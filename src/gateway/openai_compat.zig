const std = @import("std");
const secret = @import("../core/auth/secret.zig");
const stream_provider = @import("../core/agent/stream_provider.zig");
const io_mod = @import("../core/shared/io.zig");
const types = @import("../core/shared/types.zig");
const gateway_client = @import("client.zig");
const model_tool_schema = @import("../core/tooling/model_tool_schema.zig");

const Allocator = std.mem.Allocator;

/// Base URL for an OpenAI-compatible Chat Completions endpoint, for example
/// https://api.groq.com/openai/v1. HTTPS and loopback HTTP are allowed.
pub const base_url_env = "FX_OPENAI_COMPAT_BASE_URL";
pub const e2e_chat_url_env = "FX_E2E_OPENAI_COMPAT_CHAT_URL";
const max_error_body_bytes: usize = 1024 * 1024;
const max_sse_line_bytes: usize = 32 * 1024 * 1024;
const max_sse_events: usize = 100_000;
const max_tool_calls: usize = 128;
const max_tool_identity_bytes: usize = 1024;
const max_tool_arguments_bytes: usize = 4 * 1024 * 1024;
const transfer_buffer_bytes: usize = 256 * 1024;
const connect_timeout_ms: i64 = 30_000;

pub const agent_stream_provider = stream_provider.Provider{
    .stream_fn = streamCompletion,
    .build_request_fn = buildRequestForProvider,
};

fn validateModel(model: []const u8) !void {
    if (model.len == 0 or model.len > 1024) return error.InvalidOpenAiCompatModel;
    for (model) |byte| {
        if (byte <= 0x20 or byte == 0x7f) return error.InvalidOpenAiCompatModel;
    }
}

/// Resolves the Chat Completions URL. The configured base URL must be HTTPS or
/// a loopback HTTP URL so a malicious environment cannot make fx talk to an
/// unexpected plaintext host. Trailing slashes are tolerated.
pub fn resolveChatUrl(alloc: Allocator, base_url_value: ?[]const u8) ![]u8 {
    const base = base_url_value orelse return error.OpenAiCompatBaseUrlMissing;
    const trimmed = std.mem.trim(u8, base, " \t\r\n");
    if (trimmed.len == 0) return error.OpenAiCompatBaseUrlMissing;
    if (std.mem.startsWith(u8, trimmed, "https://")) {
        // HTTPS is allowed as-is.
    } else if (std.mem.startsWith(u8, trimmed, "http://")) {
        if (!gateway_client.isLoopbackHttpUrl(trimmed)) {
            return error.InvalidOpenAiCompatBaseUrl;
        }
    } else {
        return error.InvalidOpenAiCompatBaseUrl;
    }
    const stripped = std.mem.trimEnd(u8, trimmed, "/");
    return std.fmt.allocPrint(alloc, "{s}/chat/completions", .{stripped});
}

pub fn buildRequest(
    alloc: Allocator,
    request: stream_provider.RequestData,
) ![]u8 {
    try request.validatePrompt();
    try validateModel(request.model);
    if (request.budget) |budget| {
        if (budget.cancel_flag) |flag| if (flag.load(.seq_cst)) return error.Cancelled;
        _ = budget.deadline;
    }
    if (request.vision_mode == .required) return error.OpenAiCompatVisionUnsupported;
    if (request.verified_images != null or request.response_format != null) {
        return error.OpenAiCompatStructuredResponseUnsupported;
    }

    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    const writer = &out.writer;
    try writer.writeAll("{\"model\":");
    try std.json.Stringify.value(request.model, .{}, writer);
    try writer.writeAll(",\"stream\":true,\"messages\":[");
    try writeMessages(writer, alloc, request);
    try writer.writeByte(']');

    const tool_count = try writeChatTools(writer, alloc, request.tools);
    if (tool_count > 0) {
        try writer.writeAll(",\"tool_choice\":");
        try writeToolChoice(writer, request.tool_choice);
        try writer.writeAll(",\"parallel_tool_calls\":true");
    }

    if (request.max_output_tokens) |max_output_tokens| {
        try writer.print(",\"max_tokens\":{d}", .{max_output_tokens});
    }
    if (request.provider_options.reasoning) |effort| {
        if (!std.mem.eql(u8, effort.label(), "auto")) {
            try writer.writeAll(",\"reasoning_effort\":");
            try std.json.Stringify.value(effort.label(), .{}, writer);
        }
    }
    try writer.writeByte('}');
    return out.toOwnedSlice();
}

fn buildRequestForProvider(
    _: ?*anyopaque,
    alloc: Allocator,
    request: stream_provider.RequestData,
) anyerror![]u8 {
    return buildRequest(alloc, request);
}

fn writeMessages(
    writer: *std.Io.Writer,
    alloc: Allocator,
    request: stream_provider.RequestData,
) !void {
    var count: usize = 0;
    for (request.instructions) |instruction| {
        const text = instruction.content orelse continue;
        if (count > 0) try writer.writeByte(',');
        try writeTextMessage(writer, "system", text);
        count += 1;
    }
    for (request.messages) |message| {
        if (count > 0) try writer.writeByte(',');
        switch (message.role) {
            .system => {
                // The gateway prompt validator keeps system out of messages, but
                // a system lane here is harmless and some hosts place it there.
                try writeTextMessage(writer, "system", message.content orelse "");
            },
            .user => {
                const text = message.content orelse "";
                if (message.images.len > 0) {
                    // Image parts belong to a later phase; keep the text lane so
                    // a mislabeled capability cannot hide the user's question.
                }
                try writeTextMessage(writer, "user", text);
            },
            .assistant => try writeAssistantMessage(writer, alloc, message),
            .tool => {
                const tool_call_id = message.tool_call_id orelse
                    return error.OpenAiCompatToolMessageMissingCallId;
                const content = message.content orelse "";
                try writer.writeAll("{\"role\":\"tool\",\"tool_call_id\":");
                try std.json.Stringify.value(tool_call_id, .{}, writer);
                try writer.writeAll(",\"content\":");
                try std.json.Stringify.value(content, .{}, writer);
                try writer.writeByte('}');
            },
        }
        count += 1;
    }
}

fn writeTextMessage(writer: *std.Io.Writer, role: []const u8, content: []const u8) !void {
    try writer.writeAll("{\"role\":");
    try std.json.Stringify.value(role, .{}, writer);
    try writer.writeAll(",\"content\":");
    try std.json.Stringify.value(content, .{}, writer);
    try writer.writeByte('}');
}

fn writeAssistantMessage(
    writer: *std.Io.Writer,
    alloc: Allocator,
    message: types.ChatMessage,
) !void {
    _ = alloc;
    try writer.writeAll("{\"role\":\"assistant\"");
    if (message.content) |content| {
        if (content.len > 0) {
            try writer.writeAll(",\"content\":");
            try std.json.Stringify.value(content, .{}, writer);
        }
    }
    if (message.tool_calls.len > 0) {
        try writer.writeAll(",\"tool_calls\":[");
        for (message.tool_calls, 0..) |call, index| {
            if (index > 0) try writer.writeByte(',');
            try writer.writeAll("{\"id\":");
            try std.json.Stringify.value(call.id, .{}, writer);
            try writer.writeAll(",\"type\":\"function\",\"function\":{\"name\":");
            try std.json.Stringify.value(call.name, .{}, writer);
            try writer.writeAll(",\"arguments\":");
            try std.json.Stringify.value(call.arguments_json, .{}, writer);
            try writer.writeByte('}');
            try writer.writeByte('}');
        }
        try writer.writeByte(']');
    }
    try writer.writeByte('}');
}

fn writeToolChoice(writer: *std.Io.Writer, tool_choice: types.ToolChoice) !void {
    switch (tool_choice) {
        .auto, .none => try std.json.Stringify.value(tool_choice.label(), .{}, writer),
        .required => try std.json.Stringify.value("required", .{}, writer),
    }
}

fn writeChatTools(
    writer: *std.Io.Writer,
    alloc: Allocator,
    tools: stream_provider.ToolSelection,
) !usize {
    var count: usize = 0;
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    try out.writer.writeByte('[');

    for (tools.advertised_names) |name| {
        const tool = tools.advertisedFunction(name) orelse continue;
        if (count > 0) try out.writer.writeByte(',');
        try writeChatFunctionTool(&out.writer, alloc, tool);
        count += 1;
    }
    for (tools.additional_functions) |tool| {
        if (containsName(tools.advertised_names, tool.name)) continue;
        if (count > 0) try out.writer.writeByte(',');
        try writeChatFunctionTool(&out.writer, alloc, tool);
        count += 1;
    }
    for (tools.selected_dynamic) |tool| {
        if (containsName(tools.advertised_names, tool.name)) continue;
        if (count > 0) try out.writer.writeByte(',');
        if (tool.input_schema != .object) return error.InvalidToolSchema;
        var schema_out: std.Io.Writer.Allocating = .init(alloc);
        defer schema_out.deinit();
        try std.json.Stringify.value(tool.input_schema, .{}, &schema_out.writer);
        try writeChatFunctionToolJson(&out.writer, alloc, tool.name, tool.description, schema_out.written());
        count += 1;
    }
    try out.writer.writeByte(']');
    if (count > 0) {
        try writer.writeAll(",\"tools\":");
        try writer.writeAll(out.written());
    }
    return count;
}

fn containsName(names: []const []const u8, expected: []const u8) bool {
    for (names) |name| if (std.mem.eql(u8, name, expected)) return true;
    return false;
}

fn writeChatFunctionTool(
    writer: *std.Io.Writer,
    alloc: Allocator,
    tool: model_tool_schema.FunctionSchema,
) !void {
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    try model_tool_schema.writeObjectSchema(alloc, &out.writer, tool.input_schema);
    try writeChatFunctionToolJson(writer, alloc, tool.name, tool.description, out.written());
}

fn writeChatFunctionToolJson(
    writer: *std.Io.Writer,
    alloc: Allocator,
    name: []const u8,
    description: []const u8,
    parameters: []const u8,
) !void {
    if (name.len == 0) return error.InvalidToolSchema;
    try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":");
    try std.json.Stringify.value(name, .{}, writer);
    if (description.len > 0) {
        try writer.writeAll(",\"description\":");
        try model_tool_schema.writeCappedDescriptionJsonString(alloc, writer, description);
    }
    try writer.writeAll(",\"parameters\":");
    try writer.writeAll(parameters);
    try writer.writeByte('}');
    try writer.writeByte('}');
}

fn streamCompletion(
    _: ?*anyopaque,
    alloc: Allocator,
    request: stream_provider.ModelRequest,
) !stream_provider.Result {
    if (request.cancel_flag.load(.seq_cst)) return stream_provider.failResult(error.Cancelled);
    const source = request.credential.credentialSource();
    if (source != .openai_compat_api_key and source != .host_managed) {
        return stream_provider.failResult(error.OpenAiCompatApiKeyRequired);
    }
    try validateModel(request.model);
    const payload = request.prepared_request_body orelse
        try buildRequest(alloc, request.data());
    defer if (request.prepared_request_body == null) alloc.free(payload);
    var operation = PreparedStreamOperation{
        .alloc = alloc,
        .request = request,
        .payload = payload,
    };
    return (if (request.deadline) |deadline|
        gateway_client.runBoundedHttpOperation(
            stream_provider.Result,
            alloc,
            request.cancel_flag,
            deadline,
            &operation,
        )
    else
        operation.run()) catch |err| {
        if (request.cancel_flag.load(.seq_cst)) return stream_provider.failResult(error.Cancelled);
        request.attempt_evidence.network_failure = gateway_client.networkFailureEvidence(err, request.delivery.load());
        return err;
    };
}

const PreparedStreamOperation = struct {
    alloc: Allocator,
    request: stream_provider.ModelRequest,
    payload: []const u8,

    pub fn run(self: *@This()) !stream_provider.Result {
        return streamPrepared(self.alloc, self.request, self.payload);
    }
};

const OpenedRequest = struct {
    request: ?std.http.Client.Request,

    pub fn deinit(self: *OpenedRequest, _: Allocator) void {
        if (self.request) |*request| request.deinit();
        self.request = null;
    }

    pub fn take(self: *OpenedRequest) std.http.Client.Request {
        const request = self.request.?;
        self.request = null;
        return request;
    }
};

const OpenRequestOperation = struct {
    client: *std.http.Client,
    uri: std.Uri,
    auth_header: ?[]const u8,
    extra_headers: []const std.http.Header,

    pub fn run(self: *@This()) !OpenedRequest {
        var headers: std.http.Client.Request.Headers = .{
            .content_type = .{ .override = "application/json" },
            .accept_encoding = .omit,
            .user_agent = .{ .override = gateway_client.user_agent },
        };
        if (self.auth_header) |authorization| {
            headers.authorization = .{ .override = authorization };
        }
        return .{ .request = try self.client.request(.POST, self.uri, .{
            .headers = headers,
            .extra_headers = self.extra_headers,
            .keep_alive = false,
            .redirect_behavior = .unhandled,
        }) };
    }
};

fn requestAuthHeaders(alloc: Allocator, auth: stream_provider.CredentialLease) !?[]u8 {
    return switch (auth) {
        .host_managed => null,
        .direct => |direct| blk: {
            if (direct.secret_bytes.len == 0) return error.OpenAiCompatApiKeyRequired;
            break :blk try std.fmt.allocPrint(alloc, "Bearer {s}", .{direct.secret_bytes});
        },
    };
}

fn streamPrepared(
    alloc: Allocator,
    request: stream_provider.ModelRequest,
    payload: []const u8,
) !stream_provider.Result {
    if (request.cancel_flag.load(.seq_cst)) return stream_provider.failResult(error.Cancelled);
    const auth_header: ?[]u8 = try requestAuthHeaders(alloc, request.credential);
    defer if (auth_header) |value| secret.zeroAndFree(alloc, value);
    var owned_endpoint: ?[]u8 = null;
    defer if (owned_endpoint) |value| alloc.free(value);
    const request_endpoint = if (io_mod.getenv(e2e_chat_url_env)) |override| endpoint: {
        if (!gateway_client.isLoopbackHttpUrl(override)) {
            return stream_provider.failResult(error.InvalidE2EOpenAiCompatEndpoint);
        }
        break :endpoint override;
    } else endpoint: {
        const resolved = try resolveChatUrl(alloc, io_mod.getenv(base_url_env));
        owned_endpoint = resolved;
        break :endpoint resolved;
    };
    const uri = try std.Uri.parse(request_endpoint);

    var extra_headers_buf: [2]std.http.Header = undefined;
    var extra_count: usize = 0;
    extra_headers_buf[extra_count] = .{ .name = "accept", .value = "text/event-stream" };
    extra_count += 1;

    var client: std.http.Client = .{ .allocator = alloc, .io = io_mod.getIo() };
    defer client.deinit();
    var open_operation = OpenRequestOperation{
        .client = &client,
        .uri = uri,
        .auth_header = auth_header,
        .extra_headers = extra_headers_buf[0..extra_count],
    };
    const connect_deadline = std.Io.Clock.Timestamp.fromNow(io_mod.getIo(), .{
        .clock = .awake,
        .raw = .fromMilliseconds(connect_timeout_ms),
    });
    try request.admission.admit();
    var opened = try gateway_client.runBoundedHttpOperation(
        OpenedRequest,
        alloc,
        request.cancel_flag,
        connect_deadline,
        &open_operation,
    );
    var http_request = opened.take();
    defer http_request.deinit();
    var cancel_watch_done = std.atomic.Value(bool).init(false);
    const cancel_watcher = if (http_request.connection) |connection|
        try gateway_client.spawnHttpCancelWatcher(
            &cancel_watch_done,
            request.cancel_flag,
            connection.stream_writer.stream,
        )
    else
        null;
    defer {
        cancel_watch_done.store(true, .seq_cst);
        if (cancel_watcher) |thread| thread.join();
    }
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;

    http_request.transfer_encoding = .{ .content_length = payload.len };
    var send_buffer: [8192]u8 = undefined;
    request.delivery.markPossiblySent();
    var body_writer = try http_request.sendBodyUnflushed(&send_buffer);
    try body_writer.writer.writeAll(payload);
    try body_writer.end();
    if (http_request.connection) |connection| try connection.flush();
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;

    var response = try http_request.receiveHead(&.{});
    if (response.head.status != .ok) {
        var transfer: [16 * 1024]u8 = undefined;
        const reader = response.reader(&transfer);
        const body = reader.allocRemaining(alloc, .limited(max_error_body_bytes)) catch |err| switch (err) {
            error.StreamTooLong => try alloc.dupe(u8, "OpenAI-compatible error response exceeded the local limit"),
            else => return err,
        };
        return .{ .failed = .{
            .kind = failureKind(response.head.status),
            .detail = body,
            .ownership = .owned,
        } };
    }

    var transfer_buffer: [transfer_buffer_bytes]u8 = undefined;
    const reader = response.reader(&transfer_buffer);
    var events = request.events;
    const completion = try consumeSse(
        alloc,
        reader,
        &events,
        EventBridge.content,
        EventBridge.toolStart,
        EventBridge.reasoning,
        EventBridge.toolInput,
        request.cancel_flag,
        request.content_capture_limit,
    );
    errdefer {
        var owned = stream_provider.Result{ .completed = .{
            .completion = completion,
            .ownership = .owned,
        } };
        owned.deinit(alloc);
    }
    return .{ .completed = .{
        .completion = completion,
        .usage = if (completion.generation_id == null)
            .{ .unavailable = .unbilled }
        else
            .{ .unavailable = .possibly_billed },
        .ownership = .owned,
    } };
}

const EventBridge = struct {
    fn sink(raw: *anyopaque) *stream_provider.EventSink {
        return @ptrCast(@alignCast(raw));
    }

    fn content(raw: *anyopaque, chunk: []const u8) void {
        sink(raw).emit(.{ .content_delta = chunk });
    }

    fn reasoning(raw: *anyopaque, chunk: []const u8) void {
        sink(raw).emit(.{ .reasoning_delta = chunk });
    }

    fn toolInput(raw: *anyopaque, chunk: []const u8) void {
        sink(raw).emit(.{ .tool_input_delta = chunk });
    }

    fn toolStart(raw: *anyopaque, id: []const u8, name: []const u8, label: ?[]const u8) void {
        sink(raw).emit(.{ .tool_started = .{ .id = id, .name = name, .label = label } });
    }
};

const StreamCallbacks = struct {
    context: *anyopaque,
    on_content: stream_provider.StreamCallback,
    on_tool_start: ?stream_provider.ToolStartCallback = null,
    on_reasoning: ?stream_provider.StreamCallback = null,
    on_tool_input: ?stream_provider.StreamCallback = null,
};

fn failureKind(status: std.http.Status) stream_provider.FailureKind {
    return switch (status) {
        .bad_request => .invalid_request,
        .unauthorized => .unauthorized,
        .forbidden => .forbidden,
        .payload_too_large => .request_too_large,
        .too_many_requests => .rate_limited,
        .internal_server_error => .server_error,
        .bad_gateway => .bad_gateway,
        .service_unavailable => .unavailable,
        .gateway_timeout => .gateway_timeout,
        else => .provider_error,
    };
}

const SseReader = struct {
    pending_line: std.ArrayList(u8) = .empty,

    fn deinit(self: *SseReader, alloc: Allocator) void {
        self.pending_line.deinit(alloc);
    }

    fn release(self: *SseReader) void {
        self.pending_line.clearRetainingCapacity();
    }

    fn next(self: *SseReader, alloc: Allocator, reader: anytype) !?[]const u8 {
        while (true) {
            const line = try self.readLine(alloc, reader) orelse return null;
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0 or trimmed[0] == ':') {
                self.release();
                continue;
            }
            if (!std.mem.startsWith(u8, trimmed, "data:")) {
                self.release();
                continue;
            }
            const data = std.mem.trim(u8, trimmed["data:".len..], " \t");
            if (std.mem.eql(u8, data, "[DONE]")) return null;
            return data;
        }
    }

    fn readLine(self: *SseReader, alloc: Allocator, reader: anytype) !?[]const u8 {
        while (true) {
            const fragment = reader.takeDelimiter('\n') catch |err| switch (err) {
                error.StreamTooLong => {
                    const buffered = reader.buffered();
                    if (buffered.len == 0) return error.OpenAiCompatSseReadStalled;
                    if (buffered.len > max_sse_line_bytes - self.pending_line.items.len) {
                        return error.OpenAiCompatSseEventTooLarge;
                    }
                    try self.pending_line.appendSlice(alloc, buffered);
                    reader.tossBuffered();
                    continue;
                },
                error.ReadFailed => return error.ReadFailed,
            } orelse {
                if (self.pending_line.items.len > 0) return self.pending_line.items;
                return null;
            };
            if (fragment.len > max_sse_line_bytes - self.pending_line.items.len) {
                return error.OpenAiCompatSseEventTooLarge;
            }
            if (self.pending_line.items.len == 0) return fragment;
            try self.pending_line.appendSlice(alloc, fragment);
            return self.pending_line.items;
        }
    }
};

const ToolAccumulator = struct {
    index: i64,
    id: std.ArrayList(u8) = .empty,
    name: std.ArrayList(u8) = .empty,
    arguments: std.ArrayList(u8) = .empty,
    started: bool = false,

    fn deinit(self: *ToolAccumulator, alloc: Allocator) void {
        self.id.deinit(alloc);
        self.name.deinit(alloc);
        self.arguments.deinit(alloc);
    }
};

const Reducer = struct {
    const Self = @This();

    alloc: Allocator,
    event_count: usize = 0,
    content: std.ArrayList(u8) = .empty,
    tool_calls: std.ArrayList(types.ToolCall) = .empty,
    generation_id: ?[]u8 = null,
    finish_reason: ?types.ProviderFinishReason = null,
    usage: types.Usage = .{},
    cancelled: bool = false,
    accumulators: std.ArrayList(ToolAccumulator) = .empty,

    fn init(alloc: Allocator) Self {
        return .{ .alloc = alloc };
    }

    fn deinit(self: *Self) void {
        const alloc = self.alloc;
        self.content.deinit(alloc);
        // Any tool calls not transferred into a completion are freed here.
        types.freeToolCallSlice(alloc, self.tool_calls.items);
        self.tool_calls.deinit(alloc);
        if (self.generation_id) |id| alloc.free(id);
        for (self.accumulators.items) |*acc| acc.deinit(alloc);
        self.accumulators.deinit(alloc);
        self.* = undefined;
    }

    fn applyEvent(
        self: *Self,
        alloc: Allocator,
        json_text: []const u8,
        callbacks: StreamCallbacks,
        cancel_flag: *std.atomic.Value(bool),
        content_capture_limit: ?usize,
    ) !void {
        _ = cancel_flag;
        self.event_count += 1;
        if (self.event_count > max_sse_events) return error.OpenAiCompatTooManyEvents;

        var parsed = try std.json.parseFromSlice(std.json.Value, alloc, json_text, .{});
        defer parsed.deinit();
        const root = parsed.value;
        if (root != .object) return error.InvalidOpenAiCompatSseEvent;
        const object = root.object;

        if (object.get("id")) |id_value| {
            if (id_value == .string and self.generation_id == null and id_value.string.len > 0) {
                self.generation_id = try alloc.dupe(u8, id_value.string);
            }
        }
        const usage_value = object.get("usage");
        if (usage_value) |usage| if (usage == .object) self.captureUsage(usage.object);

        const choices_value = object.get("choices") orelse return;
        if (choices_value != .array or choices_value.array.items.len == 0) return;
        const choice = choices_value.array.items[0];
        if (choice != .object) return;
        const choice_object = choice.object;

        if (choice_object.get("finish_reason")) |finish_value| {
            if (finish_value == .string) {
                if (finish_value.string.len > 0) {
                    self.finish_reason = parseChatFinishReason(finish_value.string);
                }
            } else if (finish_value != .null) {
                return error.InvalidOpenAiCompatSseEvent;
            }
        }

        const delta_value = choice_object.get("delta") orelse return;
        if (delta_value != .object) return;
        const delta = delta_value.object;

        if (delta.get("reasoning_content")) |value| {
            if (value == .string and value.string.len > 0) {
                if (callbacks.on_reasoning) |cb| cb(callbacks.context, value.string);
            }
        }
        if (delta.get("content")) |value| {
            if (value == .string and value.string.len > 0) {
                try appendCaptured(alloc, &self.content, value.string, content_capture_limit);
                callbacks.on_content(callbacks.context, value.string);
            }
        }
        if (delta.get("tool_calls")) |value| {
            if (value != .array) return error.InvalidOpenAiCompatSseEvent;
            try self.applyToolCalls(alloc, value.array.items, callbacks);
        }
    }

    fn applyToolCalls(
        self: *Self,
        alloc: Allocator,
        calls: []const std.json.Value,
        callbacks: StreamCallbacks,
    ) !void {
        for (calls) |call_value| {
            if (call_value != .object) return error.InvalidOpenAiCompatSseEvent;
            const call = call_value.object;
            const index_value = call.get("index") orelse return error.InvalidOpenAiCompatSseEvent;
            if (index_value != .integer) return error.InvalidOpenAiCompatSseEvent;

            var accumulator = try self.accumulatorFor(alloc, index_value.integer);
            if (call.get("id")) |id_value| {
                if (id_value == .string and id_value.string.len > 0) {
                    if (accumulator.id.items.len == 0) {
                        try accumulator.id.appendSlice(alloc, id_value.string);
                    }
                }
            }
            const function_value = call.get("function");
            if (function_value) |function| {
                if (function != .object) return error.InvalidOpenAiCompatSseEvent;
                const function_object = function.object;
                if (function_object.get("name")) |name_value| {
                    if (name_value == .string and name_value.string.len > 0) {
                        if (accumulator.name.items.len == 0) {
                            try accumulator.name.appendSlice(alloc, name_value.string);
                        }
                    }
                }
                if (function_object.get("arguments")) |arguments_value| {
                    if (arguments_value == .string and arguments_value.string.len > 0) {
                        if (accumulator.arguments.items.len + arguments_value.string.len >
                            max_tool_arguments_bytes)
                        {
                            return error.OpenAiCompatToolArgumentsTooLarge;
                        }
                        try accumulator.arguments.appendSlice(alloc, arguments_value.string);
                        if (callbacks.on_tool_input) |cb| {
                            cb(callbacks.context, arguments_value.string);
                        }
                    }
                }
            }
            if (!accumulator.started and accumulator.id.items.len > 0 and
                accumulator.name.items.len > 0)
            {
                accumulator.started = true;
                if (callbacks.on_tool_start) |cb| {
                    cb(callbacks.context, accumulator.id.items, accumulator.name.items, null);
                }
            }
        }
    }

    fn accumulatorFor(self: *Self, alloc: Allocator, index: i64) !*ToolAccumulator {
        for (self.accumulators.items) |*accumulator| {
            if (accumulator.index == index) return accumulator;
        }
        if (self.accumulators.items.len >= max_tool_calls) {
            return error.OpenAiCompatToolCallLimitExceeded;
        }
        if (index < 0) return error.InvalidOpenAiCompatSseEvent;
        try self.accumulators.append(alloc, .{ .index = index });
        return &self.accumulators.items[self.accumulators.items.len - 1];
    }

    fn captureUsage(self: *Self, usage_object: std.json.ObjectMap) void {
        if (integerField(usage_object, "prompt_tokens")) |value| self.usage.input_tokens = value;
        if (integerField(usage_object, "completion_tokens")) |value| self.usage.output_tokens = value;
        if (integerField(usage_object, "cache_read_input_tokens")) |value| self.usage.cache_read_tokens = value;
        if (integerField(usage_object, "cache_creation_input_tokens")) |value| self.usage.cache_write_tokens = value;
        if (usage_object.get("completion_tokens_details")) |details| {
            if (details == .object) {
                if (integerField(details.object, "reasoning_tokens")) |value| {
                    self.usage.reasoning_tokens = value;
                }
            }
        }
    }

    fn finish(self: *Self, alloc: Allocator, cancel_flag: *std.atomic.Value(bool)) !types.ModelCompletion {
        if (cancel_flag.load(.seq_cst)) self.cancelled = true;

        // Materialize streamed tool calls in arrival order.
        for (self.accumulators.items) |*accumulator| {
            if (!accumulator.started) continue;
            const id = try alloc.dupe(u8, accumulator.id.items);
            errdefer alloc.free(id);
            const name = try alloc.dupe(u8, accumulator.name.items);
            errdefer alloc.free(name);
            const arguments_json = try alloc.dupe(u8, accumulator.arguments.items);
            errdefer alloc.free(arguments_json);
            try self.tool_calls.append(alloc, .{
                .id = id,
                .name = name,
                .arguments_json = arguments_json,
            });
        }

        // A healthy Chat Completions stream always reports a finish reason
        // before [DONE]. Missing it means the provider cut the stream short.
        if (!self.cancelled and self.finish_reason == null) {
            return error.OpenAiCompatStreamIncomplete;
        }

        const content = if (self.content.items.len > 0)
            try alloc.dupe(u8, self.content.items)
        else
            null;
        errdefer if (content) |value| alloc.free(value);

        // Ownership transfers to the returned completion; deinit afterwards
        // only frees whatever was not moved out.
        const tool_calls = try self.tool_calls.toOwnedSlice(alloc);
        const generation_id = self.generation_id;
        self.generation_id = null;
        return .{
            .content = content,
            .tool_calls = tool_calls,
            .generation_id = generation_id,
            .finish_reason = self.finish_reason,
            .usage = self.usage,
        };
    }
};

fn parseChatFinishReason(raw: []const u8) types.ProviderFinishReason {
    if (std.mem.eql(u8, raw, "function_call")) return .tool_calls;
    return types.ProviderFinishReason.parse_legacy(raw) orelse .other;
}

fn integerField(object: std.json.ObjectMap, key: []const u8) ?u64 {
    const value = object.get(key) orelse return null;
    if (value != .integer) return null;
    const integer = value.integer;
    if (integer < 0) return null;
    return @intCast(integer);
}

fn appendCaptured(
    alloc: Allocator,
    target: *std.ArrayList(u8),
    value: []const u8,
    capture_limit: ?usize,
) !void {
    const budget = capture_limit orelse return target.appendSlice(alloc, value);
    const current = target.items.len;
    if (current >= budget) return;
    const available = budget - current;
    try target.appendSlice(alloc, value[0..@min(available, value.len)]);
}

fn consumeSse(
    alloc: Allocator,
    reader: anytype,
    callback_ctx: *anyopaque,
    on_content_chunk: stream_provider.StreamCallback,
    on_tool_start: ?stream_provider.ToolStartCallback,
    on_reasoning_chunk: ?stream_provider.StreamCallback,
    on_tool_input_chunk: ?stream_provider.StreamCallback,
    cancel_flag: *std.atomic.Value(bool),
    content_capture_limit: ?usize,
) !types.ModelCompletion {
    var reducer = Reducer.init(alloc);
    defer reducer.deinit();
    var sse: SseReader = .{};
    defer sse.deinit(alloc);
    const callbacks = StreamCallbacks{
        .context = callback_ctx,
        .on_content = on_content_chunk,
        .on_tool_start = on_tool_start,
        .on_reasoning = on_reasoning_chunk,
        .on_tool_input = on_tool_input_chunk,
    };
    while (try sse.next(alloc, reader)) |json_text| {
        defer sse.release();
        if (cancel_flag.load(.seq_cst)) break;
        try reducer.applyEvent(alloc, json_text, callbacks, cancel_flag, content_capture_limit);
    }
    return reducer.finish(alloc, cancel_flag);
}

test "OpenAI-compatible request uses chat completions messages and nested tools" {
    const read_file_schema = model_tool_schema.FunctionSchema{
        .name = "read_file",
        .description = "Read a file",
        .input_schema = .{
            .properties = &.{.{ .name = "path", .json_type = .string }},
            .required = &.{"path"},
            .additional_properties = false,
        },
    };
    const instructions = [_]types.ChatMessage{.{ .role = .system, .content = "Be concise." }};
    const messages = [_]types.ChatMessage{
        .{ .role = .user, .content = "Read it." },
        .{
            .role = .assistant,
            .tool_calls = &.{.{ .id = "call_1", .name = "read_file", .arguments_json = "{\"path\":\"README.md\"}" }},
        },
        .{ .role = .tool, .tool_call_id = "call_1", .tool_name = "read_file", .content = "contents" },
    };
    const body = try buildRequest(std.testing.allocator, .{
        .model = "llama-3.3-70b",
        .instructions = &instructions,
        .messages = &messages,
        .tools = .{ .additional_functions = &.{read_file_schema} },
        .tool_choice = .auto,
        .provider_options = .{ .reasoning = types.ReasoningEffort.literal("high") },
        .max_output_tokens = 4096,
    });
    defer std.testing.allocator.free(body);

    try std.testing.expect(std.mem.find(u8, body, "\"model\":\"llama-3.3-70b\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"role\":\"system\",\"content\":\"Be concise.\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"role\":\"tool\",\"tool_call_id\":\"call_1\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"tool_calls\":[{\"id\":\"call_1\"") != null);
    // The messages array must close before the tools key opens; a missing
    // ",\"tools\":" delimiter produced invalid JSON that servers reject.
    try std.testing.expect(std.mem.find(u8, body, "\"content\":\"contents\"}],\"tools\":[{\"type\":\"function\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"type\":\"function\",\"function\":{\"name\":\"read_file\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"parameters\":{\"type\":\"object\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"reasoning_effort\":\"high\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"max_tokens\":4096") != null);
}

test "OpenAI-compatible reducer turns chat chunks into a tool completion" {
    const events = [_][]const u8{
        "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}",
        "{\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_9\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}}]},\"finish_reason\":null}]}",
        "{\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}",
    };
    var completion = try reduceEvents(std.testing.allocator, &events, null, null);
    defer freeCompletion(std.testing.allocator, &completion);

    try std.testing.expectEqualStrings("chatcmpl-1", completion.generation_id.?);
    try std.testing.expectEqual(types.ProviderFinishReason.tool_calls, completion.finish_reason.?);
    try std.testing.expectEqual(@as(usize, 1), completion.tool_calls.len);
    try std.testing.expectEqualStrings("call_9", completion.tool_calls[0].id);
    try std.testing.expectEqualStrings("bash", completion.tool_calls[0].name);
    try std.testing.expectEqualStrings("{\"command\":\"ls\"}", completion.tool_calls[0].arguments_json);
}

test "OpenAI-compatible reducer accumulates text and usage and stops at length" {
    const events = [_][]const u8{
        "{\"id\":\"chatcmpl-2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"},\"finish_reason\":null}]}",
        "{\"id\":\"chatcmpl-2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}",
        "{\"id\":\"chatcmpl-2\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}",
    };
    var completion = try reduceEvents(std.testing.allocator, &events, null, null);
    defer freeCompletion(std.testing.allocator, &completion);

    try std.testing.expectEqualStrings("Hello", completion.content.?);
    try std.testing.expectEqual(types.ProviderFinishReason.length, completion.finish_reason.?);
    try std.testing.expectEqual(@as(?u64, 10), completion.usage.input_tokens);
    try std.testing.expectEqual(@as(?u64, 5), completion.usage.output_tokens);
}

test "OpenAI-compatible stream without terminal events is incomplete" {
    const events = [_][]const u8{
        "{\"id\":\"chatcmpl-3\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":null}]}",
    };
    try std.testing.expectError(
        error.OpenAiCompatStreamIncomplete,
        reduceEvents(std.testing.allocator, &events, null, null),
    );
}

fn noopChunk(_: *anyopaque, _: []const u8) void {}
fn noopToolStart(_: *anyopaque, _: []const u8, _: []const u8, _: ?[]const u8) void {}

fn reduceEvents(
    alloc: Allocator,
    events: []const []const u8,
    content_limit: ?usize,
    cancel_flag: ?*std.atomic.Value(bool),
) !types.ModelCompletion {
    const Feed = struct {
        events: []const []const u8,
        cursor: usize = 0,
    };
    var feed = Feed{ .events = events };
    const Reader = struct {
        feed: *Feed,
        buffered_data: []const u8 = "",

        pub fn buffered(self: *@This()) []const u8 {
            return self.buffered_data;
        }

        pub fn tossBuffered(self: *@This()) void {
            self.buffered_data = "";
        }

        pub fn takeDelimiter(self: *@This(), delimiter: u8) error{ StreamTooLong, ReadFailed }!?[]const u8 {
            if (self.buffered_data.len == 0) {
                if (self.feed.cursor >= self.feed.events.len) return null;
                const line = self.feed.events[self.feed.cursor];
                self.feed.cursor += 1;
                const trimmed = std.mem.trim(u8, line, "\n");
                // Test events never embed a real newline; a delimiter inside a
                // fragment would only mean the fixture needs splitting.
                if (std.mem.findScalar(u8, trimmed, delimiter) != null) {
                    return error.ReadFailed;
                }
                return trimmed;
            }
            const index = std.mem.indexOfScalar(u8, self.buffered_data, delimiter) orelse {
                const out = self.buffered_data;
                self.buffered_data = "";
                return out;
            };
            const out = self.buffered_data[0..index];
            self.buffered_data = self.buffered_data[index + 1 ..];
            return out;
        }
    };
    var sink: usize = 0;
    var cancel = std.atomic.Value(bool).init(false);
    const flag = cancel_flag orelse &cancel;
    var reader = Reader{ .feed = &feed };
    return consumeSse(
        alloc,
        &reader,
        &sink,
        noopChunk,
        noopToolStart,
        noopChunk,
        noopChunk,
        flag,
        content_limit,
    );
}

fn freeCompletion(alloc: Allocator, completion: *types.ModelCompletion) void {
    if (completion.content) |content| alloc.free(content);
    if (completion.generation_id) |id| alloc.free(id);
    types.freeToolCallSlice(alloc, @constCast(completion.tool_calls));
    completion.* = .{};
}
