const std = @import("std");
const credentials = @import("../core/auth/credentials.zig");
const gateway_provider = @import("../core/gateway/gateway_provider.zig");
const model_capabilities = @import("../core/config/model_capabilities.zig");
const io_mod = @import("../core/shared/io.zig");
const model_catalog = @import("../core/gateway/model_catalog.zig");
const gateway_client = @import("client.zig");
const openai_compat = @import("openai_compat.zig");

const Allocator = std.mem.Allocator;

const max_response_bytes: usize = 8 * 1024 * 1024;
const request_timeout_ms: i64 = 15_000;

/// Capabilities for models listed by a generic OpenAI-compatible /models
/// endpoint are unknown ahead of time; known metadata comes from the catalog
/// entry and everything else keeps fx's conservative defaults.
pub fn capabilitiesForModel(_: []const u8) model_capabilities.Capabilities {
    return .{};
}

pub const cli_model_catalog_provider = gateway_provider.CliModelCatalogProvider{
    .fetch_fn = fetchCliModelCatalog,
};

pub const model_catalog_provider = model_catalog.Provider{
    .fetch_fn = fetchModelCatalog,
    .provider_id = .openai_compat,
};

fn fetchCliModelCatalog(
    _: ?*anyopaque,
    alloc: Allocator,
    input: gateway_provider.CliModelCatalogInput,
) gateway_provider.CliModelCatalogResult {
    const result = model_catalog.fetchWithPublicFallback(model_catalog_provider, alloc, .{
        .access = input.access,
        .endpoint = input.endpoint,
        .cancel_flag = input.cancel_flag,
        .view = .full,
    });
    return switch (result) {
        .loaded => |loaded| project: {
            var catalog = loaded.catalog;
            defer model_catalog.freeModelCatalog(alloc, &catalog);
            const ids = model_catalog.projectModelIds(alloc, catalog.items) catch return .{ .failure = .{
                .access = loaded.provenance.access,
                .anonymous_fallback_used = loaded.provenance.anonymous_fallback_used,
                .failure = .{ .category = .resource_exhausted },
            } };
            break :project .{ .loaded = .{
                .ids = ids,
                .provenance = loaded.provenance,
            } };
        },
        .failed => |failed| .{ .failure = failed },
    };
}

fn fetchModelCatalog(
    _: ?*anyopaque,
    alloc: Allocator,
    input: model_catalog.FetchInput,
) Allocator.Error!model_catalog.ProviderResult {
    if (input.cancel_flag) |flag| if (flag.load(.seq_cst)) {
        return .{ .failure = .{ .category = .cancellation } };
    };
    const base_url = io_mod.getenv(openai_compat.base_url_env) orelse
        return .{ .failure = .{ .category = .runtime } };
    const models_url = modelListUrl(alloc, base_url) catch
        return .{ .failure = .{ .category = .runtime } };
    defer alloc.free(models_url);

    const access = input.access;
    const api_key = switch (access) {
        .host_managed => null,
        else => access.authorizationCredential(),
    };

    const body = fetchModelsBody(alloc, models_url, api_key, input.cancel_flag) catch |err| switch (err) {
        error.Cancelled => return .{ .failure = .{ .category = .cancellation } },
        error.HttpStatus => return .{ .failure = .{ .category = .http_status } },
        error.Timeout => return .{ .failure = .{ .category = .gateway_unavailable, .retryable = true } },
        error.ResponseTooLarge => return .{ .failure = .{ .category = .malformed_response } },
        else => return .{ .failure = .{ .category = .transport } },
    };
    defer alloc.free(body);
    return parseModels(alloc, body) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => .{ .failure = .{ .category = .malformed_response } },
    };
}

fn modelListUrl(alloc: Allocator, base_url_value: []const u8) ![]u8 {
    const trimmed = std.mem.trim(u8, base_url_value, " \t\r\n");
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
    return std.fmt.allocPrint(alloc, "{s}/models", .{stripped});
}

fn fetchModelsBody(
    alloc: Allocator,
    url: []const u8,
    api_key: ?[]const u8,
    cancel_flag: ?*std.atomic.Value(bool),
) ![]u8 {
    const secret = @import("../core/auth/secret.zig");
    var client: std.http.Client = .{ .allocator = alloc, .io = io_mod.getIo() };
    defer client.deinit();
    var auth_header: ?[]u8 = null;
    defer if (auth_header) |value| secret.zeroAndFree(alloc, value);
    if (api_key) |key| {
        auth_header = try std.fmt.allocPrint(alloc, "Bearer {s}", .{key});
    }
    var headers: std.http.Client.Request.Headers = .{
        .accept_encoding = .omit,
        .user_agent = .{ .override = gateway_client.user_agent },
    };
    if (auth_header) |value| headers.authorization = .{ .override = value };
    var request = try client.request(.GET, try std.Uri.parse(url), .{
        .headers = headers,
        .extra_headers = &.{.{ .name = "accept", .value = "application/json" }},
        .keep_alive = false,
        .redirect_behavior = .unhandled,
    });
    defer request.deinit();
    try request.sendBodiless();
    if (request.connection) |conn| try conn.flush();
    var response = request.receiveHead(&.{}) catch |err| {
        if (cancel_flag) |flag| if (flag.load(.seq_cst)) return error.Cancelled;
        if (err == error.Timeout) return error.Timeout;
        return err;
    };
    if (cancel_flag) |flag| if (flag.load(.seq_cst)) return error.Cancelled;
    if (response.head.status != .ok) return error.HttpStatus;
    var transfer_buffer: [64 * 1024]u8 = undefined;
    const reader = response.reader(&transfer_buffer);
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    _ = reader.streamRemaining(&out.writer) catch |err| {
        if (cancel_flag) |flag| if (flag.load(.seq_cst)) return error.Cancelled;
        if (err == error.StreamTooLong) return error.ResponseTooLarge;
        return err;
    };
    if (out.written().len > max_response_bytes) return error.ResponseTooLarge;
    if (cancel_flag) |flag| if (flag.load(.seq_cst)) return error.Cancelled;
    return out.toOwnedSlice();
}

fn parseModels(alloc: Allocator, json_text: []const u8) !model_catalog.ProviderResult {
    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, json_text, .{});
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return error.MalformedResponse;
    const data_value = root.object.get("data") orelse return error.MalformedResponse;
    if (data_value != .array) return error.MalformedResponse;

    var entries: std.ArrayList(model_catalog.ModelCatalogEntry) = .empty;
    errdefer model_catalog.freeModelCatalog(alloc, &entries);
    for (data_value.array.items) |item| {
        if (item != .object) return error.MalformedResponse;
        const id_value = item.object.get("id") orelse continue;
        if (id_value != .string or id_value.string.len == 0) continue;
        const id = try alloc.dupe(u8, id_value.string);
        errdefer alloc.free(id);
        const model_type = try alloc.dupe(u8, "chat");
        errdefer alloc.free(model_type);
        try entries.append(alloc, .{
            .id = id,
            .model_type = model_type,
        });
    }
    return .{ .catalog = entries };
}

test "model list URL tolerates base URL slashes and rejects plain hosts" {
    const alloc = std.testing.allocator;
    const with_slash = try modelListUrl(alloc, "https://api.groq.com/openai/v1/");
    defer alloc.free(with_slash);
    try std.testing.expectEqualStrings("https://api.groq.com/openai/v1/models", with_slash);

    try std.testing.expectError(
        error.InvalidOpenAiCompatBaseUrl,
        modelListUrl(alloc, "http://example.com/v1"),
    );
    try std.testing.expectError(
        error.OpenAiCompatBaseUrlMissing,
        modelListUrl(alloc, "  "),
    );
}

test "models response maps data entries into a catalog" {
    const alloc = std.testing.allocator;
    const json_text =
        "{\"object\":\"list\",\"data\":[{\"id\":\"llama-3.3-70b-versatile\",\"object\":\"model\"},{\"id\":\"gpt-oss-120b\",\"object\":\"model\"}]}";
    var result = try parseModels(alloc, json_text);
    defer switch (result) {
        .catalog => |*catalog| model_catalog.freeModelCatalog(alloc, catalog),
        .failure => {},
    };
    const catalog = result.catalog;
    try std.testing.expectEqual(@as(usize, 2), catalog.items.len);
    try std.testing.expectEqualStrings("llama-3.3-70b-versatile", catalog.items[0].id);
    try std.testing.expectEqualStrings("gpt-oss-120b", catalog.items[1].id);
}
