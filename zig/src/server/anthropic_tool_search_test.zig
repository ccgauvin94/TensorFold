//! Anthropic deferred loading: the tool search entries become the engine's own tool, and a deferred definition stays
//! declared (so its calls parse) while leaving the published list the template renders.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const tool_specs = @import("tool_specs.zig");
const translate = @import("anthropic_translate.zig").translate;

const head = "{\"model\":\"m\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":";

fn request(a: std.mem.Allocator, tools: []const u8) !json.Value {
    const text = try std.mem.concat(a, u8, &.{ head, tools, "}" });
    return switch (json.parse(a, text) catch return error.Parse) {
        .ok => |v| v,
        .err => return error.Parse,
    };
}

fn chat(a: std.mem.Allocator, tools: []const u8) !json.Value {
    var cx: errors.Cx = .{ .a = a };
    return translate(&cx, try request(a, tools), false) catch |e| {
        std.debug.print("refused: {s}\n", .{cx.message});
        return e;
    };
}

test "a deferred tool is declared but not published" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = try chat(a, "[{\"name\":\"get_weather\",\"description\":\"w\",\"input_schema\":{\"type\":\"object\"},\"defer_loading\":true}," ++
        "{\"name\":\"search_files\",\"description\":\"f\",\"input_schema\":{\"type\":\"object\"}}]");
    var cx: errors.Cx = .{ .a = a };
    const declared = try tool_specs.active(&cx, body.get("tools"), null);
    try std.testing.expectEqual(@as(usize, 2), declared.len);
    try std.testing.expect(tool_specs.deferred(declared[0]));
    try std.testing.expect(!tool_specs.deferred(declared[1]));
    // parsing keeps both; the prompt shows only the one that was not deferred
    try std.testing.expect(tool_specs.anyDeferred(declared));
    const shown = try tool_specs.published(a, declared);
    try std.testing.expectEqual(@as(usize, 1), shown.len);
    try std.testing.expectEqualStrings("search_files", shown[0].get("function").?.get("name").?.string);
}

test "the tool search entries translate to the engine's own tool" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = try chat(a, "[{\"type\":\"tool_search_tool_regex_20251119\",\"name\":\"tool_search_tool_regex\"}," ++
        "{\"type\":\"tool_search_tool_bm25_20251119\"}," ++
        "{\"name\":\"get_weather\",\"input_schema\":{\"type\":\"object\"},\"defer_loading\":true}]");
    const tools = body.get("tools").?;
    try std.testing.expectEqual(@as(usize, 3), tools.array.len);

    const regex = tools.array[0];
    try std.testing.expectEqualStrings("tool_search_tool_regex", regex.get("function").?.get("name").?.string);
    try std.testing.expectEqualStrings("tool_search_tool_regex_20251119", regex.get("x_tensorfold_server_tool").?.string);
    const regex_props = regex.get("function").?.get("parameters").?.get("properties").?;
    try std.testing.expect(regex_props.get("pattern") != null);
    try std.testing.expect(regex_props.get("limit") != null);

    // the bm25 variant names itself and asks for a query rather than a pattern
    const bm25 = tools.array[1];
    try std.testing.expectEqualStrings("tool_search_tool_bm25", bm25.get("function").?.get("name").?.string);
    const bm25_props = bm25.get("function").?.get("parameters").?.get("properties").?;
    try std.testing.expect(bm25_props.get("query") != null);
    try std.testing.expect(bm25_props.get("pattern") == null);

    // a search tool is never deferred, so both stay published beside the one visible custom tool
    var cx: errors.Cx = .{ .a = a };
    const declared = try tool_specs.active(&cx, body.get("tools"), null);
    const shown = try tool_specs.published(a, declared);
    try std.testing.expectEqual(@as(usize, 2), shown.len);
}

test "all tools deferred is refused with Anthropic's message" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const raw = try request(a, "[{\"name\":\"get_weather\",\"input_schema\":{\"type\":\"object\"},\"defer_loading\":true}," ++
        "{\"name\":\"search_files\",\"input_schema\":{\"type\":\"object\"},\"defer_loading\":true}]");
    try std.testing.expectError(error.Refused, translate(&cx, raw, false));
    try std.testing.expectEqualStrings("At least one tool must have defer_loading=false. All tools cannot be deferred.", cx.message);
}

test "a deferred tool search tool is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const raw = try request(a, "[{\"type\":\"tool_search_tool_regex_20251119\",\"defer_loading\":true}]");
    try std.testing.expectError(error.Refused, translate(&cx, raw, false));
    try std.testing.expectEqualStrings("the tool search tool cannot be deferred", cx.message);
}

test "an unsupported tool type is still refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const raw = try request(a, "[{\"type\":\"computer_20250124\",\"name\":\"computer\"}]");
    try std.testing.expectError(error.Refused, translate(&cx, raw, false));
}

test "the search tool names what it can find, so the model can aim" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = try chat(a, "[{\"type\":\"tool_search_tool_regex_20251119\",\"name\":\"tool_search_tool_regex\"}," ++
        "{\"name\":\"get_weather\",\"input_schema\":{\"type\":\"object\"},\"defer_loading\":true}," ++
        "{\"name\":\"send_email\",\"input_schema\":{\"type\":\"object\"},\"defer_loading\":true}," ++
        "{\"name\":\"search_files\",\"input_schema\":{\"type\":\"object\"}}]");

    const search = body.get("tools").?.array[0];
    const said = search.get("function").?.get("description").?.string;
    // the deferred names are published in the search tool's own description; the eager one is not listed
    try std.testing.expect(std.mem.indexOf(u8, said, "get_weather, send_email") != null);
    try std.testing.expect(std.mem.indexOf(u8, said, "search_files") == null);
}
