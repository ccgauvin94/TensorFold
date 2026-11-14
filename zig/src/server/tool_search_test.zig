//! Anthropic's tool search: what a deferred request's ``tool_search_tool_*`` call finds, and the reply blocks it
//! produces. Pure functions over the request's own tool list.
const std = @import("std");
const json = @import("json");
const search = @import("tool_search.zig");
const errors = @import("errors.zig");
const tool_specs = @import("tool_specs.zig");
const translate = @import("anthropic_translate.zig").translate;

const tools_json =
    \\[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City and state"}}}}},
    \\ {"type":"function","function":{"name":"get_user_data","description":"Read a user's profile record","parameters":{"type":"object","properties":{"id":{"type":"string","description":"The user id"}}}}},
    \\ {"type":"function","function":{"name":"get_weather_data","description":"Hourly weather history","parameters":{"type":"object","properties":{"station":{"type":"string","description":"Weather station code"}}}}},
    \\ {"type":"function","function":{"name":"database_query","description":"Run SQL against the analytics database","parameters":{"type":"object","properties":{"sql":{"type":"string","description":"A query statement"}}}}},
    \\ {"type":"function","function":{"name":"send_email","description":"Send a message","parameters":{"type":"object","properties":{}}}}]
;

fn tools(a: std.mem.Allocator) ![]const json.Value {
    const raw = switch (json.parse(a, tools_json) catch return error.Parse) {
        .ok => |v| v,
        .err => return error.Parse,
    };
    return raw.array;
}

fn args(a: std.mem.Allocator, text: []const u8) !json.Value {
    return switch (json.parse(a, text) catch return error.Parse) {
        .ok => |v| v,
        .err => return error.Parse,
    };
}

fn has(list: []const []const u8, name: []const u8) bool {
    for (list) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

fn names(out: search.Outcome) ![]const []const u8 {
    return switch (out) {
        .found => |n| n,
        .failed => |f| {
            std.debug.print("search failed: {s}: {s}\n", .{ f.code, f.message });
            return error.SearchFailed;
        },
    };
}

test "bm25 finds tools by name, description and argument" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const list = try tools(a);

    const weather = try names(try search.run(a, list, .bm25, try args(a, "{\"query\":\"weather\"}")));
    try std.testing.expect(has(weather, "get_weather"));
    try std.testing.expect(has(weather, "get_weather_data"));

    // a word only one tool's description carries ranks that tool first
    const current = try names(try search.run(a, list, .bm25, try args(a, "{\"query\":\"current weather\"}")));
    try std.testing.expectEqualStrings("get_weather", current[0]);

    // an argument's description is searched too
    const stations = try names(try search.run(a, list, .bm25, try args(a, "{\"query\":\"weather station code\"}")));
    try std.testing.expectEqualStrings("get_weather_data", stations[0]);

    // a query nothing matches is an empty list, not an error
    const none = try names(try search.run(a, list, .bm25, try args(a, "{\"query\":\"quantum chromodynamics\"}")));
    try std.testing.expectEqual(@as(usize, 0), none.len);

    // limit caps the answer
    const one = try names(try search.run(a, list, .bm25, try args(a, "{\"query\":\"weather\",\"limit\":1}")));
    try std.testing.expectEqual(@as(usize, 1), one.len);
}

test "regex is unanchored, case-insensitive, and honours | and anchors" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const list = try tools(a);

    // the contract's own example: one pattern, two tools
    const data = try names(try search.run(a, list, .regex, try args(a, "{\"pattern\":\"get_.*_data\"}")));
    try std.testing.expectEqual(@as(usize, 2), data.len);

    // a description matches as well as a name
    const weather = try names(try search.run(a, list, .regex, try args(a, "{\"pattern\":\"WEATHER\"}")));
    try std.testing.expect(weather.len >= 2);

    // alternation, both orders
    const either = try names(try search.run(a, list, .regex, try args(a, "{\"pattern\":\"database.*query|query.*database\"}")));
    try std.testing.expectEqualStrings("database_query", either[0]);

    // an anchor keeps it to the prefix
    const anchored = try names(try search.run(a, list, .regex, try args(a, "{\"pattern\":\"^send\"}")));
    try std.testing.expectEqual(@as(usize, 1), anchored.len);
    try std.testing.expectEqualStrings("send_email", anchored[0]);

    // a class and a quantifier
    const cls = try names(try search.run(a, list, .regex, try args(a, "{\"pattern\":\"^get_[uw][se]\"}")));
    try std.testing.expectEqual(@as(usize, 3), cls.len);
}

test "a search that cannot run reports invalid_tool_input" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const list = try tools(a);

    // groups and the shorthand classes are Python's, not ours: refuse rather than match something else
    const grouped = try search.run(a, list, .regex, try args(a, "{\"pattern\":\"(weather|rain)\"}"));
    try std.testing.expectEqualStrings("invalid_tool_input", grouped.failed.code);

    const shorthand = try search.run(a, list, .regex, try args(a, "{\"pattern\":\"\\\\d+\"}"));
    try std.testing.expectEqualStrings("invalid_tool_input", shorthand.failed.code);

    // the pattern and query bounds the contract sets
    var long: [201]u8 = undefined;
    @memset(&long, 'a');
    const too_long = try search.run(a, list, .regex, try args(a, try std.fmt.allocPrint(a, "{{\"pattern\":\"{s}\"}}", .{long})));
    try std.testing.expectEqualStrings("invalid_tool_input", too_long.failed.code);

    const bad_limit = try search.run(a, list, .bm25, try args(a, "{\"query\":\"weather\",\"limit\":0}"));
    try std.testing.expectEqualStrings("invalid_tool_input", bad_limit.failed.code);
}

test "the reply blocks carry the contract's shapes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const input = try args(a, "{\"pattern\":\"weather\",\"limit\":5}");
    const call = try search.serverToolUse(a, "srvtoolu_01", "tool_search_tool_regex", input);
    try std.testing.expectEqualStrings("server_tool_use", call.get("type").?.string);
    try std.testing.expectEqualStrings("srvtoolu_01", call.get("id").?.string);
    try std.testing.expectEqualStrings("tool_search_tool_regex", call.get("name").?.string);

    const found = try search.result(a, "srvtoolu_01", &.{ "get_weather", "get_weather_data" });
    try std.testing.expectEqualStrings("tool_search_tool_result", found.get("type").?.string);
    try std.testing.expectEqualStrings("srvtoolu_01", found.get("tool_use_id").?.string);
    const content = found.get("content").?;
    try std.testing.expectEqualStrings("tool_search_tool_search_result", content.get("type").?.string);
    const refs = content.get("tool_references").?.array;
    try std.testing.expectEqual(@as(usize, 2), refs.len);
    try std.testing.expectEqualStrings("tool_reference", refs[0].get("type").?.string);
    try std.testing.expectEqualStrings("get_weather", refs[0].get("tool_name").?.string);

    const bad = try search.failure(a, "srvtoolu_02", .{ .code = "invalid_tool_input", .message = "Invalid regular expression pattern: missing ) at position 1" });
    const body = bad.get("content").?;
    try std.testing.expectEqualStrings("tool_search_tool_result_error", body.get("type").?.string);
    try std.testing.expectEqualStrings("invalid_tool_input", body.get("error_code").?.string);
}

test "the expansion renders the definitions, in order, byte-identically" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const list = try tools(a);

    const names_in = [_][]const u8{ "get_weather_data", "get_weather" };
    const one = try search.expansion(a, list, &names_in);
    try std.testing.expect(std.mem.indexOf(u8, one, "## Tool: get_weather_data") != null);
    try std.testing.expect(std.mem.indexOf(u8, one, "Hourly weather history") != null);
    // the schema is rendered as JSON, in the order the request declared its fields
    try std.testing.expect(std.mem.indexOf(u8, one, "Parameters (JSON Schema): {") != null);
    try std.testing.expect(std.mem.indexOf(u8, one, "\"type\":\"object\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, one, "\"station\"") != null);
    // the order the search returned them, not the request's order
    const data_at = std.mem.indexOf(u8, one, "## Tool: get_weather_data").?;
    const weather_at = std.mem.indexOf(u8, one, "## Tool: get_weather\n").?;
    try std.testing.expect(data_at < weather_at);

    // a later request re-derives this from the result blocks: a single differing byte would cost the prefix cache
    const two = try search.expansion(a, list, &names_in);
    try std.testing.expectEqualStrings(one, two);

    // a reference with no definition is the contract's client error, not a silent skip
    const ghost = [_][]const u8{ "get_weather", "no_such_tool" };
    try std.testing.expectEqualStrings("no_such_tool", search.missing(list, &ghost).?);
    try std.testing.expect(search.missing(list, &names_in) == null);
}

test "the expansion message carries the search and its definitions" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const list = try tools(a);

    const found_names = [_][]const u8{"get_weather"};
    const found = try search.expansionMessage(a, "system", "pattern", "weather", &found_names, list);
    try std.testing.expectEqualStrings("system", found.get("role").?.string);
    const text = found.get("content").?.string;
    try std.testing.expect(std.mem.indexOf(u8, text, "Tool search (pattern \"weather\") added 1 tool: get_weather.") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "## Tool: get_weather") != null);

    // nothing found says so, so the model does not read it as a failure
    const none = try search.expansionMessage(a, "system", "pattern", "quantum", &.{}, list);
    const none_text = none.get("content").?.string;
    try std.testing.expect(std.mem.indexOf(u8, none_text, "found no tools") != null);

    // and it is re-derivable: the same conversation renders the same message
    const again = try search.expansionMessage(a, "system", "pattern", "weather", &found_names, list);
    try std.testing.expectEqualStrings(text, again.get("content").?.string);
}

test "a replayed search derives exactly the turns the live request appended" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const list = try tools(a);

    const refs = [_][]const u8{"get_weather"};
    const live_input = try args(a, "{\"pattern\":\"weather\"}");

    // live: what the round writes into the conversation when it runs the search
    const live = try search.derive(a, "system", list, .regex, live_input, &refs);

    // replay: the two blocks a client echoes back — the call's input and the result's references, nothing else.
    // The id is deliberately different from the one the live request used: it never reaches the prompt.
    const blocks = try args(a,
        \\[{"type":"server_tool_use","id":"srvtoolu_replayed","name":"tool_search_tool_regex","input":{"pattern":"weather"}},
        \\ {"type":"tool_search_tool_result","tool_use_id":"srvtoolu_replayed","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"get_weather"}]}}]
    );
    const echoed_input = blocks.array[0].get("input").?;
    const echoed_refs = [_][]const u8{"get_weather"};
    const replayed = try search.derive(a, "system", list, .regex, echoed_input, &echoed_refs);

    try std.testing.expectEqualStrings(live.call, replayed.call);
    try std.testing.expectEqualStrings(live.expansion.get("content").?.string, replayed.expansion.get("content").?.string);
    try std.testing.expectEqualStrings("system", replayed.expansion.get("role").?.string);

    // bookkeeping never reaches the prompt, so a client that rewrites an id cannot move the prefix
    try std.testing.expect(std.mem.indexOf(u8, replayed.call, "srvtoolu") == null);
    try std.testing.expect(std.mem.indexOf(u8, replayed.expansion.get("content").?.string, "srvtoolu") == null);
    // the call line is the inputs, in one order, and reads as a search
    try std.testing.expectEqualStrings("Tool search: pattern=\"weather\"", replayed.call);

    // a limit is part of the inputs and so part of the canonical line
    const with_limit = try search.derive(a, "system", list, .bm25, try args(a, "{\"query\":\"weather\",\"limit\":3}"), &refs);
    try std.testing.expectEqualStrings("Tool search: query=\"weather\", limit=3", with_limit.call);
}

test "a replayed assistant turn is cut at the search, as the live request wrote it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const body = try args(a,
        \\{"model":"m","max_tokens":64,
        \\ "tools":[{"type":"tool_search_tool_regex_20251119","name":"tool_search_tool_regex"},
        \\          {"name":"get_weather","description":"Get the current weather","input_schema":{"type":"object","properties":{"location":{"type":"string"}}},"defer_loading":true}],
        \\ "messages":[{"role":"user","content":"what is the weather?"},
        \\  {"role":"assistant","content":[
        \\    {"type":"text","text":"Let me look for a tool."},
        \\    {"type":"server_tool_use","id":"srvtoolu_1","name":"tool_search_tool_regex","input":{"pattern":"weather"}},
        \\    {"type":"tool_search_tool_result","tool_use_id":"srvtoolu_1","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"get_weather"}]}},
        \\    {"type":"text","text":"Found it."},
        \\    {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"location":"Bozeman"}}]},
        \\  {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"12C"}]}]}
    );

    var cx: errors.Cx = .{ .a = a };
    const chat = try translate(&cx, body, false);
    const declared = try tool_specs.active(&cx, chat.get("tools"), null);

    const resolved = try search.resolve(a, &cx, chat.get("messages").?, "system", declared);
    const msgs = resolved.array;
    try std.testing.expectEqual(@as(usize, 5), msgs.len);

    // the assistant turn stops at the search: its prose and the canonical call line, and no tool call
    const first = msgs[1];
    try std.testing.expectEqualStrings("assistant", first.get("role").?.string);
    try std.testing.expect(first.get("tool_calls") == null);
    const parts = first.get("content").?.array;
    try std.testing.expectEqual(@as(usize, 2), parts.len);
    try std.testing.expectEqualStrings("Let me look for a tool.", parts[0].get("text").?.string);
    try std.testing.expectEqualStrings("Tool search: pattern=\"weather\"", parts[1].get("text").?.string);

    // then the expansion, as the late system message the template keeps
    const expansion = msgs[2];
    try std.testing.expectEqualStrings("system", expansion.get("role").?.string);
    const text = expansion.get("content").?.string;
    try std.testing.expect(std.mem.indexOf(u8, text, "## Tool: get_weather") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "Get the current weather") != null);

    // and the reply continues in a new assistant turn, which is where the client's own tool call lives
    const continued = msgs[3];
    try std.testing.expectEqualStrings("assistant", continued.get("role").?.string);
    try std.testing.expectEqualStrings("Found it.", continued.get("content").?.array[0].get("text").?.string);
    try std.testing.expectEqualStrings("get_weather", continued.get("tool_calls").?.array[0].get("function").?.get("name").?.string);

    // the tool result still ends the conversation as a tool turn
    try std.testing.expectEqualStrings("tool", msgs[4].get("role").?.string);
}

test "the gate sees what a request asks for and nothing else" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const plain = try args(a, "{\"tools\":[{\"name\":\"get_weather\",\"input_schema\":{\"type\":\"object\"}}]}");
    try std.testing.expect(search.requested(plain) == null);

    const deferred = try args(a, "{\"tools\":[{\"name\":\"get_weather\",\"input_schema\":{\"type\":\"object\"},\"defer_loading\":true}]}");
    try std.testing.expectEqualStrings("defer_loading", search.requested(deferred).?);

    const searching = try args(a, "{\"tools\":[{\"type\":\"tool_search_tool_bm25_20251119\"}]}");
    try std.testing.expectEqualStrings("the tool search tool", search.requested(searching).?);

    // a request with no tools at all is not asking for anything
    try std.testing.expect(search.requested(try args(a, "{}")) == null);
    try std.testing.expect(search.requested(try args(a, "{\"tools\":[]}")) == null);
}
