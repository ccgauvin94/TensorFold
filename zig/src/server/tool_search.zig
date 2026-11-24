//! Anthropic's tool search (``tool_search_tool_regex_20251119`` / ``tool_search_tool_bm25_20251119``): the search a
//! deferred request's tools answer, and the reply blocks the contract defines for it. Pure functions over the
//! request's own tool list, so a search costs nothing but a walk.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// Which search tool the request offered: the regex variant takes a Python ``re.search`` pattern, the bm25 one a
/// natural-language query. Both search names, descriptions, argument names and argument descriptions.
pub const Variant = enum { regex, bm25 };

/// Anthropic's default and bounds for ``limit``.
pub const default_limit: usize = 5;
pub const max_limit: usize = 10000;
pub const max_pattern: usize = 200;
pub const max_query: usize = 500;

/// A search that could not run: the contract reports these in the result block with a 200, not as a request error.
pub const Failure = struct { code: []const u8, message: []const u8 };

/// What a search call produced: the tools it found (by name, best first), or why it did not run.
pub const Outcome = union(enum) { found: []const []const u8, failed: Failure };

/// ``invalid_tool_input``: the one code a malformed pattern or query reports.
const bad_input = "invalid_tool_input";

fn fail(code: []const u8, message: []const u8) Outcome {
    return .{ .failed = .{ .code = code, .message = message } };
}

const Limit = union(enum) { ok: usize, bad: Failure };

/// The search a call's arguments ask for: ``limit`` is 1 to 10000, ``5`` when absent.
fn limitOf(args: Value) Limit {
    const raw = if (args == .object) args.get("limit") else null;
    const v = raw orelse return .{ .ok = default_limit };
    if (v == .null) return .{ .ok = default_limit };
    const n = v.int64() orelse return .{ .bad = .{ .code = bad_input, .message = "limit must be an integer" } };
    if (n < 1 or n > max_limit) return .{ .bad = .{ .code = bad_input, .message = "limit must be between 1 and 10000" } };
    return .{ .ok = @intCast(n) };
}

/// Runs ``args`` (``{pattern|query, limit}``) against ``tools``, as the request declared them.
pub fn run(a: Allocator, tools: []const Value, variant: Variant, args: Value) Allocator.Error!Outcome {
    const limit = switch (limitOf(args)) {
        .ok => |n| n,
        .bad => |f| return .{ .failed = f },
    };
    const leaf = switch (variant) {
        .regex => "pattern",
        .bm25 => "query",
    };
    const text = blk: {
        const v = if (args == .object) args.get(leaf) else null;
        const s = v orelse return fail(bad_input, "the search input must include its query");
        if (s != .string or s.string.len == 0) return fail(bad_input, "the search input must be a non-empty string");
        break :blk s.string;
    };
    return switch (variant) {
        .regex => regex(a, tools, text, limit),
        .bm25 => bm25(a, tools, text, limit),
    };
}

/// The text a tool is searched by: its name, description, and every argument's name and description.
fn fields(a: Allocator, tool: Value) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    const function = if (tool.get("function")) |f| (if (f == .object) f else tool) else tool;
    if (function.get("name")) |n| {
        if (n == .string) try out.appendSlice(a, n.string);
    }
    if (function.get("description")) |d| {
        if (d == .string) {
            try out.append(a, ' ');
            try out.appendSlice(a, d.string);
        }
    }
    if (function.get("parameters")) |p| {
        if (p == .object) {
            if (p.get("properties")) |props| {
                if (props == .object) {
                    var it = props.object.iterator();
                    while (it.next()) |entry| {
                        try out.append(a, ' ');
                        try out.appendSlice(a, entry.key_ptr.*);
                        if (entry.value_ptr.* == .object) {
                            if (entry.value_ptr.get("description")) |d| {
                                if (d == .string) {
                                    try out.append(a, ' ');
                                    try out.appendSlice(a, d.string);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return out.items;
}

fn nameOf(tool: Value) []const u8 {
    const function = if (tool.get("function")) |f| (if (f == .object) f else tool) else tool;
    const n = function.get("name") orelse return "";
    return if (n == .string) n.string else "";
}

/// ---------------------------------------------------------------------------------------------------------------
/// BM25: the natural-language variant. Tokens are runs of alphanumerics with ``_`` and ``-`` as separators, so
/// ``get_weather`` is found by ``weather``; the whole name is searched too.
/// ---------------------------------------------------------------------------------------------------------------

const Hits = struct { name: []const u8, score: f64 };

fn tokens(a: Allocator, text: []const u8) Allocator.Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var start: ?usize = null;
    for (text, 0..) |ch, i| {
        const word = std.ascii.isAlphanumeric(ch);
        if (word and start == null) start = i;
        if (!word and start != null) {
            try out.append(a, try std.ascii.allocLowerString(a, text[start.?..i]));
            start = null;
        }
    }
    if (start) |s| try out.append(a, try std.ascii.allocLowerString(a, text[s..]));
    return out.items;
}

fn count(haystack: []const []const u8, needle: []const u8) f64 {
    var n: f64 = 0;
    for (haystack) |t| if (std.mem.eql(u8, t, needle)) {
        n += 1;
    };
    return n;
}

fn bm25(a: Allocator, tools: []const Value, query: []const u8, limit: usize) Allocator.Error!Outcome {
    if (query.len > max_query) return fail(bad_input, "query must be at most 500 characters");
    const terms = try tokens(a, query);
    if (terms.len == 0) return .{ .found = &.{} };

    // one token list per tool, and the average length BM25 normalises by
    const counts = try a.alloc([]const []const u8, tools.len);
    var total: f64 = 0;
    for (tools, counts) |tool, *slot| {
        slot.* = try tokens(a, try fields(a, tool));
        total += @floatFromInt(slot.len);
    }
    const average = if (tools.len == 0) 0 else total / @as(f64, @floatFromInt(tools.len));

    var hits: std.ArrayList(Hits) = .empty;
    const k1: f64 = 1.2;
    const b: f64 = 0.75;
    for (tools, counts) |tool, doc| {
        if (doc.len == 0) continue;
        var score: f64 = 0;
        for (terms) |term| {
            const tf = count(doc, term);
            if (tf == 0) continue;
            var df: f64 = 0;
            for (counts) |other| {
                if (count(other, term) > 0) df += 1;
            }
            const idf = std.math.log1p((@as(f64, @floatFromInt(tools.len)) - df + 0.5) / (df + 0.5));
            const len = @as(f64, @floatFromInt(doc.len));
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len / average));
        }
        if (score > 0) try hits.append(a, .{ .name = nameOf(tool), .score = score });
    }
    return .{ .found = try rank(a, hits, limit) };
}

/// The best ``limit`` hits, ties broken by name so a search is deterministic.
fn rank(a: Allocator, hits: std.ArrayList(Hits), limit: usize) Allocator.Error![]const []const u8 {
    const less = struct {
        fn before(_: void, x: Hits, y: Hits) bool {
            if (x.score != y.score) return x.score > y.score;
            return std.mem.lessThan(u8, x.name, y.name);
        }
    }.before;
    std.mem.sort(Hits, hits.items, {}, less);
    const n = @min(limit, hits.items.len);
    const out = try a.alloc([]const u8, n);
    for (hits.items[0..n], out) |hit, *slot| slot.* = hit.name;
    return out;
}

/// ---------------------------------------------------------------------------------------------------------------
/// The regex variant: ``re.search`` semantics, case-insensitive, over the same fields. Supported: literals, ``.``,
/// ``*``, ``+``, ``?``, ``[...]`` classes (ranges and ``^``), ``^``/``$`` anchors, ``|`` alternation and ``\``
/// escapes. Constructs Python has and this does not (groups, backreferences, ``{m,n}``, lookaround, ``\b``) are
/// reported as ``invalid_tool_input`` rather than silently matching something else.
/// ---------------------------------------------------------------------------------------------------------------

fn regex(a: Allocator, tools: []const Value, pattern: []const u8, limit: usize) Allocator.Error!Outcome {
    if (pattern.len > max_pattern) return fail(bad_input, "pattern must be at most 200 characters");
    // `prefix(a|b|c)` is what a model actually writes; expanding it keeps the matcher free of groups
    const expanded = (try expandGroups(a, pattern)) orelse return fail(bad_input, "the pattern uses a construct this search does not support: use literals, ., *, +, ?, [...], ^, $ and |");
    if (!supported(expanded)) return fail(bad_input, "the pattern uses a construct this search does not support: use literals, ., *, +, ?, [...], ^, $ and |");
    var hits: std.ArrayList(Hits) = .empty;
    for (tools) |tool| {
        const text = try fields(a, tool);
        if (matches(expanded, text)) try hits.append(a, .{ .name = nameOf(tool), .score = 1 });
    }
    return .{ .found = try rank(a, hits, limit) };
}

/// Rewrites the groups a model writes into plain alternatives — `k_ir__(now|timeline)` becomes
/// `k_ir__now|k_ir__timeline` — so the matcher never has to implement one. Null for anything it
/// cannot express that way (nested, quantified, unbalanced, or a cross-product past the budget),
/// which is reported rather than guessed at.
fn expandGroups(a: Allocator, pattern: []const u8) Allocator.Error!?[]const u8 {
    var current: std.ArrayList([]const u8) = .empty;
    try current.append(a, pattern);
    var budget: usize = 64;
    while (true) {
        var changed = false;
        var next: std.ArrayList([]const u8) = .empty;
        for (current.items) |alt| {
            const open = std.mem.indexOfScalar(u8, alt, '(') orelse {
                try next.append(a, alt);
                continue;
            };
            var close: ?usize = null;
            var i = open + 1;
            while (i < alt.len) : (i += 1) {
                if (alt[i] == '(') return null; // nested: not expressible here
                if (alt[i] == ')') {
                    close = i;
                    break;
                }
            }
            const end = close orelse return null;
            if (end + 1 < alt.len and (alt[end + 1] == '*' or alt[end + 1] == '+' or alt[end + 1] == '?')) return null;
            const head = alt[0..open];
            const body = alt[open + 1 .. end];
            const tail = alt[end + 1 ..];
            var choices = std.mem.splitScalar(u8, body, '|');
            while (choices.next()) |choice| {
                if (budget == 0) return null;
                budget -= 1;
                try next.append(a, try std.mem.concat(a, u8, &.{ head, choice, tail }));
            }
            changed = true;
        }
        current = next;
        if (!changed) break;
    }
    var out: std.ArrayList(u8) = .empty;
    for (current.items, 0..) |alt, i| {
        if (i > 0) try out.append(a, '|');
        try out.appendSlice(a, alt);
    }
    return out.items;
}

/// False when the pattern uses something this matcher does not implement.
fn supported(pattern: []const u8) bool {
    var i: usize = 0;
    while (i < pattern.len) : (i += 1) switch (pattern[i]) {
        '\\' => {
            i += 1;
            if (i == pattern.len) return false;
            // escapes are literals; the classes Python spells with one (\d, \w, \b) are not implemented
            if (std.ascii.isAlphanumeric(pattern[i])) return false;
        },
        '(', ')', '{', '}' => return false,
        else => {},
    };
    return true;
}

fn lower(ch: u8) u8 {
    return std.ascii.toLower(ch);
}

fn atomLen(pattern: []const u8, at: usize) usize {
    return switch (pattern[at]) {
        '\\' => 2,
        '[' => blk: {
            var i = at + 1;
            if (i < pattern.len and pattern[i] == '^') i += 1;
            if (i < pattern.len and pattern[i] == ']') i += 1; // a leading ] is a member
            while (i < pattern.len and pattern[i] != ']') i += 1;
            break :blk @min(i + 1, pattern.len) - at;
        },
        else => 1,
    };
}

fn atomMatches(pattern: []const u8, at: usize, ch: u8) bool {
    return switch (pattern[at]) {
        '.' => true,
        '\\' => at + 1 < pattern.len and lower(pattern[at + 1]) == lower(ch),
        '[' => blk: {
            var i = at + 1;
            var negated = false;
            if (i < pattern.len and pattern[i] == '^') {
                negated = true;
                i += 1;
            }
            var hit = false;
            while (i < pattern.len and pattern[i] != ']') {
                if (i + 2 < pattern.len and pattern[i + 1] == '-' and pattern[i + 2] != ']') {
                    if (lower(ch) >= lower(pattern[i]) and lower(ch) <= lower(pattern[i + 2])) hit = true;
                    i += 3;
                } else {
                    if (lower(pattern[i]) == lower(ch)) hit = true;
                    i += 1;
                }
            }
            break :blk hit != negated;
        },
        else => lower(pattern[at]) == lower(ch),
    };
}

/// ``re.search``: an unanchored match somewhere in ``text``, or one at its start after ``^``.
fn matches(pattern: []const u8, text: []const u8) bool {
    var rest = pattern;
    while (rest.len > 0) {
        const cut = std.mem.indexOfScalar(u8, rest, '|') orelse rest.len;
        if (alternative(rest[0..cut], text)) return true;
        rest = if (cut == rest.len) "" else rest[cut + 1 ..];
    }
    return false;
}

fn alternative(pattern: []const u8, text: []const u8) bool {
    if (pattern.len == 0) return true;
    if (pattern[0] == '^') {
        var i: usize = 1;
        while (i < pattern.len and pattern[i] == '^') i += 1;
        return from(pattern, i, text, 0);
    }
    var start: usize = 0;
    while (start <= text.len) : (start += 1) if (from(pattern, 0, text, start)) return true;
    return false;
}

fn from(pattern: []const u8, at: usize, text: []const u8, ti: usize) bool {
    if (at == pattern.len) return true; // `re.search` needs no end anchor
    if (pattern[at] == '$' and at + 1 == pattern.len) return ti == text.len;
    const end = at + atomLen(pattern, at);
    const quant = if (end < pattern.len) pattern[end] else 0;
    switch (quant) {
        '*' => {
            var k = ti;
            while (true) {
                if (from(pattern, end + 1, text, k)) return true;
                if (k >= text.len or !atomMatches(pattern, at, text[k])) return false;
                k += 1;
            }
        },
        '+' => {
            if (ti >= text.len or !atomMatches(pattern, at, text[ti])) return false;
            var k = ti + 1;
            while (true) {
                if (from(pattern, end + 1, text, k)) return true;
                if (k >= text.len or !atomMatches(pattern, at, text[k])) return false;
                k += 1;
            }
        },
        '?' => {
            if (ti < text.len and atomMatches(pattern, at, text[ti]) and from(pattern, end + 1, text, ti + 1)) return true;
            return from(pattern, end + 1, text, ti);
        },
        else => {
            if (ti >= text.len or !atomMatches(pattern, at, text[ti])) return false;
            return from(pattern, end, text, ti + 1);
        },
    }
}

/// ---------------------------------------------------------------------------------------------------------------
/// The reply blocks the contract defines. A search the engine ran reports itself as ``server_tool_use`` and its
/// references as ``tool_search_tool_result``; the client must echo both back and never answer them.
/// ---------------------------------------------------------------------------------------------------------------

/// ``{type: "server_tool_use", id, name, input}``: the engine's own call, never the client's.
pub fn serverToolUse(a: Allocator, id: []const u8, name: []const u8, input: Value) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = "server_tool_use" });
    try o.put(a, "id", .{ .string = id });
    try o.put(a, "name", .{ .string = name });
    try o.put(a, "input", input);
    return .{ .object = o };
}

/// ``{type: "tool_search_tool_result", tool_use_id, content:{type, tool_references}}`` for a search that ran.
pub fn result(a: Allocator, tool_use_id: []const u8, names: []const []const u8) Allocator.Error!Value {
    const refs = try a.alloc(Value, names.len);
    for (names, refs) |name, *slot| {
        const ref = try json.newObject(a);
        try ref.put(a, "type", .{ .string = "tool_reference" });
        try ref.put(a, "tool_name", .{ .string = name });
        slot.* = .{ .object = ref };
    }
    const content = try json.newObject(a);
    try content.put(a, "type", .{ .string = "tool_search_tool_search_result" });
    try content.put(a, "tool_references", .{ .array = refs });
    return resultBlock(a, tool_use_id, .{ .object = content });
}

/// The same block for a search that could not run: a 200 with the contract's ``error_code`` and message.
pub fn failure(a: Allocator, tool_use_id: []const u8, why: Failure) Allocator.Error!Value {
    const content = try json.newObject(a);
    try content.put(a, "type", .{ .string = "tool_search_tool_result_error" });
    try content.put(a, "error_code", .{ .string = why.code });
    try content.put(a, "error_message", .{ .string = why.message });
    return resultBlock(a, tool_use_id, .{ .object = content });
}

fn resultBlock(a: Allocator, tool_use_id: []const u8, content: Value) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = "tool_search_tool_result" });
    try o.put(a, "tool_use_id", .{ .string = tool_use_id });
    try o.put(a, "content", content);
    return .{ .object = o };
}

/// ---------------------------------------------------------------------------------------------------------------
/// Expansion: the definitions a search's references stand for, as the text a mid-conversation system message
/// carries. Anthropic appends a ``tool_reference`` inline and expands it before the model sees it; here the
/// expansion is text, and it must be BYTE-IDENTICAL every time the same references come back — a later request
/// re-derives it from the result blocks in the conversation, and a single differing byte would move the prompt and
/// cost the whole prefix cache.
/// ---------------------------------------------------------------------------------------------------------------

/// The tool a reference names, as the request declared it.
fn definition(tools: []const Value, name: []const u8) ?Value {
    for (tools) |tool| if (std.mem.eql(u8, nameOf(tool), name)) return tool;
    return null;
}

/// The first reference the request holds no definition for: the contract's 400 (``Tool reference '<name>' not
/// found in available tools``), which is a client error rather than a failed search.
pub fn missing(tools: []const Value, names: []const []const u8) ?[]const u8 {
    for (names) |name| if (definition(tools, name) == null) return name;
    return null;
}

/// ``## Tool: <name>`` sections for ``names``, in the order the search returned them, each with its description
/// and its parameters as JSON Schema — the same shape the encoder already gives this family for a
/// ``## Response Format`` section, so the model reads it as a definition rather than as conversation.
pub fn expansion(a: Allocator, tools: []const Value, names: []const []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (names) |name| {
        const tool = definition(tools, name) orelse continue;
        const function = if (tool.get("function")) |f| (if (f == .object) f else tool) else tool;
        try out.appendSlice(a, "## Tool: ");
        try out.appendSlice(a, name);
        try out.append(a, '\n');
        if (function.get("description")) |d| {
            if (d == .string and d.string.len > 0) {
                try out.appendSlice(a, d.string);
                try out.append(a, '\n');
            }
        }
        if (function.get("parameters")) |p| {
            try out.appendSlice(a, "Parameters (JSON Schema): ");
            try out.appendSlice(a, try json.stringify(a, p, .{ .ascii = false, .compact = true })); // tight: fewer tokens, and stable
            try out.append(a, '\n');
        }
    }
    return out.items;
}

/// The whole system message a search leaves in the conversation: what was searched, what it found, and the
/// definitions. A search that found nothing still says so — the model must not think it hit an error.
pub fn expansionMessage(a: Allocator, role: []const u8, leaf: []const u8, query: []const u8, names: []const []const u8, tools: []const Value) Allocator.Error!Value {
    var text: std.ArrayList(u8) = .empty;
    if (names.len == 0) {
        try text.appendSlice(a, "Tool search (");
        try text.appendSlice(a, leaf);
        try text.appendSlice(a, " \"");
        try text.appendSlice(a, query);
        try text.appendSlice(a, "\") found no tools. Nothing was changed; use the tools you can already see.");
    } else {
        try text.appendSlice(a, "Tool search (");
        try text.appendSlice(a, leaf);
        try text.appendSlice(a, " \"");
        try text.appendSlice(a, query);
        try text.appendSlice(a, "\") added ");
        try text.appendSlice(a, try std.fmt.allocPrint(a, "{d} tool{s}: ", .{ names.len, if (names.len == 1) "" else "s" }));
        for (names, 0..) |name, i| {
            if (i > 0) try text.appendSlice(a, ", ");
            try text.appendSlice(a, name);
        }
        try text.appendSlice(a, ".\n\n");
        try text.appendSlice(a, try expansion(a, tools, names));
    }
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = text.items });
    return .{ .object = o };
}

/// ---------------------------------------------------------------------------------------------------------------
/// Replay fidelity. A client echoes the two blocks back and nothing else: the call's ``id``, ``name`` and
/// ``input``, and the result's ``tool_references``. It never echoes the markup the model wrote, so the prompt must
/// carry a CANONICAL rendering of the call derived from the block — the model's own DSML cannot be reconstructed,
/// and keeping it would move the prompt one byte and cost the prefix cache at exactly the point discovery happens.
/// The id and the tool's own name stay out of the text for the same reason: they are wire bookkeeping, not context.
/// ---------------------------------------------------------------------------------------------------------------

fn leafOf(variant: Variant) []const u8 {
    return switch (variant) {
        .regex => "pattern",
        .bm25 => "query",
    };
}

fn stringField(input: Value, name: []const u8) []const u8 {
    if (input != .object) return "";
    const v = input.get(name) orelse return "";
    return if (v == .string) v.string else "";
}

/// The canonical line a search leaves in the conversation: ``Tool search: pattern="weather", limit=5``. Built from
/// the call's inputs only, in one order, so the turn a live request writes and the turn a replay re-derives are the
/// same bytes.
pub fn callText(a: Allocator, variant: Variant, input: Value) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "Tool search: ");
    try out.appendSlice(a, leafOf(variant));
    try out.appendSlice(a, "=\"");
    try out.appendSlice(a, stringField(input, leafOf(variant)));
    try out.append(a, '"');
    if (input == .object) {
        if (input.get("limit")) |l| {
            if (l.int64()) |n| try out.appendSlice(a, try std.fmt.allocPrint(a, ", limit={d}", .{n}));
        }
    }
    return out.items;
}

/// The two turns a search leaves behind — the canonical call line and the expansion message — derived from the
/// call's input and the result's references alone. One function for both directions: what the round appends when
/// the search runs, and what a later request re-derives from the echoed blocks.
pub const Derived = struct { call: []const u8, expansion: Value };

pub fn derive(a: Allocator, role: []const u8, tools: []const Value, variant: Variant, input: Value, refs: []const []const u8) Allocator.Error!Derived {
    return .{
        .call = try callText(a, variant, input),
        .expansion = try expansionMessage(a, role, leafOf(variant), stringField(input, leafOf(variant)), refs, tools),
    };
}

/// The role a translated request carries where an expansion belongs. Never a role a client can send: the marker is
/// produced by the translate pass and consumed here.
pub const placeholder = "__tensorfold_tool_search";

pub fn isPlaceholder(m: Value) bool {
    if (m != .object) return false;
    const role = m.strField("role") orelse return false;
    return std.mem.eql(u8, role, placeholder);
}

/// The name a tool entry is declared under. A chat-body tool nests it under ``function``; an entry that never
/// went through translate carries it at the top level. Getting this wrong is silent: every lookup simply misses,
/// and a discovered tool stays deferred for ever (which reads as the model searching for it over and over).
pub fn declaredName(tool: Value) []const u8 {
    if (tool != .object) return "";
    if (tool.get("function")) |function| if (function == .object) {
        if (function.get("name")) |n| if (n == .string) return n.string;
    };
    if (tool.get("name")) |n| if (n == .string) return n.string;
    return "";
}

/// Clears ``defer_loading`` on the tools a search returned, in place: the request's tool entries are
/// shared with the chat body, so the template sees them as published.
pub fn publish(tools: []const Value, names: []const []const u8) void {
    for (tools) |tool| {
        if (tool != .object) continue;
        const name = declaredName(tool);
        for (names) |found| {
            if (!std.mem.eql(u8, name, found)) continue;
            _ = tool.object.orderedRemove("defer_loading");
            break;
        }
    }
}

/// Folds every placeholder a translated request carries back into turns the model reads: the canonical call line
/// into the assistant turn the search block was cut from, and the expansion message after it. ``role`` is the
/// server's late-system role, so a template that keeps a later system message gets one and any other gets a user
/// turn — the same message either way, only the role differs.
pub fn resolve(a: Allocator, cx: *errors.Cx, messages: Value, role: []const u8, tools: []const Value) errors.Refused!Value {
    if (messages != .array) return messages;
    var out: std.ArrayList(Value) = .empty;
    // A tool is defined ONCE per request, however many times it was found. The client
    // echoes every search it ever made (that is the contract), so re-deriving each result
    // restated definitions the model already had: a chat that searched three times saw the
    // same schemas re-added, and said so — "tool search keeps re-adding tools".
    var defined: std.StringHashMapUnmanaged(void) = .empty;
    for (messages.array) |m| {
        if (!isPlaceholder(m)) {
            try out.append(a, m);
            continue;
        }
        const input = m.get("input") orelse Value.null;
        const previous = if (out.items.len > 0) out.items[out.items.len - 1] else Value.null;
        const previous_role = if (previous == .object) (previous.strField("role") orelse "") else "";
        if (!std.mem.eql(u8, previous_role, "assistant")) return cx.refuse("a tool search result must follow the assistant turn that called it");

        // a search that could not run says so; the model must not read a silent gap as a successful search
        if (m.get("error_code")) |code| {
            if (code == .string) {
                const why = m.get("error_message") orelse Value.null;
                const text = try std.fmt.allocPrint(a, "Tool search failed ({s}): {s}. Nothing was added; use the tools you can already see.", .{ code.string, (if (why == .string) why.string else "") });
                out.items[out.items.len - 1] = try withCall(a, previous, try callText(a, variantOf(input), input));
                try out.append(a, try plainMessage(a, role, text));
                continue;
            }
        }

        // A search marked lost never got its result into the transcript: the turn was cut before it arrived, or
        // the client recorded two searches and answered only one. The engine ran that search once, so it can run
        // it again from the request's own tool list, and the replay then reads exactly as the live turn did.
        var refs: []const []const u8 = &.{};
        if (json.truthyField(m, "lost")) {
            switch (try run(a, tools, variantOf(input), input)) {
                .found => |names| refs = names,
                .failed => |why| {
                    const text = try std.fmt.allocPrint(a, "Tool search failed ({s}): {s}. Nothing was added; use the tools you can already see.", .{ why.code, why.message });
                    out.items[out.items.len - 1] = try withCall(a, previous, try callText(a, variantOf(input), input));
                    try out.append(a, try plainMessage(a, role, text));
                    continue;
                },
            }
        } else {
            const refs_value = m.get("references") orelse Value{ .array = &.{} };
            if (refs_value != .array) return cx.refuse("a tool search result must list its references");
            const listed = try a.alloc([]const u8, refs_value.array.len);
            for (refs_value.array, listed) |item, *slot| {
                const name = if (item == .object) (item.strField("tool_name") orelse "") else "";
                if (name.len == 0) return cx.refuse("a tool reference must name its tool");
                slot.* = name;
            }
            refs = listed;
        }
        // the contract's other 400: a reference with no definition in this request
        if (missing(tools, refs)) |name| return cx.fail(.request, "Tool reference '{s}' not found in available tools", .{name});

        // A tool this history discovered is loaded from here on: the request's own tool list stops
        // deferring it, so the template publishes it and the model can call it instead of only
        // reading about it.
        publish(tools, refs);

        const fresh = try a.alloc([]const u8, refs.len);
        var fresh_count: usize = 0;
        for (refs) |name| {
            if ((try defined.getOrPut(a, name)).found_existing) continue;
            fresh[fresh_count] = name;
            fresh_count += 1;
        }

        const derived = try derive(a, role, tools, variantOf(input), input, fresh[0..fresh_count]);
        out.items[out.items.len - 1] = try withCall(a, previous, derived.call);
        if (fresh_count > 0) {
            try out.append(a, derived.expansion);
        } else {
            // a repeat search must not read as a silent gap, and must not restate schemas
            const leaf = leafOf(variantOf(input));
            const text = try std.fmt.allocPrint(a, "Tool search ({s} \"{s}\") matched only tools that are already defined above. Nothing was added.", .{ leaf, stringField(input, leaf) });
            try out.append(a, try plainMessage(a, role, text));
        }
    }
    return .{ .array = out.items };
}

/// Which variant a set of search inputs belongs to: the bm25 tool asks for a query, the regex one for a pattern.
fn variantOf(input: Value) Variant {
    return if (input == .object and input.get("query") != null) .bm25 else .regex;
}

/// ``msg`` with ``call`` appended to its content, as the text part the template renders.
fn withCall(a: Allocator, msg: Value, call: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    if (msg == .object) {
        var it = msg.object.iterator();
        while (it.next()) |entry| {
            if (!std.mem.eql(u8, entry.key_ptr.*, "content")) try o.put(a, entry.key_ptr.*, entry.value_ptr.*);
        }
    }
    var parts: std.ArrayList(Value) = .empty;
    if (msg == .object) {
        if (msg.get("content")) |content| switch (content) {
            .string => |text| {
                if (text.len > 0) try parts.append(a, try messages_mod.textPart(a, text));
            },
            .array => |items| for (items) |item| try parts.append(a, item),
            else => {},
        };
    }
    try parts.append(a, try messages_mod.textPart(a, call));
    try o.put(a, "content", .{ .array = parts.items });
    return .{ .object = o };
}

fn plainMessage(a: Allocator, role: []const u8, text: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = text });
    return .{ .object = o };
}

/// What a request asks for that ``--tool-search`` gates: a search tool entry, or any ``defer_loading``. Null for
/// every request that knows nothing about the feature, which is what keeps an ungated server's behaviour identical.
pub fn requested(body: Value) ?[]const u8 {
    if (body != .object) return null;
    const tools = body.get("tools") orelse return null;
    if (tools != .array) return null;
    var deferred = false;
    for (tools.array) |tool| {
        if (tool != .object) continue;
        if (tool.strField("type")) |kind| {
            for (searches_types) |wanted| if (std.mem.eql(u8, kind, wanted)) return "the tool search tool";
        }
        if (json.truthyField(tool, "defer_loading")) deferred = true;
    }
    return if (deferred) "defer_loading" else null;
}

/// The two request types the tool search tools are declared with.
const searches_types = [_][]const u8{ "tool_search_tool_regex_20251119", "tool_search_tool_bm25_20251119" };
