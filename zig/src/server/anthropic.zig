//! Anthropic's Messages routes over the chat handler: typed SSE events, tool_use blocks, count_tokens.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const fields = @import("fields.zig");
const messages_mod = @import("messages.zig");
const openai = @import("openai.zig");
const prompt = @import("prompt.zig");
const routes = @import("routes.zig");
const sse = @import("sse.zig");
const tool_specs = @import("tool_specs.zig");
const tool_search = @import("tool_search.zig");
const ids = @import("ids.zig");
const auth = @import("auth.zig");
const translate = @import("anthropic_translate.zig").translate;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub fn route(path: []const u8) bool {
    const p = auth.routePath(path);
    for ([_][]const u8{ "/v1/messages", "/messages", "/v1/messages/count_tokens", "/messages/count_tokens" }) |r| if (std.mem.eql(u8, p, r)) return true;
    return false;
}

/// Anthropic's error object for an HTTP status.
fn errorValue(a: Allocator, message: []const u8, status: u16) Allocator.Error!Value {
    const kind = switch (status) {
        400 => "invalid_request_error",
        404 => "not_found_error",
        503 => "overloaded_error",
        else => "api_error",
    };
    const inner = try json.newObject(a);
    try inner.put(a, "type", .{ .string = kind });
    try inner.put(a, "message", .{ .string = message });
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = "error" });
    try o.put(a, "error", .{ .object = inner });
    return .{ .object = o };
}

fn sendError(conn: *Conn, a: Allocator, message: []const u8, status: u16) void {
    routes.sendValue(conn, a, status, errorValue(a, message, status) catch return);
}

/// Rendered like the chat route renders, without generating: the prompt's token count.
fn countTokens(srv: *Server, cx: *Cx, chat: Value) errors.Refused!usize {
    const thinking = try fields.thinkingFields(cx, chat, srv.effort_levels);
    const on = thinking.enable orelse srv.config.enable_thinking;
    cx.clear_thinking = thinking.clear;
    const declared = try tool_specs.active(cx, chat.get("tools"), chat.get("tool_choice"));
    // a deferred tool is declared but not shown: the count is what the prompt renders
    const tools = if (tool_specs.anyDeferred(declared)) try tool_specs.published(cx.a, declared) else declared;
    // a search block leaves a placeholder until the request's tools are known, and the count is what renders
    const resolved = try tool_search.resolve(cx.a, cx, chat.get("messages") orelse Value.null, srv.late_system, declared);
    const msgs = try messages_mod.normalizeWith(cx, resolved, "system", srv.needs_user_after_tool, .of(srv.info.vision));
    const ids_ = prompt.renderIds(srv, cx, msgs, tools, on, srv.effortFor(thinking.effort), true) catch |e| switch (e) {
        error.Refused => return cx.other(cx.message),
        else => |x| return x,
    };
    return ids_.len;
}

pub fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    const count = std.mem.endsWith(u8, auth.routePath(conn.path), "/count_tokens");
    var cx: Cx = .{ .a = a };
    const prepared = blk: {
        const raw = switch (http_body.read(conn, a, http_body.limit) catch return) {
            .ok => |b| b,
            .refused => |m| return sendError(conn, a, m, 400),
        };
        const body = switch (json.parse(a, if (raw.len == 0) "{}" else raw) catch return) {
            .ok => |v| v,
            .err => |m| return sendError(conn, a, m, 400),
        };
        // deferred loading is opt-in: without the flag this server answers what it always did, and says so rather
        // than quietly loading every deferred definition into the prompt
        if (!srv.config.tool_search) if (tool_search.requested(body)) |what| {
            const message = std.fmt.allocPrint(a, "{s} is not enabled on this server (--tool-search)", .{what}) catch return;
            return sendError(conn, a, message, 400);
        };
        const chat = translate(&cx, body, count) catch return sendError(conn, a, cx.message, cx.status());
        // Anthropic deferred loading: a replayed search carries its blocks as placeholders until the request's own
        // tools can expand them, and the expansion must sit where the live request put it or the prefix moves.
        const declared = tool_specs.active(&cx, chat.get("tools"), chat.get("tool_choice")) catch |e| switch (e) {
            error.Refused => return sendError(conn, a, cx.message, if (cx.kind == .capacity) 503 else 400),
            else => return,
        };
        const resolved = tool_search.resolve(a, &cx, chat.get("messages") orelse Value.null, srv.late_system, declared) catch |e| switch (e) {
            error.Refused => return sendError(conn, a, cx.message, if (cx.kind == .capacity) 503 else 400),
            else => return,
        };
        chat.object.put(a, "messages", resolved) catch return;
        if (count) {
            const n = countTokens(srv, &cx, chat) catch return sendError(conn, a, cx.message, if (cx.kind == .capacity) 503 else 400);
            const o = json.newObject(a) catch return;
            o.put(a, "input_tokens", json.intValue(a, n) catch return) catch return;
            return routes.sendValue(conn, a, 200, .{ .object = o });
        }
        break :blk .{ chat, body };
    };
    const body_streamed = json.truthyField(prepared[1], "stream");
    const declared = tool_specs.active(&cx, prepared[0].get("tools"), prepared[0].get("tool_choice")) catch {
        return sendError(conn, a, cx.message, 400);
    };
    const chat_body = prepared[0]; // the maps inside it are mutated between rounds

    if (body_streamed) return streamed(srv, conn, a, prepared[1], declared, chat_body);

    // Not streamed: the rounds are assembled and answered whole, and the search blocks sit in front of the
    // content the last round produced.
    var blocks: std.ArrayList(Value) = .empty;
    var rounds: usize = 0;
    while (true) : (rounds += 1) {
        var reply: Reply = .{ .a = a, .conn = conn, .model = openai.replyModel(srv, prepared[1]) };
        reply.init() catch return;
        var wire: Wire = .{ .reply = &reply };
        openai.run(srv, a, wire.out(), .{ .conn = conn }, true, chat_body);
        if (wire.opened or wire.status == null or conn.broken) return;
        const payload = wire.payload;
        if (wire.status.? != 200) {
            const problem = payload.get("error") orelse Value.null;
            const message = if (problem == .object) (if (problem.get("message")) |m| (if (m == .string) m.string else "request failed") else "request failed") else (if (problem == .string) problem.string else "request failed");
            return sendError(conn, a, message, wire.status.?);
        }
        const answered = answerSearchRound(a, srv, &blocks, declared, payload, chat_body) catch return;
        if (answered) {
            if (rounds >= max_search_rounds) return sendError(conn, a, "the model kept searching for tools without answering", 503);
            continue;
        }
        const whole = reply.completion(payload) catch |e| {
            const message = std.fmt.allocPrint(a, "invalid model response: {s}", .{reply.problem orelse @errorName(e)}) catch return;
            return sendError(conn, a, message, 500);
        };
        splice(a, whole, blocks.items) catch return;
        return routes.sendValue(conn, a, 200, whole);
    }
}

/// A streamed reply, including one that calls a search: the message is opened once and every round continues it,
/// so the deltas reach the client as the model produces them instead of arriving together at the end.
fn streamed(srv: *Server, conn: *Conn, a: Allocator, body: Value, declared: []const Value, chat_body: Value) void {
    var reply: Reply = .{ .a = a, .conn = conn, .model = openai.replyModel(srv, body), .declared = declared };
    reply.init() catch return;
    var wire: Wire = .{ .reply = &reply };
    var rounds: usize = 0;
    while (true) : (rounds += 1) {
        reply.prose.clearRetainingCapacity();
        reply.search = null;
        openai.run(srv, a, wire.out(), .{ .conn = conn }, true, chat_body);
        if (conn.broken) return;
        if (wire.status) |status| {
            // the runner answered whole although the request streamed: emit that instead of events
            if (status != 200) {
                const problem = wire.payload.get("error") orelse Value.null;
                const message = if (problem == .object) (if (problem.get("message")) |m| (if (m == .string) m.string else "request failed") else "request failed") else (if (problem == .string) problem.string else "request failed");
                return sendError(conn, a, message, status);
            }
            const whole = reply.completion(wire.payload) catch return;
            return routes.sendValue(conn, a, 200, whole);
        }
        const input = reply.search orelse return; // the runner closed the message
        if (rounds >= max_search_rounds) {
            const e = errorValue(a, "the model kept searching for tools without answering", 503) catch return;
            sse.event(conn, a, "error", e) catch {};
            return;
        }
        const variant: tool_search.Variant = if (input == .object and input.get("query") != null) .bm25 else .regex;
        const outcome = tool_search.run(a, declared, variant, input) catch return;
        const id = std.fmt.allocPrint(a, "srvtoolu_{d}", .{rounds + 1}) catch return;
        // the call block is already on the wire; what the contract wants next is its result, whole
        const result_block = switch (outcome) {
            .found => |names| tool_search.result(a, id, names) catch return,
            .failed => |why| tool_search.failure(a, id, why) catch return,
        };
        reply.wholeBlock(result_block) catch return;
        extendForSearch(a, srv, declared, chat_body, reply.prose.items, variant, input, outcome) catch return;
    }
}



/// Whether this request declares one of the engine's own search tools.
fn anyServerTool(declared: []const Value) bool {
    for (declared) |tool| if (tool.get("x_tensorfold_server_tool") != null) return true;
    return false;
}

/// A finished message as the event stream the same reply would have produced: the search blocks land mid-message
/// and the rounds after them continue it, so a reply that may carry one is assembled first and then emitted.
fn emitAsSse(conn: *Conn, a: Allocator, whole: Value) void {
    if (whole != .object) return;
    sse.open(conn) catch return;
    const head = json.copyObject(a, whole.object) catch return;
    head.put(a, "content", .{ .array = &.{} }) catch return;
    head.put(a, "stop_reason", .null) catch return;
    head.put(a, "stop_sequence", .null) catch return;
    const opened = json.newObject(a) catch return;
    opened.put(a, "type", .{ .string = "message_start" }) catch return;
    opened.put(a, "message", .{ .object = head }) catch return;
    sse.event(conn, a, "message_start", .{ .object = opened }) catch return;

    const content = whole.get("content") orelse Value{ .array = &.{} };
    if (content == .array) {
        for (content.array, 0..) |block, at| {
        const index: i64 = @intCast(at);
        const kind = if (block == .object) (block.strField("type") orelse "") else "";
        const start = json.newObject(a) catch return;
        start.put(a, "type", .{ .string = "content_block_start" }) catch return;
        start.put(a, "index", json.intValue(a, index) catch return) catch return;
        start.put(a, "content_block", block) catch return;
        sse.event(conn, a, "content_block_start", .{ .object = start }) catch return;

        // the block's changing part arrives as deltas on the block the start opened
        const delta = json.newObject(a) catch return;
        delta.put(a, "type", .{ .string = "content_block_delta" }) catch return;
        delta.put(a, "index", json.intValue(a, index) catch return) catch return;
        const inner = json.newObject(a) catch return;
        if (std.mem.eql(u8, kind, "text")) {
            inner.put(a, "type", .{ .string = "text_delta" }) catch return;
            inner.put(a, "text", block.get("text") orelse Value.null) catch return;
        } else if (std.mem.eql(u8, kind, "thinking")) {
            inner.put(a, "type", .{ .string = "thinking_delta" }) catch return;
            inner.put(a, "thinking", block.get("thinking") orelse Value.null) catch return;
        } else if (std.mem.eql(u8, kind, "tool_use")) {
            inner.put(a, "type", .{ .string = "input_json_delta" }) catch return;
            inner.put(a, "partial_json", .{ .string = json.stringify(a, block.get("input") orelse Value.null, .{ .compact = true }) catch return }) catch return;
        }
        if (inner.get("type") != null) {
            delta.put(a, "delta", .{ .object = inner }) catch return;
            sse.event(conn, a, "content_block_delta", .{ .object = delta }) catch return;
        }
        const stop = json.newObject(a) catch return;
        stop.put(a, "type", .{ .string = "content_block_stop" }) catch return;
        stop.put(a, "index", json.intValue(a, index) catch return) catch return;
            sse.event(conn, a, "content_block_stop", .{ .object = stop }) catch return;
        }
    }

    const tail = json.newObject(a) catch return;
    tail.put(a, "type", .{ .string = "message_delta" }) catch return;
    const move = json.newObject(a) catch return;
    move.put(a, "stop_reason", whole.get("stop_reason") orelse Value.null) catch return;
    move.put(a, "stop_sequence", whole.get("stop_sequence") orelse Value.null) catch return;
    tail.put(a, "delta", .{ .object = move }) catch return;
    tail.put(a, "usage", whole.get("usage") orelse Value.null) catch return;
    sse.event(conn, a, "message_delta", .{ .object = tail }) catch return;
    const end = json.newObject(a) catch return;
    end.put(a, "type", .{ .string = "message_stop" }) catch return;
    sse.event(conn, a, "message_stop", .{ .object = end }) catch return;
}

/// A search call the model made in a round: the id it gave, the tool it named, and its inputs.
const SearchCall = struct { id: Value, name: []const u8, input: Value };

/// How many searches one reply may make before it must answer (`--tool-search`): enough for a couple of hops, few
/// enough that a model that keeps searching cannot spin.
const max_search_rounds = 4;

/// Whether ``name`` is one of the tools this request declared as a server tool (a tool search).
fn serverTool(declared: []const Value, name: []const u8) bool {
    for (declared) |tool| {
        if (tool.get("x_tensorfold_server_tool") == null) continue;
        const function = tool.get("function") orelse continue;
        if (function != .object) continue;
        const declared_name = function.get("name") orelse continue;
        if (declared_name == .string and std.mem.eql(u8, declared_name.string, name)) return true;
    }
    return false;
}

/// Answers the search a finished round called, if it called one: appends the contract's two blocks to ``blocks``
/// and extends ``body`` with the turns the continuation needs — the assistant turn it produced with the call
/// rendered canonically, then the expansion. Returns false when the round called no server tool.
fn answerSearchRound(a: Allocator, srv: *Server, blocks: *std.ArrayList(Value), declared: []const Value, payload: Value, body: Value) !bool {
    const choices = payload.get("choices") orelse return false;
    if (choices != .array or choices.array.len == 0) return false;
    const message = choices.array[0].get("message") orelse return false;
    const calls = message.get("tool_calls") orelse return false;
    if (calls != .array or calls.array.len == 0) return false;

    var call: ?SearchCall = null;
    for (calls.array) |entry| {
        const function = entry.get("function") orelse continue;
        if (function != .object) continue;
        const name = function.get("name") orelse continue;
        if (name != .string or !serverTool(declared, name.string)) continue;
        const raw = function.get("arguments") orelse Value.null;
        const input = if (raw == .string) switch (try json.parseText(a, raw.string)) {
            .ok => |v| v,
            .err => Value.null,
        } else raw;
        call = .{ .id = entry.get("id") orelse Value.null, .name = name.string, .input = input };
        break;
    }
    const found = call orelse return false;

    const variant: tool_search.Variant = if (found.input == .object and found.input.get("query") != null) .bm25 else .regex;
    const outcome = try tool_search.run(a, declared, variant, found.input);
    const id = try std.fmt.allocPrint(a, "srvtoolu_{d}", .{blocks.items.len + 1});
    try blocks.append(a, try tool_search.serverToolUse(a, id, found.name, found.input));
    switch (outcome) {
        .found => |names| try blocks.append(a, try tool_search.result(a, id, names)),
        .failed => |why| try blocks.append(a, try tool_search.failure(a, id, why)),
    }

    // the continuation: everything so far, then the assistant turn as the prompt will render it on replay, then the
    // expansion in the server's late-system role
    const prose = if (message.get("content")) |p| (if (p == .string) p.string else "") else "";
    try extendForSearch(a, srv, declared, body, prose, variant, found.input, outcome);
    return true;
}

/// Extends the request with what the continuation needs: the assistant turn as the prompt will render it on replay,
/// then the expansion in the server's late-system role. Shared by the assembled and the streamed path.
fn extendForSearch(a: Allocator, srv: *Server, declared: []const Value, body: Value, prose: []const u8, variant: tool_search.Variant, input: Value, outcome: tool_search.Outcome) Allocator.Error!void {
    const messages = body.get("messages") orelse Value{ .array = &.{} };
    var out: std.ArrayList(Value) = .empty;
    if (messages == .array) for (messages.array) |m| try out.append(a, m);
    var parts: std.ArrayList(Value) = .empty;
    if (prose.len > 0) try parts.append(a, try messages_mod.textPart(a, prose));
    try parts.append(a, try messages_mod.textPart(a, try tool_search.callText(a, variant, input)));
    const turn = try json.newObject(a);
    try turn.put(a, "role", .{ .string = "assistant" });
    try turn.put(a, "content", .{ .array = parts.items });
    try out.append(a, .{ .object = turn });
    const expansion = switch (outcome) {
        .found => |names| try tool_search.derive(a, srv.late_system, declared, variant, input, names),
        .failed => null,
    };
    if (expansion) |derived| try out.append(a, derived.expansion);
    try body.object.put(a, "messages", .{ .array = out.items });
}

/// The blocks the searches produced, in front of the last round's content.
fn splice(a: Allocator, whole: Value, blocks: []const Value) Allocator.Error!void {
    if (blocks.len == 0 or whole != .object) return;
    const content = whole.get("content") orelse return;
    if (content != .array) return;
    const all = try std.mem.concat(a, Value, &.{ blocks, content.array });
    try whole.object.put(a, "content", .{ .array = all });
}

const Wire = struct {
    reply: *Reply,
    opened: bool = false,
    status: ?u16 = null,
    payload: Value = .null,

    fn out(w: *Wire) openai.Out {
        return .{ .ctx = w, .vt = &.{ .open = open, .event = event, .reply = whole, .keepalive = ping } };
    }

    fn open(ctx: *anyopaque, prompt_len: usize) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        // one message across every round: a search round is answered and the next round continues it, so a
        // second open would restart the stream the client is already reading
        if (w.opened) return;
        w.opened = true;
        // the head counts the whole prompt; message_delta's usage splits it into input and cache_read
        if (prompt_len > 0) {
            const n = json.intValue(w.reply.a, prompt_len) catch return error.Closed;
            w.reply.tokens.object.put(w.reply.a, "input_tokens", n) catch return error.Closed;
        }
        try sse.open(w.reply.conn);
        try w.reply.start();
    }

    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        return w.reply.chunk(payload);
    }

    fn whole(ctx: *anyopaque, status: u16, payload: Value) void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.status = status;
        w.payload = payload;
    }
    fn ping(ctx: *anyopaque) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        return w.reply.send("ping", try w.reply.obj("ping"));
    }
};

const Call = struct { id: []const u8 = "", name: []const u8 = "", arguments: []const u8 = "", sent: usize = 0, done: bool = false };
const Pending = struct { kind: []const u8, text: []const u8 };

/// Typed Messages events from chat chunks, each delta on a block of its own type.
const Reply = struct {
    a: Allocator,
    conn: *Conn,
    model: []const u8,
    /// The request's declared tools, so a call to one of the engine's own search tools is emitted as the
    /// `server_tool_use` block the contract asks for instead of a `tool_use` the client would try to run.
    declared: []const Value = &.{},
    /// The prose of the round in flight, which the continuation's assistant turn is built from.
    prose: std.ArrayList(u8) = .empty,
    /// Set when this round called a search: the round loop answers it and runs another round on this same
    /// reply, so the message stays open instead of being closed out at the round boundary.
    search: ?Value = null,
    base: *json.Object = undefined,
    index: i64 = -1,
    kind: ?[]const u8 = null,
    calls: std.AutoArrayHashMapUnmanaged(i64, Call) = .empty,
    pending: std.ArrayList(Pending) = .empty,
    active: ?i64 = null,
    finished: bool = false,
    finish: []const u8 = "end_turn",
    stop_sequence: Value = .null,
    tokens: Value = .null,
    problem: ?[]const u8 = null,

    const E = error{Closed};

    fn init(r: *Reply) Allocator.Error!void {
        const a = r.a;
        r.base = try json.newObject(a);
        try r.base.put(a, "id", .{ .string = try ids.make(a, "msg_", 32) });
        try r.base.put(a, "type", .{ .string = "message" });
        try r.base.put(a, "role", .{ .string = "assistant" });
        try r.base.put(a, "model", .{ .string = r.model });
        try r.base.put(a, "content", .{ .array = &.{} });
        try r.base.put(a, "stop_reason", .null);
        try r.base.put(a, "stop_sequence", .null);
        r.tokens = try usage(a, .null);
        try r.base.put(a, "usage", r.tokens);
    }

    fn send(r: *Reply, kind: []const u8, payload: *json.Object) E!void {
        return sse.event(r.conn, r.a, kind, .{ .object = payload });
    }

    fn obj(r: *Reply, kind: []const u8) E!*json.Object {
        const o = json.newObject(r.a) catch return error.Closed;
        o.put(r.a, "type", .{ .string = kind }) catch return error.Closed;
        return o;
    }

    fn put(r: *Reply, o: *json.Object, k: []const u8, v: Value) E!void {
        o.put(r.a, k, v) catch return error.Closed;
    }

    fn start(r: *Reply) E!void {
        const o = try r.obj("message_start");
        try r.put(o, "message", .{ .object = r.base });
        try r.send("message_start", o);
    }

    fn close(r: *Reply) E!void {
        const kind = r.kind orelse return;
        if (std.mem.eql(u8, kind, "thinking")) {
            const d = try r.obj("signature_delta");
            try r.put(d, "signature", .{ .string = "" }); // local reasoning is plaintext: no provider signature
            const o = try r.obj("content_block_delta");
            try r.put(o, "index", json.intValue(r.a, r.index) catch return error.Closed);
            try r.put(o, "delta", .{ .object = d });
            try r.send("content_block_delta", o);
        }
        const o = try r.obj("content_block_stop");
        try r.put(o, "index", json.intValue(r.a, r.index) catch return error.Closed);
        try r.send("content_block_stop", o);
        r.kind = null;
    }

    /// One block, opened and closed with no deltas between: a search result arrives whole.
    fn wholeBlock(r: *Reply, block: Value) E!void {
        if (block != .object) return;
        try r.close();
        r.index += 1;
        const opened = try r.obj("content_block_start");
        try r.put(opened, "index", json.intValue(r.a, r.index) catch return error.Closed);
        try r.put(opened, "content_block", block);
        try r.send("content_block_start", opened);
        const closed = try r.obj("content_block_stop");
        try r.put(closed, "index", json.intValue(r.a, r.index) catch return error.Closed);
        try r.send("content_block_stop", closed);
    }

    fn text(r: *Reply, kind: []const u8, value: []const u8) E!void {        if (value.len == 0) return;
        if (std.mem.eql(u8, kind, "text")) r.prose.appendSlice(r.a, value) catch return error.Closed;
        if (r.active != null) {
            r.pending.append(r.a, .{ .kind = kind, .text = value }) catch return error.Closed;
            return;
        }
        const key = if (std.mem.eql(u8, kind, "thinking")) "thinking" else "text";
        if (r.kind == null or !std.mem.eql(u8, r.kind.?, kind)) {
            try r.close();
            r.index += 1;
            r.kind = kind;
            const block = try r.obj(kind);
            try r.put(block, key, .{ .string = "" });
            const o = try r.obj("content_block_start");
            try r.put(o, "index", json.intValue(r.a, r.index) catch return error.Closed);
            try r.put(o, "content_block", .{ .object = block });
            try r.send("content_block_start", o);
        }
        const d = try r.obj(std.mem.concat(r.a, u8, &.{ kind, "_delta" }) catch return error.Closed);
        try r.put(d, key, .{ .string = value });
        const o = try r.obj("content_block_delta");
        try r.put(o, "index", json.intValue(r.a, r.index) catch return error.Closed);
        try r.put(o, "delta", .{ .object = d });
        try r.send("content_block_delta", o);
    }

    fn chunk(r: *Reply, payload: ?Value) E!void {
        if (r.finished) return;
        const p = payload orelse {
            try r.drain(true);
            try r.close();
            // a round that called a search is not the end of the message: the loop answers it and runs
            // another round on this same reply, so closing out here would end the stream mid-reply
            if (r.search != null) return;
            const d = json.newObject(r.a) catch return error.Closed;
            try r.put(d, "stop_reason", .{ .string = r.finish });
            try r.put(d, "stop_sequence", r.stop_sequence);
            const o = try r.obj("message_delta");
            try r.put(o, "delta", .{ .object = d });
            try r.put(o, "usage", r.tokens);
            try r.send("message_delta", o);
            try r.send("message_stop", try r.obj("message_stop"));
            r.finished = true;
            return;
        };
        if (p.has("error") and std.mem.eql(u8, r.finish, "refusal")) return; // --loop-guard-stop: the refusal says it
        if (p.has("error")) {
            try r.close();
            const problem = p.get("error").?;
            const message = if (problem.get("message")) |m| (if (m == .string) m.string else "generation failed") else "generation failed";
            const kind = problem.get("type");
            const status: u16 = if (kind != null and kind.? == .string and std.mem.eql(u8, kind.?.string, "invalid_request_error")) 400 else 500;
            const e = errorValue(r.a, message, status) catch return error.Closed;
            try r.send("error", e.object);
            r.finished = true;
            return;
        }
        if (p.get("usage")) |u| if (u != .null) {
            r.tokens = usage(r.a, u) catch return error.Closed;
        };
        if (p.get("stop_sequence")) |s| if (s != .null) {
            r.stop_sequence = s;
        };
        if (p.get("choices")) |choices| if (choices == .array) for (choices.array) |choice| {
            if (choice.get("finish_reason")) |f| if (f.truthy()) {
                r.finish = finishOf(json.strOr(f));
            };
            if (r.stop_sequence != .null) r.finish = "stop_sequence";
            const delta = choice.get("delta") orelse Value.null;
            try r.text("thinking", json.strOr(delta.get("reasoning_content")));
            try r.text("text", json.strOr(delta.get("content")));
            if (delta.get("tool_calls")) |tc| if (tc == .array) for (tc.array) |call| {
                const at: i64 = if (call.get("index")) |i| i.int64() orelse 0 else 0;
                const got = r.calls.getOrPut(r.a, at) catch return error.Closed;
                if (!got.found_existing) got.value_ptr.* = .{};
                const target = got.value_ptr;
                if (call.get("id")) |id| if (id.truthy() and id == .string) {
                    target.id = id.string;
                };
                const function = call.get("function") orelse Value.null;
                target.name = std.mem.concat(r.a, u8, &.{ target.name, json.strOr(function.get("name")) }) catch return error.Closed;
                target.arguments = std.mem.concat(r.a, u8, &.{ target.arguments, json.strOr(function.get("arguments")) }) catch return error.Closed;
            };
        };
        try r.drain(false);
    }

    fn drain(r: *Reply, final: bool) E!void {
        for (r.calls.keys(), r.calls.values()) |at, *call| {
            if (call.done) continue;
            if (call.id.len == 0 or call.name.len == 0) return;
            const server = serverTool(r.declared, call.name);
            if (r.active == null) {
                try r.close();
                r.index += 1;
                r.kind = if (server) "server_tool_use" else "tool_use";
                r.active = at;
                const block = try r.obj(r.kind.?);
                try r.put(block, "id", .{ .string = call.id });
                try r.put(block, "name", .{ .string = call.name });
                try r.put(block, "input", .{ .object = json.newObject(r.a) catch return error.Closed });
                const o = try r.obj("content_block_start");
                try r.put(o, "index", json.intValue(r.a, r.index) catch return error.Closed);
                try r.put(o, "content_block", .{ .object = block });
                try r.send("content_block_start", o);
            }
            if (at != r.active.?) return;
            const delta = call.arguments[call.sent..];
            if (delta.len > 0) {
                const d = try r.obj("input_json_delta");
                try r.put(d, "partial_json", .{ .string = delta });
                const o = try r.obj("content_block_delta");
                try r.put(o, "index", json.intValue(r.a, r.index) catch return error.Closed);
                try r.put(o, "delta", .{ .object = d });
                try r.send("content_block_delta", o);
                call.sent = call.arguments.len;
            }
            var complete = false;
            if (std.mem.endsWith(u8, std.mem.trimEnd(u8, call.arguments, " \t\r\n"), "}")) {
                if (json.parseText(r.a, call.arguments) catch null) |res| complete = res == .ok and res.ok == .object;
            }
            if (!complete and !final) return;
            try r.close();
            call.done = true;
            r.active = null;
            // a search the model asked for: the round loop answers it and continues this same message
            if (server) {
                const raw = call.arguments;
                const input = if (raw.len == 0) Value{ .object = json.newObject(r.a) catch return error.Closed } else switch (json.parseText(r.a, raw) catch return error.Closed) {
                    .ok => |v| v,
                    .err => Value{ .object = json.newObject(r.a) catch return error.Closed },
                };
                r.search = input;
            }
            const pending = r.pending.items;
            r.pending = .empty;
            for (pending) |p| try r.text(p.kind, p.text);
        }
    }

    /// A whole chat completion as one Messages reply.
    fn completion(r: *Reply, data: Value) !Value {
        const a = r.a;
        const choice = data.get("choices").?.array[0];
        const message = choice.get("message").?;
        var content: std.ArrayList(Value) = .empty;
        if (message.get("reasoning_content")) |t| if (t.truthy()) {
            const b = try json.newObject(a);
            try b.put(a, "type", .{ .string = "thinking" });
            try b.put(a, "thinking", t);
            try b.put(a, "signature", .{ .string = "" });
            try content.append(a, .{ .object = b });
        };
        if (message.get("content")) |t| if (t.truthy()) {
            const b = try json.newObject(a);
            try b.put(a, "type", .{ .string = "text" });
            try b.put(a, "text", t);
            try content.append(a, .{ .object = b });
        };
        if (message.get("tool_calls")) |calls| if (calls == .array) for (calls.array) |call| {
            const function = call.get("function").?;
            const raw = function.get("arguments").?;
            const args = if (raw == .string) switch (try json.parseText(a, raw.string)) {
                .ok => |v| v,
                .err => |m| {
                    r.problem = m;
                    return error.InvalidResponse;
                },
            } else raw;
            if (args != .object) {
                r.problem = "model tool arguments are not a JSON object";
                return error.InvalidResponse;
            }
            const b = try json.newObject(a);
            try b.put(a, "type", .{ .string = "tool_use" });
            try b.put(a, "id", call.get("id").?);
            try b.put(a, "name", function.get("name").?);
            try b.put(a, "input", args);
            try content.append(a, .{ .object = b });
        };
        const out = try json.copyObject(a, r.base);
        try out.put(a, "content", .{ .array = content.items });
        try out.put(a, "usage", try usage(a, data.get("usage") orelse .null));
        const stop = data.get("stop_sequence") orelse Value.null;
        try out.put(a, "stop_sequence", stop);
        try out.put(a, "stop_reason", .{ .string = if (stop != .null) "stop_sequence" else finishOf(json.strOr(choice.get("finish_reason"))) });
        return .{ .object = out };
    }
};

fn finishOf(reason: []const u8) []const u8 {
    if (std.mem.eql(u8, reason, "tool_calls")) return "tool_use";
    if (std.mem.eql(u8, reason, "length")) return "max_tokens";
    if (std.mem.eql(u8, reason, @import("loop_stop.zig").finish_reason)) return "refusal";
    return "end_turn";
}

/// A chat usage as Anthropic's: cached prompt tokens read from the cache, thinking tokens as output detail.
fn usage(a: Allocator, value: Value) Allocator.Error!Value {
    const prompt_tokens = (if (value.get("prompt_tokens")) |p| p.int64() else null) orelse 0;
    const details = value.get("prompt_tokens_details");
    const cached_raw = if (details != null and details.?.truthy()) (if (details.?.get("cached_tokens")) |c| c.int64() orelse 0 else 0) else 0;
    const cached = @min(prompt_tokens, @max(0, cached_raw));
    const o = try json.newObject(a);
    try o.put(a, "input_tokens", try json.intValue(a, prompt_tokens - cached));
    try o.put(a, "output_tokens", value.get("completion_tokens") orelse Value{ .int = "0" });
    try o.put(a, "cache_creation_input_tokens", .{ .int = "0" });
    try o.put(a, "cache_read_input_tokens", try json.intValue(a, cached));
    const out_details = value.get("completion_tokens_details");
    if (out_details != null and out_details.? == .object and out_details.?.has("reasoning_tokens")) {
        const t = try json.newObject(a);
        try t.put(a, "thinking_tokens", out_details.?.get("reasoning_tokens").?);
        try o.put(a, "output_tokens_details", .{ .object = t });
    }
    return .{ .object = o };
}
