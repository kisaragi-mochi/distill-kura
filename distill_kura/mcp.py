"""kura-mcp — the bridge that turns the kura's HTTP mouth into MCP (stdio JSON-RPC).

    model ── MCP host (DSH / Claude Code / anything) ── stdio ── this bridge ── HTTP ── kura

Stdlib only, one file, so it can be dropped next to any host that speaks MCP.

Mode switching, two ways:

  · **bound** — set `KURA_STORE=eq` on the bridge process. Every call goes to that
    kura. This is the DSH pattern: each agent preset mounts its own bridge, so
    switching preset switches memory (and the persona, which DSH owns).
  · **free** — leave `KURA_STORE` empty and the tools take an optional `store`
    argument, plus `kura_use` to change the session default at runtime. Use this
    when one agent needs to move between kura mid-conversation.

Conventions that are not decoration:
  · answer only initialize / tools/list / tools/call / ping; never reply to a
    notification (a reply to a notification breaks the connection)
  · failures come back as `isError: true`, never as an exception — a dropped
    connection makes the tool vanish until the host restarts it
  · descriptions say WHEN to call and WHAT AN EMPTY RESULT MEANS; that text is
    the only thing the model knows about the tool
  · recall answers lead with elapsed/how/picked/walked. `how=words` means the
    thinker was unreachable and quality silently degraded — never hide that
  · read-only by default: writing belongs to the distiller's gate, not to a model
    with a spare tool call

The resident map, over MCP:
  MCP has exactly one way for a server to put standing text in front of the model — the
  `instructions` field of the initialize result — and the spec says a client MAY use it.
  Measured support: Claude Code injects it (and TRUNCATES it at 2KB), VS Code/Copilot and
  Goose inject it; Claude Desktop, claude.ai and DSH's own mcp-client ignore it entirely.
  A 9,000-token index therefore CANNOT travel this way.
  So `instructions` carries a short pointer, and the map itself is offered as the
  `kura_map` tool, which any host can call. For a host that can inject properly, use the
  native DSH plugin or paste `kura prefill` into the system prompt.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request

KURA = os.environ.get("KURA_URL", "http://127.0.0.1:8085").rstrip("/")
_RAW_STORE = os.environ.get("KURA_STORE")
STORE = (_RAW_STORE or "").strip()                      # "" = free mode
if _RAW_STORE is not None and _RAW_STORE != "" and not STORE:
    # A preset that meant to bind, bound to nothing: whitespace collapsed to free mode
    # and the other kura's memories came back through it.
    sys.exit("KURA_STORE is set to whitespace. Unset it for free mode, or name a store.")
LABEL = os.environ.get("KURA_LABEL", "the kura")
READONLY = os.environ.get("KURA_READONLY", "1") not in ("", "0", "false", "no")
WRITE_LOG = os.environ.get("KURA_WRITE_LOG", "")
PROTOCOL_VERSION = "2025-06-18"

# Kept deliberately short: Claude Code truncates server instructions at 2KB, and a
# truncated instruction is a sentence that stops mid-thought in the model's context.
INSTRUCTIONS = """{label} is a long-term memory (a "kura") for this household or project.

It is not a search index of documents: it holds distilled facts — decisions that were
made, measurements that were taken, landmines that will recur — one fact per memory,
linked to each other.

· Call kura_recall whenever the question touches what was decided, measured, or done
  here before. Recall works by MEANING, so ask it in plain language; the words need not
  match anything.
· An empty result means it is not remembered YET. Say so plainly. Never fill the gap
  with a plausible invention — a confident guess about this household is the one failure
  mode this memory exists to prevent.
· Call kura_map to see the whole index at once when you need to know WHAT EXISTS rather
  than look one thing up.
"""

_session_store = STORE       # `kura_use` moves this in free mode


def http(method: str, path: str, body: dict | None = None, timeout: float = 240.0):
    req = urllib.request.Request(
        KURA + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _store_arg(args: dict) -> str:
    if STORE:
        return STORE                       # bound mode: the argument cannot override
    return (args.get("store") or _session_store or "").strip()


def _q(store: str) -> str:
    return f"?store={urllib.parse.quote(store)}" if store else ""



def _bound_note() -> str:
    return (f" This bridge is bound to the '{STORE}' kura." if STORE else
            " Pass `store` to reach a different kura, or call kura_use to switch for the session.")


TOOLS: list[dict] = [
    {
        "name": "kura_recall",
        "description": (
            f"Recall from {LABEL} — long-term memory retrieved by MEANING rather than keyword, "
            "then following [[links]] between memories. Call it whenever the question touches "
            "past decisions, measurements, people, machines, or anything done before — prefer "
            "it over guessing. An empty result means it is simply not remembered yet: say so "
            "plainly and never fill the gap with invention." + _bound_note()),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string",
                             "description": "What you want to remember, as a natural question"},
                "hops": {"type": "integer",
                         "description": "How many [[link]] hops to walk (default 1)"},
                "store": {"type": "string",
                          "description": "Which kura to ask (store or mode name). Omit for the current one."},
            },
            "required": ["question"],
        },
    },
    {
        "name": "kura_read",
        "description": (
            f"Read one whole memory from {LABEL} by its slug (e.g. 'storage-doctrine'). "
            "Use after kura_recall when a summary is not enough and you need the full text."),
        "inputSchema": {
            "type": "object",
            "properties": {"slug": {"type": "string", "description": "Memory slug, without .md"},
                           "store": {"type": "string", "description": "Which kura. Omit for the current one."}},
            "required": ["slug"],
        },
    },
    {
        "name": "kura_map",
        "description": (
            f"Show the whole index of {LABEL} — every memory's one-line recognition "
            "trigger, in one answer. Use it when you need to see WHAT EXISTS rather than "
            "look something up: before claiming a topic was never discussed, when "
            "choosing which memory to open, or right after switching kura. It is a map, "
            "not the contents: open a memory with kura_read for the detail."),
        "inputSchema": {"type": "object",
                        "properties": {"store": {"type": "string",
                                                 "description": "Which kura. Omit for the current one."}}},
    },
    {
        "name": "kura_doctor",
        "description": (
            f"Health check of {LABEL}: how many memories, resolved and dead [[links]], islands "
            "(memories nothing links to), index drift. Call when recall behaves oddly, or when "
            "asked about the memory system itself."),
        "inputSchema": {"type": "object",
                        "properties": {"store": {"type": "string",
                                                 "description": "Which kura. Omit for the current one."}}},
    },
    {
        "name": "kura_list",
        "description": (
            "List the kura this server holds and which agent mode each one belongs to. Call it "
            "when you are unsure which memory you are speaking from, or before switching."),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "kura_use",
        "description": (
            "Switch which kura the following calls read from, for the rest of this session. "
            "Use when the conversation moves to a different mode of work (for example from "
            "building things to talking things through). Has no effect on a bridge that was "
            "bound to a single kura at startup — it will say so."),
        "inputSchema": {
            "type": "object",
            "properties": {"store": {"type": "string", "description": "Store or mode name"}},
            "required": ["store"],
        },
    },
    {
        "name": "kura_remember",
        "description": (
            f"Write ONE fact into {LABEL}. This bypasses the distiller's evidence gate, so use "
            "it only when the human explicitly asks for something to be remembered. One fact "
            "per call, and write the date into the text."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "slug": {"type": "string", "description": "kebab-case id"},
                "description": {"type": "string", "description": "One line for the index — a recognition trigger, not a summary"},
                "body": {"type": "string", "description": "The fact itself"},
                "store": {"type": "string", "description": "Which kura. Omit for the current one."},
            },
            "required": ["slug", "description", "body"],
        },
    },
]
if STORE:
    # Bound to one kura: least disclosure. The other stores' names, labels and counts
    # are not this agent's business, and `store`/`kura_use` are dead weight in a schema
    # the model reads every turn. (Not a security boundary — that is process separation.
    # See docs/TRUST.md.)
    TOOLS = [t for t in TOOLS if t["name"] not in ("kura_use", "kura_list")]
    for _t in TOOLS:
        _t["inputSchema"].get("properties", {}).pop("store", None)

if READONLY:
    # Read-only removes the tool from the listing AND refuses the call (see call_tool).
    # Hiding a tool is not enforcement: a host that remembers an earlier listing, or a
    # model that guesses the name, still reaches `tools/call`. Found by doing exactly
    # that — the hidden tool wrote a memory.
    TOOLS = [t for t in TOOLS if t["name"] != "kura_remember"]


def _header(store: str) -> str:
    """Whoever calls must be able to see WHICH memory answered."""
    return f"[kura: {store or 'default'}]"


def call_tool(name: str, args: dict) -> str:
    global _session_store
    if READONLY and name == "kura_remember":
        raise PermissionError(
            "this bridge is read-only: memories enter through the distiller's evidence "
            "gate, not by tool call (set KURA_READONLY=0 to allow writing)")
    if name not in {t["name"] for t in TOOLS}:
        raise ValueError(f"unknown tool: {name}")
    store = _store_arg(args)

    if name == "kura_list":
        if STORE:
            return f"This agent is bound to the '{STORE}' kura."
        d = http("GET", "/stores")
        cur = store or d.get("default")
        lines = [f"current: {cur}"]
        for n, s in (d.get("stores") or {}).items():
            modes = [m for m, t in (d.get("modes") or {}).items() if t == n]
            lines.append(f"  {'*' if n == cur else '·'} {n}: {s.get('label')} — "
                         f"{s.get('memories')} memories"
                         + (f", modes={modes}" if modes else "")
                         # The store's own policy, not this bridge's: a client that
                         # hides its write tool has not made the store read-only.
                         + (f" [{s.get('write_policy')}]"
                            if s.get("write_policy") not in (None, "direct-allowed") else ""))
        return "\n".join(lines)

    if name == "kura_use":
        want = (args.get("store") or "").strip()
        if STORE:
            return (f"This bridge is bound to '{STORE}' and cannot switch. "
                    f"Switching kura here means switching agent mode/preset.")
        d = http("GET", "/stores")
        if want not in (d.get("stores") or {}) and want not in (d.get("modes") or {}):
            return f"No kura called {want!r}. Known: {sorted(d.get('stores') or {})}"
        _session_store = want
        return f"Now reading from '{want}'. (Only memory switched — persona and tools are the host's.)"

    if name == "kura_recall":
        t0 = time.time()
        d = http("POST", "/recall" + _q(store),
                 {"question": args.get("question", ""), "hops": int(args.get("hops", 1))})
        head = (f"{_header(d.get('store') or store)} "
                f"[{d.get('elapsed_s', round(time.time() - t0, 1))}s / {d.get('how', '?')}] "
                f"picked: {d.get('picked', '?')}\nwalked: {d.get('walked', [])}\n\n")
        return head + (d.get("context") or "(nothing recalled — not remembered yet)")

    if name == "kura_read":
        slug = urllib.parse.quote((args.get("slug") or "").strip())
        d = http("GET", f"/memory/{slug}" + _q(store))
        return d.get("text") or f"(no memory called {args.get('slug')!r} in {store or 'the default kura'})"

    if name == "kura_map":
        d = http("GET", "/prefill" + _q(store))
        return d.get("text") or "(the index could not be read — the map is missing, not empty)"

    if name == "kura_doctor":
        return json.dumps(http("GET", "/doctor" + _q(store)), ensure_ascii=False, indent=1)

    if name == "kura_remember":
        d = http("POST", "/remember" + _q(store),
                 {"slug": args.get("slug"), "description": args.get("description"),
                  "body": args.get("body")})
        if WRITE_LOG:
            try:
                os.makedirs(os.path.dirname(WRITE_LOG), exist_ok=True)
                with open(WRITE_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                        "store": store, "args": args, "result": d},
                                       ensure_ascii=False) + "\n")
            except OSError:
                pass
        return json.dumps(d, ensure_ascii=False)

    raise ValueError(f"unknown tool: {name}")   # unreachable: guarded above


def reply(mid, result=None, error=None) -> None:
    msg = {"jsonrpc": "2.0", "id": mid}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except ValueError:
            continue
        mid, method = m.get("id"), m.get("method", "")
        params = m.get("params") or {}
        if mid is None:
            continue                                   # never answer a notification
        if method == "initialize":
            reply(mid, {"protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "kura", "version": "0.1.0"},
                        "instructions": INSTRUCTIONS.format(label=LABEL)})
        elif method == "ping":
            reply(mid, {})
        elif method == "tools/list":
            reply(mid, {"tools": TOOLS})
        elif method == "tools/call":
            try:
                text = call_tool(params.get("name", ""), params.get("arguments") or {})
                reply(mid, {"content": [{"type": "text", "text": text}], "isError": False})
            except (PermissionError, ValueError) as e:
                # A refusal is not an outage. Saying "cannot reach the kura" here would
                # send the model looking for a broken server instead of reading the reason.
                reply(mid, {"content": [{"type": "text", "text": f"[refused] {e}"}],
                            "isError": True})
            except Exception as e:                     # keep the connection alive
                reply(mid, {"content": [{"type": "text",
                                         "text": f"[cannot reach {LABEL}] {type(e).__name__}: {e}"}],
                            "isError": True})
        else:
            reply(mid, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
