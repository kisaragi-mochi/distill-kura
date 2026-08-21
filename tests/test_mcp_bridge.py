"""The MCP bridge: protocol shape, mode switching, and read-only actually refusing.

The read-only test is here because of a live failure: the bridge removed `kura_remember`
from `tools/list` but still executed it when called by name, and wrote a memory. Hiding
a tool is not enforcement — a host that cached an older listing, or a model that simply
guesses the name, reaches `tools/call` anyway.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeKura(BaseHTTPRequestHandler):
    """Stands in for the HTTP service; records what the bridge asked for."""
    calls: list[str] = []

    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        FakeKura.calls.append("GET " + self.path)
        if self.path.startswith("/prefill"):
            return self._json({"text": "<<<KURA-MAP store=maker>>>\n- [A](a.md) — t\n"
                                       "<<<END KURA-MAP>>>\n", "etag": "e1"})
        if self.path.startswith("/stores"):
            return self._json({"default": "maker",
                               "stores": {"maker": {"label": "m", "memories": 1,
                                                    "write_policy": "direct-allowed"},
                                          "eq": {"label": "e", "memories": 2,
                                                 "write_policy": "distiller-only"}},
                               "modes": {"talking": "eq"}})
        self._json({"text": "a memory"})

    def do_POST(self):
        FakeKura.calls.append("POST " + self.path)
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        self._json({"store": "maker", "how": "meaning", "picked": ["p"], "walked": ["p"],
                    "elapsed_s": 0.1, "context": "recalled text"})


def start():
    FakeKura.calls = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeKura)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def speak(url: str, messages: list[dict], env: dict | None = None) -> list[dict]:
    """Run the bridge as a real subprocess and talk JSON-RPC to it over stdio."""
    e = {**os.environ, "KURA_URL": url, "PYTHONPATH": ROOT, **(env or {})}
    p = subprocess.run([sys.executable, "-m", "distill_kura.mcp"],
                       input="\n".join(json.dumps(m) for m in messages) + "\n",
                       capture_output=True, text=True, env=e, timeout=60)
    return [json.loads(l) for l in p.stdout.splitlines() if l.strip()]


INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}


def call(name, args, id_=2):
    return {"jsonrpc": "2.0", "id": id_, "method": "tools/call",
            "params": {"name": name, "arguments": args}}


def test_notifications_get_no_reply():
    """Answering a notification breaks the connection with most hosts."""
    srv, url = start()
    try:
        out = speak(url, [INIT, {"jsonrpc": "2.0", "method": "notifications/initialized"},
                          {"jsonrpc": "2.0", "id": 2, "method": "ping"}])
        assert [m["id"] for m in out] == [1, 2]
    finally:
        srv.shutdown()


def test_readonly_hides_and_refuses_the_write_tool():
    srv, url = start()
    try:
        out = speak(url, [INIT, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                          call("kura_remember", {"slug": "x", "description": "d", "body": "b"}, 3)])
        names = [t["name"] for t in out[1]["result"]["tools"]]
        assert "kura_remember" not in names                      # hidden…
        assert out[2]["result"]["isError"] is True               # …and refused
        assert "read-only" in out[2]["result"]["content"][0]["text"]
        assert "[refused]" in out[2]["result"]["content"][0]["text"]   # not "cannot reach"
        assert not [c for c in FakeKura.calls if "remember" in c]  # never touched the store
    finally:
        srv.shutdown()


def test_writable_bridge_offers_and_performs_the_write():
    srv, url = start()
    try:
        out = speak(url, [INIT, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                          call("kura_remember", {"slug": "x", "description": "d", "body": "b"}, 3)],
                    env={"KURA_READONLY": "0"})
        assert "kura_remember" in [t["name"] for t in out[1]["result"]["tools"]]
        assert out[2]["result"]["isError"] is False
        assert any("remember" in c for c in FakeKura.calls)
    finally:
        srv.shutdown()


def test_bound_bridge_ignores_a_store_argument():
    """A preset-bound bridge is the mode switch. An argument must not escape the binding."""
    srv, url = start()
    try:
        out = speak(url, [INIT, call("kura_recall", {"question": "q", "store": "eq"})],
                    env={"KURA_STORE": "maker"})
        assert out[1]["result"]["isError"] is False
        assert any("store=maker" in c for c in FakeKura.calls)
        assert not any("store=eq" in c for c in FakeKura.calls)
    finally:
        srv.shutdown()


def test_a_bound_bridge_does_not_advertise_the_other_kura():
    """Least disclosure: the other stores' names, labels and counts are not this agent's
    business, and `store` is dead weight in a schema the model reads every turn. Not a
    security boundary — that is process separation (docs/TRUST.md)."""
    srv, url = start()
    try:
        out = speak(url, [INIT, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}],
                    env={"KURA_STORE": "maker"})
        names = [t["name"] for t in out[1]["result"]["tools"]]
        assert "kura_list" not in names and "kura_use" not in names
        for t in out[1]["result"]["tools"]:
            assert "store" not in t["inputSchema"].get("properties", {}), t["name"]
    finally:
        srv.shutdown()


def test_a_free_bridge_keeps_the_listing_and_the_store_argument():
    srv, url = start()
    try:
        out = speak(url, [INIT, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
        names = [t["name"] for t in out[1]["result"]["tools"]]
        assert "kura_list" in names and "kura_use" in names
    finally:
        srv.shutdown()


def test_bound_bridge_refuses_to_switch():
    """The tool is not listed, but a host that cached an older listing can still call
    it by name — so the call itself has to refuse too."""
    srv, url = start()
    try:
        out = speak(url, [INIT, call("kura_use", {"store": "eq"})], env={"KURA_STORE": "maker"})
        text = out[1]["result"]["content"][0]["text"]
        assert "cannot switch" in text or "unknown tool" in text or "refused" in text
    finally:
        srv.shutdown()


def test_the_listing_shows_the_store_policy_not_the_client_switch():
    """A client that hides its write tool has not made the store read-only. Showing the
    client's own switch there would say a store is protected when only this agent is."""
    srv, url = start()
    try:
        out = speak(url, [INIT, call("kura_list", {})], env={"KURA_READONLY": "0"})
        text = out[1]["result"]["content"][0]["text"]
        assert "[distiller-only]" in text          # eq, from the server
        assert "[direct-allowed]" not in text      # the default is not worth the noise
    finally:
        srv.shutdown()


def test_free_bridge_switches_for_the_session():
    srv, url = start()
    try:
        out = speak(url, [INIT, call("kura_use", {"store": "talking"}, 2),
                          call("kura_recall", {"question": "q"}, 3)])
        assert "Now reading from 'talking'" in out[1]["result"]["content"][0]["text"]
        assert any("/recall?store=talking" in c for c in FakeKura.calls)
    finally:
        srv.shutdown()


def test_free_bridge_rejects_an_unknown_store():
    srv, url = start()
    try:
        out = speak(url, [INIT, call("kura_use", {"store": "nope"})])
        assert "No kura called" in out[1]["result"]["content"][0]["text"]
    finally:
        srv.shutdown()


def test_recall_answer_shows_which_kura_and_how():
    srv, url = start()
    try:
        out = speak(url, [INIT, call("kura_recall", {"question": "q"})])
        text = out[1]["result"]["content"][0]["text"]
        assert text.startswith("[kura: maker]")
        assert "meaning" in text and "recalled text" in text
    finally:
        srv.shutdown()


def test_an_unreachable_kura_is_an_error_not_a_crash():
    out = speak("http://127.0.0.1:1", [INIT, call("kura_recall", {"question": "q"})])
    assert out[1]["result"]["isError"] is True
    assert "cannot reach" in out[1]["result"]["content"][0]["text"]


# ── the resident map over MCP ───────────────────────────────────────────────

def test_initialize_carries_instructions_that_fit_a_2kb_cap():
    """`instructions` is the only standing-text channel MCP has, and it is a MAY: Claude
    Code injects it and truncates at 2KB, VS Code and Goose inject it, Claude Desktop,
    claude.ai and DSH's own mcp-client ignore it. So it must be short, and it must not
    try to carry the index."""
    srv, url = start()
    try:
        out = speak(url, [INIT])
        text = out[0]["result"]["instructions"]
        assert len(text.encode()) < 2048
        assert "not remembered YET" in text
        assert "kura_map" in text
    finally:
        srv.shutdown()


def test_kura_map_serves_the_whole_index():
    srv, url = start()
    try:
        out = speak(url, [INIT, call("kura_map", {})])
        assert out[1]["result"]["isError"] is False
        assert "KURA-MAP" in out[1]["result"]["content"][0]["text"]
        assert any(c.startswith("GET /prefill") for c in FakeKura.calls)
    finally:
        srv.shutdown()


def test_a_whitespace_store_name_is_refused_rather_than_unbinding():
    """`KURA_STORE=" "` collapsed to free mode, and a preset that meant to bind returned
    another kura's confidential memory."""
    srv, url = start()
    try:
        e = {**os.environ, "KURA_URL": url, "PYTHONPATH": ROOT, "KURA_STORE": " "}
        p = subprocess.run([sys.executable, "-m", "distill_kura.mcp"],
                           input=json.dumps(INIT) + "\n", capture_output=True, text=True,
                           env=e, timeout=60)
        assert p.returncode != 0
        assert "whitespace" in p.stderr
        assert p.stdout.strip() == ""      # it never served a single frame
    finally:
        srv.shutdown()
