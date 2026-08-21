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
        if self.path.startswith("/stores"):
            return self._json({"default": "maker",
                               "stores": {"maker": {"label": "m", "memories": 1},
                                          "eq": {"label": "e", "memories": 2}},
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


def test_bound_bridge_refuses_to_switch():
    srv, url = start()
    try:
        out = speak(url, [INIT, call("kura_use", {"store": "eq"})], env={"KURA_STORE": "maker"})
        assert "cannot switch" in out[1]["result"]["content"][0]["text"]
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
