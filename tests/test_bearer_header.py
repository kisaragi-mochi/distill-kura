"""The Bearer header: one rule, three callers.

`Endpoint.ask`, `Endpoint.alive` and `payforward._post` each built
`{"Authorization": "Bearer <key>"}` from `api_key_env` in their own copy of the
same block, and nothing in the suite ever inspected a request header — so a slip
in any of the three would have gone unnoticed. They share `bearer_headers` now,
and this pins what it must produce.

It also pins the one thing the old copies got wrong: `ask` wrote
`last_error = "<ENV> is not set"` BEFORE the request, where every exit path
overwrote it before returning. The note was unreachable. A missing key surfaces
as a 401, so the 401 message is where the variable has to be named.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.payforward import _post                    # noqa: E402
from distill_kura.thinker import Endpoint, bearer_headers    # noqa: E402


class Guarded(BaseHTTPRequestHandler):
    """Answers only with a key; 401s without one. Records the Authorization it saw."""
    seen: list = []

    def log_message(self, *a):
        pass

    def _reply(self):
        auth = self.headers.get("Authorization")
        Guarded.seen.append(auth)
        if auth is None:
            b = b'{"error":"missing key"}'
            self.send_response(401)
        else:
            b = json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    do_POST = _reply
    do_GET = _reply


def test_an_unset_api_key_env_is_named_in_the_401_it_causes(monkeypatch):
    Guarded.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Guarded)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        monkeypatch.delenv("KURA_TEST_KEY", raising=False)
        e = Endpoint(url=base, model="m", api_key_env="KURA_TEST_KEY", dialect="openai")
        assert e.ask("s", "u") is None
        assert "HTTP 401" in e.last_error and "KURA_TEST_KEY is not set" in e.last_error
        assert e.alive() is False
        assert Guarded.seen[-1] is None
        assert _post(base + "/x", {}, 5, "KURA_TEST_KEY")[0] == 401
        assert Guarded.seen[-1] is None

        monkeypatch.setenv("KURA_TEST_KEY", "k1")
        e2 = Endpoint(url=base, model="m", api_key_env="KURA_TEST_KEY", dialect="openai")
        assert e2.ask("s", "u") == "hello"
        assert e2.last_error == ""            # the note rides the failure, not success
        assert e2.alive() is True
        assert _post(base + "/x", {}, 5, "KURA_TEST_KEY")[0] == 200
        assert set(Guarded.seen[-3:]) == {"Bearer k1"}

        monkeypatch.setenv("KURA_TEST_KEY", "")   # empty counts as unset, everywhere
        assert bearer_headers("KURA_TEST_KEY") == {}
        assert bearer_headers(None) == {}
    finally:
        srv.shutdown()
