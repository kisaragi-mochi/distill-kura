"""The /prefill wire contract, and what a POST with an unparseable body gets.

The map is the largest thing the server hands out and the one clients re-read every
couple of minutes, so its conditional GET (ETag → 304), its `format=text` form for a
shell hook, and its per-request `?window`/`?fraction` overrides are the parts a
refactor could break silently: no test drove them before this file. No model needed.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.registry import Registry     # noqa: E402
from distill_kura.server import _make_handler  # noqa: E402
from distill_kura.store import Store           # noqa: E402
from distill_kura.thinker import Models        # noqa: E402


def serve(tmp_path):
    s = Store(name="s", path=str(tmp_path / "s"), label="the kura")
    s.init_files()
    s.remember("cooling", "how the ssd stays cool", "the fans went in first")
    reg = Registry(stores={"s": s}, modes={}, models=Models.from_config({}), default="s")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(reg))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        r = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:      # 304 arrives here, not as a reply
        return e.code, e.headers, e.read()
    return r.status, r.headers, r.read()


def test_prefill_carries_a_quoted_etag_and_answers_304_to_a_match(tmp_path):
    srv, base = serve(tmp_path)
    try:
        code, h, body = get(f"{base}/prefill")
        assert code == 200
        d = json.loads(body)
        assert h["Content-Type"] == "application/json; charset=utf-8"
        assert h["ETag"] == '"' + d["etag"] + '"'

        code, h, body = get(f"{base}/prefill", {"If-None-Match": h["ETag"]})
        assert code == 304 and body == b""
        assert h["Content-Length"] == "0"
        assert h["ETag"] == '"' + d["etag"] + '"'

        # The guard is equality, not presence: a stale validator gets the map.
        code, _, body = get(f"{base}/prefill", {"If-None-Match": '"stale"'})
        assert code == 200 and json.loads(body)["etag"] == d["etag"]
    finally:
        srv.shutdown()


def test_prefill_format_text_is_the_block_and_nothing_else(tmp_path):
    srv, base = serve(tmp_path)
    try:
        _, _, body = get(f"{base}/prefill")
        d = json.loads(body)
        code, h, raw = get(f"{base}/prefill?format=text")
        assert code == 200
        assert h["Content-Type"] == "text/plain; charset=utf-8"
        assert h["ETag"] == '"' + d["etag"] + '"'
        assert raw.decode() == d["text"]
    finally:
        srv.shutdown()


def test_prefill_query_overrides_reach_the_build(tmp_path):
    srv, base = serve(tmp_path)
    try:
        code, _, body = get(f"{base}/prefill?window=2000&fraction=0.5")
        assert code == 200
        d = json.loads(body)
        assert d["window_tokens"] == 2000 and d["text"]
    finally:
        srv.shutdown()


def test_a_post_body_that_is_not_json_is_a_400_that_says_so(tmp_path):
    srv, base = serve(tmp_path)
    try:
        req = urllib.request.Request(f"{base}/recall", data=b"{",
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req)
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400
            assert json.loads(e.read()) == {"error": "bad json"}
    finally:
        srv.shutdown()


def test_profile_survives_a_charter_configured_at_a_path_that_is_not_there(tmp_path):
    """A `charter` pointing at a missing file is a config mistake; it used to take the
    whole request down with an OSError, which reads to a client as the server dying."""
    s = Store(name="s", path=str(tmp_path / "s"), label="the kura",
              charter=str(tmp_path / "nowhere" / "charter.md"))
    s.init_files()
    reg = Registry(stores={"s": s}, modes={}, models=Models.from_config({}), default="s")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(reg))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        code, _, body = get(f"http://127.0.0.1:{srv.server_address[1]}/profile")
        assert code == 200 and json.loads(body)["charter"] == ""
    finally:
        srv.shutdown()


def test_a_path_that_merely_starts_like_a_route_is_a_404_not_that_route(tmp_path):
    """Routes were matched with `path.startswith`, so `/healthz` answered as /health,
    `/indexes` as /index and `/storesX` as /stores. A client's typo — or a probe for a
    route this build does not have — got a 200 for a DIFFERENT endpoint, which is the
    one wrong answer a caller cannot detect."""
    srv, base = serve(tmp_path)
    try:
        for phantom in ("/healthz", "/indexes", "/storesX"):
            code, _, body = get(base + phantom)
            assert code == 404, f"{phantom} answered {code}"
            assert json.loads(body) == {"error": "not found", "path": phantom}
        for real in ("/health", "/index", "/stores", "/doctor", "/prefill", "/profile"):
            assert get(base + real)[0] == 200, real
        assert get(f"{base}/memory/cooling")[0] == 200
        assert get(f"{base}/glance/cooling")[0] == 200
    finally:
        srv.shutdown()


def test_a_post_path_that_merely_starts_like_a_route_is_a_404(tmp_path):
    srv, base = serve(tmp_path)
    try:
        req = urllib.request.Request(f"{base}/rememberX", data=b"{}",
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req)
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
            assert json.loads(e.read())["path"] == "/rememberX"
    finally:
        srv.shutdown()
