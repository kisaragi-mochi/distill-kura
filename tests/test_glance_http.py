"""Glance over HTTP and CLI: exact, honest, contained (plan §4.4).

The glance route carries the `read_exact` philosophy onto the wire: a slug the
caller recognised on the map gets its ~150-token mechanical confirmation, and a
misspelling — or an escape attempt — is a 404 with no neighbour's content behind
it. Written adversarially: each HTTP test is an attempt to make glance answer for
something other than the memory that was named. No model is needed anywhere.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.cli import main              # noqa: E402
from distill_kura.registry import Registry     # noqa: E402
from distill_kura.server import _make_handler  # noqa: E402
from distill_kura.store import Store           # noqa: E402
from distill_kura.thinker import Models        # noqa: E402

SECRET = "TOP SECRET CONTENT"
OUTSIDE = "A FILE OUTSIDE EVERY STORE"


def glance_store(tmp_path) -> Store:
    """One store holding the memory the plan's examples glance at."""
    s = Store(name="m", path=str(tmp_path / "m"), label="k")
    s.init_files()
    s.remember_direct("freetoken-hybrid",
                      "FreeToken CPU hybrid — the all-hands cooperative inference push",
                      "The hybrid runs the experts on CPU.\n\nRelated: [[exl3-quantization]] "
                      "and [[../outside/private]] and [[a-memory-that-does-not-exist]].",
                      title="FreeToken CPU hybrid")
    return s


def two_stores(tmp_path):
    """A public store and a private one, side by side, plus a file outside both."""
    pub = Store(name="public", path=str(tmp_path / "public"), label="public")
    prv = Store(name="private", path=str(tmp_path / "private"), label="private")
    pub.init_files()
    prv.init_files()
    pub.remember("harmless", "a public note", "nothing sensitive")
    prv.remember("secret", "the private one", SECRET)
    (tmp_path / "outside.md").write_text(OUTSIDE, encoding="utf-8")
    return pub, prv


def serve(reg):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(reg))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# ── over HTTP ───────────────────────────────────────────────────────────────

def test_http_glance_serves_an_exact_slug(tmp_path):
    st = glance_store(tmp_path)
    reg = Registry(stores={"m": st}, modes={}, models=Models.from_config({}), default="m")
    srv, base = serve(reg)
    try:
        body = json.load(urllib.request.urlopen(f"{base}/glance/freetoken-hybrid"))
        assert body["ok"] is True
        assert body["slug"] == "freetoken-hybrid"
        assert body["title"] == "FreeToken CPU hybrid"
        assert "all-hands" in body["trigger"]
    finally:
        srv.shutdown()


def test_http_glance_refuses_a_misspelling(tmp_path):
    """A misspelling must not resolve to a neighbour: the answer is a 404 whose
    error names the name given, with no resolved slug to grab."""
    st = glance_store(tmp_path)
    reg = Registry(stores={"m": st}, modes={}, models=Models.from_config({}), default="m")
    srv, base = serve(reg)
    try:
        try:
            urllib.request.urlopen(f"{base}/glance/freetoken-hyrbid")
            assert False, "a misspelling must not answer for a neighbour"
        except urllib.error.HTTPError as e:
            assert e.code == 404
            body = json.load(e)
            assert body["ok"] is False
            assert "freetoken-hyrbid" in body["error"]
            assert "slug" not in body
    finally:
        srv.shutdown()


ESCAPES = [
    "../private/secret",
    "..%2Fprivate%2Fsecret",
    "....//private/secret",
    "../../etc/hostname",
    "public/../private/secret",
    "..\\private\\secret",
]


def test_http_glance_never_returns_another_store(tmp_path):
    pub, prv = two_stores(tmp_path)
    reg = Registry(stores={"public": pub, "private": prv}, modes={},
                   models=Models.from_config({}), default="public")
    srv, base = serve(reg)
    try:
        for probe in ESCAPES:
            url = f"{base}/glance/{urllib.parse.quote(probe, safe='')}?store=public"
            try:
                body = json.load(urllib.request.urlopen(url))
                assert SECRET not in (body.get("text") or ""), probe
                assert OUTSIDE not in (body.get("text") or ""), probe
            except urllib.error.HTTPError as e:
                assert e.code == 404, (probe, e.code)
                body = json.load(e)
                assert SECRET not in (body.get("error") or ""), probe
                assert OUTSIDE not in (body.get("error") or ""), probe
    finally:
        srv.shutdown()


def test_http_glance_store_selector_finds_or_404s(tmp_path):
    pub, prv = two_stores(tmp_path)
    reg = Registry(stores={"public": pub, "private": prv}, modes={},
                   models=Models.from_config({}), default="public")
    srv, base = serve(reg)
    try:
        # ?store= switches the store a glance answers from: private holds 'secret'.
        body = json.load(urllib.request.urlopen(f"{base}/glance/secret?store=private"))
        assert body["ok"] is True and body["slug"] == "secret"
        # The same selector, asked of a store that does not hold the slug: the 404
        # names the store the name is unknown in — never a guess at the default's.
        try:
            urllib.request.urlopen(f"{base}/glance/secret?store=public")
            assert False, "public does not hold 'secret'"
        except urllib.error.HTTPError as e:
            assert e.code == 404
            body = json.load(e)
            assert body["ok"] is False and "public" in body["error"]
        # And an unknown selector name is its own 404, before any glance happens.
        try:
            urllib.request.urlopen(f"{base}/glance/harmless?store=nope")
            assert False, "an unknown store must not serve from the default"
        except urllib.error.HTTPError as e:
            assert e.code == 404
            assert "unknown store" in json.load(e)["error"]
    finally:
        srv.shutdown()


# ── over the CLI ────────────────────────────────────────────────────────────

def test_cli_glance_prints_text_or_exits_one(tmp_path, capsys):
    st = glance_store(tmp_path)
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[stores.m]
path = "{tmp_path / 'm'}"
[models.thinker]
url = "http://127.0.0.1:9/v1"
model = "none"
""", encoding="utf-8")
    assert main(["-c", str(cfg), "-s", "m", "glance", "freetoken-hybrid"]) == 0
    assert "[freetoken-hybrid]" in capsys.readouterr().out
    # A misspelling is a refusal: exit 1, and the error is JSON in both modes.
    assert main(["-c", str(cfg), "-s", "m", "glance", "freetoken-hyrbid"]) == 1
    out = capsys.readouterr().out
    assert '"ok": false' in out and "freetoken-hyrbid" in out
    assert main(["-c", str(cfg), "-s", "m", "glance", "--json", "freetoken-hyrbid"]) == 1
    assert '"ok": false' in capsys.readouterr().out
