"""One full turn of the loop, against a fake model server.

A scripted model is not a quality test — it cannot tell you whether a real model picks
good memories. What it *does* prove is that the machinery holds when the model behaves
badly: a fabricated quote never becomes a memory, a draft that credits the human with
no evidence is refused, and the watermark moves so the same water is not drunk twice.

The fake server answers on a real socket, so the HTTP client, the endpoint dialects and
the server routes are all exercised for real.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.distill import Distiller                    # noqa: E402
from distill_kura.registry import Registry                    # noqa: E402
from distill_kura.store import Store                          # noqa: E402
from distill_kura.thinker import Models                       # noqa: E402

JOURNAL = [
    {"type": "user", "message": {"content": [
        {"type": "text", "text": "put the archive on the slow disk, the fast one is for scratch"}]}},
    {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "df -h /data"}}]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "content": [{"type": "text", "text": "/data 3.2T used 1.1T avail"}]}]}},
    {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "I think we should also mirror it, but that is only my hunch."}]}},
]

# What the fake model answers, keyed by a phrase in the system prompt.
# Keys must not span a line break — the prompts are hard-wrapped.
SCRIPT = {
    "deserves to become a permanent memory": json.dumps([
        {"topic": "archive-on-slow-disk", "kind": "project",
         "why": "a decision about where the archive lives",
         "quotes": ["[USER] put the archive on the slow disk"]},
        {"topic": "invented-thing", "kind": "project",
         "why": "the user asked for nightly snapshots",
         "quotes": ["[USER] please take nightly snapshots of everything"]},
    ]),
    "actually NEW": "NEW\nthe store has nothing about disks",
    "INDEX of everything remembered": "[]",
    "You write the final memory": (
        "SLUG: archive-on-slow-disk\n"
        "TITLE: Archive on the slow disk\n"
        "DESC: the archive lives on the slow disk; the fast one stays scratch\n"
        "BODY:\n"
        "The archive goes on the slow disk. The fast disk is scratch space.\n\n"
        "**Why:** putting it the other way round burns write endurance for nothing.\n"
        "**How to apply:** when choosing a target directory, check which disk it is on.\n"),
    "draw the last line": "POUR\nreason: inside its evidence and useful later",
}


class FakeLLM(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        system = body["messages"][0]["content"]
        answer = next((v for k, v in SCRIPT.items() if k in system), "[]")
        out = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def start_fake() -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def build(tmp_path):
    jdir = tmp_path / "journals"
    jdir.mkdir()
    with open(jdir / "session.jsonl", "w", encoding="utf-8") as f:
        for e in JOURNAL:
            f.write(json.dumps(e) + "\n")
        f.write(json.dumps({"type": "user", "message": {"content": [
            {"type": "text", "text": "padding " * 2000}]}}) + "\n")   # pass the min-drink bar

    store = Store(name="main", path=str(tmp_path / "kura"), label="test kura")
    store.init_files()
    srv, url = start_fake()
    models = Models.from_config({"thinker": {"url": url, "model": "fake"}})
    reg = Registry(stores={"main": store}, modes={}, models=models, default="main",
                   raw={"distill": {"journals": {"claude": str(jdir)}, "language": "English"}})
    return srv, reg, store, jdir


def test_one_pass_writes_a_draft_and_refuses_the_fabricated_one(tmp_path):
    srv, reg, store, _ = build(tmp_path)
    try:
        d = Distiller(reg, store)
        r = d.run(chunks=1)
        assert r["ok"]
        assert r["drafts"] == ["archive-on-slow-disk"]     # the grounded candidate
        assert r["dropped"] == 1                            # the invented one never got through
        drafts = os.listdir(d.drafts_dir)
        assert drafts == ["archive-on-slow-disk.md"]
        text = open(os.path.join(d.drafts_dir, drafts[0]), encoding="utf-8").read()
        assert "[USER] put the archive on the slow disk" in text   # evidence travels with it
        # the rejected candidate is recorded, not silently forgotten
        dropped = open(os.path.join(store.still, "dropped.jsonl"), encoding="utf-8").read()
        assert "quotes not found in the raw material" in dropped
    finally:
        srv.shutdown()


def test_drain_pours_the_draft_into_the_store(tmp_path):
    srv, reg, store, _ = build(tmp_path)
    try:
        d = Distiller(reg, store)
        d.run(chunks=1)
        out = d.drain()
        assert out["poured"] == 1 and out["left"] == 0
        assert "archive-on-slow-disk" in store.slugs()
        assert "- [Archive on the slow disk](archive-on-slow-disk.md)" in store.index_text()
        assert "slow disk" in store.read("archive-on-slow-disk")
    finally:
        srv.shutdown()


def test_the_same_water_is_not_drunk_twice(tmp_path):
    srv, reg, store, _ = build(tmp_path)
    try:
        d = Distiller(reg, store)
        d.run(chunks=1)
        again = d.run(chunks=1)
        assert again == {"ok": True, "why": "nothing worth drinking"}
    finally:
        srv.shutdown()


def test_a_draft_crediting_the_human_without_evidence_is_never_poured(tmp_path):
    srv, reg, store, _ = build(tmp_path)
    try:
        d = Distiller(reg, store)
        os.makedirs(d.drafts_dir, exist_ok=True)
        p = os.path.join(d.drafts_dir, "bad.md")
        open(p, "w", encoding="utf-8").write(
            "<!-- distilled\n     kind: project   evidence classes: SELF"
            "   🚫credits the human with no [USER] quote\n     evidence:\n-->\n"
            "TITLE: Bad\nDESC: something the human never said\n\nthe user decided to do it\n")
        r = d.pour("bad")
        assert not r["ok"]
        assert "bad" not in store.slugs()
        # and the drain agrees: it tosses it without even asking the model
        out = d.drain()
        assert out["quarantined"] == 1 and out["poured"] == 0
    finally:
        srv.shutdown()


def _compose_one(d, source, slug_line, body_line, quote):
    """One composed draft, with the scribe answering exactly what the test needs.
    compose() is the only place a slug is made, so a slug test has to go through it."""
    d._current_source = source
    d.scribe = lambda task, u, max_tokens=0: (                    # type: ignore[method-assign]
        f"SLUG: {slug_line}\nTITLE: Archive\n"
        f"DESC: the archive lives on the slow disk, the fast one stays scratch\n"
        f"BODY:\n{body_line}\n")
    return d.compose({"topic": "archive", "kind": "project", "why": "where the archive lives",
                      "evidence": [{"class": "USER", "text": quote}], "classes": ["USER"]})


def test_a_fix_with_no_body_leaves_the_draft_unpoured(tmp_path):
    """A FIX says the text as staged goes past its evidence. With no replacement BODY
    there is nothing to fix it WITH — and pouring it anyway files exactly the sentence
    the judge objected to. It stays staged for the next drain to judge cold."""
    srv, reg, store, _ = build(tmp_path)
    try:
        d = Distiller(reg, store)
        d.run(chunks=1)
        p = os.path.join(d.drafts_dir, "archive-on-slow-disk.md")
        before = open(p, encoding="utf-8").read()
        d.scribe = lambda task, u, max_tokens=0: (                # type: ignore[method-assign]
            "FIX\nreason: the last sentence goes past the evidence\n")
        out = d.drain()
        assert out["poured"] == 0 and out["fixed"] == 0 and out["tossed"] == 0
        assert out["fix_unparsed"] == 1 and out["left"] == 1
        assert open(p, encoding="utf-8").read() == before      # not touched, not re-signed
        assert "archive-on-slow-disk" not in store.slugs()
    finally:
        srv.shutdown()


def test_a_slug_with_no_ascii_still_lands_as_a_usable_draft(tmp_path):
    """The slug sanitiser keeps ASCII only: a store written in Japanese used to compose
    memories whose name reduced to the empty string, so every one of them was the same
    file. The fallback name has to be usable, stable, and its own."""
    srv, reg, store, jdir = build(tmp_path)
    try:
        d = Distiller(reg, store)
        src = str(jdir / "session.jsonl")
        c = _compose_one(d, src, "書庫は遅いディスクへ",
                         "The archive goes on the slow disk. The fast disk stays scratch.",
                         "put the archive on the slow disk")
        assert c and re.fullmatch(r"[a-z0-9][a-z0-9-]*", c["slug"])
        assert _compose_one(d, src, "書庫は遅いディスクへ", "x", "y")["slug"] == c["slug"]
        other = _compose_one(d, src, "速いディスクは作業用", "x", "y")["slug"]
        assert other != c["slug"]                 # two nameless slugs are not one memory
        p = d.stage(c, src)
        assert os.path.basename(p) == c["slug"] + ".md"
        # the mark signs the name, so the fallback name has to pour under itself
        assert d.pour(c["slug"])["ok"]
        assert c["slug"] in store.slugs()
    finally:
        srv.shutdown()


def test_two_drafts_with_one_slug_both_survive(tmp_path):
    """Staging straight to `<slug>.md` overwrote a gate-passed draft with no trace when
    two candidates sanitised to the same name. The second one takes a numbered name —
    signed under that name, so the next drain judges it instead of quarantining it."""
    srv, reg, store, jdir = build(tmp_path)
    try:
        d = Distiller(reg, store)
        src = str(jdir / "session.jsonl")
        quote = "put the archive on the slow disk"
        a = _compose_one(d, src, "archive", "The archive goes on the slow disk.", quote)
        pa = d.stage(a, src)
        b = _compose_one(d, src, "archive", "The fast disk is scratch space.", quote)
        pb = d.stage(b, src)
        assert (os.path.basename(pa), os.path.basename(pb)) == ("archive.md", "archive.2.md")
        assert (a["slug"], b["slug"]) == ("archive", "archive.2")
        assert "The archive goes on the slow disk." in open(pa, encoding="utf-8").read()
        # a fresh distiller talks to the fake server again, which answers POUR
        out = Distiller(reg, store).drain()
        assert out["quarantined"] == 0 and out["poured"] == 2 and out["left"] == 0
        # two memories, not one written twice — the store spells the numbered name
        # its own way, and both bodies are still there
        assert {"archive", "archive-2"} <= set(store.slugs())
        assert "The archive goes on the slow disk." in store.read("archive")
        assert "The fast disk is scratch space." in store.read("archive-2")
    finally:
        srv.shutdown()


def test_server_routes_the_right_store(tmp_path):
    """Two kura behind one port: a request naming a mode must not read the other's index."""
    from distill_kura.server import _make_handler
    import urllib.request

    a = Store(name="maker", path=str(tmp_path / "a"), label="maker")
    b = Store(name="eq", path=str(tmp_path / "b"), label="eq")
    for s in (a, b):
        s.init_files()
    a.remember("hammer", "about hammers", "body")
    b.remember("listening", "about listening", "body")
    reg = Registry(stores={"maker": a, "eq": b}, modes={"talking": "eq"},
                   models=Models.from_config({"thinker": {"url": "http://127.0.0.1:1/v1"}}),
                   default="maker")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(reg))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        j = lambda p: json.load(urllib.request.urlopen(base + p))   # noqa: E731
        assert "hammer" in j("/index")["index"]                     # default store
        assert "listening" in j("/index?store=eq")["index"]
        assert "listening" in j("/index?mode=talking")["index"]     # by mode name
        assert "listening" in j("/s/eq/index")["index"]             # path-prefixed form
        assert j("/health")["stores"] == {"maker": 1, "eq": 1}
        assert j("/doctor")["store"] == "maker"                     # bare = the default store
        assert set(j("/doctor?all=1")) == {"maker", "eq"}           # explicit = every store
        try:
            j("/index?store=nope")
        except urllib.error.HTTPError as e:
            assert e.code == 404
        else:
            raise AssertionError("an unknown store must 404, not fall back silently")
    finally:
        srv.shutdown()
