"""Containment: a store answers for its own memories and for nothing else.

These are regression tests for a real hole. Before the fix, `resolve()` accepted any
name whose `<store>/<name>.md` happened to exist, so all three of these leaked another
store's private memory:

    GET /memory/..%2Fprivate%2Fsecret?store=public   → 200, full text
    Store.read("../outside")                          → the file's contents
    a memory containing [[../private/secret]]         → walked there during recall

The fix is structural rather than a list of forbidden characters: every lookup resolves
INTO `slug_set()`, the set of memories the store actually holds. `../other/secret`, an
absolute path and a symlink alias are all simply not members. `contained()` sits behind
that as defence in depth, and files whose real path leaves the store are excluded from
`slugs()` entirely and reported by `doctor()`.

Written adversarially: each test is an escape attempt, not a happy path.
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

from distill_kura.registry import Registry     # noqa: E402
from distill_kura.server import _make_handler  # noqa: E402
from distill_kura.store import Store, contained  # noqa: E402
from distill_kura.thinker import Models        # noqa: E402

SECRET = "TOP SECRET CONTENT"
OUTSIDE = "A FILE OUTSIDE EVERY STORE"


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


ESCAPES = [
    "../private/secret",
    "../outside",
    "..%2Fprivate%2Fsecret",
    "....//private/secret",
    "../../etc/hostname",
    "/etc/hostname",
    "public/../private/secret",
    "./../private/secret",
    "..\\private\\secret",
]


# ── the store itself ────────────────────────────────────────────────────────

def test_no_escape_reaches_content_through_read(tmp_path):
    pub, _ = two_stores(tmp_path)
    for probe in ESCAPES:
        got = pub.read(probe)
        assert SECRET not in got, probe
        assert OUTSIDE not in got, probe


def test_no_escape_reaches_content_through_read_exact(tmp_path):
    pub, _ = two_stores(tmp_path)
    for probe in ESCAPES:
        assert pub.read_exact(probe) == "", probe


def test_every_resolution_lands_inside_the_store(tmp_path):
    """However wrong a guess is, it can only name a memory of THIS store."""
    pub, _ = two_stores(tmp_path)
    for probe in ESCAPES + ["harmless", "harmles", "nonsense", ""]:
        r = pub.resolve(probe)
        assert r is None or r in pub.slug_set(), (probe, r)


def test_an_explicit_read_is_exact_not_fuzzy(tmp_path):
    """`kura_read` promises that an unknown slug is unknown. Snapping to a neighbour
    would return a memory nobody asked for and call it the one requested."""
    pub, _ = two_stores(tmp_path)
    pub.remember("ssd-tier-mission", "running the big model off an SSD tier", "body")
    # The real misspelling this resolver exists for: a model adds a word.
    assert pub.resolve("ssd-tier-inference-mission") == "ssd-tier-mission"
    assert pub.read_exact("ssd-tier-inference-mission") == ""   # explicit: unknown is unknown
    assert pub.read_exact("ssd-tier-mission") != ""


def test_a_link_cannot_walk_out_of_the_store(tmp_path):
    pub, _ = two_stores(tmp_path)
    pub.remember("bait", "bait", "points at [[../private/secret]] and [[../outside]]")
    walked = pub.walk(["bait"], hops=2)
    assert walked == ["bait"]
    assert all(w in pub.slug_set() for w in walked)
    ctx = "\n".join(pub.read(w) for w in walked)
    assert SECRET not in ctx and OUTSIDE not in ctx


def test_an_index_title_cannot_point_outside_the_store(tmp_path):
    """The index is a text file a model also reads; a crafted title must not become a
    path. Titles resolve through the same slug set as everything else."""
    pub, _ = two_stores(tmp_path)
    with open(pub.index_path, "a", encoding="utf-8") as f:
        f.write("- [Innocent Title](../private/secret.md) — looks like an entry\n")
    pub._titles = None
    assert pub.read("Innocent Title") == "" or SECRET not in pub.read("Innocent Title")
    assert pub.resolve_exact("../private/secret") is None


def test_a_symlink_out_of_the_store_is_not_a_memory(tmp_path):
    pub, prv = two_stores(tmp_path)
    os.symlink(prv.file_of("secret"), os.path.join(pub.path, "alias.md"))
    pub._slugs_cache = None
    assert "alias" not in pub.slugs()
    assert pub.read("alias") == "" and pub.read_exact("alias") == ""
    # Excluded, but never silently: the eye has to see it.
    assert pub.doctor()["escaping"] == ["alias"]


def test_a_hardlink_is_reported_even_though_it_cannot_be_refused(tmp_path):
    """Found by an adversarial pass over the fix itself.

    A hardlink has no second path to resolve, so `contained()` passes and the file
    genuinely is in the store — content placed this way is served, correctly by the
    rules, and keeps serving the target's future edits. Refusing every `st_nlink > 1`
    file would take a store dark under any snapshot backup, which is the worse failure.
    So it is reported rather than excluded, and `docs/TRUST.md` says plainly that the
    boundary here is filesystem permissions, not name resolution."""
    pub, prv = two_stores(tmp_path)
    os.link(prv.file_of("secret"), os.path.join(pub.path, "hardpriv.md"))
    pub._slugs_cache = None
    d = pub.doctor()
    assert "hardpriv" in d["hardlinked"]
    assert d["escaping"] == []          # a path check genuinely cannot see this
    assert "hardpriv" in pub.slugs()    # and it really is a file in this store


def test_ordinary_memories_are_not_reported_as_hardlinked(tmp_path):
    pub, _ = two_stores(tmp_path)
    assert pub.doctor()["hardlinked"] == []


def test_study_subdirectory_memories_still_work(tmp_path):
    """The one legitimate slug containing a separator must keep working — a fix that
    breaks `_study/foo` has traded a hole for an outage."""
    pub, _ = two_stores(tmp_path)
    os.makedirs(os.path.join(pub.path, "_study"), exist_ok=True)
    with open(os.path.join(pub.path, "_study", "note.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: note\n---\n\na long-form note\n")
    pub._slugs_cache = None
    assert "_study/note" in pub.slugs()
    assert "long-form note" in pub.read_exact("_study/note")
    assert pub.resolve("note") == "_study/note"        # fuzzy still finds it


def test_contained_follows_symlinks(tmp_path):
    inside = tmp_path / "store"
    inside.mkdir()
    (tmp_path / "elsewhere.md").write_text("x", encoding="utf-8")
    os.symlink(tmp_path / "elsewhere.md", inside / "link.md")
    assert contained(str(inside), str(inside / "real.md"))
    assert not contained(str(inside), str(inside / "link.md"))
    assert not contained(str(inside), str(tmp_path / "elsewhere.md"))


def test_the_slug_cache_still_sees_a_new_memory(tmp_path):
    """The cache is keyed on directory mtime; a poured memory must be findable at once."""
    pub, _ = two_stores(tmp_path)
    assert pub.slug_set() == frozenset({"harmless"})
    pub.remember("brand-new", "just written", "body")
    assert "brand-new" in pub.slug_set()


# ── over HTTP ───────────────────────────────────────────────────────────────

def serve(reg):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(reg))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_http_memory_route_refuses_every_escape(tmp_path):
    pub, prv = two_stores(tmp_path)
    reg = Registry(stores={"public": pub, "private": prv}, modes={},
                   models=Models.from_config({}), default="public")
    srv, base = serve(reg)
    try:
        for probe in ESCAPES:
            url = f"{base}/memory/{urllib.parse.quote(probe, safe='')}?store=public"
            try:
                body = json.load(urllib.request.urlopen(url))
                assert SECRET not in (body.get("text") or ""), probe
                assert OUTSIDE not in (body.get("text") or ""), probe
            except urllib.error.HTTPError as e:
                assert e.code == 404, (probe, e.code)
    finally:
        srv.shutdown()


def test_http_memory_route_still_serves_a_real_memory(tmp_path):
    pub, _ = two_stores(tmp_path)
    reg = Registry(stores={"public": pub}, modes={}, models=Models.from_config({}),
                   default="public")
    srv, base = serve(reg)
    try:
        body = json.load(urllib.request.urlopen(f"{base}/memory/harmless"))
        assert body["slug"] == "harmless" and "nothing sensitive" in body["text"]
    finally:
        srv.shutdown()
