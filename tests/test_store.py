"""Store mechanics: writing, resolving, walking, and the index staying honest.

No model is involved in anything tested here — that is the point. Everything a wrong
answer could quietly corrupt is deterministic Python, and this file pins it down.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.store import Store, is_memory_file   # noqa: E402


def make(tmp_path, name="t") -> Store:
    s = Store(name=name, path=str(tmp_path / name), label=name)
    s.init_files()
    return s


def test_remember_creates_file_and_index_line(tmp_path):
    s = make(tmp_path)
    r = s.remember("first-fact", "the trigger line", "the body", title="First")
    assert r["ok"] and r["created"]
    assert os.path.exists(s.file_of("first-fact"))
    assert "- [First](first-fact.md) — the trigger line" in s.index_text()
    assert s.slugs() == ["first-fact"]


def test_rewriting_a_memory_refreshes_its_index_line(tmp_path):
    """A stale index line keeps speaking the old fact: the index is what is read every
    time, the body only when opened."""
    s = make(tmp_path)
    s.remember("fact", "old trigger", "old body", title="Fact")
    s.remember("fact", "new trigger", "new body", title="Fact")
    idx = s.index_text()
    assert "new trigger" in idx
    assert "old trigger" not in idx
    assert idx.count("(fact.md)") == 1
    assert "new body" in s.read("fact")


def test_index_title_is_never_a_truncated_description(tmp_path):
    s = make(tmp_path)
    long = "a description that runs on well past any sensible title length, and then some"
    s.remember("slug-here", long, "body")
    line = [l for l in s.index_text().splitlines() if "(slug-here.md)" in l][0]
    title = line.split("](")[0].lstrip("- [")
    assert title == "slug-here"          # falls back to the slug, not a cut-off phrase
    assert not long.startswith(title)


def line_for(s, slug):
    return [l for l in s.index_text().splitlines() if f"({slug}.md)" in l][0]


def test_the_title_fallbacks_are_the_same_on_a_new_line_and_a_rewrite(tmp_path):
    """The two renders of an entry line had two copies of the `0 < len <= 40` rule with
    two different fallbacks. Only one of them was tested."""
    s = make(tmp_path)
    # no title, description short enough to say → the description becomes the title
    s.remember("s1", "short desc", "b")
    assert line_for(s, "s1") == "- [short desc](s1.md) — short desc"
    # a title too long to say falls back the same way
    s.remember("s2", "short desc", "b", title="x" * 41)
    assert line_for(s, "s2") == "- [short desc](s2.md) — short desc"
    # on a REWRITE with no title, the line keeps the title already on it
    s.remember("s3", "d", "b", title="Keep")
    s.remember("s3", "new d", "b")
    assert line_for(s, "s3") == "- [Keep](s3.md) — new d"


def test_write_policy_distiller_only_refuses_a_direct_write_and_accepts_a_pour(tmp_path):
    """The documented meaning of the old `readonly = true`: tools may not write, the
    distiller's verified pour may. The boolean refused BOTH, so a store advertised as
    maintained-by-the-distiller was frozen solid and nothing said so."""
    s = make(tmp_path)
    s.write_policy = "distiller-only"
    r = s.remember_direct("nope", "d", "b")
    assert not r["ok"] and "direct writes are refused" in r["error"]
    assert not os.path.exists(s.file_of("nope"))
    ok = s.pour_verified("poured", "came through the gate", "body")
    assert ok["ok"] and os.path.exists(s.file_of("poured"))


def test_write_policy_frozen_refuses_both_doors(tmp_path):
    s = make(tmp_path)
    s.write_policy = "frozen"
    assert not s.remember_direct("a", "d", "b")["ok"]
    assert not s.pour_verified("b", "d", "b")["ok"]
    assert s.slugs() == []


def test_write_policy_direct_allowed_is_the_default(tmp_path):
    s = make(tmp_path)
    assert s.write_policy == "direct-allowed"
    assert s.remember_direct("a", "d", "b")["ok"]
    assert s.pour_verified("c", "d", "b")["ok"]


def test_the_deprecated_readonly_flag_means_distiller_only(tmp_path):
    """Not frozen: `readonly` was always documented as "the distiller may still write"."""
    s = Store(name="ro", path=str(tmp_path / "ro"), readonly=True)
    s.init_files()
    assert s.write_policy == "distiller-only"
    assert not s.remember_direct("x", "d", "b")["ok"]
    assert s.pour_verified("x", "d", "b")["ok"]


def test_an_unknown_write_policy_fails_at_construction(tmp_path):
    try:
        Store(name="x", path=str(tmp_path / "x"), write_policy="readonly-ish")
    except ValueError as e:
        assert "write_policy must be one of" in str(e)
    else:
        raise AssertionError("a misspelled policy must not construct")


def test_resolve_snaps_misspelled_and_titled_names(tmp_path):
    s = make(tmp_path)
    s.remember("ssd-tier-mission", "running a huge model off an SSD tier", "body",
               title="SSD tier")
    assert s.resolve("ssd-tier-mission") == "ssd-tier-mission"
    assert s.resolve("ssd-tier-inference-mission") == "ssd-tier-mission"   # model misspelling
    assert s.resolve("SSD tier") == "ssd-tier-mission"                     # index title
    assert s.resolve("ssd-tier-mission.md") == "ssd-tier-mission"
    assert s.resolve("something entirely unrelated") is None


def test_links_walk_by_hops(tmp_path):
    s = make(tmp_path)
    s.remember("a", "a", "links to [[b]]")
    s.remember("b", "b", "links to [[c]]")
    s.remember("c", "c", "the end")
    assert s.walk(["a"], hops=0) == ["a"]
    assert s.walk(["a"], hops=1) == ["a", "b"]
    assert s.walk(["a"], hops=2) == ["a", "b", "c"]


def test_links_ignore_prose_that_is_not_a_slug(tmp_path):
    s = make(tmp_path)
    body = "a real [[link-here]] and [[この文章はリンクではありません。長い文]] prose"
    assert Store.links_of(body) == ["link-here"]


def test_doctor_sees_dead_links_and_islands(tmp_path):
    s = make(tmp_path)
    s.remember("hub", "hub", "points at [[leaf]] and [[ghost]]")
    s.remember("leaf", "leaf", "no links")
    s.remember("lonely", "lonely", "nobody links here and it links nowhere")
    d = s.doctor()
    assert d["memories"] == 3
    assert d["links_resolved"] == 1
    assert d["links_dead"] == ["hub→ghost"]
    assert d["islands"] == ["lonely"]
    assert d["not_in_index"] == [] and d["index_orphans"] == []


def test_read_log_is_append_only_and_counts(tmp_path):
    s = make(tmp_path)
    s.remember("x", "x", "body")
    s.note_read(["x"], "recall")
    s.note_read(["x"], "recall")
    counts = s.read_counts()
    assert counts["x"][0] == 2


def test_bodies_are_stored_verbatim(tmp_path):
    """A store holds code, JSON and shell. Rewriting a body to make it template-safe
    corrupts exactly the memories that carry the most detail — escaping is the job of
    whoever renders a template, at render time."""
    s = make(tmp_path)
    body = 'a snippet: {"a": {{nested}}} and ${SHELL} and 100% of it'
    s.remember("verbatim", "d", body)
    assert body in s.read("verbatim")


def test_the_charter_is_not_a_memory(tmp_path):
    """Writing a store's charter made it appear as a memory the moment it existed:
    unindexed in `doctor`, and walkable by recall."""
    s = make(tmp_path)
    with open(os.path.join(s.path, "charter.md"), "w", encoding="utf-8") as f:
        f.write("you are one worker in a memory system\n")
    with open(os.path.join(s.path, "README.md"), "w", encoding="utf-8") as f:
        f.write("about this store\n")
    s.remember("real", "a real memory", "body")
    assert s.slugs() == ["real"]
    assert s.doctor()["not_in_index"] == []
    assert not s.remember("charter", "d", "b")["ok"]       # cannot be created as one either
    # ...and the same reserved names inside _study/, where a README explaining the
    # shelf is the most natural file to leave.
    os.makedirs(os.path.join(s.path, "_study"), exist_ok=True)
    with open(os.path.join(s.path, "_study", "README.md"), "w", encoding="utf-8") as f:
        f.write("what this shelf is for\n")
    s._slugs_cache = None
    assert s.slugs() == ["real"]


def test_gate_key_corruption_is_loud_never_regenerated(tmp_path):
    import pytest
    s = Store(name="m", path=str(tmp_path / "m")); s.init_files()
    k1 = s.gate_key()
    assert s.gate_key() == k1
    p = os.path.join(s.still, "gate.key")
    open(p, "wb").write(b"short")
    with pytest.raises(RuntimeError):
        s.gate_key()          # a new key would orphan every mark — refuse loudly


def test_a_memory_is_indexed_even_when_the_comment_names_its_slug(tmp_path):
    """The index header is an HTML comment holding an EXAMPLE link. Judging "already
    indexed" against the RAW index let that example stand in for the memory's own entry
    line: the file was written, nothing pointed at it, and recall could not see it —
    invisible in the one place that is read every turn."""
    s = make(tmp_path)
    assert "(its-slug.md)" in s.index_text()      # the format hint, straight from init_files
    r = s.remember("its-slug", "the real trigger", "body", title="Its Slug")
    assert r["ok"] and r["indexed"] is True
    assert "- [Its Slug](its-slug.md) — the real trigger" in s.index_text()
    assert s.known_slugs() == ["its-slug"]
    assert s.doctor()["not_in_index"] == []


def test_a_rewrite_refreshes_the_entry_line_not_the_comments_example(tmp_path):
    """The same confusion from the other side: a comment whose example is shaped exactly
    like an entry line was edited in place of the memory's real line, leaving the index
    still speaking the old fact and the hint quietly rewritten."""
    s = make(tmp_path)
    s.remember("its-slug", "old trigger", "old body", title="Its Slug")
    example = "<!-- Each entry line is:\n- [Example](its-slug.md) — an example, not a memory.\n-->"
    s._write_index(example + "\n" + s.index_text())
    s.remember("its-slug", "new trigger", "new body", title="Its Slug")
    idx = s.index_text()
    assert example in idx                                    # the hint, byte for byte
    assert "- [Its Slug](its-slug.md) — new trigger" in idx
    assert "old trigger" not in idx
    assert sum(1 for l in idx.splitlines()
               if l.startswith("- [Its Slug]")) == 1        # refreshed, not duplicated
    assert s.known_slugs() == ["its-slug"]


def test_a_wal_intent_naming_a_store_kept_file_is_quarantined(tmp_path):
    """The intent is data, not authority. `_write` will never produce `charter.md` or
    `memory.md` as a target, so replay must not accept one: a corrupted or forged
    transaction would otherwise overwrite the file every worker of the store reads
    first, on nothing but its own say-so."""
    import hashlib
    import json
    s = make(tmp_path)
    charter = os.path.join(s.path, "charter.md")
    with open(charter, "w", encoding="utf-8") as f:
        f.write("the real charter\n")

    def forge(txid, target):
        txdir = os.path.join(s._wal_dir, txid)
        os.makedirs(txdir)
        payload = b"bytes nobody poured\n"
        with open(os.path.join(txdir, "payload-0"), "wb") as f:
            f.write(payload)
        with open(os.path.join(txdir, "intent.json"), "w", encoding="utf-8") as f:
            json.dump({"txid": txid, "slug": "x", "op": "write", "next_revision": 99,
                       "files": [{"payload": "payload-0", "target": target,
                                  "sha256": hashlib.sha256(payload).hexdigest()}]}, f)
        return txdir

    assert s._wal_intact(forge("00000000000000000001-1", "charter.md")) is None
    assert s._wal_intact(forge("00000000000000000002-1", "memory.md")) is None
    rep = s._wal_replay()
    assert rep["replayed"] == []
    assert rep["quarantined"] == ["00000000000000000001-1", "00000000000000000002-1"]
    assert open(charter, encoding="utf-8").read() == "the real charter\n"
    assert not os.path.exists(os.path.join(s.path, "memory.md"))
    assert s.revision() != 99                       # the counter did not honour the promise
    assert s.doctor()["broken_wal"] == rep["quarantined"]     # loud, not swept up
    # One predicate, both doors: every name the writer refuses is a name replay refuses.
    for slug in ("charter", "memory", "MEMORY", "profile"):
        assert not s.remember(slug, "d", "b")["ok"], slug
        assert not is_memory_file(f"{slug}.md".lower()), slug


def test_titles_follow_an_index_rewritten_by_another_writer(tmp_path):
    """`resolve()` accepts an index title, so the title map is part of name resolution.
    Built once and dropped only by whichever writer remembered to drop it, it outlived a
    tidy or a rename done anywhere else — this Store kept snapping answers onto a title
    the index no longer carries, and could not find the one it does."""
    s = make(tmp_path)
    s.remember("ssd-tier-mission", "running a model off an SSD tier", "body",
               title="Old Title")
    assert s.resolve("Old Title") == "ssd-tier-mission"      # the map is now warm
    other = Store(name="t", path=s.path, label="t")          # the distiller, another process
    other._write_index(s.index_text().replace("Old Title", "New Title"))
    assert s.resolve("New Title") == "ssd-tier-mission"
    assert "old title" not in s.titles()
    assert s.resolve("Old Title") is None


def test_a_write_over_an_escaping_symlink_inherits_no_neighbours_frontmatter(tmp_path):
    """A rewrite keeps the old file's metadata, and it used to read that metadata through
    the FUZZY resolver. A symlink pointing out of the store is not a memory of the store,
    so the name snapped to the nearest neighbour by word overlap and the new memory was
    born wearing that neighbour's session id and tags — provenance invented from a name."""
    s = make(tmp_path)
    s.pour_verified("alpha-fact", "the neighbour", "neighbour body",
                    meta={"session": "neighbour-session"}, tags=["decision"])
    outside = tmp_path / "outside.md"
    outside.write_text("---\nname: outside\n---\n\nnot ours\n", encoding="utf-8")
    os.symlink(str(outside), s.file_of("alpha-fact-note"))
    assert s.escaping() == ["alpha-fact-note"]               # excluded from every lookup
    assert s.resolve("alpha-fact-note") == "alpha-fact"      # the snap that did the damage

    r = s.pour_verified("alpha-fact-note", "its own trigger", "its own body")
    assert r["ok"]
    fm = s.frontmatter_exact("alpha-fact-note")
    assert "session" not in fm and "curation_mark" not in fm
    assert s.tags("alpha-fact-note") == ()
    assert "its own body" in s.read_exact("alpha-fact-note")
    assert "neighbour body" not in s.read_exact("alpha-fact-note")
    # ...and the neighbour it used to borrow from is untouched.
    assert s.frontmatter_exact("alpha-fact")["session"] == "neighbour-session"
    assert s.tags("alpha-fact") == ("decision",)


def test_a_grouped_index_line_is_not_rewritten_and_says_so(tmp_path):
    """The refresh matches a line of the slug's own. A slug sharing a line with its
    siblings keeps the old hook — rewriting it from one slug would swallow the others
    — and the result admits it. Pinned so extending the refresh stays a choice."""
    s = make(tmp_path)
    s.remember("fact", "old trigger", "old body", title="Fact")
    with open(s.index_path, "w", encoding="utf-8") as f:
        f.write("- a family — [Fact](fact.md) — one/[Other](other.md) — two\n")
    before = s.index_text()
    r = s.remember("fact", "new trigger", "new body", title="Fact")
    assert r["ok"] and r["indexed"] is False
    assert s.index_text() == before          # the siblings are not swallowed
    assert "new body" in s.read("fact")
