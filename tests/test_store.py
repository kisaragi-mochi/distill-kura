"""Store mechanics: writing, resolving, walking, and the index staying honest.

No model is involved in anything tested here — that is the point. Everything a wrong
answer could quietly corrupt is deterministic Python, and this file pins it down.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.store import Store   # noqa: E402


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
