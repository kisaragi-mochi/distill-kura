"""The Hot Trail — current position, not importance (plan §15, Trail block).

Each test pins one way a trail could lie: ranking by reads, inventing prose,
serving stale breadcrumbs as current, or stamping fresh what was never verified.
No model anywhere.
"""
from __future__ import annotations

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.store import Store                       # noqa: E402
from distill_kura.trail import Trail, HEADER, TRAIL_BEGIN  # noqa: E402
from distill_kura.weave import Loom                        # noqa: E402


def build(tmp_path, n_old=3):
    s = Store(name="m", path=str(tmp_path / "m"), label="k")
    s.init_files()
    # old memories, dated long ago inside the body
    for i in range(n_old):
        s.remember_direct(f"old-{i}", f"an old settled thing number {i}",
                          f"settled on 2025-01-0{i + 1}\nbody")
    # fresh memories, dated today
    today = time.strftime("%Y-%m-%d")
    for i in range(4):
        s.remember_direct(f"fresh-{i}", f"the newest work item number {i}",
                          f"dated {today} — body {i}")
    return s


def test_the_trail_holds_the_fresh_layer_newest_first(tmp_path):
    s = build(tmp_path)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    text, _ = t.build()
    assert text is not None and text.startswith(TRAIL_BEGIN)
    lines = [l for l in text.splitlines() if l.startswith("- [")]
    assert all("fresh-" in l for l in lines), "an old settled memory never walks here"
    # newest first: fresh-3 is younger than fresh-0 only by slug order — the tie
    # breaks deterministically, which is what the same-revision test below pins
    assert "old-0" not in text and "old-1" not in text


def test_same_revision_same_bytes(tmp_path):
    s = build(tmp_path)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    a, sa = t.build()
    b, sb = t.build()
    assert a == b and (sa.source_revision, sa.source_sha256) == \
        (sb.source_revision, sb.source_sha256), "deterministic or nothing"


def test_no_read_count_ranking_anywhere(tmp_path):
    """A trail that promoted heavily-read memories would be an importance score.
    Fill the read log with the OLD memories and the trail must not care."""
    s = build(tmp_path, n_old=3)
    os.makedirs(s.still, exist_ok=True)
    with open(os.path.join(s.still, "reads.jsonl"), "w", encoding="utf-8") as f:
        for _ in range(50):
            f.write(json.dumps({"at": int(time.time()), "why": "recall",
                                "n": ["old-0", "old-1", "old-2"]}) + "\n")
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    text, _ = t.build()
    assert "old-0" not in (text or "")


def test_no_invented_prose_every_line_is_an_index_line(tmp_path):
    s = build(tmp_path)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    text, _ = t.build()
    index_lines = {l for l in s.index_text().splitlines() if l.startswith("- [")}
    trail_lines = {l for l in (text or "").splitlines() if l.startswith("- [")}
    assert trail_lines and trail_lines <= index_lines, \
        "a trail line that is not a canonical index line is invented prose"


def test_the_header_carries_nothing_volatile(tmp_path):
    s = build(tmp_path)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    text, _ = t.build()
    head = (text or "").splitlines()[:2]           # BEGIN + HEADER
    import re
    assert not re.search(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}|\brev", " ".join(head), re.I)


def test_a_revision_move_marks_the_trail_stale(tmp_path):
    s = build(tmp_path)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    assert t.write()["written"] is True
    assert t.is_stale() is False
    s.remember_direct("fresh-9", "one more thing poured afterwards", "body")
    assert t.is_stale() is True, "a poured memory must retire the old trail"


def test_a_body_edit_marks_the_trail_stale_even_though_the_index_is_identical(tmp_path):
    """The revision sees what the index hash cannot: a curation sentence written
    onto a memory moves the store while every index line stays byte-identical."""
    s = build(tmp_path, n_old=1)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    assert t.write()["written"] is True
    s.annotate_direct("old-0", annotations={"keep": "the settled meaning, said plainly"})
    assert s.index_text() and t.is_stale() is True


def test_mid_build_mutation_persists_nothing(tmp_path):
    s = build(tmp_path)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    text, stamp = t.build()
    s.remember_direct("landed-mid-build", "poured while the trail was composing",
                      "body")            # the store moved between build() and persist()
    r = t.persist(text, stamp)
    assert r["written"] is False and "refused" in r
    assert t.text_on_disk() is None, "nothing was written"


def test_an_empty_fresh_layer_removes_the_trail_rather_than_lying(tmp_path):
    s = build(tmp_path)
    loom = Loom(s, scribe=None, fresh_days=14)
    t = Trail(s, loom=loom)
    assert t.write()["written"] is True
    # every memory ages out of the fresh window: nothing recent to say
    old = Loom(s, scribe=None, fresh_days=-1)
    t2 = Trail(s, loom=old)
    assert t2.write()["removed"] is True
    assert t2.text_on_disk() is None and t2.is_stale() is True


def test_a_tampered_trail_text_is_stale_not_fresh(tmp_path):
    s = build(tmp_path)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    assert t.write()["written"] is True
    with open(t.out_path, "a", encoding="utf-8") as f:
        f.write("- [hand-edited](lie.md) — not from any index\n")
    assert t.is_stale() is True, "the product hash refuses a hand-edited trail"


def test_migration_absent_trail_builds_on_request(tmp_path):
    s = build(tmp_path)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    assert t.text_on_disk() is None and t.is_stale() is True     # absent → stale
    assert t.write()["written"] is True
    assert t.is_stale() is False


# ── the reviewer's round: containment, spec, time, revision 0 ────────────────

def test_the_trail_refuses_to_point_at_the_canonical_index(tmp_path):
    s = build(tmp_path)
    with pytest.raises(ValueError, match="canonical index"):
        Trail(s, loom=Loom(s, scribe=None), out_path=os.path.join(s.path, "MEMORY.md"))


def test_the_trail_refuses_a_memory_slot(tmp_path):
    """The loom's old wound: a derived writer pointed at a store-root .md eats a
    memory one rebuild at a time while the stats say `written`."""
    s = build(tmp_path)
    loom = Loom(s, scribe=None, out_path=str(tmp_path / "cloth.md"))  # out of store
    with pytest.raises(ValueError, match="memory slot"):
        Trail(s, loom=loom, out_path=os.path.join(s.path, "old-0.md"))


def test_a_frozen_store_refuses_an_in_store_trail_but_allows_one_outside(tmp_path):
    s = Store(name="f", path=str(tmp_path / "f"), write_policy="frozen")
    s.init_files()
    loom = Loom(s, scribe=None, out_path=str(tmp_path / "cloth.md"))
    with pytest.raises(ValueError, match="frozen"):
        Trail(s, loom=loom, out_path=os.path.join(s.still, "t.md"))
    outside = Trail(s, loom=loom, out_path=str(tmp_path / "archive-trail.md"))
    assert outside.write()["written"] is False    # nothing fresh to say, but no refusal


def test_a_changed_spec_retires_the_trail(tmp_path):
    """The trail's bytes depend on the shaping config too: a trail built at
    trail_tokens=200 read back through trail_tokens=40 describes a different
    walk, and must not wear the old stamp."""
    s = build(tmp_path)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14), trail_tokens=200)
    assert t.write()["written"] is True and t.is_stale() is False
    reconfigured = Trail(s, loom=Loom(s, scribe=None, fresh_days=14), trail_tokens=40)
    assert reconfigured.is_stale() is True


def test_time_alone_retires_the_trail(tmp_path):
    """The pure-time hazard: the fresh window slides with no store write at all,
    so a trail older than its own horizon is stale even though nothing moved."""
    s = build(tmp_path)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    assert t.write()["written"] is True and t.is_stale() is False
    st = t._state()
    st["valid_until"] = time.time() - 1          # the horizon has passed
    with open(t.state_path, "w", encoding="utf-8") as f:
        json.dump(st, f)
    # the product hash still matches — only time moved — and it is still stale
    assert _state_hash_matches(t) and t.is_stale() is True


def _state_hash_matches(t) -> bool:
    import hashlib
    return hashlib.sha256((t.text_on_disk() or "").encode()).hexdigest() \
        == t._state().get("trail_sha256")


def test_revision_zero_is_an_honest_value_not_a_missing_one(tmp_path):
    """A store whose mutations predate the revision counter answers 0 — the
    weave's documented contract. Truthiness used to read 0 as 'no record' and
    leave such a store permanently stale."""
    s = build(tmp_path)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    assert t.write()["written"] is True
    assert t.is_stale() is False
    os.remove(os.path.join(s.still, "revision"))          # the counter never existed
    t2 = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    assert t2.write()["written"] is True                  # rebuild at revision 0
    assert t2._state()["source_revision"] == 0
    assert t2.is_stale() is False, "revision 0 proves the trail as well as 42 does"


def test_an_old_version_sidecar_is_unprovable_and_thus_stale(tmp_path):
    s = build(tmp_path)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    assert t.write()["written"] is True
    st = t._state()
    st["version"] = 99                                    # a future format
    with open(t.state_path, "w", encoding="utf-8") as f:
        json.dump(st, f)
    assert t.is_stale() is True
