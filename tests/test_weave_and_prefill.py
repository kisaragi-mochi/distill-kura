"""The loom and the resident block.

Every test here guards a failure that was actually observed, in the live store or in
this code while it was being written. The two that matter most are the ones about *not
losing anything*: a compression step that quietly drops a memory produces a map that
looks perfectly healthy and describes a household that no longer exists.
"""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest   # noqa: E402

from distill_kura.prefill import build, etag_of, loom_for, unreachable   # noqa: E402
from distill_kura.store import Store                                     # noqa: E402
from distill_kura.tokens import estimate                                 # noqa: E402
from distill_kura.weave import Loom, WeaveError, _links_per_line         # noqa: E402


def a_store(tmp_path, n_old: int = 6) -> Store:
    s = Store(name="s", path=str(tmp_path / "s"), label="test kura")
    s.init_files()
    s.remember("doctrine-one", "always measure before claiming", "body", type_="feedback")
    s.remember("about-ken", "prefers plain answers", "body", type_="user")
    s.remember("recent-thing",
               "the run finished at 43.7 t/s after the fans went in, which was the whole point",
               "body 2026-08-22")
    for i in range(n_old):
        s.remember(f"old-{i}",
                   f"an older note number {i} with a long trailing description that keeps "
                   f"going well past any sensible trigger length and says 12.5 GB somewhere",
                   "body from 2020-01-01")
        # Age them by writing an old date in the body and pushing mtime back.
        p = s.file_of(f"old-{i}")
        old = time.time() - 400 * 86400
        os.utime(p, (old, old))
    return s


# ── layering ────────────────────────────────────────────────────────────────

def test_layers_split_pinned_fresh_and_trigger(tmp_path):
    s = a_store(tmp_path)
    loom = Loom(s, scribe=None, fresh_days=14)
    assert loom.layer_of("doctrine-one") == "pinned"     # type: feedback
    assert loom.layer_of("about-ken") == "pinned"        # type: user
    assert loom.layer_of("recent-thing") == "fresh"
    assert loom.layer_of("old-0") == "trigger"


def test_pinned_and_fresh_lines_are_byte_identical(tmp_path):
    s = a_store(tmp_path)
    cloth = Loom(s, scribe=None).weave().text
    for slug in ("doctrine-one", "about-ken", "recent-thing"):
        original = [l for l in s.index_text().splitlines() if f"({slug}.md)" in l][0]
        assert original in cloth


def test_trigger_lines_get_shorter(tmp_path):
    s = a_store(tmp_path)
    cloth = Loom(s, scribe=None, trigger_tokens=16).weave()
    line = [l for l in cloth.text.splitlines() if "(old-0.md)" in l][0]
    original = [l for l in s.index_text().splitlines() if "(old-0.md)" in l][0]
    assert len(line) < len(original)
    assert cloth.stats["tokens_est"] < cloth.stats["source_tokens_est"]


def test_a_bulk_touch_does_not_make_the_whole_store_look_fresh(tmp_path):
    """`cp -r`, a restore or a checkout resets every mtime. Trusting it turns the entire
    index fresh, nothing is trimmed, and the mechanism has switched itself off in
    silence. Measured on the real store: 50 of 214 files shared one bulk-touch day, and
    for those mtime understated the true age by a median of 11 days."""
    s = a_store(tmp_path, n_old=12)
    for slug in s.slugs():
        os.utime(s.file_of(slug), None)        # everything "modified" just now
    loom = Loom(s, scribe=None, fresh_days=14)
    layers = [loom.layer_of(x) for x in s.slugs()]
    assert layers.count("fresh") <= 2          # the dated ones stay old
    assert layers.count("trigger") >= 10


def test_a_future_date_in_a_body_is_not_a_timestamp(tmp_path):
    s = a_store(tmp_path, n_old=1)
    s.remember("plan", "a plan mentioning 2099-01-01", "we will do it on 2099-01-01")
    old = time.time() - 400 * 86400
    os.utime(s.file_of("plan"), (old, old))
    assert Loom(s, scribe=None).layer_of("plan") == "trigger"


# ── the postcondition: nothing may be lost ──────────────────────────────────

def test_every_memory_survives_the_weave(tmp_path):
    s = a_store(tmp_path, n_old=10)
    cloth = Loom(s, scribe=None, trigger_tokens=8).weave()
    assert _links_per_line(cloth.text) == _links_per_line(s.index_text())


def test_a_line_naming_several_memories_is_left_alone(tmp_path):
    """Real indexes group related memories on one line. Rewriting such a line from the
    first slug's layer would swallow the others, and a memory missing from the map does
    not exist as far as the agent is concerned."""
    s = a_store(tmp_path, n_old=2)
    with open(s.index_path, "a", encoding="utf-8") as f:
        f.write("- a family — [A](old-0.md) — one/[B](old-1.md) — two, and a long tail "
                "of description that would certainly be trimmed if we let it\n")
    cloth = Loom(s, scribe=None, trigger_tokens=8).weave()
    assert "[A](old-0.md)" in cloth.text and "[B](old-1.md)" in cloth.text
    assert cloth.stats["grouped"] == 1


def test_the_postcondition_actually_fires(tmp_path, monkeypatch):
    """The guard has to be able to fail, or it is decoration."""
    s = a_store(tmp_path, n_old=2)
    loom = Loom(s, scribe=None)
    monkeypatch.setattr(loom, "_mechanical", lambda desc: "](ghost.md) invented")
    with pytest.raises(WeaveError):
        loom.weave()


def test_a_loom_may_not_write_over_the_canonical_index(tmp_path):
    """The original implementation read its own output, so it re-wove its own cloth and
    could never see a new memory: 41 of 129 memories were missing from a cloth that
    looked perfectly healthy, and had been for 11 days."""
    s = a_store(tmp_path, n_old=1)
    with pytest.raises(ValueError, match="canonical index"):
        Loom(s, scribe=None, out_path=s.index_path)


def test_the_cloth_is_not_mistaken_for_a_memory(tmp_path):
    s = a_store(tmp_path, n_old=1)
    loom = Loom(s, scribe=None)
    loom.write()
    assert os.path.exists(loom.out_path)
    assert not any("woven" in x for x in s.slugs())
    assert s.doctor()["not_in_index"] == []


# ── trimming quality ────────────────────────────────────────────────────────

def test_trimming_never_leaves_an_unbalanced_bracket(tmp_path):
    """A cut inside `(...)` leaves an open bracket, and the next markdown reader
    swallows whatever follows — including the next entry's link."""
    s = a_store(tmp_path, n_old=1)
    s.remember("bracketed", "a recipe（小モデル=選択と集中）with more text after it that "
                            "runs past the budget so the trim has to bite somewhere", "body")
    os.utime(s.file_of("bracketed"), (time.time() - 400 * 86400,) * 2)
    cloth = Loom(s, scribe=None, trigger_tokens=10).weave()
    for line in cloth.text.splitlines():
        for op, cl in (("(", ")"), ("（", "）"), ("[", "]")):
            assert line.count(op) == line.count(cl), line


def test_trimming_never_invents_a_unit(tmp_path):
    """"3.7 seed" must not come back as "3.7s". A trimmer that manufactures a
    measurement is doing the one thing the whole project forbids."""
    loom = Loom(a_store(tmp_path, n_old=1), scribe=None, trigger_tokens=6)
    out = loom._mechanical("persona JSON plus IDENTITY and a 3.7 seed and more words here")
    assert "3.7s" not in out


def test_trimming_does_not_cut_mid_word_when_a_break_is_near(tmp_path):
    loom = Loom(a_store(tmp_path, n_old=1), scribe=None)
    out = loom._soft_cut("ここは私自身の自律進化の土壌、工房ではない話", 14)
    assert out.endswith("土壌") or out.endswith("進化の土壌")


def test_weaving_twice_changes_nothing(tmp_path):
    s = a_store(tmp_path, n_old=3)
    loom = Loom(s, scribe=None)
    first = loom.write()
    second = loom.write()
    assert first["written"] is True and second["written"] is False


def test_changing_the_budget_actually_rewrites(tmp_path):
    """The hook ledger is keyed on the description AND the budget. Keying on the
    description alone makes `trigger_tokens` a no-op for every cached line."""
    s = a_store(tmp_path, n_old=6)
    wide = Loom(s, scribe=None, trigger_tokens=40).weave()
    tight = Loom(s, scribe=None, trigger_tokens=8).weave()
    assert tight.stats["tokens_est"] < wide.stats["tokens_est"]


# ── the budget ──────────────────────────────────────────────────────────────

def test_fit_shortens_the_fresh_window_to_get_under_budget(tmp_path):
    s = a_store(tmp_path, n_old=2)
    # A cohort a month old: full lines at fresh_days=365, triggers at fresh_days=7.
    month_ago = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    for i in range(6):
        s.remember(f"mid-{i}",
                   f"a note from a month ago, number {i}, with a description long enough "
                   f"that trimming it visibly changes the size of the whole index",
                   f"written on {month_ago}")
    loom = Loom(s, scribe=None, fresh_days=365)
    wide = loom.weave().stats["tokens_est"]
    tight = Loom(s, scribe=None, fresh_days=7).weave().stats["tokens_est"]
    assert tight < wide, "the fixture must actually be compressible"

    cloth = loom.fit(window_tokens=100_000, fraction=(tight + 5) / 100_000)
    assert cloth.stats["budget_met"] is True
    assert cloth.stats["fresh_days_used"] < 365
    assert cloth.stats["tokens_est"] <= tight + 5
    assert loom.fresh_days == 365                    # the ladder does not mutate the loom


def test_an_unreachable_budget_keeps_the_better_map(tmp_path):
    """When every rung overflows, spending the vivid layer buys nothing — so it is kept.
    Paying a cost for a target you cannot reach is the worst of both."""
    s = a_store(tmp_path, n_old=4)
    loom = Loom(s, scribe=None, fresh_days=365)
    cloth = loom.fit(window_tokens=1000, fraction=0.01)
    assert cloth.stats["budget_met"] is False
    assert cloth.stats["fresh_days_used"] == 365


def test_over_budget_is_reported_and_nothing_is_dropped(tmp_path):
    s = a_store(tmp_path, n_old=20)
    cloth = Loom(s, scribe=None).fit(window_tokens=1000, fraction=0.01)
    assert cloth.stats["over_budget"] is True
    assert "weight" in cloth.stats           # says WHERE the weight is
    assert _links_per_line(cloth.text) == _links_per_line(s.index_text())


# ── the resident block ──────────────────────────────────────────────────────

def test_the_block_is_byte_stable_across_builds(tmp_path):
    s = a_store(tmp_path, n_old=3)
    loom = loom_for(s)
    loom.write()
    a = build(s, loom)
    b = build(s, loom)
    assert a.text == b.text and a.etag == b.etag


def test_the_block_carries_nothing_that_ticks(tmp_path):
    """Anything volatile in front of the index re-prices the entire prefix every turn:
    measured, an identical preamble costs 0.14s and one changed word at the front 0.66s."""
    s = a_store(tmp_path, n_old=2)
    pf = build(s, loom_for(s))
    head = pf.text.split("\n\n")[0]
    assert not re.search(r"\d{2}:\d{2}|session|uuid|elapsed", head, re.I)


def test_a_volatile_header_is_refused_at_build_time(tmp_path):
    s = a_store(tmp_path, n_old=1)
    with pytest.raises(ValueError, match="changes over time"):
        build(s, None, header="{label} — session 2026-08-22\n")


def test_template_braces_are_escaped_on_the_way_into_a_prompt(tmp_path):
    """The store keeps what its author wrote; the renderer is what has to be careful.
    `{{today}}` in a memory would otherwise be interpolated by the prompt template."""
    s = a_store(tmp_path, n_old=1)
    s.remember("braces", "mentions {{today}} in the trigger", "body")
    pf = build(s, None)
    assert "{{" not in pf.text
    assert pf.stats["braces_escaped"] >= 1
    assert "{{today}}" in s.read("braces")          # the memory itself is untouched


def test_an_oversized_map_becomes_a_stub_not_a_truncated_list(tmp_path):
    """A truncated map is the worst artifact available: it looks complete, and every
    memory below the cut appears not to exist."""
    s = a_store(tmp_path, n_old=40)
    pf = build(s, None, window_tokens=1200)
    assert pf.stats["over_ceiling"] is True and pf.stats["map_shown"] is False
    assert "(old-0.md)" not in pf.text
    assert "not the same as the memory being empty" in pf.text


def test_a_stale_cloth_is_not_served_as_current(tmp_path):
    s = a_store(tmp_path, n_old=2)
    loom = loom_for(s)
    loom.write()
    time.sleep(0.01)
    s.remember("brand-new", "something that happened after the weave", "body")
    pf = build(s, loom)
    assert pf.stats.get("stale") is True
    assert pf.stats["source"] == "canonical"        # falls back to the complete index
    assert "brand-new" in pf.text


def test_an_unreachable_store_says_so_instead_of_going_blank(tmp_path):
    text = unreachable("the kura")
    assert text.strip()
    assert "MISSING, not that it is empty" in text


def test_the_etag_changes_only_when_the_map_does(tmp_path):
    s = a_store(tmp_path, n_old=2)
    first = build(s, None).etag
    assert build(s, None).etag == first
    s.remember("another", "one more thing", "body")
    assert build(s, None).etag != first


def test_estimator_is_close_on_mixed_text():
    """The naive chars/2 is biased low by 8-23% against real tokenizers, and low is the
    direction that silently overflows a window."""
    jp = "索引を毎回の会話に常駐させる仕組み。" * 20
    en = "the quick brown fox jumps over the lazy dog. " * 20
    assert estimate(jp) > len(jp) * 0.7          # Japanese is ~1 token per character
    assert estimate(en) < len(en) * 0.45         # English is ~1 per four
    assert etag_of("a") != etag_of("b")
