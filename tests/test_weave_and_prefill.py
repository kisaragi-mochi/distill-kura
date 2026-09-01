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


def test_markers_survive_compression(tmp_path):
    """⚠️ says "this will bite again" and ★ says "this is the important one". A trimmer
    that keeps the words and drops the marker has thrown away the point of the line."""
    s = a_store(tmp_path, n_old=1)
    for slug, desc in (("warned", "⚠️ crossing this boundary dies silently with no error "
                                  "at all, every single time, and it will happen again"),
                       ("starred", "★ the real conclusion here is that the surround matters "
                                   "more than the detail, which took a blind test to learn")):
        s.remember(slug, desc, "body")
        os.utime(s.file_of(slug), (time.time() - 400 * 86400,) * 2)
    cloth = Loom(s, scribe=None, trigger_tokens=8).weave().text
    assert "⚠" in [l for l in cloth.splitlines() if "(warned.md)" in l][0]
    assert "★" in [l for l in cloth.splitlines() if "(starred.md)" in l][0]


def test_improving_the_trimmer_invalidates_the_ledger(tmp_path):
    """The ledger reuses a line when the description and budget are unchanged — which
    silently includes "and the code that wrote it". Without a version, the trimmer can
    be fixed and nothing changes."""
    from distill_kura.weave import LEDGER_VERSION
    s = a_store(tmp_path, n_old=2)
    loom = Loom(s, scribe=None)
    loom.weave()
    import json as _json
    ledger = _json.load(open(loom.hooks_path, encoding="utf-8"))
    assert all(e["v"] == LEDGER_VERSION for e in ledger.values())
    for e in ledger.values():
        e["v"] = LEDGER_VERSION - 1
        e["hook"] = "a stale line from an older trimmer"
    _json.dump(ledger, open(loom.hooks_path, "w", encoding="utf-8"))
    assert "a stale line from an older trimmer" not in loom.weave().text


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


def test_a_trigger_that_swaps_a_digit_is_not_grounded(tmp_path):
    # A one-digit swap keeps most of its 2-grams and walks over the overlap floor —
    # and a false trigger is worn on every turn, then baked into KV by pay-forward.
    s = Store(name="w", path=str(tmp_path / "w")); s.init_files()
    loom = Loom(s, scribe=None)
    title = "SAZANAMI GPUs"
    desc = "SAZANAMI runs 12 GPUs with NVFP4 and local inference"
    assert not loom._acceptable("SAZANAMI runs 99 GPUs with NVFP4 and local inference", title, desc)
    assert loom._acceptable("SAZANAMI runs 12 GPUs with NVFP4 and local inference", title, desc)


# ── the source-hash CAS: a pour during the weave must not vanish ────────────

def test_persist_refuses_when_the_source_moved_while_weaving(tmp_path):
    """weave() reads the index, then spends model time on triggers. A memory poured
    meanwhile is missing from the cloth — yet the cloth would land NEWER than the
    index, so an mtime test would call the stale cloth fresh and pay-forward would
    bake it into KV. persist() re-hashes the index under the store lock and refuses
    a cloth whose source has moved: old cloth intact, distinct outcome, no retry
    loop of its own — re-weaving is the caller's decision."""
    s = a_store(tmp_path, n_old=2)
    loom = Loom(s, scribe=None)
    loom.write()
    before = loom.cloth_on_disk()
    cloth = loom.weave()                        # a snapshot of the index as it was
    s.remember("poured-meanwhile", "a memory poured between weave and persist", "body")
    stats = loom.persist(cloth)
    assert stats["written"] is False
    assert stats["refused"] == "source moved while weaving"
    assert loom.cloth_on_disk() == before       # the old cloth is intact
    assert loom.is_stale() is True              # and nothing pretends it is current
    # The caller re-weaves; the fresh weave sees the poured memory and lands.
    again = loom.write()
    assert again["written"] is True and "refused" not in again
    assert "poured-meanwhile" in loom.cloth_on_disk()
    assert loom.is_stale() is False


def test_staleness_is_judged_by_hash_not_mtime(tmp_path):
    """A cloth NEWER than the index proves nothing — that is exactly the state a
    mid-weave pour leaves behind. Staleness is: current index hash != the hash
    persist() verified. The serving side must fall back to the canonical index."""
    s = a_store(tmp_path, n_old=2)
    loom = Loom(s, scribe=None)
    loom.write()
    assert loom.is_stale() is False             # steady state: weave → not stale
    s.remember("brand-new", "poured after the weave", "body")
    ahead = time.time() + 3600
    os.utime(loom.out_path, (ahead, ahead))     # cloth mtime NEWER than the index
    assert loom.is_stale() is True
    pf = build(s, loom)
    assert pf.stats.get("stale") is True and pf.stats["source"] == "canonical"


def test_a_cloth_without_a_source_record_is_not_trusted(tmp_path):
    """A cloth written before the source record existed cannot be proven current.
    Unprovable is served the same as stale — the canonical index is the safe
    fallback — and one no-op re-weave heals the record."""
    s = a_store(tmp_path, n_old=1)
    loom = Loom(s, scribe=None)
    loom.write()
    os.remove(loom.state_path)
    assert loom.is_stale() is True
    again = loom.write()                        # byte-identical cloth, no churn…
    assert again["written"] is False
    assert loom.is_stale() is False             # …but the record is healed
    import json as _json
    st = _json.load(open(loom.state_path, encoding="utf-8"))
    assert st["source_sha256"] and st["cloth_sha256"]   # both ends re-recorded
    assert isinstance(st["source_revision"], int)       # and the revision beside them


def test_a_mutated_cloth_cannot_wear_a_valid_freshness_stamp(tmp_path):
    """The sidecar proves the PRODUCT as well as the source: were only the index
    hashed, a cloth corrupted or hand-edited while the index sat unchanged would
    still say fresh — and prefill would wear the mutated map on every turn. Either
    hash mismatching is stale; the canonical index is the fallback; one re-weave
    heals."""
    s = a_store(tmp_path, n_old=2)
    loom = Loom(s, scribe=None)
    loom.write()
    assert loom.is_stale() is False             # steady state unchanged
    good = loom.cloth_on_disk()
    assert "43.7" in good                       # the fresh line rides in full
    with open(loom.out_path, "w", encoding="utf-8") as f:
        f.write(good.replace("43.7", "99.9"))   # a hand-edit; the index untouched
    assert loom.is_stale() is True
    pf = build(s, loom)
    assert pf.stats.get("stale") is True and pf.stats["source"] == "canonical"
    assert "99.9" not in pf.text                # the mutation is never served
    healed = loom.write()
    assert healed["written"] is True
    assert loom.cloth_on_disk() == good and loom.is_stale() is False


def test_a_body_only_change_is_caught_by_the_revision(tmp_path):
    """The weave's real input is wider than the index text: layer_of() reads memory
    types and body dates. A body rewrite through the store leaves the index
    byte-identical — no hash can see it — but bumps the store revision, and the
    revision is part of the freshness stamp."""
    s = a_store(tmp_path, n_old=2)
    loom = Loom(s, scribe=None)
    loom.write()
    assert loom.is_stale() is False
    before = s.index_text()
    s.remember("recent-thing",
               "the run finished at 43.7 t/s after the fans went in, which was the whole point",
               "body 2020-01-01")               # the date layer_of reads has moved
    assert s.index_text() == before             # the index hash alone says "fresh"
    assert loom.is_stale() is True              # the revision says the store moved
    loom.write()
    assert loom.is_stale() is False             # one re-weave heals


def test_persist_refuses_on_a_mid_weave_body_change(tmp_path):
    """Same refusal as the poured-memory case, through the counter instead of the
    hash: a body rewritten while the loom was busy leaves the index byte-identical,
    so only the captured revision can prove the source moved."""
    s = a_store(tmp_path, n_old=2)
    loom = Loom(s, scribe=None)
    loom.write()
    before_cloth = loom.cloth_on_disk()
    cloth = loom.weave()
    before = s.index_text()
    s.remember("recent-thing",
               "the run finished at 43.7 t/s after the fans went in, which was the whole point",
               "body 2020-01-01")
    assert s.index_text() == before
    stats = loom.persist(cloth)
    assert stats["written"] is False
    assert stats["refused"] == "source moved while weaving"
    assert loom.cloth_on_disk() == before_cloth


# ── the attribution floor ───────────────────────────────────────────────────

def test_a_trigger_may_not_newly_credit_the_human(tmp_path):
    """A trigger is worn on every turn. Compression that adds 「ケンが決めた」 to a
    line that never credited anyone manufactures authority — rejected exactly like
    an invented number, and the mechanical trimmer takes over."""
    s = Store(name="w", path=str(tmp_path / "w")); s.init_files()
    loom = Loom(s, scribe=None)
    title = "storage doctrine"
    desc = "資産と正典はDATA2、作業の釜はDATA1に置く"
    assert loom._acceptable(desc, title, desc)                        # the line is fine
    assert not loom._acceptable(desc + "とケンが決めた", title, desc)   # the credit is not


def test_a_source_that_credits_the_human_may_keep_a_crediting_trigger(tmp_path):
    """The floor forbids NEW attribution only: when the index line itself says the
    human decided, the trigger repeating that is compression, not invention."""
    s = Store(name="w", path=str(tmp_path / "w")); s.init_files()
    loom = Loom(s, scribe=None)
    title = "storage doctrine"
    desc = "ケンの決裁: 資産と正典はDATA2、作業はDATA1"
    assert loom._acceptable("ケンの決裁: 資産と正典はDATA2", title, desc)
