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
from distill_kura.distill.gate import composed_number_violations        # noqa: E402
from distill_kura.weave import Loom, WeaveError, _links_per_line         # noqa: E402


# Real index lines from the household this store was built for: Japanese, English, and
# both mixed with identifiers. Every one carries numbers, which is the point — a trigger
# is worn on every turn, so a number the trimmer composed by accident is a lie the agent
# reads all day and pay-forward bakes into KV.
REAL_LINES = [
    ("常駐して噛む脳", "★背景で日誌消化+前景0.15秒応答。記憶の新陳代謝はこの形。GPU普段0%=消化はタダ"),
    ("蔵サービス", ":8085 記憶への唯一の入り口。索引常駐＋意味の再認＋リンク歩行0.4秒。雲/ローカル/足軽/声が共有"),
    ("蔵の救命艇", "⚠️写しが8日止まり197本が世界に無かった。lifeboat.shで別盤・別筐体+復元ドリル済。地雷=sortロケール"),
    ("Qwen3.6-35B-A3B NVFP4", "vLLM0.22無改造163t/s。27B pi-tune=grafted-MTPをignoreに/ThinkingCap=思考46%減"),
    ("FreeToken/SAZANAMI", "★KV永続化 116秒→23ms(greedy一致)。床+限界費用で読む。投機がコードで勝った(18.907→23.139/k=4)"),
    ("Huihui-Qwen3.8", "TP=8で単流107.7/8並列547.5 t/s・⚠️APC明示必須・FP16素体の口 :8019 TP=8 xhigh(門3/3)"),
    ("GLM-5.2-for-3090検証", "⚠️実測1375KiB/expertでREADME 320KiB主張と矛盾・厳格PASS 4/75層(worst 0.866)→3090では不成立"),
    ("表流しの実証", "★冷キャッシュ+MemoryMax=90Gで実証: 表領域0.1-3.5%常駐で code 51.43 t/s維持。所要=ファイル72.9GiB+anon≈83GB"),
    ("bake and restore", "the bake took 796.5 seconds and the restore 0.655 seconds afterwards, measured cold"),
    ("CPU推論はDIMMファン待ち", "the run finished at 43.7 t/s after the fans went in, which was the whole point of the week"),
]


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
    monkeypatch.setattr(loom, "_mechanical",
                        lambda desc, title="": "](ghost.md) invented")
    # The floors now refuse this cut before it is worn; silence them so the lie
    # reaches the cloth and the postcondition — the backstop behind the floors —
    # is the thing under test, as before W2b.
    monkeypatch.setattr("distill_kura.floors.first_violation", lambda *a, **k: None)
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


def test_a_decimal_is_never_split_into_two_numbers(tmp_path):
    """The observed bug: the clause splitter treated the ASCII "." as a full stop, so
    "the bake took 796.5 seconds" was cut to "the bake took 796." and the trigger
    reported 796 and 5 — two measurements that appear nowhere in the memory. Unlike a
    model's candidate this path never faced the numeric floor, so the invented number
    went onto the resident map and was worn on every turn."""
    desc = "the bake took 796.5 seconds and the restore 0.655 seconds afterwards"
    for tokens in (6, 8, 10, 12, 16, 24):
        out = Loom(a_store(tmp_path, n_old=1), scribe=None,
                   trigger_tokens=tokens)._mechanical(desc, "bake and restore")
        assert "796." not in out or "796.5" in out, (tokens, out)
        assert not composed_number_violations(
            out, [{"text": f"bake and restore {desc}"}]), (tokens, out)


def test_a_decimal_is_never_split_in_japanese(tmp_path):
    """Same cut, a line with no spaces to fall back on. 。 and ； are real boundaries
    and stay; the period inside 796.5 is not one."""
    desc = "★焼成は796.5秒、復元は0.655秒だった。以後この形で運用する。⚠️冷キャッシュだと12,500GiB読み直し"
    for tokens in (6, 8, 10, 12, 16, 24):
        out = Loom(a_store(tmp_path, n_old=1), scribe=None,
                   trigger_tokens=tokens)._mechanical(desc, "焼成と復元")
        assert not composed_number_violations(
            out, [{"text": f"焼成と復元 {desc}"}]), (tokens, out)


def test_a_number_at_the_edge_of_the_budget_is_not_halved():
    """The budget lands where it lands. Every character position is a possible cut, and
    a cut inside a number invents one whichever mechanism made it — so the cut moves off
    the number rather than the number being trimmed to fit. "107.7/8" counts as one
    number here because the floor reads it as one claim."""
    for text in ("the restore finished in 0.655 seconds after the bake wrote 12,500 GB to disk",
                 "TP=8で単流107.7/8並列547.5 t/s・⚠️APC明示必須・FP16素体の口 :8019 xhigh",
                 "measured 2026-08-22 at 1.23e-4 error over 4/75 layers, worst 0.866"):
        for limit in range(6, len(text) + 3):
            out = Loom._soft_cut(text, limit)
            assert out.strip(), (limit, text)
            assert not composed_number_violations(out, [{"text": text}]), (limit, out)


def test_the_mechanical_trim_faces_the_numeric_floor_too(tmp_path):
    """A model-written trigger has to clear the numeric floor in `_acceptable`; the
    mechanical fallback did not, and it is the path that runs whenever the GPU is down —
    so the unchecked line is the one worn on the worst day. Same floor, same source,
    across every budget the loom is used at."""
    s = a_store(tmp_path, n_old=1)
    for tokens in (6, 8, 10, 12, 16, 24, 40):
        loom = Loom(s, scribe=None, trigger_tokens=tokens)
        for title, desc in REAL_LINES:
            out = loom._keep_markers(desc, loom._mechanical(desc, title))
            assert not composed_number_violations(
                out, [{"text": f"{title} {desc}"}]), (tokens, title, out)


def test_the_numeric_floor_on_the_trim_can_actually_fire(tmp_path, monkeypatch):
    """The guard has to be able to fail, or it is decoration. With a fragment that no
    honest cut could produce, the trim must reach for a wider one — never patch the
    number out, never come back blank."""
    desc = ("the ledger held steady all week. later the sheet showed the same figure "
            "again and then a long tail of words that blows any sensible budget")
    loom = Loom(a_store(tmp_path, n_old=1), scribe=None, trigger_tokens=16)
    monkeypatch.setattr(Loom, "_salient", staticmethod(lambda text: ["923ms"]))
    out = loom._mechanical(desc, "ledger")
    assert "923" not in out
    assert out.strip() and out in f"{desc}"


def test_the_trim_is_never_blank(tmp_path):
    """A blank trigger drops the memory off the map entirely, which is far worse than a
    mediocre one — so no rung of the fallback may end in silence."""
    s = a_store(tmp_path, n_old=1)
    for tokens in (1, 2, 4, 6, 24):
        loom = Loom(s, scribe=None, trigger_tokens=tokens)
        for title, desc in REAL_LINES + [("digits", "123456789012345678901234567890"),
                                         ("one", "0.6551234567890123456789012345")]:
            assert loom._mechanical(desc, title).strip(), (tokens, title)


def test_a_woven_line_never_carries_a_number_the_index_did_not(tmp_path):
    """End to end, with no model reachable: what lands in the cloth is what gets worn."""
    s = a_store(tmp_path, n_old=1)
    for i, (title, desc) in enumerate(REAL_LINES):
        s.remember(f"real-{i}", desc, "body")
        os.utime(s.file_of(f"real-{i}"), (time.time() - 400 * 86400,) * 2)
    cloth = Loom(s, scribe=None, trigger_tokens=12).weave().text
    source = s.index_text()
    for line in cloth.splitlines():
        assert not composed_number_violations(line, [{"text": source}]), line


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
    ledger = _json.load(open(loom.hooks_path, encoding="utf-8"))["payload"]
    assert all(e["v"] == LEDGER_VERSION for e in ledger.values())
    for e in ledger.values():
        e["v"] = LEDGER_VERSION - 1
        e["hook"] = "a stale line from an older trimmer"
    # Still marked, so the file itself is trusted: only the stale `v` retires the lines.
    _json.dump({"payload": ledger, "mark": loom._hooks_mark(ledger)},
               open(loom.hooks_path, "w", encoding="utf-8"))
    assert "a stale line from an older trimmer" not in loom.weave().text


def test_a_hand_edited_hook_line_is_not_worn_and_the_ledger_regenerates(tmp_path):
    """The defect this closes: hooks.json was plain JSON, so a hook line edited by hand
    reached the production cloth on the next weave. The file now carries the ledger's
    mark, and a file whose mark does not verify is treated as EMPTY — every hook is
    regenerated (mechanical when no scribe; that is the intended cost), never
    partially trusted."""
    import json as _json
    s = a_store(tmp_path, n_old=2)
    loom = Loom(s, scribe=None)
    loom.weave()
    env = _json.load(open(loom.hooks_path, encoding="utf-8"))
    assert {"payload", "mark"} <= set(env)
    env["payload"]["old-0"]["hook"] = "a hand-edited line worn on every turn"
    _json.dump(env, open(loom.hooks_path, "w", encoding="utf-8"))
    cloth = loom.weave()
    assert "a hand-edited line worn on every turn" not in cloth.text
    assert cloth.stats["hooks_reused"] == 0 and cloth.stats["hooks_written"] >= 1
    # The rewritten ledger verifies again: the lie did not survive on disk either.
    fresh = loom._hooks()
    assert fresh and fresh["old-0"]["hook"] != "a hand-edited line worn on every turn"


def test_a_legacy_unmarked_hooks_file_is_ignored_not_worn(tmp_path):
    """The old format was a bare slug→entry dict. An unmarked file is neither trusted
    entry-by-entry nor upgraded in place: it reads as empty, every hook regenerates,
    and nothing crashes. The entries here are otherwise reusable, so hooks_reused == 0
    shows the FILE was refused, not the entries."""
    import json as _json
    s = a_store(tmp_path, n_old=2)
    loom = Loom(s, scribe=None)
    loom.weave()
    env = _json.load(open(loom.hooks_path, encoding="utf-8"))
    _json.dump(env["payload"], open(loom.hooks_path, "w", encoding="utf-8"))
    cloth = loom.weave()
    assert cloth.stats["hooks_reused"] == 0 and cloth.stats["hooks_written"] >= 1


def test_a_frozen_store_grows_no_hooks_file(tmp_path):
    """How frozen is handled today, kept: a loom whose cloth would land inside a frozen
    store is refused at construction, before any hook is computed or saved; and the
    no-model status weave (generate=False) never saves. The mark must not add a new
    way for a frozen archive to grow."""
    s = a_store(tmp_path, n_old=2)
    s.write_policy = "frozen"
    with pytest.raises(ValueError, match="frozen"):
        Loom(s, scribe=None)                   # the default cloth lives in the store
    assert not os.path.exists(os.path.join(s.still, "hooks.json"))
    assert not os.path.exists(os.path.join(s.still, "gate.key"))
    outside = Loom(s, scribe=None, out_path=str(tmp_path / "cloth.md"))
    cloth = outside.weave(generate=False)
    assert cloth.stats["hooks_written"] == 0
    assert not os.path.exists(os.path.join(s.still, "hooks.json"))


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


# ── the hook faces the adaptive floors (W2b) ────────────────────────────────


class HookScribe:
    """A scripted scribe: answers the HOOK request for a title with fixed text."""
    def __init__(self, cues: dict[str, str]):
        self.cues = cues

    def ask(self, system, user, **kw):
        title = user.split("\n", 1)[0].removeprefix("title: ").strip()
        return self.cues.get(title, "")


def test_a_lying_hook_is_never_worn_the_mechanical_or_canonical_line_is(tmp_path):
    """The defect W2b closes: the ledger wore the scribe's answer once it cleared the
    numeric floor — the house store wore `d62189` for `6d62189` and ★ on lines that
    never had them (19/67 memories, measured 2026-09-02). A hook that invents a
    marker, cuts an identifier and re-binds a number is refused; the mechanical trim
    (or, if that lies too, the canonical line) is worn instead, and the reason is
    recorded on the entry."""
    s = Store(name="f", path=str(tmp_path / "f"), label="test kura")
    s.init_files()
    s.remember("fresh-note", "touched today, so it is worn verbatim", "body")
    s.remember("old-build",
               "the old build ds4-tp8-engine-canonical ran from 12.5 GB of weights "
               "for the whole week without a restart", "body from 2020-01-01")
    p = s.file_of("old-build")
    old = time.time() - 400 * 86400
    os.utime(p, (old, old))
    # Passes the OLD floors (grounded, no composed number) and would have been worn.
    scribe = HookScribe({"old-build": "★the ds4-tp8 build ran from 12.5 GB"})
    loom = Loom(s, scribe=scribe)
    cloth = loom.weave()
    line = next(l for l in cloth.text.splitlines() if "old-build" in l)
    for lie in ("★", "ds4-tp8 build"):          # invented marker, cut identifier
        assert lie not in line
    title = "old-build"
    desc = ("the old build ds4-tp8-engine-canonical ran from 12.5 GB of weights "
            "for the whole week without a restart")
    mech = loom._keep_markers(desc, loom._mechanical(desc, title))
    entry = loom._hooks()["old-build"]
    assert entry["hook"] in (mech, desc)        # mechanical or canonical, never the lie
    assert entry["floor"]                       # the reason is recorded
    assert isinstance(entry["floor"], str)


def test_a_clean_scribe_hook_is_worn_unchanged(tmp_path):
    """The floors are a gate, not a rewriter: an honest hook the floors accept is worn
    exactly as the scribe wrote it, with `floor` recorded as None."""
    s = a_store(tmp_path)
    loom = Loom(s, scribe=HookScribe({"old-0": "an older note number 0"}))
    cloth = loom.weave()
    entry = loom._hooks()["old-0"]
    assert entry["hook"] == "an older note number 0"
    assert entry["by"] == "model"
    assert entry["floor"] is None
    assert "— an older note number 0" in cloth.text


def test_the_postcondition_holds_when_the_scribe_invents_a_link(tmp_path):
    """Without the floors, a scribe answer naming a [[link]] the line never had would
    land in the cloth — the one layer the postcondition cannot see into, because the
    link sits after the slug. The floors refuse it and the weave still holds."""
    s = a_store(tmp_path)
    loom = Loom(s, scribe=HookScribe({"old-2": "says 12.5 GB somewhere [[g]]"}))
    raw = s.index_text()
    cloth = loom.weave()                        # raises WeaveError if the map lies
    assert _links_per_line(raw) == _links_per_line(cloth.text)
    assert "[[g]]" not in cloth.text


# ── the one place a `[prefill]` table becomes a block ─────────────────────────

def test_budget_of_reads_the_defaults_and_lets_a_caller_override_them():
    """The three numbers used to be spelled out at every call site, so a changed
    default would have moved some callers and not others."""
    from distill_kura import prefill as pf
    assert pf.budget_of({}) == (pf.DEFAULT_WINDOW_TOKENS, pf.DEFAULT_BUDGET_FRACTION,
                                pf.DEFAULT_HARD_FRACTION)
    assert pf.budget_of(None) == pf.budget_of({})
    # An empty override (an absent query string) falls through to the config; a real
    # one wins, string or number.
    assert pf.budget_of({"window_tokens": 4096, "budget_fraction": 0.1},
                        window_tokens="", fraction=None) == (4096, 0.1, 0.20)
    assert pf.budget_of({}, window_tokens="2048", fraction="0.5")[:2] == (2048, 0.5)


def test_build_from_cfg_with_no_config_is_the_bare_build(tmp_path):
    from distill_kura import prefill as pf
    s = a_store(tmp_path, n_old=4)
    assert pf.build_from_cfg(s, None, {}).etag == build(s, None).etag
    assert pf.build_from_cfg(s, None, None).etag == build(s, None).etag


def test_build_from_cfg_honours_the_window_in_the_config(tmp_path):
    from distill_kura import prefill as pf
    s = a_store(tmp_path, n_old=40)
    assert pf.build_from_cfg(s, None, {"window_tokens": 1200}).stats["over_ceiling"] is True

def test_the_loom_leaves_no_tmp_debris_and_names_its_tmp_per_process(tmp_path, monkeypatch):
    """Three writers in this module hand-rolled the same tmp+replace. The per-process
    tmp name is what keeps two weaves running side by side from tearing each other's
    ledger, and nothing checked it."""
    s = a_store(tmp_path)
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(a), real(a, b))[1])
    Loom(s, scribe=None).write()
    assert not [f for f in os.listdir(s.still) if ".tmp" in f]
    assert seen and all(t.endswith(f".tmp.{os.getpid()}") for t in seen)
