"""The adaptive minimum recognition trigger — M4, shadow mode.

Every test is written as the failure it prevents, and from BOTH review axes: the
Red Team (can a shorter cue smuggle a lie, route wrongly, or lose a memory?) and the
Warmth Team (does strictness make a genuinely recognisable cue impossible?). Nothing
here touches the production cloth unless it says so.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest   # noqa: E402

from distill_kura.adaptive import ADAPTIVE_VERSION, Adaptive          # noqa: E402
from distill_kura.store import Store                                   # noqa: E402
from distill_kura.tokens import estimate                               # noqa: E402
from distill_kura.weave import Loom                                    # noqa: E402


class Scribe:
    """A scripted scribe: answers the CUE labels with what the test dictates."""
    def __init__(self, cues: dict[str, dict[int, str]]):
        self.cues = cues            # title -> {step: text}
        self.calls = 0

    def ask(self, system, user, **kw):
        self.calls += 1
        title = user.split("\n", 1)[0].removeprefix("title: ").strip()
        got = self.cues.get(title, {})
        return "\n".join(f"CUE{s}: {t}" for s, t in got.items())


def old_store(tmp_path, entries: list[tuple[str, str, str]]) -> Store:
    """Trigger-layer memories only (aged past fresh_days), plus one pinned line."""
    s = Store(name="s", path=str(tmp_path / "s"), label="test kura")
    s.init_files()
    s.remember("doctrine", "always measure before claiming", "body", type_="feedback")
    for slug, title, desc in entries:
        s.remember(slug, desc, "body from 2020-01-01", title=title)
        p = s.file_of(slug)
        old = time.time() - 400 * 86400
        os.utime(p, (old, old))
    return s


A = ("fans-first", "CPU推論はDIMMファン待ち",
     "the run finished at 43.7 t/s only after the DIMM fans went in, the whole point of the week")
B = ("fans-second", "DIMM fans on the other box",
     "the other box also needed the DIMM fans went in before its run reached 12.5 t/s in August")
C = ("ssd-tier", "SSD tier inference",
     "running the 2.6T model off an SSD tier at 3.4 t/s; ⚠ the table must stay on disk")


# ── Red Team ────────────────────────────────────────────────────────────────

def test_an_8_token_cue_that_collides_with_a_neighbour_is_refused(tmp_path):
    """Two memories share their opening words. An 8-token cue made of those words
    recognises BOTH — so it recognises neither, and the shadow must say so."""
    s = old_store(tmp_path, [A, B, C])
    sc = Scribe({A[1]: {8: "DIMM fans went in", 12: "43.7 t/s only after the DIMM fans went in",
                        16: "run reached 43.7 t/s only after the DIMM fans went in",
                        24: "the run finished at 43.7 t/s only after the DIMM fans went in"}})
    ad = Adaptive(s, Loom(s, scribe=None, trigger_tokens=24), scribe=sc)
    rep = ad.shadow()
    r = rep["memories"]["fans-first"]
    assert r["why_not_shorter"].get("8", "").startswith(("ambiguous", "no confident hit", "ungrounded", "names another"))
    assert r["shortest_safe"] and "43.7" in r["shortest_safe"]
    assert r["shortest_safe_tokens"] <= r["current_tokens"]


def test_a_cue_that_invents_a_number_is_refused_with_the_reason(tmp_path):
    s = old_store(tmp_path, [A, C])
    sc = Scribe({A[1]: {8: "run hit 49 t/s", 24: "the run finished at 43.7 t/s only after the DIMM fans went in"}})
    ad = Adaptive(s, Loom(s, scribe=None), scribe=sc)
    r = ad.shadow()["memories"]["fans-first"]
    assert r["why_not_shorter"]["8"].startswith("invented number")
    assert "49" not in (r["shortest_safe"] or "")


def test_a_cue_that_invents_a_marker_or_a_link_is_refused(tmp_path):
    s = old_store(tmp_path, [A, C])
    sc = Scribe({A[1]: {8: "⚠ DIMM fans 43.7 t/s", 12: "43.7 t/s [[fans]] DIMM fans went in",
                        24: "the run finished at 43.7 t/s only after the DIMM fans went in"}})
    ad = Adaptive(s, Loom(s, scribe=None), scribe=sc)
    r = ad.shadow()["memories"]["fans-first"]
    assert r["why_not_shorter"]["8"].startswith("invented marker")
    assert r["why_not_shorter"]["12"].startswith("invented link")


def test_a_cue_that_credits_the_human_is_refused(tmp_path):
    s = old_store(tmp_path, [A, C])
    sc = Scribe({A[1]: {8: "Ken decided DIMM fans 43.7 t/s",
                        24: "the run finished at 43.7 t/s only after the DIMM fans went in"}})
    ad = Adaptive(s, Loom(s, scribe=None), scribe=sc)
    r = ad.shadow()["memories"]["fans-first"]
    assert "credits the human" in r["why_not_shorter"]["8"]


def test_a_cue_that_flips_a_negation_is_refused_both_ways(tmp_path):
    neg = ("no-mmap", "NAS mmap", "the model must not be loaded from the NAS: CIFS cannot mmap, it OOMs")
    pos = ("mmap-ok", "local mmap", "the model loads fine from local NVMe with mmap at 2.1 GB/s")
    s = old_store(tmp_path, [neg, pos, C])
    sc = Scribe({neg[1]: {8: "model loaded from the NAS: CIFS mmap", 24: "must not be loaded from the NAS: CIFS cannot mmap, it OOMs"},
                 pos[1]: {8: "model does not load from NVMe mmap", 24: "loads fine from local NVMe with mmap at 2.1 GB/s"}})
    ad = Adaptive(s, Loom(s, scribe=None), scribe=sc)
    rep = ad.shadow()["memories"]
    assert rep["no-mmap"]["why_not_shorter"]["8"] == "negation dropped"
    assert rep["mmap-ok"]["why_not_shorter"]["8"].startswith("negation invented")


def test_the_ladder_ends_at_the_canonical_line_never_below_recognition(tmp_path):
    """Every candidate fails, the current trigger fails too: the shadow falls back
    to the canonical line. Recognition is never traded for tokens (§7.7)."""
    s = old_store(tmp_path, [A, B])
    sc = Scribe({A[1]: {8: "DIMM fans", 12: "DIMM fans went in", 16: "DIMM fans went in, August", 24: "fans"}})
    loom = Loom(s, scribe=None, trigger_tokens=24)
    ad = Adaptive(s, loom, scribe=sc)
    r = ad.shadow()["memories"]["fans-first"]
    assert r["chosen"] in ("current", "canonical")
    assert r["shortest_safe"] in (A[2], loom._mechanical(A[2], A[1]))


# ── shadow means shadow ─────────────────────────────────────────────────────

def test_the_production_cloth_is_byte_identical_under_shadow(tmp_path):
    s = old_store(tmp_path, [A, B, C])
    loom = Loom(s, scribe=None, trigger_tokens=24)
    before = loom.weave().text
    Adaptive(s, loom, scribe=Scribe({A[1]: {8: "43.7 t/s DIMM fans"}})).shadow()
    assert loom.weave().text == before
    assert os.path.exists(os.path.join(s.still, "adaptive.json"))


def test_an_untouched_old_config_never_runs_the_shadow(tmp_path):
    from distill_kura.registry import Registry
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f'[stores.main]\npath = "{tmp_path / "main"}"\n[prefill]\ntrigger_tokens = 24\n', encoding="utf-8")
    Store(name="main", path=str(tmp_path / "main")).init_files()
    reg = Registry.load(str(cfg))
    pc = reg.prefill_cfg_for(reg.stores["main"])
    assert not pc.get("adaptive_triggers") and not pc.get("adaptive_apply")
    assert not os.path.exists(os.path.join(reg.stores["main"].still, "adaptive.json"))


def test_the_config_refuses_a_ladder_that_cannot_mean_what_it_says(tmp_path):
    from distill_kura.registry import Registry
    Store(name="main", path=str(tmp_path / "main")).init_files()
    for bad in ('trigger_steps = [12, 8]', 'trigger_steps = [8, 32]\ntrigger_tokens = 24',
                'adaptive_apply = true'):
        cfg = tmp_path / "kura.toml"
        cfg.write_text(f'[stores.main]\npath = "{tmp_path / "main"}"\n[prefill]\n{bad}\n', encoding="utf-8")
        with pytest.raises(ValueError):
            Registry.load(str(cfg))


def test_the_second_shadow_calls_no_model_and_a_callsign_rejudges_one_memory(tmp_path):
    s = old_store(tmp_path, [A, B, C])
    sc = Scribe({A[1]: {24: "the run finished at 43.7 t/s only after the DIMM fans went in"},
                 B[1]: {24: "the other box also needed DIMM fans before its run reached 12.5 t/s"},
                 C[1]: {24: "running the 2.6T model off an SSD tier at 3.4 t/s; ⚠ the table must stay on disk"}})
    loom = Loom(s, scribe=None)
    ad = Adaptive(s, loom, scribe=sc)
    ad.shadow(); first = sc.calls
    assert first == 3
    ad.shadow(); assert sc.calls == first          # steady state: zero model calls
    # A callsign lands on ONE memory: no model is called for it (candidates are the
    # model's; the callsign is the human's) — it is re-JUDGED, and only it changes.
    import distill_kura.adaptive as mod
    real = mod.Adaptive._callsigns
    mod.Adaptive._callsigns = lambda self, slug: [("全員野球", "r1")] if slug == "ssd-tier" else []
    try:
        rep = Adaptive(s, loom, scribe=sc).shadow()
        assert sc.calls == first
        assert rep["memories"]["ssd-tier"]["candidates"]["callsign"] == "全員野球"
        assert rep["memories"]["fans-first"]["candidates"]["callsign"] is None
    finally:
        mod.Adaptive._callsigns = real


def test_the_shadow_report_adds_up(tmp_path):
    s = old_store(tmp_path, [A, C])
    sc = Scribe({A[1]: {12: "43.7 t/s after the DIMM fans went in"},
                 C[1]: {12: "2.6T model off an SSD tier at 3.4 t/s ⚠"}})
    rep = Adaptive(s, Loom(s, scribe=None), scribe=sc).shadow()
    sm = rep["summary"]
    assert rep["version"] == ADAPTIVE_VERSION and sm["memories"] == 2
    assert sm["current_tokens_total"] - sm["shortest_safe_tokens_total"] == sm["saved_tokens"]
    assert sum(sm["by_shortest"].values()) == 2
    d = json.load(open(os.path.join(s.still, "adaptive.json"), encoding="utf-8"))
    assert d["summary"] == sm


# ── Warmth Team ─────────────────────────────────────────────────────────────

def test_a_japanese_only_cue_with_no_ascii_can_still_be_the_shortest(tmp_path):
    ja = ("kura-lifeboat", "蔵の救命艇", "写しが八日止まり百九十七本が世界に無かった。lifeboat.sh で別盤へ、復元ドリル済み")
    other = ("index-diet", "索引の痩身", "痩せた索引で閃きは死なず。混合索引は掟太・直近太・残りは引き金")
    s = old_store(tmp_path, [ja, other])
    sc = Scribe({ja[1]: {8: "写しが八日止まり", 24: "写しが八日止まり百九十七本が世界に無かった。lifeboat.sh で別盤へ"}})
    r = Adaptive(s, Loom(s, scribe=None), scribe=sc).shadow()["memories"]["kura-lifeboat"]
    assert r["shortest_safe"] == "写しが八日止まり", r["why_not_shorter"]


def test_a_render_wears_the_shortest_safe_cues_and_keeps_every_link(tmp_path):
    from distill_kura.weave import _links_per_line
    s = old_store(tmp_path, [A, C])
    sc = Scribe({A[1]: {12: "43.7 t/s after the DIMM fans went in"},
                 C[1]: {12: "2.6T model off an SSD tier at 3.4 t/s ⚠"}})
    loom = Loom(s, scribe=None)
    ad = Adaptive(s, loom, scribe=sc)
    text = ad.render(ad.shadow())
    assert "43.7 t/s after the DIMM fans went in" in text
    assert _links_per_line(text) == _links_per_line(s.index_text())
    assert estimate(text) < estimate(loom.weave().text)


# ── the red team's own cases ───────────────────────────────────────────────

def test_a_cue_that_keeps_the_negated_noun_and_drops_the_negator_is_refused(tmp_path):
    src = ("evolution-soil", "自律進化の土壌", "ケンの願い: ここは私自身の自律進化の土壌、工房ではない。種を育てよ")
    s = old_store(tmp_path, [src, C])
    sc = Scribe({src[1]: {8: "自律進化の土壌、工房", 24: "自律進化の土壌、工房ではない。種を育てよ"}})
    r = Adaptive(s, Loom(s, scribe=None), scribe=sc).shadow()["memories"]["evolution-soil"]
    assert r["why_not_shorter"]["8"] == "negation dropped"
    assert "ではない" in r["shortest_safe"]


def test_a_cue_that_reassigns_a_number_to_another_referent_is_refused(tmp_path):
    src = ("tp-layout", "GPU layout", "GPU 8枚 TP=4 で 116秒→23ms、KV永続化が効いた")
    s = old_store(tmp_path, [src, C])
    sc = Scribe({src[1]: {8: "GPU 4枚 TP=8", 12: "23ms→116秒 KV永続化", 24: "GPU 8枚 TP=4 で 116秒→23ms、KV永続化"}})
    r = Adaptive(s, Loom(s, scribe=None), scribe=sc).shadow()["memories"]["tp-layout"]
    assert r["why_not_shorter"]["8"].startswith("number re-bound")
    assert r["why_not_shorter"]["12"].startswith(("arrow reversed", "number re-bound"))
    assert r["chosen"] in ("24", "current", "canonical")


def test_a_cue_that_drops_the_retirement_word_is_refused(tmp_path):
    src = ("glm52-serving", "GLM-5.2 local serving", "⚠️2026-08-17退役(常設はHuihui+K3の二人だけ)。436GB GGUF・純CPU 8.97 t/s の記録")
    s = old_store(tmp_path, [src, C])
    sc = Scribe({src[1]: {8: "⚠️436GB GGUF 8.97 t/s", 24: "⚠️2026-08-17退役。436GB GGUF・純CPU 8.97 t/s"}})
    r = Adaptive(s, Loom(s, scribe=None), scribe=sc).shadow()["memories"]["glm52-serving"]
    assert r["why_not_shorter"]["8"] == "retirement word dropped"


def test_an_identifier_cut_in_the_middle_is_refused(tmp_path):
    src = ("ds4-engine", "TP8エンジンの所在", "DATA1/ds4 は lna-lab/DS4-For-TP8; ds4-tp8-engine-canonical に 8/18 の最後の姿")
    s = old_store(tmp_path, [src, C])
    sc = Scribe({src[1]: {8: "ds4-tp8 の最後の姿", 24: "ds4-tp8-engine-canonical に 8/18 の最後の姿"}})
    r = Adaptive(s, Loom(s, scribe=None), scribe=sc).shadow()["memories"]["ds4-engine"]
    assert r["why_not_shorter"]["8"].startswith("cut or invented identifier")


def test_a_newly_poured_neighbour_rejudges_a_cached_cue_without_a_model(tmp_path):
    """The verdict is a global property: a cue safe among these neighbours may be
    ambiguous once one more arrives. Candidates stay cached; the tests run again."""
    s = old_store(tmp_path, [A, C])
    sc = Scribe({A[1]: {8: "DIMM fans went in", 24: "the run finished at 43.7 t/s only after the DIMM fans went in"}})
    loom = Loom(s, scribe=None)
    first = Adaptive(s, loom, scribe=sc).shadow()["memories"]["fans-first"]
    calls = sc.calls
    s.remember(B[0], B[2], "body from 2020-01-01", title=B[1])
    p = s.file_of(B[0]); old = time.time() - 400 * 86400; os.utime(p, (old, old))
    second = Adaptive(s, loom, scribe=sc).shadow()["memories"]["fans-first"]
    assert sc.calls == calls + 1                  # B's own call; none for A
    if first["chosen"] == "8":
        assert second["chosen"] != "8" or "8" in second["why_not_shorter"] or second["shortest_safe"] != "DIMM fans went in"


def test_a_callsign_is_accepted_by_its_receipt_not_by_gram_overlap(tmp_path, monkeypatch):
    import distill_kura.adaptive as mod
    s = old_store(tmp_path, [C, A])
    sc = Scribe({C[1]: {24: "running the 2.6T model off an SSD tier at 3.4 t/s; ⚠ the table must stay on disk"}})
    monkeypatch.setattr(mod.Adaptive, "_callsigns", lambda self, slug: [("全員野球", "r1")] if slug == "ssd-tier" else [])
    monkeypatch.setattr(mod.Adaptive, "routes_by_callsign", lambda self, cand, slug: cand == "全員野球" and slug == "ssd-tier")
    r = Adaptive(s, Loom(s, scribe=None), scribe=sc).shadow()["memories"]["ssd-tier"]
    assert r["chosen"] == "callsign" and r["via"] == "receipt" and r["shortest_safe"] == "全員野球"
    assert r["callsign_routes"] is True


def test_an_ambiguous_callsign_is_never_offered(tmp_path, monkeypatch):
    """The ledger keeps ambiguous keys out of `cues`; the shadow reads only `cues`."""
    import distill_kura.adaptive as mod
    s = old_store(tmp_path, [C, A])
    sc = Scribe({C[1]: {24: "running the 2.6T model off an SSD tier at 3.4 t/s; ⚠ the table must stay on disk"}})
    monkeypatch.setattr(mod.Adaptive, "_ledger", lambda self: {"cues": {}, "ambiguous": {"全員野球": ["ssd-tier", "fans-first"]}})
    r = Adaptive(s, Loom(s, scribe=None), scribe=sc).shadow()["memories"]["ssd-tier"]
    assert r["candidates"]["callsign"] is None and r["chosen"] != "callsign"


def test_a_frozen_store_grows_no_adaptive_files(tmp_path):
    s = old_store(tmp_path, [A, C])
    s.write_policy = "frozen"
    sc = Scribe({A[1]: {24: "the run finished at 43.7 t/s only after the DIMM fans went in"}})
    loom = Loom(s, scribe=None, out_path=str(tmp_path / "cloth.md"))
    rep = Adaptive(s, loom, scribe=sc).shadow()
    assert rep["summary"]["memories"] >= 1
    assert not os.path.exists(os.path.join(s.still, "adaptive.json"))
    assert not os.path.exists(os.path.join(s.still, "adaptive.hooks.json"))


def test_a_title_rename_regenerates_the_candidates(tmp_path):
    s = old_store(tmp_path, [A, C])
    sc = Scribe({A[1]: {24: "the run finished at 43.7 t/s only after the DIMM fans went in"},
                 "renamed": {24: "the run finished at 43.7 t/s only after the DIMM fans went in"}})
    loom = Loom(s, scribe=None)
    Adaptive(s, loom, scribe=sc).shadow(); n = sc.calls
    s.remember(A[0], A[2], "body from 2020-01-01", title="renamed")
    p = s.file_of(A[0]); old = time.time() - 400 * 86400; os.utime(p, (old, old))
    Adaptive(s, loom, scribe=sc).shadow()
    assert sc.calls == n + 1


def test_a_missing_label_is_recorded_as_not_offered(tmp_path):
    s = old_store(tmp_path, [A, C])
    sc = Scribe({A[1]: {24: "the run finished at 43.7 t/s only after the DIMM fans went in"}})
    r = Adaptive(s, Loom(s, scribe=None), scribe=sc).shadow()["memories"]["fans-first"]
    assert r["why_not_shorter"]["8"] == "not offered" and r["candidates"]["8"] is None


def test_the_summary_reports_per_script(tmp_path):
    ja = ("kura-lifeboat", "蔵の救命艇", "写しが八日止まり百九十七本が世界に無かった。復元ドリル済み")
    s = old_store(tmp_path, [A, ja])
    sc = Scribe({A[1]: {24: "the run finished at 43.7 t/s only after the DIMM fans went in"},
                 ja[1]: {24: "写しが八日止まり百九十七本が世界に無かった"}})
    sm = Adaptive(s, Loom(s, scribe=None), scribe=sc).shadow()["summary"]
    assert set(sm["by_script"]) >= {"en", "ja"}


def test_the_cli_renders_the_shadow_to_a_path_but_never_onto_the_cloth(tmp_path):
    from distill_kura.cli import main
    s = old_store(tmp_path, [A, C])
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f'[stores.s]\npath = "{s.path}"\n[prefill]\ntrigger_tokens = 24\nadaptive_triggers = true\n', encoding="utf-8")
    out = tmp_path / "map.md"
    assert main(["-c", str(cfg), "-s", "s", "weave", "--no-model", "--adaptive-out", str(out)]) == 0
    assert out.exists() and "(fans-first.md)" in out.read_text(encoding="utf-8")
    import pytest as _pt
    with _pt.raises(SystemExit):
        main(["-c", str(cfg), "-s", "s", "weave", "--no-model",
              "--adaptive-out", os.path.join(s.still, "index.woven.md")])


# ── the cache is a file, not a witness (FAILURES FOUND #1) ──────────────────

def test_a_tampered_cache_entry_is_not_worn_as_is(tmp_path):
    """A cue written into adaptive.hooks.json by hand goes through the same floors
    as a fresh answer; a malformed entry (wrong label / non-string) is regenerated."""
    s = old_store(tmp_path, [A, C])
    sc = Scribe({A[1]: {24: "the run finished at 43.7 t/s only after the DIMM fans went in"}})
    loom = Loom(s, scribe=None)
    ad = Adaptive(s, loom, scribe=sc)
    ad.shadow()
    cache_path = os.path.join(s.still, "adaptive.hooks.json")
    cache = json.load(open(cache_path, encoding="utf-8"))
    # (a) a lie planted under a valid label is floored, never chosen
    cache["fans-first"]["cues"]["8"] = "the run finished at 99.9 t/s"
    json.dump(cache, open(cache_path, "w", encoding="utf-8"))
    calls = sc.calls
    r = Adaptive(s, loom, scribe=sc).shadow()
    rec = r["memories"]["fans-first"]
    assert sc.calls == calls                      # reused: no model call
    assert rec["chosen"] != "8" and rec["why_not_shorter"]["8"].startswith("invented number")
    assert r["summary"]["cues_reused"] >= 1
    # (b) a malformed entry is not reused at all
    cache["fans-first"]["cues"]["8"] = 12345
    json.dump(cache, open(cache_path, "w", encoding="utf-8"))
    calls = sc.calls
    Adaptive(s, loom, scribe=sc).shadow()
    assert sc.calls == calls + 1                  # regenerated for A


def test_an_untestable_cue_is_told_apart_from_an_ambiguous_one(tmp_path):
    """A cue made only of stop-grams (or of nothing the store knows) cannot be
    tested; that is a different fact from 'hits the wrong memory'."""
    s = old_store(tmp_path, [A, C])
    ad = Adaptive(s, Loom(s, scribe=None), scribe=None)
    why = ad.recognises("ZZQXV", "fans-first")
    assert why.startswith(("untestable", "no confident hit"))
    # and the two reasons never share a head word
    assert not (why.startswith("untestable") and "ambiguous" in why)
