"""The worldline benchmark runner — raw traces, honest skips, separated credit.

No model is needed anywhere in this file: agent-only is exercised through a stub
endpoint object, because what is under test is the RUNNER's bookkeeping (which
calls were made, what counted as opened, what is an outage versus an answer), not
any model's cleverness.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.store import Store                       # noqa: E402
from distill_kura.thinker import Models                    # noqa: E402
from distill_kura.registry import Registry                 # noqa: E402
from distill_kura.distill.pipeline import Distiller        # noqa: E402
from distill_kura import worldline as wl                   # noqa: E402


class StubModel:
    """Duck-types Endpoint.ask: the runner may not care which model it got."""
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def ask(self, system, user, max_tokens=400, timeout=None, temperature=None):
        self.calls += 1
        return self.reply


def build(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m"), label="k")
    s.init_files()
    s.remember_direct("freetoken-hybrid", "FreeToken CPU hybrid — the all-hands "
                      "CPU/GPU cooperative inference push", "body", tags=["decision"])
    s.remember_direct("old-abandoned-plan", "the early sqlite-first draft, "
                      "abandoned before the store existed", "body")
    s.remember_direct("weave-revision", "the three-point freshness proof: source, "
                      "product, revision", "body")
    s.remember_direct("gpu-power-measure", "measured the board's idle draw",
                      "tool said 311 W")
    return s


def reg_of(store):
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    return Registry(stores={"m": store}, modes={}, models=models, default="m", raw={})


def _cue_manifest(tmp_path, store, cue_text="全員野球", quote="例の全員野球でいこう"):
    """One verified cue on freetoken-hybrid, written straight to the evidence dir —
    the test_callsigns pattern: provenance only, no model."""
    src = tmp_path / "journal.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    return Distiller(reg_of(store), store)._write_manifest(
        {"slug": "freetoken-hybrid", "kind": "project",
         "evidence": [{"class": "USER", "text": quote}], "classes": ["USER"],
         "routing_cues": [{"text": cue_text, "class": "USER", "quote": quote}]},
        str(src), "test:cue")


CASES = [
    {"id": "direct", "utterance": "the freetoken-hybrid experiment, continued",
     "target_slugs": ["freetoken-hybrid"], "acceptable_related": [],
     "must_not_anchor": [], "category": "direct-name"},
    {"id": "trap", "utterance": "the old-abandoned-plan — is that still the idea?",
     "target_slugs": ["freetoken-hybrid"], "acceptable_related": [],
     "must_not_anchor": ["old-abandoned-plan"], "category": "superseded-plan"},
    {"id": "unknown", "utterance": "そちらの京都の家の改修の件はどうでしょう",
     "target_slugs": [], "acceptable_related": [], "must_not_anchor": [],
     "category": "unknown"},
    {"id": "other-house", "utterance": "something this store has never seen",
     "target_slugs": ["no-such-memory-anywhere"], "acceptable_related": [],
     "must_not_anchor": [], "category": "direct-name"},
]


# ── agent-only: the model's own recognition, and only that ──────────────────

def test_agent_only_uses_the_model_and_counts_no_helper_calls(tmp_path):
    s = build(tmp_path)
    stub = StubModel('["freetoken-hybrid"]')
    tr = wl.run_case(s, CASES[0], "agent-only", thinker=stub)
    assert stub.calls == 1, "the model saw the utterance exactly once"
    assert tr["first_tool"] == "model" and tr["opened"] == ["freetoken-hybrid"]
    assert tr["thinker_calls"] == 0 and tr["fastpath_used"] is False
    assert tr["target_reached"] is True


def test_agent_only_never_calls_fastpath_or_recall(tmp_path):
    """Routing separation is the point of the mode: silence or hits must come from
    the model alone. The runner holds no reference it could accidentally use."""
    s = build(tmp_path)
    stub = StubModel("[]")
    tr = wl.run_case(s, CASES[2], "agent-only", thinker=stub)
    assert tr["opened"] == [] and tr["target_reached"] is True, \
        "an empty answer to an unknown question is the honest answer"
    assert tr["fastpath_used"] is False


def test_agent_only_invented_slugs_are_not_landings(tmp_path):
    """Invented slugs no longer silently vanish: the real memory counts as opened,
    the invention is recorded as invalid, and it must not sink the case."""
    s = build(tmp_path)
    stub = StubModel('["freetoken-hybrid", "a-slug-that-does-not-exist"]')
    tr = wl.run_case(s, CASES[0], "agent-only", thinker=stub)
    assert tr["opened"] == ["freetoken-hybrid"], "only real memories count as opened"
    assert tr["invalid_slugs"] == ["a-slug-that-does-not-exist"]
    assert tr["proposed_slugs"] == ["freetoken-hybrid", "a-slug-that-does-not-exist"]
    assert tr["format_error"] is False and tr["target_reached"] is True


def test_agent_only_near_miss_is_not_snapped(tmp_path):
    """A misspelt slug stays a miss. resolve()'s fuzzy snap is a rescue for
    thinker picks; agent-only measures the model's EXACT recognition, so
    'freetoken-hyrbid' must not become freetoken-hybrid."""
    s = build(tmp_path)
    stub = StubModel('["freetoken-hyrbid"]')
    tr = wl.run_case(s, CASES[0], "agent-only", thinker=stub)
    assert tr["opened"] == []
    assert tr["invalid_slugs"] == ["freetoken-hyrbid"]
    assert tr["format_error"] is False and tr["target_reached"] is False


def test_agent_only_hallucination_on_unknown_is_not_a_refusal(tmp_path):
    """A confident guess on the unknown case is the one failure the prompt names;
    it must score as a miss, not vanish into a 'correct' refusal."""
    s = build(tmp_path)
    stub = StubModel('["kyoto-house"]')
    tr = wl.run_case(s, CASES[2], "agent-only", thinker=stub)
    assert tr["target_reached"] is False
    assert tr["invalid_slugs"] == ["kyoto-house"]
    assert tr["format_error"] is False


def test_agent_only_valid_empty_array_on_unknown_is_a_refusal(tmp_path):
    """'[]' is the only honest shape for an unknown: no format error, no proposal,
    so it is a correct refusal — not a degraded non-answer."""
    s = build(tmp_path)
    stub = StubModel("[]")
    tr = wl.run_case(s, CASES[2], "agent-only", thinker=stub)
    assert tr["format_error"] is False
    assert tr["proposed_slugs"] == [] and tr["target_reached"] is True


def test_agent_only_shapeless_reply_is_a_format_error(tmp_path):
    """No JSON array means no answer at all: recorded as a format error, never
    scored as a refusal, and never a crash."""
    s = build(tmp_path)
    stub = StubModel("I don't remember anything matching that.")
    tr = wl.run_case(s, CASES[2], "agent-only", thinker=stub)
    assert tr["format_error"] is True
    assert tr["proposed_slugs"] == [] and tr["invalid_slugs"] == []
    assert tr["target_reached"] is False


def test_agent_only_unreachable_model_is_an_outage_not_an_answer(tmp_path):
    """ask() → None means the endpoint was down; recording it as `target_reached`
    would pay the dead endpoint for silence."""
    s = build(tmp_path)
    stub = StubModel(None)
    tr = wl.run_case(s, CASES[2], "agent-only", thinker=stub)
    assert tr["skipped"] == "model unreachable" and tr["target_reached"] is False


def test_agent_only_without_an_endpoint_skips_honestly(tmp_path):
    tr = wl.run_case(build(tmp_path), CASES[0], "agent-only", thinker=None)
    assert tr["skipped"] and "endpoint" in tr["skipped"]


# ── fastpath: silence is silence, no thinker rescue ─────────────────────────

def test_fastpath_direct_name_hits_without_any_model(tmp_path):
    s = build(tmp_path)
    tr = wl.run_case(s, CASES[0], "fastpath")
    assert tr["first_tool"] == "fastpath" and tr["thinker_calls"] == 0
    assert tr["opened"] == ["freetoken-hybrid"] and tr["target_reached"] is True


def test_fastpath_silence_is_recorded_not_rescued(tmp_path):
    s = build(tmp_path)
    tr = wl.run_case(s, CASES[2], "fastpath")   # the unknown utterance: no name to cite
    assert tr["fastpath_used"] is False and tr["thinker_calls"] == 0, \
        "the thinker rescue belongs to `full`; calling it here would blur the modes"
    assert tr["opened"] == [] and tr["target_reached"] is True


def test_fastpath_lands_the_wrong_branch_honestly_when_baited(tmp_path):
    """The trap case NAMES the abandoned plan, so tier zero rightly hits it — and
    the trace must record that as a wrong branch, not hide it behind silence."""
    s = build(tmp_path)
    tr = wl.run_case(s, CASES[1], "fastpath")
    assert tr["fastpath_used"] is True and tr["opened"] == ["old-abandoned-plan"]
    assert tr["wrong_branch"] is True and tr["target_reached"] is False


def test_prose_around_the_array_is_a_format_error_not_a_hit(tmp_path):
    """The prompt says "Output ONLY a JSON array". A model that wraps the array
    in chatter was rescued by the embedded-array parser; format compliance is
    part of what agent-only measures."""
    s = build(tmp_path)
    stub = StubModel('I think it\'s this:\n["freetoken-hybrid"]')
    tr = wl.run_case(s, CASES[0], "agent-only", thinker=stub)
    assert tr["format_error"] is True
    assert tr["opened"] == [] and tr["target_reached"] is False


def test_agent_url_cannot_hijack_the_full_routing_thinker(tmp_path):
    """One flag quietly swapping the production path's brain would end the
    comparability the three modes exist for: full ALWAYS runs the configured
    thinker; --agent-url measures agent-only and nothing else."""
    from distill_kura import bench
    s = build(tmp_path)
    reg = reg_of(s)
    cfg = tmp_path / "cases.json"
    cfg.write_text(json.dumps({"cases": CASES}), encoding="utf-8")
    with pytest.raises(ValueError, match="agent-only"):
        bench.worldline(reg, s, str(cfg), routing="full",
                        agent_url="http://elsewhere:1/v1",
                        trace_path=str(tmp_path / "t.jsonl"))


# ── full: the production path, accounted ────────────────────────────────────

def test_full_uses_recall_and_counts_the_thinker_honestly(tmp_path):
    s = build(tmp_path)
    stub = StubModel('["freetoken-hybrid"]')
    tr = wl.run_case(s, CASES[0], "full", thinker=stub)
    assert tr["opened"], "the recall path returned something"
    assert tr["first_tool"], tr
    assert tr["thinker_calls"] in (0, 1) and tr["recall_context_tokens"] >= 0


def test_full_without_a_model_degrades_and_does_not_crash(tmp_path):
    tr = wl.run_case(build(tmp_path), CASES[0], "full", thinker=None)
    assert tr["skipped"] is None and "how" not in tr


# ── skips, traces, summary ──────────────────────────────────────────────────

def test_a_case_written_for_another_house_is_skipped_not_scored(tmp_path):
    tr = wl.run_case(build(tmp_path), CASES[3], "fastpath")
    assert tr["skipped"] == "target slugs absent from this store"
    assert tr["target_reached"] is False and tr["wrong_branch"] is False


def test_wrong_branch_is_flagged(tmp_path):
    s = build(tmp_path)
    stub = StubModel('["old-abandoned-plan"]')
    tr = wl.run_case(s, CASES[1], "agent-only", thinker=stub)
    assert tr["wrong_branch"] is True and tr["target_reached"] is False


def test_run_writes_the_trace_file_and_a_raw_summary(tmp_path):
    s = build(tmp_path)
    out = wl.run(s, CASES, "fastpath",
                 trace_path=str(tmp_path / "traces.jsonl"))
    lines = [json.loads(l) for l in
             open(tmp_path / "traces.jsonl", encoding="utf-8") if l.strip()]
    assert len(lines) == len(CASES)
    assert {t["case"] for t in lines} == {c["id"] for c in CASES}
    sm = out["summary"]
    assert sm["cases_total"] == len(CASES) and sm["skipped"] == 1
    assert "score" not in sm, "no composite score exists yet, on purpose"


def test_the_unknown_category_is_refusal_not_failure(tmp_path):
    s = build(tmp_path)
    stub = StubModel("[]")
    tr = wl.run_case(s, CASES[2], "agent-only", thinker=stub)
    assert tr["category"] == "unknown" and tr["target_reached"] is True


def test_bench_worldline_through_the_registry(tmp_path):
    from distill_kura import bench
    s = build(tmp_path)
    reg = reg_of(s)
    cfg = tmp_path / "cases.json"
    cfg.write_text(json.dumps({"cases": CASES}), encoding="utf-8")
    r = bench.worldline(reg, s, str(cfg), routing="fastpath",
                        trace_path=str(tmp_path / "t.jsonl"))
    assert r["routing"] == "fastpath" and len(r["traces"]) == len(CASES)
    assert r["traces"][0]["resident_tokens"] > 0, \
        "the resident map (woven or canonical) is what the agent wears"


def test_run_stamps_the_agent_identity_on_every_trace(tmp_path):
    s = build(tmp_path)
    agent = {"url": "http://x/v1", "model": "big"}
    stub = StubModel("[]")
    out = wl.run(s, CASES, "agent-only", thinker=stub, agent=agent)
    assert out["agent"] == agent
    assert all(t["agent"] == agent for t in out["traces"]), \
        "who was measured must be on record with every trace, skipped or not"


def test_bench_worldline_default_records_the_configured_thinker(tmp_path):
    """Without --agent-url the configured thinker plays the agent, and the trace
    must say so — an unrecorded substitute model is an unmeasurable result."""
    from distill_kura import bench
    s = build(tmp_path)
    reg = reg_of(s)
    cfg = tmp_path / "cases.json"
    cfg.write_text(json.dumps({"cases": CASES}), encoding="utf-8")
    r = bench.worldline(reg, s, str(cfg), routing="agent-only",
                        trace_path=str(tmp_path / "t.jsonl"))
    assert r["agent"] == {"url": "http://127.0.0.1:9/v1", "model": "none"}
    assert all(t["agent"] == r["agent"] for t in r["traces"])
    assert "model unreachable" in {t["skipped"] for t in r["traces"]}, \
        "the dead configured endpoint is an honest skip, not an answer"


def test_bench_worldline_agent_url_records_that_identity(tmp_path):
    """--agent-url names the measured model; the recorded identity must match the
    endpoint actually asked, even when that endpoint is unreachable."""
    from distill_kura import bench
    s = build(tmp_path)
    reg = reg_of(s)
    cfg = tmp_path / "cases.json"
    cfg.write_text(json.dumps({"cases": CASES}), encoding="utf-8")
    r = bench.worldline(reg, s, str(cfg), routing="agent-only",
                        agent_url="http://127.0.0.1:9/v1", agent_model="probe",
                        trace_path=str(tmp_path / "t.jsonl"))
    assert r["agent"] == {"url": "http://127.0.0.1:9/v1", "model": "probe"}
    assert all(t["agent"] == r["agent"] for t in r["traces"])
    assert "model unreachable" in {t["skipped"] for t in r["traces"]}


# ── cues: the callsign pre-head, threaded through the runner ───────────────

CUE_CASE = {"id": "cue", "utterance": "あの全員野球の続きなんだけど",
            "target_slugs": ["freetoken-hybrid"], "acceptable_related": [],
            "must_not_anchor": [], "category": "shared-callsign"}


def test_fastpath_with_cues_routes_via_the_callsign_pre_head(tmp_path):
    s = build(tmp_path)
    _cue_manifest(tmp_path, s)
    tr = wl.run_case(s, CUE_CASE, "fastpath", use_cues=True)
    assert tr["cue_hit"] == "全員野球" and tr["cue_ambiguous"] is False
    assert tr["opened"] == ["freetoken-hybrid"] and tr["target_reached"] is True
    assert wl.summarize([tr])["cue_direct_total"] >= 1


def test_fastpath_without_cues_is_silent_on_the_shared_word(tmp_path):
    """cues off = tier zero of before: the five heads alone stay silent on a
    callsign, and cue_hit must be None, not a guess."""
    s = build(tmp_path)
    _cue_manifest(tmp_path, s)
    tr = wl.run_case(s, CUE_CASE, "fastpath", use_cues=False)
    assert tr["cue_hit"] is None and tr["cue_ambiguous"] is False
    assert tr["opened"] == [] and tr["fastpath_used"] is False
    assert tr["target_reached"] is False, "the callsign case's target was not reached"


def test_full_with_a_dead_thinker_and_cues_threaded_does_not_crash(tmp_path):
    s = build(tmp_path)
    _cue_manifest(tmp_path, s)
    tr = wl.run_case(s, CUE_CASE, "full", thinker=None, use_cues=True)
    assert tr["skipped"] is None and "cue_hit" in tr
    assert tr["cue_hit"] == "全員野球", \
        "the cue answers before the thinker is even asked — a dead thinker is fine"
    assert tr["opened"] == ["freetoken-hybrid"] and tr["target_reached"] is True


def test_bench_worldline_no_cues_counts_no_cue_direct_hits(tmp_path):
    from distill_kura import bench
    s = build(tmp_path)
    _cue_manifest(tmp_path, s)
    reg = reg_of(s)
    cfg = tmp_path / "cases.json"
    cfg.write_text(json.dumps({"cases": [CUE_CASE] + CASES}), encoding="utf-8")
    r = bench.worldline(reg, s, str(cfg), routing="fastpath",
                        trace_path=str(tmp_path / "t.jsonl"), use_cues=False)
    assert r["summary"]["cue_direct_total"] == 0
    assert all(t["cue_hit"] is None for t in r["traces"])
