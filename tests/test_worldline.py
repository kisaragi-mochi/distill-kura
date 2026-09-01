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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.store import Store                       # noqa: E402
from distill_kura.thinker import Models                    # noqa: E402
from distill_kura.registry import Registry                 # noqa: E402
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
    s = build(tmp_path)
    stub = StubModel('["freetoken-hybrid", "a-slug-that-does-not-exist"]')
    tr = wl.run_case(s, CASES[0], "agent-only", thinker=stub)
    assert tr["opened"] == ["freetoken-hybrid"], "only real memories count as opened"


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
