"""Worldline recovery — the benchmark that asks what the map exists for.

Retrieval precision asks "did the right file come back". This asks a harder thing:
the person opens a session with a breadcrumb — a shared callsign, an ellipsis, a
plan they abandoned months ago — and the question is whether the agent returns to
the right shared world, refuses honestly when nothing is remembered, and does not
resurrect the abandoned plan as if it were current.

Three routing modes keep the credit honest (the plan's §1.3: never hide the model's
own cognition behind a retriever):

    agent-only   the conversation model reads the resident map and names slugs
                 itself. No kura tools, no fastpath, no thinker. What this
                 measures is the MODEL's recognition.
    fastpath     tier zero only. Its silence is recorded as silence — the thinker
                 rescue is a different mode's measurement.
    full         the production recall path.

Traces are raw metrics only. There is deliberately no composite score: report the
counts and the costs and read them together, or the number starts optimizing for
itself.
"""
from __future__ import annotations

import json
import re
import time

from . import fastpath
from .recall import recall
from .store import Store
from .thinker import Endpoint
from .tokens import estimate

ROUTES = ("agent-only", "fastpath", "full")

# The conversation model, playing itself: it sees exactly what a real agent wears
# (the resident map) and answers with slugs. Nothing about kura's retrieval stack
# appears here — that separation is the point of agent-only.
AGENT_SYS = (
    "You are an agent at the start of a session. Below is the RESIDENT MEMORY MAP "
    "of this household/project — one recognition line per memory. The user opens "
    "with a single utterance; it may be vague, a shared nickname, or the continuation "
    "of past work.\n\n"
    "Name the memories this utterance refers to or continues. Judge by meaning, not "
    "by shared words. If the map holds nothing that truly matches, answer an empty "
    "array — \"not remembered\" is a correct answer, and a confident guess is the "
    "one failure this exists to catch.\n\n"
    "Output ONLY a JSON array of exact slugs from the map (no .md, no path), empty "
    "if nothing matches.\n\n"
    "=== RESIDENT MAP ===\n"
)


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list):
        raise ValueError(f"{path}: expected a JSON array of cases (or {{\"cases\": [...]}})")
    return cases


def _clean(name: str) -> str:
    n = str(name).strip().strip("[]()`\"' ")
    return n[:-3] if n.endswith(".md") else n


def _answer_slugs(raw: str, store: Store) -> list[str]:
    """The model's answer, snapped to real memories. Whatever the format, only a
    name that lands inside this store counts as `opened` — an invented slug is
    not a landing, it is nothing (and `resolve()` cannot leave the store)."""
    names: list[str] | None = None
    m = re.search(r"\[.*?\]", raw, re.S)
    if m:
        try:
            got = json.loads(m.group(0))
            names = [_clean(x) for x in got if isinstance(x, str)]
        except ValueError:
            names = None
    if not names:
        names = [_clean(x) for x in (raw or "").splitlines() if x.strip()]
    out: list[str] = []
    for n in names:
        s = store.resolve(n)
        if s and s not in out:
            out.append(s)
    return out


def run_case(store: Store, case: dict, routing: str, thinker: Endpoint | None = None,
             resident: str | None = None, fastpath_cfg: dict | None = None,
             hops: int = 1) -> dict:
    """One utterance, one routing mode, one honest trace row."""
    t0 = time.perf_counter()
    resident = store.index_text() if resident is None else resident
    tr = {"case": case.get("id", ""), "category": case.get("category", ""),
          "routing": routing, "resident_tokens": estimate(resident),
          "first_tool": "", "opened": [], "related_reached": [],
          "thinker_calls": 0, "fastpath_used": False, "recall_context_tokens": 0,
          "target_reached": False, "wrong_branch": False, "skipped": None}

    known = store.slug_set()
    want = [s for s in (case.get("target_slugs") or []) if s]
    if want and not any(s in known for s in want):
        # A case written against another house is not a measurement here. Skip it
        # with the reason; never score what could not run.
        tr["skipped"] = "target slugs absent from this store"
        tr["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return tr

    if routing == "agent-only":
        if thinker is None:
            tr["skipped"] = "agent-only needs a model endpoint (the conversation model)"
        else:
            tr["first_tool"] = "model"          # the agent's own cognition, not a helper
            raw = thinker.ask(AGENT_SYS + resident, case["utterance"], max_tokens=300)
            if raw is None:
                # An unreachable model is an outage, not an honest "nothing matched";
                # recording it as target_reached would reward the dead endpoint.
                tr["skipped"] = "model unreachable"
            else:
                tr["opened"] = _answer_slugs(raw, store)
    elif routing == "fastpath":
        tr["first_tool"] = "fastpath"
        cfg = fastpath_cfg or {}
        fp = fastpath.lookup(store, case["utterance"], top=3,
                             gate=cfg.get("gate", fastpath.DEFAULT_GATE))
        tr["fastpath_used"] = bool(fp["hits"])
        tr["opened"] = [h["slug"] for h in fp["hits"]]
        # Deliberately no thinker here: tier zero's silence IS the measurement.
        # The rescue path belongs to `full`.
    elif routing == "full":
        d = recall(store, thinker, case["utterance"], hops=hops, fastpath_cfg=fastpath_cfg)
        tr["opened"] = list(d.get("included") or [])
        how = d.get("how", "")
        tr["first_tool"] = how
        tr["fastpath_used"] = how == "fastpath"
        tr["thinker_calls"] = 1 if how in ("meaning", "meaning→none") else 0
        tr["recall_context_tokens"] = estimate(d.get("context", ""))
    else:
        raise ValueError(f"routing must be one of {ROUTES}, got {routing!r}")

    if tr["skipped"] is None:
        if want:
            tr["target_reached"] = any(s in tr["opened"] for s in want)
        else:
            # The unknown category: the honest answer is opening NOTHING. A store
            # that hands back look-alikes for a question it knows nothing about
            # scores worse than one that says "not remembered".
            tr["target_reached"] = not tr["opened"]
        tr["related_reached"] = [s for s in (case.get("acceptable_related") or [])
                                 if s in tr["opened"]]
        tr["wrong_branch"] = any(s in tr["opened"] for s in (case.get("must_not_anchor") or []))
    tr["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return tr


def summarize(traces: list[dict]) -> dict:
    """Raw aggregates over the cases that actually ran. No composite score — the
    day one number exists, someone optimizes it (the plan's §3.2)."""
    ran = [t for t in traces if not t.get("skipped")]
    n = len(ran)

    def mean(xs: list) -> float:
        return round(sum(xs) / len(xs), 1) if xs else 0

    return {"cases_total": len(traces), "runnable": n,
            "skipped": len(traces) - n,
            "target_reached": sum(1 for t in ran if t["target_reached"]),
            "wrong_branch": sum(1 for t in ran if t["wrong_branch"]),
            "related_touched": sum(len(t["related_reached"]) for t in ran),
            "unknown_refused_correctly": sum(
                1 for t in ran if t["category"] == "unknown" and t["target_reached"]),
            "thinker_calls_total": sum(t["thinker_calls"] for t in ran),
            "fastpath_direct_total": sum(1 for t in ran if t["fastpath_used"]),
            "resident_tokens_mean": mean([t["resident_tokens"] for t in ran]),
            "opened_mean": mean([len(t["opened"]) for t in ran]),
            "elapsed_ms_mean": mean([t["elapsed_ms"] for t in ran])}


def run(store: Store, cases: list[dict], routing: str = "full",
        thinker: Endpoint | None = None, resident: str | None = None,
        fastpath_cfg: dict | None = None, hops: int = 1,
        trace_path: str | None = None) -> dict:
    if routing not in ROUTES:
        raise ValueError(f"routing must be one of {ROUTES}, got {routing!r}")
    traces = [run_case(store, c, routing, thinker=thinker, resident=resident,
                       fastpath_cfg=fastpath_cfg, hops=hops) for c in cases]
    if trace_path:
        with open(trace_path, "a", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return {"store": store.name, "routing": routing, "cases": len(traces),
            "summary": summarize(traces), "traces": traces}
