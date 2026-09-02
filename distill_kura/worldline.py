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

Resident variants (the guide's §9) put the SAME cases in front of different maps:
`canonical` (the full index), `woven` (the production cloth, no model calls), and
any named text handed in from a file — so a map produced by a module that does not
exist yet can be measured the day it does. agent-only is the variant judge: the
other two routes let a rescue path hide a map that has grown too thin.

Traces are raw metrics only. There is deliberately no composite score: report the
counts and the costs and read them together, or the number starts optimizing for
itself. `explanation_burden` is left out on purpose — it needs a human in the
loop (M8) and a proxy would be measured instead of the thing.
"""
from __future__ import annotations

import hashlib
import json
import re
import time

from . import edges
from . import fastpath
from .recall import recall
from .store import Store
from .thinker import Endpoint
from .tokens import estimate

ROUTES = ("agent-only", "fastpath", "full")
BUILTIN_VARIANTS = ("canonical", "woven")

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


def seed(store: Store, path: str) -> list[str]:
    """Plant the benchmark's fixture memories (bench/worldline/memories.json) into
    a store. The shipped cases name these slugs, so a fresh store seeded from the
    file runs every case instead of skipping them all — and no case ever has to
    point at a house's private memories to be runnable."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    mems = data.get("memories") if isinstance(data, dict) else data
    if not isinstance(mems, list):
        raise ValueError(f"{path}: expected a JSON array of memories (or {{\"memories\": [...]}})")
    planted: list[str] = []
    for m in mems:
        r = store.remember_direct(m["slug"], m["description"], m.get("body", ""),
                                  type_=m.get("type", "project"), tags=m.get("tags"))
        if not r.get("ok", True):
            # A refused write is a store policy, not a fixture — say so, or the
            # whole run skips honestly for a reason nobody can see.
            raise RuntimeError(f"seeding {m['slug']!r} refused: {r.get('error')}")
        planted.append(m["slug"])
    return planted


def resident_variants(store: Store, names, files: dict[str, str] | None = None,
                      prefill_cfg: dict | None = None) -> dict[str, str]:
    """name → resident text, for every variant the caller asked for.

    `canonical` is the store's index; `woven` is what the loom would give right
    now WITHOUT a model (`generate=False`) — a benchmark must not spend GPU
    seconds, and a cloth that needed a model to exist is not the cloth in
    production anyway. Anything else must be in `files` (name → path): an unknown
    name is refused loudly, because a typo that fell back to canonical would
    print a perfectly healthy comparison of one map against itself."""
    files = dict(files or {})
    out: dict[str, str] = {}
    for name in names:
        if name in out:
            continue
        if name in files:
            with open(files[name], encoding="utf-8") as f:
                out[name] = f.read()
        elif name == "canonical":
            out[name] = store.index_text()
        elif name == "woven":
            from .prefill import loom_for
            out[name] = loom_for(store, prefill_cfg).weave(generate=False).text
        else:
            raise ValueError(f"unknown resident variant {name!r}: builtins are "
                             f"{BUILTIN_VARIANTS}, files are {sorted(files) or 'none'}")
    for name in files:
        if name not in out:
            out[name] = open(files[name], encoding="utf-8").read()
    return out


def _clean(name: str) -> str:
    n = str(name).strip().strip("[]()`\"' ")
    return n[:-3] if n.endswith(".md") else n


def _agent_answer(raw: str, store: Store) -> dict:
    """The conversation model's reply, read STRICTLY. The WHOLE reply must be a
    JSON array of strings — no prose around it, no array embedded in a sentence.
    This is a benchmark of the model's own ability, and following the output
    format it was given is part of that ability; a lenient parser here would
    rescue exactly the models agent-only exists to measure honestly. Only a
    cleaned name that is EXACTLY in slug_set() is `opened` — no resolve(), no
    fuzzy anything: the snap was a rescue for thinker picks, and it turned a
    misspelt slug into a hit while letting a hallucinated ["kyoto-house"]
    vanish and score as a correct refusal."""
    def bad() -> dict:
        return {"proposed_slugs": [], "invalid_slugs": [], "opened": [],
                "format_error": True}

    try:
        got = json.loads((raw or "").strip())
    except ValueError:
        return bad()
    if not isinstance(got, list) or not all(isinstance(x, str) for x in got):
        return bad()
    proposed: list[str] = []
    for x in got:
        c = _clean(x)
        if c and c not in proposed:
            proposed.append(c)
    known = store.slug_set()
    return {"proposed_slugs": proposed,
            "invalid_slugs": [n for n in proposed if n not in known],
            "opened": [n for n in proposed if n in known],
            "format_error": False}


def _map_sha(text: str) -> str:
    """Identity of the resident map that was actually worn — the name alone
    ("adaptive") cannot say WHICH trigger set a trace was measured against."""
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def run_case(store: Store, case: dict, routing: str, thinker: Endpoint | None = None,
             resident: str | None = None, fastpath_cfg: dict | None = None,
             hops: int = 1, agent: dict | None = None,
             use_cues: bool = True, resident_variant: str = "canonical") -> dict:
    """One utterance, one routing mode, one resident map, one honest trace row."""
    t0 = time.perf_counter()
    resident = store.index_text() if resident is None else resident
    tr = {"case": case.get("id", ""), "category": case.get("category", ""),
          "routing": routing, "resident_variant": resident_variant,
          "resident_sha": _map_sha(resident),
          "resident_tokens": estimate(resident),
          "first_tool": "", "opened": [], "related_reached": [],
          "thinker_calls": 0, "fastpath_used": False, "recall_context_tokens": 0,
          "target_reached": False, "wrong_branch": False, "obsolete_branch": False,
          "edge_says_obsolete": False,
          "remembered_but_unreachable": False, "unnecessary_opens": [],
          "skipped": None,
          "proposed_slugs": [], "invalid_slugs": [], "format_error": False,
          "reply_head": None, "reply_chars": 0,
          "cue_hit": None, "cue_ambiguous": False,
          "agent": agent if routing == "agent-only" else None}

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
                a = _agent_answer(raw, store)
                # What the model actually said, in part: a format_error is
                # otherwise a verdict with no witness (prose? a fenced array?
                # reasoning cut by the output cap?). The head is kept, not the
                # whole reply — a trace row is a measurement, not a transcript.
                tr["reply_head"] = raw[:240]
                tr["reply_chars"] = len(raw)
                tr["opened"] = a["opened"]
                tr["proposed_slugs"] = a["proposed_slugs"]
                tr["invalid_slugs"] = a["invalid_slugs"]
                tr["format_error"] = a["format_error"]
    elif routing == "fastpath":
        tr["first_tool"] = "fastpath"
        cfg = fastpath_cfg or {}
        fp = fastpath.lookup(store, case["utterance"], top=3,
                             gate=cfg.get("gate", fastpath.DEFAULT_GATE),
                             cues=use_cues)
        tr["fastpath_used"] = bool(fp["hits"])
        tr["opened"] = [h["slug"] for h in fp["hits"]]
        # cue_ambiguous stays False on purpose: the pre-head is silent on
        # ambiguity, indistinguishable from absence — that IS the honest reading.
        tr["cue_hit"] = fp.get("cue")
        # Deliberately no thinker here: tier zero's silence IS the measurement.
        # The rescue path belongs to `full`.
    elif routing == "full":
        cfg = dict(fastpath_cfg or {})   # copies: recall must not mutate the caller's table
        cfg["cues"] = use_cues
        d = recall(store, thinker, case["utterance"], hops=hops, fastpath_cfg=cfg)
        tr["opened"] = list(d.get("included") or [])
        how = d.get("how", "")
        tr["first_tool"] = how
        tr["fastpath_used"] = how == "fastpath"
        tr["thinker_calls"] = 1 if how in ("meaning", "meaning→none") else 0
        tr["recall_context_tokens"] = estimate(d.get("context", ""))
        tr["cue_hit"] = d.get("fastpath_cue")
    else:
        raise ValueError(f"routing must be one of {ROUTES}, got {routing!r}")

    if tr["skipped"] is None:
        if want:
            tr["target_reached"] = any(s in tr["opened"] for s in want)
        elif routing == "agent-only":
            # The unknown category, measured against the MODEL: only a valid empty
            # array is the honest "not remembered". A hallucinated slug is the
            # failure the prompt names, not a refusal — resolve() used to drop it
            # to nothing and pay the guess as a correct refusal.
            tr["target_reached"] = not tr["format_error"] and not tr["proposed_slugs"]
        else:
            # The unknown category: the honest answer is opening NOTHING. A store
            # that hands back look-alikes for a question it knows nothing about
            # scores worse than one that says "not remembered".
            tr["target_reached"] = not tr["opened"]
        related = [s for s in (case.get("acceptable_related") or []) if s]
        tr["related_reached"] = [s for s in related if s in tr["opened"]]
        # An obsolete memory is one the fixture MARKS obsolete: it was thrown away,
        # and landing on it resurrects a dead plan. Counted apart from an ordinary
        # wrong branch (a live neighbour), because the two fail differently and a
        # single count would let a rise in resurrections hide under a fall in
        # near-misses. The map is not consulted for the mark — a benchmark that
        # read tags to decide what counts would move with the store under test.
        obsolete = [s for s in (case.get("obsolete_slugs") or []) if s]
        tr["obsolete_branch"] = any(s in tr["opened"] for s in obsolete)
        # Does the derived edge map (M7) independently mark what the fixture calls
        # obsolete — a `supersedes` edge pointing at each obsolete slug? A raw
        # metric beside obsolete_branch, never a scoring input.
        supersedes_targets = {e["target"] for e in edges.current(store).get("edges", [])
                              if e["type"] == "supersedes"}
        tr["edge_says_obsolete"] = bool(obsolete) and all(
            s in supersedes_targets for s in obsolete)
        tr["wrong_branch"] = any(s in tr["opened"] and s not in obsolete
                                 for s in (case.get("must_not_anchor") or []))
        # "The memory exists, the door was too narrow": the target is IN the store
        # (the absent case skipped above) and the map-reading route did not reach
        # it. Only the routes that answer from the map count — in `full` the
        # thinker's rescue is exactly what hides a thin map.
        tr["remembered_but_unreachable"] = bool(
            want and routing in ("agent-only", "fastpath") and not tr["target_reached"])
        # Doors opened that were neither the target nor a neighbour the fixture
        # accepts: the cost side of a wider trigger. For the unknown category every
        # opening is unnecessary, which is the same statement as target_reached.
        tr["unnecessary_opens"] = [s for s in tr["opened"] if s not in want and s not in related]
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
            "obsolete_branch": sum(1 for t in ran if t.get("obsolete_branch")),
            "related_touched": sum(len(t["related_reached"]) for t in ran),
            # The unknown category answered from nothing. `honest_unknown` is the
            # guide's name (§9); the older key stays so existing readers keep working.
            "honest_unknown": sum(
                1 for t in ran if t["category"] == "unknown" and t["target_reached"]),
            "unknown_refused_correctly": sum(
                1 for t in ran if t["category"] == "unknown" and t["target_reached"]),
            "remembered_but_unreachable": sum(
                1 for t in ran if t.get("remembered_but_unreachable")),
            "unnecessary_opens": sum(len(t.get("unnecessary_opens") or []) for t in ran),
            "thinker_calls_total": sum(t["thinker_calls"] for t in ran),
            "fastpath_direct_total": sum(1 for t in ran if t["fastpath_used"]),
            "cue_direct_total": sum(1 for t in ran if t.get("cue_hit") is not None),
            "resident_tokens_mean": mean([t["resident_tokens"] for t in ran]),
            "opened_mean": mean([len(t["opened"]) for t in ran]),
            "elapsed_ms_mean": mean([t["elapsed_ms"] for t in ran])}


def run(store: Store, cases: list[dict], routing: str = "full",
        thinker: Endpoint | None = None, resident: str | None = None,
        fastpath_cfg: dict | None = None, hops: int = 1,
        trace_path: str | None = None, agent: dict | None = None,
        use_cues: bool = True,
        resident_variants: dict[str, str] | None = None) -> dict:
    """Every case under every resident variant, in one result.

    `resident_variants` (name → map text) is the comparison the guide's §9 asks
    for; without it the single `resident` runs as variant "canonical" (or
    "resident" when the caller handed in its own text, so a trace never claims
    a map it did not wear). The cases are the same objects for every variant —
    the point is that only the map changes."""
    if routing not in ROUTES:
        raise ValueError(f"routing must be one of {ROUTES}, got {routing!r}")
    if not resident_variants:
        resident_variants = {"canonical" if resident is None else "resident": resident}
    traces: list[dict] = []
    variants: dict[str, dict] = {}
    for name, text in resident_variants.items():
        rows = [run_case(store, c, routing, thinker=thinker, resident=text,
                         fastpath_cfg=fastpath_cfg, hops=hops, agent=agent,
                         use_cues=use_cues, resident_variant=name) for c in cases]
        variants[name] = {"resident_tokens": estimate(text or ""),
                          "resident_sha": _map_sha(text or ""),
                          "summary": summarize(rows)}
        traces.extend(rows)
    if trace_path:
        with open(trace_path, "a", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    result = {"store": store.name, "routing": routing, "cases": len(cases),
              "variants": variants,
              "summary": summarize(traces), "traces": traces}
    if routing == "agent-only":
        # Bookkeeping only: who was measured must be on record with the numbers.
        result["agent"] = agent
    return result
