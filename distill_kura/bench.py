"""Measure what the distiller actually costs and keeps — with a real tokenizer.

Two ratios get confused with each other, and neither one has a number until someone
counts. Naming them is half the work:

    store_ratio  = tokens in the canonical memories and index
                   / tokens of raw journal consumed to produce them
                 → "how much smaller is the memory than the conversation"

    map_ratio    = tokens in the woven resident map
                   / tokens in the canonical index
                 → "how much cheaper is the map than the index it came from"

    window_share = tokens in the resident map / the model's context window
                 → the "spring source" condition, and the only one the code enforces

They answer different questions and are routinely quoted as one number. A low
`store_ratio` is also not automatically good: past a point it stops being compression
and becomes omission, which is what the retention side has to measure.

The core stays dependency-free, so exact counting is delegated to a command you name:

    kura bench compress --tokenizer-command "./count-tokens"

which must read text on stdin and print an integer. Without one, the built-in
estimator is used and every figure is labelled `estimated`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess

from .distill import Distiller
from .registry import Registry
from .store import Store
from .tokens import estimate


def worldline(reg: Registry, store: Store, cases_path: str, routing: str = "full",
              hops: int = 1, trace_path: str | None = None,
              agent_url: str | None = None, agent_model: str | None = None,
              use_cues: bool = True) -> dict:
    """Can the house return to the right shared world from a small breadcrumb?

    Raw traces, no composite score — see `distill_kura/worldline.py` and
    `bench/worldline/README.md`. The resident context is what the agent actually
    wears: the woven cloth when current, the canonical index otherwise.

    `agent_url` names the conversation model that plays the agent in agent-only
    mode (agent-only measures the MODEL's recognition, and the recorded identity
    must always be the endpoint actually asked). Without it the configured
    thinker plays the agent and is recorded as such. It is refused for any other
    routing: `full` always runs the CONFIGURED thinker, or the modes stop being
    comparable — one flag would quietly swap the production path's brain.
    """
    from . import worldline as wl
    from .prefill import build, loom_for, trail_for
    from .thinker import Endpoint
    loom = loom_for(store, reg.prefill_cfg_for(store))
    pf = build(store, loom, trail=trail_for(store, reg.prefill_cfg_for(store), loom=loom))
    thinker = reg.models_for(store).thinker
    identity = None
    if routing == "agent-only":
        if agent_url:
            thinker = Endpoint(url=agent_url, model=agent_model or "agent")
        identity = {"url": thinker.url, "model": thinker.model}
    elif agent_url or agent_model:
        raise ValueError("--agent-url/--agent-model measure agent-only routing; "
                         f"--routing {routing!r} always uses the configured thinker")
    return wl.run(store, wl.load_cases(cases_path), routing=routing,
                  thinker=thinker, resident=pf.text,
                  fastpath_cfg=reg.fastpath_cfg_for(store), hops=hops,
                  trace_path=trace_path
                  or os.path.join(store.still, "worldline-traces.jsonl"),
                  agent=identity, use_cues=use_cues)


def counter(command: str | None):
    """A token counter: the named command, or the built-in estimate."""
    if not command:
        return estimate, "estimated"

    def count(text: str) -> int:
        p = subprocess.run(command, shell=True, input=text, capture_output=True,
                           text=True, timeout=300)
        try:
            return int(p.stdout.strip().split()[-1])
        except (ValueError, IndexError):
            raise RuntimeError(f"tokenizer command printed no integer: {p.stdout[:200]!r}")
    return count, f"exact:{command}"


def store_tokens(store: Store, count) -> dict:
    bodies = "\n".join(store.read_exact(s) for s in store.slugs())
    idx = store.index_text()
    return {"memories": len(store.slugs()),
            "body_tokens": count(bodies),
            "index_tokens": count(idx)}


def retention(reg: Registry, store: Store, questions_path: str,
              hops: int = 1, top: int = 3) -> dict:
    """Did the memory keep what mattered, or only what was easy?

    `store_ratio` answers "how much smaller"; nothing yet answers "and what was lost".
    A store that keeps one memory out of a hundred has a spectacular ratio and may be
    useless, so the two numbers have to be read together or neither means anything.

    The score is deliberately MODEL-FREE: each planted fact carries a marker that must
    appear in what recall returns. That measures whether the fact is *findable*, which
    is the coverage question. It does not measure whether the answer reads well — that
    needs a judge, and a judge needs a model, and then the benchmark stops being
    reproducible on someone else's machine.

    Distractors invert the test: a fact marked `must_not_store` is a point LOST if the
    store kept it. A memory system is judged by what it declines as much as by what it
    keeps.
    """
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    thinker = reg.models_for(store).thinker
    from .recall import recall as do_recall

    rows, by_cat = [], {}
    for q in questions:
        try:
            d = do_recall(store, thinker, q["question"], hops=hops, top=top)
            found = bool(re.search(q["expect"], d.get("context", ""), re.I))
            wanted = not q.get("must_not_store")
            ok = found if wanted else not found
            rows.append({"id": q["id"], "category": q["category"], "found": found,
                         "expected_to_be_found": wanted, "pass": ok,
                         "walked": d.get("walked", []), "how": d.get("how")})
            c = by_cat.setdefault(q["category"], {"pass": 0, "total": 0})
            c["total"] += 1
            c["pass"] += 1 if ok else 0
        except (re.error, KeyError) as e:
            # One malformed question must abort the run naming the offender, not be
            # skipped silently — a benchmark that quietly drops a question overstates
            # the store.
            raise RuntimeError(
                f"retention: question {q.get('id')!r} is malformed "
                f"({type(e).__name__}: {e})") from e

    passed = sum(1 for r in rows if r["pass"])
    return {
        "store": store.name,
        "questions": len(rows),
        "passed": passed,
        "score": round(passed / max(1, len(rows)), 3),
        "by_category": {k: {**v, "score": round(v["pass"] / v["total"], 3)}
                        for k, v in sorted(by_cat.items())},
        "failures": [r for r in rows if not r["pass"]],
        "rows": rows,
        "note": ("A marker match means the fact is FINDABLE, not that the answer is good. "
                 "Distractors invert: keeping them costs a point."),
    }


def compress(reg: Registry, store: Store, tokenizer_command: str | None = None,
             session: str | None = None) -> dict:
    """What this store cost, against the journal it was made from.

    Reads the distiller's own metric log, so the raw side is what was actually consumed
    rather than what happens to be lying in the journal directory today.
    """
    count, how = counter(tokenizer_command)
    dis = Distiller(reg, store)
    metrics_path = os.path.join(store.still, "metrics.jsonl")
    raw_tokens = batches = 0
    malformed_metric_lines = 0
    recorded_keys: set[str] = set()
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if session and session not in str(row.get("source_key", "")):
                        continue
                    # raw_tokens_est can be a JSON type int() refuses (a list, a dict)
                    # — one such line used to crash the whole benchmark.
                    raw_tokens += int(row.get("raw_tokens_est") or 0)
                    recorded_keys.add(str(row.get("source_key", "")))
                    batches += 1
                except (ValueError, TypeError, AttributeError):
                    malformed_metric_lines += 1

    st = store_tokens(store, count)
    # The numerator must be ONLY what came out of the recorded batches. Dividing the
    # whole store by the raw material of a few batches gave 6.3 on a store that predated
    # its metrics — a number in the wrong direction by an order of magnitude. A memory
    # is attributed to a batch through its evidence manifest, which names the source.
    scoped_bodies = []
    unattributed = 0
    ev_dir = os.path.join(store.path, "_evidence")
    for slug in store.slugs():
        ref = store.frontmatter(slug).get("evidence_manifest", "")
        digest = ref.split("sha256:", 1)[1] if "sha256:" in ref else ""
        mpath = os.path.join(ev_dir, f"{digest}.json") if digest else ""
        if not (mpath and os.path.exists(mpath)):
            unattributed += 1
            continue
        try:
            with open(mpath, encoding="utf-8") as f:
                src_key = json.load(f).get("source_key", "")
        except (OSError, ValueError):
            unattributed += 1
            continue
        if src_key in recorded_keys:
            scoped_bodies.append(store.read_exact(slug))
    scoped_tokens = count("\n".join(scoped_bodies)) if scoped_bodies else 0
    cloth = None
    from .prefill import loom_for
    loom = loom_for(store, reg.prefill_cfg_for(store))
    text = loom.cloth_on_disk()
    if text is not None:
        cloth = count(text)

    out = {
        "store": store.name,
        "counted_by": how,
        # The raw side is always the distiller's own estimate at drink time: the journal
        # is not re-read here. An exact tokenizer therefore counts the canonical side
        # exactly and divides it by an estimate. Say so, rather than print a ratio that
        # looks more precise than its worse half.
        "raw_counted_by": "estimated (at drink time)",
        "batches_recorded": batches,
        "raw_tokens_consumed": raw_tokens or None,
        "memories_from_recorded_batches": len(scoped_bodies),
        "memories_unattributed": unattributed,
        "canonical_tokens_from_recorded_batches": scoped_tokens or None,
        "store_ratio": (round(scoped_tokens / raw_tokens, 4)
                        if raw_tokens and scoped_bodies else None),
        "store_ratio_units": ("mixed: canonical exact / raw estimated"
                              if tokenizer_command else "estimated / estimated"),
        "whole_store": {"memories": st["memories"],
                        "body_tokens": st["body_tokens"],
                        "index_tokens": st["index_tokens"]},
        "map_tokens": cloth,
        "map_ratio": (round(cloth / st["index_tokens"], 4)
                      if cloth and st["index_tokens"] else None),
        "journals": sorted(dis.journals),
    }
    if raw_tokens and not scoped_bodies:
        out["note"] = ("batches were recorded but no memory carries an evidence manifest "
                       "pointing at them, so store_ratio is undefined. Memories poured "
                       "before manifests existed cannot be attributed to a batch.")
    if malformed_metric_lines:
        # The count only lands in the report when it is non-zero: a healthy log does
        # not deserve a field everyone has to re-read.
        out["malformed_metric_lines"] = malformed_metric_lines
    if not raw_tokens:
        out["note"] = ("no distiller metrics yet, so store_ratio cannot be computed. "
                       "Run `kura distill run` (metrics land in _still/metrics.jsonl); "
                       "a ratio against journals this store never drank would be a "
                       "number with no meaning.")
    if how.startswith("estimated"):
        out["warning"] = ("counted with the built-in estimator, which is within a few "
                          "percent of the tokenizers it was fitted against and up to "
                          "~20% out against others. Pass --tokenizer-command before "
                          "quoting these figures anywhere.")
    return out
