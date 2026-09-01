# Worldline benchmark

> Can the house return to the right shared world from the smallest breadcrumb?

This is NOT a retrieval-precision benchmark. Each case is an opening utterance —
often vague, sometimes a shared callsign, sometimes a trap that names an abandoned
plan — and the question is whether the path the agent takes lands on the right
worldline, refuses honestly when nothing is remembered, and does not resurrect a
plan that was already thrown away.

## Cases (`cases.json`)

| field | meaning |
|---|---|
| `id` | stable case name |
| `utterance` | what the person says at the start of a session |
| `target_slugs` | reaching any ONE of these is a correct landing (empty ⇒ the honest answer is "not remembered") |
| `acceptable_related` | neighbours that also count as landing well — reported, never required |
| `must_not_anchor` | landing here is a WRONG branch (usually the superseded plan) |
| `category` | one of: direct-name, shared-callsign, ellipsis, reversal, superseded-plan, cross-domain, recent-thread, old-doctrine, near-collision, unknown |

The shipped slugs are placeholders for a house store. A case whose targets do not
exist in the store under test is **skipped with a reason**, never scored — adapting
the file to your own slugs is the intended use.

## Trace format (JSONL, one line per case)

```json
{
  "case": "wf-callsign-1",
  "category": "shared-callsign",
  "routing": "agent-only",
  "resident_tokens": 1840,
  "first_tool": "model",
  "opened": ["freetoken-hybrid"],
  "related_reached": ["exl3-quantization"],
  "thinker_calls": 0,
  "fastpath_used": false,
  "recall_context_tokens": 0,
  "target_reached": true,
  "wrong_branch": false,
  "skipped": null,
  "elapsed_ms": 820
}
```

Raw metrics only. There is deliberately **no composite score**: report the counts,
the token costs and the wrong-branch rate, and read them together.

## Routing modes

```bash
kura bench worldline --cases bench/worldline/cases.json --routing agent-only
```

| mode | what answers the utterance | what it isolates |
|---|---|---|
| `agent-only` | the conversation model reads the resident map and names slugs itself — no kura tools, no fastpath, no thinker | the MODEL's own recognition ability |
| `fastpath` | tier zero only; silence is recorded as no-direct-hit (no thinker rescue) | the deterministic recognizer |
| `full` | the real recall path (fastpath → thinker → word fallback) | today's production stack |

`agent-only` needs a model endpoint (the configured thinker plays the conversation
model); the other two modes run without one.

## What a run prints

A JSON object: per-case `traces`, and a `summary` of raw aggregates — runnable and
skipped case counts, target-reached and wrong-branch counts, thinker/fastpath use,
mean resident tokens and mean opened memories. Traces are also appended to
`<store>/_still/worldline-traces.jsonl` (override with `--trace PATH`).

Compare variants (§11 of the plan) by running the same cases against the same
canonical store snapshot with different `--routing`, and — as later milestones
land — different resident shapes.
