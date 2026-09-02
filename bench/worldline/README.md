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
| `must_not_anchor` | landing here is a WRONG branch (a live neighbour, the other half of a collision) |
| `obsolete_slugs` | optional; landing here is an OBSOLETE branch — the fixture says this memory was thrown away. Counted apart from `wrong_branch` |
| `category` | one of: direct-name, shared-callsign, callsign, ellipsis, reversal, superseded-plan, obsolete-resurrection, cross-domain, recent-thread, old-doctrine, near-collision, prefix-collision, japanese-only, unknown |

Every slug the shipped cases name exists in `memories.json` — a synthetic fixture
store in the spirit of `bench/fixtures` (publishable; nothing from a real house).
Seed a fresh store with `distill_kura.worldline.seed(store, "bench/worldline/memories.json")`
and every case runs. Against any other store a case whose targets do not exist is
**skipped with a reason**, never scored — adapting the file to your own slugs is
the intended use.

The cases added for M4 (adaptive triggers) are the shapes a shorter trigger must
not break: `prefix-collision` (two memories whose triggers share their first words;
the utterance names only the distinguishing tail), `callsign` (the utterance IS a
house callsign and nothing else), `obsolete-resurrection` (the utterance names the
thrown-away rule by its old words), `japanese-only` (no ASCII anywhere), and a
third `unknown`.

## Trace format (JSONL, one line per case)

```json
{
  "case": "wf-callsign-1",
  "category": "shared-callsign",
  "routing": "agent-only",
  "resident_variant": "canonical",
  "resident_tokens": 1840,
  "first_tool": "model",
  "opened": ["freetoken-hybrid"],
  "related_reached": ["exl3-quantization"],
  "thinker_calls": 0,
  "fastpath_used": false,
  "recall_context_tokens": 0,
  "target_reached": true,
  "wrong_branch": false,
  "obsolete_branch": false,
  "remembered_but_unreachable": false,
  "unnecessary_opens": [],
  "skipped": null,
  "elapsed_ms": 820
}
```

Raw metrics only. There is deliberately **no composite score**: report the counts,
the token costs and the wrong-branch rate, and read them together.

| metric | meaning |
|---|---|
| `target_reached` | a target slug was opened (unknown category: NOTHING was opened) |
| `honest_unknown` (summary) | the unknown cases answered from nothing — the same count as the older `unknown_refused_correctly` |
| `wrong_branch` | a `must_not_anchor` slug was opened |
| `obsolete_branch` | an `obsolete_slugs` slug was opened: a dead plan resurrected. Disjoint from `wrong_branch` |
| `remembered_but_unreachable` | the target IS in the store and the map-reading route (`agent-only`, `fastpath`) did not reach it — "the memory exists, the door was too narrow". Always false under `full`, where the thinker's rescue is what hides a thin map |
| `unnecessary_opens` | opened slugs that are neither a target nor `acceptable_related` — the cost side of a wider trigger |

`explanation_burden` (how much the agent had to be told before it landed) is NOT
in v1: it needs a human-in-the-loop definition, and a proxy would get optimized
instead of the thing. It belongs to M8.

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

`--no-cues` runs tier zero without the callsign pre-head so `fastpath` with and
without it can be compared on the same cases (cue_hit / cue_direct_total in the
trace and summary say which answered).

## Resident variants

The same cases, worn with different maps (the guide's §9; the plan's §11):

```bash
kura bench worldline --routing agent-only --resident canonical,woven
kura bench worldline --routing agent-only --resident canonical,woven,adaptive12 \
    --resident-file adaptive12=/path/to/shadow-12.md
```

| variant | the map the agent wears |
|---|---|
| `canonical` | the store's full index (`store.index_text()`) — the default |
| `woven` | the production cloth as the loom would give it now, **no model calls** (`weave(generate=False)`) |
| `NAME` from `--resident-file NAME=PATH` | any text file — an adaptive shadow, a hand-trimmed map, a map from a module that does not exist yet |

A name that is neither a builtin nor a `--resident-file` is refused loudly: a typo
that fell back to canonical would print a healthy comparison of one map against
itself. `agent-only` is the judge between variants — the other two routes let a
rescue path hide a map that has grown too thin.

## What a run prints

By default a table, one row per variant: `resident_tokens` and the raw counts side
by side (runnable, target_reached, wrong_branch, obsolete_branch, honest_unknown,
remembered_but_unreachable, unnecessary_opens, thinker_calls_total, opened_mean).
`--json` prints the full result instead: per-case `traces` (each stamped with its
`resident_variant`), a per-variant `variants` block, and the `summary` over all
traces. Traces are also appended to `<store>/_still/worldline-traces.jsonl`
(override with `--trace PATH`).

Exit code: 0 when something ran and no case landed a wrong or obsolete branch;
1 otherwise.
