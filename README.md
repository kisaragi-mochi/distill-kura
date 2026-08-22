# 蒸留蔵 — distill-kura

**A long-term memory for agents that is distilled, not accumulated.**
Recall works by *meaning*, writing is gated by *evidence*, and one server can hold
several separate memories — one per agent mode — so switching mode switches what the
agent remembers.

Ships as a [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) plugin,
an MCP server for any other host, an HTTP service, and a Python library. Standard
library only; no vector database, no embeddings, no framework.

```
        ┌── recall ──────────────────────────────────────────────┐
        │  question → whole index in one prompt → picked slugs   │
        │           → walk [[links]] → the neighbourhood         │  ~0.4 s
        └────────────────────────────────────────────────────────┘
        ┌── distil ──────────────────────────────────────────────┐
        │  journal → classed evidence → candidates → GATE        │
        │  → new? → composed → draft → judged → poured           │
        └────────────────────────────────────────────────────────┘
```

---

## Why this exists

Two failures kill an agent's long-term memory, and they kill it from opposite sides.

**Retrieval by keyword misses the thing you needed.** A question about *"SSD inference
chips"* shares no word with a memory titled *"running the 2.6T model off an SSD tier"* —
yet they are the same subject. Word search returns nothing; the agent answers from
nowhere. The fix here is not embeddings but *recognition*: the entire index (one line
per memory, written as a recognition trigger) goes into one prompt, and a small model
names what bears on the question. An index of ~500 memories is around 6k tokens — a
few percent of a modern context window, and it sits in the prefix cache.

**Writing everything poisons the store.** An agent asserts something; a naive distiller
records the assertion as a fact; the next agent reads it back as ground truth and
repeats it with more confidence. That loop is self-reinforcing, and prompt instructions
do not stop it — measured, not assumed. So the write path is gated by deterministic
Python: every candidate memory must carry quotes that exist *character-for-character*
in the raw material, tagged with where they came from.

| class | what it is | what it licenses |
|---|---|---|
| `[USER]` | the human's own words | "they decided", "they asked" |
| `[TOOL]` | machine output | **numbers — the only source** |
| `[ACT]` | a tool that was invoked | "this was done" |
| `[SELF]` | the agent's own prose | a judgement, in the first person, never a bare fact |

A quote that is not found verbatim is discarded. A candidate with no surviving quote is
thrown away. A number with no `[TOOL]` behind it is stripped. Text crediting the human
with a decision, when no `[USER]` quote survived, is refused at the last gate. Ideas are
welcome — they go to a seed file, never to the store, and graduate only when later
evidence confirms them.

---

## Quick start

```bash
git clone https://github.com/lna-lab/distill-kura && cd distill-kura
pip install -e .                       # or just run: python3 -m distill_kura.cli

cp kura.example.toml kura.toml         # edit: one model endpoint is enough to start
kura init main --path ~/kura/main      # create an empty store
kura serve                             # http://127.0.0.1:8085
```

```bash
curl -s -X POST localhost:8085/recall -H 'content-type: application/json' \
     -d '{"question":"what did we decide about the archive disk?","hops":1}'
```

Wear the index, so the agent always knows what is known:

```bash
kura weave                             # build the three-layer cloth
kura prefill                           # the block to put in the system prompt
```

Feed it your agent transcripts:

```bash
kura distill run      # drink a batch → candidates → gate → drafts
kura distill drafts   # look at what it wants to write
kura distill drain    # the scribe re-reads each draft cold: pour / fix / toss
kura distill night    # stay resident and do it whenever things go quiet
```

Nothing enters the store until `drain` (or a hand-run `pour`). Drafts carry their
evidence in an HTML comment, so you can always see *why* a memory exists.

---

## The resident map

Recall-by-tool answers *"what do you know about X?"* — but only once the agent has
decided to ask. It never answers the question the agent does not think to ask: **is
there anything here at all?** An agent that cannot see the map does not know what it is
missing, so it guesses, and a confident guess about your household is precisely the
failure this project exists to prevent.

So the index is also worn: a standing block in the system prompt, on every turn.

```bash
kura weave      # re-weave the index into the three-layer cloth
kura prefill    # print the block a host should inject
```

### Three layers, because detail only pays for recent things

A blind A/B test — 20 questions, fat index vs slimmed index, scored without knowing
which was which — settled the shape:

| band | fat | slim |
|---|---|---|
| overall | 9 | 11 |
| recent events | **4** | 1 |
| doctrine | 1 | **4** |
| cross-domain leaps | 1 | **4** |

The doctrine lines were byte-identical in both indexes, and the slim index still won
that band: **a lighter surround makes the standing lines work better.** Detail is not
the source of insight. It earns its place only where things are still moving.

| layer | rule | line |
|---|---|---|
| pinned | frontmatter `type` in `pinned_types` | kept in full |
| fresh | changed within `fresh_days` | kept in full |
| trigger | everything else | compressed to ~`trigger_tokens` |

Trigger lines are written by the `scribe` model and cached in a ledger keyed on the
description *and* the budget, so a re-weave in the steady state costs nothing. With no
model reachable the loom trims mechanically instead — a memory system must not go blank
because a GPU is down.

**Age is not mtime.** `cp -r`, a restore or a checkout resets every timestamp, the whole
index turns "fresh", nothing is trimmed, and the mechanism has silently switched itself
off. So the loom prefers a date written *inside* the memory, and distrusts any mtime
that a fifth of the store shares with one calendar day.

### Where it goes, and why that is a cache decision

```yaml
- id: kura
  name: distill-kura
  config: { store: eq, promptOrder: -50 }   # before the persona
```

A prefix cache is lost from the first changed byte onward — measured on one local
server: an identical 4,029-token preamble reprices from 0.68 s to 0.14 s, appending at
the **end** stays 0.14 s, and **one word added at the front costs the whole cache
(0.66 s)**. The persona commonly carries a clock, so it changes every minute; the map is
the largest block in the prompt and changes a few times a day. The big stable thing goes
in front of the thing that ticks.

The block itself therefore contains no date, no clock, no counter — and `build()`
refuses a header that does, at build time rather than through mysteriously slow turns
three weeks later.

### It never hands over half a map

| situation | what the agent gets |
|---|---|
| all well | the map, between `<<<KURA-MAP>>>` markers |
| over `budget_fraction` | the **whole** map, and a warning in the JSON (never in the text — a banner is volatile content) |
| over `hard_fraction` | a **stub** with no index lines, saying the map is missing rather than empty |
| kura unreachable | an explicit note that the map is missing, never an empty string |

A truncated map is the worst artifact available: it looks complete, and every memory
below the cut appears not to exist. `weave` will shorten the fresh window to fit, but it
will never drop a line — and if no setting reaches the budget it says so, keeps the
better map, and tells you where the weight is.

### Getting it into a host

| host | mechanism |
|---|---|
| DSH | native plugin — a `systemPrompt.section`, refreshed in the background |
| Claude Code, VS Code, Goose | MCP `instructions` carries a short pointer (**2KB cap**); the map itself comes from the `kura_map` tool or a session hook running `kura prefill` |
| Claude Desktop, claude.ai | ignore `instructions` entirely — use `kura_map` |
| anything else | `GET /prefill?format=text`, or `kura prefill` in a shell hook |

The MCP `instructions` field is a `MAY` in the spec, and a 9,000-token index cannot
travel through a 2KB cap regardless, so this project does not pretend otherwise.

---

## Modes: more than one kura

A single memory that serves both "help me build this" and "help me think this through"
serves neither well: the recall that helps you debug is noise in a conversation about
what to do next. So a store is a directory, and a mode maps to a store.

```toml
[stores.maker]
path = "~/kura/maker"
label = "maker mode — building things"

[stores.eq]
path = "~/kura/eq"
label = "EQ mode — talking things through"

[modes]
maker = "maker"
eq    = "eq"
```

Every route takes a selector, so one process serves them all:

```bash
curl -s -X POST localhost:8085/recall -d '{"question":"...","mode":"eq"}'
curl -s localhost:8085/index?store=maker
curl -s localhost:8085/s/eq/doctor          # path form, for clients that only vary a base URL
```

The stores share no memories, no index, and no distiller watermark. Switching mode
genuinely changes what is remembered — not the same memory in a different voice.

**Independent as routing, not as confidentiality.** The server has no authentication, so
any process that can reach its port can name any store it holds. Binding an agent keeps
a *model* in its lane; it does not keep a *process* out. One trust level per process —
[`docs/TRUST.md`](docs/TRUST.md) is short and worth reading before a private store goes
in. It also covers the two boundaries that are easy to miss: two stores drinking from
one journal root, and two stores behind one model endpoint.

### With DeepSeek Harness

DSH switches **persona and tools** by agent preset. distill-kura switches **memory** by
store. Bind them and one preset change moves the whole self:

```yaml
# .agent-presets/eq/agent.cordis.yml
- id: kura-eq
  name: distill-kura
  config:
    url: http://127.0.0.1:8085
    store: eq            # this preset's memory
    readonly: true       # the CLIENT's own switch: do not even offer a write tool
    # (the store's own `write_policy` is the authority; this just keeps the tool
    #  out of the model's hands. Naming a store already binds the preset.)
```

Leave `allowSwitch` at its default and the agent also gets `kura_use`, so it can move
between kura mid-conversation without a preset change. Tools: `kura_recall`,
`kura_read`, `kura_doctor`, `kura_list`, `kura_use`, and `kura_remember` (only when the
store is writable). Full wiring, including the MCP bridge and the `isolate` realm rule
for service rows, is in [`examples/dsh-presets/`](examples/dsh-presets/).

**Persona is the host's business, not ours.** This project never renders or injects a
persona; it only records, per store, which persona file belongs with it, readable at
`GET /profile?store=eq` so the two halves can be kept in step by whoever owns the
preset. Agent instructions likewise stay with the host's `AGENTS.md` mechanism — see
[`AGENTS.md`](AGENTS.md) in this repo for the conventions an agent working *on* this
codebase should follow.

### With any MCP host

```jsonc
{ "mcpServers": { "kura": {
    "command": "python3", "args": ["-m", "distill_kura.mcp"],
    "env": { "KURA_URL": "http://127.0.0.1:8085", "KURA_STORE": "eq", "KURA_READONLY": "1" }
}}}
```

Leave `KURA_STORE` unset for free mode: the tools take an optional `store` argument and
`kura_use` switches for the session.

---

## Models: one by default, upgrade a role at a time

Three **roles**, not three machines:

| role | when it runs | wants |
|---|---|---|
| `thinker` | every recall | small and fast; must judge relevance by meaning |
| `brain` | distilling: reads a whole batch of journal | context length and patience |
| `scribe` | distilling: writes the memory, then judges drafts | good prose in your language, judgement |

Declare only `[models.thinker]` and one model does all three. Upgrade either of the
others independently — a bigger local model, or an online API (any OpenAI-compatible
`/chat/completions`; the key is read from an environment variable you name, never
stored in the config):

```toml
[models.thinker]                       # always-on, local, small
url = "http://127.0.0.1:8000/v1"
model = "local-small"

[models.scribe]                        # upgrade just the writing
url = "https://api.example.com/v1"
model = "big-model"
api_key_env = "EXAMPLE_API_KEY"
```

Two things this handles for you: reasoning-effort dialects differ per model family
(`reasoning_effort`, `thinking_effort`, `enable_thinking`), so **all** of them are sent
— an unknown one is ignored by the template, while a model left on deep-thinking by
default can spend its whole budget reasoning and return nothing. And the charter text
is placed byte-identically at the head of every role's prompt, so on a slow local model
the three roles share one cached prefix instead of paying three prefills.

**If the thinker is down, recall does not go silent** — it falls back to word overlap
and labels the answer `how=words`, which the tools surface as `⚠ degraded`. Quiet
degradation is worse than degradation.

---

## What a memory looks like

One file, one fact.

```markdown
---
name: archive-on-slow-disk
description: the archive lives on the slow disk; the fast one stays scratch
metadata:
  type: project          # user | feedback | project | reference
---

The archive goes on the slow disk. The fast disk is scratch space.

**Why:** the other way round burns write endurance for nothing.
**How to apply:** check which disk a target directory is on before writing there.
Related: [[disk-layout]]
```

And one line in `MEMORY.md`:

```markdown
- [Archive on the slow disk](archive-on-slow-disk.md) — the archive lives on the slow disk; the fast one stays scratch
```

That line is the only thing read *every single time*. It is a **recognition trigger**,
not a summary: proper nouns, numbers, ⚠️ landmines, the conclusion reached. If a line
could be swapped with another memory's line and still read fine, it is not doing its
job — `kura distill tidy` finds the mechanically detectable cases and rewrites them.

`kura doctor` reports counts, dead links, islands (memories nothing links to), and
index drift. It is the eye the metabolism needs.

---

## The HTTP surface

| route | what it does |
|---|---|
| `POST /recall` | `{question, hops, top, chars, total_chars, store\|mode}` → picked, walked, context. `chars` is per memory; `total_chars` is a hard ceiling on the whole context |
| `POST /remember` | `{slug, description, body, type, title}` — a DIRECT write, refused unless `write_policy = "direct-allowed"` |
| `GET /index` | the raw index |
| `GET /prefill` | the resident block, ready to inject (`&format=text` for a hook) |
| `GET /memory/<slug>` | one memory in full |
| `GET /doctor` | health of one store (`?all=1` for every store) |
| `GET /stores` | stores, modes, and which model fills each role |
| `GET /profile` | the store's charter, and a pointer to its persona (never rendered here) |
| `GET /health` | liveness |

Any route accepts `?store=` / `?mode=`, a `store`/`mode` field in the body, or the
`/s/<name>/…` path prefix. No authentication: bind to loopback, or put something in
front of it.

---

## Design notes worth reading before you change things

- **[docs/DESIGN.md](docs/DESIGN.md)** — why recognition beats search, what the gate
  buys, and the failure that motivated each mechanism.
- **[docs/OPERATING.md](docs/OPERATING.md)** — running it resident, schedulers and exit
  codes, backups, what to watch.
- **[docs/TRUST.md](docs/TRUST.md)** — what a store boundary is and is not, write
  policies, and the two boundaries that are easy to miss (shared journals, shared
  models). Read it before a private store goes in.

A few decisions that look odd until you hit the thing they prevent:

- **Reserve before drinking.** The distiller claims a stretch of journal *before*
  reading it, under a lock, and watermarks only ever move forward. Two distillers each
  writing back their own snapshot erased each other's progress and re-drank the same
  water a dozen times.
- **Watermarks are per-adapter units.** Byte offsets for append-only transcripts,
  sequence numbers for archives that get rewritten (a byte offset into a recompressed
  file is a lie).
- **Echo suppression.** A quote that already exists in the store is not new material —
  it is the store reading itself back through a tool result. Without this, a memory
  system rediscovers and re-records its own contents forever.
- **The last gate is a model, not a human.** If a person must approve every draft, the
  system has quietly made that person its bottleneck, and drafts pile up forever.
  Nothing in the loop may require someone who is not always present.
- **`kura distill run` exits 2 when there was nothing to do.** A scheduler must be able
  to tell "did work" from "found nothing", or a watchdog spins on an empty queue and
  starves the steps that need the idle time.

## Measuring it, instead of claiming it

Two questions get answered with one number and should not be.

**How much smaller?** `store_ratio` = tokens in the memories and index / tokens of raw
journal actually consumed. **What was lost?** That is a different measurement, and a
store that keeps one memory in a hundred scores beautifully on the first while being
useless.

```bash
kura bench compress                       # what this store cost, from the distiller's own metrics
kura bench compress --tokenizer-command "./count-tokens"   # exact, not estimated
kura bench retention --questions bench/fixtures/questions.json
```

Measured here, with the shipped fixtures and the built-in estimator:

| corpus | `store_ratio` |
|---|---|
| `scripts/demo-clean-room.sh` (ordinary chat, mostly filler) | **0.18** |
| `bench/fixtures/corpus.jsonl` (dense: every line is signal) | **1.14** |

The second one is not a bug. On material where nothing is filler, distilling does not
compress — each memory adds its *why* and *how to apply*, and the store comes out
slightly larger than the transcript. **The ratio is a property of the corpus, not of
this tool**, which is why there is no headline number here and why the command reports
what it counted with.

Retention is scored model-free: each planted fact carries a marker that must appear in
what recall returns, so the score is reproducible on someone else's machine. Distractors
invert — a fact marked `must_not_store` costs a point if the store kept it, because a
memory system is judged by what it declines as much as by what it keeps.

    score 1.0 (10/10)   decision 1/1  number 2/2  negation 1/1  reversal 1/1
                        conditional 1/1  landmine 1/1  returning 1/1  distractor 2/2

That is ten planted facts in a synthetic fixture, distilled by a local Qwen3.8-27B
(NVFP4) as brain and scribe with `max_items = 8, coverage_passes = 2`, and scored with
the same model as thinker. A different model will give a different score: the score
measures a *pipeline-plus-model*, and the fixture exists so the model is the only thing
that varies. It measures whether a fact is *findable*, not whether the answer reads
well — judging prose needs a model, and then the benchmark stops being reproducible.

`kura distill run` writes one line per batch to `_still/metrics.jsonl`, which is where
the raw side comes from. The canonical side counts **only memories whose evidence
manifest points at a recorded batch** — dividing a whole store by the raw material of a
few batches is a number in the wrong direction by an order of magnitude, and the first
version of this command did exactly that. Memories that predate manifests are reported
as `unattributed`, not silently included. The raw side is always the distiller's
estimate at drink time, so with `--tokenizer-command` the ratio is labelled `mixed`.

## What this runs against

| | requirement |
|---|---|
| Python | 3.11+ (no dependencies; `pip install -e ".[dev]"` only adds pytest) |
| Node | 20+, for the DSH plugin only |
| `zstd` | only to read DSH session archives |
| model endpoint | anything answering `POST <url>/chat/completions` in the OpenAI shape |

**"OpenAI-compatible" is narrower than "any provider."** A vendor's native API needs an
OpenAI-compatible gateway in front of it; its own URL will not do. A strict service also
rejects unknown top-level fields, so set `dialect = "openai"` (or `"generic"`) — the
default `"vllm"` sends `chat_template_kwargs`, which local servers want and a strict one
400s on. The client retries once with a plain body and records *why* a call failed
rather than collapsing every cause into a silent `None`.

## Tests

```bash
python3 -m pytest tests -q                              # 145 tests, no model required
cd dsh-plugin && npm test                               # 24 more for the plugin
```

The gate is tested adversarially: every case is a way a real model actually tried to
smuggle something past it. `test_containment.py` is written the same way — every case is
an escape attempt, not a happy path — because it guards a hole that was real: a store
used to answer for any file whose path you could spell. The end-to-end test runs a full
distil→drain cycle against a scripted model server on a real socket.

## License

MIT.
