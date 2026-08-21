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
    readonly: true       # writing goes through the distiller's gate
    allowSwitch: false   # bound: the preset IS the mode switch
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
| `POST /recall` | `{question, hops, top, chars, store\|mode}` → picked, walked, context |
| `POST /remember` | `{slug, description, body, type, title}` — refused on a read-only store |
| `GET /index` | the whole index, for a client that wants it resident |
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

## Tests

```bash
python3 -m pytest tests -q     # 44 tests, no model required
```

The gate is tested adversarially: every case is a way a real model actually tried to
smuggle something past it. The end-to-end test runs a full distil→drain cycle against a
scripted model server on a real socket.

## License

MIT.
