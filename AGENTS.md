# AGENTS.md — working on distill-kura

Instructions for an agent (or a person) changing this codebase. Hosts that read
`AGENTS.md` — DeepSeek Harness via `dsh-agent-instructions`, Claude Code, others — pick
this up automatically.

## What this project is

A long-term memory for agents: recall by meaning, writing gated by evidence, several
independent stores so an agent mode change is a memory change. Standard library only,
Python ≥ 3.11. No dependencies is a feature — it lets the whole thing be dropped next
to any host, and it keeps the trust surface small for something that decides what an
agent believes.

## The one rule that outranks the others

**No model output becomes a stored fact without a mechanical check.**

Everything in `distill_kura/distill/gate.py` is deterministic Python, and it stays that
way. If you are tempted to "let the model decide" something the gate currently decides,
you are re-opening the failure this project exists to close: an agent asserts something,
the distiller records the assertion, the next agent reads it back as ground truth. That
loop is self-reinforcing. Prompt instructions do not stop it — that was measured, not
assumed.

Prompts may *help a model pass the gate honestly*. They may not *replace* it.

## Layout

```
distill_kura/
  store.py       one kura: files, index, [[links]], doctor. No model calls at all.
  recall.py      recognition: whole index → picked slugs → walk links → context
  registry.py    stores + modes + model roles, loaded from kura.toml
  thinker.py     one OpenAI-compatible client; three roles (thinker/brain/scribe)
  server.py      the HTTP mouth, store-selectable on every route
  mcp.py         MCP stdio bridge (stdlib only, single file, droppable anywhere)
  cli.py         `kura`
  distill/
    sources.py   journal adapters + evidence classing. Add formats here.
    prompts.py   every prompt. Shared charter head = one cached prefix.
    gate.py      ← the deterministic floor. Treat changes here as load-bearing.
    watermark.py claim-before-drink, flock + max() merge
    seeds.py     ideas (no evidence required) and their graduation
    pipeline.py  the pass: sip → spot → gate → novelty → compose → stage → drain
dsh-plugin/      the DeepSeek Harness plugin (JavaScript)
examples/        a runnable config, two demo stores, DSH preset wiring
tests/           44 tests, no model needed
```

## House style

**Comments explain *why*, and name the failure that produced the rule.** A comment that
restates the code is noise; a comment that says "a byte offset into a recompressed
archive is a lie" saves the next person a day. Where a number or a behaviour came from
a measurement, say it was measured. Never write an estimate in the shape of a
measurement — that is the same sin the gate exists to prevent, committed in source.

**Fail loudly at load.** A bad config value throws with the offending value named. A
silently skipped plugin or a silently ignored field looks exactly like a working one,
and that is the worst failure mode there is.

**Degrade visibly.** When the thinker is unreachable, recall falls back to word overlap
*and labels the answer* `how=words`; the tools surface it as `⚠ degraded`. Never let
quality drop quietly.

**Tools: ASCII names, human descriptions.** A tool name is a function-calling key. The
description is the only thing the model knows about the tool, so it must say when to
call it *and what an empty result means* — otherwise an empty memory gets filled with
invention.

**Registration must be reversible.** In the DSH plugin, every `register()`/`guard()`
disposer goes on `ctx.effect()`. Unloading leaves no debris.

**Policy lives outside the tool.** Read-only is enforced by a monotonic
`ctx.tools.guard()`, not by an `if` inside the tool body — a guard's denial cannot be
overturned by another listener.

## Changing prompts

The charter text sits byte-identically at the head of every role's system prompt. On a
slow local model that is one cached prefix instead of three prefills — a real
measurement, not a theory. **Do not reword the charter per call site.** Add
task-specific text after the separator instead.

Reasoning-effort dialects (`reasoning_effort`, `thinking_effort`, `enable_thinking`) are
all sent at once on purpose: templates ignore unknown variables, and a model left on its
default deep-thinking setting can spend the whole token budget reasoning and return an
empty answer. Do not "clean this up" to one dialect.

## Changing the distiller

- Adding a journal format: subclass `Source` in `sources.py`, register it in `SOURCES`,
  and decide the watermark unit deliberately. Append-only file → byte offset. Anything
  that gets rewritten → a sequence number carried in the events.
- The watermark needs **both** halves: `flock` to serialise, and `max()` to merge. A
  lock alone still lets a stale snapshot win.
- Claim *before* reading, never after. Advance-after-read leaves a window where a
  parallel runner starts at the same offset.
- Anything that bounds coverage (top-N, sampling, a retry cap) must be logged. Silent
  truncation reads as "covered everything".

## Tests

```bash
python3 -m pytest tests -q
```

New behaviour needs a test that fails without it. For anything touching the gate, write
the test **adversarially**: not "does the happy path work" but "what is the shape of the
lie that would get through". The existing gate tests are each a real smuggling attempt.

Tests must not need a model. Use a stub endpoint (see `StubThinker`) or the scripted
HTTP server in `test_pipeline_e2e.py`.

## What belongs to the host, not to us

**Persona.** This project never renders or injects a persona. A store may record which
persona file belongs with it, exposed as a pointer at `GET /profile`; rendering and
injection are the harness's job (`dsh-persona`, `AGENTS.md`, or whatever the host uses).
Keep it that way — memory and identity switch together because the *host* binds them,
not because we grew a second identity system.

**Agent instructions.** Same: `AGENTS.md` is read by the host. We ship one for this
repo; we do not implement the mechanism.

**Authentication.** There is none. The server binds to loopback by default. If you need
auth, put something in front of it rather than growing a half-built auth layer here.
