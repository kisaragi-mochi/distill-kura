# Five rooms

A layout for one person who talks in several registers, and wants each register to
remember its own side of the same afternoon.

| selector | room | what it keeps |
|---|---|---|
| `research` | Research | what became known: hypotheses, sources, measurements with their conditions, refutations, open questions |
| `develop` | Develop | how it was made to work: designs and why, reproducible procedures, bugs and their real fixes, tests, landmines |
| `manage` | Manage | what it changed in the plan: decisions and reversals, priorities, deadlines, resources, dependencies, status |
| `eq` | EQ | what was felt and what settled it: turning points, how this person likes to be met, preferences, boundaries, promises |
| `user` | USER | the person over time, with no narrower purpose: enduring threads, current interests, everyday context, how they like to be helped |

The core of distill-kura does not know these names. It serves any set of stores and
any set of selectors; this directory is a recommended shape, not a built-in one.

## The rules the layout stands on

**The room is chosen before the conversation.** By the host — a DSH preset, a
`KURA_STORE` in an MCP environment, `-s` on the CLI. Not by a model reading the first
message. A conversation that drifts from building into feeling stays in the room it
started in; the host may *offer* another room for the next session, the person decides.
An unknown selector is an error at the door, never a quiet fall to the default.

**One room, many tags.** A memory lives in exactly one store. It may carry several
tags (`decision`, `landmine`, `emotion-carried`, …) that describe its character. A
Develop memory tagged `emotion-carried` is a Develop memory. Nothing in the code
moves, copies or re-files a memory by its tags, and there is no move command.

**Each room drinks from its own journal.** With more than one store, nothing inherits
the global journal root — see `kura.rooms.example.toml`, where every store names its
own. The same topic raised in two rooms produces two memories, each distilled from
that room's own evidence. That is not duplication; the Research "what we learned" and
the Develop "what we did" are different facts.

**The narrow rooms are sharp; the wide room is soft.** Research, Develop, Manage and
EQ have fixed charters and a small vocabulary, so recognition lands cleanly. USER
accepts anything, so its recall is expected to be a little looser. In exchange, USER is
the one room whose understanding may grow: a `profile.md` beside its charter, written
in sentences, read after the charter, updated from an observed draft and never by a
model on its own. See `docs/OPERATING.md`, "The wide room".

## Wiring it

```bash
# the stores
for r in research develop manage eq user; do
  python3 -m distill_kura.cli init $r --path ~/kura/$r
  cp examples/rooms/$r/charter.md ~/kura/$r/charter.md
done
# the config
cp examples/rooms/kura.rooms.example.toml kura.toml   # then edit paths and journals
```

In DeepSeek Harness, one preset per room, each naming its store in the plugin config
(`store: research`, …) — the preset picker *is* the room picker, and it is read when
the session is created. See `../dsh-presets/README.md`. For an MCP host, one server
entry per room with `KURA_STORE=research` etc.; the host's profile or workspace
selects which server is mounted.

## What this layout does not do

It does not route. It does not rank one room's memories above another's. It does not
forget anything yet — capacity is *observed* (`kura doctor` → `capacity`) and nothing
acts on it; what happens when a shelf is full is a conversation to have with real
memories in front of you, not a default. `docs/DESIGN.md` §8.
