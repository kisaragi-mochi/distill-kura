# Operating a kura

## Run it resident

The server is a single Python process, stdlib only. systemd (user unit):

```ini
# ~/.config/systemd/user/kura.service
[Unit]
Description=distill-kura
After=network.target

[Service]
Environment=KURA_CONFIG=%h/kura/kura.toml
ExecStart=/usr/bin/python3 -m distill_kura.cli serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now kura
curl -s localhost:8085/health
```

The distiller is separate on purpose — the mouth must stay up even when the distiller
is wedged, and the distiller must be killable without taking recall down with it.

```ini
# ~/.config/systemd/user/kura-tend@.service      (one instance per store)
[Unit]
Description=distill-kura watcher for %i
After=kura.service

[Service]
Environment=KURA_CONFIG=%h/kura/kura.toml
ExecStart=/usr/bin/python3 -m distill_kura.cli -s %i tend
Restart=always
RestartSec=30

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now kura-tend@maker kura-tend@eq
curl -s 'localhost:8085/doctor?store=maker' | python3 -c 'import json,sys; print(json.load(sys.stdin)["tending"])'
```

`Restart=always`, not `on-failure`: the first watcher this replaced was started by a
boot script, died with the machine, and was not missed for twelve days. `tend` works
only once the journals have been quiet for `idle_min` (10) minutes, rests a track that
had nothing to do for `backoff_min` (20), stops a running track when the journal
changes (unless `yield_on_return = false` — an editor on its own seat), and keeps every
track's output in `_still/tend.log`. `distill night` still exists for a bare
one-pass-when-quiet loop, but `tend` is the one to run.

## Keeping the resident map current

The map the agent wears is only as good as its last weave.

```ini
# ~/.config/systemd/user/kura-weave.service   (+ a .timer, or a cron line)
[Service]
Type=oneshot
Environment=KURA_CONFIG=%h/kura/kura.toml
ExecStart=/usr/bin/python3 -m distill_kura.cli weave
```

Weave after the distiller pours, and once more in the morning:

```bash
kura distill drain && kura weave
```

A weave in the steady state costs nothing: trigger lines are cached in a ledger keyed on
the description *and* the budget, so only genuinely changed lines reach a model, and an
unchanged cloth is not rewritten at all (`written: false`). Re-weaving on a tight timer
is still pointless — the index changes a few times a day, not a few times a minute.

**What to watch:**

| signal | meaning |
|---|---|
| `kura prefill` exits 2 | no cloth yet, or the cloth is stale — the full index is being served instead |
| `over_budget: true` | the map costs more than `budget_fraction`. Nothing was dropped |
| `budget_met: false` | no setting reaches the budget; the vivid layer was kept deliberately |
| `over_ceiling: true` | the map is not being shown at all — a stub went out instead. Act on this |
| `hooks_mechanical` high | the scribe is failing the quality bar, or is unreachable |
| `grouped` large | many lines name several memories each; those are never trimmed |

If the map will not fit: lower `trigger_tokens`, narrow `pinned_types`, shorten
`fresh_days`, raise `budget_fraction` if the window can afford it — or split the store,
which is the honest answer past a few hundred memories.

**Never** hand-edit the woven cloth. It is derived; the next weave overwrites it. Edit
the canonical `MEMORY.md`, or the memory itself, and re-weave.

**Never** hand-write a draft into `_still/drafts/` and expect it to pour on a
`distiller-only` store: the gate signs what it stages and the pour checks the signature.
Write the memory with `kura remember` on a `direct-allowed` store, or let the distiller
produce it.

## Scheduling by hand, and exit codes

```bash
kura distill run    # 0 = did work, 2 = nothing worth drinking
kura distill drain  # 0 = poured or tossed something, 2 = no drafts
kura distill tidy   # 0 = repaired a line, 2 = index is clean
kura prefill        # 0 = a current cloth was served, 2 = no cloth or it is stale
```

**Exit 2 means "there was nothing to do".** A scheduler must distinguish it from
success. A loop that treats "found nothing" as "did work" spins, and the spinning
crowds out the steps that need idle time — measured once as 184 empty passes in 42
minutes, during which the pour step never ran at all.

A reasonable cron shape (one queue, in order, stop at the first idle step):

```bash
kura distill run && kura distill drain; kura distill tidy -n 3
```

## Watch these

```bash
curl -s localhost:8085/doctor | python3 -m json.tool
```

| signal | meaning | do |
|---|---|---|
| `how: words` in recall answers | the thinker is unreachable; quality has dropped | restart the model endpoint |
| `links_dead` growing | memories reference slugs that do not exist | fix the links or write the missing memory |
| `islands` growing | memories nothing links to; they will rarely be recalled | link them, or accept they are archive |
| `index_tokens_est` near your window | the index no longer fits in one prompt | split into a second kura, or a two-level index |
| `not_in_index` | a memory exists but nothing points at it | it is invisible to recall — index it |
| drafts piling up in `_still/drafts/` | `drain` is not running | run it; check the scribe endpoint |
| `_still/tossed.jsonl` growing fast | the brain is proposing junk | look at what it proposes; the prompt or the journal source may be wrong |
| `invalid_tags` non-empty | a `tags:` line a reader cannot parse; the memory reads as untagged | fix the line by hand (it is a JSON array of lower-case kebab words) |
| `missing_manifest` non-empty | a memory points at an evidence manifest that is gone | restore `_evidence/` from backup; the memory still reads, its provenance does not |
| `curation.tampered` non-empty | a memory's tags or sentences were edited after the gate signed them | read the file; restore from the manifest or re-pour. On a `distiller-only` store this is the one thing nobody but the gate should have done |
| `curation.unsigned_names` non-empty | tags or sentences with no mark, on a store where only the gate writes | someone wrote frontmatter by hand; decide whether it stays (it reads fine; it just is not the gate's word) |
| `tending.alive: false` | no watcher is tending this store (never started, or died) | `systemctl --user status kura-tend@<store>`; the heartbeat is `_still/tend.json` |
| `learned_profile.state: broken` | `profile.md` is unreadable or carries numbers about how much things matter; it is NOT read and the fixed charter carries on | read `why`; rewrite the profile in sentences |
| `capacity.*` | the store's size in four units — memories, index tokens, body tokens, bytes. `limit` and `pressure` are `None` on purpose | nothing acts on these yet. Watch them, and see the next section |

## When a shelf is full

Nothing is forgotten automatically, and `capacity` has no limit until a person sets
one. The unit a store is measured in, the limit, whether a wide room shares it, how
candidates are compared, where a garaged memory goes and for how long, which tags
protect absolutely and which only argue, who approves, and what finally deletes a file
— none of that is decided, and `docs/DESIGN.md` §8 says why it will be decided with
real memories in front of the people whose memories they are rather than by a default.

What you can do today is keep the material that decision will need:

- the distiller writes `belongs_because` / `keep` / `may_fade` on every memory it pours,
  so "what does this still do here?" has a first answer on file
- `kura annotate <slug> --keep "…" --may-fade "…"` adds them to older memories by hand
  (direct door: `direct-allowed` stores only)
- `kura doctor` once a week into a log, so the four capacity units have a history

When the first forgetting pass comes it will be a dry run — store, pressure, candidates,
their tags and sentences, the reason each could be released, what must be kept, and the
proposed action — and it will modify nothing.

## Back it up

A kura is plain markdown in a directory. Anything works — `git` is a good default,
because a memory store's history is worth having:

```bash
cd ~/kura/main && git init && git add . && git commit -m "kura snapshot"
```

Back up to a **different physical device**, and verify a restore. A copy that has been
silently failing for a week is the same as no copy: it is worth checking that the
newest file in the backup is actually new.

What to back up, and what not to:

| directory | keep it? |
|---|---|
| the memories and `MEMORY.md` | **yes** — this is the store |
| `_evidence/` | **yes** — the manifests memories point at. Losing them makes "why does this memory exist?" unanswerable, which is the question the whole gate exists to keep answerable |
| `_still/` | no. Workshop: drafts, watermarks, the read log, dropped and tossed candidates, metrics. Deleting `watermark.json` makes the distiller re-drink every journal from the beginning — expensive, not harmful |

`_still/metrics.jsonl` is worth keeping if you care about trend: it is one line per
distilled batch and the only place `kura bench compress` can learn how much raw material
a store actually consumed.

## Feeding it

```toml
[distill.journals]
claude = "~/.claude/projects/my-project"   # Claude Code transcripts (.jsonl)
dsh    = "~/dsh/sessions"                  # DSH sessions (.jsonl.zstd; needs `zstd`)
text   = "~/notes"                         # .md/.txt/.log, all treated as [USER]
evidence = "~/journals"                    # classified .evidence.jsonl (USER/SELF/ACT/TOOL)
```

Newest journals are drunk first: today's decisions are worth the most. A batch is
`chunk_chars` (default 200k) of classed text — aim it at what your brain model swallows
comfortably in one gulp, because a bigger batch means fewer, better-contextualised
candidates rather than more of them.

Per-store journals, when two kura are fed from different places:

```toml
[stores.eq.distill]
language = "日本語"
[stores.eq.distill.journals]
text = "~/notes/dialogue"
```

## Multiple kura on one server

Everything is store-selectable, so one process is normally right. Watch two things:

- **Watermarks are per store.** Distilling the same journal into two stores is allowed
  and each keeps its own position — useful when a conversation is relevant to both, but
  it doubles model cost, so bind journals per store when you can.
- **`write_policy = "distiller-only"`** on a store refuses direct writes (a tool call,
  the CLI, `POST /remember`) while still accepting the distiller's verified pour. That is
  the recommended setting for anything an agent reads from constantly: every memory then
  has to pass the evidence gate. `frozen` refuses the pour too. The deprecated
  `readonly = true` maps to `distiller-only`. See `docs/TRUST.md`.

- **Bind each store to its own journal root.** With more than one store configured,
  nothing inherits `[distill.journals]`; a store with no `[stores.<name>.distill.journals]`
  drinks nothing, and says so in `kura stores`. This is the whole reason two rooms can
  remember two sides of one afternoon: each only ever sees its own conversations. The
  five-room example in `examples/rooms/` shows the shape.
- **A memory never changes store.** Tags do not move it, a mode change does not move
  it, and there is no move or copy command. If the same topic belongs in two rooms,
  each room distils its own memory from its own evidence.

## The wide room

A store whose charter is "understand the person, without a narrower purpose" may keep a
`profile.md` beside its charter: a few sections in sentences — enduring threads, current
interests, everyday context, conversation preferences, unresolved threads. That store's
distiller reads it after the charter, so what the room keeps from then on follows what
it has come to understand. It never enters the resident map, and `GET /profile` hands it
to a host with its state.

```bash
kura -s user profile show      # state (absent / present / broken), the text, and whether a draft waits
kura -s user profile draft     # → _still/profile.draft.md, from THIS store's memories only
$EDITOR ~/kura/user/_still/profile.draft.md
kura -s user profile apply     # a person copying a file they have read. Never automatic
```

A profile is text. One that carries numbers about how much things matter (`trading:
0.8`, `interest score 7`) is the weight this project refuses to store: it is reported as
`broken`, not read, and the fixed charter carries on — visibly, in `doctor`, not as a
quiet fallback. Whether and when a draft should ever be applied without a person is a
decision to make with real drafts in hand; the command exists so those drafts can be
collected.

## Adding a journal format

Subclass `Source` in `distill_kura/distill/sources.py`, register it in `SOURCES`, and
choose the watermark unit deliberately: byte offset for append-only files, a sequence
number carried in the events for anything that gets rewritten. Then add a key under
`[distill.journals]`. The evidence classes are the contract — get the mapping right
(`USER` / `TOOL` / `ACT` / `SELF`, and drop reasoning blocks) and everything downstream
works unchanged.

## Measuring rather than guessing

```bash
kura bench compress     # store_ratio and map_ratio, from the distiller's own metrics
kura bench retention --questions bench/fixtures/questions.json
```

`bench compress` refuses to compute a ratio against journals the store never drank. Pass
`--tokenizer-command` before quoting a figure anywhere: without it the numbers carry an
`estimated` label and a warning, because the built-in estimator is up to ~20% out against
tokenizers it was not fitted on.

`bench retention` exits non-zero below 0.9, so it can sit in a scheduled job. Write your
own questions against your own corpus — the shipped fixture measures the tool, not your
store.

## Security

There is no authentication. The default bind is `127.0.0.1`. If it must be reachable
from elsewhere, put a reverse proxy with auth in front; do not add a half-built auth
layer to the server. Remember that a kura is a fairly intimate document — it holds what
someone decided, what annoyed them, and what they came back to.
