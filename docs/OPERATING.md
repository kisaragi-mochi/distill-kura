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
# ~/.config/systemd/user/kura-distill.service
[Service]
Environment=KURA_CONFIG=%h/kura/kura.toml
ExecStart=/usr/bin/python3 -m distill_kura.cli distill night --idle-min 20
Restart=on-failure
```

`night` only runs a pass once the journals have been quiet for `--idle-min`, so it stays
out of the way of live work.

## Scheduling by hand, and exit codes

```bash
kura distill run    # 0 = did work, 2 = nothing worth drinking
kura distill drain  # 0 = poured or tossed something, 2 = no drafts
kura distill tidy   # 0 = repaired a line, 2 = index is clean
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

## Back it up

A kura is plain markdown in a directory. Anything works — `git` is a good default,
because a memory store's history is worth having:

```bash
cd ~/kura/main && git init && git add . && git commit -m "kura snapshot"
```

Back up to a **different physical device**, and verify a restore. A copy that has been
silently failing for a week is the same as no copy: it is worth checking that the
newest file in the backup is actually new.

`_still/` is the distiller's workshop (drafts, watermarks, read log, dropped and tossed
candidates). It is not part of the memory and does not need backing up — but deleting
`watermark.json` makes the distiller re-drink every journal from the beginning, which
is expensive rather than harmful.

## Feeding it

```toml
[distill.journals]
claude = "~/.claude/projects/my-project"   # Claude Code transcripts (.jsonl)
dsh    = "~/dsh/sessions"                  # DSH sessions (.jsonl.zstd; needs `zstd`)
text   = "~/notes"                         # .md/.txt/.log, all treated as [USER]
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
- **`readonly = true`** on a store means tools cannot write to it at all; the distiller's
  `pour` path is the only way in. That is the recommended setting for anything an agent
  reads from constantly.

## Adding a journal format

Subclass `Source` in `distill_kura/distill/sources.py`, register it in `SOURCES`, and
choose the watermark unit deliberately: byte offset for append-only files, a sequence
number carried in the events for anything that gets rewritten. Then add a key under
`[distill.journals]`. The evidence classes are the contract — get the mapping right
(`USER` / `TOOL` / `ACT` / `SELF`, and drop reasoning blocks) and everything downstream
works unchanged.

## Security

There is no authentication. The default bind is `127.0.0.1`. If it must be reachable
from elsewhere, put a reverse proxy with auth in front; do not add a half-built auth
layer to the server. Remember that a kura is a fairly intimate document — it holds what
someone decided, what annoyed them, and what they came back to.
