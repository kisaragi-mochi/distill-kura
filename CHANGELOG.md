# Changelog

## 0.3.0 — unreleased

### Pay it forward (new)

The resident map is byte-stable so a prefix cache can hold it — but the first turn
after a re-weave still pays the whole cold prefill, and on a slow mouth that is
minutes. `kura pay-forward` pays it once, in the quiet hours: right after a weave
changes the map, it pushes the new map through each `[[payforward.mouths]]` entry (a
llama.cpp server started with `--slot-save-path`) and saves the slot's KV to disk, so
even a mouth restart wakes up warm. Measured on one machine — 320B pure-CPU llama.cpp,
16,444-token map: bake 796 s, save 283 ms (1.5 GB on NVMe), restore after killing and
rebooting the server 655 ms, first turn after it reprocessed 18 prompt tokens. Named
after the film: the cold turn is paid forward so the next one receives it warm.

Slot files are content-addressed on the map's etag (`kura-<store>-<etag…>.bin`), so a
changed etag tries a restore before it bakes — a file left by a lost state file or a
parallel runner is still the right bytes — and a fresh etag is proven, never assumed:
a restore shows the file still exists, a one-token probe with `cache_prompt: true`
reads `timings.prompt_n`, and small means warm. A restore whose 200 was a lie is
caught by the same probe, and the probe's own prefill is saved rather than paid twice;
a reply with no timings at all proves nothing and is refused — fail closed — because
llama.cpp always sends them. An unreachable mouth is a loud, labeled skip that never
advances the per-store state (`_still/payforward.json`, written only after a confirmed
success, read-modify-written under the slot's lock). One runner per physical slot: the
whole sequence holds a machine-local flock keyed on (normalized base url, slot) — the
runners that can collide, `kura tend` and the systemd restart hook, are machine-local —
and a second runner skips cleanly instead of racing. Two config entries naming one
physical slot are refused at load. The ledger itself takes a second lock — the slot
lock cannot cover it: two mouths of one store hold two different slot locks and share
one `payforward.json`, so `--mouth A` / `--mouth B` running together was a lost update.
The read-modify-write now holds a millisecond flock beside the file. And busy is no
longer fresh: a held slot lock proves another runner exists, not that it is warming
your etag, so `skipped-locked` exits 1 (transient — retry) and only a verified
all-fresh run exits 2. Exit precedence puts not-covered first: {A baked, B busy}
exits 1 so the scheduler comes back for B — 0 means the whole fleet is covered, never
"something, at least, worked". Old slot files are
not pruned, because the slots API can save and restore a filename but cannot list the
directory. Exit 2 when every mouth is fresh, so a scheduler can tell "nothing to do"
from work; a failed mouth is exit 1, because a failure is neither. `kura tend` runs it
as a track after each weave, counting bakes and restores — work, never launches.

### The gate reaches the composed text (gate_version 3)

The candidate's quotes were verified; the scribe's finished text was not — and the
scribe is a model too: told "write no numbers", it could still write one with nothing
behind it. Now every numeric token of two or more digits in the final DESC+BODY must
already exist in a verified quote or in the evidence's own date, checked by the same
deterministic substring floor as the quotes themselves. One retry with the violations
named, then the draft is dropped. A derived number — a ratio the scribe computed — is
refused on purpose: arithmetic the evidence never did is a claim the evidence never
made. (Found in an outside review; confirmed against the code before fixing.)

The second review then bent the floor, and it hardened: numbers are matched token
by token, canonicalised (commas forgiven, nothing else) — never against the
evidence's concatenated digits, which had let "899 ms … 2.3 ms" vouch for an
invented "923". A sign is meaning ("-12.5" is not "+12.5"); a range is one claim
("12-16" is not licensed by "12" and "16"); the evidence's date is no longer
auto-allowed in the body — an extension heading gets its date from code, after
verification. The gate's test-file contract now states exactly which classes the
deterministic floor covers, no more.

The third review found the door missing where it mattered most: the judge's FIX
is the LAST model to touch the text, and used to be re-signed without
re-verification. Now nothing earns a mark without passing the floor — a FIX is
re-checked against the draft's full evidence manifest (fail closed if the
manifest is unreadable) and refused if it cannot pass, leaving the draft as
staged. The floor also widened to the whole model-written surface: title,
recognition trigger, section heading and the curation sentences, not just
DESC+BODY — a title lands in the resident map, and "99-GPU構成" must not enter
through it. Tokens are Unicode-normalised (en/em dashes between digits, the
true minus, full-width digits), scientific notation is one token, and single
digits are verified too — "8 GPUs", "4-bit" and "2x" are exactly the claims a
local-model house invents — with ordered-list markers mechanically excluded.
Manifests written from here on say `gate_version: 4`, so provenance can tell
which floor a memory passed.

The fourth review followed the truth downstream: the memory can be right while
the memory the agent WEARS is wrong. Two more writers of resident truth now
stand behind the same floor — `tidy`, the only path that puts model prose into
the canonical index (an invented "99-GPU" title used to walk straight into
recall and the resident map; now the memory itself is the evidence and the
rewrite is refused), and the weave's trigger scribe, whose one-digit swap kept
enough 2-grams to clear the grounding overlap (numbers now need the title or
description to contain them; LEDGER_VERSION 6, so every cached hook re-earns
its place under the new floor). The numeric tokenizer closes its last known
seam — a signed mantissa ("-1e9" no longer decomposes into an evidenced "-1"
and "9") — and content-addressed provenance now means it: manifests are
re-hashed on every read (a FIX over a tampered manifest fails closed) and
`doctor` reports `tampered_manifest` alongside `missing_manifest`.

### An explicit nothing is an answer

`pick_by_meaning` distinguished "the thinker is unreachable" (None) from "the thinker
read the whole index and named nothing" ([]) — and recall then overrode the second
with word overlap anyway, handing back look-alikes for questions the store knows
nothing about. The explicit empty pick is now respected (`how: "meaning→none"`);
word overlap remains the fallback only when the thinker is unreachable.

### Docs

`allowSwitch` in the README now matches the code: it follows `store` (naming a store
binds the preset, fail-closed) rather than defaulting open.

## 0.2.0

The first release shaped by outside review: a security/isolation review, an adversarial
pass over the fixes it produced, and a third-party reproducibility review.

### Tier zero of recall: the fast path (new)

A deterministic five-head recognizer (slug/title containment, word IDF, character
3-grams with stop-grams, character 2-grams, the opening of the body) answers a DIRECT
question — one that names a memory — in well under a millisecond, skipping the
thinker's ~17k-token index prefill entirely. An honesty gate (top1 >= `gate`,
top1/top2 >= 1.15) keeps every paraphrase with the thinker: blind-tested against it
before porting — 14/14 agreement on direct questions, zero wrong answers, silent on
everything semantic. Its index is built lazily from the store's own data and cached
in-process, keyed on the canonical index's mtime and the memory count.

`[fastpath]` in the config (`enabled`, `gate`; per-store overridable), a
`fastpath_verdict` / `fastpath_ms` pair in every recall reply beside `how = "fastpath"`,
and a `fastpath` block in `doctor`. A side effect worth naming: a direct question now
still finds its memory when the thinker is down, instead of degrading straight to
word overlap.

### The resident map (new)

The index is now *worn*, not merely queried: a standing block in the system prompt so an
agent can see what is known — and, more importantly, what is not. Three layers (pinned /
fresh / trigger) from a blind A/B test showing that detail earns its place only for
recent things. Byte-stable by contract, degrades to an honest note rather than to silence
or to half a map, and never truncates the list to fit.

`kura weave`, `kura prefill`, `GET /prefill`, a `systemPrompt.section` in the DSH plugin,
and a `kura_map` tool for hosts that cannot inject.

### Rooms, tags, and the sentences that go with a memory (new)

The room is chosen before the conversation and a memory never leaves it; a memory
may carry several **tags** — words about its character, never weights — and three
curation sentences: `belongs_because`, `keep`, `may_fade`. The distiller proposes
them against the store's charter; a tag that claims something about the human
(`entrusted`, `emotion-carried`, `recurred`) is checked deterministically against
the quotes, and both the basis and every refusal go into the evidence manifest
(`gate_version` 2, additive). `recurred` is written once, by the distiller, when
the human raises a covered topic again from another journal — decided, never
proposed, never counted.

The prompts no longer rank every store alike (decisions, then emotion, then topics
returned to …): the charter ranks, and emotion and recurrence are things not to
walk past. `examples/rooms/` carries a five-room layout — Research / Develop /
Manage / EQ / USER — with charters and a config where each room drinks from its
own journal. The core still serves any stores and any selectors.

A wide room may keep a learned `profile.md` beside its charter, in sentences, read
after the charter by its own distiller and never entering the resident map.
`kura profile draft` writes one from that store's memories; `kura profile apply` is
a person copying a file they have read. A profile carrying numbers about how much
things matter is reported as broken and not read.

A claiming tag needs its evidence, and `landmine` needs an *actual* failure — an
error in `[TOOL]` output, or a warning or correction in the human's words; a quiet
`df` line is tool output and nothing else. The verified door signs the curation it
writes (`curation_mark`, same per-store key as the draft gate mark), and `doctor`
names a hand-edited tag on a `distiller-only` store as `tampered` or `unsigned`.

`POST /annotate`, `kura annotate`, `tags`/`belongs_because`/`keep`/`may_fade` on
`/remember` and the MCP `kura_remember`; `GET /memory` returns them; `doctor` reports
`invalid_tags`, `missing_manifest`, `learned_profile` and **capacity in four units
with `limit` and `pressure` left `None`**.

**Not in this release:** forgetting. Nothing is garaged, settled, absorbed,
released or deleted, and no unit or limit is chosen. `docs/DESIGN.md` §8 says what
is undecided and why the first pass will be a dry run.

### The editor, and the watcher (new)

The model you talk to is also the **editor** that writes and judges memories in its
idle minutes — that is the default, and a GPU model does it acceptably (measured on
the house's Qwen: six drafts in 33 s, judged in 5 s, with reasons in the evidence
vocabulary). The upgrade path is an editor on its own seat, including a CPU model
that never competes with the conversation. `kura tend` is the watcher: quiet is the
newest journal's mtime; after `idle_min` it drains, distils, re-weaves once per
silence and tidies once; a track with nothing to do exits 2 and rests
`backoff_min`; work is counted, launches are not; every track's output is kept;
a heartbeat that `doctor` reads says whether anyone is tending the store; the human's
return stops a running track unless `yield_on_return = false`. Rebuilt from the
five-day record of the house's first watcher and the four ways it went wrong.

`kura distill catchup` marks every journal as drunk up to now, so pointing a
distiller at an existing history does not start by re-reading all of it. Forward
only — it cannot lose progress.

An extension's heading now carries the evidence's date (the journal file's mtime)
and a heading that says otherwise is corrected mechanically — 30 of 39 extension
headings in the house had been dated before the distiller existed.

Not shipped, on purpose: an autonomous research loop. It stays on the house side.

### Boundaries

- **Containment.** Every lookup resolves into the set of memories a store actually holds.
  `GET /memory/..%2Fother%2Fsecret` used to return another store's memory; a `[[../x]]`
  link used to walk there. Explicit reads are exact; only a model's pick is fuzzy.
- **Write authority.** `write_policy = direct-allowed | distiller-only | frozen`, with
  `remember_direct()` and `pour_verified()` as separate doors. The deprecated
  `readonly = true` now means `distiller-only`, which is what it always claimed — it
  used to refuse the distiller's pour as well. The gate signs what it stages and the
  pour verifies it, so a hand-written draft does not pour.
- **Isolation.** Per-store journal roots (no implicit inheritance above one store),
  per-store `model_profile`, and load-time refusal of aliased, nested or
  journal-overlapping stores, mode/store name collisions, unknown or mistyped keys, and
  partial model profiles.
- **`docs/TRUST.md`**, which states plainly that several kura behind one server are
  independent as *routing*, not as confidentiality: one trust level per process.

### Measurement (new)

`kura bench compress` and `kura bench retention`, `_still/metrics.jsonl`, and
content-addressed evidence manifests under `_evidence/` referenced from each memory's
frontmatter, so "why does this memory exist?" stays answerable after the draft is gone.

Measured with the shipped fixtures: `store_ratio` 0.18 on ordinary chat and 1.14 on dense
material. The ratio is a property of the corpus, not of the tool.

### Fixed

- a FIX verdict kept only the first draft header line, so DESC was lost and the
  memory poured with its slug as the index trigger
- a Claude Code subagent transcript records the PARENT MODEL's prompt as `type: user`
  with `isSidechain: true`; it was classed [USER], so a model's "the owner approved X"
  could pass the gate as the human's decision. Sidechain text is [SELF] now (tool
  results stay [TOOL]). 360 of the house's 391 journal files were sidechains
- `tidy()` wrote the index with a bare `open()`, outside the store lock and the atomic
  replace every other index write uses
- an EXTENDS pour overwrote `evidence_manifest`, erasing where the memory came from;
  the first manifest is now pinned as `origin_manifest` and `recur()` reads that
- `profile_draft()` read `_study/` notes first (underscore sorts before letters) and
  a few long notes spent the whole budget before it saw a memory
- `known_slugs()` matched `(AGENTS.md)` inside an index line's prose and reported an
  orphan that was never a memory; only link targets count now
- `commitment` passed `verify_tags()` unconditionally; it is a claim about the human
  and now needs a [USER] quote like `emotion-carried`
- the resident map had no store identity: a failed switch left the previous kura's index
  in the prompt while recall went to the new one
- the trigger quality gate tested the alphabet, rejecting good Japanese triggers (and ★)
- a memory and its index line were two writes; concurrent writers lost index lines
- `chars` in recall was per memory and read as a total; `total_chars` is a hard ceiling
- `doctor` and `/index` still used `len//2`, biased low against every real tokenizer
- the loom could write into a memory slot, destroying it one weave at a time
- `tidy()` and `init_files()` wrote into frozen stores
- `pour '../../../x'` read a file from anywhere on the filesystem into the store
- hardlinks: reported on the read side, filtered by inode on the intake side
- the DSH plugin pinned `@deepseek-ai/dsh-tools` as a direct dependency, so a profile
  could load a second physical copy and split its module-local Symbol identity — the
  first tool call died on `undefined.prepare`. Now a `"*"` peer: the host supplies the
  one copy it already has. First outside contribution, by @kisaragi-mochi (#1)

### Compatibility

Endpoints have a `dialect` (`vllm` | `openai` | `generic`) and record why a call failed.
"OpenAI-compatible" means it answers `POST <url>/chat/completions` in that shape — a
vendor's native API needs a gateway in front of it.

## 0.1.0

First public release: recall by recognition, the evidence gate, several kura behind one
server, the distiller, the DSH plugin and the MCP bridge.
