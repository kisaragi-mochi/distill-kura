# Changelog

## 0.2.0 — unreleased

The first release shaped by outside review: a security/isolation review, an adversarial
pass over the fixes it produced, and a third-party reproducibility review.

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

`POST /annotate`, `kura annotate`, `tags`/`belongs_because`/`keep`/`may_fade` on
`/remember` and the MCP `kura_remember`; `GET /memory` returns them; `doctor` reports
`invalid_tags`, `missing_manifest`, `learned_profile` and **capacity in four units
with `limit` and `pressure` left `None`**.

**Not in this release:** forgetting. Nothing is garaged, settled, absorbed,
released or deleted, and no unit or limit is chosen. `docs/DESIGN.md` §8 says what
is undecided and why the first pass will be a dry run.

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
