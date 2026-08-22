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

### Compatibility

Endpoints have a `dialect` (`vllm` | `openai` | `generic`) and record why a call failed.
"OpenAI-compatible" means it answers `POST <url>/chat/completions` in that shape — a
vendor's native API needs a gateway in front of it.

## 0.1.0

First public release: recall by recognition, the evidence gate, several kura behind one
server, the distiller, the DSH plugin and the MCP bridge.
