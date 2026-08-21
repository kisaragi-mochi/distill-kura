# Design notes

Each section is a mechanism and the failure that produced it. Nothing here is
speculative architecture; every rule was paid for.

---

## 1. Recall is recognition, not search

**The failure.** A question about *"SSD inference chips"* returned nothing, while the
store held a memory about *"running the 2.6T model off an SSD tier"*. Word search
cannot connect them: no term is shared. Worse, the agent did not know it had missed
anything, so it answered from nowhere.

**The mechanism.** Put the *entire index* into one prompt and let a small model name
what bears on the question. One line per memory, written as a recognition trigger. The
model answers with slugs; we walk `[[links]]` from there and return the neighbourhood.

**Why it is affordable.** ~50 tokens per index line. 500 memories ≈ 6k tokens — a few
percent of a modern context window, and it is a stable prefix, so it stays in the
prefix cache. Measured end-to-end at ~0.4 s on a local 27B-class model.

**Why not embeddings.** Embeddings answer "what is similar to this text". Recognition
answers "what bears on this question", which is a judgement, not a distance. The
semantic gap in the failure above is exactly the kind a cosine score misses and a
reader catches. A vector index is also a second source of truth to keep in sync; here
the index *is* the artifact, human-readable and hand-editable.

**Where it stops scaling.** When the index no longer fits comfortably in the window.
Around 500 memories on a 131k window, at which point the answer is not a vector store
but a two-level index (sections, then lines) or a second kura — which is one reason
stores are cheap to create.

**What happens when the model is down.** Recall falls back to word overlap and labels
the result `how=words`. It never returns silence and never pretends the answer came
from meaning.

---

## 2. The gate: why a model may not decide what is true

**The failure.** The distiller recorded the agent's own assertion as a fact. The next
agent read it back as ground truth and restated it with more confidence, which the
distiller recorded again. Confabulation compounds. Turning the temperature down and
strengthening the instruction did not stop it.

**The mechanism.** Every candidate memory must carry quotes that exist
character-for-character in the raw material, each tagged with its evidence class. The
check is `substring in text`. A fabricated quote cannot pass it, whatever the model's
confidence; a paraphrase cannot pass it either, which is deliberate — a paraphrase may
be perfectly faithful, but it is not checkable, so it is treated as invention.

What survives determines what may be claimed:

| surviving classes | licence |
|---|---|
| `TOOL` or `ACT` present | numbers may be written |
| `USER` present | the human may be credited with a decision |
| `SELF` only | it is a judgement, and must say so in the first person |
| nothing | discarded |

**Echo suppression.** A quote that already appears verbatim in the store is dropped as
an echo. Journals contain tool results from *reading the store*; without this, a memory
system rediscovers and re-records its own contents forever.

**The idea hatch, and the hole in it.** Ideas need no quotes — a new thought is by
definition absent from the material, and requiring evidence would either kill it or
push the model to disguise it as a fact. Ideas go to a seed file, never the store.
Observed in the wild: factual reports relabelled `kind: idea` to skip verification
("the user approved X", no quotes). Anything of that grammatical shape is now dropped.

**The last check before writing.** Prompt instructions get broken, so the composed text
is scanned mechanically: if it credits the human with a decision and no `USER` quote
survived, the draft is flagged and can never be poured.

---

## 3. Two models, three roles

| role | frequency | needs |
|---|---|---|
| thinker | every recall | speed; relevance judgement by meaning |
| brain | per distil batch | context length; reads a whole batch at once |
| scribe | per memory, per draft | prose in the target language; judgement |

Default is one model in all three chairs. The roles exist so each can be upgraded
independently — the original deployment ran a fast GPU model as brain and a very slow,
very large CPU-only model as scribe: they never contend for the same hardware, so both
can run at once, all night.

**The shared charter is a speed feature.** The same text, byte for byte, heads every
role's system prompt. Measured on a CPU-only model: a 1,225-token preamble cost 74.7 s
on first sight and 4.9 s on the second — the server keeps a common prefix in the slot.
Three differently-worded preambles paid that cost three times. Giving every worker the
same understanding of the household and making the system faster turned out to be the
same act.

**All effort dialects, always.** `reasoning_effort`, `thinking_effort`,
`enable_thinking` are sent together; templates ignore what they do not know. A model
left on its default deep-thinking setting can consume the whole `max_tokens` budget on
reasoning and return an empty string — which shows up not as an error but as "the
distiller found nothing today", intermittently. That is the worst kind of bug.

---

## 4. Watermarks, and drinking the same water twice

**The failure.** Two distillers in parallel. Each read the marks file, each wrote its
own version back, and the later write erased the other's progress. The same stretch of
journal was processed a dozen times, producing duplicate memories that then had to be
de-duplicated by hand.

**The mechanism, both halves:**
1. `flock` around read-modify-write, so the two never interleave.
2. `max()` on merge, so a stale value cannot pull a mark backwards.

A lock alone is not enough: a runner holding an old snapshot still writes a smaller
number legally. Merge-forward is what makes it safe.

**Claim before drinking.** The stretch is reserved (mark moved) *before* it is read.
Advance-after-read leaves a window in which a parallel runner starts at the same offset.

**Units are the adapter's business.** Append-only transcripts use byte offsets.
Archives that get rewritten — a compressed session log, for instance — use the event
sequence number, because a byte offset into a recompressed file is simply a lie. The
watermark key must also be unique per source: every DSH session file is literally named
`session.jsonl.zstd`, so the key carries the directory.

---

## 5. Drafts, and who is allowed to be the bottleneck

Composing a memory does not store it. Drafts land in `_still/drafts/` with their
evidence attached in an HTML comment — you can always see *why* a memory exists.

The temptation is to have a person approve each draft. That quietly makes the person a
required step in a loop that runs every night, and drafts pile up unread. So the scribe
re-reads each draft cold, with the evidence, and returns POUR / FIX / TOSS — and is told
plainly not to be afraid to TOSS, because a store is worth what it returns when queried,
not what it weighs. Tossed drafts are logged, not deleted into nothing.

**Nothing in an autonomous loop may require someone who is not always present.** A
person can still read drafts, override, and pour by hand; the loop just does not wait
for them.

---

## 6. The index line is the product

The index is the only thing read *in full, every time*. A body is opened when needed.
So an index line is not a summary — it is a recognition trigger, and its quality is the
quality of the whole system.

Strong: proper nouns, numbers, ⚠️ landmines, the conclusion reached.
Weak: "about X", "notes on X", "important findings" — phrases that fit any memory.

The test: **if a line could swap places with another memory's line and still read
fine, it is not doing its job.**

This decays without maintenance. `kura distill tidy` finds the mechanically detectable
rot — a title that is a truncated description, a trigger too short to recognise, a
title left as the raw slug — and has the scribe rewrite those lines from the body. It
is a metabolism, not a one-time cleanup: hand-fixing ten lines proves the rot returns.

---

## 7. Several kura, one server

A single memory serving both "help me build this" and "help me think this through"
serves neither: the recall that helps you debug is noise in a conversation about what
to do next, and the reverse is worse.

A store is a directory. A mode maps to a store. They share no memories, no index, and
no distiller watermark — so a mode switch genuinely changes what is remembered, not
merely the tone of the answer.

The host binds mode to store. In DSH, an agent preset carries both the persona and the
kura binding, so one preset change moves the whole self. This project deliberately does
not own the persona side: a second identity mechanism living here would drift out of
step with the host's, and the host is the one that actually decides who is speaking. We
carry a pointer (`persona = "..."`, readable at `GET /profile`) and nothing more.

---

## 8. What this is not

- **Not a RAG pipeline.** Nothing is chunked, embedded or scored. The unit is a fact a
  person could read aloud.
- **Not a knowledge graph.** `[[links]]` are hand- or model-written associations walked
  breadth-first, with no ontology and no inference.
- **Not an archive.** The archive is the journal. This holds what was distilled *out* of
  it — which is why TOSS is a good outcome and growth is not a metric.
