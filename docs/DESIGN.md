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

**Reservation is consumption.** `claim` writes the end `claim_bound` returns, and
`advance` only takes `max()`. If that end is past the last byte `sip` will actually
drink — a partial tail, or a char-budget stop mid-record — the unread stretch is
skipped forever. Sources that walk records reserve the same complete-record end `sip`
returns.

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

## 6b. The resident map: recall you did not have to ask for

**The failure.** Tool-based recall only fires once the agent decides to ask. Asked about
something the household had measured, an agent that had not thought to call the tool
answered from general knowledge — fluently, and wrongly. It had no way to know there was
anything to look up. The gap is not retrieval quality; it is that the agent cannot see
the shape of what it knows.

**The mechanism.** The index is also worn: a standing block in the system prompt, every
turn. That is affordable exactly when the index is small — the spring-source condition,
measured at 4.6% of the window when this was first built (it has since grown to 6.9%,
which is what forced the loom).

**Three layers, and why not just "trim everything".** A blind A/B test (20 questions,
fat vs slimmed index, scored without knowing which was which) came out 9–11 overall — a
coin toss — but the bands told the real story: detail won for *recent events* (4–1) and
lost everywhere else, including doctrine (1–4) where the lines were byte-identical in
both indexes. A lighter surround makes the standing lines work better. So: doctrine and
recent memories keep their full line; everything else is compressed to a trigger.

**Why age is not mtime.** `cp -r`, a restore, a checkout, a bulk `sed` — every timestamp
resets, the whole index becomes "fresh", nothing is trimmed, and the mechanism has
switched itself off without a word. Measured on the live store: 50 of 214 files shared a
single bulk-touch day, and for those mtime understated the true age by a median of 11
days (worst case 425). So the loom prefers a date written inside the memory, and
distrusts any mtime shared by a fifth of the store on one calendar day. A date more than
a day in the future is a plan, not a stamp — but *exactly* one day of slack is allowed,
because "today" is written in local time and at 06:00 in Tokyo the UTC date is still
yesterday. Without that slack the freshest memories are discarded every morning.

**Where the block goes is a cache decision.** A prefix cache is lost from the first
changed byte onward: identical preamble 0.14 s, appended-at-the-end 0.14 s, one word
added at the front 0.66 s. The persona commonly embeds a clock, so it changes every
minute; the map is the largest block and changes a few times a day. The big stable thing
goes in front of the thing that ticks — `promptOrder: -50`, before the persona. And the
block itself contains nothing volatile; `build()` refuses a header carrying a date or a
clock at build time.

**Never half a map.** Over the soft budget, the whole map is still emitted and the
warning goes in the JSON, never in the text — a banner inside the block is volatile
content and costs the prefix. Over the hard ceiling, the block becomes a stub with no
index lines at all. Truncating to fit would be the worst possible artifact: it looks
complete, and every memory below the cut appears not to exist.

**And two bugs inherited from the first implementation**, both of which failed silently
and are now impossible by construction:

- *The loom read its own output.* It preferred the woven file as its source when one
  existed, so it re-wove its own cloth and could never see a new memory. On the live
  store, 41 of 129 memories — doctrine included — had been missing from a healthy-looking
  cloth for 11 days. The source is now always the canonical index, and a loom pointed at
  it as an output refuses to construct.
- *The cloth lived in the store.* Written beside the memories, it was picked up as one.
  It belongs in `_still/`, the workshop, which is never walked.

The postcondition that makes this class of bug unshippable: **the cloth must name
exactly the same memories, in the same order, as the index it came from.** Checked on
every weave, raising if not. Compression may shorten a description; it may never lose,
reorder or invent a link.

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

## 8. The room decides what matters — and what a full shelf means

### One principle, four questions

What to remember, how to write it, what to recall and — one day — what to let go are
not four ranking problems. They are one question asked four times: *what is this store
for, and which side of this memory serves that?* The answer lives in the store's
charter, which is why the charter sits byte-identically at the head of every role's
prompt. The same afternoon yields a Research memory ("what became known"), a Develop
memory ("how it was made to work"), a Manage memory ("what it changed in the plan") and
an EQ memory ("what was felt, and what settled it"). These are different facts about
one event, and none of them is the "real" one.

The floor does not move with the room. A quote exists verbatim or the candidate dies;
a number comes from `[TOOL]` or it is not written; a decision is the human's words or
it is not a decision; the agent's prose is a judgement and is written as one; nothing
crosses a store boundary. What changes by room is not what is true but which aspect of
the truth is kept, and for what.

### Why there is no universal priority

An earlier SPOT prompt ranked, for every store alike: decisions first, then emotion,
then topics returned to, then numbers and landmines, then doctrine. It was not wrong
about what to notice; it was wrong about who decides. Emotion is the *reason* an EQ
store exists and a *distraction* in a Develop store. A topic returned to is the heart
of a USER store and noise in a Research store that already has the result. A list that
puts emotion at #2 everywhere makes four rooms into one room with four doors.

So emotion, recurrence, numbers and reversals are now *things not to walk past* —
observations — and the charter ranks. A candidate must say why it belongs in *this*
store, and a sentence that would fit any store ("it is important") is treated as "it
does not belong anywhere yet".

### The room is the person's choice

A store is chosen before the conversation, by the host, and is the whole session's
home. The model does not read the first message and pick a room; it does not move the
session when the topic drifts; it does not file a memory into a different store because
a tag suggests one. A Develop memory tagged `emotion-carried` is a Develop memory.
There is no move, no copy and no re-filing, and a mode change affects only future
sessions. When the same topic is raised in another room, that room distils its own
memory from its own evidence. This looks like duplication and is not: two rooms, two
facts.

### Tags are words, not weights

A memory may carry several tags. They describe its character — `decision`, `landmine`,
`entrusted`, `emotion-carried`, `recurred`, `settled`, the store's own words — and
nothing ranks, sorts or counts by them. Two things keep that honest:

1. **A claiming tag needs the evidence that would make it true.** `entrusted` says the
   human asked for this to be kept, so it needs a `[USER]` quote that asks.
   `emotion-carried` needs the human's words at all. `landmine` and `formative` need
   more than the agent's prose. The check is deterministic (`gate.verify_tags`), and
   both the basis and every refusal go into the evidence manifest.
2. **`recurred` is decided, not proposed.** A model proposing it would be a popularity
   counter in disguise. It is written once, by the distiller, when a candidate covered
   by an existing memory carries the human's own words from a *different journal* than
   the memory was distilled from. A memory with no manifest has no known origin, and is
   left alone — logged, not guessed. There is no `recurrence_count`, and a second
   recurrence does not touch the file.

Seven further words — `superseded`, `absorbed`, `fulfilled`, `expired`, `corrected`,
`released`, `incidental` — are reserved for a forgetting pass that does not exist yet,
and a model may not assign them.

Beside the tags, three sentences: **belongs_because** (why this store wants it),
**keep** (the meaning that must outlive any thinning) and **may_fade** (the detail that
need not). They are the scribe's curation judgement against the charter, not new facts,
and they decide nothing today. They are written now so that when a shelf is full, the
question "what does this memory still do here?" already has a first answer on file.

### The wide room

Four narrow rooms with fixed charters recognise sharply. A USER room, whose charter is
"understand the person, without a narrower purpose", cannot be sharp in the same way,
and the design accepts that: its recall is expected to be a little softer, and in
exchange it is the one room whose understanding grows. The growth is a `profile.md`
beside the charter — enduring threads, current interests, everyday context,
conversation preferences, unresolved threads — in sentences. It is read after the
charter by that store's distiller, so what the room keeps from then on is shaped by
what it has come to understand; it never enters the resident map; it is store-local
and drafted from the store's own memories; a draft is a file a person reads, and
applying it is a person's act. A profile that carries numbers about how much things
matter is the weight this project refuses to store, and is reported as broken and not
read. When, or whether, a draft should be applied without a person is a decision to
make with real drafts in hand.

### Garage-sale forgetting (not built)

Forgetting here is not a function of age or of how rarely something is read. It is a
consequence of a finite shelf: when the 101st memory arrives at a store that holds a
hundred, *all 101* are asked the same question — *is there still a reason for you to
be here?* — and the one with the weakest answer against the charter is the candidate.
Neither the old hundred nor the newcomer has a claim by default; not storing the
newcomer is a correct outcome. And the absence of a protecting tag is not a reason to
forget: forgetting needs a positive account — "this has done its work" — which is what
the reserved words above are for.

The shapes it might take: KEEP; SETTLE (thin the detail, keep the meaning — `may_fade`
says which is which); ABSORB (fold the meaning into another memory *of the same
store*); GARAGE (out of the active index, into a grace state); RELEASE (out of the
active memory, with an explicit reason). MOVE is not on the list and will not be.

What exists today is the observation point: `kura doctor` reports capacity in four
units side by side — memories, index tokens, body tokens, bytes — with `limit` and
`pressure` left `None`. Which unit a shelf is measured in, what the limit is, whether
the wide room shares it, how candidates are compared, where the garage is and how long
grace lasts, whether a garaged memory can still be read by name, which tags protect
absolutely and which only argue, who approves, and what finally deletes a file — every
one of those is undecided, and will be decided with real memories, a real map, and real
mis-selections in front of the people whose memories they are. Until then, detecting a
full shelf reports it and changes nothing. The first implementation, whenever it comes,
is a dry run that shows the store, the pressure, the candidates, their tags and
sentences, the reason each could be released, what must be kept, and the proposed
action — and modifies no memory, no index, no frontmatter.

**The seam it would plug into, and the ledger it would write** (a design sketch, so
that when the pass is built it lands where the rest of the system expects it — none of
this exists in code):

- *Where a dry run reads from.* `doctor()["capacity"]` for the pressure, each memory's
  `tags()` and `annotations()` for its own account of itself, the store's charter for
  the question. Nothing else — no read log, no model-side ranking.
- *Where a proposal goes.* `_still/garage/proposals/<date>.jsonl`, one line per
  candidate: `{id, slug, index_line (verbatim bytes), section, tags, annotations,
  reason (words), proposed: KEEP|SETTLE|ABSORB|GARAGE|RELEASE, protected_by: [...]}`.
  A proposal is a file a person reads; it expires with the next night's proposal.
- *Where an act is recorded, if one is ever taken.* `_still/garage/ledger.jsonl`, one
  line per act: the proposal id, who decided (a person, never a model), the memory's
  `sha256` before, the index line removed byte-for-byte, and what would restore it.
  The ledger is append-only and is what `kura garage repair` would rebuild a
  half-applied state from. A memory that is garaged keeps its file and its slug, gains
  `state: garaged` in its frontmatter through `annotate_verified`, and leaves the index;
  `read_exact` still answers for it. Nothing in the ledger deletes.
- *What `doctor` would add.* `garaged` (state and ledger agree), `half_garaged`
  (they do not), `lost` (in no index and in no ledger — today's silent forgetting),
  and `links_released` kept apart from `links_dead`.

---

## 9. What this is not

- **Not a RAG pipeline.** Nothing is chunked, embedded or scored. The unit is a fact a
  person could read aloud.
- **Not a knowledge graph.** `[[links]]` are hand- or model-written associations walked
  breadth-first, with no ontology and no inference.
- **Not an archive.** The archive is the journal. This holds what was distilled *out* of
  it — which is why TOSS is a good outcome and growth is not a metric.
- **Not a context-stuffer.** The resident map is the *index*, never the bodies. The
  point of distilling is that the map stays small enough to carry; a design that pastes
  memories into every prompt has given up on that and will hit the window instead.
