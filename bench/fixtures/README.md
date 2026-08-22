# Retention fixtures

A synthetic journal with facts planted in it on purpose, and the questions those facts
answer. Synthetic because a retention benchmark has to be publishable: a real transcript
cannot be, and a benchmark nobody can run is a claim, not a measurement.

Each planted fact carries a category, because they fail differently:

| category | why it is here |
|---|---|
| `decision` | the thing a memory store exists for |
| `number` | must survive exactly, or not at all |
| `negation` | "we are NOT doing X" — the easiest thing to drop and the worst to lose |
| `reversal` | a decision changed later; keeping only the first is worse than keeping neither |
| `conditional` | a rule with an exception attached |
| `chronology` | which came first |
| `landmine` | a failure that will recur |
| `returning` | a topic raised again after a gap |
| `distractor` | **must NOT be stored** — chit-chat, or something a config file already records |

`expect` is a literal string or a regex that must appear in what recall returns. Matching
a marker rather than judging prose keeps the score model-free and reproducible: it
measures whether the fact is *findable*, which is the coverage question. It does not
measure whether the answer reads well — that needs a judge, and a judge needs a model.
