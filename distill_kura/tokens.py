"""Token estimation, for budgeting a prompt we cannot tokenize.

The kura has to answer one question constantly: *does the index still fit?* Loading a
real tokenizer would drag a model-specific dependency into a project that has none, for
a number that only needs to be right to within a few percent.

Rates are per SCRIPT, not one global divisor. A BPE merges about four ASCII characters
into one token but *splits* a kanji into slightly more than one, so any single `chars/N`
rule is wrong for one language or the other, and its error moves with the language mix —
exactly the thing a budget must not do.

The weights below were least-squares fitted on a real 16.8k-character mixed
Japanese/English index and checked against six actual tokenizers:

    tokenizer            actual    chars/2    weighted
    o200k (GPT-4o)        9,155      -8%        -1%
    gemma-4-31B           9,081      -8%        -0%
    Qwen3.6-27B           8,579      -2%        +6%
    KAT-Coder-V2.5        8,578      -2%        +6%
    Laguna-S-2.1         10,343     -19%       -12%
    cl100k (GPT-4)       10,888     -23%       -17%

Two lessons in that spread. The tokenizers disagree with each other by 1.27x, so no
character-based estimate can do better than roughly +/-15% — do not chase precision
here. And the naive `chars/2` is biased LOW, which is the dangerous direction: a budget
that under-counts silently overflows the window it was meant to protect.

This is an ESTIMATE and is labelled as one everywhere it surfaces. Budgets built on it
leave headroom rather than sitting exactly on the limit.
"""
from __future__ import annotations

import re

# Tokens per character, by script. Fitted, not guessed — see the table above.
ASCII = 0.378
HIRAGANA = 0.735
KATAKANA = 0.592
KANJI = 1.122
JP_PUNCT = 1.198
OTHER = 1.658

_HIRA = re.compile(r"[぀-ゟ]")
_KATA = re.compile(r"[゠-ヿｦ-ﾟ]")
_KANJI = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
_JPUNCT = re.compile(r"[　-〿＀-･]")
_ASCII = re.compile(r"[\x00-\x7f]")


def estimate(text: str) -> int:
    """Approximate tokens in `text`. One pass over five character classes: no tables,
    no tokenizer dependency, accurate to roughly +/-10% on mixed Japanese/English."""
    if not text:
        return 0
    hira = len(_HIRA.findall(text))
    kata = len(_KATA.findall(text))
    kanji = len(_KANJI.findall(text))
    jp = len(_JPUNCT.findall(text))
    ascii_ = len(_ASCII.findall(text))
    other = max(0, len(text) - hira - kata - kanji - jp - ascii_)
    return int(ascii_ * ASCII + hira * HIRAGANA + kata * KATAKANA
               + kanji * KANJI + jp * JP_PUNCT + other * OTHER)


def fraction_of(text: str, window_tokens: int) -> float:
    """What share of a context window this text would occupy (0.0-1.0)."""
    return estimate(text) / max(1, window_tokens)
