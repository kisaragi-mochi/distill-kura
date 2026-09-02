"""The adaptive floors, one candidate at a time, with the reason spelled out.

Extracted verbatim from `Adaptive.floor` (W2a): same checks, same order, same
reason strings. `Adaptive.floor` delegates here; nothing else changed.
"""
from __future__ import annotations

import re
import unicodedata

from .distill.gate import attributes_to_human, composed_number_violations
from .weave import DEAD_WORDS, MARKERS

# Meaning flips cheaply while 2-grams stay put. Blunt and two-directional on purpose.
_NEG_EN = ("not", "no", "never", "none", "without", "cannot", "can't", "don't",
           "doesn't", "isn't", "aren't", "won't", "wasn't", "didn't", "must not")
_NEG_JA = ("ではない", "するな", "しないで", "ません", "ない", "ぬ", "ず", "なく", "せず",
           "禁止", "不可", "未", "非", "不")
# A line that says a thing is over must not shrink into a line that says it is current.
_OBSOLETE = ("退役", "畳んだ", "撤退", "廃止", "廃案", "封印", "superseded", "deprecated",
             "retired", "abandoned", "obsolete", "撤収")
_LINK = re.compile(r"\[\[([^\[\]\n]+)\]\]")
_NUM = re.compile(r"[+-]?\d+(?:\.\d+)?[eE][+-]?\d+|[+-]?\d[\d,.:/-]*\d|[+-]?\d")
_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_+./-]{2,24}")
_ARROW = re.compile(r"(\S+)\s*(?:→|->|⇒)\s*(\S+)")


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _negations(text: str) -> set[str]:
    t = text.lower()
    found = {w for w in _NEG_EN if re.search(rf"(?<![a-z']){re.escape(w)}(?![a-z'])", t)}
    found |= {w for w in _NEG_JA if w in text}
    return found


def _bound_numbers(text: str) -> set[str]:
    """Each number tied to what it measures: the token glued to its RIGHT ("43.7 t/s",
    "8枚"), and the token glued to its LEFT only when nothing separates them ("TP=8",
    "v2"). A number that opens a cue has no left neighbour and must not be blamed for
    it; a digit inside an identifier (ds4-tp8) is not a measurement at all."""
    out = set()
    for m in _NUM.finditer(text):
        a, b = m.start(), m.end()
        before = text[a - 1] if a else ""
        after = text[b] if b < len(text) else ""
        if (before.isalpha() and before.isascii()) or (after.isalpha() and after.isascii() and
                                                        (before in "-_/" or (a and text[a-1].isalnum()))):
            continue                         # part of an identifier, not a quantity
        right = re.match(r"[^\s、。,;:)）」』]*", text[b:]).group(0)
        if right and right[0] in "-_./":     # ds4-tp8 style tails are identifiers too
            right = ""
        left = ""
        if a and not text[a - 1].isspace():
            m_left = re.search(r"[A-Za-z_=+#]+$", text[:a])
            left = m_left.group(0) if m_left else ""
        out.add(f"{left}|{m.group(0)}|{right}")
    return out


def _idents(text: str) -> set[str]:
    """Code-like tokens only — anything with a digit, an inner -_./+, or two capitals.
    Plain words are not identifiers: a shorter cue may drop or add a function word,
    but it may never cut `ds4-tp8-engine-canonical` down to `ds4-tp8`."""
    out = set()
    for i in _IDENT.findall(text):
        i = i.rstrip("./-+")
        if (any(ch.isdigit() for ch in i) or re.search(r"[A-Za-z0-9][-_./+][A-Za-z0-9]", i)
                or sum(1 for ch in i if ch.isupper()) >= 2):
            out.add(i)
    return out


def first_violation(cand: str, title: str, desc: str, loom) -> str | None:
    """→ None when the candidate may be worn; else the FIRST reason it may not.
        Reuses the production floors; adds only what a shorter cue newly risks."""
    t = _nfkc(cand).strip()
    title_n, desc_n = _nfkc(title), _nfkc(desc)
    src = f"{title_n} {desc_n}"
    if not t:
        return "empty"
    if t.lower() in DEAD_WORDS:
        return "dead word"
    if t.strip("*`★⚠️ 　").lower() == title_n.strip().lower():
        return "restates the title"
    bad = composed_number_violations(t, [{"text": src}])
    if bad:
        return f"invented number: {', '.join(bad)}"
    bound_src = _bound_numbers(src)
    for b in _bound_numbers(t):
        if b not in bound_src:
            return f"number re-bound: {b.strip('|')}"
    arrows_src = set(_ARROW.findall(src))
    for a in _ARROW.findall(t):
        if a not in arrows_src:
            return f"arrow reversed or invented: {a[0]}→{a[1]}"
    if attributes_to_human(t, []) and not attributes_to_human(src, []):
        return "credits the human where the line does not"
    for mark in MARKERS:
        if mark in t and mark not in src:
            return f"invented marker {mark}"
    have = {_nfkc(x) for x in _LINK.findall(desc)}
    for name in _LINK.findall(t):
        if _nfkc(name) not in have:
            return f"invented link [[{name}]]"
    idents_src = _idents(src)
    for ident in _idents(t):
        if ident not in idents_src:
            return f"cut or invented identifier: {ident}"
    neg_c, neg_s = _negations(t), _negations(src)
    if neg_c - neg_s:
        return "negation invented: " + ", ".join(sorted(neg_c - neg_s))
    if neg_s and not neg_c:
        return "negation dropped"
    low_src, low_t = src.lower(), t.lower()
    if any(w in low_src for w in _OBSOLETE) and not any(w in low_t for w in _OBSOLETE):
        return "retirement word dropped"
    if not loom._grounded(t, desc_n):
        return "ungrounded (2-gram overlap below the floor)"
    return None
