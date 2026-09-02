"""Was a transition PROVEN — old → new, in the human's own sentence?

The retirement face rewrites the map's most-read line, so the only thing allowed to
trigger it is a human saying, in one breath, that the old thing is over AND what
takes its place. A model PROPOSING `superseded` proves nothing; the gate refusing
that proposal proves nothing either. Proposed ≠ proven.

Pure and model-free on purpose: this is the same relation for the pipeline (which
decides whether to knock) and for `Store.retire` (which must not trust its caller).

The three things ONE [USER] quote must carry, all of them, none stitched together
from two quotes:
  (i)   the OLD memory, by exact slug or exact index title;
  (ii)  an explicit replacement/retirement construction — the sentence has to SAY the
        change, not merely mention both names;
  (iii) the NEW memory, in the SAME quote: exact slug, exact title, or ≥2 whole words
        of its title/topic.
A topic-shift clause ("ところで…", "by the way …") is cut off before (ii) and (iii)
are looked for, so "old-way はやめよう。ところで GPU 温度を測ろう" can never make the
GPU memory the successor of old-way.

A quote that only retires — `やめる`, `廃止`, `stop` — without naming a successor is a
`retired-only` result: retirement is proven, succession is NOT. No caller writes a
face for it (a face without a successor is not implemented); it is returned so the
callers can say WHY they stayed silent.
"""
from __future__ import annotations

import re
import unicodedata

# ── the constructions that SAY a change, per language ───────────────────────
# Each is (name, pattern). The name is the receipt: what the human's sentence was
# read as. `…` in a construction is a gap the pattern spans loosely.
_REPLACEMENT = [
    ("やめて…で行く", r"やめ(?:て|で)[^。\n]{0,40}?(?:で|に)(?:行く|いく|する)"),
    ("に代えて", r"に代えて"),
    ("代わりに", r"代わりに"),
    ("今後は", r"今後は"),
    ("に変更", r"に(?:変更|変え)"),
    ("に置き換え", r"に置(?:き)?換え"),
    ("→", r"→"),
    ("から…へ", r"から[^。\n]{0,40}?へ(?:移|切|変|$|[^\w])"),
    ("instead of", r"\binstead of\b"),
    ("instead", r"\binstead\b(?! of)"),
    ("replace … with", r"\breplac(?:e|ed|es|ing)\b[^.\n]{0,60}?\bwith\b"),
    ("switch to", r"\bswitch(?:ed|es|ing)?\s+to\b"),
    ("now use", r"\bnow\s+(?:use|using|we use|we're using)\b"),
    ("superseded by", r"\bsuperseded by\b"),
]

# Retirement without a successor: proves the old thing is over, nothing more.
_RETIREMENT = [
    ("やめる", r"やめ(?:る|た|よう|ます|る事|ること)?"),
    ("廃止", r"廃止|廃す|打ち切"),
    ("stop", r"\bstop(?:ped|ping|s)?\s+(?:using|doing)?"),
    ("drop", r"\bdrop(?:ped|ping|s)?\b"),
    ("retire", r"\bretir(?:e|ed|es|ing)\b"),
    ("done with", r"\bdone with\b"),
]

# A clause that changes the subject can never supply the successor.
_SHIFT = re.compile(r"(ところで|別件|余談|それはそうと|by the way|anyway|"
                    r"on another note|unrelated)", re.I)

_SENT = re.compile(r"[。．.!?！？\n]")
_WORD = re.compile(r"[0-9A-Za-z_\-]+|[぀-ヿ㐀-鿿]+")
_ASCII = re.compile(r"^[0-9A-Za-z_\-]+$")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s or "")).lower()


def _clauses(text: str) -> str:
    """The quote with every topic-shift sentence removed (from the marker onward)."""
    keep = []
    for part in _SENT.split(text):
        if _SHIFT.search(part):
            break
        keep.append(part)
    return " ".join(keep)


def _names(text: str, *candidates: str) -> str | None:
    """The first candidate that appears in `text` as a whole name (already NFKC-low)."""
    for c in candidates:
        c = _norm(c).strip()
        if not c:
            continue
        pat = (rf"(?<![0-9A-Za-z_\-]){re.escape(c)}(?![0-9A-Za-z_\-])"
               if _ASCII.match(c) else re.escape(c))
        if re.search(pat, text):
            return c
    return None


def _words(*sources: str) -> list[str]:
    """The words a title/topic contributes, deduped, order kept."""
    out: list[str] = []
    for s in sources:
        for w in _WORD.findall(_norm(s)):
            if len(w) >= 2 and w not in out:
                out.append(w)
    return out


def _word_hits(text: str, words: list[str]) -> list[str]:
    hits = []
    for w in words:
        pat = (rf"(?<![0-9A-Za-z_\-]){re.escape(w)}(?![0-9A-Za-z_\-])"
               if _ASCII.match(w) else re.escape(w))
        if re.search(pat, text):
            hits.append(w)
    return hits


def _matched(text: str, table) -> list[str]:
    return [name for name, pat in table if re.search(pat, text)]


def find_transition(evidence: list[dict], old: dict, new: dict) -> dict | None:
    """Did ONE [USER] quote prove `old` → `new`?

    → `{"kind": "superseded", ...}` when all three halves are in one quote,
      `{"kind": "retired-only", ...}` when the human retired the old thing but named
      no successor, and None when nothing was proven. The quote and the matched
      constructions come back as the receipt: what sentence, read how.
    """
    old_slug, old_title = str(old.get("slug") or ""), str(old.get("title") or "")
    new_slug, new_title = str(new.get("slug") or ""), str(new.get("title") or "")
    new_topic = str(new.get("topic") or "")
    if not old_slug or not new_slug or old_slug == new_slug:
        return None
    new_words = _words(new_title, new_topic)
    retired_only = None
    for q in (evidence or []):
        if not isinstance(q, dict) or q.get("class") != "USER":
            continue
        whole = _norm(q.get("text"))
        if not _names(whole, old_slug, old_title):
            continue                       # (i) — the old memory, by name
        text = _clauses(whole)             # a "ところで" clause is not the human's ruling
        if not _names(text, old_slug, old_title):
            continue
        repl = _matched(text, _REPLACEMENT)
        retire = _matched(text, _RETIREMENT)
        if not (repl or retire):
            continue                       # (ii) — the sentence must SAY the change
        named = _names(text, new_slug, new_title)
        hits = _word_hits(text, new_words)
        if named or len(hits) >= 2:        # (iii) — and name the successor in it
            return {"kind": "superseded", "old": old_slug, "new": new_slug,
                    "quote": str(q.get("text") or ""),
                    "constructions": repl + retire,
                    "new_named_by": named or hits}
        if retire and retired_only is None:
            retired_only = {"kind": "retired-only", "old": old_slug, "new": None,
                            "quote": str(q.get("text") or ""),
                            "constructions": retire}
    return retired_only
