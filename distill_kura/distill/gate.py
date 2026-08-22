"""The gate — the one place a model never touches.

A candidate arrives with quotes. We check, by plain substring match, that each quote
really exists in the raw material of the class it claims. A fabricated quote cannot
pass a substring check, no matter how confident the model is. Everything downstream
(the writer, the pourer) is a model again; this is the deterministic floor under them.

Four things happen here:

1. **Quote verification.** No surviving quote → the candidate is dropped.
2. **Echo suppression.** A quote that already exists verbatim in the store is not new
   material — it is the store reading itself back through a tool result. Without this,
   a store re-discovers and re-records its own contents forever.
3. **Class arithmetic.** Which classes survived decides what may be claimed:
   [TOOL]/[ACT] present → grounded; [USER] present → attributable to the human;
   [SELF] only → it is a judgement and must say so.
4. **The idea escape hatch, closed against smuggling.** Ideas need no quotes (a new
   thought is by definition not in the material) — but "the human approved X" is a
   factual report wearing an idea's coat, and is dropped.
"""
from __future__ import annotations

import re

from ..store import InvalidTag, normalize_tags
from .sources import Segment

MAX_QUOTE = 400
MIN_QUOTE = 12

# "the human decided/approved/said …" — a factual report, not an idea.
_FACT_IN_IDEA_CLOTHING = re.compile(
    r"\b(user|human|ken|owner|they)\b.{0,24}\b(proposed|approved|decided|asked|chose|said|"
    r"confirmed|requested|wants|noted|instructed)\b", re.I)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def gate(cands: list[dict], segs: list[Segment], store_text: str = "") -> tuple[list[dict], list[dict], list[dict]]:
    """→ (kept, dropped, ideas). `store_text` is the normalised full text of the store."""
    hay = {c: norm("\n".join(s.text for s in segs if s.cls == c))
           for c in ("USER", "TOOL", "ACT", "SELF")}
    kept: list[dict] = []
    dropped: list[dict] = []
    ideas: list[dict] = []

    for c in cands:
        if str(c.get("kind", "")).lower() == "idea":
            blob = f"{c.get('topic', '')} {c.get('why', '')}"
            if _FACT_IN_IDEA_CLOTHING.search(blob):
                dropped.append({**c, "why_dropped": "a factual report dressed as an idea"})
                continue
            ideas.append(c)
            continue

        good: list[dict] = []
        classes: set[str] = set()
        echoed = 0
        for q in (c.get("quotes") or [])[:6]:
            q = str(q)
            m = re.match(r"\[(USER|TOOL|ACT|SELF|KEN|ME)\]\s*(.*)", q, re.S)
            claimed, body = (m.group(1), m.group(2)) if m else (None, q)
            claimed = {"KEN": "USER", "ME": "SELF"}.get(claimed, claimed)   # legacy tags
            body = norm(body)[:MAX_QUOTE]
            if len(body) < MIN_QUOTE:
                continue
            found = [k for k, v in hay.items() if body and body in v]
            if not found:
                continue                       # not in the material: fabricated or paraphrased
            if store_text and body in store_text:
                echoed += 1                    # the store reading itself back
                continue
            cls = claimed if claimed in found else found[0]
            good.append({"class": cls, "text": body})
            classes.add(cls)

        if not good:
            dropped.append({**c, "why_dropped": ("echo of text already in the store" if echoed
                                                 else "quotes not found in the raw material")})
            continue

        judgement = classes == {"SELF"}
        if judgement and not re.search(r"judge|judgement|judgment|opinion|read it|見立て|判断",
                                       f"{c.get('why', '')} {c.get('topic', '')}", re.I):
            dropped.append({**c, "why_dropped": "turning the agent's own words into a fact"})
            continue

        claims_number = bool(re.search(r"\d", f"{c.get('why', '')} {c.get('topic', '')}"))
        grounded = bool(classes & {"TOOL", "ACT"})
        kept.append({**c,
                     "evidence": good,
                     "classes": sorted(classes),
                     "judgement": judgement,
                     "unverified_numbers": claims_number and not grounded})
    return kept, dropped, ideas


# ── tags: a model proposes, the evidence decides ─────────────────────────
#
# A tag is a word, but some words are claims. `entrusted` says the human asked for this
# to be kept; `emotion-carried` says they reacted; `recurred` says they brought it up
# again. A model that could attach those freely could immortalise whatever it liked —
# so each claiming tag needs the class of evidence that would make it true, checked
# here and recorded in the manifest. The seven words reserved for a future forgetting
# pass are refused outright: no model assigns them until that pass is designed.
#
# Everything else — `decision`, `landmine`'s sibling `hypothesis`, a store's own words —
# is curation, not a claim about the human, and passes as proposed.
_ENTRUST = re.compile(
    r"(remember (this|that|it)|don'?t forget|do not forget|keep (this|that|it) in mind|"
    r"write (this|that|it) down|覚えて|忘れないで|記憶して|メモして|覚えておいて|覚えといて)", re.I)
FORGETTING_TAGS = frozenset({"superseded", "absorbed", "fulfilled", "expired",
                             "corrected", "released", "incidental"})


def verify_tags(proposed, evidence: list[dict], recurred_ok: bool = False
                ) -> tuple[tuple[str, ...], dict[str, dict], dict[str, str]]:
    """→ (kept, basis, refused).

    `basis[tag]` names the evidence a kept claiming tag rests on, so the manifest can
    say why the tag exists. `refused[tag]` says why one did not make it — written to
    the manifest too, because a silently dropped tag looks like one never proposed.
    `recurred` is decided by the caller (it needs a prior memory and a different
    occasion, which a single candidate cannot see) and passes only with `recurred_ok`.
    """
    try:
        tags = normalize_tags(proposed)
    except InvalidTag as e:
        return (), {}, {"*": str(e)}
    classes = {e["class"] for e in evidence}
    user_quotes = [e["text"] for e in evidence if e["class"] == "USER"]
    kept: list[str] = []
    basis: dict[str, dict] = {}
    refused: dict[str, str] = {}
    for t in tags:
        if t in FORGETTING_TAGS:
            refused[t] = "reserved for the forgetting pass; a model may not assign it"
        elif t in ("emotion-carried", "commitment"):
            # Both are claims about the human — what they felt, what they undertook
            # — and neither can rest on tool output or the agent's prose.
            if user_quotes:
                kept.append(t); basis[t] = {"class": "USER", "quote": user_quotes[0]}
            else:
                refused[t] = "needs the human's own words; no [USER] quote survived"
        elif t == "entrusted":
            q = next((q for q in user_quotes if _ENTRUST.search(q)), None)
            if q:
                kept.append(t); basis[t] = {"class": "USER", "quote": q}
            else:
                refused[t] = "needs a [USER] quote that asks for this to be kept"
        elif t == "recurred":
            if recurred_ok:
                kept.append(t); basis[t] = {"class": "USER", "quote": user_quotes[0] if user_quotes else ""}
            else:
                refused[t] = "recurrence is decided against a prior memory, not proposed"
        elif t in ("formative", "landmine"):
            if classes - {"SELF"}:
                kept.append(t); basis[t] = {"class": sorted(classes - {"SELF"})[0]}
            else:
                refused[t] = "needs more than the agent's own prose"
        else:
            kept.append(t)
    return tuple(kept), basis, refused


def salvage(raw: str) -> list[dict]:
    """Recover complete objects from a truncated JSON array.

    Readers copy quotes generously and run out of budget mid-array. The array will not
    parse, but every closed `{...}` inside it is still valid — take those.
    """
    out: list[dict] = []
    depth, start, instr, esc = 0, None, False, False
    for i, ch in enumerate(raw or ""):
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    o = __import__("json").loads(raw[start:i + 1])
                    if isinstance(o, dict) and (o.get("quotes") or
                                                str(o.get("kind", "")).lower() == "idea"):
                        out.append(o)
                except ValueError:
                    pass
                start = None
    return out


def attributes_to_human(text: str, classes: list[str], words: list[str] | None = None) -> bool:
    """True when prose credits the human with a decision but no [USER] quote survived.
    Prompt instructions get broken; this is checked mechanically instead."""
    if "USER" in classes:
        return False
    pat = words or [r"ケン(は|が|の指示|の決裁|さんが)", r"\b(the user|the human|the owner)\b\s+\w*\s*"
                                                        r"(decided|asked|approved|chose|instructed|said)"]
    return any(re.search(p, text, re.I) for p in pat)
