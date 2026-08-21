"""Recall = recognition, not search.

The whole index (one line per memory, ~50 tokens each) is handed to the thinker,
which names the memories that *bear on* the question by meaning. Then we walk
[[links]] from the picked memories and return the neighbourhood as context.

Why not grep / embeddings: a question about "SSD inference chips" shares no word
with a memory titled "run the 2.6T model from an SSD tier", yet they are the same
problem. A resident index + a small model recognises that in ~0.4 s. If the
thinker is unreachable we fall back to word overlap and *say so* in `how`.
"""
from __future__ import annotations

import json
import re
import time

from .store import Store
from .thinker import Endpoint

PICK_SYS = (
    "You are the long-term memory of {label}. Below is the INDEX of everything remembered — "
    "one line per memory, written as a recognition trigger, not a title.\n\n"
    "Given the question, name the memories that genuinely bear on it. Judge by MEANING, not by "
    "shared words: a question about an outside topic is relevant when this store is working on "
    "the same problem, even if no word matches. Prefer few and right over many.\n"
    "Output ONLY a JSON array of file slugs (no .md, no path). Empty array if nothing truly "
    "relates.\n\n=== INDEX ===\n"
)


def pick_by_meaning(store: Store, thinker: Endpoint, question: str, top: int) -> list[str] | None:
    raw = thinker.ask(PICK_SYS.format(label=store.label) + store.index_text(), question,
                      max_tokens=500)
    if raw is None:
        return None
    m = re.search(r"\[.*?\]", raw, re.S)
    if m:
        try:
            got = json.loads(m.group(0))
            picked = [str(x) for x in got if isinstance(x, str)][:top]
            if picked:
                return picked
        except ValueError:
            pass
    # Deterministic net: whatever the format, a real slug in the answer is a pick.
    hits: list[str] = []
    for slug in store.known_slugs():
        if slug in raw and slug not in hits:
            hits.append(slug)
    return hits[:top]


def pick_by_words(store: Store, question: str, top: int) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9ァ-ヴー一-龠]{2,}", question)
    scored = []
    for line in store.index_text().splitlines():
        m = re.search(r"\(([^)]+)\.md\)", line)
        if not m:
            continue
        s = sum(line.lower().count(t.lower()) for t in terms)
        if s:
            scored.append((s, m.group(1)))
    scored.sort(reverse=True)
    return [s for _, s in scored[:top]]


def fit(text: str, question: str, budget: int) -> str:
    """Trim a memory to `budget` chars WITHOUT cutting from the top: keep the
    frontmatter + opening, then prefer paragraphs that mention the question's words.
    (Long memories keep their conclusions at the bottom; head-truncation loses them.)"""
    if len(text) <= budget:
        return text
    head_end = text.find("\n\n", text.find("---", 4) + 3)
    head = text[:max(0, head_end)][:600] if head_end > 0 else text[:600]
    terms = [w for w in re.findall(r"[A-Za-z0-9]{2,}|[ァ-ヴー]{2,}|[一-龠]{2,}", question)]
    rest = text[len(head):]
    paras = [x for x in rest.split("\n\n") if x.strip()]
    if any(len(x) > budget // 3 for x in paras):      # giant paragraph → split by line
        split: list[str] = []
        for x in paras:
            split += [ln for ln in x.split("\n") if ln.strip()] if len(x) > budget // 3 else [x]
        paras = split
    scored = sorted(((sum(p.lower().count(t.lower()) for t in terms), i, p)
                     for i, p in enumerate(paras)), key=lambda x: (-x[0], x[1]))
    keep, used = [], len(head)
    for _, i, para in scored:
        if used + len(para) + 2 > budget:
            continue
        keep.append((i, para)); used += len(para) + 2
    keep.sort()
    return head + "\n\n" + "\n".join(p for _, p in keep) + ("\n\n…(trimmed)" if used < len(text) else "")


def recall(store: Store, thinker: Endpoint | None, question: str, hops: int = 1,
           top: int = 3, chars: int = 6000) -> dict:
    t0 = time.time()
    picked = pick_by_meaning(store, thinker, question, top) if thinker else None
    if picked is None:
        picked, how = pick_by_words(store, question, top), "words(thinker unreachable)"
    elif not picked:
        picked, how = pick_by_words(store, question, top), "meaning→none→words"
    else:
        how = "meaning"
    order = store.walk(picked, hops)
    ctx = "\n\n".join(f"=== {store.label}: {s} ===\n{fit(store.read(s), question, chars)}"
                      for s in order)
    store.note_read(order, "recall")
    return {"store": store.name, "question": question, "how": how, "picked": picked,
            "walked": order, "context": ctx, "chars": len(ctx),
            "elapsed_s": round(time.time() - t0, 2)}
