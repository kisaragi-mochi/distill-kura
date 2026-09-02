"""Tier zero of recall: a deterministic recognizer for DIRECT questions.

`recall` hands the whole index to a model on every question (~17k tokens of
prefill, ~900 ms). Most questions do not need a model: when someone names a
memory — its slug, its title, its own rare words — recognising it is string
work, not judgement. Five small "heads" vote over an inverted index of the
store (slug/title containment; word IDF; character 3-grams with stop-grams;
character 2-grams; the opening of the body), each normalised by the evidence
the QUESTION could have reached, and an honesty gate refuses to answer unless
the winner is both strong (top1 >= gate) and clear of the runner-up
(top1/top2 >= 1.15).

The gate is the whole point. A hit here is served with full confidence and no
model in the loop, so the design errs toward silence: a paraphrase ("SSD
inference chips" for the ssd-tier memory) scores wide and low and falls through
to the thinker, which is the tier that judges by meaning. The prototype was
blind-tested against the thinker before porting: 14/14 agreement on direct
questions, zero wrong answers, silent on every paraphrase — at ~0.5 ms.

The index is built lazily from the store's own reading APIs (index line title
and hook, frontmatter description, the first 500 chars of body, `[[link]]`
names) and cached in-process, keyed on the store's revision counter as well as
the canonical index's mtime and the memory count — so a poured memory is
recognisable on the next recall without anyone restarting anything.
"""
from __future__ import annotations

import math
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .store import LINK                  # the index-link grammar, defined once

if TYPE_CHECKING:                       # types only: this module is a layer ON TOP
    from .store import Store            # of the store API, never under it

# The heads' weights in the vote. `name` dominates because a whole slug or title
# inside the question is nearly a citation; `char2` barely counts because two
# characters match everything a little.
HEAD_WEIGHTS = {"name": 3.0, "word": 1.5, "char3": 1.0, "char2": 0.4, "body": 0.5}
# A gram in over a fifth of the store separates nothing and only drowns the honest
# ones — dropped at build. The count floor matters: in a store of four memories the
# ratio alone would stop every gram that exists, and tier zero would go mute.
STOP_DF_RATIO = 0.20
STOP_DF_MIN = 3
GATE_RATIO = 1.15       # the winner must be CLEAR of the runner-up, not just tall
DEFAULT_GATE = 1.0
# Bumped whenever heads, weights, stop-gram rules or the gate change: the adaptive
# trigger shadow keys its cache on it, so a cue judged "distinguishable" by an
# older recognizer is re-judged rather than trusted across an algorithm change.
RECOGNIZER_VERSION = 2   # 2: 'untestable' (no scoreable gram) is told apart from a sub-gate hit
BODY_CHARS = 500

_TOKEN = re.compile(
    r":[0-9]{2,5}"                    # a port (:8085) is one word
    r"|[a-z][a-z0-9]*(?:\.[0-9]+)*"   # ASCII words; qwen3.8 stays one word
    r"|[0-9]+(?:\.[0-9]+)?"           # numbers
    r"|[ァ-ヿ]+"                       # katakana runs
)


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).casefold()


def _words(s: str) -> set[str]:
    return set(_TOKEN.findall(_norm(s)))


def _grams(s: str, n: int) -> set[str]:
    t = re.sub(r"\s+", " ", _norm(s)).strip()
    if len(t) < n:
        return set()
    return {t[i:i + n] for i in range(len(t) - n + 1)} - {" " * n}


# ── building ─────────────────────────────────────────────────────────────

@dataclass
class _Index:
    stamp: tuple[int, int, int]         # (index mtime_ns, memory count, revision) built at
    built_at: float
    build_ms: float
    n: int
    mems: list[dict]                    # slug / nm_slug / nm_title
    heads: dict[str, dict]              # head → {key: (idf, [memory ids])}
    stops: dict[str, frozenset[str]]


def _stamp(store: Store) -> tuple[int, int, int]:
    """What the cache is keyed on. Every number moves when a memory is poured,
    rewritten or removed; none of them moves on a mere read.

    The revision is here because the first two are blind to a change that touches
    only a memory FILE. A body rewrite of a memory whose index line is a grouped
    family line (`- topic — [A](a.md)/[B](b.md)`) leaves the index byte-identical
    and the count the same, so the recognizer kept serving the old body and
    `doctor` called itself fresh while saying it. The store counts every committed
    mutation; asking it is the only stamp that cannot miss one."""
    try:
        m = os.stat(store.index_path).st_mtime_ns
    except OSError:
        m = 0
    return (m, len(store.slug_set()), store.revision())


def _index_lines(store: Store) -> tuple[dict[str, str], dict[str, str]]:
    """Title and hook per slug, from the index. Every link on a line counts (real
    indexes group families of memories on one line), and each linked memory hears
    the whole line's prose as its hook — the hook is the recognition trigger, and
    it triggers for everything it names."""
    titles: dict[str, str] = {}
    hooks: dict[str, str] = {}
    for line in store._uncommented(store.index_text()).splitlines():
        pairs = LINK.findall(line)
        if not pairs:
            continue
        prose = LINK.sub(lambda m: m.group(1), line).strip().lstrip("-• \t")
        for title, slug in pairs:
            slug = slug.strip()
            titles.setdefault(slug, title.strip())
            hooks[slug] = (hooks.get(slug, "") + " " + prose).strip()
    return titles, hooks


def _invert(sets: list[set[str]], n: int) -> tuple[dict, frozenset[str]]:
    """gram → (idf, [memory ids]). Grams past the stop bar are dropped but
    remembered, so the query side can keep them out of the denominator too."""
    df: dict[str, int] = {}
    lists: dict[str, list[int]] = {}
    for i, s in enumerate(sets):
        for g in s:
            df[g] = df.get(g, 0) + 1
            lists.setdefault(g, []).append(i)
    post: dict[str, tuple[float, list[int]]] = {}
    stop: set[str] = set()
    for g, d in df.items():
        if d >= STOP_DF_MIN and d / n > STOP_DF_RATIO:
            stop.add(g)
        else:
            post[g] = (math.log(n / d), lists[g])
    return post, frozenset(stop)


def _clean_title(t: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", _norm(t))


def _build(store: Store, stamp: tuple[int, int], body: bool = True) -> _Index:
    t0 = time.perf_counter()
    titles, hooks = _index_lines(store)
    mems: list[dict] = []
    word_sets, meta3, meta2, body3 = [], [], [], []
    for slug in store.slugs():
        text = store.read(slug)
        fm = store.frontmatter(slug)
        body = store._split(text)[1]
        title = titles.get(slug) or fm.get("name") or slug
        hook = hooks.get(slug, "")
        mems.append({"slug": slug, "nm_slug": _norm(slug),
                     "nm_title": _clean_title(title)})
        word_sets.append(_words(" ".join(
            [title, fm.get("description", ""), hook] + store.links_of(text))))
        meta = " ".join([title, fm.get("description", ""), hook])
        meta3.append(_grams(meta, 3))
        meta2.append(_grams(meta, 2))
        # The adaptive shadow judges cues against what the AGENT sees — the resident
        # map — so it asks for an index without the body head; with it, a cue would
        # be "recognised" by prose the resident line never shows.
        body3.append(_grams(body[:BODY_CHARS], 3) if body else set())
    n = len(mems)
    heads: dict[str, dict] = {}
    stops: dict[str, frozenset[str]] = {}
    for hname, sets in (("word", word_sets), ("char3", meta3),
                        ("char2", meta2), ("body", body3)):
        heads[hname], stops[hname] = _invert(sets, max(1, n))
    return _Index(stamp=stamp, built_at=time.time(),
                  build_ms=round((time.perf_counter() - t0) * 1000, 1),
                  n=n, mems=mems, heads=heads, stops=stops)


_CACHE: dict[str, _Index] = {}
_LOCK = threading.Lock()


def index_for(store: Store, body: bool = True) -> _Index:
    """The store's recognizer index, built on first use (~100 ms per ~300 memories)
    and rebuilt when the store moves underneath it. The lock is for the threaded
    server: two concurrent recalls must not both pay the build. `body=False` is a
    second, separately cached index over the resident lines only."""
    stamp = _stamp(store)
    key = store.path if body else store.path + "|resident-only"
    idx = _CACHE.get(key)
    if idx is not None and idx.stamp == stamp:
        return idx
    with _LOCK:
        idx = _CACHE.get(key)
        if idx is None or idx.stamp != stamp:
            idx = _build(store, stamp, body=body)
            _CACHE[key] = idx
    return idx


# ── scoring ──────────────────────────────────────────────────────────────

def _cited(name: str, qn: str) -> bool:
    """A slug named by the question — as a WORD, not a substring. "ai" inside
    "training" and "go" inside "algorithm" used to score a full name-head hit;
    with no runner-up the margin gate was skipped entirely and tier zero answered
    an unrelated memory with full confidence. Three characters minimum, and
    boundaries at anything that is not slug alphabet (a trailing "-mission" means
    the question named a longer, different slug)."""
    return (len(name) >= 3
            and re.search(rf"(?<![a-z0-9-]){re.escape(name)}(?![a-z0-9-])", qn) is not None)


def _score(idx: _Index, question: str, top: int, gate: float) -> tuple[list[dict], str]:
    qn = re.sub(r"\s+", " ", _norm(question)).strip()
    q_tok, q3, q2 = _words(question), _grams(question, 3), _grams(question, 2)

    # name head: the slug or the index title, whole, inside the question.
    name_sc: dict[int, float] = {}
    for i, m in enumerate(idx.mems):
        s = 0.0
        if _cited(m["nm_slug"], qn):
            s += 1.0
        base = m["nm_slug"].rsplit("/", 1)[-1]
        if base != m["nm_slug"] and _cited(base, qn):
            s += 1.0
        if len(m["nm_title"]) >= 3 and m["nm_title"] in qn:
            s += 1.0
        if s:
            name_sc[i] = s

    # Voting heads: idf votes, normalised by what the QUESTION could have reached —
    # a known gram counts its idf, an unknown one counts log(n) (absent from the
    # store = maximal information), a stop-gram counts nowhere. Normalising by the
    # best MEMORY instead would let one lucky gram reach 1.0 and the gate would die.
    unk = math.log(max(2, idx.n))

    def vote(head: str, keys: set[str]) -> tuple[dict[int, float], float]:
        post, stop = idx.heads[head], idx.stops[head]
        sc: dict[int, float] = {}
        qmass = 0.0
        for k in keys:
            e = post.get(k)
            if e:
                idf, ids = e
                qmass += idf
                for i in ids:
                    sc[i] = sc.get(i, 0.0) + idf
            elif k not in stop:
                qmass += unk
        return sc, qmass

    scores = {"name": (name_sc, max(name_sc.values()) if name_sc else 0.0)}
    for head, keys in (("word", q_tok), ("char3", q3), ("char2", q2), ("body", q3)):
        scores[head] = vote(head, keys)

    combined: dict[int, float] = {}
    per_head: dict[int, dict[str, float]] = {}
    for head, (sc, mx) in scores.items():
        if not sc or mx <= 0:
            continue
        w = HEAD_WEIGHTS[head]
        for i, v in sc.items():
            nv = v / mx
            combined[i] = combined.get(i, 0.0) + w * nv
            per_head.setdefault(i, {})[head] = round(nv, 3)
    ranked = sorted(combined.items(), key=lambda kv: (-kv[1], idx.mems[kv[0]]["slug"]))

    # The honesty gate: strong AND clear, or nothing. Silence is the correct answer
    # for a paraphrase — that question belongs to the thinker.
    if not ranked:
        # Nothing voted: every gram of the question is a stop-gram or unknown to the
        # store. That is not "the store said no" — it is "the store could not be
        # asked". The two must not share a name (a cue made only of stop-grams
        # would otherwise read as merely ambiguous).
        return [], "untestable"
    top1 = ranked[0][1]
    top2 = ranked[1][1] if len(ranked) > 1 else 0.0
    if top1 < gate or (top2 > 0 and top1 / top2 < GATE_RATIO):
        return [], "no-confident-hit"
    hits = []
    for i, s in ranked[:max(1, top)]:
        if s < gate:
            break
        hits.append({"slug": idx.mems[i]["slug"], "score": round(s, 4),
                     "heads": per_head[i]})
    return hits, "ok"


def lookup(store: Store, question: str, top: int = 3,
           gate: float = DEFAULT_GATE, cues: bool = True, body: bool = True) -> dict:
    """→ {"hits": [{slug, score, heads}…], "verdict": "ok"|"no-confident-hit"|"untestable", "ms"}.

    Empty hits is the honest answer, never a failure: it means "this question needs
    judgement", and the caller falls through to the thinker.

    Before the five heads run, one exact pre-head: a verified USER callsign the
    question contains, unique in the store, routes straight to its memory. It is
    NOT mixed into the scoring (no weights, no gate arithmetic) — the shared word
    either names one memory or the pre-head is silent, and the five heads run
    exactly as before."""
    t0 = time.perf_counter()
    if cues:
        from .cues import CueLedger
        cue = CueLedger(store).direct(question)
        if cue:
            return {"hits": [{"slug": cue["slug"], "score": "cue",
                              "heads": {"cue": cue["cue"]}}],
                    "verdict": "ok", "cue": cue["cue"],
                    "ms": round((time.perf_counter() - t0) * 1000, 3)}
    idx = index_for(store, body=body)
    hits, verdict = _score(idx, question, top, float(gate))
    return {"hits": hits, "verdict": verdict, "cue": None,
            "ms": round((time.perf_counter() - t0) * 1000, 3)}


def doctor_info(store: Store) -> dict:
    """What `doctor` shows: is tier zero standing, and how big are its heads.
    Never builds — doctor observes; the build belongs to the first recall."""
    idx = _CACHE.get(store.path)
    if idx is None:
        return {"built": False}
    return {
        "built": True,
        "fresh": idx.stamp == _stamp(store),    # False = the next recall rebuilds
        "age_s": int(time.time() - idx.built_at),
        "build_ms": idx.build_ms,
        "memories": idx.n,
        "head_vocab": {h: len(p) for h, p in idx.heads.items()},
        "stop_grams": {h: len(s) for h, s in idx.stops.items()},
    }
