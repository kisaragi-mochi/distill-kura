"""Typed worldline edges (M7) — DERIVED routing state, never canonical.

A canonical memory names its neighbourhood with [[links]] and nothing else; an
edge type (`continues` / `next` / `supersedes` / `rejected` / `blocked-by`) is a
READING of that neighbourhood, recomputed from the store on demand and kept in
ONE derived, marked file — `_still/edges.json`, the cue ledger's
`{"payload": ..., "mark": hmac}` shape, signed with the store's own gate key. A
mis-marked file reads as empty and is rebuilt. No edge is ever written into
frontmatter or a body, and on a frozen store the derivation still answers — in
memory, writing nothing.

Floors, every edge: source and target are EXISTING exact slugs of THIS store
(resolved through `resolve`, kept only when the hit is in `slug_set()`), source
!= target, and the target must be among the source's own [[links]] — prose alone
invents nothing. Evidence floors: `supersedes`/`rejected` need USER evidence in
the source's verified manifest, `blocked-by` needs USER, TOOL or ACT; a memory
whose manifest is missing or unverifiable yields none of the three, and the gap
is counted in `unevidenced` rather than passed over in silence.

Deterministic and byte-stable at a revision. No model calls.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re

from .store import FROZEN

EDGES_VERSION = 1
_MARK_EXTRA = ("superseded", "退役", "畳んだ", "撤退", "旧")
_SUPERSEDES_CACHE: tuple[str, ...] | None = None


def _supersedes_words() -> tuple[str, ...]:
    """The retirement words: floors._OBSOLETE plus the plan's extras. `floors`
    pulls in distill.gate, whose package import pulls the whole distiller —
    loading it here (first use, not import time) is what keeps
    registry → prefill → trail → edges from closing a circular import."""
    global _SUPERSEDES_CACHE
    if _SUPERSEDES_CACHE is None:
        from . import floors
        _SUPERSEDES_CACHE = tuple(dict.fromkeys(floors._OBSOLETE + _MARK_EXTRA))
    return _SUPERSEDES_CACHE


MARK_PREFIX = "edge-map-v1"
# One (source, target, type) at most once; a line carrying several cues loses to
# the most specific, checked in this order.
EDGE_TYPES = ("supersedes", "rejected", "blocked-by", "next", "continues")

_REJECTED = ("却下", "不採用", "棄却", "rejected", "refuted", "撤回", "やめた", "採らない")
_BLOCKED = ("blocked", "待ち", "止まって", "停止中", "blocked-by", "待機")
_CONTINUES = ("続き", "継続", "継承", "continues", "後継", "再戦", "再開")
_NEXT_START = ("次=", "次:", "next:", "次の一手")          # the explicit pointer
_CONTINUES_START = ("→", "次", "Next")                     # the open onward line

# `廃案 (→ old-plan)`: the bare-name arrow shape. A candidate from here still has
# to pass the neighbourhood floor — the name must be [[linked]] somewhere in the
# memory, or the prose invented an edge.
_ARROW_BARE = re.compile(r"\((?:→|->)\s*([^()\n]{1,60}?)\)")

_MANIFEST_CLASSES = ("USER", "TOOL", "ACT")


def _canon(obj) -> str:
    """One deterministic serialisation for hashing and signing."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hit(words: tuple[str, ...], line: str) -> str | None:
    """First word present in the line — ASCII case-insensitively (the floors'
    rule: 'Superseded' and 'superseded' are the same retirement word)."""
    low = line.lower()
    for w in words:
        if w in (low if w.isascii() else line):
            return w
    return None


def _first_pos(words: tuple[str, ...], line: str) -> int:
    low = line.lower()
    best = -1
    for w in words:
        i = low.find(w) if w.isascii() else line.find(w)
        if i >= 0 and (best < 0 or i < best):
            best = i
    return best


def _line_candidates(store, line: str) -> list[tuple[str, str, str]]:
    """(name, type, cue) candidates on one line, before any floor. The cue that
    fired is recorded on the edge — a derivation that cannot say WHY it said
    something is a guess wearing a derivation's clothes."""
    t = cue = None
    if w := _hit(_supersedes_words(), line):
        t, cue = "supersedes", w
    elif w := _hit(_REJECTED, line):
        t, cue = "rejected", w
    elif w := _hit(_BLOCKED, line):
        t, cue = "blocked-by", w
    else:
        stripped = line.lstrip()
        for pref in _NEXT_START:
            if stripped.startswith(pref):
                t, cue = "next", pref
                break
        if t is None:
            for pref in _CONTINUES_START:
                if stripped.startswith(pref):
                    t, cue = "continues", pref
                    break
            if t is None and (w := _hit(_CONTINUES, line)):
                t, cue = "continues", w
    out: list[tuple[str, str, str]] = []
    if t is not None:
        for name in store.links_of(line):
            out.append((name, t, cue))
    if cue and t == "supersedes":
        pos = _first_pos(_supersedes_words(), line)
        for m in _ARROW_BARE.finditer(line):
            if m.start() > pos:                    # the retirement word came first
                out.append((store._clean(m.group(1)), t, cue))
    return out


def _manifest_classes(store, source: str) -> set[str] | None:
    """The evidence classes of the memory's verified manifest — None when the
    pointer is absent or the manifest fails its re-hash (fail closed: a broken
    provenance yields no supersedes/rejected/blocked-by edge at all)."""
    ref = store.frontmatter_exact(source).get("evidence_manifest", "")
    hexd = ref.split("sha256:", 1)[1] if ref.startswith("sha256:") else ""
    man = store.load_manifest_verified(hexd) if hexd else None
    if man is None:
        return None
    return {q.get("class") for q in (man.get("quotes") or [])
            if isinstance(q, dict) and isinstance(q.get("class"), str)}


def derive(store) -> dict:
    """The whole edge map for THIS revision, recomputed from the canonical files.

    → {"version", "revision", "edges": [{source, target, type, cue, evidence}],
       "counts": {type: n}, "unevidenced": n, "dropped": {reason: n}}"""
    known = store.slug_set()
    revision = store.revision()
    edges: dict[tuple[str, str, str], dict] = {}
    dropped: dict[str, int] = {}
    unevidenced = 0
    classes: dict[str, set[str] | None] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for source in sorted(known):
        text = store.read_exact(source)
        _fm_raw, body = store._split(text)
        desc = store.frontmatter_exact(source).get("description", "")
        surface = f"{desc}\n{body}"                   # frontmatter excluded except description
        # The neighbourhood, resolved once: every candidate target must already be
        # a link of THIS memory, or it was invented from prose.
        linked = {r for r in (store.resolve(n) for n in store.links_of(surface))
                  if r}
        for line in surface.splitlines():
            for name, typ, cue in _line_candidates(store, line):
                target = store.resolve(name)
                if target is None:
                    drop("unknown-target")
                    continue
                if target == source:
                    drop("self")
                    continue
                if target not in linked:
                    drop("not-linked")
                    continue
                if (source, target, typ) in edges:
                    continue
                evidence = None
                if typ in ("supersedes", "rejected", "blocked-by"):
                    if source not in classes:
                        classes[source] = _manifest_classes(store, source)
                    cls = classes[source]
                    if typ == "blocked-by":
                        allowed = [c for c in _MANIFEST_CLASSES if cls and c in cls]
                        if not allowed:
                            unevidenced += 1
                            continue
                        evidence = allowed[0]         # USER preferred, then TOOL, then ACT
                    elif cls and "USER" in cls:
                        evidence = "USER"
                    else:
                        unevidenced += 1
                        continue
                edges[(source, target, typ)] = {"source": source, "target": target,
                                                "type": typ, "cue": cue,
                                                "evidence": evidence}
    ordered = [edges[k] for k in sorted(
        edges, key=lambda k: (k[0], k[1], EDGE_TYPES.index(k[2])))]
    counts = {t: sum(1 for e in ordered if e["type"] == t) for t in EDGE_TYPES}
    return {"version": EDGES_VERSION, "revision": revision, "edges": ordered,
            "counts": counts, "unevidenced": unevidenced, "dropped": dropped}


def _path(store) -> str:
    return os.path.join(store.still, "edges.json")


def _mark(store, payload: dict) -> str:
    return hmac.new(store.gate_key(), (MARK_PREFIX + _canon(payload)).encode("utf-8"),
                    hashlib.sha256).hexdigest()


def write(store, payload: dict | None = None) -> bool:
    """Mark and persist the payload. Skipped on a frozen store — derivation
    still answers there, in memory; frozen means the world does not grow, not
    even its derived caches. A cache that cannot persist is a cache miss."""
    if payload is None:
        payload = derive(store)
    if store.write_policy == FROZEN:
        return False
    blob = json.dumps({"payload": payload, "mark": _mark(store, payload)},
                      ensure_ascii=False, indent=1, sort_keys=True)
    try:
        os.makedirs(store.still, exist_ok=True)
        store._replace_file(_path(store), blob.encode("utf-8"))
        return True
    except OSError:
        return False


def load(store) -> dict:
    """The stored payload, only when its mark and revision prove it current.
    Anything else — unreadable, mis-marked, wrong version, stale revision —
    reads as {} and the next `edges_of` rebuilds it."""
    try:
        with open(_path(store), encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    p, mark = (d or {}).get("payload"), (d or {}).get("mark")
    if not isinstance(p, dict) or not isinstance(mark, str) or not mark.isascii():
        return {}
    if not hmac.compare_digest(_mark(store, p), mark):
        return {}
    if p.get("version") != EDGES_VERSION or p.get("revision") != store.revision():
        return {}
    return p


_memo: tuple[str, int, dict] | None = None


def current(store) -> dict:
    """The payload for this revision — the file when it proves itself, else a
    fresh derivation. Never writes: this is the door for readers that must stay
    read-only (doctor, bench). Memoised per (store, revision)."""
    global _memo
    rev = store.revision()
    if _memo and _memo[0] == store.path and _memo[1] == rev:
        return _memo[2]
    payload = load(store) or derive(store)
    _memo = (store.path, rev, payload)
    return payload


def stamp_sha(store) -> str:
    """Hash of the current payload — folded into the trail's spec so a changed
    edge set re-prices the trail's freshness even though no canonical byte moved
    (the cue ledger's cue_stamp, for the same reason)."""
    return hashlib.sha256(_canon(current(store)).encode("utf-8")).hexdigest()


def edges_of(store, slug: str) -> list[dict]:
    """One slug's outgoing + incoming edges as {"type", "other", "direction"} —
    from the marked file, rebuilt (and re-marked, unless frozen) whenever the
    stored revision is not this one."""
    payload = load(store)
    if not payload:
        payload = derive(store)
        write(store, payload)
    s = store.resolve_exact(slug)
    if not s:
        return []
    out = [{"type": e["type"], "other": e["target"], "direction": "out"}
           for e in payload.get("edges", []) if e["source"] == s]
    out += [{"type": e["type"], "other": e["source"], "direction": "in"}
            for e in payload.get("edges", []) if e["target"] == s]
    return out
