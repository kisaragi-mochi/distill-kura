"""The cue ledger — a derived routing cache, never a canonical fact.

Verified callsigns live in content-addressed evidence manifests; `_still/cues.json`
is only their INDEX, rebuilt from those manifests whenever it is absent or stale.
Delete it any time: the same ledger comes back from the same provenance.

Every entry had to pass the gate before it got here — the phrase was an exact
substring of a surviving [USER] quote — and every entry is re-proven on the way
IN again: manifests are hash-verified, and a manifest whose `memory_slug` names a
memory this store no longer holds is not routing material. A cue that points at
two memories is AMBIGUOUS, and ambiguous is the same silence as absent: a wrong
direct route is the one unforgivable outcome (tier zero's founding rule).
"""
from __future__ import annotations

import json
import os
import re

from .distill.gate import cue_key
from .store import Store

CUES_VERSION = 1


class CueLedger:
    def __init__(self, store: Store):
        self.store = store
        self.path = os.path.join(store.still, "cues.json")
        self._memo: tuple[int, dict] | None = None

    # ── building, from provenance only ──────────────────────────────────
    def build(self) -> dict:
        """The ledger as a pure function of the store's verified manifests and its
        current slug set. Nothing else may feed it."""
        known = self.store.slug_set()
        by_key: dict[str, dict[str, dict]] = {}
        ev_dir = os.path.join(self.store.path, "_evidence")
        for fname in sorted(os.listdir(ev_dir)) if os.path.isdir(ev_dir) else []:
            digest = fname[:-5] if fname.endswith(".json") else ""
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                continue
            man = self.store.load_manifest_verified(digest)
            if man is None:
                continue    # tampered, unreadable, or gone: unprovable provenance
            slug = man.get("memory_slug")
            cues = man.get("routing_cues") or []
            if not isinstance(slug, str) or not cues or slug not in known:
                continue    # routes at a memory this store does not hold
            for cue in cues:
                text = cue.get("text") if isinstance(cue, dict) else None
                if not isinstance(text, str) or not text:
                    continue
                k = cue_key(text)
                if not k:
                    continue
                by_key.setdefault(k, {}).setdefault(
                    slug, {"display": text, "manifest": f"sha256:{digest}"})
        cues: dict[str, dict] = {}
        ambiguous: dict[str, list[str]] = {}
        for k, per_slug in sorted(by_key.items()):
            if len(per_slug) == 1:
                slug, e = next(iter(per_slug.items()))
                cues[k] = {"display": e["display"], "slug": slug,
                           "manifest": e["manifest"]}
            else:
                # The same shared word for two memories is a real thing ("あの件").
                # It is kept, visibly — but it may never pick one of them for us.
                ambiguous[k] = sorted(per_slug)
        return {"version": CUES_VERSION, "source_revision": self.store.revision(),
                "cues": cues, "ambiguous": ambiguous}

    def write(self, ledger: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + f".tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ledger, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    def ledger(self) -> dict:
        """The current ledger: the file on disk when it is provably current,
        rebuilt from provenance otherwise. Derived state, self-healing."""
        rev = self.store.revision()
        if self._memo and self._memo[0] == rev:
            return self._memo[1]
        d = None
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = None
        if not (isinstance(d, dict) and d.get("version") == CUES_VERSION
                and d.get("source_revision") == rev
                and isinstance(d.get("cues"), dict)
                and isinstance(d.get("ambiguous"), dict)):
            d = self.build()
            self.write(d)
        self._memo = (rev, d)
        return d

    # ── the question it exists to answer ────────────────────────────────
    def direct(self, question: str) -> dict | None:
        """The ONE verified cue the question contains, as {"slug", "cue"} — or
        None. Absent, ambiguous, and stale-able-to-rebuild are the same silence."""
        led = self.ledger()
        qn = " ".join(str(question or "").split())
        qkey = cue_key(qn)
        if not qkey:
            return None
        hit = None
        for k, e in led["cues"].items():
            if self._contains(qkey, k):
                if hit is not None:
                    return None      # two different cues, two worlds: silence
                hit = e
        return {"slug": hit["slug"], "cue": hit["display"]} if hit else None

    @staticmethod
    def _contains(qkey: str, key: str) -> bool:
        # An ASCII-bearing cue must appear as a WORD (the _cited lesson: "hybrid"
        # inside "hybrids" is a different word); a CJK-only cue matches by
        # containment, because CJK text has no spaces to bound words.
        if re.search(r"[a-z0-9]", key):
            return re.search(rf"(?<![a-z0-9-]){re.escape(key)}(?![a-z0-9-])",
                             qkey) is not None
        return key in qkey
