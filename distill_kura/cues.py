"""The cue ledger — receipts are the authority, everything else is cache.

A callsign becomes a ROUTE only through an immutable receipt: content-addressed,
HMAC-signed with the store's own gate key (no second key system), and issued at
the moment the association becomes REAL — a draft that actually POURED, an
extension that actually POURED, or a COVERED verdict against a slug that exists.
A staged draft's manifest carries the cues as provenance; it carries no route,
so a TOSSed or quarantined draft can never grow one.

The reader trusts nothing it is handed, not even its own cache:

* `build()` re-verifies EVERY receipt mechanically — file hash, HMAC, schema,
  slug membership, the referenced manifest's hash, `routing_cues_version`, the
  cue's class, the quote's presence in the manifest's USER evidence, the text's
  exact-substring relationship, and the length rules. "It was gated when it was
  issued" is not an argument; the reader has its own floor.
* `_still/cues.json` is a marked cache, never a truth source. A bad mark, a
  moved stamp, or a write that cannot happen — each means rebuild (in memory if
  the disk refuses), and a cache failure never becomes a recall failure.
* A cue-side provenance stamp (a hash over the receipt set) rides beside the
  store revision: a COVERED cue moves no canonical byte, and freshness must see
  it anyway.

And a cache may only ever pick a route the receipts still prove: on a direct
hit the pointed-to receipt is verified again before the slug is returned.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import uuid

from .distill.gate import cue_key
from .store import Store

CUES_VERSION = 2
RECEIPT_SCHEMA = 1
RECEIPT_PREFIX = "cue-receipt-v1"
LEDGER_MARK_PREFIX = "cue-ledger-v2"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canon(obj) -> str:
    """One deterministic serialisation for hashing and signing."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CueLedger:
    def __init__(self, store: Store):
        self.store = store
        self.path = os.path.join(store.still, "cues.json")
        self.receipts = os.path.join(store.still, "cue-receipts")
        self._memo: tuple[tuple[int, str], dict] | None = None

    # ── issuing: the one moment a route becomes real ────────────────────
    def issue(self, memory_slug: str, evidence_manifest: str, routing_cues: list,
              accepted_via: str) -> str | None:
        """Write the immutable receipt for a callsign association, content-
        addressed, signed with the gate key. Idempotent: the same receipt is the
        same file. Returns the receipt digest, or None when the cues are not
        signable (nothing explodes — the route simply does not come to be)."""
        cues = [{"text": c.get("text"), "class": c.get("class"), "quote": c.get("quote")}
                for c in (routing_cues or []) if isinstance(c, dict) and c.get("text")]
        receipt = {"schema": RECEIPT_SCHEMA, "memory_slug": memory_slug,
                   "evidence_manifest": evidence_manifest,
                   "routing_cues": cues, "accepted_via": accepted_via}
        mark = hmac.new(self.store.gate_key(),
                        (RECEIPT_PREFIX + _canon(receipt)).encode("utf-8"),
                        hashlib.sha256).hexdigest()
        digest = _sha(_canon(receipt))
        blob = json.dumps({"receipt": receipt, "mark": mark},
                          ensure_ascii=False, indent=1, sort_keys=True)
        try:
            os.makedirs(self.receipts, exist_ok=True)
            # Unique even across parallel rebuilds: a shared temp name would let
            # one writer clobber another's os.replace mid-flight.
            tmp = os.path.join(self.receipts, f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(blob)
            os.replace(tmp, os.path.join(self.receipts, f"{digest}.json"))
        except OSError:
            return None                     # a route that cannot be proven is no route
        self._memo = None
        return digest

    # ── verifying: the reader's own mechanical floor ────────────────────
    def receipt_digests(self) -> list[str]:
        try:
            return sorted(f[:-5] for f in os.listdir(self.receipts)
                          if f.endswith(".json") and _HEX64.match(f[:-5]))
        except OSError:
            return []

    def stamp(self) -> tuple[int, str]:
        """The two-ended freshness source: the canonical revision AND the cue
        provenance (a hash over the receipt set — a COVERED cue moves no
        canonical byte, and this is what sees it anyway)."""
        return (self.store.revision(), _sha("\n".join(self.receipt_digests())))

    def _verify_receipt(self, digest: str) -> dict | None:
        """The receipt, only if every mechanical check passes. This runs on
        BUILD and again on every DIRECT HIT — a cache entry may only ever name
        a route the receipts still prove."""
        try:
            with open(os.path.join(self.receipts, f"{digest}.json"),
                      encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            return None
        receipt, mark = (d or {}).get("receipt"), (d or {}).get("mark")
        if not isinstance(receipt, dict) or not isinstance(mark, str):
            return None
        if _sha(_canon(receipt)) != digest:            # content must hash to its name
            return None
        want = hmac.new(self.store.gate_key(),
                        (RECEIPT_PREFIX + _canon(receipt)).encode("utf-8"),
                        hashlib.sha256).hexdigest()
        if not hmac.compare_digest(want, mark):
            return None
        if receipt.get("schema") != RECEIPT_SCHEMA:
            return None
        slug = receipt.get("memory_slug")
        ref = receipt.get("evidence_manifest") or ""
        if not isinstance(slug, str) or slug not in self.store.slug_set():
            return None
        hexd = ref.split("sha256:", 1)[1] if ref.startswith("sha256:") else ""
        if not _HEX64.match(hexd):
            return None
        man = self.store.load_manifest_verified(hexd)
        if man is None:                                # missing or tampered provenance
            return None
        if man.get("routing_cues_version") != 1:
            return None
        user_quotes = [q.get("text") for q in (man.get("quotes") or [])
                       if isinstance(q, dict) and q.get("class") == "USER"]
        cues = []
        for c in receipt.get("routing_cues") or []:
            if not isinstance(c, dict):
                return None
            text, quote = c.get("text"), c.get("quote")
            if c.get("class") != "USER" or not isinstance(text, str) \
                    or not isinstance(quote, str):
                return None
            key = cue_key(text)
            if not key or not any(ch.isalnum() for ch in key) or not (3 <= len(key) <= 40):
                return None
            if quote not in user_quotes:               # the human's words, on record
                return None
            if text not in quote:                      # and the cue is part of them
                return None
            cues.append({"text": text, "class": "USER", "quote": quote})
        if not cues:
            return None
        return {"memory_slug": slug, "evidence_manifest": f"sha256:{hexd}",
                "routing_cues": cues, "accepted_via": receipt.get("accepted_via"),
                "receipt": digest}

    # ── building: receipts only ─────────────────────────────────────────
    def build(self) -> dict:
        known = self.store.slug_set()
        by_key: dict[str, dict[str, dict]] = {}
        for digest in self.receipt_digests():
            r = self._verify_receipt(digest)
            if r is None:
                continue
            for cue in r["routing_cues"]:
                k = cue_key(cue["text"])
                by_key.setdefault(k, {}).setdefault(
                    r["memory_slug"], {"display": cue["text"], "slug": r["memory_slug"],
                                       "manifest": r["evidence_manifest"],
                                       "receipt": digest})
        cues: dict[str, dict] = {}
        ambiguous: dict[str, list[str]] = {}
        for k, per_slug in sorted(by_key.items()):
            if len(per_slug) == 1:
                slug, e = next(iter(per_slug.items()))
                if slug in known:                       # belt: verified above already
                    cues[k] = e
            else:
                ambiguous[k] = sorted(per_slug)
        return {"version": CUES_VERSION, "source_revision": self.store.revision(),
                "cue_stamp": self.stamp()[1], "cues": cues, "ambiguous": ambiguous}

    # ── the cache: marked, never authoritative ──────────────────────────
    def _mark(self, payload: dict) -> str:
        return hmac.new(self.store.gate_key(),
                        (LEDGER_MARK_PREFIX + _canon(payload)).encode("utf-8"),
                        hashlib.sha256).hexdigest()

    def write(self, ledger: dict) -> bool:
        payload = {k: ledger[k] for k in ("version", "source_revision", "cue_stamp",
                                          "cues", "ambiguous")}
        blob = json.dumps({"payload": payload, "mark": self._mark(payload)},
                          ensure_ascii=False, indent=1, sort_keys=True)
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(blob)
            os.replace(tmp, self.path)
            return True
        except OSError:
            return False                # a cache that cannot persist is a cache miss

    def ledger(self) -> dict:
        """The current ledger. The file is trusted only when its mark and BOTH
        stamps say so; anything else rebuilds — and if the disk refuses to hold
        the rebuild, the in-memory copy still answers (a cache failure is not a
        recall failure)."""
        key = self.stamp()
        if self._memo and self._memo[0] == key:
            return self._memo[1]
        d = None
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = None
        if isinstance(d, dict) and isinstance(d.get("payload"), dict) \
                and isinstance(d.get("mark"), str):
            p = d["payload"]
            if hmac.compare_digest(self._mark(p), d["mark"]) \
                    and p.get("version") == CUES_VERSION \
                    and p.get("source_revision") == key[0] \
                    and p.get("cue_stamp") == key[1]:
                self._memo = (key, p)
                return p
        p = self.build()
        self.write(p)                                  # best effort, never fatal
        self._memo = (key, p)
        return p

    # ── the question it exists to answer ────────────────────────────────
    def direct(self, question: str) -> dict | None:
        """The ONE memory the question's verified cues name — or None.

        Several cues may agree on the same memory (a world has several names);
        cues that disagree about the world are silence, as are absent and
        ambiguous. The chosen entry's receipt is verified AGAIN before the slug
        is returned: a cache may only ever repeat what the receipts still prove.
        Any failure below is None — silence over a wrong route, always."""
        try:
            led = self.ledger()
            qn = " ".join(str(question or "").split())
            qkey = cue_key(qn)
            if not qkey:
                return None
            slug = cue = entry = None
            for k, e in led["cues"].items():
                if self._contains(qkey, k):
                    if slug is not None and e["slug"] != slug:
                        return None        # two cues, two worlds: no route
                    if slug is None:
                        slug, cue, entry = e["slug"], e["display"], e
            if slug is None:
                return None
            r = self._verify_receipt(entry["receipt"])
            if r is None or r["memory_slug"] != slug \
                    or slug not in self.store.slug_set():
                return None
            return {"slug": slug, "cue": cue}
        except Exception:
            return None          # the cache is an optimisation; recall must not fall

    @staticmethod
    def _contains(qkey: str, key: str) -> bool:
        # An ASCII-bearing cue must appear as a WORD (the _cited lesson: "hybrid"
        # inside "hybrids" is a different word); a CJK-only cue matches by
        # containment, because CJK text has no spaces to bound words.
        if re.search(r"[a-z0-9]", key):
            return re.search(rf"(?<![a-z0-9-]){re.escape(key)}(?![a-z0-9-])",
                             qkey) is not None
        return key in qkey
