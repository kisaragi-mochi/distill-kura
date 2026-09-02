"""The Hot Trail — where in this world the house was walking recently.

The map (the woven index) answers "what exists". The trail answers a smaller,
warmer question: *what were we just doing?* A session that opens with "続きやろう"
does not need the whole continent — it needs the last few breadcrumbs, and it
needs them WITHOUT re-reading anything.

Selection is deliberately dumb (plan §5.3): the FRESH layer, newest internal date
first, until a small token budget, each memory once, and every line is an EXISTING
recognition line reused verbatim. No prose is written here, no ranking exists
here, and the read log is not consulted — a trail that promoted what was often
read would be an importance score wearing a path costume.

It is a derived artifact in every particular: `_still/trailhead.md` can be deleted
and rebuilt from the canonical store at any time, and its freshness is proven the
way the cloth's is — a sidecar recording the source revision, the source index
hash and the trail's own hash. Unprovable means stale; one rebuild heals.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

from . import edges
from .store import Store, contained
from .tokens import estimate
from .weave import Loom

# The honesty note rides INSIDE the marker. It used to be a prose line under the
# marker ("CURRENT PATH — these are recent breadcrumbs, not the whole memory:"),
# and measured on the house store (2026-09-02, 42 cases, one reader) that one
# sentence made the agent start narrating: format errors 13→19, recovery 22→17,
# while the same words folded into the marker cost nothing (25/42, 15). A prose
# line in a block of index lines reads as an invitation to write prose.
TRAIL_BEGIN = "<<<KURA-TRAIL — recent breadcrumbs, not the whole memory>>>"
TRAIL_END = "<<<END KURA-TRAIL>>>"
# Constant on purpose: a header that carried a date or a revision number would
# re-price the prefix cache every time the trail was rebuilt.
HEADER = ""   # kept for importers; the note lives in TRAIL_BEGIN now
STATE_VERSION = 3   # 3: the prose header moved into the marker (a trail on disk rebuilds)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class TrailStamp:
    """What the trail was built FROM — canonical state AND the knobs that shape it.
    The revision is captured BEFORE the index is read (the weave's rule), and the
    spec/valid_until halves exist because the trail's own docstring says the fresh
    window slides with time: a trail built on day 13 read on day 15 with an
    unchanged store would otherwise wear a valid freshness stamp over breadcrumbs
    that have aged out of the window."""
    source_revision: int
    source_sha256: str
    spec_sha256: str
    valid_until: float


class Trail:
    def __init__(self, store: Store, loom: Loom | None = None,
                 trail_tokens: int = 200, out_path: str | None = None):
        self.store = store
        self.loom = loom or Loom(store)
        self.trail_tokens = int(trail_tokens)
        self.out_path = out_path or os.path.join(store.still, "trailhead.md")
        # The same three guards the loom carries — a derived writer must never
        # touch a canonical file. The trail is rebuildable from the store at any
        # time; a memory it overwrote is not.
        if os.path.abspath(self.out_path) == os.path.abspath(store.index_path):
            raise ValueError(
                f"the trail would overwrite the canonical index ({store.index_path}). "
                "Trail to a different file.")
        inside = contained(store.path, self.out_path)
        if inside and not contained(store.still, self.out_path):
            # The loom's wound, reopened: `out_path` at a store-root `.md` silently
            # eats a memory one rebuild at a time while the stats say `written`.
            raise ValueError(
                f"the trail would be written into a memory slot ({self.out_path}); "
                f"rebuilding there destroys that memory. Put it under {store.still}, "
                f"or outside the store entirely.")
        if inside and store.write_policy == "frozen":
            raise ValueError(f"store '{store.name}' is frozen: nothing may be written "
                             f"inside it, including the trail. Point the trail outside "
                             f"the store to keep one for an archive.")
        self.state_path = self.out_path + ".state.json"

    def _spec(self) -> dict:
        """Everything besides the canonical state that shapes the trail's bytes.
        The edges hash rides here (M7): a `continues` line is part of the trail's
        bytes, so a changed edge set must re-price the trail's freshness even
        though no canonical byte moved — the cue ledger's cue_stamp, same shape."""
        return {"fresh_days": self.loom.fresh_days,
                "pinned_types": list(self.loom.pinned_types),
                "trail_tokens": self.trail_tokens,
                "bulk_touch_share": self.loom.bulk_touch_share,
                "edges_sha256": edges.stamp_sha(self.store)}

    def spec_sha256(self) -> str:
        return _sha256(json.dumps(self._spec(), sort_keys=True))

    # ── building ─────────────────────────────────────────────────────────
    def build(self) -> tuple[str | None, TrailStamp]:
        """The trail text (None when the fresh layer is empty — an empty CURRENT
        PATH block would cost prompt tokens to say nothing), and the stamp of the
        canonical state AND shaping spec it was built from."""
        now = time.time()
        stamp = TrailStamp(self.store.revision(), _sha256(self.store.index_text()),
                           self.spec_sha256(), 0.0)
        fresh = [s for s in self.store.slugs()
                 if self.loom.layer_of(s, now) == "fresh"]
        # Newest first by the memory's own internal date (the loom's age logic:
        # a date written inside the memory, mtime only when nothing bulk-touched
        # it). Ties break on the slug, so the same store always yields the same
        # bytes at the same revision.
        ages = {s: self.loom.age_days(s, now) for s in fresh}
        fresh.sort(key=lambda s: (ages[s], s))

        lines: list[str] = []
        included: list[str] = []                # the fresh slugs whose lines made it
        seen_line: set[str] = set()
        used = 0
        included_ages: list[float] = []         # ages of the breadcrumbs included
        for slug in fresh:
            line = self._index_line(slug)
            if line is None or line in seen_line:      # a grouped line names several
                continue
            cost = estimate(line)
            if lines and used + cost > self.trail_tokens:
                break
            seen_line.add(line)
            lines.append(line)
            included.append(slug)
            used += cost
            included_ages.append(ages[slug])
        if not lines:
            return None, stamp
        # How long this trail can call itself current: until the FIRST included
        # breadcrumb ages out of the fresh window. Past that moment the text is a
        # lie about the present even though the store never moved — the pure-time
        # hazard the revision and the index hash cannot see.
        horizon_days = min(self.loom.fresh_days - a for a in included_ages)
        stamp.valid_until = now + max(0.0, horizon_days) * 86400.0
        body = "\n".join(lines)
        onward = self._onward_lines(included)
        if onward:
            body += "\n" + "\n".join(onward)
        text = f"{TRAIL_BEGIN}\n{body}\n{TRAIL_END}\n"
        return text, stamp

    def _onward_lines(self, slugs: list[str]) -> list[str]:
        """The `↳ source continues → target` lines (M7): one optional tail, only
        when a fresh breadcrumb itself has an onward `continues`/`next` edge.
        Max 3, in the edge map's own order — deterministic, and a trail that
        grew relations without bound would be a second map."""
        by_source: dict[str, list[dict]] = {}
        for e in edges.current(self.store).get("edges", []):
            if e["type"] in ("continues", "next"):
                by_source.setdefault(e["source"], []).append(e)
        out: list[str] = []
        for slug in slugs:
            for e in by_source.get(slug, ()):
                out.append(f"↳ {slug} {e['type']} → {e['target']}")
                if len(out) >= 3:
                    return out
        return out

    def _index_line(self, slug: str) -> str | None:
        """The canonical recognition line, verbatim — a fresh-layer line is kept
        whole by the loom, so the canonical line IS the woven line. Nothing is
        written here; a trail that composed its own prose would be a new fact
        factory (plan §1.4)."""
        import re
        for line in self.store.index_text().splitlines():
            if re.search(rf"\({re.escape(slug)}\.md\)", line):
                return line
        return None

    # ── persisting, with the same two-ended proof as the cloth ──────────
    def persist(self, text: str | None, stamp: TrailStamp) -> dict:
        """Put the built trail on disk, atomically, only if the store has not
        moved since the build. `None` (nothing fresh) REMOVES the trail: a
        surviving "current path" of stale breadcrumbs is a lie about the present."""
        if self.store.write_policy == "frozen":
            # Reachable only for a trail OUTSIDE the store (the constructor refuses
            # in-store paths on a frozen store): taking the store lock would write
            # a lock file inside it, so the CAS check runs unlocked — and nothing
            # can move a frozen store's index anyway.
            return self._persist_checked(text, stamp)
        with self.store._locked():
            return self._persist_checked(text, stamp)

    def _persist_checked(self, text: str | None, stamp: TrailStamp) -> dict:
        if (self.store.revision() != stamp.source_revision
                or _sha256(self.store.index_text()) != stamp.source_sha256):
            # A memory landed while the trail was being built. Nothing is written;
            # the caller rebuilds — retrying here could chase a busy store forever.
            return {"written": False, "refused": "source moved while trail building"}
        if text is None:
            for p in (self.out_path, self.state_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
            return {"written": False, "removed": True}
        if self.text_on_disk() == text and self._state() == self._state_for(stamp, _sha256(text)):
            return {"written": False, "unchanged": True}   # no churn when nothing moved
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        tmp = self.out_path + f".tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, self.out_path)
        # Trail first, record second: a crash between the two leaves an unprovable
        # trail (read as stale), never a fresh stamp on unverified bytes.
        self._record(stamp, _sha256(text))
        return {"written": True, "tokens": estimate(text), "lines":
                text.count("\n- ") if "\n- " in text else 0}

    def _record(self, stamp: TrailStamp, trail_sha: str) -> None:
        record = self._state_for(stamp, trail_sha)
        if self._state() == record:
            return                               # no churn when nothing changed
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp = self.state_path + f".tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, self.state_path)

    @staticmethod
    def _state_for(stamp: TrailStamp, trail_sha: str) -> dict:
        return {"version": STATE_VERSION, "source_revision": stamp.source_revision,
                "source_sha256": stamp.source_sha256, "trail_sha256": trail_sha,
                "spec_sha256": stamp.spec_sha256, "valid_until": stamp.valid_until}

    # ── reading it back ──────────────────────────────────────────────────
    def text_on_disk(self) -> str | None:
        try:
            return open(self.out_path, encoding="utf-8", errors="ignore").read()
        except OSError:
            return None

    def _state(self) -> dict:
        try:
            with open(self.state_path, encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def is_stale(self) -> bool:
        """True unless the trail on disk is provably the product of the current
        canonical state, shaped by the current spec, and still inside its own
        time horizon. No file, no (complete, current-version) record, a moved
        index, a moved revision, a changed config, a text that no longer hashes
        to its own record, or a passed valid_until — each is stale, and one
        rebuild heals all of them.

        Revision 0 is an HONEST value (a store whose mutations predate the
        counter, the weave's documented contract) and is checked by type, not
        truthiness — `if not st.get(...)` used to read 0 as "missing" and leave
        such a store permanently stale."""
        text = self.text_on_disk()
        if text is None:
            return True
        st = self._state()
        if st.get("version") != STATE_VERSION:
            return True                       # a future format is not ours to trust
        for k in ("source_sha256", "trail_sha256", "spec_sha256", "valid_until"):
            if not st.get(k):
                return True                   # half a record proves nothing
        revision = st.get("source_revision")
        if not (isinstance(revision, int) and not isinstance(revision, bool)):
            return True
        return (self.store.revision() != revision
                or _sha256(self.store.index_text()) != st["source_sha256"]
                or _sha256(text) != st["trail_sha256"]
                or self.spec_sha256() != st["spec_sha256"]
                or time.time() > float(st["valid_until"]))

    # ── one-shot ─────────────────────────────────────────────────────────
    def write(self) -> dict:
        text, stamp = self.build()
        return self.persist(text, stamp)
