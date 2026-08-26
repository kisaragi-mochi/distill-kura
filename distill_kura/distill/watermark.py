"""How far we have drunk from each journal — and how two distillers avoid the same water.

Two mistakes were made here and both are fixed in this file:

**Rewind.** Two distillers running in parallel each read the marks dict, each wrote
their own back, and the later write erased the other's progress — the same stretch was
re-processed a dozen times. The fix is BOTH halves: `flock` to serialise, and `max()`
to merge. A lock alone still lets a stale snapshot win.

**Drinking before reserving.** Reserving the stretch *before* reading it is what keeps
two runners apart. Advance-after-read leaves a window where the other runner starts on
the same offset.

Watermark units are the source adapter's business (byte offset for append-only files,
sequence number for rewritten archives) — this module only stores integers.
"""
from __future__ import annotations

import fcntl
import json
import os

from .sources import Source, source_for


class Watermarks:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def read(self) -> dict[str, int]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except ValueError:
            return {}

    def _write(self, cur: dict) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def advance(self, key: str, pos: int) -> None:
        """Move forward only. A stale value must never pull the mark backwards."""
        with open(self.path + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                cur = self.read()
                cur[key] = max(cur.get(key, 0), int(pos))
                self._write(cur)
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def claim(self, files: list[str], budget_chars: int,
              min_chars: int) -> tuple[str, int, Source, int] | None:
        """Reserve the next stretch worth drinking.

        Returns (path, start, source, reserved_end). `budget_chars` is passed to
        `claim_bound` unchanged — it is the same budget `sip` will receive. A
        global byte-slack here reserved past what a record-walking source would
        drink; max-forward then skipped the unread tail forever. Adapters that
        estimate bytes apply their own slack inside `claim_bound`.
        """
        with open(self.path + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                cur = self.read()
                for path in files:
                    src = source_for(path)
                    if not src:
                        continue
                    k = src.key(path)
                    start = cur.get(k, 0)
                    end, approx = src.claim_bound(path, start, budget_chars)
                    if approx < min_chars or end <= start:
                        continue
                    cur[k] = end
                    self._write(cur)
                    return path, start, src, end
                return None
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)
