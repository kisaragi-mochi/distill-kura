"""The seed field — where ideas live, because they are not facts.

An idea has no quote by definition: the point of an idea is that it is *not* in the
material. Requiring evidence for it would either throw it away or push the model to
disguise it as a fact. So ideas go to `_still/seeds.jsonl`, never into the store.

They are not filed away and forgotten. Every time an evidence-backed candidate makes
it through the gate, the open seeds are checked against it: an idea that turns out to
have been right graduates once, with a note of what confirmed it. A seed field nobody
revisits is a junk drawer.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()[:19]


class Seeds:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    @contextlib.contextmanager
    def _locked(self):
        """One writer at a time, like Watermarks.advance.

        confirm() is a read-modify-write of the whole ledger and sow() appends to it:
        parallel runners (a hand `kura distill run` beside the tended one) interleaved
        those and lost seeds — an append landing between confirm's read and its
        replace was simply dropped, and two confirms sharing one tmp file truncated
        each other's inode."""
        with open(self.path + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def _all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        for line in open(self.path, encoding="utf-8", errors="ignore"):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def sow(self, text: str, who: str, topic: str = "") -> dict:
        rec = {"kind": "IDEA", "text": text[:300], "from": who, "topic": topic,
               "at": _now(), "confirmed": None}
        with self._locked():
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def open_seeds(self, limit: int = 30) -> list[dict]:
        return [s for s in self._all() if not s.get("confirmed")][-limit:]

    def confirm(self, seed_text: str, how: str, memory: str = "") -> bool:
        with self._locked():
            rows = self._all()
            hit = False
            for r in rows:
                if not r.get("confirmed") and r.get("text", "")[:60] == seed_text[:60]:
                    r["confirmed"] = {"how": how[:300], "memory": memory, "at": _now()}
                    hit = True
                    break
            if hit:
                # Per-process tmp: a fixed name is one file two writers open("w") on,
                # and the loser's os.replace raises FileNotFoundError mid-pass.
                tmp = self.path + f".tmp.{os.getpid()}"
                with open(tmp, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                os.replace(tmp, self.path)
        return hit
