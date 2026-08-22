"""`kura tend` — the watcher that keeps a store in the house's quiet hours.

What it does, in order, every time the house has been quiet for `idle_min`:

    drain   drafts waiting → the editor reads each one cold and pours / fixes / tosses
    distill no drafts     → one pass over the journal: sip → spot → gate → stage
    weave   something poured this silence → re-weave the resident map, once
    tidy    once per silence, only if the index has mechanically ragged lines

"Quiet" is the simplest signal there is: the newest journal file's mtime. No model is
asked whether the human is busy; a conversation file that has not changed in ten
minutes is a person who is not typing. That is what the house's first watcher used
for five days (2026-08-07 → 08-11, `nemuri.py`), and it was enough.

Lessons from that watcher, written into this one:

- **"Nothing to do" is exit code 2**, not success. A track that returns 2 is put to
  sleep for `backoff_min`, so an empty journal does not spin every fifteen seconds
  and starve the other tracks — that spin once ran `tidy` 3,122 times in a night and
  did real work three times.
- **Count work, never launches.** The summary says what was poured, tossed, fixed
  and drafted. "495 passes" was a launch counter and it was mistaken for output.
- **Keep every decision.** Track output goes to `_still/tend.log`, never to
  /dev/null — the brain's stdout was discarded once and the candidate counts of
  five days can no longer be recovered.
- **Yield when the human returns** (`yield_on_return`, default on): a running track
  is terminated the moment the journal changes, because the editor is usually the
  same GPU the conversation needs. When the editor is a separate model that does not
  compete for the same seat — a CPU model, another machine — set it off, so a verdict
  in flight is not thrown away.
- **Be easy to watch.** A heartbeat in `_still/tend.json` every tick; `doctor`
  reads it and says whether the watcher is alive. The first watcher died with the
  machine and nobody noticed for twelve days.

The watcher spawns `kura` subcommands as subprocesses rather than calling in-process,
so a track can be killed cleanly and its exit code means what the CLI says it means.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

from .distill.sources import discover_all
from .registry import Registry
from .store import Store

TRACKS = ("drain", "distill", "weave", "tidy")


def _log(path: str, s: str) -> None:
    line = f"{datetime.now().strftime('%m-%d %H:%M:%S')} {s}"
    print(line, flush=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


class Tender:
    def __init__(self, reg: Registry, store: Store, config_path: str | None,
                 idle_min: float | None = None, poll_s: float = 15.0,
                 backoff_min: float | None = None, yield_on_return: bool | None = None):
        self.reg, self.store, self.config_path = reg, store, config_path
        cfg = (reg.raw.get("distill") or {})
        scfg = store.extra.get("distill") if isinstance(store.extra.get("distill"), dict) else {}
        pick = lambda k, d: scfg.get(k, cfg.get(k, d))   # noqa: E731
        self.idle_s = float(idle_min if idle_min is not None else pick("idle_min", 10)) * 60
        self.backoff_s = float(backoff_min if backoff_min is not None else pick("backoff_min", 20)) * 60
        self.yield_on_return = bool(yield_on_return if yield_on_return is not None
                                    else pick("yield_on_return", True))
        self.poll_s = poll_s
        # Journals are the store's own — the same roots the distiller drinks from, so
        # the watcher and the distiller agree on what "the conversation" is.
        from .distill import Distiller
        self.journals = Distiller(reg, store).journals
        self.exclude = [st.path for st in reg.stores.values()]
        os.makedirs(store.still, exist_ok=True)
        self.log_path = os.path.join(store.still, "tend.log")
        self.beat_path = os.path.join(store.still, "tend.json")
        self.next_ok: dict[str, float] = {t: 0.0 for t in TRACKS}
        self.proc: subprocess.Popen | None = None
        self.proc_track = ""
        self.done = {"poured": 0, "tossed": 0, "fixed": 0, "drafts": 0, "woven": 0, "tidied": 0}
        self._woven_this_silence = False
        self._tidied_this_silence = False

    # ── the signal ────────────────────────────────────────────────────────
    def newest_mtime(self) -> float:
        fs = discover_all(self.journals, exclude_roots=self.exclude) if self.journals else []
        best = 0.0
        for f in fs:
            try:
                best = max(best, os.path.getmtime(f))
            except OSError:
                continue
        return best

    # ── running a track ───────────────────────────────────────────────────
    def _cmd(self, track: str) -> list[str]:
        base = [sys.executable, "-m", "distill_kura.cli"]
        if self.config_path:
            base += ["-c", self.config_path]
        base += ["-s", self.store.name]
        return base + {"drain": ["distill", "drain"], "distill": ["distill", "run", "--chunks", "1"],
                       "weave": ["weave"], "tidy": ["distill", "tidy"]}[track]

    def start(self, track: str) -> None:
        _log(self.log_path, f"→ {track}")
        # Output goes to a file, not a pipe: a pass that prints more than the pipe
        # buffer holds would block on write and the watcher would wait on it forever.
        self._out_path = os.path.join(self.store.still, f"tend.{track}.out")
        out = open(self._out_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(self._cmd(track), stdout=out, stderr=subprocess.STDOUT, text=True)
        out.close()
        self.proc_track = track

    def reap(self) -> bool:
        """Collect a finished track. Returns True when one was collected."""
        if not self.proc or self.proc.poll() is None:
            return False
        try:
            out = open(self._out_path, encoding="utf-8", errors="ignore").read()
        except OSError:
            out = ""
        rc = self.proc.returncode
        track, self.proc, self.proc_track = self.proc_track, None, ""
        for line in out.strip().splitlines():
            _log(self.log_path, f"   {line[:400]}")
        if rc == 2:
            self.next_ok[track] = time.time() + self.backoff_s
            _log(self.log_path, f"· {track}: nothing to do — resting {int(self.backoff_s / 60)} min")
            return True
        if rc != 0:
            self.next_ok[track] = time.time() + self.backoff_s
            _log(self.log_path, f"✗ {track} failed (rc={rc}) — resting {int(self.backoff_s / 60)} min")
            return True
        last = next((l for l in reversed(out.strip().splitlines()) if l.startswith("{")), "")
        try:
            r = json.loads(last) if last else {}
        except ValueError:
            r = {}
        if track == "drain":
            self.done["poured"] += int(r.get("poured") or 0)
            self.done["tossed"] += int(r.get("tossed") or 0)
            self.done["fixed"] += int(r.get("fixed") or 0)
            if r.get("poured"):
                self._woven_this_silence = False       # the map has something new to say
        elif track == "distill":
            self.done["drafts"] += len(r.get("drafts") or [])
        elif track == "weave":
            self.done["woven"] += 1
            self._woven_this_silence = True
        elif track == "tidy":
            self.done["tidied"] += int(r.get("fixed") or 0)
        return True

    def kill(self, why: str) -> None:
        if self.proc and self.proc.poll() is None:
            _log(self.log_path, f"⏹ {self.proc_track} stopped — {why}")
            try:
                self.proc.send_signal(signal.SIGTERM)
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()
            self.proc = None
            self.proc_track = ""

    # ── choosing the next track ───────────────────────────────────────────
    def choose(self, now: float) -> str | None:
        drafts = os.path.join(self.store.still, "drafts")
        have_drafts = any(f.endswith(".md") for f in os.listdir(drafts)) if os.path.isdir(drafts) else False
        order = (["drain"] if have_drafts else ["distill"])
        if not self._woven_this_silence and self.done["poured"]:
            order.append("weave")
        if not self._tidied_this_silence:
            order.append("tidy")
        for t in order:
            if now >= self.next_ok[t]:
                return t
        return None

    # ── heartbeat, so doctor can say whether the watcher is alive ─────────
    def beat(self, idle: float, stamp: float) -> None:
        try:
            tmp = self.beat_path + f".tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "at": int(time.time()), "idle_s": int(idle),
                           "journal_mtime": int(stamp), "running": self.proc_track,
                           "next_ok": {k: int(v) for k, v in self.next_ok.items()},
                           "done": self.done}, f)
            os.replace(tmp, self.beat_path)
        except OSError:
            pass

    # ── one tick ──────────────────────────────────────────────────────────
    def tick(self, stamp_seen: float) -> float:
        """Returns the journal mtime as of this tick."""
        now = time.time()
        stamp = self.newest_mtime()
        self.reap()
        if stamp != stamp_seen and stamp_seen:
            # The human is back. Say what was done — work, not launches — and reset.
            if self.yield_on_return:
                self.kill("the journal changed: the human is back")
            if any(self.done.values()):
                _log(self.log_path, "the human is back: " + ", ".join(f"{k} {v}" for k, v in self.done.items() if v))
            self.done = {k: 0 for k in self.done}
            self._woven_this_silence = False
            self._tidied_this_silence = False
        idle = now - stamp if stamp else 0.0
        if stamp and idle >= self.idle_s and not self.proc:
            t = self.choose(now)
            if t:
                if t == "tidy":
                    self._tidied_this_silence = True
                self.start(t)
        self.beat(idle, stamp)
        return stamp

    def watch(self) -> None:
        _log(self.log_path, f"tending '{self.store.name}': quiet after {int(self.idle_s / 60)} min, "
                            f"rest {int(self.backoff_s / 60)} min on nothing-to-do, "
                            f"yield_on_return={'on' if self.yield_on_return else 'off'}, "
                            f"journals={list(self.journals) or 'NONE'}")
        if not self.journals:
            _log(self.log_path, "⚠ this store has no journal roots: nothing will ever be quiet or busy. "
                                "Bind [stores.<name>.distill.journals] first.")
        stamp = 0.0
        try:
            while True:
                stamp = self.tick(stamp)
                time.sleep(self.poll_s)
        except KeyboardInterrupt:
            self.kill("watcher stopped")


def heartbeat(store: Store, stale_after_s: float = 120.0) -> dict:
    """What `doctor` reports: is someone tending this store?"""
    return store.tend_state(stale_after_s)
