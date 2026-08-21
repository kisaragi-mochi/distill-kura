"""Where the raw material comes from, and how it is CLASSED.

Everything the distiller reads is turned into segments carrying an evidence class:

    [USER]  the human's own words      — primary evidence
    [TOOL]  machine output             — the ONLY place a measured number may come from
    [ACT]   a tool the agent invoked   — evidence that an action was taken
    [SELF]  the agent's own prose      — a judgement worth keeping, never a bare fact

Reasoning / thinking blocks are dropped: an inner monologue is not evidence.
Injected content (system reminders, runtime context) is not the human speaking.

Three adapters ship here; add your own by subclassing `Source` and registering it
in `SOURCES`. `watermark` semantics differ per adapter, so each one owns them:
byte offset for append-only files, sequence number for rewritten archives.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
from dataclasses import dataclass

MAX_TOOL = 1500      # tools are verbose; the head is enough to ground a number
MAX_SEG = 4000

CLASSES = ("USER", "TOOL", "ACT", "SELF")


@dataclass
class Segment:
    cls: str
    text: str

    def as_line(self) -> str:
        return f"[{self.cls}] {self.text}"


def as_evidence(segs: list[Segment]) -> str:
    """What the model sees. The class tags stay on — they are the judgement material."""
    return "\n".join(s.as_line() for s in segs)


class Source:
    """One kind of journal. Watermarks are opaque ints owned by the adapter."""
    name = "base"

    def matches(self, path: str) -> bool:
        raise NotImplementedError

    def key(self, path: str) -> str:
        """Watermark key. Must be unique across files (basenames often collide)."""
        return f"{self.name}:{os.path.abspath(path)}"

    def discover(self, root: str) -> list[str]:
        raise NotImplementedError

    def sip(self, path: str, start: int, limit_chars: int) -> tuple[list[Segment], int]:
        """Read past the watermark. Returns (segments, new watermark)."""
        raise NotImplementedError

    def claim_bound(self, path: str, start: int, budget_chars: int) -> tuple[int, int]:
        """Reserve a stretch before drinking it, so parallel runs never overlap.
        Returns (end watermark, approximate chars in the stretch)."""
        raise NotImplementedError


# ── Claude Code / plain JSONL transcripts (append-only → byte watermark) ─────

class ClaudeCodeSource(Source):
    """`~/.claude/projects/<project>/<session>.jsonl`, one JSON event per line."""
    name = "claude"

    def matches(self, path: str) -> bool:
        return path.endswith(".jsonl")

    def key(self, path: str) -> str:
        return "claude:" + os.path.basename(path)

    def discover(self, root: str) -> list[str]:
        return sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True),
                      key=os.path.getmtime, reverse=True)

    @staticmethod
    def _text_of(part) -> str:
        if isinstance(part, str):
            return part
        if isinstance(part, dict):
            if part.get("type") == "text":
                return part.get("text") or ""
            if part.get("type") == "tool_result":
                c = part.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return " ".join(x.get("text", "") for x in c if isinstance(x, dict))
        return ""

    def sip(self, path: str, start: int, limit_chars: int) -> tuple[list[Segment], int]:
        segs: list[Segment] = []
        total = 0
        with open(path, "rb") as h:
            h.seek(start)
            while True:
                line = h.readline()
                if not line:
                    break
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                t = d.get("type")
                c = (d.get("message") or {}).get("content")
                parts = c if isinstance(c, list) else ([c] if isinstance(c, str) else [])
                for p in parts:
                    cls = None
                    if t == "user":
                        cls = "TOOL" if (isinstance(p, dict) and p.get("type") == "tool_result") else "USER"
                    elif t == "assistant":
                        if isinstance(p, dict) and p.get("type") == "text":
                            cls = "SELF"
                        elif isinstance(p, dict) and p.get("type") == "tool_use":
                            txt = f"{p.get('name')} {json.dumps(p.get('input', {}), ensure_ascii=False)[:600]}"
                            segs.append(Segment("ACT", txt)); total += len(txt)
                            continue
                    if not cls:
                        continue
                    txt = self._text_of(p).strip()
                    if not txt or "system-reminder" in txt or txt.startswith("<local-command"):
                        continue
                    if cls == "TOOL":
                        txt = txt[:MAX_TOOL]
                    segs.append(Segment(cls, txt[:MAX_SEG]))
                    total += min(len(txt), MAX_SEG)
                if total >= limit_chars:
                    return segs, h.tell()
            return segs, h.tell()

    def claim_bound(self, path: str, start: int, budget_chars: int) -> tuple[int, int]:
        size = os.path.getsize(path)
        end = min(start + budget_chars, size)
        return end, max(0, end - start)


# ── DeepSeek Harness sessions (zstd, rewritten → sequence watermark) ─────────

class DshSource(Source):
    """`<DSH_HOME>/sessions/<dir>/session.jsonl.zstd`.

    The archive is REWRITTEN, not appended, so a byte offset lies. The event
    `seq` counter is the honest watermark. The key must include the session
    directory: every file is literally named `session.jsonl.zstd`.
    """
    name = "dsh"

    def matches(self, path: str) -> bool:
        return path.endswith(".jsonl.zstd")

    def key(self, path: str) -> str:
        return "dsh:" + os.path.basename(os.path.dirname(path))

    def discover(self, root: str) -> list[str]:
        return sorted(glob.glob(os.path.join(root, "**", "session.jsonl.zstd"), recursive=True),
                      key=os.path.getmtime, reverse=True)

    @staticmethod
    def _lines(path: str):
        p = subprocess.run(["zstd", "-dc", path], capture_output=True, timeout=300)
        for line in p.stdout.splitlines():
            try:
                yield json.loads(line)
            except ValueError:
                continue

    @staticmethod
    def _classify(d: dict) -> Segment | None:
        t = d.get("type")
        data = d.get("data") or {}
        if t == "user/message":
            if (data.get("source") or {}).get("kind") != "user":
                return None                       # injected context is not the human
            txt = " ".join(c.get("text", "") for c in (data.get("content") or [])
                           if isinstance(c, dict) and c.get("type") == "text").strip()
            if not txt or txt.startswith("<system-reminder") or txt.startswith("Current runtime context"):
                return None
            return Segment("USER", txt)
        if t == "assistant/chunk":
            c = data.get("chunk") or {}
            if c.get("type") == "block-end":
                b = c.get("block") or {}
                if b.get("type") == "text" and (b.get("text") or "").strip():
                    return Segment("SELF", b["text"].strip())
            return None                            # reasoning blocks are dropped
        if t == "tool/call":
            return Segment("ACT", f"{data.get('name')} {(data.get('arguments') or '')[:600]}")
        if t == "tool/result":
            parts = []
            for c in (data.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool-result":
                    for cc in c.get("content") or []:
                        if isinstance(cc, dict) and cc.get("type") == "text":
                            parts.append(cc.get("text", ""))
            txt = "\n".join(parts).strip()
            return Segment("TOOL", txt) if txt else None
        return None

    def sip(self, path: str, start: int, limit_chars: int) -> tuple[list[Segment], int]:
        segs: list[Segment] = []
        total, last = 0, start
        for d in self._lines(path):
            seq = d.get("seq")
            if seq is None or seq <= start:
                continue
            last = max(last, seq)
            s = self._classify(d)
            if not s:
                continue
            s.text = s.text[:MAX_TOOL if s.cls == "TOOL" else MAX_SEG]
            segs.append(s)
            total += len(s.text)
            if total >= limit_chars:
                break
        return segs, last

    def claim_bound(self, path: str, start: int, budget_chars: int) -> tuple[int, int]:
        total, end = 0, start
        for d in self._lines(path):
            seq = d.get("seq")
            if seq is None or seq <= start:
                continue
            s = self._classify(d)
            if s:
                total += min(len(s.text), MAX_TOOL if s.cls == "TOOL" else MAX_SEG)
            end = max(end, seq)
            if total >= budget_chars:
                break
        return end, total


# ── Plain text / markdown notes (append-only → byte watermark) ──────────────

class TextSource(Source):
    """A directory of notes or logs. Everything is [USER] — a human wrote it.
    Useful for distilling meeting notes, diaries, or exported chat logs."""
    name = "text"

    def matches(self, path: str) -> bool:
        return path.endswith((".md", ".txt", ".log"))

    def key(self, path: str) -> str:
        return "text:" + os.path.abspath(path)

    def discover(self, root: str) -> list[str]:
        out = []
        for ext in ("*.md", "*.txt", "*.log"):
            out += glob.glob(os.path.join(root, "**", ext), recursive=True)
        return sorted(out, key=os.path.getmtime, reverse=True)

    def sip(self, path: str, start: int, limit_chars: int) -> tuple[list[Segment], int]:
        with open(path, "rb") as h:
            h.seek(start)
            raw = h.read(limit_chars * 4).decode("utf-8", errors="ignore")
            pos = h.tell()
        segs = [Segment("USER", p.strip()[:MAX_SEG]) for p in raw.split("\n\n") if p.strip()]
        return segs, pos

    def claim_bound(self, path: str, start: int, budget_chars: int) -> tuple[int, int]:
        size = os.path.getsize(path)
        end = min(start + budget_chars, size)
        return end, max(0, end - start)


SOURCES: dict[str, Source] = {s.name: s for s in (ClaudeCodeSource(), DshSource(), TextSource())}


def source_for(path: str) -> Source | None:
    for s in (SOURCES["dsh"], SOURCES["claude"], SOURCES["text"]):
        if s.matches(path):
            return s
    return None


def _inside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) \
            == os.path.realpath(root)
    except (ValueError, OSError):
        return False


def discover_all(roots: dict, exclude_roots: list[str] | None = None) -> list[str]:
    """Journals to drink from, newest first — today's decisions are worth the most.

    `roots` maps a source kind to either a path or a table::

        {"dsh": "~/dsh/sessions"}
        {"dsh": {"root": "~/dsh/sessions-maker",
                 "include_glob": ["**/session.jsonl.zstd"],
                 "exclude_glob": ["**/scratch/**"]}}

    Per-source globs exist because one sessions directory usually holds every mode's
    conversations. Pointing two stores at it distils all of them into both: the memory
    directories are separate but the INTAKE is shared, and contamination happens there.

    `exclude_roots` (the store directories) is defence in depth: a journal root that
    contains a store would re-ingest memories as if a human had written them, which
    launders model-written text into [USER] evidence. The registry refuses that overlap
    at load; this catches it again at discovery.
    """
    files: list[str] = []
    for kind, spec in (roots or {}).items():
        src = SOURCES.get(kind)
        if not src:
            continue
        if isinstance(spec, dict):
            root = os.path.expanduser(str(spec.get("root", "")))
            include = spec.get("include_glob") or []
            exclude = spec.get("exclude_glob") or []
        else:
            root, include, exclude = os.path.expanduser(str(spec)), [], []
        if not root or not os.path.isdir(root):
            continue
        found = src.discover(root)
        if include:
            keep: list[str] = []
            for pat in include:
                keep += glob.glob(os.path.join(root, pat), recursive=True)
            found = [f for f in found if f in set(keep)]
        for pat in exclude:
            dropped = set(glob.glob(os.path.join(root, pat), recursive=True))
            found = [f for f in found if f not in dropped]
        files += found
    for root in (exclude_roots or []):
        files = [f for f in files if not _inside(f, root)]
    return sorted(set(files), key=os.path.getmtime, reverse=True)
