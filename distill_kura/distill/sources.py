"""Where the raw material comes from, and how it is CLASSED.

Everything the distiller reads is turned into segments carrying an evidence class:

    [USER]  the human's own words      — primary evidence
    [TOOL]  machine output             — the ONLY place a measured number may come from
    [ACT]   a tool the agent invoked   — evidence that an action was taken
    [SELF]  the agent's own prose      — a judgement worth keeping, never a bare fact

Reasoning / thinking blocks are dropped: an inner monologue is not evidence.
Injected content (system reminders, runtime context) is not the human speaking.

Four adapters ship here; add your own by subclassing `Source` and registering it
in `SOURCES`. `watermark` semantics differ per adapter, so each one owns them:
byte offset for append-only files, sequence number for rewritten archives.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime

MAX_TOOL = 1500      # tools are verbose; the head is enough to ground a number
MAX_SEG = 4000
MAX_LINE = 32 * 1024  # raw JSONL line, including the newline; bound before json.loads
MAX_ID = 256          # event_id / session_id / turn_id; oversized is skipped, never sliced
MAX_TIMESTAMP = 40    # RFC3339 date-time with timezone; ordinary values fit with room
MAX_CLASS = 32

CLASSES = ("USER", "TOOL", "ACT", "SELF")

# RFC3339 date-time with a timezone. The clock is not consulted; a miss is a skip.
# Accepted: 2026-08-27T00:00:00Z | .123Z | +09:00 | -00:00
# Rejected: missing/non-string, date-only, naive, space-separator, leap seconds,
# offsets with seconds. Parsed by datetime.fromisoformat after Z → +00:00.
_RFC3339 = re.compile(
    r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})\Z"
)


def _rfc3339(value: object) -> bool:
    if not isinstance(value, str) or not value or len(value) > MAX_TIMESTAMP:
        return False
    if _RFC3339.match(value) is None:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


@dataclass
class IntakeSkip:
    """One skipped record. Offset and size only — never the payload."""
    reason: str
    at: int
    size: int


@dataclass
class IntakeReport:
    """Bounded skip accounting for one sip. A diagnostic must not flood or throw.

    Reasons are a closed set: malformed, unknown_version, unknown_class, missing,
    blank, oversized, partial, invalid. Samples cap at MAX_SAMPLES; counts do not.
    Nothing here is a path, an id, a credential, or evidence text.
    """
    skipped: dict[str, int] = field(default_factory=dict)
    samples: list[IntakeSkip] = field(default_factory=list)
    MAX_SAMPLES = 16
    MAX_SIZE_REPORTED = MAX_LINE

    def note(self, reason: str, at: int, size: int) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1
        if len(self.samples) < self.MAX_SAMPLES:
            self.samples.append(IntakeSkip(
                reason, max(0, int(at)),
                min(max(0, int(size)), self.MAX_SIZE_REPORTED)))

    @property
    def total(self) -> int:
        return sum(self.skipped.values())

    def as_dict(self) -> dict:
        return {
            "skipped": dict(self.skipped),
            "samples": [{"reason": s.reason, "at": s.at, "size": s.size}
                        for s in self.samples],
        }


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

    def sip(self, path: str, start: int, limit_chars: int,
            until: int | None = None, report: IntakeReport | None = None
            ) -> tuple[list[Segment], int]:
        """Read past the watermark. Returns (segments, new watermark).

        `until` is the reserved end from `claim` — sip must not drink past it.
        `report` collects bounded skip reasons; it must never carry payloads.
        """
        raise NotImplementedError

    def claim_bound(self, path: str, start: int, budget_chars: int) -> tuple[int, int]:
        """Reserve a stretch before drinking it, so parallel runs never overlap.

        `budget_chars` is the same budget `sip` will receive. The returned end
        must be a watermark `sip` will actually reach: `claim` writes it before
        the drink, and `advance` only takes max(), so over-reservation skips
        unread bytes forever. Returns (end watermark, approximate chars).
        """
        raise NotImplementedError


# ── Claude Code / plain JSONL transcripts (append-only → byte watermark) ─────

class ClaudeCodeSource(Source):
    """`~/.claude/projects/<project>/<session>.jsonl`, one JSON event per line."""
    name = "claude"

    def matches(self, path: str) -> bool:
        return path.endswith(".jsonl") and not path.endswith(".evidence.jsonl")

    def key(self, path: str) -> str:
        return "claude:" + os.path.basename(path)

    def discover(self, root: str) -> list[str]:
        found = [f for f in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)
                 if not f.endswith(".evidence.jsonl")]
        return sorted(found, key=os.path.getmtime, reverse=True)

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

    def sip(self, path: str, start: int, limit_chars: int,
            until: int | None = None, report: IntakeReport | None = None
            ) -> tuple[list[Segment], int]:
        return self._drink(path, start, limit_chars, until=until)

    def _drink(self, path: str, start: int, limit_chars: int,
               until: int | None = None) -> tuple[list[Segment], int]:
        segs: list[Segment] = []
        total = 0
        with open(path, "rb") as h:
            h.seek(start)
            while True:
                line_start = h.tell()
                if until is not None and line_start >= until:
                    return segs, line_start
                line = h.readline()
                if not line:
                    break
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                t = d.get("type")
                # A subagent's transcript records the PARENT MODEL's instructions as
                # `type: user` with `isSidechain: true`. That text is model-written:
                # classed [USER] it would be the human's own words, and "the owner
                # approved X" in a delegation prompt would pass the gate as a decision.
                # Tool results stay [TOOL]; everything else in a sidechain is [SELF].
                side = bool(d.get("isSidechain"))
                c = (d.get("message") or {}).get("content")
                parts = c if isinstance(c, list) else ([c] if isinstance(c, str) else [])
                for p in parts:
                    cls = None
                    if t == "user":
                        if isinstance(p, dict) and p.get("type") == "tool_result":
                            cls = "TOOL"
                        else:
                            cls = "SELF" if side else "USER"
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
        # Same walk sip uses, same budget Distiller will pass to sip. A 2.2× byte
        # estimate reserved past the char-budget record end; max-forward then
        # skipped the unread head of the next compact JSONL line forever.
        # approx is bytes walked, not truncated segment text: MAX_SEG caps a
        # long USER line at 4000 chars, and using that sum as min_chars would
        # refuse a journal whose padding was sized to pass MIN_DRINK on bytes.
        _, end = self._drink(path, start, budget_chars)
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

    def sip(self, path: str, start: int, limit_chars: int,
            until: int | None = None, report: IntakeReport | None = None
            ) -> tuple[list[Segment], int]:
        segs: list[Segment] = []
        total, last = 0, start
        for d in self._lines(path):
            seq = d.get("seq")
            if seq is None or seq <= start:
                continue
            if until is not None and seq > until:
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


# ── Classified evidence JSONL (append-only → byte watermark) ───────────────

class EvidenceJsonlSource(Source):
    """`*.evidence.jsonl` — one versioned, class-tagged event per line.

    Writers append complete JSON objects; a crash may leave a partial final line.
    The watermark stops before that line so the next append can finish it. Invalid
    lines are dropped, never reclassified, and counted on an IntakeReport.

    Minimum shape (schema_version 1)::

        {"schema_version": 1, "event_id": "…", "session_id": "…", "turn_id": "…",
         "class": "USER"|"SELF"|"ACT"|"TOOL", "text": "…",
         "timestamp": "RFC3339 date-time with timezone"}

    `timestamp` is a JSON string matching `YYYY-MM-DDTHH:MM:SS[.frac](Z|±HH:MM)`,
    parsed by `datetime.fromisoformat` after a trailing `Z` is rewritten to
    `+00:00`. Date-only, naive, non-string, space-separator, leap-second, and
    `±HH:MM:SS` values are rejected. The clock is not consulted and the record
    is not rewritten; the timestamp is a gate, not a stored field.

    Identity fields (`event_id`, `session_id`, `turn_id`) are at most 256
    characters; a raw line is at most 32 KiB. Oversized values are skipped,
    never truncated into a valid identity. Ordinary UUIDs, ULIDs, and hex
    digests fit.
    """
    name = "evidence"

    def matches(self, path: str) -> bool:
        return path.endswith(".evidence.jsonl")

    def key(self, path: str) -> str:
        return "evidence:" + os.path.abspath(path)

    def discover(self, root: str) -> list[str]:
        return sorted(glob.glob(os.path.join(root, "**", "*.evidence.jsonl"), recursive=True),
                      key=os.path.getmtime, reverse=True)

    @staticmethod
    def _note(report: IntakeReport | None, reason: str, at: int, size: int) -> None:
        if report is not None:
            report.note(reason, at, size)

    @staticmethod
    def _read_record(h) -> tuple[bytes | None, str]:
        """Bounded read of one JSONL record. Never readline()s the rest of the file.

        Status: 'eof' | 'ok' | 'partial' | 'oversized'.
        'ok' payload includes the newline and is at most MAX_LINE bytes.
        """
        chunk = h.readline(MAX_LINE + 1)
        if not chunk:
            return None, "eof"
        if chunk.endswith(b"\n"):
            if len(chunk) > MAX_LINE:
                return None, "oversized"
            return chunk, "ok"
        if len(chunk) <= MAX_LINE:
            return chunk, "partial"
        while True:
            more = h.readline(MAX_LINE)
            if not more:
                return None, "partial"
            if more.endswith(b"\n"):
                return None, "oversized"

    @staticmethod
    def _parse(raw: bytes) -> tuple[str | None, Segment | None]:
        try:
            d = json.loads(raw)
        except ValueError:
            return "malformed", None
        if not isinstance(d, dict):
            return "malformed", None
        if "schema_version" not in d:
            return "missing", None
        ver = d.get("schema_version")
        # `True == 1` in Python; a JSON true must not pass as version 1.
        if type(ver) is not int or ver != 1:
            return "unknown_version", None
        if "class" not in d:
            return "missing", None
        cls = d.get("class")
        if not isinstance(cls, str):
            return "unknown_class", None
        if len(cls) > MAX_CLASS:
            return "oversized", None
        if cls not in CLASSES:
            return "unknown_class", None
        for name in ("event_id", "session_id", "turn_id"):
            if name not in d:
                return "missing", None
            val = d[name]
            if not isinstance(val, str):
                return "invalid", None
            if not val.strip():
                return "blank", None
            if len(val) > MAX_ID:
                return "oversized", None
        if "text" not in d:
            return "missing", None
        text = d.get("text")
        if not isinstance(text, str):
            return "invalid", None
        if not text.strip():
            return "blank", None
        cap = MAX_TOOL if cls == "TOOL" else MAX_SEG
        if len(text) > cap:
            return "oversized", None
        if "timestamp" not in d:
            return "missing", None
        ts = d.get("timestamp")
        if not isinstance(ts, str):
            return "invalid", None
        if not ts.strip():
            return "blank", None
        if len(ts) > MAX_TIMESTAMP:
            return "oversized", None
        if not _rfc3339(ts):
            return "invalid", None
        return None, Segment(cls, text.strip())

    def _drink(self, path: str, start: int, limit_chars: int,
               until: int | None = None,
               report: IntakeReport | None = None) -> tuple[list[Segment], int]:
        segs: list[Segment] = []
        total = 0
        with open(path, "rb") as h:
            h.seek(start)
            while True:
                line_start = h.tell()
                if until is not None and line_start >= until:
                    return segs, line_start
                raw, status = self._read_record(h)
                if status == "eof":
                    return segs, line_start
                if status == "partial":
                    self._note(report, "partial", line_start, len(raw or b""))
                    return segs, line_start
                if status == "oversized":
                    self._note(report, "oversized", line_start, MAX_LINE + 1)
                    continue
                reason, seg = self._parse(raw or b"")
                if reason:
                    self._note(report, reason, line_start, len(raw or b""))
                    continue
                assert seg is not None
                segs.append(seg)
                total += len(seg.text)
                if total >= limit_chars:
                    return segs, h.tell()
            return segs, h.tell()

    def sip(self, path: str, start: int, limit_chars: int,
            until: int | None = None, report: IntakeReport | None = None
            ) -> tuple[list[Segment], int]:
        return self._drink(path, start, limit_chars, until=until, report=report)

    def claim_bound(self, path: str, start: int, budget_chars: int) -> tuple[int, int]:
        # Same walk sip uses, same budget Distiller will pass to sip. A byte
        # estimate (file size, or start+budget) reserved past the last complete
        # record; max-forward then skipped the unread tail forever.
        segs, end = self._drink(path, start, budget_chars)
        text = sum(len(s.text) for s in segs)
        # Junk-only stretches still have to move: approx=0 would refuse the claim
        # and never report the skips. Bytes walked let the watermark advance.
        return end, text if text else max(0, end - start)


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

    def sip(self, path: str, start: int, limit_chars: int,
            until: int | None = None, report: IntakeReport | None = None
            ) -> tuple[list[Segment], int]:
        with open(path, "rb") as h:
            h.seek(start)
            cap = limit_chars * 4
            if until is not None:
                cap = min(cap, max(0, until - start))
            raw = h.read(cap).decode("utf-8", errors="ignore")
            pos = h.tell()
            if until is not None:
                pos = min(pos, until)
        segs = [Segment("USER", p.strip()[:MAX_SEG]) for p in raw.split("\n\n") if p.strip()]
        return segs, pos

    def claim_bound(self, path: str, start: int, budget_chars: int) -> tuple[int, int]:
        size = os.path.getsize(path)
        end = min(start + int(budget_chars * 2.2), size)
        return end, max(0, end - start)


SOURCES: dict[str, Source] = {
    s.name: s for s in (ClaudeCodeSource(), DshSource(), EvidenceJsonlSource(), TextSource())
}


def source_for(path: str) -> Source | None:
    for s in (SOURCES["dsh"], SOURCES["evidence"], SOURCES["claude"], SOURCES["text"]):
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
    # Path exclusion is not enough: a HARDLINK to a memory, sitting in an otherwise clean
    # journal root, is a different path to the same inode. It walked through and was
    # sipped as [USER] evidence — model-written memory laundered into the human's words,
    # which is the one thing the evidence gate exists to prevent. Compare identities.
    ids = set()
    for root in (exclude_roots or []):
        for p in glob.glob(os.path.join(root, "**", "*.md"), recursive=True):
            try:
                st = os.stat(p)
                ids.add((st.st_dev, st.st_ino))
            except OSError:
                continue
    if ids:
        keep = []
        for f in files:
            try:
                st = os.stat(f)
            except OSError:
                continue
            if (st.st_dev, st.st_ino) not in ids:
                keep.append(f)
        files = keep
    return sorted(set(files), key=os.path.getmtime, reverse=True)
