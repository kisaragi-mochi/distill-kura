"""Classified `.evidence.jsonl` intake: schema, safety, watermarks, discovery."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.distill.sources import (                  # noqa: E402
    MAX_SEG,
    MAX_TOOL,
    ClaudeCodeSource,
    EvidenceJsonlSource,
    discover_all,
    source_for,
)
from distill_kura.store import Store                          # noqa: E402


def _event(cls: str, text: str, **extra) -> dict:
    base = {
        "schema_version": 1,
        "event_id": "evt-1",
        "session_id": "sess-1",
        "turn_id": "turn-1",
        "class": cls,
        "text": text,
        "timestamp": "2026-08-27T00:00:00Z",
    }
    base.update(extra)
    return base


def _write(path, *events, trailing_partial: str | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        if trailing_partial is not None:
            f.write(trailing_partial)


# ── classification ──────────────────────────────────────────────────────────

def test_all_four_evidence_classes_are_preserved(tmp_path):
    p = tmp_path / "j.evidence.jsonl"
    _write(p,
           _event("USER", "human words"),
           _event("SELF", "assistant prose", event_id="e2"),
           _event("ACT", "tool_call read_file", event_id="e3"),
           _event("TOOL", "file contents here", event_id="e4"))
    segs, end = EvidenceJsonlSource().sip(str(p), 0, 10_000)
    assert [(s.cls, s.text) for s in segs] == [
        ("USER", "human words"),
        ("SELF", "assistant prose"),
        ("ACT", "tool_call read_file"),
        ("TOOL", "file contents here"),
    ]
    assert end == p.stat().st_size


def test_source_for_prefers_evidence_over_claude(tmp_path):
    p = tmp_path / "x.evidence.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    assert source_for(str(p)).name == "evidence"
    assert not ClaudeCodeSource().matches(str(p))


def test_claude_discover_omits_evidence_jsonl(tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    (root / "plain.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "tagged.evidence.jsonl").write_text("{}\n", encoding="utf-8")
    found = ClaudeCodeSource().discover(str(root))
    assert str(root / "plain.jsonl") in found
    assert str(root / "tagged.evidence.jsonl") not in found


# ── malformed / invalid lines ─────────────────────────────────────────────

def test_malformed_json_is_skipped_not_reclassified(tmp_path):
    p = tmp_path / "bad.evidence.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write("not json at all\n")
        f.write(json.dumps(_event("USER", "good line", event_id="e2")) + "\n")
    segs, _ = EvidenceJsonlSource().sip(str(p), 0, 10_000)
    assert len(segs) == 1 and segs[0].cls == "USER" and segs[0].text == "good line"


def test_unknown_schema_version_is_skipped(tmp_path):
    p = tmp_path / "v2.evidence.jsonl"
    _write(p, {**_event("USER", "future"), "schema_version": 2})
    assert EvidenceJsonlSource().sip(str(p), 0, 10_000)[0] == []


def test_unknown_class_is_skipped_not_mapped_to_user(tmp_path):
    p = tmp_path / "cls.evidence.jsonl"
    _write(p, {**_event("USER", "ok"), "class": "SYSTEM"})
    segs, _ = EvidenceJsonlSource().sip(str(p), 0, 10_000)
    assert segs == []


def test_missing_or_blank_ids_are_skipped(tmp_path):
    p = tmp_path / "ids.evidence.jsonl"
    _write(p,
           {**_event("USER", "no event_id"), "event_id": ""},
           {**_event("USER", "no session"), "session_id": "  "},
           {**_event("USER", "missing turn"), "turn_id": None})
    assert EvidenceJsonlSource().sip(str(p), 0, 10_000)[0] == []


def test_blank_or_non_string_text_is_skipped(tmp_path):
    p = tmp_path / "text.evidence.jsonl"
    _write(p,
           {**_event("USER", ""), "text": ""},
           {**_event("USER", "x"), "text": 42},
           {**_event("USER", "x"), "text": "   "})
    assert EvidenceJsonlSource().sip(str(p), 0, 10_000)[0] == []


def test_oversized_text_is_skipped_not_truncated(tmp_path):
    p = tmp_path / "big.evidence.jsonl"
    _write(p,
           _event("USER", "x" * (MAX_SEG + 1), event_id="big-user"),
           _event("TOOL", "y" * (MAX_TOOL + 1), event_id="big-tool"),
           _event("USER", "fits", event_id="ok"))
    segs, _ = EvidenceJsonlSource().sip(str(p), 0, 10_000)
    assert len(segs) == 1 and segs[0].text == "fits"


# ── incomplete final line / watermark ─────────────────────────────────────

def test_incomplete_final_line_does_not_advance_watermark(tmp_path):
    p = tmp_path / "tail.evidence.jsonl"
    good = json.dumps(_event("USER", "first"))
    with open(p, "wb") as f:
        f.write((good + "\n").encode())
        f.write(b'{"schema_version": 1, "event_id": "e2"')
    src = EvidenceJsonlSource()
    segs1, pos1 = src.sip(str(p), 0, 10_000)
    assert len(segs1) == 1 and segs1[0].text == "first"
    assert pos1 == len(good) + 1

    with open(p, "ab") as f:
        f.write(b', "session_id": "s", "turn_id": "t", "class": "USER", "text": "second"}\n')
    segs2, pos2 = src.sip(str(p), pos1, 10_000)
    assert len(segs2) == 1 and segs2[0].text == "second"
    assert pos2 == p.stat().st_size


def test_watermark_resume_skips_already_drunk_lines(tmp_path):
    p = tmp_path / "resume.evidence.jsonl"
    _write(p,
           _event("USER", "one", event_id="e1"),
           _event("USER", "two", event_id="e2"))
    src = EvidenceJsonlSource()
    first_line = (json.dumps(_event("USER", "one", event_id="e1")) + "\n").encode()
    segs, pos = src.sip(str(p), len(first_line), 10_000)
    assert len(segs) == 1 and segs[0].text == "two"
    assert pos == p.stat().st_size


def test_duplicate_basenames_get_distinct_watermark_keys(tmp_path):
    src = EvidenceJsonlSource()
    a = tmp_path / "a" / "j.evidence.jsonl"
    b = tmp_path / "b" / "j.evidence.jsonl"
    _write(a, _event("USER", "from a"))
    _write(b, _event("USER", "from b"))
    assert src.key(str(a)) != src.key(str(b))
    assert src.key(str(a)).startswith("evidence:")


# ── discovery: include/exclude and store isolation ──────────────────────────

def test_include_and_exclude_globs_narrow_evidence_root(tmp_path):
    _write(tmp_path / "logs" / "keep" / "a.evidence.jsonl", _event("USER", "keep me"))
    _write(tmp_path / "logs" / "skip" / "b.evidence.jsonl", _event("USER", "drop me"))
    root = str(tmp_path / "logs")
    kept = discover_all({"evidence": {"root": root, "exclude_glob": ["skip/**"]}})
    assert len(kept) == 1 and "keep" in kept[0]
    only = discover_all({"evidence": {"root": root, "include_glob": ["skip/**"]}})
    assert len(only) == 1 and "skip" in only[0]


def test_a_hardlinked_memory_in_an_evidence_root_is_not_discovered(tmp_path):
    st = Store(name="s", path=str(tmp_path / "s"))
    st.init_files()
    st.remember("mem", "d", "MODEL-WRITTEN MEMORY BODY")
    jr = tmp_path / "jr"
    jr.mkdir()
    _write(jr / "real.evidence.jsonl", _event("USER", "human note"))
    os.link(st.file_of("mem"), jr / "hardlinked.evidence.jsonl")
    found = discover_all({"evidence": str(jr)}, exclude_roots=[st.path])
    names = [os.path.basename(f) for f in found]
    assert names == ["real.evidence.jsonl"]
