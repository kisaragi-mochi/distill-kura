"""Classified `.evidence.jsonl` intake: schema, safety, watermarks, discovery."""
from __future__ import annotations

import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.distill.pipeline import Distiller, MIN_DRINK   # noqa: E402
from distill_kura.distill.sources import (                  # noqa: E402
    MAX_ID,
    MAX_LINE,
    MAX_SEG,
    MAX_TOOL,
    SCAN_LIMIT,
    ClaudeCodeSource,
    EvidenceJsonlSource,
    IntakeReport,
    Segment,
    Source,
    call_sip,
    discover_all,
    source_for,
)
from distill_kura.distill.watermark import Watermarks          # noqa: E402
from distill_kura.registry import Registry                    # noqa: E402
from distill_kura.store import Store                          # noqa: E402
from distill_kura.thinker import Models                       # noqa: E402


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
    end, _ = src.claim_bound(str(p), 0, 10_000)
    assert end == pos1 < p.stat().st_size

    with open(p, "ab") as f:
        f.write(b', "session_id": "s", "turn_id": "t", "class": "USER", '
                b'"text": "second", "timestamp": "2026-08-27T00:00:01Z"}\n')
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


# ── timestamp contract ──────────────────────────────────────────────────────

def test_missing_or_non_string_timestamp_is_skipped(tmp_path):
    p = tmp_path / "ts.evidence.jsonl"
    _write(p,
           {k: v for k, v in _event("USER", "no ts").items() if k != "timestamp"},
           {**_event("USER", "numeric", event_id="e2"), "timestamp": 1724716800},
           {**_event("USER", "null", event_id="e3"), "timestamp": None})
    segs, _ = EvidenceJsonlSource().sip(str(p), 0, 10_000)
    assert segs == []


def test_malformed_and_naive_timestamps_are_skipped(tmp_path):
    p = tmp_path / "naive.evidence.jsonl"
    _write(p,
           {**_event("USER", "date-only"), "timestamp": "2026-08-27"},
           {**_event("USER", "naive", event_id="e2"), "timestamp": "2026-08-27T00:00:00"},
           {**_event("USER", "space", event_id="e3"), "timestamp": "2026-08-27 00:00:00Z"},
           {**_event("USER", "leap", event_id="e4"), "timestamp": "2026-06-30T23:59:60Z"})
    assert EvidenceJsonlSource().sip(str(p), 0, 10_000)[0] == []


def test_rfc3339_z_offset_and_fractional_timestamps_are_accepted(tmp_path):
    p = tmp_path / "ok-ts.evidence.jsonl"
    _write(p,
           _event("USER", "z"),
           _event("USER", "frac", event_id="e2", timestamp="2026-08-27T00:00:00.123Z"),
           _event("USER", "offset", event_id="e3", timestamp="2026-08-27T09:00:00+09:00"))
    segs, _ = EvidenceJsonlSource().sip(str(p), 0, 10_000)
    assert [s.text for s in segs] == ["z", "frac", "offset"]


def test_timestamp_is_a_gate_not_a_stored_or_filled_field(tmp_path):
    """A missing timestamp is skipped. The clock is not a fallback, and the
    segment never gains a timestamp field — evidence is not rewritten."""
    p = tmp_path / "clock.evidence.jsonl"
    _write(p, {k: v for k, v in _event("USER", "no ts").items() if k != "timestamp"})
    segs, _ = EvidenceJsonlSource().sip(str(p), 0, 10_000)
    assert segs == []
    assert set(Segment.__dataclass_fields__) == {"cls", "text"}


def test_json_true_is_not_schema_version_one(tmp_path):
    p = tmp_path / "bool.evidence.jsonl"
    _write(p, {**_event("USER", "bool ver"), "schema_version": True})
    assert EvidenceJsonlSource().sip(str(p), 0, 10_000)[0] == []


# ── bounded parsing ─────────────────────────────────────────────────────────

def test_oversized_ids_are_skipped_not_truncated(tmp_path):
    p = tmp_path / "ids-size.evidence.jsonl"
    _write(p,
           _event("USER", "too long", event_id="e" * (MAX_ID + 1)),
           _event("USER", "ordinary-uuid", event_id="550e8400-e29b-41d4-a716-446655440000"))
    segs, _ = EvidenceJsonlSource().sip(str(p), 0, 10_000)
    assert len(segs) == 1 and segs[0].text == "ordinary-uuid"


def test_oversized_line_is_skipped_without_json_loads(tmp_path, monkeypatch):
    p = tmp_path / "huge.evidence.jsonl"
    huge = b'{"schema_version": 1, "event_id": "e", "text": "' + (b"A" * (MAX_LINE + 50)) + b'"}\n'
    good = (json.dumps(_event("USER", "after the dump", event_id="ok")) + "\n").encode()
    with open(p, "wb") as f:
        f.write(huge)
        f.write(good)
    called = {"n": 0}
    real = json.loads

    def spy(s, *a, **k):
        called["n"] += 1
        if isinstance(s, (bytes, bytearray)) and len(s) > MAX_LINE:
            raise AssertionError("json.loads on an oversized line")
        if isinstance(s, str) and len(s.encode()) > MAX_LINE:
            raise AssertionError("json.loads on an oversized line")
        return real(s, *a, **k)
    monkeypatch.setattr("distill_kura.distill.sources.json.loads", spy)
    segs, end = EvidenceJsonlSource().sip(str(p), 0, 10_000)
    assert len(segs) == 1 and segs[0].text == "after the dump"
    assert end == p.stat().st_size
    assert called["n"] == 1


# ── runtime reporting ───────────────────────────────────────────────────────

def test_skips_are_reported_without_payloads_or_paths(tmp_path):
    p = tmp_path / "secret-dir" / "j.evidence.jsonl"
    secret = "credential-hunter2-not-for-logs"
    os.makedirs(p.parent, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("this is not json and contains " + secret + "\n")
        f.write(json.dumps({**_event("USER", "future " + secret), "schema_version": 2}) + "\n")
        f.write(json.dumps({**_event("USER", "sys " + secret), "class": "SYSTEM"}) + "\n")
        f.write(json.dumps({**_event("USER", "blank " + secret), "event_id": ""}) + "\n")
        f.write(json.dumps({**_event("USER", ""), "text": ""}) + "\n")
        f.write(json.dumps(_event("USER", "x" * (MAX_SEG + 1))) + "\n")
        f.write(json.dumps({k: v for k, v in _event("USER", "no ts " + secret).items()
                            if k != "timestamp"}) + "\n")
        f.write('{"schema_version": 1, "event_id": "partial-' + secret + '"')
    report = IntakeReport()
    segs, pos = EvidenceJsonlSource().sip(str(p), 0, 10_000, report=report)
    assert segs == []
    assert pos < p.stat().st_size
    assert report.skipped.get("malformed")
    assert report.skipped.get("unknown_version")
    assert report.skipped.get("unknown_class")
    assert report.skipped.get("blank")
    assert report.skipped.get("oversized")
    assert report.skipped.get("missing")
    assert report.skipped.get("partial")
    blob = json.dumps(report.as_dict())
    assert secret not in blob
    assert "secret-dir" not in blob
    assert str(p) not in blob
    assert len(report.samples) <= IntakeReport.MAX_SAMPLES


def test_reporting_is_bounded_on_a_flood_of_junk(tmp_path):
    p = tmp_path / "flood.evidence.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for _ in range(200):
            f.write("not-json\n")
    report = IntakeReport()
    EvidenceJsonlSource().sip(str(p), 0, 10_000, report=report)
    assert report.skipped["malformed"] == 200
    assert len(report.samples) == IntakeReport.MAX_SAMPLES


def _blob(tag: str, n: int = 3500) -> str:
    return (tag + " " + ("x" * n))[:n]


def _distiller(tmp_path, journal_dir, chunk_chars=4000):
    st = Store(name="s", path=str(tmp_path / "s"))
    st.init_files()
    models = Models.from_config({})
    reg = Registry(stores={"s": st}, modes={}, models=models, default="s",
                   raw={"distill": {"journals": {"evidence": str(journal_dir)}}})
    return Distiller(reg, st, chunk_chars=chunk_chars), st


# ── real claim + sip_one durability ─────────────────────────────────────────

def test_sip_one_partial_tail_does_not_skip_unread_bytes(tmp_path):
    """Luna: claim reserved byte 192, sip returned 156, max-forward kept 192."""
    jr = tmp_path / "jr"
    jr.mkdir()
    p = jr / "j.evidence.jsonl"
    events = [_event("USER", _blob(f"keep-{i}"), event_id=f"e{i}") for i in range(2)]
    _write(p, *events, trailing_partial='{"schema_version": 1, "event_id": "tail')
    complete = sum(len((json.dumps(e) + "\n").encode()) for e in events)
    assert complete < p.stat().st_size
    d, st = _distiller(tmp_path, jr, chunk_chars=20_000)
    got = d.sip_one()
    assert got is not None
    segs, path, key = got
    assert [s.text[:6] for s in segs] == ["keep-0", "keep-1"]
    mark = d.marks.read()[key]
    assert mark == complete
    assert mark < p.stat().st_size
    # completing the tail must be drinkable, not skipped
    with open(p, "ab") as f:
        f.write(b"\n")
        f.write((json.dumps(_event("USER", _blob("keep-2"), event_id="e2")) + "\n").encode())
        f.write((json.dumps(_event("USER", _blob("keep-3"), event_id="e3")) + "\n").encode())
    got2 = d.sip_one()
    assert got2 is not None
    assert {s.text[:6] for s in got2[0]} == {"keep-2", "keep-3"}


def test_sip_one_large_complete_events_are_not_skipped_by_overclaim(tmp_path):
    """Luna: claim reserved 22000 while sip consumed only through 12309.

    A 2.2× byte/char fudge in claim() reserved past sip's char-budget stop;
    max-forward then skipped the unread complete record forever.
    """
    jr = tmp_path / "jr"
    jr.mkdir()
    p = jr / "j.evidence.jsonl"
    events = [_event("USER", _blob(f"evt-{i}"), event_id=f"e{i}") for i in range(4)]
    _write(p, *events)
    d, _ = _distiller(tmp_path, jr, chunk_chars=4000)
    first = d.sip_one()
    second = d.sip_one()
    assert first is not None and second is not None
    assert [s.text[:5] for s in first[0] + second[0]] == ["evt-0", "evt-1", "evt-2", "evt-3"]
    assert d.sip_one() is None
    assert d.marks.read()[first[2]] == p.stat().st_size


def test_sip_one_resume_does_not_redrink_or_skip(tmp_path):
    jr = tmp_path / "jr"
    jr.mkdir()
    p = jr / "j.evidence.jsonl"
    events = [_event("USER", _blob(f"n{i}"), event_id=f"e{i}") for i in range(4)]
    _write(p, *events)
    d, _ = _distiller(tmp_path, jr, chunk_chars=4000)
    first = d.sip_one()
    second = d.sip_one()
    third = d.sip_one()
    assert first and second
    assert third is None
    seen = [s.text[:2] for s in first[0] + second[0]]
    assert seen == ["n0", "n1", "n2", "n3"]
    assert d.marks.read()[first[2]] == p.stat().st_size


def test_evidence_claim_bound_is_where_the_read_stops(tmp_path):
    p = tmp_path / "bound.evidence.jsonl"
    events = [_event("USER", _blob(f"b{i}"), event_id=f"e{i}") for i in range(6)]
    _write(p, *events)
    src = EvidenceJsonlSource()
    for budget in (500, 4000, 20_000):
        end, _ = src.claim_bound(str(p), 0, budget)
        _, stop = src.sip(str(p), 0, budget)
        assert end == stop


def test_parallel_claims_reserve_disjoint_complete_records(tmp_path):
    jr = tmp_path / "jr"
    jr.mkdir()
    p = jr / "j.evidence.jsonl"
    events = [_event("USER", _blob(f"p{i}"), event_id=f"e{i}") for i in range(8)]
    _write(p, *events)
    d, st = _distiller(tmp_path, jr, chunk_chars=4000)
    first = d.marks.claim([str(p)], 4000, MIN_DRINK)
    assert first is not None
    path_a, start_a, end_a, src_a = first
    key = src_a.key(str(p))
    reserved_a = d.marks.read()[key]
    second = d.marks.claim([str(p)], 4000, MIN_DRINK)
    assert second is not None
    path_b, start_b, end_b, src_b = second
    reserved_b = d.marks.read()[key]
    assert start_a < start_b
    assert reserved_a <= start_b
    assert reserved_b > start_b
    # sip of each reservation drinks only that stretch
    src = EvidenceJsonlSource()
    a, a_end = call_sip(src, path_a, start_a, 4000, bound_end=reserved_a)
    b, b_end = call_sip(src, path_b, start_b, 4000, bound_end=reserved_b)
    assert a_end == reserved_a and b_end == reserved_b
    assert {s.text[:2] for s in a}.isdisjoint({s.text[:2] for s in b})
    d.marks.advance(src.key(str(p)), a_end)
    d.marks.advance(src.key(str(p)), b_end)
    assert d.marks.read()[src.key(str(p))] == max(a_end, b_end)


def test_parallel_sip_one_does_not_overlap_or_skip(tmp_path):
    jr = tmp_path / "jr"
    jr.mkdir()
    p = jr / "j.evidence.jsonl"
    events = [_event("USER", _blob(f"t{i}"), event_id=f"e{i}") for i in range(8)]
    _write(p, *events)
    d1, st = _distiller(tmp_path, jr, chunk_chars=4000)
    d2, _ = _distiller(tmp_path, jr, chunk_chars=4000)
    d2.marks = d1.marks
    bag = []

    def worker(d):
        got = d.sip_one()
        if got:
            bag.append([s.text[:2] for s in got[0]])

    threads = [threading.Thread(target=worker, args=(d,)) for d in (d1, d2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    flat = [x for row in bag for x in row]
    assert len(flat) == len(set(flat))
    assert set(flat) <= {f"t{i}" for i in range(8)}
    # leftover complete records are still drinkable
    rest = d1.sip_one()
    if rest:
        flat += [s.text[:2] for s in rest[0]]
    rest2 = d1.sip_one()
    if rest2:
        flat += [s.text[:2] for s in rest2[0]]
    assert sorted(flat) == [f"t{i}" for i in range(8)]


def test_sip_one_writes_bounded_intake_without_payloads(tmp_path):
    jr = tmp_path / "secret-root"
    jr.mkdir()
    p = jr / "j.evidence.jsonl"
    secret = "token-abc-not-for-logs"
    events = [_event("USER", _blob(f"ok{i}"), event_id=f"e{i}") for i in range(2)]
    with open(p, "w", encoding="utf-8") as f:
        f.write("not json " + secret + "\n")
        for e in events:
            f.write(json.dumps(e) + "\n")
        f.write(json.dumps({**_event("USER", "sys " + secret), "class": "SYSTEM"}) + "\n")
    d, st = _distiller(tmp_path, jr, chunk_chars=20_000)
    got = d.sip_one()
    assert got is not None
    intake = open(os.path.join(st.still, "intake.jsonl"), encoding="utf-8").read()
    assert "malformed" in intake and "unknown_class" in intake
    assert secret not in intake
    assert "secret-root" not in intake
    assert str(p) not in intake
    assert "j.evidence.jsonl" in intake


def test_intake_write_failure_does_not_break_sip_one(tmp_path):
    jr = tmp_path / "jr"
    jr.mkdir()
    p = jr / "j.evidence.jsonl"
    _write(p,
           _event("USER", _blob("a"), event_id="e1"),
           _event("USER", _blob("b"), event_id="e2"))
    with open(p, "a", encoding="utf-8") as f:
        f.write("not-json\n")
    d, st = _distiller(tmp_path, jr, chunk_chars=20_000)
    os.makedirs(os.path.join(st.still, "intake.jsonl"))
    got = d.sip_one()
    assert got is not None
    assert [s.text[:1] for s in got[0]] == ["a", "b"]


# ── Luna rework: reserved end, legacy sip, bounded tail ─────────────────────

class _LegacySource(Source):
    """Pre-existing custom adapter: sip(path, start, limit_chars) only."""
    name = "legacy"

    def matches(self, path: str) -> bool:
        return path.endswith(".legacy.txt")

    def key(self, path: str) -> str:
        return "legacy:" + os.path.abspath(path)

    def discover(self, root: str) -> list[str]:
        return []

    def sip(self, path: str, start: int, limit_chars: int) -> tuple[list[Segment], int]:
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(limit_chars)
        return [Segment("USER", data.decode(errors="replace"))], start + len(data)

    def claim_bound(self, path: str, start: int, budget_chars: int) -> tuple[int, int]:
        end = min(os.path.getsize(path), start + budget_chars)
        return end, max(0, end - start)


def test_call_sip_legacy_three_arg_source_ignores_report(tmp_path):
    p = tmp_path / "x.legacy.txt"
    p.write_bytes(b"hello world")
    src = _LegacySource()
    segs, stop = call_sip(src, str(p), 0, 5, report=IntakeReport())
    assert len(segs) == 1 and segs[0].text == "hello"
    assert stop == 5


def test_sip_one_with_legacy_three_arg_source(tmp_path, monkeypatch):
    legacy = _LegacySource()
    jr = tmp_path / "jr"
    jr.mkdir()
    p = jr / "note.legacy.txt"
    p.write_bytes(b"abcdefghij")
    d, st = _distiller(tmp_path, jr, chunk_chars=4)
    monkeypatch.setattr("distill_kura.distill.pipeline.MIN_DRINK", 1)
    monkeypatch.setattr(d, "files", lambda session=None: [str(p)])
    real_source_for = source_for

    def fake_source_for(path):
        return legacy if legacy.matches(path) else real_source_for(path)

    monkeypatch.setattr("distill_kura.distill.sources.source_for", fake_source_for)
    monkeypatch.setattr("distill_kura.distill.watermark.source_for", fake_source_for)
    got = d.sip_one()
    assert got is not None
    segs, _, key = got
    assert segs[0].text == "abcd"
    assert d.marks.read()[key] == 4


def test_first_sip_respects_reserved_end_after_second_claim(tmp_path):
    """Adversarial: claim, append, second claim, first sip — no overlap or skip."""
    jr = tmp_path / "jr"
    jr.mkdir()
    p = jr / "j.evidence.jsonl"
    events = [_event("USER", _blob(f"c{i}"), event_id=f"e{i}") for i in range(4)]
    _write(p, *events)
    d, _ = _distiller(tmp_path, jr, chunk_chars=4000)
    first = d.marks.claim([str(p)], 4000, MIN_DRINK)
    assert first is not None
    path_a, start_a, end_a, src_a = first
    key = src_a.key(str(p))
    with open(p, "ab") as f:
        for i in range(4):
            f.write((json.dumps(_event("USER", _blob(f"late{i}"), event_id=f"late{i}"))
                     + "\n").encode())
    second = d.marks.claim([str(p)], 4000, MIN_DRINK)
    assert second is not None
    path_b, start_b, end_b, src_b = second
    assert start_b == end_a
    src = EvidenceJsonlSource()
    a, a_end = call_sip(src, path_a, start_a, 4000, bound_end=end_a)
    b, b_end = call_sip(src, path_b, start_b, 4000, bound_end=end_b)
    assert a_end == end_a <= start_b
    assert b_end == end_b
    assert {s.text[:4] for s in a}.isdisjoint({s.text[:4] for s in b})
    d.marks.advance(key, a_end)
    d.marks.advance(key, b_end)
    assert d.marks.read()[key] == max(a_end, b_end)


def test_unterminated_oversized_tail_stays_bounded_per_attempt(tmp_path, monkeypatch):
    p = tmp_path / "tail.evidence.jsonl"
    p.write_bytes(b"x" * (MAX_LINE * 10))
    consumed: list[int] = []
    real_open = open

    def open_wrapper(path, mode="r", *args, **kwargs):
        fh = real_open(path, mode, *args, **kwargs)
        if os.path.abspath(str(path)) == os.path.abspath(str(p)) and "b" in mode:
            total = 0
            base_readline = fh.readline

            def readline(size=-1):
                nonlocal total
                chunk = base_readline(size)
                if chunk:
                    total += len(chunk)
                return chunk

            fh.readline = readline
            base_close = fh.close

            def close():
                consumed.append(total)
                return base_close()

            fh.close = close
        return fh

    monkeypatch.setattr("builtins.open", open_wrapper)
    src = EvidenceJsonlSource()
    segs, pos = src.sip(str(p), 0, 10_000)
    assert segs == [] and pos == 0
    assert consumed and consumed[0] <= MAX_LINE * 10 + MAX_LINE
    segs2, pos2 = src.sip(str(p), 0, 10_000)
    assert segs2 == [] and pos2 == 0
    assert len(consumed) == 2 and consumed[1] <= MAX_LINE * 10 + MAX_LINE


def test_completed_oversized_line_then_valid_event(tmp_path):
    p = tmp_path / "mix.evidence.jsonl"
    huge = (b'{"schema_version": 1, "event_id": "e", "session_id": "s", "turn_id": "t", '
            b'"class": "USER", "text": "' + (b"A" * (MAX_LINE + 50))
            + b'", "timestamp": "2026-08-27T00:00:00Z"}\n')
    good = (json.dumps(_event("USER", "after", event_id="ok")) + "\n").encode()
    with open(p, "wb") as f:
        f.write(huge)
        f.write(good)
    segs, end = EvidenceJsonlSource().sip(str(p), 0, 10_000)
    assert len(segs) == 1 and segs[0].text == "after"
    assert end == p.stat().st_size

def test_unterminated_tail_larger_than_scan_limit_stays_bounded(tmp_path, monkeypatch):
    """Garbage tail with no newline: capped per attempt, watermark unchanged."""
    p = tmp_path / "tail.evidence.jsonl"
    p.write_bytes(b"x" * (SCAN_LIMIT + 50_000))
    consumed: list[int] = []
    real_open = open

    def open_wrapper(path, mode="r", *args, **kwargs):
        fh = real_open(path, mode, *args, **kwargs)
        if os.path.abspath(str(path)) == os.path.abspath(str(p)) and "b" in mode:
            total = 0
            base_readline = fh.readline

            def readline(size=-1):
                nonlocal total
                chunk = base_readline(size)
                if chunk:
                    total += len(chunk)
                return chunk

            fh.readline = readline
            base_close = fh.close

            def close():
                consumed.append(total)
                return base_close()

            fh.close = close
        return fh

    monkeypatch.setattr("builtins.open", open_wrapper)
    src = EvidenceJsonlSource()
    segs, pos = src.sip(str(p), 0, 10_000)
    assert segs == [] and pos == 0
    assert consumed and consumed[0] <= SCAN_LIMIT + MAX_LINE
    segs2, pos2 = src.sip(str(p), 0, 10_000)
    assert segs2 == [] and pos2 == 0
    assert len(consumed) == 2 and consumed[1] <= SCAN_LIMIT + MAX_LINE


def test_unterminated_json_tail_larger_than_scan_limit_stays_bounded(tmp_path, monkeypatch):
    """Unterminated JSON-shaped tail: same bounded cap, no { prefix escape hatch."""
    p = tmp_path / "json-tail.evidence.jsonl"
    p.write_bytes(b'{"schema_version": 1, "event_id": "e"' + b"x" * (SCAN_LIMIT + 50_000))
    consumed: list[int] = []
    real_open = open

    def open_wrapper(path, mode="r", *args, **kwargs):
        fh = real_open(path, mode, *args, **kwargs)
        if os.path.abspath(str(path)) == os.path.abspath(str(p)) and "b" in mode:
            total = 0
            base_readline = fh.readline

            def readline(size=-1):
                nonlocal total
                chunk = base_readline(size)
                if chunk:
                    total += len(chunk)
                return chunk

            fh.readline = readline
            base_close = fh.close

            def close():
                consumed.append(total)
                return base_close()

            fh.close = close
        return fh

    monkeypatch.setattr("builtins.open", open_wrapper)
    src = EvidenceJsonlSource()
    segs, pos = src.sip(str(p), 0, 10_000)
    assert segs == [] and pos == 0
    assert consumed and consumed[0] <= SCAN_LIMIT + MAX_LINE
    segs2, pos2 = src.sip(str(p), 0, 10_000)
    assert segs2 == [] and pos2 == 0
    assert len(consumed) == 2 and consumed[1] <= SCAN_LIMIT + MAX_LINE


def test_completed_nonjson_oversized_line_past_scan_limit_then_valid_event(tmp_path):
    """Completed non-JSON line longer than SCAN_LIMIT must not block later evidence."""
    p = tmp_path / "xline.evidence.jsonl"
    huge = b"x" * (SCAN_LIMIT + 50_000) + b"\n"
    good = (json.dumps(_event("USER", "after", event_id="ok")) + "\n").encode()
    with open(p, "wb") as f:
        f.write(huge)
        f.write(good)
    segs, end = EvidenceJsonlSource().sip(str(p), 0, 10_000)
    assert len(segs) == 1 and segs[0].text == "after"
    assert end == p.stat().st_size


def test_completed_oversized_line_past_scan_limit_then_valid_event(tmp_path):
    """Completed invalid line longer than SCAN_LIMIT must not block later evidence."""
    p = tmp_path / "past-scan.evidence.jsonl"
    prefix = (b'{"schema_version": 1, "event_id": "e", "session_id": "s", "turn_id": "t", '
              b'"class": "USER", "text": "')
    suffix = b'", "timestamp": "2026-08-27T00:00:00Z"}\n'
    text_len = SCAN_LIMIT - len(prefix) - len(suffix) + 50_000
    huge = prefix + (b"A" * text_len) + suffix
    assert len(huge) > SCAN_LIMIT and huge.endswith(b"\n")
    good = (json.dumps(_event("USER", "after", event_id="ok")) + "\n").encode()
    with open(p, "wb") as f:
        f.write(huge)
        f.write(good)
    segs, end = EvidenceJsonlSource().sip(str(p), 0, 10_000)
    assert len(segs) == 1 and segs[0].text == "after"
    assert end == p.stat().st_size


def test_completed_oversized_past_scan_limit_respects_bound_end(tmp_path):
    """Skip a completed oversized line inside the reservation; do not drink past it."""
    p = tmp_path / "bound-scan.evidence.jsonl"
    prefix = (b'{"schema_version": 1, "event_id": "e", "session_id": "s", "turn_id": "t", '
              b'"class": "USER", "text": "')
    suffix = b'", "timestamp": "2026-08-27T00:00:00Z"}\n'
    text_len = SCAN_LIMIT - len(prefix) - len(suffix) + 50_000
    huge = prefix + (b"A" * text_len) + suffix
    in_bound = (json.dumps(_event("USER", "inside", event_id="in")) + "\n").encode()
    outside = (json.dumps(_event("USER", "outside", event_id="out")) + "\n").encode()
    with open(p, "wb") as f:
        f.write(huge)
        f.write(in_bound)
        f.write(outside)
    bound_end = len(huge) + len(in_bound)
    segs, end = EvidenceJsonlSource().sip(str(p), 0, 10_000, bound_end=bound_end)
    assert len(segs) == 1 and segs[0].text == "inside"
    assert end == bound_end
    segs2, end2 = EvidenceJsonlSource().sip(str(p), bound_end, 10_000)
    assert len(segs2) == 1 and segs2[0].text == "outside"
    assert end2 == p.stat().st_size
