"""The richness gauge (plan §15), pinned against synthetic `_still` logs whose
numbers are known in advance: every metric computed from lines the writers would
have produced, a malformed line counted and skipped rather than fatal, the
"not recorded" cells that refuse to invent proxies, and the §15 warning firing on
the exact up-and-down pattern and on nothing else. No model, and the gauge never
writes — the directory is compared before and after.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.cli import main          # noqa: E402
from distill_kura.richness import gauge, table   # noqa: E402
from distill_kura.store import Store       # noqa: E402


def make_store(tmp_path, name="m") -> Store:
    s = Store(name=name, path=str(tmp_path / name))
    s.init_files()
    return s


def write(tmp_path, name, lines, store="m"):
    still = tmp_path / store / "_still"
    with open(os.path.join(str(still), name), "a", encoding="utf-8") as f:
        for l in lines:
            f.write((l if isinstance(l, str) else json.dumps(l, ensure_ascii=False)) + "\n")


NOW = datetime.now(timezone.utc)


def iso(days=0, hours=0, full=False):
    t = (NOW - timedelta(days=days, hours=hours))
    return t.isoformat() if full else t.isoformat()[:19]


def metric_row(days, **kw):
    r = {"source_key": "j.log", "segments": 0, "by_class": {}, "candidates": 0,
         "gated_kept": 0, "gated_dropped": 0, "ideas": 0,
         "cue_receipts": 0, "cue_receipt_failures": 0, "at": iso(days)}
    r.update(kw)
    return r


def test_candidate_rate_with_per_source_spread(tmp_path):
    s = make_store(tmp_path)
    write(tmp_path, "metrics.jsonl", [
        metric_row(2, source_key="a.log", segments=10, candidates=5, gated_kept=3),
        metric_row(1, source_key="b.log", segments=10, candidates=1),
        metric_row(1, source_key="b.log", segments=5, candidates=1),
    ])
    m = gauge(s)["metrics"]
    cr = m["candidate_rate"]
    assert cr["candidates"] == 7 and cr["segments"] == 25
    assert cr["rate"] == round(7 / 25, 4) and cr["batches"] == 3
    assert cr["per_source"]["a.log"] == {"candidates": 5, "segments": 10,
                                         "rate": 0.5}
    assert cr["per_source"]["b.log"]["rate"] == round(2 / 15, 4)


def test_rejection_reasons_and_unverified_numbers(tmp_path):
    s = make_store(tmp_path)
    write(tmp_path, "dropped.jsonl", [
        {"topic": "x", "why_dropped": "quotes not found in the raw material",
         "unverified_numbers": True, "at": iso(1, full=True)},
        {"topic": "y", "why_dropped": "COVERED by alpha", "reason": "same fact",
         "routing_cues_refused": {"zap": "ambiguous"}, "at": iso(1, full=True)},
        {"topic": "z", "why_dropped": "quotes not found in the raw material",
         "at": iso(1, full=True)},
    ])
    rj = gauge(s)["metrics"]["rejections"]
    assert rj["dropped"] == 3
    assert rj["by_why_dropped"]["quotes not found in the raw material"] == 2
    assert rj["reason_present"] == 1
    assert rj["unverified_numbers"] == 1


def test_user_survival_says_not_recorded_not_invented(tmp_path):
    """The writer never records per-class keeps; the gauge must say so, not
    extrapolate a proxy from gated_kept."""
    s = make_store(tmp_path)
    write(tmp_path, "metrics.jsonl", [
        metric_row(1, by_class={"USER": 4, "SELF": 1}, gated_kept=3)])
    us = gauge(s)["metrics"]["user_evidence_survival"]
    assert us["user_candidates"] == 4 and us["user_candidates_n"] == 5
    assert us["gated_kept"] == 3
    assert us["user_kept"] == "not recorded by the writer"


def test_seed_retreat_and_toss_verdicts(tmp_path):
    s = make_store(tmp_path)
    write(tmp_path, "metrics.jsonl", [metric_row(1, ideas=3, gated_kept=6)])
    write(tmp_path, "seeds.jsonl", [
        {"kind": "IDEA", "text": "t", "from": "brain/spot", "topic": "",
         "at": iso(1), "confirmed": None},
        {"kind": "IDEA", "text": "u", "from": "brain/spot", "topic": "",
         "at": iso(1), "confirmed": {"how": "h", "memory": "a", "at": iso(0)}},
    ])
    write(tmp_path, "tossed.jsonl", [
        {"slug": "d1", "verdict": "TOSS", "why": "beyond evidence", "at": iso(1)},
        {"slug": "d2", "verdict": "QUARANTINE", "why": "no valid mark", "at": iso(1)},
    ])
    sj = gauge(s)["metrics"]["self_judgement_seeds"]
    assert sj["ideas"] == 3 and sj["gated_kept"] == 6 and sj["retreat_rate"] == 0.5
    assert sj["seed_entries"] == 2 and sj["seeds_confirmed"] == 1
    assert sj["tossed"]["by_verdict"] == {"TOSS": 1, "QUARANTINE": 1}


def test_callsign_receipts_and_refusals(tmp_path):
    s = make_store(tmp_path)
    write(tmp_path, "metrics.jsonl", [
        metric_row(2, cue_receipts=2, cue_receipt_failures=1),
        metric_row(1, cue_receipts=0, cue_receipt_failures=3),
    ])
    write(tmp_path, "dropped.jsonl", [
        {"topic": "y", "why_dropped": "COVERED by alpha",
         "routing_cues_refused": {"zap": "ambiguous", "zed": "two stores"},
         "at": iso(1, full=True)},
    ])
    cs = gauge(s)["metrics"]["callsigns"]
    assert cs["cue_receipts"] == 2 and cs["cue_receipt_failures"] == 4
    assert cs["dropped_with_routing_cues_refused"] == 1 and cs["dropped_n"] == 1
    assert cs["refused_cue_texts"] == 2


def test_unreachable_per_variant_and_sha_identity(tmp_path):
    s = make_store(tmp_path)
    write(tmp_path, "worldline-traces.jsonl", [
        {"case": "c1", "resident_variant": "canonical", "resident_sha": "aaa",
         "skipped": None, "remembered_but_unreachable": True,
         "target_reached": False, "thinker_calls": 1},
        {"case": "c2", "resident_variant": "canonical", "resident_sha": "aaa",
         "skipped": None, "remembered_but_unreachable": False,
         "target_reached": True, "thinker_calls": 0},
        {"case": "c3", "resident_variant": "woven", "resident_sha": "bbb",
         "skipped": "target slugs absent from this store",
         "remembered_but_unreachable": False, "target_reached": False,
         "thinker_calls": 0},
        {"case": "c4", "resident_variant": "woven", "resident_sha": "bbb",
         "skipped": None, "remembered_but_unreachable": False,
         "target_reached": True, "thinker_calls": 0},
    ])
    m = gauge(s)["metrics"]
    assert m["remembered_but_unreachable"]["canonical"] == {
        "remembered_but_unreachable": 1, "runnable": 2, "rate": 0.5}
    # the sha is the identity; the variant name is only its label
    assert m["trigger_shortening_by_sha"]["aaa"]["variants"] == ["canonical"]
    assert m["trigger_shortening_by_sha"]["aaa"]["target_reached"] == 1
    assert m["trigger_shortening_by_sha"]["aaa"]["runnable"] == 2
    assert "bbb" in m["trigger_shortening_by_sha"]


def test_fallback_thinker_reads_and_how_not_recorded(tmp_path):
    s = make_store(tmp_path)
    write(tmp_path, "reads.jsonl", [
        {"at": 1750000000, "why": "recall", "n": ["alpha"]},
        {"at": 1750000001, "why": "hand", "n": ["beta"]},
    ])
    fb = gauge(s)["metrics"]["fallback_thinker"]
    assert fb["reads_by_why"] == {"recall": 1, "hand": 1} and fb["reads_n"] == 2
    assert fb["how_fastpath_vs_meaning"].startswith("not recorded")
    assert "Store.note_read" in fb["how_fastpath_vs_meaning"]


def test_malformed_line_counted_and_skipped(tmp_path):
    s = make_store(tmp_path)
    write(tmp_path, "metrics.jsonl", [
        metric_row(1, segments=10, candidates=5),
        '{"source_key": "j.log", "segments": 3, ',      # torn line
        "not json at all",
    ])
    r = gauge(s)
    assert r["files"]["metrics.jsonl"] == {"lines": 3, "bad_lines": 2}
    assert r["metrics"]["candidate_rate"]["segments"] == 10   # the good row still counts


def test_absent_files_give_empty_cells_not_crashes(tmp_path):
    s = make_store(tmp_path)
    m = gauge(s)["metrics"]
    assert m["candidate_rate"]["rate"] is None and m["candidate_rate"]["batches"] == 0
    assert m["rejections"]["dropped"] == 0
    assert m["remembered_but_unreachable"] == {}
    assert m["fallback_thinker"]["reads_n"] == 0
    assert gauge(s)["windows"]            # a range with no rows still tiles


def test_since_bounds_the_range(tmp_path):
    s = make_store(tmp_path)
    write(tmp_path, "metrics.jsonl", [
        metric_row(30, segments=10, candidates=5),
        metric_row(1, segments=10, candidates=1),
    ])
    r = gauge(s, since_days=7)
    cr = r["metrics"]["candidate_rate"]
    assert cr["candidates"] == 1 and cr["segments"] == 10
    assert r["range"]["since_days"] == 7


def test_warn_fires_only_on_rejections_up_AND_kept_down(tmp_path):
    s = make_store(tmp_path)
    # inside the two newest windows (7d default), relative to the newest row
    write(tmp_path, "metrics.jsonl", [
        metric_row(9, gated_kept=5, gated_dropped=1),   # previous window
        metric_row(1, gated_kept=2, gated_dropped=4),   # latest window
    ])
    r = gauge(s, window_days=7)
    assert r["warnings"] == ["WARN richness: rejections up (1→4), kept down (5→2) "
                             "— is the gate learning honesty or silence?"]
    # rejections up but kept up too: honesty learning, no warning
    s2 = make_store(tmp_path, "m2")
    write(tmp_path, "metrics.jsonl", [
        metric_row(9, gated_kept=5, gated_dropped=1),
        metric_row(1, gated_kept=6, gated_dropped=4),
    ], store="m2")
    assert gauge(s2, window_days=7)["warnings"] == []
    # rejections down while kept fell: the gate is not the suspect, no warning
    s3 = make_store(tmp_path, "m3")
    write(tmp_path, "metrics.jsonl", [
        metric_row(9, gated_kept=5, gated_dropped=4),
        metric_row(1, gated_kept=2, gated_dropped=1),
    ], store="m3")
    assert gauge(s3, window_days=7)["warnings"] == []
    # previous window empty: nothing to compare against, no warning
    s4 = make_store(tmp_path, "m4")
    write(tmp_path, "metrics.jsonl", [metric_row(1, gated_kept=2, gated_dropped=4)],
          store="m4")
    assert gauge(s4, window_days=7)["warnings"] == []


def test_json_shape_via_cli(tmp_path, capsys):
    s = make_store(tmp_path)
    write(tmp_path, "metrics.jsonl", [metric_row(1, segments=4, candidates=2)])
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[stores.m]
path = "{tmp_path / 'm'}"
[models.thinker]
url = "http://127.0.0.1:9/v1"
model = "none"
""", encoding="utf-8")
    rc = main(["-c", str(cfg), "-s", "m", "metrics", "richness", "--json"])
    assert rc == 0                       # a gauge, not a gate: even the WARN exits 0
    r = json.loads(capsys.readouterr().out)
    assert set(r) == {"store", "retired", "range", "files", "metrics", "windows", "warnings"}
    assert r["store"] == "m"
    assert r["files"]["metrics.jsonl"] == {"lines": 1, "bad_lines": 0}
    assert set(r["metrics"]) == {"candidate_rate", "rejections",
                                 "user_evidence_survival", "self_judgement_seeds",
                                 "callsigns", "remembered_but_unreachable",
                                 "trigger_shortening_by_sha", "fallback_thinker"}
    for w in r["windows"]:
        assert set(w) == {"start", "end", "metrics"}
    rc = main(["-c", str(cfg), "-s", "m", "metrics", "richness"])
    out = capsys.readouterr().out
    assert rc == 0 and "candidate rate" in out and "not recorded by the writer" in out


def test_gauge_never_writes(tmp_path):
    """A gauge that writes anything into the store it watches has confused reading
    with writing — the one rule every lookup into a store answers to."""
    s = make_store(tmp_path)
    write(tmp_path, "metrics.jsonl", [metric_row(1, segments=4, candidates=2)])
    still = s.still
    before = sorted((p, os.stat(os.path.join(still, p)).st_mtime_ns,
                     os.stat(os.path.join(still, p)).st_size)
                    for p in os.listdir(still))
    gauge(s, since_days=90, window_days=1)
    table(gauge(s))
    after = sorted((p, os.stat(os.path.join(still, p)).st_mtime_ns,
                    os.stat(os.path.join(still, p)).st_size)
                   for p in os.listdir(still))
    assert before == after
    assert sorted(os.listdir(tmp_path / "m")) == ["MEMORY.md", "_still"]
