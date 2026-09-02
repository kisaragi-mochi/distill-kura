"""The richness gauge (plan §15): did the store STOP REMEMBERING LIES, or stop
remembering?

Rejections going up is ambiguous — the gate may be learning honesty, or the writer
may have gone quiet and the gate is all that is left. `gate rejection rate` alone
cannot tell those apart; only reading rejections against `gated_kept`, seed retreat
and what recall actually reached can. This module is pure aggregation over the
`_still/*.jsonl` logs their writers already record:

    metrics.jsonl             pipeline._metric — one row per batch
    dropped.jsonl             pipeline.run — gate rejections and COVERED verdicts
    tossed.jsonl              pipeline.drain — TOSS / QUARANTINE verdicts
    seeds.jsonl               Seeds.sow — ideas that must not pass as facts
    reads.jsonl               Store.note_read — what recall returned (never ranked on)
    worldline-traces.jsonl    worldline.run — one row per measured case

No model, never writes. Every number carries its denominator (n) — a bare ratio
invites exactly the misreading this gauge exists to prevent. A malformed line is
counted (`bad_lines`) and skipped, never fatal: a gauge that dies on one bad line
stops watching precisely when the store needs watching.

What is NOT here is as deliberate as what is: the recall path's "how" (fastpath vs
meaning) is computed in recall.py and dropped on the floor — no log carries it — so
that cell says "not recorded" and names the writer, rather than inventing a proxy.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone

from .store import Store

FILES = ("metrics.jsonl", "dropped.jsonl", "tossed.jsonl", "seeds.jsonl",
         "reads.jsonl", "worldline-traces.jsonl")

WARN_LINE = ("WARN richness: rejections up ({a}→{b}), kept down ({c}→{d}) — "
             "is the gate learning honesty or silence?")


def _read_jsonl(path: str) -> tuple[list[dict], int, int]:
    """(parsed rows, physical lines, bad lines). Blank lines are neither data nor
    lies, so they count as lines and move on; a non-blank line that will not parse
    is a bad line — counted, never fatal."""
    rows, lines, bad = [], 0, 0
    if not os.path.exists(path):
        return rows, lines, bad
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            lines += 1
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(r, dict):
                rows.append(r)
    return rows, lines, bad


def _row_time(row: dict) -> datetime | None:
    """UTC time of a row, or None. metrics/tossed/seeds write ISO[:19] (naive UTC),
    dropped writes full ISO with offset, reads writes an epoch int. worldline traces
    carry no timestamp at all — dateless rows stay in the whole range, out of windows."""
    at = row.get("at")
    if isinstance(at, (int, float)) and not isinstance(at, bool):
        try:
            return datetime.fromtimestamp(at, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(at, str) and at:
        t = at.replace("Z", "+00:00")
        try:
            d = datetime.fromisoformat(t)
        except ValueError:
            return None
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
    return None


def _num(v) -> int:
    """A count from a log that may hold anything — bench hit this with
    raw_tokens_est as a list, so arithmetic here coerces rather than crashes."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return 0
    return int(v)


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _metrics(rows: dict[str, list[dict]]) -> dict:
    """Every metric over one set of rows — used for the whole range and for each
    window, so a window is the same gauge at a smaller n, never a second impl."""
    m = rows.get("metrics.jsonl") or []
    d = rows.get("dropped.jsonl") or []
    to = rows.get("tossed.jsonl") or []
    s = rows.get("seeds.jsonl") or []
    rd = rows.get("reads.jsonl") or []
    wl = rows.get("worldline-traces.jsonl") or []
    out: dict = {}

    # 1. raw journal → candidate rate, with the per-source spread the source_key
    #    carries (a rate that mixes all journals hides the one that drifted).
    segs = sum(_num(r.get("segments")) for r in m)
    cands = sum(_num(r.get("candidates")) for r in m)
    per_source: dict[str, dict] = {}
    for r in m:
        k = str(r.get("source_key", ""))
        ps = per_source.setdefault(k, {"candidates": 0, "segments": 0})
        ps["candidates"] += _num(r.get("candidates"))
        ps["segments"] += _num(r.get("segments"))
    out["candidate_rate"] = {
        "candidates": cands, "segments": segs, "rate": _rate(cands, segs),
        "batches": len(m),
        "per_source": {k: {**v, "rate": _rate(v["candidates"], v["segments"])}
                       for k, v in sorted(per_source.items())},
    }

    # 2. gate rejection reasons — the shape of what the gate refuses, and how much
    #    of the refusal carried a reason at all.
    why = Counter(str(r.get("why_dropped") or "(no why_dropped)") for r in d)
    out["rejections"] = {
        "dropped": len(d),
        "by_why_dropped": dict(why.most_common()),
        "reason_present": sum(1 for r in d if "reason" in r),
        "unverified_numbers": sum(1 for r in d if r.get("unverified_numbers")),
        # the writer's own per-batch count — gated_kept's exact counterpart, and
        # the number the §15 window warning reads (dropped.jsonl also holds COVERED
        # verdicts, which are novelty rejections, not gate rejections)
        "gated_dropped": sum(_num(r.get("gated_dropped")) for r in m),
    }

    # 3. USER evidence survival. The writer records by_class at spot time and
    #    gated_kept at the end of the batch, but never keeps-per-class — and this
    #    gauge does not invent a proxy for a number nobody wrote down.
    user = sum(_num((r.get("by_class") or {}).get("USER")) for r in m)
    kept = sum(_num(r.get("gated_kept")) for r in m)
    out["user_evidence_survival"] = {
        "user_candidates": user, "user_candidates_n": sum(
            _num(x) for r in m for x in (r.get("by_class") or {}).values()),
        "gated_kept": kept,
        "user_kept": "not recorded by the writer",
    }

    # 4. SELF judgement and seeds: candidates that retreated to the seed field
    #    instead of becoming facts, and what the drain tossed.
    kinds = Counter(str(r.get("kind") or "(no kind)") for r in s)
    out["self_judgement_seeds"] = {
        "ideas": sum(_num(r.get("ideas")) for r in m),
        "gated_kept": kept, "retreat_rate": None,          # filled below
        "seed_entries": len(s), "seed_kinds": dict(kinds.most_common()),
        "seeds_confirmed": sum(1 for r in s if r.get("confirmed")),
        "tossed": {"total": len(to),
                   "by_verdict": dict(Counter(str(r.get("verdict") or "(no verdict)")
                                              for r in to).most_common())},
    }
    out["self_judgement_seeds"]["retreat_rate"] = _rate(
        out["self_judgement_seeds"]["ideas"], kept)

    # 5. callsigns: proposed-and-accepted receipts, failures, refusals the gate
    #    recorded on dropped candidates.
    receipts = sum(_num(r.get("cue_receipts")) for r in m)
    failures = sum(_num(r.get("cue_receipt_failures")) for r in m)
    refused_rows = [r for r in d if r.get("routing_cues_refused")]
    out["callsigns"] = {
        "cue_receipts": receipts, "cue_receipt_failures": failures,
        "batches_n": len(m),
        "dropped_with_routing_cues_refused": len(refused_rows), "dropped_n": len(d),
        "refused_cue_texts": sum(len(r.get("routing_cues_refused") or {}) for r in d),
    }

    # 6. remembered but unreachable: the map said it, recall could not open it —
    #    per resident variant, since a variant IS the map that was worn.
    by_var: dict[str, dict] = {}
    for r in wl:
        v = str(r.get("resident_variant") or "(no variant)")
        bv = by_var.setdefault(v, {"remembered_but_unreachable": 0, "runnable": 0})
        if r.get("skipped"):
            continue                       # a case against another house never ran
        bv["runnable"] += 1
        if r.get("remembered_but_unreachable"):
            bv["remembered_but_unreachable"] += 1
    out["remembered_but_unreachable"] = {
        v: {**bv, "rate": _rate(bv["remembered_but_unreachable"], bv["runnable"])}
        for v, bv in sorted(by_var.items())}

    # 7. trigger shortening, before/after a re-weave: the sha IS the identity of
    #    the map a trace measured against; the variant name is only its label.
    by_sha: dict[str, dict] = {}
    for r in wl:
        if r.get("skipped"):
            continue
        sha = str(r.get("resident_sha") or "(no sha)")
        bs = by_sha.setdefault(sha, {"variants": set(), "target_reached": 0,
                                     "runnable": 0})
        bs["variants"].add(str(r.get("resident_variant") or "(no variant)"))
        bs["runnable"] += 1
        if r.get("target_reached"):
            bs["target_reached"] += 1
    out["trigger_shortening_by_sha"] = {
        sha: {"variants": sorted(v["variants"]), "target_reached": v["target_reached"],
              "runnable": v["runnable"],
              "rate": _rate(v["target_reached"], v["runnable"])}
        for sha, v in sorted(by_sha.items())}

    # 8. fallback thinker: reads by why, thinker bails-in per trace — and the
    #    honest hole: nothing logs recall's own "how".
    out["fallback_thinker"] = {
        "reads_by_why": dict(Counter(str(r.get("why") or "(no why)")
                                     for r in rd).most_common()),
        "reads_n": len(rd),
        "traces_with_thinker_call": sum(1 for r in wl
                                        if not r.get("skipped")
                                        and _num(r.get("thinker_calls")) > 0),
        "traces_runnable": sum(1 for r in wl if not r.get("skipped")),
        "how_fastpath_vs_meaning": ("not recorded — recall's how (fastpath vs "
                                    "meaning) reaches no log; Store.note_read "
                                    "(reads.jsonl) is the writer that would have "
                                    "to carry it"),
    }
    return out


def _windows(end: datetime, floor: datetime | None, wdays: float,
             first_seen: datetime | None) -> list[tuple[datetime, datetime]]:
    """Consecutive [start, end) buckets, newest first, covering either the --since
    floor or (with no --since) back to the first dated row."""
    out = []
    e = end
    while True:
        s = e - timedelta(days=wdays)
        lo = s if floor is None or s >= floor else floor
        out.append((lo, e))
        e = s
        if floor is not None and e <= floor:
            break
        if floor is None and first_seen is not None and e <= first_seen:
            break
        if floor is None and first_seen is None:
            break
    return out


def gauge(store: Store, since_days: float | None = None,
          window_days: float = 7.0) -> dict:
    """The full richness dict: whole range, rolling windows, warnings. Read-only —
    it opens the `_still` logs and touches nothing."""
    now = datetime.now(timezone.utc)
    floor = now - timedelta(days=since_days) if since_days is not None else None

    files: dict[str, dict] = {}
    rows: dict[str, list[dict]] = {}
    dated: list[datetime] = []
    for name in FILES:
        rs, lines, bad = _read_jsonl(os.path.join(store.still, name))
        kept_rows = []
        for r in rs:
            t = _row_time(r)
            if t is not None:
                dated.append(t)
                if floor is not None and t < floor:
                    continue               # --since bounds the range it can prove
            kept_rows.append(r)
        rows[name] = kept_rows
        files[name] = {"lines": lines, "bad_lines": bad}

    first_seen = min(dated) if dated else None
    last_seen = max(dated) if dated else None
    end = last_seen if last_seen is not None else now
    if floor is not None and floor > end:
        end = floor                     # a range with no rows still tiles honestly

    windows = []
    for i, (lo, hi) in enumerate(_windows(end, floor, window_days, first_seen)):
        # The newest window is closed on the right: the row that DEFINES the end of
        # the range must land in the window it defines, not fall between buckets.
        newest = i == 0

        def in_window(r, lo=lo, hi=hi, newest=newest):
            t = _row_time(r)
            if t is None:
                return False
            return lo <= t <= hi if newest else lo <= t < hi

        wr = {name: [r for r in rows[name] if in_window(r)] for name in FILES}
        windows.append({"start": lo.isoformat(timespec="seconds"),
                        "end": hi.isoformat(timespec="seconds"),
                        "metrics": _metrics(wr)})

    # §15's warning, computed not judged: rejections up AND kept down, window over
    # window, both windows populated. A gauge, not a gate — it never sets an exit code.
    warnings: list[str] = []
    if len(windows) >= 2:
        a, b = windows[0]["metrics"], windows[1]["metrics"]
        ka, da = a["candidate_rate"]["batches"], a["rejections"]["gated_dropped"]
        kb, db = b["candidate_rate"]["batches"], b["rejections"]["gated_dropped"]
        kept_a, kept_b = a["user_evidence_survival"]["gated_kept"], \
            b["user_evidence_survival"]["gated_kept"]
        if ka and kb and da > db and kept_a < kept_b:
            warnings.append(WARN_LINE.format(a=db, b=da, c=kept_b, d=kept_a))

    return {
        "store": store.name,
        # Read from the store, not from a log: a retirement is a state the index
        # carries, and a gauge that could only see it in a jsonl would miss every
        # transition a person made with `kura retire`.
        "retired": len(store.faced()),
        "range": {"window_days": window_days, "since_days": since_days,
                  "first_seen": first_seen.isoformat(timespec="seconds")
                  if first_seen else None,
                  "last_seen": last_seen.isoformat(timespec="seconds")
                  if last_seen else None},
        "files": files,
        "metrics": _metrics(rows),
        "windows": list(reversed(windows)),   # oldest first: a trend reads left→right
        "warnings": warnings,
    }


def _cell(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def _frac(num: int, den: int, rate) -> str:
    return f"{_cell(rate)}  ({num}/{den})"


def table(r: dict) -> str:
    """The human surface: the whole range first, then one line per window so the
    trend — the thing §15 warns about — is visible without a screen full of JSON."""
    m = r["metrics"]
    rg = r["range"]
    out = [f"richness  store={r['store']}  "
           f"range={rg['first_seen'] or '∅'}..{rg['last_seen'] or '∅'}  "
           f"window={_cell(rg['window_days'])}d  "
           f"since={_cell(rg['since_days']) + 'd' if rg['since_days'] is not None else 'all'}"]

    out.append(f"retired: {_cell(r.get('retired'))} memories wear a retirement face")
    bad = {n: f["bad_lines"] for n, f in r["files"].items()}
    total_lines = sum(f["lines"] for f in r["files"].values())
    out.append(f"files: {total_lines} lines, bad lines: "
               + (", ".join(f"{n.replace('.jsonl', '')}={b}" for n, b in bad.items())
                  or "0"))

    cr = m["candidate_rate"]
    out.append(f"1 candidate rate        {_frac(cr['candidates'], cr['segments'], cr['rate'])}"
               f"  over {cr['batches']} batches")
    # The spread, not the census: the five sources that fed the most segments, and
    # a count for the rest. On the house store the full list was 60 agent
    # transcripts and buried the table it was meant to explain.
    srcs = sorted(cr["per_source"].items(), key=lambda kv: (-kv[1]["segments"], kv[0]))
    for k, v in srcs[:5]:
        out.append(f"    {k[:48]:<48} {_frac(v['candidates'], v['segments'], v['rate'])}")
    if len(srcs) > 5:
        rest = srcs[5:]
        c = sum(v["candidates"] for _, v in rest); n = sum(v["segments"] for _, v in rest)
        out.append(f"    {'+' + str(len(rest)) + ' more sources':<48} {_frac(c, n, _rate(c, n))}")

    rj = m["rejections"]
    out.append(f"2 rejections            {_cell(rj['dropped'])} dropped")
    for k, v in rj["by_why_dropped"].items():
        out.append(f"    {k[:60]:<60} {v}")
    out.append(f"    reason present       "
               f"{_frac(rj['reason_present'], rj['dropped'], _rate(rj['reason_present'], rj['dropped']))}")
    out.append(f"    unverified_numbers   {rj['unverified_numbers']}")

    us = m["user_evidence_survival"]
    out.append(f"3 USER survival          candidates "
               f"{us['user_candidates']}/{us['user_candidates_n']} by_class  ·  "
               f"kept per class: {us['user_kept']}  (gated_kept {us['gated_kept']})")

    sj = m["self_judgement_seeds"]
    out.append(f"4 seeds / SELF retreat   {_frac(sj['ideas'], sj['gated_kept'], sj['retreat_rate'])}"
               f"  · seeds {sj['seed_entries']} {sj['seed_kinds']}"
               f"  · confirmed {sj['seeds_confirmed']}"
               f"  · tossed {sj['tossed']['total']} {sj['tossed']['by_verdict']}")

    cs = m["callsigns"]
    out.append(f"5 callsigns              receipts {cs['cue_receipts']}"
               f" / failures {cs['cue_receipt_failures']} over {cs['batches_n']} batches"
               f"  · refused on dropped {cs['dropped_with_routing_cues_refused']}"
               f"/{cs['dropped_n']} ({cs['refused_cue_texts']} cues)")

    out.append("6 remembered_but_unreachable (per resident variant)")
    for v, x in m["remembered_but_unreachable"].items():
        out.append(f"    {v[:40]:<40} "
                   f"{_frac(x['remembered_but_unreachable'], x['runnable'], x['rate'])}")

    out.append("7 target reached per resident_sha (sha = the map identity)")
    for sha, x in m["trigger_shortening_by_sha"].items():
        out.append(f"    {sha[:20]:<20} {'/'.join(x['variants'])[:28]:<28} "
                   f"{_frac(x['target_reached'], x['runnable'], x['rate'])}")

    fb = m["fallback_thinker"]
    out.append(f"8 fallback thinker       reads {fb['reads_by_why']} (n={fb['reads_n']})"
               f"  · thinker in {fb['traces_with_thinker_call']}"
               f"/{fb['traces_runnable']} traces")
    out.append(f"    {fb['how_fastpath_vs_meaning']}")

    out.append(f"windows ({_cell(rg['window_days'])}d, oldest first): "
               "window · batches · kept · dropped · candidates")
    for w in r["windows"]:
        wm = w["metrics"]
        out.append(f"    {w['start'][:19]}..{w['end'][:19]}  "
                   f"{wm['candidate_rate']['batches']:>4}  "
                   f"{wm['user_evidence_survival']['gated_kept']:>5}  "
                   f"{wm['rejections']['dropped']:>7}  "
                   f"{wm['candidate_rate']['candidates']:>10}")

    out.extend(r["warnings"])
    return "\n".join(out)
