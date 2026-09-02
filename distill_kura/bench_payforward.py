"""`kura bench payforward` — measure what the Stable Spine + Hot Trail actually buys.

`kura pay-forward` claims the map comes back warm from a slot file; this command makes
the mouth price its own claims, one `[[payforward.mouths]]` entry at a time. Every
condition sends one chat call and reads `timings.prompt_n` back — the number of prompt
tokens the server says it reprocessed, which is the only witness that counts (a restore
answering 200 proves a file was read, not that the KV came back). The conditions, in
order:

    cold-full                the whole resident block, cache_prompt off — what a turn
                             pays with no spine at all (the measured 796 s case on the
                             320B CPU mouth). Skippable with --skip-cold.
    bake-spine               only when no slot file exists for the current etag: the
                             map-only spine is baked exactly as pay-forward does
                             (probe with cache_prompt on, then action=save, then the
                             ledger advance through `_advance`) and shown as its own row.
    restore-spine+trail      the slot file is restored, then the full map+trail block is
                             sent — prompt_n should be ≈ trail + probe tokens.
    trail-changed            one synthetic line appended to the trail (NEVER written to
                             disk) — prompt_n should be ≈ the changed tail only.
    map-changed              spine restored again, one character changed in the LAST
                             index line — prompt_n ≈ from the change to the end.
    map-changed-first-line   same, but the change is in the FIRST index line —
                             prompt_n ≈ the whole map. This is the proof that a volatile
                             header (a date, a clock) re-prices everything behind it.
    warm-repeat              the exact restore-spine+trail request again — the mouth
                             serves it from its in-process prefix cache for the price of
                             the trail, no disk involved.

`warm-repeat` re-arms the slot from the spine file first (the pay-forward restore,
sub-second). It has to: the changed-text rows before it leave their MODIFIED texts in
the slot's KV, and a request measured against those would re-price the whole map and
report a cold payment as "warm". The re-arm is part of the row and named in its note.
After the last row the current-etag file is restored once more, so the mouth is left
exactly as pay-forward keeps it.

Nothing is ever written to the store, and no modified map or trail reaches disk — the
variants exist only inside the request bodies. Exit 0 always, except when the mouth is
unreachable (or another pay-forward runner holds the slot): exit 1 with the reason.
"""
from __future__ import annotations

import fcntl
import time

from . import payforward
from . import prefill as prefill_mod
from . import trail as trail_mod
from .registry import Registry, mouth_base

# Inserted before the trail's END marker for the trail-changed row. One line, clearly
# synthetic, never persisted — it exists only to move the tail of the request.
SYNTH_LINE = "- (bench-synthetic.md) appended by `kura bench payforward`; never written to disk"


def _chat(base: str, mouth: dict, system_text: str, cache_prompt: bool,
          timeout: float) -> tuple[int | None, float | None, float]:
    """One chat call wearing `system_text`, → (timings.prompt_n, timings.prompt_ms,
    wall seconds). prompt_n is None when the reply carries no timings — reported as
    such, never read as zero."""
    body = {
        "model": mouth["model"],
        "messages": [{"role": "system", "content": system_text},
                     {"role": "user", "content": payforward.PROBE}],
        "max_tokens": 1,
        "temperature": 0,
        "cache_prompt": cache_prompt,
        "id_slot": mouth["slot"],   # always pinned: save/restore act on this slot, and a
                                    # multi-slot mouth must not answer on a different one
    }
    t0 = time.perf_counter()
    code, reply = payforward._post(f"{base}/v1/chat/completions", body, timeout,
                                   mouth.get("api_key_env"))
    wall = round(time.perf_counter() - t0, 2)
    if code != 200:
        raise OSError(f"chat refused: {payforward._err(code, reply)}")
    t = reply.get("timings") or {}

    def _num(v):
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    return _num(t.get("prompt_n")), _num(t.get("prompt_ms")), wall


def _row(condition: str, prompt_n, prompt_ms, wall_s, map_t, trail_t, note: str) -> dict:
    return {"condition": condition, "prompt_n": prompt_n, "prompt_ms": prompt_ms,
            "wall_s": wall_s, "map_tokens_est": map_t, "trail_tokens_est": trail_t,
            "note": note}


def _flip_one_char(line: str) -> str:
    """One character changed, same length — so the divergence the mouth reports is
    the change itself, not a shift."""
    for i, ch in enumerate(line):
        if ch.isalnum():
            return line[:i] + ("X" if ch != "X" else "Y") + line[i + 1:]
    return line + "x"


def _map_with_line_changed(map_text: str, last: bool) -> str | None:
    """The map with one character changed in its last (or first) index line — the
    lines `(...​.md)` names — or None when the map shows no index line at all."""
    lines = map_text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ".md)" in ln]
    if not idx:
        return None
    i = idx[-1] if last else idx[0]
    lines[i] = _flip_one_char(lines[i])
    return "\n".join(lines)


def _restore(base: str, mouth: dict, fname: str, key) -> bool:
    code, _ = payforward._post(f"{base}/slots/{mouth['slot']}?action=restore",
                               {"filename": fname}, payforward.SLOT_TIMEOUT_S, key)
    return code == 200


def run(reg: Registry, mouth: str, skip_cold: bool = False) -> dict:
    """Measure one mouth by its [[payforward.mouths]] name. An unknown name raises —
    a typo that silently measured nothing would read exactly like a warm fleet."""
    picked = [m for m in reg.payforward_mouths if m["name"] == mouth]
    if not picked:
        raise KeyError(f"unknown mouth: {mouth!r}. known: "
                       f"{[m['name'] for m in reg.payforward_mouths] or 'none configured'}")
    return measure(reg, picked[0], skip_cold=skip_cold)


def measure(reg: Registry, mouth: dict, skip_cold: bool = False) -> dict:
    store = reg.stores[mouth["store"]]
    # The map alone — the etag the slot file is keyed on. The trail rides AFTER the
    # map (prefill.build), so it never enters the filename.
    pf = payforward._build_prefill(reg, store)
    cfg = reg.prefill_cfg_for(store)
    loom = prefill_mod.loom_for(store, cfg)
    trail = prefill_mod.trail_for(store, cfg, loom=loom)
    full = prefill_mod.build(store, loom, header=cfg.get("header"),
                             window_tokens=int(cfg.get("window_tokens", 131072)),
                             fraction=float(cfg.get("budget_fraction", 0.05)),
                             hard_fraction=float(cfg.get("hard_fraction", 0.20)),
                             trail=trail)

    raw = full.stats.get("trail", "")
    trail_state = ("appended" if raw.startswith("appended")
                   else "stale" if raw.startswith("stale") else "absent")
    # The appended block is the exact suffix of the served text: split there rather
    # than re-deriving the trail, so the variants below differ from production by the
    # intended characters and nothing else.
    if trail_state == "appended" and full.text.startswith(pf.text):
        map_text, trail_part = pf.text, full.text[len(pf.text):]
    else:
        map_text, trail_part = full.text, ""
    map_t = pf.tokens
    trail_t = full.stats.get("trail_tokens") if trail_state == "appended" else None

    base = mouth_base(mouth["url"])
    key = mouth.get("api_key_env")
    fname = payforward.slot_filename(store.name, pf.etag)
    rows: list[dict] = []
    out: dict = {"mouth": mouth["name"], "store": store.name, "etag": pf.etag,
                 "map_tokens_est": map_t, "trail_tokens_est": trail_t,
                 "trail": trail_state, "rows": rows}

    # One runner per physical slot, the same lock pay-forward holds: a measurement
    # interleaved with a real bake would price the other runner's requests.
    lock = open(payforward._lock_path(base, mouth["slot"]), "w")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise OSError("another pay-forward runner holds this slot right now "
                          "— busy is not a measurement; retry later")

        if not skip_cold:
            pn, ms, wall = _chat(base, mouth, full.text, False, payforward.BAKE_TIMEOUT_S)
            rows.append(_row("cold-full", pn, ms, wall, map_t, trail_t,
                             "cache_prompt off — the whole resident block is paid"))

        if not _restore(base, mouth, fname, key):
            # Absent file: bake the spine exactly as pay-forward does — probe with
            # cache_prompt on, then save, then the ledger advance (allowed to write
            # _still/payforward.json because a real bake happened).
            t0 = time.perf_counter()
            pn, _ = payforward._probe(base, mouth, pf.text, payforward.BAKE_TIMEOUT_S)
            bake_s = round(time.perf_counter() - t0, 2)
            t0 = time.perf_counter()
            code, reply = payforward._post(f"{base}/slots/{mouth['slot']}?action=save",
                                           {"filename": fname},
                                           payforward.SLOT_TIMEOUT_S, key)
            save_s = round(time.perf_counter() - t0, 2)
            rows.append(_row("bake-spine", pn, None, round(bake_s + save_s, 2),
                             map_t, None,
                             "no slot file for the current etag — baked the map-only "
                             "spine as pay-forward does: probe, then save"))
            if code != 200:
                # The spine never reached disk: every restored row would silently
                # re-price against whatever the slot holds, so stop rather than
                # mislabel — and the ledger does not advance, the same rule the
                # pay-forward bake obeys on a failed save.
                out["warning"] = (f"the bake's save failed (HTTP {code}); was the mouth "
                                  f"started with --slot-save-path? No spine to measure "
                                  f"against — rows after the bake were skipped")
                return out
            payforward._advance(store, mouth["name"], {
                "did": "baked", "etag": pf.etag, "file": fname, "slot": mouth["slot"],
                "prompt_n": pn, "bake_s": bake_s, "save_s": save_s,
                "bytes": reply.get("n_written")})

        pn, ms, wall = _chat(base, mouth, full.text, True, payforward.BAKE_TIMEOUT_S)
        rows.append(_row("restore-spine+trail", pn, ms, wall, map_t, trail_t,
                         "slot restored from the current-etag file — only the trail "
                         "and the probe are reprocessed"))

        if trail_part:
            changed = trail_part.replace(trail_mod.TRAIL_END,
                                         SYNTH_LINE + "\n" + trail_mod.TRAIL_END, 1)
            if changed == trail_part:                   # frame absent: plain append
                changed = trail_part + SYNTH_LINE + "\n"
            pn, ms, wall = _chat(base, mouth, map_text + changed, True,
                                 payforward.BAKE_TIMEOUT_S)
            rows.append(_row("trail-changed", pn, ms, wall, map_t, trail_t,
                             "one synthetic line appended to the trail (never written "
                             "to disk) — only the changed tail is reprocessed"))
        else:
            rows.append(_row("trail-changed", None, None, None, map_t, trail_t,
                             f"no trail appended ({trail_state}) — nothing to change"))

        for cond, note in (
            ("map-changed",
             "one character changed in the LAST index line (never written to disk) "
             "— reprocessed from the change to the end"),
            ("map-changed-first-line",
             "one character changed in the FIRST index line — a change at the front "
             "re-prices nearly the whole map (the volatile-header proof)"),
        ):
            variant = _map_with_line_changed(map_text, last=(cond == "map-changed"))
            if variant is None:
                rows.append(_row(cond, None, None, None, map_t, trail_t,
                                 "no index line in the map to change"))
            elif not _restore(base, mouth, fname, key):
                rows.append(_row(cond, None, None, None, map_t, trail_t,
                                 "restore failed — the spine file disappeared mid-run"))
            else:
                pn, ms, wall = _chat(base, mouth, variant + trail_part, True,
                                     payforward.BAKE_TIMEOUT_S)
                rows.append(_row(cond, pn, ms, wall, map_t, trail_t, note))

        # Re-arm from the spine file: the rows above churned the KV with modified
        # texts, and a repeat measured against those would re-price the whole map —
        # the restore is the pay-forward primitive, sub-second where a bake is minutes.
        warm = _restore(base, mouth, fname, key)
        pn, ms, wall = _chat(base, mouth, full.text, True, payforward.BAKE_TIMEOUT_S)
        rows.append(_row("warm-repeat", pn, ms, wall, map_t, trail_t,
                         "the restore-spine+trail request again, no bake — the map "
                         "comes from KV, only the trail and the probe are paid"))
        if not warm:
            out["warning"] = "the slot file vanished mid-run; warm-repeat ran against a cold slot"

        # Leave the mouth exactly as pay-forward keeps it: warm on the current-etag spine.
        if not _restore(base, mouth, fname, key):
            out["final_restore_error"] = (f"could not restore {fname} at the end; "
                                          "the mouth is not left on the spine")
        return out
    finally:
        lock.close()                                # closing the fd releases the flock
