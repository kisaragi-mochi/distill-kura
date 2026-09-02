"""`kura pay-forward` — pay the map's cold prefill in the quiet hours, once.

The resident block (`prefill.py`) is byte-stable so a prefix cache can HOLD it — but
the first turn after a re-weave still pays the whole cold prefill, and on a slow mouth
that is not a detail. A llama.cpp server started with `--slot-save-path` can persist a
slot's KV to disk and load it back (`POST /slots/<id>?action=save` / `?action=restore`,
body `{"filename": ...}`), and the load is sub-second where the bake was minutes —
measured on one machine (a 320B pure-CPU llama.cpp mouth, 16,444-token map,
2026-09-01): the bake 796 s, the save 283 ms (1.5 GB on NVMe), the restore after
killing and rebooting the server 655 ms, and the first turn after it reprocessed 18
prompt tokens — the map came back warm. So right after a re-weave changes the map,
this module pushes the new map through each registered mouth once and saves the slot:
the 13-minute turn is paid ahead of need, and even a mouth restart wakes up warm.

Named after the film: the cold turn is paid forward, so the next turn — whoever's turn
it is — receives it warm.

The slot filename carries the map's etag (`kura-<store>-<etag prefix>.bin`), which makes
the files content-addressed: a file with the right name holds the right bytes no matter
who wrote it or when. That buys the whole algorithm, per mouth:

    try restore first    — always, fresh etag or changed. On a fresh etag it is the
                           cheap proof the file still exists; on a changed etag it
                           catches a map some earlier runner already baked (a lost
                           state file, a parallel run) and skips the expensive part.
    verify, never assume — one tiny probe with `cache_prompt: true`, `max_tokens: 1`
                           and the map as system. `timings.prompt_n` says how much the
                           server actually reprocessed: a handful of tokens means the
                           slot is warm; the whole map means the restore's 200 was no
                           good — and in THAT case the probe itself just paid the
                           prefill, so the run saves the probe's work instead of
                           paying a second time. A reply with no timings at all
                           proves nothing and is refused — fail closed; llama.cpp
                           always sends them.
    bake                 — the same probe call, knowingly cold, with a very long
                           timeout (the long wait is the whole point), then save,
                           then record.
    unreachable          — a loud, labeled skip. Never a crash, never a state advance.
    one runner per slot  — the whole sequence holds an flock keyed on the slot's
                           physical identity (normalized base url + slot id), in the
                           system temp directory: machine-local, because the runners
                           that can collide — `kura tend` and the systemd restart
                           hook — are machine-local too. A second runner reports
                           busy (`skipped-locked`) instead of racing the first — and
                           busy is NOT fresh: the lock proves another runner exists,
                           not that it is warming YOUR etag, so the run exits as
                           transient (retry), never as all-fresh.

State lives per store in `_still/payforward.json`, keyed by mouth name — workshop
bookkeeping like the tend heartbeat, written atomically and only ever after a confirmed
success, so a crash or a dead mouth leaves the last good record standing.

Old slot files are NOT pruned: the slots API can save and restore a filename but cannot
list the directory, so any pruning here would be a guess about files it cannot see.
Sweep the mouth's `--slot-save-path` directory by hand when it grows — a slot file
scales with map size times the model's KV width (measured: 1.5 GB for the 16,444-token
map on the 320B mouth), so it will.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime

from . import prefill as prefill_mod
from .registry import Registry, mouth_base as _base
from .store import Store
from .thinker import bearer_headers

# 12 of the etag's 16 hex chars: enough to keep two maps' files apart, short enough
# that the filename stays readable in a directory listing.
ETAG_CHARS = 12

# The bake timeout bounds the one call this feature exists to make: the cold prefill
# it pays was measured at 796 s (320B pure-CPU, 16,444-token map), and a bigger map or
# a busier mouth only grows it. An hour is a ceiling, not an expectation.
BAKE_TIMEOUT_S = 3600.0
# A save is a file written and a restore a file read (measured: 283 ms and 655 ms for
# the same 1.5 GB slot file on NVMe). Minutes of margin, not an hour — slower disks
# and bigger maps exist.
SLOT_TIMEOUT_S = 300.0

# "Small means warm": the probe is one character plus the chat template's framing —
# tens of tokens at the very most — while the map is thousands. 128 is a threshold
# chosen to separate those two regimes, not a measurement; for a map so small that 128
# would not separate them, half the map's own size draws the line instead (and being
# wrong there costs a prefill that was trivial anyway).
WARM_PROMPT_N = 128
PROBE = "."


def slot_filename(store_name: str, etag: str) -> str:
    """Filename only — the server prepends its own `--slot-save-path`."""
    return f"kura-{store_name}-{etag[:ETAG_CHARS]}.bin"


def state_path(store: Store) -> str:
    return os.path.join(store.still, "payforward.json")


def _lock_path(base: str, slot: int) -> str:
    """One lock per PHYSICAL slot (normalized base url + slot id), in the system temp
    directory. Machine-local on purpose: the runners that can collide — `kura tend`
    and a systemd hook firing on the mouth's restart — live on this machine, and a
    slot is one server's resource, so a wider lock would guard nothing real."""
    key = hashlib.sha1(f"{base}|{slot}".encode()).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"kura-payforward-{key}.lock")


# ── plumbing ─────────────────────────────────────────────────────────────

def _post(url: str, body: dict, timeout: float, api_key_env: str | None) -> tuple[int, dict]:
    """POST json → (status, parsed reply). An HTTP error status is an ANSWER — a 400
    from a restore means "no such file", which the caller acts on — so it comes back
    as a status, not an exception. Only never-reached-the-server raises (OSError)."""
    headers = {"Content-Type": "application/json", **bearer_headers(api_key_env)}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            try:
                return r.status, json.load(r)
            except ValueError:
                return r.status, {}
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except ValueError:
            return e.code, {}
    except TimeoutError as e:
        raise OSError(f"timed out after {timeout:.0f}s") from e


def _err(code: int, reply: dict) -> str:
    return f"HTTP {code}: {json.dumps(reply, ensure_ascii=False)[:200]}"


def _build_prefill(reg: Registry, store: Store):
    """The store's map and etag, exactly the way the server's GET /prefill route builds
    them — in-process, never over HTTP to ourselves."""
    cfg = reg.prefill_cfg_for(store)
    loom = prefill_mod.loom_for(store, cfg)
    return prefill_mod.build_from_cfg(store, loom, cfg)


def _probe(base: str, mouth: dict, map_text: str,
           timeout: float) -> tuple[int | None, float]:
    """One tiny chat call wearing the map as system. With `cache_prompt` on, a warm
    slot reprocesses only the probe's few tokens and a cold one pays the whole map —
    which is exactly how a bake is paid, so the same call serves both jobs.
    → (timings.prompt_n, wall seconds). prompt_n is None when the reply carries no
    timings (a mouth that is not llama.cpp): warmth is then unverifiable, not false."""
    body = {
        "model": mouth["model"],
        "messages": [{"role": "system", "content": map_text},
                     {"role": "user", "content": PROBE}],
        "max_tokens": 1,
        "temperature": 0,
        "cache_prompt": True,       # llama.cpp: reuse the slot's KV up to the first changed token
        "id_slot": mouth["slot"],   # keep the probe on the slot save/restore acts on
    }
    t0 = time.perf_counter()
    code, reply = _post(f"{base}/v1/chat/completions", body, timeout, mouth.get("api_key_env"))
    if code != 200:
        raise OSError(f"chat probe refused: {_err(code, reply)}")
    pn = (reply.get("timings") or {}).get("prompt_n")
    return (int(pn) if isinstance(pn, (int, float)) and not isinstance(pn, bool) else None,
            round(time.perf_counter() - t0, 2))


# ── state ────────────────────────────────────────────────────────────────

def _read_state(store: Store) -> dict:
    try:
        with open(state_path(store), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_state(store: Store, state: dict) -> None:
    # Atomic like the tend heartbeat: a run killed mid-write must not leave a torn
    # file that reads as "nothing was ever baked" and triggers a pointless re-bake.
    os.makedirs(store.still, exist_ok=True)
    tmp = state_path(store) + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    os.replace(tmp, state_path(store))


def _advance(store: Store, name: str, result: dict) -> None:
    """Record a CONFIRMED success. Never called on a failure path, so an unreachable
    mouth or a failed save leaves the last good record standing.

    Two locks, two resources. The SLOT lock (in `pay_one`) guards one mouth's KV and
    is held for the whole restore/bake — but it cannot guard this file: the ledger is
    one file per STORE, and two mouths of one store hold two DIFFERENT slot locks, so
    `--mouth A` and `--mouth B` running together would read-modify-write over each
    other and the slower writer would erase the faster one's record. So the ledger's
    read-modify-write takes its own lock, held for milliseconds — blocking, because
    the wait is a file read and a file write, never a bake."""
    os.makedirs(store.still, exist_ok=True)
    lock = open(state_path(store) + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = _read_state(store)               # re-read INSIDE the lock: merge-forward
        m = state.setdefault("mouths", {}).setdefault(name, {})
        now = datetime.now().isoformat(timespec="seconds")
        m.update({"etag": result["etag"], "filename": result["file"],
                  "slot": result["slot"], "checked_at": now})
        for k in ("bake_s", "restore_s", "probe_s", "save_s", "prompt_n", "bytes"):
            if result.get(k) is not None:
                m[k] = result[k]
        if result["did"] == "baked":
            m["baked_at"] = now
        elif result["did"] == "restored":
            m["restored_at"] = now
        _write_state(store, state)
    finally:
        lock.close()


# ── the run ──────────────────────────────────────────────────────────────

def pay_one(reg: Registry, mouth: dict, force: bool = False) -> dict:
    """One mouth: restore-first, verify, bake only what restore could not cover.

    → {"mouth", "store", "slot", "etag", "file",
       "did": "baked" | "restored" | "skipped-fresh" | "skipped-locked"
              | "unreachable" | "save-failed" | "unverified",
       wall times, "prompt_n", "bytes", and "error"/"note" where they apply}.
    """
    store = reg.stores[mouth["store"]]
    pf = _build_prefill(reg, store)
    fname = slot_filename(store.name, pf.etag)
    r: dict = {"mouth": mouth["name"], "store": store.name, "slot": mouth["slot"],
               "etag": pf.etag, "file": fname, "map_tokens_est": pf.tokens}
    base = _base(mouth["url"])
    key = mouth.get("api_key_env")
    warm_bar = min(WARM_PROMPT_N, max(8, pf.tokens // 2))

    # One runner per physical slot. Without this, `kura tend` and the systemd
    # restart hook can interleave restore/bake/save on one slot — and the state
    # read-modify-write below must see the winner's record, not race it.
    lock = open(_lock_path(base, mouth["slot"]), "w")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            r["did"] = "skipped-locked"
            r["note"] = ("another pay-forward holds this slot right now — busy, not "
                         "fresh: it may be finishing an older map. Retry later.")
            return r
        return _pay_locked(reg, mouth, force, store, pf, fname, r, base, key, warm_bar)
    finally:
        lock.close()                        # closing the fd releases the flock


def _pay_locked(reg: Registry, mouth: dict, force: bool, store: Store, pf, fname: str,
                r: dict, base: str, key: str | None, warm_bar: int) -> dict:
    # The state is read INSIDE the lock: a concurrent runner may have just advanced
    # it, and acting on a pre-lock snapshot would re-bake what it already baked.
    mstate = (_read_state(store).get("mouths") or {}).get(mouth["name"]) or {}
    fresh = (not force) and mstate.get("etag") == pf.etag

    t0 = time.perf_counter()
    try:
        need_bake = True
        if not force:
            rt0 = time.perf_counter()
            code, reply = _post(f"{base}/slots/{mouth['slot']}?action=restore",
                                {"filename": fname}, SLOT_TIMEOUT_S, key)
            if code == 200:
                r["restore_s"] = round(time.perf_counter() - rt0, 2)
                pn, r["probe_s"] = _probe(base, mouth, pf.text, BAKE_TIMEOUT_S)
                r["prompt_n"] = pn
                if pn is None:
                    # Fail closed: warmth is PROVEN, never assumed, and a reply
                    # without timings.prompt_n proves nothing. llama.cpp always
                    # sends timings, so a mouth that omits them is not one this
                    # feature can vouch for — loud skip, no state advance.
                    r["did"] = "unverified"
                    r["error"] = ("the probe's reply carried no timings.prompt_n, so "
                                  "warmth cannot be verified. llama.cpp always sends "
                                  "timings — is this mouth something else?")
                    r["wall_s"] = round(time.perf_counter() - t0, 2)
                    return r
                if pn <= warm_bar:
                    # Warm. "skipped-fresh" when the state already knew this etag —
                    # the everyday nothing-to-do — and "restored" when the file alone
                    # carried it back (a lost state, a parallel runner's bake).
                    r["did"] = "skipped-fresh" if fresh else "restored"
                    r["wall_s"] = round(time.perf_counter() - t0, 2)
                    _advance(store, mouth["name"], r)
                    return r
                # Cold despite the 200: the probe itself just paid the whole prefill,
                # so the slot is warm NOW. Keep that work — save it, never pay twice.
                need_bake = False
                r["note"] = (f"restore answered 200 but the probe reprocessed {pn} "
                             f"tokens; kept the probe's work and saved it")
            else:
                r["restore_error"] = _err(code, reply)
        if need_bake:
            bt0 = time.perf_counter()
            r["prompt_n"], _ = _probe(base, mouth, pf.text, BAKE_TIMEOUT_S)
            r["bake_s"] = round(time.perf_counter() - bt0, 2)
        st0 = time.perf_counter()
        code, reply = _post(f"{base}/slots/{mouth['slot']}?action=save",
                            {"filename": fname}, SLOT_TIMEOUT_S, key)
        if code != 200:
            # The prefill was paid but nothing reached disk: warm until the next mouth
            # restart, then cold again — and this run will repeat until it is fixed.
            r["did"] = "save-failed"
            r["error"] = (f"the prefill was paid but the slot could not be saved "
                          f"({_err(code, reply)}). Was the mouth started with "
                          f"--slot-save-path? Without it the warmth dies with the "
                          f"server, and every run pays the bake again.")
            r["wall_s"] = round(time.perf_counter() - t0, 2)
            return r                                    # no state advance: nothing on disk to trust
        r["save_s"] = round(time.perf_counter() - st0, 2)
        r["bytes"] = reply.get("n_written")
        r["did"] = "baked"
        r["wall_s"] = round(time.perf_counter() - t0, 2)
        _advance(store, mouth["name"], r)
        return r
    except OSError as e:
        # Unreachable, timed out, or a refused probe: a loud, labeled skip. The state
        # never moves, so the next run starts from the last confirmed truth.
        r["did"] = "unreachable"
        r["error"] = f"{type(e).__name__}: {e}"
        r["wall_s"] = round(time.perf_counter() - t0, 2)
        return r


def run(reg: Registry, store: str | None = None, mouth: str | None = None,
        force: bool = False) -> dict:
    """Every configured mouth (optionally narrowed to one store or one mouth).

    → {"results": [...], "baked", "restored", "fresh", "failed", "worked"}.
    `worked` is what a scheduler counts: bakes and restores, never launches.
    An unknown mouth NAME raises — a typo that silently exits "nothing to do" would
    read exactly like a fleet that is warm."""
    mouths = reg.payforward_mouths
    if store:
        st = reg.store(store)                   # accepts a mode name, like every selector
        mouths = [m for m in mouths if m["store"] == st.name]
    if mouth:
        picked = [m for m in mouths if m["name"] == mouth]
        if not picked:
            raise KeyError(f"unknown mouth: {mouth!r}. known: "
                           f"{[m['name'] for m in mouths] or 'none configured'}")
        mouths = picked
    results = [pay_one(reg, m, force=force) for m in mouths]
    tally = {"baked": 0, "restored": 0, "fresh": 0, "locked": 0, "failed": 0}
    label = {"baked": "baked", "restored": "restored", "skipped-fresh": "fresh",
             "skipped-locked": "locked",
             "unreachable": "failed", "save-failed": "failed", "unverified": "failed"}
    for x in results:
        tally[label[x["did"]]] += 1
    out = {"results": results, **tally, "worked": tally["baked"] + tally["restored"]}
    if not mouths:
        out["note"] = "no [[payforward.mouths]] configured" if not reg.payforward_mouths \
            else "no mouth wears this store's map"
    return out
