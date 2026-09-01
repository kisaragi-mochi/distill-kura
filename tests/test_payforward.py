"""`kura pay-forward` — the map's cold prefill, paid once in the quiet hours.

The fake mouth answers on a real socket and keeps the two pieces of state a real
llama.cpp server keeps: which slot files exist (save / restore) and what the slot is
currently warm on (a probe wearing the text the slot holds reprocesses a handful of
tokens; anything else pays the whole prompt, and afterwards the slot holds THAT).
The tests check the contract, not the model: a bake happens once per etag and is
saved; a fresh etag is proven — restore first, then a cached probe — and reads as
nothing-to-do (exit 2); a lost state file is repaired by the content-addressed
restore instead of a re-bake; a failed restore falls back to the bake; a restore
whose 200 was a lie does not get paid for twice; and an unreachable mouth is a loud,
labeled skip that never advances the state.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura import cli, payforward               # noqa: E402
from distill_kura.registry import Registry             # noqa: E402
from distill_kura.store import Store                   # noqa: E402

WARM_N, COLD_N = 7, 4000


class FakeMouth:
    """The state behind the handler. `files` maps a slot filename to the system text
    that was in the slot when it was saved — which is exactly what restore brings
    back. `log` records every request, in order, for sequence assertions."""

    def __init__(self):
        self.files: dict[str, str] = {}
        self.warm: str | None = None            # what the slot holds right now
        self.has_slot_save_path = True          # False = started without --slot-save-path
        self.send_timings = True                # False = a mouth that is not llama.cpp
        self.log: list[str] = []


def make_handler(m: FakeMouth):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code: int, obj) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path.startswith("/slots/"):
                action = self.path.split("action=", 1)[1]
                fname = body.get("filename", "")
                m.log.append(f"{action}:{fname}")
                if not m.has_slot_save_path:
                    return self._json(400, {"error": {
                        "message": "This server does not support slots action"}})
                if action == "save":
                    m.files[fname] = m.warm or ""
                    return self._json(200, {"id_slot": 0, "filename": fname,
                                            "n_saved": 16444, "n_written": 1566456688,
                                            "timings": {"save_ms": 274.318}})
                if action == "restore":
                    if fname not in m.files:
                        return self._json(400, {"error": {
                            "message": f"slot file not found: {fname}"}})
                    m.warm = m.files[fname]
                    return self._json(200, {"id_slot": 0, "filename": fname,
                                            "n_restored": 16444, "n_read": 1566456688,
                                            "timings": {"restore_ms": 31.391}})
            if self.path == "/v1/chat/completions":
                system = body["messages"][0]["content"]
                cached = ":cached" if body.get("cache_prompt") else ""
                state = "warm" if system == m.warm else "cold"
                m.log.append(f"chat:{state}{cached}")
                prompt_n = WARM_N if system == m.warm else COLD_N
                m.warm = system                 # the probe leaves the slot holding its prompt
                reply = {"choices": [{"message": {"content": "."}}],
                         "timings": {"prompt_n": prompt_n, "prompt_ms": 1.0}}
                if not m.send_timings:
                    del reply["timings"]
                return self._json(200, reply)
            self._json(404, {"error": "unknown route"})
    return H


def start_mouth() -> tuple[ThreadingHTTPServer, FakeMouth, str]:
    m = FakeMouth()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(m))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, m, f"http://127.0.0.1:{srv.server_address[1]}"


def build(tmp_path, url: str, mouth_extra: str = ""):
    st = Store(name="m", path=str(tmp_path / "m"), label="m")
    st.init_files()
    st.remember("ssd-tier", "running a huge model off an SSD tier", "body")
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[stores.m]
path = "{tmp_path / 'm'}"
[models.thinker]
url = "http://127.0.0.1:9/v1"
model = "none"
[[payforward.mouths]]
name = "slow"
url = "{url}"
store = "m"
{mouth_extra}
""", encoding="utf-8")
    reg = Registry.load(str(cfg))
    return reg, reg.store("m"), str(cfg)


def state_of(store: Store) -> dict:
    return json.load(open(payforward.state_path(store), encoding="utf-8"))


# ── the bake ────────────────────────────────────────────────────────────────

def test_a_changed_etag_bakes_saves_and_records(tmp_path):
    srv, m, url = start_mouth()
    try:
        reg, st, cfg = build(tmp_path, url)
        r = payforward.run(reg)
        x = r["results"][0]
        assert x["did"] == "baked" and r["worked"] == 1
        assert x["prompt_n"] == COLD_N                      # the bake paid the map
        assert x["bytes"] == 1566456688                     # file size, from the save
        assert x["bake_s"] is not None and x["wall_s"] is not None
        # the file is content-addressed on the etag, name only — no path
        assert x["file"] == f"kura-m-{x['etag'][:payforward.ETAG_CHARS]}.bin"
        assert "/" not in x["file"]
        assert x["file"] in m.files
        # state written only now, after the confirmed save
        rec = state_of(st)["mouths"]["slow"]
        assert rec["etag"] == x["etag"] and rec["filename"] == x["file"]
        assert rec["baked_at"] and rec["prompt_n"] == COLD_N
    finally:
        srv.shutdown()


def test_cli_exit_codes_zero_for_work_two_for_all_fresh(tmp_path, capsys):
    srv, m, url = start_mouth()
    try:
        build(tmp_path, url)
        assert cli.main(["-c", str(tmp_path / "kura.toml"), "pay-forward"]) == 0
        capsys.readouterr()
        assert cli.main(["-c", str(tmp_path / "kura.toml"), "pay-forward"]) == 2
        out = json.loads(capsys.readouterr().out)
        assert out["fresh"] == 1 and out["worked"] == 0
        assert out["results"][0]["did"] == "skipped-fresh"
    finally:
        srv.shutdown()


# ── the fresh path: restore first, verify always ────────────────────────────

def test_a_fresh_etag_restores_first_and_probes_with_cache_prompt(tmp_path):
    srv, m, url = start_mouth()
    try:
        reg, st, cfg = build(tmp_path, url)
        payforward.run(reg)
        m.log.clear()
        r = payforward.run(reg)
        assert r["results"][0]["did"] == "skipped-fresh"
        assert r["results"][0]["prompt_n"] == WARM_N
        # the sequence IS the contract: restore proves the file, the cached probe
        # proves the warmth, and no save is issued for a slot that did not change
        assert m.log[0].startswith("restore:kura-m-")
        assert m.log[1] == "chat:warm:cached"
        assert len(m.log) == 2
    finally:
        srv.shutdown()


def test_a_lost_state_file_is_repaired_by_restore_not_a_rebake(tmp_path):
    srv, m, url = start_mouth()
    try:
        reg, st, cfg = build(tmp_path, url)
        payforward.run(reg)
        os.remove(payforward.state_path(st))    # the state is gone …
        m.warm = None                           # … and the mouth restarted (slot cold)
        m.log.clear()
        r = payforward.run(reg)
        x = r["results"][0]
        assert x["did"] == "restored" and r["worked"] == 1
        assert not any(e.startswith("chat:cold") for e in m.log)   # no prefill was paid
        assert state_of(st)["mouths"]["slow"]["restored_at"]       # state rebuilt
    finally:
        srv.shutdown()


def test_a_failed_restore_falls_back_to_the_bake(tmp_path):
    srv, m, url = start_mouth()
    try:
        reg, st, cfg = build(tmp_path, url)
        payforward.run(reg)
        m.files.clear()                         # the operator swept --slot-save-path
        m.warm = None                           # and the mouth restarted
        m.log.clear()
        r = payforward.run(reg)
        x = r["results"][0]
        assert x["did"] == "baked"
        assert "restore_error" in x             # the fallback is visible, not silent
        assert m.log[0].startswith("restore:")
        assert "chat:cold:cached" in m.log
        assert any(e.startswith("save:") for e in m.log)
    finally:
        srv.shutdown()


def test_a_restore_that_lies_is_caught_and_the_probes_work_is_saved(tmp_path):
    """restore says 200 but the slot comes back holding something else. The probe pays
    the whole prefill discovering that — and the run must save THAT work rather than
    issuing a second bake call."""
    srv, m, url = start_mouth()
    try:
        reg, st, cfg = build(tmp_path, url)
        r0 = payforward.run(reg)
        fname = r0["results"][0]["file"]
        m.files[fname] = "the wrong bytes"      # a corrupt or mismatched slot file
        m.warm = None
        m.log.clear()
        r = payforward.run(reg)
        x = r["results"][0]
        assert x["did"] == "baked"
        assert "kept the probe's work" in x["note"]
        chats = [e for e in m.log if e.startswith("chat:")]
        assert chats == ["chat:cold:cached"]    # one prefill, never two
        assert any(e.startswith("save:") for e in m.log)
    finally:
        srv.shutdown()


def test_force_rebakes_without_asking_the_etag(tmp_path):
    srv, m, url = start_mouth()
    try:
        reg, st, cfg = build(tmp_path, url)
        payforward.run(reg)
        m.log.clear()
        r = payforward.run(reg, force=True)
        assert r["results"][0]["did"] == "baked"
        assert not any(e.startswith("restore:") for e in m.log)    # straight to the bake
        assert any(e.startswith("save:") for e in m.log)
    finally:
        srv.shutdown()


def test_a_second_runner_on_the_same_slot_skips_cleanly(tmp_path):
    """`kura tend` and the systemd restart hook can fire together. flock conflicts
    across open file descriptions, so a second handle in this very process stands in
    for the second process."""
    import fcntl
    srv, m, url = start_mouth()
    try:
        reg, st, cfg = build(tmp_path, url)
        holder = open(payforward._lock_path(payforward._base(url), 0), "w")
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = payforward.run(reg)
        x = r["results"][0]
        assert x["did"] == "skipped-locked" and r["locked"] == 1
        assert r["worked"] == 0 and r["failed"] == 0           # not work, not a failure
        assert m.log == []                                     # the mouth was never touched
        assert not os.path.exists(payforward.state_path(st))   # and the state never moved
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()
        assert payforward.run(reg)["results"][0]["did"] == "baked"   # released → work proceeds
        assert state_of(st)["mouths"]["slow"]["etag"]          # one consistent record
    finally:
        srv.shutdown()


def test_a_probe_without_timings_is_unverifiable_and_moves_nothing(tmp_path):
    """Warmth is proven, never assumed: a reply with no timings.prompt_n proves
    nothing, so the mouth is refused loudly instead of trusted on the restore's 200.
    llama.cpp always sends timings — nothing real regresses."""
    srv, m, url = start_mouth()
    try:
        reg, st, cfg = build(tmp_path, url)
        payforward.run(reg)                                    # bake; state recorded
        before = state_of(st)
        m.send_timings = False
        r = payforward.run(reg)
        x = r["results"][0]
        assert x["did"] == "unverified" and r["failed"] == 1 and r["worked"] == 0
        assert "timings" in x["error"]                         # loud, and names the gap
        assert state_of(st) == before                          # no state advance
        assert cli.main(["-c", cfg, "pay-forward"]) == 1       # unverifiable is a failure
    finally:
        srv.shutdown()


# ── failure is loud and moves nothing ───────────────────────────────────────

def test_an_unreachable_mouth_is_a_loud_skip_that_never_advances_state(tmp_path):
    reg, st, cfg = build(tmp_path, "http://127.0.0.1:9")   # nothing listens there
    r = payforward.run(reg)
    x = r["results"][0]
    assert x["did"] == "unreachable" and x["error"]
    assert r["failed"] == 1 and r["worked"] == 0
    assert not os.path.exists(payforward.state_path(st))   # no state advance
    assert cli.main(["-c", cfg, "pay-forward"]) == 1       # a failure is not "nothing to do"


def test_a_mouth_without_slot_save_path_is_named_and_state_stays_put(tmp_path):
    srv, m, url = start_mouth()
    try:
        m.has_slot_save_path = False
        reg, st, cfg = build(tmp_path, url)
        r = payforward.run(reg)
        x = r["results"][0]
        assert x["did"] == "save-failed"
        assert "--slot-save-path" in x["error"]            # the fix is in the message
        assert not os.path.exists(payforward.state_path(st))
    finally:
        srv.shutdown()


def test_a_typoed_mouth_name_fails_loudly_instead_of_reading_as_all_warm(tmp_path):
    reg, st, cfg = build(tmp_path, "http://127.0.0.1:9")
    try:
        payforward.run(reg, mouth="slw")
    except KeyError as e:
        assert "slw" in e.args[0] and "slow" in e.args[0]
    else:
        raise AssertionError("an unknown mouth name must raise, not skip")


# ── config validation, at load ──────────────────────────────────────────────

def _load_with_mouths(tmp_path, mouths_toml: str) -> Registry:
    Store(name="m", path=str(tmp_path / "vm")).init_files()
    cfg = tmp_path / "v.toml"
    cfg.write_text(f"""
[stores.m]
path = "{tmp_path / 'vm'}"
[models.thinker]
url = "http://127.0.0.1:9/v1"
model = "none"
{mouths_toml}
""", encoding="utf-8")
    return Registry.load(str(cfg))


def test_a_bad_mouth_is_a_load_error_with_the_offender_named(tmp_path):
    cases = [
        ('[[payforward.mouths]]\nname = "a"\nurl = "http://x"\n',
         "needs `store`"),
        ('[[payforward.mouths]]\nname = "a"\nurl = "http://x"\nstore = "nope"\n',
         "not a configured store"),
        ('[[payforward.mouths]]\nname = "a"\nurl = "http://x"\nstore = "m"\nslott = 1\n',
         "unknown key"),
        ('[[payforward.mouths]]\nname = "a"\nurl = "http://x"\nstore = "m"\nslot = "0"\n',
         "slot must be int"),
        ('[[payforward.mouths]]\nname = "a"\nurl = "http://x"\nstore = "m"\n'
         '[[payforward.mouths]]\nname = "a"\nurl = "http://y"\nstore = "m"\n',
         "two mouths named"),
        ('[payforward]\nmouthes = []\n',
         "unknown key"),
    ]
    for toml, expect in cases:
        try:
            _load_with_mouths(tmp_path, toml)
        except ValueError as e:
            assert expect in str(e), f"{expect!r} not in {e}"
        else:
            raise AssertionError(f"loaded silently: {toml!r}")


def test_two_names_for_one_physical_slot_are_refused_at_load(tmp_path):
    """The name keys the state, but (normalized base url, slot) is the physical
    identity: two entries on one slot would race on its KV."""
    same = ('[[payforward.mouths]]\nname = "a"\nurl = "http://127.0.0.1:8014"\nstore = "m"\n'
            '[[payforward.mouths]]\nname = "b"\nurl = "http://127.0.0.1:8014/v1/"\nstore = "m"\n')
    try:
        _load_with_mouths(tmp_path, same)                      # /v1/ variant: same base
    except ValueError as e:
        assert "same physical slot" in str(e) and "'a'" in str(e)
    else:
        raise AssertionError("two names for one url+slot must be refused")
    ok = ('[[payforward.mouths]]\nname = "a"\nurl = "http://127.0.0.1:8014"\nstore = "m"\n'
          '[[payforward.mouths]]\nname = "b"\nurl = "http://127.0.0.1:8014"\nstore = "m"\nslot = 1\n')
    assert len(_load_with_mouths(tmp_path, ok).payforward_mouths) == 2   # another slot is fine


def test_valid_mouths_get_defaults_and_show_in_describe(tmp_path):
    reg = _load_with_mouths(tmp_path, '[[payforward.mouths]]\nname = "a"\n'
                                      'url = "http://127.0.0.1:9/v1/"\nstore = "m"\n')
    mo = reg.payforward_mouths[0]
    assert mo["slot"] == 0 and mo["model"] == "default"
    assert reg.describe()["payforward"]["mouths"][0]["name"] == "a"
    # a /v1 suffix is the certain slip (every other url in the file carries one);
    # the slots API lives beside /v1, so it is stripped rather than punished
    assert payforward._base(mo["url"]) == "http://127.0.0.1:9"
