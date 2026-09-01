"""The watcher: quiet is a journal that has not changed; nothing-to-do is exit 2 and a
rest; the human's return stops the track (unless the editor sits elsewhere); a
heartbeat says whether anyone is tending the store at all.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from distill_kura.registry import Registry     # noqa: E402
from distill_kura.store import Store           # noqa: E402
from distill_kura.tend import Tender           # noqa: E402


def build(tmp_path, **distill):
    Store(name="m", path=str(tmp_path / "m")).init_files()
    jdir = tmp_path / "journals"; jdir.mkdir()
    (jdir / "s.jsonl").write_text(json.dumps({"type": "user", "message": {"content": [
        {"type": "text", "text": "hello " * 3000}]}}) + "\n", encoding="utf-8")
    extra = "".join(f"{k} = {json.dumps(v)}\n" for k, v in distill.items())
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[stores.m]
path = "{tmp_path / 'm'}"
[models.thinker]
url = "http://127.0.0.1:9/v1"
model = "none"
[distill]
{extra}
[distill.journals]
claude = "{jdir}"
""", encoding="utf-8")
    reg = Registry.load(str(cfg))
    return reg, reg.store("m"), str(cfg), jdir / "s.jsonl"


def test_quiet_is_the_journals_mtime_and_a_fresh_journal_is_not_quiet(tmp_path):
    reg, st, cfg, j = build(tmp_path)
    t = Tender(reg, st, cfg, idle_min=10)
    stamp = t.tick(0.0)
    assert stamp == os.path.getmtime(j)
    assert t.proc is None                                   # just written: not quiet
    assert st.tend_state()["alive"]                         # but the heartbeat is there


def test_nothing_to_do_is_exit_two_and_a_rest_not_a_spin(tmp_path):
    reg, st, cfg, j = build(tmp_path, backoff_min=20)
    old = time.time() - 3600
    os.utime(j, (old, old))                                 # quiet for an hour
    t = Tender(reg, st, cfg, idle_min=10)
    t.tick(0.0)
    assert t.proc_track == "distill"                        # no drafts → distil
    t.proc.wait(timeout=120)
    t.reap()
    # a dead thinker finds no candidates → "nothing worth drinking" → rc 2 → rest
    assert t.next_ok["distill"] > time.time() + 19 * 60
    log = open(os.path.join(st.still, "tend.log"), encoding="utf-8").read()
    assert "nothing to do — resting 20 min" in log
    assert "distill run" not in log or "→ distill" in log   # decisions kept, never /dev/null
    # the next tick does not relaunch the resting track; it moves on to tidy, once
    t.tick(os.path.getmtime(j))
    assert t.proc_track == "tidy"
    t.proc.wait(timeout=120); t.reap()
    t.tick(os.path.getmtime(j))
    assert t.proc is None                                   # everything is resting or done


def test_the_humans_return_stops_a_running_track_unless_told_not_to(tmp_path):
    reg, st, cfg, j = build(tmp_path)
    old = time.time() - 3600
    os.utime(j, (old, old))
    t = Tender(reg, st, cfg, idle_min=10)
    t._cmd = lambda track: [sys.executable, "-c", "import time; time.sleep(60)"]   # type: ignore
    stamp = t.tick(0.0)
    assert t.proc is not None
    with open(j, "a", encoding="utf-8") as f:
        f.write("\n")                                       # the human types
    t.tick(stamp)
    assert t.proc is None
    assert "stopped — the journal changed" in open(os.path.join(st.still, "tend.log"), encoding="utf-8").read()
    # with the editor on its own seat, the verdict in flight is left to finish
    os.utime(j, (old, old))
    t2 = Tender(reg, st, cfg, idle_min=10, yield_on_return=False)
    t2._cmd = t._cmd                                        # type: ignore
    stamp = t2.tick(0.0)
    with open(j, "a", encoding="utf-8") as f:
        f.write("\n")
    t2.tick(stamp)
    assert t2.proc is not None and t2.proc.poll() is None
    t2.kill("test over")


def test_work_is_counted_and_launches_are_not(tmp_path):
    reg, st, cfg, j = build(tmp_path)
    t = Tender(reg, st, cfg, idle_min=10)
    assert set(t.done) == {"poured", "tossed", "fixed", "drafts", "woven", "paid", "tidied"}
    assert not any(k.endswith("_runs") or "launch" in k for k in t.done)


def test_payforward_is_scheduled_after_a_weave_and_only_then(tmp_path):
    """The map cannot have changed without a weave, so the payforward track waits for
    one — and runs once per weave, not once per tick (a mouth restart is the systemd
    hook's job, not the watcher's)."""
    reg, st, cfg, j = build(tmp_path)
    t = Tender(reg, st, cfg, idle_min=10)
    now = time.time()
    for track in ("drain", "distill", "tidy"):
        t.next_ok[track] = now + 9999           # only the question at hand remains
    assert t.choose(now) is None                # no weave yet → no payforward
    t._woven_this_silence = True
    assert t.choose(now) == "payforward"
    t._paid_this_silence = True
    assert t.choose(now) is None                # once per weave, not once per tick


def test_doctor_reports_a_dead_watcher(tmp_path):
    reg, st, cfg, j = build(tmp_path)
    assert st.doctor()["tending"] == {"alive": False, "why": "no watcher has ever run here"}
    t = Tender(reg, st, cfg, idle_min=10)
    t.tick(0.0)
    assert st.doctor()["tending"]["alive"]
    p = os.path.join(st.still, "tend.json")
    d = json.load(open(p)); d["at"] = int(time.time()) - 600
    json.dump(d, open(p, "w"))
    assert not st.doctor()["tending"]["alive"]
    assert "heartbeat" in st.doctor()["tending"]["why"]


def test_cli_once_runs_a_tick_and_exits(tmp_path):
    reg, st, cfg, j = build(tmp_path)
    old = time.time() - 3600
    os.utime(j, (old, old))
    e = {**os.environ, "PYTHONPATH": ROOT}
    p = subprocess.run([sys.executable, "-m", "distill_kura.cli", "-c", cfg, "-s", "m", "tend", "--once"],
                       capture_output=True, text=True, env=e, timeout=300)
    # The dead thinker leaves no drafts, so the tick did no work — that is exit 2
    # ("nothing to do"), which the always-0 return used to hide from schedulers.
    assert p.returncode == 2, p.stderr
    last = [l for l in p.stdout.splitlines() if l.startswith("{")][-1]
    out = json.loads(last)
    assert out["store"] == "m" and "done" in out
    assert not any(out["done"].values())            # nothing was poured or fixed
    assert os.path.exists(os.path.join(st.still, "tend.log"))


def test_a_store_with_no_journals_says_so_instead_of_waiting_forever(tmp_path):
    Store(name="a", path=str(tmp_path / "a")).init_files()
    Store(name="b", path=str(tmp_path / "b")).init_files()
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[stores.a]
path = "{tmp_path / 'a'}"
[stores.b]
path = "{tmp_path / 'b'}"
[models.thinker]
url = "http://127.0.0.1:9/v1"
model = "none"
""", encoding="utf-8")
    reg = Registry.load(str(cfg))
    t = Tender(reg, reg.store("a"), str(cfg))
    assert t.journals == {}
    assert t.newest_mtime() == 0.0
    t.tick(0.0)
    assert t.proc is None


def test_catchup_starts_from_today_without_losing_further_marks(tmp_path):
    """Pointing a distiller at an old journal would drink all of it. catchup moves the
    marks to the end of every journal — forward only, so a mark already past stays."""
    from distill_kura.distill import Distiller
    reg, st, cfg, j = build(tmp_path)
    d = Distiller(reg, st)
    r = d.catch_up()
    assert r["ok"] and r["journals"] == 1 and r["moved"] == 1
    assert d.sip_one() is None                      # nothing left to drink
    marks = d.marks.read()
    key = "claude:" + os.path.basename(str(j))
    assert marks[key] == os.path.getsize(j)
    d.marks.advance(key, marks[key] + 10_000)       # someone is further along
    d.catch_up()
    assert d.marks.read()[key] == marks[key] + 10_000   # never pulled backwards
