"""`kura bench payforward` — six conditions, priced by a fake mouth that counts like
llama.cpp does.

The fake keeps, per slot, the prompt text its KV currently holds. A chat with
`cache_prompt` on reprocesses only what lies beyond the longest common prefix with that
text (everything when the flag is off) and then leaves the FULL new text in the slot —
the two rules a real mouth follows. Save/restore copy that text to/from a dict by
filename, so a restored slot is warm exactly as far as the file's bytes. `estimate()`
from tokens.py is the tokenizer, so the numbers compare with the bench's own
`map_tokens_est` / `trail_tokens_est`.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura import bench_payforward, cli, payforward       # noqa: E402
from distill_kura import prefill as prefill_mod                  # noqa: E402
from distill_kura.registry import Registry                       # noqa: E402
from distill_kura.store import Store                             # noqa: E402
from distill_kura.tokens import estimate                         # noqa: E402


class FakeMouth:
    """`kv`: what each slot holds right now; `files`: what save put on disk. The
    request text is the messages rendered deterministically — the same string a real
    server would run through its chat template."""

    def __init__(self):
        self.kv: dict[int, str] = {}
        self.files: dict[str, str] = {}
        self.log: list[str] = []


def _render(body: dict) -> str:
    return "\x00".join(f"{m['role']}:{m['content']}" for m in body["messages"])


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
                head, _, action = self.path.partition("?action=")
                slot = int(head.rsplit("/", 1)[1])
                fname = body.get("filename", "")
                m.log.append(f"{action}:{fname}")
                if action == "save":
                    m.files[fname] = m.kv.get(slot, "")
                    return self._json(200, {"n_written": len(m.files[fname])})
                if action == "restore":
                    if fname not in m.files:
                        return self._json(400, {"error": {"message": f"no file {fname}"}})
                    m.kv[slot] = m.files[fname]
                    return self._json(200, {})
                return self._json(400, {"error": {"message": "unknown action"}})
            if self.path == "/v1/chat/completions":
                slot = int(body.get("id_slot") or 0)
                text = _render(body)
                if body.get("cache_prompt"):
                    shared = len(os.path.commonprefix([text, m.kv.get(slot, "")]))
                    prompt_n = estimate(text[shared:])
                else:
                    prompt_n = estimate(text)
                m.kv[slot] = text               # the request leaves its whole prompt in the slot
                m.log.append(f"chat:{prompt_n}")
                return self._json(200, {"choices": [{"message": {"content": "."}}],
                                        "timings": {"prompt_n": prompt_n,
                                                    "prompt_ms": prompt_n * 0.5}})
            return self._json(404, {"error": "unknown route"})
    return H


def start_mouth() -> tuple[ThreadingHTTPServer, FakeMouth, str]:
    m = FakeMouth()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(m))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, m, f"http://127.0.0.1:{srv.server_address[1]}"


def build(tmp_path, url: str):
    """A store with several index lines (so first-line vs last-line differ) and a
    CURRENT trail on disk, built through the same helpers the bench uses, so the
    freshness stamps the bench verifies are the real ones."""
    st = Store(name="m", path=str(tmp_path / "m"), label="m")
    st.init_files()
    # A date in the body, because five remembers in one sitting make today a bulk-touch
    # day and mtime age reads as unknown — the trail's fresh layer would come out empty.
    today = datetime.now(timezone.utc).date().isoformat()
    for i in range(12):                     # enough index lines that the map, not the
        st.remember(f"mem-{i:02d}-note",    # ~200-token trail, dominates the block
                    f"bench fixture memory number {i} of this store",
                    f"observed {today}")
    loom = prefill_mod.loom_for(st, {})
    prefill_mod.trail_for(st, {}, loom=loom).write()
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
""", encoding="utf-8")
    return Registry.load(str(cfg)), st, str(cfg)


def snapshot(st: Store) -> dict:
    """Every byte under the store path — the bench may add payforward bookkeeping to
    _still (a real bake happened) and nothing else anywhere."""
    out = {}
    for root, _dirs, files in os.walk(st.path):
        for f in files:
            p = os.path.join(root, f)
            out[os.path.relpath(p, st.path)] = open(p, "rb").read()
    return out


ROW_KEYS = {"condition", "prompt_n", "prompt_ms", "wall_s",
            "map_tokens_est", "trail_tokens_est", "note"}


def by_condition(r: dict) -> dict:
    assert all(set(x) == ROW_KEYS for x in r["rows"])
    return {x["condition"]: x for x in r["rows"]}


def test_the_six_conditions_are_priced_by_the_mouth(tmp_path):
    srv, m, url = start_mouth()
    try:
        reg, st, _cfg = build(tmp_path, url)
        before = snapshot(st)
        r = bench_payforward.run(reg, mouth="slow")
        assert [x["condition"] for x in r["rows"]] == [
            "cold-full", "bake-spine", "restore-spine+trail", "trail-changed",
            "map-changed", "map-changed-first-line", "warm-repeat"]
        assert r["trail"] == "appended"
        assert r["trail_tokens_est"] and r["map_tokens_est"]
        b = by_condition(r)
        # the restored spine pays the trail, not the map
        assert 0 < b["restore-spine+trail"]["prompt_n"] < b["cold-full"]["prompt_n"]
        # a changed trail reprocesses only its tail
        assert 0 < b["trail-changed"]["prompt_n"] < b["restore-spine+trail"]["prompt_n"]
        # position matters: a change in the FIRST line re-prices ~the whole map,
        # the same change in the LAST line only the end
        assert b["map-changed"]["prompt_n"] < b["map-changed-first-line"]["prompt_n"]
        assert b["map-changed-first-line"]["prompt_n"] > 0.75 * b["cold-full"]["prompt_n"]
        # the repeated request is served from the prefix cache: no more than row 2 paid
        assert b["warm-repeat"]["prompt_n"] <= b["restore-spine+trail"]["prompt_n"]
        assert b["warm-repeat"]["prompt_n"] < b["map-changed-first-line"]["prompt_n"]
        # a real bake happened, so the ledger may have moved — inside _still, nowhere else
        after = snapshot(st)
        diff = {k for k in set(before) ^ set(after)}
        changed = {k for k in set(before) & set(after) if before[k] != after[k]}
        assert not changed
        assert diff <= {"_still/payforward.json", "_still/payforward.json.lock"}
        # the trail on disk never learned the synthetic line, the index never the flip
        assert "bench-synthetic" not in (st.index_text() + open(
            os.path.join(st.still, "trailhead.md"), encoding="utf-8").read())
    finally:
        srv.shutdown()


def test_a_second_run_skips_cold_and_bake_and_skip_cold_omits_row_one(tmp_path):
    srv, m, url = start_mouth()
    try:
        reg, st, _cfg = build(tmp_path, url)
        bench_payforward.run(reg, mouth="slow")
        m.log.clear()                           # the second run must re-bake nothing
        r = bench_payforward.run(reg, mouth="slow", skip_cold=True)
        conds = [x["condition"] for x in r["rows"]]
        assert "cold-full" not in conds and "bake-spine" not in conds
        assert conds[0] == "restore-spine+trail" and conds[-1] == "warm-repeat"
        assert any(e.startswith("restore:") for e in m.log)   # warmth came from the file
        assert not any(e.startswith("save:") for e in m.log)  # so nothing was saved again
    finally:
        srv.shutdown()


def test_cli_json_flag_and_unknown_mouth_exit(tmp_path, capsys):
    srv, m, url = start_mouth()
    try:
        reg, st, cfg = build(tmp_path, url)
        rc = cli.main(["-c", cfg, "bench", "payforward", "--mouth", "slow",
                       "--skip-cold", "--json"])
        assert rc == 0
        r = json.loads(capsys.readouterr().out)
        assert r["mouth"] == "slow" and r["store"] == "m" and r["rows"]
        assert "cold-full" not in [x["condition"] for x in r["rows"]]
        try:
            cli.main(["-c", cfg, "bench", "payforward", "--mouth", "nope"])
            raised = False
        except SystemExit as e:
            raised, code = True, e.code
        assert raised and isinstance(code, str) and "nope" in code
    finally:
        srv.shutdown()
