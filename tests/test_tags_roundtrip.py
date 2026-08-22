"""Tags and annotations survive every surface: HTTP, the MCP bridge, the CLI.

A tag that can be written but not read back, or read on one surface and dropped on
another, is a tag nobody can rely on. So each surface is driven for real — the HTTP
server in a thread, the bridge as a subprocess talking to that server, the CLI as a
subprocess — and the same memory is inspected on the way out.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from distill_kura.registry import Registry       # noqa: E402
from distill_kura.server import _make_handler    # noqa: E402
from distill_kura.store import Store             # noqa: E402


def registry(tmp_path, policy="direct-allowed") -> tuple[Registry, str]:
    Store(name="main", path=str(tmp_path / "main")).init_files()
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[stores.main]
path = "{tmp_path / 'main'}"
write_policy = "{policy}"
[models.thinker]
url = "http://127.0.0.1:9/v1"
model = "none"
""", encoding="utf-8")
    return Registry.load(str(cfg)), str(cfg)


def serve(reg) -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(reg))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def post(base, path, obj):
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return json.load(r)


def test_http_round_trip_of_tags_and_annotations(tmp_path):
    reg, _ = registry(tmp_path)
    srv, base = serve(reg)
    try:
        code, r = post(base, "/remember", {
            "slug": "m", "description": "trigger", "body": "the body",
            "tags": ["landmine", "decision", "landmine"],
            "belongs_because": "because this store cuts outages",
            "annotations": {"keep": "the order"}})
        assert code == 200 and r["ok"], r
        d = get(base, "/memory/m")
        assert d["tags"] == ["decision", "landmine"]
        assert d["annotations"] == {"belongs_because": "because this store cuts outages",
                                    "keep": "the order"}
        # /annotate merges; the same merge twice changes nothing
        code, r = post(base, "/annotate", {"slug": "m", "tags": ["recurred"], "may_fade": "chatter"})
        assert code == 200 and r["changed"]
        code, r = post(base, "/annotate", {"slug": "m", "tags": ["recurred"], "may_fade": "chatter"})
        assert code == 200 and not r["changed"]
        d = get(base, "/memory/m")
        assert d["tags"] == ["decision", "landmine", "recurred"]
        assert d["annotations"]["may_fade"] == "chatter"
        # and the index is unchanged by annotation
        assert get(base, "/index")["index"].count("(m.md)") == 1
        # a bad tag is a refusal with the tag named, not a silent drop
        code, r = post(base, "/annotate", {"slug": "m", "tags": ["Not Kebab"]})
        assert code == 403 and "Not Kebab" in r["error"]
        assert get(base, "/memory/m")["tags"] == ["decision", "landmine", "recurred"]
    finally:
        srv.shutdown()


def test_http_annotate_obeys_distiller_only(tmp_path):
    """The direct door is the direct door, whether it carries a body or a tag."""
    reg, _ = registry(tmp_path, policy="distiller-only")
    reg.store("main").pour_verified("m", "d", "b")
    srv, base = serve(reg)
    try:
        code, r = post(base, "/annotate", {"slug": "m", "tags": ["entrusted"]})
        assert code == 403 and "distiller-only" in r["error"]
        assert get(base, "/memory/m")["tags"] == []
    finally:
        srv.shutdown()


def test_mcp_bridge_carries_tags_and_annotations_into_the_store(tmp_path):
    reg, _ = registry(tmp_path)
    srv, base = serve(reg)
    try:
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "kura_remember",
                            "arguments": {"slug": "via-mcp", "description": "d", "body": "b",
                                          "tags": ["entrusted"], "keep": "the promise"}}}]
        e = {**os.environ, "KURA_URL": base, "PYTHONPATH": ROOT, "KURA_READONLY": "0"}
        p = subprocess.run([sys.executable, "-m", "distill_kura.mcp"],
                           input="\n".join(json.dumps(m) for m in msgs) + "\n",
                           capture_output=True, text=True, env=e, timeout=60)
        out = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
        assert out[1]["result"]["content"][0]["text"].count('"ok": true') == 1
        st = reg.store("main")
        assert st.tags("via-mcp") == ("entrusted",)
        assert st.annotations("via-mcp") == {"keep": "the promise"}
        # the tool schema advertises the fields a model may fill
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
        p = subprocess.run([sys.executable, "-m", "distill_kura.mcp"],
                           input="\n".join(json.dumps(m) for m in msgs) + "\n",
                           capture_output=True, text=True, env=e, timeout=60)
        out = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
        schema = [t for t in out[1]["result"]["tools"] if t["name"] == "kura_remember"][0]
        assert {"tags", "belongs_because", "keep", "may_fade"} <= set(schema["inputSchema"]["properties"])
        assert "tags" not in schema["inputSchema"]["required"]
    finally:
        srv.shutdown()


def test_cli_round_trip(tmp_path):
    reg, cfg = registry(tmp_path)
    e = {**os.environ, "PYTHONPATH": ROOT, "KURA_CONFIG": cfg}
    run = lambda *a: subprocess.run([sys.executable, "-m", "distill_kura.cli", *a],   # noqa: E731
                                    capture_output=True, text=True, env=e, timeout=60)
    p = run("remember", "cli-m", "trigger", "body", "--tag", "decision", "--tag", "landmine",
            "--keep", "the decision")
    assert p.returncode == 0, p.stderr
    p = run("annotate", "cli-m", "--tag", "decision", "--belongs-because", "it is ours")
    assert p.returncode == 0 and json.loads(p.stdout)["changed"]
    p = run("annotate", "cli-m", "--tag", "decision")
    assert p.returncode == 0 and not json.loads(p.stdout)["changed"]
    st = reg.store("main")
    assert st.tags("cli-m") == ("decision", "landmine")
    assert st.annotations("cli-m") == {"keep": "the decision", "belongs_because": "it is ours"}
    p = run("annotate", "cli-m", "--tag", "Bad")
    assert p.returncode == 1 and "Bad" in p.stdout
