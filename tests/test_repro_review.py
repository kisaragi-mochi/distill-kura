"""Findings from a third-party reproducibility review, held shut.

The review's question was not "is this safe" but "can someone else get the same result
from it". Most of what it found is that behaviour and documentation had drifted apart —
a parameter that reads as a total and is per-item, an estimator described as more
accurate than it is, a compatibility claim wider than the client, a quality gate that
tests the alphabet rather than the content.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.recall import recall           # noqa: E402
from distill_kura.store import Store             # noqa: E402
from distill_kura.thinker import Endpoint        # noqa: E402
from distill_kura.tokens import estimate         # noqa: E402
from distill_kura.weave import Loom              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Stub:
    def __init__(self, answer):
        self.answer = answer

    def ask(self, system, user, **kw):
        return self.answer


# ── P1-5: the trigger gate tested the alphabet, not the content ─────────────

def test_a_japanese_trigger_with_no_ascii_or_digits_is_accepted(tmp_path):
    """The gate demanded a digit, three ASCII letters, or ⚠ as proof of "specificity".
    That is a test of SCRIPT: 「周囲の軽さが詳細より重要」 and 「★夜空が澄むと星座が見える」
    carry none of them, so a model that wrote well was overruled and the mechanical
    trimmer used instead. ★ — the store's own marker — was not even in the list."""
    loom = Loom(Store(name="s", path=str(tmp_path / "s")), scribe=None)
    desc = "★夜空が澄むと星座が見える。周囲の軽さが詳細より重要で、掟の行は同一なのに痩せた側が勝った"
    for good in ("★夜空が澄むと星座が見える", "周囲の軽さが詳細より重要", "掟の行は同一"):
        assert loom._acceptable(good, "題名", desc), good


def test_an_invented_trigger_is_still_rejected(tmp_path):
    """Dropping the alphabet test must not drop the point of having a test."""
    loom = Loom(Store(name="s", path=str(tmp_path / "s")), scribe=None)
    desc = "★夜空が澄むと星座が見える。周囲の軽さが詳細より重要"
    assert not loom._acceptable("全く関係のない捏造された主張です", "題名", desc)
    assert not loom._acceptable("the model made this up entirely", "題名", desc)


def test_grounding_survives_light_paraphrase(tmp_path):
    loom = Loom(Store(name="s", path=str(tmp_path / "s")), scribe=None)
    desc = "索引を毎回の会話に常駐させると、モデルは自分が何を知らないかを知る"
    assert loom._grounded("索引を毎回の会話に常駐させる", desc)
    assert not loom._grounded("まったく別の話題について書かれた行", desc)


# ── P2-3 / P2-2: one estimator, honestly described ──────────────────────────

def test_doctor_and_index_use_the_fitted_estimator(tmp_path):
    """`len(text)//2` is biased 8-23% low against real tokenizers, and low is the
    direction that silently overflows a window. Two places still used it."""
    s = Store(name="s", path=str(tmp_path / "s"))
    s.init_files()
    s.remember("jp", "日本語の説明文がここに入ります。索引は毎回まるごと読まれます", "body")
    idx = s.index_text()
    assert s.doctor()["index_tokens_est"] == estimate(idx)
    assert s.doctor()["index_tokens_est"] != len(idx) // 2


# ── P2-1: `chars` was per memory and read as a total ────────────────────────

def test_recall_can_bound_the_whole_context_not_just_each_memory(tmp_path):
    s = Store(name="s", path=str(tmp_path / "s"))
    s.init_files()
    for i in range(6):
        # Several paragraphs, so trimming has something to keep: a single giant
        # paragraph is dropped whole by `fit`, and then no budget is ever spent.
        body = "\n\n".join(f"paragraph {j} " + "word " * 60 for j in range(10))
        s.remember(f"m{i}", f"memory {i}", body + f"\n\n[[m{(i + 1) % 6}]]")
    wide = recall(s, Stub('["m0"]'), "anything", hops=3, chars=3000)
    tight = recall(s, Stub('["m0"]'), "anything", hops=3, chars=3000, total_chars=4000)
    assert wide["chars"] > 4000                      # per-memory budget, many memories
    assert tight["chars"] <= 4000
    assert tight["dropped_for_budget"]               # and it says which ones it left out
    assert tight["chars_per_memory"] == 3000 and tight["total_chars"] == 4000

    # A ceiling exceeded by a little is not a ceiling: `fit` keeps a memory's opening
    # whole, so a piece can come back bigger than the room it was given.
    for total in (4000, 2000, 900):
        d = recall(s, Stub('["m0"]'), "anything", hops=3, chars=3000, total_chars=total)
        assert d["chars"] <= total, (total, d["chars"])

    # And a budget too small for even one memory answers with a cut memory, never with
    # silence: an empty context reads as "nothing is remembered".
    tiny = recall(s, Stub('["m0"]'), "anything", hops=3, chars=3000, total_chars=700)
    assert tiny["chars"] and tiny["chars"] <= 700
    assert tiny["included"] == ["m0"] and "truncated" in tiny["context"]


# ── P1-8: what "OpenAI-compatible" actually covers ──────────────────────────

class Strict(BaseHTTPRequestHandler):
    """A service that 400s on an unknown top-level field, as a strict one does."""
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        Strict.seen.append(sorted(body))
        if "chat_template_kwargs" in body:
            self.send_response(400)
            b = b'{"error":"unknown field chat_template_kwargs"}'
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            return self.wfile.write(b)
        out = json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def test_a_strict_service_gets_a_second_plainer_attempt():
    """"The server rejected a field" and "the server is down" are different problems,
    and only one of them is worth giving up on."""
    Strict.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Strict)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        e = Endpoint(url=f"http://127.0.0.1:{srv.server_address[1]}/v1", model="m")
        assert e.ask("s", "u") == "hello"
        assert any("chat_template_kwargs" in body for body in Strict.seen)   # tried
        assert any("chat_template_kwargs" not in body for body in Strict.seen)  # then plainer
        assert e.last_error == ""
    finally:
        srv.shutdown()


def test_the_openai_dialect_never_sends_the_local_server_field():
    Strict.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Strict)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        e = Endpoint(url=f"http://127.0.0.1:{srv.server_address[1]}/v1", model="m",
                     dialect="openai")
        assert e.ask("s", "u") == "hello"
        assert all("chat_template_kwargs" not in body for body in Strict.seen)
    finally:
        srv.shutdown()


def test_a_failure_says_which_kind_of_failure_it_was():
    """None used to mean everything: wrong key, wrong url, wrong model, rejected field."""
    e = Endpoint(url="http://127.0.0.1:1/v1", model="m")
    assert e.ask("s", "u") is None
    assert "unreachable" in e.last_error
    assert Endpoint().ask("s", "u") is None
    assert Endpoint(url="", model="m").last_error in ("", "no url configured")


# ── P1-7: a memory and its index line are one change ────────────────────────

def test_concurrent_writers_do_not_lose_index_lines(tmp_path):
    """Two writers interleaved and the second one's read-modify-write of MEMORY.md
    dropped the first's line: the memory existed and nothing pointed at it, which makes
    it invisible to recall."""
    path = str(tmp_path / "s")
    Store(name="s", path=path).init_files()
    worker = tmp_path / "w.py"
    worker.write_text(
        "import sys\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "from distill_kura.store import Store\n"
        "w, path = sys.argv[1], sys.argv[2]\n"
        "st = Store(name='s', path=path)\n"
        "[st.remember(f'w{w}-m{i}', f'memory {w}-{i}', 'body') for i in range(8)]\n",
        encoding="utf-8")
    procs = [subprocess.Popen([sys.executable, str(worker), str(w), path]) for w in range(8)]
    for p in procs:
        assert p.wait() == 0
    d = Store(name="s", path=path).doctor()
    assert d["memories"] == 64
    assert d["not_in_index"] == [] and d["index_orphans"] == []


def test_the_index_is_replaced_atomically(tmp_path):
    """A crash mid-append left a half-written line in the one file read every turn."""
    s = Store(name="s", path=str(tmp_path / "s"))
    s.init_files()
    s.remember("a", "first", "body")
    before = s.index_text()
    s.remember("b", "second", "body")
    assert before in s.index_text()          # append preserved what was there
    assert s.index_text().endswith("\n")     # and left no partial line
