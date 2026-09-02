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

from distill_kura.distill.seeds import Seeds    # noqa: E402
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


# The calibration table for the grounding rule. The constant in `Loom` was chosen from
# these seven cases, all real: change GRAM or GROUNDING_FLOOR and this will tell you what
# it costs. The one that matters most is the inverted negation — a trigger that turns
# "this is NOT a workshop" into "soil and workshop" reads perfectly and is a lie.
GROUNDING_CASES = [
    ("heavy compression, particles dropped",
     "★背景で日誌消化+前景0.15秒応答。記憶の新陳代謝はこの形。GPU普段0%=消化はタダ",
     "★背景日誌消化+前景0.15s応答、GPU0%=タダ", True),
    ("heavy compression, mixed scripts",
     ":8085 記憶への唯一の入り口。索引常駐＋意味の再認＋リンク歩行0.4秒。雲/ローカル/足軽/声が共有",
     ":8085 記憶唯一入口。索引常駐+再認+リンク0.4秒。雲/ローカル/足軽/声共有", True),
    ("compression with English identifiers",
     "persona JSON + IDENTITY.md(同期), 3.7 seed, ケンの関係性・正典決定",
     "YUKI persona JSON + IDENTITY.md, 3.7 seed, ケン関係性・正典", True),
    ("short trigger, verbatim", "★夜空が澄むと星座が見える。周囲の軽さが詳細より重要で、掟の行は同一",
     "★夜空が澄むと星座が見える", True),
    ("INVERTED NEGATION", "ケンの願い: ここは私自身の自律進化の土壌、工房ではない。種を育てよ",
     "YUKI進化プラットフォーム: ケンの願い、自律進化の土壌・工房、種を育てよ", False),
    ("invention", "★夜空が澄むと星座が見える。周囲の軽さが詳細より重要",
     "全く関係のない捏造された主張です", False),
    ("a restatement of nothing", "索引を毎回の会話に常駐させると、モデルは自分が何を知らないかを知る",
     "常在の地図についての重要な知見", False),
]


def test_grounding_calibration(tmp_path):
    loom = Loom(Store(name="s", path=str(tmp_path / "s")), scribe=None)
    for label, desc, trigger, expected in GROUNDING_CASES:
        assert loom._grounded(trigger, desc) is expected, label


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


# ── found by self-review, after the outside reviews ─────────────────────────

def test_rewriting_a_memory_keeps_metadata_it_did_not_write(tmp_path):
    """Regenerating the frontmatter from the template silently dropped every metadata
    key the template did not know — a session id, a node type, a stamp written by
    another tool. On the live store every memory the distiller EXTENDS would have lost
    its provenance fields."""
    s = Store(name="s", path=str(tmp_path / "s"))
    s.init_files()
    with open(s.file_of("old"), "w", encoding="utf-8") as f:
        f.write("---\nname: old\ndescription: d\nmetadata:\n  type: feedback\n"
                "  originSessionId: abc-123\n  node_type: memory\n---\n\nbody\n")
    with open(s.index_path, "a", encoding="utf-8") as f:
        f.write("- [old](old.md) — d\n")
    s.remember_direct("old", "new desc", "new body", type_="feedback")
    fm = s.frontmatter("old")
    assert fm["originSessionId"] == "abc-123" and fm["node_type"] == "memory"
    assert fm["description"] == "new desc"
    assert s.read_exact("old").count("metadata:") == 1


def test_store_ratio_counts_only_what_the_recorded_batches_produced(tmp_path):
    """Dividing the WHOLE store by the raw material of a few batches gave 6.3 on a store
    that predated its metrics — wrong by an order of magnitude, in the direction that
    makes the tool look bad, which is the only reason it was noticed."""
    import hashlib
    from distill_kura.bench import compress
    from distill_kura.registry import Registry
    from distill_kura.thinker import Models
    st = Store(name="m", path=str(tmp_path / "m"))
    st.init_files()
    for i in range(20):
        st.remember(f"pre-{i}", "older, from before any metrics", "a body " * 30)
    os.makedirs(st.still, exist_ok=True)
    with open(os.path.join(st.still, "metrics.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"source_key": "x", "raw_tokens_est": 500}) + "\n")
    man = json.dumps({"source_key": "x"})
    digest = hashlib.sha256(man.encode()).hexdigest()
    os.makedirs(os.path.join(st.path, "_evidence"), exist_ok=True)
    with open(os.path.join(st.path, "_evidence", f"{digest}.json"), "w") as f:
        f.write(man)
    st.pour_verified("one", "from the recorded batch", "short body",
                     meta={"evidence_manifest": f"sha256:{digest}"})
    reg = Registry(stores={"m": st}, modes={}, models=Models.from_config({}), default="m")
    r = compress(reg, st)
    assert r["memories_from_recorded_batches"] == 1
    assert r["memories_unattributed"] == 20
    assert r["store_ratio"] is not None and r["store_ratio"] < 1.0
    assert r["store_ratio_units"] == "estimated / estimated"
    exact = compress(reg, st, tokenizer_command="wc -c")
    assert exact["store_ratio_units"].startswith("mixed")   # the raw side is never exact


def test_a_tampered_manifest_counts_as_unattributed_in_compress(tmp_path):
    """Content-addressed means the name is the hash of the bytes. bench read the
    manifest raw, so a file whose bytes no longer hash to its name still attributed
    its memory to a recorded batch — while doctor called the same file tampered."""
    import hashlib
    from distill_kura.bench import compress
    from distill_kura.registry import Registry
    from distill_kura.thinker import Models
    st = Store(name="m", path=str(tmp_path / "m"))
    st.init_files()
    os.makedirs(st.still, exist_ok=True)
    with open(os.path.join(st.still, "metrics.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"source_key": "x", "raw_tokens_est": 500}) + "\n")
    man = json.dumps({"source_key": "x"})
    digest = hashlib.sha256(man.encode()).hexdigest()
    os.makedirs(os.path.join(st.path, "_evidence"), exist_ok=True)
    mpath = os.path.join(st.path, "_evidence", f"{digest}.json")
    with open(mpath, "w") as f:
        f.write(man)
    st.pour_verified("one", "from the recorded batch", "short body",
                     meta={"evidence_manifest": f"sha256:{digest}"})
    with open(mpath, "a") as f:
        f.write(" ")                                        # same name, other bytes
    reg = Registry(stores={"m": st}, modes={}, models=Models.from_config({}), default="m")
    r = compress(reg, st)
    assert r["memories_from_recorded_batches"] == 0
    assert r["memories_unattributed"] == 1


def test_the_global_distill_table_is_type_checked_too(tmp_path):
    """The per-store table was checked; the global one was not, so the same truthy
    string that was refused under [stores.x.distill] slipped through under [distill]."""
    from distill_kura.registry import Registry
    os.makedirs(tmp_path / "s")
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f'[stores.s]\npath = "{tmp_path / "s"}"\n'
                   '[distill]\ninherit_global_journals = "false"\n', encoding="utf-8")
    try:
        Registry.load(str(cfg))
    except ValueError as e:
        assert "inherit_global_journals must be bool" in str(e)
    else:
        raise AssertionError("a truthy string must not pass for a boolean, globally either")


# ── the seed ledger is one writer at a time ─────────────────────────────────

def test_concurrent_seed_confirms_do_not_lose_seeds(tmp_path):
    """confirm() read the whole ledger, rewrote it through a FIXED `.tmp` name and
    replaced it, with no lock and with sow() appending unlocked beside it. Two
    runners — a hand `kura distill run` next to the tended one — shared that tmp
    file: the second open("w") truncated the first's inode, both wrote into it, and
    one os.replace hit FileNotFoundError while whole seeds vanished from the ledger."""
    path = str(tmp_path / "still" / "seeds.jsonl")
    worker = tmp_path / "sw.py"
    worker.write_text(
        "import sys\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "from distill_kura.distill.seeds import Seeds\n"
        "w, path = sys.argv[1], sys.argv[2]\n"
        "s = Seeds(path)\n"
        "[s.sow(f'idea {w}-{i}', 'test') for i in range(8)]\n"
        "[s.confirm(f'idea {w}-{i}', 'how') for i in range(8)]\n",
        encoding="utf-8")
    procs = [subprocess.Popen([sys.executable, str(worker), str(w), path]) for w in range(8)]
    for p in procs:
        assert p.wait() == 0
    rows = Seeds(path)._all()
    assert len(rows) == 64
    assert all(r.get("confirmed") for r in rows)
