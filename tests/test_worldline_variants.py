"""Resident variants and the M4 metrics of the worldline benchmark.

The same cases, worn with different maps: what changes between rows must be the
map and nothing else, or the comparison the guide's §9 asks for is measuring the
runner instead. No model anywhere — agent-only runs on a stub endpoint, as in
test_worldline.py, and `woven` is built with generate=False.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.cli import main                          # noqa: E402
from distill_kura.store import Store                       # noqa: E402
from distill_kura import worldline as wl                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORIES = os.path.join(ROOT, "bench", "worldline", "memories.json")
CASES = os.path.join(ROOT, "bench", "worldline", "cases.json")


class StubModel:
    """Duck-types Endpoint.ask; answers by utterance so one stub can play a whole
    run. Records every resident map it was shown."""
    def __init__(self, replies: dict[str, str], default: str = "[]"):
        self.replies, self.default = replies, default
        self.seen: list[str] = []

    def ask(self, system, user, max_tokens=400, timeout=None, temperature=None):
        self.seen.append(system)
        return self.replies.get(user, self.default)

    def ask_full(self, system, user, max_tokens=400, timeout=None, temperature=None):
        raw = self.ask(system, user, max_tokens, timeout, temperature)
        if raw is None:
            return None
        return {"content": raw, "reasoning": "", "finish_reason": "stop"}


def build(tmp_path) -> Store:
    s = Store(name="m", path=str(tmp_path / "m"), label="k")
    s.init_files()
    wl.seed(s, MEMORIES)
    return s


def case(id_):
    return next(c for c in wl.load_cases(CASES) if c["id"] == id_)


# ── the fixture itself ──────────────────────────────────────────────────────

def test_every_shipped_case_is_runnable_against_the_seeded_store(tmp_path):
    """A case whose slugs are not in memories.json is skipped honestly — and a
    shipped fixture that skips is a benchmark nobody can run."""
    s = build(tmp_path)
    out = wl.run(s, wl.load_cases(CASES), "fastpath")
    assert out["summary"]["skipped"] == 0, [t["case"] for t in out["traces"] if t["skipped"]]


def test_the_m4_shapes_are_present():
    cats = [c["category"] for c in wl.load_cases(CASES)]
    assert cats.count("prefix-collision") >= 2
    assert "callsign" in cats and "obsolete-resurrection" in cats
    assert cats.count("unknown") >= 1 and "japanese-only" in cats
    ja = case("wf-ja-only-1")["utterance"]
    assert not any(ord(ch) < 128 for ch in ja), "the Japanese-only case must carry no ASCII"


# ── variants: same cases, different maps ────────────────────────────────────

def test_variants_run_the_same_cases_and_report_tokens_per_variant(tmp_path):
    s = build(tmp_path)
    cases = [case("wf-direct-1"), case("wf-unknown-1")]
    stub = StubModel({})
    vs = wl.resident_variants(s, ["canonical", "woven"])
    out = wl.run(s, cases, "agent-only", thinker=stub, resident_variants=vs)
    assert set(out["variants"]) == {"canonical", "woven"}
    assert len(out["traces"]) == 2 * len(cases)
    for name in vs:
        rows = [t for t in out["traces"] if t["resident_variant"] == name]
        assert [t["case"] for t in rows] == [c["id"] for c in cases], \
            "every variant sees the same cases in the same order"
        assert out["variants"][name]["resident_tokens"] > 0
        assert all(t["resident_tokens"] == out["variants"][name]["resident_tokens"]
                   for t in rows)
    assert vs["woven"] != vs["canonical"], "the loom trimmed at least one trigger"
    assert len(stub.seen) == 2 * len(cases) and vs["woven"] in "".join(stub.seen[2:]), \
        "the woven text is what the agent actually wore for the woven rows"


def test_woven_variant_calls_no_model(tmp_path):
    """A benchmark must not spend GPU seconds building its own input; the woven
    map is what the loom would give NOW, mechanically."""
    s = build(tmp_path)
    vs = wl.resident_variants(s, ["woven"])
    assert "freetoken-hybrid.md" in vs["woven"]


def test_resident_file_is_honoured(tmp_path):
    """The adaptive shadow does not exist yet; a file is how it gets measured."""
    s = build(tmp_path)
    shadow = tmp_path / "shadow.md"
    shadow.write_text("# shadow\n- [freetoken-hybrid](freetoken-hybrid.md) — all hands\n",
                      encoding="utf-8")
    vs = wl.resident_variants(s, ["canonical", "adaptive8"], {"adaptive8": str(shadow)})
    assert vs["adaptive8"] == shadow.read_text(encoding="utf-8")
    stub = StubModel({})
    out = wl.run(s, [case("wf-direct-1")], "agent-only", thinker=stub, resident_variants=vs)
    a8 = out["variants"]["adaptive8"]["resident_tokens"]
    assert 0 < a8 < out["variants"]["canonical"]["resident_tokens"]
    assert any(vs["adaptive8"] in seen for seen in stub.seen)


def test_unknown_variant_name_fails_loudly(tmp_path):
    """A typo that fell back to canonical would print a healthy comparison of one
    map against itself."""
    s = build(tmp_path)
    with pytest.raises(ValueError, match="unknown resident variant 'wovne'"):
        wl.resident_variants(s, ["canonical", "wovne"])


def test_single_resident_without_variants_is_named_honestly(tmp_path):
    s = build(tmp_path)
    out = wl.run(s, [case("wf-direct-1")], "fastpath")
    assert out["traces"][0]["resident_variant"] == "canonical"
    out = wl.run(s, [case("wf-direct-1")], "fastpath", resident="# hand-made\n")
    assert out["traces"][0]["resident_variant"] == "resident", \
        "a caller's own text must not be recorded as the canonical index"


# ── the M4 metrics ──────────────────────────────────────────────────────────

def test_remembered_but_unreachable_counts_exactly_the_narrow_doors(tmp_path):
    """Three cases: a direct name tier zero hits; a Japanese ellipsis it stays
    silent on; an unknown with nothing to reach. Only the ellipsis is a memory
    that exists behind a door too narrow."""
    s = build(tmp_path)
    cases = [case("wf-direct-1"), case("wf-ja-only-1"), case("wf-unknown-1")]
    out = wl.run(s, cases, "fastpath")
    by = {t["case"]: t for t in out["traces"]}
    assert by["wf-direct-1"]["target_reached"] is True
    assert by["wf-direct-1"]["remembered_but_unreachable"] is False
    assert by["wf-ja-only-1"]["target_reached"] is False
    assert by["wf-ja-only-1"]["remembered_but_unreachable"] is True
    assert by["wf-unknown-1"]["remembered_but_unreachable"] is False, \
        "nothing was remembered, so nothing was unreachable"
    assert out["summary"]["remembered_but_unreachable"] == 1


def test_remembered_but_unreachable_is_never_charged_to_full(tmp_path):
    """In `full` the thinker's rescue is exactly what hides a thin map, so the
    metric is defined only for the map-reading routes."""
    s = build(tmp_path)
    tr = wl.run_case(s, case("wf-callsign-only-1"), "full", thinker=None)
    assert tr["remembered_but_unreachable"] is False


def test_agent_only_miss_on_an_existing_target_is_unreachable(tmp_path):
    s = build(tmp_path)
    stub = StubModel({}, default="[]")
    tr = wl.run_case(s, case("wf-ja-only-1"), "agent-only", thinker=stub)
    assert tr["target_reached"] is False and tr["remembered_but_unreachable"] is True


def test_unnecessary_opens_excludes_target_and_related(tmp_path):
    s = build(tmp_path)
    c = case("wf-direct-1")               # target freetoken-hybrid, related exl3-cpu
    stub = StubModel({}, default='["freetoken-hybrid", "exl3-cpu", "gpu-power-measure"]')
    tr = wl.run_case(s, c, "agent-only", thinker=stub)
    assert tr["target_reached"] is True and tr["related_reached"] == ["exl3-cpu"]
    assert tr["unnecessary_opens"] == ["gpu-power-measure"]
    assert wl.summarize([tr])["unnecessary_opens"] == 1


def test_obsolete_branch_is_counted_apart_from_wrong_branch(tmp_path):
    """Resurrecting a thrown-away plan and landing on a live neighbour fail
    differently; one count would let a rise in the first hide under a fall in
    the second."""
    s = build(tmp_path)
    dead = StubModel({}, default='["archive-on-slow-disk"]')
    tr = wl.run_case(s, case("wf-obsolete-1"), "agent-only", thinker=dead)
    assert tr["obsolete_branch"] is True and tr["wrong_branch"] is False
    assert tr["target_reached"] is False
    live = StubModel({}, default='["fan-install-dimm-side"]')
    tr2 = wl.run_case(s, case("wf-prefix-1"), "agent-only", thinker=live)
    assert tr2["wrong_branch"] is True and tr2["obsolete_branch"] is False
    sm = wl.summarize([tr, tr2])
    assert sm["obsolete_branch"] == 1 and sm["wrong_branch"] == 1


def test_honest_unknown_is_named_in_the_summary(tmp_path):
    s = build(tmp_path)
    stub = StubModel({}, default="[]")
    out = wl.run(s, [case("wf-unknown-1"), case("wf-unknown-3")], "agent-only", thinker=stub)
    sm = out["summary"]
    assert sm["honest_unknown"] == 2 == sm["unknown_refused_correctly"]


def test_prefix_collision_hit_on_the_right_tail_is_clean(tmp_path):
    s = build(tmp_path)
    stub = StubModel({}, default='["fan-install-throughput"]')
    tr = wl.run_case(s, case("wf-prefix-1"), "agent-only", thinker=stub)
    assert tr["target_reached"] and not tr["wrong_branch"] and tr["unnecessary_opens"] == []


# ── the CLI: table by default, --json for traces, exit code contract ────────

def _cfg(tmp_path) -> str:
    build(tmp_path)
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[stores.m]
path = "{tmp_path / 'm'}"
[models.thinker]
url = "http://127.0.0.1:9/v1"
model = "none"
""", encoding="utf-8")
    return str(cfg)


def test_cli_prints_one_row_per_variant_with_tokens(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    shadow = tmp_path / "shadow.md"
    shadow.write_text("# shadow\n- [freetoken-hybrid](freetoken-hybrid.md) — all hands\n",
                      encoding="utf-8")
    rc = main(["-c", cfg, "-s", "m", "bench", "worldline", "--cases", CASES,
               "--routing", "fastpath", "--resident", "canonical,woven,shadow",
               "--resident-file", f"shadow={shadow}",
               "--trace", str(tmp_path / "t.jsonl")])
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.strip()]
    assert lines[1].startswith("variant") and "resident_tokens" in lines[1]
    assert "remembered_but_unreachable" in lines[1] and "obsolete_branch" in lines[1]
    first = lines[:next(i for i, l in enumerate(lines)
                        if l.startswith("paired-format-valid"))]
    rows = {l.split()[0]: l for l in first[3:]}
    assert set(rows) == {"canonical", "woven", "shadow"}
    assert int(rows["shadow"].split()[1]) < int(rows["canonical"].split()[1])
    # tier zero never names a dead plan the fixture does not bait it with by slug,
    # so the run passes; a wrong or obsolete landing would have returned 1
    assert rc in (0, 1)


def test_cli_json_dumps_traces_stamped_with_their_variant(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    rc = main(["-c", cfg, "-s", "m", "bench", "worldline", "--cases", CASES,
               "--routing", "fastpath", "--resident", "canonical,woven", "--json",
               "--trace", str(tmp_path / "t.jsonl")])
    r = json.loads(capsys.readouterr().out)
    assert {t["resident_variant"] for t in r["traces"]} == {"canonical", "woven"}
    assert set(r["variants"]) == {"canonical", "woven"}
    assert "score" not in r["summary"]
    expected = 0 if (r["summary"]["runnable"] and not r["summary"]["wrong_branch"]
                     and not r["summary"]["obsolete_branch"]) else 1
    assert rc == expected


def test_cli_unknown_variant_exits_with_the_name(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(["-c", cfg, "-s", "m", "bench", "worldline", "--cases", CASES,
              "--routing", "fastpath", "--resident", "canonical,wovne"])
    assert "wovne" in str(e.value)


def test_cli_resident_file_wants_name_equals_path(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(["-c", cfg, "-s", "m", "bench", "worldline", "--cases", CASES,
              "--routing", "fastpath", "--resident-file", "nopath"])
    assert "NAME=PATH" in str(e.value)


def test_every_trace_names_the_map_it_wore_by_hash(tmp_path):
    """'adaptive' is a label; the sha says WHICH trigger set was measured (FAILURES FOUND #2)."""
    s = build(tmp_path)
    stub = StubModel({})
    vs = wl.resident_variants(s, ["canonical", "woven"])
    out = wl.run(s, [case("wf-direct-1")], "agent-only", thinker=stub, resident_variants=vs)
    shas = {t["resident_variant"]: t["resident_sha"] for t in out["traces"]}
    assert len(shas["canonical"]) == 12 and shas["canonical"] != shas["woven"]
    for name, v in out["variants"].items():
        assert v["resident_sha"] == shas[name]


def test_a_format_error_keeps_the_head_of_the_reply_as_its_witness(tmp_path):
    """FAILURES FOUND (house run 2026-09-02): 32/42 rows were format_error with no way
    to tell prose from a fenced array from a reply cut by the output cap."""
    s = build(tmp_path)
    stub = StubModel({}, default="Looking at the map, the most relevant entry is freetoken-hybrid ...")
    out = wl.run(s, [case("wf-direct-1")], "agent-only", thinker=stub)
    t = out["traces"][0]
    assert t["format_error"] and t["reply_head"].startswith("Looking at the map") \
        and t["reply_chars"] == len(stub.default)


# ── the visible answer, and the two ways it goes missing ────────────────────

class SplitStub:
    """Duck-types Endpoint.ask_full: a server that answers on two channels. The
    benchmark must read `content` alone — the reasoning is a witness, never an
    answer."""
    def __init__(self, content="", reasoning="", finish_reason="stop"):
        self.content, self.reasoning, self.finish_reason = content, reasoning, finish_reason
        self.ask_calls = 0

    def ask(self, system, user, max_tokens=400, timeout=None, temperature=None):
        # Present so a mistaken caller is caught by the assertions, not by AttributeError.
        self.ask_calls += 1
        return self.content or self.reasoning

    def ask_full(self, system, user, max_tokens=400, timeout=None, temperature=None):
        return {"content": self.content, "reasoning": self.reasoning,
                "finish_reason": self.finish_reason}


def _row(store, stub):
    return wl.run(store, [case("wf-direct-1")], "agent-only", thinker=stub)["traces"][0]


def test_a_clean_answer_is_neither_truncated_nor_reasoning_only(tmp_path):
    s = build(tmp_path)
    t = _row(s, SplitStub(content='["freetoken-hybrid"]'))
    assert t["target_reached"] and not t["format_error"]
    assert not t["truncated"] and not t["reasoning_only"]


def test_the_reasoning_channel_is_never_read_as_the_answer(tmp_path):
    """ask() would hand back the reasoning when the content is empty. That fallback
    is a thinker-side kindness; here it would score a model that said nothing."""
    s = build(tmp_path)
    stub = SplitStub(content="", reasoning='The map has it: ["freetoken-hybrid"]')
    t = _row(s, stub)
    assert stub.ask_calls == 0                  # ask_full, not ask
    assert t["reasoning_only"] and t["format_error"] and not t["target_reached"]
    assert t["opened"] == [] and t["reply_chars"] == 0
    assert t["reply_head"].startswith("[reasoning] The map has it")


def test_a_truncated_reply_is_still_a_format_error(tmp_path):
    """Both flags can be true at once: the cap explains the failure, it does not
    excuse it — an excused row would hide 'raise max_tokens' behind a healthy count."""
    s = build(tmp_path)
    t = _row(s, SplitStub(content='["freetoken-hyb', finish_reason="length"))
    assert t["truncated"] and t["format_error"] and not t["reasoning_only"]


def test_a_complete_valid_answer_can_still_be_marked_truncated(tmp_path):
    """finish_reason is an observation about the call, not a verdict on the parse."""
    s = build(tmp_path)
    t = _row(s, SplitStub(content='["freetoken-hybrid"]', finish_reason="length"))
    assert t["truncated"] and not t["format_error"] and t["target_reached"]


def test_reasoning_only_and_truncated_can_both_be_true(tmp_path):
    """The house's own failure: the whole budget went into thinking and the cap hit."""
    s = build(tmp_path)
    t = _row(s, SplitStub(content="   ", reasoning="thinking..." * 40,
                          finish_reason="length"))
    assert t["truncated"] and t["reasoning_only"] and t["format_error"]


def test_content_present_beside_reasoning_keeps_the_content_as_the_head(tmp_path):
    s = build(tmp_path)
    t = _row(s, SplitStub(content="I think so", reasoning="long private thought"))
    assert not t["reasoning_only"] and t["format_error"]
    assert t["reply_head"] == "I think so"


def test_an_unreachable_model_is_still_a_skip(tmp_path):
    """Unchanged by the split: an outage is not an honest 'nothing matched'."""
    class Dead:
        def ask_full(self, *a, **k):
            return None
    t = _row(build(tmp_path), Dead())
    assert t["skipped"] == "model unreachable" and not t["truncated"]


def test_the_two_counts_reach_the_summary_and_the_table(tmp_path):
    s = build(tmp_path)
    out = wl.run(s, [case("wf-direct-1")], "agent-only",
                 thinker=SplitStub(content="", reasoning="thought", finish_reason="length"))
    assert out["summary"]["truncated"] == 1 and out["summary"]["reasoning_only"] == 1
    from distill_kura.cli import _worldline_table
    head = _worldline_table(out).splitlines()[1]
    assert "truncated" in head and "reasoning_only" in head


# ── which questions were answered ───────────────────────────────────────────

def test_two_byte_different_case_files_have_different_shas(tmp_path):
    """The digest is of the file's bytes, so a case edited in place — same count,
    same path — cannot pass itself off as the old set."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    cases = wl.load_cases(CASES)
    a.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    b.write_text(json.dumps({"cases": cases}) + "\n", encoding="utf-8")
    ca, sa = wl.load_case_set(str(a))
    cb, sb = wl.load_case_set(str(b))
    assert ca == cb and len(sa) == 64 and sa != sb
    assert wl.load_cases(str(a)) == ca        # the old shape still works


def test_the_case_set_sha_is_on_every_row_and_on_the_result(tmp_path):
    s = build(tmp_path)
    cases, sha = wl.load_case_set(CASES)
    out = wl.run(s, cases, "fastpath", case_set_sha=sha,
                 resident_variants=wl.resident_variants(s, ["canonical", "woven"]))
    assert out["case_set_sha"] == sha
    assert out["traces"] and all(t["case_set_sha"] == sha for t in out["traces"])


def test_the_cli_header_carries_the_case_set_sha(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    main(["-c", cfg, "-s", "m", "bench", "worldline", "--cases", CASES,
          "--routing", "fastpath"])
    sha = wl.load_case_set(CASES)[1]
    assert f"case_set_sha={sha[:12]}" in capsys.readouterr().out


# ── the promotion view: paired over the format-valid cases, and compare ──────

class ByVariantStub:
    """Answers differently depending on which map it is shown, so a paired
    comparison has something to be paired about."""
    def __init__(self, by_marker: dict, default="[]"):
        self.by_marker, self.default = by_marker, default

    def ask_full(self, system, user, max_tokens=400, timeout=None, temperature=None):
        content = self.default
        for marker, reply in self.by_marker.items():
            if marker in system:
                content = reply.get(user, reply.get("*", self.default))
                break
        return {"content": content, "reasoning": "", "finish_reason": "stop"}


def _two_variant_run(tmp_path, marker_reply):
    s = build(tmp_path)
    cases, sha = wl.load_case_set(CASES)
    thin = tmp_path / "thin.md"
    thin.write_text("# thin map — MARKER-THIN\n", encoding="utf-8")
    vs = {"canonical": s.index_text(), "thin": thin.read_text(encoding="utf-8")}
    return wl.run(s, cases[:4], "agent-only", thinker=marker_reply,
                  resident_variants=vs, case_set_sha=sha)


def test_paired_valid_counts_only_the_cases_readable_everywhere(tmp_path):
    """A variant that garbles the cases it finds hard must not look better for it:
    such a row is a failure for one side and simply absent for the other."""
    out = _two_variant_run(tmp_path, ByVariantStub(
        {"MARKER-THIN": {"*": "sorry, I can't tell"}}, default="[]"))
    pv = out["paired_valid"]
    assert set(pv) == {"canonical", "thin"}
    # `thin` was unreadable on every case, so nothing is paired at all.
    assert pv["thin"]["cases"] == 0 and pv["canonical"]["cases"] == 0
    # ...while the all-cases summary still shows canonical answering.
    assert out["variants"]["canonical"]["summary"]["format_error"] == 0
    assert out["variants"]["thin"]["summary"]["format_error"] > 0


def test_paired_valid_matches_the_summary_when_every_row_is_readable(tmp_path):
    out = _two_variant_run(tmp_path, ByVariantStub({}, default="[]"))
    for name, v in out["variants"].items():
        assert out["paired_valid"][name]["cases"] == v["summary"]["runnable"]
        assert (out["paired_valid"][name]["target_reached"]
                == v["summary"]["target_reached"])


def test_the_paired_table_is_printed_under_the_first(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    main(["-c", cfg, "-s", "m", "bench", "worldline", "--cases", CASES,
          "--routing", "fastpath"])
    out = capsys.readouterr().out
    assert "paired-format-valid" in out
    tail = out.split("paired-format-valid", 1)[1]
    for col in ("cases", "target_reached", "wrong_branch", "obsolete_branch",
                "remembered_but_unreachable"):
        assert col in tail


def _result_file(tmp_path, name, sha, *, canonical_target=1, thin_target=0,
                 wrong=0, fmt=0):
    """A minimal worldline result, hand-built so a compare test says exactly what
    it means instead of depending on how a fixture happens to score."""
    traces = []
    for variant, hits in (("canonical", canonical_target), ("thin", thin_target)):
        for i in range(2):
            traces.append({"case": f"c{i}", "resident_variant": variant,
                           "case_set_sha": sha, "skipped": None,
                           "format_error": False, "target_reached": i < hits,
                           "wrong_branch": False, "obsolete_branch": False,
                           "remembered_but_unreachable": False})
    variants = {
        "canonical": {"resident_tokens": 10, "resident_sha": "x",
                      "summary": {"runnable": 2, "target_reached": canonical_target,
                                  "wrong_branch": wrong, "obsolete_branch": 0,
                                  "remembered_but_unreachable": 0,
                                  "format_error": fmt}},
        "thin": {"resident_tokens": 5, "resident_sha": "y",
                 "summary": {"runnable": 2, "target_reached": thin_target,
                             "wrong_branch": 0, "obsolete_branch": 0,
                             "remembered_but_unreachable": 0, "format_error": 0}}}
    p = tmp_path / name
    p.write_text(json.dumps({"store": "m", "routing": "agent-only", "cases": 2,
                             "case_set_sha": sha, "variants": variants,
                             "traces": traces}), encoding="utf-8")
    return str(p)


def test_compare_refuses_two_different_case_sets(tmp_path):
    a = _result_file(tmp_path, "a.json", "aaa")
    b = _result_file(tmp_path, "b.json", "bbb")
    with pytest.raises(SystemExit) as e:
        main(["bench", "worldline-compare", a, b])
    msg = str(e.value)
    assert "case sets differ" in msg and "must not be compared" in msg


def test_compare_prints_recovery_twice_and_the_safety_deltas(tmp_path, capsys):
    a = _result_file(tmp_path, "a.json", "s1", canonical_target=1, fmt=2)
    b = _result_file(tmp_path, "b.json", "s2".replace("s2", "s1"),
                     canonical_target=2, fmt=0)
    rc = main(["bench", "worldline-compare", a, b])
    out = capsys.readouterr().out
    assert rc == 0
    assert "paired-format-valid cases (valid in every variant of BOTH runs): 2" in out
    assert "all cases" in out and "paired valid" in out
    assert "format_error       A 2  B 0  delta -2" in out
    for k in ("wrong_branch", "obsolete_branch", "remembered_but_unreachable"):
        assert k in out
    assert "delta +0" in out                # an unmoved safety count still shows its sign


def test_compare_json_carries_the_numbers(tmp_path, capsys):
    a = _result_file(tmp_path, "a.json", "s1", canonical_target=1)
    b = _result_file(tmp_path, "b.json", "s1", canonical_target=2)
    main(["bench", "worldline-compare", a, b, "--json"])
    c = json.loads(capsys.readouterr().out)
    assert c["paired_valid_cases"] == 2
    v = c["variants"]["canonical"]
    assert v["all_cases"]["recovery_a"] == 0.5 and v["all_cases"]["recovery_b"] == 1.0
    assert v["paired_valid"]["cases"] == 2
    assert v["safety"]["wrong_branch"]["delta"] == 0
    assert "score" not in json.dumps(c)     # no composite, on purpose


def test_compare_pairs_only_cases_valid_in_both_files(tmp_path):
    a = json.loads(open(_result_file(tmp_path, "a.json", "s1"), encoding="utf-8").read())
    b = json.loads(open(_result_file(tmp_path, "b.json", "s1"), encoding="utf-8").read())
    # one case went unreadable in B's thin variant: it leaves the paired set for BOTH.
    for t in b["traces"]:
        if t["case"] == "c1" and t["resident_variant"] == "thin":
            t["format_error"] = True
    c = wl.compare(a, b)
    assert c["paired_valid_cases"] == 1
    assert c["variants"]["canonical"]["paired_valid"]["cases"] == 1
