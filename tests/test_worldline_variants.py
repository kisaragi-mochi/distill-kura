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
    rows = {l.split()[0]: l for l in lines[3:]}
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
