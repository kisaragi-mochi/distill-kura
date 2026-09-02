"""Tier zero of recall: the deterministic five-head recognizer.

These tests check the *contract*, not the tuning: a direct question that names a
memory is answered without the thinker (even with no thinker at all), an ambiguous
or nonsense question falls through — silence is the honest answer — the in-process
index follows the store when it changes, and the config switch really switches it
off. A wrong answer from tier zero would be served with full confidence and no
model in the loop, which is why the falling-through cases matter as much as the hit.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura import fastpath                       # noqa: E402
from distill_kura.recall import recall                  # noqa: E402
from distill_kura.registry import Registry              # noqa: E402
from distill_kura.store import Store                    # noqa: E402


class StubThinker:
    """Stands in for an endpoint. `answer=None` means unreachable."""
    def __init__(self, answer):
        self.answer = answer
        self.seen: list[str] = []

    def ask(self, system, user, **kw):
        self.seen.append(user)
        return self.answer


def a_store(tmp_path, name="s") -> Store:
    s = Store(name=name, path=str(tmp_path / name), label=name)
    s.init_files()
    s.remember("ssd-tier-mission", "running a huge model off an SSD tier", "body [[cooling]]")
    s.remember("cooling", "the fans had to go in before the CPU run", "body")
    s.remember("poker-ladder", "decision making under uncertainty, poker as a rung", "body")
    return s


# ── (a) a confident hit skips the thinker ───────────────────────────────────

def test_a_direct_question_answers_with_no_thinker_at_all(tmp_path):
    """The robustness win: recall used to degrade straight to word overlap when the
    thinker was down. A direct question now still finds its memory, confidently."""
    s = a_store(tmp_path)
    d = recall(s, None, "what did we decide about the ssd-tier-mission?", hops=1)
    assert d["how"] == "fastpath"
    assert d["picked"] == ["ssd-tier-mission"]
    assert d["walked"] == ["ssd-tier-mission", "cooling"]   # the [[link]] still walks
    assert d["fastpath_verdict"] == "ok"
    assert d["fastpath_ms"] is not None
    assert "ssd-tier-mission" in d["context"]


def test_a_confident_hit_never_asks_the_thinker(tmp_path):
    s = a_store(tmp_path)
    t = StubThinker('["cooling"]')          # would answer differently; must not be asked
    d = recall(s, t, "ssd-tier-mission?", hops=0)
    assert d["how"] == "fastpath"
    assert d["picked"] == ["ssd-tier-mission"]
    assert t.seen == []                     # the ~900 ms prefill simply did not happen


# ── (b) an ambiguous question falls through ─────────────────────────────────

def test_an_ambiguous_question_falls_through_to_the_thinker(tmp_path):
    """Two memories that score identically must produce silence, not a coin toss:
    the ratio gate (top1/top2 >= 1.15) hands the question to the tier that judges."""
    s = Store(name="s", path=str(tmp_path / "s"), label="s")
    s.init_files()
    s.remember("north-rack-cooling", "cooling the north rack: fans and airflow", "body")
    s.remember("south-rack-cooling", "cooling the south rack: fans and airflow", "body")
    s.remember("gpu-topology", "which GPU sits on which PCIe switch", "body")
    t = StubThinker('["north-rack-cooling"]')
    d = recall(s, t, "rack cooling fans airflow", hops=0)
    assert d["fastpath_verdict"] == "no-confident-hit"
    assert d["how"] == "meaning"            # the thinker path ran, unchanged
    assert len(t.seen) == 1
    assert d["picked"] == ["north-rack-cooling"]


# ── (c) gate honesty ────────────────────────────────────────────────────────

def test_nonsense_yields_no_hits(tmp_path):
    s = a_store(tmp_path)
    fp = fastpath.lookup(s, "purple elephant marmalade zeppelin")
    assert fp["hits"] == []
    assert fp["verdict"] == "no-confident-hit"
    # and through recall: the old degradation path is untouched
    d = recall(s, None, "purple elephant marmalade zeppelin", hops=0)
    assert d["how"].startswith("words")
    assert d["fastpath_verdict"] == "no-confident-hit"


def test_a_raised_gate_makes_even_a_direct_hit_shy(tmp_path):
    s = a_store(tmp_path)
    assert fastpath.lookup(s, "ssd-tier-mission?")["hits"]
    assert fastpath.lookup(s, "ssd-tier-mission?", gate=99.0)["hits"] == []


# ── (d) the cache follows the store ─────────────────────────────────────────

def test_the_index_follows_the_store(tmp_path):
    s = a_store(tmp_path)
    assert fastpath.lookup(s, "quantization-bake?")["hits"] == []
    s.remember("quantization-bake", "baking NVFP4 quantized models", "body")
    fp = fastpath.lookup(s, "quantization-bake?")
    assert [h["slug"] for h in fp["hits"]] == ["quantization-bake"]


def test_a_rewrite_that_leaves_the_index_alone_still_refreshes_tier_zero(tmp_path):
    """A memory whose index line is a grouped family line (`- topic — [A](a.md)/[B](b.md)`)
    is rewritten: `_write` cannot refresh that line, so it reports `indexed: False`, the
    index bytes and the memory count both stand still, and a stamp made of those two
    alone would keep serving the old description while `doctor` called itself fresh.
    The store revision is the third number that cannot miss the write.

    The rewrite changes the DESCRIPTION, not the body: the body head weighs 0.5 against
    a gate of 1.0, so a body-only edit could never gate through and the test would pass
    or fail for the wrong reason."""
    s = a_store(tmp_path)
    s.remember("alpha-one", "first thing", "body one")
    s.remember("beta-two", "second thing", "body two")
    with open(s.index_path, "w", encoding="utf-8") as f:
        f.write("- greek letters — [Alpha](alpha-one.md)/[Beta](beta-two.md)\n")
    assert fastpath.lookup(s, "axolotl narwhal")["hits"] == []      # prime the cache

    r = s.remember("alpha-one", "axolotl narwhal is the new description", "body")
    assert r["indexed"] is False        # the precondition: the index line stayed put
    assert s.doctor()["fastpath"]["fresh"] is False
    assert [h["slug"] for h in fastpath.lookup(s, "axolotl narwhal")["hits"]] == ["alpha-one"]


def test_doctor_reports_the_fastpath_block(tmp_path):
    s = a_store(tmp_path)
    assert s.doctor()["fastpath"] == {"built": False}    # lazy: nothing until a recall
    fastpath.lookup(s, "ssd-tier-mission?")
    d = s.doctor()["fastpath"]
    assert d["built"] is True and d["fresh"] is True
    assert d["memories"] == 3
    assert set(d["head_vocab"]) == {"word", "char3", "char2", "body"}
    assert all(v > 0 for v in d["head_vocab"].values())
    s.remember("new-memory", "something else entirely", "body")
    assert s.doctor()["fastpath"]["fresh"] is False      # the next recall rebuilds


# ── (e) the config switch ───────────────────────────────────────────────────

def test_the_config_switch_really_switches_it_off(tmp_path):
    s = a_store(tmp_path)
    d = recall(s, None, "ssd-tier-mission?", hops=0, fastpath_cfg={"enabled": False})
    assert d["how"].startswith("words")                  # back to the old behaviour
    assert d["fastpath_verdict"] == "disabled"
    assert d["fastpath_ms"] is None


def test_fastpath_config_loads_and_overrides_per_store(tmp_path):
    for n in ("maker", "eq"):
        Store(name=n, path=str(tmp_path / n)).init_files()
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[fastpath]
enabled = true
gate = 1.2
[stores.maker]
path = "{tmp_path / 'maker'}"
[stores.eq]
path = "{tmp_path / 'eq'}"
[stores.eq.fastpath]
enabled = false
""", encoding="utf-8")
    reg = Registry.load(str(cfg))
    assert reg.fastpath_cfg_for(reg.stores["maker"]) == {"enabled": True, "gate": 1.2}
    eq = reg.fastpath_cfg_for(reg.stores["eq"])
    assert eq["enabled"] is False and eq["gate"] == 1.2  # override merges, not replaces


def test_a_wrong_fastpath_type_fails_at_load(tmp_path):
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[fastpath]
gate = "high"
[stores.main]
path = "{tmp_path / 'main'}"
""", encoding="utf-8")
    try:
        Registry.load(str(cfg))
    except ValueError as e:
        assert "gate" in str(e)
    else:
        raise AssertionError("a wrong type must not load")


def test_a_grouped_index_line_gives_every_slug_on_it_the_same_hook(tmp_path):
    """`LINK.sub` — the substitution that turns a line into its prose — had no test
    with more than one link on the line, which is the only case it exists for."""
    s = a_store(tmp_path)
    with open(s.index_path, "a", encoding="utf-8") as f:
        f.write("- optane family — [A](ssd-tier-mission.md)/[B](cooling.md) — the D-type board\n")
    titles, hooks = fastpath._index_lines(s)
    assert titles["ssd-tier-mission"] and titles["cooling"]
    for slug in ("ssd-tier-mission", "cooling"):
        assert "optane family" in hooks[slug] and "D-type board" in hooks[slug]
        assert "](" not in hooks[slug]          # the links themselves are gone
