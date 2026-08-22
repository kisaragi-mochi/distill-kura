"""The five-room layout: chosen before the conversation, one room per memory, no
inheritance between rooms, no router, no move.

These tests load the shipped example config against temporary directories, so the
example stays a working example and the invariants it claims are the code's.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from distill_kura.distill import Distiller              # noqa: E402
from distill_kura.distill.prompts import DEFAULT_CHARTER  # noqa: E402
from distill_kura.registry import Registry              # noqa: E402
from distill_kura.store import Store                    # noqa: E402

ROOMS = ("research", "develop", "manage", "eq", "user")


def rooms_registry(tmp_path) -> Registry:
    src = open(os.path.join(ROOT, "examples", "rooms", "kura.rooms.example.toml"),
               encoding="utf-8").read()
    text = src.replace("~/kura/", str(tmp_path / "kura") + "/") \
              .replace("~/dsh/sessions/", str(tmp_path / "sessions") + "/") \
              .replace('charter = "examples/rooms/', f'charter = "{ROOT}/examples/rooms/') \
              .replace('url = "http://127.0.0.1:8011/v1"', 'url = "http://127.0.0.1:9/v1"')
    for r in ROOMS:
        Store(name=r, path=str(tmp_path / "kura" / r)).init_files()
        os.makedirs(tmp_path / "sessions" / f"work-{r}", exist_ok=True)
    cfg = tmp_path / "kura.toml"
    cfg.write_text(text, encoding="utf-8")
    return Registry.load(str(cfg))


def test_the_example_loads_with_five_rooms_and_five_selectors(tmp_path):
    reg = rooms_registry(tmp_path)
    assert set(reg.stores) == set(ROOMS)
    assert reg.modes == {r: r for r in ROOMS}
    for r in ROOMS:
        st = reg.store(r)
        assert st.write_policy == "distiller-only"
        assert st.charter and os.path.exists(st.charter)


def test_an_unknown_room_is_an_error_at_the_door_not_the_default(tmp_path):
    reg = rooms_registry(tmp_path)
    with pytest.raises(KeyError):
        reg.store("feelings")           # a plausible name that is not a room
    with pytest.raises(KeyError):
        reg.store("USER")               # selectors are lower-case on the wire


def test_every_room_drinks_from_its_own_journal_and_nothing_else(tmp_path):
    reg = rooms_registry(tmp_path)
    seen = {}
    for r in ROOMS:
        d = Distiller(reg, reg.store(r))
        assert list(d.journals) == ["dsh"], r
        root = d.journals["dsh"]
        assert root.endswith(f"work-{r}"), (r, root)
        seen[r] = root
    assert len(set(seen.values())) == len(ROOMS)          # five roots, none shared


def test_each_charter_starts_with_the_shared_evidence_spine():
    """A room's charter replaces the default one at the head of every prompt, so it
    must still carry the evidence classes — otherwise the room's distiller would
    never be told what [USER] and [TOOL] mean."""
    for r in ROOMS:
        t = open(os.path.join(ROOT, "examples", "rooms", r, "charter.md"), encoding="utf-8").read()
        assert t.startswith(DEFAULT_CHARTER.rstrip("\n")), r
        assert "## This room:" in t
        low = t.lower()
        for w in ("score", "salience", "importance"):
            assert w not in low, (r, w)


def test_the_same_topic_in_two_rooms_makes_two_memories_and_copies_nothing(tmp_path):
    """Research's 'what we learned' and Develop's 'what we did' are different facts.
    Nothing crosses the boundary to deduplicate them."""
    reg = rooms_registry(tmp_path)
    res, dev = reg.store("research"), reg.store("develop")
    res.pour_verified("slow-disk", "the archive on the slow disk", "we learned the slow disk is fine",
                      tags=["research-result"])
    dev.pour_verified("slow-disk", "the archive on the slow disk", "we moved the archive there",
                      tags=["implementation"])
    assert "learned" in res.read_exact("slow-disk") and "moved" not in res.read_exact("slow-disk")
    assert "moved" in dev.read_exact("slow-disk") and "learned" not in dev.read_exact("slow-disk")
    # a Develop memory may carry an EQ-flavoured tag and remains a Develop memory
    dev.annotate_verified("slow-disk", tags=["emotion-carried"])
    assert dev.tags("slow-disk") == ("emotion-carried", "implementation")
    assert reg.store("eq").slugs() == []
    # the mode table is the only map, and it is one-to-one
    assert all(reg.modes[r] == r for r in ROOMS)


def test_no_router_and_no_move_exist_in_the_codebase():
    """Spec 8.1: no path saves into another store without a chosen mode, and no
    move/copy function was added. Checked against the source, not a docstring."""
    src = ""
    for dirpath, _, files in os.walk(os.path.join(ROOT, "distill_kura")):
        for f in files:
            if f.endswith(".py"):
                src += open(os.path.join(dirpath, f), encoding="utf-8").read()
    for word in ("def move_memory", "def copy_memory", "def migrate", "def auto_route",
                 "def classify_mode", "importance_score", "salience_score", "priority_score",
                 "recurrence_count"):
        assert word not in src, word
    # read counts exist for diagnostics and are never consulted for a decision
    assert not re.search(r"read_counts\(\)[^\n]*\n[^\n]*(rank|sort|prior|keep|drop)", src)


def test_a_mode_switch_affects_future_sessions_only_and_moves_nothing(tmp_path):
    """Spec 8.4 #8, written as the act itself: a memory distilled while the mode was
    `develop` stays in develop after the mode table is pointed elsewhere. The only
    thing a switch changes is which store the NEXT request resolves to."""
    reg = rooms_registry(tmp_path)
    dev = reg.store("develop")
    dev.pour_verified("black-screen", "recovery order after a GPU change", "the order that worked",
                      tags=["landmine"])
    before = open(dev.file_of("black-screen"), "rb").read()
    # the host's selector `work` pointed at develop; it is re-bound to eq for future
    # sessions (a selector that IS a store name cannot be re-bound — by design)
    reg.modes["work"] = "develop"
    assert reg.store("work") is dev
    reg.modes["work"] = "eq"
    assert reg.store("work").name == "eq"                      # future requests go there
    assert reg.store("eq").slugs() == []                        # nothing travelled with the switch
    assert dev.slugs() == ["black-screen"]
    assert open(dev.file_of("black-screen"), "rb").read() == before
    assert dev.tags("black-screen") == ("landmine",)
