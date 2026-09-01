"""Glance — mechanical micro-recall, exact and honest (plan §15, Glance block).

Every test here is one of the ways a glance could lie: reading a neighbour for a
misspelt name, showing a tampered KEEP sentence as authority, or leaking a link
out of the store. No model is needed anywhere.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.glance import glance                     # noqa: E402
from distill_kura.store import Store                       # noqa: E402


def build(tmp_path) -> Store:
    s = Store(name="m", path=str(tmp_path / "m"), label="k")
    s.init_files()
    s.remember_direct("freetoken-hybrid",
                      "FreeToken CPU hybrid — the all-hands cooperative inference push",
                      "The hybrid runs the experts on CPU.\n\nRelated: [[exl3-quantization]] "
                      "and [[../outside/private]] and [[a-memory-that-does-not-exist]].",
                      title="FreeToken CPU hybrid")
    s.remember_direct("exl3-quantization", "EXL3 quantization — what the CPU side eats",
                      "body")
    return s


def test_glance_reads_an_exact_slug(tmp_path):
    g = glance(build(tmp_path), "freetoken-hybrid")
    assert g["ok"] and g["slug"] == "freetoken-hybrid"
    assert g["title"] == "FreeToken CPU hybrid"
    assert "all-hands" in g["trigger"]
    assert "[freetoken-hybrid]" in g["text"]


def test_glance_refuses_every_fuzzy_shape(tmp_path):
    s = build(tmp_path)
    for bad in ("freetoken-hyrbid",              # the classic model misspelling
                "Freetoken-Hybrid",              # case
                "the freetoken hybrid",          # a title, not a name
                "../freetoken-hybrid",           # a path
                "freetoken"):                    # a prefix
        g = glance(s, bad)
        assert g["ok"] is False and "no memory" in g["error"], bad


def test_glance_accepts_the_md_suffix_as_shape_not_fuzz(tmp_path):
    """`.md` is the shape models answer in, stripped everywhere in this codebase —
    normalisation, not fuzzy matching. The name is still exact."""
    assert glance(build(tmp_path), "freetoken-hybrid.md")["ok"] is True


def test_an_unknown_slug_is_an_honest_nothing(tmp_path):
    g = glance(build(tmp_path), "never-heard-of-it")
    assert g["ok"] is False
    assert "kura_recall" in g["error"], "the error says where to find the name"


def test_a_verified_keep_is_shown(tmp_path):
    s = build(tmp_path)
    s.annotate_verified("freetoken-hybrid",
                        annotations={"keep": "CPU is a member of the team, not an offload target"})
    g = glance(s, "freetoken-hybrid")
    assert g["keep_state"] == "verified"
    assert g["keep"] == "CPU is a member of the team, not an offload target"
    assert "KEEP:" in g["text"] and "CPU is a member" in g["text"]


def test_a_tampered_keep_is_never_authority(tmp_path):
    """Hand-edit the KEEP sentence after the distiller signed it and the mark no
    longer matches: the sentence is REFUSED as authority, not shown."""
    s = build(tmp_path)
    s.annotate_verified("freetoken-hybrid", annotations={"keep": "the original signed sentence"})
    f = s.file_of("freetoken-hybrid")
    text = open(f, encoding="utf-8").read()
    with open(f, "w", encoding="utf-8") as fh:
        fh.write(text.replace("the original signed sentence", "A SENTENCE NOBODY SIGNED"))
    g = glance(s, "freetoken-hybrid")
    assert g["keep_state"] == "tampered"
    assert g["keep"] is None and "KEEP:" not in g["text"]
    assert "NOBODY SIGNED" not in g["text"]


def test_an_unsigned_keep_is_not_shown_either(tmp_path):
    s = build(tmp_path)
    s.annotate_direct("freetoken-hybrid", annotations={"keep": "said by no one verified"})
    g = glance(s, "freetoken-hybrid")
    assert g["keep_state"] == "unsigned"
    assert g["keep"] is None and "KEEP:" not in g["text"]


def test_links_cannot_leave_the_store(tmp_path):
    g = glance(build(tmp_path), "freetoken-hybrid")
    assert g["links"] == ["exl3-quantization"], \
        "a dead link and a path link are simply not links"


def test_glance_needs_no_model(tmp_path):
    """The whole point: a glance is strings the store already has. This test
    constructs no endpoint, no server, nothing — and the glance answers."""
    g = glance(build(tmp_path), "exl3-quantization")
    assert g["ok"] and g["relations"] == []
