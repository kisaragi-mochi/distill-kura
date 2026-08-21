"""The gate — the floor under every model in the system.

If these tests pass, a model cannot get an invented fact into the store no matter how
confidently it words it. That is the whole claim of this project, so it is tested
adversarially: each case is a way a real model actually tried to get something through.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.distill.gate import attributes_to_human, gate, salvage   # noqa: E402
from distill_kura.distill.sources import Segment                           # noqa: E402

SEGS = [
    Segment("USER", "let's move the index to a separate file, the current one is too big"),
    Segment("TOOL", "elapsed 4.21s, 118 memories, 402 links resolved"),
    Segment("ACT", "Bash grep -c '' MEMORY.md"),
    Segment("SELF", "I think the index will need pruning before long"),
]


def test_verbatim_quote_survives():
    kept, dropped, ideas = gate([{
        "topic": "index-split", "kind": "project", "why": "a decision about the index",
        "quotes": ["[USER] let's move the index to a separate file"],
    }], SEGS)
    assert len(kept) == 1 and not dropped
    assert kept[0]["classes"] == ["USER"]


def test_fabricated_quote_is_dropped():
    kept, dropped, _ = gate([{
        "topic": "invented", "kind": "project", "why": "sounds plausible",
        "quotes": ["[USER] we agreed to rewrite the whole thing in Rust"],
    }], SEGS)
    assert not kept
    assert dropped[0]["why_dropped"] == "quotes not found in the raw material"


def test_paraphrase_is_dropped_even_when_faithful():
    """A paraphrase may be perfectly accurate — it still cannot be checked, so it is
    treated exactly like an invention."""
    kept, dropped, _ = gate([{
        "topic": "paraphrase", "kind": "project", "why": "same meaning, different words",
        "quotes": ["[USER] the user wants the index moved out into its own file"],
    }], SEGS)
    assert not kept and dropped


def test_agent_prose_alone_cannot_become_a_fact():
    kept, dropped, _ = gate([{
        "topic": "pruning-needed", "kind": "project", "why": "the index needs pruning",
        "quotes": ["[SELF] I think the index will need pruning before long"],
    }], SEGS)
    assert not kept
    assert dropped[0]["why_dropped"] == "turning the agent's own words into a fact"


def test_agent_prose_survives_when_it_names_itself_a_judgement():
    kept, _, _ = gate([{
        "topic": "pruning-needed", "kind": "feedback",
        "why": "my judgement: the index will need pruning",
        "quotes": ["[SELF] I think the index will need pruning before long"],
    }], SEGS)
    assert len(kept) == 1 and kept[0]["judgement"] is True


def test_number_without_tool_backing_is_flagged():
    kept, _, _ = gate([{
        "topic": "402-links", "kind": "project", "why": "there are 402 links now",
        "quotes": ["[USER] let's move the index to a separate file"],
    }], SEGS)
    assert kept[0]["unverified_numbers"] is True


def test_number_with_tool_backing_is_grounded():
    kept, _, _ = gate([{
        "topic": "402-links", "kind": "project", "why": "there are 402 links now",
        "quotes": ["[TOOL] elapsed 4.21s, 118 memories, 402 links resolved"],
    }], SEGS)
    assert kept[0]["unverified_numbers"] is False
    assert "TOOL" in kept[0]["classes"]


def test_quote_already_in_the_store_is_an_echo_not_new_material():
    """A tool result that read the store back is not a discovery. Without this, a store
    re-finds and re-records its own contents forever."""
    store_text = "elapsed 4.21s, 118 memories, 402 links resolved"
    kept, dropped, _ = gate([{
        "topic": "echo", "kind": "project", "why": "counts",
        "quotes": ["[TOOL] elapsed 4.21s, 118 memories, 402 links resolved"],
    }], SEGS, store_text)
    assert not kept
    assert dropped[0]["why_dropped"] == "echo of text already in the store"


def test_ideas_need_no_quotes():
    _, _, ideas = gate([{
        "topic": "try-a-bloom-filter", "kind": "idea",
        "why": "a bloom filter might shortcut the resolve step", "quotes": [],
    }], SEGS)
    assert len(ideas) == 1


def test_a_factual_report_cannot_hide_inside_an_idea():
    """The one hole in the idea hatch, found in the wild: a factual claim with no quotes
    relabelled `kind: idea` to skip verification."""
    _, dropped, ideas = gate([{
        "topic": "approval", "kind": "idea",
        "why": "the user approved moving the index into its own file", "quotes": [],
    }], SEGS)
    assert not ideas
    assert dropped[0]["why_dropped"] == "a factual report dressed as an idea"


def test_class_tag_is_corrected_to_where_the_text_really_is():
    """A quote labelled [USER] that actually lives in tool output is re-filed, not
    trusted — otherwise mislabelling would launder a machine line into a human decision."""
    kept, _, _ = gate([{
        "topic": "mislabelled", "kind": "project", "why": "counts",
        "quotes": ["[USER] elapsed 4.21s, 118 memories, 402 links resolved"],
    }], SEGS)
    assert kept[0]["classes"] == ["TOOL"]


def test_too_short_quotes_do_not_count():
    kept, dropped, _ = gate([{
        "topic": "tiny", "kind": "project", "why": "x", "quotes": ["[USER] the"],
    }], SEGS)
    assert not kept and dropped


def test_salvage_recovers_objects_from_a_truncated_array():
    raw = ('[{"topic":"a","kind":"project","why":"one","quotes":["[USER] x"]},'
           '{"topic":"b","kind":"project","why":"two","quotes":["[USER] y')
    got = salvage(raw)
    assert len(got) == 1 and got[0]["topic"] == "a"


def test_salvage_ignores_braces_inside_strings():
    raw = '[{"topic":"a","why":"contains a } brace","quotes":["[USER] x"]}]'
    assert len(salvage(raw)) == 1


def test_attribution_check_is_mechanical():
    assert attributes_to_human("the user decided to drop it", [])
    assert attributes_to_human("ケンが決めた", [])
    assert not attributes_to_human("the user decided to drop it", ["USER"])
    assert not attributes_to_human("the index was moved", [])
