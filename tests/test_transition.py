"""The verified transition relation — proposed ≠ proven.

Every test is named for the failure it prevents. The relation is pure and knows
nothing about the model's proposal on purpose: what writes `現在は [[new]]` into
canonical is the human's own explicit old → new sentence, or nothing.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.distill.transition import find_transition       # noqa: E402


def _t(text, old=("old-way", "the old way"), new=("new-way", "the new way"), topic=""):
    return find_transition([{"class": "USER", "text": text}],
                           {"slug": old[0], "title": old[1]},
                           {"slug": new[0], "title": new[1], "topic": topic})


def test_a_by_the_way_clause_never_supplies_the_successor():
    """The whole defect in one line: "old-way はやめよう。ところで別件で GPU 温度…"
    must never make the GPU memory old-way's successor."""
    r = _t("old-way はもうやめよう。ところで別件で GPU 温度の記録を取ろう",
           new=("gpu-temperature", "GPU 温度の記録"))
    assert r is not None and r["kind"] == "retired-only" and r["new"] is None


def test_naming_both_memories_without_a_construction_proves_nothing():
    """Two names in one sentence is a mention, not a ruling."""
    assert _t("old-way と new-way はどちらもよく使う") is None
    assert _t("old-way and new-way are both fine") is None


def test_the_explicit_forms_prove_succession_in_either_language():
    for text in ("old-way はやめて、今後は new-way で行く",
                 "old-way に代えて new-way を使う",
                 "old-way → new-way",
                 "we replaced old-way with new-way",
                 "stop using old-way, now use new-way",
                 "switch to new-way, old-way is done"):
        r = _t(text)
        assert r and r["kind"] == "superseded" and r["new"] == "new-way", text
        assert r["constructions"] and r["quote"] == text, text


def test_a_transition_is_never_stitched_across_two_quotes():
    r = find_transition([{"class": "USER", "text": "old-way はもうやめる"},
                         {"class": "USER", "text": "今後は new-way で行く"}],
                        {"slug": "old-way", "title": "the old way"},
                        {"slug": "new-way", "title": "the new way"})
    assert r is not None and r["kind"] == "retired-only"


def test_only_user_class_evidence_can_prove_a_transition():
    ev = [{"class": c, "text": "old-way はやめて、今後は new-way で行く"}
          for c in ("TOOL", "SELF", "ACT")]
    assert find_transition(ev, {"slug": "old-way", "title": "the old way"},
                           {"slug": "new-way", "title": "the new way"}) is None


def test_two_title_or_topic_words_can_stand_in_for_the_new_name():
    r = _t("old-way はやめて、今後は GPU 温度の記録を取る",
           new=("gpu-temperature", "GPU 温度の記録"))
    assert r and r["kind"] == "superseded"
    # one word alone is not a name
    assert _t("old-way はやめて、今後は温度を見る",
              new=("gpu-temperature", "GPU 温度の記録"))["kind"] == "retired-only"


def test_a_paraphrase_of_the_old_memory_is_not_its_name():
    """Exact slug or exact index title only — a model's paraphrase cannot retire."""
    assert _t("the way we used to do it is over, now use new-way") is None
