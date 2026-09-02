"""Tags and annotations: words on a memory, never weights.

One store owns a memory; several tags describe it. Nothing here ranks, counts or
scores — and these tests pin that down from the file up: the same set of tags always
renders to the same bytes, merging nothing new touches nothing on disk, and a tag that
cannot be read is reported rather than silently treated as 'untagged'.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.store import (ANNOTATION_KEYS, RESERVED_TAGS, InvalidTag,   # noqa: E402
                                Store, normalize_tags)


def make(tmp_path, name="t", policy="direct-allowed") -> Store:
    s = Store(name=name, path=str(tmp_path / name), label=name, write_policy=policy)
    s.init_files()
    return s


# ── normalisation ────────────────────────────────────────────────────────

def test_tags_dedupe_and_order_deterministically():
    assert normalize_tags(["landmine", "decision", "landmine"]) == ("decision", "landmine")
    assert normalize_tags(("decision", "landmine")) == normalize_tags(["landmine", "decision"])
    assert normalize_tags('["b", "a", "b"]') == ("a", "b")       # a JSON array in a string
    assert normalize_tags("a, b b") == ("a", "b")                 # a loose list
    assert normalize_tags(None) == () and normalize_tags([]) == ()


@pytest.mark.parametrize("bad", ["Landmine", "land mine", "-lead", "", "x" * 41, "日本語"])
def test_a_tag_that_is_not_a_kebab_word_is_refused_not_dropped(bad):
    with pytest.raises(InvalidTag):
        normalize_tags([bad])


def test_a_non_string_tag_is_refused():
    with pytest.raises(InvalidTag):
        normalize_tags([3])


def test_the_reserved_words_are_themselves_valid_tags():
    assert normalize_tags(sorted(RESERVED_TAGS)) == tuple(sorted(RESERVED_TAGS))


# ── writing and reading back ─────────────────────────────────────────────

def test_tags_and_annotations_round_trip_through_the_file(tmp_path):
    s = make(tmp_path)
    r = s.remember_direct("black-screen", "recovery order after a GPU change", "body",
                          tags=["landmine", "implementation", "landmine"],
                          annotations={"belongs_because": "so the next outage is cut in the same order",
                                       "keep": "the order and the success condition",
                                       "may_fade": "the chat order of that afternoon"})
    assert r["ok"]
    assert s.tags("black-screen") == ("implementation", "landmine")
    assert s.annotations("black-screen") == {
        "belongs_because": "so the next outage is cut in the same order",
        "keep": "the order and the success condition",
        "may_fade": "the chat order of that afternoon"}
    text = s.read_exact("black-screen")
    # The shape the spec shows: tags under metadata as a JSON array, the three
    # sentences at the top level, body untouched.
    assert '  tags: ["implementation", "landmine"]' in text
    assert "\nbelongs_because: so the next" in text
    assert text.rstrip("\n").endswith("---\n\nbody")
    assert s.frontmatter("black-screen")["type"] == "project"


def test_a_memory_without_tags_reads_as_untagged_and_is_not_an_error(tmp_path):
    """Every memory written before today has no tags line at all."""
    s = make(tmp_path)
    s.remember_direct("old", "an old one", "body")
    assert s.tags("old") == ()
    assert s.annotations("old") == {}
    assert s.tag_problems("old") is None
    assert "tags:" not in s.read_exact("old")


def test_rewriting_the_body_keeps_tags_and_annotations(tmp_path):
    """A caller that does not mention tags has not asked to remove them."""
    s = make(tmp_path)
    s.remember_direct("m", "d", "v1", tags=["decision"], annotations={"keep": "the decision"})
    s.remember_direct("m", "d2", "v2")
    assert s.read("m").rstrip("\n").endswith("v2")
    assert s.tags("m") == ("decision",)
    assert s.annotations("m") == {"keep": "the decision"}
    # and tags given on a rewrite are a union — nothing is silently dropped
    s.remember_direct("m", "d3", "v3", tags=["landmine"])
    assert s.tags("m") == ("decision", "landmine")


def test_a_bad_tag_refuses_the_whole_write_before_anything_is_touched(tmp_path):
    s = make(tmp_path)
    r = s.remember_direct("m", "d", "b", tags=["Fine", "ok"])
    assert not r["ok"] and "Fine" in r["error"]
    assert not os.path.exists(s.file_of("m"))
    assert "(m.md)" not in s.index_text()


def test_an_unknown_annotation_key_is_refused(tmp_path):
    s = make(tmp_path)
    r = s.remember_direct("m", "d", "b", annotations={"importance": "high"})
    assert not r["ok"] and "importance" in r["error"]
    assert set(ANNOTATION_KEYS) == {"belongs_because", "keep", "may_fade"}


def test_the_annotate_door_also_refuses_an_unknown_annotation_key(tmp_path):
    """Both doors carry their own copy of the key check; only the write door was
    tested, so a slip on the annotate side would have been silent."""
    s = make(tmp_path)
    s.remember_direct("m", "d", "b")
    before = open(s.file_of("m"), encoding="utf-8").read()
    r = s.annotate_direct("m", annotations={"importance": "high"})
    assert not r["ok"] and "importance" in r["error"]
    assert open(s.file_of("m"), encoding="utf-8").read() == before


# ── annotating in place ──────────────────────────────────────────────────

def test_annotate_merges_tags_and_a_second_identical_merge_touches_nothing(tmp_path):
    s = make(tmp_path)
    s.remember_direct("m", "d", "b", tags=["decision"])
    r1 = s.annotate_direct("m", tags=["recurred"])
    assert r1["ok"] and r1["changed"] and r1["tags"] == ["decision", "recurred"]
    before = open(s.file_of("m"), "rb").read()
    st = os.stat(s.file_of("m"))
    os.utime(s.file_of("m"), (st.st_atime, st.st_mtime - 100))   # make any rewrite visible
    m0 = os.stat(s.file_of("m")).st_mtime
    r2 = s.annotate_direct("m", tags=["recurred"])
    r3 = s.annotate_direct("m", tags=["recurred", "decision"])
    assert r2["ok"] and not r2["changed"]
    assert r3["ok"] and not r3["changed"]
    assert open(s.file_of("m"), "rb").read() == before
    assert os.stat(s.file_of("m")).st_mtime == m0            # not even the mtime moved
    assert s.tags("m") == ("decision", "recurred")            # one `recurred`, no count


def test_annotate_does_not_touch_body_description_or_index(tmp_path):
    s = make(tmp_path)
    s.remember_direct("m", "the trigger", "the body", title="M")
    idx = s.index_text()
    s.annotate_direct("m", tags=["landmine"], annotations={"keep": "the body"})
    assert s.read("m").rstrip("\n").endswith("the body")
    assert s.frontmatter("m")["description"] == "the trigger"
    assert s.index_text() == idx


def test_annotate_is_exact_and_never_decorates_a_neighbour(tmp_path):
    s = make(tmp_path)
    s.remember_direct("ssd-tier-mission", "d", "b")
    r = s.annotate_direct("ssd-tier-inference-mission", tags=["landmine"])   # fuzzy-close
    assert not r["ok"]
    assert s.tags("ssd-tier-mission") == ()


def test_annotation_write_authority_follows_the_store_policy(tmp_path):
    """direct-allowed: both doors. distiller-only: only the verified door. frozen: neither.
    A tool that could write `entrusted` on a distiller-only store could immortalise
    anything it liked."""
    cases = {"direct-allowed": (True, True), "distiller-only": (False, True), "frozen": (False, False)}
    for policy, (direct, verified) in cases.items():
        s = make(tmp_path, name=policy, policy="direct-allowed")
        s.remember_direct("m", "d", "b")          # seed, then apply the policy under test
        s.write_policy = policy
        rd = s.annotate_direct("m", tags=["entrusted"])
        rv = s.annotate_verified("m", tags=["formative"])
        assert rd["ok"] is direct, (policy, rd)
        assert rv["ok"] is verified, (policy, rv)
        want = set()
        if direct:
            want.add("entrusted")
        if verified:
            want.add("formative")
        assert set(s.tags("m")) == want, policy


def test_annotate_verified_can_carry_provenance_into_metadata(tmp_path):
    s = make(tmp_path, policy="distiller-only")
    s.pour_verified("m", "d", "b")
    s.annotate_verified("m", tags=["recurred"], meta={"recurred_manifest": "sha256:abc"})
    assert s.frontmatter("m")["recurred_manifest"] == "sha256:abc"
    assert s.tags("m") == ("recurred",)


# ── doctor ───────────────────────────────────────────────────────────────

def test_doctor_names_a_tags_line_it_cannot_read_instead_of_calling_it_untagged(tmp_path):
    s = make(tmp_path)
    s.remember_direct("m", "d", "b")
    p = s.file_of("m")
    t = open(p, encoding="utf-8").read().replace("  type: project\n",
                                                 "  type: project\n  tags: [\"Bad Tag\"]\n")
    open(p, "w", encoding="utf-8").write(t)
    assert s.tags("m") == ()                      # a reader gets the honest empty answer
    d = s.doctor()
    assert "m" in d["invalid_tags"] and "Bad Tag" in d["invalid_tags"]["m"]


def test_doctor_reports_a_manifest_the_memory_points_at_but_which_is_gone(tmp_path):
    s = make(tmp_path)
    s.pour_verified("m", "d", "b", meta={"evidence_manifest": "sha256:" + "0" * 64})
    assert s.doctor()["missing_manifest"] == ["m"]
    os.makedirs(os.path.join(s.path, "_evidence"))
    open(os.path.join(s.path, "_evidence", "0" * 64 + ".json"), "w").write("{}")
    assert s.doctor()["missing_manifest"] == []


def test_doctor_reports_a_manifest_pointer_that_is_not_even_a_digest(tmp_path):
    """A pointer that is not `sha256:<64 hex>` can never be looked up, so it is neither
    missing nor tampered — it has its own doctor key, and nothing exercised it."""
    s = make(tmp_path)
    s.pour_verified("m", "d", "b", meta={"evidence_manifest": "not-a-digest"})
    d = s.doctor()
    assert d["invalid_manifest_pointer"] == ["m"]
    assert d["missing_manifest"] == [] and d["tampered_manifest"] == []


def test_doctor_observes_capacity_in_four_units_and_decides_nothing(tmp_path):
    """The unit a shelf is measured in has not been chosen. Reporting all four keeps
    that choice open; `limit` and `pressure` stay None until a person sets them."""
    s = make(tmp_path)
    s.remember_direct("a", "d", "x" * 100, tags=["decision"])
    s.remember_direct("b", "d", "y" * 100)
    cap = s.doctor()["capacity"]
    assert cap["memories"] == 2
    assert cap["body_tokens_est"] > 0 and cap["index_tokens_est"] > 0 and cap["bytes"] > 200
    assert cap["unit"] is None and cap["limit"] is None and cap["pressure"] is None
    assert s.doctor()["tagged"] == 1
    # Nothing was changed by observing.
    assert s.slugs() == ["a", "b"]
    assert s.tags("a") == ("decision",) and s.tags("b") == ()


def test_a_tagged_memory_is_byte_stable_across_a_read_write_cycle(tmp_path):
    """Two writers that agree on the parts produce the same file. Otherwise a merge
    that changed nothing would still show up as a change."""
    s = make(tmp_path)
    s.remember_direct("m", "d", "b", tags=["b-tag", "a-tag"],
                      annotations={"may_fade": "details"})
    first = open(s.file_of("m"), "rb").read()
    s.annotate_direct("m", tags=["a-tag"], annotations={"may_fade": "details"})
    assert open(s.file_of("m"), "rb").read() == first
    # The file is legible as the JSON it claims to be.
    line = [l for l in first.decode().splitlines() if l.strip().startswith("tags:")][0]
    assert json.loads(line.split(":", 1)[1]) == ["a-tag", "b-tag"]


def test_an_unterminated_frontmatter_block_is_no_frontmatter_at_all(tmp_path):
    """`_frontmatter_of` and `_split` must agree on where the block ends — they used
    to carry two copies of the rule. An opening `---` with no closing one is the edge
    that would show a disagreement, and nothing exercised it."""
    s = make(tmp_path)
    s.remember_direct("m", "d", "b")
    torn = "---\nname: m\ndescription: d\nno closing fence here\n"
    with open(s.file_of("m"), "w", encoding="utf-8") as f:
        f.write(torn)
    assert s.frontmatter("m") == {}
    assert Store._split(torn) == ("", torn)
