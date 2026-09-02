"""Defects found by the 2026-08-22 design review, each written as the lie it lets through.

Every test here failed before its fix. They stay as the regression floor.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.distill import Distiller                    # noqa: E402
from distill_kura.distill.gate import verify_tags             # noqa: E402
from distill_kura.distill.sources import ClaudeCodeSource    # noqa: E402
from distill_kura.registry import Registry                    # noqa: E402
from distill_kura.store import Store                          # noqa: E402
from distill_kura.thinker import Models                       # noqa: E402


# ── 1. a subagent's transcript is not the human speaking ─────────────────

def test_a_sidechain_user_line_is_the_parent_models_prompt_not_the_human(tmp_path):
    """In Claude Code, a subagent's transcript records the PARENT MODEL's instructions
    as `type: user` with `isSidechain: true`. Classed as [USER], a model's sentence
    "the owner approved the migration" becomes the human's own words — the exact
    laundering the evidence classes exist to prevent. 360 of the house's 391 journal
    files are sidechains (measured 2026-08-23)."""
    p = tmp_path / "agent-x.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "user", "isSidechain": True, "message": {"content": [
            {"type": "text", "text": "the owner approved the migration to the slow disk"}]}}) + "\n")
        f.write(json.dumps({"type": "assistant", "isSidechain": True, "message": {"content": [
            {"type": "text", "text": "I looked and found the config points at /data"}]}}) + "\n")
        f.write(json.dumps({"type": "user", "message": {"content": [
            {"type": "text", "text": "a real human line in the same file"}]}}) + "\n")
    segs, _ = ClaudeCodeSource().sip(str(p), 0, 10_000)
    by = {s.text: s.cls for s in segs}
    assert by["the owner approved the migration to the slow disk"] == "SELF"   # model-written
    assert by["I looked and found the config points at /data"] == "SELF"
    assert by["a real human line in the same file"] == "USER"                   # untouched


# ── 2. tidy writes the index under the lock, atomically ──────────────────

def test_tidy_replaces_the_index_atomically_under_the_lock(tmp_path, monkeypatch):
    store = Store(name="m", path=str(tmp_path / "m")); store.init_files()
    store.remember_direct("a-very-long-slug-name-for-this-memory-x", "tiny", "the body")   # trigger too short → ragged
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    reg = Registry(stores={"m": store}, modes={}, models=models, default="m", raw={})
    d = Distiller(reg, store)
    d.scribe = lambda task, u, max_tokens=0: "TITLE: Things\nDESC: a proper trigger about the body"  # type: ignore
    seen = {}
    real = store._write_index

    def spy(text):
        seen["via_write_index"] = True
        return real(text)
    monkeypatch.setattr(store, "_write_index", spy)
    r = d.tidy()
    assert r["fixed"] == 1
    assert seen.get("via_write_index"), "tidy wrote the index with a bare open(), outside the atomic path"
    assert not [f for f in os.listdir(store.path) if ".tmp." in f]
    assert "a proper trigger about the body" in store.index_text()


# ── 3. an EXTENDS pour keeps the memory's first manifest ─────────────────

def test_extending_a_memory_keeps_its_origin_manifest(tmp_path):
    """`meta` on a pour overrides frontmatter keys. An EXTENDS pour wrote a new
    `evidence_manifest`, so the first one — the memory's ORIGIN, which `recur()` needs
    to decide 'another occasion' — was gone after the first extension."""
    s = Store(name="m", path=str(tmp_path / "m"), write_policy="distiller-only"); s.init_files()
    s.pour_verified("x", "d", "first body", meta={"evidence_manifest": "sha256:" + "a" * 64})
    s.pour_verified("x", "d", "first body\n\nmore", meta={"evidence_manifest": "sha256:" + "b" * 64})
    fm = s.frontmatter("x")
    assert fm["evidence_manifest"] == "sha256:" + "b" * 64          # latest evidence
    assert fm["origin_manifest"] == "sha256:" + "a" * 64            # the origin survives
    s.pour_verified("x", "d", "first body\n\nmore\n\nagain", meta={"evidence_manifest": "sha256:" + "c" * 64})
    assert s.frontmatter("x")["origin_manifest"] == "sha256:" + "a" * 64   # and is never overwritten


def test_recur_reads_the_origin_not_the_latest_extension(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m"), write_policy="distiller-only"); s.init_files()
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    reg = Registry(stores={"m": s}, modes={}, models=models, default="m", raw={})
    d = Distiller(reg, s)
    os.makedirs(d._evidence_dir(), exist_ok=True)
    # Content-addressed for real: the verified loader re-hashes the bytes, so a
    # fixture manifest must be named by its own hash like a genuine one.
    import hashlib as _hl
    refs = {}
    for key in ("claude:first.jsonl", "claude:second.jsonl"):
        blob = json.dumps({"source_key": key})
        h = _hl.sha256(blob.encode()).hexdigest()
        open(os.path.join(d._evidence_dir(), h + ".json"), "w").write(blob)
        refs[key] = h
    s.pour_verified("x", "d", "body", meta={"evidence_manifest": "sha256:" + refs["claude:first.jsonl"]})
    s.pour_verified("x", "d", "body\n\nmore", meta={"evidence_manifest": "sha256:" + refs["claude:second.jsonl"]})
    assert d._origin_key("x") == "claude:first.jsonl"


# ── 4. a profile draft reads the memories before the study shelf ─────────

def test_profile_draft_reads_memories_before_study_notes(tmp_path):
    """`slugs()` sorts `_study/...` first (underscore sorts before letters), so a
    store with a few long notes spent the whole reading budget on the shelf and the
    draft never saw a single memory."""
    s = Store(name="u", path=str(tmp_path / "u")); s.init_files()
    os.makedirs(os.path.join(s.path, "_study"))
    open(os.path.join(s.path, "_study", "big-note.md"), "w", encoding="utf-8").write(
        "---\nname: big-note\ndescription: d\nmetadata:\n  type: reference\n---\n\n" + "x" * 70_000)
    s.remember_direct("small-memory", "the thing they return to", "they keep returning to memory design")
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    reg = Registry(stores={"u": s}, modes={}, models=models, default="u", raw={})
    d = Distiller(reg, s)
    seen = {}

    def brain(task, user, max_tokens=0):
        seen["user"] = user
        return "## Enduring threads\nmemory design.\n"
    d.brain = brain   # type: ignore
    assert d.profile_draft()["ok"]
    assert "=== small-memory ===" in seen["user"]


# ── 5. an index reference in prose is not a memory ───────────────────────

def test_known_slugs_only_counts_link_targets(tmp_path):
    """`作法(AGENTS.md)` in an index line's prose matched `\\(([^)]+)\\.md\\)`, so doctor
    reported an index orphan called AGENTS that was never a memory."""
    s = Store(name="m", path=str(tmp_path / "m")); s.init_files()
    s.remember_direct("real", "d", "b", title="Real")
    with open(s.index_path, "a", encoding="utf-8") as f:
        f.write("- [Forge](real.md) — read the craft (ORDER_CRAFT.md) and the rules(AGENTS.md) first\n")
    assert set(s.known_slugs()) == {"real"}
    assert s.doctor()["index_orphans"] == []


# ── 6. commitment is a claim about the human and needs their words ───────

def test_commitment_needs_the_humans_words():
    tool_only = [{"class": "TOOL", "text": "/data 3.2T used 1.1T avail"}]
    kept, _, refused = verify_tags(["commitment"], tool_only)
    assert kept == () and "commitment" in refused
    kept, basis, _ = verify_tags(["commitment"], [{"class": "USER", "text": "I will move the archive tonight"}])
    assert kept == ("commitment",) and basis["commitment"]["class"] == "USER"


# ── 7. landmine needs an actual failure, danger, correction or warning ───

def test_landmine_needs_a_failure_in_the_evidence_not_just_a_tool_line():
    """The spec: a landmine rests on an actual failure, danger, correction or warning.
    A quiet `df` line is tool output and is not any of those — tagging it `landmine`
    turned every bake log into a minefield."""
    quiet = [{"class": "TOOL", "text": "/data 3.2T used 1.1T avail"}]
    kept, _, refused = verify_tags(["landmine"], quiet)
    assert kept == () and "landmine" in refused
    failed = [{"class": "TOOL", "text": "RPC call to sample_tokens timed out; engine dead"}]
    kept, basis, _ = verify_tags(["landmine"], failed)
    assert kept == ("landmine",) and basis["landmine"]["class"] == "TOOL"
    assert "timed out" in basis["landmine"]["quote"]
    jp = [{"class": "TOOL", "text": "CUDA graph capture で落ちた: out of memory"}]
    assert verify_tags(["landmine"], jp)[0] == ("landmine",)


def test_landmine_from_the_human_needs_a_warning_or_a_correction():
    plain = [{"class": "USER", "text": "put the archive on the slow disk"}]
    assert verify_tags(["landmine"], plain)[0] == ()
    warned = [{"class": "USER", "text": "⚠ NASから直接ロードしないで、OOMする"}]
    kept, basis, _ = verify_tags(["landmine"], warned)
    assert kept == ("landmine",) and basis["landmine"]["class"] == "USER"
    corrected = [{"class": "USER", "text": "いや、それは違う。x4縮退は接触で移動する"}]
    assert verify_tags(["landmine"], corrected)[0] == ("landmine",)
    en = [{"class": "USER", "text": "don't load from the NAS directly, it OOMs"}]
    assert verify_tags(["landmine"], en)[0] == ("landmine",)


def test_landmine_never_rests_on_the_agents_prose():
    self_only = [{"class": "SELF", "text": "I think this failed because of the driver"}]
    assert verify_tags(["landmine"], self_only)[0] == ()
    # an [ACT] alone is an action, not a failure
    act = [{"class": "ACT", "text": "Bash {\"command\": \"docker rm -f huihui\"}"}]
    assert verify_tags(["landmine"], act)[0] == ()


# ── 8. curation is signed on the verified door ───────────────────────────

def test_a_verified_pour_signs_its_tags_and_a_hand_edit_is_tampering(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m"), write_policy="distiller-only"); s.init_files()
    s.pour_verified("x", "d", "b", tags=["decision"], annotations={"keep": "the decision"})
    assert s.curation_state("x") == "verified"
    assert s.doctor()["curation"]["tampered"] == [] and s.doctor()["curation"]["verified"] == 1
    # someone edits the file by hand and promotes the memory to `entrusted`
    p = s.file_of("x")
    t = open(p, encoding="utf-8").read().replace('["decision"]', '["decision", "entrusted"]')
    open(p, "w", encoding="utf-8").write(t)
    assert s.tags("x") == ("decision", "entrusted")            # the reader still reads it…
    assert s.curation_state("x") == "tampered"                 # …and doctor names it
    assert s.doctor()["curation"]["tampered"] == ["x"]


def test_the_direct_door_writes_unsigned_and_that_is_normal_where_it_is_allowed(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m")); s.init_files()          # direct-allowed
    s.remember_direct("x", "d", "b", tags=["decision"])
    assert s.curation_state("x") == "unsigned"
    d = s.doctor()["curation"]
    assert d["unsigned"] == 1 and d["tampered"] == [] and d["unsigned_names"] == []   # not listed: allowed here
    # a verified annotation on top signs it; a later direct annotation drops the mark
    s.annotate_verified("x", tags=["landmine"])
    assert s.curation_state("x") == "verified"
    s.annotate_direct("x", tags=["reference"])
    assert s.curation_state("x") == "unsigned"
    assert s.doctor()["curation"]["tampered"] == []


def test_on_a_distiller_only_store_hand_written_tags_are_named(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m"), write_policy="distiller-only"); s.init_files()
    s.pour_verified("x", "d", "b")                              # no curation, no mark: fine
    assert s.curation_state("x") == "none"
    p = s.file_of("x")
    t = open(p, encoding="utf-8").read().replace("  type: project\n", '  type: project\n  tags: ["entrusted"]\n')
    open(p, "w", encoding="utf-8").write(t)
    assert s.curation_state("x") == "unsigned"
    assert s.doctor()["curation"]["unsigned_names"] == ["x"]   # listed: nobody but the gate should write here


def test_the_mark_is_stable_so_an_identical_merge_still_touches_nothing(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m"), write_policy="distiller-only"); s.init_files()
    s.pour_verified("x", "d", "b", tags=["decision"])
    before = open(s.file_of("x"), "rb").read()
    r = s.annotate_verified("x", tags=["decision"])
    assert not r["changed"] and open(s.file_of("x"), "rb").read() == before
    assert s.curation_state("x") == "verified"


def test_recurred_is_signed_too(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m"), write_policy="distiller-only"); s.init_files()
    s.pour_verified("x", "d", "b")
    s.annotate_verified("x", tags=["recurred"], meta={"recurred_manifest": "sha256:" + "f" * 64})
    assert s.curation_state("x") == "verified"
    assert s.frontmatter("x")["recurred_manifest"].startswith("sha256:")


# ── 9. an extension's heading carries the evidence's date, not an invented one ──

def test_extension_heading_date_is_the_journals_not_the_models(tmp_path):
    """30 of 39 extension headings in the house were dated before the distiller
    existed; one said 2025. The date now comes from the journal file's mtime and a
    heading that says otherwise is corrected mechanically."""
    import time
    from distill_kura.distill.pipeline import Distiller as D
    s = Store(name="m", path=str(tmp_path / "m"), write_policy="distiller-only"); s.init_files()
    s.pour_verified("x", "the slow disk", "The archive goes on the slow disk.")
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    reg = Registry(stores={"m": s}, modes={}, models=models, default="m", raw={})
    d = D(reg, s)
    src = tmp_path / "j.jsonl"; src.write_text("{}\n")
    t = time.mktime((2026, 8, 20, 12, 0, 0, 0, 0, -1)); os.utime(src, (t, t))
    d._current_source = str(src)
    d.scribe = lambda task, u, max_tokens=0: "SECTION: ## 2025-06-09 注ぎ手の判断記録\nBODY:\nnew fact\n"   # type: ignore
    c = {"extends": "x", "extends_why": "adds", "evidence": [{"class": "USER", "text": "move it tonight"}],
         "classes": ["USER"], "kind": "project"}
    out = d._compose_extension(c)
    assert out["body"].startswith("## 2026-08-20 注ぎ手の判断記録")
    assert "2025" not in out["body"]
    # a heading with no date at all gets the date put in front
    d.scribe = lambda task, u, max_tokens=0: "SECTION: ## the verdict log\nBODY:\nnew fact\n"   # type: ignore
    assert d._compose_extension(c)["body"].startswith("## 2026-08-20 the verdict log")


# ── round four: the index-line rewriter wears the floor ──────────────────────

def test_tidy_cannot_invent_a_number_into_the_index(tmp_path):
    store = Store(name="m", path=str(tmp_path / "m")); store.init_files()
    store.remember_direct("gpu-notes", "tiny", "the machine ran with its usual boards")
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    reg = Registry(stores={"m": store}, modes={}, models=models, default="m", raw={})
    d = Distiller(reg, store)
    d.scribe = lambda task, u, max_tokens=0: "TITLE: 99-GPU rig\nDESC: a proper trigger about the boards"  # type: ignore
    r = d.tidy()
    assert r["fixed"] == 0
    assert "99-GPU" not in store.index_text()


def test_tidy_may_cite_the_memorys_own_numbers(tmp_path):
    store = Store(name="m", path=str(tmp_path / "m")); store.init_files()
    store.remember_direct("gpu-notes", "tiny", "the machine ran with 12 boards")
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    reg = Registry(stores={"m": store}, modes={}, models=models, default="m", raw={})
    d = Distiller(reg, store)
    d.scribe = lambda task, u, max_tokens=0: "TITLE: 12-board rig\nDESC: a proper trigger about the boards"  # type: ignore
    r = d.tidy()
    assert r["fixed"] == 1
    assert "12-board rig" in store.index_text()


# ── round five: tidy merges under the lock instead of writing its snapshot ──

def test_tidy_does_not_erase_a_memory_poured_while_it_thought(tmp_path):
    store = Store(name="m", path=str(tmp_path / "m")); store.init_files()
    store.remember_direct("gpu-notes", "tiny", "the machine ran with 12 boards")
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    reg = Registry(stores={"m": store}, modes={}, models=models, default="m", raw={})
    d = Distiller(reg, store)

    def racing_scribe(task, u, max_tokens=0):
        # A pour lands while the model is composing the new line.
        store.remember_direct("landed-mid-tidy", "a proper trigger line about landing", "body two")
        return "TITLE: 12-board rig\nDESC: a proper trigger about the boards"
    d.scribe = racing_scribe  # type: ignore
    r = d.tidy()
    assert r["fixed"] == 1 and r["skipped_stale"] == 0
    idx = store.index_text()
    assert "landed-mid-tidy" in idx            # the racer survived
    assert "12-board rig" in idx               # and the tidy still landed


def test_doctor_audits_every_manifest_pointer(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m"), write_policy="distiller-only"); s.init_files()
    s.pour_verified("x", "a proper description line", "body",
                    meta={"origin_manifest": "sha256:" + "c" * 64})
    assert s.doctor()["missing_manifest"] == ["x"]


def test_verified_loader_defines_what_a_digest_is(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m")); s.init_files()
    assert s.load_manifest_verified("../../../etc/passwd") is None
    assert s.load_manifest_verified("deadbeef") is None
