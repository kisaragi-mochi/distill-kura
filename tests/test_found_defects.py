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
    for h, key in (("a" * 64, "claude:first.jsonl"), ("b" * 64, "claude:second.jsonl")):
        open(os.path.join(d._evidence_dir(), h + ".json"), "w").write(json.dumps({"source_key": key}))
    s.pour_verified("x", "d", "body", meta={"evidence_manifest": "sha256:" + "a" * 64})
    s.pour_verified("x", "d", "body\n\nmore", meta={"evidence_manifest": "sha256:" + "b" * 64})
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
