"""Write authority, at every door — not just the two obvious ones.

An adversarial pass over the first write-policy fix put it plainly: the policy was
"correctly wired into `remember_direct` and `pour_verified` and correctly refuses over
HTTP — and then `tidy()`, `Loom.persist()` and `init_files()` write into the same store
without ever asking". A memory was created in a frozen store, and another's body
destroyed, using nothing but the shipped `kura weave --no-model`.

Every test here is one of those doors, held shut.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest   # noqa: E402

from distill_kura.distill import Distiller                    # noqa: E402
from distill_kura.registry import Registry                    # noqa: E402
from distill_kura.store import Store                          # noqa: E402
from distill_kura.thinker import Models                       # noqa: E402
from distill_kura.weave import Loom                           # noqa: E402


def a_store(tmp_path, policy="direct-allowed", name="main"):
    s = Store(name=name, path=str(tmp_path / name))
    s.init_files()
    s.remember("d", "a description long enough to be trimmed by the mechanical trimmer",
               "ORIGINAL BODY: the real memory, which must survive")
    s.write_policy = policy
    return s


def a_distiller(store):
    reg = Registry(stores={store.name: store}, modes={},
                   models=Models.from_config({}), default=store.name)
    return Distiller(reg, store)


# ── the loom is a door too ──────────────────────────────────────────────────

def test_the_cloth_may_not_be_woven_into_a_memory_slot(tmp_path):
    """`cloth_path` pointed at a store-root `.md` ate that memory one weave at a time,
    while the stats block said `written: true` and looked perfectly healthy."""
    s = a_store(tmp_path)
    with pytest.raises(ValueError, match="memory slot"):
        Loom(s, scribe=None, out_path=s.file_of("d"))
    assert "ORIGINAL BODY" in s.read_exact("d")


def test_nothing_is_woven_inside_a_frozen_store(tmp_path):
    s = a_store(tmp_path, policy="frozen")
    with pytest.raises(ValueError, match="frozen"):
        Loom(s, scribe=None)
    assert not os.path.exists(os.path.join(s.still, "index.woven.md"))


def test_a_frozen_store_can_still_have_a_resident_map_kept_outside_it(tmp_path):
    """Refusing to write inside an archive must not mean the archive cannot be read."""
    s = a_store(tmp_path, policy="frozen")
    out = str(tmp_path / "outside-cloth.md")
    loom = Loom(s, scribe=None, out_path=out)
    assert loom.write()["written"] is True
    assert os.path.exists(out)


def test_tidy_does_not_rewrite_a_frozen_index(tmp_path):
    """The index is read on every recall and every prefill, and tidy is the only path
    that puts MODEL-authored prose into it."""
    s = a_store(tmp_path, policy="frozen")
    with open(s.index_path, "a", encoding="utf-8") as f:
        f.write("- [a title that is definitely longer than forty characters here](d.md) — t\n")
    d = a_distiller(s)
    d.scribe = lambda *a, **k: "TITLE: PWNED\nDESC: rewritten by a model"
    r = d.tidy(6)
    assert r["ok"] is False and "frozen" in r["why"]
    assert "PWNED" not in s.index_text()


def test_init_files_does_not_create_an_index_in_a_frozen_store(tmp_path):
    s = Store(name="arch", path=str(tmp_path / "arch"), write_policy="frozen")
    s.init_files()
    assert not os.path.exists(s.index_path)


# ── the pour ────────────────────────────────────────────────────────────────

def test_a_draft_is_named_by_a_slug_not_a_path(tmp_path):
    """`pour '../../../out/o'` read a file from anywhere on the filesystem into the
    store and renamed the original."""
    s = a_store(tmp_path, policy="distiller-only")
    outside = tmp_path / "out"
    outside.mkdir()
    (outside / "o.md").write_text("TITLE: outside\nDESC: I was never a draft\n\nbody\n",
                                  encoding="utf-8")
    d = a_distiller(s)
    rel = os.path.relpath(str(outside / "o"), d.drafts_dir)
    r = d.pour(rel)
    assert r["ok"] is False and "bare slug" in r["why"]
    assert (outside / "o.md").exists()          # the outside file was not renamed
    assert "out-o" not in s.slugs()


def test_a_draft_nobody_gated_does_not_pour(tmp_path):
    """`distiller-only` has to mean "this passed the evidence gate". File existence in
    `_still/drafts/` is not that: a hand-written draft poured straight into a store
    whose direct door refuses everything."""
    s = a_store(tmp_path, policy="distiller-only")
    d = a_distiller(s)
    os.makedirs(d.drafts_dir, exist_ok=True)
    with open(os.path.join(d.drafts_dir, "x.md"), "w", encoding="utf-8") as f:
        f.write("TITLE: x\nDESC: nobody verified this\n\nbody\n")
    r = d.pour("x")
    assert r["ok"] is False and "gate mark" in r["why"]
    assert "x" not in s.slugs()


def test_a_draft_the_gate_staged_does_pour(tmp_path):
    s = a_store(tmp_path, policy="distiller-only")
    d = a_distiller(s)
    d.stage({"slug": "real", "title": "Real", "description": "the trigger", "body": "the body",
             "kind": "project", "evidence": [{"class": "USER", "text": "they said so"}],
             "classes": ["USER"], "unverified_numbers": False, "judgement": False,
             "attributed_to_human": False}, "session.jsonl")
    assert d.pour("real")["ok"] is True
    assert "real" in s.slugs()


def test_editing_a_staged_draft_invalidates_its_mark(tmp_path):
    """The mark signs the body, so a draft edited after staging is no longer gated."""
    s = a_store(tmp_path, policy="distiller-only")
    d = a_distiller(s)
    p = d.stage({"slug": "real", "title": "Real", "description": "trigger", "body": "the body",
                 "kind": "project", "evidence": [{"class": "USER", "text": "x"}],
                 "classes": ["USER"], "unverified_numbers": False, "judgement": False,
                 "attributed_to_human": False}, "s.jsonl")
    raw = open(p, encoding="utf-8").read()
    open(p, "w", encoding="utf-8").write(raw.replace("the body", "something else entirely"))
    r = d.pour("real")
    assert r["ok"] is False and "gate mark" in r["why"]


# ── the store's own doors ───────────────────────────────────────────────────

def test_a_frozen_refusal_does_not_point_at_a_door_that_is_also_shut(tmp_path):
    s = a_store(tmp_path, policy="frozen")
    err = s.remember_direct("x", "d", "b")["error"]
    assert "frozen" in err and "distiller" not in err


def test_swapping_the_directory_for_a_symlink_afterwards_is_refused(tmp_path):
    """`rmdir scratch; ln -s private scratch` turned a permissive store into a writable
    alias of a protected one, while the Store object looked unchanged."""
    (tmp_path / "private").mkdir()
    (tmp_path / "scratch").mkdir()
    s = Store(name="scratch", path=str(tmp_path / "scratch"))
    os.rmdir(tmp_path / "scratch")
    os.symlink(tmp_path / "private", tmp_path / "scratch")
    r = s.remember_direct("planted", "d", "b")
    assert r["ok"] is False and "no longer resolves" in r["error"]
    assert not (tmp_path / "private" / "planted.md").exists()


def test_a_second_pour_of_one_slug_does_not_destroy_the_first_poured_draft(tmp_path):
    """`os.rename(p, p + ".poured")` overwrote the archive of the earlier draft: once
    the first draft was poured its file left `*.md`, so the next draft of the same
    slug was staged under the plain name again and its pour renamed straight onto the
    older `.poured` — the first draft's text, evidence header and mark simply gone."""
    s = a_store(tmp_path, policy="distiller-only")
    d = a_distiller(s)
    base = {"slug": "real", "title": "Real", "description": "the trigger",
            "kind": "project", "evidence": [{"class": "USER", "text": "they said so"}],
            "classes": ["USER"], "unverified_numbers": False, "judgement": False,
            "attributed_to_human": False}
    d.stage({**base, "body": "the first body"}, "session.jsonl")
    assert d.pour("real")["ok"] is True
    first = open(os.path.join(d.drafts_dir, "real.md.poured"), encoding="utf-8").read()
    d.stage({**base, "body": "the second body"}, "session.jsonl")
    assert d.pour("real")["ok"] is True
    assert open(os.path.join(d.drafts_dir, "real.md.poured"),
                encoding="utf-8").read() == first
    poured = sorted(f for f in os.listdir(d.drafts_dir) if f.endswith(".poured"))
    assert len(poured) == 2, poured
    assert "the second body" in "".join(
        open(os.path.join(d.drafts_dir, f), encoding="utf-8").read() for f in poured)
