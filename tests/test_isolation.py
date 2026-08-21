"""Isolation between stores: journals in, models out.

Separating memories on disk buys nothing if the *intake* or the *models* are shared.
Two stores fed from one sessions directory distil each other's conversations, and two
stores behind one thinker hand that endpoint both indexes. Both failures leave the
directories looking perfectly separate.

Reported by Sol Pro as P0-3, P1-1, P1-2 and P1-3.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.distill import Distiller                    # noqa: E402
from distill_kura.distill.sources import discover_all         # noqa: E402
from distill_kura.registry import Registry                    # noqa: E402
from distill_kura.store import Store                          # noqa: E402


def write_config(tmp_path, text: str) -> str:
    p = tmp_path / "kura.toml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def a_journal(path, name="session.jsonl", who="the human said something here"):
    os.makedirs(path, exist_ok=True)
    f = os.path.join(path, name)
    with open(f, "w", encoding="utf-8") as h:
        for _ in range(40):
            h.write(json.dumps({"type": "user",
                                "message": {"content": [{"type": "text", "text": who * 40}]}}) + "\n")
    return f


# ── journal intake ──────────────────────────────────────────────────────────

def test_two_stores_do_not_share_a_journal_root_by_default(tmp_path):
    """With more than one store and no per-store journals, the distiller refuses to
    guess. Feeding both from one root is how one mode's conversations end up distilled
    into another mode's memory."""
    a_journal(str(tmp_path / "logs"))
    for n in ("maker", "eq"):
        Store(name=n, path=str(tmp_path / n)).init_files()
    cfg = write_config(tmp_path, f"""
[stores.maker]
path = "{tmp_path / 'maker'}"
[stores.eq]
path = "{tmp_path / 'eq'}"
[distill.journals]
claude = "{tmp_path / 'logs'}"
""")
    reg = Registry.load(cfg)
    assert Distiller(reg, reg.store("maker")).journals == {}
    assert Distiller(reg, reg.store("eq")).journals == {}


def test_a_single_store_still_inherits_the_global_journals(tmp_path):
    """The simple deployment must not need extra ceremony."""
    a_journal(str(tmp_path / "logs"))
    Store(name="main", path=str(tmp_path / "main")).init_files()
    cfg = write_config(tmp_path, f"""
[stores.main]
path = "{tmp_path / 'main'}"
[distill.journals]
claude = "{tmp_path / 'logs'}"
""")
    reg = Registry.load(cfg)
    assert "claude" in Distiller(reg, reg.store("main")).journals


def test_an_empty_per_store_journals_table_inherits_nothing(tmp_path):
    """`journals = {}` used to fall through to the global roots, because an empty dict
    is falsey: "this store inherits nothing" silently meant "inherits everything"."""
    a_journal(str(tmp_path / "logs"))
    Store(name="main", path=str(tmp_path / "main")).init_files()
    cfg = write_config(tmp_path, f"""
[stores.main]
path = "{tmp_path / 'main'}"
[stores.main.distill]
inherit_global_journals = false
[stores.main.distill.journals]
[distill.journals]
claude = "{tmp_path / 'logs'}"
""")
    reg = Registry.load(cfg)
    assert Distiller(reg, reg.store("main")).journals == {}


def test_each_store_can_be_bound_to_its_own_root(tmp_path):
    a_journal(str(tmp_path / "logs-maker"))
    a_journal(str(tmp_path / "logs-eq"))
    for n in ("maker", "eq"):
        Store(name=n, path=str(tmp_path / n)).init_files()
    cfg = write_config(tmp_path, f"""
[stores.maker]
path = "{tmp_path / 'maker'}"
[stores.maker.distill.journals]
claude = "{tmp_path / 'logs-maker'}"
[stores.eq]
path = "{tmp_path / 'eq'}"
[stores.eq.distill.journals]
claude = "{tmp_path / 'logs-eq'}"
""")
    reg = Registry.load(cfg)
    maker = Distiller(reg, reg.store("maker")).files()
    eq = Distiller(reg, reg.store("eq")).files()
    assert maker and eq
    assert not (set(maker) & set(eq))          # no file is visible to both


def test_a_journal_inside_a_store_is_never_discovered(tmp_path):
    """Defence in depth behind the registry's load-time refusal: re-ingesting memories
    as raw material files model-written text as the human's words, which breaks the one
    guarantee the evidence gate gives."""
    store = Store(name="main", path=str(tmp_path / "main"))
    store.init_files()
    store.remember("a-memory", "written by the distiller", "body")
    inside = a_journal(str(tmp_path / "main" / "_still"), name="stray.jsonl")
    found = discover_all({"claude": str(tmp_path / "main")}, exclude_roots=[store.path])
    assert inside not in found and found == []


def test_include_and_exclude_globs_narrow_one_root(tmp_path):
    """One sessions directory usually holds every mode's conversations."""
    a_journal(str(tmp_path / "logs" / "maker"), name="s.jsonl")
    a_journal(str(tmp_path / "logs" / "scratch"), name="s.jsonl")
    found = discover_all({"claude": {"root": str(tmp_path / "logs"),
                                     "exclude_glob": ["scratch/**"]}})
    assert len(found) == 1 and "maker" in found[0]
    only = discover_all({"claude": {"root": str(tmp_path / "logs"),
                                    "include_glob": ["scratch/**"]}})
    assert len(only) == 1 and "scratch" in only[0]


# ── model isolation ─────────────────────────────────────────────────────────

def test_a_store_can_bind_its_own_model_endpoints(tmp_path):
    """One shared thinker sees every store's whole index, so separating them on disk
    buys nothing against the model."""
    for n in ("shared", "project"):
        Store(name=n, path=str(tmp_path / n)).init_files()
    cfg = write_config(tmp_path, f"""
[models.thinker]
url = "http://127.0.0.1:8000/v1"
model = "shared-model"
[model_profiles.private.thinker]
url = "http://127.0.0.1:8100/v1"
model = "private-model"
[stores.shared]
path = "{tmp_path / 'shared'}"
[stores.project]
path = "{tmp_path / 'project'}"
model_profile = "private"
""")
    reg = Registry.load(cfg)
    assert reg.models_for(reg.store("shared")).thinker.model == "shared-model"
    assert reg.models_for(reg.store("project")).thinker.model == "private-model"
    assert Distiller(reg, reg.store("project")).models.thinker.model == "private-model"


def test_an_undefined_model_profile_fails_at_load(tmp_path):
    """Never a quiet fall back to the shared endpoint — that is exactly how a private
    store's index reaches a model it was never meant to see."""
    Store(name="project", path=str(tmp_path / "project")).init_files()
    cfg = write_config(tmp_path, f"""
[stores.project]
path = "{tmp_path / 'project'}"
model_profile = "does-not-exist"
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "not defined" in str(e)
    else:
        raise AssertionError("an undefined profile must not load")


def test_profiles_inherit_role_defaults_within_themselves(tmp_path):
    Store(name="p", path=str(tmp_path / "p")).init_files()
    cfg = write_config(tmp_path, f"""
[model_profiles.solo.thinker]
url = "http://127.0.0.1:9/v1"
model = "one"
[stores.p]
path = "{tmp_path / 'p'}"
model_profile = "solo"
""")
    m = Registry.load(cfg).models_for(Registry.load(cfg).store("p"))
    assert m.brain.model == "one" and m.scribe.model == "one"
