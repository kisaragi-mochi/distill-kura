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


# ── the edges an adversarial pass found in the first isolation fix ──────────

def test_a_profile_that_leaves_a_role_out_fails_at_load(tmp_path):
    """Models chains thinker -> brain -> scribe, so a role missing at the head landed on
    the built-in default endpoint. A profile defining only `brain` sent a store's whole
    CONFIDENTIAL index to an endpoint named nowhere in the file — the exact fallback the
    feature exists to forbid."""
    Store(name="project", path=str(tmp_path / "project")).init_files()
    cfg = write_config(tmp_path, f"""
[models.thinker]
url = "http://127.0.0.1:18022/v1"
[model_profiles.private.brain]
url = "http://127.0.0.1:18021/v1"
[stores.project]
path = "{tmp_path / 'project'}"
model_profile = "private"
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "thinker.url" in str(e)
    else:
        raise AssertionError("a partial profile must not load")


def test_an_endpoint_with_no_url_is_unreachable_not_somewhere_else(tmp_path):
    from distill_kura.thinker import Endpoint
    assert Endpoint().url == ""
    assert Endpoint().ask("s", "u") is None      # never a guess at 127.0.0.1:8000
    assert Endpoint().alive() is False


def test_two_journal_roots_may_not_nest(tmp_path):
    """`_check_paths` compared journal roots against store roots only, so the outer
    store drank the inner store's entire intake."""
    for d in ("s1", "s2", "logs/eq"):
        os.makedirs(tmp_path / d, exist_ok=True)
    cfg = write_config(tmp_path, f"""
[stores.maker]
path = "{tmp_path / 's1'}"
[stores.maker.distill.journals]
text = "{tmp_path / 'logs'}"
[stores.eq]
path = "{tmp_path / 's2'}"
[stores.eq.distill.journals]
text = "{tmp_path / 'logs' / 'eq'}"
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "nested" in str(e)
    else:
        raise AssertionError("nested journal roots must not load")


def test_the_table_form_of_a_journal_root_is_checked_too(tmp_path):
    """`_real(str(root))` stringified the dict, so the documented table form skipped the
    store-overlap refusal entirely."""
    os.makedirs(tmp_path / "s1")
    cfg = write_config(tmp_path, f"""
[stores.main]
path = "{tmp_path / 's1'}"
[distill.journals]
text = {{ root = "{tmp_path / 's1'}" }}
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "overlaps" in str(e)
    else:
        raise AssertionError("a table-form journal root overlapping a store must not load")


def test_a_string_where_a_boolean_belongs_fails_at_load(tmp_path):
    """`inherit_global_journals = "false"` is a STRING, therefore truthy: the store
    inherited the global intake it had explicitly declined."""
    for n in ("s1", "s2"):
        Store(name=n, path=str(tmp_path / n)).init_files()
    cfg = write_config(tmp_path, f"""
[stores.s1]
path = "{tmp_path / 's1'}"
[stores.s1.distill]
inherit_global_journals = "false"
[stores.s2]
path = "{tmp_path / 's2'}"
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "inherit_global_journals must be bool" in str(e)
    else:
        raise AssertionError("a truthy string must not pass for a boolean")


def test_a_hardlinked_memory_in_a_journal_root_is_not_drunk(tmp_path):
    """Path exclusion is not enough: a hardlink to a memory sitting in an otherwise clean
    journal root is a different path to the same inode. It was sipped as [USER] evidence —
    model-written memory laundered into the human's words."""
    st = Store(name="s", path=str(tmp_path / "s"))
    st.init_files()
    st.remember("mem", "d", "MODEL-WRITTEN MEMORY BODY")
    jr = tmp_path / "jr"
    jr.mkdir()
    (jr / "note.md").write_text("a real human note\n", encoding="utf-8")
    os.link(st.file_of("mem"), jr / "hardlinked-memory.md")
    found = discover_all({"text": str(jr)}, exclude_roots=[st.path])
    # Compare BASENAMES: pytest names tmp_path after the test, so this very test's
    # directory contains the word "hardlinked" and a whole-path substring check passes
    # no matter what the code does.
    names = [os.path.basename(f) for f in found]
    assert names == ["note.md"]


def test_a_store_with_no_name_is_refused(tmp_path):
    os.makedirs(tmp_path / "s1")
    cfg = write_config(tmp_path, f"""
[stores.""]
path = "{tmp_path / 's1'}"
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "needs a name" in str(e)
    else:
        raise AssertionError("an unaddressable store must not load")


def test_a_contradictory_write_config_is_refused(tmp_path):
    """The deprecated key was applied last and always won, so tightening a store while a
    stale `readonly = false` sat in the file produced a fully writable one."""
    os.makedirs(tmp_path / "s1")
    cfg = write_config(tmp_path, f"""
[stores.main]
path = "{tmp_path / 's1'}"
write_policy = "frozen"
readonly = false
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "readonly" in str(e) and "write_policy" in str(e)
    else:
        raise AssertionError("a contradictory write config must not load")
