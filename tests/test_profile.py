"""The learned profile: the wide room's understanding, in sentences, applied by a person.

What is pinned here: the profile is store-local and reads nothing from another store;
it is text, and a table of numbers is refused as broken rather than read; a missing
one and a broken one are both visible in `doctor` while the fixed charter carries on;
a draft is a file a person reads, never an application; and the resident map — the
byte-stable block — does not change when a profile appears.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from distill_kura import prefill, weave                      # noqa: E402
from distill_kura.distill import Distiller                   # noqa: E402
from distill_kura.distill.prompts import DEFAULT_CHARTER     # noqa: E402
from distill_kura.registry import Registry                   # noqa: E402
from distill_kura.store import Store                         # noqa: E402
from distill_kura.thinker import Models                      # noqa: E402

GOOD = """## Enduring threads
They keep returning to the question of where memories should live.

## Conversation preferences
Plain answers first; the reasoning after, if asked.
"""


def two(tmp_path):
    user = Store(name="user", path=str(tmp_path / "user"), write_policy="distiller-only")
    other = Store(name="other", path=str(tmp_path / "other"))
    for s in (user, other):
        s.init_files()
    user.pour_verified("memory-home", "where memories live", "they keep asking where memories should live")
    other.remember_direct("secret-plan", "the other room's secret", "THE OTHER ROOM KNOWS THIS")
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    reg = Registry(stores={"user": user, "other": other}, modes={}, models=models,
                   default="user", raw={"distill": {"language": "English"}})
    return reg, user, other


def test_profile_is_not_a_memory_and_its_state_is_visible(tmp_path):
    reg, user, _ = two(tmp_path)
    assert user.doctor()["learned_profile"]["state"] == "absent"
    open(user.profile_path, "w", encoding="utf-8").write(GOOD)
    assert "profile" not in " ".join(user.slugs())
    d = user.doctor()
    assert d["learned_profile"]["state"] == "present" and d["memories"] == 1
    assert user.profile_text() == GOOD


def test_a_table_of_numbers_is_broken_not_read(tmp_path):
    reg, user, _ = two(tmp_path)
    for bad in ("## Interests\n- trading: 0.8\n- gardening: 0.2\n",
                "## Threads\nmemory design, interest score 7\n",
                "| topic | weight |\n| memory | 9 |\n", "   \n"):
        open(user.profile_path, "w", encoding="utf-8").write(bad)
        st = user.profile_state()
        assert st["state"] == "broken", bad
        assert user.profile_text() == ""
        # the distiller names it and carries on with the fixed charter alone
        d = Distiller(reg, user)
        assert d.profile["state"] == "broken"
        assert d.charter == DEFAULT_CHARTER
    os.remove(user.profile_path)
    assert Distiller(reg, user).charter == DEFAULT_CHARTER


def test_a_sound_profile_is_read_after_the_charter_by_this_store_only(tmp_path):
    reg, user, other = two(tmp_path)
    open(user.profile_path, "w", encoding="utf-8").write(GOOD)
    d = Distiller(reg, user)
    assert d.charter.startswith(DEFAULT_CHARTER.rstrip("\n"))
    assert d.charter.rstrip().endswith("if asked.")
    assert Distiller(reg, other).charter == DEFAULT_CHARTER   # the other store is untouched
    # and still one head per store: every role's system prompt shares the prefix
    assert d._sys("A").startswith(d.charter) and d._sys("B").startswith(d.charter)


def test_draft_reads_only_this_stores_memories_and_applies_nothing(tmp_path):
    reg, user, other = two(tmp_path)
    d = Distiller(reg, user)
    seen = {}

    def brain(task, userx, max_tokens=0):
        seen["task"], seen["user"] = task, userx
        return GOOD
    d.brain = brain                                           # type: ignore[method-assign]
    r = d.profile_draft()
    assert r["ok"], r
    assert "memory-home" in seen["user"]
    assert "secret-plan" not in seen["user"] and "THE OTHER ROOM" not in seen["user"]
    assert "NO numbers about how much" in seen["task"]
    assert os.path.exists(r["draft"]) and r["draft"].endswith("_still/profile.draft.md")
    assert not os.path.exists(user.profile_path)              # drafted, not applied
    assert user.profile_state()["state"] == "absent"


def test_a_draft_that_would_be_a_broken_profile_is_refused(tmp_path):
    reg, user, _ = two(tmp_path)
    d = Distiller(reg, user)
    d.brain = lambda task, u, max_tokens=0: "## Interests\n- memory: 0.9\n"   # type: ignore
    r = d.profile_draft()
    assert not r["ok"] and "table of numbers" in r["why"]
    assert not os.path.exists(os.path.join(user.still, "profile.draft.md"))
    d.brain = lambda task, u, max_tokens=0: "Sure! Here is the profile:\n## x\n"   # type: ignore
    assert not Distiller.profile_draft(d)["ok"]


def test_a_draft_is_refused_in_the_same_words_the_store_would_use(tmp_path):
    """The draft check and the on-disk check are one check. When they drifted apart,
    a draft was refused with half a sentence ("carries a score or weight") while the
    store, told the same text, gave the reason the docs quote."""
    reg, user, _ = two(tmp_path)
    d = Distiller(reg, user)
    text = "## Threads\nmemory design, interest score 7\n"
    d.brain = lambda task, u, max_tokens=0: text   # type: ignore
    r = d.profile_draft()
    assert not r["ok"] and "a profile holds no numbers about how much things matter" in r["why"]
    open(user.profile_path, "w", encoding="utf-8").write(text)
    assert r["why"] == f"draft refused: {user.profile_state()['why']}"


def test_an_empty_store_has_nothing_to_draft_from(tmp_path):
    s = Store(name="e", path=str(tmp_path / "e")); s.init_files()
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    reg = Registry(stores={"e": s}, modes={}, models=models, default="e", raw={})
    assert not Distiller(reg, s).profile_draft()["ok"]


def test_the_resident_map_does_not_change_when_a_profile_appears(tmp_path):
    """The profile is for the distiller and the host's /profile; the byte-stable block
    an agent reads every turn is built from the index alone."""
    reg, user, _ = two(tmp_path)
    loom = weave.Loom(user)
    loom.weave(generate=False)
    before = prefill.build(user, loom=loom).text
    open(user.profile_path, "w", encoding="utf-8").write(GOOD)
    loom.weave(generate=False)
    after = prefill.build(user, loom=loom).text
    assert before == after
    assert "Enduring threads" not in after


def test_cli_show_draft_apply_and_frozen(tmp_path):
    reg, user, _ = two(tmp_path)
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[stores.user]
path = "{user.path}"
write_policy = "distiller-only"
[models.thinker]
url = "http://127.0.0.1:9/v1"
model = "none"
""", encoding="utf-8")
    e = {**os.environ, "PYTHONPATH": ROOT, "KURA_CONFIG": str(cfg)}
    run = lambda *a: subprocess.run([sys.executable, "-m", "distill_kura.cli", "-s", "user", *a],  # noqa: E731
                                    capture_output=True, text=True, env=e, timeout=60)
    p = run("profile", "show")
    assert p.returncode == 0 and json.loads(p.stdout)["state"] == "absent"
    p = run("profile", "apply")
    assert p.returncode == 1 and "no draft" in p.stdout
    os.makedirs(user.still, exist_ok=True)
    open(os.path.join(user.still, "profile.draft.md"), "w", encoding="utf-8").write(GOOD)
    assert json.loads(run("profile", "show").stdout)["draft_waiting"] is True
    p = run("profile", "apply")
    assert p.returncode == 0 and json.loads(p.stdout)["state"] == "present"
    assert user.profile_text() == GOOD
    assert not os.path.exists(os.path.join(user.still, "profile.draft.md"))
    # frozen: nothing is written there by anyone through this tool
    cfg.write_text(cfg.read_text().replace("distiller-only", "frozen"), encoding="utf-8")
    open(os.path.join(user.still, "profile.draft.md"), "w", encoding="utf-8").write(GOOD)
    p = run("profile", "apply")
    assert p.returncode == 1 and "frozen" in p.stdout
