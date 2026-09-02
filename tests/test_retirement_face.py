"""The retirement face — the one way canonical may say "this is over".

Every test here is named for the failure it prevents. The shape of the danger is
always the same: a memory that is no longer true keeps being read as current, or —
the opposite and worse — something that is not the human's own word quietly rewrites
the map's most-read line.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.store import Store                          # noqa: E402
from distill_kura.weave import Loom                           # noqa: E402


def a_store(tmp_path, policy="direct-allowed"):
    s = Store(name="m", path=str(tmp_path / "m"))
    s.init_files()
    s.remember("old-way", "the old way of doing it, kept because it is still true history",
               "OLD BODY")
    s.remember("new-way", "the new way, which replaced it", "NEW BODY")
    s.write_policy = policy
    return s


def a_manifest(store: Store, quotes: list[dict]) -> str:
    """A manifest named by the hash of its own bytes, like a genuine one."""
    blob = json.dumps({"gate_version": 6, "quotes": quotes}, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    os.makedirs(os.path.join(store.path, "_evidence"), exist_ok=True)
    with open(os.path.join(store.path, "_evidence", h + ".json"), "w", encoding="utf-8") as f:
        f.write(blob)
    return h


def user_manifest(store: Store) -> str:
    return a_manifest(store, [{"class": "USER",
                               "text": "stop using old-way, now use new-way"}])


def hook_of(store: Store, slug: str) -> str:
    for line in store.index_text().splitlines():
        if f"({slug}.md)" in line:
            return line.split(" — ", 1)[1]
    raise AssertionError(f"{slug} has no index line")


# ── the door works ──────────────────────────────────────────────────────────

def test_a_verified_transition_faces_the_old_line_and_nothing_else(tmp_path):
    s = a_store(tmp_path)
    h = user_manifest(s)
    before_new = s.read_exact("new-way")
    r = s.retire("old-way", "new-way", h)
    assert r["ok"] and r["changed"]
    assert hook_of(s, "old-way").startswith("superseded: ")
    assert "[[new-way]]" in hook_of(s, "old-way")
    assert f"retired: superseded by [[new-way]] (manifest sha256:{h})" in s.read_exact("old-way")
    assert "OLD BODY" in s.read_exact("old-way"), "the old memory is never emptied"
    assert s.read_exact("new-way") == before_new, "the new memory is not touched"


def test_the_face_is_written_in_the_script_of_the_old_trigger(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m"))
    s.init_files()
    s.remember("k3-plan", "K3をSSD階層で走らせる計画。純CPUで10 tok/sを狙う", "本文")
    s.remember("k3-new", "新しい計画", "本文")
    h = a_manifest(s, [{"class": "USER", "text": "k3-plan はやめて、今後は k3-new で行く"}])
    assert s.retire("k3-plan", "k3-new", h)["ok"]
    hook = hook_of(s, "k3-plan")
    assert hook.startswith("退役: ") and "／現在は [[k3-new]]" in hook


def test_a_user_quote_naming_the_old_title_is_enough(tmp_path):
    """The human says the memory's NAME, not its slug — the way people talk."""
    s = a_store(tmp_path)
    title = next(t for t, sl in s.titles().items() if sl == "old-way")
    h = a_manifest(s, [{"class": "USER", "text": f"instead of {title} we now use new-way"}])
    assert s.retire("old-way", "new-way", h)["ok"]


# ── the failures it prevents ────────────────────────────────────────────────

def test_a_derived_edge_alone_cannot_retire(tmp_path):
    """M7 edges are a READING of the store. A reading that could rewrite what it read
    would let the store's own prose retire its memories — no human in the loop at all."""
    import inspect

    from distill_kura import edges

    src = inspect.getsource(edges)
    assert ".retire(" not in src and "store.retire" not in src, \
        "edges.py must never reach the retirement door"
    s = a_store(tmp_path)
    # And the derived map, built over a store whose bodies talk about superseding,
    # changes no index line.
    s.remember("old-way", "the old way of doing it, kept because it is still true history",
               "OLD BODY. this supersedes nothing; see [[new-way]] which replaced it")
    before = s.index_text()
    edges.current(s)
    assert s.index_text() == before
    assert not Store.is_faced(hook_of(s, "old-way"))


def test_a_tool_only_manifest_cannot_retire(tmp_path):
    """A `git log` line saying "removed the old path" is not the human retiring a plan."""
    s = a_store(tmp_path)
    h = a_manifest(s, [{"class": "TOOL", "text": "deleted old-way.py, the new-way path wins"},
                       {"class": "SELF", "text": "I judge old-way to be finished"},
                       {"class": "ACT", "text": "ran the new-way script instead of old-way"}])
    r = s.retire("old-way", "new-way", h)
    assert not r["ok"] and "[USER]" in r["error"]
    assert not Store.is_faced(hook_of(s, "old-way"))


def test_a_manifest_that_only_names_the_old_memory_cannot_retire(tmp_path):
    """The door does not trust its caller. A [USER] quote that merely MENTIONS the
    old memory — or retires it without naming a successor — is not the human saying
    old → new, and `現在は [[new]]` may only be written by that sentence."""
    s = a_store(tmp_path)
    for text in ("old-way is what got us here, remember",
                 "old-way はもうやめよう。ところで別件で new-way の話だけど",
                 "old-way はもうやめる"):
        h = a_manifest(s, [{"class": "USER", "text": text}])
        r = s.retire("old-way", "new-way", h)
        assert not r["ok"], text
        assert "no explicit succession in the human's words" in r["error"], text
        assert not Store.is_faced(hook_of(s, "old-way")), text


def test_a_tampered_manifest_cannot_retire(tmp_path):
    """Content-addressed means the name is the hash. Edit the bytes, lose the door."""
    s = a_store(tmp_path)
    h = user_manifest(s)
    p = os.path.join(s.path, "_evidence", h + ".json")
    man = json.load(open(p, encoding="utf-8"))
    man["quotes"][0]["text"] = "stop using new-way, old-way is fine"
    open(p, "w", encoding="utf-8").write(json.dumps(man, ensure_ascii=False, sort_keys=True))
    r = s.retire("old-way", "new-way", h)
    assert not r["ok"] and "tampered" in r["error"]


def test_a_missing_manifest_cannot_retire(tmp_path):
    s = a_store(tmp_path)
    r = s.retire("old-way", "new-way", "f" * 64)
    assert not r["ok"] and "missing" in r["error"]


def test_a_memory_cannot_supersede_itself(tmp_path):
    s = a_store(tmp_path)
    r = s.retire("old-way", "old-way", user_manifest(s))
    assert not r["ok"] and "itself" in r["error"]


def test_an_absent_old_or_new_memory_is_a_refusal(tmp_path):
    s = a_store(tmp_path)
    h = user_manifest(s)
    assert not s.retire("no-such", "new-way", h)["ok"]
    r = s.retire("old-way", "no-such-either", h)
    assert not r["ok"] and "nothing to point at" in r["error"]
    assert not Store.is_faced(hook_of(s, "old-way"))


def test_the_second_call_is_already_and_writes_nothing(tmp_path):
    s = a_store(tmp_path)
    h = user_manifest(s)
    assert s.retire("old-way", "new-way", h)["ok"]
    faced_index, faced_body, rev = s.index_text(), s.read_exact("old-way"), s.revision()
    r = s.retire("old-way", "new-way", h)
    assert r["ok"] and r["already"] and not r["changed"]
    assert s.index_text() == faced_index and s.read_exact("old-way") == faced_body
    assert s.revision() == rev, "a no-op must not announce a mutation"


def test_a_second_different_successor_is_refused(tmp_path):
    """A face is a ruling, not a stack: overwriting it would erase the first one."""
    s = a_store(tmp_path)
    s.remember("third-way", "a later idea", "BODY")
    h = user_manifest(s)
    assert s.retire("old-way", "new-way", h)["ok"]
    h2 = a_manifest(s, [{"class": "USER", "text": "we replaced old-way with third-way"}])
    r = s.retire("old-way", "third-way", h2)
    assert not r["ok"] and "already wears" in r["error"]
    assert "[[new-way]]" in hook_of(s, "old-way")


def test_the_retired_memory_is_still_listed_and_still_readable(tmp_path):
    """Never deleted, never hidden, never re-slugged: forgetting is not how a store
    records that the world moved on."""
    s = a_store(tmp_path)
    assert s.retire("old-way", "new-way", user_manifest(s))["ok"]
    assert "old-way" in s.slugs() and "old-way" in s.known_slugs()
    assert s.resolve_exact("old-way") == "old-way"
    assert "OLD BODY" in s.read("old-way")
    assert s.doctor()["not_in_index"] == [] and s.doctor()["index_orphans"] == []


def test_a_frozen_store_refuses_to_retire(tmp_path):
    s = a_store(tmp_path, policy="frozen")
    h = user_manifest(s)
    r = s.retire("old-way", "new-way", h)
    assert not r["ok"] and "frozen" in r["error"]
    assert not Store.is_faced(hook_of(s, "old-way"))


def test_a_grouped_index_line_is_left_to_its_author(tmp_path):
    """Rewriting a grouped line from one slug would swallow its siblings — the same
    rule `_write` keeps."""
    s = a_store(tmp_path)
    s._write_index("# m\n\n- ways — [old](old-way.md)/[new](new-way.md)\n")
    r = s.retire("old-way", "new-way", user_manifest(s))
    assert not r["ok"] and "no index line of its own" in r["error"]


# ── the face survives the map ───────────────────────────────────────────────

def _aged(store: Store, slug: str, days: float = 400) -> None:
    """Push a memory out of the `fresh` layer so its line is actually compressed."""
    old = time.time() - days * 86400
    os.utime(store.file_of(slug), (old, old))


def a_long_faced_store(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m"))
    s.init_files()
    s.remember("k3-plan",
               "K3をSSD階層で走らせる計画。純CPUで10 tok/sを狙う。作戦帳はCAMPAIGN.mdに置き、"
               "職人はDSH-Qwenが担当し、監督は雲のユキが受け持つ長い長い説明の行", "本文")
    s.remember("k3-new", "新しい計画", "本文")
    h = a_manifest(s, [{"class": "USER", "text": "k3-plan はやめて、今後は k3-new で行く"}])
    assert s.retire("k3-plan", "k3-new", h)["ok"]
    _aged(s, "k3-plan")
    return s


def test_the_weave_never_compresses_the_face_away(tmp_path):
    """The face's tail is exactly what a length cut removes — and a worn map that
    says a dead plan is current, with no link to what replaced it, is worse than no
    map at all."""
    s = a_long_faced_store(tmp_path)
    canonical = next(l for l in s.index_text().splitlines() if "(k3-plan.md)" in l)
    for generate in (False, True):
        cloth = Loom(s, scribe=None).weave(generate=generate)
        line = next(l for l in cloth.text.splitlines() if "(k3-plan.md)" in l)
        assert "退役: " in line and "[[k3-new]]" in line, line
        assert len(line) < len(canonical), "and it did compress"


def test_a_scribes_answer_cannot_drop_the_face_either(tmp_path):
    """The scribe is shown only the part inside the face, so no answer of its own
    can lose the retirement word or the successor link."""
    class Scribe:
        def ask(self, system, user, **kw):
            return "K3をSSD階層で走らせる計画"     # grounded, and faceless

    s = a_long_faced_store(tmp_path)
    line = next(l for l in Loom(s, scribe=Scribe()).weave().text.splitlines()
                if "(k3-plan.md)" in l)
    assert line.endswith("／現在は [[k3-new]]") and "退役: " in line
    assert "K3をSSD階層で走らせる計画" in line, "and the scribe's compression is still worn"


def test_the_floors_accept_a_faced_trigger(tmp_path):
    """floors._OBSOLETE refuses a cut that DROPS a retirement word; it must not refuse
    one that carries it."""
    from distill_kura import floors

    s = a_long_faced_store(tmp_path)
    desc = next(l for l in s.index_text().splitlines()
                if "(k3-plan.md)" in l).split(" — ", 1)[1]
    loom = Loom(s, scribe=None)
    trigger = loom._mechanical(desc, "k3-plan")
    assert floors.first_violation(trigger, "k3-plan", desc, loom) is None
    # And the guard it exists for still bites: the same cut without the face.
    faceless = trigger.split("／")[0].removeprefix("退役: ")
    assert floors.first_violation(faceless, "k3-plan", desc, loom) == "retirement word dropped"


# ── the distiller's own path ────────────────────────────────────────────────
#
# Built through the pipeline's own `_write_manifest` and `stage`, never a hand-made
# hash: the point of these is that the REAL path fires, and a fixture that mints its
# own provenance proves nothing about it.

def a_distiller(tmp_path, store):
    from distill_kura.distill.pipeline import Distiller
    from distill_kura.registry import Registry
    from distill_kura.thinker import Models

    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    reg = Registry(stores={store.name: store}, modes={}, models=models, default=store.name,
                   raw={"distill": {"journals": {"claude": str(tmp_path)}}})
    return Distiller(reg, store)


def stage_and_pour(tmp_path, dis, evidence, classes, slug="newer-way",
                   title="the newer way", description="what we do instead now"):
    src = tmp_path / "journal.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    dis._current_key = "test:retire"
    d = {"slug": slug, "kind": "project", "title": title,
         "description": description, "body": "BODY",
         "evidence": evidence, "classes": classes,
         "tags": [], "tag_basis": {},
         # exactly what the gate leaves behind: `superseded` is proposed and REFUSED
         "tags_refused": {"superseded": "reserved for the forgetting pass; "
                                        "a model may not assign it"}}
    dis.stage(d, str(src))
    return dis.pour(d["slug"])


def test_the_distiller_retires_on_the_humans_own_words(tmp_path):
    s = a_store(tmp_path, policy="distiller-only")
    dis = a_distiller(tmp_path, s)
    r = stage_and_pour(tmp_path, dis,
                       [{"class": "USER", "text": "stop using old-way — switch to newer-way"}],
                       ["USER"])
    assert r["ok"] and r["retired"] == "old-way"
    assert Store.is_faced(hook_of(s, "old-way")) and "[[newer-way]]" in hook_of(s, "old-way")


def test_the_agents_own_prose_retires_nothing(tmp_path):
    """A [SELF] quote is the agent talking about the store. If that could retire a
    memory, the store would be able to retire itself by narrating."""
    s = a_store(tmp_path, policy="distiller-only")
    dis = a_distiller(tmp_path, s)
    r = stage_and_pour(tmp_path, dis,
                       [{"class": "SELF", "text": "I read old-way as finished; this replaces it"},
                        {"class": "TOOL", "text": "old-way.sh: No such file or directory"}],
                       ["SELF", "TOOL"])
    assert r["ok"] and not r.get("retired")
    assert not Store.is_faced(hook_of(s, "old-way"))


def test_a_pour_that_names_no_existing_memory_retires_nothing(tmp_path):
    s = a_store(tmp_path, policy="distiller-only")
    dis = a_distiller(tmp_path, s)
    r = stage_and_pour(tmp_path, dis,
                       [{"class": "USER", "text": "stop doing it the way we used to, this is better"}],
                       ["USER"])
    assert r["ok"] and not r.get("retired")
    assert not Store.is_faced(hook_of(s, "old-way"))


def test_a_quote_that_merely_names_the_old_memory_retires_nothing(tmp_path):
    """People talk about their memories all the time. Naming one is not retiring it,
    with or without a `superseded` tag anywhere near the draft."""
    s = a_store(tmp_path, policy="distiller-only")
    dis = a_distiller(tmp_path, s)
    r = stage_and_pour(tmp_path, dis,
                       [{"class": "USER", "text": "old-way is what got us here, remember"}],
                       ["USER"])
    assert r["ok"] and not r.get("retired")
    assert not Store.is_faced(hook_of(s, "old-way"))


def test_a_proposed_supersede_plus_a_by_the_way_quote_writes_no_successor(tmp_path):
    """THE attack. The human retires old-way and then, in a `ところで` clause, starts
    an unrelated subject; the model wrongly proposes `superseded` on the memory that
    grew out of that clause. Reading the refused proposal as a signal wrote
    `退役: …／現在は [[gpu-temperature]]` into canonical — a successor nobody chose."""
    s = a_store(tmp_path, policy="distiller-only")
    dis = a_distiller(tmp_path, s)
    before = next(l for l in s.index_text().splitlines() if "(old-way.md)" in l)
    r = stage_and_pour(
        tmp_path, dis,
        [{"class": "USER",
          "text": "old-way はもうやめよう。ところで別件で GPU 温度の記録を取ろう"}],
        ["USER"], slug="gpu-temperature", title="GPU 温度の記録",
        description="GPU の温度を毎分記録して残す")
    assert r["ok"] and not r.get("retired")
    assert next(l for l in s.index_text().splitlines()
                if "(old-way.md)" in l) == before, "canonical's line must be untouched"
    assert not Store.is_faced(hook_of(s, "old-way"))


def test_a_retirement_without_a_successor_writes_no_face(tmp_path):
    """`old-way はもうやめる` proves retirement, not succession. A successor-less face
    is not implemented, so the distiller writes nothing at all rather than guess."""
    s = a_store(tmp_path, policy="distiller-only")
    dis = a_distiller(tmp_path, s)
    r = stage_and_pour(tmp_path, dis,
                       [{"class": "USER", "text": "old-way はもうやめる"}], ["USER"])
    assert r["ok"] and not r.get("retired")
    assert not Store.is_faced(hook_of(s, "old-way"))


def test_the_distiller_retires_on_an_explicit_japanese_sentence(tmp_path):
    s = a_store(tmp_path, policy="distiller-only")
    dis = a_distiller(tmp_path, s)
    r = stage_and_pour(tmp_path, dis,
                       [{"class": "USER", "text": "old-way はやめて、今後は newer-way で行く"}],
                       ["USER"])
    assert r["ok"] and r["retired"] == "old-way"
    assert "／現在は [[newer-way]]" in hook_of(s, "old-way") or \
           "[[newer-way]]" in hook_of(s, "old-way")


def test_a_transition_split_across_two_quotes_retires_nothing(tmp_path):
    """One quote names the old memory, another names the new one. Stitching them
    would let two unrelated sentences become a ruling neither of them made."""
    s = a_store(tmp_path, policy="distiller-only")
    dis = a_distiller(tmp_path, s)
    r = stage_and_pour(tmp_path, dis,
                       [{"class": "USER", "text": "old-way はもうやめる"},
                        {"class": "USER", "text": "今後は newer-way で行く"}], ["USER"])
    assert r["ok"] and not r.get("retired")
    assert not Store.is_faced(hook_of(s, "old-way"))


def test_the_pour_prompt_still_demands_a_verified_transition(tmp_path):
    """The doctrine the writer is shown is unchanged by this fix: the face is worn
    only when the transition is VERIFIED from the human's own words."""
    from distill_kura.distill import prompts

    p = prompts.INDEX_CRAFT
    assert "Retired things wear it" in p
    assert "VERIFIED" in p and "never a guess from prose" in p


def test_the_drain_counts_the_transition_in_its_metrics_row(tmp_path):
    """A face appearing on the map with nothing in the metrics is a change to
    canonical that nobody can add up afterwards."""
    s = a_store(tmp_path, policy="distiller-only")
    dis = a_distiller(tmp_path, s)
    src = tmp_path / "journal.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    dis._current_key = "test:retire"
    dis.stage({"slug": "newer-way", "kind": "project", "title": "the newer way",
               "description": "what we do instead now", "body": "BODY",
               "evidence": [{"class": "USER",
                             "text": "stop using old-way — switch to newer-way"}],
               "classes": ["USER"], "tags": [], "tag_basis": {},
               "tags_refused": {"superseded": "reserved for the forgetting pass"}},
              str(src))
    dis.judge_draft = lambda p: {"slug": os.path.basename(p)[:-3], "verdict": "POUR",
                                 "why": "scripted", "judged_sha": None}
    out = dis.drain()
    assert out["poured"] == 1 and out["retired"] == 1
    assert Store.is_faced(hook_of(s, "old-way"))
    rows = [json.loads(l) for l in
            open(os.path.join(s.still, "metrics.jsonl"), encoding="utf-8")]
    assert rows[-1]["op"] == "drain" and rows[-1]["retired"] == 1
    assert rows[-1]["retired_slugs"] == ["old-way"]


# ── the human-driven transition, and the counters ───────────────────────────

def a_config(tmp_path, store):
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f'default = "m"\n[stores.m]\npath = "{store.path}"\n'
                   '[models.thinker]\nurl = "http://127.0.0.1:9/v1"\nmodel = "none"\n',
                   encoding="utf-8")
    return str(cfg)


def test_the_cli_retires_and_says_what_it_did(tmp_path, capsys):
    from distill_kura.cli import main

    s = a_store(tmp_path)
    h = user_manifest(s)
    cfg = a_config(tmp_path, s)
    assert main(["-c", cfg, "-s", "m", "retire", "old-way", "new-way", "--manifest", h]) == 0
    assert "superseded by new-way" in capsys.readouterr().out
    assert Store.is_faced(hook_of(s, "old-way"))
    # sha256:-prefixed is the spelling the memories themselves carry; both work.
    assert main(["-c", cfg, "-s", "m", "retire", "old-way", "new-way",
                 "--manifest", "sha256:" + h]) == 0
    assert "already" in capsys.readouterr().out


def test_the_cli_exits_1_with_the_reason_on_a_refusal(tmp_path, capsys):
    """A scheduler that reads 0 for a refused retirement would report a transition
    that never happened."""
    from distill_kura.cli import main

    s = a_store(tmp_path)
    cfg = a_config(tmp_path, s)
    bad = a_manifest(s, [{"class": "TOOL", "text": "old-way.py removed, new-way stands"}])
    assert main(["-c", cfg, "-s", "m", "retire", "old-way", "new-way", "--manifest", bad]) == 1
    assert "[USER]" in capsys.readouterr().err
    assert not Store.is_faced(hook_of(s, "old-way"))


def test_doctor_and_richness_count_the_faced_memories(tmp_path):
    from distill_kura import richness

    s = a_store(tmp_path)
    assert s.doctor()["retired"] == 0
    assert richness.gauge(s)["retired"] == 0
    assert s.retire("old-way", "new-way", user_manifest(s))["ok"]
    d = s.doctor()
    assert d["retired"] == 1 and d["retired_names"] == ["old-way"]
    assert d["memories"] == 2, "a retired memory is still a memory"
    r = richness.gauge(s)
    assert r["retired"] == 1
    assert "retired: 1" in richness.table(r)


def test_a_worldline_case_reports_whether_canonical_says_obsolete(tmp_path):
    """`edge_says_obsolete` is what the DERIVED map thinks; `obsolete_faced` is what
    canonical says. Raw metrics, side by side — neither moves a score."""
    from distill_kura import worldline as wl

    class StubModel:
        def ask_full(self, system, user, **kw):
            return {"content": '["new-way"]', "finish_reason": "stop"}

    s = a_store(tmp_path)
    case = {"id": "c1", "utterance": "the old way — is that still the idea?",
            "target_slugs": ["new-way"], "acceptable_related": [],
            "must_not_anchor": [], "obsolete_slugs": ["old-way"],
            "category": "superseded-plan"}
    tr = wl.run_case(s, case, "agent-only", thinker=StubModel())
    assert tr["obsolete_faced"] is False and tr["target_reached"] is True
    assert s.retire("old-way", "new-way", user_manifest(s))["ok"]
    tr = wl.run_case(s, case, "agent-only", thinker=StubModel())
    assert tr["obsolete_faced"] is True
    assert tr["target_reached"] is True, "the face changes no score"
