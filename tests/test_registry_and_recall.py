"""Multi-kura routing, watermark safety, and recall's behaviour when the model is down.

The recall tests use a stub thinker, so they check the *plumbing* around the model:
that a picked slug is walked, that a wrong-format answer is still salvaged, and that
an unreachable thinker degrades to word matching and says so instead of going silent.
"""
from __future__ import annotations

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.distill.watermark import Watermarks   # noqa: E402
from distill_kura.recall import fit, recall             # noqa: E402
from distill_kura.registry import Registry              # noqa: E402
from distill_kura.store import Store                    # noqa: E402


class StubThinker:
    """Stands in for an endpoint. `answer=None` means unreachable."""
    def __init__(self, answer):
        self.answer = answer
        self.seen: list[str] = []

    def ask(self, system, user, **kw):
        self.seen.append(user)
        return self.answer


def a_store(tmp_path, name="s") -> Store:
    s = Store(name=name, path=str(tmp_path / name), label=name)
    s.init_files()
    s.remember("ssd-tier-mission", "running a huge model off an SSD tier", "body [[cooling]]")
    s.remember("cooling", "the fans had to go in before the CPU run", "body")
    return s


# ── registry ────────────────────────────────────────────────────────────────

def write_config(tmp_path, text: str) -> str:
    p = tmp_path / "kura.toml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_modes_resolve_to_stores(tmp_path):
    for n in ("maker", "eq"):
        Store(name=n, path=str(tmp_path / n)).init_files()
    cfg = write_config(tmp_path, f"""
[server]
default = "maker"
[models.thinker]
url = "http://127.0.0.1:1/v1"
model = "m"
[stores.maker]
path = "{tmp_path / 'maker'}"
[stores.eq]
path = "{tmp_path / 'eq'}"
[modes]
building = "maker"
talking = "eq"
""")
    reg = Registry.load(cfg)
    assert reg.store().name == "maker"            # default
    assert reg.store("eq").name == "eq"           # by store name
    assert reg.store("talking").name == "eq"      # by mode name


def test_an_unknown_mode_raises_instead_of_answering_from_the_default(tmp_path):
    """The old behaviour turned a typo in a mode name into "a different household's
    memory answered, fluently" — the opposite of failing loudly, and invisible from
    outside. The fallback still exists, under a name that admits what it does."""
    for n in ("maker", "eq"):
        Store(name=n, path=str(tmp_path / n)).init_files()
    cfg = write_config(tmp_path, f"""
[server]
default = "maker"
[stores.maker]
path = "{tmp_path / 'maker'}"
[stores.eq]
path = "{tmp_path / 'eq'}"
[modes]
talking = "eq"
""")
    reg = Registry.load(cfg)
    try:
        reg.store_for_mode("takling")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown mode must not silently answer from another store")
    assert reg.store_for_mode_or_default("takling").name == "maker"


def test_a_mode_named_after_a_different_store_is_refused(tmp_path):
    for n in ("maker", "eq"):
        Store(name=n, path=str(tmp_path / n)).init_files()
    cfg = write_config(tmp_path, f"""
[stores.maker]
path = "{tmp_path / 'maker'}"
[stores.eq]
path = "{tmp_path / 'eq'}"
[modes]
eq = "maker"
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "collides" in str(e)
    else:
        raise AssertionError("an ambiguous selector must not load")


def test_a_typo_in_a_store_key_fails_at_load(tmp_path):
    """`readnoly = true` used to land in `extra` and do nothing: a store that reads as
    protected and is not."""
    cfg = write_config(tmp_path, f"""
[stores.main]
path = "{tmp_path / 'main'}"
readnoly = true
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "readnoly" in str(e)
    else:
        raise AssertionError("an unknown store key must not be accepted silently")


def test_an_x_prefixed_key_is_allowed_through(tmp_path):
    cfg = write_config(tmp_path, f"""
[stores.main]
path = "{tmp_path / 'main'}"
x_my_extension = "hello"
""")
    reg = Registry.load(cfg)
    assert reg.stores["main"].extra["x_my_extension"] == "hello"


def test_a_wrong_type_fails_at_load(tmp_path):
    cfg = write_config(tmp_path, f"""
[stores.main]
path = "{tmp_path / 'main'}"
readonly = "yes"
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "must be bool" in str(e)
    else:
        raise AssertionError("a wrong type must not load")


def test_two_stores_may_not_share_one_directory(tmp_path):
    (tmp_path / "one").mkdir()
    cfg = write_config(tmp_path, f"""
[stores.a]
path = "{tmp_path / 'one'}"
[stores.b]
path = "{tmp_path / 'one'}"
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "same directory" in str(e)
    else:
        raise AssertionError("aliased stores must not load")


def test_a_store_inside_another_store_is_refused(tmp_path):
    (tmp_path / "outer" / "inner").mkdir(parents=True)
    cfg = write_config(tmp_path, f"""
[stores.outer]
path = "{tmp_path / 'outer'}"
[stores.inner]
path = "{tmp_path / 'outer' / 'inner'}"
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "nested" in str(e)
    else:
        raise AssertionError("nested stores must not load")


def test_a_symlinked_store_alias_is_refused(tmp_path):
    (tmp_path / "real").mkdir()
    os.symlink(tmp_path / "real", tmp_path / "alias")
    cfg = write_config(tmp_path, f"""
[stores.a]
path = "{tmp_path / 'real'}"
[stores.b]
path = "{tmp_path / 'alias'}"
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "same directory" in str(e)
    else:
        raise AssertionError("a symlink alias must not load as a second store")


def test_a_journal_root_may_not_contain_a_store(tmp_path):
    """The distiller would re-ingest memories as raw material and file model-written
    text as the human's words, which breaks the one guarantee the gate gives."""
    (tmp_path / "notes" / "kura").mkdir(parents=True)
    cfg = write_config(tmp_path, f"""
[stores.main]
path = "{tmp_path / 'notes' / 'kura'}"
[distill.journals]
text = "{tmp_path / 'notes'}"
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "overlaps" in str(e)
    else:
        raise AssertionError("a journal root containing a store must not load")


def test_path_overlap_can_be_accepted_explicitly(tmp_path):
    (tmp_path / "notes" / "kura").mkdir(parents=True)
    cfg = write_config(tmp_path, f"""
[server]
allow_path_overlap = true
[stores.main]
path = "{tmp_path / 'notes' / 'kura'}"
[distill.journals]
text = "{tmp_path / 'notes'}"
""")
    assert Registry.load(cfg).stores["main"].name == "main"


def test_a_mode_pointing_nowhere_fails_at_load(tmp_path):
    """Silent misconfiguration is the failure mode that looks like working software."""
    cfg = write_config(tmp_path, f"""
[stores.maker]
path = "{tmp_path / 'maker'}"
[modes]
eq = "does-not-exist"
""")
    try:
        Registry.load(cfg)
    except ValueError as e:
        assert "does-not-exist" in str(e)
    else:
        raise AssertionError("a mode pointing at no store must not load quietly")


def test_brain_and_scribe_inherit_the_thinker_by_default(tmp_path):
    cfg = write_config(tmp_path, f"""
[models.thinker]
url = "http://127.0.0.1:9/v1"
model = "one-model"
[stores.main]
path = "{tmp_path / 'main'}"
""")
    reg = Registry.load(cfg)
    assert reg.models.brain.model == "one-model"
    assert reg.models.scribe.model == "one-model"
    assert reg.models.describe()["single_model"] is True


def test_upgrading_one_role_leaves_the_others_alone(tmp_path):
    cfg = write_config(tmp_path, f"""
[models.thinker]
url = "http://127.0.0.1:9/v1"
model = "small"
[models.scribe]
url = "https://api.example.com/v1"
model = "big"
api_key_env = "EXAMPLE_KEY"
[stores.main]
path = "{tmp_path / 'main'}"
""")
    reg = Registry.load(cfg)
    assert reg.models.thinker.model == "small"
    assert reg.models.brain.model == "small"          # not upgraded
    assert reg.models.scribe.model == "big"
    assert reg.models.scribe.api_key_env == "EXAMPLE_KEY"
    assert reg.models.scribe.temperature == 0.4       # writing gets a little room
    assert reg.models.describe()["single_model"] is False


def test_no_config_still_gives_one_working_store(tmp_path, monkeypatch):
    monkeypatch.setenv("KURA_DIR", str(tmp_path / "solo"))
    monkeypatch.delenv("KURA_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    reg = Registry.load(None)
    assert list(reg.stores) == ["main"]


# ── recall ──────────────────────────────────────────────────────────────────

def test_recall_picks_by_meaning_and_walks_links(tmp_path):
    s = a_store(tmp_path)
    d = recall(s, StubThinker('["ssd-tier-mission"]'), "what about SSD inference?", hops=1)
    assert d["how"] == "meaning"
    assert d["picked"] == ["ssd-tier-mission"]
    assert d["walked"] == ["ssd-tier-mission", "cooling"]     # followed the [[link]]
    assert "ssd-tier-mission" in d["context"]


def test_picks_are_tidied_of_md_suffixes_and_brackets(tmp_path):
    """Models answer with `slug`, `slug.md` or `[slug]`; the caller should not care."""
    s = a_store(tmp_path)
    d = recall(s, StubThinker('["ssd-tier-mission.md", "[cooling]"]'), "q", hops=0)
    assert d["picked"] == ["ssd-tier-mission", "cooling"]
    assert d["walked"] == ["ssd-tier-mission", "cooling"]


def test_recall_salvages_a_slug_from_prose(tmp_path):
    """Models answer in prose maybe one time in three. A real slug in the text is a pick,
    whatever shape it arrived in."""
    s = a_store(tmp_path)
    d = recall(s, StubThinker("I think ssd-tier-mission is the relevant one here."),
               "SSD?", hops=0)
    assert d["picked"] == ["ssd-tier-mission"]


def test_recall_degrades_to_words_and_says_so(tmp_path):
    s = a_store(tmp_path)
    d = recall(s, StubThinker(None), "tell me about cooling", hops=0)
    assert d["how"].startswith("words")            # visible degradation, not silence
    assert d["walked"] == ["cooling"]


def test_recall_with_no_thinker_at_all(tmp_path):
    s = a_store(tmp_path)
    d = recall(s, None, "cooling", hops=0)
    assert d["walked"] == ["cooling"]


def test_empty_answer_is_reported_as_nothing_remembered(tmp_path):
    s = a_store(tmp_path)
    d = recall(s, StubThinker("[]"), "a topic nobody ever discussed", hops=0)
    assert d["walked"] == [] and d["context"] == ""


def test_fit_keeps_the_head_and_the_relevant_paragraph(tmp_path):
    """Trimming from the top loses conclusions, which live at the bottom of a memory."""
    text = ("---\nname: x\n---\n\n" + "opening line\n\n"
            + "\n\n".join(f"filler paragraph {i}" for i in range(40))
            + "\n\nthe measured answer was 42 tokens per second")
    out = fit(text, "how many tokens per second?", 400)
    assert "opening line" in out
    assert "42 tokens per second" in out


# ── watermarks ──────────────────────────────────────────────────────────────

def test_watermark_never_moves_backwards(tmp_path):
    w = Watermarks(str(tmp_path / "_still" / "watermark.json"))
    w.advance("k", 100)
    w.advance("k", 40)            # a stale writer
    assert w.read()["k"] == 100


def test_parallel_advances_do_not_lose_each_other(tmp_path):
    """The original bug: two distillers each wrote back their own snapshot and erased
    the other's progress, so the same stretch was drunk a dozen times."""
    w = Watermarks(str(tmp_path / "_still" / "watermark.json"))
    keys = [f"k{i}" for i in range(8)]

    def worker(k):
        for pos in range(1, 21):
            w.advance(k, pos)

    threads = [threading.Thread(target=worker, args=(k,)) for k in keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    marks = w.read()
    assert all(marks[k] == 20 for k in keys)


def test_claim_reserves_before_reading(tmp_path):
    """Claiming a stretch must move the mark immediately, so a second runner starting
    in the same instant gets different water."""
    j = tmp_path / "j.jsonl"
    j.write_text("x" * 500_000, encoding="utf-8")
    w = Watermarks(str(tmp_path / "_still" / "watermark.json"))
    first = w.claim([str(j)], budget_chars=100_000, min_chars=1000)
    second = w.claim([str(j)], budget_chars=100_000, min_chars=1000)
    assert first is not None and second is not None
    assert second[1] > first[1]          # different starting offsets
