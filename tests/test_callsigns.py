"""USER callsigns — the shared vernacular that leads back to a memory.

A callsign is NOT content: it is the two-of-us word ("全員野球") that ROUTES to a
memory. Because a routing word is worth exactly as much as its provenance, the
rules are absolute and tested adversarially here, gate-level first:

* only the human's own words, verbatim inside a SURVIVING [USER] quote —
  a phrase the agent or a tool merely used is not a shared vocabulary;
* the model may not paraphrase, invent, or pick targets — verify_callsigns sees
  proposals and evidence and nothing else.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.distill.gate import verify_callsigns    # noqa: E402


def ev(cls, text):
    return {"class": cls, "text": text}


U = ev("USER", "例の全員野球でいこう、CPUもGPUも同じチームで回す")
T = ev("TOOL", "quartus: 全員野球 mode engaged, 4 devices online")
S = ev("SELF", "全員野球という呼び方が定着したと思う")


# ── the ten attacks, gate level (1–7) ───────────────────────────────────────

def test_1_a_phrase_only_the_agent_used_is_refused():
    kept, refused = verify_callsigns(["全員野球"], [S])
    assert kept == [] and "human" in refused["全員野球"]


def test_2_a_phrase_only_the_tools_said_is_refused():
    kept, refused = verify_callsigns(["全員野球"], [T])
    assert kept == [] and "human" in refused["全員野球"]


def test_3_the_humans_own_words_are_accepted_with_their_quote():
    kept, refused = verify_callsigns(["全員野球"], [U, T, S])
    assert kept == [{"text": "全員野球", "class": "USER",
                     "quote": U["text"]}], kept
    assert refused == {}


def test_4_a_paraphrase_of_the_humans_words_is_refused():
    """The USER said 全員野球; the model proposing みんなで協力 is inventing a
    nickname the human never used — exactly the failure the gate exists for."""
    kept, refused = verify_callsigns(["みんなで協力"], [U])
    assert kept == [] and "みんなで協力" in refused


def test_5_two_codepoints_are_too_short_to_route():
    kept, refused = verify_callsigns(["go", "寝る"], [ev("USER", "go 寝る now")])
    assert kept == [] and len(refused) == 2


def test_6_forty_one_codepoints_are_not_a_callsign():
    long = "あ" * 41
    kept, refused = verify_callsigns([long], [ev("USER", f"彼は{long}と言った")])
    assert kept == [] and long in refused


def test_7_the_gated_value_is_the_round_trip_value():
    """Whitespace around a proposal is shape noise: the gate accepts the collapsed
    form, and what it returns must equal what a JSON manifest round-trip gives
    back — the annotation-signing lesson, applied before the wound this time."""
    kept, _ = verify_callsigns(["  全員野球 \n"], [U])
    assert kept and kept[0]["text"] == "全員野球"
    assert json.loads(json.dumps(kept))[0]["text"] == kept[0]["text"]


# ── gate-level rules the attacks imply ──────────────────────────────────────

def test_at_most_two_callsigns_survive():
    quotes = ev("USER", "全員野球 と 知性備蓄部門 と 蔵坚固 の三つで進める")
    kept, refused = verify_callsigns(["全員野球", "知性備蓄部門", "蔵坚固"], [quotes])
    assert [k["text"] for k in kept] == ["全員野球", "知性備蓄部門"]
    assert refused["蔵坚固"].startswith("at most two")


def test_whitespace_and_punctuation_only_is_refused():
    kept, refused = verify_callsigns(["・・・", " — "], [ev("USER", "・・・ — 本気")])
    assert kept == [] and len(refused) == 2


def test_the_same_callsign_proposed_twice_is_kept_once():
    kept, _ = verify_callsigns(["全員野球", "全員野球"], [U])
    assert [k["text"] for k in kept] == ["全員野球"]


def test_the_routing_key_normalises_but_the_display_keeps_the_human_words():
    """Comparison happens on NFKC+casefold; the display stays what the human
    typed — 'FreeToken' and 'freetoken' route the same, and neither is rewritten."""
    kept, _ = verify_callsigns(["FreeToken"], [ev("USER", "FreeTokenの件ね")])
    assert kept[0]["text"] == "FreeToken"
    from distill_kura.distill.gate import cue_key
    assert cue_key("ＦｒｅｅＴｏｋｅｎ") == cue_key("freetoken")


# ── the ledger (attacks 8–10) and its derived nature ────────────────────────

from distill_kura.cues import CueLedger                        # noqa: E402
from distill_kura.distill.pipeline import Distiller            # noqa: E402
from distill_kura.registry import Registry                     # noqa: E402
from distill_kura.store import Store                           # noqa: E402
from distill_kura.thinker import Models                        # noqa: E402


def _store_with_memory(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m"), label="k")
    s.init_files()
    s.remember_direct("freetoken-hybrid", "the all-hands CPU/GPU hybrid push",
                      "body", title="FreeToken hybrid")
    s.remember_direct("other-memory", "an unrelated settled thing", "body")
    return s


def _dis(s):
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    reg = Registry(stores={"m": s}, modes={}, models=models, default="m", raw={})
    return Distiller(reg, s)


def _cue_manifest(tmp_path, dis, slug, cue_text, quote="例の全員野球でいこう"):
    """A cue that became REAL: manifest (provenance) plus receipt (authority)."""
    src = tmp_path / "journal.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    cues = [{"text": cue_text, "class": "USER", "quote": quote}]
    digest = dis._write_manifest(
        {"slug": slug, "kind": "project",
         "evidence": [{"class": "USER", "text": quote}], "classes": ["USER"],
         "routing_cues": cues},
        str(src), "test:cue")
    CueLedger(dis.store).issue(memory_slug=slug, evidence_manifest=f"sha256:{digest}",
                               routing_cues=cues, accepted_via="new")
    return digest


def test_8_the_same_cue_on_two_memories_is_ambiguous_never_directed(tmp_path):
    s = _store_with_memory(tmp_path)
    d = _dis(s)
    _cue_manifest(tmp_path, d, "freetoken-hybrid", "全員野球")
    _cue_manifest(tmp_path, d, "other-memory", "全員野球",
                  quote="また全員野球の話?")          # the human reuses their own word
    led = CueLedger(s).build()
    assert led["cues"] == {} and "全員野球" in " ".join(led["ambiguous"]) or \
        led["ambiguous"], led
    assert CueLedger(s).direct("あの全員野球の続き") is None, \
        "a wrong direct route is the one unforgivable outcome"


def test_9_the_same_cue_many_times_on_one_memory_stays_unique(tmp_path):
    s = _store_with_memory(tmp_path)
    d = _dis(s)
    for _ in range(3):
        _cue_manifest(tmp_path, d, "freetoken-hybrid", "全員野球")
    led = CueLedger(s).build()
    assert list(led["cues"]) == ["全員野球"]
    assert CueLedger(s).direct("あの全員野球の続き")["slug"] == "freetoken-hybrid"


def test_10_a_manifest_pointing_outside_slug_set_is_not_routing_material(tmp_path):
    s = _store_with_memory(tmp_path)
    d = _dis(s)
    _cue_manifest(tmp_path, d, "a-memory-that-left", "全員野球")
    led = CueLedger(s).build()
    assert led["cues"] == {} and led["ambiguous"] == {}


def test_deleting_the_ledger_rebuilds_it_identically_from_provenance(tmp_path):
    s = _store_with_memory(tmp_path)
    d = _dis(s)
    _cue_manifest(tmp_path, d, "freetoken-hybrid", "全員野球")
    led = CueLedger(s)
    first = led.ledger()
    assert os.path.exists(led.path)
    os.remove(led.path)
    rebuilt = CueLedger(s).ledger()
    assert rebuilt["cues"] == first["cues"] and rebuilt["ambiguous"] == first["ambiguous"]


def test_a_tampered_manifest_is_not_routing_material(tmp_path):
    s = _store_with_memory(tmp_path)
    d = _dis(s)
    digest = _cue_manifest(tmp_path, d, "freetoken-hybrid", "全員野球")
    p = os.path.join(s.path, "_evidence", f"{digest}.json")
    man = json.load(open(p))
    man["routing_cues"].append({"text": "偽装した合言葉", "class": "USER", "quote": "x"})
    json.dump(man, open(p, "w"))            # the content no longer hashes to its name
    led = CueLedger(s).build()
    assert led["cues"] == {} and led["ambiguous"] == {}


# ── the fastpath pre-head ────────────────────────────────────────────────────

from distill_kura import fastpath                           # noqa: E402


def test_a_unique_verified_cue_routes_before_the_five_heads(tmp_path):
    s = _store_with_memory(tmp_path)
    _cue_manifest(tmp_path, _dis(s), "freetoken-hybrid", "全員野球")
    r = fastpath.lookup(s, "あの全員野球の続きなんだけど")
    assert r["verdict"] == "ok" and r["cue"] == "全員野球"
    assert r["hits"][0]["slug"] == "freetoken-hybrid"
    assert r["hits"][0]["heads"] == {"cue": "全員野球"}


def test_the_pre_head_off_changes_nothing_about_the_five_heads(tmp_path):
    s = _store_with_memory(tmp_path)
    _cue_manifest(tmp_path, _dis(s), "freetoken-hybrid", "全員野球")
    off = fastpath.lookup(s, "あの全員野球の続きなんだけど", cues=False)
    assert off["cue"] is None and off["hits"] == [], \
        "cues off = the tier zero of before, byte for byte in behaviour"


def test_an_ascii_cue_must_appear_as_a_word_not_inside_a_longer_one(tmp_path):
    s = _store_with_memory(tmp_path)
    _cue_manifest(tmp_path, _dis(s), "freetoken-hybrid", "freetoken",
                  quote="freetoken でいこう")
    assert fastpath.lookup(s, "the freetoken plan, continued")["cue"] == "freetoken"
    assert fastpath.lookup(s, "what about freetokens plural")["cue"] is None, \
        "'freetoken' inside 'freetokens' is a different word"


def test_two_different_cues_in_one_question_are_silence(tmp_path):
    s = _store_with_memory(tmp_path)
    d = _dis(s)
    _cue_manifest(tmp_path, d, "freetoken-hybrid", "全員野球")
    _cue_manifest(tmp_path, d, "other-memory", "知性備蓄部門", quote="知性備蓄部門の予算")
    assert CueLedger(s).direct("全員野球と知性備蓄部門の件") is None


def test_recall_reports_the_cue_that_answered(tmp_path):
    from distill_kura.recall import recall as do_recall
    s = _store_with_memory(tmp_path)
    _cue_manifest(tmp_path, _dis(s), "freetoken-hybrid", "全員野球")
    r = do_recall(s, None, "あの全員野球の続き", fastpath_cfg={})
    assert r["how"] == "fastpath-cue" and r["fastpath_cue"] == "全員野球"
    assert "freetoken-hybrid" in r["walked"]


# ── COVERED keeps a late-born cue; the canonical store never moves ──────────

def _covered_run(tmp_path, callsigns):
    import json as _j
    s = _store_with_memory(tmp_path)
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    reg = Registry(stores={"m": s}, modes={}, models=models, default="m",
                   raw={"distill": {"journals": {"claude": str(tmp_path)}}})
    d = Distiller(reg, s)
    j = tmp_path / "j.jsonl"
    j.write_text(_j.dumps({"type": "user", "message": {"content": [
        {"type": "text", "text": "例の全員野球でいこう、freetoken-hybrid の続き"}]}}) + "\n"
        + _j.dumps({"type": "user", "message": {"content": [
            {"type": "text", "text": "padding " * 2000}]}}) + "\n", encoding="utf-8")
    d._current_source = str(j)

    def brain(task, user, max_tokens=0):
        if "deserves to become a permanent memory" in task:
            return _j.dumps([{"topic": "all-hands", "kind": "project",
                              "why": "the hybrid freetoken-hybrid continues",
                              "callsigns": callsigns,
                              "quotes": ["[USER] 例の全員野球でいこう、freetoken-hybrid の続き"]}])
        if "actually NEW" in task:
            return "COVERED freetoken-hybrid\nalready there"
        return ""
    d.brain = brain                                   # type: ignore[method-assign]
    return s, d, d.run(chunks=1)


def test_a_covered_memory_keeps_a_late_born_callsign(tmp_path):
    """Memory novelty = COVERED, routing novelty = NEW: nothing enters the store,
    but the new shared word is provenanced against the EXISTING slug."""
    s, d, r = _covered_run(tmp_path, ["全員野球"])
    assert r["covered"] >= 1 and r["drafts"] == []
    assert set(s.slugs()) == {"freetoken-hybrid", "other-memory"}, "no memory added"
    led = CueLedger(s).build()
    assert led["cues"].get("全員野球", {}).get("slug") == "freetoken-hybrid"
    man = json.load(open(os.path.join(
        s.path, "_evidence", led["cues"]["全員野球"]["manifest"].split(":", 1)[1] + ".json")))
    assert man["memory_slug"] == "freetoken-hybrid" and man["routing_cues_version"] == 1


def test_a_callsign_never_moves_a_canonical_byte(tmp_path):
    before = (_store_bytes(tmp_path, with_cues=False))
    s, d, r = _covered_run(tmp_path, ["全員野球"])
    after = _store_bytes(tmp_path, existing=s)
    assert before == after, "memory bodies and the index are byte-identical"


def _store_bytes(tmp_path, with_cues=False, existing=None):
    s = existing or _store_with_memory(tmp_path)
    out = {}
    for slug in s.slugs():
        out[slug] = open(s.file_of(slug), "rb").read()
    out["MEMORY.md"] = open(s.index_path, "rb").read()
    return out


def test_the_callsign_machinery_is_removable_without_a_trace(tmp_path):
    """Stop-conditions #7: with cues off everywhere, recall is exactly itself —
    and the store's routing works again the moment the machinery returns."""
    s = _store_with_memory(tmp_path)
    _cue_manifest(tmp_path, _dis(s), "freetoken-hybrid", "全員野球")
    q = "あの全員野球の続き"
    assert fastpath.lookup(s, q, cues=False)["hits"] == []
    assert fastpath.lookup(s, q)["verdict"] == "ok"
