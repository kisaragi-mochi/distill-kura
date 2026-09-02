"""Typed worldline edges (M7) — derived routing state, and every way it could lie.

Each test is one floor or one honesty property: an edge invented from prose, an
edge pointing out of the store, an unevidenced supersedes smuggled through, a
tampered cache, a frozen store that grew a file anyway. The manifest is built the
way the pour path builds it — content-addressed, written by `_write_manifest` —
never a fake hash. No model anywhere.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura import edges as edges_mod                    # noqa: E402
from distill_kura.glance import glance                         # noqa: E402
from distill_kura.store import Store                           # noqa: E402
from distill_kura.trail import Trail, TRAIL_END                # noqa: E402
from distill_kura.weave import Loom                            # noqa: E402
from distill_kura.worldline import run_case                    # noqa: E402


def build(tmp_path) -> Store:
    s = Store(name="m", path=str(tmp_path / "m"), label="k")
    s.init_files()
    return s


def reg_of(store):
    from distill_kura.registry import Registry
    from distill_kura.thinker import Models
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    return Registry(stores={"m": store}, modes={}, models=models, default="m", raw={})


def pour_with_manifest(store, slug, desc, body, classes=("USER",), tmp_path=None):
    """A memory whose provenance is REAL: the manifest is written by the same
    `_write_manifest` the pour path uses (content-addressed under `_evidence/`)
    and referenced from the frontmatter through `pour_verified` — no fake hashes."""
    from distill_kura.distill.pipeline import Distiller
    jdir = str(tmp_path) if tmp_path is not None else store.still
    os.makedirs(jdir, exist_ok=True)
    src = os.path.join(jdir, "journal.jsonl")
    if not os.path.exists(src):
        with open(src, "w", encoding="utf-8") as f:
            f.write("{}\n")
    d = Distiller(reg_of(store), store)._write_manifest(
        {"slug": slug, "kind": "project",
         "evidence": [{"class": c, "text": f"{slug} said so via {c}"} for c in classes],
         "classes": list(classes)},
        src, "test:edges")
    r = store.pour_verified(slug, desc, body, meta={"evidence_manifest": f"sha256:{d}"})
    assert r.get("ok"), r
    return d


# ── the floors ───────────────────────────────────────────────────────────────

def test_an_unknown_target_is_dropped_and_counted(tmp_path):
    s = build(tmp_path)
    s.remember_direct("wishful", "points at nothing",
                      "撤退した: [[ghost-memory]]")
    d = edges_mod.derive(s)
    assert d["edges"] == []
    assert d["dropped"]["unknown-target"] == 1


def test_a_self_edge_is_dropped(tmp_path):
    s = build(tmp_path)
    s.remember_direct("loop", "points at itself", "やめた: [[loop]]")
    d = edges_mod.derive(s)
    assert d["edges"] == []
    assert d["dropped"]["self"] == 1


def test_prose_alone_invents_nothing(tmp_path):
    """The bare `(→ old-plan)` shape fires only when the name is ALSO [[linked]]
    by the source — the neighbourhood floor, not the prose, decides."""
    s = build(tmp_path)
    s.remember_direct("old-plan", "the old plan", "body")
    s.remember_direct("prose-only", "names the old plan in prose only",
                      "廃案 (→ old-plan)")
    d = edges_mod.derive(s)
    assert d["edges"] == []
    assert d["dropped"]["not-linked"] == 1
    # the same shape WITH the link elsewhere in the memory — and USER evidence —
    # is kept (the bare-arrow candidate still passes the neighbourhood floor)
    pour_with_manifest(s, "linked", "names and links the old plan",
                       "see [[old-plan]] for context\n廃案 (→ old-plan)",
                       classes=("USER",), tmp_path=tmp_path)
    d = edges_mod.derive(s)
    assert [(e["source"], e["target"], e["type"]) for e in d["edges"]] == \
        [("linked", "old-plan", "supersedes")]


def test_supersedes_without_user_evidence_is_dropped_and_counted(tmp_path):
    s = build(tmp_path)
    s.remember_direct("old-plan", "the old plan", "body")
    s.remember_direct("new-plan", "the new plan",
                      "The old approach is retired: [[old-plan]].")
    d = edges_mod.derive(s)
    assert d["edges"] == []
    assert d["unevidenced"] == 1, "a silent drop would read as 'no claim was made'"


def test_supersedes_with_a_real_user_manifest_is_kept(tmp_path):
    s = build(tmp_path)
    s.remember_direct("old-plan", "the old plan", "body")
    pour_with_manifest(s, "new-plan", "the new plan",
                       "旧案は退役: [[old-plan]]", classes=("USER",), tmp_path=tmp_path)
    d = edges_mod.derive(s)
    assert len(d["edges"]) == 1
    e = d["edges"][0]
    assert (e["source"], e["target"], e["type"], e["cue"], e["evidence"]) == \
        ("new-plan", "old-plan", "supersedes", "退役", "USER")


def test_rejected_takes_user_only_but_blocked_by_takes_tool(tmp_path):
    s = build(tmp_path)
    s.remember_direct("idea", "the idea", "body")
    s.remember_direct("blocked-work", "waiting on the idea", "blocked-by: [[idea]]")
    pour_with_manifest(s, "blocked-work", "waiting on the idea", "blocked-by: [[idea]]",
                       classes=("TOOL",), tmp_path=tmp_path)
    d = edges_mod.derive(s)
    assert [(e["target"], e["type"], e["evidence"]) for e in d["edges"]] == \
        [("idea", "blocked-by", "TOOL")]
    # the same manifest class cannot buy a `rejected` edge — that needs USER
    pour_with_manifest(s, "reject-tool", "rejecting with tool evidence",
                       "不採用: [[idea]]", classes=("TOOL",), tmp_path=tmp_path)
    d = edges_mod.derive(s)
    assert not any(e["source"] == "reject-tool" for e in d["edges"])
    assert d["unevidenced"] == 1


def test_a_missing_or_tampered_manifest_yields_no_strict_edge(tmp_path):
    s = build(tmp_path)
    s.remember_direct("idea", "the idea", "body")
    digest = pour_with_manifest(s, "with-man", "has provenance",
                                "却下: [[idea]]", classes=("USER",), tmp_path=tmp_path)
    s.remember_direct("no-man", "no provenance", "却下: [[idea]]")
    d = edges_mod.derive(s)
    assert [(e["source"], e["type"]) for e in d["edges"]] == [("with-man", "rejected")]
    assert d["unevidenced"] == 1
    # tamper with the manifest's bytes: the re-hash fails, the edge is gone
    mpath = os.path.join(s.path, "_evidence", f"{digest}.json")
    man = json.loads(open(mpath, encoding="utf-8").read())
    man["quotes"][0]["text"] = "forged after the fact"
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(json.dumps(man, ensure_ascii=False, sort_keys=True, indent=1))
    assert edges_mod.derive(s)["edges"] == []
    # BOTH strict-edge sources now count as unevidenced: no-man has no manifest,
    # with-man's manifest fails its re-hash
    assert edges_mod.derive(s)["unevidenced"] == 2


def test_continues_and_next_need_no_manifest_and_record_their_cue(tmp_path):
    s = build(tmp_path)
    s.remember_direct("older", "the earlier work", "body")
    s.remember_direct("cont", "continuation", "前の件の続き: [[older]]")
    s.remember_direct("arrowcont", "onward line", "→ [[older]]")
    s.remember_direct("nxt", "the explicit pointer", "次: [[older]]")
    d = edges_mod.derive(s)
    got = {(e["source"]): (e["type"], e["cue"], e["evidence"]) for e in d["edges"]}
    assert got["cont"] == ("continues", "続き", None)
    assert got["arrowcont"] == ("continues", "→", None)
    assert got["nxt"] == ("next", "次:", None)
    assert d["unevidenced"] == 0


def test_specificity_order_and_one_edge_per_triple(tmp_path):
    s = build(tmp_path)
    s.remember_direct("older", "the earlier work", "body")
    # rejected (cue 撤回) outranks continues (cue 継続) on the same line — and the
    # evidence floor applies to the WINNING type
    pour_with_manifest(s, "both", "two cues, one line", "撤回して継続: [[older]]",
                       classes=("USER",), tmp_path=tmp_path)
    # and the same (source, target, type) twice is still one edge
    s.remember_direct("twice", "same line twice",
                      "前の件の続き: [[older]]\nまた続き: [[older]]")
    d = edges_mod.derive(s)
    got = {e["source"]: e for e in d["edges"]}
    assert got["both"]["type"] == "rejected"
    assert sum(1 for e in d["edges"] if e["source"] == "twice") == 1
    assert d["counts"] == {"supersedes": 0, "rejected": 1, "blocked-by": 0,
                           "next": 0, "continues": 1}


def test_derivation_is_deterministic_and_bytes_stable(tmp_path):
    s = build(tmp_path)
    s.remember_direct("older", "the earlier work", "body")
    s.remember_direct("cont", "continuation", "前の件の続き: [[older]]")
    a = edges_mod.derive(s)
    b = edges_mod.derive(s)
    assert a == b
    assert edges_mod.write(s) and edges_mod.write(s)
    blob = open(edges_mod._path(s), "rb").read()
    edges_mod.write(s)
    assert open(edges_mod._path(s), "rb").read() == blob


# ── the marked cache ─────────────────────────────────────────────────────────

def test_a_tampered_file_reads_as_empty_and_is_rebuilt(tmp_path):
    s = build(tmp_path)
    s.remember_direct("older", "the earlier work", "body")
    s.remember_direct("cont", "continuation", "前の件の続き: [[older]]")
    assert edges_mod.edges_of(s, "cont")
    good = open(edges_mod._path(s), encoding="utf-8").read()
    d = json.loads(good)
    d["payload"]["edges"] = []                     # the lie: the map, emptied
    with open(edges_mod._path(s), "w", encoding="utf-8") as f:
        f.write(json.dumps(d, ensure_ascii=False))
    assert edges_mod.load(s) == {}, "a mis-marked file is not authority"
    rows = edges_mod.edges_of(s, "cont")           # rebuilt, and the file healed
    assert rows and rows[0]["type"] == "continues"
    assert edges_mod.load(s).get("edges")


def test_frozen_store_computes_in_memory_and_writes_nothing(tmp_path):
    s = build(tmp_path)
    s.remember_direct("older", "the earlier work", "body")
    s.remember_direct("cont", "continuation", "前の件の続き: [[older]]")
    frozen = Store(name="m", path=s.path, label="k", write_policy="frozen",
                   readonly=None)
    assert edges_mod.write(frozen) is False
    rows = edges_mod.edges_of(frozen, "cont")
    assert rows and rows[0]["direction"] == "out"
    assert not os.path.exists(edges_mod._path(frozen)), "frozen means nothing grows"


def test_edges_of_covers_both_directions_and_refuses_unknown(tmp_path):
    s = build(tmp_path)
    s.remember_direct("older", "the earlier work", "body")
    s.remember_direct("cont", "continuation", "前の件の続き: [[older]]")
    out_rows = edges_mod.edges_of(s, "cont")
    in_rows = edges_mod.edges_of(s, "older")
    assert out_rows == [{"type": "continues", "other": "older", "direction": "out"}]
    assert in_rows == [{"type": "continues", "other": "cont", "direction": "in"}]
    assert edges_mod.edges_of(s, "no-such-slug") == []


# ── the consumers: glance, trail, worldline ─────────────────────────────────

def test_glance_shows_relations(tmp_path):
    s = build(tmp_path)
    s.remember_direct("exl3-quantization", "EXL3 quantization", "body")
    s.remember_direct("freetoken-hybrid", "the hybrid push",
                      "前の件の続き: [[exl3-quantization]]")
    g = glance(s, "freetoken-hybrid")
    assert g["relations"] == [{"type": "continues", "other": "exl3-quantization",
                               "direction": "out"}]
    assert "RELATIONS:" in g["text"] and "→ continues exl3-quantization" in g["text"]
    g2 = glance(s, "exl3-quantization")
    assert g2["relations"] == [{"type": "continues", "other": "freetoken-hybrid",
                                "direction": "in"}]
    assert "← continues freetoken-hybrid" in g2["text"]


def _trail_store(tmp_path, fresh0_extra: str = "") -> Store:
    s = build(tmp_path)
    today = time.strftime("%Y-%m-%d")
    s.remember_direct("old-0", "an old settled thing", "settled on 2025-01-01\nbody")
    s.remember_direct("fresh-0", "the newest work item",
                      f"dated {today} — body{fresh0_extra}")
    s.remember_direct("fresh-1", "the other fresh item", f"dated {today} — body")
    return s


def test_trail_onward_line_appears_only_with_an_edge(tmp_path):
    plain = _trail_store(tmp_path / "a")
    edged = _trail_store(tmp_path / "b", fresh0_extra="\n前の件の続き: [[fresh-1]]")
    ta = Trail(plain, loom=Loom(plain, scribe=None, fresh_days=14))
    tb = Trail(edged, loom=Loom(edged, scribe=None, fresh_days=14))
    text_a, _ = ta.build()
    text_b, _ = tb.build()
    assert "↳" not in text_a
    assert "↳ fresh-0 continues → fresh-1" in text_b
    # the trail lines themselves are byte-identical; the ↳ tail is the only delta
    lines_b = [l for l in text_b.splitlines() if l.startswith("- [")]
    lines_a = [l for l in text_a.splitlines() if l.startswith("- [")]
    assert lines_a == lines_b
    assert text_b.index("↳ fresh-0") < text_b.index(TRAIL_END)


def test_trail_stays_stale_until_rebuilt_when_edges_change(tmp_path):
    s = _trail_store(tmp_path)
    t = Trail(s, loom=Loom(s, scribe=None, fresh_days=14))
    text, stamp = t.build()
    res = t.persist(text, stamp)
    assert res.get("written") and not t.is_stale()
    # a body write that adds an edge moves the revision AND the edge payload; the
    # spec hash is what proves the FOLD (the payload sha rides in the spec), which
    # is what keeps a rebuilt edge set from leaving a provably-fresh trail behind.
    spec_before = t.spec_sha256()
    s.remember_direct("fresh-1", "the other fresh item",
                      f"dated {time.strftime('%Y-%m-%d')} — body\n次: [[fresh-0]]")
    assert t.spec_sha256() != spec_before, "a changed edge set must re-price the trail"
    assert t.is_stale()


def test_worldline_edge_says_obsolete(tmp_path):
    s = build(tmp_path)
    s.remember_direct("old-abandoned-plan", "the early draft", "body")
    pour_with_manifest(s, "new-plan", "the successor",
                       "初期案は退役: [[old-abandoned-plan]]",
                       classes=("USER",), tmp_path=tmp_path)
    marked = {"id": "c1", "utterance": "whatever", "target_slugs": [],
              "acceptable_related": [], "must_not_anchor": [],
              "obsolete_slugs": ["old-abandoned-plan"], "category": "superseded-plan"}
    unmarked = {"id": "c2", "utterance": "whatever", "target_slugs": [],
                "acceptable_related": [], "must_not_anchor": [],
                "obsolete_slugs": ["never-existed"], "category": "superseded-plan"}
    tr = run_case(s, marked, "fastpath")
    assert tr["edge_says_obsolete"] is True
    tr = run_case(s, unmarked, "fastpath")
    assert tr["edge_says_obsolete"] is False
    assert tr["obsolete_branch"] is False


def test_doctor_reports_edge_counts(tmp_path):
    s = build(tmp_path)
    s.remember_direct("older", "the earlier work", "body")
    s.remember_direct("cont", "continuation", "前の件の続き: [[older]]")
    d = s.doctor()
    assert d["edges"] == {"counts": {"supersedes": 0, "rejected": 0, "blocked-by": 0,
                                     "next": 0, "continues": 1},
                          "unevidenced": 0}
    assert not os.path.exists(edges_mod._path(s)), "doctor reads, it does not write"
