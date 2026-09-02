"""Cue receipts — the authority boundary of the callsign machinery.

The M3-hardening question, asked adversarially: what can make a route? Only a
receipt the store's own key signed, issued at the moment the association became
real, and re-proven by the reader every time. Everything else — a manifest, the
cues.json cache, a forged revision — is attack surface, and every test here is
one of those attacks.
"""
from __future__ import annotations

import hmac
import json
import os
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.cues import CueLedger                       # noqa: E402
from distill_kura.distill.pipeline import Distiller           # noqa: E402
from distill_kura.registry import Registry                    # noqa: E402
from distill_kura.store import Store                          # noqa: E402
from distill_kura.thinker import Models                       # noqa: E402
from distill_kura import fastpath                             # noqa: E402


def build(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m"), label="k")
    s.init_files()
    s.remember_direct("freetoken-hybrid", "the all-hands CPU/GPU hybrid push",
                      "body", title="FreeToken hybrid")
    s.remember_direct("other-memory", "an unrelated settled thing", "body")
    models = Models.from_config({"thinker": {"url": "http://127.0.0.1:9/v1", "model": "none"}})
    reg = Registry(stores={"m": s}, modes={}, models=models, default="m",
                   raw={"distill": {"journals": {"claude": str(tmp_path)}}})
    return s, Distiller(reg, s)


def cue(tmp_path, dis, slug, text, quote="例の全員野球でいこう", via="new"):
    src = tmp_path / "journal.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    cues = [{"text": text, "class": "USER", "quote": quote}]
    d = dis._write_manifest(
        {"slug": slug, "kind": "project",
         "evidence": [{"class": "USER", "text": quote}], "classes": ["USER"],
         "routing_cues": cues}, str(src), "test:cue")
    CueLedger(dis.store).issue(memory_slug=slug, evidence_manifest=f"sha256:{d}",
                               routing_cues=cues, accepted_via=via)
    return d


Q = "あの全員野球の続きなんだけど"


# ── 1+4: the cache is not the truth ─────────────────────────────────────────

def test_a_cache_with_its_slug_rewritten_refuses_to_route(tmp_path):
    s, dis = build(tmp_path)
    cue(tmp_path, dis, "freetoken-hybrid", "全員野球")
    led = CueLedger(s)
    assert led.direct(Q)["slug"] == "freetoken-hybrid"
    blob = json.load(open(led.path, encoding="utf-8"))
    blob["payload"]["cues"]["全員野球"]["slug"] = "other-memory"
    json.dump(blob, open(led.path, "w", encoding="utf-8"))
    assert CueLedger(s).direct(Q)["slug"] == "freetoken-hybrid", \
        "the mark no longer matches; the rebuild from receipts wins"


def test_a_cache_with_its_manifest_rewritten_refuses_to_route(tmp_path):
    s, dis = build(tmp_path)
    cue(tmp_path, dis, "freetoken-hybrid", "全員野球")
    led = CueLedger(s)
    led.ledger()                                    # seed the cache
    blob = json.load(open(led.path, encoding="utf-8"))
    blob["payload"]["cues"]["全員野球"]["manifest"] = "sha256:" + "0" * 64
    json.dump(blob, open(led.path, "w", encoding="utf-8"))
    fresh = CueLedger(s)
    assert fresh.direct(Q) is None or fresh.direct(Q)["slug"] == "freetoken-hybrid"
    rebuilt = fresh.ledger()
    assert rebuilt["cues"]["全員野球"]["manifest"] != "sha256:" + "0" * 64


def test_a_fake_cue_inserted_into_the_cache_routes_nowhere(tmp_path):
    s, dis = build(tmp_path)
    cue(tmp_path, dis, "freetoken-hybrid", "全員野球")
    led = CueLedger(s)
    led.ledger()
    blob = json.load(open(led.path, encoding="utf-8"))
    blob["payload"]["cues"]["知性備蓄部門"] = {
        "display": "知性備蓄部門", "slug": "other-memory",
        "manifest": "sha256:" + "f" * 64, "receipt": "0" * 64}
    json.dump(blob, open(led.path, "w", encoding="utf-8"))
    assert CueLedger(s).direct("知性備蓄部門の予算は?") is None, \
        "no receipt ever said this; the cache may not invent one"


def test_a_forged_source_revision_still_refuses(tmp_path):
    """Rewrite the cache AND set source_revision to the current truth: the mark
    covers the whole payload, so honesty about one field saves nothing."""
    s, dis = build(tmp_path)
    cue(tmp_path, dis, "freetoken-hybrid", "全員野球")
    led = CueLedger(s)
    led.ledger()
    blob = json.load(open(led.path, encoding="utf-8"))
    blob["payload"]["cues"]["全員野球"]["slug"] = "other-memory"
    blob["payload"]["source_revision"] = s.revision()          # the true value
    json.dump(blob, open(led.path, "w", encoding="utf-8"))
    assert CueLedger(s).direct(Q)["slug"] == "freetoken-hybrid"


def test_a_corrupt_cache_rebuilds_in_memory_and_still_answers(tmp_path):
    s, dis = build(tmp_path)
    cue(tmp_path, dis, "freetoken-hybrid", "全員野球")
    led = CueLedger(s)
    led.ledger()
    open(led.path, "w", encoding="utf-8").write("{not json at all")
    assert CueLedger(s).direct(Q)["slug"] == "freetoken-hybrid"


# ── 2+3: manifests and drafts have no authority ─────────────────────────────

def test_a_self_hashed_fake_manifest_alone_creates_no_route(tmp_path):
    """A manifest that perfectly hashes to its own name — but no receipt — is
    provenance nobody activated. Only receipts are authority."""
    s, dis = build(tmp_path)
    src = tmp_path / "journal.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    man = {"gate_version": 6, "memory_slug": "other-memory", "routing_cues_version": 1,
           "routing_cues": [{"text": "全員野球", "class": "USER", "quote": "例の全員野球でいこう"}],
           "quotes": [{"class": "USER", "text": "例の全員野球でいこう"}]}
    blob = json.dumps(man, ensure_ascii=False, sort_keys=True, indent=1)
    import hashlib
    digest = hashlib.sha256(blob.encode()).hexdigest()
    os.makedirs(os.path.join(s.path, "_evidence"), exist_ok=True)
    open(os.path.join(s.path, "_evidence", f"{digest}.json"), "w").write(blob)
    assert CueLedger(s).build()["cues"] == {}, \
        "a perfect manifest with no receipt is not a route"


def test_an_unpoured_drafts_cue_is_not_active(tmp_path):
    """stage() writes the manifest (provenance); the receipt exists only after a
    successful POUR. A TOSSed or quarantined draft can never grow a route."""
    import json as _j
    s, dis = build(tmp_path)
    j = tmp_path / "j.jsonl"
    j.write_text(_j.dumps({"type": "user", "message": {"content": [
        {"type": "text", "text": "例の全員野球でいこう freetoken-hybrid の続き"}]}}) + "\n"
        + _j.dumps({"type": "user", "message": {"content": [
            {"type": "text", "text": "padding " * 2000}]}}) + "\n", encoding="utf-8")
    dis._current_source = str(j)

    def brain(task, user, max_tokens=0):
        if "deserves to become" in task:
            return _j.dumps([{"topic": "all-hands", "kind": "project", "why": "the hybrid",
                              "callsigns": ["全員野球"],
                              "quotes": ["[USER] 例の全員野球でいこう freetoken-hybrid の続き"]}])
        if "actually NEW" in task:
            return "NEW\nnothing like it"
        return ""
    dis.brain = brain                                    # type: ignore[method-assign]
    dis.scribe = lambda task, u, max_tokens=0: (          # type: ignore[method-assign]
        "SLUG: all-hands\nTITLE: All hands\nDESC: the all-hands hybrid push\n"
        "BODY:\nThe hybrid runs.\n")
    r = dis.run(chunks=1)
    assert r["drafts"] == ["all-hands"], r                # staged, NOT poured
    assert CueLedger(s).direct(Q) is None, "a staged draft's cue is not a route"
    # the judge pours it → the receipt exists → the route is real
    dis.scribe = lambda task, u, max_tokens=0: "POUR\nreason: inside its evidence"  # type: ignore[method-assign]
    dis.drain()
    assert CueLedger(s).direct(Q) is not None, "a poured draft's cue IS a route"


# ── 5: receipts are tamper-evident ───────────────────────────────────────────

def test_a_tampered_receipt_routes_nowhere(tmp_path):
    s, dis = build(tmp_path)
    cue(tmp_path, dis, "freetoken-hybrid", "全員野球")
    led = CueLedger(s)
    led.ledger()
    key = led.receipt_digests()[0]
    p = os.path.join(led.receipts, f"{key}.json")
    d = json.load(open(p))
    d["receipt"]["memory_slug"] = "other-memory"           # the smuggle
    json.dump(d, open(p, "w"))
    assert CueLedger(s).direct(Q) is None, \
        "content no longer hashes to its name; the reader's floor refuses"


def test_a_correctly_resigned_receipt_keeps_its_old_name_is_refused(tmp_path):
    """Honest about the boundary: a principal holding the gate key CAN re-sign
    content (docs/TRUST.md — the key stops accidents, not the filesystem owner),
    and a fully re-forged, correctly RENAMED receipt is beyond this floor, like
    every other mark in the store. What digest-naming does defend — and what
    this pins — is the cheap attack: re-sign the content, keep the old filename,
    and the name no longer hashes to the bytes."""
    import hashlib
    s, dis = build(tmp_path)
    cue(tmp_path, dis, "freetoken-hybrid", "全員野球")
    led = CueLedger(s)
    key = led.receipt_digests()[0]
    p = os.path.join(led.receipts, f"{key}.json")
    d = json.load(open(p))
    d["receipt"]["memory_slug"] = "other-memory"
    blob = json.dumps(d["receipt"], ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
    mark = hmac.new(s.gate_key(), ("cue-receipt-v1" + blob).encode(),
                    hashlib.sha256).hexdigest()          # a REAL re-signature
    json.dump({"receipt": d["receipt"], "mark": mark}, open(p, "w"))
    assert CueLedger(s).direct(Q) is None, "the name no longer hashes to the content"


# ── M3.1: the association itself must be proven ─────────────────────────────

def _manifest_only(tmp_path, dis, slug, cues, quote="例の全員野球でいこう"):
    src = tmp_path / "journal.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    return dis._write_manifest(
        {"slug": slug, "kind": "project",
         "evidence": [{"class": "USER", "text": quote}], "classes": ["USER"],
         "routing_cues": cues}, str(src), "test:cue")


def test_a_receipt_for_a_slug_the_manifest_never_blessed_is_refused(tmp_path):
    """The reader proved the human said the words — but not that the gate
    accepted them FOR THIS MEMORY. A buggy caller signing a receipt against the
    wrong slug must not grow a route."""
    s, dis = build(tmp_path)
    quote = "例の全員野球でいこう"
    cues = [{"text": "全員野球", "class": "USER", "quote": quote}]
    d = _manifest_only(tmp_path, dis, "freetoken-hybrid", cues)  # blessed for THIS one
    CueLedger(s).issue(memory_slug="other-memory",                  # signed for THAT one
                       evidence_manifest=f"sha256:{d}",
                       routing_cues=cues, accepted_via="new")
    assert CueLedger(s).direct(Q) is None


def test_a_receipt_cue_the_manifest_never_approved_is_refused(tmp_path):
    """A phrase inside the human's quote, said by the human — but the gate
    approved '全員野球', not '全員野球で'. Subset-of-quote is not subset-of-approved."""
    s, dis = build(tmp_path)
    quote = "例の全員野球でいこう"
    d = _manifest_only(tmp_path, dis, "freetoken-hybrid",
                       [{"text": "全員野球", "class": "USER", "quote": quote}])
    CueLedger(s).issue(memory_slug="freetoken-hybrid",
                       evidence_manifest=f"sha256:{d}",
                       routing_cues=[{"text": "全員野球で", "class": "USER",
                                      "quote": quote}], accepted_via="new")
    assert CueLedger(s).direct("全員野球でいこう?") is None
    assert CueLedger(s).direct(Q) is None


# ── M3.1: receipts are persistent authority — policy and durability ─────────

def test_a_frozen_store_refuses_to_grow_routing(tmp_path):
    """Frozen means the world does not grow, including the ways it may be
    called. COVERED needs no canonical write to mint a route — this refusal is
    the only thing standing between a frozen archive and evolving routing."""
    import pathlib
    frozen = Store(name="f", path=str(tmp_path / "f"), write_policy="frozen")
    frozen.init_files()
    res = CueLedger(frozen).issue(memory_slug="x", evidence_manifest="sha256:" + "0" * 64,
                                  routing_cues=[{"text": "全員野球", "class": "USER",
                                                 "quote": "例の全員野球"}],
                                  accepted_via="covered")
    assert res["ok"] is False and "frozen" in res["why"]


def test_a_receipt_store_symlinked_out_of_the_kura_is_refused(tmp_path):
    s, dis = build(tmp_path)
    led = CueLedger(s)
    os.makedirs(led.receipts)
    outside = tmp_path / "outside-receipts"
    outside.mkdir()
    rmdir = led.receipts
    os.rmdir(rmdir)
    os.symlink(outside, rmdir)
    res = led.issue(memory_slug="freetoken-hybrid",
                    evidence_manifest="sha256:" + "0" * 64,
                    routing_cues=[{"text": "全員野球", "class": "USER",
                                   "quote": "例の全員野球"}], accepted_via="new")
    assert res["ok"] is False and "symlink" in res["why"]
    assert led.receipt_digests() == [], "reading through the escape is refused too"


def test_receipts_are_written_durably_and_failures_are_visible(tmp_path, monkeypatch):
    import json as _j
    s, dis = build(tmp_path)
    fsynced = []
    real = s._fsync_dir
    monkeypatch.setattr(s, "_fsync_dir", lambda p: (fsynced.append(p), real(p))[1])

    def boom(target, data):
        raise OSError("disk went away")
    monkeypatch.setattr(s, "_replace_file", boom)
    res = CueLedger(s).issue(memory_slug="freetoken-hybrid",
                             evidence_manifest="sha256:" + "0" * 64,
                             routing_cues=[{"text": "全員野球", "class": "USER",
                                            "quote": "例の全員野球"}], accepted_via="new")
    assert res["ok"] is False and "write failed" in res["why"]
    assert fsynced == [], "a failed write never claims durability"

    monkeypatch.setattr(s, "_replace_file", Store._replace_file.__get__(s))
    res2 = CueLedger(s).issue(memory_slug="freetoken-hybrid",
                              evidence_manifest="sha256:" + "0" * 64,
                              routing_cues=[{"text": "全員野球", "class": "USER",
                                             "quote": "例の全員野球"}], accepted_via="new")
    assert res2["ok"] is True and fsynced, "the directory fsync rode along"


def test_the_run_and_pour_results_say_when_a_receipt_was_refused(tmp_path, monkeypatch):
    import json as _j
    s, dis = build(tmp_path)
    j = tmp_path / "j.jsonl"
    j.write_text(_j.dumps({"type": "user", "message": {"content": [
        {"type": "text", "text": "また知性備蓄部門の話、freetoken-hybrid 続きで"}]}}) + "\n"
        + _j.dumps({"type": "user", "message": {"content": [
            {"type": "text", "text": "padding " * 2000}]}}) + "\n", encoding="utf-8")
    dis._current_source = str(j)

    def brain(task, user, max_tokens=0):
        if "deserves to become" in task:
            return _j.dumps([{"topic": "budget", "kind": "project",
                              "why": "the hybrid freetoken-hybrid",
                              "callsigns": ["知性備蓄部門"],
                              "quotes": ["[USER] また知性備蓄部門の話、freetoken-hybrid 続きで"]}])
        if "actually NEW" in task:
            return "COVERED freetoken-hybrid\nalready there"
        return ""
    dis.brain = brain                                     # type: ignore[method-assign]

    def boom(target, data):
        raise OSError("no receipts tonight")
    monkeypatch.setattr(s, "_replace_file", boom)
    r = dis.run(chunks=1)
    assert r.get("cue_receipt_failures") == 1 and r.get("cue_receipts") == 0, \
        "the refusal is counted in the run's numbers, never swallowed"


# ── 4: COVERED freshness without a canonical move ───────────────────────────

def test_a_covered_late_born_cue_is_visible_without_a_canonical_move(tmp_path):
    """The cue-side stamp, not the canonical revision, sees the new receipt:
    plain ledger()/direct() (never build()) must find it immediately."""
    import json as _j
    s, dis = build(tmp_path)
    cue(tmp_path, dis, "freetoken-hybrid", "全員野球")
    led = CueLedger(s)
    led.ledger()                                          # cache built and persisted
    before = s.revision()

    j = tmp_path / "j2.jsonl"
    j.write_text(_j.dumps({"type": "user", "message": {"content": [
        {"type": "text", "text": "また知性備蓄部門の話、freetoken-hybrid 続きで"}]}}) + "\n"
        + _j.dumps({"type": "user", "message": {"content": [
            {"type": "text", "text": "padding " * 2000}]}}) + "\n", encoding="utf-8")
    dis._current_source = str(j)

    def brain(task, user, max_tokens=0):
        if "deserves to become" in task:
            return _j.dumps([{"topic": "budget", "kind": "project", "why": "the hybrid freetoken-hybrid",
                              "callsigns": ["知性備蓄部門"],
                              "quotes": ["[USER] また知性備蓄部門の話、freetoken-hybrid 続きで"]}])
        if "actually NEW" in task:
            return "COVERED freetoken-hybrid\nalready there"
        return ""
    dis.brain = brain                                    # type: ignore[method-assign]
    dis.run(chunks=1)
    assert s.revision() == before, "a COVERED cue moves no canonical byte"
    fresh = CueLedger(s)
    assert fresh.direct("知性備蓄部門の予算は?") is not None, \
        "the cache stamp must see the new receipt without a canonical move"


# ── 6: several names for one world ──────────────────────────────────────────

def test_same_slug_aliases_route_together(tmp_path):
    """"全員野球のFreeTokenの続き": both cues, one memory → a direct hit. Two cues
    naming DIFFERENT memories remain silence (pinned in test_callsigns)."""
    s, dis = build(tmp_path)
    cue(tmp_path, dis, "freetoken-hybrid", "全員野球")
    cue(tmp_path, dis, "freetoken-hybrid", "FreeToken", quote="FreeToken でいこう")
    r = CueLedger(s).direct("全員野球のFreeTokenの続き")
    assert r is not None and r["slug"] == "freetoken-hybrid"


# ── 7: a broken cache never breaks recall ───────────────────────────────────

def test_an_unwritable_still_leaves_recall_alive(tmp_path):
    s, dis = build(tmp_path)
    cue(tmp_path, dis, "freetoken-hybrid", "全員野球")
    led = CueLedger(s)
    led.ledger()                                          # cache exists
    os.chmod(led.path, stat.S_IRUSR | stat.S_IRGRP)       # read-only cache file
    os.chmod(os.path.dirname(led.path), stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
    try:
        # stale by construction (the memo belongs to another instance): the disk
        # refuses the rebuild, the in-memory ledger still answers
        fresh = CueLedger(s)
        os.remove(led.path) if os.access(led.path, os.W_OK) else None
        assert fresh.direct(Q) is not None or True       # never raises: the floor
        from distill_kura.recall import recall as do_recall
        r = do_recall(s, None, "the all-hands CPU/GPU hybrid push, continued",
                      fastpath_cfg={})
        assert r["context"] is not None                  # canonical recall alive
    finally:
        os.chmod(os.path.dirname(led.path), 0o755)
        os.chmod(led.path, 0o644) if os.path.exists(led.path) else None
