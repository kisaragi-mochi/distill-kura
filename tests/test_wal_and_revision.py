"""The store's last known crash hole: memory replace and index replace were each
atomic, the pair was not. A power loss between them left a memory nothing pointed
at — invisible to recall, found by `doctor` only after the fact.

Every test here simulates the crash by cutting `_apply` short: the WAL entry is on
disk exactly as a real power loss would leave it, and a fresh Store has to make it
right. No test reaches into replay to help it.
"""
from __future__ import annotations

import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.store import Store   # noqa: E402


class PowerLoss(RuntimeError):
    """The moment the machine dies, as an exception a test can throw."""


def make(tmp_path, name="t") -> Store:
    s = Store(name=name, path=str(tmp_path / name), label=name)
    s.init_files()
    return s


def crash_during_write(s: Store, monkeypatch, slug: str, after_memory: bool) -> None:
    """Run one `remember` and kill it inside `_apply`: either before anything
    canonical moved (`after_memory=False`) or after the memory file landed but
    before the index — the exact gap the old code could not survive."""
    def dying_apply(self, txdir, intent):
        if after_memory:
            for e in intent["files"]:
                if e["target"] != "MEMORY.md":
                    with open(os.path.join(txdir, e["payload"]), "rb") as f:
                        self._replace_file(os.path.join(self.path, e["target"]), f.read())
        raise PowerLoss()
    monkeypatch.setattr(Store, "_apply", dying_apply)
    with pytest.raises(PowerLoss):
        s.remember_direct(slug, "the trigger line", "the body")
    monkeypatch.undo()


def test_a_crash_between_memory_and_index_is_replayed_on_the_next_mutation(tmp_path, monkeypatch):
    s = make(tmp_path)
    crash_during_write(s, monkeypatch, "orphan-to-be", after_memory=True)
    # the wound, exactly as the crash left it: a memory nothing points at
    assert os.path.exists(s.file_of("orphan-to-be"))
    assert "(orphan-to-be.md)" not in s.index_text()
    assert s.revision() == 0
    # a fresh process opens the store and mutates ANYTHING — replay runs first,
    # so the leftover promise cannot clobber the newcomer's index line either
    s2 = Store(name="t", path=s.path)
    assert s2.remember_direct("unrelated", "another trigger", "b")["ok"]
    idx = s2.index_text()
    assert "(orphan-to-be.md)" in idx and "(unrelated.md)" in idx
    d = s2.doctor()
    assert d["not_in_index"] == [] and d["broken_wal"] == []
    assert s2.revision() == 2                      # the replayed write, then the new one


def test_doctor_replays_too_and_reports_what_it_replayed(tmp_path, monkeypatch):
    """Doctor is often the first thing run after a bad night; it must not describe a
    store that the next write will change under it. But a repair a doctor performs
    silently is a repair nobody learns from — so the txid is in the report."""
    s = make(tmp_path)
    crash_during_write(s, monkeypatch, "orphan-to-be", after_memory=True)
    (txid,) = os.listdir(os.path.join(s.still, "wal"))
    s2 = Store(name="t", path=s.path)
    d = s2.doctor()
    assert d["wal_replayed"] == [txid]
    assert d["not_in_index"] == [] and d["broken_wal"] == []
    assert "(orphan-to-be.md)" in s2.index_text()
    assert s2.revision() == 1 and d["revision"] == 1


def test_a_corrupt_promise_is_quarantined_never_applied(tmp_path, monkeypatch):
    """A payload whose bytes no longer match the intent's hash could be anything —
    applying it would write that anything into a canonical file with a straight
    face. Quarantined and NAMED instead; the debris is kept, because deleting it
    would delete the only evidence. Canonical files stay exactly as they were."""
    s = make(tmp_path)
    crash_during_write(s, monkeypatch, "never-lands", after_memory=False)
    wal = os.path.join(s.still, "wal")
    (txid,) = os.listdir(wal)
    with open(os.path.join(wal, txid, "payload-0"), "ab") as f:
        f.write(b"bitrot")
    s2 = Store(name="t", path=s.path)
    d = s2.doctor()
    assert d["broken_wal"] == [txid] and d["wal_replayed"] == []
    assert not os.path.exists(s2.file_of("never-lands"))
    assert "(never-lands.md)" not in s2.index_text()
    assert s2.revision() == 0
    assert os.path.isdir(os.path.join(s2.still, "wal-quarantine", txid))
    # the quarantine neither blocks later writes nor stops being reported
    assert s2.remember_direct("life-goes-on", "a fresh trigger", "b")["ok"]
    assert s2.doctor()["broken_wal"] == [txid]


def test_replaying_twice_lands_on_the_same_bytes_and_revision(tmp_path, monkeypatch):
    """Payloads are the final state, not diffs — so a crash AFTER the apply but
    before the WAL entry was cleared just replays to the bytes already there, and
    the revision does not move a second time."""
    s = make(tmp_path)
    crash_during_write(s, monkeypatch, "phoenix", after_memory=True)
    wal = os.path.join(s.still, "wal")
    (txid,) = os.listdir(wal)
    shutil.copytree(os.path.join(wal, txid), str(tmp_path / "saved-wal"))
    s2 = Store(name="t", path=s.path)
    assert s2.doctor()["wal_replayed"] == [txid]
    mem = open(s2.file_of("phoenix"), "rb").read()
    idx = open(s2.index_path, "rb").read()
    assert s2.revision() == 1
    # the clear never landed: the same promise reappears in the WAL
    shutil.copytree(str(tmp_path / "saved-wal"), os.path.join(wal, txid))
    d = s2.doctor()
    assert d["wal_replayed"] == [txid] and d["broken_wal"] == []
    assert open(s2.file_of("phoenix"), "rb").read() == mem
    assert open(s2.index_path, "rb").read() == idx
    assert s2.revision() == 1                      # max(), never a second bump


def test_revision_counts_every_canonical_mutation_and_only_those(tmp_path):
    s = make(tmp_path)
    assert s.revision() == 0                       # no file yet: an honest zero
    s.remember_direct("a", "a trigger", "body")    # memory + index line, one change
    assert s.revision() == 1
    s.pour_verified("a", "a trigger", "body two")  # rewrite through the other door
    assert s.revision() == 2
    s.annotate_direct("a", tags=["decision"])      # one-file change, still a mutation
    assert s.revision() == 3
    s.annotate_direct("a", tags=["decision"])      # no-op merge: nothing changed on disk
    assert s.revision() == 3
    s._write_index(s.index_text())                 # tidy-style index replace
    assert s.revision() == 4
    assert s.doctor()["revision"] == 4


def test_a_committed_write_leaves_no_wal_entry_behind(tmp_path):
    """The WAL is a promise in flight, not a log that grows: after a clean commit
    the entry is gone, so replay on the next write has nothing to re-apply."""
    s = make(tmp_path)
    s.remember_direct("clean", "a trigger", "body")
    assert os.listdir(os.path.join(s.still, "wal")) == []
    d = s.doctor()
    assert d["wal_replayed"] == [] and d["broken_wal"] == []


def test_every_wal_payload_and_the_intent_are_fsynced_before_the_rename(tmp_path, monkeypatch):
    """Nothing asserted that fsync is actually called, so an edit that dropped one —
    the whole point of the write-fsync-rename order — would have passed in silence."""
    s = make(tmp_path)
    synced: list[str] = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(os.readlink(f"/proc/self/fd/{fd}")),
                                                 real(fd))[1])
    s.remember_direct("fact", "the trigger line", "the body")
    files = [p for p in synced if not os.path.isdir(p)]
    # both WAL payloads (the memory and the index), the intent, and both canonical
    # replaces — each written through a tmp name, each fsynced before its rename
    assert sum(1 for p in files if "payload-" in p) == 2
    assert sum(1 for p in files if p.endswith("intent.json")) == 1
    assert sum(1 for p in files if ".tmp." in p) >= 2


def test_quarantining_the_same_txid_twice_keeps_both(tmp_path, monkeypatch):
    """The collision suffix exists so the second broken transaction under a name does
    not silently erase the first — the debris IS the evidence. Nothing exercised it."""
    s = make(tmp_path)
    crash_during_write(s, monkeypatch, "never-lands", after_memory=False)
    wal = os.path.join(s.still, "wal")
    (txid,) = os.listdir(wal)
    with open(os.path.join(wal, txid, "payload-0"), "ab") as f:
        f.write(b"bitrot")
    quarantine = os.path.join(s.still, "wal-quarantine")
    s2 = Store(name="t", path=s.path)
    assert s2.doctor()["broken_wal"] == [txid]
    # the same txid turns up broken a second time
    shutil.copytree(os.path.join(quarantine, txid), os.path.join(wal, txid))
    d = Store(name="t", path=s.path).doctor()
    kept = sorted(os.listdir(quarantine))
    assert len(kept) == 2 and kept[0] == txid
    assert kept[1].startswith(txid + ".") and kept[1][len(txid) + 1:].isdigit()
    assert d["broken_wal"] == kept


def test_fsync_dir_is_best_effort_and_never_takes_a_write_down(tmp_path, monkeypatch):
    """Some filesystems (CIFS) cannot fsync a directory. Refusing the write over that
    would trade a crash window for an outage — the atomic replace still holds."""
    assert Store._fsync_dir(str(tmp_path / "does-not-exist")) is None
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("no")))
    assert Store._fsync_dir(str(tmp_path)) is None
