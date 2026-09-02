"""The constellation (M6) — the sector map a store over the honest ceiling wears.

The invariant is the whole feature: every slug in `slug_set()` lands in exactly one
sector, so the map can claim "a memory not named here may still exist inside a
sector" without any memory falling between the lines. Each test here is a way that
claim could quietly become false.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest   # noqa: E402

from distill_kura import cli, constellation                             # noqa: E402
from distill_kura.prefill import build, loom_for, trail_for             # noqa: E402
from distill_kura.registry import Registry                              # noqa: E402
from distill_kura.store import Store                                    # noqa: E402


TODAY = "2026-09-02"


def a_sector_store(tmp_path) -> Store:
    """A store with two real headings, a grouped line, and memories the index
    never names. The index is written by hand AFTER the remembers: the writer
    appends new lines at the end, which would land in the last heading's sector."""
    s = Store(name="c", path=str(tmp_path / "c"), label="sector kura")
    s.init_files()
    for slug, desc in (("doc-a", "always measure before claiming"),
                       ("doc-b", "the map outranks the trail"),
                       ("op-one", "the run finished at 43.7 t/s"),
                       ("g-a", "the bake took 796.5 seconds"),
                       ("g-b", "the restore took 0.655 seconds"),
                       ("no-line-1", "a memory with no index line at all"),
                       ("no-line-2", "another memory the index never names")):
        s.remember(slug, desc, f"observed {TODAY}")
    open(s.index_path, "w", encoding="utf-8").write(
        "# sector kura — index\n"
        "<!-- hint: entries look like - [Title](slug.md) — trigger; this comment's\n"
        "     [example](example.md) link is not a memory -->\n"
        "## Doctrine\n"
        "- [always measure](doc-a.md) — always measure before claiming\n"
        "- [map outranks](doc-b.md) — the map outranks the trail\n"
        "## Operations\n"
        "- [the run](op-one.md) — the run finished at 43.7 t/s\n"
        "- bakes — [bake](g-a.md) — 796.5 s/[restore](g-b.md) — 0.655 s\n")
    return s


def a_big_store(tmp_path, n: int = 42) -> Store:
    """Enough index that the full map alone is over a 1200-token window's ceiling."""
    s = Store(name="b", path=str(tmp_path / "b"), label="big kura")
    s.init_files()
    for i in range(n):
        s.remember(f"mem-{i:02d}-note", f"bench fixture memory number {i} of this store",
                   f"observed {TODAY}")
    return s


# ── the invariant: every memory in exactly one sector ──────────────────────

def test_every_memory_is_in_exactly_one_sector(tmp_path):
    s = a_sector_store(tmp_path)
    secs = constellation.sectors(s)
    assert [x.name for x in secs] == ["Doctrine", "Operations", "UNSECTIONED"]
    by = {x.name: x for x in secs}
    assert by["Doctrine"].slugs == ["doc-a", "doc-b"]
    # A grouped line puts EACH of its slugs in that heading's sector, once.
    assert by["Operations"].slugs == ["op-one", "g-a", "g-b"]
    # No index line at all → UNSECTIONED, sorted, last.
    assert by["UNSECTIONED"].slugs == ["no-line-1", "no-line-2"]
    check = constellation.check(s)
    assert check["invariant_ok"] is True and check["unsectioned"] == 2
    assert sum(check["sector_counts"].values()) == len(s.slug_set())


def test_the_invariant_actually_holds_everywhere(tmp_path):
    """Property, not example: sum(sector counts) == len(slug_set()) on both fixtures."""
    for mk in (a_sector_store, a_big_store):
        s = mk(tmp_path)
        check = constellation.check(s)
        assert check["covered"] == check["memories"] == len(s.slug_set())
        assert check["invariant_ok"] is True


def test_the_first_index_line_wins(tmp_path):
    """A memory named twice belongs to the heading above its FIRST line; a later
    mention moves nothing (the sector is where it was first put, not where it is
    also mentioned)."""
    s = a_sector_store(tmp_path)
    with open(s.index_path, "a", encoding="utf-8") as f:
        f.write("## Later\n- [again](doc-a.md) — mentioned again under a new heading\n")
    secs = constellation.sectors(s)
    by = {x.name: x for x in secs}
    assert by["Doctrine"].slugs.count("doc-a") == 1
    assert by["Later"].slugs == []


def test_an_index_comment_is_not_a_sector_or_a_memory(tmp_path):
    """The format hint's example link must not become a phantom memory, and the
    comment text must not become a heading (`_uncommented` runs first)."""
    s = a_sector_store(tmp_path)
    names = [x.name for x in constellation.sectors(s)]
    assert "example" not in names and "UNSECTIONED" in names
    assert all("example.md" not in x.slugs for x in constellation.sectors(s))


def test_doctor_reports_sector_and_unsectioned_counts(tmp_path):
    s = a_sector_store(tmp_path)
    assert s.doctor()["constellation"] == {"sectors": 3, "unsectioned": 2}


# ── absence semantics: the two sentences, never the map's wording ──────────

def test_the_two_sentences_carry_and_the_map_wording_does_not(tmp_path):
    s = a_sector_store(tmp_path)
    text = constellation.render(s, "sector kura")
    assert "This is a map of sectors, not individual memories." in text
    assert "A memory not named here may still exist inside a sector." in text
    assert "NOT on this list" not in text
    # "right after the frame header": BEGIN, the header line, then the sentences.
    lines = text.splitlines()
    assert lines[0] == "<<<KURA-MAP store=c>>>"
    assert lines[2] == "This is a map of sectors, not individual memories."
    assert lines[-1] == "<<<END KURA-MAP>>>"


def test_sector_lines_are_hints_not_listings(tmp_path):
    s = a_sector_store(tmp_path)
    text = constellation.render(s, "sector kura")
    line = [l for l in text.splitlines() if l.startswith("- Doctrine")][0]
    assert line == "- Doctrine — 2 memories (e.g. always measure / map outranks)"
    assert "(doc-a.md)" not in text              # slugs never leak into the sector map


def test_a_custom_header_still_wears_the_two_sentences(tmp_path):
    """The sentences are render's contract, not the default header's: a custom
    header must not silently restore the full map's absence wording."""
    s = a_sector_store(tmp_path)
    text = constellation.render(s, "sector kura", header="=== custom ===\n")
    assert "This is a map of sectors, not individual memories." in text
    assert "A memory not named here may still exist inside a sector." in text


# ── the resident modes ──────────────────────────────────────────────────────

def test_full_is_the_default_and_byte_identical_to_it(tmp_path):
    s = a_sector_store(tmp_path)
    loom = loom_for(s)
    loom.write()
    a = build(s, loom)
    b = build(s, loom, resident_mode="full")
    assert a.text == b.text and a.etag == b.etag
    assert a.stats["resident_mode"] == "full" and a.stats["map_shown"] is True


def test_auto_wears_the_map_while_it_fits(tmp_path):
    s = a_sector_store(tmp_path)
    pf = build(s, None, resident_mode="auto")
    assert pf.stats["map_shown"] is True
    assert pf.stats.get("constellation_shown") is False
    assert pf.stats["source"] in ("canonical", "woven")
    assert "(doc-a.md)" in pf.text
    assert "This is a map of sectors" not in pf.text


def test_auto_switches_to_the_constellation_over_the_ceiling(tmp_path):
    s = a_big_store(tmp_path)
    pf = build(s, None, resident_mode="auto", window_tokens=1200)
    assert pf.stats["over_ceiling"] is False        # the constellation itself fits
    assert pf.stats["map_shown"] is False           # it is NOT the map
    assert pf.stats["constellation_shown"] is True
    assert pf.stats["source"] == "constellation"
    assert "(mem-00-note.md)" not in pf.text        # no per-memory lines went out
    assert "This is a map of sectors, not individual memories." in pf.text
    assert "NOT on this list" not in pf.text


def test_constellation_mode_always_wears_the_sector_map(tmp_path):
    """Explicit `constellation` wears the sector map even while the map still fits:
    that is what the setting means."""
    s = a_sector_store(tmp_path)
    pf = build(s, None, resident_mode="constellation")
    assert pf.stats["constellation_shown"] is True and pf.stats["map_shown"] is False
    assert "- Doctrine — 2 memories" in pf.text
    assert "(doc-a.md)" not in pf.text


def test_a_constellation_over_the_ceiling_becomes_the_stub(tmp_path):
    s = a_sector_store(tmp_path)
    pf = build(s, None, resident_mode="constellation",
               window_tokens=80, hard_fraction=0.20)
    assert pf.stats["over_ceiling"] is True
    assert pf.stats["map_shown"] is False
    assert pf.stats.get("constellation_shown") is False
    assert "not the same as the memory being empty" in pf.text
    assert "Doctrine" not in pf.text                # no truncated sector list either


def test_the_trail_rides_after_the_constellation(tmp_path):
    s = a_big_store(tmp_path)
    loom = loom_for(s)
    trail = trail_for(s, {}, loom=loom)
    trail.write()
    pf = build(s, loom, trail=trail, resident_mode="constellation")
    assert pf.stats["trail"] == "appended"
    assert pf.text.index("<<<END KURA-MAP>>>") < pf.text.index("<<<KURA-TRAIL>>>")


def test_byte_stable_across_renders(tmp_path):
    """Same store revision → same bytes: the sector map carries no clock either."""
    s = a_sector_store(tmp_path)
    a = build(s, None, resident_mode="constellation")
    b = build(s, None, resident_mode="constellation")
    assert a.text == b.text and a.etag == b.etag
    assert constellation.render(s, "sector kura") == \
        constellation.render(s, "sector kura")


def test_a_volatile_header_is_refused_here_too(tmp_path):
    s = a_sector_store(tmp_path)
    with pytest.raises(ValueError, match="changes over time"):
        constellation.render(s, "sector kura", header="{label} — session 2026\n")


# ── escaping and refusal ────────────────────────────────────────────────────

def test_braces_are_escaped_on_the_way_into_a_prompt(tmp_path):
    s = a_sector_store(tmp_path)
    # The braces must sit where the sector map actually shows them: an example title.
    text = s.index_text().replace("- [always measure](doc-a.md)",
                                  "- [uses {{today}}](doc-a.md)", 1)
    open(s.index_path, "w", encoding="utf-8").write(text)
    text = constellation.render(s, "sector kura")
    assert "{{" not in text and "｛｛" in text
    pf = build(s, None, resident_mode="constellation")
    assert pf.stats["braces_escaped"] >= 1
    # The store keeps what its author wrote; only the RENDER escapes.
    assert "{{today}}" in s.index_text()


def test_build_refuses_a_resident_mode_outside_the_three(tmp_path):
    s = a_sector_store(tmp_path)
    with pytest.raises(ValueError, match="resident_mode"):
        build(s, None, resident_mode="sectors")


def test_config_refuses_a_bad_resident_mode_at_load(tmp_path):
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[stores.m]
path = "{tmp_path / 'm'}"
[stores.m.prefill]
resident_mode = "sectors"
""", encoding="utf-8")
    with pytest.raises(ValueError, match="resident_mode"):
        Registry.load(str(cfg))


def test_config_refuses_a_bad_global_resident_mode_at_load(tmp_path):
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[prefill]
resident_mode = "galaxy"
[stores.m]
path = "{tmp_path / 'm'}"
""", encoding="utf-8")
    with pytest.raises(ValueError, match="resident_mode"):
        Registry.load(str(cfg))


def test_prefill_cfg_for_passes_resident_mode_through(tmp_path):
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[prefill]
resident_mode = "auto"
[stores.m]
path = "{tmp_path / 'm'}"
[stores.other]
path = "{tmp_path / 'other'}"
""", encoding="utf-8")
    reg = Registry.load(str(cfg))
    assert reg.prefill_cfg_for(reg.stores["m"])["resident_mode"] == "auto"
    assert reg.prefill_cfg_for(reg.stores["other"])["resident_mode"] == "auto"


# ── the CLI ─────────────────────────────────────────────────────────────────

def test_kura_constellation_prints_the_table_and_the_invariant(tmp_path, capsys):
    s = a_sector_store(tmp_path)
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f'[stores.c]\npath = "{s.path}"\n', encoding="utf-8")
    rc = cli.main(["-c", str(cfg), "-s", "c", "constellation"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "- Doctrine — 2 memories (e.g. always measure / map outranks)" in out
    assert "UNSECTIONED — 2 memories" in out
    assert "invariant: sum(sector counts) = 7 memories, store holds 7 — ok" in out


def test_kura_constellation_json(tmp_path, capsys):
    import json as _json
    s = a_sector_store(tmp_path)
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f'[stores.c]\npath = "{s.path}"\n', encoding="utf-8")
    rc = cli.main(["-c", str(cfg), "-s", "c", "constellation", "--json"])
    assert rc == 0
    r = _json.loads(capsys.readouterr().out)
    assert r["invariant_ok"] is True and r["unsectioned"] == 2
    assert {"name": "Operations", "count": 3, "titles": ["the run", "bake", "restore"]} \
        in r["sectors_detail"]
