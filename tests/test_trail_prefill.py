"""Stable Spine + Hot Trail: the map's bytes end where the trail begins.

The one property this file exists to pin (plan §8.3): when the trail changes
ALONE — the fresh window slides with time, no store write anywhere near — the
prefill's map portion must be byte-identical, because a prefix cache is lost from
the first changed byte and the map is the big stable thing.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura import prefill                                      # noqa: E402
from distill_kura.store import Store                                  # noqa: E402
from distill_kura.trail import Trail                                  # noqa: E402
from distill_kura.weave import Loom                                   # noqa: E402


def build(tmp_path):
    s = Store(name="m", path=str(tmp_path / "m"), label="k")
    s.init_files()
    today = time.strftime("%Y-%m-%d")
    for i in range(3):
        s.remember_direct(f"fresh-{i}", f"recent work item {i}", f"dated {today}")
    return s


def _loom(s):
    return Loom(s, scribe=None, fresh_days=14)


def _map_bytes(pf) -> str:
    """The prefill's text up to and including the MAP's end marker."""
    end = prefill.END + "\n"
    return pf.text.split(end, 1)[0] + end if end in pf.text else pf.text


def test_the_trail_is_appended_after_the_map(tmp_path):
    s = build(tmp_path)
    t = Trail(s, loom=_loom(s))
    assert t.write()["written"] is True
    pf = prefill.build(s, _loom(s), trail=t)
    assert pf.stats["trail"] == "appended"
    assert "<<<END KURA-MAP>>>" in pf.text and "<<<KURA-TRAIL>>>" in pf.text
    assert pf.text.index("<<<END KURA-MAP>>>") < pf.text.index("<<<KURA-TRAIL>>>")


def test_a_trail_only_change_leaves_the_map_bytes_identical(tmp_path):
    """§8.3, as the test the plan asks for: new_prefill.startswith(old_map_bytes).
    The trail is rebuilt SHORTER (a smaller budget is a legal config change; the
    fresh window sliding with time does the same) with the store untouched."""
    s = build(tmp_path)
    t = Trail(s, loom=_loom(s), trail_tokens=200)
    assert t.write()["written"] is True
    pf1 = prefill.build(s, _loom(s), trail=t)
    # a shorter budget: fewer breadcrumbs, same store, same map
    t2 = Trail(s, loom=_loom(s), trail_tokens=10)
    assert t2.write()["written"] is True
    pf2 = prefill.build(s, _loom(s), trail=t2)
    assert pf2.text.startswith(_map_bytes(pf1)), \
        "the map's bytes are the cache; a trail change must not touch them"
    assert pf2.etag != pf1.etag, "the combined etag does move with the trail"


def test_no_trail_leaves_the_prefill_exactly_as_it_was(tmp_path):
    s = build(tmp_path)
    plain = prefill.build(s, _loom(s))
    absent = prefill.build(s, _loom(s), trail=Trail(s, loom=_loom(s)))
    assert absent.text == plain.text
    assert "absent" in absent.stats["trail"]


def test_a_stale_trail_is_not_appended(tmp_path):
    s = build(tmp_path)
    t = Trail(s, loom=_loom(s))
    assert t.write()["written"] is True
    s.remember_direct("landed-after", "poured after the trail was built", "body")
    pf = prefill.build(s, _loom(s), trail=t)
    assert pf.stats.get("trail_stale") is True
    assert "<<<KURA-TRAIL>>>" not in pf.text, \
        "a stale trail would lie about the present; the map stands alone"


def test_the_cli_trail_command_builds_and_reports(tmp_path):
    from distill_kura.cli import main
    s = build(tmp_path)
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f'[stores.m]\npath = "{s.path}"\n'
                   '[models.thinker]\nurl = "http://127.0.0.1:9/v1"\n', encoding="utf-8")
    assert main(["-c", str(cfg), "-s", "m", "trail"]) == 0
    assert Trail(s, loom=_loom(s)).text_on_disk() is not None
    # nothing fresh (the window closes): nothing to say is exit 2
    cfg2 = tmp_path / "kura2.toml"
    cfg2.write_text(f'[stores.m]\npath = "{s.path}"\n'
                    '[prefill]\nfresh_days = -1\n'
                    '[models.thinker]\nurl = "http://127.0.0.1:9/v1"\n', encoding="utf-8")
    assert main(["-c", str(cfg2), "-s", "m", "trail"]) == 2


def test_trail_tokens_is_a_known_prefill_key(tmp_path):
    from distill_kura.registry import Registry
    cfg = tmp_path / "k.toml"
    cfg.write_text('[stores.m]\npath = "%s"\n[prefill]\ntrail_tokens = 120\n'
                   '[models.thinker]\nurl = "http://127.0.0.1:9/v1"\n'
                   % (tmp_path / "m"), encoding="utf-8")
    assert Registry.load(str(cfg)) is not None
