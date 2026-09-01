"""Exit codes are a contract: 2 = "there was nothing to do", 1 = "tried and failed".

A scheduler that misreads them either spins on an empty queue or rests forever, so the
codes themselves are pinned here. No model is needed for any of this.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.cli import main          # noqa: E402
from distill_kura.store import Store       # noqa: E402


def make_store(tmp_path) -> str:
    Store(name="m", path=str(tmp_path / "m")).init_files()
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[stores.m]
path = "{tmp_path / 'm'}"
[models.thinker]
url = "http://127.0.0.1:9/v1"
model = "none"
""", encoding="utf-8")
    return str(cfg)


def test_pour_all_with_no_drafts_is_exit_two(tmp_path):
    """`--all` over zero drafts is "nothing to do", not success — the old always-0
    made a scheduler see success and never come back."""
    cfg = make_store(tmp_path)
    rc = main(["-c", cfg, "-s", "m", "distill", "pour", "--all"])
    assert rc == 2


def test_pour_of_a_slug_without_ok_is_exit_one(tmp_path):
    """A single pour that found no valid draft failed — it must not read as success."""
    cfg = make_store(tmp_path)
    rc = main(["-c", cfg, "-s", "m", "distill", "pour", "no-such-draft"])
    assert rc == 1
