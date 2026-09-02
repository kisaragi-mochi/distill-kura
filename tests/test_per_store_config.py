"""One merge rule for every per-store table.

`prefill_cfg_for` and `fastpath_cfg_for` were the same five lines with the table
name swapped, and `Tender` hand-rolled a third copy for `[distill]`. Only the
fastpath branch had a test (tests/test_fastpath.py:160); the prefill and distill
branches were unpinned, so a drift in either would have passed the suite.

The rule is key PRESENCE: a store's explicit `0` is that store's answer, not a
request to fall back to the global value.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura.registry import Registry              # noqa: E402
from distill_kura.store import Store                    # noqa: E402


def _reg(tmp_path):
    for n in ("maker", "eq"):
        Store(name=n, path=str(tmp_path / n)).init_files()
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
[prefill]
window_tokens = 1000
budget_fraction = 0.05
[distill]
idle_min = 7
backoff_min = 20
[stores.maker]
path = "{tmp_path / 'maker'}"
[stores.eq]
path = "{tmp_path / 'eq'}"
[stores.eq.prefill]
window_tokens = 4000
[stores.eq.distill]
idle_min = 0
""", encoding="utf-8")
    return Registry.load(str(cfg))


def test_a_store_without_its_own_table_gets_the_global_one(tmp_path):
    reg = _reg(tmp_path)
    maker = reg.stores["maker"]
    assert reg.prefill_cfg_for(maker) == {"window_tokens": 1000, "budget_fraction": 0.05}
    assert reg.distill_cfg_for(maker) == {"idle_min": 7, "backoff_min": 20}


def test_a_store_table_overrides_key_by_key_and_an_explicit_zero_wins(tmp_path):
    reg = _reg(tmp_path)
    eq = reg.stores["eq"]
    pf = reg.prefill_cfg_for(eq)
    assert pf["window_tokens"] == 4000            # overridden
    assert pf["budget_fraction"] == 0.05          # merged, not replaced
    d = reg.distill_cfg_for(eq)
    assert d["idle_min"] == 0                     # presence, not truthiness
    assert d["backoff_min"] == 20
    assert reg.prefill_cfg == {"window_tokens": 1000, "budget_fraction": 0.05}   # global intact
