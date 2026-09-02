"""What `kura` returns to a scheduler.

Exit 2 means "there was nothing to do" and 0 means "it is done"; a watchdog that
cannot tell them apart spins on an empty queue. The mapping lives in one 300-line
main(), where a moved branch can flip 0 and 2 without any other test noticing —
these drive it in-process, no model and no subprocess. Weave and prefill's producers
are tested at function level elsewhere; what is pinned here is the mapping.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_kura import cli                  # noqa: E402
from distill_kura.store import Store          # noqa: E402


@pytest.fixture
def house(tmp_path):
    """Two stores — one normal, one frozen — and a thinker on a dead port."""
    m = Store(name="m", path=str(tmp_path / "m"), label="m")
    f = Store(name="f", path=str(tmp_path / "f"), label="f")
    m.init_files()
    f.init_files()
    m.remember("cooling", "how the ssd stays cool", "the fans went in first")
    cfg = tmp_path / "kura.toml"
    cfg.write_text(f"""
default = "m"
[stores.m]
path = "{m.path}"
[stores.f]
path = "{f.path}"
write_policy = "frozen"
[modes]
talking = "m"
[models.thinker]
url = "http://127.0.0.1:9/v1"
model = "none"
""", encoding="utf-8")
    return str(cfg), tmp_path


def run(cfg, *argv):
    return cli.main(["-c", cfg, *argv])


def test_no_subcommand_prints_help_and_succeeds(house, capsys):
    cfg, _ = house
    assert run(cfg) == 0
    assert "kura" in capsys.readouterr().out


def test_stores_and_doctor_answer_json_and_zero(house, capsys):
    cfg, _ = house
    assert run(cfg, "stores") == 0
    assert set(json.loads(capsys.readouterr().out)["stores"]) == {"m", "f"}
    assert run(cfg, "-s", "m", "doctor") == 0
    assert json.loads(capsys.readouterr().out)["store"] == "m"
    assert run(cfg, "doctor", "--all") == 0
    assert set(json.loads(capsys.readouterr().out)) == {"m", "f"}


def test_init_creates_the_store_and_prints_the_toml(house, capsys, tmp_path):
    cfg, _ = house
    assert run(cfg, "init", "n", "--path", str(tmp_path / "n")) == 0
    assert os.path.isdir(str(tmp_path / "n"))
    assert "[stores.n]" in capsys.readouterr().out


def test_an_unknown_selector_is_loud_and_names_what_exists(house):
    cfg, _ = house
    with pytest.raises(SystemExit) as e:
        run(cfg, "-s", "nope", "doctor")
    assert str(e.value.code).startswith("unknown store or mode")


def test_a_write_to_a_frozen_store_exits_one(house, capsys):
    cfg, _ = house
    assert run(cfg, "-s", "f", "remember", "slug", "a description", "a body") == 1
    assert "frozen" in capsys.readouterr().out


def test_prefill_says_two_when_the_block_is_not_what_it_should_be(house, capsys):
    """Usable text still goes out — the 2 is for the hook that can re-weave."""
    cfg, _ = house
    code = run(cfg, "-s", "m", "prefill")
    assert "KURA-MAP" in capsys.readouterr().out    # the text still goes out
    assert code == 2                                # no cloth has been woven yet


def test_a_bare_group_command_names_its_subcommands(house):
    cfg, _ = house
    for group, one in (("distill", "catchup"), ("bench", "retention"),
                       ("metrics", "richness")):
        with pytest.raises(SystemExit) as e:
            run(cfg, "-s", "m", group)
        assert one in str(e.value.code)
