from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from sts_lightspeed_baseline import card_name, import_sts, run_trial


class FakeCardId(Enum):
    BASH = 1


def test_card_name_preserves_upgrade_state() -> None:
    assert (
        card_name(SimpleNamespace(id=FakeCardId.BASH, upgraded=False, upgrade_count=0)) == "BASH"
    )
    assert card_name(SimpleNamespace(id=FakeCardId.BASH, upgraded=True, upgrade_count=1)) == "BASH+"
    assert (
        card_name(SimpleNamespace(id=FakeCardId.BASH, upgraded=True, upgrade_count=2)) == "BASH+2"
    )
    assert card_name(SimpleNamespace(id=FakeCardId.BASH, upgraded=True)) == "BASH+"


def test_lightspeed_external_trial_from_env() -> None:
    module_dir = os.environ.get("STS_LIGHTSPEED_MODULE_DIR")
    if not module_dir:
        pytest.skip("set STS_LIGHTSPEED_MODULE_DIR to run the external sts_lightspeed smoke test")

    sts = import_sts(Path(module_dir))
    result = run_trial(
        sts=sts,
        seed=1,
        ascension=0,
        simulation_count=1,
        boss_multiplier=1.0,
        deck_policy="heuristic",
        max_decisions=500,
    )

    assert result.outcome in {"PLAYER_LOSS", "PLAYER_VICTORY"}
    assert result.floor >= 0
    assert result.deck
    assert result.reward_metrics_available
