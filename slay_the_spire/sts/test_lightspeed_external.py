from __future__ import annotations

import os
from pathlib import Path

import pytest

from sts_lightspeed_baseline import import_sts, run_trial


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

