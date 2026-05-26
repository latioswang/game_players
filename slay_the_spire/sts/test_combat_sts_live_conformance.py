from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


MODULE_ENTRY_POINTS = ("CharacterClass", "GameContext", "init_combat")
COMBAT_HANDLE_METHODS = (
    "observe",
    "legal_actions",
    "apply",
    "clone",
    "advance_to_player_decision",
    "is_terminal",
    "metrics",
    "exit_to_game_context",
)


def test_future_sts_lightspeed_combat_entry_points_from_env() -> None:
    sts = _import_sts_from_env()

    missing_module_entries = [name for name in MODULE_ENTRY_POINTS if not hasattr(sts, name)]
    if missing_module_entries and not _strict_combat_conformance_enabled():
        pytest.skip(
            "set STS_LIGHTSPEED_COMBAT_CONFORMANCE=1 to require future combat entry points; "
            f"missing: {', '.join(missing_module_entries)}"
        )
    assert missing_module_entries == []

    game_context = sts.GameContext(sts.CharacterClass.IRONCLAD, 1, 0)
    combat = sts.init_combat(game_context)
    _assert_combat_handle_shape(combat)

    combat.advance_to_player_decision()
    observation = combat.observe()
    legal_actions = combat.legal_actions()
    terminal = combat.is_terminal()
    metrics = combat.metrics()

    assert isinstance(observation, Mapping)
    assert isinstance(legal_actions, list)
    assert all(isinstance(action, Mapping) for action in legal_actions)
    assert isinstance(terminal, bool)
    assert isinstance(metrics, Mapping)

    clone = combat.clone()
    _assert_combat_handle_shape(clone)

    if not terminal:
        assert legal_actions
        selected_action = _deterministic_smoke_action(legal_actions)
        result = clone.apply(selected_action)
        assert isinstance(result, Mapping)
        assert combat.observe() == observation


def _import_sts_from_env() -> ModuleType:
    module_dir = os.environ.get("STS_LIGHTSPEED_MODULE_DIR")
    if not module_dir:
        pytest.skip("set STS_LIGHTSPEED_MODULE_DIR to run live sts_lightspeed combat conformance")

    path = Path(module_dir).expanduser()
    if not path.exists():
        pytest.skip(f"STS_LIGHTSPEED_MODULE_DIR does not exist: {path}")

    original_sys_path = list(sys.path)
    previous_module = sys.modules.pop("slaythespire", None)
    sys.path.insert(0, str(path.resolve()))
    try:
        module = importlib.import_module("slaythespire")
    except ImportError as exc:
        pytest.skip(f"could not import slaythespire from STS_LIGHTSPEED_MODULE_DIR: {exc}")
    finally:
        sys.path[:] = original_sys_path
        if previous_module is None:
            sys.modules.pop("slaythespire", None)
        else:
            sys.modules["slaythespire"] = previous_module
    return module


def _strict_combat_conformance_enabled() -> bool:
    return os.environ.get("STS_LIGHTSPEED_COMBAT_CONFORMANCE") == "1"


def _assert_combat_handle_shape(combat: Any) -> None:
    missing = [
        method_name
        for method_name in COMBAT_HANDLE_METHODS
        if not callable(getattr(combat, method_name, None))
    ]
    assert missing == []


def _deterministic_smoke_action(actions: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    for action in actions:
        action_id = action.get("id") or action.get("stable_id") or action.get("action_id")
        action_type = action.get("type") or action.get("action_type")
        if action_id == "end_turn" or action_type == "end_turn":
            return action
    return actions[0]
