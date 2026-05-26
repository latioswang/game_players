from __future__ import annotations

import json
from pathlib import Path

import pytest

from combat_trajectory import (
    CombatActionLog,
    CombatStateLog,
    CombatStepLog,
    CombatTrajectory,
    CombatValueLog,
    read_jsonl,
    trajectory_from_json,
    trajectory_from_json_line,
    trajectory_to_json,
    trajectory_to_json_line,
    validate_json_value,
    write_jsonl,
)


def test_combat_trajectory_round_trips_as_strict_json_line() -> None:
    trajectory = sample_trajectory()

    line = trajectory_to_json_line(trajectory)
    decoded = json.loads(line)
    restored = trajectory_from_json_line(line)

    assert "\n" not in line
    assert decoded["kind"] == "sts_combat_trajectory"
    assert decoded["format_version"] == 1
    assert decoded["steps"][0]["state"]["state"]["hand"] == ["STRIKE+", "BASH"]
    assert restored == trajectory


def test_trajectory_to_json_returns_plain_python_data() -> None:
    trajectory = sample_trajectory()
    data = trajectory_to_json(trajectory)
    restored = trajectory_from_json(data)

    assert restored == trajectory
    assert isinstance(data, dict)
    assert isinstance(data["steps"], list)
    assert isinstance(data["steps"][0]["state"], dict)
    assert isinstance(data["steps"][0]["action"], dict)
    assert isinstance(data["steps"][0]["value"], dict)
    json.dumps(data, allow_nan=False)

    data["steps"][0]["state"]["state"]["player"]["hp"] = 1
    assert trajectory.steps[0].state.state["player"] == {"hp": 70, "block": 0, "energy": 3}


def test_json_values_are_copied_and_normalized() -> None:
    state = {"hand": ("STRIKE", "DEFEND"), "player": {"hp": 70}}
    log = CombatStateLog(step_index=0, turn=0, phase="player", state=state)
    state["player"]["hp"] = 1

    assert log.state["hand"] == ["STRIKE", "DEFEND"]
    assert log.state["player"] == {"hp": 70}
    assert validate_json_value({"cards": ("BASH",)}) == {"cards": ["BASH"]}


def test_rejects_non_json_safe_payloads() -> None:
    with pytest.raises(TypeError, match="state.bad"):
        CombatStateLog(step_index=0, state={"bad": Path("not-json")})

    with pytest.raises(TypeError, match="state key"):
        CombatStateLog(step_index=0, state={1: "non-string-key"})

    with pytest.raises(ValueError, match="finite"):
        CombatValueLog(step_index=0, value=float("nan"))


def test_rejects_mismatched_step_indexes() -> None:
    state = CombatStateLog(step_index=2, state={"player": {"hp": 70}})

    with pytest.raises(ValueError, match="action.step_index"):
        CombatStepLog(
            state=state,
            action=CombatActionLog(step_index=3, legal_actions=[]),
        )

    with pytest.raises(ValueError, match="value.step_index"):
        CombatStepLog(
            state=state,
            value=CombatValueLog(step_index=3, value=1.0),
        )


def test_rejects_out_of_order_trajectory_steps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CombatTrajectory(
            trajectory_id="bad-order",
            seed=1,
            combat_id="cultist",
            steps=[
                CombatStepLog(state=CombatStateLog(step_index=2, state={})),
                CombatStepLog(state=CombatStateLog(step_index=1, state={})),
            ],
        )


def test_read_write_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "combat.jsonl"
    trajectories = [
        sample_trajectory("seed-1-cultist", 1),
        sample_trajectory("seed-2-jaw-worm", 2),
    ]

    write_jsonl(path, trajectories)

    assert read_jsonl(path) == trajectories
    assert path.read_text(encoding="utf-8").count("\n") == 2


def sample_trajectory(trajectory_id: str = "seed-1-cultist", seed: int = 1) -> CombatTrajectory:
    state = CombatStateLog(
        step_index=0,
        turn=1,
        phase="player",
        state={
            "player": {"hp": 70, "block": 0, "energy": 3},
            "monsters": [{"id": "cultist", "hp": 48, "intent": "attack"}],
            "hand": ["STRIKE+", "BASH"],
            "draw_pile": ["DEFEND"],
            "discard_pile": [],
            "exhaust_pile": [],
        },
    )
    action = CombatActionLog(
        step_index=0,
        policy="beam-search",
        legal_actions=[
            {"id": "play:0:monster-0", "type": "play_card", "card": "STRIKE+", "target": 0},
            {"id": "end_turn", "type": "end_turn"},
        ],
        selected_action={"id": "play:0:monster-0", "type": "play_card", "card": "STRIKE+", "target": 0},
    )
    value = CombatValueLog(
        step_index=0,
        value=12.5,
        action_values={"play:0:monster-0": 12.5, "end_turn": -4.0},
        details={"win_probability": 0.96, "expected_hp_loss": 2.0},
    )
    return CombatTrajectory(
        trajectory_id=trajectory_id,
        seed=seed,
        combat_id="act1_floor1_cultist",
        steps=[CombatStepLog(state=state, action=action, value=value, reward=1.0, done=False)],
        outcome={"won": True, "turns": 3, "hp_loss": 2},
        metadata={"character": "IRONCLAD", "ascension": 0},
    )
