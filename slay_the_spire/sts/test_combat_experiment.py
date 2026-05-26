from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from combat_experiment import (
    FixtureActionOutcome,
    FixtureCombatRecord,
    FixtureCombatStepRecord,
    RegressionConfig,
    compare_fixed_seed_policies,
    fixture_action_key,
    run_fixture_combat_policy,
    track_policy_regressions,
)
from combat_trajectory import read_jsonl


def test_experiment_module_imports_as_package_module() -> None:
    module = importlib.import_module("slay_the_spire.sts.combat_experiment")

    assert module.FixtureCombatRecord is not None


def safe_policy(state: dict[str, object], legal_actions: tuple[dict[str, object], ...]) -> dict[str, object]:
    del state
    by_key = {fixture_action_key(action): action for action in legal_actions}
    return by_key.get("defend", by_key["strike"])


def risky_policy(state: dict[str, object], legal_actions: tuple[dict[str, object], ...]) -> str:
    del state, legal_actions
    return "strike"


def slow_policy(state: dict[str, object], legal_actions: tuple[dict[str, object], ...]) -> str:
    del state, legal_actions
    return "wait"


def wait_then_strike_policy(state: dict[str, object], legal_actions: tuple[dict[str, object], ...]) -> str:
    del state
    keys = {fixture_action_key(action) for action in legal_actions}
    if "wait" in keys:
        return "wait"
    return "strike"


def test_replays_policy_against_local_fixture_record() -> None:
    result = run_fixture_combat_policy(
        two_step_record(),
        wait_then_strike_policy,
        policy_id="safe",
    )

    assert result.metrics.survived is True
    assert result.metrics.hp_loss == 2
    assert result.metrics.turns == 2
    assert result.selected_action_keys == ("wait", "strike")
    assert result.trajectory.trajectory_id == "safe:9:two-step-cultist"
    assert result.trajectory.steps[0].action is not None
    assert result.trajectory.steps[0].action.selected_action == {"id": "wait", "type": "end_turn"}
    assert result.trajectory.outcome["metrics"]["hp_loss"] == 2
    assert result.trajectory.metadata["source"] == "fixture_combat_record"


def test_fixed_seed_comparison_tracks_paired_regressions_and_summaries() -> None:
    comparison = compare_fixed_seed_policies(
        [single_step_record(seed=2, combat_id="jaw-worm"), single_step_record(seed=1, combat_id="cultist")],
        {"safe": safe_policy, "risky": risky_policy},
        regression_config=RegressionConfig(
            baseline_policy_id="safe",
            hp_loss_tolerance=0,
            turn_tolerance=0,
        ),
    )

    safe = comparison.policies["safe"].summary
    risky = comparison.policies["risky"].summary

    assert safe.combats == 2
    assert safe.survival_rate == 1.0
    assert safe.average_hp_loss == pytest.approx(3.0)
    assert risky.average_hp_loss == pytest.approx(12.0)
    assert [(regression.combat_id, regression.metric) for regression in comparison.regressions] == [
        ("cultist", "hp_loss"),
        ("cultist", "turns"),
        ("jaw-worm", "hp_loss"),
        ("jaw-worm", "turns"),
    ]
    assert comparison.regression_combat_ids("risky") == ("cultist", "jaw-worm")
    assert comparison.to_json()["policies"]["safe"]["combats"] == 2


def test_trajectory_saving_hook_writes_policy_decisions(tmp_path: Path) -> None:
    path = tmp_path / "combat-trajectories.jsonl"

    compare_fixed_seed_policies(
        [single_step_record(seed=3, combat_id="louse")],
        {"safe": safe_policy, "risky": risky_policy},
        baseline_policy_id="safe",
        trajectory_path=path,
    )

    trajectories = read_jsonl(path)

    assert [trajectory.trajectory_id for trajectory in trajectories] == [
        "risky:3:louse",
        "safe:3:louse",
    ]
    assert trajectories[0].metadata["policy_id"] == "risky"
    assert trajectories[0].steps[0].action is not None
    assert trajectories[0].steps[0].action.selected_action["id"] == "strike"


def test_track_policy_regressions_can_be_called_on_comparison_results() -> None:
    comparison = compare_fixed_seed_policies(
        [death_record()],
        {"safe": safe_policy, "risky": risky_policy},
        baseline_policy_id="safe",
    )

    regressions = track_policy_regressions(
        comparison.policies,
        RegressionConfig(baseline_policy_id="safe"),
    )

    assert [(regression.metric, regression.baseline, regression.candidate) for regression in regressions] == [
        ("survived", True, False),
        ("hp_loss", 4, 70),
    ]


def test_rejects_illegal_policy_action_and_incomplete_fixture() -> None:
    record = single_step_record(seed=1, combat_id="cultist")

    with pytest.raises(ValueError, match="illegal action"):
        run_fixture_combat_policy(record, lambda _state, _actions: "missing", policy_id="bad")

    with pytest.raises(ValueError, match="missing fixture outcomes"):
        FixtureCombatStepRecord(
            step_index=0,
            state={"turn": 1},
            legal_actions=[{"id": "strike"}, {"id": "defend"}],
            outcomes={"strike": terminal_metrics(hp_loss=3, turns=1)},
        )

    with pytest.raises(ValueError, match="non-terminal"):
        FixtureCombatRecord(
            seed=1,
            combat_id="dangling",
            steps=[
                FixtureCombatStepRecord(
                    step_index=0,
                    state={"turn": 1},
                    legal_actions=[{"id": "wait"}],
                    outcomes={"wait": FixtureActionOutcome(done=False)},
                )
            ],
        )


def single_step_record(*, seed: int, combat_id: str) -> FixtureCombatRecord:
    return FixtureCombatRecord(
        seed=seed,
        combat_id=combat_id,
        steps=[
            FixtureCombatStepRecord(
                step_index=0,
                state={"turn": 1, "phase": "player", "enemy": combat_id},
                legal_actions=[
                    {"id": "strike", "type": "play_card", "card": "Strike"},
                    {"id": "defend", "type": "play_card", "card": "Defend"},
                ],
                outcomes={
                    "strike": terminal_metrics(hp_loss=12, turns=3),
                    "defend": terminal_metrics(hp_loss=3, turns=2),
                },
            )
        ],
        metadata={"fixture": "act1-small"},
    )


def two_step_record() -> FixtureCombatRecord:
    return FixtureCombatRecord(
        seed=9,
        combat_id="two-step-cultist",
        steps=[
            FixtureCombatStepRecord(
                step_index=0,
                state={"turn": 1, "phase": "player", "enemy": "cultist"},
                legal_actions=[
                    {"id": "wait", "type": "end_turn"},
                    {"id": "strike", "type": "play_card", "card": "Strike"},
                ],
                outcomes={
                    "wait": FixtureActionOutcome(next_step_index=1, reward=0.0, done=False),
                    "strike": terminal_metrics(hp_loss=8, turns=1),
                },
            ),
            FixtureCombatStepRecord(
                step_index=1,
                state={"turn": 2, "phase": "player", "enemy": "cultist"},
                legal_actions=[
                    {"id": "strike", "type": "play_card", "card": "Strike"},
                ],
                outcomes={"strike": terminal_metrics(hp_loss=2, turns=2, reward=1.0)},
            ),
        ],
    )


def death_record() -> FixtureCombatRecord:
    return FixtureCombatRecord(
        seed=4,
        combat_id="bad-lagavulin",
        steps=[
            FixtureCombatStepRecord(
                step_index=0,
                state={"turn": 1, "phase": "player", "enemy": "lagavulin"},
                legal_actions=[
                    {"id": "strike", "type": "play_card", "card": "Strike"},
                    {"id": "defend", "type": "play_card", "card": "Defend"},
                ],
                outcomes={
                    "strike": terminal_metrics(survived=False, hp_loss=70, turns=4),
                    "defend": terminal_metrics(hp_loss=4, turns=3),
                },
            )
        ],
    )


def terminal_metrics(
    *,
    survived: bool = True,
    hp_loss: int,
    turns: int,
    reward: float | None = None,
) -> FixtureActionOutcome:
    return FixtureActionOutcome(
        metrics={
            "survived": survived,
            "hp_loss": hp_loss,
            "turns": turns,
        },
        outcome={"won": survived},
        reward=reward,
    )
