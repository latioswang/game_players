from __future__ import annotations

import pytest

from combat_training import (
    CombatValueTrainingExample,
    LinearCombatValueFunction,
    combat_state_features,
    combat_value_training_examples_from_trajectories,
    fit_linear_combat_value_function,
    train_linear_combat_value_function,
)
from combat_trajectory import (
    CombatStateLog,
    CombatStepLog,
    CombatTrajectory,
    CombatValueLog,
)


def test_extracts_examples_from_trajectory_value_logs() -> None:
    trajectory = CombatTrajectory(
        trajectory_id="tiny-combat",
        seed=7,
        combat_id="cultist",
        steps=[
            step(0, hp=50, value=10.0),
            CombatStepLog(state=CombatStateLog(step_index=1, state={"player": {"hp": 45}})),
            step(2, hp=40, value=-2.5),
        ],
    )

    examples = combat_value_training_examples_from_trajectories((trajectory,))

    assert len(examples) == 2
    assert examples[0].source == "tiny-combat:0"
    assert examples[0].features["player_hp"] == 50.0
    assert examples[0].target == 10.0
    assert examples[1].source == "tiny-combat:2"
    assert examples[1].target == -2.5


def test_linear_training_fits_tiny_feature_fixture_exactly() -> None:
    examples = (
        CombatValueTrainingExample(features={"x": 0.0}, target=1.0),
        CombatValueTrainingExample(features={"x": 2.0}, target=5.0),
    )

    result = fit_linear_combat_value_function(examples, feature_names=("x",), l2=0.0)

    assert result.summary.example_count == 2
    assert result.summary.feature_names == ("x",)
    assert result.summary.mean_squared_error == pytest.approx(0.0)
    assert result.model.intercept == pytest.approx(1.0)
    assert result.model.weights == {"x": pytest.approx(2.0)}
    assert result.model.predict_features({"x": 3.0}) == pytest.approx(7.0)


def test_training_from_combat_trajectories_predicts_state_values() -> None:
    trajectories = (
        CombatTrajectory(
            trajectory_id="combat-1",
            seed=1,
            combat_id="cultist",
            steps=[
                step(0, hp=10, value=7.0),
                step(1, hp=20, value=12.0),
            ],
        ),
    )

    model = train_linear_combat_value_function(
        trajectories,
        feature_names=("player_hp",),
        l2=0.0,
    )

    assert model.predict_state({"player": {"hp": 30}}) == pytest.approx(17.0)
    assert model.evaluate({"player": {"hp": 0}}) == pytest.approx(2.0)


def test_linear_model_round_trips_as_plain_json_data() -> None:
    model = LinearCombatValueFunction(weights={"player_hp": 0.5}, intercept=2.0)

    restored = LinearCombatValueFunction.from_json(model.to_json())

    assert restored == model
    assert restored.predict_features({"player_hp": 10}) == pytest.approx(7.0)


def test_combat_state_features_include_terminal_flags() -> None:
    features = combat_state_features(
        {
            "player": {"hp": 15, "max_hp": 30, "block": 3, "energy": 2},
            "monsters": [{"hp": 0, "alive": False}],
            "hand": ["Strike", "Defend"],
        }
    )

    assert features["player_hp"] == 15.0
    assert features["player_hp_ratio"] == pytest.approx(0.5)
    assert features["hand_size"] == 2.0
    assert features["terminal_victory"] == 1.0
    assert features["terminal_defeat"] == 0.0


def test_rejects_non_finite_targets_and_singular_unregularized_fit() -> None:
    with pytest.raises(ValueError, match="finite"):
        CombatValueTrainingExample(features={"x": 1.0}, target=float("nan"))

    examples = (
        CombatValueTrainingExample(features={"x": 1.0}, target=1.0),
        CombatValueTrainingExample(features={"x": 1.0}, target=2.0),
    )

    with pytest.raises(ValueError, match="singular"):
        fit_linear_combat_value_function(examples, feature_names=("x",), l2=0.0)


def step(index: int, *, hp: int, value: float) -> CombatStepLog:
    return CombatStepLog(
        state=CombatStateLog(
            step_index=index,
            state={
                "player": {"hp": hp, "max_hp": 30, "block": 0, "energy": 3},
                "monsters": [{"hp": 20, "intent": {"damage": 4}}],
            },
        ),
        value=CombatValueLog(step_index=index, value=value),
    )
