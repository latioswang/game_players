from __future__ import annotations

from dataclasses import dataclass

import pytest

from combat_policy import (
    CallableCombatPolicyPrior,
    CombatActionRanker,
    MappingCombatPolicyPrior,
    UniformCombatPolicyPrior,
    default_action_key,
)


@dataclass(frozen=True)
class ToyCombatState:
    incoming_damage: int


@dataclass(frozen=True)
class ToyAction:
    key: str
    damage: int = 0
    block: int = 0


def test_mapping_prior_ranks_legal_actions_deterministically() -> None:
    actions = (
        ToyAction("strike", damage=6),
        ToyAction("defend", block=5),
        ToyAction("bash", damage=8),
    )
    ranker = CombatActionRanker(
        MappingCombatPolicyPrior(
            {
                "strike": 0.25,
                "defend": 0.5,
                "bash": 0.5,
            }
        )
    )

    ranked = ranker.rank_actions(ToyCombatState(incoming_damage=7), actions)

    assert [entry.action_key for entry in ranked] == ["bash", "defend", "strike"]
    assert [entry.rank for entry in ranked] == [1, 2, 3]
    assert [entry.prior for entry in ranked] == [0.5, 0.5, 0.25]


def test_callable_prior_can_use_state_and_action_features() -> None:
    actions = (
        ToyAction("strike", damage=6),
        ToyAction("defend", block=5),
        ToyAction("shrug", block=8),
    )
    state = ToyCombatState(incoming_damage=7)

    def safety_prior(combat_state: ToyCombatState, action: ToyAction) -> float:
        return action.damage + min(action.block, combat_state.incoming_damage)

    ranker = CombatActionRanker(CallableCombatPolicyPrior(safety_prior))

    assert [action.key for action in ranker.select_actions(state, actions, limit=2)] == [
        "shrug",
        "strike",
    ]


def test_uniform_prior_uses_action_key_tie_break() -> None:
    actions = (
        ToyAction("strike"),
        ToyAction("bash"),
        ToyAction("defend"),
    )

    ranked = CombatActionRanker(UniformCombatPolicyPrior()).rank_actions(None, actions)

    assert [entry.action.key for entry in ranked] == ["bash", "defend", "strike"]


def test_ranker_rejects_misaligned_or_invalid_prior_scores() -> None:
    class TooFewScores:
        def score_actions(self, state: object, legal_actions: tuple[ToyAction, ...]) -> tuple[float, ...]:
            del state, legal_actions
            return (1.0,)

    class InvalidScore:
        def score_actions(self, state: object, legal_actions: tuple[ToyAction, ...]) -> tuple[float, ...]:
            del state, legal_actions
            return (float("nan"), 1.0)

    actions = (ToyAction("strike"), ToyAction("defend"))

    with pytest.raises(ValueError, match="wrong number"):
        CombatActionRanker(TooFewScores()).rank_actions(None, actions)

    with pytest.raises(ValueError, match="finite"):
        CombatActionRanker(InvalidScore()).rank_actions(None, actions)


def test_default_action_key_supports_explicit_key_method() -> None:
    class MethodKeyAction:
        def action_key(self) -> str:
            return "play:card:0:target:1"

    assert default_action_key(MethodKeyAction()) == "play:card:0:target:1"
