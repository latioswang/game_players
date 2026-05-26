from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from combat_hybrid import HybridCombatConfig, HybridCombatPlayer
from combat_policy import CombatActionRanker, MappingCombatPolicyPrior, UniformCombatPolicyPrior
from combat_search import BeamSearchConfig


@dataclass(frozen=True)
class ToyCombatState:
    player_hp: int
    enemy_hp: int
    turn: int = 0
    prepared: bool = False
    history: tuple[str, ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.player_hp <= 0 or self.enemy_hp <= 0 or self.turn >= 2


@dataclass(frozen=True)
class ToyOutcome:
    before: ToyCombatState
    action: str
    after: ToyCombatState


class ToyCombatSimulator:
    def __init__(self) -> None:
        self.steps: list[str] = []

    def legal_actions(self, state: ToyCombatState) -> tuple[str, ...]:
        if state.is_terminal:
            return ()
        if not state.history:
            return ("reckless", "prepare", "defend")
        if state.prepared:
            return ("finisher", "wait")
        return ("strike", "wait")

    def step(self, state: ToyCombatState, action: str) -> ToyOutcome:
        self.steps.append(action)
        if action == "reckless":
            after = ToyCombatState(
                player_hp=state.player_hp,
                enemy_hp=0,
                turn=state.turn + 1,
                history=(*state.history, action),
            )
        elif action == "prepare":
            after = ToyCombatState(
                player_hp=state.player_hp - 3,
                enemy_hp=state.enemy_hp,
                turn=state.turn + 1,
                prepared=True,
                history=(*state.history, action),
            )
        elif action == "defend":
            after = ToyCombatState(
                player_hp=state.player_hp - 1,
                enemy_hp=state.enemy_hp,
                turn=state.turn + 1,
                history=(*state.history, action),
            )
        elif action == "finisher":
            after = ToyCombatState(
                player_hp=state.player_hp,
                enemy_hp=0,
                turn=state.turn + 1,
                prepared=state.prepared,
                history=(*state.history, action),
            )
        elif action == "strike":
            after = ToyCombatState(
                player_hp=state.player_hp - 2,
                enemy_hp=state.enemy_hp - 5,
                turn=state.turn + 1,
                prepared=state.prepared,
                history=(*state.history, action),
            )
        elif action == "wait":
            after = ToyCombatState(
                player_hp=state.player_hp - 4,
                enemy_hp=state.enemy_hp,
                turn=state.turn + 1,
                prepared=state.prepared,
                history=(*state.history, action),
            )
        else:
            raise ValueError(f"unknown action: {action}")
        return ToyOutcome(before=state, action=action, after=after)

    def clone(self, state: ToyCombatState) -> ToyCombatState:
        return state

    def advance_to_decision(self, state: ToyCombatState) -> ToyCombatState:
        return state


class ToyValueEvaluator:
    def evaluate(self, state: ToyCombatState) -> SimpleNamespace:
        if state.enemy_hp <= 0:
            return SimpleNamespace(score=1_000.0 + state.player_hp)
        if state.player_hp <= 0:
            return SimpleNamespace(score=-1_000.0)
        prepared_bonus = 8.0 if state.prepared else 0.0
        return SimpleNamespace(score=state.player_hp + prepared_bonus - (state.enemy_hp * 4.0))


def test_hybrid_prunes_with_policy_then_searches_by_leaf_value() -> None:
    simulator = ToyCombatSimulator()
    ranker = CombatActionRanker(
        MappingCombatPolicyPrior(
            {
                "prepare": 1.0,
                "defend": 0.9,
                "finisher": 1.0,
                "strike": 0.8,
                "wait": 0.0,
                "reckless": -10.0,
            },
            action_key=str,
        )
    )
    player = HybridCombatPlayer[ToyCombatState, str](
        simulator=simulator,
        ranker=ranker,
        value_evaluator=ToyValueEvaluator(),
        config=HybridCombatConfig(
            search=BeamSearchConfig(max_depth=2, beam_width=4),
            policy_action_limit=2,
        ),
    )

    result = player.search(ToyCombatState(player_hp=30, enemy_hp=10))

    assert result is not None
    assert [entry.action for entry in result.ranked_actions] == ["prepare", "defend", "reckless"]
    assert [entry.action for entry in result.pruned_actions] == ["prepare", "defend"]
    assert result.principal_variation == ("prepare", "finisher")
    assert result.action == "prepare"
    assert result.score == pytest.approx(1_027.0)
    assert "reckless" not in simulator.steps


def test_hybrid_uses_policy_order_for_deterministic_value_ties() -> None:
    player = HybridCombatPlayer[ToyCombatState, str](
        simulator=ToyCombatSimulator(),
        ranker=CombatActionRanker(UniformCombatPolicyPrior()),
        value_evaluator=lambda _state: 0.0,
        config=HybridCombatConfig(search=BeamSearchConfig(max_depth=1, beam_width=3)),
    )

    result = player.search(ToyCombatState(player_hp=30, enemy_hp=10))

    assert result is not None
    assert [entry.action for entry in result.ranked_actions] == ["defend", "prepare", "reckless"]
    assert result.action == "defend"
    assert result.principal_variation == ("defend",)


@dataclass(frozen=True)
class DictOutcome:
    before: dict[str, Any]
    action: str
    after: dict[str, Any]


class DictCombatSimulator:
    def legal_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state.get("terminal"):
            return ()
        return ("defend", "strike")

    def step(self, state: dict[str, Any], action: str) -> DictOutcome:
        after = deepcopy(state)
        history = tuple(after.get("history", ()))
        after["history"] = (*history, action)
        if action == "strike":
            after["monsters"] = [{"hp": 0, "alive": False}]
            after["terminal"] = True
            after["result"] = "victory"
        elif action == "defend":
            after["player"]["block"] = 8
        else:
            raise ValueError(f"unknown action: {action}")
        return DictOutcome(before=state, action=action, after=after)

    def clone(self, state: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(state)

    def advance_to_decision(self, state: dict[str, Any]) -> dict[str, Any]:
        return state


def test_hybrid_default_value_evaluator_scores_abstract_dict_states() -> None:
    state = {
        "player": {"hp": 32, "max_hp": 80, "block": 0},
        "monsters": [{"hp": 5, "intent_damage": 6}],
        "history": (),
    }
    player = HybridCombatPlayer[dict[str, Any], str](
        simulator=DictCombatSimulator(),
        ranker=CombatActionRanker(UniformCombatPolicyPrior()),
        config=HybridCombatConfig(search=BeamSearchConfig(max_depth=1, beam_width=2)),
    )

    result = player.search(state)

    assert result is not None
    assert result.action == "strike"
    assert result.leaf_state["result"] == "victory"
    assert result.score > 900_000
    assert state["monsters"][0]["hp"] == 5


def test_hybrid_returns_none_for_terminal_or_empty_pruned_actions() -> None:
    player = HybridCombatPlayer[ToyCombatState, str](
        simulator=ToyCombatSimulator(),
        ranker=CombatActionRanker(UniformCombatPolicyPrior()),
        value_evaluator=ToyValueEvaluator(),
        config=HybridCombatConfig(policy_action_limit=0),
    )

    assert player.choose_action(ToyCombatState(player_hp=0, enemy_hp=10)) is None
    assert player.choose_action(ToyCombatState(player_hp=30, enemy_hp=10)) is None


def test_hybrid_config_rejects_negative_policy_limit() -> None:
    with pytest.raises(ValueError, match="policy_action_limit"):
        HybridCombatConfig(policy_action_limit=-1)
