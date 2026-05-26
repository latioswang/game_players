from __future__ import annotations

from dataclasses import dataclass

import pytest

from combat_search import BeamSearchCombatPlayer, BeamSearchConfig


@dataclass(frozen=True)
class ToyCombatState:
    enemy_hp: int
    player_hp: int = 10
    turn: int = 0
    history: tuple[str, ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.enemy_hp <= 0 or self.player_hp <= 0 or self.turn >= 3


@dataclass(frozen=True)
class ToyOutcome:
    before: ToyCombatState
    action: str
    after: ToyCombatState


class ToyCombatSimulator:
    def legal_actions(self, state: ToyCombatState) -> tuple[str, ...]:
        if state.is_terminal:
            return ()
        return ("strike", "big_hit", "defend")

    def step(self, state: ToyCombatState, action: str) -> ToyOutcome:
        if action == "strike":
            after = ToyCombatState(
                enemy_hp=state.enemy_hp - 3,
                player_hp=state.player_hp - 2,
                turn=state.turn + 1,
                history=(*state.history, action),
            )
        elif action == "big_hit":
            after = ToyCombatState(
                enemy_hp=state.enemy_hp - 5,
                player_hp=state.player_hp - 5,
                turn=state.turn + 1,
                history=(*state.history, action),
            )
        elif action == "defend":
            after = ToyCombatState(
                enemy_hp=state.enemy_hp,
                player_hp=state.player_hp - 1,
                turn=state.turn + 1,
                history=(*state.history, action),
            )
        else:
            raise ValueError(f"unknown action: {action}")
        return ToyOutcome(before=state, action=action, after=after)

    def clone(self, state: ToyCombatState) -> ToyCombatState:
        return state

    def advance_to_decision(self, state: ToyCombatState) -> ToyCombatState:
        return state


@dataclass
class MutableToyCombatState:
    enemy_hp: int
    history: list[str]

    @property
    def is_terminal(self) -> bool:
        return self.enemy_hp <= 0


@dataclass(frozen=True)
class MutableToyOutcome:
    before: MutableToyCombatState
    action: str
    after: MutableToyCombatState


class MutableToyCombatSimulator:
    def legal_actions(self, state: MutableToyCombatState) -> tuple[str, ...]:
        if state.is_terminal:
            return ()
        return ("poke",)

    def step(self, state: MutableToyCombatState, action: str) -> MutableToyOutcome:
        if action != "poke":
            raise ValueError(f"unknown action: {action}")
        before = MutableToyCombatState(state.enemy_hp, list(state.history))
        state.enemy_hp -= 1
        state.history.append(action)
        return MutableToyOutcome(before=before, action=action, after=state)

    def clone(self, state: MutableToyCombatState) -> MutableToyCombatState:
        return MutableToyCombatState(state.enemy_hp, list(state.history))

    def advance_to_decision(self, state: MutableToyCombatState) -> MutableToyCombatState:
        return state


def toy_evaluator(state: ToyCombatState) -> float:
    if state.enemy_hp <= 0:
        return 1_000.0 + state.player_hp
    if state.player_hp <= 0:
        return -1_000.0
    return state.player_hp - (state.enemy_hp * 3.0)


def mutable_toy_evaluator(state: MutableToyCombatState) -> float:
    if state.enemy_hp <= 0:
        return 100.0
    return -float(state.enemy_hp)


def test_beam_search_returns_best_first_action_and_line() -> None:
    player = BeamSearchCombatPlayer[ToyCombatState, str](
        simulator=ToyCombatSimulator(),
        evaluator=toy_evaluator,
        config=BeamSearchConfig(max_depth=2, beam_width=3),
    )

    result = player.search(ToyCombatState(enemy_hp=6))

    assert result is not None
    assert result.action == "strike"
    assert result.principal_variation == ("strike", "strike")
    assert result.score == pytest.approx(1_006.0)
    assert result.leaf_state.history == ("strike", "strike")
    assert result.expanded_nodes == 12


def test_beam_search_is_deterministic_when_scores_tie() -> None:
    player = BeamSearchCombatPlayer[ToyCombatState, str](
        simulator=ToyCombatSimulator(),
        evaluator=lambda _state: 0.0,
        config=BeamSearchConfig(max_depth=1, beam_width=3),
    )

    result = player.search(ToyCombatState(enemy_hp=10))

    assert result is not None
    assert result.action == "strike"
    assert result.principal_variation == ("strike",)


def test_beam_search_supports_mutating_simulator_protocol() -> None:
    player = BeamSearchCombatPlayer[MutableToyCombatState, str](
        simulator=MutableToyCombatSimulator(),
        evaluator=mutable_toy_evaluator,
        config=BeamSearchConfig(max_depth=2, beam_width=1),
    )
    root = MutableToyCombatState(enemy_hp=2, history=[])

    result = player.search(root)

    assert result is not None
    assert result.principal_variation == ("poke", "poke")
    assert result.score == pytest.approx(100.0)
    assert root.enemy_hp == 2
    assert root.history == []


def test_choose_action_returns_none_for_terminal_or_stuck_state() -> None:
    player = BeamSearchCombatPlayer[ToyCombatState, str](
        simulator=ToyCombatSimulator(),
        evaluator=toy_evaluator,
    )

    assert player.choose_action(ToyCombatState(enemy_hp=0)) is None
    assert player.choose_action(ToyCombatState(enemy_hp=4, turn=3)) is None


def test_config_rejects_non_positive_budgets() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        BeamSearchConfig(max_depth=0)
    with pytest.raises(ValueError, match="beam_width"):
        BeamSearchConfig(beam_width=0)
