from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Sequence

import pytest

from combat_api import (
    ActionOutcome,
    ActionType,
    CardState,
    CardType,
    CombatAction,
    CombatMetrics,
    CombatPhase,
    CombatPolicy,
    CombatResult,
    CombatSimulator,
    CombatState,
    MonsterState,
    PlayerState,
    PotionState,
    legal_player_actions,
)


def make_state() -> CombatState:
    return CombatState(
        player=PlayerState(
            hp=68,
            max_hp=80,
            block=5,
            energy=3,
            potions=(PotionState(slot=0, name="Fire Potion"), PotionState(slot=1, name=None)),
        ),
        monsters=(
            MonsterState(
                monster_id="jaw-worm-0",
                name="Jaw Worm",
                hp=40,
                max_hp=40,
                intent="attack",
                intent_damage=11,
            ),
            MonsterState(
                monster_id="cultist-0",
                name="Cultist",
                hp=0,
                max_hp=48,
                targetable=True,
            ),
        ),
        hand=(
            CardState(
                card_id="strike_r",
                name="Strike",
                instance_id="card-1",
                card_type=CardType.ATTACK,
                cost=1,
            ),
            CardState(
                card_id="bash",
                name="Bash",
                instance_id="card-2",
                card_type=CardType.ATTACK,
                cost=2,
                upgraded=True,
            ),
            CardState(
                card_id="impervious",
                name="Impervious",
                instance_id="card-3",
                card_type=CardType.SKILL,
                cost=4,
            ),
        ),
    )


def test_combat_state_derives_terminal_and_target_information() -> None:
    state = make_state()

    assert not state.is_terminal
    assert state.incoming_damage == 11
    assert [monster.monster_id for monster in state.living_monsters] == ["jaw-worm-0"]
    assert [monster.monster_id for monster in state.targetable_monsters] == ["jaw-worm-0"]
    assert state.card_in_hand("card-2").display_name == "Bash+"
    assert state.card_in_hand("missing") is None
    assert state.monster_by_id("cultist-0").is_alive is False


def test_value_objects_are_frozen_and_tuple_backed() -> None:
    state = make_state()

    with pytest.raises(FrozenInstanceError):
        state.player.hp = 1

    assert isinstance(state.hand, tuple)
    assert isinstance(state.player.potions, tuple)


def test_legal_player_actions_are_conservative_and_typed() -> None:
    state = make_state()

    actions = legal_player_actions(state)

    assert CombatAction.play_card("card-1", "jaw-worm-0") in actions
    assert CombatAction.play_card("card-2", "jaw-worm-0") in actions
    assert CombatAction.play_card("card-3", "jaw-worm-0") not in actions
    assert CombatAction.use_potion(0, "jaw-worm-0") in actions
    assert CombatAction.use_potion(1, "jaw-worm-0") not in actions
    assert actions[-1] == CombatAction.end_turn()
    assert {action.action_type for action in actions} == {
        ActionType.PLAY_CARD,
        ActionType.USE_POTION,
        ActionType.END_TURN,
    }


def test_enum_like_strings_are_normalized_for_adapter_payloads() -> None:
    state = CombatState(
        player=PlayerState(hp=40, max_hp=80, energy=1),
        monsters=(
            MonsterState(
                monster_id="cultist-0",
                name="Cultist",
                hp=48,
                max_hp=48,
                intent_damage=6,
            ),
        ),
        hand=(
            CardState(
                card_id="strike_r",
                name="Strike",
                instance_id="card-1",
                card_type="attack",
                cost=1,
            ),
        ),
        phase="player_turn",
        result="ongoing",
    )
    action = CombatAction("play_card", card_instance_id="card-1", monster_id="cultist-0")

    assert not state.is_terminal
    assert state.phase is CombatPhase.PLAYER_TURN
    assert state.result is CombatResult.ONGOING
    assert state.hand[0].card_type is CardType.ATTACK
    assert action.action_type is ActionType.PLAY_CARD
    assert action.action_key() == "play:card-1:cultist-0"
    assert legal_player_actions(state) == (action, CombatAction.end_turn())


def test_no_legal_player_actions_after_terminal_or_enemy_turn() -> None:
    state = make_state()

    assert legal_player_actions(replace(state, phase=CombatPhase.ENEMY_TURN)) == ()
    assert (
        legal_player_actions(
            replace(state, phase=CombatPhase.COMPLETE, result=CombatResult.PLAYER_VICTORY)
        )
        == ()
    )


def test_negative_potion_slot_is_rejected() -> None:
    with pytest.raises(ValueError, match="potion slot"):
        CombatAction.use_potion(-1)


def test_future_binding_action_shapes_cover_potion_discard_and_card_select() -> None:
    discard = CombatAction.discard_potion(2)
    select = CombatAction.select_card(
        "card-9",
        stable_id="select:discard:card-9",
        metadata={"source_zone": "discard", "task": "headbutt"},
    )

    assert discard.action_type is ActionType.DISCARD_POTION
    assert discard.action_key() == "discard_potion:2"
    assert select.action_type is ActionType.SELECT_CARD
    assert select.action_key() == "select:discard:card-9"
    assert select.to_json()["metadata"] == {"source_zone": "discard", "task": "headbutt"}


class FakeSimulator:
    def legal_actions(self, state: CombatState) -> Sequence[CombatAction]:
        return legal_player_actions(state)

    def step(self, state: CombatState, action: CombatAction) -> ActionOutcome:
        after = replace(
            state,
            phase=CombatPhase.COMPLETE,
            result=CombatResult.PLAYER_VICTORY,
            action_history=state.action_history + (action,),
            metrics=replace(state.metrics, cards_played=state.metrics.cards_played + 1),
        )
        return ActionOutcome(
            before=state,
            action=action,
            after=after,
            metrics_delta=CombatMetrics(cards_played=1),
            events=("fake action applied",),
        )

    def clone(self, state: CombatState) -> CombatState:
        return replace(state)

    def advance_to_decision(self, state: CombatState) -> CombatState:
        return state


class FirstActionPolicy:
    def choose_action(
        self, state: CombatState, legal_actions: Sequence[CombatAction]
    ) -> CombatAction:
        if not legal_actions:
            raise ValueError(f"no legal actions for {state.phase.value}")
        return legal_actions[0]


def test_protocols_accept_adapter_and_policy_without_sts_lightspeed() -> None:
    simulator = FakeSimulator()
    policy = FirstActionPolicy()
    state = make_state()

    assert isinstance(simulator, CombatSimulator)
    assert isinstance(policy, CombatPolicy)

    cloned = simulator.clone(state)
    action = policy.choose_action(cloned, simulator.legal_actions(cloned))
    outcome = simulator.step(cloned, action)

    assert outcome.is_terminal
    assert outcome.result is CombatResult.PLAYER_VICTORY
    assert outcome.after.action_history == (action,)
    assert outcome.metrics_delta.cards_played == 1
