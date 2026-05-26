from __future__ import annotations

import importlib
import json
from dataclasses import replace

import pytest

from combat_api import (
    CardState,
    CardType,
    CombatAction,
    CombatPhase,
    CombatResult,
    CombatSimulator,
)
from combat_fixture import (
    CombatFixtureSimulator,
    combat_state_from_json,
    combat_state_to_json,
    make_fixture_state,
)


def test_fixture_simulator_satisfies_combat_simulator_protocol() -> None:
    assert isinstance(CombatFixtureSimulator(), CombatSimulator)


def test_fixture_module_imports_as_package_module() -> None:
    module = importlib.import_module("slay_the_spire.sts.combat_fixture")

    assert module.CombatFixtureSimulator is not None


def test_legal_actions_are_deterministic_for_simple_cards() -> None:
    simulator = CombatFixtureSimulator()
    state = make_fixture_state()

    actions = simulator.legal_actions(state)

    assert actions == (
        CombatAction.play_card("strike-0", "fixture-monster-0"),
        CombatAction.play_card("defend-0"),
        CombatAction.play_card("bash-0", "fixture-monster-0"),
        CombatAction.end_turn(),
    )


def test_strike_defend_and_bash_apply_expected_fixture_mechanics() -> None:
    simulator = CombatFixtureSimulator()
    state = make_fixture_state(player_energy=4)

    strike = simulator.step(state, CombatAction.play_card("strike-0", "fixture-monster-0"))
    assert strike.after.monsters[0].hp == 26
    assert strike.after.player.energy == 3
    assert strike.after.metrics.cards_played == 1
    assert strike.after.metrics.damage_dealt == 6

    defend = simulator.step(strike.after, CombatAction.play_card("defend-0"))
    assert defend.after.player.block == 5
    assert defend.after.player.energy == 2

    bash = simulator.step(defend.after, CombatAction.play_card("bash-0", "fixture-monster-0"))
    assert bash.after.monsters[0].hp == 18
    assert bash.after.monsters[0].powers[0].id == "Vulnerable"
    assert bash.after.monsters[0].powers[0].amount == 2
    assert bash.after.metrics.cards_played == 3
    assert bash.after.action_history == (
        CombatAction.play_card("strike-0", "fixture-monster-0"),
        CombatAction.play_card("defend-0"),
        CombatAction.play_card("bash-0", "fixture-monster-0"),
    )


def test_vulnerable_increases_later_attack_damage() -> None:
    simulator = CombatFixtureSimulator()
    state = make_fixture_state(player_energy=3, monster_hp=40)

    bashed = simulator.step(state, CombatAction.play_card("bash-0", "fixture-monster-0")).after
    struck = simulator.step(bashed, CombatAction.play_card("strike-0", "fixture-monster-0")).after

    assert bashed.monsters[0].hp == 32
    assert struck.monsters[0].hp == 23
    assert struck.metrics.damage_dealt == 17


def test_end_turn_and_advance_resolve_enemy_attack_and_deterministic_draw() -> None:
    simulator = CombatFixtureSimulator(hand_size=3)
    state = make_fixture_state()

    defended = simulator.step(state, CombatAction.play_card("defend-0")).after
    ended = simulator.step(defended, CombatAction.end_turn()).after

    assert ended.phase is CombatPhase.ENEMY_TURN
    assert simulator.legal_actions(ended) == ()
    assert ended.hand == ()
    assert [card.instance_id for card in ended.discard_pile] == [
        "defend-0",
        "strike-0",
        "bash-0",
    ]

    advanced = simulator.advance_to_decision(ended)

    assert advanced.phase is CombatPhase.PLAYER_TURN
    assert advanced.turn == 2
    assert advanced.player.hp == 79
    assert advanced.player.block == 0
    assert advanced.player.energy == 3
    assert [card.instance_id for card in advanced.hand] == [
        "strike-1",
        "defend-1",
        "defend-0",
    ]
    assert [card.instance_id for card in advanced.draw_pile] == ["strike-0", "bash-0"]
    assert advanced.discard_pile == ()
    assert advanced.metrics.hp_loss == 1
    assert advanced.metrics.turns_taken == 1


def test_fixture_marks_victory_and_death_terminal() -> None:
    simulator = CombatFixtureSimulator()
    nearly_won = make_fixture_state(monster_hp=6)

    victory = simulator.step(nearly_won, CombatAction.play_card("strike-0", "fixture-monster-0"))

    assert victory.after.is_terminal
    assert victory.after.result is CombatResult.PLAYER_VICTORY
    assert victory.after.metrics.result is CombatResult.PLAYER_VICTORY
    assert victory.after.metrics.turns_taken == 1
    assert simulator.legal_actions(victory.after) == ()

    dying = replace(
        make_fixture_state(player_hp=4, monster_intent_damage=10),
        hand=(),
        draw_pile=(),
    )
    enemy_turn = simulator.step(dying, CombatAction.end_turn()).after
    death = simulator.advance_to_decision(enemy_turn)

    assert death.is_terminal
    assert death.result is CombatResult.PLAYER_DEATH
    assert death.player.hp == 0
    assert death.metrics.hp_loss == 4


def test_clone_round_trips_json_and_does_not_alias_metadata() -> None:
    simulator = CombatFixtureSimulator()
    state = replace(make_fixture_state(), metadata={"nested": {"value": 1}})

    cloned = simulator.clone(state)

    assert cloned == state
    assert cloned is not state
    assert cloned.player is not state.player
    assert cloned.metadata is not state.metadata

    nested = cloned.metadata["nested"]
    assert isinstance(nested, dict)
    nested["value"] = 2
    assert state.metadata == {"nested": {"value": 1}}


def test_serialization_is_json_safe_and_deterministic() -> None:
    state = make_fixture_state()
    payload = combat_state_to_json(state)
    restored = combat_state_from_json(payload)

    assert restored == state
    assert json.dumps(payload, sort_keys=True) == json.dumps(
        combat_state_to_json(restored),
        sort_keys=True,
    )


def test_rejects_illegal_or_unsupported_actions() -> None:
    simulator = CombatFixtureSimulator()
    state = make_fixture_state()

    with pytest.raises(ValueError, match="illegal"):
        simulator.step(state, CombatAction.play_card("missing", "fixture-monster-0"))

    unsupported = replace(
        state,
        hand=(CardState("anger", "Anger", "anger-0", CardType.ATTACK, 0),),
    )
    assert simulator.legal_actions(unsupported) == (CombatAction.end_turn(),)
