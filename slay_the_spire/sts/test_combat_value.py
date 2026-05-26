from __future__ import annotations

from types import SimpleNamespace

from combat_api import CardState, CardType, CombatState as ApiCombatState, MonsterState, PlayerState
from combat_value import (
    CombatState,
    FallbackCombatAction,
    FallbackCombatState,
    FallbackMonsterState,
    HandcraftedCombatValueEvaluator,
)


def test_terminal_outcomes_dominate_score() -> None:
    evaluator = HandcraftedCombatValueEvaluator()

    victory = evaluator.evaluate(
        FallbackCombatState(
            player_hp=4,
            player_max_hp=80,
            monsters=(FallbackMonsterState(hp=0, alive=False),),
        )
    )
    defeat = evaluator.evaluate(
        FallbackCombatState(
            player_hp=0,
            player_max_hp=80,
            monsters=(FallbackMonsterState(hp=20),),
        )
    )

    assert victory.terminal == "victory"
    assert defeat.terminal == "defeat"
    assert victory.score > 900_000
    assert defeat.score < -900_000


def test_block_against_incoming_damage_is_valued_deterministically() -> None:
    evaluator = HandcraftedCombatValueEvaluator()
    monster = FallbackMonsterState(hp=20, intent_damage=12)

    unblocked = evaluator.evaluate(
        FallbackCombatState(
            player_hp=40,
            player_max_hp=80,
            player_block=0,
            energy=3,
            monsters=(monster,),
        )
    )
    blocked = evaluator.evaluate(
        FallbackCombatState(
            player_hp=40,
            player_max_hp=80,
            player_block=12,
            energy=3,
            monsters=(monster,),
        )
    )

    assert unblocked.incoming_damage == 12
    assert unblocked.expected_hp_loss == 12
    assert blocked.expected_hp_loss == 0
    assert blocked.score > unblocked.score


def test_scores_dict_backed_combat_api_shaped_state() -> None:
    evaluator = HandcraftedCombatValueEvaluator()

    value = evaluator.evaluate(
        {
            "player": {
                "hp": 40,
                "max_hp": 80,
                "block": 5,
                "powers": {"Strength": 2, "Vulnerable": 1},
            },
            "energy": 2,
            "hand": ["Strike", "Defend", "Bash"],
            "draw_pile": ["Strike", "Defend"],
            "monsters": [
                {
                    "hp": 12,
                    "intent": {"damage": 7, "hits": 2},
                    "powers": {"Weak": 1},
                }
            ],
            "turn": 2,
        }
    )

    assert value.terminal is None
    assert value.player_hp == 40
    assert value.player_block == 5
    assert value.incoming_damage == 14
    assert value.expected_hp_loss == 9
    assert value.enemy_hp == 12
    assert value.details["player_strength"] == 4.0
    assert value.details["enemy_weak"] == 1.5


def test_scores_plain_object_state_with_sts_style_names() -> None:
    evaluator = HandcraftedCombatValueEvaluator()
    state = SimpleNamespace(
        cur_hp=25,
        max_hp=70,
        player_block=3,
        energy=1,
        hand=("Strike",),
        monsters=(SimpleNamespace(current_hp=18, intent_damage=6, intent_hits=2),),
        turn_number=3,
    )

    value = evaluator.evaluate(state)

    assert value.player_hp == 25
    assert value.player_max_hp == 70
    assert value.incoming_damage == 12
    assert value.expected_hp_loss == 9
    assert value.alive_enemies == 1


def test_scores_combat_api_state_player_energy() -> None:
    evaluator = HandcraftedCombatValueEvaluator()
    value = evaluator.evaluate(
        ApiCombatState(
            player=PlayerState(hp=40, max_hp=80, block=4, energy=3),
            monsters=(
                MonsterState(
                    monster_id="jaw-worm-0",
                    name="Jaw Worm",
                    hp=40,
                    max_hp=40,
                    intent_damage=11,
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
            ),
        )
    )

    assert value.incoming_damage == 11
    assert value.expected_hp_loss == 7
    assert value.details["energy"] == 3.75


def test_non_targetable_alive_monster_is_not_terminal_victory() -> None:
    evaluator = HandcraftedCombatValueEvaluator()

    value = evaluator.evaluate(
        ApiCombatState(
            player=PlayerState(hp=40, max_hp=80),
            monsters=(
                MonsterState(
                    monster_id="lagavulin-0",
                    name="Lagavulin",
                    hp=109,
                    max_hp=109,
                    targetable=False,
                ),
            ),
        )
    )

    assert value.terminal is None
    assert value.alive_enemies == 1
    assert value.enemy_hp == 109
    assert value.score < 100_000


def test_evaluate_actions_uses_successor_states_when_available() -> None:
    evaluator = HandcraftedCombatValueEvaluator()
    current = FallbackCombatState(
        player_hp=30,
        player_max_hp=80,
        monsters=(FallbackMonsterState(hp=20, intent_damage=10),),
    )
    losing_state = FallbackCombatState(
        player_hp=0,
        player_max_hp=80,
        monsters=(FallbackMonsterState(hp=20),),
    )
    winning_state = FallbackCombatState(
        player_hp=30,
        player_max_hp=80,
        monsters=(FallbackMonsterState(hp=0, alive=False),),
    )

    best = evaluator.best_action(
        current,
        (
            FallbackCombatAction("lose", next_state=losing_state),
            FallbackCombatAction("win", next_state=winning_state),
        ),
    )

    assert best.action.name == "win"
    assert best.value.terminal == "victory"


def test_combat_state_alias_is_available_for_future_integration() -> None:
    assert CombatState is not None
