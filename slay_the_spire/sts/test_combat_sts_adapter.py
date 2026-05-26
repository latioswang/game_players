from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from combat_api import ActionType, CardType, CombatAction, CombatResult, CombatSimulator


adapter_module = pytest.importorskip(
    "combat_sts_adapter",
    reason=(
        "future payload-based sts_lightspeed adapter is not implemented yet; "
        "expected module is combat_sts_adapter"
    ),
)
StsLightspeedCombatAdapter = adapter_module.StsLightspeedPayloadCombatBackend
StaleCombatAction = getattr(adapter_module, "StaleCombatAction", ValueError)
CombatBackendSimulator = adapter_module.CombatBackendSimulator


class FakeCombatHandle:
    """Small deterministic stand-in for a future sts_lightspeed CombatHandle."""

    def __init__(
        self,
        *,
        monster_hp: int = 40,
        player_hp: int = 70,
        turn: int = 1,
        decision_id: int = 0,
        hand: list[dict[str, Any]] | None = None,
        metrics: dict[str, Any] | None = None,
        applied_commands: list[str] | None = None,
    ) -> None:
        self.monster_hp = monster_hp
        self.player_hp = player_hp
        self.turn = turn
        self.decision_id = decision_id
        self.hand = _starting_hand() if hand is None else deepcopy(hand)
        self._metrics = _empty_metrics() if metrics is None else deepcopy(metrics)
        self.applied_commands = [] if applied_commands is None else list(applied_commands)

    def observe(self) -> dict[str, Any]:
        return {
            "player": {
                "hp": self.player_hp,
                "max_hp": 80,
                "block": 0,
                "energy": 3,
                "powers": [],
                "relics": ["Burning Blood"],
                "potions": [{"slot": 0, "name": "Fire Potion"}],
            },
            "monsters": [
                {
                    "monster_id": "jaw-worm-0",
                    "name": "Jaw Worm",
                    "hp": self.monster_hp,
                    "max_hp": 40,
                    "block": 0,
                    "intent": "attack",
                    "intent_damage": 11,
                    "powers": [],
                    "targetable": self.monster_hp > 0,
                }
            ],
            "hand": deepcopy(self.hand),
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
            "turn": self.turn,
            "phase": "player_turn",
            "result": "ongoing",
            "action_history": [],
            "metrics": self.metrics(),
            "metadata": {
                "encounter": "Jaw Worm",
                "input_state": "PLAYER_NORMAL",
                "decision_id": self._decision_key(),
            },
        }

    def legal_actions(self) -> list[dict[str, Any]]:
        actions = []
        for hand_index, card in enumerate(self.hand):
            if card["card_type"] == "attack" and card["cost"] <= 3:
                actions.append(
                    {
                        "action_type": "play_card",
                        "stable_id": f"play:{card['instance_id']}:jaw-worm-0",
                        "card_instance_id": card["instance_id"],
                        "monster_id": "jaw-worm-0",
                        "label": f"{card['name']} -> Jaw Worm",
                        "command": f"play {hand_index} 0",
                        "binding_action_id": f"{self._decision_key()}:play:{hand_index}:0",
                        "decision_id": self._decision_key(),
                    }
                )

        actions.append(
            {
                "action_type": "end_turn",
                "stable_id": "end_turn",
                "label": "End Turn",
                "command": "end",
                "binding_action_id": f"{self._decision_key()}:end",
                "decision_id": self._decision_key(),
            }
        )
        return actions

    def apply(self, action: dict[str, Any] | str) -> dict[str, Any]:
        action_payload = self._resolve_action(action)
        command = action_payload["command"]
        self.applied_commands.append(command)

        if command == "play 0 0":
            self.monster_hp -= 6
            self.hand = [card for card in self.hand if card["instance_id"] != "card-101"]
            self._metrics["cards_played"] += 1
            self._metrics["damage_dealt"] += 6
        elif command == "end":
            self.turn += 1
            self.player_hp -= 5
            self.hand = [_defend_card()]
            self._metrics["turns_taken"] += 1
            self._metrics["hp_loss"] += 5
        else:
            raise AssertionError(f"unexpected fake command: {command}")

        self.decision_id += 1
        return {
            "events": [f"applied:{command}"],
            "metrics_delta": self.metrics(),
            "observation": self.observe(),
        }

    def clone(self) -> "FakeCombatHandle":
        return type(self)(
            monster_hp=self.monster_hp,
            player_hp=self.player_hp,
            turn=self.turn,
            decision_id=self.decision_id,
            hand=self.hand,
            metrics=self._metrics,
            applied_commands=[],
        )

    def advance_to_player_decision(self) -> None:
        return None

    def is_terminal(self) -> bool:
        return False

    def metrics(self) -> dict[str, Any]:
        return deepcopy(self._metrics)

    def _resolve_action(self, action: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(action, str):
            for legal_action in self.legal_actions():
                if legal_action["stable_id"] == action or legal_action["command"] == action:
                    return legal_action
            raise AssertionError(f"fake binding received stale action string: {action}")

        current_by_binding_id = {
            legal_action["binding_action_id"]: legal_action for legal_action in self.legal_actions()
        }
        binding_action_id = action.get("binding_action_id")
        if binding_action_id not in current_by_binding_id:
            raise AssertionError(f"fake binding received stale action payload: {action}")
        return current_by_binding_id[binding_action_id]

    def _decision_key(self) -> str:
        return f"decision-{self.decision_id}"


def test_observe_and_legal_actions_convert_payloads_to_combat_api_objects() -> None:
    adapter = StsLightspeedCombatAdapter(FakeCombatHandle())

    state = adapter.observe()
    actions = tuple(adapter.legal_actions())

    assert state.player.hp == 70
    assert state.player.energy == 3
    assert state.player.potions[0].name == "Fire Potion"
    assert state.monsters[0].monster_id == "jaw-worm-0"
    assert state.monsters[0].intent_damage == 11
    assert state.hand[0].card_type is CardType.ATTACK
    assert state.metadata["sts_payload"]["metadata"]["decision_id"] == "decision-0"

    assert [action.action_key() for action in actions] == [
        "play:card-101:jaw-worm-0",
        "end_turn",
    ]
    assert actions[0].action_type is ActionType.PLAY_CARD
    assert actions[0].card_instance_id == "card-101"
    assert actions[0].monster_id == "jaw-worm-0"
    assert actions[0].metadata["command"] == "play 0 0"
    assert actions[0].metadata["sts_payload"]["binding_action_id"] == "decision-0:play:0:0"
    assert actions[1].action_type is ActionType.END_TURN
    assert actions[1].action_key() == CombatAction.end_turn().action_key()
    assert actions[1].metadata["command"] == "end"


def test_apply_forwards_current_payload_and_updates_observation_and_metrics() -> None:
    handle = FakeCombatHandle()
    adapter = StsLightspeedCombatAdapter(handle)
    strike = adapter.legal_actions()[0]

    adapter.apply(strike)

    after = adapter.observe()
    metrics = adapter.metrics()

    assert handle.applied_commands == ["play 0 0"]
    assert after.monsters[0].hp == 34
    assert [card.instance_id for card in after.hand] == []
    assert after.metadata["sts_payload"]["metadata"]["decision_id"] == "decision-1"
    assert metrics.cards_played == 1
    assert metrics.damage_dealt == 6
    assert metrics.result is CombatResult.ONGOING


def test_clone_is_independent_and_keeps_deterministic_payload_state() -> None:
    handle = FakeCombatHandle()
    adapter = StsLightspeedCombatAdapter(handle)
    clone = adapter.clone()

    clone.apply(clone.legal_actions()[0])

    assert adapter.observe().monsters[0].hp == 40
    assert adapter.metrics().damage_dealt == 0
    assert handle.applied_commands == []

    assert clone.observe().monsters[0].hp == 34
    assert clone.metrics().damage_dealt == 6
    assert [action.action_key() for action in adapter.legal_actions()] == [
        "play:card-101:jaw-worm-0",
        "end_turn",
    ]
    assert [action.action_key() for action in clone.legal_actions()] == ["end_turn"]


def test_apply_rejects_stale_action_id_from_previous_decision() -> None:
    handle = FakeCombatHandle()
    adapter = StsLightspeedCombatAdapter(handle)
    stale_strike, end_turn = adapter.legal_actions()

    adapter.apply(end_turn)

    with pytest.raises(StaleCombatAction, match="play:card-101:jaw-worm-0"):
        adapter.apply(stale_strike)

    assert handle.applied_commands == ["end"]
    assert [action.action_key() for action in adapter.legal_actions()] == ["end_turn"]


def test_apply_rejects_reused_stable_key_from_previous_decision() -> None:
    handle = FakeCombatHandle()
    adapter = StsLightspeedCombatAdapter(handle)
    _, stale_end_turn = adapter.legal_actions()

    adapter.apply(stale_end_turn)

    with pytest.raises(StaleCombatAction, match="end_turn"):
        adapter.apply(stale_end_turn)

    assert handle.applied_commands == ["end"]


def test_apply_rejects_live_actions_without_freshness_metadata() -> None:
    handle = NoFreshnessCombatHandle()
    adapter = StsLightspeedCombatAdapter(handle)
    _, end_turn = adapter.legal_actions()

    with pytest.raises(StaleCombatAction, match="end_turn"):
        adapter.apply(end_turn)

    assert handle.applied_commands == []


def test_backend_simulator_wraps_live_backend_for_search_callers() -> None:
    backend = StsLightspeedCombatAdapter(FakeCombatHandle())
    simulator = CombatBackendSimulator()
    state = simulator.wrap(backend)
    before_action_keys = [action.action_key() for action in simulator.legal_actions(state)]

    assert isinstance(simulator, CombatSimulator)
    assert state.player.hp == 70

    outcome = simulator.step(state, simulator.legal_actions(state)[0])

    assert outcome.before.monsters[0].hp == 40
    assert outcome.after.monsters[0].hp == 34
    assert outcome.metrics_delta.cards_played == 1
    assert outcome.metrics_delta.damage_dealt == 6
    assert state.monsters[0].hp == 40
    assert [action.action_key() for action in simulator.legal_actions(outcome.before)] == before_action_keys
    assert simulator.clone(outcome.before).monsters[0].hp == 40
    assert backend.metrics().damage_dealt == 0


def test_backend_state_snapshots_terminal_status_before_mutating_backend() -> None:
    backend = StsLightspeedCombatAdapter(TerminalAfterStrikeHandle())
    simulator = CombatBackendSimulator()
    state = simulator.wrap(backend)

    outcome = simulator.step(state, simulator.legal_actions(state)[0])

    assert not outcome.before.is_terminal
    assert outcome.after.is_terminal


def test_advance_to_decision_does_not_mutate_input_backend_state() -> None:
    backend = StsLightspeedCombatAdapter(AdvancingCombatHandle())
    simulator = CombatBackendSimulator()
    state = simulator.wrap(backend)

    advanced = simulator.advance_to_decision(state)

    assert state.turn == 1
    assert advanced.turn == 2
    assert simulator.clone(state).turn == 1
    assert backend.observe().turn == 1


def _starting_hand() -> list[dict[str, Any]]:
    return [
        {
            "card_id": "strike_r",
            "name": "Strike",
            "instance_id": "card-101",
            "card_type": "attack",
            "cost": 1,
            "upgraded": False,
            "tags": ["starter"],
        }
    ]


def _defend_card() -> dict[str, Any]:
    return {
        "card_id": "defend_r",
        "name": "Defend",
        "instance_id": "card-202",
        "card_type": "skill",
        "cost": 1,
        "upgraded": False,
        "tags": ["starter"],
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "result": "ongoing",
        "hp_loss": 0,
        "turns_taken": 0,
        "damage_dealt": 0,
        "potions_used": 0,
        "cards_played": 0,
        "search_time_ms": 0.0,
    }


class TerminalAfterStrikeHandle(FakeCombatHandle):
    def is_terminal(self) -> bool:
        return self.monster_hp <= 34


class AdvancingCombatHandle(FakeCombatHandle):
    def advance_to_player_decision(self) -> None:
        self.turn += 1


class NoFreshnessCombatHandle(FakeCombatHandle):
    def legal_actions(self) -> list[dict[str, Any]]:
        return [
            {
                key: value
                for key, value in action.items()
                if key not in {"binding_action_id", "decision_id"}
            }
            for action in super().legal_actions()
        ]

    def _resolve_action(self, action: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(action, str):
            return super()._resolve_action(action)
        for legal_action in self.legal_actions():
            if (
                legal_action.get("stable_id") == action.get("stable_id")
                and legal_action.get("command") == action.get("command")
            ):
                return legal_action
        raise AssertionError(f"fake binding received stale action payload: {action}")
