"""Deterministic local combat fixture for tests.

This module is a small Python test adapter for the ``combat_api`` simulator
protocol.  It intentionally models only a narrow, deterministic subset of
Slay the Spire combat: Strike-like attacks, Defend-like block, Bash-like attack
plus Vulnerable, end turn, enemy intent damage, and deterministic deck cycling.
It is not a replacement for a future action-level ``sts_lightspeed`` backend.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Mapping, Sequence

try:
    from .combat_api import (
        ActionOutcome,
        ActionType,
        CardState,
        CardType,
        CombatAction,
        CombatMetrics,
        CombatPhase,
        CombatPower,
        CombatResult,
        CombatState,
        JsonValue,
        MonsterState,
        PlayerState,
        PotionState,
    )
except ImportError:
    from combat_api import (
        ActionOutcome,
        ActionType,
        CardState,
        CardType,
        CombatAction,
        CombatMetrics,
        CombatPhase,
        CombatPower,
        CombatResult,
        CombatState,
        JsonValue,
        MonsterState,
        PlayerState,
        PotionState,
    )


DEFAULT_MAX_ENERGY = 3
DEFAULT_HAND_SIZE = 5


class CombatFixtureSimulator:
    """Pure deterministic combat simulator for local tests."""

    def __init__(
        self,
        *,
        max_energy: int = DEFAULT_MAX_ENERGY,
        hand_size: int = DEFAULT_HAND_SIZE,
    ) -> None:
        if max_energy < 0:
            raise ValueError("max_energy must be non-negative")
        if hand_size < 0:
            raise ValueError("hand_size must be non-negative")
        self.max_energy = max_energy
        self.hand_size = hand_size

    def legal_actions(self, state: CombatState) -> tuple[CombatAction, ...]:
        """Return legal simple-card actions plus end turn."""

        if (
            state.is_terminal
            or state.phase is not CombatPhase.PLAYER_TURN
            or not state.living_monsters
        ):
            return ()

        actions: list[CombatAction] = []
        targets = state.targetable_monsters
        for card in state.hand:
            if card.cost > state.player.energy or not _is_supported_card(card):
                continue
            if card.card_type is CardType.ATTACK:
                actions.extend(
                    CombatAction.play_card(card.instance_id, target.monster_id)
                    for target in targets
                )
            elif card.card_type is CardType.SKILL:
                actions.append(CombatAction.play_card(card.instance_id))

        actions.append(CombatAction.end_turn())
        return tuple(actions)

    def step(self, state: CombatState, action: CombatAction) -> ActionOutcome:
        """Apply one legal player action."""

        legal_by_key = {legal.action_key(): legal for legal in self.legal_actions(state)}
        if action.action_key() not in legal_by_key:
            raise ValueError(f"illegal combat fixture action: {action.action_key()}")
        action = legal_by_key[action.action_key()]

        if action.action_type is ActionType.END_TURN:
            return self._end_turn(state, action)
        if action.action_type is not ActionType.PLAY_CARD:
            raise ValueError(f"unsupported combat fixture action: {action.action_type.value}")
        return self._play_card(state, action)

    def clone(self, state: CombatState) -> CombatState:
        """Return an independent JSON round-trip clone of ``state``."""

        return combat_state_from_json(combat_state_to_json(state))

    def advance_to_decision(self, state: CombatState) -> CombatState:
        """Resolve automatic fixture phases until terminal or player decision."""

        if state.result is not CombatResult.ONGOING:
            return _complete_state(state, state.result)
        if not state.living_monsters:
            return _complete_state(state, CombatResult.PLAYER_VICTORY)
        if state.phase is not CombatPhase.ENEMY_TURN:
            return state

        incoming_damage = _enemy_intent_damage(state)
        hp_loss = min(state.player.hp, max(0, incoming_damage - state.player.block))
        hp = state.player.hp - hp_loss
        metrics = replace(state.metrics, hp_loss=state.metrics.hp_loss + hp_loss)

        if hp <= 0:
            return replace(
                state,
                player=replace(state.player, hp=0, block=0),
                phase=CombatPhase.COMPLETE,
                result=CombatResult.PLAYER_DEATH,
                metrics=replace(metrics, result=CombatResult.PLAYER_DEATH),
            )

        hand, draw_pile, discard_pile = _draw_cards(
            state.draw_pile,
            state.discard_pile,
            self.hand_size,
        )
        return replace(
            state,
            player=replace(state.player, hp=hp, block=0, energy=self.max_energy),
            hand=hand,
            draw_pile=draw_pile,
            discard_pile=discard_pile,
            turn=state.turn + 1,
            phase=CombatPhase.PLAYER_TURN,
            metrics=metrics,
        )

    def serialize_state(self, state: CombatState) -> dict[str, JsonValue]:
        """Return a JSON-safe fixture state payload."""

        return combat_state_to_json(state)

    def deserialize_state(self, payload: Mapping[str, JsonValue]) -> CombatState:
        """Build a fixture state from a JSON-safe payload."""

        return combat_state_from_json(payload)

    def _play_card(self, state: CombatState, action: CombatAction) -> ActionOutcome:
        card = state.card_in_hand(_require_value(action.card_instance_id, "card_instance_id"))
        if card is None:
            raise ValueError(f"card is not in hand: {action.card_instance_id}")

        hand = _remove_card(state.hand, card.instance_id)
        discard_pile = (*state.discard_pile, card)
        player = replace(state.player, energy=state.player.energy - card.cost)
        monsters = state.monsters
        damage_dealt = 0
        events: list[str] = []

        family = _card_family(card)
        if family == "strike":
            monsters, damage_dealt = _damage_target(state, action, _attack_damage(card, state))
            events.append(f"{card.display_name} dealt {damage_dealt} damage")
        elif family == "bash":
            monsters, damage_dealt = _damage_target(state, action, _attack_damage(card, state))
            monsters = _apply_monster_power(
                monsters,
                _require_value(action.monster_id, "monster_id"),
                "Vulnerable",
                3 if card.upgraded else 2,
            )
            events.append(f"{card.display_name} dealt {damage_dealt} damage and applied Vulnerable")
        elif family == "defend":
            block = _defend_block(card, state)
            player = replace(player, block=player.block + block)
            events.append(f"{card.display_name} gained {block} block")
        else:
            raise ValueError(f"unsupported fixture card: {card.name}")

        result = CombatResult.ONGOING
        phase = CombatPhase.PLAYER_TURN
        if not any(monster.hp > 0 for monster in monsters):
            result = CombatResult.PLAYER_VICTORY
            phase = CombatPhase.COMPLETE

        metrics = replace(
            state.metrics,
            result=result,
            cards_played=state.metrics.cards_played + 1,
            damage_dealt=state.metrics.damage_dealt + damage_dealt,
            turns_taken=state.turn if result is CombatResult.PLAYER_VICTORY else state.metrics.turns_taken,
        )
        after = replace(
            state,
            player=player,
            monsters=monsters,
            hand=hand,
            discard_pile=discard_pile,
            phase=phase,
            result=result,
            action_history=(*state.action_history, action),
            metrics=metrics,
        )
        return ActionOutcome(
            before=state,
            action=action,
            after=after,
            metrics_delta=CombatMetrics(cards_played=1, damage_dealt=damage_dealt),
            events=tuple(events),
        )

    def _end_turn(self, state: CombatState, action: CombatAction) -> ActionOutcome:
        metrics = replace(
            state.metrics,
            turns_taken=max(state.metrics.turns_taken, state.turn),
        )
        after = replace(
            state,
            player=replace(state.player, energy=0),
            hand=(),
            discard_pile=(*state.discard_pile, *state.hand),
            phase=CombatPhase.ENEMY_TURN,
            action_history=(*state.action_history, action),
            metrics=metrics,
        )
        return ActionOutcome(
            before=state,
            action=action,
            after=after,
            metrics_delta=CombatMetrics(turns_taken=1),
            events=("ended player turn",),
        )


LocalCombatFixtureSimulator = CombatFixtureSimulator


def combat_state_to_json(state: CombatState) -> dict[str, JsonValue]:
    """Return a deterministic JSON-safe payload for a combat state."""

    return json.loads(json.dumps(state.to_json(), sort_keys=True))


def combat_state_from_json(payload: Mapping[str, JsonValue]) -> CombatState:
    """Deserialize a payload produced by ``combat_state_to_json``."""

    return CombatState(
        player=_player_from_json(_mapping(payload["player"], "player")),
        monsters=tuple(_monster_from_json(item) for item in _sequence(payload["monsters"], "monsters")),
        hand=tuple(_card_from_json(item) for item in _sequence(payload.get("hand", ()), "hand")),
        draw_pile=tuple(
            _card_from_json(item) for item in _sequence(payload.get("draw_pile", ()), "draw_pile")
        ),
        discard_pile=tuple(
            _card_from_json(item)
            for item in _sequence(payload.get("discard_pile", ()), "discard_pile")
        ),
        exhaust_pile=tuple(
            _card_from_json(item)
            for item in _sequence(payload.get("exhaust_pile", ()), "exhaust_pile")
        ),
        turn=int(payload.get("turn", 1)),
        phase=payload.get("phase", CombatPhase.PLAYER_TURN.value),
        result=payload.get("result", CombatResult.ONGOING.value),
        action_history=tuple(
            _action_from_json(item)
            for item in _sequence(payload.get("action_history", ()), "action_history")
        ),
        metrics=_metrics_from_json(_mapping(payload.get("metrics", {}), "metrics")),
        metadata=dict(_mapping(payload.get("metadata", {}), "metadata")),
    )


def make_fixture_state(
    *,
    player_hp: int = 80,
    player_max_hp: int = 80,
    player_energy: int = DEFAULT_MAX_ENERGY,
    monster_hp: int = 32,
    monster_intent_damage: int = 6,
) -> CombatState:
    """Create a minimal deterministic Strike/Defend/Bash fixture combat."""

    return CombatState(
        player=PlayerState(hp=player_hp, max_hp=player_max_hp, energy=player_energy),
        monsters=(
            MonsterState(
                monster_id="fixture-monster-0",
                name="Fixture Monster",
                hp=monster_hp,
                max_hp=monster_hp,
                intent="attack",
                intent_damage=monster_intent_damage,
            ),
        ),
        hand=(
            CardState("strike_r", "Strike", "strike-0", CardType.ATTACK, 1),
            CardState("defend_r", "Defend", "defend-0", CardType.SKILL, 1),
            CardState("bash", "Bash", "bash-0", CardType.ATTACK, 2),
        ),
        draw_pile=(
            CardState("strike_r", "Strike", "strike-1", CardType.ATTACK, 1),
            CardState("defend_r", "Defend", "defend-1", CardType.SKILL, 1),
        ),
    )


def _card_from_json(payload: JsonValue) -> CardState:
    data = _mapping(payload, "card")
    return CardState(
        card_id=str(data["card_id"]),
        name=str(data["name"]),
        instance_id=str(data["instance_id"]),
        card_type=data["card_type"],
        cost=int(data["cost"]),
        upgraded=bool(data.get("upgraded", False)),
        tags=tuple(str(tag) for tag in _sequence(data.get("tags", ()), "card.tags")),
    )


def _power_from_json(payload: JsonValue) -> CombatPower:
    data = _mapping(payload, "power")
    return CombatPower(id=str(data["id"]), amount=int(data.get("amount", 0)))


def _player_from_json(payload: Mapping[str, JsonValue]) -> PlayerState:
    return PlayerState(
        hp=int(payload["hp"]),
        max_hp=int(payload["max_hp"]),
        block=int(payload.get("block", 0)),
        energy=int(payload.get("energy", 0)),
        powers=tuple(_power_from_json(item) for item in _sequence(payload.get("powers", ()), "powers")),
        relics=tuple(str(relic) for relic in _sequence(payload.get("relics", ()), "relics")),
        potions=tuple(
            PotionState(slot=int(_mapping(item, "potion")["slot"]), name=_optional_str(_mapping(item, "potion").get("name")))
            for item in _sequence(payload.get("potions", ()), "potions")
        ),
    )


def _monster_from_json(payload: JsonValue) -> MonsterState:
    data = _mapping(payload, "monster")
    return MonsterState(
        monster_id=str(data["monster_id"]),
        name=str(data["name"]),
        hp=int(data["hp"]),
        max_hp=int(data["max_hp"]),
        block=int(data.get("block", 0)),
        intent=_optional_str(data.get("intent")),
        intent_damage=int(data.get("intent_damage", 0)),
        powers=tuple(_power_from_json(item) for item in _sequence(data.get("powers", ()), "monster.powers")),
        targetable=bool(data.get("targetable", True)),
    )


def _action_from_json(payload: JsonValue) -> CombatAction:
    data = _mapping(payload, "action")
    return CombatAction(
        action_type=data["action_type"],
        card_instance_id=_optional_str(data.get("card_instance_id")),
        monster_id=_optional_str(data.get("monster_id")),
        potion_slot=None if data.get("potion_slot") is None else int(data["potion_slot"]),
        stable_id=_optional_str(data.get("stable_id")),
        label=_optional_str(data.get("label")),
        metadata=_mapping(data.get("metadata", {}), "action.metadata"),
    )


def _metrics_from_json(payload: Mapping[str, JsonValue]) -> CombatMetrics:
    return CombatMetrics(
        result=payload.get("result", CombatResult.ONGOING.value),
        hp_loss=int(payload.get("hp_loss", 0)),
        turns_taken=int(payload.get("turns_taken", 0)),
        damage_dealt=int(payload.get("damage_dealt", 0)),
        potions_used=int(payload.get("potions_used", 0)),
        cards_played=int(payload.get("cards_played", 0)),
        search_time_ms=float(payload.get("search_time_ms", 0.0)),
    )


def _draw_cards(
    draw_pile: Sequence[CardState],
    discard_pile: Sequence[CardState],
    count: int,
) -> tuple[tuple[CardState, ...], tuple[CardState, ...], tuple[CardState, ...]]:
    draw = list(draw_pile)
    discard = list(discard_pile)
    hand: list[CardState] = []
    for _ in range(count):
        if not draw:
            if not discard:
                break
            draw = discard
            discard = []
        hand.append(draw.pop(0))
    return tuple(hand), tuple(draw), tuple(discard)


def _damage_target(
    state: CombatState,
    action: CombatAction,
    damage: int,
) -> tuple[tuple[MonsterState, ...], int]:
    monster_id = _require_value(action.monster_id, "monster_id")
    target = state.monster_by_id(monster_id)
    if target is None or not target.is_alive or not target.targetable:
        raise ValueError(f"invalid target: {monster_id}")

    remaining = max(0, damage)
    new_monsters: list[MonsterState] = []
    hp_damage = 0
    for monster in state.monsters:
        if monster.monster_id != monster_id:
            new_monsters.append(monster)
            continue
        if _power_amount(monster.powers, "Vulnerable") > 0:
            remaining = remaining * 3 // 2
        block_damage = min(monster.block, remaining)
        remaining -= block_damage
        hp_damage = min(monster.hp, remaining)
        new_monsters.append(
            replace(
                monster,
                block=monster.block - block_damage,
                hp=monster.hp - hp_damage,
            )
        )
    return tuple(new_monsters), hp_damage


def _apply_monster_power(
    monsters: Sequence[MonsterState],
    monster_id: str,
    power_id: str,
    amount: int,
) -> tuple[MonsterState, ...]:
    return tuple(
        replace(monster, powers=_add_power(monster.powers, power_id, amount))
        if monster.monster_id == monster_id and monster.is_alive
        else monster
        for monster in monsters
    )


def _add_power(
    powers: Sequence[CombatPower],
    power_id: str,
    amount: int,
) -> tuple[CombatPower, ...]:
    normalized = _normalize_power_id(power_id)
    updated: list[CombatPower] = []
    found = False
    for power in powers:
        if _normalize_power_id(power.id) == normalized:
            updated.append(replace(power, amount=power.amount + amount))
            found = True
        else:
            updated.append(power)
    if not found:
        updated.append(CombatPower(power_id, amount))
    return tuple(updated)


def _attack_damage(card: CardState, state: CombatState) -> int:
    if _card_family(card) == "bash":
        base = 10 if card.upgraded else 8
    else:
        base = 9 if card.upgraded else 6
    return max(0, base + _power_amount(state.player.powers, "Strength"))


def _defend_block(card: CardState, state: CombatState) -> int:
    base = 8 if card.upgraded else 5
    return max(0, base + _power_amount(state.player.powers, "Dexterity"))


def _enemy_intent_damage(state: CombatState) -> int:
    total = 0
    player_vulnerable = _power_amount(state.player.powers, "Vulnerable") > 0
    for monster in state.living_monsters:
        damage = max(0, monster.intent_damage + _power_amount(monster.powers, "Strength"))
        if player_vulnerable:
            damage = damage * 3 // 2
        total += damage
    return total


def _power_amount(powers: Sequence[CombatPower], power_id: str) -> int:
    normalized = _normalize_power_id(power_id)
    return sum(power.amount for power in powers if _normalize_power_id(power.id) == normalized)


def _normalize_power_id(power_id: str) -> str:
    return power_id.replace("_", "").replace(" ", "").casefold()


def _is_supported_card(card: CardState) -> bool:
    return _card_family(card) in {"strike", "defend", "bash"}


def _card_family(card: CardState) -> str | None:
    key = f"{card.card_id} {card.name}".replace("_", " ").casefold()
    if "bash" in key:
        return "bash"
    if "strike" in key:
        return "strike"
    if "defend" in key:
        return "defend"
    return None


def _remove_card(hand: Sequence[CardState], instance_id: str) -> tuple[CardState, ...]:
    removed = False
    remaining: list[CardState] = []
    for card in hand:
        if not removed and card.instance_id == instance_id:
            removed = True
            continue
        remaining.append(card)
    if not removed:
        raise ValueError(f"card is not in hand: {instance_id}")
    return tuple(remaining)


def _complete_state(state: CombatState, result: CombatResult) -> CombatState:
    return replace(
        state,
        phase=CombatPhase.COMPLETE,
        result=result,
        metrics=replace(state.metrics, result=result),
    )


def _mapping(value: JsonValue, context: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _sequence(value: JsonValue, context: str) -> Sequence[JsonValue]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{context} must be a sequence")
    return value


def _optional_str(value: JsonValue) -> str | None:
    return None if value is None else str(value)


def _require_value(value: str | None, context: str) -> str:
    if value is None:
        raise ValueError(f"{context} is required")
    return value


__all__ = [
    "CombatFixtureSimulator",
    "LocalCombatFixtureSimulator",
    "combat_state_from_json",
    "combat_state_to_json",
    "make_fixture_state",
]
