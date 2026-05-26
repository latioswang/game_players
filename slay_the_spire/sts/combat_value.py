"""Lightweight combat-state value evaluation.

This module intentionally has no ML dependency.  It provides a small evaluator
interface plus a deterministic handcrafted scorer that can run against plain
objects, dictionaries, the local fallback dataclasses below, or a future
``combat_api`` module with equivalent state/action concepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class FallbackCombatAction:
    """Minimal action placeholder until a real combat_api action is available."""

    name: str
    next_state: Any | None = None


@dataclass(frozen=True)
class FallbackMonsterState:
    """Minimal monster placeholder used by tests and early adapters."""

    hp: int
    max_hp: int | None = None
    block: int = 0
    intent_damage: int = 0
    intent_hits: int = 1
    powers: Mapping[str, int] = field(default_factory=dict)
    targetable: bool = True
    alive: bool = True


@dataclass(frozen=True)
class FallbackCombatState:
    """Minimal combat-state placeholder until the action-level API lands."""

    player_hp: int
    player_max_hp: int
    player_block: int = 0
    energy: int = 0
    monsters: Sequence[FallbackMonsterState] = field(default_factory=tuple)
    hand: Sequence[Any] = field(default_factory=tuple)
    draw_pile: Sequence[Any] = field(default_factory=tuple)
    discard_pile: Sequence[Any] = field(default_factory=tuple)
    exhaust_pile: Sequence[Any] = field(default_factory=tuple)
    powers: Mapping[str, int] = field(default_factory=dict)
    turn: int = 1
    terminal_outcome: str | None = None


try:  # Prefer a package-local combat API when another implementation provides it.
    from .combat_api import CombatAction as CombatAction  # type: ignore[import-not-found]
    from .combat_api import CombatState as CombatState  # type: ignore[import-not-found]

    HAS_COMBAT_API = True
except ImportError:
    try:  # Also support direct script-style imports from slay_the_spire/sts.
        from combat_api import CombatAction as CombatAction  # type: ignore[import-not-found]
        from combat_api import CombatState as CombatState  # type: ignore[import-not-found]

        HAS_COMBAT_API = True
    except ImportError:
        CombatAction = FallbackCombatAction
        CombatState = FallbackCombatState
        HAS_COMBAT_API = False


class CombatValueEvaluator(Protocol):
    """Small common interface for handcrafted, search, or learned evaluators."""

    def evaluate(self, state: Any) -> "CombatValue":
        """Return a higher-is-better value estimate for ``state``."""


@dataclass(frozen=True)
class CombatValue:
    """Deterministic, inspectable value estimate for a combat state."""

    score: float
    terminal: str | None
    player_hp: int
    player_max_hp: int
    player_block: int
    incoming_damage: int
    expected_hp_loss: int
    enemy_hp: int
    alive_enemies: int
    details: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionValue:
    """A legal action paired with its successor-state value."""

    action: Any
    value: CombatValue


@dataclass(frozen=True)
class HandcraftedValueWeights:
    """Tunable coefficients for the deterministic baseline scorer."""

    terminal_win: float = 1_000_000.0
    terminal_loss: float = -1_000_000.0
    lethal_setup: float = 100_000.0
    death_risk: float = -100_000.0
    hp: float = 6.0
    hp_ratio: float = 35.0
    expected_hp_loss: float = -8.0
    incoming_damage: float = -0.25
    useful_block: float = 1.5
    excess_block: float = 0.15
    enemy_hp: float = -1.2
    alive_enemy: float = -4.0
    energy: float = 1.25
    hand_card: float = 0.45
    draw_card: float = 0.08
    discard_card: float = 0.03
    player_strength: float = 2.0
    player_dexterity: float = 2.0
    player_weak: float = -4.0
    player_vulnerable: float = -6.0
    enemy_strength: float = -1.0
    enemy_weak: float = 1.5
    enemy_vulnerable: float = 2.0
    turn: float = -0.35


class HandcraftedCombatValueEvaluator:
    """Deterministic baseline scorer for combat states.

    The score is intentionally simple and inspectable: terminal outcomes
    dominate, survival is valued strongly, unblocked incoming damage is costly,
    and lower remaining monster HP is better.  It is a baseline leaf evaluator,
    not a replacement for exact combat mechanics.
    """

    def __init__(self, weights: HandcraftedValueWeights | None = None) -> None:
        self.weights = weights or HandcraftedValueWeights()

    def evaluate(self, state: Any) -> CombatValue:
        snapshot = _snapshot(state)
        terminal = _terminal_outcome(state, snapshot)
        weights = self.weights

        score = 0.0
        if terminal == "victory":
            score += weights.terminal_win
        elif terminal == "defeat":
            score += weights.terminal_loss

        if terminal is None and snapshot.enemy_hp <= 0:
            score += weights.lethal_setup
            terminal = "victory"
        if terminal is None and snapshot.player_hp <= 0:
            score += weights.death_risk
            terminal = "defeat"
        if terminal is None and snapshot.player_hp - snapshot.expected_hp_loss <= 0:
            score += weights.death_risk

        hp_ratio = snapshot.player_hp / max(1, snapshot.player_max_hp)
        useful_block = min(snapshot.player_block, snapshot.incoming_damage)
        excess_block = max(0, snapshot.player_block - snapshot.incoming_damage)

        details = {
            "hp": snapshot.player_hp * weights.hp,
            "hp_ratio": hp_ratio * weights.hp_ratio,
            "expected_hp_loss": snapshot.expected_hp_loss * weights.expected_hp_loss,
            "incoming_damage": snapshot.incoming_damage * weights.incoming_damage,
            "useful_block": useful_block * weights.useful_block,
            "excess_block": excess_block * weights.excess_block,
            "enemy_hp": snapshot.enemy_hp * weights.enemy_hp,
            "alive_enemy": snapshot.alive_enemies * weights.alive_enemy,
            "energy": snapshot.energy * weights.energy,
            "hand_card": snapshot.hand_size * weights.hand_card,
            "draw_card": snapshot.draw_size * weights.draw_card,
            "discard_card": snapshot.discard_size * weights.discard_card,
            "player_strength": snapshot.player_strength * weights.player_strength,
            "player_dexterity": snapshot.player_dexterity * weights.player_dexterity,
            "player_weak": snapshot.player_weak * weights.player_weak,
            "player_vulnerable": snapshot.player_vulnerable * weights.player_vulnerable,
            "enemy_strength": snapshot.enemy_strength * weights.enemy_strength,
            "enemy_weak": snapshot.enemy_weak * weights.enemy_weak,
            "enemy_vulnerable": snapshot.enemy_vulnerable * weights.enemy_vulnerable,
            "turn": snapshot.turn * weights.turn,
        }
        score += sum(details.values())

        return CombatValue(
            score=score,
            terminal=terminal,
            player_hp=snapshot.player_hp,
            player_max_hp=snapshot.player_max_hp,
            player_block=snapshot.player_block,
            incoming_damage=snapshot.incoming_damage,
            expected_hp_loss=snapshot.expected_hp_loss,
            enemy_hp=snapshot.enemy_hp,
            alive_enemies=snapshot.alive_enemies,
            details=details,
        )

    def evaluate_actions(self, state: Any, actions: Iterable[Any]) -> list[ActionValue]:
        """Score actions with available successor states.

        The future combat API should provide exact transition results.  Until
        then, this helper only consumes common successor attributes such as
        ``next_state``/``result_state``/``state_after``.  Actions without a
        successor are scored as the current state, which keeps the interface
        usable while making the integration gap explicit.
        """

        return [ActionValue(action=action, value=self.evaluate(_successor_state(state, action))) for action in actions]

    def best_action(self, state: Any, actions: Iterable[Any]) -> ActionValue:
        """Return the highest-valued action, with deterministic tie-breaking."""

        values = self.evaluate_actions(state, actions)
        if not values:
            raise ValueError("best_action requires at least one action")
        return max(values, key=lambda item: (item.value.score, _action_name(item.action)))


@dataclass(frozen=True)
class _StateSnapshot:
    player_hp: int
    player_max_hp: int
    player_block: int
    energy: int
    incoming_damage: int
    expected_hp_loss: int
    enemy_hp: int
    alive_enemies: int
    hand_size: int
    draw_size: int
    discard_size: int
    player_strength: int
    player_dexterity: int
    player_weak: int
    player_vulnerable: int
    enemy_strength: int
    enemy_weak: int
    enemy_vulnerable: int
    turn: int


def _snapshot(state: Any) -> _StateSnapshot:
    player = _get(state, "player", "character", "hero", default=state)
    player_hp = _int(_get(player, "player_hp", "cur_hp", "current_hp", "hp", default=0))
    player_max_hp = _int(_get(player, "player_max_hp", "max_hp", "maximum_hp", default=max(1, player_hp)))
    player_block = _int(_get(player, "player_block", "block", default=_get(state, "player_block", default=0)))
    energy = _int(
        _get(
            player,
            "energy",
            "player_energy",
            "current_energy",
            default=_get(state, "energy", "player_energy", "current_energy", default=0),
        )
    )

    monsters = _monsters(state)
    alive_monsters = [monster for monster in monsters if _is_alive(monster)]
    incoming_damage = sum(_monster_intent_damage(monster) for monster in alive_monsters)
    expected_hp_loss = max(0, incoming_damage - player_block)
    enemy_hp = sum(max(0, _int(_get(monster, "cur_hp", "current_hp", "hp", default=0))) for monster in alive_monsters)

    player_strength = _power_amount(player, "strength")
    player_dexterity = _power_amount(player, "dexterity", "dex")
    player_weak = _power_amount(player, "weak")
    player_vulnerable = _power_amount(player, "vulnerable", "vuln")

    enemy_strength = sum(_power_amount(monster, "strength") for monster in alive_monsters)
    enemy_weak = sum(_power_amount(monster, "weak") for monster in alive_monsters)
    enemy_vulnerable = sum(_power_amount(monster, "vulnerable", "vuln") for monster in alive_monsters)

    return _StateSnapshot(
        player_hp=player_hp,
        player_max_hp=player_max_hp,
        player_block=player_block,
        energy=energy,
        incoming_damage=incoming_damage,
        expected_hp_loss=expected_hp_loss,
        enemy_hp=enemy_hp,
        alive_enemies=len(alive_monsters),
        hand_size=_size(_get(state, "hand", "cards_in_hand", default=())),
        draw_size=_size(_get(state, "draw_pile", "draw", "deck_draw", default=())),
        discard_size=_size(_get(state, "discard_pile", "discard", default=())),
        player_strength=player_strength,
        player_dexterity=player_dexterity,
        player_weak=player_weak,
        player_vulnerable=player_vulnerable,
        enemy_strength=enemy_strength,
        enemy_weak=enemy_weak,
        enemy_vulnerable=enemy_vulnerable,
        turn=_int(_get(state, "turn", "turn_number", default=0)),
    )


def _terminal_outcome(state: Any, snapshot: _StateSnapshot) -> str | None:
    if snapshot.player_hp <= 0:
        return "defeat"
    if snapshot.alive_enemies == 0 and _monsters(state):
        return "victory"

    outcome = _get(
        state,
        "terminal_outcome",
        "outcome",
        "result",
        "combat_result",
        default=None,
    )
    normalized = _normalized_name(outcome)
    if normalized in {
        "victory",
        "win",
        "won",
        "player_victory",
        "combat_victory",
        "monster_defeat",
    }:
        return "victory"
    if normalized in {
        "defeat",
        "loss",
        "lost",
        "player_loss",
        "player_death",
        "combat_loss",
        "dead",
    }:
        return "defeat"

    done = _get(state, "done", "terminal", "is_terminal", default=False)
    if bool(done):
        won = _get(state, "won", "player_won", "combat_won", default=None)
        if won is not None:
            return "victory" if bool(won) else "defeat"
    return None


def _successor_state(state: Any, action: Any) -> Any:
    for name in ("next_state", "result_state", "state_after", "after_state"):
        value = _get(action, name, default=None)
        if value is not None:
            return value() if callable(value) else value
    return state


def _monsters(state: Any) -> list[Any]:
    monsters = _get(state, "monsters", "enemies", "monster_states", default=None)
    if monsters is None:
        monster = _get(state, "monster", "enemy", default=None)
        monsters = () if monster is None else (monster,)
    if isinstance(monsters, Mapping):
        return list(monsters.values())
    if isinstance(monsters, Iterable) and not isinstance(monsters, (str, bytes)):
        return list(monsters)
    return [monsters]


def _is_alive(monster: Any) -> bool:
    alive = _get(monster, "alive", "is_alive", default=None)
    if alive is not None and not bool(alive):
        return False
    hp = _int(_get(monster, "cur_hp", "current_hp", "hp", default=0))
    targetable = _get(monster, "targetable", "is_targetable", default=True)
    return hp > 0 and bool(targetable)


def _monster_intent_damage(monster: Any) -> int:
    intent = _get(monster, "intent", default=None)
    damage = _get(monster, "intent_damage", "damage", "attack_damage", default=None)
    hits = _get(monster, "intent_hits", "hits", "attack_hits", default=None)

    if damage is None and intent is not None:
        damage = _get(intent, "damage", "intent_damage", "attack_damage", default=0)
    if hits is None and intent is not None:
        hits = _get(intent, "hits", "intent_hits", "attack_hits", default=1)

    base_damage = _int(damage)
    hit_count = max(1, _int(hits, default=1))
    strength = _power_amount(monster, "strength")
    return max(0, base_damage + strength) * hit_count


def _power_amount(source: Any, *names: str) -> int:
    for name in names:
        direct = _get(source, name, default=None)
        if direct is not None:
            return _int(direct)

    powers = _get(source, "powers", "power_state", default=None)
    if powers is None:
        return 0

    aliases = {_normalized_name(name) for name in names}
    if isinstance(powers, Mapping):
        for key, value in powers.items():
            if _normalized_name(key) in aliases:
                return _int(value)
        return 0

    if isinstance(powers, Iterable) and not isinstance(powers, (str, bytes)):
        for power in powers:
            power_name = _get(power, "id", "name", "power_id", default=power)
            if _normalized_name(power_name) in aliases:
                return _int(_get(power, "amount", "stacks", default=1))
    return 0


def _get(source: Any, *names: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return default
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _size(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return sum(1 for _ in value)


def _normalized_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        value = value.name
    return str(value).rsplit(".", 1)[-1].lower()


def _action_name(action: Any) -> str:
    name = _get(action, "name", "id", "action_id", default=action)
    return _normalized_name(name)


__all__ = [
    "ActionValue",
    "CombatAction",
    "CombatState",
    "CombatValue",
    "CombatValueEvaluator",
    "FallbackCombatAction",
    "FallbackCombatState",
    "FallbackMonsterState",
    "HandcraftedCombatValueEvaluator",
    "HandcraftedValueWeights",
    "HAS_COMBAT_API",
]
