"""Combat-only API contracts for Slay the Spire policy work.

The current local ``sts_lightspeed`` binding exposes full playouts and card
reward pauses, but not action-level combat control. This module defines the
Python-side contract that planners and learned policies can target now. A real
adapter should delegate mechanics to the C++ simulator once bindings expose
combat decisions, legal actions, cloning, and action application.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class ActionType(str, Enum):
    """Stable high-level categories for combat decisions."""

    PLAY_CARD = "play_card"
    USE_POTION = "use_potion"
    END_TURN = "end_turn"


ActionKind = ActionType


class CardType(str, Enum):
    ATTACK = "attack"
    SKILL = "skill"
    POWER = "power"
    STATUS = "status"
    CURSE = "curse"


class CombatPhase(str, Enum):
    PLAYER_TURN = "player_turn"
    ENEMY_TURN = "enemy_turn"
    COMPLETE = "complete"


class CombatResult(str, Enum):
    ONGOING = "ongoing"
    PLAYER_VICTORY = "player_victory"
    PLAYER_DEATH = "player_death"


CombatOutcome = CombatResult


@dataclass(frozen=True)
class CardState:
    """Serializable view of one combat card instance."""

    card_id: str
    name: str
    instance_id: str
    card_type: CardType
    cost: int
    upgraded: bool = False
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "card_type", _coerce_enum(CardType, self.card_type, "card_type"))
        object.__setattr__(self, "tags", tuple(self.tags))

    @property
    def display_name(self) -> str:
        return f"{self.name}+" if self.upgraded else self.name

    def to_json(self) -> dict[str, JsonValue]:
        return _json_dict(asdict(self))


CombatCard = CardState


@dataclass(frozen=True)
class CombatPower:
    """Serializable view of a player or monster power."""

    id: str
    amount: int = 0

    def to_json(self) -> dict[str, JsonValue]:
        return _json_dict(asdict(self))


@dataclass(frozen=True)
class PotionState:
    """Serializable view of one potion slot."""

    slot: int
    name: str | None

    def __post_init__(self) -> None:
        if self.slot < 0:
            raise ValueError("potion slot must be non-negative")

    @property
    def is_filled(self) -> bool:
        return self.name is not None

    def to_json(self) -> dict[str, JsonValue]:
        return _json_dict(asdict(self))


@dataclass(frozen=True)
class PlayerState:
    """Serializable view of the player side of combat."""

    hp: int
    max_hp: int
    block: int = 0
    energy: int = 0
    powers: tuple[CombatPower, ...] = ()
    relics: tuple[str, ...] = ()
    potions: tuple[PotionState, ...] = ()

    def to_json(self) -> dict[str, JsonValue]:
        return _json_dict(asdict(self))


CombatPlayer = PlayerState


@dataclass(frozen=True)
class MonsterState:
    """Serializable view of an enemy combatant."""

    monster_id: str
    name: str
    hp: int
    max_hp: int
    block: int = 0
    intent: str | None = None
    intent_damage: int = 0
    powers: tuple[CombatPower, ...] = ()
    targetable: bool = True

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    def to_json(self) -> dict[str, JsonValue]:
        return _json_dict(asdict(self))


CombatMonster = MonsterState


@dataclass(frozen=True)
class CombatAction:
    """One legal combat action."""

    action_type: ActionType
    card_instance_id: str | None = None
    monster_id: str | None = None
    potion_slot: int | None = None
    stable_id: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_type", _coerce_enum(ActionType, self.action_type, "action_type"))
        if self.potion_slot is not None and self.potion_slot < 0:
            raise ValueError("potion slot must be non-negative")

    @classmethod
    def play_card(cls, card_instance_id: str, monster_id: str | None = None) -> "CombatAction":
        return cls(
            action_type=ActionType.PLAY_CARD,
            card_instance_id=card_instance_id,
            monster_id=monster_id,
        )

    @classmethod
    def use_potion(cls, potion_slot: int, monster_id: str | None = None) -> "CombatAction":
        return cls(
            action_type=ActionType.USE_POTION,
            potion_slot=potion_slot,
            monster_id=monster_id,
        )

    @classmethod
    def end_turn(cls) -> "CombatAction":
        return cls(action_type=ActionType.END_TURN)

    @property
    def kind(self) -> ActionType:
        return self.action_type

    def action_key(self) -> str:
        if self.stable_id is not None:
            return self.stable_id
        if self.action_type is ActionType.PLAY_CARD:
            return f"play:{self.card_instance_id}:{self.monster_id}"
        if self.action_type is ActionType.USE_POTION:
            return f"potion:{self.potion_slot}:{self.monster_id}"
        return "end_turn"

    def to_json(self) -> dict[str, JsonValue]:
        return _json_dict(asdict(self))


@dataclass(frozen=True)
class CombatMetrics:
    """Reward-independent metrics accumulated during one combat."""

    result: CombatResult = CombatResult.ONGOING
    hp_loss: int = 0
    turns_taken: int = 0
    damage_dealt: int = 0
    potions_used: int = 0
    cards_played: int = 0
    search_time_ms: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "result", _coerce_enum(CombatResult, self.result, "result"))

    @property
    def outcome(self) -> CombatResult:
        return self.result

    def to_json(self) -> dict[str, JsonValue]:
        return _json_dict(asdict(self))


@dataclass(frozen=True)
class CombatState:
    """Serializable combat state at a policy decision boundary."""

    player: PlayerState
    monsters: tuple[MonsterState, ...]
    hand: tuple[CardState, ...] = ()
    draw_pile: tuple[CardState, ...] = ()
    discard_pile: tuple[CardState, ...] = ()
    exhaust_pile: tuple[CardState, ...] = ()
    turn: int = 1
    phase: CombatPhase = CombatPhase.PLAYER_TURN
    result: CombatResult = CombatResult.ONGOING
    action_history: tuple[CombatAction, ...] = ()
    metrics: CombatMetrics = field(default_factory=CombatMetrics)
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "monsters", tuple(self.monsters))
        object.__setattr__(self, "hand", tuple(self.hand))
        object.__setattr__(self, "draw_pile", tuple(self.draw_pile))
        object.__setattr__(self, "discard_pile", tuple(self.discard_pile))
        object.__setattr__(self, "exhaust_pile", tuple(self.exhaust_pile))
        object.__setattr__(self, "action_history", tuple(self.action_history))
        object.__setattr__(self, "phase", _coerce_enum(CombatPhase, self.phase, "phase"))
        object.__setattr__(self, "result", _coerce_enum(CombatResult, self.result, "result"))

    @property
    def is_terminal(self) -> bool:
        return self.result is not CombatResult.ONGOING or self.phase is CombatPhase.COMPLETE

    @property
    def living_monsters(self) -> tuple[MonsterState, ...]:
        return tuple(monster for monster in self.monsters if monster.is_alive)

    @property
    def targetable_monsters(self) -> tuple[MonsterState, ...]:
        return tuple(monster for monster in self.living_monsters if monster.targetable)

    @property
    def incoming_damage(self) -> int:
        return sum(monster.intent_damage for monster in self.living_monsters)

    @property
    def outcome(self) -> CombatResult:
        return self.result

    def card_in_hand(self, instance_id: str) -> CardState | None:
        return next((card for card in self.hand if card.instance_id == instance_id), None)

    def monster_by_id(self, monster_id: str) -> MonsterState | None:
        return next((monster for monster in self.monsters if monster.monster_id == monster_id), None)

    def to_json(self) -> dict[str, JsonValue]:
        return _json_dict(asdict(self))


CombatObservation = CombatState


@dataclass(frozen=True)
class ActionOutcome:
    """Result of applying one combat action."""

    before: CombatState
    action: CombatAction
    after: CombatState
    metrics_delta: CombatMetrics = field(default_factory=CombatMetrics)
    events: tuple[str, ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.after.is_terminal

    @property
    def result(self) -> CombatResult:
        return self.after.result


@runtime_checkable
class CombatSimulator(Protocol):
    """Pure state-transition interface used by Python combat policies."""

    def legal_actions(self, state: CombatState) -> Sequence[CombatAction]:
        """Return legal actions for ``state``."""

    def step(self, state: CombatState, action: CombatAction) -> ActionOutcome:
        """Apply exactly one combat action and return the transition."""

    def clone(self, state: CombatState) -> CombatState:
        """Return an independent copy of ``state``."""

    def advance_to_decision(self, state: CombatState) -> CombatState:
        """Advance non-player phases until terminal or next player decision."""


@runtime_checkable
class CombatPolicy(Protocol):
    """Policy interface for choosing one action from legal combat actions."""

    def choose_action(
        self, state: CombatState, legal_actions: Sequence[CombatAction]
    ) -> CombatAction | None:
        """Choose one action, or None when no legal action exists."""


@runtime_checkable
class CombatBackend(Protocol):
    """Action-level simulator boundary required by future sts_lightspeed adapters."""

    def observe(self) -> CombatObservation:
        """Return the current decision-state observation."""

    def legal_actions(self) -> Sequence[CombatAction]:
        """Return legal actions for the current decision state."""

    def apply(self, action: CombatAction) -> None:
        """Apply exactly one legal combat action in-place."""

    def clone(self) -> "CombatBackend":
        """Return an independent copy with identical deterministic state."""

    def advance_to_player_decision(self) -> None:
        """Advance non-player phases until terminal or next player decision."""

    def is_terminal(self) -> bool:
        """Return whether the combat is complete."""

    def metrics(self) -> CombatMetrics:
        """Return combat metrics accumulated so far."""


class UnsupportedCombatBinding(RuntimeError):
    """Raised when the current external binding cannot expose combat control."""


class StsLightspeedCombatBackend:
    """Placeholder adapter for future action-level ``sts_lightspeed`` bindings."""

    def __init__(self, game_context: Any) -> None:
        self.game_context = game_context

    def observe(self) -> CombatObservation:
        raise UnsupportedCombatBinding(
            "Current sts_lightspeed Python bindings do not expose combat observations."
        )

    def legal_actions(self) -> Sequence[CombatAction]:
        raise UnsupportedCombatBinding(
            "Current sts_lightspeed Python bindings do not expose legal combat actions."
        )

    def apply(self, action: CombatAction) -> None:
        raise UnsupportedCombatBinding(
            "Current sts_lightspeed Python bindings do not expose single-action application."
        )

    def clone(self) -> CombatBackend:
        raise UnsupportedCombatBinding(
            "Current sts_lightspeed Python bindings do not expose combat-state cloning."
        )

    def advance_to_player_decision(self) -> None:
        raise UnsupportedCombatBinding(
            "Current sts_lightspeed Python bindings do not expose combat decision pauses."
        )

    def is_terminal(self) -> bool:
        raise UnsupportedCombatBinding(
            "Current sts_lightspeed Python bindings do not expose combat terminal state."
        )

    def metrics(self) -> CombatMetrics:
        raise UnsupportedCombatBinding(
            "Current sts_lightspeed Python bindings do not expose combat-only metrics."
        )


def fixture_legal_player_actions(state: CombatState) -> tuple[CombatAction, ...]:
    """Return conservative legal actions for deterministic Python fixtures.

    This helper is intentionally simple. The C++ simulator remains the source
    of truth for exact card/relic/enemy mechanics once action bindings exist.
    Real ``sts_lightspeed`` adapters must use simulator-provided legal actions,
    not this fixture approximation.
    """

    if state.is_terminal or state.phase is not CombatPhase.PLAYER_TURN:
        return ()

    actions: list[CombatAction] = []
    targets = state.targetable_monsters

    for card in state.hand:
        if card.cost > state.player.energy:
            continue
        if card.card_type is CardType.ATTACK:
            actions.extend(CombatAction.play_card(card.instance_id, target.monster_id) for target in targets)
        elif card.card_type in {CardType.SKILL, CardType.POWER}:
            actions.append(CombatAction.play_card(card.instance_id))

    for potion in state.player.potions:
        if not potion.is_filled:
            continue
        if targets:
            actions.extend(CombatAction.use_potion(potion.slot, target.monster_id) for target in targets)
        else:
            actions.append(CombatAction.use_potion(potion.slot))

    actions.append(CombatAction.end_turn())
    return tuple(actions)


def legal_player_actions(state: CombatState) -> tuple[CombatAction, ...]:
    """Compatibility alias for ``fixture_legal_player_actions``."""

    return fixture_legal_player_actions(state)


def _coerce_enum(enum_type: type[Enum], value: Any, field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid {enum_type.__name__}") from exc


def _json_dict(raw: dict[str, Any]) -> dict[str, JsonValue]:
    return {key: _json_value(value) for key, value in raw.items()}


def _json_value(value: Any) -> JsonValue:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "to_json"):
        return value.to_json()
    return value
