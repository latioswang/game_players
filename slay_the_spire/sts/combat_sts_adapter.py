"""Payload adapter for future action-level ``sts_lightspeed`` combat bindings.

The current public binding does not expose combat decisions yet. This module is
the Python-side bridge for the binding shape documented in
``STS_LIGHTSPEED_BINDING_PLAN.md``: upstream returns plain payload dictionaries,
and the local planner consumes the stable ``combat_api`` dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:
    from .combat_api import (
        ActionType,
        CardState,
        CardType,
        CombatAction,
        CombatBackend,
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
        ActionType,
        CardState,
        CardType,
        CombatAction,
        CombatBackend,
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


class StaleCombatAction(ValueError):
    """Raised when a previously legal live action no longer applies."""


class StsLightspeedPayloadCombatBackend:
    """Adapt a pybind combat handle with dict payload methods to ``CombatBackend``.

    The wrapped handle is expected to expose ``observe``, ``legal_actions``,
    ``apply``, ``clone``, ``advance_to_player_decision``, ``is_terminal``, and
    ``metrics``. Legal actions are validated against the latest live action list
    before forwarding to catch stale stable ids after state mutations.
    """

    def __init__(self, handle: Any) -> None:
        self.handle = handle

    def observe(self) -> CombatState:
        return combat_state_from_payload(_call(self.handle, "observe"))

    def legal_actions(self) -> tuple[CombatAction, ...]:
        return tuple(combat_action_from_payload(payload) for payload in self._legal_payloads())

    def apply(self, action: CombatAction) -> None:
        payload = self._matching_current_payload(action)
        if payload is None:
            raise StaleCombatAction(f"stale or illegal live combat action: {action.action_key()}")
        self.handle.apply(payload)

    def clone(self) -> CombatBackend:
        return type(self)(_call(self.handle, "clone"))

    def advance_to_player_decision(self) -> None:
        self.handle.advance_to_player_decision()

    def is_terminal(self) -> bool:
        return bool(_call(self.handle, "is_terminal"))

    def metrics(self) -> CombatMetrics:
        return combat_metrics_from_payload(_call(self.handle, "metrics"))

    def _legal_payloads(self) -> tuple[Mapping[str, Any], ...]:
        payloads = _call(self.handle, "legal_actions")
        if not isinstance(payloads, Sequence) or isinstance(payloads, (str, bytes, bytearray)):
            raise TypeError("legal_actions() must return a sequence of action payloads")
        return tuple(_mapping(payload, "legal action payload") for payload in payloads)

    def _matching_current_payload(self, action: CombatAction) -> Mapping[str, Any] | None:
        for payload in self._legal_payloads():
            current = combat_action_from_payload(payload)
            if current.action_key() == action.action_key() and _live_identity_matches(action, current):
                return payload
        return None


def combat_state_from_payload(payload: Mapping[str, Any]) -> CombatState:
    """Convert an upstream observation payload into ``CombatState``."""

    payload = _mapping(payload, "combat observation payload")
    combat_payload = _mapping(_get(payload, "combat", default={}), "combat payload")
    metrics_payload = _get(payload, "metrics", default={})
    history_payload = _sequence(_get(payload, "action_history", "actions_taken", default=()))

    return CombatState(
        player=player_from_payload(_mapping(_get(payload, "player"), "player payload")),
        monsters=tuple(monster_from_payload(monster) for monster in _sequence(_get(payload, "monsters"))),
        hand=tuple(card_from_payload(card) for card in _zone(payload, "hand")),
        draw_pile=tuple(card_from_payload(card) for card in _zone(payload, "draw_pile", "draw")),
        discard_pile=tuple(card_from_payload(card) for card in _zone(payload, "discard_pile", "discard")),
        exhaust_pile=tuple(card_from_payload(card) for card in _zone(payload, "exhaust_pile", "exhaust")),
        turn=_int(_get(combat_payload, "turn", default=_get(payload, "turn", default=1)), "turn"),
        phase=_phase_from_payload(payload),
        result=_result_from_payload(payload),
        action_history=tuple(combat_action_from_payload(action) for action in history_payload),
        metrics=combat_metrics_from_payload(_mapping(metrics_payload, "metrics payload")),
        metadata=_state_metadata(
            payload,
            {
                "sts_payload": _metadata_payload(payload),
                "input_state": _get(combat_payload, "input_state", default=_get(payload, "input_state", default=None)),
                "encounter": _get(combat_payload, "encounter", default=_get(payload, "encounter", default=None)),
                "floor": _get(combat_payload, "floor", default=_get(payload, "floor", default=None)),
            },
        ),
    )


def player_from_payload(payload: Mapping[str, Any]) -> PlayerState:
    return PlayerState(
        hp=_int(_get(payload, "hp", "current_hp", "currentHealth"), "player hp"),
        max_hp=_int(_get(payload, "max_hp", "maxHealth"), "player max_hp"),
        block=_int(_get(payload, "block", default=0), "player block"),
        energy=_int(_get(payload, "energy", default=0), "player energy"),
        powers=tuple(power_from_payload(power) for power in _sequence(_get(payload, "powers", default=()))),
        relics=tuple(str(relic) for relic in _sequence(_get(payload, "relics", default=()))),
        potions=tuple(potion_from_payload(potion, index) for index, potion in enumerate(_sequence(_get(payload, "potions", default=())))),
    )


def monster_from_payload(payload: Mapping[str, Any]) -> MonsterState:
    payload = _mapping(payload, "monster payload")
    monster_id = str(_get(payload, "monster_id", "id", "stable_id", "index"))
    hp = _int(_get(payload, "hp", "current_hp"), "monster hp")
    return MonsterState(
        monster_id=monster_id,
        name=str(_get(payload, "name", "monster_id", "id", default=monster_id)),
        hp=hp,
        max_hp=_int(_get(payload, "max_hp", default=hp), "monster max_hp"),
        block=_int(_get(payload, "block", default=0), "monster block"),
        intent=_optional_str(_get(payload, "intent", "move", "move_id", default=None)),
        intent_damage=_int(_get(payload, "intent_damage", "move_damage", default=0), "intent_damage"),
        powers=tuple(power_from_payload(power) for power in _sequence(_get(payload, "powers", default=()))),
        targetable=bool(_get(payload, "targetable", default=_get(payload, "alive", default=hp > 0))),
    )


def card_from_payload(payload: Mapping[str, Any]) -> CardState:
    payload = _mapping(payload, "card payload")
    card_id = str(_get(payload, "card_id", "id"))
    instance_id = str(_get(payload, "instance_id", "unique_id", "uuid", "card_unique_id", default=card_id))
    return CardState(
        card_id=card_id,
        name=str(_get(payload, "name", default=card_id)),
        instance_id=instance_id,
        card_type=_card_type_from_payload(payload),
        cost=_int(_get(payload, "cost", "current_cost", "cost_for_turn", default=0), "card cost"),
        upgraded=bool(_get(payload, "upgraded", default=False)),
        tags=tuple(str(tag) for tag in _sequence(_get(payload, "tags", default=()))),
    )


def potion_from_payload(payload: Any, default_slot: int = 0) -> PotionState:
    if payload is None:
        return PotionState(slot=default_slot, name=None)
    if isinstance(payload, str):
        return PotionState(slot=default_slot, name=payload or None)
    payload = _mapping(payload, "potion payload")
    return PotionState(
        slot=_int(_get(payload, "slot", "potion_slot", "index", default=default_slot), "potion slot"),
        name=_optional_str(_get(payload, "name", "potion_id", default=None)),
    )


def power_from_payload(payload: Any) -> CombatPower:
    if isinstance(payload, str):
        return CombatPower(id=payload)
    payload = _mapping(payload, "power payload")
    return CombatPower(id=str(_get(payload, "id", "power_id", "name")), amount=_int(_get(payload, "amount", default=0), "power amount"))


def combat_action_from_payload(payload: Mapping[str, Any]) -> CombatAction:
    """Convert a live legal-action payload into ``CombatAction``."""

    payload = _mapping(payload, "combat action payload")
    action_type = _action_type_from_payload(payload)
    stable_id = _optional_str(_get(payload, "stable_id", "action_id", "id", default=None))
    metadata = _action_metadata(payload)

    return CombatAction(
        action_type=action_type,
        card_instance_id=_optional_str(_get(payload, "card_instance_id", "card_unique_id", "unique_id", default=None)),
        monster_id=_optional_str(_get(payload, "monster_id", "target_id", "target_idx", "target", default=None)),
        potion_slot=_optional_int(_get(payload, "potion_slot", "potion_idx", "slot", default=None), "potion slot"),
        stable_id=stable_id,
        label=_optional_str(_get(payload, "label", default=None)),
        metadata=metadata,
    )


def combat_metrics_from_payload(payload: Mapping[str, Any] | None) -> CombatMetrics:
    payload = {} if payload is None else _mapping(payload, "metrics payload")
    return CombatMetrics(
        result=_result_from_metrics(payload),
        hp_loss=_int(_get(payload, "hp_loss", default=0), "hp_loss"),
        turns_taken=_int(_get(payload, "turns_taken", "turns", default=0), "turns_taken"),
        damage_dealt=_int(_get(payload, "damage_dealt", default=0), "damage_dealt"),
        potions_used=_int(_get(payload, "potions_used", default=0), "potions_used"),
        cards_played=_int(_get(payload, "cards_played", default=0), "cards_played"),
        search_time_ms=float(_get(payload, "search_time_ms", default=0.0)),
    )


class StsLightspeedCombatAdapter(StsLightspeedPayloadCombatBackend):
    """Compatibility name for the payload-backed live combat adapter."""


@dataclass(frozen=True)
class BackendCombatState:
    """Search-facing wrapper around a stateful live combat backend."""

    backend: CombatBackend
    observation: CombatState
    terminal: bool = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.observation, name)

    @property
    def is_terminal(self) -> bool:
        return self.observation.is_terminal or self.terminal

    def to_json(self) -> dict[str, JsonValue]:
        return self.observation.to_json()


@dataclass(frozen=True)
class BackendActionOutcome:
    """Result of one action applied through a stateful live backend."""

    before: BackendCombatState
    action: CombatAction
    after: BackendCombatState
    metrics_delta: CombatMetrics = field(default_factory=CombatMetrics)
    events: tuple[str, ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.after.is_terminal

    @property
    def result(self) -> CombatResult:
        return self.after.result


class CombatBackendSimulator:
    """Expose a stateful ``CombatBackend`` through the pure simulator protocol."""

    def wrap(self, backend: CombatBackend) -> BackendCombatState:
        observation = backend.observe()
        return BackendCombatState(
            backend=backend,
            observation=observation,
            terminal=observation.is_terminal or backend.is_terminal(),
        )

    def legal_actions(self, state: BackendCombatState) -> tuple[CombatAction, ...]:
        if state.is_terminal:
            return ()
        return tuple(state.backend.legal_actions())

    def step(self, state: BackendCombatState, action: CombatAction) -> BackendActionOutcome:
        before = state
        before_metrics = state.backend.metrics()
        state.backend.apply(action)
        after = self.wrap(state.backend)
        return BackendActionOutcome(
            before=before,
            action=action,
            after=after,
            metrics_delta=combat_metrics_delta(before_metrics, state.backend.metrics()),
        )

    def clone(self, state: BackendCombatState) -> BackendCombatState:
        return self.wrap(state.backend.clone())

    def advance_to_decision(self, state: BackendCombatState) -> BackendCombatState:
        if not state.is_terminal:
            state.backend.advance_to_player_decision()
        return self.wrap(state.backend)


def combat_metrics_delta(before: CombatMetrics, after: CombatMetrics) -> CombatMetrics:
    """Return per-action metric deltas for monotonic live counters."""

    return CombatMetrics(
        result=after.result,
        hp_loss=after.hp_loss - before.hp_loss,
        turns_taken=after.turns_taken - before.turns_taken,
        damage_dealt=after.damage_dealt - before.damage_dealt,
        potions_used=after.potions_used - before.potions_used,
        cards_played=after.cards_played - before.cards_played,
        search_time_ms=after.search_time_ms - before.search_time_ms,
    )


def _action_type_from_payload(payload: Mapping[str, Any]) -> ActionType:
    raw = str(_get(payload, "action_type", "type", "kind")).lower()
    aliases = {
        "play": ActionType.PLAY_CARD,
        "play_card": ActionType.PLAY_CARD,
        "card": ActionType.PLAY_CARD,
        "potion": ActionType.USE_POTION,
        "use_potion": ActionType.USE_POTION,
        "drink": ActionType.USE_POTION,
        "drink_potion": ActionType.USE_POTION,
        "discard": ActionType.DISCARD_POTION,
        "discard_potion": ActionType.DISCARD_POTION,
        "select": ActionType.SELECT_CARD,
        "select_card": ActionType.SELECT_CARD,
        "card_select": ActionType.SELECT_CARD,
        "end": ActionType.END_TURN,
        "end_turn": ActionType.END_TURN,
    }
    if raw not in aliases:
        raise ValueError(f"unsupported live combat action type: {raw}")
    return aliases[raw]


def _card_type_from_payload(payload: Mapping[str, Any]) -> CardType:
    raw = str(_get(payload, "card_type", "type", default=CardType.STATUS.value)).lower()
    aliases = {
        "attack": CardType.ATTACK,
        "skill": CardType.SKILL,
        "power": CardType.POWER,
        "status": CardType.STATUS,
        "curse": CardType.CURSE,
    }
    return aliases.get(raw, CardType.STATUS)


def _phase_from_payload(payload: Mapping[str, Any]) -> CombatPhase:
    combat_payload = _mapping(_get(payload, "combat", default={}), "combat payload")
    raw = str(_get(combat_payload, "phase", default=_get(payload, "phase", "input_state", default="player_turn"))).lower()
    aliases = {
        "player_turn": CombatPhase.PLAYER_TURN,
        "player_normal": CombatPhase.PLAYER_TURN,
        "card_select": CombatPhase.PLAYER_TURN,
        "enemy_turn": CombatPhase.ENEMY_TURN,
        "monster_turn": CombatPhase.ENEMY_TURN,
        "complete": CombatPhase.COMPLETE,
        "battle_complete": CombatPhase.COMPLETE,
    }
    return aliases.get(raw, CombatPhase.PLAYER_TURN)


def _result_from_payload(payload: Mapping[str, Any]) -> CombatResult:
    combat_payload = _mapping(_get(payload, "combat", default={}), "combat payload")
    return _result_from_raw(_get(combat_payload, "result", "outcome", default=_get(payload, "result", "outcome", default="ongoing")))


def _result_from_metrics(payload: Mapping[str, Any]) -> CombatResult:
    return _result_from_raw(_get(payload, "result", "outcome", default=CombatResult.ONGOING.value))


def _result_from_raw(raw: Any) -> CombatResult:
    value = str(raw).lower()
    aliases = {
        "ongoing": CombatResult.ONGOING,
        "none": CombatResult.ONGOING,
        "player_victory": CombatResult.PLAYER_VICTORY,
        "victory": CombatResult.PLAYER_VICTORY,
        "won": CombatResult.PLAYER_VICTORY,
        "player_death": CombatResult.PLAYER_DEATH,
        "player_loss": CombatResult.PLAYER_DEATH,
        "loss": CombatResult.PLAYER_DEATH,
        "death": CombatResult.PLAYER_DEATH,
        "lost": CombatResult.PLAYER_DEATH,
    }
    if value not in aliases:
        raise ValueError(f"unsupported combat result: {raw}")
    return aliases[value]


def _zone(payload: Mapping[str, Any], *names: str) -> Sequence[Any]:
    for name in names:
        if name in payload:
            return _sequence(payload[name])
    piles = _get(payload, "piles", "cards", default={})
    if isinstance(piles, Mapping):
        for name in names:
            if name in piles:
                return _sequence(piles[name])
    return ()


def _metadata_payload(payload: Mapping[str, Any]) -> dict[str, JsonValue]:
    return {str(key): _json_value(value) for key, value in payload.items() if key != "sts_payload"}


def _state_metadata(payload: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, JsonValue]:
    metadata = dict(base)
    nested = _get(payload, "metadata", default={})
    if isinstance(nested, Mapping):
        metadata.update(nested)
    return _json_mapping(metadata)


def _action_metadata(payload: Mapping[str, Any]) -> dict[str, JsonValue]:
    metadata = _json_mapping({"sts_payload": _metadata_payload(payload)})
    for key, value in payload.items():
        if key in {
            "action_type",
            "type",
            "kind",
            "stable_id",
            "action_id",
            "id",
            "card_instance_id",
            "card_unique_id",
            "unique_id",
            "monster_id",
            "target_id",
            "target_idx",
            "target",
            "potion_slot",
            "potion_idx",
            "slot",
            "label",
        }:
            continue
        metadata[str(key)] = _json_value(value)
    return metadata


def _live_identity_matches(selected: CombatAction, current: CombatAction) -> bool:
    selected_identity = _live_identity(selected)
    if not selected_identity:
        return True
    current_identity = _live_identity(current)
    return all(current_identity.get(key) == value for key, value in selected_identity.items())


def _live_identity(action: CombatAction) -> dict[str, JsonValue]:
    identity_keys = ("binding_action_id", "decision_id")
    return {
        key: action.metadata[key]
        for key in identity_keys
        if key in action.metadata and action.metadata[key] is not None
    }


def _json_mapping(payload: Mapping[str, Any]) -> dict[str, JsonValue]:
    return {str(key): _json_value(value) for key, value in payload.items() if value is not None}


def _json_value(value: Any) -> JsonValue:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _call(handle: Any, method_name: str) -> Any:
    method = getattr(handle, method_name, None)
    if method is None:
        raise TypeError(f"combat handle does not expose {method_name}()")
    return method()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError("payload value must be a sequence")


def _get(payload: Mapping[str, Any], *names: str, default: Any = ...) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    if default is not ...:
        return default
    raise KeyError(f"missing required payload field: {'/'.join(names)}")


def _int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _int(value, label)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
