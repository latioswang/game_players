"""JSONL-safe trajectory records for Slay the Spire combat logs.

The combat adapter is expected to hand this module plain Python state/action
payloads.  These helpers validate and copy those payloads so each trajectory can
be written as one strict JSON object per JSONL row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, TypeAlias


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict: TypeAlias = dict[str, JsonValue]

FORMAT_VERSION = 1
TRAJECTORY_KIND = "sts_combat_trajectory"


@dataclass(frozen=True)
class CombatStateLog:
    """One combat state observed by a policy or value function."""

    step_index: int
    state: JsonDict
    turn: int | None = None
    phase: str | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int(self.step_index, "step_index")
        if self.turn is not None:
            _require_non_negative_int(self.turn, "turn")
        object.__setattr__(self, "state", _require_json_dict(self.state, "state"))

    def to_json(self) -> JsonDict:
        return _drop_none(
            {
                "step_index": self.step_index,
                "turn": self.turn,
                "phase": self.phase,
                "state": _require_json_dict(self.state, "state"),
            }
        )

    @classmethod
    def from_json(cls, data: JsonDict) -> "CombatStateLog":
        data = _require_json_dict(data, "state_log")
        return cls(
            step_index=_require_int(data.get("step_index"), "step_index"),
            turn=_optional_int(data.get("turn"), "turn"),
            phase=_optional_str(data.get("phase"), "phase"),
            state=_require_json_dict(data.get("state"), "state"),
        )


@dataclass(frozen=True)
class CombatActionLog:
    """Legal and selected action data for one combat decision."""

    step_index: int
    legal_actions: list[JsonDict]
    selected_action: JsonDict | None = None
    policy: str | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int(self.step_index, "step_index")
        object.__setattr__(
            self,
            "legal_actions",
            [
                _require_json_dict(action, f"legal_actions[{index}]")
                for index, action in enumerate(self.legal_actions)
            ],
        )
        if self.selected_action is not None:
            object.__setattr__(
                self,
                "selected_action",
                _require_json_dict(self.selected_action, "selected_action"),
            )

    def to_json(self) -> JsonDict:
        return _drop_none(
            {
                "step_index": self.step_index,
                "policy": self.policy,
                "legal_actions": [
                    _require_json_dict(action, f"legal_actions[{index}]")
                    for index, action in enumerate(self.legal_actions)
                ],
                "selected_action": None
                if self.selected_action is None
                else _require_json_dict(self.selected_action, "selected_action"),
            }
        )

    @classmethod
    def from_json(cls, data: JsonDict) -> "CombatActionLog":
        data = _require_json_dict(data, "action_log")
        raw_legal_actions = data.get("legal_actions")
        if not isinstance(raw_legal_actions, list):
            raise TypeError("legal_actions must be a list")
        selected_action = data.get("selected_action")
        return cls(
            step_index=_require_int(data.get("step_index"), "step_index"),
            policy=_optional_str(data.get("policy"), "policy"),
            legal_actions=[
                _require_json_dict(action, f"legal_actions[{index}]")
                for index, action in enumerate(raw_legal_actions)
            ],
            selected_action=None
            if selected_action is None
            else _require_json_dict(selected_action, "selected_action"),
        )


@dataclass(frozen=True)
class CombatValueLog:
    """Value estimates associated with one combat state or action choice."""

    step_index: int
    value: float | None = None
    action_values: dict[str, float] = field(default_factory=dict)
    details: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_negative_int(self.step_index, "step_index")
        if self.value is not None:
            object.__setattr__(self, "value", _require_finite_float(self.value, "value"))
        object.__setattr__(
            self,
            "action_values",
            {
                _require_str(action_id, "action_values key"): _require_finite_float(
                    estimate, f"action_values[{action_id!r}]"
                )
                for action_id, estimate in self.action_values.items()
            },
        )
        object.__setattr__(self, "details", _require_json_dict(self.details, "details"))

    def to_json(self) -> JsonDict:
        return _drop_none(
            {
                "step_index": self.step_index,
                "value": self.value,
                "action_values": dict(self.action_values),
                "details": _require_json_dict(self.details, "details"),
            }
        )

    @classmethod
    def from_json(cls, data: JsonDict) -> "CombatValueLog":
        data = _require_json_dict(data, "value_log")
        raw_action_values = data.get("action_values", {})
        if not isinstance(raw_action_values, dict):
            raise TypeError("action_values must be a dict")
        return cls(
            step_index=_require_int(data.get("step_index"), "step_index"),
            value=_optional_float(data.get("value"), "value"),
            action_values={
                _require_str(action_id, "action_values key"): _require_finite_float(
                    estimate, f"action_values[{action_id!r}]"
                )
                for action_id, estimate in raw_action_values.items()
            },
            details=_require_json_dict(data.get("details", {}), "details"),
        )


@dataclass(frozen=True)
class CombatStepLog:
    """State/action/value record for one combat policy step."""

    state: CombatStateLog
    action: CombatActionLog | None = None
    value: CombatValueLog | None = None
    reward: float | None = None
    done: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, CombatStateLog):
            raise TypeError("state must be a CombatStateLog")
        if self.action is not None and not isinstance(self.action, CombatActionLog):
            raise TypeError("action must be a CombatActionLog or None")
        if self.action is not None and self.action.step_index != self.state.step_index:
            raise ValueError("action.step_index must match state.step_index")
        if self.value is not None and not isinstance(self.value, CombatValueLog):
            raise TypeError("value must be a CombatValueLog or None")
        if self.value is not None and self.value.step_index != self.state.step_index:
            raise ValueError("value.step_index must match state.step_index")
        if self.reward is not None:
            object.__setattr__(self, "reward", _require_finite_float(self.reward, "reward"))

    @property
    def step_index(self) -> int:
        return self.state.step_index

    def to_json(self) -> JsonDict:
        return _drop_none(
            {
                "state": self.state.to_json(),
                "action": None if self.action is None else self.action.to_json(),
                "value": None if self.value is None else self.value.to_json(),
                "reward": self.reward,
                "done": self.done,
            }
        )

    @classmethod
    def from_json(cls, data: JsonDict) -> "CombatStepLog":
        data = _require_json_dict(data, "step_log")
        action = data.get("action")
        value = data.get("value")
        return cls(
            state=CombatStateLog.from_json(_require_json_dict(data.get("state"), "state")),
            action=None
            if action is None
            else CombatActionLog.from_json(_require_json_dict(action, "action")),
            value=None
            if value is None
            else CombatValueLog.from_json(_require_json_dict(value, "value")),
            reward=_optional_float(data.get("reward"), "reward"),
            done=_require_bool(data.get("done", False), "done"),
        )


@dataclass(frozen=True)
class CombatTrajectory:
    """One combat-only trajectory suitable for one JSONL row."""

    trajectory_id: str
    seed: int
    combat_id: str
    steps: list[CombatStepLog] = field(default_factory=list)
    outcome: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)
    format_version: int = FORMAT_VERSION
    kind: str = TRAJECTORY_KIND

    def __post_init__(self) -> None:
        _require_non_negative_int(self.seed, "seed")
        if self.format_version != FORMAT_VERSION:
            raise ValueError(f"format_version must be {FORMAT_VERSION}")
        if self.kind != TRAJECTORY_KIND:
            raise ValueError(f"kind must be {TRAJECTORY_KIND!r}")
        object.__setattr__(self, "trajectory_id", _require_str(self.trajectory_id, "trajectory_id"))
        object.__setattr__(self, "combat_id", _require_str(self.combat_id, "combat_id"))
        object.__setattr__(self, "steps", list(self.steps))
        previous_step_index = -1
        for index, step in enumerate(self.steps):
            if not isinstance(step, CombatStepLog):
                raise TypeError(f"steps[{index}] must be a CombatStepLog")
            if step.step_index <= previous_step_index:
                raise ValueError("steps must have strictly increasing step_index values")
            previous_step_index = step.step_index
        object.__setattr__(self, "outcome", _require_json_dict(self.outcome, "outcome"))
        object.__setattr__(self, "metadata", _require_json_dict(self.metadata, "metadata"))

    def to_json(self) -> JsonDict:
        return {
            "format_version": self.format_version,
            "kind": self.kind,
            "trajectory_id": self.trajectory_id,
            "seed": self.seed,
            "combat_id": self.combat_id,
            "steps": [step.to_json() for step in self.steps],
            "outcome": _require_json_dict(self.outcome, "outcome"),
            "metadata": _require_json_dict(self.metadata, "metadata"),
        }

    @classmethod
    def from_json(cls, data: JsonDict) -> "CombatTrajectory":
        data = _require_json_dict(data, "trajectory")
        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list):
            raise TypeError("steps must be a list")
        return cls(
            format_version=_require_int(data.get("format_version"), "format_version"),
            kind=_require_str(data.get("kind"), "kind"),
            trajectory_id=_require_str(data.get("trajectory_id"), "trajectory_id"),
            seed=_require_int(data.get("seed"), "seed"),
            combat_id=_require_str(data.get("combat_id"), "combat_id"),
            steps=[
                CombatStepLog.from_json(_require_json_dict(step, f"steps[{index}]"))
                for index, step in enumerate(raw_steps)
            ],
            outcome=_require_json_dict(data.get("outcome", {}), "outcome"),
            metadata=_require_json_dict(data.get("metadata", {}), "metadata"),
        )

    def to_json_line(self) -> str:
        return json.dumps(
            self.to_json(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_json_line(cls, line: str) -> "CombatTrajectory":
        return cls.from_json(json.loads(line))


def validate_json_value(value: Any, path: str = "value") -> JsonValue:
    """Return a JSON-safe copy of ``value`` or raise for unsafe data."""

    return _normalize_json_value(value, path)


def trajectory_to_json(trajectory: CombatTrajectory) -> JsonDict:
    return trajectory.to_json()


def trajectory_from_json(data: JsonDict) -> CombatTrajectory:
    return CombatTrajectory.from_json(data)


def trajectory_to_json_line(trajectory: CombatTrajectory) -> str:
    return trajectory.to_json_line()


def trajectory_from_json_line(line: str) -> CombatTrajectory:
    return CombatTrajectory.from_json_line(line)


def write_jsonl(
    path: str | Path,
    trajectories: Iterable[CombatTrajectory],
    append: bool = False,
) -> None:
    mode = "a" if append else "w"
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open(mode, encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(trajectory.to_json_line())
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[CombatTrajectory]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [
            CombatTrajectory.from_json_line(line)
            for line in handle
            if line.strip()
        ]


def _normalize_json_value(value: Any, path: str) -> JsonValue:
    if value is None or isinstance(value, str) or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _require_finite_float(value, path)
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            _require_str(key, f"{path} key"): _normalize_json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    raise TypeError(f"{path} must contain only JSON-safe plain Python values")


def _require_json_dict(value: Any, path: str) -> JsonDict:
    normalized = _normalize_json_value(value, path)
    if not isinstance(normalized, dict):
        raise TypeError(f"{path} must be a dict")
    return normalized


def _drop_none(data: dict[str, JsonValue]) -> JsonDict:
    return {key: value for key, value in data.items() if value is not None}


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a bool")
    return value


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    return value


def _optional_str(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, path)


def _require_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an int")
    return value


def _optional_int(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, path)


def _require_non_negative_int(value: Any, path: str) -> None:
    number = _require_int(value, path)
    if number < 0:
        raise ValueError(f"{path} must be non-negative")


def _require_finite_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


def _optional_float(value: Any, path: str) -> float | None:
    if value is None:
        return None
    return _require_finite_float(value, path)
