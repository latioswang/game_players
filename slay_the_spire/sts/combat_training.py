"""No-dependency training helpers for combat value functions.

The helpers in this module intentionally operate on plain Python data and the
JSON-safe trajectory records from ``combat_trajectory``.  They are small enough
for deterministic fixtures while still being useful as the first learned value
baseline for combat search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from .combat_value import _snapshot as _snapshot_combat_state
    from .combat_value import _terminal_outcome as _terminal_outcome
except ImportError:
    from combat_value import _snapshot as _snapshot_combat_state
    from combat_value import _terminal_outcome as _terminal_outcome


FeatureMap = Mapping[str, float]
FeatureExtractor = Callable[[Any], FeatureMap]

BIAS_FEATURE = "__bias__"


@dataclass(frozen=True)
class CombatValueTrainingExample:
    """One numeric value target paired with combat-state features."""

    features: FeatureMap
    target: float
    weight: float = 1.0
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _finite_feature_dict(self.features, "features"))
        object.__setattr__(self, "target", _finite_float(self.target, "target"))
        object.__setattr__(self, "weight", _positive_finite_float(self.weight, "weight"))


@dataclass(frozen=True)
class LinearCombatValueFunction:
    """Sparse linear combat value function.

    Missing features evaluate as zero.  The intercept is stored separately so
    serialized weights remain easy to inspect and can be feature-pruned without
    special casing a bias entry.
    """

    weights: FeatureMap = field(default_factory=dict)
    intercept: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _finite_feature_dict(self.weights, "weights"))
        object.__setattr__(self, "intercept", _finite_float(self.intercept, "intercept"))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.weights)

    def predict_features(self, features: Mapping[str, Any]) -> float:
        feature_values = _finite_feature_dict(features, "features")
        return self.intercept + sum(
            weight * feature_values.get(name, 0.0)
            for name, weight in self.weights.items()
        )

    def evaluate_features(self, features: Mapping[str, Any]) -> float:
        return self.predict_features(features)

    def predict_state(self, state: Any, feature_extractor: FeatureExtractor | None = None) -> float:
        extractor = feature_extractor or combat_state_features
        return self.predict_features(extractor(state))

    def evaluate(self, state: Any) -> float:
        return self.predict_state(state)

    def to_json(self) -> dict[str, Any]:
        return {
            "intercept": self.intercept,
            "weights": dict(self.weights),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "LinearCombatValueFunction":
        return cls(
            intercept=_finite_float(data.get("intercept", 0.0), "intercept"),
            weights=_require_mapping(data.get("weights", {}), "weights"),
        )


@dataclass(frozen=True)
class CombatValueTrainingSummary:
    """Fit diagnostics for a deterministic linear value-function run."""

    example_count: int
    feature_names: tuple[str, ...]
    l2: float
    mean_squared_error: float


@dataclass(frozen=True)
class CombatValueTrainingResult:
    """A fitted linear value function and basic diagnostics."""

    model: LinearCombatValueFunction
    summary: CombatValueTrainingSummary


def combat_state_features(state: Any) -> dict[str, float]:
    """Extract stable numeric features from a combat state-shaped object."""

    snapshot = _snapshot_combat_state(state)
    terminal = _terminal_outcome(state, snapshot)
    player_max_hp = max(1.0, float(snapshot.player_max_hp))
    return {
        "player_hp": float(snapshot.player_hp),
        "player_max_hp": float(snapshot.player_max_hp),
        "player_hp_ratio": float(snapshot.player_hp) / player_max_hp,
        "player_block": float(snapshot.player_block),
        "energy": float(snapshot.energy),
        "incoming_damage": float(snapshot.incoming_damage),
        "expected_hp_loss": float(snapshot.expected_hp_loss),
        "enemy_hp": float(snapshot.enemy_hp),
        "alive_enemies": float(snapshot.alive_enemies),
        "hand_size": float(snapshot.hand_size),
        "draw_size": float(snapshot.draw_size),
        "discard_size": float(snapshot.discard_size),
        "player_strength": float(snapshot.player_strength),
        "player_dexterity": float(snapshot.player_dexterity),
        "player_weak": float(snapshot.player_weak),
        "player_vulnerable": float(snapshot.player_vulnerable),
        "enemy_strength": float(snapshot.enemy_strength),
        "enemy_weak": float(snapshot.enemy_weak),
        "enemy_vulnerable": float(snapshot.enemy_vulnerable),
        "turn": float(snapshot.turn),
        "terminal_victory": 1.0 if terminal == "victory" else 0.0,
        "terminal_defeat": 1.0 if terminal == "defeat" else 0.0,
    }


def combat_value_training_examples(
    records: Iterable[Any],
    *,
    feature_extractor: FeatureExtractor = combat_state_features,
    skip_missing_targets: bool = True,
) -> tuple[CombatValueTrainingExample, ...]:
    """Normalize trajectories, step logs, or value records into examples.

    Supported inputs include ``CombatTrajectory`` objects, ``CombatStepLog``
    objects, their ``to_json`` dictionaries, ``CombatValueTrainingExample``
    instances, ``(features, target)`` pairs, and mappings with either
    ``features``/``target`` or ``state``/``value``.
    """

    examples: list[CombatValueTrainingExample] = []
    for record in records:
        examples.extend(
            _examples_from_record(
                record,
                feature_extractor=feature_extractor,
                skip_missing_targets=skip_missing_targets,
                trajectory_id=None,
            )
        )
    return tuple(examples)


def combat_value_training_examples_from_trajectories(
    trajectories: Iterable[Any],
    *,
    feature_extractor: FeatureExtractor = combat_state_features,
    skip_missing_targets: bool = True,
) -> tuple[CombatValueTrainingExample, ...]:
    """Collect state-value examples from trajectory step value logs."""

    return combat_value_training_examples(
        trajectories,
        feature_extractor=feature_extractor,
        skip_missing_targets=skip_missing_targets,
    )


def fit_linear_combat_value_function(
    records: Iterable[Any],
    *,
    feature_names: Sequence[str] | None = None,
    feature_extractor: FeatureExtractor = combat_state_features,
    l2: float = 1e-6,
    skip_missing_targets: bool = True,
) -> CombatValueTrainingResult:
    """Fit a sparse linear value function with closed-form ridge regression."""

    examples = combat_value_training_examples(
        records,
        feature_extractor=feature_extractor,
        skip_missing_targets=skip_missing_targets,
    )
    if not examples:
        raise ValueError("at least one training example is required")

    l2 = _non_negative_finite_float(l2, "l2")
    names = _feature_names(examples, feature_names)
    coefficients = _fit_weighted_normal_equations(examples, names, l2)
    intercept = coefficients[0]
    weights = {
        name: coefficient
        for name, coefficient in zip(names, coefficients[1:])
        if coefficient != 0.0
    }
    model = LinearCombatValueFunction(weights=weights, intercept=intercept)
    mse = _mean_squared_error(model, examples)
    return CombatValueTrainingResult(
        model=model,
        summary=CombatValueTrainingSummary(
            example_count=len(examples),
            feature_names=names,
            l2=l2,
            mean_squared_error=mse,
        ),
    )


def train_linear_combat_value_function(
    records: Iterable[Any],
    *,
    feature_names: Sequence[str] | None = None,
    feature_extractor: FeatureExtractor = combat_state_features,
    l2: float = 1e-6,
    skip_missing_targets: bool = True,
) -> LinearCombatValueFunction:
    """Fit and return only the linear combat value function."""

    return fit_linear_combat_value_function(
        records,
        feature_names=feature_names,
        feature_extractor=feature_extractor,
        l2=l2,
        skip_missing_targets=skip_missing_targets,
    ).model


def fit_linear_combat_value(
    records: Iterable[Any],
    *,
    feature_names: Sequence[str] | None = None,
    feature_extractor: FeatureExtractor = combat_state_features,
    l2: float = 1e-6,
    skip_missing_targets: bool = True,
) -> CombatValueTrainingResult:
    """Alias for ``fit_linear_combat_value_function``."""

    return fit_linear_combat_value_function(
        records,
        feature_names=feature_names,
        feature_extractor=feature_extractor,
        l2=l2,
        skip_missing_targets=skip_missing_targets,
    )


def train_linear_combat_value(
    records: Iterable[Any],
    *,
    feature_names: Sequence[str] | None = None,
    feature_extractor: FeatureExtractor = combat_state_features,
    l2: float = 1e-6,
    skip_missing_targets: bool = True,
) -> LinearCombatValueFunction:
    """Alias for ``train_linear_combat_value_function``."""

    return train_linear_combat_value_function(
        records,
        feature_names=feature_names,
        feature_extractor=feature_extractor,
        l2=l2,
        skip_missing_targets=skip_missing_targets,
    )


def _examples_from_record(
    record: Any,
    *,
    feature_extractor: FeatureExtractor,
    skip_missing_targets: bool,
    trajectory_id: str | None,
) -> list[CombatValueTrainingExample]:
    if isinstance(record, CombatValueTrainingExample):
        return [record]

    if _has_steps(record):
        record_trajectory_id = str(_get_value(record, "trajectory_id", default=trajectory_id or "trajectory"))
        return [
            example
            for step in _get_steps(record)
            for example in _examples_from_record(
                step,
                feature_extractor=feature_extractor,
                skip_missing_targets=skip_missing_targets,
                trajectory_id=record_trajectory_id,
            )
        ]

    pair = _feature_target_pair(record)
    if pair is not None:
        features, target = pair
        return [CombatValueTrainingExample(features=features, target=target)]

    state_record = _get_value(record, "state", default=None)
    value_record = _get_value(record, "value", default=None)
    if state_record is None or value_record is None:
        if skip_missing_targets:
            return []
        raise ValueError("record must contain state and value data")

    target = _target_from_value_record(value_record)
    if target is None:
        if skip_missing_targets:
            return []
        raise ValueError("value record must contain a numeric value target")

    state = _state_payload(state_record)
    source = _source_key(record, trajectory_id)
    return [
        CombatValueTrainingExample(
            features=feature_extractor(state),
            target=target,
            source=source,
        )
    ]


def _feature_target_pair(record: Any) -> tuple[FeatureMap, float] | None:
    if isinstance(record, Mapping):
        features = record.get("features")
        target = record.get("target")
        if features is not None and target is not None:
            return _require_mapping(features, "features"), _finite_float(target, "target")
    if isinstance(record, tuple) and len(record) == 2:
        features, target = record
        return _require_mapping(features, "features"), _finite_float(target, "target")
    return None


def _target_from_value_record(value_record: Any) -> float | None:
    if isinstance(value_record, (int, float)) and not isinstance(value_record, bool):
        return _finite_float(value_record, "value")
    value = _get_value(value_record, "value", default=None)
    if value is None:
        value = _get_value(value_record, "target", default=None)
    if value is None:
        return None
    return _finite_float(value, "value")


def _state_payload(state_record: Any) -> Any:
    inner_state = _get_value(state_record, "state", default=None)
    if inner_state is not None:
        return inner_state
    return state_record


def _source_key(record: Any, trajectory_id: str | None) -> str | None:
    step_index = _get_value(record, "step_index", default=None)
    if step_index is None:
        state_record = _get_value(record, "state", default=None)
        step_index = _get_value(state_record, "step_index", default=None)
    if trajectory_id is None and step_index is None:
        return None
    prefix = trajectory_id or "record"
    return f"{prefix}:{step_index}" if step_index is not None else prefix


def _has_steps(record: Any) -> bool:
    return _get_value(record, "steps", default=None) is not None


def _get_steps(record: Any) -> Iterable[Any]:
    steps = _get_value(record, "steps", default=())
    if isinstance(steps, Iterable) and not isinstance(steps, (str, bytes, Mapping)):
        return steps
    raise TypeError("steps must be an iterable of step records")


def _get_value(source: Any, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _feature_names(
    examples: Sequence[CombatValueTrainingExample],
    feature_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if feature_names is not None:
        names = tuple(feature_names)
        for name in names:
            if not isinstance(name, str) or not name:
                raise ValueError("feature_names must contain non-empty strings")
            if name == BIAS_FEATURE:
                raise ValueError(f"{BIAS_FEATURE!r} is reserved for the model intercept")
        return names

    return tuple(sorted({name for example in examples for name in example.features}))


def _fit_weighted_normal_equations(
    examples: Sequence[CombatValueTrainingExample],
    feature_names: Sequence[str],
    l2: float,
) -> list[float]:
    dimension = len(feature_names) + 1
    matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    vector = [0.0 for _ in range(dimension)]

    for example in examples:
        row = [1.0]
        row.extend(example.features.get(name, 0.0) for name in feature_names)
        for i, left in enumerate(row):
            weighted_left = example.weight * left
            vector[i] += weighted_left * example.target
            for j, right in enumerate(row):
                matrix[i][j] += weighted_left * right

    for index in range(1, dimension):
        matrix[index][index] += l2

    return _solve_linear_system(matrix, vector)


def _solve_linear_system(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    size = len(vector)
    augmented = [list(row) + [vector[index]] for index, row in enumerate(matrix)]

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        pivot = augmented[pivot_row][column]
        if abs(pivot) <= 1e-12:
            raise ValueError("training system is singular; use l2 > 0 or fewer features")
        if pivot_row != column:
            augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]

        for row in range(column + 1, size):
            factor = augmented[row][column] / augmented[column][column]
            if factor == 0.0:
                continue
            for item in range(column, size + 1):
                augmented[row][item] -= factor * augmented[column][item]

    solution = [0.0 for _ in range(size)]
    for row in range(size - 1, -1, -1):
        remainder = augmented[row][size] - sum(
            augmented[row][column] * solution[column]
            for column in range(row + 1, size)
        )
        solution[row] = remainder / augmented[row][row]
    return solution


def _mean_squared_error(
    model: LinearCombatValueFunction,
    examples: Sequence[CombatValueTrainingExample],
) -> float:
    total_weight = sum(example.weight for example in examples)
    return sum(
        example.weight * (model.predict_features(example.features) - example.target) ** 2
        for example in examples
    ) / total_weight


def _finite_feature_dict(features: Mapping[str, Any], path: str) -> dict[str, float]:
    data = _require_mapping(features, path)
    normalized: dict[str, float] = {}
    for name, value in data.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path} keys must be non-empty strings")
        if name == BIAS_FEATURE:
            raise ValueError(f"{BIAS_FEATURE!r} is reserved for the model intercept")
        normalized[name] = _finite_float(value, f"{path}[{name!r}]")
    return dict(sorted(normalized.items()))


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _finite_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


def _positive_finite_float(value: Any, path: str) -> float:
    number = _finite_float(value, path)
    if number <= 0:
        raise ValueError(f"{path} must be positive")
    return number


def _non_negative_finite_float(value: Any, path: str) -> float:
    number = _finite_float(value, path)
    if number < 0:
        raise ValueError(f"{path} must be non-negative")
    return number


__all__ = [
    "BIAS_FEATURE",
    "CombatValueTrainingExample",
    "CombatValueTrainingResult",
    "CombatValueTrainingSummary",
    "FeatureExtractor",
    "FeatureMap",
    "LinearCombatValueFunction",
    "combat_state_features",
    "combat_value_training_examples",
    "combat_value_training_examples_from_trajectories",
    "fit_linear_combat_value",
    "fit_linear_combat_value_function",
    "train_linear_combat_value",
    "train_linear_combat_value_function",
]
