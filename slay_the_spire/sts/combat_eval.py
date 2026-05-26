"""Combat-only evaluation metric aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class CombatEvaluationRecord:
    """Metrics for one completed combat."""

    combat_id: str
    survived: bool
    hp_before: int
    hp_after: int
    turns: int
    potions_used: int = 0
    search_time_ms: float = 0.0
    elite: bool = False
    boss: bool = False

    def __post_init__(self) -> None:
        _require_non_negative(self.hp_before, "hp_before")
        _require_non_negative(self.hp_after, "hp_after")
        _require_non_negative(self.turns, "turns")
        _require_non_negative(self.potions_used, "potions_used")
        if self.search_time_ms < 0:
            raise ValueError("search_time_ms must be non-negative")

    @property
    def hp_loss(self) -> int:
        return max(0, self.hp_before - self.hp_after)


@dataclass(frozen=True)
class CombatEvaluationSummary:
    """Aggregate combat-only metrics for a deterministic seed suite."""

    combat_count: int
    survival_rate: float
    average_hp_loss: float
    median_hp_loss: float
    average_turns: float
    average_potions_used: float
    average_search_time_ms: float
    elite_survival_rate: float | None = None
    boss_survival_rate: float | None = None
    worst_hp_loss: int = 0
    regression_combat_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CombatEvaluationBaseline:
    """Comparable baseline metrics keyed by combat id."""

    hp_loss_by_combat_id: dict[str, int] = field(default_factory=dict)
    survived_by_combat_id: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricStats:
    """Aggregate statistics for one numeric combat metric."""

    count: int
    total: float
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int(self.count, "count")
        _require_finite_number(self.total, "total")
        if self.count == 0:
            if self.total != 0:
                raise ValueError("empty stats must have zero total")
            if self.minimum is not None or self.maximum is not None:
                raise ValueError("empty stats must not have min/max")
            return

        if self.minimum is None or self.maximum is None:
            raise ValueError("non-empty stats require min/max")
        _require_finite_number(self.minimum, "minimum")
        _require_finite_number(self.maximum, "maximum")
        if self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "MetricStats":
        numbers = [_require_finite_number(value, "value") for value in values]
        if not numbers:
            return cls(count=0, total=0.0)
        return cls(
            count=len(numbers),
            total=sum(numbers),
            minimum=min(numbers),
            maximum=max(numbers),
        )

    def to_json(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "total": self.total,
            "mean": self.mean,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True)
class CombatEvalMetrics:
    """Reward-independent outcome metrics for one completed combat."""

    survived: bool
    hp_loss: int
    turns: int
    potions_used: int = 0
    search_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.survived, bool):
            raise TypeError("survived must be a bool")
        _require_non_negative_int(self.hp_loss, "hp_loss")
        _require_non_negative_int(self.turns, "turns")
        _require_non_negative_int(self.potions_used, "potions_used")
        object.__setattr__(
            self,
            "search_time_ms",
            _require_non_negative_number(self.search_time_ms, "search_time_ms"),
        )

    @property
    def survival(self) -> int:
        return int(self.survived)

    def to_json(self) -> dict[str, bool | int | float]:
        return {
            "survived": self.survived,
            "hp_loss": self.hp_loss,
            "turns": self.turns,
            "potions_used": self.potions_used,
            "search_time_ms": self.search_time_ms,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CombatEvalMetrics":
        """Normalize a dict-shaped combat metric record."""

        return cls(
            survived=_survived_from_data(data),
            hp_loss=_int_from_data(data, "hp_loss"),
            turns=_int_from_data(data, "turns", "turns_taken"),
            potions_used=_int_from_data(data, "potions_used", "potion_usage", default=0),
            search_time_ms=_float_from_data(data, "search_time_ms", default=0.0),
        )

    @classmethod
    def from_object(cls, metrics: Any) -> "CombatEvalMetrics":
        """Normalize a plain object or dataclass with combat metric attributes."""

        if isinstance(metrics, Mapping):
            return cls.from_mapping(metrics)
        return cls.from_mapping(_object_mapping(metrics))


@dataclass(frozen=True)
class CombatEvalSummary:
    """Aggregate combat-only policy metrics for a batch of combats."""

    combats: int
    survival: MetricStats
    hp_loss: MetricStats
    turns: MetricStats
    potions_used: MetricStats
    search_time_ms: MetricStats

    def __post_init__(self) -> None:
        _require_non_negative_int(self.combats, "combats")
        for name in ("survival", "hp_loss", "turns", "potions_used", "search_time_ms"):
            stats = getattr(self, name)
            if not isinstance(stats, MetricStats):
                raise TypeError(f"{name} must be a MetricStats")
            if stats.count != self.combats:
                raise ValueError(f"{name}.count must match combats")

    @property
    def survival_rate(self) -> float:
        return self.survival.mean

    @property
    def average_hp_loss(self) -> float:
        return self.hp_loss.mean

    @property
    def average_turns(self) -> float:
        return self.turns.mean

    @property
    def average_potions_used(self) -> float:
        return self.potions_used.mean

    @property
    def average_search_time_ms(self) -> float:
        return self.search_time_ms.mean

    def to_json(self) -> dict[str, Any]:
        return {
            "combats": self.combats,
            "survival_rate": self.survival_rate,
            "average_hp_loss": self.average_hp_loss,
            "average_turns": self.average_turns,
            "average_potions_used": self.average_potions_used,
            "average_search_time_ms": self.average_search_time_ms,
            "survival": self.survival.to_json(),
            "hp_loss": self.hp_loss.to_json(),
            "turns": self.turns.to_json(),
            "potions_used": self.potions_used.to_json(),
            "search_time_ms": self.search_time_ms.to_json(),
        }


def summarize_combat_records(
    records: Iterable[CombatEvaluationRecord],
    *,
    baseline: CombatEvaluationBaseline | None = None,
) -> CombatEvaluationSummary:
    """Summarize combat records with optional regression detection."""

    combat_records = tuple(records)
    if not combat_records:
        return CombatEvaluationSummary(
            combat_count=0,
            survival_rate=0.0,
            average_hp_loss=0.0,
            median_hp_loss=0.0,
            average_turns=0.0,
            average_potions_used=0.0,
            average_search_time_ms=0.0,
        )

    hp_losses = [record.hp_loss for record in combat_records]
    elites = tuple(record for record in combat_records if record.elite)
    bosses = tuple(record for record in combat_records if record.boss)

    return CombatEvaluationSummary(
        combat_count=len(combat_records),
        survival_rate=_rate(record.survived for record in combat_records),
        average_hp_loss=statistics.fmean(hp_losses),
        median_hp_loss=float(statistics.median(hp_losses)),
        average_turns=statistics.fmean(record.turns for record in combat_records),
        average_potions_used=statistics.fmean(record.potions_used for record in combat_records),
        average_search_time_ms=statistics.fmean(record.search_time_ms for record in combat_records),
        elite_survival_rate=None if not elites else _rate(record.survived for record in elites),
        boss_survival_rate=None if not bosses else _rate(record.survived for record in bosses),
        worst_hp_loss=max(hp_losses),
        regression_combat_ids=_regression_ids(combat_records, baseline),
    )


def aggregate_combat_metrics(
    records: Iterable[CombatEvalMetrics | Mapping[str, Any] | Any],
) -> CombatEvalSummary:
    """Aggregate per-combat metrics into a batch summary."""

    normalized = [normalize_combat_metrics(record) for record in records]
    return CombatEvalSummary(
        combats=len(normalized),
        survival=MetricStats.from_values(record.survival for record in normalized),
        hp_loss=MetricStats.from_values(record.hp_loss for record in normalized),
        turns=MetricStats.from_values(record.turns for record in normalized),
        potions_used=MetricStats.from_values(record.potions_used for record in normalized),
        search_time_ms=MetricStats.from_values(record.search_time_ms for record in normalized),
    )


def normalize_combat_metrics(record: CombatEvalMetrics | Mapping[str, Any] | Any) -> CombatEvalMetrics:
    """Return a ``CombatEvalMetrics`` instance from a record-like input."""

    if isinstance(record, CombatEvalMetrics):
        return record
    if isinstance(record, Mapping):
        return CombatEvalMetrics.from_mapping(record)
    return CombatEvalMetrics.from_object(record)


def combine_combat_summaries(summaries: Iterable[CombatEvalSummary]) -> CombatEvalSummary:
    """Combine already aggregated summaries without needing the raw combats."""

    summary_list = list(summaries)
    combats = sum(summary.combats for summary in summary_list)
    return CombatEvalSummary(
        combats=combats,
        survival=_combine_stats(summary.survival for summary in summary_list),
        hp_loss=_combine_stats(summary.hp_loss for summary in summary_list),
        turns=_combine_stats(summary.turns for summary in summary_list),
        potions_used=_combine_stats(summary.potions_used for summary in summary_list),
        search_time_ms=_combine_stats(summary.search_time_ms for summary in summary_list),
    )


def _combine_stats(stats: Iterable[MetricStats]) -> MetricStats:
    stats_list = list(stats)
    count = sum(item.count for item in stats_list)
    if count == 0:
        return MetricStats(count=0, total=0.0)

    non_empty = [item for item in stats_list if item.count > 0]
    return MetricStats(
        count=count,
        total=sum(item.total for item in stats_list),
        minimum=min(item.minimum for item in non_empty if item.minimum is not None),
        maximum=max(item.maximum for item in non_empty if item.maximum is not None),
    )


def _survived_from_data(data: Mapping[str, Any]) -> bool:
    for key in ("survived", "won"):
        if key in data:
            value = data[key]
            if not isinstance(value, bool):
                raise TypeError(f"{key} must be a bool")
            return value

    for key in ("result", "outcome"):
        if key not in data:
            continue
        value = data[key]
        normalized = str(getattr(value, "value", value)).lower()
        if normalized in {"player_victory", "victory", "win", "won", "success"}:
            return True
        if normalized in {"player_death", "defeat", "loss", "lost", "death", "dead"}:
            return False
        raise ValueError(f"{key} does not identify a terminal combat outcome: {value!r}")

    raise KeyError("metrics must include survived, won, result, or outcome")


def _int_from_data(data: Mapping[str, Any], *keys: str, default: int | None = None) -> int:
    value = _value_from_data(data, *keys, default=default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{keys[0]} must be an int")
    if value < 0:
        raise ValueError(f"{keys[0]} must be non-negative")
    return value


def _float_from_data(data: Mapping[str, Any], *keys: str, default: float | None = None) -> float:
    value = _value_from_data(data, *keys, default=default)
    return _require_non_negative_number(value, keys[0])


def _value_from_data(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    if default is not None:
        return default
    joined = ", ".join(keys)
    raise KeyError(f"metrics must include one of: {joined}")


def _object_mapping(metrics: Any) -> dict[str, Any]:
    return {
        name: getattr(metrics, name)
        for name in (
            "survived",
            "won",
            "result",
            "outcome",
            "hp_loss",
            "turns",
            "turns_taken",
            "potions_used",
            "potion_usage",
            "search_time_ms",
        )
        if hasattr(metrics, name)
    }


def _regression_ids(
    records: tuple[CombatEvaluationRecord, ...],
    baseline: CombatEvaluationBaseline | None,
) -> tuple[str, ...]:
    if baseline is None:
        return ()

    regressions: list[str] = []
    for record in records:
        baseline_survived = baseline.survived_by_combat_id.get(record.combat_id)
        if baseline_survived is True and not record.survived:
            regressions.append(record.combat_id)
            continue

        baseline_hp_loss = baseline.hp_loss_by_combat_id.get(record.combat_id)
        if baseline_hp_loss is not None and record.hp_loss > baseline_hp_loss:
            regressions.append(record.combat_id)

    return tuple(regressions)


def _rate(values: Iterable[bool]) -> float:
    flags = tuple(values)
    if not flags:
        return 0.0
    return sum(flags) / len(flags)


def _require_non_negative(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_non_negative_number(value: Any, name: str) -> float:
    number = _require_finite_number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number
