"""Deterministic combat-policy experiments over local fixture records.

The helpers in this module intentionally avoid the external simulator.  A
fixture combat record is a small decision graph: each step contains a JSON-safe
state, legal actions, and deterministic outcomes keyed by stable action ids.
Policies are replayed against those records with fixed seed metadata, producing
comparable metrics, regression reports, and optional combat trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    from .combat_eval import CombatEvalMetrics, CombatEvalSummary, aggregate_combat_metrics
    from .combat_trajectory import (
        CombatActionLog,
        CombatStateLog,
        CombatStepLog,
        CombatTrajectory,
        JsonDict,
        JsonValue,
        validate_json_value,
        write_jsonl,
    )
except ImportError:
    from combat_eval import CombatEvalMetrics, CombatEvalSummary, aggregate_combat_metrics
    from combat_trajectory import (
        CombatActionLog,
        CombatStateLog,
        CombatStepLog,
        CombatTrajectory,
        JsonDict,
        JsonValue,
        validate_json_value,
        write_jsonl,
    )


PolicyFn = Callable[[JsonDict, Sequence[JsonDict]], Any]
RunKey = tuple[int, str]


class FixtureCombatPolicy(Protocol):
    """Policy interface accepted by fixture combat experiments."""

    def choose_action(self, state: JsonDict, legal_actions: Sequence[JsonDict]) -> Any:
        """Choose one legal action, a legal action key, or None when no action exists."""


@dataclass(frozen=True)
class FixtureActionOutcome:
    """Deterministic result of choosing one action in a fixture step."""

    next_step_index: int | None = None
    metrics: CombatEvalMetrics | Mapping[str, Any] | Any | None = None
    outcome: Mapping[str, Any] = field(default_factory=dict)
    reward: float | None = None
    done: bool | None = None

    def __post_init__(self) -> None:
        if self.next_step_index is not None and self.next_step_index < 0:
            raise ValueError("next_step_index must be non-negative")
        if self.reward is not None:
            object.__setattr__(self, "reward", _require_finite_float(self.reward, "reward"))
        if self.done is not None and not isinstance(self.done, bool):
            raise TypeError("done must be a bool")
        object.__setattr__(self, "outcome", _json_dict(self.outcome, "outcome"))
        if self.metrics is not None and not isinstance(self.metrics, CombatEvalMetrics):
            object.__setattr__(self, "metrics", CombatEvalMetrics.from_object(self.metrics))

    @property
    def is_terminal(self) -> bool:
        if self.done is not None:
            return self.done
        return self.next_step_index is None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FixtureActionOutcome":
        """Build an outcome from a plain fixture mapping."""

        next_step_index = data.get("next_step_index", data.get("next_step"))
        metrics = data.get("metrics", data.get("terminal_metrics"))
        return cls(
            next_step_index=next_step_index,
            metrics=metrics,
            outcome=data.get("outcome", {}),
            reward=data.get("reward"),
            done=data.get("done"),
        )

    def outcome_json(self) -> JsonDict:
        """Return terminal outcome metadata, including metrics when present."""

        data = dict(self.outcome)
        if self.metrics is not None:
            data.setdefault("metrics", self.metrics.to_json())
        return data


@dataclass(frozen=True)
class FixtureCombatStepRecord:
    """One policy decision point in a local fixture combat."""

    step_index: int
    state: Mapping[str, Any]
    legal_actions: Sequence[Mapping[str, Any]]
    outcomes: Mapping[str, FixtureActionOutcome | Mapping[str, Any]]

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")

        state = _json_dict(self.state, "state")
        legal_actions = tuple(_json_dict(action, f"legal_actions[{index}]") for index, action in enumerate(self.legal_actions))
        action_keys = tuple(fixture_action_key(action) for action in legal_actions)
        if len(set(action_keys)) != len(action_keys):
            raise ValueError("legal action keys must be unique within a fixture step")

        outcomes = {
            str(action_key): _coerce_outcome(outcome)
            for action_key, outcome in self.outcomes.items()
        }
        missing_outcomes = sorted(set(action_keys) - set(outcomes))
        if missing_outcomes:
            raise ValueError(f"missing fixture outcomes for legal actions: {missing_outcomes}")

        object.__setattr__(self, "state", state)
        object.__setattr__(self, "legal_actions", legal_actions)
        object.__setattr__(self, "outcomes", outcomes)

    def legal_action_by_key(self) -> dict[str, JsonDict]:
        return {fixture_action_key(action): action for action in self.legal_actions}


@dataclass(frozen=True)
class FixtureCombatRecord:
    """A fixed-seed combat fixture that can be replayed without a simulator."""

    seed: int
    combat_id: str
    steps: Sequence[FixtureCombatStepRecord]
    initial_step_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(self.combat_id, str) or not self.combat_id:
            raise ValueError("combat_id must be a non-empty string")
        if self.initial_step_index < 0:
            raise ValueError("initial_step_index must be non-negative")

        steps = tuple(self.steps)
        by_index = {step.step_index: step for step in steps}
        if len(by_index) != len(steps):
            raise ValueError("fixture step indexes must be unique")
        if self.initial_step_index not in by_index:
            raise ValueError("initial_step_index must identify a fixture step")

        for step in steps:
            for outcome in step.outcomes.values():
                if outcome.next_step_index is not None and outcome.next_step_index not in by_index:
                    raise ValueError("outcome next_step_index must identify a fixture step")
                if not outcome.is_terminal and outcome.next_step_index is None:
                    raise ValueError("non-terminal fixture outcomes require next_step_index")
                if outcome.is_terminal and outcome.metrics is None:
                    raise ValueError("terminal fixture outcomes require metrics")

        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "metadata", _json_dict(self.metadata, "metadata"))

    def step_by_index(self) -> dict[int, FixtureCombatStepRecord]:
        return {step.step_index: step for step in self.steps}


@dataclass(frozen=True)
class PolicyRunResult:
    """Result of replaying one policy on one fixed-seed fixture combat."""

    policy_id: str
    seed: int
    combat_id: str
    metrics: CombatEvalMetrics
    selected_action_keys: tuple[str, ...]
    trajectory: CombatTrajectory

    @property
    def run_key(self) -> RunKey:
        return (self.seed, self.combat_id)


@dataclass(frozen=True)
class PolicyComparisonResult:
    """Aggregate result for one policy over a fixture seed suite."""

    policy_id: str
    runs: tuple[PolicyRunResult, ...]
    summary: CombatEvalSummary


@dataclass(frozen=True)
class RegressionConfig:
    """Thresholds for paired candidate-vs-baseline regression tracking."""

    baseline_policy_id: str
    hp_loss_tolerance: int = 0
    turn_tolerance: int | None = None
    potion_tolerance: int | None = None
    require_survival_preserved: bool = True

    def __post_init__(self) -> None:
        if not self.baseline_policy_id:
            raise ValueError("baseline_policy_id must be non-empty")
        if self.hp_loss_tolerance < 0:
            raise ValueError("hp_loss_tolerance must be non-negative")
        if self.turn_tolerance is not None and self.turn_tolerance < 0:
            raise ValueError("turn_tolerance must be non-negative")
        if self.potion_tolerance is not None and self.potion_tolerance < 0:
            raise ValueError("potion_tolerance must be non-negative")


@dataclass(frozen=True)
class CombatPolicyRegression:
    """One paired metric regression against a fixed-seed baseline run."""

    policy_id: str
    seed: int
    combat_id: str
    metric: str
    baseline: JsonValue
    candidate: JsonValue


@dataclass(frozen=True)
class FixedSeedComparison:
    """Deterministic comparison of policies over the same fixture records."""

    policies: dict[str, PolicyComparisonResult]
    regressions: tuple[CombatPolicyRegression, ...] = ()
    baseline_policy_id: str | None = None

    def regression_combat_ids(self, policy_id: str | None = None) -> tuple[str, ...]:
        """Return unique combat ids with regressions, preserving report order."""

        ids: list[str] = []
        for regression in self.regressions:
            if policy_id is not None and regression.policy_id != policy_id:
                continue
            if regression.combat_id not in ids:
                ids.append(regression.combat_id)
        return tuple(ids)

    def to_json(self) -> JsonDict:
        return {
            "baseline_policy_id": self.baseline_policy_id,
            "policies": {
                policy_id: result.summary.to_json()
                for policy_id, result in self.policies.items()
            },
            "regressions": [
                {
                    "policy_id": regression.policy_id,
                    "seed": regression.seed,
                    "combat_id": regression.combat_id,
                    "metric": regression.metric,
                    "baseline": regression.baseline,
                    "candidate": regression.candidate,
                }
                for regression in self.regressions
            ],
        }


def run_fixture_combat_policy(
    record: FixtureCombatRecord,
    policy: FixtureCombatPolicy | PolicyFn,
    *,
    policy_id: str,
    max_steps: int = 100,
) -> PolicyRunResult:
    """Replay one policy on one fixture combat record."""

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if not policy_id:
        raise ValueError("policy_id must be non-empty")

    steps_by_index = record.step_by_index()
    current_step_index = record.initial_step_index
    selected_action_keys: list[str] = []
    trajectory_steps: list[CombatStepLog] = []

    for decision_index in range(max_steps):
        step = steps_by_index[current_step_index]
        legal_actions = tuple(step.legal_actions)
        if not legal_actions:
            raise ValueError(f"fixture step {step.step_index} has no legal actions and no terminal outcome")

        legal_by_key = step.legal_action_by_key()
        selected = _choose_action(policy, step.state, legal_actions)
        selected_key = fixture_action_key(selected)
        if selected_key not in legal_by_key:
            raise ValueError(
                f"policy {policy_id!r} selected illegal action {selected_key!r} "
                f"for combat {record.combat_id!r} seed {record.seed}"
            )

        outcome = step.outcomes[selected_key]
        selected_action_keys.append(selected_key)
        trajectory_steps.append(
            _trajectory_step(
                decision_index=decision_index,
                policy_id=policy_id,
                step=step,
                selected_action=legal_by_key[selected_key],
                outcome=outcome,
            )
        )

        if outcome.is_terminal:
            if outcome.metrics is None:
                raise ValueError("terminal fixture outcomes require metrics")
            trajectory = _trajectory(record, policy_id, trajectory_steps, outcome)
            return PolicyRunResult(
                policy_id=policy_id,
                seed=record.seed,
                combat_id=record.combat_id,
                metrics=outcome.metrics,
                selected_action_keys=tuple(selected_action_keys),
                trajectory=trajectory,
            )

        if outcome.next_step_index is None:
            raise ValueError("non-terminal fixture outcome requires next_step_index")
        current_step_index = outcome.next_step_index

    raise RuntimeError(
        f"policy {policy_id!r} exceeded max_steps={max_steps} on "
        f"combat {record.combat_id!r} seed {record.seed}"
    )


def compare_fixed_seed_policies(
    records: Sequence[FixtureCombatRecord],
    policies: Mapping[str, FixtureCombatPolicy | PolicyFn],
    *,
    baseline_policy_id: str | None = None,
    regression_config: RegressionConfig | None = None,
    trajectory_path: str | Path | None = None,
    max_steps: int = 100,
) -> FixedSeedComparison:
    """Run policies over the same fixed-seed fixtures and track regressions."""

    if not policies:
        raise ValueError("at least one policy is required")
    if baseline_policy_id is not None and baseline_policy_id not in policies:
        raise ValueError("baseline_policy_id must identify one of the policies")
    if regression_config is not None and regression_config.baseline_policy_id not in policies:
        raise ValueError("regression_config.baseline_policy_id must identify one of the policies")

    ordered_records = tuple(sorted(records, key=lambda record: (record.seed, record.combat_id)))
    results: dict[str, PolicyComparisonResult] = {}
    trajectories: list[CombatTrajectory] = []

    for policy_id in sorted(policies):
        runs = tuple(
            run_fixture_combat_policy(
                record,
                policies[policy_id],
                policy_id=policy_id,
                max_steps=max_steps,
            )
            for record in ordered_records
        )
        trajectories.extend(run.trajectory for run in runs)
        results[policy_id] = PolicyComparisonResult(
            policy_id=policy_id,
            runs=runs,
            summary=aggregate_combat_metrics(run.metrics for run in runs),
        )

    effective_baseline = baseline_policy_id
    if regression_config is not None:
        effective_baseline = regression_config.baseline_policy_id
    regressions: tuple[CombatPolicyRegression, ...] = ()
    if effective_baseline is not None:
        config = regression_config or RegressionConfig(baseline_policy_id=effective_baseline)
        regressions = tuple(track_policy_regressions(results, config))

    if trajectory_path is not None:
        write_jsonl(trajectory_path, trajectories)

    return FixedSeedComparison(
        policies=results,
        regressions=regressions,
        baseline_policy_id=effective_baseline,
    )


def track_policy_regressions(
    results: Mapping[str, PolicyComparisonResult],
    config: RegressionConfig,
) -> tuple[CombatPolicyRegression, ...]:
    """Detect paired per-combat regressions against a baseline policy."""

    if config.baseline_policy_id not in results:
        raise ValueError("baseline policy is missing from comparison results")

    baseline_runs = {
        run.run_key: run
        for run in results[config.baseline_policy_id].runs
    }
    regressions: list[CombatPolicyRegression] = []
    for policy_id in sorted(results):
        if policy_id == config.baseline_policy_id:
            continue
        for run in results[policy_id].runs:
            baseline = baseline_runs.get(run.run_key)
            if baseline is None:
                raise ValueError(f"candidate run has no paired baseline: {run.run_key}")
            regressions.extend(_regressions_for_run(run, baseline, config))
    return tuple(regressions)


def fixture_action_key(action: Any) -> str:
    """Return the stable key used by fixture legal actions and policy outputs."""

    if action is None:
        return ""
    if isinstance(action, str):
        return action
    if isinstance(action, Mapping):
        for key in ("action_key", "stable_id", "id", "key", "name"):
            value = action.get(key)
            if value is not None:
                return str(value)
        raise ValueError("fixture action mappings require action_key, stable_id, id, key, or name")

    explicit_key = getattr(action, "action_key", None)
    if callable(explicit_key):
        return str(explicit_key())
    if explicit_key is not None:
        return str(explicit_key)

    for attribute in ("stable_id", "id", "key", "name"):
        value = getattr(action, attribute, None)
        if value is not None:
            return str(value)

    return repr(action)


def _choose_action(policy: FixtureCombatPolicy | PolicyFn, state: JsonDict, legal_actions: Sequence[JsonDict]) -> Any:
    chooser = getattr(policy, "choose_action", None)
    if callable(chooser):
        return chooser(state, legal_actions)
    if callable(policy):
        return policy(state, legal_actions)
    raise TypeError("policy must be callable or provide choose_action")


def _trajectory_step(
    *,
    decision_index: int,
    policy_id: str,
    step: FixtureCombatStepRecord,
    selected_action: JsonDict,
    outcome: FixtureActionOutcome,
) -> CombatStepLog:
    return CombatStepLog(
        state=CombatStateLog(
            step_index=decision_index,
            turn=_optional_int(step.state.get("turn")),
            phase=_optional_str(step.state.get("phase")),
            state=step.state,
        ),
        action=CombatActionLog(
            step_index=decision_index,
            policy=policy_id,
            legal_actions=list(step.legal_actions),
            selected_action=selected_action,
        ),
        reward=outcome.reward,
        done=outcome.is_terminal,
    )


def _trajectory(
    record: FixtureCombatRecord,
    policy_id: str,
    steps: Sequence[CombatStepLog],
    outcome: FixtureActionOutcome,
) -> CombatTrajectory:
    return CombatTrajectory(
        trajectory_id=f"{policy_id}:{record.seed}:{record.combat_id}",
        seed=record.seed,
        combat_id=record.combat_id,
        steps=list(steps),
        outcome=outcome.outcome_json(),
        metadata={
            **dict(record.metadata),
            "policy_id": policy_id,
            "source": "fixture_combat_record",
        },
    )


def _regressions_for_run(
    candidate: PolicyRunResult,
    baseline: PolicyRunResult,
    config: RegressionConfig,
) -> list[CombatPolicyRegression]:
    regressions: list[CombatPolicyRegression] = []
    candidate_metrics = candidate.metrics
    baseline_metrics = baseline.metrics

    if (
        config.require_survival_preserved
        and baseline_metrics.survived
        and not candidate_metrics.survived
    ):
        regressions.append(
            _regression(candidate, "survived", baseline_metrics.survived, candidate_metrics.survived)
        )

    if candidate_metrics.hp_loss > baseline_metrics.hp_loss + config.hp_loss_tolerance:
        regressions.append(
            _regression(candidate, "hp_loss", baseline_metrics.hp_loss, candidate_metrics.hp_loss)
        )

    if (
        config.turn_tolerance is not None
        and candidate_metrics.turns > baseline_metrics.turns + config.turn_tolerance
    ):
        regressions.append(
            _regression(candidate, "turns", baseline_metrics.turns, candidate_metrics.turns)
        )

    if (
        config.potion_tolerance is not None
        and candidate_metrics.potions_used > baseline_metrics.potions_used + config.potion_tolerance
    ):
        regressions.append(
            _regression(candidate, "potions_used", baseline_metrics.potions_used, candidate_metrics.potions_used)
        )

    return regressions


def _regression(
    candidate: PolicyRunResult,
    metric: str,
    baseline: JsonValue,
    value: JsonValue,
) -> CombatPolicyRegression:
    return CombatPolicyRegression(
        policy_id=candidate.policy_id,
        seed=candidate.seed,
        combat_id=candidate.combat_id,
        metric=metric,
        baseline=baseline,
        candidate=value,
    )


def _coerce_outcome(outcome: FixtureActionOutcome | Mapping[str, Any]) -> FixtureActionOutcome:
    if isinstance(outcome, FixtureActionOutcome):
        return outcome
    if isinstance(outcome, Mapping):
        return FixtureActionOutcome.from_mapping(outcome)
    raise TypeError("fixture outcomes must be FixtureActionOutcome or mapping")


def _json_dict(data: Mapping[str, Any], path: str) -> JsonDict:
    normalized = validate_json_value(dict(data), path)
    if not isinstance(normalized, dict):
        raise TypeError(f"{path} must be a JSON object")
    return normalized


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return value


def _require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number
