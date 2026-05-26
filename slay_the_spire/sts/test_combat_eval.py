from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from combat_eval import (
    CombatEvalMetrics,
    CombatEvalSummary,
    CombatEvaluationBaseline,
    CombatEvaluationRecord,
    MetricStats,
    aggregate_combat_metrics,
    combine_combat_summaries,
    normalize_combat_metrics,
    summarize_combat_records,
)


def test_summarizes_combat_records() -> None:
    summary = summarize_combat_records(
        (
            CombatEvaluationRecord(
                combat_id="seed-1-floor-1",
                survived=True,
                hp_before=80,
                hp_after=72,
                turns=3,
                potions_used=0,
                search_time_ms=2.0,
            ),
            CombatEvaluationRecord(
                combat_id="seed-1-elite-1",
                survived=False,
                hp_before=72,
                hp_after=0,
                turns=7,
                potions_used=1,
                search_time_ms=8.0,
                elite=True,
            ),
            CombatEvaluationRecord(
                combat_id="seed-1-boss",
                survived=True,
                hp_before=65,
                hp_after=51,
                turns=9,
                potions_used=1,
                search_time_ms=10.0,
                boss=True,
            ),
        )
    )

    assert summary.combat_count == 3
    assert summary.survival_rate == pytest.approx(2 / 3)
    assert summary.average_hp_loss == pytest.approx((8 + 72 + 14) / 3)
    assert summary.median_hp_loss == 14
    assert summary.average_turns == pytest.approx(19 / 3)
    assert summary.average_potions_used == pytest.approx(2 / 3)
    assert summary.average_search_time_ms == pytest.approx(20 / 3)
    assert summary.elite_survival_rate == 0
    assert summary.boss_survival_rate == 1
    assert summary.worst_hp_loss == 72


def test_empty_summary_is_defined() -> None:
    summary = summarize_combat_records(())

    assert summary.combat_count == 0
    assert summary.survival_rate == 0.0
    assert summary.elite_survival_rate is None
    assert summary.boss_survival_rate is None


def test_detects_regression_against_baseline() -> None:
    summary = summarize_combat_records(
        (
            CombatEvaluationRecord(
                combat_id="same-survival-more-loss",
                survived=True,
                hp_before=80,
                hp_after=60,
                turns=4,
            ),
            CombatEvaluationRecord(
                combat_id="lost-after-baseline-survived",
                survived=False,
                hp_before=80,
                hp_after=0,
                turns=5,
            ),
            CombatEvaluationRecord(
                combat_id="improved",
                survived=True,
                hp_before=80,
                hp_after=76,
                turns=2,
            ),
        ),
        baseline=CombatEvaluationBaseline(
            hp_loss_by_combat_id={
                "same-survival-more-loss": 10,
                "improved": 8,
            },
            survived_by_combat_id={"lost-after-baseline-survived": True},
        ),
    )

    assert summary.regression_combat_ids == (
        "same-survival-more-loss",
        "lost-after-baseline-survived",
    )


def test_rejects_negative_metrics() -> None:
    with pytest.raises(ValueError, match="hp_before"):
        CombatEvaluationRecord(
            combat_id="bad",
            survived=True,
            hp_before=-1,
            hp_after=0,
            turns=1,
        )

    with pytest.raises(ValueError, match="search_time_ms"):
        CombatEvaluationRecord(
            combat_id="bad",
            survived=True,
            hp_before=1,
            hp_after=1,
            turns=1,
            search_time_ms=-0.1,
        )


def test_combat_eval_metrics_are_frozen_and_json_safe() -> None:
    metrics = CombatEvalMetrics(
        survived=True,
        hp_loss=6,
        turns=4,
        potions_used=1,
        search_time_ms=12.5,
    )

    with pytest.raises(FrozenInstanceError):
        metrics.hp_loss = 1

    assert metrics.survival == 1
    assert metrics.to_json() == {
        "survived": True,
        "hp_loss": 6,
        "turns": 4,
        "potions_used": 1,
        "search_time_ms": 12.5,
    }


def test_aggregate_combat_metrics_tracks_survival_and_averages() -> None:
    summary = aggregate_combat_metrics(
        (
            CombatEvalMetrics(True, hp_loss=2, turns=3, potions_used=0, search_time_ms=4.0),
            CombatEvalMetrics(False, hp_loss=18, turns=5, potions_used=1, search_time_ms=8.0),
            CombatEvalMetrics(True, hp_loss=7, turns=4, potions_used=2, search_time_ms=18.0),
        )
    )

    assert summary.combats == 3
    assert summary.survival_rate == pytest.approx(2 / 3)
    assert summary.average_hp_loss == pytest.approx(9.0)
    assert summary.average_turns == pytest.approx(4.0)
    assert summary.average_potions_used == pytest.approx(1.0)
    assert summary.average_search_time_ms == pytest.approx(10.0)
    assert summary.hp_loss.minimum == 2
    assert summary.hp_loss.maximum == 18
    assert summary.to_json()["survival"]["total"] == 2.0


def test_aggregate_empty_batch_returns_zeroed_summary() -> None:
    summary = aggregate_combat_metrics(())

    assert summary == CombatEvalSummary(
        combats=0,
        survival=MetricStats(count=0, total=0.0),
        hp_loss=MetricStats(count=0, total=0.0),
        turns=MetricStats(count=0, total=0.0),
        potions_used=MetricStats(count=0, total=0.0),
        search_time_ms=MetricStats(count=0, total=0.0),
    )
    assert summary.survival_rate == 0.0


def test_normalizes_mapping_aliases_from_logs_and_api_metrics() -> None:
    assert normalize_combat_metrics(
        {
            "won": True,
            "hp_loss": 3,
            "turns": 2,
            "potion_usage": 1,
            "search_time_ms": 5,
        }
    ) == CombatEvalMetrics(True, hp_loss=3, turns=2, potions_used=1, search_time_ms=5.0)

    assert normalize_combat_metrics(
        {
            "result": "player_death",
            "hp_loss": 40,
            "turns_taken": 6,
        }
    ) == CombatEvalMetrics(False, hp_loss=40, turns=6)


def test_normalizes_object_shaped_metrics_without_combat_api_dependency() -> None:
    metrics = SimpleNamespace(
        outcome=SimpleNamespace(value="player_victory"),
        hp_loss=8,
        turns_taken=4,
        potions_used=1,
        search_time_ms=21.25,
    )

    assert CombatEvalMetrics.from_object(metrics) == CombatEvalMetrics(
        survived=True,
        hp_loss=8,
        turns=4,
        potions_used=1,
        search_time_ms=21.25,
    )


def test_combine_combat_summaries_matches_raw_aggregation() -> None:
    first = aggregate_combat_metrics(
        (
            CombatEvalMetrics(True, hp_loss=2, turns=3, search_time_ms=4.0),
            CombatEvalMetrics(False, hp_loss=10, turns=5, search_time_ms=8.0),
        )
    )
    second = aggregate_combat_metrics(
        (CombatEvalMetrics(True, hp_loss=6, turns=2, potions_used=1, search_time_ms=12.0),)
    )

    combined = combine_combat_summaries((first, second))
    direct = aggregate_combat_metrics(
        (
            CombatEvalMetrics(True, hp_loss=2, turns=3, search_time_ms=4.0),
            CombatEvalMetrics(False, hp_loss=10, turns=5, search_time_ms=8.0),
            CombatEvalMetrics(True, hp_loss=6, turns=2, potions_used=1, search_time_ms=12.0),
        )
    )

    assert combined == direct


def test_validation_rejects_invalid_normalized_metrics() -> None:
    with pytest.raises(TypeError, match="survived"):
        CombatEvalMetrics(survived=1, hp_loss=0, turns=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="hp_loss"):
        CombatEvalMetrics(survived=True, hp_loss=-1, turns=0)

    with pytest.raises(ValueError, match="search_time_ms"):
        CombatEvalMetrics(survived=True, hp_loss=0, turns=0, search_time_ms=float("inf"))

    with pytest.raises(ValueError, match="outcome"):
        normalize_combat_metrics({"outcome": "ongoing", "hp_loss": 0, "turns": 1})
