"""Training metric persistence and plotting."""

from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Any


METRIC_FIELDS = (
    "episode",
    "train_score",
    "train_tile",
    "eval_avg_score",
    "eval_best_tile",
    "eval_tile_counts",
    "pct_512",
    "pct_1024",
    "pct_2048",
    "alpha",
    "epsilon",
    "weight_count",
    "weight_l2",
    "weight_abs_mean",
    "weight_delta_l2",
    "td_error_abs_avg",
    "td_updates",
    "agent_metrics",
)


def append_metrics(path: str, row: dict[str, Any]) -> None:
    metric_path = Path(path)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    exists = metric_path.exists() and metric_path.stat().st_size > 0
    with metric_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})


def plot_metrics(metrics_path: str, output_path: str, series: list[str], watch_seconds: float | None = None) -> None:
    while True:
        _plot_once(metrics_path, output_path, series)
        if watch_seconds is None:
            return
        time.sleep(watch_seconds)


def _plot_once(metrics_path: str, output_path: str, series: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segments: list[tuple[list[int], dict[str, list[float]]]] = []
    episodes: list[int] = []
    values: dict[str, list[float]] = {name: [] for name in series}
    previous_episode: int | None = None
    with Path(metrics_path).open(newline="") as file:
        for row in csv.DictReader(file):
            if not row.get("episode"):
                continue
            episode = int(row["episode"])
            if previous_episode is not None and episode < previous_episode and episodes:
                segments.append((episodes, values))
                episodes = []
                values = {name: [] for name in series}
            previous_episode = episode
            row_values: dict[str, float] = {}
            for name in series:
                value = row.get(name)
                if not value and name.startswith("pct_"):
                    value = _derive_tile_pct(row, name)
                if not value:
                    break
                row_values[name] = float(value)
            else:
                episodes.append(episode)
                for name, value in row_values.items():
                    values[name].append(value)
    if episodes:
        segments.append((episodes, values))

    if not segments:
        raise SystemExit(f"no plottable metric rows found in {metrics_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(series), 1, figsize=(10, max(4, 3 * len(series))), sharex=True)
    if len(series) == 1:
        axes = [axes]
    for axis, name in zip(axes, series):
        for segment_episodes, segment_values in segments:
            axis.plot(segment_episodes, segment_values[name], marker="o", linewidth=1.5, markersize=3)
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.3)
    axes[0].set_title("2048 Training Progress")
    axes[-1].set_xlabel("Episode")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _derive_tile_pct(row: dict[str, str], name: str) -> str:
    try:
        threshold = int(name.removeprefix("pct_"))
    except ValueError:
        return ""
    counts = _parse_tile_counts(row.get("eval_tile_counts", ""))
    total = sum(counts.values())
    if total == 0:
        return ""
    reached = sum(count for tile, count in counts.items() if tile >= threshold)
    return str(reached / total)


def _parse_tile_counts(text: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for item in text.split(","):
        if not item:
            continue
        tile_text, count_text = item.split(":", 1)
        counts[int(tile_text)] = int(count_text)
    return counts
