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


def plot_metrics(metrics_path: str, output_path: str, watch_seconds: float | None = None) -> None:
    while True:
        _plot_once(metrics_path, output_path)
        if watch_seconds is None:
            return
        time.sleep(watch_seconds)


def _plot_once(metrics_path: str, output_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes: list[int] = []
    avg_scores: list[float] = []
    with Path(metrics_path).open(newline="") as file:
        for row in csv.DictReader(file):
            if not row.get("episode") or not row.get("eval_avg_score"):
                continue
            episodes.append(int(row["episode"]))
            avg_scores.append(float(row["eval_avg_score"]))

    if not episodes:
        raise SystemExit(f"no plottable metric rows found in {metrics_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, avg_scores, marker="o", linewidth=1.5, markersize=3)
    plt.title("2048 Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Average Evaluation Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

