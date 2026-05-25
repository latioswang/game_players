"""Run one fixed-budget 2048 n-tuple autoresearch experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


RESULT_FIELDS = (
    "timestamp",
    "commit",
    "tag",
    "eval_avg_score",
    "eval_best_tile",
    "eval_tile_counts",
    "episodes",
    "eval_games",
    "seconds",
    "status",
    "description",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one n-tuple autoresearch experiment.")
    parser.add_argument("--tag", required=True, help="short run label used for output paths")
    parser.add_argument("--description", required=True, help="short text description for results.tsv")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", default="results.tsv")
    parser.add_argument("--runs-dir", default="models/autoresearch")
    args = parser.parse_args()

    tag = _clean_tag(args.tag)
    run_dir = Path(args.runs_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "agent.pkl"
    best_model_path = run_dir / "agent.best.pkl"
    metrics_path = run_dir / "metrics.csv"
    log_path = run_dir / "run.log"

    command = [
        sys.executable,
        "-m",
        "game_players.cli",
        "train",
        "--agent",
        "ntuple",
        "--fresh",
        "--episodes",
        str(args.episodes),
        "--eval-every",
        str(args.eval_every),
        "--eval-games",
        str(args.eval_games),
        "--seed",
        str(args.seed),
        "--model",
        str(model_path),
        "--best-model",
        str(best_model_path),
        "--metrics",
        str(metrics_path),
        "--save-every",
        "0",
    ]

    start = time.monotonic()
    with log_path.open("w") as log_file:
        completed = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    seconds = time.monotonic() - start

    status = "keep" if completed.returncode == 0 else "crash"
    row = _last_metric_row(metrics_path) if completed.returncode == 0 else {}
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "tag": tag,
        "eval_avg_score": row.get("eval_avg_score", "0.0"),
        "eval_best_tile": row.get("eval_best_tile", "0"),
        "eval_tile_counts": row.get("eval_tile_counts", ""),
        "episodes": str(args.episodes),
        "eval_games": str(args.eval_games),
        "seconds": f"{seconds:.1f}",
        "status": status,
        "description": args.description,
    }
    _append_result(Path(args.results), result)

    print(f"status: {status}")
    print(f"eval_avg_score: {result['eval_avg_score']}")
    print(f"eval_best_tile: {result['eval_best_tile']}")
    print(f"eval_tile_counts: {result['eval_tile_counts']}")
    print(f"seconds: {result['seconds']}")
    print(f"log: {log_path}")
    print(f"results: {args.results}")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _clean_tag(tag: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "-" for char in tag.strip())
    if not cleaned:
        raise SystemExit("--tag must contain at least one safe path character")
    return cleaned


def _last_metric_row(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"training completed but metrics file was not written: {path}")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"training completed but metrics file has no rows: {path}")
    return rows[-1]


def _append_result(path: Path, row: dict[str, str]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


if __name__ == "__main__":
    main()
