"""Command line entry points for the 2048 Expectimax player."""

from __future__ import annotations

import argparse
import random

import glog as log

from .expectimax_agent import (
    MAX_DEPTH,
    ExpectimaxAgent,
    action_name,
    auto_worker_count,
    evaluate_games,
    validate_depth,
)
from .game2048 import render


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a heuristic Expectimax 2048 player.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("eval", help="evaluate Expectimax over many games")
    evaluate.add_argument("--games", type=int, default=100)
    evaluate.add_argument("--seed", type=int, default=1)
    evaluate.add_argument("--depth", type=int, default=2, help=f"Expectimax depth, 1-{MAX_DEPTH}")
    evaluate.add_argument("--workers", default="auto", help="parallel eval workers: positive integer or 'auto'")

    play = subparsers.add_parser("play", help="run one Expectimax game")
    play.add_argument("--seed", type=int, default=1)
    play.add_argument("--depth", type=int, default=2, help=f"Expectimax depth, 1-{MAX_DEPTH}")
    play.add_argument("--show-board", action="store_true")

    args = parser.parse_args()
    if args.command == "eval":
        evaluate_agent(args)
    elif args.command == "play":
        play_game(args)


def evaluate_agent(args: argparse.Namespace) -> None:
    _validate_depth_or_exit(args.depth)
    if args.games < 1:
        raise SystemExit("--games must be at least 1")
    workers = _resolve_workers(args.workers)
    agent = ExpectimaxAgent(depth=args.depth)
    agent.warm_up()
    summary = evaluate_games(agent, games=args.games, seed=args.seed, workers=workers)
    log.info("games=%s", summary.games)
    log.info("depth=%s", args.depth)
    log.info("workers=%s", workers)
    log.info("avg_score=%.1f", summary.avg_score)
    log.info("best_score=%s", summary.best_score)
    log.info("avg_max_tile=%.1f", summary.avg_max_tile)
    log.info("best_max_tile=%s", summary.best_max_tile)
    log.info("tile_counts=%s", _format_tile_counts(summary.tile_counts))
    log.info("wins_2048=%s", summary.wins_2048)
    log.info("win_rate_2048=%.3f", summary.win_rate_2048)
    log.info("avg_moves=%.1f", summary.avg_moves)
    log.info("avg_seconds_per_game=%.3f", summary.avg_seconds_per_game)
    log.info("total_seconds=%.3f", summary.total_seconds)


def play_game(args: argparse.Namespace) -> None:
    _validate_depth_or_exit(args.depth)
    agent = ExpectimaxAgent(depth=args.depth)
    agent.warm_up()
    rng = random.Random(args.seed)
    result = agent.play_episode(rng)
    log.info("depth=%s", args.depth)
    log.info("score=%s", result.score)
    log.info("max_tile=%s", result.max_tile)
    log.info("moves=%s", result.moves)
    log.info("next_action=%s", action_name(agent.choose_action(_pack_for_next_action(result.board))))
    if args.show_board:
        log.info("final_board:\n%s", render(result.board))


def _validate_depth_or_exit(depth: int) -> None:
    try:
        validate_depth(depth)
    except ValueError as exc:
        raise SystemExit(f"--depth {exc}") from exc


def _resolve_workers(value: str) -> int:
    if value == "auto":
        return auto_worker_count()
    try:
        workers = int(value)
    except ValueError as exc:
        raise SystemExit("--workers must be a positive integer or 'auto'") from exc
    if workers < 1:
        raise SystemExit("--workers must be at least 1")
    return workers


def _format_tile_counts(tile_counts: dict[int, int]) -> str:
    return ",".join(f"{tile}:{count}" for tile, count in sorted(tile_counts.items()))


def _pack_for_next_action(board):
    from .expectimax_agent import pack_board

    return pack_board(board)


if __name__ == "__main__":
    main()
