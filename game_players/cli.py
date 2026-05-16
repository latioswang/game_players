"""Command line entry points for training and evaluating the 2048 agent."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import glog as log

from .game2048 import render
from .device import best_torch_device
from .ntuple_agent import NTupleAgent, summarize_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate a 2048 RL agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train an n-tuple TD agent")
    train.add_argument("--episodes", type=int, default=1000)
    train.add_argument("--model", default="models/2048-agent.pkl")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--alpha", type=float, default=0.01)
    train.add_argument("--epsilon", type=float, default=0.05)
    train.add_argument("--eval-every", type=int, default=100)
    train.add_argument("--eval-games", type=int, default=20)
    train.add_argument("--fresh", action="store_true", help="start a new model even if --model already exists")
    train.add_argument("--agent", choices=["auto", "ntuple", "dqn"], default="auto")
    train.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    evaluate = subparsers.add_parser("eval", help="evaluate a saved agent")
    evaluate.add_argument("--model", default="models/2048-agent.pkl")
    evaluate.add_argument("--games", type=int, default=100)
    evaluate.add_argument("--seed", type=int, default=1)
    evaluate.add_argument("--show-board", action="store_true")
    evaluate.add_argument("--agent", choices=["auto", "ntuple", "dqn"], default="auto")
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    args = parser.parse_args()
    if args.command == "train":
        train_agent(args)
    elif args.command == "eval":
        evaluate_agent(args)


def train_agent(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    model_path = Path(args.model)
    checkpoint_kind = _checkpoint_kind(args.model) if model_path.exists() and not args.fresh else None
    agent_kind, device = _select_agent(args)
    if checkpoint_kind is not None:
        agent_kind = checkpoint_kind
        if args.agent != "auto" and args.agent != checkpoint_kind:
            raise SystemExit(f"{args.model} is a {checkpoint_kind} checkpoint, but --agent {args.agent} was requested")

    if model_path.exists() and not args.fresh and agent_kind == "dqn":
        from .dqn_agent import DQNAgent

        agent = DQNAgent.load(args.model, device=device)
        log.info("resuming DQN %s from episode %s on %s", args.model, agent.episodes_trained, device)
    elif model_path.exists() and not args.fresh:
        agent = NTupleAgent.load(args.model)
        agent.alpha = args.alpha
        agent.epsilon = args.epsilon
        log.info("resuming n-tuple %s from episode %s on cpu", args.model, agent.episodes_trained)
    else:
        if agent_kind == "dqn":
            from .dqn_agent import DQNAgent

            agent = DQNAgent(device=device)
            log.info("starting a new DQN model on %s", device)
        else:
            agent = NTupleAgent(alpha=args.alpha, epsilon=args.epsilon)
            log.info("starting a new n-tuple model on cpu")
        if args.fresh:
            log.info("existing checkpoint will be overwritten at save time")
        else:
            log.info("no checkpoint found at %s", args.model)

    for episode in range(1, args.episodes + 1):
        result = agent.train_episode(rng)
        agent.episodes_trained += 1
        if episode == 1 or episode % args.eval_every == 0:
            eval_results = [agent.play_episode(rng) for _ in range(args.eval_games)]
            summary = summarize_results(eval_results)
            log.info(
                "episode=%s train_score=%s train_tile=%s eval_avg_score=%.1f eval_best_tile=%.0f",
                agent.episodes_trained,
                result.score,
                result.max_tile,
                summary["avg_score"],
                summary["best_max_tile"],
            )

    _ensure_parent(args.model)
    agent.save(args.model)
    log.info("saved model to %s", args.model)


def evaluate_agent(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    if _should_load_dqn(args.model, args.agent):
        from .dqn_agent import DQNAgent

        _, device = _select_agent(args)
        agent = DQNAgent.load(args.model, device=device)
    else:
        agent = NTupleAgent.load(args.model)
    results = [agent.play_episode(rng) for _ in range(args.games)]
    summary = summarize_results(results)
    log.info("games=%.0f", summary["games"])
    log.info("avg_score=%.1f", summary["avg_score"])
    log.info("best_score=%.0f", summary["best_score"])
    log.info("avg_max_tile=%.1f", summary["avg_max_tile"])
    log.info("best_max_tile=%.0f", summary["best_max_tile"])
    if args.show_board:
        best = max(results, key=lambda result: result.score)
        log.info("best_board:\n%s", render(best.board))


def _ensure_parent(path: str) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _select_agent(args: argparse.Namespace) -> tuple[str, str]:
    requested_device, reason = best_torch_device() if args.device == "auto" else (args.device, "requested explicitly")
    if args.agent == "ntuple":
        log.info("using n-tuple agent on cpu")
        return "ntuple", "cpu"
    if args.agent == "dqn":
        _require_torch()
        log.info("using DQN agent on %s: %s", requested_device, reason)
        return "dqn", requested_device
    if requested_device != "cpu":
        log.info("using DQN agent on %s: %s", requested_device, reason)
        return "dqn", requested_device
    log.info("using n-tuple agent on cpu: %s", reason)
    return "ntuple", "cpu"


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit("DQN training requires PyTorch. Install torch, then rerun the command.") from exc


def _should_load_dqn(path: str, requested_agent: str) -> bool:
    if requested_agent == "dqn":
        return True
    if requested_agent == "ntuple":
        return False
    return _checkpoint_kind(path) == "dqn"


def _checkpoint_kind(path: str) -> str:
    try:
        with open(path, "rb") as file:
            header = file.read(2)
    except FileNotFoundError:
        return "ntuple"
    return "dqn" if header == b"PK" else "ntuple"


if __name__ == "__main__":
    main()
