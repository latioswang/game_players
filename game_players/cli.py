"""Command line entry points for training and evaluating the 2048 agent."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import glog as log

from .game2048 import render
from .device import best_torch_device
from .metrics import append_metrics, plot_metrics
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
    train.add_argument("--min-alpha", type=float, default=0.001)
    train.add_argument("--min-epsilon", type=float, default=0.005)
    train.add_argument("--alpha-decay", type=float, default=0.99995)
    train.add_argument("--epsilon-decay", type=float, default=0.9999)
    train.add_argument("--eval-every", type=int, default=500)
    train.add_argument("--eval-games", type=int, default=50)
    train.add_argument("--fresh", action="store_true", help="start a new model even if --model already exists")
    train.add_argument("--no-symmetry", action="store_true", help="disable n-tuple symmetry sharing")
    train.add_argument("--agent", choices=["auto", "ntuple", "dqn"], default="auto")
    train.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    train.add_argument("--metrics", default="models/training-metrics.csv")
    train.add_argument("--save-every", type=int, default=1000, help="save the model every N completed episodes")
    train.add_argument("--best-model", default=None, help="save a copy whenever eval_avg_score improves")

    evaluate = subparsers.add_parser("eval", help="evaluate a saved agent")
    evaluate.add_argument("--model", default="models/2048-agent.pkl")
    evaluate.add_argument("--games", type=int, default=100)
    evaluate.add_argument("--seed", type=int, default=1)
    evaluate.add_argument("--show-board", action="store_true")
    evaluate.add_argument("--agent", choices=["auto", "ntuple", "dqn"], default="auto")
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")

    plot = subparsers.add_parser("plot", help="plot training metrics")
    plot.add_argument("--metrics", default="models/training-metrics.csv")
    plot.add_argument("--output", default="models/training-progress.png")
    plot.add_argument("--watch", type=float, default=None, help="refresh the output image every N seconds")

    args = parser.parse_args()
    if args.command == "train":
        train_agent(args)
    elif args.command == "eval":
        evaluate_agent(args)
    elif args.command == "plot":
        plot_metrics(args.metrics, args.output, args.watch)
        log.info("wrote training plot to %s", args.output)


def train_agent(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    model_path = Path(args.model)
    checkpoint_kind = _checkpoint_kind(args.model) if model_path.exists() and not args.fresh else None
    if checkpoint_kind is not None:
        if args.agent != "auto" and args.agent != checkpoint_kind:
            raise SystemExit(f"{args.model} is a {checkpoint_kind} checkpoint, but --agent {args.agent} was requested")
        selection_args = argparse.Namespace(**vars(args))
        selection_args.agent = checkpoint_kind
        agent_kind, device = _select_agent(selection_args)
    else:
        agent_kind, device = _select_agent(args)

    if model_path.exists() and not args.fresh and agent_kind == "dqn":
        from .dqn_agent import DQNAgent

        agent = DQNAgent.load(args.model, device=device)
        log.info("resuming DQN %s from episode %s on %s", args.model, agent.episodes_trained, device)
    elif model_path.exists() and not args.fresh:
        agent = NTupleAgent.load(args.model)
        _configure_ntuple_agent(agent, args)
        log.info(
            "resuming n-tuple %s from episode %s on cpu with %s patterns symmetry=%s",
            args.model,
            agent.episodes_trained,
            len(agent.patterns),
            agent.use_symmetry,
        )
    else:
        if agent_kind == "dqn":
            from .dqn_agent import DQNAgent

            agent = DQNAgent(device=device)
            log.info("starting a new DQN model on %s", device)
        else:
            agent = NTupleAgent()
            _configure_ntuple_agent(agent, args)
            log.info("starting a new n-tuple model on cpu")
        if args.fresh:
            log.info("existing checkpoint will be overwritten at save time")
        else:
            log.info("no checkpoint found at %s", args.model)

    best_eval_score = float("-inf")
    best_model_path = args.best_model or _default_best_model_path(args.model)

    try:
        for episode in range(1, args.episodes + 1):
            result = agent.train_episode(rng)
            agent.episodes_trained += 1
            if episode == 1 or episode % args.eval_every == 0:
                eval_results = [agent.play_episode(rng) for _ in range(args.eval_games)]
                summary = summarize_results(eval_results)
                agent_metrics = _agent_metrics(agent)
                tile_counts = _format_tile_counts(summary["tile_counts"])
                log.info(
                    "episode=%s train_score=%s train_tile=%s eval_avg_score=%.1f eval_best_tile=%.0f %s eval_tiles=%s",
                    agent.episodes_trained,
                    result.score,
                    result.max_tile,
                    summary["avg_score"],
                    summary["best_max_tile"],
                    agent_metrics,
                    tile_counts,
                )
                append_metrics(
                    args.metrics,
                    {
                        "episode": agent.episodes_trained,
                        "train_score": result.score,
                        "train_tile": result.max_tile,
                        "eval_avg_score": f"{summary['avg_score']:.1f}",
                        "eval_best_tile": f"{summary['best_max_tile']:.0f}",
                        "eval_tile_counts": tile_counts,
                        "agent_metrics": agent_metrics,
                    },
                )
                if float(summary["avg_score"]) > best_eval_score:
                    best_eval_score = float(summary["avg_score"])
                    _save_agent(agent, best_model_path)
                    log.info("new best eval_avg_score=%.1f saved to %s", best_eval_score, best_model_path)
            if args.save_every > 0 and episode % args.save_every == 0:
                _save_agent(agent, args.model)
    except KeyboardInterrupt:
        _save_agent(agent, args.model)
        log.info("interrupted; saved model at episode %s", agent.episodes_trained)
        raise SystemExit(130) from None

    _save_agent(agent, args.model)


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
    log.info("tile_counts=%s", _format_tile_counts(summary["tile_counts"]))
    if args.show_board:
        best = max(results, key=lambda result: result.score)
        log.info("best_board:\n%s", render(best.board))


def _ensure_parent(path: str) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _save_agent(agent: object, path: str) -> None:
    _ensure_parent(path)
    agent.save(path)
    log.info("saved model to %s at episode %s", path, getattr(agent, "episodes_trained", "unknown"))


def _configure_ntuple_agent(agent: NTupleAgent, args: argparse.Namespace) -> None:
    agent.alpha = args.alpha
    agent.epsilon = args.epsilon
    agent.min_alpha = args.min_alpha
    agent.min_epsilon = args.min_epsilon
    agent.alpha_decay = args.alpha_decay
    agent.epsilon_decay = args.epsilon_decay
    agent.use_symmetry = not args.no_symmetry


def _default_best_model_path(model_path: str) -> str:
    path = Path(model_path)
    return str(path.with_name(f"{path.stem}.best{path.suffix}"))


def _agent_metrics(agent: object) -> str:
    if hasattr(agent, "replay"):
        loss = getattr(agent, "last_loss", None)
        loss_text = "none" if loss is None else f"{loss:.5f}"
        config = getattr(agent, "config")
        return (
            f"epsilon={config.epsilon:.4f} "
            f"loss={loss_text} "
            f"replay={len(getattr(agent, 'replay'))} "
            f"steps={getattr(agent, 'steps')}"
        )
    if hasattr(agent, "weights"):
        return (
            f"epsilon={getattr(agent, 'epsilon'):.4f} "
            f"alpha={getattr(agent, 'alpha'):.5f} "
            f"weights={len(getattr(agent, 'weights'))}"
        )
    return ""


def _format_tile_counts(tile_counts: dict[int, int]) -> str:
    return ",".join(f"{tile}:{count}" for tile, count in sorted(tile_counts.items()))


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
