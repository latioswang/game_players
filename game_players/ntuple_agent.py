"""N-tuple temporal-difference learner for 2048."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import pickle
import random
from typing import DefaultDict, Iterable

from .game2048 import Action, Board, GameResult, legal_actions, max_tile, move, new_game, step


Pattern = tuple[int, ...]


BASE_PATTERNS: tuple[Pattern, ...] = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 15),
    (0, 4, 8, 12),
    (1, 5, 9, 13),
    (2, 6, 10, 14),
    (3, 7, 11, 15),
    (0, 1, 4, 5),
    (1, 2, 5, 6),
    (2, 3, 6, 7),
    (4, 5, 8, 9),
    (5, 6, 9, 10),
    (6, 7, 10, 11),
    (8, 9, 12, 13),
    (9, 10, 13, 14),
    (10, 11, 14, 15),
)


@dataclass
class NTupleAgent:
    """A compact value-function agent using board pattern lookup tables."""

    alpha: float = 0.01
    gamma: float = 1.0
    epsilon: float = 0.05
    episodes_trained: int = 0
    patterns: tuple[Pattern, ...] = BASE_PATTERNS
    weights: DefaultDict[tuple[int, Pattern], float] = field(default_factory=lambda: defaultdict(float))

    def value(self, board: Board) -> float:
        return sum(self.weights[(index, _feature(board, pattern))] for index, pattern in enumerate(self.patterns))

    def update_value(self, board: Board, target: float) -> float:
        prediction = self.value(board)
        error = target - prediction
        scaled = self.alpha * error / len(self.patterns)
        for index, pattern in enumerate(self.patterns):
            self.weights[(index, _feature(board, pattern))] += scaled
        return error

    def action_values(self, board: Board) -> list[tuple[Action, float, Board, int]]:
        values = []
        for action in legal_actions(board):
            after, reward = move(board, action)
            values.append((action, reward + self.gamma * self.value(after), after, reward))
        return values

    def choose_action(self, board: Board, rng: random.Random | None = None, explore: bool = True) -> tuple[Action, Board, int]:
        rng = rng or random
        values = self.action_values(board)
        if not values:
            raise ValueError("cannot choose an action from a terminal board")
        if explore and rng.random() < self.epsilon:
            action, _, after, reward = rng.choice(values)
            return action, after, reward
        best_score = max(score for _, score, _, _ in values)
        best = [(action, after, reward) for action, score, after, reward in values if score == best_score]
        return rng.choice(best)

    def train_episode(self, rng: random.Random | None = None) -> GameResult:
        rng = rng or random
        board = new_game(rng)
        previous_after: Board | None = None
        total_reward = 0
        moves = 0

        while True:
            if not legal_actions(board):
                if previous_after is not None:
                    self.update_value(previous_after, 0.0)
                return GameResult(total_reward, max_tile(board), moves, board)

            action, after, reward = self.choose_action(board, rng, explore=True)
            if previous_after is not None:
                self.update_value(previous_after, reward + self.gamma * self.value(after))

            board, _, _ = step(board, action, rng)
            previous_after = after
            total_reward += reward
            moves += 1

    def play_episode(self, rng: random.Random | None = None) -> GameResult:
        rng = rng or random
        board = new_game(rng)
        total_reward = 0
        moves = 0
        while legal_actions(board):
            _, after, reward = self.choose_action(board, rng, explore=False)
            board = _spawn_from_after(after, rng)
            total_reward += reward
            moves += 1
        return GameResult(total_reward, max_tile(board), moves, board)

    def save(self, path: str) -> None:
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "episodes_trained": self.episodes_trained,
            "patterns": self.patterns,
            "weights": dict(self.weights),
        }
        with open(path, "wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "NTupleAgent":
        with open(path, "rb") as file:
            payload = pickle.load(file)
        agent = cls(
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            episodes_trained=payload.get("episodes_trained", 0),
            patterns=tuple(payload["patterns"]),
        )
        agent.weights.update(payload["weights"])
        return agent


def _feature(board: Board, pattern: Pattern) -> tuple[int, ...]:
    return tuple(board[index] for index in pattern)


def _spawn_from_after(after: Board, rng: random.Random) -> Board:
    from .game2048 import add_random_tile

    return add_random_tile(after, rng)


def summarize_results(results: Iterable[GameResult]) -> dict[str, float]:
    items = list(results)
    if not items:
        return {"games": 0}
    scores = [result.score for result in items]
    tiles = [result.max_tile for result in items]
    return {
        "games": len(items),
        "avg_score": sum(scores) / len(scores),
        "best_score": max(scores),
        "avg_max_tile": sum(tiles) / len(tiles),
        "best_max_tile": max(tiles),
    }
