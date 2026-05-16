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


SIX_TUPLE_PATTERNS: tuple[Pattern, ...] = (
    (0, 1, 2, 4, 5, 6),
    (1, 2, 3, 5, 6, 7),
    (4, 5, 6, 8, 9, 10),
    (5, 6, 7, 9, 10, 11),
    (8, 9, 10, 12, 13, 14),
    (9, 10, 11, 13, 14, 15),
    (0, 4, 8, 1, 5, 9),
    (2, 6, 10, 3, 7, 11),
)


DEFAULT_PATTERNS: tuple[Pattern, ...] = BASE_PATTERNS + SIX_TUPLE_PATTERNS
SYMMETRY_MAPS: tuple[Pattern, ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    (12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3),
    (15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    (3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12),
    (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12),
    (12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3),
    (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15),
    (15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0),
)


@dataclass
class NTupleAgent:
    """A compact value-function agent using board pattern lookup tables."""

    alpha: float = 0.01
    gamma: float = 1.0
    epsilon: float = 0.05
    min_alpha: float = 0.001
    min_epsilon: float = 0.005
    alpha_decay: float = 0.99995
    epsilon_decay: float = 0.9999
    use_symmetry: bool = True
    episodes_trained: int = 0
    patterns: tuple[Pattern, ...] = DEFAULT_PATTERNS
    weights: DefaultDict[tuple[int, Pattern], float] = field(default_factory=lambda: defaultdict(float))

    def value(self, board: Board) -> float:
        boards = _symmetries(board) if self.use_symmetry else (board,)
        total = 0.0
        for transformed in boards:
            total += sum(self.weights[(index, _feature(transformed, pattern))] for index, pattern in enumerate(self.patterns))
        return total / len(boards)

    def update_value(self, board: Board, target: float) -> float:
        prediction = self.value(board)
        error = target - prediction
        boards = _symmetries(board) if self.use_symmetry else (board,)
        scaled = self.alpha * error / (len(self.patterns) * len(boards))
        for transformed in boards:
            for index, pattern in enumerate(self.patterns):
                self.weights[(index, _feature(transformed, pattern))] += scaled
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
                self.decay_learning_rates()
                return GameResult(total_reward, max_tile(board), moves, board)

            action, after, reward = self.choose_action(board, rng, explore=True)
            if previous_after is not None:
                self.update_value(previous_after, reward + self.gamma * self.value(after))

            board, _, _ = step(board, action, rng)
            previous_after = after
            total_reward += reward
            moves += 1

    def decay_learning_rates(self) -> None:
        self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

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
            "min_alpha": self.min_alpha,
            "min_epsilon": self.min_epsilon,
            "alpha_decay": self.alpha_decay,
            "epsilon_decay": self.epsilon_decay,
            "use_symmetry": self.use_symmetry,
            "episodes_trained": self.episodes_trained,
            "patterns": self.patterns,
            "weights": dict(self.weights),
        }
        with open(path, "wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str, upgrade_patterns: bool = True) -> "NTupleAgent":
        with open(path, "rb") as file:
            payload = pickle.load(file)
        patterns = tuple(payload["patterns"])
        if upgrade_patterns and _is_prefix(patterns, DEFAULT_PATTERNS):
            patterns = DEFAULT_PATTERNS
        agent = cls(
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            min_alpha=payload.get("min_alpha", 0.001),
            min_epsilon=payload.get("min_epsilon", 0.005),
            alpha_decay=payload.get("alpha_decay", 0.99995),
            epsilon_decay=payload.get("epsilon_decay", 0.9999),
            use_symmetry=payload.get("use_symmetry", True),
            episodes_trained=payload.get("episodes_trained", 0),
            patterns=patterns,
        )
        agent.weights.update(payload["weights"])
        return agent


def _feature(board: Board, pattern: Pattern) -> tuple[int, ...]:
    return tuple(board[index] for index in pattern)


def _symmetries(board: Board) -> tuple[Board, ...]:
    return tuple(tuple(board[index] for index in mapping) for mapping in SYMMETRY_MAPS)


def _is_prefix(prefix: tuple[Pattern, ...], patterns: tuple[Pattern, ...]) -> bool:
    return len(prefix) <= len(patterns) and patterns[: len(prefix)] == prefix


def _spawn_from_after(after: Board, rng: random.Random) -> Board:
    from .game2048 import add_random_tile

    return add_random_tile(after, rng)


def summarize_results(results: Iterable[GameResult]) -> dict[str, object]:
    items = list(results)
    if not items:
        return {"games": 0}
    scores = [result.score for result in items]
    tiles = [result.max_tile for result in items]
    tile_counts: dict[int, int] = {}
    for tile in tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    return {
        "games": len(items),
        "avg_score": sum(scores) / len(scores),
        "best_score": max(scores),
        "avg_max_tile": sum(tiles) / len(tiles),
        "best_max_tile": max(tiles),
        "tile_counts": tile_counts,
    }
