"""Fast heuristic Expectimax player for 2048."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os
import random
import time
from typing import Iterable

import numpy as np
from numba import njit

from .game2048 import ACTION_NAMES, DOWN, LEFT, RIGHT, UP, Action, Board, GameResult, max_tile, new_game


MAX_DEPTH = 5
ROW_COUNT = 1 << 16
ROW_MASK = np.uint64(0xFFFF)
TILE_MASK = np.uint64(0xF)
FULL_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
MAX_NODE = 0
CHANCE_NODE = 1
ACTION_ORDER: tuple[Action, ...] = (UP, RIGHT, DOWN, LEFT)


@dataclass(frozen=True)
class EvaluationSummary:
    games: int
    avg_score: float
    best_score: int
    avg_max_tile: float
    best_max_tile: int
    tile_counts: dict[int, int]
    wins_2048: int
    win_rate_2048: float
    avg_moves: float
    avg_seconds_per_game: float
    total_seconds: float


class ExpectimaxAgent:
    """A deterministic 2048 player using exact chance nodes and board heuristics."""

    def __init__(self, depth: int = 3) -> None:
        validate_depth(depth)
        self.depth = depth

    def warm_up(self) -> None:
        board = pack_board((1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        move_packed(np.uint64(board), LEFT)
        heuristic_score(np.uint64(board))
        self.choose_action(board)

    def choose_action(self, board: int) -> Action | None:
        action = int(choose_action_numba(np.uint64(board), self.depth))
        return None if action < 0 else action

    def play_episode(self, rng: random.Random) -> GameResult:
        board = pack_board(new_game(rng))
        total_reward = 0
        moves = 0
        while True:
            action = self.choose_action(board)
            if action is None:
                return GameResult(total_reward, packed_max_tile(board), moves, unpack_board(board))
            after, reward = move_packed(np.uint64(board), action)
            board = add_random_tile_packed(int(after), rng)
            total_reward += int(reward)
            moves += 1


def validate_depth(depth: int) -> None:
    if depth < 1 or depth > MAX_DEPTH:
        raise ValueError(f"depth must be between 1 and {MAX_DEPTH}")


def evaluate_games(agent: ExpectimaxAgent, games: int, seed: int, workers: int = 1) -> EvaluationSummary:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    game_seeds = _game_seeds(games, seed)
    start = time.monotonic()
    if workers == 1:
        results = [_play_seed(agent.depth, game_seed) for game_seed in game_seeds]
    else:
        worker_count = min(workers, games)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(lambda game_seed: _play_seed(agent.depth, game_seed), game_seeds))
    total_seconds = time.monotonic() - start
    return summarize_results(results, total_seconds)


def auto_worker_count() -> int:
    return os.cpu_count() or 1


def _game_seeds(games: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(0, 2**63) for _ in range(games)]


def _play_seed(depth: int, seed: int) -> GameResult:
    return ExpectimaxAgent(depth).play_episode(random.Random(seed))


def summarize_results(results: Iterable[GameResult], total_seconds: float = 0.0) -> EvaluationSummary:
    items = list(results)
    if not items:
        return EvaluationSummary(0, 0.0, 0, 0.0, 0, {}, 0, 0.0, 0.0, 0.0, total_seconds)
    scores = [result.score for result in items]
    tiles = [result.max_tile for result in items]
    moves = [result.moves for result in items]
    tile_counts: dict[int, int] = {}
    for tile in tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    wins = sum(1 for tile in tiles if tile >= 2048)
    games = len(items)
    return EvaluationSummary(
        games=games,
        avg_score=sum(scores) / games,
        best_score=max(scores),
        avg_max_tile=sum(tiles) / games,
        best_max_tile=max(tiles),
        tile_counts=tile_counts,
        wins_2048=wins,
        win_rate_2048=wins / games,
        avg_moves=sum(moves) / games,
        avg_seconds_per_game=total_seconds / games if games else 0.0,
        total_seconds=total_seconds,
    )


def pack_board(board: Board) -> int:
    packed = 0
    for index, value in enumerate(board):
        if value < 0 or value > 15:
            raise ValueError("packed board supports tile exponents from 0 to 15")
        packed |= int(value) << (index * 4)
    return packed


def unpack_board(packed: int) -> Board:
    return tuple((int(packed) >> (index * 4)) & 0xF for index in range(16))


def add_random_tile_packed(board: int, rng: random.Random) -> int:
    empties = [index for index in range(16) if ((board >> (index * 4)) & 0xF) == 0]
    if not empties:
        return board
    index = rng.choice(empties)
    tile = 1 if rng.random() < 0.9 else 2
    return board | (tile << (index * 4))


def packed_max_tile(board: int) -> int:
    high = max((board >> (index * 4)) & 0xF for index in range(16))
    return 0 if high == 0 else 1 << high


def spawn_probability_sum(board: int) -> float:
    empties = sum(1 for index in range(16) if ((board >> (index * 4)) & 0xF) == 0)
    if empties == 0:
        return 0.0
    return empties * ((0.9 / empties) + (0.1 / empties))


def action_name(action: Action | None) -> str:
    return "none" if action is None else ACTION_NAMES[action]


def _chance_value(after: int, depth: int, cache: dict[tuple[int, int, int], float]) -> float:
    key = (after, depth, CHANCE_NODE)
    cached = cache.get(key)
    if cached is not None:
        return cached

    empties = [index for index in range(16) if ((after >> (index * 4)) & 0xF) == 0]
    if not empties:
        value = 0.0
    else:
        probability_per_cell = 1.0 / len(empties)
        value = 0.0
        for index in empties:
            shift = index * 4
            value += 0.9 * probability_per_cell * _max_value(after | (1 << shift), depth, cache)
            value += 0.1 * probability_per_cell * _max_value(after | (2 << shift), depth, cache)
    cache[key] = value
    return value


def _max_value(board: int, depth: int, cache: dict[tuple[int, int, int], float]) -> float:
    key = (board, depth, MAX_NODE)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if depth <= 0:
        value = float(heuristic_score(np.uint64(board)))
        cache[key] = value
        return value

    best = float("-inf")
    for action in ACTION_ORDER:
        after, reward = move_packed(np.uint64(board), action)
        if after == board:
            continue
        value = float(reward) + _chance_value(int(after), depth - 1, cache)
        if value > best:
            best = value
    if best == float("-inf"):
        best = 0.0
    cache[key] = best
    return best


def _build_row_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_rows = np.zeros(ROW_COUNT, dtype=np.uint16)
    right_rows = np.zeros(ROW_COUNT, dtype=np.uint16)
    left_rewards = np.zeros(ROW_COUNT, dtype=np.int64)
    right_rewards = np.zeros(ROW_COUNT, dtype=np.int64)
    for row_id in range(ROW_COUNT):
        row = [(row_id >> (index * 4)) & 0xF for index in range(4)]
        merged, reward = _merge_row_left(row)
        left_rows[row_id] = _pack_row(merged)
        left_rewards[row_id] = reward
        reversed_merged, reward = _merge_row_left(list(reversed(row)))
        right_rows[row_id] = _pack_row(list(reversed(reversed_merged)))
        right_rewards[row_id] = reward
    return left_rows, right_rows, left_rewards, right_rewards


def _merge_row_left(row: list[int]) -> tuple[list[int], int]:
    compact = [value for value in row if value]
    merged: list[int] = []
    reward = 0
    index = 0
    while index < len(compact):
        value = compact[index]
        if index + 1 < len(compact) and compact[index + 1] == value:
            reward += 1 << (value + 1)
            value = min(value + 1, 15)
            index += 2
        else:
            index += 1
        merged.append(value)
    merged.extend([0] * (4 - len(merged)))
    return merged, reward


def _pack_row(row: list[int]) -> int:
    return sum(value << (index * 4) for index, value in enumerate(row))


ROW_LEFT, ROW_RIGHT, ROW_LEFT_REWARD, ROW_RIGHT_REWARD = _build_row_tables()
SNAKE_WEIGHTS = np.array(
    [
        [16, 15, 14, 13, 9, 10, 11, 12, 8, 7, 6, 5, 1, 2, 3, 4],
        [13, 14, 15, 16, 12, 11, 10, 9, 5, 6, 7, 8, 4, 3, 2, 1],
        [4, 3, 2, 1, 5, 6, 7, 8, 12, 11, 10, 9, 13, 14, 15, 16],
        [1, 2, 3, 4, 8, 7, 6, 5, 9, 10, 11, 12, 16, 15, 14, 13],
    ],
    dtype=np.float64,
)


@njit(cache=True, nogil=True)
def _get_tile(board: int, index: int) -> int:
    return int((np.uint64(board) >> (index * 4)) & TILE_MASK)


@njit(cache=True, nogil=True)
def _set_tile(board: np.uint64, index: int, value: int) -> np.uint64:
    shift = index * 4
    clear_mask = FULL_MASK ^ (TILE_MASK << shift)
    return (board & clear_mask) | (np.uint64(value) << shift)


@njit(cache=True, nogil=True)
def move_packed(board: int, action: int) -> tuple[np.uint64, int]:
    source = np.uint64(board)
    result = np.uint64(0)
    reward = 0
    if action == LEFT:
        for row in range(4):
            shift = row * 16
            row_id = int((source >> shift) & ROW_MASK)
            result |= np.uint64(ROW_LEFT[row_id]) << shift
            reward += int(ROW_LEFT_REWARD[row_id])
        return result, reward
    if action == RIGHT:
        for row in range(4):
            shift = row * 16
            row_id = int((source >> shift) & ROW_MASK)
            result |= np.uint64(ROW_RIGHT[row_id]) << shift
            reward += int(ROW_RIGHT_REWARD[row_id])
        return result, reward
    if action == UP:
        for col in range(4):
            row_id = 0
            for row in range(4):
                row_id |= _get_tile(source, row * 4 + col) << (row * 4)
            next_row = int(ROW_LEFT[row_id])
            reward += int(ROW_LEFT_REWARD[row_id])
            for row in range(4):
                result = _set_tile(result, row * 4 + col, (next_row >> (row * 4)) & 0xF)
        return result, reward
    if action == DOWN:
        for col in range(4):
            row_id = 0
            for row in range(4):
                row_id |= _get_tile(source, row * 4 + col) << (row * 4)
            next_row = int(ROW_RIGHT[row_id])
            reward += int(ROW_RIGHT_REWARD[row_id])
            for row in range(4):
                result = _set_tile(result, row * 4 + col, (next_row >> (row * 4)) & 0xF)
        return result, reward
    return source, 0


@njit(cache=True, nogil=True)
def heuristic_score(board: int) -> float:
    empty_count = 0
    max_exp = 0
    max_index = 0
    value_sum = 0.0
    exponents = np.empty(16, dtype=np.int64)
    values = np.empty(16, dtype=np.float64)
    for index in range(16):
        exp = _get_tile(board, index)
        exponents[index] = exp
        if exp == 0:
            empty_count += 1
            values[index] = 0.0
            continue
        if exp > max_exp:
            max_exp = exp
            max_index = index
        tile_value = 2.0**exp
        values[index] = tile_value
        value_sum += tile_value

    best_snake = 0.0
    for path in range(4):
        score = 0.0
        for index in range(16):
            score += values[index] * SNAKE_WEIGHTS[path, index]
        if score > best_snake:
            best_snake = score

    smoothness = 0.0
    merge_potential = 0.0
    for row in range(4):
        for col in range(3):
            left = exponents[row * 4 + col]
            right = exponents[row * 4 + col + 1]
            if left != 0 and right != 0:
                diff = left - right
                smoothness += abs(diff)
                if left == right:
                    merge_potential += 2.0 ** (left + 1)
    for col in range(4):
        for row in range(3):
            top = exponents[row * 4 + col]
            bottom = exponents[(row + 1) * 4 + col]
            if top != 0 and bottom != 0:
                diff = top - bottom
                smoothness += abs(diff)
                if top == bottom:
                    merge_potential += 2.0 ** (top + 1)

    corner_bonus = 0.0
    if max_index == 0 or max_index == 3 or max_index == 12 or max_index == 15:
        corner_bonus = (2.0**max_exp) * 8.0

    return (
        value_sum * 2.0
        + float(empty_count) * 10000.0
        + best_snake * 8.0
        + merge_potential * 25.0
        + corner_bonus
        - smoothness * 250.0
    )


@njit(nogil=True)
def choose_action_numba(board: np.uint64, depth: int) -> int:
    best_action = -1
    best_value = -1.0e30
    for action in range(4):
        after, reward = move_packed(board, action)
        if after == board:
            continue
        value = float(reward) + _expectimax_value(after, depth - 1, True)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action


@njit(nogil=True)
def _expectimax_value(board: np.uint64, depth: int, is_chance: bool) -> float:
    if is_chance:
        empty_count = 0
        for index in range(16):
            if _get_tile(board, index) == 0:
                empty_count += 1
        if empty_count == 0:
            return _expectimax_value(board, depth, False)
        probability_per_cell = 1.0 / float(empty_count)
        value = 0.0
        for index in range(16):
            if _get_tile(board, index) == 0:
                shift = index * 4
                value += 0.9 * probability_per_cell * _expectimax_value(board | (np.uint64(1) << shift), depth, False)
                value += 0.1 * probability_per_cell * _expectimax_value(board | (np.uint64(2) << shift), depth, False)
        return value

    if depth <= 0:
        return heuristic_score(board)

    best = -1.0e30
    for action in range(4):
        after, reward = move_packed(board, action)
        if after == board:
            continue
        value = float(reward) + _expectimax_value(after, depth - 1, True)
        if value > best:
            best = value
    if best == -1.0e30:
        return 0.0
    return best
