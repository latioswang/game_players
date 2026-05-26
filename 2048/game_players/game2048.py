"""Core 2048 game mechanics.

Tiles are stored as exponents: 0 means empty, 1 means 2, 2 means 4, etc.
This keeps feature keys compact and avoids repeated log2 conversions.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence


Board = tuple[int, ...]
Action = int

UP: Action = 0
RIGHT: Action = 1
DOWN: Action = 2
LEFT: Action = 3
ACTION_NAMES = {
    UP: "up",
    RIGHT: "right",
    DOWN: "down",
    LEFT: "left",
}


def empty_board() -> Board:
    return (0,) * 16


def new_game(rng: random.Random | None = None) -> Board:
    rng = rng or random
    board = empty_board()
    board = add_random_tile(board, rng)
    return add_random_tile(board, rng)


def board_score(board: Board) -> int:
    return sum((1 << value) for value in board if value)


def max_tile(board: Board) -> int:
    high = max(board)
    return 0 if high == 0 else 1 << high


def render(board: Board) -> str:
    values = [str(1 << value) if value else "." for value in board]
    width = max(len(value) for value in values)
    rows = []
    for r in range(4):
        rows.append(" ".join(value.rjust(width) for value in values[r * 4 : (r + 1) * 4]))
    return "\n".join(rows)


def add_random_tile(board: Board, rng: random.Random | None = None) -> Board:
    rng = rng or random
    empties = [index for index, value in enumerate(board) if value == 0]
    if not empties:
        return board
    index = rng.choice(empties)
    tile = 1 if rng.random() < 0.9 else 2
    board_list = list(board)
    board_list[index] = tile
    return tuple(board_list)


def can_move(board: Board, action: Action) -> bool:
    moved, _ = move(board, action)
    return moved != board


def legal_actions(board: Board) -> list[Action]:
    return [action for action in (UP, RIGHT, DOWN, LEFT) if can_move(board, action)]


def is_terminal(board: Board) -> bool:
    return not legal_actions(board)


def move(board: Board, action: Action) -> tuple[Board, int]:
    """Apply a deterministic slide/merge move without adding a random tile."""
    if action == LEFT:
        rows = [_merge_line(board[r * 4 : (r + 1) * 4]) for r in range(4)]
        return _rows_to_board([line for line, _ in rows]), sum(score for _, score in rows)
    if action == RIGHT:
        rows = [_merge_line(reversed(board[r * 4 : (r + 1) * 4])) for r in range(4)]
        merged = [tuple(reversed(line)) for line, _ in rows]
        return _rows_to_board(merged), sum(score for _, score in rows)
    if action == UP:
        cols = [_merge_line(board[c::4]) for c in range(4)]
        return _cols_to_board([line for line, _ in cols]), sum(score for _, score in cols)
    if action == DOWN:
        cols = [_merge_line(reversed(board[c::4])) for c in range(4)]
        merged = [tuple(reversed(line)) for line, _ in cols]
        return _cols_to_board(merged), sum(score for _, score in cols)
    raise ValueError(f"unknown action: {action}")


def step(board: Board, action: Action, rng: random.Random | None = None) -> tuple[Board, int, bool]:
    after, reward = move(board, action)
    if after == board:
        return board, 0, is_terminal(board)
    next_board = add_random_tile(after, rng)
    return next_board, reward, is_terminal(next_board)


def _merge_line(line: Iterable[int]) -> tuple[tuple[int, int, int, int], int]:
    compact = [value for value in line if value]
    merged: list[int] = []
    reward = 0
    index = 0
    while index < len(compact):
        value = compact[index]
        if index + 1 < len(compact) and compact[index + 1] == value:
            value += 1
            reward += 1 << value
            index += 2
        else:
            index += 1
        merged.append(value)
    merged.extend([0] * (4 - len(merged)))
    return (merged[0], merged[1], merged[2], merged[3]), reward


def _rows_to_board(rows: Sequence[Sequence[int]]) -> Board:
    return tuple(value for row in rows for value in row)


def _cols_to_board(cols: Sequence[Sequence[int]]) -> Board:
    board = [0] * 16
    for c, col in enumerate(cols):
        for r, value in enumerate(col):
            board[r * 4 + c] = value
    return tuple(board)


@dataclass(frozen=True)
class GameResult:
    score: int
    max_tile: int
    moves: int
    board: Board

