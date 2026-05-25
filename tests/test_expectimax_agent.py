import pytest

from game_players.cli import _resolve_workers, _validate_depth_or_exit
from game_players.expectimax_agent import (
    ExpectimaxAgent,
    evaluate_games,
    heuristic_score,
    move_packed,
    pack_board,
    spawn_probability_sum,
    unpack_board,
)
from game_players.game2048 import DOWN, LEFT, RIGHT, UP, legal_actions, move


def test_row_transition_table_matches_reference_moves():
    board = (
        1, 1, 1, 0,
        2, 2, 2, 2,
        0, 0, 0, 0,
        3, 0, 3, 3,
    )
    packed = pack_board(board)

    for action in (UP, RIGHT, DOWN, LEFT):
        expected_board, expected_reward = move(board, action)
        actual_board, actual_reward = move_packed(packed, action)
        assert unpack_board(int(actual_board)) == expected_board
        assert actual_reward == expected_reward


def test_spawn_probabilities_sum_to_one():
    board = (
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 0,
    )

    assert spawn_probability_sum(pack_board(board)) == pytest.approx(1.0)


def test_expectimax_returns_legal_action():
    board = (
        1, 0, 0, 0,
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    )
    action = ExpectimaxAgent(depth=2).choose_action(pack_board(board))

    assert action in legal_actions(board)


def test_expectimax_is_deterministic_across_depths():
    board = (
        4, 3, 2, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    )
    packed = pack_board(board)

    assert ExpectimaxAgent(depth=1).choose_action(packed) == ExpectimaxAgent(depth=1).choose_action(packed)
    assert ExpectimaxAgent(depth=3).choose_action(packed) == ExpectimaxAgent(depth=3).choose_action(packed)


def test_heuristic_prefers_empty_cornered_monotonic_boards():
    sparse_cornered = pack_board(
        (
            4, 3, 2, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        )
    )
    crowded_cornered = pack_board(
        (
            4, 3, 2, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        )
    )
    centered_max = pack_board(
        (
            3, 2, 1, 0,
            0, 4, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        )
    )
    nonmonotonic = pack_board(
        (
            2, 4, 1, 3,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        )
    )

    assert heuristic_score(sparse_cornered) > heuristic_score(crowded_cornered)
    assert heuristic_score(sparse_cornered) > heuristic_score(centered_max)
    assert heuristic_score(sparse_cornered) > heuristic_score(nonmonotonic)


def test_cli_rejects_invalid_depth():
    with pytest.raises(SystemExit):
        _validate_depth_or_exit(0)
    with pytest.raises(SystemExit):
        _validate_depth_or_exit(6)


def test_parallel_evaluation_keeps_requested_game_count():
    agent = ExpectimaxAgent(depth=1)
    agent.warm_up()

    summary = evaluate_games(agent, games=2, seed=1, workers=2)

    assert summary.games == 2
    assert sum(summary.tile_counts.values()) == 2


def test_worker_resolution_accepts_auto_and_rejects_invalid_values():
    assert _resolve_workers("1") == 1
    assert _resolve_workers("auto") >= 1
    with pytest.raises(SystemExit):
        _resolve_workers("0")
    with pytest.raises(SystemExit):
        _resolve_workers("many")
