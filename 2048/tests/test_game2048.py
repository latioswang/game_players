from game_players.game2048 import DOWN, LEFT, RIGHT, UP, legal_actions, move, new_game


def test_left_move_merges_once_per_pair():
    board = (
        1, 1, 1, 0,
        2, 2, 2, 2,
        0, 0, 0, 0,
        3, 0, 3, 3,
    )

    after, reward = move(board, LEFT)

    assert after == (
        2, 1, 0, 0,
        3, 3, 0, 0,
        0, 0, 0, 0,
        4, 3, 0, 0,
    )
    assert reward == 4 + 8 + 8 + 16


def test_directional_moves():
    board = (
        1, 0, 0, 0,
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    )

    assert move(board, UP)[0][0] == 2
    assert move(board, DOWN)[0][12] == 2
    assert move(board, RIGHT)[0][3] == 1


def test_new_game_starts_with_two_tiles():
    board = new_game()

    assert sum(1 for value in board if value) == 2
    assert legal_actions(board)

