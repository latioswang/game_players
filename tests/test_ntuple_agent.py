from game_players.ntuple_agent import NTupleAgent


def test_symmetry_value_is_shared_across_reflections():
    agent = NTupleAgent(alpha=1.0, use_symmetry=True)
    board = (
        1, 2, 3, 4,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    )
    reflected = (
        4, 3, 2, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    )

    agent.update_value(board, 100.0)

    assert agent.value(board) == agent.value(reflected)


def test_learning_rate_decay_respects_floors():
    agent = NTupleAgent(
        alpha=0.01,
        epsilon=0.05,
        min_alpha=0.009,
        min_epsilon=0.04,
        alpha_decay=0.1,
        epsilon_decay=0.1,
    )

    agent.decay_learning_rates()

    assert agent.alpha == 0.009
    assert agent.epsilon == 0.04

