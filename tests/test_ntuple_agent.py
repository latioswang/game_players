from argparse import Namespace

from game_players.cli import _configure_ntuple_agent
from game_players.ntuple_agent import NTupleAgent


def test_fresh_ntuple_defaults_use_fixed_epsilon_and_no_symmetry():
    agent = NTupleAgent(use_symmetry=True, epsilon_decay=0.5)
    args = Namespace(
        alpha=None,
        epsilon=None,
        min_alpha=None,
        min_epsilon=None,
        alpha_decay=None,
        epsilon_decay=None,
        symmetry="auto",
    )

    _configure_ntuple_agent(agent, args, is_fresh=True)

    assert agent.use_symmetry is False
    assert agent.epsilon_decay == 1.0


def test_resume_ntuple_preserves_saved_values_by_default():
    agent = NTupleAgent(use_symmetry=True, epsilon_decay=0.75)
    args = Namespace(
        alpha=None,
        epsilon=None,
        min_alpha=None,
        min_epsilon=None,
        alpha_decay=None,
        epsilon_decay=None,
        symmetry="auto",
    )

    _configure_ntuple_agent(agent, args, is_fresh=False)

    assert agent.use_symmetry is True
    assert agent.epsilon_decay == 0.75


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
