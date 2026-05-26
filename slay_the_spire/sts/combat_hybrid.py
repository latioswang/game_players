"""Hybrid combat policy orchestration.

The hybrid player combines three independent pieces:

* ``CombatActionRanker`` supplies deterministic policy-prior ordering.
* ``BeamSearchCombatPlayer`` searches only the ranked/pruned action set.
* a value evaluator scores abstract leaf states.

The implementation intentionally depends only on the local simulator protocol
from ``combat_search``.  It works with dicts, dataclasses, or future
``combat_api`` state objects as long as the simulator can enumerate legal
actions, clone states, apply actions, and advance to a decision boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, Sequence, TypeVar

try:
    from .combat_policy import CombatActionRanker, RankedCombatAction
    from .combat_search import BeamSearchCombatPlayer, BeamSearchConfig, SearchResult
    from .combat_value import HandcraftedCombatValueEvaluator
except ImportError:  # Support direct imports from slay_the_spire/sts tests.
    from combat_policy import CombatActionRanker, RankedCombatAction
    from combat_search import BeamSearchCombatPlayer, BeamSearchConfig, SearchResult
    from combat_value import HandcraftedCombatValueEvaluator


ActionT = TypeVar("ActionT")
StateT = TypeVar("StateT")


class ValueEvaluator(Protocol[StateT]):
    """Evaluator shape accepted by ``HybridCombatPlayer``."""

    def evaluate(self, state: StateT) -> Any:
        """Return either a numeric score or an object with a numeric score."""


ValueScorer = Callable[[StateT], float]
ValueEvaluatorLike = ValueEvaluator[StateT] | ValueScorer[StateT]


@dataclass(frozen=True)
class HybridCombatConfig:
    """Budget and pruning knobs for the hybrid combat player."""

    search: BeamSearchConfig = field(default_factory=BeamSearchConfig)
    policy_action_limit: int | None = None

    def __post_init__(self) -> None:
        if self.policy_action_limit is not None and self.policy_action_limit < 0:
            raise ValueError("policy_action_limit must be non-negative")


@dataclass(frozen=True)
class HybridSearchResult(Generic[StateT, ActionT]):
    """Search result plus the root policy-prior ranking used for pruning."""

    action: ActionT
    score: float
    principal_variation: tuple[ActionT, ...]
    leaf_state: StateT
    expanded_nodes: int
    ranked_actions: tuple[RankedCombatAction, ...]
    pruned_actions: tuple[RankedCombatAction, ...]
    search_result: SearchResult[StateT, ActionT]


class HybridCombatPlayer(Generic[StateT, ActionT]):
    """Choose combat actions by policy pruning plus value-guided beam search."""

    def __init__(
        self,
        simulator: Any,
        ranker: CombatActionRanker,
        value_evaluator: ValueEvaluatorLike[StateT] | None = None,
        config: HybridCombatConfig | None = None,
    ) -> None:
        self.simulator = simulator
        self.ranker = ranker
        self.value_evaluator = value_evaluator or HandcraftedCombatValueEvaluator()
        self.config = config or HybridCombatConfig()

    def choose_action(
        self,
        state: StateT,
        legal_actions: Sequence[ActionT] | None = None,
    ) -> ActionT | None:
        """Return the first action of the best hybrid line, or None."""

        result = self.search(state, legal_actions=legal_actions)
        if result is None:
            return None
        return result.action

    def search(
        self,
        state: StateT,
        legal_actions: Sequence[ActionT] | None = None,
    ) -> HybridSearchResult[StateT, ActionT] | None:
        """Rank root actions, prune them, then run value-guided beam search."""

        decision_state = self.simulator.advance_to_decision(self.simulator.clone(state))
        if _is_terminal(decision_state):
            return None

        if legal_actions is None:
            root_actions = tuple(self.simulator.legal_actions(decision_state))
        else:
            root_actions = tuple(legal_actions)
        if not root_actions:
            return None

        ranked_actions = self.ranker.rank_actions(decision_state, root_actions)
        pruned_actions = _prune_ranked(ranked_actions, self.config.policy_action_limit)
        if not pruned_actions:
            return None

        pruned_simulator = _PolicyPrunedCombatSimulator[StateT, ActionT](
            simulator=self.simulator,
            ranker=self.ranker,
            action_limit=self.config.policy_action_limit,
        )
        search_player = BeamSearchCombatPlayer[StateT, ActionT](
            simulator=pruned_simulator,
            evaluator=self._score_state,
            config=self.config.search,
        )
        search_result = search_player.search(
            decision_state,
            legal_actions=tuple(entry.action for entry in pruned_actions),
        )
        if search_result is None:
            return None

        return HybridSearchResult(
            action=search_result.action,
            score=search_result.score,
            principal_variation=search_result.principal_variation,
            leaf_state=search_result.leaf_state,
            expanded_nodes=search_result.expanded_nodes,
            ranked_actions=ranked_actions,
            pruned_actions=pruned_actions,
            search_result=search_result,
        )

    def _score_state(self, state: StateT) -> float:
        evaluator = self.value_evaluator
        if hasattr(evaluator, "evaluate"):
            raw_value = evaluator.evaluate(state)  # type: ignore[union-attr]
        else:
            raw_value = evaluator(state)  # type: ignore[misc, operator]
        score = getattr(raw_value, "score", raw_value)
        return float(score)


@dataclass(frozen=True)
class _PolicyPrunedCombatSimulator(Generic[StateT, ActionT]):
    simulator: Any
    ranker: CombatActionRanker
    action_limit: int | None

    def legal_actions(self, state: StateT) -> tuple[ActionT, ...]:
        legal_actions = self.simulator.legal_actions(state)
        ranked_actions = self.ranker.rank_actions(state, legal_actions, limit=self.action_limit)
        return tuple(entry.action for entry in ranked_actions)

    def step(self, state: StateT, action: ActionT) -> Any:
        return self.simulator.step(state, action)

    def clone(self, state: StateT) -> StateT:
        return self.simulator.clone(state)

    def advance_to_decision(self, state: StateT) -> StateT:
        return self.simulator.advance_to_decision(state)


def _prune_ranked(
    ranked_actions: tuple[RankedCombatAction, ...],
    limit: int | None,
) -> tuple[RankedCombatAction, ...]:
    if limit is None:
        return ranked_actions
    return ranked_actions[:limit]


def _is_terminal(state: Any) -> bool:
    terminal = getattr(state, "is_terminal", False)
    if callable(terminal):
        return bool(terminal())
    if terminal:
        return True
    if isinstance(state, dict):
        return bool(state.get("is_terminal", state.get("terminal", False)))
    return False


__all__ = [
    "HybridCombatConfig",
    "HybridCombatPlayer",
    "HybridSearchResult",
]
