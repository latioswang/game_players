"""Small beam-search skeleton for abstract combat_api adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, Sequence, TypeVar


ActionT = TypeVar("ActionT")
StateT = TypeVar("StateT")


class ActionOutcome(Protocol[StateT]):
    """Minimal outcome shape returned by a combat simulator step."""

    @property
    def after(self) -> StateT:
        """State after applying the action."""


class CombatSimulator(Protocol[StateT, ActionT]):
    """Protocol subset expected from combat_api.CombatSimulator."""

    def legal_actions(self, state: StateT) -> Sequence[ActionT]:
        """Return actions that can be passed to step for this state."""

    def step(self, state: StateT, action: ActionT) -> ActionOutcome[StateT]:
        """Apply one legal action and return the transition outcome."""

    def clone(self, state: StateT) -> StateT:
        """Return an isolated copy or equivalent immutable snapshot."""

    def advance_to_decision(self, state: StateT) -> StateT:
        """Advance automatic phases until the next policy decision or terminal."""


Evaluator = Callable[[StateT], float]


@dataclass(frozen=True)
class BeamSearchConfig:
    """Budget knobs for a deterministic player-turn beam search."""

    max_depth: int = 6
    beam_width: int = 8

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.beam_width < 1:
            raise ValueError("beam_width must be at least 1")


@dataclass(frozen=True)
class SearchResult(Generic[StateT, ActionT]):
    """Best line found by beam search."""

    action: ActionT
    score: float
    principal_variation: tuple[ActionT, ...]
    leaf_state: StateT
    expanded_nodes: int


@dataclass(frozen=True)
class _BeamNode(Generic[StateT, ActionT]):
    state: StateT
    score: float
    line: tuple[ActionT, ...]
    order: int


class BeamSearchCombatPlayer(Generic[StateT, ActionT]):
    """Choose combat actions by scoring a bounded beam of action sequences."""

    def __init__(
        self,
        simulator: CombatSimulator[StateT, ActionT],
        evaluator: Evaluator[StateT],
        config: BeamSearchConfig | None = None,
    ) -> None:
        self.simulator = simulator
        self.evaluator = evaluator
        self.config = config or BeamSearchConfig()

    def choose_action(
        self,
        state: StateT,
        legal_actions: Sequence[ActionT] | None = None,
    ) -> ActionT | None:
        """Return the first action of the best line, or None if no action exists."""

        result = self.search(state, legal_actions=legal_actions)
        if result is None:
            return None
        return result.action

    def search(
        self,
        state: StateT,
        legal_actions: Sequence[ActionT] | None = None,
    ) -> SearchResult[StateT, ActionT] | None:
        """Run deterministic beam search from the supplied state."""

        state = self.simulator.advance_to_decision(self.simulator.clone(state))
        if _is_terminal(state):
            return None

        if legal_actions is None:
            first_actions = tuple(self.simulator.legal_actions(state))
        else:
            first_actions = tuple(legal_actions)
        if not first_actions:
            return None

        beam = [
            _BeamNode(
                state=state,
                score=self.evaluator(state),
                line=(),
                order=0,
            )
        ]
        expanded_nodes = 0
        next_order = 1

        for _depth in range(self.config.max_depth):
            candidates: list[_BeamNode[StateT, ActionT]] = []
            for node in beam:
                if _is_terminal(node.state):
                    candidates.append(node)
                    continue

                if not node.line:
                    actions = first_actions
                else:
                    actions = tuple(self.simulator.legal_actions(node.state))
                if not actions:
                    candidates.append(node)
                    continue

                for action in actions:
                    successor = self._successor(node.state, action)
                    expanded_nodes += 1
                    candidates.append(
                        _BeamNode(
                            state=successor,
                            score=self.evaluator(successor),
                            line=(*node.line, action),
                            order=next_order,
                        )
                    )
                    next_order += 1

            if not candidates:
                break

            candidates.sort(key=lambda candidate: (-candidate.score, candidate.order))
            beam = candidates[: self.config.beam_width]
            if all(_is_terminal(node.state) for node in beam):
                break

        best = max(beam, key=lambda node: (node.score, -node.order))
        if not best.line:
            return None

        return SearchResult(
            action=best.line[0],
            score=best.score,
            principal_variation=best.line,
            leaf_state=best.state,
            expanded_nodes=expanded_nodes,
        )

    def _successor(self, state: StateT, action: ActionT) -> StateT:
        cloned = self.simulator.clone(state)
        outcome = self.simulator.step(cloned, action)
        return self.simulator.advance_to_decision(outcome.after)


def _is_terminal(state: Any) -> bool:
    terminal = getattr(state, "is_terminal", False)
    if callable(terminal):
        return bool(terminal())
    return bool(terminal)
