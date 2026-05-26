"""Policy-prior helpers for ranking legal combat actions.

This module intentionally stays independent of the external ``sts_lightspeed``
binding.  The ranker only needs a combat state object, an iterable of legal
action objects, and a prior scorer that assigns each legal action a numeric
score.  That keeps the interface usable for handcrafted toy tests now and for a
learned policy model later.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


ActionKey = str
ActionKeyFn = Callable[[Any], ActionKey]
ActionScoreFn = Callable[[Any, Any], float]


class CombatPolicyPrior(Protocol):
    """Batch scoring interface for legal combat actions.

    Larger scores are better.  Scores may be probabilities, log-probabilities,
    logits, or heuristic values as long as they are finite and comparable for a
    single call.  Implementations must return one score for each action in the
    same order as ``legal_actions``.
    """

    def score_actions(self, state: Any, legal_actions: Sequence[Any]) -> Sequence[float]:
        """Return prior scores aligned with ``legal_actions``."""


@dataclass(frozen=True)
class RankedCombatAction:
    """One legal action annotated with its policy-prior rank."""

    action: Any
    prior: float
    rank: int
    action_key: ActionKey


@dataclass(frozen=True)
class UniformCombatPolicyPrior:
    """Assign the same prior score to every legal action."""

    score: float = 0.0

    def score_actions(self, state: Any, legal_actions: Sequence[Any]) -> tuple[float, ...]:
        del state
        return tuple(self.score for _ in legal_actions)


@dataclass(frozen=True)
class MappingCombatPolicyPrior:
    """Score actions from a mapping keyed by ``action_key``."""

    scores_by_key: Mapping[ActionKey, float]
    default_score: float = 0.0
    action_key: ActionKeyFn | None = None

    def score_actions(self, state: Any, legal_actions: Sequence[Any]) -> tuple[float, ...]:
        del state
        key_fn = self.action_key or default_action_key
        return tuple(float(self.scores_by_key.get(key_fn(action), self.default_score)) for action in legal_actions)


@dataclass(frozen=True)
class CallableCombatPolicyPrior:
    """Adapt a per-action scoring callable to the batch prior protocol."""

    score_action: ActionScoreFn

    def score_actions(self, state: Any, legal_actions: Sequence[Any]) -> tuple[float, ...]:
        return tuple(float(self.score_action(state, action)) for action in legal_actions)


def default_action_key(action: Any) -> ActionKey:
    """Best-effort stable key for sorting and mapping arbitrary action objects."""

    explicit_key = getattr(action, "action_key", None)
    if callable(explicit_key):
        return str(explicit_key())
    if explicit_key is not None:
        return str(explicit_key)

    for attribute in ("key", "id", "name"):
        value = getattr(action, attribute, None)
        if value is not None:
            return str(value)

    return repr(action)


@dataclass(frozen=True)
class CombatActionRanker:
    """Rank legal combat actions by policy prior with deterministic tie breaks."""

    prior: CombatPolicyPrior
    action_key: ActionKeyFn = default_action_key

    def rank_actions(
        self,
        state: Any,
        legal_actions: Iterable[Any],
        *,
        limit: int | None = None,
    ) -> tuple[RankedCombatAction, ...]:
        """Return legal actions sorted by descending prior.

        Ties are resolved by ``action_key`` and then original index, which makes
        rankings reproducible for toy fixtures and simulator action objects.
        ``limit`` can be used as the first policy-pruning hook.
        """

        actions = tuple(legal_actions)
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if not actions or limit == 0:
            return ()

        scores = tuple(float(score) for score in self.prior.score_actions(state, actions))
        if len(scores) != len(actions):
            raise ValueError(
                "policy prior returned the wrong number of scores: "
                f"expected {len(actions)}, got {len(scores)}"
            )

        keyed = []
        for index, (action, score) in enumerate(zip(actions, scores)):
            if not math.isfinite(score):
                raise ValueError(f"policy prior score must be finite for action index {index}: {score!r}")
            keyed.append((action, score, self.action_key(action), index))

        keyed.sort(key=lambda item: (-item[1], item[2], item[3]))
        if limit is not None:
            keyed = keyed[:limit]

        return tuple(
            RankedCombatAction(action=action, prior=score, rank=rank, action_key=action_key)
            for rank, (action, score, action_key, _index) in enumerate(keyed, start=1)
        )

    def select_actions(
        self,
        state: Any,
        legal_actions: Iterable[Any],
        *,
        limit: int | None = None,
    ) -> tuple[Any, ...]:
        """Return only action objects from ``rank_actions``."""

        return tuple(ranked.action for ranked in self.rank_actions(state, legal_actions, limit=limit))
