# Combat Policy Plan

This plan is scoped to Slay the Spire combat only. Deck building, route
selection, shops, rests, upgrades, and reward picking are intentionally out of
scope here.

## Goal

Build a long-run combat policy that can eventually outperform handcrafted rules
and the current external-agent baseline while remaining measurable,
inspectable, and compatible with the `sts_lightspeed` simulator.

## 1. Expose a Combat Environment API

The first requirement is a binding layer that makes combat a real decision
environment instead of an opaque call to `Agent.playout`.

Needed state:

- player HP, max HP, block, energy, powers, relic state, potion state
- hand, draw pile, discard pile, exhaust pile
- card upgrade and temporary card state
- monster HP, block, powers, intents, and targetability
- combat phase, turn number, action history, and terminal outcome

Needed operations:

- enumerate legal actions
- apply one action
- clone combat state cheaply
- advance non-player phases
- detect lethal, player death, and combat completion
- report combat reward-independent metrics such as HP loss and turns taken

This API is the foundation for every stronger combat policy. Without it, the
local code can only tune high-level `sts_lightspeed.Agent` settings.

## 2. Build a Search-Based Combat Player

Once action-level combat control exists, build a planner before training a
model. Slay the Spire combat is sequence-heavy, so search should produce useful
policies and high-quality training data earlier than pure RL.

Preferred first planner:

- beam search over player-turn action sequences
- deterministic state cloning for exact card/action effects
- leaf evaluation for expected HP, survival, lethal setup, and future hand
  quality
- explicit tactical checks for immediate lethal and imminent player death

Possible extensions:

- expectimax over draw and enemy randomness
- Monte Carlo tree search if cloning is cheap enough
- dynamic programming for deterministic turn fragments
- action pruning from simple combat rules or a learned policy

Core evaluation terms:

- win combat immediately when possible
- never accept player death unless all lines die
- minimize expected HP loss
- prevent incoming damage efficiently
- value vulnerable, weak, strength, dexterity, draw, and energy setup
- preserve rare resources such as potions when alternatives are close
- avoid wasting energy, damage, block, and exhaust effects
- prefer scaling only when the player is safe enough to benefit

## 3. Generate Combat Trajectories

Use the search player to produce deterministic, reproducible combat data.

Each trajectory should include:

- seed and combat identity
- initial deck, relics, potions, HP, and enemy encounter
- every combat state observed by the policy
- legal actions and selected action
- search scores or action-value estimates when available
- resulting state after each action
- final outcome, HP loss, turns taken, and resources consumed

Keep generated datasets out of commits unless explicitly requested. Store only
small fixtures or schemas in the repository.

## 4. Train a Combat Value Function

The highest long-run payoff is a learned value model trained from planner and
rollout data.

Primary target:

```text
combat state -> expected combat value
```

Useful value targets:

- win probability
- expected HP after combat
- expected HP loss
- death probability
- expected number of turns remaining
- resource-adjusted value that accounts for potions and exhaust costs

The value model should be used inside search as a leaf evaluator, replacing or
augmenting handcrafted evaluation terms.

## 5. Train a Combat Policy Model

Train a policy model from planner decisions once the planner produces reliable
actions.

Primary target:

```text
combat state -> action prior or ranked legal actions
```

The policy model is most useful for:

- pruning weak branches in beam search or MCTS
- speeding up large seed sweeps
- providing a fallback action when search budgets are low
- distilling expensive search into a cheaper policy

It should not replace tactical safety checks for lethal, player death, or
obvious resource waste.

## 6. Combine Policy, Value, and Search

The strongest long-run architecture is hybrid:

- handcrafted rules handle critical tactical invariants
- a policy model proposes promising actions
- search explores action sequences
- a value model scores leaf states
- deterministic seed sweeps compare the hybrid policy against baselines

This keeps the combat player strong, inspectable, and fast enough to improve
iteratively.

## Evaluation

Use deterministic seed suites and compare against the current external-agent
baseline. Until action-level combat bindings exist, live comparisons are limited
to full-run metrics exposed by the current `sts_lightspeed` Python binding.
Combat-only metrics become live baseline metrics once the binding can pause at
combat decisions, apply one action, and report combat outcomes.

Track:

- win rate
- average floor reached in full-run tests
- combat survival rate
- average HP loss per combat
- boss and elite survival
- turns per combat
- potion usage
- search time per decision
- regression seeds where the new policy performs worse

Tests should cover mechanics-sensitive behavior with fixed seeds and small
combat fixtures once the combat API supports them.

## Recommended Sequence

1. Add action-level combat bindings or a local adapter around equivalent
   `sts_lightspeed` internals.
2. Implement deterministic combat state serialization and cloning tests.
3. Build a beam-search combat player with handcrafted leaf evaluation.
4. Run fixed-seed comparisons against `sts_lightspeed.Agent`.
5. Save planner trajectories for model training.
6. Train a value function and use it as the planner leaf evaluator.
7. Train a policy model for action pruning and low-budget decisions.
8. Iterate on the hybrid policy with regression seed tracking.
