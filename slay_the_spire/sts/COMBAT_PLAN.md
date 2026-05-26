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

Current checkpoint:

- Completed locally: combat state/action contracts, fixture and protocol
  adapters, deterministic serialization helpers, fixture cloning tests,
  beam-search planning over abstract combat simulators, JSONL trajectory
  records, handcrafted value scoring, linear value-function training,
  policy action priors, hybrid policy orchestration, fixed-seed fixture
  experiments, regression tracking, combat metric aggregation, a dict-payload
  `sts_lightspeed` combat adapter, a `CombatBackendSimulator` bridge for search
  callers, fake-handle adapter tests, and env-gated live conformance smoke
  tests.
- Not completed live: the Python `sts_lightspeed` binding still does not
  expose combat legal-action enumeration, single-action application,
  action-level combat pause/resume, cheap live combat cloning, or live
  combat-only outcome metrics.

Current Recommended Sequence increment:

1. The local payload adapter boundary is implemented. It consumes stable
   `sts_lightspeed` legal-action payloads, emits the existing combat action
   contract without changing fixture behavior, re-enumerates current legal
   payloads at apply time, validates live freshness metadata such as
   `binding_action_id` and `decision_id`, and rejects stale action keys
   explicitly. Missing freshness metadata is treated as an unsafe live payload.
2. Payload adapter acceptance is covered by fake-handle tests for card/target,
   end-turn, metrics, clone independence, search-facing simulator wrapping, and
   stale action rejection, including reused stable keys such as `end_turn`.
   Potion, potion-discard, and card-select payloads remain
   binding-conformance cases once upstream emits real examples.
3. Add the upstream live-binding bridge behind the adapter. It should pause a fixed-seed
   combat at each player decision, enumerate payload-backed legal actions,
   apply exactly one selected action, refresh the combat snapshot, and return
   combat metrics such as HP loss, turns, potion use, survival, and terminal
   outcome. The local simulator bridge computes per-action deltas from
   before/after backend metrics.
4. Accept the live-binding bridge only when a deterministic smoke combat can
   execute several player decisions through the bridge while preserving legal
   action identity, selected action application, state refresh, and metric
   accounting.
5. Extend deterministic serialization and cloning coverage to live
   `sts_lightspeed` combat states once upstream exposes either cheap cloning or
   deterministic replay support.
6. Run the existing beam-search combat player against live fixed-seed combats.
7. Compare live combat-only metrics against `sts_lightspeed.Agent`.
8. Save live planner trajectories for model training.
9. Train or tune the value function as the live planner leaf evaluator.
10. Train or tune the policy model for action pruning and low-budget decisions.
11. Iterate on the hybrid policy with regression seed tracking.

Remaining upstream binding work:

- expose stable legal-action payloads with command type, card index, target
  index, potion index, card-select metadata, and discard metadata
- expose a pybind operation that applies one combat action and returns the
  updated combat snapshot or a resumable decision handle
- preserve action-level pause/resume across normal hand actions, card-select
  tasks, potion usage, potion discard, monster turns, and combat completion
- provide cheap live combat cloning or deterministic replay suitable for search
- surface combat-only outcome, turn, HP-loss, resource-use, and terminal-state
  metrics without relying on full-run aggregate stats
