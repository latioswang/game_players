# sts_lightspeed Combat Binding Plan

This note records the live binding work needed to move the combat policy stack
from deterministic local fixtures to real `sts_lightspeed` combats.

The current repository does not vendor `sts_lightspeed`, and the installed
Python binding used by this repo exposes `GameContext`, `Agent.playout`, card
reward hooks, and run-level fields only. It does not expose action-level combat
control. The local combat modules therefore remain a target interface until the
upstream binding adds the methods below.

## Upstream Internals To Expose

Read-only upstream inspection found these relevant C++ structures:

- `BattleContext` owns combat state, RNGs, player, monsters, cards, potions,
  action queues, turn number, input state, and combat outcome.
- `InputState::PLAYER_NORMAL` is the main player-decision state for normal
  card and potion decisions.
- `BattleSimulator` already wraps a `BattleContext` and has `initBattle`,
  `printActions`, `takeAction`, `takeNormalAction`, `takePotionAction`,
  `isBattleComplete`, and `exitBattle`.
- `BattleContext` has a default copy constructor, which makes cheap cloned
  search states plausible if the binding owns a value copy or copies the
  simulator's `BattleContext`.
- `CardManager` exposes hand, draw pile, discard pile, exhaust pile, and unique
  card ids.
- `Monster` exposes HP, block, targetability, move state, powers, and intent
  helpers such as `getMoveBaseDamage`.

## Required Python Binding Surface

Add a combat adapter class in the upstream pybind module, or equivalent methods
on an exposed simulator object:

- `init_combat(game_context, encounter=None) -> CombatHandle`; implementing the
  optional encounter requires wrapping `BattleContext` directly or adding a
  `BattleSimulator::initBattle(gc, encounter)` overload because the current
  `BattleSimulator::initBattle` accepts only `GameContext`
- `CombatHandle.observe() -> dict`
- `CombatHandle.legal_actions() -> list[dict]`
- `CombatHandle.apply(action: dict | str) -> dict`
- `CombatHandle.clone() -> CombatHandle`
- `CombatHandle.advance_to_player_decision() -> None`
- `CombatHandle.is_terminal() -> bool`
- `CombatHandle.metrics() -> dict`
- `CombatHandle.exit_to_game_context(game_context) -> None`

The returned payloads should map cleanly to `combat_api.py`:

- player fields map to `PlayerState`
- card instance fields map to `CardState`
- monster fields map to `MonsterState`
- legal action payloads map to `CombatAction`
- terminal and per-step metrics map to `CombatMetrics` and `ActionOutcome`

## Local Payload Adapter Path

Keep the pybind surface dictionary-based and add the local conformance layer in
this repository. The adapter should wrap a live `CombatHandle`, validate each
payload, and expose the existing `CombatBackend` protocol:

- `observe()` converts the live snapshot into `CombatObservation`
- `legal_actions()` converts stable live action payloads into `CombatAction`
- `apply(action)` re-enumerates current legal payloads, matches the selected
  action by stable `CombatAction.action_key()` plus freshness metadata such as
  `binding_action_id` and `decision_id`, rejects stale ids, and forwards the
  current full payload to the handle
- `clone()` wraps `CombatHandle.clone()` and preserves deterministic state
- `advance_to_player_decision()`, `is_terminal()`, and `metrics()` forward to
  the handle and normalize the result into local combat types
- `CombatBackendSimulator` wraps the stateful backend for search callers and
  computes per-action metric deltas from before/after backend metrics. Wrapped
  states snapshot terminal status so historical outcomes are not affected by
  later backend mutations.

This keeps search, value models, trajectories, and evaluation coupled only to
`CombatBackend`, while the adapter owns all pybind payload drift and validation.
The conformance tests should use fixed seeds and run only when
`STS_LIGHTSPEED_MODULE_DIR` points at a compiled `slaythespire` module.

## Legal Action Encoding

Use action payloads with both a stable search id and the current upstream
command/index data needed for application. `BattleSimulator::takeAction`
currently consumes hand indexes and potion indexes, so the adapter must resolve
stable card ids back to the current live hand index at apply time and reject
stale ids after clone/apply mutations.

Core stable ids:

- `end_turn`
- `play:<card_unique_id>:<target_idx>`
- `drink:<potion_idx>:<target_idx>`
- `discard_potion:<potion_idx>`
- card-select actions such as
  `select:<task>:<source_zone>:<card_unique_id-or-indexes>` when
  `InputState::CARD_SELECT` is active

The binding should expose both a stable id and the original simulator command
string accepted by `BattleSimulator::takeAction`. Search can use the stable id;
the adapter stores command/index metadata for auditability but applies the
freshly matched current payload, avoiding stale hand or potion indexes from an
older decision.

Card-select actions need task metadata, not just a hand card id. Some upstream
choices refer to hand, discard pile, exhaust pile, draw pile, generated Codex or
Discovery options, and multi-select index lists such as Gamble or exhaust-many
tasks. The payload should include at least `task`, `source_zone`,
`selected_indexes`, `card_unique_ids` when available, and `command`.

Potion discard is represented locally by `ActionType.DISCARD_POTION`; live
adapters should map it to the upstream `discard <potion_idx>` command.

## Snapshot Fields

The minimum useful `observe()` payload:

- player: HP, max HP, block, energy, powers, relic counters relevant in combat,
  potions
- piles: hand, draw, discard, exhaust, with card id, unique id, current cost,
  upgrades, temporary flags, and card type
- monsters: index/stable id, monster id/name, HP, max HP, block, powers,
  move/intent id, estimated intent damage, targetable, alive
- combat: input state, turn, floor, encounter, outcome, action history if
  available
- metrics: initial HP, current HP loss, turns taken, cards played, damage dealt,
  potions used, undefined-behavior flags if present. Some counters are not
  native `BattleContext` fields today, so the adapter should own initial
  snapshots and per-action counters or upstream should add instrumentation.

## Validation Path

1. Add upstream pybind smoke tests that initialize a one-combat Ironclad state,
   call `observe`, enumerate actions, clone, apply `end_turn`, and verify the
   original clone is unchanged.
2. Add local payload-adapter conformance tests behind `STS_LIGHTSPEED_MODULE_DIR`
   that convert upstream payloads into `CombatObservation`, `CombatAction`, and
   `CombatMetrics`, and assert the wrapper satisfies `CombatBackend`.
3. Run the existing `CombatFixtureSimulator` tests unchanged to preserve local
   determinism.
4. Run fixed-seed live comparisons with `combat_experiment.py` once live
   metrics are available.

## Recommendation

Do not port the full C++ combat engine into this repository. Keep
`sts_lightspeed` as the mechanics source of truth, add a focused upstream
pybind combat adapter, and keep this repository's Python code responsible for
planning, value/policy models, trajectories, and evaluation.
