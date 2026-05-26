# Slay-the-Spire Baseline Runner

This folder contains the Slay-the-Spire-style experiment hooks. It adds a thin
Python runner for the external `gamerpuppy/sts_lightspeed` simulator. The
simulator is not vendored into this repository; build it separately and point
the runner at the compiled Python module directory.

## Build sts_lightspeed

The upstream project currently builds its Python module with bundled
`pybind11` 2.7.1. On this machine, that binding fails against Python 3.12 but
works against system Python 3.9:

```bash
git clone --recursive https://github.com/gamerpuppy/sts_lightspeed.git /tmp/codex-sts-sim-eval/sts_lightspeed
cd /tmp/codex-sts-sim-eval/sts_lightspeed
cmake -S . -B build-py39 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPYTHON_EXECUTABLE=/usr/bin/python3
cmake --build build-py39 --target slaythespire --config Release -j 8
```

## Test Python Simulator Setup

From this repository root, verify that Python can import and run the simulator:

```bash
PYTHONPATH=/tmp/codex-sts-sim-eval/sts_lightspeed/build-py39 \
/usr/bin/python3 - <<'PY'
import slaythespire as sts

gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 1, 0)
agent = sts.Agent()
agent.simulation_count_base = 1
agent.playout(gc)

print("seed", sts.get_seed_str(1))
print("outcome", gc.outcome)
print("floor", gc.floor_num)
print("hp", gc.cur_hp)
print("deck_size", len(gc.deck))
PY
```

Expected shape:

```text
seed 1
outcome GameOutcome.PLAYER_LOSS
floor 2
hp 0
deck_size 12
```

## Run Baselines

```bash
/usr/bin/python3 slay_the_spire/sts/sts_lightspeed_baseline.py \
  --module-dir /tmp/codex-sts-sim-eval/sts_lightspeed/build-py39 \
  --games 100 \
  --start-seed 1 \
  --simulation-count 5 \
  --deck-policy heuristic
```

The script evaluates sequential seeds using Ironclad at ascension 0. It supports
four deck policies:

- `agent`: let the upstream `sts_lightspeed` agent run the whole playout. The
  current binding does not expose the agent's card-reward choices separately, so
  reward pick/skip metrics are reported as unavailable for this policy.
- `skip`: pause at card rewards and skip every card.
- `first`: pause at card rewards and take the first offered card.
- `heuristic`: pause at card rewards and score cards with a small Python rule
  table.

Use `--jsonl path/to/file.jsonl` to write per-run decks and card-reward
decisions for later analysis.

## Combat-Only Policy Work

The combat-only long-run plan is in `slay_the_spire/sts/COMBAT_PLAN.md`. It is
intentionally separate from deck building, route selection, shops, rests,
upgrades, and reward picking.

Current implementation checkpoints:

- `combat_api.py`: serializable combat state/action contracts plus a placeholder
  `sts_lightspeed` adapter that raises until action-level bindings exist.
- `combat_search.py`: deterministic beam-search skeleton over an abstract combat
  simulator.
- `combat_trajectory.py`: JSONL-safe combat trajectory records for future data
  collection.
- `combat_value.py` and `combat_policy.py`: handcrafted value scoring and policy
  prior/ranking helpers for local adapters and fixtures.
- `combat_eval.py`: combat-only metric aggregation for deterministic seed-suite
  comparisons once live combat metrics are available.
- `combat_fixture.py`: deterministic local Strike/Defend/Bash-style combat
  fixture with cloning and JSON state round trips.
- `combat_training.py`: no-dependency sparse linear value-function training from
  trajectory value records.
- `combat_hybrid.py`: policy-pruned, value-guided beam-search orchestration.
- `combat_experiment.py`: fixed-seed fixture policy comparisons, regression
  tracking, and optional JSONL trajectory output.
- `STS_LIGHTSPEED_BINDING_PLAN.md`: concrete upstream binding checklist for the
  remaining live simulator blocker.

Recommended Sequence checkpoint:

- Completed in local fixture/protocol space: state/action contracts,
  serialization helpers, cloning tests, beam-search planning, trajectory
  records, handcrafted value scoring, policy ranking, and metric aggregation.
- Still blocked in live `sts_lightspeed`: legal combat-action enumeration,
  single-action application, action-level combat pause/resume, cheap live state
  cloning, and combat-only outcome reporting are not exposed by the current
  Python binding.
- Added the local payload adapter in `combat_sts_adapter.py`. It converts
  future pybind `CombatHandle.observe()`, `legal_actions()`, `apply()`,
  `clone()`, and `metrics()` dictionaries into the existing `CombatBackend`
  protocol, and `CombatBackendSimulator` lets search/model callers use a live
  backend through the existing simulator methods.
- Next live work item: add the upstream pybind `CombatHandle` surface so the
  adapter can run fixed-seed simulator combats instead of fake payload tests.

Live conformance checks should stay environment-gated until the external module
is available. Use `STS_LIGHTSPEED_MODULE_DIR` for pytest checks that import the
compiled pybind module and skip cleanly while future combat entry points are
absent. Set `STS_LIGHTSPEED_COMBAT_CONFORMANCE=1` to require those future entry
points. Fake-handle tests cover payload conversion to
`CombatObservation`/`CombatAction`/`CombatMetrics`, clone independence, and
stale-action rejection until upstream exposes real combat payloads.

Run the local combat tests from the repository root:

```bash
python -m pytest slay_the_spire/sts/test_combat_*.py -q
```

These tests use local fake, dict-backed, and protocol adapters. They do not
exercise live `sts_lightspeed` combat action bindings, because the current
Python binding still does not expose legal combat actions or single-action
application. Passing these tests means the local fixture/planner/model layer is
ready for a future live adapter; it does not mean live combat bindings exist.

## Basic Baseline Strategy Tests

Run the simulator smoke test before larger experiments:

```bash
/usr/bin/python3 slay_the_spire/sts/run_lightspeed_smoke.py \
  --module-dir /tmp/codex-sts-sim-eval/sts_lightspeed/build-py39 \
  --games 5 \
  --start-seed 1 \
  --simulation-count 1
```

This script actually imports the compiled `slaythespire` extension, performs a
direct simulator playout, then runs three baseline strategy paths:

- `agent`: one call into the upstream `sts_lightspeed` agent.
- `skip`: Python pauses at card rewards and skips them.
- `heuristic`: Python pauses at card rewards, scores the choices, and records
  every decision.

You can also run the strategy commands individually:

The same smoke test can also be collected by `pytest`; it is skipped unless the
external simulator module path is provided. Use the same Python interpreter that
was used to build the `slaythespire` extension when running the full external
simulator test through pytest:

```bash
STS_LIGHTSPEED_MODULE_DIR=/tmp/codex-sts-sim-eval/sts_lightspeed/build-py39 \
python -m pytest slay_the_spire/sts/test_lightspeed_external.py -q
```

```bash
/usr/bin/python3 slay_the_spire/sts/sts_lightspeed_baseline.py \
  --module-dir /tmp/codex-sts-sim-eval/sts_lightspeed/build-py39 \
  --games 5 \
  --start-seed 1 \
  --simulation-count 1 \
  --deck-policy agent

/usr/bin/python3 slay_the_spire/sts/sts_lightspeed_baseline.py \
  --module-dir /tmp/codex-sts-sim-eval/sts_lightspeed/build-py39 \
  --games 5 \
  --start-seed 1 \
  --simulation-count 1 \
  --deck-policy skip

/usr/bin/python3 slay_the_spire/sts/sts_lightspeed_baseline.py \
  --module-dir /tmp/codex-sts-sim-eval/sts_lightspeed/build-py39 \
  --games 5 \
  --start-seed 1 \
  --simulation-count 1 \
  --deck-policy heuristic \
  --show-trials
```

For the `heuristic --show-trials` smoke test, the output should include a
summary followed by JSON rows containing `reward_options`, `picked`, final
`deck`, and final `floor`.

## Trial Results

All runs used seeds `1..100`, ascension `0`.

| Policy | Simulation Count | Games | Wins | Avg Floor | Best Floor | Avg Picked | Avg Skipped |
|---|---:|---:|---:|---:|---:|---:|---:|
| `agent` | 1 | 100 | 0 | 5.55 | 16 | unavailable | unavailable |
| `heuristic` | 1 | 100 | 0 | 5.90 | 16 | 3.13 | 0.01 |
| `agent` | 5 | 100 | 0 | 9.75 | 19 | unavailable | unavailable |
| `heuristic` | 5 | 100 | 0 | 9.29 | 21 | 5.03 | 0.06 |

Interpretation:

- The runner is wired correctly and can produce deterministic baseline metrics.
- At this weak baseline level, increasing combat/search simulations matters
  more than the simple Python card-pick heuristic.
- The heuristic is intentionally simple; it is useful as a first training
  baseline and data-collection hook, not as a strong deckbuilder.

## Does It Support Deck Building?

Yes, but not as a polished Gym-style API. The Python binding exposes enough
deck-building hooks to build an adapter:

- `GameContext.deck` exposes the current deck.
- `GameContext.get_card_reward()` returns card reward options when paused on a
  reward screen.
- `GameContext.pick_reward_card(card)` adds the selected card.
- `GameContext.skip_reward_cards()` skips the reward.
- `GameContext.obtain_card(card)` and `GameContext.remove_card(index)` allow
  direct deck mutation for experiments.
- `Agent.pause_on_card_reward = True` lets Python take over card reward
  decisions between simulator playout chunks.

The simulator also models out-of-combat run flow according to its README and
source structure, including map, rewards, shops, relics, events, and combat.
The current binding exposes card rewards most directly; additional Python
bindings may be needed for full route/shop/rest policy learning.

## Deck-Building Approach

Use a staged approach:

1. Keep `sts_lightspeed` as the mechanics engine and add a thin local adapter
   rather than copying simulator code into this repo.
2. Start with card-reward policy learning because the binding already exposes
   that decision cleanly.
3. Use the upstream search agent for combat at first, so early deck-building
   experiments do not need to solve card sequencing and deck construction at
   the same time.
4. Collect JSONL trajectories containing seed, floor, deck, reward options,
   chosen card, final floor, and outcome.
5. Train a supervised reward picker from strong heuristic/search data, then
   fine-tune against held-out seeds with RL.
6. Add bindings for map routing, shops, rests/upgrades, removals, and potions
   once card rewards are measurable.

The first model should probably be a small contextual card picker:

```text
(deck tokens, relic tokens, floor/act/boss/hp/gold, reward card tokens)
  -> score each offered card plus skip
```

That keeps the action space small and lets combat remain controlled by
`sts_lightspeed` search while deck-building improves.
