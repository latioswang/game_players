# sts_lightspeed Baseline Runner

This worktree adds a thin Python runner for the external
`gamerpuppy/sts_lightspeed` simulator. The simulator is not vendored into this
repository; build it separately and point the runner at the compiled Python
module directory.

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

## Run Baselines

```bash
/usr/bin/python3 scripts/sts_lightspeed_baseline.py \
  --module-dir /tmp/codex-sts-sim-eval/sts_lightspeed/build-py39 \
  --games 100 \
  --start-seed 1 \
  --simulation-count 5 \
  --deck-policy heuristic
```

The script evaluates sequential seeds using Ironclad at ascension 0. It supports
four deck policies:

- `agent`: let the upstream `sts_lightspeed` agent run the whole playout.
- `skip`: pause at card rewards and skip every card.
- `first`: pause at card rewards and take the first offered card.
- `heuristic`: pause at card rewards and score cards with a small Python rule
  table.

Use `--jsonl path/to/file.jsonl` to write per-run decks and card-reward
decisions for later analysis.

## Trial Results

All runs used seeds `1..100`, ascension `0`.

| Policy | Simulation Count | Games | Wins | Avg Floor | Best Floor | Avg Picked | Avg Skipped |
|---|---:|---:|---:|---:|---:|---:|---:|
| `agent` | 1 | 100 | 0 | 5.55 | 16 | n/a | n/a |
| `heuristic` | 1 | 100 | 0 | 5.90 | 16 | 3.13 | 0.01 |
| `agent` | 5 | 100 | 0 | 9.75 | 19 | n/a | n/a |
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

