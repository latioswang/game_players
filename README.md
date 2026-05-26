# game_players

A fast 2048 player using heuristic Expectimax. This version focuses on the
best-performing approach from the referenced video: search over future swipes,
exact random tile spawns, and a handcrafted board heuristic. There is no
training phase.

## Set Up

```bash
python3 -m venv venv
venv/bin/python -m pip install -e ".[dev,plot]"
```

Runtime dependencies are `numpy`, `numba`, and `glog`.

## Run It

Evaluate the player over many games:

```bash
python -m game_players.cli eval --games 100 --seed 1 --depth 2
```

Evaluation uses all CPU cores by default. Pin the run to one worker when you
need a single-threaded baseline:

```bash
python -m game_players.cli eval --games 100 --seed 1 --depth 2 --workers 1
```

Run one game and print the final board:

```bash
python -m game_players.cli play --seed 1 --depth 2 --show-board
```

Depth controls how many player moves Expectimax searches. Valid depths are
`1` through `5`; depth `2` is the default because exact depth `3` expands a
large chance tree and is much slower.

## How It Works

1. The board stores tile exponents, so `1` means `2`, `2` means `4`, and so on.
2. Row move transitions are precomputed for all `16^4` possible rows.
3. Numba-compiled helpers apply moves and score leaf boards quickly.
4. Expectimax max nodes choose the best swipe.
5. Chance nodes enumerate every empty-cell spawn with `2` probability `0.9` and `4` probability `0.1`.
6. Leaf boards are scored with empty-cell, monotonicity, smoothness, merge-potential, corner-max, and large-tile bonuses.

Evaluation reports:

- `games`
- `avg_score`
- `best_score`
- `avg_max_tile`
- `best_max_tile`
- `tile_counts`
- `wins_2048`
- `win_rate_2048`
- `avg_moves`
- `avg_seconds_per_game`
- `total_seconds`

The `--workers` option accepts a positive integer or `auto`. Parallel runs use
one deterministic seed per game, so changing worker count changes runtime
without changing the evaluated games.

## Files

- `game_players/game2048.py`: reference 2048 game rules and rendering.
- `game_players/expectimax_agent.py`: packed-board engine and Expectimax player.
- `game_players/cli.py`: `eval` and `play` commands.
- `tests/test_game2048.py`: reference mechanics tests.
- `tests/test_expectimax_agent.py`: Expectimax and heuristic tests.
