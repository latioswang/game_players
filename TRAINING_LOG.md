# 2048 Training Log

This file records training observations, command changes, and experiment results so we can distinguish real learning effects from mixed-run chart artifacts.

## 2026-05-26 07:50 CST

### Parallel Expectimax Evaluation

Goal: use the Apple M3 CPU hardware more effectively. The machine has 8 CPU cores and Numba reports 8 available threads with the `workqueue` threading layer.

Code changes:

- Added `eval --workers`, accepting a positive integer or `auto`.
- Changed evaluation to generate one deterministic seed per game, so single-worker and multi-worker runs evaluate the same games.
- Parallelized independent games with a thread pool. The hot recursive Expectimax functions are Numba-compiled with `nogil=True`, allowing game searches to run concurrently.
- Set `--workers auto` as the default for `eval`; use `--workers 1` for single-worker baselines.

Validation:

```bash
venv/bin/pytest -q
```

Result: `11 passed in 1.79s`.

Benchmark:

```bash
venv/bin/python -m game_players.cli eval --games 100 --seed 1 --depth 2 --workers 1
venv/bin/python -m game_players.cli eval --games 100 --seed 1 --depth 2 --workers auto
```

| Workers | Games | Depth | Avg Score | Best Score | Best Tile | Tile Counts | Wins 2048 | Win Rate | Avg Moves | Total Seconds | Avg Seconds/Game |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 100 | 2 | 24492.2 | 81244 | 4096 | `256:3,512:18,1024:29,2048:39,4096:11` | 50 | 0.500 | 1289.2 | 37.201 | 0.372 |
| 8 (`auto`) | 100 | 2 | 24492.2 | 81244 | 4096 | `256:3,512:18,1024:29,2048:39,4096:11` | 50 | 0.500 | 1289.2 | 8.535 | 0.085 |

Speedup: `37.201 / 8.535 = 4.36x`. The score rows match exactly, confirming worker count changes runtime without changing the evaluated game set.

## 2026-05-26 07:36 CST

### Fast Heuristic Expectimax Replacement

Replaced the n-tuple/DQN public agent stack with the transcript winner: heuristic Expectimax. There is no training phase now.

Code changes:

- Added `game_players/expectimax_agent.py` with packed boards, precomputed row transitions, Numba-compiled move/heuristic/search kernels, exact random-spawn chance nodes, and summary metrics.
- Replaced `game_players/cli.py` with `eval` and `play` only.
- Removed the old n-tuple agent, DQN agent, device selector, autoresearch script, and n-tuple program notes.
- Updated `README.md` around the no-training Expectimax workflow.
- Added `numpy`, `numba`, and `glog` as runtime dependencies in `pyproject.toml`.

Validation:

```bash
venv/bin/pytest -q
```

Result: `9 passed in 1.23s`.

Smoke benchmark:

```bash
venv/bin/python -m game_players.cli eval --games 5 --seed 1 --depth 2
```

Result: `avg_score=29401.6`, `best_score=71760`, `avg_max_tile=2048.0`, `best_max_tile=4096`, `tile_counts=1024:2,2048:2,4096:1`, `wins_2048=3`, `win_rate_2048=0.600`, `avg_moves=1472.6`, `avg_seconds_per_game=0.403`, `total_seconds=2.016`.

Full completed benchmark:

```bash
venv/bin/python -m game_players.cli eval --games 100 --seed 1 --depth 2
```

Result: `avg_score=25491.2`, `best_score=71760`, `avg_max_tile=1720.3`, `best_max_tile=4096`, `tile_counts=512:10,1024:33,2048:49,4096:8`, `wins_2048=57`, `win_rate_2048=0.570`, `avg_moves=1339.3`, `avg_seconds_per_game=0.629`, `total_seconds=62.914`.

Depth-3 note:

```bash
venv/bin/python -m game_players.cli eval --games 100 --seed 1 --depth 3
```

Stopped twice after exceeding practical runtime, including one run stopped at about `2:48` with no completed output. Exact depth-3 chance expansion is too expensive for the current full-game benchmark, even with Numba recursion. The CLI default was set to depth `2` so the documented 100-game benchmark completes in about one minute on this machine. Depth `3` remains accepted for small experiments.

## 2026-05-26 07:15 CST

### Expectimax Lookahead Policy

Reference: the requested YouTube comparison emphasized search methods for 2048, especially expectimax-style play over random tile spawns. The first local implementation keeps the existing n-tuple TD evaluator and adds expectimax only at evaluation time.

Code changes:

- `README.md`: added the improvement roadmap and the `--lookahead-depth` evaluation command.
- `game_players/ntuple_agent.py`: added expectimax action scoring, exact random-spawn enumeration, per-decision memoization, and shared action-outcome generation.
- `game_players/cli.py`: added `eval --lookahead-depth`; depth `1` preserves the previous greedy n-tuple policy.
- `tests/test_ntuple_agent.py`: covered spawn probabilities and a fixture where depth-2 expectimax prefers a lower-immediate-reward move because of future spawn value.

Validation:

```bash
venv/bin/pytest -q
```

Result: `9 passed in 0.03s`.

Evaluation checkpoint:

```bash
models/2048-agent.best.pkl
```

Same-seed smoke comparison:

| Command | Games | Seed | Depth | Avg Score | Best Score | Avg Max Tile | Best Max Tile | Tile Counts | Wall Time |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `venv/bin/python -m game_players.cli eval --model models/2048-agent.best.pkl --agent ntuple --games 3 --seed 1 --lookahead-depth 1` | 3 | 1 | 1 | 12416.0 | 15668 | 853.3 | 1024 | `512:1,1024:2` | 14.463s |
| `venv/bin/python -m game_players.cli eval --model models/2048-agent.best.pkl --agent ntuple --games 3 --seed 1 --lookahead-depth 2` | 3 | 1 | 2 | 14946.7 | 16792 | 1024.0 | 1024 | `1024:3` | 52.893s |

Additional baseline:

```bash
venv/bin/python -m game_players.cli eval --model models/2048-agent.best.pkl --agent ntuple --games 30 --seed 1 --lookahead-depth 1
```

Result: `avg_score=6948.5`, `best_score=15668`, `avg_max_tile=460.8`, `best_max_tile=1024`, `tile_counts=128:2,256:11,512:13,1024:4`.

Attempted matching 30-game depth-2 eval, but stopped it after it exceeded two minutes without finishing. The three-game result suggests expectimax can improve decision quality, but the current dict-backed n-tuple value lookup is too slow for routine depth-2 or depth-3 benchmarks. Next implementation target should be faster n-tuple storage or a bounded/pruned expectimax mode before relying on larger search evaluations.

## 2026-05-17 04:45 CST

### Four-Way Symmetry and Epsilon Comparison

All runs used fresh n-tuple starts with:

```bash
--fresh --agent ntuple --episodes 3000 --eval-every 500 --eval-games 30 --save-every 0 --seed 0
```

The two fixed-epsilon partial files `models/compare-sym-off-fixed-epsilon.csv` and `models/compare-sym-on-fixed-epsilon.csv` were not used for the final decision because the main `.pkl` checkpoints were missing/incomplete and one CSV contained duplicate episode-1/header rows. Fresh fixed-epsilon runs were written to `models/compare-final-*.csv`.

| Option | Metrics CSV | Best eval_avg_score | Best Episode | Final eval_avg_score | Final Best Tile | Final pct_512 | Final pct_1024 | Final pct_2048 | Final Tile Counts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Symmetry off + epsilon decay default | `models/compare-sym-off-decay.csv` | 5753.2 | 2500 | 5491.6 | 1024 | 0.433 | 0.033 | 0.000 | `128:2,256:15,512:12,1024:1` |
| Symmetry on + epsilon decay default | `models/compare-sym-on-decay.csv` | 5173.7 | 3000 | 5173.7 | 1024 | 0.567 | 0.033 | 0.000 | `128:1,256:12,512:16,1024:1` |
| Symmetry off + fixed epsilon (`--epsilon-decay 1.0`) | `models/compare-final-sym-off-fixed-epsilon.csv` | 5992.1 | 3000 | 5992.1 | 1024 | 0.600 | 0.100 | 0.000 | `128:2,256:10,512:15,1024:3` |
| Symmetry on + fixed epsilon (`--epsilon-decay 1.0`) | `models/compare-final-sym-on-fixed-epsilon.csv` | 4843.7 | 3000 | 4843.7 | 1024 | 0.400 | 0.033 | 0.000 | `128:6,256:12,512:11,1024:1` |

Decision:

- Symmetry was not clearly helpful. Both symmetry-on runs underperformed their symmetry-off counterpart, and symmetry-on was materially slower wall-clock because each value/update path expands across eight board transforms.
- Fixed epsilon beat epsilon decay in the best symmetry-off comparison: `5992.1` vs `5753.2` best eval average, and `0.100` vs `0.033` final pct_1024.
- Fresh n-tuple defaults are now symmetry off and `epsilon_decay=1.0`.
- Resume behavior still preserves saved `alpha`, `epsilon`, decay settings, and symmetry mode unless explicit flags override them.

Code changes:

- `game_players/ntuple_agent.py`: default `use_symmetry=False`, default `epsilon_decay=1.0`, named default constants.
- `game_players/cli.py`: fresh auto symmetry uses the n-tuple default, fresh epsilon decay uses the n-tuple default, and eval scheduling uses global `agent.episodes_trained` so continued runs evaluate at clean global episode multiples.
- `tests/test_ntuple_agent.py`: covered fresh defaults and resume preservation.
- `README.md`: documented the new n-tuple defaults.

Caveat: this is one seed with 30 eval games per checkpoint, so it is enough to set pragmatic local defaults but not a statistical proof of long-run superiority.

## 2026-05-17 04:10 CST

Code baseline: `9dfcf8b` plus local uncommitted diagnostics and symmetry-resume fixes.

### Context

The `models/score-tiles.png` chart showed a sharp score drop after the symmetry/decay/best-checkpoint change. The backing CSV, `models/training-metrics.csv`, was mixed across several training regimes:

- Early rows used the old schema without `alpha`, `weight_delta_l2`, or tile percentage columns.
- Rows before roughly episode `22004` were from the pre-symmetry run and were around `9k-11.8k` eval average.
- Rows after roughly episode `22004` resumed with symmetry enabled and dropped to roughly `5k-7.6k`.
- The CSV also contains several fresh/restarted runs, so line plots must break when episode numbers reset.

### Findings

1. The plot was misleading because it connected separate runs into one line.
2. Enabling symmetry on an already-trained non-symmetric checkpoint caused a real regression.
3. Resume was resetting `alpha` and `epsilon` to defaults, which also distorted continued training.

Quick checkpoint comparison, same saved weights:

| Checkpoint | Symmetry | Eval Games | Avg Score | Best Tile | Tile Counts |
|---|---:|---:|---:|---:|---|
| `models/2048-agent.pkl` | on | 100 | 7162.5 | 2048 | `{512: 49, 1024: 19, 256: 28, 128: 3, 2048: 1}` |
| `models/2048-agent.pkl` | off | 100 | 9821.8 | 1024 | `{512: 50, 1024: 35, 256: 15}` |
| `models/2048-agent.best.pkl` | on | 100 | 7184.0 | 2048 | `{256: 30, 512: 51, 1024: 15, 2048: 2, 128: 2}` |
| `models/2048-agent.best.pkl` | off | 100 | 9341.0 | 1024 | `{512: 64, 1024: 28, 256: 7, 128: 1}` |

Conclusion: for existing checkpoints, recover with:

```bash
venv/bin/python -m game_players.cli train --episodes 100000 --symmetry off
```

### Code Fixes Made Locally

- Plotting now breaks lines when episode numbers reset.
- Plotting can derive `pct_512`, `pct_1024`, and `pct_2048` from old `eval_tile_counts` rows.
- Resumed n-tuple checkpoints preserve saved `alpha`, `epsilon`, decay settings, and symmetry mode unless explicit flags override them.
- Fresh n-tuple runs now default to symmetry off after the four-way comparison below.
- Diagnostic metrics added:
  - `weight_l2`
  - `weight_abs_mean`
  - `weight_delta_l2`
  - `td_error_abs_avg`
  - `td_updates`
  - `pct_512`
  - `pct_1024`
  - `pct_2048`

### Cold-Start Comparison In Progress

Goal: compare fresh symmetric vs non-symmetric n-tuple training, and later compare epsilon decay vs fixed epsilon.

Common settings:

```bash
--fresh --agent ntuple --episodes 3000 --eval-every 500 --eval-games 30 --save-every 0
```

#### Symmetry Off + Decay

Command:

```bash
venv/bin/python -m game_players.cli train \
  --fresh --agent ntuple --symmetry off \
  --episodes 3000 --eval-every 500 --eval-games 30 --save-every 0 \
  --model models/compare-sym-off-decay.pkl \
  --metrics models/compare-sym-off-decay.csv
```

Results:

| Episode | Eval Avg Score | Best Tile | Epsilon | Alpha | Weights | Tile Counts |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 3182.0 | 512 | 0.0500 | 0.01000 | 80953 | `32:1,128:10,256:13,512:6` |
| 500 | 3144.4 | 512 | 0.0476 | 0.00975 | 346573 | `128:8,256:18,512:4` |
| 1000 | 3126.9 | 512 | 0.0452 | 0.00951 | 496101 | `64:1,128:7,256:18,512:4` |
| 1500 | 4208.3 | 512 | 0.0430 | 0.00928 | 616453 | `128:4,256:17,512:9` |
| 2000 | 5154.8 | 1024 | 0.0409 | 0.00905 | 734903 | `128:2,256:15,512:12,1024:1` |
| 2500 | 5753.2 | 1024 | 0.0389 | 0.00882 | 844529 | `256:16,512:12,1024:2` |
| 3000 | 5491.6 | 1024 | 0.0370 | 0.00861 | 939356 | `128:2,256:15,512:12,1024:1` |

Best checkpoint saved at episode 2500.

#### Symmetry On + Decay

Command:

```bash
venv/bin/python -m game_players.cli train \
  --fresh --agent ntuple --symmetry on \
  --episodes 3000 --eval-every 500 --eval-games 30 --save-every 0 \
  --model models/compare-sym-on-decay.pkl \
  --metrics models/compare-sym-on-decay.csv
```

Partial results:

| Episode | Eval Avg Score | Best Tile | Epsilon | Alpha | Weights | Tile Counts |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1996.4 | 256 | 0.0500 | 0.01000 | 322857 | `64:3,128:13,256:14` |
| 500 | 4201.6 | 512 | 0.0476 | 0.00975 | 1664261 | `128:6,256:12,512:12` |
| 1000 | 3368.1 | 512 | 0.0452 | 0.00951 | 2163787 | `64:5,128:8,256:9,512:8` |
| 1500 | 5054.3 | 1024 | 0.0430 | 0.00928 | 2523007 | `128:2,256:15,512:12,1024:1` |
| 2000 | 4807.6 | 1024 | 0.0409 | 0.00905 | 2840861 | `128:5,256:13,512:10,1024:2` |
| 2500 | 4725.2 | 1024 | 0.0389 | 0.00882 | 3102953 | `64:3,128:4,256:12,512:9,1024:2` |
| 3000 | 5173.7 | 1024 | 0.0370 | 0.00861 | 3286623 | `128:1,256:12,512:16,1024:1` |

Status: completed. Best checkpoint saved at episode 3000.

Interpretation at 3000 episodes:

- Symmetry on eventually caught up close to symmetry off, but it was much slower wall-clock and still slightly behind the symmetry-off best result at 3000 episodes.
- Symmetry-off best eval average: `5753.2` at episode 2500.
- Symmetry-on best eval average: `5173.7` at episode 3000.
- This is not enough evidence to prove symmetry is worse long-term, but it is enough evidence that symmetry is not an immediate win and should not be forced onto existing runs.

### Working Hypotheses

1. Symmetry sharing may be implemented in a way that over-averages directional strategy. 2048 is symmetric under board transforms mathematically, but the learned strategy may benefit from orientation-specific specialization unless action transforms are handled more carefully.
2. The update normalization for symmetry may be too conservative: each transformed feature receives `alpha / (patterns * symmetries)`, which may slow learning substantially.
3. The wider 6-tuple pattern expansion plus symmetry creates many more active keys, so early learning can look worse even if it eventually catches up.
4. Epsilon decay may be too aggressive for the first few thousand episodes, but the symmetry-off run improved despite decay, so it is not the primary cause of the observed drop.

### Next Experiments

Run after the symmetric 3000-episode run finishes:

```bash
venv/bin/python -m game_players.cli train \
  --fresh --agent ntuple --symmetry off \
  --episodes 3000 --eval-every 500 --eval-games 30 --save-every 0 \
  --epsilon-decay 1.0 \
  --model models/compare-sym-off-fixed-epsilon.pkl \
  --metrics models/compare-sym-off-fixed-epsilon.csv

venv/bin/python -m game_players.cli train \
  --fresh --agent ntuple --symmetry on \
  --episodes 3000 --eval-every 500 --eval-games 30 --save-every 0 \
  --epsilon-decay 1.0 \
  --model models/compare-sym-on-fixed-epsilon.pkl \
  --metrics models/compare-sym-on-fixed-epsilon.csv
```

Also consider testing symmetry update scaling:

- Current: `alpha * error / (patterns * symmetries)`
- Candidate: `alpha * error / patterns`, applied to all symmetries
- Candidate: canonicalize board to one symmetry instead of updating all 8
