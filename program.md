# game_players autoresearch

This is the research protocol for improving the 2048 n-tuple learner.

## Scope

The target is `game_players/ntuple_agent.py`. Optimize the n-tuple agent only.

Read these files before changing anything:

- `README.md` for project behavior and commands.
- `game_players/game2048.py` for the fixed 2048 environment.
- `game_players/ntuple_agent.py` for the learner.
- `game_players/cli.py` for the training and evaluation harness.
- `game_players/metrics.py` for metric output.
- `tests/test_game2048.py` and `tests/test_ntuple_agent.py` for guardrails.

Do not modify the game rules, scoring, random tile spawning, metric definition, or tests to make a result look better. Those are the benchmark.

## Metric

The primary metric is `eval_avg_score`; higher is better. Use `eval_best_tile` and `eval_tile_counts` as secondary tie-breakers. If two runs are statistically similar, keep the simpler implementation.

Because 2048 is stochastic, treat small single-run changes as weak evidence. Re-run promising ideas with more evaluation games or more seeds before considering them real.

## Baseline

The first experiment in a run should be a baseline with the current code:

```bash
python scripts/autoresearch_2048.py --tag baseline --description baseline
```

The runner trains a fresh n-tuple checkpoint, captures logs, reads the final metrics row, and appends a line to `results.tsv`.

## Experiment Loop

Use a branch named like `autoresearch/may17-ntuple` for longer runs.

Loop:

1. Check git status and current best row in `results.tsv`.
2. Change only `game_players/ntuple_agent.py` unless the experiment requires a tiny CLI option.
3. Run tests:

   ```bash
   pytest
   ```

4. Run one candidate:

   ```bash
   python scripts/autoresearch_2048.py --tag short-description --description "what changed"
   ```

5. Compare the row appended to `results.tsv`.
6. Keep the commit only if it improves `eval_avg_score` enough to justify its complexity.
7. Discard or revise regressions.

## Good Ideas To Try

- Pattern sets: snake paths, corner-heavy patterns, larger rectangles, fewer redundant patterns.
- Learning update normalization in `update_value`.
- Alpha and epsilon schedules.
- Value or reward scaling.
- Tie-breaking between equal action values.
- Small TD update variants that preserve afterstate learning.

Avoid adding dependencies for n-tuple experiments. Keep the code understandable and fast enough for many short trials.
