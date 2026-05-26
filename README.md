# Game Players

Monorepo for game-specific reinforcement-learning and search players. Each game
AI owns its package metadata, source code, tests, and local documentation in its
own subdirectory.

## Projects

| Project | Status | Description |
|---|---|---|
| [`2048`](2048/) | Active | Heuristic Expectimax player for 2048. |
| [`slay_the_spire`](slay_the_spire/) | Planned | Slay the Spire AI workspace. |

## Working With A Project

Treat the repository root as the working directory for commands and Python
module execution. Game projects should behave like packages inside a single
monorepo, similar to Google's monorepo style, so examples and automation should
start from the repo root rather than requiring `cd` into a subproject.

```bash
python -m pip install -e "2048[dev]"
python -m pytest 2048/tests
python -m game_players.cli eval --games 100 --seed 1 --depth 2
```

Generated models, plots, metrics, and large experiment outputs should stay out
of commits unless they are explicitly requested.
