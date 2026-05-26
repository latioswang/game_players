# AGENTS.md

## Repository expectations

- This is a Python package for reinforcement-learning game players.
- For each new Codex implementation or fix chat session, create a fresh Git worktree before modifying repository files. Use a dedicated `codex/...` branch/worktree, work only inside that worktree, and leave the primary checkout untouched unless the user explicitly asks otherwise.
- Install local development dependencies with `python -m pip install -e ".[dev]"`.
- Run the test suite with `pytest`.
- Keep generated models, plots, metrics, and large experiment outputs out of commits unless explicitly requested.
- Prefer deterministic tests. Use fixed seeds when testing training, move selection, or stochastic game behavior.

## Review guidelines

- Treat incorrect 2048 move mechanics, scoring, terminal-state handling, or afterstate evaluation as high-priority issues.
- Check that checkpoint loading and model serialization changes remain backward compatible with existing saved agents.
- Check CLI changes for clear defaults, resumable training behavior, and safe handling of interrupt/save paths.
- Require tests for changes to game mechanics, n-tuple feature extraction, TD updates, checkpoint migration, or CLI argument behavior.
- Do not flag style-only nits unless they hide a behavioral risk.
