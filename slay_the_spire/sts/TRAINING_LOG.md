# Slay the Spire Combat Training Log

## 2026-05-27

- Added the combat-only long-run implementation plan in `COMBAT_PLAN.md`.
- Scoped the work away from deck building, route selection, shops, rests,
  upgrades, and reward picking so the combat policy effort can proceed
  independently.
- Established the initial target path: expose a combat API, build a search
  player, collect trajectories, train value and policy models, combine them in
  hybrid search, and evaluate against deterministic baselines.
