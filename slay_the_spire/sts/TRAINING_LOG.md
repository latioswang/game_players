# Slay the Spire Combat Training Log

## 2026-05-27

- Added the combat-only long-run implementation plan in `COMBAT_PLAN.md`.
- Scoped the work away from deck building, route selection, shops, rests,
  upgrades, and reward picking so the combat policy effort can proceed
  independently.
- Established the initial target path: expose a combat API, build a search
  player, collect trajectories, train value and policy models, combine them in
  hybrid search, and evaluate against deterministic baselines.
- Documented the combat-only plan location and local checkpoint modules in
  `README.md`.
- Added the repository-root local combat test command:
  `python -m pytest slay_the_spire/sts/test_combat_*.py -q`.
- Noted that current local tests do not claim live `sts_lightspeed` combat
  action bindings; those remain pending behind the adapter boundary.
- Added combat API/state contracts, fixture-only legal action helpers, an
  unsupported `sts_lightspeed` combat adapter boundary, beam search, trajectory
  JSONL records, handcrafted value scoring, policy prior ranking, and combat
  evaluation aggregation.
- Addressed integration-review risks by normalizing enum-like adapter payloads,
  reading energy from `PlayerState`, validating trajectory step ordering, and
  clarifying that live combat metrics are gated on future action-level
  bindings.
- Verified local combat tests with
  `python -m pytest slay_the_spire/sts/test_combat_*.py -q`: 42 passed.
- Verified the Slay-the-Spire suite with `python -m pytest slay_the_spire/sts -q`:
  43 passed, 1 skipped because `STS_LIGHTSPEED_MODULE_DIR` is not set.
