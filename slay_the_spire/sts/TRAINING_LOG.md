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
- Fixed Codex review finding: non-targetable monsters with positive HP now
  remain alive for value evaluation and no longer produce false terminal
  victories.
- Advanced the `COMBAT_PLAN.md` Recommended Sequence checkpoint wording to
  separate completed local fixture/planner/model work from the remaining live
  simulator integration work.
- Current completed local layer: combat contracts, fixture/protocol adapters,
  serialization helpers, cloning tests, beam search, trajectory records,
  handcrafted value evaluation, policy ranking, and metric aggregation.
- Current blocker: live `sts_lightspeed` action bindings still do not expose
  legal combat actions, single-action application, action-level combat
  pause/resume, cheap live state cloning, or combat-only outcome metrics.
- Next checkpoint: implement and validate the live action-binding adapter, then
  run the existing planner/value/policy helpers against fixed-seed live combats.
- Added deterministic local sequence-completion modules:
  `combat_fixture.py`, `combat_training.py`, `combat_hybrid.py`, and
  `combat_experiment.py`.
- Added `STS_LIGHTSPEED_BINDING_PLAN.md` after inspecting upstream
  `BattleContext`, `BattleSimulator`, `InputState`, `CardManager`, and
  `Monster` internals. The recommendation remains to expose action-level
  pybind hooks upstream rather than porting the C++ combat engine into Python.
- Addressed final integration review findings: new modules now support package
  imports, fixture Vulnerable affects later attack damage, non-terminal fixture
  outcomes must point to a next step, and the binding plan now specifies
  command/index resolution, card-select task metadata, potion discard actions,
  and adapter-owned metrics counters.
