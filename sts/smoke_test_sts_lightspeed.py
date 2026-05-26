"""Smoke-test the external sts_lightspeed simulator and local baseline policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from sts_lightspeed_baseline import import_sts, run_trial, summarize_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sts_lightspeed simulator smoke tests.")
    parser.add_argument(
        "--module-dir",
        type=Path,
        required=True,
        help="Directory containing slaythespire.cpython-*.so from sts_lightspeed.",
    )
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--simulation-count", type=int, default=1)
    args = parser.parse_args()

    sts = import_sts(args.module_dir)
    run_direct_simulator_smoke(sts)
    for policy in ("agent", "skip", "heuristic"):
        results = [
            run_trial(
                sts=sts,
                seed=args.start_seed + offset,
                ascension=0,
                simulation_count=args.simulation_count,
                boss_multiplier=1.0,
                deck_policy=policy,
                max_decisions=500,
            )
            for offset in range(args.games)
        ]
        assert len(results) == args.games
        assert all(result.outcome in {"PLAYER_LOSS", "PLAYER_VICTORY"} for result in results)
        if policy == "skip":
            assert all(not result.cards_picked for result in results)
        if policy == "heuristic":
            assert any(result.decision_count > 0 for result in results)

        print(f"policy={policy}")
        for line in summarize_results(results):
            print(line)


def run_direct_simulator_smoke(sts: object) -> None:
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 1, 0)
    agent = sts.Agent()
    agent.simulation_count_base = 1
    agent.playout(gc)

    assert sts.get_seed_str(1) == "1"
    assert gc.floor_num >= 0
    assert len(gc.deck) > 0
    assert gc.outcome in {sts.GameOutcome.PLAYER_LOSS, sts.GameOutcome.PLAYER_VICTORY}

    print("direct_simulator=ok")
    print(f"seed={sts.get_seed_str(1)}")
    print(f"outcome={gc.outcome}")
    print(f"floor={gc.floor_num}")
    print(f"hp={gc.cur_hp}")
    print(f"deck_size={len(gc.deck)}")


if __name__ == "__main__":
    main()

