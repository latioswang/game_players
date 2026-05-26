"""Run seeded baseline trials through the external sts_lightspeed simulator.

The sts_lightspeed project is intentionally not vendored here. Build its
Python module externally, then point this script at the build directory:

    /usr/bin/python3 sts/sts_lightspeed_baseline.py \
      --module-dir /tmp/codex-sts-sim-eval/sts_lightspeed/build-py39 \
      --games 20 --simulation-count 1 --deck-policy heuristic

The Python version running this script must match the Python ABI used to build
the slaythespire extension module. On this machine, Python 3.9 worked with the
upstream bundled pybind11; Python 3.12 did not.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


GOOD_CARD_BONUSES = {
    "ANGER": 3.5,
    "ARMAMENTS": 4.0,
    "BATTLE_TRANCE": 5.0,
    "BLOOD_FOR_BLOOD": 3.0,
    "BODY_SLAM": 2.0,
    "CARNAGE": 5.0,
    "CORRUPTION": 4.5,
    "DEMON_FORM": 4.0,
    "DISARM": 5.0,
    "DARK_EMBRACE": 4.5,
    "FEEL_NO_PAIN": 4.0,
    "FIEND_FIRE": 5.5,
    "FLAME_BARRIER": 5.0,
    "FLEX": 1.5,
    "HEAVY_BLADE": 2.0,
    "IMPERVIOUS": 4.5,
    "INFLAME": 4.0,
    "OFFERING": 6.0,
    "POMMEL_STRIKE": 4.0,
    "POWER_THROUGH": 4.0,
    "SHOCKWAVE": 5.5,
    "SHRUG_IT_OFF": 5.0,
    "SPOT_WEAKNESS": 4.5,
    "TRUE_GRIT": 3.0,
    "TWIN_STRIKE": 3.0,
    "UPPERCUT": 4.5,
    "WHIRLWIND": 4.0,
}

BAD_CARD_PENALTIES = {
    "CLASH": -4.0,
    "FIRE_BREATHING": -2.0,
    "FORETHOUGHT": -3.0,
    "HAVOC": -2.0,
    "JUGGERNAUT": -1.5,
    "MAGNETISM": -2.0,
    "MAYHEM": -3.5,
    "RAGE": -2.0,
    "SECRET_TECHNIQUE": -1.5,
    "SECRET_WEAPON": -1.5,
    "TRANSMUTATION": -2.0,
}


@dataclass(frozen=True)
class DecisionLog:
    floor: int
    deck_size_before: int
    reward_options: list[str]
    picked: str | None
    scores: dict[str, float]


@dataclass(frozen=True)
class TrialResult:
    seed: int
    ascension: int
    simulation_count: int
    deck_policy: str
    outcome: str
    won: bool
    floor: int
    hp: int
    max_hp: int
    gold: int
    deck_size: int
    relic_count: int
    decision_count: int
    cards_picked: list[str]
    cards_skipped: int
    deck: list[str]
    decisions: list[DecisionLog]


def main() -> None:
    args = parse_args()
    sts = import_sts(args.module_dir)
    results = [
        run_trial(
            sts=sts,
            seed=args.start_seed + offset,
            ascension=args.ascension,
            simulation_count=args.simulation_count,
            boss_multiplier=args.boss_simulation_multiplier,
            deck_policy=args.deck_policy,
            max_decisions=args.max_decisions,
        )
        for offset in range(args.games)
    ]

    if args.jsonl:
        write_jsonl(args.jsonl, results)

    print_summary(results)
    if args.show_trials:
        for result in results:
            print(json.dumps(result_to_json(result), sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sts_lightspeed seeded baseline trials through its Python binding."
    )
    parser.add_argument(
        "--module-dir",
        type=Path,
        required=True,
        help="Directory containing slaythespire.cpython-*.so from sts_lightspeed.",
    )
    parser.add_argument("--games", type=int, default=20, help="Number of sequential seeds to evaluate.")
    parser.add_argument("--start-seed", type=int, default=1, help="First seed to evaluate.")
    parser.add_argument("--ascension", type=int, default=0, help="Ascension level for GameContext.")
    parser.add_argument(
        "--simulation-count",
        type=int,
        default=1,
        help="Base search simulations per decision for sts_lightspeed Agent.",
    )
    parser.add_argument(
        "--boss-simulation-multiplier",
        type=float,
        default=1.0,
        help="Agent boss-fight simulation multiplier.",
    )
    parser.add_argument(
        "--deck-policy",
        choices=("agent", "skip", "first", "heuristic"),
        default="heuristic",
        help=(
            "agent lets sts_lightspeed handle rewards; the other policies pause at "
            "card rewards and choose in Python."
        ),
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=500,
        help="Safety cap for Python reward decisions in one run.",
    )
    parser.add_argument("--jsonl", type=Path, help="Optional JSONL output path for per-run details.")
    parser.add_argument("--show-trials", action="store_true", help="Print each trial as JSON.")
    return parser.parse_args()


def import_sts(module_dir: Path) -> Any:
    module_dir = module_dir.expanduser().resolve()
    if not module_dir.exists():
        raise SystemExit(f"--module-dir does not exist: {module_dir}")
    sys.path.insert(0, str(module_dir))
    try:
        import slaythespire as sts  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Could not import slaythespire. Build sts_lightspeed's Python module "
            "and pass its build directory with --module-dir."
        ) from exc
    return sts


def run_trial(
    *,
    sts: Any,
    seed: int,
    ascension: int,
    simulation_count: int,
    boss_multiplier: float,
    deck_policy: str,
    max_decisions: int,
) -> TrialResult:
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
    agent = sts.Agent()
    agent.simulation_count_base = simulation_count
    agent.boss_simulation_multiplier = boss_multiplier
    agent.pause_on_card_reward = deck_policy != "agent"
    agent.print_logs = False

    decisions: list[DecisionLog] = []
    cards_picked: list[str] = []
    cards_skipped = 0

    if deck_policy == "agent":
        agent.playout(gc)
    else:
        while gc.outcome == sts.GameOutcome.UNDECIDED and len(decisions) < max_decisions:
            before = state_key(gc)
            agent.playout(gc)
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                break
            rewards = list(gc.get_card_reward())
            if rewards:
                picked, scores = choose_reward(gc, rewards, deck_policy)
                decision = DecisionLog(
                    floor=int(gc.floor_num),
                    deck_size_before=len(gc.deck),
                    reward_options=[card_name(card) for card in rewards],
                    picked=card_name(picked) if picked is not None else None,
                    scores=scores,
                )
                decisions.append(decision)
                if picked is None:
                    gc.skip_reward_cards()
                    cards_skipped += 1
                else:
                    gc.pick_reward_card(picked)
                    cards_picked.append(card_name(picked))
                continue

            after = state_key(gc)
            if after == before:
                raise RuntimeError(f"sts_lightspeed playout stalled at {after}")
        if gc.outcome == sts.GameOutcome.UNDECIDED:
            raise RuntimeError(
                "trial reached --max-decisions before terminal outcome; "
                f"seed={seed} policy={deck_policy} max_decisions={max_decisions}"
            )

    return TrialResult(
        seed=seed,
        ascension=ascension,
        simulation_count=simulation_count,
        deck_policy=deck_policy,
        outcome=enum_name(gc.outcome),
        won=gc.outcome == sts.GameOutcome.PLAYER_VICTORY,
        floor=int(gc.floor_num),
        hp=int(gc.cur_hp),
        max_hp=int(gc.max_hp),
        gold=int(gc.gold),
        deck_size=len(gc.deck),
        relic_count=len(gc.relics),
        decision_count=len(decisions),
        cards_picked=cards_picked,
        cards_skipped=cards_skipped,
        deck=[card_name(card) for card in gc.deck],
        decisions=decisions,
    )


def choose_reward(gc: Any, rewards: list[Any], deck_policy: str) -> tuple[Any | None, dict[str, float]]:
    if deck_policy == "skip":
        return None, {card_name(card): 0.0 for card in rewards}
    if deck_policy == "first":
        return rewards[0], {card_name(card): float(index == 0) for index, card in enumerate(rewards)}
    scored = {card_name(card): score_card_reward(gc, card) for card in rewards}
    best = max(rewards, key=lambda card: scored[card_name(card)])
    best_score = scored[card_name(best)]
    skip_threshold = skip_score_threshold(gc)
    return (best if best_score >= skip_threshold else None), scored


def score_card_reward(gc: Any, card: Any) -> float:
    name = card_name(card)
    deck = [card_name(deck_card) for deck_card in gc.deck]
    deck_size = len(deck)
    attack_count = count_cards_by_type(gc.deck, "ATTACK")
    skill_count = count_cards_by_type(gc.deck, "SKILL")

    score = 0.0
    score += GOOD_CARD_BONUSES.get(name, 0.0)
    score += BAD_CARD_PENALTIES.get(name, 0.0)

    rarity = enum_name(card.rarity)
    card_type = enum_name(card.type)
    if rarity == "RARE":
        score += 1.0
    elif rarity == "UNCOMMON":
        score += 0.5

    if card_type == "ATTACK":
        score += 1.5 if gc.floor_num < 18 else 0.25
        if attack_count <= 5:
            score += 2.0
    elif card_type == "SKILL":
        score += 0.5
        if skill_count <= 4 and gc.floor_num > 8:
            score += 1.0
    elif card_type == "POWER":
        score += 0.75 if gc.floor_num > 8 else -0.25

    if "STRIKE" in name and name != "POMMEL_STRIKE":
        score -= 1.0
    if "DEFEND" in name:
        score -= 1.0
    if name in deck:
        score -= 0.75
    if deck_size >= 25:
        score -= 1.5
    elif deck_size >= 20:
        score -= 0.75

    if "BASH" in deck and name in {"UPPERCUT", "SHOCKWAVE"}:
        score += 0.5
    if any(card_name in deck for card_name in ("FEEL_NO_PAIN", "DARK_EMBRACE", "CORRUPTION")):
        if name in {"TRUE_GRIT", "FIEND_FIRE", "SEVER_SOUL", "SECOND_WIND"}:
            score += 2.0
    return score


def skip_score_threshold(gc: Any) -> float:
    deck_size = len(gc.deck)
    if deck_size < 14:
        return 0.25
    if deck_size < 20:
        return 1.0
    return 2.0


def count_cards_by_type(cards: Iterable[Any], type_name: str) -> int:
    return sum(1 for card in cards if enum_name(card.type) == type_name)


def state_key(gc: Any) -> tuple[str, int, str, int, int]:
    return (enum_name(gc.outcome), int(gc.floor_num), enum_name(gc.screen_state), int(gc.cur_hp), len(gc.deck))


def enum_name(value: Any) -> str:
    text = str(value)
    return text.rsplit(".", 1)[-1]


def card_name(card: Any) -> str:
    return enum_name(card.id)


def print_summary(results: list[TrialResult]) -> None:
    for line in summarize_results(results):
        print(line)


def summarize_results(results: list[TrialResult]) -> list[str]:
    if not results:
        return ["No trials ran."]
    wins = sum(result.won for result in results)
    floors = [result.floor for result in results]
    hp_values = [result.hp for result in results]
    picks = [len(result.cards_picked) for result in results]
    skips = [result.cards_skipped for result in results]
    return [
        f"games={len(results)}",
        f"wins={wins}",
        f"win_rate={wins / len(results):.3f}",
        f"avg_floor={statistics.fmean(floors):.2f}",
        f"best_floor={max(floors)}",
        f"avg_final_hp={statistics.fmean(hp_values):.2f}",
        f"avg_cards_picked={statistics.fmean(picks):.2f}",
        f"avg_card_rewards_skipped={statistics.fmean(skips):.2f}",
        "outcomes=" + json.dumps(count_by(result.outcome for result in results), sort_keys=True),
    ]


def write_jsonl(path: Path, results: list[TrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result_to_json(result), sort_keys=True) + "\n")


def result_to_json(result: TrialResult) -> dict[str, Any]:
    data = asdict(result)
    data["decisions"] = [asdict(decision) for decision in result.decisions]
    return data


def count_by(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    main()
