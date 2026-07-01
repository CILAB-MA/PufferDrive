#!/usr/bin/env python3
"""Find the worst-performing maps in a per-scenario log JSON (see scenario_log.py).

Input is the ``<id>_selfplay.json`` (or any ``--eval.scenario-log-path`` output)
produced during zeroshot/selfplay runs: a dict keyed by scenario/map id -> metrics
(score, collision_rate, offroad_rate, dnf_rate, ...).

Usage:
    python3 analyze/find_worst_maps.py /data/puffer/results/selfplay/selfplay/scenario_logs/e729wx4e_selfplay.json
    python3 analyze/find_worst_maps.py <path> --top 50 --sort-by collision_rate --out worst.csv
"""

import argparse
import json

import pandas as pd

DISPLAY_COLUMNS = [
    "map_id",
    "score",
    "collision_rate",
    "offroad_rate",
    "dnf_rate",
    "completion_rate",
    "lane_alignment_rate",
    "episode_return",
]


def load_scenario_log(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict keyed by scenario id in {path}, got {type(data)}")

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "scenario_id"
    df = df.reset_index()
    df["scenario_id"] = df["scenario_id"].astype(int)
    if "map_id" in df.columns:
        df["map_id"] = df["map_id"].astype(int)
    return df


def add_failure_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_collision"] = df.get("collision_rate", 0) > 0
    df["is_offroad"] = df.get("offroad_rate", 0) > 0
    df["is_dnf"] = df.get("dnf_rate", 0) > 0
    df["did_not_solve"] = df.get("score", 1) < 1
    df["failure_score"] = (
        (1 - df.get("score", 1))
        + df.get("collision_rate", 0)
        + df.get("offroad_rate", 0)
        + df.get("dnf_rate", 0)
    )
    return df


def print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"Total maps: {n}")
    print(f"  mean score:            {df['score'].mean():.4f}")
    print(f"  solved (score==1):     {(df['score'] >= 1).sum()} ({(df['score'] >= 1).mean() * 100:.1f}%)")
    print(f"  any collision:         {df['is_collision'].sum()} ({df['is_collision'].mean() * 100:.1f}%)")
    print(f"  any offroad:           {df['is_offroad'].sum()} ({df['is_offroad'].mean() * 100:.1f}%)")
    print(f"  dnf:                   {df['is_dnf'].sum()} ({df['is_dnf'].mean() * 100:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Path to a <id>_selfplay.json scenario log file")
    parser.add_argument("--top", type=int, default=30, help="Number of worst maps to show (default: 30)")
    parser.add_argument(
        "--sort-by",
        default="failure_score",
        choices=["failure_score", "score", "collision_rate", "offroad_rate", "dnf_rate"],
        help="Ranking metric (default: failure_score, a combined badness score)",
    )
    parser.add_argument("--out", default=None, help="Write the full sorted table to this CSV path")
    args = parser.parse_args()

    df = load_scenario_log(args.path)
    df = add_failure_flags(df)
    print_summary(df)

    ascending = args.sort_by == "score"
    ranked = df.sort_values(args.sort_by, ascending=ascending)

    if args.out:
        ranked.to_csv(args.out, index=False)
        print(f"Wrote full sorted table ({len(ranked)} rows) to {args.out}")

    cols = [c for c in DISPLAY_COLUMNS if c in ranked.columns]
    print(f"Top {args.top} worst maps by {args.sort_by}:")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(ranked[cols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
