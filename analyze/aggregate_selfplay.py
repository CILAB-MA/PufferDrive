#!/usr/bin/env python3
"""Aggregate population self-play zeroshot_reactive.json into mean/std."""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np


def is_num(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def load_entries(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    return []


def aggregate_selfplay(entries, key_filter=None):
    if key_filter is None:
        key_filter = lambda _: True
    bucket = defaultdict(list)
    models = []
    for d in entries:
        if not isinstance(d, dict):
            continue
        for matchup, metrics in d.items():
            if not matchup.endswith("_vs_selfplay") or not isinstance(metrics, dict):
                continue
            models.append(matchup[: -len("_vs_selfplay")])
            for k, v in metrics.items():
                if key_filter(k) and is_num(v):
                    bucket[k].append(float(v))
    mean_d, std_d = {}, {}
    for k, vs in bucket.items():
        if vs:
            arr = np.array(vs)
            mean_d[k] = float(np.mean(arr))
            std_d[k] = float(np.std(arr)) if len(vs) > 1 else 0.0
    return sorted(models), mean_d, std_d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", help="zeroshot_reactive.json from population_play.sh")
    parser.add_argument(
        "--out",
        default=None,
        help="Write summary JSON here (default: <results_json_dir>/population_selfplay_summary.json)",
    )
    parser.add_argument("--ego-only", action="store_true", help="Only aggregate ego_* metrics")
    args = parser.parse_args()

    key_filter = (lambda k: k.startswith("ego_")) if args.ego_only else (lambda _: True)
    entries = load_entries(args.results_json)
    models, mean_d, std_d = aggregate_selfplay(entries, key_filter=key_filter)
    if not mean_d:
        raise SystemExit(f"No *_vs_selfplay entries in {args.results_json}")

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.results_json)),
        "population_selfplay_summary.json",
    )
    summary = {
        "n_models": len(models),
        "models": models,
        "mean": mean_d,
        "std": std_d,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Aggregated {len(models)} self-play runs: {', '.join(models)}")
    for k in sorted(mean_d):
        print(f"  {k}: {mean_d[k]:.6f} ± {std_d[k]:.6f}")
    print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
