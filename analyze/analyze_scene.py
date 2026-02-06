import json
import math
import os
from collections import defaultdict

import pandas as pd
import argparse
from scipy import stats
import matplotlib.pyplot as plt


NOMINAL_DIR = "/data/puffer/results/nominal"


def parse_args():
    parser = argparse.ArgumentParser("Analyze scene")
    parser.add_argument("--long-tail", "-l", type=str, default="nominal")
    parser.add_argument("--out-path", "-o", type=str, default="/data/puffer/results/lane_breaker/correlation")
    parser.add_argument("--mode", "-m", type=str, default="replay", choices=["replay", "reactive", "nominal"])
    return parser.parse_args()

def scenarios_in_quantiles(df_mean, key, quantile, is_low=True):
    s = df_mean[key].astype(float)

    q = s.quantile(quantile)
    if is_low:
        df  = df_mean.loc[s <= q, ["scenario_id", key]].sort_values(key, ascending=True)
    else:
        df = df_mean.loc[s >= q, ["scenario_id", key]].sort_values(key, ascending=False)

    return df, q
    
def by_seed(df, scen_ids, metric_cols):
        sub = df[df["scenario_id"].isin(scen_ids)].copy()
        return (sub.groupby(["scenario_id", "seed"], as_index=False)[metric_cols]
                   .mean()
                   .sort_values(["scenario_id", "seed"]))

def df_to_seedkey_json(df, out_path=None):
    metric_cols = [c for c in df.columns if c != "seed"]
    d = {
        str(row["seed"]): {c: (None if row[c] is None else float(row[c])) for c in metric_cols}
        for _, row in df.iterrows()
    }

    out = [{k: v} for k, v in d.items()]

    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def analyze_scene():
    args = parse_args()
    long_tail_dir = f"/data/puffer/results/{args.long_tail}"
    scene_paths = [os.path.join(long_tail_dir, p) for p in os.listdir(long_tail_dir) if "wosac_scene" in p]

    total_wosac_results = []
    for seed in scene_paths:
        with open(seed, "r", encoding="utf-8") as f:
            wosac_scene_score = json.load(f)
        for wosac_result in wosac_scene_score:
            seed_id = seed[-13:-5]
            wosac_result["seed"] = seed_id
            total_wosac_results.append(wosac_result)
    df = pd.DataFrame(total_wosac_results)
    num_cols = df.select_dtypes("number").columns.drop("scenario_id", errors="ignore")
    df_mean = df.groupby("scenario_id", as_index=False)[num_cols].mean()
    df_mean["n_rows"] = df.groupby("scenario_id").size().values
    df_score, q_score = scenarios_in_quantiles(df_mean,"realism_meta_score", 0.1)
    df_collision, q_collision = scenarios_in_quantiles(df_mean, "likelihood_collision_indication", 0.1)
    df_dist, q_dist = scenarios_in_quantiles(df_mean, "likelihood_distance_to_nearest_object", 0.1)

    score_ids = df_score["scenario_id"].tolist()
    coll_ids  = df_collision["scenario_id"].tolist()
    dist_ids  = df_dist["scenario_id"].tolist()
    metric_cols = df.select_dtypes("number").columns.difference(["scenario_id"])
    score_by_seed = by_seed(df, score_ids, metric_cols)
    score_mean = score_by_seed.groupby("seed", as_index=False).mean()
    coll_by_seed = by_seed(df, coll_ids, metric_cols)
    coll_mean = coll_by_seed.groupby("seed", as_index=False).mean()
    dist_by_seed = by_seed(df, dist_ids, metric_cols)
    dist_mean = dist_by_seed.groupby("seed", as_index=False).mean()
    # save result

    df_to_seedkey_json(score_mean, os.path.join(long_tail_dir, "wosac_tail_score.json"))
    df_to_seedkey_json(coll_mean, os.path.join(long_tail_dir, "wosac_tail_collision.json"))
    df_to_seedkey_json(dist_mean, os.path.join(long_tail_dir, "wosac_tail_distance.json"))
if __name__ == "__main__":
    analyze_scene()