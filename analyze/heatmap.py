import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser("Make a heatmap with different range")
    parser.add_argument('--base-path', '-b', type=str, default="/data/puffer/results")
    parser.add_argument('--type', '-t', type=str, default="lane_breaker")
    parser.add_argument("--mode", "-m", type=str, default="replay", choices=["reactive", "replay"])
    args = parser.parse_args()
    return args

def should_ignore(agent_id, ignores):
    return any(s in agent_id for s in ignores)

if __name__ == "__main__":
    args = parse_args()
    b = args.base_path
    out_path = os.path.join(b, args.type, "heatmap")
    filename = "zeroshot" if args.mode == "replay" else "zeroshot_reactive"
    json_path = os.path.join(b, args.type, f"{filename}.json")
    os.makedirs(out_path, exist_ok=True)
    with open(json_path, "r", encoding="utf-8") as f:
        matches = json.load(f)
    
    # score will be calculated by xp - sp
    # this is sp
    # when we use replay method, we should calculate it as ((xp - rp) - (sp - rp))
    with open(f"/data/puffer/results/nominal/{filename}.json", "r", encoding="utf-8") as f:
        ego_matches = json.load(f)
    def only_key(d):
        return next(iter(d))
    def only_item(d):
        return next(iter(d.items()))
    if args.mode == "replay":
        new_ego_matches = []
        for m in ego_matches:
            k, v = only_item(m)
            if k[:8] == k[-8:]:
                new_k = k[:-8] + "selfplay"
                new_ego_matches.append({new_k: v})
        ego_matches = new_ego_matches
    else:
        ego_matches = [m for m in ego_matches
                    if (k := only_key(m)).endswith("selfplay")]
    matchups = {}
    for entry in ego_matches:
        if isinstance(entry, dict):
            matchups.update(entry)
    for entry in matches:
        if isinstance(entry, dict):
            matchups.update(entry)
    
    ego_agents = sorted({ k.split("_vs_", 1)[0] for k in matchups.keys() if "_vs_" in k } - {"selfplay"})[:10]
    other_agents = sorted({ k.split("_vs_", 1)[1] for k in matchups.keys() if "_vs_" in k } - {"selfplay"})[:10]

    # agents.remove("nggfacko")
    # ego metrics
    print(ego_agents, len(ego_agents),other_agents, len(other_agents))
    sample = next(iter(matchups.values()))
    ego_metrics = sorted([k for k in sample.keys() if k.startswith("ego_")])

    def build_delta_df(metric):
        df = pd.DataFrame(index=ego_agents, columns=other_agents, dtype=float)
        for one in ego_agents:
            base_key = f"{one}_vs_selfplay"
            base = matchups.get(base_key, {}).get(metric, np.nan)
            if np.isnan(base):
                diag_key = f"{one}_vs_{one}"
                base = matchups.get(diag_key, {}).get(metric, np.nan)
            for other in other_agents:
                key = f"{one}_vs_{other}"
                val = matchups.get(key, {}).get(metric, np.nan)
                df.loc[one, other] = val - base
        return df

    for metric in ego_metrics:
        if "ego_n" == metric:
            continue
    
        df = build_delta_df(metric)
        plt.figure(figsize=(0.45 * len(ego_agents) + 4, 0.45 * len(other_agents) + 3))
        ax = sns.heatmap(
            df,
            cmap="coolwarm",
            center=0.0,
            square=False,
            linewidths=0.2,
            linecolor="white",
            cbar_kws={"label": f"Δ {metric} (vs selfplay)"},
        )
        ax.set_title(f"{metric}: (one_vs_other) - (one_vs_selfplay)")
        ax.set_xlabel("other")
        ax.set_ylabel("one")
        plt.tight_layout()

        out = os.path.join(out_path, f"{args.mode}_{metric}.png")
        plt.savefig(out, dpi=200)
        plt.close()

    print("saved metrics:", ego_metrics)
    print("output dir:", out_path)
