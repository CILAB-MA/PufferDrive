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
    parser = argparse.ArgumentParser("Correlation plots overlay: nominal(blue) vs others(orange)")
    parser.add_argument("--long-tail", "-l", type=str, default="lane_breaker")
    parser.add_argument("--out-path", "-o", type=str, default="/data/puffer/results/lane_breaker/correlation")
    parser.add_argument("--mode", "-m", type=str, default="replay", choices=["replay", "reactive", "nominal"])
    parser.add_argument("--wosac", "-w", type=str, default="all", choices=["all", "collision", "distance", "score"])
    return parser.parse_args()


def is_num(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def load_maybe_list_of_singletons(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        out = {}
        for d in obj:
            if isinstance(d, dict):
                out.update(d)
        return out
    raise TypeError(f"Unexpected JSON top-level type: {type(obj)}")


def load_matchups_as_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [{k: v} for k, v in obj.items()]
    raise TypeError(f"Unexpected JSON top-level type: {type(obj)}")


def aggregate_xp_by_one(matchups_list, ego_ids):

    bucket = defaultdict(lambda: defaultdict(list))
    for matchup in matchups_list:
        if not isinstance(matchup, dict):
            continue
        for key, metrics in matchup.items():
            if "_vs_" not in key or not isinstance(metrics, dict):
                continue
            one, other = key.split("_vs_", 1)
            if other in ["npyqjrgh", "selfplay"]:
                continue
            if one not in ego_ids:
                continue
            for metric_name, value in metrics.items():
                if is_num(value):
                    bucket[one][metric_name].append(float(value))

    xp = {
        one: {mn: sum(vs) / len(vs) for mn, vs in ms.items() if vs}
        for one, ms in bucket.items()
    }
    return xp


def compute_diff_xp(mode: str, long_tail_dir: str, ego_ids):
    reactive = (mode == "reactive")

    # SP comes from nominal, mode-dependent file
    sp_src = os.path.join(NOMINAL_DIR, "zeroshot_reactive.json" if reactive else "zeroshot_replay.json")
    sp_flat = load_maybe_list_of_singletons(sp_src)

    sp_keys = [f"{one}_vs_selfplay" for one in ego_ids]
    sp = {k[:8]: sp_flat[k] for k in sp_keys if k in sp_flat}  # id -> metrics

    # XP comes from long_tail_dir, mode-dependent file
    xp_src = os.path.join(long_tail_dir, "zeroshot_reactive.json" if reactive else "zeroshot.json")
    matchups_list = load_matchups_as_list(xp_src)
    xp = aggregate_xp_by_one(matchups_list, ego_ids)

    play_metric_keys = []
    diff_xp = {}
    for ego, sp_m in sp.items():
        xp_m = xp.get(ego)
        if xp_m is None:
            continue
        d2 = {}
        for mn in (sp_m.keys() & xp_m.keys()):
            if "ego" in mn:
                play_metric_keys.append(mn)
                a, b = sp_m[mn], xp_m[mn]
                if is_num(a) and is_num(b):
                    d2[mn] = float(b) - float(a)  # XP - SP
        if d2:
            diff_xp[ego] = d2

    return diff_xp, list(set(play_metric_keys))


def _scatter_one(df, label, color):
    if len(df) < 3:
        return f"{label}: n={len(df)} (skip)"

    pear_r, pear_p = stats.pearsonr(df["x"], df["y"])
    plt.scatter(df["x"], df["y"], color=color, alpha=0.85, label=f"{label} | r={pear_r:.2f} p={pear_p:.2g}")

    # OLS line (only if non-constant)
    if df["x"].nunique() >= 2 and df["y"].nunique() >= 2:
        slope, intercept, *_ = stats.linregress(df["x"], df["y"])
        xline = pd.Series([df["x"].min(), df["x"].max()])
        plt.plot(xline, slope * xline + intercept, color=color, linewidth=1.5)

    return None


def plot_corr_overlay(metric_name, play_metric, wosac_dict, series, out_dir):
    plt.figure()
    any_points = False

    for label, diff_xp, color in series:
        rows = []
        common_ids = set(wosac_dict.keys()) & set(diff_xp.keys())
        for ego in common_ids:
            x = wosac_dict[ego].get(metric_name, None)
            y = diff_xp[ego].get(play_metric, None)
            if is_num(x) and is_num(y):
                rows.append({"id": ego, "x": float(x), "y": float(y)})

        df = pd.DataFrame(rows)
        if len(df) > 0:
            any_points = True
        _scatter_one(df, label, color)

    if not any_points:
        plt.close()
        return

    plt.xlabel(f"WOSAC: {metric_name}")
    plt.ylabel(f"XP - SP: {play_metric}")
    plt.title(f"{metric_name} vs {play_metric}")
    plt.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    safe_play = play_metric.replace("/", "_")
    safe_met = metric_name.replace("/", "_")
    plt.savefig(os.path.join(out_dir, f"{safe_met}_vs_{safe_play}.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    long_tail_dir = f"/data/puffer/results/{args.long_tail}"

    # ---- WOSAC (nominal) ----
    if args.wosac == "all":
        wosac_file = "wosac.json"
    else:
        wosac_file = f"wosac_tail_{args.wosac}.json"
    with open(os.path.join(NOMINAL_DIR, wosac_file), "r", encoding="utf-8") as f:
        wosac_score = json.load(f)

    first_metrics = next(iter(wosac_score[0].values()))
    metric_keys = list(first_metrics.keys())
    wosac_dict = {next(iter(d)): next(iter(d.values())) for d in wosac_score}
    ego_ids = [next(iter(d)) for d in wosac_score]

    # ---- always compute nominal baseline (blue) using replay-style diff (nominal XP vs nominal SP) ----
    # baseline mode is always "replay" for nominal curve
    nominal_diff_xp, nominal_play_keys = compute_diff_xp("replay", NOMINAL_DIR, ego_ids)

    # ---- decide which extra modes to overlay (orange) ----
    if args.mode == "nominal":
        overlay_modes = ["replay", "reactive"]
    else:
        overlay_modes = [args.mode]

    overlays = []
    play_metric_keys = set(nominal_play_keys)

    for m in overlay_modes:
        if m == "nominal":
            continue
        diff_xp_m, play_keys_m = compute_diff_xp(m, long_tail_dir, ego_ids)
        overlays.append((m, diff_xp_m, "tab:orange"))
        play_metric_keys |= set(play_keys_m)

    out_dir = args.out_path + f"/{args.mode}_{args.wosac}"
    os.makedirs(out_dir, exist_ok=True)

    for play_metric in sorted(play_metric_keys):
        if "ego_" not in play_metric:
            continue
        if play_metric == "ego_n":
            continue
        for metric in metric_keys:
            if metric not in ["realism_meta_score", "ade"]:
                continue

            series = [("nominal", nominal_diff_xp, "tab:blue")] + overlays
            plot_corr_overlay(metric, play_metric, wosac_dict, series, out_dir)
