#!/usr/bin/env python3
"""Compare logreplay.json and wosac.json across experiment folders.

For each EXP folder, loads results and takes mean across seeds (multiple model IDs).
Outputs comparison table across EXPs.
"""

import json
import math
import os
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


RESULTS_BASE = "/data/puffer/results"


def is_num(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def load_json_list(path: str):
    """Load JSON file - may be list of {model_id: metrics} or single dict."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    return []


def aggregate_seeds(entries, key_filter=None):
    """Take mean and std of numeric metrics across multiple seed entries.

    Returns (mean_dict, std_dict).
    key_filter: callable(metric_name) -> bool, or None to include all.
    """
    if key_filter is None:
        key_filter = lambda k: True
    bucket = defaultdict(list)
    for d in entries:
        if not isinstance(d, dict):
            continue
        for model_id, metrics in d.items():
            if not isinstance(metrics, dict):
                continue
            for k, v in metrics.items():
                if key_filter(k) and is_num(v):
                    bucket[k].append(float(v))
    mean_d = {}
    std_d = {}
    for k, vs in bucket.items():
        if vs:
            arr = np.array(vs)
            mean_d[k] = float(np.mean(arr))
            std_d[k] = float(np.std(arr)) if len(vs) > 1 else 0.0
    return mean_d, std_d


def _logreplay_key_filter(k):
    return k.startswith("ego_")


def _wosac_key_filter(k):
    return k != "num_agents"


def aggregate_zeroshot_matchups(entries, key_filter=None):
    """Aggregate zeroshot_reactive results.

    Keys are {ego}_vs_{other}. First: mean over others per ego (per seed).
    Then: mean and std over egos (seeds).
    Returns (mean_dict, std_dict).
    """
    if key_filter is None:
        key_filter = lambda k: True
    # Step 1: group by ego, for each ego take mean over others
    ego_bucket = defaultdict(lambda: defaultdict(list))
    for d in entries:
        if not isinstance(d, dict):
            continue
        for key, metrics in d.items():
            if "_vs_" not in key or not isinstance(metrics, dict):
                continue
            ego, other = key.split("_vs_", 1)
            if other == "selfplay":
                continue
            for k, v in metrics.items():
                if key_filter(k) and is_num(v):
                    ego_bucket[ego][k].append(float(v))

    # Per-ego means (one value per seed)
    per_ego_means = []
    for ego, metrics in ego_bucket.items():
        row = {}
        for k, vs in metrics.items():
            if vs:
                row[k] = sum(vs) / len(vs)
        if row:
            per_ego_means.append(row)

    # Step 2: mean and std across seeds (egos)
    if not per_ego_means:
        return {}, {}
    all_metrics = set()
    for row in per_ego_means:
        all_metrics.update(row.keys())
    mean_d = {}
    std_d = {}
    for k in all_metrics:
        vals = [r[k] for r in per_ego_means if k in r and is_num(r[k])]
        if vals:
            arr = np.array(vals)
            mean_d[k] = float(np.mean(arr))
            std_d[k] = float(np.std(arr)) if len(vals) > 1 else 0.0
    return mean_d, std_d


def collect_exp_results(base_path: str):
    """Collect (logreplay, wosac, unseen_other_seeds, unseen_other_rewards) aggregated by exp."""
    logreplay_by_exp = {}
    wosac_by_exp = {}
    unseen_seeds_by_exp = {}
    unseen_rewards_by_exp = {}

    if not os.path.isdir(base_path):
        return logreplay_by_exp, wosac_by_exp, unseen_seeds_by_exp, unseen_rewards_by_exp

    for exp in sorted(os.listdir(base_path)):
        exp_dir = os.path.join(base_path, exp)
        if not os.path.isdir(exp_dir):
            continue

        # logreplay.json: only ego_* metrics
        lr_path = os.path.join(exp_dir, "logreplay.json")
        if os.path.isfile(lr_path):
            entries = load_json_list(lr_path)
            if entries:
                logreplay_by_exp[exp] = aggregate_seeds(entries, key_filter=_logreplay_key_filter)

        # wosac.json: exclude num_agents
        wosac_path = os.path.join(exp_dir, "wosac.json")
        if os.path.isfile(wosac_path):
            entries = load_json_list(wosac_path)
            if entries:
                wosac_by_exp[exp] = aggregate_seeds(entries, key_filter=_wosac_key_filter)

        # unseen_other_seeds/zeroshot_reactive.json: ego_vs_other -> mean over others, then mean over seeds
        for mode, out in [("unseen_other_seeds", unseen_seeds_by_exp), ("unseen_other_rewards", unseen_rewards_by_exp)]:
            zs_path = os.path.join(exp_dir, mode, "zeroshot_reactive.json")
            if os.path.isfile(zs_path):
                entries = load_json_list(zs_path)
                if entries:
                    out[exp] = aggregate_zeroshot_matchups(entries, key_filter=_logreplay_key_filter)

    return logreplay_by_exp, wosac_by_exp, unseen_seeds_by_exp, unseen_rewards_by_exp


def build_comparison_dfs(data_by_exp: dict, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build DataFrames for mean and std. data_by_exp[exp] = (mean_dict, std_dict)."""
    if not data_by_exp:
        return pd.DataFrame(), pd.DataFrame()
    all_metrics = set()
    for val in data_by_exp.values():
        mean_d = val[0] if isinstance(val, tuple) else val
        all_metrics.update(mean_d.keys())
    all_metrics = sorted(all_metrics)

    rows_mean, rows_std = [], []
    for exp in sorted(data_by_exp.keys()):
        val = data_by_exp[exp]
        mean_d, std_d = val if isinstance(val, tuple) else (val, {})
        row_m, row_s = {"exp": exp}, {"exp": exp}
        for m in all_metrics:
            vm = mean_d.get(m)
            vs = std_d.get(m) if std_d else 0.0
            row_m[m] = float(vm) if is_num(vm) else None
            row_s[m] = float(vs) if is_num(vs) else (0.0 if vm is not None else None)
        rows_mean.append(row_m)
        rows_std.append(row_s)

    df_m = pd.DataFrame(rows_mean).set_index("exp")
    df_s = pd.DataFrame(rows_std).set_index("exp")
    df_m.index.name = df_s.index.name = label
    return df_m, df_s


def plot_bar_comparison(df_mean: pd.DataFrame, df_std: pd.DataFrame, out_dir: str, prefix: str):
    """Draw one bar chart per metric: EXPs on x-axis, value on y-axis, with std error bars.
    Y-axis is zoomed to highlight differences (data range + margin).
    """
    if df_mean.empty:
        return
    metrics = [c for c in df_mean.columns if df_mean[c].notna().any()]
    if not metrics:
        return
    exps = df_mean.index.tolist()
    n_exp = len(exps)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_exp, 1)))

    for metric in metrics:
        vals = np.array([float(x) if is_num(x) else np.nan for x in df_mean[metric].values])
        errs = np.zeros_like(vals)
        if df_std is not None and not df_std.empty and metric in df_std.columns:
            errs = np.where(np.isnan(df_std[metric].values), 0.0, df_std[metric].values)
        valid = ~np.isnan(vals)
        if not np.any(valid):
            continue
        fig, ax = plt.subplots(figsize=(max(6, n_exp * 0.8), 5))
        x = np.arange(n_exp)
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors[:n_exp], edgecolor="gray", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(exps, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{prefix}: {metric}")
        ax.grid(axis="y", alpha=0.3)
        # Zoom y-axis to highlight differences
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        span = vmax - vmin
        margin = max(span * 0.15, 0.02) if span > 0 else 0.1
        ax.set_ylim(vmin - margin, vmax + margin)
        plt.tight_layout()
        safe_name = metric.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(out_dir, f"{prefix}_{safe_name}.png"), dpi=150)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser("Compare logreplay, wosac, unseen_other_seeds, unseen_other_rewards")
    parser.add_argument("--base-path", "-b", type=str, default=RESULTS_BASE)
    parser.add_argument("--out-dir", "-o", type=str, default=None)
    parser.add_argument(
        "--format", "-f", type=str, default="all",
        choices=["logreplay", "wosac", "unseen_seeds", "unseen_rewards", "both", "all"]
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip bar plot generation")
    return parser.parse_args()


def _run_format(name, data, args, out_dir):
    """Process one format: print, save CSV, optionally plot."""
    if not data:
        return
    df_m, df_s = build_comparison_dfs(data, "exp")
    print(f"\n--- {name} (mean ± std across seeds) ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(df_m.to_string())
    safe = name.replace(" ", "_").replace(".", "_")
    df_m.to_csv(os.path.join(out_dir, f"{safe}_compare.csv"))
    df_s.to_csv(os.path.join(out_dir, f"{safe}_std.csv"))
    print(f"\nSaved: {out_dir}/{safe}_compare.csv, {safe}_std.csv")
    if not args.no_plot:
        plot_bar_comparison(df_m, df_s, out_dir, safe)


if __name__ == "__main__":
    args = parse_args()
    lr, wosac, unseen_seeds, unseen_rewards = collect_exp_results(args.base_path)

    out_dir = args.out_dir or os.path.join(args.base_path, "compare")
    os.makedirs(out_dir, exist_ok=True)

    fmt = args.format
    run_all = fmt == "all"
    run_both = fmt == "both"  # logreplay + wosac only
    if (fmt == "logreplay" or run_all or run_both) and lr:
        _run_format("logreplay", lr, args, out_dir)
    if (fmt == "wosac" or run_all or run_both) and wosac:
        _run_format("wosac", wosac, args, out_dir)
    if (fmt == "unseen_seeds" or run_all) and unseen_seeds:
        _run_format("unseen_other_seeds", unseen_seeds, args, out_dir)
    if (fmt == "unseen_rewards" or run_all) and unseen_rewards:
        _run_format("unseen_other_rewards", unseen_rewards, args, out_dir)

    if not any([lr, wosac, unseen_seeds, unseen_rewards]):
        print(f"No results found under {args.base_path}")
