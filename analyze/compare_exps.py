#!/usr/bin/env python3
"""Compare logreplay.json and wosac.json across experiment folders.

For each EXP folder, loads results and takes mean across seeds (multiple model IDs).
Outputs comparison table across EXPs.

Plots (bar / subplot figures) only show:
collision_per_agent, offroad_per_agent, score, lane_alignment_rate
(matching ``ego_*``-prefixed columns when present).
"""

import json
import math
import os
import re
from collections import defaultdict
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import argparse


RESULTS_BASE = "/data/puffer/results"

# Plots only these metrics (CSV tables still include all keys). Matches ``ego_*`` prefix in data.
PLOT_METRICS_ORDER = (
    "collision_per_agent",
    "offroad_per_agent",
    "score",
    "lane_alignment_rate",
)

# Exact / prefixed synonyms first; see ``_fuzzy_metric_column`` for avg_* / naming drift.
PLOT_METRIC_SYNONYMS = {
    "collision_per_agent": (
        "collision_per_agent",
        "collisions_per_agent",
        "avg_collision_per_agent",
        "avg_collisions_per_agent",
    ),
    "offroad_per_agent": (
        "offroad_per_agent",
        "off_road_per_agent",
        "avg_offroad_per_agent",
        "avg_off_road_per_agent",
    ),
    "score": ("score", "avg_score", "total_score"),
    "lane_alignment_rate": (
        "lane_alignment_rate",
        "lane_alignment",
        "avg_lane_alignment_rate",
    ),
}

# Google / Material brand-adjacent palette (per experiment bar)
GOOGLE_PALETTE = (
    "#4285F4",  # blue
    "#EA4335",  # red
    "#FBBC04",  # yellow
    "#34A853",  # green
    "#A142F4",  # purple
    "#00BCD4",  # cyan
    "#FF6D00",  # orange
    "#9AA0A6",  # gray
)


def _bar_colors_google(n: int) -> list:
    return [GOOGLE_PALETTE[i % len(GOOGLE_PALETTE)] for i in range(max(n, 1))]


def _fuzzy_metric_column(df_mean: pd.DataFrame, base: str) -> Optional[str]:
    """Last resort: match benchmark-style names (avg_collisions_per_agent, etc.)."""
    cols = [c for c in df_mean.columns if df_mean[c].notna().any()]
    if base == "collision_per_agent":
        for c in cols:
            lc = c.lower()
            if ("collision" in lc or "collisions" in lc) and "per_agent" in lc:
                return c
    if base == "offroad_per_agent":
        for c in cols:
            lc = c.lower()
            if ("offroad" in lc or "off_road" in lc) and "per_agent" in lc:
                return c
    if base == "lane_alignment_rate":
        for c in cols:
            lc = c.lower()
            if "lane" in lc and "alignment" in lc:
                return c
    if base == "score":
        hits = []
        for c in cols:
            lc = c.lower()
            if lc in ("score", "ego_score"):
                hits.append((0, len(c), c))
            elif lc.endswith("_score") and "collision" not in lc and "offroad" not in lc:
                hits.append((1, len(c), c))
        if hits:
            hits.sort(key=lambda x: (x[0], x[1], x[2]))
            return hits[0][2]
    return None


def _plot_metric_columns(df_mean: pd.DataFrame) -> list:
    """Resolve column names for PLOT_METRICS_ORDER (synonyms, ego_, fuzzy)."""
    if df_mean.empty:
        return []
    available = list(df_mean.columns)
    result = []
    for base in PLOT_METRICS_ORDER:
        found: Optional[str] = None
        synonyms = PLOT_METRIC_SYNONYMS.get(base, (base,))
        for syn in synonyms:
            for c in (syn, f"ego_{syn}"):
                if c in df_mean.columns and df_mean[c].notna().any():
                    found = c
                    break
            if found:
                break
        if found is None:
            for c in available:
                if (c == base or c.endswith(base)) and df_mean[c].notna().any():
                    found = c
                    break
        if found is None:
            found = _fuzzy_metric_column(df_mean, base)
        if found is not None and found not in result:
            result.append(found)
    return result


def _metric_column_for_base(df_mean: pd.DataFrame, base: str) -> Optional[str]:
    """Resolved column in ``df_mean`` for a canonical PLOT_METRICS_ORDER key, or None."""
    for c in _plot_metric_columns(df_mean):
        if _canonical_base_for_column(c) == base:
            return c
    return None


def _slice_plot_metrics(
    df_mean: pd.DataFrame, df_std: Optional[pd.DataFrame]
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    cols = _plot_metric_columns(df_mean)
    if not cols:
        return pd.DataFrame(index=df_mean.index), None
    dm = df_mean[cols].copy()
    if df_std is None or df_std.empty:
        return dm, None
    ds = pd.DataFrame(index=df_std.index)
    for c in cols:
        ds[c] = df_std[c] if c in df_std.columns else 0.0
    return dm, ds


def _canonical_base_for_column(column: str) -> str:
    """Map resolved JSON column to canonical metric key for titles."""
    col = str(column)
    for base, syns in PLOT_METRIC_SYNONYMS.items():
        for syn in syns:
            if col == syn or col == f"ego_{syn}" or col.endswith(syn):
                return base
    for base in PLOT_METRICS_ORDER:
        if col == base or col == f"ego_{base}" or col.endswith(base):
            return base
    return col


def _metric_plot_title(column: str) -> str:
    """Human-readable subplot title for a resolved column (title case; score → Success Score)."""
    base = _canonical_base_for_column(column)
    if base == "score":
        return "Success Score"
    s = format_metric_display(str(base).replace("_", " "))
    return s.title()


def _metric_ylabel(column: str) -> str:
    """Y-axis label: per-agent counts → 횟수; score / rates → %."""
    base = _canonical_base_for_column(column)
    if base in ("collision_per_agent", "offroad_per_agent"):
        return "Count per Episode"
    if base in ("score", "lane_alignment_rate"):
        return "Percent (%)"
    b = str(base).lower()
    if "per_agent" in b:
        return "Count per Episode"
    if "rate" in b:
        return "Percent (%)"
    return "Value"


def _ylim_with_errs(vals: np.ndarray, errs: np.ndarray) -> tuple[float, float]:
    """Y-limits that include error bars + padding (caps stay inside axes)."""
    v = np.asarray(vals, dtype=float)
    e = np.asarray(errs, dtype=float)
    mask = np.isfinite(v)
    if not np.any(mask):
        return 0.0, 1.0
    e = np.where(np.isfinite(e), e, 0.0)
    low = float(np.nanmin((v - e)[mask]))
    high = float(np.nanmax((v + e)[mask]))
    span = max(high - low, abs(high) * 1e-6, 1e-9)
    pad = max(span * 0.12, 0.02 * max(abs(high), abs(low), 1.0))
    return low - pad, high + pad


# Repo-root times.ttf (analyze/compare_exps.py -> .. /times.ttf)
_TIMES_TTF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "times.ttf")
)
_registered_compare_font: Optional[str] = None


def _compare_serif_font_name() -> str:
    global _registered_compare_font
    if _registered_compare_font is not None:
        return _registered_compare_font
    if os.path.isfile(_TIMES_TTF):
        try:
            font_manager.fontManager.addfont(_TIMES_TTF)
            _registered_compare_font = font_manager.FontProperties(
                fname=_TIMES_TTF
            ).get_name()
            return _registered_compare_font
        except (OSError, ValueError, RuntimeError):
            pass
    _registered_compare_font = "DejaVu Serif"
    return _registered_compare_font


def format_metric_display(metric: str) -> str:
    """Subplot title text for a metric (underscores already normalized by caller)."""
    return str(metric)


def _exp_lane_nominal_mix_marker(exp: str) -> bool:
    """True if name encodes the Lane+Nominal mix token ``0.25`` (or ``_0_25``) — shown as ``(L+N)`` on plots."""
    s = str(exp)
    if re.search(r"(?<![0-9.])0\.25(?![0-9])", s):
        return True
    if re.search(r"(?:^|_)0_25(?:_|$)", s):
        return True
    return False


def _apply_ln_mix_display(label: str, orig_exp: str) -> str:
    """Replace ``0.25`` / ``0_25`` with ``(L+N)``; if mix was only in a stripped suffix, append `` (L+N)``."""
    out = re.sub(r"(?<![0-9.])0\.25(?![0-9])", "(L+N)", label)
    out = re.sub(r"(?<![0-9])0_25(?![0-9])", "(L+N)", out)
    if _exp_lane_nominal_mix_marker(orig_exp) and "(L+N)" not in out:
        out = f"{out} (L+N)"
    return out


def format_exp_xlabel(exp: str) -> str:
    """X-axis tick labels for subplot/bar figures.

    ``reactive_<v>`` → ``Reactive-PBT (<V>)``, ``replay_<v>`` → ``ReCord (<V>)`` with ``<V>`` from
    ``_format_type_name``; ``selfplay`` → ``SP``; ``logreplay`` is never rewritten as *replay*.
    ``0.25`` / ``0_25`` (Lane+Nominal mix) → ``(L+N)`` in the label (or trailing `` (L+N)`` if stripped).
    Non-matching names still get generic ``reactive`` / ``replay`` word replacements.
    """
    orig = str(exp)
    s = orig
    s = re.sub(r"(?:_\d+(?:\.\d+)?)+$", "", s)
    if re.search("selfplay", s, re.I):
        s = re.sub("selfplay", "SP", s, flags=re.I)
    m = re.search(r"(?:^|_)reactive_([a-zA-Z0-9_]+)$", s)
    if m:
        return _apply_ln_mix_display(
            f"Reactive-PBT ({_format_type_name(m.group(1))})", orig
        )
    m = re.search(r"(?:^|_)(?<!log)replay_([a-zA-Z0-9_]+)$", s)
    if m:
        return _apply_ln_mix_display(
            f"ReCord ({_format_type_name(m.group(1))})", orig
        )
    s = re.sub(r"(?<!log)replay", "ReCord", s)
    s = re.sub(r"reactive", "Reactive-PBT", s, flags=re.I)
    return _apply_ln_mix_display(s, orig)


def _format_type_name(type_name: str) -> str:
    """Normalize variant tokens: snake_case -> 'Snake Case'."""
    toks = [t for t in str(type_name).split("_") if t]
    return " ".join(t[:1].upper() + t[1:] for t in toks)


def _disambiguated_family_label(exp: str) -> Optional[str]:
    """When tick labels collide, use same style as ``format_exp_xlabel`` (variant in parentheses)."""
    orig = str(exp)
    s = orig
    s = re.sub(r"(?:_\d+(?:\.\d+)?)+$", "", s)
    m = re.search(r"(?:^|_)reactive_([a-zA-Z0-9_]+)$", s)
    if m:
        return _apply_ln_mix_display(
            f"Reactive-PBT ({_format_type_name(m.group(1))})", orig
        )
    m = re.search(r"(?:^|_)(?<!log)replay_([a-zA-Z0-9_]+)$", s)
    if m:
        return _apply_ln_mix_display(
            f"ReCord ({_format_type_name(m.group(1))})", orig
        )
    return None


def _exp_replay_reactive_family(exp: str) -> Optional[str]:
    """``replay`` or ``reactive`` if folder matches replay_* / reactive_* (excluding logreplay); else None."""
    s = str(exp)
    if "selfplay" in s.lower():
        return None
    s = re.sub(r"(?:_\d+(?:\.\d+)?)+$", "", s)
    if re.search(r"(?:^|_)reactive_", s):
        return "reactive"
    if re.search(r"(?:^|_)(?<!log)replay_", s):
        return "replay"
    return None


def _replay_reactive_variant_kind(exp: str) -> Optional[tuple[str, str]]:
    """Parse ``replay_*`` / ``reactive_*`` tail into ``(variant_slug, 'replay'|'reactive')``."""
    s = str(exp)
    if "selfplay" in s.lower():
        return None
    s = re.sub(r"(?:_\d+(?:\.\d+)?)+$", "", s)
    m = re.search(r"(?:^|_)reactive_([a-zA-Z0-9_]+)$", s)
    if m:
        return (m.group(1), "reactive")
    m = re.search(r"(?:^|_)(?<!log)replay_([a-zA-Z0-9_]+)$", s)
    if m:
        return (m.group(1), "replay")
    return None


# Variant column-group order in LaTeX (then alphabetical for unknown slugs).
_LATEX_VARIANT_GROUP_PRIORITY = ("nominal", "lane")


def _latex_variant_group_sort_key(slug: str) -> tuple[int, str]:
    if slug in _LATEX_VARIANT_GROUP_PRIORITY:
        return (_LATEX_VARIANT_GROUP_PRIORITY.index(slug), slug)
    return (len(_LATEX_VARIANT_GROUP_PRIORITY), slug)


def _latex_replay_reactive_variant_group_layout(
    exps: list[str],
) -> Optional[tuple[list[str], list[str], list[str]]]:
    """If every exp is ``replay_<v>`` / ``reactive_<v>`` with a full grid per ``v``, return column layout.

    Returns ``(exps_left_to_right, group_titles_tex, variant_slugs)`` where columns are
    ``[replay_v1, reactive_v1, replay_v2, reactive_v2, ...]`` grouped by variant ``v`` (nominal before lane,
    then other slugs alphabetically). ``group_titles`` are LaTeX-escaped display names for headers;
    ``variant_slugs`` matches one slug per variant group (same order as column pairs).
    """
    parsed: dict[str, tuple[str, str]] = {}
    for e in exps:
        vk = _replay_reactive_variant_kind(e)
        if vk is None:
            return None
        parsed[e] = vk
    by_var: dict[str, dict[str, str]] = defaultdict(dict)
    for e, (slug, kind) in parsed.items():
        if kind in by_var[slug]:
            return None
        by_var[slug][kind] = e
    for slug, kinds in by_var.items():
        if set(kinds.keys()) != {"replay", "reactive"}:
            return None
    ordered_vars = sorted(by_var.keys(), key=_latex_variant_group_sort_key)
    col_order: list[str] = []
    group_tex: list[str] = []
    variant_slugs: list[str] = []
    for v in ordered_vars:
        col_order.append(by_var[v]["replay"])
        col_order.append(by_var[v]["reactive"])
        group_tex.append(_latex_escape(_format_type_name(v)))
        variant_slugs.append(v)
    if len(col_order) != len(exps):
        return None
    return (col_order, group_tex, variant_slugs)


def _rr_variant_pairs(col_order: list[str]) -> list[tuple[str, str]]:
    """``col_order`` is ``[replay_v1, reactive_v1, ...]`` → ``[(r1,re1), ...]``."""
    return [(col_order[i], col_order[i + 1]) for i in range(0, len(col_order), 2)]


def _diff_mean_std(
    dm: pd.DataFrame,
    ds: Optional[pd.DataFrame],
    exp_r: str,
    exp_re: str,
    base: str,
) -> tuple[Optional[float], Optional[float]]:
    """``μ_replay − μ_reactive`` and ``σ_Δ ≈ \\sqrt{σ_r^2 + σ_{re}^2}`` (seed stds treated as independent)."""
    col = _metric_column_for_base(dm, base)
    if col is None or exp_r not in dm.index or exp_re not in dm.index:
        return None, None
    mr, mre = dm.loc[exp_r, col], dm.loc[exp_re, col]
    if not is_num(mr) or not is_num(mre):
        return None, None
    d_mean = float(mr) - float(mre)
    sr, sre = 0.0, 0.0
    if ds is not None and not ds.empty and col in ds.columns:
        if exp_r in ds.index:
            vr = ds.loc[exp_r, col]
            if is_num(vr):
                sr = float(vr)
        if exp_re in ds.index:
            ve = ds.loc[exp_re, col]
            if is_num(ve):
                sre = float(ve)
    d_std = math.hypot(sr, sre)
    return d_mean, d_std


def _row_best_rr_diff(
    dm: pd.DataFrame,
    ds: Optional[pd.DataFrame],
    pairs: list[tuple[str, str]],
    base: str,
    *,
    atol: float = 1e-9,
) -> set[int]:
    """Variant index whose Replay$-$Reactive mean is best for this metric (same min/max rule as raw means)."""
    maximize = _metric_higher_is_better(base)
    scored: list[tuple[int, float]] = []
    for k, (er, ee) in enumerate(pairs):
        m, _ = _diff_mean_std(dm, ds, er, ee, base)
        if m is None:
            continue
        scored.append((k, float(m)))
    if not scored:
        return set()
    if maximize:
        best = max(m for _, m in scored)
        return {k for k, m in scored if m + atol >= best}
    best = min(m for _, m in scored)
    return {k for k, m in scored if m - atol <= best}


def _combined_diff_cell(
    dm: pd.DataFrame,
    ds: Optional[pd.DataFrame],
    exp_r: str,
    exp_re: str,
    base: str,
    *,
    bold: bool = False,
) -> str:
    vm, vs = _diff_mean_std(dm, ds, exp_r, exp_re, base)
    if vm is None:
        return "\\multicolumn{1}{c}{--}"
    vstd = float(vs) if vs is not None else 0.0
    s = f"{float(vm):+.3f} \\pm {vstd:.3f}"
    if bold:
        return f"$\\bm{{{s}}}$"
    return f"${s}$"


def format_exp_xlabels(exps: list[str]) -> list[str]:
    """Format x labels; if duplicates occur, disambiguate or fall back to the raw experiment name."""
    labels = [format_exp_xlabel(e) for e in exps]
    counts = defaultdict(int)
    for lb in labels:
        counts[lb] += 1
    out = []
    for exp, lb in zip(exps, labels):
        if counts[lb] > 1:
            dis = _disambiguated_family_label(exp)
            if dis and dis != lb:
                out.append(dis)
            else:
                out.append(str(exp))
        else:
            out.append(lb)
    return out


def is_num(x):
    """True for finite reals (includes ``numpy`` scalar types from ``DataFrame.loc``)."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError, OverflowError):
        return False


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


def collect_exp_results(base_path: str, exp_names=None):
    """Collect (logreplay, wosac, unseen_other_seeds, unseen_other_rewards) aggregated by exp."""
    logreplay_by_exp = {}
    wosac_by_exp = {}
    unseen_seeds_by_exp = {}
    unseen_rewards_by_exp = {}

    if not os.path.isdir(base_path):
        return logreplay_by_exp, wosac_by_exp, unseen_seeds_by_exp, unseen_rewards_by_exp

    if exp_names:
        exp_iter = list(exp_names)
    else:
        exp_iter = sorted(os.listdir(base_path))

    for exp in exp_iter:
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
    font = _compare_serif_font_name()
    df_mean, df_std = _slice_plot_metrics(df_mean, df_std)
    if df_mean.empty or len(df_mean.columns) == 0:
        return
    metrics = [c for c in df_mean.columns if df_mean[c].notna().any()]
    if not metrics:
        return
    exps = df_mean.index.tolist()
    n_exp = len(exps)
    colors = _bar_colors_google(n_exp)

    for metric in metrics:
        vals = np.array([float(x) if is_num(x) else np.nan for x in df_mean[metric].values])
        errs = np.zeros_like(vals)
        if df_std is not None and not df_std.empty and metric in df_std.columns:
            errs = np.where(np.isnan(df_std[metric].values), 0.0, df_std[metric].values)
        valid = ~np.isnan(vals)
        if not np.any(valid):
            continue
        with plt.rc_context(
            {
                "font.family": font,
                "font.size": 12,
                "axes.titlesize": 15,
            }
        ):
            fig, ax = plt.subplots(figsize=(max(6, n_exp * 0.8), 5))
            x = np.arange(n_exp)
            ax.bar(
                x,
                vals,
                yerr=errs,
                capsize=4,
                color=colors[:n_exp],
                edgecolor="gray",
                linewidth=0.5,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(format_exp_xlabels(exps), rotation=45, ha="right")
            mt = _metric_plot_title(metric)
            ax.set_ylabel(_metric_ylabel(metric))
            ax.set_title(f"{prefix}: {mt}", fontsize=15)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(*_ylim_with_errs(vals, errs))
            plt.tight_layout()
            safe_name = metric.replace("/", "_").replace(" ", "_")
            plt.savefig(os.path.join(out_dir, f"{prefix}_{safe_name}.png"), dpi=150)
            plt.close()


def plot_metrics_subplot_figure(
    df_mean: pd.DataFrame,
    df_std: pd.DataFrame,
    out_dir: str,
    prefix: str,
    *,
    figure_title: str,
):
    """One figure: grid of subplots, one subplot per metric (exp comparison + error bars)."""
    if df_mean.empty:
        return
    font = _compare_serif_font_name()
    df_mean, df_std = _slice_plot_metrics(df_mean, df_std)
    if df_mean.empty or len(df_mean.columns) == 0:
        return
    metrics = [c for c in df_mean.columns if df_mean[c].notna().any()]
    if not metrics:
        return
    nrows, ncols = 1, 4
    exps = df_mean.index.tolist()
    n_exp = len(exps)
    colors = _bar_colors_google(n_exp)
    x = np.arange(n_exp)

    rc = {
        "font.family": font,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.6 * ncols + 0.5, 3.9),
            constrained_layout=True,
        )
        axes_arr = np.atleast_1d(axes).ravel()
        for idx, metric in enumerate(metrics):
            ax = axes_arr[idx]
            vals = np.array(
                [float(v) if is_num(v) else np.nan for v in df_mean[metric].values]
            )
            errs = np.zeros_like(vals)
            if df_std is not None and not df_std.empty and metric in df_std.columns:
                errs = np.nan_to_num(df_std[metric].values, nan=0.0)
            if not np.any(~np.isnan(vals)):
                ax.set_visible(False)
                continue
            ax.bar(
                x,
                vals,
                yerr=errs,
                capsize=3,
                color=colors[:n_exp],
                edgecolor="gray",
                linewidth=0.5,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(format_exp_xlabels(exps), rotation=35, ha="right")
            disp = _metric_plot_title(metric)
            ax.set_title(disp, fontsize=14)
            ax.set_ylabel(_metric_ylabel(metric))
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(*_ylim_with_errs(vals, errs))
        for j in range(len(metrics), len(axes_arr)):
            axes_arr[j].set_visible(False)
        fig.suptitle(figure_title, fontsize=18, fontweight="bold")
        base = os.path.join(out_dir, f"{prefix}_subplots")
        plt.savefig(f"{base}.png", dpi=200)
        plt.savefig(f"{base}.pdf")
        plt.close()


def plot_combined_metrics_grid_3x4(
    blocks: list[tuple[pd.DataFrame, pd.DataFrame, str]],
    out_dir: str,
    *,
    figure_title: str,
    basename: str = "combined_logreplay_seeds_rewards_subplots",
    replay_minus_reactive: bool = False,
):
    """One 3×4 figure: row = logreplay / unseen seeds / unseen rewards; col = four plot metrics.

    If ``replay_minus_reactive`` and experiment names form a full ``replay_<v>`` / ``reactive_<v>``
    grid, each subplot uses one bar per variant: ``μ_replay − μ_reactive`` (error bars from combined std).
    """
    if len(blocks) != 3:
        return
    font = _compare_serif_font_name()
    nrows, ncols = 3, 4

    all_exps: set[str] = set()
    for df_m, df_s, _ in blocks:
        dm, _ = _slice_plot_metrics(df_m, df_s)
        if not dm.empty:
            all_exps.update(str(e) for e in dm.index)
    exps_sorted = sorted(all_exps)
    layout = _latex_replay_reactive_variant_group_layout(exps_sorted)
    use_rr_diff = bool(replay_minus_reactive and layout is not None)
    if replay_minus_reactive and layout is None:
        print(
            "Warning: replay-minus-reactive combined plot requested but experiment names are not a "
            "full replay_<v> / reactive_<v> grid; plotting raw per-exp bars instead."
        )
    pairs_rr = _rr_variant_pairs(layout[0]) if use_rr_diff and layout else None
    variant_slugs = layout[2] if use_rr_diff and layout else None

    rc = {
        "font.family": font,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.5 * ncols + 0.5, 3.35 * nrows + 1.0),
            constrained_layout=True,
        )
        for r, (df_m, df_s, row_lbl) in enumerate(blocks):
            df_m_sliced, df_s_sliced = _slice_plot_metrics(df_m, df_s)
            if df_m_sliced.empty or not len(df_m_sliced.columns):
                for k in range(ncols):
                    axes[r, k].set_visible(False)
                continue
            if use_rr_diff and pairs_rr is not None and variant_slugs is not None:
                n_exp = len(pairs_rr)
                x = np.arange(n_exp)
                xtick_labels = [_format_type_name(s) for s in variant_slugs]
            else:
                exps = df_m_sliced.index.tolist()
                n_exp = len(exps)
                x = np.arange(n_exp)
                xtick_labels = format_exp_xlabels(exps)
            colors = _bar_colors_google(max(n_exp, 1))
            for k in range(ncols):
                ax = axes[r, k]
                base = PLOT_METRICS_ORDER[k]
                metric = _metric_column_for_base(df_m_sliced, base)
                if metric is None or metric not in df_m_sliced.columns:
                    ax.set_visible(False)
                    continue
                if use_rr_diff and pairs_rr is not None:
                    vals_list: list[float] = []
                    errs_list: list[float] = []
                    for er, ee in pairs_rr:
                        dvm, dvs = _diff_mean_std(df_m_sliced, df_s_sliced, er, ee, base)
                        vals_list.append(float(dvm) if dvm is not None else np.nan)
                        errs_list.append(float(dvs) if dvs is not None else 0.0)
                    vals = np.array(vals_list, dtype=float)
                    errs = np.array(errs_list, dtype=float)
                else:
                    vals = np.array(
                        [float(v) if is_num(v) else np.nan for v in df_m_sliced[metric].values]
                    )
                    errs = np.zeros_like(vals)
                    if df_s_sliced is not None and not df_s_sliced.empty and metric in df_s_sliced.columns:
                        errs = np.nan_to_num(df_s_sliced[metric].values, nan=0.0)
                if not np.any(~np.isnan(vals)):
                    ax.set_visible(False)
                    continue
                ax.bar(
                    x,
                    vals,
                    yerr=errs,
                    capsize=3,
                    color=colors[:n_exp],
                    edgecolor="gray",
                    linewidth=0.5,
                )
                if use_rr_diff:
                    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
                ax.set_xticks(x)
                rot = 18 if use_rr_diff else 35
                ax.set_xticklabels(xtick_labels, rotation=rot, ha="right")
                disp = _metric_plot_title(metric)
                ax.set_title(disp, fontsize=14)
                ylab = _metric_ylabel(metric)
                y_prefix = "Δ (replay−reactive)\n" if use_rr_diff else ""
                if k == 0:
                    ax.set_ylabel(f"{row_lbl}\n{y_prefix}{ylab}", fontsize=11)
                else:
                    ax.set_ylabel(f"{y_prefix}{ylab}")
                ax.grid(axis="y", alpha=0.3)
                ax.set_ylim(*_ylim_with_errs(vals, errs))

        fig.suptitle(figure_title, fontsize=18, fontweight="bold")
        path_base = os.path.join(out_dir, basename)
        plt.savefig(f"{path_base}.png", dpi=200)
        plt.savefig(f"{path_base}.pdf")
        plt.close()


# Short column headers (legacy / other tools); LaTeX combined table uses full names + arrows below.
_METRIC_LATEX_HEADER = {
    "collision_per_agent": "Coll.",
    "offroad_per_agent": "Off.",
    "score": "Succ.",
    "lane_alignment_rate": "Lane",
}

# LaTeX table: full metric names with “better” direction (↑ higher, ↓ lower).
_METRIC_LATEX_FULL_ROW = {
    "collision_per_agent": (r"Collision per agent", r"$\downarrow$"),
    "offroad_per_agent": (r"Off-road per agent", r"$\downarrow$"),
    "score": (r"Success score", r"$\uparrow$"),
    "lane_alignment_rate": (r"Lane alignment rate", r"$\uparrow$"),
}


def _metric_higher_is_better(base: str) -> bool:
    """True if larger mean is better (score, lane alignment); else lower is better."""
    return base in ("score", "lane_alignment_rate")


def _metric_latex_row_label(base: str) -> str:
    """Second-column header: full name + direction arrow (math)."""
    if base in _METRIC_LATEX_FULL_ROW:
        name, arr = _METRIC_LATEX_FULL_ROW[base]
        return f"\\textbf{{{name} {arr}}}"
    name = _latex_escape(str(base).replace("_", " ").title())
    return f"\\textbf{{{name}}}"


def _row_best_exps(
    dm: pd.DataFrame,
    exps: list[str],
    base: str,
    *,
    atol: float = 1e-9,
) -> set[str]:
    """Within this row (one block $\\times$ one metric), bold experiment(s) with best mean only.

    Collision / off-road: lower mean is better. Score / lane alignment: higher mean is better.
    Ties within ``atol`` on the mean are all included.
    """
    maximize = _metric_higher_is_better(base)
    pairs: list[tuple[str, float]] = []
    for e in exps:
        col = _metric_column_for_base(dm, base)
        if col is None or e not in dm.index:
            continue
        m = dm.loc[e, col]
        if not is_num(m):
            continue
        pairs.append((e, float(m)))
    if not pairs:
        return set()
    if maximize:
        best = max(m for _, m in pairs)
        return {e for e, m in pairs if m + atol >= best}
    best = min(m for _, m in pairs)
    return {e for e, m in pairs if m - atol <= best}


def _latex_escape(s: str) -> str:
    """Escape special characters for LaTeX text mode."""
    t = str(s)
    out: list[str] = []
    for ch in t:
        if ch == "\\":
            out.append("\\textbackslash{}")
        elif ch == "&":
            out.append("\\&")
        elif ch == "%":
            out.append("\\%")
        elif ch == "$":
            out.append("\\$")
        elif ch == "#":
            out.append("\\#")
        elif ch == "_":
            out.append("\\_")
        elif ch == "{":
            out.append("\\{")
        elif ch == "}":
            out.append("\\}")
        elif ch == "~":
            out.append("\\textasciitilde{}")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


def _latex_exp_column_headers(exps: list[str]) -> list[str]:
    """LaTeX table column titles: ``Replay`` / ``Reactive`` when exps are only those families (no selfplay)."""
    families = [_exp_replay_reactive_family(e) for e in exps]
    if not exps or any(f is None for f in families):
        return [_latex_escape(e) for e in exps]
    short_base = {"replay": "Replay", "reactive": "Reactive"}
    labels = [short_base[f] for f in families]
    counts = defaultdict(int)
    for lb in labels:
        counts[lb] += 1
    out: list[str] = []
    for exp, lb in zip(exps, labels):
        if counts[lb] > 1:
            dis = _disambiguated_family_label(exp)
            out.append(_latex_escape(dis if dis else lb))
        else:
            out.append(_latex_escape(lb))
    return out


def _block_title_latex(row_lbl: str) -> str:
    """Map plot row labels to concise LaTeX multicolumn titles."""
    key = str(row_lbl).strip()
    if key == "Log replay":
        return "Log replay"
    if "seeds" in key.lower():
        return "Unseen other (seeds)"
    if "rewards" in key.lower():
        return "Unseen other (rewards)"
    return _latex_escape(key)


def _combined_metric_cell(
    dm: pd.DataFrame,
    ds: Optional[pd.DataFrame],
    exp: str,
    base: str,
    *,
    bold: bool = False,
) -> str:
    col = _metric_column_for_base(dm, base)
    if col is None or exp not in dm.index:
        return "\\multicolumn{1}{c}{--}"
    vm = dm.loc[exp, col]
    if not is_num(vm):
        return "\\multicolumn{1}{c}{--}"
    vs = 0.0
    if ds is not None and not ds.empty and col in ds.columns and exp in ds.index:
        vv = ds.loc[exp, col]
        if is_num(vv):
            vs = float(vv)
    s = f"{float(vm):.3f} \\pm {vs:.3f}"
    if bold:
        return f"$\\bm{{{s}}}$"
    return f"${s}$"


def export_combined_metrics_latex_table(
    out_dir: str,
    basename: str,
    blocks: list[tuple[pd.DataFrame, pd.DataFrame, str]],
    *,
    caption: str = "Zero-shot coordination metrics (mean $\\pm$ std across seeds).",
    label: str = "tab:combined_pufferdrive_metrics",
    replay_minus_reactive: bool = False,
) -> Optional[str]:
    """Write a full ``table*`` environment: rows = 3 evaluation blocks $\\times$ 4 metrics (vertical);
    columns = experiments. Metric column uses ``\\textbf``; best-in-row values use ``\\bm{...}`` in math.
    When exps form a full ``replay_<v>`` / ``reactive_<v>`` grid per variant ``v``, columns are grouped
    by ``v`` (nominal before lane, then others) with sub-headers Replay / Reactive; otherwise short
    Replay/Reactive or raw folder names (see ``_latex_exp_column_headers``).
    If ``replay_minus_reactive`` and the grid layout applies, one numeric column per variant:
    $\\mu_{\\mathrm{replay}} - \\mu_{\\mathrm{reactive}}$ with $\\sigma_\\Delta \\approx
    \\sqrt{\\sigma_r^2 + \\sigma_{re}^2}$; best-in-row compares variants on that difference."""
    if len(blocks) != 3:
        return None
    sliced: list[tuple[pd.DataFrame, Optional[pd.DataFrame], str]] = []
    all_exps: set[str] = set()
    for df_m, df_s, row_lbl in blocks:
        dm, ds = _slice_plot_metrics(df_m, df_s)
        sliced.append((dm, ds, row_lbl))
        if not dm.empty:
            all_exps.update(str(e) for e in dm.index)
    if not all_exps:
        return None

    exps_sorted = sorted(all_exps)
    rr_groups = _latex_replay_reactive_variant_group_layout(exps_sorted)
    use_rr_diff = bool(replay_minus_reactive and rr_groups is not None)
    if replay_minus_reactive and rr_groups is None:
        print(
            "Warning: replay-minus-reactive table requested but experiment names are not a "
            "full replay_<v> / reactive_<v> grid; writing absolute per-exp columns instead."
        )

    col_order: Optional[list[str]] = None
    group_tex: Optional[list[str]] = None
    pairs_rr: Optional[list[tuple[str, str]]] = None
    if rr_groups is not None:
        col_order, group_tex, _ = rr_groups
        if use_rr_diff:
            pairs_rr = _rr_variant_pairs(col_order)
    grouped_header = rr_groups is not None and not use_rr_diff
    n_metrics = len(PLOT_METRICS_ORDER)
    if use_rr_diff and pairs_rr is not None:
        n_data_cols = len(pairs_rr)
    elif col_order is not None:
        n_data_cols = len(col_order)
    else:
        n_data_cols = len(exps_sorted)
    col_spec = f"ll*{{{n_data_cols}}}{{c}}"

    lines: list[str] = [
        "% Auto-generated by analyze/compare_exps.py",
        "% Requires: \\usepackage{booktabs}",
        "% Requires: \\usepackage{multirow}",
        "% Requires: \\usepackage{bm}   % best-in-row values use $\\bm{...}$ (\\textbf only on metric names).",
        "% Best mean in that row (collision/off-road: min; score/lane: max); ties get \\bm.",
        "\\begin{table*}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]
    if use_rr_diff and group_tex is not None:
        lines.append(
            "% Columns: $\\Delta = \\mu_{\\mathrm{replay}} - \\mu_{\\mathrm{reactive}}$ "
            "(mean $\\pm$ $\\sqrt{\\sigma_r^2 + \\sigma_{re}^2}$ over seeds)."
        )
        lines.append("Evaluation & Metric & " + " & ".join(group_tex) + " \\\\")
    elif grouped_header and group_tex is not None:
        lines.append(
            "% Columns: grouped by variant (multicolumn); within each group: Replay then Reactive."
        )
        mc = " & ".join(f"\\multicolumn{{2}}{{c}}{{{g}}}" for g in group_tex)
        lines.append(f"Evaluation & Metric & {mc} \\\\")
        cmid = " ".join(
            f"\\cmidrule(lr){{{3 + 2 * i}}}{{{4 + 2 * i}}}" for i in range(len(group_tex))
        )
        lines.append(cmid)
        sub = " & ".join(["Replay", "Reactive"] * len(group_tex))
        lines.append(f" & & {sub} \\\\")
    else:
        col_headers = _latex_exp_column_headers(exps_sorted)
        lines.append("Evaluation & Metric & " + " & ".join(col_headers) + " \\\\")
    lines.append("\\midrule")

    for bi, (dm, ds, row_lbl) in enumerate(sliced):
        block_title = _block_title_latex(row_lbl)
        block_tex = _latex_escape(block_title)
        if use_rr_diff and pairs_rr is not None:
            winners_by_base = {
                base: _row_best_rr_diff(dm, ds, pairs_rr, base) for base in PLOT_METRICS_ORDER
            }
        elif col_order is not None:
            winners_by_base = {
                base: _row_best_exps(dm, col_order, base) for base in PLOT_METRICS_ORDER
            }
        else:
            winners_by_base = {
                base: _row_best_exps(dm, exps_sorted, base) for base in PLOT_METRICS_ORDER
            }
        for mi, base in enumerate(PLOT_METRICS_ORDER):
            mname = _metric_latex_row_label(base)
            if mi == 0:
                c_eval = f"\\multirow{{{n_metrics}}}{{*}}{{{block_tex}}}"
            else:
                c_eval = ""
            win = winners_by_base[base]
            if use_rr_diff and pairs_rr is not None:
                cells = [
                    _combined_diff_cell(
                        dm, ds, er, ee, base, bold=(k in win)
                    )
                    for k, (er, ee) in enumerate(pairs_rr)
                ]
            elif col_order is not None:
                cells = [
                    _combined_metric_cell(dm, ds, exp, base, bold=(exp in win))
                    for exp in col_order
                ]
            else:
                cells = [
                    _combined_metric_cell(dm, ds, exp, base, bold=(exp in win))
                    for exp in exps_sorted
                ]
            lines.append(f"{c_eval} & {mname} & " + " & ".join(cells) + " \\\\")
        if bi < len(sliced) - 1:
            lines.append("\\midrule")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])

    out_path = os.path.join(out_dir, f"{basename}_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser("Compare logreplay, wosac, unseen_other_seeds, unseen_other_rewards")
    parser.add_argument("--base-path", "-b", type=str, default=RESULTS_BASE)
    parser.add_argument(
        "--exps",
        nargs="+",
        default=None,
        help="Specific experiment folder names to compare (e.g. --exps exp_a exp_b). "
             "If omitted, compares all experiments under --base-path.",
    )
    parser.add_argument("--out-dir", "-o", type=str, default=None)
    parser.add_argument(
        "--format", "-f", type=str, default="all",
        choices=["logreplay", "wosac", "unseen_seeds", "unseen_rewards", "both", "all"]
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip bar plot generation")
    parser.add_argument(
        "--combined-rr-diff",
        action="store_true",
        help=(
            "Combined 3×4 figure + LaTeX: one value per variant = replay mean − reactive mean "
            "(requires replay_<v> / reactive_<v> grid). Saves basename *_rr_diff.*; "
            "σ_Δ ≈ sqrt(σ_replay² + σ_reactive²)."
        ),
    )
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
        # One multi-panel figure for these blocks; per-metric PNGs for wosac (large tables).
        subplot_titles = {
            "logreplay": "Log-Replay (LR)",
            "unseen_other_rewards": "Unseen other rewards (UOR)",
            "unseen_other_seeds": "Unseen other seeds (UOS))",
        }
        if name in subplot_titles:
            plot_metrics_subplot_figure(
                df_m,
                df_s,
                out_dir,
                safe,
                figure_title=subplot_titles[name],
            )
        else:
            plot_bar_comparison(df_m, df_s, out_dir, safe)


if __name__ == "__main__":
    args = parse_args()
    lr, wosac, unseen_seeds, unseen_rewards = collect_exp_results(
        args.base_path, exp_names=args.exps
    )

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

    if (
        not args.no_plot
        and lr
        and unseen_seeds
        and unseen_rewards
    ):
        df_lm, df_ls = build_comparison_dfs(lr, "exp")
        df_sm, df_ss = build_comparison_dfs(unseen_seeds, "exp")
        df_rm, df_rs = build_comparison_dfs(unseen_rewards, "exp")
        combined_blocks = [
            (df_lm, df_ls, "Log replay"),
            (df_sm, df_ss, "Unseen other · seeds"),
            (df_rm, df_rs, "Unseen other · rewards"),
        ]
        _fig_title = "Zero-shot Coordination Results in PufferDrive"
        _basename = "combined_logreplay_seeds_rewards_subplots"
        _rr_diff = bool(getattr(args, "combined_rr_diff", False))
        if _rr_diff:
            _basename += "_rr_diff"
            _fig_title += r" (Replay $-$ Reactive, per variant)"
        if _rr_diff:
            _cap = (
                _fig_title
                + r" ($\Delta = \mu_{\mathrm{replay}} - \mu_{\mathrm{reactive}}$ "
                r"mean $\pm$ $\sqrt{\sigma_r^2 + \sigma_{re}^2}$)."
            )
        else:
            _cap = f"{_fig_title} (mean $\\pm$ std across seeds)."
        plot_combined_metrics_grid_3x4(
            combined_blocks,
            out_dir,
            figure_title=_fig_title,
            basename=_basename,
            replay_minus_reactive=_rr_diff,
        )
        print(f"\nSaved: {out_dir}/{_basename}.png, .pdf")
        tex_path = export_combined_metrics_latex_table(
            out_dir,
            _basename,
            combined_blocks,
            caption=_cap,
            replay_minus_reactive=_rr_diff,
        )
        if tex_path:
            print(f"Saved: {tex_path}")

    if not any([lr, wosac, unseen_seeds, unseen_rewards]):
        print(f"No results found under {args.base_path}")
