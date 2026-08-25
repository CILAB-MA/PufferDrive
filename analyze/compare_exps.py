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
    # "lane_alignment_rate",
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
        empty = pd.DataFrame(index=df_mean.index)
        if "n_seeds" in getattr(df_mean, "attrs", {}):
            empty.attrs["n_seeds"] = pd.DataFrame(index=df_mean.index)
        return empty, None
    dm = df_mean[cols].copy()
    n_src = getattr(df_mean, "attrs", {}).get("n_seeds")
    if isinstance(n_src, pd.DataFrame) and not n_src.empty:
        dn = pd.DataFrame(index=n_src.index)
        for c in cols:
            dn[c] = n_src[c] if c in n_src.columns else 0
        dm.attrs["n_seeds"] = dn
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


def _is_selfplay_exp(exp: str) -> bool:
    """True for the self-play baseline folder (``selfplay`` / ``*_selfplay``)."""
    s = os.path.basename(str(exp).strip().rstrip("/")).lower()
    return s == "selfplay" or s.endswith("_selfplay") or s.startswith("selfplay_")


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


_PBT_STRATEGY_DISPLAY = {
    "curriculum": "Curriculum",
    "uniform": "Uniform",
    "prioritized": "Prioritized",
    "plr": "Prioritized",
}

# ``record-curriculum-popul_curriculum-wandb``, ``reactive-uniform-popul_lane_nominal-wandb``
_PBT_TRAIN_FOLDER_RE = re.compile(
    r"(?i)^(record|replay|reactive)[-_](curriculum|prioritized|uniform|plr)"
    r"(?:-popul_[A-Za-z0-9_]+)?(?:-wandb)?$"
)


def _parse_pbt_exp_name(exp: str) -> Optional[tuple[str, str]]:
    """Train-folder names → ``(kind, strategy)``. ``kind`` is ``replay`` or ``reactive``.

    ``record`` / ``replay`` both map to ``replay`` so ReCord vs Reactive-PBT pairing still works.
    """
    s = os.path.basename(str(exp).strip().rstrip("/"))
    s = re.sub(r"(?:_\d+(?:\.\d+)?)+$", "", s)
    if "selfplay" in s.lower() or s.lower().startswith("logreplay"):
        return None
    m = _PBT_TRAIN_FOLDER_RE.match(s)
    if not m:
        return None
    mode = m.group(1).lower()
    strategy = m.group(2).lower()
    if strategy == "plr":
        strategy = "prioritized"
    kind = "reactive" if mode == "reactive" else "replay"
    return (kind, strategy)


def _pbt_mode_display(kind: str) -> str:
    return "Reactive-PBT" if kind == "reactive" else "ReCord"


def _pbt_strategy_display(strategy: str) -> str:
    slug = str(strategy).lower()
    if slug == _SELFPLAY_SLUG or slug == "self-play":
        return "Self-play"
    if slug in _PBT_STRATEGY_DISPLAY:
        return _PBT_STRATEGY_DISPLAY[slug]
    return _format_type_name(strategy)


def format_exp_xlabel(exp: str) -> str:
    """X-axis tick labels for subplot/bar figures.

    Train folders ``record|reactive-<strategy>-popul_*-wandb`` → ``ReCord|Reactive-PBT (<Strategy>)``.
    Legacy ``reactive_<v>`` / ``replay_<v>`` → ``Reactive-PBT|ReCord (<V>)``.
    ``selfplay`` → ``SP``; ``logreplay`` is never rewritten as *replay*.
    ``0.25`` / ``0_25`` (Lane+Nominal mix) → ``(L+N)`` in the label (or trailing `` (L+N)`` if stripped).
    Non-matching names still get generic ``reactive`` / ``replay`` / ``record`` word replacements.
    """
    orig = str(exp)
    parsed = _parse_pbt_exp_name(orig)
    if parsed is not None:
        kind, strategy = parsed
        return _apply_ln_mix_display(
            f"{_pbt_mode_display(kind)} ({_pbt_strategy_display(strategy)})", orig
        )
    s = orig
    s = re.sub(r"(?:_\d+(?:\.\d+)?)+$", "", s)
    if re.search("selfplay", s, re.I):
        return "Self-play"
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
    s = re.sub(r"(?i)(?<!log)record", "ReCord", s)
    s = re.sub(r"reactive", "Reactive-PBT", s, flags=re.I)
    s = re.sub(r"(?i)curriculum", "Curriculum", s)
    s = re.sub(r"(?i)prioritized|\bplr\b", "Prioritized", s)
    s = re.sub(r"(?i)uniform", "Uniform", s)
    return _apply_ln_mix_display(s, orig)


def _format_type_name(type_name: str) -> str:
    """Normalize variant tokens: snake_case -> 'Snake Case'."""
    toks = [t for t in str(type_name).split("_") if t]
    return " ".join(t[:1].upper() + t[1:] for t in toks)


def _disambiguated_family_label(exp: str) -> Optional[str]:
    """When tick labels collide, use same style as ``format_exp_xlabel`` (variant in parentheses)."""
    orig = str(exp)
    parsed = _parse_pbt_exp_name(orig)
    if parsed is not None:
        kind, strategy = parsed
        return _apply_ln_mix_display(
            f"{_pbt_mode_display(kind)} ({_pbt_strategy_display(strategy)})", orig
        )
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
    """``replay`` or ``reactive`` if folder matches record/replay/reactive train names; else None."""
    parsed = _parse_pbt_exp_name(exp)
    if parsed is not None:
        return parsed[0]
    s = str(exp)
    if "selfplay" in s.lower():
        return None
    s = re.sub(r"(?:_\d+(?:\.\d+)?)+$", "", s)
    if re.search(r"(?:^|[_-])reactive[_-]", s):
        return "reactive"
    if re.search(r"(?:^|[_-])(?<!log)(?:replay|record)[_-]", s):
        return "replay"
    return None


def _replay_reactive_variant_kind(exp: str) -> Optional[tuple[str, str]]:
    """Parse train names into ``(variant_slug, 'replay'|'reactive')``.

    Hyphen folders use strategy (curriculum / uniform / prioritized) as the slug.
    Legacy ``replay_nominal`` still uses the name tail.
    """
    parsed = _parse_pbt_exp_name(exp)
    if parsed is not None:
        kind, strategy = parsed
        return (strategy, kind)
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
_LATEX_VARIANT_GROUP_PRIORITY = ("curriculum", "prioritized", "uniform", "nominal", "lane")
# Plot groups: Uniform / Curriculum / Prioritized on the big axis; Reactive then ReCord inside.
_PLOT_VARIANT_GROUP_PRIORITY = ("uniform", "curriculum", "prioritized", "nominal", "lane")
_PLOT_KIND_ORDER = {"reactive": 0, "replay": 1}
_PLOT_KIND_DISPLAY = {"reactive": "Reactive-PBT", "replay": "ReCord"}
_PLOT_KIND_COLORS = {"reactive": GOOGLE_PALETTE[0], "replay": GOOGLE_PALETTE[1]}
_SELFPLAY_BAR_COLOR = GOOGLE_PALETTE[7]  # gray baseline bar
_SELFPLAY_SLUG = "selfplay"


def _latex_variant_group_sort_key(slug: str) -> tuple[int, str]:
    if slug in _LATEX_VARIANT_GROUP_PRIORITY:
        return (_LATEX_VARIANT_GROUP_PRIORITY.index(slug), slug)
    return (len(_LATEX_VARIANT_GROUP_PRIORITY), slug)


def _plot_variant_group_sort_key(slug: str) -> tuple[int, str]:
    if slug in _PLOT_VARIANT_GROUP_PRIORITY:
        return (_PLOT_VARIANT_GROUP_PRIORITY.index(slug), slug)
    return (len(_PLOT_VARIANT_GROUP_PRIORITY), slug)


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
        if _is_selfplay_exp(e):
            continue
        vk = _replay_reactive_variant_kind(e)
        if vk is None:
            return None
        parsed[e] = vk
    if not parsed:
        return None
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
    non_sp = [str(e) for e in exps if not _is_selfplay_exp(e)]
    if len(col_order) != len(non_sp):
        return None
    return (col_order, group_tex, variant_slugs)


def _pick_selfplay_exp(exps) -> Optional[str]:
    """Prefer folder named exactly ``selfplay`` when several match."""
    sp_names = [str(e) for e in exps if _is_selfplay_exp(str(e))]
    if not sp_names:
        return None
    return next((e for e in sp_names if e.lower() == "selfplay"), sp_names[0])


def _ordered_exps_for_plot(exps: list[str]) -> list[str]:
    """Fallback bar order: Uniform/Curriculum/Prioritized (Reactive then ReCord), Self-play last."""
    sp = _pick_selfplay_exp(exps)
    names = [str(e) for e in exps if not _is_selfplay_exp(e)]
    if not names:
        return [sp] if sp else []
    keyed: list[tuple[tuple[int, str], int, str]] = []
    for e in names:
        vk = _replay_reactive_variant_kind(e)
        if vk is None:
            keyed.append((_plot_variant_group_sort_key("~"), 99, e))
            continue
        slug, kind = vk
        keyed.append((_plot_variant_group_sort_key(slug), _PLOT_KIND_ORDER.get(kind, 2), e))
    keyed.sort()
    ordered = [row[-1] for row in keyed]
    if sp:
        ordered.append(sp)
    return ordered


def _strategy_kind_groups(
    exps,
) -> Optional[tuple[list[str], list[str], dict[str, dict[str, str]], Optional[str]]]:
    """Parse experiments into (strategy slugs, kinds, slug→kind→exp, selfplay_exp|None).

    Returns None if the non-selfplay exps are not a Reactive/ReCord strategy grid.
    """
    by_var: dict[str, dict[str, str]] = defaultdict(dict)
    sp_exp = _pick_selfplay_exp(exps)
    for e in exps:
        if _is_selfplay_exp(str(e)):
            continue
        vk = _replay_reactive_variant_kind(str(e))
        if vk is None:
            return None
        slug, kind = vk
        if kind in by_var[slug]:
            return None
        by_var[slug][kind] = str(e)
    if not by_var:
        return None
    slugs = sorted(by_var.keys(), key=_plot_variant_group_sort_key)
    kinds = [k for k in ("reactive", "replay") if any(k in by_var[s] for s in slugs)]
    if len(slugs) < 1 or len(kinds) < 2:
        return None
    return slugs, kinds, {s: dict(by_var[s]) for s in slugs}, sp_exp


def _metric_val_err(
    df_mean: pd.DataFrame,
    df_std: Optional[pd.DataFrame],
    exp: str,
    metric: str,
) -> tuple[float, float]:
    if (
        exp not in df_mean.index
        or metric not in df_mean.columns
        or not is_num(df_mean.loc[exp, metric])
    ):
        return float("nan"), 0.0
    val = float(df_mean.loc[exp, metric])
    err = 0.0
    if (
        df_std is not None
        and not df_std.empty
        and metric in df_std.columns
        and exp in df_std.index
        and is_num(df_std.loc[exp, metric])
    ):
        err = float(df_std.loc[exp, metric])
    return val, err


def _draw_grouped_rr_bars(
    ax,
    df_mean: pd.DataFrame,
    df_std: Optional[pd.DataFrame],
    metric: str,
    slugs: list[str],
    kinds: list[str],
    by_var: dict[str, dict[str, str]],
    sp_exp: Optional[str] = None,
):
    """Grouped bars: x = strategy (+ Self-play), hue = Reactive-PBT then ReCord.

    Self-play (if present) is a single bar on its own x-tick after the strategy groups.
    """
    has_sp = bool(sp_exp) and sp_exp in df_mean.index
    n_strat = len(slugs)
    x = np.arange(n_strat, dtype=float)
    n_k = len(kinds)
    width = 0.36 if n_k == 2 else max(0.18, 0.8 / max(n_k, 1))
    all_vals: list[float] = []
    all_errs: list[float] = []
    for i, kind in enumerate(kinds):
        vals = []
        errs = []
        for slug in slugs:
            exp = by_var.get(slug, {}).get(kind)
            if exp is None:
                vals.append(np.nan)
                errs.append(0.0)
            else:
                v, e = _metric_val_err(df_mean, df_std, exp, metric)
                vals.append(v)
                errs.append(e)
        offset = (i - (n_k - 1) / 2.0) * width
        ax.bar(
            x + offset,
            vals,
            width,
            yerr=errs,
            capsize=3,
            color=_PLOT_KIND_COLORS.get(kind, GOOGLE_PALETTE[i % len(GOOGLE_PALETTE)]),
            edgecolor="gray",
            linewidth=0.5,
            label=_PLOT_KIND_DISPLAY.get(kind, kind),
        )
        all_vals.extend(vals)
        all_errs.extend(errs)
    tick_pos = list(x)
    tick_labels = [_pbt_strategy_display(s) for s in slugs]
    if has_sp:
        sp_x = float(n_strat)
        sp_val, sp_err = _metric_val_err(df_mean, df_std, sp_exp, metric)
        ax.bar(
            [sp_x],
            [sp_val],
            width=min(0.5, width * 1.25),
            yerr=[sp_err],
            capsize=3,
            color=_SELFPLAY_BAR_COLOR,
            edgecolor="gray",
            linewidth=0.5,
            label="Self-play",
        )
        all_vals.append(sp_val)
        all_errs.append(sp_err)
        tick_pos.append(sp_x)
        tick_labels.append("Self-play")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    return np.asarray(all_vals, dtype=float), np.asarray(all_errs, dtype=float)


def _reindex_plot_exps(df_mean: pd.DataFrame, df_std: Optional[pd.DataFrame]):
    order = _ordered_exps_for_plot(list(df_mean.index.astype(str)))
    order = [e for e in order if e in df_mean.index]
    leftover = [e for e in df_mean.index.astype(str) if e not in order]
    order = order + leftover
    dm = df_mean.reindex(order)
    ds = df_std
    if df_std is not None and not df_std.empty:
        ds = df_std.reindex(order)
    return dm, ds


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

    Returns ``(mean_dict, std_dict, n_dict)``.
    ``std`` uses population std (``ddof=0``), matching prior CSVs; Welch tests convert to sample std.
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
    n_d = {}
    for k, vs in bucket.items():
        if vs:
            arr = np.array(vs)
            mean_d[k] = float(np.mean(arr))
            std_d[k] = float(np.std(arr)) if len(vs) > 1 else 0.0
            n_d[k] = int(len(vs))
    return mean_d, std_d, n_d


def _logreplay_key_filter(k):
    return k.startswith("ego_")


def _wosac_key_filter(k):
    return k != "num_agents"


def aggregate_zeroshot_matchups(entries, key_filter=None):
    """Aggregate zeroshot_reactive results.

    Keys are {ego}_vs_{other}. First: mean over others per ego (per seed).
    Then: mean and std over egos (seeds).
    Returns ``(mean_dict, std_dict, n_dict)``.
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
        return {}, {}, {}
    all_metrics = set()
    for row in per_ego_means:
        all_metrics.update(row.keys())
    mean_d = {}
    std_d = {}
    n_d = {}
    for k in all_metrics:
        vals = [r[k] for r in per_ego_means if k in r and is_num(r[k])]
        if vals:
            arr = np.array(vals)
            mean_d[k] = float(np.mean(arr))
            std_d[k] = float(np.std(arr)) if len(vals) > 1 else 0.0
            n_d[k] = int(len(vals))
    return mean_d, std_d, n_d


_UNSEEN_REACTIVE_JSON = "zeroshot_reactive.json"
_UNSEEN_REPLAY_JSON = "zeroshot.json"
_UNSEEN_REPLAY_JSON_ALT = "zeroshot_replay.json"  # e.g. selfplay folder naming


def _ego_ids_from_zeroshot_entries(entries) -> set:
    ids = set()
    for d in entries:
        if not isinstance(d, dict):
            continue
        for k in d:
            if "_vs_" in k:
                ids.add(k.split("_vs_", 1)[0])
    return ids


def _ego_ids_for_unseen_exp(exp: str, exp_dir: str, unseen_mode: str) -> set:
    """Ego wandb ids for this experiment (json keys, scenario logs, checkpoints)."""
    ids: set = set()
    for fname in (_UNSEEN_REACTIVE_JSON, _UNSEEN_REPLAY_JSON, _UNSEEN_REPLAY_JSON_ALT):
        ids |= _ego_ids_from_zeroshot_entries(
            load_json_list(os.path.join(exp_dir, unseen_mode, fname))
        )
    log_dir = os.path.join(exp_dir, unseen_mode, "scenario_logs")
    if os.path.isdir(log_dir):
        for name in os.listdir(log_dir):
            m = re.search(r"(?:pbt_)?([A-Za-z0-9]{8})_vs_", name)
            if m:
                ids.add(m.group(1))
    ckpt_dir = os.path.join("/data/puffer/experiments", exp)
    if os.path.isdir(ckpt_dir):
        for name in os.listdir(ckpt_dir):
            if name.startswith("puffer_drive_") and name.endswith(".pt"):
                stem = name[len("puffer_drive_") : -3]
                ids.add(stem)
                if len(stem) >= 8:
                    ids.add(stem[-8:])
    return ids


def _filter_zeroshot_entries_by_ego(entries, ego_ids: set):
    if not ego_ids:
        return []
    out = []
    for d in entries:
        if not isinstance(d, dict):
            continue
        kept = {
            k: v
            for k, v in d.items()
            if "_vs_" in k and k.split("_vs_", 1)[0] in ego_ids
        }
        if kept:
            out.append(kept)
    return out


def _load_unseen_replay_entries(exp: str, exp_dir: str, unseen_mode: str, base_path: str, ego_ids: set):
    """Replay-eval matchups: local ``zeroshot.json`` / ``zeroshot_replay.json``, else shared dump."""
    for fname in (_UNSEEN_REPLAY_JSON, _UNSEEN_REPLAY_JSON_ALT):
        local = load_json_list(os.path.join(exp_dir, unseen_mode, fname))
        if local:
            return local
    for fname in (_UNSEEN_REPLAY_JSON, _UNSEEN_REPLAY_JSON_ALT):
        shared = load_json_list(os.path.join(base_path, unseen_mode, fname))
        if shared:
            return _filter_zeroshot_entries_by_ego(shared, ego_ids)
    return []


def _load_unseen_eval_by_protocol(exp: str, exp_dir: str, unseen_mode: str, base_path: str):
    """Load both zero-shot eval protocols for one exp × unseen corpus.

    Returns ``{"reactive-play": entries, "replay": entries}`` (missing protocol → []).
    """
    ego_ids = _ego_ids_for_unseen_exp(exp, exp_dir, unseen_mode)
    return {
        "reactive-play": load_json_list(
            os.path.join(exp_dir, unseen_mode, _UNSEEN_REACTIVE_JSON)
        ),
        "replay": _load_unseen_replay_entries(
            exp, exp_dir, unseen_mode, base_path, ego_ids
        ),
    }


def collect_exp_results(base_path: str, exp_names=None):
    """Collect logreplay, wosac, and unseen (seeds/rewards × replay/reactive-play) by exp.

    Returns
    -------
    logreplay, wosac,
    unseen_seeds_reactive, unseen_seeds_replay,
    unseen_rewards_reactive, unseen_rewards_replay
    """
    logreplay_by_exp = {}
    wosac_by_exp = {}
    unseen_seeds_reactive = {}
    unseen_seeds_replay = {}
    unseen_rewards_reactive = {}
    unseen_rewards_replay = {}

    empty = (
        logreplay_by_exp,
        wosac_by_exp,
        unseen_seeds_reactive,
        unseen_seeds_replay,
        unseen_rewards_reactive,
        unseen_rewards_replay,
    )
    if not os.path.isdir(base_path):
        return empty

    if exp_names:
        exp_iter = list(exp_names)
    else:
        exp_iter = sorted(os.listdir(base_path))

    unseen_outs = {
        ("unseen_other_seeds", "reactive-play"): unseen_seeds_reactive,
        ("unseen_other_seeds", "replay"): unseen_seeds_replay,
        ("unseen_other_rewards", "reactive-play"): unseen_rewards_reactive,
        ("unseen_other_rewards", "replay"): unseen_rewards_replay,
    }

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

        # unseen: both eval protocols (frozen-other replay + live reactive-play)
        for corpus in ("unseen_other_seeds", "unseen_other_rewards"):
            by_proto = _load_unseen_eval_by_protocol(exp, exp_dir, corpus, base_path)
            for proto, entries in by_proto.items():
                if not entries:
                    continue
                unseen_outs[(corpus, proto)][exp] = aggregate_zeroshot_matchups(
                    entries, key_filter=_logreplay_key_filter
                )

    return empty


def build_comparison_dfs(data_by_exp: dict, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build DataFrames for mean and std. data_by_exp[exp] = (mean_dict, std_dict[, n_dict]).

    Sample counts are attached as ``df_mean.attrs['n_seeds']`` (DataFrame, same index/columns).
    """
    if not data_by_exp:
        return pd.DataFrame(), pd.DataFrame()
    all_metrics = set()
    for val in data_by_exp.values():
        mean_d = val[0] if isinstance(val, tuple) else val
        all_metrics.update(mean_d.keys())
    all_metrics = sorted(all_metrics)

    rows_mean, rows_std, rows_n = [], [], []
    for exp in sorted(data_by_exp.keys()):
        val = data_by_exp[exp]
        if isinstance(val, tuple):
            mean_d = val[0]
            std_d = val[1] if len(val) > 1 else {}
            n_d = val[2] if len(val) > 2 else {}
        else:
            mean_d, std_d, n_d = val, {}, {}
        row_m, row_s, row_n = {"exp": exp}, {"exp": exp}, {"exp": exp}
        for m in all_metrics:
            vm = mean_d.get(m)
            vs = std_d.get(m) if std_d else 0.0
            vn = n_d.get(m) if n_d else None
            row_m[m] = float(vm) if is_num(vm) else None
            row_s[m] = float(vs) if is_num(vs) else (0.0 if vm is not None else None)
            row_n[m] = int(vn) if vn is not None and is_num(vn) else (
                None if vm is None else 0
            )
        rows_mean.append(row_m)
        rows_std.append(row_s)
        rows_n.append(row_n)

    df_m = pd.DataFrame(rows_mean).set_index("exp")
    df_s = pd.DataFrame(rows_std).set_index("exp")
    df_n = pd.DataFrame(rows_n).set_index("exp")
    df_m.index.name = df_s.index.name = df_n.index.name = label
    df_m.attrs["n_seeds"] = df_n
    return df_m, df_s


def plot_bar_comparison(df_mean: pd.DataFrame, df_std: pd.DataFrame, out_dir: str, prefix: str):
    """Draw one bar chart per metric: EXPs on x-axis, value on y-axis, with std error bars.
    Y-axis is zoomed to highlight differences (data range + margin).
    Self-play (if present) is its own x-tick bar after the strategy groups.
    """
    if df_mean.empty:
        return
    font = _compare_serif_font_name()
    df_mean, df_std = _slice_plot_metrics(df_mean, df_std)
    if df_mean.empty or len(df_mean.columns) == 0:
        return
    groups = _strategy_kind_groups(df_mean.index)
    if groups is None:
        df_mean, df_std = _reindex_plot_exps(df_mean, df_std)
    metrics = [c for c in df_mean.columns if df_mean[c].notna().any()]
    if not metrics:
        return
    exps = df_mean.index.tolist()
    n_exp = len(exps)
    colors = _bar_colors_google(n_exp)
    n_x = (len(groups[0]) + (1 if groups[3] else 0)) if groups is not None else n_exp

    for metric in metrics:
        with plt.rc_context(
            {
                "font.family": font,
                "font.size": 12,
                "axes.titlesize": 15,
            }
        ):
            fig, ax = plt.subplots(figsize=(max(6, n_x * 0.95), 5))
            if groups is not None:
                slugs, kinds, by_var, sp_exp = groups
                vals, errs = _draw_grouped_rr_bars(
                    ax, df_mean, df_std, metric, slugs, kinds, by_var, sp_exp=sp_exp
                )
            else:
                vals = np.array([float(x) if is_num(x) else np.nan for x in df_mean[metric].values])
                errs = np.zeros_like(vals)
                if df_std is not None and not df_std.empty and metric in df_std.columns:
                    errs = np.where(np.isnan(df_std[metric].values), 0.0, df_std[metric].values)
                valid = ~np.isnan(vals)
                if not np.any(valid):
                    plt.close()
                    continue
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
            if not np.any(~np.isnan(vals)):
                plt.close()
                continue
            ax.legend(frameon=False)
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
    include_selfplay: bool = True,
):
    """One figure: 4 metric panels. X = Uniform / Curriculum / Prioritized [/ Self-play];
    grouped bars = Reactive-PBT then ReCord (Self-play is a single bar when enabled).
    """
    if df_mean.empty:
        return
    font = _compare_serif_font_name()
    df_mean, df_std = _slice_plot_metrics(df_mean, df_std)
    if df_mean.empty or len(df_mean.columns) == 0:
        return
    groups = _strategy_kind_groups(df_mean.index)
    if groups is None:
        df_mean, df_std = _reindex_plot_exps(df_mean, df_std)
        if not include_selfplay:
            keep = [e for e in df_mean.index if not _is_selfplay_exp(str(e))]
            df_mean = df_mean.reindex(keep)
            if df_std is not None and not df_std.empty:
                df_std = df_std.reindex(keep)
    metrics = [c for c in df_mean.columns if df_mean[c].notna().any()]
    if not metrics:
        return
    nrows, ncols = 1, len(PLOT_METRICS_ORDER)
    exps = df_mean.index.tolist()
    n_exp = len(exps)
    colors = _bar_colors_google(n_exp)
    x = np.arange(n_exp)
    sp_for_width = groups[3] if (groups is not None and include_selfplay) else None
    fig_w = 3.6 * ncols + (0.9 if sp_for_width else 0.5)

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
            figsize=(fig_w, 3.4),
            constrained_layout=True,
        )
        axes_arr = np.atleast_1d(axes).ravel()
        legend_ax = None
        for idx, metric in enumerate(metrics):
            ax = axes_arr[idx]
            if groups is not None:
                slugs, kinds, by_var, sp_exp = groups
                if not include_selfplay:
                    sp_exp = None
                vals, errs = _draw_grouped_rr_bars(
                    ax, df_mean, df_std, metric, slugs, kinds, by_var, sp_exp=sp_exp
                )
                if legend_ax is None:
                    legend_ax = ax
            else:
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
            if not np.any(~np.isnan(vals)):
                ax.set_visible(False)
                continue
            disp = _metric_plot_title(metric)
            ax.set_title(disp, fontsize=14)
            ax.set_ylabel(_metric_ylabel(metric))
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(*_ylim_with_errs(vals, errs))
        for j in range(len(metrics), len(axes_arr)):
            axes_arr[j].set_visible(False)
        if legend_ax is not None:
            handles, labels = legend_ax.get_legend_handles_labels()
            # Deduplicate legend entries (Reactive / ReCord / Self-play).
            seen = set()
            uniq_h, uniq_l = [], []
            for h, lab in zip(handles, labels):
                if lab in seen:
                    continue
                seen.add(lab)
                uniq_h.append(h)
                uniq_l.append(lab)
            fig.legend(
                uniq_h,
                uniq_l,
                loc="outside upper center",
                ncol=max(len(uniq_l), 1),
                frameon=False,
            )
        base = os.path.join(out_dir, f"{prefix}_subplots")
        plt.savefig(f"{base}.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
        plt.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.25)
        plt.close()


def _is_logreplay_block(row_lbl: str) -> bool:
    low = str(row_lbl).strip().lower()
    return ("log" in low and "replay" in low) or low in ("lr", "log-replay", "log replay")


def plot_combined_metrics_grid_3x4(
    blocks: list[tuple[pd.DataFrame, pd.DataFrame, str]],
    out_dir: str,
    *,
    figure_title: str,
    basename: str = "combined_logreplay_seeds_rewards_subplots",
    replay_minus_reactive: bool = False,
):
    """One N×4 figure: each row = an evaluation block; col = four plot metrics.

    If ``replay_minus_reactive`` and experiment names form a full ``replay_<v>`` / ``reactive_<v>``
    grid, each subplot uses one bar per variant: ``μ_replay − μ_reactive`` (error bars from combined std).
    """
    if not blocks:
        return
    font = _compare_serif_font_name()
    nrows, ncols = len(blocks), len(PLOT_METRICS_ORDER)

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
        has_any_sp = any(
            _pick_selfplay_exp(dm.index)
            for dm, _, _ in (
                (_slice_plot_metrics(df_m, df_s)[0], df_s, lbl) for df_m, df_s, lbl in blocks
            )
            if not dm.empty
        )
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.5 * ncols + (1.0 if has_any_sp else 0.5), 2.65 * nrows + 0.7),
            constrained_layout=True,
            squeeze=False,
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
                groups = None
                sp_exp = None
            else:
                groups = _strategy_kind_groups(df_m_sliced.index)
                skip_sp = _is_logreplay_block(row_lbl)
                if groups is not None:
                    slugs, kinds, by_var, sp_exp = groups
                    if skip_sp:
                        sp_exp = None
                    xtick_labels = [_pbt_strategy_display(s) for s in slugs]
                    if sp_exp:
                        xtick_labels.append("Self-play")
                    n_exp = len(xtick_labels)
                    x = np.arange(n_exp)
                    groups = (slugs, kinds, by_var, sp_exp)
                else:
                    sp_exp = None
                    df_m_sliced, df_s_sliced = _reindex_plot_exps(df_m_sliced, df_s_sliced)
                    if skip_sp:
                        keep = [e for e in df_m_sliced.index if not _is_selfplay_exp(str(e))]
                        df_m_sliced = df_m_sliced.reindex(keep)
                        if df_s_sliced is not None and not df_s_sliced.empty:
                            df_s_sliced = df_s_sliced.reindex(keep)
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
                drew_grouped = False
                if use_rr_diff and pairs_rr is not None:
                    vals_list: list[float] = []
                    errs_list: list[float] = []
                    for er, ee in pairs_rr:
                        dvm, dvs = _diff_mean_std(df_m_sliced, df_s_sliced, er, ee, base)
                        vals_list.append(float(dvm) if dvm is not None else np.nan)
                        errs_list.append(float(dvs) if dvs is not None else 0.0)
                    vals = np.array(vals_list, dtype=float)
                    errs = np.array(errs_list, dtype=float)
                elif groups is not None:
                    slugs, kinds, by_var, sp_exp = groups
                    vals, errs = _draw_grouped_rr_bars(
                        ax,
                        df_m_sliced,
                        df_s_sliced,
                        metric,
                        slugs,
                        kinds,
                        by_var,
                        sp_exp=sp_exp,
                    )
                    drew_grouped = True
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
                if not drew_grouped:
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
                    rot = 18 if use_rr_diff else 35
                    ha = "right"
                    ax.set_xticklabels(xtick_labels, rotation=rot, ha=ha)
                else:
                    # Grouped drawer already set ticks (incl. Self-play); keep rotation readable.
                    for lbl in ax.get_xticklabels():
                        lbl.set_rotation(0)
                        lbl.set_ha("center")
                if use_rr_diff:
                    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
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

        if not use_rr_diff:
            # Prefer an axes that drew Self-play (log-replay row omits it).
            uniq_h, uniq_l = [], []
            seen: set[str] = set()
            for ax in axes.ravel():
                for h, lab in zip(*ax.get_legend_handles_labels()):
                    if lab in seen:
                        continue
                    seen.add(lab)
                    uniq_h.append(h)
                    uniq_l.append(lab)
            if uniq_h:
                fig.legend(
                    uniq_h,
                    uniq_l,
                    loc="outside upper center",
                    ncol=max(len(uniq_l), 1),
                    frameon=False,
                )
        path_base = os.path.join(out_dir, basename)
        plt.savefig(f"{path_base}.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
        plt.savefig(f"{path_base}.pdf", bbox_inches="tight", pad_inches=0.25)
        plt.close()


# Short column headers (legacy / other tools); LaTeX combined table uses full names + arrows below.
_METRIC_LATEX_HEADER = {
    "collision_per_agent": "Coll.",
    "offroad_per_agent": "Off.",
    "score": "Succ.",
    "lane_alignment_rate": "Lane",
}

# LaTeX table: short metric names with “better” direction (↑ higher, ↓ lower).
_METRIC_LATEX_FULL_ROW = {
    "collision_per_agent": (r"Collision", r"$\downarrow$"),
    "offroad_per_agent": (r"Off-road", r"$\downarrow$"),
    "score": (r"Success", r"$\uparrow$"),
    "lane_alignment_rate": (r"Lane", r"$\uparrow$"),
}

# Paper table (Reactive-PBT / \ours / Self-play).
LATEX_PAPER_METRICS_ORDER = (
    "collision_per_agent",
    "offroad_per_agent",
    "score",
    # "lane_alignment_rate",
)

# Preferred strategy when collapsing Uniform/Curriculum/Prioritized → one Reactive / one ReCord column.
_LATEX_STRATEGY_PREF = ("uniform", "prioritized", "curriculum")

# Plot row label → paper Evaluation cell (short name + acronym).
_PAPER_BLOCK_TITLES = {
    "Log replay": "Log-replay (LR)",
    "Unseen seeds · reactive-play": "Unseen seeds (US)",
    "Unseen seeds · replay-eval": "US + replay (USR)",
    "Unseen rewards · reactive-play": "Unseen rewards (UR)",
    "Unseen rewards · replay-eval": "UR + replay (URR)",
}

# Paper table row order (US → UR → USR → URR → LR).
_PAPER_BLOCK_ORDER = (
    "Unseen seeds · reactive-play",
    "Unseen rewards · reactive-play",
    "Unseen seeds · replay-eval",
    "Unseen rewards · replay-eval",
    "Log replay",
)


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
    """Map plot row labels to concise LaTeX Evaluation titles (paper acronyms)."""
    key = str(row_lbl).strip()
    if key in _PAPER_BLOCK_TITLES:
        return _PAPER_BLOCK_TITLES[key]
    low = key.lower()
    if "seeds" in low and "reactive" in low:
        return "Unseen seeds (US)"
    if "seeds" in low and "replay" in low:
        return "US + replay (USR)"
    if "rewards" in low and "reactive" in low:
        return "Unseen rewards (UR)"
    if "rewards" in low and "replay" in low:
        return "UR + replay (URR)"
    if "log" in low and "replay" in low:
        return "Log-replay (LR)"
    if "seeds" in low:
        return "Unseen seeds (US)"
    if "rewards" in low:
        return "Unseen rewards (UR)"
    return key


def _reorder_blocks_for_paper(
    blocks: list[tuple[pd.DataFrame, pd.DataFrame, str]],
) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Order evaluation blocks US → USR → UR → URR → LR; append any unknown labels last."""
    by_lbl = {str(lbl).strip(): (dm, ds, lbl) for dm, ds, lbl in blocks}
    out: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    seen: set[str] = set()
    for key in _PAPER_BLOCK_ORDER:
        if key in by_lbl:
            out.append(by_lbl[key])
            seen.add(key)
    for dm, ds, lbl in blocks:
        if str(lbl).strip() not in seen:
            out.append((dm, ds, lbl))
    return out


# Preferred strategy when collapsing Uniform/Curriculum/Prioritized → one Reactive / one ReCord column.
_LATEX_STRATEGY_PREF = ("uniform", "prioritized", "curriculum")
# Default paper table: show both Uniform and Prioritized (with strategy in parentheses).
_LATEX_MULTI_STRATEGIES = ("uniform", "prioritized")


def _paper_method_column_specs(
    all_exps: set[str],
    *,
    strategy: Optional[str] = None,
) -> tuple[list[Optional[str]], list[tuple[str, int]], list[str], str]:
    """Build paper-table method columns.

    Default: Reactive-PBT / \\ours for each of Uniform and Prioritized (if present), then Self-play.

    Returns ``(method_exps, top_groups, sub_headers, comment)`` where
    ``top_groups`` is ``[(display_name, n_cols), ...]`` for the first header row
    (strategy names / Self-play) and ``sub_headers`` is the second row
    (``Reactive-PBT`` / ``\\ours`` / empty under Self-play).
    """
    by_strat: dict[str, dict[str, str]] = defaultdict(dict)
    orphans_re: list[str] = []
    orphans_rr: list[str] = []
    for e in sorted(all_exps):
        if _is_selfplay_exp(e):
            continue
        vk = _replay_reactive_variant_kind(e)
        if vk is None:
            fam = _exp_replay_reactive_family(e)
            if fam == "reactive":
                orphans_re.append(e)
            elif fam == "replay":
                orphans_rr.append(e)
            continue
        slug, kind = vk
        by_strat[slug][kind] = e
    sp = _pick_selfplay_exp(all_exps)

    strat_arg = (str(strategy).lower() if strategy else None)
    if strat_arg in (None, "auto", "all", ""):
        wanted = [s for s in _LATEX_MULTI_STRATEGIES if s in by_strat]
        if not wanted:
            wanted = [s for s in _LATEX_STRATEGY_PREF if s in by_strat]
        if not wanted:
            wanted = sorted(by_strat.keys())
    else:
        wanted = [strat_arg]

    method_exps: list[Optional[str]] = []
    top_groups: list[tuple[str, int]] = []
    sub_headers: list[str] = []
    used: list[str] = []
    for slug in wanted:
        kinds = by_strat.get(slug, {})
        re_e = kinds.get("reactive")
        rr_e = kinds.get("replay")
        if re_e is None and rr_e is None:
            continue
        disp = _pbt_strategy_display(slug)
        method_exps.extend([re_e, rr_e])
        top_groups.append((disp, 2))
        sub_headers.extend(["Reactive-PBT", "\\ours"])
        used.append(slug)

    if not method_exps and (orphans_re or orphans_rr):
        method_exps = [
            orphans_re[0] if orphans_re else None,
            orphans_rr[0] if orphans_rr else None,
        ]
        top_groups = [("Methods", 2)]
        sub_headers = ["Reactive-PBT", "\\ours"]
        used = ["orphan"]

    method_exps.append(sp)
    top_groups.append(("Self-play", 1))
    sub_headers.append("")  # covered by multirow Self-play on row 1
    comment = (
        f"strategies={'+'.join(used)}; selfplay={sp}; cols="
        + ",".join(str(e) for e in method_exps)
    )
    return method_exps, top_groups, sub_headers, comment


def _paper_method_columns(
    all_exps: set[str],
    *,
    strategy: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Legacy single-pair picker (first Reactive / ReCord / Self-play)."""
    exps, _groups, _subs, comment = _paper_method_column_specs(all_exps, strategy=strategy)
    re_e = next((e for e in exps[:-1:2] if e), None) if exps else None
    rr_e = next((e for e in exps[1:-1:2] if e), None) if exps else None
    sp = exps[-1] if exps else None
    slug = None
    if "strategies=" in comment:
        part = comment.split("strategies=", 1)[1].split(";", 1)[0]
        slug = part.split("+", 1)[0] if part and part != "orphan" else None
    return re_e, rr_e, sp, slug


def _latex_two_row_method_headers(
    top_groups: list[tuple[str, int]],
    sub_headers: list[str],
) -> list[str]:
    """Build two header rows: strategy names on top, Reactive-PBT / \\ours below."""
    top_parts: list[str] = []
    cmid_parts: list[str] = []
    col = 3  # after Evaluation & Metric
    for name, n in top_groups:
        if n > 1:
            top_parts.append(f"\\multicolumn{{{n}}}{{c}}{{{name}}}")
            cmid_parts.append(f"\\cmidrule(lr){{{col}-{col + n - 1}}}")
        else:
            # Self-play spans both header rows
            top_parts.append(f"\\multirow{{2}}{{*}}{{{name}}}")
        col += n
    row1 = "Evaluation & Metric & " + " & ".join(top_parts) + " \\\\"
    row_cmid = " ".join(cmid_parts) if cmid_parts else ""
    # Second row: leave blank under Evaluation/Metric and under Self-play multirow
    sub_cells = [h if h else "" for h in sub_headers]
    row2 = " & & " + " & ".join(sub_cells) + " \\\\"
    out = [row1]
    if row_cmid:
        out.append(row_cmid)
    out.append(row2)
    return out


def _combined_metric_cell(
    dm: pd.DataFrame,
    ds: Optional[pd.DataFrame],
    exp: Optional[str],
    base: str,
    *,
    bold: bool = False,
) -> str:
    if not exp:
        return "\\multicolumn{1}{c}{--}"
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


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-7) -> float:
    """Continued fraction for incomplete beta (Numerical Recipes)."""
    am, bm = 1.0, 1.0
    az = 1.0
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    bz = 1.0 - qab * x / qap
    for m in range(1, max_iter + 1):
        em = float(m)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem))
        app = ap + d * az
        bpp = bp + d * bz
        am, bm = ap / bpp, bp / bpp
        az_prev = az
        az, bz = app / bpp, 1.0
        if abs(az - az_prev) < eps * abs(az):
            return az
    return az


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if a <= 0.0 or b <= 0.0:
        return float("nan")
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(max(x, 1e-300)) * a + math.log(max(1.0 - x, 1e-300)) * b - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _welch_ttest_pvalue(m1: float, s_pop1: float, n1: int, m2: float, s_pop2: float, n2: int) -> float:
    """Two-sided Welch's t-test p-value from means and population stds (ddof=0)."""
    if n1 < 2 or n2 < 2:
        return 1.0
    s1 = s_pop1 * math.sqrt(n1 / (n1 - 1)) if n1 > 1 else 0.0
    s2 = s_pop2 * math.sqrt(n2 / (n2 - 1)) if n2 > 1 else 0.0
    v1, v2 = s1 * s1, s2 * s2
    denom = math.sqrt(v1 / n1 + v2 / n2)
    if denom < 1e-15:
        return 1.0 if abs(m1 - m2) < 1e-12 else 0.0
    t = abs(m1 - m2) / denom
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = 0.0
    if n1 > 1:
        df_den += (v1 / n1) ** 2 / (n1 - 1)
    if n2 > 1:
        df_den += (v2 / n2) ** 2 / (n2 - 1)
    if df_den <= 0.0:
        return 1.0
    df = df_num / df_den
    x = df / (df + t * t)
    p = _betai(0.5 * df, 0.5, x)
    if not math.isfinite(p):
        return 1.0
    return float(min(max(p, 0.0), 1.0))


def _welch_ttest_pvalue_better(
    m_a: float,
    s_pop_a: float,
    n_a: int,
    m_b: float,
    s_pop_b: float,
    n_b: int,
    *,
    a_better_if_higher: bool,
) -> float:
    """One-sided Welch p-value for H1: method A is better than B."""
    if n_a < 2 or n_b < 2:
        return 1.0
    s1 = s_pop_a * math.sqrt(n_a / (n_a - 1)) if n_a > 1 else 0.0
    s2 = s_pop_b * math.sqrt(n_b / (n_b - 1)) if n_b > 1 else 0.0
    v1, v2 = s1 * s1, s2 * s2
    denom = math.sqrt(v1 / n_a + v2 / n_b)
    if denom < 1e-15:
        if abs(m_a - m_b) < 1e-12:
            return 1.0
        better = (m_a > m_b) if a_better_if_higher else (m_a < m_b)
        return 0.0 if better else 1.0
    # Positive t ⇒ A looks better under the chosen direction.
    diff = (m_a - m_b) if a_better_if_higher else (m_b - m_a)
    t = diff / denom
    df_num = (v1 / n_a + v2 / n_b) ** 2
    df_den = 0.0
    if n_a > 1:
        df_den += (v1 / n_a) ** 2 / (n_a - 1)
    if n_b > 1:
        df_den += (v2 / n_b) ** 2 / (n_b - 1)
    if df_den <= 0.0:
        return 1.0
    df = df_num / df_den
    x = df / (df + t * t)
    p_two = _betai(0.5 * df, 0.5, x)
    if not math.isfinite(p_two):
        return 1.0
    # Upper-tail one-sided p for signed t.
    p_one = 0.5 * p_two if t >= 0.0 else 1.0 - 0.5 * p_two
    return float(min(max(p_one, 0.0), 1.0))


def _row_best_methods(
    dm: pd.DataFrame,
    method_exps: list[Optional[str]],
    base: str,
    *,
    atol: float = 1e-9,
) -> set[int]:
    """Indices in ``method_exps`` with the best mean for this metric."""
    maximize = _metric_higher_is_better(base)
    pairs: list[tuple[int, float]] = []
    for i, e in enumerate(method_exps):
        if not e:
            continue
        col = _metric_column_for_base(dm, base)
        if col is None or e not in dm.index:
            continue
        m = dm.loc[e, col]
        if not is_num(m):
            continue
        pairs.append((i, float(m)))
    if not pairs:
        return set()
    if maximize:
        best = max(m for _, m in pairs)
        return {i for i, m in pairs if m + atol >= best}
    best = min(m for _, m in pairs)
    return {i for i, m in pairs if m - atol <= best}


def _exp_algo_kind(exp: Optional[str]) -> Optional[str]:
    """Algorithm family for significance grouping: ``reactive`` / ``replay`` / ``selfplay``."""
    if not exp:
        return None
    if _is_selfplay_exp(exp):
        return "selfplay"
    vk = _replay_reactive_variant_kind(exp)
    if vk is not None:
        return vk[1]
    return _exp_replay_reactive_family(exp)


def _row_method_stats(
    dm: pd.DataFrame,
    ds: Optional[pd.DataFrame],
    method_exps: list[Optional[str]],
    base: str,
) -> list[Optional[tuple[float, float, int]]]:
    """Per-column ``(mean, pop_std, n)`` for ``base``, or ``None`` if missing."""
    col = _metric_column_for_base(dm, base)
    if col is None:
        return [None] * len(method_exps)
    dn = getattr(dm, "attrs", {}).get("n_seeds")
    out: list[Optional[tuple[float, float, int]]] = []
    for e in method_exps:
        if not e or e not in dm.index:
            out.append(None)
            continue
        m = dm.loc[e, col]
        if not is_num(m):
            out.append(None)
            continue
        s = 0.0
        if ds is not None and not ds.empty and col in ds.columns and e in ds.index:
            sv = ds.loc[e, col]
            if is_num(sv):
                s = float(sv)
        n = 0
        if isinstance(dn, pd.DataFrame) and col in dn.columns and e in dn.index:
            nv = dn.loc[e, col]
            if is_num(nv):
                n = int(nv)
        out.append((float(m), s, n))
    return out


def _row_sig_best_methods(
    dm: pd.DataFrame,
    ds: Optional[pd.DataFrame],
    method_exps: list[Optional[str]],
    base: str,
    *,
    alpha: float = 0.05,
    atol: float = 1e-9,
) -> set[int]:
    """Bold within each Reactive-PBT / \\ours strategy pair under two-sided Welch.

    For Uniform and Prioritized columns, compare the matched Reactive vs \\ours pair.
    Bold the better mean only if two-sided Welch p < ``alpha``. Self-play is never bolded.
    Direction for “better”: collision/off-road lower-better; score/lane higher-better.
    """
    maximize = _metric_higher_is_better(base)
    stats = _row_method_stats(dm, ds, method_exps, base)
    wins: set[int] = set()
    i = 0
    n = len(method_exps)
    while i + 1 < n:
        e0, e1 = method_exps[i], method_exps[i + 1]
        k0, k1 = _exp_algo_kind(e0), _exp_algo_kind(e1)
        if e0 and e1 and {k0, k1} == {"reactive", "replay"}:
            st0, st1 = stats[i], stats[i + 1]
            if st0 is not None and st1 is not None:
                m0, s0, n0 = st0
                m1, s1, n1 = st1
                if n0 >= 2 and n1 >= 2:
                    if maximize:
                        better = i if m0 > m1 + atol else (i + 1 if m1 > m0 + atol else None)
                    else:
                        better = i if m0 < m1 - atol else (i + 1 if m1 < m0 - atol else None)
                    if better is not None:
                        p = _welch_ttest_pvalue(m0, s0, n0, m1, s1, n1)
                        if p < alpha:
                            wins.add(better)
            i += 2
            continue
        i += 1
    return wins


def _row_stat_best_methods(
    dm: pd.DataFrame,
    ds: Optional[pd.DataFrame],
    method_exps: list[Optional[str]],
    base: str,
    *,
    alpha: float = 0.05,
    atol: float = 1e-9,
) -> set[int]:
    """Deprecated alias → ``_row_sig_best_methods``."""
    return _row_sig_best_methods(
        dm, ds, method_exps, base, alpha=alpha, atol=atol
    )


def export_combined_metrics_latex_table(
    out_dir: str,
    basename: str,
    blocks: list[tuple[pd.DataFrame, pd.DataFrame, str]],
    *,
    caption: str = (
        "Comparison on zero-shot coordination performance "
        "(mean $\\pm$ std across seeds). "
        "Bold: within each strategy, Reactive-PBT vs \\ours "
        "(two-sided Welch $t$-test, $\\alpha=0.05$; "
        "collision/off-road $\\downarrow$, success $\\uparrow$)."
    ),
    label: str = "tab:pbt_init",
    replay_minus_reactive: bool = False,
    latex_strategy: Optional[str] = None,
    include_lane: bool = False,
) -> Optional[str]:
    """Write a ``table*`` in the paper layout.

    Absolute mode (default): columns ``Reactive-PBT (Uniform)``, ``\\ours (Uniform)``,
    ``Reactive-PBT (Prioritized)``, ``\\ours (Prioritized)``, ``Self-play``.
    Rows US / UR / USR / URR / LR × metrics. Use ``latex_strategy`` to restrict to one strategy.
    """
    if not blocks:
        return None

    if replay_minus_reactive:
        return _export_combined_rr_diff_latex_table(
            out_dir, basename, blocks, caption=caption, label=label
        )

    blocks = _reorder_blocks_for_paper(blocks)
    metrics = list(LATEX_PAPER_METRICS_ORDER)
    if include_lane and "lane_alignment_rate" not in metrics:
        metrics.append("lane_alignment_rate")
    n_metrics = len(metrics)

    sliced: list[tuple[pd.DataFrame, Optional[pd.DataFrame], str]] = []
    all_exps: set[str] = set()
    for df_m, df_s, row_lbl in blocks:
        dm, ds = _slice_plot_metrics(df_m, df_s)
        sliced.append((dm, ds, row_lbl))
        if not dm.empty:
            all_exps.update(str(e) for e in dm.index)
    if not all_exps:
        return None

    method_exps, top_groups, sub_headers, col_comment = _paper_method_column_specs(
        all_exps, strategy=latex_strategy
    )
    n_data_cols = len(method_exps)
    col_spec = f"ll*{{{n_data_cols}}}{{c}}"

    lines: list[str] = [
        "% Auto-generated by analyze/compare_exps.py",
        "% Requires: \\usepackage{booktabs}",
        "% Requires: \\usepackage{multirow}",
        "% Requires: \\usepackage{bm}",
        "% Define \\ours in the paper (e.g. ReCord).",
        "% Bold = within each strategy, better of Reactive-PBT vs \\ours if two-sided "
        "Welch p < 0.05 (collision/off-road: min; score: max). Self-play is never bolded.",
        f"% {col_comment}",
        "% Header row 1 = strategy; row 2 = Reactive-PBT / \\ours.",
    ]
    lines.extend(
        [
            "\\begin{table*}[h]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "% Slightly tighter than default to fit 5 method columns in \\textwidth.",
            "\\small",
            "\\setlength{\\tabcolsep}{3.5pt}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
        ]
    )
    lines.extend(_latex_two_row_method_headers(top_groups, sub_headers))
    lines.append("\\midrule")

    for bi, (dm, ds, row_lbl) in enumerate(sliced):
        if dm.empty:
            continue
        block_tex = _latex_escape(_block_title_latex(row_lbl))
        # Prefer method exps present in this block; still compare winners on available cells.
        row_methods = [
            (e if e and e in dm.index else None) for e in method_exps
        ]
        if not any(row_methods):
            continue
        for mi, base in enumerate(metrics):
            mname = _metric_latex_row_label(base)
            if mi == 0:
                c_eval = f"\\multirow{{{n_metrics}}}{{*}}{{{block_tex}}}"
            else:
                c_eval = ""
            win_idx = _row_sig_best_methods(dm, ds, row_methods, base)
            cells = [
                _combined_metric_cell(dm, ds, e, base, bold=(i in win_idx))
                for i, e in enumerate(row_methods)
            ]
            lines.append(f"{c_eval} & {mname} & " + " & ".join(cells) + " \\\\")
        if bi < len(sliced) - 1:
            lines.append("\\midrule")

    # Drop trailing midrule if last block(s) were skipped empty
    while lines and lines[-1] == "\\midrule":
        lines.pop()

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])

    out_path = os.path.join(out_dir, f"{basename}_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def _export_combined_rr_diff_latex_table(
    out_dir: str,
    basename: str,
    blocks: list[tuple[pd.DataFrame, pd.DataFrame, str]],
    *,
    caption: str,
    label: str,
) -> Optional[str]:
    """Legacy wide Δ table: one column per variant = μ_replay − μ_reactive."""
    sliced: list[tuple[pd.DataFrame, Optional[pd.DataFrame], str]] = []
    all_exps: set[str] = set()
    for df_m, df_s, row_lbl in blocks:
        dm, ds = _slice_plot_metrics(df_m, df_s)
        sliced.append((dm, ds, row_lbl))
        if not dm.empty:
            all_exps.update(str(e) for e in dm.index)
    if not all_exps:
        return None

    exps_sorted = sorted(e for e in all_exps if not _is_selfplay_exp(e))
    rr_groups = _latex_replay_reactive_variant_group_layout(exps_sorted)
    if rr_groups is None:
        print(
            "Warning: replay-minus-reactive table requested but experiment names are not a "
            "full replay_<v> / reactive_<v> grid; skipping LaTeX."
        )
        return None
    col_order, group_tex, _ = rr_groups
    pairs_rr = _rr_variant_pairs(col_order)
    n_metrics = len(PLOT_METRICS_ORDER)
    n_data_cols = len(pairs_rr)
    col_spec = f"ll*{{{n_data_cols}}}{{c}}"

    lines: list[str] = [
        "% Auto-generated by analyze/compare_exps.py (replay − reactive per variant)",
        "% Requires: \\usepackage{booktabs}",
        "% Requires: \\usepackage{multirow}",
        "% Requires: \\usepackage{bm}",
        "\\begin{table*}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "% Columns: $\\Delta = \\mu_{\\mathrm{replay}} - \\mu_{\\mathrm{reactive}}$ "
        "(mean $\\pm$ $\\sqrt{\\sigma_r^2 + \\sigma_{re}^2}$ over seeds).",
        "Evaluation & Metric & " + " & ".join(group_tex) + " \\\\",
        "\\midrule",
    ]
    for bi, (dm, ds, row_lbl) in enumerate(sliced):
        block_tex = _latex_escape(_block_title_latex(row_lbl))
        winners_by_base = {
            base: _row_best_rr_diff(dm, ds, pairs_rr, base) for base in PLOT_METRICS_ORDER
        }
        for mi, base in enumerate(PLOT_METRICS_ORDER):
            mname = _metric_latex_row_label(base)
            c_eval = (
                f"\\multirow{{{n_metrics}}}{{*}}{{{block_tex}}}" if mi == 0 else ""
            )
            win = winners_by_base[base]
            cells = [
                _combined_diff_cell(dm, ds, er, ee, base, bold=(k in win))
                for k, (er, ee) in enumerate(pairs_rr)
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
    parser = argparse.ArgumentParser(
        "Compare logreplay, wosac, unseen (seeds/rewards × replay/reactive-play eval)"
    )
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
        choices=[
            "logreplay",
            "wosac",
            "unseen_seeds",
            "unseen_rewards",
            "unseen_seeds_replay",
            "unseen_seeds_reactive",
            "unseen_rewards_replay",
            "unseen_rewards_reactive",
            "both",
            "all",
        ],
        help="Which result blocks to emit. unseen_seeds / unseen_rewards = both eval protocols.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip bar plot generation")
    parser.add_argument(
        "--combined-rr-diff",
        action="store_true",
        help=(
            "Combined figure + LaTeX: one value per variant = train-replay mean − train-reactive mean "
            "(requires replay_<v> / reactive_<v> grid). Saves basename *_rr_diff.*; "
            "σ_Δ ≈ sqrt(σ_replay² + σ_reactive²)."
        ),
    )
    parser.add_argument(
        "--latex-strategy",
        type=str,
        default=None,
        choices=["uniform", "curriculum", "prioritized", "auto", "all"],
        help=(
            "Paper LaTeX columns. Default/auto/all: Reactive-PBT/\\ours for Uniform and "
            "Prioritized (with strategy in parentheses) + Self-play. "
            "A single slug restricts to that pair only."
        ),
    )
    parser.add_argument(
        "--latex-include-lane",
        action="store_true",
        help="Include lane alignment rate row in the paper LaTeX table (omitted by default).",
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
            "unseen_other_seeds_reactive": "Unseen other seeds · reactive-play eval (UOS)",
            "unseen_other_seeds_replay": "Unseen other seeds · replay eval (UOS)",
            "unseen_other_rewards_reactive": "Unseen other rewards · reactive-play eval (UOR)",
            "unseen_other_rewards_replay": "Unseen other rewards · replay eval (UOR)",
        }
        if name in subplot_titles:
            plot_metrics_subplot_figure(
                df_m,
                df_s,
                out_dir,
                safe,
                figure_title=subplot_titles[name],
                include_selfplay=(name != "logreplay"),
            )
        else:
            plot_bar_comparison(df_m, df_s, out_dir, safe)


if __name__ == "__main__":
    args = parse_args()
    (
        lr,
        wosac,
        unseen_seeds_reactive,
        unseen_seeds_replay,
        unseen_rewards_reactive,
        unseen_rewards_replay,
    ) = collect_exp_results(args.base_path, exp_names=args.exps)

    out_dir = args.out_dir or os.path.join(args.base_path, "compare")
    os.makedirs(out_dir, exist_ok=True)

    fmt = args.format
    run_all = fmt == "all"
    run_both = fmt == "both"  # logreplay + wosac only

    if (fmt == "logreplay" or run_all or run_both) and lr:
        _run_format("logreplay", lr, args, out_dir)
    if (fmt == "wosac" or run_all or run_both) and wosac:
        _run_format("wosac", wosac, args, out_dir)

    emit_seeds_reactive = fmt in ("all", "unseen_seeds", "unseen_seeds_reactive")
    emit_seeds_replay = fmt in ("all", "unseen_seeds", "unseen_seeds_replay")
    emit_rewards_reactive = fmt in ("all", "unseen_rewards", "unseen_rewards_reactive")
    emit_rewards_replay = fmt in ("all", "unseen_rewards", "unseen_rewards_replay")

    if emit_seeds_reactive and unseen_seeds_reactive:
        _run_format("unseen_other_seeds_reactive", unseen_seeds_reactive, args, out_dir)
    if emit_seeds_replay and unseen_seeds_replay:
        _run_format("unseen_other_seeds_replay", unseen_seeds_replay, args, out_dir)
    if emit_rewards_reactive and unseen_rewards_reactive:
        _run_format("unseen_other_rewards_reactive", unseen_rewards_reactive, args, out_dir)
    if emit_rewards_replay and unseen_rewards_replay:
        _run_format("unseen_other_rewards_replay", unseen_rewards_replay, args, out_dir)

    if (
        not args.no_plot
        and lr
        and (unseen_seeds_reactive or unseen_seeds_replay)
        and (unseen_rewards_reactive or unseen_rewards_replay)
    ):
        df_lm, df_ls = build_comparison_dfs(lr, "exp")
        combined_blocks = [(df_lm, df_ls, "Log replay")]
        if unseen_seeds_reactive:
            combined_blocks.append(
                (*build_comparison_dfs(unseen_seeds_reactive, "exp"), "Unseen seeds · reactive-play")
            )
        if unseen_seeds_replay:
            combined_blocks.append(
                (*build_comparison_dfs(unseen_seeds_replay, "exp"), "Unseen seeds · replay-eval")
            )
        if unseen_rewards_reactive:
            combined_blocks.append(
                (*build_comparison_dfs(unseen_rewards_reactive, "exp"), "Unseen rewards · reactive-play")
            )
        if unseen_rewards_replay:
            combined_blocks.append(
                (*build_comparison_dfs(unseen_rewards_replay, "exp"), "Unseen rewards · replay-eval")
            )
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
            _label = "tab:combined_pufferdrive_metrics_rr_diff"
        else:
            _cap = (
                "Comparison on zero-shot coordination performance "
                r"(mean $\pm$ std across seeds). "
                r"Bold: within each strategy, Reactive-PBT vs \ours "
                r"(two-sided Welch $t$-test, $\alpha=0.05$; "
                r"collision/off-road $\downarrow$, success $\uparrow$)."
            )
            _label = "tab:pbt_init"
        plot_combined_metrics_grid_3x4(
            combined_blocks,
            out_dir,
            figure_title=_fig_title,
            basename=_basename,
            replay_minus_reactive=_rr_diff,
        )
        print(f"\nSaved: {out_dir}/{_basename}.png, .pdf")
        _strat = getattr(args, "latex_strategy", None)
        if _strat == "auto":
            _strat = None
        tex_path = export_combined_metrics_latex_table(
            out_dir,
            _basename,
            combined_blocks,
            caption=_cap,
            label=_label,
            replay_minus_reactive=_rr_diff,
            latex_strategy=_strat,
            include_lane=bool(getattr(args, "latex_include_lane", False)),
        )
        if tex_path:
            print(f"Saved: {tex_path}")

    if not any(
        [
            lr,
            wosac,
            unseen_seeds_reactive,
            unseen_seeds_replay,
            unseen_rewards_reactive,
            unseen_rewards_replay,
        ]
    ):
        print(f"No results found under {args.base_path}")
