#!/usr/bin/env python3
"""Plot collected SPS summaries (train / fair / paper / hybrid mixed).

Reads ``/data/puffer/experiments/sps_{strategy}_{mode}/sps_summary.json``
(mode: record → ReCord, reactive → Reactive-PBT, mixed_p* → hybrid).

Modes:
  (default)  raw train SPS — Reactive-PBT vs ReCord
  --fair     effective SPS = N / (N/sps + T_prep)  (prep amortized)
  --paper    dual panel: Training SPS | ReCord prep overhead %
  --mixed    SPS vs partner_replay_prob (via scripts/pbt_sps_mixed_sweep.sh)

Prep wall time default (~440s) reverse-engineered from existing
``sps_fair*.png`` / ``sps_paper.png`` (≈2.7% overhead at 2B steps).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

GOOGLE_BLUE = "#4285F4"
GOOGLE_RED = "#EA4335"

STRATEGY_ORDER = ("uniform", "curriculum", "prioritized")
STRATEGY_LABEL = {
    "uniform": "Uniform",
    "curriculum": "Curriculum",
    "prioritized": "Prioritized",
}

# Default partner-corpus prep wall (seconds). Matches prior sps_fair / sps_paper figs.
DEFAULT_PREP_SECONDS = 440.0

# Hybrid continuum tags (match scripts/pbt_sps_seed.sh).
_MIXED_PROB_TAGS = (
    (0.0, "reactive", None),  # pure reactive folder sps_{strategy}_reactive
    (0.25, "mixed", "p025"),
    (0.5, "mixed", "p05"),
    (0.75, "mixed", "p075"),
    (1.0, "record", None),  # pure record folder sps_{strategy}_record
)

_TIMES_TTF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "times.ttf"))
_registered_font: Optional[str] = None


def _serif_font_name() -> str:
    global _registered_font
    if _registered_font is not None:
        return _registered_font
    if os.path.isfile(_TIMES_TTF):
        try:
            font_manager.fontManager.addfont(_TIMES_TTF)
            _registered_font = font_manager.FontProperties(fname=_TIMES_TTF).get_name()
            return _registered_font
        except (OSError, ValueError, RuntimeError):
            pass
    _registered_font = "DejaVu Serif"
    return _registered_font


def _load_summary(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_sps(
    experiments_root: str,
) -> dict[str, dict[str, tuple[float, float, int]]]:
    """strategy → kind → (mean, std, n) of ``mean_sps_second_half`` across seeds."""
    out: dict[str, dict[str, tuple[float, float, int]]] = {}
    for strategy in STRATEGY_ORDER:
        for kind in ("reactive", "record"):
            folder = os.path.join(experiments_root, f"sps_{strategy}_{kind}")
            summary = _load_summary(os.path.join(folder, "sps_summary.json"))
            if summary is None:
                continue
            vals = [
                float(s["mean_sps_second_half"])
                for s in summary.get("seeds", [])
                if "mean_sps_second_half" in s
            ]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr)) if len(arr) > 1 else 0.0
            out.setdefault(strategy, {})[kind] = (mean, std, len(arr))
    return out


def collect_mixed_sps(
    experiments_root: str,
    strategy: str = "uniform",
) -> list[tuple[float, float, float, int, str]]:
    """Collect (p, mean, std, n, folder) along the hybrid continuum for one strategy.

    Endpoints use ``sps_{strategy}_reactive`` (p=0) and ``sps_{strategy}_record`` (p=1).
    Interior points use ``sps_{strategy}_mixed_{tag}``.
    """
    rows: list[tuple[float, float, float, int, str]] = []
    for p, kind, tag in _MIXED_PROB_TAGS:
        if kind == "mixed":
            folder_name = f"sps_{strategy}_mixed_{tag}"
        else:
            folder_name = f"sps_{strategy}_{kind}"
        folder = os.path.join(experiments_root, folder_name)
        summary = _load_summary(os.path.join(folder, "sps_summary.json"))
        if summary is None:
            continue
        # Prefer summary partner_replay_prob when present.
        if summary.get("partner_replay_prob") is not None and kind == "mixed":
            p = float(summary["partner_replay_prob"])
        vals = [
            float(s["mean_sps_second_half"])
            for s in summary.get("seeds", [])
            if "mean_sps_second_half" in s
        ]
        if not vals and summary.get("mean_sps_across_seeds") is not None:
            vals = [float(summary["mean_sps_across_seeds"])]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr)) if len(arr) > 1 else 0.0
        rows.append((float(p), mean, std, len(arr), folder_name))
    rows.sort(key=lambda r: r[0])
    return rows


def plot_mixed_sps(
    rows: list[tuple[float, float, float, int, str]],
    out_path: str,
    *,
    strategy: str = "uniform",
    sps_scale: float = 1e3,
) -> None:
    """Line plot: train SPS vs partner_replay_prob (hybrid continuum)."""
    if not rows:
        raise SystemExit(
            f"No mixed SPS summaries found for strategy={strategy}. "
            "Run scripts/pbt_sps_mixed_sweep.sh first."
        )
    font = _serif_font_name()
    xs = np.asarray([r[0] for r in rows], dtype=float)
    ys = np.asarray([r[1] / sps_scale for r in rows], dtype=float)
    es = np.asarray([r[2] / sps_scale for r in rows], dtype=float)
    rc = {
        "font.family": font,
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
        ax.errorbar(
            xs,
            ys,
            yerr=es,
            fmt="o-",
            color=GOOGLE_BLUE,
            ecolor=GOOGLE_BLUE,
            elinewidth=1.0,
            capsize=3.5,
            markersize=7.5,
            linewidth=2.0,
            label=STRATEGY_LABEL.get(strategy, strategy),
            zorder=3,
        )
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(
            ["0\n(Reactive-PBT)", "0.25", "0.5", "0.75", "1\n(ReCord)"]
        )
        ax.set_xlabel("Replay Ratio")
        ax.set_ylabel("SPS" if sps_scale == 1.0 else "SPS (×10³)")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        hi = float(np.nanmax(ys + es)) if len(ys) else 1.0
        ax.set_ylim(0.0, hi * 1.15 if hi > 0 else 1.0)
        ax.legend(loc="best", frameon=False)
        _save(fig, out_path)


def effective_sps(train_sps: float, *, amortize_steps: float, prep_seconds: float) -> float:
    """N / (N/sps + T_prep)."""
    if train_sps <= 0:
        return 0.0
    return 1.0 / (1.0 / train_sps + float(prep_seconds) / float(amortize_steps))


def with_effective_record(
    data: dict[str, dict[str, tuple[float, float, int]]],
    *,
    amortize_steps: float,
    prep_seconds: float,
) -> dict[str, dict[str, tuple[float, float, int]]]:
    """Replace record series with prep-amortized effective SPS (reactive unchanged)."""
    out: dict[str, dict[str, tuple[float, float, int]]] = {}
    n = float(amortize_steps)
    t_prep = float(prep_seconds)
    for strategy, kinds in data.items():
        out[strategy] = {}
        if "reactive" in kinds:
            out[strategy]["reactive"] = kinds["reactive"]
        if "record" in kinds:
            m, sd, n_seeds = kinds["record"]
            eff_m = effective_sps(m, amortize_steps=n, prep_seconds=t_prep)
            if m > 0 and sd > 0:
                denom = n + m * t_prep
                d_eff = (n * n) / (denom * denom)
                eff_sd = abs(d_eff) * sd
            else:
                eff_sd = 0.0
            out[strategy]["record"] = (float(eff_m), float(eff_sd), n_seeds)
    return out


def _save(fig, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    base, _ext = os.path.splitext(out_path)
    fig.savefig(f"{base}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {base}.png, {base}.pdf")


def plot_sps(
    data: dict[str, dict[str, tuple[float, float, int]]],
    out_path: str,
    *,
    sps_scale: float = 1e3,
    xlabel: str = "SPS",
    record_label: str = "ReCord",
) -> None:
    """Horizontal grouped-bar: Reactive-PBT (blue) + record series (red)."""
    strategies = [s for s in STRATEGY_ORDER if s in data]
    if not strategies:
        raise SystemExit("No SPS summaries found.")

    kind_order = ("reactive", "record")
    kind_label = {
        "reactive": "Reactive-PBT",
        "record": record_label,
    }
    kind_color = {
        "reactive": GOOGLE_BLUE,
        "record": GOOGLE_RED,
    }

    font = _serif_font_name()
    n_strat = len(strategies)
    n_kind = len(kind_order)
    y = np.arange(n_strat, dtype=float)[::-1]
    height = 0.32
    offsets = np.linspace(-(n_kind - 1) / 2, (n_kind - 1) / 2, n_kind) * height

    rc = {
        "font.family": font,
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
        handles, labels = [], []
        for ki, kind in enumerate(kind_order):
            means, errs = [], []
            for s in strategies:
                if kind in data[s]:
                    m, sd, _n = data[s][kind]
                    means.append(m / sps_scale)
                    errs.append(sd / sps_scale)
                else:
                    means.append(np.nan)
                    errs.append(0.0)
            bars = ax.barh(
                y + offsets[ki],
                means,
                height=height * 0.95,
                xerr=errs,
                capsize=3,
                color=kind_color[kind],
                edgecolor="gray",
                linewidth=0.5,
                label=kind_label[kind],
                zorder=3,
            )
            handles.append(bars)
            labels.append(kind_label[kind])

        ax.set_yticks(y)
        ax.set_yticklabels([STRATEGY_LABEL[s] for s in strategies])
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        hi = 0.0
        for s in strategies:
            for kind in kind_order:
                if kind in data[s]:
                    m, sd, _ = data[s][kind]
                    hi = max(hi, (m + sd) / sps_scale)
        xmax = hi * 1.12 if hi > 0 else 1.0
        ax.set_xlim(0.0, xmax)
        if sps_scale == 1e3:
            ticks = [t for t in ax.get_xticks() if 0 <= t <= xmax + 1e-9]
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [f"{int(t)}K" if float(t).is_integer() else f"{t:g}K" for t in ticks]
            )

        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=len(labels),
            frameon=False,
        )
        _save(fig, out_path)


def plot_paper(
    data: dict[str, dict[str, tuple[float, float, int]]],
    out_path: str,
    *,
    amortize_steps: float,
    prep_seconds: float,
    sps_scale: float = 1e3,
) -> None:
    """Dual panel matching ``sps_paper.png``: Training SPS | prep overhead %."""
    strategies = [s for s in STRATEGY_ORDER if s in data]
    if not strategies:
        raise SystemExit("No SPS summaries found.")

    font = _serif_font_name()
    n_strat = len(strategies)
    y = np.arange(n_strat, dtype=float)[::-1]
    height = 0.32
    offsets = np.linspace(-0.5, 0.5, 2) * height

    rc = {
        "font.family": font,
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
    }
    with plt.rc_context(rc):
        fig, (ax0, ax1) = plt.subplots(
            1, 2, figsize=(10.0, 3.6), constrained_layout=True, sharey=True
        )
        handles, labels = [], []

        # Left: raw training SPS
        for ki, kind in enumerate(("reactive", "record")):
            means, errs = [], []
            for s in strategies:
                m, sd, _ = data[s][kind]
                means.append(m / sps_scale)
                errs.append(sd / sps_scale)
            color = GOOGLE_BLUE if kind == "reactive" else GOOGLE_RED
            label = "Reactive-PBT" if kind == "reactive" else "ReCord"
            bars = ax0.barh(
                y + offsets[ki],
                means,
                height=height * 0.95,
                xerr=errs,
                capsize=3,
                color=color,
                edgecolor="gray",
                linewidth=0.5,
                label=label,
                zorder=3,
            )
            handles.append(bars)
            labels.append(label)

        ax0.set_yticks(y)
        ax0.set_yticklabels([STRATEGY_LABEL[s] for s in strategies])
        ax0.set_xlabel("Training SPS")
        ax0.grid(axis="x", alpha=0.3, zorder=0)
        ax0.set_axisbelow(True)
        hi = max(
            (data[s][k][0] + data[s][k][1]) / sps_scale
            for s in strategies
            for k in ("reactive", "record")
        )
        xmax = hi * 1.12
        ax0.set_xlim(0.0, xmax)
        if sps_scale == 1e3:
            ticks = [t for t in ax0.get_xticks() if 0 <= t <= xmax + 1e-9]
            ax0.set_xticks(ticks)
            ax0.set_xticklabels(
                [f"{int(t)}K" if float(t).is_integer() else f"{t:g}K" for t in ticks]
            )

        # Right: ReCord prep overhead % at amortize_steps
        n = float(amortize_steps)
        t_prep = float(prep_seconds)
        oh = []
        for s in strategies:
            sps = data[s]["record"][0]
            t_train = n / sps if sps > 0 else np.nan
            oh.append(100.0 * t_prep / (t_prep + t_train))
        # Format horizon for xlabel, e.g. 2B / 100M
        if amortize_steps >= 1e9 and float(amortize_steps).is_integer():
            hor = f"{int(amortize_steps / 1e9)}B"
        elif amortize_steps >= 1e6 and float(amortize_steps).is_integer():
            hor = f"{int(amortize_steps / 1e6)}M"
        else:
            hor = f"{amortize_steps:g}"
        ax1.barh(
            y,
            oh,
            height=height * 0.95,
            color=GOOGLE_RED,
            edgecolor="gray",
            linewidth=0.5,
            zorder=3,
        )
        ax1.set_xlabel(f"ReCord prep overhead ({hor}, %)")
        ax1.grid(axis="x", alpha=0.3, zorder=0)
        ax1.set_axisbelow(True)
        ax1.set_xlim(0.0, max(oh) * 1.25 if oh else 1.0)

        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=2,
            frameon=False,
        )
        _save(fig, out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiments-root", default="/data/puffer/experiments")
    p.add_argument("--out", default="/data/puffer/results/compare/sps")
    p.add_argument("--no-scale", action="store_true", help="Plot raw SPS instead of ×10³")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--fair",
        action="store_true",
        help="Effective SPS (prep amortized); red = effective ReCord (except preparation)",
    )
    mode.add_argument(
        "--paper",
        action="store_true",
        help="Dual panel: Training SPS + ReCord prep overhead %%",
    )
    mode.add_argument(
        "--mixed",
        action="store_true",
        help="Hybrid continuum: SPS vs partner_replay_prob (reactive…mixed…record)",
    )
    p.add_argument(
        "--strategy",
        default="uniform",
        choices=list(STRATEGY_ORDER),
        help="Strategy for --mixed curve (default: uniform)",
    )
    p.add_argument(
        "--amortize-steps",
        type=float,
        default=None,
        help="Steps used to amortize prep (fair default 1e8, paper/fair_2b default 2e9)",
    )
    p.add_argument(
        "--prep-seconds",
        type=float,
        default=DEFAULT_PREP_SECONDS,
        help="Partner corpus prep wall seconds (default matches prior fair figs)",
    )
    args = p.parse_args()

    scale = 1.0 if args.no_scale else 1e3
    out = args.out
    if out.lower().endswith((".png", ".pdf")):
        out = os.path.splitext(out)[0]

    if args.mixed:
        rows = collect_mixed_sps(args.experiments_root, strategy=args.strategy)
        print(f"[mixed] strategy={args.strategy}")
        print("p        mean_sps    std     n   folder")
        for p_val, m, sd, n, folder in rows:
            print(f"{p_val:<8g} {m:10.1f} {sd:7.1f} {n:3d}  {folder}")
        missing = [
            f"p={p:g}"
            for p, kind, tag in _MIXED_PROB_TAGS
            if not any(abs(r[0] - p) < 1e-9 for r in rows)
        ]
        if missing:
            print(f"Missing points: {', '.join(missing)}")
        mixed_out = out if out.endswith("mixed") else f"{out}_mixed_{args.strategy}"
        plot_mixed_sps(rows, mixed_out, strategy=args.strategy, sps_scale=scale)
        return

    data = collect_sps(args.experiments_root)

    if args.fair:
        amort = args.amortize_steps if args.amortize_steps is not None else 1e8
        fair = with_effective_record(
            data, amortize_steps=amort, prep_seconds=args.prep_seconds
        )
        print(
            f"[fair] amortize_steps={amort:g} prep_seconds={args.prep_seconds:g}"
        )
        print("strategy          kind         mean_sps    std     n")
        for s in STRATEGY_ORDER:
            if s not in fair:
                continue
            for k in ("reactive", "record"):
                if k not in fair[s]:
                    continue
                m, sd, n = fair[s][k]
                print(f"{s:<16} {k:<12} {m:10.1f} {sd:7.1f} {n:3d}")
        plot_sps(
            fair,
            out,
            sps_scale=scale,
            xlabel="Effective SPS",
            record_label="effective ReCord (except preparation)",
        )
    elif args.paper:
        amort = args.amortize_steps if args.amortize_steps is not None else 2e9
        print(f"[paper] amortize_steps={amort:g} prep_seconds={args.prep_seconds:g}")
        print("strategy          kind         mean_sps    std     n")
        for s in STRATEGY_ORDER:
            if s not in data:
                continue
            for k in ("reactive", "record"):
                if k not in data[s]:
                    continue
                m, sd, n = data[s][k]
                print(f"{s:<16} {k:<12} {m:10.1f} {sd:7.1f} {n:3d}")
        plot_paper(
            data,
            out,
            amortize_steps=amort,
            prep_seconds=args.prep_seconds,
            sps_scale=scale,
        )
    else:
        print("strategy          kind         mean_sps    std     n")
        for s in STRATEGY_ORDER:
            if s not in data:
                continue
            for k in ("reactive", "record"):
                if k not in data[s]:
                    continue
                m, sd, n = data[s][k]
                print(f"{s:<16} {k:<12} {m:10.1f} {sd:7.1f} {n:3d}")
        plot_sps(data, out, sps_scale=scale, xlabel="SPS", record_label="ReCord")


if __name__ == "__main__":
    main()
