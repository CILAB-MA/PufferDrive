#!/usr/bin/env python3
"""Plot SPS summaries (train-only or fair amortized).

Train mode reads ``sps_{strategy}_{mode}/sps_summary.json``.
Fair mode amortizes ReCord collect (+ optional storage) from
``collect_bench_*/fair_sps_table.json``::

  eff_i = N / (N / sps_i + (T_collect + T_storage) / M)

Paper mode: main ghost-outline figure @ 2B (collect+storage) + appendix @ 100M.

Examples::

  python analyze/plot_sps.py --mode paper
  python analyze/plot_sps.py --mode train
  python analyze/plot_sps.py --mode fair --N 100000000 --include-storage
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Optional

FULL_TRAIN_N = 2_000_000_000.0
SPS_PROBE_N = 100_000_000.0

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# Match analyze/compare_exps.py
GOOGLE_PALETTE = (
    "#4285F4",  # Reactive-PBT
    "#EA4335",  # ReCord
)

STRATEGY_ORDER = ("uniform", "curriculum", "prioritized")
STRATEGY_LABEL = {
    "uniform": "Uniform",
    "curriculum": "Curriculum",
    "prioritized": "Prioritized",
}
KIND_ORDER = ("reactive", "record")  # record folder = ReCord / replay train
KIND_LABEL = {
    "reactive": "Reactive-PBT",
    "record": "ReCord",
}
KIND_COLOR = {
    "reactive": GOOGLE_PALETTE[0],
    "record": GOOGLE_PALETTE[1],
}

# strategy → directory under experiments/ with fair_sps_table.json
FAIR_TABLE_DIR = {
    "uniform": "collect_bench_lane_nominal",
    "prioritized": "collect_bench_lane_nominal_prioritized",
    "curriculum": "collect_bench_curriculum",
}

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


def _seed_sps(summary: dict) -> list[float]:
    return [
        float(s["mean_sps_second_half"])
        for s in summary.get("seeds", [])
        if "mean_sps_second_half" in s
    ]


def collect_sps(
    experiments_root: str,
) -> dict[str, dict[str, tuple[float, float, int]]]:
    """strategy → kind → (mean, std, n) of ``mean_sps_second_half`` across seeds."""
    out: dict[str, dict[str, tuple[float, float, int]]] = {}
    for strategy in STRATEGY_ORDER:
        for kind in KIND_ORDER:
            folder = os.path.join(experiments_root, f"sps_{strategy}_{kind}")
            summary = _load_summary(os.path.join(folder, "sps_summary.json"))
            if summary is None:
                continue
            vals = _seed_sps(summary)
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr)) if len(arr) > 1 else 0.0
            out.setdefault(strategy, {})[kind] = (mean, std, len(arr))
    return out


def _fair_eff(sps: float, N: float, T_cs: float, M: float) -> float:
    """eff = N / (N/sps + T_cs/M)."""
    t_train = N / max(float(sps), 1e-12)
    t_extra = float(T_cs) / max(float(M), 1e-12)
    return N / max(t_train + t_extra, 1e-12)


def collect_fair_sps(
    experiments_root: str,
    *,
    N: Optional[float] = None,
    M: Optional[float] = None,
    include_storage: bool = False,
) -> dict[str, dict[str, tuple[float, float, int]]]:
    """Per-seed amortized effective SPS; Reactive extra=0, ReCord uses T_collect."""
    out: dict[str, dict[str, tuple[float, float, int]]] = {}
    for strategy in STRATEGY_ORDER:
        fair_dir = FAIR_TABLE_DIR.get(strategy)
        if not fair_dir:
            continue
        fair = _load_summary(
            os.path.join(experiments_root, fair_dir, "fair_sps_table.json")
        )
        if fair is None:
            print(f"[warn] missing fair table for {strategy}: {fair_dir}")
            continue
        inputs = fair.get("inputs") or {}
        N_use = float(N if N is not None else fair.get("N_ego_steps") or 1e8)
        M_use = float(M if M is not None else fair.get("M_seeds") or 4.0)
        T_collect = float(inputs.get("T_collect_est") or 0.0)
        T_storage = (
            float(inputs.get("t_storage") or 0.0) if include_storage else 0.0
        )
        T_cs = T_collect + T_storage

        for kind in KIND_ORDER:
            folder = os.path.join(experiments_root, f"sps_{strategy}_{kind}")
            summary = _load_summary(os.path.join(folder, "sps_summary.json"))
            if summary is None:
                continue
            train_vals = _seed_sps(summary)
            if not train_vals:
                continue
            if kind == "reactive":
                eff_vals = train_vals
            else:
                eff_vals = [_fair_eff(s, N_use, T_cs, M_use) for s in train_vals]
            arr = np.asarray(eff_vals, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr)) if len(arr) > 1 else 0.0
            out.setdefault(strategy, {})[kind] = (mean, std, len(arr))
    return out


def _print_table(
    data: dict[str, dict[str, tuple[float, float, int]]],
    *,
    label: str,
) -> None:
    print(label)
    print("strategy          kind         mean_sps    std     n")
    for s in STRATEGY_ORDER:
        if s not in data:
            continue
        for k in KIND_ORDER:
            if k not in data[s]:
                continue
            m, sd, n = data[s][k]
            print(f"{s:<16} {k:<12} {m:10.1f} {sd:7.1f} {n:3d}")


def plot_sps_stacked(
    train_data: dict[str, dict[str, tuple[float, float, int]]],
    fair_data: dict[str, dict[str, tuple[float, float, int]]],
    out_path: str,
    *,
    sps_scale: float = 1e3,
    budget_label: str = "2B",
) -> None:
    """Single panel: reactive = blue; ReCord = ghost outline (raw) + solid red (eff).

    Full raw-training SPS is a light outline/ghost bar; solid red fills only the
    effective SPS after amortized collect+storage at ``budget_label``. The hollow
    tip is the prep penalty.
    """
    strategies = [s for s in STRATEGY_ORDER if s in train_data]
    if not strategies:
        raise SystemExit("No SPS summaries for stacked figure.")

    font = _serif_font_name()
    n_strat = len(strategies)
    y = np.arange(n_strat, dtype=float)[::-1]
    height = 0.32
    offsets = np.linspace(-0.5, 0.5, 2) * height  # reactive, record

    rc = {
        "font.family": font,
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)

        for si, strat in enumerate(strategies):
            yy = y[si]
            # Reactive-PBT
            if "reactive" in train_data[strat]:
                m, sd, _ = train_data[strat]["reactive"]
                w = m / sps_scale
                e = sd / sps_scale
                ax.barh(
                    yy + offsets[0],
                    w,
                    height=height * 0.95,
                    xerr=e,
                    capsize=3,
                    color=KIND_COLOR["reactive"],
                    edgecolor="gray",
                    linewidth=0.5,
                    zorder=3,
                )
            # ReCord: ghost = full raw train; solid red = effective (hollow tip = prep)
            if "record" in train_data[strat] and strat in fair_data and "record" in fair_data[strat]:
                m_train, sd_train, _ = train_data[strat]["record"]
                m_eff, _sd_eff, _ = fair_data[strat]["record"]
                w_train = max(m_train, 0.0) / sps_scale
                w_eff = max(m_eff, 0.0) / sps_scale
                y_rec = yy + offsets[1]
                # Ghost / outline for raw training SPS
                ax.barh(
                    y_rec,
                    w_train,
                    height=height * 0.95,
                    facecolor="#F2F2F2",
                    edgecolor="#666666",
                    linewidth=1.0,
                    linestyle="--",
                    zorder=2,
                )
                # Solid effective SPS
                ax.barh(
                    y_rec,
                    w_eff,
                    height=height * 0.95,
                    color=KIND_COLOR["record"],
                    edgecolor="gray",
                    linewidth=0.5,
                    zorder=3,
                )
                if sd_train > 0:
                    ax.errorbar(
                        m_train / sps_scale,
                        y_rec,
                        xerr=sd_train / sps_scale,
                        fmt="none",
                        ecolor="0.25",
                        capsize=3,
                        zorder=4,
                    )

        ax.set_yticks(y)
        ax.set_yticklabels([STRATEGY_LABEL[s] for s in strategies])
        ax.set_xlabel("Steps per second")
        ax.grid(axis="x", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        hi = 0.0
        for s in strategies:
            if "reactive" in train_data[s]:
                m, sd, _ = train_data[s]["reactive"]
                hi = max(hi, (m + sd) / sps_scale)
            if "record" in train_data[s]:
                m, sd, _ = train_data[s]["record"]
                hi = max(hi, (m + sd) / sps_scale)
        xmax = hi * 1.12 if hi > 0 else 1.0
        ax.set_xlim(0.0, xmax)
        if sps_scale == 1e3:
            ticks = ax.get_xticks()
            ticks = [t for t in ticks if 0 <= t <= xmax + 1e-9]
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [f"{int(t)}K" if float(t).is_integer() else f"{t:g}K" for t in ticks]
            )

        from matplotlib.patches import Patch

        legend_handles = [
            Patch(facecolor=KIND_COLOR["reactive"], edgecolor="gray", label="Reactive-PBT"),
            Patch(
                facecolor=KIND_COLOR["record"],
                edgecolor="gray",
                label="ReCord (effective)",
            ),
            Patch(
                facecolor="#F2F2F2",
                edgecolor="#666666",
                linestyle="--",
                linewidth=1.0,
                label=f"ReCord raw train (prep @{budget_label})",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="outside upper center",
            ncol=3,
            frameon=False,
        )

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        base, _ext = os.path.splitext(out_path)
        fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"Saved: {base}.pdf")


def plot_sps(
    data: dict[str, dict[str, tuple[float, float, int]]],
    out_path: str,
    *,
    sps_scale: float = 1e3,
    xlabel: str = "SPS",
) -> None:
    """Horizontal grouped-bar figure (log-replay colors, no titles)."""
    strategies = [s for s in STRATEGY_ORDER if s in data]
    if not strategies:
        raise SystemExit("No SPS summaries found.")

    font = _serif_font_name()
    n_strat = len(strategies)
    n_kind = len(KIND_ORDER)
    # Top → bottom: Uniform, Curriculum, Prioritized
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
        handles = []
        labels = []
        for ki, kind in enumerate(KIND_ORDER):
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
                color=KIND_COLOR[kind],
                edgecolor="gray",
                linewidth=0.5,
                label=KIND_LABEL[kind],
                zorder=3,
            )
            handles.append(bars)
            labels.append(KIND_LABEL[kind])

        ax.set_yticks(y)
        ax.set_yticklabels([STRATEGY_LABEL[s] for s in strategies])
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        hi = 0.0
        for s in strategies:
            for kind in KIND_ORDER:
                if kind in data[s]:
                    m, sd, _ = data[s][kind]
                    hi = max(hi, (m + sd) / sps_scale)
        xmax = hi * 1.12 if hi > 0 else 1.0
        ax.set_xlim(0.0, xmax)
        if sps_scale == 1e3:
            ticks = ax.get_xticks()
            ticks = [t for t in ticks if 0 <= t <= xmax + 1e-9]
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

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        base, _ext = os.path.splitext(out_path)
        fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"Saved: {base}.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiments-root",
        default="/data/puffer/experiments",
        help="Root containing sps_{strategy}_{mode}/ and collect_bench_*/",
    )
    p.add_argument(
        "--mode",
        choices=("train", "fair", "paper"),
        default="paper",
        help="train | fair (amortized) | paper (main @2B + appendix @100M)",
    )
    p.add_argument(
        "--include-storage",
        action="store_true",
        help="With --mode fair, also amortize disk write time",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output path without or with extension (writes .pdf at dpi=300)",
    )
    p.add_argument(
        "--no-scale",
        action="store_true",
        help="Plot raw SPS instead of ×10³",
    )
    p.add_argument(
        "--N",
        type=float,
        default=None,
        help="Fair window ego-steps (default: from fair_sps_table.json)",
    )
    p.add_argument(
        "--M",
        type=float,
        default=None,
        help="Fair amortization #seeds (default: from fair_sps_table.json)",
    )
    args = p.parse_args()
    scale = 1.0 if args.no_scale else 1e3
    root = args.experiments_root
    compare = "/data/puffer/results/compare"

    if args.mode == "paper":
        train_data = collect_sps(root)
        fair_2b = collect_fair_sps(
            root,
            N=FULL_TRAIN_N,
            M=args.M,
            include_storage=True,
        )
        _print_table(train_data, label="mode=train")
        _print_table(
            fair_2b,
            label=f"mode=fair N={FULL_TRAIN_N/1e9:.0f}B "
            f"(collect+storage, amortized)",
        )
        main_out = f"{compare}/sps"
        plot_sps_stacked(
            train_data,
            fair_2b,
            main_out,
            sps_scale=scale,
            budget_label=f"{FULL_TRAIN_N/1e9:.0f}B",
        )
        # Alias for older paper paths that still point at sps_paper.pdf
        paper_alias = f"{compare}/sps_paper.pdf"
        shutil.copy2(f"{main_out}.pdf", paper_alias)
        print(f"Saved: {paper_alias}")

        fair_100m = collect_fair_sps(
            root,
            N=SPS_PROBE_N,
            M=args.M,
            include_storage=True,
        )
        _print_table(
            fair_100m,
            label=f"mode=fair N={SPS_PROBE_N/1e6:.0f}M (appendix)",
        )
        plot_sps(
            fair_100m,
            f"{compare}/sps_fair_100m",
            sps_scale=scale,
            xlabel="Effective SPS",
        )
        return

    if args.mode == "fair":
        data = collect_fair_sps(
            root,
            N=args.N,
            M=args.M,
            include_storage=bool(args.include_storage),
        )
        xlabel = "Effective SPS"
        default_out = f"{compare}/sps_fair"
    else:
        data = collect_sps(root)
        xlabel = "Training SPS"
        default_out = f"{compare}/sps"

    _print_table(data, label=f"mode={args.mode}")
    out = args.out or default_out
    if out.lower().endswith((".png", ".pdf")):
        out = os.path.splitext(out)[0]
    plot_sps(
        data,
        out,
        sps_scale=scale,
        xlabel=xlabel,
    )


if __name__ == "__main__":
    main()
