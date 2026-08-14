#!/usr/bin/env python3
"""Apply criticality_metrics.py to the packs written by run_coordination.sh and
run_ego_readout.sh, and organize the results into one per-method (record / reactive /
selfplay) comparison report.

This is a pure post-hoc analysis over already-saved .npz packs (see rollout.py's
save_ego_pack / load_ego_pack) -- it does not load policies or step the environment, so it
does not need pufferlib and can run on CPU.

Usage:
    python criticality_report.py --out-root /data/puffer/results/coordination
    ./run_criticality_report.sh                      # shell wrapper, same defaults

Inputs (must already exist, i.e. run_coordination.sh / run_ego_readout.sh already ran):
    <out-root>/divergence_scenes/packs/seed{i}_{alias}.npz   (scalar-only packs)
    <out-root>/ego_readout/packs/seed{i}_{alias}.npz         (full-trajectory packs)

Outputs:
    <out-root>/criticality_report/summary.json
    <out-root>/criticality_report/report.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from common import ORDER, PRETTY, aggregate_numeric_across_seeds, jsonable
from criticality_metrics import (
    APPROXIMATE_METRICS,
    EXACT_METRICS,
    NOT_APPLICABLE,
    compute_all_metrics,
    has_readout,
    has_trajectories,
    monte_carlo_collision_probability,
    summarize_metrics,
)

_SEED_RE = re.compile(r"^seed(\d+)_(.+)\.npz$")


def _load_pack(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def discover_packs(pack_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """alias -> sorted list of (seed_index, path) for every seed{i}_{alias}.npz found."""
    by_alias: dict[str, list[tuple[int, Path]]] = {alias: [] for alias in ORDER}
    if not pack_dir.is_dir():
        return by_alias
    for p in sorted(pack_dir.glob("seed*_*.npz")):
        m = _SEED_RE.match(p.name)
        if not m:
            continue
        seed_idx, alias = int(m.group(1)), m.group(2)
        if alias in by_alias:
            by_alias[alias].append((seed_idx, p))
    for alias in by_alias:
        by_alias[alias].sort(key=lambda t: t[0])
    return by_alias


def analyze_source(pack_dir: Path, *, label: str) -> dict[str, Any] | None:
    by_alias = discover_packs(pack_dir)
    if not any(by_alias.values()):
        return None

    per_alias: dict[str, Any] = {}
    for alias in ORDER:
        entries = by_alias[alias]
        if not entries:
            continue
        packs = [_load_pack(p) for _, p in entries]

        per_seed_means: list[dict[str, float]] = []
        for pack in packs:
            metrics = compute_all_metrics(pack)
            per_seed_means.append(summarize_metrics(metrics))
        across = aggregate_numeric_across_seeds(per_seed_means, skip=set())

        mc = monte_carlo_collision_probability(packs)

        per_alias[alias] = {
            "n_seeds": len(packs),
            "n_ego_per_seed": [int(p["collided"].shape[0]) for p in packs],
            "has_trajectories": bool(packs and has_trajectories(packs[0])),
            "has_readout": bool(packs and has_readout(packs[0])),
            "metrics": {
                k: {"mean": v["mean"], "std": v["std"], "n": v["n"]} for k, v in across.items()
            },
            "monte_carlo_collision_probability": {
                "overall_mean": mc["overall_mean"],
                "n_scenes": mc["n_scenes"],
                "n_seeds": mc["n_seeds"],
            },
        }

    return {"label": label, "pack_dir": str(pack_dir), "by_method": per_alias}


def _fmt(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.4g}"


def render_markdown(sources: list[dict[str, Any]]) -> str:
    lines: list[str] = ["# Criticality metrics report", ""]
    lines.append(
        "Metrics from Westhofen et al. (2022), *Criticality Metrics for Automated "
        "Driving: A Review and Suitability Analysis of the State of the Art* "
        "(https://doi.org/10.1007/s11831-022-09788-7). See `criticality_metrics.py` for "
        "exact formulas, approximation caveats, and the full not-applicable list."
    )
    lines.append("")
    lines.append(
        f"Implemented: {len(EXACT_METRICS)} exact + {len(APPROXIMATE_METRICS)} "
        f"approximate (assumptions noted per-metric below). Not applicable given this "
        f"pipeline's state budget: {len(NOT_APPLICABLE)} (reasons in "
        f"`criticality_metrics.NOT_APPLICABLE`)."
    )
    lines.append("")

    for src in sources:
        lines.append(f"## {src['label']} ({src['pack_dir']})")
        lines.append("")
        by_method = src["by_method"]
        aliases = [a for a in ORDER if a in by_method]
        if not aliases:
            lines.append("_no packs found_")
            lines.append("")
            continue

        n_seeds = {a: by_method[a]["n_seeds"] for a in aliases}
        lines.append("Seeds: " + ", ".join(f"{PRETTY[a]}={n_seeds[a]}" for a in aliases))
        lines.append("")
        lines.append(
            "Monte-Carlo collision probability (fraction of seed replicates that "
            "collide, per map, averaged over maps):"
        )
        lines.append("")
        lines.append("| Method | P-MC |")
        lines.append("|---|---|")
        for a in aliases:
            mc = by_method[a]["monte_carlo_collision_probability"]["overall_mean"]
            lines.append(f"| {PRETTY[a]} | {_fmt(mc)} |")
        lines.append("")

        all_metric_names: list[str] = []
        seen = set()
        for a in aliases:
            for k in by_method[a]["metrics"]:
                if k not in seen:
                    seen.add(k)
                    all_metric_names.append(k)

        header = "| Metric | " + " | ".join(PRETTY[a] for a in aliases) + " |"
        sep = "|---|" + "---|" * len(aliases)
        lines.append(header)
        lines.append(sep)
        for k in all_metric_names:
            row = [k]
            for a in aliases:
                m = by_method[a]["metrics"].get(k)
                row.append(_fmt(m["mean"]) if m else "--")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Criticality-metrics report across methods")
    p.add_argument("--out-root", type=str, default="/data/puffer/results/coordination")
    args = p.parse_args()

    out_root = Path(args.out_root)
    sources = []
    for subdir, label in (
        ("divergence_scenes", "run_coordination.sh (scalar-only packs)"),
        ("ego_readout", "run_ego_readout.sh (full-trajectory packs)"),
    ):
        result = analyze_source(out_root / subdir / "packs", label=label)
        if result is not None:
            sources.append(result)

    if not sources:
        raise SystemExit(
            f"No packs found under {out_root}/{{divergence_scenes,ego_readout}}/packs/. "
            "Run run_coordination.sh and/or run_ego_readout.sh first."
        )

    report_dir = out_root / "criticality_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "sources": jsonable(sources),
        "applicability": {
            "exact": EXACT_METRICS,
            "approximate": APPROXIMATE_METRICS,
            "not_applicable": NOT_APPLICABLE,
        },
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    md = render_markdown(sources)
    (report_dir / "report.md").write_text(md)

    print(md)
    print(f"\nWrote {report_dir / 'summary.json'}")
    print(f"Wrote {report_dir / 'report.md'}")


if __name__ == "__main__":
    main()