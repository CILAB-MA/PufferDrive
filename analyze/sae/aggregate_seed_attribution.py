#!/usr/bin/env python3
"""Aggregate projection-attribution results across policy seeds.

Reads::

    <OUT_ROOT>/seed*/validation/attribution_validation_summary.json

Writes::

    <OUT_ROOT>/seed_level_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SAE_DIR = Path(__file__).resolve().parent
if str(_SAE_DIR) not in sys.path:
    sys.path.insert(0, str(_SAE_DIR))

from stats_utils import bootstrap_ci  # noqa: E402


ALIASES = ("record", "reactive", "selfplay")
PRETTY = {"record": "ReCord", "reactive": "Reactive", "selfplay": "Self-play"}
VARIANTS = ("raw", "cos", "scaled")


def _load_headline(seed_dir: Path) -> dict | None:
    slim = seed_dir / "validation" / "attribution_validation_summary.json"
    full = seed_dir / "validation" / "attribution_validation.json"
    if slim.is_file():
        return json.loads(slim.read_text())
    if full.is_file():
        data = json.loads(full.read_text())
        # older / full JSON may already embed headline-like fields
        if "headline" in data:
            return data
        if data.get("runs"):
            # rebuild slim-like from full
            run0 = data["runs"][0]
            headline = {}
            for pm, block in run0.get("pool_comparisons", {}).items():
                headline[pm] = {}
                for v in VARIANTS:
                    pr = (block.get("paired") or {}).get(v) or {}
                    pooled = pr.get("pooled") or {}
                    w = pooled.get("wilcoxon_record_gt_reactive") or {}
                    headline[pm][v] = {
                        "frac_record_gt_reactive": pooled.get("frac_record_gt_reactive"),
                        "median_delta": pooled.get("median_delta"),
                        "wilcoxon_p": w.get("pvalue"),
                        "median_attr_by_policy": pr.get("mean_median_attr"),
                    }
            return {
                "matched_features": data.get("matched_features"),
                "pool_modes": data.get("pool_modes"),
                "headline": headline,
                "seed_info": _read_seed_info(seed_dir),
            }
    return None


def _read_seed_info(seed_dir: Path) -> dict | None:
    path = seed_dir / "seed_info.json"
    if path.is_file():
        return json.loads(path.read_text())
    return None


def _seed_mean_std(vals: list[float]) -> dict:
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None, "values": []}
    ci = bootstrap_ci(arr, statistic="mean", seed=0)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "ci95": [ci["ci_low"], ci["ci_high"]],
        "values": arr.tolist(),
    }


def aggregate(out_root: Path, *, primary_pool: str = "max", primary_variant: str = "raw") -> dict:
    seed_dirs = sorted(
        [p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("seed")]
    )
    per_seed = []
    for sd in seed_dirs:
        headline = _load_headline(sd)
        if headline is None:
            continue
        info = headline.get("seed_info") or _read_seed_info(sd) or {}
        row = {
            "seed_dir": sd.name,
            "seed_index": info.get("seed_index"),
            "policy_runs": info.get("policy_runs"),
            "matched_features": headline.get("matched_features"),
            "headline": headline.get("headline"),
        }
        # flatten primary metric for easy tables
        h = (headline.get("headline") or {}).get(primary_pool, {}).get(primary_variant, {})
        row["primary"] = {
            "pool": primary_pool,
            "variant": primary_variant,
            "median_attr_by_policy": h.get("median_attr_by_policy"),
            "frac_record_gt_reactive": h.get("frac_record_gt_reactive"),
            "median_delta": h.get("median_delta"),
            "wilcoxon_p": h.get("wilcoxon_p"),
        }
        per_seed.append(row)

    # Cross-seed: for each alias, collect median attrs
    by_policy: dict[str, list[float]] = {a: [] for a in ALIASES}
    deltas: list[float] = []
    fracs: list[float] = []
    order_ok = 0
    for row in per_seed:
        med = (row.get("primary") or {}).get("median_attr_by_policy") or {}
        for a in ALIASES:
            if a in med and med[a] is not None:
                by_policy[a].append(float(med[a]))
        if "record" in med and "reactive" in med:
            deltas.append(float(med["record"]) - float(med["reactive"]))
        fr = (row.get("primary") or {}).get("frac_record_gt_reactive")
        if fr is not None:
            fracs.append(float(fr))
        if (
            med.get("record") is not None
            and med.get("reactive") is not None
            and med.get("selfplay") is not None
            and med["record"] > med["reactive"] > med["selfplay"]
        ):
            order_ok += 1

    summary = {
        "n_seeds": len(per_seed),
        "primary_pool": primary_pool,
        "primary_variant": primary_variant,
        "per_seed": per_seed,
        "across_seeds": {
            "median_attr_p_brake": {a: _seed_mean_std(by_policy[a]) for a in ALIASES},
            "record_minus_reactive": _seed_mean_std(deltas),
            "frac_record_gt_reactive": _seed_mean_std(fracs),
            "frac_seeds_order_R_gt_Rea_gt_SP": float(order_ok / len(per_seed))
            if per_seed
            else None,
            "frac_seeds_record_gt_reactive": float(
                np.mean([d > 0 for d in deltas])
            )
            if deltas
            else None,
        },
        "note": (
            "Each seed: independent collect → SAE → matching → attribution. "
            "across_seeds reports mean±std of per-seed median attributions."
        ),
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate seed-level attribution stats")
    p.add_argument(
        "--out-root",
        type=str,
        default="/data/puffer/sae/runs/attribution_policy_seeds",
    )
    p.add_argument("--primary-pool", type=str, default="max")
    p.add_argument("--primary-variant", type=str, default="raw")
    args = p.parse_args()

    out_root = Path(args.out_root)
    summary = aggregate(
        out_root, primary_pool=args.primary_pool, primary_variant=args.primary_variant
    )
    out_path = out_root / "seed_level_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"n_seeds={summary['n_seeds']}  wrote {out_path}")
    across = summary["across_seeds"]
    print(f"\n=== Across seeds ({args.primary_pool}/{args.primary_variant} median attr) ===")
    for a in ALIASES:
        s = across["median_attr_p_brake"][a]
        print(
            f"  {PRETTY[a]:<10} mean={s['mean']}  std={s['std']}  "
            f"n={s['n']}  values={s['values']}"
        )
    d = across["record_minus_reactive"]
    print(
        f"  Δ(R−Rea)   mean={d['mean']}  std={d['std']}  "
        f"frac(Δ>0)={across['frac_seeds_record_gt_reactive']}"
    )
    print(
        f"  order R>Rea>SP on {across['frac_seeds_order_R_gt_Rea_gt_SP']} of seeds"
    )


if __name__ == "__main__":
    main()
