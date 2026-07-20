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
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_SAE_DIR = Path(__file__).resolve().parent
if str(_SAE_DIR) not in sys.path:
    sys.path.insert(0, str(_SAE_DIR))

from stats_utils import bootstrap_ci  # noqa: E402
from feature_steering import EXTENDED_PRIMARY_METRICS  # noqa: E402

PRIMARY_METRIC = "attr_p_brake"


ALIASES = ("record", "reactive", "selfplay")
PRETTY = {"record": "ReCord", "reactive": "Reactive", "selfplay": "Self-play"}
VARIANTS = ("raw", "cos", "scaled")
PAIRING_MODES = ("auto", "paired", "unpaired")


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
                    paired_v = (block.get("paired") or {}).get(v) or {}
                    if isinstance(paired_v, dict) and PRIMARY_METRIC in paired_v:
                        primary_block = paired_v.get(PRIMARY_METRIC) or {}
                        pooled = primary_block.get("pooled") or {}
                        w = pooled.get("wilcoxon_record_gt_reactive") or {}
                        by_metric = {}
                        for metric in EXTENDED_PRIMARY_METRICS:
                            mb = paired_v.get(metric) or {}
                            mp = mb.get("pooled") or {}
                            mw = mp.get("wilcoxon_record_gt_reactive") or {}
                            by_metric[metric] = {
                                "frac_record_gt_reactive": mp.get("frac_record_gt_reactive"),
                                "median_delta": mp.get("median_delta"),
                                "wilcoxon_p": mw.get("pvalue"),
                                "median_attr_by_policy": mb.get("mean_median_attr"),
                            }
                        headline[pm][v] = {
                            "frac_record_gt_reactive": pooled.get("frac_record_gt_reactive"),
                            "median_delta": pooled.get("median_delta"),
                            "wilcoxon_p": w.get("pvalue"),
                            "median_attr_by_policy": primary_block.get("mean_median_attr"),
                            "by_metric": by_metric,
                            "cross_metric_consistency": (
                                block.get("cross_metric_consistency") or {}
                            ).get(v),
                        }
                        continue
                    pooled = paired_v.get("pooled") or {}
                    w = pooled.get("wilcoxon_record_gt_reactive") or {}
                    headline[pm][v] = {
                        "frac_record_gt_reactive": pooled.get("frac_record_gt_reactive"),
                        "median_delta": pooled.get("median_delta"),
                        "wilcoxon_p": w.get("pvalue"),
                        "median_attr_by_policy": paired_v.get("mean_median_attr"),
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


def _pairing_assessment(per_seed: list[dict], pairing_mode: str) -> dict:
    if pairing_mode == "paired":
        return {
            "mode": "paired",
            "is_paired": True,
            "reason": "Forced by CLI flag.",
        }
    if pairing_mode == "unpaired":
        return {
            "mode": "unpaired",
            "is_paired": False,
            "reason": "Forced by CLI flag.",
        }

    shared_keys = []
    for row in per_seed:
        info = row.get("seed_info") or {}
        if info.get("shared_training_seed") is not None:
            shared_keys.append(info["shared_training_seed"])
        elif info.get("pairing_key") is not None:
            shared_keys.append(info["pairing_key"])
        else:
            return {
                "mode": "unpaired",
                "is_paired": False,
                "reason": (
                    "No explicit shared seed / pairing key found in seed metadata. "
                    "Directory sort order alone is not treated as evidence of true pairing."
                ),
            }
    return {
        "mode": "paired",
        "is_paired": True,
        "reason": (
            "All seeds provide an explicit shared seed / pairing key; paired summaries are allowed."
        ),
        "pairing_keys": shared_keys,
    }


def _exact_unpaired_permutation_test(xs: list[float], ys: list[float]) -> dict | None:
    if not xs or not ys:
        return None
    pooled = np.asarray(xs + ys, dtype=np.float64)
    n_x = len(xs)
    observed = float(np.mean(xs) - np.mean(ys))
    total = math.comb(len(pooled), n_x)
    ge = 0
    abs_ge = 0
    for idxs in combinations(range(len(pooled)), n_x):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(idxs)] = True
        diff = float(pooled[mask].mean() - pooled[~mask].mean())
        if diff >= observed - 1e-15:
            ge += 1
        if abs(diff) >= abs(observed) - 1e-15:
            abs_ge += 1
    return {
        "statistic": "mean_difference",
        "observed_record_minus_reactive": observed,
        "pvalue_one_sided_record_gt_reactive": float(ge / total),
        "pvalue_two_sided": float(abs_ge / total),
        "n_assignments": int(total),
        "note": "Exact unpaired permutation test over seed-level values.",
    }


def _cross_product_deltas(xs: list[float], ys: list[float]) -> dict:
    deltas = [float(x - y) for x in xs for y in ys]
    out = _seed_mean_std(deltas)
    out["frac_gt_zero"] = float(np.mean(np.asarray(deltas) > 0)) if deltas else None
    out["note"] = "All record-reactive seed cross-product differences."
    return out


def _metric_block(per_seed: list[dict], pool: str, variant: str, metric: str) -> dict:
    """Within-seed paired summaries for one attribution metric."""
    by_policy: dict[str, list[float]] = {a: [] for a in ALIASES}
    paired_deltas: list[float] = []
    fracs: list[float] = []
    order_ok = 0
    n_rows = 0
    for row in per_seed:
        h = (((row.get("headline") or {}).get(pool) or {}).get(variant) or {})
        by_metric = h.get("by_metric") or {}
        mblock = by_metric.get(metric) or {}
        if metric == PRIMARY_METRIC and not mblock:
            med = h.get("median_attr_by_policy") or {}
            fr = h.get("frac_record_gt_reactive")
            md = h.get("median_delta")
            if med:
                mblock = {
                    "median_attr_by_policy": med,
                    "frac_record_gt_reactive": fr,
                    "median_delta": md,
                }
        med = mblock.get("median_attr_by_policy") or {}
        if not med:
            continue
        n_rows += 1
        for a in ALIASES:
            if a in med and med[a] is not None:
                by_policy[a].append(float(med[a]))
        if "record" in med and "reactive" in med:
            paired_deltas.append(float(med["record"]) - float(med["reactive"]))
        fr = mblock.get("frac_record_gt_reactive")
        if fr is not None:
            fracs.append(float(fr))
        if (
            med.get("record") is not None
            and med.get("reactive") is not None
            and med.get("selfplay") is not None
            and med["record"] > med["reactive"] > med["selfplay"]
        ):
            order_ok += 1

    record_vals = by_policy["record"]
    reactive_vals = by_policy["reactive"]
    return {
        "metric": metric,
        "n_seed_triplets": n_rows,
        "median_attr_by_policy": {a: _seed_mean_std(by_policy[a]) for a in ALIASES},
        "record_minus_reactive_within_seed": _seed_mean_std(paired_deltas),
        "frac_record_gt_reactive_obs_pooled": _seed_mean_std(fracs),
        "frac_seeds_record_gt_reactive_within_seed": (
            float(np.mean([d > 0 for d in paired_deltas])) if paired_deltas else None
        ),
        "frac_seeds_order_R_gt_Rea_gt_SP": float(order_ok / n_rows) if n_rows else None,
        "record_minus_reactive_cross_product": _cross_product_deltas(
            record_vals, reactive_vals
        ),
        "record_vs_reactive_unpaired_permutation": _exact_unpaired_permutation_test(
            record_vals, reactive_vals
        ),
    }


def _metric_summary(per_seed: list[dict], pool: str, variant: str) -> dict:
    """Legacy wrapper: primary brake metric + extended set."""
    blocks = {
        metric: _metric_block(per_seed, pool, variant, metric)
        for metric in EXTENDED_PRIMARY_METRICS
    }
    primary = blocks[PRIMARY_METRIC]
    return {
        **primary,
        "median_attr_p_brake": primary["median_attr_by_policy"],
        "by_metric": blocks,
    }


def aggregate(
    out_root: Path,
    *,
    primary_pool: str = "max",
    primary_variant: str = "scaled",
    pairing_mode: str = "auto",
) -> dict:
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
            "seed_info": info,
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

    pairing = _pairing_assessment(per_seed, pairing_mode)
    headline_data = [row.get("headline") or {} for row in per_seed]
    pools = sorted({pool for h in headline_data for pool in h.keys()})
    metric_summaries = {
        pool: {
            variant: _metric_summary(per_seed, pool, variant)
            for variant in VARIANTS
            if any(variant in (h.get(pool) or {}) for h in headline_data)
        }
        for pool in pools
    }

    summary = {
        "n_seeds": len(per_seed),
        "primary_pool": primary_pool,
        "primary_variant": primary_variant,
        "pairing": pairing,
        "per_seed": per_seed,
        "across_seed_triplets": metric_summaries.get(primary_pool, {}).get(primary_variant, {}),
        "all_pool_variant_summaries": metric_summaries,
        "note": (
            "Each seed triplet: independent collect → SAE → matching → attribution. "
            "Method comparison is within-seed (record vs reactive vs selfplay in same triplet), "
            "then aggregated across seed triplets. Index-aligned triplets are not guaranteed "
            "to share the same PBT training seed unless seed_info.shared_training_seed is set."
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
    p.add_argument("--primary-variant", type=str, default="scaled")
    p.add_argument(
        "--pairing-mode",
        type=str,
        choices=PAIRING_MODES,
        default="auto",
        help="auto requires explicit seed metadata before allowing paired summaries",
    )
    args = p.parse_args()

    out_root = Path(args.out_root)
    summary = aggregate(
        out_root,
        primary_pool=args.primary_pool,
        primary_variant=args.primary_variant,
        pairing_mode=args.pairing_mode,
    )
    out_path = out_root / "seed_level_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"n_seeds={summary['n_seeds']}  wrote {out_path}")
    print(
        f"pairing={summary['pairing']['mode']}  reason={summary['pairing']['reason']}"
    )
    across = summary["across_seed_triplets"]
    print(f"\n=== Across seed triplets ({args.primary_pool}/{args.primary_variant}) ===")
    for metric in EXTENDED_PRIMARY_METRICS:
        block = across.get("by_metric", {}).get(metric) or across
        if metric != PRIMARY_METRIC and "by_metric" in across:
            block = across["by_metric"][metric]
        print(f"\n--- {metric} ---")
        for a in ALIASES:
            med_key = "median_attr_by_policy" if metric != PRIMARY_METRIC else "median_attr_p_brake"
            policy_stats = block.get(med_key) or block.get("median_attr_by_policy") or {}
            s = policy_stats.get(a) or {}
            print(
                f"  {PRETTY[a]:<10} mean={s.get('mean')}  std={s.get('std')}  "
                f"n={s.get('n')}  values={s.get('values')}"
            )
        wd = block.get("record_minus_reactive_within_seed") or {}
        print(
            f"  Δ(R−Rea) within-seed mean={wd.get('mean')}  std={wd.get('std')}  "
            f"frac(Δ>0)={block.get('frac_seeds_record_gt_reactive_within_seed')}"
        )
    d = across.get("record_minus_reactive_cross_product") or {}
    print(
        f"\n  Δ(R−Rea) cross-product (unpaired, primary): mean={d.get('mean')}  "
        f"std={d.get('std')}  frac(Δ>0)={d.get('frac_gt_zero')}"
    )
    perm = across.get("record_vs_reactive_unpaired_permutation")
    if perm:
        print(
            "  unpaired perm "
            f"p(one-sided R>Rea)={perm['pvalue_one_sided_record_gt_reactive']}  "
            f"p(two-sided)={perm['pvalue_two_sided']}"
        )
    print(
        f"  order R>Rea>SP on {across.get('frac_seeds_order_R_gt_Rea_gt_SP')} of seeds"
    )


if __name__ == "__main__":
    main()
