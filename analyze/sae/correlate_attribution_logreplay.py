#!/usr/bin/env python3
"""Correlate per-policy attribution with human-replay driving performance.

Each policy checkpoint (record/reactive/selfplay × seed) gets:
  - attribution median (from seed*/validation summary)
  - logreplay metrics (ego_score, collision_rate, ... from logreplay.json)

Also reports within-seed-triplet paired deltas vs performance deltas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

_SAE_DIR = Path(__file__).resolve().parent
if str(_SAE_DIR) not in sys.path:
    sys.path.insert(0, str(_SAE_DIR))

ALIASES = ("record", "reactive", "selfplay")
EXP_MAP = {
    "record": "replay_0.25",
    "reactive": "reactive_0.25",
    "selfplay": "selfplay",
}
DEFAULT_METRICS = (
    "ego_score",
    "score",
    "completion_rate",
    "ego_collision_rate",
    "collision_rate",
    "ego_collisions_per_agent",
    "collisions_per_agent",
    "episode_return",
    "ego_lane_alignment_rate",
)


def load_logreplay(results_root: Path) -> dict[str, dict[str, float]]:
    """model_id -> {metric: value}"""
    out: dict[str, dict[str, float]] = {}
    for alias, exp in EXP_MAP.items():
        path = results_root / exp / "logreplay.json"
        if not path.is_file():
            continue
        entries = json.loads(path.read_text())
        for block in entries:
            if not isinstance(block, dict):
                continue
            for model_id, metrics in block.items():
                if not isinstance(metrics, dict):
                    continue
                out[f"{alias}:{model_id}"] = {
                    k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
                }
                out[model_id] = out[f"{alias}:{model_id}"]
    return out


def _normalize_run_id(run_path: str) -> str:
    p = Path(run_path)
    name = p.name
    if name.startswith("puffer_drive_"):
        name = name[len("puffer_drive_") :]
    if name.endswith(".pt"):
        name = name[:-3]
    if name.startswith("pbt_"):
        name = name[4:]
    return name


def load_seed_rows(
    out_root: Path,
    *,
    pool: str = "max",
    variant: str = "raw",
) -> list[dict]:
    rows = []
    for sd in sorted(out_root.glob("seed*/")):
        slim = sd / "validation" / "attribution_validation_summary.json"
        info_path = sd / "seed_info.json"
        if not slim.is_file() or not info_path.is_file():
            continue
        slim_d = json.loads(slim.read_text())
        info = json.loads(info_path.read_text())
        h = (slim_d.get("headline") or {}).get(pool, {}).get(variant, {})
        med = h.get("median_attr_by_policy") or {}
        policy_runs = info.get("policy_runs") or {}
        row = {
            "seed_dir": sd.name,
            "seed_index": info.get("seed_index"),
            "frac_record_gt_reactive": h.get("frac_record_gt_reactive"),
            "median_delta_record_reactive": h.get("median_delta"),
            "attr": {a: med.get(a) for a in ALIASES},
            "run_ids": {a: _normalize_run_id(policy_runs[a]) for a in ALIASES if a in policy_runs},
        }
        rows.append(row)
    return rows


def flatten_policy_rows(seed_rows: list[dict], logreplay: dict[str, dict[str, float]]) -> list[dict]:
    flat = []
    for sr in seed_rows:
        for alias in ALIASES:
            run_id = sr["run_ids"].get(alias)
            attr = sr["attr"].get(alias)
            if run_id is None or attr is None:
                continue
            perf = logreplay.get(run_id) or logreplay.get(f"{alias}:{run_id}")
            flat.append(
                {
                    "seed_dir": sr["seed_dir"],
                    "seed_index": sr["seed_index"],
                    "alias": alias,
                    "run_id": run_id,
                    "attr_median": float(attr),
                    "logreplay": perf or {},
                }
            )
    return flat


def corr_safe(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return {"n": int(x.size), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr = scipy_stats.pearsonr(x, y)
    sr = scipy_stats.spearmanr(x, y)
    return {
        "n": int(x.size),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Attribution vs log-replay performance correlation")
    p.add_argument(
        "--out-root",
        type=str,
        default="/data/puffer/sae/runs/attribution_policy_seeds",
    )
    p.add_argument(
        "--results-root",
        type=str,
        default="/data/puffer/results",
    )
    p.add_argument("--pool", type=str, default="max")
    p.add_argument("--variant", type=str, default="raw")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    out_root = Path(args.out_root)
    logreplay = load_logreplay(Path(args.results_root))
    seed_rows = load_seed_rows(out_root, pool=args.pool, variant=args.variant)
    flat = flatten_policy_rows(seed_rows, logreplay)

    report: dict = {
        "pool": args.pool,
        "variant": args.variant,
        "n_seed_triplets": len(seed_rows),
        "n_policy_rows": len(flat),
        "note": (
            "Each seed triplet = one aligned index (rec_i, rea_i, sp_i). "
            "Attribution is computed per triplet with that triplet's own SAE/matching. "
            "Correlations use n=9 independent policy checkpoints (3 methods × 3 seeds)."
        ),
        "per_policy": flat,
        "correlations_attr_vs_logreplay": {},
        "within_seed_triplet": [],
    }

    attrs = np.array([r["attr_median"] for r in flat], dtype=np.float64)
    for metric in DEFAULT_METRICS:
        ys = np.array([r["logreplay"].get(metric, np.nan) for r in flat], dtype=np.float64)
        if np.isfinite(ys).sum() >= 3:
            report["correlations_attr_vs_logreplay"][metric] = corr_safe(attrs, ys)

    # By method only (n=3 each — descriptive only)
    report["correlations_by_method"] = {}
    for alias in ALIASES:
        sub = [r for r in flat if r["alias"] == alias]
        xs = np.array([r["attr_median"] for r in sub])
        report["correlations_by_method"][alias] = {}
        for metric in DEFAULT_METRICS:
            ys = np.array([r["logreplay"].get(metric, np.nan) for r in sub])
            if np.isfinite(ys).sum() >= 3:
                report["correlations_by_method"][alias][metric] = corr_safe(xs, ys)

    # Within-seed paired: attr delta vs perf delta (n=3 seed triplets)
    for sr in seed_rows:
        perf = {}
        attr = sr["attr"]
        for alias in ALIASES:
            rid = sr["run_ids"].get(alias)
            if rid:
                perf[alias] = logreplay.get(rid) or logreplay.get(f"{alias}:{rid}") or {}
        trip = {
            "seed_dir": sr["seed_dir"],
            "seed_index": sr["seed_index"],
            "attr_record_minus_reactive": (
                float(attr["record"]) - float(attr["reactive"])
                if attr.get("record") is not None and attr.get("reactive") is not None
                else None
            ),
            "frac_record_gt_reactive": sr.get("frac_record_gt_reactive"),
            "perf_record_minus_reactive": {},
        }
        if perf.get("record") and perf.get("reactive"):
            for metric in DEFAULT_METRICS:
                if metric in perf["record"] and metric in perf["reactive"]:
                    trip["perf_record_minus_reactive"][metric] = float(
                        perf["record"][metric] - perf["reactive"][metric]
                    )
        report["within_seed_triplet"].append(trip)

    # Correlate within-seed attr delta vs perf delta (n=3)
    attr_deltas = []
    perf_deltas: dict[str, list[float]] = {m: [] for m in DEFAULT_METRICS}
    for trip in report["within_seed_triplet"]:
        ad = trip.get("attr_record_minus_reactive")
        if ad is None:
            continue
        attr_deltas.append(ad)
        for m, v in trip.get("perf_record_minus_reactive", {}).items():
            perf_deltas[m].append(v)
    attr_deltas = np.asarray(attr_deltas, dtype=np.float64)
    report["correlations_within_seed_delta"] = {}
    for m, vals in perf_deltas.items():
        if len(vals) == len(attr_deltas) and len(vals) >= 3:
            report["correlations_within_seed_delta"][m] = corr_safe(
                attr_deltas, np.asarray(vals, dtype=np.float64)
            )

    out_path = Path(args.out) if args.out else out_root / "attribution_logreplay_correlation.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    print(f"n_seed_triplets={report['n_seed_triplets']}  n_policy_rows={report['n_policy_rows']}")
    print("\n=== attr vs logreplay (all 9 policies, Spearman) ===")
    for m, c in report["correlations_attr_vs_logreplay"].items():
        print(f"  {m}: rho={c['spearman_r']:+.3f} p={c['spearman_p']:.4f} (n={c['n']})")
    print("\n=== within-seed Δ(attr) vs Δ(perf) R−Rea (n=3) ===")
    for m, c in report.get("correlations_within_seed_delta", {}).items():
        print(f"  {m}: rho={c['spearman_r']:+.3f} p={c['spearman_p']:.4f}")


if __name__ == "__main__":
    main()
