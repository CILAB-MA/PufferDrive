"""Semantic enrichment for SAE features (preferred over top-K purity).

Computes Enrichment = P(label | highly active) / P(label) at multiple
active sets: top-32, top-100, top-1%, all firing samples.

Also keeps lightweight tags from deduplicated top scenes for quick scanning.
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

from feature_matching import resolve_model_name  # noqa: E402
from scene_metrics import (  # noqa: E402
    aggregate_profile,
    binary_labels,
    compute_row_metrics,
    enrichment,
)

ENRICH_LABELS = (
    "low_ttc",
    "closing",
    "path_overlap",
    "other_not_yielding",
    "conflict",
)


def _load_sample_metrics(exp_dir: Path) -> dict[str, np.ndarray]:
    keys = dict(np.load(exp_dir / "sample_keys.npz"))
    return compute_row_metrics(
        ego_state=keys["ego_state"],
        other_state=keys["other_state"],
        future_traj=keys.get("future_traj"),
        dist_at_t=keys.get("dist_at_t"),
    )


def build_enrichment_profiles(exp_dir: Path) -> dict:
    acts_path = exp_dir / "feature_acts.npy"
    if not acts_path.is_file():
        raise FileNotFoundError(acts_path)
    acts = np.load(acts_path, mmap_mode="r")
    metrics = _load_sample_metrics(exp_dir)
    labels = binary_labels(metrics)
    stats = np.load(exp_dir / "feature_stats.npz")
    alive = stats["alive"].astype(bool) if "alive" in stats.files else (stats["n_fire"] > 0)

    # Base rates over full validation/training split.
    base_rates = {name: float(labels[name].mean()) for name in ENRICH_LABELS}

    top = None
    top_path = exp_dir / "top_activations.npz"
    if top_path.is_file():
        top = np.load(top_path)

    profiles = []
    for fid in np.flatnonzero(alive).tolist():
        row: dict = {
            "feature_id": int(fid),
            "density": float(stats["density"][fid]),
            "n_fire": int(stats["n_fire"][fid]),
            "base_rates": base_rates,
            "enrichment": {},
        }
        for name in ENRICH_LABELS:
            row["enrichment"][name] = enrichment(
                labels[name], acts=acts, feature_id=int(fid)
            )

        # Deduped top-set profile (for tags / diversity echo).
        if top is not None and "feature_id" in top.files:
            ids = top["feature_id"]
            pos = np.flatnonzero(ids == fid)
            if pos.size:
                i = int(pos[0])
                valid = top["top_valid"][i] if "top_valid" in top.files else top["top_row_idx"][i] >= 0
                if np.any(valid):
                    top_metrics = compute_row_metrics(
                        ego_state=top["top_ego_state"][i][valid],
                        other_state=top["top_other_state"][i][valid],
                        future_traj=top["top_future_traj"][i][valid]
                        if "top_future_traj" in top.files
                        else None,
                        dist_at_t=top["top_dist_at_t"][i][valid]
                        if "top_dist_at_t" in top.files
                        else None,
                    )
                    row["top_profile"] = aggregate_profile(top_metrics)
                    row["n_top_scenes"] = int(valid.sum())
                    if "diversity_n_unique_scene" in top.files:
                        row["n_unique_scene"] = int(top["diversity_n_unique_scene"][i])
                        row["n_unique_traj"] = int(top["diversity_n_unique_traj"][i])
                        row["valid_future_frac"] = float(
                            top["diversity_valid_future_frac"][i]
                        )
                        row["nan_frac"] = float(top["diversity_nan_frac"][i])

        # Primary score: low_ttc enrichment at top1% (stable across thresholds preferred).
        e_low = row["enrichment"]["low_ttc"]
        e_conf = row["enrichment"]["conflict"]
        row["enrichment_score"] = float(
            np.nanmean(
                [
                    e_low.get("enrichment_top1pct", np.nan),
                    e_low.get("enrichment_top32", np.nan),
                    e_conf.get("enrichment_top1pct", np.nan),
                ]
            )
        )
        # Tag if enrichment sustained (top32 and top1pct both >= 1.5).
        tags = []
        for name in ENRICH_LABELS:
            e = row["enrichment"][name]
            if (
                np.isfinite(e.get("enrichment_top32", np.nan))
                and np.isfinite(e.get("enrichment_top1pct", np.nan))
                and e["enrichment_top32"] >= 1.5
                and e["enrichment_top1pct"] >= 1.5
            ):
                tags.append(f"enriched_{name}")
        # ReCord-relevant composites
        if "enriched_other_not_yielding" in tags and (
            "enriched_path_overlap" in tags or "enriched_conflict" in tags
        ):
            tags.append("record_relevant")
        row["tags"] = tags
        profiles.append(row)

    profiles.sort(key=lambda r: -(r.get("enrichment_score") or 0.0))
    return {
        "profiles": profiles,
        "base_rates": base_rates,
        "n_alive": int(alive.sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="SAE feature semantic enrichment")
    p.add_argument("--analysis-dir", type=str, required=True)
    p.add_argument(
        "--experiments",
        type=str,
        default="record,reactive,selfplay",
        help="Comma-separated model aliases or folder names",
    )
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "semantics"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = {"experiments": []}
    for raw in args.experiments.split(","):
        raw = raw.strip()
        if not raw:
            continue
        exp = resolve_model_name(raw)
        exp_dir = analysis_dir / exp
        if not (exp_dir / "feature_acts.npy").is_file():
            print(f"skip {exp}: no feature_acts.npy")
            continue
        print(f"=== enrichment {exp} ===")
        result = build_enrichment_profiles(exp_dir)
        path = out_dir / f"{exp}_enrichment.json"
        path.write_text(json.dumps(result["profiles"], indent=2))
        # Compact scores for temporal / matching consumers
        compact = [
            {
                "feature_id": r["feature_id"],
                "enrichment_score": r["enrichment_score"],
                "tags": r["tags"],
                "low_ttc_top32": r["enrichment"]["low_ttc"].get("enrichment_top32"),
                "low_ttc_top1pct": r["enrichment"]["low_ttc"].get("enrichment_top1pct"),
                "conflict_top1pct": r["enrichment"]["conflict"].get("enrichment_top1pct"),
                "other_not_yielding_top1pct": r["enrichment"]["other_not_yielding"].get(
                    "enrichment_top1pct"
                ),
                "path_overlap_top1pct": r["enrichment"]["path_overlap"].get(
                    "enrichment_top1pct"
                ),
            }
            for r in result["profiles"]
        ]
        (out_dir / f"{exp}_enrichment_scores.json").write_text(
            json.dumps(compact, indent=2)
        )
        (out_dir / f"{exp}_base_rates.json").write_text(
            json.dumps(result["base_rates"], indent=2)
        )
        index["experiments"].append(exp)
        top5 = result["profiles"][:5]
        print(f"  base_rates={result['base_rates']}")
        for t in top5:
            print(
                f"  feat {t['feature_id']}: score={t['enrichment_score']:.2f} "
                f"tags={t['tags']} "
                f"low_ttc_top1%={t['enrichment']['low_ttc'].get('enrichment_top1pct')}"
            )
        print(f"  wrote {path}")

    (out_dir / "index.json").write_text(
        json.dumps({"experiments": index["experiments"], "out_dir": str(out_dir)}, indent=2)
    )


if __name__ == "__main__":
    main()
