"""Small shared stats helpers for SAE attribution / analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats


def load_enrichment_scores(semantics_dir: Path, exp: str) -> dict[int, float]:
    path = semantics_dir / f"{exp}_enrichment_scores.json"
    if not path.is_file():
        path = semantics_dir / f"{exp}_enrichment.json"
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text())
    out = {}
    for r in rows:
        fid = int(r["feature_id"])
        score = r.get("enrichment_score")
        if score is None:
            score = r.get("low_ttc_top1pct") or r.get("conflict_top1pct") or 0.0
        out[fid] = float(score) if score is not None else 0.0
    return out


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: str = "median",
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "point": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    fn = np.median if statistic == "median" else np.mean
    point = float(fn(values))
    boots = np.empty(n_boot, dtype=np.float64)
    n = values.size
    for i in range(n_boot):
        boots[i] = fn(values[rng.integers(0, n, size=n)])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {"n": int(n), "point": point, "ci_low": lo, "ci_high": hi, "statistic": statistic}


def wilcoxon_onesided_greater(deltas: np.ndarray) -> dict:
    """H1: median(Δ) > 0."""
    x = np.asarray(deltas, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[x != 0]
    if x.size < 5:
        return {"n": int(x.size), "statistic": None, "pvalue": None, "note": "too few non-zero pairs"}
    try:
        res = scipy_stats.wilcoxon(x, alternative="greater", zero_method="wilcox")
        return {
            "n": int(x.size),
            "statistic": float(res.statistic),
            "pvalue": float(res.pvalue),
        }
    except ValueError as exc:
        return {"n": int(x.size), "statistic": None, "pvalue": None, "note": str(exc)}
