"""Shared scene geometry metrics + behavior buckets for SAE sampling/enrichment.

Labels are for **sampling balance and analysis only** — never SAE loss.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DT_S = 0.1

# Mutually-exclusive sampling buckets (priority: rare first, normal last).
BUCKET_NAMES = (
    "closing_low_ttc",
    "intersection_conflict",
    "merge_cutin",
    "strong_accel_decel",
    "normal_interaction",
)
BUCKET_ID = {name: i for i, name in enumerate(BUCKET_NAMES)}

# Target mix for balanced SAE minibatch sampling (replacement OK for rare buckets).
DEFAULT_BUCKET_FRACS = {
    "closing_low_ttc": 0.25,
    "intersection_conflict": 0.15,
    "merge_cutin": 0.15,
    "strong_accel_decel": 0.15,
    "normal_interaction": 0.30,
}


def _safe_nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(~np.isfinite(x)):
        return float("nan")
    return float(np.nanmean(x))


def compute_row_metrics(
    *,
    ego_state: np.ndarray,
    other_state: np.ndarray,
    future_traj: np.ndarray | None = None,
    dist_at_t: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-row kinematic / future-path metrics. Arrays shaped ``(N,)``."""
    ego = np.asarray(ego_state, dtype=np.float32)
    other = np.asarray(other_state, dtype=np.float32)
    rel_xy = other[:, :2] - ego[:, :2]
    rel_dist = (
        dist_at_t.astype(np.float32)
        if dist_at_t is not None
        else np.linalg.norm(rel_xy, axis=-1).astype(np.float32)
    )
    eh = ego[:, 2]
    oh = other[:, 2]
    ego_dir = np.stack([np.cos(eh), np.sin(eh)], axis=-1)
    other_dir = np.stack([np.cos(oh), np.sin(oh)], axis=-1)
    ego_vel = ego_dir * ego[:, 3:4]
    other_vel = other_dir * other[:, 3:4]
    rel_vel = other_vel - ego_vel

    dist_safe = np.maximum(rel_dist, 1e-3)
    unit_rel = rel_xy / dist_safe[:, None]
    closing = -np.sum(rel_vel * unit_rel, axis=-1)
    # lateral approach (sideways relative to ego heading)
    ego_lat = np.stack([-np.sin(eh), np.cos(eh)], axis=-1)
    lateral_closing = -np.sum(rel_vel * ego_lat, axis=-1)
    heading_diff = np.abs(np.arctan2(np.sin(oh - eh), np.cos(oh - eh)))

    ttc = np.where(closing > 0.1, rel_dist / closing, np.nan).astype(np.float32)
    rel_speed = np.linalg.norm(rel_vel, axis=-1).astype(np.float32)

    n = ego.shape[0]
    future_min_dist = np.full(n, np.nan, dtype=np.float32)
    path_overlap = np.full(n, np.nan, dtype=np.float32)
    other_accel = np.full(n, np.nan, dtype=np.float32)
    valid_future_frac = np.zeros(n, dtype=np.float32)

    if future_traj is not None:
        ft = np.asarray(future_traj, dtype=np.float32)
        valid = np.isfinite(ft[..., 0])
        valid_future_frac = valid.mean(axis=1).astype(np.float32)
        dxy = ft[..., :2] - ego[:, None, :2]
        d = np.linalg.norm(dxy, axis=-1)
        d = np.where(valid, d, np.nan)
        with np.errstate(all="ignore"):
            future_min_dist = np.nanmin(d, axis=1).astype(np.float32)
            path_overlap = np.nanmean((d < 5.0).astype(np.float32), axis=1)
            sp = np.where(valid, ft[..., 3], np.nan)
            dv = sp[:, 1:] - sp[:, :-1]
            other_accel = np.nanmean(dv / DT_S, axis=1).astype(np.float32)

    return {
        "rel_dist": rel_dist,
        "rel_speed": rel_speed,
        "closing_speed": closing.astype(np.float32),
        "lateral_closing": lateral_closing.astype(np.float32),
        "heading_diff": heading_diff.astype(np.float32),
        "ttc": ttc,
        "future_min_dist": future_min_dist,
        "path_overlap": path_overlap.astype(np.float32),
        "other_accel": other_accel,
        "valid_future_frac": valid_future_frac,
        "other_speed": other[:, 3].astype(np.float32),
        "ego_speed": ego[:, 3].astype(np.float32),
    }


def binary_labels(metrics: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Boolean semantic tags used for enrichment numerators/denominators."""
    ttc = metrics["ttc"]
    close = metrics["closing_speed"]
    fmin = metrics["future_min_dist"]
    po = metrics["path_overlap"]
    oa = metrics["other_accel"]
    hd = metrics["heading_diff"]
    lat = metrics["lateral_closing"]

    low_ttc = np.isfinite(ttc) & (ttc < 4.0)
    closing = close > 1.0
    path_overlap = np.isfinite(po) & (po >= 0.3)
    other_not_yielding = (~np.isfinite(oa)) | (oa >= -0.5)
    # partner rarely braking among finite: treat missing accel as non-yield
    other_decel = np.isfinite(oa) & (oa < -0.5)
    conflict = low_ttc | (
        np.isfinite(fmin) & (fmin < 8.0) & (close > 0.5)
    )
    intersectionish = (hd > (np.pi / 6.0)) & (hd < (5.0 * np.pi / 6.0)) & (
        low_ttc | path_overlap | (close > 0.5)
    )
    merge_cutin = (hd <= (np.pi / 6.0)) & (
        (np.abs(lat) > 0.5) | path_overlap
    ) & (close > 0.3)
    strong_accel = np.isfinite(oa) & (np.abs(oa) > 1.5)

    return {
        "low_ttc": low_ttc,
        "closing": closing,
        "path_overlap": path_overlap,
        "other_not_yielding": other_not_yielding & ~other_decel,
        "other_decel": other_decel,
        "conflict": conflict,
        "intersectionish": intersectionish,
        "merge_cutin": merge_cutin,
        "strong_accel_decel": strong_accel,
    }


def assign_behavior_buckets(metrics: dict[str, np.ndarray]) -> np.ndarray:
    """Mutually exclusive bucket ids (int8). Rare classes win over normal."""
    labels = binary_labels(metrics)
    n = next(iter(metrics.values())).shape[0]
    out = np.full(n, BUCKET_ID["normal_interaction"], dtype=np.int8)

    # Priority order (most specific / sparse first).
    mask = labels["closing"] | labels["low_ttc"]
    out[mask] = BUCKET_ID["closing_low_ttc"]

    mask = (out == BUCKET_ID["normal_interaction"]) & labels["intersectionish"]
    out[mask] = BUCKET_ID["intersection_conflict"]

    mask = (out == BUCKET_ID["normal_interaction"]) & labels["merge_cutin"]
    out[mask] = BUCKET_ID["merge_cutin"]

    mask = (out == BUCKET_ID["normal_interaction"]) & labels["strong_accel_decel"]
    out[mask] = BUCKET_ID["strong_accel_decel"]
    return out


def bucket_counts(bucket_ids: np.ndarray) -> dict[str, int]:
    counts = {name: 0 for name in BUCKET_NAMES}
    for name, bid in BUCKET_ID.items():
        counts[name] = int((bucket_ids == bid).sum())
    return counts


def enrichment(
    label: np.ndarray,
    *,
    acts: np.ndarray,
    feature_id: int,
    modes: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Enrichment = P(label | highly active) / P(label) for several active sets.

    Parameters
    ----------
    label:
        Boolean array ``(N,)``.
    acts:
        Feature activations ``(N, d)`` or column ``(N,)``.
    """
    y = np.asarray(label, dtype=bool)
    col = acts[:, feature_id] if acts.ndim == 2 else np.asarray(acts, dtype=np.float32)
    base = float(y.mean()) if y.size else float("nan")
    if modes is None:
        n = col.shape[0]
        modes = {
            "top32": {"kind": "topk", "k": 32},
            "top100": {"kind": "topk", "k": 100},
            "top1pct": {"kind": "topk", "k": max(int(np.ceil(0.01 * n)), 1)},
            "firing": {"kind": "fire"},
        }

    out: dict[str, float] = {"base_rate": base, "feature_id": float(feature_id)}
    order = np.argsort(-col)
    for name, spec in modes.items():
        if spec["kind"] == "topk":
            k = min(int(spec["k"]), order.size)
            idx = order[:k]
        elif spec["kind"] == "fire":
            idx = np.flatnonzero(col > 0)
            if idx.size == 0:
                out[f"p_given_{name}"] = float("nan")
                out[f"enrichment_{name}"] = float("nan")
                out[f"n_{name}"] = 0.0
                continue
        else:
            raise ValueError(spec)
        p_given = float(y[idx].mean()) if idx.size else float("nan")
        out[f"p_given_{name}"] = p_given
        out[f"n_{name}"] = float(idx.size)
        if base > 1e-12 and np.isfinite(p_given):
            out[f"enrichment_{name}"] = p_given / base
        else:
            out[f"enrichment_{name}"] = float("nan")
    return out


def normalize_feature_column(col: np.ndarray) -> np.ndarray:
    """ã = (a - median) / (Q0.95 - median); clamp denom."""
    col = np.asarray(col, dtype=np.float64)
    med = float(np.median(col))
    q95 = float(np.quantile(col, 0.95))
    denom = max(q95 - med, 1e-6)
    return ((col - med) / denom).astype(np.float32)


def aggregate_profile(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    """Scalar summary over a (usually top-K) row subset."""
    ttc = metrics["ttc"]
    labels = binary_labels(metrics)
    return {
        "rel_dist_mean": _safe_nanmean(metrics["rel_dist"]),
        "rel_speed_mean": _safe_nanmean(metrics["rel_speed"]),
        "closing_speed_mean": _safe_nanmean(metrics["closing_speed"]),
        "ttc_mean": _safe_nanmean(ttc),
        "ttc_frac_below_4s": float(np.nanmean(ttc < 4.0)) if np.any(np.isfinite(ttc)) else float("nan"),
        "future_min_dist_mean": _safe_nanmean(metrics["future_min_dist"]),
        "path_overlap_mean": _safe_nanmean(metrics["path_overlap"]),
        "other_accel_mean": _safe_nanmean(metrics["other_accel"]),
        "other_decel_frac": float(labels["other_decel"].mean()) if labels["other_decel"].size else float("nan"),
        "other_not_yielding_frac": float(labels["other_not_yielding"].mean())
        if labels["other_not_yielding"].size
        else float("nan"),
        "conflict_purity": float(labels["conflict"].mean()) if labels["conflict"].size else float("nan"),
        "ego_speed_mean": _safe_nanmean(metrics["ego_speed"]),
        "other_speed_mean": _safe_nanmean(metrics["other_speed"]),
        "valid_future_frac_mean": _safe_nanmean(metrics["valid_future_frac"]),
    }
