#!/usr/bin/env python3
"""Shared constants and small helpers for coordination analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

METHODS = {
    "record": "replay_0.25",
    "reactive": "reactive_0.25",
    "selfplay": "selfplay",
}
PRETTY = {"record": "ReCord", "reactive": "Reactive", "selfplay": "Self-play"}
ORDER = ("record", "reactive", "selfplay")

ACCEL_VALUES_NP = np.array([-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0], dtype=np.float64)
ACCEL_VALUES = torch.tensor(
    [-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0], dtype=torch.float32
)
STEER_VALUES = torch.tensor(
    [
        -1.000,
        -0.833,
        -0.667,
        -0.500,
        -0.333,
        -0.167,
        0.000,
        0.167,
        0.333,
        0.500,
        0.667,
        0.833,
        1.000,
    ],
    dtype=torch.float32,
)
N_STEER = 13
N_ACTIONS = 7 * N_STEER

DEFAULT_THRESHOLDS = dict(
    ttc_approach=4.0,
    closing_approach=0.5,
    dist_approach=25.0,
    ttc_tight=1.5,
    closing_tight=1.0,
    dist_tight=12.0,
    pre_window=20,
    hard_brake=-2.667,
)

def list_policy_ckpts(experiments_root: Path, exp: str, *, max_n: int | None = 3) -> list[Path]:
    preferred = {
        "replay_0.25": ("pbt_2iko0ovs", "pbt_ob5mu6jp", "pbt_veyjl2oa"),
        "reactive_0.25": ("pbt_da3kfc4q", "pbt_rq33g301", "pbt_vjy3d8rz"),
        "selfplay": ("e729wx4e", "q2lad528", "zfvp9288"),
    }
    root = experiments_root / exp
    out: list[Path] = []
    for pid in preferred.get(exp, ()):
        path = root / f"puffer_drive_{pid}.pt"
        if path.is_file():
            out.append(path)
    if out:
        return out[: max_n or len(out)]
    paths = sorted(root.glob("puffer_drive_*.pt"))
    if max_n is not None:
        paths = paths[:max_n]
    return paths


def nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def frac(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


def subset_mean(x: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return nanmean(x[mask])


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def accel_from_actions(action_np: np.ndarray) -> np.ndarray:
    a = np.asarray(action_np)
    if a.ndim >= 1 and np.issubdtype(a.dtype, np.floating) and a.size and float(np.nanmax(np.abs(a))) <= 5.0:
        return a.reshape(-1, a.shape[-1])[:, 0].astype(np.float64) if a.ndim > 1 else a.astype(np.float64)
    flat = a.reshape(-1).astype(np.int64)
    accel_idx = (flat // N_STEER).clip(0, 6)
    return ACCEL_VALUES_NP[accel_idx]


def nearest_from_states(
    ego_xy: np.ndarray,
    ego_heading: float,
    ego_speed: float,
    other_xy: np.ndarray,
    other_heading: np.ndarray,
    other_speed: np.ndarray,
    other_id: np.ndarray,
) -> tuple[float, float, float]:
    valid = other_id >= 0
    if not np.any(valid):
        return float("nan"), 0.0, float("nan")
    rel = other_xy[valid] - ego_xy[None, :]
    dist = np.linalg.norm(rel, axis=-1)
    j = int(np.argmin(dist))
    d = float(dist[j])
    if d < 1e-3:
        return d, 0.0, 0.0
    unit = rel[j] / d
    eh = float(ego_heading)
    oh = float(other_heading[valid][j])
    ev = np.array([np.cos(eh), np.sin(eh)], dtype=np.float64) * float(ego_speed)
    ov = np.array([np.cos(oh), np.sin(oh)], dtype=np.float64) * float(other_speed[valid][j])
    closing = float(-np.dot(ov - ev, unit))
    ttc = d / closing if closing > 0.1 else float("nan")
    return d, closing, ttc


def action_stats_from_logits(actions) -> dict[str, torch.Tensor]:
    """Brake / throttle / steer / wait / gap-press / entropy from 91-bin policy logits."""
    if isinstance(actions, (tuple, list)):
        logit = actions[0]
    elif torch.is_tensor(actions):
        logit = actions
    else:
        raise TypeError(f"unsupported action type: {type(actions)}")

    if logit.shape[-1] != N_ACTIONS:
        raise ValueError(
            f"expected discrete classic Drive logits with last dim {N_ACTIONS}, "
            f"got {tuple(logit.shape)}"
        )

    probs = torch.softmax(logit.float(), dim=-1)
    device = logit.device
    accel_tbl = ACCEL_VALUES.to(device)
    steer_tbl = STEER_VALUES.to(device)
    a_idx = torch.arange(N_ACTIONS, device=device) // N_STEER
    s_idx = torch.arange(N_ACTIONS, device=device) % N_STEER
    accel = (probs * accel_tbl[a_idx]).sum(dim=-1)
    steer = (probs * steer_tbl[s_idx]).sum(dim=-1)
    p_brake = probs[:, a_idx < 3].sum(dim=-1)
    p_throttle = probs[:, a_idx > 3].sum(dim=-1)
    p_wait = probs[:, a_idx == 3].sum(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return {
        "accel": accel,
        "steer": steer,
        "steer_mag": steer.abs(),
        "p_brake": p_brake,
        "p_throttle": p_throttle,
        "p_neg_accel": probs[:, accel_tbl[a_idx] < 0].sum(dim=-1),
        "p_strong_brake": probs[:, a_idx == 0].sum(dim=-1),
        "brake_proxy": -accel,
        "log_p_brake": p_brake.clamp_min(1e-8).log(),
        "brake_minus_throttle": p_brake - p_throttle,
        "entropy": entropy,
        "p_wait": p_wait,
        "gap_press": p_throttle - p_brake,
        "p_yield": p_brake,
    }


def aggregate_numeric_across_seeds(
    per_seed: list[dict[str, Any]],
    *,
    skip: set[str] | None = None,
) -> dict[str, Any]:
    """Mean/std over seed replicates for flat numeric fields."""
    skip = skip or {"seed_index", "ckpts", "headline"}
    keys = [
        k
        for k, v in per_seed[0].items()
        if k not in skip and isinstance(v, (int, float)) and v is not None
    ]
    across: dict[str, Any] = {}
    for k in keys:
        vs = []
        for s in per_seed:
            val = s.get(k)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                continue
            vs.append(float(val))
        across[k] = {
            "values": vs,
            "n": len(vs),
            "mean": float(np.mean(vs)) if vs else None,
            "std": float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0,
        }
    return across


def paired_mean_ci(
    diff: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Map-level paired mean difference with percentile bootstrap CI."""
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "alpha": alpha,
        }
    mean = float(d.mean())
    if n == 1 or n_boot <= 0:
        return {"n": n, "mean": mean, "ci_low": mean, "ci_high": mean, "alpha": alpha}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return {"n": n, "mean": mean, "ci_low": lo, "ci_high": hi, "alpha": alpha}


def hierarchical_seed_map_ci(
    per_seed_diffs: list[np.ndarray],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Hierarchical bootstrap: resample seeds, then maps within each seed.

    Each element of ``per_seed_diffs`` is a 1-D array of map-level paired
    differences for one training seed. The statistic is the mean of seed means.
    """
    cleaned: list[np.ndarray] = []
    for d in per_seed_diffs:
        arr = np.asarray(d, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            cleaned.append(arr)
    n_seeds = len(cleaned)
    if n_seeds == 0:
        return {
            "n_seeds": 0,
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "alpha": alpha,
        }
    seed_means = np.array([float(a.mean()) for a in cleaned], dtype=np.float64)
    mean = float(seed_means.mean())
    if n_seeds == 1 or n_boot <= 0:
        # Fall back to map bootstrap within the single seed.
        return {
            **paired_mean_ci(cleaned[0], n_boot=n_boot, alpha=alpha, seed=seed),
            "n_seeds": n_seeds,
            "mean": mean,
        }
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        vals = []
        for si in seed_idx:
            d = cleaned[int(si)]
            map_idx = rng.integers(0, d.size, size=d.size)
            vals.append(float(d[map_idx].mean()))
        boots[b] = float(np.mean(vals))
    return {
        "n_seeds": n_seeds,
        "mean": mean,
        "ci_low": float(np.quantile(boots, alpha / 2.0)),
        "ci_high": float(np.quantile(boots, 1.0 - alpha / 2.0)),
        "alpha": alpha,
        "seed_means": seed_means.tolist(),
    }


def mcnemar_exact(n_pos: int, n_neg: int) -> dict[str, float | int]:
    """Exact two-sided McNemar test on off-diagonal counts (Bin(n, 1/2))."""
    n_pos = int(n_pos)
    n_neg = int(n_neg)
    n = n_pos + n_neg
    out: dict[str, float | int] = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_discordant": n,
        "p_value": float("nan"),
    }
    if n == 0:
        return out
    # Two-sided exact binomial: sum dens of outcomes with dens <= dens(n_pos).
    # Under p=1/2, dens(k) = C(n,k) / 2^n.
    log_c = np.zeros(n + 1, dtype=np.float64)
    for k in range(1, n + 1):
        log_c[k] = log_c[k - 1] + np.log(n - k + 1) - np.log(k)
    log_dens = log_c - n * np.log(2.0)
    dens = np.exp(log_dens - log_dens.max())
    dens = dens / dens.sum()
    thresh = dens[n_pos]
    p = float(dens[dens <= thresh + 1e-15].sum())
    out["p_value"] = float(min(1.0, max(0.0, p)))
    return out


def stage_probs(pack: dict[str, np.ndarray]) -> dict[str, float]:
    """Interaction-stage probabilities for one method pack.

    Code definitions: resolved = approach & ~tight, escalated = approach & tight.
    Headline reports Pr(resolved | approach) and Pr(collided | tight).
    """
    ap = pack["had_approach"].astype(bool)
    tight = pack["had_tight"].astype(bool)
    resolved = pack["resolved"].astype(bool)
    coll = pack["collided"].astype(bool)
    n_ap = int(ap.sum())
    n_tight = int(tight.sum())
    return {
        "pr_approach": frac(ap),
        "pr_tight_given_approach": float(tight[ap].mean()) if n_ap else float("nan"),
        "pr_resolved_given_approach": float(resolved[ap].mean()) if n_ap else float("nan"),
        "pr_collided_given_tight": float(coll[tight].mean()) if n_tight else float("nan"),
    }


def auroc_score(y: np.ndarray, scores: np.ndarray) -> float:
    """Mann–Whitney AUROC (no sklearn)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    pos, neg = s[y], s[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Vectorized pairwise: P(pos > neg) + 0.5 P(pos == neg)
    # Sort and rank for efficiency
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # Average ranks for ties
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + 1 + j + 1)
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos_ranks = float(ranks[y].sum())
    n_pos, n_neg = float(pos.size), float(neg.size)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auprc_score(y: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (AUPRC) without sklearn."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order].astype(np.float64)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / float(n_pos)
    # AP = sum (R_n - R_{n-1}) * P_n
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def braking_lead_tau(
    p_brake_traj: np.ndarray,
    tight_onset: int,
    *,
    gamma: float,
    consec: int,
    window: int,
) -> float:
    """Earliest tau<=0 where p_brake > gamma for `consec` consecutive steps."""
    e = int(tight_onset)
    if e < 0:
        return float("nan")
    t_max = int(p_brake_traj.shape[0])
    for tau in range(-window, 1):
        ok = True
        for k in range(consec):
            t = e + tau + k
            if t < 0 or t >= t_max or t > e:
                ok = False
                break
            v = float(p_brake_traj[t])
            if not (np.isfinite(v) and v > gamma):
                ok = False
                break
        if ok:
            return float(tau)
    return float("nan")


def geom_matched_paired_delta(
    z_rec: np.ndarray,
    z_rea: np.ndarray,
    ttc_rec: np.ndarray,
    ttc_rea: np.ndarray,
    speed_rec: np.ndarray,
    speed_rea: np.ndarray,
    mask: np.ndarray,
    *,
    n_ttc_bins: int = 4,
    n_speed_bins: int = 4,
) -> dict[str, float]:
    """Within-bin Rec−Rea Δ using quantile bins on pooled (TTC, speed)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {"n_maps": 0, "n_bins_used": 0, "delta": float("nan")}

    # Pooled edges from both methods on masked maps
    ttc_all = np.concatenate([ttc_rec[idx], ttc_rea[idx]])
    spd_all = np.concatenate([speed_rec[idx], speed_rea[idx]])
    ttc_ok = ttc_all[np.isfinite(ttc_all)]
    spd_ok = spd_all[np.isfinite(spd_all)]
    if ttc_ok.size < n_ttc_bins or spd_ok.size < n_speed_bins:
        return {"n_maps": int(idx.size), "n_bins_used": 0, "delta": float("nan")}

    ttc_edges = np.unique(np.quantile(ttc_ok, np.linspace(0, 1, n_ttc_bins + 1)))
    spd_edges = np.unique(np.quantile(spd_ok, np.linspace(0, 1, n_speed_bins + 1)))
    if ttc_edges.size < 2 or spd_edges.size < 2:
        return {"n_maps": int(idx.size), "n_bins_used": 0, "delta": float("nan")}

    def _bin(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return np.digitize(x, edges[1:-1], right=False)

    bin_deltas: list[float] = []
    bin_weights: list[float] = []
    for bt in range(ttc_edges.size - 1):
        for bs in range(spd_edges.size - 1):
            rec_sel = []
            rea_sel = []
            for i in idx:
                if not (
                    np.isfinite(z_rec[i])
                    and np.isfinite(z_rea[i])
                    and np.isfinite(ttc_rec[i])
                    and np.isfinite(ttc_rea[i])
                    and np.isfinite(speed_rec[i])
                    and np.isfinite(speed_rea[i])
                ):
                    continue
                if _bin(np.array([ttc_rec[i]]), ttc_edges)[0] == bt and _bin(
                    np.array([speed_rec[i]]), spd_edges
                )[0] == bs:
                    rec_sel.append(float(z_rec[i]))
                if _bin(np.array([ttc_rea[i]]), ttc_edges)[0] == bt and _bin(
                    np.array([speed_rea[i]]), spd_edges
                )[0] == bs:
                    rea_sel.append(float(z_rea[i]))
            if not rec_sel or not rea_sel:
                continue
            w = min(len(rec_sel), len(rea_sel))
            bin_deltas.append(float(np.mean(rec_sel) - np.mean(rea_sel)))
            bin_weights.append(float(w))

    if not bin_deltas:
        return {"n_maps": int(idx.size), "n_bins_used": 0, "delta": float("nan")}
    w = np.asarray(bin_weights, dtype=np.float64)
    d = np.asarray(bin_deltas, dtype=np.float64)
    return {
        "n_maps": int(idx.size),
        "n_bins_used": int(len(bin_deltas)),
        "delta": float(np.average(d, weights=w)),
    }


def kinematic_adjusted_method_effect(
    z_rec: np.ndarray,
    z_rea: np.ndarray,
    v_rec: np.ndarray,
    v_rea: np.ndarray,
    ttc_rec: np.ndarray,
    ttc_rea: np.ndarray,
    d_rec: np.ndarray,
    d_rea: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Map-FE pooled regression: z ~ I[Rec] + speed + TTC + dist.

    Within-map demeaning implements u_i. Returns beta_1 (Rec method effect)
    after kinematic adjustment, plus unadjusted paired mean for reference.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {
            "n_maps": 0,
            "beta_rec": float("nan"),
            "unadjusted_paired_delta": float("nan"),
        }

    rows_z: list[float] = []
    rows_rec: list[float] = []
    rows_v: list[float] = []
    rows_ttc: list[float] = []
    rows_d: list[float] = []
    map_ids: list[int] = []
    for j, i in enumerate(idx):
        for is_rec, z, v, ttc, d in (
            (1.0, z_rec[i], v_rec[i], ttc_rec[i], d_rec[i]),
            (0.0, z_rea[i], v_rea[i], ttc_rea[i], d_rea[i]),
        ):
            vals = (z, v, ttc, d)
            if not all(np.isfinite(vals)):
                continue
            rows_z.append(float(z))
            rows_rec.append(is_rec)
            rows_v.append(float(v))
            rows_ttc.append(float(ttc))
            rows_d.append(float(d))
            map_ids.append(j)

    if len(rows_z) < 4:
        return {
            "n_maps": int(idx.size),
            "beta_rec": float("nan"),
            "unadjusted_paired_delta": float("nan"),
        }

    z = np.asarray(rows_z, dtype=np.float64)
    rec = np.asarray(rows_rec, dtype=np.float64)
    v = np.asarray(rows_v, dtype=np.float64)
    ttc = np.asarray(rows_ttc, dtype=np.float64)
    d = np.asarray(rows_d, dtype=np.float64)
    mid = np.asarray(map_ids, dtype=np.int64)

    def _demean(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for m in np.unique(mid):
            sel = mid == m
            out[sel] -= out[sel].mean()
        return out

    y = _demean(z)
    X = np.column_stack([_demean(rec), _demean(v), _demean(ttc), _demean(d)])
    # Drop maps/rows that became all-zero covariates after demean (single obs).
    keep = np.any(np.abs(X) > 1e-12, axis=1) & np.isfinite(y)
    y = y[keep]
    X = X[keep]
    if y.size < 4 or X.shape[0] < X.shape[1]:
        return {
            "n_maps": int(idx.size),
            "beta_rec": float("nan"),
            "unadjusted_paired_delta": float("nan"),
        }
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        beta_rec = float(beta[0])
    except np.linalg.LinAlgError:
        beta_rec = float("nan")

    # Unadjusted paired delta on maps where both z are finite.
    both = np.isfinite(z_rec[idx]) & np.isfinite(z_rea[idx])
    unadj = (
        float(np.mean(z_rec[idx][both] - z_rea[idx][both])) if both.any() else float("nan")
    )
    return {
        "n_maps": int(idx.size),
        "n_rows": int(y.size),
        "beta_rec": beta_rec,
        "unadjusted_paired_delta": unadj,
    }
