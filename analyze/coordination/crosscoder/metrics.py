"""Crosscoder evaluation metrics (divergences, subspace, ISTA codes)."""

from __future__ import annotations

import numpy as np
import torch

from crosscoder.model import Crosscoder, ista_sparse_code, reconstruct_from_codes

EPS = 1e-8


def kl_softmax(log_p: np.ndarray, log_q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.exp(log_p - log_p.max(axis=-1, keepdims=True))
    q = np.exp(log_q - log_q.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True).clip(min=eps)
    q = q / q.sum(axis=-1, keepdims=True).clip(min=eps)
    return np.sum(p * (np.log(p.clip(min=eps)) - np.log(q.clip(min=eps))), axis=-1)


def category_masks(tight: np.ndarray, approach: np.ndarray) -> dict[str, np.ndarray]:
    t = tight.astype(bool)
    a = approach.astype(bool)
    return {
        "nominal": ~t,
        "nominal_strict": ~t & ~a,
        "approach_only": a & ~t,
        "tight": t,
        "tight_and_approach": t & a,
    }


def l2_rows(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=-1)


def hidden_divergences(h_r: np.ndarray, h_a: np.ndarray) -> dict[str, np.ndarray]:
    d = h_r - h_a
    l2 = l2_rows(d)
    nr = l2_rows(h_r)
    na = l2_rows(h_a)
    norm = l2 / (0.5 * (nr + na) + EPS)
    rr = np.sum(h_r * h_a, axis=-1) / (nr * na + EPS)
    cosine_dist = 1.0 - np.clip(rr, -1.0, 1.0)
    return {
        "d_h": l2.astype(np.float32),
        "d_h_norm": norm.astype(np.float32),
        "cosine_dist": cosine_dist.astype(np.float32),
        "norm_record": nr.astype(np.float32),
        "norm_reactive": na.astype(np.float32),
    }


def code_divergences(c_r: np.ndarray, c_a: np.ndarray) -> dict[str, np.ndarray]:
    k = max(c_r.shape[-1], 1)
    abs_mean = np.abs(c_r - c_a).mean(axis=-1)
    l1 = np.abs(c_r - c_a).sum(axis=-1)
    l2 = l2_rows(c_r - c_a)
    n1r = np.abs(c_r).sum(axis=-1)
    n1a = np.abs(c_a).sum(axis=-1)
    n2r = l2_rows(c_r)
    n2a = l2_rows(c_a)
    return {
        "d_c": abs_mean.astype(np.float32),
        "d_c_l1_norm": (l1 / (0.5 * (n1r + n1a) + EPS)).astype(np.float32),
        "d_c_l2_norm": (l2 / (0.5 * (n2r + n2a) + EPS)).astype(np.float32),
        "l0_record": (c_r > 0).sum(axis=-1).astype(np.float32),
        "l0_reactive": (c_a > 0).sum(axis=-1).astype(np.float32),
        "l1_record": n1r.astype(np.float32),
        "l1_reactive": n1a.astype(np.float32),
        "dict_size": np.array(k),
    }


def summarize_by_category(values: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    out: dict = {}
    for name, m in masks.items():
        v = values[m]
        out[name] = {
            "mean": float(v.mean()) if v.size else float("nan"),
            "median": float(np.median(v)) if v.size else float("nan"),
            "n": int(v.size),
        }
    t_mean = out.get("tight", {}).get("mean", float("nan"))
    n_mean = out.get("nominal", {}).get("mean", float("nan"))
    out["tight_over_nominal"] = (
        float(t_mean / (n_mean + EPS)) if np.isfinite(t_mean) and np.isfinite(n_mean) else float("nan")
    )
    return out


def scene_level_delta(
    scene: np.ndarray,
    values: np.ndarray,
    tight: np.ndarray,
) -> dict:
    """Per-scene (mean_tight − mean_nominal); only scenes with both categories."""
    scene = np.asarray(scene, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    tight = np.asarray(tight, dtype=bool)
    deltas = []
    for s in np.unique(scene):
        m = scene == s
        vt, vn = values[m & tight], values[m & ~tight]
        if vt.size == 0 or vn.size == 0:
            continue
        deltas.append(float(vt.mean() - vn.mean()))
    d = np.asarray(deltas, dtype=np.float64)
    return {
        "num_eligible_scenes": int(d.size),
        "mean_delta": float(d.mean()) if d.size else float("nan"),
        "fraction_positive": float((d > 0).mean()) if d.size else float("nan"),
        "deltas": d,
    }


def bootstrap_ci(x: np.ndarray, *, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = x.size
    for i in range(n_boot):
        means[i] = x[rng.integers(0, n, n)].mean()
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return [float(lo), float(hi)]


def orthonormal_basis(rows: np.ndarray) -> np.ndarray:
    """rows: (K, D) → Q (D, K) with orthonormal columns, or fewer if rank-deficient."""
    if rows.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    q, _r = np.linalg.qr(rows.T)
    rank = min(rows.shape[0], rows.shape[1])
    return q[:, :rank]


def subspace_overlap(rows_a: np.ndarray, rows_b: np.ndarray) -> dict[str, float]:
    qa = orthonormal_basis(rows_a)
    qb = orthonormal_basis(rows_b)
    k = min(qa.shape[1], qb.shape[1], max(rows_a.shape[0], 1))
    if k == 0:
        return {"overlap": float("nan"), "mean_principal_cosine": float("nan"), "k": 0}
    s = np.linalg.svd(qa[:, :k].T @ qb[:, :k], compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return {
        "overlap": float(np.sum(s**2) / k),
        "mean_principal_cosine": float(s.mean()) if s.size else float("nan"),
        "mean_principal_angle_deg": float(angles.mean()) if angles.size else float("nan"),
        "k": int(k),
    }


def interaction_sensitivity(abs_delta: np.ndarray, tight: np.ndarray) -> np.ndarray:
    t = tight.astype(bool)
    mean_t = abs_delta[t].mean(axis=0) if np.any(t) else np.zeros(abs_delta.shape[1])
    mean_n = abs_delta[~t].mean(axis=0) if np.any(~t) else np.zeros(abs_delta.shape[1])
    return mean_t - mean_n


def tight_onset(tight: np.ndarray) -> np.ndarray:
    n, _tlen = tight.shape
    out = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        hits = np.flatnonzero(tight[i])
        if hits.size:
            out[i] = int(hits[0])
    return out


@torch.no_grad()
def infer_codes_and_recon(
    model: Crosscoder,
    h_r: np.ndarray,
    h_a: np.ndarray,
    *,
    l1_coeff: float,
    device: str,
    batch: int = 4096,
    n_iters: int = 80,
) -> dict[str, np.ndarray | float]:
    model.eval()
    d_r = model.decoder_columns("record").to(device)
    d_a = model.decoder_columns("reactive").to(device)
    b_r = model.decoder_bias("record").to(device)
    b_a = model.decoder_bias("reactive").to(device)
    codes_r, codes_a = [], []
    recon_r, recon_a = [], []
    shared_z, shared_rr, shared_ra = [], [], []
    for i in range(0, h_r.shape[0], batch):
        hr = torch.from_numpy(h_r[i : i + batch]).to(device)
        ha = torch.from_numpy(h_a[i : i + batch]).to(device)
        cr = ista_sparse_code(hr, d_r, b_r, l1_coeff=l1_coeff, n_iters=n_iters)
        ca = ista_sparse_code(ha, d_a, b_a, l1_coeff=l1_coeff, n_iters=n_iters)
        rr = reconstruct_from_codes(cr, d_r, b_r)
        ra = reconstruct_from_codes(ca, d_a, b_a)
        z_r, z_a, sr, sa = model.forward(hr, ha)
        codes_r.append(cr.cpu().numpy())
        codes_a.append(ca.cpu().numpy())
        recon_r.append(((hr - rr).pow(2).mean(dim=-1)).cpu().numpy())
        recon_a.append(((ha - ra).pow(2).mean(dim=-1)).cpu().numpy())
        shared_z.append(z_r.cpu().numpy())
        shared_rr.append(((hr - sr).pow(2).mean(dim=-1)).cpu().numpy())
        shared_ra.append(((ha - sa).pow(2).mean(dim=-1)).cpu().numpy())
    c_r = np.concatenate(codes_r, axis=0)
    c_a = np.concatenate(codes_a, axis=0)
    e_r = np.concatenate(recon_r, axis=0)
    e_a = np.concatenate(recon_a, axis=0)
    z = np.concatenate(shared_z, axis=0)
    se_r = np.concatenate(shared_rr, axis=0)
    se_a = np.concatenate(shared_ra, axis=0)
    support_r = c_r > 0
    support_a = c_a > 0
    inter = (support_r & support_a).sum(axis=-1)
    union = (support_r | support_a).sum(axis=-1).clip(min=1)
    return {
        "c_record": c_r,
        "c_reactive": c_a,
        "z_encoder": z,
        "sep_recon_record": e_r,
        "sep_recon_reactive": e_a,
        "shared_recon_record": se_r,
        "shared_recon_reactive": se_a,
        "mean_sep_recon_record": float(e_r.mean()),
        "mean_sep_recon_reactive": float(e_a.mean()),
        "mean_shared_recon_record": float(se_r.mean()),
        "mean_shared_recon_reactive": float(se_a.mean()),
        "mean_sep_l0_record": float((c_r > 0).sum(axis=-1).mean()),
        "mean_sep_l0_reactive": float((c_a > 0).sum(axis=-1).mean()),
        "mean_shared_l0": float((z > 0).sum(axis=-1).mean()),
        "dead_frac_record": float((c_r.max(axis=0) <= 0).mean()),
        "dead_frac_reactive": float((c_a.max(axis=0) <= 0).mean()),
        "mean_support_jaccard": float((inter / union).mean()),
        "mean_l1_record": float(np.abs(c_r).mean()),
        "mean_l1_reactive": float(np.abs(c_a).mean()),
        "mean_code_norm_record": float(l2_rows(c_r).mean()),
        "mean_code_norm_reactive": float(l2_rows(c_a).mean()),
    }
