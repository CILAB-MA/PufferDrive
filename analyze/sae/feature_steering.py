"""Feature steering: causal sensitivity of policy outputs to SAE features.

Hook (default)::

    z = partner_encoder(slot)     # SAE input
    f' = f + α e_j
    z' = decode(f')
    encoded = pool/shared(z')     # remaining Drive.encode
    h = LSTMCell(encoded, h0=0)   # remaining recurrent
    π, V = decode_actions(h)

Gradient::

    ∂ P(brake)/∂f_j ,  ∂ E[accel]/∂f_j ,  ∂ log P(brake)/∂f_j ,  ∂ V/∂f_j ,  ∂ H/∂f_j

Finite difference (α ∈ {-2σ,-σ,+σ,+2σ} by default)::

    S_j(α) = metric(z + α d_j) − metric(z)

Requires ``obs`` in activations.npz (``SAVE_OBS=1``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SAE_DIR = Path(__file__).resolve().parent
if str(_SAE_DIR) not in sys.path:
    sys.path.insert(0, str(_SAE_DIR))

from feature_matching import load_alive_mask, load_feature_acts, resolve_model_name  # noqa: E402
from stats_utils import load_enrichment_scores  # noqa: E402
from partner_features import (  # noqa: E402
    action_stats_from_logits,
    encode_with_slot_override,
    tracked_slot_is_maxpool_winner,
)
from sae_model import SparseAutoencoder  # noqa: E402

MODELS = ("record", "reactive", "selfplay")
PRETTY = {"record": "ReCord", "reactive": "Reactive", "selfplay": "Self-play"}
ACTIVATIONS_FILENAME = "activations.npz"
DEFAULT_ALPHAS = (-5.0, 5.0, 10.0)


def activation_key(exp: str) -> str:
    return f"activation__{exp}"


def intervene_decode(
    sae: SparseAutoencoder,
    x: torch.Tensor,
    *,
    feature_id: int,
    mode: str,
    alpha: float | torch.Tensor,
) -> torch.Tensor:
    """SAE encode → intervene on feature → decode. Gradients flow through decode + α."""
    with torch.no_grad():
        feats = sae.encode(x)
    feats = feats.detach()
    if mode == "boost":
        # Avoid in-place leaf writes so α stays in the autograd graph.
        eye = torch.zeros_like(feats)
        eye[:, feature_id] = 1.0
        feats_int = feats + eye * alpha
    elif mode == "ablate":
        feats_int = feats.clone()
        feats_int[:, feature_id] = 0.0
    elif mode == "set":
        eye = torch.zeros_like(feats)
        eye[:, feature_id] = 1.0
        feats_int = feats * (1.0 - eye) + eye * alpha
    else:
        raise ValueError(mode)
    return sae.decode(feats_int)


def _sample_indices(
    n_total: int, max_rows: int, seed: int, mask: np.ndarray | None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mask is not None:
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            cand = np.arange(n_total)
    else:
        cand = np.arange(n_total)
    n = min(max_rows, int(cand.size))
    return rng.choice(cand, size=n, replace=False)


def conflict_mask_from_activations(sae_root: Path, probe_step: int, data_mode: str) -> np.ndarray | None:
    """Approximate conflict rows from ego/other_state in activations.npz."""
    from scene_metrics import binary_labels, compute_row_metrics

    path = (
        sae_root
        / "human_replay"
        / data_mode
        / f"step_{probe_step:06d}"
        / ACTIVATIONS_FILENAME
    )
    if not path.is_file():
        return None
    with np.load(path) as data:
        if "ego_state" not in data.files or "other_state" not in data.files:
            return None
        metrics = compute_row_metrics(
            ego_state=data["ego_state"],
            other_state=data["other_state"],
            future_traj=data["future_traj"] if "future_traj" in data.files else None,
            dist_at_t=data["dist_at_t"] if "dist_at_t" in data.files else None,
        )
    return binary_labels(metrics)["conflict"]


STAT_KEYS = (
    "p_brake",
    "p_throttle",
    "p_neg_accel",
    "p_strong_brake",
    "accel",
    "steer",
    "steer_mag",
    "brake_proxy",
    "log_p_brake",
    "entropy",
)
GRAD_MAP = (
    ("p_brake", "d_p_brake"),
    ("p_throttle", "d_p_throttle"),
    ("p_neg_accel", "d_p_neg_accel"),
    ("p_strong_brake", "d_p_strong_brake"),
    ("accel", "d_accel"),
    ("steer", "d_steer"),
    ("steer_mag", "d_steer_mag"),
    ("brake_proxy", "d_brake_proxy"),
    ("log_p_brake", "d_log_p_brake"),
    ("entropy", "d_entropy"),
)
# Metrics we attribute onto SAE decoder directions (paper-facing).
ATTR_METRICS = (
    ("p_brake", "attr_p_brake"),
    ("p_neg_accel", "attr_p_neg_accel"),
    ("p_strong_brake", "attr_p_strong_brake"),
    ("accel", "attr_accel"),
    ("log_p_brake", "attr_log_p_brake"),
    ("entropy", "attr_entropy"),
    ("steer_mag", "attr_steer_mag"),
)


def decoder_directions(sae: SparseAutoencoder) -> torch.Tensor:
    """Feature dictionary: ``W_dec[j]`` is direction ``d_j`` in partner-embedding space.

    Shape ``(d_sae, d_in)``. Rows are typically unit-norm after training.
    """
    return sae.W_dec.detach()


def feature_activation_scales(
    sae: SparseAutoencoder,
    x: np.ndarray,
    feature_ids: list[int],
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> dict[str, np.ndarray]:
    """Per-feature σ and IQR of SAE activations (active-only when enough fires)."""
    sae.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
            chunks.append(sae.encode(xb).cpu().numpy())
    feats = np.concatenate(chunks, axis=0)
    sigmas = np.zeros(len(feature_ids), dtype=np.float64)
    iqrs = np.zeros(len(feature_ids), dtype=np.float64)
    for i, fid in enumerate(feature_ids):
        col = feats[:, fid].astype(np.float64)
        active = col[col > 0]
        src = active if active.size >= 32 else col
        sigmas[i] = max(float(np.std(src)), 1e-3)
        q75, q25 = np.quantile(src, [0.75, 0.25])
        iqrs[i] = max(float(q75 - q25), 1e-3)
    return {"sigma": sigmas, "iqr": iqrs}


def _combine_row_masks(
    *masks: np.ndarray | None,
    n_total: int,
) -> np.ndarray | None:
    out = np.ones(n_total, dtype=bool)
    for m in masks:
        if m is None:
            continue
        out &= np.asarray(m, dtype=bool)
    return out if out.any() else out


def _attribution_pool_mode(pool_mode: str) -> str:
    """Natural ∂π/∂z uses real max-pool (or slot_only); not delta residual.

    ``delta`` needs ``baseline_slot_embedding`` and is only for finite steering /
    SAE-intervention paths. Passing ``delta`` here would raise or silently break.
    """
    if pool_mode == "slot_only":
        return "slot_only"
    return "max"


def projection_attribution_obs_level(
    *,
    policy,
    sae: SparseAutoencoder,
    obs: np.ndarray,
    partner_slot: np.ndarray,
    x: np.ndarray,
    feature_ids: list[int],
    device: torch.device,
    batch_size: int = 256,
    max_rows: int = 2048,
    pool_mode: str = "max",
    row_mask: np.ndarray | None = None,
    winning_slot_only: bool = False,
    skip_lstm: bool = False,
    sample_seed: int = 2,
) -> dict:
    """Per-observation attribution with raw / cos / scaled variants.

    Returns arrays shaped ``(N, K)`` for each metric variant plus row indices.
    """
    n_total = obs.shape[0]
    winner_mask = None
    if winning_slot_only:
        winner_chunks = []
        for start in range(0, n_total, batch_size):
            stop = min(start + batch_size, n_total)
            ob = torch.from_numpy(obs[start:stop].astype(np.float32)).to(device)
            sl = torch.from_numpy(partner_slot[start:stop].astype(np.int64)).to(device)
            winner_chunks.append(tracked_slot_is_maxpool_winner(policy, ob, sl).cpu().numpy())
        winner_mask = np.concatenate(winner_chunks)

    combined = _combine_row_masks(row_mask, winner_mask, n_total=n_total)
    idx = _sample_indices(n_total, max_rows, sample_seed, combined)
    n = int(idx.size)
    obs_t = torch.from_numpy(obs[idx].astype(np.float32)).to(device)
    slot_t = torch.from_numpy(partner_slot[idx].astype(np.int64)).to(device)
    x_np = x[idx].astype(np.float32)

    scales = feature_activation_scales(
        sae, x_np, feature_ids, device=device, batch_size=batch_size
    )
    sigma_k = scales["sigma"]

    dirs = decoder_directions(sae).to(device=device, dtype=torch.float32)
    d_sel = dirs[feature_ids]
    d_norm = d_sel.norm(dim=-1).clamp_min(1e-8)

    policy.train(False)
    sae.train(False)

    k = len(feature_ids)
    metric_keys = [name for _, name in ATTR_METRICS] + ["attr_value"]
    chunks: dict[str, list[np.ndarray]] = {m: [] for m in metric_keys}
    chunks_raw: dict[str, list[np.ndarray]] = {m: [] for m in metric_keys}
    chunks_cos: dict[str, list[np.ndarray]] = {m: [] for m in metric_keys}
    chunks_scaled: dict[str, list[np.ndarray]] = {m: [] for m in metric_keys}
    grad_norms: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        ob = obs_t[start:stop]
        sl = slot_t[start:stop]
        xb = torch.from_numpy(x_np[start:stop]).to(device)
        z = xb.detach().clone().requires_grad_(True)
        attr_pool = _attribution_pool_mode(pool_mode)
        _, actions, value = encode_with_slot_override(
            policy,
            ob,
            sl,
            z,
            pool_mode=attr_pool,
            skip_lstm=skip_lstm,
        )
        stats = action_stats_from_logits(actions)
        value_s = value.float().reshape(-1)
        bsz = int(z.shape[0])

        grads_z: dict[str, torch.Tensor] = {}
        for key, _ in ATTR_METRICS:
            g = torch.autograd.grad(
                stats[key].mean(), z, retain_graph=True, allow_unused=True
            )[0]
            grads_z[key] = g if g is not None else torch.zeros_like(z)
        gv = torch.autograd.grad(value_s.mean(), z, allow_unused=True)[0]
        if gv is None:
            gv = torch.zeros_like(z)
        grads_z["value"] = gv

        g_norm = grads_z["p_brake"].detach().norm(dim=-1).cpu().numpy()
        grad_norms.append(g_norm)

        for key, out_name in ATTR_METRICS:
            g = grads_z[key]
            raw = (g @ d_sel.T).detach().cpu().numpy().astype(np.float64)
            cos = (raw / (g.norm(dim=-1, keepdim=True).clamp_min(1e-8).cpu().numpy() * d_norm.cpu().numpy())).astype(np.float64)
            scaled = raw * sigma_k.reshape(1, -1)
            chunks_raw[out_name].append(raw)
            chunks_cos[out_name].append(cos)
            chunks_scaled[out_name].append(scaled)
            chunks[out_name].append(raw)

        raw_v = (gv @ d_sel.T).detach().cpu().numpy().astype(np.float64)
        cos_v = raw_v / (
            gv.norm(dim=-1, keepdim=True).clamp_min(1e-8).cpu().numpy() * d_norm.cpu().numpy()
        )
        chunks_raw["attr_value"].append(raw_v)
        chunks_cos["attr_value"].append(cos_v.astype(np.float64))
        chunks_scaled["attr_value"].append(raw_v * sigma_k.reshape(1, -1))
        chunks["attr_value"].append(raw_v)

    arrays = {
        "raw": {m: np.concatenate(v, axis=0) for m, v in chunks_raw.items()},
        "cos": {m: np.concatenate(v, axis=0) for m, v in chunks_cos.items()},
        "scaled": {m: np.concatenate(v, axis=0) for m, v in chunks_scaled.items()},
    }
    return {
        "feature_ids": [int(f) for f in feature_ids],
        "row_indices": idx.astype(np.int64),
        "n": n,
        "pool_mode": pool_mode,
        "forward_pool_mode": _attribution_pool_mode(pool_mode),
        "winning_slot_only": winning_slot_only,
        "feature_scales": {k: v.tolist() for k, v in scales.items()},
        "mean_grad_norm_p_brake": float(np.concatenate(grad_norms).mean()),
        "arrays": arrays,
    }


def projection_attribution(
    *,
    policy,
    sae: SparseAutoencoder,
    obs: np.ndarray,
    partner_slot: np.ndarray,
    x: np.ndarray,
    feature_ids: list[int] | None,
    device: torch.device,
    batch_size: int = 256,
    max_rows: int = 2048,
    pool_mode: str = "delta",
    row_mask: np.ndarray | None = None,
    skip_lstm: bool = False,
) -> dict:
    """Feature attribution without intervening::

        g_z = ∂ metric / ∂ z
        attr_j = E[ ⟨g_z , d_j⟩ ]     where d_j = W_dec[j]

    One backward for ∂π/∂z, then cheap dots onto all (or selected) SAE directions.
    This is the local, observational measure of *how strongly the policy uses
    feature j at the current observation* — preferred over finite steering for
    paper claims.

    Also returns ``value`` attributions and per-row mean |g_z|.
    """
    idx = _sample_indices(obs.shape[0], max_rows, 2, row_mask)
    n = int(idx.size)
    obs_t = torch.from_numpy(obs[idx].astype(np.float32)).to(device)
    slot_t = torch.from_numpy(partner_slot[idx].astype(np.int64)).to(device)
    x_np = x[idx].astype(np.float32)

    dirs = decoder_directions(sae).to(device=device, dtype=torch.float32)  # (F, d)
    if feature_ids is None:
        feature_ids = list(range(dirs.shape[0]))
    d_sel = dirs[feature_ids]  # (K, d)

    policy.train(False)
    sae.train(False)

    # Accumulators: sum of attr over batches, divide by n at end
    k = len(feature_ids)
    sum_attr = {name: np.zeros(k, dtype=np.float64) for _, name in ATTR_METRICS}
    sum_attr_value = np.zeros(k, dtype=np.float64)
    sum_grad_norm = 0.0
    n_done = 0

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        ob = obs_t[start:stop]
        sl = slot_t[start:stop]
        xb = torch.from_numpy(x_np[start:stop]).to(device)
        # Natural partner embedding as leaf; SAE directions live in this space.
        z = xb.detach().clone().requires_grad_(True)
        # Natural forward: put z in its slot and use real max-pool.
        # ``delta`` is for FD/steering only (needs baseline); ignore here.
        _, actions, value = encode_with_slot_override(
            policy,
            ob,
            sl,
            z,
            pool_mode=_attribution_pool_mode(pool_mode),
            skip_lstm=skip_lstm,
        )
        stats = action_stats_from_logits(actions)
        value_s = value.float().reshape(-1)
        bsz = int(z.shape[0])

        # One graph for all metrics via retain_graph
        grads_z: dict[str, torch.Tensor] = {}
        for key, _ in ATTR_METRICS:
            g = torch.autograd.grad(
                stats[key].mean(), z, retain_graph=True, allow_unused=True
            )[0]
            grads_z[key] = g if g is not None else torch.zeros_like(z)
        gv = torch.autograd.grad(value_s.mean(), z, allow_unused=True)[0]
        if gv is None:
            gv = torch.zeros_like(z)

        sum_grad_norm += float(grads_z["p_brake"].detach().norm(dim=-1).sum().cpu())
        # ⟨g, d_j⟩ per row → mean over batch, accumulate
        for key, name in ATTR_METRICS:
            # (B, d) @ (d, K) → (B, K) then sum over B
            dots = grads_z[key] @ d_sel.T  # (B, K)
            sum_attr[name] += dots.detach().double().sum(dim=0).cpu().numpy()
        sum_attr_value += (gv @ d_sel.T).detach().double().sum(dim=0).cpu().numpy()
        n_done += bsz

    assert n_done == n
    rows = []
    for i, fid in enumerate(feature_ids):
        row = {
            "feature_id": int(fid),
            "n": int(n),
            "method": "grad_z_dot_decoder",
            "hook": "per_other_pre_lstm" if not skip_lstm else "per_other_skip_lstm",
            "pool_mode": pool_mode,
            "attr_value": float(sum_attr_value[i] / n),
        }
        for _, name in ATTR_METRICS:
            row[name] = float(sum_attr[name][i] / n)
        rows.append(row)

    return {
        "features": rows,
        "mean_grad_norm_p_brake": float(sum_grad_norm / n),
        "n": int(n),
        "formula": "attr_j = E[ <∂metric/∂z , W_dec[j]> ]",
    }


def gradient_sensitivity(
    *,
    policy,
    sae: SparseAutoencoder,
    obs: np.ndarray,
    partner_slot: np.ndarray,
    x: np.ndarray,
    feature_ids: list[int],
    device: torch.device,
    batch_size: int = 256,
    max_rows: int = 2048,
    pool_mode: str = "delta",
    row_mask: np.ndarray | None = None,
    skip_lstm: bool = False,
) -> list[dict]:
    """Legacy path: ∂/∂α through SAE boost (equals projection if decode is linear).

    Prefer :func:`projection_attribution` for paper-facing numbers.
    """
    idx = _sample_indices(obs.shape[0], max_rows, 0, row_mask)
    n = int(idx.size)
    obs_t = torch.from_numpy(obs[idx].astype(np.float32)).to(device)
    slot_t = torch.from_numpy(partner_slot[idx].astype(np.int64)).to(device)
    x_t = torch.from_numpy(x[idx].astype(np.float32)).to(device)

    policy.train(False)
    sae.train(False)
    rows = []
    for fid in feature_ids:
        acc = {out_key: [] for _, out_key in GRAD_MAP}
        acc["d_value"] = []
        for start in range(0, n, batch_size):
            ob = obs_t[start : start + batch_size]
            sl = slot_t[start : start + batch_size]
            xb = x_t[start : start + batch_size]
            with torch.no_grad():
                x_base = sae(xb)
            alpha = torch.zeros((), device=device, requires_grad=True)
            x_hat = intervene_decode(sae, xb, feature_id=fid, mode="boost", alpha=alpha)
            _, actions, value = encode_with_slot_override(
                policy,
                ob,
                sl,
                x_hat,
                pool_mode=pool_mode,
                baseline_slot_embedding=x_base if pool_mode == "delta" else None,
                skip_lstm=skip_lstm,
            )
            stats = action_stats_from_logits(actions)
            value_s = value.float().reshape(-1)

            for key, out_key in GRAD_MAP:
                g = torch.autograd.grad(
                    stats[key].mean(), alpha, retain_graph=True, allow_unused=True
                )[0]
                acc[out_key].append(float(g.detach().cpu()) if g is not None else 0.0)

            gv = torch.autograd.grad(value_s.mean(), alpha, allow_unused=True)[0]
            acc["d_value"].append(float(gv.detach().cpu()) if gv is not None else 0.0)

        rows.append(
            {
                "feature_id": int(fid),
                "n": int(n),
                "pool_mode": pool_mode,
                "hook": "per_other_pre_lstm" if not skip_lstm else "per_other_skip_lstm",
                "through_lstm": (not skip_lstm),
                "method": "d_alpha_through_decode",
                **{k: float(np.mean(v)) for k, v in acc.items()},
            }
        )
    return rows


@torch.no_grad()
def feature_activation_sigma(
    sae: SparseAutoencoder,
    x: torch.Tensor,
    feature_id: int,
    *,
    batch_size: int = 2048,
) -> float:
    """Std of SAE pre-intervention activations for feature ``feature_id``."""
    chunks = []
    for start in range(0, x.shape[0], batch_size):
        feats = sae.encode(x[start : start + batch_size])
        chunks.append(feats[:, feature_id].float().cpu().numpy())
    arr = np.concatenate(chunks)
    active = arr[arr > 0]
    src = active if active.size >= 32 else arr
    sig = float(np.std(src))
    return max(sig, 1e-3)


def sigma_alphas(
    sigma: float, multipliers: tuple[float, ...] = (-2.0, -1.0, 1.0, 2.0)
) -> tuple[float, ...]:
    return tuple(float(m * sigma) for m in multipliers)


@torch.no_grad()
def finite_difference_sweep(
    *,
    policy,
    sae: SparseAutoencoder,
    obs: np.ndarray,
    partner_slot: np.ndarray,
    x: np.ndarray,
    feature_ids: list[int],
    device: torch.device,
    alpha_scale: str = "sigma",
    fixed_alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    sigma_mults: tuple[float, ...] = (-2.0, -1.0, 1.0, 2.0),
    batch_size: int = 256,
    max_rows: int = 2048,
    pool_mode: str = "delta",
    row_mask: np.ndarray | None = None,
    skip_lstm: bool = False,
) -> list[dict]:
    """Finite intervention S_j(α)=π(z+α d_j)−π(z); α in σ-units by default."""
    idx = _sample_indices(obs.shape[0], max_rows, 1, row_mask)
    n = int(idx.size)
    obs_t = torch.from_numpy(obs[idx].astype(np.float32)).to(device)
    slot_t = torch.from_numpy(partner_slot[idx].astype(np.int64)).to(device)
    x_t = torch.from_numpy(x[idx].astype(np.float32)).to(device)

    policy.eval()
    sae.eval()
    x_recon = sae(x_t)

    def run_slot(x_slot: torch.Tensor) -> dict[str, np.ndarray]:
        chunks: dict[str, list] = {k: [] for k in STAT_KEYS}
        chunks["value"] = []
        for start in range(0, n, batch_size):
            ob = obs_t[start : start + batch_size]
            sl = slot_t[start : start + batch_size]
            xs = x_slot[start : start + batch_size]
            xb = x_recon[start : start + batch_size]
            _, actions, value = encode_with_slot_override(
                policy,
                ob,
                sl,
                xs,
                pool_mode=pool_mode,
                baseline_slot_embedding=xb if pool_mode == "delta" else None,
                skip_lstm=skip_lstm,
            )
            stats = action_stats_from_logits(actions)
            for k in STAT_KEYS:
                chunks[k].append(stats[k].cpu().numpy())
            chunks["value"].append(value.float().reshape(-1).cpu().numpy())
        return {k: np.concatenate(v) for k, v in chunks.items()}

    rows = []
    for fid in feature_ids:
        sigma = feature_activation_sigma(sae, x_t, fid)
        if alpha_scale == "sigma":
            alphas = sigma_alphas(sigma, sigma_mults)
            alpha_meta = {"scale": "sigma", "sigma": sigma, "mults": list(sigma_mults)}
        else:
            alphas = fixed_alphas
            alpha_meta = {"scale": "fixed", "sigma": sigma}

        x_ablate = intervene_decode(sae, x_t, feature_id=fid, mode="ablate", alpha=0.0)
        base = run_slot(x_recon)
        ablate = run_slot(x_ablate)
        entry: dict = {
            "feature_id": int(fid),
            "n": int(n),
            "pool_mode": pool_mode,
            "hook": "per_other_pre_lstm" if not skip_lstm else "per_other_skip_lstm",
            "through_lstm": (not skip_lstm),
            "alpha_meta": alpha_meta,
            "base": {k: float(v.mean()) for k, v in base.items()},
            "ablate": {
                **{k: float(v.mean()) for k, v in ablate.items()},
                **{f"delta_{k}": float((ablate[k] - base[k]).mean()) for k in base},
            },
            "boosts": {},
        }
        for alpha in alphas:
            x_b = intervene_decode(sae, x_t, feature_id=fid, mode="boost", alpha=float(alpha))
            boosted = run_slot(x_b)
            key = f"{alpha:.4g}"
            if alpha_scale == "sigma":
                mult = alpha / sigma if sigma > 0 else 0.0
                key = f"{mult:+.0f}σ({alpha:.3g})"
            entry["boosts"][key] = {
                "alpha": float(alpha),
                **{k: float(v.mean()) for k, v in boosted.items()},
                **{f"delta_{k}": float((boosted[k] - base[k]).mean()) for k in base},
                "frac_p_brake_up": float(np.mean(boosted["p_brake"] > base["p_brake"])),
            }
        rows.append(entry)
    return rows


@torch.no_grad()
def embedding_sensitivity(
    sae: SparseAutoencoder,
    x: np.ndarray,
    feature_ids: list[int],
    *,
    device: torch.device,
    alpha: float = 5.0,
    batch_size: int = 2048,
    max_rows: int = 4096,
) -> list[dict]:
    rng = np.random.default_rng(0)
    n = min(max_rows, x.shape[0])
    idx = rng.choice(x.shape[0], size=n, replace=False)
    xb = torch.from_numpy(x[idx]).to(device)
    sae.eval()
    rows = []
    for fid in feature_ids:
        deltas, cosines = [], []
        for start in range(0, n, batch_size):
            batch = xb[start : start + batch_size]
            x0 = sae(batch)
            x1 = intervene_decode(sae, batch, feature_id=fid, mode="boost", alpha=alpha)
            deltas.append((x1 - x0).norm(dim=-1).cpu().numpy())
            cosines.append(
                torch.nn.functional.cosine_similarity(x0, x1, dim=-1).cpu().numpy()
            )
        rows.append(
            {
                "feature_id": int(fid),
                "alpha_boost": alpha,
                "mean_l2_delta": float(np.concatenate(deltas).mean()),
                "mean_cosine": float(np.concatenate(cosines).mean()),
            }
        )
    return rows


def pick_candidate_features(
    analysis_dir: Path,
    semantics_dir: Path,
    matching_dir: Path,
    *,
    top_k: int = 5,
    min_enrichment: float = 1.5,
) -> dict[str, list[int]]:
    """Blind primary set: matched triples + conflict enrichment (before attribution).

    Falls back to enrichment ranking only if no triples exist.
    """
    out: dict[str, list[int]] = {}
    model_dirs = {a: resolve_model_name(a) for a in MODELS}

    # Prefer matched triples + enrichment threshold (selection bias control)
    try:
        from projection_attribution_validation import pick_blind_conflict_features

        blind = pick_blind_conflict_features(
            analysis_dir,
            semantics_dir,
            matching_dir,
            min_enrichment=min_enrichment,
            top_k=top_k,
        )
        if any(blind.get(a) for a in MODELS):
            return blind
    except Exception:
        pass

    for a in MODELS:
        enrich = load_enrichment_scores(semantics_dir, model_dirs[a])
        ranked = sorted(enrich.items(), key=lambda kv: -kv[1])
        out[a] = [int(fid) for fid, _ in ranked[:top_k]]
    return out


def load_matched_triples(matching_dir: Path) -> list[dict]:
    path = matching_dir / "matched_triples.json"
    if not path.is_file():
        # try common alternate names
        for alt in ("triples.json", "mutual_nn_triples.json"):
            if (matching_dir / alt).is_file():
                path = matching_dir / alt
                break
        else:
            return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("triples") or data.get("matched_triples") or []
    return list(data)


def discover_sae_ckpts(run_dir: Path, ckpt_name: str) -> dict[str, Path]:
    found = {}
    for a in MODELS:
        exp = resolve_model_name(a)
        path = run_dir / exp / ckpt_name
        if path.is_file():
            found[a] = path
    return found


def load_obs_bundle(sae_root: Path, probe_step: int, data_mode: str, exp: str):
    path = (
        sae_root
        / "human_replay"
        / data_mode
        / f"step_{probe_step:06d}"
        / ACTIVATIONS_FILENAME
    )
    if not path.is_file():
        return None
    with np.load(path) as data:
        if "obs" not in data.files:
            return None
        return {
            "obs": np.asarray(data["obs"], dtype=np.float32),
            "partner_slot": np.asarray(data["partner_slot"], dtype=np.int64),
            "x": np.asarray(data[activation_key(exp)], dtype=np.float32),
            "path": str(path),
        }


def _print_grad_row(alias: str, r: dict) -> None:
    print(
        f"  grad f{r['feature_id']}: "
        f"∂P(brk)/∂f={r['d_p_brake']:+.5f}  "
        f"∂E[acc]/∂f={r['d_accel']:+.5f}  "
        f"∂logP(brk)/∂f={r.get('d_log_p_brake', 0):+.5f}  "
        f"∂H/∂f={r.get('d_entropy', 0):+.5f}  "
        f"∂V/∂f={r['d_value']:+.5f}"
        f"  [lstm={r.get('through_lstm')}]"
    )


def _print_fd_row(r: dict) -> None:
    fid = r["feature_id"]
    am = r.get("alpha_meta") or {}
    print(
        f"  fd f{fid}: base P(brk)={r['base']['p_brake']:.4f} "
        f"E[acc]={r['base']['accel']:.3f} H={r['base'].get('entropy', float('nan')):.3f} "
        f"V={r['base']['value']:.3f}  σ={am.get('sigma')}"
    )
    for a_str, b in r["boosts"].items():
        print(
            f"       {a_str:>12}: ΔP(brk)={b['delta_p_brake']:+.4f} "
            f"Δacc={b['delta_accel']:+.4f} Δ|steer|={b.get('delta_steer_mag', 0):+.4f} "
            f"ΔH={b.get('delta_entropy', 0):+.4f} ΔV={b['delta_value']:+.4f} "
            f"frac↑brk={b['frac_p_brake_up']:.3f}"
        )
    ab = r["ablate"]
    print(
        f"       {'ablate':>12}: ΔP(brk)={ab['delta_p_brake']:+.4f} "
        f"Δacc={ab['delta_accel']:+.4f} ΔV={ab['delta_value']:+.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="SAE feature steering (grad + FD)")
    p.add_argument("--analysis-dir", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--ckpt-name", type=str, default="sae_step_0001000.pt")
    p.add_argument("--sae-root", type=str, default="/data/puffer/sae")
    p.add_argument("--probe-step", type=int, default=1908)
    p.add_argument("--data-mode", type=str, default="validation")
    p.add_argument("--base-path", type=str, default="/data/puffer/experiments")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--alphas",
        type=str,
        default="5,10,-5",
        help="fixed alphas when --alpha-scale=fixed (use --alphas=5,10,-5)",
    )
    p.add_argument(
        "--alpha-scale",
        type=str,
        default="sigma",
        choices=["sigma", "fixed"],
        help="FD α in units of feature σ (default) or fixed absolute values",
    )
    p.add_argument(
        "--sigma-mults",
        type=str,
        default="-2,-1,1,2",
        help="multipliers of σ when --alpha-scale=sigma",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-rows", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--skip-policy", action="store_true")
    p.add_argument("--features", type=str, default=None, help="override: record=27,64;reactive=7")
    p.add_argument(
        "--pool-mode",
        type=str,
        default="delta",
        choices=["max", "delta", "slot_only"],
        help="How steered slot enters partner pool (delta=guaranteed residual effect)",
    )
    p.add_argument(
        "--conflict-only",
        action="store_true",
        help="Evaluate sensitivity only on conflict-labeled rows",
    )
    p.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Ablation: bypass LSTM (old incorrect path)",
    )
    args = p.parse_args()

    fixed_alphas = tuple(float(x) for x in args.alphas.split(",") if x.strip())
    sigma_mults = tuple(float(x) for x in args.sigma_mults.split(",") if x.strip())
    analysis_dir = Path(args.analysis_dir)
    run_dir = Path(args.run_dir)
    semantics_dir = analysis_dir / "semantics"
    matching_dir = analysis_dir / "matching"
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    row_mask = None
    if args.conflict_only:
        row_mask = conflict_mask_from_activations(
            Path(args.sae_root), args.probe_step, args.data_mode
        )
        n_c = int(row_mask.sum()) if row_mask is not None else 0
        print(f"conflict-only filter: {n_c} rows")
    print(
        f"hook=per_other→pool→shared→LSTM→head  "
        f"pool_mode={args.pool_mode} alpha_scale={args.alpha_scale} "
        f"skip_lstm={args.skip_lstm}"
    )

    ckpts = discover_sae_ckpts(run_dir, args.ckpt_name)
    if not ckpts:
        raise FileNotFoundError(f"No {args.ckpt_name} under {run_dir}")

    if args.features:
        candidates: dict[str, list[int]] = {a: [] for a in MODELS}
        for part in args.features.split(";"):
            if not part.strip() or "=" not in part:
                continue
            alias, ids = part.split("=", 1)
            alias = alias.strip()
            candidates[alias] = [int(x) for x in ids.split(",") if x.strip()]
    else:
        candidates = pick_candidate_features(
            analysis_dir, semantics_dir, matching_dir, top_k=args.top_k
        )
    print("Candidate features:", {a: candidates.get(a) for a in MODELS})

    results: dict = {
        "embedding": {},
        "attribution": {},  # primary: ⟨∂π/∂z, d_j⟩
        "gradient": {},  # legacy α-path (sanity check vs attribution)
        "finite_difference": {},  # secondary corroboration
        "alpha_scale": args.alpha_scale,
        "fixed_alphas": list(fixed_alphas),
        "sigma_mults": list(sigma_mults),
        "pool_mode": args.pool_mode,
        "conflict_only": bool(args.conflict_only),
        "through_lstm": (not args.skip_lstm),
        "hook": "per_other_embedding",
        "note": (
            "Primary: attr_j = E[<∂metric/∂z, W_dec[j]>] at natural partner "
            "embedding z (no intervention). Remaining net = pool→shared→LSTM→head. "
            "Hook stays at per-other (other-modeling layer), not LSTM hidden. "
            "Steering (FD) is secondary nonlinear check."
        ),
    }

    if args.skip_policy:
        for alias, ckpt in ckpts.items():
            exp = resolve_model_name(alias)
            sae = SparseAutoencoder.load(ckpt, map_location=device).to(device)
            bundle = load_obs_bundle(
                Path(args.sae_root), args.probe_step, args.data_mode, exp
            )
            act_path = (
                Path(args.sae_root)
                / "human_replay"
                / args.data_mode
                / f"step_{args.probe_step:06d}"
                / ACTIVATIONS_FILENAME
            )
            with np.load(act_path) as data:
                x_in = np.asarray(data[activation_key(exp)], dtype=np.float32)
            fids = candidates.get(alias) or []
            emb = embedding_sensitivity(sae, x_in, fids, device=device, alpha=5.0)
            results["embedding"][alias] = emb
        (out_dir / "steering_summary.json").write_text(json.dumps(results, indent=2))
        print(f"Wrote {out_dir / 'steering_summary.json'}")
        return

    # Shared obs check once
    probe_bundle = load_obs_bundle(
        Path(args.sae_root), args.probe_step, args.data_mode, resolve_model_name("record")
    )
    if probe_bundle is None:
        results["note"] = (
            "obs not stored in activations.npz; policy steering skipped. "
            "Re-collect with SAVE_OBS=1 / --save-obs then re-run."
        )
        print(results["note"])
        # still write embedding if possible
        for alias, ckpt in ckpts.items():
            exp = resolve_model_name(alias)
            sae = SparseAutoencoder.load(ckpt, map_location=device).to(device)
            act_path = (
                Path(args.sae_root)
                / "human_replay"
                / args.data_mode
                / f"step_{args.probe_step:06d}"
                / ACTIVATIONS_FILENAME
            )
            if act_path.is_file():
                with np.load(act_path) as data:
                    x_in = np.asarray(data[activation_key(exp)], dtype=np.float32)
                emb = embedding_sensitivity(
                    sae,
                    x_in,
                    candidates.get(alias) or [],
                    device=device,
                    alpha=5.0,
                )
                results["embedding"][alias] = emb
                for r in emb:
                    print(
                        f"  [{PRETTY[alias]}] emb f{r['feature_id']}: "
                        f"L2Δ={r['mean_l2_delta']:.4f}"
                    )
        (out_dir / "steering_summary.json").write_text(json.dumps(results, indent=2))
        print(f"Wrote {out_dir / 'steering_summary.json'}")
        return

    from sae_rollout import (  # noqa: WPS433
        build_human_replay_drive_args,
        create_policy,
        create_vecenv,
        load_policy_from_checkpoint,
        pick_checkpoint,
        resolve_run_dir,
        safe_close_vecenv,
    )

    for alias, ckpt in ckpts.items():
        exp = resolve_model_name(alias)
        print(f"\n=== {PRETTY[alias]} ({exp}) ===")
        sae = SparseAutoencoder.load(ckpt, map_location=device).to(device)
        bundle = load_obs_bundle(
            Path(args.sae_root), args.probe_step, args.data_mode, exp
        )
        assert bundle is not None
        fids = candidates.get(alias) or []
        if not fids:
            print("  no candidate features")
            continue

        emb = embedding_sensitivity(
            sae, bundle["x"], fids, device=device, alpha=5.0, max_rows=args.max_rows
        )
        results["embedding"][alias] = emb

        run = resolve_run_dir(args.base_path, exp)
        ckpt_pol = pick_checkpoint(run, device=str(device), probe_step=args.probe_step)
        drive_args = build_human_replay_drive_args(
            None, num_maps=1, device=str(device), data_mode=args.data_mode
        )
        vecenv = create_vecenv(drive_args, env_name="puffer_drive")
        policy = create_policy(drive_args, vecenv, env_name="puffer_drive")
        load_policy_from_checkpoint(policy, ckpt_pol.state_dict)
        policy.to(device)
        try:
            # Primary: ⟨∂π/∂z, d_j⟩ — no finite intervention
            attr = projection_attribution(
                policy=policy,
                sae=sae,
                obs=bundle["obs"],
                partner_slot=bundle["partner_slot"],
                x=bundle["x"],
                feature_ids=fids,
                device=device,
                batch_size=args.batch_size,
                max_rows=args.max_rows,
                pool_mode=args.pool_mode,
                row_mask=row_mask,
                skip_lstm=args.skip_lstm,
            )
            results["attribution"][alias] = attr
            print(f"  attribution ‖∂P(brk)/∂z‖ mean={attr['mean_grad_norm_p_brake']:.4f}")
            for r in attr["features"]:
                print(
                    f"  attr f{r['feature_id']}: "
                    f"⟨∂P(brk)/∂z,d⟩={r['attr_p_brake']:+.5f}  "
                    f"⟨∂E[acc]/∂z,d⟩={r['attr_accel']:+.5f}  "
                    f"⟨∂logP/∂z,d⟩={r['attr_log_p_brake']:+.5f}  "
                    f"⟨∂V/∂z,d⟩={r['attr_value']:+.5f}"
                )

            # Secondary: FD steering in σ units
            fd = finite_difference_sweep(
                policy=policy,
                sae=sae,
                obs=bundle["obs"],
                partner_slot=bundle["partner_slot"],
                x=bundle["x"],
                feature_ids=fids,
                device=device,
                alpha_scale=args.alpha_scale,
                fixed_alphas=fixed_alphas,
                sigma_mults=sigma_mults,
                batch_size=args.batch_size,
                max_rows=args.max_rows,
                pool_mode=args.pool_mode,
                row_mask=row_mask,
                skip_lstm=args.skip_lstm,
            )
            results["finite_difference"][alias] = fd
            for r in fd:
                _print_fd_row(r)
        finally:
            safe_close_vecenv(vecenv)

    # Cross-model attribution comparison by candidate rank
    if all(a in results["attribution"] for a in MODELS):
        cmp_rows = []
        for i in range(args.top_k):
            row: dict = {"rank": i}
            for a in MODELS:
                feats = results["attribution"][a]["features"]
                if i < len(feats):
                    row[f"{a}_feature"] = feats[i]["feature_id"]
                    row[f"{a}_attr_p_brake"] = feats[i]["attr_p_brake"]
                    row[f"{a}_attr_accel"] = feats[i]["attr_accel"]
                    row[f"{a}_attr_value"] = feats[i]["attr_value"]
            cmp_rows.append(row)
        results["attribution_comparison_by_rank"] = cmp_rows
        print("\n=== Attribution ⟨∂P(brake)/∂z, d_j⟩ (NonYieldSpec rank) ===")
        for row in cmp_rows:
            print(
                f"  rank{row['rank']}: "
                f"R={row.get('record_attr_p_brake')} "
                f"Rea={row.get('reactive_attr_p_brake')} "
                f"SP={row.get('selfplay_attr_p_brake')}"
            )
        mean = {
            a: float(
                np.mean(
                    [r[f"{a}_attr_p_brake"] for r in cmp_rows if f"{a}_attr_p_brake" in r]
                )
            )
            for a in MODELS
        }
        mean_acc = {
            a: float(
                np.mean(
                    [r[f"{a}_attr_accel"] for r in cmp_rows if f"{a}_attr_accel" in r]
                )
            )
            for a in MODELS
        }
        mean_v = {
            a: float(
                np.mean(
                    [r[f"{a}_attr_value"] for r in cmp_rows if f"{a}_attr_value" in r]
                )
            )
            for a in MODELS
        }
        results["mean_attr_p_brake_top_k"] = mean
        results["mean_attr_accel_top_k"] = mean_acc
        results["mean_attr_value_top_k"] = mean_v
        print(f"  mean ⟨∂P(brake)/∂z,d⟩: {mean}")
        print(f"  mean ⟨∂E[accel]/∂z,d⟩: {mean_acc}")
        print(f"  mean ⟨∂V/∂z,d⟩:         {mean_v}")
        order_ok = mean["record"] > mean["reactive"] > mean["selfplay"]
        results["claim_brake_attr_R_gt_Rea_gt_SP"] = bool(order_ok)
        print(f"  claim ReCord>Rea>SP on brake attr: {order_ok}")

    (out_dir / "steering_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote steering → {out_dir / 'steering_summary.json'}")


if __name__ == "__main__":
    main()
