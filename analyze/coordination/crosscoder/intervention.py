#!/usr/bin/env python3
"""Immediate same-state causal patch on Reactive hiddens.

  delta_U = U U^T (h_R - h_Rea)
  h'      = h_Rea + α delta_U

One shot: control bases → D_U thresholds → patch α-sweep → summary.

  python analyze/coordination/crosscoder/intervention.py --out-root ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_COORD = Path(__file__).resolve().parent.parent
if str(_COORD) not in sys.path:
    sys.path.insert(0, str(_COORD))

from common import ACCEL_VALUES, N_ACTIONS, N_STEER, STEER_VALUES  
from crosscoder.collect import load_named_pair  
from crosscoder.frozen_config import (  
    ALPHAS,
    FROZEN_K,
    HIDDEN_DIM,
    PRIMARY_ALPHA,
    RESULTS_MECHANISM,
)
from crosscoder.metrics import bootstrap_ci, orthonormal_basis  
from crosscoder.pipeline import _write_json  
from runtime import load_policy_from_ckpt  

# Re-exported for persistence / transient_causal / tests.
HIDDEN = HIDDEN_DIM
EPS = 1e-8
N_RANDOM = 3
DECODE_BS = 65536


def _inner(pol):
    return pol.policy if hasattr(pol, "policy") and hasattr(pol.policy, "decode_actions") else pol


def _as_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@torch.no_grad()
def decode_t(inner, h: torch.Tensor, bs: int = DECODE_BS) -> torch.Tensor:
    outs = []
    for i in range(0, h.shape[0], bs):
        a, _ = inner.decode_actions(h[i : i + bs])
        t = a[0] if isinstance(a, (tuple, list)) else a
        outs.append(t.float())
    return torch.cat(outs, 0)


def kl_t(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(log_p.float(), dim=-1)
    return (p * (torch.log_softmax(log_p.float(), dim=-1) - torch.log_softmax(log_q.float(), dim=-1))).sum(-1)


def js_t(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(log_p.float(), dim=-1)
    q = torch.softmax(log_q.float(), dim=-1)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p.clamp_min(EPS).log() - m.clamp_min(EPS).log())).sum(-1) + 0.5 * (
        q * (q.clamp_min(EPS).log() - m.clamp_min(EPS).log())
    ).sum(-1)


def expected_control_t(logits: torch.Tensor, accel_tbl: torch.Tensor, steer_tbl: torch.Tensor):
    p = torch.softmax(logits.float(), dim=-1)
    nact = logits.shape[-1]
    a_idx = torch.arange(nact, device=logits.device) // N_STEER
    s_idx = torch.arange(nact, device=logits.device) % N_STEER
    return (p * accel_tbl[a_idx]).sum(-1), (p * steer_tbl[s_idx]).sum(-1)


@torch.no_grad()
def patch_metrics_t(log_r, log_a, log_p, accel_tbl, steer_tbl) -> dict[str, np.ndarray]:
    d_base = kl_t(log_r, log_a)
    d_patch = kl_t(log_r, log_p)
    a_star = log_r.argmax(-1)
    rows = torch.arange(log_r.shape[0], device=log_r.device)
    p_a = torch.softmax(log_a.float(), dim=-1)
    p_p = torch.softmax(log_p.float(), dim=-1)
    acc_b, st_b = expected_control_t(log_a, accel_tbl, steer_tbl)
    acc_p, st_p = expected_control_t(log_p, accel_tbl, steer_tbl)
    acc_r, st_r = expected_control_t(log_r, accel_tbl, steer_tbl)
    return {
        "dkl": (d_base - d_patch).cpu().numpy().astype(np.float32),
        "d_base": d_base.cpu().numpy().astype(np.float32),
        "d_patch": d_patch.cpu().numpy().astype(np.float32),
        "js_base": js_t(log_r, log_a).cpu().numpy().astype(np.float32),
        "js_patch": js_t(log_r, log_p).cpu().numpy().astype(np.float32),
        "p_star_gain": (p_p[rows, a_star] - p_a[rows, a_star]).cpu().numpy().astype(np.float32),
        "argmax_agree_base": (log_a.argmax(-1) == a_star).float().cpu().numpy().astype(np.float32),
        "argmax_agree_patch": (log_p.argmax(-1) == a_star).float().cpu().numpy().astype(np.float32),
        "d_accel_to_r_base": (acc_b - acc_r).abs().cpu().numpy().astype(np.float32),
        "d_accel_to_r_patch": (acc_p - acc_r).abs().cpu().numpy().astype(np.float32),
        "d_steer_to_r_base": (st_b - st_r).abs().cpu().numpy().astype(np.float32),
        "d_steer_to_r_patch": (st_p - st_r).abs().cpu().numpy().astype(np.float32),
    }


def project(delta: np.ndarray, q: np.ndarray) -> np.ndarray:
    """q: (D, r) columns. Returns P_Q delta."""
    return (delta @ q) @ q.T


def match_norm(vec: np.ndarray, ref: np.ndarray) -> np.ndarray:
    n_v = np.linalg.norm(vec, axis=-1, keepdims=True)
    n_r = np.linalg.norm(ref, axis=-1, keepdims=True)
    return (vec * (n_r / (n_v + EPS))).astype(np.float32)


def scene_means(values: np.ndarray, scene: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if mask is not None:
        v = np.where(mask, v, np.nan)
    uniq, inv = np.unique(scene.astype(np.int64), return_inverse=True)
    finite = np.isfinite(v)
    sums = np.bincount(inv, weights=np.where(finite, v, 0.0), minlength=uniq.size)
    cnt = np.bincount(inv, weights=finite.astype(np.float64), minlength=uniq.size)
    means = sums / np.clip(cnt, 1.0, None)
    return means[cnt > 0]


def summarize(values: np.ndarray, scene: np.ndarray, mask: np.ndarray | None = None) -> dict:
    sm = scene_means(values, scene, mask)
    return {
        "n_frames": int(mask.sum()) if mask is not None else int(values.size),
        "n_scenes": int(sm.size),
        "mean": float(sm.mean()) if sm.size else float("nan"),
        "median": float(np.median(sm)) if sm.size else float("nan"),
        "fraction_positive": float((sm > 0).mean()) if sm.size else float("nan"),
        "bootstrap_ci": bootstrap_ci(sm),
    }


def complement_basis(u: np.ndarray, seed: int = 0) -> np.ndarray:
    d, r = u.shape
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(np.concatenate([u, rng.normal(size=(d, max(d - r, 1)))], axis=1))
    return q[:, r:]


def _load_u(root: Path, pid: str) -> np.ndarray:
    u = np.load(root / "subspace" / "consensus" / f"{pid}_U.npy").astype(np.float64)
    if u.ndim != 2:
        raise ValueError(f"U {pid} shape {u.shape}")
    if u.shape[0] != HIDDEN:
        if u.shape[1] == HIDDEN:
            u = u.T
        else:
            raise ValueError(f"U shape {u.shape} is not a 256-d basis")
    return u


def _pad_basis(q: np.ndarray, rank: int) -> np.ndarray:
    full = np.zeros((HIDDEN, rank), dtype=np.float64)
    r = min(q.shape[1], rank)
    full[:, :r] = q[:, :r]
    return full


def control_bases_dir(root: Path) -> Path:
    modern = root / "subspace" / "controls"
    legacy = root / "semantics" / "subspace"
    if (modern / "control_subspaces.pt").exists():
        return modern
    if (legacy / "control_subspaces.pt").exists():
        return legacy
    return modern


def ensure_control_bases(root: Path, pairs_doc: dict, *, force: bool = False) -> Path:
    """Build non-interaction / random controls from consensus U (no feature IDs)."""
    out_dir = root / "subspace" / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy = root / "semantics" / "subspace"
    if not force and (out_dir / "control_subspaces.pt").exists():
        return out_dir
    if not force and (legacy / "control_subspaces.pt").exists():
        return legacy

    rng = np.random.default_rng(0)
    q_rand = _pad_basis(orthonormal_basis(rng.normal(size=(FROZEN_K, HIDDEN))), FROZEN_K)
    rec: dict[str, dict] = {}
    for pair in pairs_doc["selected_pairs"]:
        pid = pair["pair_id"]
        u = _load_u(root, pid)
        q_int = _pad_basis(u, FROZEN_K)
        q_non = _pad_basis(complement_basis(u, seed=0), FROZEN_K)
        rec[pid] = {
            "record": q_int,
            "reactive": q_int,
            "record_nonint": q_non,
            "reactive_nonint": q_non,
        }
        print(f"  control bases {pid} U={u.shape} nonint={q_non.shape}", flush=True)

    torch.save(
        {
            "per_pair": {
                k: {kk: torch.from_numpy(vv) for kk, vv in v.items()} for k, v in rec.items()
            },
            "random": torch.from_numpy(q_rand),
            "construction": "nonint = QR complement of consensus U; random = QR N(0,1)",
        },
        out_dir / "control_subspaces.pt",
    )
    torch.save({pid: torch.from_numpy(rec[pid]["record"]) for pid in rec}, out_dir / "record_basis.pt")
    torch.save({pid: torch.from_numpy(rec[pid]["reactive"]) for pid in rec}, out_dir / "reactive_basis.pt")
    _write_json(out_dir / "metadata.json", {"rank": FROZEN_K, "pairs": list(rec)})
    return out_dir


def load_pair_bases(root: Path, pid: str) -> dict:
    u = _load_u(root, pid)
    bdir = control_bases_dir(root)
    q_r = torch.load(bdir / "record_basis.pt", weights_only=False)
    q_a = torch.load(bdir / "reactive_basis.pt", weights_only=False)
    ctrl = torch.load(bdir / "control_subspaces.pt", weights_only=False)
    per = ctrl["per_pair"][pid]
    return {
        "U": u,
        "Q_record": _as_np(q_r[pid]),
        "Q_reactive": _as_np(q_a[pid]),
        "Q_nonint_rea": _as_np(per["reactive_nonint"]),
        "Q_nonint_rec": _as_np(per["record_nonint"]),
        "Q_random": _as_np(ctrl["random"]),
        "rank": int(u.shape[1]),
        "controls_dir": str(bdir),
    }


def subset_pack(dkl, extra, scene, tight, high, low, *, rank: int, delta_norm: np.ndarray | None = None) -> dict:
    out = {
        "all": summarize(dkl, scene),
        "tight": summarize(dkl, scene, tight),
        "nominal": summarize(dkl, scene, ~tight),
        "high_DU": summarize(dkl, scene, high),
        "low_DU": summarize(dkl, scene, low),
        "e_dim_all": float(np.mean(dkl) / max(rank, 1)),
        "e_norm_all": float(np.mean(dkl) / (float(np.mean(delta_norm)) + EPS)) if delta_norm is not None else float("nan"),
    }
    if extra:
        out["p_star_gain_all"] = summarize(extra["p_star_gain"], scene)
        out["argmax_agree_delta"] = summarize(extra["argmax_agree_patch"] - extra["argmax_agree_base"], scene)
        out["js_reduction"] = summarize(extra["js_base"] - extra["js_patch"], scene)
        out["accel_err_reduction"] = summarize(extra["d_accel_to_r_base"] - extra["d_accel_to_r_patch"], scene)
        out["steer_err_reduction"] = summarize(extra["d_steer_to_r_base"] - extra["d_steer_to_r_patch"], scene)
    return out


def freeze_du_thresholds(root: Path, pairs_doc: dict, acts_root: Path) -> dict:
    """Percentiles of D_U on policy-train maps."""
    path = root / "intervention" / "divergence_bins" / "thresholds.json"
    if path.is_file():
        print("  reuse frozen D_U thresholds", flush=True)
        raw = json.loads(path.read_text())
        for rec in raw.values():
            edges = []
            for e in rec["edges"]:
                if isinstance(e, dict):
                    lab, lo, hi = e["lab"], e["lo"], e["hi"]
                else:
                    lab, lo, hi = e
                hi_f = float("inf") if str(hi) in ("inf", "Infinity") or hi is None else float(hi)
                edges.append((lab, float(lo), hi_f))
            rec["edges"] = edges
        return raw

    train_dir = Path(acts_root) / "train_dataset"
    shards = sorted((train_dir / "shards").glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"Missing train shards under {train_dir}")
    out: dict = {}
    for pair in pairs_doc["selected_pairs"]:
        pid = pair["pair_id"]
        u = _load_u(root, pid)
        rec_k, rea_k = pair["record"]["key"], pair["reactive"]["key"]
        dus = []
        for pth in shards:
            z = np.load(pth, allow_pickle=True)
            dlt = z[f"h_{rec_k}"].reshape(-1, HIDDEN).astype(np.float32) - z[f"h_{rea_k}"].reshape(-1, HIDDEN).astype(
                np.float32
            )
            dus.append(np.linalg.norm(dlt @ u, axis=-1))
            del z, dlt
        du = np.concatenate(dus)
        q50, q75, q90 = np.quantile(du, [0.5, 0.75, 0.9])
        out[pid] = {
            "q50": float(q50),
            "q75": float(q75),
            "q90": float(q90),
            "edges": [
                ["p0_50", 0.0, float(q50)],
                ["p50_75", float(q50), float(q75)],
                ["p75_90", float(q75), float(q90)],
                ["p90_100", float(q90), None],
            ],
        }
        print(f"  freeze D_U {pid} q50={q50:.3f} q90={q90:.3f} n={du.size}", flush=True)
    _write_json(path, out)
    return freeze_du_thresholds(root, pairs_doc, acts_root)


def run_pair(args, root: Path, pair: dict, bases: dict, q90: float, q50: float, bins: dict, acts_root: Path) -> dict:
    pid = pair["pair_id"]
    rec_k, rea_k = pair["record"]["key"], pair["reactive"]["key"]
    print(f"########## intervene {pid} ##########", flush=True)
    val = load_named_pair(Path(acts_root) / "validation_dataset", rec_k, rea_k)
    n, tlen, d = val["h_record"].shape
    h_r = val["h_record"].reshape(-1, d).astype(np.float32)
    h_a = val["h_reactive"].reshape(-1, d).astype(np.float32)
    log_r = val["logits_record"].reshape(-1, N_ACTIONS)
    log_a = val["logits_reactive"].reshape(-1, N_ACTIONS)
    scene = np.repeat(val["scene_id"].astype(np.int64), tlen)
    tight = val["tight"].reshape(-1)
    max_n = int(getattr(args, "max_frames", 0) or 0)
    if max_n > 0:
        h_r, h_a, log_r, log_a, scene, tight = (
            h_r[:max_n],
            h_a[:max_n],
            log_r[:max_n],
            log_a[:max_n],
            scene[:max_n],
            tight[:max_n],
        )

    delta_full = h_r - h_a
    u = bases["U"]
    delta_u = project(delta_full, u).astype(np.float32)
    du = np.linalg.norm(h_r @ u - h_a @ u, axis=-1)
    high, low = du >= q90, du <= q50
    rel_u = np.linalg.norm(delta_u, axis=-1) / (np.linalg.norm(h_a, axis=-1) + EPS)

    pol = load_policy_from_ckpt(Path(pair["reactive"]["ckpt"]), args.device)
    inner = _inner(pol)
    device = args.device
    h_a_t = torch.from_numpy(np.ascontiguousarray(h_a)).to(device, non_blocking=True)
    log_r_t = torch.from_numpy(np.ascontiguousarray(log_r)).to(device, non_blocking=True)
    log_a_t = torch.from_numpy(np.ascontiguousarray(log_a)).to(device, non_blocking=True)
    probe_n = min(4096, h_a_t.shape[0])
    recon_err = float((decode_t(inner, h_a_t[:probe_n]) - log_a_t[:probe_n]).abs().mean().item())
    print(f"  unpatched decode vs stored logits MAE={recon_err:.5f}", flush=True)
    accel_tbl = ACCEL_VALUES.to(device)
    steer_tbl = STEER_VALUES.to(device)

    delta_non = match_norm(project(delta_full, bases["Q_nonint_rea"]), delta_u)
    delta_perp = match_norm(project(delta_full, complement_basis(u, seed=1)), delta_u)
    rng = np.random.default_rng(0)
    rand_deltas = []
    for _ in range(N_RANDOM):
        qr = _pad_basis(orthonormal_basis(rng.normal(size=(FROZEN_K, HIDDEN))), FROZEN_K)
        rand_deltas.append(match_norm(project(delta_full, qr), delta_u))

    conditions = {
        "interaction": delta_u,
        "noninteraction": delta_non,
        "orthogonal": delta_perp,
        "full_hidden": delta_full.astype(np.float32),
        "sign_reverse": (-delta_u).astype(np.float32),
    }
    for i, rd in enumerate(rand_deltas):
        conditions[f"random_{i}"] = rd

    rec: dict = {
        "recon_mae": recon_err,
        "by_alpha": {},
        "perturbation": {
            "median_rel_norm_alpha1": float(np.median(rel_u)),
            "p90_rel_norm_alpha1": float(np.quantile(rel_u, 0.9)),
            "median_du": float(np.median(du)),
        },
        "corr_du_dkl": {},
        "bins": {},
        "du_q90": float(q90),
        "du_q50": float(q50),
    }

    for alpha in ALPHAS:
        akey = str(alpha)
        rec["by_alpha"][akey] = {}
        rec["perturbation"][akey] = {
            "median_r": float(np.median(alpha * rel_u)),
            "p95_r": float(np.quantile(alpha * rel_u, 0.95)),
        }
        for name, dlt in conditions.items():
            dlt_t = torch.from_numpy(np.ascontiguousarray(dlt)).to(device)
            log_p = decode_t(inner, h_a_t + float(alpha) * dlt_t)
            m = patch_metrics_t(log_r_t, log_a_t, log_p, accel_tbl, steer_tbl)
            pack = subset_pack(
                m["dkl"],
                m,
                scene,
                tight,
                high,
                low,
                rank=int(u.shape[1]),
                delta_norm=np.linalg.norm(dlt, axis=-1),
            )
            rec["by_alpha"][akey][name] = pack
            if name == "interaction":
                rec["corr_du_dkl"][akey] = float(np.corrcoef(du, m["dkl"])[0, 1]) if np.std(du) > 0 else float("nan")
                rec["bins"][akey] = {}
                for lab, lo, hi in bins["edges"]:
                    mask = (du >= lo) & (du < hi) if hi < np.inf else (du >= lo)
                    rec["bins"][akey][lab] = summarize(m["dkl"], scene, mask)
            print(
                f"  a={alpha:.2f} {name:16s} meanΔKL={pack['all']['mean']:.4f} "
                f"tight={pack['tight']['mean']:.4f} frac+={pack['all']['fraction_positive']:.3f}",
                flush=True,
            )
            del log_p, m, dlt_t

        rec["by_alpha"][akey]["random"] = {
            k: {
                "mean": float(np.mean([rec["by_alpha"][akey][f"random_{i}"][k]["mean"] for i in range(N_RANDOM)])),
                "fraction_positive": float(
                    np.mean([rec["by_alpha"][akey][f"random_{i}"][k]["fraction_positive"] for i in range(N_RANDOM)])
                ),
            }
            for k in ("all", "tight", "nominal", "high_DU", "low_DU")
        }

    # reverse: patch ReCord toward Reactive at α=1
    del h_a_t, log_r_t, log_a_t
    pol_r = load_policy_from_ckpt(Path(pair["record"]["ckpt"]), args.device)
    inner_r = _inner(pol_r)
    h_r_t = torch.from_numpy(np.ascontiguousarray(h_r)).to(device)
    log_r_t = torch.from_numpy(np.ascontiguousarray(log_r)).to(device)
    log_a_t = torch.from_numpy(np.ascontiguousarray(log_a)).to(device)
    dlt_t = torch.from_numpy(np.ascontiguousarray(-delta_u)).to(device)
    rev = patch_metrics_t(log_a_t, log_r_t, decode_t(inner_r, h_r_t + dlt_t), accel_tbl, steer_tbl)
    rec["reverse_alpha1"] = subset_pack(
        rev["dkl"], rev, scene, tight, high, low, rank=int(u.shape[1]), delta_norm=np.linalg.norm(delta_u, axis=-1)
    )
    print(f"  reverse a=1 meanΔKL={rec['reverse_alpha1']['all']['mean']:.4f}", flush=True)

    seed_dir = root / "intervention" / "immediate" / f"seed_{pair['train_seed']}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    _write_json(seed_dir / "metrics.json", rec)
    del val, h_r, h_a, pol, pol_r, h_r_t, log_r_t, log_a_t, dlt_t
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rec


def _across_pairs(all_pairs: dict) -> dict:
    def grab(alpha: str, cond: str, subset: str = "tight") -> dict:
        vals = [float(all_pairs[p]["by_alpha"][alpha][cond][subset]["mean"]) for p in all_pairs]
        a = np.asarray(vals, dtype=np.float64)
        return {"per_pair": vals, "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0}

    return {
        "interaction_tight_a0.5": grab("0.5", "interaction"),
        "interaction_tight_a1.0": grab("1.0", "interaction"),
        "interaction_nominal_a1.0": grab("1.0", "interaction", "nominal"),
        "random_tight_a1.0": grab("1.0", "random"),
        "noninteraction_tight_a1.0": grab("1.0", "noninteraction"),
        "sign_reverse_tight_a1.0": grab("1.0", "sign_reverse"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default=str(RESULTS_MECHANISM))
    p.add_argument(
        "--acts-root",
        default="",
        help="Collect root with {train,validation}_dataset (default: <out-root>/policy_seed_replication)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all val frames")
    args = p.parse_args()

    root = Path(args.out_root)
    acts_root = Path(args.acts_root) if args.acts_root else root / "policy_seed_replication"
    for sub in ("intervention/immediate", "intervention/divergence_bins", "subspace/controls"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    if not (root / "policy_seed_replication" / "summary.json").is_file():
        raise SystemExit(f"Missing replicate summary under {root}; run replicate.py first")
    if not (root / "subspace" / "consensus").is_dir():
        raise SystemExit(f"Missing subspace/consensus under {root}")
    if not (acts_root / "validation_dataset" / "shards").is_dir():
        raise SystemExit(f"Missing collect validation shards under {acts_root}")

    pairs_doc = json.loads((root / "policy_seed_replication" / "policy_pairs.json").read_text())
    force = bool(int(os.environ.get("FORCE", "0")))

    print(f"intervene out={root} acts={acts_root}", flush=True)
    print("========== control bases ==========", flush=True)
    ensure_control_bases(root, pairs_doc, force=force)
    print("========== D_U thresholds ==========", flush=True)
    thr = freeze_du_thresholds(root, pairs_doc, acts_root)

    all_pairs = {}
    for pair in pairs_doc["selected_pairs"]:
        pid = pair["pair_id"]
        bases = load_pair_bases(root, pid)
        all_pairs[pid] = run_pair(args, root, pair, bases, thr[pid]["q90"], thr[pid]["q50"], thr[pid], acts_root)

    across = _across_pairs(all_pairs)
    _write_json(root / "intervention" / "immediate" / "summary.json", all_pairs)
    summary = {
        "config": {
            "primary_delta": "delta_U = U U^T (h_R - h_Rea); h'_Rea = h_Rea + alpha * delta_U",
            "subspace_rank": FROZEN_K,
            "alphas": list(ALPHAS),
            "primary_alpha": PRIMARY_ALPHA,
        },
        "by_policy_seed": all_pairs,
        "across_pairs": across,
    }
    _write_json(root / "intervention" / "summary.json", summary)
    if (root / "summary.json").is_file():
        top = json.loads((root / "summary.json").read_text())
        top["intervention"] = across
        _write_json(root / "summary.json", top)
    print(json.dumps(across, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
