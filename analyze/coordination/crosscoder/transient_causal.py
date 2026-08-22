#!/usr/bin/env python3
"""Characterize states carrying the transient causal intervention effect.

Offline from same-state val packs + consensus U, α=PRIMARY_ALPHA.
Reports sparsity / criticality / event / regeneration / actions / controls.

  python analyze/coordination/crosscoder/transient_causal.py --out-root ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_COORD = Path(__file__).resolve().parent.parent
if str(_COORD) not in sys.path:
    sys.path.insert(0, str(_COORD))

from common import ACCEL_VALUES, N_ACTIONS, N_STEER, STEER_VALUES  
from criticality_metrics import A_MAX  
from crosscoder.collect import load_named_pair  
from crosscoder.frozen_config import (  
    EVENT_WINDOW_HI,
    EVENT_WINDOW_LO,
    FROZEN_K,
    PRIMARY_ALPHA as ALPHA,
    RESULTS_MECHANISM,
)
from crosscoder.intervention import (  
    HIDDEN,
    _inner,
    decode_t,
    kl_t,
    load_pair_bases,
    match_norm,
    project,
)
from crosscoder.metrics import bootstrap_ci, orthonormal_basis, tight_onset  
from crosscoder.pipeline import _write_json  
from runtime import load_policy_from_ckpt  

EPS = 1e-8
EVENT_WINDOW = np.arange(EVENT_WINDOW_LO, EVENT_WINDOW_HI + 1)
HIGH_PCT = 90
MED_LO, MED_HI = 40, 60
LOW_PCT = 50
CUM_PCTS = (1, 5, 10, 20, 50)
CONDS = ("interaction", "random", "noninteraction", "sign_reverse")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    ra = a[m].argsort().argsort().astype(np.float64)
    rb = b[m].argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _cohens_d(hi: np.ndarray, lo: np.ndarray) -> float:
    a, b = hi[np.isfinite(hi)], lo[np.isfinite(lo)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    sp = math.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / max(a.size + b.size - 2, 1))
    return float((a.mean() - b.mean()) / (sp + EPS))


def _group_stats(x: np.ndarray, mask: np.ndarray) -> dict:
    v = x[mask & np.isfinite(x)]
    return {
        "n": int(v.size),
        "mean": float(v.mean()) if v.size else float("nan"),
        "median": float(np.median(v)) if v.size else float("nan"),
        "bootstrap_ci": bootstrap_ci(v) if v.size else [float("nan"), float("nan")],
    }


def _dist_summary(x: np.ndarray) -> dict:
    v = x[np.isfinite(x)]
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std()),
        "fraction_positive": float((v > 0).mean()),
        "percentiles": {str(q): float(np.percentile(v, q)) for q in (50, 75, 90, 95, 99)},
    }


def _cumulative_positive(dkl: np.ndarray) -> dict:
    pos = dkl[dkl > 0]
    if pos.size == 0:
        return {str(p): float("nan") for p in CUM_PCTS}
    order = np.sort(pos)[::-1]
    total = float(order.sum())
    return {
        str(p): float(order[: max(1, int(round(pos.size * p / 100.0)))].sum() / (total + EPS)) for p in CUM_PCTS
    }


def _criticality_frame(ttc, dist, closing) -> dict[str, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        a_req = np.where(closing > 0, -(closing**2) / (2.0 * np.maximum(dist, EPS)), 0.0)
        a_req = np.where(np.isfinite(dist) & np.isfinite(closing), a_req, np.nan)
        btn = a_req / (-A_MAX)
    return {
        "ttc": ttc.astype(np.float32),
        "dce_proxy": dist.astype(np.float32),
        "closing": closing.astype(np.float32),
        "a_long_req": a_req.astype(np.float32),
        "btn": btn.astype(np.float32),
    }


def _action_prims(logits: torch.Tensor, accel_tbl, steer_tbl) -> dict[str, np.ndarray]:
    p = torch.softmax(logits.float(), dim=-1)
    nact = logits.shape[-1]
    a_idx = torch.arange(nact, device=logits.device) // N_STEER
    s_idx = torch.arange(nact, device=logits.device) % N_STEER
    return {
        "accel": (p * accel_tbl[a_idx]).sum(-1).cpu().numpy().astype(np.float32),
        "steer": (p * steer_tbl[s_idx]).sum(-1).cpu().numpy().astype(np.float32),
        "entropy": (-(p * p.clamp_min(EPS).log()).sum(-1)).cpu().numpy().astype(np.float32),
        "p_brake": p[:, a_idx < 3].sum(-1).cpu().numpy().astype(np.float32),
        "p_strong_brake": p[:, a_idx == 0].sum(-1).cpu().numpy().astype(np.float32),
        "p_accel": p[:, a_idx > 3].sum(-1).cpu().numpy().astype(np.float32),
        "argmax": logits.argmax(-1).cpu().numpy().astype(np.int32),
    }


@torch.no_grad()
def compute_pair_frames(args, root: Path, pair: dict, bases: dict, acts_root: Path) -> dict:
    pid = pair["pair_id"]
    print(f"########## transient frames {pid} ##########", flush=True)
    val = load_named_pair(
        Path(acts_root) / "validation_dataset",
        pair["record"]["key"],
        pair["reactive"]["key"],
    )
    n, tlen, d = val["h_record"].shape
    h_r = val["h_record"].reshape(-1, d).astype(np.float32)
    h_a = val["h_reactive"].reshape(-1, d).astype(np.float32)
    log_r = val["logits_record"].reshape(-1, N_ACTIONS)
    log_a = val["logits_reactive"].reshape(-1, N_ACTIONS)
    scene = np.repeat(val["scene_id"].astype(np.int64), tlen)
    tight = val["tight"].reshape(-1)
    approach = val["approach"].reshape(-1)
    ttc = val["ttc"].reshape(-1).astype(np.float32)
    dist = val["min_dist"].reshape(-1).astype(np.float32)
    closing = val["closing"].reshape(-1).astype(np.float32)
    onset = tight_onset(val["tight"])

    u = bases["U"]
    delta_u = project(h_r - h_a, u).astype(np.float32)
    d_u = np.linalg.norm(h_r @ u - h_a @ u, axis=-1).astype(np.float32)
    delta_norm = np.linalg.norm(delta_u, axis=-1).astype(np.float32)
    d_u_ep = d_u.reshape(n, tlen)

    pol = load_policy_from_ckpt(Path(pair["reactive"]["ckpt"]), args.device)
    inner = _inner(pol)
    device = args.device
    h_a_t = torch.from_numpy(np.ascontiguousarray(h_a)).to(device)
    log_r_t = torch.from_numpy(np.ascontiguousarray(log_r)).to(device)
    log_a_t = torch.from_numpy(np.ascontiguousarray(log_a)).to(device)
    accel_tbl, steer_tbl = ACCEL_VALUES.to(device), STEER_VALUES.to(device)

    delta_non = match_norm(project(h_r - h_a, bases["Q_nonint_rea"]), delta_u)
    rng = np.random.default_rng(0)
    qr = orthonormal_basis(rng.normal(size=(FROZEN_K, HIDDEN)))
    if qr.shape[1] < FROZEN_K:
        pad = np.zeros((HIDDEN, FROZEN_K), dtype=np.float64)
        pad[:, : qr.shape[1]] = qr
        qr = pad
    else:
        qr = qr[:, :FROZEN_K]
    conditions = {
        "interaction": delta_u,
        "random": match_norm(project(h_r - h_a, qr), delta_u),
        "noninteraction": delta_non,
        "sign_reverse": (-delta_u).astype(np.float32),
    }

    prim_r = _action_prims(log_r_t, accel_tbl, steer_tbl)
    prim_a = _action_prims(log_a_t, accel_tbl, steer_tbl)
    dkl_base = kl_t(log_r_t, log_a_t).cpu().numpy().astype(np.float32)

    out_cond = {}
    for name, dlt in conditions.items():
        print(f"  decode α={ALPHA} {name}", flush=True)
        dlt_t = torch.from_numpy(np.ascontiguousarray(dlt)).to(device)
        log_p = decode_t(inner, h_a_t + float(ALPHA) * dlt_t)
        dkl = (dkl_base - kl_t(log_r_t, log_p).cpu().numpy()).astype(np.float32)
        prim_p = _action_prims(log_p, accel_tbl, steer_tbl)
        a_star = log_r_t.argmax(-1)
        rows = torch.arange(log_r_t.shape[0], device=device)
        p_a = torch.softmax(log_a_t.float(), -1)
        p_p = torch.softmax(log_p.float(), -1)
        out_cond[name] = {
            "dkl": dkl,
            "p_star_gain": (p_p[rows, a_star] - p_a[rows, a_star]).cpu().numpy().astype(np.float32),
            "accel_patch": prim_p["accel"],
            "p_brake_patch": prim_p["p_brake"],
            "p_accel_patch": prim_p["p_accel"],
            "entropy_patch": prim_p["entropy"],
            "argmax_agree_base": (prim_a["argmax"] == prim_r["argmax"]).astype(np.float32),
            "argmax_agree_patch": (prim_p["argmax"] == prim_r["argmax"]).astype(np.float32),
        }
        del log_p, dlt_t

    thr = json.loads((root / "intervention" / "divergence_bins" / "thresholds.json").read_text())
    q90, q50 = float(thr[pid]["q90"]), float(thr[pid]["q50"])
    crit = _criticality_frame(ttc, dist, closing)
    pack = {
        "scene_id": scene,
        "tight": tight.astype(bool),
        "approach": approach.astype(bool),
        "d_u": d_u,
        "delta_norm": delta_norm,
        "dkl_base": dkl_base,
        "accel_r": prim_r["accel"],
        "accel_a": prim_a["accel"],
        "steer_r": prim_r["steer"],
        "steer_a": prim_a["steer"],
        "entropy_a": prim_a["entropy"],
        "p_brake_a": prim_a["p_brake"],
        "p_accel_a": prim_a["p_accel"],
        "argmax_r": prim_r["argmax"],
        "argmax_a": prim_a["argmax"],
        "du_high": d_u >= q90,
        "du_low": d_u <= q50,
        "d_u_ep": d_u_ep,
        "onset_ep": onset.astype(np.int32),
        "ttc_ep": val["ttc"].astype(np.float32),
        "dist_ep": val["min_dist"].astype(np.float32),
        "closing_ep": val["closing"].astype(np.float32),
        **crit,
        **{f"{k}_{ck}": v for ck, cd in out_cond.items() for k, v in cd.items()},
    }
    del val, h_r, h_a, pol, h_a_t, log_r_t, log_a_t
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return pack


def effect_groups(dkl: np.ndarray) -> dict[str, np.ndarray]:
    finite = np.isfinite(dkl)
    thr_hi = np.percentile(dkl[finite], HIGH_PCT)
    thr_med_lo = np.percentile(dkl[finite], MED_LO)
    thr_med_hi = np.percentile(dkl[finite], MED_HI)
    thr_lo = np.percentile(dkl[finite], LOW_PCT)
    return {
        "high": finite & (dkl >= thr_hi),
        "medium": finite & (dkl >= thr_med_lo) & (dkl <= thr_med_hi),
        "low": finite & (dkl <= thr_lo),
        "thresholds": {
            "high_p90": float(thr_hi),
            "med_p40": float(thr_med_lo),
            "med_p60": float(thr_med_hi),
            "low_p50": float(thr_lo),
        },
    }


def analyze_sparsity(dkl: np.ndarray) -> dict:
    return {
        "distribution": _dist_summary(dkl),
        "cumulative_positive_mass": _cumulative_positive(dkl),
        "groups": effect_groups(dkl)["thresholds"],
    }


def analyze_criticality(pack: dict, dkl: np.ndarray, groups: dict) -> dict:
    out: dict = {"associations": {}}
    for m in ("ttc", "dce_proxy", "closing", "btn"):
        x = pack[m]
        out["associations"][m] = {
            "spearman_vs_dkl": _spearman(dkl, x),
            "high": _group_stats(x, groups["high"]),
            "low": _group_stats(x, groups["low"]),
            "cohens_d_high_vs_low": _cohens_d(x[groups["high"]], x[groups["low"]]),
        }
    out["tight"] = {
        "mean_dkl_tight": float(dkl[pack["tight"]].mean()) if pack["tight"].any() else float("nan"),
        "mean_dkl_nominal": float(dkl[~pack["tight"]].mean()),
        "frac_high_that_are_tight": float(pack["tight"][groups["high"]].mean()) if groups["high"].any() else float("nan"),
    }
    return out


def event_align_series(ep: np.ndarray, onset: np.ndarray, window=EVENT_WINDOW) -> dict:
    vals = []
    for i in range(ep.shape[0]):
        e = int(onset[i])
        if e < 0:
            continue
        row = []
        for tau in window:
            t = e + int(tau)
            row.append(float(ep[i, t]) if 0 <= t < ep.shape[1] and np.isfinite(ep[i, t]) else np.nan)
        vals.append(row)
    arr = np.asarray(vals, dtype=np.float64) if vals else np.zeros((0, window.size))
    means, meds, ns, cis = [], [], [], []
    for j in range(window.size):
        col = arr[:, j] if arr.size else np.array([])
        col = col[np.isfinite(col)]
        ns.append(int(col.size))
        means.append(float(col.mean()) if col.size else float("nan"))
        meds.append(float(np.median(col)) if col.size else float("nan"))
        cis.append(bootstrap_ci(col) if col.size else [float("nan"), float("nan")])
    return {"tau": window.tolist(), "mean": means, "median": meds, "n": ns, "bootstrap_ci": cis}


def analyze_event_aligned(pack: dict) -> dict:
    n, tlen = pack["d_u_ep"].shape
    dkl = pack["dkl_interaction"].reshape(n, tlen)
    return {
        "dkl_interaction": event_align_series(dkl, pack["onset_ep"]),
        "d_u": event_align_series(pack["d_u_ep"], pack["onset_ep"]),
        "n_events": int((pack["onset_ep"] >= 0).sum()),
    }


def analyze_natural_regeneration(pack: dict, artificial_half_life: float = 1.0) -> dict:
    d_u, onset = pack["d_u_ep"], pack["onset_ep"]
    thr = float(np.nanpercentile(d_u, 75))
    elevated = d_u >= thr
    durations = []
    for i in range(d_u.shape[0]):
        e = int(onset[i])
        if e < 0:
            continue
        if not elevated[i, e]:
            near = np.where(elevated[i, max(0, e - 5) : min(d_u.shape[1], e + 6)])[0]
            if near.size == 0:
                continue
            e = max(0, e - 5) + int(near[np.argmin(np.abs(near - 5))])
        lo = e
        while lo > 0 and elevated[i, lo - 1]:
            lo -= 1
        hi = e
        while hi + 1 < d_u.shape[1] and elevated[i, hi + 1]:
            hi += 1
        durations.append(hi - lo + 1)
    dur = np.asarray(durations, dtype=np.float64)
    return {
        "elevated_threshold_p75": thr,
        "n_events_with_elevated": int(dur.size),
        "duration_mean": float(dur.mean()) if dur.size else float("nan"),
        "duration_median": float(np.median(dur)) if dur.size else float("nan"),
        "duration_bootstrap_ci": bootstrap_ci(dur) if dur.size else [float("nan"), float("nan")],
        "artificial_patch_half_life_steps": artificial_half_life,
        "duration_gt_artificial_half_life_frac": float((dur > artificial_half_life).mean()) if dur.size else float("nan"),
    }


def analyze_actions(pack: dict, groups: dict) -> dict:
    hi = groups["high"]
    if not hi.any():
        return {}
    da = pack["accel_patch_interaction"] - pack["accel_a"]
    return {
        "n_high": int(hi.sum()),
        "delta_accel": _group_stats(da, hi),
        "delta_p_brake": _group_stats(pack["p_brake_patch_interaction"] - pack["p_brake_a"], hi),
        "delta_p_accel": _group_stats(pack["p_accel_patch_interaction"] - pack["p_accel_a"], hi),
        "p_star_gain": _group_stats(pack["p_star_gain_interaction"], hi),
        "fraction_more_braking": float(((pack["p_brake_patch_interaction"] - pack["p_brake_a"])[hi] > 0).mean()),
        "fraction_less_accel": float((da[hi] < 0).mean()),
    }


def analyze_controls(pack: dict, groups_inter: dict) -> dict:
    d_i = pack["dkl_interaction"]
    out = {}
    for c in ("random", "noninteraction", "sign_reverse"):
        d_c = pack[f"dkl_{c}"]
        out[c] = {
            "distribution": _dist_summary(d_c),
            "mean_dkl_tight": float(d_c[pack["tight"]].mean()) if pack["tight"].any() else float("nan"),
            "mean_dkl_high_inter_states": float(d_c[groups_inter["high"]].mean()) if groups_inter["high"].any() else float("nan"),
            "specific_tight": float((d_i - d_c)[pack["tight"]].mean()) if pack["tight"].any() else float("nan"),
        }
    return out


def _read_artificial_half_life(root: Path, pid: str) -> float:
    path = root / "intervention_persistence" / "summary.json"
    if not path.is_file():
        return 1.0
    pers = json.loads(path.read_text())
    one = pers.get("one_shot", {}).get(pid, {})
    hl = one.get("half_life_RU", 1.0)
    return float(hl) if np.isfinite(hl) else 1.0


def _across(all_pairs: dict) -> dict:
    def grab(fn):
        vals = [fn(all_pairs[p]) for p in all_pairs]
        a = np.asarray(vals, dtype=np.float64)
        return {"per_pair": vals, "mean": float(np.nanmean(a))}

    return {
        "mean_dkl": grab(lambda r: r["sparsity"]["distribution"]["mean"]),
        "cum10_positive_mass": grab(lambda r: r["sparsity"]["cumulative_positive_mass"]["10"]),
        "spearman_ttc": grab(lambda r: r["criticality"]["associations"]["ttc"]["spearman_vs_dkl"]),
        "tight_minus_nominal_dkl": grab(
            lambda r: r["criticality"]["tight"]["mean_dkl_tight"] - r["criticality"]["tight"]["mean_dkl_nominal"]
        ),
        "natural_elevated_duration": grab(lambda r: r["natural_regeneration"]["duration_mean"]),
        "fraction_more_braking_high": grab(lambda r: r["actions"].get("fraction_more_braking", float("nan"))),
        "interaction_minus_random_tight": grab(
            lambda r: r["criticality"]["tight"]["mean_dkl_tight"] - r["controls"]["random"]["mean_dkl_tight"]
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default=str(RESULTS_MECHANISM))
    p.add_argument(
        "--acts-root",
        default="",
        help="Collect root with validation_dataset (default: <out-root>/policy_seed_replication)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--pairs", default="all", help="comma pair_ids or all")
    args = p.parse_args()

    root = Path(args.out_root)
    acts_root = Path(args.acts_root) if args.acts_root else root / "policy_seed_replication"
    out = root / "transient_causal_characterization"
    (out / "per_pair").mkdir(parents=True, exist_ok=True)

    pairs_doc = json.loads((root / "policy_seed_replication" / "policy_pairs.json").read_text())
    pairs = pairs_doc["selected_pairs"]
    if args.pairs != "all":
        want = set(args.pairs.split(","))
        pairs = [x for x in pairs if x["pair_id"] in want]

    cfg = {
        "alpha": ALPHA,
        "subspace_rank": FROZEN_K,
        "policy_pairs": [x["pair_id"] for x in pairs],
        "acts_root": str(acts_root),
    }
    _write_json(out / "config.json", cfg)
    print(f"transient out={root} acts={acts_root}", flush=True)

    all_pairs: dict = {}
    for pair in pairs:
        pid = pair["pair_id"]
        bases = load_pair_bases(root, pid)
        pack = compute_pair_frames(args, root, pair, bases, acts_root)
        dkl = pack["dkl_interaction"]
        groups = effect_groups(dkl)
        hl = _read_artificial_half_life(root, pid)

        np.savez_compressed(
            out / "per_pair" / f"{pid}_frames.npz",
            d_u=pack["d_u"],
            dkl_interaction=dkl,
            tight=pack["tight"],
            ttc=pack["ttc"],
        )

        rec = {
            "sparsity": analyze_sparsity(dkl),
            "criticality": analyze_criticality(pack, dkl, groups),
            "event_aligned": analyze_event_aligned(pack),
            "natural_regeneration": analyze_natural_regeneration(pack, artificial_half_life=hl),
            "actions": analyze_actions(pack, groups),
            "controls": analyze_controls(pack, groups),
        }
        all_pairs[pid] = rec
        print(
            f"  {pid} meanΔKL={rec['sparsity']['distribution']['mean']:.4f} "
            f"cum10={rec['sparsity']['cumulative_positive_mass']['10']:.3f} "
            f"ρ(ttc)={rec['criticality']['associations']['ttc']['spearman_vs_dkl']:.3f} "
            f"nat_dur={rec['natural_regeneration']['duration_mean']:.2f}",
            flush=True,
        )

    across = _across(all_pairs)
    summary = {
        "config": cfg,
        "by_policy_seed": all_pairs,
        "across_pairs": across,
    }
    _write_json(out / "summary.json", summary)
    if (root / "summary.json").is_file():
        top = json.loads((root / "summary.json").read_text())
        top["transient"] = across
        _write_json(root / "summary.json", top)
    print(json.dumps(across, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
