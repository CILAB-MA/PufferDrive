#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

import numpy as np

from common import (
    ORDER,
    aggregate_numeric_across_seeds,
    braking_lead_tau,
    geom_matched_paired_delta,
    kinematic_adjusted_method_effect,
    nanmean,
    paired_mean_ci,
    subset_mean,
)
from divergence import _flatten_numeric, _set_path, divergence_masks
import argparse
import json
from pathlib import Path

from common import DEFAULT_THRESHOLDS, METHODS, PRETTY, hierarchical_seed_map_ci, jsonable, list_policy_ckpts
from rollout import load_ego_pack, rollout_per_ego, save_ego_pack

READOUT_SCALARS = (
    "pre_entry_speed",
    "entry_speed",
    "pre_entry_exp_accel",
    "pre_entry_p_brake",
    "pre_entry_p_yield",
    "pre_entry_gap_press",
    "pre_entry_entropy",
    "tight_exp_accel",
    "tight_p_brake",
    "tight_p_yield",
    "tight_gap_press",
    "tight_entropy",
    "pre_tight_min_dist",
    "pre_tight_ttc",
)

# Primary paired readout keys that get map-level CIs + kinematic adjustment.
PRIMARY_READOUT = (
    "pre_entry_exp_accel",
    "pre_entry_p_brake",
    "pre_entry_gap_press",
    "pre_entry_entropy",
)

CURVE_KEYS = ("exp_accel", "p_brake", "entropy", "speed", "min_dist", "ttc")
TRAJ_REQUIRED = (
    "exp_accel_traj",
    "p_brake_traj",
    "entropy_traj",
    "speed_traj",
    "min_dist_traj",
    "ttc_traj",
    "tight_onset",
)
BRAKE_GAMMAS = (0.15, 0.20, 0.25)
BRAKE_CONSEC = 3
DEFAULT_WINDOW = 20


def pack_has_curves(pack: dict[str, np.ndarray]) -> bool:
    if not all(k in pack for k in TRAJ_REQUIRED):
        return False
    # Reject corrupt packs from interrupted rollouts (all-zero speed).
    speed = pack["speed_traj"]
    if not np.isfinite(speed).any() or float(np.nanmean(np.abs(speed))) < 1e-3:
        return False
    return True


def _paired_delta(rec: np.ndarray, rea: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    d = rec[mask].astype(np.float64) - rea[mask].astype(np.float64)
    d = d[np.isfinite(d)]
    return float(d.mean()) if d.size else float("nan")


def _paired_diff_array(rec: np.ndarray, rea: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.asarray([], dtype=np.float64)
    d = rec[mask].astype(np.float64) - rea[mask].astype(np.float64)
    return d[np.isfinite(d)]


def _traj_at_tau(pack: dict[str, np.ndarray], key: str, tau: int) -> np.ndarray:
    return _traj_at_shared_onset(pack, key, pack["tight_onset"], tau)


def _traj_at_shared_onset(
    pack: dict[str, np.ndarray],
    key: str,
    onset: np.ndarray,
    tau: int,
) -> np.ndarray:
    """Read traj at shared wall-clock onset[i] + tau (same map time for Rec/Rea)."""
    traj_name = {
        "exp_accel": "exp_accel_traj",
        "p_brake": "p_brake_traj",
        "entropy": "entropy_traj",
        "speed": "speed_traj",
        "min_dist": "min_dist_traj",
        "ttc": "ttc_traj",
    }[key]
    traj = pack[traj_name]
    n = int(onset.shape[0])
    out = np.full(n, np.nan, dtype=np.float64)
    t_max = int(traj.shape[1])
    for i in range(n):
        e = int(onset[i])
        if e < 0:
            continue
        t = e + tau
        if 0 <= t < t_max:
            out[i] = float(traj[i, t])
    return out


def analyze_event_curves(
    by_alias: dict[str, dict[str, np.ndarray]],
    mask: np.ndarray,
    *,
    window: int = DEFAULT_WINDOW,
    shared_onset: np.ndarray | None = None,
) -> dict[str, Any]:
    """Paired Rec/Rea curves. If shared_onset is set, both use that wall-clock anchor."""
    rec, rea = by_alias["record"], by_alias["reactive"]
    taus = list(range(-window, 1))
    curves: dict[str, Any] = {"taus": taus, "n_maps": int(mask.sum())}
    onset = shared_onset

    def _at(pack, key, tau):
        if onset is None:
            return _traj_at_tau(pack, key, tau)
        return _traj_at_shared_onset(pack, key, onset, tau)

    for key in CURVE_KEYS:
        means_rec, means_rea, deltas, ci_lo, ci_hi = [], [], [], [], []
        for tau in taus:
            zr = _at(rec, key, tau)
            za = _at(rea, key, tau)
            diffs = _paired_diff_array(zr, za, mask)
            ci = paired_mean_ci(diffs, seed=(abs(tau) * 17 + hash(key)) % (2**31))
            means_rec.append(subset_mean(zr, mask))
            means_rea.append(subset_mean(za, mask))
            deltas.append(ci["mean"])
            ci_lo.append(ci["ci_low"])
            ci_hi.append(ci["ci_high"])
        curves[key] = {
            "record": means_rec,
            "reactive": means_rea,
            "delta": deltas,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        }

    # Braking lead time (both-crossing primary; censored sensitivity)
    lead: dict[str, Any] = {}
    for gamma in BRAKE_GAMMAS:
        tau_rec = np.full(rec["tight_onset"].shape[0], np.nan)
        tau_rea = np.full(rea["tight_onset"].shape[0], np.nan)
        for i in np.flatnonzero(mask):
            e_rec = int(onset[i]) if onset is not None else int(rec["tight_onset"][i])
            e_rea = int(onset[i]) if onset is not None else int(rea["tight_onset"][i])
            tau_rec[i] = braking_lead_tau(
                rec["p_brake_traj"][i],
                e_rec,
                gamma=gamma,
                consec=BRAKE_CONSEC,
                window=window,
            )
            tau_rea[i] = braking_lead_tau(
                rea["p_brake_traj"][i],
                e_rea,
                gamma=gamma,
                consec=BRAKE_CONSEC,
                window=window,
            )
        both_cross = mask & np.isfinite(tau_rec) & np.isfinite(tau_rea)
        d_both = _paired_diff_array(tau_rec, tau_rea, both_cross)
        ci_both = paired_mean_ci(d_both, seed=int(gamma * 1000))

        # Right-censor non-crossers at end of window (tau = 0) for sensitivity.
        tau_rec_c = tau_rec.copy()
        tau_rea_c = tau_rea.copy()
        for i in np.flatnonzero(mask):
            if not np.isfinite(tau_rec_c[i]):
                tau_rec_c[i] = 0.0
            if not np.isfinite(tau_rea_c[i]):
                tau_rea_c[i] = 0.0
        d_cens = _paired_diff_array(tau_rec_c, tau_rea_c, mask)
        ci_cens = paired_mean_ci(d_cens, seed=int(gamma * 1000) + 7)

        lead[f"gamma_{gamma:g}"] = {
            "record_mean_both_cross": subset_mean(tau_rec, both_cross),
            "reactive_mean_both_cross": subset_mean(tau_rea, both_cross),
            "delta": ci_both["mean"],
            "ci_low": ci_both["ci_low"],
            "ci_high": ci_both["ci_high"],
            "n_both_cross": ci_both["n"],
            "frac_defined_rec": float(np.isfinite(tau_rec[mask]).mean()) if mask.any() else float("nan"),
            "frac_defined_rea": float(np.isfinite(tau_rea[mask]).mean()) if mask.any() else float("nan"),
            "censored_delta": ci_cens["mean"],
            "censored_ci_low": ci_cens["ci_low"],
            "censored_ci_high": ci_cens["ci_high"],
            "censored_n": ci_cens["n"],
            "record_mean": subset_mean(tau_rec, both_cross),
            "reactive_mean": subset_mean(tau_rea, both_cross),
            "n": ci_both["n"],
        }
    curves["brake_lead"] = {"consec": BRAKE_CONSEC, "by_gamma": lead}


    # Threshold-free window averages over τ ∈ [-W, 0]
    window_avg: dict[str, Any] = {}
    for key in ("exp_accel", "p_brake"):
        rec_wa = np.full(rec["tight_onset"].shape[0], np.nan)
        rea_wa = np.full(rea["tight_onset"].shape[0], np.nan)
        for i in np.flatnonzero(mask):
            vals_r, vals_a = [], []
            for tau in taus:
                vr = _at(rec, key, tau)[i]
                va = _at(rea, key, tau)[i]
                if np.isfinite(vr):
                    vals_r.append(float(vr))
                if np.isfinite(va):
                    vals_a.append(float(va))
            if vals_r:
                rec_wa[i] = float(np.mean(vals_r))
            if vals_a:
                rea_wa[i] = float(np.mean(vals_a))
        diffs = _paired_diff_array(rec_wa, rea_wa, mask)
        ci = paired_mean_ci(diffs, seed=hash(f"wa_{key}") % (2**31))
        window_avg[key] = {
            "record": subset_mean(rec_wa, mask),
            "reactive": subset_mean(rea_wa, mask),
            "delta": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "n": ci["n"],
        }
        curves[f"_diff_window_{key}"] = diffs
    curves["window_avg"] = window_avg

    # Geom-matched at selected taus
    matched = {}
    for tau in (-5, -1, 0):
        if tau < -window:
            continue
        block = {}
        for key in ("exp_accel", "p_brake"):
            zr = _at(rec, key, tau)
            za = _at(rea, key, tau)
            tr = _at(rec, "ttc", tau)
            ta = _at(rea, "ttc", tau)
            sr = _at(rec, "speed", tau)
            sa = _at(rea, "speed", tau)
            block[key] = geom_matched_paired_delta(zr, za, tr, ta, sr, sa, mask)
        matched[f"tau_{tau}"] = block
    curves["geom_matched"] = matched
    return curves


def _metric_block(
    rec: dict[str, np.ndarray],
    rea: dict[str, np.ndarray],
    key: str,
    mask: np.ndarray,
    *,
    sp: dict[str, np.ndarray] | None = None,
    with_ci: bool = False,
    with_adj: bool = False,
) -> dict[str, Any]:
    if key not in rec:
        return {}
    blk: dict[str, Any] = {
        "record": subset_mean(rec[key], mask),
        "reactive": subset_mean(rea[key], mask),
        "paired_delta": _paired_delta(rec[key], rea[key], mask),
    }
    if sp is not None and key in sp:
        blk["selfplay"] = subset_mean(sp[key], mask)
    if with_ci:
        diffs = _paired_diff_array(rec[key], rea[key], mask)
        ci = paired_mean_ci(diffs, seed=hash(key) % (2**31))
        blk["ci_low"] = ci["ci_low"]
        blk["ci_high"] = ci["ci_high"]
        blk["n_paired"] = ci["n"]
    if with_adj and "pre_tight_min_dist" in rec and "pre_tight_ttc" in rec:
        adj = kinematic_adjusted_method_effect(
            rec[key],
            rea[key],
            rec["pre_entry_speed"],
            rea["pre_entry_speed"],
            rec["pre_tight_ttc"],
            rea["pre_tight_ttc"],
            rec["pre_tight_min_dist"],
            rea["pre_tight_min_dist"],
            mask,
        )
        blk["adj_beta_rec"] = adj["beta_rec"]
        blk["adj_unadjusted"] = adj["unadjusted_paired_delta"]
        blk["adj_n_maps"] = adj["n_maps"]
    return blk


def analyze_readout_triplet(by_alias: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    rec, rea, sp = by_alias["record"], by_alias["reactive"], by_alias["selfplay"]
    n = int(rec["collided"].size)
    if rea["collided"].size != n or sp["collided"].size != n:
        raise ValueError("ego counts differ across methods — maps not aligned")

    m = divergence_masks(by_alias)
    both_tight = rec["had_tight"] & rea["had_tight"]
    # Paired readout primary: both methods have tight anchors.
    hard_tight = m["hard_tail"] & both_tight
    rec_ok = m["rec_ok_rea_coll"] & both_tight
    rea_ok = m["rec_coll_rea_ok"] & both_tight
    # Prevent slices must NOT require both-tight (one side resolved ⇒ no tight).
    res_esc = m["rec_res_rea_esc"]
    esc_res = m["rec_esc_rea_res"]

    def _subset_metrics(
        mask: np.ndarray,
        *,
        with_sp: bool = False,
        primary: bool = False,
    ) -> dict[str, Any]:
        block: dict[str, Any] = {"n": int(mask.sum())}
        for key in READOUT_SCALARS:
            if key not in rec:
                continue
            block[key] = _metric_block(
                rec,
                rea,
                key,
                mask,
                sp=sp if with_sp else None,
                with_ci=primary and key in PRIMARY_READOUT,
                with_adj=primary and key in PRIMARY_READOUT,
            )
        return block

    out: dict[str, Any] = {
        "n_ego": n,
        "has_curves": pack_has_curves(rec) and pack_has_curves(rea),
        "hard_tail": _subset_metrics(hard_tight, with_sp=True, primary=True),
        "on_rec_ok_rea_collide": _subset_metrics(rec_ok),
        "on_rec_coll_rea_ok": _subset_metrics(rea_ok),
        "on_rec_resolve_rea_escalate": _subset_metrics(res_esc),
        "on_rec_escalate_rea_resolve": _subset_metrics(esc_res),
    }

    if out["has_curves"]:
        curves = analyze_event_curves(by_alias, hard_tight)
        out["curves"] = curves
        # Prevent: align both trajs to the escalator's tight onset (shared wall-clock).
        res_esc_valid = res_esc & (rea["tight_onset"] >= 0)
        esc_res_valid = esc_res & (rec["tight_onset"] >= 0)
        out["curves_prevent_res_esc"] = analyze_event_curves(
            by_alias, res_esc_valid, shared_onset=rea["tight_onset"]
        )
        out["curves_prevent_esc_res"] = analyze_event_curves(
            by_alias, esc_res_valid, shared_onset=rec["tight_onset"]
        )
        # Prevent scalars from pack pre_entry are undefined on the resolve side;
        # surface shared-onset window averages instead.
        for subset_key, curve_key in (
            ("on_rec_resolve_rea_escalate", "curves_prevent_res_esc"),
            ("on_rec_escalate_rea_resolve", "curves_prevent_esc_res"),
        ):
            wa = (out.get(curve_key) or {}).get("window_avg") or {}
            if subset_key in out and wa:
                out[subset_key]["window_avg"] = wa
        out["_diff_pre_entry_exp_accel"] = _paired_diff_array(
            rec["pre_entry_exp_accel"], rea["pre_entry_exp_accel"], hard_tight
        )
        out["_diff_pre_entry_p_brake"] = _paired_diff_array(
            rec["pre_entry_p_brake"], rea["pre_entry_p_brake"], hard_tight
        )
        out["_diff_window_exp_accel"] = curves.get("_diff_window_exp_accel", np.asarray([]))
        out["_diff_window_p_brake"] = curves.get("_diff_window_p_brake", np.asarray([]))

    return out


def aggregate_readout_across_seeds(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std over seed replicates for nested readout fields + average curves."""
    flat_seeds = []
    for s in per_seed:
        slim = {
            k: v
            for k, v in s.items()
            if k
            not in {
                "curves",
                "curves_prevent_res_esc",
                "curves_prevent_esc_res",
                "has_curves",
                "seed_index",
                "ckpts",
            }
            and not str(k).startswith("_diff_")
        }
        flat_seeds.append(_flatten_numeric(slim))

    across = aggregate_numeric_across_seeds(flat_seeds, skip=set())
    means: dict[str, Any] = {}
    for path, block in across.items():
        if block["mean"] is not None:
            _set_path(means, path, block["mean"])

    # Hierarchical CIs for primary paired deltas → overlay on hard_tail metrics.
    for key, stash in (
        ("pre_entry_exp_accel", "_diff_pre_entry_exp_accel"),
        ("pre_entry_p_brake", "_diff_pre_entry_p_brake"),
        ("window_exp_accel", "_diff_window_exp_accel"),
        ("window_p_brake", "_diff_window_p_brake"),
    ):
        diffs = [np.asarray(s.get(stash, []), dtype=np.float64) for s in per_seed if stash in s]
        if not diffs:
            continue
        hci = hierarchical_seed_map_ci(diffs, seed=hash(key) % (2**31))
        means.setdefault("hierarchical", {})[key] = {
            "mean": hci["mean"],
            "ci_low": hci["ci_low"],
            "ci_high": hci["ci_high"],
            "n_seeds": hci.get("n_seeds"),
        }
        if key.startswith("pre_entry_"):
            blk = means.setdefault("hard_tail", {}).setdefault(key, {})
            blk["paired_delta"] = hci["mean"]
            blk["ci_low"] = hci["ci_low"]
            blk["ci_high"] = hci["ci_high"]

    # Average event curves across seeds that have them
    curve_seeds = [s["curves"] for s in per_seed if s.get("curves")]
    if curve_seeds:
        taus = curve_seeds[0]["taus"]
        merged: dict[str, Any] = {
            "taus": taus,
            "n_maps": nanmean(np.array([c["n_maps"] for c in curve_seeds], dtype=np.float64)),
        }
        for key in CURVE_KEYS:
            block: dict[str, list[float]] = {
                "record": [],
                "reactive": [],
                "delta": [],
                "ci_low": [],
                "ci_high": [],
            }
            for j, _tau in enumerate(taus):
                for field in block:
                    vals = []
                    for c in curve_seeds:
                        arr = (c.get(key) or {}).get(field, [])
                        if j < len(arr) and arr[j] is not None and np.isfinite(arr[j]):
                            vals.append(float(arr[j]))
                    block[field].append(nanmean(np.array(vals)) if vals else float("nan"))
            merged[key] = block
        by_gamma: dict[str, Any] = {}
        for gkey in curve_seeds[0]["brake_lead"]["by_gamma"]:
            fields = [
                "record_mean",
                "reactive_mean",
                "delta",
                "ci_low",
                "ci_high",
                "n",
                "n_both_cross",
                "frac_defined_rec",
                "frac_defined_rea",
                "censored_delta",
                "censored_ci_low",
                "censored_ci_high",
                "censored_n",
                "record_mean_both_cross",
                "reactive_mean_both_cross",
            ]
            by_gamma[gkey] = {
                f: nanmean(
                    np.array(
                        [
                            c["brake_lead"]["by_gamma"][gkey].get(f, float("nan"))
                            for c in curve_seeds
                        ],
                        dtype=np.float64,
                    )
                )
                for f in fields
            }
        merged["brake_lead"] = {"consec": BRAKE_CONSEC, "by_gamma": by_gamma}
        wa: dict[str, Any] = {}
        for metric in ("exp_accel", "p_brake"):
            wa[metric] = {
                f: nanmean(
                    np.array(
                        [
                            (c.get("window_avg") or {}).get(metric, {}).get(f, float("nan"))
                            for c in curve_seeds
                        ],
                        dtype=np.float64,
                    )
                )
                for f in ("record", "reactive", "delta", "ci_low", "ci_high", "n")
            }
            hk = f"window_{metric}"
            hier = (means.get("hierarchical") or {}).get(hk)
            if hier:
                wa[metric]["delta"] = hier["mean"]
                wa[metric]["ci_low"] = hier["ci_low"]
                wa[metric]["ci_high"] = hier["ci_high"]
        merged["window_avg"] = wa
        gm: dict[str, Any] = {}
        for tau_key, block0 in curve_seeds[0]["geom_matched"].items():
            gm[tau_key] = {}
            for metric in block0:
                gm[tau_key][metric] = {
                    "delta": nanmean(
                        np.array(
                            [
                                c["geom_matched"][tau_key][metric].get("delta", float("nan"))
                                for c in curve_seeds
                            ],
                            dtype=np.float64,
                        )
                    ),
                    "n_bins_used": nanmean(
                        np.array(
                            [
                                c["geom_matched"][tau_key][metric].get(
                                    "n_bins_used", float("nan")
                                )
                                for c in curve_seeds
                            ],
                            dtype=np.float64,
                        )
                    ),
                }
        merged["geom_matched"] = gm
        means["curves"] = merged

    def _avg_curve_key(key: str) -> None:
        seeds = [s[key] for s in per_seed if s.get(key)]
        if not seeds:
            return
        taus = seeds[0]["taus"]
        out_c: dict[str, Any] = {
            "taus": taus,
            "n_maps": nanmean(np.array([c["n_maps"] for c in seeds], dtype=np.float64)),
        }
        for ck in CURVE_KEYS:
            block: dict[str, list[float]] = {
                "record": [],
                "reactive": [],
                "delta": [],
                "ci_low": [],
                "ci_high": [],
            }
            for j, _tau in enumerate(taus):
                for field in block:
                    vals = []
                    for c in seeds:
                        arr = (c.get(ck) or {}).get(field, [])
                        if j < len(arr) and arr[j] is not None and np.isfinite(arr[j]):
                            vals.append(float(arr[j]))
                    block[field].append(nanmean(np.array(vals)) if vals else float("nan"))
            out_c[ck] = block
        wa: dict[str, Any] = {}
        for metric in ("exp_accel", "p_brake"):
            wa[metric] = {
                f: nanmean(
                    np.array(
                        [
                            (c.get("window_avg") or {}).get(metric, {}).get(f, float("nan"))
                            for c in seeds
                        ],
                        dtype=np.float64,
                    )
                )
                for f in ("record", "reactive", "delta", "ci_low", "ci_high", "n")
            }
        out_c["window_avg"] = wa
        means[key] = out_c

    _avg_curve_key("curves_prevent_res_esc")
    _avg_curve_key("curves_prevent_esc_res")

    means["has_curves"] = bool(any(s.get("has_curves") for s in per_seed))
    return {"aggregate": means, "across_seeds": across}


def main() -> None:

    parser = argparse.ArgumentParser(description="Ego policy readout on divergence scenes")
    parser.add_argument("--experiments-root", type=str, default="/data/puffer/experiments")
    parser.add_argument("--out-root", type=str, default="/data/puffer/results/coordination")
    parser.add_argument("--num-maps", type=int, default=600)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-seeds", type=int, default=3)
    parser.add_argument("--reuse-packs", action="store_true")
    parser.add_argument("--ttc-approach", type=float, default=DEFAULT_THRESHOLDS["ttc_approach"])
    parser.add_argument("--closing-approach", type=float, default=DEFAULT_THRESHOLDS["closing_approach"])
    parser.add_argument("--dist-approach", type=float, default=DEFAULT_THRESHOLDS["dist_approach"])
    parser.add_argument("--ttc-tight", type=float, default=DEFAULT_THRESHOLDS["ttc_tight"])
    parser.add_argument("--closing-tight", type=float, default=DEFAULT_THRESHOLDS["closing_tight"])
    parser.add_argument("--dist-tight", type=float, default=DEFAULT_THRESHOLDS["dist_tight"])
    parser.add_argument("--pre-window", type=int, default=DEFAULT_THRESHOLDS["pre_window"])
    parser.add_argument("--hard-brake", type=float, default=DEFAULT_THRESHOLDS["hard_brake"])
    args = parser.parse_args()

    out_dir = Path(args.out_root) / "ego_readout"
    pack_dir = out_dir / "packs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_dir.mkdir(parents=True, exist_ok=True)

    ckpts = {
        alias: list_policy_ckpts(Path(args.experiments_root), exp, max_n=args.max_seeds)
        for alias, exp in METHODS.items()
    }
    n_seeds = min(len(ckpts[a]) for a in ORDER)
    if n_seeds <= 0:
        raise SystemExit("no checkpoints found")

    thr = dict(
        ttc_approach=args.ttc_approach,
        closing_approach=args.closing_approach,
        dist_approach=args.dist_approach,
        ttc_tight=args.ttc_tight,
        closing_tight=args.closing_tight,
        dist_tight=args.dist_tight,
        pre_window=args.pre_window,
        hard_brake=args.hard_brake,
    )

    per_seed: list[dict] = []
    for i in range(n_seeds):
        print(f"\n########## seed index {i} (ego readout) ##########")
        by_alias: dict[str, dict[str, np.ndarray]] = {}
        for alias in ORDER:
            ckpt = ckpts[alias][i]
            pack_path = pack_dir / f"seed{i}_{alias}.npz"
            reuse = False
            if args.reuse_packs and pack_path.is_file():
                pack = load_ego_pack(pack_path)
                if pack_has_curves(pack):
                    print(f"  [{PRETTY[alias]}] reuse {pack_path.name}")
                    by_alias[alias] = pack
                    reuse = True
                else:
                    print(f"  [{PRETTY[alias]}] stale pack (no traj) — recapture")
            if not reuse:
                print(f"  [{PRETTY[alias]}] {ckpt.name} maps={args.num_maps} capture_readout")
                by_alias[alias] = rollout_per_ego(
                    ckpt=ckpt, num_maps=args.num_maps, device=args.device,
                    capture_readout=True, **thr,
                )
                save_ego_pack(pack_path, by_alias[alias])
            r = by_alias[alias]

        stats = analyze_readout_triplet(by_alias)
        stats["seed_index"] = i
        stats["ckpts"] = {a: ckpts[a][i].name for a in ORDER}
        per_seed.append(stats)
        ht = stats.get("hard_tail") or {}
        spd = ht.get("pre_entry_speed") or {}
        acc = ht.get("pre_entry_exp_accel") or {}

    across = aggregate_readout_across_seeds(per_seed)
    # Drop internal map-diff stashes from saved JSON (used only for hierarchical CI).
    clean_seeds = []
    for s in per_seed:
        so = {k: v for k, v in s.items() if not str(k).startswith("_diff_")}
        for ck in ("curves", "curves_prevent_res_esc", "curves_prevent_esc_res"):
            if isinstance(so.get(ck), dict):
                so[ck] = {
                    k: v for k, v in so[ck].items() if not str(k).startswith("_diff_")
                }
        clean_seeds.append(so)
    summary = {
        "config": {"num_maps": args.num_maps, **thr, "max_seeds": n_seeds},
        "aggregate": across["aggregate"],
        "across_seeds": across["across_seeds"],
        "per_seed": [jsonable(s) for s in clean_seeds],
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
