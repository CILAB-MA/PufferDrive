#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_THRESHOLDS,
    METHODS,
    PRETTY,
    aggregate_numeric_across_seeds,
    hierarchical_seed_map_ci,
    jsonable,
    list_policy_ckpts,
    nanmean,
    paired_mean_ci,
)
from divergence import _flatten_numeric, _set_path
from rollout import load_ego_pack, rollout_per_ego, save_ego_pack

DEFAULT_SWEEP_CONDITIONS: list[tuple[str, float, int, int, str]] = [
    # dose @ canonical window
    ("bias1.0_t-15_-6_tight", 1.0, -15, -6, "tight"),
    ("bias1.5_t-15_-6_tight", 1.5, -15, -6, "tight"),
    ("bias2.5_t-15_-6_tight", 2.5, -15, -6, "tight"),
    ("bias3.5_t-15_-6_tight", 3.5, -15, -6, "tight"),
    # window @ bias 1.5
    ("bias1.5_t-20_-10_tight", 1.5, -20, -10, "tight"),
    ("bias1.5_t-10_-1_tight", 1.5, -10, -1, "tight"),
    ("bias1.5_t-20_-1_tight", 1.5, -20, -1, "tight"),
    # approach-onset: first 1.5s of approach
    ("bias1.5_t0_14_approach", 1.5, 0, 14, "approach"),
    ("bias2.5_t0_14_approach", 2.5, 0, 14, "approach"),
]


def _frac(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        return float(np.mean(x.astype(np.float64)))
    if not np.any(mask):
        return float("nan")
    return float(np.mean(x[mask].astype(np.float64)))


def _window_mean_traj(
    traj: np.ndarray,
    onset: np.ndarray,
    mask: np.ndarray,
    tau_lo: int,
    tau_hi: int,
) -> np.ndarray:
    n = int(onset.shape[0])
    out = np.full(n, np.nan, dtype=np.float64)
    t_max = int(traj.shape[1])
    for i in np.flatnonzero(mask):
        e = int(onset[i])
        if e < 0:
            continue
        vals = []
        for tau in range(tau_lo, tau_hi + 1):
            t = e + tau
            if 0 <= t < t_max and np.isfinite(traj[i, t]):
                vals.append(float(traj[i, t]))
        if vals:
            out[i] = float(np.mean(vals))
    return out


def _concat_packs(packs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = packs[0].keys()
    out: dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.concatenate([p[k] for p in packs], axis=0)
    return out


def approach_onset_from_pack(
    pack: dict[str, np.ndarray],
    *,
    ttc_approach: float,
    closing_approach: float,
    dist_approach: float,
) -> np.ndarray:
    d = pack["min_dist_traj"]
    ttc = pack["ttc_traj"]
    cl = pack["closing_traj"]
    n, t_max = d.shape
    out = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        for t in range(t_max):
            tt, dd, cc = ttc[i, t], d[i, t], cl[i, t]
            if not (np.isfinite(tt) and np.isfinite(dd) and np.isfinite(cc)):
                continue
            if cc >= closing_approach and tt < ttc_approach and dd < dist_approach:
                out[i] = t
                break
    return out


def _mask_stats(
    base: dict[str, np.ndarray],
    patched: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    onset: np.ndarray,
    tau_lo: int,
    tau_hi: int,
    tag: str,
) -> dict[str, Any]:
    if not np.any(mask):
        return {"n": 0}
    d_coll = patched["collided"][mask].astype(np.float64) - base["collided"][mask].astype(
        np.float64
    )
    d_tight = patched["had_tight"][mask].astype(np.float64) - base["had_tight"][mask].astype(
        np.float64
    )
    d_res = patched["resolved"][mask].astype(np.float64) - base["resolved"][mask].astype(
        np.float64
    )
    pb0 = _window_mean_traj(base["p_brake_traj"], onset, mask, tau_lo, tau_hi)
    pb1 = _window_mean_traj(patched["p_brake_traj"], onset, mask, tau_lo, tau_hi)
    ea0 = _window_mean_traj(base["exp_accel_traj"], onset, mask, tau_lo, tau_hi)
    ea1 = _window_mean_traj(patched["exp_accel_traj"], onset, mask, tau_lo, tau_hi)
    d_pb = (pb1 - pb0)[mask]
    d_ea = (ea1 - ea0)[mask]
    d_pb = d_pb[np.isfinite(d_pb)]
    d_ea = d_ea[np.isfinite(d_ea)]
    ci_c = paired_mean_ci(d_coll, seed=hash(tag + "c") % (2**31))
    ci_t = paired_mean_ci(d_tight, seed=hash(tag + "t") % (2**31))
    ci_r = paired_mean_ci(d_res, seed=hash(tag + "r") % (2**31))
    ci_pb = paired_mean_ci(d_pb, seed=hash(tag + "pb") % (2**31))
    ci_ea = paired_mean_ci(d_ea, seed=hash(tag + "ea") % (2**31))
    return {
        "n": int(mask.sum()),
        "coll_baseline": _frac(base["collided"], mask),
        "coll_patched": _frac(patched["collided"], mask),
        "delta_coll": ci_c["mean"],
        "delta_coll_ci_low": ci_c["ci_low"],
        "delta_coll_ci_high": ci_c["ci_high"],
        "ci_low": ci_c["ci_low"],
        "ci_high": ci_c["ci_high"],
        "tight_baseline": _frac(base["had_tight"], mask),
        "tight_patched": _frac(patched["had_tight"], mask),
        "delta_tight": ci_t["mean"],
        "delta_tight_ci_low": ci_t["ci_low"],
        "delta_tight_ci_high": ci_t["ci_high"],
        "resolved_baseline": _frac(base["resolved"], mask),
        "resolved_patched": _frac(patched["resolved"], mask),
        "delta_resolved": ci_r["mean"],
        "delta_resolved_ci_low": ci_r["ci_low"],
        "delta_resolved_ci_high": ci_r["ci_high"],
        "still_tight_patched": _frac(patched["had_tight"], mask),
        "early_p_brake_baseline": float(nanmean(pb0[mask])),
        "early_p_brake_patched": float(nanmean(pb1[mask])),
        "delta_early_p_brake": ci_pb["mean"],
        "delta_early_p_brake_ci_low": ci_pb["ci_low"],
        "delta_early_p_brake_ci_high": ci_pb["ci_high"],
        "early_exp_accel_baseline": float(nanmean(ea0[mask])),
        "early_exp_accel_patched": float(nanmean(ea1[mask])),
        "delta_early_exp_accel": ci_ea["mean"],
        "delta_early_exp_accel_ci_low": ci_ea["ci_low"],
        "delta_early_exp_accel_ci_high": ci_ea["ci_high"],
        "_diff_coll": d_coll,
        "_diff_tight": d_tight,
        "_diff_resolved": d_res,
        "_diff_pb": d_pb,
        "_diff_ea": d_ea,
    }


def compare_baseline_vs_patch(
    base: dict[str, np.ndarray],
    patched: dict[str, np.ndarray],
    *,
    tau_lo: int,
    tau_hi: int,
    rec: dict[str, np.ndarray] | None = None,
    bias_onset: np.ndarray | None = None,
    readout_onset: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare baseline vs patch. Optional rec pack enables R+/R− slices."""
    if patched["collided"].size != base["collided"].size:
        raise ValueError("baseline/patch ego counts differ")
    if not np.array_equal(base["scene_id"], patched["scene_id"]):
        raise ValueError("scene_id mismatch between baseline and patch")

    onset_bias = (
        bias_onset if bias_onset is not None else base["tight_onset"].astype(np.int32)
    )
    onset_read = readout_onset if readout_onset is not None else base["tight_onset"]
    n = int(base["collided"].size)
    m_tight = base["had_tight"].astype(bool) & (base["tight_onset"] >= 0)
    m_ap = base["had_approach"].astype(bool)

    out: dict[str, Any] = {
        "n_ego": n,
        "n_baseline_tight": int(m_tight.sum()),
        "n_baseline_approach": int(m_ap.sum()),
        "fullpop": {
            "coll_baseline": _frac(base["collided"]),
            "coll_patched": _frac(patched["collided"]),
            "delta_coll": _frac(patched["collided"]) - _frac(base["collided"]),
            "tight_baseline": _frac(base["had_tight"]),
            "tight_patched": _frac(patched["had_tight"]),
            "delta_tight": _frac(patched["had_tight"]) - _frac(base["had_tight"]),
        },
        "on_baseline_tight": _mask_stats(
            base, patched, m_tight, onset=onset_read, tau_lo=tau_lo, tau_hi=tau_hi, tag="tight"
        ),
        "on_baseline_approach": _mask_stats(
            base, patched, m_ap, onset=onset_read, tau_lo=tau_lo, tau_hi=tau_hi, tag="ap"
        ),
    }
    # Flat diffs for single-mode hierarchical aggregate (legacy keys).
    ot = out["on_baseline_tight"]
    ap = out["on_baseline_approach"]
    out["_diff_coll_tight"] = ot.get("_diff_coll", np.asarray([]))
    out["_diff_tight_ap"] = ap.get("_diff_tight", np.asarray([]))
    out["_diff_early_p_brake"] = ot.get("_diff_pb", np.asarray([]))
    out["_diff_early_exp_accel"] = ot.get("_diff_ea", np.asarray([]))

    if rec is not None:
        if int(rec["collided"].size) != n or not np.array_equal(rec["scene_id"], base["scene_id"]):
            raise ValueError("rec pack not aligned with reactive baseline")
        both_ap = rec["had_approach"] & base["had_approach"]
        m_rp = both_ap & rec["resolved"].astype(bool) & base["escalated"].astype(bool)
        m_rm = both_ap & rec["escalated"].astype(bool) & base["resolved"].astype(bool)
        onset_rp = np.where(base["tight_onset"] >= 0, base["tight_onset"], onset_bias)
        out["on_r_plus"] = _mask_stats(
            base, patched, m_rp, onset=onset_rp, tau_lo=tau_lo, tau_hi=tau_hi, tag="rp"
        )
        out["on_r_minus"] = _mask_stats(
            base, patched, m_rm, onset=onset_rp, tau_lo=tau_lo, tau_hi=tau_hi, tag="rm"
        )
        out["rec_reference"] = {
            "early_p_brake_on_baseline_tight": float(
                nanmean(
                    _window_mean_traj(
                        rec["p_brake_traj"], base["tight_onset"], m_tight, tau_lo, tau_hi
                    )[m_tight]
                )
            ),
            "early_p_brake_on_r_plus": float(
                nanmean(
                    _window_mean_traj(rec["p_brake_traj"], onset_rp, m_rp, tau_lo, tau_hi)[m_rp]
                )
            )
            if np.any(m_rp)
            else float("nan"),
        }
    return out


def aggregate_mechanism(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate single-condition seeds (legacy flat _diff_* stashes)."""
    flat_seeds = []
    for s in per_seed:
        slim = {k: v for k, v in s.items() if not str(k).startswith("_diff_")}
        clean: dict[str, Any] = {}
        for k, v in slim.items():
            if isinstance(v, dict):
                clean[k] = {kk: vv for kk, vv in v.items() if not str(kk).startswith("_diff_")}
            else:
                clean[k] = v
        flat_seeds.append(_flatten_numeric(clean))
    across = aggregate_numeric_across_seeds(flat_seeds)

    means: dict[str, Any] = {}
    for path, block in across.items():
        if isinstance(block, dict) and "mean" in block:
            _set_path(means, path, block["mean"])

    for key, stash in (
        ("delta_coll_on_baseline_tight", "_diff_coll_tight"),
        ("delta_tight_on_baseline_approach", "_diff_tight_ap"),
        ("delta_early_p_brake", "_diff_early_p_brake"),
        ("delta_early_exp_accel", "_diff_early_exp_accel"),
        ("delta_coll_r_plus", None),
        ("delta_resolved_r_plus", None),
    ):
        if stash is not None:
            diffs = [
                np.asarray(s.get(stash, []), dtype=np.float64) for s in per_seed if stash in s
            ]
        elif key == "delta_coll_r_plus":
            diffs = [
                np.asarray((s.get("on_r_plus") or {}).get("_diff_coll", []), dtype=np.float64)
                for s in per_seed
                if (s.get("on_r_plus") or {}).get("_diff_coll") is not None
            ]
        else:
            diffs = [
                np.asarray((s.get("on_r_plus") or {}).get("_diff_resolved", []), dtype=np.float64)
                for s in per_seed
                if (s.get("on_r_plus") or {}).get("_diff_resolved") is not None
            ]
        diffs = [d for d in diffs if d.size]
        if not diffs:
            continue
        hci = hierarchical_seed_map_ci(diffs, seed=hash(key) % (2**31))
        means.setdefault("hierarchical", {})[key] = {
            "mean": hci["mean"],
            "ci_low": hci["ci_low"],
            "ci_high": hci["ci_high"],
            "n_seeds": hci.get("n_seeds"),
        }

    return {"aggregate": means, "across_seeds": across}


def aggregate_condition(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one sweep condition across seeds (nested _diff_*)."""
    flat_seeds = []
    for s in per_seed:
        slim = {k: v for k, v in s.items() if not str(k).startswith("_diff_") and k != "condition"}
        clean: dict[str, Any] = {}
        for k, v in slim.items():
            if isinstance(v, dict):
                clean[k] = {kk: vv for kk, vv in v.items() if not str(kk).startswith("_diff_")}
            else:
                clean[k] = v
        flat_seeds.append(_flatten_numeric(clean))
    across = aggregate_numeric_across_seeds(flat_seeds)
    means: dict[str, Any] = {}
    for path, block in across.items():
        if isinstance(block, dict) and "mean" in block:
            _set_path(means, path, block["mean"])

    for key, path_stash in (
        ("delta_coll_tight", ("on_baseline_tight", "_diff_coll")),
        ("delta_coll_r_plus", ("on_r_plus", "_diff_coll")),
        ("delta_resolved_r_plus", ("on_r_plus", "_diff_resolved")),
        ("delta_tight_approach", ("on_baseline_approach", "_diff_tight")),
    ):
        diffs = []
        for s in per_seed:
            blk: Any = s
            ok = True
            for p in path_stash[:-1]:
                if p not in blk:
                    ok = False
                    break
                blk = blk[p]
            if not ok or path_stash[-1] not in blk:
                continue
            diffs.append(np.asarray(blk[path_stash[-1]], dtype=np.float64))
        if not diffs:
            continue
        hci = hierarchical_seed_map_ci(diffs, seed=hash(key) % (2**31))
        means.setdefault("hierarchical", {})[key] = {
            "mean": hci["mean"],
            "ci_low": hci["ci_low"],
            "ci_high": hci["ci_high"],
            "n_seeds": hci.get("n_seeds"),
        }
    return {"aggregate": means, "across_seeds": across}


def rollout_patch_sharded(
    *,
    ckpt: Path,
    bias_onset: np.ndarray,
    num_maps: int,
    shard_size: int,
    device: str,
    brake_logit_bias: float,
    bias_tau_lo: int,
    bias_tau_hi: int,
    thr: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Patch rollout in shards (heap-safe for large num_maps)."""
    if num_maps <= shard_size:
        return rollout_per_ego(
            ckpt=ckpt,
            num_maps=num_maps,
            device=device,
            capture_readout=True,
            brake_logit_bias=float(brake_logit_bias),
            bias_onset=bias_onset,
            bias_tau_lo=int(bias_tau_lo),
            bias_tau_hi=int(bias_tau_hi),
            map_start=0,
            **thr,
        )
    parts: list[dict[str, np.ndarray]] = []
    for start in range(0, num_maps, shard_size):
        n = min(shard_size, num_maps - start)
        print(f"    shard map_start={start} n={n}", flush=True)
        parts.append(
            rollout_per_ego(
                ckpt=ckpt,
                num_maps=n,
                device=device,
                capture_readout=True,
                brake_logit_bias=float(brake_logit_bias),
                bias_onset=bias_onset[start : start + n],
                bias_tau_lo=int(bias_tau_lo),
                bias_tau_hi=int(bias_tau_hi),
                map_start=start,
                **thr,
            )
        )
    return _concat_packs(parts)


def _load_or_roll_baseline(
    *,
    ckpt: Path,
    seed_i: int,
    num_maps: int,
    shard_size: int,
    device: str,
    reuse_baseline: bool,
    base_path: Path,
    baseline_dir: Path,
    thr: dict[str, Any],
) -> dict[str, np.ndarray]:
    if reuse_baseline:
        for cand in (base_path, baseline_dir / f"seed{seed_i}_reactive.npz"):
            if not cand.is_file():
                continue
            pack = load_ego_pack(cand)
            n_pack = int(pack["collided"].size)
            if n_pack < num_maps or "p_brake_traj" not in pack or "tight_onset" not in pack:
                continue
            if n_pack > num_maps:
                pack = {k: v[:num_maps] for k, v in pack.items()}
                print(f"  [baseline] reuse {cand}[:{num_maps}] (from n={n_pack})", flush=True)
            else:
                print(f"  [baseline] reuse {cand}", flush=True)
            if cand != base_path or n_pack != num_maps:
                save_ego_pack(base_path, pack)
            return pack

    print(f"  [baseline] {ckpt.name} maps={num_maps} shard={shard_size}", flush=True)
    if num_maps <= shard_size:
        base = rollout_per_ego(
            ckpt=ckpt, num_maps=num_maps, device=device, capture_readout=True, **thr
        )
    else:
        parts = []
        for start in range(0, num_maps, shard_size):
            n = min(shard_size, num_maps - start)
            print(f"    baseline shard map_start={start} n={n}", flush=True)
            parts.append(
                rollout_per_ego(
                    ckpt=ckpt,
                    num_maps=n,
                    device=device,
                    capture_readout=True,
                    map_start=start,
                    **thr,
                )
            )
        base = _concat_packs(parts)
    save_ego_pack(base_path, base)
    return base


def _load_rec_prefix(baseline_dir: Path, seed_i: int, num_maps: int) -> dict[str, np.ndarray] | None:
    cand = baseline_dir / f"seed{seed_i}_record.npz"
    if not cand.is_file():
        return None
    rp = load_ego_pack(cand)
    if int(rp["collided"].size) < num_maps:
        return None
    return {k: v[:num_maps] for k, v in rp.items()}


def _strip_diffs(s: dict[str, Any]) -> dict[str, Any]:
    so: dict[str, Any] = {}
    for k, v in s.items():
        if str(k).startswith("_diff_"):
            continue
        if isinstance(v, dict):
            so[k] = {kk: vv for kk, vv in v.items() if not str(kk).startswith("_diff_")}
        else:
            so[k] = v
    return so


def run_single(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_root) / "mechanism_patch"
    pack_dir = out_dir / "packs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = (
        Path(args.baseline_pack_dir)
        if args.baseline_pack_dir
        else Path(args.out_root) / "ego_readout" / "packs"
    )
    ckpts = list_policy_ckpts(
        Path(args.experiments_root), METHODS["reactive"], max_n=args.max_seeds
    )
    if not ckpts:
        raise SystemExit("no reactive checkpoints")
    n_seeds = len(ckpts)
    shard_size = int(args.shard_size) if int(args.shard_size) > 0 else int(args.num_maps)
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
        print(f"\n########## seed index {i} (mechanism single) ##########", flush=True)
        ckpt = ckpts[i]
        base_path = pack_dir / f"seed{i}_reactive_baseline.npz"
        patch_path = pack_dir / f"seed{i}_reactive_patch.npz"
        base = _load_or_roll_baseline(
            ckpt=ckpt,
            seed_i=i,
            num_maps=int(args.num_maps),
            shard_size=shard_size,
            device=args.device,
            reuse_baseline=bool(args.reuse_baseline),
            base_path=base_path,
            baseline_dir=baseline_dir,
            thr=thr,
        )
        rec = _load_rec_prefix(baseline_dir, i, int(args.num_maps))
        bias_onset = base["tight_onset"].astype(np.int32)

        print(
            f"  [patch] bias={args.brake_logit_bias} "
            f"tau=[{args.bias_tau_lo},{args.bias_tau_hi}] "
            f"maps={args.num_maps} shard={shard_size}",
            flush=True,
        )
        patched = rollout_patch_sharded(
            ckpt=ckpt,
            bias_onset=bias_onset,
            num_maps=int(args.num_maps),
            shard_size=shard_size,
            device=args.device,
            brake_logit_bias=float(args.brake_logit_bias),
            bias_tau_lo=int(args.bias_tau_lo),
            bias_tau_hi=int(args.bias_tau_hi),
            thr=thr,
        )
        save_ego_pack(patch_path, patched)

        stats = compare_baseline_vs_patch(
            base,
            patched,
            tau_lo=int(args.bias_tau_lo),
            tau_hi=int(args.bias_tau_hi),
            rec=rec,
            bias_onset=bias_onset,
        )
        stats["seed_index"] = i
        stats["ckpt"] = ckpt.name
        stats["method"] = PRETTY["reactive"]
        per_seed.append(stats)
        ot = stats["on_baseline_tight"]
        print(
            f"  Δcoll|tight0={ot['delta_coll']:+.4f}  "
            f"Δp_brake_early={ot['delta_early_p_brake']:+.4f}  "
            f"n_tight0={stats['n_baseline_tight']}",
            flush=True,
        )

    across = aggregate_mechanism(per_seed)
    summary = {
        "config": {
            "mode": "single",
            "num_maps": args.num_maps,
            "shard_size": shard_size,
            "max_seeds": n_seeds,
            "brake_logit_bias": args.brake_logit_bias,
            "bias_tau_lo": args.bias_tau_lo,
            "bias_tau_hi": args.bias_tau_hi,
            "anchor": "tight",
            "target_alias": "reactive",
            **thr,
        },
        "aggregate": across["aggregate"],
        "across_seeds": across["across_seeds"],
        "per_seed": [jsonable(_strip_diffs(s)) for s in per_seed],
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    print(f"\nWrote {out_dir / 'summary.json'}", flush=True)


def run_sweep(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_root) / "mechanism_deep"
    pack_dir = out_dir / "packs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = (
        Path(args.baseline_pack_dir)
        if args.baseline_pack_dir
        else Path(args.out_root) / "ego_readout" / "packs"
    )
    ckpts = list_policy_ckpts(
        Path(args.experiments_root), METHODS["reactive"], max_n=args.max_seeds
    )
    if not ckpts:
        raise SystemExit("no reactive checkpoints")
    n_seeds = len(ckpts)
    shard = int(args.shard_size) if int(args.shard_size) > 0 else int(args.num_maps)
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
    conditions = DEFAULT_SWEEP_CONDITIONS

    by_condition: dict[str, Any] = {}
    for cond_name, bias, tau_lo, tau_hi, anchor in conditions:
        print(f"\n======== condition {cond_name} ========", flush=True)
        per_seed: list[dict] = []
        for i in range(n_seeds):
            print(f"\n## seed {i}", flush=True)
            ckpt = ckpts[i]
            base_path = pack_dir / f"seed{i}_reactive_baseline.npz"
            base = _load_or_roll_baseline(
                ckpt=ckpt,
                seed_i=i,
                num_maps=int(args.num_maps),
                shard_size=shard,
                device=args.device,
                reuse_baseline=bool(args.reuse_baseline),
                base_path=base_path,
                baseline_dir=baseline_dir,
                thr=thr,
            )
            rec = _load_rec_prefix(baseline_dir, i, int(args.num_maps))
            ap_onset = approach_onset_from_pack(
                base,
                ttc_approach=thr["ttc_approach"],
                closing_approach=thr["closing_approach"],
                dist_approach=thr["dist_approach"],
            )
            if anchor == "tight":
                bias_onset = base["tight_onset"].astype(np.int32)
            elif anchor == "approach":
                bias_onset = ap_onset
            else:
                raise ValueError(anchor)
            readout_onset = ap_onset if anchor == "approach" else base["tight_onset"]

            patch_path = pack_dir / f"seed{i}_{cond_name}.npz"
            patched = None
            if patch_path.is_file():
                cand = load_ego_pack(patch_path)
                if int(cand["collided"].size) == args.num_maps:
                    print(f"  reuse patch {patch_path.name}", flush=True)
                    patched = cand
            if patched is None:
                print(
                    f"  patch bias={bias} tau=[{tau_lo},{tau_hi}] anchor={anchor}",
                    flush=True,
                )
                patched = rollout_patch_sharded(
                    ckpt=ckpt,
                    bias_onset=bias_onset,
                    num_maps=int(args.num_maps),
                    shard_size=shard,
                    device=args.device,
                    brake_logit_bias=float(bias),
                    bias_tau_lo=int(tau_lo),
                    bias_tau_hi=int(tau_hi),
                    thr=thr,
                )
                save_ego_pack(patch_path, patched)

            stats = compare_baseline_vs_patch(
                base,
                patched,
                tau_lo=tau_lo,
                tau_hi=tau_hi,
                rec=rec,
                bias_onset=bias_onset,
                readout_onset=readout_onset,
            )
            stats["seed_index"] = i
            stats["condition"] = cond_name
            per_seed.append(stats)
            ot = stats["on_baseline_tight"]
            rp = stats.get("on_r_plus") or {}
            print(
                f"  tight Δcoll={ot.get('delta_coll', float('nan')):+.4f} "
                f"Δpb={ot.get('delta_early_p_brake', float('nan')):+.4f} | "
                f"R+ n={rp.get('n', 0)} Δcoll={rp.get('delta_coll', float('nan')):+.4f} "
                f"Δres={rp.get('delta_resolved', float('nan')):+.4f}",
                flush=True,
            )

        across = aggregate_condition(per_seed)
        by_condition[cond_name] = {
            "config": {"bias": bias, "tau_lo": tau_lo, "tau_hi": tau_hi, "anchor": anchor},
            "aggregate": across["aggregate"],
            "across_seeds": across["across_seeds"],
            "per_seed": [jsonable(_strip_diffs(s)) for s in per_seed],
        }

    match_note: dict[str, Any] = {}
    for name, block in by_condition.items():
        if block["config"]["anchor"] != "tight":
            continue
        agg = block["aggregate"]
        ot = agg.get("on_baseline_tight") or {}
        ref = (agg.get("rec_reference") or {}).get("early_p_brake_on_baseline_tight")
        pb = ot.get("early_p_brake_patched")
        if ref is None or pb is None:
            continue
        match_note[name] = {
            "patched_p_brake": pb,
            "rec_p_brake": ref,
            "abs_gap": abs(float(pb) - float(ref)),
            "delta_coll": ot.get("delta_coll"),
        }
    best_match = (
        min(match_note.items(), key=lambda kv: kv[1]["abs_gap"])[0] if match_note else None
    )

    summary = {
        "config": {
            "mode": "sweep",
            "num_maps": args.num_maps,
            "max_seeds": n_seeds,
            "shard_size": shard,
            "conditions": [c[0] for c in conditions],
        },
        "dose_match_to_rec": {"by_condition": match_note, "best": best_match},
        "by_condition": by_condition,
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    print(f"\nWrote {out_dir / 'summary.json'}", flush=True)
    if best_match:
        print(f"Closest Rec p_brake match: {best_match}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brake-logit patch mechanism (single|sweep)")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=("single", "sweep"),
        help="single=one condition → mechanism_patch/; sweep=grid → mechanism_deep/",
    )
    parser.add_argument("--experiments-root", type=str, default="/data/puffer/experiments")
    parser.add_argument("--out-root", type=str, default="/data/puffer/results/coordination")
    parser.add_argument("--baseline-pack-dir", type=str, default="")
    parser.add_argument("--num-maps", type=int, default=2000)
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Max maps per vecenv (heap-safe). 0 = no shard.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-seeds", type=int, default=3)
    parser.add_argument("--reuse-baseline", action="store_true")
    parser.add_argument("--brake-logit-bias", type=float, default=1.5)
    parser.add_argument("--bias-tau-lo", type=int, default=-15)
    parser.add_argument("--bias-tau-hi", type=int, default=-6)
    parser.add_argument("--ttc-approach", type=float, default=DEFAULT_THRESHOLDS["ttc_approach"])
    parser.add_argument("--closing-approach", type=float, default=DEFAULT_THRESHOLDS["closing_approach"])
    parser.add_argument("--dist-approach", type=float, default=DEFAULT_THRESHOLDS["dist_approach"])
    parser.add_argument("--ttc-tight", type=float, default=DEFAULT_THRESHOLDS["ttc_tight"])
    parser.add_argument("--closing-tight", type=float, default=DEFAULT_THRESHOLDS["closing_tight"])
    parser.add_argument("--dist-tight", type=float, default=DEFAULT_THRESHOLDS["dist_tight"])
    parser.add_argument("--pre-window", type=int, default=DEFAULT_THRESHOLDS["pre_window"])
    parser.add_argument("--hard-brake", type=float, default=DEFAULT_THRESHOLDS["hard_brake"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "single":
        run_single(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
