#!/usr/bin/env python3
"""Rigorous projection-attribution validation for paper claims.

Checks (see user spec):
  1. Observation-level median, bootstrap 95% CI, frac(ReCord>Reactive), Wilcoxon
  2. Raw / cos / σ-scaled attribution
  3. Max-pool vs winning-slot vs slot-only paths
  4. Extended action metrics (brake, throttle, steer, net slowing, accel, value, entropy)
  5. Finite-difference dose-response monotonicity (steering causal check)
  6. Blind feature selection (matched triples + enrichment, fixed before attribution)
  7. Negative controls (random dir, lane/non-conflict, permuted pairing)
  8. Policy-seed aggregation (multiple PBT runs per method)

Requires ``obs`` in activations.npz (``SAVE_OBS=1``).
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

from feature_matching import (  # noqa: E402
    build_triplet_matches,
    load_alive_mask,
    load_feature_acts,
    resolve_model_name,
)
from feature_steering import (  # noqa: E402
    EXTENDED_PRIMARY_METRICS,
    MODELS,
    PRETTY,
    discover_sae_ckpts,
    finite_difference_sweep,
    load_obs_bundle,
    projection_attribution_obs_level,
)
from stats_utils import bootstrap_ci, load_enrichment_scores, wilcoxon_onesided_greater  # noqa: E402
from sae_model import SparseAutoencoder  # noqa: E402

POOL_MODES = ("max", "winning_slot", "slot_only")
ATTR_VARIANTS = ("raw", "cos", "scaled")
PRIMARY_METRIC = "attr_p_brake"


def _load_matching_triples(matching_dir: Path) -> list[dict]:
    for name in (
        "triplet_matches.json",
        "matched_triples.json",
        "triples.json",
        "mutual_nn_triples.json",
    ):
        path = matching_dir / name
        if path.is_file():
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return list(data.get("triples") or data.get("matched_triples") or [])
            return list(data)
    # rebuild from pairwise match files
    rec_rea = matching_dir / "match_record_reactive.json"
    rec_sp = matching_dir / "match_record_selfplay.json"
    if rec_rea.is_file() and rec_sp.is_file():
        return build_triplet_matches(
            json.loads(rec_rea.read_text()),
            json.loads(rec_sp.read_text()),
        )
    return []


def pick_blind_conflict_features(
    analysis_dir: Path,
    semantics_dir: Path,
    matching_dir: Path,
    *,
    min_enrichment: float = 1.5,
    min_density: float = 1e-4,
    top_k: int = 5,
) -> dict[str, list[int]]:
    """Primary feature set fixed *before* attribution (semantic + matching criteria)."""
    triples = _load_matching_triples(matching_dir)
    if not triples:
        raise FileNotFoundError(f"No matched triples under {matching_dir}")

    enrich = {
        a: load_enrichment_scores(semantics_dir, resolve_model_name(a)) for a in MODELS
    }
    alive = {
        a: load_alive_mask(analysis_dir / resolve_model_name(a), min_density=min_density)
        for a in MODELS
    }

    scored: list[tuple[float, dict]] = []
    for trip in triples:
        rec = int(trip["record"])
        rea = int(trip["reactive"])
        sp = int(trip["selfplay"])
        if not (alive["record"][rec] and alive["reactive"][rea] and alive["selfplay"][sp]):
            continue
        e_rec = enrich["record"].get(rec, 0.0)
        e_rea = enrich["reactive"].get(rea, 0.0)
        e_sp = enrich["selfplay"].get(sp, 0.0)
        if min(e_rec, e_rea, e_sp) < min_enrichment:
            continue
        score = float(trip.get("score_min") or min(e_rec, e_rea, e_sp))
        scored.append((score, {"record": rec, "reactive": rea, "selfplay": sp, "score_min": score}))

    scored.sort(key=lambda x: -x[0])
    picked = [t for _, t in scored[:top_k]]
    out = {a: [t[a] for t in picked] for a in MODELS}
    return out


def pick_negative_control_features(
    analysis_dir: Path,
    semantics_dir: Path,
    *,
    n_random: int = 3,
    seed: int = 0,
) -> dict[str, dict[str, list[int]]]:
    """Lane-keeping / non-conflict features per model (low conflict enrichment)."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, list[int]]] = {a: {"lane_like": [], "non_conflict": []} for a in MODELS}
    for alias in MODELS:
        exp = resolve_model_name(alias)
        enrich = load_enrichment_scores(semantics_dir, exp)
        alive = load_alive_mask(analysis_dir / exp)
        ids = [fid for fid, e in enrich.items() if alive[fid] and np.isfinite(e)]
        if not ids:
            continue
        ranked = sorted(ids, key=lambda fid: enrich[fid])
        out[alias]["non_conflict"] = ranked[:n_random]
        out[alias]["lane_like"] = ranked[:n_random]
        # random alive features (not enrichment-selected)
        alive_ids = np.flatnonzero(alive).tolist()
        rng.shuffle(alive_ids)
        out[alias]["random_alive"] = [int(x) for x in alive_ids[:n_random]]
    return out


def random_decoder_directions(
    sae: SparseAutoencoder,
    feature_ids: list[int],
    *,
    seed: int,
) -> np.ndarray:
    """Unit-norm random directions replacing W_dec[j] for negative control.

    Reserved helper; primary neg controls use low-enrichment / random-alive
    feature ids instead of substituting W_dec rows.
    """
    rng = np.random.default_rng(seed)
    d_in = sae.W_dec.shape[1]
    out = np.zeros((len(feature_ids), d_in), dtype=np.float32)
    for i in range(len(feature_ids)):
        v = rng.standard_normal(d_in).astype(np.float32)
        v /= max(float(np.linalg.norm(v)), 1e-8)
        out[i] = v
    return out


def summarize_obs_attributions(
    arrays: np.ndarray,
    *,
    bootstrap_seed: int = 0,
) -> dict:
    """Observation-level summary: median, bootstrap CI, mean."""
    x = np.asarray(arrays, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    med_ci = bootstrap_ci(x, seed=bootstrap_seed, statistic="median")
    mean_ci = bootstrap_ci(x, seed=bootstrap_seed + 1, statistic="mean")
    return {
        "n": int(x.size),
        "median": med_ci["point"],
        "median_ci95": [med_ci["ci_low"], med_ci["ci_high"]],
        "mean": mean_ci["point"],
        "mean_ci95": [mean_ci["ci_low"], mean_ci["ci_high"]],
    }


def paired_policy_comparison(
    record_vals: np.ndarray,
    reactive_vals: np.ndarray,
    *,
    bootstrap_seed: int = 0,
) -> dict:
    """Paired obs-level comparison: ReCord − Reactive."""
    r = np.asarray(record_vals, dtype=np.float64).ravel()
    a = np.asarray(reactive_vals, dtype=np.float64).ravel()
    n = min(r.size, a.size)
    r, a = r[:n], a[:n]
    mask = np.isfinite(r) & np.isfinite(a)
    r, a = r[mask], a[mask]
    delta = r - a
    return {
        "n_paired": int(delta.size),
        "frac_record_gt_reactive": float(np.mean(delta > 0)) if delta.size else None,
        "median_delta": float(np.median(delta)) if delta.size else None,
        "mean_delta": float(np.mean(delta)) if delta.size else None,
        "delta_median_ci95": bootstrap_ci(delta, seed=bootstrap_seed, statistic="median"),
        "wilcoxon_record_gt_reactive": wilcoxon_onesided_greater(delta),
    }


def fd_monotonicity_report(fd_rows: list[dict]) -> list[dict]:
    """Check dose-response: +α ⇒ brake↑, −α ⇒ brake↓ (batch means)."""
    out = []
    for row in fd_rows:
        boosts = row.get("boosts") or {}
        ordered = []
        for key, b in boosts.items():
            alpha = float(b.get("alpha", 0.0))
            ordered.append((alpha, float(b.get("delta_p_brake", 0.0))))
        ordered.sort(key=lambda t: t[0])
        if len(ordered) < 2:
            mono = None
        else:
            deltas = [d for _, d in ordered]
            mono = all(deltas[i] <= deltas[i + 1] for i in range(len(deltas) - 1))
        pos_ok = all(d > 0 for a, d in ordered if a > 0) if ordered else None
        neg_ok = all(d < 0 for a, d in ordered if a < 0) if ordered else None
        out.append(
            {
                "feature_id": row["feature_id"],
                "monotonic_in_alpha": mono,
                "positive_alpha_increases_brake": pos_ok,
                "negative_alpha_decreases_brake": neg_ok,
                "ordered_deltas": [{"alpha": a, "delta_p_brake": d} for a, d in ordered],
            }
        )
    return out


def _load_policy(base_path: str, exp: str, probe_step: int, device, run_dir: str | None = None):
    from sae_rollout import (  # noqa: WPS433
        build_human_replay_drive_args,
        create_policy,
        create_vecenv,
        load_policy_from_checkpoint,
        pick_checkpoint,
        resolve_policy_location,
    )

    loc = resolve_policy_location(base_path, exp, run_dir, prefer_final=True)
    ckpt = pick_checkpoint(loc, device=str(device), probe_step=probe_step)
    drive_args = build_human_replay_drive_args(
        None, num_maps=1, device=str(device), data_mode="validation"
    )
    vec = create_vecenv(drive_args, env_name="puffer_drive")
    pol = create_policy(drive_args, vec, env_name="puffer_drive")
    load_policy_from_checkpoint(pol, ckpt.state_dict)
    pol.to(device).eval()
    return pol, vec, loc


def run_single_policy_attribution(
    *,
    alias: str,
    feature_ids: list[int],
    sae_ckpt: Path,
    sae_root: Path,
    probe_step: int,
    data_mode: str,
    base_path: str,
    device,
    pool_mode: str,
    max_rows: int,
    batch_size: int,
    row_mask: np.ndarray | None,
    policy_run_dir: str | None = None,
    skip_lstm: bool = False,
) -> dict:
    exp = resolve_model_name(alias)
    sae = SparseAutoencoder.load(sae_ckpt, map_location=device).to(device)
    bundle = load_obs_bundle(sae_root, probe_step, data_mode, exp)
    if bundle is None:
        raise FileNotFoundError("obs missing in activations.npz")

    winning_only = pool_mode == "winning_slot"
    actual_pool = "max" if winning_only else pool_mode

    policy, vecenv, run_used = _load_policy(
        base_path, exp, probe_step, device, run_dir=policy_run_dir
    )
    try:
        obs_level = projection_attribution_obs_level(
            policy=policy,
            sae=sae,
            obs=bundle["obs"],
            partner_slot=bundle["partner_slot"],
            x=bundle["x"],
            feature_ids=feature_ids,
            device=device,
            batch_size=batch_size,
            max_rows=max_rows,
            pool_mode=actual_pool,
            row_mask=row_mask,
            winning_slot_only=winning_only,
            skip_lstm=skip_lstm,
        )
        fd = finite_difference_sweep(
            policy=policy,
            sae=sae,
            obs=bundle["obs"],
            partner_slot=bundle["partner_slot"],
            x=bundle["x"],
            feature_ids=feature_ids,
            device=device,
            batch_size=batch_size,
            max_rows=max_rows,
            pool_mode=actual_pool,
            row_mask=row_mask,
            skip_lstm=skip_lstm,
        )
    finally:
        from sae_rollout import safe_close_vecenv  # noqa: WPS433

        safe_close_vecenv(vecenv)

    # Per-feature summaries for each variant/metric
    summaries = {}
    for variant in ATTR_VARIANTS:
        summaries[variant] = {}
        for metric_name, arr in obs_level["arrays"][variant].items():
            for j, fid in enumerate(feature_ids):
                summaries[variant][f"f{fid}_{metric_name}"] = summarize_obs_attributions(
                    arr[:, j]
                )

    feature_rows = []
    for j, fid in enumerate(feature_ids):
        row = {"feature_id": int(fid)}
        for variant in ATTR_VARIANTS:
            for metric_name in obs_level["arrays"][variant]:
                arr = obs_level["arrays"][variant][metric_name][:, j]
                s = summarize_obs_attributions(arr)
                row[f"{variant}_{metric_name}_median"] = s["median"]
                row[f"{variant}_{metric_name}_mean"] = s["mean"]
        feature_rows.append(row)

    obs_level_out = {
        "row_indices": obs_level["row_indices"].tolist(),
        "feature_scales": obs_level["feature_scales"],
    }
    for metric_name in EXTENDED_PRIMARY_METRICS:
        for variant in ATTR_VARIANTS:
            obs_level_out[f"{variant}_{metric_name}"] = obs_level["arrays"][variant][
                metric_name
            ].tolist()

    return {
        "alias": alias,
        "exp": exp,
        "policy_run_dir": run_used,
        "pool_mode": pool_mode,
        "n_rows": obs_level["n"],
        "feature_ids": feature_ids,
        "feature_summaries": summaries,
        "feature_rows": feature_rows,
        "obs_level": {
            **obs_level_out,
        },
        "fd_monotonicity": fd_monotonicity_report(fd),
        "fd": fd,
    }


def run_negative_controls(
    *,
    alias: str,
    sae_ckpt: Path,
    sae_root: Path,
    probe_step: int,
    data_mode: str,
    base_path: str,
    device,
    feature_ids: list[int],
    neg_features: dict[str, list[int]],
    max_rows: int,
    batch_size: int,
    row_mask: np.ndarray | None,
    policy_run_dir: str | None = None,
) -> dict:
    """Random direction + semantic negative features."""
    primary = run_single_policy_attribution(
        alias=alias,
        feature_ids=feature_ids,
        sae_ckpt=sae_ckpt,
        sae_root=sae_root,
        probe_step=probe_step,
        data_mode=data_mode,
        base_path=base_path,
        device=device,
        pool_mode="max",
        max_rows=max_rows,
        batch_size=batch_size,
        row_mask=row_mask,
        policy_run_dir=policy_run_dir,
    )
    neg: dict = {"primary_conflict": primary}

    for ctrl_name, fids in neg_features.items():
        if not fids:
            continue
        neg[ctrl_name] = run_single_policy_attribution(
            alias=alias,
            feature_ids=fids,
            sae_ckpt=sae_ckpt,
            sae_root=sae_root,
            probe_step=probe_step,
            data_mode=data_mode,
            base_path=base_path,
            device=device,
            pool_mode="max",
            max_rows=max_rows,
            batch_size=batch_size,
            row_mask=row_mask,
            policy_run_dir=policy_run_dir,
        )

    # Shuffled obs–feature pairing: permute attribution rows within feature
    raw = np.asarray(primary["obs_level"]["raw_attr_p_brake"], dtype=np.float64)
    if raw.ndim == 2 and raw.shape[0] > 1:
        rng = np.random.default_rng(42)
        shuffled = raw.copy()
        for j in range(shuffled.shape[1]):
            rng.shuffle(shuffled[:, j])
        neg["shuffled_pairing"] = summarize_obs_attributions(shuffled)
    return neg


def cross_policy_paired_stats(
    results_by_alias: dict[str, dict],
    matched_features: dict[str, list[int]],
    *,
    variant: str = "raw",
    metric: str = PRIMARY_METRIC,
) -> dict:
    """Paired obs-level ReCord vs Reactive for each matched feature rank."""
    if "record" not in results_by_alias or "reactive" not in results_by_alias:
        return {}

    rec_raw = np.asarray(
        results_by_alias["record"]["obs_level"][f"{variant}_{metric}"],
        dtype=np.float64,
    )
    rea_raw = np.asarray(
        results_by_alias["reactive"]["obs_level"][f"{variant}_{metric}"],
        dtype=np.float64,
    )
    n_feat = min(rec_raw.shape[1], rea_raw.shape[1], len(matched_features["record"]))
    per_feature = []
    all_deltas = []
    for j in range(n_feat):
        cmp = paired_policy_comparison(rec_raw[:, j], rea_raw[:, j])
        cmp["record_feature"] = matched_features["record"][j]
        cmp["reactive_feature"] = matched_features["reactive"][j]
        per_feature.append(cmp)
        if rec_raw.shape[0] == rea_raw.shape[0]:
            all_deltas.extend((rec_raw[:, j] - rea_raw[:, j]).tolist())

    pooled = {
        "n_paired": int(len(all_deltas)),
        "frac_record_gt_reactive": float(np.mean(np.asarray(all_deltas) > 0))
        if all_deltas
        else None,
        "median_delta": float(np.median(all_deltas)) if all_deltas else None,
        "mean_delta": float(np.mean(all_deltas)) if all_deltas else None,
        "delta_median_ci95": bootstrap_ci(np.asarray(all_deltas), statistic="median")
        if all_deltas
        else None,
        "wilcoxon_record_gt_reactive": wilcoxon_onesided_greater(np.asarray(all_deltas))
        if all_deltas
        else None,
        "note": "pooled over all matched features (ReCord−Reactive per obs×feature)",
    }
    return {
        "variant": variant,
        "metric": metric,
        "per_feature": per_feature,
        "pooled": pooled,
        "mean_median_attr": {
            a: float(
                np.median(
                    np.asarray(results_by_alias[a]["obs_level"][f"{variant}_{metric}"])
                )
            )
            for a in results_by_alias
        },
    }


def cross_metric_sign_consistency(
    results_by_alias: dict[str, dict],
    *,
    variant: str = "raw",
) -> dict:
    """Brake vs accel attribution should oppose at feature median (brake↑ ⇒ accel↓)."""
    rec = results_by_alias.get("record")
    if not rec:
        return {}
    obs = rec.get("obs_level") or {}
    brake_key = f"{variant}_attr_p_brake"
    accel_key = f"{variant}_attr_accel"
    if brake_key not in obs or accel_key not in obs:
        return {}
    brake = np.asarray(obs[brake_key], dtype=np.float64)
    accel = np.asarray(obs[accel_key], dtype=np.float64)
    if brake.ndim != 2 or accel.shape != brake.shape:
        return {}
    per_feature = []
    consistent = 0
    for j in range(brake.shape[1]):
        b_med = float(np.median(brake[:, j]))
        a_med = float(np.median(accel[:, j]))
        ok = (b_med == 0.0 and a_med == 0.0) or (np.sign(b_med) != np.sign(a_med))
        if ok:
            consistent += 1
        per_feature.append(
            {
                "feature_index": int(j),
                "median_attr_p_brake": b_med,
                "median_attr_accel": a_med,
                "sign_consistent": bool(ok),
            }
        )
    n_feat = brake.shape[1]
    return {
        "variant": variant,
        "n_features": int(n_feat),
        "frac_sign_consistent": float(consistent / n_feat) if n_feat else None,
        "per_feature": per_feature,
        "note": "median(attr_p_brake) and median(attr_accel) should have opposite signs",
    }


def aggregate_policy_seeds(
    seed_results: list[dict],
    *,
    metrics: tuple[str, ...] = EXTENDED_PRIMARY_METRICS,
) -> dict:
    """Mean ± std across policy training seeds."""
    if not seed_results:
        return {}
    aliases = seed_results[0].get("by_alias", {}).keys()
    out: dict = {}
    for metric in metrics:
        metric_out: dict = {}
        for alias in aliases:
            medians = []
            for run in seed_results:
                if alias not in run.get("by_alias", {}):
                    continue
                rows = run["by_alias"][alias]["feature_rows"]
                if not rows:
                    continue
                key = f"raw_{metric}_median"
                medians.append(float(np.mean([r.get(key, np.nan) for r in rows])))
            if medians:
                metric_out[alias] = {
                    "n_seeds": len(medians),
                    f"mean_median_{metric}": float(np.mean(medians)),
                    f"std_median_{metric}": float(np.std(medians)),
                    "per_seed": medians,
                }
        if metric_out:
            out[metric] = metric_out
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Projection attribution validation")
    p.add_argument("--analysis-dir", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--ckpt-name", type=str, default="sae_step_0001000.pt")
    p.add_argument("--sae-root", type=str, default="/data/puffer/sae")
    p.add_argument("--probe-step", type=int, default=1908)
    p.add_argument("--data-mode", type=str, default="validation")
    p.add_argument("--base-path", type=str, default="/data/puffer/experiments")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-enrichment", type=float, default=1.5)
    p.add_argument("--max-rows", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--conflict-only", action="store_true", default=True)
    p.add_argument("--no-conflict-only", action="store_true")
    p.add_argument(
        "--pool-modes",
        type=str,
        default="max,winning_slot,slot_only",
        help="comma-separated pool modes to compare",
    )
    p.add_argument("--skip-fd", action="store_true")
    p.add_argument("--skip-negative-controls", action="store_true")
    p.add_argument(
        "--policy-run-dirs",
        type=str,
        default=None,
        help="override policy ckpt paths: record=/path/puffer_drive_id.pt;reactive=path;selfplay=path",
    )
    args = p.parse_args()

    import torch

    analysis_dir = Path(args.analysis_dir)
    run_dir = Path(args.run_dir)
    semantics_dir = analysis_dir / "semantics"
    matching_dir = analysis_dir / "matching"
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "attribution_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    pool_modes = [m.strip() for m in args.pool_modes.split(",") if m.strip()]
    conflict_only = not args.no_conflict_only

    row_mask = None
    if conflict_only:
        from feature_steering import conflict_mask_from_activations  # noqa: WPS433

        row_mask = conflict_mask_from_activations(
            Path(args.sae_root), args.probe_step, args.data_mode
        )
        print(f"conflict-only: {int(row_mask.sum()) if row_mask is not None else 0} rows")

    matched = pick_blind_conflict_features(
        analysis_dir,
        semantics_dir,
        matching_dir,
        min_enrichment=args.min_enrichment,
        top_k=args.top_k,
    )
    print("Blind matched conflict features:", matched)

    neg_feats = pick_negative_control_features(analysis_dir, semantics_dir)
    ckpts = discover_sae_ckpts(run_dir, args.ckpt_name)

    # Optional per-alias policy run dirs (PBT seeds)
    policy_run_dirs: dict[str, str | None] = {a: None for a in MODELS}
    if args.policy_run_dirs:
        for part in args.policy_run_dirs.split(";"):
            if not part.strip() or "=" not in part:
                continue
            alias, path = part.split("=", 1)
            policy_run_dirs[alias.strip()] = path.strip()

    all_seed_results: list[dict] = []
    full_report: dict = {
        "matched_features": matched,
        "pool_modes": pool_modes,
        "conflict_only": conflict_only,
        "policy_run_dirs": {k: v for k, v in policy_run_dirs.items() if v},
        "note": (
            "Features selected blind (matched triples + enrichment) before attribution. "
            "Policy-seed variance requires separate activation/SAE runs per seed; "
            "use run_attribution_policy_seeds.sh for full replication."
        ),
        "runs": [],
    }

    print("\n======== Attribution validation ========")
    run_entry: dict = {"pool_comparisons": {}, "by_alias": {}}

    for pool_mode in pool_modes:
        print(f"\n--- pool_mode={pool_mode} ---")
        by_alias = {}
        for alias in MODELS:
            if alias not in ckpts:
                continue
            fids = matched.get(alias) or []
            if not fids:
                continue
            pol_run = policy_run_dirs.get(alias)
            print(f"  {PRETTY[alias]} features={fids} policy_run={pol_run or 'default'}")
            try:
                res = run_single_policy_attribution(
                    alias=alias,
                    feature_ids=fids,
                    sae_ckpt=ckpts[alias],
                    sae_root=Path(args.sae_root),
                    probe_step=args.probe_step,
                    data_mode=args.data_mode,
                    base_path=args.base_path,
                    device=device,
                    pool_mode=pool_mode,
                    max_rows=args.max_rows,
                    batch_size=args.batch_size,
                    row_mask=row_mask,
                    policy_run_dir=pol_run,
                )
            except FileNotFoundError as exc:
                msg = str(exc)
                if "No checkpoint step" in msg or msg.startswith("skip:"):
                    print(f"  SKIP {PRETTY[alias]}: {exc}")
                    continue
                raise
            if args.skip_fd:
                res.pop("fd", None)
            by_alias[alias] = res

        run_entry["pool_comparisons"][pool_mode] = {
            "by_alias": {a: by_alias[a] for a in by_alias},
            "paired": {
                v: {
                    metric: cross_policy_paired_stats(
                        by_alias, matched, variant=v, metric=metric
                    )
                    for metric in EXTENDED_PRIMARY_METRICS
                }
                for v in ATTR_VARIANTS
            },
            "cross_metric_consistency": {
                v: cross_metric_sign_consistency(by_alias, variant=v) for v in ATTR_VARIANTS
            },
        }
        if pool_mode == pool_modes[0]:
            run_entry["by_alias"] = by_alias

    if not args.skip_negative_controls and "record" in ckpts:
        run_entry["negative_controls_record"] = run_negative_controls(
            alias="record",
            sae_ckpt=ckpts["record"],
            sae_root=Path(args.sae_root),
            probe_step=args.probe_step,
            data_mode=args.data_mode,
            base_path=args.base_path,
            device=device,
            feature_ids=matched.get("record", []),
            neg_features=neg_feats.get("record", {}),
            max_rows=args.max_rows,
            batch_size=args.batch_size,
            row_mask=row_mask,
            policy_run_dir=policy_run_dirs.get("record"),
        )

    all_seed_results.append(run_entry)
    full_report["runs"].append(run_entry)

    primary = run_entry["pool_comparisons"].get(pool_modes[0], {})
    paired_raw = (primary.get("paired", {}).get("raw") or {}).get(PRIMARY_METRIC) or {}
    if paired_raw:
        p = paired_raw.get("pooled", {})
        print(
            f"\n  Paired ReCord>Reactive (raw brake attr): "
            f"frac={p.get('frac_record_gt_reactive')} "
            f"medianΔ={p.get('median_delta')} "
            f"p={p.get('wilcoxon_record_gt_reactive', {}).get('pvalue')}"
        )
        print(f"  Mean median attr by policy: {paired_raw.get('mean_median_attr')}")
    for metric in EXTENDED_PRIMARY_METRICS:
        if metric == PRIMARY_METRIC:
            continue
        block = (primary.get("paired", {}).get("raw") or {}).get(metric) or {}
        p = block.get("pooled") or {}
        if not p:
            continue
        print(
            f"  Paired ReCord>Reactive (raw {metric}): "
            f"frac={p.get('frac_record_gt_reactive')} "
            f"medianΔ={p.get('median_delta')}"
        )
    cm = (primary.get("cross_metric_consistency") or {}).get("raw") or {}
    if cm:
        print(
            f"  Brake/accel sign consistency (ReCord): "
            f"frac={cm.get('frac_sign_consistent')}"
        )

    full_report["policy_seed_aggregate"] = aggregate_policy_seeds(all_seed_results)

    out_path = out_dir / "attribution_validation.json"
    # Drop bulky per-obs arrays before writing full JSON
    for run in full_report.get("runs", []):
        for pm, block in run.get("pool_comparisons", {}).items():
            for alias, res in block.get("by_alias", {}).items():
                res.pop("obs_level", None)
                res.pop("fd", None)
        for alias, res in run.get("by_alias", {}).items():
            res.pop("obs_level", None)
            res.pop("fd", None)
        nc = run.get("negative_controls_record") or {}
        for name, block in list(nc.items()):
            if isinstance(block, dict):
                block.pop("obs_level", None)
                block.pop("fd", None)

    out_path.write_text(json.dumps(full_report, indent=2))
    print(f"\nWrote {out_path}")

    # Compact headline summary for paper tables
    slim = {
        "matched_features": full_report["matched_features"],
        "pool_modes": pool_modes,
        "conflict_only": conflict_only,
        "headline": {},
    }
    seed_info_path = out_dir.parent / "seed_info.json"
    if seed_info_path.is_file():
        slim["seed_info"] = json.loads(seed_info_path.read_text())
    for pm, block in full_report["runs"][0]["pool_comparisons"].items():
        slim["headline"][pm] = {}
        for v in ATTR_VARIANTS:
            paired_v = block.get("paired", {}).get(v) or {}
            primary_block = paired_v.get(PRIMARY_METRIC) or {}
            pooled = primary_block.get("pooled") or {}
            slim["headline"][pm][v] = {
                "frac_record_gt_reactive": pooled.get("frac_record_gt_reactive"),
                "median_delta": pooled.get("median_delta"),
                "wilcoxon_p": (pooled.get("wilcoxon_record_gt_reactive") or {}).get(
                    "pvalue"
                ),
                "median_attr_by_policy": primary_block.get("mean_median_attr"),
                "by_metric": {
                    metric: {
                        "frac_record_gt_reactive": (
                            (paired_v.get(metric) or {}).get("pooled") or {}
                        ).get("frac_record_gt_reactive"),
                        "median_delta": (
                            (paired_v.get(metric) or {}).get("pooled") or {}
                        ).get("median_delta"),
                        "wilcoxon_p": (
                            ((paired_v.get(metric) or {}).get("pooled") or {})
                            .get("wilcoxon_record_gt_reactive", {})
                            .get("pvalue")
                        ),
                        "median_attr_by_policy": (paired_v.get(metric) or {}).get(
                            "mean_median_attr"
                        ),
                    }
                    for metric in EXTENDED_PRIMARY_METRICS
                },
                "cross_metric_consistency": (
                    block.get("cross_metric_consistency") or {}
                ).get(v),
            }
    slim_path = out_dir / "attribution_validation_summary.json"
    slim_path.write_text(json.dumps(slim, indent=2))
    print(f"Wrote {slim_path}")


if __name__ == "__main__":
    main()
