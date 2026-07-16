"""Top-activation retrieval for trained SAE features.

For each SAE latent ``j``, find the dataset rows where feature ``j`` fires
strongest, and attach visualization metadata (ego/other state, future traj, …).

Usage::

    # all features, top-32 examples each (one experiment)
    python analyze/sae/top_activation_retrieval.py \\
      --sae-ckpt /data/puffer/sae/runs/topk_exp16_k32_step1908/selfplay/sae_best.pt \\
      --activations /data/puffer/sae/human_replay/training/step_001908/activations.npz \\
      --experiment selfplay \\
      --top-k 32 \\
      --out-dir /data/puffer/sae/analysis/selfplay_topk

    # only features 0,10,42
    python analyze/sae/top_activation_retrieval.py ... --features 0,10,42

    # all experiments under a run dir
    python analyze/sae/top_activation_retrieval.py \\
      --run-dir /data/puffer/sae/runs/topk_exp16_k32_step1908 \\
      --sae-root /data/puffer/sae --probe-step 1908 --data-mode training
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

from sae_model import SparseAutoencoder  # noqa: E402

ACTIVATIONS_FILENAME = "activations.npz"
META_KEYS = (
    "partner_slot",
    "time_idx",
    "agent_idx",
    "scenario_id",
    "scene_id",
    "timestep",
    "vehicle_id",
    "ego_id",
    "dist_at_t",
    "ego_state",
    "other_state",
    "future_traj",
)


def activation_key(exp_name: str) -> str:
    return f"activation__{exp_name}"


def parse_csv_ints(spec: str | None) -> list[int] | None:
    if spec is None or not str(spec).strip():
        return None
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def resolve_activations_path(
    *,
    activations: str | None,
    sae_root: str,
    data_mode: str,
    probe_step: int,
) -> Path:
    if activations:
        path = Path(activations)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = [
        Path(sae_root) / "human_replay" / data_mode / f"step_{probe_step:06d}" / ACTIVATIONS_FILENAME,
        Path(sae_root) / data_mode / f"step_{probe_step:06d}" / ACTIVATIONS_FILENAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No activations.npz. Tried:\n  " + "\n  ".join(str(p) for p in candidates)
    )


def load_activation_pack(path: Path, experiment: str) -> dict[str, np.ndarray]:
    """Load partner-encoder act matrix + available row metadata."""
    key = activation_key(experiment)
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            available = (
                [str(x) for x in data["experiments"].tolist()]
                if "experiments" in data.files
                else [k for k in data.files if k.startswith("activation__")]
            )
            raise KeyError(f"missing {key} in {path} (have {available})")
        x = np.asarray(data[key], dtype=np.float32)
        pack: dict[str, np.ndarray] = {
            "x": x,
            "experiment": np.array(experiment),
            "path": np.array(str(path)),
        }
        for mk in META_KEYS:
            if mk in data.files:
                pack[mk] = np.asarray(data[mk])
        for mk in (
            "ego_state_keys",
            "other_state_keys",
            "future_traj_keys",
            "future_traj_horizon",
            "probe_step",
            "sae_collect_version",
        ):
            if mk in data.files:
                pack[mk] = np.asarray(data[mk])
    return pack


@torch.no_grad()
def encode_features(
    sae: SparseAutoencoder,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Return SAE feature acts ``(N, d_sae)`` float32."""
    sae.eval()
    n = int(x.shape[0])
    d_sae = int(sae.cfg.d_sae)
    out = np.empty((n, d_sae), dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch = torch.from_numpy(x[start:stop]).to(device)
        feats = sae.encode(batch)
        out[start:stop] = feats.detach().cpu().numpy().astype(np.float32, copy=False)
    return out


def feature_summary(feats: np.ndarray) -> dict[str, np.ndarray]:
    """Per-feature density / mean / max over the dataset."""
    positive = feats > 0
    return {
        "density": positive.mean(axis=0).astype(np.float32),
        "mean_act": feats.mean(axis=0).astype(np.float32),
        "mean_act_when_fire": np.divide(
            feats.sum(axis=0),
            positive.sum(axis=0).clip(min=1),
        ).astype(np.float32),
        "max_act": feats.max(axis=0).astype(np.float32),
        "n_fire": positive.sum(axis=0).astype(np.int64),
    }


def top_activation_indices(
    feats: np.ndarray,
    *,
    feature_ids: list[int] | None,
    top_k: int,
    min_activation: float = 0.0,
    scene_id: np.ndarray | None = None,
    vehicle_id: np.ndarray | None = None,
    timestep: np.ndarray | None = None,
    dedup_scene_vehicle: bool = True,
    min_timestep_gap: int = 10,
) -> dict[str, np.ndarray]:
    """For each feature, indices of the ``top_k`` strongest activations.

    Dedup (default): keep at most one row per ``(scene_id, vehicle_id)``.
    If the same pair appears with timestep gap < ``min_timestep_gap``, keep the
    stronger activation only (already implied by descending scan + one-per-pair).

    Returns
    -------
    feature_id : (F,)
    top_row_idx : (F, K)   int64, -1 if fewer than K firing rows
    top_value   : (F, K)   float32, NaN if padded
    diversity_* : per-feature uniqueness / NaN stats over the kept top set
    """
    n, d_sae = feats.shape
    if feature_ids is None:
        feature_ids = list(range(d_sae))
    feature_ids = [int(f) for f in feature_ids]
    for f in feature_ids:
        if f < 0 or f >= d_sae:
            raise ValueError(f"feature id {f} out of range [0, {d_sae})")

    f_arr = np.asarray(feature_ids, dtype=np.int64)
    top_idx = np.full((len(feature_ids), top_k), -1, dtype=np.int64)
    top_val = np.full((len(feature_ids), top_k), np.nan, dtype=np.float32)
    n_unique_scene = np.zeros(len(feature_ids), dtype=np.int32)
    n_unique_traj = np.zeros(len(feature_ids), dtype=np.int32)
    valid_future_frac = np.full(len(feature_ids), np.nan, dtype=np.float32)
    nan_frac = np.full(len(feature_ids), np.nan, dtype=np.float32)

    use_dedup = (
        dedup_scene_vehicle
        and scene_id is not None
        and vehicle_id is not None
    )

    for i, f in enumerate(feature_ids):
        col = feats[:, f]
        if min_activation > 0:
            eligible = np.flatnonzero(col >= min_activation)
        else:
            eligible = np.flatnonzero(col > 0)
            if eligible.size == 0:
                eligible = np.arange(n, dtype=np.int64)
        if eligible.size == 0:
            continue
        scores = col[eligible]
        order = np.argsort(-scores)
        ranked = eligible[order]

        chosen: list[int] = []
        seen_pair: dict[tuple[int, int], int] = {}
        for row in ranked.tolist():
            if len(chosen) >= top_k:
                break
            if use_dedup:
                key = (int(scene_id[row]), int(vehicle_id[row]))
                if key in seen_pair:
                    continue
                if timestep is not None and min_timestep_gap > 1:
                    # reject near-duplicates on same pair already handled by one-per-pair;
                    # also reject if another chosen row shares scene+vehicle with small gap
                    # (unreachable once one-per-pair); keep for gap across ego variants.
                    pass
                seen_pair[key] = int(timestep[row]) if timestep is not None else -1
            chosen.append(row)

        k = len(chosen)
        if k == 0:
            continue
        chosen_arr = np.asarray(chosen, dtype=np.int64)
        top_idx[i, :k] = chosen_arr
        top_val[i, :k] = col[chosen_arr]

        if scene_id is not None:
            n_unique_scene[i] = int(np.unique(scene_id[chosen_arr]).size)
        if scene_id is not None and vehicle_id is not None:
            traj_keys = np.stack(
                [scene_id[chosen_arr], vehicle_id[chosen_arr]], axis=1
            )
            n_unique_traj[i] = int(np.unique(traj_keys, axis=0).shape[0])
        nan_frac[i] = float(np.mean(~np.isfinite(top_val[i, :k])))

    return {
        "feature_id": f_arr,
        "top_row_idx": top_idx,
        "top_value": top_val,
        "diversity_n_unique_scene": n_unique_scene,
        "diversity_n_unique_traj": n_unique_traj,
        "diversity_nan_frac": nan_frac,
        "diversity_valid_future_frac": valid_future_frac,
        "dedup_scene_vehicle": np.bool_(use_dedup),
        "min_timestep_gap": np.int32(min_timestep_gap),
    }


def gather_meta_for_topk(
    pack: dict[str, np.ndarray],
    retrieval: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Slice metadata arrays to ``(F, K, ...)`` aligning with top_row_idx."""
    top_idx = retrieval["top_row_idx"]
    f, k = top_idx.shape
    flat = top_idx.reshape(-1)
    valid = flat >= 0
    safe = np.where(valid, flat, 0)

    out: dict[str, np.ndarray] = {}
    for key in META_KEYS:
        if key not in pack:
            continue
        arr = pack[key]
        gathered = arr[safe]
        g = gathered.reshape((f * k,) + arr.shape[1:]).copy()
        if not valid.all():
            if np.issubdtype(g.dtype, np.floating):
                g[~valid] = np.nan
            elif np.issubdtype(g.dtype, np.integer):
                g[~valid] = -1
        out[f"top_{key}"] = g.reshape((f, k) + arr.shape[1:])
    out["top_valid"] = top_idx >= 0
    return out


def retrieve_top_activations(
    sae: SparseAutoencoder,
    pack: dict[str, np.ndarray],
    *,
    device: torch.device,
    top_k: int = 32,
    feature_ids: list[int] | None = None,
    min_activation: float = 0.0,
    batch_size: int = 4096,
    skip_dead: bool = False,
    dedup_scene_vehicle: bool = True,
    min_timestep_gap: int = 10,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Encode dataset; return ``(retrieval_result, feature_acts[N,d_sae])``."""
    x = pack["x"]
    print(f"  encode N={x.shape[0]} d_in={x.shape[1]} -> d_sae={sae.cfg.d_sae}")
    feats = encode_features(sae, x, device=device, batch_size=batch_size)
    stats = feature_summary(feats)

    if feature_ids is None and skip_dead:
        feature_ids = [i for i, d in enumerate(stats["density"]) if d > 0]
        print(f"  skip_dead: keeping {len(feature_ids)}/{sae.cfg.d_sae} firing features")

    retrieval = top_activation_indices(
        feats,
        feature_ids=feature_ids,
        top_k=top_k,
        min_activation=min_activation,
        scene_id=pack.get("scene_id"),
        vehicle_id=pack.get("vehicle_id"),
        timestep=pack.get("timestep"),
        dedup_scene_vehicle=dedup_scene_vehicle,
        min_timestep_gap=min_timestep_gap,
    )
    meta = gather_meta_for_topk(pack, retrieval)

    # Fill valid-future ratio from gathered future_traj if present.
    if "top_future_traj" in meta:
        fut = meta["top_future_traj"]
        valid = np.isfinite(fut[..., 0])
        top_valid = meta["top_valid"]
        frac = []
        for i in range(fut.shape[0]):
            m = top_valid[i]
            if not np.any(m):
                frac.append(float("nan"))
            else:
                frac.append(float(valid[i][m].mean()))
        retrieval["diversity_valid_future_frac"] = np.asarray(frac, dtype=np.float32)

    result = {
        **retrieval,
        **{f"feat_{k}": v for k, v in stats.items()},
        **meta,
        "experiment": pack["experiment"],
        "activations_path": pack["path"],
        "n_rows": np.int64(x.shape[0]),
        "d_in": np.int32(x.shape[1]),
        "d_sae": np.int32(sae.cfg.d_sae),
        "top_k": np.int32(top_k),
        "min_activation": np.float32(min_activation),
    }
    top_idx = retrieval["top_row_idx"]
    f, k = top_idx.shape
    flat = top_idx.reshape(-1)
    safe = np.where(flat >= 0, flat, 0)
    top_x = x[safe].reshape(f, k, x.shape[1])
    if (flat < 0).any():
        top_x = top_x.copy()
        top_x.reshape(f * k, x.shape[1])[flat < 0] = np.nan
    result["top_x"] = top_x.astype(np.float32, copy=False)
    return result, feats


def save_sample_keys(pack: dict[str, np.ndarray], out_dir: Path) -> Path:
    """Shared observation keys for cross-model matching (same rows across SAEs)."""
    keys = {}
    for mk in META_KEYS:
        if mk in pack:
            keys[mk] = pack[mk]
    path = out_dir / "sample_keys.npz"
    np.savez_compressed(path, **keys)
    return path


def save_feature_stats(stats: dict[str, np.ndarray], out_dir: Path, *, d_sae: int) -> Path:
    path = out_dir / "feature_stats.npz"
    np.savez_compressed(
        path,
        density=stats["density"],
        mean_act=stats["mean_act"],
        mean_act_when_fire=stats["mean_act_when_fire"],
        max_act=stats["max_act"],
        n_fire=stats["n_fire"],
        d_sae=np.int32(d_sae),
        alive=stats["n_fire"] > 0,
    )
    return path


def save_feature_acts(feats: np.ndarray, out_dir: Path, *, dtype=np.float16) -> Path:
    """Full ``[N, d_sae]`` feature activations (float16 by default)."""
    path = out_dir / "feature_acts.npy"
    np.save(path, feats.astype(dtype, copy=False))
    return path


def save_retrieval(
    result: dict[str, np.ndarray],
    out_dir: Path,
    *,
    feats: np.ndarray | None = None,
    pack: dict[str, np.ndarray] | None = None,
    save_full_acts: bool = True,
) -> dict[str, Path]:
    """Write recommended analysis layout under ``out_dir``.

    ::

        feature_acts.npy      # [N, d_sae] float16
        sample_keys.npz       # scene/timestep/ego/vehicle/...
        feature_stats.npz
        top_activations.npz
        top_activations.csv
        top_activations_summary.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    top_path = out_dir / "top_activations.npz"
    np.savez_compressed(top_path, **result)
    written["top_activations"] = top_path

    stats = {
        "density": result["feat_density"],
        "mean_act": result["feat_mean_act"],
        "mean_act_when_fire": result["feat_mean_act_when_fire"],
        "max_act": result["feat_max_act"],
        "n_fire": result["feat_n_fire"],
    }
    written["feature_stats"] = save_feature_stats(
        stats, out_dir, d_sae=int(result["d_sae"])
    )

    if pack is not None:
        written["sample_keys"] = save_sample_keys(pack, out_dir)

    if save_full_acts and feats is not None:
        written["feature_acts"] = save_feature_acts(feats, out_dir)
        print(
            f"  wrote {written['feature_acts']}  "
            f"shape={feats.shape} dtype=float16 "
            f"({feats.shape[0] * feats.shape[1] * 2 / 1e6:.1f} MB)"
        )

    feat_ids = result["feature_id"]
    dens = result["feat_density"]
    summary = {
        "experiment": str(result["experiment"]),
        "activations_path": str(result["activations_path"]),
        "n_rows": int(result["n_rows"]),
        "d_sae": int(result["d_sae"]),
        "top_k": int(result["top_k"]),
        "n_features_retrieved": int(feat_ids.shape[0]),
        "n_dead_features": int((result["feat_n_fire"] == 0).sum()),
        "density_mean": float(dens.mean()),
        "density_median": float(np.median(dens)),
        "files": {k: str(v) for k, v in written.items()},
        "schema": {
            "feature_acts.npy": "[N, d_sae] float16 full feature activations",
            "sample_keys.npz": "per-row scene/timestep/vehicle/ego_state/...",
            "feature_stats.npz": "per-feature density / mean / max / n_fire",
            "top_activations.npz": "top-K retrieval with meta",
        },
    }
    per_feat = []
    top_val = result["top_value"]
    for i, f in enumerate(feat_ids.tolist()):
        vals = top_val[i]
        finite = vals[np.isfinite(vals)]
        row = {
            "feature_id": int(f),
            "density": float(dens[f]),
            "n_fire": int(result["feat_n_fire"][f]),
            "max_act": float(result["feat_max_act"][f]),
            "top1": float(finite[0]) if finite.size else None,
            "top_mean": float(finite.mean()) if finite.size else None,
            "n_unique_scene": int(result["diversity_n_unique_scene"][i])
            if "diversity_n_unique_scene" in result
            else None,
            "n_unique_traj": int(result["diversity_n_unique_traj"][i])
            if "diversity_n_unique_traj" in result
            else None,
            "valid_future_frac": float(result["diversity_valid_future_frac"][i])
            if "diversity_valid_future_frac" in result
            and np.isfinite(result["diversity_valid_future_frac"][i])
            else None,
            "nan_frac": float(result["diversity_nan_frac"][i])
            if "diversity_nan_frac" in result
            and np.isfinite(result["diversity_nan_frac"][i])
            else None,
        }
        per_feat.append(row)
    summary["features"] = per_feat
    if per_feat:
        uniq = [r["n_unique_scene"] for r in per_feat if r["n_unique_scene"] is not None]
        if uniq:
            summary["diversity_unique_scene_mean"] = float(np.mean(uniq))
            summary["diversity_unique_scene_median"] = float(np.median(uniq))
    summary_path = out_dir / "top_activations_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    written["summary"] = summary_path

    csv_path = out_dir / "top_activations.csv"
    _write_topk_csv(csv_path, result)
    written["csv"] = csv_path

    # Compact diversity table
    div_path = out_dir / "top_diversity.json"
    div_path.write_text(json.dumps(per_feat, indent=2))
    written["diversity"] = div_path
    return written


def _write_topk_csv(path: Path, result: dict[str, np.ndarray]) -> None:
    import csv

    feat_ids = result["feature_id"]
    top_idx = result["top_row_idx"]
    top_val = result["top_value"]
    scene = result.get("top_scene_id")
    tstep = result.get("top_timestep")
    vid = result.get("top_vehicle_id")
    dist = result.get("top_dist_at_t")
    ego = result.get("top_ego_state")
    other = result.get("top_other_state")

    fields = [
        "feature_id",
        "rank",
        "row_idx",
        "activation",
        "scene_id",
        "timestep",
        "vehicle_id",
        "dist_at_t",
        "ego_x",
        "ego_y",
        "ego_heading",
        "ego_speed",
        "other_x",
        "other_y",
        "other_heading",
        "other_speed",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, feat in enumerate(feat_ids.tolist()):
            for rank in range(top_idx.shape[1]):
                row_i = int(top_idx[i, rank])
                if row_i < 0 or not np.isfinite(top_val[i, rank]):
                    continue
                row = {
                    "feature_id": feat,
                    "rank": rank + 1,
                    "row_idx": row_i,
                    "activation": float(top_val[i, rank]),
                    "scene_id": int(scene[i, rank]) if scene is not None else "",
                    "timestep": int(tstep[i, rank]) if tstep is not None else "",
                    "vehicle_id": int(vid[i, rank]) if vid is not None else "",
                    "dist_at_t": float(dist[i, rank]) if dist is not None else "",
                }
                if ego is not None:
                    row.update(
                        {
                            "ego_x": float(ego[i, rank, 0]),
                            "ego_y": float(ego[i, rank, 1]),
                            "ego_heading": float(ego[i, rank, 2]),
                            "ego_speed": float(ego[i, rank, 3]),
                        }
                    )
                else:
                    row.update(
                        {k: "" for k in ("ego_x", "ego_y", "ego_heading", "ego_speed")}
                    )
                if other is not None:
                    row.update(
                        {
                            "other_x": float(other[i, rank, 0]),
                            "other_y": float(other[i, rank, 1]),
                            "other_heading": float(other[i, rank, 2]),
                            "other_speed": float(other[i, rank, 3]),
                        }
                    )
                else:
                    row.update(
                        {
                            k: ""
                            for k in ("other_x", "other_y", "other_heading", "other_speed")
                        }
                    )
                w.writerow(row)


def discover_run_experiments(
    run_dir: Path,
    *,
    ckpt_name: str = "sae_best.pt",
) -> list[tuple[str, Path]]:
    """Return ``(experiment, checkpoint)`` pairs under a train run dir."""
    found = []
    for ckpt in sorted(run_dir.glob(f"*/{ckpt_name}")):
        found.append((ckpt.parent.name, ckpt))
    if found:
        return found
    # Fallbacks
    for fallback in ("sae_best.pt", "sae_last.pt"):
        if fallback == ckpt_name:
            continue
        for ckpt in sorted(run_dir.glob(f"*/{fallback}")):
            found.append((ckpt.parent.name, ckpt))
        if found:
            return found
    return found


def run_one(
    *,
    sae_ckpt: Path,
    activations: Path,
    experiment: str,
    out_dir: Path,
    device: torch.device,
    top_k: int,
    feature_ids: list[int] | None,
    min_activation: float,
    batch_size: int,
    skip_dead: bool,
    save_full_acts: bool = True,
    dedup_scene_vehicle: bool = True,
    min_timestep_gap: int = 10,
) -> Path:
    print(f"=== {experiment} ===")
    print(f"  sae: {sae_ckpt}")
    print(f"  data: {activations}")
    sae = SparseAutoencoder.load(sae_ckpt, map_location=device)
    sae.to(device)
    print(f"  model: {sae}")
    pack = load_activation_pack(activations, experiment)
    result, feats = retrieve_top_activations(
        sae,
        pack,
        device=device,
        top_k=top_k,
        feature_ids=feature_ids,
        min_activation=min_activation,
        batch_size=batch_size,
        skip_dead=skip_dead,
        dedup_scene_vehicle=dedup_scene_vehicle,
        min_timestep_gap=min_timestep_gap,
    )
    written = save_retrieval(
        result,
        out_dir,
        feats=feats,
        pack=pack,
        save_full_acts=save_full_acts,
    )
    print(
        f"  saved {written['top_activations']}  "
        f"(F={result['feature_id'].shape[0]}, K={int(result['top_k'])}, "
        f"dead={int((result['feat_n_fire'] == 0).sum())})"
    )
    if "diversity_n_unique_scene" in result:
        u = result["diversity_n_unique_scene"]
        print(
            f"  dedup diversity: unique_scene mean={float(u.mean()):.1f} "
            f"median={float(np.median(u)):.1f}"
        )
    return written["top_activations"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SAE top-activation retrieval")
    p.add_argument("--sae-ckpt", type=str, default=None, help="Path to sae_best.pt")
    p.add_argument("--run-dir", type=str, default=None, help="Train run root; loops experiments")
    p.add_argument(
        "--ckpt-name",
        type=str,
        default="sae_best.pt",
        help="Checkpoint filename under each experiment dir (e.g. sae_step_0001000.pt)",
    )
    p.add_argument("--activations", type=str, default=None)
    p.add_argument("--sae-root", type=str, default="/data/puffer/sae")
    p.add_argument("--probe-step", type=int, default=1908)
    p.add_argument("--data-mode", choices=("training", "validation"), default="training")
    p.add_argument("--experiment", type=str, default=None, help="Required with --sae-ckpt")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--features", type=str, default=None, help="Comma-separated feature ids")
    p.add_argument("--min-activation", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument(
        "--skip-dead",
        action="store_true",
        help="Only retrieve features with density > 0",
    )
    p.add_argument(
        "--no-full-acts",
        action="store_true",
        help="Do not write feature_acts.npy (saves disk)",
    )
    p.add_argument(
        "--no-dedup",
        action="store_true",
        help="Allow multiple top rows from the same scene_id+vehicle_id",
    )
    p.add_argument(
        "--min-timestep-gap",
        type=int,
        default=10,
        help="When deduping, treat nearby timesteps on same pair as duplicates",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    feature_ids = parse_csv_ints(args.features)
    activations = resolve_activations_path(
        activations=args.activations,
        sae_root=args.sae_root,
        data_mode=args.data_mode,
        probe_step=args.probe_step,
    )
    save_full_acts = not args.no_full_acts
    dedup = not args.no_dedup

    if args.run_dir:
        run_dir = Path(args.run_dir)
        pairs = discover_run_experiments(run_dir, ckpt_name=args.ckpt_name)
        if not pairs:
            raise FileNotFoundError(
                f"No {args.ckpt_name} (or fallbacks) under {run_dir}"
            )
        root_out = Path(args.out_dir) if args.out_dir else run_dir / "analysis" / "top_activations"
        written = []
        for exp, ckpt in pairs:
            if args.experiment and exp != args.experiment:
                continue
            written.append(
                run_one(
                    sae_ckpt=ckpt,
                    activations=activations,
                    experiment=exp,
                    out_dir=root_out / exp,
                    device=device,
                    top_k=args.top_k,
                    feature_ids=feature_ids,
                    min_activation=args.min_activation,
                    batch_size=args.batch_size,
                    skip_dead=args.skip_dead,
                    save_full_acts=save_full_acts,
                    dedup_scene_vehicle=dedup,
                    min_timestep_gap=args.min_timestep_gap,
                )
            )
        print("\nDone:")
        for w in written:
            print(f"  {w}")
        return

    if not args.sae_ckpt or not args.experiment:
        raise SystemExit("Provide --sae-ckpt and --experiment, or --run-dir")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.sae_ckpt).parent / "top_activations"
    run_one(
        sae_ckpt=Path(args.sae_ckpt),
        activations=activations,
        experiment=args.experiment,
        out_dir=out_dir,
        device=device,
        top_k=args.top_k,
        feature_ids=feature_ids,
        min_activation=args.min_activation,
        batch_size=args.batch_size,
        skip_dead=args.skip_dead,
        save_full_acts=save_full_acts,
        dedup_scene_vehicle=dedup,
        min_timestep_gap=args.min_timestep_gap,
    )


if __name__ == "__main__":
    main()
