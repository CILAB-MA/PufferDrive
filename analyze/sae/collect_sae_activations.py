"""Collect shared human-replay partner_encoder activations for SAE training.

One human-replay rollout defines shared obs; then each of
``selfplay``, ``reactive_0.25``, ``replay_0.25`` encodes the same
(obs, partner_slot) rows. Writes ``activations.npz`` with activations **and**
per-row visualization metadata (no LP ``future_*.npz`` labels).

Layout::

    <sae_root>/human_replay/{training,validation}/step_XXXXXX/
      activations.npz
        activation__selfplay           (N, d)
        activation__reactive_0.25
        activation__replay_0.25
        scene_id / timestep / vehicle_id / ego_id
        ego_state                      (N, 4) = x, y, heading, speed
        other_state                    (N, 4) = x, y, heading, speed
        future_traj                    (N, H, 4) partner traj at t..t+H-1
        partner_slot, agent_idx, dist_at_t, ...

Usage::

    ./analyze/sae/run_onpolicy_sae_pipeline.sh
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

_SAE_DIR = os.path.dirname(os.path.abspath(__file__))
if _SAE_DIR not in sys.path:
    sys.path.insert(0, _SAE_DIR)

from partner_features import batch_partner_context  # noqa: E402
from sae_rollout import (  # noqa: E402
    DEFAULT_EXPERIMENTS,
    REPR_LAYER,
    ROLLOUT_MODE,
    build_human_replay_drive_args,
    collect_selfplay_rollout,
    create_policy,
    create_vecenv,
    load_policy_from_checkpoint,
    parse_csv_list,
    partner_dist_at_t_from_obs_batch,
    partner_vehicle_size_mask,
    pick_checkpoint,
    resolve_output_dir,
    resolve_run_dir,
    safe_close_vecenv,
    save_npz_atomic,
    save_scene_context,
)

SAE_COLLECT_VERSION = 6
ACTIVATIONS_FILENAME = "activations.npz"
# Future partner trajectory window stored for visualization (steps after t, inclusive of t).
FUTURE_TRAJ_HORIZON = 20
# Prefer scene diversity over dense adjacent timesteps (same ego–partner track).
DEFAULT_MAX_TIMESTEPS_PER_PAIR = 4
DEFAULT_MIN_TIMESTEP_GAP = 8
DEFAULT_MAX_SAMPLES_PER_SCENE = 64
# Matches drive.h MAX_SPEED (obs stores signed_speed / MAX_SPEED).
OBS_MAX_SPEED_M_S = 100.0
EGO_STATE_KEYS = ("x", "y", "heading", "speed")
OTHER_STATE_KEYS = ("x", "y", "heading", "speed")
META_ROW_KEYS = (
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


def _empty_meta_rows(obs_dim: int = 0) -> dict[str, np.ndarray]:
    h = FUTURE_TRAJ_HORIZON
    return {
        "obs": np.zeros((0, obs_dim), dtype=np.float32),
        "partner_slot": np.zeros((0,), dtype=np.int64),
        "time_idx": np.zeros((0,), dtype=np.int64),
        "agent_idx": np.zeros((0,), dtype=np.int64),
        "scenario_id": np.zeros((0,), dtype=np.int32),
        "scene_id": np.zeros((0,), dtype=np.int32),
        "timestep": np.zeros((0,), dtype=np.int64),
        "vehicle_id": np.zeros((0,), dtype=np.int32),
        "ego_id": np.zeros((0,), dtype=np.int32),
        "dist_at_t": np.zeros((0,), dtype=np.float32),
        "ego_state": np.zeros((0, 4), dtype=np.float32),
        "other_state": np.zeros((0, 4), dtype=np.float32),
        "future_traj": np.full((0, h, 4), np.nan, dtype=np.float32),
    }


def _gather_future_traj(
    other_traj: dict,
    a_idx: np.ndarray,
    p_idx: np.ndarray,
    t_idx: np.ndarray,
    vehicle_id: np.ndarray,
    *,
    horizon: int = FUTURE_TRAJ_HORIZON,
) -> np.ndarray:
    """Partner (x, y, heading, speed) for t..t+H-1; NaN where missing / id mismatch."""
    n = int(a_idx.shape[0])
    num_steps = int(other_traj["other_x"].shape[2])
    out = np.full((n, horizon, 4), np.nan, dtype=np.float32)
    if n == 0:
        return out

    offsets = np.arange(horizon, dtype=np.int64)
    t_fut = t_idx[:, None] + offsets[None, :]
    in_bounds = t_fut < num_steps
    t_clip = np.minimum(t_fut, num_steps - 1)

    a = a_idx[:, None]
    p = p_idx[:, None]
    fx = other_traj["other_x"][a, p, t_clip]
    fy = other_traj["other_y"][a, p, t_clip]
    fh = other_traj["other_heading"][a, p, t_clip]
    fs = other_traj["other_speed"][a, p, t_clip]
    fid = other_traj["other_id"][a, p, t_clip]
    same = (fid == vehicle_id[:, None]) & (vehicle_id[:, None] != -1) & in_bounds

    stacked = np.stack([fx, fy, fh, fs], axis=-1).astype(np.float32, copy=False)
    out[same] = stacked[same]
    return out


def thin_transition_indices(
    *,
    scene_id: np.ndarray,
    vehicle_id: np.ndarray,
    ego_id: np.ndarray,
    timestep: np.ndarray,
    max_timesteps_per_pair: int,
    min_timestep_gap: int,
    max_samples_per_scene: int | None,
) -> np.ndarray:
    """Keep spaced timesteps; favor scene diversity over adjacent frames.

    Rules (applied in sort order by timestep ascending within each ego–partner)::

      - at most ``max_timesteps_per_pair`` rows per (scene, ego, vehicle)
      - consecutive kept timesteps differ by >= ``min_timestep_gap``
      - optional hard cap ``max_samples_per_scene`` across all partners in a scene
    """
    n = int(timestep.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    order = np.argsort(timestep, kind="mergesort")
    keep: list[int] = []
    pair_kept: dict[tuple[int, int, int], list[int]] = {}
    scene_count: dict[int, int] = {}

    for idx in order.tolist():
        scen = int(scene_id[idx])
        ego = int(ego_id[idx])
        vid = int(vehicle_id[idx])
        t = int(timestep[idx])
        key = (scen, ego, vid)

        if max_samples_per_scene is not None and scene_count.get(scen, 0) >= max_samples_per_scene:
            continue

        prev = pair_kept.get(key)
        if prev is not None:
            if len(prev) >= max_timesteps_per_pair:
                continue
            if (t - prev[-1]) < min_timestep_gap:
                continue
        else:
            pair_kept[key] = []
            prev = pair_kept[key]

        keep.append(idx)
        prev.append(t)
        scene_count[scen] = scene_count.get(scen, 0) + 1

    keep_arr = np.asarray(keep, dtype=np.int64)
    keep_arr.sort()
    return keep_arr


def build_sae_transition_rows(
    ego_traj: dict,
    other_traj: dict,
    termination_mask: np.ndarray,
    agent_scenario: np.ndarray,
    *,
    max_min_dist_m: float | None = None,
    future_traj_horizon: int = FUTURE_TRAJ_HORIZON,
    max_timesteps_per_pair: int = DEFAULT_MAX_TIMESTEPS_PER_PAIR,
    min_timestep_gap: int = DEFAULT_MIN_TIMESTEP_GAP,
    max_samples_per_scene: int | None = DEFAULT_MAX_SAMPLES_PER_SCENE,
) -> dict[str, np.ndarray]:
    """Select (agent, partner_slot, t) rows + visualization metadata.

    Current-time filters only (no LP future labels). Each kept row includes
    ego/other state and a short partner ``future_traj`` window for plotting.
    Timesteps are thinned so SAE sees more independent scenes, not adjacent frames.
    """
    other_ids = other_traj["other_id"]
    other_speed = other_traj["other_speed"]
    obs_traj = ego_traj["obs"]  # (A, obs_dim, T)
    num_agents, num_slots, num_steps = other_ids.shape

    valid = (other_ids != -1) & (other_speed > 0)
    valid = valid & termination_mask[:, None, :]
    valid = valid & partner_vehicle_size_mask(
        obs_traj, num_partner_slots=num_slots
    )

    a_idx_grid = np.broadcast_to(
        np.arange(num_agents, dtype=np.int64)[:, None, None], valid.shape
    )
    p_idx_grid = np.broadcast_to(
        np.arange(num_slots, dtype=np.int64)[None, :, None], valid.shape
    )
    t_idx_grid = np.broadcast_to(
        np.arange(num_steps, dtype=np.int64)[None, None, :], valid.shape
    )

    a_idx = a_idx_grid[valid]
    p_idx = p_idx_grid[valid]
    t_idx = t_idx_grid[valid]

    obs_at = np.transpose(obs_traj, (0, 2, 1))  # (A, T, D)
    obs_full = np.broadcast_to(
        obs_at[:, None, :, :], (num_agents, num_slots, num_steps, obs_at.shape[-1])
    )
    o_obs = np.asarray(obs_full[valid], dtype=np.float32)
    scen = agent_scenario[a_idx]

    n_raw = int(a_idx.shape[0])
    if n_raw == 0:
        print("  SAE transitions: 0 (no valid partners at t)")
        return _empty_meta_rows(obs_dim=int(obs_traj.shape[1]))

    dist_at_t = partner_dist_at_t_from_obs_batch(o_obs, p_idx)
    if max_min_dist_m is not None:
        keep = dist_at_t <= float(max_min_dist_m)
        print(
            f"  dist@t filter (dist_at_t <= {max_min_dist_m:.2f} m): "
            f"kept {int(keep.sum())}/{n_raw} transitions"
        )
        a_idx = a_idx[keep]
        p_idx = p_idx[keep]
        t_idx = t_idx[keep]
        o_obs = o_obs[keep]
        scen = scen[keep]
        dist_at_t = dist_at_t[keep]
    else:
        print(f"  SAE transitions: {n_raw} (no dist@t filter)")

    vehicle_id = other_traj["other_id"][a_idx, p_idx, t_idx].astype(np.int32, copy=False)
    ego_id = other_traj["ego_id"][a_idx, t_idx].astype(np.int32, copy=False)
    ego_speed = (o_obs[:, 2] * OBS_MAX_SPEED_M_S).astype(np.float32, copy=False)
    ego_state = np.stack(
        [
            ego_traj["ego_x"][a_idx, t_idx],
            ego_traj["ego_y"][a_idx, t_idx],
            ego_traj["ego_heading"][a_idx, t_idx],
            ego_speed,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    other_state = np.stack(
        [
            other_traj["other_x"][a_idx, p_idx, t_idx],
            other_traj["other_y"][a_idx, p_idx, t_idx],
            other_traj["other_heading"][a_idx, p_idx, t_idx],
            other_traj["other_speed"][a_idx, p_idx, t_idx],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    future_traj = _gather_future_traj(
        other_traj,
        a_idx,
        p_idx,
        t_idx,
        vehicle_id,
        horizon=int(future_traj_horizon),
    )

    n_before_thin = int(a_idx.shape[0])
    if (
        max_timesteps_per_pair > 0
        or (min_timestep_gap > 1)
        or (max_samples_per_scene is not None and max_samples_per_scene > 0)
    ):
        keep = thin_transition_indices(
            scene_id=scen.astype(np.int32, copy=False),
            vehicle_id=vehicle_id,
            ego_id=ego_id,
            timestep=t_idx.astype(np.int64, copy=False),
            max_timesteps_per_pair=max(1, int(max_timesteps_per_pair)),
            min_timestep_gap=max(1, int(min_timestep_gap)),
            max_samples_per_scene=(
                None
                if max_samples_per_scene is None or max_samples_per_scene <= 0
                else int(max_samples_per_scene)
            ),
        )
        a_idx = a_idx[keep]
        p_idx = p_idx[keep]
        t_idx = t_idx[keep]
        o_obs = o_obs[keep]
        scen = scen[keep]
        dist_at_t = dist_at_t[keep]
        vehicle_id = vehicle_id[keep]
        ego_id = ego_id[keep]
        ego_state = ego_state[keep]
        other_state = other_state[keep]
        future_traj = future_traj[keep]
        print(
            f"  timestep thin (max_per_pair={max_timesteps_per_pair}, "
            f"min_gap={min_timestep_gap}, max_per_scene={max_samples_per_scene}): "
            f"kept {int(keep.shape[0])}/{n_before_thin} "
            f"({int(np.unique(scen).size)} scenes)"
        )

    return {
        "obs": o_obs,
        "partner_slot": p_idx.astype(np.int64, copy=False),
        "time_idx": t_idx.astype(np.int64, copy=False),
        "agent_idx": a_idx.astype(np.int64, copy=False),
        "scenario_id": scen.astype(np.int32, copy=False),
        # Aliases matching the viz schema.
        "scene_id": scen.astype(np.int32, copy=False),
        "timestep": t_idx.astype(np.int64, copy=False),
        "vehicle_id": vehicle_id,
        "ego_id": ego_id,
        "dist_at_t": dist_at_t.astype(np.float32, copy=False),
        "ego_state": ego_state,
        "other_state": other_state,
        "future_traj": future_traj,
    }


def write_activations_npz(
    step_dir: str,
    *,
    rows: dict[str, np.ndarray],
    activations: dict[str, np.ndarray],
    experiments: list[str],
    reference_exp: str,
    probe_step: int,
) -> str:
    out_path = os.path.join(step_dir, ACTIVATIONS_FILENAME)
    n = int(rows["partner_slot"].shape[0])

    def _base_fields() -> dict[str, np.ndarray]:
        fields: dict[str, np.ndarray] = {
            "activation_dim": np.int32(0 if n == 0 else activations[experiments[0]].shape[1]),
            "n_unique_transitions": np.int32(n),
            "repr_layer": np.array(REPR_LAYER),
            "rollout_mode": np.array(ROLLOUT_MODE),
            "reference_exp": np.array(reference_exp),
            "shared_obs": np.bool_(True),
            "experiments": np.array(experiments),
            "probe_step": np.int32(probe_step),
            "sae_collect_version": np.int32(SAE_COLLECT_VERSION),
            "ego_state_keys": np.array(EGO_STATE_KEYS),
            "other_state_keys": np.array(OTHER_STATE_KEYS),
            "future_traj_horizon": np.int32(FUTURE_TRAJ_HORIZON),
            "future_traj_keys": np.array(OTHER_STATE_KEYS),
            "schema": np.array(
                "activation / scene_id / timestep / vehicle_id / "
                "ego_state[x,y,heading,speed] / other_state[...] / future_traj[H,4]"
            ),
        }
        for key in META_ROW_KEYS:
            if key in rows:
                fields[key] = rows[key]
        # Optional raw obs for policy steering (large); not in META_ROW_KEYS.
        if "obs" in rows:
            fields["obs"] = rows["obs"]
        return fields

    if n == 0:
        fields = _base_fields()
        for exp_name in experiments:
            fields[activation_key(exp_name)] = np.zeros((0, 0), dtype=np.float32)
        save_npz_atomic(out_path, **fields)
        print(f"  wrote {out_path} (0 activations)")
        return out_path

    sample = activations[experiments[0]]
    fields = _base_fields()
    fields["activation_dim"] = np.int32(sample.shape[1])
    for exp_name in experiments:
        act = activations[exp_name]
        if act.shape[0] != n:
            raise RuntimeError(
                f"{exp_name}: activation rows {act.shape[0]} != transitions {n}"
            )
        fields[activation_key(exp_name)] = act.astype(np.float32)

    save_npz_atomic(out_path, **fields)
    obs_note = f", obs={fields['obs'].shape}" if "obs" in fields else ""
    print(
        f"  wrote {out_path}: {n} transitions, dim={sample.shape[1]}, "
        f"meta=ego/other_state+future_traj[{FUTURE_TRAJ_HORIZON}]{obs_note}, "
        f"experiments={experiments}"
    )
    return out_path


def collect_shared_activations(
    args: dict,
    vecenv,
    policy,
    *,
    reference_exp: str,
    reference_state_dict: dict,
    reference_step: int,
    experiments: list[str],
    base_path: str,
    output_dir: str,
    data_mode: str,
    save_raw: bool = False,
    save_obs: bool = False,
    max_min_dist_m: float | None = None,
    max_timesteps_per_pair: int = DEFAULT_MAX_TIMESTEPS_PER_PAIR,
    min_timestep_gap: int = DEFAULT_MIN_TIMESTEP_GAP,
    max_samples_per_scene: int | None = DEFAULT_MAX_SAMPLES_PER_SCENE,
    policy_run_dirs: dict[str, str] | None = None,
) -> str:
    """Human-replay rollout → multi-policy activations.npz (+ viz metadata)."""
    policy_run_dirs = policy_run_dirs or {}
    load_policy_from_checkpoint(policy, reference_state_dict)
    ego_traj, other_traj, termination_mask, agent_scenario = collect_selfplay_rollout(
        args, vecenv, policy
    )

    os.makedirs(output_dir, exist_ok=True)
    out_dir = os.path.join(output_dir, f"step_{reference_step:06d}")
    os.makedirs(out_dir, exist_ok=True)
    save_scene_context(
        vecenv.driver_env,
        os.path.join(out_dir, "scene_context.npz"),
        data_mode=data_mode,
    )

    if save_raw:
        save_npz_atomic(
            os.path.join(out_dir, "trajectories.npz"),
            **{f"ego_{k}": v for k, v in ego_traj.items()},
            **{f"other_{k}": v for k, v in other_traj.items()},
            termination_mask=termination_mask,
            agent_scenario=agent_scenario,
            rollout_mode=np.array(ROLLOUT_MODE),
            reference_exp=np.array(reference_exp),
        )

    rows = build_sae_transition_rows(
        ego_traj,
        other_traj,
        termination_mask,
        agent_scenario,
        max_min_dist_m=max_min_dist_m,
        max_timesteps_per_pair=max_timesteps_per_pair,
        min_timestep_gap=min_timestep_gap,
        max_samples_per_scene=max_samples_per_scene,
    )
    n = int(rows["partner_slot"].shape[0])
    print(f"  transitions for SAE encode: {n}")

    device = torch.device(args["train"]["device"])
    activations: dict[str, np.ndarray] = {}
    for exp_name in experiments:
        run_dir = policy_run_dirs.get(exp_name) or resolve_run_dir(base_path, exp_name)
        ckpt = pick_checkpoint(run_dir, device=str(device), probe_step=reference_step)
        assert ckpt.state_dict is not None
        load_policy_from_checkpoint(policy, ckpt.state_dict)
        policy.eval()
        act = batch_partner_context(
            policy,
            rows["obs"],
            rows["partner_slot"],
            device=device,
        )
        activations[exp_name] = act
        print(
            f"  [{exp_name}] step {ckpt.step:06d} ({os.path.basename(run_dir)}) "
            f"-> {activation_key(exp_name)} shape={act.shape}"
        )

    # Prefer saving obs for steering / policy hooks (optional; large).
    rows_meta = {k: rows[k] for k in META_ROW_KEYS if k in rows}
    if save_obs and "obs" in rows:
        rows_meta["obs"] = rows["obs"]
    write_activations_npz(
        out_dir,
        rows=rows_meta,
        activations=activations,
        experiments=experiments,
        reference_exp=reference_exp,
        probe_step=reference_step,
    )

    _drop_lp_artifacts(out_dir)
    return out_dir


def _drop_lp_artifacts(step_dir: str) -> None:
    """Remove leftover label-bearing LP files if present."""
    for path in glob.glob(os.path.join(step_dir, "future_*.npz")):
        os.remove(path)
        print(f"  removed leftover {os.path.basename(path)}")
    trend = os.path.join(step_dir, "label_distance_trend.npz")
    if os.path.isfile(trend):
        os.remove(trend)
        print("  removed leftover label_distance_trend.npz")


def _activations_ok(path: str, experiments: list[str]) -> bool:
    try:
        with np.load(path) as data:
            ver = (
                int(data["sae_collect_version"])
                if "sae_collect_version" in data.files
                else 0
            )
            if ver < SAE_COLLECT_VERSION:
                return False
            for exp_name in experiments:
                key = activation_key(exp_name)
                if key not in data.files:
                    return False
            # Viz metadata required since v5.
            for key in (
                "scene_id",
                "timestep",
                "vehicle_id",
                "ego_state",
                "other_state",
                "future_traj",
            ):
                if key not in data.files:
                    return False
            return True
    except Exception:
        return False


def parse_policy_run_dirs(spec: str | None) -> dict[str, str]:
    """Parse ``exp=/path;...``. Accepts aliases record/reactive/selfplay."""
    aliases = {
        "record": "replay_0.25",
        "rec": "replay_0.25",
        "replay": "replay_0.25",
        "replay_0.25": "replay_0.25",
        "reactive": "reactive_0.25",
        "reactive_0.25": "reactive_0.25",
        "selfplay": "selfplay",
        "sp": "selfplay",
    }
    out: dict[str, str] = {}
    if not spec:
        return out
    for part in spec.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, path = part.split("=", 1)
        key = name.strip().lower()
        exp = aliases.get(key, name.strip())
        path = path.strip()
        if exp and path:
            out[exp] = os.path.abspath(path)
    return out


def verify_sae_root(
    output_dir: str,
    data_mode: str,
    *,
    experiments: list[str],
    steps: set[int] | None = None,
) -> list[str]:
    root = resolve_output_dir(output_dir, data_mode)
    errors: list[str] = []
    for path in sorted(glob.glob(os.path.join(root, "step_*"))):
        match = re.search(r"step_(\d+)$", os.path.basename(path))
        if not match:
            continue
        step = int(match.group(1))
        if steps is not None and step not in steps:
            continue
        act_path = os.path.join(path, ACTIVATIONS_FILENAME)
        if not os.path.isfile(act_path):
            errors.append(f"missing {act_path}")
            continue
        if not _activations_ok(act_path, experiments):
            errors.append(f"corrupt/stale or missing experiment keys in {act_path}")
            continue
        with np.load(act_path) as data:
            n = int(data["n_unique_transitions"])
            for exp_name in experiments:
                key = activation_key(exp_name)
                if data[key].shape[0] != n:
                    errors.append(f"{act_path}: {key} length != n_unique_transitions")
        leftovers = glob.glob(os.path.join(path, "future_*.npz"))
        if leftovers:
            errors.append(f"unexpected future_*.npz under {path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Human-replay shared obs → multi-policy partner_encoder activations "
            "(activations.npz only)"
        )
    )
    parser.add_argument("--base-path", type=str, default="/data/puffer/experiments")
    parser.add_argument(
        "--reference-exp",
        type=str,
        default="replay_0.25",
        help="Policy that drives the human-replay rollout (defines shared obs)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=",".join(DEFAULT_EXPERIMENTS),
        help="Comma-separated experiments to encode on the shared obs",
    )
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument(
        "--policy-run-dirs",
        type=str,
        default=None,
        help=(
            "Per-experiment encode checkpoints: "
            "replay_0.25=/path;reactive_0.25=/path;selfplay=/path "
            "(aliases record/reactive/selfplay also ok). "
            "Overrides default resolve_run_dir for encoding; "
            "--run-dir still selects the reference rollout policy."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="SAE root; writes <output-dir>/human_replay/{training,validation}/",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num-maps", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-mode", type=str, default="training", choices=["training", "validation"])
    parser.add_argument("--probe-step", type=int, default=None)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument(
        "--save-obs",
        action="store_true",
        help="Also store raw obs in activations.npz (needed for policy feature steering)",
    )
    parser.add_argument(
        "--max-min-dist-m",
        type=float,
        default=10.0,
        help="Keep transitions with partner dist@t <= this (m). Negative = no filter.",
    )
    parser.add_argument(
        "--max-timesteps-per-pair",
        type=int,
        default=DEFAULT_MAX_TIMESTEPS_PER_PAIR,
        help="Max kept timesteps per (scene, ego, partner vehicle).",
    )
    parser.add_argument(
        "--min-timestep-gap",
        type=int,
        default=DEFAULT_MIN_TIMESTEP_GAP,
        help="Min step spacing between kept timesteps for the same ego–partner.",
    )
    parser.add_argument(
        "--max-samples-per-scene",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_SCENE,
        help="Hard cap on kept rows per scene (0 = unlimited).",
    )
    parser.add_argument("--force-collect", action="store_true")
    args_cli = parser.parse_args()

    experiments = parse_csv_list(args_cli.experiments)
    if not experiments:
        raise ValueError("--experiments is empty")
    max_min_dist_m = None if args_cli.max_min_dist_m < 0 else float(args_cli.max_min_dist_m)

    reference_run = resolve_run_dir(args_cli.base_path, args_cli.reference_exp, args_cli.run_dir)
    reference_ckpt = pick_checkpoint(
        reference_run, device=args_cli.device, probe_step=args_cli.probe_step
    )
    probe_step = int(reference_ckpt.step)
    assert reference_ckpt.state_dict is not None
    policy_run_dirs = parse_policy_run_dirs(args_cli.policy_run_dirs)
    if policy_run_dirs:
        print(f"  policy_run_dirs={policy_run_dirs}")

    output_dir = resolve_output_dir(args_cli.output_dir, args_cli.data_mode)
    step_dir = os.path.join(output_dir, f"step_{probe_step:06d}")
    act_path = os.path.join(step_dir, ACTIVATIONS_FILENAME)
    if (
        not args_cli.force_collect
        and os.path.isfile(act_path)
        and _activations_ok(act_path, experiments)
    ):
        print(f"skip collect ({act_path} already ok)")
        print("done")
        return

    print(
        f"Human-replay SAE collect "
        f"(num_maps={args_cli.num_maps}, device={args_cli.device}, "
        f"data_mode={args_cli.data_mode}, probe_step={probe_step})"
    )
    print(f"  reference_exp={args_cli.reference_exp} (defines shared obs)")
    print(f"  experiments={experiments}")
    print(f"  representation={REPR_LAYER}")
    print(f"  output={output_dir}")
    print("  disk: activations.npz (+ ego/other_state, future_traj)")
    if args_cli.save_obs:
        print("  save_obs=True (raw obs stored for policy steering)")
    if max_min_dist_m is not None:
        print(f"  dist@t filter: partner dist_at_t <= {max_min_dist_m:.2f} m")
    print(
        f"  diversity: max_timesteps_per_pair={args_cli.max_timesteps_per_pair} "
        f"min_gap={args_cli.min_timestep_gap} "
        f"max_per_scene={args_cli.max_samples_per_scene}"
    )

    args = build_human_replay_drive_args(
        args_cli.config,
        num_maps=args_cli.num_maps,
        device=args_cli.device,
        data_mode=args_cli.data_mode,
    )
    vecenv = create_vecenv(args, env_name="puffer_drive")
    policy = create_policy(args, vecenv, env_name="puffer_drive")

    out_dir = collect_shared_activations(
        args,
        vecenv,
        policy,
        reference_exp=args_cli.reference_exp,
        reference_state_dict=reference_ckpt.state_dict,
        reference_step=probe_step,
        experiments=experiments,
        base_path=args_cli.base_path,
        output_dir=output_dir,
        data_mode=args_cli.data_mode,
        save_raw=args_cli.save_raw,
        save_obs=bool(args_cli.save_obs),
        max_min_dist_m=max_min_dist_m,
        max_timesteps_per_pair=int(args_cli.max_timesteps_per_pair),
        min_timestep_gap=int(args_cli.min_timestep_gap),
        max_samples_per_scene=(
            None
            if int(args_cli.max_samples_per_scene) <= 0
            else int(args_cli.max_samples_per_scene)
        ),
        policy_run_dirs=policy_run_dirs,
    )
    errors = verify_sae_root(
        args_cli.output_dir,
        args_cli.data_mode,
        experiments=experiments,
        steps={probe_step},
    )
    safe_close_vecenv(vecenv)
    if errors:
        raise RuntimeError("SAE collect verify failed:\n  " + "\n  ".join(errors))
    print(f"  -> {out_dir}/{ACTIVATIONS_FILENAME}")
    print("done")


if __name__ == "__main__":
    main()
