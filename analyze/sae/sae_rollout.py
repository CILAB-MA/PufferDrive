"""Rollout / env helpers used by SAE activation collect (human-replay shared obs)."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

_DATA_MODES = frozenset({"training", "validation"})

HUMAN_REPLAY_SUBDIR = "human_replay"
ROLLOUT_MODE = "human_replay"
REPR_LAYER = "partner_encoder_slot"
DEFAULT_EXPERIMENTS = ("selfplay", "reactive_0.25", "replay_0.25")

# drive.h partner obs: width/length normalized by these maxima
MAX_VEH_WIDTH_M = 15.0
MAX_VEH_LEN_M = 30.0
EGO_OBS_DIM = 7
PARTNER_OBS_DIM = 7
MAX_PARTNER_SLOTS = 31  # drive.h MAX_AGENTS - 1
MIN_VEHICLE_AREA_M2 = 3.0
MIN_VEHICLE_LENGTH_M = 2.5
OBS_PARTNER_SCALE = 0.02


def parse_csv_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def resolve_output_dir(output_dir: str, data_mode: str) -> str:
    """Place npz under human_replay/training/ or human_replay/validation/."""
    if data_mode not in _DATA_MODES:
        raise ValueError(f"data_mode must be one of {sorted(_DATA_MODES)}, got {data_mode!r}")
    base = os.path.normpath(output_dir)
    parts = base.split(os.sep)
    if len(parts) >= 2 and parts[-1] in _DATA_MODES and parts[-2] == HUMAN_REPLAY_SUBDIR:
        return base
    if parts and parts[-1] == HUMAN_REPLAY_SUBDIR:
        return os.path.join(base, data_mode)
    return os.path.join(base, HUMAN_REPLAY_SUBDIR, data_mode)


def build_human_replay_drive_args(
    config_path: str | None = None,
    *,
    num_maps: int = 300,
    device: str = "cuda",
    data_mode: str = "training",
) -> dict:
    """puffer_drive args aligned with pufferl human_replay / log-replay eval."""
    from pufferlib.pufferl import load_config

    env_name = "puffer_drive"
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]
        args = load_config(env_name, config_dir=config_path)
    finally:
        sys.argv = saved_argv

    map_section = args.get(data_mode) or args["eval"]
    args["env"]["map_dir"] = map_section["map_dir"]
    args["env"]["num_maps"] = num_maps
    args["env"]["sequential_map_sampling"] = True
    args["env"]["episode_length"] = 91
    args["env"]["termination_mode"] = 0
    args["env"]["control_mode"] = args["eval"].get("human_replay_control_mode", "control_sdc_only")
    args["vec"] = dict(backend=args["eval"].get("backend", "PufferEnv"), num_envs=1)
    args["train"]["device"] = device
    args["load_model_path"] = None
    return args


def is_flat_policy_ckpt(path: str) -> bool:
    from load_ckpt import is_flat_policy_ckpt as _is_flat

    return _is_flat(path)


def resolve_policy_location(
    base_path: str,
    exp_name: str,
    location: str | None = None,
    *,
    sweep_id: str | None = None,
    prefer_final: bool = True,
) -> str:
    from load_ckpt import resolve_policy_location as _resolve

    return _resolve(
        base_path,
        exp_name,
        location,
        sweep_id=sweep_id,
        prefer_final=prefer_final,
    )


def resolve_run_dir(base_path: str, exp_name: str, run_dir: str | None = None) -> str:
    """Legacy helper: returns a run directory. Prefer :func:`resolve_policy_location`."""
    from load_ckpt import resolve_policy_location

    if run_dir and (run_dir.endswith(".pt") or is_flat_policy_ckpt(run_dir)):
        return os.path.abspath(run_dir)
    try:
        return resolve_policy_location(base_path, exp_name, run_dir, prefer_final=False)
    except (FileNotFoundError, ValueError):
        from load_ckpt import find_run_dirs

        if run_dir:
            return os.path.abspath(run_dir)
        exp_path = os.path.join(base_path, exp_name)
        run_dirs = find_run_dirs(exp_path)
        if not run_dirs:
            raise FileNotFoundError(f"No run dirs under {exp_path}")
        return run_dirs[0][1]


def pick_checkpoint(
    location: str,
    *,
    device: str,
    probe_step: int | None,
):
    """Load checkpoint tensors on CPU; ``load_state_dict`` copies them to the policy.

    Loading directly onto CUDA temporarily duplicates the model weights and can
    OOM an otherwise viable rollout.
    """
    from load_ckpt import Checkpoint, iter_checkpoints, is_flat_policy_ckpt, load_checkpoint

    if is_flat_policy_ckpt(location) or (
        location.endswith(".pt") and os.path.isfile(location)
    ):
        state_dict = load_checkpoint(location, device="cpu")
        step = int(probe_step) if probe_step is not None else 0
        return Checkpoint(step=step, path=location, state_dict=state_dict)

    checkpoints = list(
        iter_checkpoints(
            location,
            device="cpu",
            map_location="cpu",
            load_state=True,
        )
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {location}")
    if probe_step is None:
        return checkpoints[-1]
    for ckpt in checkpoints:
        if ckpt.step == probe_step:
            return ckpt
    available = ", ".join(f"{c.step:06d}" for c in checkpoints)
    raise FileNotFoundError(
        f"No checkpoint step {probe_step:06d} in {location} (have: {available})"
    )


def save_npz_atomic(path: str, **arrays) -> None:
    """Write npz atomically so readers never see a partial zip."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=dir_name)
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        for candidate in (tmp_path, f"{tmp_path}.npz"):
            if os.path.exists(candidate):
                os.remove(candidate)
        raise


def create_vecenv(args: dict, env_name: str = "puffer_drive"):
    from pufferlib.pufferl import load_env

    return load_env(env_name, args)


def safe_close_vecenv(vecenv) -> None:
    if vecenv is None:
        return
    try:
        vecenv.close()
    except Exception:
        pass


def create_policy(args: dict, vecenv, env_name: str = "puffer_drive"):
    from pufferlib.pufferl import load_policy

    policy_args = {**args, "load_model_path": None}
    policy = load_policy(policy_args, vecenv, env_name)
    policy.eval()
    return policy


def load_policy_from_checkpoint(policy, state_dict: dict) -> None:
    policy.load_state_dict(state_dict)


def ego_indices_from_reset(args: dict, driver, infos) -> np.ndarray:
    """First controlled agent per map."""
    ego: list[int] = []
    stride = int(args["env"]["num_agents"])
    if infos:
        for env_i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            ao = np.asarray(info.get("agent_offsets", driver.agent_offsets), dtype=np.int64)
            ego.extend((ao[:-1] + stride * env_i).tolist())
    if not ego:
        ao = np.asarray(driver.agent_offsets, dtype=np.int64)
        ego.extend(ao[:-1].tolist())
    out = np.asarray(ego, dtype=np.int64)
    if out.size == 0:
        raise RuntimeError("no per-map ego indices from reset")
    return out


def _init_ego_buffers(
    num_ego: int,
    obs_dim: int,
    sim_steps: int,
    num_partners: int = 31,
) -> tuple[dict, dict]:
    other_trajectories = {
        "other_x": np.zeros((num_ego, num_partners, sim_steps), dtype=np.float32),
        "other_y": np.zeros((num_ego, num_partners, sim_steps), dtype=np.float32),
        "other_heading": np.zeros((num_ego, num_partners, sim_steps), dtype=np.float32),
        "other_id": np.zeros((num_ego, num_partners, sim_steps), dtype=np.int32),
        "ego_id": np.zeros((num_ego, sim_steps), dtype=np.int32),
        "other_speed": np.zeros((num_ego, num_partners, sim_steps), dtype=np.float32),
    }
    trajectories = {
        "obs": np.zeros((num_ego, obs_dim, sim_steps), dtype=np.float32),
        "ego_x": np.zeros((num_ego, sim_steps), dtype=np.float32),
        "ego_y": np.zeros((num_ego, sim_steps), dtype=np.float32),
        "ego_heading": np.zeros((num_ego, sim_steps), dtype=np.float32),
    }
    return trajectories, other_trajectories


def _record_ego_timestep(
    *,
    local_i: int,
    g_idx: int,
    time_idx: int,
    obs,
    driver,
    trajectories: dict,
    other_trajectories: dict,
) -> None:
    partner_state = driver.get_global_partner_state()
    agent_state = driver.get_global_agent_state()
    other_trajectories["other_x"][local_i, :, time_idx] = partner_state["x"][g_idx]
    other_trajectories["other_y"][local_i, :, time_idx] = partner_state["y"][g_idx]
    other_trajectories["other_speed"][local_i, :, time_idx] = partner_state["speed"][g_idx]
    other_trajectories["other_heading"][local_i, :, time_idx] = partner_state["heading"][g_idx]
    other_trajectories["other_id"][local_i, :, time_idx] = partner_state["other_id"][g_idx]
    other_trajectories["ego_id"][local_i, time_idx] = partner_state["ego_id"][g_idx]
    trajectories["obs"][local_i, :, time_idx] = obs[g_idx]
    trajectories["ego_x"][local_i, time_idx] = agent_state["x"][g_idx]
    trajectories["ego_y"][local_i, time_idx] = agent_state["y"][g_idx]
    trajectories["ego_heading"][local_i, time_idx] = agent_state["heading"][g_idx]


def _episode_info_completed(info_list) -> bool:
    if not info_list:
        return False
    for info in info_list:
        if not isinstance(info, dict):
            continue
        if any(k in info for k in ("episode_return", "episode_length", "score", "ego_score")):
            return True
    return False


def _update_ego_end_steps(
    *,
    time_idx: int,
    obs: np.ndarray,
    ego_indices: np.ndarray,
    end_step: np.ndarray,
    driver,
    respawn_idx: int,
    prev_respawn: np.ndarray,
    info_list,
    sim_steps: int,
) -> np.ndarray:
    respawn = obs[:, respawn_idx] > 0.5
    newly_respawned = respawn & ~prev_respawn
    prev_respawn[:] = respawn
    terminals = driver.terminals.astype(bool)
    truncations = driver.truncations.astype(bool)
    for local_i, g_idx in enumerate(ego_indices):
        if time_idx < end_step[local_i] and (
            terminals[g_idx] or truncations[g_idx] or newly_respawned[g_idx]
        ):
            end_step[local_i] = min(end_step[local_i], time_idx + 1)
    if _episode_info_completed(info_list) and time_idx + 1 >= sim_steps:
        end_step[:] = np.minimum(end_step, time_idx + 1)
    return prev_respawn


def _agent_scenario_ids(driver, num_agents: int) -> np.ndarray:
    ao = np.asarray(driver.agent_offsets, dtype=np.int64)
    map_ids = np.asarray(driver.map_ids, dtype=np.int32)
    out = np.zeros(num_agents, dtype=np.int32)
    for env_i in range(len(map_ids)):
        out[ao[env_i] : ao[env_i + 1]] = map_ids[env_i]
    return out


def partner_dist_at_t_from_obs_batch(
    obs: np.ndarray,
    partner_slot: np.ndarray,
) -> np.ndarray:
    """Partner ‖rel‖ from ego at observation time t (meters)."""
    slots = partner_slot.astype(np.int64)
    rows = np.arange(obs.shape[0], dtype=np.int64)
    bases = EGO_OBS_DIM + slots * PARTNER_OBS_DIM
    rel_x = obs[rows, bases].astype(np.float64) / OBS_PARTNER_SCALE
    rel_y = obs[rows, bases + 1].astype(np.float64) / OBS_PARTNER_SCALE
    return np.hypot(rel_x, rel_y).astype(np.float32)


def partner_vehicle_size_mask(
    obs_t: np.ndarray,
    *,
    num_partner_slots: int = MAX_PARTNER_SLOTS,
) -> np.ndarray:
    """True where partner obs slot looks vehicle-sized (not tiny ped/bike)."""
    slots = np.arange(num_partner_slots)
    idx_w = EGO_OBS_DIM + slots * PARTNER_OBS_DIM + 2
    idx_l = EGO_OBS_DIM + slots * PARTNER_OBS_DIM + 3
    width_m = obs_t[:, idx_w, :] * MAX_VEH_WIDTH_M
    length_m = obs_t[:, idx_l, :] * MAX_VEH_LEN_M
    area_m = width_m * length_m
    return (area_m >= MIN_VEHICLE_AREA_M2) & (length_m >= MIN_VEHICLE_LENGTH_M)


def save_scene_context(driver, path: str, *, data_mode: str) -> None:
    """Save road-edge polylines and per-agent scenario ids for map rendering."""
    poly = driver.get_road_edge_polylines()
    save_npz_atomic(
        path,
        road_edge_x=poly["x"],
        road_edge_y=poly["y"],
        road_edge_lengths=poly["lengths"],
        road_edge_scenario_id=poly["scenario_id"],
        map_ids=np.asarray(driver.map_ids, dtype=np.int32),
        agent_offsets=np.asarray(driver.agent_offsets, dtype=np.int64),
        data_mode=np.array(data_mode),
    )


def collect_selfplay_rollout(
    args: dict,
    vecenv,
    policy,
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Human-replay / selfplay-style rollout: record ego + partner trajectories."""
    import torch
    import pufferlib

    driver = vecenv.driver_env
    sim_steps = int(args["env"]["episode_length"])
    num_agents_global = vecenv.observation_space.shape[0]
    obs_dim = int(np.prod(vecenv.single_observation_space.shape))
    device = args["train"]["device"]
    respawn_idx = 9 if args["env"].get("dynamics_model") == "jerk" else 6

    obs, infos = vecenv.reset()
    lp_ego_indices = ego_indices_from_reset(args, driver, infos)
    num_lp_ego = int(lp_ego_indices.size)
    agent_scenario_all = _agent_scenario_ids(driver, num_agents_global)
    agent_scenario = agent_scenario_all[lp_ego_indices]

    trajectories, other_trajectories = _init_ego_buffers(num_lp_ego, obs_dim, sim_steps)
    end_step = np.full(num_lp_ego, sim_steps, dtype=np.int32)
    prev_respawn = np.zeros(num_agents_global, dtype=bool)

    state = {}
    if args["train"]["use_rnn"]:
        state = dict(
            lstm_h=torch.zeros(num_agents_global, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents_global, policy.hidden_size, device=device),
        )

    print(
        f"  rollout ego: {num_lp_ego}/{num_agents_global} agents "
        f"(one per map)"
    )

    for time_idx in range(sim_steps):
        for local_i, g_idx in enumerate(lp_ego_indices):
            if time_idx >= end_step[local_i]:
                continue
            _record_ego_timestep(
                local_i=local_i,
                g_idx=int(g_idx),
                time_idx=time_idx,
                obs=obs,
                driver=driver,
                trajectories=trajectories,
                other_trajectories=other_trajectories,
            )

        with torch.no_grad():
            ob_tensor = torch.as_tensor(obs).to(device)
            logits, _ = policy.forward_eval(ob_tensor, state)
            action, _, _ = pufferlib.pytorch.sample_logits(logits)
            action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)
        if isinstance(logits, torch.distributions.Normal):
            action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high)

        obs, _, _, _, info_list = vecenv.step(action_np)

        prev_respawn = _update_ego_end_steps(
            time_idx=time_idx,
            obs=obs,
            ego_indices=lp_ego_indices,
            end_step=end_step,
            driver=driver,
            respawn_idx=respawn_idx,
            prev_respawn=prev_respawn,
            info_list=info_list,
            sim_steps=sim_steps,
        )

    termination_mask = np.arange(sim_steps)[None, :] < end_step[:, None]
    return trajectories, other_trajectories, termination_mask, agent_scenario
