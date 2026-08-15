#!/usr/bin/env python3
"""Same-map per-ego rollout packs for coordination analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from common import (
    DEFAULT_THRESHOLDS,
    accel_from_actions,
    action_stats_from_logits,
    nearest_from_states,
    nearest_from_states_with_pose,
)
from runtime import (
    agent_scenario_ids,
    build_human_replay_drive_args,
    create_policy,
    create_vecenv,
    ego_indices_from_reset,
    pick_checkpoint,
    safe_close_vecenv,
)

OBS_COLLISION_IDX = 5
DT = 0.1


def _first_true(mask: np.ndarray) -> int:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else -1


def rollout_per_ego(
    *,
    ckpt: Path,
    num_maps: int,
    device: str,
    data_mode: str = "validation",
    ttc_approach: float = DEFAULT_THRESHOLDS["ttc_approach"],
    closing_approach: float = DEFAULT_THRESHOLDS["closing_approach"],
    dist_approach: float = DEFAULT_THRESHOLDS["dist_approach"],
    ttc_tight: float = DEFAULT_THRESHOLDS["ttc_tight"],
    closing_tight: float = DEFAULT_THRESHOLDS["closing_tight"],
    dist_tight: float = DEFAULT_THRESHOLDS["dist_tight"],
    pre_window: int = DEFAULT_THRESHOLDS["pre_window"],
    hard_brake: float = DEFAULT_THRESHOLDS["hard_brake"],
    capture_readout: bool = False,
    capture_obs: bool = False,
    capture_pose: bool = False,
    capture_map_geometry: bool = False,
    brake_logit_bias: float = 0.0,
    bias_onset: np.ndarray | None = None,
    bias_tau_lo: int = -15,
    bias_tau_hi: int = -6,
    map_start: int = 0,
) -> dict[str, np.ndarray]:
    import pufferlib
    from common import N_ACTIONS, N_STEER

    if data_mode not in ("training", "validation"):
        raise ValueError(f"data_mode must be training|validation, got {data_mode!r}")
    args = build_human_replay_drive_args(
        num_maps=num_maps, device=device, data_mode=data_mode, map_start=map_start
    )
    vecenv = create_vecenv(args, env_name="puffer_drive")
    policy = create_policy(args, vecenv, env_name="puffer_drive")
    ck = pick_checkpoint(str(ckpt), device="cpu", probe_step=None)
    policy.load_state_dict(ck.state_dict)
    policy.to(device).eval()

    driver = vecenv.driver_env
    sim_steps = int(args["env"]["episode_length"])
    obs, infos = vecenv.reset()
    ego_idx = ego_indices_from_reset(args, driver, infos)
    n_ego = int(ego_idx.size)
    num_agents = int(vecenv.observation_space.shape[0])

    map_geometry: dict[str, np.ndarray] = {}
    if capture_map_geometry:
        # Map geometry is static per scenario -- captured once right after reset, NOT
        # per-step, and MUST happen before the env is closed (safe_close_vecenv below);
        # calling it after close is a use-after-free (crashed with a segfault when first
        # tried at the end of this function, hence capturing it here instead).
        lanes = driver.get_lane_polylines()
        map_geometry["lane_polyline_x"] = lanes["x"]
        map_geometry["lane_polyline_y"] = lanes["y"]
        map_geometry["lane_polyline_lengths"] = lanes["lengths"]
        map_geometry["lane_polyline_scenario_id"] = lanes["scenario_id"]
        crosswalks = driver.get_crosswalk_polylines()
        map_geometry["crosswalk_polyline_x"] = crosswalks["x"]
        map_geometry["crosswalk_polyline_y"] = crosswalks["y"]
        map_geometry["crosswalk_polyline_lengths"] = crosswalks["lengths"]
        map_geometry["crosswalk_polyline_scenario_id"] = crosswalks["scenario_id"]
        # get_road_edge_polylines() already existed before this session (used elsewhere);
        # just wasn't captured into analysis packs. Needed for PF's U_road component.
        edges = driver.get_road_edge_polylines()
        map_geometry["road_edge_polyline_x"] = edges["x"]
        map_geometry["road_edge_polyline_y"] = edges["y"]
        map_geometry["road_edge_polyline_lengths"] = edges["lengths"]
        map_geometry["road_edge_polyline_scenario_id"] = edges["scenario_id"]

    num_agents_global = int(getattr(driver, "num_agents", num_agents))
    try:
        scene_ids = agent_scenario_ids(driver, num_agents_global)[ego_idx]
    except Exception:
        scene_ids = np.arange(n_ego, dtype=np.int64)

    state = {}
    if args["train"]["use_rnn"]:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    min_dist = np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
    closing_s = np.zeros((n_ego, sim_steps), dtype=np.float32)
    ttc_s = np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
    speed_s = np.zeros((n_ego, sim_steps), dtype=np.float32)
    accel_s = np.zeros((n_ego, sim_steps), dtype=np.float32)
    approach = np.zeros((n_ego, sim_steps), dtype=bool)
    tight = np.zeros((n_ego, sim_steps), dtype=bool)
    env_coll = np.zeros((n_ego, sim_steps), dtype=bool)
    prev_xy = np.full((n_ego, 2), np.nan, dtype=np.float64)

    if capture_pose:
        ego_x_s = np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
        ego_y_s = np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
        ego_heading_s = np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
        other_x_s = np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
        other_y_s = np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
        other_heading_s = np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
        # Raw entity index of whichever agent was nearest at each step -- "nearest" is
        # recomputed independently every step, so this can (and does, near crossovers)
        # switch identity mid-episode. Any metric using other_x/y/heading_traj as a
        # single coherent trajectory must first check this stays constant over the
        # window it analyzes.
        other_id_s = np.full((n_ego, sim_steps), -1, dtype=np.int32)
        other_speed_s = np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
        other_length_s = np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
        other_width_s = np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
        # Vehicle length/width are static per episode (entity property, not per-step
        # state) -- captured once, from the first step's agent_state.
        ego_length = np.full(n_ego, np.nan, dtype=np.float32)
        ego_width = np.full(n_ego, np.nan, dtype=np.float32)

    readout_out = {
        "accel": "exp_accel",
        "p_brake": "p_brake",
        "p_yield": "p_yield",
        "gap_press": "gap_press",
        "entropy": "entropy",
    }
    readout_s: dict[str, np.ndarray] = {}
    obs_traj = None
    obs_dim = int(np.prod(vecenv.single_observation_space.shape))
    if capture_readout:
        readout_s = {
            out: np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
            for out in readout_out.values()
        }
    if capture_obs:
        obs_traj = np.full((n_ego, sim_steps, obs_dim), np.nan, dtype=np.float32)

    for t in range(sim_steps):
        partner_state = driver.get_global_partner_state()
        agent_state = driver.get_global_agent_state()
        for li, g in enumerate(ego_idx):
            g = int(g)
            xy = np.array([agent_state["x"][g], agent_state["y"][g]], dtype=np.float64)
            spd = (
                float(np.linalg.norm(xy - prev_xy[li]) / DT)
                if np.isfinite(prev_xy[li]).all()
                else 0.0
            )
            prev_xy[li] = xy
            speed_s[li, t] = spd
            if capture_pose:
                (
                    d,
                    cl,
                    ttc,
                    other_x,
                    other_y,
                    other_heading,
                    other_id,
                    other_speed,
                    other_length,
                    other_width,
                ) = nearest_from_states_with_pose(
                    xy,
                    float(agent_state["heading"][g]),
                    spd,
                    np.stack([partner_state["x"][g], partner_state["y"][g]], axis=-1).astype(np.float64),
                    partner_state["heading"][g],
                    partner_state["speed"][g],
                    partner_state["other_id"][g],
                    partner_state["length"][g],
                    partner_state["width"][g],
                )
                ego_x_s[li, t] = xy[0]
                ego_y_s[li, t] = xy[1]
                ego_heading_s[li, t] = float(agent_state["heading"][g])
                other_x_s[li, t] = other_x
                other_y_s[li, t] = other_y
                other_heading_s[li, t] = other_heading
                other_id_s[li, t] = other_id
                other_speed_s[li, t] = other_speed
                other_length_s[li, t] = other_length
                other_width_s[li, t] = other_width
                if t == 0:
                    ego_length[li] = float(agent_state["length"][g])
                    ego_width[li] = float(agent_state["width"][g])
            else:
                d, cl, ttc = nearest_from_states(
                    xy,
                    float(agent_state["heading"][g]),
                    spd,
                    np.stack([partner_state["x"][g], partner_state["y"][g]], axis=-1).astype(np.float64),
                    partner_state["heading"][g],
                    partner_state["speed"][g],
                    partner_state["other_id"][g],
                )
            min_dist[li, t] = d
            closing_s[li, t] = cl
            ttc_s[li, t] = ttc
            finite = np.isfinite(ttc) and np.isfinite(d)
            approach[li, t] = (
                finite and (cl >= closing_approach) and (ttc < ttc_approach) and (d < dist_approach)
            )
            tight[li, t] = (
                finite and (cl >= closing_tight) and (ttc < ttc_tight) and (d < dist_tight)
            )

        with torch.no_grad():
            ob_tensor = torch.as_tensor(obs).to(device)
            ego_t = torch.as_tensor(ego_idx, device=device, dtype=torch.long)
            if capture_obs:
                obs_traj[:, t] = ob_tensor.index_select(0, ego_t).detach().cpu().numpy()
            logits, _ = policy.forward_eval(ob_tensor, state)
            if (
                brake_logit_bias != 0.0
                and bias_onset is not None
                and int(bias_onset.shape[0]) == n_ego
            ):
                logit_t = logits[0] if isinstance(logits, (tuple, list)) else logits
                a_idx = torch.arange(N_ACTIONS, device=device) // N_STEER
                brake_cols = (a_idx < 3).nonzero(as_tuple=False).squeeze(-1)
                onsets = torch.as_tensor(bias_onset, device=device, dtype=torch.long)
                in_win = (onsets >= 0) & ((onsets + int(bias_tau_lo)) <= t) & (
                    t <= (onsets + int(bias_tau_hi))
                )
                if bool(in_win.any()):
                    g_sel = ego_t[in_win]
                    logit_t[g_sel.unsqueeze(1), brake_cols] = logit_t[
                        g_sel.unsqueeze(1), brake_cols
                    ] + float(brake_logit_bias)
            action, _, _ = pufferlib.pytorch.sample_logits(logits)
            action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)

        accel_all = accel_from_actions(action_np)
        if accel_all.shape[0] == num_agents:
            accel_s[:, t] = accel_all[ego_idx]
        else:
            accel_s[:, t] = np.nan

        if capture_readout:
            stats = action_stats_from_logits(logits)
            for src, out in readout_out.items():
                readout_s[out][:, t] = stats[src].index_select(0, ego_t).detach().cpu().numpy()

        obs, _, _, _, _ = vecenv.step(action_np)
        for li, g in enumerate(ego_idx):
            row = obs[int(g)]
            env_coll[li, t] = row.shape[0] > OBS_COLLISION_IDX and float(row[OBS_COLLISION_IDX]) == 1.0

    try:
        safe_close_vecenv(vecenv)
    except Exception:
        pass

    collided = np.any(env_coll, axis=1)
    had_approach = np.any(approach, axis=1)
    had_tight = np.any(tight, axis=1)
    resolved = had_approach & (~had_tight)
    escalated = had_approach & had_tight

    ep_min_dist = np.full(n_ego, np.nan, dtype=np.float64)
    ep_min_ttc = np.full(n_ego, np.nan, dtype=np.float64)
    for i in range(n_ego):
        d_row = min_dist[i]
        d_ok = np.isfinite(d_row)
        if d_ok.any():
            ep_min_dist[i] = float(np.min(d_row[d_ok]))
        t_row = ttc_s[i]
        t_ok = np.isfinite(t_row)
        if t_ok.any():
            ep_min_ttc[i] = float(np.min(t_row[t_ok]))

    pre_speed = np.full(n_ego, np.nan, dtype=np.float64)
    entry_speed = np.full(n_ego, np.nan, dtype=np.float64)
    cpa_dist = np.full(n_ego, np.nan, dtype=np.float64)
    cpa_speed = np.full(n_ego, np.nan, dtype=np.float64)
    for i in range(n_ego):
        if had_tight[i]:
            e = _first_true(tight[i])
            if e >= 0:
                entry_speed[i] = float(speed_s[i, e])
                lo = max(0, e - pre_window)
                if e - lo >= 3:
                    pre_speed[i] = float(speed_s[i, lo:e].mean())
        if resolved[i]:
            m = approach[i]
            d_row = min_dist[i].copy()
            d_row[~m] = np.inf
            t_cpa = int(np.argmin(d_row))
            cpa_dist[i] = float(min_dist[i, t_cpa])
            cpa_speed[i] = float(speed_s[i, t_cpa])

    tight_onset = np.full(n_ego, -1, dtype=np.int32)
    for i in range(n_ego):
        if had_tight[i]:
            tight_onset[i] = _first_true(tight[i])

    pack: dict[str, np.ndarray] = {
        "scene_id": scene_ids.astype(np.int64, copy=False),
        "collided": collided.astype(bool),
        "had_approach": had_approach.astype(bool),
        "had_tight": had_tight.astype(bool),
        "resolved": resolved.astype(bool),
        "escalated": escalated.astype(bool),
        "ep_min_dist": ep_min_dist,
        "ep_min_ttc": ep_min_ttc,
        "pre_entry_speed": pre_speed,
        "entry_speed": entry_speed,
        "cpa_dist": cpa_dist,
        "cpa_speed": cpa_speed,
        "mean_hard_brake": (accel_s <= hard_brake).mean(axis=1).astype(np.float64),
        "tight_onset": tight_onset,
    }

    if capture_readout or capture_obs:
        pack["min_dist_traj"] = min_dist.astype(np.float32)
        pack["ttc_traj"] = ttc_s.astype(np.float32)
        pack["closing_traj"] = closing_s.astype(np.float32)
        pack["speed_traj"] = speed_s.astype(np.float32)
        # The actually-sampled/executed acceleration -- was already computed every step
        # (used only for mean_hard_brake below) but never persisted as a trajectory
        # until now. For dynamics_model="classic" (what this pipeline uses), this is the
        # real physically-applied longitudinal acceleration, not a policy-expectation
        # proxy like exp_accel_traj -- use this for LongJ instead of exp_accel_traj when
        # available (see criticality_metrics.py's longitudinal_jerk_proxy).
        pack["accel_traj"] = accel_s.astype(np.float32)
        # Guard against silent frozen-ego rollouts (e.g. drive.ini failed → dt=0).
        mean_speed = float(np.nanmean(np.abs(speed_s)))
        if not np.isfinite(mean_speed) or mean_speed < 1e-3:
            raise RuntimeError(
                f"rollout_per_ego: ego speed_traj is ~0 (mean_abs={mean_speed:.3g}). "
                "Likely drive.ini failed to load (dt=0) or control is broken; "
                "refuse to save/use this pack."
            )
    if capture_obs and obs_traj is not None:
        pack["obs_traj"] = obs_traj

    if capture_pose:
        pack["ego_x_traj"] = ego_x_s
        pack["ego_y_traj"] = ego_y_s
        pack["ego_heading_traj"] = ego_heading_s
        pack["ego_length"] = ego_length
        pack["ego_width"] = ego_width
        # Nearest-partner pose. NOTE: the partner's own length/width are NOT captured
        # here -- other_id (see nearest_from_states_with_pose) is a raw simulator
        # entity index, a different numbering than agent_state's track-id-based "id",
        # so it cannot be safely matched against get_global_agent_state()'s output to
        # look up the partner's dimensions. That would need a small dedicated C-side
        # accessor keyed by entity index (not implemented here).
        pack["other_x_traj"] = other_x_s
        pack["other_y_traj"] = other_y_s
        pack["other_heading_traj"] = other_heading_s
        pack["other_id_traj"] = other_id_s
        pack["other_speed_traj"] = other_speed_s
        pack["other_length_traj"] = other_length_s
        pack["other_width_traj"] = other_width_s

    if capture_map_geometry:
        # Flat, NOT ego-indexed (unlike every other pack field): 'lengths'[k] is the
        # point-count of polyline k, and the corresponding x/y run is the next
        # lengths[k] entries of the flattened x/y arrays. Captured earlier (right after
        # reset, before the env closes) -- see map_geometry above.
        pack.update(map_geometry)

    if capture_readout:
        for key in readout_out.values():
            pack[f"{key}_traj"] = readout_s[key]

        pre_readout = {k: np.full(n_ego, np.nan, dtype=np.float64) for k in readout_out.values()}
        tight_readout = {k: np.full(n_ego, np.nan, dtype=np.float64) for k in readout_out.values()}
        pre_tight_min_dist = np.full(n_ego, np.nan, dtype=np.float64)
        pre_tight_ttc = np.full(n_ego, np.nan, dtype=np.float64)
        for i in range(n_ego):
            if had_tight[i]:
                e = _first_true(tight[i])
                if e >= 0:
                    for key in readout_out.values():
                        tight_readout[key][i] = float(readout_s[key][i, e])
                    lo = max(0, e - pre_window)
                    if e - lo >= 3:
                        for key in readout_out.values():
                            pre_readout[key][i] = float(readout_s[key][i, lo:e].mean())
                    pre_step = max(0, e - 1)
                    pre_tight_min_dist[i] = float(min_dist[i, pre_step])
                    ttc_val = ttc_s[i, pre_step]
                    if np.isfinite(ttc_val):
                        pre_tight_ttc[i] = float(ttc_val)

        for key in readout_out.values():
            pack[f"pre_entry_{key}"] = pre_readout[key]
            pack[f"tight_{key}"] = tight_readout[key]
        pack["pre_tight_min_dist"] = pre_tight_min_dist
        pack["pre_tight_ttc"] = pre_tight_ttc

    return pack


def save_ego_pack(path: Path, pack: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in pack.items()})


def load_ego_pack(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
