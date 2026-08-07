#!/usr/bin/env python3
"""Same-map per-ego rollout packs for coordination analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from common import DEFAULT_THRESHOLDS, accel_from_actions, action_stats_from_logits, nearest_from_states
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
    brake_logit_bias: float = 0.0,
    bias_onset: np.ndarray | None = None,
    bias_tau_lo: int = -15,
    bias_tau_hi: int = -6,
) -> dict[str, np.ndarray]:
    import pufferlib
    from common import N_ACTIONS, N_STEER

    if data_mode not in ("training", "validation"):
        raise ValueError(f"data_mode must be training|validation, got {data_mode!r}")
    args = build_human_replay_drive_args(num_maps=num_maps, device=device, data_mode=data_mode)
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
