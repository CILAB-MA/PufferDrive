#!/usr/bin/env python3
"""One-shot persistence of the interaction-sensitive patch.

duration=1 artificial transplant → R_U half-life (typically ≈ 1 step).

  python analyze/coordination/crosscoder/persistence.py --out-root ...
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

from common import DEFAULT_THRESHOLDS, nearest_from_states 
from crosscoder.collect import (
    MAINTAIN_ACTION,
    _logits_tensor,
    _make_shard_map_dir,
    _new_lstm_state,
)
from crosscoder.frozen_config import PRIMARY_ALPHA as ALPHA  
from crosscoder.frozen_config import FROZEN_K, RESULTS_MECHANISM  
from crosscoder.intervention import HIDDEN, load_pair_bases  
from crosscoder.metrics import bootstrap_ci, kl_softmax, orthonormal_basis  
from crosscoder.pipeline import _write_json  
from runtime import (  
    agent_scenario_ids,
    build_human_replay_drive_args,
    create_policy,
    create_vecenv,
    ego_indices_from_reset,
    pick_checkpoint,
    resolve_drive_map_dir,
    safe_close_vecenv,
)

BRANCH_T = 40
HORIZON = 40
DECAY_T = 21
EPS = 1e-8
SAVE_TAUS = (0, 1, 2, 3, 5, 10, 20)
CONDS = ("interaction", "random", "noninteraction", "sign_reverse")


def _curve(v: np.ndarray, mask: np.ndarray | None = None) -> dict:
    x = np.asarray(v, dtype=np.float64)
    out = {"mean": [], "median": [], "bootstrap_ci": [], "n": []}
    for t in range(x.shape[1]):
        col = x[:, t]
        if mask is not None:
            col = col[mask]
        col = col[np.isfinite(col)]
        out["n"].append(int(col.size))
        if col.size == 0:
            out["mean"].append(float("nan"))
            out["median"].append(float("nan"))
            out["bootstrap_ci"].append([float("nan"), float("nan")])
            continue
        out["mean"].append(float(col.mean()))
        out["median"].append(float(np.median(col)))
        out["bootstrap_ci"].append(bootstrap_ci(col))
    return out


def _at(curve: dict, taus=SAVE_TAUS) -> dict:
    return {
        str(t): {
            "mean": curve["mean"][t],
            "median": curve["median"][t],
            "bootstrap_ci": curve["bootstrap_ci"][t],
            "n": curve["n"][t],
        }
        for t in taus
        if t < len(curve["mean"])
    }


def _half_life(mean_r: list[float], thresh: float) -> float:
    for t, v in enumerate(mean_r):
        if t == 0:
            continue
        if np.isfinite(v) and v < thresh:
            return float(t)
    return float("nan")


def _make_delta(dlt: torch.Tensor, condition: str, u, qn, qr) -> torch.Tensor:
    d_u = (dlt @ u) @ u.T
    n_u = d_u.norm(dim=-1, keepdim=True).clamp_min(EPS)
    if condition == "interaction":
        return d_u
    if condition == "sign_reverse":
        return -d_u
    if condition == "random":
        d = (dlt @ qr) @ qr.T
        return d * (n_u / d.norm(dim=-1, keepdim=True).clamp_min(EPS))
    if condition == "noninteraction":
        d = (dlt @ qn) @ qn.T
        return d * (n_u / d.norm(dim=-1, keepdim=True).clamp_min(EPS))
    if condition in ("record", "reactive", "nopatch"):
        return torch.zeros_like(dlt)
    raise ValueError(condition)


def _setup_env(rec_ckpt: Path, rea_ckpt: Path, num_maps: int, device: str, map_dir: str):
    args = build_human_replay_drive_args(
        num_maps=num_maps, device=device, data_mode="validation", map_dir=map_dir, map_start=0
    )
    vecenv = create_vecenv(args, env_name="puffer_drive")
    rec = create_policy(args, vecenv, env_name="puffer_drive")
    rea = create_policy(args, vecenv, env_name="puffer_drive")
    rec.load_state_dict(pick_checkpoint(str(rec_ckpt), device="cpu", probe_step=None).state_dict)
    rea.load_state_dict(pick_checkpoint(str(rea_ckpt), device="cpu", probe_step=None).state_dict)
    rec.to(device).eval()
    rea.to(device).eval()
    return args, vecenv, rec, rea


@torch.no_grad()
def rollout(
    *,
    args,
    vecenv,
    rec,
    rea,
    device: str,
    u: np.ndarray,
    q_non: np.ndarray,
    q_rand: np.ndarray,
    condition: str,
    duration: int,
    alpha: float,
    branch_t: int,
    horizon: int,
) -> dict:
    """Maintain prefix, then ego-controlled rollout with optional one-shot/maintained patch."""
    driver = vecenv.driver_env
    obs, infos = vecenv.reset()
    ego_idx = ego_indices_from_reset(args, driver, infos)
    n_ego = int(ego_idx.size)
    num_agents = int(vecenv.observation_space.shape[0])
    print(f"    n_ego={n_ego} cond={condition} dur={duration}", flush=True)
    rec_st = _new_lstm_state(num_agents, int(rec.hidden_size), device)
    shadow_st = _new_lstm_state(num_agents, int(rea.hidden_size), device)
    patch_st = _new_lstm_state(num_agents, int(rea.hidden_size), device)
    try:
        scene_ids = agent_scenario_ids(driver, int(getattr(driver, "num_agents", num_agents)))[ego_idx]
    except Exception:
        scene_ids = np.arange(n_ego, dtype=np.int64)
    ego_t = torch.as_tensor(ego_idx, device=device, dtype=torch.long)
    actions = np.full(vecenv.action_space.shape, MAINTAIN_ACTION, dtype=np.int64)
    th = DEFAULT_THRESHOLDS
    prev_xy = np.full((n_ego, 2), np.nan, dtype=np.float64)
    for _ in range(int(branch_t)):
        ob = torch.as_tensor(obs).to(device)
        rec.forward_eval(ob, rec_st)
        rea.forward_eval(ob, shadow_st)
        rea.forward_eval(ob, patch_st)
        ag = driver.get_global_agent_state()
        for li, g in enumerate(ego_idx):
            prev_xy[li] = [ag["x"][int(g)], ag["y"][int(g)]]
        obs, _, _, _, _ = vecenv.step(actions)

    u_t = torch.from_numpy(u.astype(np.float32)).to(device)
    qn_t = torch.from_numpy(q_non.astype(np.float32)).to(device)
    qr_t = torch.from_numpy(q_rand.astype(np.float32)).to(device)
    hid = int(rec.hidden_size)
    h_rec = np.zeros((n_ego, horizon, hid), dtype=np.float32)
    h_shadow = np.zeros((n_ego, horizon, hid), dtype=np.float32)
    h_ctrl = np.zeros((n_ego, horizon, hid), dtype=np.float32)
    log_rec = np.zeros((n_ego, horizon, 91), dtype=np.float32)
    log_sh = np.zeros((n_ego, horizon, 91), dtype=np.float32)
    log_ctrl = np.zeros((n_ego, horizon, 91), dtype=np.float32)
    xy = np.zeros((n_ego, horizon + 1, 2), dtype=np.float32)
    acts = np.zeros((n_ego, horizon), dtype=np.int64)
    r_rel = np.zeros((n_ego, horizon), dtype=np.float32)
    tight0 = np.zeros(n_ego, dtype=bool)
    du0 = np.zeros(n_ego, dtype=np.float32)

    for h in range(horizon):
        ob = torch.as_tensor(obs).to(device)
        rec.forward_eval(ob, rec_st)
        rea.forward_eval(ob, shadow_st)
        rea.forward_eval(ob, patch_st)
        h_r, h_s, h_p = rec_st["hidden"], shadow_st["hidden"], patch_st["hidden"]
        dlt = h_r - h_s
        if h == 0:
            du0 = (dlt.index_select(0, ego_t) @ u_t).norm(dim=-1).cpu().numpy().astype(np.float32)
        rel_scale = torch.zeros(h_p.shape[0], 1, device=device)
        if h < int(duration) and condition not in ("record", "reactive", "nopatch"):
            delta = _make_delta(dlt, condition, u_t, qn_t, qr_t)
            h_p = h_s + float(alpha) * delta
            patch_st["hidden"] = patch_st["lstm_h"] = h_p
            rel_scale = (float(alpha) * delta.norm(dim=-1, keepdim=True)) / (h_s.norm(dim=-1, keepdim=True) + EPS)
        if condition == "record":
            ctrl_h, ctrl_pol = rec_st["hidden"], rec
        elif condition in ("reactive", "nopatch"):
            ctrl_h, ctrl_pol = shadow_st["hidden"], rea
        else:
            ctrl_h, ctrl_pol = patch_st["hidden"], rea
        lr = _logits_tensor(rec.policy.decode_actions(rec_st["hidden"])[0])
        ls = _logits_tensor(rea.policy.decode_actions(shadow_st["hidden"])[0])
        lc = _logits_tensor(ctrl_pol.policy.decode_actions(ctrl_h)[0])
        h_rec[:, h] = h_r.index_select(0, ego_t).cpu().numpy()
        h_shadow[:, h] = h_s.index_select(0, ego_t).cpu().numpy()
        h_ctrl[:, h] = ctrl_h.index_select(0, ego_t).cpu().numpy()
        log_rec[:, h] = lr.index_select(0, ego_t).float().cpu().numpy()
        log_sh[:, h] = ls.index_select(0, ego_t).float().cpu().numpy()
        log_ctrl[:, h] = lc.index_select(0, ego_t).float().cpu().numpy()
        r_rel[:, h] = rel_scale.index_select(0, ego_t).squeeze(-1).cpu().numpy()
        act = lc.index_select(0, ego_t).argmax(dim=-1).cpu().numpy().astype(np.int64)
        acts[:, h] = act
        actions[:] = MAINTAIN_ACTION
        ego_np = np.asarray(ego_idx)
        actions[ego_np] = act.reshape(-1, *actions[ego_np].shape[1:])
        ag = driver.get_global_agent_state()
        xs = np.asarray(ag["x"])[ego_np]
        ys = np.asarray(ag["y"])[ego_np]
        xy[:, h, 0], xy[:, h, 1] = xs, ys
        if h == 0:
            partner = driver.get_global_partner_state()
            heading = np.asarray(ag["heading"])[ego_np]
            for li, g in enumerate(ego_idx):
                g = int(g)
                now = np.array([xs[li], ys[li]], dtype=np.float64)
                spd = float(np.linalg.norm(now - prev_xy[li]) / 0.1) if np.isfinite(prev_xy[li]).all() else 0.0
                d, cl, ttc = nearest_from_states(
                    now,
                    float(heading[li]),
                    spd,
                    np.stack([partner["x"][g], partner["y"][g]], axis=-1).astype(np.float64),
                    partner["heading"][g],
                    partner["speed"][g],
                    partner["other_id"][g],
                )
                finite = np.isfinite(ttc) and np.isfinite(d)
                tight0[li] = bool(
                    finite and cl >= th["closing_tight"] and ttc < th["ttc_tight"] and d < th["dist_tight"]
                )
                prev_xy[li] = now
        else:
            prev_xy[:, 0], prev_xy[:, 1] = xs, ys
        obs, _, _, _, _ = vecenv.step(actions)

    ag = driver.get_global_agent_state()
    for li, g in enumerate(ego_idx):
        xy[li, -1, 0] = ag["x"][int(g)]
        xy[li, -1, 1] = ag["y"][int(g)]
    return {
        "scene_id": scene_ids.astype(np.int64),
        "h_record": h_rec,
        "h_shadow": h_shadow,
        "h_ctrl": h_ctrl,
        "logits_record": log_rec,
        "logits_shadow": log_sh,
        "logits_ctrl": log_ctrl,
        "xy": xy,
        "actions": acts,
        "rel_norm": r_rel,
        "tight0": tight0,
        "du0": du0,
    }


def persistence_from_packs(patch: dict, reactive: dict, record: dict, u: np.ndarray, decay_t: int = DECAY_T) -> dict:
    t = min(decay_t, patch["h_ctrl"].shape[1], reactive["h_ctrl"].shape[1], record["h_ctrl"].shape[1])
    hp, ha, hr = patch["h_ctrl"][:, :t], reactive["h_ctrl"][:, :t], record["h_ctrl"][:, :t]
    hs = patch["h_shadow"][:, :t]
    e = (hp - ha) @ u
    e_s = (hp - hs) @ u
    d_nat = (hr - ha) @ u
    p_u = np.linalg.norm(e, axis=-1)
    p_us = np.linalg.norm(e_s, axis=-1)
    r_u = p_u / (p_u[:, :1] + EPS)
    r_us = p_us / (p_us[:, :1] + EPS)
    r_full = np.linalg.norm(hp - ha, axis=-1) / (np.linalg.norm(hp[:, :1] - ha[:, :1], axis=-1, keepdims=True) + EPS)
    a_u = np.sum(e * d_nat, axis=-1) / ((np.linalg.norm(e, axis=-1) * np.linalg.norm(d_nat, axis=-1)) + EPS)
    dkl = kl_softmax(record["logits_ctrl"][:, :t], reactive["logits_ctrl"][:, :t]) - kl_softmax(
        record["logits_ctrl"][:, :t], patch["logits_ctrl"][:, :t]
    )
    dkl0 = dkl[:, 0:1]
    ok = dkl0[:, 0] > 0
    r_kl = np.full_like(dkl, np.nan)
    r_kl[ok] = dkl[ok] / (dkl0[ok] + EPS)
    return {
        "R_U": r_u.astype(np.float32),
        "R_U_sameobs": r_us.astype(np.float32),
        "R_full": r_full.astype(np.float32),
        "A_U": a_u.astype(np.float32),
        "dkl": dkl.astype(np.float32),
        "R_KL": r_kl.astype(np.float32),
        "tight0": patch["tight0"],
        "du0": patch["du0"],
        "scene_id": patch["scene_id"],
        "rel_norm": patch["rel_norm"][:, :t],
    }


def summarize_pers(p: dict, q90: float, q50: float) -> dict:
    tight, du = p["tight0"], p["du0"]
    high, low = du >= q90, du <= q50
    masks = {"all": None, "tight": tight, "nominal": ~tight, "high_DU": high, "low_DU": low}
    rec: dict = {}
    for name, key in (
        ("representation_decay", "R_U"),
        ("sameobs_decay", "R_U_sameobs"),
        ("full_hidden_decay", "R_full"),
        ("direction_alignment", "A_U"),
        ("action_effect_decay", "dkl"),
    ):
        rec[name] = {mname: _curve(p[key], mask) for mname, mask in masks.items()}
        rec[name]["at"] = _at(rec[name]["all"])
    rec["half_life_RU"] = _half_life(rec["representation_decay"]["all"]["mean"], 0.5)
    rec["one_over_e_RU"] = _half_life(rec["representation_decay"]["all"]["mean"], float(math.e**-1))
    rec["half_life_RU_tight"] = _half_life(rec["representation_decay"]["tight"]["mean"], 0.5)
    dkl_mean = rec["action_effect_decay"]["all"]["mean"]
    rec["half_life_dkl"] = _half_life([x / (dkl_mean[0] + EPS) for x in dkl_mean], 0.5)
    rec["n_tight"] = int(tight.sum())
    rec["median_rel_norm_t0"] = float(np.median(p["rel_norm"][:, 0]))
    return rec


def traj_delta_h40(patch: dict, reactive: dict, record: dict) -> dict:
    H = 40
    d_base = np.linalg.norm(reactive["xy"][:, 1 : H + 1] - record["xy"][:, 1 : H + 1], axis=-1).mean(-1)
    d_p = np.linalg.norm(patch["xy"][:, 1 : H + 1] - record["xy"][:, 1 : H + 1], axis=-1).mean(-1)
    delta = d_base - d_p
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "bootstrap_ci": bootstrap_ci(delta),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default=str(RESULTS_MECHANISM))
    p.add_argument("--device", default="cuda")
    p.add_argument("--maps", type=int, default=1000)
    args = p.parse_args()

    root = Path(args.out_root)
    out = root / "intervention_persistence"
    (out / "one_shot").mkdir(parents=True, exist_ok=True)

    pairs_doc = json.loads((root / "policy_seed_replication" / "policy_pairs.json").read_text())
    thr = json.loads((root / "intervention" / "divergence_bins" / "thresholds.json").read_text())
    src = Path(resolve_drive_map_dir("validation", None))
    tmp = out / "_maps"
    _make_shard_map_dir(src, tmp, 0, int(args.maps))

    rng = np.random.default_rng(0)
    q_rand = orthonormal_basis(rng.normal(size=(FROZEN_K, HIDDEN)))
    if q_rand.shape[1] < FROZEN_K:
        pad = np.zeros((HIDDEN, FROZEN_K), dtype=np.float64)
        pad[:, : q_rand.shape[1]] = q_rand
        q_rand = pad
    else:
        q_rand = q_rand[:, :FROZEN_K]

    cfg = {
        "alpha": ALPHA,
        "subspace_rank": FROZEN_K,
        "decay_horizon": DECAY_T - 1,
        "patch_duration": 1,
        "maps": int(args.maps),
        "branch_t": BRANCH_T,
    }
    _write_json(out / "config.json", cfg)

    all_pairs: dict = {}
    for pair in pairs_doc["selected_pairs"]:
        pid = pair["pair_id"]
        print(f"########## persistence {pid} ##########", flush=True)
        bases = load_pair_bases(root, pid)
        env_args, vecenv, rec, rea = _setup_env(
            Path(pair["record"]["ckpt"]), Path(pair["reactive"]["ckpt"]), int(args.maps), args.device, str(tmp)
        )
        kw = dict(
            args=env_args,
            vecenv=vecenv,
            rec=rec,
            rea=rea,
            device=args.device,
            u=bases["U"],
            q_non=bases["Q_nonint_rea"],
            q_rand=q_rand,
            alpha=ALPHA,
            branch_t=BRANCH_T,
            horizon=HORIZON,
        )
        packs = {c: rollout(condition=c, duration=0, **kw) for c in ("record", "reactive", "nopatch")}
        err_h = float(np.max(np.abs(packs["nopatch"]["h_ctrl"] - packs["reactive"]["h_ctrl"])))
        print(f"  nopatch vs reactive max|h|={err_h:.4g}", flush=True)

        q90, q50 = float(thr[pid]["q90"]), float(thr[pid]["q50"])
        one_shot: dict = {"nopatch_vs_reactive_max_h": err_h}
        closed: dict = {}
        for cond in CONDS:
            print(f"  duration=1 {cond}", flush=True)
            pk = rollout(condition=cond, duration=1, **kw)
            pers = persistence_from_packs(pk, packs["reactive"], packs["record"], bases["U"])
            summ = summarize_pers(pers, q90, q50)
            one_shot[cond] = summ
            closed[cond] = {"H40_delta_D": traj_delta_h40(pk, packs["reactive"], packs["record"])}
            if cond == "interaction":
                np.savez_compressed(
                    out / "one_shot" / f"seed_{pair['train_seed']}_{cond}.npz",
                    R_U=pers["R_U"],
                    dkl=pers["dkl"],
                    tight0=pers["tight0"],
                    du0=pers["du0"],
                    scene_id=pers["scene_id"],
                )

        all_pairs[pid] = {"one_shot": one_shot, "closed_loop": closed}
        seed_dir = out / "one_shot" / f"seed_{pair['train_seed']}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        _write_json(seed_dir / "metrics.json", one_shot)
        print(
            f"  half-life R_U={one_shot['interaction']['half_life_RU']} "
            f"ΔKL0={one_shot['interaction']['action_effect_decay']['all']['mean'][0]:.3f}",
            flush=True,
        )
        safe_close_vecenv(vecenv)
        del rec, rea, vecenv

    hls = [all_pairs[p]["one_shot"]["interaction"]["half_life_RU"] for p in all_pairs]
    across = {
        "half_life_RU": {
            "per_pair": hls,
            "mean": float(np.nanmean(hls)),
        },
        "dkl0": {
            "per_pair": [all_pairs[p]["one_shot"]["interaction"]["action_effect_decay"]["all"]["mean"][0] for p in all_pairs],
        },
        "dkl5": {
            "per_pair": [all_pairs[p]["one_shot"]["interaction"]["action_effect_decay"]["all"]["mean"][5] for p in all_pairs],
        },
    }
    summary = {
        "config": cfg,
        "one_shot": {pid: all_pairs[pid]["one_shot"]["interaction"] for pid in all_pairs},
        "controls": {
            pid: {c: _at(all_pairs[pid]["one_shot"][c]["action_effect_decay"]["all"]) for c in CONDS if c != "interaction"}
            for pid in all_pairs
        },
        "closed_loop": {pid: all_pairs[pid]["closed_loop"] for pid in all_pairs},
        "across_pairs": across,
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "one_shot" / "summary.json", summary["one_shot"])
    print(json.dumps(across, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
