
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from common import DEFAULT_THRESHOLDS, N_STEER, jsonable, nearest_from_states
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

DT = 0.1
# accel=0 (idx 3), steer=0 (idx 6) — neither ReCord nor Reactive.
MAINTAIN_ACTION = 3 * N_STEER + 6

# Ego control for the physical rollout. All modes still forward every policy on
# the resulting identical obs history; only the state distribution changes.
EGO_MODES = ("maintain", "record", "reactive")
DEFAULT_DRIVER_SEED = 42

MAP_FILENAME = "map_{idx:03d}.bin"


def normalize_ego_mode(ego_mode: str) -> str:
    m = str(ego_mode).strip().lower()
    if m not in EGO_MODES:
        raise ValueError(f"ego_mode must be one of {EGO_MODES}, got {ego_mode!r}")
    return m


def collect_split_dir(replication_root: Path, ego_mode: str, split: str) -> Path:
    """maintain keeps the legacy flat layout; record/reactive go under ego_*.

    split: 'train_dataset' | 'validation_dataset'
    """
    mode = normalize_ego_mode(ego_mode)
    root = Path(replication_root)
    if mode == "maintain":
        return root / split
    return root / f"ego_{mode}" / split


def _new_lstm_state(n: int, hidden: int, device: str) -> dict[str, torch.Tensor]:
    return dict(
        lstm_h=torch.zeros(n, hidden, device=device),
        lstm_c=torch.zeros(n, hidden, device=device),
    )


def _logits_tensor(logits) -> torch.Tensor:
    t = logits[0] if isinstance(logits, (tuple, list)) else logits
    return t


def _set_ego_actions_from_logits(
    actions: np.ndarray,
    ego_idx: np.ndarray,
    logits,
    ego_t: torch.Tensor,
) -> None:
    """Write argmax actions for ego slots; leave others at maintain (experts ignore)."""
    act = _logits_tensor(logits).index_select(0, ego_t).argmax(dim=-1).cpu().numpy().astype(np.int64)
    ego_np = np.asarray(ego_idx)
    actions[:] = MAINTAIN_ACTION
    actions[ego_np] = act.reshape(-1, *actions[ego_np].shape[1:])


def _map_bin(map_dir: Path, idx: int) -> Path:
    return map_dir / MAP_FILENAME.format(idx=idx)


def _make_shard_map_dir(src_dir: Path, dest: Path, start: int, n: int) -> None:
    """Symlink maps [start, start+n) as map_000.bin ... so Drive can load them from 0."""
    dest.mkdir(parents=True, exist_ok=True)
    for local, src_i in enumerate(range(start, start + n)):
        src = _map_bin(src_dir, src_i)
        if not src.is_file():
            raise FileNotFoundError(f"missing map binary {src}")
        link = dest / MAP_FILENAME.format(idx=local)
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(src, link)


@torch.no_grad()
def collect_named_policies(
    *,
    policies: dict[str, Path],
    num_maps: int,
    device: str,
    data_mode: str,
    map_dir: str | None = None,
    map_id_offset: int = 0,
    ego_mode: str = "maintain",
    driver_name: str | None = None,
    driver_ckpt: Path | None = None,
) -> dict[str, np.ndarray]:
    """Log-replay partners; ego_mode controls ego; forward all named policies.

    When ego_mode is record/reactive, driver_name (+ ckpt) selects whose argmax
    steps the env. Driver may be outside the current GPU group.
    """
    if not policies:
        raise ValueError("policies is empty")
    ego_mode = normalize_ego_mode(ego_mode)
    if ego_mode != "maintain":
        if not driver_name or driver_ckpt is None:
            raise ValueError(f"ego_mode={ego_mode} requires driver_name and driver_ckpt")
    resolved = resolve_drive_map_dir(data_mode, map_dir)
    args = build_human_replay_drive_args(
        num_maps=num_maps, device=device, data_mode=data_mode, map_start=0, map_dir=resolved
    )
    vecenv = create_vecenv(args, env_name="puffer_drive")
    names = list(policies.keys())
    models = []
    states = []
    driver_pol = None
    driver_state = None
    driver_in_group = False
    try:
        driver = vecenv.driver_env
        sim_steps = int(args["env"]["episode_length"])
        obs, infos = vecenv.reset()
        ego_idx = ego_indices_from_reset(args, driver, infos)
        n_ego = int(ego_idx.size)
        num_agents = int(vecenv.observation_space.shape[0])
        for _name, ckpt in policies.items():
            pol = create_policy(args, vecenv, env_name="puffer_drive")
            pol.load_state_dict(pick_checkpoint(str(ckpt), device="cpu", probe_step=None).state_dict)
            pol.to(device).eval()
            models.append(pol)
            states.append(_new_lstm_state(num_agents, int(pol.hidden_size), device))
        if ego_mode != "maintain":
            if driver_name in policies:
                driver_in_group = True
                driver_pol = models[names.index(driver_name)]
                driver_state = states[names.index(driver_name)]
            else:
                driver_pol = create_policy(args, vecenv, env_name="puffer_drive")
                driver_pol.load_state_dict(
                    pick_checkpoint(str(driver_ckpt), device="cpu", probe_step=None).state_dict
                )
                driver_pol.to(device).eval()
                driver_state = _new_lstm_state(num_agents, int(driver_pol.hidden_size), device)
        hidden = int(models[0].hidden_size)
        try:
            scene_ids = agent_scenario_ids(driver, int(getattr(driver, "num_agents", num_agents)))[ego_idx]
        except Exception:
            scene_ids = np.arange(n_ego, dtype=np.int64)
        scene_ids = scene_ids.astype(np.int64) + int(map_id_offset)

        h_all = {n: np.zeros((n_ego, sim_steps, hidden), dtype=np.float32) for n in names}
        log_all = {n: np.zeros((n_ego, sim_steps, 91), dtype=np.float32) for n in names}
        min_dist = np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
        ttc_s = np.full((n_ego, sim_steps), np.nan, dtype=np.float32)
        closing_s = np.zeros((n_ego, sim_steps), dtype=np.float32)
        prev_xy = np.full((n_ego, 2), np.nan, dtype=np.float64)
        ttc_tight = DEFAULT_THRESHOLDS["ttc_tight"]
        closing_tight = DEFAULT_THRESHOLDS["closing_tight"]
        dist_tight = DEFAULT_THRESHOLDS["dist_tight"]
        ttc_approach = DEFAULT_THRESHOLDS["ttc_approach"]
        closing_approach = DEFAULT_THRESHOLDS["closing_approach"]
        dist_approach = DEFAULT_THRESHOLDS["dist_approach"]
        tight = np.zeros((n_ego, sim_steps), dtype=bool)
        approach = np.zeros((n_ego, sim_steps), dtype=bool)
        ego_t = torch.as_tensor(ego_idx, device=device, dtype=torch.long)
        actions = np.full(vecenv.action_space.shape, MAINTAIN_ACTION, dtype=np.int64)

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
            ob = torch.as_tensor(obs).to(device)
            drive_logits = None
            for name, pol, st in zip(names, models, states):
                logits, _ = pol.forward_eval(ob, st)
                h_all[name][:, t] = st["hidden"].index_select(0, ego_t).cpu().numpy()
                log_all[name][:, t] = _logits_tensor(logits).index_select(0, ego_t).cpu().numpy()
                if driver_in_group and name == driver_name:
                    drive_logits = logits
            if ego_mode == "maintain":
                actions[:] = MAINTAIN_ACTION
            else:
                if not driver_in_group:
                    drive_logits, _ = driver_pol.forward_eval(ob, driver_state)
                _set_ego_actions_from_logits(actions, ego_idx, drive_logits, ego_t)
            obs, _, _, _, _ = vecenv.step(actions)
    finally:
        safe_close_vecenv(vecenv)

    out = {
        "scene_id": scene_ids.astype(np.int64),
        "min_dist": min_dist,
        "ttc": ttc_s,
        "closing": closing_s,
        "tight": tight,
        "approach": approach,
        "data_mode": np.array(data_mode),
        "map_dir": np.array(resolved),
        "map_id_offset": np.array(int(map_id_offset)),
        "policy_names": np.array(names),
        "ego_mode": np.array(ego_mode),
        "driver_name": np.array("" if driver_name is None else driver_name),
    }
    for name in names:
        out[f"h_{name}"] = h_all[name]
        out[f"logits_{name}"] = log_all[name]
        out[f"ckpt_{name}"] = np.array(str(policies[name]))
    return out


def _shard_policy_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    z = np.load(path, allow_pickle=True)
    names = {k[2:] for k in z.files if k.startswith("h_")}
    if "policy_names" in z.files:
        names |= {str(x) for x in np.asarray(z["policy_names"]).reshape(-1).tolist()}
    return names


def _merge_named_shard(path: Path, pack: dict[str, np.ndarray]) -> None:
    """Add newly collected policy keys into an existing shard, keeping scene metadata."""
    if not path.is_file():
        np.savez(path, **pack)
        return
    z = np.load(path, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    old_names = [str(x) for x in np.asarray(data.get("policy_names", [])).reshape(-1).tolist()]
    new_names = [str(x) for x in np.asarray(pack["policy_names"]).reshape(-1).tolist()]
    merged = old_names + [n for n in new_names if n not in old_names]
    data["policy_names"] = np.array(merged)
    skip = {
        "scene_id",
        "min_dist",
        "ttc",
        "closing",
        "tight",
        "approach",
        "data_mode",
        "map_dir",
        "map_id_offset",
        "policy_names",
        "ego_mode",
        "driver_name",
    }
    for k, v in pack.items():
        if k in skip:
            continue
        data[k] = v
    np.savez(path, **data)


def collect_named_policies_sharded(
    *,
    policies: dict[str, Path],
    num_maps: int,
    device: str,
    data_mode: str,
    out_dir: Path,
    shard_size: int = 1000,
    map_dir: str | None = None,
    group_size: int = 4,
    ego_mode: str = "maintain",
    driver_name: str | None = None,
) -> None:
    """Collect policies in small GPU groups and merge keys into the same shards."""
    ego_mode = normalize_ego_mode(ego_mode)
    driver_ckpt = None
    if ego_mode != "maintain":
        if not driver_name or driver_name not in policies:
            raise ValueError(
                f"ego_mode={ego_mode} needs driver_name in policies; got {driver_name!r}"
            )
        driver_ckpt = policies[driver_name]
    src = Path(resolve_drive_map_dir(data_mode, map_dir))
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    items = list(policies.items())
    groups = [dict(items[i : i + int(group_size)]) for i in range(0, len(items), int(group_size))]
    (out_dir / "policy_names.json").write_text(
        json.dumps(
            {
                "names": list(policies.keys()),
                "ckpts": {k: str(v) for k, v in policies.items()},
                "group_size": int(group_size),
                "n_groups": len(groups),
                "ego_mode": ego_mode,
                "driver_name": driver_name,
            },
            indent=2,
        )
    )
    wanted = set(policies)
    for start in range(0, int(num_maps), int(shard_size)):
        n = min(int(shard_size), int(num_maps) - start)
        path = shard_dir / f"shard_{start:05d}_{n:04d}.npz"
        have = _shard_policy_names(path)
        missing_groups = [g for g in groups if not set(g).issubset(have)]
        if not missing_groups and wanted.issubset(have):
            print(f"  reuse shard {path.name} policies={sorted(have)}", flush=True)
            continue
        print(
            f"  collect maps [{start}, {start + n}) ego_mode={ego_mode} "
            f"missing_groups={len(missing_groups)} from {src}",
            flush=True,
        )
        tmp = out_dir / "_tmp_maps"
        if tmp.exists():
            shutil.rmtree(tmp)
        _make_shard_map_dir(src, tmp, start, n)
        try:
            for gi, g in enumerate(missing_groups):
                print(f"    group {gi + 1}/{len(missing_groups)} policies={list(g)}", flush=True)
                pack = collect_named_policies(
                    policies=g,
                    num_maps=n,
                    device=device,
                    data_mode=data_mode,
                    map_dir=str(tmp),
                    map_id_offset=start,
                    ego_mode=ego_mode,
                    driver_name=driver_name,
                    driver_ckpt=driver_ckpt,
                )
                _merge_named_shard(path, pack)
                del pack
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        have = _shard_policy_names(path)
        print(f"  wrote {path} policies={sorted(have)}", flush=True)


def load_named_pair(
    dataset_dir: Path,
    name_a: str,
    name_b: str,
) -> dict[str, np.ndarray]:
    """Load two named policies from multi-policy shards as a paired episode pack."""
    shard_dir = dataset_dir / "shards"
    shards = sorted(shard_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    parts = []
    for p in shards:
        z = np.load(p, allow_pickle=True)
        ka, kb = f"h_{name_a}", f"h_{name_b}"
        if ka not in z.files or kb not in z.files:
            raise KeyError(f"{p} missing {ka} or {kb}; have {list(z.files)[:12]}")
        parts.append(
            {
                "scene_id": z["scene_id"],
                "h_record": z[ka],
                "h_reactive": z[kb],
                "logits_record": z[f"logits_{name_a}"],
                "logits_reactive": z[f"logits_{name_b}"],
                "min_dist": z["min_dist"],
                "ttc": z["ttc"],
                "closing": z["closing"],
                "tight": z["tight"],
                "approach": z["approach"],
            }
        )
    return concat_episode_packs(parts)


def concat_episode_packs(packs: list[dict]) -> dict[str, np.ndarray]:
    keys = [
        "scene_id",
        "h_record",
        "h_reactive",
        "logits_record",
        "logits_reactive",
        "min_dist",
        "ttc",
        "closing",
        "tight",
        "approach",
    ]
    out = {k: np.concatenate([p[k] for p in packs], axis=0) for k in keys}
    last = packs[-1]
    for meta in ("rec_ckpt", "rea_ckpt", "data_mode", "map_dir", "ego_mode", "driver_name"):
        if meta in last:
            out[meta] = np.array(last[meta])
    return out


def flatten_pack(pack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """(n_ego, T, ...) → per-state rows with scene_id / timestep."""
    n, t = pack["h_record"].shape[:2]
    scene = np.repeat(pack["scene_id"][:, None], t, axis=1).reshape(-1)
    step = np.tile(np.arange(t, dtype=np.int32), n)
    return {
        "scene_id": scene,
        "timestep": step,
        "h_record": pack["h_record"].reshape(n * t, -1),
        "h_reactive": pack["h_reactive"].reshape(n * t, -1),
        "logits_record": pack["logits_record"].reshape(n * t, -1),
        "logits_reactive": pack["logits_reactive"].reshape(n * t, -1),
        "min_dist": pack["min_dist"].reshape(-1),
        "ttc": pack["ttc"].reshape(-1),
        "closing": pack["closing"].reshape(-1),
        "tight": pack["tight"].reshape(-1),
        "approach": pack["approach"].reshape(-1),
    }


def dataset_stats(pack: dict[str, np.ndarray], extra: dict | None = None) -> dict:
    tight = pack["tight"].astype(bool)
    ap = pack["approach"].astype(bool)
    n_scenes = int(pack["scene_id"].shape[0]) if pack["h_record"].ndim == 3 else int(np.unique(pack["scene_id"]).size)
    n_states = int(tight.size)
    stats = {
        "n_scenes": n_scenes,
        "n_states": n_states,
        "frac_tight": float(tight.mean()),
        "frac_approach": float(ap.mean()),
        "mean_min_dist": float(np.nanmean(pack["min_dist"])),
        "tight_definition": {
            "ttc": DEFAULT_THRESHOLDS["ttc_tight"],
            "closing": DEFAULT_THRESHOLDS["closing_tight"],
            "dist": DEFAULT_THRESHOLDS["dist_tight"],
        },
        "approach_definition": {
            "ttc": DEFAULT_THRESHOLDS["ttc_approach"],
            "closing": DEFAULT_THRESHOLDS["closing_approach"],
            "dist": DEFAULT_THRESHOLDS["dist_approach"],
        },
    }
    if extra:
        stats.update(extra)
    return jsonable(stats)


def collect_all_policy_pairs(
    *,
    out_root: Path,
    policies: dict[str, Path],
    num_train_maps: int = 10000,
    num_val_maps: int = 10000,
    shard_size: int = 1000,
    device: str = "cuda",
    force: bool = False,
    ego_mode: str = "maintain",
    driver_seed: int = DEFAULT_DRIVER_SEED,
) -> dict:
    ego_mode = normalize_ego_mode(ego_mode)
    driver_name = None
    if ego_mode == "record":
        driver_name = f"record_s{int(driver_seed)}"
    elif ego_mode == "reactive":
        driver_name = f"reactive_s{int(driver_seed)}"
    if driver_name is not None and driver_name not in policies:
        raise KeyError(f"driver {driver_name} not in policies {sorted(policies)}")

    base = out_root / "policy_seed_replication"
    results = {}
    for data_mode, n_maps, sub in (
        ("training", num_train_maps, "train_dataset"),
        ("validation", num_val_maps, "validation_dataset"),
    ):
        out = collect_split_dir(base, ego_mode, sub)
        shard_dir = out / "shards"
        if not force and shard_dir.is_dir() and any(shard_dir.glob("shard_*.npz")):
            print(f"[reuse] {out} (set force=True to regenerate)", flush=True)
            results[data_mode] = {"path": str(out), "reused": True, "ego_mode": ego_mode}
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(
            f"########## collect {data_mode} n={n_maps} ego_mode={ego_mode} "
            f"driver={driver_name} policies={sorted(policies)} ##########",
            flush=True,
        )
        collect_named_policies_sharded(
            policies=policies,
            num_maps=int(n_maps),
            device=device,
            data_mode=data_mode,
            out_dir=out,
            shard_size=int(shard_size),
            ego_mode=ego_mode,
            driver_name=driver_name,
        )
        keys = sorted(policies)
        if len(keys) >= 2:
            pack = load_named_pair(out, keys[0], keys[1])
            stats = dataset_stats(
                pack,
                extra={
                    "data_mode": data_mode,
                    "n_policies": len(policies),
                    "policy_keys": keys,
                    "ego_mode": ego_mode,
                    "driver_name": driver_name,
                    "driver_seed": int(driver_seed) if driver_name else None,
                },
            )
            (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
            print(json.dumps(stats, indent=2), flush=True)
        results[data_mode] = {
            "path": str(out),
            "reused": False,
            "n_policies": len(policies),
            "ego_mode": ego_mode,
            "driver_name": driver_name,
        }
    return results


def main() -> int:
    """Canonical activation collection for the final mechanism story."""
    import argparse
    import sys

    _coord = Path(__file__).resolve().parent.parent
    if str(_coord) not in sys.path:
        sys.path.insert(0, str(_coord))

    # Lazy import: replicate depends on collect at module level.
    from crosscoder.frozen_config import RESULTS_MECHANISM  # noqa: WPS433
    from crosscoder.replicate import _all_policies, phase_pairs  # noqa: WPS433

    p = argparse.ArgumentParser(
        description=(
            "Collect 10K train + 10K val paired activations for all matched policy seeds. "
            "ego_mode selects the physical ego controller; all policies still forward on "
            "the same obs. Primary/frozen path is maintain."
        )
    )
    p.add_argument("--out-root", default=str(RESULTS_MECHANISM))
    p.add_argument("--num-train-maps", type=int, default=10000)
    p.add_argument("--num-val-maps", type=int, default=10000)
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true", help="Regenerate even if shards exist")
    p.add_argument(
        "--ego-mode",
        default="maintain",
        choices=[*EGO_MODES, "all"],
        help="maintain (primary) | record | reactive | all three",
    )
    p.add_argument(
        "--driver-seed",
        type=int,
        default=DEFAULT_DRIVER_SEED,
        help="Which matched seed's policy drives ego for record/reactive modes",
    )
    args = p.parse_args()

    root = Path(args.out_root)
    pairs_doc = phase_pairs(root)
    if pairs_doc.get("missing_ckpts"):
        raise SystemExit(f"missing checkpoints: {pairs_doc['missing_ckpts']}")
    policies = _all_policies(pairs_doc)
    force = bool(args.force or int(os.environ.get("FORCE", "0")))
    modes = list(EGO_MODES) if args.ego_mode == "all" else [args.ego_mode]
    all_out = {}
    for mode in modes:
        print(f"===== ego_mode={mode} =====", flush=True)
        all_out[mode] = collect_all_policy_pairs(
            out_root=root,
            policies=policies,
            num_train_maps=args.num_train_maps,
            num_val_maps=args.num_val_maps,
            shard_size=args.shard_size,
            device=args.device,
            force=force,
            ego_mode=mode,
            driver_seed=args.driver_seed,
        )
    print(json.dumps({"collect": all_out, "n_policies": len(policies)}, indent=2), flush=True)
    print(
        "Done. pipeline/replicate read maintain at "
        f"{root / 'policy_seed_replication'}/{{train,validation}}_dataset; "
        "record/reactive robustness under "
        f"{root / 'policy_seed_replication'}/ego_{{record,reactive}}/.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
