#!/usr/bin/env python3
"""Checkpoint resolution, env/policy setup, and policy forward helpers."""

from __future__ import annotations

import glob
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

_CKPT_RE = re.compile(r"^model_.+_(\d{6})\.pt$")
_FINAL_CKPT_RE = re.compile(r"^puffer_drive_(.+)\.pt$")
_RUN_DIR_RE = re.compile(r"^puffer_drive_(.+)$")


@dataclass(frozen=True)
class Checkpoint:
    step: int
    path: str
    state_dict: dict | None = None


def checkpoint_step(path: str) -> int:
    match = _CKPT_RE.match(os.path.basename(path))
    if not match:
        raise ValueError(f"Not a step checkpoint: {path}")
    return int(match.group(1))


def is_flat_policy_ckpt(path: str) -> bool:
    return os.path.isfile(path) and bool(_FINAL_CKPT_RE.match(os.path.basename(path)))


def find_final_policy_ckpts(exp_path: str) -> list[tuple[str, str]]:
    exp_path = os.path.abspath(exp_path)
    if not os.path.isdir(exp_path):
        raise FileNotFoundError(f"Experiment path does not exist: {exp_path}")
    runs: list[tuple[str, str]] = []
    for path in sorted(glob.glob(os.path.join(exp_path, "puffer_drive_*.pt"))):
        if not os.path.isfile(path):
            continue
        match = _FINAL_CKPT_RE.match(os.path.basename(path))
        if match:
            runs.append((match.group(1), path))
    return runs


def resolve_final_policy_ckpt(exp_path: str, *, sweep_id: str | None = None) -> str:
    finals = find_final_policy_ckpts(exp_path)
    if not finals:
        raise FileNotFoundError(f"No flat puffer_drive_*.pt checkpoints under {exp_path}")
    if sweep_id is not None:
        for sid, path in finals:
            if sid == sweep_id:
                return path
        ids = ", ".join(s for s, _ in finals)
        raise FileNotFoundError(f"No puffer_drive_{sweep_id}.pt under {exp_path} (have: {ids})")
    if len(finals) == 1:
        return finals[0][1]
    ids = ", ".join(s for s, _ in finals)
    raise ValueError(f"Multiple flat puffer_drive_*.pt under {exp_path}; pass sweep_id. Found: {ids}")


def resolve_run_dir(path: str, run_name: str | None = None) -> str:
    path = os.path.abspath(path)
    if run_name:
        path = os.path.join(path, run_name)
    if glob.glob(os.path.join(path, "model_*.pt")):
        return path
    run_dirs = sorted(
        d
        for d in glob.glob(os.path.join(path, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "model_*.pt"))
    )
    if len(run_dirs) == 1:
        return run_dirs[0]
    if not run_dirs:
        raise FileNotFoundError(f"No model_*.pt checkpoints under {path}")
    names = ", ".join(os.path.basename(d) for d in run_dirs)
    raise ValueError(f"Multiple run dirs under {path}; pass run_name. Found: {names}")


def resolve_policy_location(
    base_path: str,
    exp_name: str,
    location: str | None = None,
    *,
    sweep_id: str | None = None,
    prefer_final: bool = True,
) -> str:
    if location:
        loc = os.path.abspath(location)
        if is_flat_policy_ckpt(loc):
            return loc
        if os.path.isdir(loc) and glob.glob(os.path.join(loc, "model_*.pt")):
            return loc
        if os.path.isfile(loc) and loc.endswith(".pt"):
            return loc
        raise FileNotFoundError(f"Not a policy checkpoint: {loc}")

    exp_path = os.path.join(os.path.abspath(base_path), exp_name)
    if prefer_final:
        try:
            return resolve_final_policy_ckpt(exp_path, sweep_id=sweep_id)
        except FileNotFoundError:
            pass
        raise FileNotFoundError(
            f"No flat puffer_drive_*.pt under {exp_path}. "
            "Export final weights to the experiment root or pass an explicit --run-dir / --policy-run-dirs path."
        )
    return resolve_run_dir(exp_path)


def sweep_name_from_run_dir(run_dir: str) -> str:
    match = _RUN_DIR_RE.match(os.path.basename(os.path.abspath(run_dir)))
    if not match:
        raise ValueError(f"Run dir does not match puffer_drive_{{sweep_name}}: {run_dir}")
    return match.group(1)


def find_run_dirs(exp_path: str, prefix: str = "puffer_drive_") -> list[tuple[str, str]]:
    exp_path = os.path.abspath(exp_path)
    if not os.path.isdir(exp_path):
        raise FileNotFoundError(f"Experiment path does not exist: {exp_path}")
    runs: list[tuple[str, str]] = []
    for entry in sorted(glob.glob(os.path.join(exp_path, f"{prefix}*"))):
        if not os.path.isdir(entry):
            continue
        if not glob.glob(os.path.join(entry, "model_*.pt")):
            continue
        runs.append((os.path.basename(entry)[len(prefix) :], entry))
    if not runs:
        raise FileNotFoundError(f"No puffer_drive_* run dirs with checkpoints under {exp_path}")
    return runs


def list_checkpoints(run_dir: str) -> list[tuple[int, str]]:
    run_dir = resolve_run_dir(run_dir)
    paths = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No model_*.pt checkpoints in {run_dir}")
    steps_and_paths = [(checkpoint_step(p), p) for p in paths]
    steps_and_paths.sort(key=lambda x: x[0])
    return steps_and_paths


def iter_checkpoints(
    run_dir: str,
    *,
    device: str | torch.device = "cpu",
    load_state: bool = True,
    map_location: str | torch.device | None = None,
) -> Iterator[Checkpoint]:
    if map_location is None:
        map_location = device
    for step, path in list_checkpoints(run_dir):
        state_dict = None
        if load_state:
            state_dict = torch.load(path, map_location=map_location, weights_only=True)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        yield Checkpoint(step=step, path=path, state_dict=state_dict)


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> dict:
    state_dict = torch.load(path, map_location=device, weights_only=True)
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


DRIVE_MAP_DIRS = {
    "training": "/data/puffer/resources/drive/binaries/training",
    "validation": "/data/puffer/resources/drive/binaries/validation",
}


def resolve_drive_map_dir(data_mode: str, map_dir: str | None = None) -> str:
    if map_dir:
        return str(map_dir)
    if data_mode not in DRIVE_MAP_DIRS:
        raise ValueError(f"data_mode must be training|validation, got {data_mode!r}")
    return DRIVE_MAP_DIRS[data_mode]


def build_human_replay_drive_args(
    config_path: str | None = None,
    *,
    num_maps: int = 300,
    device: str = "cuda",
    data_mode: str = "training",
    map_start: int = 0,
    map_dir: str | None = None,
) -> dict:
    from pufferlib.pufferl import load_config

    env_name = "puffer_drive"
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]
        args = load_config(env_name, config_dir=config_path)
    finally:
        sys.argv = saved_argv

    if map_start != 0:
        # Drive.__init__ (pufferlib/ocean/drive/drive.py) has no map_start parameter, and
        # the C binding's map_idx (binding.c) always starts at 0 -- there is no native
        # starting-offset concept to forward this to. Fail loudly rather than silently
        # ignoring the requested shard offset (mechanism.py's sharding callers need this
        # implemented, e.g. via a C-side offset or a map_dir subset, before map_start != 0
        # can work).
        raise NotImplementedError(
            f"map_start={map_start} requested, but the Drive env has no starting-map-offset "
            "support (see binding.c's map_idx). Only map_start=0 works today."
        )

    if data_mode not in ("training", "validation"):
        raise ValueError(f"data_mode must be training|validation, got {data_mode!r}")
    map_section = args.get(data_mode) or {}
    args["env"]["map_dir"] = (
        str(map_dir)
        if map_dir
        else (map_section.get("map_dir") or resolve_drive_map_dir(data_mode))
    )
    args["env"]["num_maps"] = num_maps
    args["env"]["sequential_map_sampling"] = True
    args["env"]["episode_length"] = 91
    args["env"]["termination_mode"] = 0
    args["env"]["control_mode"] = args["eval"].get("human_replay_control_mode", "control_sdc_only")
    args["vec"] = dict(backend=args["eval"].get("backend", "PufferEnv"), num_envs=1)
    args["train"]["device"] = device
    args["load_model_path"] = None
    return args


def pick_checkpoint(location: str, *, device: str, probe_step: int | None):
    if is_flat_policy_ckpt(location) or (location.endswith(".pt") and os.path.isfile(location)):
        state_dict = load_checkpoint(location, device="cpu")
        step = int(probe_step) if probe_step is not None else 0
        return Checkpoint(step=step, path=location, state_dict=state_dict)

    checkpoints = list(iter_checkpoints(location, device="cpu", map_location="cpu", load_state=True))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {location}")
    if probe_step is None:
        return checkpoints[-1]
    for ckpt in checkpoints:
        if ckpt.step == probe_step:
            return ckpt
    available = ", ".join(f"{c.step:06d}" for c in checkpoints)
    raise FileNotFoundError(f"No checkpoint step {probe_step:06d} in {location} (have: {available})")


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

    policy = load_policy({**args, "load_model_path": None}, vecenv, env_name)
    policy.eval()
    return policy


def ego_indices_from_reset(args: dict, driver, infos) -> np.ndarray:
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


def agent_scenario_ids(driver, num_agents: int) -> np.ndarray:
    ao = np.asarray(driver.agent_offsets, dtype=np.int64)
    map_ids = np.asarray(driver.map_ids, dtype=np.int32)
    out = np.zeros(num_agents, dtype=np.int32)
    for env_i in range(len(map_ids)):
        out[ao[env_i] : ao[env_i + 1]] = map_ids[env_i]
    return out


def load_policy_from_ckpt(ckpt: Path, device: str) -> torch.nn.Module:
    args = build_human_replay_drive_args(num_maps=1, device=device)
    vecenv = create_vecenv(args)
    policy = create_policy(args, vecenv)
    ck = pick_checkpoint(str(ckpt), device="cpu", probe_step=None)
    policy.load_state_dict(ck.state_dict)
    policy.to(device).eval()
    safe_close_vecenv(vecenv)
    return policy


