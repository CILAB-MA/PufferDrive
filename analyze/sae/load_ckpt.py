"""Load per-epoch checkpoints from a PufferLib training run directory.

Example run dir:
  /data/nocturne/puffer/experiments/reactive_0.25/puffer_drive_pbt_da3kfc4q

Checkpoints are saved as:
  model_{env_name}_{update:06d}.pt
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Iterator

import torch

_CKPT_RE = re.compile(r"^model_.+_(\d{6})\.pt$")


def checkpoint_step(path: str) -> int:
    """Parse training update/epoch from a checkpoint filename."""
    name = os.path.basename(path)
    match = _CKPT_RE.match(name)
    if not match:
        raise ValueError(f"Not a step checkpoint: {path}")
    return int(match.group(1))


def resolve_run_dir(path: str, run_name: str | None = None) -> str:
    """Resolve a run directory that contains model_*.pt checkpoints.

    Accepts either:
      - the run dir itself (…/puffer_drive_pbt_da3kfc4q)
      - an experiment parent plus optional run_name
      - an experiment parent with exactly one run subdir
    """
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


_RUN_DIR_RE = re.compile(r"^puffer_drive_(.+)$")


def sweep_name_from_run_dir(run_dir: str) -> str:
    """Extract sweep_name from a run directory named puffer_drive_{sweep_name}."""
    name = os.path.basename(os.path.abspath(run_dir))
    match = _RUN_DIR_RE.match(name)
    if not match:
        raise ValueError(f"Run dir does not match puffer_drive_{{sweep_name}}: {run_dir}")
    return match.group(1)


def find_run_dirs(exp_path: str, prefix: str = "puffer_drive_") -> list[tuple[str, str]]:
    """Find run dirs under an experiment path.

    Returns:
        List of (sweep_name, run_dir) sorted by sweep_name.
        Only includes directories (not .pt files) that contain model_*.pt.
    """
    exp_path = os.path.abspath(exp_path)
    if not os.path.isdir(exp_path):
        raise FileNotFoundError(f"Experiment path does not exist: {exp_path}")

    runs: list[tuple[str, str]] = []
    for entry in sorted(glob.glob(os.path.join(exp_path, f"{prefix}*"))):
        if not os.path.isdir(entry):
            continue
        if not glob.glob(os.path.join(entry, "model_*.pt")):
            continue
        sweep_name = os.path.basename(entry)[len(prefix) :]
        runs.append((sweep_name, entry))

    if not runs:
        raise FileNotFoundError(f"No puffer_drive_* run dirs with checkpoints under {exp_path}")
    return runs


def list_checkpoints(run_dir: str) -> list[tuple[int, str]]:
    """Return (step, path) pairs sorted by training step."""
    run_dir = resolve_run_dir(run_dir)
    paths = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No model_*.pt checkpoints in {run_dir}")

    steps_and_paths = [(checkpoint_step(p), p) for p in paths]
    steps_and_paths.sort(key=lambda x: x[0])
    return steps_and_paths


@dataclass(frozen=True)
class Checkpoint:
    step: int
    path: str
    state_dict: dict | None = None


def iter_checkpoints(
    run_dir: str,
    *,
    device: str | torch.device = "cpu",
    load_state: bool = True,
    map_location: str | torch.device | None = None,
) -> Iterator[Checkpoint]:
    """Yield checkpoints in step order.

    Args:
        run_dir: Run directory or experiment parent (see resolve_run_dir).
        device: Target device when load_state=True.
        load_state: If True, torch.load each checkpoint into state_dict.
        map_location: Passed to torch.load (defaults to device).
    """
    if map_location is None:
        map_location = device

    for step, path in list_checkpoints(run_dir):
        state_dict = None
        if load_state:
            state_dict = torch.load(path, map_location=map_location, weights_only=True)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        yield Checkpoint(step=step, path=path, state_dict=state_dict)


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> dict:
    """Load a single checkpoint state_dict."""
    state_dict = torch.load(path, map_location=device, weights_only=True)
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="List checkpoints for a training run")
    parser.add_argument("--base-path", type=str, default="/data/puffer/experiments")
    parser.add_argument("--exp-name", type=str, default="reactive_0.25")
    args = parser.parse_args()

    exp_path = os.path.join(args.base_path, args.exp_name)
    for sweep_name, run_dir in find_run_dirs(exp_path):
        print(f"\n[{sweep_name}] {run_dir}")
        for step, path in list_checkpoints(run_dir):
            print(f"  {step:06d}  {path}")
