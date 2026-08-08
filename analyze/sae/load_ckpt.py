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
_FINAL_CKPT_RE = re.compile(r"^puffer_drive_(.+)\.pt$")


def checkpoint_step(path: str) -> int:
    """Parse training update/epoch from a checkpoint filename."""
    name = os.path.basename(path)
    match = _CKPT_RE.match(name)
    if not match:
        raise ValueError(f"Not a step checkpoint: {path}")
    return int(match.group(1))


def is_flat_policy_ckpt(path: str) -> bool:
    """True for exported ``puffer_drive_{id}.pt`` files (not run-dir checkpoints)."""
    return os.path.isfile(path) and bool(_FINAL_CKPT_RE.match(os.path.basename(path)))


def find_final_policy_ckpts(exp_path: str) -> list[tuple[str, str]]:
    """Flat exported policies at experiment root: ``puffer_drive_{id}.pt``.

    Does **not** descend into ``puffer_drive_*`` run subdirectories.
    """
    exp_path = os.path.abspath(exp_path)
    if not os.path.isdir(exp_path):
        raise FileNotFoundError(f"Experiment path does not exist: {exp_path}")

    runs: list[tuple[str, str]] = []
    for path in sorted(glob.glob(os.path.join(exp_path, "puffer_drive_*.pt"))):
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        match = _FINAL_CKPT_RE.match(name)
        if match:
            runs.append((match.group(1), path))
    return runs


def resolve_final_policy_ckpt(
    exp_path: str,
    *,
    sweep_id: str | None = None,
) -> str:
    """Resolve a single flat ``puffer_drive_{id}.pt`` under an experiment directory."""
    finals = find_final_policy_ckpts(exp_path)
    if not finals:
        raise FileNotFoundError(
            f"No flat puffer_drive_*.pt checkpoints under {exp_path}"
        )
    if sweep_id is not None:
        for sid, path in finals:
            if sid == sweep_id:
                return path
        ids = ", ".join(s for s, _ in finals)
        raise FileNotFoundError(
            f"No puffer_drive_{sweep_id}.pt under {exp_path} (have: {ids})"
        )
    if len(finals) == 1:
        return finals[0][1]
    ids = ", ".join(s for s, _ in finals)
    raise ValueError(
        f"Multiple flat puffer_drive_*.pt under {exp_path}; pass sweep_id. Found: {ids}"
    )


def resolve_policy_location(
    base_path: str,
    exp_name: str,
    location: str | None = None,
    *,
    sweep_id: str | None = None,
    prefer_final: bool = True,
) -> str:
    """Resolve a policy checkpoint path.

    Priority:
      1. Explicit ``location`` (.pt file or run dir with ``model_*.pt``)
      2. Flat ``puffer_drive_{id}.pt`` at experiment root (default)
      3. Legacy run subdirectory (only when ``prefer_final=False`` or no flat ckpts)
    """
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

    if prefer_final:
        raise FileNotFoundError(
            f"No flat puffer_drive_*.pt under {exp_path}. "
            "Export final weights to the experiment root or pass an explicit --run-dir / --policy-run-dirs path."
        )
    return resolve_run_dir(exp_path)


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
