#!/usr/bin/env python3
"""Concatenate save-population replay shards into Drive_PBT replay .npy files.

Shards are written by puffer zeroshot save-population to::

    <population_path>/splits/actions_{start:06d}_{end:06d}.npy
    <population_path>/splits/agent_offsets_{start:06d}_{end:06d}.npy
    <population_path>/splits/map_ids_{start:06d}_{end:06d}.npy

where ``[start, end)`` is a half-open global rollout index range (same as
``collect_start_idx`` / ``collect_end_idx`` in the ``[pbt]`` config).

Parallel example (total 50 rollouts)::

    # job A
    puffer zeroshot ... --pbt.num-collect-rollout=50 --pbt.collect-start-idx=0 --pbt.collect-end-idx=25 \\
        ...
    # job B
    puffer zeroshot ... --pbt.num-collect-rollout=50 --pbt.collect-start-idx=25 --pbt.collect-end-idx=50 \\

Then::

    python data_concat.py --population-path /path/to/population --total-rollouts 50

This writes ``saved/other_actions_actions.npy`` (and offsets / map_ids) for ``Drive_PBT`` replay mode.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Tuple

import numpy as np

_SPLIT_RE = re.compile(r"^actions_(\d+)_(\d+)\.npy$")


def _discover_shards(splits_dir: str) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    if not os.path.isdir(splits_dir):
        raise FileNotFoundError(f"Missing splits directory: {splits_dir}")
    for name in os.listdir(splits_dir):
        m = _SPLIT_RE.match(name)
        if not m:
            continue
        start, end = int(m.group(1)), int(m.group(2))
        if start >= end:
            raise ValueError(f"Invalid shard range in filename {name}: start={start} end={end}")
        out.append((start, end, os.path.join(splits_dir, name)))
    if not out:
        raise FileNotFoundError(f"No actions_*_*.npy shards under {splits_dir}")
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _validate_coverage(shards: List[Tuple[int, int, str]], total_rollouts: int) -> None:
    if shards[0][0] != 0:
        raise ValueError(f"Shards must start at 0; first shard starts at {shards[0][0]}")
    if shards[-1][1] != total_rollouts:
        raise ValueError(
            f"Last shard must end at total_rollouts={total_rollouts}; got {shards[-1][1]}"
        )
    for i in range(len(shards) - 1):
        a0, a1, _ = shards[i]
        b0, b1, _ = shards[i + 1]
        if a1 != b0:
            raise ValueError(
                f"Gap or overlap between shards: [{a0},{a1}) and [{b0},{b1})"
            )


def _matching_paths(splits_dir: str, start: int, end: int) -> Tuple[str, str, str, str]:
    tag = f"{start:06d}_{end:06d}"
    return (
        os.path.join(splits_dir, f"actions_{tag}.npy"),
        os.path.join(splits_dir, f"agent_offsets_{tag}.npy"),
        os.path.join(splits_dir, f"map_ids_{tag}.npy"),
        os.path.join(splits_dir, f"agent_ids_{tag}.npy"),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--population-path",
        required=True,
        help="Population directory (pbt.population_path); shard npy files live in <path>/splits/",
    )
    p.add_argument(
        "--total-rollouts",
        type=int,
        required=True,
        help="Must match pbt.num_collect_rollout used when collecting shards",
    )
    args = p.parse_args()

    pop = os.path.abspath(args.population_path)
    splits_dir = os.path.join(pop, "splits")
    saved_dir = os.path.join(pop, "saved")

    shards = _discover_shards(splits_dir)
    _validate_coverage(shards, args.total_rollouts)

    # Shape / dtype from first shard
    start0, end0, actions_path = shards[0]
    _, ao_path, m_path, id_path = _matching_paths(splits_dir, start0, end0)

    first_a = np.load(actions_path, mmap_mode="r")
    first_ao = np.load(ao_path, mmap_mode="r")
    first_m = np.load(m_path, mmap_mode="r")
    has_agent_ids = os.path.isfile(id_path)
    if first_a.shape[0] != end0 - start0:
        raise ValueError(f"{actions_path}: leading dim {first_a.shape[0]} != {end0 - start0}")
    na, T, c = int(first_a.shape[1]), int(first_a.shape[2]), int(first_a.shape[3])
    jo = int(first_ao.shape[1])
    km = int(first_m.shape[1])
    dtype_a = first_a.dtype
    if has_agent_ids:
        first_id = np.load(id_path, mmap_mode="r")
        if first_id.shape != (end0 - start0, na):
            raise ValueError(f"{id_path}: shape {first_id.shape} != ({end0 - start0}, {na})")

    for start, end, ap in shards[1:]:
        _, ao_p, m_p, id_p = _matching_paths(splits_dir, start, end)
        aa = np.load(ap, mmap_mode="r")
        aao = np.load(ao_p, mmap_mode="r")
        am = np.load(m_p, mmap_mode="r")
        if aa.dtype != dtype_a or tuple(aa.shape[1:]) != (na, T, c):
            raise ValueError(f"Shape/dtype mismatch: {ap} vs {actions_path}")
        if aao.shape[1:] != (jo,) or am.shape[1:] != (km,):
            raise ValueError(f"offsets/map_ids shape mismatch: shard [{start},{end})")
        if has_agent_ids:
            if not os.path.isfile(id_p):
                raise ValueError(f"Missing agent_ids shard: {id_p}")
            aid = np.load(id_p, mmap_mode="r")
            if aid.shape != (end - start, na):
                raise ValueError(f"agent_ids shape mismatch: {id_p}")
        if aa.shape[0] != end - start:
            raise ValueError(f"{ap}: leading dim {aa.shape[0]} != {end - start}")

    out_a = os.path.join(saved_dir, "other_actions_actions.npy")
    out_ao = os.path.join(saved_dir, "other_actions_agent_offsets.npy")
    out_m = os.path.join(saved_dir, "other_actions_map_ids.npy")
    out_id = os.path.join(saved_dir, "other_actions_agent_ids.npy")
    os.makedirs(saved_dir, exist_ok=True)

    mm_a = np.lib.format.open_memmap(
        out_a, mode="w+", dtype=dtype_a, shape=(args.total_rollouts, na, T, c)
    )
    mm_ao = np.lib.format.open_memmap(out_ao, mode="w+", dtype=np.int32, shape=(args.total_rollouts, jo))
    mm_m = np.lib.format.open_memmap(out_m, mode="w+", dtype=np.int32, shape=(args.total_rollouts, km))
    mm_id = None
    if has_agent_ids:
        mm_id = np.lib.format.open_memmap(
            out_id, mode="w+", dtype=np.int32, shape=(args.total_rollouts, na)
        )

    offset = 0
    for start, end, ap in shards:
        _, ao_p, m_p, id_p = _matching_paths(splits_dir, start, end)
        sl = end - start
        aa = np.load(ap, mmap_mode="r")
        aao = np.load(ao_p, mmap_mode="r")
        am = np.load(m_p, mmap_mode="r")
        mm_a[offset : offset + sl] = np.ascontiguousarray(aa)
        mm_ao[offset : offset + sl] = np.ascontiguousarray(aao)
        mm_m[offset : offset + sl] = np.ascontiguousarray(am)
        if has_agent_ids:
            mm_id[offset : offset + sl] = np.ascontiguousarray(np.load(id_p, mmap_mode="r"))
        offset += sl
    del mm_a, mm_ao, mm_m
    if mm_id is not None:
        del mm_id

    print(f"Wrote {out_a}")
    print(f"Wrote {out_ao}")
    print(f"Wrote {out_m}")
    if has_agent_ids:
        print(f"Wrote {out_id}")
    print(f"total_rollouts={args.total_rollouts} shards={len(shards)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
