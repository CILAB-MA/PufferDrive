#!/usr/bin/env python3
"""Merge collect shards under <population>/splits/ into <population>/saved/."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from typing import List, Optional, Tuple

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


def _matching_paths(splits_dir: str, start: int, end: int) -> dict:
    tag = f"{start:06d}_{end:06d}"
    return {
        "actions": os.path.join(splits_dir, f"actions_{tag}.npy"),
        "agent_offsets": os.path.join(splits_dir, f"agent_offsets_{tag}.npy"),
        "map_ids": os.path.join(splits_dir, f"map_ids_{tag}.npy"),
        "global_ids": os.path.join(splits_dir, f"global_ids_{tag}.npy"),
        "types": os.path.join(splits_dir, f"types_{tag}.npy"),
        "population_keys": os.path.join(splits_dir, f"population_keys_{tag}.npy"),
    }


def _load_corpus_global_ids(global_ids_path: str) -> np.ndarray:
    """Return 2D global_ids LUT from rollout 0 of a shard file."""
    gid = np.load(global_ids_path, mmap_mode="r")
    if gid.ndim == 3:
        return np.ascontiguousarray(gid[0])
    if gid.ndim == 2:
        return np.ascontiguousarray(gid)
    raise ValueError(f"{global_ids_path}: expected 2D or 3D array, got shape {gid.shape}")


def _require_optional_consistency(
    label: str,
    first_path: str,
    shard_path: str,
    start: int,
    end: int,
    first_has: bool,
) -> Optional[np.ndarray]:
    exists = os.path.isfile(shard_path)
    if first_has and not exists:
        raise ValueError(f"Missing {label} shard: {shard_path}")
    if (not first_has) and exists:
        raise ValueError(
            f"{label} shard present for [{start},{end}) but missing on first shard; "
            "collect with consistent settings"
        )
    if not exists:
        return None
    return np.load(shard_path, mmap_mode="r")


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
    paths0 = _matching_paths(splits_dir, start0, end0)

    first_a = np.load(actions_path, mmap_mode="r")
    first_ao = np.load(paths0["agent_offsets"], mmap_mode="r")
    first_m = np.load(paths0["map_ids"], mmap_mode="r")
    has_global_ids = os.path.isfile(paths0["global_ids"])
    has_types = os.path.isfile(paths0["types"])
    has_pop = os.path.isfile(paths0["population_keys"])
    if first_a.shape[0] != end0 - start0:
        raise ValueError(f"{actions_path}: leading dim {first_a.shape[0]} != {end0 - start0}")
    na, T, c = int(first_a.shape[1]), int(first_a.shape[2]), int(first_a.shape[3])
    jo = int(first_ao.shape[1])
    km = int(first_m.shape[1])
    dtype_a = first_a.dtype
    if has_global_ids:
        first_gid = np.load(paths0["global_ids"], mmap_mode="r")
        if first_gid.ndim == 3:
            if first_gid.shape[0] != end0 - start0:
                raise ValueError(
                    f"{paths0['global_ids']}: leading dim {first_gid.shape[0]} != {end0 - start0}"
                )
            num_maps, max_entity = int(first_gid.shape[1]), int(first_gid.shape[2])
        elif first_gid.ndim == 2:
            num_maps, max_entity = int(first_gid.shape[0]), int(first_gid.shape[1])
        else:
            raise ValueError(
                f"{paths0['global_ids']}: expected shape (shard, num_maps, max_entity) "
                "or (num_maps, max_entity)"
            )
    if has_types:
        first_types = np.load(paths0["types"], mmap_mode="r")
        if first_types.shape != (end0 - start0,):
            raise ValueError(
                f"{paths0['types']}: expected shape ({end0 - start0},), got {first_types.shape}"
            )
    if has_pop:
        first_pop = np.load(paths0["population_keys"], mmap_mode="r")
        if first_pop.shape != (end0 - start0, na):
            raise ValueError(
                f"{paths0['population_keys']}: expected shape ({end0 - start0}, {na}), "
                f"got {first_pop.shape}"
            )

    for start, end, ap in shards[1:]:
        paths = _matching_paths(splits_dir, start, end)
        aa = np.load(ap, mmap_mode="r")
        aao = np.load(paths["agent_offsets"], mmap_mode="r")
        am = np.load(paths["map_ids"], mmap_mode="r")
        if aa.dtype != dtype_a or tuple(aa.shape[1:]) != (na, T, c):
            raise ValueError(f"Shape/dtype mismatch: {ap} vs {actions_path}")
        if aao.shape[1:] != (jo,) or am.shape[1:] != (km,):
            raise ValueError(f"offsets/map_ids shape mismatch: shard [{start},{end})")
        if has_global_ids:
            if not os.path.isfile(paths["global_ids"]):
                raise ValueError(f"Missing global_ids shard: {paths['global_ids']}")
            gid = np.load(paths["global_ids"], mmap_mode="r")
            if gid.ndim == 3:
                if gid.shape != (end - start, num_maps, max_entity):
                    raise ValueError(f"global_ids shape mismatch: {paths['global_ids']}")
            elif gid.ndim == 2:
                if gid.shape != (num_maps, max_entity):
                    raise ValueError(f"global_ids shape mismatch: {paths['global_ids']}")
            else:
                raise ValueError(f"global_ids shape mismatch: {paths['global_ids']}")
        tt = _require_optional_consistency("types", paths0["types"], paths["types"], start, end, has_types)
        if tt is not None and tt.shape != (end - start,):
            raise ValueError(f"types shape mismatch: {paths['types']} got {tt.shape}")
        pp = _require_optional_consistency(
            "population_keys",
            paths0["population_keys"],
            paths["population_keys"],
            start,
            end,
            has_pop,
        )
        if pp is not None and pp.shape != (end - start, na):
            raise ValueError(
                f"population_keys shape mismatch: {paths['population_keys']} got {pp.shape}"
            )
        if aa.shape[0] != end - start:
            raise ValueError(f"{ap}: leading dim {aa.shape[0]} != {end - start}")

    out_a = os.path.join(saved_dir, "other_actions_actions.npy")
    out_ao = os.path.join(saved_dir, "other_actions_agent_offsets.npy")
    out_m = os.path.join(saved_dir, "other_actions_map_ids.npy")
    out_gid = os.path.join(saved_dir, "global_ids.npy")
    out_types = os.path.join(saved_dir, "difficulty_types.npy")
    out_pop = os.path.join(saved_dir, "population_keys.npy")
    os.makedirs(saved_dir, exist_ok=True)

    mm_a = np.lib.format.open_memmap(
        out_a, mode="w+", dtype=dtype_a, shape=(args.total_rollouts, na, T, c)
    )
    mm_ao = np.lib.format.open_memmap(out_ao, mode="w+", dtype=np.int32, shape=(args.total_rollouts, jo))
    mm_m = np.lib.format.open_memmap(out_m, mode="w+", dtype=np.int32, shape=(args.total_rollouts, km))
    mm_types = None
    if has_types:
        mm_types = np.lib.format.open_memmap(
            out_types, mode="w+", dtype=np.int32, shape=(args.total_rollouts,)
        )
    mm_pop = None
    if has_pop:
        mm_pop = np.lib.format.open_memmap(
            out_pop, mode="w+", dtype=np.int32, shape=(args.total_rollouts, na)
        )

    offset = 0
    for start, end, ap in shards:
        paths = _matching_paths(splits_dir, start, end)
        sl = end - start
        aa = np.load(ap, mmap_mode="r")
        aao = np.load(paths["agent_offsets"], mmap_mode="r")
        am = np.load(paths["map_ids"], mmap_mode="r")
        mm_a[offset : offset + sl] = np.ascontiguousarray(aa)
        mm_ao[offset : offset + sl] = np.ascontiguousarray(aao)
        mm_m[offset : offset + sl] = np.ascontiguousarray(am)
        if mm_types is not None:
            tt = np.load(paths["types"], mmap_mode="r")
            mm_types[offset : offset + sl] = np.ascontiguousarray(tt)
        if mm_pop is not None:
            pp = np.load(paths["population_keys"], mmap_mode="r")
            mm_pop[offset : offset + sl] = np.ascontiguousarray(pp)
        offset += sl
    del mm_a, mm_ao, mm_m
    if mm_types is not None:
        del mm_types
    if mm_pop is not None:
        del mm_pop

    print(f"Wrote {out_a}")
    print(f"Wrote {out_ao}")
    print(f"Wrote {out_m}")
    if has_global_ids:
        corpus_global_ids = _load_corpus_global_ids(paths0["global_ids"])
        np.save(out_gid, corpus_global_ids)
        print(f"Wrote {out_gid} shape={corpus_global_ids.shape} (corpus reference rollout 0)")
    if has_types:
        print(f"Wrote {out_types} shape=({args.total_rollouts},)")
    if has_pop:
        print(f"Wrote {out_pop} shape=({args.total_rollouts}, {na})")
        manifest_src = os.path.join(pop, "population_manifest.json")
        manifest_dst = os.path.join(saved_dir, "population_manifest.json")
        if os.path.isfile(manifest_src):
            shutil.copy2(manifest_src, manifest_dst)
            print(f"Wrote {manifest_dst}")
        else:
            print(
                f"Warning: {manifest_src} missing; population_keys.npy indices "
                "cannot be resolved to checkpoint paths",
                file=sys.stderr,
            )
    print(f"total_rollouts={args.total_rollouts} shards={len(shards)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
