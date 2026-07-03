#!/usr/bin/env python3
"""Build a map-binary subset folder from a scenario CSV (see find_worst_maps.py).

Copies the map_*.bin files referenced by a CSV's map_id column into a new
folder under the binaries root, renumbering them sequentially (map_000.bin,
map_001.bin, ...) in CSV row order, so the folder can be used directly as
--env.map-dir with --env.num-maps set to the row count.

Usage:
    python3 analyze/build_map_subset.py analyze/score_le1_high_collision.csv --dest-name score_le1_high_collision
    python3 analyze/build_map_subset.py analyze/score_le1_high_collision.csv --dest-name worst_100 --limit 100
"""

import argparse
import csv
import os
import shutil


def load_map_ids(csv_path, column):
    ids = []
    seen = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column not in reader.fieldnames:
            raise ValueError(f"Column {column!r} not found in {csv_path} (available: {', '.join(reader.fieldnames)})")
        for row in reader:
            map_id = int(float(row[column]))
            if map_id not in seen:
                seen.add(map_id)
                ids.append(map_id)
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="CSV with a map_id column (e.g. output of find_worst_maps.py)")
    parser.add_argument("--dest-name", required=True, help="New folder name created under --binaries-root")
    parser.add_argument("--map-id-col", default="map_id", help="CSV column holding the source map id (default: map_id)")
    parser.add_argument(
        "--src-dir",
        default="/data/puffer/resources/drive/binaries/training",
        help="Directory containing the source map_*.bin files",
    )
    parser.add_argument(
        "--binaries-root",
        default="/data/puffer/resources/drive/binaries",
        help="Root directory the new subset folder is created under",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only take the first N map ids (CSV row order)")
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copy (saves disk, breaks if src moves)")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing destination folder first")
    args = parser.parse_args()

    map_ids = load_map_ids(args.csv_path, args.map_id_col)
    if args.limit is not None:
        map_ids = map_ids[: args.limit]
    if not map_ids:
        raise SystemExit(f"No map ids found in {args.csv_path}")

    dest_dir = os.path.join(args.binaries_root, args.dest_name)
    if os.path.exists(dest_dir):
        if not args.overwrite:
            raise SystemExit(f"{dest_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    mapping_path = os.path.join(dest_dir, "source_map_ids.csv")
    missing = []
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["new_index", "source_map_id"])
        for new_idx, src_id in enumerate(map_ids):
            src_path = os.path.join(args.src_dir, f"map_{src_id:03d}.bin")
            if not os.path.isfile(src_path):
                missing.append(src_id)
                continue
            dst_path = os.path.join(dest_dir, f"map_{new_idx:03d}.bin")
            if args.symlink:
                os.symlink(src_path, dst_path)
            else:
                shutil.copyfile(src_path, dst_path)
            writer.writerow([new_idx, src_id])

    n_ok = len(map_ids) - len(missing)
    print(f"Wrote {n_ok} map binaries to {dest_dir}")
    print(f"  source: {args.src_dir}")
    print(f"  mapping: {mapping_path}")
    if missing:
        preview = missing[:10]
        suffix = "..." if len(missing) > 10 else ""
        print(f"  WARNING: {len(missing)} source maps not found: {preview}{suffix}")
    print(f"Use with: --env.map-dir {dest_dir} --env.num-maps {n_ok}")


if __name__ == "__main__":
    main()
