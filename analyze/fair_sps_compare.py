#!/usr/bin/env python3
"""Amortized fair Record vs Reactive SPS table (collect time included).

    eff = N_ego / (T_train + T_collect / M)

Storage / disk-write time is reported in the JSON but **excluded** from ``eff``
by default (``--include-storage`` to add it back).

Record uses collect_summary.json from ``analyze/bench_replay_collect.py``.
Reactive has zero collect overhead (policy cost already in train SPS).

Example:
  python analyze/fair_sps_compare.py \\
    --record-sps /data/puffer/experiments/sps_uniform_record/sps_summary.json \\
    --reactive-sps /data/puffer/experiments/sps_uniform_reactive/sps_summary.json \\
    --collect /data/puffer/experiments/collect_bench_lane_nominal/collect_summary.json \\
    --N 100000000 --M 4 \\
    --out /data/puffer/experiments/collect_bench_lane_nominal/fair_sps_table.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record-sps", required=True, help="sps_summary.json for Record")
    p.add_argument("--reactive-sps", required=True, help="sps_summary.json for Reactive")
    p.add_argument(
        "--collect",
        required=True,
        help="collect_summary.json from bench_replay_collect.py",
    )
    p.add_argument(
        "--N",
        type=float,
        default=100_000_000,
        help="Ego steps for comparison window (default 100M)",
    )
    p.add_argument(
        "--M",
        type=float,
        default=4,
        help="Amortization factor = # train seeds reusing one corpus (default 4)",
    )
    p.add_argument(
        "--include-storage",
        action="store_true",
        help="Add t_storage into the amortized extra term (default: collect only)",
    )
    p.add_argument("--out", default="", help="Write fair_sps_table.json here")
    return p.parse_args()


def _mean_sps(summary: dict) -> float:
    if "mean_sps_across_seeds" in summary:
        return float(summary["mean_sps_across_seeds"])
    seeds = summary.get("seeds") or []
    vals = [float(s["mean_sps_second_half"]) for s in seeds if s.get("mean_sps_second_half")]
    if not vals:
        raise ValueError("No mean_sps_across_seeds / seed stats in SPS summary")
    return sum(vals) / len(vals)


def _row(
    mode: str,
    train_sps: float,
    N: float,
    T_collect: float,
    T_storage: float,
    M: float,
) -> dict[str, Any]:
    T_train = N / max(train_sps, 1e-12)
    extra = (T_collect + T_storage) / max(M, 1e-12) if mode == "record" else 0.0
    T_total = T_train + extra
    eff = N / max(T_total, 1e-12)
    return {
        "mode": mode,
        "train_sps": train_sps,
        "T_train_s": T_train,
        "T_collect_s": T_collect if mode == "record" else 0.0,
        "T_storage_s": T_storage if mode == "record" else 0.0,
        "amortized_collect_storage_s": extra,
        "T_total_s": T_total,
        "eff_steps_per_s": eff,
        "N_ego_steps": N,
        "M": M,
    }


def main():
    cli = _parse_args()
    with open(cli.record_sps) as f:
        rec = json.load(f)
    with open(cli.reactive_sps) as f:
        rea = json.load(f)
    with open(cli.collect) as f:
        col = json.load(f)

    N = float(cli.N)
    M = float(cli.M)
    rec_sps = _mean_sps(rec)
    rea_sps = _mean_sps(rea)
    T_collect = float(col.get("T_collect_est") or 0.0)
    T_storage_raw = float(col.get("t_storage") or 0.0)
    T_storage = T_storage_raw if cli.include_storage else 0.0

    record_row = _row("record", rec_sps, N, T_collect, T_storage, M)
    reactive_row = _row("reactive", rea_sps, N, 0.0, 0.0, M)

    table = {
        "formula": (
            "eff = N / (T_train + (T_collect + T_storage) / M); Reactive extra=0"
            if cli.include_storage
            else "eff = N / (T_train + T_collect / M); Reactive extra=0; storage excluded"
        ),
        "N_ego_steps": N,
        "M_seeds": M,
        "include_storage": bool(cli.include_storage),
        "inputs": {
            "record_sps_path": os.path.abspath(cli.record_sps),
            "reactive_sps_path": os.path.abspath(cli.reactive_sps),
            "collect_path": os.path.abspath(cli.collect),
            "record_train_sps": rec_sps,
            "reactive_train_sps": rea_sps,
            "T_collect_est": T_collect,
            "t_storage": T_storage_raw,
            "t_storage_used_in_eff": T_storage,
            "collect_sps_like": col.get("mean_sps_like"),
            "S_agent_steps": col.get("S_agent_steps"),
            "bytes": col.get("bytes"),
        },
        "rows": [record_row, reactive_row],
        "speedup_eff_record_over_reactive": (
            record_row["eff_steps_per_s"] / max(reactive_row["eff_steps_per_s"], 1e-12)
        ),
        "note": (
            "Legacy zeroshot save-population wall-time is excluded; "
            "T_collect uses train-path collect_sps. "
            "Reactive policy cost is already inside train SPS. "
            + (
                "Storage write time included in eff."
                if cli.include_storage
                else "Storage write time excluded from eff (reported only)."
            )
        ),
    }

    out = cli.out or os.path.join(
        os.path.dirname(os.path.abspath(cli.collect)), "fair_sps_table.json"
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    def _fmt(row: dict) -> str:
        return (
            f"{row['mode']:10s}  train_sps={row['train_sps']:>10,.0f}  "
            f"T_train={row['T_train_s']:>8.1f}s  "
            f"amort_c={row['amortized_collect_storage_s']:>8.1f}s  "
            f"eff={row['eff_steps_per_s']:>10,.0f}"
        )

    print("========== Fair SPS table ==========")
    print(f"  N={N:,.0f}  M={M:g}  include_storage={cli.include_storage}")
    print(
        f"  T_collect={T_collect:.1f}s  t_storage(raw)={T_storage_raw:.1f}s  "
        f"used={T_storage:.1f}s"
    )
    for row in table["rows"]:
        print("  " + _fmt(row))
    print(
        f"  eff speedup Record/Reactive = "
        f"{table['speedup_eff_record_over_reactive']:.3f}x"
    )
    print(f"  wrote {out}")
    print("====================================")


if __name__ == "__main__":
    main()
