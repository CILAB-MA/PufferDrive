#!/usr/bin/env python3

from __future__ import annotations
from typing import Any
import numpy as np
from common import (
    ORDER,
    aggregate_numeric_across_seeds,
    frac,
    hierarchical_seed_map_ci,
    mcnemar_exact,
    paired_mean_ci,
    stage_probs,
    subset_mean,
)
import argparse
import json
from pathlib import Path

from common import DEFAULT_THRESHOLDS, METHODS, PRETTY, jsonable, list_policy_ckpts
from rollout import load_ego_pack, rollout_per_ego, save_ego_pack

def divergence_masks(by_alias: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    rec, rea, sp = by_alias["record"], by_alias["reactive"], by_alias["selfplay"]

    coll_disagree_rea = rec["collided"] != rea["collided"]
    coll_disagree_sp = rec["collided"] != sp["collided"]
    rec_ok_rea_coll = (~rec["collided"]) & rea["collided"]
    rec_coll_rea_ok = rec["collided"] & (~rea["collided"])
    rec_ok_sp_coll = (~rec["collided"]) & sp["collided"]
    rec_coll_sp_ok = rec["collided"] & (~sp["collided"])

    either_ap_rea = rec["had_approach"] | rea["had_approach"]
    either_ap_sp = rec["had_approach"] | sp["had_approach"]
    either_ap_rea_sp = rea["had_approach"] | sp["had_approach"]
    both_ap_rea = rec["had_approach"] & rea["had_approach"]
    both_ap_sp = rec["had_approach"] & sp["had_approach"]

    tight_disagree_rea = either_ap_rea & (rec["had_tight"] != rea["had_tight"])
    tight_disagree_sp = either_ap_sp & (rec["had_tight"] != sp["had_tight"])
    tight_disagree_rea_sp = either_ap_rea_sp & (rea["had_tight"] != sp["had_tight"])
    rec_res_rea_esc = both_ap_rea & rec["resolved"] & rea["escalated"]
    rec_esc_rea_res = both_ap_rea & rec["escalated"] & rea["resolved"]
    rec_res_sp_esc = both_ap_sp & rec["resolved"] & sp["escalated"]
    rec_esc_sp_res = both_ap_sp & rec["escalated"] & sp["resolved"]

    hard_tail = (
        rec["collided"]
        | rea["collided"]
        | sp["collided"]
        | tight_disagree_rea
        | tight_disagree_sp
        | tight_disagree_rea_sp
    )

    return {
        "coll_disagree_rea": coll_disagree_rea,
        "coll_disagree_sp": coll_disagree_sp,
        "rec_ok_rea_coll": rec_ok_rea_coll,
        "rec_coll_rea_ok": rec_coll_rea_ok,
        "rec_ok_sp_coll": rec_ok_sp_coll,
        "rec_coll_sp_ok": rec_coll_sp_ok,
        "either_ap_rea": either_ap_rea,
        "either_ap_sp": either_ap_sp,
        "both_ap_rea": both_ap_rea,
        "both_ap_sp": both_ap_sp,
        "tight_disagree_rea": tight_disagree_rea,
        "tight_disagree_sp": tight_disagree_sp,
        "tight_disagree_rea_sp": tight_disagree_rea_sp,
        "rec_res_rea_esc": rec_res_rea_esc,
        "rec_esc_rea_res": rec_esc_rea_res,
        "rec_res_sp_esc": rec_res_sp_esc,
        "rec_esc_sp_res": rec_esc_sp_res,
        "hard_tail": hard_tail,
    }


def _by_method(packs: dict[str, dict[str, np.ndarray]], key: str) -> dict[str, float]:
    return {alias: frac(packs[alias][key]) for alias in ORDER}


def _by_method_masked(
    packs: dict[str, dict[str, np.ndarray]],
    key: str,
    mask: np.ndarray,
) -> dict[str, float | None]:
    if not mask.any():
        return {alias: None for alias in ORDER}
    return {alias: frac(packs[alias][key][mask]) for alias in ORDER}


def _safe_frac(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def analyze_triplet(by_alias: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    rec, rea, sp = by_alias["record"], by_alias["reactive"], by_alias["selfplay"]
    n = int(rec["collided"].size)
    if rea["collided"].size != n or sp["collided"].size != n:
        raise ValueError("ego counts differ across methods — maps not aligned")

    m = divergence_masks(by_alias)
    both_ap_rea_n = int(m["both_ap_rea"].sum())
    n_c_pos = int(m["rec_ok_rea_coll"].sum())
    n_c_neg = int(m["rec_coll_rea_ok"].sum())
    n_c_pos_sp = int(m["rec_ok_sp_coll"].sum())
    n_c_neg_sp = int(m["rec_coll_sp_ok"].sum())

    paired_coll = rea["collided"].astype(np.float64) - rec["collided"].astype(np.float64)
    coll_ci = paired_mean_ci(paired_coll, seed=0)

    disagree_signed = np.zeros(n, dtype=np.float64)
    disagree_signed[m["rec_ok_rea_coll"]] = 1.0
    disagree_signed[m["rec_coll_rea_ok"]] = -1.0
    disagree_ci = paired_mean_ci(disagree_signed, seed=1)

    frac_c_pos = frac(m["rec_ok_rea_coll"])
    frac_c_neg = frac(m["rec_coll_rea_ok"])
    frac_res_esc = _safe_frac(int(m["rec_res_rea_esc"].sum()), both_ap_rea_n)
    frac_esc_res = _safe_frac(int(m["rec_esc_rea_res"].sum()), both_ap_rea_n)
    delta_prevent = (
        frac_res_esc - frac_esc_res
        if frac_res_esc is not None and frac_esc_res is not None
        else None
    )

    hard = m["hard_tail"]
    c_pos, c_neg = m["rec_ok_rea_coll"], m["rec_coll_rea_ok"]
    res_esc, esc_res = m["rec_res_rea_esc"], m["rec_esc_rea_res"]

    out: dict[str, Any] = {
        "n_ego": n,
        "alignment": {
            "scene_id_match_rec_rea": float(np.mean(rec["scene_id"] == rea["scene_id"])),
            "scene_id_match_rec_sp": float(np.mean(rec["scene_id"] == sp["scene_id"])),
        },
        "fullpop": {
            "delta_coll_rea_minus_rec": coll_ci["mean"],
            "ci_low": coll_ci["ci_low"],
            "ci_high": coll_ci["ci_high"],
            "coll": _by_method(by_alias, "collided"),
            "tight": _by_method(by_alias, "had_tight"),
        },
        "stages": {alias: stage_probs(by_alias[alias]) for alias in ORDER},
        "asymmetry_vs_reactive": {
            "n_rec_ok_rea_coll": n_c_pos,
            "n_rec_coll_rea_ok": n_c_neg,
            "frac_rec_ok_rea_coll": frac_c_pos,
            "frac_rec_coll_rea_ok": frac_c_neg,
            "delta_disagree": frac_c_pos - frac_c_neg,
            "delta_disagree_ci_low": disagree_ci["ci_low"],
            "delta_disagree_ci_high": disagree_ci["ci_high"],
            "mcnemar_p_value": mcnemar_exact(n_c_pos, n_c_neg)["p_value"],
            "frac_rec_resolve_rea_escalate|both_approach": frac_res_esc,
            "frac_rec_escalate_rea_resolve|both_approach": frac_esc_res,
            "delta_prevent": delta_prevent,
        },
        "asymmetry_vs_selfplay": {
            "n_rec_ok_sp_coll": n_c_pos_sp,
            "n_rec_coll_sp_ok": n_c_neg_sp,
            "frac_rec_ok_sp_coll": frac(m["rec_ok_sp_coll"]),
            "frac_rec_coll_sp_ok": frac(m["rec_coll_sp_ok"]),
            "delta_disagree": frac(m["rec_ok_sp_coll"]) - frac(m["rec_coll_sp_ok"]),
            "mcnemar_p_value": mcnemar_exact(n_c_pos_sp, n_c_neg_sp)["p_value"],
        },
        "hard_tail": {
            "n": int(hard.sum()),
            "frac": frac(hard),
            "frac_coll_disagree_vs_rea": frac(m["coll_disagree_rea"]),
            "frac_coll_disagree_vs_sp": frac(m["coll_disagree_sp"]),
            "coll": _by_method_masked(by_alias, "collided", hard),
            "tight": _by_method_masked(by_alias, "had_tight", hard),
            "pre_entry_speed": {
                alias: subset_mean(
                    by_alias[alias]["pre_entry_speed"],
                    hard & by_alias[alias]["had_tight"],
                )
                for alias in ORDER
            },
        },
        "on_rec_ok_rea_collide": {
            "n": n_c_pos,
            "rec_min_dist": subset_mean(rec["ep_min_dist"], c_pos),
            "rea_min_dist": subset_mean(rea["ep_min_dist"], c_pos),
            "rec_min_ttc": subset_mean(rec["ep_min_ttc"], c_pos),
            "rea_min_ttc": subset_mean(rea["ep_min_ttc"], c_pos),
            "rea_pre_entry_speed": subset_mean(
                rea["pre_entry_speed"], c_pos & rea["had_tight"]
            ),
        },
        "on_rec_coll_rea_ok": {
            "n": n_c_neg,
            "rec_min_dist": subset_mean(rec["ep_min_dist"], c_neg),
            "rea_min_dist": subset_mean(rea["ep_min_dist"], c_neg),
            "rec_min_ttc": subset_mean(rec["ep_min_ttc"], c_neg),
            "rea_min_ttc": subset_mean(rea["ep_min_ttc"], c_neg),
            "rec_pre_entry_speed": subset_mean(
                rec["pre_entry_speed"], c_neg & rec["had_tight"]
            ),
        },
        "on_rec_resolve_rea_escalate": {
            "n": int(res_esc.sum()),
            "rec_cpa_dist": subset_mean(rec["cpa_dist"], res_esc),
            "rec_cpa_speed": subset_mean(rec["cpa_speed"], res_esc),
            "rea_pre_entry_speed": subset_mean(rea["pre_entry_speed"], res_esc),
            "rea_entry_speed": subset_mean(rea["entry_speed"], res_esc),
        },
        "on_rec_escalate_rea_resolve": {
            "n": int(esc_res.sum()),
            "rea_cpa_dist": subset_mean(rea["cpa_dist"], esc_res),
            "rea_cpa_speed": subset_mean(rea["cpa_speed"], esc_res),
            "rec_pre_entry_speed": subset_mean(rec["pre_entry_speed"], esc_res),
            "rec_entry_speed": subset_mean(rec["entry_speed"], esc_res),
        },
        "_diff_coll_rea_minus_rec": paired_coll,
        "_diff_disagree_signed": disagree_signed,
    }
    return out


def _flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested dicts of numbers for seed aggregation."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).startswith("_") or k in {"ckpts", "seed_index"}:
                continue
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten_numeric(v, path))
            elif isinstance(v, (int, float)) and v is not None and np.isfinite(float(v)):
                out[path] = float(v)
    return out


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = root
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def aggregate_across_seeds(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std over seed replicates + hierarchical CI for primary deltas."""
    flat_seeds = [_flatten_numeric(s) for s in per_seed]
    across = aggregate_numeric_across_seeds(flat_seeds, skip=set())

    means: dict[str, Any] = {}
    for path, block in across.items():
        if block["mean"] is not None:
            _set_path(means, path, block["mean"])

    coll_diffs = [np.asarray(s["_diff_coll_rea_minus_rec"], dtype=np.float64) for s in per_seed]
    hci = hierarchical_seed_map_ci(coll_diffs, seed=0)
    means.setdefault("fullpop", {})
    means["fullpop"]["delta_coll_rea_minus_rec"] = hci["mean"]
    means["fullpop"]["ci_low"] = hci["ci_low"]
    means["fullpop"]["ci_high"] = hci["ci_high"]
    means["fullpop"]["hierarchical_bootstrap"] = {
        "n_seeds": hci.get("n_seeds"),
        "seed_means": hci.get("seed_means"),
        "ci_low": hci.get("ci_low"),
        "ci_high": hci.get("ci_high"),
    }
    means.setdefault("asymmetry_vs_reactive", {})
    means["asymmetry_vs_reactive"]["delta_disagree_ci_low"] = hci["ci_low"]
    means["asymmetry_vs_reactive"]["delta_disagree_ci_high"] = hci["ci_high"]
    # Averaged McNemar p is not a valid combined test — keep per-seed only.
    means["asymmetry_vs_reactive"].pop("mcnemar_p_value", None)
    means.setdefault("asymmetry_vs_selfplay", {}).pop("mcnemar_p_value", None)
    means["asymmetry_vs_reactive"]["mcnemar_per_seed"] = [
        {
            "seed_index": s.get("seed_index"),
            "n_c_pos": s["asymmetry_vs_reactive"]["n_rec_ok_rea_coll"],
            "n_c_neg": s["asymmetry_vs_reactive"]["n_rec_coll_rea_ok"],
            "p_value": s["asymmetry_vs_reactive"]["mcnemar_p_value"],
        }
        for s in per_seed
    ]
    return {"aggregate": means, "across_seeds": across}

def main():
    p = argparse.ArgumentParser(
        description="Divergence-conditioned Rec/Rea/SP comparison (canonical)"
    )
    p.add_argument("--experiments-root", type=str, default="/data/puffer/experiments")
    p.add_argument("--out-root", type=str, default="/data/puffer/results/coordination")
    p.add_argument("--num-maps", type=int, default=600)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-seeds", type=int, default=3)
    p.add_argument(
        "--reuse-packs",
        action="store_true",
        help="Reuse saved per-ego npz under out-root/divergence_scenes/packs/ if present",
    )
    p.add_argument("--ttc-approach", type=float, default=DEFAULT_THRESHOLDS["ttc_approach"])
    p.add_argument("--closing-approach", type=float, default=DEFAULT_THRESHOLDS["closing_approach"])
    p.add_argument("--dist-approach", type=float, default=DEFAULT_THRESHOLDS["dist_approach"])
    p.add_argument("--ttc-tight", type=float, default=DEFAULT_THRESHOLDS["ttc_tight"])
    p.add_argument("--closing-tight", type=float, default=DEFAULT_THRESHOLDS["closing_tight"])
    p.add_argument("--dist-tight", type=float, default=DEFAULT_THRESHOLDS["dist_tight"])
    p.add_argument("--pre-window", type=int, default=DEFAULT_THRESHOLDS["pre_window"])
    p.add_argument("--hard-brake", type=float, default=DEFAULT_THRESHOLDS["hard_brake"])
    args = p.parse_args()

    out_dir = Path(args.out_root) / "divergence_scenes"
    pack_dir = out_dir / "packs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_dir.mkdir(parents=True, exist_ok=True)

    ckpts = {
        alias: list_policy_ckpts(Path(args.experiments_root), exp, max_n=args.max_seeds)
        for alias, exp in METHODS.items()
    }
    n_seeds = min(len(ckpts[a]) for a in ORDER)
    if n_seeds <= 0:
        raise SystemExit("no checkpoints found")

    thresholds = dict(
        ttc_approach=args.ttc_approach,
        closing_approach=args.closing_approach,
        dist_approach=args.dist_approach,
        ttc_tight=args.ttc_tight,
        closing_tight=args.closing_tight,
        dist_tight=args.dist_tight,
        pre_window=args.pre_window,
        hard_brake=args.hard_brake,
    )

    per_seed: list[dict] = []
    for i in range(n_seeds):
        print(f"\n########## seed index {i} (paired same-map) ##########")
        by_alias: dict[str, dict[str, np.ndarray]] = {}
        for alias in ORDER:
            ckpt = ckpts[alias][i]
            pack_path = pack_dir / f"seed{i}_{alias}.npz"
            if args.reuse_packs and pack_path.is_file():
                print(f"  [{PRETTY[alias]}] reuse {pack_path.name}")
                by_alias[alias] = load_ego_pack(pack_path)
            else:
                print(f"  [{PRETTY[alias]}] {ckpt.name} maps={args.num_maps}")
                by_alias[alias] = rollout_per_ego(
                    ckpt=ckpt,
                    num_maps=args.num_maps,
                    device=args.device,
                    **thresholds,
                )
                save_ego_pack(pack_path, by_alias[alias])
            r = by_alias[alias]

        stats = analyze_triplet(by_alias)
        stats["seed_index"] = i
        stats["ckpts"] = {a: ckpts[a][i].name for a in ORDER}
        per_seed.append(stats)

    across = aggregate_across_seeds(per_seed)
    per_seed_out = []
    for s in per_seed:
        so = {k: v for k, v in s.items() if not str(k).startswith("_diff_")}
        per_seed_out.append(jsonable(so))
    summary = {
        "config": {"num_maps": args.num_maps, **thresholds, "max_seeds": n_seeds},
        "aggregate": across["aggregate"],
        "across_seeds": across["across_seeds"],
        "per_seed": per_seed_out,
    }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(jsonable(summary), indent=2))
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()
