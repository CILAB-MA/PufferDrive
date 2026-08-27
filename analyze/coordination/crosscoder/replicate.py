#!/usr/bin/env python3
"""Matched-seed Crosscoder replication + interaction subspace.

Runs (one shot):
  pairs → replicate → subspace → report

  python analyze/coordination/crosscoder/replicate.py --out-root ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_COORD = Path(__file__).resolve().parent.parent
if str(_COORD) not in sys.path:
    sys.path.insert(0, str(_COORD))

from crosscoder.collect import EGO_MODES, flatten_pack, load_named_pair, normalize_ego_mode  
from crosscoder.frozen_config import (
    FROZEN_DICT,
    FROZEN_K,
    FROZEN_LAMBDA,
    REA_DIR,
    REC_DIR,
    RESULTS_MECHANISM,
    SEED_ORDER,
    resolve_split_path,
)
from crosscoder.metrics import (
    bootstrap_ci,
    category_masks,
    code_divergences,
    hidden_divergences,
    infer_codes_and_recon,
    interaction_sensitivity,
    orthonormal_basis,
    scene_level_delta,
    subspace_overlap,
    summarize_by_category,
)
from crosscoder.pipeline import (
    _apply_norm,
    _load_model,
    _normalize_fit,
    _oversample_mix,
    _save_model,
    _train_crosscoder,
    _write_json,
)

# Re-exported for persistence / transient_causal / intervention / collect.
__all__ = ["FROZEN_K", "phase_pairs", "_all_policies", "build_policy_pairs"]

WANDB_SEEDS = {
    "record": {42: "yb0uds6n", 3: "i2oeu54g", 11: "w1wu2uom", 0: "x9etv2km"},
    "reactive": {42: "xc4bxfyr", 3: "t6u7nnbx", 11: "5vf9d37q", 0: "757932iw"},
}


def resolve_acts_root(acts_root: Path | str, ego_mode: str) -> Path:
    """policy_seed_replication root → mode-specific collect dir parent.

    maintain: .../policy_seed_replication
    record:   .../policy_seed_replication/ego_record
    """
    mode = normalize_ego_mode(ego_mode)
    root = Path(acts_root)
    if mode == "maintain" or root.name == f"ego_{mode}":
        return root
    return root / f"ego_{mode}"


def resolve_out_root(out_root: Path | str, ego_mode: str) -> Path:
    """Redirect non-maintain away from the primary mechanism tree."""
    mode = normalize_ego_mode(ego_mode)
    out = Path(out_root)
    if mode != "maintain" and out.resolve() == RESULTS_MECHANISM.resolve():
        out = out.parent / f"crosscoder_mechanism_ego_{mode}"
        print(f"ego_mode={mode}: out → {out}", flush=True)
    return out


def _split_dirs(acts_root: Path) -> tuple[Path, Path]:
    return acts_root / "train_dataset", acts_root / "validation_dataset"


def _ckpt(method: str, run_id: str) -> Path:
    root = REC_DIR if method == "record" else REA_DIR
    return root / f"puffer_drive_pbt_{run_id}.pt"


def _policy_key(method: str, seed: int) -> str:
    return f"{method}_s{seed}"


def build_policy_pairs() -> dict:
    rec, rea = [], []
    for seed in SEED_ORDER:
        rec.append(
            {
                "train_seed": seed,
                "wandb_id": WANDB_SEEDS["record"][seed],
                "ckpt": str(_ckpt("record", WANDB_SEEDS["record"][seed])),
                "key": _policy_key("record", seed),
            }
        )
        rea.append(
            {
                "train_seed": seed,
                "wandb_id": WANDB_SEEDS["reactive"][seed],
                "ckpt": str(_ckpt("reactive", WANDB_SEEDS["reactive"][seed])),
                "key": _policy_key("reactive", seed),
            }
        )
    missing = [e for e in rec + rea if not Path(e["ckpt"]).is_file()]
    pairs = []
    for seed in SEED_ORDER:
        r = next(x for x in rec if x["train_seed"] == seed)
        a = next(x for x in rea if x["train_seed"] == seed)
        pairs.append(
            {
                "pair_id": f"matched_seed_{seed}",
                "train_seed": seed,
                "record": r,
                "reactive": a,
            }
        )
    return {
        "record_runs": rec,
        "reactive_runs": rea,
        "missing_ckpts": missing,
        "selected_pairs": pairs,
        "original_10k_pair": {
            "pair_id": "original_10k_mismatched",
            "record_wandb": "i2oeu54g",
            "reactive_wandb": "5vf9d37q",
            "note": "pipeline confirmatory pair (seeds 3 vs 11); not a matched-seed replicate",
        },
    }


def _all_policies(pairs_doc: dict) -> dict[str, Path]:
    return {e["key"]: Path(e["ckpt"]) for e in pairs_doc["record_runs"] + pairs_doc["reactive_runs"]}


def phase_pairs(root: Path) -> dict:
    doc = build_policy_pairs()
    path = root / "policy_seed_replication" / "policy_pairs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, doc)
    print(f"wrote {path} n_pairs={len(doc['selected_pairs'])}", flush=True)
    return doc


def _train_pair_crosscoder(
    train_pack: dict,
    *,
    out_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Path:
    ckpt = out_dir / f"crosscoder_seed{seed}.pt"
    if ckpt.is_file():
        print(f"  reuse {ckpt}", flush=True)
        return ckpt
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = flatten_pack(train_pack)
    mean_r, std_r = _normalize_fit(flat["h_record"])
    mean_a, std_a = _normalize_fit(flat["h_reactive"])
    np.savez(out_dir / "activation_norm.npz", mean_r=mean_r, std_r=std_r, mean_a=mean_a, std_a=std_a)
    h_r = _apply_norm(flat["h_record"], mean_r, std_r)
    h_a = _apply_norm(flat["h_reactive"], mean_a, std_a)
    rng = np.random.default_rng(0)
    mix = _oversample_mix(np.arange(h_r.shape[0]), flat["tight"], 0.25, rng)
    print(f"  train Crosscoder n={mix.size} seed={seed} λ={FROZEN_LAMBDA}", flush=True)
    model, hist = _train_crosscoder(
        h_r[mix],
        h_a[mix],
        dict_size=FROZEN_DICT,
        l1_coeff=FROZEN_LAMBDA,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        device=device,
        seed=seed,
    )
    _save_model(ckpt, model, {"lambda_train": FROZEN_LAMBDA, "seed": seed})
    _write_json(out_dir / f"train_log_seed{seed}.json", hist)
    return ckpt


def _codes_on_pack(model, pack, norm_path: Path, device: str) -> dict:
    norm = np.load(norm_path)
    n, t, d = pack["h_record"].shape
    h_r = _apply_norm(pack["h_record"].reshape(n * t, d), norm["mean_r"], norm["std_r"])
    h_a = _apply_norm(pack["h_reactive"].reshape(n * t, d), norm["mean_a"], norm["std_a"])
    return infer_codes_and_recon(model, h_r, h_a, l1_coeff=FROZEN_LAMBDA, device=device, n_iters=80)


def _eval_pair(pack: dict, inf: dict, hid: dict) -> dict:
    flat_t = pack["tight"].reshape(-1)
    masks = category_masks(flat_t, pack["approach"].reshape(-1))
    n_scene, tlen = pack["h_record"].shape[:2]
    scene = np.repeat(pack["scene_id"][:, None], tlen, axis=1).reshape(-1)
    cd = code_divergences(inf["c_record"], inf["c_reactive"])
    out = {
        "raw_hidden": {k: summarize_by_category(hid[k], masks) for k in ("d_h", "d_h_norm")},
        "latent": {k: summarize_by_category(cd[k], masks) for k in ("d_c", "d_c_l1_norm")},
        "recon": {
            "sep_record": inf["mean_sep_recon_record"],
            "sep_reactive": inf["mean_sep_recon_reactive"],
            "l0_record": inf["mean_sep_l0_record"],
            "l0_reactive": inf["mean_sep_l0_reactive"],
        },
        "scene_level": {},
    }
    for name, arr in (("d_c", cd["d_c"]), ("d_h_norm", hid["d_h_norm"])):
        sl = scene_level_delta(scene, arr, flat_t)
        out["scene_level"][name] = {
            "num_eligible_scenes": sl["num_eligible_scenes"],
            "mean_delta": sl["mean_delta"],
            "fraction_positive": sl["fraction_positive"],
            "bootstrap_ci": bootstrap_ci(sl["deltas"]),
        }
    return out


def _aggregate(rows: list[dict]) -> dict:
    def grab(*path: str):
        vals = []
        for r in rows:
            cur: object = r
            for k in path:
                cur = cur[k]  # type: ignore[index]
            vals.append(float(cur))  # type: ignore[arg-type]
        a = np.asarray(vals, dtype=np.float64)
        return {
            "per_pair": vals,
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        }

    return {
        "n_pairs": len(rows),
        "pairs": rows,
        "across_pairs": {
            "d_c_tight_over_nominal": grab("latent", "d_c", "tight_over_nominal"),
            "d_c_l1_norm_tight_over_nominal": grab("latent", "d_c_l1_norm", "tight_over_nominal"),
            "d_h_norm_tight_over_nominal": grab("raw_hidden", "d_h_norm", "tight_over_nominal"),
            "scene_frac_positive_d_c": grab("scene_level", "d_c", "fraction_positive"),
        },
    }


def replicate(args, root: Path, pairs_doc: dict, acts_root: Path) -> dict:
    """Train one Crosscoder per matched seed; eval |Δc| on val."""
    train_dir, val_dir = _split_dirs(acts_root)
    rows = []
    for pair in pairs_doc["selected_pairs"]:
        pid = pair["pair_id"]
        pdir = root / "policy_seed_replication" / pid
        rec_k, rea_k = pair["record"]["key"], pair["reactive"]["key"]
        print(f"########## replicate {pid} {rec_k} vs {rea_k} ##########", flush=True)
        train_pack = load_named_pair(train_dir, rec_k, rea_k)
        val_pack = load_named_pair(val_dir, rec_k, rea_k)
        ckpt = _train_pair_crosscoder(
            train_pack,
            out_dir=pdir,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=0,
        )
        model = _load_model(ckpt, args.device)
        inf = _codes_on_pack(model, val_pack, pdir / "activation_norm.npz", args.device)
        hid = hidden_divergences(
            val_pack["h_record"].reshape(-1, val_pack["h_record"].shape[-1]),
            val_pack["h_reactive"].reshape(-1, val_pack["h_reactive"].shape[-1]),
        )
        metrics = _eval_pair(val_pack, inf, hid)
        row = {
            "pair_id": pid,
            "train_seed": pair["train_seed"],
            "record": pair["record"],
            "reactive": pair["reactive"],
            "ego_mode": args.ego_mode,
            **metrics,
        }
        _write_json(pdir / "val_metrics.json", row)
        rows.append(row)
        print(
            f"  |Δc|={row['latent']['d_c']['tight_over_nominal']:.3f} "
            f"scene+={row['scene_level']['d_c']['fraction_positive']:.3f}",
            flush=True,
        )
        del inf, hid, train_pack, val_pack, model
    summary = _aggregate(rows)
    summary["ego_mode"] = args.ego_mode
    summary["acts_root"] = str(acts_root)
    _write_json(root / "policy_seed_replication" / "summary.json", summary)
    print(json.dumps(summary["across_pairs"], indent=2), flush=True)
    return summary


def _mask_pack(pack: dict, scene_mask: np.ndarray) -> dict:
    n = pack["scene_id"].shape[0]
    return {k: (v[scene_mask] if isinstance(v, np.ndarray) and v.shape[0] == n else v) for k, v in pack.items()}


def subspace(args, root: Path, pairs_doc: dict, acts_root: Path) -> dict:
    """Consensus U at K=FROZEN_K from train/dev; eval D_U on val."""
    split_path = resolve_split_path()
    split = json.loads(split_path.read_text())
    print(f"  subspace split={split_path}", flush=True)
    dev_ids = np.asarray(split["dev_scene_ids"], dtype=np.int64)
    train_dir, val_dir = _split_dirs(acts_root)
    k = int(FROZEN_K)
    n_seeds = int(args.cc_seeds)
    (root / "subspace" / "consensus").mkdir(parents=True, exist_ok=True)
    (root / "subspace" / "per_crosscoder_seed").mkdir(parents=True, exist_ok=True)

    all_pair = {}
    consensus = {}
    for pair in pairs_doc["selected_pairs"]:
        pid = pair["pair_id"]
        rec_k, rea_k = pair["record"]["key"], pair["reactive"]["key"]
        pdir = root / "policy_seed_replication" / pid
        train_pack = load_named_pair(train_dir, rec_k, rea_k)
        val_pack = load_named_pair(val_dir, rec_k, rea_k)
        dev_pack = _mask_pack(train_pack, np.isin(train_pack["scene_id"].astype(np.int64), dev_ids))
        print(f"########## subspace {pid} K={k} seeds={n_seeds} ##########", flush=True)

        bases = []
        for s in range(n_seeds):
            ckpt = _train_pair_crosscoder(
                train_pack,
                out_dir=pdir,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=s,
            )
            m = _load_model(ckpt, args.device)
            inf = _codes_on_pack(m, dev_pack, pdir / "activation_norm.npz", args.device)
            sens = interaction_sensitivity(
                np.abs(inf["c_record"] - inf["c_reactive"]), dev_pack["tight"].reshape(-1)
            )
            idx = np.argsort(-sens)[:k]
            stacked = np.concatenate(
                [m.decoder_columns("record").cpu().numpy()[idx], m.decoder_columns("reactive").cpu().numpy()[idx]],
                axis=0,
            )
            bases.append(stacked)
            del inf, m

        overlaps = []
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                overlaps.append(subspace_overlap(bases[i], bases[j])["overlap"])
        mean_overlap = float(np.mean(overlaps)) if overlaps else float("nan")

        qs = [orthonormal_basis(b) for b in bases]
        pavg = sum(q @ q.T for q in qs) / max(len(qs), 1)
        evals, evecs = np.linalg.eigh(pavg)
        order = np.argsort(-evals)
        mass = np.cumsum(evals[order])
        mass = mass / max(mass[-1], 1e-8)
        r = int(np.searchsorted(mass, 0.9) + 1)
        r = max(1, min(r, k))
        U = evecs[:, order[:r]].astype(np.float64)
        np.save(root / "subspace" / "consensus" / f"{pid}_U.npy", U)
        consensus[pid] = torch.from_numpy(U)

        n_v, t_v, hid = val_pack["h_record"].shape
        delta = val_pack["h_record"].reshape(n_v * t_v, hid) - val_pack["h_reactive"].reshape(n_v * t_v, hid)
        du = np.linalg.norm(delta @ U, axis=-1)
        masks = category_masks(val_pack["tight"].reshape(-1), val_pack["approach"].reshape(-1))
        du_sum = summarize_by_category(du.astype(np.float32), masks)
        pair_out = {
            "stability": {"mean_overlap": mean_overlap, "n_seed_pairs": len(overlaps), "k": k},
            "consensus_rank": r,
            "val_subspace": du_sum,
        }
        _write_json(root / "subspace" / "per_crosscoder_seed" / f"{pid}.json", pair_out)
        _write_json(
            root / "subspace" / "consensus" / f"{pid}_metadata.json",
            {"pair_id": pid, "k": k, "rank": r, "mean_overlap": mean_overlap},
        )
        all_pair[pid] = pair_out
        print(f"  overlap={mean_overlap:.3f} D_U ratio={du_sum['tight_over_nominal']:.3f}", flush=True)
        del train_pack, val_pack, dev_pack

    torch.save(consensus, root / "subspace" / "consensus" / "consensus_subspace.pt")
    _write_json(root / "subspace" / "stability.json", all_pair)
    ov = [all_pair[p]["stability"]["mean_overlap"] for p in all_pair]
    sub_summary = {
        "ego_mode": args.ego_mode,
        "acts_root": str(acts_root),
        "frozen_k": k,
        "mean_overlap": float(np.mean(ov)) if ov else float("nan"),
        "per_pair_overlap": ov,
        "pairs": list(all_pair.keys()),
    }
    _write_json(root / "subspace" / "summary.json", sub_summary)
    print(json.dumps(sub_summary, indent=2), flush=True)
    return {"pairs": all_pair, "summary": sub_summary}


def report(root: Path, *, ego_mode: str, acts_root: Path) -> dict:
    cross = json.loads((root / "policy_seed_replication" / "summary.json").read_text())
    sub_path = root / "subspace" / "summary.json"
    sub = json.loads(sub_path.read_text()) if sub_path.is_file() else {}
    summary = {
        "ego_mode": ego_mode,
        "acts_root": str(acts_root),
        "policy_seed_replication": cross.get("across_pairs"),
        "subspace": sub,
    }
    _write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default=str(RESULTS_MECHANISM))
    p.add_argument(
        "--acts-root",
        default=str(RESULTS_MECHANISM / "policy_seed_replication"),
        help="Collect root (maintain flat; ego_* appended for biased modes)",
    )
    p.add_argument("--ego-mode", default="maintain", choices=list(EGO_MODES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--cc-seeds", type=int, default=5, help="Crosscoder seeds for subspace consensus")
    args = p.parse_args()

    args.ego_mode = normalize_ego_mode(args.ego_mode)
    acts_root = resolve_acts_root(args.acts_root, args.ego_mode)
    root = resolve_out_root(args.out_root, args.ego_mode)
    (root / "policy_seed_replication").mkdir(parents=True, exist_ok=True)
    (root / "subspace").mkdir(parents=True, exist_ok=True)

    train_shards = acts_root / "train_dataset" / "shards"
    if not train_shards.is_dir() or not any(train_shards.glob("shard_*.npz")):
        raise SystemExit(
            f"Missing collect shards at {train_shards}. "
            "Run collect / run_collect_all_ego.sh first."
        )

    print(f"replicate ego_mode={args.ego_mode} acts={acts_root} out={root}", flush=True)
    pairs_doc = phase_pairs(root)
    if pairs_doc.get("missing_ckpts"):
        raise SystemExit(f"missing checkpoints: {pairs_doc['missing_ckpts']}")

    replicate(args, root, pairs_doc, acts_root)
    subspace(args, root, pairs_doc, acts_root)
    report(root, ego_mode=args.ego_mode, acts_root=acts_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
