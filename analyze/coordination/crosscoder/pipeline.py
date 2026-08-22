#!/usr/bin/env python3
"""Frozen ReCord vs Reactive Crosscoder.

Runs train → validate → report in one shot.

  python analyze/coordination/crosscoder/pipeline.py --ego-mode maintain
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_COORD = Path(__file__).resolve().parent.parent
if str(_COORD) not in sys.path:
    sys.path.insert(0, str(_COORD))

from common import jsonable  # noqa: E402
from crosscoder.collect import EGO_MODES, flatten_pack, load_named_pair, normalize_ego_mode  # noqa: E402
from crosscoder.frozen_config import (  # noqa: E402
    FROZEN_DICT,
    FROZEN_LAMBDA,
    RESULTS_10K,
    RESULTS_MECHANISM,
)
from crosscoder.metrics import (  
    bootstrap_ci,
    category_masks,
    code_divergences,
    hidden_divergences,
    infer_codes_and_recon,
    scene_level_delta,
    summarize_by_category,
)
from crosscoder.model import Crosscoder  

DEFAULT_ACTS_ROOT = RESULTS_MECHANISM / "policy_seed_replication"
DEFAULT_OUT_ROOT = RESULTS_10K
DEFAULT_REC_KEY = "record_s3"
DEFAULT_REA_KEY = "reactive_s11"
LAMBDA = FROZEN_LAMBDA
DICT_SIZE = FROZEN_DICT


# --- paths / IO (also used by replicate, intervention, …) ---


def resolve_acts_root(acts_root: Path | str, ego_mode: str) -> Path:
    mode = normalize_ego_mode(ego_mode)
    root = Path(acts_root)
    if mode == "maintain" or root.name == f"ego_{mode}":
        return root
    return root / f"ego_{mode}"


def resolve_out_root(out_root: Path | str, ego_mode: str) -> Path:
    mode = normalize_ego_mode(ego_mode)
    out = Path(out_root)
    if mode != "maintain" and out.resolve() == DEFAULT_OUT_ROOT.resolve():
        out = out.parent / f"crosscoder_10k_ego_{mode}"
        print(f"ego_mode={mode}: out → {out}", flush=True)
    return out


def _out(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "config": root / "config",
        "final_train": root / "final_train",
        "validation": root / "validation",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2))


# --- data ---


def _load_pair_pack(acts_root: Path, split: str, rec_key: str, rea_key: str) -> dict[str, np.ndarray]:
    d = acts_root / ("train_dataset" if split == "train" else "validation_dataset")
    return load_named_pair(d, rec_key, rea_key)


def _oversample_mix(
    idx: np.ndarray, tight: np.ndarray, crit_frac: float, rng: np.random.Generator
) -> np.ndarray:
    tight = tight.astype(bool)
    t_idx, n_idx = idx[tight[idx]], idx[~tight[idx]]
    if t_idx.size == 0 or n_idx.size == 0:
        return idx.copy()
    n_t = max(int(t_idx.size), int(round(crit_frac * n_idx.size / max(1e-8, 1.0 - crit_frac))))
    out = np.concatenate([n_idx, rng.choice(t_idx, size=n_t, replace=True)])
    rng.shuffle(out)
    return out


def _normalize_fit(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = h.mean(axis=0).astype(np.float32)
    std = h.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _apply_norm(h: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((h - mean) / std).astype(np.float32)


# --- model (imported by replicate.py) ---


def _train_crosscoder(
    h_r: np.ndarray,
    h_a: np.ndarray,
    *,
    dict_size: int,
    l1_coeff: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    train_mode: str = "shared",
    topk: int | None = None,
) -> tuple[Crosscoder, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Crosscoder(h_r.shape[1], dict_size, train_mode=train_mode, topk=topk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(h_r), torch.from_numpy(h_a)),
        batch_size=batch_size,
        shuffle=True,
    )
    history = []
    model.train()
    for epoch in range(epochs):
        acc = {k: 0.0 for k in ("loss", "recon", "l0")}
        n = 0
        fired = torch.zeros(dict_size, device=device, dtype=torch.bool)
        last = None
        for br, ba in loader:
            br, ba = br.to(device), ba.to(device)
            last = (br, ba)
            opt.zero_grad(set_to_none=True)
            loss, stats = model.loss(br, ba, l1_coeff=l1_coeff)
            loss.backward()
            opt.step()
            model.normalize_decoders()
            with torch.no_grad():
                zr, za, _, _ = model.forward(br, ba)
                fired |= (zr > 0).any(0) | (za > 0).any(0)
            for k in acc:
                acc[k] += stats[k]
            n += 1
        n_dead = int((~fired).sum().item())
        if last is not None and n_dead > int(0.8 * dict_size):
            n_dead = model.resample_dead_features(fired, last[0], last[1])
        else:
            n_dead = 0
        row = {k: v / max(n, 1) for k, v in acc.items()}
        row.update(epoch=epoch, n_dead_resampled=n_dead)
        history.append(row)
        print(f"    epoch {epoch:03d} recon={row['recon']:.4f} l0={row['l0']:.1f}", flush=True)
    return model, history


def _save_model(path: Path, model: Crosscoder, extra: dict | None = None) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "dict_size": model.dict_size,
            "hidden_dim": model.hidden_dim,
            "train_mode": model.train_mode,
            "topk": model.topk,
            **(extra or {}),
        },
        path,
    )


def _load_model(path: Path, device: str) -> Crosscoder:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = Crosscoder(
        int(ck.get("hidden_dim", 256)),
        int(ck["dict_size"]),
        train_mode=str(ck.get("train_mode", "shared")),
        topk=ck.get("topk"),
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


# --- train / validate / report ---


def train(args, dirs: dict[str, Path]) -> dict:
    pack = _load_pair_pack(Path(args.acts_root), "train", args.rec_key, args.rea_key)
    flat = flatten_pack(pack)
    mean_r, std_r = _normalize_fit(flat["h_record"])
    mean_a, std_a = _normalize_fit(flat["h_reactive"])
    np.savez(dirs["final_train"] / "activation_norm.npz", mean_r=mean_r, std_r=std_r, mean_a=mean_a, std_a=std_a)
    h_r = _apply_norm(flat["h_record"], mean_r, std_r)
    h_a = _apply_norm(flat["h_reactive"], mean_a, std_a)
    rng = np.random.default_rng(0)
    mix = _oversample_mix(np.arange(h_r.shape[0]), flat["tight"], args.crit_frac, rng)

    seeds = list(range(int(args.seeds)))
    for seed in seeds:
        out = dirs["final_train"] / f"mixed_seed{seed}"
        ckpt = out / "crosscoder.pt"
        if ckpt.is_file() and not args.overwrite:
            print(f"reuse {ckpt}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"########## train seed{seed} n={mix.size} λ={LAMBDA} ##########", flush=True)
        model, hist = _train_crosscoder(
            h_r[mix],
            h_a[mix],
            dict_size=int(args.dict_size),
            l1_coeff=float(LAMBDA),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            device=args.device,
            seed=seed,
        )
        _save_model(ckpt, model, {"lambda_train": float(LAMBDA), "seed": seed})
        _write_json(out / "train_log.json", hist)

    cfg = {
        "dictionary_size": int(args.dict_size),
        "lambda_train": float(LAMBDA),
        "lambda_infer": float(LAMBDA),
        "train_mode": "shared",
        "crosscoder_seeds": seeds,
        "ego_mode": args.ego_mode,
        "acts_root": str(args.acts_root),
        "rec_key": args.rec_key,
        "rea_key": args.rea_key,
        "n_train_scenes": int(pack["scene_id"].shape[0]),
    }
    _write_json(dirs["config"] / "selected_crosscoder_config.json", cfg)
    return cfg


def validate(args, dirs: dict[str, Path], cfg: dict) -> dict:
    pack = _load_pair_pack(Path(args.acts_root), "val", args.rec_key, args.rea_key)
    flat = flatten_pack(pack)
    norm = np.load(dirs["final_train"] / "activation_norm.npz")
    h_r = _apply_norm(flat["h_record"], norm["mean_r"], norm["std_r"])
    h_a = _apply_norm(flat["h_reactive"], norm["mean_a"], norm["std_a"])
    masks = category_masks(flat["tight"], flat["approach"])

    hid = hidden_divergences(flat["h_record"], flat["h_reactive"])
    raw = {k: summarize_by_category(hid[k], masks) for k in ("d_h", "d_h_norm")}

    seed = int(cfg["crosscoder_seeds"][0])
    model = _load_model(dirs["final_train"] / f"mixed_seed{seed}" / "crosscoder.pt", args.device)
    inf = infer_codes_and_recon(
        model, h_r, h_a, l1_coeff=float(cfg["lambda_infer"]), device=args.device, n_iters=int(args.ista_iters)
    )
    cd = code_divergences(inf["c_record"], inf["c_reactive"])
    latent = {k: summarize_by_category(cd[k], masks) for k in ("d_c", "d_c_l1_norm")}

    sl = scene_level_delta(flat["scene_id"], cd["d_c"], flat["tight"])
    scene = {
        "num_eligible_scenes": sl["num_eligible_scenes"],
        "mean_delta": sl["mean_delta"],
        "fraction_positive": sl["fraction_positive"],
        "bootstrap_ci": bootstrap_ci(sl["deltas"]),
    }
    recon = {
        "mean_sep": 0.5 * (inf["mean_sep_recon_record"] + inf["mean_sep_recon_reactive"]),
        "mean_l0": 0.5 * (inf["mean_sep_l0_record"] + inf["mean_sep_l0_reactive"]),
    }
    val = {
        "num_maps": int(pack["scene_id"].shape[0]),
        "num_states": int(flat["scene_id"].size),
        "tight_fraction": float(flat["tight"].mean()),
        "raw_hidden": raw,
        "latent": latent,
        "scene_level": {"d_c": scene},
        "recon": recon,
    }
    _write_json(dirs["validation"] / "validation_raw.json", val)
    print(
        f"  |Δc| ratio={latent['d_c']['tight_over_nominal']:.3f}  "
        f"d_h_norm={raw['d_h_norm']['tight_over_nominal']:.3f}  "
        f"scene+={scene['fraction_positive']:.3f}  "
        f"recon={recon['mean_sep']:.4f}",
        flush=True,
    )
    return val


def report(args, dirs: dict[str, Path], cfg: dict, val: dict) -> None:
    dc = val["latent"]["d_c"]["tight_over_nominal"]
    dhn = val["raw_hidden"]["d_h_norm"]["tight_over_nominal"]
    summary = {
        "ego_mode": args.ego_mode,
        "rec_key": args.rec_key,
        "rea_key": args.rea_key,
        "config": {"dict": cfg["dictionary_size"], "lambda": cfg["lambda_infer"]},
        "validation": {
            "d_c_tight_over_nominal": dc,
            "d_c_l1_norm_tight_over_nominal": val["latent"]["d_c_l1_norm"]["tight_over_nominal"],
            "d_h_norm_tight_over_nominal": dhn,
            "scene_frac_positive_d_c": val["scene_level"]["d_c"]["fraction_positive"],
            "mean_sep_recon": val["recon"]["mean_sep"],
            "mean_sep_l0": val["recon"]["mean_l0"],
            "num_maps": val["num_maps"],
            "num_states": val["num_states"],
        },
    }
    _write_json(dirs["root"] / "summary.json", summary)
    print(json.dumps(summary["validation"], indent=2), flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--acts-root", default=str(DEFAULT_ACTS_ROOT))
    p.add_argument("--ego-mode", default="maintain", choices=list(EGO_MODES))
    p.add_argument("--rec-key", default=DEFAULT_REC_KEY)
    p.add_argument("--rea-key", default=DEFAULT_REA_KEY)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dict-size", type=int, default=DICT_SIZE)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--crit-frac", type=float, default=0.25)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--ista-iters", type=int, default=80)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    args.ego_mode = normalize_ego_mode(args.ego_mode)
    args.acts_root = str(resolve_acts_root(args.acts_root, args.ego_mode))
    args.out_root = str(resolve_out_root(args.out_root, args.ego_mode))
    dirs = _out(Path(args.out_root))
    print(f"pipeline ego_mode={args.ego_mode} acts={args.acts_root} out={args.out_root}", flush=True)

    cfg = train(args, dirs)
    val = validate(args, dirs, cfg)
    report(args, dirs, cfg, val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
