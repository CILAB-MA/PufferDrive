"""Train one Sparse Autoencoder per experiment on collected activations.

Reads ``activations.npz`` produced by ``collect_sae_activations.py`` and
trains a **separate** SAE for each of ``selfplay`` / ``reactive_0.25`` /
``replay_0.25`` (not a shared model).

Usage::

    python analyze/sae/train_sae.py \\
      --sae-root /data/puffer/sae \\
      --probe-step 1908 \\
      --experiments selfplay,reactive_0.25,replay_0.25 \\
      --architecture topk --k 32 --expansion 16 \\
      --device cuda --out-dir /data/puffer/sae/runs/topk_k32

    # writes:
    #   <out-dir>/selfplay/sae_best.pt
    #   <out-dir>/reactive_0.25/sae_best.pt
    #   <out-dir>/replay_0.25/sae_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SAE_DIR = Path(__file__).resolve().parent
if str(_SAE_DIR) not in sys.path:
    sys.path.insert(0, str(_SAE_DIR))

from sae_model import SparseAutoencoder, build_sae  # noqa: E402

ACTIVATIONS_FILENAME = "activations.npz"
HUMAN_REPLAY_SUBDIR = "human_replay"


class MetricsLogger:
    """Accumulate train/eval rows and flush visualization-friendly exports.

    Writes under ``out_dir``:
      - ``train_log.jsonl``     raw event stream
      - ``metrics_history.npz`` aligned arrays (steps, loss, mse, ...)
      - ``metrics_history.csv`` same curves for pandas / sheets
      - ``summary.json``        final scalars + paths
    """

    def __init__(self, out_dir: Path, *, experiment: str):
        self.out_dir = Path(out_dir)
        self.experiment = experiment
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "train_log.jsonl"
        # wipe previous run artifacts in this dir
        if self.jsonl_path.is_file():
            self.jsonl_path.unlink()
        self.train_rows: list[dict] = []
        self.eval_rows: list[dict] = []

    def log_train(self, row: dict) -> None:
        row = {"kind": "train", "experiment": self.experiment, **row}
        self.train_rows.append(row)
        self._append_jsonl(row)

    def log_eval(self, row: dict) -> None:
        row = {"kind": "eval", "experiment": self.experiment, **row}
        self.eval_rows.append(row)
        self._append_jsonl(row)

    def _append_jsonl(self, row: dict) -> None:
        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def flush(self, *, extra_summary: dict | None = None) -> dict[str, Path]:
        """Write npz / csv / summary. Safe to call repeatedly."""
        written: dict[str, Path] = {"jsonl": self.jsonl_path}

        arrays: dict[str, np.ndarray] = {"experiment": np.array(self.experiment)}
        if self.train_rows:
            arrays.update(_rows_to_arrays(self.train_rows, prefix="train_"))
        if self.eval_rows:
            arrays.update(_rows_to_arrays(self.eval_rows, prefix="eval_"))

        npz_path = self.out_dir / "metrics_history.npz"
        np.savez_compressed(npz_path, **arrays)
        written["npz"] = npz_path

        csv_path = self.out_dir / "metrics_history.csv"
        _write_metrics_csv(csv_path, self.train_rows, self.eval_rows)
        written["csv"] = csv_path

        summary = {
            "experiment": self.experiment,
            "n_train_logs": len(self.train_rows),
            "n_eval_logs": len(self.eval_rows),
            "files": {k: str(v) for k, v in written.items()},
            "last_train": self.train_rows[-1] if self.train_rows else None,
            "last_eval": self.eval_rows[-1] if self.eval_rows else None,
            "best_eval": _best_eval_row(self.eval_rows),
        }
        if extra_summary:
            summary.update(extra_summary)
        summary_path = self.out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        written["summary"] = summary_path
        return written


def _numeric_keys(rows: list[dict]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        for k, v in row.items():
            if k in ("kind", "experiment", "split"):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                keys.add(k)
    # stable, useful order
    preferred = [
        "step",
        "tokens",
        "sec",
        "loss",
        "mse",
        "l0",
        "l1",
        "aux",
        "dead_frac",
        "train_mse",
        "train_l0",
        "train_explained_variance",
        "train_dead_frac",
        "val_mse",
        "val_l0",
        "val_explained_variance",
        "val_dead_frac",
    ]
    rest = sorted(keys - set(preferred))
    return [k for k in preferred if k in keys] + rest


def _rows_to_arrays(rows: list[dict], *, prefix: str) -> dict[str, np.ndarray]:
    keys = _numeric_keys(rows)
    out: dict[str, np.ndarray] = {}
    n = len(rows)
    for key in keys:
        vals = []
        for row in rows:
            v = row.get(key, np.nan)
            vals.append(float(v) if v is not None else np.nan)
        out[f"{prefix}{key}"] = np.asarray(vals, dtype=np.float64)
    out[f"{prefix}n"] = np.int32(n)
    return out


def _write_metrics_csv(
    path: Path,
    train_rows: list[dict],
    eval_rows: list[dict],
) -> None:
    """One CSV with kind column so train/eval curves share a file."""
    import csv

    all_rows = list(train_rows) + list(eval_rows)
    if not all_rows:
        path.write_text("kind,experiment,step\n")
        return
    keys = ["kind", "experiment"] + _numeric_keys(all_rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _best_eval_row(eval_rows: list[dict]) -> dict | None:
    if not eval_rows:
        return None
    key = "val_mse" if "val_mse" in eval_rows[0] else "train_mse"
    scored = [r for r in eval_rows if r.get(key) is not None]
    if not scored:
        return eval_rows[-1]
    return min(scored, key=lambda r: float(r[key]))


def write_run_comparison(root_out: Path, experiment_dirs: list[Path]) -> Path | None:
    """Merge per-exp metrics into root ``comparison_metrics.npz`` for multi-curve plots."""
    bundles: dict[str, dict[str, np.ndarray]] = {}
    for exp_dir in experiment_dirs:
        hist = exp_dir / "metrics_history.npz"
        if not hist.is_file():
            continue
        with np.load(hist, allow_pickle=False) as data:
            bundles[exp_dir.name] = {k: data[k] for k in data.files}

    if not bundles:
        return None

    flat: dict[str, np.ndarray] = {"experiments": np.array(list(bundles.keys()))}
    for exp_name, arrays in bundles.items():
        safe = exp_name.replace(".", "_")
        for key, arr in arrays.items():
            if key == "experiment":
                continue
            flat[f"{safe}__{key}"] = arr

    out = root_out / "comparison_metrics.npz"
    np.savez_compressed(out, **flat)

    # Also a small index for loaders / dashboards.
    index = {
        "experiments": list(bundles.keys()),
        "per_experiment": {
            name: str(d / "metrics_history.npz")
            for name, d in zip(bundles.keys(), experiment_dirs)
            if (d / "metrics_history.npz").is_file()
        },
        "comparison_npz": str(out),
    }
    (root_out / "comparison_index.json").write_text(json.dumps(index, indent=2))
    return out


def activation_key(exp_name: str) -> str:
    return f"activation__{exp_name}"


def parse_csv_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def resolve_step_dir(sae_root: str, data_mode: str, probe_step: int) -> Path:
    """``<sae_root>/human_replay/{mode}/step_XXXXXX`` (or already nested)."""
    root = Path(sae_root)
    candidates = [
        root / HUMAN_REPLAY_SUBDIR / data_mode / f"step_{probe_step:06d}",
        root / data_mode / f"step_{probe_step:06d}",
        root / f"step_{probe_step:06d}",
    ]
    for path in candidates:
        if (path / ACTIVATIONS_FILENAME).is_file():
            return path
    tried = "\n  ".join(str(p / ACTIVATIONS_FILENAME) for p in candidates)
    raise FileNotFoundError(f"No {ACTIVATIONS_FILENAME} for step {probe_step}. Tried:\n  {tried}")


def load_activation_matrix(
    step_dir: Path,
    experiment: str,
) -> tuple[np.ndarray, dict]:
    """Load one ``activation__<exp>`` matrix. Returns ``(X[N,d], meta)``."""
    path = step_dir / ACTIVATIONS_FILENAME
    key = activation_key(experiment)
    with np.load(path, allow_pickle=False) as data:
        available = (
            [str(x) for x in data["experiments"].tolist()] if "experiments" in data.files else []
        )
        if key not in data.files:
            hint = f" (file has {available})" if available else ""
            raise KeyError(f"missing {key} in {path}{hint}")
        x = np.asarray(data[key], dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"{key} must be 2-D, got {x.shape}")
        meta: dict = {
            "path": str(path),
            "experiment": experiment,
            "activation_key": key,
            "n": int(x.shape[0]),
            "d_in": int(x.shape[1]),
            "probe_step": int(data["probe_step"]) if "probe_step" in data.files else None,
            "repr_layer": str(data["repr_layer"]) if "repr_layer" in data.files else None,
            "sae_collect_version": (
                int(data["sae_collect_version"]) if "sae_collect_version" in data.files else None
            ),
        }
        # Optional visualization fields used for behavior-bucket sampling.
        for mk in ("ego_state", "other_state", "future_traj", "dist_at_t"):
            if mk in data.files:
                meta[mk] = np.asarray(data[mk])
    return x, meta


def compute_train_buckets(meta: dict) -> np.ndarray | None:
    """Return per-row bucket ids if ego/other(/future) meta is present."""
    if "ego_state" not in meta or "other_state" not in meta:
        return None
    from scene_metrics import assign_behavior_buckets, bucket_counts, compute_row_metrics

    metrics = compute_row_metrics(
        ego_state=meta["ego_state"],
        other_state=meta["other_state"],
        future_traj=meta.get("future_traj"),
        dist_at_t=meta.get("dist_at_t"),
    )
    buckets = assign_behavior_buckets(metrics)
    counts = bucket_counts(buckets)
    total = max(int(buckets.shape[0]), 1)
    print("  behavior buckets (sampling only):")
    for name, n in counts.items():
        print(f"    {name:24s} {n:7d} ({100.0 * n / total:5.1f}%)")
    return buckets


class ActivationBuffer:
    """Shuffle activations in-memory and yield fixed-size batches."""

    def __init__(self, x: np.ndarray, *, seed: int = 0):
        self.x = np.asarray(x, dtype=np.float32)
        self.n = int(self.x.shape[0])
        self.d = int(self.x.shape[1])
        self.rng = np.random.default_rng(seed)
        self._perm = self.rng.permutation(self.n)
        self._pos = 0

    def next_batch(self, batch_size: int) -> np.ndarray:
        if self.n == 0:
            return np.zeros((0, self.d), dtype=np.float32)
        out = np.empty((batch_size, self.d), dtype=np.float32)
        filled = 0
        while filled < batch_size:
            if self._pos >= self.n:
                self._perm = self.rng.permutation(self.n)
                self._pos = 0
            take = min(batch_size - filled, self.n - self._pos)
            out[filled : filled + take] = self.x[self._perm[self._pos : self._pos + take]]
            self._pos += take
            filled += take
        return out


class BalancedActivationBuffer:
    """Sample minibatches with fixed behavior-bucket fractions (not for SAE loss)."""

    def __init__(
        self,
        x: np.ndarray,
        bucket_ids: np.ndarray,
        *,
        fracs: dict[str, float] | None = None,
        seed: int = 0,
    ):
        from scene_metrics import BUCKET_ID, BUCKET_NAMES, DEFAULT_BUCKET_FRACS

        self.x = np.asarray(x, dtype=np.float32)
        self.n = int(self.x.shape[0])
        self.d = int(self.x.shape[1])
        self.rng = np.random.default_rng(seed)
        self.fracs = dict(fracs or DEFAULT_BUCKET_FRACS)
        # renormalize in case of custom fracs
        s = sum(self.fracs.get(name, 0.0) for name in BUCKET_NAMES)
        if s <= 0:
            raise ValueError("bucket fracs sum to 0")
        self.fracs = {name: float(self.fracs.get(name, 0.0)) / s for name in BUCKET_NAMES}

        self._indices: dict[str, np.ndarray] = {}
        self._pos: dict[str, int] = {}
        for name, bid in BUCKET_ID.items():
            idx = np.flatnonzero(bucket_ids == bid)
            if idx.size == 0:
                # fall back: borrow from normal / all rows
                idx = np.flatnonzero(bucket_ids == BUCKET_ID["normal_interaction"])
                if idx.size == 0:
                    idx = np.arange(self.n, dtype=np.int64)
            self._indices[name] = self.rng.permutation(idx)
            self._pos[name] = 0
        self._names = [n for n in BUCKET_NAMES if self.fracs[n] > 0]

    def _take(self, name: str, k: int) -> np.ndarray:
        if k <= 0:
            return np.zeros(0, dtype=np.int64)
        idx = self._indices[name]
        out = np.empty(k, dtype=np.int64)
        filled = 0
        while filled < k:
            if self._pos[name] >= idx.size:
                self._indices[name] = self.rng.permutation(idx)
                idx = self._indices[name]
                self._pos[name] = 0
            take = min(k - filled, idx.size - self._pos[name])
            out[filled : filled + take] = idx[self._pos[name] : self._pos[name] + take]
            self._pos[name] += take
            filled += take
        return out

    def next_batch(self, batch_size: int) -> np.ndarray:
        if self.n == 0:
            return np.zeros((0, self.d), dtype=np.float32)
        # multinomial allocation of slots, then residual to largest frac
        counts = {n: int(round(self.fracs[n] * batch_size)) for n in self._names}
        diff = batch_size - sum(counts.values())
        if diff != 0:
            top = max(self._names, key=lambda n: self.fracs[n])
            counts[top] = max(0, counts[top] + diff)
        parts = []
        for name in self._names:
            parts.append(self._take(name, counts[name]))
        rows = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        self.rng.shuffle(rows)
        return self.x[rows]


@torch.no_grad()
def evaluate_sae(
    sae: SparseAutoencoder,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> dict[str, float]:
    sae.eval()
    n = int(x.shape[0])
    if n == 0:
        return {"mse": 0.0, "l0": 0.0, "explained_variance": 0.0, "dead_frac": 1.0}

    mse_sum = 0.0
    l0_sum = 0.0
    fired = torch.zeros(sae.cfg.d_sae, dtype=torch.float64)
    ev_sum = 0.0

    for start in range(0, n, batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        out = sae.training_loss(batch)
        b = batch.shape[0]
        mse_sum += float(out["mse_loss"].item()) * b
        l0_sum += float(out["l0"].item()) * b
        feats = out["feature_acts"]
        fired += (feats > 0).any(dim=0).cpu().to(torch.float64)
        ev_sum += float(sae.explained_variance(batch, out["x_hat"]).item()) * b

    return {
        "mse": mse_sum / n,
        "l0": l0_sum / n,
        "explained_variance": ev_sum / n,
        "dead_frac": float((fired == 0).sum().item()) / sae.cfg.d_sae,
    }


def train_one_experiment(args: argparse.Namespace, experiment: str, out_dir: Path) -> Path:
    """Train a dedicated SAE for a single experiment's activations."""
    train_dir = resolve_step_dir(args.sae_root, "training", args.probe_step)
    x_train, train_meta = load_activation_matrix(train_dir, experiment)
    print(
        f"[{experiment}] train: {train_meta['path']}  "
        f"N={train_meta['n']}  d={train_meta['d_in']}"
    )

    x_val: np.ndarray | None = None
    val_meta: dict | None = None
    if not args.skip_val:
        try:
            val_dir = resolve_step_dir(args.sae_root, "validation", args.probe_step)
            x_val, val_meta = load_activation_matrix(val_dir, experiment)
            print(
                f"[{experiment}] val:   {val_meta['path']}  "
                f"N={val_meta['n']}  d={val_meta['d_in']}"
            )
            for mk in ("ego_state", "other_state", "future_traj", "dist_at_t"):
                val_meta.pop(mk, None)
        except FileNotFoundError as exc:
            print(f"[{experiment}] warning: no validation ({exc}); continuing without val")

    d_in = int(train_meta["d_in"])
    d_sae = args.d_sae if args.d_sae is not None else args.expansion * d_in

    device = torch.device(args.device)
    sae = build_sae(
        d_in=d_in,
        d_sae=d_sae,
        expansion=args.expansion,
        architecture=args.architecture,
        k=args.k,
        l1_coefficient=args.l1_coefficient,
        device=str(device),
        apply_b_dec_to_input=not args.no_b_dec_to_input,
        normalize_decoder=not args.no_normalize_decoder,
        aux_loss_coefficient=args.aux_loss_coefficient,
        repr_layer=str(train_meta.get("repr_layer") or "partner_encoder_slot"),
        experiment=experiment,
        extra={
            "probe_step": args.probe_step,
            "sae_root": args.sae_root,
            "experiment": experiment,
            "train_n": train_meta["n"],
        },
    )
    sae.cfg.device = str(device)
    sae.to(device)
    print(f"[{experiment}] SAE: {sae}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "experiment": experiment,
                "train_meta": train_meta,
                "val_meta": val_meta,
                "sae_cfg": sae.cfg.to_dict(),
            },
            indent=2,
            default=str,
        )
    )

    opt = torch.optim.Adam(
        sae.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    buffer: ActivationBuffer | BalancedActivationBuffer
    if args.no_balanced_sampling:
        buffer = ActivationBuffer(x_train, seed=args.seed)
        print(f"[{experiment}] sampling: uniform shuffle")
    else:
        buckets = compute_train_buckets(train_meta)
        if buckets is None:
            buffer = ActivationBuffer(x_train, seed=args.seed)
            print(f"[{experiment}] sampling: uniform (no ego/other meta for buckets)")
        else:
            buffer = BalancedActivationBuffer(x_train, buckets, seed=args.seed)
            from scene_metrics import DEFAULT_BUCKET_FRACS

            print(f"[{experiment}] sampling: balanced behavior buckets {DEFAULT_BUCKET_FRACS}")
            print(
                f"[{experiment}]   rare buckets use replacement "
                f"(shuffle+reloop when exhausted)"
            )

    # Drop heavy meta arrays from memory after buckets are built.
    for mk in ("ego_state", "other_state", "future_traj", "dist_at_t"):
        train_meta.pop(mk, None)

    tokens_since_fire = torch.zeros(d_sae, dtype=torch.long, device=device)
    dead_threshold = int(args.dead_feature_window)

    logger = MetricsLogger(out_dir, experiment=experiment)
    best_val_mse = float("inf")
    t0 = time.time()
    tokens_seen = 0
    checkpoint_steps = set()
    if args.checkpoint_steps.strip():
        checkpoint_steps = {
            int(x.strip()) for x in args.checkpoint_steps.split(",") if x.strip()
        }

    sae.train()
    for step in range(1, args.num_steps + 1):
        batch_np = buffer.next_batch(args.batch_size)
        batch = torch.from_numpy(batch_np).to(device, non_blocking=True)

        dead_mask = None
        if args.architecture == "topk":
            dead_mask = tokens_since_fire >= dead_threshold

        out = sae.training_loss(batch, dead_neuron_mask=dead_mask)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        opt.step()
        if sae.cfg.normalize_decoder:
            sae.normalize_decoder_()

        with torch.no_grad():
            fired = (out["feature_acts"] > 0).any(dim=0)
            tokens_since_fire += args.batch_size
            tokens_since_fire[fired] = 0

        tokens_seen += args.batch_size

        if step % args.log_every == 0 or step == 1:
            row = {
                "step": step,
                "tokens": tokens_seen,
                "loss": float(loss.item()),
                "mse": float(out["mse_loss"].item()),
                "l0": float(out["l0"].item()),
                "dead_frac": float((tokens_since_fire >= dead_threshold).float().mean().item()),
                "sec": round(time.time() - t0, 2),
                "lr": float(opt.param_groups[0]["lr"]),
            }
            if "l1_loss" in out:
                row["l1"] = float(out["l1_loss"].item())
            if "aux_loss" in out:
                row["aux"] = float(out["aux_loss"].item())
            logger.log_train(row)
            print(
                f"[{experiment}] step {step:6d}  loss={row['loss']:.5f}  mse={row['mse']:.5f}  "
                f"l0={row['l0']:.2f}  dead={row['dead_frac']:.3f}"
                + (f"  l1={row['l1']:.5f}" if "l1" in row else "")
                + (f"  aux={row['aux']:.5f}" if "aux" in row else "")
            )

        if step % args.eval_every == 0 or step == args.num_steps:
            metrics: dict = {"step": step, "tokens": tokens_seen, "sec": round(time.time() - t0, 2)}
            idx = np.random.default_rng(step).choice(
                x_train.shape[0], size=min(args.eval_samples, x_train.shape[0]), replace=False
            )
            train_m = evaluate_sae(sae, x_train[idx], device=device, batch_size=args.batch_size)
            metrics.update({f"train_{k}": v for k, v in train_m.items()})
            if x_val is not None:
                val_m = evaluate_sae(sae, x_val, device=device, batch_size=args.batch_size)
                metrics.update({f"val_{k}": v for k, v in val_m.items()})
                print(
                    f"[{experiment}] eval  train_mse={train_m['mse']:.5f} "
                    f"train_ev={train_m['explained_variance']:.4f}  "
                    f"val_mse={val_m['mse']:.5f} val_ev={val_m['explained_variance']:.4f}  "
                    f"val_l0={val_m['l0']:.2f} val_dead={val_m['dead_frac']:.3f}"
                )
                if val_m["mse"] < best_val_mse:
                    best_val_mse = val_m["mse"]
                    sae.save(out_dir / "sae_best.pt")
                    metrics["is_best"] = 1
                    print(f"[{experiment}] saved best -> {out_dir / 'sae_best.pt'}")
                else:
                    metrics["is_best"] = 0
            else:
                print(
                    f"[{experiment}] eval  train_mse={train_m['mse']:.5f} "
                    f"train_ev={train_m['explained_variance']:.4f} train_l0={train_m['l0']:.2f}"
                )
            logger.log_eval(metrics)
            # Refresh plottable artifacts during long runs.
            if step % max(args.eval_every, 1) == 0:
                logger.flush()
            sae.train()

        if (args.save_every > 0 and step % args.save_every == 0) or step in checkpoint_steps:
            ckpt_path = out_dir / f"sae_step_{step:07d}.pt"
            sae.save(ckpt_path)
            print(f"[{experiment}] checkpoint -> {ckpt_path}")

    sae.save(out_dir / "sae_last.pt")
    paths = logger.flush(
        extra_summary={
            "best_val_mse": best_val_mse if best_val_mse < float("inf") else None,
            "sae_best": str(out_dir / "sae_best.pt") if (out_dir / "sae_best.pt").is_file() else None,
            "sae_last": str(out_dir / "sae_last.pt"),
            "elapsed_sec": round(time.time() - t0, 2),
        }
    )
    print(f"[{experiment}] done. last -> {out_dir / 'sae_last.pt'}")
    print(
        f"[{experiment}] metrics -> {paths['npz'].name}, "
        f"{paths['csv'].name}, {paths['summary'].name}"
    )
    return out_dir


def train(args: argparse.Namespace) -> list[Path]:
    experiments = parse_csv_list(args.experiments)
    if not experiments:
        raise ValueError("--experiments is empty")

    root_out = Path(args.out_dir)
    root_out.mkdir(parents=True, exist_ok=True)
    (root_out / "run_config.json").write_text(
        json.dumps(
            {
                "experiments": experiments,
                "note": "One SAE per experiment under <out-dir>/<experiment>/",
                "artifacts": {
                    "per_experiment": [
                        "train_log.jsonl",
                        "metrics_history.npz",
                        "metrics_history.csv",
                        "summary.json",
                        "config.json",
                        "sae_best.pt",
                        "sae_last.pt",
                    ],
                    "run_root": [
                        "run_config.json",
                        "comparison_metrics.npz",
                        "comparison_index.json",
                    ],
                },
                "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            },
            indent=2,
            default=str,
        )
    )

    written: list[Path] = []
    for i, exp in enumerate(experiments):
        print()
        print("=" * 72)
        print(f"Training SAE {i + 1}/{len(experiments)}: {exp}")
        print("=" * 72)
        # Independent RNG stream per experiment for reproducibility.
        torch.manual_seed(args.seed + i)
        np.random.seed(args.seed + i)
        written.append(train_one_experiment(args, exp, root_out / exp))

    cmp_path = write_run_comparison(root_out, written)
    if cmp_path is not None:
        print(f"\nComparison metrics -> {cmp_path}")
    return written

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train one SAE per experiment on collected activations.npz"
    )
    p.add_argument("--sae-root", type=str, default="/data/puffer/sae")
    p.add_argument("--probe-step", type=int, default=1908)
    p.add_argument(
        "--experiments",
        type=str,
        default="selfplay,reactive_0.25,replay_0.25",
        help="Comma-separated experiments; each gets its own SAE under out-dir/<exp>/",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/data/puffer/sae/runs/default",
        help="Root run dir; writes <out-dir>/<experiment>/sae_*.pt",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-val", action="store_true")

    p.add_argument("--architecture", choices=("standard", "topk"), default="topk")
    p.add_argument("--expansion", type=int, default=16, help="d_sae = expansion * d_in if --d-sae unset")
    p.add_argument("--d-sae", type=int, default=None)
    p.add_argument("--k", type=int, default=32, help="TopK sparsity (topk arch)")
    p.add_argument("--l1-coefficient", type=float, default=1e-3, help="Standard SAE L1 coef")
    p.add_argument("--aux-loss-coefficient", type=float, default=1.0)
    p.add_argument("--no-b-dec-to-input", action="store_true")
    p.add_argument("--no-normalize-decoder", action="store_true")

    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--num-steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--dead-feature-window",
        type=int,
        default=10_000_000,
        help="Tokens without fire before a feature is marked dead (topk aux)",
    )

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-samples", type=int, default=16384)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument(
        "--checkpoint-steps",
        type=str,
        default="1000,2000",
        help="Extra commas of step numbers to always write sae_step_XXXXXXX.pt",
    )
    p.add_argument(
        "--no-balanced-sampling",
        action="store_true",
        help="Disable behavior-bucket minibatch rebalancing",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    paths = train(args)
    print()
    print("All SAEs written:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
