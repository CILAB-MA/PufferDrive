#!/usr/bin/env python3
"""Fair Record collect bench: train-path policy SPS + storage I/O timing.

Phases (logged separately into ``collect_summary.json``):

1. **t_env_policy** — same vec / partner ``forward_eval`` path as
   ``puffer train_pbt`` reactive evaluate (no learn). Throughput =
   agent-steps / wall after warmup.
2. **t_flush_write** — write a corpus-shaped ``other_actions_actions.npy``
   (default: match ``{population}/saved/other_actions_actions.npy`` header)
   via memmap; report bytes / wall.
3. **t_storage_copy** (optional) — copy that file to ``--storage-copy-dest``.

Estimated collect wall for a full corpus:

    T_collect ≈ S / collect_sps
    S = n_combos * n_agents * horizon   (from corpus shape)

Does **not** use the slow zeroshot ``save-population`` path.

Example:
  CUDA_VISIBLE_DEVICES=0 python analyze/bench_replay_collect.py \\
    --population-path /data/puffer/popul_lane_nominal \\
    --out-dir /data/puffer/experiments/collect_bench_lane_nominal \\
    --total-timesteps 20000000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Optional

import numpy as np
import torch

import pufferlib
import pufferlib.pufferl as pufferl
from pufferlib.ocean.drive_pbt.drive_pbt import resolve_reactive_policy_files


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-name", default="puffer_drive_pbt")
    p.add_argument(
        "--config-dir",
        default="pufferlib/ocean/drive_pbt",
        help="Same config root as train_pbt",
    )
    p.add_argument(
        "--population-path",
        required=True,
        help="Population dir (manifest or *.pt), same as train",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Result dir (default: /data/puffer/experiments/collect_bench_{pop_short}/)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=20_000_000,
        help="Agent-steps for env/policy phase (mask.sum() over recv batches)",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=100_000,
        help="Agent-steps before SPS timer starts",
    )
    p.add_argument(
        "--num-checkpoints",
        type=int,
        default=0,
        help="Max partner policies to load (0 = all)",
    )
    p.add_argument("--log-interval", type=float, default=0.25)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument(
        "--puffer-arg",
        action="append",
        default=[],
        help="Extra arg for load_config, e.g. --puffer-arg=--train.device=cuda",
    )
    p.add_argument(
        "--corpus-actions",
        default="",
        help="Existing other_actions_actions.npy to take shape/dtype from "
        "(default: <population>/saved/other_actions_actions.npy)",
    )
    p.add_argument(
        "--n-combos",
        type=int,
        default=10,
        help="Rollouts/combos for T_collect / storage estimate (default 10). "
        "0 = use corpus file shape[0]",
    )
    p.add_argument(
        "--flush-max-bytes",
        type=int,
        default=0,
        help="Cap flush write size for smoke tests (0 = full corpus nbytes)",
    )
    p.add_argument(
        "--skip-env-policy",
        action="store_true",
        help="Skip phase 1 (reuse collect_sps from --env-policy-summary)",
    )
    p.add_argument(
        "--env-policy-summary",
        default="",
        help="JSON with overall_sps / mean_sps_second_half if skipping phase 1",
    )
    p.add_argument(
        "--skip-flush",
        action="store_true",
        help="Skip memmap flush phase",
    )
    p.add_argument(
        "--keep-flush-file",
        action="store_true",
        help="Keep the written flush npy (default: delete after timing)",
    )
    p.add_argument(
        "--storage-copy-dest",
        default="",
        help="If set, time a copy of the flush file (or corpus) to this path",
    )
    p.add_argument(
        "--hardware-note",
        default="",
        help="Free-text hardware note stored in summary",
    )
    return p.parse_args()


def _pop_short(population_path: str) -> str:
    name = os.path.basename(os.path.abspath(population_path).rstrip("/"))
    body = name[len("popul_") :] if name.startswith("popul_") else name
    return body


def _default_out_dir(population_path: str) -> str:
    return f"/data/puffer/experiments/collect_bench_{_pop_short(population_path)}"


def _load_train_config(env_name: str, config_dir: str, puffer_argv: list[str]) -> dict:
    saved = sys.argv[:]
    try:
        sys.argv = [saved[0], *puffer_argv]
        return pufferl.load_config(env_name, config_dir=config_dir)
    finally:
        sys.argv = saved


def _apply_cli_overrides(args: dict, cli) -> None:
    args["pbt"]["population_path"] = os.path.abspath(cli.population_path)
    args["train"]["seed"] = int(cli.seed)
    args["vec"]["seed"] = int(cli.seed)
    args["wandb"] = False
    args["neptune"] = False
    if cli.num_envs is not None:
        args["vec"]["num_envs"] = int(cli.num_envs)
    if cli.batch_size is not None:
        args["vec"]["batch_size"] = int(cli.batch_size)
    if cli.num_workers is not None:
        args["vec"]["num_workers"] = int(cli.num_workers)


def _load_partner_policies(args, vecenv, env_name: str, num_checkpoints: int):
    files = resolve_reactive_policy_files(args["pbt"]["population_path"])
    if num_checkpoints and num_checkpoints > 0:
        files = files[: int(num_checkpoints)]
    if not files:
        raise FileNotFoundError(
            f"No partner policies under {args['pbt']['population_path']}"
        )
    policies = []
    for path, name in files:
        a2 = dict(args)
        a2["load_model_path"] = path
        pol = pufferl.load_policy(a2, vecenv, env_name)
        pol.eval()
        for p in pol.parameters():
            p.requires_grad_(False)
        policies.append(pol)
        print(f"  loaded partner: {name} <- {path}")
    return policies


def _partition_counts(n_agents: int, n_pol: int) -> list[int]:
    base, rem = divmod(n_agents, n_pol)
    return [base + (i < rem) for i in range(n_pol)]


def _resolve_corpus_meta(cli) -> dict[str, Any]:
    """Shape/dtype/bytes for the Record action dump used in training."""
    pop = os.path.abspath(cli.population_path)
    path = cli.corpus_actions or os.path.join(
        pop, "saved", "other_actions_actions.npy"
    )
    if not os.path.isfile(path):
        # Fallback: typical nominal layout if file missing (smoke / new pop).
        n_combos = int(cli.n_combos) if cli.n_combos > 0 else 10
        shape = (n_combos, 62151, 910, 1)
        dtype = np.dtype(np.float64)
        nbytes = int(np.prod(shape) * dtype.itemsize)
        return {
            "corpus_actions_path": path,
            "exists": False,
            "shape": list(shape),
            "dtype": str(dtype),
            "nbytes": nbytes,
            "n_combos": n_combos,
            "n_agents": shape[1],
            "horizon": shape[2],
            "S_agent_steps": int(n_combos * shape[1] * shape[2]),
            "note": "corpus file missing; using default (n_combos, 62151, 910, 1) float64",
        }
    arr = np.load(path, mmap_mode="r")
    shape = tuple(int(x) for x in arr.shape)
    dtype = arr.dtype
    n_agents = int(shape[1]) if len(shape) > 1 else 0
    horizon = int(shape[2]) if len(shape) > 2 else 0
    channels = int(shape[3]) if len(shape) > 3 else 1
    if cli.n_combos > 0:
        n_combos = int(cli.n_combos)
    else:
        n_combos = int(shape[0])
    nbytes = int(n_combos * n_agents * horizon * channels * dtype.itemsize)
    S = int(n_combos * n_agents * horizon)
    return {
        "corpus_actions_path": path,
        "exists": True,
        "shape": [n_combos, n_agents, horizon, channels],
        "file_shape": list(shape),
        "dtype": str(dtype),
        "nbytes": nbytes,
        "n_combos": n_combos,
        "n_agents": n_agents,
        "horizon": horizon,
        "S_agent_steps": S,
    }


def run_env_policy_phase(cli, args: dict, out_dir: str) -> dict:
    """Phase 1: train-matched partner forward + env step SPS."""
    log_path = os.path.join(out_dir, f"collect_sps_seed_{cli.seed}.jsonl")
    print("========== phase 1: t_env_policy (train-matched) ==========")
    print(
        f"  vec: num_envs={args['vec'].get('num_envs')} "
        f"batch_size={args['vec'].get('batch_size')} "
        f"num_workers={args['vec'].get('num_workers')}"
    )
    print(f"  target_agent_steps={cli.total_timesteps}")

    vecenv = pufferl.load_env(cli.env_name, args)
    vecenv.async_reset(int(cli.seed))
    policies = _load_partner_policies(
        args, vecenv, cli.env_name, cli.num_checkpoints
    )
    device = args["train"]["device"]
    use_rnn = bool(args["train"].get("use_rnn"))
    n_pol = len(policies)
    hidden = int(getattr(policies[0], "hidden_size", 256))
    agents_per_batch = int(vecenv.agents_per_batch)
    action_buf = np.zeros((agents_per_batch, 1), dtype=np.int64)

    target = int(cli.total_timesteps)
    warmup_steps = max(0, int(cli.warmup_steps))
    log_interval = float(cli.log_interval)
    agent_steps = 0
    epoch = 0
    t0 = None
    steps_at_timer_start = 0
    last_log_t = None
    last_log_steps = 0
    sps_samples: list[float] = []

    # Wall only around recv→forward→send after warmup (excludes load).
    t_env_policy = 0.0

    with open(log_path, "w", encoding="utf-8") as log_f:
        while agent_steps < target:
            loop_t0 = time.perf_counter()
            o, r, d, t, info, env_id, mask = vecenv.recv()
            n = int(o.shape[0])
            if n != agents_per_batch:
                action_buf = np.zeros((n, 1), dtype=np.int64)
                agents_per_batch = n

            counts = _partition_counts(n, n_pol)
            pool = np.random.permutation(n)
            ob_t = torch.as_tensor(o, device=device)

            with torch.inference_mode():
                p = 0
                for pi, policy in enumerate(policies):
                    count = counts[pi]
                    if count <= 0:
                        continue
                    idx = pool[p : p + count]
                    p += count
                    state = {}
                    if use_rnn:
                        state["lstm_h"] = torch.zeros(count, hidden, device=device)
                        state["lstm_c"] = torch.zeros(count, hidden, device=device)
                    logits, _ = policy.forward_eval(ob_t[idx], state)
                    action, _, _ = pufferlib.pytorch.sample_logits(logits)
                    action_np = action.detach().cpu().numpy()
                    if action_np.ndim == 1:
                        action_np = action_np.reshape(-1, 1)
                    if isinstance(logits, torch.distributions.Normal):
                        action_np = np.clip(
                            action_np,
                            vecenv.action_space.low,
                            vecenv.action_space.high,
                        )
                    action_buf[idx] = action_np

            vecenv.send(action_buf)
            loop_dt = time.perf_counter() - loop_t0
            step = int(np.asarray(mask).sum()) if mask is not None else n
            agent_steps += step

            if t0 is None:
                if agent_steps < warmup_steps:
                    continue
                t0 = time.time()
                steps_at_timer_start = agent_steps
                last_log_t = t0
                last_log_steps = agent_steps
                print(f"  warmup done at steps={agent_steps}; timer start")
                continue

            t_env_policy += loop_dt
            now = time.time()
            if now - last_log_t >= log_interval and agent_steps > last_log_steps:
                epoch += 1
                dt = now - last_log_t
                sps = (agent_steps - last_log_steps) / max(dt, 1e-9)
                uptime = now - t0
                row = {
                    "SPS": sps,
                    "agent_steps": agent_steps,
                    "uptime": uptime,
                    "epoch": epoch,
                    "seed": cli.seed,
                    "pbt_mode": "collect_bench",
                    "wall_time": now,
                    "num_policies": n_pol,
                    "agents_per_batch": n,
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
                sps_samples.append(sps)
                print(
                    f"  epoch={epoch:4d}  steps={agent_steps:>10d}  "
                    f"SPS={sps:,.0f}  uptime={uptime:.1f}s"
                )
                last_log_t = now
                last_log_steps = agent_steps

    if t0 is None:
        vecenv.close()
        raise RuntimeError(
            f"Did not finish warmup ({warmup_steps}); "
            f"increase --total-timesteps (got agent_steps={agent_steps})"
        )

    wall = time.time() - t0
    measured_steps = max(agent_steps - steps_at_timer_start, 0)
    # Prefer precise perf_counter sum; fall back to wall if empty.
    t_env = t_env_policy if t_env_policy > 0 else wall
    overall_sps = measured_steps / max(t_env, 1e-9)
    half = sps_samples[len(sps_samples) // 2 :] if sps_samples else []
    mean_second = float(np.mean(half)) if half else overall_sps

    vecenv.close()
    result = {
        "t_env_policy": t_env,
        "wall_s_post_warmup": wall,
        "agent_steps": agent_steps,
        "measured_steps": measured_steps,
        "overall_sps": overall_sps,
        "mean_sps_second_half": mean_second,
        "n_sps_logs": len(sps_samples),
        "num_policies": n_pol,
        "log": os.path.basename(log_path),
        "vec": {
            "num_envs": args["vec"].get("num_envs"),
            "batch_size": args["vec"].get("batch_size"),
            "num_workers": args["vec"].get("num_workers"),
            "backend": args["vec"].get("backend"),
        },
        "env_num_agents": args["env"].get("num_agents"),
    }
    print(
        f"  t_env_policy={t_env:.2f}s  measured_steps={measured_steps}  "
        f"collect_sps={overall_sps:,.1f}"
    )
    return result


def run_flush_write_phase(
    out_dir: str,
    corpus: dict,
    flush_max_bytes: int,
    keep_file: bool,
) -> dict:
    """Phase 2: time writing corpus-shaped actions npy (memmap)."""
    print("========== phase 2: t_flush_write ==========")
    shape = tuple(corpus["shape"])
    dtype = np.dtype(corpus["dtype"])
    nbytes_full = int(corpus["nbytes"])
    write_bytes = nbytes_full
    write_shape = shape

    if flush_max_bytes and flush_max_bytes > 0 and flush_max_bytes < nbytes_full:
        # Shrink n_combos so product ≈ flush_max_bytes (keep trailing dims).
        trailing = int(np.prod(shape[1:])) * dtype.itemsize
        n_combos = max(1, flush_max_bytes // max(trailing, 1))
        write_shape = (n_combos, *shape[1:])
        write_bytes = int(np.prod(write_shape) * dtype.itemsize)
        print(
            f"  capped flush: shape {write_shape} ({write_bytes / 1e9:.3f} GB) "
            f"vs full {nbytes_full / 1e9:.3f} GB"
        )
    else:
        print(f"  full flush: shape {write_shape} ({write_bytes / 1e9:.3f} GB)")

    out_path = os.path.join(out_dir, "flush_other_actions_actions.npy")
    if os.path.exists(out_path):
        os.remove(out_path)

    # Fill with a cheap pattern so the write is real (not sparse holes only).
    t0 = time.perf_counter()
    mm = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=dtype, shape=write_shape
    )
    # Chunk along combo axis to bound RAM.
    chunk = max(1, min(write_shape[0], 2))
    for i in range(0, write_shape[0], chunk):
        j = min(write_shape[0], i + chunk)
        # Deterministic tiny values; avoids depending on RNG speed.
        mm[i:j] = np.zeros((j - i, *write_shape[1:]), dtype=dtype)
        mm.flush()
    del mm
    # Ensure durable size visible to OS.
    with open(out_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        final_size = f.tell()
    t_flush = time.perf_counter() - t0
    mb_s = (write_bytes / 1e6) / max(t_flush, 1e-9)

    # Extrapolate to full corpus if capped.
    if write_bytes < nbytes_full:
        t_flush_full_est = t_flush * (nbytes_full / max(write_bytes, 1))
    else:
        t_flush_full_est = t_flush

    result = {
        "t_flush_write": t_flush,
        "t_flush_write_full_est": t_flush_full_est,
        "flush_path": out_path,
        "flush_shape": list(write_shape),
        "flush_nbytes": write_bytes,
        "flush_file_size": final_size,
        "write_MBps": mb_s,
        "capped": write_bytes < nbytes_full,
    }
    print(
        f"  t_flush_write={t_flush:.2f}s  {mb_s:.1f} MB/s  "
        f"full_est={t_flush_full_est:.2f}s"
    )

    if not keep_file and not os.environ.get("PUFFER_KEEP_FLUSH"):
        # Keep file only if storage-copy needs it; caller may delete later.
        pass
    return result


def run_storage_copy_phase(
    src_path: str,
    dest: str,
    nbytes_ref: int,
) -> dict:
    """Phase 3: time copying flush/corpus file to dest (local or remote path)."""
    print("========== phase 3: t_storage_copy ==========")
    print(f"  src={src_path}")
    print(f"  dest={dest}")
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    if os.path.exists(dest):
        os.remove(dest)
    t0 = time.perf_counter()
    shutil.copy2(src_path, dest)
    t_copy = time.perf_counter() - t0
    size = os.path.getsize(dest)
    mb_s = (size / 1e6) / max(t_copy, 1e-9)
    if size < nbytes_ref:
        t_copy_full_est = t_copy * (nbytes_ref / max(size, 1))
    else:
        t_copy_full_est = t_copy
    print(
        f"  t_storage_copy={t_copy:.2f}s  {mb_s:.1f} MB/s  "
        f"full_est={t_copy_full_est:.2f}s"
    )
    return {
        "t_storage_copy": t_copy,
        "t_storage_copy_full_est": t_copy_full_est,
        "storage_copy_dest": dest,
        "storage_copy_nbytes": size,
        "copy_MBps": mb_s,
    }


def build_summary(cli, corpus: dict, env_pol: dict, flush: Optional[dict], copy: Optional[dict]) -> dict:
    collect_sps = float(env_pol.get("overall_sps") or env_pol.get("mean_sps_second_half") or 0.0)
    S = int(corpus["S_agent_steps"])
    t_collect_est = (S / collect_sps) if collect_sps > 0 else float("inf")

    t_flush = float((flush or {}).get("t_flush_write_full_est") or (flush or {}).get("t_flush_write") or 0.0)
    t_copy = float((copy or {}).get("t_storage_copy_full_est") or (copy or {}).get("t_storage_copy") or 0.0)
    # Local-only: storage = flush write. If copy measured, add it (upload).
    t_storage = t_flush + t_copy

    hw = cli.hardware_note or _default_hardware_note()

    return {
        "mode": "collect_bench",
        "seed": cli.seed,
        "population_path": os.path.abspath(cli.population_path),
        "hardware_note": hw,
        "corpus": corpus,
        "env_policy": env_pol,
        "flush_write": flush,
        "storage_copy": copy,
        "t_env_policy": float(env_pol.get("t_env_policy") or 0.0),
        "t_flush_write": t_flush,
        "t_storage_copy": t_copy,
        "t_storage": t_storage,
        "mean_sps_like": collect_sps,
        "overall_sps": collect_sps,
        "S_agent_steps": S,
        "T_collect_est": t_collect_est,
        "T_collect_plus_storage_est": t_collect_est + t_storage,
        "bytes": int(corpus["nbytes"]),
        "n_combos": int(corpus["n_combos"]),
        "formula": {
            "T_collect": "S_agent_steps / collect_sps",
            "T_storage": "t_flush_write_full_est + t_storage_copy_full_est",
            "note": (
                "collect_sps is train-matched partner+env throughput "
                "(all agents in batch). Legacy zeroshot save-population excluded."
            ),
        },
    }


def _default_hardware_note() -> str:
    note = []
    try:
        import subprocess

        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        ).strip().splitlines()
        if gpu:
            note.append("GPU: " + gpu[0].strip())
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    note.append("CPU: " + line.split(":", 1)[1].strip())
                    break
    except Exception:
        pass
    return "; ".join(note) if note else ""


def main():
    cli = _parse_args()
    pop = os.path.abspath(cli.population_path)
    out_dir = cli.out_dir or _default_out_dir(pop)
    os.makedirs(out_dir, exist_ok=True)

    corpus = _resolve_corpus_meta(cli)
    print("========== Fair collect bench ==========")
    print(f"  population={pop}")
    print(f"  out_dir={out_dir}")
    print(
        f"  corpus shape={corpus['shape']} dtype={corpus['dtype']} "
        f"nbytes={corpus['nbytes'] / 1e9:.2f}GB S={corpus['S_agent_steps']:,}"
    )
    print("========================================")

    env_pol: dict
    if cli.skip_env_policy:
        if not cli.env_policy_summary or not os.path.isfile(cli.env_policy_summary):
            raise SystemExit("--skip-env-policy requires --env-policy-summary JSON")
        with open(cli.env_policy_summary) as f:
            env_pol = json.load(f)
        # Normalize keys if a partial env-policy summary is reused.
        if "t_env_policy" not in env_pol:
            env_pol["t_env_policy"] = float(env_pol.get("wall_s") or 0.0)
        if "overall_sps" not in env_pol and "mean_sps_second_half" in env_pol:
            env_pol["overall_sps"] = env_pol["mean_sps_second_half"]
        print(f"  reused env_policy summary from {cli.env_policy_summary}")
    else:
        args = _load_train_config(cli.env_name, cli.config_dir, list(cli.puffer_arg or []))
        _apply_cli_overrides(args, cli)
        env_pol = run_env_policy_phase(cli, args, out_dir)

    flush = None
    flush_path = None
    if not cli.skip_flush:
        flush = run_flush_write_phase(
            out_dir,
            corpus,
            int(cli.flush_max_bytes or 0),
            keep_file=bool(cli.keep_flush_file or cli.storage_copy_dest),
        )
        flush_path = flush.get("flush_path")

    copy = None
    if cli.storage_copy_dest:
        src = flush_path
        if not src or not os.path.isfile(src):
            # Fall back to existing corpus file for copy timing.
            src = corpus["corpus_actions_path"] if corpus.get("exists") else None
        if not src or not os.path.isfile(src):
            raise SystemExit("storage-copy needs a flush file or existing corpus npy")
        copy = run_storage_copy_phase(src, cli.storage_copy_dest, int(corpus["nbytes"]))

    # Cleanup flush unless explicitly kept.
    if flush_path and os.path.isfile(flush_path) and not cli.keep_flush_file:
        try:
            os.remove(flush_path)
        except OSError:
            pass

    summary = build_summary(cli, corpus, env_pol, flush, copy)
    out_path = os.path.join(out_dir, "collect_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("---------- collect_summary ----------")
    print(f"  mean_sps_like     = {summary['mean_sps_like']:,.1f}")
    print(f"  T_collect_est     = {summary['T_collect_est']:.1f}s "
          f"({summary['T_collect_est'] / 3600:.2f}h)")
    print(f"  t_storage         = {summary['t_storage']:.1f}s")
    print(f"  T_collect+storage = {summary['T_collect_plus_storage_est']:.1f}s")
    print(f"  wrote {out_path}")
    print("-------------------------------------")
    return summary


if __name__ == "__main__":
    main()
