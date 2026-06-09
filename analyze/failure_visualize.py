#!/usr/bin/env python3
"""
Failure-aware visualization for PufferDrive eval / zeroshot scenario logs.

Workflows::

    # 1) Find collision scenes from zeroshot per-pair JSON (keys = map index)
    python analyze/failure_visualize.py \\
        --from-scenario-log /data/puffer/results/reactive_nominal/unseen_other_rewards/scenario_logs \\
        --map-dir /data/puffer/resources/drive/binaries/training \\
        --ego-collision-threshold 0.01 \\
        --out /tmp/failures.json

    # 2) Emit shell to render one GIF per failed map (symlink map_XXX.bin -> staging/map_000.bin)
    python analyze/failure_visualize.py \\
        --emit-viz-commands-only /tmp/failures.json \\
        --model /data/puffer/experiments/reactive_nominal/puffer_drive_MP1.pt \\
        --viz-use-xvfb \\
        --write-viz-script /tmp/run_failure_viz.sh

    # Optional: live rollout scan (aggregate vec_log only)
    python analyze/failure_visualize.py \\
        --env puffer_drive --model /path/to/policy.pt \\
        --map-dir /data/puffer/resources/drive/binaries/training \\
        --num-maps 8 --max-steps 400 --out /tmp/failures.json
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import asdict, dataclass, field
from glob import glob
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch

    import pufferlib.pytorch
    import pufferlib.vector as vector
    from pufferlib.pufferl import load_config, load_env, load_policy

    _IMPORT_ERROR = None
except ImportError as exc:
    torch = None  # type: ignore
    _IMPORT_ERROR = exc
    load_config = load_env = load_policy = None  # type: ignore
    vector = None  # type: ignore


@dataclass
class RolloutFailureRecord:
    """Aggregate collision signal from a short live rollout scan."""

    seed: int
    first_collision_step: int
    max_ego_collision_rate: float
    max_collision_rate: float
    map_ids_snapshot: List[int] = field(default_factory=list)
    notes: str = ""


@dataclass
class SceneFailureRecord:
    """One map/scene flagged from a zeroshot scenario log JSON."""

    map_id: int
    scenario_key: str
    ego_collision_rate: float
    collision_rate: float
    metrics: Dict[str, float] = field(default_factory=dict)
    source_log: str = ""
    pair_tag: str = ""
    notes: str = ""


def _collision_trigger(metrics: Dict[str, float], ego_threshold: float, any_threshold: float) -> bool:
    ego = metrics.get("ego_collision_rate", 0.0)
    coll = metrics.get("collision_rate", 0.0)
    ego_n = metrics.get("ego_collisions_per_agent", 0.0)
    coll_n = metrics.get("collisions_per_agent", 0.0)
    if ego > ego_threshold or ego_n > ego_threshold:
        return True
    if coll > any_threshold or coll_n > any_threshold:
        return True
    return False


def _float_metrics(row: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def load_scenario_log_entries(path: str) -> Dict[str, Dict[str, Any]]:
    """Load per-scene metrics dict (supports legacy ``scenario`` list format)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {}

    entries: Dict[str, Dict[str, Any]] = {}
    legacy = raw.get("scenario")
    if isinstance(legacy, list):
        for row in legacy:
            if not isinstance(row, dict):
                continue
            key = None
            if "map_id" in row:
                key = str(int(row["map_id"]))
            elif "scenario_id" in row:
                key = str(int(row["scenario_id"]))
            if key:
                entries[key] = dict(row)

    for key, value in raw.items():
        if key == "scenario" or not isinstance(value, dict):
            continue
        entries[str(key)] = dict(value)
    return entries


def failures_from_scenario_log(
    log_path: str,
    *,
    ego_collision_threshold: float,
    collision_threshold: float,
    max_failures: Optional[int] = None,
    pair_tag: Optional[str] = None,
) -> List[SceneFailureRecord]:
    """Parse one ``MP1_vs_MP2.json`` (or similar) and return scenes above collision thresholds."""
    if pair_tag is None:
        pair_tag = os.path.splitext(os.path.basename(log_path))[0]
    entries = load_scenario_log_entries(log_path)
    failures: List[SceneFailureRecord] = []

    for scenario_key, row in entries.items():
        metrics = _float_metrics(row)
        if not _collision_trigger(metrics, ego_collision_threshold, collision_threshold):
            continue
        map_id = int(metrics.get("map_id", scenario_key))
        failures.append(
            SceneFailureRecord(
                map_id=map_id,
                scenario_key=scenario_key,
                ego_collision_rate=metrics.get("ego_collision_rate", 0.0),
                collision_rate=metrics.get("collision_rate", 0.0),
                metrics=metrics,
                source_log=os.path.abspath(log_path),
                pair_tag=pair_tag,
                notes="From zeroshot scenario log (map index key; not Waymo hex scenario_id).",
            )
        )

    failures.sort(key=lambda r: (r.ego_collision_rate, r.collision_rate), reverse=True)
    if max_failures is not None and max_failures > 0:
        failures = failures[:max_failures]
    return failures


def failures_from_scenario_log_path(
    path: str,
    *,
    ego_collision_threshold: float,
    collision_threshold: float,
    max_failures_per_log: Optional[int] = None,
    max_failures_total: Optional[int] = None,
) -> List[SceneFailureRecord]:
    """Accept a single JSON file or a directory of ``*_vs_*.json`` logs."""
    if os.path.isdir(path):
        patterns = [
            os.path.join(path, "*.json"),
            os.path.join(path, "**", "*.json"),
        ]
        files: List[str] = []
        for pat in patterns:
            files.extend(glob(pat, recursive=True))
        files = sorted({os.path.abspath(f) for f in files})
    else:
        files = [os.path.abspath(path)]

    all_failures: List[SceneFailureRecord] = []
    for fp in files:
        if not os.path.isfile(fp):
            continue
        all_failures.extend(
            failures_from_scenario_log(
                fp,
                ego_collision_threshold=ego_collision_threshold,
                collision_threshold=collision_threshold,
                max_failures=max_failures_per_log,
            )
        )

    all_failures.sort(key=lambda r: (r.ego_collision_rate, r.collision_rate), reverse=True)
    if max_failures_total is not None and max_failures_total > 0:
        all_failures = all_failures[:max_failures_total]
    return all_failures


def prepare_single_map_staging(map_dir: str, map_id: int, staging_dir: str) -> str:
    """Symlink ``map_{id:03d}.bin`` as ``map_000.bin`` so ``num_maps=1`` loads that scene only."""
    os.makedirs(staging_dir, exist_ok=True)
    src = os.path.join(map_dir, f"map_{map_id:03d}.bin")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Missing map binary: {src}")
    dst = os.path.join(staging_dir, "map_000.bin")
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)
    return staging_dir


def _extract_log_metrics(info_list: Any) -> Optional[Dict[str, float]]:
    if not info_list:
        return None
    item = info_list[0]
    if isinstance(item, list) and item:
        item = item[0]
    if not isinstance(item, dict):
        return None
    out: Dict[str, float] = {}
    for key in ("ego_collision_rate", "collision_rate", "ego_collisions_per_agent", "collisions_per_agent"):
        if key not in item:
            continue
        try:
            out[key] = float(item[key])
        except (TypeError, ValueError):
            continue
    return out or None


def _map_ids_from_reset_info(infos: Any) -> List[int]:
    flat: List[Dict[str, Any]] = []
    if isinstance(infos, dict):
        flat = [infos]
    elif isinstance(infos, (list, tuple)):
        for x in infos:
            if isinstance(x, dict):
                flat.append(x)
            elif isinstance(x, (list, tuple)):
                for y in x:
                    if isinstance(y, dict):
                        flat.append(y)
    mids: List[int] = []
    for d in flat:
        raw = d.get("map_ids")
        if raw is None:
            continue
        arr = np.asarray(raw).reshape(-1)
        mids.extend(int(x) for x in arr.tolist())
    return mids


def scan_rollout_for_collisions(
    *,
    env_name: str,
    model_path: str,
    map_dir: str,
    num_maps: int,
    max_steps: int,
    seed: int,
    ego_collision_threshold: float,
    collision_threshold: float,
    device: Optional[str] = None,
) -> Tuple[Optional[RolloutFailureRecord], Dict[str, Any]]:
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "pufferlib / torch import failed. Run from repo root with the same env as ``puffer`` CLI."
        ) from _IMPORT_ERROR

    args = load_config(env_name)
    args["env"]["map_dir"] = map_dir
    args["env"]["num_maps"] = int(num_maps)
    args["env"]["sequential_map_sampling"] = True
    args["vec"] = dict(backend="PufferEnv", num_envs=1)
    args["load_model_path"] = model_path
    if device:
        args["train"]["device"] = device

    vecenv = load_env(env_name, args)
    policy = load_policy(args, vecenv, env_name).eval()
    num_agents = vecenv.observation_space.shape[0]
    dev = args["train"]["device"]

    state: Dict[str, Any] = {}
    if args["train"]["use_rnn"]:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=dev),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=dev),
        )

    ob, infos = vector.reset(vecenv, seed=seed)
    map_snapshot = _map_ids_from_reset_info(infos)

    first_hit: Optional[int] = None
    max_ego = 0.0
    max_coll = 0.0

    for t in range(max_steps):
        with torch.inference_mode():
            ob_t = torch.as_tensor(ob, device=dev)
            logits, _value = policy.forward_eval(ob_t, state)
            action, _lp, _ = pufferlib.pytorch.sample_logits(logits)
            action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)
            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high)

        _ob, _r, _d, _tr, info_list = vector.step(vecenv, action_np)
        ob = _ob

        metrics = _extract_log_metrics(info_list)
        if metrics:
            max_ego = max(max_ego, metrics.get("ego_collision_rate", 0.0))
            max_coll = max(max_coll, metrics.get("collision_rate", 0.0))
            if _collision_trigger(metrics, ego_collision_threshold, collision_threshold):
                if first_hit is None:
                    first_hit = t

    meta = {
        "env_name": env_name,
        "map_dir": map_dir,
        "num_maps": num_maps,
        "max_steps": max_steps,
        "seed": seed,
        "ego_collision_threshold": ego_collision_threshold,
        "collision_threshold": collision_threshold,
        "max_ego_collision_rate_seen": max_ego,
        "max_collision_rate_seen": max_coll,
    }

    if first_hit is None:
        return None, meta

    rec = RolloutFailureRecord(
        seed=seed,
        first_collision_step=first_hit,
        max_ego_collision_rate=max_ego,
        max_collision_rate=max_coll,
        map_ids_snapshot=map_snapshot,
        notes="Aggregate vec_log collision from live rollout scan.",
    )
    return rec, meta


def _failure_row_tag(row: Dict[str, Any], index: int) -> str:
    if row.get("pair_tag") and row.get("map_id") is not None:
        return f"{row['pair_tag']}_map{int(row['map_id']):03d}"
    if row.get("map_id") is not None:
        return f"map{int(row['map_id']):03d}"
    seed = int(row.get("seed", 0))
    return f"failure_{index:03d}_seed{seed}"


def emit_viz_commands(
    records_path: str,
    *,
    env_name: str,
    model_path: str,
    map_dir: str,
    python_exe: str,
    xvfb: bool,
    frames: int,
    fps: float,
    out_dir: str,
    staging_parent: str,
    episode_length: int,
) -> str:
    """Build shell script: one ``analyze/viz.py`` call per failed scene (single-map staging)."""
    with open(records_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    failures: List[Dict[str, Any]] = payload.get("failures", [])
    meta = payload.get("meta", {})
    map_dir = meta.get("map_dir") or map_dir

    lines: List[str] = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(os.getcwd())}"]
    analyze_dir = os.path.dirname(os.path.abspath(__file__))
    viz_py = os.path.join(analyze_dir, "viz.py")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(staging_parent, exist_ok=True)

    total = len(failures)
    for i, row in enumerate(failures):
        tag = _failure_row_tag(row, i)
        out_gif = os.path.join(out_dir, f"{tag}.gif")
        lines.append(f"echo '[{i + 1}/{total}] {tag} -> {out_gif}'")
        lines.append(f"if [[ -f {shlex.quote(out_gif)} ]]; then")
        lines.append("  echo '      skip (exists)'")
        lines.append("else")

        if row.get("map_id") is not None:
            map_id = int(row["map_id"])
            src_bin = os.path.join(map_dir, f"map_{map_id:03d}.bin")
            cmd = [
                python_exe,
                viz_py,
                "--env",
                env_name,
                "--model",
                model_path,
                "--map-bin",
                os.path.abspath(src_bin),
                "--out",
                out_gif,
                "--view",
                "topdown",
                "--frame-skip",
                "5",
                "--heartbeat-sec",
                "30",
            ]
            weights_bin = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "pufferlib",
                "resources",
                "drive",
                "puffer_drive_weights.bin",
            )
            if os.path.isfile(weights_bin):
                cmd.extend(["--reuse-weights", weights_bin])
            if xvfb:
                cmd.append("--use-xvfb")
        else:
            seed = int(row.get("seed", 0))
            cmd = [
                python_exe,
                viz_py,
                "--env",
                env_name,
                "--model",
                model_path,
                "--out",
                out_gif,
                "--",
                "--eval.wosac-num-maps",
                "3",
                "--train.seed",
                str(seed),
            ]

        if xvfb and row.get("map_id") is None:
            line = "xvfb-run -a -s " + shlex.quote("-screen 0 1920x1080x24") + " " + " ".join(shlex.quote(c) for c in cmd)
        else:
            line = " ".join(shlex.quote(c) for c in cmd)
        lines.append(f"  {line}")
        lines.append("fi")

    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", type=str, default="puffer_drive", help="Env name for load_config / load_env.")
    p.add_argument("--model", type=str, default=None, help="Checkpoint .pt (ego policy for viz).")
    p.add_argument("--map-dir", type=str, default=None, help="Drive map_dir (parent of map_XXX.bin).")

    p.add_argument(
        "--from-scenario-log",
        type=str,
        default=None,
        metavar="PATH",
        help="Zeroshot scenario log JSON file or directory of *_vs_*.json logs.",
    )
    p.add_argument(
        "--max-failures-per-log",
        type=int,
        default=None,
        help="Cap failures taken from each scenario log file (after sorting by ego collision).",
    )
    p.add_argument(
        "--max-failures-total",
        type=int,
        default=None,
        help="Cap total failures when scanning a directory of logs.",
    )

    p.add_argument("--num-maps", type=int, default=5, help="For live --scan rollout only.")
    p.add_argument("--max-steps", type=int, default=300, help="For live --scan rollout only.")
    p.add_argument("--seed", type=int, default=0, help="vecenv reset seed for live scan.")
    p.add_argument("--ego-collision-threshold", type=float, default=1e-3, help="Trigger if ego_collision_rate exceeds this.")
    p.add_argument("--collision-threshold", type=float, default=1e-2, help="Trigger if collision_rate exceeds this.")
    p.add_argument("--device", type=str, default=None, help="Override train device for live scan.")
    p.add_argument("--out", type=str, default="failure_visualize_records.json", help="Write JSON results here.")

    p.add_argument(
        "--emit-viz-commands-only",
        type=str,
        default=None,
        metavar="RECORDS_JSON",
        help="Read failures JSON and write/print a viz shell script.",
    )
    p.add_argument("--viz-out-dir", type=str, default="./failure_viz_runs", help="GIF output directory.")
    p.add_argument("--viz-staging-dir", type=str, default="./failure_viz_map_staging", help="Per-scene symlink dirs.")
    p.add_argument("--viz-frames", type=int, default=91, help="Frames for GIF (WOMD episode length).")
    p.add_argument("--viz-fps", type=float, default=15.0, help="GIF FPS.")
    p.add_argument("--viz-episode-length", type=int, default=91, help="--env.episode-length forwarded to eval.")
    p.add_argument("--viz-python", type=str, default=sys.executable, help="Python for viz.py.")
    p.add_argument("--viz-use-xvfb", action="store_true", help="Wrap viz with xvfb-run -a.")
    p.add_argument(
        "--write-viz-script",
        type=str,
        default=None,
        metavar="SH_PATH",
        help="Write shell script path (with --emit-viz-commands-only).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.emit_viz_commands_only:
        model = args.model or input_json_model(args.emit_viz_commands_only)
        if not model:
            print("Need --model or 'model_path' inside JSON for emit mode.", file=sys.stderr)
            return 2
        if not args.map_dir:
            try:
                with open(args.emit_viz_commands_only, "r", encoding="utf-8") as f:
                    args.map_dir = json.load(f).get("meta", {}).get("map_dir")
            except OSError:
                pass
        if not args.map_dir:
            print("Need --map-dir or map_dir in JSON meta for per-scene staging.", file=sys.stderr)
            return 2

        script = emit_viz_commands(
            args.emit_viz_commands_only,
            env_name=args.env,
            model_path=os.path.abspath(model),
            map_dir=os.path.abspath(args.map_dir),
            python_exe=args.viz_python,
            xvfb=args.viz_use_xvfb,
            frames=args.viz_frames,
            fps=args.viz_fps,
            out_dir=os.path.abspath(args.viz_out_dir),
            staging_parent=os.path.abspath(args.viz_staging_dir),
            episode_length=args.viz_episode_length,
        )
        if args.write_viz_script:
            with open(args.write_viz_script, "w", encoding="utf-8") as f:
                f.write(script)
            os.chmod(args.write_viz_script, 0o755)
            print(f"Wrote {args.write_viz_script}")
        else:
            print(script, end="")
        return 0

    if args.from_scenario_log:
        if not args.map_dir:
            print("--map-dir is required with --from-scenario-log (for viz staging paths in meta).", file=sys.stderr)
            return 2
        failures = failures_from_scenario_log_path(
            args.from_scenario_log,
            ego_collision_threshold=args.ego_collision_threshold,
            collision_threshold=args.collision_threshold,
            max_failures_per_log=args.max_failures_per_log,
            max_failures_total=args.max_failures_total,
        )
        meta = {
            "source": "scenario_log",
            "scenario_log_path": os.path.abspath(args.from_scenario_log),
            "map_dir": os.path.abspath(args.map_dir),
            "ego_collision_threshold": args.ego_collision_threshold,
            "collision_threshold": args.collision_threshold,
            "failure_count": len(failures),
        }
        payload: Dict[str, Any] = {
            "meta": meta,
            "model_path": os.path.abspath(args.model) if args.model else None,
            "failures": [asdict(f) for f in failures],
        }
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Found {len(failures)} failed scene(s). Wrote {out_path}")
        if failures:
            print("Example:", asdict(failures[0]))
        if args.model:
            print(
                "\nNext:\n  python analyze/failure_visualize.py "
                f"--emit-viz-commands-only {shlex.quote(out_path)} "
                f"--model {shlex.quote(args.model)} --map-dir {shlex.quote(args.map_dir)} "
                "--viz-use-xvfb "
                f"--write-viz-script {shlex.quote(os.path.join(os.path.dirname(out_path), 'run_failure_viz.sh'))}"
            )
        return 0

    if not args.model or not args.map_dir:
        print("Use --from-scenario-log, or provide --model and --map-dir for live scan.", file=sys.stderr)
        return 2

    rec, meta = scan_rollout_for_collisions(
        env_name=args.env,
        model_path=args.model,
        map_dir=args.map_dir,
        num_maps=args.num_maps,
        max_steps=args.max_steps,
        seed=args.seed,
        ego_collision_threshold=args.ego_collision_threshold,
        collision_threshold=args.collision_threshold,
        device=args.device,
    )

    payload = {"meta": meta, "model_path": os.path.abspath(args.model), "failures": []}
    if rec is not None:
        payload["failures"].append(asdict(rec))
        print("Collision-like signal detected:", asdict(rec))
    else:
        print("No collision trigger in this rollout.")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


def input_json_model(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            p = json.load(f)
        m = p.get("model_path")
        return str(m) if m else None
    except OSError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
