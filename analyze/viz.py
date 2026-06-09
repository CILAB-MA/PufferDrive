#!/usr/bin/env python3
"""Render PufferDrive rollouts to video/GIF via the C ``./visualize`` binary."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import IO, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained Drive policy on one map.")
    parser.add_argument("--env", type=str, default="puffer_drive")
    parser.add_argument("--model", type=str, required=True, help="Checkpoint .pt")
    parser.add_argument("--map-bin", type=str, required=True, help="Path to map_XXX.bin")
    parser.add_argument("--out", type=str, default="viz.gif", help="Output .gif or .mp4")
    parser.add_argument("--view", type=str, default="topdown", choices=["topdown", "agent", "both"])
    parser.add_argument("--frame-skip", type=int, default=5, help="Render every N sim steps.")
    parser.add_argument("--episode-length", type=int, default=91)
    parser.add_argument("--heartbeat-sec", type=int, default=30, help="Print alive message every N sec.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--visualize-bin", type=str, default="./visualize")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-xvfb", action="store_true", help="Required headless.")
    parser.add_argument(
        "--reuse-weights",
        nargs="?",
        const="pufferlib/resources/drive/puffer_drive_weights.bin",
        default=None,
        metavar="WEIGHTS_BIN",
        help="Skip export. Default bin: pufferlib/resources/drive/puffer_drive_weights.bin",
    )
    return parser.parse_args()


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _export_weights_fast(model_path: str, env_name: str, out_bin: str) -> None:
    """Export without spawning 16 vec workers (default drive.ini vec.num_workers=16)."""
    from pufferlib.pufferl import export, load_config, load_env, load_policy

    argv = sys.argv
    sys.argv = [
        argv[0],
        "export",
        env_name,
        "--load-model-path",
        os.path.abspath(model_path),
        "--env.num-maps",
        "1",
        "--vec.num-envs",
        "1",
        "--vec.num-workers",
        "1",
    ]
    try:
        args = load_config(env_name)
        args["load_model_path"] = os.path.abspath(model_path)
        vecenv = load_env(env_name, args)
        policy = load_policy(args, vecenv, env_name)
        export(env_name=env_name, args=args, vecenv=vecenv, policy=policy, path=out_bin, silent=True)
    finally:
        sys.argv = argv


def _mp4_to_gif(mp4_path: str, gif_path: str, fps: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            mp4_path,
            "-vf",
            f"fps={fps}",
            "-loop",
            "0",
            gif_path,
        ],
        check=True,
    )


def _heartbeat(stop: threading.Event, label: str, interval: int) -> None:
    while not stop.wait(interval):
        print(f"[{time.strftime('%H:%M:%S')}] still running: {label}", flush=True)


def _run_subprocess_stream(
    cmd: List[str],
    cwd: str,
    *,
    phase: str,
    heartbeat_sec: int,
    expected_frames: Optional[int] = None,
) -> None:
    env = os.environ.copy()
    env.setdefault("ASAN_OPTIONS", "exitcode=0:print_summary=0")

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    stop = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(stop, phase, heartbeat_sec), daemon=True)
    hb.start()

    bar = None
    if tqdm is not None and expected_frames is not None and expected_frames > 0:
        bar = tqdm(total=expected_frames, desc=phase, unit="frame", file=sys.stderr)

    assert proc.stdout is not None
    asan_hits = 0
    try:
        for line in proc.stdout:
            stripped = line.rstrip()
            if "AddressSanitizer:DEADLYSIGNAL" in line:
                asan_hits += 1
                if asan_hits == 1:
                    print(
                        "\nERROR: ./visualize crashed (ASAN build). Rebuild without sanitizer:\n"
                        "  bash scripts/build_ocean.sh visualize fast\n",
                        flush=True,
                    )
                if asan_hits >= 3:
                    proc.kill()
                    raise RuntimeError("visualize keeps segfaulting (rebuild with: bash scripts/build_ocean.sh visualize fast)")
            if stripped:
                print(stripped, flush=True)
            if bar is not None:
                if "Recording topdown" in line or "Recording agent" in line:
                    bar.set_description(f"{phase} (rendering)")
                if "Wrote" in line and "frames" in line:
                    bar.n = bar.total
                    bar.refresh()
            elif "Recording topdown" in line or "Wrote" in line:
                print(f">>> {stripped}", flush=True)
    finally:
        stop.set()
        rc = proc.wait()
    if bar is not None:
        bar.close()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def run_drive_visualize(
    *,
    model_path: str,
    map_bin: str,
    out_path: str,
    env_name: str,
    view: str,
    frame_skip: int,
    episode_length: int,
    heartbeat_sec: int,
    python_exe: str,
    visualize_bin: str,
    use_xvfb: bool,
    dry_run: bool,
    reuse_weights: Optional[str],
) -> None:
    repo = _repo_root()
    map_bin = os.path.abspath(map_bin)
    if not os.path.isfile(map_bin):
        raise FileNotFoundError(f"Map binary not found: {map_bin}")

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    want_gif = out_path.lower().endswith(".gif")
    mp4_path = out_path if out_path.lower().endswith(".mp4") else out_path.rsplit(".", 1)[0] + ".mp4"
    frame_skip = max(1, frame_skip)
    est_frames = max(1, (episode_length + frame_skip - 1) // frame_skip)

    if not use_xvfb:
        print("Warning: use --use-xvfb on headless machines.", flush=True)

    with tempfile.TemporaryDirectory(prefix="puffer_viz_") as tmp:
        if reuse_weights is not None:
            weights_bin = (
                os.path.abspath(reuse_weights)
                if os.path.isabs(reuse_weights)
                else os.path.join(repo, reuse_weights)
            )
            if not os.path.isfile(weights_bin):
                raise FileNotFoundError(f"Weights not found: {weights_bin}")
        else:
            weights_bin = os.path.join(tmp, "policy_weights.bin")
            if not dry_run:
                print("[1/3] Export weights (single env, no layer spam) ...", flush=True)
                t0 = time.time()
                _export_weights_fast(model_path, env_name, weights_bin)
                print(f"      done in {time.time() - t0:.1f}s -> {weights_bin}", flush=True)

        viz_bin = visualize_bin if os.path.isabs(visualize_bin) else os.path.join(repo, visualize_bin)
        if not os.path.isfile(viz_bin):
            raise FileNotFoundError(f"Missing {viz_bin}. Build: bash scripts/build_ocean.sh visualize fast")

        # visualize.c only honors --output-topdown when --output-agent is also set
        agent_mp4 = mp4_path.replace(".mp4", "_agent.mp4").replace(".gif", "_agent.mp4")
        cmd = [
            viz_bin,
            "--map-name",
            map_bin,
            "--policy-name",
            weights_bin,
            "--output-topdown",
            mp4_path,
            "--output-agent",
            agent_mp4,
            "--view",
            view,
            "--frame-skip",
            str(frame_skip),
        ]
        if use_xvfb:
            cmd = ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24", *cmd]

        print(f"[2/3] Visualize (~{est_frames} frames, CPU raylib; heartbeat every {heartbeat_sec}s) ...", flush=True)
        print("      ", " ".join(shlex.quote(c) for c in cmd), flush=True)
        if dry_run:
            return

        t0 = time.time()
        _run_subprocess_stream(
            cmd,
            repo,
            phase="visualize",
            heartbeat_sec=heartbeat_sec,
            expected_frames=est_frames if view != "both" else est_frames * 2,
        )
        print(f"      visualize done in {time.time() - t0:.1f}s", flush=True)

    if not os.path.isfile(mp4_path):
        raise RuntimeError(f"visualize did not create {mp4_path}")

    if want_gif:
        print(f"[3/3] ffmpeg -> GIF ...", flush=True)
        _mp4_to_gif(mp4_path, out_path, fps=15.0)
        print(f"Done: {out_path}", flush=True)
    else:
        print(f"Done: {mp4_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.env != "puffer_drive":
        raise SystemExit("Only puffer_drive is supported.")

    run_drive_visualize(
        model_path=args.model,
        map_bin=args.map_bin,
        out_path=args.out,
        env_name=args.env,
        view=args.view,
        frame_skip=args.frame_skip,
        episode_length=args.episode_length,
        heartbeat_sec=args.heartbeat_sec,
        python_exe=args.python,
        visualize_bin=args.visualize_bin,
        use_xvfb=args.use_xvfb,
        dry_run=args.dry_run,
        reuse_weights=args.reuse_weights,
    )


if __name__ == "__main__":
    main()
