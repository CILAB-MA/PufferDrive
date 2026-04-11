#!/usr/bin/env python3
"""Run model visualization via pufferl eval and save GIF output.

This is a thin wrapper around:
    python -m pufferlib.pufferl eval <env_name> ...

It reuses the existing eval/render pipeline in ``pufferlib/pufferl.py`` and
provides a simpler interface for saving rollout visualizations.
"""

import argparse
import os
import shlex
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained model and save rollout GIF."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="puffer_drive",
        help="Environment name passed to pufferl eval.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Checkpoint path (.pt) for --load-model-path.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="viz.gif",
        help="Output GIF path.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Number of frames to save (maps to --save-frames).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="GIF FPS (maps to --fps).",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="raylib",
        choices=["auto", "human", "ansi", "rgb_array", "raylib", "None"],
        help="Render mode passed to pufferl eval.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to launch pufferl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command only (do not execute).",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help=(
            "Additional args forwarded to pufferl eval. "
            "Use '--' before them, e.g. -- --eval.map-dir /path --env.num-maps 4"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    extra = list(args.extra_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    cmd = [
        args.python,
        "-m",
        "pufferlib.pufferl",
        "eval",
        args.env,
        "--load-model-path",
        os.path.abspath(args.model),
        "--save-frames",
        str(max(args.frames, 1)),
        "--gif-path",
        out_path,
        "--fps",
        str(args.fps),
        "--render-mode",
        args.render_mode,
        *extra,
    ]

    print("Running:")
    print(" ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return

    subprocess.run(cmd, check=True)
    if os.path.isfile(out_path):
        print(f"\nSaved visualization: {out_path}")
    else:
        print("\nFinished, but output GIF was not found. Check eval logs/options.")


if __name__ == "__main__":
    main()

