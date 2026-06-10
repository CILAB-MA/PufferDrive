#!/usr/bin/env python3
"""Smoke-test Drive_PBT minimum_distance + score_metric at resample.

Prints per-step minimum_distance stats and, at each resample boundary:
  - pre-commit preview (partner slots, ego return, commit candidates)
  - post-commit score_metric updates

Example:
  python analyze/debug_minimum_distance.py \\
    --population-path /data/puffer/popul_lane_nominal \\
    --steps 25 --resample-frequency 20
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.ocean.drive_pbt.drive_pbt import Drive_PBT


def _reverse_entity_lookup(env):
    out = {}
    if env._entity_to_corpus is None:
        return out
    for (map_id, entity_id), corpus_idx in env._entity_to_corpus.items():
        out[int(corpus_idx)] = (int(map_id), int(entity_id))
    return out


def summarize(env, step: int, top_k: int = 8) -> None:
    md = env.minimum_distance[:, 0]
    n_ego = int(np.sum(md == -np.inf))
    finite = np.isfinite(md) & (md != -np.inf)
    n_partner = int(np.sum(finite))
    n_inf = int(np.sum(np.isinf(md)))

    print(f"\n=== step {step} tick={env.tick} ===")
    print(f"corpus size={env.total_agents}  ego(-inf)={n_ego}  partner(updated)={n_partner}  untouched(inf)={n_inf}")

    if n_partner == 0:
        return

    idx = np.flatnonzero(finite)
    order = idx[np.argsort(md[idx])][:top_k]
    rev = _reverse_entity_lookup(env)
    mi = env.minimum_index[:, 0]
    print(f"{'corpus':>8} {'map':>6} {'entity':>8} {'ego_idx':>8} {'min_dist(m)':>12}")
    for g in order:
        map_id, entity_id = rev.get(int(g), (-1, -1))
        print(f"{g:8d} {map_id:6d} {entity_id:8d} {mi[g]:8d} {md[g]:12.2f}")


def _step_visible_audit(env, max_rows: int = 8) -> None:
    rel_xy = env._partner_obs()[:, :, :2]
    visible = (rel_xy[:, :, 0] != 0) | (rel_xy[:, :, 1] != 0)
    partner_states = env.get_global_partner_state()
    other_ids = partner_states["other_id"]
    ao = np.asarray(env.agent_offsets, dtype=np.int64)
    md = env.minimum_distance[:, 0]

    hits = misses = skipped_ego = would_update = 0
    rows = []
    for env_i in range(env.num_envs):
        map_id = int(env.map_ids[env_i])
        cur, nxt = ao[env_i], ao[env_i + 1]
        for ego_idx in env.ego_indices:
            if ego_idx < cur or ego_idx >= nxt:
                continue
            for slot in np.flatnonzero(visible[ego_idx]):
                entity_id = int(other_ids[ego_idx, slot])
                corpus_idx = env._corpus_index(map_id, entity_id)
                dist = float(np.linalg.norm(rel_xy[ego_idx, slot]) / 0.02)
                if corpus_idx is None:
                    misses += 1
                    slot_kind = "miss"
                else:
                    hits += 1
                    if md[corpus_idx] == -np.inf:
                        skipped_ego += 1
                        slot_kind = "ego(-inf)"
                    else:
                        would_update += 1
                        slot_kind = "partner"
                if len(rows) < max_rows:
                    rows.append((map_id, entity_id, corpus_idx, dist, slot_kind))
    print(
        f"visible lookup: hit={hits} miss={misses}  "
        f"skipped_ego_slot={skipped_ego}  would_update_partner={would_update}"
    )
    if rows:
        print(f"{'map':>6} {'entity':>8} {'corpus':>8} {'dist(m)':>10} {'slot':>12}")
        for map_id, entity_id, corpus_idx, dist, slot_kind in rows:
            c = "-" if corpus_idx is None else str(corpus_idx)
            print(f"{map_id:6d} {entity_id:8d} {c:>8} {dist:10.2f} {slot_kind:>12}")


def summarize_ego_returns(env, top_k: int = 8) -> None:
    """Per-ego rollout return (score_metric uses one linked ego per partner slot, not this sum)."""
    ego_ret = env._episode_return[env.ego_indices]
    print(
        f"ego rollout return: sum={float(ego_ret.sum()):.3f}  "
        f"mean={float(ego_ret.mean()):.3f}  "
        f"min={float(ego_ret.min()):.3f}  max={float(ego_ret.max()):.3f}  "
        f"steps={env.tick}"
    )
    order = np.argsort(np.abs(ego_ret))[::-1]
    print(f"{'ego_idx':>8} {'return':>10} {'last_step_r':>12}")
    for i in order[:top_k]:
        idx = int(env.ego_indices[i])
        print(f"{idx:8d} {ego_ret[i]:10.3f} {env.rewards[idx]:12.3f}")


def summarize_commit_preview(env, rollout: int, top_k: int = 10) -> None:
    md = env.minimum_distance[:, 0]
    partner_mask = np.isfinite(md) & (md != -np.inf)
    committed = partner_mask & (env.minimum_index[:, 0] >= 0)

    print(f"\n--- resample preview (rollout {rollout}, tick={env.tick}) ---")
    print(
        f"partner_slots={int(partner_mask.sum())}  "
        f"commit_candidates={int(committed.sum())}"
    )
    summarize_ego_returns(env)
    if committed.sum() == 0:
        print("(no commit candidates — visible partner never hit a corpus partner slot this rollout)")
        return

    rev = _reverse_entity_lookup(env)
    print("score_metric will copy linked ego return (not ego sum above):")
    print(f"{'corpus':>8} {'map':>6} {'entity':>8} {'ego':>6} {'dist':>8} {'return':>10}")
    for g in np.flatnonzero(committed)[:top_k]:
        ego_idx = int(env.minimum_index[g, 0])
        map_id, entity_id = rev.get(int(g), (-1, -1))
        print(
            f"{g:8d} {map_id:6d} {entity_id:8d} {ego_idx:6d} "
            f"{md[g]:8.2f} {env._episode_return[ego_idx]:10.3f}"
        )


def summarize_resample_commit(env, rollout: int, score_before: np.ndarray, top_k: int = 10) -> None:
    sm = env.score_metric[:, 0]
    changed = np.flatnonzero(sm != score_before)
    md = env.minimum_distance[:, 0]
    n_partner = int(np.sum(np.isfinite(md) & (md != -np.inf)))

    print(f"\n--- resample committed (rollout {rollout}) ---")
    print(
        f"score_metric changed={changed.size}  "
        f"_episode_return sum={float(env._episode_return.sum()):.3f} (expect 0)  "
        f"partner slots after reset={n_partner} (expect 0)"
    )
    if changed.size == 0:
        print("(score_metric unchanged — check commit_candidates / ego return)")
        return

    rev = _reverse_entity_lookup(env)
    order = changed[np.argsort(np.abs(sm[changed]))[::-1]][:top_k]
    print(f"{'corpus':>8} {'map':>6} {'entity':>8} {'before':>10} {'after':>10}")
    for g in order:
        map_id, entity_id = rev.get(int(g), (-1, -1))
        print(
            f"{g:8d} {map_id:6d} {entity_id:8d} "
            f"{score_before[g]:10.3f} {sm[g]:10.3f}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--population-path", default="/data/puffer/popul_lane_nominal")
    p.add_argument("--map-dir", default="/data/puffer/resources/drive/binaries/training")
    p.add_argument("--num-maps", type=int, default=2)
    p.add_argument("--num-agents", type=int, default=64)
    p.add_argument("--ego-ratio", type=float, default=0.25)
    p.add_argument("--pbt-mode", default="reactive", choices=["reactive", "replay"])
    p.add_argument("--resample-frequency", type=int, default=20)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose-steps", action="store_true", help="Print minimum_distance table every step")
    args = p.parse_args()

    saved = os.path.join(args.population_path, "saved")
    for name in ("other_actions_agent_offsets.npy", "other_actions_map_ids.npy", "other_actions_agent_ids.npy"):
        path = os.path.join(saved, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing corpus file: {path}")

    env = Drive_PBT(
        num_maps=args.num_maps,
        num_agents=args.num_agents,
        map_dir=args.map_dir,
        ego_ratio=args.ego_ratio,
        population_path=args.population_path,
        pbt_mode=args.pbt_mode,
        agent_sampling=True,
        render_mode=None,
        episode_length=91,
        resample_frequency=args.resample_frequency,
    )

    env.reset(seed=args.seed)
    print(f"resample_frequency={args.resample_frequency}  steps={args.steps}")
    print(f"live maps={env.map_ids[:env.num_envs]}  ego_indices={env.ego_indices.tolist()}")
    print(f"corpus lookup entries={len(env._entity_to_corpus)}")
    summarize(env, step=0)
    _step_visible_audit(env)

    rollout = 0
    for t in range(1, args.steps + 1):
        if env.tick + 1 == args.resample_frequency:
            summarize_commit_preview(env, rollout)

        score_before = env.score_metric[:, 0].copy()
        env.step(np.zeros((env.num_agents, 1), dtype=np.int32))

        if env.tick == 0 and t > 0:
            summarize_resample_commit(env, rollout, score_before)
            rollout += 1

        if args.verbose_steps or env.tick == 0 or t <= 2:
            summarize(env, step=t)
        if t == 1:
            _step_visible_audit(env)

    print(
        "\nNotes:"
        "\n  - ego rollout return sum CAN be large while score_metric stays small:"
        "\n    score_metric[g] = _episode_return[minimum_index[g]] (closest ego only)"
        "\n  - rollout with partner_slots=0: returns still accumulate, but nothing commits"
        "\n  - many egos show ~-0.02 over 20 steps with zero actions (only tiny per-step reward)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
