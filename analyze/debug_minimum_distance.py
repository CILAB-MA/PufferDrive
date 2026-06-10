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


def _tracked_mask(env):
    md = env.minimum_distance
    return np.isfinite(md) & (env.minimum_ego_idx >= 0) & (env.minimum_other_idx >= 0)


def summarize(env, step: int, top_k: int = 8) -> None:
    md = env.minimum_distance
    tracked = _tracked_mask(env)
    n_partner = int(np.sum(tracked))
    n_untracked = int(np.sum(np.isinf(md)))

    print(f"\n=== step {step} tick={env.tick} ===")
    print(
        f"other agents={md.size}  partner(tracked)={n_partner}  "
        f"untracked(inf)={n_untracked}"
    )

    if n_partner == 0:
        return

    idx = np.flatnonzero(tracked)
    order = idx[np.argsort(md[idx])][:top_k]
    rev = _reverse_entity_lookup(env)
    print(
        f"{'slot':>6} {'other':>6} {'corpus':>8} {'map':>6} {'entity':>8} "
        f"{'ego_idx':>8} {'min_dist(m)':>12}"
    )
    for slot in order:
        corpus_idx = int(env.minimum_other_idx[slot])
        other_idx = int(env.other_indices_arr[slot])
        map_id, entity_id = rev.get(corpus_idx, (-1, -1))
        print(
            f"{slot:6d} {other_idx:6d} {corpus_idx:8d} {map_id:6d} {entity_id:8d} "
            f"{env.minimum_ego_idx[slot]:8d} {md[slot]:12.2f}"
        )


def _step_visible_audit(env, max_rows: int = 8) -> None:
    rel_xy = env._partner_obs()[:, :, :2]
    visible = (rel_xy[:, :, 0] != 0) | (rel_xy[:, :, 1] != 0)
    partner_states = env.get_global_partner_state()
    other_ids = partner_states["other_id"]
    entity_to_agent = env._live_entity_to_agent()
    ao = np.asarray(env.agent_offsets, dtype=np.int64)

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
                partner_idx = entity_to_agent.get((map_id, entity_id))
                if corpus_idx is None:
                    misses += 1
                    slot_kind = "miss"
                elif partner_idx is None or partner_idx in env._ego_index_set:
                    skipped_ego += 1
                    slot_kind = "ego/skip"
                elif env._other_slot_for_agent(partner_idx) is None:
                    slot_kind = "not-other"
                else:
                    hits += 1
                    would_update += 1
                    slot_kind = "partner"
                if len(rows) < max_rows:
                    rows.append((map_id, entity_id, corpus_idx, dist, slot_kind))
    print(
        f"visible lookup: hit={hits} miss={misses}  "
        f"skipped_ego={skipped_ego}  would_update_partner={would_update}"
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
    tracked = _tracked_mask(env)
    md = env.minimum_distance

    print(f"\n--- resample preview (rollout {rollout}, tick={env.tick}) ---")
    print(
        f"partner_slots={int(tracked.sum())}  "
        f"commit_candidates={int(tracked.sum())}"
    )
    summarize_ego_returns(env)
    if tracked.sum() == 0:
        print("(no commit candidates — visible partner never matched a live other agent)")
        return

    rev = _reverse_entity_lookup(env)
    print("score_metric[other_slot] will copy linked ego return (not ego sum above):")
    print(
        f"{'slot':>6} {'corpus':>8} {'map':>6} {'entity':>8} {'other':>6} "
        f"{'ego':>6} {'dist':>8} {'return':>10}"
    )
    order = np.flatnonzero(tracked)
    order = order[np.argsort(md[order])][:top_k]
    for slot in order:
        corpus_idx = int(env.minimum_other_idx[slot])
        ego_idx = int(env.minimum_ego_idx[slot])
        other_idx = int(env.other_indices_arr[slot])
        map_id, entity_id = rev.get(corpus_idx, (-1, -1))
        print(
            f"{slot:6d} {corpus_idx:8d} {map_id:6d} {entity_id:8d} {other_idx:6d} "
            f"{ego_idx:6d} {md[slot]:8.2f} {env._episode_return[ego_idx]:10.3f}"
        )


def summarize_resample_commit(
    env,
    rollout: int,
    score_before: np.ndarray,
    new_score_before: np.ndarray | None,
    top_k: int = 10,
) -> None:
    changed_slots = np.flatnonzero(score_before != 0)
    n_partner = int(_tracked_mask(env).sum())

    print(f"\n--- resample committed (rollout {rollout}) ---")
    print(
        f"score_metric slots set={changed_slots.size}  "
        f"_episode_return sum={float(env._episode_return.sum()):.3f} (expect 0)  "
        f"partner slots after reset={n_partner}"
    )
    if changed_slots.size > 0:
        rev = _reverse_entity_lookup(env)
        print(f"{'slot':>6} {'corpus':>8} {'map':>6} {'entity':>8} {'score':>10}")
        order = changed_slots[np.argsort(np.abs(score_before[changed_slots]))[::-1]][:top_k]
        for slot in order:
            corpus_idx = int(env.minimum_other_idx[slot])
            map_id, entity_id = rev.get(corpus_idx, (-1, -1))
            print(
                f"{slot:6d} {corpus_idx:8d} {map_id:6d} {entity_id:8d} "
                f"{score_before[slot]:10.3f}"
            )
    elif changed_slots.size == 0:
        print("(no per-slot score_metric — check commit_candidates / ego return)")

    sampler = getattr(env, "agent_sampler", None)
    if sampler is not None:
        ns = sampler.new_score
        if new_score_before is None:
            changed_mask = ns != 0
        else:
            changed_mask = ns != new_score_before
        corpus_idx, policy_idx = np.where(changed_mask)
        print(f"agent_sampler.new_score changed={corpus_idx.size}")
        if corpus_idx.size > 0:
            rev = _reverse_entity_lookup(env)
            order = np.argsort(np.abs(ns[corpus_idx, policy_idx]))[::-1][:top_k]
            print(f"{'corpus':>8} {'policy':>6} {'map':>6} {'entity':>8} {'score':>10}")
            for i in order:
                g, p = int(corpus_idx[i]), int(policy_idx[i])
                map_id, entity_id = rev.get(g, (-1, -1))
                print(f"{g:8d} {p:6d} {map_id:6d} {entity_id:8d} {ns[g, p]:10.3f}")


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
    print(f"other agents={env.other_indices_arr.size}  corpus lookup entries={len(env._entity_to_corpus)}")
    summarize(env, step=0)
    _step_visible_audit(env)

    rollout = 0
    for t in range(1, args.steps + 1):
        if env.tick + 1 == args.resample_frequency:
            summarize_commit_preview(env, rollout)

        score_before = env.score_metric.copy()
        sampler = getattr(env, "agent_sampler", None)
        new_score_before = sampler.new_score.copy() if sampler is not None else None
        env.step(np.zeros((env.num_agents, 1), dtype=np.int32))

        if env.tick == 0 and t > 0:
            summarize_resample_commit(env, rollout, score_before, new_score_before)
            rollout += 1

        if args.verbose_steps or env.tick == 0 or t <= 2:
            summarize(env, step=t)
        if t == 1:
            _step_visible_audit(env)

    print(
        "\nNotes:"
        "\n  - minimum_distance is indexed by live other agent (size num_agents - num_ego)"
        "\n  - minimum_other_idx stores corpus index; minimum_ego_idx stores closest ego"
        "\n  - score_metric[slot] = _episode_return[minimum_ego_idx] at resample"
        "\n  - agent_sampler.new_score[corpus] from distance-filtered score_metric"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
