#!/usr/bin/env python3
"""Smoke-test Drive_PBT minimum_distance + corpus entity-id lookup.

Prints a small table each step so you can verify:
  - ego corpus slots -> -inf after reset
  - visible partners -> finite min distance at corpus_idx
  - (map_id, entity_id) resolves via _entity_to_corpus

Example:
  python analyze/debug_minimum_distance.py \\
    --population-path /data/puffer/popul_lane_nominal \\
    --steps 5
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
    ref_ids = np.asarray(env.actions_agent_ids[0], dtype=np.int64)
    for (map_id, entity_id), corpus_idx in env._entity_to_corpus.items():
        out[int(corpus_idx)] = (int(map_id), int(entity_id))
    return out


def summarize(env, step: int, top_k: int = 12) -> None:
    md = env.minimum_distance[:, 0]
    n_ego = int(np.sum(md == -np.inf))
    finite = np.isfinite(md) & (md != -np.inf)
    n_partner = int(np.sum(finite))
    n_inf = int(np.sum(np.isinf(md)))

    print(f"\n=== step {step} ===")
    print(f"corpus size={env.total_agents}  ego(-inf)={n_ego}  partner(updated)={n_partner}  untouched(inf)={n_inf}")

    if n_partner == 0:
        print(
            "(no partner min_dist yet — visible hits may all map to ego corpus slots marked -inf; "
            "see skipped_ego_slot in audit below)"
        )
        return

    idx = np.flatnonzero(finite)
    order = idx[np.argsort(md[idx])][:top_k]
    rev = _reverse_entity_lookup(env)
    print(f"{'corpus':>8} {'map':>6} {'entity':>8} {'min_dist(m)':>12}")
    for g in order:
        map_id, entity_id = rev.get(int(g), (-1, -1))
        print(f"{g:8d} {map_id:6d} {entity_id:8d} {md[g]:12.2f}")


def _step_visible_audit(env, max_rows: int = 12) -> None:
    """One-step audit: visible partner -> corpus_idx lookup + ego-slot skip."""
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--population-path", default="/data/puffer/popul_lane_nominal")
    p.add_argument("--map-dir", default="/data/puffer/resources/drive/binaries/training")
    p.add_argument("--num-maps", type=int, default=2)
    p.add_argument("--num-agents", type=int, default=64)
    p.add_argument("--ego-ratio", type=float, default=0.25)
    p.add_argument("--pbt-mode", default="reactive", choices=["reactive", "replay"])
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
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
        resample_frequency=10_000,
    )

    obs, _ = env.reset(seed=args.seed)
    print(f"live maps={env.map_ids[:env.num_envs]}  ego_indices={env.ego_indices.tolist()}")
    print(f"corpus lookup entries={len(env._entity_to_corpus)}")
    summarize(env, step=0)
    _step_visible_audit(env)

    for t in range(1, args.steps + 1):
        actions = np.zeros((env.num_agents, 1), dtype=np.int32)
        env.step(actions)
        summarize(env, step=t)
        if t == 1:
            _step_visible_audit(env)

    print(
        "\nInterpretation:"
        "\n  hit>0           -> (map_id, entity_id) lookup OK"
        "\n  skipped_ego_slot -> visible agent is a live ego; corpus slot stays -inf (expected)"
        "\n  would_update_partner>0 -> partner corpus slots getting finite min_dist"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
