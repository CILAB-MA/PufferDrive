#!/usr/bin/env python3
"""Stage A/B debugger for Drive_PBT partner sampling.

Stage A (default): small env, index/LUT identity + minimum_distance correctness.
Stage B: shorter episode_length and/or injected reset() within a rollout.

Examples:
  # Stage A — toy env, index + distance audits
  python analyze/debug_minimum_distance.py \\
    --population-path /data/puffer/popul_lane_nominal

  # Stage B — episode reset inside rollout (before resample)
  python analyze/debug_minimum_distance.py \\
    --population-path /data/puffer/popul_lane_nominal \\
    --episode-length 30 --resample-frequency 50 --inject-reset-at 15

  # Verbose tables every step
  python analyze/debug_minimum_distance.py --verbose-steps
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.ocean.drive_pbt.drive_pbt import Drive_PBT, _PARTNER_REL_SCALE

_PARTNER_REL_SCALE_F = float(_PARTNER_REL_SCALE)


@dataclass
class AuditReport:
    label: str
    ok: list[str] = field(default_factory=list)
    fail: list[str] = field(default_factory=list)

    def check(self, cond: bool, ok_msg: str, fail_msg: str) -> None:
        if cond:
            self.ok.append(ok_msg)
        else:
            self.fail.append(fail_msg)

    def merge(self, other: AuditReport) -> None:
        self.ok.extend(other.ok)
        self.fail.extend(other.fail)

    def print_summary(self) -> None:
        print(f"\n=== audit summary: {self.label} ===")
        print(f"  passed: {len(self.ok)}  failed: {len(self.fail)}")
        for msg in self.fail[:20]:
            print(f"  FAIL: {msg}")
        if len(self.fail) > 20:
            print(f"  ... and {len(self.fail) - 20} more failures")


def _reverse_entity_lookup(env):
    out = {}
    gid = env.global_ids
    if gid is None:
        return out
    for map_id in range(gid.shape[0]):
        row = gid[map_id]
        for entity_id in np.flatnonzero(row >= 0):
            out[int(row[entity_id])] = (int(map_id), int(entity_id))
    return out


def _tracked_mask(env):
    md = env.minimum_distance
    return (
        np.isfinite(md)
        & (env.minimum_ego_idx >= 0)
        & (env.minimum_other_global_idx >= 0)
    )


def _info_dict(env, info):
    if isinstance(info, list):
        return info[0] if info else {}
    return info or {}


def audit_slot_identity(env, label: str = "slot identity", max_failures: int = 8) -> AuditReport:
    """Stage A #2: local_idx, global_idx, LUT round-trip for each other slot."""
    rep = AuditReport(label)

    gid = env.global_ids
    lut = env._env_entity_to_other_slot
    map_per_agent = env._map_per_agent()
    live_entity_ids = env.get_global_partner_state()["ego_id"].astype(np.int64)

    n_other = int(env.other_indices_arr.size)
    rep.check(
        env.minimum_other_local_idx.shape == (n_other,)
        and env.minimum_other_global_idx.shape == (n_other,)
        and lut is not None,
        "tracking buffers allocated",
        "missing minimum_other_* or _env_entity_to_other_slot",
    )

    env_per_agent = env._env_per_agent()
    shown = 0
    pairs: dict[tuple[int, int], list[int]] = {}
    for slot in range(n_other):
        local = int(env.minimum_other_local_idx[slot])
        if local < 0:
            continue
        other_idx = int(env.other_indices_arr[slot])
        env_i = int(env_per_agent[other_idx])
        pairs.setdefault((env_i, local), []).append(slot)

    dupes = {k: v for k, v in pairs.items() if len(v) > 1}
    if dupes:
        sample = list(dupes.items())[:3]
        rep.fail.append(
            f"{len(dupes)} duplicate (env_i, entity) across slots (e.g. {sample})"
        )

    for slot in range(n_other):
        other_idx = int(env.other_indices_arr[slot])
        local = int(env.minimum_other_local_idx[slot])
        global_g = int(env.minimum_other_global_idx[slot])
        map_id = int(map_per_agent[other_idx])
        env_i = int(env_per_agent[other_idx])

        if local < 0:
            continue

        live_local = int(live_entity_ids[other_idx])
        rep.check(
            live_local == local,
            f"slot {slot}: live ego_id matches minimum_other_local_idx ({local})",
            f"slot {slot}: live ego_id={live_local} != minimum_other_local_idx={local} "
            f"(other_idx={other_idx}, env={env_i}, map={map_id})",
        )

        if 0 <= map_id < gid.shape[0] and 0 <= local < gid.shape[1]:
            expected_g = int(gid[map_id, local])
            rep.check(
                expected_g == global_g,
                f"slot {slot}: global_ids[{map_id},{local}] == minimum_other_global_idx",
                f"slot {slot}: global_ids[{map_id},{local}]={expected_g} != "
                f"minimum_other_global_idx={global_g}",
            )
        else:
            rep.check(
                False,
                "",
                f"slot {slot}: map/local out of global_ids bounds "
                f"(map={map_id}, local={local})",
            )

        if 0 <= env_i < lut.shape[0] and 0 <= local < lut.shape[1]:
            lut_slot = int(lut[env_i, local])
            rep.check(
                lut_slot == slot,
                f"slot {slot}: LUT round-trip lut[{env_i},{local}]==slot",
                f"slot {slot}: LUT[{env_i},{local}]={lut_slot} (expected {slot})",
            )
        else:
            rep.check(
                False,
                "",
                f"slot {slot}: env/local out of LUT bounds (env={env_i}, local={local})",
            )

        if rep.fail and shown < max_failures:
            shown += 1

    valid_slots = int(np.sum(env.minimum_other_local_idx >= 0))
    rep.check(
        valid_slots > 0,
        f"{valid_slots}/{n_other} other slots have valid local entity id",
        "no other slot has valid minimum_other_local_idx after reset",
    )
    return rep


def audit_policy_assignment(env, label: str = "policy assignment") -> AuditReport:
    rep = AuditReport(label)
    if env.pbt_mode != "reactive":
        return rep

    flat = getattr(env, "policy_per_slot_flatten", None)
    groups = getattr(env, "policy_per_slot", None)
    rep.check(flat is not None and groups is not None, "policy_per_slot buffers exist", "missing policy_per_slot")

    if flat is None or groups is None:
        return rep

    n_other = int(env.other_indices_arr.size)
    rep.check(flat.shape == (n_other,), f"policy_per_slot_flatten shape ({n_other},)", f"flat shape {flat.shape}")

    union = np.concatenate([np.asarray(g, dtype=np.int64) for g in groups if g.size], dtype=np.int64)
    rep.check(
        union.size == n_other and np.array_equal(np.sort(union), np.sort(env.other_indices_arr)),
        "policy_per_slot lists partition other_indices_arr",
        f"policy groups cover {union.size} agents, expected {n_other}",
    )

    info_groups = groups
    for policy_idx, agents in enumerate(info_groups):
        if agents.size == 0:
            continue
        slot_idx = np.searchsorted(env.other_indices_arr, agents)
        rep.check(
            np.all(flat[slot_idx] == policy_idx),
            f"policy {policy_idx}: {agents.size} agents",
            f"policy {policy_idx}: flat assignment mismatch",
        )
    return rep


def _bruteforce_minimum_distance(env):
    """Recompute per-slot min (dist, ego_idx) from partner obs (reference implementation)."""
    n_other = int(env.other_indices_arr.size)
    best_dist = np.full(n_other, np.inf, dtype=np.float64)
    best_ego = np.full(n_other, -1, dtype=np.int64)

    partner_states = env.get_global_partner_state()
    other_ids = partner_states["other_id"].astype(np.int64)
    rel_xy = env._partner_obs()[:, :, :2]
    dist = np.linalg.norm(rel_xy, axis=2) / _PARTNER_REL_SCALE_F
    env_per_agent = env._env_per_agent()
    env_ids = np.broadcast_to(env_per_agent[:, np.newaxis], other_ids.shape)
    lut = env._env_entity_to_other_slot

    ego_mask = np.zeros(env.num_agents, dtype=bool)
    ego_mask[env.ego_indices] = True
    valid = (
        ego_mask[:, np.newaxis]
        & (other_ids >= 0)
        & (env_ids >= 0)
        & (env_ids < lut.shape[0])
        & (other_ids < lut.shape[1])
    )
    other_slot = np.full(other_ids.shape, -1, dtype=np.int64)
    other_slot[valid] = lut[env_ids[valid], other_ids[valid]]
    valid &= other_slot >= 0

    ego_idx, pslot = np.where(valid)
    for i in range(ego_idx.size):
        slot = int(other_slot[ego_idx[i], pslot[i]])
        d = float(dist[ego_idx[i], pslot[i]])
        if d < best_dist[slot]:
            best_dist[slot] = d
            best_ego[slot] = int(ego_idx[i])

    return best_dist, best_ego


def audit_minimum_distance(env, label: str = "minimum_distance", tol: float = 1e-4) -> AuditReport:
    """Stage A #3: env.minimum_distance / minimum_ego_idx vs brute-force partner obs."""
    rep = AuditReport(label)
    if not env._score_tracking_enabled:
        return rep

    expected_dist, expected_ego = _bruteforce_minimum_distance(env)
    actual_dist = env.minimum_distance.astype(np.float64)
    actual_ego = env.minimum_ego_idx

    mismatches = 0
    for slot in range(actual_dist.size):
        exp_d = expected_dist[slot]
        act_d = actual_dist[slot]
        exp_e = int(expected_ego[slot])
        act_e = int(actual_ego[slot])

        if not np.isfinite(exp_d):
            if np.isfinite(act_d):
                rep.fail.append(f"slot {slot}: env tracked dist={act_d:.2f} but brute-force has no partner")
                mismatches += 1
            continue

        if not np.isfinite(act_d):
            rep.fail.append(
                f"slot {slot}: brute-force dist={exp_d:.2f} ego={exp_e} but env dist=inf"
            )
            mismatches += 1
            continue

        if abs(act_d - exp_d) > tol or act_e != exp_e:
            rep.fail.append(
                f"slot {slot}: env (d={act_d:.4f}, ego={act_e}) != "
                f"brute-force (d={exp_d:.4f}, ego={exp_e})"
            )
            mismatches += 1

    tracked = int(np.sum(np.isfinite(actual_dist)))
    rep.check(
        mismatches == 0,
        f"minimum_distance matches brute-force for {tracked} tracked slots",
        f"{mismatches} slot(s) mismatch brute-force minimum_distance",
    )
    return rep


def audit_metric_reset(
    env,
    before_dist: np.ndarray,
    before_ego: np.ndarray,
    label: str,
    expect_identity_refresh: bool = True,
) -> AuditReport:
    """Stage B: after reset/resample boundary (post _update_minimum_distance at tick 0)."""
    rep = AuditReport(label)

    rep.check(env.tick == 0, f"tick==0 after {label}", f"tick={env.tick} expected 0 after {label}")

    rep.check(
        np.all(env.score_metric == 0),
        "score_metric cleared",
        f"score_metric not zero: {int(np.sum(env.score_metric != 0))} slots",
    )
    if hasattr(env, "_episode_return"):
        rep.check(
            float(env._episode_return.sum()) == 0.0,
            "_episode_return cleared",
            f"_episode_return sum={float(env._episode_return.sum()):.3f} (expected 0)",
        )

    # reset()/resample: _reset_other_indices() then optional _update_minimum_distance() at tick 0.
    if env._score_tracking_enabled:
        rep.merge(audit_minimum_distance(env, f"{label}/distance_tick0"))

    if expect_identity_refresh:
        rep.merge(audit_slot_identity(env, f"{label}/identity"))
        rep.merge(audit_policy_assignment(env, f"{label}/policy"))

    if before_dist is not None and np.any(np.isfinite(before_dist)):
        n_before = int(np.sum(np.isfinite(before_dist)))
        n_after = int(np.sum(np.isfinite(env.minimum_distance)))
        rep.check(
            True,
            f"distance tracking refreshed: {n_before} finite before boundary, {n_after} at tick 0",
            "",
        )
    return rep


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
        corpus_idx = int(env.minimum_other_global_idx[slot])
        entity_id = int(env.minimum_other_local_idx[slot])
        other_idx = int(env.other_indices_arr[slot])
        map_id = rev.get(corpus_idx, (-1, -1))[0]
        print(
            f"{slot:6d} {other_idx:6d} {corpus_idx:8d} {map_id:6d} {entity_id:8d} "
            f"{env.minimum_ego_idx[slot]:8d} {md[slot]:12.2f}"
        )


def _step_visible_audit(env, max_rows: int = 8) -> None:
    partner_states = env.get_global_partner_state()
    other_ids = partner_states["other_id"].astype(np.int64)
    rel_xy = env._partner_obs()[:, :, :2]
    dist = np.linalg.norm(rel_xy, axis=2) / _PARTNER_REL_SCALE_F
    env_per_agent = env._env_per_agent()
    env_ids = np.broadcast_to(env_per_agent[:, np.newaxis], other_ids.shape)
    lut = env._env_entity_to_other_slot

    ego_mask = np.zeros(env.num_agents, dtype=bool)
    ego_mask[env.ego_indices] = True
    valid = (
        ego_mask[:, np.newaxis]
        & (other_ids >= 0)
        & (env_ids >= 0)
        & (env_ids < lut.shape[0])
        & (other_ids < lut.shape[1])
    )

    hits = misses = 0
    rows = []
    ego_idx, partner_slot = np.where(valid)
    for i in range(ego_idx.size):
        e, p = int(ego_idx[i]), int(partner_slot[i])
        env_i = int(env_ids[e, p])
        entity_id = int(other_ids[e, p])
        other_slot = int(lut[env_i, entity_id])
        d = float(dist[e, p])
        if other_slot < 0:
            misses += 1
            slot_kind = "not-other"
        else:
            hits += 1
            slot_kind = f"slot={other_slot}"
        if len(rows) < max_rows:
            rows.append((env_i, entity_id, other_slot, d, slot_kind))
    print(f"partner obs LUT: hit={hits} miss={misses}")
    if rows:
        print(f"{'env':>6} {'entity':>8} {'slot':>8} {'dist(m)':>10} {'kind':>12}")
        for env_i, entity_id, other_slot, d, slot_kind in rows:
            s = "-" if other_slot < 0 else str(other_slot)
            print(f"{env_i:6d} {entity_id:8d} {s:>8} {d:10.2f} {slot_kind:>12}")


def summarize_ego_returns(env, top_k: int = 8) -> None:
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
    print(f"partner_slots={int(tracked.sum())}  commit_candidates={int(tracked.sum())}")
    summarize_ego_returns(env)
    if tracked.sum() == 0:
        print("(no commit candidates — no tracked partner distance this rollout)")
        return

    print("score_metric[slot] will copy linked ego return:")
    print(
        f"{'slot':>6} {'corpus':>8} {'map':>6} {'entity':>8} {'other':>6} "
        f"{'ego':>6} {'dist':>8} {'return':>10}"
    )
    order = np.flatnonzero(tracked)
    order = order[np.argsort(md[order])][:top_k]
    rev = _reverse_entity_lookup(env)
    for slot in order:
        corpus_idx = int(env.minimum_other_global_idx[slot])
        entity_id = int(env.minimum_other_local_idx[slot])
        ego_idx = int(env.minimum_ego_idx[slot])
        other_idx = int(env.other_indices_arr[slot])
        map_id = rev.get(corpus_idx, (-1, -1))[0]
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
            corpus_idx = int(env.minimum_other_global_idx[slot])
            entity_id = int(env.minimum_other_local_idx[slot])
            map_id = rev.get(corpus_idx, (-1, -1))[0]
            print(f"{slot:6d} {corpus_idx:8d} {map_id:6d} {entity_id:8d} {score_before[slot]:10.3f}")
    else:
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


def run_stage_a(env, label: str, report: AuditReport, verbose: bool = False) -> None:
    print(f"\n{'=' * 60}\nStage A audits: {label}\n{'=' * 60}")
    id_rep = audit_slot_identity(env, f"{label}/identity")
    pol_rep = audit_policy_assignment(env, f"{label}/policy")
    dist_rep = audit_minimum_distance(env, f"{label}/distance")
    report.merge(id_rep)
    report.merge(pol_rep)
    report.merge(dist_rep)
    id_rep.print_summary()
    pol_rep.print_summary()
    dist_rep.print_summary()
    if verbose:
        _step_visible_audit(env)


def run_stage_b_reset(
    env,
    report: AuditReport,
    step: int,
    before_dist: np.ndarray,
    before_ego: np.ndarray,
    before_policy: np.ndarray | None,
    info: dict,
) -> None:
    print(f"\n{'=' * 60}\nStage B: injected reset at step {step}\n{'=' * 60}")
    report.check(
        bool(info.get("partner_resampled")),
        "info['partner_resampled'] is True after reset()",
        "info['partner_resampled'] missing or False after reset()",
    )
    report.merge(
        audit_metric_reset(
            env,
            before_dist,
            before_ego,
            f"reset@step{step}",
            expect_identity_refresh=True,
        )
    )
    if before_policy is not None and hasattr(env, "policy_per_slot_flatten"):
        same = np.array_equal(before_policy, env.policy_per_slot_flatten)
        print(
            f"  policy_per_slot_flatten changed after reset: {not same} "
            f"(sampler re-draws each reset)"
        )


def _parse_int_list(s: str) -> list[int]:
    if not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--population-path", default="/data/puffer/popul_lane_nominal")
    p.add_argument("--map-dir", default="/data/puffer/resources/drive/binaries/training")
    p.add_argument("--num-maps", type=int, default=2, help="Stage A: small map count")
    p.add_argument("--num-agents", type=int, default=64, help="Stage A: small agent count")
    p.add_argument("--ego-ratio", type=float, default=0.25)
    p.add_argument("--pbt-mode", default="reactive", choices=["reactive", "replay"])
    p.add_argument("--resample-frequency", type=int, default=10, help="Stage A: hit resample 2-3x in --steps")
    p.add_argument("--episode-length", type=int, default=91, help="Stage B: set < resample-frequency for in-rollout episode end")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--inject-reset-at",
        default="",
        help="Stage B: comma-separated step numbers to call env.reset() mid-rollout (e.g. 15)",
    )
    p.add_argument("--skip-stage-a", action="store_true", help="Only run boundary/resample logging")
    p.add_argument("--verbose-steps", action="store_true", help="Print tables + distance audit every step")
    args = p.parse_args()

    inject_reset_at = set(_parse_int_list(args.inject_reset_at))
    stage_b = args.episode_length < args.resample_frequency or bool(inject_reset_at)
    if stage_b:
        print(
            f"Stage B enabled: episode_length={args.episode_length}, "
            f"resample_frequency={args.resample_frequency}, inject_reset_at={sorted(inject_reset_at)}"
        )

    saved = os.path.join(args.population_path, "saved")
    for name in ("other_actions_agent_offsets.npy", "other_actions_map_ids.npy", "global_ids.npy"):
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
        render_mode=None,
        episode_length=args.episode_length,
        resample_frequency=args.resample_frequency,
    )

    report = AuditReport("total")

    _, reset_info = env.reset(seed=args.seed)
    reset_info = _info_dict(env, reset_info)
    print(f"resample_frequency={args.resample_frequency}  episode_length={args.episode_length}  steps={args.steps}")
    print(f"live maps={env.map_ids[:env.num_envs]}  ego_indices={env.ego_indices.tolist()}")
    print(f"other agents={env.other_indices_arr.size}  global_ids shape={env.global_ids.shape}")
    print(f"after initial reset: partner_resampled={reset_info.get('partner_resampled')}")

    if not args.skip_stage_a:
        run_stage_a(env, "after initial reset", report, verbose=True)
    summarize(env, step=0)

    rollout = 0
    for t in range(1, args.steps + 1):
        if t in inject_reset_at:
            before_dist = env.minimum_distance.copy()
            before_ego = env.minimum_ego_idx.copy()
            before_policy = (
                env.policy_per_slot_flatten.copy()
                if hasattr(env, "policy_per_slot_flatten")
                else None
            )
            _, reset_info = env.reset(seed=args.seed + t)
            run_stage_b_reset(
                env, report, t, before_dist, before_ego, before_policy, _info_dict(env, reset_info)
            )
            if not args.skip_stage_a:
                run_stage_a(env, f"after injected reset @ step {t}", report)
            summarize(env, step=t)

        if env.tick + 1 == args.resample_frequency:
            summarize_commit_preview(env, rollout)

        score_before = env.score_metric.copy()
        sampler = getattr(env, "agent_sampler", None)
        new_score_before = sampler.new_score.copy() if sampler is not None else None
        dist_before_resample = env.minimum_distance.copy()
        ego_before_resample = env.minimum_ego_idx.copy()
        policy_before_resample = (
            env.policy_per_slot_flatten.copy() if hasattr(env, "policy_per_slot_flatten") else None
        )

        _, _, _, _, step_info = env.step(np.zeros((env.num_agents, 1), dtype=np.int32))
        step_info = _info_dict(env, step_info)

        if env.tick == 0 and t > 0:
            print(f"\n{'=' * 60}\nStage B: resample boundary at step {t}\n{'=' * 60}")
            print(f"  partner_resampled={step_info.get('partner_resampled')}")
            report.check(
                bool(step_info.get("partner_resampled")),
                f"resample@step{t}: partner_resampled True in step info",
                f"resample@step{t}: partner_resampled missing/False",
            )
            report.merge(
                audit_metric_reset(
                    env,
                    dist_before_resample,
                    ego_before_resample,
                    f"resample@step{t}",
                    expect_identity_refresh=True,
                )
            )
            summarize_resample_commit(env, rollout, score_before, new_score_before)
            if not args.skip_stage_a:
                run_stage_a(env, f"after resample @ step {t}", report)
            if policy_before_resample is not None:
                changed = not np.array_equal(policy_before_resample, env.policy_per_slot_flatten)
                print(f"  policy_per_slot_flatten changed after resample: {changed}")
            rollout += 1

        if args.verbose_steps or env.tick == 0 or t <= 2:
            summarize(env, step=t)
            if not args.skip_stage_a and (args.verbose_steps or env.tick == 0):
                dist_rep = audit_minimum_distance(env, f"step {t}")
                if dist_rep.fail:
                    dist_rep.print_summary()
                    report.merge(dist_rep)

    report.print_summary()
    print(
        "\nNotes:"
        "\n  Stage A: audit_slot_identity + audit_minimum_distance (brute-force)"
        "\n  Stage B: inject-reset-at and/or episode_length < resample_frequency"
        "\n  minimum_other_local_idx = map entity id; minimum_other_global_idx = corpus g"
        "\n  score_metric[slot] = _episode_return[minimum_ego_idx] at resample"
    )
    return 1 if report.fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
