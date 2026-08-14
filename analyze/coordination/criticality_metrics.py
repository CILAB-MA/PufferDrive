#!/usr/bin/env python3
"""Criticality metrics for automated driving, applied to PufferDrive coordination packs.

Every metric here is grounded in:

    Westhofen, Neurohr, Koopmann, Butz, Schuett, Utesch, Kramer, Gutenkunst, Boede (2022),
    "Criticality Metrics for Automated Driving: A Review and Suitability Analysis of the
    State of the Art", Archives of Computational Methods in Engineering.
    https://doi.org/10.1007/s11831-022-09788-7

which reviews 43 criticality metrics across 8 categories (Time/Distance/Velocity/
Acceleration/Jerk/Index/Probability/Potential-scale). The original paper each metric is
attributed to by that survey is cited in the relevant function's docstring.

Data budget. Packs written by rollout_per_ego() (see rollout.py) carry, per ego and per
0.1s step (DT below): distance to the nearest interacting agent (min_dist_traj), closing
speed along the line of sight (closing_traj), TTC as computed by the simulator itself
(ttc_traj), ego speed (speed_traj) -- and, only when capture_readout=True, i.e. packs
written by run_ego_readout.sh but NOT run_coordination.sh -- the policy's action-
distribution readouts (exp_accel_traj, p_brake_traj, ...). There is no lane/road
geometry, no conflict-area definition, no vehicle footprint/mass, no per-step position or
heading, and no *other* agent's full state beyond what nearest_from_states() already
reduced it to (distance / closing speed / the sim's own TTC).

That budget makes roughly half of the survey's 43 metrics inapplicable outright -- they
need a conflict area, a reaction-time driver model, vehicle footprints, or an externally
calibrated probability distribution the source paper itself says needs dedicated data.
See NOT_APPLICABLE at the bottom of this file for the full list with per-metric reasons.
Of the rest, some are computed exactly as the survey defines them (EXACT_METRICS) and
some require a stand-in assumption the paper leaves open (e.g. an unmeasured vehicle
mass, reaction time, or "maximum available deceleration" constant) -- those are computed
too, but flagged in APPROXIMATE_METRICS with the specific assumption made.

Usage:
    from criticality_metrics import compute_all_metrics, summarize_metrics
    metrics = compute_all_metrics(pack)              # dict[str, np.ndarray], one row/ego
    means = summarize_metrics(metrics)                # dict[str, float]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from common import ACCEL_VALUES_NP, DEFAULT_THRESHOLDS

DT = 0.1  # seconds/step, matches rollout.py
EPS = 1e-6

# Maximum available longitudinal acceleration magnitude (m/s^2), taken directly from the
# policy's own discrete action bins (ACCEL_VALUES_NP in common.py) rather than an assumed
# literature constant -- this is the actual physical limit the ego can exert in PufferDrive.
A_MAX = float(np.max(np.abs(ACCEL_VALUES_NP)))


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float = EPS) -> np.ndarray:
    den = np.where(np.abs(den) < eps, np.nan, den)
    return num / den


def has_trajectories(pack: dict[str, np.ndarray]) -> bool:
    """True for packs from run_ego_readout.sh (capture_readout=True); False for the
    scalar-only packs written by run_coordination.sh."""
    return all(k in pack for k in ("min_dist_traj", "ttc_traj", "closing_traj", "speed_traj"))


def has_readout(pack: dict[str, np.ndarray]) -> bool:
    return "exp_accel_traj" in pack


# ============================================================================
# 1. TIME-SCALE METRICS
# ============================================================================


def ttc_scalar(pack: dict[str, np.ndarray]) -> np.ndarray:
    """TTC -- Time To Collision [survey formula: min{t | d(p1(t+t),p2(t+t))=0}].

    Reused as-is: the simulator's own nearest_from_states() (rollout.py) already computes
    exactly this quantity per step (distance / closing speed while closing), so ep_min_ttc
    (worst TTC realized over the episode) is an exact instance of the metric, not an
    approximation. Falls back to nanmin(ttc_traj) if only the trajectory is present.
    """
    if "ep_min_ttc" in pack:
        return pack["ep_min_ttc"]
    return np.nanmin(pack["ttc_traj"], axis=1)


def dce_scalar(pack: dict[str, np.ndarray]) -> np.ndarray:
    """DCE -- Distance of Closest Encounter [Eggert2014].

    DCE(A1,A2,t) = min_{t~>=0} d(p1(t+t~), p2(t+t~)). Exact: equals ep_min_dist, the
    minimum inter-agent distance realized over the episode.
    """
    if "ep_min_dist" in pack:
        return pack["ep_min_dist"]
    return np.nanmin(pack["min_dist_traj"], axis=1)


def closest_encounter(pack: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """DCE and TTCE -- Time To Closest Encounter [Eggert2014].

    TTCE(A1,A2,t) = argmin_{t~>=0} d(p1(t+t~),p2(t+t~)); as DCE->0, TTCE->TTC. Requires
    the full min_dist_traj (not just its scalar minimum) to recover the *time* of closest
    approach, so only available on run_ego_readout.sh packs.
    """
    d = pack["min_dist_traj"]
    n = d.shape[0]
    dce = np.full(n, np.nan, dtype=np.float64)
    ttce = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        row = d[i]
        if np.isfinite(row).any():
            j = int(np.nanargmin(row))
            dce[i] = float(row[j])
            ttce[i] = float(j) * DT
    return dce, ttce


def headway_traj(pack: dict[str, np.ndarray]) -> np.ndarray:
    """HW -- Headway [Jansson2005]. HW(A1,A2,t) = d(p1(t),p2(t)); the instantaneous spatial
    gap. Exact: this is literally min_dist_traj, formalized under its survey name."""
    return pack["min_dist_traj"]


def time_headway_traj(pack: dict[str, np.ndarray]) -> np.ndarray:
    """THW -- Time Headway [Jansson2005]. THW(A1,A2,t) = min{t~>=0 | p1(t+t~)=p2(t)}: time
    for A1 to reach A2's current (fixed) position at A1's current speed. Exact given the
    stored state: d(t) / max(speed(t), eps)."""
    return _safe_div(pack["min_dist_traj"], pack["speed_traj"])


def time_exposed(traj: np.ndarray, tau: float, *, dt: float = DT) -> np.ndarray:
    """TET -- Time Exposed <underlying metric> [Minderhoud2001].

    TET(A1,A2,tau) = integral_[t0,te] 1{metric(t) <= tau} dt. The survey notes this
    "aggregation below a target value" construction is independent of TTC and adapts to
    any underlying metric -- used below for both TTC and, via time_integrated, a_long_req.
    """
    below = np.where(np.isfinite(traj), traj <= tau, False)
    return dt * below.sum(axis=1)


def time_integrated(traj: np.ndarray, tau: float, *, dt: float = DT) -> np.ndarray:
    """TIT -- Time Integrated <underlying metric> [Minderhoud2001].

    TIT(A1,A2,tau) = integral_[t0,te] 1{metric(t)<=tau} * (tau - metric(t)) dt. Retains the
    *margin* below tau at each step rather than TET's binary indicator, which the survey
    states "reflects criticality more accurately than TET".
    """
    margin = np.where(np.isfinite(traj) & (traj <= tau), tau - traj, 0.0)
    return dt * margin.sum(axis=1)


def worst_case_ttc_traj(pack: dict[str, np.ndarray], *, extra_accel: float = 2 * A_MAX) -> np.ndarray:
    """WTTC -- Worst Time To Collision [Wachenfeld2016]. APPROXIMATE (see
    APPROXIMATE_METRICS['WTTC']).

    The survey's WTTC takes the min TTC over *all* trajectories reachable by both actors'
    DMMs (a true reachable-set computation). Lacking that, we substitute a single
    worst-case scenario: the current closing speed instantaneously ramps up at a fixed
    extra acceleration (default: twice the env's own max accel/decel bound, i.e. both
    agents adversarially closing as hard as physically possible), and solve the resulting
    quadratic gap(t) = d - closing*t - 0.5*extra_accel*t^2 = 0 for its smallest positive
    root.
    """
    d = pack["min_dist_traj"]
    cl = np.maximum(pack["closing_traj"], 0.0)
    disc = cl**2 + 2.0 * extra_accel * d
    with np.errstate(invalid="ignore"):
        t = (-cl + np.sqrt(np.maximum(disc, 0.0))) / extra_accel
    valid = np.isfinite(pack["min_dist_traj"]) & np.isfinite(pack["closing_traj"])
    return np.where(valid, t, np.nan)


def time_to_brake_traj(pack: dict[str, np.ndarray], *, a_max: float = A_MAX) -> np.ndarray:
    """TTB -- Time To Brake, the maneuver='brake' special case of TTM [Mages2009;
    Hillenbrand2006; Tamke2011]. APPROXIMATE (see APPROXIMATE_METRICS['TTB']).

    TTM(A1,A2,t,m) is "the latest time in [0,TTC] such that performing maneuver m from
    then on still avoids collision". Using a constant-max-deceleration braking model as m
    (the env's own a_max bound), the time needed to bring the closing speed to zero is
    closing/a_max, so TTB = TTC - closing/a_max.
    """
    ttc = pack["ttc_traj"]
    cl = pack["closing_traj"]
    brake_time = np.where(cl > 0, cl / a_max, np.nan)
    return ttc - brake_time


# ============================================================================
# 2. DISTANCE-SCALE METRICS -- HW, DCE handled above (Time-Scale section, per the
#    survey's own cross-references: "HW: refer to THW", "DCE: refer to TTCE").
#    AGS and PSD: NOT_APPLICABLE (need gap-acceptance / conflict-area models).
# ============================================================================


# ============================================================================
# 3. VELOCITY-SCALE METRICS
# ============================================================================


def delta_v_proxy(pack: dict[str, np.ndarray]) -> np.ndarray:
    """Delta-v -- [Gabauer2006; Carlson1979]. APPROXIMATE (see
    APPROXIMATE_METRICS['Delta-v']).

    Two-actor predictive formula: Delta-v(A1,A2,t) = m2/(m1+m2) * ||v2(t)-v1(t)||. With no
    mass data, we assume equal mass (weight 0.5) and evaluate the closing speed at the
    timestep of closest approach (TTCE) as a stand-in for ||v2-v1|| at the moment that
    matters most -- not a physically measured post-collision speed change.
    """
    d = pack["min_dist_traj"]
    cl = pack["closing_traj"]
    n = d.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        row = d[i]
        if not np.isfinite(row).any():
            continue
        j = int(np.nanargmin(row))
        c = cl[i, j]
        if np.isfinite(c):
            out[i] = 0.5 * abs(float(c))
    return out


# CS (Conflict Severity): NOT_APPLICABLE -- explicitly not run-time capable per the survey
# itself (needs a-posteriori "evasive maneuver" identification plus both agents' masses).


# ============================================================================
# 4. ACCELERATION-SCALE METRICS
# ============================================================================


def required_long_decel_traj(pack: dict[str, np.ndarray]) -> np.ndarray:
    """a_long,req -- Required Longitudinal Acceleration, a.k.a. DRAC (Deceleration Rate To
    Avoid Crash) [Jansson2005; alias Archer2005].

    Constant-acceleration special case (assuming the lead/other agent holds its current
    acceleration, i.e. a2=0): a_long,req = min(a2 + closing^2/(2d), 0). Exact given the
    stored state (closing_traj IS v1_long - v2_long by construction of
    nearest_from_states()). Returned as a magnitude with sign convention <= 0
    (0 = no braking needed).
    """
    d = pack["min_dist_traj"]
    cl = pack["closing_traj"]
    with np.errstate(divide="ignore", invalid="ignore"):
        a_req = np.where(cl > 0, -(cl**2) / (2.0 * np.maximum(d, EPS)), 0.0)
    valid = np.isfinite(d) & np.isfinite(cl)
    return np.where(valid, a_req, np.nan)


def brake_threat_number_traj(pack: dict[str, np.ndarray], *, a_min_brake: float = -A_MAX) -> np.ndarray:
    """BTN -- Brake Threat Number [Jansson2005; multi-actor extension Eidehall2011].

    BTN = a_long,req / a1,long,min. By definition BTN >= 1 means a braking maneuver cannot
    avoid the collision under the assumed model. a_min_brake defaults to the environment's
    own maximum-braking bound (-A_MAX), which is exact for this policy's action space
    rather than an assumed literature constant.
    """
    return required_long_decel_traj(pack) / a_min_brake


def deceleration_to_safety_time_traj(pack: dict[str, np.ndarray], *, ts: float = 0.0) -> np.ndarray:
    """DST -- Deceleration to Safety Time [Hupfer1997; Schubert2010]. APPROXIMATE for
    ts > 0 (see APPROXIMATE_METRICS['DST']); EXACT at ts=0.

    DST(A1,A2,t,ts) = (v1_long-v2_long)^2 / (2*(d - v2_long*ts)). At ts=0 this is exactly
    the constant-acceleration a_long,req (the survey states the two agree). For ts>0 we
    need v2 in isolation, which isn't stored -- approximated as
    max(ego_speed - closing_speed, 0), a car-following-geometry assumption.
    """
    d = pack["min_dist_traj"]
    cl = pack["closing_traj"]
    v1 = pack["speed_traj"]
    v2 = np.clip(v1 - cl, 0.0, None)
    denom = d - v2 * ts
    with np.errstate(divide="ignore", invalid="ignore"):
        dst = (cl**2) / (2.0 * denom)
    return np.where(np.abs(denom) < EPS, np.nan, dst)


# a_lat,req, STN, a_req (combined norm): NOT_APPLICABLE -- see dict below (no steer/lateral
# signal persisted in the packs).


# ============================================================================
# 5. JERK-SCALE METRICS
# ============================================================================


def longitudinal_jerk_proxy(pack: dict[str, np.ndarray]) -> np.ndarray:
    """LongJ -- Longitudinal Jerk [general concept; curve-safety use in Ambros2019].
    APPROXIMATE (see APPROXIMATE_METRICS['LongJ']).

    LongJ(A1,t) = j1,long(t) = d(a1,long)/dt. The actually-*executed* per-step acceleration
    isn't persisted in the saved packs (only its episode-level hard-braking fraction is);
    what IS available is exp_accel_traj, the policy's softmax-expected acceleration. We
    finite-difference that as a proxy for the policy's decision smoothness, not the
    vehicle's physically realized jerk.
    """
    a = pack["exp_accel_traj"]
    jerk = np.diff(a, axis=1) / DT
    pad = np.full((jerk.shape[0], 1), np.nan, dtype=jerk.dtype)
    return np.concatenate([pad, jerk], axis=1)


# LatJ: NOT_APPLICABLE -- same missing-steer-signal reason as a_lat,req.


# ============================================================================
# 6. INDEX-SCALE METRICS
# ============================================================================


def accident_metric(pack: dict[str, np.ndarray]) -> np.ndarray:
    """AM -- Accident Metric [general/implicit, e.g. GIDAS database usage].

    AM(Sc) = 0 if no accident happened, 1 otherwise. Exact: this is exactly the pack's own
    `collided` flag, formalized under its survey name.
    """
    return pack["collided"].astype(np.float64)


def crash_potential_index_deterministic(pack: dict[str, np.ndarray], *, a_min_brake: float = -A_MAX) -> np.ndarray:
    """CPI -- Crash Potential Index [Cunto2007; Cunto2008]. APPROXIMATE (see
    APPROXIMATE_METRICS['CPI_deterministic']).

    CPI(A1,A2) = (1/(te-t0)) * integral P(a_long,req(t) < a_long,min(t)) dt, where the
    literature fits a_long,min to a *normal distribution* from empirical braking-capability
    studies. We have no such calibration data, so we use the environment's own fixed
    a_max bound as a deterministic threshold instead: this collapses the smooth
    probability into the fraction of the episode where a_long,req already exceeds the
    hard physical limit (a stricter, deterministic reading of the same idea).
    """
    a_req = required_long_decel_traj(pack)
    valid = np.isfinite(a_req)
    exceed = np.where(valid, a_req <= a_min_brake, False)
    denom = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cpi = exceed.sum(axis=1) / np.where(denom > 0, denom, np.nan)
    return cpi


def rss_longitudinal_min_distance_traj(
    pack: dict[str, np.ndarray],
    *,
    rho: float = 1.0,
    a_max_accel: float = A_MAX,
    a_min_brake: float = A_MAX,
    a_max_brake: float = A_MAX,
) -> np.ndarray:
    """Longitudinal half of RSS-DS's d_min [Shalev-Shwartz2017]. APPROXIMATE (see
    APPROXIMATE_METRICS['RSS_long_violation']).

    d_min = v_r*rho + 0.5*a_max_accel*rho^2 + (v_r+rho*a_max_accel)^2/(2*a_min_brake)
            - v_f^2/(2*a_max_brake)
    with v_r = ego (rear/following) speed, v_f = lead speed (approximated as
    ego_speed - closing_speed, same assumption as DST/CPI's v2 above). rho (reaction time)
    and the accel/brake bounds are assumed constants, since neither is measured.
    """
    v1 = pack["speed_traj"]
    cl = pack["closing_traj"]
    v2 = np.clip(v1 - cl, 0.0, None)
    d_min = (
        v1 * rho
        + 0.5 * a_max_accel * rho**2
        + (v1 + rho * a_max_accel) ** 2 / (2.0 * a_min_brake)
        - v2**2 / (2.0 * a_max_brake)
    )
    return np.maximum(d_min, 0.0)


def rss_longitudinal_violation_traj(pack: dict[str, np.ndarray], **kwargs: Any) -> np.ndarray:
    """RSS_long_violation -- longitudinal-only proxy for RSS-DS [Shalev-Shwartz2017].
    APPROXIMATE / PARTIAL (see APPROXIMATE_METRICS['RSS_long_violation']).

    True RSS-DS = 1 iff BOTH the lateral and longitudinal safe distances are violated
    simultaneously. We can only evaluate the longitudinal half (no lane geometry for the
    lateral half exists), so this necessarily over-flags relative to the true metric.
    """
    d_min = rss_longitudinal_min_distance_traj(pack, **kwargs)
    return pack["min_dist_traj"] < d_min


# ACI, CI, PRI, SOI, TCI, PSD, AGS, STN: NOT_APPLICABLE -- see dict below.


# ============================================================================
# 7. PROBABILITY-SCALE METRICS
# ============================================================================


def monte_carlo_collision_probability(packs_by_seed: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    """P-MC -- Collision Probability via Monte Carlo [Broadhurst2005]. APPROXIMATE
    reinterpretation: exact as an *empirical* Monte-Carlo estimate, not the paper's
    control-input-integral formulation.

    The survey's P-MC integrates a collision probability over a prior distribution of
    control-input hypotheses for every actor. We don't have that generative model, but the
    coordination pipeline already runs the SAME policy checkpoint family across multiple
    training seeds on the SAME map sequence (see run_coordination.sh's MAX_SEEDS) -- so the
    empirical fraction of seed replicates that collide on a given map IS a direct
    Monte-Carlo estimate of collision probability for that scene, just drawing its
    "hypotheses" from seed variance rather than a control-input prior.

    Args:
        packs_by_seed: per-seed pack dicts for the SAME method (record/reactive/selfplay),
            each pack must contain 'scene_id' and 'collided' (any pack schema works, since
            this only touches scalar fields present in both run_coordination.sh and
            run_ego_readout.sh packs).
    """
    scene_to_colls: dict[int, list[bool]] = {}
    for pack in packs_by_seed:
        for sid, coll in zip(pack["scene_id"].tolist(), pack["collided"].tolist()):
            scene_to_colls.setdefault(int(sid), []).append(bool(coll))
    per_scene = {sid: float(np.mean(vals)) for sid, vals in scene_to_colls.items()}
    overall = float(np.mean(list(per_scene.values()))) if per_scene else float("nan")
    return {
        "per_scene": per_scene,
        "n_scenes": len(per_scene),
        "overall_mean": overall,
        "n_seeds": len(packs_by_seed),
    }


# P-SMH, P-SRS: NOT_APPLICABLE -- see dict below.


# ============================================================================
# 8. POTENTIAL-SCALE METRICS -- PF, SP: NOT_APPLICABLE (see dict below; both need
#    per-object-type potential functions built on lane/road geometry or vehicle
#    footprints, neither of which the packs carry).
# ============================================================================


# ============================================================================
# Aggregate entry points
# ============================================================================


def compute_all_metrics(
    pack: dict[str, np.ndarray],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute every applicable metric for one method's pack. Returns one array per metric,
    each of length n_ego (one row per ego / map). Trajectory-dependent metrics are skipped
    (not filled with NaN columns) when the pack lacks trajectories, e.g. packs from
    run_coordination.sh -- check `has_trajectories(pack)` / `has_readout(pack)` beforehand
    if you need to know which tier ran.
    """
    thr = thresholds or DEFAULT_THRESHOLDS
    out: dict[str, np.ndarray] = {
        "AM_accident_metric": accident_metric(pack),
        "DCE_distance_closest_encounter_m": dce_scalar(pack),
        "TTC_min_s": ttc_scalar(pack),
    }

    if has_trajectories(pack):
        _dce_t, ttce_t = closest_encounter(pack)
        out["TTCE_time_to_closest_encounter_s"] = ttce_t
        out["HW_mean_m"] = np.nanmean(headway_traj(pack), axis=1)
        out["THW_mean_s"] = np.nanmean(time_headway_traj(pack), axis=1)

        ttc = pack["ttc_traj"]
        out["TET_tight_s"] = time_exposed(ttc, thr["ttc_tight"])
        out["TET_approach_s"] = time_exposed(ttc, thr["ttc_approach"])
        out["TIT_tight_s2"] = time_integrated(ttc, thr["ttc_tight"])
        out["TIT_approach_s2"] = time_integrated(ttc, thr["ttc_approach"])

        a_req = required_long_decel_traj(pack)
        out["a_long_req_worst_mps2"] = np.nanmin(a_req, axis=1)
        btn = brake_threat_number_traj(pack)
        out["BTN_max"] = np.nanmax(btn, axis=1)
        out["BTN_frac_unavoidable"] = np.nanmean(
            np.where(np.isfinite(btn), (btn >= 1.0).astype(np.float64), np.nan), axis=1
        )

        out["DST_ts0_worst_mps2"] = np.nanmax(deceleration_to_safety_time_traj(pack, ts=0.0), axis=1)
        out["TTB_min_s"] = np.nanmin(time_to_brake_traj(pack), axis=1)
        out["WTTC_min_s_approx"] = np.nanmin(worst_case_ttc_traj(pack), axis=1)
        out["CPI_deterministic_approx"] = crash_potential_index_deterministic(pack)
        out["DeltaV_proxy_mps_approx"] = delta_v_proxy(pack)
        out["RSS_long_violation_frac_approx"] = np.mean(rss_longitudinal_violation_traj(pack), axis=1)

    if has_readout(pack):
        jerk = longitudinal_jerk_proxy(pack)
        out["LongJ_policy_proxy_mean_abs_mps3"] = np.nanmean(np.abs(jerk), axis=1)

    return out


def summarize_metrics(
    metrics: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Collapse per-ego metric arrays into scalar means (NaN-safe), optionally restricted
    to a boolean subset mask (e.g. the pipeline's own 'had_tight' / 'hard_tail' masks)."""
    out: dict[str, float] = {}
    for key, arr in metrics.items():
        vals = arr[mask] if mask is not None else arr
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[key] = float(vals.mean()) if vals.size else float("nan")
    return out


# ============================================================================
# Applicability registry -- what's implemented exactly, what's approximated and how, and
# what's excluded and why. Printed by criticality_report.py for transparency.
# ============================================================================

EXACT_METRICS: dict[str, str] = {
    "AM": "Accident Metric == the pack's own `collided` flag.",
    "DCE": "Distance of Closest Encounter == ep_min_dist / nanmin(min_dist_traj).",
    "TTC": "Time To Collision == the simulator's own ttc_traj / ep_min_ttc (already computed by nearest_from_states in rollout.py).",
    "TTCE": "Time To Closest Encounter == argmin(min_dist_traj) * dt.",
    "HW": "Headway == min_dist_traj itself.",
    "THW": "Time Headway == min_dist_traj / speed_traj.",
    "TET": "Time Exposed TTC -- generic threshold-crossing integral over ttc_traj.",
    "TIT": "Time Integrated TTC -- generic margin integral over ttc_traj.",
    "a_long,req (DRAC)": "closing_traj is v1_long - v2_long by construction, so the constant-acceleration special case is exact.",
    "BTN": "= a_long,req / a_min_brake, with a_min_brake the env's own exact action-space bound.",
    "DST (ts=0)": "agrees with a_long,req at ts=0, per the survey's own note.",
}

APPROXIMATE_METRICS: dict[str, str] = {
    "DST (ts>0)": "other agent's longitudinal speed v2 is approximated as ego_speed - closing_speed (assumes the closing-speed axis is approximately the longitudinal/car-following axis).",
    "TTB": "uses a fixed maximum-braking maneuver model (a_max = env's own action-space bound) as TTM's required 'brake' maneuver model, rather than a calibrated driver/vehicle braking model.",
    "WTTC": "substitutes a single worst-case constant extra closing-acceleration (2x the env's max accel bound) for the paper's true reachable-trajectory-set computation.",
    "CPI_deterministic": "the paper's CPI integrates P(a_req < a_min) under an externally-fitted normal distribution of a_min from empirical braking-capability studies; we don't have that calibration data, so we use the env's fixed action bound as a deterministic threshold, collapsing the metric to an indicator average rather than a smooth probability.",
    "Delta-v": "assumes equal vehicle mass (no mass data available) and uses the closing speed at the timestep of minimum distance as a stand-in for pre-impact relative velocity, not a physically measured post-collision speed change.",
    "RSS_long_violation": "implements only the longitudinal half of RSS-DS's simultaneous lateral+longitudinal violation test (no lane geometry for the lateral half); uses assumed constants (reaction time rho=1.0s, accel/brake bounds = env's action-space bound). Over-flags relative to the true RSS-DS.",
    "LongJ": "computed on the policy's *expected* acceleration (softmax-weighted mean over the discrete action distribution) since the actually-sampled/executed per-step acceleration isn't persisted in the packs -- a policy-smoothness proxy, not physically realized vehicle jerk.",
    "P-MC": "reinterpreted as an empirical Monte-Carlo estimate: the fraction of training-seed replicates (same method, same map) that collide, rather than the paper's control-input-prior integral -- both estimate the same quantity (a collision probability over stochastic realizations), just from different sources of stochasticity.",
}

NOT_APPLICABLE: dict[str, str] = {
    "ET": "needs a defined conflict area (CA) -- a lane/road-geometry construct not present in the saved packs.",
    "PET": "same CA requirement as ET, plus needs the *other* agent's own entry/exit times for that CA, i.e. its full trajectory (only its distance/closing state is stored).",
    "PrET / SPrET / TA": "needs a per-actor Dynamic Motion Model predicting future trajectories and an explicit predicted-path intersection point; only the current reduced state (distance, closing speed, TTC) is available, not full predicted paths for both actors.",
    "TTZ": "pedestrian-crossing-specific; needs a crosswalk/zebra position and the pipeline carries no VRU/crosswalk semantics.",
    "PSD": "requires a conflict-area distance p_CA(t); substituting the nearest agent's position would deviate from Allen et al.'s own definition (relative to an intersection point, not another vehicle), so this was intentionally not approximated.",
    "AGS": "the survey itself states the formulation 'has not yet been generalized' beyond an empirically-fit gap-acceptance model needing driver-demographic/road-condition covariates not in our state at all.",
    "CI (Conflict Index)": "built on top of PET (a-posteriori CA-crossing time) plus vehicle masses and approach headings at CA entry/exit -- none available.",
    "PRI": "pedestrian-specific; needs a driver reaction-time model and a crosswalk conflict window, neither available.",
    "TCI": "needs lane-relative reference positions (a following-distance reference r_x, a lateral-clearance reference r_y) derived from road geometry -- none in the packs.",
    "ACI": "requires an externally supplied probabilistic causal 'collision tree' with per-branch conditional probabilities calibrated from a dedicated dataset -- not derivable from rollout packs alone.",
    "SOI": "requires a per-actor-type 'personal space' footprint definition and considers ALL actors' overlaps (retrospective multi-agent); packs only track the single nearest interacting agent per ego.",
    "PF (Potential Functions)": "requires a hand-designed potential function per object type (lane markings, road geometry, other agents, VRUs) -- no lane/road-geometry channel to build one from.",
    "SP (Safety Potential / SFF)": "requires a defined set of safe fallback control policies and their resulting occupied/claimed sets (effectively vehicle footprints over time) -- footprints aren't tracked.",
    "P-SMH": "needs a finite set of scored hypothesis trajectories for BOTH the ego (two-track model) and the other agent (one-track model); the other agent's action distribution was never captured (only the ego policy's logits are read out).",
    "P-SRS": "needs precomputed offline Markov-chain reachability abstractions over a discretized state/input space for the other agent -- no such precomputation exists in this pipeline.",
    "a_lat,req / STN / LatJ": "all three need a lateral-acceleration or steering signal per timestep. common.py's action_stats_from_logits() DOES compute 'steer'/'steer_mag' from the policy logits, but rollout.py's readout_out capture map only pulls accel/p_brake/p_yield/gap_press/entropy into the saved pack -- steer was never persisted. Adding a 'steer': 'steer' entry to that dict (and re-running run_ego_readout.sh) would unlock these; not done here since it requires re-collecting rollout data.",
    "a_req (combined norm)": "= sqrt(a_long_req^2 + a_lat_req^2); depends on a_lat,req (not applicable, see above). Reporting only the longitudinal term would silently misrepresent the combined metric, so it is omitted rather than degraded.",
    "PTTC": "assumes actor A1 moves at constant velocity while only A2 decelerates, needing A2's acceleration in isolation; the pipeline only ever measures the *relative* closing acceleration, with no way to attribute it to one side. a_long,req/DST already cover the same 'required deceleration' family without that attribution ambiguity.",
    "CS (Conflict Severity)": "explicitly NOT run-time capable per the survey itself -- requires a-posteriori identification of which timestep was an 'evasive maneuver', plus both agents' masses.",
    "TTS / TTK / TTR / generic TTM": "TTB (TTM's maneuver='brake' case) is implemented using the env's own max-deceleration bound as the maneuver model. TTS (maneuver='steer') needs the same lateral signal missing for a_lat,req. TTK (maneuver='kickdown') is directional (only meaningful when approached from behind, which the packs don't distinguish). TTR = max over {brake,steer,kickdown}; with only TTB reliably defined it would just collapse to TTB, so it's not reported separately.",
    "RSS-DS (full)": "the true metric requires a SIMULTANEOUS lateral AND longitudinal safe-distance violation; only the longitudinal half is implemented (see APPROXIMATE_METRICS['RSS_long_violation']) since no lane/lateral data exists.",
}