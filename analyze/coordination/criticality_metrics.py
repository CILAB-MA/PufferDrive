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
# Used as a_min_brake/a_max wherever a metric asks for THIS vehicle's own capability.
A_MAX = float(np.max(np.abs(ACCEL_VALUES_NP)))

# RSS-DS's OTHER TWO acceleration parameters (Shalev-Shwartz, Shammah, Shashua 2017,
# "On a Formal Model of Safe and Scalable Self-Driving Cars", arXiv:1708.06374). The RSS
# paper itself declines to publish numeric values ("parameters should be determined... by
# regulation"); these are the field-standard values from Intel's reference implementation
# (github.com/intel/ad-rss-lib, "Parameter Discussion" appendix), which explicitly cites
# them as the "German driving school rule of thumb" and are the values actually shipped in
# that RSS reference implementation. Do NOT reuse A_MAX for these -- RSS models a generic
# road user's capability, not this policy's own action-space bound.
A_RSS_MAX_ACCEL = 3.5  # a_max,accel: max comfortable acceleration a car might apply
A_RSS_MAX_BRAKE_OTHER = 8.0  # a_max,brake: max braking the OTHER car might apply


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float = EPS) -> np.ndarray:
    den = np.where(np.abs(den) < eps, np.nan, den)
    return num / den


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via a numpy-vectorized erf approximation (Abramowitz & Stegun
    7.1.26, max abs error ~1.5e-7) -- avoids adding a scipy dependency solely for CPI's
    normal-CDF term."""
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x) / np.sqrt(2.0)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return 0.5 * (1.0 + sign * y)


def has_trajectories(pack: dict[str, np.ndarray]) -> bool:
    """True for packs from run_ego_readout.sh (capture_readout=True); False for the
    scalar-only packs written by run_coordination.sh."""
    return all(k in pack for k in ("min_dist_traj", "ttc_traj", "closing_traj", "speed_traj"))


def has_readout(pack: dict[str, np.ndarray]) -> bool:
    return "exp_accel_traj" in pack


def has_width(pack: dict[str, np.ndarray]) -> bool:
    """True when both agents' vehicle widths are captured (ego_width scalar,
    other_width_traj per-step) -- needed for a_lat,req/STN/a_req (those need width only,
    not length). Verified against a real compiled rollout: other_width_traj ~2.0m,
    realistic."""
    return "ego_width" in pack and "other_width_traj" in pack


def has_dimensions(pack: dict[str, np.ndarray]) -> bool:
    """True when BOTH length and width are captured for both agents -- needed for
    encroachment_times_traj (ET/PET) and time_to_steer_traj (TTS), which build full
    oriented rectangles/discs from vehicle geometry, not just a width-only lateral term
    the way a_lat,req does. Found via code review (not a crash observed in practice,
    since rollout.py's capture_pose always sets ego_length alongside ego_width in the
    same block, and other_length_traj alongside other_width_traj) that the ET/PET call
    site was gated only on "other_length_traj" in pack, silently assuming width would
    also be present rather than checking -- this helper makes that assumption explicit
    and centralizes it so any future change to what rollout.py captures together can't
    silently reintroduce a KeyError."""
    return "ego_length" in pack and has_width(pack) and "other_length_traj" in pack


def has_pose(pack: dict[str, np.ndarray]) -> bool:
    """True for packs from rollout_per_ego(..., capture_pose=True): ego and nearest-
    partner (x, y, heading, speed) trajectories, ego vehicle dimensions, and the
    nearest-partner's raw identity per step (other_id_traj). Verified end-to-end against
    a real compiled rollout (not just synthetic data) -- see git history for
    rollout.py/common.py's capture_pose additions."""
    return all(
        k in pack
        for k in (
            "ego_x_traj",
            "ego_y_traj",
            "ego_heading_traj",
            "other_x_traj",
            "other_y_traj",
            "other_heading_traj",
            "other_speed_traj",
            "other_id_traj",
        )
    )


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
    """TTB -- Time To Brake, the maneuver='brake' special case of TTM [Hillenbrand 2007
    KIT dissertation, "Fahrerassistenz zur Kollisionsvermeidung", Sec. 5.2.3, eq. 5.4-5.6;
    journal version Hillenbrand, Spieker, Kroschel 2006]. EXACT (see
    EXACT_METRICS['TTB']).

    TTM(A1,A2,t,m) is "the latest time in [0,TTC] such that performing maneuver m from
    then on still avoids collision". The dissertation confirms the 1D/no-lateral-
    intersection case (our setup) is pure longitudinal kinematics: the ego rides out its
    current motion until switching to full braking at a_min (an EGO-OWN property --
    explicitly "the magnitude-maximal braking deceleration the ego vehicle can produce",
    i.e. exactly this env's own a_max=4.0 m/s^2 bound, not an assumed external constant).
    Time needed to bring the closing speed to zero at that rate is closing/a_max, so
    TTB = TTC - closing/a_max.
    """
    ttc = pack["ttc_traj"]
    cl = pack["closing_traj"]
    brake_time = np.where(cl > 0, cl / a_max, np.nan)
    return ttc - brake_time


def time_to_kickdown_traj(pack: dict[str, np.ndarray], *, a_max: float = A_MAX) -> np.ndarray:
    """TTK -- Time To Kickdown, the maneuver='kickdown' (full acceleration) special case
    of TTM [Hillenbrand 2007 dissertation, Sec. 5.2.5 -- presented as TTB's mirror image:
    "since this is a one-dimensional problem", ego switches to +a_max instead of -a_max].
    EXACT when pose is available, APPROXIMATE/conditional otherwise (see
    APPROXIMATE_METRICS['TTK']).

    Algebraically identical in form to TTB (TTK = TTC - closing/a_max, using the
    accelerate-away bound in place of the brake bound). Kickdown only helps when
    accelerating REDUCES the closing rate (escaping a rear threat or racing to clear a
    crossing point) -- it is actively counterproductive if the interacting agent is a
    lead vehicle ahead (accelerating would INCREASE the closing rate there). Earlier
    versions of this pipeline had no way to tell these two geometric configurations
    apart (closing_traj's sign alone is direction-agnostic). Now that capture_pose
    provides real position/heading, this determines the true ahead/behind relationship
    by projecting the other agent's relative position onto the ego's own heading (the
    same technique already used by rss_full_violation_traj/required_lat_accel_traj) and
    masks TTK to NaN whenever the other agent is genuinely ahead (kickdown would not
    help there, so the value is undefined rather than misleadingly reported). Falls back
    to the old unconditional (direction-unverified) computation when pose isn't
    available in the pack.
    """
    ttc = pack["ttc_traj"]
    cl = pack["closing_traj"]
    kick_time = np.where(cl > 0, cl / a_max, np.nan)
    ttk = ttc - kick_time
    if "ego_x_traj" in pack and "other_x_traj" in pack:
        ex, ey, eh = pack["ego_x_traj"], pack["ego_y_traj"], pack["ego_heading_traj"]
        ox, oy = pack["other_x_traj"], pack["other_y_traj"]
        d_long = (ox - ex) * np.cos(eh) + (oy - ey) * np.sin(eh)
        ttk = np.where(d_long < 0, ttk, np.nan)  # other genuinely behind ego
    return ttk


def time_to_react_approx_traj(pack: dict[str, np.ndarray], *, a_max: float = A_MAX) -> np.ndarray:
    """TTR -- Time To React [Hillenbrand 2007 dissertation eq. 5.16: TTR ~= max(TTB, TTS,
    TTK); Tamke, Dang, Breuel 2011 generalized the same max-over-maneuvers structure].
    APPROXIMATE, conservative, CHEAP VARIANT (see APPROXIMATE_METRICS['TTR']).

    This is the trajectory-only variant: TTR_approx = max(TTB, TTK), which omits TTS (the
    steer-maneuver term). TTS *is* now computable (see time_to_steer_traj below) when
    pose+width are captured, but it is far more expensive per-step than every other
    metric in this module (a discretized forward-simulation search, not a closed-form
    solve) -- compute_all_metrics() computes the fuller max(TTB, TTK, TTS) separately
    (as 'TTR_full_min_s_approx') when pose+width are available, rather than making every
    caller of this cheap function pay TTS's cost implicitly. Since dropping a term from a
    max can only reduce the result, this function's TTR_approx <= the true TTR always --
    a systematic *underestimate* of how much time remains, i.e. it biases toward treating
    situations as more urgent than they truly are, which is the safe direction for a
    criticality metric to be wrong in.
    """
    # np.fmax (not np.maximum) is required here: TTK is frequently NaN (the direction-
    # gating fix means "kickdown isn't a valid maneuver here", not "unknown data"), and
    # np.maximum propagates NaN through the whole max -- silently making TTR NaN even
    # when TTB is a perfectly good, available answer. np.fmax correctly ignores a NaN
    # operand and falls back to the other maneuver's value instead (found via testing
    # this file's own real captured data, which routinely has TTK=NaN).
    return np.fmax(time_to_brake_traj(pack, a_max=a_max), time_to_kickdown_traj(pack, a_max=a_max))


def time_to_steer_traj(
    pack: dict[str, np.ndarray],
    *,
    a_lat_mag: float = 7.0,
    n_tau: int = 9,
    horizon_s: float = 3.0,
    dt_grid: float = 0.2,
) -> np.ndarray:
    """TTS -- Time To Steer, the maneuver='steer' special case of TTM [Hillenbrand 2007
    dissertation Sec. 5.2.4]. APPROXIMATE, requires pose+width (see
    APPROXIMATE_METRICS['TTS']).

    Hillenbrand's own TTS is a circular-arc turning-radius model solved by nested
    interval bisection -- genuinely hard to reproduce faithfully. Instead, this follows
    the architecturally SIMPLER approach used by the actively-maintained CommonRoad-CriMe
    reference toolbox (commonroad_crime/measure/time/tts.py, confirmed by reading its
    actual solver code): bisect over the maneuver-start offset tau in [0, TTC], and at
    each candidate tau, forward-simulate a point-mass steering maneuver (constant lateral
    acceleration a_lat_mag, starting from zero lateral velocity at tau; longitudinal speed
    held at its current value) against the other agent's constant-velocity-extrapolated
    path, checking disc-vs-disc collision (radius = each agent's own circumscribing-circle
    radius from length/width, the same disc approximation used by WTTC and
    CommonRoad-CriMe's own multi-disc chains, simplified here to one disc per vehicle
    instead of three). TTS = max(TTM(steer left), TTM(steer right)), each computed by
    scanning n_tau candidate tau values on [0,TTC] and taking the largest one that avoids
    collision over the whole horizon_s window (a discretized, not exact-bisection, search
    -- coarser than CommonRoad-CriMe's binary_search() but avoids implementing their full
    jerk-limited SimulationLat point-mass integrator). If no candidate tau (including
    tau=0, i.e. steer immediately) avoids collision, returns -inf (matches the survey's
    own {-inf} U [0,inf) output scale for TTM/TTS: no steer maneuver, however early,
    avoids the collision under this model).

    Deliberately NOT vectorized over (ego, t, tau, s) simultaneously -- this is
    substantially more expensive per-call than every other metric in this module (each
    (ego,t) pair does 2 * n_tau forward simulations of horizon_s/dt_grid steps); the
    n_tau/horizon_s/dt_grid defaults trade resolution for runtime. If you want a fuller
    TTR than time_to_react_approx_traj's max(TTB,TTK), combine this with those:
    max(TTB, TTK, TTS).
    """
    ex, ey, eh, ev = pack["ego_x_traj"], pack["ego_y_traj"], pack["ego_heading_traj"], pack["speed_traj"]
    ox, oy, oh, ov = (
        pack["other_x_traj"],
        pack["other_y_traj"],
        pack["other_heading_traj"],
        pack["other_speed_traj"],
    )
    ttc = pack["ttc_traj"]
    r_ego_arr = np.hypot(pack["ego_length"] / 2.0, pack["ego_width"] / 2.0)
    r_other_arr = np.hypot(pack["other_length_traj"] / 2.0, pack["other_width_traj"] / 2.0)

    n, T = ex.shape
    out = np.full((n, T), np.nan, dtype=np.float64)
    s_grid = np.arange(0.0, horizon_s + dt_grid, dt_grid)

    for i in range(n):
        r_sum_base = r_ego_arr[i]
        for t in range(T):
            tt = ttc[i, t]
            if not (np.isfinite(tt) and tt > 0):
                continue
            r_sum = r_sum_base + r_other_arr[i, t]
            if not np.isfinite(r_sum):
                continue
            heading, speed = eh[i, t], ev[i, t]
            hux, huy = np.cos(heading), np.sin(heading)
            pux, puy = -np.sin(heading), np.cos(heading)
            ovx = ov[i, t] * np.cos(oh[i, t])
            ovy = ov[i, t] * np.sin(oh[i, t])
            ex0, ey0, ox0, oy0 = ex[i, t], ey[i, t], ox[i, t], oy[i, t]
            long_d = speed * s_grid
            oth_x = ox0 + s_grid * ovx
            oth_y = oy0 + s_grid * ovy

            taus = np.linspace(0.0, float(tt), n_tau)
            best = -np.inf
            for sign in (1.0, -1.0):
                for tau in taus:
                    lat_d = np.where(s_grid >= tau, 0.5 * sign * a_lat_mag * (s_grid - tau) ** 2, 0.0)
                    eg_x = ex0 + long_d * hux + lat_d * pux
                    eg_y = ey0 + long_d * huy + lat_d * puy
                    d = np.hypot(eg_x - oth_x, eg_y - oth_y)
                    if d.min() > r_sum:
                        best = max(best, float(tau))
            out[i, t] = best
    return out


def potential_ttc_traj(pack: dict[str, np.ndarray], *, assumed_decel: float = A_MAX) -> np.ndarray:
    """PTTC -- Potential Time To Collision [Wakabayashi, Takahashi, Niimi, Renge 2003,
    "Traffic Conflict Analysis using Vehicle Tracking System/Digital VCR and Proposal of a
    New Conflict Indicator", Infrastructure Planning Review 20:949-956, Sec. 6].
    APPROXIMATE (see APPROXIMATE_METRICS['PTTC']).

    PTTC(A1,A2,t) = (1/a)*(-d_dot +/- sqrt(d_dot^2+2*a*d)): TTC generalized to assume the
    closing rate additionally decelerates at rate `a` (nominally the OTHER agent braking).
    The original paper never measures the other agent's true deceleration either -- Sec. 6
    states they used a PRESET constant, empirically derived from three braking-severity
    classes measured on real vehicles ("mild"=foot off accelerator~=0.93 m/s^2,
    "medium"=light braking~=2.78 m/s^2, "hard"=moderately hard braking~=5.56 m/s^2), and
    plotted three PTTC severity variants per conflict event. That is exactly what makes
    this usable here: pass assumed_decel=0.93/2.78/5.56 to reproduce Wakabayashi's own
    three variants exactly; the default below (this env's own A_MAX=4.0) falls between
    their medium and hard presets.
    """
    d = pack["min_dist_traj"]
    cl = pack["closing_traj"]
    a = assumed_decel
    disc = cl**2 - 2.0 * a * d
    with np.errstate(invalid="ignore"):
        t = (cl - np.sqrt(np.maximum(disc, 0.0))) / a
    valid = np.isfinite(d) & np.isfinite(cl) & (cl > 0) & (disc >= 0)
    return np.where(valid, t, np.inf)


def _time_advantage_traj(pack: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Shared solve for TA / PrET (constant-velocity case) and SPrET.

    Requires pose (see has_pose). Solves p1(t) + t1*v1(t) = p2(t) + t2*v2(t) for the two
    straight-line paths' crossing times t1, t2 (both agents' velocity vectors
    reconstructed from heading + speed, per Neurohr et al. 2021's own constant-velocity
    model). A per-instant computation -- like TTC/PTTC, this only uses state AT time t,
    so it does NOT need other_id continuity across steps (unlike a hypothetical ET/PET
    implementation, which would).
    """
    ex, ey, eh, ev = pack["ego_x_traj"], pack["ego_y_traj"], pack["ego_heading_traj"], pack["speed_traj"]
    ox, oy, oh, ov = (
        pack["other_x_traj"],
        pack["other_y_traj"],
        pack["other_heading_traj"],
        pack["other_speed_traj"],
    )
    oid = pack["other_id_traj"]

    v1x, v1y = ev * np.cos(eh), ev * np.sin(eh)
    v2x, v2y = ov * np.cos(oh), ov * np.sin(oh)
    dx, dy = ox - ex, oy - ey
    det = v2x * v1y - v1x * v2y
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-dx * v2y + v2x * dy) / det
        t2 = (v1x * dy - v1y * dx) / det
    valid = (
        (oid >= 0)
        & (np.abs(det) > 1e-6)
        & np.isfinite(t1)
        & np.isfinite(t2)
        & (t1 >= 0)
        & (t2 >= 0)
    )
    return np.where(valid, t1, np.nan), np.where(valid, t2, np.nan)


def _crossing_point_traj(
    pack: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like _time_advantage_traj, but also returns the PREDICTED crossing point
    (px, py) = ego_pos(t) + t1*v1(t) -- reuses the same validated constant-velocity
    solve (see _time_advantage_traj's own hand-verified test cases) rather than trying
    to find where the two agents' actual (noisy, 10Hz) observed paths literally cross,
    which is a much more fragile computation. Used by encroachment_times_traj() below."""
    ex, ey, eh, ev = pack["ego_x_traj"], pack["ego_y_traj"], pack["ego_heading_traj"], pack["speed_traj"]
    t1, t2 = _time_advantage_traj(pack)
    v1x, v1y = ev * np.cos(eh), ev * np.sin(eh)
    px = np.where(np.isfinite(t1), ex + t1 * v1x, np.nan)
    py = np.where(np.isfinite(t1), ey + t1 * v1y, np.nan)
    return t1, t2, px, py


def _point_in_oriented_rect(
    px: np.ndarray, py: np.ndarray, cx: float, cy: float, heading: np.ndarray, half_l: np.ndarray, half_w: np.ndarray
) -> np.ndarray:
    """Whether point (px,py) [world frame] falls inside a rectangle centered at (cx,cy),
    oriented along `heading`, with half-length half_l and half-width half_w."""
    dx, dy = px - cx, py - cy
    hux, huy = np.cos(heading), np.sin(heading)
    pux, puy = -np.sin(heading), np.cos(heading)
    along = dx * hux + dy * huy
    across = dx * pux + dy * puy
    return (np.abs(along) <= half_l) & (np.abs(across) <= half_w)


def encroachment_times_traj(pack: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """ET -- Encroachment Time [Allen, Shin, Cooper 1978] and PET -- Post Encroachment
    Time [same source]. APPROXIMATE, scenario-level (one value per ego, not per-step --
    see APPROXIMATE_METRICS['ET/PET']).

    ET(A1,CA) = t_exit(A1,CA) - t_entry(A1,CA); PET(A1,A2,CA) = t_entry(A2,CA) -
    t_exit(A1,CA), defined only when A1 leaves CA before or at the time A2 enters it.
    The conflict area CA is operationally defined as each agent's own real, ORIENTED
    rectangular footprint (length x width, at its own per-step heading -- using width
    was a later upgrade; an earlier version used a circular disc of radius=length/2,
    under-using data this pipeline already captures) sweeping through the predicted
    path-crossing point (see _crossing_point_traj), matching Laureshyn et al.'s own
    practical rectangular-footprint refinement of Allen et al.'s definition more closely
    than a disc does. Computed once per maximal other_id-stable window (the window's
    crossing-point prediction closest to, but not after, the event is used as the
    anchor), then each agent's own entry/exit is the first/last timestep within that
    window where the crossing point falls inside its own oriented rectangle. If an ego
    has multiple such windows (interacting with different agents at different times),
    the MOST critical (smallest) ET/PET across windows is reported, matching this
    module's worst-case-aggregate convention elsewhere (TTC_min, TTB_min, ...).
    """
    ex, ey, eh = pack["ego_x_traj"], pack["ego_y_traj"], pack["ego_heading_traj"]
    ox, oy, oh = pack["other_x_traj"], pack["other_y_traj"], pack["other_heading_traj"]
    elen, ewid = pack["ego_length"], pack["ego_width"]
    olen, owid = pack["other_length_traj"], pack["other_width_traj"]
    oid = pack["other_id_traj"]
    _t1, _t2, px, py = _crossing_point_traj(pack)

    n, T = ex.shape
    et_out = np.full(n, np.nan, dtype=np.float64)
    pet_out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        ids = oid[i]
        t = 0
        while t < T:
            if ids[t] < 0:
                t += 1
                continue
            t_start = t
            while t + 1 < T and ids[t + 1] == ids[t]:
                t += 1
            t_end = t  # inclusive; [t_start, t_end] is one maximal stable-id window

            # The crossing prediction is only valid (t1>=0) for timesteps BEFORE the
            # predicted event -- it goes NaN once the crossing has passed (t1 would be
            # negative). Anchor on the timestep with the smallest nonnegative t1 within
            # the window: the prediction made closest to, but not after, the event.
            window_t1 = _t1[i, t_start : t_end + 1]
            valid_window = np.isfinite(window_t1)
            if not valid_window.any():
                t += 1
                continue
            anchor = t_start + int(np.flatnonzero(valid_window)[np.argmin(window_t1[valid_window])])

            if np.isfinite(px[i, anchor]):
                xstar, ystar = float(px[i, anchor]), float(py[i, anchor])
                sl = slice(t_start, t_end + 1)

                # _point_in_oriented_rect(px,py, cx,cy, heading, half_l, half_w): test
                # whether the fixed crossing point (xstar,ystar) falls inside a rectangle
                # centered at each agent's own moving position -- xstar,ystar go in the
                # function's "point" slot, the agent's per-step position/heading go in
                # its "rectangle" slot.
                ego_valid = np.isfinite(ex[i, sl])
                near_ego = ego_valid & _point_in_oriented_rect(
                    xstar, ystar, ex[i, sl], ey[i, sl], eh[i, sl], elen[i] / 2.0, ewid[i] / 2.0
                )
                other_valid = np.isfinite(ox[i, sl])
                near_other = other_valid & _point_in_oriented_rect(
                    xstar, ystar, ox[i, sl], oy[i, sl], oh[i, sl], olen[i, sl] / 2.0, owid[i, sl] / 2.0
                )

                if near_ego.any():
                    idx_ego = np.flatnonzero(near_ego)
                    t_entry_ego, t_exit_ego = idx_ego[0], idx_ego[-1]
                    et_val = float(t_exit_ego - t_entry_ego) * DT
                    et_out[i] = et_val if np.isnan(et_out[i]) else min(et_out[i], et_val)

                    if near_other.any():
                        idx_other = np.flatnonzero(near_other)
                        t_entry_other = idx_other[0]
                        if t_entry_other >= t_exit_ego:
                            pet_val = float(t_entry_other - t_exit_ego) * DT
                            pet_out[i] = pet_val if np.isnan(pet_out[i]) else min(pet_out[i], pet_val)
            t += 1

    return et_out, pet_out


def time_advantage_traj(pack: dict[str, np.ndarray]) -> np.ndarray:
    """TA -- Time Advantage [Hansson 1975; formalized by Laureshyn, Svensson, Hyden
    2010]; also the constant-velocity special case of PrET -- Predictive Encroachment
    Time [Neurohr, Bussler, Koopmann, Kamran, Reich 2021, "Criticality Analysis for the
    Verification and Validation of Automated Vehicles", IEEE Access, Sec. V.A.6.d]. EXACT
    given captured pose (see EXACT_METRICS['TA/PrET']).

    TA(A1,A2,t) = |t1~-t2~| where t1~,t2~ solve p1(t+t1~) = p2(t+t2~) under each agent's
    own constant-velocity extrapolation (heading+speed reconstructed into a velocity
    vector) -- i.e. the predicted PET assuming both agents hold their current path.
    Undefined (NaN) when the two paths are parallel (never cross) or the crossing lies in
    either agent's past.
    """
    t1, t2 = _time_advantage_traj(pack)
    return np.abs(t1 - t2)


def scaled_predictive_encroachment_time_traj(pack: dict[str, np.ndarray]) -> np.ndarray:
    """SPrET -- Scaled Predictive Encroachment Time [Neurohr et al. 2021, eq. (1)].
    EXACT given captured pose (see EXACT_METRICS['TA/PrET']).

    SPrET = |t1~^2 - t2~^2| = (t1~+t2~)*|t1~-t2~| -- down-weights situations far before
    the predicted intersection relative to plain TA/PrET, "incorporates prediction
    uncertainty" per the original paper.
    """
    t1, t2 = _time_advantage_traj(pack)
    return np.abs(t1**2 - t2**2)


def has_crosswalks(pack: dict[str, np.ndarray]) -> bool:
    """True when capture_map_geometry crosswalk fields are present AND non-empty for at
    least one scenario (a map with zero crosswalks in view still has the keys, just with
    size-0 arrays -- confirmed on a real map, which had 5 crosswalk polylines / 20 pts)."""
    return "crosswalk_polyline_lengths" in pack and pack["crosswalk_polyline_lengths"].size > 0


def _unpack_polylines(
    x: np.ndarray, y: np.ndarray, lengths: np.ndarray, scenario_ids: np.ndarray
) -> list[tuple[int, np.ndarray]]:
    """Reconstruct (scenario_id, points[k,2]) polylines from capture_map_geometry's flat
    encoding (x/y concatenated across all polylines, lengths[k] = point count of
    polyline k)."""
    out: list[tuple[int, np.ndarray]] = []
    offset = 0
    for length, sid in zip(lengths.tolist(), scenario_ids.tolist()):
        pts = np.stack([x[offset : offset + length], y[offset : offset + length]], axis=-1)
        out.append((int(sid), pts))
        offset += length
    return out


def _ray_polyline_ttc(px: float, py: float, vx: float, vy: float, segments: np.ndarray) -> float:
    """Smallest t>=0 at which the ray p+t*v (v already scaled so t comes out in seconds)
    crosses any of the given line segments (shape (S,2,2): S segments, 2 endpoints, xy).
    Returns inf if v is ~0 or no segment is crossed ahead."""
    if (vx * vx + vy * vy) < 1e-6:
        return float("inf")
    ax, ay = segments[:, 0, 0], segments[:, 0, 1]
    bx, by = segments[:, 1, 0], segments[:, 1, 1]
    ex, ey = bx - ax, by - ay
    denom = ex * vy - ey * vx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ex * (ay - py) - ey * (ax - px)) / denom
        s = (vx * (ay - py) - vy * (ax - px)) / denom
    valid = np.isfinite(t) & np.isfinite(s) & (t >= 0) & (s >= 0) & (s <= 1) & (np.abs(denom) > 1e-9)
    if not valid.any():
        return float("inf")
    return float(t[valid].min())


def time_to_zebra_traj(pack: dict[str, np.ndarray]) -> np.ndarray:
    """TTZ -- Time To Zebra [Varhelyi 1998]. APPROXIMATE, requires pose + crosswalk
    geometry (see APPROXIMATE_METRICS['TTZ']) -- unlocked by capture_map_geometry's new
    crosswalk-polyline accessor (verified against a real compiled map: 5 crosswalk
    polylines, 20 points, coordinate frame matches the existing road-edge getter).

    TTZ(A1,CA,t) = min({t~>=0 | d(p1(t+t~), p_CA(t+t~)) = 0} u {inf}): time until the ego,
    extrapolated at its CURRENT heading and speed (a static-target special case of the
    same constant-velocity model used for TA/PrET), first reaches a crosswalk polygon
    boundary. Solved as a ray-vs-polyline-segment intersection (smallest nonnegative
    crossing time over all segments of any crosswalk in the ego's own scenario).
    APPROXIMATE rather than EXACT because: crosswalks are treated as their boundary
    polyline (a thin region), not the paper's zebra "position" abstraction exactly, and
    the constant-velocity extrapolation (like TA/PrET, TTB, etc. elsewhere in this
    module) assumes the ego holds its current heading/speed rather than using a full
    trajectory predictor.
    """
    ex, ey, eh, ev = pack["ego_x_traj"], pack["ego_y_traj"], pack["ego_heading_traj"], pack["speed_traj"]
    n, T = ex.shape
    out = np.full((n, T), np.nan, dtype=np.float64)
    if not has_crosswalks(pack):
        return out

    polylines = _unpack_polylines(
        pack["crosswalk_polyline_x"],
        pack["crosswalk_polyline_y"],
        pack["crosswalk_polyline_lengths"],
        pack["crosswalk_polyline_scenario_id"],
    )
    scene_id = pack["scene_id"]
    segments_by_scene: dict[int, np.ndarray] = {}
    for sid, pts in polylines:
        segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (k-1, 2, 2)
        if segs.shape[0] == 0:
            continue
        segments_by_scene.setdefault(sid, []).append(segs)
    segments_by_scene = {
        sid: np.concatenate(chunks, axis=0) for sid, chunks in segments_by_scene.items()
    }

    for i in range(n):
        segs = segments_by_scene.get(int(scene_id[i]))
        if segs is None:
            continue
        for t in range(T):
            speed = float(ev[i, t])
            heading = float(eh[i, t])
            vx, vy = speed * np.cos(heading), speed * np.sin(heading)
            tt = _ray_polyline_ttc(float(ex[i, t]), float(ey[i, t]), vx, vy, segs)
            if np.isfinite(tt):
                out[i, t] = tt
    return out


# ============================================================================
# 2. DISTANCE-SCALE METRICS -- HW, DCE handled above (Time-Scale section, per the
#    survey's own cross-references: "HW: refer to THW", "DCE: refer to TTCE").
# ============================================================================


def proportion_of_stopping_distance_traj(
    pack: dict[str, np.ndarray], *, a_long_min: float = A_MAX
) -> np.ndarray:
    """PSD -- Proportion of Stopping Distance [Allen 1978; car-following form: Astarita,
    Guido, Vitale, Giofre 2012, "A new microsimulation model for the evaluation of traffic
    safety performances", Eq. 3]. EXACT for car-following/approach interactions (see
    EXACT_METRICS['PSD']) -- this pipeline has no intersection scenarios.

    PSD(A1,CA,t) = d(p1(t),p_CA(t)) / MSD(A1,t), MSD = ||v1(t)||^2/(2*|a1,long,min(t)|).
    Astarita et al. (2012) explicitly define, for rear-end/car-following interactions
    specifically, the "conflict area" p_CA as the LEAD VEHICLE'S OWN POSITION (their
    "remaining distance to the potential point of collision" = the following-to-lead-
    vehicle gap used in their own DRAC formula) -- not an independently-defined
    intersection polygon. That is exactly this pipeline's min_dist_traj, so no
    approximation is needed for the car-following case.
    """
    d = pack["min_dist_traj"]
    v = pack["speed_traj"]
    msd = (v**2) / (2.0 * a_long_min)
    return _safe_div(d, msd)


def accepted_gap_ttc(pack: dict[str, np.ndarray]) -> np.ndarray:
    """AGS -- Accepted Gap Size, reframed as a purely temporal gap [Alhajyaseen, Asano,
    Nakamura 2013's univariate cumulative-Weibull gap-time model; Petzoldt 2014's
    TTA-substitution precedent]. APPROXIMATE (see APPROXIMATE_METRICS['AGS']).

    The interACT project's critique of AGS ("depends on driver age, gender, waiting time,
    road condition...") describes gap-acceptance modeling in general, not what
    Alhajyaseen's own fitted model actually takes as input: their acceptance-probability
    function is univariate over the gap/lag TIME alone, P(x) = 1 - exp(-(x/alpha)^beta) --
    those other covariates explain why DIFFERENT SITES have different alpha/beta, not
    additional function arguments. This surfaces the raw observable half of that model
    (no alpha/beta needed): the realized TTC at the last pre-tight step, i.e. the gap the
    ego implicitly accepted by continuing rather than fully yielding. Pair with
    weibull_gap_acceptance_probability() below if you have domain-fitted alpha/beta.
    """
    ttc = pack["ttc_traj"]
    onset = pack["tight_onset"]
    n = onset.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        e = int(onset[i])
        if e <= 0:
            continue
        v = ttc[i, e - 1]
        if np.isfinite(v):
            out[i] = float(v)
    return out


def weibull_gap_acceptance_probability(gap: np.ndarray, *, alpha: float, beta: float) -> np.ndarray:
    """Cumulative-Weibull gap-acceptance probability [Alhajyaseen, Asano, Nakamura 2013,
    Eq. 2]: P(x) = 1 - exp(-(x/alpha)^beta). alpha (scale, ~ mean accepted gap, seconds)
    and beta (shape) have NO default on purpose -- Alhajyaseen's own fitted values
    (alpha=3.27-7.62s, beta=2.29-4.88) are for a pedestrian left-turn conflict, not this
    domain. Fit your own alpha/beta from accepted_gap_ttc() observations on this pipeline
    if you want a calibrated curve rather than borrowing a domain-inappropriate one.
    """
    gap = np.asarray(gap, dtype=np.float64)
    return 1.0 - np.exp(-np.power(np.maximum(gap, 0.0) / alpha, beta))


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


def conflict_severity_at_onset(pack: dict[str, np.ndarray]) -> np.ndarray:
    """CS -- Conflict Severity [Bagdadi 2013, "Estimation of the severity of safety
    critical events"]. APPROXIMATE, event-triggered (see
    APPROXIMATE_METRICS['CS']).

    CS(A1,A2) = Delta_v(t_evasive) - TTA(A1,A2)*||a1(t_evasive)||*m2/(m1+m2). The survey
    calls this "not run-time capable... TTA can only be computed once evasive maneuver has
    been identified" -- but that is a PRECONDITION (needing an identified onset), not a
    look-ahead requirement: every quantity in the formula is evaluated AT t_evasive using
    only state available at that instant (same computability class as TTC). The Swedish
    Traffic Conflicts Technique manual, which underlies this literature, defines its
    analogous severity moment identically: "at the moment when one of the road users start
    taking an evasive action." This pipeline already detects such an onset causally
    (tight_onset), so CS is evaluated the instant it fires: Delta_v ~ 0.5*|closing| at
    onset (equal-mass convention, same as delta_v_proxy), TTA ~ ttc_traj at onset, and the
    ego's own realized deceleration a1 from a one-step finite difference of speed_traj
    (more physically grounded than the policy's expected-acceleration proxy used
    elsewhere, since this evaluates a single known instant rather than averaging a jerk
    trajectory).
    """
    onset = pack["tight_onset"]
    speed = pack["speed_traj"]
    ttc = pack["ttc_traj"]
    closing = pack["closing_traj"]
    n = onset.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        e = int(onset[i])
        if e <= 0:
            continue
        cl = closing[i, e]
        tta = ttc[i, e]
        if not (np.isfinite(cl) and np.isfinite(tta)):
            continue
        dv = 0.5 * abs(float(cl))
        a1 = abs(float(speed[i, e] - speed[i, e - 1])) / DT
        out[i] = dv - tta * a1 * 0.5
    return out


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


def required_lat_accel_traj(pack: dict[str, np.ndarray]) -> np.ndarray:
    """a_lat,req -- Required Lateral Acceleration [Jansson 2005, "Collision Avoidance
    Theory: With Application to Automotive Collision Mitigation", PhD thesis, Linkoping
    University, Eq. 5.46-5.49 constant-acceleration special case]. APPROXIMATE, requires
    pose+width (see APPROXIMATE_METRICS['a_lat,req']) -- unlocked by capture_pose +
    capture_map_geometry's incidental partner-width plumbing (verified against a real
    compiled rollout: other_width_traj range ~2.0m, realistic).

    a1,lat,k(t) = a2,lat + 2*(v2,lat-v1,lat)/TTC + (2/TTC^2)*[k*(w1+w2)/2 + (p2,lat-p1,lat)],
    for k in {+1,-1} (steer left/right); a_lat,req = min(|a1,lat,left|, |a1,lat,right|) --
    the minimal average lateral acceleration A1 needs, in either direction, to avoid a
    future collision by steering. Decomposes relative position/velocity onto ego's
    heading-perpendicular axis (same technique as rss_full_violation_traj's lateral
    half). v1,lat is the ego's own lateral velocity in its own per-step heading frame --
    unlike the other agent (only ~1/3 of steps valid, see other_id_traj), ego position is
    captured every single step, so this is computed by finite-differencing (ex,ey) and
    projecting onto the perpendicular axis, rather than assumed 0 as an earlier version
    of this function did. a2,lat=0 (the other agent's lateral acceleration) remains
    unmeasured/assumed 0 (same "assume 0" convention already used for a_long,req's a2) --
    other's data coverage is too sparse to finite-difference reliably the way ego's can
    be. w1, w2 are ego_width and other_width.
    """
    ex, ey, eh = pack["ego_x_traj"], pack["ego_y_traj"], pack["ego_heading_traj"]
    ox, oy, oh, ov = (
        pack["other_x_traj"],
        pack["other_y_traj"],
        pack["other_heading_traj"],
        pack["other_speed_traj"],
    )
    ttc = pack["ttc_traj"]
    w1 = pack["ego_width"][:, None]
    w2 = pack["other_width_traj"]

    perp_ux, perp_uy = -np.sin(eh), np.cos(eh)
    dx, dy = ox - ex, oy - ey
    d_lat = dx * perp_ux + dy * perp_uy

    if ex.shape[1] >= 2:
        ego_vx = np.gradient(ex, DT, axis=1)
        ego_vy = np.gradient(ey, DT, axis=1)
        v1_lat = ego_vx * perp_ux + ego_vy * perp_uy
    else:
        v1_lat = np.zeros_like(ex)  # single-step pack: fall back to the old v1_lat=0 assumption

    v2x, v2y = ov * np.cos(oh), ov * np.sin(oh)
    v_lat_diff = (v2x * perp_ux + v2y * perp_uy) - v1_lat  # (v2,lat - v1,lat)

    with np.errstate(divide="ignore", invalid="ignore"):
        base = 2.0 * v_lat_diff / ttc
        term = 2.0 / ttc**2
        a_left = base + term * ((w1 + w2) / 2.0 + d_lat)
        a_right = base + term * (-(w1 + w2) / 2.0 + d_lat)
    a_req = np.minimum(np.abs(a_left), np.abs(a_right))
    valid = np.isfinite(ttc) & (ttc > 0) & np.isfinite(d_lat) & np.isfinite(w2)
    return np.where(valid, a_req, np.nan)


def steer_threat_number_traj(pack: dict[str, np.ndarray], *, a_lat_min: float = 7.0) -> np.ndarray:
    """STN -- Steer Threat Number [Jansson 2005; multi-actor extension Eidehall 2011].
    APPROXIMATE (see APPROXIMATE_METRICS['a_lat,req']).

    STN = a_lat,req / a_lat,min. By definition STN >= 1 means a steering maneuver cannot
    avoid the collision under the assumed model. a_lat_min defaults to 7.0 m/s^2, Jansson's
    own demonstrator-vehicle deployed lateral-capability bound (Table 8.3) -- the more
    conservative of his two reported values (the other, 9.82 m/s^2 =~ 1g, is his idealized
    physical limit; pass a_lat_min=9.82 to use that instead).
    """
    return required_lat_accel_traj(pack) / a_lat_min


def combined_required_accel_traj(pack: dict[str, np.ndarray]) -> np.ndarray:
    """a_req -- combined Required Acceleration [Jansson 2005]. APPROXIMATE (see
    APPROXIMATE_METRICS['a_lat,req']).

    a_req = sqrt(a_long,req^2 + a_lat,req^2). Jansson's thesis actually gives THREE
    distinct combination formulas (a plain min(), this sqrt-of-squares "SOCC" form, and a
    friction-ellipse feasibility check) -- this implements the sqrt form since that's
    what the Westhofen survey itself attributes to Jansson and is the most commonly cited
    version, but per a direct read of Jansson Eq. 5.59-5.61 it is itself already a
    simplification of his true joint-optimal (A_x, A_y) solve, not a literal transcription
    -- flagged explicitly since this implementation combines two INDEPENDENTLY computed
    1D terms (required_long_decel_traj, required_lat_accel_traj) post-hoc, which is the
    survey's reading of Jansson, not Jansson's own coupled optimization.
    """
    a_long = required_long_decel_traj(pack)
    a_lat = required_lat_accel_traj(pack)
    return np.sqrt(a_long**2 + a_lat**2)


# ============================================================================
# 5. JERK-SCALE METRICS
# ============================================================================


def longitudinal_jerk_proxy(pack: dict[str, np.ndarray]) -> np.ndarray:
    """LongJ -- Longitudinal Jerk [general concept; curve-safety use in Ambros2019].
    EXACT when accel_traj is present, APPROXIMATE otherwise (see
    APPROXIMATE_METRICS['LongJ']).

    LongJ(A1,t) = j1,long(t) = d(a1,long)/dt. rollout.py now persists accel_traj -- the
    actually-sampled/executed acceleration for dynamics_model="classic" (this pipeline's
    default) -- which was already computed every step but previously discarded after
    only feeding mean_hard_brake. When present, LongJ is the exact finite-difference
    jerk of that real signal. Packs saved before this addition (or using
    dynamics_model="jerk", where the C side's own jerk_long field would be the right
    source instead -- not currently exposed) only have exp_accel_traj, the policy's
    softmax-expected acceleration -- finite-differencing that remains a decision-
    smoothness proxy, not physically realized jerk, for those older packs.
    """
    a = pack["accel_traj"] if "accel_traj" in pack else pack["exp_accel_traj"]
    jerk = np.diff(a, axis=1) / DT
    pad = np.full((jerk.shape[0], 1), np.nan, dtype=jerk.dtype)
    return np.concatenate([pad, jerk], axis=1)


# LatJ: NOT_APPLICABLE -- needs an executed lateral-accel signal (steer not persisted),
# unlike a_lat,req below which only needs pose+width (see NOT_APPLICABLE['LatJ']).


# ============================================================================
# 6. INDEX-SCALE METRICS
# ============================================================================


def accident_metric(pack: dict[str, np.ndarray]) -> np.ndarray:
    """AM -- Accident Metric [general/implicit, e.g. GIDAS database usage].

    AM(Sc) = 0 if no accident happened, 1 otherwise. Exact: this is exactly the pack's own
    `collided` flag, formalized under its survey name.
    """
    return pack["collided"].astype(np.float64)


def crash_potential_index(
    pack: dict[str, np.ndarray],
    *,
    madr_mean: float = 8.45,
    madr_std: float = 1.40,
    madr_bounds: tuple[float, float] = (4.23, 12.69),
) -> np.ndarray:
    """CPI -- Crash Potential Index [Cunto, F.J.C. (2008), "Assessing Safety Performance
    of Transportation Systems Using Microscopic Simulation", PhD thesis, U. Waterloo,
    Sec. 4.3, Eq. 4.1-4.2; Cunto & Saccomanno 2007/2008]. APPROXIMATE (see
    APPROXIMATE_METRICS['CPI']).

    CPI(A1,A2) = (1/(te-t0)) * integral P(MADR <= DRAC(t)) dt: the time-averaged
    probability that a vehicle's own Maximum Available Deceleration Rate (MADR, modeled by
    Cunto as normally distributed across the vehicle fleet, fit to field braking-test
    data) is insufficient to avoid the collision implied by the current DRAC (= this
    module's a_long,req). Cunto's thesis gives the actual fitted parameters for passenger
    cars: mean=8.45 m/s^2, std=1.40 m/s^2, truncated to [4.23, 12.69] m/s^2 (5.01/1.40/
    [2.05,7.98] for trucks) -- used verbatim here, replacing an earlier deterministic-
    threshold version of this function that used the environment's own -4.0 m/s^2 action
    bound as a crude 0/1 cutoff. Note: this env's action-space bound (4.0 m/s^2) sits
    ~3.2 std below Cunto's mean, deep in the tail where a real car would virtually always
    still be capable of braking harder -- so CPI computed this way will generally run low,
    reflecting the environment's own conservative braking bound relative to real vehicles,
    not necessarily an error in the policy or the metric.
    """
    a_req = required_long_decel_traj(pack)
    drac = np.abs(a_req)
    lo, hi = madr_bounds
    drac_clipped = np.clip(drac, lo, hi)
    p = _norm_cdf((drac_clipped - madr_mean) / madr_std)
    p = np.where(np.isfinite(drac), p, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(p, axis=1)


def rss_longitudinal_min_distance_traj(
    pack: dict[str, np.ndarray],
    *,
    rho: float = 1.0,
    a_max_accel: float = A_RSS_MAX_ACCEL,
    a_min_brake: float = A_MAX,
    a_max_brake: float = A_RSS_MAX_BRAKE_OTHER,
) -> np.ndarray:
    """Longitudinal half of RSS-DS's d_min [Shalev-Shwartz, Shammah, Shashua 2017, "On a
    Formal Model of Safe and Scalable Self-Driving Cars", arXiv:1708.06374, Definition 1 +
    Lemma 2]. APPROXIMATE (see APPROXIMATE_METRICS['RSS_long_violation']).

    d_min = v_r*rho + 0.5*a_max_accel*rho^2 + (v_r+rho*a_max_accel)^2/(2*a_min_brake)
            - v_f^2/(2*a_max_brake)
    with v_r = ego (rear/following) speed, v_f = lead speed (approximated as
    ego_speed - closing_speed, same assumption as DST/CPI's v2 above). rho and the three
    accel/brake bounds are three DISTINCT parameters the original paper deliberately
    leaves unspecified ("should be determined... by regulation"); defaults here are the
    field-standard values from Intel's ad-rss-lib reference implementation (rho=1.0s,
    a_max_accel=3.5, a_min_brake=4.0 [this env's own bound, "German driving school" value],
    a_max_brake=8.0 for what the OTHER car might do -- NOT the same as a_min_brake, a
    mistake an earlier version of this function made by reusing a single A_MAX for both).
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


def rss_full_violation_traj(
    pack: dict[str, np.ndarray],
    *,
    rho: float = 1.0,
    a_max_accel_long: float = A_RSS_MAX_ACCEL,
    a_min_brake_long: float = A_MAX,
    a_max_brake_long: float = A_RSS_MAX_BRAKE_OTHER,
    a_max_accel_lat: float = 0.2,
    a_min_brake_lat: float = 0.8,
    mu: float = 0.1,
) -> np.ndarray:
    """RSS-DS -- full Responsibility-Sensitive Safety Dangerous-Situation flag, both axes
    [Shalev-Shwartz, Shammah, Shashua 2017, Definitions 1+2 (longitudinal, Lemma 2) and
    Definition 6 (lateral, Lemma 4)]. APPROXIMATE, requires pose (see
    APPROXIMATE_METRICS['RSS_full']) -- this SUPERSEDES rss_longitudinal_violation_traj
    (which cannot determine true ahead/behind or evaluate the lateral axis at all) when
    pose is available.

    RSS-DS = 1 iff BOTH the longitudinal AND lateral safe distances are simultaneously
    violated (Definition 9). Longitudinal front/rear roles are now determined from the
    real sign of the projected relative position (previously assumed ego=rear always).
    Lateral closing is evaluated via ego-heading-relative projection: since ego's own
    lateral velocity in its own frame is ~0 by construction (only longitudinal motion is
    directly measured), the "positive-side car" / "negative-side car" role assignment in
    the paper's Definition 6 is resolved by the sign of the projected lateral offset. The
    lateral accel/brake bounds (a_max_accel_lat=0.2, a_min_brake_lat=0.8, mu=0.1m) are the
    same Intel ad-rss-lib reference values used for the longitudinal bounds elsewhere in
    this module -- the original paper itself publishes no numeric values for either axis.
    Flagged APPROXIMATE rather than EXACT because: (a) the lateral sign-role assignment is
    this implementation's own resolution of an ambiguity in the paper's v_i +/- rho*a_lat
    notation, not something verified against a reference implementation, and (b) ego's own
    lateral velocity is assumed zero rather than measured.
    """
    ex, ey, eh, ev = pack["ego_x_traj"], pack["ego_y_traj"], pack["ego_heading_traj"], pack["speed_traj"]
    ox, oy, oh, ov = (
        pack["other_x_traj"],
        pack["other_y_traj"],
        pack["other_heading_traj"],
        pack["other_speed_traj"],
    )

    heading_ux, heading_uy = np.cos(eh), np.sin(eh)
    perp_ux, perp_uy = -np.sin(eh), np.cos(eh)

    dx, dy = ox - ex, oy - ey
    d_long = dx * heading_ux + dy * heading_uy
    d_lat = dx * perp_ux + dy * perp_uy

    v2x, v2y = ov * np.cos(oh), ov * np.sin(oh)
    v2_long = v2x * heading_ux + v2y * heading_uy
    v2_lat = v2x * perp_ux + v2y * perp_uy
    v1_long = ev
    v1_lat = np.zeros_like(ev)

    # Longitudinal: assign front/rear roles from the measured sign of d_long.
    ahead = d_long >= 0
    v_front = np.clip(np.where(ahead, v2_long, v1_long), 0.0, None)
    v_rear = np.clip(np.where(ahead, v1_long, v2_long), 0.0, None)
    d_min_long = np.maximum(
        v_rear * rho
        + 0.5 * a_max_accel_long * rho**2
        + (v_rear + rho * a_max_accel_long) ** 2 / (2.0 * a_min_brake_long)
        - v_front**2 / (2.0 * a_max_brake_long),
        0.0,
    )

    # Lateral: assign positive-/negative-side roles from the measured sign of d_lat.
    other_is_positive = d_lat >= 0
    v1_l = np.where(other_is_positive, v2_lat, v1_lat)
    v2_l = np.where(other_is_positive, v1_lat, v2_lat)
    gap_lat = np.abs(d_lat)
    v1_rho = v1_l + rho * a_max_accel_lat
    v2_rho = v2_l - rho * a_max_accel_lat
    d_min_lat = mu + np.maximum(
        0.0,
        (v1_l + v1_rho) / 2.0 * rho
        + v1_rho**2 / (2.0 * a_min_brake_lat)
        - ((v2_l + v2_rho) / 2.0 * rho - v2_rho**2 / (2.0 * a_min_brake_lat)),
    )

    valid = np.isfinite(d_long) & np.isfinite(d_lat)
    violation = (np.abs(d_long) < d_min_long) & (gap_lat < d_min_lat)
    return np.where(valid, violation, False)


def safety_potential_traj(
    pack: dict[str, np.ndarray], *, a_min: float = A_MAX, k: float = 2.0
) -> np.ndarray:
    """SP -- Safety Potential / Safety Force Field [Nister, Lee, Ng, Wang 2019, NVIDIA
    "The Safety Force Field" whitepaper, worked "Implementation Example"; companion doc
    "An Introduction to the Safety Force Field" for the plain-language 1D worked cases].
    APPROXIMATE (see APPROXIMATE_METRICS['SP']).

    Implements the whitepaper's own single-control-policy worked example:
    SP = ||(t_ego,finish - t_int, t_other,finish - t_int)||_k, with t_i,finish = v_i/|a_min|
    (time for actor i to reach a full stop under a fixed max-braking safety procedure) and
    t_int the earliest claimed-set intersection time -- which, in the paper's own 1D
    worked case (no footprints for the general 2D claimed-set geometry, which this
    pipeline doesn't have), collapses to exactly this pipeline's own TTC. v_other is
    approximated the same way as elsewhere (ego_speed - closing_speed). Neither the paper
    nor its companion doc publishes a numeric a_min -- "implementors will... add margins
    for their reaction time" -- so a_min defaults to this env's own A_MAX bound, same
    convention as BTN/DST/TTB. NOTE: the source's own sign/monotonicity convention for
    when SP indicates "more dangerous" is subtle (its own unsafe-set inequalities are
    stated as ">= 0" almost everywhere) -- treat this implementation as the paper's literal
    formula, but verify against the original before using SP values to gate any decision.
    """
    v1 = pack["speed_traj"]
    cl = pack["closing_traj"]
    ttc = pack["ttc_traj"]
    v2 = np.clip(v1 - cl, 0.0, None)
    t1 = v1 / a_min
    t2 = v2 / a_min
    d1 = t1 - ttc
    d2 = t2 - ttc
    valid = np.isfinite(d1) & np.isfinite(d2)
    sp = np.where(valid, (np.abs(d1) ** k + np.abs(d2) ** k) ** (1.0 / k), np.nan)
    return sp


# ACI, CI, PRI, SOI, TCI, STN: NOT_APPLICABLE -- see dict below.


# ============================================================================
# 7. PROBABILITY-SCALE METRICS
# ============================================================================


def monte_carlo_collision_probability(packs_by_seed: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    """P-MC-seed -- policy-checkpoint outcome variance, NOT Broadhurst's per-instant P-MC
    [Broadhurst, Baker, Kanade 2005, "Monte Carlo Road Safety Reasoning"; primary-source
    detail from the open CMU-RI-TR-04-11 precursor report]. APPROXIMATE, RELABELED (see
    APPROXIMATE_METRICS['P-MC']).

    IMPORTANT DISTINCTION, confirmed against the primary source: Broadhurst's
    P-MC(A1,S,t) = integral P(C|U)P(U)dU is a *state-conditioned, instantaneous* quantity
    -- "given the scene right now, what is the probability of a collision under
    uncertainty about what everyone does over the next t_H seconds" (aleatoric uncertainty
    over control-input choices AT THIS MOMENT). What this function computes is different:
    the empirical fraction of independently-trained-and-seeded policy checkpoints (same
    method, same map) that collide anywhere over a full episode -- a policy-level,
    trajectory-level statistic marginalizing over WHICH CHECKPOINT is deployed, not over
    per-instant action-choice uncertainty in a fixed scene. Both are legitimate collision-
    probability-under-stochasticity estimates, but they are not the same random variable;
    this one is better read as a policy-class robustness/reliability statistic.

    A genuinely closer P-MC approximation IS conceptually available (the ego's own
    per-bin action-distribution probabilities from action_stats_from_logits() are
    structurally exactly Broadhurst's P(u_ego) term -- his own first implementation didn't
    model the other agent's actions either, "the decision tree only contains actions of
    our own car", mirroring this pipeline's own data gap), but it needs the FULL per-bin
    action-probability vector per step, which rollout.py currently does NOT persist (only
    scalar aggregates like p_brake/exp_accel are saved) -- the same class of missing-
    capture issue documented for the steer signal (see NOT_APPLICABLE['LatJ']).

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
# 8. POTENTIAL-SCALE METRICS -- SP is implemented in the Index-Scale section above
#    (was ported there before this comment was updated; left in place to avoid an
#    unnecessary code move). PF is implemented below.
# ============================================================================


def _nearest_point_to_rect_distance(
    px: np.ndarray, py: np.ndarray, half_l: float | np.ndarray, half_w: np.ndarray
) -> np.ndarray:
    """Distance from point (px,py), given in a rectangle's own body frame (rectangle
    centered at origin, half-length half_l along x, half-width half_w along y), to the
    rectangle's boundary. 0 if the point is inside/on the rectangle."""
    dx = np.maximum(np.abs(px) - half_l, 0.0)
    dy = np.maximum(np.abs(py) - half_w, 0.0)
    return np.hypot(dx, dy)


def potential_functions_traj(
    pack: dict[str, np.ndarray],
    *,
    a_car: float = 10.0,
    alpha_car: float = 0.5,
    beta_car: float = 0.6,
    t_follow: float = 3.0,
    d0_car: float = 50.0,
    a_lane: float = 2.0,
    sigma_lane: float | None = None,
    eta_road: float = 3.0,
    gamma_vel: float = 0.2,
    v_des: float = 25.0,
) -> np.ndarray:
    """PF -- Potential Functions as Superposition of Scoring Functions [Wolf, Burdick
    2008, "Artificial Potential Functions for Highway Driving with Collision Avoidance",
    IEEE ICRA 2008; full text + Table I parameters obtained directly]. APPROXIMATE (see
    APPROXIMATE_METRICS['PF']).

    U = U_lane + U_road + U_car + U_vel. Defaults are Wolf & Burdick's own Table I
    simulation parameters (labeled by them as scenario-specific demonstration values,
    not universal constants -- d0_car is the one exception: it is referenced in their
    text as "max distance at which U_car has influence" but not actually listed in
    Table I, so 50.0m here is this implementation's own choice, not theirs).

    U_car,m(K) = A_car * exp(-alpha*K) / K, a Yukawa potential in a pseudo-distance K to
    the nearest other agent, measured in THAT AGENT'S OWN body frame (forward = its
    heading). K is the nearest-point-to-rectangle distance (rectangle = that agent's own
    length x width) EXCEPT behind the agent, where the paper appends a rearward "wedge"
    (encouraging lane-changing over stopping) via a velocity-dependent longitudinal
    rescaling xi_m(v) = xi0(v)*exp(-beta*(v-v_m)), xi0(v) = d0/(Tf*v) if v>=d0/Tf else 1
    -- APPROXIMATED HERE by nearest-rectangle-distance on the RESCALED body-frame
    position rather than reconstructing the paper's exact triangular wedge polygon
    (which needs figure-level geometric detail beyond the text).

    U_lane, U_road need a road-relative lateral (y) coordinate; this pipeline has no
    road-aligned coordinate frame (WOMD scenarios include curves/intersections, not just
    Wolf & Burdick's straight highway). APPROXIMATED via signed perpendicular distance
    from the ego's position to the nearest lane-centerline / road-edge polyline segment
    (reusing the point-to-segment-distance building block also used by
    _nearest_point_to_rect_distance, applied per-segment) as a stand-in for their
    y-y_c,i / y-y_0,j terms -- a real deviation from the paper's straight-highway
    parameterization, not just a missing constant.

    U_vel = gamma*(v-v_des)*x, x interpreted here as the ego's own cumulative
    along-path distance traveled since episode start (not raw world (x,y), which
    would make the potential depend on absolute map position -- clearly not the
    paper's intent for a "progress" term).
    """
    ex, ey, eh, ev = pack["ego_x_traj"], pack["ego_y_traj"], pack["ego_heading_traj"], pack["speed_traj"]
    ox, oy, oh, ov = (
        pack["other_x_traj"],
        pack["other_y_traj"],
        pack["other_heading_traj"],
        pack["other_speed_traj"],
    )
    n, T = ex.shape

    # --- U_car ---
    other_l = pack.get("other_length_traj")
    other_w = pack.get("other_width_traj")
    u_car = np.zeros((n, T), dtype=np.float64)
    if other_l is not None and other_w is not None:
        dx, dy = ex - ox, ey - oy  # ego relative to other, world frame
        cos_oh, sin_oh = np.cos(oh), np.sin(oh)
        x_body = dx * cos_oh + dy * sin_oh
        y_body = -dx * sin_oh + dy * cos_oh
        v_threshold = d0_car / t_follow
        with np.errstate(divide="ignore", invalid="ignore"):
            xi0 = np.where(ev >= v_threshold, d0_car / np.maximum(t_follow * ev, EPS), 1.0)
        xi_m = xi0 * np.exp(-beta_car * (ev - ov))
        x_body_eff = np.where(x_body < 0, xi_m * x_body, x_body)
        k = _nearest_point_to_rect_distance(x_body_eff, y_body, other_l / 2.0, other_w / 2.0)
        k = np.maximum(k, EPS)
        u_car_valid = a_car * np.exp(-alpha_car * k) / k
        valid = np.isfinite(x_body) & np.isfinite(y_body) & (pack.get("other_id_traj", np.ones((n, T))) >= 0)
        u_car = np.where(valid, u_car_valid, 0.0)

    # --- U_lane, U_road (perpendicular distance to nearest polyline segment) ---
    def _min_perp_dist_to_polylines(
        px_t: np.ndarray, py_t: np.ndarray, x_key: str, y_key: str, len_key: str, sid_key: str
    ) -> np.ndarray:
        out = np.full((n, T), np.nan, dtype=np.float64)
        if len_key not in pack or pack[len_key].size == 0:
            return out
        polylines = _unpack_polylines(pack[x_key], pack[y_key], pack[len_key], pack[sid_key])
        scene_id = pack["scene_id"]
        segs_by_scene: dict[int, np.ndarray] = {}
        for sid, pts in polylines:
            if pts.shape[0] < 2:
                continue
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            segs_by_scene.setdefault(sid, []).append(segs)
        segs_by_scene = {sid: np.concatenate(v, axis=0) for sid, v in segs_by_scene.items()}
        for i in range(n):
            segs = segs_by_scene.get(int(scene_id[i]))
            if segs is None:
                continue
            a = segs[:, 0, :]  # (S,2)
            b = segs[:, 1, :]
            ab = b - a
            ab_len2 = np.maximum((ab**2).sum(axis=-1), EPS)
            for t in range(T):
                p = np.array([px_t[i, t], py_t[i, t]])
                tproj = np.clip(((p - a) * ab).sum(axis=-1) / ab_len2, 0.0, 1.0)
                closest = a + tproj[:, None] * ab
                d = np.hypot(*(p - closest).T)
                out[i, t] = float(d.min())
        return out

    d_lane = _min_perp_dist_to_polylines(
        ex, ey, "lane_polyline_x", "lane_polyline_y", "lane_polyline_lengths", "lane_polyline_scenario_id"
    )
    d_road = _min_perp_dist_to_polylines(
        ex, ey, "road_edge_polyline_x", "road_edge_polyline_y", "road_edge_polyline_lengths",
        "road_edge_polyline_scenario_id",
    )
    sigma = sigma_lane if sigma_lane is not None else 1.2  # 0.3 * Wolf&Burdick's own lane width (4m)
    u_lane = np.where(np.isfinite(d_lane), a_lane * np.exp(-(d_lane**2) / (2 * sigma**2)), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_road = np.where(np.isfinite(d_road), (eta_road / 2.0) / np.maximum(d_road, EPS) ** 2, 0.0)

    # --- U_vel (x = cumulative along-path distance traveled, not raw world x) ---
    step_dist = np.hypot(np.diff(ex, axis=1, prepend=ex[:, :1]), np.diff(ey, axis=1, prepend=ey[:, :1]))
    path_dist = np.cumsum(step_dist, axis=1)
    u_vel = gamma_vel * (ev - v_des) * path_dist

    return u_lane + u_road + u_car + u_vel


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
        out["TTK_min_s_approx"] = np.nanmin(time_to_kickdown_traj(pack), axis=1)
        out["TTR_min_s_approx"] = np.nanmin(time_to_react_approx_traj(pack), axis=1)
        out["PTTC_min_s_approx"] = np.nanmin(potential_ttc_traj(pack), axis=1)
        out["WTTC_min_s_approx"] = np.nanmin(worst_case_ttc_traj(pack), axis=1)
        out["PSD_mean"] = np.nanmean(proportion_of_stopping_distance_traj(pack), axis=1)
        out["CPI_approx"] = crash_potential_index(pack)
        out["DeltaV_proxy_mps_approx"] = delta_v_proxy(pack)
        out["RSS_long_violation_frac_approx"] = np.mean(rss_longitudinal_violation_traj(pack), axis=1)
        out["SP_worst_approx"] = np.nanmin(safety_potential_traj(pack), axis=1)
        out["AGS_accepted_gap_ttc_s_approx"] = accepted_gap_ttc(pack)
        out["CS_conflict_severity_approx"] = conflict_severity_at_onset(pack)

    if has_readout(pack):
        jerk = longitudinal_jerk_proxy(pack)
        out["LongJ_policy_proxy_mean_abs_mps3"] = np.nanmean(np.abs(jerk), axis=1)

    if has_pose(pack):
        ta = time_advantage_traj(pack)
        out["TA_PrET_min_s"] = np.nanmin(ta, axis=1)
        sprEt = scaled_predictive_encroachment_time_traj(pack)
        out["SPrET_min_s2"] = np.nanmin(sprEt, axis=1)
        rss_full = rss_full_violation_traj(pack)
        out["RSS_full_violation_frac_approx"] = np.mean(rss_full, axis=1)

    if has_pose(pack) and has_crosswalks(pack):
        ttz = time_to_zebra_traj(pack)
        out["TTZ_min_s_approx"] = np.nanmin(ttz, axis=1)

    if has_pose(pack) and has_dimensions(pack):
        et_scenario, pet_scenario = encroachment_times_traj(pack)
        out["ET_s_approx"] = et_scenario
        out["PET_s_approx"] = pet_scenario

    if has_pose(pack) and "other_length_traj" in pack:
        # potential_functions_traj degrades gracefully via .get()/None-checks if width
        # is absent (u_car contributes 0 rather than crashing) -- unlike ET/PET and TTS
        # below, which index length/width directly and need has_dimensions' full check.
        pf = potential_functions_traj(pack)
        out["PF_mean_approx"] = np.nanmean(pf, axis=1)
        out["PF_worst_approx"] = np.nanmax(pf, axis=1)

    if has_pose(pack) and has_width(pack):
        a_lat = required_lat_accel_traj(pack)
        out["a_lat_req_worst_mps2_approx"] = np.nanmax(a_lat, axis=1)
        stn = steer_threat_number_traj(pack)
        out["STN_max_approx"] = np.nanmax(stn, axis=1)
        a_comb = combined_required_accel_traj(pack)
        out["a_req_combined_worst_mps2_approx"] = np.nanmax(a_comb, axis=1)

    if has_pose(pack) and has_dimensions(pack):
        tts = time_to_steer_traj(pack)
        # nanmin naturally preserves -inf (genuinely "no steer maneuver, however early,
        # avoids collision") as the most-critical value, while skipping NaN (undefined,
        # e.g. no TTC) -- this is the correct worst-case aggregate, not a filtering bug.
        out["TTS_min_s_approx"] = np.nanmin(tts, axis=1)
        # Fuller TTR = max(TTB, TTK, TTS) -- see time_to_react_approx_traj's docstring for
        # why the cheap TTR_min_s_approx above (computed earlier, traj-only) omits TTS.
        # np.fmax, not np.maximum -- same NaN-propagation reasoning as
        # time_to_react_approx_traj above (TTK and/or TTS are frequently NaN here too).
        ttr_full = np.fmax(time_to_brake_traj(pack), np.fmax(time_to_kickdown_traj(pack), tts))
        out["TTR_full_min_s_approx"] = np.nanmin(ttr_full, axis=1)

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
    "PSD": "Astarita, Guido, Vitale, Giofre (2012) confirm, from the original paper directly, that for car-following/rear-end interactions (this pipeline's scenario type) the 'conflict point' p_CA is defined as the lead vehicle's own position -- exactly min_dist_traj -- not an independently-defined intersection polygon, so no substitution/approximation is needed here.",
    "TTB": "Hillenbrand (2007) KIT dissertation, read directly, confirms the 1D/no-lateral-intersection case (this pipeline's setup) is pure longitudinal kinematics using the EGO's OWN max-braking bound -- exactly this env's own A_MAX action-space bound, not an assumed external constant.",
    "TA/PrET/SPrET": "unlocked by capture_pose (verified end-to-end against a real compiled rollout, not just synthetic data): with ego and nearest-partner (x, y, heading, speed) now captured, Neurohr et al. (2021)'s constant-velocity path-intersection solve is a direct 2x2 linear system -- exactly what the formula requires, no substitution needed. Undefined (NaN) when paths are parallel or the crossing lies in either agent's past, per the formula's own definition.",
}

APPROXIMATE_METRICS: dict[str, str] = {
    "DST (ts>0)": "other agent's longitudinal speed v2 is approximated as ego_speed - closing_speed (assumes the closing-speed axis is approximately the longitudinal/car-following axis). No paper found (Hupfer 1997's original PDF was located but is an unOCR'd scan) recommends a standard non-zero ts value either, so ts remains a caller-supplied choice.",
    "WTTC": "Wachenfeld et al. (2016)'s original is paywalled, but its cited open-source reference implementation (CommonRoad-CriMe) reveals the true formula applies the SAME a_max independently to both vehicles and grows an isotropic 2D disc (from vehicle footprint radii) at combined rate (a1+a2). Under a symmetric-vehicle assumption this validates '2x this env's a_max' as a real consequence of the original's own structure (not an arbitrary multiplier) -- but the isotropic-2D-to-1D-closing-direction collapse and the disc-to-point-vehicle simplification (no footprint radii available) are real, additional departures from the original geometry.",
    "CPI": "Cunto's 2008 PhD thesis (the primary source, read directly) gives the actual normal-distribution parameters for Maximum Available Deceleration Rate (mean=8.45, std=1.40 m/s^2, cars) used here; this is now much closer to the original formula (P(MADR<=DRAC) via the normal CDF) than a prior deterministic-threshold version of this metric, but MADR is still a fleet-wide human-vehicle distribution being applied to an RL policy operating under a fixed, narrower discrete action space, which is itself an approximation of what CPI is meant to model.",
    "Delta-v": "assumes equal vehicle mass (Shelby 2011's own convention when masses are unavailable, read directly from the original -- not an arbitrary guess) and uses the closing speed at the timestep of minimum distance as a stand-in for pre-impact relative velocity (Shelby 2011 does exactly this substitution -- 'predicted-collision relative velocity' -- when extending Delta-v to non-collision conflicts, confirmed from his original text), not a physically measured post-collision speed change.",
    "RSS_long_violation": "scalar-only fallback (works on any pack, even without capture_pose) that implements just the longitudinal half of RSS-DS's simultaneous lateral+longitudinal violation test, and blindly assumes ego=rear/following (it has no real position data to determine true ahead/behind). SUPERSEDED by RSS_full when pose is available (see below) -- kept only for packs from run_coordination.sh's scalar-only rollouts. Uses field-standard constants from Intel's ad-rss-lib reference implementation (rho=1.0s, a_max_accel=3.5, a_min_brake=4.0, a_max_brake=8.0) since the original paper deliberately declines to publish numeric values.",
    "RSS_full": "unlocked by capture_pose (verified end-to-end against a real compiled rollout): now correctly determines true front/rear roles from the measured sign of the longitudinal offset (rather than always assuming ego=rear), and evaluates the lateral axis too (Definition 6/Lemma 4) via projection onto ego's heading-perpendicular direction, using ego's own lateral velocity in its own frame as 0. Passed hand-computable sanity tests (stationary-adjacent-lane no-violation, closing-fast tiny-gap violation, etc.) but is flagged APPROXIMATE rather than EXACT because the lateral formula's v_i +/- rho*a_lat sign convention (which of the two cars is the 'positive-side' one) is this implementation's own resolution of an ambiguity in the source notation, not verified against Intel's ad-rss-lib or another reference implementation. Same field-standard longitudinal constants as RSS_long_violation, plus a_max_accel_lat=0.2, a_min_brake_lat=0.8, mu=0.1m for the lateral axis (also Intel ad-rss-lib defaults).",
    "LongJ": "UPGRADED: rollout.py now persists accel_traj, the actually-sampled/executed per-step acceleration -- this was already computed every step (used only for mean_hard_brake) but discarded before this fix. For dynamics_model='classic' (this pipeline's default) this is the exact physically-applied longitudinal acceleration (confirmed by reading the C integration step directly: `signed_speed += acceleration*dt` uses this exact same decoded value), so LongJ computed from it is EXACT, not approximate. Still listed here (not in EXACT_METRICS) because packs saved before this fix -- or using dynamics_model='jerk', where the C side's own jerk_long field would be the right source instead, not currently exposed -- only have exp_accel_traj (the policy's softmax-expected acceleration), for which LongJ remains a decision-smoothness proxy, not physically realized jerk.",
    "P-MC": "RELABELED from an earlier version of this metric: what's actually computed (empirical fraction of training-seed replicates that collide, same method/map) estimates policy-checkpoint outcome variance, confirmed (via Broadhurst's own CMU-RI-TR-04-11 precursor report) to be a DIFFERENT random variable than the original paper's per-instant control-input-uncertainty integral, not merely a looser version of it. See monte_carlo_collision_probability()'s docstring for the full distinction and for what a truer one-step approximation would need (full per-bin action probabilities, not currently persisted).",
    "PTTC": "Wakabayashi et al. (2003), read directly (original Japanese text via J-STAGE), reveals the original paper ALSO never isolates the other agent's true acceleration -- it substitutes a preset constant from three empirically-measured braking-severity classes (0.93/2.78/5.56 m/s^2). This pipeline follows the same substitution (default: this env's own A_MAX, which falls between Wakabayashi's medium/hard presets), so the approximation directly mirrors the original method rather than deviating from it.",
    "TTK": "UPGRADED: Hillenbrand (2007) confirms the underlying 1D kinematics are identical to TTB, and the direction ambiguity (kickdown only helps when the interacting agent is behind, not ahead) is now RESOLVED when pose is available -- time_to_kickdown_traj projects the other agent's relative position onto the ego's own heading (same technique as RSS_full/a_lat,req) and returns NaN whenever the other agent is genuinely ahead, rather than reporting an unverified value. Hand-tested (other-ahead -> NaN, other-behind -> finite, no-pose -> old unconditional fallback). On the one real episode tested, every interaction happened to have the other agent ahead, so TTK came out entirely NaN for it -- a more honest result than the previous version's misleading finite value for the same data. Still listed as approximate (not exact) because packs without capture_pose fall back to the old direction-unverified computation.",
    "TTR": "compute_all_metrics reports TWO variants. 'TTR_min_s_approx' (cheap, always available on any trajectory pack) = max(TTB, TTK) only, omitting TTS since it's far more expensive to compute (a forward-simulation search, not a closed-form solve) -- provably an underestimate of the true TTR (dropping a max term can only lower it), the safe-biased direction to be wrong in. 'TTR_full_min_s_approx' (only when pose+width are available) = max(TTB, TTK, TTS), the fuller value. Earlier text here claimed TTS was 'not applicable' -- that was true when this note was first written but is now stale; TTS has since been implemented (see its own entry above) and TTR_full uses it. BUG FOUND AND FIXED during verification: both variants originally used np.maximum, which PROPAGATES NaN through the whole max -- since TTK is frequently and legitimately NaN (the direction-gating fix means 'not a valid maneuver here', not 'unknown'), this silently made TTR NaN even when TTB was a perfectly good answer, confirmed on real captured data (TTK all-NaN for one episode's ego made TTR_min_s_approx report NaN instead of falling back to TTB). Switched to np.fmax, which correctly ignores a NaN operand; re-verified TTR_min_s_approx now equals TTB_min_s exactly wherever TTK is NaN.",
    "AGS": "reframed, per Alhajyaseen et al. (2013)'s own univariate cumulative-Weibull gap-acceptance model (confirmed from the original: the function itself takes only gap TIME as input, not the demographic covariates the survey's abstraction implies are required), as a purely temporal quantity -- the realized TTC at the last pre-tight step. A full acceptance-PROBABILITY curve additionally needs alpha/beta parameters fitted to THIS domain; Alhajyaseen's own fitted values are for a pedestrian left-turn scenario and are not reused here as defaults.",
    "CS (Conflict Severity)": "Bagdadi (2013)'s original text is paywalled and unreachable, but the survey's 'not run-time capable' framing was re-examined: the actual blocker is a PRECONDITION (needing an identified evasive-maneuver onset), not a look-ahead requirement, and this pipeline already detects such an onset causally via tight_onset (paralleling how the Swedish Traffic Conflicts Technique manual itself defines its analogous severity moment). Computed the instant tight_onset fires, using the equal-mass convention and a one-step finite-difference of speed_traj for the ego's realized deceleration (no mass data, and Bagdadi's own numeric CS target value is for his own naturalistic-driving population, not transferable here as a hard threshold).",
    "SP (Safety Potential / SFF)": "implements the NVIDIA whitepaper's own explicitly-labeled single-control-policy 'Implementation Example' (read directly, both the full whitepaper and its plain-language companion doc), not the full general n-actor/arbitrary-control-policy SFF theory (which needs footprints/claimed-set geometry this pipeline doesn't have). t_int is approximated by this pipeline's own TTC (a faithful substitution specifically for the paper's 1D worked case, confirmed against its prose description of that case) and v_other by the same ego_speed-minus-closing-speed approximation used elsewhere; a_min is an implementer's choice by the source's own design (the paper publishes no numeric value), defaulted to this env's own A_MAX for consistency with BTN/DST/TTB.",
    "a_lat,req": "unlocked by capture_pose + capture_map_geometry's incidental partner-width plumbing (both agents' widths verified against a real compiled rollout, ~2.0-2.3m, realistic): Jansson (2005)'s constant-acceleration formula (Eq. 5.46-5.49) implemented directly, using the same heading-perpendicular lateral decomposition already built for RSS_full. UPGRADED: v1,lat (ego's own lateral velocity in its own frame) is now finite-differenced from the ego's own (100%-covered) position trajectory rather than assumed 0 -- confirmed via a straight-line-motion test (recovers the same value as the old v1,lat=0 assumption) and a lateral-drift test (produces a genuinely different, larger value, confirming the signal is live). a2,lat (the other agent's lateral acceleration) remains assumed 0 -- the other agent's position coverage (~1/3 of steps) is too sparse to finite-difference as reliably as ego's. Hand-computed sanity test (head-on, zero lateral offset, known TTC and widths) matched the closed-form expectation exactly. STN = a_lat,req / 7.0 m/s^2 (Jansson's own deployed-demonstrator lateral bound).",
    "a_req (combined norm)": "= sqrt(a_long_req^2 + a_lat_req^2), now computable since both terms are (see a_lat,req above). Per the survey's own attributed (simplified) combination -- Jansson's thesis has two OTHER combination formulas (a plain min() and a coupled friction-ellipse solve) that are not implemented; this is the survey's reading of Jansson, not a literal transcription of his joint-optimal (A_x,A_y) solve (Eq. 5.59-5.61).",
    "TTS": "Hillenbrand (2007)'s own approach is a genuine 2D circular-arc turning-radius model solved by nested interval bisection -- not reproduced faithfully. Instead follows the architecturally simpler approach found in the actively-maintained CommonRoad-CriMe reference toolbox's actual solver code (confirmed by reading commonroad_crime/measure/time/tts.py directly): single-level bisection over the maneuver-start offset tau, forward-simulating a constant-lateral-acceleration point-mass steer maneuver against the other agent's constant-velocity path, checked via single-disc (circumscribing-circle) collision rather than CommonRoad-CriMe's own 3-disc chain or Hillenbrand's exact arc geometry. Hand-verified: sane finite TTS for a head-on scenario, correctly -inf when even immediate steering can't avoid collision (verified with deliberately very-wide vehicles), correctly NaN when TTC is undefined, and a clean monotonic trend (narrower vehicles -> larger TTS) across a parameter sweep. TTM's m='steer' case is exactly this. Substantially more expensive to compute than every other metric in this module (a discretized 2D forward-simulation search per (ego, timestep) pair, ~1.85s for a single 91-step episode in testing) -- n_tau/horizon_s/dt_grid are exposed as tunable resolution/runtime knobs.",
    "ET/PET": "unlocked by capture_pose's ego+other position/heading and other_length_traj/ego_length/ego_width/other_width_traj (real dimensions for both agents, both axes). Operationalizes the conflict area (CA) as each agent's own ORIENTED RECTANGLE (length x width, at its own per-step heading) sweeping through the predicted path-crossing point -- itself computed by reusing the already-validated constant-velocity crossing solve from TA/PrET (_crossing_point_traj), anchored at the timestep within each other_id-stable window whose predicted crossing time is smallest-but-still-nonnegative (closest to, but not after, the event -- the forward-only t>=0 constraint means the raw prediction goes undefined once a crossing has passed, which an earlier draft of this function got wrong by anchoring on the window's last timestep instead). UPGRADED from an earlier version that used a circular disc of radius=length/2 (leaving width unused despite being captured) to a proper oriented-rectangle point-containment test -- closer to Laureshyn et al.'s own practical rectangular-footprint refinement of Allen et al.'s definition. Re-verified after the upgrade with the same 3 hand-computed synthetic cases: simultaneous arrival (ET well-defined per agent, PET correctly undefined per Allen et al.'s own assumption -- rectangle ET came out larger than the old disc version, 0.4s vs 0.2s, which is the physically expected direction: a car's own length dominates dwell time when driving straight through a point, which a disc of radius=length/2 under-counts relative to a length-2 rectangle), staggered arrival (matches closed-form expectations), and non-crossing parallel paths (NaN). Scenario-level (one value per ego, not per-step), reporting the most-critical (smallest) ET/PET across all stable-id windows if an ego has more than one. On the one real map available for testing, this pipeline's captured episode happened to produce NaN (no genuine path-crossing detected in that specific rollout, plausible for an interaction dominated by car-following/adjacent-lane geometry rather than intersection-crossing) -- not evidence of a bug, but a reminder that ET/PET's finite-rate will depend heavily on how many genuine crossing conflicts a given map/policy combination produces.",
    "PF (Potential Functions)": "Wolf & Burdick (2008) -- note: 2008, not 2018 as an earlier pass at this registry stated, a citekey typo traced and corrected -- full text obtained on a second attempt (a different, working Caltech repository record for the same paper). Confirmed the vehicle-avoidance potential (U_car) IS velocity-dependent, not distance-only: a Yukawa potential A_car*exp(-alpha*K)/K in a pseudo-distance K, with a rearward 'wedge' behind the obstacle rescaled by xi_m(v)=xi0(v)*exp(-beta*(v-v_other)) -- implemented here using nearest-point-to-rectangle distance (rectangle = the other agent's real length x width) instead of reconstructing the paper's exact triangular wedge polygon (needs figure-level detail beyond the text), and Wolf & Burdick's own Table I values as defaults EXCEPT d0 (referenced in their text as 'max distance at which U_car has influence' but never actually listed in Table I -- 50.0m here is this implementation's own choice). U_lane/U_road need a road-relative lateral coordinate the paper assumes (straight highway); this pipeline's WOMD scenarios include curves/intersections, so these are approximated via signed perpendicular distance to the nearest lane-centerline/road-edge polyline segment instead -- a real deviation from the paper's coordinate frame, not just a missing constant. U_vel's 'x' term is interpreted as cumulative along-path distance since episode start (not raw world (x,y), which would make potential depend on absolute map position). CAVEAT discovered during testing: with Wolf & Burdick's own beta=0.6, the wedge rescaling saturates extremely fast for agents with even moderately different speeds (e.g. a 15 m/s speed differential over 50m already pins K at its numerical floor) -- verified this is the formula's actual documented behavior, not an implementation bug, but it means PF may report near-identical extreme values across many differing scenarios unless retuned for this pipeline's speed range. Also worth noting: the Westhofen survey's own authors, in their worked example, independently excluded PF (and SP) as insufficiently validated even with full geometric data available.",
    "TTZ": "unlocked by capture_map_geometry's new crosswalk-polyline accessor (verified against a real compiled map: 5 crosswalk polylines / 20 points, coordinate frame confirmed against the existing road-edge getter). Solved as ray-vs-polyline-segment intersection under constant-velocity extrapolation (ego's current heading+speed) -- hand-verified with straight-ahead, behind, missed-to-the-side, and diagonal-approach test cases, all matching closed-form expectations exactly. Approximate rather than exact because crosswalks are treated as boundary polylines (not the paper's abstract 'position') and the constant-velocity model is the same simplification used elsewhere (TA/PrET, TTB, ...) rather than a full trajectory predictor. Still genuinely vehicle-only: this pipeline has no VRU/pedestrian-presence signal, so TTZ here answers 'when would the ego geometrically reach the crosswalk', not 'is a pedestrian there' -- pair with an external pedestrian-presence source if you need the paper's full VRU-conflict framing.",
}

NOT_APPLICABLE: dict[str, str] = {
    "CI (Conflict Index)": "built on top of PET (now available; see APPROXIMATE_METRICS) plus vehicle masses and approach headings at CA entry/exit (masses still unavailable, same gap as Delta-v/CS). Even granting those, the standalone probability-proxy term e^(-beta*PET) still needs the calibration constant beta -- confirmed a genuine dead end on a second, deeper pass: the Procedia Computer Science version of Alhajyaseen's paper is legitimately gold-OA per Unpaywall but every automated fetch is Cloudflare-bot-walled (not a subscription paywall -- a human browser would likely succeed where this couldn't); the Arab J Sci Eng version has no OA copy anywhere; no citing paper in the broader conflict-severity-index literature reports an example beta either. So CI stays blocked by a genuinely missing constant with no accessible route to it, not by missing PET anymore.",
    "PRI": "confirmed genuinely and completely pedestrian/crosswalk-specific from the original paper (Cafiso et al. 2011, read in full) -- needs the pedestrian's own position/walking-speed trajectory toward a defined crosswalk conflict area, with no vehicle-vehicle analogue anywhere in the source.",
    "TCI": "Junietz's 2019 dissertation (read directly, the fullest available source), confirms TCI is computed via constrained trajectory (MPC) optimization needing world-frame (x,y)/heading vehicle state and a genuine lane-relative lateral offset d_lat distinct from the longitudinal gap. The author explicitly states 'there is no additional longitudinal component' and declines to define a degenerate/longitudinal-only fallback himself.",
    "ACI": "Kuang et al.'s original AAP 2015 paper is paywalled, but a companion paper by the same two lead authors (Kuang & Qu 2015, EPPM conference, openly hosted) restates the FULL top-level aggregation math -- the probability tree's combination formula (their eq. 1), the ACI sum (eq. 2), and a time-averaging extension (eq. 3) not even in the Westhofen survey -- plus confirms MADR is drawn from a truncated normal (AASHTO 2004 / Cunto & Saccomanno 2008) rather than a plain normal. What remains genuinely locked behind the paywall: the companion paper explicitly references 'Table 1' for what each of the 4 condition levels / 8 leaf nodes physically means (e.g. which node represents 'lead vehicle brakes and follower fails to react in time'), but does not reprint that table -- so the aggregation MATH is now known, but the domain-specific TREE STRUCTURE is not, and fabricating plausible-sounding leaf-node semantics would misrepresent the metric rather than approximate it. Would need either primary-paper access (confirmed genuinely unreachable via Unpaywall/ResearchGate/institutional repos/citing-paper search) or a caller willing to supply their own tree structure to a generic N-level implementation -- not attempted here since guessing the semantics defeats the purpose of citing Kuang et al. specifically. Reaction time (if you build your own tree): LogNormal(mean=0.92s, SD=0.28s), Triggs & Harris (1982), confirmed used throughout this sub-literature.",
    "SOI": "Ogawa (2007) and Johnsson et al. (2018), both read in full, confirm the 'personal space' buffer is an oriented rectangle along the direction of travel -- the ORIENTATION half of this is no longer a blocker now that capture_pose provides heading for both agents, but the SIZE half still is: the only published numeric areas in the accessible literature are for pedestrians (5.0 m^2) and bicycles (12.8 m^2), not cars. Fabricating a car-scale personal-space area (e.g. from vehicle footprint * some multiplier) would not be reusing a literature constant, it would be inventing one, so this stays not-applicable rather than shipping an ungrounded threshold.",
    "P-SMH": "Sanchez Morales et al. (2019), read in full, confirms algebraically that a degenerate N=M=1 hypothesis-per-side instantiation collapses the formula to exactly the binary collision indicator (AM), not a meaningful trivial P-SMH -- the metric's entire value-add is the weighted sum over a NONTRIVIAL (N,M>1) hypothesis set, which needs full trajectory generation (two-track ego model, one-track other-agent model, lane-topology-dependent scoring penalties) this pipeline doesn't produce.",
    "P-SRS": "Althoff et al. (2009)'s original PDF is bot-blocked on every mirror, but the survey's formula-level (not just prose) paraphrase, cross-checked against independent secondary sources, confirms the method needs offline-precomputed Markov-chain reachability tables over a discretized, ROAD-RELATIVE position x velocity partition -- genuinely requiring lane/road geometry this pipeline doesn't have, not just an online-computation shortcut.",
    "LatJ": "needs an EXECUTED lateral-acceleration time series to differentiate into jerk -- distinct from a_lat,req (now available; see APPROXIMATE_METRICS), which is a 'required to avoid collision' threat quantity, not what the vehicle actually did. common.py's action_stats_from_logits() DOES compute a 'steer'/'steer_mag' signal from the policy logits, but rollout.py's readout_out capture map only pulls accel/p_brake/p_yield/gap_press/entropy into the saved pack -- steer was never persisted. Adding a 'steer': 'steer' entry to that dict (and re-running run_ego_readout.sh) would unlock this; not done here since it requires re-collecting rollout data.",
}