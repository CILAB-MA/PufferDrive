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
    APPROXIMATE (see APPROXIMATE_METRICS['TTK']).

    Algebraically identical in form to TTB (TTK = TTC - closing/a_max, using the
    accelerate-away bound in place of the brake bound -- the same magnitude A_MAX here,
    since this env's action space is symmetric). The reason this is APPROXIMATE rather
    than EXACT despite the dissertation confirming the 1D math: kickdown only helps when
    accelerating REDUCES the closing rate (e.g. the ego is racing to clear a crossing
    point, or outrunning something approaching from behind) -- it is actively
    counterproductive if the interacting agent is a lead vehicle ahead (accelerating would
    INCREASE the closing rate there). The pack's closing_traj sign convention does not
    distinguish these two geometric configurations, so TTK values should be treated as a
    conditional "if kickdown is even the right maneuver here" quantity, not a
    universally-valid time margin the way TTB is.
    """
    ttc = pack["ttc_traj"]
    cl = pack["closing_traj"]
    kick_time = np.where(cl > 0, cl / a_max, np.nan)
    return ttc - kick_time


def time_to_react_approx_traj(pack: dict[str, np.ndarray], *, a_max: float = A_MAX) -> np.ndarray:
    """TTR -- Time To React [Hillenbrand 2007 dissertation eq. 5.16: TTR ~= max(TTB, TTS,
    TTK); Tamke, Dang, Breuel 2011 generalized the same max-over-maneuvers structure].
    APPROXIMATE, conservative (see APPROXIMATE_METRICS['TTR']).

    TTS (the steer-maneuver term) is not computable here (needs vehicle widths + minimum
    turning radius -- see NOT_APPLICABLE). TTR_approx = max(TTB, TTK) omits it. Since
    dropping a term from a max can only reduce the result, TTR_approx <= TTR_true always
    -- a systematic *underestimate* of how much time remains, i.e. it biases toward
    treating situations as more urgent than they truly are, which is the safe direction
    for a criticality metric to be wrong in.
    """
    return np.maximum(time_to_brake_traj(pack, a_max=a_max), time_to_kickdown_traj(pack, a_max=a_max))


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
    capture issue documented for the steer signal (see NOT_APPLICABLE['a_lat,req...']).

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
}

APPROXIMATE_METRICS: dict[str, str] = {
    "DST (ts>0)": "other agent's longitudinal speed v2 is approximated as ego_speed - closing_speed (assumes the closing-speed axis is approximately the longitudinal/car-following axis). No paper found (Hupfer 1997's original PDF was located but is an unOCR'd scan) recommends a standard non-zero ts value either, so ts remains a caller-supplied choice.",
    "WTTC": "Wachenfeld et al. (2016)'s original is paywalled, but its cited open-source reference implementation (CommonRoad-CriMe) reveals the true formula applies the SAME a_max independently to both vehicles and grows an isotropic 2D disc (from vehicle footprint radii) at combined rate (a1+a2). Under a symmetric-vehicle assumption this validates '2x this env's a_max' as a real consequence of the original's own structure (not an arbitrary multiplier) -- but the isotropic-2D-to-1D-closing-direction collapse and the disc-to-point-vehicle simplification (no footprint radii available) are real, additional departures from the original geometry.",
    "CPI": "Cunto's 2008 PhD thesis (the primary source, read directly) gives the actual normal-distribution parameters for Maximum Available Deceleration Rate (mean=8.45, std=1.40 m/s^2, cars) used here; this is now much closer to the original formula (P(MADR<=DRAC) via the normal CDF) than a prior deterministic-threshold version of this metric, but MADR is still a fleet-wide human-vehicle distribution being applied to an RL policy operating under a fixed, narrower discrete action space, which is itself an approximation of what CPI is meant to model.",
    "Delta-v": "assumes equal vehicle mass (Shelby 2011's own convention when masses are unavailable, read directly from the original -- not an arbitrary guess) and uses the closing speed at the timestep of minimum distance as a stand-in for pre-impact relative velocity (Shelby 2011 does exactly this substitution -- 'predicted-collision relative velocity' -- when extending Delta-v to non-collision conflicts, confirmed from his original text), not a physically measured post-collision speed change.",
    "RSS_long_violation": "implements only the longitudinal half of RSS-DS's simultaneous lateral+longitudinal violation test; confirmed directly from Shalev-Shwartz et al. (2017) that the lateral formula needs an independently-decomposed lateral velocity component that cannot be recovered from a single radial closing-speed scalar without heading/bearing information (not a missing-assumption problem, a missing-input-category problem). Uses field-standard constants from Intel's ad-rss-lib reference implementation (rho=1.0s, a_max_accel=3.5, a_min_brake=4.0, a_max_brake=8.0) since the original paper deliberately declines to publish numeric values. Over-flags relative to the true RSS-DS (longitudinal violation alone is a necessary but not sufficient condition for the full metric).",
    "LongJ": "computed on the policy's *expected* acceleration (softmax-weighted mean over the discrete action distribution) since the actually-sampled/executed per-step acceleration isn't persisted in the packs -- a policy-smoothness proxy, not physically realized vehicle jerk.",
    "P-MC": "RELABELED from an earlier version of this metric: what's actually computed (empirical fraction of training-seed replicates that collide, same method/map) estimates policy-checkpoint outcome variance, confirmed (via Broadhurst's own CMU-RI-TR-04-11 precursor report) to be a DIFFERENT random variable than the original paper's per-instant control-input-uncertainty integral, not merely a looser version of it. See monte_carlo_collision_probability()'s docstring for the full distinction and for what a truer one-step approximation would need (full per-bin action probabilities, not currently persisted).",
    "PTTC": "Wakabayashi et al. (2003), read directly (original Japanese text via J-STAGE), reveals the original paper ALSO never isolates the other agent's true acceleration -- it substitutes a preset constant from three empirically-measured braking-severity classes (0.93/2.78/5.56 m/s^2). This pipeline follows the same substitution (default: this env's own A_MAX, which falls between Wakabayashi's medium/hard presets), so the approximation directly mirrors the original method rather than deviating from it.",
    "TTK": "Hillenbrand (2007) confirms the underlying 1D kinematics are identical to TTB, but the SIGN/DIRECTION of when kickdown actually helps (only when accelerating reduces the closing rate, e.g. escaping a rear threat or a crossing point -- actively wrong if the interacting agent is a lead vehicle ahead) cannot be determined from this pipeline's direction-agnostic closing-speed scalar, so TTK values should be read as conditional on a maneuver-applicability assumption this pipeline cannot verify.",
    "TTR": "= max(TTB, TTK) only (the true TTR = max(TTB, TTS, TTK) also includes TTS, which is not applicable here -- see NOT_APPLICABLE). Provably a systematic underestimate of the true TTR (dropping a term from a max can only lower it), which is the safe-biased direction for a criticality metric to be wrong in.",
    "AGS": "reframed, per Alhajyaseen et al. (2013)'s own univariate cumulative-Weibull gap-acceptance model (confirmed from the original: the function itself takes only gap TIME as input, not the demographic covariates the survey's abstraction implies are required), as a purely temporal quantity -- the realized TTC at the last pre-tight step. A full acceptance-PROBABILITY curve additionally needs alpha/beta parameters fitted to THIS domain; Alhajyaseen's own fitted values are for a pedestrian left-turn scenario and are not reused here as defaults.",
    "CS (Conflict Severity)": "Bagdadi (2013)'s original text is paywalled and unreachable, but the survey's 'not run-time capable' framing was re-examined: the actual blocker is a PRECONDITION (needing an identified evasive-maneuver onset), not a look-ahead requirement, and this pipeline already detects such an onset causally via tight_onset (paralleling how the Swedish Traffic Conflicts Technique manual itself defines its analogous severity moment). Computed the instant tight_onset fires, using the equal-mass convention and a one-step finite-difference of speed_traj for the ego's realized deceleration (no mass data, and Bagdadi's own numeric CS target value is for his own naturalistic-driving population, not transferable here as a hard threshold).",
    "SP (Safety Potential / SFF)": "implements the NVIDIA whitepaper's own explicitly-labeled single-control-policy 'Implementation Example' (read directly, both the full whitepaper and its plain-language companion doc), not the full general n-actor/arbitrary-control-policy SFF theory (which needs footprints/claimed-set geometry this pipeline doesn't have). t_int is approximated by this pipeline's own TTC (a faithful substitution specifically for the paper's 1D worked case, confirmed against its prose description of that case) and v_other by the same ego_speed-minus-closing-speed approximation used elsewhere; a_min is an implementer's choice by the source's own design (the paper publishes no numeric value), defaulted to this env's own A_MAX for consistency with BTN/DST/TTB.",
}

NOT_APPLICABLE: dict[str, str] = {
    "ET": "needs a defined conflict area (CA) -- a lane/road-geometry construct not present in the saved packs. Confirmed via the CommonRoad-CriMe reference implementation (which computes ET from lanelet-polygon intersections and full vehicle footprints): no formulation found, in the original literature or any practical implementation, that drops the CA/footprint requirement.",
    "PET": "same CA requirement as ET, plus needs the *other* agent's own entry/exit times for that CA, i.e. its full (x,y) trajectory and footprint dimensions -- confirmed via Laureshyn et al.'s own practical/video-based PET method (which replaces the *lane-defined* CA with a *footprint-derived* one, but still fundamentally needs both agents' trajectories and rectangular dimensions, neither of which this pipeline captures).",
    "PrET / SPrET / TA": "Neurohr et al. (2021), read in full, confirms all three reduce to 'find where two predicted straight-line 2D paths cross', needing each agent's absolute (x,y) position AND individual velocity vector/heading in isolation. This pipeline's 'closing speed' is a scalar projection onto the line-of-sight, not a recoverable 2D vector -- there is no assumption that lets a single relative-projection scalar stand in for two independent heading/velocity vectors.",
    "TTZ": "pedestrian-crossing-specific; needs a crosswalk/zebra position and the pipeline carries no VRU/crosswalk semantics.",
    "CI (Conflict Index)": "built on top of PET, which this pipeline does not have (TTC is a closing-rate projection, not a footprint-clearance timing measure -- the two are not interchangeable), plus vehicle masses and approach headings at CA entry/exit. Even the standalone probability-proxy term e^(-beta*PET) cannot be evaluated without PET itself, independent of the missing calibration constant beta (confirmed: beta is described everywhere in the literature as site-calibrated, with no universal published value either).",
    "PRI": "confirmed genuinely and completely pedestrian/crosswalk-specific from the original paper (Cafiso et al. 2011, read in full) -- needs the pedestrian's own position/walking-speed trajectory toward a defined crosswalk conflict area, with no vehicle-vehicle analogue anywhere in the source.",
    "TCI": "Junietz's 2019 dissertation (read directly, the fullest available source), confirms TCI is computed via constrained trajectory (MPC) optimization needing world-frame (x,y)/heading vehicle state and a genuine lane-relative lateral offset d_lat distinct from the longitudinal gap. The author explicitly states 'there is no additional longitudinal component' and declines to define a degenerate/longitudinal-only fallback himself.",
    "ACI": "Kuang et al. (2015) is paywalled and unreachable in full text. A plausible standard reaction-time constant for one branch of its causal tree was located (lognormal, mean=0.92s, SD=0.28s, Triggs & Harris 1982, used throughout this sub-literature) but is unconfirmed against Kuang's own text, and the tree still needs the LEAD vehicle's kinematics modeled independently of the follower (this pipeline only has the relative/closing projection) plus the following vehicle's own braking-capacity distribution and the tree's full ~8-branch conditional structure, none of which are recoverable regardless of the reaction-time constant.",
    "SOI": "Ogawa (2007) and Johnsson et al. (2018), both read in full, confirm the 'personal space' buffer is (a) an oriented rectangle along the direction of travel, needing heading (excluded from this data budget), and (b) only has published numeric areas for pedestrians (5.0 m^2) and bicycles (12.8 m^2) in the accessible literature -- no car-scale constant exists to substitute even if the orientation requirement were relaxed to an isotropic circle.",
    "PF (Potential Functions)": "Wolf & Burdick (2008) -- note: 2008, not 2018 as an earlier pass at this registry stated, a citekey typo traced and corrected -- could not be read in full text (403 from every mirror including the open Caltech repository). Abstract-level evidence across independent secondary sources indicates the vehicle-avoidance potential term depends on relative velocity and surrounding traffic context, not distance alone, undermining a hoped-for pure-distance-decay simplification; the lane-marking/road-geometry potential terms remain unconditionally unavailable regardless. Also worth noting: the Westhofen survey's own authors, in their worked example, independently excluded PF (and SP) as insufficiently validated even with full geometric data available.",
    "P-SMH": "Sanchez Morales et al. (2019), read in full, confirms algebraically that a degenerate N=M=1 hypothesis-per-side instantiation collapses the formula to exactly the binary collision indicator (AM), not a meaningful trivial P-SMH -- the metric's entire value-add is the weighted sum over a NONTRIVIAL (N,M>1) hypothesis set, which needs full trajectory generation (two-track ego model, one-track other-agent model, lane-topology-dependent scoring penalties) this pipeline doesn't produce.",
    "P-SRS": "Althoff et al. (2009)'s original PDF is bot-blocked on every mirror, but the survey's formula-level (not just prose) paraphrase, cross-checked against independent secondary sources, confirms the method needs offline-precomputed Markov-chain reachability tables over a discretized, ROAD-RELATIVE position x velocity partition -- genuinely requiring lane/road geometry this pipeline doesn't have, not just an online-computation shortcut.",
    "a_lat,req / STN / LatJ": "Jansson (2005), read in full, confirms the required-lateral-acceleration formula (Eq. 5.46-5.49) intrinsically needs both vehicles' WIDTHS (the lateral clearance the maneuver must achieve is a function of vehicle geometry, not just capability) plus the other agent's independent lateral position/velocity -- there is no width-free version of a_lat,req itself in the source (only the downstream capability-normalization step, i.e. STN = a_lat,req/a_lat,min, is width-free, but that still needs a_lat,req first). Separately: common.py's action_stats_from_logits() DOES compute a 'steer'/'steer_mag' signal from the policy logits, but rollout.py's readout_out capture map only pulls accel/p_brake/p_yield/gap_press/entropy into the saved pack -- steer was never persisted, so even the ego's OWN lateral effort isn't available today, let alone the other agent's position/width needed for the true formula. If vehicle width/other-agent lateral position were ever added, literature-grounded a_lat,min values from Jansson's own demonstrator: 7.0 m/s^2 (deployed-system bound) or 9.82 m/s^2 (idealized physical limit, ~1g).",
    "a_req (combined norm)": "downstream of a_lat,req (not applicable, see above) regardless of which of Jansson's three original combination formulas is used (a plain min(), the paper's own sqrt-of-squares joint-optimal solution -- which is itself NOT simply two independent 1D formulas combined post-hoc, per a direct read of Jansson Eq. 5.59-5.61 -- or a friction-ellipse feasibility check). Reporting only the longitudinal term would silently misrepresent whichever of these three the reader assumes is meant, so it is omitted rather than degraded.",
    "TTS (and TTM's m='steer' case)": "Hillenbrand (2007), read in full, confirms TTS is a genuine 2D circular-arc turning-radius model (needing the ego's minimum turning radius -- itself a function of vehicle geometry and a friction-limited bound -- plus the other agent's lateral offset and both vehicles' width/length), with explicitly NO simplified constant-lateral-clearance fallback anywhere in the source -- the actual model is geometrically more detailed than a clearance heuristic would be, not less. TTB and TTK (TTM's m='brake'/'kickdown' cases) ARE implemented -- see EXACT_METRICS/APPROXIMATE_METRICS.",
}