"""
path_planner.py - from pen strokes to timed, minimum-energy motor trajectories.

Pipeline
  1. order_strokes   greedy nearest-neighbour, strokes may be reversed and
                     closed loops re-started at their nearest point
                     (cuts pen-up travel)
  2. time_path       human-like timing:
                       pen down : 2/3 power law  v = k * R^(1/3)  (slow in tight
                                  curves / corners) + acceleration limit,
                                  start and end each stroke at rest
                       pen up   : minimum-jerk straight-line reach, duration
                                  from a Fitts'-law-style log rule
  3. plan_redundancy dynamic programming over the redundant hand angle phi3:
                       minimise  integral[ qd' M(q) qd  +  w_c * discomfort(q) ] dt
                     (kinetic-energy-weighted motion + human range-of-motion
                     comfort + transmission margin), globally over the path
  4. dynamics        motor speed, torque (exact rigid-body model for the
                     absolute-angle coordinates), power and work per motor
"""
import math
from dataclasses import dataclass, field

import numpy as np

from config import CFG
from arm_kinematics import (HAND_GRID, ik_grid, inverse_dynamics, mass_matrix,
                            mass_params, wrap_np, kinetic_energy, solve_ik)


# ------------------------------------------------------------------ params
_P = CFG["plan"]


@dataclass
class PlanParams:
    """Defaults come from params.toml [plan]."""
    dt: float = _P["dt"]                   # s, trajectory sample time
    power_gain: float = _P["power_gain"]   # k in v = k R^(1/3)
    v_min: float = _P["v_min"]             # mm/s
    v_max: float = _P["v_max"]             # mm/s, pen-down cap
    a_max: float = _P["a_max"]             # mm/s^2, tangential acceleration cap
    pen_dwell: float = _P["pen_dwell"]     # s to lift or lower the pen
    w_energy: float = _P["w_energy"]       # weight of kinetic-energy term
    w_comfort: float = _P["w_comfort"]     # weight of ROM-discomfort term
    w_hand: float = _P["w_hand"]           # extra penalty on hand-angle rate
    band: int = _P["band"]                 # DP: max hand-angle grid steps per DP sample
    dp_stride: int = _P["dp_stride"]       # DP runs on every n-th sample


# ------------------------------------------------------------ 1. ordering
def order_strokes(strokes, start):
    todo = [np.asarray(s, float) for s in strokes if len(s) > 1]
    out, pos = [], np.asarray(start, float)
    while todo:
        best = None
        for i, s in enumerate(todo):
            closed = np.hypot(*(s[0] - s[-1])) < 1e-6
            if closed:
                d = np.hypot(*(s - pos).T)
                k = int(np.argmin(d))
                cand = (d[k], i, np.vstack([s[k:], s[1:k + 1]]) if k else s)
            else:
                d0, d1 = np.hypot(*(s[0] - pos)), np.hypot(*(s[-1] - pos))
                cand = (d0, i, s) if d0 <= d1 else (d1, i, s[::-1])
            if best is None or cand[0] < best[0]:
                best = cand
        _, i, s = best
        out.append(s)
        pos = s[-1]
        todo.pop(i)
    return out


# --------------------------------------------------------------- 2. timing
def _resample(poly, step):
    seg = np.hypot(*np.diff(poly, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 1e-9:
        return poly[:1], s[:1]
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    ss = np.linspace(0, s[-1], n)
    return np.stack([np.interp(ss, s, poly[:, 0]), np.interp(ss, s, poly[:, 1])], 1), ss


def _stroke_timing(poly, p):
    """Arc-length samples with times following the 2/3 power law."""
    pts, s = _resample(poly, 0.5)
    if len(pts) < 2:
        return pts, np.zeros(1)
    # curvature from turning angle per unit length (robust at corners)
    d = np.diff(pts, axis=0)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    kappa = np.zeros(len(pts))
    kappa[1:-1] = np.abs(np.diff(th)) / 0.5
    w = 7
    kappa = np.convolve(np.pad(kappa, w // 2, mode="edge"), np.ones(w) / w, mode="valid")
    R = 1.0 / np.maximum(kappa, 1e-6)
    v = np.clip(p.power_gain * np.cbrt(R), p.v_min, p.v_max)
    v[0] = v[-1] = 0.0
    ds = np.diff(s)
    for i in range(1, len(v)):                        # accel limit, forward
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * p.a_max * ds[i - 1]))
    for i in range(len(v) - 2, -1, -1):               # decel limit, backward
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * p.a_max * ds[i]))
    vm = np.maximum((v[1:] + v[:-1]) / 2, 1e-3)
    t = np.r_[0, np.cumsum(ds / vm)]
    return pts, t


def _minjerk(a, b, dt):
    dist = float(np.hypot(*(b - a)))
    T = 0.15 + 0.10 * math.log2(1 + dist / 5.0)         # Fitts-like duration
    n = max(2, int(round(T / dt)) + 1)
    tau = np.linspace(0, 1, n)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    return a + (b - a) * s[:, None], np.linspace(0, T, n)


def time_path(strokes, start, p):
    """-> t, xy (N,2), pen (N,) bool sampled every p.dt."""
    T, XY, PEN = [], [], []
    t0, pos = 0.0, np.asarray(start, float)

    def add(pts, ts, pen):
        nonlocal t0
        T.append(t0 + ts)
        XY.append(pts)
        PEN.append(np.full(len(ts), pen))
        t0 = t0 + ts[-1]

    for s in strokes:
        if np.hypot(*(s[0] - pos)) > 1e-6:
            pts, ts = _minjerk(pos, s[0], p.dt)
            add(pts, ts, False)
        add(np.vstack([s[0], s[0]]), np.array([0, p.pen_dwell]), False)       # lower
        pts, ts = _stroke_timing(s, p)
        add(pts, ts, True)
        add(np.vstack([s[-1], s[-1]]), np.array([0, p.pen_dwell]), False)     # lift
        pos = s[-1]
    t = np.concatenate(T)
    xy = np.vstack(XY)
    pen = np.concatenate(PEN)
    # uniform time grid (keep the first sample at duplicate times)
    keep = np.r_[True, np.diff(t) > 1e-9]
    t, xy, pen = t[keep], xy[keep], pen[keep]
    tu = np.arange(0, t[-1] + 1e-9, p.dt)
    xu = np.interp(tu, t, xy[:, 0])
    yu = np.interp(tu, t, xy[:, 1])
    idx = np.clip(np.searchsorted(t, tu, side="right") - 1, 0, len(t) - 1)
    # a sample is pen-down only if the segment it lies on is pen-down at both ends
    pu = pen[idx] & pen[np.clip(idx + 1, 0, len(t) - 1)]
    return tu, np.stack([xu, yu], 1), pu


# --------------------------------------------------------- 3. redundancy DP
def plan_redundancy(g, xy, p, chunk=1500):
    """Viterbi over the hand-angle grid.  Returns phi (N,3), valid mask."""
    K = len(HAND_GRID)
    step = HAND_GRID[1] - HAND_GRID[0]
    H = mass_params(g)
    B = p.band
    shifts = np.arange(-B, B + 1)
    src = (np.arange(K)[None, :] - shifts[:, None]) % K          # [d, k] -> previous k
    N = len(xy)
    bp = np.zeros((N, K), np.int8)
    P1 = np.zeros((N, K), np.float32)
    P2 = np.zeros((N, K), np.float32)
    restart = np.zeros(N, bool)
    seg_end = {}                    # cost vectors where a restart happened
    valid = np.ones(N, bool)
    prev = None
    prev_q = None
    for c0 in range(0, N, chunk):
        c1 = min(N, c0 + chunk)
        px, py = xy[c0:c1, 0:1], xy[c0:c1, 1:2]
        phi1, phi2, ok, sc = ik_grid(g, px, py, HAND_GRID[None, :])
        P1[c0:c1], P2[c0:c1] = phi1, phi2
        state = np.where(ok, p.w_comfort * sc * p.dt, np.inf)
        q = np.stack([phi1, phi2, np.broadcast_to(HAND_GRID, phi1.shape)], -1)     # (n,K,3)
        # transition energy for every sample, shift, destination: dq' M(q_dest) dq / dt
        qprev = np.concatenate([prev_q[None] if prev_q is not None else q[:1], q[:-1]], 0)
        dq = wrap_np(q[:, None, :, :] - qprev[:, src, :])                       # (n,d,K,3)
        M = mass_matrix(H, q)                                                     # (n,K,3,3)
        trans = p.w_energy * np.einsum("ndki,nkij,ndkj->ndk", dq, M, dq) / p.dt
        if p.w_hand:
            trans = trans + p.w_hand * dq[..., 2] ** 2 / p.dt
        for i in range(c1 - c0):
            n = c0 + i
            if prev is None:
                cur = state[i].copy()
                restart[n] = True
            else:
                cand = prev[src] + trans[i]                                      # (d,K)
                j = np.argmin(cand, 0)
                cur = cand[j, np.arange(K)] + state[i]
                bp[n] = j
                if not np.isfinite(cur).any():                                   # band too tight / jump
                    seg_end[n - 1] = prev
                    cur = state[i].copy()
                    restart[n] = True
            if not np.isfinite(cur).any():                                       # unreachable point
                valid[n] = False
                cur = prev if prev is not None else np.zeros(K)
                restart[n] = prev is None
                bp[n] = B                                                        # shift 0
            prev = cur
        prev_q = q[-1]
    # backtrack
    k = np.zeros(N, int)
    k[-1] = int(np.argmin(prev))
    for n in range(N - 1, 0, -1):
        if restart[n]:          # chain restarted: previous segment ends at its own optimum
            k[n - 1] = int(np.argmin(seg_end[n - 1]))
        else:
            k[n - 1] = (k[n] - shifts[bp[n, k[n]]]) % K
    phi3 = HAND_GRID[k]
    phi = np.stack([P1[np.arange(N), k], P2[np.arange(N), k], phi3], 1).astype(float)
    for n in range(N):          # unreachable samples hold the last good pose
        if not valid[n]:
            phi[n] = phi[n - 1] if n else phi[n]
    return _smooth(g, xy, phi, valid), valid


def _smooth(g, xy, phi, valid, sigma=4.0):
    """Remove the 2-degree grid quantisation from phi3, then re-solve phi1/2."""
    from scipy.ndimage import gaussian_filter1d
    p3 = np.unwrap(phi[:, 2])
    p3s = gaussian_filter1d(p3, sigma, mode="nearest")
    a1, a2, ok, _ = ik_grid(g, xy[:, 0], xy[:, 1], p3s)
    use = ok & valid
    out = phi.copy()
    out[use, 0], out[use, 1], out[use, 2] = a1[use], a2[use], wrap_np(p3s[use])
    return out


def plan_strided(g, xy, p):
    """Run the DP on a decimated path (dt * stride), then interpolate the hand
    angle back to full rate and re-solve the arm exactly."""
    k = max(1, int(p.dp_stride))
    if k == 1:
        return plan_redundancy(g, xy, p)
    idx = np.r_[np.arange(0, len(xy), k), len(xy) - 1] if (len(xy) - 1) % k else np.arange(0, len(xy), k)
    pk = PlanParams(**{**p.__dict__, "dt": p.dt * k, "band": p.band * k})
    phik, validk = plan_redundancy(g, xy[idx], pk)
    n = np.arange(len(xy))
    p3 = np.interp(n, idx, np.unwrap(phik[:, 2]))
    a1, a2, ok, _ = ik_grid(g, xy[:, 0], xy[:, 1], p3)
    near = np.clip(np.round(n / k).astype(int), 0, len(idx) - 1)
    valid = ok & validk[near]
    phi = np.stack([a1, a2, wrap_np(p3)], 1)
    for i in np.flatnonzero(~valid):            # hold last good pose
        phi[i] = phi[i - 1] if i else phik[0]
    return phi, valid


def greedy_redundancy(g, xy):
    """Baseline: per-sample local IK (what the interactive mouse mode does)."""
    phi = np.zeros((len(xy), 3))
    valid = np.ones(len(xy), bool)
    prev = None
    for n, P in enumerate(xy):
        s = solve_ik(g, P, prev)
        if s is None:
            valid[n] = False
            phi[n] = phi[n - 1] if n else 0
            continue
        prev = s["phi3"]
        phi[n] = s["phi1"], s["phi2"], s["phi3"]
    return phi, valid


# ---------------------------------------------------------------- 4. dynamics
@dataclass
class Plan:
    t: np.ndarray
    xy: np.ndarray
    pen: np.ndarray
    phi: np.ndarray            # (N,3) absolute link angles
    valid: np.ndarray
    motors: np.ndarray         # (N,3) motor shaft angles (rad, unwrapped)
    omega: np.ndarray          # (N,3) rad/s
    alpha: np.ndarray          # (N,3) rad/s^2
    tau: np.ndarray            # (N,3) N m
    power: np.ndarray          # (N,3) W
    stats: dict = field(default_factory=dict)


def dynamics(g, t, xy, pen, phi, valid, dt):
    from scipy.ndimage import gaussian_filter1d
    q = np.unwrap(phi, axis=0)
    motors = q + np.array([0.0, math.pi, g.off_a])
    qd = np.gradient(q, dt, axis=0)
    qdd = gaussian_filter1d(np.gradient(qd, dt, axis=0), 2.0, axis=0, mode="nearest")
    H = mass_params(g)
    tau = inverse_dynamics(H, q, qd, qdd)
    power = tau * qd
    seg = np.hypot(*np.diff(xy, axis=0).T)
    stats = dict(
        duration_s=float(t[-1]),
        ink_mm=float(seg[pen[1:] & pen[:-1]].sum()),
        travel_mm=float(seg[~(pen[1:] & pen[:-1])].sum()),
        unreachable=int((~valid).sum()),
        peak_speed_rpm=(np.abs(qd).max(0) * 60 / (2 * math.pi)).tolist(),
        peak_torque_Nm=np.abs(tau).max(0).tolist(),
        rms_torque_Nm=np.sqrt((tau ** 2).mean(0)).tolist(),
        pos_work_J=(np.clip(power, 0, None).sum(0) * dt).tolist(),
        abs_work_J=(np.abs(power).sum(0) * dt).tolist(),
        kinetic_integral=float((2 * kinetic_energy(H, q, qd)).sum() * dt),   # integral qd'Mqd dt
        motor_range_deg=np.degrees(motors.max(0) - motors.min(0)).tolist(),
    )
    return Plan(t, xy, pen, phi, valid, motors, qd, qdd, tau, power, stats)


def plan_job(g, strokes, start, p=None, method="dp"):
    p = p or PlanParams()
    strokes = order_strokes(strokes, start)
    t, xy, pen = time_path(strokes, start, p)
    if method == "dp":
        phi, valid = plan_strided(g, xy, p)
    else:
        phi, valid = greedy_redundancy(g, xy)
    pen = pen & valid
    return dynamics(g, t, xy, pen, phi, valid, p.dt)


def export_csv(plan, path):
    hdr = ("t_s,pen,x_mm,y_mm,phi1_deg,phi2_deg,phi3_deg,"
           "M1_deg,M2_deg,M3_deg,w1_dps,w2_dps,w3_dps,tau1_Nm,tau2_Nm,tau3_Nm")
    data = np.column_stack([plan.t, plan.pen.astype(int), plan.xy,
                            np.degrees(plan.phi), np.degrees(plan.motors),
                            np.degrees(plan.omega), plan.tau])
    np.savetxt(path, data, delimiter=",", header=hdr, comments="",
               fmt=["%.3f", "%d"] + ["%.4f"] * 14)
