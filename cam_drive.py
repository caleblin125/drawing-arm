"""
cam_drive.py - encode a planned drawing onto a stack of groove cams driven by
ONE motor.

Layout (top view, same frame as the arm, mm):
    S  shoulder axle (the three arm hubs: M1 upper arm, M2 elbow crank, M3 wrist crank)
    O  camshaft, D mm from S in direction gamma_O (behind the shoulder, toward
       the body; the arm/cranks never sweep there - see README)
    For each arm input i:
        hub i carries a spur-gear SECTOR (z_hub teeth, module m) at its own layer
        follower i pivots at F_i, carries the mating sector (z_f teeth) and an
        arm of length l_i ending in a roller (MR74 bearing) that rides in the
        groove of cam i.  hub angle change = -(z_f/z_hub) * follower change
    Pen-lift cam: its follower lever pulls a Bowden cable to the pen holder.

One camshaft revolution = one complete drawing (plan is closed into a loop:
after the last stroke the arm returns to the start pose, pen up).

Synthesis per axis: grid search over gear ratio, follower arm length and pitch
radius, keeping max pressure angle <= alpha_max, groove curvature > roller
radius, monotone polar angle (no self-intersection), and picking the
smallest cam.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from arm_kinematics import Geometry
from config import CFG

D = math.radians


_DR = CFG["drive"]


@dataclass
class DriveParams:
    """Defaults come from params.toml [drive]."""
    D_SO: float = _DR["D_SO"]                    # shoulder axle -> camshaft distance (mm)
    gamma_O: float = D(_DR["gamma_O"])           # direction of the camshaft from S
    module: float = _DR["module"]                # gear module (mm)
    z_hub: int = _DR["z_hub"]                    # teeth on the hub sectors
    z_follow: tuple = tuple(_DR["z_follow"])     # follower sector teeth candidates
    roller_r: float = _DR["roller"]["od"] / 2    # cam roller bearing radius
    roller_w: float = _DR["roller"]["width"]
    wall: float = _DR["wall"]                    # groove wall thickness
    shaft_r: float = _DR["shaft_d"] / 2
    alpha_max: float = D(_DR["alpha_max"])       # max pressure angle
    min_wall_radius: float = _DR["min_wall_radius"]
    n: int = _DR["n"]                            # cam samples per revolution
    return_time: float = _DR["return_time"]      # s, pen-up return to the start pose
    pen_rise: float = _DR["pen_rise"]            # mm roller rise for pen lift
    pen_ramp_deg: float = _DR["pen_ramp_deg"]    # cam degrees per pen up/down ramp
    pen_ramp: float = 0.25                       # s (set by plan_for_cams)
    cam_stack: tuple = tuple(_DR["cam_stack"])   # cams bottom -> top
    standoff_r: float = CFG["parts"]["standoff_od"] / 2


# ------------------------------------------------------------ loop closing
def _minjerk(a, b, n):
    tau = np.linspace(0, 1, n)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    return a + (b - a) * s[:, None]


def periodic_profiles(plan, p: DriveParams):
    """-> theta (n,), motor angles (n,3) rad, pen (n,) in [0,1] (1 = down)."""
    dt = plan.t[1] - plan.t[0]
    M = plan.motors
    nr = max(3, int(p.return_time / dt))
    ret = _minjerk(M[-1], M[0], nr + 2)[1:-1]
    Mall = np.vstack([M, ret])
    pen = np.r_[plan.pen.astype(float), np.zeros(len(ret))]
    T = len(Mall) * dt
    t = np.arange(len(Mall)) * dt
    tu = np.linspace(0, T, p.n, endpoint=False)
    Mu = np.stack([np.interp(tu, t, Mall[:, k], period=T) for k in range(3)], 1)
    # pen: cycloidal ramps of pen_ramp seconds (centered on each transition)
    pen_u = np.interp(tu, t, pen, period=T)
    w = max(1, int(round(p.pen_ramp / (T / p.n))))
    ker = 1 - np.cos(np.linspace(0, 2 * np.pi, w + 2)[1:-1])
    ker /= ker.sum()
    pen_s = np.real(np.fft.ifft(np.fft.fft(pen_u) * np.fft.fft(np.r_[ker, np.zeros(p.n - len(ker))])))
    pen_s = np.roll(pen_s, -(len(ker) // 2))
    return 2 * np.pi * tu / T, Mu, np.clip(pen_s, 0, 1), T


# ---------------------------------------------------------- cam geometry
def _pitch(theta, O, F, l, psi):
    """Roller-centre path in the cam frame (cam turns CW: frame angle +theta)."""
    R = F + l * np.stack([np.cos(psi), np.sin(psi)], 1)
    v = R - O
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1), R


def evaluate_cam(theta, O, F, l, psi, p):
    pc, R = _pitch(theta, O, F, l, psi)
    dth = theta[1] - theta[0]
    d1 = (np.roll(pc, -1, 0) - np.roll(pc, 1, 0)) / (2 * dth)
    d2 = (np.roll(pc, -1, 0) - 2 * pc + np.roll(pc, 1, 0)) / dth ** 2
    sp = np.hypot(d1[:, 0], d1[:, 1])
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    rho = sp ** 3 / np.where(np.abs(cross) < 1e-12, 1e-12, cross)
    n_c = np.stack([-d1[:, 1], d1[:, 0]], 1) / sp[:, None]
    c, s = np.cos(-theta), np.sin(-theta)                     # back to ground frame
    n_g = np.stack([c * n_c[:, 0] - s * n_c[:, 1], s * n_c[:, 0] + c * n_c[:, 1]], 1)
    v = np.stack([-np.sin(psi), np.cos(psi)], 1)              # roller velocity dir
    alpha = np.arccos(np.clip(np.abs((n_g * v).sum(1)), 0, 1))
    polar = np.unwrap(np.arctan2(pc[:, 1], pc[:, 0]))
    monotone = np.all(np.diff(polar) > 0) or np.all(np.diff(polar) < 0)
    r = np.hypot(pc[:, 0], pc[:, 1])
    return dict(pitch=pc, R=R, alpha=alpha, rho=rho, monotone=monotone,
                r_min=r.min(), r_max=r.max(),
                outer=r.max() + p.roller_r + p.wall,
                inner=r.min() - p.roller_r - p.wall)


def _circle_x(c0, r0, c1, r1):
    d = np.hypot(*(c1 - c0))
    if d > r0 + r1 or d < abs(r0 - r1) or d < 1e-9:
        return []
    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(max(r0 ** 2 - a ** 2, 0))
    m = c0 + a * (c1 - c0) / d
    perp = np.array([-(c1 - c0)[1], (c1 - c0)[0]]) / d
    return [m + h * perp, m - h * perp]


def synth_axis(theta, rel, p, S, O, geared=True, name="", min_pivot=0.0, keepout=None):
    """rel: (n,) desired OUTPUT angle relative to its mean (rad).
    min_pivot: the follower pivot (and its standoff) must be at least this far
    from the camshaft, i.e. clear of every cam lower in the stack.
    keepout(F) -> clearance (mm) of the pivot standoff to lower followers' swept areas.
    Returns the best design dict or None."""
    best = None
    ratios = [(zf, zf / p.z_hub) for zf in p.z_follow] if geared else [(0, 1.0)]
    for zf, G in ratios:
        a = p.module * (p.z_hub + zf) / 2 if geared else None
        for l in np.arange(25, 86, 5.0):
            psi_rel = -rel / G if geared else rel * p.pen_rise / l
            if np.ptp(psi_rel) > D(70):
                break
            for Rp in np.arange(20, 111, 5.0):
                c = math.hypot(Rp, l)
                if geared:
                    Fs = _circle_x(S, a, O, c)
                else:
                    Fs = [O + c * np.array([math.cos(ang), math.sin(ang)]) for ang in np.radians(np.arange(0, 360, 30))]
                for F in Fs:
                    for Rm in _circle_x(O, Rp, F, l):
                        psi0 = math.atan2(*(Rm - F)[::-1])
                        psi = psi0 + psi_rel - psi_rel.mean()
                        ev = evaluate_cam(theta, O, F, l, psi, p)
                        if not ev["monotone"] or ev["alpha"].max() > p.alpha_max:
                            continue
                        if np.abs(ev["rho"]).min() < p.roller_r + p.min_wall_radius:
                            continue
                        if ev["inner"] < p.shaft_r + 3:
                            continue
                        # follower pivot boss must clear the cam
                        if np.hypot(*(F - O)) < max(ev["outer"], min_pivot) + p.standoff_r + 1:
                            continue
                        if keepout is not None and keepout(F) < 1.0:
                            continue
                        if geared and np.hypot(*(S - O)) < ev["outer"] + p.module * p.z_hub / 2 + 3:
                            continue
                        score = ev["outer"] + 4 * G
                        if best is None or score < best["score"]:
                            best = dict(name=name, score=score, zf=zf, G=G, l=l, Rp=Rp, F=F, psi=psi,
                                        a=a, **ev)
    return best


def _seg_dist(q, a, b):
    ab = b - a
    t = np.clip(np.dot(q - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.hypot(*(q - (a + t * ab))))


def swept_keepout(designs, p, arm_r=7.0):
    """Clearance of a vertical standoff at F to the swept arms/sectors of the
    given (lower) follower designs."""
    def fn(F):
        best = 1e9
        for d in designs:
            idx = np.linspace(0, len(d["psi"]) - 1, 36).astype(int)
            for k in idx:
                R = d["F"] + d["l"] * np.array([math.cos(d["psi"][k]), math.sin(d["psi"][k])])
                best = min(best, _seg_dist(F, d["F"], R) - arm_r - p.standoff_r)
            if d.get("zf"):
                rs = p.module * d["zf"] / 2
                dd = np.hypot(*(F - d["F"]))
                a_m = math.atan2(-d["F"][1], -d["F"][0])
                aq = math.atan2(*(F - d["F"])[::-1])
                if abs((aq - a_m + math.pi) % (2 * math.pi) - math.pi) < np.ptp(d["psi"]) / 2 + 0.3:
                    best = min(best, abs(dd - (rs - 2)) - (1.5 * p.module + 5) - p.standoff_r)
            best = min(best, np.hypot(*(F - d["F"])) - 2 * p.standoff_r)
        return best
    return fn


@dataclass
class CamSet:
    params: DriveParams
    theta: np.ndarray
    motors: np.ndarray
    pen: np.ndarray
    period: float
    O: np.ndarray
    axes: list = field(default_factory=list)       # M1, M2, M3 designs
    pen_axis: dict = None


def design_cams(plan, g: Geometry, p: DriveParams = None):
    p = p or DriveParams()
    theta, M, pen, T = periodic_profiles(plan, p)
    S = np.zeros(2)
    O = p.D_SO * np.array([math.cos(p.gamma_O), math.sin(p.gamma_O)])
    names = {"M1": (0, "M1 shoulder"), "M2": (1, "M2 elbow"), "M3": (2, "M3 wrist")}
    axes = [None, None, None]
    pen_axis = None
    lower_outer = 0.0                            # largest cam below the current follower
    lower = []                                   # designs of followers below
    for key in p.cam_stack:                      # bottom -> top: pivots must clear lower cams
        if key == "pen":
            pen_rel = (1 - pen)                  # 1 = lifted; scaled to a rise per arm length
            pen_axis = synth_axis(theta, pen_rel - pen_rel.mean(), p, S, O, False, "pen lift",
                                  min_pivot=lower_outer, keepout=swept_keepout(lower, p))
            des = pen_axis
        else:
            k, name = names[key]
            rel = M[:, k] - M[:, k].mean()
            des = synth_axis(theta, rel, p, S, O, True, name, min_pivot=lower_outer,
                             keepout=swept_keepout(lower, p))
            if des is not None:
                des["M_mean"] = float(M[:, k].mean())
            axes[k] = des
        if des is not None:
            lower_outer = max(lower_outer, des["outer"])
            lower.append(des)
    return CamSet(p, theta, M, pen, T, O, axes, pen_axis)


def report(cs: CamSet):
    lines = [f"camshaft period {cs.period:.1f} s per drawing  (motor -> worm -> camshaft at "
             f"{60 / cs.period:.2f} rpm)"]
    for a in cs.axes + [cs.pen_axis]:
        if a is None:
            lines.append("  (axis infeasible: drawing too busy for one cam revolution)")
            continue
        lines.append(f"  {a['name']:12s} cam OD {2 * a['outer']:5.0f} mm  ratio {a['G']:.2f} "
                     f"(z {a['zf']}:{cs.params.z_hub})  arm {a['l']:.0f}  pitch r {a['Rp']:.0f}  "
                     f"max pressure {math.degrees(a['alpha'].max()):4.1f} deg  min rho {np.abs(a['rho']).min():.1f}")
    return "\n".join(lines)


def plot_cams(cs: CamSet, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axes = [a for a in cs.axes + [cs.pen_axis] if a is not None]
    fig, ax = plt.subplots(1, len(axes), figsize=(4 * len(axes), 4.4))
    ax = np.atleast_1d(ax)
    p = cs.params
    for a, A in zip(axes, ax):
        pc = a["pitch"]
        nrm = np.gradient(pc, axis=0)
        nrm = np.stack([-nrm[:, 1], nrm[:, 0]], 1) / np.hypot(*nrm.T)[:, None]
        for sgn in (-1, 1):
            w = pc + sgn * p.roller_r * nrm
            A.plot(*np.vstack([w, w[:1]]).T, color="#1a237e", lw=1)
        t = np.linspace(0, 2 * np.pi, 200)
        A.plot(a["outer"] * np.cos(t), a["outer"] * np.sin(t), color="#888", lw=0.8)
        A.plot(p.shaft_r * np.cos(t), p.shaft_r * np.sin(t), color="#888", lw=0.8)
        A.plot(*pc.T, color="#e08a1e", lw=0.6, ls="--")
        A.set_aspect("equal")
        A.set_title(f"{a['name']}\nOD {2 * a['outer']:.0f} mm, max α {math.degrees(a['alpha'].max()):.0f}°",
                    fontsize=9)
        A.axis("off")
    fig.suptitle(f"groove cams (one revolution = {cs.period:.1f} s drawing)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plan_for_cams(g, strokes, p: DriveParams = None, iters=3):
    """Plan tuned for cams: gentler accelerations, smooth hand, the loop starts
    on the drawing, and pen ramps sized in cam degrees (overlapping travel)."""
    from path_planner import PlanParams, plan_job, order_strokes
    p = p or DriveParams()
    strokes = order_strokes(strokes, strokes[0][0])
    start = strokes[0][0]
    pp = PlanParams(**CFG["plan"].get("cam", {}))
    plan = None
    for _ in range(iters):
        plan = plan_job(g, strokes, start, pp)
        T = plan.t[-1] + p.return_time
        p.pen_ramp = p.pen_ramp_deg / 360 * T
        pp.pen_dwell = p.pen_ramp / 2           # other half overlaps the travel
    return plan, p


if __name__ == "__main__":
    import sys
    from arm_linkage import PAPER, builtin_shape
    from path_planner import PlanParams, plan_job
    from vector_input import fit_to_box, load_svg

    src = sys.argv[1] if len(sys.argv) > 1 else "heart"
    strokes = load_svg(src) if src.endswith(".svg") else builtin_shape(src)
    g = Geometry()
    plan, p = plan_for_cams(g, fit_to_box(strokes, PAPER))
    cs = design_cams(plan, g, p)
    print(report(cs))
    plot_cams(cs, f"cad/cams_{src.split('/')[-1].split('.')[0]}.png")
