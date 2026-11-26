"""
cartridge.py - swappable cam cartridge: FIXED followers, replaceable cams.

    python -m drawarm cartridge [drawing ...]   -> report, output/cartridge_<name>.png

Why: in cam mode (cams.py) every drawing re-optimises the follower pivots,
so new drawings need new followers and a new base plate.  Here the followers
are designed ONCE (from a few reference drawings and the whole paper range)
and only the cam stack changes:

* single-sided PERIPHERY cams: the roller rides on the cam's rim, held on it
  by the anti-backlash torsion springs (spring direction chosen per axis so
  the roller is always pushed toward the camshaft).  No grooves, so the cam
  stack can move away from the rollers.
* cam carriage: camshaft + cams + the motor/worm sit on a carriage that
  slides out along `slide` (away from the followers).  The rollers follow the
  cams a few degrees and come to rest on STOPS; the cams then slide clear.
  Lift the old stack off the D-shaft, drop the new one on, slide back, latch.
* every contact point faces the followers (inside a cone opposite the slide
  direction), so sliding out always moves the cams away from the rollers.

Follower angle law (fixed for every drawing, through the 3:1 gear mesh):
    psi_k(t) = psi_ref_k - (M_k(t) - M_ref_k) / G      (geared axes)
    psi_pen(t) = psi_ref + pen_rise / l * (1 - pen(t))  (pen-lift lever)
"""
import math
from dataclasses import dataclass, field, replace

import numpy as np

from .cams import (CamSet, DriveParams, _circle_x, cam_profiles, evaluate_cam, groove_contact,
                   plan_for_cams, swept_keepout)
from .config import CFG, PAPER
from .kinematics import Geometry, paper_motor_range

_CP = CFG.get("cartridge", {})
NAMES = ("M1", "M2", "M3", "pen")


@dataclass
class CartridgeParams:
    """Defaults from params.toml [cartridge] (all optional)."""
    reference: tuple = tuple(_CP.get("reference", ["circle", "heart", "star", "spiral"]))
    max_od: float = _CP.get("max_od", 220.0)              # largest cam the carriage takes (print bed)
    contact_cone: float = math.radians(_CP.get("contact_cone", 75.0))   # contact points vs. -slide
    stop_margin: float = math.radians(_CP.get("stop_margin", 4.0))      # follower stop beyond travel
    slide_clear: float = _CP.get("slide_clear", 2.0)      # roller <-> cam while sliding out (mm)
    pen_lift: float = _CP.get("pen_lift", 3.0)            # pen travel at the hand (mm)
    pen_rise: float = _CP.get("pen_rise", 3.0)            # pen-lift roller rise on the cam (mm)
    compensate: bool = _CP.get("compensate", True)        # pre-distort cams for predicted deflection
    compensate_smooth_deg: float = _CP.get("compensate_smooth_deg", 6.0)   # smoothing of the correction


@dataclass
class Followers:
    """The fixed follower set (one design for every drawing)."""
    params: DriveParams
    cart: CartridgeParams
    O: np.ndarray
    slide: np.ndarray                                    # unit vector, carriage slide-out direction
    axes: dict = field(default_factory=dict)             # name -> F, l, G, zf, psi_ref, M_ref, psi_lo/hi ...
    spring_sign: tuple = (1, 1, 1)                        # hub spring direction that pushes rollers inward


# ------------------------------------------------------------ angle laws
def follower_psi(ax, M=None, pen=None):
    if ax["zf"]:
        return ax["psi_ref"] - (M - ax["M_ref"]) / ax["G"]
    return ax["psi_ref"] + ax["rise"] / ax["l"] * (1 - pen)


def _inward_sign(F, l, psi, O):
    """+1 if increasing psi moves the roller toward O."""
    R = F + l * np.array([math.cos(psi), math.sin(psi)])
    v = np.array([-math.sin(psi), math.cos(psi)])
    return 1 if np.dot(R - O, v) < 0 else -1


def periphery_eval(theta, O, F, l, psi, p):
    """evaluate_cam, then single-sided quantities: cam rim = pitch curve offset
    toward O by the roller; envelope = rim max radius."""
    ev = evaluate_cam(theta, O, F, l, psi, p)
    ev["outer"] = ev["r_max"] - p.roller_r                       # cam rim, no groove wall
    ev["inner"] = ev["r_min"] - p.roller_r
    return ev


def contact_dirs_ok(R, O, slide, cone):
    """All roller centres must lie within `cone` of -slide (seen from O)."""
    d = R - O
    c = -(d @ slide) / np.linalg.norm(d, axis=1)
    return bool(np.all(c >= math.cos(cone)))


# --------------------------------------------------------- follower design
def design_followers(g: Geometry, p: DriveParams = None, cp: CartridgeParams = None, refs=None, verbose=False):
    """Choose ONE follower geometry per axis that works for every reference
    drawing (and stays inside the carriage envelope over the whole paper)."""
    from .inputs import fit_to_box
    from .shapes import load_drawing
    cp = cp or CartridgeParams()
    p = replace(p or DriveParams(), pen_rise=cp.pen_rise)
    S = np.zeros(2)
    O = p.D_SO * np.array([math.cos(p.gamma_O), math.sin(p.gamma_O)])
    slide = O / np.linalg.norm(O)                                # away from the shoulder
    M_paper = paper_motor_range(g)
    M_ref = np.array([(lo + hi) / 2 for lo, hi in M_paper])
    # reference profiles (re-timed cam angle), computed once
    if refs is None:
        refs = []
        for name in cp.reference:
            plan, pp = plan_for_cams(g, fit_to_box(load_drawing(name), PAPER), replace(p))
            th, M, pen, T, _ = cam_profiles(plan, pp)
            refs.append((name, th, M, pen, plan))
    pre = CFG["drive"].get("preload", {})
    hub_T = np.array(pre.get("hub_torque", [0, 0, 0]), float) * (1 + pre.get("torque_variation", 0))
    axes, lower = {}, []
    lower_outer = 0.0
    for key in p.cam_stack:
        geared = key != "pen"
        k = NAMES.index(key)
        best = None
        zf = max(p.z_follow) if geared else 0
        G = zf / p.z_hub if geared else 1.0
        a = p.module * (p.z_hub + zf) / 2 if geared else None
        for l in np.arange(30, 91, 5.0):
            rise = p.pen_rise
            for Rp in np.arange(25, 96, 5.0):
                c = math.hypot(Rp, l)
                if geared:
                    Fs = _circle_x(S, a, O, c)
                else:
                    Fs = [O + c * np.array([math.cos(t), math.sin(t)])
                          for t in np.radians(np.arange(0, 360, p.pen_pivot_step))]
                for F in Fs:
                    for Rm in _circle_x(O, Rp, F, l):
                        psi_ref = math.atan2(*(Rm - F)[::-1])
                        ax = dict(name=key, F=F, l=l, G=G, zf=zf, psi_ref=psi_ref,
                                  M_ref=M_ref[k] if geared else 0.0, rise=rise, Rp=Rp, a=a)
                        # whole-paper travel -> envelope and pivot clearance
                        if geared:
                            ps = follower_psi(ax, M=np.array(M_paper[k]))
                        else:
                            ps = follower_psi(ax, pen=np.array([1.0, 0.0]))
                        ps_lo, ps_hi = ps.min(), ps.max()
                        arc = np.linspace(ps_lo, ps_hi, 25)
                        Rarc = F + l * np.stack([np.cos(arc), np.sin(arc)], 1)
                        r_arc = np.hypot(*(Rarc - O).T)
                        env = r_arc.max() - p.roller_r
                        if 2 * env > cp.max_od or r_arc.min() - p.roller_r < p.shaft_r + 4:
                            continue
                        if np.hypot(*(F - O)) < max(env, lower_outer) + p.standoff_r + 1:
                            continue
                        if geared and np.hypot(*(S - O)) < env + p.module * p.z_hub / 2 + 3:
                            continue
                        if not contact_dirs_ok(Rarc, O, slide, cp.contact_cone):
                            continue
                        if lower and swept_keepout(lower, p)(F) < 1.0:
                            continue
                        # every reference drawing must give a valid cam
                        worst = 0.0
                        ok = True
                        for name, th, M, pen, plan in refs:
                            psi = follower_psi(ax, M=M[:, k]) if geared else follower_psi(ax, pen=pen)
                            ev = periphery_eval(th, O, F, l, psi, p)
                            if not ev["monotone"]:
                                ok = False
                                break
                            r1 = ev["alpha"].max() / p.alpha_max
                            r2 = (p.roller_r + p.min_wall_radius) / max(np.abs(ev["rho"]).min(), 1e-6)
                            r3 = 0.0
                            if geared and p.max_contact:
                                tq = hub_T[k] + np.abs(plan.tau[:, k]).max()
                                r3 = groove_contact(ev, l, G, tq, p) / p.max_contact
                            worst = max(worst, r1, r2, r3)
                            if worst > 1.0:
                                ok = False
                                break
                        if not ok:
                            continue
                        score = env + 40 * max(0.0, worst - 0.85)       # smallest envelope with margin
                        if best is None or score < best["score"]:
                            best = dict(ax, score=score, worst=worst, env=env, psi_lo=ps_lo, psi_hi=ps_hi,
                                        inward=_inward_sign(F, l, psi_ref, O),
                                        psi=np.linspace(ps_lo, ps_hi, 50))    # swept, for keep-outs
        if best is None:
            raise SystemExit(f"no fixed follower for {key} works for all reference drawings "
                             f"{cp.reference}: raise [cartridge].max_od or simplify the references")
        # stop: just beyond the travel on the inward side (rollers rest there when the cams slide out)
        best["psi_stop"] = (best["psi_hi"] + cp.stop_margin) if best["inward"] > 0 else (best["psi_lo"] - cp.stop_margin)
        axes[key] = best
        lower.append(best)
        lower_outer = max(lower_outer, best["env"])
        if verbose:
            print(f"  {key}: l {best['l']:.0f} Rp {best['Rp']:.0f} env OD {2 * best['env']:.0f} worst {best['worst']:.2f}")
    # hub spring direction: follower torque = -G * hub torque must push the roller inward
    signs = tuple(int(-axes[nm]["inward"]) if axes[nm]["zf"] else 1 for nm in ("M1", "M2", "M3"))
    return Followers(p, cp, O, slide, axes, signs)


# -------------------------------------------------------------- cam stack
def compliance_offsets(cs, g, fol):
    """Predicted follower-angle correction (per axis, on the cam grid) that
    cancels the static deflection of the drive: gear-tooth/spoke compliance,
    the backlash flank the springs hold, and roller-contact + follower-arm
    compliance (x3 through the gear).  Loads = re-timed dynamics + pen drag -
    spring preload, i.e. what the gear mesh must actually carry."""
    from .cams import cam_motion
    from .structural import dynamic_case
    from . import springs as spr
    ph = CFG.get("physics", {})
    k_g = ph.get("k_mesh", 300.0)                                   # N m/rad at the hub
    k_c = ph.get("k_contact", 2.0e4)                                # N/m at the roller
    r_hub = fol.params.module * fol.params.z_hub / 2 / 1000
    b2 = CFG["parts"]["backlash"] * 2 / 1000 / r_hub / 2            # half the hub backlash (rad)
    motion = cam_motion(cs, g)
    X, L = dynamic_case(g, motion)
    M = np.unwrap(X["phi"], axis=0) + np.array([0.0, math.pi, g.off_a])
    sp = spr.design(paper_motor_range(g), fol.spring_sign)
    out = {}
    for k, nm in enumerate(("M1", "M2", "M3")):
        ax = fol.axes[nm]
        tau_g = L["tau"][:, k] - spr.hub_spring_torque(sp[nm], M[:, k])      # gear torque on the hub
        f = -tau_g / k_g                                              # mesh spring stretch
        delta = f + np.sign(f) * b2                                   # + backlash flank
        gap = np.abs(ax["G"] * tau_g) / (k_c * (ax["l"] / 1000) ** 2)  # roller sinks into the cam
        off = delta / ax["G"] - ax["inward"] * gap
        # the loads come from finite-differenced accelerations: smooth over a few
        # degrees of cam so the correction does not add sharp features to the rim
        from scipy.ndimage import gaussian_filter1d
        sig = max(1.0, fol.cart.compensate_smooth_deg / 360 * len(off))
        out[nm] = gaussian_filter1d(off, sig, mode="wrap")
    return out


def cams_for(plan, g: Geometry, fol: Followers, tune=True):
    """Cam stack for a drawing on the fixed followers.  Returns a CamSet (so
    parts/structural keep working) plus a list of problems (empty = OK).
    tune: if the default re-timing breaks a limit, retry other re-timing
    settings and keep the stack with the smallest worst-case ratio."""
    best = None
    variants = [(None, None)] + ([(0.85, 8.0), (1.0, 8.0), (0.85, 4.0), (1.0, 4.0), (0.7, 16.0), (1.0, 16.0)]
                                 if tune else [])
    for rt, ar in variants:
        q = fol.params if rt is None else replace(fol.params, retime=rt, retime_accel_ratio=ar)
        cs, probs, worst = _cams_for(plan, g, fol, q)
        if fol.cart.compensate:                                       # second pass with deflection offsets
            cs, probs, worst = _cams_for(plan, g, fol, q, compliance_offsets(cs, g, fol))
        if best is None or worst < best[2]:
            best = (cs, probs, worst)
        if not probs:
            break
    return best[0], best[1]


def _cams_for(plan, g, fol, p, offsets=None):
    worst = 0.0
    th, M, pen, T, plan_time = cam_profiles(plan, p)
    pre = CFG["drive"].get("preload", {})
    hub_T = np.array(pre.get("hub_torque", [0, 0, 0]), float) * (1 + pre.get("torque_variation", 0))
    axes, pen_axis, problems = [None] * 3, None, []
    for key, ax in fol.axes.items():
        k = NAMES.index(key)
        psi = follower_psi(ax, M=M[:, k]) if ax["zf"] else follower_psi(ax, pen=pen)
        if offsets and key in offsets:
            psi = psi + offsets[key]
        ev = periphery_eval(th, fol.O, ax["F"], ax["l"], psi, p)
        des = dict(name=key, F=ax["F"], l=ax["l"], G=ax["G"], zf=ax["zf"], Rp=ax["Rp"], a=ax["a"], psi=psi,
                   score=0, **ev)
        if ax["zf"]:
            des["M_mean"] = float(M[:, k].mean())
            if p.max_contact:
                des["contact"] = groove_contact(ev, ax["l"], ax["G"], hub_T[k] + np.abs(plan.tau[:, k]).max(), p)
                worst = max(worst, des["contact"] / p.max_contact)
                if des["contact"] > p.max_contact:
                    problems.append(f"{key}: rim contact {des['contact']:.0f} MPa > {p.max_contact:.0f}")
        a_max = math.degrees(ev["alpha"].max())
        worst = max(worst, ev["alpha"].max() / p.alpha_max,
                    (p.roller_r + p.min_wall_radius) / max(np.abs(ev["rho"]).min(), 1e-6))
        if ev["alpha"].max() > p.alpha_max:
            problems.append(f"{key}: pressure angle {a_max:.0f} deg > {math.degrees(p.alpha_max):.0f}")
        if np.abs(ev["rho"]).min() < p.roller_r + p.min_wall_radius:
            problems.append(f"{key}: rim curvature radius {np.abs(ev['rho']).min():.1f} mm too tight")
        if not ev["monotone"]:
            problems.append(f"{key}: profile folds over")
        if 2 * ev["outer"] > fol.cart.max_od:
            problems.append(f"{key}: cam OD {2 * ev['outer']:.0f} > carriage {fol.cart.max_od:.0f}")
        if key == "pen":
            pen_axis = des
        else:
            axes[k] = des
    cs = CamSet(p, th, M, pen, T, fol.O, axes, pen_axis, plan_time)
    return cs, problems, worst


def slide_out_check(cs, fol: Followers, n_pts=400, s_max=None, ds=0.5):
    """Slide the cams out along `slide` (cam angle 0).  Each roller follows its
    cam inward until it reaches its stop; from then on the cam must never touch
    it again.  Returns the minimum clearance (mm) after separation (negative =
    a rim comes back into a parked roller), and the slide needed to clear all."""
    p = fol.params
    worst, s_free = 1e9, 0.0
    by_name = {d["name"]: d for d in cs.axes + [cs.pen_axis] if d is not None}
    for key, ax in fol.axes.items():
        des = by_name.get(key)
        if des is None:
            continue
        Rstop = ax["F"] + ax["l"] * np.array([math.cos(ax["psi_stop"]), math.sin(ax["psi_stop"])])
        pc = des["pitch"][:: max(1, len(des["pitch"]) // n_pts)]
        d = np.gradient(pc, axis=0)
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.hypot(*d.T)[:, None]
        a, b = pc + p.roller_r * nrm, pc - p.roller_r * nrm
        rim = (b if np.hypot(*b.T).mean() < np.hypot(*a.T).mean() else a) + fol.O
        span = s_max or 2 * (np.hypot(*(rim - fol.O).T).max() + p.roller_r) + 10
        sv = np.arange(0, span, ds)
        from matplotlib.path import Path
        poly = Path(rim)
        f = np.empty(len(sv))
        for i, s_ in enumerate(sv):
            q = Rstop - s_ * fol.slide                        # roller in the cam's moving frame
            dmin = np.hypot(*(q - rim).T).min() - p.roller_r
            f[i] = -dmin - p.roller_r if poly.contains_point(q) else dmin
        sep = np.flatnonzero(f >= fol.cart.slide_clear)
        if len(sep) == 0:
            worst = min(worst, float(f.max()))
            continue
        i0 = sep[0]
        worst = min(worst, float(f[i0:].min()))
        s_free = max(s_free, float(sv[i0]))
        # never re-touch once the cam has passed the roller completely
    return worst, s_free


def report(fol: Followers, results):
    lines = ["fixed followers (designed once):"]
    for key, ax in fol.axes.items():
        lines.append(f"  {key:4s} pivot ({ax['F'][0]:6.1f},{ax['F'][1]:6.1f})  arm {ax['l']:.0f}  "
                     f"envelope OD {2 * ax['env']:.0f}  worst ref margin {ax['worst']:.2f}  "
                     f"stop at psi {math.degrees(ax['psi_stop']):.1f} deg")
    lines.append(f"  hub spring signs (roller pushed onto the cam): {fol.spring_sign}")
    for name, cs, probs, slide in results:
        ods = " / ".join("--" if d is None else f"{2 * d['outer']:.0f}" for d in cs.axes + [cs.pen_axis])
        lines.append(f"{name:22s} cams OD {ods} mm, slide-out clearance {slide:.1f} mm  "
                     + ("OK" if not probs and slide > 0 else "PROBLEMS: " + "; ".join(probs)
                        + ("" if slide > 0 else "; rollers hit cams when sliding out")))
    return "\n".join(lines)


# ------------------------------------------------------------ cam STL files
def cam_solids(cs, fol: Followers, thickness=None):
    """CadQuery solids of the periphery cams (rim + D-bore + timing notch at
    cam angle 0 + lightening holes), one per axis, flat at z = 0."""
    import cadquery as cq
    from .config import LAYER
    p = fol.params
    th = CFG["parts"]["layer_thickness"]
    axle = CFG["parts"]["axle_bore"]
    out = {}
    for des in [d for d in cs.axes + [cs.pen_axis] if d is not None]:
        nm = des["name"]
        t = thickness or th[LAYER["cam_" + nm]]
        pc = des["pitch"]
        d = np.gradient(pc, axis=0)
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.hypot(*d.T)[:, None]
        a, b = pc + p.roller_r * nrm, pc - p.roller_r * nrm
        rim = b if np.hypot(*b.T).mean() < np.hypot(*a.T).mean() else a
        cam = cq.Workplane("XY").polyline([tuple(q) for q in rim[::2]]).close().extrude(t)
        flat = axle / 2 - 0.8
        bore = (cq.Workplane("XY").circle(axle / 2).extrude(t)
                .cut(cq.Workplane("XY").center(flat + 5, 0).rect(10, 10).extrude(t)))
        cam = cam.cut(bore)
        r_in = np.hypot(*rim.T).min()
        cam = cam.cut(cq.Workplane("XY").center(r_in - 2.5, 0).rect(2, 2).extrude(t))   # timing mark
        hole = (r_in - 12) / 2
        if hole > 4:
            for ang in np.radians([45, 135, 225, 315]):
                c = (10 + hole) * np.array([math.cos(ang), math.sin(ang)])
                cam = cam.cut(cq.Workplane("XY").center(*c).circle(hole * 0.8).extrude(t))
        # engrave the axis name on the top face
        cam = cam.cut(cq.Workplane("XY").workplane(offset=t - 0.6)
                      .center(0, -(r_in * 0.55)).text(nm, 7, 0.6, combine=False))
        out[nm] = cam
    return out


def plot_cartridge(cs, fol: Followers, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = fol.params
    fig, ax = plt.subplots(figsize=(8, 8))
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(*(fol.O[:, None] + fol.cart.max_od / 2 * np.stack([np.cos(t), np.sin(t)])), color="#bbb", ls=":",
            lw=1, label="carriage envelope")
    cols = {"M1": "#2f6db5", "M2": "#1aa3a3", "M3": "#8e55c9", "pen": "#e08a1e"}
    for des in [d for d in cs.axes + [cs.pen_axis] if d is not None]:
        nm = des["name"]
        pc = des["pitch"]
        d = np.gradient(pc, axis=0)
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.hypot(*d.T)[:, None]
        a, b = pc + p.roller_r * nrm, pc - p.roller_r * nrm
        rim = (b if np.hypot(*b.T).mean() < np.hypot(*a.T).mean() else a) + fol.O
        ax.plot(*np.vstack([rim, rim[:1]]).T, color=cols[nm], lw=1.2, label=f"cam {nm}")
        axd = fol.axes[nm]
        R0 = des["R"][0]
        ax.plot(*np.array([axd["F"], R0]).T, color=cols[nm], lw=3, alpha=0.5)
        ax.add_patch(plt.Circle(R0, p.roller_r, fill=False, color=cols[nm]))
        ax.plot(*axd["F"], "o", color=cols[nm])
    ax.arrow(*fol.O, *(40 * fol.slide), width=1.5, color="#c62828")
    ax.text(*(fol.O + 48 * fol.slide), "slide out", color="#c62828")
    ax.plot(0, 0, "k+", ms=12)
    ax.text(3, 3, "shoulder S")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(title or "cam cartridge (cam angle 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
