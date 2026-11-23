"""
drum.py - multi-turn cam drum: one motor, a much longer drawing.

    python -m drawarm drum [drawing]   -> report, output/drum.stl, output/drum_<name>.png

Mechanism
    A vertical drum at the camshaft position O turns `turns` times per drawing
    and rises one lead-screw lead (`pitch`) per turn (drum on a lead screw
    running in a fixed nut; the motor drives it through a sliding key/spline).
    Each follower (the same geared followers as the cam drive, same layers)
    carries a thin FINGER (separate flat part, dowel-keyed to the follower on
    the same pivot, just below it) with the roller underneath; the roller
    rests on a raised helical BAND at that height.  Only the finger and roller
    enter the drum envelope, and they stay inside their own axial slot, so
    ribs of the other bands (at other heights mod the lead) never touch them.  Seen
    from the follower's fixed plane, a band is a plate cam unrolled over
    `turns` revolutions, so the synthesis is exactly the cam synthesis with
    theta running 0 .. 2*pi*turns.  The bands are interleaved like the starts
    of a multi-start thread; the pitch is picked so neighbouring bands never
    overlap at any follower.
    The anti-backlash torsion springs on the shoulder hubs keep every roller
    pressed on its band, so the bands are single-sided ribs, not grooves.
    Default contact = "below_base": the drum hangs in a bay under the base
    plate; each follower becomes a rocker on a steel 8 mm D-shaft through the
    base plate (follower sector on top, finger + roller at the bottom), so the
    rollers stack evenly one band slot apart.  contact = "cam_layers" puts the
    fingers in the empty cam layers instead (no rocker shafts, longer lead).
    End of drawing: reverse the motor (the arm retraces the drawing) or use a
    half-nut + spring to drop the drum back (followers lifted).

Capacity: track per band = 2*pi*R*turns instead of 2*pi*R.
"""
import math
import struct
from dataclasses import dataclass, field

import numpy as np

from .cams import DriveParams, cam_profiles, swept_keepout, synth_axis
from .config import CFG, LAYER
from .kinematics import Geometry

_DM = CFG.get("drum", {})


@dataclass
class DrumParams:
    """Defaults come from params.toml [drum] (all optional)."""
    turns: int = _DM.get("turns", 4)
    n_per_turn: int = _DM.get("n_per_turn", 720)
    pitch: float = _DM.get("pitch", 0.0)                  # lead per turn (0 = smallest that fits)
    pitch_options: tuple = tuple(_DM.get("pitch_options", [8, 10, 12, 14, 16, 20, 24, 30, 36, 40]))
    band_clearance: float = _DM.get("band_clearance", 1.0)   # axial gap between a finger and a rib
    finger_t: float = _DM.get("finger_t", 3.0)            # finger plate thickness
    finger_gap: float = _DM.get("finger_gap", 0.5)        # finger <-> roller
    contact: str = _DM.get("contact", "below_base")       # "below_base" (rocker shafts) or "cam_layers"
    base_gap: float = _DM.get("base_gap", 1.5)            # drum core below the lowest band surface
    end_margin: float = _DM.get("end_margin", 5.0)        # extra drum length at each end


@dataclass
class DrumSet:
    """Duck-types CamSet (params, theta, motors, pen, period, O, axes, pen_axis)
    so parts.py can build the followers; plus drum geometry."""
    params: DriveParams
    drum: DrumParams
    theta: np.ndarray
    motors: np.ndarray
    pen: np.ndarray
    period: float
    O: np.ndarray
    axes: list = field(default_factory=list)
    pen_axis: dict = None
    plan_time: np.ndarray = None
    pitch: float = 0.0
    z_contact: dict = field(default_factory=dict)     # band name -> follower contact height (mm)
    R_outer: float = 0.0
    R_core: float = 0.0
    length: float = 0.0
    z_bottom: float = 0.0                              # drum bottom at theta = 0 (frame z)


# ------------------------------------------------------------- band layout
def contact_heights(p: DriveParams, dp: "DrumParams" = None):
    """Height of each roller's mid-plane: follower plate, gap, finger, gap, roller."""
    dp = dp or DrumParams()
    th, gap = CFG["parts"]["layer_thickness"], CFG["parts"]["gap"]
    z0 = lambda L: sum(th[i] + gap for i in range(L))
    return {nm: z0(LAYER["follow_" + nm]) - gap - dp.finger_t - dp.finger_gap - p.roller_w / 2
            for nm in ("M1", "M2", "M3", "pen")}


def contact_ranges(p: DriveParams, dp: "DrumParams" = None, step=0.5):
    """Allowed roller mid-plane heights per axis.  In drum mode the cam layers
    (and layer 4) are empty on the drive side, so a finger + roller stack may
    sit in any of those layers below its follower plate (the finger reaches its
    follower through a sleeve on the pivot pin, dowel-keyed)."""
    dp = dp or DrumParams()
    th, gap = CFG["parts"]["layer_thickness"], CFG["parts"]["gap"]
    z0 = lambda L: sum(th[i] + gap for i in range(L))
    follower_layers = {LAYER["follow_" + nm] for nm in ("M1", "M2", "M3", "pen")}
    stack_up = p.roller_w / 2 + dp.finger_gap + dp.finger_t
    out = {}
    for nm in ("M1", "M2", "M3", "pen"):
        fl = LAYER["follow_" + nm]
        vals = []
        for L in range(fl):
            if L in follower_layers:
                continue
            lo = z0(L) - gap / 2 + p.roller_w / 2
            hi = z0(L + 1) - gap / 2 - stack_up
            vals += list(np.arange(lo, hi + 1e-9, step))
        out[nm] = sorted(set(round(v, 3) for v in vals))
    return out


def place_bands(cands, band_w, options, fixed=0.0):
    """Smallest lead (and a roller height per axis from its candidates) such
    that every pair of rollers is on its own band: circular distance of their
    heights mod lead >= band_w.  Depth-first search, nearest-to-follower first."""
    names = list(cands)
    order = {nm: sorted(v, reverse=True) for nm, v in cands.items()}       # prefer high (short sleeves)
    fine = list(np.arange(band_w * len(names), 150.0, 0.5))

    def dfs(i, chosen, pch):
        if i == len(names):
            return chosen
        for zv in order[names[i]]:
            if all(min((zv - c) % pch, pch - (zv - c) % pch) >= band_w for c in chosen.values()):
                r = dfs(i + 1, {**chosen, names[i]: zv}, pch)
                if r:
                    return r
        return None

    for pch in ([fixed] if fixed else sorted(options) + fine):
        if pch < band_w * len(names):
            continue
        r = dfs(0, {}, pch)
        if r:
            return float(pch), {k: float(v) for k, v in r.items()}
    return None, {nm: float(v[-1]) for nm, v in order.items()}


def slot_width(p: DriveParams, dp: "DrumParams"):
    """Axial spacing two bands need: rib half-width + roller/finger stack + clearance."""
    rib_half = p.roller_w / 2 + 0.5
    stack_up = p.roller_w / 2 + dp.finger_gap + dp.finger_t          # above the roller mid-plane
    return rib_half + stack_up + dp.band_clearance


def choose_pitch(z, band_w, options, fixed=0.0):
    """Smallest lead such that every pair of followers sits on different
    bands: circular distance of their heights (mod pitch) >= band width."""
    vals = list(z.values())
    fine = list(np.arange(band_w * len(vals), 150.0, 0.5))   # custom (printed) lead screw
    cands = [fixed] if fixed else sorted(options) + fine
    for pch in cands:
        ok = True
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                d = (vals[i] - vals[j]) % pch
                if min(d, pch - d) < band_w:
                    ok = False
        if ok and pch >= band_w * len(vals):
            return pch
    return None


def _seg_dist(q, a, b):
    ab = b - a
    t = np.clip(np.dot(q - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.hypot(*(q - (a + t * ab))))


def finger_order(designs, finger_r=6.0, shaft_r=4.5):
    """Order of the fingers in the drum bay (top -> bottom) so that no rocker
    shaft running down to a deeper finger crosses a shallower finger's swing.
    designs: {name: design}.  Returns (order, min clearance)."""
    import itertools
    names = list(designs)
    paths = {}
    for nm, d in designs.items():
        idx = np.linspace(0, len(d["psi"]) - 1, 60).astype(int)
        paths[nm] = [(d["F"], d["F"] + d["l"] * np.array([math.cos(d["psi"][i]), math.sin(d["psi"][i])]))
                     for i in idx]
    best = None
    for order in itertools.permutations(names):
        worst = 1e9
        for i, k in enumerate(order):                    # finger k at depth i
            for j in order[i + 1:]:                       # shafts of deeper fingers pass it
                Fj = designs[j]["F"]
                worst = min(worst, min(_seg_dist(Fj, a, b) for a, b in paths[k]) - finger_r - shaft_r)
        if best is None or worst > best[1]:
            best = (list(order), worst)
    return best


# ----------------------------------------------------------------- design
def design_drum(plan, g: Geometry, p: DriveParams = None, dp: DrumParams = None, passes=3):
    from dataclasses import replace
    p = p or DriveParams()
    p = replace(p, pen_pivot_step=_DM.get("pen_pivot_step", 10.0),
                pen_ramp_deg=_DM.get("pen_ramp_deg", 3 * p.pen_ramp_deg))   # drum degrees are cheap
    dp = dp or DrumParams()
    theta, M, pen, T, plan_time = cam_profiles(plan, p, turns=dp.turns, n_per_turn=dp.n_per_turn)
    S = np.zeros(2)
    O = p.D_SO * np.array([math.cos(p.gamma_O), math.sin(p.gamma_O)])
    names = {"M1": (0, "M1 shoulder"), "M2": (1, "M2 elbow"), "M3": (2, "M3 wrist")}
    pre = CFG["drive"].get("preload", {})
    hub_pre = np.array(pre.get("hub_torque", [0, 0, 0]), float) * (1 + pre.get("torque_variation", 0))
    design_torque = hub_pre + (np.abs(plan.tau).max(0) if hasattr(plan, "tau") else 0)

    R_req = 0.0                                    # every pivot must clear the whole drum
    axes, pen_axis = [None] * 3, None
    for _ in range(passes):
        axes, pen_axis, lower = [None] * 3, None, []
        for key in p.cam_stack:
            if key == "pen":
                rel = (1 - pen)
                des = synth_axis(theta, rel - rel.mean(), p, S, O, False, "pen lift band",
                                 min_pivot=R_req, keepout=swept_keepout(lower, p), body_r=R_req)
                pen_axis = des
            else:
                k, name = names[key]
                rel = M[:, k] - M[:, k].mean()
                des = synth_axis(theta, rel, p, S, O, True, name + " band", min_pivot=R_req,
                                 keepout=swept_keepout(lower, p), hub_torque=design_torque[k], body_r=R_req)
                if des is not None:
                    des["M_mean"] = float(M[:, k].mean())
                axes[k] = des
            if des is not None:
                lower.append(des)
        found = [d for d in axes + [pen_axis] if d is not None]
        R_new = max((d["r_max"] - p.roller_r + 1.0 for d in found), default=0.0)   # band surface max
        if R_new <= R_req + 0.5:
            break
        R_req = R_new

    band_w = slot_width(p, dp)
    if dp.contact == "below_base":
        # drum bay under the base plate: rollers evenly stacked, one band slot apart
        need = band_w * len(p.cam_stack)
        pitch = dp.pitch or min([o for o in dp.pitch_options if o >= need] or [need])
        # the drum rises `turns * pitch` while drawing: keep its top below the base plate
        top = -CFG["parts"]["gap"] - CFG["parts"]["base_thickness"] - 3.0 - dp.turns * pitch - dp.end_margin
        named = {nm: d for nm, d in zip(("M1", "M2", "M3", "pen"), axes + [pen_axis]) if d is not None}
        order, _ = finger_order(named) if named else (list(p.cam_stack[::-1]), 0)
        order += [nm for nm in p.cam_stack if nm not in order]
        z = {nm: top - (i + 0.5) * band_w for i, nm in enumerate(order)}
    else:
        pitch, z = place_bands(contact_ranges(p, dp), band_w, dp.pitch_options, dp.pitch)
    found = [d for d in axes + [pen_axis] if d is not None]
    r_surf_min = min((d["r_min"] - p.roller_r for d in found), default=10.0)
    R_core = max(p.shaft_r + 4, r_surf_min - dp.base_gap)
    R_outer = max((d["r_max"] - p.roller_r for d in found), default=R_core)
    travel = dp.turns * (pitch or 0)
    zmin, zmax = min(z.values()), max(z.values())
    z_bottom = zmin - travel - band_w / 2 - dp.end_margin     # drum rises by `travel` while drawing
    length = (zmax + band_w / 2 + dp.end_margin) - z_bottom
    return DrumSet(p, dp, theta, M, pen, T, O, axes, pen_axis, plan_time, pitch or 0.0,
                   z, R_outer, R_core, length, z_bottom)


def report(ds: DrumSet):
    lines = [f"drum: {ds.drum.turns} turns per drawing, {ds.period:.1f} s "
             f"({60 * ds.drum.turns / ds.period:.1f} rpm), re-timing {ds.params.retime:g}",
             (f"  lead {ds.pitch:g} mm/turn (travel {ds.drum.turns * ds.pitch:.0f} mm), "
              f"surface radius {ds.R_core:.0f}-{ds.R_outer:.0f} mm (OD {2 * ds.R_outer:.0f}), "
              f"length {ds.length:.0f} mm, drum bay {-ds.z_bottom:.0f} mm deep below the arm") if ds.pitch else
             "  NO lead fits: followers too close in height for the band width (raise pitch_options)"]
    for a in ds.axes + [ds.pen_axis]:
        if a is None:
            lines.append("  (band infeasible: drawing too busy even for this many turns)")
            continue
        lines.append(f"  {a['name']:18s} track {2 * math.pi * a['Rp'] * ds.drum.turns / 1000:4.2f} m  "
                     f"ratio {a['G']:.2f}  arm {a['l']:.0f}  pitch r {a['Rp']:.0f}  "
                     f"max pressure {math.degrees(a['alpha'].max()):4.1f} deg  min rho {np.abs(a['rho']).min():.1f}")
    return "\n".join(lines)


# ------------------------------------------------------------------- mesh
def _band_surface(des, p):
    """Rib-top curve in the drum frame: pitch curve offset inward by the roller."""
    pc = des["pitch"]
    d = np.gradient(pc, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.hypot(*d.T)[:, None]
    a, b = pc + p.roller_r * nrm, pc - p.roller_r * nrm
    return b if np.hypot(*b.T).mean() < np.hypot(*a.T).mean() else a


def drum_triangles(ds: DrumSet, n_core=180):
    """Triangles (k,3,3) of the drum: core cylinder + one helical rib per band.
    Shells overlap (rib roots sit inside the core); slicers union them."""
    p, dp = ds.params, ds.drum
    tris = []

    def quad(a, b, c, d):
        tris.append((a, b, c))
        tris.append((a, c, d))

    # core cylinder with an 8 mm bore (drum frame: z from 0 to length)
    L, Rc, Rb = ds.length, ds.R_core, CFG["parts"]["axle_bore"] / 2
    ang = np.linspace(0, 2 * np.pi, n_core + 1)
    for i in range(n_core):
        a0, a1 = ang[i], ang[i + 1]
        po = lambda r, a, z: (r * math.cos(a), r * math.sin(a), z)
        quad(po(Rc, a0, 0), po(Rc, a1, 0), po(Rc, a1, L), po(Rc, a0, L))          # outer
        quad(po(Rb, a1, 0), po(Rb, a0, 0), po(Rb, a0, L), po(Rb, a1, L))          # bore
        quad(po(Rb, a0, 0), po(Rb, a1, 0), po(Rc, a1, 0), po(Rc, a0, 0))          # bottom ring
        quad(po(Rc, a0, L), po(Rc, a1, L), po(Rb, a1, L), po(Rb, a0, L))          # top ring
    # ribs
    half = p.roller_w / 2 + 0.5
    for nm, des in zip(("M1", "M2", "M3", "pen"), ds.axes + [ds.pen_axis]):
        if des is None:
            continue
        top = _band_surface(des, p)
        r_out = np.hypot(*top.T)
        phi = np.arctan2(top[:, 1], top[:, 0])
        # contact height in the drum frame drops by `pitch` per turn as the drum rises
        zc = ds.z_contact[nm] - ds.z_bottom - ds.pitch * ds.theta / (2 * np.pi)
        r_in = Rc - 1.0
        ring = []
        for k in range(len(top)):
            c, s_ = math.cos(phi[k]), math.sin(phi[k])
            ring.append([(r_in * c, r_in * s_, zc[k] - half), (r_out[k] * c, r_out[k] * s_, zc[k] - half),
                         (r_out[k] * c, r_out[k] * s_, zc[k] + half), (r_in * c, r_in * s_, zc[k] + half)])
        for k in range(len(ring) - 1):
            A, B = ring[k], ring[k + 1]
            for j in range(4):
                quad(A[j], B[j], B[(j + 1) % 4], A[(j + 1) % 4])
        quad(*ring[0][::-1])                                                     # end caps
        quad(*ring[-1])
    return np.array(tris, float)


def write_stl(tris, path):
    with open(path, "wb") as f:
        f.write(b"drawarm drum".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            n = n / (np.linalg.norm(n) + 1e-12)
            f.write(struct.pack("<12fH", *n, *t[0], *t[1], *t[2], 0))


def plot_drum(ds: DrumSet, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = ds.params
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for nm, des in zip(("M1", "M2", "M3", "pen"), ds.axes + [ds.pen_axis]):
        if des is None:
            continue
        r = np.hypot(*_band_surface(des, p).T)
        ax[0].plot(np.degrees(ds.theta) / 360, r, lw=0.9, label=nm)
        zc = ds.z_contact[nm] - ds.z_bottom - ds.pitch * ds.theta / (2 * np.pi)
        ax[1].plot((np.degrees(ds.theta) % 360), zc, ".", ms=0.6, label=nm)
    ax[0].set_xlabel("drum turns")
    ax[0].set_ylabel("band surface radius (mm)")
    ax[0].set_title("band profiles")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("drum angle (deg)")
    ax[1].set_ylabel("height on drum (mm)")
    ax[1].set_title(f"band layout (unrolled), lead {ds.pitch:g} mm")
    ax[1].set_ylim(0, ds.length)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
