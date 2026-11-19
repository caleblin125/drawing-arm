"""
parts.py - parametric 3D-printable parts for the drawing arm + cam drive (CadQuery).

    python parts.py [shape|file.svg]     -> cad/stl/*.stl, cad/assembly.step, cad/preview.png

Everything is generated from Geometry (arm_kinematics) and the CamSet
(cam_drive), so changing a link length or the drawing regenerates the parts.

Layer stack (each plate T thick, layer pitch P = T + washer gap), bottom -> top:

  layer  arm side                          drive side (behind the shoulder)
  -----  --------------------------------  ---------------------------------
   0                                       cam M2
   1     M2 elbow crank (hub on S axle)    follower M2 (sector gear + roller)
   2     coupler E  (C2-T2)                cam M1
   3     upper arm (hub on S axle)         follower M1
   4     forearm (tail T2, E, W)
   5     bell-crank arm B, hand + crank    cam M3
   6     coupler B  (B1-B2), M3 gear (hub) follower M3
   7     bell-crank arm A, M3 crank (hub)  cam pen-lift
   8     coupler A  (A1-A2)                follower pen-lift (cable lever)
  base plate below layer 0 carries the S axle, camshaft and follower pivots.

Every pin joins ADJACENT layers except T2 (layers 2-4, clears the upper arm
by >= 20 mm thanks to the 20 deg elbow limit), and the bell-crank hub that
spans 5-7 (keyed to the arm-A plate with a dowel).  The M3 hub is split the
same way: gear sector on layer 6, crank on layer 7, keyed by a dowel, so the
follower M3 sector never meets the crank arm.
Hardware: 8 mm rod (shoulder axle, camshaft), 4 mm pins/shoulder bolts,
MR74ZZ bearings (4x7x2.5) as cam rollers, M3 set screw for the pen.
"""
import math
import os
import sys

import numpy as np
import cadquery as cq

from arm_kinematics import Geometry, Linkage
from cam_drive import DriveParams, design_cams, plan_for_cams, report, plot_cams

T = 5.0            # plate thickness
GAP = 1.0          # washer gap
P = T + GAP        # layer pitch
PIN = 4.3          # 4 mm pin clearance bore
AXLE = 8.25        # 8 mm rod clearance bore
FIT_PIN = 3.95     # press fit for 4 mm pins (roller studs)
W_LINK = 14.0      # link width
BACKLASH = 0.15    # mm, gear tooth thinning

LAYER = dict(cam_M2=0, crank_M2=1, follow_M2=1, coupler_E=2, cam_M1=2, upper_arm=3, follow_M1=3,
             forearm=4, bell_B=5, hand=5, cam_M3=5, coupler_B=6, gear_M3=6, follow_M3=6,
             bell_A=7, crank_M3=7, cam_pen=7, coupler_A=8, follow_pen=8)


def z0(layer):
    return layer * P


# ------------------------------------------------------------------ 2D helpers
def v2(p):
    return np.asarray(p, float)


def ang(a, b):
    d = v2(b) - v2(a)
    return math.atan2(d[1], d[0])


def slot_link(a, b, ra, rb, t=T):
    """Stadium/hull of two circles (a link body)."""
    a, b = v2(a), v2(b)
    d = b - a
    L = float(np.hypot(*d))
    th = math.atan2(d[1], d[0])
    pts = []
    for k in np.linspace(math.pi / 2, 3 * math.pi / 2, 24):
        pts.append(a + ra * np.array([math.cos(th + k), math.sin(th + k)]))
    for k in np.linspace(-math.pi / 2, math.pi / 2, 24):
        pts.append(b + rb * np.array([math.cos(th + k), math.sin(th + k)]))
    return cq.Workplane("XY").polyline([tuple(p) for p in pts]).close().extrude(t)


def holes(wp, pts, d, t=T):
    for p in pts:
        wp = wp.cut(cq.Workplane("XY").center(*p).circle(d / 2).extrude(t))
    return wp


def disc(c, r, t=T):
    return cq.Workplane("XY").center(*c).circle(r).extrude(t)


# ------------------------------------------------------------------ gears
def _inv(a):
    return math.tan(a) - a


def gear_sector_pts(center, m, z, a_center, half_span, gap_on_center=False, inner_r=None,
                    anchor=None, pa=math.radians(20)):
    """Involute spur-gear SECTOR polygon: teeth whose centres fall within
    a_center +- half_span, phased so a tooth (or a gap) sits on `anchor`
    (the mesh line in the assembled pose)."""
    anchor = a_center if anchor is None else anchor
    c = v2(center)
    rp = m * z / 2
    rb = rp * math.cos(pa)
    ra = rp + m
    rf = rp - 1.25 * m
    inner_r = inner_r if inner_r is not None else max(rf - 2.5 * m, 1)
    pitch = 2 * math.pi / z
    half_tooth_p = math.pi / (2 * z) - BACKLASH / (2 * rp)
    radii = np.linspace(max(rb, rf), ra, 8)

    def half_angle(r):
        al = math.acos(min(1, rb / r))
        return half_tooth_p + _inv(pa) - _inv(al)

    off = pitch / 2 if gap_on_center else 0.0
    n = int((half_span + abs(a_center - anchor)) / pitch) + 2
    centers = [anchor + off + k * pitch for k in range(-n, n + 1)]
    centers = [a for a in centers if abs(a - a_center) <= half_span + 1e-9]
    pol = lambda r, a: tuple(c + r * np.array([math.cos(a), math.sin(a)]))
    pts = []
    a_start = centers[0] - pitch / 2
    pts.append(pol(inner_r, a_start))
    pts.append(pol(rf, a_start))
    for tc in centers:
        right = [pol(r, tc - half_angle(r)) for r in radii]
        left = [pol(r, tc + half_angle(r)) for r in radii[::-1]]
        root_r = [pol(rf, tc - half_angle(radii[0]) - 0.02)] if rf < rb else []
        root_l = [pol(rf, tc + half_angle(radii[0]) + 0.02)] if rf < rb else []
        pts += root_r + right + left + root_l
    a_end = centers[-1] + pitch / 2
    pts.append(pol(rf, a_end))
    pts.append(pol(inner_r, a_end))
    # inner arc back
    for a in np.linspace(a_end, a_start, 12)[1:-1]:
        pts.append(pol(inner_r, a))
    return pts


def gear_sector(center, m, z, a_center, half_span, gap_on_center=False, inner_r=None, t=T, anchor=None):
    pts = gear_sector_pts(center, m, z, a_center, half_span, gap_on_center, inner_r, anchor)
    return cq.Workplane("XY").polyline(pts).close().extrude(t)


def paper_motor_range(g):
    """Motor angle range needed to reach the whole paper (any drawing)."""
    from arm_kinematics import solve_ik
    from arm_linkage import PAPER
    xs = np.linspace(PAPER[0], PAPER[2], 14)
    ys = np.linspace(PAPER[1], PAPER[3], 12)
    M = []
    for x in xs:
        for y in ys:
            s = solve_ik(g, (x, y))
            if s:
                M.append(s["inputs"])
    M = np.unwrap(np.array(M), axis=0)
    return list(zip(M.min(0), M.max(0)))


# ================================================================== parts
class Builder:
    def __init__(self, g: Geometry, cs):
        self.g, self.cs = g, cs
        self.p = cs.params
        M0 = cs.motors[0]                          # assembled pose = cam angle 0
        self.M_paper = paper_motor_range(g)        # gears cover the whole paper (reusable)
        self.st = Linkage(g).solve(*M0)
        self.M0 = M0
        self.parts = {}                            # name -> (solid in world, layer, colour)

    def add(self, name, solid, layer, color):
        self.parts[name] = (solid.translate((0, 0, z0(layer))), layer, color)

    # ---------------------------------------------------------- arm
    def hub_sector(self, axis_idx, solid):
        """Add the hub gear sector that meshes with follower axis_idx."""
        des = self.cs.axes[axis_idx]
        S = np.zeros(2)
        a_mesh = ang(S, des["F"])
        lo, hi = self.M_paper[axis_idx]
        lo, hi = min(lo, self.cs.motors[:, axis_idx].min()), max(hi, self.cs.motors[:, axis_idx].max())
        span = (hi - lo) / 2 + math.radians(12)
        # world angle of hub material at the mesh = a_mesh - (M - M0)
        dM = np.array([lo, hi]) - self.M0[axis_idx]
        centre = a_mesh - (dM.max() + dM.min()) / 2
        sec = gear_sector(S, self.p.module, self.p.z_hub, centre, span, gap_on_center=False,
                          inner_r=4.0, anchor=a_mesh)
        # tooth must sit centred on the mesh line in the assembled pose
        return solid.union(sec)

    def build_arm(self):
        g, st = self.g, self.st
        S, E, W, Pp = (v2(st[k]) for k in ("S", "E", "W", "P"))

        crank2 = slot_link(S, st["C2"], 10, 7).union(disc(S, 10))
        crank2 = self.hub_sector(1, crank2) if self.cs.axes[1] else crank2
        crank2 = holes(crank2, [S], AXLE)
        crank2 = holes(crank2, [st["C2"]], PIN)
        self.add("crank_M2", crank2, LAYER["crank_M2"], "#1aa3a3")

        cE = holes(slot_link(st["C2"], st["T2"], 6, 6), [st["C2"], st["T2"]], PIN)
        self.add("coupler_E", cE, LAYER["coupler_E"], "#7fd6d6")

        ua = slot_link(S, E, 11, 9).union(disc(S, 11))
        ua = self.hub_sector(0, ua) if self.cs.axes[0] else ua
        ua = holes(ua, [S], AXLE)
        ua = holes(ua, [E], PIN)
        # lightening slot
        L = np.hypot(*(E - S))
        u = (E - S) / L
        ua = ua.cut(slot_link(S + u * 28, E - u * 28, 3, 3))
        self.add("upper_arm", ua, LAYER["upper_arm"], "#2f6db5")

        fa = slot_link(st["T2"], E, 7, 9).union(slot_link(E, W, 9, 7.5))
        fa = holes(fa, [st["T2"], E, W], PIN)
        u = (W - E) / np.hypot(*(W - E))
        fa = fa.cut(slot_link(E + u * 26, W - u * 26, 2.5, 2.5))
        self.add("forearm", fa, LAYER["forearm"], "#2f9e57")

        # bell-crank: lower plate (arm B) with hub boss up through layer 6,
        # upper plate (arm A) keyed by a 3 mm dowel
        dow = E + 6.5 * np.array([math.cos(ang(E, st["A2"]) + 1.0), math.sin(ang(E, st["A2"]) + 1.0)])
        bl = slot_link(E, st["B1"], 9, 6).union(disc(E, 9, 2 * P))
        bl = holes(bl, [E, st["B1"]], PIN, 2 * P)
        bl = holes(bl, [dow], 2.9, 2 * P)
        self.add("bell_crank_lower", bl, LAYER["bell_B"], "#8e55c9")
        bu = slot_link(E, st["A2"], 9, 6)
        bu = holes(bu, [E, st["A2"]], PIN)
        bu = holes(bu, [dow], 3.1)
        self.add("bell_crank_upper", bu, LAYER["bell_A"], "#8e55c9")

        cB = holes(slot_link(st["B1"], st["B2"], 5.5, 5.5), [st["B1"], st["B2"]], PIN)
        self.add("coupler_B", cB, LAYER["coupler_B"], "#c7a8ea")

        dow3 = S + 6.5 * np.array([math.cos(ang(S, st["A1"])), math.sin(ang(S, st["A1"]))])
        crank3 = slot_link(S, st["A1"], 10, 6).union(disc(S, 10))
        crank3 = holes(crank3, [S], AXLE)
        crank3 = holes(crank3, [st["A1"]], PIN)
        crank3 = holes(crank3, [dow3], 3.1)
        self.add("crank_M3", crank3, LAYER["crank_M3"], "#8e55c9")
        gear3 = disc(S, 10)
        gear3 = self.hub_sector(2, gear3) if self.cs.axes[2] else gear3
        gear3 = holes(gear3, [S], AXLE)
        gear3 = holes(gear3, [dow3], 2.9)
        self.add("gear_M3", gear3, LAYER["gear_M3"], "#8e55c9")

        cA = holes(slot_link(st["A1"], st["A2"], 5.5, 5.5), [st["A1"], st["A2"]], PIN)
        self.add("coupler_A", cA, LAYER["coupler_A"], "#c7a8ea")

        # hand: wrist boss + crank arm to B2 + pen holder at P (pen points down)
        hand = slot_link(W, Pp, 8, 9).union(slot_link(W, st["B2"], 8, 6))
        hand = hand.union(disc(Pp, 9, T + 6))                       # taller pen collar
        hand = holes(hand, [W, st["B2"]], PIN)
        hand = holes(hand, [Pp], 10.4, T + 6)                       # ~10 mm marker
        u = (Pp - W) / np.hypot(*(Pp - W))
        setscrew = (cq.Workplane("YZ").circle(1.3).extrude(12)
                    .rotate((0, 0, 0), (0, 0, 1), math.degrees(math.atan2(u[1], u[0])))
                    .translate((Pp[0], Pp[1], T / 2 + 3)))
        hand = hand.cut(setscrew)
        self.add("hand", hand, LAYER["hand"], "#e08a1e")

        # shoulder axle spacers (fill layers without a hub)
        for name, lay, n in (("spacer_S_2", 2, 1), ("spacer_S_45", 4, 2)):
            sp = disc(S, 7, n * P - GAP).cut(disc(S, AXLE / 2, n * P))
            self.add(name, sp.translate((0, 0, -GAP / 2 + GAP / 2)), lay, "#bbbbbb")

    # ---------------------------------------------------------- drive
    def build_drive(self):
        cs, p = self.cs, self.p
        O = cs.O
        lay = {0: ("M1", "cam_M1", "follow_M1"), 1: ("M2", "cam_M2", "follow_M2"),
               2: ("M3", "cam_M3", "follow_M3")}
        designs = [(k, cs.axes[k]) + lay[k] for k in range(3)] + [(3, cs.pen_axis, "pen", "cam_pen", "follow_pen")]
        for k, des, nm, cam_l, fol_l in designs:
            if des is None:
                continue
            # ---- cam (groove on top face), D-flat bore, timing notch at theta = 0
            pc = des["pitch"]
            d = np.gradient(pc, axis=0)
            nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.hypot(*d.T)[:, None]
            outer_w = pc + p.roller_r * nrm
            inner_w = pc - p.roller_r * nrm
            cam = disc((0, 0), des["outer"])
            groove = (cq.Workplane("XY").polyline([tuple(q) for q in outer_w[::2]]).close().extrude(3.2)
                      .cut(cq.Workplane("XY").polyline([tuple(q) for q in inner_w[::2]]).close().extrude(3.2)))
            cam = cam.cut(groove.translate((0, 0, T - 3.2)))
            bore = disc((0, 0), AXLE / 2).cut(cq.Workplane("XY").center(AXLE / 2 - 0.6 + 2, 0).rect(4, 10).extrude(T))
            cam = cam.cut(bore)
            cam = cam.cut(cq.Workplane("XY").center(des["outer"], 0).rect(3, 3).extrude(T))   # timing notch
            # lightening holes inside the groove
            r_hole = (des["inner"] - 7) / 2
            if r_hole > 4:
                for a in np.arange(0, 360, 90) + 45:
                    cpt = (7 + r_hole) * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
                    cam = cam.cut(disc(cpt, r_hole * 0.8))
            self.add(f"cam_{nm}", cam.translate((O[0], O[1], 0)), LAYER[cam_l], "#666666")

            # ---- follower: pivot boss, arm to roller, sector gear toward S
            F = des["F"]
            psi0 = des["psi"][0]
            R0 = F + des["l"] * np.array([math.cos(psi0), math.sin(psi0)])
            fol = slot_link(F, R0, 7, 6)
            if k < 3:
                S = np.zeros(2)
                a_mesh = ang(F, S)
                lo, hi = self.M_paper[k]
                lo = min(lo, cs.motors[:, k].min())
                hi = max(hi, cs.motors[:, k].max())
                dpsi = -(np.array([lo, hi]) - self.M0[k]) / des["G"]
                span = abs(dpsi[1] - dpsi[0]) / 2 + math.radians(6)
                centre = a_mesh - (dpsi.max() + dpsi.min()) / 2
                rs = p.module * des["zf"] / 2
                # spoke from pivot to sector (stops short of the mesh)
                fol = fol.union(slot_link(F, F + (rs - 8) * np.array([math.cos(centre), math.sin(centre)]), 7, 4))
                fol = fol.union(gear_sector(F, p.module, des["zf"], centre, span, gap_on_center=True,
                                            inner_r=rs - 5, anchor=a_mesh))
            else:
                # cable lever: extend beyond the pivot, cable hole at the end
                tip = F - 45 * np.array([math.cos(psi0), math.sin(psi0)])
                fol = fol.union(slot_link(F, tip, 7, 5))
                fol = holes(fol, [tip], 2.0)
            fol = holes(fol, [F], PIN)
            fol = holes(fol, [R0], FIT_PIN)                          # roller stud (pin + MR74 below)
            self.add(f"follower_{nm}", fol, LAYER[fol_l], "#9e9e9e")

    def build_base(self):
        cs = self.cs
        pts = [np.zeros(2), cs.O] + [a["F"] for a in cs.axes + [cs.pen_axis] if a is not None]
        hull = []
        for q in pts:
            for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                hull.append(q + 16 * np.array([math.cos(a), math.sin(a)]))
        hull = np.array(hull)
        from scipy.spatial import ConvexHull
        h = hull[ConvexHull(hull).vertices]
        base = cq.Workplane("XY").polyline([tuple(q) for q in h]).close().extrude(6)
        base = holes(base, [np.zeros(2), cs.O], AXLE, 6)
        base = holes(base, [a["F"] for a in cs.axes + [cs.pen_axis] if a is not None], PIN, 6)
        self.parts["base_plate"] = (base.translate((0, 0, -6 - GAP)), -1, "#d7ccc8")

    def build(self):
        self.build_arm()
        self.build_drive()
        self.build_base()
        return self.parts


# ============================================================ clearance check
def drive_clearance(cs, p=None, n_pose=72):
    """Each follower pivot pin rises from the base plate through every lower
    layer.  Check it against the cams (static discs) and the swept followers
    (approximated as pivot->roller bars + sector arcs) on those layers.
    Returns list of (pin, obstacle, clearance_mm); negative = collision."""
    p = cs.params
    items = []
    names = ["M1", "M2", "M3", "pen"]
    des = cs.axes + [cs.pen_axis]
    for i, (nm, d) in enumerate(zip(names, des)):
        if d is not None:
            items.append((nm, d, LAYER["follow_" + nm], LAYER["cam_" + nm]))
    out = []
    for nm, d, lay, _ in items:
        pin = d["F"]
        for nm2, d2, lay2, cam_lay2 in items:
            if nm2 == nm:
                continue
            if cam_lay2 < lay:                     # cam disc below this follower
                c = np.hypot(*(pin - cs.O)) - d2["outer"] - 3.0
                out.append((nm, "cam " + nm2, c))
            if lay2 < lay:                         # follower swept below
                idx = np.linspace(0, len(d2["psi"]) - 1, n_pose).astype(int)
                best = 1e9
                for k in idx:
                    ps = d2["psi"][k]
                    R = d2["F"] + d2["l"] * np.array([math.cos(ps), math.sin(ps)])
                    a, b = d2["F"], R
                    t_ = np.clip(np.dot(pin - a, b - a) / np.dot(b - a, b - a), 0, 1)
                    best = min(best, np.hypot(*(pin - (a + t_ * (b - a)))) - 7 - 2.2)
                    if nm2 != "pen":
                        rs = p.module * d2["zf"] / 2 + p.module
                        dd = np.hypot(*(pin - d2["F"]))
                        if abs(dd - (rs - 2.5)) < 3.5 + 2.2:
                            a_m = math.atan2(*(np.zeros(2) - d2["F"])[::-1])
                            ap = math.atan2(*(pin - d2["F"])[::-1])
                            if abs((ap - a_m + math.pi) % (2 * math.pi) - math.pi) < np.ptp(d2["psi"]) / 2 + 0.3:
                                best = min(best, -1.0)
                out.append((nm, "follower " + nm2, best))
        # shoulder hubs / arm parts never reach the drive area (checked by angle in README)
    return out


# ================================================================== output
def export(parts, outdir="cad"):
    os.makedirs(f"{outdir}/stl", exist_ok=True)
    for f in os.listdir(f"{outdir}/stl"):          # drop stale parts
        if f.endswith(".stl"):
            os.remove(f"{outdir}/stl/{f}")
    asm = cq.Assembly(name="drawing_arm")
    for name, (solid, layer, color) in parts.items():
        asm.add(solid, name=name, color=cq.Color(color))
        bb = solid.val().BoundingBox()
        flat = solid.translate((-bb.center.x, -bb.center.y, -bb.zmin))
        cq.exporters.export(flat, f"{outdir}/stl/{name}.stl", tolerance=0.05, angularTolerance=0.1)
    asm.export(f"{outdir}/assembly.step")


def preview(parts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(14, 6.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for name, (solid, layer, color) in sorted(parts.items(), key=lambda kv: kv[1][1]):
        verts, tris = solid.val().tessellate(0.3, 0.3)
        V = np.array([(v.x, v.y, v.z) for v in verts])
        F = np.array(tris)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        nz = np.cross(b - a, c - a)
        up = nz[:, 2] / (np.linalg.norm(nz, axis=1) + 1e-12) > 0.99
        for tri in F[up]:                               # top view: upward faces
            ax1.fill(V[tri, 0], V[tri, 1], color=color, lw=0, alpha=0.85)
        pc = Poly3DCollection(V[F], facecolor=color, edgecolor="none", alpha=0.95)
        ax2.add_collection3d(pc)
    ax1.set_aspect("equal")
    ax1.set_title("top view (assembled at cam angle 0)")
    ax1.grid(alpha=0.2)
    allv = np.vstack([np.array([(v.x, v.y, v.z) for v in s.val().tessellate(1, 1)[0]]) for s, _, _ in parts.values()])
    lo, hi = allv.min(0), allv.max(0)
    mid, rng = (lo + hi) / 2, (hi - lo).max() / 2
    ax2.set_xlim(mid[0] - rng, mid[0] + rng)
    ax2.set_ylim(mid[1] - rng, mid[1] + rng)
    ax2.set_zlim(mid[2] - rng / 3, mid[2] + rng / 3)
    ax2.view_init(elev=35, azim=-60)
    ax2.set_box_aspect((1, 1, 1 / 3))
    ax2.set_title("isometric")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    from arm_linkage import PAPER, builtin_shape
    from vector_input import fit_to_box, load_svg

    src = sys.argv[1] if len(sys.argv) > 1 else "heart"
    strokes = load_svg(src) if src.endswith(".svg") else builtin_shape(src)
    g = Geometry()
    plan, dp = plan_for_cams(g, fit_to_box(strokes, PAPER))
    cs = design_cams(plan, g, dp)
    print(report(cs))
    if any(a is None for a in cs.axes + [cs.pen_axis]):
        sys.exit("some cams are infeasible for this drawing - try a simpler one")
    plot_cams(cs, "cad/cams.png")
    bad = [c for c in drive_clearance(cs) if c[2] < 0]
    for pin, obs, c in drive_clearance(cs):
        print(f"  pivot pin {pin:4s} vs {obs:12s}: {c:6.1f} mm" + ("   <-- COLLISION" if c < 0 else ""))
    parts = Builder(g, cs).build()
    export(parts)
    preview(parts, "cad/preview.png")
    print(f"{len(parts)} parts -> cad/stl/*.stl, cad/assembly.step, cad/preview.png, cad/cams.png")
