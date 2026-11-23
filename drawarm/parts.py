"""
parts.py - parametric 3D-printable parts for the drawing arm + cam drive (CadQuery).

    python -m drawarm parts [shape|file.svg]   -> output/stl/*.stl, output/assembly.step,
                                                  output/preview.png, output/cams.png, output/springs.md

All dimensions come from params.toml ([parts], [drive], [arm]); geometry from
Geometry (kinematics.py) and the CamSet (cams.py).

Layer stack, bottom -> top (thickness per layer from [parts].layer_thickness):

  layer  arm side                          drive side (behind the shoulder)
  -----  --------------------------------  ---------------------------------
   0                                       cam M2
   1     M2 elbow crank (hub on S axle)    follower M2 (sector gear + roller)
   2     coupler E  (C2-T2)                cam M1
   3     upper arm (hub on S axle)         follower M1
   4     forearm (tail T2, E, W)
   5     bell-crank arm B, hand + crank    cam M3
   6     coupler B, M3 gear (hub)          follower M3
   7     bell-crank arm A, M3 crank (hub)  cam pen-lift
   8     coupler A  (A1-A2)                follower pen-lift (cable lever)
  base plate below layer 0: S axle (D-flat, fixed), camshaft, follower standoffs.
  torsion springs + keyed collars fill the shoulder-axle gaps on layers 0, 2, 4-5.

Every pin joins ADJACENT layers except T2 (layers 2-4, clears the upper arm
thanks to the 20 deg elbow limit), the bell-crank hub (5-7, dowel-keyed to the
arm-A plate) and the M3 hub (gear on 6, crank on 7, dowel-keyed).
Anti-backlash: a torsion spring around the fixed shoulder axle under each hub
(keyed collar + leg hole in the hub, see springs.py) holds a hub torque larger
than the arm ever needs ([drive.preload]), so gear and cam flanks never change
side.
"""
import math
import os

import numpy as np
import cadquery as cq

from .kinematics import Geometry, Linkage, paper_motor_range
from .cams import design_cams, plan_for_cams, report
from . import springs as spr
from .config import CFG, PAPER, LAYER, OUT

PC, PR, DR = CFG["parts"], CFG["parts"]["radius"], CFG["drive"]
GAP = PC["gap"]
TH = list(PC["layer_thickness"])
PIN, AXLE, FIT_PIN = PC["pin_bore"], PC["axle_bore"], PC["fit_pin"]
BACKLASH = PC["backlash"]
STANDOFF_R = PC["standoff_od"] / 2
SPACER_R = PC["spacer_od"] / 2

HUB_LAYERS = (1, 3, 6)                       # layers with a gear on the shoulder axle
CAM_LAYERS = (0, 2, 5, 7)


def z0(layer):
    return sum(TH[i] + GAP for i in range(layer))


def tk(layer):
    return TH[layer]


# ------------------------------------------------------------------ 2D helpers
def v2(p):
    return np.asarray(p, float)


def ang(a, b):
    d = v2(b) - v2(a)
    return math.atan2(d[1], d[0])


def upol(a):
    return np.array([math.cos(a), math.sin(a)])


def slot_link(a, b, ra, rb, t):
    """Stadium/hull of two circles (a link body)."""
    a, b = v2(a), v2(b)
    d = b - a
    th = math.atan2(d[1], d[0])
    pts = [a + ra * upol(th + k) for k in np.linspace(math.pi / 2, 3 * math.pi / 2, 24)]
    pts += [b + rb * upol(th + k) for k in np.linspace(-math.pi / 2, math.pi / 2, 24)]
    return cq.Workplane("XY").polyline([tuple(p) for p in pts]).close().extrude(t)


def holes(wp, pts, d, t):
    for p in pts:
        wp = wp.cut(cq.Workplane("XY").center(*p).circle(d / 2).extrude(t))
    return wp


def disc(c, r, t):
    return cq.Workplane("XY").center(*c).circle(r).extrude(t)


def tube(c, r_out, r_in, t):
    return disc(c, r_out, t).cut(disc(c, r_in, t))


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
    pol = lambda r, a: tuple(c + r * upol(a))
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
    for a in np.linspace(a_end, a_start, 12)[1:-1]:
        pts.append(pol(inner_r, a))
    return pts


def gear_sector(center, m, z, a_center, half_span, t, gap_on_center=False, inner_r=None, anchor=None):
    pts = gear_sector_pts(center, m, z, a_center, half_span, gap_on_center, inner_r, anchor)
    return cq.Workplane("XY").polyline(pts).close().extrude(t)


# ================================================== drive layout & clearances
def seg_dist(p, a, b):
    ab = b - a
    t_ = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.hypot(*(p - (a + t_ * ab))))


def seg_seg_dist(a, b, c, d, n=12):
    return min(min(seg_dist(a + (b - a) * k / n, c, d) for k in range(n + 1)),
               min(seg_dist(c + (d - c) * k / n, a, b) for k in range(n + 1)))


class Layout:
    """Plan-view obstacles per layer in the drive area, for checking vertical
    standoffs (which pass through every lower layer)."""

    def __init__(self, g, cs, M_paper):
        self.g, self.cs, self.p = g, cs, cs.params
        self.M_paper = M_paper
        self.items = []                         # (layer, name, fn(point)->clearance, seg_fn)
        p = self.p
        names = ["M1", "M2", "M3", "pen"]
        self.designs = [(nm, d) for nm, d in zip(names, cs.axes + [cs.pen_axis]) if d is not None]
        self.drum = hasattr(cs, "drum")             # DrumSet: no cams above the base plate
        for nm, d in self.designs:
            O, R = cs.O, d["outer"]
            if not self.drum:
                self.items.append((LAYER["cam_" + nm], "cam " + nm, lambda q, O=O, R=R: np.hypot(*(q - O)) - R))
            self.items.append((LAYER["follow_" + nm], "follower " + nm, self._follower_fn(nm, d)))
        # shoulder axle hardware
        r_tip = p.module * p.z_hub / 2 + p.module
        for lay in range(len(TH)):
            r = r_tip if lay in HUB_LAYERS else max(SPACER_R, DR["preload"].get("max_od", 21.0) / 2 + 1)
            self.items.append((lay, "shoulder hub/spacer", lambda q, r=r: np.hypot(*q) - r))
            if lay not in CAM_LAYERS and not hasattr(cs, "drum"):
                self.items.append((lay, "camshaft spacer", lambda q: np.hypot(*(q - cs.O)) - SPACER_R))
        # arm-side sweeps near the shoulder (cranks/couplers over the whole paper)
        from .kinematics import solve_ik
        lk = Linkage(g)
        poses = [lk.solve(*solve_ik(g, (x, y))["inputs"])
                 for x in np.linspace(PAPER[0], PAPER[2], 7) for y in np.linspace(PAPER[1], PAPER[3], 5)]
        segs = {1: [], 2: [], 7: [], 8: []}
        for st in poses:
            segs[1].append((v2(st["S"]), v2(st["C2"]), PR["crank_M2"][0]))
            segs[2].append((v2(st["C2"]), v2(st["T2"]), PR["coupler_E"]))
            segs[7].append((v2(st["S"]), v2(st["A1"]), PR["crank_M3"][0]))
            segs[8].append((v2(st["A1"]), v2(st["A2"]), PR["coupler_A"]))
        for lay, ss in segs.items():
            self.items.append((lay, "arm sweep", lambda q, ss=ss: min(seg_dist(q, a, b) - r for a, b, r in ss)))

    def _follower_fn(self, nm, d):
        p = self.p
        idx = np.linspace(0, len(d["psi"]) - 1, 48).astype(int)
        F = d["F"]
        geared = nm != "pen"
        rs = p.module * d["zf"] / 2 if geared else 0
        a_m = ang(F, np.zeros(2))
        half = np.ptp(d["psi"]) / 2 + math.radians(8)

        def fn(q):
            best = 1e9
            for k in idx:
                R = F + d["l"] * upol(d["psi"][k])
                best = min(best, seg_dist(q, F, R) - PR["follower_arm"][0])
            if geared:
                dd = np.hypot(*(q - F))
                aq = ang(F, q)
                if abs((aq - a_m + math.pi) % (2 * math.pi) - math.pi) < half:
                    best = min(best, abs(dd - (rs - 2)) - (p.module * 1.5 + 5))
            return best
        return fn

    def column_clearance(self, q, top_layer, r, exclude=()):
        """Vertical column of radius r at q from the base up to (not incl.) top_layer."""
        out = []
        for lay, name, fn in self.items:
            if lay < top_layer and name not in exclude:
                out.append((fn(q) - r, name, lay))
        return min(out) if out else (1e9, "", -1)

    def segment_clearance(self, a, b, layer, r, exclude=()):
        out = []
        for lay, name, fn in self.items:
            if lay == layer and name not in exclude:
                ts = np.linspace(0, 1, 9)
                out.append((min(fn(a + (b - a) * t_) for t_ in ts) - r, name))
        return min(out) if out else (1e9, "")


def drive_clearance(layout):
    """Clearance of every follower standoff (columns rising from the base
    plate) against everything on the layers they pass."""
    res = []
    for nm, d in layout.designs:
        c, who, lay = layout.column_clearance(d["F"], LAYER["follow_" + nm], STANDOFF_R)
        res.append((f"standoff {nm}", who, lay, c))
    return res


# ================================================================== parts
class Builder:
    def __init__(self, g: Geometry, cs):
        self.g, self.cs = g, cs
        self.p = cs.params
        M0 = cs.motors[0]                          # assembled pose = cam angle 0
        self.M_paper = paper_motor_range(g)        # gears cover the whole paper (reusable)
        self.st = Linkage(g).solve(*M0)
        self.M0 = M0
        self.layout = Layout(g, cs, self.M_paper)
        self.springs = spr.design(self.M_paper)        # torsion springs on the shoulder axle
        self.parts = {}                            # name -> (solid in world, layer, colour)

    def add(self, name, solid, layer, color, z=None):
        self.parts[name] = (solid.translate((0, 0, z0(layer) if z is None else z)), layer, color)

    # ---------------------------------------------------------- arm
    def hub_sector(self, axis_idx, solid, t):
        des = self.cs.axes[axis_idx]
        S = np.zeros(2)
        a_mesh = ang(S, des["F"])
        lo, hi = self.M_paper[axis_idx]
        lo, hi = min(lo, self.cs.motors[:, axis_idx].min()), max(hi, self.cs.motors[:, axis_idx].max())
        span = (hi - lo) / 2 + math.radians(12)
        dM = np.array([lo, hi]) - self.M0[axis_idx]           # hub material at mesh = a_mesh - dM
        centre = a_mesh - (dM.max() + dM.min()) / 2
        sec = gear_sector(S, self.p.module, self.p.z_hub, centre, span, t, gap_on_center=False,
                          inner_r=4.0, anchor=a_mesh)
        return solid.union(sec)

    def leg_hole(self, solid, axis, t):
        """Hole for the torsion-spring leg, on the side away from the gear mesh."""
        sp = self.springs[axis]
        k = ["M1", "M2", "M3"].index(axis)
        a = ang(np.zeros(2), self.cs.axes[k]["F"]) + math.pi if self.cs.axes[k] else 0.0
        r = min(sp["D"] / 2, PR["hub"] - sp["d"] / 2 - 1.0)
        return holes(solid, [r * upol(a)], sp["d"] + 0.4, t)

    def build_arm(self):
        g, st = self.g, self.st
        S, E, W, Pp = (v2(st[k]) for k in ("S", "E", "W", "P"))
        L = LAYER

        t = tk(L["crank_M2"])
        r0, r1 = PR["crank_M2"]
        crank2 = slot_link(S, st["C2"], r0, r1, t).union(disc(S, PR["hub"], t))
        crank2 = self.hub_sector(1, crank2, t) if self.cs.axes[1] else crank2
        crank2 = holes(holes(crank2, [S], AXLE, t), [st["C2"]], PIN, t)
        crank2 = self.leg_hole(crank2, "M2", t)
        self.add("crank_M2", crank2, L["crank_M2"], "#1aa3a3")

        t = tk(L["coupler_E"])
        r = PR["coupler_E"]
        self.add("coupler_E", holes(slot_link(st["C2"], st["T2"], r, r, t), [st["C2"], st["T2"]], PIN, t),
                 L["coupler_E"], "#7fd6d6")

        t = tk(L["upper_arm"])
        r0, r1 = PR["upper_arm"]
        ua = slot_link(S, E, r0, r1, t).union(disc(S, r0, t))
        ua = self.hub_sector(0, ua, t) if self.cs.axes[0] else ua
        ua = holes(holes(ua, [S], AXLE, t), [E], PIN, t)
        ua = self.leg_hole(ua, "M1", t)
        u = (E - S) / np.hypot(*(E - S))
        sw, si = PR["upper_arm_slot"]
        ua = ua.cut(slot_link(S + u * si, E - u * si, sw, sw, t))
        self.add("upper_arm", ua, L["upper_arm"], "#2f6db5")

        t = tk(L["forearm"])
        rT, rE, rW = PR["forearm"]
        fa = slot_link(st["T2"], E, rT, rE, t).union(slot_link(E, W, rE, rW, t))
        fa = holes(fa, [st["T2"], E, W], PIN, t)
        u = (W - E) / np.hypot(*(W - E))
        sw, si = PR["forearm_slot"]
        fa = fa.cut(slot_link(E + u * si, W - u * si, sw, sw, t))
        self.add("forearm", fa, L["forearm"], "#2f9e57")

        # bell-crank: lower plate (arm B) + hub boss up to layer 7, upper plate (arm A) dowel-keyed
        rb0, rb1 = PR["bell"]
        h_boss = z0(L["bell_A"]) - z0(L["bell_B"])
        dow = E + 6.5 * upol(ang(E, st["A2"]) + 1.0)
        t5 = tk(L["bell_B"])
        bl = slot_link(E, st["B1"], rb0, rb1, t5).union(disc(E, rb0, h_boss))
        bl = holes(holes(bl, [E, st["B1"]], PIN, h_boss), [dow], PC["dowel_fit"], h_boss)
        self.add("bell_crank_lower", bl, L["bell_B"], "#8e55c9")
        t7 = tk(L["bell_A"])
        bu = holes(holes(slot_link(E, st["A2"], rb0, rb1, t7), [E, st["A2"]], PIN, t7), [dow], PC["dowel_free"], t7)
        self.add("bell_crank_upper", bu, L["bell_A"], "#8e55c9")

        t = tk(L["coupler_B"])
        r = PR["coupler_B"]
        self.add("coupler_B", holes(slot_link(st["B1"], st["B2"], r, r, t), [st["B1"], st["B2"]], PIN, t),
                 L["coupler_B"], "#c7a8ea")

        # M3: crank (layer 7) + gear (layer 6, boss up to touch the crank), dowel-keyed
        dow3 = S + 6.5 * upol(ang(S, st["A1"]))
        t = tk(L["crank_M3"])
        r0, r1 = PR["crank_M3"]
        crank3 = slot_link(S, st["A1"], r0, r1, t).union(disc(S, PR["hub"], t))
        crank3 = holes(holes(holes(crank3, [S], AXLE, t), [st["A1"]], PIN, t), [dow3], PC["dowel_free"], t)
        self.add("crank_M3", crank3, L["crank_M3"], "#8e55c9")
        t6 = tk(L["gear_M3"])
        h3 = z0(L["crank_M3"]) - z0(L["gear_M3"])
        gear3 = disc(S, PR["hub"], h3)
        gear3 = self.hub_sector(2, gear3, t6) if self.cs.axes[2] else gear3
        gear3 = holes(holes(gear3, [S], AXLE, h3), [dow3], PC["dowel_fit"], h3)
        gear3 = self.leg_hole(gear3, "M3", h3)
        self.add("gear_M3", gear3, L["gear_M3"], "#8e55c9")

        t = tk(L["coupler_A"])
        r = PR["coupler_A"]
        self.add("coupler_A", holes(slot_link(st["A1"], st["A2"], r, r, t), [st["A1"], st["A2"]], PIN, t),
                 L["coupler_A"], "#c7a8ea")

        # hand: wrist boss + crank arm to B2 + pen holder at P (pen points down)
        t = tk(L["hand"])
        h0, h1 = PR["hand"]
        c0, c1 = PR["hand_crank"]
        hc = t + PC["pen_collar_extra"]
        hand = slot_link(W, Pp, h0, h1, t).union(slot_link(W, st["B2"], c0, c1, t))
        hand = hand.union(disc(Pp, h1, hc))
        hand = holes(hand, [W, st["B2"]], PIN, t)
        hand = holes(hand, [Pp], PC["pen_bore"], hc)
        u = (Pp - W) / np.hypot(*(Pp - W))
        setscrew = (cq.Workplane("YZ").circle(PC["setscrew_d"] / 2).extrude(h1 + 3)
                    .rotate((0, 0, 0), (0, 0, 1), math.degrees(math.atan2(u[1], u[0])))
                    .translate((Pp[0], Pp[1], hc / 2)))
        self.add("hand", hand.cut(setscrew), L["hand"], "#e08a1e")

        # keyed spring collars on the fixed shoulder axle (one torsion spring each)
        gaps = [("M2", -GAP, z0(1)), ("M1", z0(1) + tk(1), z0(3)), ("M3", z0(3) + tk(3), z0(6))]
        for nm, za, zb in gaps:
            sp = self.springs[nm]
            h = zb - za
            r_out = sp["ID"] / 2 - 0.4                                  # spring slides over it
            col = disc(S, r_out, h)
            flat = AXLE / 2 - 0.8                                       # D-flat on the axle
            col = col.cut(disc(S, AXLE / 2, h).cut(cq.Workplane("XY").center(flat + 5, 0).rect(10, 10).extrude(h)))
            col = col.cut(cq.Workplane("XY").center(-r_out, 0).rect(2.2, sp["d"] + 0.6).extrude(h))  # leg slot
            # flange under the spring (bears on the hub/base below)
            col = col.union(tube(S, sp["OD"] / 2 + 1.0, AXLE / 2, 1.0).cut(
                cq.Workplane("XY").center(flat + 5, 0).rect(10, 10).extrude(1.0)))
            self.add(f"spring_collar_{nm}", col, -1, "#bbbbbb", z=za)

    # ---------------------------------------------------------- drive
    def build_drive(self):
        if self.layout.drum:
            return self.build_drum_drive()
        cs, p = self.cs, self.p
        O = cs.O
        rr = p.roller_r + PC["groove_clearance"] / 2
        depth = p.roller_w + 0.5
        for nm, des in self.layout.designs:
            k = ["M1", "M2", "M3", "pen"].index(nm)
            cl, fl = LAYER["cam_" + nm], LAYER["follow_" + nm]
            tc, tf = tk(cl), tk(fl)
            # ---- cam: groove on the top face, D-flat bore, timing notch at theta = 0
            pc = des["pitch"]
            d = np.gradient(pc, axis=0)
            nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.hypot(*d.T)[:, None]
            outer_w, inner_w = pc + rr * nrm, pc - rr * nrm
            cam = disc((0, 0), des["outer"], tc)
            groove = (cq.Workplane("XY").polyline([tuple(q) for q in outer_w[::2]]).close().extrude(depth)
                      .cut(cq.Workplane("XY").polyline([tuple(q) for q in inner_w[::2]]).close().extrude(depth)))
            cam = cam.cut(groove.translate((0, 0, tc - depth)))
            cam = cam.cut(disc((0, 0), AXLE / 2, tc).cut(
                cq.Workplane("XY").center(AXLE / 2 + 1.4, 0).rect(4, 10).extrude(tc)))
            cam = cam.cut(cq.Workplane("XY").center(des["outer"], 0).rect(3, 3).extrude(tc))
            r_hole = (des["inner"] - 8) / 2
            if r_hole > 4:
                for a in np.radians(np.arange(45, 360, 90)):
                    cam = cam.cut(disc((8 + r_hole) * upol(a), r_hole * 0.8, tc))
            self.add(f"cam_{nm}", cam.translate((O[0], O[1], 0)), cl, "#666666")

            # ---- follower: pivot boss, arm to roller, sector gear toward S
            F = des["F"]
            psi0 = des["psi"][0]
            R0 = F + des["l"] * upol(psi0)
            ra0, ra1 = PR["follower_arm"]
            fol = slot_link(F, R0, ra0, ra1, tf)
            if nm != "pen":
                S = np.zeros(2)
                a_mesh = ang(F, S)
                lo, hi = self.M_paper[k]
                lo, hi = min(lo, cs.motors[:, k].min()), max(hi, cs.motors[:, k].max())
                dpsi = -(np.array([lo, hi]) - self.M0[k]) / des["G"]
                span = abs(dpsi[1] - dpsi[0]) / 2 + math.radians(6)
                centre = a_mesh - (dpsi.max() + dpsi.min()) / 2
                rs = p.module * des["zf"] / 2
                s0, s1 = PR["follower_spoke"]
                fol = fol.union(slot_link(F, F + (rs - 3 * p.module - 2) * upol(centre), s0, s1, tf))
                fol = fol.union(gear_sector(F, p.module, des["zf"], centre, span, tf, gap_on_center=True,
                                            inner_r=rs - 3 * p.module - 2, anchor=a_mesh))
            else:
                tip = F - 45 * upol(psi0)                                     # cable lever
                fol = holes(fol.union(slot_link(F, tip, ra0, 5, tf)), [tip], 2.0, tf)
            fol = holes(holes(fol, [F], PIN, tf), [R0], FIT_PIN, tf)          # pivot + roller stud
            self.add(f"follower_{nm}", fol, fl, "#9e9e9e")
            # standoff under the follower pivot
            self.add(f"standoff_{nm}", tube(F, STANDOFF_R, PIN / 2, z0(fl) - GAP + GAP), -1, "#bcaaa4", z=-GAP)

        # camshaft spacers between the cams
        cams = sorted(LAYER[f"cam_{nm}"] for nm, _ in self.layout.designs)
        za = -GAP
        for i, lay in enumerate(cams):
            if z0(lay) - za > 0.1:
                self.add(f"spacer_O_{i}", tube(O, SPACER_R, AXLE / 2, z0(lay) - za), -1, "#bbbbbb", z=za)
            za = z0(lay) + tk(lay)

    def _follower_plate(self, nm, des, tf, with_arm=True):
        """Follower body at its layer: pivot boss, (arm to roller), sector gear
        toward the shoulder (geared axes) or cable lever (pen)."""
        cs, p = self.cs, self.p
        k = ["M1", "M2", "M3", "pen"].index(nm)
        F = des["F"]
        psi0 = des["psi"][0]
        R0 = F + des["l"] * upol(psi0)
        ra0, ra1 = PR["follower_arm"]
        fol = slot_link(F, R0, ra0, ra1, tf) if with_arm else disc(F, ra0 + 2, tf)
        if nm != "pen":
            S = np.zeros(2)
            a_mesh = ang(F, S)
            lo, hi = self.M_paper[k]
            lo, hi = min(lo, cs.motors[:, k].min()), max(hi, cs.motors[:, k].max())
            dpsi = -(np.array([lo, hi]) - self.M0[k]) / des["G"]
            span = abs(dpsi[1] - dpsi[0]) / 2 + math.radians(6)
            centre = a_mesh - (dpsi.max() + dpsi.min()) / 2
            rs = p.module * des["zf"] / 2
            s0, s1 = PR["follower_spoke"]
            fol = fol.union(slot_link(F, F + (rs - 3 * p.module - 2) * upol(centre), s0, s1, tf))
            fol = fol.union(gear_sector(F, p.module, des["zf"], centre, span, tf, gap_on_center=True,
                                        inner_r=rs - 3 * p.module - 2, anchor=a_mesh))
        else:
            tip = F - 45 * upol(psi0)                                     # cable lever
            fol = holes(fol.union(slot_link(F, tip, ra0, 5, tf)), [tip], 2.0, tf)
        return fol, F, R0

    def _d_bore(self, solid, c, t):
        """8 mm D-shaft bore (rocker shafts in drum mode)."""
        flat = AXLE / 2 - 0.8
        cut = disc(c, AXLE / 2, t).cut(cq.Workplane("XY").center(c[0] + flat + 5, c[1]).rect(10, 10).extrude(t))
        return solid.cut(cut)

    def build_drum_drive(self):
        """Drum mode: follower plates (sector/lever only) on steel 8 mm D-shaft
        rockers through the base plate; a finger + roller stud on each rocker in
        the drum bay; the drum itself is exported separately as a mesh."""
        cs, p, dp = self.cs, self.p, self.cs.drum
        self.rockers = {}
        for nm, des in self.layout.designs:
            fl = LAYER["follow_" + nm]
            tf = tk(fl)
            fol, F, R0 = self._follower_plate(nm, des, tf, with_arm=False)
            self.add(f"follower_{nm}", self._d_bore(fol, F, tf), fl, "#9e9e9e")
            # finger in the bay: rocker -> roller stud (roller hangs below the finger)
            zf = cs.z_contact[nm] + p.roller_w / 2 + dp.finger_gap
            fin = slot_link(F, R0, 6.0, 5.0, dp.finger_t).union(disc(F, 8.0, dp.finger_t + 4))   # hub collar
            fin = holes(self._d_bore(fin, F, dp.finger_t + 4), [R0], FIT_PIN, dp.finger_t)
            self.add(f"finger_{nm}", fin, -1, "#9e9e9e", z=zf)
            # spacer from the base top to the follower plate
            self.add(f"standoff_{nm}", tube(F, STANDOFF_R, AXLE / 2, z0(fl) - GAP + GAP), -1, "#bcaaa4", z=-GAP)
            self.rockers[nm] = dict(F=F, z_bottom=zf, z_top=z0(fl) + tf, length=z0(fl) + tf - zf + 5)

    def bay_clearance(self):
        """Drum bay: every rocker shaft that passes a finger's height must clear
        that finger over its whole swing (fingers 6 mm half-width, shafts 4 mm)."""
        out = []
        des = dict(self.layout.designs)
        for k, rk in self.rockers.items():
            d = des[k]
            idx = np.linspace(0, len(d["psi"]) - 1, 60).astype(int)
            for j, rj in self.rockers.items():
                if j == k or rj["z_bottom"] > rk["z_bottom"]:          # shaft j does not reach down here
                    continue
                c = min(seg_dist(rj["F"], d["F"], d["F"] + d["l"] * upol(d["psi"][i])) for i in idx) - 6 - 4.5
                out.append((f"finger {k} vs rocker {j}", c))
        return out

    def build_base(self):
        cs = self.cs
        pts = [np.zeros(2), cs.O] + [d["F"] for _, d in self.layout.designs]
        hull = np.array([q + 16 * upol(a) for q in pts for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)])
        from scipy.spatial import ConvexHull
        h = hull[ConvexHull(hull).vertices]
        bt = PC["base_thickness"]
        base = cq.Workplane("XY").polyline([tuple(q) for q in h]).close().extrude(bt)
        base = holes(base, [np.zeros(2), cs.O], AXLE, bt)
        pivot = AXLE if self.layout.drum else PIN                      # rocker shafts in drum mode
        base = holes(base, [d["F"] for _, d in self.layout.designs], pivot, bt)
        self.parts["base_plate"] = (base.translate((0, 0, -bt - GAP)), -1, "#d7ccc8")

    def build(self):
        self.build_arm()
        self.build_drive()
        self.build_base()
        return self.parts


# ================================================================== output
def export(parts, outdir=OUT):
    os.makedirs(f"{outdir}/stl", exist_ok=True)
    for f in os.listdir(f"{outdir}/stl"):
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
    allv = []
    for name, (solid, layer, color) in sorted(parts.items(), key=lambda kv: kv[1][1]):
        verts, tris = solid.val().tessellate(0.3, 0.3)
        V = np.array([(v.x, v.y, v.z) for v in verts])
        F = np.array(tris)
        allv.append(V)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        nz = np.cross(b - a, c - a)
        up = nz[:, 2] / (np.linalg.norm(nz, axis=1) + 1e-12) > 0.99
        for tri in F[up]:
            ax1.fill(V[tri, 0], V[tri, 1], color=color, lw=0, alpha=0.85)
        ax2.add_collection3d(Poly3DCollection(V[F], facecolor=color, edgecolor="none", alpha=0.95))
    ax1.set_aspect("equal")
    ax1.set_title("top view (assembled at cam angle 0)")
    ax1.grid(alpha=0.2)
    allv = np.vstack(allv)
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


def build_all(src="heart", cad=True):
    from .shapes import load_drawing
    from .inputs import fit_to_box
    strokes = load_drawing(src)
    g = Geometry()
    plan, dp = plan_for_cams(g, fit_to_box(strokes, PAPER))
    cs = design_cams(plan, g, dp)
    if any(a is None for a in cs.axes + [cs.pen_axis]):
        raise SystemExit(report(cs) + "\nsome cams are infeasible for this drawing - try a simpler one")
    b = Builder(g, cs)
    parts = b.build() if cad else None
    return g, plan, cs, b, parts
