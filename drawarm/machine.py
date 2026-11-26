"""
machine.py - the complete single-motor automaton with a swappable cam cartridge.

    python -m drawarm machine [drawing]   -> output/machine/<drawing>/
                                             fixed/*.stl      printed once
                                             cams/*.stl       the swappable cam set for this drawing
                                             assembly.step, preview.png, BOM.md

What is built (all from params.toml):
  arm          links, couplers, bell-crank, hand, hubs with gear sectors,
               torsion-spring collars (spring directions from the cartridge)
  followers    FIXED geometry (cartridge.design_followers): sector gear + roller
               arm; a stop post for each, where it parks when the cams slide out
  pen lift     lever on the pen follower -> Bowden cable -> spring-loaded pen
               carrier sliding in the hand (spring = pen pressure, cable = lift)
  cartridge    carriage tongue sliding in a channel of the base plate (latch
               pin), two posts on the far side, top plate with the camshaft
               bearing, the ONE motor (N20-style gearmotor) and a worm driving a
               worm wheel on the camshaft; cams + keyed spacers on the camshaft
  base plate   shoulder axle, follower pivots, stop posts, channel, Bowden anchor

Swapping cams: pull the latch pin, slide the carriage out (rollers park on
their stops), lift the cam stack off the camshaft, drop the new stack on (all
cams share the D-flat, timing marks aligned), turn the camshaft to its index
mark, slide back in (the cams push the rollers back onto their rims), latch.
"""
import math
from dataclasses import dataclass

import numpy as np
import cadquery as cq

from .config import CFG, LAYER
from .parts import (AXLE, FIT_PIN, GAP, PC, PIN, PR, STANDOFF_R, Builder, disc, holes, seg_dist,
                    slot_link, tk, tube, upol, z0)

_MC = CFG.get("machine", {})


@dataclass
class MachineParams:
    """Defaults from params.toml [machine] (all optional)."""
    motor_rpm: float = _MC.get("motor_rpm", 200.0)        # gearmotor output speed
    motor_body: tuple = tuple(_MC.get("motor_body", [12.0, 10.0, 24.0]))   # N20: w, h, length (mm)
    motor_shaft_d: float = _MC.get("motor_shaft_d", 3.0)  # D-shaft
    worm_module: float = _MC.get("worm_module", 1.5)
    worm_r: float = _MC.get("worm_r", 6.0)                # worm pitch radius
    worm_len: float = _MC.get("worm_len", 16.0)
    wheel_teeth: int = _MC.get("wheel_teeth", 30)         # single-start worm -> ratio = teeth
    wheel_t: float = _MC.get("wheel_t", 8.0)
    channel_w: float = _MC.get("channel_w", 30.0)         # carriage tongue width
    tongue_clear: float = _MC.get("tongue_clear", 0.3)
    stop_r: float = _MC.get("stop_r", 4.0)                # stop post radius
    pen_spring_force: float = _MC.get("pen_spring_force", 1.0)   # N pen pressure
    guide_d: float = _MC.get("guide_d", 16.0)             # pen carrier guide bore in the hand
    top_gap: float = _MC.get("top_gap", 3.0)              # carriage top plate above the top layer


def top_z():
    return z0(len(PC["layer_thickness"]) - 1) + PC["layer_thickness"][-1]


class Machine:
    def __init__(self, g, plan, fol, cs, mp: MachineParams = None):
        self.g, self.plan, self.fol, self.cs = g, plan, fol, cs
        self.mp = mp or MachineParams()
        self.p = fol.params
        self.b = Builder(g, cs, spring_sign=fol.spring_sign)
        self.parts = self.b.parts                    # fixed + cam parts, name -> (solid, layer, colour)
        self.cam_names = []
        self.notes = []

    # ------------------------------------------------------------ followers
    TAIL = 24.0                                       # follower tail length (stop arm)

    def tail_dir(self, nm, psi):
        return psi + math.pi + self.tail_off.get(nm, 0.0)

    def stop_point(self, nm):
        """Stop post against the follower's TAIL (roughly opposite the roller
        arm), so the post stands outside the cam envelope.  When parking, the
        follower turns the `inward` way; the tail's leading side is then
        u(tail + inward*90)."""
        ax = self.fol.axes[nm]
        td = self.tail_dir(nm, ax["psi_stop"])
        return ax["F"] + self.TAIL * upol(td) + (5.0 + self.mp.stop_r) * upol(td + ax["inward"] * math.pi / 2)

    def _tail_clear(self, nm, cols):
        ax = self.fol.axes[nm]
        ps = np.linspace(min(ax["psi_lo"], ax["psi_stop"]), max(ax["psi_hi"], ax["psi_stop"]), 30)
        return min((seg_dist(q, ax["F"], ax["F"] + self.TAIL * upol(self.tail_dir(nm, a_))) - 5 - r
                    for a_ in ps for q, r in cols), default=1e9)

    def place_tails(self):
        """Top follower first: pick each tail angle so its stop post clears what
        it passes and its tail clears the columns of every higher follower."""
        self.tail_off = {}
        order = sorted(self.fol.axes, key=lambda nm: -LAYER["follow_" + nm])
        placed_cols = []
        for nm in order:
            ax = self.fol.axes[nm]
            best = None
            for off in np.radians(np.arange(-80, 81, 10)):
                self.tail_off[nm] = off
                P = self.stop_point(nm)
                c1, _, _ = self.b.layout.column_clearance(P, LAYER["follow_" + nm], self.mp.stop_r,
                                                          exclude=("follower " + nm,))
                c2 = self._tail_clear(nm, placed_cols)
                score = min(c1, c2) - 0.05 * abs(math.degrees(off))
                if best is None or score > best[0]:
                    best = (score, off)
            self.tail_off[nm] = best[1]
            placed_cols += [(ax["F"], STANDOFF_R), (self.stop_point(nm), self.mp.stop_r)]

    def build_followers(self):
        p, cs = self.p, self.cs
        for des in [d for d in cs.axes + [cs.pen_axis] if d is not None]:
            nm = des["name"]
            fl = LAYER["follow_" + nm]
            tf = tk(fl)
            if nm == "pen":
                fol, F, R0 = self.pen_lever(des, tf)
            else:
                fol, F, R0 = self.b._follower_plate(nm, des, tf, with_arm=True)
            fol = fol.union(slot_link(F, F + self.TAIL * upol(self.tail_dir(nm, des["psi"][0])), 7, 5, tf))  # tail
            fol = holes(holes(fol, [F], PIN, tf), [R0], FIT_PIN, tf)
            self.b.add(f"follower_{nm}", fol, fl, "#9e9e9e")
            self.b.add(f"standoff_{nm}", tube(F, STANDOFF_R, PIN / 2, z0(fl)), -1, "#bcaaa4", z=-GAP)
            P = self.stop_point(nm)
            self.b.add(f"stop_post_{nm}", disc(P, self.mp.stop_r, z0(fl) + tf), -1, "#a1887f", z=-GAP)

    def pen_lever(self, des, tf):
        """Pen follower: roller arm + cable lever on the side where the cable
        tension (pen spring) pushes the roller onto its cam."""
        ax = self.fol.axes["pen"]
        F, l = ax["F"], ax["l"]
        psi0 = des["psi"][0]
        R0 = F + l * upol(psi0)
        lever = self.fol.cart.pen_lift / max(self.p.pen_rise, 1e-6) * l          # tip travel = pen lift
        # Bowden anchor: a column clear of everything it passes, near the shoulder
        # (cable route), with a lever tip whose cable tension pushes the roller inward
        top_layer = len(PC["layer_thickness"])
        best = None
        for rad in np.arange(40, 151, 10.0):
            for th in np.radians(np.arange(0, 360, 10)):
                ex = F + rad * upol(th)
                c, _, _ = self.b.layout.column_clearance(ex, top_layer, 5.0)
                if c < 2.0:
                    continue
                for a in np.radians(np.arange(0, 360, 15)):
                    tip = F + lever * upol(psi0 + a)
                    v = ex - tip
                    if np.linalg.norm(v) < 15:
                        continue
                    torque = np.cross(tip - F, v / np.linalg.norm(v))
                    if np.sign(torque) != ax["inward"] or abs(torque) < 0.6 * lever:
                        continue
                    score = np.linalg.norm(ex) - 20 * min(c, 10.0)        # near the shoulder, clear
                    if best is None or score < best[0]:
                        best = (score, tip, ex)
        if best is None:
            raise SystemExit("no clear Bowden anchor position for the pen lever")
        _, tip, exit_pt = best
        self.pen_tip, self.pen_exit = tip, exit_pt
        ra0, ra1 = PR["follower_arm"]
        fol = slot_link(F, R0, ra0, ra1, tf).union(slot_link(F, tip, ra0, 5, tf))
        fol = holes(fol, [tip], 2.0, tf)                            # cable nipple hole
        return fol, F, R0

    # ------------------------------------------------------------ pen carrier
    def build_pen_carrier(self):
        """Hand: replace the fixed pen collar by a guide bore; the carrier
        (pen clamp + spring seat + cable eye) slides in it."""
        st = self.b.st
        Pp = np.asarray(st["P"])
        W = np.asarray(st["W"])
        t = tk(LAYER["hand"])
        solid, layer, col = self.parts["hand"]
        zH = z0(LAYER["hand"])
        gd = self.mp.guide_d
        h = t + 12
        collar = disc(Pp, gd / 2 + 3, h).translate((0, 0, zH))
        u = (Pp - W) / np.linalg.norm(Pp - W)
        stop_post = disc(Pp - (gd / 2 + 6) * u, 3.0, h + 10).translate((0, 0, zH))       # Bowden stop
        hand = solid.union(collar).union(stop_post)
        hand = hand.cut(disc(Pp, gd / 2 + 0.2, h).translate((0, 0, zH)))
        hand = hand.cut(disc(Pp - (gd / 2 + 6) * u, 1.1, h + 10).translate((0, 0, zH)))  # cable housing hole
        self.parts["hand"] = (hand, layer, col)
        # carrier: tube that slides in the guide, pen bore + set screw, flange with cable eye on top
        L = h + 14
        car = disc(Pp, gd / 2 - 0.2, L).cut(disc(Pp, PC["pen_bore"] / 2, L))
        car = car.union(tube(Pp, gd / 2 + 5, PC["pen_bore"] / 2, 3).translate((0, 0, L - 3)))   # flange
        eye = Pp - (gd / 2 + 3) * u
        car = car.cut(disc(eye, 1.0, L + 1))
        self.b.add("pen_carrier", car, -1, "#e08a1e", z=zH - 6)
        self.notes.append(f"pen carrier: compression spring (OD < {gd + 8:.0f} mm, ~{self.mp.pen_spring_force:g} N at "
                          f"pen-down) between the hand collar top and the carrier flange; cable from the pen "
                          f"lever eye through the Bowden stop on the hand; pen lift {self.fol.cart.pen_lift:g} mm")

    # ------------------------------------------------------------ cartridge
    def build_cartridge(self):
        cs, p, mp, fol = self.cs, self.p, self.mp, self.fol
        O, d = fol.O, fol.slide
        n = np.array([-d[1], d[0]])
        # cams (periphery) + camshaft spacers
        from .cartridge import cam_solids
        cams = cam_solids(cs, fol)
        za = -GAP
        for nm, solid in sorted(cams.items(), key=lambda kv: LAYER["cam_" + kv[0]]):
            L = LAYER["cam_" + nm]
            if z0(L) - za > 0.1:
                self.b.add(f"cam_spacer_{nm}", self._dtube(O, PC["spacer_od"] / 2, z0(L) - za), -1, "#bbbbbb", z=za)
            self.b.add(f"cam_{nm}", solid.translate((O[0], O[1], 0)), L, "#666666")
            self.cam_names.append(f"cam_{nm}")
            za = z0(L) + tk(L)
        zt = top_z() + mp.top_gap
        self.b.add("cam_spacer_top", self._dtube(O, PC["spacer_od"] / 2, zt - za), -1, "#bbbbbb", z=za)
        self.cam_names += [k for k in self.parts if k.startswith("cam_spacer")]
        # carriage: tongue (in the base channel), posts on the far side, top plate
        env = fol.cart.max_od / 2
        far = O + (env + 10) * d
        bt = PC["base_thickness"]
        tongue_pts = [O - 18 * d + mp.channel_w / 2 * n, far + 12 * d + mp.channel_w / 2 * n,
                      far + 12 * d - mp.channel_w / 2 * n, O - 18 * d - mp.channel_w / 2 * n]
        tongue = cq.Workplane("XY").polyline([tuple(q) for q in tongue_pts]).close().extrude(bt)
        tongue = holes(tongue, [O], AXLE, bt)
        latch = O + (env + 2) * d
        tongue = holes(tongue, [latch], PIN, bt)
        posts = [far + 10 * n, far - 10 * n]
        tongue = holes(tongue, posts, 3.3, bt)                     # M3 screws up the posts
        self.b.add("carriage_tongue", tongue, -1, "#90a4ae", z=-GAP - bt)
        for i, q in enumerate(posts):
            self.b.add(f"carriage_post_{i}", tube(q, 5.0, 1.6, zt + GAP), -1, "#90a4ae", z=-GAP)
        top_pts = [O - 22 * d + 22 * n, far + 12 * d + 16 * n, far + 12 * d - 16 * n, O - 22 * d - 22 * n]
        top = cq.Workplane("XY").polyline([tuple(q) for q in top_pts]).close().extrude(5)
        top = holes(holes(top, [O], AXLE, 5), posts, 3.3, 5)
        self.b.add("carriage_top", top, -1, "#90a4ae", z=zt)
        self.carriage = dict(posts=posts, latch=latch, far=far, zt=zt, tongue=tongue_pts)
        self.build_drive(zt + 5)

    def _dtube(self, c, r, h):
        flat = AXLE / 2 - 0.8
        return disc(c, r, h).cut(disc(c, AXLE / 2, h).cut(
            cq.Workplane("XY").center(c[0] + flat + 5, c[1]).rect(10, 10).extrude(h)))

    def build_drive(self, z):
        """Worm wheel on the camshaft above the carriage top plate; worm on the
        gearmotor shaft; motor clamp on the top plate."""
        mp, O, d = self.mp, self.fol.O, self.fol.slide
        n = np.array([-d[1], d[0]])
        m, zw = mp.worm_module, mp.wheel_teeth
        r_wheel = m * zw / 2
        lead = math.atan(m / (2 * mp.worm_r))                     # single start
        twist = math.degrees(mp.wheel_t * math.tan(lead) / r_wheel)
        from .parts import gear_sector_pts
        pts = gear_sector_pts(O, m, zw, 0.0, math.pi - 1e-6, inner_r=AXLE / 2 + 3)
        wheel = cq.Workplane("XY").polyline(pts).close().twistExtrude(mp.wheel_t, twist)
        wheel = wheel.cut(self._dtube(O, AXLE / 2 + 0.01, mp.wheel_t).union(disc(O, AXLE / 2, mp.wheel_t)))
        self.b.add("worm_wheel", wheel, -1, "#ffb74d", z=z)
        # worm: helical thread swept along a helix, axis along n, tangent to the wheel
        a_c = r_wheel + mp.worm_r                                  # centre distance
        wc = O + a_c * d                                           # worm axis passes here (far side)
        pitch = math.pi * m
        L = mp.worm_len
        helix = cq.Wire.makeHelix(pitch, L, mp.worm_r - 1.25 * m)
        prof = (cq.Workplane("XZ").center(mp.worm_r - 1.25 * m, 0)
                .polyline([(0, -0.9 * pitch / 2), (2.25 * m, -0.35 * pitch / 2),
                           (2.25 * m, 0.35 * pitch / 2), (0, 0.9 * pitch / 2)]).close())
        try:
            thread = prof.sweep(cq.Workplane("XY").add(helix), isFrenet=True)
            worm = cq.Workplane("XY").circle(mp.worm_r - 1.25 * m + 0.05).extrude(L).union(thread)
        except Exception:                                          # fall back to a plain core
            worm = cq.Workplane("XY").circle(mp.worm_r + m).extrude(L)
            self.notes.append("worm thread sweep failed in CadQuery: buy a module-1.5 single-start worm")
        worm = worm.cut(cq.Workplane("XY").circle(mp.motor_shaft_d / 2 + 0.1).extrude(L))
        ang = math.degrees(math.atan2(n[1], n[0]))
        worm = (worm.translate((0, 0, -L / 2)).rotate((0, 0, 0), (0, 1, 0), 90)
                .rotate((0, 0, 0), (0, 0, 1), ang).translate((wc[0], wc[1], z + mp.wheel_t / 2)))
        self.parts["worm"] = (worm, -1, "#ffb74d")
        # motor clamp: a block holding the motor body, axis along n
        bw, bh, bl = mp.motor_body
        mc = wc - (L / 2 + 2 + bl / 2) * n
        clamp = (cq.Workplane("XY").rect(bl + 4, bw + 6).extrude(bh + 4 + mp.wheel_t / 2 + 2)
                 .cut(cq.Workplane("XY").workplane(offset=mp.wheel_t / 2 + 2 - bh / 2).rect(bl, bw + 0.4)
                      .extrude(bh + 0.4 + 10)))
        clamp = clamp.rotate((0, 0, 0), (0, 0, 1), ang).translate((mc[0], mc[1], z - 5))
        self.parts["motor_clamp"] = (clamp, -1, "#90a4ae")
        rpm_cam = 60.0 / self.cs.period
        self.notes.append(f"motor: N20-style gearmotor ~{rpm_cam * zw:.0f} rpm ({mp.motor_rpm:g} rpm part + "
                          f"voltage trim), single-start worm m{m:g} -> {zw}:1 wheel -> camshaft {rpm_cam:.1f} rpm "
                          f"({self.cs.period:.1f} s per drawing); worm is self-locking")

    # ------------------------------------------------------------ base plate
    def build_base(self):
        fol, mp = self.fol, self.mp
        O, d = fol.O, fol.slide
        n = np.array([-d[1], d[0]])
        pts = [np.zeros(2), O] + [ax["F"] for ax in fol.axes.values()] + \
              [self.stop_point(nm) for nm in fol.axes] + [self.pen_exit]
        from scipy.spatial import ConvexHull
        hull = np.array([q + 18 * upol(a) for q in pts for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)])
        h = hull[ConvexHull(hull).vertices]
        bt = PC["base_thickness"]
        base = cq.Workplane("XY").polyline([tuple(q) for q in h]).close().extrude(bt)
        # channel for the carriage tongue (open toward the slide direction)
        w = mp.channel_w + 2 * mp.tongue_clear
        ch = [O - (18 + mp.tongue_clear) * d + w / 2 * n, O + 400 * d + w / 2 * n,
              O + 400 * d - w / 2 * n, O - (18 + mp.tongue_clear) * d - w / 2 * n]
        base = base.cut(cq.Workplane("XY").polyline([tuple(q) for q in ch]).close().extrude(bt))
        base = holes(base, [np.zeros(2)], AXLE, bt)
        base = holes(base, [ax["F"] for ax in fol.axes.values()], PIN, bt)
        base = holes(base, [self.stop_point(nm) for nm in fol.axes], 3.2, bt)    # stop-post screws
        # latch ear: a tab beside the channel with the latch-pin hole lined up with the tongue's
        ear_c = self.carriage["latch"] + (w / 2 + 8) * n
        ear = slot_link(self.carriage["latch"] + (w / 2) * n, ear_c, 7, 7, bt)
        base = base.union(ear)
        base = base.cut(cq.Workplane("XY").polyline([tuple(q) for q in ch]).close().extrude(bt))
        self.notes.append("latch: a 4 mm pin through the base ear and the carriage tongue (horizontal hole, drill/ream)")
        # Bowden anchor post for the pen cable
        base = base.union(disc(self.pen_exit, 5, bt))
        self.b.add("bowden_anchor", tube(self.pen_exit, 5, 1.1, z0(LAYER["follow_pen"]) + 6), -1, "#a1887f", z=-GAP)
        self.parts["base_plate"] = (base.translate((0, 0, -bt - GAP)), -1, "#d7ccc8")

    # ------------------------------------------------------------ checks
    def checks(self):
        """Clearances: standoffs / stop posts vs everything they pass, the
        carriage channel vs fixed columns, the slide-out."""
        from .cartridge import slide_out_check
        from .parts import drive_clearance
        out = [(what, c) for what, who, lay, c in drive_clearance(self.b.layout)]
        for nm in self.fol.axes:
            c, who, lay = self.b.layout.column_clearance(self.stop_point(nm), LAYER["follow_" + nm], self.mp.stop_r,
                                                         exclude=("follower " + nm,))
            out.append((f"stop post {nm} (vs {who} L{lay})", c))
        c, who, lay = self.b.layout.column_clearance(self.pen_exit, len(PC["layer_thickness"]), 5.0)
        out.append((f"Bowden anchor column (vs {who} L{lay})", c))
        # fixed columns must stay out of the channel (the tongue slides through it)
        O, d = self.fol.O, self.fol.slide
        n = np.array([-d[1], d[0]])
        half = self.mp.channel_w / 2 + self.mp.tongue_clear
        cols = [(f"standoff {nm}", ax["F"], STANDOFF_R) for nm, ax in self.fol.axes.items()] + \
               [(f"stop post {nm}", self.stop_point(nm), self.mp.stop_r) for nm in self.fol.axes] + \
               [("shoulder axle", np.zeros(2), 10.0), ("Bowden anchor", self.pen_exit, 5.0)]
        for name, q, r in cols:
            rel = q - O
            along, across = rel @ d, rel @ n
            if along > -18 - r:
                out.append((f"{name} vs carriage channel", abs(across) - half - r))
        # follower tails sweep their layer: columns of higher followers pass through it
        cols_by = {nm: [(self.fol.axes[nm]["F"], STANDOFF_R), (self.stop_point(nm), self.mp.stop_r)]
                   for nm in self.fol.axes}
        for nm, ax in self.fol.axes.items():
            ps = np.linspace(min(ax["psi_lo"], ax["psi_stop"]), max(ax["psi_hi"], ax["psi_stop"]), 40)
            for other, cl in cols_by.items():
                if other == nm or LAYER["follow_" + other] <= LAYER["follow_" + nm]:
                    continue
                out.append((f"tail {nm} vs columns of {other}", self._tail_clear(nm, cl)))
        clear, s_free = slide_out_check(self.cs, self.fol)
        out.append((f"slide-out (cams clear after {s_free:.0f} mm)", clear))
        return out

    def build(self):
        self.place_tails()
        self.b.build_arm()
        self.build_pen_carrier()
        self.build_followers()
        self.build_cartridge()
        self.build_base()
        return self.parts

    def bom(self):
        sp = self.b.springs
        lines = ["# Bill of materials", "",
                 "## Printed (fixed: print once)", ""]
        fixed = [k for k in self.parts if k not in self.cam_names]
        lines += [f"- {k}" for k in sorted(fixed)]
        lines += ["", "## Printed (cam set for this drawing: swap these)", ""]
        lines += [f"- {k}" for k in sorted(self.cam_names)]
        lines += ["", "## Hardware", "",
                  "- 1 x N20-style 6 V gearmotor (D-shaft 3 mm), the ONLY motor",
                  "- 1 x 8 mm steel rod with D-flat, shoulder axle; 1 x 8 mm steel rod with D-flat, camshaft",
                  "- ~12 x 4 mm steel pins / M4 shoulder bolts (link joints, follower pivots), 1 x 4 mm latch pin",
                  "- 4 x 684ZZ bearings (4x9x4) as rollers + 4 mm stub pins",
                  "- 2 x 3 mm dowels (bell-crank key, M3 hub key)",
                  "- 2 x M3 x ~90 mm screws (carriage posts), M3 screws for the stop posts",
                  "- Bowden cable (inner ~0.8 mm, housing ~2 mm OD) + 1 compression spring for the pen carrier",
                  "- washers 8 mm / 4 mm (1 mm thick) between layers"]
        lines += ["", "## Torsion springs (shoulder axle)", ""]
        for nm, s in sp.items():
            lines.append(f"- {nm}: music wire {s['d']} mm, mean D {s['D']:.1f} mm, {s['n']} coils, "
                         f"wind {s['wind_deg']:.0f} deg, direction {'CCW' if s['sign'] > 0 else 'CW'} on the hub")
        lines += ["", "## Notes", ""] + [f"- {x}" for x in self.notes]
        return "\n".join(lines)
