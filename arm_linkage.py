"""
arm_linkage.py - planar 3-joint drawing arm, all three inputs at the shoulder.

Layout (top view, XY plane, units = mm, angles = rad, CCW positive)
--------------------------------------------------------------------
    S  shoulder pivot (fixed to body).  M1, M2, M3 are coaxial shafts here.
    E  elbow,  W  wrist,  P  pen tip

    Main chain:   S --L1-- E --L2-- W --L3-- P        (upper arm, forearm, hand)

    Elbow drive  ("5-bar" with both inputs at S -> parallelogram):
        M1 drives the upper arm directly.
        M2 drives a crank S->C2 (radius r_e).  The forearm has a tail E->T2
        (also r_e).  Coupler C2-T2 has length L1, so S-E-T2-C2 is a
        parallelogram and the forearm's ABSOLUTE angle follows M2.

    Wrist drive  (two parallelograms in series, with an idler bell-crank at E):
        M3 drives crank S->A1 (r_a).  Coupler A1-A2 (length L1) turns a free
        bell-crank at the elbow (arm E->A2 of r_a, arm E->B1 of r_b, the two
        arms off_b-off_a apart).  Coupler B1-B2 (length L2) turns the hand crank
        W->B2 (r_b), which is rigid with the hand.
        => the hand's ABSOLUTE angle follows M3.

    Consequence: inputs map straight to absolute link angles
        M1 = phi1 (upper arm),  M2 = phi2 + pi (forearm),  M3 = phi3 + off_a (hand)
    so IK is a plain 2-link solve plus a choice of hand angle, and FK is
    fully decoupled (turning M1 alone translates forearm+hand without rotating
    them).

Dead points: a parallelogram is singular when its crank lines up with the
link it rides on.  Transmission "margin" = |sin(crank - link)|, 1 = ideal.
    elbow stage   : |sin(elbow bend)|        -> elbow bend kept in 20..160 deg
    wrist stage A : |sin(phi3+off_a - phi1)|
    wrist stage B : |sin(phi3+off_b - phi2)|
Making the elbow bell-crank an L (off_b - off_a = 90 deg) lets both wrist
stages be perfect at a 90 deg elbow bend and stay >= ~0.8 over the range.
The 3rd DOF (hand angle) is redundant for 2D pen position, so the IK uses it
to maximise the worst margin.

Run:  python arm_linkage.py          (GUI)
      python arm_linkage.py --test   (IK->FK consistency check)
"""
import math
import random
import sys
from dataclasses import dataclass

TAU = 2 * math.pi
D = math.radians


# ----------------------------------------------------------------- geometry
@dataclass
class Geometry:
    L1: float = 300.0          # upper arm   (shoulder -> elbow)
    L2: float = 250.0          # forearm     (elbow -> wrist)
    L3: float = 70.0           # hand        (wrist -> pen tip)
    r_e: float = 60.0          # elbow drive crank at S  = forearm tail at E
    r_a: float = 45.0          # wrist crank at S        = bell-crank arm A at E
    r_b: float = 40.0          # bell-crank arm B at E   = hand crank at W
    off_a: float = 0.0         # crank A angle relative to hand direction
    off_b: float = D(90)       # crank B angle relative to hand direction
    elbow_sign: int = +1       # +1: elbow bends CCW (forearm left of upper arm)
    elbow_min: float = D(20)   # |elbow bend| limits (parallelogram dead points at 0/180)
    elbow_max: float = D(160)
    wrist_max: float = D(100)  # |hand - forearm| limit (human-ish)
    margin_min: float = 0.30   # reject IK solutions with worse transmission


# ---------------------------------------------------------------- 2D helpers
def polar(p, r, a):
    return (p[0] + r * math.cos(a), p[1] + r * math.sin(a))


def ang(p, q):
    return math.atan2(q[1] - p[1], q[0] - p[0])


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def dist(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])


def circle_x(c0, r0, c1, r1):
    """Intersections of two circles (0, 1 or 2 points)."""
    dx, dy = c1[0] - c0[0], c1[1] - c0[1]
    d = math.hypot(dx, dy)
    if d < 1e-12 or d > r0 + r1 + 1e-9 or d < abs(r0 - r1) - 1e-9:
        return []
    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h = math.sqrt(max(r0 * r0 - a * a, 0.0))
    mx, my = c0[0] + a * dx / d, c0[1] + a * dy / d
    return [(mx - h * dy / d, my + h * dx / d), (mx + h * dy / d, my - h * dx / d)]


def nearest(pts, ref):
    return min(pts, key=lambda p: dist(p, ref))


# ------------------------------------------------------------------- margins
def margins(g, phi1, phi2, phi3):
    return (abs(math.sin(phi2 - phi1)),                 # elbow parallelogram
            abs(math.sin(phi3 + g.off_a - phi1)),       # wrist stage A (upper arm)
            abs(math.sin(phi3 + g.off_b - phi2)))       # wrist stage B (forearm)


# ------------------------------------------------------------- forward kin.
class Linkage:
    """Solves the closed loops from the three input angles by circle
    intersection (works even if couplers aren't exact parallelograms).
    Branch choice: nearest to the previous solution if track=True (what a real
    mechanism does - it can flip at a dead point), otherwise nearest to the
    ideal parallelogram position."""

    def __init__(self, g):
        self.g = g
        self.prev = None

    def solve(self, m1, m2, m3, track=False):
        g = self.g
        pv = self.prev if track else None
        S = (0.0, 0.0)
        E = polar(S, g.L1, m1)

        # elbow parallelogram  S-E-T2-C2
        C2 = polar(S, g.r_e, m2)
        sols = circle_x(C2, g.L1, E, g.r_e)
        if not sols:
            return None
        T2 = nearest(sols, pv["T2"] if pv else polar(E, g.r_e, m2))
        phi2 = ang(E, T2) - math.pi
        W = polar(E, g.L2, phi2)

        # wrist stage A  S-E-A2-A1
        A1 = polar(S, g.r_a, m3)
        sols = circle_x(A1, g.L1, E, g.r_a)
        if not sols:
            return None
        A2 = nearest(sols, pv["A2"] if pv else polar(E, g.r_a, m3))
        bell = ang(E, A2)                            # bell-crank arm A angle
        B1 = polar(E, g.r_b, bell - g.off_a + g.off_b)

        # wrist stage B  E-W-B2-B1
        sols = circle_x(B1, g.L2, W, g.r_b)
        if not sols:
            return None
        bB = bell - g.off_a + g.off_b
        B2 = nearest(sols, pv["B2"] if pv else polar(W, g.r_b, bB))
        phi3 = ang(W, B2) - g.off_b
        P = polar(W, g.L3, phi3)

        phi1 = m1
        st = dict(S=S, E=E, W=W, P=P, C2=C2, T2=T2, A1=A1, A2=A2, B1=B1, B2=B2,
                  phi1=phi1, phi2=phi2, phi3=phi3, inputs=(m1, m2, m3),
                  margins=margins(g, phi1, phi2, phi3))
        # flipped into an anti-parallelogram?
        st["flipped"] = (abs(wrap(phi2 + math.pi - m2)) > 1e-3 or
                         abs(wrap(phi3 + g.off_a - m3)) > 1e-3)
        self.prev = st
        return st


# ------------------------------------------------------------- inverse kin.
def ik_2link(g, Wp):
    """Shoulder/elbow angles that put the wrist at Wp (fixed elbow branch)."""
    d2 = Wp[0] ** 2 + Wp[1] ** 2
    c = (d2 - g.L1 ** 2 - g.L2 ** 2) / (2 * g.L1 * g.L2)
    if abs(c) > 1:
        return None
    e = g.elbow_sign * math.acos(c)                  # relative elbow bend
    if not (g.elbow_min <= abs(e) <= g.elbow_max):
        return None
    phi1 = math.atan2(Wp[1], Wp[0]) - math.atan2(g.L2 * math.sin(e), g.L1 + g.L2 * math.cos(e))
    return phi1, phi1 + e


def solve_ik(g, P, prev_phi3=None, fixed_phi3=None, smooth=0.15):
    """Pen tip P -> input angles.  The hand angle (redundant DOF) is either
    fixed, or chosen to maximise the worst transmission margin, with a
    penalty on jumping away from the previous hand angle."""
    if fixed_phi3 is not None:
        cands = [fixed_phi3]
    elif prev_phi3 is None:
        cands = [D(k) for k in range(-180, 180)]
    else:
        cands = [prev_phi3 + D(0.5 * k) for k in range(-60, 61)]

    best, best_cost = None, float("inf")
    for phi3 in cands:
        arm = ik_2link(g, polar(P, -g.L3, phi3))
        if arm is None:
            continue
        phi1, phi2 = arm
        if abs(wrap(phi3 - phi2)) > g.wrist_max:
            continue
        m = margins(g, phi1, phi2, phi3)
        if min(m) < g.margin_min and fixed_phi3 is None:
            continue
        cost = -min(m)
        if prev_phi3 is not None:
            cost += smooth * abs(wrap(phi3 - prev_phi3))
        if cost < best_cost:
            best_cost, best = cost, (phi1, phi2, phi3, m)

    if best is None:
        if prev_phi3 is not None and fixed_phi3 is None:   # local search failed
            return solve_ik(g, P, None, None, smooth)
        return None
    phi1, phi2, phi3, m = best
    return dict(phi1=phi1, phi2=phi2, phi3=phi3, margins=m,
                inputs=(phi1, phi2 + math.pi, phi3 + g.off_a))


# --------------------------------------------------------------- self-test
def self_test(n=5000):
    g = Geometry()
    lk = Linkage(g)
    worst, solved = 0.0, 0
    for _ in range(n):
        P = (random.uniform(250, 510), random.uniform(-110, 110))
        sol = solve_ik(g, P)
        if not sol:
            continue
        solved += 1
        st = lk.solve(*sol["inputs"])
        worst = max(worst, dist(st["P"], P))
    print(f"IK solved {solved}/{n} points on the paper; max |FK(IK(P)) - P| = {worst:.2e} mm")

    # decoupling: rotating M1 alone must not rotate forearm or hand
    s0 = lk.solve(D(10), D(200), D(30))
    s1 = lk.solve(D(40), D(200), D(30))
    print(f"M1 +30 deg -> forearm rotates {math.degrees(wrap(s1['phi2']-s0['phi2'])):.2e} deg, "
          f"hand rotates {math.degrees(wrap(s1['phi3']-s0['phi3'])):.2e} deg")

    # worst margin over the paper with auto hand angle
    mins = []
    for x in range(250, 511, 10):
        for y in range(-110, 111, 10):
            s = solve_ik(g, (x, y))
            if s:
                mins.append(min(s["margins"]))
    print(f"paper grid: worst transmission margin {min(mins):.2f}, mean {sum(mins)/len(mins):.2f}")
    return worst < 1e-6 and solved == n


# ===================================================================== GUI
def run_gui():
    import tkinter as tk

    BLUE, GREEN, ORANGE = "#2f6db5", "#2f9e57", "#e08a1e"
    TEAL, PURPLE, GREY = "#1aa3a3", "#8e55c9", "#9aa0a6"
    PAPER = (250, -110, 510, 110)                  # x0, y0, x1, y1 in mm

    def resample(pts, step=2.0):
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            n = max(1, int(dist(a, b) / step))
            out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
        return out

    def shape(name):
        cx, cy = (PAPER[0] + PAPER[2]) / 2, (PAPER[1] + PAPER[3]) / 2
        strokes = []
        if name == "circle":
            strokes = [[(cx + 85 * math.cos(t * TAU / 200), cy + 85 * math.sin(t * TAU / 200)) for t in range(201)]]
        elif name == "square":
            s = 80
            strokes = [[(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s), (cx - s, cy - s)]]
        elif name == "star":
            pts = []
            for k in range(11):
                r = 100 if k % 2 == 0 else 40
                a = math.pi / 2 + k * math.pi / 5
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            strokes = [pts]
        elif name == "spiral":
            strokes = [[(cx + (5 + 3.2 * t / 10) * math.cos(t / 10), cy + (5 + 3.2 * t / 10) * math.sin(t / 10))
                        for t in range(0, 290)]]
        elif name == "heart":
            pts = []
            for k in range(201):
                t = k * TAU / 200
                x = 16 * math.sin(t) ** 3
                y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
                pts.append((cx + 5.5 * x, cy + 5.5 * y + 10))
            strokes = [pts]
        elif name == "lissajous":
            strokes = [[(cx + 110 * math.sin(3 * t * TAU / 600), cy + 95 * math.sin(4 * t * TAU / 600))
                        for t in range(601)]]
        elif name == "grid":
            for x in range(PAPER[0] + 10, PAPER[2], 40):
                strokes.append([(x, PAPER[1] + 10), (x, PAPER[3] - 10)])
            for y in range(PAPER[1] + 10, PAPER[3], 40):
                strokes.append([(PAPER[0] + 10, y), (PAPER[2] - 10, y)])
        return strokes

    class App:
        def __init__(self, root):
            self.root = root
            root.title("3-joint drawing arm - planar linkage")
            self.g = Geometry()
            self.link = Linkage(self.g)
            self.s = 0.95                           # px per mm
            self.ox, self.oy = 170, 360
            self.queue = []
            self.mouse_down = False
            self.target = ((PAPER[0] + PAPER[2]) / 2, 0.0)
            self.prev_phi3 = None
            self.last_pen = None
            self.pen = False
            self.unreachable = False

            self.cv = tk.Canvas(root, width=840, height=720, bg="white", highlightthickness=0)
            self.cv.grid(row=0, column=0, rowspan=2)
            side = tk.Frame(root, padx=8, pady=8)
            side.grid(row=0, column=1, sticky="n")

            self.mode = tk.StringVar(value="ik")
            tk.Label(side, text="Mode", font=("Segoe UI", 10, "bold")).pack(anchor="w")
            tk.Radiobutton(side, text="IK: drag mouse / play shapes", variable=self.mode, value="ik",
                           command=self.on_mode).pack(anchor="w")
            tk.Radiobutton(side, text="Manual: drive the 3 input shafts", variable=self.mode, value="fk",
                           command=self.on_mode).pack(anchor="w")

            tk.Label(side, text="Hand angle (redundant DOF)", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
            self.hand_mode = tk.StringVar(value="auto")
            tk.Radiobutton(side, text="auto: best transmission", variable=self.hand_mode, value="auto").pack(anchor="w")
            tk.Radiobutton(side, text="fixed absolute angle:", variable=self.hand_mode, value="fixed").pack(anchor="w")
            self.hand_deg = tk.DoubleVar(value=20)
            tk.Scale(side, from_=-180, to=180, orient="horizontal", length=260, variable=self.hand_deg).pack(anchor="w")

            tk.Label(side, text="Shapes", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
            bf = tk.Frame(side)
            bf.pack(anchor="w")
            for i, name in enumerate(["circle", "square", "star", "spiral", "heart", "lissajous", "grid"]):
                tk.Button(bf, text=name, width=9, command=lambda n=name: self.play(n)).grid(row=i // 3, column=i % 3, padx=1, pady=1)
            cf = tk.Frame(side)
            cf.pack(anchor="w", pady=4)
            tk.Button(cf, text="Stop", width=9, command=lambda: self.queue.clear()).pack(side="left", padx=1)
            tk.Button(cf, text="Clear ink", width=9, command=lambda: self.cv.delete("ink")).pack(side="left", padx=1)
            self.speed = tk.DoubleVar(value=150)
            tk.Label(side, text="Pen speed (mm/s)").pack(anchor="w")
            tk.Scale(side, from_=20, to=600, orient="horizontal", length=260, variable=self.speed).pack(anchor="w")

            tk.Label(side, text="Input shafts (manual mode), deg", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
            self.mvars = []
            for lbl, col in (("M1 shoulder", BLUE), ("M2 elbow drive", TEAL), ("M3 wrist drive", PURPLE)):
                v = tk.DoubleVar()
                tk.Scale(side, label=lbl, fg=col, from_=-180, to=360, resolution=0.5, orient="horizontal",
                         length=260, variable=v).pack(anchor="w")
                self.mvars.append(v)
            self.pen_var = tk.BooleanVar(value=False)
            tk.Checkbutton(side, text="pen down (manual mode)", variable=self.pen_var).pack(anchor="w")

            self.info = tk.Label(side, justify="left", font=("Consolas", 9), anchor="w")
            self.info.pack(anchor="w", pady=(8, 0))

            self.cv.bind("<ButtonPress-1>", self.on_press)
            self.cv.bind("<B1-Motion>", self.on_drag)
            self.cv.bind("<Motion>", self.on_hover)
            self.cv.bind("<ButtonRelease-1>", lambda e: setattr(self, "mouse_down", False))

            self.draw_static()
            self.tick()

        # ---- coordinate transforms
        def w2c(self, p):
            return self.ox + self.s * p[0], self.oy - self.s * p[1]

        def c2w(self, x, y):
            return (x - self.ox) / self.s, (self.oy - y) / self.s

        # ---- events
        def on_mode(self):
            if self.mode.get() == "fk" and self.link.prev:
                for v, a in zip(self.mvars, self.link.prev["inputs"]):
                    v.set(round(math.degrees(a) % 360 if math.degrees(a) < -180 else math.degrees(a), 1))
            self.queue.clear()

        def on_press(self, e):
            self.mouse_down = True
            self.queue.clear()
            self.target = self.c2w(e.x, e.y)

        def on_drag(self, e):
            self.target = self.c2w(e.x, e.y)

        def on_hover(self, e):
            if not self.queue:
                self.target = self.c2w(e.x, e.y)

        def play(self, name):
            self.mode.set("ik")
            start = self.link.prev["P"] if self.link.prev else self.target
            q = []
            for stroke in shape(name):
                for p in resample([start, stroke[0]], 1.0)[1:]:
                    q.append((p, False))                 # pen-up travel
                for p in resample(stroke, 1.0):
                    q.append((p, True))
                start = stroke[-1]
            self.queue = q

        # ---- drawing
        def draw_static(self):
            c = self.cv
            x0, y0 = self.w2c((PAPER[0], PAPER[3]))
            x1, y1 = self.w2c((PAPER[2], PAPER[1]))
            c.create_rectangle(x0, y0, x1, y1, fill="#fbf8ee", outline="#d8d2bd")
            c.create_text(x0 + 4, y0 + 4, anchor="nw", text="paper", fill="#b5ad93", font=("Segoe UI", 8))
            # body block behind shoulder
            bx0, by0 = self.w2c((-160, 130))
            bx1, by1 = self.w2c((-25, -130))
            c.create_rectangle(bx0, by0, bx1, by1, fill="#eceff1", outline="#cfd8dc")
            c.create_text((bx0 + bx1) / 2, (by0 + by1) / 2, text="body\n\nM1 M2 M3\ncoaxial at\nshoulder",
                          fill="#78909c", font=("Segoe UI", 8), justify="center")
            # reach circles for the pen (approx, hand not counted)
            g = self.g
            rmax = math.sqrt(g.L1 ** 2 + g.L2 ** 2 + 2 * g.L1 * g.L2 * math.cos(g.elbow_min)) + g.L3
            for r, dash in ((rmax, (4, 4)),):
                a, b = self.w2c((-r, r)), self.w2c((r, -r))
                c.create_oval(*a, *b, outline="#cfd8dc", dash=dash)
            c.create_text(10, 10, anchor="nw", fill="#607d8b", font=("Segoe UI", 9), text=(
                "blue = upper arm   green = forearm (+ tail)   orange = hand/pen\n"
                "teal = elbow parallelogram (M2)   purple = wrist parallelograms + elbow bell-crank (M3)"))

        def render(self, st):
            c = self.cv
            c.delete("arm")

            def ln(p, q, w, col, **kw):
                c.create_line(*self.w2c(p), *self.w2c(q), width=w, fill=col, capstyle="round", tags="arm", **kw)

            def dot(p, r, col, fill="white"):
                x, y = self.w2c(p)
                c.create_oval(x - r, y - r, x + r, y + r, outline=col, fill=fill, width=2, tags="arm")

            S, E, W, P = st["S"], st["E"], st["W"], st["P"]
            # wrist chain
            c.create_polygon(*self.w2c(E), *self.w2c(st["A2"]), *self.w2c(st["B1"]),
                             fill="#e6d9f5", outline=PURPLE, width=2, tags="arm")
            ln(S, st["A1"], 5, PURPLE)
            ln(st["A1"], st["A2"], 2, PURPLE)
            ln(st["B1"], st["B2"], 2, PURPLE)
            ln(W, st["B2"], 5, PURPLE)
            # elbow drive
            ln(S, st["C2"], 6, TEAL)
            ln(st["C2"], st["T2"], 2.5, TEAL)
            # main links
            ln(S, E, 12, BLUE)
            ln(st["T2"], W, 9, GREEN)
            ln(W, P, 6, ORANGE)
            for p in (st["A1"], st["A2"], st["B1"], st["B2"], st["C2"], st["T2"]):
                dot(p, 3, GREY)
            for p, col in ((E, BLUE), (W, GREEN)):
                dot(p, 6, col)
            for r, col in ((14, PURPLE), (11, TEAL), (8, BLUE)):       # coaxial shafts
                dot(S, r, col, fill="")
            dot(P, 4, "red", fill="red" if self.pen else "white")
            if self.mode.get() == "ik":
                x, y = self.w2c(self.target)
                col = "#e53935" if self.unreachable else "#90a4ae"
                c.create_line(x - 7, y, x + 7, y, fill=col, tags="arm")
                c.create_line(x, y - 7, x, y + 7, fill=col, tags="arm")

        def readout(self, st):
            d = math.degrees
            m = st["margins"]
            m1, m2, m3 = (d(a) for a in st["inputs"])
            txt = (f"inputs   M1 {m1:7.1f}  M2 {m2:7.1f}  M3 {m3:7.1f}\n"
                   f"shoulder {d(st['phi1']):7.1f}\n"
                   f"elbow    {d(wrap(st['phi2'] - st['phi1'])):7.1f}  (relative)\n"
                   f"wrist    {d(wrap(st['phi3'] - st['phi2'])):7.1f}  (relative)\n"
                   f"hand abs {d(wrap(st['phi3'])):7.1f}\n"
                   f"pen      ({st['P'][0]:6.1f}, {st['P'][1]:6.1f}) mm\n\n"
                   f"transmission margin (1 = ideal, 0 = dead point)\n"
                   f"  elbow pgram   {m[0]:.2f}\n"
                   f"  wrist stage A {m[1]:.2f}\n"
                   f"  wrist stage B {m[2]:.2f}\n")
            if st["flipped"]:
                txt += "\n!! passed a dead point: a parallelogram\n   flipped to anti-parallelogram"
            elif min(m) < 0.25:
                txt += "\n!  near a dead point"
            if self.unreachable:
                txt += "\n!  target unreachable / outside limits"
            self.info.config(text=txt, fg="#c62828" if st["flipped"] else "black")

        # ---- main loop
        def tick(self):
            dt = 0.016
            g = self.g
            if self.mode.get() == "ik":
                if self.queue:
                    n = max(1, int(self.speed.get() * dt))     # queue is 1 mm spaced
                    for _ in range(n):
                        if not self.queue:
                            break
                        self.target, self.pen = self.queue.pop(0)
                        self.step_ik(g, draw=True)
                else:
                    self.pen = self.mouse_down
                    self.step_ik(g, draw=True)
            else:
                self.pen = self.pen_var.get()
                st = self.link.solve(*(math.radians(v.get()) for v in self.mvars), track=True)
                self.unreachable = st is None
                if st:
                    self.ink(st)
            if self.link.prev:
                self.render(self.link.prev)
                self.readout(self.link.prev)
            self.root.after(int(dt * 1000), self.tick)

        def step_ik(self, g, draw):
            fixed = math.radians(self.hand_deg.get()) if self.hand_mode.get() == "fixed" else None
            sol = solve_ik(g, self.target, self.prev_phi3, fixed)
            self.unreachable = sol is None
            if sol is None:
                return
            self.prev_phi3 = sol["phi3"]
            st = self.link.solve(*sol["inputs"])
            if st and draw:
                self.ink(st)

        def ink(self, st):
            P = st["P"]
            if self.pen and self.last_pen is not None and dist(P, self.last_pen) < 20:
                self.cv.create_line(*self.w2c(self.last_pen), *self.w2c(P), fill="#1a237e", width=2,
                                    capstyle="round", tags="ink")
            self.last_pen = P if self.pen else None

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    run_gui()
