"""
arm_linkage.py - tkinter simulator for the planar 3-joint drawing arm.

    python arm_linkage.py           GUI
    python arm_linkage.py --test    numerical self-checks

Modules
    arm_kinematics.py   geometry, human range-of-motion profiles, FK/IK, dynamics
    path_planner.py     stroke ordering, human-like timing, min-energy redundancy DP
    vector_input.py     SVG / raster image -> strokes
"""
import math
import sys
import threading
import time

import numpy as np

from arm_kinematics import (D, DEFAULT_ROM, ROM_PROFILES, Geometry, Linkage,
                            dist, inverse_dynamics, kinetic_energy, mass_params,
                            reachable_grid, solve_ik, wrap)
from path_planner import PlanParams, export_csv, plan_job
from vector_input import fit_to_box, load_image, load_svg

from config import PAPER                       # x0, y0, x1, y1 (mm) from params.toml


# ------------------------------------------------------------ built-in shapes
def builtin_shape(name):
    """Strokes in a 0..200 box, y down (same convention as SVG/images)."""
    c = np.array([100.0, 100.0])
    t = np.linspace(0, 2 * math.pi, 241)
    if name == "circle":
        return [c + 90 * np.stack([np.cos(t), np.sin(t)], 1)]
    if name == "star":
        k = np.arange(11)
        r = np.where(k % 2 == 0, 95, 38)
        a = -math.pi / 2 + k * math.pi / 5
        return [c + np.stack([r * np.cos(a), r * np.sin(a)], 1)]
    if name == "spiral":
        tt = np.linspace(0, 6 * math.pi, 600)
        r = 4 + 4.5 * tt
        return [c + np.stack([r * np.cos(tt), r * np.sin(tt)], 1)]
    if name == "heart":
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        return [c + 5.5 * np.stack([x, y], 1)]
    if name == "lissajous":
        tt = np.linspace(0, 2 * math.pi, 800)
        return [c + np.stack([95 * np.sin(3 * tt), 80 * np.sin(4 * tt)], 1)]
    if name == "flower":
        r = 50 + 40 * np.cos(5 * t)
        return [c + np.stack([r * np.cos(t), r * np.sin(t)], 1)]
    if name == "grid":
        s = []
        for v in range(10, 191, 30):
            s += [np.array([[v, 10.0], [v, 190.0]]), np.array([[10.0, v], [190.0, v]])]
        return s
    raise KeyError(name)


SHAPES = ["circle", "star", "spiral", "heart", "lissajous", "flower", "grid"]


# ================================================================ self-test
def self_test():
    ok = True
    g = Geometry()
    lk = Linkage(g)
    rng = np.random.default_rng(0)

    worst, solved, n = 0.0, 0, 3000
    for _ in range(n):
        P = (rng.uniform(PAPER[0], PAPER[2]), rng.uniform(PAPER[1], PAPER[3]))
        s = solve_ik(g, P)
        if s:
            solved += 1
            worst = max(worst, dist(lk.solve(*s["inputs"])["P"], P))
    print(f"[IK/FK] solved {solved}/{n}, max |FK(IK(P)) - P| = {worst:.1e} mm")
    ok &= solved == n and worst < 1e-6

    s0, s1 = lk.solve(D(-40), D(230), D(40)), lk.solve(D(-10), D(230), D(40))
    print(f"[decoupling] M1 +30 deg -> forearm {math.degrees(wrap(s1['phi2'] - s0['phi2'])):.1e} deg, "
          f"hand {math.degrees(wrap(s1['phi3'] - s0['phi3'])):.1e} deg")

    for name, rom in ROM_PROFILES.items():
        _, _, r = reachable_grid(Geometry(rom=rom), *PAPER[::2], *PAPER[1::2], 10)
        print(f"[ROM] {name:40s} paper reachable {100 * r.mean():5.1f}%")
        ok &= r.all()

    H = mass_params(g)
    t = np.linspace(0, 2, 20001)
    dt = t[1] - t[0]
    q = np.stack([0.5 * np.sin(3 * t), 1 + 0.8 * np.sin(2 * t + 1), 0.3 * np.cos(5 * t)], 1)
    qd = np.gradient(q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    tau = inverse_dynamics(H, q, qd, qdd)
    err = np.abs(np.gradient(kinetic_energy(H, q, qd), dt) - (tau * qd).sum(1))[5:-5].max()
    print(f"[dynamics] |dKE/dt - sum(tau*w)| = {err:.1e} W")
    ok &= err < 1e-5

    strokes = fit_to_box(builtin_shape("flower") + builtin_shape("grid"), PAPER)
    res = {}
    for m in ("greedy", "dp"):
        t0 = time.time()
        res[m] = plan_job(g, strokes, (380, 0), method=m)
        st = res[m].stats
        print(f"[plan:{m:6s}] {time.time() - t0:4.1f}s compute, {st['duration_s']:.1f}s job, "
              f"int(qd'Mqd)={st['kinetic_integral']:.4f}, |work|={sum(st['abs_work_J']):.3f} J, "
              f"peak tau={np.round(st['peak_torque_Nm'], 3)} Nm, unreachable={st['unreachable']}")
    save = 1 - res["dp"].stats["kinetic_integral"] / res["greedy"].stats["kinetic_integral"]
    print(f"[plan] min-energy DP saves {100 * save:.0f}% kinetic effort vs greedy IK")
    ok &= res["dp"].stats["unreachable"] == 0
    print("PASS" if ok else "FAIL")
    return ok


# ===================================================================== GUI
def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from PIL import Image, ImageTk

    if sys.platform == "win32":                 # crisp rendering + correct sizes on HiDPI
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    BLUE, GREEN, ORANGE = "#2f6db5", "#2f9e57", "#e08a1e"
    TEAL, PURPLE, GREY = "#1aa3a3", "#8e55c9", "#9aa0a6"
    MOTOR_COLS = (BLUE, TEAL, PURPLE)

    class App:
        def __init__(self, root):
            self.root = root
            root.title("3-joint drawing arm")
            self.g = Geometry()
            self.params = PlanParams()
            self.link = Linkage(self.g)
            sh = root.winfo_screenheight()
            ch = max(420, min(720, sh - 170))
            cw = int(ch * 1.12)
            self.s = ch / 760.0
            self.ox, self.oy = 180 * self.s, ch / 2
            self.target = (380.0, 0.0)
            self.prev_phi3 = None
            self.mouse_down = False
            self.last_pen = None
            self.unreachable = False
            self.plan = None
            self.play_t = 0.0
            self.playing = False
            self.last_idx = 0
            self.strokes_src = None
            self.busy = False
            self.cur_idx = None
            self.pen = False

            self.cv = tk.Canvas(root, width=cw, height=ch, bg="white", highlightthickness=0)
            self.cv.grid(row=0, column=0, rowspan=2, sticky="nsew")
            root.columnconfigure(0, weight=1)
            root.rowconfigure(0, weight=1)
            self.cv.bind("<Configure>", self.on_resize)
            side = tk.Frame(root, padx=6, pady=6)
            side.grid(row=0, column=1, sticky="n")
            nb = ttk.Notebook(side)
            nb.pack(fill="x")
            self.build_draw_tab(nb)
            self.build_arm_tab(nb)
            self.build_manual_tab(nb)
            self.build_stats_tab(nb)
            self.info = tk.Label(side, justify="left", font=("Consolas", 9), anchor="w")
            self.info.pack(anchor="w", pady=(8, 0), fill="x")

            self.cv.bind("<ButtonPress-1>", self.on_press)
            self.cv.bind("<B1-Motion>", self.on_move)
            self.cv.bind("<Motion>", self.on_move)
            self.cv.bind("<ButtonRelease-1>", lambda e: setattr(self, "mouse_down", False))

            self.draw_static()
            self.tick()

        # ------------------------------------------------------------ tabs
        def build_draw_tab(self, nb):
            f = tk.Frame(nb, padx=6, pady=6)
            nb.add(f, text="Draw")
            self.mode = tk.StringVar(value="live")
            for txt, val in (("live: arm follows mouse (hold = ink)", "live"),
                             ("planned job playback", "plan"),
                             ("manual: drive the 3 shafts", "manual")):
                tk.Radiobutton(f, text=txt, variable=self.mode, value=val, command=self.on_mode).pack(anchor="w")

            tk.Label(f, text="Built-in shapes", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(6, 0))
            bf = tk.Frame(f)
            bf.pack(anchor="w")
            for i, n in enumerate(SHAPES):
                tk.Button(bf, text=n, width=8, command=lambda n=n: self.load_strokes(builtin_shape(n), n)
                          ).grid(row=i // 4, column=i % 4, padx=1, pady=1)

            tk.Label(f, text="Files", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(6, 0))
            ff = tk.Frame(f)
            ff.pack(anchor="w")
            tk.Button(ff, text="Load SVG...", width=12, command=self.open_svg).pack(side="left", padx=1)
            tk.Button(ff, text="Load image...", width=12, command=self.open_image).pack(side="left", padx=1)
            im = tk.Frame(f)
            im.pack(anchor="w")
            tk.Label(im, text="image →").pack(side="left")
            self.img_mode = tk.StringVar(value="outline")
            for v in ("outline", "hatch", "both"):
                tk.Radiobutton(im, text=v, variable=self.img_mode, value=v).pack(side="left")
            self.hatch = tk.DoubleVar(value=3.0)
            tk.Scale(f, label="hatch spacing (px)", from_=1.5, to=8, resolution=0.5,
                     orient="horizontal", length=250, variable=self.hatch).pack(anchor="w")

            tk.Label(f, text="Playback", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(6, 0))
            pf = tk.Frame(f)
            pf.pack(anchor="w")
            tk.Button(pf, text="▶ / ❚❚", width=7, command=self.toggle_play).pack(side="left", padx=1)
            tk.Button(pf, text="⟲ restart", width=8, command=self.restart).pack(side="left", padx=1)
            tk.Button(pf, text="Clear ink", width=8, command=lambda: self.cv.delete("ink")).pack(side="left", padx=1)
            self.mult = tk.DoubleVar(value=1.0)
            tk.Scale(f, label="time scale (× real time)", from_=0.25, to=20, resolution=0.25,
                     orient="horizontal", length=250, variable=self.mult).pack(anchor="w")
            self.prog = ttk.Progressbar(f, length=250, maximum=1000)
            self.prog.pack(anchor="w")
            self.status = tk.Label(f, text="", fg="#555", anchor="w", justify="left", wraplength=290)
            self.status.pack(anchor="w", fill="x")

        def build_arm_tab(self, nb):
            f = tk.Frame(nb, padx=6, pady=6)
            nb.add(f, text="Arm")
            tk.Label(f, text="Range-of-motion profile", font=("Segoe UI", 9, "bold")).pack(anchor="w")
            self.rom_name = tk.StringVar(value=DEFAULT_ROM)
            ttk.Combobox(f, textvariable=self.rom_name, values=list(ROM_PROFILES), state="readonly",
                         width=40).pack(anchor="w")
            self.rom_name.trace_add("write", lambda *a: self.on_rom())
            self.show_reach = tk.BooleanVar(value=True)
            tk.Checkbutton(f, text="shade reachable pen area", variable=self.show_reach,
                           command=self.draw_reach).pack(anchor="w")

            tk.Label(f, text="Planner", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(6, 0))
            self.w_comfort = tk.DoubleVar(value=self.params.w_comfort)
            tk.Scale(f, label="comfort weight (vs kinetic energy)", from_=0, to=0.5, resolution=0.01,
                     orient="horizontal", length=250, variable=self.w_comfort).pack(anchor="w")
            self.gain = tk.DoubleVar(value=self.params.power_gain)
            tk.Scale(f, label="drawing speed gain k  (v = k·R^⅓)", from_=8, to=80, resolution=1,
                     orient="horizontal", length=250, variable=self.gain).pack(anchor="w")
            self.vmax = tk.DoubleVar(value=self.params.v_max)
            tk.Scale(f, label="max pen-down speed (mm/s)", from_=40, to=600, resolution=10,
                     orient="horizontal", length=250, variable=self.vmax).pack(anchor="w")
            tk.Button(f, text="Re-plan current drawing", command=self.replan).pack(anchor="w", pady=2)

            tk.Label(f, text="Live mode hand angle", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(6, 0))
            self.hand_mode = tk.StringVar(value="auto")
            hf = tk.Frame(f)
            hf.pack(anchor="w")
            tk.Radiobutton(hf, text="auto (comfort)", variable=self.hand_mode, value="auto").pack(side="left")
            tk.Radiobutton(hf, text="fixed:", variable=self.hand_mode, value="fixed").pack(side="left")
            self.hand_deg = tk.DoubleVar(value=20)
            tk.Scale(f, from_=-180, to=180, orient="horizontal", length=250, variable=self.hand_deg).pack(anchor="w")

        def build_manual_tab(self, nb):
            f = tk.Frame(nb, padx=6, pady=6)
            nb.add(f, text="Manual")
            tk.Label(f, text="Manual shafts (deg) - select 'manual' mode", font=("Segoe UI", 9, "bold")).pack(anchor="w")
            self.mvars = []
            for lbl, col in (("M1 shoulder", BLUE), ("M2 elbow drive", TEAL), ("M3 wrist drive", PURPLE)):
                v = tk.DoubleVar()
                tk.Scale(f, label=lbl, fg=col, from_=-180, to=360, resolution=0.5, orient="horizontal",
                         length=250, variable=v).pack(anchor="w")
                self.mvars.append(v)
            self.pen_var = tk.BooleanVar(value=False)
            tk.Checkbutton(f, text="pen down", variable=self.pen_var).pack(anchor="w")

        def build_stats_tab(self, nb):
            f = tk.Frame(nb, padx=6, pady=6)
            nb.add(f, text="Motors")
            self.stats_lbl = tk.Label(f, text="(plan a drawing first)", justify="left",
                                      font=("Consolas", 9), anchor="w")
            self.stats_lbl.pack(anchor="w")
            bf = tk.Frame(f)
            bf.pack(anchor="w", pady=4)
            tk.Button(bf, text="Motor plots", command=self.show_plots).pack(side="left", padx=1)
            tk.Button(bf, text="Export CSV...", command=self.export).pack(side="left", padx=1)
            tk.Button(bf, text="Compare vs greedy IK", command=self.compare).pack(side="left", padx=1)

        # ------------------------------------------------------- transforms
        def w2c(self, p):
            return self.ox + self.s * p[0], self.oy - self.s * p[1]

        def c2w(self, x, y):
            return (x - self.ox) / self.s, (self.oy - y) / self.s

        # ----------------------------------------------------------- events
        def on_resize(self, e):
            s_new = min(e.width / 840.0, e.height / 760.0)
            ox_new, oy_new = 180 * s_new, e.height / 2
            if abs(s_new - self.s) < 1e-6 and abs(oy_new - self.oy) < 0.5:
                return
            k = s_new / self.s
            for tag in ("ink", "preview"):
                self.cv.scale(tag, self.ox, self.oy, k, k)
                self.cv.move(tag, ox_new - self.ox, oy_new - self.oy)
            self.s, self.ox, self.oy = s_new, ox_new, oy_new
            self.cv.delete("paper")
            self.draw_static()
            self.cv.tag_lower("preview")
            self.cv.tag_lower("reach")
            self.cv.tag_lower("paper")

        def on_mode(self):
            if self.mode.get() == "manual" and self.link.prev:
                for v, a in zip(self.mvars, self.link.prev["inputs"]):
                    d = math.degrees(a)
                    v.set(round(d + 360 if d < -180 else d, 1))
            if self.mode.get() != "plan":
                self.playing = False
            self.last_pen = None

        def on_press(self, e):
            self.mouse_down = True
            if self.mode.get() != "live":
                self.mode.set("live")
                self.on_mode()
            self.target = self.c2w(e.x, e.y)

        def on_move(self, e):
            self.target = self.c2w(e.x, e.y)

        def on_rom(self):
            self.g.rom = ROM_PROFILES[self.rom_name.get()]
            self.prev_phi3 = None
            self.draw_reach()
            if self.strokes_src is not None:
                self.replan()

        # ------------------------------------------------------ input files
        def open_svg(self):
            p = filedialog.askopenfilename(filetypes=[("SVG", "*.svg"), ("All", "*.*")])
            if p:
                try:
                    self.load_strokes(load_svg(p), p.split("/")[-1])
                except Exception as ex:
                    messagebox.showerror("SVG", f"Could not read SVG:\n{ex}")

        def open_image(self):
            p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                                                      ("All", "*.*")])
            if not p:
                return
            mode = self.img_mode.get()
            try:
                kw = {"spacing": self.hatch.get()} if mode == "hatch" else {}
                strokes = load_image(p, mode, **kw)
            except Exception as ex:
                messagebox.showerror("Image", f"Could not trace image:\n{ex}")
                return
            self.load_strokes(strokes, p.split("/")[-1])

        def load_strokes(self, strokes, name):
            self.strokes_src = (strokes, name)
            self.replan()

        def replan(self):
            if self.strokes_src is None or self.busy:
                return
            strokes, name = self.strokes_src
            fitted = fit_to_box(strokes, PAPER)
            if not fitted:
                self.status.config(text=f"{name}: nothing to draw")
                return
            self.params.w_comfort = self.w_comfort.get()
            self.params.power_gain = self.gain.get()
            self.params.v_max = self.vmax.get()
            start = self.link.prev["P"] if self.link.prev else self.target
            self.busy = True
            self.playing = False
            npts = sum(len(s) for s in fitted)
            self.status.config(text=f"planning {name}: {len(fitted)} strokes, {npts} vertices ...")
            self.cv.delete("preview")
            for s in fitted:
                pts = [c for p in s for c in self.w2c(p)]
                if len(pts) >= 4:
                    self.cv.create_line(*pts, fill="#cfd8dc", tags="preview")
            self.cv.tag_lower("preview")
            self.cv.tag_lower("reach")
            self.cv.tag_lower("paper")

            def work():
                try:
                    plan = plan_job(self.g, fitted, start, self.params)
                    self.root.after(0, lambda: self.plan_ready(plan, name))
                except Exception as ex:
                    msg = str(ex)
                    self.root.after(0, lambda: self.plan_failed(msg))

            threading.Thread(target=work, daemon=True).start()

        def plan_ready(self, plan, name):
            self.busy = False
            self.plan = plan
            self.play_t = 0.0
            self.last_idx = 0
            self.last_pen = None
            self.mode.set("plan")
            self.playing = True
            st = plan.stats
            self.status.config(text=f"{name}: {st['duration_s']:.1f}s job, ink {st['ink_mm']:.0f} mm, "
                                    f"travel {st['travel_mm']:.0f} mm"
                                    + (f", {st['unreachable']} unreachable samples" if st["unreachable"] else ""))
            self.show_stats()

        def plan_failed(self, msg):
            self.busy = False
            self.status.config(text=f"planning failed: {msg}")

        def toggle_play(self):
            if self.plan:
                self.mode.set("plan")
                self.playing = not self.playing

        def restart(self):
            if self.plan:
                self.mode.set("plan")
                self.play_t, self.last_idx, self.last_pen = 0.0, 0, None
                self.playing = True

        # ----------------------------------------------------------- stats
        def show_stats(self):
            st = self.plan.stats
            row = lambda k, f: "  ".join(f"{v:{f}}" for v in st[k])
            self.stats_lbl.config(text=(
                f"job time        {st['duration_s']:8.1f} s\n"
                f"ink / travel    {st['ink_mm']:6.0f} / {st['travel_mm']:.0f} mm\n"
                f"∫ q̇ᵀMq̇ dt      {st['kinetic_integral']:10.5f}\n\n"
                f"                 M1       M2       M3\n"
                f"range   deg  {row('motor_range_deg', '8.1f')}\n"
                f"peak    rpm  {row('peak_speed_rpm', '8.2f')}\n"
                f"peak    N·m  {row('peak_torque_Nm', '8.4f')}\n"
                f"rms     N·m  {row('rms_torque_Nm', '8.4f')}\n"
                f"+work   J    {row('pos_work_J', '8.4f')}\n"
                f"|work|  J    {row('abs_work_J', '8.4f')}\n\n"
                f"masses {self.g.m1}/{self.g.m2}/{self.g.m3} kg, reflected motor\n"
                f"inertia {self.g.J_motor[0]:.0e} kg·m² each, horizontal plane\n"
                f"(no gravity, no friction)"))

        def export(self):
            if not self.plan:
                return
            p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if p:
                export_csv(self.plan, p)
                self.status.config(text=f"exported {len(self.plan.t)} samples to {p}")

        def compare(self):
            if self.strokes_src is None or self.busy:
                return
            fitted = fit_to_box(self.strokes_src[0], PAPER)
            start = self.link.prev["P"] if self.link.prev else self.target
            self.status.config(text="planning greedy baseline ...")
            self.busy = True

            def work():
                gr = plan_job(self.g, fitted, start, self.params, method="greedy")
                self.root.after(0, lambda: done(gr))

            def done(gr):
                self.busy = False
                a, b = gr.stats, self.plan.stats
                pct = lambda k: 100 * (1 - sum(np.atleast_1d(b[k])) / max(1e-12, sum(np.atleast_1d(a[k]))))
                messagebox.showinfo("Min-energy DP vs greedy IK", (
                    f"                 greedy      DP     saving\n"
                    f"∫ q̇ᵀMq̇ dt     {a['kinetic_integral']:8.4f} {b['kinetic_integral']:8.4f}  {pct('kinetic_integral'):5.1f}%\n"
                    f"|work| J       {sum(a['abs_work_J']):8.4f} {sum(b['abs_work_J']):8.4f}  {pct('abs_work_J'):5.1f}%\n"
                    f"peak τ sum N·m {sum(a['peak_torque_Nm']):8.4f} {sum(b['peak_torque_Nm']):8.4f}  {pct('peak_torque_Nm'):5.1f}%\n"
                    f"rms τ sum N·m  {sum(a['rms_torque_Nm']):8.4f} {sum(b['rms_torque_Nm']):8.4f}  {pct('rms_torque_Nm'):5.1f}%"))
                self.status.config(text="comparison done")

            threading.Thread(target=work, daemon=True).start()

        def show_plots(self):
            if not self.plan:
                return
            pl = self.plan
            w = tk.Toplevel(self.root)
            w.title("Motor trajectories")
            W, Hh = 900, 640
            c = tk.Canvas(w, width=W, height=Hh, bg="white")
            c.pack()
            step = max(1, len(pl.t) // 1500)
            t = pl.t[::step]
            panels = [("shaft angle (deg)", np.degrees(pl.motors - pl.motors[0])[::step]),
                      ("speed (rpm)", (pl.omega * 60 / (2 * math.pi))[::step]),
                      ("torque (N·m)", pl.tau[::step])]
            ph = Hh / 3
            for i, (title, Y) in enumerate(panels):
                y0, y1 = i * ph + 22, (i + 1) * ph - 14
                lo, hi = float(Y.min()), float(Y.max())
                if hi - lo < 1e-12:
                    hi = lo + 1
                X = lambda tt: 60 + (W - 80) * tt / max(t[-1], 1e-9)
                Yc = lambda v: y1 - (y1 - y0) * (v - lo) / (hi - lo)
                c.create_rectangle(60, y0, W - 20, y1, outline="#ddd")
                if lo < 0 < hi:
                    c.create_line(60, Yc(0), W - 20, Yc(0), fill="#eee")
                c.create_text(64, y0 - 10, anchor="w", text=title, font=("Segoe UI", 9, "bold"))
                c.create_text(56, y0, anchor="e", text=f"{hi:.3g}", font=("Segoe UI", 8))
                c.create_text(56, y1, anchor="e", text=f"{lo:.3g}", font=("Segoe UI", 8))
                for m in range(3):
                    pts = [v for tt, yy in zip(t, Y[:, m]) for v in (X(tt), Yc(yy))]
                    c.create_line(*pts, fill=MOTOR_COLS[m], width=1.5)
            c.create_text(W - 20, Hh - 4, anchor="se", text=f"time 0 - {t[-1]:.1f} s   "
                          "blue M1 shoulder · teal M2 elbow · purple M3 wrist", font=("Segoe UI", 8))

        # --------------------------------------------------------- drawing
        def draw_static(self):
            c = self.cv
            x0, y0 = self.w2c((PAPER[0], PAPER[3]))
            x1, y1 = self.w2c((PAPER[2], PAPER[1]))
            c.create_rectangle(x0, y0, x1, y1, fill="#fbf8ee", outline="#d8d2bd", tags="paper")
            bx0, by0 = self.w2c((-160, 130))
            bx1, by1 = self.w2c((-25, -130))
            c.create_rectangle(bx0, by0, bx1, by1, fill="#eceff1", outline="#cfd8dc", tags="paper")
            c.create_text((bx0 + bx1) / 2, (by0 + by1) / 2, text="body\n\nM1 M2 M3\ncoaxial at\nshoulder",
                          fill="#78909c", font=("Segoe UI", 8), justify="center", tags="paper")
            c.create_text(10, 10, anchor="nw", fill="#607d8b", font=("Segoe UI", 9), tags="paper", text=(
                "right arm, top view.  blue upper arm · green forearm · orange hand/pen\n"
                "teal elbow parallelogram (M2) · purple wrist parallelograms + bell-crank (M3)\n"
                "green shading = pen positions reachable within the range-of-motion profile"))
            self.draw_reach()

        def draw_reach(self):
            self.cv.delete("reach")
            if not self.show_reach.get():
                return
            x0w, y1w = self.c2w(0, 0)
            cw = self.cv.winfo_width() if self.cv.winfo_width() > 10 else int(self.cv["width"])
            ch = self.cv.winfo_height() if self.cv.winfo_height() > 10 else int(self.cv["height"])
            x1w, y0w = self.c2w(cw, ch)
            step = 6.0
            xs, ys, ok = reachable_grid(self.g, x0w, x1w, y0w, y1w, step)
            rgba = np.zeros(ok.shape + (4,), np.uint8)
            rgba[ok] = (120, 200, 140, 60)
            img = Image.fromarray(rgba[::-1], "RGBA")
            wpx = int(round(len(xs) * step * self.s))
            hpx = int(round(len(ys) * step * self.s))
            img = img.resize((wpx, hpx), Image.NEAREST)
            bg = Image.new("RGBA", img.size, (255, 255, 255, 0))
            self._reach_img = ImageTk.PhotoImage(Image.alpha_composite(bg, img))
            cx, cy = self.w2c((xs[0] - step / 2, ys[-1] + step / 2))
            self.cv.create_image(cx, cy, anchor="nw", image=self._reach_img, tags="reach")
            self.cv.tag_raise("reach", "paper")

        def render(self, st):
            c = self.cv
            c.delete("arm")

            def ln(p, q, w, col):
                c.create_line(*self.w2c(p), *self.w2c(q), width=w, fill=col, capstyle="round", tags="arm")

            def dot(p, r, col, fill="white"):
                x, y = self.w2c(p)
                c.create_oval(x - r, y - r, x + r, y + r, outline=col, fill=fill, width=2, tags="arm")

            S, E, W, P = st["S"], st["E"], st["W"], st["P"]
            c.create_polygon(*self.w2c(E), *self.w2c(st["A2"]), *self.w2c(st["B1"]),
                             fill="#e6d9f5", outline=PURPLE, width=2, tags="arm")
            ln(S, st["A1"], 5, PURPLE)
            ln(st["A1"], st["A2"], 2, PURPLE)
            ln(st["B1"], st["B2"], 2, PURPLE)
            ln(W, st["B2"], 5, PURPLE)
            ln(S, st["C2"], 6, TEAL)
            ln(st["C2"], st["T2"], 2.5, TEAL)
            ln(S, E, 12, BLUE)
            ln(st["T2"], W, 9, GREEN)
            ln(W, P, 6, ORANGE)
            for p in (st["A1"], st["A2"], st["B1"], st["B2"], st["C2"], st["T2"]):
                dot(p, 3, GREY)
            dot(E, 6, BLUE)
            dot(W, 6, GREEN)
            for r, col in ((14, PURPLE), (11, TEAL), (8, BLUE)):
                dot(S, r, col, fill="")
            dot(P, 4, "red", fill="red" if self.pen else "white")
            if self.mode.get() == "live":
                x, y = self.w2c(self.target)
                col = "#e53935" if self.unreachable else "#90a4ae"
                c.create_line(x - 7, y, x + 7, y, fill=col, tags="arm")
                c.create_line(x, y - 7, x, y + 7, fill=col, tags="arm")

        def readout(self, st):
            d = math.degrees
            m = st["margins"]
            r = self.g.rom
            el, wr = wrap(st["phi2"] - st["phi1"]), wrap(st["phi3"] - st["phi2"])

            def rng(v, lim):
                f = (v - lim[0]) / (lim[1] - lim[0])
                bar = "".join("█" if abs(i / 9 - f) < 0.06 else "·" for i in range(10))
                return f"[{bar}]"

            txt = (f"inputs  M1 {d(st['inputs'][0]):7.1f}  M2 {d(st['inputs'][1]):7.1f}  M3 {d(st['inputs'][2]):7.1f}\n"
                   f"shoulder {d(st['phi1']):7.1f}  {rng(st['phi1'], r.shoulder)}  ({d(st['phi1']) + 90:+.0f} from lateral)\n"
                   f"elbow    {d(el):7.1f}  {rng(el, r.elbow)}  flexion\n"
                   f"wrist    {d(wr):7.1f}  {rng(wr, r.wrist)}\n"
                   f"pen ({st['P'][0]:6.1f}, {st['P'][1]:6.1f}) mm\n"
                   f"transmission  elbow {m[0]:.2f}  wristA {m[1]:.2f}  wristB {m[2]:.2f}\n")
            if self.mode.get() == "plan" and self.plan is not None and self.cur_idx is not None:
                i = self.cur_idx
                w = self.plan.omega[i] * 60 / (2 * math.pi)
                tq = self.plan.tau[i]
                txt += (f"t {self.plan.t[i]:6.2f}/{self.plan.t[-1]:.2f} s\n"
                        f"rpm  {w[0]:7.2f} {w[1]:7.2f} {w[2]:7.2f}\n"
                        f"N·m  {tq[0]:7.4f} {tq[1]:7.4f} {tq[2]:7.4f}\n")
            if st["flipped"]:
                txt += "!! dead point passed: parallelogram flipped\n"
            elif min(m) < 0.25:
                txt += "!  near a dead point\n"
            if self.unreachable:
                txt += "!  target outside range of motion\n"
            self.info.config(text=txt, fg="#c62828" if st["flipped"] else "black")

        # ------------------------------------------------------- main loop
        def tick(self):
            frame = 0.016
            mode = self.mode.get()
            if mode == "plan" and self.plan is not None:
                self.step_plan(frame)
            elif mode == "manual":
                self.pen = self.pen_var.get()
                st = self.link.solve(*(math.radians(v.get()) for v in self.mvars), track=True)
                self.unreachable = st is None
                if st:
                    self.ink([st["P"]])
            else:
                self.pen = self.mouse_down
                fixed = math.radians(self.hand_deg.get()) if self.hand_mode.get() == "fixed" else None
                sol = solve_ik(self.g, self.target, self.prev_phi3, fixed)
                self.unreachable = sol is None
                if sol:
                    self.prev_phi3 = sol["phi3"]
                    st = self.link.solve(*sol["inputs"])
                    if st:
                        self.ink([st["P"]])
            if self.link.prev:
                self.render(self.link.prev)
                self.readout(self.link.prev)
            self.root.after(int(frame * 1000), self.tick)

        def step_plan(self, frame):
            pl = self.plan
            if self.playing:
                self.play_t = min(self.play_t + frame * self.mult.get(), pl.t[-1])
            idx = min(int(self.play_t / self.params.dt), len(pl.t) - 1)
            pts = []
            for i in range(self.last_idx, idx + 1):
                self.pen = bool(pl.pen[i])
                st = self.link.solve(*pl.motors[i])          # FK of the real linkage
                if st is None:
                    continue
                if self.pen:
                    pts.append(st["P"])
                else:
                    self.ink(pts)
                    pts = []
                    self.last_pen = None
            self.ink(pts)
            self.last_idx = max(self.last_idx, idx + 1)
            self.cur_idx = idx
            self.unreachable = not pl.valid[idx]
            self.prog["value"] = 1000 * self.play_t / max(pl.t[-1], 1e-9)
            if self.play_t >= pl.t[-1]:
                self.playing = False

        def ink(self, pts):
            if not self.pen and not pts:
                self.last_pen = None
                return
            if not pts:
                return
            chain = ([self.last_pen] if self.last_pen is not None else []) + list(pts)
            if len(chain) >= 2:
                xy = [v for p in chain for v in self.w2c(p)]
                self.cv.create_line(*xy, fill="#1a237e", width=2, capstyle="round", joinstyle="round", tags="ink")
            self.last_pen = chain[-1] if self.pen else None

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    run_gui()
