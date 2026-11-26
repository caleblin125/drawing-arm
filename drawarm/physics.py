"""
physics.py - time-domain physics test of the complete single-motor machine,
with Monte-Carlo manufacturing tolerances (runs on a laptop or a Slurm cluster).

    python -m drawarm physics prepare heart -o scenario.pkl      # cams, followers, springs (numpy only after this)
    python -m drawarm physics run scenario.pkl --first 0 --count 10 -o results/   # Monte-Carlo runs
    python -m drawarm physics summarize results/                  # report + plots

Model (planar, SI units inside)
  motor      DC gearmotor, linear torque-speed (stall torque, no-load speed, voltage)
  worm       single-start, ratio N, compliant mesh k_w; forward efficiency eta,
             self-locking backwards (back-drive efficiency 2 - 1/eta < 0)
  camshaft   inertia J_c; every cam pushes back through its roller contact
  rollers    one-sided penalty contact on the cam profile psi_cam(theta)
             (+ random print error of the profile); contact torque -> camshaft by
             virtual work; contact LOSS = lift-off
  followers  massless (quasi-static): mesh spring (with backlash) vs contact
             spring, solved exactly each step (piecewise linear in psi)
  gears      hub <-> follower mesh, stiffness k_g, backlash b (dead zone)
  hub springs torsion springs S(theta) on the shoulder hubs (preload)
  arm        3-link rigid dynamics in absolute angles (exact mass matrix)
  pen        Coulomb drag (smoothed) while the pen is down; pen follower cable
             tension keeps the pen roller on its cam
  tolerances link lengths and crank radii (pen position via the full closed-loop
             FK of the toleranced linkage), cam profile error, backlash,
             preload, stiffness, friction, motor voltage / torque
Outputs per run: pen-tip error vs the ideal drawing (RMS, max), roller lift-off
time and depth, gear load reversals, min contact force, motor torque margin,
camshaft speed ripple.
"""
import json
import math
import os
import pickle
import time

import numpy as np

from .kinematics import Geometry, Linkage, mass_params

_PH = {}
try:
    from .config import CFG
    _PH = CFG.get("physics", {})
except Exception:                                                     # pragma: no cover
    pass


DEFAULTS = dict(
    dt=_PH.get("dt", 2e-4),                    # s
    revolutions=_PH.get("revolutions", 2.0),   # start from rest; the last revolution is scored
    k_mesh=_PH.get("k_mesh", 300.0),           # N m/rad at the hub (printed teeth + spokes)
    c_mesh=_PH.get("c_mesh", 0.05),            # N m s/rad
    k_contact=_PH.get("k_contact", 2.0e4),     # N/m roller contact incl. follower arm bending
    c_contact=_PH.get("c_contact", 5.0),       # N s/m
    k_worm=_PH.get("k_worm", 150.0),           # N m/rad at the camshaft
    c_worm=_PH.get("c_worm", 0.05),
    eta_worm=_PH.get("eta_worm", 0.4),
    J_cam=_PH.get("J_cam", 2.0e-3),            # kg m^2 camshaft + cams + wheel
    J_motor=_PH.get("J_motor", 1.0e-6),        # kg m^2 at the gearmotor output
    stall=_PH.get("stall", 0.08),              # N m gearmotor stall torque at nominal voltage
    no_load_rpm=_PH.get("no_load_rpm", 230.0), # gearmotor no-load speed at nominal voltage
    pen_mu=_PH.get("pen_mu", 0.4),
    pen_normal=_PH.get("pen_normal", 1.0),     # N
    arm_damping=_PH.get("arm_damping", 0.002), # N m s/rad per joint
)

TOL = dict(                                    # Monte-Carlo ranges (uniform +-)
    link_mm=_PH.get("tol_link_mm", 0.15),
    cam_mm=_PH.get("tol_cam_mm", 0.10),        # cam profile print error (smooth, random)
    backlash_mm=tuple(_PH.get("tol_backlash_mm", [0.03, 0.20])),
    preload=_PH.get("tol_preload", 0.20),
    stiffness=_PH.get("tol_stiffness", 0.30),
    pen_mu=tuple(_PH.get("tol_pen_mu", [0.2, 0.6])),
    voltage=_PH.get("tol_voltage", 0.10),
    stall=_PH.get("tol_stall", 0.15),
)


# ================================================================ prepare
def prepare(drawing="heart"):
    """Design the machine for a drawing and freeze everything the simulation
    needs into plain numpy / float data (no CadQuery needed afterwards)."""
    from .cams import plan_for_cams
    from .cartridge import CartridgeParams, cams_for, design_followers
    from .config import PAPER
    from .inputs import fit_to_box
    from .kinematics import paper_motor_range
    from .shapes import load_drawing
    from . import springs as spr

    g = Geometry()
    fol = design_followers(g)
    plan, _ = plan_for_cams(g, fit_to_box(load_drawing(drawing), PAPER))
    cs, probs = cams_for(plan, g, fol)
    sp = spr.design(paper_motor_range(g), fol.spring_sign)
    cp = CartridgeParams()
    axes = {}
    for key, ax in fol.axes.items():
        des = cs.pen_axis if key == "pen" else cs.axes[("M1", "M2", "M3").index(key)]
        axes[key] = dict(psi=np.asarray(des["psi"]), alpha=np.asarray(des["alpha"]), l=ax["l"] / 1000,
                         G=ax["G"], zf=ax["zf"], inward=ax["inward"], M_ref=ax["M_ref"], psi_ref=ax["psi_ref"])
    pen_lever = cp.pen_lift / cp.pen_rise * fol.axes["pen"]["l"]
    sc = dict(
        drawing=drawing, problems=probs,
        geometry={k: (list(v) if isinstance(v, tuple) else v) for k, v in g.__dict__.items() if k != "rom"},
        theta=np.asarray(cs.theta), motors=np.asarray(cs.motors), pen=np.asarray(cs.pen), period=cs.period,
        axes=axes, springs={k: dict(P=v["P"], k=v["k"], sign=v["sign"], theta_mid=v["theta_mid"])
                            for k, v in sp.items()},
        module=fol.params.module, z_hub=fol.params.z_hub,
        pen_lift_per_rad=pen_lever / 1000 * 1.0,                       # m of pen per rad of pen follower
        pen_cable_torque=1.0 * pen_lever / 1000 * 0.8,                  # N m (pen spring 1 N on the lever)
        worm_ratio=CFG.get("machine", {}).get("wheel_teeth", 30),
    )
    return sc


# ============================================================== simulate
def _solve_pl(resid, pts, fallback):
    """Root of a decreasing piecewise-linear function with kinks at pts."""
    xs = [pts[0] - 1.0] + list(pts) + [pts[-1] + 1.0]
    rs = [resid(x) for x in xs]
    for (x0, r0), (x1, r1) in zip(zip(xs, rs), zip(xs[1:], rs[1:])):
        if r0 >= 0 >= r1:
            if r0 == r1:                       # flat zero: floating in backlash, touching the cam
                return min(max(fallback, x0), x1)
            return x0 + (x1 - x0) * r0 / (r0 - r1)
    if rs[0] < 0:                              # beyond the left end (linear)
        return xs[0] - (xs[0] - xs[1]) * rs[0] / (rs[0] - rs[1]) if rs[0] != rs[1] else xs[0]
    return xs[-1] + (xs[-1] - xs[-2]) * rs[-1] / (rs[-2] - rs[-1]) if rs[-2] != rs[-1] else xs[-1]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _smooth_noise(rng, n, amp, modes=12):
    """Smooth periodic random profile error with RMS ~ amp/sqrt(2)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out = np.zeros(n)
    for m in range(1, modes + 1):
        out += rng.normal() / m * np.cos(m * th) + rng.normal() / m * np.sin(m * th)
    out *= amp / (np.abs(out).max() + 1e-12)
    return out


def sample_tolerances(seed):
    """One Monte-Carlo draw (seed 0 = nominal)."""
    rng = np.random.default_rng(seed)
    if seed == 0:
        return dict(seed=0, link=np.zeros(6), cam=None, backlash_mm=[0.10, 0.10, 0.10], preload=[1, 1, 1],
                    stiff=1.0, pen_mu=DEFAULTS["pen_mu"], voltage=1.0, stall=1.0)
    u = lambda a: rng.uniform(-a, a)
    return dict(seed=int(seed),
                link=np.array([u(TOL["link_mm"]) for _ in range(6)]),
                cam=int(rng.integers(1 << 30)),
                backlash_mm=[float(rng.uniform(*TOL["backlash_mm"])) for _ in range(3)],
                preload=[1 + u(TOL["preload"]) for _ in range(3)],
                stiff=1 + u(TOL["stiffness"]),
                pen_mu=float(rng.uniform(*TOL["pen_mu"])),
                voltage=1 + u(TOL["voltage"]),
                stall=1 + u(TOL["stall"]))


def simulate(sc, tol=None, opts=None, trace=False):
    """One run.  Returns a dict of metrics (and traces if trace=True)."""
    o = dict(DEFAULTS, **(opts or {}))
    tol = tol or sample_tolerances(0)
    t_wall = time.time()
    gd = dict(sc["geometry"])
    g_nom = Geometry(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in gd.items()})
    g = Geometry(**{**g_nom.__dict__, "J_motor": (0.0, 0.0, 0.0)})
    H = mass_params(g)                                                 # SI (kg m^2)
    Lm = np.array([g.L1, g.L2, g.L3]) / 1000
    # toleranced linkage for the pen position (closed-loop FK)
    dl = tol["link"]
    g_tol = Geometry(**{**g_nom.__dict__, "L1": g.L1 + dl[0], "L2": g.L2 + dl[1], "L3": g.L3 + dl[2],
                        "r_e": g.r_e + dl[3], "r_a": g.r_a + dl[4], "r_b": g.r_b + dl[5]})
    link_tol, link_nom = Linkage(g_tol), Linkage(g_nom)
    offs = np.array([0.0, math.pi, g.off_a])

    th_grid = sc["theta"]
    n = len(th_grid)
    rng = np.random.default_rng(tol["cam"]) if tol["cam"] is not None else None
    ax_names = ["M1", "M2", "M3", "pen"]
    psi_cam, dpsi_cam, alpha = {}, {}, {}
    for nm in ax_names:
        a = sc["axes"][nm]
        p = a["psi"].copy()
        if rng is not None:                                            # print error of the cam rim
            p = p + a["inward"] * _smooth_noise(rng, n, TOL["cam_mm"] / 1000) / (a["l"] * np.cos(a["alpha"]))
        psi_cam[nm] = p
        dpsi_cam[nm] = np.gradient(np.unwrap(p), th_grid)
        alpha[nm] = a["alpha"]
    r_hub = sc["module"] * sc["z_hub"] / 2 / 1000
    back = np.array(tol["backlash_mm"]) * 2 / 1000 / r_hub             # rad at the hub (both gears thinned)
    k_g = o["k_mesh"] * tol["stiff"]
    k_c = o["k_contact"] * tol["stiff"]
    sp = sc["springs"]
    N = sc["worm_ratio"]
    eta = o["eta_worm"]
    w0 = o["no_load_rpm"] * 2 * math.pi / 60 * tol["voltage"]
    t_stall = o["stall"] * tol["stall"] * tol["voltage"]

    def interp(arr, th):
        x = (th % (2 * np.pi)) / (2 * np.pi) * n
        i = int(x) % n
        f = x - int(x)
        return arr[i] * (1 - f) + arr[(i + 1) % n] * f

    # ---- state: arm q (absolute link angles), qd; camshaft th, w; motor angle (camshaft units) thm, wm
    q = sc["motors"][0] - offs
    qd = np.zeros(3)
    th, w = 0.0, 0.0
    thm, wm = 0.0, 0.0
    dt = o["dt"]
    T_rev = sc["period"]
    steps = int(3 * o["revolutions"] * T_rev / dt)                  # cap (a stalled motor ends the run)
    rec = []
    lift = np.zeros(4)
    lift_depth = np.zeros(4)
    min_F = np.full(4, np.inf)
    reversals = np.zeros(3, int)
    last_sign = np.zeros(3)
    peak_motor = 0.0
    psi = np.array([interp(psi_cam[nm], 0.0) for nm in ax_names])
    pen_ref = sc["axes"]["pen"]["psi_ref"]
    th_end = o["revolutions"] * 2 * math.pi
    score_from = th_end - 2 * math.pi
    k = 0
    while th < th_end and k < steps:
        k += 1
        M = q + offs
        # --- hub springs
        tau_s = np.zeros(3)
        for i, nm in enumerate(("M1", "M2", "M3")):
            s = sp[nm]
            d = _wrap(M[i] - s["theta_mid"])
            tau_s[i] = (s["sign"] * s["P"] * tol["preload"][i] - s["k"] * d) / 1000
        # --- followers (quasi-static): mesh spring vs one-sided contact
        tau_mesh = np.zeros(3)
        tau_cam = 0.0
        contact_torque = np.zeros(4)
        for i, nm in enumerate(ax_names):
            a = sc["axes"][nm]
            pc = interp(psi_cam[nm], th)
            kc_t = k_c * a["l"] ** 2                                    # contact, torsional at the follower
            if nm != "pen":
                G = a["G"]
                kg_t = k_g * G * G                                      # mesh, seen from the follower
                hub_term = _wrap(M[i] - a["M_ref"])
                b2 = back[i] / 2
                # delta(psi) = hub_term + G (psi - psi_ref); mesh torque on follower = -G k_g f(delta)
                # contact torque = -inward k_c,t max(0, inward (psi - pc))
                def resid(ps):
                    dlt = hub_term + G * (ps - a["psi_ref"])
                    f = dlt - b2 if dlt > b2 else (dlt + b2 if dlt < -b2 else 0.0)
                    gap = a["inward"] * (ps - pc)
                    return -G * k_g * f - a["inward"] * kc_t * max(0.0, gap)
                ps = _solve_pl(resid, sorted([a["psi_ref"] + (-hub_term - b2) / G,
                                              a["psi_ref"] + (-hub_term + b2) / G, pc]), pc)
                dlt = hub_term + G * (ps - a["psi_ref"])
                f = dlt - b2 if dlt > b2 else (dlt + b2 if dlt < -b2 else 0.0)
                tau_mesh[i] = -k_g * f                                  # on the hub
                gap = a["inward"] * (ps - pc)
                ct = kc_t * max(0.0, gap)
                sgn = np.sign(f)
                if sgn != 0 and last_sign[i] != 0 and sgn != last_sign[i] and th >= score_from:
                    reversals[i] += 1
                if sgn != 0:
                    last_sign[i] = sgn
            else:
                # pen follower: cable tension pushes inward, contact pushes back
                tq = sc["pen_cable_torque"]
                gap0 = tq / kc_t                                        # equilibrium penetration
                ps = pc + a["inward"] * gap0
                ct = tq
            psi[i] = ps
            contact_torque[i] = ct
            Fn = ct / (a["l"] * max(math.cos(interp(alpha[nm], th)), 0.2))
            if th >= score_from:
                min_F[i] = min(min_F[i], Fn)
                if ct <= 0:
                    lift[i] += dt
                    lift_depth[i] = max(lift_depth[i], abs(a["inward"] * (ps - pc)) * a["l"] * 1000)
            # reaction on the camshaft (virtual work: Q_theta = -dV/dtheta = ct * inward * psi_cam')
            tau_cam += ct * a["inward"] * interp(dpsi_cam[nm], th)
        # --- pen
        pen_h = abs(psi[3] - pen_ref) * sc["pen_lift_per_rad"]
        pen_down = pen_h < 0.3e-3
        c = np.cos(q)
        s_ = np.sin(q)
        Pd = np.array([-(Lm * s_ * qd).sum(), (Lm * c * qd).sum()])    # pen velocity (m/s)
        tau_pen = np.zeros(3)
        if pen_down:
            v = np.hypot(*Pd)
            Fp = -tol["pen_mu"] * DEFAULTS["pen_normal"] * Pd / math.sqrt(v * v + 1e-6)
            for j in range(3):
                tau_pen[j] = Lm[j] * (-s_[j] * Fp[0] + c[j] * Fp[1])
        # --- arm dynamics (absolute coordinates)
        dq = q[:, None] - q[None, :]
        Mm = H * np.cos(dq)
        h = (H * np.sin(dq) * (qd ** 2)[None, :]).sum(1)
        tau = tau_mesh + tau_s + tau_pen - o["arm_damping"] * qd
        qdd = np.linalg.solve(Mm, tau - h)
        # --- worm, camshaft, motor
        tw = o["k_worm"] * (thm - th) + o["c_worm"] * (wm - w)          # worm -> camshaft torque
        load_m = tw / (N * eta) if tw >= 0 else tw * (2 - 1 / eta) / N   # motor-side load (self-locking)
        wmot = wm * N
        tm = t_stall * (1 - wmot / w0)
        tm = min(max(tm, -t_stall), t_stall)
        peak_motor = max(peak_motor, abs(load_m) / max(t_stall, 1e-9)) if th >= score_from else peak_motor
        thm_dd = (tm - load_m) / (o["J_motor"] * N)                     # motor angle in camshaft units
        w_dot = (tw + tau_cam) / o["J_cam"]
        # --- integrate (semi-implicit Euler)
        qd = qd + dt * qdd
        q = q + dt * qd
        w = w + dt * w_dot
        th = th + dt * w
        wm = wm + dt * thm_dd
        thm = thm + dt * wm
        if trace or (th >= score_from and k % 25 == 0):
            rec.append((k * dt, th, *(q + offs), pen_down, w, load_m / t_stall))
    rec = np.array(rec, float)
    # ---- score the last revolution: pen tip (toleranced FK) vs ideal (nominal FK at the same cam angle)
    errs = []
    speed = []
    for row in rec:
        t_, th_, m1, m2, m3, pdn, w_, ld = row
        if th_ < score_from:
            continue
        st = link_tol.solve(m1, m2, m3, track=True)
        mi = np.array([interp(sc["motors"][:, j], th_) for j in range(3)])
        ideal = link_nom.solve(*mi, track=True)
        if st is None or ideal is None:
            continue
        if pdn > 0.5 or interp(sc["pen"], th_) > 0.5:
            errs.append(math.hypot(st["P"][0] - ideal["P"][0], st["P"][1] - ideal["P"][1]))
        speed.append(w_)
    errs = np.array(errs) if errs else np.array([np.nan])
    speed = np.array(speed) if speed else np.array([np.nan])
    w_nom = 2 * math.pi / T_rev
    res = dict(
        seed=tol["seed"], drawing=sc["drawing"], wall_s=round(time.time() - t_wall, 1),
        pen_err_rms_mm=float(np.sqrt(np.nanmean(errs ** 2))), pen_err_max_mm=float(np.nanmax(errs)),
        lift_off_ms=[float(1000 * x) for x in lift], lift_depth_mm=[float(x) for x in lift_depth],
        min_contact_N=[float(x) if np.isfinite(x) else None for x in min_F],
        mesh_reversals=[int(x) for x in reversals],
        motor_peak_load=float(peak_motor),                              # fraction of stall torque
        cam_speed_ratio=float(np.nanmean(speed) / w_nom), cam_speed_ripple=float(np.nanstd(speed) / w_nom),
        stalled=bool(th < th_end),
        tolerances={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in tol.items()},
    )
    if trace:
        res["trace"] = rec
    return res


# ============================================================== CLI helpers
def run_batch(scenario, first, count, outdir, opts=None):
    with open(scenario, "rb") as f:
        sc = pickle.load(f)
    os.makedirs(outdir, exist_ok=True)
    for seed in range(first, first + count):
        r = simulate(sc, sample_tolerances(seed), opts)
        with open(os.path.join(outdir, f"run_{seed:05d}.json"), "w") as f:
            json.dump(r, f)
        print(f"seed {seed}: pen err rms {r['pen_err_rms_mm']:.3f} max {r['pen_err_max_mm']:.3f} mm, "
              f"lift-off {max(r['lift_off_ms']):.1f} ms, reversals {r['mesh_reversals']}, "
              f"motor {100 * r['motor_peak_load']:.0f}% stall, {r['wall_s']} s", flush=True)


def summarize(resdir, out_md=None):
    rs = []
    for fn in sorted(os.listdir(resdir)):
        if fn.endswith(".json"):
            with open(os.path.join(resdir, fn)) as f:
                rs.append(json.load(f))
    if not rs:
        return "no results"
    arr = lambda k: np.array([r[k] for r in rs], float)
    rms, mx = arr("pen_err_rms_mm"), arr("pen_err_max_mm")
    lift = np.array([max(r["lift_off_ms"]) for r in rs])
    rev = np.array([sum(r["mesh_reversals"]) for r in rs])
    mot = arr("motor_peak_load")
    stalled = sum(r["stalled"] for r in rs)
    pct = lambda x, q: float(np.nanpercentile(x, q))
    nom = next((r for r in rs if r["seed"] == 0), None)
    lines = [f"# Physics Monte-Carlo – `{rs[0]['drawing']}` ({len(rs)} runs)", "",
             "| metric | nominal | median | 95th pct | worst |", "|---|---|---|---|---|"]

    def row(name, x, nv, fmt="{:.3f}"):
        lines.append(f"| {name} | {fmt.format(nv) if nv is not None else '–'} | {fmt.format(pct(x, 50))} | "
                     f"{fmt.format(pct(x, 95))} | {fmt.format(np.nanmax(x))} |")
    row("pen error RMS (mm)", rms, nom and nom["pen_err_rms_mm"])
    row("pen error max (mm)", mx, nom and nom["pen_err_max_mm"])
    row("roller lift-off (ms, worst axis)", lift, nom and max(nom["lift_off_ms"]), "{:.1f}")
    row("gear load reversals (sum)", rev, nom and sum(nom["mesh_reversals"]), "{:.0f}")
    row("motor peak load (x stall)", mot, nom and nom["motor_peak_load"], "{:.2f}")
    lines += ["", f"stalled runs: {stalled} / {len(rs)}; runs with any lift-off: {(lift > 0).sum()}; "
                  f"runs with gear reversals: {(rev > 0).sum()}"]
    # sensitivity: correlation of pen error with each tolerance
    feats = {"backlash (mean mm)": [np.mean(r["tolerances"]["backlash_mm"]) for r in rs],
             "preload (min factor)": [min(r["tolerances"]["preload"]) for r in rs],
             "stiffness factor": [r["tolerances"]["stiff"] for r in rs],
             "pen friction mu": [r["tolerances"]["pen_mu"] for r in rs],
             "link error (max |mm|)": [max(abs(np.array(r["tolerances"]["link"]))) for r in rs],
             "voltage factor": [r["tolerances"]["voltage"] for r in rs]}
    lines += ["", "| tolerance | correlation with pen error RMS |", "|---|---|"]
    for k, v in feats.items():
        v = np.array(v, float)
        c = np.corrcoef(v, rms)[0, 1] if np.nanstd(v) > 0 and np.nanstd(rms) > 0 else float("nan")
        lines.append(f"| {k} | {c:+.2f} |")
    text = "\n".join(lines)
    if out_md:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(text)
    return text
