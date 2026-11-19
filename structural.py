"""
structural.py - loads, stress and strain in every link, pin joint, gear and cam
of the drawing arm, plus the pen-tip error they cause.

    python structural.py [shape|file.svg]   -> prints a report, writes cad/strain_<name>.md

Load cases
  A  cam drive   : plan_for_cams (the hardware as designed), pen drag on
  B  fast motors : default planner (220 mm/s pen, 1.5 m/s^2), pen drag on
  C  handling    : 5 N static push on the pen tip, every direction, whole paper

Model (planar, all vectorised over time samples, units mm / N / N*mm)
  * rigid links with the Geometry masses (uniform rods), inertia from the planned
    motion (COM accelerations + I*alpha); couplers are massless two-force members
  * pen drag  F = -mu * N_pen * v_hat while the pen is down and moving
  * Newton-Euler, distal to proximal:
        hand  (about W)  -> coupler B tension
        bell-crank (about E, massless) -> coupler A tension
        forearm (about E) -> coupler E tension
        upper arm (about S) -> M1 torque; cranks -> M2, M3 torques
    (checked against the closed-form inverse dynamics: same motor torques)
  * each link = in-plane beam of its printed outline (width tapering between
    the end bosses, lightening slots, pin holes); stress = M/Z + N/A at 80
    sections along it.  Pin lugs: bearing + net-section tension (Kt = 3).
  * gears: Lewis bending (+ Kv) and Hertz contact; cam rollers: Hertz line contact.
  * anti-backlash torsion springs (springs.py): the gear/cam carry tau - S(theta);
    if that never changes sign the backlash/groove slop no longer moves the pen.
  * strain = stress / E (printed material, in-plane).
All dimensions, materials and loads come from params.toml.
"""
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

from arm_kinematics import Geometry, Linkage, inverse_dynamics, mass_params, paper_motor_range
from config import CFG, PAPER, thickness
import springs as spr

PC, PR, DR, LD = CFG["parts"], CFG["parts"]["radius"], CFG["drive"], CFG["loads"]
PIN_D, AXLE_D = PC["pin_bore"], PC["axle_bore"]
BACKLASH = PC["backlash"]                   # mm per gear (tooth thinning)
GROOVE_CLEAR = PC["groove_clearance"]       # mm roller <-> groove


@dataclass
class Material:
    name: str = CFG["material"]["name"]
    E: float = CFG["material"]["E"]               # MPa
    strength: float = CFG["material"]["strength"] # MPa
    nu: float = CFG["material"]["nu"]


MAT = Material()
E_STEEL, NU_STEEL = 200e3, 0.3


def cx(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# =================================================================== loads
def kinematic_states(g, motors):
    lk = Linkage(g)
    keys = ("S", "E", "W", "P", "C2", "T2", "A1", "A2", "B1", "B2")
    out = {k: np.zeros((len(motors), 2)) for k in keys}
    phi = np.zeros((len(motors), 3))
    margins = np.zeros((len(motors), 3))
    for i, m in enumerate(motors):
        st = lk.solve(*m)
        for k in keys:
            out[k][i] = st[k]
        phi[i] = st["phi1"], st["phi2"], st["phi3"]
        margins[i] = st["margins"]
    out["phi"], out["margins"] = phi, margins
    return out


def solve_loads(g, X, acc, alpha, F_pen):
    """X: kinematic arrays (N,2); acc: dict of COM accelerations (N,2) mm/s^2;
    alpha: (N,3) rad/s^2; F_pen: (N,2) N.  Returns a dict of loads."""
    m1, m2, m3 = g.m1, g.m2, g.m3
    I = np.array([g.m1 * g.L1 ** 2, g.m2 * g.L2 ** 2, g.m3 * g.L3 ** 2]) / 12   # kg mm^2
    S, E, W, P = X["S"], X["E"], X["W"], X["P"]
    c1, c2, c3 = (S + E) / 2, (E + W) / 2, (W + P) / 2
    Fi1, Fi2, Fi3 = -m1 * acc["c1"] / 1000, -m2 * acc["c2"] / 1000, -m3 * acc["c3"] / 1000   # N
    Mi = -I[None, :] * alpha * 1e-3                                                          # N mm

    uB = unit(X["B2"] - X["B1"])
    uA = unit(X["A2"] - X["A1"])
    ue = unit(X["T2"] - X["C2"])

    # hand about W: sum moments = 0, coupler B pulls B2 toward B1 with T_B
    m_ext = cx(P - W, F_pen) + cx(c3 - W, Fi3) + Mi[:, 2]
    T_B = m_ext / cx(X["B2"] - W, uB)
    R_W = -(F_pen + Fi3 - T_B[:, None] * uB)               # force ON hand FROM forearm at W

    # bell-crank about E (massless)
    T_A = cx(X["B1"] - E, T_B[:, None] * uB) / cx(X["A2"] - E, uA)
    R_Eb = -(T_B[:, None] * uB - T_A[:, None] * uA)         # on bell FROM elbow pin

    # forearm about E
    F_W_on_fa = -R_W
    m_fa = cx(W - E, F_W_on_fa) + cx(c2 - E, Fi2) + Mi[:, 1]
    T_e = m_fa / cx(X["T2"] - E, ue)
    R_Ef = -(F_W_on_fa + Fi2 - T_e[:, None] * ue)           # on forearm FROM elbow pin

    # upper arm about S (elbow pin fixed in the upper arm)
    F_E_on_ua = -R_Ef - R_Eb
    tau1 = -(cx(E - S, F_E_on_ua) + cx(c1 - S, Fi1) + Mi[:, 0])
    tau2 = -cx(X["C2"] - S, T_e[:, None] * ue)
    tau3 = -cx(X["A1"] - S, T_A[:, None] * uA)
    R_S = -(F_E_on_ua + Fi1)

    return dict(T_B=T_B, T_A=T_A, T_e=T_e, R_W=R_W, R_Eb=R_Eb, R_Ef=R_Ef, R_S=R_S,
                tau=np.stack([tau1, tau2, tau3], 1) / 1000,      # N m
                F_pen=F_pen, Fi=(Fi1, Fi2, Fi3), Mi=Mi, c=(c1, c2, c3),
                uA=uA, uB=uB, ue=ue, F_E_on_ua=F_E_on_ua, F_W_on_fa=F_W_on_fa)


# =================================================================== beams
def beam_stress(a, b, ra, rb, loads, couples=(), slot=None, holes=(), n=80, profile=None, t=5.0):
    """In-plane beam from a (root) to b.  loads: list of (point (N,2), force (N,2));
    couples: list of (point, M (N,)).  The bending moment at a section is taken
    from the loads lying beyond it (toward b).  slot = (r, inset_a, inset_b);
    holes = [(s_pos, d)] with s_pos = 0 (at a) or -1 (at b).
    profile = [(s_abs, radius), ...] overrides the linear ra->rb half-width.
    Returns dict: sigma (N,) peak |stress|, M (N,) peak |moment|,
    delta (N,) tip deflection perpendicular to a->b relative to a clamp at a,
    theta (N,) tip rotation, and the section arrays for chord corrections."""
    L = np.linalg.norm(b - a, axis=1)                          # (N,)
    u = (b - a) / L[:, None]
    s_frac = np.linspace(0.0, 1.0, n)
    s = L[:, None] * s_frac[None, :]                           # (N,n)
    if profile:
        ps, pr = zip(*profile)
        w = 2 * np.interp(s, ps, pr)
    else:
        w = 2 * (ra + (rb - ra) * s_frac)[None, :] * np.ones_like(s)
    ws = np.zeros_like(s)
    if slot:
        r, ia, ib = slot
        ws = np.where((s >= ia) & (s <= L[:, None] - ib), 2 * r, 0.0)
    for sp, d in holes:
        centre = np.zeros_like(L)[:, None] if sp == 0 else L[:, None]
        chord = 2 * np.sqrt(np.clip((d / 2) ** 2 - (s - centre) ** 2, 0, None))
        ws = np.maximum(ws, chord)
    Iz = t * (w ** 3 - ws ** 3) / 12
    Z = Iz / (w / 2)
    A = t * (w - ws)
    xs = a[:, None, :] + s[..., None] * u[:, None, :]          # (N,n,2)
    M = np.zeros_like(s)
    Nax = np.zeros_like(s)
    for p, F in loads:
        proj = ((p - a) * u).sum(1)                            # (N,)
        beyond = proj[:, None] > s
        M += np.where(beyond, cx(p[:, None, :] - xs, F[:, None, :]), 0.0)
        Nax += np.where(beyond, (F * u).sum(1)[:, None], 0.0)
    for p, C in couples:
        proj = ((p - a) * u).sum(1)
        M += np.where(proj[:, None] > s, C[:, None], 0.0)
    sig = np.abs(M) / Z + np.abs(Nax) / A
    k = np.argmax(sig, axis=1)
    curv = M / (MAT.E * Iz)                                    # 1/mm
    theta = np.trapezoid(curv, s, axis=1)
    delta = np.trapezoid(curv * (L[:, None] - s), s, axis=1)  # moment-area, clamp at a
    return dict(sigma=sig[np.arange(len(k)), k], M=np.abs(M).max(1), delta=delta, theta=theta,
                s=s, curv=curv, L=L)


def deflection_at(beam, s_target):
    """Perpendicular deflection at distance s_target (N,) from the clamp."""
    s, curv = beam["s"], beam["curv"]
    lever = np.clip(s_target[:, None] - s, 0, None)
    return np.trapezoid(curv * lever, s, axis=1)


def lug(F, w, d=PIN_D, t=5.0, Kt=3.0):
    """Pin joint in a printed plate: bearing and net-section tension stresses."""
    Fm = np.linalg.norm(F, axis=-1) if F.ndim > 1 else np.abs(F)
    return Fm / (d * t), Kt * Fm / ((w - d) * t)


def member_table(g, X, L):
    """Peak stress per member and per pin joint for one load case."""
    S, E, W, P = X["S"], X["E"], X["W"], X["P"]
    (c1, c2, c3), (Fi1, Fi2, Fi3), Mi = L["c"], L["Fi"], L["Mi"]
    rows = {}

    defl = {}

    def add(name, bm, key=None):
        rows[name] = dict(sigma=float(np.max(bm["sigma"])), M=float(np.max(bm["M"])), F=None)
        if key:
            defl[key] = bm

    ua_r, (ua_sw, ua_si) = PR["upper_arm"], PR["upper_arm_slot"]
    bm = beam_stress(S, E, *ua_r, [(E, L["F_E_on_ua"]), (c1, Fi1)], [(c1, Mi[:, 0])],
                     slot=(ua_sw, ua_si, ua_si), holes=[(0, AXLE_D), (-1, PIN_D)], t=thickness("upper_arm"))
    add("upper arm (S-E)", bm, "ua")
    rT, rE, rW = PR["forearm"]
    fsw, fsi = PR["forearm_slot"]
    bm = beam_stress(X["T2"], W, rT, rW,
                     [(E, L["R_Ef"]), (W, L["F_W_on_fa"]), (c2, Fi2)], [(c2, Mi[:, 1])],
                     slot=(fsw, g.r_e + fsi, fsi), holes=[(0, PIN_D), (-1, PIN_D)],
                     profile=[(0, rT), (g.r_e, rE), (g.r_e + g.L2, rW)], t=thickness("forearm"))
    add("forearm (T2-E-W)", bm, "fa")
    bm = beam_stress(W, P, *PR["hand"], [(P, L["F_pen"]), (c3, Fi3)], [(c3, Mi[:, 2])],
                     holes=[(0, PIN_D)], t=thickness("hand"))
    add("hand (W-pen)", bm, "hand")
    bm = beam_stress(W, X["B2"], *PR["hand_crank"], [(X["B2"], -L["T_B"][:, None] * L["uB"])],
                     holes=[(0, PIN_D), (-1, PIN_D)], t=thickness("hand"))
    add("hand crank (W-B2)", bm, "hcrank")
    bm = beam_stress(S, X["C2"], *PR["crank_M2"], [(X["C2"], L["T_e"][:, None] * L["ue"])],
                     holes=[(0, AXLE_D), (-1, PIN_D)], t=thickness("crank_M2"))
    add("M2 elbow crank (S-C2)", bm, "c2")
    bm = beam_stress(S, X["A1"], *PR["crank_M3"], [(X["A1"], L["T_A"][:, None] * L["uA"])],
                     holes=[(0, AXLE_D), (-1, PIN_D)], t=thickness("crank_M3"))
    add("M3 wrist crank (S-A1)", bm, "c3")
    bm = beam_stress(E, X["B1"], *PR["bell"], [(X["B1"], L["T_B"][:, None] * L["uB"])],
                     holes=[(0, PIN_D), (-1, PIN_D)], t=thickness("bell_B"))
    add("bell-crank arm B", bm, "bellB")
    bm = beam_stress(E, X["A2"], *PR["bell"], [(X["A2"], -L["T_A"][:, None] * L["uA"])],
                     holes=[(0, PIN_D), (-1, PIN_D)], t=thickness("bell_A"))
    add("bell-crank arm A", bm, "bellA")

    # couplers: axial (+ Euler buckling, weak axis = plate thickness)
    for name, key, Tn, Lc in (("coupler E (C2-T2)", "coupler_E", L["T_e"], g.L1),
                              ("coupler A (A1-A2)", "coupler_A", L["T_A"], g.L1),
                              ("coupler B (B1-B2)", "coupler_B", L["T_B"], g.L2)):
        w, t = 2 * PR[key], thickness(key)
        sig = np.abs(Tn) / ((w - PIN_D) * t)
        Pcr = math.pi ** 2 * MAT.E * (w * t ** 3 / 12) / Lc ** 2
        rows[name] = dict(sigma=float(sig.max()), M=None, F=float(np.abs(Tn).max()),
                          buckle_SF=float(Pcr / max(1e-9, np.max(-Tn))) if np.max(-Tn) > 0 else float("inf"),
                          Pcr=Pcr)

    # pin joints (worst lug at each joint)
    joints = {
        "S hub (axle 8 mm)": (L["R_S"], 2 * PR["upper_arm"][0], AXLE_D, thickness("upper_arm")),
        "elbow E (upper arm lug)": (L["F_E_on_ua"], 2 * PR["upper_arm"][1], PIN_D, thickness("upper_arm")),
        "elbow E (forearm lug)": (L["R_Ef"], 2 * PR["forearm"][1], PIN_D, thickness("forearm")),
        "wrist W": (L["R_W"], 2 * PR["forearm"][2], PIN_D, thickness("forearm")),
        "T2 (coupler E)": (L["T_e"], 2 * PR["coupler_E"], PIN_D, thickness("coupler_E")),
        "A1/A2 (coupler A)": (L["T_A"], 2 * PR["coupler_A"], PIN_D, thickness("coupler_A")),
        "B1/B2 (coupler B)": (L["T_B"], 2 * PR["coupler_B"], PIN_D, thickness("coupler_B")),
    }
    jrows = {}
    for name, (F, w, d, t) in joints.items():
        sb, st = lug(F, w, d, t)
        jrows[name] = dict(F=float(np.max(np.linalg.norm(F, axis=-1) if F.ndim > 1 else np.abs(F))),
                           bearing=float(sb.max()), net=float(st.max()))
    return rows, jrows, defl


# ================================================================ gears & cams
LEWIS = [(12, .245), (15, .290), (17, .303), (20, .322), (24, .337), (30, .359), (40, .389),
         (50, .408), (75, .435), (100, .447), (150, .460)]


def lewis_Y(z):
    zz, yy = zip(*LEWIS)
    return float(np.interp(z, zz, yy))


GEAR_LAYER = {"M1": "upper_arm", "M2": "crank_M2", "M3": "gear_M3"}


def gear_torque(tau_req, motors, springs):
    """Torque the gear mesh must deliver to each hub (N m): required minus spring."""
    out = np.array(tau_req, float).copy()
    if springs:
        for k, nm in enumerate(("M1", "M2", "M3")):
            out[:, k] -= spr.hub_spring_torque(springs[nm], motors[:, k])
    return out


def gear_table(tau_gear, cs):
    """tau_gear (N,3) N m -> tooth root bending (Lewis) and contact (Hertz) stresses."""
    p = cs.params
    m, z_hub, Kv = p.module, p.z_hub, LD["gear_Kv"]
    rows = {}
    pa = math.radians(20)
    Es = 1 / (2 * (1 - MAT.nu ** 2) / MAT.E)                 # same printed material both gears
    for k, name in enumerate(("M1", "M2", "M3")):
        des = cs.axes[k]
        zf = des["zf"] if des else p.z_follow[0]
        b = thickness(GEAR_LAYER[name])                       # face width = layer thickness
        r_h, r_f = m * z_hub / 2, m * zf / 2
        Ft = np.abs(tau_gear[:, k]) * 1000 / r_h              # N
        sb_h = Kv * Ft / (b * m * lewis_Y(z_hub))
        sb_f = Kv * Ft / (b * m * lewis_Y(zf))
        rho = 1 / (r_h * math.sin(pa)) + 1 / (r_f * math.sin(pa))
        sH = np.sqrt(Kv * Ft / math.cos(pa) * Es * rho / (math.pi * b))
        sgn = np.sign(tau_gear[:, k])
        rows[f"{name} hub z{z_hub} / follower z{zf} (m{m:g}, b{b:g})"] = dict(
            Ft=float(Ft.max()), root_hub=float(sb_h.max()), root_fol=float(sb_f.max()),
            contact=float(sH.max()), reverses=bool(sgn.min() < 0 < sgn.max()),
            min_load=float(np.abs(tau_gear[:, k]).min()))
    return rows


def cam_table(tau_gear, t, cs):
    """Roller force = follower torque / (arm * cos(pressure angle)); Hertz on the groove wall."""
    rows = {}
    if cs is None:
        return rows
    p = cs.params
    th = 2 * np.pi * t / cs.period
    Es = 1 / ((1 - MAT.nu ** 2) / MAT.E + (1 - NU_STEEL ** 2) / E_STEEL)
    for k, name in enumerate(("M1", "M2", "M3")):
        des = cs.axes[k]
        if des is None:
            continue
        G = des["G"]
        alpha = np.interp(th, cs.theta, des["alpha"], period=2 * np.pi)
        rho = np.interp(th, cs.theta, np.abs(des["rho"]), period=2 * np.pi)
        tf = np.abs(tau_gear[:, k]) * G                       # N m at the follower
        Fr = tf * 1000 / (des["l"] * np.cos(alpha))           # N
        rho_wall = np.maximum(rho - p.roller_r, 0.5)
        R = 1 / (1 / p.roller_r + 1 / rho_wall)
        pH = np.sqrt(Fr * Es / (math.pi * p.roller_w * R))
        w_arm, t_f = 2 * PR["follower_arm"][0], thickness(f"follow_{name}")
        rows[f"cam {name}"] = dict(F=float(Fr.max()), contact=float(pH.max()),
                                   follower_root=float((tf * 1000 / (t_f * w_arm ** 2 / 6)).max()))
    return rows


def camshaft_torque(tau_gear, omega_hub, period):
    """Power balance: camshaft torque = sum(tau_gear_i * w_i) / w_cam (ideal, no friction)."""
    w_cam = 2 * np.pi / period
    return (tau_gear * omega_hub).sum(1) / w_cam


# ============================================================ pen-tip errors
def tip_error(g, X, L, cs, defl, tau_gear=None):
    """Pen-tip displacement (upper bound per sample, then max) from
    (a) coupler stretch, (b) bending of every link and crank (moment-area on
    the real moment diagrams), (c) gear backlash + groove clearance."""
    S, E, W, P = X["S"], X["E"], X["W"], X["P"]
    m = np.maximum(X["margins"], 0.05)
    A_c = lambda key: (2 * PR[key] - PIN_D) * thickness(key)
    d_e = np.abs(L["T_e"]) * g.L1 / (MAT.E * A_c("coupler_E"))
    d_A = np.abs(L["T_A"]) * g.L1 / (MAT.E * A_c("coupler_A"))
    d_B = np.abs(L["T_B"]) * g.L2 / (MAT.E * A_c("coupler_B"))
    r_PE = np.linalg.norm(P - E, axis=1)
    stretch = d_e / (g.r_e * m[:, 0]) * r_PE + (d_A / (g.r_a * m[:, 1]) + d_B / (g.r_b * m[:, 2])) * g.L3

    ua = np.abs(defl["ua"]["delta"])                              # elbow moves -> pen moves
    fa = defl["fa"]                                               # rooted at T2; chord through T2 and E
    L_TE = np.linalg.norm(E - X["T2"], axis=1)
    fa_W = np.abs(fa["delta"] - deflection_at(fa, L_TE) * fa["L"] / L_TE)
    hand = np.abs(defl["hand"]["delta"])
    rot_hand = (np.abs(defl["hcrank"]["delta"]) / g.r_b + np.abs(defl["c3"]["delta"]) / g.r_a
                + np.abs(defl["bellA"]["delta"]) / g.r_a + np.abs(defl["bellB"]["delta"]) / g.r_b)
    rot_fore = np.abs(defl["c2"]["delta"]) / g.r_e
    bend_parts = dict(upper_arm=ua.max(), forearm=fa_W.max(), hand=hand.max(),
                      cranks=(rot_hand * g.L3 + rot_fore * r_PE).max())
    bend = (ua + fa_W + hand + rot_hand * g.L3 + rot_fore * r_PE).max()

    slop = []
    r_hub = cs.params.module * cs.params.z_hub / 2
    for k in range(3):
        G = cs.axes[k]["G"] if cs.axes[k] else 3.0
        l = cs.axes[k]["l"] if cs.axes[k] else 55.0
        s_k = 2 * BACKLASH / r_hub + G * GROOVE_CLEAR / l         # rad at the hub
        if tau_gear is not None:                                   # preloaded and never reversing
            sg = np.sign(tau_gear[:, k])
            if sg.min() == sg.max() and sg[0] != 0:
                s_k = 0.0
        slop.append(s_k)
    lever = [np.linalg.norm(P - S, axis=1).max(), r_PE.max(), g.L3]
    backlash = sum(s_ * l_ for s_, l_ in zip(slop, lever))
    return dict(stretch=float(stretch.max()), bend=float(bend), bend_parts=bend_parts,
                backlash=float(backlash), slop_deg=[math.degrees(s_) for s_ in slop])


# ================================================================= cases
def smooth(a, sigma=2.0):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(a, sigma, axis=0, mode="nearest")


def dynamic_case(g, plan, mu=LD["pen_mu"], N_pen=LD["pen_normal"]):
    X = kinematic_states(g, plan.motors)
    dt = plan.t[1] - plan.t[0]
    acc = {}
    for i, (a, b) in enumerate((("S", "E"), ("E", "W"), ("W", "P"))):
        c = (X[a] + X[b]) / 2
        acc[f"c{i + 1}"] = smooth(np.gradient(np.gradient(c, dt, axis=0), dt, axis=0))
    v = np.gradient(X["P"], dt, axis=0)
    sp = np.linalg.norm(v, axis=1, keepdims=True)
    F_pen = np.where((plan.pen[:, None]) & (sp > 1.0), -mu * N_pen * v / np.maximum(sp, 1e-9), 0.0)
    L = solve_loads(g, X, acc, plan.alpha, F_pen)
    return X, L


def static_case(g, F=LD["push"], ndir=16):
    from arm_kinematics import solve_ik
    M = []
    for x in np.linspace(PAPER[0], PAPER[2], 12):
        for y in np.linspace(PAPER[1], PAPER[3], 10):
            s = solve_ik(g, (x, y))
            if s:
                M.append(s["inputs"])
    M = np.repeat(np.array(M), ndir, axis=0)
    ang = np.tile(np.linspace(0, 2 * np.pi, ndir, endpoint=False), len(M) // ndir)
    X = kinematic_states(g, M)
    zero = np.zeros((len(M), 2))
    F_pen = F * np.stack([np.cos(ang), np.sin(ang)], 1)
    L = solve_loads(g, X, {"c1": zero, "c2": zero, "c3": zero}, np.zeros((len(M), 3)), F_pen)
    return X, L


def validate(g, plan):
    """Statics chain (no pen) vs closed-form inverse dynamics (no motor inertia)."""
    X = kinematic_states(g, plan.motors[:400])
    dt = plan.t[1] - plan.t[0]
    acc = {}
    for i, (a, b) in enumerate((("S", "E"), ("E", "W"), ("W", "P"))):
        c = (X[a] + X[b]) / 2
        acc[f"c{i + 1}"] = np.gradient(np.gradient(c, dt, axis=0), dt, axis=0)
    q = np.unwrap(X["phi"], axis=0)
    qd = np.gradient(q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    L = solve_loads(g, X, acc, qdd, np.zeros((len(q), 2)))
    g0 = Geometry(**{**g.__dict__, "J_motor": (0, 0, 0)})
    tau_ref = inverse_dynamics(mass_params(g0), q, qd, qdd)
    sl = slice(5, -5)
    return float(np.abs(L["tau"][sl] - tau_ref[sl]).max()), float(np.abs(tau_ref[sl]).max())


# ================================================================== report
def fmt_rows(rows):
    out = ["| member | peak axial F (N) | peak moment (N·mm) | peak stress (MPa) | strain (µε) | SF |",
           "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        sig = r["sigma"]
        sf = MAT.strength / max(sig, 1e-9)
        extra = f" (buckling SF {r['buckle_SF']:.0f})" if "buckle_SF" in r and np.isfinite(r["buckle_SF"]) else ""
        out.append(f"| {k} | {'' if r['F'] is None else f'{r['F']:.2f}'} | "
                   f"{'' if r['M'] is None else f'{r['M']:.0f}'} | {sig:.3f} | {1e6 * sig / MAT.E:.0f} | "
                   f"{sf:.0f}{extra} |")
    return "\n".join(out)


def fmt_joints(j):
    out = ["| joint | peak force (N) | bearing (MPa) | net tension ×Kt3 (MPa) | SF |", "|---|---|---|---|---|"]
    for k, r in j.items():
        out.append(f"| {k} | {r['F']:.2f} | {r['bearing']:.3f} | {r['net']:.3f} | "
                   f"{MAT.strength / max(r['net'], r['bearing'], 1e-9):.0f} |")
    return "\n".join(out)


def run(src="heart"):
    from arm_linkage import builtin_shape
    from cam_drive import design_cams, plan_for_cams
    from path_planner import plan_job
    from vector_input import fit_to_box, load_svg

    g = Geometry()
    strokes = fit_to_box(load_svg(src) if src.endswith(".svg") else builtin_shape(src), PAPER)
    th = PC["layer_thickness"]
    md = [f"# Structural check – `{src}`", "",
          f"Parameters: `params.toml`. Material: {MAT.name}, E = {MAT.E:.0f} MPa, strength "
          f"{MAT.strength:.0f} MPa. Layers {th} mm. Gears module {DR['module']}, z{DR['z_hub']}/"
          f"{DR['z_follow'][0]}. Rollers {DR['roller']['name']}. Masses {g.m1}/{g.m2}/{g.m3} kg. "
          f"Pen drag {LD['pen_mu'] * LD['pen_normal']:.2f} N.", ""]

    plan_cam, dp = plan_for_cams(g, strokes)
    cs = design_cams(plan_cam, g, dp)
    springs = spr.design(paper_motor_range(g))
    err, ref = validate(g, plan_cam)
    md += [f"Validation: statics chain vs inverse dynamics, max motor-torque difference "
           f"{err:.1e} N·m (peak {ref:.1e} N·m).", "", spr.report(springs).split("\n", 2)[2], ""]

    plan_fast = plan_job(g, strokes, (380, 0))
    cases = [("A – cam drive (as designed, spring-preloaded)", plan_cam, cs, springs),
             ("B – fast motors (no cams, no springs; reference)", plan_fast, None, None)]
    for title, plan, cset, sp in cases:
        X, L = dynamic_case(g, plan)
        rows, joints, defl = member_table(g, X, L)
        tau_req = L["tau"]
        motors = np.unwrap(X["phi"], axis=0) + np.array([0.0, math.pi, g.off_a])
        tau_gear = gear_torque(tau_req, motors, sp)
        md += [f"## Case {title}", f"duration {plan.t[-1]:.1f} s, peak torque the arm needs "
               f"{np.abs(tau_req).max(0).round(3).tolist()} N·m", "", fmt_rows(rows), "", fmt_joints(joints), ""]
        if cset is not None:
            gears = gear_table(tau_gear, cs)
            md += ["| gear mesh | Ft (N) | root σ hub (MPa) | root σ follower (MPa) | contact (MPa) "
                   "| hub-root strain (µε) | SF | load reverses? |", "|---|---|---|---|---|---|---|---|"]
            for k, r in gears.items():
                md.append(f"| {k} | {r['Ft']:.1f} | {r['root_hub']:.2f} | {r['root_fol']:.2f} | {r['contact']:.1f} | "
                          f"{1e6 * r['root_hub'] / MAT.E:.0f} | {MAT.strength / r['root_hub']:.1f} | "
                          f"{'YES' if r['reverses'] else 'no (min ' + format(r['min_load'], '.3f') + ' N·m)'} |")
            cams = cam_table(tau_gear, plan.t, cset)
            omega = np.gradient(np.unwrap(X["phi"], axis=0), plan.t[1] - plan.t[0], axis=0)
            Tcs = camshaft_torque(tau_gear, omega, cs.period)
            md += ["", "| cam | roller force (N) | Hertz on groove (MPa) | SF | follower root σ (MPa) |",
                   "|---|---|---|---|---|"]
            for k, r in cams.items():
                md.append(f"| {k} | {r['F']:.1f} | {r['contact']:.1f} | {MAT.strength / r['contact']:.1f} | "
                          f"{r['follower_root']:.2f} |")
            md += ["", f"Camshaft torque (ideal, springs included): peak {np.abs(Tcs).max():.3f} N·m, "
                   f"RMS {np.sqrt((Tcs ** 2).mean()):.3f} N·m at {60 / cs.period:.1f} rpm. With a ~40 %-efficient "
                   f"worm the motor must supply ≈ {np.abs(Tcs).max() / 0.4:.3f} N·m ÷ worm ratio."]
            tip = tip_error(g, X, L, cs, defl, tau_gear)
            bp = tip["bend_parts"]
            md += ["", f"Pen-tip error: coupler stretch {1000 * tip['stretch']:.0f} µm, link/crank bending "
                   f"{1000 * tip['bend']:.0f} µm (upper arm {1000 * bp['upper_arm']:.0f}, forearm "
                   f"{1000 * bp['forearm']:.0f}, hand {1000 * bp['hand']:.0f}, cranks {1000 * bp['cranks']:.0f}), "
                   f"backlash + groove slop {tip['backlash']:.2f} mm "
                   f"(hub slop {', '.join(f'{x:.2f}°' for x in tip['slop_deg'])}).", ""]

    X, L = static_case(g)
    rows, joints, defl = member_table(g, X, L)
    M_static = np.unwrap(X["phi"], axis=0) + np.array([0.0, math.pi, g.off_a])
    tau_gear = gear_torque(L["tau"], M_static, springs)
    gears = gear_table(tau_gear, cs)
    tip = tip_error(g, X, L, cs, defl, tau_gear)
    md += [f"## Case C – {LD['push']:g} N push on the pen tip (any direction, whole paper, springs on)", "",
           f"peak torque at the hubs {np.abs(L['tau']).max(0).round(3).tolist()} N·m", "",
           fmt_rows(rows), "", fmt_joints(joints), "",
           "| gear mesh | Ft (N) | root σ hub (MPa) | SF | contact (MPa) |", "|---|---|---|---|---|"]
    for k, r in gears.items():
        md.append(f"| {k} | {r['Ft']:.1f} | {r['root_hub']:.1f} | {MAT.strength / r['root_hub']:.1f} | {r['contact']:.0f} |")
    bp = tip["bend_parts"]
    md += ["", f"Pen-tip deflection under the push: coupler stretch {1000 * tip['stretch']:.0f} µm, "
           f"link/crank bending {1000 * tip['bend']:.0f} µm (upper arm {1000 * bp['upper_arm']:.0f}, "
           f"forearm {1000 * bp['forearm']:.0f}, hand {1000 * bp['hand']:.0f}, cranks {1000 * bp['cranks']:.0f}). "
           "A push larger than the spring preload reverses the gear load: the arm then moves through the "
           "backlash and springs back when released.", ""]
    text = "\n".join(md)
    os.makedirs("cad", exist_ok=True)
    name = os.path.splitext(os.path.basename(src))[0]
    with open(f"cad/strain_{name}.md", "w", encoding="utf-8") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(run(sys.argv[1] if len(sys.argv) > 1 else "heart"))
