"""
arm_kinematics.py - geometry, range of motion, FK/IK and rigid-body dynamics
for the planar 3-joint drawing arm (all inputs coaxial at the shoulder).

Frame (top view): shoulder S at the origin, body behind it (x < 0), "forward"
is +x, the body midline is on the +y side (this is a RIGHT arm).
Units: mm for geometry, kg for mass, rad for angles, s for time.

Mechanism recap (see README):
    M1 = phi1            upper-arm absolute angle (direct drive)
    M2 = phi2 + pi       forearm absolute angle   (parallelogram across upper arm)
    M3 = phi3 + off_a    hand absolute angle      (2 parallelograms + elbow bell-crank)
Because every motor sets an ABSOLUTE link angle, the absolute angles are also
the generalised coordinates of the dynamics, and the motor torques are exactly
the generalised forces for them (virtual work through ideal parallelograms).
"""
import math
from dataclasses import dataclass, field, replace

import numpy as np

TAU = 2 * math.pi
D = math.radians


# ------------------------------------------------------------ range of motion
@dataclass(frozen=True)
class ROM:
    """Joint limits (rad).  shoulder = absolute phi1, elbow = phi2-phi1,
    wrist = phi3-phi2 (all CCW-positive; for a right arm, CCW = toward the
    body midline = flexion / horizontal adduction)."""
    name: str
    shoulder: tuple
    elbow: tuple
    wrist: tuple


# Typical adult active ROM (clinical goniometry norms, e.g. AAOS):
#   shoulder horizontal adduction ~130 deg, horizontal abduction ~45 deg
#     (measured from the arm pointing straight out sideways = phi1 -90 deg)
#   elbow flexion 0-145 deg (0 is a parallelogram dead point -> 20 deg min)
#   wrist flexion 80 / extension 70, radial dev 20 / ulnar dev 30
ROM_PROFILES = {
    "human: palm sideways (wrist flex/ext)": ROM(
        "human: palm sideways (wrist flex/ext)",
        shoulder=(D(-90 - 45), D(-90 + 130)), elbow=(D(20), D(145)), wrist=(D(-70), D(80))),
    "human: palm down (radial/ulnar dev)": ROM(
        "human: palm down (radial/ulnar dev)",
        shoulder=(D(-90 - 45), D(-90 + 130)), elbow=(D(20), D(145)), wrist=(D(-30), D(20))),
    "mechanical only": ROM(
        "mechanical only",
        shoulder=(D(-180), D(180)), elbow=(D(20), D(160)), wrist=(D(-100), D(100))),
}
DEFAULT_ROM = "human: palm sideways (wrist flex/ext)"


# -------------------------------------------------------------------- geometry
@dataclass
class Geometry:
    L1: float = 300.0          # upper arm   (shoulder -> elbow)   ~ human humerus
    L2: float = 250.0          # forearm     (elbow -> wrist)      ~ human ulna
    L3: float = 70.0           # hand        (wrist -> pen tip)
    r_e: float = 60.0          # elbow drive crank at S  = forearm tail at E
    r_a: float = 45.0          # wrist crank at S        = bell-crank arm A at E
    r_b: float = 40.0          # bell-crank arm B at E   = hand crank at W
    off_a: float = 0.0         # crank A angle relative to hand direction
    off_b: float = D(90)       # crank B angle relative to hand direction (L bell-crank)
    margin_min: float = 0.30   # min |sin| transmission margin allowed in IK
    rom: ROM = field(default_factory=lambda: ROM_PROFILES[DEFAULT_ROM])
    # mass model (uniform rods + pen/hand lump) - used for energy & torque
    m1: float = 0.25           # upper arm + couplers riding on it
    m2: float = 0.20           # forearm + couplers
    m3: float = 0.08           # hand + pen holder + pen
    # rotor + gearbox inertia reflected to each output shaft (kg m^2).
    # Each motor turns an ABSOLUTE angle, so this just adds to the diagonal.
    J_motor: tuple = (2e-4, 2e-4, 2e-4)

    @property
    def lengths_m(self):
        return np.array([self.L1, self.L2, self.L3]) / 1000.0


# --------------------------------------------------------------- 2D helpers
def polar(p, r, a):
    return (p[0] + r * math.cos(a), p[1] + r * math.sin(a))


def ang(p, q):
    return math.atan2(q[1] - p[1], q[0] - p[0])


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def dist(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])


def circle_x(c0, r0, c1, r1):
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


# ------------------------------------------------------- margins & comfort
def margins(g, phi1, phi2, phi3, xp=math):
    """|sin| transmission margins: elbow pgram, wrist stage A, wrist stage B."""
    return (abs(xp.sin(phi2 - phi1)),
            abs(xp.sin(phi3 + g.off_a - phi1)),
            abs(xp.sin(phi3 + g.off_b - phi2)))


def discomfort(g, phi1, phi2, phi3):
    """0 at mid-range, ~1 at a joint limit (quartic), summed over joints.
    Humans avoid end-of-range postures; this term models that."""
    r = g.rom
    tot = 0.0
    for q, (lo, hi) in ((phi1, r.shoulder), (wrap_np(phi2 - phi1), r.elbow), (wrap_np(phi3 - phi2), r.wrist)):
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        tot = tot + ((q - mid) / half) ** 4
    return tot


def wrap_np(a):
    return (a + np.pi) % TAU - np.pi


# ------------------------------------------------------------ forward kin.
class Linkage:
    """Closed-loop FK by circle intersection.  track=True follows the previous
    branch (like real hardware, can flip at a dead point)."""

    def __init__(self, g):
        self.g = g
        self.prev = None

    def solve(self, m1, m2, m3, track=False):
        g = self.g
        pv = self.prev if track else None
        S = (0.0, 0.0)
        E = polar(S, g.L1, m1)
        C2 = polar(S, g.r_e, m2)
        sols = circle_x(C2, g.L1, E, g.r_e)
        if not sols:
            return None
        T2 = nearest(sols, pv["T2"] if pv else polar(E, g.r_e, m2))
        phi2 = ang(E, T2) - math.pi
        W = polar(E, g.L2, phi2)

        A1 = polar(S, g.r_a, m3)
        sols = circle_x(A1, g.L1, E, g.r_a)
        if not sols:
            return None
        A2 = nearest(sols, pv["A2"] if pv else polar(E, g.r_a, m3))
        bB = ang(E, A2) - g.off_a + g.off_b
        B1 = polar(E, g.r_b, bB)
        sols = circle_x(B1, g.L2, W, g.r_b)
        if not sols:
            return None
        B2 = nearest(sols, pv["B2"] if pv else polar(W, g.r_b, bB))
        phi3 = ang(W, B2) - g.off_b
        P = polar(W, g.L3, phi3)

        st = dict(S=S, E=E, W=W, P=P, C2=C2, T2=T2, A1=A1, A2=A2, B1=B1, B2=B2,
                  phi1=m1, phi2=phi2, phi3=phi3, inputs=(m1, m2, m3),
                  margins=margins(g, m1, phi2, phi3))
        st["flipped"] = (abs(wrap(phi2 + math.pi - m2)) > 1e-3 or
                         abs(wrap(phi3 + g.off_a - m3)) > 1e-3)
        self.prev = st
        return st


def inputs_from_angles(g, phi1, phi2, phi3):
    return phi1, phi2 + math.pi, phi3 + g.off_a


# ------------------------------------------------------------ inverse kin.
def ik_grid(g, Px, Py, phi3):
    """Vectorised IK.  Px, Py broadcast against phi3 (hand angle candidates).
    Returns phi1, phi2, feasible-mask, state cost (discomfort + margin penalty)."""
    Wx = Px - g.L3 * np.cos(phi3)
    Wy = Py - g.L3 * np.sin(phi3)
    c = (Wx ** 2 + Wy ** 2 - g.L1 ** 2 - g.L2 ** 2) / (2 * g.L1 * g.L2)
    ok = np.abs(c) <= 1
    e = np.arccos(np.clip(c, -1, 1))                 # elbow always bends CCW (right arm)
    phi1 = np.arctan2(Wy, Wx) - np.arctan2(g.L2 * np.sin(e), g.L1 + g.L2 * np.cos(e))
    phi1 = wrap_np(phi1)
    phi2 = phi1 + e
    r = g.rom
    w = wrap_np(phi3 - phi2)
    ok &= (e >= r.elbow[0]) & (e <= r.elbow[1])
    ok &= (phi1 >= r.shoulder[0]) & (phi1 <= r.shoulder[1])
    ok &= (w >= r.wrist[0]) & (w <= r.wrist[1])
    m = np.minimum(np.minimum(np.abs(np.sin(e)), np.abs(np.sin(phi3 + g.off_a - phi1))),
                   np.abs(np.sin(phi3 + g.off_b - phi2)))
    ok &= m >= g.margin_min
    cost = discomfort(g, phi1, phi2, phi3) + 0.5 * (1 - m) ** 2
    return phi1, phi2, ok, cost


HAND_GRID = np.radians(np.arange(-180, 180, 2.0))     # 2 deg resolution


def solve_ik(g, P, prev_phi3=None, fixed_phi3=None, smooth=2.0):
    """Single-point IK for interactive use: picks the hand angle that minimises
    discomfort + margin penalty (+ a penalty for jumping from prev_phi3)."""
    if fixed_phi3 is not None:
        cands = np.array([fixed_phi3])
    elif prev_phi3 is None:
        cands = HAND_GRID
    else:
        cands = prev_phi3 + np.radians(np.arange(-20, 20.01, 0.5))
    phi1, phi2, ok, cost = ik_grid(g, P[0], P[1], cands)
    if prev_phi3 is not None and fixed_phi3 is None:
        cost = cost + smooth * np.abs(wrap_np(cands - prev_phi3))
    if not ok.any():
        if prev_phi3 is not None and fixed_phi3 is None:
            return solve_ik(g, P, None, None, smooth)
        return None
    i = int(np.argmin(np.where(ok, cost, np.inf)))
    p1, p2, p3 = float(phi1[i]), float(phi2[i]), float(cands[i])
    return dict(phi1=p1, phi2=p2, phi3=p3, margins=margins(g, p1, p2, p3),
                inputs=inputs_from_angles(g, p1, p2, p3))


def reachable_grid(g, x0, x1, y0, y1, step=10.0):
    """Boolean grid of pen positions reachable under the ROM profile."""
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    _, _, ok, _ = ik_grid(g, X[..., None], Y[..., None], np.radians(np.arange(-180, 180, 4.0)))
    return xs, ys, ok.any(axis=-1)


# ------------------------------------------------------------------ dynamics
def mass_params(g):
    """H_jk coefficients for a planar chain in ABSOLUTE angles:
       KE = 1/2 sum_jk H_jk cos(q_j - q_k) qd_j qd_k
    with uniform-rod links (COM at mid-length, I = m l^2 / 12)."""
    l = g.lengths_m
    m = np.array([g.m1, g.m2, g.m3])
    c = l / 2
    I = m * l ** 2 / 12
    a = np.zeros((3, 3))                     # a[i, j]: lever of q_j on COM of link i
    for i in range(3):
        a[i, :i] = l[:i]
        a[i, i] = c[i]
    H = np.einsum("i,ij,ik->jk", m, a, a) + np.diag(I) + np.diag(g.J_motor)
    return H


def mass_matrix(H, q):
    """q: (..., 3) absolute angles -> M (..., 3, 3)."""
    dq = q[..., :, None] - q[..., None, :]
    return H * np.cos(dq)


def inverse_dynamics(H, q, qd, qdd):
    """Motor torques (N m) for absolute-angle coordinates (horizontal plane,
    no gravity):  tau_j = sum_k H_jk [cos(q_j-q_k) qdd_k + sin(q_j-q_k) qd_k^2]."""
    dq = q[..., :, None] - q[..., None, :]
    return (np.einsum("...jk,...k->...j", H * np.cos(dq), qdd) +
            np.einsum("...jk,...k->...j", H * np.sin(dq), qd ** 2))


def kinetic_energy(H, q, qd):
    M = mass_matrix(H, q)
    return 0.5 * np.einsum("...j,...jk,...k->...", qd, M, qd)
