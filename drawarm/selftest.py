"""Numerical self-checks: IK/FK round trip, decoupling, ROM reach, dynamics
power balance, and min-energy planning vs greedy IK."""
import math
import time

import numpy as np

from .config import PAPER
from .inputs import fit_to_box
from .kinematics import (D, ROM_PROFILES, Geometry, Linkage, dist, inverse_dynamics,
                         kinetic_energy, mass_params, reachable_grid, solve_ik, wrap)
from .planner import plan_job
from .shapes import builtin_shape


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
