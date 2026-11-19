"""
springs.py - anti-backlash torsion springs on the shoulder axle.

Each shoulder hub (M2 crank, M1 upper arm, M3 gear) gets a torsion spring that
sits in the spacer gap below it, around the FIXED 8 mm shoulder axle:
    one leg in a keyed collar (D-flat on the axle, cannot turn)
    other leg in a hole in the hub.
The spring applies a hub torque S(theta) = sign * P - k * (theta - theta_mid)
that is larger than the biggest torque the arm ever needs, so the gear mesh and
the cam roller are always loaded on the same flank: backlash and groove
clearance no longer move the pen.

Sizing (music wire, close-wound helical torsion spring):
    k  = E d^4 / (64 D n)          N mm / rad
    stress = Ki * 32 M / (pi d^3), Ki = (4C^2 - C - 1) / (4C(C - 1)), C = D/d
    allowed = 0.75 * UTS,  UTS = 2211 / d^0.145 MPa (ASTM A228)
Pick the smallest wire whose rate keeps the torque within +-variation over
the hub's travel (whole paper) and whose stress is allowed.
"""
import math

from .config import CFG, LAYER

E_WIRE = 207e3                      # MPa
WIRES = (0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0)


def gap_lengths():
    """Axial room (mm) for the spring under each hub: M2 (layer 1), M1 (3), M3 gear (6)."""
    th, gap = CFG["parts"]["layer_thickness"], CFG["parts"]["gap"]
    z0 = lambda L: sum(th[i] + gap for i in range(L))
    return {"M2": z0(LAYER["crank_M2"]) + gap,
            "M1": z0(LAYER["upper_arm"]) - (z0(LAYER["crank_M2"]) + th[LAYER["crank_M2"]]),
            "M3": z0(LAYER["gear_M3"]) - (z0(LAYER["upper_arm"]) + th[LAYER["upper_arm"]])}


def design(motor_ranges):
    """motor_ranges: [(lo, hi)] rad for M1, M2, M3 (whole paper).
    Returns dict per axis with the spring spec and its torque law."""
    pre = CFG["drive"]["preload"]
    axle = CFG["drive"]["shaft_d"]
    gaps = gap_lengths()
    out = {}
    for k, nm in enumerate(("M1", "M2", "M3")):
        P = pre["hub_torque"][k] * 1000                # N mm
        sign = pre["sign"][k]
        lo, hi = motor_ranges[k]
        travel = hi - lo
        L_avail = gaps[nm] - 0.5
        cands = [(d, D) for d in WIRES for D in [axle + 2.5 + d + 2 * j for j in range(6)]
                 if D + d <= pre.get("max_od", 21.0)]
        options = []
        for d, D in cands:
            n = math.floor(L_avail / d) - 1             # close wound: body length (n + 1) d
            if n < 2:
                continue
            kr = E_WIRE * d ** 4 / (64 * D * n)         # N mm / rad
            var = kr * travel / 2 / P
            M_max = P + kr * travel / 2
            C = D / d
            Ki = (4 * C * C - C - 1) / (4 * C * (C - 1))
            sigma = Ki * 32 * M_max / (math.pi * d ** 3)
            allow = 0.75 * 2211 / d ** 0.145
            options.append(dict(axis=nm, d=d, D=D, n=n, body=(n + 1) * d, k=kr, P=P, sign=sign,
                                wind_deg=math.degrees(P / kr), var=var, sigma=sigma, allow=allow,
                                travel_deg=math.degrees(travel), theta_mid=(lo + hi) / 2,
                                ok=var <= pre["torque_variation"] and sigma <= allow,
                                gap=gaps[nm], OD=D + d, ID=D - d))
        ok = [o for o in options if o["ok"]]
        # best stress margin among valid springs; otherwise the least-bad one
        best = (min(ok, key=lambda o: o["sigma"] / o["allow"]) if ok else
                min(options, key=lambda o: max(o["var"] / pre["torque_variation"], o["sigma"] / o["allow"])))
        out[nm] = best
    return out


def hub_spring_torque(spec, theta):
    """Spring torque ON the hub (N m) at hub angle theta (rad)."""
    d = (theta - spec["theta_mid"] + math.pi) % (2 * math.pi) - math.pi      # wrap to +-pi
    return (spec["sign"] * spec["P"] - spec["k"] * d) / 1000


def report(specs):
    lines = ["# Anti-backlash torsion springs (shoulder axle)", "",
             "Each spring sits in the spacer gap under its hub, around the fixed shoulder axle: one leg in "
             "the keyed collar (D-flat), the other in the hub's leg hole. Wind it by `wind` before "
             "inserting the hub leg.", "",
             "| hub | preload (N·m) | wire d (mm) | mean D (mm) | OD / ID (mm) | coils | body / gap (mm) "
             "| rate (N·mm/rad) | wind (°) | torque ± over travel | stress / allowed (MPa) | ok |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm, s in specs.items():
        lines.append(f"| {nm} | {s['sign'] * s['P'] / 1000:+.2f} | {s['d']} | {s['D']:.1f} | {s['OD']:.1f} / {s['ID']:.1f} "
                     f"| {s['n']} | {s['body']:.1f} / {s['gap']:.1f} | {s['k']:.0f} | {s['wind_deg']:.0f} | "
                     f"{100 * s['var']:.0f} % over {s['travel_deg']:.0f}° | {s['sigma']:.0f} / {s['allow']:.0f} | "
                     f"{'yes' if s['ok'] else 'NO'} |")
    return "\n".join(lines)
