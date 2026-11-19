# drawing-arm

Kinematics, motion planning and a tkinter simulator for a human-like drawing
automaton: a planar shoulder–elbow–wrist arm whose **three input shafts sit
coaxially at the shoulder** (on the body), with the elbow and wrist driven
remotely through parallelogram linkages.

```
pip install -r requirements.txt
python arm_linkage.py          # simulator
python arm_linkage.py --test   # numerical self-checks
```

## Mechanism

| Input | Drives | Linkage |
|---|---|---|
| M1 | upper arm | direct |
| M2 | forearm (absolute angle) | parallelogram across the upper arm (shoulder crank → coupler → forearm tail): the "5-bar with both inputs at the shoulder" |
| M3 | hand (absolute angle) | parallelogram across the upper arm → free L-shaped bell-crank at the elbow → parallelogram across the forearm → hand crank |

Because every motor sets an **absolute** link angle, the kinematics decouple:
turning M1 alone translates the forearm and hand without rotating them.

Default sizes (mm): upper arm 300, forearm 250, hand→pen 70; cranks 60 / 45 / 40.
Edit `Geometry` in `arm_kinematics.py`.

**Dead points.** A parallelogram can flip into an anti-parallelogram when its
crank lines up with its link. The elbow is kept between 20° and 160° of bend,
and the 90° elbow bell-crank keeps both wrist stages well conditioned. The
simulator shows the transmission margin |sin| for every stage (1 = ideal),
and manual mode lets you drive through a dead point to watch the flip.

## Human range of motion

Profiles in `arm_kinematics.ROM_PROFILES` (typical adult active ROM,
clinical goniometry norms):

* shoulder horizontal abduction 45° / adduction 130°, measured from the arm pointing straight out sideways
* elbow flexion 20°–145° (0° is the parallelogram dead point)
* wrist flexion 80° / extension 70° (palm sideways), or radial 20° / ulnar 30° deviation (palm down)

The IK rejects poses outside the profile and adds a quartic *discomfort* cost
that pushes joints toward mid-range, as people do. Green shading in the GUI
shows the reachable pen area.

## Motion planning (`path_planner.py`)

1. **Stroke order**: greedy nearest-neighbour. Strokes may be reversed, and closed loops start at their nearest point.
2. **Human-like timing**:
   * pen down follows the **2/3 power law** `v = k·R^(1/3)`, so the pen slows in tight curves and corners, with an acceleration limit. Each stroke starts and ends at rest.
   * pen up is a **minimum-jerk** straight reach, with a Fitts-style log duration.
3. **Minimum-energy redundancy resolution**: the hand angle is redundant for
   2-D pen position. Dynamic programming over a 2° grid of hand angles minimises
   `∫ [ q̇ᵀ M(q) q̇ + w_c · discomfort(q) + margin penalty ] dt`
   over the whole job (globally, not point by point). The planner then smooths
   the result and re-solves the arm exactly. It typically uses 10–15% less
   kinetic effort than greedy per-point IK, with lower RMS torque.
4. **Dynamics**: exact planar rigid-body model in absolute coordinates, so
   motor torque is the generalised force:
   `τ_j = Σ_k H_jk [cos(q_j−q_k) q̈_k + sin(q_j−q_k) q̇_k²]`, plus the
   reflected motor inertia. The self-test checks it with a power balance.

The **Motors** tab shows range, peak/RMS speed and torque, and work per motor,
plots the trajectories, and exports a CSV (`t, pen, x, y, joint angles, motor
angles, speeds, torques`) for sizing motors and gearboxes. Masses and inertias
are placeholders until the CAD exists.

## Inputs (`vector_input.py`)

* **SVG**: paths (all commands including arcs), line, polyline, polygon, rect, circle, ellipse, and nested transforms
* **Images** (PNG/JPG/...): `outline` (iso-contours of brightness), `hatch` (tone-dependent cross-hatching with alternating direction), or `both`
* Drawings are scaled to fit the paper (260 × 220 mm, 250 mm in front of the shoulder)

`examples/sample.svg` is a small test drawing.

## Not modelled yet

* rotation of the whole arm plane about a vertical axis
* pen-lift actuator (pen up/down is a flag)
* gravity, friction, link compliance, backlash
