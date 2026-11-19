"""drawarm - a single-motor, cam-programmed drawing automaton.

Modules
    config      loads params.toml (every default lives there)
    kinematics  geometry, human range of motion, FK/IK, dynamics
    planner     stroke ordering, human-like timing, min-energy redundancy DP
    inputs      SVG / raster image -> strokes
    shapes      built-in test drawings, load_drawing()
    cams        encode a plan onto groove cams (one motor)
    springs     anti-backlash torsion springs
    parts       CadQuery printable parts, STL/STEP export
    structural  loads, stress, strain, pen-tip error
    sim         tkinter simulator
    selftest    numerical self-checks

Run `python -m drawarm --help`.
"""
