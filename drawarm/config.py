"""
config.py - loads params.toml (the single source of truth for every module).

    from .config import CFG          # nested dict, angles still in degrees
    CFG["arm"]["L1"]

Environment overrides:
    DRAWING_ARM_PARAMS=/path/to/other.toml   use another parameter file
    DRAWING_ARM_OUT=/path/to/dir             where generated files go (default ./output)
"""
import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.environ.get("DRAWING_ARM_PARAMS", os.path.join(ROOT, "params.toml"))
OUT = os.environ.get("DRAWING_ARM_OUT", os.path.join(ROOT, "output"))

with open(PATH, "rb") as _f:
    CFG = tomllib.load(_f)

PAPER = (CFG["paper"]["x0"], CFG["paper"]["y0"], CFG["paper"]["x1"], CFG["paper"]["y1"])

# Which layer every part lives on (layer thicknesses: [parts].layer_thickness).
# The cam order must match [drive].cam_stack.
LAYER = dict(cam_M2=0, crank_M2=1, follow_M2=1, coupler_E=2, cam_M1=2, upper_arm=3, follow_M1=3,
             forearm=4, bell_B=5, hand=5, cam_M3=5, coupler_B=6, gear_M3=6, follow_M3=6,
             bell_A=7, crank_M3=7, cam_pen=7, coupler_A=8, follow_pen=8)


def thickness(part):
    """Plate thickness (mm) of a part, from its layer."""
    return CFG["parts"]["layer_thickness"][LAYER[part]]


def out_path(*parts):
    """Path inside the output folder (created on demand)."""
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
