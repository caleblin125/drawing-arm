"""Command line:  python -m drawarm <command> [drawing]

drawing = a built-in shape (circle, star, spiral, heart, lissajous, flower,
grid), an .svg file, or an image.  Generated files go to ./output.
"""
import argparse
import sys


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m drawarm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sim", help="open the tkinter simulator")
    sub.add_parser("test", help="run the numerical self-checks")
    for name, text in (("cams", "design the cam stack for a drawing (+ output/cams_<name>.png)"),
                       ("parts", "generate printable parts: STL, STEP, preview, spring spec"),
                       ("strain", "structural report: stress, strain, pen error")):
        p = sub.add_parser(name, help=text)
        p.add_argument("drawing", nargs="?", default="heart")
    a = ap.parse_args(argv)

    if a.cmd == "sim":
        from .sim import run_gui
        run_gui()
    elif a.cmd == "test":
        from .selftest import self_test
        sys.exit(0 if self_test() else 1)
    elif a.cmd == "cams":
        from .cams import design_cams, plan_for_cams, plot_cams, report
        from .config import PAPER, out_path
        from .inputs import fit_to_box
        from .kinematics import Geometry
        from .shapes import drawing_name, load_drawing
        g = Geometry()
        plan, p = plan_for_cams(g, fit_to_box(load_drawing(a.drawing), PAPER))
        cs = design_cams(plan, g, p)
        print(report(cs))
        if all(x is not None for x in cs.axes + [cs.pen_axis]):
            plot_cams(cs, out_path(f"cams_{drawing_name(a.drawing)}.png"))
    elif a.cmd == "parts":
        from . import springs as spr
        from .cams import plot_cams, report
        from .config import OUT, out_path
        from .parts import build_all, drive_clearance, export, preview
        g, plan, cs, b, parts = build_all(a.drawing)
        print(report(cs))
        plot_cams(cs, out_path("cams.png"))
        for what, who, lay, c in drive_clearance(b.layout):
            print(f"  {what:24s} {c:6.1f} mm  (nearest: {who} L{lay})" + ("   <-- COLLISION" if c < 0 else ""))
        srep = spr.report(b.springs)
        print(srep)
        with open(out_path("springs.md"), "w", encoding="utf-8") as f:
            f.write(srep)
        export(parts)
        preview(parts, out_path("preview.png"))
        print(f"{len(parts)} parts -> {OUT}: stl/*.stl, assembly.step, preview.png, cams.png, springs.md")
    elif a.cmd == "strain":
        from .structural import run
        print(run(a.drawing))


if __name__ == "__main__":
    main()
