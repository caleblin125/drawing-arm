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
    pd = sub.add_parser("drum", help="multi-turn cam drum: report, output/drum.stl, output/drum_<name>.png")
    pd.add_argument("drawing", nargs="?", default="heart")
    pd.add_argument("--turns", type=int, default=None, help="drum turns per drawing (default: params.toml)")
    pd.add_argument("--parts", action="store_true", help="also build the drum-mode printable parts")
    pc_ = sub.add_parser("cartridge", help="fixed followers + swappable cam stacks (output/cartridge/<name>/*.stl)")
    pc_.add_argument("drawings", nargs="*", default=["heart"])
    pc_.add_argument("--no-stl", action="store_true", help="report only")
    pm = sub.add_parser("machine", help="complete single-motor automaton with swappable cams (output/machine/<name>)")
    pm.add_argument("drawing", nargs="?", default="heart")
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
    elif a.cmd == "drum":
        from .cams import plan_for_cams
        from .config import PAPER, out_path
        from .drum import DrumParams, design_drum, drum_triangles, plot_drum, report, write_stl
        from .inputs import fit_to_box
        from .kinematics import Geometry
        from .shapes import drawing_name, load_drawing
        g = Geometry()
        dp = DrumParams() if a.turns is None else DrumParams(turns=a.turns)
        plan, p = plan_for_cams(g, fit_to_box(load_drawing(a.drawing), PAPER))
        ds = design_drum(plan, g, p, dp)
        print(report(ds))
        if ds.pitch and all(x is not None for x in ds.axes + [ds.pen_axis]):
            plot_drum(ds, out_path(f"drum_{drawing_name(a.drawing)}.png"))
            write_stl(drum_triangles(ds), out_path("drum.stl"))
            print(f"-> {out_path('drum.stl')}, {out_path(f'drum_{drawing_name(a.drawing)}.png')}")
            if a.parts:
                from . import springs as spr
                from .parts import Builder, export, preview, drive_clearance
                b = Builder(g, ds)
                parts = b.build()
                for what, who, lay, c in drive_clearance(b.layout):
                    print(f"  {what:24s} {c:6.1f} mm  (nearest: {who} L{lay})" + ("   <-- COLLISION" if c < 0 else ""))
                for what, c in b.bay_clearance():
                    print(f"  {what:24s} {c:6.1f} mm" + ("   <-- COLLISION" if c < 0 else ""))
                for nm, r in b.rockers.items():
                    print(f"  rocker shaft {nm:4s}: 8 mm D-shaft, {r['length']:.0f} mm long")
                outdir = out_path("drum_parts", "x")[:-2]
                export(parts, outdir)
                write_stl(drum_triangles(ds), out_path("drum_parts", "stl", "drum.stl"))
                preview(parts, out_path("drum_parts", "preview.png"))
                with open(out_path("drum_parts", "springs.md"), "w", encoding="utf-8") as f:
                    f.write(spr.report(b.springs))
                print(f"{len(parts) + 1} parts -> {outdir}")
    elif a.cmd == "cartridge":
        import cadquery as cq
        from .cams import plan_for_cams
        from .cartridge import (cam_solids, cams_for, design_followers, plot_cartridge, report,
                                slide_out_check)
        from .config import PAPER, out_path
        from .inputs import fit_to_box
        from .kinematics import Geometry
        from .shapes import drawing_name, load_drawing
        g = Geometry()
        fol = design_followers(g)
        res = []
        for src in a.drawings:
            plan, _ = plan_for_cams(g, fit_to_box(load_drawing(src), PAPER))
            cs, probs = cams_for(plan, g, fol)
            clear, s_free = slide_out_check(cs, fol)
            res.append((drawing_name(src), cs, probs, clear))
            name = drawing_name(src)
            plot_cartridge(cs, fol, out_path("cartridge", name, "layout.png"), f"cam cartridge: {name}")
            if not a.no_stl:
                for nm, solid in cam_solids(cs, fol).items():
                    cq.exporters.export(solid, out_path("cartridge", name, f"cam_{nm}.stl"),
                                        tolerance=0.05, angularTolerance=0.1)
        print(report(fol, res))
        print(f"-> {out_path('cartridge')}")
    elif a.cmd == "machine":
        import cadquery as cq
        from .cams import plan_for_cams
        from .cartridge import cams_for, design_followers
        from .config import PAPER, out_path
        from .inputs import fit_to_box
        from .kinematics import Geometry
        from .machine import Machine
        from .parts import preview
        from .shapes import drawing_name, load_drawing
        g = Geometry()
        fol = design_followers(g)
        plan, _ = plan_for_cams(g, fit_to_box(load_drawing(a.drawing), PAPER))
        cs, probs = cams_for(plan, g, fol)
        if probs:
            print("cam stack problems: " + "; ".join(probs))
        m = Machine(g, plan, fol, cs)
        parts = m.build()
        bad = False
        for what, c in m.checks():
            print(f"  {what:48s} {c:7.1f} mm" + ("   <-- COLLISION" if c < 0 else ""))
            bad |= c < 0
        name = drawing_name(a.drawing)
        asm = cq.Assembly(name="drawing_automaton")
        for k, (solid, lay, col) in parts.items():
            asm.add(solid, name=k, color=cq.Color(col))
            sub_ = "cams" if k in m.cam_names else "fixed"
            bb = solid.val().BoundingBox()
            cq.exporters.export(solid.translate((-bb.center.x, -bb.center.y, -bb.zmin)),
                                out_path("machine", name, sub_, f"{k}.stl"), tolerance=0.05, angularTolerance=0.1)
        asm.export(out_path("machine", name, "assembly.step"))
        preview(parts, out_path("machine", name, "preview.png"))
        with open(out_path("machine", name, "BOM.md"), "w", encoding="utf-8") as f:
            f.write(m.bom())
        print(m.bom())
        print(f"{len(parts)} parts -> {out_path('machine', name)}" + ("  (fix collisions first!)" if bad else ""))
    elif a.cmd == "strain":
        from .structural import run
        print(run(a.drawing))


if __name__ == "__main__":
    main()
