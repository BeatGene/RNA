python
from pathlib import Path
from pymol import cmd

bundle = Path.cwd()
targets = ('8hb8-assembly1',)
for target in targets:
    cmd.reinitialize()
    cmd.load(str(bundle / "structures" / (target + "_input.cif")), "before")
    cmd.load(str(bundle / "structures" / (target + "_refined.cif")), "after")

    # The FoldBench exports use predicted chain A.  Keep the target RNA only.
    for obj in ("before", "after"):
        if cmd.count_atoms(obj + " and chain A"):
            cmd.remove(obj + " and not chain A")
    before_p = cmd.count_atoms("before and name P")
    after_p = cmd.count_atoms("after and name P")
    if before_p < 3 or before_p != after_p:
        raise RuntimeError(f"{target}: phosphate atom counts differ: {before_p} vs {after_p}")

    # Rigidly superpose the refined RNA onto its own input, using phosphate
    # atoms and no outlier rejection.  This is only for display, not a new
    # benchmark RMSD calculation.
    cmd.align("after and name P", "before and name P", cycles=0)
    cmd.hide("everything")
    cmd.set_color("before_orange", [0.85, 0.45, 0.29])
    cmd.set_color("after_teal", [0.09, 0.49, 0.54])
    cmd.color("before_orange", "before")
    cmd.color("after_teal", "after")
    cmd.show("cartoon", "before")
    cmd.show("cartoon", "after")
    cmd.set("cartoon_transparency", 0.45, "before")
    cmd.set("cartoon_transparency", 0.0, "after")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("orthoscopic", 1)
    cmd.set("ray_shadows", 0)
    cmd.orient("before or after")
    cmd.zoom("before or after", buffer=4)

    (bundle / "images").mkdir(exist_ok=True)
    (bundle / "sessions").mkdir(exist_ok=True)
    cmd.save(str(bundle / "sessions" / (target + ".pse")))
    cmd.png(str(bundle / "images" / (target + "_overlay.png")),
            width=1800, height=1350, dpi=300, ray=1)
    print("RENDERED", target, "P_atoms", before_p)
python end
