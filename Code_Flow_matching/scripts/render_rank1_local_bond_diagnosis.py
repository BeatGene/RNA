"""Render two mapped local chemical failures with the installed PyMOL engine."""

from pathlib import Path
import csv
import json

import pymol
from pymol2 import SingletonPyMOL


def main():
    workspace = Path(__file__).resolve().parents[2]
    bundle = workspace / "pymol_bundle"
    output = workspace / "Code_Flow_matching/evaluation/local_rank1_diagnosis_20261013"
    output.mkdir(parents=True, exist_ok=True)
    pymol.invocation.parse_args(["pymol", "-cq"])
    engine = SingletonPyMOL()
    engine.start()
    cmd = engine.cmd
    cmd.cd(str(bundle))
    cmd.run(str(bundle / "rna_pymol_viewer.py"), namespace="module")
    for pdb, residue, a, b in (("8JHP", 6, "C6", "N6"), ("7ZJ4", 199, "C2", "N3")):
        cmd.do(f"rna_open test, {pdb}")
        cmd.hide("everything", "native or before or after")
        focus = f"(native or before or after) and resi {residue}"
        cmd.show("sticks", focus)
        cmd.show("spheres", focus)
        cmd.set("sphere_scale", 0.18)
        cmd.set("stick_radius", 0.10)
        cmd.set("label_size", 22)
        cmd.set("label_color", "black")
        cmd.set("ray_shadows", 0)
        for name in ("native", "before", "after"):
            left = f"{name} and resi {residue} and name {a}"
            right = f"{name} and resi {residue} and name {b}"
            cmd.distance("bond_" + name, left, right)
            cmd.set("dash_color", {"native": "gray60", "before": "marine", "after": "orange"}[name], "bond_" + name)
        cmd.orient(f"native and resi {residue}")
        cmd.zoom(focus, buffer=3)
        path = output / f"{pdb}_residue{residue}_bond.png"
        cmd.png(str(path), width=1500, height=1100, dpi=200, ray=1)
        cmd.save(str(output / f"{pdb}_residue{residue}_bond.pse"))
        print("LOCAL_BOND_RENDERED", path, flush=True)
        for name in ("native", "before", "after"):
            cmd.delete("bond_" + name)
    engine.stop()


if __name__ == "__main__":
    main()
