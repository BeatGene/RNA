"""Portable interactive viewer. Run this INSIDE PyMOL from the bundle root."""

from pathlib import Path
import csv
import json

from pymol import cmd
from chempy import Bond

BUNDLE = Path(__file__).resolve().parent
STATE = {"split": None, "pdb": None, "fit": "all_atom", "observed": False}
OBJECTS = ("native", "before", "after")
BACKBONE = "name P+OP1+OP2+OP3+O5'+C5'+C4'+O4'+C3'+O3'+C2'+O2'+C1'"
COLORS = {"native": "gray60", "before": "marine", "after": "orange"}


def table(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def target_root():
    if STATE["split"] is None:
        raise RuntimeError("Open a target first: rna_open test, 9OD4")
    return BUNDLE / STATE["split"] / STATE["pdb"]


def subset():
    return "native or before_observed or after_observed" if STATE["observed"] else "native or before or after"


def install_topology(name, metadata):
    """Replace guessed bonds, preserving exact mapped atom IDs and positions."""
    model = cmd.get_model(name, state=1)
    expected = metadata["observed_atom_ids"] if name == "native" else metadata["atom_ids"]
    id_to_index = {atom.id: index for index, atom in enumerate(model.atom)}
    if len(id_to_index) != len(model.atom) or set(id_to_index) != set(expected):
        raise RuntimeError(f"{name}: PyMOL did not preserve exported atom IDs")
    ids_by_identity = {(atom.resi, atom.name): atom.id for atom in model.atom}
    if len(ids_by_identity) != len(model.atom):
        raise RuntimeError(f"{name}: duplicate residue/atom identities")
    model.bond = []
    for a, b in metadata["bonds"]:
        if a in id_to_index and b in id_to_index:
            bond = Bond()
            bond.index = [id_to_index[a], id_to_index[b]]
            bond.order = 1
            model.bond.append(bond)
    cmd.delete(name)
    cmd.load_model(model, name, state=1, discrete=1, zoom=0)
    # load_model can renumber IDs (notably the gaps in observed native).
    cmd.alter(name, "ID=ids_by_identity[(resi,name)]", space={"ids_by_identity": ids_by_identity})
    cmd.sort(name)
    if {atom.id for atom in cmd.get_model(name).atom} != set(expected):
        raise RuntimeError(f"{name}: mapped IDs changed while replacing topology")
    if len(cmd.get_bonds(name)) != len(model.bond):
        raise RuntimeError(f"{name}: template bond count mismatch")


def rna_open(split, pdb, fit="all_atom", preserve_view="0"):
    split, pdb, fit = str(split).strip(), str(pdb).strip().upper(), str(fit).strip()
    if split not in ("val", "test") or fit not in ("all_atom", "backbone"):
        raise ValueError("rna_open val|test, PDB_ID, all_atom|backbone")
    root = BUNDLE / split / pdb
    metadata = json.loads((root / "display.json").read_text(encoding="utf-8"))
    camera = cmd.get_view() if int(preserve_view) else None
    # Only replace this viewer's objects/selections; other user objects stay available.
    for name in (*OBJECTS, "before_observed", "after_observed", "rna_focus", "rna_worse"):
        cmd.delete(name)
    for name in OBJECTS:
        cmd.load(str(root / fit / (name + ".cif")), name, state=1, zoom=0)
        install_topology(name, metadata)
    STATE.update(split=split, pdb=pdb, fit=fit, observed=False)
    cmd.select_list("before_observed", "before", metadata["observed_atom_ids"], mode="id")
    cmd.select_list("after_observed", "after", metadata["observed_atom_ids"], mode="id")
    cmd.disable("before_observed")
    cmd.disable("after_observed")
    cmd.set("valence", 0, "native or before or after")
    cmd.set("orthoscopic", 1)
    cmd.bg_color("white")
    rna_overview()
    if camera:
        cmd.set_view(camera)
    else:
        cmd.orient("native")
        cmd.zoom("native or before or after", buffer=4)
    metric = metadata["transforms"][fit]
    print(f"RNA {split}/{pdb} | {fit} fit | native=gray before=blue after=orange")
    print(f"RMSD {metric['input_rmsd_a']:.4f} -> {metric['refined_rmsd_a']:.4f} A")
    print("Commands: rna_fit backbone / rna_overview / rna_sticks / rna_observed 1")
    print("          rna_residue 12 / rna_color_change / rna_worst / rna_next / rna_save")


def rna_fit(mode="all_atom"):
    rna_open(STATE["split"], STATE["pdb"], mode, preserve_view="1")


def rna_overview():
    target_root()
    selection = "native or before or after"
    cmd.hide("everything", selection)
    for name in OBJECTS:
        cmd.color(COLORS[name], name)
    # Exact atom-to-atom backbone lines retain gaps in native and deformed bonds.
    cmd.show("lines", f"({subset()}) and ({BACKBONE})")
    cmd.set("line_width", 2.0, selection)
    cmd.show("spheres", f"({subset()}) and name P")
    cmd.set("sphere_scale", 0.18, selection)


def rna_sticks():
    target_root()
    cmd.hide("everything", "native or before or after")
    for name in OBJECTS:
        cmd.color(COLORS[name], name)
    cmd.show("sticks", subset())
    cmd.set("stick_radius", 0.12, "native or before or after")


def rna_observed(enabled="1"):
    target_root()
    STATE["observed"] = bool(int(enabled))
    rna_overview()
    print("Observed native correspondence only:", STATE["observed"])


def rna_residue(start, end=None):
    first = int(start)
    last = int(end) if end not in (None, "") else first
    if first < 1 or last < first:
        raise ValueError("residue range must satisfy 1 <= start <= end")
    rna_sticks()
    cmd.hide("sticks", "native or before or after")
    cmd.select("rna_focus", f"({subset()}) and resi {first}-{last}")
    if not cmd.count_atoms("rna_focus"):
        raise ValueError("no atoms in that residue range")
    cmd.show("sticks", "rna_focus")
    cmd.show("lines", f"({subset()}) and ({BACKBONE})")
    cmd.zoom("rna_focus", buffer=5)
    cmd.disable("rna_focus")
    for row in table(target_root() / "residue_metrics.tsv"):
        if first <= int(row["residue"]) <= last:
            print("RESIDUE", row["residue"], row["base"], "observed", row["native_observed_atoms"],
                  "error_before", row["input_rms_error_all_atom_fit_a"],
                  "error_after", row["refined_rms_error_all_atom_fit_a"],
                  "improvement_A", row["improvement_a"])


def rna_color_change(limit="0.5"):
    """Color after by per-residue improvement after global all-atom fitting."""
    scale = float(limit)
    if scale <= 0:
        raise ValueError("color limit must be positive")
    cmd.color("gray70", "after")
    for row in table(target_root() / "residue_metrics.tsv"):
        if row["improvement_a"] == "":
            continue
        value = max(-1.0, min(1.0, float(row["improvement_a"]) / scale))
        color = [1.0, 1.0 + value, 1.0 + value] if value < 0 else [1.0 - value, 1.0, 1.0 - value]
        name = "rna_change_" + row["residue"]
        cmd.set_color(name, color)
        cmd.color(name, "after and resi " + row["residue"])
    print("after: red=worse green=better white=unchanged gray=no native coordinates")
    print(f"Color saturates at +/- {scale} A; values always use global ALL-ATOM fits.")


def rna_worst(count="10"):
    rows = [row for row in table(target_root() / "residue_metrics.tsv") if row["improvement_a"] != ""]
    rows.sort(key=lambda row: float(row["improvement_a"]))
    for row in rows[:int(count)]:
        print("WORST_RESIDUE", row["residue"], row["base"], "improvement_A", row["improvement_a"])
    if rows:
        rna_residue(rows[0]["residue"])


def navigate(offset):
    rows = table(BUNDLE / ("inspection_order_" + STATE["split"] + ".tsv"))
    index = next(i for i, row in enumerate(rows) if row["pdb_id"].upper() == STATE["pdb"])
    selected = rows[(index + offset) % len(rows)]
    rna_open(STATE["split"], selected["pdb_id"], STATE["fit"])


def rna_next():
    target_root()
    navigate(1)


def rna_prev():
    target_root()
    navigate(-1)


def rna_save():
    target_root()
    directory = BUNDLE / "sessions" / STATE["split"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (STATE["pdb"] + "_" + STATE["fit"] + ".pse")
    cmd.save(str(path))
    print("SESSION_SAVED", path)


def rna_render_all():
    for row in table(BUNDLE / "manifest.tsv"):
        for mode in ("all_atom", "backbone"):
            rna_open(row["split"], row["pdb_id"], mode)
            rna_save()
            directory = BUNDLE / "images" / row["split"]
            directory.mkdir(parents=True, exist_ok=True)
            cmd.png(str(directory / (row["pdb_id"] + "_" + mode + ".png")),
                    width=1400, height=1000, dpi=150, ray=0)
    print("RNA_RENDER_ALL_COMPLETE")


for function in (rna_open, rna_fit, rna_overview, rna_sticks, rna_observed, rna_residue,
                 rna_color_change, rna_worst, rna_next, rna_prev, rna_save, rna_render_all):
    cmd.extend(function.__name__, function)
