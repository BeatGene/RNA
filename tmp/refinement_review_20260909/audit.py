"""Read-only audit of project code; temporary fixtures are removed on exit.

Run using Python with torch and gemmi. Does not claim to run the full network.
The RMS method is extracted unchanged from source; scatter(sum) uses index_add.
"""
import ast
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

import gemmi
import torch

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'Code_Flow_matching'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture = load('audit_fixture', PROJECT / 'scripts/test_build_refinement_pt.py')
builder = fixture.module


def scatter_sum(src, index, dim=0, dim_size=None, reduce='sum'):
    assert dim == 0 and reduce == 'sum'
    return src.new_zeros(dim_size).index_add(0, index, src)


tree = ast.parse((PROJECT / 'etflow/models/model.py').read_text(encoding='utf-8'))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseFlow')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'residue_rms_error')
method.decorator_list = []
namespace = dict(torch=torch, Tensor=torch.Tensor, Optional=Optional, scatter=scatter_sum)
exec(compile(ast.Module(body=[method], type_ignores=[]), 'extracted_residue_rms_error', 'exec'), namespace)
residue_rms_error = namespace['residue_rms_error']

report = {}
for name, mask, coords in [
    ('missing_residue', [True, False], [[1., 0., 0.], [2., 0., 0.]]),
    ('zero_observed_error', [True, True], [[0., 0., 0.], [2., 0., 0.]]),
]:
    prediction = torch.tensor(coords, requires_grad=True)
    error, valid = residue_rms_error(prediction, torch.zeros_like(prediction), torch.tensor([0, 1]), 2, torch.tensor(mask))
    loss = error[valid].mean()
    loss.backward()
    report[name] = dict(loss=float(loss.detach()), gradient=prediction.grad.tolist(), finite_gradient=bool(prediction.grad.isfinite().all()))

with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / 'test.cif'
    fixture.write_cif(path, [('P', 'P', 'A', 'A', 1, 0., 0., 0.), ('C', "C4'", 'A', 'A', 1, 1., 0., 0.)], 'A')
    block, atoms = builder.read_atoms(path)
    raw = list(block.find_values('_atom_site.label_atom_id'))
    report['cif_names'] = dict(raw=raw, decoded=[gemmi.cif.as_string(s) for s in raw], parsed=[a.atom_name for a in atoms], ids=[builder.ATOM_NAME_TO_ID.get(a.atom_name, 0) for a in atoms])
    rows = [('P', 'P', 'A', 'A', 1, 0., 0., 0.), ('C', "C4'", 'A', 'A', 1, 1., 0., 0.), ('C', "C1'", 'A', 'A', 1, 0., 1., 0.), ('N', 'N9', 'A', 'A', 1, 0., 0., 1.)]
    pred, native, confidence, fm = [Path(temp) / n for n in ['pred.cif', 'native.cif', 'conf.json', 'fm.pt']]
    fixture.write_cif(pred, rows, 'A')
    fixture.write_cif(native, rows, 'A')
    # Same chemical names, differing legal CIF quoting.
    native_text = native.read_text(encoding='utf-8')
    for name in ['P', "C4'", "C1'", 'N9']:
        native_text = native_text.replace("'" + name + "'", '"' + name + '"')
    native.write_text(native_text, encoding='utf-8')
    confidence.write_text(json.dumps(dict(atom_to_token_idx=[0]*4, atom_plddt=[0.9]*4, token_pair_pae=[[0.]], token_pair_pde=[[0.]], contact_probs=[[1.]])), encoding='utf-8')
    torch.save(dict(residue_embedding=torch.zeros(1,640), sequences=['A'], chain_offsets=[0,1]), fm)
    try:
        builder.build_sample(pred, confidence, native, fm, 'test', 0, 0)
        report['different_cif_quoting'] = 'accepted'
    except ValueError as exc:
        report['different_cif_quoting'] = str(exc)
    # Original test uses identical quoting on both sides, hiding graph corruption.
    fixture.write_cif(native, rows, 'A')
    builder.native_chains.cache_clear()
    sample = builder.build_sample(pred, confidence, native, fm, 'test', 0, 0)
    report['identical_quoted_fixture'] = dict(atom_name_id=sample['atom_name_id'].tolist(), num_bonds=sample['geometry_bond_index'].shape[1], observed=int(sample['target_mask'].sum()))

report['purine_bonds'] = {}
for symbol in ['A', 'G']:
    names = list(dict.fromkeys(a for bond in builder.BASE_BONDS[symbol] for a in bond[:2]))
    lookup = {(0, n): i for i, n in enumerate(names)}
    _, _, bonds, _, exclusions = builder.build_graph(symbol, lookup)
    pair = sorted([lookup[(0, 'N9')], lookup[(0, 'C4')]])
    report['purine_bonds'][symbol] = dict(n9_c4_bond=pair in bonds.T.tolist(), n9_c4_clash_excluded=pair in exclusions.T.tolist())

torch.manual_seed(7)
mobile = torch.randn(8, 3)
rotation, _ = torch.linalg.qr(torch.randn(3, 3))
rotation[:, -1] *= torch.det(rotation)
fixed = mobile @ rotation + torch.tensor([3., -4., 2.])
aligned, _, _, rmsd = builder.kabsch_align(mobile, fixed)
report['kabsch_rotation_translation'] = dict(rmsd=rmsd, max_error=float((aligned-fixed).abs().max()))
v = torch.randn(8,3,16)
linear = torch.nn.Linear(16,32,bias=False)
rotate = lambda x: torch.einsum('ij,njc->nic', rotation, x)
report['channel_linear_equivariance_max_error'] = float((linear(rotate(v))-rotate(linear(v))).abs().max().detach())
edges = torch.tensor([[1.e-3,0.,0.], [0.,0.,0.]])
length = torch.linalg.vector_norm(edges,dim=-1,keepdim=True)
denom = length.masked_fill(length == 0, 1.)
report['proposed_normalization'] = (edges/denom).tolist()
print(json.dumps(report, indent=2, allow_nan=True))
