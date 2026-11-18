#!/usr/bin/env python3
"""Read-only, streaming audit of the three RNA refinement PT pools.

Requires Python >=3.9, torch, numpy and (unless --sources none) gemmi. No
Lightning, PyG or GPU. Default rebuild mode imports a pinned source builder
and its constants, independently reconstructing each sample from raw inputs.
PT inputs are loaded weights_only=True.
Reports are written only in a NEW output directory. Exit 0=complete/no errors,
1=complete with findings, 2=incomplete/fatal. Warnings do not change exit status.
"""
from __future__ import annotations

import argparse
import collections
import csv
import functools
import hashlib
import importlib.util
import json
import logging
import math
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SPLITS = ('train', 'val', 'test', 'holdout_foldbench_val_homology')
ATOM_NAMES = ("P", "OP1", "OP2", "OP3", "O5'", "C5'", "C4'", "O4'",
              "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7",
              "C5", "C6", "N1", "C2", "N3", "C4", "N6", "O6", "N2", "O2", "N4", "O4")
REQUIRED = ('pos pos_pred target_mask atomic_numbers atom_name_id residue_index sequence '
            'residue_type_id rnafm_embedding edge_index edge_attr geometry_bond_index '
            'ideal_bond_length atom_plddt atom_to_token_idx token_pair_pae token_pair_pde contact_probs').split()
SHAPES = {
    'pos': ['N', 3], 'pos_pred': ['N', 3], 'target_mask': ['N'],
    'atomic_numbers': ['N'], 'atom_name_id': ['N'], 'residue_index': ['N'],
    'atom_plddt': ['N'], 'atom_to_token_idx': ['N'], 'residue_type_id': ['L'],
    'rnafm_embedding': ['L', 640], 'edge_index': [2, 'E'], 'edge_attr': ['E', 7],
    'geometry_bond_index': [2, 'B'], 'ideal_bond_length': ['B'],
    'clash_exclusion_index': [2, 'C'], 'token_pair_pae': ['L', 'L'],
    'token_pair_pde': ['L', 'L'], 'contact_probs': ['L', 'L'],
    'predicted_atom_site_row': ['N'], 'native_atom_site_row': ['N'],
    'observed_residue_mask': ['L'], 'modified_residue_mask': ['L'],
    'prediction_to_native_residue_index': ['L'], 'protenix_original_token_index': ['L'],
    'protenix_original_atom_token_index': ['N'], 'protenix_residue_token_offsets': ['L+1'],
    'protenix_residue_token_index': ['K'], 'native_to_prediction_rotation': [3, 3],
    'native_to_prediction_translation': [3],
}
MEANINGS = {
    'pos': 'Target: aligned native on observed atoms; prediction placeholders elsewhere (angstrom).',
    'pos_pred': 'Unmodified selected decoy coordinates (angstrom).',
    'target_mask': 'True only where a native atom supplies supervision.',
    'residue_index': 'Zero-based local canonical residue index of each atom.',
    'atom_to_token_idx': 'Rebuilt residue index, NOT necessarily original Protenix token index.',
    'atom_name_id': 'Atom name IDs: 1..28, 0=unknown; see ATOM_NAMES in script.',
    'residue_type_id': 'A=0 C=1 G=2 U=3 (4=unknown in loader).',
    'rnafm_embedding': 'Residue-level RNA-FM embedding, broadcast to atoms by the loader.',
    'edge_index': 'Directed static edges; dynamic radius edges are constructed by the model.',
    'edge_attr': 'Seven edge channels; builder uses 0=intraresidue, 1=O3-P, 2=C4-C4.',
    'geometry_bond_index': 'Undirected chemical bonds for geometry regularization.',
    'ideal_bond_length': 'Template bond lengths in angstrom, not native measurements.',
    'clash_exclusion_index': 'Exclude chemical 1-2 and 1-3 pairs from clash scoring.',
    'atom_plddt': 'Atomic confidence in [0,1].',
    'token_pair_pae': 'Residue-pair PAE in angstrom before loader normalization.',
    'token_pair_pde': 'Residue-pair PDE in angstrom before loader normalization.',
    'contact_probs': 'Residue-pair contact probability in [0,1].',
    'native_to_prediction_rotation': 'Row-vector transform: native @ R + t = stored target.',
    'native_to_prediction_translation': 'Translation t of native-to-decoy transform (angstrom).',
    'pre_refinement_aligned_rmsd': 'Observed-atom proper-rotation Kabsch RMSD in angstrom.',
    'dataset_max_pre_refinement_rmsd': 'Build-time filter setting, not proof every stored PT meets it.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_atom(name):
    return {'O1P': 'OP1', 'O2P': 'OP2', 'O3P': 'OP3'}.get(name, name).replace('*', "'")


def fitted_rmsd(a, b):
    a, b = a.double(), b.double()
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vh = torch.linalg.svd(a.T @ b)
    fix = torch.eye(3, dtype=torch.float64)
    fix[-1, -1] = torch.sign(torch.linalg.det(u @ vh))
    return float(((a @ (u @ fix @ vh) - b).square().sum(-1).mean()).sqrt())


def fingerprint(value, h=None):
    """Content digest independent of torch.save ZIP metadata and dictionary order."""
    h = hashlib.sha256() if h is None else h
    if isinstance(value, torch.Tensor):
        v = value.detach().cpu().contiguous()
        h.update(('tensor:' + str(v.dtype) + ':' + str(tuple(v.shape))).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        h.update(b'dict')
        for k in sorted(value):
            fingerprint(k, h); fingerprint(value[k], h)
    elif isinstance(value, (tuple, list)):
        h.update(type(value).__name__.encode())
        for v in value:
            fingerprint(v, h)
    else:
        h.update((type(value).__name__ + ':' + repr(value) + '\n').encode())
    return h.hexdigest()


def tensor_close(actual, expected, name, atol=0.002, rtol=0.002):
    require(actual.shape == expected.shape, name + ': shape differs from source')
    require(torch.allclose(actual.double(), expected.double(), atol=atol, rtol=rtol),
            name + ': values differ from source')


def validate(d):
    """Validate model-facing contract; failure does not stop the other PTs."""
    require(isinstance(d, dict), 'PT must contain a dictionary')
    require(not (set(REQUIRED) - d.keys()), 'missing required keys: ' + str(sorted(set(REQUIRED)-d.keys())))
    seq = d['sequence']
    require(isinstance(seq, str) and seq and set(seq) <= set('ACGU'), 'noncanonical/empty sequence')
    n, length = d['pos_pred'].shape[0], len(seq)
    require(n >= 3, 'fewer than 3 atoms')
    dims = {'N': n, 'L': length, 'L+1': length+1}
    for key, shape in SHAPES.items():
        if key not in d:
            continue
        v = d[key]
        require(isinstance(v, torch.Tensor), key + ' is not a tensor')
        require(v.ndim == len(shape), key + ': invalid rank')
        for actual, expected in zip(v.shape, shape):
            if isinstance(expected, int) or expected in dims:
                require(actual == dims.get(expected, expected), key + ': invalid shape ' + str(tuple(v.shape)))
    for key, v in d.items():
        if isinstance(v, torch.Tensor):
            require(v.layout == torch.strided and not v.is_complex(), key + ': unsupported tensor')
            if v.is_floating_point():
                require(bool(torch.isfinite(v).all()), key + ': NaN/Inf')
    for key in ('pos', 'pos_pred', 'rnafm_embedding', 'edge_attr', 'ideal_bond_length',
                'atom_plddt', 'token_pair_pae', 'token_pair_pde', 'contact_probs'):
        require(d[key].is_floating_point(), key + ': expected floating dtype')
    for key in ('atomic_numbers', 'atom_name_id', 'residue_index', 'residue_type_id',
                'atom_to_token_idx', 'edge_index', 'geometry_bond_index', 'clash_exclusion_index',
                'predicted_atom_site_row', 'native_atom_site_row', 'prediction_to_native_residue_index',
                'protenix_original_token_index', 'protenix_original_atom_token_index',
                'protenix_residue_token_offsets', 'protenix_residue_token_index'):
        if key in d:
            require(d[key].dtype == torch.int64, key + ': expected int64')
    require(d['target_mask'].dtype == torch.bool, 'target_mask must be bool')
    mask = d['target_mask']
    require(int(mask.sum()) >= 3, 'fewer than 3 supervised atoms')
    for key in ('residue_index', 'atom_to_token_idx'):
        require(bool(((d[key] >= 0) & (d[key] < length)).all()), key + ': out of bounds')
    require(torch.equal(d['residue_index'], d['atom_to_token_idx']), 'residue/token indices disagree')
    require(torch.unique(torch.stack([d['residue_index'], d['atom_name_id']],dim=1),dim=0).shape[0] == n,
            'duplicate atom identities within residues')
    require(bool(((d['atom_name_id'] >= 0) & (d['atom_name_id'] <= 28)).all()), 'atom name ID out of range')
    require(bool(((d['atomic_numbers'] > 0) & (d['atomic_numbers'] < 20)).all()), 'atomic number outside model max_z=20')
    require(torch.equal(d['residue_type_id'], torch.tensor(['ACGU'.index(x) for x in seq])), 'residue IDs disagree with sequence')
    for key in ('atom_plddt', 'contact_probs'):
        require(bool(((d[key] >= 0) & (d[key] <= 1)).all()), key + ': outside [0,1]')
    for key in ('token_pair_pae', 'token_pair_pde'):
        require(bool((d[key] >= 0).all()), key + ': negative error')
    for key in ('edge_index', 'geometry_bond_index', 'clash_exclusion_index'):
        if key in d:
            v = d[key]
            require(bool(((v >= 0) & (v < n)).all()), key + ': atom index out of bounds')
            require(bool((v[0] != v[1]).all()), key + ': self edges')
            require(torch.unique(v.T, dim=0).shape[0] == v.shape[1], key + ': duplicate edges')
    ei, ea, bonds = d['edge_index'], d['edge_attr'], d['geometry_bond_index']
    require(ea.shape[0] == ei.shape[1], 'edge attributes count mismatch')
    require(bool(((ea == 0) | (ea == 1)).all()) and bool((ea.sum(-1) == 1).all()), 'edge_attr is not one-hot')
    require(d['ideal_bond_length'].numel() == bonds.shape[1], 'bond length count mismatch')
    require(bool((d['ideal_bond_length'] > 0).all()), 'nonpositive ideal bond length')
    edges = set(map(tuple, ei.T.tolist()))
    require(all((j, i) in edges for i, j in edges), 'static edges not bidirectional')
    require(all((i, j) in edges and (j, i) in edges for i, j in bonds.T.tolist()), 'chemical bonds absent from static graph')
    if 'clash_exclusion_index' in d:
        exclusions = {tuple(sorted(p)) for p in d['clash_exclusion_index'].T.tolist()}
        adj = collections.defaultdict(set)
        for i, j in bonds.T.tolist():
            require(tuple(sorted((i,j))) in exclusions, 'bond absent from clash exclusions')
            adj[i].add(j); adj[j].add(i)
        for ns in adj.values():
            require(all(tuple(sorted((i,j))) in exclusions for i in ns for j in ns if i < j), '1-3 pair absent from clash exclusions')
    if 'native_atom_site_row' in d:
        require(torch.equal(mask, d['native_atom_site_row'] >= 0), 'native row/mask mismatch')
    if 'observed_atom_count' in d:
        require(int(mask.sum()) == d['observed_atom_count'], 'observed atom count mismatch')
    if 'observed_atom_fraction' in d:
        require(abs(float(mask.float().mean())-d['observed_atom_fraction']) < 1e-6, 'observed fraction mismatch')
    if 'observed_residue_mask' in d:
        expected = torch.zeros(length, dtype=torch.bool); expected[d['residue_index'][mask].unique()] = True
        require(torch.equal(expected, d['observed_residue_mask']), 'observed residue mask mismatch')
    if bool((~mask).any()):
        tensor_close(d['pos'][~mask], d['pos_pred'][~mask], 'unobserved placeholders', 1e-5, 0)
    if 'protenix_residue_token_offsets' in d:
        off, tokens = d['protenix_residue_token_offsets'], d['protenix_residue_token_index']
        require(int(off[0]) == 0 and int(off[-1]) == tokens.numel() and bool((off[1:] > off[:-1]).all()), 'invalid residue token offsets')
        require(bool((tokens >= 0).all()) and tokens.unique().numel() == tokens.numel(), 'invalid/shared original tokens')
    if 'native_to_prediction_rotation' in d:
        rot = d['native_to_prediction_rotation'].double()
        tensor_close(rot.T @ rot, torch.eye(3), 'rotation orthogonality', 1e-4, 0)
        require(abs(float(torch.linalg.det(rot))-1) < 1e-4, 'rotation is not proper')
    rmsd = fitted_rmsd(d['pos_pred'][mask], d['pos'][mask])
    frame = float((d['pos_pred'][mask].double()-d['pos'][mask].double()).square().sum(-1).mean().sqrt())
    require(abs(frame-rmsd) <= 0.01, 'stored pair is not already in a common optimally fitted frame')
    if 'pre_refinement_aligned_rmsd' in d:
        require(abs(rmsd-float(d['pre_refinement_aligned_rmsd'])) <= 0.01, 'stored/recomputed RMSD mismatch')
    return n, length, rmsd


@functools.lru_cache(maxsize=4)
def cif_data(path):
    import gemmi
    block = gemmi.cif.read_file(str(path)).sole_block()
    def col(tag):
        return [gemmi.cif.as_string(v) for v in block.find_values(tag)]
    xyz = torch.tensor(list(zip(*(map(float,col('_atom_site.Cartn_'+axis)) for axis in 'xyz'))), dtype=torch.float64)
    return {'xyz': xyz, 'name': col('_atom_site.label_atom_id'), 'element': col('_atom_site.type_symbol'),
            'chain': col('_atom_site.label_asym_id'), 'seq': col('_atom_site.label_seq_id'),
            'asym': col('_struct_asym.id'), 'entity': col('_struct_asym.entity_id'),
            'poly_entity': col('_entity_poly.entity_id'), 'poly_type': col('_entity_poly.type'),
            'poly_sequence': col('_entity_poly.pdbx_seq_one_letter_code_can'),
            'seq_entity': col('_entity_poly_seq.entity_id'), 'seq_num': col('_entity_poly_seq.num'),
            'seq_comp': col('_entity_poly_seq.mon_id')}


@functools.lru_cache(maxsize=2)
def fm_data(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def resolve_source(recorded, fallback):
    # Prefer explicitly configured server roots; retain metadata paths as fallback.
    for p in [*fallback, Path(recorded) if recorded else None]:
        if p is not None and p.is_file():
            return p
    raise FileNotFoundError('source missing; searched: ' + str(fallback) + ' recorded=' + str(recorded))


def source_check(d, args, split, pdb, seed, sample):
    warnings = []
    needed = ('predicted_atom_site_row', 'native_atom_site_row', 'native_to_prediction_rotation',
              'native_to_prediction_translation', 'predicted_label_seq_ids', 'native_label_seq_ids',
              'prediction_to_native_residue_index')
    require(all(k in d for k in needed), 'missing source provenance fields: '+str([k for k in needed if k not in d]))
    pred_dir = args.prediction_root / split / pdb
    if not pred_dir.exists():
        pred_dir = args.prediction_root / split / pdb.upper()
    candidates = sorted(pred_dir.glob('seed_'+str(seed)+'/predictions/*_sample_'+str(sample)+'.cif'))
    candidates = [p for p in candidates if not p.name.endswith('_wounresol.cif')]
    require(len(candidates)==1, 'expected exactly one decoy CIF in configured Data_V2; found '+str(len(candidates)))
    pred = resolve_source(d.get('source_predicted_cif'), candidates)
    native = resolve_source(d.get('source_native_cif'), [args.native_root/(pdb+'.cif'),args.native_root/(pdb.upper()+'.cif')])
    pc, nc = cif_data(str(pred)), cif_data(str(native))
    rows, nr, mask = d['predicted_atom_site_row'], d['native_atom_site_row'], d['target_mask']
    require(bool(((rows >= 0) & (rows < len(pc['xyz']))).all()) and rows.unique().numel() == rows.numel(), 'prediction CIF row range/uniqueness')
    require(bool((nr[mask] < len(nc['xyz'])).all()) and nr[mask].unique().numel() == int(mask.sum()), 'native CIF row range/uniqueness')
    tensor_close(d['pos_pred'], pc['xyz'][rows], 'prediction CIF coordinates', 0.002, 0)
    target = nc['xyz'][nr[mask]] @ d['native_to_prediction_rotation'].double() + d['native_to_prediction_translation'].double()
    tensor_close(d['pos'][mask], target, 'native CIF transformed coordinates', 0.01, 0)
    for i, row in enumerate(rows.tolist()):
        aid, residue = int(d['atom_name_id'][i]), int(d['residue_index'][i])
        require(aid > 0 and normalize_atom(pc['name'][row]) == ATOM_NAMES[aid-1], 'prediction atom name mapping')
        require(pc['chain'][row] == d['predicted_chain_id'], 'prediction chain mapping')
        require(pc['seq'][row] == str(d['predicted_label_seq_ids'][residue]), 'prediction residue mapping')
        import gemmi
        require(gemmi.Element(pc['element'][row]).atomic_number == int(d['atomic_numbers'][i]), 'atomic number/source element mismatch')
        if bool(mask[i]):
            j = int(nr[i]); native_res = int(d['prediction_to_native_residue_index'][residue])
            require(normalize_atom(nc['name'][j]) == ATOM_NAMES[aid-1], 'native atom name mapping')
            require(nc['element'][j].upper() == pc['element'][row].upper(), 'native/predicted element mismatch')
            require(nc['seq'][j] == str(d['native_label_seq_ids'][native_res]), 'native residue mapping')
            if 'native_label_chain_id' in d:
                require(nc['chain'][j] == d['native_label_chain_id'], 'native chain mapping')
    types = dict(zip(nc['poly_entity'], nc['poly_type']))
    rna_chains = [a for a,e in zip(nc['asym'],nc['entity']) if types.get(e,'').lower() == 'polyribonucleotide']
    require(len(rna_chains) == 1, 'native RNA chain count is '+str(len(rna_chains))+' (expected exactly one declared RNA chain)')
    entity = dict(zip(nc['asym'],nc['entity']))[rna_chains[0]]
    declared_ids = sorted({int(num) for ent,num in zip(nc['seq_entity'],nc['seq_num']) if ent==entity})
    require([str(i) for i in declared_ids] == [str(i) for i in d['native_label_seq_ids']], 'native declared residue numbering differs from PT')
    canon = dict(zip(nc['poly_entity'],nc['poly_sequence'])).get(entity,'')
    canon = ''.join(canon.split()).upper()
    if not canon:
        residues = {int(num):comp.upper() for ent,num,comp in zip(nc['seq_entity'],nc['seq_num'],nc['seq_comp']) if ent==entity}
        canon = ''.join(residues[k] for k in sorted(residues))
    if canon and set(canon) <= set('ACGU'):
        require(canon == d['sequence'], 'native declared canonical sequence differs from PT sequence')
    else:
        # Modified components need a CCD mapping; do not claim independent
        # sequence validation when only raw atom identities were checkable.
        warnings.append('native canonical sequence needs CCD mapping; atom mapping checked, full canonical sequence NOT independently verified')
    if args.sources != 'full':
        return warnings
    fm_path = resolve_source(d.get('source_rnafm_pt'), [args.rnafm_root/pdb.upper()/'rnafm_t12_residue_embeddings.pt'])
    fm = fm_data(str(fm_path))
    matches = [i for i,s in enumerate(fm['sequences']) if str(s).upper() == d['sequence']]
    if len(matches) > 1:
        expected = fm.get('expected_protenix_chain_ids', [])
        matches = [i for i in matches if i < len(expected) and str(expected[i]) == d['predicted_chain_id']]
    require(len(matches) == 1, 'RNA-FM sequence/chain match is absent or ambiguous')
    i = matches[0]; off = fm['chain_offsets']
    tensor_close(d['rnafm_embedding'], fm['residue_embedding'][int(off[i]):int(off[i+1])].half(), 'RNA-FM embedding', 0, 0)
    prefix = re.sub(r'_sample_\d+\.cif$', '', pred.name)
    conf = resolve_source(d.get('source_confidence_json'), [pred.with_name(prefix+'_full_data_sample_'+str(sample)+'.json')])
    with conf.open() as handle:
        c = json.load(handle)
    pl = torch.as_tensor(c['atom_plddt'], dtype=torch.float32)
    at = torch.as_tensor(c['atom_to_token_idx'], dtype=torch.int64)
    require(pl.numel() == len(pc['xyz']) and at.numel() == len(pc['xyz']), 'confidence/CIF atom count mismatch')
    selected = pl[rows]
    if float(selected.max()) > 1:
        selected = selected/100
    tensor_close(d['atom_plddt'], selected, 'source pLDDT', 1e-6, 0)
    tensor_close(d['protenix_original_atom_token_index'], at[rows], 'source atom/token mapping', 0, 0)
    off, flat = d['protenix_residue_token_offsets'], d['protenix_residue_token_index']
    groups = [flat[int(off[i]):int(off[i+1])] for i in range(len(d['sequence']))]
    counts = torch.tensor([len(g) for g in groups], dtype=torch.float32)
    group_idx = torch.repeat_interleave(torch.arange(len(groups)), counts.long())
    require(bool((flat >= 0).all()), 'negative original token index')
    for key in ('token_pair_pae', 'token_pair_pde', 'contact_probs'):
        matrix = torch.as_tensor(c[key], dtype=torch.float32)
        require(matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and int(flat.max()) < matrix.shape[0], 'source pair matrix shape/range')
        require(bool(torch.isfinite(matrix).all()), 'source pair matrix NaN/Inf')
        selected = matrix[flat][:,flat]
        rowmean = torch.zeros((len(groups),len(flat))).index_add_(0,group_idx,selected)/counts[:,None]
        agg = torch.zeros((len(groups),len(groups))).index_add_(1,group_idx,rowmean)/counts[None,:]
        tensor_close(d[key], agg.half(), key+' source aggregation', 0, 0)
    return warnings


def load_reference_builder(path):
    """Load a reference source file, never a module supplied by a PT payload."""
    require(path.is_file(), 'reference builder missing: '+str(path)+'; upload/extract the rebuild bundle')
    spec = importlib.util.spec_from_file_location('_pt_audit_reference_builder',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Avoid writing __pycache__ next to a reference builder in a source tree.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    require(callable(getattr(module,'build_sample',None)), 'reference has no build_sample')
    return module


def file_sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def rebuild_from_raw(args, builder, split, pdb, seed, sample):
    """Inputs come ONLY from configured roots + relative PT identity, not old mappings."""
    folder = args.prediction_root/split/pdb
    if not folder.exists(): folder=args.prediction_root/split/pdb.upper()
    candidates = [p for p in folder.glob(f'seed_{seed}/predictions/*_sample_{sample}.cif')
                  if not p.name.endswith('_wounresol.cif')]
    require(len(candidates)==1,'expected exactly one original prediction; found '+str(len(candidates)))
    pred=candidates[0]
    prefix=re.sub(r'_sample_\d+\.cif$','',pred.name)
    confidence=pred.with_name(f'{prefix}_full_data_sample_{sample}.json')
    native=resolve_source(None,[args.native_root/(pdb+'.cif'),args.native_root/(pdb.upper()+'.cif')])
    nc=cif_data(str(native)); types=dict(zip(nc['poly_entity'],nc['poly_type']))
    chains=[a for a,e in zip(nc['asym'],nc['entity']) if types.get(e,'').lower()=='polyribonucleotide']
    require(len(chains)==1,'native declares '+str(len(chains))+' RNA chains; expected single-RNA scope')
    fm=args.rnafm_root/pdb.upper()/'rnafm_t12_residue_embeddings.pt'
    # Rebuild chain selection, residue/atom mapping, masks, alignment, graph,
    # confidence aggregation, embedding and provenance from the raw files.
    result=builder.build_sample(pred,confidence,native,fm,pdb,seed,sample,args.ccd_components_file)
    result['dataset_max_pre_refinement_rmsd']=args.rmsd_cutoff
    return result


PROVENANCE_ONLY = {
    'schema_version','generator_version','mapping_policy','atom_mapping_version',
    'edge_type_version','geometry_template_version','protenix_model_name','rnafm_model_name',
    'dataset_max_pre_refinement_rmsd',
}


def compare_rebuilt(old, new, atol=0.002):
    """Return field-level differences. Equal serialization bytes are not required."""
    differences=[]
    for key in sorted(old.keys() | new.keys()):
        metadata=key.startswith('source_') or key in PROVENANCE_ONLY
        severity='WARNING' if metadata else 'ERROR'
        if key not in old or key not in new:
            differences.append((severity,key,'missing_in_existing' if key not in old else 'extra_in_existing',''))
            continue
        a,b=old[key],new[key]
        if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
            if a.shape!=b.shape:
                differences.append((severity,key,'shape',f'{list(a.shape)} != {list(b.shape)}'));continue
            if a.dtype!=b.dtype:
                differences.append((severity,key,'dtype',f'{a.dtype} != {b.dtype}'))
            if a.is_floating_point() and b.is_floating_point():
                # Only spatial coordinates/transforms/RMSD allow rounding noise;
                # FP16 features, templates and confidence must reproduce exactly.
                tol=atol if key in ('pos','pos_pred','native_to_prediction_rotation','native_to_prediction_translation') else 0.
                same=torch.allclose(a.double(),b.double(),atol=tol,rtol=0,equal_nan=False)
                if not same:
                    delta=(a.double()-b.double()).abs()
                    differences.append((severity,key,'values',f'max_abs={float(delta.max()) if delta.numel() else 0}; atol={tol}'))
            elif not torch.equal(a,b):
                differences.append((severity,key,'values','integer/bool mismatch'))
        elif isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
            differences.append((severity,key,'type',f'{type(a).__name__} != {type(b).__name__}'))
        elif isinstance(a,float) and isinstance(b,(float,int)):
            tolerance=atol if key=='pre_refinement_aligned_rmsd' else 1e-6 if key=='observed_atom_fraction' else 0.
            if not math.isclose(a,b,rel_tol=0,abs_tol=tolerance):
                differences.append((severity,key,'values',f'{a!r} != {b!r}; atol={tolerance}'))
        elif a!=b:
            differences.append((severity,key,'values',f'{repr(a)[:160]} != {repr(b)[:160]}'))
    return differences


class Audit:
    def __init__(self, args):
        self.args = args
        self.out = args.output
        self.out.mkdir(parents=True, exist_ok=False)
        self.log = logging.getLogger('pt-audit'); self.log.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(self.out/'run.log')):
            h.setFormatter(fmt); self.log.addHandler(h)
        self.counts = collections.Counter(); self.fields = {}; self.total = 0
        self.versions = collections.defaultdict(collections.Counter)
        self.handles = []
        self.issues = self.writer('issues.tsv', ['severity','code','path','detail'])
        self.rebuild_diffs = self.writer('rebuild_differences.tsv',['path','severity','field','kind','detail'])
        self.rebuild_rows = self.writer('rebuild_samples.tsv',['path','status','reference_fields','differences','seconds'])
        self.rebuild_counts = collections.Counter({'rebuilt':0,'matched':0,'different':0,'metadata_only':0,'data_matched':0,'failed':0})
        self.files = self.writer('files.tsv', ['pool','split','pdb','seed','sample','path','status','atoms','length','rmsd','seconds'])
        self.db = sqlite3.connect(self.out/'audit.sqlite')
        self.db.execute('create table files(pool text,split text,pdb text,seed integer,sample integer,path text,sequence text,digest text,rmsd real, unique(pool,split,pdb,seed,sample))')

    def writer(self, name, fields):
        h = (self.out/name).open('w', newline='', encoding='utf-8'); self.handles.append(h)
        w = csv.writer(h, delimiter='\t'); w.writerow(fields); return w

    def issue(self, level, code, path, detail):
        self.counts[level+':'+code] += 1
        self.issues.writerow([level,code,str(path),str(detail).replace('\n',' ')])
        # Every finding is in TSV; log first 5 of each code to avoid huge logs.
        if self.counts[level+':'+code] <= 5:
            self.log.warning('%s %s %s %s',level,code,path,detail)

    def schema(self, d):
        for key,v in d.items():
            f = self.fields.setdefault(key, {'present':0,'types':collections.Counter(),'shapes':collections.Counter(),'shape_ranges':{},'example':None})
            f['present'] += 1
            typ = 'Tensor/'+str(v.dtype) if isinstance(v,torch.Tensor) else type(v).__name__
            f['types'][typ] += 1
            if isinstance(v,torch.Tensor):
                f['shapes'][str(SHAPES.get(key,list(v.shape)))] += 1
                dims = f['shape_ranges'].setdefault(str(v.ndim), [[int(s),int(s)] for s in v.shape])
                for limits,size in zip(dims,v.shape):
                    limits[0]=min(limits[0],int(size)); limits[1]=max(limits[1],int(size))
            elif f['example'] is None:
                f['example'] = repr(v)[:240]
            if key in ('schema_version','generator_version','mapping_policy','atom_mapping_version','edge_type_version','geometry_template_version','dataset_max_pre_refinement_rmsd'):
                self.versions[key][str(v)] += 1

    def inspect(self, pool, root, path):
        start = time.monotonic(); row = [pool,'','','','',str(path),'ERROR','','','']
        errors_before = sum(v for k,v in self.counts.items() if k.startswith('ERROR:'))
        try:
            parts = path.relative_to(root).parts
            require(len(parts) == 4 and parts[0] in SPLITS, 'unexpected PT path layout')
            split,pdb,sd,fn = parts; pdb = pdb.lower()
            seed = int(re.fullmatch(r'seed_(\d+)',sd).group(1))
            sample = int(re.fullmatch(r'sample_(\d+)\.pt',fn).group(1))
            row[1:5] = [split,pdb,seed,sample]
            if pool == 'archive':
                require(split in ('val','test'), 'archive split must be val/test')
            before = path.stat()
            d = torch.load(path,map_location='cpu',weights_only=True)
            after = path.stat()
            require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns), 'file changed during loading')
            if isinstance(d,dict):
                self.total += 1; self.schema(d)
            if self.args.sources=='rebuild':
                t0=time.monotonic()
                try:
                    fresh=rebuild_from_raw(self.args,self.builder,split,pdb,seed,sample)
                    differences=compare_rebuilt(d,fresh,self.args.compare_atol)
                    for severity,key,kind,detail in differences:
                        self.rebuild_diffs.writerow([str(path),severity,key,kind,detail])
                        self.issue(severity,'rebuild_field_difference',path,key+': '+kind+' '+detail)
                        self.rebuild_counts['different_field:'+key]+=1
                    data_diff=any(item[0]=='ERROR' for item in differences)
                    rebuild_status='DIFFERENT' if data_diff else 'METADATA_ONLY' if differences else 'MATCH'
                    self.rebuild_rows.writerow([str(path),rebuild_status,len(fresh),len(differences),f'{time.monotonic()-t0:.4f}'])
                    self.rebuild_counts['rebuilt']+=1
                    self.rebuild_counts['different' if data_diff else 'metadata_only' if differences else 'matched']+=1
                    if not data_diff: self.rebuild_counts['data_matched']+=1
                    del fresh
                except Exception as exc:
                    self.issue('ERROR','rebuild_failed',path,repr(exc))
                    self.rebuild_counts['failed']+=1
                    self.rebuild_rows.writerow([str(path),'FAILED','','',f'{time.monotonic()-t0:.4f}'])
            n,length,rmsd = validate(d)
            for key, expected in [('native_structure_id',pdb),('protenix_seed',seed),('protenix_sample',sample)]:
                if key not in d:
                    self.issue('WARNING','missing_provenance',path,key)
                else:
                    actual = str(d[key]).lower() if key=='native_structure_id' else d[key]
                    require(actual == expected, 'metadata/path mismatch: '+key)
            if 'sample_id' in d:
                require(d['sample_id'] == f'{pdb}_seed_{seed}_sample_{sample}', 'sample_id/path mismatch')
            if 'native_sequence' in d:
                require(d['native_sequence'] == d['sequence'], 'native/decoy sequence mismatch')
            if 'clash_exclusion_index' not in d:
                self.issue('WARNING','legacy_clash_fallback',path,'Loader falls back to direct bonds only')
            if float(d['target_mask'].float().mean()) < self.args.min_observed:
                self.issue('WARNING','low_observed_fraction',path,float(d['target_mask'].float().mean()))
            cutoff = float(d.get('dataset_max_pre_refinement_rmsd',self.args.rmsd_cutoff))
            if pool == 'high' and rmsd <= self.args.rmsd_cutoff:
                self.issue('ERROR','high_pool_below_cutoff',path,rmsd)
            if pool != 'high' and rmsd > cutoff+1e-4:
                self.issue('WARNING','main_pool_above_metadata_cutoff',path,f'{rmsd} > {cutoff}; inspect historical override')
            if self.args.sources in ('coordinates','full'):
                try:
                    for warning in source_check(d,self.args,split,pdb,seed,sample):
                        self.issue('WARNING','source_sequence_unverified',path,warning)
                except Exception as exc:
                    self.issue('ERROR','source_check',path,repr(exc)); row[6] = 'SOURCE_ERROR'
            digest = fingerprint(d)
            try:
                self.db.execute('insert into files values(?,?,?,?,?,?,?,?,?)',
                                (pool,split,pdb,seed,sample,str(path),d['sequence'],digest,rmsd))
            except sqlite3.IntegrityError:
                self.issue('ERROR','duplicate_identity',path,'pool/split/pdb/seed/sample collision')
            row[7:10] = [n,length,rmsd]
            if row[6] != 'SOURCE_ERROR':
                row[6] = 'FINDINGS' if sum(v for k,v in self.counts.items() if k.startswith('ERROR:')) > errors_before else 'OK'
        except Exception as exc:
            self.issue('ERROR','invalid_pt',path,repr(exc))
        self.files.writerow(row+[f'{time.monotonic()-start:.4f}'])

    def raw_scores(self, split, pdb):
        directory = self.args.prediction_root/split/pdb
        if not directory.exists(): directory = self.args.prediction_root/split/pdb.upper()
        scores = {}
        for p in directory.glob('seed_*/predictions/*_summary_confidence_sample_*.json'):
            try:
                seed = int(re.fullmatch(r'seed_(\d+)',p.parent.parent.name).group(1))
                sample = int(re.search(r'_sample_(\d+)\.json$',p.name).group(1))
                score = float(json.loads(p.read_text())['ranking_score'])
                require(math.isfinite(score),'nonfinite ranking_score')
                require((seed,sample) not in scores,'duplicate raw ranking identity')
                scores[(seed,sample)] = score
            except Exception as exc:
                self.issue('ERROR','ranking_json',p,repr(exc))
        return scores

    def cross_checks(self):
        self.db.commit()
        for key,values in self.versions.items():
            if len(values)>1:
                self.issue('WARNING','mixed_metadata_versions',key,dict(values))
        report = self.writer('pdb_counts.tsv',['pool','split','pdb','candidates'])
        for pool,split,pdb,num in self.db.execute('select pool,split,pdb,count(*) from files group by pool,split,pdb order by pool,split,pdb'):
            report.writerow([pool,split,pdb,num])
            if pool=='current' and split in ('val','test') and num != 1:
                self.issue('ERROR','not_rank1',f'{pool}/{split}/{pdb}',num)
            if pool=='archive' and num != 200:
                self.issue('WARNING','archive_not_200',f'{pool}/{split}/{pdb}',num)
        for pool in ('current','archive','high'):
            for pdb,ns in self.db.execute('select pdb,count(distinct split) from files where pool=? group by pdb having count(distinct split)>1',(pool,)):
                self.issue('ERROR','pdb_cross_split',pool+'/'+pdb,ns)
        for pdb,ns in self.db.execute('select pdb,count(distinct split) from files group by pdb having count(distinct split)>1'):
            self.issue('ERROR','pdb_cross_split_across_pools',pdb,ns)
        overlaps = self.writer('sequence_overlaps.tsv',['split_a','pdb_a','split_b','pdb_b','sequence'])
        sql = '''with t as (select distinct split,pdb,sequence from files where pool='current')
                 select a.split,a.pdb,b.split,b.pdb,a.sequence from t a join t b
                 on a.sequence=b.sequence and a.split<b.split'''
        overlap_count = 0
        for row in self.db.execute(sql): overlaps.writerow(row); overlap_count += 1
        if overlap_count:
            self.issue('WARNING','exact_sequence_overlap','current',str(overlap_count)+' pairs; not a full homology audit')
        sql = '''select a.path,b.path,a.pool,b.pool,a.digest=b.digest from files a join files b
                 on a.split=b.split and a.pdb=b.pdb and a.seed=b.seed and a.sample=b.sample and a.pool<b.pool'''
        for a,b,ap,bp,equal in self.db.execute(sql):
            if 'high' in (ap,bp): self.issue('ERROR','high_pool_overlap',a,b)
            elif not equal: self.issue('ERROR','current_archive_payload_diff',a,b)
        ranking = self.writer('rank1_audit.tsv',['split','pdb','selected_seed','selected_sample','archive_count','scored_archive','best_available','best_raw'])
        for split,pdb,seed,sample,path in self.db.execute("select split,pdb,seed,sample,path from files where pool='current' and split in ('val','test')"):
            archive = list(self.db.execute("select seed,sample from files where pool='archive' and split=? and pdb=?",(split,pdb)))
            require_archive = (seed,sample) in archive
            if not require_archive: self.issue('ERROR','selected_not_in_archive',path,len(archive))
            scores = self.raw_scores(split,pdb)
            available = [x for x in archive if x in scores]
            if len(available)!=len(archive) or not available:
                self.issue('ERROR','rank1_unverifiable',path,f'archive={len(archive)} scored={len(available)}')
            best = sorted(available,key=lambda x:(-scores[x],*x))[0] if available else None
            raw_best = sorted(scores,key=lambda x:(-scores[x],*x))[0] if scores else None
            if best is not None and best != (seed,sample): self.issue('ERROR','wrong_available_rank1',path,str(best))
            if raw_best is not None and raw_best != (seed,sample):
                self.issue('WARNING','raw_rank1_differs',path,f'raw best={raw_best}; may have been filtered before PT selection')
            ranking.writerow([split,pdb,seed,sample,len(archive),len(available),best,raw_best])
        for split,pdb in self.db.execute("select distinct split,pdb from files where pool='archive' except select distinct split,pdb from files where pool='current'"):
            self.issue('ERROR','archive_without_current',split+'/'+pdb,'no selected PT')

    def write_schema(self):
        result = {'loaded_dictionaries':self.total,'versions':{k:dict(v) for k,v in self.versions.items()},'dimensions':{'N':'selected heavy atoms','L':'canonical residues','E':'directed static edges','B':'chemical bonds','C':'clash exclusion pairs','K':'original tokens across residues'},'fields':{}}
        lines = ['# Observed PT schema','',f'Loaded dictionaries: {self.total}. Presence counts include invalid dictionaries; inspect issues.tsv.',
                 '', '| Field | Type / dtype | Shape | Present | Meaning |','|---|---|---|---|---|']
        for key,f in sorted(self.fields.items()):
            entry = dict(f); entry['types']=dict(f['types']); entry['shapes']=dict(f['shapes'])
            entry['missing']=self.total-f['present']; entry['meaning']=MEANINGS.get(key,'Provenance/metadata; inspect example and builder.')
            result['fields'][key]=entry
            lines.append('| '+ ' | '.join(str(x).replace('|','/') for x in [key,', '.join(f['types']),', '.join(f['shapes']),str(f['present'])+'/'+str(self.total),entry['meaning']])+' |')
        lines += ['', 'Coordinates: each PT is a paired decoy/native frame. Different decoys need not share a global frame.',
                  'Model-facing node_attr (674), confidence edges (+4), num_tokens, and mobility summaries are constructed by dataset.py; they need not be serialized in PT.',
                  'This schema is observed, not a claim that all files passed. Exact sequence overlap does not replace homology analysis.']
        (self.out/'schema.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
        (self.out/'schema.md').write_text('\n'.join(lines)+'\n')

    def run(self):
        summary = {'complete':False,'sources':self.args.sources,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'torch':torch.__version__,'started_utc':datetime.now(timezone.utc).isoformat(),
                   'arguments':{k:str(v) for k,v in vars(self.args).items()}}
        status = 2
        try:
            started = time.monotonic(); last = started; count = 0
            if self.args.sources=='rebuild':
                self.builder=load_reference_builder(self.args.builder)
                self.args.ccd_components_file=self.builder.resolve_ccd_components_file(self.args.ccd_components_file)
                constants=self.args.builder.resolve().parents[1]/'etflow/data/constants.py'
                summary['reference']={'builder_path':str(self.args.builder),
                    'builder_sha256':hashlib.sha256(self.args.builder.read_bytes()).hexdigest(),
                    'constants_sha256':hashlib.sha256(constants.read_bytes()).hexdigest(),
                    'generator_version':self.builder.GENERATOR_VERSION,
                    'mapping_policy':self.builder.MAPPING_POLICY,
                    'ccd_components_file':str(self.args.ccd_components_file),
                    'ccd_sha256':file_sha256(self.args.ccd_components_file) if self.args.ccd_components_file else None,
                    'independent_from_existing_pt_mappings':True}
                self.log.info('REFERENCE %s',summary['reference'])
            for pool,root in [('current',self.args.current_root),('archive',self.args.archive_root),('high',self.args.high_root)]:
                require(root.is_dir(),'missing PT root: '+str(root))
                for split in (('val','test') if pool=='archive' else SPLITS):
                    if not (root/split).is_dir(): self.issue('WARNING','missing_split',root/split,'not present')
                found = 0
                self.log.info('SCAN pool=%s root=%s',pool,root)
                for path in sorted(root.rglob('*.pt')):
                    # Report unexpected layout too, rather than silently accepting extra PTs.
                    self.inspect(pool,root,path); count+=1; found+=1
                    if count % 100 == 0: self.db.commit()
                    if time.monotonic()-last >= self.args.progress_seconds:
                        self.log.info('PROGRESS files=%d elapsed=%.1fs findings=%s',count,time.monotonic()-started,dict(self.counts))
                        for h in self.handles: h.flush()
                        self.db.commit(); last=time.monotonic()
                    if self.args.limit and found>=self.args.limit: break
                summary[pool+'_files_scanned']=found
                if not found and pool!='high': self.issue('ERROR','empty_pool',root,'no PTs')
                for partial in root.rglob('*.pt.tmp'): self.issue('WARNING','partial_pt',partial,'incomplete builder output')
            if self.args.limit:
                self.issue('WARNING','limited_run',self.out,'cross-pool/rank checks skipped; this is not a complete audit')
            else:
                self.log.info('CROSS_CHECKS'); self.cross_checks()
            summary['native_cif_count']=sum(1 for _ in self.args.native_root.glob('*.cif'))
            summary['complete']=not bool(self.args.limit)
            summary['files_scanned']=count; summary['elapsed_seconds']=time.monotonic()-started
            status=1 if any(k.startswith('ERROR:') for k in self.counts) else 0
        except Exception as exc:
            summary['fatal']=repr(exc); self.log.exception('FATAL audit incomplete')
        finally:
            self.db.commit(); self.write_schema()
            summary['findings']=dict(self.counts); summary['exit_code']=status
            summary['rebuild']=dict(self.rebuild_counts)
            summary['finished_utc']=datetime.now(timezone.utc).isoformat()
            (self.out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
            for h in self.handles: h.flush(); h.close()
            self.db.close()
            self.log.info('AUDIT_FINISHED complete=%s exit=%d output=%s',summary['complete'],status,self.out)
        return status


def main():
    base=Path('/storage9920/home/tinghao.xia')
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('current-root','Data_PT_V2'),('archive-root','Data_PT_V2_all_candidates_20261013'),('high-root','Data_PT_V2_RMSD_GT30'),('prediction-root','Data_V2'),('native-root','pdb_data'),('rnafm-root','Data_FM/RNA_FM_embeddings')]:
        p.add_argument('--'+name,type=Path,default=base/default)
    p.add_argument('--output',type=Path,required=True,help='NEW report directory; cannot be inside any input root')
    reference=Path(__file__).resolve().parent/'reference/scripts/build_refinement_pt.py'
    p.add_argument('--sources',choices=('rebuild','none','coordinates','full'),default='rebuild',help='rebuild independently regenerates the entire payload; full is the older partial source check')
    p.add_argument('--builder',type=Path,default=reference if reference.is_file() else Path(__file__).resolve().with_name('build_refinement_pt.py'))
    p.add_argument('--ccd-components-file',type=Path,help='CCD used for canonical modified-residue mapping; auto-discovered if omitted')
    p.add_argument('--compare-atol',type=float,default=0.002,help='absolute tolerance for coordinates/transforms/RMSD; other numerical tensors must match exactly')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--progress-seconds',type=float,default=60)
    p.add_argument('--min-observed',type=float,default=0.5)
    p.add_argument('--rmsd-cutoff',type=float,default=30)
    p.add_argument('--limit',type=int,help='Smoke only: max files per pool; cross-checks skipped')
    a=p.parse_args()
    require(a.threads>0 and a.progress_seconds>0 and 0<=a.min_observed<=1 and a.rmsd_cutoff>0,'invalid thresholds')
    require(a.limit is None or a.limit>0,'limit must be positive')
    require(a.compare_atol>=0 and math.isfinite(a.compare_atol),'compare-atol must be finite and nonnegative')
    for key,value in vars(a).items():
        if isinstance(value,Path): setattr(a,key,value.expanduser().resolve())
    roots=[a.current_root,a.archive_root,a.high_root,a.prediction_root,a.native_root,a.rnafm_root]
    require(all(a.output!=r and r not in a.output.parents for r in roots),'output must be outside all input roots')
    global torch
    import torch
    require(hasattr(torch,'__version__'),'torch installation is incomplete')
    torch.empty(0).numpy()  # Fail early on an unavailable/incompatible NumPy bridge.
    torch.set_num_threads(a.threads)
    if a.sources!='none':
        import gemmi
    return Audit(a).run()


if __name__=='__main__':
    sys.exit(main())
