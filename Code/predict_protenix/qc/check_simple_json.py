#!/usr/bin/env python3
"""Read-only audit of raw Protenix JSON against the local pure-RNA CIF snapshot.

Python >=3.9, Gemmi and the SERVER'S installed Protenix are required. No prep,
MSA search, prediction or external network request is performed by this script.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import signal
import sys
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HOME = Path('/storage9920/home/tinghao.xia')
ID_RE = re.compile(r'^[A-Z0-9]{4}$')
RAW_RE = re.compile(r'^([A-Za-z0-9]{4})\.json$', re.IGNORECASE)
ENTITY_TYPES = {'rnaSequence', 'dnaSequence', 'proteinChain', 'ligand'}
PREP_FIELDS = {'unpairedMsaPath', 'pairedMsaPath', 'msa', 'template',
               'templates', 'templatePath'}
LABEL_FIELDS = {'label_entity_id', 'label_asym_id'}
FIELDS = ['PDB_ID', 'STATUS', 'ISSUES', 'WARNINGS', 'CIF_PATH', 'JSON_PATH',
          'CIF_SHA256', 'JSON_SHA256', 'REFERENCE_CORE_SHA256',
          'RAW_RNA_CHAINS', 'REFERENCE_RNA_CHAINS', 'CIF_DECLARED_RNA_CHAINS',
          'CIF_FIRST_MODEL_OBSERVED_RNA_CHAINS', 'RNA_SEQUENCE_COPY_MATCH',
          'ENTITY_ORDER_MATCH', 'ALL_ENTITY_CONTENT_MATCH', 'MODIFICATIONS_MATCH',
          'COVALENT_BONDS_MATCH', 'ASSEMBLY_MATCH', 'CIF_CANONICAL_SEQUENCE_COPY_MATCH',
          'DIFF_PATHS', 'REFERENCE_JSON', 'CONVERTER_LOG', 'CHECKED_AT_UTC']


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stat_key(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def atomic(path, text):
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    try:
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def stable(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def load_json(path):
    def invalid_constant(value):
        raise ValueError(f'invalid JSON numeric constant: {value}')
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique_keys,
                      parse_constant=invalid_constant)


def positive_integer(value):
    return type(value) is int and value > 0


def normalize_task(payload, pdb_id):
    """Return comparable input semantics, keeping the entity list order intact."""
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError('expected exactly one task in a top-level JSON list')
    task = payload[0]
    if not isinstance(task.get('name'), str) or task['name'].strip().upper() != pdb_id:
        raise ValueError('task name does not match CIF/PDB ID')
    entries = task.get('sequences')
    if not isinstance(entries, list) or not entries:
        raise ValueError('missing/empty sequences list')
    warnings, review = [], []
    if set(task) - {'name', 'sequences', 'covalent_bonds', 'assembly_id'}:
        review.append('EXTRA_TASK_FIELDS: ' + ','.join(sorted(set(task) - {
            'name', 'sequences', 'covalent_bonds', 'assembly_id'})))
    entities, rna = [], Counter()
    for index, item in enumerate(entries, 1):
        if not isinstance(item, dict) or len(item) != 1 or not (set(item) <= ENTITY_TYPES):
            raise ValueError(f'entity {index}: expected one recognized entity type')
        kind, body = next(iter(item.items()))
        if not isinstance(body, dict):
            raise ValueError(f'entity {index}: body must be an object')
        count = body.get('count', 1)
        if not positive_integer(count):
            raise ValueError(f'entity {index}: count must be a positive integer (not bool/float/string)')
        if 'count' not in body:
            warnings.append(f'entity {index}: COUNT_DEFAULTS_TO_ONE')
        component = body.get('ligand' if kind == 'ligand' else 'sequence')
        if not isinstance(component, str) or not component:
            raise ValueError(f'entity {index}: missing sequence/ligand')
        if kind != 'ligand' and not re.fullmatch(r'[A-Z]+', component):
            raise ValueError(f'entity {index}: sequence must be uppercase letters without spaces')
        if kind == 'rnaSequence':
            rna[component] += count
            if set(component) - set('ACGU'):
                warnings.append(f'entity {index}: NON_ACGU_PRESENT={"".join(sorted(set(component)-set("ACGU")))}')
        if kind in {'proteinChain', 'dnaSequence'}:
            review.append(f'NON_RNA_POLYMER_IN_PURE_TARGET: entity {index}, {kind}')
        mods = body.get('modifications', [])
        if not isinstance(mods, list):
            raise ValueError(f'entity {index}: modifications must be a list')
        position_field, type_field = ('ptmPosition', 'ptmType') if kind == 'proteinChain' else (
            'basePosition', 'modificationType')
        seen = set()
        for mod in mods:
            if not isinstance(mod, dict):
                raise ValueError(f'entity {index}: invalid modification')
            position = mod.get(position_field)
            if not positive_integer(position) or position > len(component) or position in seen:
                raise ValueError(f'entity {index}: invalid/duplicate modification position')
            seen.add(position)
            if not isinstance(mod.get(type_field), str) or not re.fullmatch(r'CCD_[A-Za-z0-9]+', mod[type_field]):
                raise ValueError(f'entity {index}: invalid modification CCD identifier')
        known = {'sequence', 'ligand', 'count', 'modifications'}
        prep = sorted(set(body) & PREP_FIELDS)
        if prep:
            review.append(f'RAW_CONTAINS_PREP_FIELDS: entity {index}, {prep}')
        extras = set(body) - known - PREP_FIELDS - LABEL_FIELDS
        if extras:
            review.append(f'EXTRA_ENTITY_FIELDS: entity {index}, {sorted(extras)}')
        entities.append({'type': kind, 'component': component, 'count': count,
                         'modifications': sorted(mods, key=stable)})
    if not rna:
        raise ValueError('no RNA entity in raw JSON')
    bonds = task.get('covalent_bonds', [])
    if not isinstance(bonds, list):
        raise ValueError('covalent_bonds must be a list')
    for bond in bonds:
        if not isinstance(bond, dict):
            raise ValueError('invalid covalent bond')
        for end in (1, 2):
            entity = bond.get(f'entity{end}')
            position = bond.get(f'position{end}')
            if not positive_integer(entity) or entity > len(entities) or not positive_integer(position):
                raise ValueError('invalid bond entity/position reference')
            entry = entities[entity-1]
            if entry['type'] != 'ligand' and position > len(entry['component']):
                raise ValueError('bond residue position is outside entity sequence')
            if not isinstance(bond.get(f'atom{end}'), str) or not bond[f'atom{end}']:
                raise ValueError('bond atom name is missing')
            copy = bond.get(f'copy{end}', 1)
            if not positive_integer(copy) or copy > entry['count']:
                raise ValueError('invalid bond copy reference')
    core = {'entities': entities, 'covalent_bonds': sorted(bonds, key=stable),
            'assembly_id': task.get('assembly_id')}
    return core, rna, warnings, review


def cif_inventory(path, pdb_id):
    """Independent declared sequences/copies; missing coordinates do not truncate sequences."""
    from gemmi import cif
    document = cif.read_file(str(path), check_level=2)
    if len(document) != 1:
        raise ValueError('CIF must have one data block')
    block = document[0]
    entry = block.find_value('_entry.id')
    if not entry or cif.as_string(entry).upper() != pdb_id:
        raise ValueError('CIF entry ID mismatch')
    poly = block.get_mmcif_category('_entity_poly.')
    asym = block.get_mmcif_category('_struct_asym.')
    ids, types = poly.get('entity_id', []), poly.get('type', [])
    if len(ids) != len(types) or not ids:
        raise ValueError('missing/inconsistent _entity_poly')
    rna_ids = {entity for entity, kind in zip(ids, types) if kind == 'polyribonucleotide'}
    if not rna_ids or any(kind != 'polyribonucleotide' for kind in types):
        raise ValueError('expected exclusively RNA polymers (nonpolymer ligands are allowed)')
    chains = Counter(entity for entity in asym.get('entity_id', []) if entity in rna_ids)
    if any(not chains[x] for x in rna_ids):
        raise ValueError('RNA entity has no _struct_asym chain mapping')
    canonical = poly.get('pdbx_seq_one_letter_code_can', [None] * len(ids))
    if len(canonical) != len(ids):
        raise ValueError('inconsistent canonical RNA sequence column')
    sequences = Counter()
    for entity, seq in zip(ids, canonical):
        if entity in rna_ids:
            if not isinstance(seq, str) or not re.fullmatch(r'[A-Za-z\s]+', seq):
                raise ValueError('canonical RNA sequence unavailable; requires manual CCD-aware review')
            sequences[re.sub(r'\s+', '', seq).upper()] += chains[entity]
    atom_entities = block.find_values('_atom_site.label_entity_id')
    atom_chains = block.find_values('_atom_site.label_asym_id')
    models = block.find_values('_atom_site.pdbx_PDB_model_num')
    if not atom_entities or len(atom_entities) != len(atom_chains) or (models and len(models) != len(atom_entities)):
        raise ValueError('missing/inconsistent atom-site entity, chain or model columns')
    observed = set()
    first_model = models[0] if models else None
    for i, (entity, chain) in enumerate(zip(atom_entities, atom_chains)):
        if entity in rna_ids and (first_model is None or models[i] == first_model):
            observed.add(chain)
    return sequences, sum(chains.values()), len(observed)


def difference_paths(a, b, prefix='', limit=40):
    result = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                result.append(f'{prefix}/{key}')
            else:
                result += difference_paths(a[key], b[key], f'{prefix}/{key}', limit)
            if len(result) >= limit:
                break
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            result.append(prefix + '/length')
        for i, (left, right) in enumerate(zip(a, b)):
            result += difference_paths(left, right, f'{prefix}/{i}', limit)
            if len(result) >= limit:
                break
    elif a != b:
        result.append(prefix)
    return result[:limit]


def audit_one(pdb_id, path, json_paths, converter, report):
    row = dict.fromkeys(FIELDS, '')
    row.update(PDB_ID=pdb_id, STATUS='PASS', CIF_PATH=str(path))
    problems, reviews, warnings = [], [], []
    before_cif = stat_key(path)
    row['CIF_SHA256'] = sha(path)
    local = None
    if len(json_paths) != 1:
        problems.append('MISSING_RAW_JSON' if not json_paths else 'DUPLICATE_RAW_JSON')
    else:
        local = json_paths[0]
        row['JSON_PATH'] = str(local)
        before_json = stat_key(local)
        row['JSON_SHA256'] = sha(local)
    raw_core = raw_rna = None
    if local:
        try:
            raw_core, raw_rna, w, r = normalize_task(load_json(local), pdb_id)
            warnings += w
            reviews += r
            row['RAW_RNA_CHAINS'] = sum(raw_rna.values())
        except Exception as error:
            problems.append(f'INVALID_RAW_JSON: {type(error).__name__}: {error}')
    try:
        canonical, declared, observed = cif_inventory(path, pdb_id)
        row.update(CIF_DECLARED_RNA_CHAINS=declared,
                   CIF_FIRST_MODEL_OBSERVED_RNA_CHAINS=observed)
        if raw_rna is not None:
            row['CIF_CANONICAL_SEQUENCE_COPY_MATCH'] = raw_rna == canonical
            if raw_rna != canonical:
                reviews.append('CIF_CANONICAL_SEQUENCE_OR_CHAIN_COPY_REVIEW')
        if declared != observed:
            warnings.append('CIF_HAS_CHAINS_WITHOUT_FIRST_MODEL_COORDINATES')
    except Exception as error:
        reviews.append(f'CIF_INDEPENDENT_CHECK_REVIEW: {type(error).__name__}: {error}')
    converter_log = report / 'converter_logs' / f'{pdb_id}.log'
    row['CONVERTER_LOG'] = str(converter_log)
    reference = None
    try:
        with tempfile.TemporaryDirectory(prefix='reference_', dir=report) as temp:
            output = Path(temp) / 'reference.json'
            with converter_log.open('w', encoding='utf-8') as f, \
                    contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                converted = converter(str(path), output_json=str(output))
            reference = load_json(output) if output.exists() else [converted] if isinstance(converted, dict) else converted
            ref_core, ref_rna, _, ref_review = normalize_task(reference, pdb_id)
            if ref_review:
                reviews.append('GENERATED_REFERENCE_SCHEMA_REVIEW: ' + '; '.join(ref_review))
            row['REFERENCE_RNA_CHAINS'] = sum(ref_rna.values())
            row['REFERENCE_CORE_SHA256'] = hashlib.sha256(stable(ref_core).encode()).hexdigest()
            if raw_core is not None:
                row.update(
                    RNA_SEQUENCE_COPY_MATCH=raw_rna == ref_rna,
                    ENTITY_ORDER_MATCH=raw_core['entities'] == ref_core['entities'],
                    ALL_ENTITY_CONTENT_MATCH=sorted(raw_core['entities'], key=stable) == sorted(ref_core['entities'], key=stable),
                    MODIFICATIONS_MATCH=[x['modifications'] for x in raw_core['entities']] == [x['modifications'] for x in ref_core['entities']],
                    COVALENT_BONDS_MATCH=raw_core['covalent_bonds'] == ref_core['covalent_bonds'],
                    ASSEMBLY_MATCH=raw_core['assembly_id'] == ref_core['assembly_id'],
                )
                if raw_core != ref_core:
                    problems.append('DIFFERENT_FROM_INSTALLED_PROTENIX_CONVERTER')
                    row['DIFF_PATHS'] = ';'.join(difference_paths(raw_core, ref_core))
    except Exception as error:
        reviews.append(f'REFERENCE_GENERATION_FAILED: {type(error).__name__}: {error}')
    if stat_key(path) != before_cif or (local and stat_key(local) != before_json):
        problems.append('SOURCE_CHANGED_DURING_CHECK')
    row['STATUS'] = 'PROBLEM' if problems else 'REVIEW' if reviews else 'PASS'
    if reference is not None and row['STATUS'] != 'PASS':
        saved = report / 'reference_json' / f'{pdb_id}.json'
        atomic(saved, json.dumps(reference, ensure_ascii=False, indent=2) + '\n')
        row['REFERENCE_JSON'] = str(saved)
    row.update(ISSUES='; '.join(problems + reviews), WARNINGS='; '.join(warnings), CHECKED_AT_UTC=now())
    return row


def provenance(converter):
    import protenix
    root = Path(protenix.__file__).resolve().parent
    source = Path(inspect.getsourcefile(converter)).resolve()
    paths = [source, root / 'data/core/parser.py', root / 'data/core/ccd.py', root / 'data/constants.py']
    data = {'protenix_version': importlib.metadata.version('protenix'),
            'python': sys.executable, 'source_files': {},
            'PROTENIX_ROOT_DIR': os.environ.get('PROTENIX_ROOT_DIR'),
            'converter_signature': str(inspect.signature(converter))}
    for path in paths:
        if path.is_file():
            data['source_files'][str(path)] = sha(path)
    common = Path(os.environ['PROTENIX_ROOT_DIR']) / 'common'
    data['ccd_files'] = {}
    for path in (common / 'components.cif', common / 'components.cif.rdkit_mol.pkl'):
        if path.is_file():
            s = path.stat()
            data['ccd_files'][str(path)] = {'size': s.st_size, 'mtime_ns': s.st_mtime_ns}
    return data


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cif-dir', type=Path, default=HOME / 'pdb_data')
    p.add_argument('--json-dir', type=Path, default=HOME / 'Json_data/Simple_json')
    p.add_argument('--expected-count', type=int, default=2241)
    p.add_argument('--report-dir', type=Path,
                   default=HOME / ('simple_json_audit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    p.add_argument('--expected-protenix-version', default='1.0.5')
    args = p.parse_args(argv)
    if args.expected_count < 1:
        p.error('--expected-count must be positive')
    for key in ('cif_dir', 'json_dir', 'report_dir'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if not args.cif_dir.is_dir() or not args.json_dir.is_dir():
        p.error('CIF and Simple_json input directories must exist')
    if args.report_dir == args.cif_dir or args.report_dir == args.json_dir:
        p.error('report directory must be separate from source directories')
    if args.report_dir.exists():
        p.error('report directory already exists; choose a new --report-dir')
    args.report_dir.mkdir(parents=True)
    report = args.report_dir
    for name in ('converter_logs', 'reference_json'):
        (report / name).mkdir()
    console = sys.stdout

    def log(text):
        print(f'[{now()}] {text}', file=console, flush=True)

    files = sorted(x for x in args.cif_dir.iterdir() if x.is_file() and x.suffix.lower() == '.cif')
    cifs, raw = {}, {}
    inventory_issues = []
    if len(files) != args.expected_count:
        inventory_issues.append(f'expected {args.expected_count} CIFs, found {len(files)}')
    for path in files:
        identifier = path.stem.upper()
        if not ID_RE.fullmatch(identifier) or identifier in cifs:
            inventory_issues.append('invalid/duplicate CIF identifier: ' + path.name)
        else:
            cifs[identifier] = path
    ignored = []
    for path in sorted(args.json_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != '.json':
            continue
        match = RAW_RE.fullmatch(path.name)
        if not match:
            ignored.append(path.name)
            continue
        raw.setdefault(match.group(1).upper(), []).append(path)
    extra = sorted(set(raw) - set(cifs))
    atomic(report / 'ignored_non_raw_json.txt', ''.join(x + '\n' for x in ignored))
    atomic(report / 'extra_raw_json_ids.txt', ''.join(x + '\n' for x in extra))
    results = {}
    stop = threading.Event()
    done = threading.Event()
    active = {'pdb_id': 'initializing converter'}
    started = now()
    meta = {}

    def checkpoint(status, fatal=''):
        counts = Counter(x['STATUS'] for x in results.values())
        unresolved = sorted(x for x in cifs if x not in results or results[x]['STATUS'] != 'PASS')
        atomic(report / 'needs_review_pdb_ids.txt', ''.join(x + '\n' for x in unresolved))
        atomic(report / 'summary.json', json.dumps({
            'status': status, 'started_at_utc': started, 'updated_at_utc': now(),
            'expected_cifs': args.expected_count, 'found_cifs': len(files),
            'checked_pdbs': len(results), 'unchecked_pdbs': len(cifs) - len(results),
            'counts': dict(counts), 'inventory_issues': inventory_issues,
            'extra_raw_json_ids': extra, 'ignored_non_raw_json_count': len(ignored),
            'all_targets_passed': status == 'COMPLETED' and not inventory_issues
                and len(results) == len(cifs) == args.expected_count and counts['PASS'] == len(cifs),
            'fatal_error': fatal, 'report_dir': str(report), 'provenance': meta,
            'scope': 'raw JSON only; includes RNA copies, modifications, ligands and covalent bonds; '
                'independent CIF canonical sequences/copies plus installed converter reproduction. '
                'No prep/MSA/GPU inference. Model input compatibility is not certified by this audit.',
        }, ensure_ascii=False, indent=2) + '\n')

    def heartbeat():
        while not done.wait(30):
            log(f'Heartbeat: checked={len(results)}/{len(cifs)}, active={active["pdb_id"]}, stop={stop.is_set()}')

    handlers = {sig: signal.signal(sig, lambda signum, frame: stop.set())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    threading.Thread(target=heartbeat, daemon=True).start()
    checkpoint('RUNNING')
    log(f'REPORT_DIR={report}; CIFs={len(files)}; raw JSON IDs={len(raw)}; ignored prep/other JSON={len(ignored)}')
    fatal = ''
    try:
        version = importlib.metadata.version('protenix')
        if version != args.expected_protenix_version:
            raise RuntimeError(f'installed Protenix={version}; expected={args.expected_protenix_version}')
        os.environ.setdefault('PROTENIX_ROOT_DIR', str(HOME / 'protenix_data'))
        with (report / 'converter_import.log').open('w', encoding='utf-8') as f, \
                contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            from protenix.data.inference.json_maker import cif_to_input_json
            import gemmi
        meta = provenance(cif_to_input_json)
        meta['gemmi_version'] = gemmi.__version__
        checkpoint('RUNNING')
        if inventory_issues:
            raise ValueError('; '.join(inventory_issues))
        with (report / 'per_pdb.tsv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter='\t')
            writer.writeheader()
            f.flush()
            for pdb_id, path in sorted(cifs.items()):
                if stop.is_set():
                    break
                active['pdb_id'] = pdb_id
                try:
                    row = audit_one(pdb_id, path, raw.get(pdb_id, []), cif_to_input_json, report)
                except Exception as error:
                    row = dict.fromkeys(FIELDS, '')
                    row.update(PDB_ID=pdb_id, STATUS='REVIEW', CIF_PATH=str(path),
                               ISSUES=f'CHECK_FAILED: {type(error).__name__}: {error}', CHECKED_AT_UTC=now())
                results[pdb_id] = row
                writer.writerow(row)
                f.flush()
                checkpoint('RUNNING')
                log(f'{len(results)}/{len(cifs)} {pdb_id} {row["STATUS"]} {row["ISSUES"]}')
    except Exception as error:
        fatal = f'{type(error).__name__}: {error}'
        log('FATAL: ' + fatal)
    finally:
        done.set()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    status = 'FAILED' if fatal else 'INTERRUPTED' if stop.is_set() else 'COMPLETED'
    checkpoint(status, fatal)
    log(f'FINAL STATUS={status}; report={report}')
    return 2 if fatal else 130 if stop.is_set() else 1 if any(x['STATUS'] != 'PASS' for x in results.values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
