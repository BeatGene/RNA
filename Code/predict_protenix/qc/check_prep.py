#!/usr/bin/env python3
"""Read-only audit of existing pure-RNA Protenix prep (no search or prediction).

Defaults: 2241 CIF/raw-JSON targets, 2227 final JSONs and prep directories.
Missing final JSONs are listed separately; their abandonment reason is NOT inferred.
Standalone Python >=3.9 script. Uses the active Protenix 1.0.5 MSA parser.
Stale JSON paths are diagnosed separately from verified local MSA content.
Relocation proposals are written to the report only; input files are never changed.
Also audits the installed prediction wrapper's actual MSA selection (no prediction).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
import re
import signal
import string
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HOME = Path('/storage9920/home/tinghao.xia')
ID_RE = re.compile(r'^[A-Z0-9]{4}$')
RAW_RE = re.compile(r'^([A-Z0-9]{4})\.json$', re.I)
FINAL_RE = re.compile(r'^([A-Z0-9]{4})-final-updated\.json$', re.I)
PREP_RE = re.compile(r'^prep_output_([A-Z0-9]{4})$', re.I)
PREP_KEYS = {'unpairedMsaPath', 'unpairedMsa', 'pairedMsaPath', 'pairedMsa',
             'templatesPath', 'templatePath', 'templates', 'msa'}
ASCII_SEQ = set(string.ascii_letters + '-.')
PDB_FIELDS = ['PDB_ID', 'STATUS', 'CONTENT_STATUS', 'CONTENT_ISSUES', 'FINAL_MSA_PATHS_READABLE', 'ISSUES', 'WARNINGS', 'RAW_JSON', 'FINAL_JSON',
              'RUNTIME_MAPPING_STATUS', 'RUNTIME_MAPPING_ISSUES',
              'RAW_SHA256', 'FINAL_SHA256', 'PREP_DIR', 'INPUT_CONTENT_MATCH',
              'DIFF_PATHS', 'RNA_ENTITIES', 'RNA_COPIES', 'MSA_FILES_IN_PREP',
              'RNA_PASS', 'RNA_PROBLEM', 'RNA_REVIEW', 'CHECKED_AT_UTC']
RNA_FIELDS = ['PDB_ID', 'ENTITY_INDEX', 'RNA_INDEX', 'STATUS', 'CONTENT_STATUS', 'CONTENT_ISSUES', 'ISSUES', 'WARNINGS',
              'SEQUENCE_LENGTH', 'COUNT', 'SEQUENCE_SHA256', 'MSA_PATH',
              'RECORDED_PATH_READABLE', 'RECORDED_PATH_ERROR', 'RELOCATION_STATUS',
              'INLINE_MSA_OVERRIDES_PATH',
              'PROPOSED_MSA_PATH', 'RELOCATION_CANDIDATES',
              'MSA_RESOLVED_PATH', 'PATH_IS_SYMLINK', 'MSA_SHA256', 'MSA_BYTES',
              'MSA_DEPTH', 'UNIQUE_ALIGNED_ROWS', 'QUERY_HEADER', 'QUERY_CORE',
              'QUERY_MATCH', 'QUERY_TU_ONLY_MATCH', 'MIN_COLUMNS', 'MAX_COLUMNS',
              'OFFICIAL_ROWS_CHECKED', 'OFFICIAL_PARSER_PASS', 'DATABASE_SET_ID',
              'CACHE_DIGEST_MATCH', 'COMPAT_PATHS', 'COMPAT_SAME_CONTENT']
MSA_FIELDS = ['RESOLVED_PATH', 'PATH_READABLE', 'FILE_BYTES', 'SHA256', 'STATUS', 'ISSUES', 'WARNINGS',
              'DEPTH', 'UNIQUE_ALIGNED_ROWS', 'QUERY_HEADER', 'QUERY_CORE', 'MIN_COLUMNS',
              'MAX_COLUMNS', 'OFFICIAL_ROWS_CHECKED', 'OFFICIAL_PARSER_PASS', 'SECONDS']
LINK_FIELDS = ['PDB_ID', 'PATH', 'IS_SYMLINK', 'RESOLVED_PATH', 'FILE_STATUS',
               'SHA256', 'QUERY_SHA256', 'QUERY_IN_RAW', 'ISSUES']
RELOCATION_FIELDS = ['PDB_ID', 'ENTITY_INDEX', 'RNA_INDEX', 'FINAL_JSON', 'FINAL_SHA256',
                     'OLD_MSA_PATH', 'PROPOSED_MSA_PATH', 'MSA_SHA256', 'CONTENT_STATUS',
                     'PROPOSAL_STATUS', 'CANDIDATES', 'EVIDENCE']
RUNTIME_FIELDS = ['PDB_ID', 'ENTITY_INDEX', 'RNA_INDEX', 'STATUS', 'ISSUES',
                  'PIPELINE_PREP_STATUS', 'RUNTIME_MSA_PATH', 'RESOLVED_PATH',
                  'MSA_SHA256', 'QUERY_MATCH', 'FULL_MSA_STATUS']
SCRIPT_VERSION = '3.0-prediction-mapping'


class AuditInterrupted(Exception):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def stable(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def signature(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def atomic(path, text):
    temp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    try:
        temp.write_text(text, encoding='utf-8')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError('nonstandard JSON constant: ' + value)


def load_json(path):
    data = path.read_bytes()
    value = json.loads(data.decode('utf-8-sig'), object_pairs_hook=json_pairs,
                       parse_constant=invalid_constant)
    return value, hashlib.sha256(data).hexdigest()


def status(problems, reviews):
    return 'PROBLEM' if problems else 'REVIEW' if reviews else 'PASS'


def add_issue(items, message):
    if message not in items and len(items) < 25:
        items.append(message)


def core_task(payload, pdb_id, raw=False):
    """Strip only known prep fields; keep entity order and all chemical content."""
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError('expected exactly one task in a JSON list')
    task = copy.deepcopy(payload[0])
    if not isinstance(task.get('name'), str) or task['name'].upper() != pdb_id:
        raise ValueError('task name does not match PDB ID')
    task['name'] = pdb_id
    if not isinstance(task.get('sequences'), list) or not task['sequences']:
        raise ValueError('missing/empty sequences')
    rnas, reviews = [], []
    for index, item in enumerate(task['sequences']):
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f'entity {index}: expected one entity type')
        kind, body = next(iter(item.items()))
        if kind not in {'rnaSequence', 'dnaSequence', 'proteinChain', 'ligand'} or not isinstance(body, dict):
            raise ValueError(f'entity {index}: unsupported entity')
        count = body.setdefault('count', 1)
        if type(count) is not int or count <= 0:
            raise ValueError(f'entity {index}: count must be a positive integer')
        value = body.get('ligand' if kind == 'ligand' else 'sequence')
        if not isinstance(value, str) or not value:
            raise ValueError(f'entity {index}: missing sequence/ligand')
        if kind != 'ligand' and not re.fullmatch(r'[A-Z]+', value):
            raise ValueError(f'entity {index}: invalid sequence spelling')
        if kind == 'rnaSequence':
            rnas.append((index, value, count, payload[0]['sequences'][index][kind]))
            for key in PREP_KEYS:
                present = key in body
                removed = body.pop(key, None)
                if raw and present:
                    reviews.append(f'RAW_HAS_PREP_FIELD: entity={index}, {key}')
                elif key not in {'unpairedMsaPath', 'unpairedMsa'} and removed not in (None, '', [], {}):
                    reviews.append(f'UNSUPPORTED_RNA_PREP_FIELD_NEEDS_REVIEW: entity={index}, {key}')
        elif kind in {'proteinChain', 'dnaSequence'}:
            reviews.append(f'NON_RNA_POLYMER_IN_PURE_TARGET: entity={index}')
        mods = body.setdefault('modifications', []) if kind != 'ligand' else body.get('modifications', [])
        if not isinstance(mods, list):
            raise ValueError('modifications must be a list')
        seen = set()
        for mod in mods:
            if not isinstance(mod, dict):
                raise ValueError('invalid modification object')
            pos = mod.get('ptmPosition' if kind == 'proteinChain' else 'basePosition')
            code = mod.get('ptmType' if kind == 'proteinChain' else 'modificationType')
            if type(pos) is not int or not 1 <= pos <= len(value) or pos in seen:
                raise ValueError('invalid/duplicate modification position')
            if not isinstance(code, str) or not re.fullmatch(r'CCD_[A-Za-z0-9]+', code):
                raise ValueError('invalid modification CCD identifier')
            seen.add(pos)
        if 'modifications' in body:
            body['modifications'] = sorted(mods, key=stable)
    if not rnas:
        raise ValueError('no RNA entries')
    bonds = task.setdefault('covalent_bonds', [])
    if not isinstance(bonds, list) or any(not isinstance(x, dict) for x in bonds):
        raise ValueError('invalid covalent_bonds')
    task['covalent_bonds'] = sorted(bonds, key=stable)
    task.setdefault('assembly_id', None)
    return task, rnas, reviews


def differences(a, b, prefix='', limit=30):
    result = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            result += [prefix+'/'+key] if key not in a or key not in b else differences(a[key], b[key], prefix+'/'+key)
            if len(result) >= limit:
                break
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            result.append(prefix+'/length')
        for i, (left, right) in enumerate(zip(a, b)):
            result += differences(left, right, prefix+f'/{i}')
            if len(result) >= limit:
                break
    elif type(a) is not type(b) or a != b:
        result.append(prefix)
    return result[:limit]


class OfficialReader:
    """Use installed 1.0.5 parsing/encoding in bounded batches, without model loading."""
    def __init__(self):
        import protenix
        from protenix.data.constants import RNA_CHAIN, MSA_RNA_SEQ_TO_ID
        from protenix.data.msa.msa_utils import RawMsa, MSACore
        self.RawMsa, self.MSACore = RawMsa, MSACore
        self.chain, self.mapping = RNA_CHAIN, dict(MSA_RNA_SEQ_TO_ID)
        if not set('ACGUN-') <= set(self.mapping):
            raise RuntimeError('unexpected installed RNA alphabet mapping')
        root = Path(protenix.__file__).resolve().parent
        self.provenance = {'protenix_version': importlib.metadata.version('protenix'),
            'python': sys.executable, 'rna_mapping': self.mapping, 'source_files': {},
            'batch_validation': 'RawMsa.from_a3m(dedup=False) + MSACore.sequences_to_array; all rows',
            'full_structure_featurization_tested': False}
        for path in [Path(inspect.getsourcefile(RawMsa)), root/'data/constants.py',
                     root/'data/msa/msa_featurizer.py', root/'data/tools/common.py']:
            if path.is_file():
                self.provenance['source_files'][str(path)] = sha_file(path)

    def check_batch(self, query, records):
        text = ''.join(f'>{head}\n{seq}\n' for head, seq in records)
        parsed = self.RawMsa.from_a3m(query=query, ctype=self.chain, a3m=text, dedup=False)
        expected = [seq for _, seq in records]
        if list(parsed.seqs) != expected:
            raise ValueError('installed parser changed/dropped input rows')
        arr, deletion = self.MSACore.sequences_to_array(parsed.seqs, self.chain)
        if tuple(arr.shape) != (len(records), len(query)) or deletion.shape != arr.shape:
            raise ValueError('unexpected installed encoder matrix shape')


def scan_a3m(stream, reader, stop, active):
    """Stream every byte/record, independent strict lengths + installed parser checks."""
    result = dict.fromkeys(MSA_FIELDS, '')
    problems, reviews, warnings = [], [], []
    digest, unique_rows = hashlib.sha256(), set()
    depth = checked = size = 0
    min_cols = max_cols = None
    header, pieces, query_core, batch = None, [], '', []
    hit_nonstandard = Counter()
    comment_count = 0
    started = datetime.now(timezone.utc)
    last_byte = b''

    def flush_batch():
        nonlocal checked
        if batch:
            try:
                reader.check_batch(query_core, batch)
                checked += len(batch)
            except Exception as error:
                add_issue(problems, f'OFFICIAL_PARSER_FAILED: {type(error).__name__}: {error}')
            batch.clear()

    def finish_record():
        nonlocal depth, query_core, min_cols, max_cols
        if header is None:
            return
        depth += 1
        seq = ''.join(pieces)
        if not seq:
            add_issue(problems, f'EMPTY_RECORD: row={depth}')
            return
        bad = set(seq) - ASCII_SEQ
        if bad:
            add_issue(problems, f'INVALID_SEQUENCE_CHARACTERS: row={depth}, {sorted(bad)}')
            return
        aligned = ''.join(c for c in seq if c in string.ascii_uppercase or c == '-')
        cols = len(aligned)
        min_cols = cols if min_cols is None else min(min_cols, cols)
        max_cols = cols if max_cols is None else max(max_cols, cols)
        unique_rows.add(hashlib.sha256(aligned.encode('ascii')).digest())
        if depth == 1:
            query_core = aligned
            result.update(QUERY_HEADER=header, QUERY_CORE=query_core)
            if not query_core or '-' in query_core:
                add_issue(problems, 'QUERY_EMPTY_OR_HAS_ALIGNMENT_GAPS')
            if any(c in string.ascii_lowercase or c == '.' for c in seq):
                add_issue(reviews, 'NONCANONICAL_QUERY_INSERTION_FORMAT')
        if not query_core or cols != len(query_core):
            add_issue(problems, f'ALIGNMENT_COLUMN_MISMATCH: row={depth}, columns={cols}, expected={len(query_core)}')
            return
        if any(c not in reader.mapping for c in aligned):
            add_issue(problems, f'CHAR_NOT_IN_INSTALLED_RNA_MAPPING: row={depth}')
            return
        if depth > 1:
            hit_nonstandard.update(c for c in aligned if c not in 'ACGUN-')
        batch.append((header, seq))
        if len(batch) >= min(128, max(1, 250000 // max(1, len(query_core)))):
            flush_batch()
        if depth % 1000 == 0:
            active['msa_rows'] = depth

    try:
        for line_number, raw_line in enumerate(stream, 1):
            if stop.is_set():
                raise AuditInterrupted()
            digest.update(raw_line)
            size += len(raw_line)
            last_byte = raw_line[-1:]
            text = raw_line.decode('utf-8', errors='strict').strip()
            if not text:
                continue
            if text.startswith('#'):
                comment_count += 1
                continue
            if text.startswith('>'):
                finish_record()
                header, pieces = text[1:], []
                if not header:
                    add_issue(problems, f'EMPTY_RECORD_HEADER: line={line_number}')
            elif header is None:
                add_issue(problems, f'SEQUENCE_BEFORE_HEADER: line={line_number}')
            else:
                pieces.append(text)
        finish_record()
        flush_batch()
        result['SHA256'] = digest.hexdigest()
        if not depth:
            add_issue(problems, 'NO_FASTA_RECORDS')
        if checked != depth:
            add_issue(problems, f'NOT_ALL_ROWS_ACCEPTED_BY_INSTALLED_PARSER: checked={checked}, rows={depth}')
        if depth == 1:
            warnings.append('QUERY_ONLY_MSA_NO_HOMOLOGOUS_HITS')
        if size and last_byte not in (b'\n', b'\r'):
            warnings.append('NO_TERMINAL_NEWLINE')
        if comment_count:
            warnings.append(f'COMMENT_LINES_IGNORED_BY_PARSER={comment_count}')
        if hit_nonstandard:
            warnings.append('NON_ACGUN_HIT_LETTERS=' + stable(dict(hit_nonstandard)))
        if hit_nonstandard['T'] and reader.mapping.get('T') != reader.mapping.get('U'):
            reviews.append('HIT_T_IS_NOT_ENCODED_AS_U_BY_INSTALLED_PROTENIX')
    except AuditInterrupted:
        raise
    except Exception as error:
        add_issue(problems, f'A3M_READ_FAILED: {type(error).__name__}: {error}')
    result.update(STATUS=status(problems, reviews), ISSUES='; '.join(problems + reviews),
        WARNINGS='; '.join(warnings), FILE_BYTES=size, DEPTH=depth,
        UNIQUE_ALIGNED_ROWS=len(unique_rows), MIN_COLUMNS=min_cols, MAX_COLUMNS=max_cols,
        OFFICIAL_ROWS_CHECKED=checked, OFFICIAL_PARSER_PASS=bool(depth and checked == depth and not problems),
        SECONDS=round((datetime.now(timezone.utc)-started).total_seconds(), 3))
    return result


class MsaCache:
    def __init__(self, reader, stop, active):
        self.reader, self.stop, self.active = reader, stop, active
        self.records, self.hits = {}, 0

    def inspect(self, path):
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError('MSA path is not a regular file')
            before = signature(resolved)
            key = (str(resolved), before)
            if key in self.records:
                self.hits += 1
                return self.records[key]
            self.active.update(msa_file=str(resolved), msa_rows=0)
            with resolved.open('rb') as stream:
                result = scan_a3m(stream, self.reader, self.stop, self.active)
            result['RESOLVED_PATH'] = str(resolved)
            result['PATH_READABLE'] = True
            if signature(resolved) != before or path.resolve(strict=True) != resolved:
                result['STATUS'] = 'PROBLEM'
                result['ISSUES'] += '; SOURCE_CHANGED_DURING_AUDIT'
            self.records[key] = result
            return result
        except AuditInterrupted:
            raise
        except Exception as error:
            result = dict.fromkeys(MSA_FIELDS, '')
            result.update(STATUS='PROBLEM', PATH_READABLE=False,
                          ISSUES=f'MSA_PATH_UNREADABLE: {type(error).__name__}: {error}')
            return result


def source_info(path, expected_sequence):
    parts = Path(path).parts
    result = {'DATABASE_SET_ID': '', 'CACHE_DIGEST_MATCH': ''}
    if 'rna_msa_sequence_cache' in parts:
        index = parts.index('rna_msa_sequence_cache')
        if len(parts) > index + 4 and parts[index+2] == 'entries':
            result['DATABASE_SET_ID'] = parts[index+1]
            expected = sha_text(expected_sequence.replace('T', 'U'))
            result['CACHE_DIGEST_MATCH'] = parts[index+3] == expected
    return result


def prep_files(folder):
    """Follow directory symlinks too, with ancestor inode checks to detect cycles."""
    result = []
    def visit(path, ancestors):
        s = path.stat()
        identity = (s.st_dev, s.st_ino)
        if identity in ancestors:
            raise ValueError('directory symlink cycle: ' + str(path))
        for item in sorted(path.iterdir()):
            if item.name == 'rna_msa.a3m':
                result.append(item)
            elif item.is_dir():
                visit(item, ancestors | {identity})
    visit(folder, set())
    return result


def audit_target(pdb_id, raw_paths, final_paths, prep_paths, cache, args):
    row = dict.fromkeys(PDB_FIELDS, '')
    row['PDB_ID'] = pdb_id
    problems, reviews, warnings, rna_rows, link_rows = [], [], [], [], []
    snapshots = []
    raw_core = final_core = None
    raw_rnas = final_rnas = []
    for label, paths in [('RAW', raw_paths), ('FINAL', final_paths)]:
        if len(paths) != 1:
            problems.append(f'{label}_JSON_COUNT={len(paths)}')
            continue
        path = paths[0]
        row[label+'_JSON'] = str(path)
        try:
            before = signature(path)
            snapshots.append((path, before))
            payload, row[label+'_SHA256'] = load_json(path)
            core, rnas, extra_reviews = core_task(payload, pdb_id, raw=label == 'RAW')
            reviews += extra_reviews
            if label == 'RAW':
                raw_core, raw_rnas = core, rnas
            else:
                final_core, final_rnas = core, rnas
        except Exception as error:
            problems.append(f'{label}_JSON_INVALID: {type(error).__name__}: {error}')
    if raw_core is not None and final_core is not None:
        changed = differences(raw_core, final_core)
        row['INPUT_CONTENT_MATCH'] = not changed
        if changed:
            problems.append('RAW_FINAL_INPUT_CONTENT_CHANGED')
            row['DIFF_PATHS'] = ';'.join(changed)
    content_problems, content_reviews = list(problems), list(reviews)
    row['RNA_ENTITIES'] = len(final_rnas)
    row['RNA_COPIES'] = sum(x[2] for x in final_rnas)
    compat = []
    if len(prep_paths) != 1:
        problems.append(f'PREP_DIR_COUNT={len(prep_paths)}')
        content_reviews.append('PREP_DIRECTORY_CONTENT_UNVERIFIED')
    else:
        row['PREP_DIR'] = str(prep_paths[0])
        try:
            compat = [(path, cache.inspect(path)) for path in prep_files(prep_paths[0])]
            if not compat:
                problems.append('NO_RNA_MSA_IN_PREP_DIR')
                content_reviews.append('NO_PREP_MSA_CONTENT_TO_VERIFY')
        except AuditInterrupted:
            raise
        except Exception as error:
            problems.append(f'PREP_DIRECTORY_SCAN_FAILED: {type(error).__name__}: {error}')
            content_reviews.append('PREP_DIRECTORY_CONTENT_UNVERIFIED')
    row['MSA_FILES_IN_PREP'] = len(compat)
    expected_queries = {x[1] for x in raw_rnas}
    for path, msa in compat:
        readable = msa['PATH_READABLE'] is True
        has_query = readable and bool(msa['QUERY_CORE'])
        is_expected = msa['QUERY_CORE'] in expected_queries if has_query else ''
        link_rows.append(dict(PDB_ID=pdb_id, PATH=str(path), IS_SYMLINK=path.is_symlink(),
            RESOLVED_PATH=msa['RESOLVED_PATH'], FILE_STATUS=msa['STATUS'],
            SHA256=msa['SHA256'], QUERY_SHA256=sha_text(msa['QUERY_CORE']) if has_query else '',
            QUERY_IN_RAW=is_expected, ISSUES=msa['ISSUES']))
        if not readable:
            add_issue(problems, 'PREP_MSA_PATH_UNREADABLE: ' + str(path) + ': ' + msa['ISSUES'])
            add_issue(content_reviews, 'UNREADABLE_PREP_MSA_CONTENT_UNVERIFIED: ' + str(path))
        elif msa['STATUS'] == 'PROBLEM':
            message = 'INVALID_PREP_MSA_CONTENT: ' + str(path) + ': ' + msa['ISSUES']
            add_issue(problems, message)
            add_issue(content_problems, message)
        elif msa['STATUS'] == 'REVIEW':
            message = 'PREP_MSA_CONTENT_NEEDS_REVIEW: ' + str(path) + ': ' + msa['ISSUES']
            add_issue(reviews, message)
            add_issue(content_reviews, message)
        if is_expected is False:
            message = 'PREP_MSA_QUERY_NOT_IN_RAW_JSON: ' + str(path)
            add_issue(reviews, message)
            add_issue(content_reviews, message)
    for rna_index, (entity_index, sequence, count, body) in enumerate(final_rnas):
        entry = dict.fromkeys(RNA_FIELDS, '')
        entry.update(PDB_ID=pdb_id, ENTITY_INDEX=entity_index, RNA_INDEX=rna_index,
                     SEQUENCE_LENGTH=len(sequence), COUNT=count, SEQUENCE_SHA256=sha_text(sequence),
                     INLINE_MSA_OVERRIDES_PATH=body.get('unpairedMsa') is not None,
                     RELOCATION_STATUS='NOT_NEEDED')
        ep, er, ew, cp, cr = [], [], [], [], []
        expected_sequence = next((x[1] for x in raw_rnas if x[0] == entity_index), None)
        text = body.get('unpairedMsaPath')
        msa = None
        candidates = [(p, m) for p, m in compat
                      if m['PATH_READABLE'] is True and m['QUERY_CORE'] == sequence]
        entry['COMPAT_PATHS'] = ';'.join(str(p) for p, _ in candidates)
        if not isinstance(text, str) or not text.strip():
            ep.append('MISSING_UNPAIRED_MSA_PATH')
            cr.append('MSA_CONTENT_UNVERIFIED_NO_RECORDED_PATH')
            entry.update(RECORDED_PATH_READABLE=False, RELOCATION_STATUS='MISSING_PATH')
        else:
            entry['MSA_PATH'] = text
            path = Path(text)
            if not path.is_absolute():
                er.append('RELATIVE_MSA_PATH_DEPENDS_ON_PREDICTION_WORKING_DIRECTORY')
                path = Path.cwd()/path
            # Literal open(path), matching Protenix; no '~' or environment expansion.
            entry['PATH_IS_SYMLINK'] = path.is_symlink()
            recorded = cache.inspect(path)
            entry['RECORDED_PATH_READABLE'] = recorded['PATH_READABLE'] is True
            if recorded['PATH_READABLE'] is True:
                # A readable but corrupt/wrong-query file is NOT replaced by a better one.
                msa = recorded
            else:
                entry['RECORDED_PATH_ERROR'] = recorded['ISSUES']
                ew.append(recorded['ISSUES'])
                # Only inspect already present files in this PDB's own prep directory.
                # Identical-byte aliases are safe to collapse; different alignments
                # with the same query remain ambiguous and must not be chosen silently.
                options = [(p, m) for p, m in candidates
                           if sequence == expected_sequence and m['STATUS'] != 'PROBLEM'
                           and m['OFFICIAL_PARSER_PASS'] is True and m['SHA256']]
                by_sha = {}
                for p, m in options:
                    by_sha.setdefault(m['SHA256'], []).append((p, m))
                entry['RELOCATION_CANDIDATES'] = stable([
                    {'path': m['RESOLVED_PATH'], 'sha256': m['SHA256']}
                    for _, m in sorted(options, key=lambda x: str(x[0]))])
                if len(by_sha) == 1:
                    _, msa = sorted(next(iter(by_sha.values())), key=lambda x: str(x[0]))[0]
                    entry.update(RELOCATION_STATUS='PROPOSED', PROPOSED_MSA_PATH=msa['RESOLVED_PATH'])
                    er.append('RECORDED_MSA_PATH_UNREADABLE_LOCAL_CONTENT_VERIFIED')
                    ew.append('LOCAL_QUERY_MATCH_DOES_NOT_PROVE_IDENTITY_TO_UNREADABLE_OLD_SERVER_MSA')
                elif len(by_sha) > 1:
                    entry['RELOCATION_STATUS'] = 'AMBIGUOUS'
                    ep.append('RECORDED_MSA_PATH_UNREADABLE_MULTIPLE_DIFFERENT_LOCAL_MSAS')
                    cr.append('RELOCATION_AMBIGUOUS_CONTENT_SELECTION_NOT_VERIFIED')
                else:
                    entry['RELOCATION_STATUS'] = 'UNAVAILABLE'
                    ep.append('RECORDED_MSA_PATH_UNREADABLE_NO_VERIFIED_LOCAL_MATCH')
                    cr.append('MSA_CONTENT_UNVERIFIED_NO_READABLE_UNIQUE_LOCAL_MATCH')
            if msa is not None:
                if msa['STATUS'] == 'PROBLEM':
                    ep.append(msa['ISSUES'])
                    cp.append(msa['ISSUES'])
                elif msa['STATUS'] == 'REVIEW':
                    er.append(msa['ISSUES'])
                    cr.append(msa['ISSUES'])
                ew += [msa['WARNINGS']] if msa['WARNINGS'] else []
                entry.update(MSA_RESOLVED_PATH=msa['RESOLVED_PATH'], MSA_SHA256=msa['SHA256'],
                    MSA_BYTES=msa['FILE_BYTES'], MSA_DEPTH=msa['DEPTH'],
                    UNIQUE_ALIGNED_ROWS=msa['UNIQUE_ALIGNED_ROWS'], QUERY_HEADER=msa['QUERY_HEADER'],
                    QUERY_CORE=msa['QUERY_CORE'], MIN_COLUMNS=msa['MIN_COLUMNS'], MAX_COLUMNS=msa['MAX_COLUMNS'],
                    OFFICIAL_ROWS_CHECKED=msa['OFFICIAL_ROWS_CHECKED'], OFFICIAL_PARSER_PASS=msa['OFFICIAL_PARSER_PASS'])
                # Absence/unreadability is not evidence of sequence/content disagreement.
                if msa['QUERY_CORE']:
                    exact = msa['QUERY_CORE'] == sequence == expected_sequence
                    tu_only = not exact and expected_sequence is not None and msa['QUERY_CORE'].replace('T','U') == sequence.replace('T','U') == expected_sequence.replace('T','U')
                    entry.update(QUERY_MATCH=exact, QUERY_TU_ONLY_MATCH=tu_only)
                    if not exact:
                        message = 'MSA_QUERY_SEQUENCE_MISMATCH' + ('_TU_ONLY_DIFFERENCE' if tu_only else '')
                        ep.append(message)
                        cp.append(message)
                src = source_info(msa['RESOLVED_PATH'], sequence)
                entry.update(src)
                if src['CACHE_DIGEST_MATCH'] is False:
                    er.append('CACHE_DIRECTORY_SEQUENCE_HASH_MISMATCH')
                    cr.append('CACHE_DIRECTORY_SEQUENCE_HASH_MISMATCH')
                if args.expected_database_set_id and src['DATABASE_SET_ID'] and src['DATABASE_SET_ID'] != args.expected_database_set_id:
                    er.append('UNEXPECTED_CACHE_DATABASE_SET_ID')
                    cr.append('UNEXPECTED_CACHE_DATABASE_SET_ID')
                if msa['SHA256']:
                    same = any(m['SHA256'] == msa['SHA256'] for _, m in candidates)
                    entry['COMPAT_SAME_CONTENT'] = same
                    if not candidates:
                        er.append('NO_COMPAT_MSA_WITH_MATCHING_QUERY')
                        cr.append('NO_COMPAT_MSA_WITH_MATCHING_QUERY')
                    elif not same:
                        er.append('FINAL_PATH_AND_PREP_MSA_CONTENT_DIFFER')
                        cr.append('FINAL_PATH_AND_PREP_MSA_CONTENT_DIFFER')
        if body.get('unpairedMsa') is not None:
            er.append('INLINE_MSA_OVERRIDES_PATH_REQUIRES_REVIEW')
            cr.append('INLINE_MSA_OVERRIDES_PATH_REQUIRES_REVIEW')
            inline = body['unpairedMsa']
            if not isinstance(inline, str) or not inline:
                ep.append('EMPTY_OR_INVALID_INLINE_MSA_OVERRIDES_PATH')
                cp.append('EMPTY_OR_INVALID_INLINE_MSA_OVERRIDES_PATH')
            else:
                inline_result = scan_a3m(io.BytesIO(inline.encode('utf-8')), cache.reader, cache.stop, cache.active)
                if inline_result['STATUS'] == 'PROBLEM' or inline_result['QUERY_CORE'] != sequence:
                    message = 'INLINE_MSA_INVALID_OR_QUERY_MISMATCH: ' + inline_result['ISSUES']
                    ep.append(message)
                    cp.append(message)
        entry.update(STATUS=status(ep, er), CONTENT_STATUS=status(cp, cr), CONTENT_ISSUES='; '.join(cp+cr),
                     ISSUES='; '.join(ep+er), WARNINGS='; '.join(ew))
        rna_rows.append(entry)
        if ep:
            add_issue(problems, f'RNA_ENTITY_{entity_index}: ' + entry['ISSUES'])
        elif er:
            add_issue(reviews, f'RNA_ENTITY_{entity_index}: ' + entry['ISSUES'])
        if cp:
            add_issue(content_problems, f'RNA_ENTITY_{entity_index}: ' + '; '.join(cp))
        elif cr:
            add_issue(content_reviews, f'RNA_ENTITY_{entity_index}: ' + '; '.join(cr))
        if ew:
            add_issue(warnings, f'RNA_ENTITY_{entity_index}: ' + '; '.join(ew))
    for path, before in snapshots:
        try:
            if signature(path) != before:
                problems.append('JSON_CHANGED_DURING_AUDIT: ' + str(path))
                content_problems.append('JSON_CHANGED_DURING_AUDIT')
        except OSError:
            problems.append('JSON_DISAPPEARED_DURING_AUDIT: ' + str(path))
            content_problems.append('JSON_DISAPPEARED_DURING_AUDIT')
    counts = Counter(x['STATUS'] for x in rna_rows)
    row.update(STATUS=status(problems, reviews), CONTENT_STATUS=status(content_problems, content_reviews),
        CONTENT_ISSUES='; '.join(content_problems+content_reviews),
        FINAL_MSA_PATHS_READABLE=bool(rna_rows) and all(x['RECORDED_PATH_READABLE'] is True for x in rna_rows),
        ISSUES='; '.join(problems+reviews), WARNINGS='; '.join(warnings), RNA_PASS=counts['PASS'],
        RNA_PROBLEM=counts['PROBLEM'], RNA_REVIEW=counts['REVIEW'], CHECKED_AT_UTC=utc_now())
    return row, rna_rows, link_rows


def relocation_rows(row, details):
    result = []
    for entry in details:
        if entry['RELOCATION_STATUS'] not in {'PROPOSED', 'AMBIGUOUS', 'UNAVAILABLE'}:
            continue
        verified = (entry['RELOCATION_STATUS'] == 'PROPOSED'
                    and row['CONTENT_STATUS'] == 'PASS' and entry['CONTENT_STATUS'] == 'PASS')
        result.append(dict(PDB_ID=row['PDB_ID'], ENTITY_INDEX=entry['ENTITY_INDEX'],
            RNA_INDEX=entry['RNA_INDEX'], FINAL_JSON=row['FINAL_JSON'], FINAL_SHA256=row['FINAL_SHA256'],
            OLD_MSA_PATH=entry['MSA_PATH'], PROPOSED_MSA_PATH=entry['PROPOSED_MSA_PATH'],
            MSA_SHA256=entry['MSA_SHA256'], CONTENT_STATUS=entry['CONTENT_STATUS'],
            PROPOSAL_STATUS='LOCAL_CONTENT_VERIFIED_PROPOSAL' if verified else 'REVIEW_REQUIRED',
            CANDIDATES=entry['RELOCATION_CANDIDATES'],
            EVIDENCE='Local full-row audit + exact raw/final query match + one distinct local MSA SHA256. '
                'Old inaccessible file bytes were NOT compared. No source changes applied.' if verified
                else 'No unambiguous fully verified proposal; no source changes applied.'))
    return result


class PredictionResolver:
    """Load the existing wrapper and call only its read-only inspect_prep()."""
    def __init__(self, explicit_path=None):
        self.module = None
        self.path = None
        self.source_signature = None
        self.provenance = {'available': False, 'source': '', 'sha256': '', 'error': ''}
        paths = [explicit_path] if explicit_path is not None else [
            HOME/'Code/predict_protenix/stage2_decoy_pipeline.py',
            HOME/'Code/predict_protenix/share_predict_core/stage2_decoy_pipeline.py']
        try:
            path = next((Path(x).expanduser().resolve() for x in paths if Path(x).expanduser().is_file()), None)
            if path is None:
                raise FileNotFoundError('prediction wrapper not found; set --prediction-pipeline to the actual stage2_decoy_pipeline.py')
            self.provenance['source'] = str(path)
            before, digest = signature(path), sha_file(path)
            name = '_prep_audit_existing_prediction_wrapper'
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            # Importing the trusted existing wrapper must not create __pycache__.
            old_flag = sys.dont_write_bytecode
            try:
                sys.dont_write_bytecode = True
                spec.loader.exec_module(module)
            finally:
                sys.dont_write_bytecode = old_flag
            if not callable(getattr(module, 'inspect_prep', None)):
                raise ValueError('prediction wrapper does not expose inspect_prep')
            if signature(path) != before or sha_file(path) != digest:
                raise ValueError('prediction wrapper changed while importing')
            self.module, self.path, self.source_signature = module, path, before
            self.provenance.update(available=True, sha256=digest,
                operation='inspect_prep only; no run_pred/runtime JSON writes/search/GPU inference')
        except Exception as error:
            self.provenance['error'] = f'{type(error).__name__}: {error}'

    def resolve(self, pdb_id, final, prep):
        if self.module is None:
            raise RuntimeError(self.provenance['error'])
        if signature(self.path) != self.source_signature:
            raise RuntimeError('prediction wrapper source changed during audit')
        return self.module.inspect_prep(pdb_id, final, prep)


def check_prediction_mapping(row, details, final_paths, prep_paths, cache, resolver):
    """Validate exactly the paths the existing wrapper would put into pred_batches."""
    result, overall_problems, overall_reviews = [], [], []
    resolved, pipeline_status = [], ''
    try:
        if not details or len(final_paths) != 1 or len(prep_paths) != 1:
            raise ValueError('unique final/prep and parsed RNA entries are required')
        info = resolver.resolve(row['PDB_ID'], final_paths[0], prep_paths[0])
        pipeline_status = info.status
        resolved = list(info.resolved_msa_paths)
        if info.status not in {'COMPLETE', 'COMPLETE_REBASABLE'}:
            overall_problems.append('PREDICTOR_PREP_NOT_READY: '+str(info.status))
        if len(resolved) != len(details):
            overall_problems.append('PREDICTOR_MSA_PATH_COUNT_MISMATCH')
    except Exception as error:
        overall_reviews.append(f'PREDICTOR_MAPPING_UNVERIFIED: {type(error).__name__}: {error}')
    for index, entry in enumerate(details):
        out = dict.fromkeys(RUNTIME_FIELDS, '')
        out.update(PDB_ID=row['PDB_ID'], ENTITY_INDEX=entry['ENTITY_INDEX'], RNA_INDEX=entry['RNA_INDEX'],
                   PIPELINE_PREP_STATUS=pipeline_status)
        problems, reviews = list(overall_problems), list(overall_reviews)
        text = resolved[index] if index < len(resolved) else ''
        if not text:
            if not reviews:
                problems.append('PREDICTOR_MSA_PATH_MISSING')
        else:
            out['RUNTIME_MSA_PATH'] = text
            msa = cache.inspect(Path(text))
            out.update(RESOLVED_PATH=msa['RESOLVED_PATH'], MSA_SHA256=msa['SHA256'], FULL_MSA_STATUS=msa['STATUS'])
            if msa['STATUS'] == 'PROBLEM':
                problems.append('PREDICTOR_SELECTED_INVALID_MSA: '+msa['ISSUES'])
            elif msa['STATUS'] != 'PASS':
                reviews.append('PREDICTOR_SELECTED_MSA_NEEDS_REVIEW: '+msa['ISSUES'])
            if msa['QUERY_CORE']:
                matches = (sha_text(msa['QUERY_CORE']) == entry['SEQUENCE_SHA256']
                           and len(msa['QUERY_CORE']) == entry['SEQUENCE_LENGTH'])
                out['QUERY_MATCH'] = matches
                if not matches:
                    problems.append('PREDICTOR_RNA_MSA_QUERY_MISMATCH')
            elif msa['PATH_READABLE'] is not True:
                reviews.append('PREDICTOR_QUERY_UNVERIFIED_UNREADABLE_FILE')
        if row['INPUT_CONTENT_MATCH'] is not True:
            reviews.append('RAW_FINAL_CONTENT_NOT_VERIFIED')
        if entry['INLINE_MSA_OVERRIDES_PATH']:
            reviews.append('INLINE_MSA_OVERRIDE_PATH_MAPPING_ALONE_IS_INSUFFICIENT')
        out.update(STATUS=status(problems, reviews), ISSUES='; '.join(problems+reviews))
        result.append(out)
    # Protect the combined conclusion if input JSONs changed between both checks.
    try:
        for prefix in ('RAW', 'FINAL'):
            if row[prefix+'_JSON'] and sha_file(Path(row[prefix+'_JSON'])) != row[prefix+'_SHA256']:
                row['CONTENT_STATUS'] = 'REVIEW'
                row['CONTENT_ISSUES'] += '; JSON_CHANGED_BETWEEN_CONTENT_AND_RUNTIME_CHECKS'
                overall_reviews.append('JSON_CHANGED_BETWEEN_CONTENT_AND_RUNTIME_CHECKS')
    except OSError as error:
        overall_reviews.append(f'JSON_RECHECK_FAILED: {error}')
        row['CONTENT_STATUS'] = 'REVIEW'
    if any(x['STATUS']=='PROBLEM' for x in result):
        overall_problems.append('PREDICTOR_RUNTIME_MAPPING_FAILED')
    elif any(x['STATUS']!='PASS' for x in result):
        overall_reviews.append('PREDICTOR_RUNTIME_MAPPING_NEEDS_REVIEW')
    if not result:
        overall_reviews.append('NO_RUNTIME_MAPPINGS_VERIFIED')
    row['RUNTIME_MAPPING_STATUS'] = status(overall_problems, overall_reviews)
    row['RUNTIME_MAPPING_ISSUES'] = '; '.join(overall_problems+overall_reviews)
    if row['RUNTIME_MAPPING_STATUS'] == 'PROBLEM':
        row['STATUS'] = 'PROBLEM'
    elif row['RUNTIME_MAPPING_STATUS'] == 'REVIEW' and row['STATUS'] == 'PASS':
        row['STATUS'] = 'REVIEW'
    if row['RUNTIME_MAPPING_STATUS'] != 'PASS':
        row['ISSUES'] += '; '+row['RUNTIME_MAPPING_ISSUES']
    return result


def directory_metadata(root):
    result = {}
    for name in ('common', 'search_database', 'search_database_legacy'):
        folder = root/name
        if folder.is_dir():
            values = {}
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    s = path.stat()
                    values[path.name] = {'size': s.st_size, 'mtime_ns': s.st_mtime_ns}
            result[name] = values
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cif-dir', type=Path, default=HOME/'pdb_data')
    p.add_argument('--json-dir', type=Path, default=HOME/'Json_data/Simple_json')
    p.add_argument('--prep-dir', type=Path, default=HOME/'Json_data/Complex_json')
    p.add_argument('--protenix-data-dir', type=Path, default=Path(os.environ.get('PROTENIX_ROOT_DIR', str(HOME/'protenix_data'))))
    p.add_argument('--expected-total', type=int, default=2241)
    p.add_argument('--expected-prepared', type=int, default=2227)
    p.add_argument('--excluded-ids', type=Path, help='Optional approved skipped PDB IDs, one per line; must equal absent final IDs')
    p.add_argument('--expected-database-set-id', help='Optional requirement for sequence-cache namespace')
    p.add_argument('--expected-protenix-version', default='1.0.5')
    p.add_argument('--prediction-pipeline', type=Path,
                   help='Existing trusted stage2_decoy_pipeline.py actually used for prediction; auto-detects two standard server locations')
    p.add_argument('--report-dir', type=Path, default=HOME/('prep_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    args = p.parse_args(argv)
    if not 0 < args.expected_prepared <= args.expected_total:
        p.error('require 0 < expected-prepared <= expected-total')
    for name in ('cif_dir', 'json_dir', 'prep_dir', 'protenix_data_dir', 'report_dir'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for directory in (args.cif_dir, args.json_dir, args.prep_dir):
        if not directory.is_dir():
            p.error('input directory missing: ' + str(directory))
    if args.report_dir.exists():
        p.error('report directory already exists; choose a new --report-dir')
    if any(parent == source for parent in [args.report_dir]+list(args.report_dir.parents)
           for source in (args.cif_dir, args.json_dir, args.prep_dir, args.protenix_data_dir)):
        p.error('report directory must be outside all input/data directories')
    args.report_dir.mkdir(parents=True)
    report = args.report_dir
    console = sys.stdout
    def log(message):
        print(f'[{utc_now()}] {message}', file=console, flush=True)
    inventory, cifs, raw, final, prep = [], {}, {}, {}, {}
    ignored = []
    for path in sorted(args.cif_dir.iterdir()):
        if path.suffix.lower() != '.cif' or not path.is_file():
            continue
        identifier = path.stem.upper()
        if not ID_RE.fullmatch(identifier) or identifier in cifs:
            inventory.append('INVALID_OR_DUPLICATE_CIF_ID: '+path.name)
        else:
            cifs[identifier] = path
    for path in sorted(args.json_dir.iterdir()):
        if not (path.is_file() or path.is_symlink()):
            continue
        r, f = RAW_RE.fullmatch(path.name), FINAL_RE.fullmatch(path.name)
        if r:
            raw.setdefault(r[1].upper(), []).append(path)
        elif f:
            final.setdefault(f[1].upper(), []).append(path)
        elif path.suffix.lower() == '.json':
            ignored.append(path.name)
    for path in sorted(args.prep_dir.iterdir()):
        m = PREP_RE.fullmatch(path.name)
        if m:
            prep.setdefault(m[1].upper(), []).append(path)
    prepared = sorted(set(cifs) & set(final))
    skipped = sorted(set(cifs)-set(final))
    if len(cifs) != args.expected_total:
        inventory.append(f'EXPECTED_CIF_IDS={args.expected_total}, FOUND={len(cifs)}')
    if len(prepared) != args.expected_prepared:
        inventory.append(f'EXPECTED_PREPARED_IDS={args.expected_prepared}, FOUND={len(prepared)}')
    if len(prep) != args.expected_prepared:
        inventory.append(f'EXPECTED_PREP_DIR_IDS={args.expected_prepared}, FOUND={len(prep)}')
    for label, index in [('RAW', raw), ('FINAL', final), ('PREP_DIR', prep)]:
        extra = sorted(set(index)-set(cifs))
        duplicates = sorted(key for key, paths in index.items() if len(paths) != 1)
        if extra:
            inventory.append(f'EXTRA_{label}_IDS: '+','.join(extra))
        if duplicates:
            inventory.append(f'DUPLICATE_{label}_IDS: '+','.join(duplicates))
    if set(raw) != set(cifs):
        inventory.append('RAW_JSON_TARGET_SET_DIFFERS_FROM_CIF_IDS')
    if set(prep) != set(final):
        inventory.append('PREP_DIR_TARGET_SET_DIFFERS_FROM_FINAL_IDS')
    if args.excluded_ids:
        approved = set()
        for line in args.excluded_ids.read_text(encoding='utf-8-sig').splitlines():
            value = line.strip().upper()
            if not value or value.startswith('#'):
                continue
            if not ID_RE.fullmatch(value):
                p.error('invalid excluded ID: '+value)
            approved.add(value)
        if approved != set(skipped):
            inventory.append('APPROVED_EXCLUSIONS_DIFFER_FROM_ABSENT_FINAL_IDS')
    atomic(report/'skipped_no_final_json_ids.txt', ''.join(x+'\n' for x in skipped))
    atomic(report/'ignored_json_files.txt', ''.join(x+'\n' for x in ignored))
    stop, done = threading.Event(), threading.Event()
    active = {'pdb_id':'initializing', 'msa_file':'', 'msa_rows':0}
    cache, meta, rows, rna_counts, db_counts = None, {}, {}, Counter(), Counter()
    relocation_counts = Counter()
    started = utc_now()
    def checkpoint(state, fatal=''):
        counts = Counter(x['STATUS'] for x in rows.values())
        content_counts = Counter(x['CONTENT_STATUS'] for x in rows.values())
        runtime_counts = Counter(x['RUNTIME_MAPPING_STATUS'] for x in rows.values())
        unchecked = sorted(set(prepared)-set(rows))
        unresolved = sorted(set(unchecked) | {key for key, row in rows.items() if row['STATUS'] != 'PASS'})
        atomic(report/'problem_or_review_pdb_ids.txt', ''.join(x+'\n' for x in unresolved))
        content_unresolved = sorted(set(unchecked) | {key for key, row in rows.items() if row['CONTENT_STATUS'] != 'PASS'})
        atomic(report/'content_problem_or_review_pdb_ids.txt', ''.join(x+'\n' for x in content_unresolved))
        runtime_unresolved = sorted(set(unchecked) | {key for key, row in rows.items() if row['RUNTIME_MAPPING_STATUS'] != 'PASS'})
        atomic(report/'runtime_mapping_problem_or_review_pdb_ids.txt', ''.join(x+'\n' for x in runtime_unresolved))
        summary = {'status':state, 'script_version':SCRIPT_VERSION, 'pid':os.getpid(), 'started_at_utc':started, 'updated_at_utc':utc_now(),
            'expected_total':args.expected_total, 'expected_prepared':args.expected_prepared,
            'expected_skipped':args.expected_total-args.expected_prepared,
            'found_cif_ids':len(cifs), 'found_raw_json_ids':len(raw), 'found_final_json_ids':len(final),
            'found_prep_dir_ids':len(prep), 'checked_prepared_pdbs':len(rows),
            'unchecked_prepared_pdbs':len(unchecked), 'prepared_counts':dict(counts),
            'prepared_content_counts':dict(content_counts), 'rna_relocation_counts':dict(relocation_counts),
            'runtime_mapping_counts':dict(runtime_counts),
            'skipped_no_final_json_count':len(skipped), 'skipped_ids':skipped,
            'skipped_id_list_confirmed':bool(args.excluded_ids) and not any('EXCLUSIONS' in x for x in inventory),
            'skipped_reason':'Absent final JSON only; user reports 14 abandoned long targets. Identity/reason not independently proven.',
            'rna_entry_counts':dict(rna_counts), 'database_set_counts':dict(db_counts),
            'unique_msa_files_checked':len(cache.records) if cache else 0,
            'shared_msa_cache_hits':cache.hits if cache else 0,
            'query_only_unique_msa_files':sum(x['DEPTH']==1 for x in cache.records.values()) if cache else 0,
            'inventory_issues':inventory, 'fatal_error':fatal,
            'all_expected_prepared_passed':state=='COMPLETED' and not inventory and len(rows)==args.expected_prepared and counts['PASS']==len(rows),
            'all_expected_prepared_content_passed':state=='COMPLETED' and not inventory and len(rows)==args.expected_prepared and content_counts['PASS']==len(rows),
            'all_expected_runtime_msa_mappings_passed':state=='COMPLETED' and not inventory and len(rows)==args.expected_prepared
                and runtime_counts['PASS']==len(rows),
            'all_expected_prepared_usable_with_prediction_wrapper':state=='COMPLETED' and not inventory and len(rows)==args.expected_prepared
                and content_counts['PASS']==len(rows) and runtime_counts['PASS']==len(rows),
            'all_final_json_msa_paths_readable':state=='COMPLETED' and not inventory and len(rows)==args.expected_prepared
                and all(row['FINAL_MSA_PATHS_READABLE'] is True for row in rows.values()),
            'content_verified_but_path_relocation_needed_pdbs':sum(row['CONTENT_STATUS']=='PASS'
                and row['FINAL_MSA_PATHS_READABLE'] is False for row in rows.values()),
            'all_2241_have_prep':len(prepared)==args.expected_total and not skipped,
            'report_dir':str(report), 'provenance':meta,
            'scope':'Existing raw/final JSON equivalence, all A3M rows, query identity, actual paths/symlinks and installed MSA parser/encoding. '
                    'No prep, search, network request or GPU inference is invoked. Bounded batches; full atom/MSA assembly not tested. '
                    'For unreadable recorded JSON paths, exact-query local prep files can be audited; '
                    'unique-content relocation proposals are report-only, NOT proof of byte identity to inaccessible old-server files. '
                    'The existing prediction wrapper inspect_prep() is called read-only, and every selected runtime MSA query/content is checked. '
                    'Database metadata/namespace are recorded, not proof of historical search completeness or biological homology quality.',
            'database_provenance_verified':False}
        atomic(report/'summary.json', json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    def heartbeat():
        while not done.wait(30):
            log(f'Heartbeat checked={len(rows)}/{len(prepared)} active={active["pdb_id"]} rows={active["msa_rows"]} msa={active["msa_file"]}')
    handlers = {s:signal.signal(s, lambda signum, frame:stop.set()) for s in (signal.SIGINT, signal.SIGTERM)}
    threading.Thread(target=heartbeat, daemon=True).start()
    checkpoint('RUNNING')
    log(f'VERSION={SCRIPT_VERSION}; REPORT_DIR={report}; prepared={len(prepared)}, no-final={len(skipped)}, inventory_issues={len(inventory)}')
    fatal = ''
    try:
        version = importlib.metadata.version('protenix')
        if version != args.expected_protenix_version:
            raise RuntimeError(f'installed Protenix={version}; expected={args.expected_protenix_version}')
        with (report/'parser_import.log').open('w', encoding='utf-8') as output, contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            reader = OfficialReader()
            resolver = PredictionResolver(args.prediction_pipeline)
        cache = MsaCache(reader, stop, active)
        meta = dict(reader.provenance, protenix_data_dir=str(args.protenix_data_dir),
                    data_files_metadata=directory_metadata(args.protenix_data_dir), script_sha256=sha_file(Path(__file__)),
                    prediction_wrapper=resolver.provenance)
        if not resolver.provenance['available']:
            log('PREDICTION_WRAPPER_UNAVAILABLE: '+resolver.provenance['error']+'; content auditing continues; runtime mapping cannot PASS')
        with contextlib.ExitStack() as stack:
            writers = {}
            for name, fields in [('per_pdb.tsv',PDB_FIELDS), ('per_rna.tsv',RNA_FIELDS), ('prep_links.tsv',LINK_FIELDS),
                                 ('msa_path_relocation_plan.tsv',RELOCATION_FIELDS),
                                 ('runtime_msa_mapping.tsv',RUNTIME_FIELDS)]:
                file = stack.enter_context((report/name).open('w', encoding='utf-8-sig', newline=''))
                writer = csv.DictWriter(file, fieldnames=fields, delimiter='\t')
                writer.writeheader()
                writers[name] = (writer, file)
            for pdb_id in prepared:
                if stop.is_set():
                    break
                active.update(pdb_id=pdb_id, msa_file='', msa_rows=0)
                try:
                    row, details, links = audit_target(pdb_id, raw.get(pdb_id,[]), final[pdb_id], prep.get(pdb_id,[]), cache, args)
                    runtime_rows = check_prediction_mapping(row, details, final[pdb_id], prep.get(pdb_id,[]), cache, resolver)
                except AuditInterrupted:
                    stop.set()
                    break
                except Exception as error:
                    row = dict.fromkeys(PDB_FIELDS, '')
                    row.update(PDB_ID=pdb_id, STATUS='REVIEW', CONTENT_STATUS='REVIEW', FINAL_MSA_PATHS_READABLE=False,
                               RUNTIME_MAPPING_STATUS='REVIEW',
                               ISSUES=f'UNEXPECTED_CHECK_FAILURE: {type(error).__name__}: {error}', CHECKED_AT_UTC=utc_now())
                    details, links, runtime_rows = [], [], []
                rows[pdb_id] = row
                rna_counts.update(x['STATUS'] for x in details)
                db_counts.update(x['DATABASE_SET_ID'] or 'not_a_sequence_cache_path' for x in details)
                relocation_counts.update(x['RELOCATION_STATUS'] for x in details)
                writers['per_pdb.tsv'][0].writerow(row)
                writers['per_rna.tsv'][0].writerows(details)
                writers['prep_links.tsv'][0].writerows(links)
                writers['msa_path_relocation_plan.tsv'][0].writerows(relocation_rows(row, details))
                writers['runtime_msa_mapping.tsv'][0].writerows(runtime_rows)
                for _, file in writers.values():
                    file.flush()
                checkpoint('RUNNING')
                log(f'{len(rows)}/{len(prepared)} {pdb_id} {row["STATUS"]} '
                    f'content={row["CONTENT_STATUS"]} runtime_mapping={row["RUNTIME_MAPPING_STATUS"]} '
                    f'recorded_paths_readable={row["FINAL_MSA_PATHS_READABLE"]} {row["ISSUES"]}')
    except AuditInterrupted:
        stop.set()
    except Exception as error:
        fatal = f'{type(error).__name__}: {error}'
        log('FATAL: '+fatal)
    finally:
        done.set()
        for s, handler in handlers.items():
            signal.signal(s, handler)
    if cache:
        with (report/'unique_msa_files.tsv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=MSA_FIELDS, delimiter='\t')
            writer.writeheader()
            writer.writerows(cache.records.values())
    state = 'FAILED' if fatal else 'INTERRUPTED' if stop.is_set() else 'COMPLETED'
    checkpoint(state, fatal)
    log(f'FINAL STATUS={state}; report={report}')
    usable = (len(rows)==args.expected_prepared
              and all(row['CONTENT_STATUS']=='PASS' and row['RUNTIME_MAPPING_STATUS']=='PASS' for row in rows.values()))
    return 2 if fatal or inventory else 130 if stop.is_set() else 0 if usable else 1


if __name__ == '__main__':
    raise SystemExit(main())
