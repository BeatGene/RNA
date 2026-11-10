#!/usr/bin/env python3
"""Read-only audit of local CIF files against current official PDB archives.

Requires Python >=3.9 and gemmi. Only reports and temporary downloads are
written; source CIFs are never changed. TXT lists mismatches AND unverified IDs.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import http.client
import json
import math
import os
import re
import signal
import sys
import tempfile
import threading
import zlib
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HOME = Path('/storage9920/home/tinghao.xia')
URL_TEMPLATES = {
    'rcsb': 'https://files.rcsb.org/download/{pdb_id}.cif.gz',
    'pdbe': ('https://ftp.ebi.ac.uk/pub/databases/pdb/data/structures/'
             'divided/mmCIF/{middle}/{pdb_id}.cif.gz'),
    'pdbj': ('https://ftp.pdbj.org/pub/pdb/data/structures/'
             'divided/mmCIF/{middle}/{pdb_id}.cif.gz'),
}
ID_PATTERN = re.compile(r'^[A-Z0-9]{4}$')
CHUNK = 1024 * 1024
FIELDS = [
    'DATASET', 'PDB_ID', 'LOCAL_PATH', 'STATUS', 'REASONS',
    'LOCAL_VALID', 'REMOTE_VALID', 'LOCAL_BYTES', 'REMOTE_BYTES',
    'LOCAL_SHA256', 'REMOTE_SHA256', 'LOCAL_ATOMS', 'REMOTE_ATOMS',
    'LOCAL_REVISION_DATE', 'REMOTE_REVISION_DATE', 'SOURCE', 'SOURCE_URL',
    'REFERENCE_MODE', 'REFERENCE_DOWNLOADED_AT_UTC',
    'HTTP_LAST_MODIFIED', 'ATTEMPTS', 'CHECKED_AT_UTC',
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def reference_url(pdb_id: str, source: str) -> str:
    identifier = pdb_id.lower()
    return URL_TEMPLATES[source].format(pdb_id=identifier, middle=identifier[1:3])


def log(message: str) -> None:
    print(f'[{now()}] {message}', flush=True)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def inspect_cif(path: Path, pdb_id: str) -> dict:
    """Syntax/basic metadata/coordinate checks, not stereochemical validation."""
    from gemmi import cif

    result = {'valid': False, 'error': '', 'atoms': 0, 'revision': ''}
    try:
        if path.stat().st_size == 0:
            raise ValueError('empty file')
        document = cif.read_file(str(path), check_level=2)
        if len(document) != 1:
            raise ValueError(f'expected one data block, found {len(document)}')
        block = document[0]
        entry = block.find_value('_entry.id')
        entry_id = cif.as_string(entry).strip().upper() if entry else ''
        if entry_id != pdb_id:
            raise ValueError(f'_entry.id={entry_id!r}, expected {pdb_id}')
        if not block.find_values('_exptl.method'):
            raise ValueError('missing _exptl.method')
        if not block.find_values('_entity_poly.entity_id'):
            raise ValueError('missing _entity_poly.entity_id')
        atoms = block.find_values('_atom_site.id')
        if not atoms:
            raise ValueError('missing atom coordinates')
        result['atoms'] = len(atoms)
        for axis in ('x', 'y', 'z'):
            values = block.find_values(f'_atom_site.Cartn_{axis}')
            if len(values) != len(atoms):
                raise ValueError(f'coordinate column {axis} length mismatch')
            for index, raw in enumerate(values, 1):
                if not math.isfinite(float(raw)):
                    raise ValueError(f'non-finite coordinate {axis}, atom row {index}')
        dates = [cif.as_string(x) for x in block.find_values(
            '_pdbx_audit_revision_history.revision_date'
        )]
        result['revision'] = max((x for x in dates if x not in ('.', '?', '')), default='')
        result['valid'] = True
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result


@dataclass
class Job:
    dataset: str
    pdb_id: str
    path: Path
    flags: list[str]


class DownloadFailure(RuntimeError):
    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


def inventory(args: argparse.Namespace) -> tuple[list[Job], list[str]]:
    jobs: list[Job] = []
    issues: list[str] = []
    for dataset, directory, expected in (
        ('pure', args.pure_dir, args.expected_pure),
        ('mixed', args.mixed_dir, args.expected_mixed),
    ):
        if not directory.is_dir():
            issues.append(f'{dataset}: directory does not exist: {directory}')
            continue
        files = sorted(
            (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == '.cif'),
            key=lambda p: p.name,
        )
        log(f'{dataset}: found {len(files)} CIF files, expected {expected}: {directory}')
        if len(files) != expected:
            issues.append(f'{dataset}: expected {expected} CIF files, found {len(files)}; '
                          'missing IDs cannot be inferred without an authoritative ID list')
        for path in files:
            pdb_id = path.stem.upper()
            flags = []
            if not ID_PATTERN.fullmatch(pdb_id):
                flags.append('INVALID_FILENAME')
            else:
                expected_name = (pdb_id.lower() if dataset == 'pure' else pdb_id) + '.cif'
                if path.name != expected_name:
                    flags.append('FILENAME_CASE_MISMATCH')
            jobs.append(Job(dataset, pdb_id, path, flags))
    counts = Counter(job.pdb_id for job in jobs)
    for job in jobs:
        if counts[job.pdb_id] > 1:
            job.flags.append('DUPLICATE_PDB_ID')
    return jobs, issues


def download_reference(job: Job, target: Path, args: argparse.Namespace,
                       stop: threading.Event) -> tuple[int, str]:
    url = reference_url(job.pdb_id, args.source)
    for attempt in range(1, args.attempts + 1):
        if stop.is_set():
            raise InterruptedError('stop requested before remote verification')
        try:
            request = Request(url, headers={
                'User-Agent': 'RNA-CIF-integrity-check/1.0 (research)',
                'Accept-Encoding': 'identity',
            })
            with urlopen(request, timeout=args.timeout) as response:
                last_modified = response.headers.get('Last-Modified', '')
                # Reading to EOF verifies gzip trailer/CRC, including truncation.
                with gzip.GzipFile(fileobj=response) as uncompressed:
                    with target.open('wb') as handle:
                        while True:
                            if stop.is_set():
                                raise InterruptedError('stop requested during download')
                            chunk = uncompressed.read(CHUNK)
                            if not chunk:
                                break
                            handle.write(chunk)
            if target.stat().st_size == 0:
                raise ValueError('Official PDB archive returned an empty CIF')
            return attempt, last_modified
        except InterruptedError:
            raise
        except (OSError, EOFError, ValueError, URLError, http.client.HTTPException,
                zlib.error) as error:
            # Missing/obsolete entries do not improve with immediate retries.
            retryable = not isinstance(error, HTTPError) or error.code in (
                408, 429, 500, 502, 503, 504,
            )
            if attempt == args.attempts or not retryable:
                raise DownloadFailure(
                    f'after {attempt} attempt(s): {type(error).__name__}: {error}', attempt
                ) from error
            if stop.wait(min(2 ** attempt, 30)):
                raise InterruptedError('stop requested during retry delay')
    raise RuntimeError('no download attempts performed')


def read_reference_manifest(directory: Path) -> dict:
    path = directory / 'reference_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('records'), dict):
        raise ValueError(f'Invalid reference manifest: {path}')
    return manifest


def reference_record_ok(pdb_id: str, record: dict) -> bool:
    return (isinstance(record, dict) and record.get('source') in URL_TEMPLATES
            and record.get('source_url') == reference_url(pdb_id, record['source'])
            and bool(re.fullmatch(r'[0-9a-f]{64}', str(record.get('sha256', ''))))
            and isinstance(record.get('bytes'), int)
            and isinstance(record.get('downloaded_at_utc'), str))


def prepare_reference(job: Job, args: argparse.Namespace, records: dict,
                      stop: threading.Event) -> dict:
    """Fetch one reference on an online machine; never trust an unrecorded cache."""
    if stop.is_set():
        return {'pdb_id': job.pdb_id, 'error': 'INTERRUPTED'}
    previous = records.get(job.pdb_id, {})
    destination = args.reference_dir / f'{job.pdb_id}.cif'
    try:
        if (not args.refresh_references and reference_record_ok(job.pdb_id, previous)
                and previous['source'] == args.source and destination.is_file()
                and destination.stat().st_size == previous['bytes']
                and sha256_file(destination) == previous['sha256']):
            return {'pdb_id': job.pdb_id, 'record': previous, 'action': 'REUSED_VERIFIED'}
        with tempfile.TemporaryDirectory(prefix='.download_', dir=args.reference_dir) as tmp:
            temporary = Path(tmp) / 'reference.cif'
            attempts, modified = download_reference(job, temporary, args, stop)
            parsed = inspect_cif(temporary, job.pdb_id)
            if not parsed['valid']:
                raise ValueError('Official reference failed validation: ' + parsed['error'])
            record = {
                'source': args.source, 'source_url': reference_url(job.pdb_id, args.source),
                'downloaded_at_utc': now(), 'http_last_modified': modified,
                'sha256': sha256_file(temporary), 'bytes': temporary.stat().st_size,
                'revision_date': parsed['revision'], 'atoms': parsed['atoms'],
                'attempts': attempts,
            }
            os.replace(temporary, destination)
        return {'pdb_id': job.pdb_id, 'record': record, 'action': 'DOWNLOADED'}
    except Exception as error:
        return {'pdb_id': job.pdb_id, 'error': f'{type(error).__name__}: {error}'}


def download_reference_set(args: argparse.Namespace) -> int:
    ids = sorted({line.strip().upper() for line in args.download_id_list.read_text(
        encoding='utf-8-sig').splitlines() if line.strip()})
    if not ids or any(not ID_PATTERN.fullmatch(pdb_id) for pdb_id in ids):
        raise SystemExit('ID list must contain one valid four-character PDB ID per line')
    args.reference_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.reference_dir / 'reference_manifest.json'
    stop = threading.Event()
    records = read_reference_manifest(args.reference_dir)['records'] if manifest_path.exists() else {}
    errors = {}
    completed = set()
    started = now()
    with (args.reference_dir / '.download.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log('Another download task holds the reference-directory lock')
            return 2
        previous_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, lambda signum, frame: stop.set())

        def save(status: str) -> None:
            atomic_text(manifest_path, json.dumps({
                'schema_version': 1, 'status': status, 'started_at_utc': started,
                'updated_at_utc': now(), 'requested_ids': ids, 'requested_count': len(ids),
                'processed_count': len(completed), 'failed_ids': errors,
                'unchecked_ids': sorted(set(ids) - completed),
                'all_references_ready': status == 'COMPLETED' and not errors
                    and len(completed) == len(ids),
                'records': records,
            }, ensure_ascii=False, indent=2) + '\n')

        log(f'Downloading {len(ids)} official references from {args.source}; '
            f'cache={args.reference_dir}; workers={args.workers}')
        save('RUNNING')
        fatal = ''
        pool = ThreadPoolExecutor(max_workers=args.workers)
        try:
            cursor = 0
            active = {}
            while active or (cursor < len(ids) and not stop.is_set()):
                while cursor < len(ids) and len(active) < args.workers and not stop.is_set():
                    pdb_id = ids[cursor]
                    job = Job('reference', pdb_id, args.reference_dir / f'{pdb_id}.cif', [])
                    active[pool.submit(prepare_reference, job, args, records, stop)] = pdb_id
                    cursor += 1
                if not active:
                    break
                done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                if not done:
                    log(f'Download heartbeat: {len(completed)}/{len(ids)}; stopping={stop.is_set()}')
                for future in done:
                    pdb_id = active.pop(future)
                    result = future.result()
                    completed.add(pdb_id)
                    if 'error' in result:
                        # Do not make an older cached record appear verified by this attempt.
                        records.pop(pdb_id, None)
                        errors[pdb_id] = result['error']
                        log(f'{len(completed)}/{len(ids)} {pdb_id} FAILED {result["error"]}')
                    else:
                        records[pdb_id] = result['record']
                        log(f'{len(completed)}/{len(ids)} {pdb_id} {result["action"]}')
                    save('RUNNING')
        except Exception as error:
            fatal = f'{type(error).__name__}: {error}'
            stop.set()
            log('DOWNLOAD FATAL: ' + fatal)
        finally:
            pool.shutdown(wait=True)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        status = 'FAILED' if fatal else 'INTERRUPTED' if stop.is_set() else 'COMPLETED'
        save(status)
        log(f'FINAL DOWNLOAD STATUS: {status}; ready={len(completed) - len(errors)}/{len(ids)}; '
            f'manifest={manifest_path}')
        return 2 if fatal else 130 if stop.is_set() else 1 if errors else 0


def check_job(job: Job, args: argparse.Namespace, stop: threading.Event) -> dict:
    row = dict.fromkeys(FIELDS, '')
    row.update(DATASET=job.dataset, PDB_ID=job.pdb_id, LOCAL_PATH=str(job.path),
               STATUS='PROBLEM', SOURCE=args.source,
               SOURCE_URL=reference_url(job.pdb_id, args.source),
               REFERENCE_MODE='offline' if args.reference_dir else 'online',
               ATTEMPTS=0)
    reasons = list(job.flags)
    try:
        if 'INVALID_FILENAME' in reasons:
            return row | {'REASONS': '; '.join(reasons), 'CHECKED_AT_UTC': now()}
        before = signature(job.path)
        row['LOCAL_BYTES'] = before[2]
        row['LOCAL_SHA256'] = sha256_file(job.path)
        local = inspect_cif(job.path, job.pdb_id)
        row.update(LOCAL_VALID=local['valid'], LOCAL_ATOMS=local['atoms'],
                   LOCAL_REVISION_DATE=local['revision'])
        if not local['valid']:
            reasons.append('LOCAL_INVALID: ' + local['error'])
        if signature(job.path) != before:
            reasons.append('LOCAL_CHANGED_DURING_CHECK')
        # Each worker keeps only one temporary decompressed reference file.
        with tempfile.TemporaryDirectory(prefix='pdb_rcsb_check_', dir=args.temp_dir) as tmp:
            reference = Path(tmp) / f'{job.pdb_id}.cif'
            try:
                reference_before = None
                if args.reference_dir:
                    record = args.reference_records.get(job.pdb_id)
                    if not reference_record_ok(job.pdb_id, record):
                        raise ValueError('missing or invalid official reference manifest record')
                    reference = args.reference_dir / f'{job.pdb_id}.cif'
                    reference_before = signature(reference)
                    row.update(SOURCE=record['source'], SOURCE_URL=record['source_url'],
                               REFERENCE_DOWNLOADED_AT_UTC=record['downloaded_at_utc'])
                    if (reference_before[2] != record['bytes']
                            or sha256_file(reference) != record['sha256']):
                        raise ValueError('reference file was modified or damaged after download')
                    attempts, modified = 0, record.get('http_last_modified', '')
                else:
                    attempts, modified = download_reference(job, reference, args, stop)
                    row['REFERENCE_DOWNLOADED_AT_UTC'] = now()
            except Exception as error:
                row['ATTEMPTS'] = getattr(error, 'attempts', 0)
                reasons.append(f'REMOTE_UNVERIFIED: {type(error).__name__}: {error}')
            else:
                row.update(ATTEMPTS=attempts, HTTP_LAST_MODIFIED=modified,
                           REMOTE_BYTES=reference.stat().st_size,
                           REMOTE_SHA256=sha256_file(reference))
                if row['LOCAL_SHA256'] == row['REMOTE_SHA256']:
                    # Same bytes imply identical parsing; avoid parsing huge CIF twice.
                    remote = local
                else:
                    remote = inspect_cif(reference, job.pdb_id)
                    reasons.append('CONTENT_DIFFERENT_FROM_DOWNLOADED_PDB_REFERENCE'
                                   if args.reference_dir else 'CONTENT_DIFFERENT_FROM_CURRENT_PDB')
                row.update(REMOTE_VALID=remote['valid'], REMOTE_ATOMS=remote['atoms'],
                           REMOTE_REVISION_DATE=remote['revision'])
                if not remote['valid']:
                    reasons.append('REMOTE_INVALID: ' + remote['error'])
                if reference_before is not None and signature(reference) != reference_before:
                    reasons.append('REFERENCE_CHANGED_DURING_CHECK')
        if signature(job.path) != before and 'LOCAL_CHANGED_DURING_CHECK' not in reasons:
            reasons.append('LOCAL_CHANGED_DURING_CHECK')
    except Exception as error:
        reasons.append(f'CHECK_ERROR: {type(error).__name__}: {error}')
    row.update(STATUS='PASS' if not reasons else 'PROBLEM',
               REASONS='; '.join(reasons), CHECKED_AT_UTC=now())
    return row


# Deliberately narrow: unknown category changes require review instead of being
# silently called harmless metadata. Coordinates are compared without fitting.
METADATA_CATEGORIES = {
    '_pdbx_audit_revision_history.', '_pdbx_audit_revision_details.',
    '_pdbx_audit_revision_group.', '_pdbx_audit_revision_category.',
    '_pdbx_audit_revision_item.', '_pdbx_audit_support.', '_audit_author.',
    '_audit_conform.', '_audit.', '_citation.', '_citation_author.',
    '_citation_editor.', '_database_2.', '_pdbx_database_status.',
    '_pdbx_database_related.', '_pdbx_database_pdb_obs_spr.', '_software.',
}
SEQUENCE_FIELDS = {'entity_id', 'type', 'pdbx_seq_one_letter_code',
                   'pdbx_seq_one_letter_code_can'}
MAPPING_CATEGORIES = {'_struct_asym.', '_pdbx_poly_seq_scheme.',
                      '_pdbx_nonpoly_scheme.', '_pdbx_branch_scheme.'}
DIFF_FIELDS = [
    'DATASET', 'PDB_ID', 'CLASSIFICATION', 'DETAIL', 'CHANGED_CATEGORIES',
    'LOCAL_ATOMS', 'REFERENCE_ATOMS', 'COMMON_ATOMS', 'LOCAL_ONLY_ATOMS',
    'REFERENCE_ONLY_ATOMS', 'MOVED_ATOMS', 'DIRECT_RMSD_ANGSTROM',
    'MAX_DISPLACEMENT_ANGSTROM', 'COORDINATES_EQUAL', 'SEQUENCES_EQUAL',
    'CHAIN_MAPPING_EQUAL', 'ATOM_LABELS_EQUAL', 'ATOM_SITE_EQUAL',
    'LOCAL_RELEASE_DATE', 'REFERENCE_RELEASE_DATE', 'LOCAL_REVISION_DATE',
    'REFERENCE_REVISION_DATE', 'LOCAL_SHA256', 'REFERENCE_SHA256',
    'REFERENCE_DOWNLOADED_AT_UTC', 'LOCAL_PATH', 'REFERENCE_PATH',
]


def category_digest(block, name: str, fields=None) -> tuple[str, int]:
    """Compare unordered rows with their columns intact; ignore quotes/layout."""
    from gemmi import cif
    data = {k.lower(): v for k, v in block.get_mmcif_category(name, raw=True).items()}
    keys = sorted(k for k in data if fields is None or k in fields)
    digest = hashlib.sha256(json.dumps(keys).encode())
    if not keys:
        return digest.hexdigest(), 0
    count = len(data[keys[0]])
    if any(len(data[k]) != count for k in keys):
        raise ValueError(f'inconsistent category column lengths: {name}')
    row_hashes = []
    for index in range(count):
        values = []
        for key in keys:
            raw = data[key][index]
            if raw in ('.', '?'):
                values.append(('NULL', raw))
                continue
            value = cif.as_string(raw)
            if name.lower() == '_entity_poly.' and key.startswith('pdbx_seq_one_letter_code'):
                value = re.sub(r'\s+', '', value)
            elif name.lower() == '_atom_site.' and (
                key.startswith(('cartn_', 'fract_')) or key in {
                    'occupancy', 'occupancy_esd', 'b_iso_or_equiv',
                    'b_iso_or_equiv_esd', 'pdbx_formal_charge',
                }
            ):
                number = Decimal(value)
                value = '0' if number == 0 else str(number.normalize())
            values.append(('VALUE', value))
        row_hashes.append(hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).digest())
    for row_hash in sorted(row_hashes):
        digest.update(row_hash)
    return digest.hexdigest(), count


def atom_records(block) -> dict:
    """Align atoms by label/model/altloc; use auth residue IDs for nonpolymers."""
    from gemmi import cif
    data = {k.lower(): v for k, v in block.get_mmcif_category('_atom_site.', raw=True).items()}
    required = ('label_asym_id', 'label_seq_id', 'label_comp_id', 'label_atom_id',
                'cartn_x', 'cartn_y', 'cartn_z')
    if any(key not in data for key in required):
        raise ValueError('atom identifiers/coordinates missing; cannot establish an unambiguous mapping')
    n = len(data['label_atom_id'])
    if not n or any(len(values) != n for values in data.values()):
        raise ValueError('empty/inconsistent atom table')

    def cell(key, index, default=''):
        raw = data[key][index] if key in data else default
        return '' if raw in ('.', '?', '') else cif.as_string(raw)

    records = {}
    for i in range(n):
        sequence_id = cell('label_seq_id', i)
        key = (cell('pdbx_pdb_model_num', i, '1'), cell('label_asym_id', i),
               sequence_id, cell('label_comp_id', i), cell('label_atom_id', i),
               cell('label_alt_id', i), cell('pdbx_pdb_ins_code', i),
               '' if sequence_id else cell('auth_seq_id', i))
        if key in records:
            raise ValueError('duplicate atom identity: ambiguous mapping; manual review required')
        xyz = tuple(float(cell('cartn_' + axis, i)) for axis in 'xyz')
        if not all(math.isfinite(x) for x in xyz):
            raise ValueError('non-finite atom coordinates')
        labels = tuple(cell(k, i) for k in (
            'id', 'label_entity_id', 'auth_asym_id', 'auth_seq_id',
            'auth_comp_id', 'auth_atom_id',
        ))
        records[key] = (xyz, labels)
    return records


def diff_snapshot(path: Path, pdb_id: str) -> dict:
    from gemmi import cif
    document = cif.read_file(str(path), check_level=2)
    if len(document) != 1:
        raise ValueError('expected one CIF data block')
    block = document[0]
    value = block.find_value('_entry.id')
    if not value or cif.as_string(value).strip().upper() != pdb_id:
        raise ValueError('entry ID mismatch')
    for item in block:
        if item.frame is not None:
            raise ValueError('save frames are not supported in semantic comparison')
    categories = {name.lower(): category_digest(block, name)
                  for name in block.get_mmcif_category_names()}
    poly = block.get_mmcif_category('_entity_poly.')
    if not any(poly.get(k) for k in ('pdbx_seq_one_letter_code', 'pdbx_seq_one_letter_code_can')):
        raise ValueError('missing polymer sequence; cannot compare sequences')
    dates = [x for x in block.find_values('_pdbx_audit_revision_history.revision_date')
             if x not in ('.', '?')]
    dates = [cif.as_string(x) for x in dates]
    original = block.find_value('_database_PDB_rev.date_original')
    return {
        'document': document, 'categories': categories,
        'atoms': atom_records(block),
        'sequence': (category_digest(block, '_entity_poly.', SEQUENCE_FIELDS),
                     categories.get('_entity_poly_seq.')),
        'mapping': (category_digest(block, '_entity_poly.', {'entity_id', 'pdbx_strand_id'}),
                    tuple((k, categories.get(k)) for k in sorted(MAPPING_CATEGORIES))),
        'release': min(dates) if dates else cif.as_string(original) if original else '',
        'revision': max(dates) if dates else '',
        'methods': sorted(cif.as_string(x) for x in block.find_values('_exptl.method')),
    }


def analyze_difference_row(audit: dict, args: argparse.Namespace) -> tuple[dict, list]:
    row = dict.fromkeys(DIFF_FIELDS, '')
    pdb_id = audit['PDB_ID'].strip().upper()
    local = Path(audit['LOCAL_PATH']).expanduser().resolve()
    reference = args.reference_dir / f'{pdb_id}.cif'
    row.update(DATASET=audit['DATASET'], PDB_ID=pdb_id, CLASSIFICATION='ERROR',
               LOCAL_PATH=str(local), REFERENCE_PATH=str(reference))
    category_rows = []
    try:
        record = args.reference_records.get(pdb_id)
        if not ID_PATTERN.fullmatch(pdb_id) or not reference_record_ok(pdb_id, record):
            raise ValueError('missing or invalid reference provenance')
        local_before, reference_before = signature(local), signature(reference)
        local_sha, reference_sha = sha256_file(local), sha256_file(reference)
        row.update(LOCAL_SHA256=local_sha, REFERENCE_SHA256=reference_sha,
                   REFERENCE_DOWNLOADED_AT_UTC=record['downloaded_at_utc'])
        if reference_sha != record['sha256'] or reference_before[2] != record['bytes']:
            raise ValueError('official reference changed/damaged after download')
        if audit.get('LOCAL_SHA256') and local_sha != audit['LOCAL_SHA256']:
            raise ValueError('local file changed since the original audit; rerun integrity audit first')
        if audit.get('REMOTE_SHA256') and reference_sha != audit['REMOTE_SHA256']:
            raise ValueError('reference changed since the original audit; rerun integrity audit first')
        left, right = diff_snapshot(local, pdb_id), diff_snapshot(reference, pdb_id)
        changed = sorted(k for k in set(left['categories']) | set(right['categories'])
                         if left['categories'].get(k) != right['categories'].get(k))
        for category in changed:
            category_rows.append({
                'PDB_ID': pdb_id, 'CATEGORY': category,
                'LOCAL_ROWS': left['categories'].get(category, ('', 0))[1],
                'REFERENCE_ROWS': right['categories'].get(category, ('', 0))[1],
                'LOCAL_SHA256': left['categories'].get(category, ('', 0))[0],
                'REFERENCE_SHA256': right['categories'].get(category, ('', 0))[0],
            })
        a, b = left['atoms'], right['atoms']
        common = a.keys() & b.keys()
        squared = [sum((a[k][0][axis] - b[k][0][axis]) ** 2 for axis in range(3))
                   for k in common]
        moved = sum(value > 0 for value in squared)
        same_atoms = a.keys() == b.keys()
        coords_equal = same_atoms and moved == 0
        labels_equal = same_atoms and all(a[k][1] == b[k][1] for k in common)
        sequence_equal = left['sequence'] == right['sequence']
        mapping_equal = left['mapping'] == right['mapping']
        atom_site_equal = '_atom_site.' not in changed
        row.update(
            CHANGED_CATEGORIES=';'.join(changed), LOCAL_ATOMS=len(a), REFERENCE_ATOMS=len(b),
            COMMON_ATOMS=len(common), LOCAL_ONLY_ATOMS=len(a.keys() - b.keys()),
            REFERENCE_ONLY_ATOMS=len(b.keys() - a.keys()), MOVED_ATOMS=moved,
            DIRECT_RMSD_ANGSTROM=math.sqrt(math.fsum(squared) / len(squared)) if squared else '',
            MAX_DISPLACEMENT_ANGSTROM=math.sqrt(max(squared)) if squared else '',
            COORDINATES_EQUAL=coords_equal, SEQUENCES_EQUAL=sequence_equal,
            CHAIN_MAPPING_EQUAL=mapping_equal, ATOM_LABELS_EQUAL=labels_equal,
            ATOM_SITE_EQUAL=atom_site_equal, LOCAL_RELEASE_DATE=left['release'],
            REFERENCE_RELEASE_DATE=right['release'], LOCAL_REVISION_DATE=left['revision'],
            REFERENCE_REVISION_DATE=right['revision'],
        )
        if local_sha == reference_sha:
            classification = 'IDENTICAL'
        elif not sequence_equal or not mapping_equal:
            classification = 'SEQUENCE_OR_CHAIN_MAPPING_CHANGED'
        elif not same_atoms:
            classification = 'ATOM_SET_OR_IDENTIFIERS_CHANGED'
        elif moved:
            classification = 'COORDINATES_CHANGED'
        elif not labels_equal:
            classification = 'ATOM_LABELS_OR_NUMBERING_CHANGED'
        elif not atom_site_equal:
            classification = 'ATOM_ATTRIBUTES_CHANGED'
        elif left['release'] != right['release'] or left['methods'] != right['methods']:
            classification = 'RELEASE_DATE_OR_METHOD_CHANGED'
        elif any(k.startswith(('_chem_comp', '_pdbx_chem_comp')) or k == '_struct_conn.' for k in changed):
            classification = 'CHEMISTRY_OR_CONNECTIVITY_CHANGED'
        elif any(k not in METADATA_CATEGORIES for k in changed):
            classification = 'OTHER_CATEGORY_CHANGED_NEEDS_REVIEW'
        else:
            classification = 'METADATA_ONLY' if changed else 'FORMAT_ONLY'
        if signature(local) != local_before or signature(reference) != reference_before:
            raise ValueError('input file changed during difference analysis')
        row.update(CLASSIFICATION=classification,
                   DETAIL='All atoms/models compared by identifiers; exact numeric coordinates; '
                          'no fitting. Multiple change flags may coexist; classification shows priority.')
    except Exception as error:
        row.update(CLASSIFICATION='ERROR', DETAIL=f'{type(error).__name__}: {error}')
    return row, category_rows


def analyze_differences(args: argparse.Namespace) -> int:
    args.reference_records = read_reference_manifest(args.reference_dir)['records']
    with args.analyze_differences.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if not {'DATASET', 'PDB_ID', 'LOCAL_PATH', 'STATUS'}.issubset(reader.fieldnames or []):
            raise SystemExit('Invalid integrity details TSV')
        selected = [row for row in reader if row['STATUS'] != 'PASS']
    report = args.difference_report
    category_path = report.with_suffix('.categories.tsv')
    summary_path = report.with_suffix('.summary.json')
    lock_path = report.with_suffix('.lock')
    protected = {args.analyze_differences, args.reference_dir / 'reference_manifest.json'}
    protected.update(Path(row['LOCAL_PATH']).expanduser().resolve() for row in selected)
    protected.update(args.reference_dir / f'{row["PDB_ID"].upper()}.cif' for row in selected)
    if any(p.resolve() in protected for p in (report, category_path, summary_path, lock_path)):
        raise SystemExit('Difference report would overwrite an input file')
    report.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    started = now()
    stop = threading.Event()
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log('Another difference-analysis process holds this report lock')
            return 2

        def save(status):
            atomic_text(summary_path, json.dumps({
                'status': status, 'started_at_utc': started, 'updated_at_utc': now(),
                'selected_files': len(selected), 'processed_files': sum(counts.values()),
                'unchecked_files': len(selected) - sum(counts.values()),
                'classification_counts': dict(counts), 'report_tsv': str(report),
                'category_changes_tsv': str(category_path),
                'scope': 'All atoms/models, polymer sequences and chain mapping. '
                    'Atom row order and numeric formatting ignored; no rigid alignment. '
                    'Only a narrow allowlist of provenance/literature categories counts as METADATA_ONLY.',
            }, ensure_ascii=False, indent=2) + '\n')

        handlers = {sig: signal.signal(sig, lambda signum, frame: stop.set())
                    for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            save('RUNNING')
            with report.open('w', encoding='utf-8-sig', newline='') as table, \
                    category_path.open('w', encoding='utf-8-sig', newline='') as category_table:
                writer = csv.DictWriter(table, fieldnames=DIFF_FIELDS, delimiter='\t')
                category_writer = csv.DictWriter(category_table, fieldnames=[
                    'PDB_ID', 'CATEGORY', 'LOCAL_ROWS', 'REFERENCE_ROWS',
                    'LOCAL_SHA256', 'REFERENCE_SHA256',
                ], delimiter='\t')
                writer.writeheader()
                category_writer.writeheader()
                log(f'Offline difference analysis: {len(selected)} files; sequential to bound memory')
                for audit in selected:
                    if stop.is_set():
                        break
                    row, categories = analyze_difference_row(audit, args)
                    writer.writerow(row)
                    category_writer.writerows(categories)
                    table.flush()
                    category_table.flush()
                    counts[row['CLASSIFICATION']] += 1
                    save('RUNNING')
                    log(f'{sum(counts.values())}/{len(selected)} {row["PDB_ID"]} '
                        f'{row["CLASSIFICATION"]} moved_atoms={row["MOVED_ATOMS"]}')
            save('INTERRUPTED' if stop.is_set() else 'COMPLETED')
        except BaseException:
            save('FAILED')
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    log(f'FINAL DIFFERENCE SUMMARY: {dict(counts)}; {summary_path}')
    return 130 if stop.is_set() else 1 if counts['ERROR'] else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pure-dir', type=Path, default=HOME / 'pdb_data')
    p.add_argument('--mixed-dir', type=Path, default=HOME / 'pdb_data_mixed')
    p.add_argument('--expected-pure', type=int, default=2241)
    p.add_argument('--expected-mixed', type=int, default=77)
    p.add_argument('--output', type=Path, default=HOME / 'pdb_cif_problem_ids.txt')
    p.add_argument('--source', choices=tuple(URL_TEMPLATES), default='rcsb',
                   help='official archive mirror: rcsb (default), pdbe, pdbj')
    p.add_argument('--workers', type=int, default=2, help='simultaneous files (default 2)')
    p.add_argument('--attempts', type=int, default=3, help='HTTP attempts per ID')
    p.add_argument('--timeout', type=float, default=90, help='HTTP socket timeout in seconds')
    p.add_argument('--temp-dir', type=Path, help='temporary download parent; default system temp')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--export-id-list', type=Path,
                       help='only inventory server directories and export IDs; no network/Gemmi')
    modes.add_argument('--download-id-list', type=Path,
                       help='download official references for an exported ID list on an online machine')
    modes.add_argument('--analyze-differences', type=Path,
                       help='offline semantic analysis of non-PASS rows in integrity details TSV')
    p.add_argument('--difference-report', type=Path, default=HOME / 'pdb_cif_difference_report.tsv')
    p.add_argument('--reference-dir', type=Path,
                   help='download cache destination, or official reference cache for offline comparison')
    p.add_argument('--refresh-references', action='store_true',
                   help='download mode: refetch even previously verified cached references')
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.workers < 1 or args.attempts < 1 or args.timeout <= 0:
        raise SystemExit('workers/attempts/timeout must be positive')
    if min(args.expected_pure, args.expected_mixed) < 0:
        raise SystemExit('expected counts must be nonnegative')
    for name in ('pure_dir', 'mixed_dir', 'output', 'reference_dir',
                 'export_id_list', 'download_id_list', 'analyze_differences', 'difference_report'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.reference_dir in (args.pure_dir, args.mixed_dir):
        raise SystemExit('Reference directory must be separate from both source CIF directories')
    if args.export_id_list:
        jobs, issues = inventory(args)
        if issues or any('INVALID_FILENAME' in job.flags or 'DUPLICATE_PDB_ID' in job.flags
                         for job in jobs):
            log(f'Cannot export an incomplete/ambiguous inventory: {issues}')
            return 2
        if args.export_id_list in {job.path.resolve() for job in jobs}:
            raise SystemExit('ID-list output would overwrite a source CIF')
        args.export_id_list.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(args.export_id_list, ''.join(pdb_id + '\n' for pdb_id in
                                               sorted({job.pdb_id for job in jobs})))
        log(f'Exported {len(jobs)} PDB IDs to {args.export_id_list}')
        return 0
    try:
        import gemmi
    except ImportError:
        raise SystemExit('Missing gemmi. Install in this Python environment: python -m pip install "gemmi>=0.7.5,<0.8"')
    if args.analyze_differences:
        if not args.reference_dir:
            raise SystemExit('--analyze-differences requires --reference-dir')
        return analyze_differences(args)
    if args.download_id_list:
        if not args.reference_dir:
            raise SystemExit('--download-id-list requires --reference-dir')
        if args.download_id_list.parent == args.reference_dir:
            raise SystemExit('Keep the input ID list outside the reference directory')
        return download_reference_set(args)
    args.reference_records = (read_reference_manifest(args.reference_dir)['records']
                              if args.reference_dir else {})
    if args.temp_dir:
        args.temp_dir = args.temp_dir.expanduser().resolve()
        if not args.temp_dir.is_dir():
            raise SystemExit(f'Temporary directory does not exist: {args.temp_dir}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    details = args.output.with_suffix('.details.tsv')
    summary_path = args.output.with_suffix('.summary.json')
    lock_path = args.output.with_suffix('.lock')
    stop = threading.Event()
    started = now()
    jobs, inventory_issues = inventory(args)
    protected = {job.path.resolve() for job in jobs}
    if any(path.resolve() in protected for path in (args.output, details, summary_path, lock_path)):
        raise SystemExit('Report path would overwrite a source CIF; choose another --output')
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f'Another checker holds {lock_path}; refusing concurrent run')
            return 2
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + '\n')
        lock.flush()
        log(f'Start: {len(jobs)} files; workers={args.workers}; '
            f'gemmi={gemmi.__version__}; '
            f'source={args.reference_dir or args.source}')
        log(f'Problem/unverified IDs: {args.output}; details: {details}')
        for issue in inventory_issues:
            log('INVENTORY ERROR: ' + issue)
        log('TXT includes unchecked IDs while RUNNING; trust final summary status.')
        results: dict[int, dict] = {}
        old_handlers = {}

        def request_stop(signum, frame):
            stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, request_stop)

        def checkpoint(status: str, fatal: str = '') -> None:
            unresolved = {job.pdb_id for i, job in enumerate(jobs)
                          if i not in results or results[i]['STATUS'] != 'PASS'}
            # Invalid filenames are escaped so one problematic name stays on one line.
            atomic_text(args.output, ''.join(x.replace('\r', '\\r').replace('\n', '\\n') + '\n'
                                            for x in sorted(unresolved)))
            matched = sum(row['STATUS'] == 'PASS' for row in results.values())
            payload = {
                'status': status, 'started_at_utc': started, 'updated_at_utc': now(),
                'pid': os.getpid(), 'expected_pure': args.expected_pure,
                'expected_mixed': args.expected_mixed, 'found_files': len(jobs),
                'checked_files': len(results), 'identical_valid_files': matched,
                'problem_files_checked': len(results) - matched,
                'unchecked_files': len(jobs) - len(results),
                'problem_or_unverified_id_count': len(unresolved),
                'inventory_issues': inventory_issues, 'fatal_error': fatal,
                'all_files_identical_and_valid': bool(jobs) and status == 'COMPLETED'
                    and not inventory_issues and matched == len(jobs),
                'reference_source': 'reference_manifest' if args.reference_dir else args.source,
                'reference_mode': 'offline' if args.reference_dir else 'online',
                'reference_directory': str(args.reference_dir) if args.reference_dir else None,
                'comparison': 'SHA256 of exact local bytes vs official PDB reference bytes; '
                    'offline references reflect the recorded download time',
                'scope': 'syntax, entry ID, basic metadata, finite coordinates; not experimental or stereochemical validation',
                'problem_ids_txt': str(args.output), 'details_tsv': str(details),
            }
            atomic_text(summary_path, json.dumps(payload, ensure_ascii=False, indent=2) + '\n')

        pool = ThreadPoolExecutor(max_workers=args.workers)
        fatal = ''
        try:
            with details.open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter='\t')
                writer.writeheader()
                handle.flush()
                checkpoint('RUNNING')
                cursor = 0
                active = {}
                while active or (cursor < len(jobs) and not stop.is_set()):
                    while cursor < len(jobs) and len(active) < args.workers and not stop.is_set():
                        active[pool.submit(check_job, jobs[cursor], args, stop)] = cursor
                        cursor += 1
                    if not active:
                        break
                    done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                    if not done:
                        log(f'Heartbeat: checked={len(results)}/{len(jobs)}; active={len(active)}; stopping={stop.is_set()}')
                        continue
                    for future in done:
                        index = active.pop(future)
                        row = future.result()
                        results[index] = row
                        writer.writerow(row)
                        handle.flush()
                        checkpoint('RUNNING')
                        log(f'{len(results)}/{len(jobs)} {row["DATASET"]} {row["PDB_ID"]} '
                            f'{row["STATUS"]} {row["REASONS"]}')
        except Exception as error:
            stop.set()
            fatal = f'{type(error).__name__}: {error}'
            log('FATAL: ' + fatal)
        finally:
            pool.shutdown(wait=True)
            for sig, previous in old_handlers.items():
                signal.signal(sig, previous)
        status = 'FAILED' if fatal else 'INTERRUPTED' if stop.is_set() else 'COMPLETED'
        checkpoint(status, fatal)
        log(f'FINAL STATUS: {status}; summary: {summary_path}')
        if fatal or inventory_issues:
            return 2
        if stop.is_set():
            return 130
        return 0 if jobs and all(row['STATUS'] == 'PASS' for row in results.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
