#!/usr/bin/env python3
"""Read-only audit of local CIF files against freshly downloaded RCSB files.

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
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HOME = Path('/storage9920/home/tinghao.xia')
URL_TEMPLATE = 'https://files.rcsb.org/download/{pdb_id}.cif.gz'
ID_PATTERN = re.compile(r'^[A-Z0-9]{4}$')
CHUNK = 1024 * 1024
FIELDS = [
    'DATASET', 'PDB_ID', 'LOCAL_PATH', 'STATUS', 'REASONS',
    'LOCAL_VALID', 'REMOTE_VALID', 'LOCAL_BYTES', 'REMOTE_BYTES',
    'LOCAL_SHA256', 'REMOTE_SHA256', 'LOCAL_ATOMS', 'REMOTE_ATOMS',
    'LOCAL_REVISION_DATE', 'REMOTE_REVISION_DATE', 'SOURCE_URL',
    'HTTP_LAST_MODIFIED', 'ATTEMPTS', 'CHECKED_AT_UTC',
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


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
    url = URL_TEMPLATE.format(pdb_id=job.pdb_id.lower())
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
                raise ValueError('RCSB returned an empty CIF')
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


def check_job(job: Job, args: argparse.Namespace, stop: threading.Event) -> dict:
    row = dict.fromkeys(FIELDS, '')
    row.update(DATASET=job.dataset, PDB_ID=job.pdb_id, LOCAL_PATH=str(job.path),
               STATUS='PROBLEM', SOURCE_URL=URL_TEMPLATE.format(pdb_id=job.pdb_id.lower()),
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
                attempts, modified = download_reference(job, reference, args, stop)
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
                    reasons.append('CONTENT_DIFFERENT_FROM_CURRENT_RCSB')
                row.update(REMOTE_VALID=remote['valid'], REMOTE_ATOMS=remote['atoms'],
                           REMOTE_REVISION_DATE=remote['revision'])
                if not remote['valid']:
                    reasons.append('REMOTE_INVALID: ' + remote['error'])
        if signature(job.path) != before and 'LOCAL_CHANGED_DURING_CHECK' not in reasons:
            reasons.append('LOCAL_CHANGED_DURING_CHECK')
    except Exception as error:
        reasons.append(f'CHECK_ERROR: {type(error).__name__}: {error}')
    row.update(STATUS='PASS' if not reasons else 'PROBLEM',
               REASONS='; '.join(reasons), CHECKED_AT_UTC=now())
    return row


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pure-dir', type=Path, default=HOME / 'pdb_data')
    p.add_argument('--mixed-dir', type=Path, default=HOME / 'pdb_data_mixed')
    p.add_argument('--expected-pure', type=int, default=2241)
    p.add_argument('--expected-mixed', type=int, default=77)
    p.add_argument('--output', type=Path, default=HOME / 'pdb_cif_problem_ids.txt')
    p.add_argument('--workers', type=int, default=2, help='simultaneous files (default 2)')
    p.add_argument('--attempts', type=int, default=3, help='HTTP attempts per ID')
    p.add_argument('--timeout', type=float, default=90, help='HTTP socket timeout in seconds')
    p.add_argument('--temp-dir', type=Path, help='temporary download parent; default system temp')
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.workers < 1 or args.attempts < 1 or args.timeout <= 0:
        raise SystemExit('workers/attempts/timeout must be positive')
    if min(args.expected_pure, args.expected_mixed) < 0:
        raise SystemExit('expected counts must be nonnegative')
    try:
        import gemmi
    except ImportError:
        raise SystemExit('Missing gemmi. Install in this Python environment: python -m pip install "gemmi>=0.7.5,<0.8"')
    for name in ('pure_dir', 'mixed_dir', 'output'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
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
        log(f'Start: {len(jobs)} files; workers={args.workers}; gemmi={gemmi.__version__}')
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
                'comparison': 'SHA256 of exact local bytes vs freshly decompressed current RCSB CIF',
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
