#!/usr/bin/env python3
"""Read-only, resumable integrity audit of Data_V2 decoys in all four splits.

Requires the server's existing numpy and gemmi. No Protenix model/search/GPU is
used. Full confidence JSON matrices are decoded one row at a time, never loaded
as a whole. --resume reuses PASS results only when all input fingerprints match.
"""
from __future__ import annotations

import argparse
import codecs
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import signal
import sqlite3
import sys
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HOME = Path('/storage9920/home/tinghao.xia')
VERSION = '1.0-full-decoy-audit'
SPLITS = ('train', 'val', 'test', 'holdout_foldbench_val_homology')
CATALOG_SOURCE = 'DATA_V2_FOLDBENCH_TEST_OVERRIDE_20260930T081218Z/final_manifest.tsv'
CATALOG_SOURCE_SHA256 = 'e36ff665557c55f1e3db1d03f11e73058776a96e822e04a26d15fae20db0671e'
CATALOG = {
    'train': """
        17RA 1A51 1A60 1A9L 1AFX 1AJF 1AJU 1AKX 1AM0 1ANR 1AQO 1ARJ 1ATO 1ATV 1ATW 1B36 1BGZ 1BN0 1BVJ 1BYJ 1BZ2 1BZ3 1BZT 1BZU
        1C0O 1C2W 1C2X 1CQ5 1CQL 1D0T 1D0U 1DUH 1E4P 1E95 1EBQ 1EBR 1EBS 1EHT 1EHZ 1EI2 1ESH 1ESY 1EVV 1F1T 1F6X 1F6Z 1F78 1F79
        1F7F 1F7G 1F7H 1F7I 1F84 1F85 1F9L 1FEQ 1FHK 1FIR 1FL8 1FMN 1FQZ 1FYO 1FYP 1G2J 1HLX 1HS1 1HS2 1HS3 1HS4 1HS8 1HWQ 1I3X
        1I3Y 1I46 1I4B 1I4C 1I9V 1IDV 1IE1 1IE2 1IK1 1IKD 1J4Y 1JO7 1JOX 1JP0 1JTJ 1JTW 1JU7 1JUR 1JWC 1JZC 1K2G 1K4A 1K4B 1K5I
        1K6G 1K6H 1KAJ 1KFO 1KH6 1KKA 1KKS 1KOS 1KP7 1KPD 1KPY 1KPZ 1KXK 1L1W 1L2X 1L3D 1LC6 1LDZ 1LU3 1LUU 1LUX 1LVJ 1M5L 1M82
        1MDG 1ME1 1MFJ 1MFK 1MFY 1MNX 1MSY 1MT4 1N66 1N8X 1NA2 1NBK 1NBR 1NC0 1NEM 1NJN 1NZ1 1O15 1OQ0 1OSU 1OSW 1OW9 1P5M 1P5N
        1P5O 1P5P 1P79 1P9X 1PBR 1PJY 1Q75 1Q8N 1Q9A 1QC8 1QD3 1QWA 1QWB 1QZA 1QZB 1R2P 1R4H 1R7W 1R7Z 1RAW 1RFR 1RHT 1RMN 1RNG
        1RNK 1ROQ 1RXA 1S2F 1S34 1S9S 1SCL 1SLO 1SLP 1SY4 1SYZ 1SZY 1T28 1TBK 1TFN 1TJZ 1TLR 1TN1 1TN2 1TOB 1TRA 1TXS 1U2A 1U3K
        1U9S 1UTS 1UUD 1UUI 1UUU 1VOP 1VTQ 1WKS 1WTS 1WTT 1XHP 1XJR 1XSG 1XSH 1XST 1XSU 1XV6 1XWP 1XWU 1Y26 1Y27 1YFG 1YG3 1YG4
        1YLG 1YMO 1YN1 1YN2 1YNC 1YNE 1YNG 1YSV 1Z2J 1Z30 1Z31 1Z43 1Z58 1ZC5 1ZIF 1ZIG 1ZIH 255D 283D 28SP 28SR 2A2E 2A43 2A64
        2A9L 2AHT 2AP0 2AP5 2AU4 2AWQ 2B57 2B7G 2CD1 2CD3 2CD5 2CD6 2EES 2EET 2EEU 2EEV 2EEW 2ES5 2EUY 2EVY 2F87 2F88 2FDT 2FEY
        2G1G 2G1W 2G9C 2GBH 2GIO 2GIP 2GIS 2GRW 2GV3 2GV4 2GVO 2HEM 2HNS 2HOJ 2HOK 2HOL 2HOM 2HOO 2HOP 2HUA 2IXY 2IXZ 2JJA 2JR4
        2JRG 2JRQ 2JSE 2JSG 2JTP 2JUK 2JWV 2JXV 2JYM 2K4C 2K5Z 2K66 2K95 2K96 2KD8 2KE6 2KEZ 2KF0 2KGP 2KHY 2KOC 2KPC 2KPD 2KPV
        2KRL 2KRP 2KRQ 2KRV 2KRW 2KRY 2KRZ 2KTZ 2KU0 2KUR 2KUU 2KUV 2KUW 2KVN 2KX8 2KXM 2KYE 2KZL 2L1V 2L2J 2L3E 2L5Z 2L6I 2L8H
        2L94 2L9E 2LA9 2LAC 2LBJ 2LBK 2LBL 2LBQ 2LBR 2LC8 2LDL 2LDT 2LDZ 2LHP 2LI4 2LJJ 2LK3 2LKR 2LP9 2LPA 2LPS 2LPT 2LQZ 2LU0
        2LUB 2LUN 2LV0 2LWK 2M12 2M21 2M22 2M24 2M4Q 2M4W 2M57 2M58 2M5U 2M8K 2MEQ 2MER 2MFD 2MHI 2MIS 2MIY 2MN0 2MNC 2MQT 2MTJ
        2MTK 2MXJ 2MXL 2MXS 2N0J 2N0R 2N1Q 2N2O 2N2P 2N3Q 2N3R 2N4L 2N6S 2N6T 2N6W 2N6X 2N7M 2N7X 2N8V 2NBX 2NBY 2NBZ 2NC0 2NC1
        2NCI 2O32 2O33 2O43 2O44 2O45 2OJ7 2OJ8 2PCV 2Q1R 2QBZ 2QH2 2QH3 2QH4 2RLU 2RO2 2RP0 2RP1 2RPK 2RPT 2RRC 2RVO 2TOB 2TPK
        2TRA 2U2A 2WNA 2XC6 2XNW 2XNZ 2XO0 2XO1 2Y95 2YDH 2YGH 2Z9Q 2ZY6 333D 387D 3A3A 3BBV 3BNT 3BWP 3CW5 3CW6 3CZW 3D0M 3D0U
        3D0X 3DHS 3DIG 3DIL 3DIM 3DIO 3DIQ 3DIR 3DIS 3DIX 3DIY 3DIZ 3DJ0 3DJ2 3DVZ 3DW4 3DW5 3DW6 3DW7 3E5C 3E5E 3E5F 3F2Q 3F2T
        3F2W 3F2X 3F2Y 3F30 3FO4 3FO6 3G4M 3GAO 3GCA 3GER 3GES 3GOG 3GOT 3GX2 3GX3 3GX5 3GX6 3GX7 3HGA 3IQN 3IQP 3IQR 3IZD 3J28
        3J29 3J2A 3J2B 3J2D 3J2E 3J2F 3J2G 3J2H 3K1V 3L0U 3LA5 3MIJ 3ND3 3ND4 3NJ6 3NPN 3PHP 3Q50 3Q51 3R4F 3S7C 3S8U 3SD3 3SUH
        3SUX 3SUY 3T4B 3TRA 3VRS 413D 430D 437D 480D 483D 4A4R 4A4S 4A4T 4A4U 4AOB 4B5R 4C40 4C4Q 4CS1 4DS6 4E8M 4E8N 4E8P 4E8Q
        4E8R 4E8V 4EN5 4ENA 4ENB 4ENC 4ERJ 4ERL 4FAQ 4FAU 4FAX 4FB0 4FE5 4FEJ 4FEL 4FEN 4FEO 4FEP 4FNJ 4GMA 4GXY 4JF2 4JIY 4K27
        4KQY 4L81 4LVV 4LVW 4LVX 4LVY 4LVZ 4LW0 4LX5 4LX6 4MS9 4NLF 4NMG 4NXH 4NYB 4NYC 4NYD 4NYG 4OJI 4OQU 4P5J 4P8Z 4P9R 4PCJ
        4PQV 4QK8 4QK9 4QKA 4QLM 4QLN 4R4P 4R4V 4RBQ 4RBY 4RBZ 4RC0 4RUM 4RZD 4TNA 4TRA 4TZX 4TZY 4WFL 4XNR 4XW1 4XW7 4XWF 4Y27
        5A17 5A18 5C5W 5C7U 5C7W 5D99 5DA6 5DO5 5DUN 5FJC 5FK1 5FK2 5FK3 5FK4 5FK5 5FK6 5FKD 5FKE 5FKF 5FKG 5FKH 5G2Y 5HBY 5IEM
        5J01 5K8H 5KH8 5KMZ 5KPY 5KQE 5L4O 5LSN 5LWJ 5M0H 5MRX 5MWI 5N5C 5NEF 5NEO 5NEP 5NEQ 5NEX 5NOM 5NQI 5NXT 5NZ6 5OB3 5SWE
        5T83 5TPY 5U3G 5UF3 5UZA 5UZT 5V0K 5V16 5V17 5V3I 5WQ1 5XWG 5Z1I 6AAS 6AAU 6BY4 6BY5 6CU1 6D3P 6DLQ 6DLR 6DLS 6DLT 6DME
        6DNR 6DVK 6E1S 6E1T 6E1U 6E1V 6E1W 6E7L 6E80 6E81 6E82 6E84 6E8U 6EZ0 6FZ0 6GYV 6GZK 6GZR 6HAG 6HBT 6HBX 6HC5 6HYK 6IZP
        6JJH 6JJI 6JQ6 6JXM 6K84 6MCI 6MXQ 6N5K 6N5N 6N5O 6N5P 6N5Q 6N5S 6N5T 6NOA 6OL3 6P2H 6PK9 6PQ7 6Q57 6Q8V 6QIV 6T3N 6T3S
        6TB7 6TF0 6TF1 6TF2 6TF3 6TFE 6TFF 6TFG 6TFH 6TNA 6U79 6UBU 6UC7 6UC8 6UC9 6UES 6UET 6UGI 6UGJ 6VAR 6VMY 6VU1 6VUH 6VUI
        6VVJ 6VZC 6W3M 6WJR 6WJS 6WLJ 6WLK 6WLL 6WLM 6WLN 6WLO 6WLQ 6WLR 6WLS 6WLT 6WLU 6WTL 6WTR 6XB7 6XRZ 6XUR 6XUS 6XWJ 6XWW
        6XXA 6XXB 6Y0Y 6Y3G 6ZYB 7D12 7D7W 7D7X 7D7Y 7D7Z 7D81 7D82 7DD4 7EAF 7EZ0 7JJD 7JJE 7JJF 7JRR 7JU1 7K16 7K4L 7KD1 7KJU
        7L0Z 7L3R 7LVA 7LYF 7LYG 7LYJ 7MKY
    """.split(),
    'val': """
        6XKN 6XKO 7E9E 7E9I 7EDN 7EEM 7EFG 7EFH 7EOG 7EOH 7EOI 7EOJ 7EOK 7EOL 7EOM 7EON 7EOO 7EOP 7EXY 7FHI 7FJ0 7KUB 7KUC 7KUD
        7KVT 7KVU 7KVV 7M3V 7MKT 7MLW 7POF 7PS8 7PTK 7PTL 7PTQ 7PTS 7Q48 7Q6L 7QA2 7QDU 7QP2 7QSH 7QUA 7R6N 7RWR 7SAM 7SHX 7TD7
        7TDA 7TDB 7TDC 7TZT 7TZU 7U4A 7UCR 7UGA 7UMC 7UMD 7UME 7UQ6 7URI 7URM 7UVT 7UZ0 7V06 7V9E 7VFT 7XD3 7XD4 7XD7 7XSK 7XSL
        7XSM 7XSN 7Y2B 7YC8 7ZFW 8BTZ 8BWT 8CLR 8CQ1 8FCS 8FW4 8HD6 8HD7 8I43 8I44 8I45 8I46 8I7N 8S95 8SCF 8SCH 8SP9 8THV 8TJQ
        8TJV 8TJX 8TNS 8U3M 8UYE 8UYG 8UYJ 8UYK 8UYL 8UYM 8UYP 8UYS
    """.split(),
    'test': """
        10ZT 23JZ 30SC 7SXP 7WIA 7WII 7ZJ4 8BU8 8HB8 8ITS 8JHP 8K15 8K7W 8Q4O 8Q6N 8QO2 8QO3 8T5O 8TVZ 8UPT 8UPY 8V1H 8VCI 8VT5
        8XTS 8XZO 8ZNQ 9BCI 9BLM 9BZC 9C6J 9CXF 9DE7 9E74 9E75 9ECQ 9ELY 9EOW 9EWW 9FM4 9FO8 9FO9 9G4I 9G4R 9G7C 9HRF 9HRO 9IOS
        9ISV 9J4N 9J4O 9K9G 9K9K 9KGG 9L8F 9LEC 9LEL 9LHK 9LKE 9M79 9MFH 9MQT 9NAM 9NBC 9NDV 9OBM 9OD4 9QTJ 9QZJ 9R7Q 9R7W 9R82
        9SY8 9UUG 9UW0 9W1L 9X6B 9Z6I 9ZC9
    """.split(),
    'holdout_foldbench_val_homology': """
        7WI9 7WIB 7WIE 7WIF 7ZJ5
    """.split(),
}
SEEDS = tuple(range(300, 350))
SAMPLES = tuple(range(4))
ID_RE = re.compile(r'^[A-Za-z0-9]{4}$')
SEED_RE = re.compile(r'^seed_(\d+)$')
OUTPUT_RE = re.compile(r'^([A-Za-z0-9]{4})_(sample|summary_confidence_sample|full_data_sample)_(\d+)\.(cif|json)$', re.I)
UNRESOLVED_RE = re.compile(r'^[A-Za-z0-9]{4}_sample_\d+_wounresol\.cif$', re.I)
SAMPLE_FIELDS = ['SPLIT','PDB_ID','SEED','SAMPLE','STATUS','ISSUES','WARNINGS','N_ATOMS','N_TOKENS',
    'RANKING_SCORE','ATOM_IDENTITY_SHA256','COORDINATE_SHA256','CIF_SHA256','SUMMARY_SHA256','FULL_SHA256',
    'FULL_BYTES','MATRIX_VALUES_CHECKED','B_PLDDT_MAX_DIFF','CHECKED_AT_UTC']
SEED_FIELDS = ['SPLIT','PDB_ID','SEED','STATUS','ISSUES','WARNINGS','PREDICTIONS_PATH',
    'RESOLVED_PREDICTIONS_PATH','PASS_SAMPLES','PROBLEM_SAMPLES','MISSING_SAMPLES','REVIEW_SAMPLES']
PDB_FIELDS = ['SPLIT','PDB_ID','STATUS','ISSUES','WARNINGS','IS_SYMLINK','RESOLVED_PATH',
    'FOUND_SEEDS','PASS_SEEDS','PASS_SAMPLES','PROBLEM_SAMPLES','MISSING_SAMPLES','REVIEW_SAMPLES']
NP = GEMMI = None


class Interrupted(Exception): pass
class ResourceLimit(Exception): pass


def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')
def stable(value): return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def sha_text(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()


def atomic(path,text):
    tmp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    try:
        tmp.write_text(text,encoding='utf-8');os.replace(tmp,path)
    finally: tmp.unlink(missing_ok=True)


def unique_keys(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('duplicate JSON key: '+key)
        result[key]=value
    return result


def bad_constant(value): raise ValueError('invalid JSON number: '+value)
DECODER=json.JSONDecoder(object_pairs_hook=unique_keys,parse_constant=bad_constant)


def small_json(path):
    data=path.read_bytes()
    value=json.loads(data.decode('utf-8-sig'),object_pairs_hook=unique_keys,parse_constant=bad_constant)
    return value,hashlib.sha256(data).hexdigest()


def fingerprint(path):
    resolved=path.resolve(strict=True)
    if not resolved.is_file(): raise ValueError('not a regular file: '+str(path))
    stat=resolved.stat()
    return [str(resolved),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]


def state(issues,reviews=(),missing=False):
    return 'PROBLEM' if issues else 'MISSING' if missing else 'REVIEW' if reviews else 'PASS'


def aggregate(statuses):
    values=set(statuses)
    return next((x for x in ('PROBLEM','MISSING','REVIEW','PASS') if x in values),'REVIEW')


def add(items,message):
    if message not in items and len(items)<25: items.append(message)


class StreamJSON:
    """Strict JSON envelope, C JSON decoder for bounded numeric vectors/rows."""
    def __init__(self,stream,stop,max_value_bytes=64*1024*1024):
        self.stream,self.stop,self.max_value_bytes=stream,stop,max_value_bytes
        self.buf='';self.pos=0;self.eof=False
        self.digest=hashlib.sha256();self.bytes=0
        self.decoder=codecs.getincrementaldecoder('utf-8')('strict')

    def fill(self):
        if self.stop.is_set(): raise Interrupted()
        if self.eof:return False
        self.buf=self.buf[self.pos:];self.pos=0
        chunk=self.stream.read(1024*1024)
        self.digest.update(chunk);self.bytes+=len(chunk)
        self.eof=not chunk
        self.buf+=self.decoder.decode(chunk,final=self.eof)
        return bool(chunk)

    def peek(self):
        while True:
            while self.pos<len(self.buf) and self.buf[self.pos] in ' \r\n\t': self.pos+=1
            if self.pos<len(self.buf):return self.buf[self.pos]
            if not self.fill():return ''

    def expect(self,char):
        got=self.peek()
        if got!=char:raise ValueError(f'JSON expected {char!r}, got {got!r}')
        self.pos+=1

    def small(self):
        self.peek()
        while True:
            try:
                value,end=DECODER.raw_decode(self.buf,self.pos)
                # Require a complete token even at a read-buffer boundary.
                if end==len(self.buf) and not self.eof:
                    self.fill();continue
                if end<len(self.buf) and self.buf[end] not in ' \r\n\t,]}:':
                    if not self.eof:
                        self.fill()
                        if len(self.buf)>self.max_value_bytes:raise ResourceLimit('JSON value exceeds memory limit')
                        continue
                    raise ValueError('invalid JSON token delimiter')
                if end-self.pos>self.max_value_bytes:raise ResourceLimit('JSON value exceeds memory limit')
                self.pos=end
                return value
            except json.JSONDecodeError:
                if self.eof:raise
                if len(self.buf)-self.pos>self.max_value_bytes:raise ResourceLimit('JSON value exceeds memory limit')
                self.fill()

    def array(self,read_value):
        self.expect('[');count=0
        if self.peek()==']':self.pos+=1;return 0
        while True:
            read_value(count);count+=1
            separator=self.peek()
            if separator==']':self.pos+=1;return count
            self.expect(',')
            if self.peek()==']':raise ValueError('trailing JSON array comma')

    def obj(self,read_value,depth=0):
        if depth>32:raise ResourceLimit('JSON nesting exceeds limit')
        self.expect('{');seen=set()
        if self.peek()=='}':self.pos+=1;return
        while True:
            key=self.small()
            if not isinstance(key,str):raise ValueError('JSON object key must be a string')
            if key in seen:raise ValueError('duplicate JSON key: '+key)
            seen.add(key);self.expect(':');read_value(key)
            if self.peek()=='}':self.pos+=1;return
            self.expect(',')
            if self.peek()=='}':raise ValueError('trailing JSON object comma')

    def skip(self,depth=0):
        if depth>32:raise ResourceLimit('JSON nesting exceeds limit')
        char=self.peek()
        if char=='[':self.array(lambda index:self.skip(depth+1))
        elif char=='{':self.obj(lambda key:self.skip(depth+1),depth+1)
        else:
            value=self.small()
            if type(value)in {int,float} and not math.isfinite(value):raise ValueError('nonfinite numeric JSON value')

    def finish(self):
        if self.peek():raise ValueError('trailing data after JSON object')
        if not self.eof:raise ValueError('JSON was not consumed to EOF')


def numbers(values,label,low=None,high=None,integer=False,boolean=False):
    if not isinstance(values,list):raise ValueError(label+' must be an array')
    allowed={int,float} if not integer else {int}
    if boolean:allowed={bool,int}
    if not set(map(type,values))<=allowed:raise ValueError(label+' contains invalid numeric types')
    arr=NP.asarray(values,dtype=NP.int64 if integer or boolean else NP.float64)
    if not NP.isfinite(arr).all():raise ValueError(label+' contains nonfinite values')
    if low is not None and (arr<low-1e-7).any():raise ValueError(label+' below allowed range')
    if high is not None and (arr>high+1e-7).any():raise ValueError(label+' above allowed range')
    return arr


def full_confidence(path,stop,max_value_bytes):
    required={'atom_plddt','token_pair_pae','token_pair_pde','contact_probs','atom_to_token_idx'}
    vectors={};matrices={};unknown=[];checked=0
    with path.open('rb') as f:
        reader=StreamJSON(f,stop,max_value_bytes)
        def read_field(key):
            nonlocal checked
            if key in {'token_pair_pae','token_pair_pde','contact_probs'}:
                width=None
                def read_row(index):
                    nonlocal width,checked
                    row=numbers(reader.small(),key,low=0,high=1 if key=='contact_probs' else None)
                    if width is None:width=len(row)
                    if not width or len(row)!=width:raise ValueError(key+' has inconsistent row lengths')
                    checked+=len(row)
                rows=reader.array(read_row)
                if not rows or rows!=width:raise ValueError(key+' is not a nonempty square matrix')
                matrices[key]=rows
            elif key in {'atom_plddt','atom_to_token_idx','token_has_frame','token_asym_id'}:
                vectors[key]=numbers(reader.small(),key,low=0,
                    high=1 if key in {'atom_plddt','token_has_frame'} else None,
                    integer=key in {'atom_to_token_idx','token_asym_id'},boolean=key=='token_has_frame')
            else:
                unknown.append(key);reader.skip()
        reader.obj(read_field);reader.finish()
        missing=required-set(vectors)-set(matrices)
        if missing:raise ValueError('missing full-confidence fields: '+','.join(sorted(missing)))
        dims=set(matrices.values())
        if len(dims)!=1:raise ValueError('PAE/PDE/contact matrix dimensions differ')
        nt=dims.pop();na=len(vectors['atom_plddt']);mapping=vectors['atom_to_token_idx']
        if not na or len(mapping)!=na:raise ValueError('atom_plddt/atom_to_token_idx lengths differ or are empty')
        if (mapping>=nt).any():raise ValueError('atom_to_token_idx outside matrix token range')
        if not NP.array_equal(NP.unique(mapping),NP.arange(nt)):raise ValueError('atom_to_token_idx does not cover exactly tokens 0..N-1')
        for key in ('token_has_frame','token_asym_id'):
            if key in vectors and len(vectors[key])!=nt:raise ValueError(key+' length differs from matrix token count')
        return dict(vectors=vectors,n_atoms=na,n_tokens=nt,sha256=reader.digest.hexdigest(),
                    bytes=reader.bytes,matrix_values=checked,unknown=unknown)


def expected_input(path):
    payload,digest=small_json(path)
    if not isinstance(payload,list) or len(payload)!=1 or not isinstance(payload[0],dict):raise ValueError('raw JSON must contain one task')
    task=payload[0]
    if str(task.get('name','')).upper()!=path.stem.upper():raise ValueError('raw JSON task name mismatch')
    if not isinstance(task.get('sequences'),list) or not task['sequences']:raise ValueError('raw JSON has no sequences')
    expected=Counter()
    for item in task['sequences']:
        if not isinstance(item,dict) or len(item)!=1:raise ValueError('invalid input entity')
        kind,body=next(iter(item.items()));count=body.get('count',1)
        if type(count)is not int or count<1:raise ValueError('invalid input count')
        if kind=='rnaSequence':
            seq=body.get('sequence','')
            if not isinstance(seq,str) or not re.fullmatch('[A-Z]+',seq):raise ValueError('invalid RNA sequence')
            residues=list(seq);seen=set()
            for mod in body.get('modifications',[]):
                pos,code=mod.get('basePosition'),mod.get('modificationType')
                if type(pos)is not int or not 1<=pos<=len(seq) or pos in seen or not isinstance(code,str) or not re.fullmatch('CCD_[A-Za-z0-9]+',code):raise ValueError('invalid RNA modification')
                residues[pos-1]=code[4:];seen.add(pos)
            expected[('rnaSequence',tuple(residues))]+=count
        elif kind=='ligand':
            code=body.get('ligand','')
            if not isinstance(code,str) or not code.startswith('CCD_'):raise ValueError('only CCD ligands are supported for reference comparison')
            expected[('ligand',tuple(code[4:].split('_')))]+=count
        else:raise ValueError('non-RNA polymer in pure-RNA target')
    return {'entities':expected,'sha256':digest}


def cif_data(path,pdb_id,expected):
    data=path.read_bytes();document=GEMMI.cif.read_string(data.decode('utf-8'),check_level=2)
    if len(document)!=1:raise ValueError('CIF must have exactly one data block')
    block=document[0]
    entry=block.find_value('_entry.id')
    if not entry or GEMMI.cif.as_string(entry).upper()!=pdb_id:raise ValueError('CIF entry ID mismatch')
    atoms=block.get_mmcif_category('_atom_site.')
    names=['id','type_symbol','label_atom_id','label_comp_id','label_asym_id','label_entity_id','Cartn_x','Cartn_y','Cartn_z','B_iso_or_equiv']
    if any(key not in atoms for key in names):raise ValueError('CIF lacks required atom-site fields')
    n=len(atoms['id'])
    if not n or any(len(value)!=n for value in atoms.values()):raise ValueError('CIF atom-site table is empty/inconsistent')
    if len(set(atoms['id']))!=n:raise ValueError('duplicate CIF atom IDs')
    if 'pdbx_PDB_model_num' in atoms and len(set(atoms['pdbx_PDB_model_num']))!=1:raise ValueError('predicted CIF has multiple models')
    coord=NP.asarray([[float(x),float(y),float(z)] for x,y,z in zip(atoms['Cartn_x'],atoms['Cartn_y'],atoms['Cartn_z'])])
    b=NP.asarray([float(x) for x in atoms['B_iso_or_equiv']])
    if not NP.isfinite(coord).all() or not NP.isfinite(b).all():raise ValueError('nonfinite CIF coordinates/B factors')
    if (b<-.01).any() or (b>100.01).any():raise ValueError('predicted CIF B-factor/pLDDT outside 0..100')
    if 'occupancy' in atoms:
        occ=NP.asarray([float(x) for x in atoms['occupancy']])
        if not NP.isfinite(occ).all() or (occ<0).any() or (occ>1.00001).any():raise ValueError('invalid CIF occupancy')
    chains=atoms['label_asym_id'];entities=atoms['label_entity_id'];comp=atoms['label_comp_id']
    label=atoms.get('label_seq_id',['.']*n);auth=atoms.get('auth_seq_id',['.']*n)
    alt=atoms.get('label_alt_id',['.']*n);ins=atoms.get('pdbx_PDB_ins_code',['?']*n)
    identities=[];residue_keys=[];chain_residues=defaultdict(dict);chain_entities={}
    for i in range(n):
        if not all(isinstance(atoms[key][i],str) and atoms[key][i] not in {'','?','.'} for key in ['label_atom_id','label_comp_id','label_asym_id','label_entity_id','type_symbol']):raise ValueError('empty CIF atom identity field')
        seq_id=label[i] if label[i] not in (None,False,'.','?','') else auth[i]
        if seq_id in (None,False,'.','?',''):raise ValueError('CIF residue has no usable numbering')
        rkey=(chains[i],str(seq_id),str(ins[i]),comp[i]);residue_keys.append(rkey)
        identities.append(rkey+(atoms['label_atom_id'][i],str(alt[i]),atoms['type_symbol'][i]))
        prior=chain_residues[chains[i]].setdefault((str(seq_id),str(ins[i])),comp[i])
        if prior!=comp[i]:raise ValueError('multiple residue identities at one predicted residue position')
        if chain_entities.setdefault(chains[i],entities[i])!=entities[i]:raise ValueError('one predicted chain has multiple entity IDs')
    if len(set(identities))!=n:raise ValueError('duplicate predicted atom identities')
    poly=block.get_mmcif_category('_entity_poly.')
    poly_types=dict(zip(poly.get('entity_id',[]),poly.get('type',[])))
    observed=Counter();declared=block.get_mmcif_category('_entity_poly_seq.');declared_by_entity=defaultdict(dict)
    for entity,num,mon in zip(declared.get('entity_id',[]),declared.get('num',[]),declared.get('mon_id',[])):
        position=int(num)
        if position in declared_by_entity[entity]:raise ValueError('duplicate polymer sequence position')
        declared_by_entity[entity][position]=mon
    for chain,residues in chain_residues.items():
        def order(item):
            pos,_=item[0]
            try:return (0,int(pos),item[0][1])
            except ValueError:return (1,pos,item[0][1])
        sequence=tuple(value for _,value in sorted(residues.items(),key=order))
        entity=chain_entities[chain];kind=poly_types.get(entity)
        if kind=='polyribonucleotide':
            observed[('rnaSequence',sequence)]+=1
            if not declared_by_entity[entity]:raise ValueError('RNA entity lacks declared polymer sequence')
            declared_sequence=tuple(v for _,v in sorted(declared_by_entity[entity].items()))
            if sequence!=declared_sequence:raise ValueError('coordinate RNA residues differ from declared polymer sequence')
        elif kind is None:observed[('ligand',sequence)]+=1
        else:raise ValueError('non-RNA polymer in predicted CIF')
    if expected is not None and observed!=expected['entities']:raise ValueError('CIF residue sequences/modifications/entity copies differ from raw input JSON')
    warnings=[]
    if n>3 and float(NP.ptp(coord,axis=0).max())<1e-6:warnings.append('ALL_COORDINATES_COLLAPSED')
    return dict(n_atoms=n,b=b,chains=chains,residues=residue_keys,sha256=hashlib.sha256(data).hexdigest(),
                identity_sha256=sha_text(stable(identities)),coordinate_sha256=hashlib.sha256(coord.astype('<f8').tobytes()).hexdigest(),warnings=warnings)


def check_summary(path,n_chains):
    obj,digest=small_json(path)
    if not isinstance(obj,dict):raise ValueError('summary must be a JSON object')
    if n_chains is None:
        candidates=[len(v) for k,v in obj.items() if k.startswith('chain_') and not k.startswith('chain_pair_') and isinstance(v,list)]
        n_chains=candidates[0] if candidates else None
    def visit(value,key=''):
        if isinstance(value,dict):
            for k,v in value.items():visit(v,k)
        elif isinstance(value,list):
            for v in value:visit(v,key)
        elif type(value)in {int,float} and not math.isfinite(value):raise ValueError('nonfinite summary value: '+key)
    visit(obj)
    for key in ['ranking_score','plddt','ptm','iptm']:
        if key not in obj or type(obj[key])not in {int,float}:raise ValueError('missing/non-numeric summary '+key)
    if not 0<=obj['plddt']<=100:raise ValueError('summary plddt outside 0..100')
    for key in ['ptm','iptm','disorder']:
        if key in obj and (type(obj[key])not in {int,float} or not 0<=obj[key]<=1):raise ValueError('summary '+key+' outside 0..1')
    if 'has_clash' in obj and type(obj['has_clash'])is not bool:raise ValueError('summary has_clash is not boolean')
    if 'num_recycles' in obj and (type(obj['num_recycles'])is not int or obj['num_recycles']<0):raise ValueError('summary num_recycles is not a nonnegative integer')
    if 'gpde' in obj and (type(obj['gpde'])not in {int,float} or obj['gpde']<0):raise ValueError('invalid summary gpde')
    for key,value in obj.items():
        if key.startswith('chain_pair_'):
            if not isinstance(value,list) or (n_chains is not None and len(value)!=n_chains):raise ValueError(key+' chain matrix dimension mismatch')
            for v in value:
                arr=numbers(v,key,low=0,high=None if key.endswith('gpde') else 1)
                if len(arr)!=len(value):raise ValueError(key+' chain matrix is not square')
        elif key.startswith('chain_'):
            arr=numbers(value,key,low=0,high=None if key.endswith('gpde') else 1)
            if n_chains is not None and len(arr)!=n_chains:raise ValueError(key+' chain vector dimension mismatch')
    return obj,digest


def sample_audit(split,pdb_id,seed,index,paths,expected,stop,max_value_bytes):
    row=dict.fromkeys(SAMPLE_FIELDS,'');row.update(SPLIT=split,PDB_ID=pdb_id,SEED=seed,SAMPLE=index)
    issues,reviews,warnings=[],[],[];missing=False;cif=full=summary=None
    before={}
    for kind,path in paths.items():
        try:
            if path is None:raise FileNotFoundError('required output missing')
            before[kind]=fingerprint(path)
            if not before[kind][3]:raise ValueError('empty file')
            if kind=='cif':
                cif=cif_data(path,pdb_id,expected)
                row.update(N_ATOMS=cif['n_atoms'],CIF_SHA256=cif['sha256'],ATOM_IDENTITY_SHA256=cif['identity_sha256'],
                           COORDINATE_SHA256=cif['coordinate_sha256'])
                warnings+=cif['warnings']
            elif kind=='full':
                full=full_confidence(path,stop,max_value_bytes)
                row.update(N_TOKENS=full['n_tokens'],FULL_SHA256=full['sha256'],FULL_BYTES=full['bytes'],MATRIX_VALUES_CHECKED=full['matrix_values'])
                if full['unknown']:warnings.append('ADDITIONAL_FULL_FIELDS='+','.join(full['unknown']))
            # Summary is checked after CIF, regardless of any earlier CIF error.
        except Interrupted:raise
        except FileNotFoundError:
            missing=True;add(warnings,'MISSING_'+kind.upper())
            if path is not None and path.is_symlink():add(warnings,'BROKEN_SYMLINK_'+kind.upper())
        except ResourceLimit as error:add(reviews,kind.upper()+'_RESOURCE_LIMIT: '+str(error))
        except Exception as error:add(issues,kind.upper()+'_INVALID: '+type(error).__name__+': '+str(error))
    if paths['summary'] is not None and 'summary' in before:
        try:
            summary,sha=check_summary(paths['summary'],len(set(cif['chains'])) if cif else None)
            row.update(RANKING_SCORE=summary['ranking_score'],SUMMARY_SHA256=sha)
        except Exception as error:add(issues,'SUMMARY_INVALID: '+type(error).__name__+': '+str(error))
    if cif is not None and full is not None:
        try:
            if cif['n_atoms']!=full['n_atoms']:raise ValueError('CIF atom count differs from full-confidence atom vectors')
            mapping=full['vectors']['atom_to_token_idx'];plddt=full['vectors']['atom_plddt']
            max_diff=float(NP.abs(cif['b']-100*plddt).max())
            row['B_PLDDT_MAX_DIFF']=round(max_diff,6)
            if max_diff>0.511:raise ValueError('CIF B-factor vs full atom_plddt differs beyond rounding tolerance (0.511/100)')
            token_residue={};token_chain={}
            for i,token in enumerate(mapping.tolist()):
                if token_residue.setdefault(token,cif['residues'][i])!=cif['residues'][i]:raise ValueError('one token maps atoms from different CIF residues')
                token_chain[token]=cif['chains'][i]
            if 'token_asym_id' in full['vectors']:
                asym=full['vectors']['token_asym_id'];chain_to_ids=defaultdict(set)
                for token,chain in token_chain.items():chain_to_ids[chain].add(int(asym[token]))
                if any(len(v)!=1 for v in chain_to_ids.values()) or len({next(iter(v)) for v in chain_to_ids.values()})!=len(chain_to_ids):raise ValueError('token_asym_id chain assignments disagree with CIF')
        except Exception as error:add(issues,'CROSS_FILE_MISMATCH: '+str(error))
    if expected is None:reviews.append('RAW_INPUT_REFERENCE_UNAVAILABLE_SEQUENCE_NOT_VERIFIED')
    for kind,before_value in before.items():
        try:
            if fingerprint(paths[kind])!=before_value:add(reviews,'SOURCE_CHANGED_DURING_CHECK: '+kind)
        except OSError:add(reviews,'SOURCE_DISAPPEARED_DURING_CHECK: '+kind)
    row.update(STATUS=state(issues,reviews,missing),ISSUES='; '.join(issues+reviews),WARNINGS='; '.join(warnings),CHECKED_AT_UTC=now())
    return row,before


class ResultCache:
    def __init__(self,path):
        self.db=sqlite3.connect(str(path))
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS samples (key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT NOT NULL)')

    def get(self,key,stamp):
        found=self.db.execute('SELECT fingerprint,result FROM samples WHERE key=?',(key,)).fetchone()
        if found and found[0]==stable(stamp):
            row=json.loads(found[1])
            if row['STATUS']=='PASS':return row
        return None

    def put(self,key,stamp,row):
        if row['STATUS']=='PASS':
            self.db.execute('INSERT OR REPLACE INTO samples VALUES (?,?,?)',(key,stable(stamp),stable(row)))

    def commit(self):self.db.commit()
    def close(self):self.db.commit();self.db.close()


def lock_report(report):
    handle=(report/'.audit.lock').open('a+')
    try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except Exception:
        handle.close();raise
    return handle


def classify_files(folder,pdb_id):
    index=defaultdict(list);issues=[];warnings=[]
    if not folder.is_dir():return index,issues,warnings
    for path in sorted(folder.iterdir()):
        match=OUTPUT_RE.fullmatch(path.name)
        if match:
            stem,kind,num,ext=match.groups();kind=kind.lower();num=int(num)
            wanted_ext='cif' if kind=='sample' else 'json'
            if stem.upper()!=pdb_id:add(issues,'WRONG_PDB_PREFIX: '+path.name)
            elif ext.lower()!=wanted_ext:add(issues,'WRONG_OUTPUT_EXTENSION: '+path.name)
            elif num not in SAMPLES:add(issues,'UNEXPECTED_SAMPLE_INDEX: '+path.name)
            else:
                label={'sample':'cif','summary_confidence_sample':'summary','full_data_sample':'full'}[kind]
                index[(label,num)].append(path)
        elif UNRESOLVED_RE.fullmatch(path.name):
            add(warnings,'OPTIONAL_WOUNRESOL_VARIANTS_PRESENT_NOT_COUNTED_AS_DECOYS')
        elif not path.name.startswith('.'):
            add(warnings,'UNRECOGNIZED_EXTRA_ENTRY: '+path.name)
    for (kind,num),paths in index.items():
        if len(paths)>1:add(issues,f'DUPLICATE_OUTPUT: {kind} sample={num}')
    return index,issues,warnings


def resolve_paths(index,num):
    return {kind:index[(kind,num)][0] if len(index[(kind,num)])==1 else None for kind in ('cif','summary','full')}


def load_manifest(path):
    expected={split:set() for split in SPLITS}
    if path is None:
        return {key:set(value) for key,value in CATALOG.items()}
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t')
        for row in reader:
            if row.get('FINAL_STATUS')=='KEPT':split=row.get('FINAL_SPLIT','')
            elif row.get('FINAL_STATUS')=='HOLDOUT_VAL_HOMOLOG':split=SPLITS[3]
            elif 'SPLIT' in row and 'PDB_ID' in row:split=row['SPLIT']
            else:continue
            pdb_id=row['PDB_ID'].upper()
            if split not in expected or not ID_RE.fullmatch(pdb_id):raise ValueError('invalid manifest split/ID')
            if any(pdb_id in values for values in expected.values()):raise ValueError('duplicate manifest PDB ID: '+pdb_id)
            expected[split].add(pdb_id)
    if not sum(map(len,expected.values())):raise ValueError('manifest has no targets')
    return expected


def main(argv=None):
    global NP,GEMMI
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=HOME/'Data_V2')
    p.add_argument('--input-json-dir',type=Path,default=HOME/'Json_data/Simple_json')
    p.add_argument('--manifest',type=Path,help='Optional replacement target manifest; default embeds the verified 2026-09-30 split including five holdouts')
    p.add_argument('--report-dir',type=Path,default=HOME/('decoy_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    p.add_argument('--resume',action='store_true',help='Reuse unchanged PASS samples from this report directory; recheck failures and changed files')
    p.add_argument('--max-value-mib',type=int,default=64,help='Memory limit for one JSON vector/matrix row, not the entire file')
    args=p.parse_args(argv)
    for key in ('data_root','input_json_dir','report_dir'):
        setattr(args,key,getattr(args,key).expanduser().resolve())
    if not args.data_root.is_dir():p.error('Data_V2 root does not exist')
    if args.max_value_mib<1:p.error('--max-value-mib must be positive')
    if args.report_dir.is_relative_to(args.data_root) or args.report_dir.is_relative_to(args.input_json_dir):p.error('report directory must be outside source directories')
    if args.report_dir.exists() and not args.resume:p.error('report already exists; use a new path or --resume')
    if args.resume and not (args.report_dir/'config.json').is_file():p.error('--resume requires an existing valid report/config.json')
    try:
        import numpy as np
        import gemmi
        NP,GEMMI=np,gemmi
        expected=load_manifest(args.manifest)
    except Exception as error:p.error(f'dependency/manifest error: {type(error).__name__}: {error}')
    script_digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config={'script_version':VERSION,'script_sha256':script_digest,'numpy_version':np.__version__,
        'gemmi_version':gemmi.__version__,'data_root':str(args.data_root),'input_json_dir':str(args.input_json_dir),
        'expected_catalog':{k:sorted(v) for k,v in expected.items()},'seeds':list(SEEDS),'samples':list(SAMPLES),
        'max_value_mib':args.max_value_mib,'catalog_source':str(args.manifest) if args.manifest else CATALOG_SOURCE,
        'catalog_source_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else CATALOG_SOURCE_SHA256}
    if args.resume:
        previous=json.loads((args.report_dir/'config.json').read_text())
        if previous!=config:p.error('resume configuration/script/dependency versions differ; choose a new report directory')
    args.report_dir.mkdir(parents=True,exist_ok=True)
    report=args.report_dir
    try:report_lock=lock_report(report)
    except BlockingIOError:p.error('another checker is already using this report directory')
    atomic(report/'config.json',json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    stop,done=threading.Event(),threading.Event()
    active={'target':'inventory','seed':'','sample':'','phase':'inventory'}
    counters=Counter();seed_counts=Counter();sample_counts=Counter();pdb_counts=Counter();split_counts=defaultdict(Counter)
    inventory=[];targets=[];pdb_results={}
    actual_sets={};reference_index=defaultdict(list)
    if args.input_json_dir.is_dir():
        for path in args.input_json_dir.iterdir():
            if re.fullmatch('[A-Za-z0-9]{4}\\.json',path.name,re.I):reference_index[path.stem.upper()].append(path)
    else:inventory.append('RAW_INPUT_JSON_DIRECTORY_MISSING')
    for split in SPLITS:
        folder=args.data_root/split;index=defaultdict(list)
        if not folder.is_dir():inventory.append('MISSING_SPLIT_DIRECTORY: '+split)
        else:
            for path in sorted(folder.iterdir()):
                if path.name.startswith('.'):continue
                if ID_RE.fullmatch(path.name):index[path.name.upper()].append(path)
                else:inventory.append('UNEXPECTED_SPLIT_ENTRY: '+str(path))
        actual_sets[split]=set(index)
        for identifier in sorted(expected[split]|set(index)):
            paths=index.get(identifier,[])
            targets.append((split,identifier,paths))
            if identifier not in expected[split]:inventory.append(f'UNEXPECTED_PDB_IN_SPLIT: {split}/{identifier}')
            if identifier in expected[split] and not paths:inventory.append(f'MISSING_PDB_DIRECTORY: {split}/{identifier}')
            if len(paths)>1:inventory.append(f'DUPLICATE_PDB_DIRECTORY: {split}/{identifier}')
    for path in args.data_root.iterdir():
        if path.is_dir() and path.name not in SPLITS:inventory.append('UNEXPECTED_ROOT_DIRECTORY: '+path.name)
    assigned=defaultdict(list)
    for split,ids in actual_sets.items():
        for identifier in ids:assigned[identifier].append(split)
    for identifier,splits in assigned.items():
        if len(splits)>1:inventory.append('PDB_PRESENT_IN_MULTIPLE_SPLITS: '+identifier+':'+','.join(splits))
    cache=ResultCache(report/'audit_state.sqlite3')
    started=now();fatal=''
    def log(message):print(f'[{now()}] {message}',flush=True)
    def checkpoint(status):
        pending=[identifier for split,identifier,_ in targets
                 if (split,identifier) not in pdb_results or pdb_results[(split,identifier)]['STATUS']!='PASS']
        atomic(report/'problem_pdb_ids.txt',''.join(x+'\n' for x in sorted(set(pending))))
        expected_pdbs=sum(map(len,expected.values()))
        payload={'status':status,'script_version':VERSION,'pid':os.getpid(),'started_at_utc':started,'updated_at_utc':now(),
            'resume':args.resume,'data_root':str(args.data_root),'report_dir':str(report),
            'expected_pdb_counts':{k:len(v) for k,v in expected.items()},
            'found_pdb_counts':{k:len(v) for k,v in actual_sets.items()},'expected_pdbs':expected_pdbs,
            'scheduled_pdbs':len(targets),'checked_pdbs':len(pdb_results),'unchecked_pdbs':len(targets)-len(pdb_results),
            'expected_seeds_per_pdb':len(SEEDS),'seed_range':[SEEDS[0],SEEDS[-1]],
            'expected_samples_per_seed':len(SAMPLES),'expected_decoys':expected_pdbs*len(SEEDS)*len(SAMPLES),
            'expected_required_files':expected_pdbs*len(SEEDS)*len(SAMPLES)*3,
            'pdb_counts':dict(pdb_counts),'seed_counts':dict(seed_counts),'sample_counts':dict(sample_counts),
            'split_counts':{k:dict(v) for k,v in split_counts.items()},'work_counters':dict(counters),
            'active':dict(active),'inventory_issues':inventory,'fatal_error':fatal,
            'all_expected_decoys_passed':status=='COMPLETED' and not inventory and len(pdb_results)==expected_pdbs
                and pdb_counts['PASS']==expected_pdbs and sample_counts['PASS']==expected_pdbs*len(SEEDS)*len(SAMPLES),
            'scope':'Read-only. Exact target/seed/sample inventory; full CIF parsing/finite coordinates and input residue/copy matching; '
                'strict whole-file JSON syntax, all matrix values/ranges/dimensions, atom/token mapping, CIF B-factor vs atom pLDDT, '
                'summary shapes and within-seed ranking. No native-coordinate RMSD/accuracy, stereochemical proof or model inference. '
                'All four splits including holdout; abandoned/previously failed targets are NOT exempt. '
                'Resume trusts a prior PASS only when script/config/input reference and source metadata fingerprints remain unchanged.'}
        atomic(report/'summary.json',json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    def heartbeat():
        while not done.wait(30):
            log(f'Heartbeat pdbs={len(pdb_results)}/{len(targets)} samples={sum(sample_counts.values())} '
                f'active={active["target"]} seed={active["seed"]} sample={active["sample"]} phase={active["phase"]}')
    handlers={sig:signal.signal(sig,lambda signum,frame:stop.set()) for sig in (signal.SIGINT,signal.SIGTERM)}
    threading.Thread(target=heartbeat,daemon=True).start()
    checkpoint('RUNNING')
    log(f'VERSION={VERSION}; REPORT_DIR={report}; targets={len(targets)}; inventory_issues={len(inventory)}')
    files=[]
    try:
        writers={}
        for name,fields in [('per_pdb.tsv',PDB_FIELDS),('per_seed.tsv',SEED_FIELDS),('per_sample.tsv',SAMPLE_FIELDS)]:
            f=(report/name).open('w',encoding='utf-8-sig',newline='');files.append(f)
            writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader();writers[name]=writer
        for split,identifier,pdb_paths in targets:
            if stop.is_set():break
            active.update(target=split+'/'+identifier,seed='',sample='',phase='reference')
            pi=[];pw=[];pr=[];expected_input_value=None
            refs=reference_index.get(identifier,[])
            if len(refs)!=1:pr.append('RAW_JSON_COUNT='+str(len(refs)))
            else:
                try:expected_input_value=expected_input(refs[0])
                except Exception as error:pr.append('RAW_JSON_REFERENCE_INVALID: '+str(error))
            reference_stamp=expected_input_value['sha256'] if expected_input_value else 'unavailable'
            if len(pdb_paths)>1:pi.append('DUPLICATE_PDB_DIRECTORIES')
            if identifier not in expected[split]:pi.append('PDB_NOT_IN_EXPECTED_SPLIT_CATALOG')
            folder=pdb_paths[0] if len(pdb_paths)==1 else args.data_root/split/identifier.lower()
            p_row=dict.fromkeys(PDB_FIELDS,'');p_row.update(SPLIT=split,PDB_ID=identifier,IS_SYMLINK=folder.is_symlink())
            seed_index=defaultdict(list)
            if folder.is_dir():
                p_row['RESOLVED_PATH']=str(folder.resolve())
                for entry in sorted(folder.iterdir()):
                    if entry.name.startswith('.'):continue
                    match=SEED_RE.fullmatch(entry.name)
                    if match:seed_index[int(match[1])].append(entry)
                    else:add(pw,'UNEXPECTED_PDB_ENTRY: '+entry.name)
            elif folder.is_symlink():pi.append('BROKEN_OR_NON_DIRECTORY_PDB_SYMLINK')
            elif folder.exists():pi.append('PDB_ENTRY_IS_NOT_A_DIRECTORY')
            p_row['FOUND_SEEDS']=len(seed_index)
            extras=sorted(set(seed_index)-set(SEEDS))
            if extras:pi.append('UNEXPECTED_SEEDS='+','.join(map(str,extras)))
            per_pdb_samples=Counter();per_pdb_seeds=Counter();identity_hash=None;coordinate_hashes={}
            for seed in sorted(set(SEEDS)|set(seed_index)):
                if stop.is_set():raise Interrupted()
                active.update(seed=seed,sample='',phase='seed inventory')
                s_issues=[];s_warnings=[];s_reviews=[]
                locations=seed_index.get(seed,[])
                if len(locations)>1:s_issues.append('DUPLICATE_SEED_DIRECTORIES')
                if seed not in SEEDS:s_issues.append('UNEXPECTED_SEED')
                if len(locations)==1 and locations[0].name!=f'seed_{seed}':s_issues.append('NONCANONICAL_SEED_DIRECTORY_NAME')
                pred=(locations[0]/'predictions') if len(locations)==1 else folder/f'seed_{seed}'/'predictions'
                if pred.exists() and not pred.is_dir():s_issues.append('PREDICTIONS_PATH_IS_NOT_A_DIRECTORY')
                elif pred.is_symlink() and not pred.is_dir():s_issues.append('BROKEN_PREDICTIONS_DIRECTORY_SYMLINK')
                index,file_issues,file_warnings=classify_files(pred,identifier)
                s_issues+=file_issues;s_warnings+=file_warnings
                seed_rows=[]
                for num in SAMPLES:
                    if stop.is_set():raise Interrupted()
                    active.update(sample=num,phase='sample validation')
                    paths=resolve_paths(index,num)
                    key=f'{split}/{identifier}/{seed}/{num}'
                    stamps={}
                    for kind,path in paths.items():
                        if path is not None:
                            try:stamps[kind]=fingerprint(path)
                            except Exception:pass
                    stamp={'files':stamps,'reference':reference_stamp}
                    sample=cache.get(key,stamp) if args.resume and len(stamps)==3 else None
                    if sample is not None:counters['reused_pass_samples']+=1
                    else:
                        sample,checked_stamps=sample_audit(split,identifier,seed,num,paths,expected_input_value,stop,args.max_value_mib*1024*1024)
                        counters['freshly_checked_samples']+=1
                        counters['full_json_bytes_scanned']+=int(sample['FULL_BYTES'] or 0)
                        counters['matrix_values_checked']+=int(sample['MATRIX_VALUES_CHECKED'] or 0)
                        # A cache hit on a future run also requires unchanged pre/post fingerprints.
                        if checked_stamps==stamps and len(stamps)==3:cache.put(key,stamp,sample)
                    if sample['STATUS']=='PASS' and sample['ATOM_IDENTITY_SHA256']:
                        if identity_hash is None:identity_hash=sample['ATOM_IDENTITY_SHA256']
                        elif sample['ATOM_IDENTITY_SHA256']!=identity_hash:
                            s_reviews.append('ATOM_IDENTITY_OR_ORDER_DIFFERS_ACROSS_SAMPLES')
                        coordinate_key=(sample['ATOM_IDENTITY_SHA256'],sample['COORDINATE_SHA256'])
                        previous=coordinate_hashes.setdefault(coordinate_key,(seed,num))
                        if previous!=(seed,num):add(s_warnings,f'IDENTICAL_COORDINATES: sample={num}, same as seed={previous[0]} sample={previous[1]}; not automatically invalid')
                    seed_rows.append(sample);writers['per_sample.tsv'].writerow(sample)
                    sample_counts[sample['STATUS']]+=1;per_pdb_samples[sample['STATUS']]+=1
                scores=[float(r['RANKING_SCORE']) for r in seed_rows if r['RANKING_SCORE']!='']
                if len(scores)==4 and any(scores[i]+1e-7<scores[i+1] for i in range(3)):
                    s_reviews.append('SAMPLE_RANKING_SCORE_ORDER_NOT_DESCENDING')
                s_status=aggregate([x['STATUS'] for x in seed_rows]+[state(s_issues,s_reviews)])
                s_row=dict.fromkeys(SEED_FIELDS,'');s_row.update(SPLIT=split,PDB_ID=identifier,SEED=seed,
                    STATUS=s_status,ISSUES='; '.join(s_issues+s_reviews),WARNINGS='; '.join(s_warnings),
                    PREDICTIONS_PATH=str(pred),RESOLVED_PREDICTIONS_PATH=str(pred.resolve()) if pred.is_dir() else '',
                    PASS_SAMPLES=sum(x['STATUS']=='PASS' for x in seed_rows),PROBLEM_SAMPLES=sum(x['STATUS']=='PROBLEM' for x in seed_rows),
                    MISSING_SAMPLES=sum(x['STATUS']=='MISSING' for x in seed_rows),REVIEW_SAMPLES=sum(x['STATUS']=='REVIEW' for x in seed_rows))
                writers['per_seed.tsv'].writerow(s_row);seed_counts[s_status]+=1;per_pdb_seeds[s_status]+=1
                if s_status!='PASS':add(pw,f'seed_{seed}: '+s_status)
                cache.commit()
                for f in files:f.flush()
                checkpoint('RUNNING')
            if expected_input_value and len(refs)==1:
                try:
                    if small_json(refs[0])[1]!=reference_stamp:pr.append('RAW_JSON_CHANGED_DURING_PDB_AUDIT')
                except Exception:pr.append('RAW_JSON_DISAPPEARED_DURING_PDB_AUDIT')
            p_status=aggregate(list(per_pdb_seeds)+[state(pi,pr)])
            p_row.update(STATUS=p_status,ISSUES='; '.join(pi+pr),WARNINGS='; '.join(pw),PASS_SEEDS=per_pdb_seeds['PASS'],
                PASS_SAMPLES=per_pdb_samples['PASS'],PROBLEM_SAMPLES=per_pdb_samples['PROBLEM'],
                MISSING_SAMPLES=per_pdb_samples['MISSING'],REVIEW_SAMPLES=per_pdb_samples['REVIEW'])
            writers['per_pdb.tsv'].writerow(p_row);pdb_results[(split,identifier)]=p_row
            pdb_counts[p_status]+=1;split_counts[split][p_status]+=1
            for f in files:f.flush()
            checkpoint('RUNNING');log(f'{len(pdb_results)}/{len(targets)} {split}/{identifier} {p_status} samples={dict(per_pdb_samples)} {p_row["ISSUES"]}')
    except Interrupted:stop.set()
    except Exception as error:
        fatal=f'{type(error).__name__}: {error}';log('FATAL: '+fatal)
    finally:
        for f in files:f.close()
        cache.close();done.set()
        for sig,handler in handlers.items():signal.signal(sig,handler)
    final_status='FAILED' if fatal else 'INTERRUPTED' if stop.is_set() else 'COMPLETED'
    active['phase']='finished';checkpoint(final_status);log(f'FINAL STATUS={final_status}; report={report}')
    report_lock.close()
    return 2 if fatal or inventory else 130 if stop.is_set() else 0 if pdb_counts['PASS']==len(targets) else 1


if __name__=='__main__':raise SystemExit(main())
