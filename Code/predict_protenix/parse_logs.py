"""
Parse Protenix prediction logs to find which PDB IDs succeeded/failed.
"""
import os
import re
import glob

LOG_DIR = r"C:\Users\49586\Desktop\Learning\Laboratory\Admis\graduate_first\RNA\Code\predict_protenix"

# Patterns
success_pat = re.compile(r"\[(\w+?) \| Seed (\d+)\] Pred 成功完成")
failure_pat = re.compile(r"\[(\w+?) \| Seed (\d+)\] Pred 阶段失败")
unknown_pat = re.compile(r"\[(\w+?) \| Seed (\d+)\] 发生未知错误")

success = {}  # pdb_id -> set of seeds
failure = {}  # pdb_id -> set of seeds
unknown = {}  # pdb_id -> set of seeds

for log_file in sorted(glob.glob(os.path.join(LOG_DIR, "nohup*.log"))):
    basename = os.path.basename(log_file)
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = success_pat.search(line)
            if m:
                pdb, seed = m.group(1).lower(), int(m.group(2))
                success.setdefault(pdb, set()).add(seed)
                continue
            m = failure_pat.search(line)
            if m:
                pdb, seed = m.group(1).lower(), int(m.group(2))
                failure.setdefault(pdb, set()).add(seed)
                continue
            m = unknown_pat.search(line)
            if m:
                pdb, seed = m.group(1).lower(), int(m.group(2))
                unknown.setdefault(pdb, set()).add(seed)

all_pdbs = set(success.keys()) | set(failure.keys()) | set(unknown.keys())
expected_seeds = {42, 43, 44, 45}

print("=" * 60)
print("Protenix Prediction Log Analysis (nohup1-9.log)")
print("=" * 60)

print(f"\nTotal unique PDB IDs in logs: {len(all_pdbs)}")
print(f"Expected seeds per ID: {sorted(expected_seeds)}")

# Per-PDB status
fully_success = []
partial_fail = []
all_fail = []
no_record_seeds = []

for pdb in sorted(all_pdbs):
    succ_seeds = success.get(pdb, set())
    fail_seeds = failure.get(pdb, set())
    unk_seeds = unknown.get(pdb, set())
    total_seen = succ_seeds | fail_seeds | unk_seeds
    missing = expected_seeds - total_seen

    if succ_seeds == expected_seeds:
        fully_success.append(pdb)
    elif fail_seeds == expected_seeds or (fail_seeds | unk_seeds) == expected_seeds:
        all_fail.append(pdb)
    else:
        partial_fail.append(pdb)

    if missing:
        no_record_seeds.append((pdb, missing))

print(f"\n  Fully successful (4/4 seeds):  {len(fully_success)}")
print(f"  Partially failed:              {len(partial_fail)}")
print(f"  All 4 seeds failed:            {len(all_fail)}")
print(f"  IDs with missing seed records: {len(no_record_seeds)}")

# Details
if partial_fail:
    print(f"\n─ Partial failures (some seeds failed) ─")
    for pdb in sorted(partial_fail):
        s = success.get(pdb, set())
        f = failure.get(pdb, set())
        u = unknown.get(pdb, set())
        print(f"  {pdb}: success={sorted(s)}, failed={sorted(f)}, unknown={sorted(u)}")

if all_fail:
    print(f"\n─ All-4-seeds failed ─")
    for pdb in sorted(all_fail):
        f = failure.get(pdb, set())
        u = unknown.get(pdb, set())
        print(f"  {pdb}: failed={sorted(f)}, unknown={sorted(u)}")

if no_record_seeds:
    print(f"\n─ Missing seed records (no success/fail line found) ─")
    for pdb, missing in sorted(no_record_seeds):
        print(f"  {pdb}: missing seeds={sorted(missing)}")

# Summary
total_tasks = sum(len(v) for v in success.values())
total_failed = sum(len(v) for v in failure.values())
total_unknown = sum(len(v) for v in unknown.values())
print(f"\n─ Summary ─")
print(f"  Total successful predictions:  {total_tasks}")
print(f"  Total failed predictions:      {total_failed}")
print(f"  Total unknown errors:          {total_unknown}")
print(f"  Overall success rate:          {total_tasks}/{total_tasks+total_failed+total_unknown} = {total_tasks/(total_tasks+total_failed+total_unknown)*100:.1f}%")
