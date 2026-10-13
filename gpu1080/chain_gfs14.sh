#!/usr/bin/env bash
# Build 14-lead areal GFS for the SAME 115 init dates as the registered lead-1
# artifact, so a lead-aware alignment screen isolates the lead extension as the
# only change. Stride 14 -- NOT the builder's old 28 default, which yields 58
# dates, a different sample.
#
# ~6 min/init with the decode lock off, so ~11.5 h for 115. Network+CPU only:
# no GPU, safe alongside the foreign training job.
set -u
cd /home/nakas/riverwatch2
exec >>logs/m1_gfs14_chain.log 2>&1
echo "=== chain start $(date -u +%FT%TZ)"

EXPECTED=$(python3 -c "
from datetime import date,timedelta
d=date(2021,5,1); n=0
while d<=date(2025,9,17): n+=1; d+=timedelta(days=14)
print(n)")
echo "expecting $EXPECTED init dates"

nice -n 10 gpu1080/.venv/bin/python scripts_modern/build_gfs_areal_14lead.py \
  --start 2021-05-01 --end 2025-09-17 --stride 14 --leads 14 --workers 8 \
  --verify-lead1 --out data/mblstm/gfs_fcst_areal14
echo "[$(date -u +%T)] builder exited rc=$?"

# ⚠️ THE GATE MUST ASSERT COMPLETENESS, NOT JUST VALIDATE WHAT EXISTS.
# The previous version printed "FILE GATE: PASS" over 48 files when 115 were
# expected: it checked every file it found and said nothing at all about the
# 67 it did not. A gate that only inspects present artifacts cannot distinguish
# "all good" from "mostly missing", and a stride sample with holes is not the
# registered sample however clean the survivors are.
echo "[$(date -u +%T)] gating: completeness FIRST, then every file"
nice -n 10 gpu1080/.venv/bin/python - "$EXPECTED" <<'PY'
import glob, sys, pandas as pd
expected = int(sys.argv[1])
fs = sorted(glob.glob("data/mblstm/gfs_fcst_areal14/*.csv.gz"))
print(f"files: {len(fs)} of {expected} expected")
complete = len(fs) == expected
if not complete:
    from datetime import date, timedelta
    want, d = [], date(2021, 5, 1)
    while d <= date(2025, 9, 17):
        want.append(d.strftime("%Y%m%d")); d += timedelta(days=14)
    have = {p.split("/")[-1].split(".")[0] for p in fs}
    miss = [w for w in want if w not in have]
    print(f"  MISSING {len(miss)}: {miss[:8]}{' ...' if len(miss) > 8 else ''}")
bad = 0
for p in fs:
    try:
        d = pd.read_csv(p, compression="gzip", dtype={"station_id": str})
    except Exception as e:
        print(f"  {p}: unreadable {e}"); bad += 1; continue
    m = []
    if sorted(d.lead_day.unique()) != list(range(1, 15)): m.append("leads")
    if len(d.groupby("lead_day").size().unique()) != 1: m.append("ragged")
    if d.isna().sum().sum(): m.append("NaN")
    if (d.precipitation_sum < 0).any() or (d.precipitation_sum > 500).any(): m.append("precip")
    if ((d.temperature_2m_min > d.temperature_2m_mean + 1e-6) |
        (d.temperature_2m_mean > d.temperature_2m_max + 1e-6)).any(): m.append("temp order")
    if m:
        print(f"  {p.split('/')[-1]}: {'; '.join(m)}"); bad += 1
print(f"per-file problems: {bad}")
ok = complete and bad == 0
print("FILE GATE:", "PASS" if ok else
      f"FAIL (complete={complete}, bad_files={bad})")
PY
echo "=== chain done $(date -u +%FT%TZ)"
