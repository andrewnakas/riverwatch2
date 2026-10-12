#!/usr/bin/env bash
# Build 14-lead areal GFS for the SAME 115 init dates as the registered lead-1
# artifact, so the two are directly comparable, then gate every output.
#
# Why these dates: benchmarks/m1_gfs_areal_screen.json was computed on them, so
# re-running the alignment screen lead-aware on the same sample isolates the
# lead extension as the only change. Picking fresh dates would confound it.
#
# ~5.9 min/init measured with the decode lock off (byte-identical output at
# ~1.9x), so ~11.3 h for 115. Network+CPU only -- no GPU, so this is safe
# alongside the foreign training job holding the card.
set -u
cd /home/nakas/riverwatch2
exec >>logs/m1_gfs14_chain.log 2>&1
echo "=== chain start $(date -u +%FT%TZ)"

nice -n 10 gpu1080/.venv/bin/python scripts_modern/build_gfs_areal_14lead.py \
  --start 2021-05-01 --end 2025-09-17 --stride 28 --leads 14 --workers 8 \
  --verify-lead1 --out data/mblstm/gfs_fcst_areal14
rc=$?
echo "[$(date -u +%T)] builder exited rc=$rc"

# GATE ON THE ARTIFACT. Every file must have 14 x n_basins rows, all 14 leads,
# no NaN, and physical bounds -- the builder asserts these per lead already, but
# a file on disk is the thing the screen will read, so check the files.
echo "[$(date -u +%T)] gating produced files"
nice -n 10 gpu1080/.venv/bin/python - <<'PY'
import glob, pandas as pd, numpy as np
fs = sorted(glob.glob("data/mblstm/gfs_fcst_areal14/*.csv.gz"))
print(f"files: {len(fs)}")
bad = 0
for p in fs:
    try:
        d = pd.read_csv(p, compression="gzip", dtype={"station_id": str})
    except Exception as e:
        print(f"  {p}: unreadable {e}"); bad += 1; continue
    leads = sorted(d.lead_day.unique())
    per = d.groupby("lead_day").size().unique()
    msgs = []
    if leads != list(range(1, 15)): msgs.append(f"leads {leads}")
    if len(per) != 1: msgs.append(f"ragged basins/lead {per}")
    if d.isna().sum().sum(): msgs.append("NaN")
    if (d.precipitation_sum < 0).any() or (d.precipitation_sum > 500).any(): msgs.append("precip out of bounds")
    if ((d.temperature_2m_min > d.temperature_2m_mean + 1e-6) |
        (d.temperature_2m_mean > d.temperature_2m_max + 1e-6)).any(): msgs.append("temp order")
    if msgs:
        print(f"  {p.split('/')[-1]}: {'; '.join(msgs)}"); bad += 1
print(f"FILE GATE: {'PASS' if bad == 0 else f'FAIL ({bad} bad)'}")
PY
echo "=== chain done $(date -u +%FT%TZ)"
