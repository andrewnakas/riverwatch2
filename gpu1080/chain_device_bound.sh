#!/usr/bin/env bash
# Bound the CPU-vs-GPU difference between GATE 3's two arms, per
# PREREG_v2.md GATE 3 ADDENDUM 2026-10-17 (ii).
#
# The treatment arm is dumped on the CPU (the 1080 is held by an unrelated
# serving process); the control arm already exists and was dumped on the GPU.
# GATE 3's deciding statistic is a difference of LEVELS between two disjoint
# basin sets, which cannot absorb a systematic offset, so the device difference
# has to be measured rather than assumed negligible.
#
# Method: re-dump a handful of COHORT basins on the CPU and compare them with
# those same basins' rows in the existing GPU control. That bounds the device
# difference AND the subset-corpus difference together, which is sufficient: if
# the combined effect is negligible, neither part of it matters.
set -u
cd ~/riverwatch2
exec 9>/tmp/rw2_devbound.lock
flock -n 9 || { echo "another device-bound run holds the lock"; exit 0; }
PY=gpu1080/.venv/bin/python
DD=data/mblstm/l51_dumps
LOG=logs/m1_gate3_device_bound.log
exec >>"$LOG" 2>&1
echo "=== device bound start $(date -u +%FT%TZ)"

N=${1:-8}
CORP=gpu1080/corpora671/m1_devbound${N}
mkdir -p "$CORP"; rm -f "$CORP"/*.csv.gz
$PY - "$N" "$CORP" <<'PYG'
import json, os, shutil, sys
n = int(sys.argv[1]); corp = sys.argv[2]
cohort = json.load(open("data/m1_cohort.json"))["cohort"]
src = "data/modern_corpus/camels_corpus_nldas_modern_areal"
# Deterministic, and NOT the first n by id: USGS ids run roughly
# geographically, so a prefix is a regional sample.
import random
random.Random("devbound").shuffle(cohort)
picked = sorted(cohort[:n])
for sid in picked:
    p = os.path.join(src, f"{sid}.csv.gz")
    assert os.path.exists(p), p
    os.symlink(os.path.realpath(p), os.path.join(corp, f"{sid}.csv.gz"))
json.dump(picked, open("/tmp/devbound_ids.json", "w"))
print("picked:", picked)
PYG
[ $? -eq 0 ] || { echo "ABORT: corpus build failed"; exit 1; }

S=501
CKPT="data/mblstm/gpu_ckpts/m1_nldasm9_s${S}.pt"
OUT="$DD/devcpu_nldasm9_s${S}_val1.csv.gz"
rm -f "$OUT"
echo "[$(date -u +%T)] CPU dump of $N cohort basins, seed $S"
nice -n 15 $PY scripts/train_mblstm.py --epochs 0 --device cpu \
  --init-ckpt "$CKPT" --out "$CKPT.dumpsink_dev" \
  --head quantile --enc-vars camels1f --q-transform linear \
  --static-set camels --hidden 256 --corpus-dir "$CORP" \
  --val-start 2014-10-01 --val-end 2018-09-30 --val-stride 1 \
  --dump-day1 "$OUT" >> logs/m1_devbound_dump.log 2>&1
RC=$?    # before anything else runs
echo "[$(date -u +%T)] dump rc=$RC"
[ "$RC" = "0" ] || { echo "ABORT: dump rc=$RC"; exit 1; }
gzip -t "$OUT" || { echo "ABORT: failed gzip -t"; exit 1; }

$PY - "$OUT" "$DD/m1_nldasm9_s${S}_val1.csv.gz" <<'PYC'
import json, sys
import numpy as np, pandas as pd
cpu_p, gpu_p = sys.argv[1], sys.argv[2]
ids = set(json.load(open("/tmp/devbound_ids.json")))
cpu = pd.read_csv(cpu_p, dtype={"station_id": str})
gpu = pd.read_csv(gpu_p, dtype={"station_id": str})
for d in (cpu, gpu):
    d["station_id"] = d.station_id.str.zfill(8)
cpu = cpu[(cpu.h == 1) & cpu.station_id.isin(ids)]
gpu = gpu[(gpu.h == 1) & gpu.station_id.isin(ids)]
m = cpu.merge(gpu, on=["station_id", "t0"], suffixes=("_cpu", "_gpu"))
print(f"basins {cpu.station_id.nunique()} (cpu) / {gpu.station_id.nunique()} (gpu); "
      f"joined rows {len(m):,}")
if m.empty:
    sys.exit("ABORT: the two dumps share no (station_id, t0) rows")
# Truth must agree first: if it does not, this is a target revision or a bad
# join, not a device difference, and nothing below means anything.
dt = np.abs(m.truth_cpu.to_numpy(float) - m.truth_gpu.to_numpy(float))
print(f"truth max |diff| = {np.nanmax(dt):.6g}")
rows = []
for sid, g in m.groupby("station_id"):
    y = g.truth_cpu.to_numpy(float)
    for tag in ("cpu", "gpu"):
        pass
    pc = ((g.ylo_cpu + g.yhi_cpu) / 2).to_numpy(float)
    pg = ((g.ylo_gpu + g.yhi_gpu) / 2).to_numpy(float)
    k = np.isfinite(y) & np.isfinite(pc) & np.isfinite(pg)
    y, pc, pg = y[k], pc[k], pg[k]
    if len(y) < 20 or np.var(y) < 1e-9:
        continue
    nse_c = 1 - np.mean((y - pc) ** 2) / np.var(y)
    nse_g = 1 - np.mean((y - pg) ** 2) / np.var(y)
    rows.append((sid, nse_c, nse_g, nse_c - nse_g,
                 float(np.nanmax(np.abs(pc - pg)))))
df = pd.DataFrame(rows, columns=["station_id", "nse_cpu", "nse_gpu",
                                 "d_nse", "max_abs_pred_diff"])
print(df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
worst = float(df.d_nse.abs().max())
print(f"\nmax |NSE difference| over {len(df)} basins: {worst:.8g}")
print(f"max |prediction difference|: {float(df.max_abs_pred_diff.max()):.8g} cfs")
verdict = ("PROCEED AS REGISTERED" if worst < 0.001 else
           "RE-DUMP THE CONTROL ON CPU before any verdict")
print(f"registered threshold 0.001 -> {verdict}")
json.dump({"_note": ("Bounds the CPU-vs-GPU and subset-vs-full-corpus "
                     "differences TOGETHER for GATE 3's two arms. Registered "
                     "in PREREG_v2.md GATE 3 ADDENDUM 2026-10-17 (ii). May "
                     "not be used to adjust either arm."),
           "seed": 501, "frame": "val1", "basins": len(df),
           "rows_joined": int(len(m)),
           "truth_max_abs_diff": float(np.nanmax(dt)),
           "max_abs_nse_diff": worst,
           "max_abs_pred_diff_cfs": float(df.max_abs_pred_diff.max()),
           "registered_threshold": 0.001,
           "verdict": verdict,
           "per_basin": df.to_dict(orient="records")},
          open("benchmarks/m1_gate3_device_bound.json", "w"), indent=1)
print("-> benchmarks/m1_gate3_device_bound.json")
PYC
rm -f "$CKPT.dumpsink_dev"
echo "=== device bound done $(date -u +%FT%TZ)"
