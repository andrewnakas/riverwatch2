#!/usr/bin/env bash
# MODERN-1 PHASE 5 — the NO-Q arm. The second half of the original request
# ("both with q and no q"), entirely unbuilt until now.
#
# Why now: NLDAS slowed ~18x (3.75 -> ~70 min/month, ~63 h left for the last 76 months)
# and the GPU has been idle. The no-q arm needs no NLDAS, so it is the right use of an
# idle card while the 4th product finishes.
#
# ⛔ lstm_nldasm is NOT in this batch -- its corpus does not exist yet (0 basins). It is
# added when chain_fuse4 builds nldas_m3_cohort, rather than queued now to fail GUARD 0.
#
# Ordered highest-value first, so a breakage shows up on the run that matters:
#   lstm_fused3md3  the registered test of whether the all-product DECODER fix transfers
#                   to an arm where forcings are the ONLY input (+0.015528 paired with-q)
#   lstm_fused3m    multi-forcing, the strongest no-q class in the frozen record
#   single-product members, then the no-q Gate B (34-year window)
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
LOG=logs/m1_noq_chain.log
exec >>"$LOG" 2>&1
echo "=== NOQ CHAIN START $(date +%F_%T) ==="

M1_QUEUE=queue_m1_noq.sh M1_LOG=m1_batch_noq M1_ALLLEADS=0 \
  bash gpu1080/run_m1_batch.sh \
  lstm_fused3md3:501 lstm_fused3md3:502 lstm_fused3md3:503 \
  lstm_fused3m:501 lstm_fused3m:502 lstm_fused3m:503 \
  lstm_gridmetm:501 lstm_gridmetm:502 lstm_gridmetm:503 \
  lstm_nclimgridm:501 lstm_nclimgridm:502 lstm_nclimgridm:503 \
  lstm_daymetm:501 lstm_daymetm:502 lstm_daymetm:503

echo "[$(date +%T)] core no-q members done; now the no-q GATE B (34-year window)"
M1_QUEUE=queue_m1_noq.sh M1_LOG=m1_batch_noq_gateb M1_ALLLEADS=0 \
  bash gpu1080/run_m1_batch.sh \
  lstm_gridmetm34:501 lstm_gridmetm34:502 lstm_gridmetm34:503

# ---- verify the ARM on every dump before scoring anything -------------------
# Both arms share a protocol and a frame; only the sidecar separates them.
echo "[$(date +%T)] arm audit on every no-q sidecar"
$PY - <<'PYA'
import glob, json
bad = []
n = 0
for f in sorted(glob.glob("data/mblstm/l51_dumps/m1_lstm_*_val1.json")):
    d = json.load(open(f))
    n += 1
    if d.get("arm") != "noq" or d.get("no_q_input") is not True:
        bad.append((f, d.get("arm"), d.get("no_q_input")))
print(f"  checked {n} no-q sidecars; violations: {len(bad)}")
for b in bad:
    print(f"    {b}")
assert not bad, "a no-q dump does not declare arm=noq"
PYA
[ $? -eq 0 ] || { echo "ABORT: arm audit failed"; exit 7; }

# ---- the registered questions ----------------------------------------------
echo "[$(date +%T)] === Q1: does the all-product DECODER fix transfer to no-q? ==="
echo "    registered +0.004..+0.012 solo; with-q measured +0.015528 paired"
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members lstm_fused3md3 --against lstm_fused3m --seeds 501,502,503 \
    --out benchmarks/m1_noq_decoder_transfer.json

echo "[$(date +%T)] === Q2: no-q GATE B, 34 years vs 9 ==="
echo "    registered +0.002..+0.010 (larger than with-q's +0.001370, since removing the"
echo "    discharge input should make the model lean harder on seeing more weather)"
$PY analysis/m1_gateb_verdict.py --frame val1 --long lstm_gridmetm34 --short lstm_gridmetm \
    --long-protocol modern-v3 --short-protocol modern-v3-9yr --seeds 501,502,503 \
    --out benchmarks/m1_noq_gateb.json

echo "[$(date +%T)] === Q3: the no-q composition (equal weight, LOO, err-corr) ==="
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members lstm_fused3md3,lstm_fused3m,lstm_gridmetm,lstm_nclimgridm,lstm_daymetm \
    --seeds 501,502,503 --corr --dup-control \
    --out benchmarks/m1_noq_composition.json
echo "=== NOQ CHAIN DONE $(date +%F_%T) ==="
