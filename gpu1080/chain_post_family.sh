#!/usr/bin/env bash
# MODERN-1 — everything that runs AFTER the fused family, in one ordered chain.
#
# ⚠️ WHY ONE CHAIN AND NOT TWO WAITERS: chain_nh3b was already parked on "exclusive GPU
# slot", and adding a second waiter for the decade probe would have had both fire on the
# same drain -- the oversubscription race this box has already paid for once. One waiter,
# explicit order. The cheap, interpretation-critical probe goes FIRST (~1 h) ahead of the
# 18 h NH training.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
V3=data/modern_corpus/v3
DD=data/mblstm/l51_dumps
LOG=logs/m1_post_family.log
exec >>"$LOG" 2>&1
echo "=== POST-FAMILY CHAIN START $(date +%F_%T) ==="

# ---- 1. wait for ONE exclusive slot -----------------------------------------
QUIET=0
for i in $(seq 1 2160); do
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  B=$(ps -eo cmd --no-headers | grep -c "run_m1_batc[h]\.sh")
  Q=$(ps -eo cmd --no-headers | grep -c "queue_m1_with[q]\.sh")
  if [ "$N" -eq 0 ] && [ "$B" -eq 0 ] && [ "$Q" -eq 0 ]; then QUIET=$((QUIET+1)); else QUIET=0; fi
  [ "$QUIET" -ge 3 ] && { echo "[$(date +%T)] exclusive slot on 3 polls"; break; }
  [ $((i % 30)) -eq 0 ] && echo "[$(date +%T)] waiting: gpu=$N batch=$B queue=$Q quiet=$QUIET/3"
  sleep 120
done
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)" -ne 0 ] && {
  echo "never got an exclusive slot"; exit 5; }

# ---- 2. THE DECADE PROBE: one checkpoint, a second evaluation decade --------
# fused3md3q trained 2005-10-01..2014-09-30. Dump it on the LEGACY decade
# 1989-10-01..1999-09-30, which is fully outside its training window (no leakage), and
# pair per-basin against its own 2018-2025 test1.
CORP="$V3/camels_corpus_fused3m_m3_cohort"
ENC=camelsm3v2d3
for S in 501 502 503; do
  CKPT="data/mblstm/gpu_ckpts/m1_fused3md3q_s${S}.pt"
  OUT="$DD/m1_fused3md3q_s${S}_xdecade1.csv.gz"
  [ -f "$OUT" ] && { echo "[$(date +%T)] s$S xdecade1 exists"; continue; }
  [ -f "$CKPT" ] || { echo "[$(date +%T)] s$S: NO CHECKPOINT - skipping"; continue; }
  echo "[$(date +%T)] dumping fused3md3q s$S on the legacy decade 1989-10-01..1999-09-30"
  nice -n 10 $PY scripts/train_mblstm.py --epochs 0 --init-ckpt "$CKPT" \
    --out "$CKPT.dumpsink_xdecade1" \
    --head quantile --enc-vars "$ENC" --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --val-start 1989-10-01 --val-end 1999-09-30 --val-stride 1 \
    --dump-day1 "$OUT" --dec-q >> "logs/m1_dump_fused3md3q_s${S}_xdecade1.log" 2>&1
  RC=$?
  rm -f "$CKPT.dumpsink_xdecade1"
  echo "[$(date +%T)] s$S dump rc=$RC"
  gzip -t "$OUT" 2>/dev/null || { echo "  ABORT s$S: dump failed gzip -t"; rm -f "$OUT"; continue; }
  # sidecar: the TRAINING window is what the guard checks, and it is unchanged. The
  # frame name is distinct so this can never be mistaken for the member's own test1.
  $PY - "$OUT" "$CKPT" "$S" <<'PYS'
import gzip, hashlib, json, os, sys
out, ck, seed = sys.argv[1:4]
n = sum(1 for _ in gzip.open(out, "rt")) - 1
assert n > 1000, f"only {n} rows"
hdr = gzip.open(out, "rt").readline().strip()
assert hdr == "station_id,t0,h,truth,ylo,ymed,yhi", hdr
json.dump({"member": "fused3md3q", "seed": int(seed), "frame": "xdecade1", "rows": n,
           "ckpt": ck, "ckpt_md5": hashlib.md5(open(ck, "rb").read()).hexdigest(),
           "train_start": "2005-10-01", "train_end": "2014-09-30",
           "protocol": "modern-v3-9yr", "all_leads": False,
           "eval_window": "1989-10-01..1999-09-30",
           "_note": "DECADE PROBE: the member's own test window is 2018-2025. This frame "
                    "re-evaluates the SAME weights on the legacy decade. Fully outside the "
                    "training window, so not in-sample.",
           "bytes": os.path.getsize(out), "ledger": "MODERN-1"},
          open(out.replace(".csv.gz", ".json"), "w"), indent=1)
print(f"  sidecar written: {n} rows")
PYS
done
NX=$(ls "$DD"/m1_fused3md3q_s*_xdecade1.csv.gz 2>/dev/null | wc -l)
echo "[$(date +%T)] xdecade1 dumps: $NX/3"
if [ "$NX" -ge 3 ]; then
  echo "[$(date +%T)] scoring the decade probe"
  $PY analysis/m1_decade_probe.py --member fused3md3q --seeds 501,502,503 \
      --out benchmarks/m1_decade_probe.json
else
  echo "[$(date +%T)] fewer than 3 xdecade1 dumps - NOT scoring (a 1-2 seed read cannot decide)"
fi

# ---- 3. hand over to the NH member (it re-verifies its own prep and slot) ---
echo "[$(date +%T)] handing over to chain_nh3b"
bash gpu1080/chain_nh3b.sh
echo "=== POST-FAMILY CHAIN DONE $(date +%F_%T) ==="
