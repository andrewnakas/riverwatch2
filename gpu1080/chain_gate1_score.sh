#!/usr/bin/env bash
# GATE 1 scoring — dump `nldasm9` on val1 under BOTH forcings, then read the verdict.
#
# Waits for the Open-Meteo corpus chain to reach its registered 150 basins, then
# dumps 5 seeds x 2 corpora and runs analysis/m1_gate1_verdict.py. Both arms use
# the SAME checkpoints and a 150-basin symlink tree of the SAME NLDAS source the
# Open-Meteo corpus was copied from (verified byte-identical), so forcing is the
# only difference.
#
# Gates on the RESOURCE, not a log marker: it refuses to start if anything else
# holds the GPU (calling a queue script onto a full card once caused a CUDA OOM
# that got mis-diagnosed as a hardware fault), and its wait keys on the corpus
# file COUNT increasing rather than on a wall clock.
set -u
cd "$HOME/riverwatch2" || exit 1

PY=gpu1080/.venv/bin/python
V3=data/modern_corpus/v3
OM="$V3/camels_corpus_openmeteo_m3_cohort"
NL_SRC=data/modern_corpus/camels_corpus_nldas_modern_areal
NL_SUB="$V3/camels_corpus_nldas_g1sub"
DD=data/mblstm/l51_dumps
LOG=logs/gate1_score.log
TARGET=150
MEMBER=nldasm9
VS=2014-10-01; VE=2018-09-30
SEEDS="501 502 503 504 505"

say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

# ---- 1. wait for the corpus, keyed on progress -----------------------------
# ⛔ The first version aborted after 90 min with no new basin, reasoning that a
# stalled corpus meant a dead upstream. That threshold was SHORTER THAN THE
# NATURAL PROGRESS INTERVAL: the upstream advances once per Open-Meteo hourly
# quota window, and a window whose quota is still pinned legitimately adds
# nothing, so 90 quiet minutes is one normal window. It duly killed a perfectly
# healthy wait at 12:51Z.
#
# Keying on progress is right; the threshold has to EXCEED the slowest
# legitimate progress interval. So the real "this will never finish" signal is
# used instead: the upstream chain is gone while the corpus is incomplete.
# Elapsed time is only a backstop.
last=-1
for i in $(seq 1 1200); do         # 1200 x 90s = 30h backstop: the upstream now
                                   # waits for a DAILY quota reset, so 12h was short
  n=$(ls "$OM" 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -ge "$TARGET" ] && { say "corpus complete: $n/$TARGET"; break; }
  if [ "$n" -ne "$last" ]; then
    last=$n
    say "corpus at $n/$TARGET"
  fi
  if ! pgrep -f "chain_gate1_o[m]\.sh" > /dev/null; then
    say "ABORT: upstream chain_gate1_om.sh is not running and corpus is $n/$TARGET"
    exit 1
  fi
  sleep 90
done
n=$(ls "$OM" 2>/dev/null | wc -l | tr -d ' ')
[ "$n" -ge "$TARGET" ] || { say "ABORT: corpus only $n/$TARGET"; exit 1; }

# ---- 2. the GPU must be free ------------------------------------------------
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
if [ "${busy:-0}" -gt 0 ]; then
  say "ABORT: $busy compute process(es) already on the GPU"
  exit 1
fi

# ---- 3. matched 150-basin NLDAS tree ---------------------------------------
rm -rf "$NL_SUB"; mkdir -p "$NL_SUB"
for f in "$OM"/*.csv.gz; do
  g=$(basename "$f")
  ln -sf "$HOME/riverwatch2/$NL_SRC/$g" "$NL_SUB/$g"
done
say "matched NLDAS tree: $(ls "$NL_SUB" | wc -l) basins"

# ---- 4. dumps: 5 seeds x 2 forcings ----------------------------------------
mkdir -p "$DD"
dump () {  # <prefix> <corpus>
  local PFX="$1" CORP="$2" S OUT CKPT
  for S in $SEEDS; do
    CKPT="data/mblstm/gpu_ckpts/m1_${MEMBER}_s${S}.pt"
    OUT="$DD/${PFX}${MEMBER}_s${S}_val1.csv.gz"
    [ -f "$OUT" ] && { say "$PFX s$S exists"; continue; }
    say "dumping $PFX s$S"
    nice -n 10 $PY scripts/train_mblstm.py --epochs 0 --init-ckpt "$CKPT" \
      --out "$CKPT.dumpsink_g1$PFX" \
      --head quantile --enc-vars camels1f --q-transform linear \
      --static-set camels --hidden 256 --corpus-dir "$CORP" \
      --val-start "$VS" --val-end "$VE" --val-stride 1 \
      --dump-day1 "$OUT" >> "logs/gate1_dump_${PFX}s${S}.log" 2>&1
    rm -f "$CKPT.dumpsink_g1$PFX"
    gzip -t "$OUT" || { say "ABORT: $PFX s$S failed gzip -t"; return 1; }
    # Sidecar, with the row floor DERIVED from this corpus rather than the full
    # 490 cohort -- a hard-coded floor once failed a perfectly good dump here.
    $PY - "$OUT" "$CKPT" "$MEMBER" "$S" "$VS" "$VE" "$CORP" <<'PYS'
import datetime as dt, gzip, hashlib, json, os, sys
out, ck, member, seed, a, b, corp = sys.argv[1:8]
import torch as _t
cfg = _t.load(ck, map_location="cpu", weights_only=False)["cfg"]
n = sum(1 for _ in gzip.open(out, "rt")) - 1
basins = len([f for f in os.listdir(corp) if f.endswith(".csv.gz")])
days = (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days + 1
floor = int(0.40 * basins * days)
assert n >= floor, f"{n} rows < derived floor {floor} ({basins} basins x {days} d)"
hdr = gzip.open(out, "rt").readline().strip()
assert hdr == "station_id,t0,h,truth,ylo,ymed,yhi", hdr
json.dump({"member": member, "seed": int(seed), "frame": "val1", "rows": n,
           "ckpt": ck, "ckpt_md5": hashlib.md5(open(ck, "rb").read()).hexdigest(),
           "train_start": cfg.get("train_start"), "train_end": cfg.get("train_end"),
           "protocol": "modern-v3-9yr", "all_leads": False,
           "bytes": os.path.getsize(out), "ledger": "MODERN-1-GATE1",
           "corpus": corp, "basins_in_corpus": basins,
           "enc_lead": int(cfg.get("enc_lead", 0) or 0),
           "dec_lead": int(cfg.get("dec_lead", 0) or 0)},
          open(out.replace(".csv.gz", ".json"), "w"), indent=1)
print(f"DUMP OK {out}: {n} rows (floor {floor})")
PYS
    [ $? -eq 0 ] || { say "ABORT: $PFX s$S sidecar failed"; return 1; }
  done
}

dump g1nl_ "$NL_SUB" || exit 1
dump g1om_ "$OM"     || exit 1

# ---- 5. the verdict ---------------------------------------------------------
say "scoring"
$PY analysis/m1_gate1_verdict.py >> "$LOG" 2>&1
say "GATE 1 done"
