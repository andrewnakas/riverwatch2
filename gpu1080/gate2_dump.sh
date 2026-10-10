#!/bin/bash
# GATE 2 treatment arm: NLDAS-2 areal everywhere EXCEPT the newest 4 encoder
# days, which come from Open-Meteo point. Control is the EXISTING g1nl_ dump,
# so the baseline cannot drift. Registered in PREREG_v2.md #GATE 2.
set -u
cd ~/riverwatch2
PY=gpu1080/.venv/bin/python
V3=data/modern_corpus/v3
OM="$V3/camels_corpus_openmeteo_m3_cohort"
NL_SUB="$V3/camels_corpus_nldas_g1sub"
DD=data/mblstm/l51_dumps
MEMBER=nldasm9
VS=2014-10-01; VE=2018-09-30
SEEDS="501 502 503 504 505"
K=4
PFX="g2sp${K}_"
say () { echo "[$(date -u +%FT%TZ)] $*"; }

[ -d "$NL_SUB" ] || { say "ABORT: $NL_SUB missing"; exit 1; }
nb=$(ls "$NL_SUB" | wc -l); no=$(ls "$OM" | wc -l)
say "control corpus $nb basins · alt corpus $no basins · splicing newest $K days"

for S in $SEEDS; do
  CKPT="data/mblstm/gpu_ckpts/m1_${MEMBER}_s${S}.pt"
  OUT="$DD/${PFX}${MEMBER}_s${S}_val1.csv.gz"
  [ -f "$OUT" ] && { say "$PFX s$S exists"; continue; }
  say "dumping $PFX s$S"
  nice -n 10 $PY scripts/train_mblstm.py --epochs 0 --init-ckpt "$CKPT" \
    --out "$CKPT.dumpsink_g2" \
    --head quantile --enc-vars camels1f --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$NL_SUB" \
    --splice-recent-dir "$OM" --splice-recent-days "$K" \
    --val-start "$VS" --val-end "$VE" --val-stride 1 \
    --dump-day1 "$OUT" >> "logs/gate2_dump_s${S}.log" 2>&1 || { say "ABORT: dump s$S failed"; exit 1; }
  rm -f "$CKPT.dumpsink_g2"
  gzip -t "$OUT" || { say "ABORT: $PFX s$S failed gzip -t"; exit 1; }
  $PY - "$OUT" "$CKPT" "$MEMBER" "$S" "$VS" "$VE" "$NL_SUB" <<'PYS'
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
           "bytes": os.path.getsize(out), "ledger": "MODERN-1-GATE2",
           "corpus": corp, "basins_in_corpus": basins,
           "splice_recent_dir": os.environ.get("G2_ALT", ""), "splice_recent_days": 4,
           "enc_lead": int(cfg.get("enc_lead", 0) or 0),
           "dec_lead": int(cfg.get("dec_lead", 0) or 0)},
          open(out.replace(".csv.gz", ".json"), "w"), indent=1)
print(f"DUMP OK {out}: {n} rows (floor {floor})")
PYS
  [ $? -eq 0 ] || { say "ABORT: $PFX s$S sidecar failed"; exit 1; }
done
say "GATE 2 dumps complete"
