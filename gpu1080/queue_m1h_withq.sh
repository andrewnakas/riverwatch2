#!/usr/bin/env bash
# MODERN-1 — the WINDOW-MATCHED 4-product members. usage: <member> <seed>
#
# WHY THIS EXISTS. The 4-product members (fused4m, fused4md4) were defined without a TRS
# override, so they inherited this launcher's 34-year default (1980-10-01) while the entire
# roster -- and fused3md3, their comparison baseline -- trains on 9 years (2005-10-01).
# Comparing them would have conflated "a 4th forcing product" with "a 3.75x longer training
# window" and reported the sum as the product effect.
#
# ⭐ The scorer's protocol guard CAUGHT it: it refused the dump outright, because the sidecar
# said train 1980-10-01..2014-09-30 against a modern-v3-9yr request. That guard exists
# because its absence once produced a model trained on its own test decade (the retracted
# 0.9253). It has now also caught a window confound, which is a second thing it buys.
#
# ⚠️ I had already spotted this exact trap for daymetm/nldasm and added 9-year twins; I
# missed it for the fused4m family. The lesson is to derive the member table from the
# protocol, not to patch members one at a time as they are noticed.
#
# A SEPARATE FILE because queue_m1_withq.sh is running the p4 batch right now.
# MODERN-1 — train one with-q member on the modern corpus. usage: <member> <seed>
#
# Protocol P-MODERN, pre-registered in PREREG_v2.md MODERN-1 before any data existed:
#   train 1980-10-01..2014-09-30 | val 2014-10-01..2018-09-30 | test 2018-10-01..2025-09-30
# Cohort: data/m1_cohort.json (490 basins), frozen from discharge coverage BEFORE
# any score was seen. The 531 are NOT the modern cohort -- 24 of their gauges are
# entirely dead in the test window.
set -u
cd ~/riverwatch2 || exit 1
M="${1:?member}"; S="${2:?seed}"
PY=gpu1080/.venv/bin/python
DD=data/mblstm/l51_dumps
V3=data/modern_corpus/v3
LOG="logs/m1_${M}_s${S}.log"
CKPT="data/mblstm/gpu_ckpts/m1_${M}_s${S}.pt"
TRS=1980-10-01; TRE=2014-09-30
VS=2014-10-01;  VE=2018-09-30
TS=2018-10-01;  TE=2025-09-30
SMOKE=0
case "$M" in *smoke*) SMOKE=1 ;; esac
LIMIT=""; [ "$SMOKE" = "1" ] && LIMIT="--limit-stations 8"

EXTRA=""
case "$M" in
  daymetm|daymetmsmoke)      CORP="$V3/camels_corpus_daymet_m3_cohort";    ENC="camels1f" ;;
  nldasm|nldasmsmoke)        CORP="$V3/camels_corpus_nldas_m3_cohort";     ENC="camels1f" ;;
  gridmetm|gridmetmsmoke)    CORP="$V3/camels_corpus_gridmet_m3_cohort";   ENC="camels1f" ;;
  nclimgridm|nclimgridmsmoke)CORP="$V3/camels_corpus_nclimgrid_m3_cohort"; ENC="camels1f" ;;
  # ---- GATE B: identical recipe, 9-year training window instead of 34 ----------
  # P-MODERN offers 34 years; the record used 9 only for comparability with
  # Nearing 2022. Nothing here has ever tested whether the extra 25 buy skill, and
  # the answer is also a BUDGET decision: 34 years is ~6.0M train windows against
  # ~1.6M, i.e. ~3.5x the GPU cost per seed, across ~50 runs.
  gridmetm9)                 CORP="$V3/camels_corpus_gridmet_m3_cohort";   ENC="camels1f"; TRS=2005-10-01 ;;
  nclimgridm9)               CORP="$V3/camels_corpus_nclimgrid_m3_cohort"; ENC="camels1f"; TRS=2005-10-01 ;;
  # ---- the 3-PRODUCT family: daymet+gridmet+nclimgrid, available FIRST ---------
  # NLDAS was deferred because its remaining transfer is ~490 GB against Daymet's
  # 4.5 GB. The fused + decoder-fixed + dec-q members are where the measured value
  # is, so the critical path never needed all four products.
  fused3m|fused3msmoke)      CORP="$V3/camels_corpus_fused3m_m3_cohort";   ENC="camelsm3v2";   TRS=2005-10-01 ;;
  fused3md3|fused3md3smoke)  CORP="$V3/camels_corpus_fused3m_m3_cohort";   ENC="camelsm3v2d3"; TRS=2005-10-01 ;;
  # ⭐ the best member class measured on the CAMELS analogue: all-product decoder
  # PLUS --dec-q, solo +0.006384 CI [+0.004554,+0.008198]
  fused3md3q|fused3md3qsmoke) CORP="$V3/camels_corpus_fused3m_m3_cohort";  ENC="camelsm3v2d3"; EXTRA="--dec-q"; TRS=2005-10-01 ;;
  # the ADMITTING hybrid shape (1-product encoder / all-product decoder)
  daymetmd3)                 CORP="$V3/camels_corpus_fused3m_m3_cohort";   ENC="camelsm1f_daymet_d3";    TRS=2005-10-01 ;;
  gridmetmd3)                CORP="$V3/camels_corpus_fused3m_m3_cohort";   ENC="camelsm1f_gridmet_d3";   TRS=2005-10-01 ;;
  nclimgridmd3)              CORP="$V3/camels_corpus_fused3m_m3_cohort";   ENC="camelsm1f_nclimgrid_d3"; TRS=2005-10-01 ;;
  # 24-var encoder, single-product (daymet) decoder -- the OPERATIONALLY servable shape
  fused4m|fused4msmoke)      CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm4v2" ;;
  # 24-var encoder AND decoder -- the LEDGER 56 fix
  fused4md4|fused4md4smoke)  CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm4v2d4" ;;
  # ⭐ + --dec-q: the assimilated discharge reaches the DECODER too. Measured on
  # the CAMELS analogue 2026-09-25: fused3h1d3q solo +0.006384 CI
  # [+0.004554,+0.008198] -- the best single member class this campaign has
  # produced, ahead of the all-product decoder alone (+0.005604).
  fused4md4q|fused4md4qsmoke)CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm4v2d4"; EXTRA="--dec-q" ;;
  # ⭐ --dec-q on the SINGLE-product members. Pre-registered in PREREG_v2.md: the
  # decoder-discharge defect is orthogonal to the forcing set, and by the 1/N rule
  # a uniform per-member lift beats any new member by ~an order of magnitude.
  # Judged on the COMPOSITION, with pairwise error correlation reported, because
  # the analogous d3 fix bought skill and spent distinctness (0.8358 -> 0.9046).
  # GATE B adopted the 9-year window (TRS=2005-10-01): 34 years measured only
  # +0.001370 for ~3.75x the compute, and --dec-q is registered at +0.003..+0.008.
  # These are seed-matched against gridmetm9/nclimgridm9, so they MUST share that
  # window -- comparing a dec-q member on 34 years against a plain one on 9 would
  # confound the two changes.
  daymetmq)                  CORP="$V3/camels_corpus_daymet_m3_cohort";    ENC="camels1f"; EXTRA="--dec-q"; TRS=2005-10-01 ;;
  nldasmq)                   CORP="$V3/camels_corpus_nldas_m3_cohort";     ENC="camels1f"; EXTRA="--dec-q"; TRS=2005-10-01 ;;
  gridmetmq)                 CORP="$V3/camels_corpus_gridmet_m3_cohort";   ENC="camels1f"; EXTRA="--dec-q"; TRS=2005-10-01 ;;
  nclimgridmq)               CORP="$V3/camels_corpus_nclimgrid_m3_cohort"; ENC="camels1f"; EXTRA="--dec-q"; TRS=2005-10-01 ;;
  # single-product ENCODER + all-product DECODER. Measured ADMITTING on the CAMELS
  # analogue (daymeth1d3 +0.000261, nldash1d3 +0.000209, both CI-positive), unlike
  # the mirror shape (all-product encoder / single-product decoder), which read NULL.
  daymetmd4)                 CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm1f_daymet_d4" ;;
  nldasmd4)                  CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm1f_nldas_d4" ;;
  gridmetmd4)                CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm1f_gridmet_d4" ;;
  nclimgridmd4)              CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm1f_nclimgrid_d4" ;;
  # ---- NLDAS single-product, WINDOW-MATCHED (the existing nldasm is 34-year) --
  nldasm9)                   CORP="$V3/camels_corpus_nldas_m3_cohort";     ENC="camels1f"; TRS=2005-10-01 ;;
  # ---- the 3rd single-product member, WINDOW-MATCHED to the others -----------
  # ⚠️ `daymetm` above inherits the default 34-year window, so it is NOT comparable to
  # gridmetm9 / nclimgridm9 / the fused family, all of which are 9-year (Gate B: 34 years
  # bought only +0.001370 for ~3.75x the compute). This is the 9-year twin, and it also
  # supplies the missing datum for the decoder-product probe: the standalone strength of
  # Daymet V4R1 on this corpus.
  daymetm9)                  CORP="$V3/camels_corpus_daymet_m3_cohort";    ENC="camels1f"; TRS=2005-10-01 ;;
  # ---- THE DECODER-PRODUCT PROBE ---------------------------------------------
  # All-product encoder, ONE-product decoder, swapping which product the decoder sees.
  # `fused3m` is already the daymet cell of this family (ENC=camelsm3v2), so only the
  # other two need building. Tests whether the +0.027060 all-product-decoder gain scales
  # with how weak the decoder's sole product is.
  fused3mgd)                 CORP="$V3/camels_corpus_fused3m_m3_cohort";   ENC="camelsm3v2_dec_gridmet";   TRS=2005-10-01 ;;
  fused3mnd)                 CORP="$V3/camels_corpus_fused3m_m3_cohort";   ENC="camelsm3v2_dec_nclimgrid"; TRS=2005-10-01 ;;
  # ---- the REGISTRATION-CORRECTED gridMET member (leak audit 2026-10-06) -----
  # gridMET row t carried rain(t+1) (corr 0.957 vs daymet[t+1], 0.103 at k=0) while every
  # other product row t carries rain(t). Daymet got the +1 shift at build time; gridMET did
  # not, because Gate 1 Part A recommended it and I dismissed that on the referee's say-so.
  # Identical to gridmetm9 except the corpus. If this scores HIGHER the misregistration was
  # a COST (record understated, not false); if LOWER the original was using future rain.
  gridmetm9fix3)             CORP="$V3/camels_corpus_gridmetfix3_m3_cohort"; ENC="camels1f"; TRS=2005-10-01 ;;
  gridmetm9fix2)             CORP="$V3/camels_corpus_gridmetfix2_m3_cohort"; ENC="camels1f"; TRS=2005-10-01 ;;
  gridmetm9fix)              CORP="$V3/camels_corpus_gridmetfix_m3_cohort"; ENC="camels1f"; TRS=2005-10-01 ;;
  # ---- the WINDOW-MATCHED 4-product members (9-year, like the whole roster) ---
  daymetm9L1)                CORP="$V3/camels_corpus_daymet_m3_cohort";    ENC="camels1f"; EXTRA="--enc-lead 1"; TRS=2005-10-01 ;;
  nldasm9L1)                 CORP="$V3/camels_corpus_nldas_m3_cohort";     ENC="camels1f"; EXTRA="--enc-lead 1"; TRS=2005-10-01 ;;
  nclimgridm9L1)             CORP="$V3/camels_corpus_nclimgrid_m3_cohort"; ENC="camels1f"; EXTRA="--enc-lead 1"; TRS=2005-10-01 ;;
  fused4m9fix3L1)            CORP="$V3/camels_corpus_fused4mfix3_m3_cohort"; ENC="camelsm4v2";   EXTRA="--enc-lead 1"; TRS=2005-10-01 ;;
  fused4md49fix3L1)          CORP="$V3/camels_corpus_fused4mfix3_m3_cohort"; ENC="camelsm4v2d4"; EXTRA="--enc-lead 1"; TRS=2005-10-01 ;;
  # ⛔ DIAGNOSTIC ONLY -- reads decoder forcing one day BEYOND the day it predicts, which is
  # outside perfect-forcing entitlement and unservable. check_sidecar refuses dec_lead!=0
  # dumps unless M1_ALLOW_DIAGNOSTIC=1, so this can be measured but never frozen.
  # ⛔ DIAGNOSTIC ONLY. Both leads together shift every column one day, which should be
  # EQUIVALENT to the misregistered corpus with no flags. That equivalence is the test:
  # it decomposes the original into (led encoder) + (decoder reading rain(T+1)), and the
  # second term is forcing beyond entitlement.
  gridmetm9fix3L1D1)         CORP="$V3/camels_corpus_gridmetfix3_m3_cohort"; ENC="camels1f"; EXTRA="--enc-lead 1 --dec-lead 1"; TRS=2005-10-01 ;;
  gridmetm9fix3D1)           CORP="$V3/camels_corpus_gridmetfix3_m3_cohort"; ENC="camels1f"; EXTRA="--dec-lead 1"; TRS=2005-10-01 ;;
  gridmetm9fix3L1)           CORP="$V3/camels_corpus_gridmetfix3_m3_cohort"; ENC="camels1f"; EXTRA="--enc-lead 1"; TRS=2005-10-01 ;;
  fused4m9fix3)              CORP="$V3/camels_corpus_fused4mfix3_m3_cohort"; ENC="camelsm4v2";   TRS=2005-10-01 ;;
  fused4md49fix3)            CORP="$V3/camels_corpus_fused4mfix3_m3_cohort"; ENC="camelsm4v2d4"; TRS=2005-10-01 ;;
  fused4m9)                  CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm4v2";   TRS=2005-10-01 ;;
  fused4md49)                CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm4v2d4"; TRS=2005-10-01 ;;
  # ---- GATE A: the registered pipeline control (PREREG_v2.md MODERN-1) --------
  # Identical corpus and recipe to fused3m, but on P-NEARING -- the FROZEN
  # record's own window -- so the comparison against frozen fused3h1 isolates
  # PIPELINE + PRODUCT change from WINDOW change. It is the registered STOP gate
  # and therefore runs BEFORE the P-MODERN family, not after.
  # Band -0.015..+0.005 solo; halt below -0.03. Scored by analysis/m1_gate_a.py.
  fused3mN|fused3mNsmoke)    CORP="$V3/camels_corpus_fused3m_m3_cohort";   ENC="camelsm3v2"
                             TRS=1999-10-01; TRE=2008-09-30
                             VS=1980-10-01;  VE=1989-09-30
                             TS=1989-10-01;  TE=1999-09-30 ;;
  *) echo "unknown member $M"; exit 2 ;;
esac

# Smoke members run against the SYNTHETIC proxy corpus (product suffixes renamed,
# dates shifted) so the whole launcher can be exercised before the real corpus
# exists. ⛔ Never a scientific result -- see scripts_modern/make_m1_smoke_corpus.py.
# ⚠️ And they run on the CPU by default: the 8 GB card is already at the L56
# batch's MAX=3, and adding an uncounted 4th job is exactly how this box got a
# CUDA OOM that was first misdiagnosed as a hardware fault.
DEV=""
if [ "$SMOKE" = "1" ]; then
  CORP="${M1_SMOKE_CORPUS:-data/modern_corpus/SMOKE_fused4m_cohort}"
  DEV="--device ${M1_SMOKE_DEVICE:-cpu}"
fi

exec >>"$LOG" 2>&1
echo "=== M1 $M s$S $(date +%F_%T) ==="

# ---- GUARD 0: the corpus must be exactly the frozen cohort -----------------
NEED=$($PY -c 'import json;print(len(json.load(open("data/m1_cohort.json"))["cohort"]))')
HAVE=$(ls "$CORP"/*.csv.gz 2>/dev/null | wc -l)
if [ "$SMOKE" = "0" ] && [ "$HAVE" != "$NEED" ]; then
  echo "ABORT GUARD 0: corpus $CORP has $HAVE basins, cohort is $NEED"; exit 1
fi
echo "GUARD 0 OK: $HAVE/$NEED cohort basins in $CORP (enc=$ENC smoke=$SMOKE)"

# ---- train -----------------------------------------------------------------
if [ ! -f "$CKPT" ]; then
  echo "=== TRAIN $M s$S $(date +%F_%T) ==="
  # shellcheck disable=SC2086
  nice -n 5 $PY scripts/train_mblstm.py \
    --epochs 30 --seed "$S" --out "$CKPT" \
    --head quantile --enc-vars "$ENC" --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --train-start "$TRS" --train-end "$TRE" \
    --val-start "$VS" --val-end "$VE" \
    --h1-weight 0.5 $EXTRA $LIMIT $DEV
  echo "=== trained rc=$? $(date +%T) ==="
fi

# ---- GUARD 1: read the protocol back out of the CHECKPOINT, never argv -----
M1_EXTRA="$EXTRA" $PY - "$CKPT" "$LOG" "$ENC" "$TRS" "$TRE" "$VS" "$VE" <<'PYG'
import os, re, sys, torch
ck, log, enc, trs, tre, vs, ve = sys.argv[1:8]
sd = torch.load(ck, map_location="cpu")
cfg = sd["cfg"]
assert cfg["train_start"] == trs, f"train_start {cfg['train_start']} != {trs}"
assert cfg["train_end"] == tre, f"train_end {cfg['train_end']} != {tre}"
assert cfg["val_range"] == [vs, ve], f"val {cfg['val_range']} != {[vs, ve]}"
assert cfg["no_q_input"] is False, "this is the WITH-Q arm"
assert abs(cfg.get("h1_weight", -1) - 0.5) < 1e-9, cfg.get("h1_weight")
ndec = len(cfg["dec_vars"])
# the checkpoint stores weights under "state_dict", not "model" -- caught by
# the proxy-corpus smoke before the real corpus existed
w = sd["state_dict"]["decoder.weight_ih_l0"].shape[1]
# --dec-q appends the last observed discharge AND its mask to the decoder input,
# so the width grows by exactly 2. Measured: dec=6 +2 +1 +27 statics = 36 without
# dec_q, 38 with it. Omitting this aborted GUARD 1 on every dec-q member AFTER a
# successful 30-epoch train (checkpoints were fine; only the guard failed).
dq = bool(cfg.get("dec_q", False))
exp = ndec + 2 + 1 + len(cfg["static_feats"]) + (2 if dq else 0)
assert w == exp, f"decoder width {w} != {exp} (dec={ndec} dec_q={dq})"
# ⭐ And assert the ARM ACTUALLY APPLIED, which matters more than the width: a
# member named *q whose cfg says dec_q=False silently trained the wrong recipe.
if "--dec-q" in os.environ.get("M1_EXTRA", ""):
    assert dq is True, "member expects --dec-q but the checkpoint says dec_q=False"
# Same reasoning for --enc-lead: it changes no tensor SHAPE, so the width check above cannot see
# it. A silently-dropped flag would train a byte-identical duplicate of the unled member and the
# resulting null would be read as a measurement.
_m = re.search(r"--enc-lead\s+(\d+)", os.environ.get("M1_EXTRA", ""))
_want = int(_m.group(1)) if _m else 0
assert int(cfg.get("enc_lead", 0)) == _want, (
    f"enc_lead {cfg.get('enc_lead')} != requested {_want}")
_md = re.search(r"--dec-lead\s+(\d+)", os.environ.get("M1_EXTRA", ""))
_wantd = int(_md.group(1)) if _md else 0
assert int(cfg.get("dec_lead", 0)) == _wantd, (
    f"dec_lead {cfg.get('dec_lead')} != requested {_wantd}")

prods = {v.rsplit("_", 1)[1] for v in cfg["enc_vars"] if "_" in v}
txt = open(log, encoding="utf-8", errors="replace").read()
assert re.search(r"^epoch 30/30 ", txt, re.M), "no 'epoch 30/30' line - partial checkpoint"
if _want:
    assert re.search(rf"^enc_lead={_want} ", txt, re.M), "no enc_lead line in the training log"
m = re.search(r"windows: train=(\d+).*?val=(\d+)", txt, re.S)
print(f"GUARD 1 OK: stations={cfg['n_stations']} train={cfg['train_start']}..{cfg['train_end']} "
      f"enc={len(cfg['enc_vars'])} dec={ndec} products={sorted(prods)} dec_in={w}")
PYG
[ $? -eq 0 ] || { echo "ABORT: GUARD 1 failed"; exit 1; }

# ---- dumps -----------------------------------------------------------------
dump () {  # <tag> <start> <end> <stride> <allleads 0|1>
  local TAG="$1" A="$2" B="$3" ST="$4" AL="$5"
  local OUT="$DD/m1_${M}_s${S}_${TAG}.csv.gz"
  [ -f "$OUT" ] && { echo "[$(date +%T)] $TAG dump exists"; return 0; }
  local FLAG=""; [ "$AL" = "1" ] && FLAG="--dump-all-leads"
  echo "=== DUMP $TAG ($A..$B stride $ST allleads=$AL) $(date +%F_%T) ==="
  # shellcheck disable=SC2086
  nice -n 10 $PY scripts/train_mblstm.py --epochs 0 --init-ckpt "$CKPT" \
    --out "$CKPT.dumpsink_$TAG" \
    --head quantile --enc-vars "$ENC" --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --val-start "$A" --val-end "$B" --val-stride "$ST" \
    --dump-day1 "$OUT" $FLAG $EXTRA $LIMIT $DEV >> "logs/m1_dump_${M}_s${S}_${TAG}.log" 2>&1
  rm -f "$CKPT.dumpsink_$TAG"
  gzip -t "$OUT" || { echo "ABORT: $TAG dump failed gzip -t"; return 1; }
  $PY - "$OUT" "$CKPT" "$TAG" "$M" "$S" "$SMOKE" "$A" "$B" "$AL" "$TRS" "$TRE" <<'PYS'
import datetime as dt, gzip, hashlib, json, os, sys
out, ck, tag, member, seed, smoke, a, b, al, trs, tre = sys.argv[1:12]
# Read the lead settings back out of the CHECKPOINT, not argv: the sidecar has to describe what
# the model actually does, and check_sidecar refuses dec_lead!=0 dumps for any record read.
import torch as _t
_cfg = _t.load(ck, map_location="cpu")["cfg"]
_el, _dl = int(_cfg.get("enc_lead", 0) or 0), int(_cfg.get("dec_lead", 0) or 0)
n = sum(1 for _ in gzip.open(out, "rt")) - 1
assert n > 1000, f"only {n} rows - the dump path emitted nothing"
# ⚠️ The row bar is DERIVED, not a magic constant. LEDGER 56 hard-coded
# `n > 1_500_000`, which was calibrated on the fused3 frame and then wrongly
# failed a perfectly good fused4 dump of 1,458,982 rows, losing its sidecar and
# making it unscoreable. Derive from cohort x window and allow for warmup/gaps.
if smoke == "0":
    ids = json.load(open("data/m1_cohort.json"))["cohort"]
    days = (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days + 1
    leads = 14 if al == "1" else 1
    floor = int(0.40 * len(ids) * days * leads)
    assert n >= floor, f"{n} rows < derived floor {floor} ({len(ids)} basins x {days} d x {leads})"
hdr = gzip.open(out, "rt").readline().strip()
assert hdr == "station_id,t0,h,truth,ylo,ymed,yhi", hdr
json.dump({"member": member, "seed": int(seed), "frame": tag, "rows": n,
           "ckpt": ck, "ckpt_md5": hashlib.md5(open(ck, "rb").read()).hexdigest(),
           # the LABEL must match the window actually trained: Gate B's short arm
           # has train_start 2005-10-01, and calling it "modern-v3" would make the
           # scorer refuse the dump against its own protocol table
           "train_start": trs, "train_end": tre,
           "protocol": ("modern-v3" if trs == "1980-10-01" else
                        "nearing2022" if trs == "1999-10-01" else "modern-v3-9yr"),
           "all_leads": al == "1", "bytes": os.path.getsize(out), "ledger": "MODERN-1",
           "enc_lead": _el, "dec_lead": _dl},
          open(out.replace(".csv.gz", ".json"), "w"), indent=1)
print(f"DUMP OK {tag}: {n} rows -> {out}")
PYS
}

dump val1      "$VS" "$VE" 1 0   # selection frame, every day, lead 1
dump test1     "$TS" "$TE" 1 0   # THE HEADLINE FRAME - read once, at the end

# ---- the all-leads dumps are OPT-IN, and that is a throughput decision ------
# MEASURED 2026-09-29: a 14-lead dump is 490 basins x ~3300 days x 14 leads ~= 22M
# rows. Each one ran >12 h at 5-6.5 GiB RSS and only ~34% GPU -- they are
# RAM/CPU-bound, not GPU-bound. Three of them held the card for 12 h while the
# Gate A verdict, which needs ONLY test1, had been sitting on disk since 08:00.
# With the family at 9 runs that would have been ~108 h of dumping for a table
# nothing had asked these members for.
# ⇒ A SCREEN needs val1 (+ test1 at the end). Only the members that reach the FINAL
# composition need the all-leads table, and they can be re-dumped on demand because
# the checkpoint is kept and `dump` skips any frame whose file already exists.
# Set M1_ALLLEADS=1 for those. Gate A never needed it at all: it is a pipeline
# control, not a reported member.
if [ "${M1_ALLLEADS:-0}" = "1" ]; then
  dump val14   "$VS" "$VE" 1 1   # all-leads table (selection side)
  dump test14  "$TS" "$TE" 1 1   # all-leads table (reported with the headline)
else
  echo "[$(date +%T)] all-leads dumps SKIPPED (M1_ALLLEADS!=1). Re-dump later with"
  echo "  M1_ALLLEADS=1 bash gpu1080/queue_m1_withq.sh $M $S   # training is skipped, the ckpt is kept"
fi
echo "=== M1 $M s$S ALL DONE $(date +%F_%T) ==="
