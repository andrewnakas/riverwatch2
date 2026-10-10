"""GATE 2 structural test: prove the splice touches EXACTLY the newest K encoder
forcing rows and nothing else. A splice that silently no-ops would read
delta = 0.000000 and look like a clean pass."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "scripts")
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("tm", "scripts/train_mblstm.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)

NL = Path("data/modern_corpus/v3/camels_corpus_nldas_g1sub")
OM = Path("data/modern_corpus/v3/camels_corpus_openmeteo_m3_cohort")
import json
ck = "data/mblstm/gpu_ckpts/m1_nldasm9_s501.pt"
import torch
cfg = torch.load(ck, map_location="cpu", weights_only=False)["cfg"]
LV = cfg["enc_vars"]
files = sorted(NL.glob("*.csv.gz"))[:3]
stations = [tm.load_station(f, LV) for f in files]
stations = [s for s in stations if s]
attrs = json.load(open("data/camels_attrs.json"))
corpus = tm.Corpus(stations, attrs, pd.Timestamp(cfg.get("train_end", "2014-09-30")),
                   LV, LV, stats=cfg, q_transform=cfg.get("q_transform", "linear"),
                   static_feats=cfg["static_feats"], head="quantile", load_vars=LV)
corpus.no_q_input = False; corpus.enc_lead = 0; corpus.dec_lead = 0
corpus.ar_mask_p = 0.0; corpus.forcing_noise = 0.0; corpus.gfs = None
corpus.mix_lookups = None; corpus.dec_q = False

w = corpus.window_index(pd.Timestamp("2015-01-01"), pd.Timestamp("2018-09-01"))
assert len(w), "no windows"
si, t0 = int(w[0][0]), int(w[0][1])

base_enc = corpus.sample(si, t0, None)[1 - 1]      # x_enc is element 0
base_enc = np.array(base_enc, copy=True)

K = 4
n = tm.attach_alt_forcing(corpus, OM, K)
assert n > 0, "no stations wired"
spl_enc = np.array(corpus.sample(si, t0, None)[0], copy=True)

NW = len(LV)
fail = []
if base_enc.shape != spl_enc.shape: fail.append("shape changed")
older_b, older_s = base_enc[:-K, :NW], spl_enc[:-K, :NW]
if not np.array_equal(older_b, older_s):
    fail.append(f"OLDER rows changed ({int((older_b!=older_s).sum())} cells) -- splice is not confined to the tail")
tail_b, tail_s = base_enc[-K:, :NW], spl_enc[-K:, :NW]
nd = int((tail_b != tail_s).sum())
if nd == 0:
    fail.append("NEWEST rows are IDENTICAL -- the splice is a silent no-op")
if not np.array_equal(base_enc[:, NW:], spl_enc[:, NW:]):
    fail.append("non-forcing channels (q / mask / doy / statics) changed")
print(f"station={corpus.stations[si]['id']} t0={t0} enc={base_enc.shape} wx_cols={NW}")
print(f"  older rows identical : {np.array_equal(older_b, older_s)}")
print(f"  newest {K} rows differ: {nd}/{tail_b.size} cells")
print(f"  q/mask/doy/static untouched: {np.array_equal(base_enc[:, NW:], spl_enc[:, NW:])}")
print(f"  mean |delta| on spliced tail (z units): {np.abs(tail_b-tail_s).mean():.4f}")
print("FAIL: " + "; ".join(fail) if fail else "PASS: splice touches exactly the newest %d forcing rows" % K)
sys.exit(1 if fail else 0)
