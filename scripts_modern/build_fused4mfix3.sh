#!/usr/bin/env bash
# Build the 4-product fused corpus on the CORRECTED gridMET, needed in BOTH verdict branches:
# if fix3 wins, the fused members should be rebuilt on the corrected product to capture the gain;
# if fix3 loses, this is what the retraction rebuilds from.
#
# ⭐ The trick that avoids a column rename: build_fused_corpus.py names columns {var}_{product},
# so passing product "gridmetfix3" would emit _gridmetfix3 columns and break the camelsm4v2 var-set
# names, which expect _gridmet. Instead a STAGING src root is assembled from symlinks where
# camels_corpus_gridmet_m3 points at the corrected data. The originals are never touched and the
# output is a drop-in for every existing --enc-vars set.
set -eu
cd ~/riverwatch2
V3=data/modern_corpus/v3
STAGE=$V3/.stage_fix3
OUT=$V3/camels_corpus_fused4mfix3_m3

rm -rf "$STAGE"; mkdir -p "$STAGE"
ln -s "$(readlink -f $V3/camels_corpus_daymet_m3)"      "$STAGE/camels_corpus_daymet_m3"
ln -s "$(readlink -f $V3/camels_corpus_nldas_m3)"       "$STAGE/camels_corpus_nldas_m3"
ln -s "$(readlink -f $V3/camels_corpus_nclimgrid_m3)"   "$STAGE/camels_corpus_nclimgrid_m3"
ln -s "$(readlink -f $V3/camels_corpus_gridmetfix3_m3)" "$STAGE/camels_corpus_gridmet_m3"
echo "staged:"; ls -l "$STAGE" | sed 's/^/  /'

# assert the staged gridmet really IS the corrected one, not the original
A=$(gpu1080/.venv/bin/python - <<'PY'
import pandas as pd, glob
f = sorted(glob.glob("data/modern_corpus/v3/.stage_fix3/camels_corpus_gridmet_m3/*.csv.gz"))[0]
g = f.replace("/.stage_fix3/camels_corpus_gridmet_m3/", "/camels_corpus_gridmet_m3/")
a = pd.read_csv(f, dtype={"date": str}).set_index("date")["precipitation_sum"]
b = pd.read_csv(g, dtype={"date": str}).set_index("date")["precipitation_sum"]
i = a.index.intersection(b.index)
import numpy as np
same = float(np.mean(np.isclose(a.loc[i], b.loc[i], atol=1e-9)))
print("CORRECTED" if same < 0.95 else "ORIGINAL")
PY
)
echo "staged gridmet is: $A"
[ "$A" = "CORRECTED" ] || { echo "ABORT: staging points at the ORIGINAL gridMET"; exit 1; }

nice -n 18 gpu1080/.venv/bin/python build_fused_corpus.py \
  --src "$STAGE" --suffix _m3 --products daymet,nldas,gridmet,nclimgrid \
  --out "$OUT" 2>&1 | tail -12

N=$(ls "$OUT"/*.csv.gz 2>/dev/null | wc -l)
echo "fused basins: $N"
[ "$N" -ge 600 ] || { echo "ABORT: only $N basins"; exit 1; }

echo "=== column check (must carry _gridmet, never _gridmetfix3) ==="
gpu1080/.venv/bin/python - <<'PY'
import glob, pandas as pd
f = sorted(glob.glob("data/modern_corpus/v3/camels_corpus_fused4mfix3_m3/*.csv.gz"))[0]
c = list(pd.read_csv(f, nrows=2).columns)
print(f"{len(c)} columns; gridmet-suffixed: {[x for x in c if x.endswith('_gridmet')]}")
assert not any(x.endswith("_gridmetfix3") for x in c), "WRONG SUFFIX"
assert sum(x.endswith("_gridmet") for x in c) == 6, "expected 6 _gridmet columns"
ref = sorted(pd.read_csv(sorted(glob.glob(
    "data/modern_corpus/v3/camels_corpus_fused4m_m3/*.csv.gz"))[0], nrows=2).columns)
assert sorted(c) == ref, f"schema differs from fused4m_m3:\n  extra {set(c)-set(ref)}\n  missing {set(ref)-set(c)}"
print("✅ schema byte-identical to camels_corpus_fused4m_m3 -- drop-in for every var-set")
PY

# cohort tree
CO="${OUT}_cohort"; rm -rf "$CO"; mkdir -p "$CO"
gpu1080/.venv/bin/python - <<'PY'
import json, os
out = "data/modern_corpus/v3/camels_corpus_fused4mfix3_m3"
co = out + "_cohort"
n = 0
for g in json.load(open("data/m1_cohort.json"))["cohort"]:
    s = os.path.abspath(f"{out}/{g}.csv.gz")
    if os.path.exists(s):
        os.symlink(s, f"{co}/{g}.csv.gz"); n += 1
print(f"cohort tree: {n} symlinks")
assert n == 490, n
PY
rm -rf "$STAGE"
echo "DONE: $OUT (+ _cohort)"
