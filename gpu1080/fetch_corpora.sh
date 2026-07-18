#!/bin/bash
# Download + normalize the 3 CAMELS corpora from Kaggle → ./corpora/camels_corpus_<f>_v2/*.csv.gz
# Needs ~/.kaggle/kaggle.json (a FULL key — scoped KGAT tokens 403 on bulk download
# from datacenter IPs but work from a home IP).
set -e
cd "$(dirname "$0")"
mkdir -p corpora
for F in daymet nldas maurer; do
  OUT="corpora/camels_corpus_${F}_v2"
  if [ "$(ls $OUT/*.csv.gz 2>/dev/null | wc -l)" -ge 500 ]; then echo "$F already have"; continue; fi
  mkdir -p "$OUT" "raw_$F"
  echo "downloading $F …"
  kaggle datasets download "andrewnakas/rw2-camels-corpus-$F" -p "raw_$F" --unzip
  python3 - "$F" << 'PYEOF'
import gzip, shutil, os, glob, sys
f = sys.argv[1]; out = f"corpora/camels_corpus_{f}_v2"; n = 0
for p in glob.glob(f"raw_{f}/**/*.csv", recursive=True):
    b = os.path.basename(p)
    if b.startswith("._"): continue
    with open(p, "rb") as fi, gzip.open(f"{out}/{b.split('.')[0]}.csv.gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    n += 1
print(f"{f}: gzipped {n} basins")
PYEOF
  rm -rf "raw_$F"
done
echo "corpora ready under ./corpora/"
