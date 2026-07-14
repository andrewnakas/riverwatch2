#!/bin/bash
# Package the CAMELS corpora + static attrs as Kaggle Datasets for the no-q
# training campaign. Run LOCALLY (needs the SD-mounted corpora + kaggle CLI).
#
# Prereqs:
#   pip install kaggle
#   mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#   (kaggle.json = {"username":"<u>","key":"<KEY>"} — NEVER commit it.)
#
# Datasets created (slugs are <username>/<name>):
#   rw2-camels-corpus-daymet / -maurer / -nldas   (~265MB each)
#   rw2-camels-static                             (attrs + gauge ids + registry)
#
# Usage:  bash notebooks/package_kaggle_data.sh <kaggle-username>
set -euo pipefail
USER=${1:?"pass your kaggle username: bash package_kaggle_data.sh <username>"}
SD=${SD:-/Volumes/STORAGE_SD/riverwatch2_data}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAGE=$(mktemp -d)
echo "staging in $STAGE"

command -v kaggle >/dev/null || { echo "install the kaggle CLI: pip install kaggle"; exit 1; }

# --- static Dataset (small, load-bearing) ---
mkdir -p "$STAGE/static"
cp "$REPO/data/camels_attrs.json" "$REPO/data/camels_gauge_ids.json" \
   "$REPO/data/stations_40_enriched.json" "$STAGE/static/"
cat > "$STAGE/static/dataset-metadata.json" <<EOF
{"title":"rw2-camels-static","id":"$USER/rw2-camels-static","licenses":[{"name":"CC0-1.0"}]}
EOF
echo "creating rw2-camels-static ..."
kaggle datasets create -p "$STAGE/static" --dir-mode zip || \
  kaggle datasets version -p "$STAGE/static" -m "update" --dir-mode zip

# --- per-forcing corpora (strip AppleDouble ._ files from the exFAT mount) ---
for F in daymet maurer nldas; do
  SRC="$SD/camels_corpus_${F}_v2"
  [ -d "$SRC" ] || { echo "MISSING corpus $SRC — is the SD mounted?"; exit 1; }
  DST="$STAGE/corpus_$F/camels_corpus_${F}_v2"
  mkdir -p "$DST"
  echo "copying $F corpus (stripping ._ files) ..."
  # copy only the real csv.gz, skip macOS AppleDouble siblings
  find "$SRC" -name '*.csv.gz' ! -name '._*' -exec cp {} "$DST/" \;
  n=$(ls "$DST" | wc -l | tr -d ' ')
  echo "  $F: $n basin files"
  cat > "$STAGE/corpus_$F/dataset-metadata.json" <<EOF
{"title":"rw2-camels-corpus-$F","id":"$USER/rw2-camels-corpus-$F","licenses":[{"name":"CC0-1.0"}]}
EOF
  echo "creating rw2-camels-corpus-$F ..."
  kaggle datasets create -p "$STAGE/corpus_$F" --dir-mode zip || \
    kaggle datasets version -p "$STAGE/corpus_$F" -m "update" --dir-mode zip
done

echo "DONE. Datasets under $USER/. Add them as inputs to the Kaggle notebooks:"
echo "  rw2-camels-corpus-{daymet,maurer,nldas}, rw2-camels-static"
rm -rf "$STAGE"
