#!/bin/bash
# Build neuralhydrology GenericDataset data (netCDF + attributes.csv) for all 3
# forcings from ./corpora/ using the repo adapter. → ./nh_data/{daymet,nldas,maurer}/
set -e
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"
ADAPTER=../scripts/corpus_to_nh.py    # reads ../data/camels_{gauge_ids,attrs}.json
for F in daymet nldas maurer; do
  if [ -f "nh_data/$F/basins.txt" ]; then echo "$F NH data exists"; continue; fi
  echo "building NH data: $F"
  python3 "$ADAPTER" --forcing "$F" \
    --corpus-dir "corpora/camels_corpus_${F}_v2" \
    --out "nh_data/$F"
done
echo "NH data ready under ./nh_data/"
