#!/bin/bash
# Ledger 45 track 2: batched re-fetch of the remaining CCI SM files.
# Replaces the xargs-per-file fetchers: one bash+TLS handshake PER FILE was the
# bottleneck (27 files/min); curl's native parallel mode reuses connections and
# measures ~70 files/min per process. Idempotent -- filters the URL list down to
# files not already present at >10 KB before each batch.
# Usage: ledger45_sat_fetch_fast.sh <variant-dir> <url-list> [parallel-max]
set -u
dest="$1"; list="$2"; pmax="${3:-24}"
mkdir -p "$dest"; cd "$dest" || exit 1
while :; do
    missing=$(mktemp)
    while read -r u; do
        f=$(basename "$u")
        if [ ! -s "$f" ] || [ "$(stat -c%s "$f")" -lt 10240 ]; then echo "$u"; fi
    done < "$list" > "$missing"
    n=$(wc -l < "$missing")
    echo "$(date +%H:%M:%S) remaining: $n in $dest"
    if [ "$n" -eq 0 ]; then rm -f "$missing"; echo "FAST_FETCH_COMPLETE $dest"; exit 0; fi
    # batches of 200 so a failure costs little and progress is visible
    split -l 200 "$missing" /tmp/batch_$$_
    for b in /tmp/batch_$$_*; do
        curl -sS --parallel --parallel-max "$pmax" --retry 2 --retry-delay 3 \
             --connect-timeout 20 --max-time 300 --remote-name-all \
             $(tr '\n' ' ' < "$b") 2>/dev/null
        rm -f "$b"
    done
    rm -f "$missing"
    # loop re-checks; exits when nothing is missing
done
