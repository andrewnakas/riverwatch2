#!/bin/bash
# Ledger 45 track 2: CCI SM COMBINED + break_adjusted_COMBINED v07.1, 1980-2008,
# run concurrently with the globsnow pass of ledger45_sat_fetch.sh.
# Uses PID-suffixed temp files so a concurrent second pass cannot corrupt.
set -u
BASE=~/riverwatch2/data/satellite
CEDA=https://dap.ceda.ac.uk/neodc/esacci/soil_moisture/data/daily_files
mkdir -p $BASE/cci_combined_v071 $BASE/cci_adjusted_v071 $BASE/fetch_lists
for variant in COMBINED break_adjusted_COMBINED; do
    if [ "$variant" = COMBINED ]; then dest=$BASE/cci_combined_v071; else dest=$BASE/cci_adjusted_v071; fi
    CLIST=$BASE/fetch_lists/cci_${variant}_urls.txt
    if [ ! -s $CLIST ]; then
        : > $CLIST.tmp2
        for y in $(seq 1980 2008); do
            curl -s --retry 3 "$CEDA/$variant/v07.1/$y/" \
              | grep -oE 'href="ESACCI[^"]+\.nc"' | sed 's/href="//;s/"//' \
              | awk -v u="$CEDA/$variant/v07.1/$y" '{print u"/"$0}' >> $CLIST.tmp2
        done
        sort -u $CLIST.tmp2 > $CLIST && rm $CLIST.tmp2
    fi
    echo "cci $variant urls: $(wc -l < $CLIST)"
    cat "$CLIST" | xargs -P 6 -I{} bash -c '
        u="{}"; d="'"$dest"'"; f="$d/$(basename $u)"
        if [ -s "$f" ] && [ $(stat -c%s "$f") -gt 10240 ]; then exit 0; fi
        t="$f.part.$$"
        curl -s --retry 3 --retry-delay 5 -o "$t" "$u" && mv "$t" "$f"
    '
    echo "CCI_${variant}_STANDALONE_DONE $(ls $dest/*.nc 2>/dev/null | wc -l)"
done
echo "CCI_STANDALONE_ALL_DONE"
