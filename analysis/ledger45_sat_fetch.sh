#!/bin/bash
# Ledger 45 track 2 stage 2: full acquisition 1980-01-01..2008-12-31.
#   1. GlobSnow v3.0 L3A daily SWE (open HTTPS, flat NetCDF4 dir)
#   2. ESA CCI SM COMBINED v07.1 (CEDA, anonymous)
#   3. ESA CCI SM break_adjusted_COMBINED v07.1 (CEDA, anonymous)
# Idempotent: skips files that already exist with size > 10 KB.
set -u
BASE=~/riverwatch2/data/satellite
mkdir -p $BASE/globsnow_full $BASE/cci_combined_v071 $BASE/cci_adjusted_v071 $BASE/fetch_lists

fetch_list () {  # $1 = url list file, $2 = dest dir, $3 = parallelism
    cat "$1" | xargs -P "$3" -I{} bash -c '
        u="{}"; d="'"$2"'"; f="$d/$(basename $u)"
        if [ -s "$f" ] && [ $(stat -c%s "$f") -gt 10240 ]; then exit 0; fi
        curl -s --retry 3 --retry-delay 5 -o "$f.part" "$u" && mv "$f.part" "$f"
    '
}

# ---- 1. GlobSnow ----------------------------------------------------------
GS=https://www.globsnow.info/swe/archive_v3.0/L3A_daily_SWE/NetCDF4
GLIST=$BASE/fetch_lists/globsnow_urls.txt
if [ ! -s $GLIST ]; then
    curl -s "$GS/" | grep -oE 'href="[0-9]{8}_northern_hemisphere_swe_0.25grid.nc"' \
      | sed 's/href="//;s/"//' \
      | awk -v gs="$GS" '{d=substr($0,1,8); if (d>="19800101" && d<="20081231") print gs"/"$0}' \
      | sort -u > $GLIST.tmp && mv $GLIST.tmp $GLIST
fi
echo "globsnow urls: $(wc -l < $GLIST)"
fetch_list $GLIST $BASE/globsnow_full 4
echo "GLOBSNOW_FETCH_PASS_DONE $(ls $BASE/globsnow_full/*.nc 2>/dev/null | wc -l)"

# ---- 2+3. CCI SM, both variants ------------------------------------------
CEDA=https://dap.ceda.ac.uk/neodc/esacci/soil_moisture/data/daily_files
for variant in COMBINED break_adjusted_COMBINED; do
    if [ "$variant" = COMBINED ]; then dest=$BASE/cci_combined_v071; else dest=$BASE/cci_adjusted_v071; fi
    CLIST=$BASE/fetch_lists/cci_${variant}_urls.txt
    if [ ! -s $CLIST ]; then
        : > $CLIST.tmp
        for y in $(seq 1980 2008); do
            curl -s --retry 3 "$CEDA/$variant/v07.1/$y/" \
              | grep -oE 'href="ESACCI[^"]+\.nc"' | sed 's/href="//;s/"//' \
              | awk -v u="$CEDA/$variant/v07.1/$y" '{print u"/"$0}' >> $CLIST.tmp
        done
        sort -u $CLIST.tmp > $CLIST && rm $CLIST.tmp
    fi
    echo "cci $variant urls: $(wc -l < $CLIST)"
    fetch_list $CLIST $dest 6
    echo "CCI_${variant}_FETCH_PASS_DONE $(ls $dest/*.nc 2>/dev/null | wc -l)"
done
echo "ALL_FETCH_PASSES_DONE"
