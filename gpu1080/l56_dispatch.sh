#!/usr/bin/env bash
# LEDGER 56 — route a member to the launcher that owns it, so ONE batch
# controller can schedule arms from all three sub-ledgers at a fixed concurrency.
# usage: l56_dispatch.sh <member> <seed>
set -u
cd ~/riverwatch2
M="${1:?member}"; S="${2:?seed}"
case "$M" in
  fused3h1d3|fused4h1d4|fused3h1d3smoke)            Q=gpu1080/queue_l56_withq.sh ;;
  daymeth1d3|maurerh1d3|nldash1d3|aorch1d4|daymeth1d3smoke) Q=gpu1080/queue_l56b_withq.sh ;;
  fused3h1d3q|fused3h1q|fused3h1d3qsmoke)           Q=gpu1080/queue_l56c_withq.sh ;;
  fused3h1decm|fused3h1decn|fused3h1decmsmoke)      Q=gpu1080/queue_l56d_withq.sh ;;
  *) echo "l56_dispatch: unknown member $M" >&2; exit 1 ;;
esac
exec bash "$Q" "$M" "$S"
