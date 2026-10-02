#!/usr/bin/env bash
# MODERN-1 — re-arm every endgame chain whose work is unfinished. Idempotent; safe to run
# repeatedly, by hand or from @reboot (m1_boot_resume.sh delegates here).
#
# ⚠️ REWRITTEN 2026-10-02. The previous version was written for an earlier stage and
# referenced chains that have all since completed (chain_fused3m2, chain_nldas4b's
# predecessor, chain_nh3b). A resume script that restarts the wrong chains is worse than
# none: it reports success while recovering nothing. Keyed on the CURRENT six.
#
# ⛔ GPU SERIALISATION SURVIVES A RE-ARM, so starting all six together is safe: each chain
# gates itself on its own predecessor and on batch=0/gpu=0 over three consecutive polls.
#   chain_nldas4b  network only
#   chain_fuse4    CPU only, build-only, launches no GPU work
#   chain_shipbar  GPU, gated on the no-q chain (long gone)
#   chain_h1w      GPU, gated on chain_shipbar being gone
#   chain_p4       GPU, gated on the 4-product corpus + a free card
#   chain_p4noq    GPU, gated on chain_p4 being gone
#
# usage:  bash gpu1080/m1_rearm.sh     ·  disarm: touch gpu1080/M1_STOP
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_rearm.log
exec >>"$LOG" 2>&1
echo "=== M1 RE-ARM $(date +%F_%T) uptime:$(uptime -p) ==="
[ -f gpu1080/M1_STOP ] && { echo "M1_STOP present - disarmed"; exit 0; }

alive () { [ "$(ps -eo cmd --no-headers | awk -v n="$1" 'index($0,n)>0 && $0 !~ /grep/ {c++} END{print c+0}')" -gt 0 ]; }
start () { echo "[$(date +%T)] starting $1"; nohup setsid bash "gpu1080/$1" >/dev/null 2>&1 </dev/null & sleep 3; }

NL=$(ls data/modern_corpus/_nldas_monthly/*.npz 2>/dev/null | sed 's/.*nldas_//;s/\.npz//' | awk '$1<201400' | wc -l)
F4=$(ls data/modern_corpus/v3/camels_corpus_fused4m_m3_cohort/*.csv.gz 2>/dev/null | wc -l)
SB=$(ls data/mblstm/l51_dumps/m1_*_s50[45]_val1.csv.gz 2>/dev/null | wc -l)
echo "state: nldas=$NL/408 fused4m=$F4 shipbar_dumps=$SB"

# <chain> <done-condition>
try () { # <script> <done 0|1> <label>
  if [ "$2" = "1" ]; then echo "  $3: already complete"
  elif alive "$1"; then echo "  $3: already running"
  else start "$1"; fi
}
try chain_nldas4b.sh "$([ "$NL" -ge 408 ] && echo 1 || echo 0)" "NLDAS fetch"
try chain_fuse4.sh   "$([ "$F4" -ge 400 ] && echo 1 || echo 0)" "4-product corpus"
try chain_shipbar.sh "$([ -f benchmarks/m1_withq_roster_5seed.json ] && echo 1 || echo 0)" "with-q ship bar"
try chain_h1w.sh     "$([ -f benchmarks/m1_h1w_08_vs_05.json ] && echo 1 || echo 0)" "h1-weight ladder"
try chain_p4.sh      "$([ -f benchmarks/m1_p4_roster.json ] && echo 1 || echo 0)" "4-product with-q"
try chain_p4noq.sh   "$([ -f benchmarks/m1_p4noq_roster.json ] && echo 1 || echo 0)" "4-product no-q"

echo "--- after re-arm ---"
for n in chain_nldas4b.sh chain_fuse4.sh chain_shipbar.sh chain_h1w.sh chain_p4.sh chain_p4noq.sh; do
  echo "  $n: $(ps -eo cmd --no-headers | awk -v n="$n" 'index($0,n)>0 && $0 !~ /grep/ {c++} END{print c+0}')"
done
echo "=== M1 RE-ARM DONE $(date +%F_%T) ==="
