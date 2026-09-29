#!/usr/bin/env bash
# MODERN-1 — re-arm every chain whose work is unfinished. Idempotent: safe to run
# repeatedly, by hand or from @reboot.
#
# WHY: all four MODERN-1 chains are setsid shells, not cron jobs, so a reboot stops
# every one of them. The previous resume script was written for an earlier stage (it
# resumes the Daymet build, now complete, and a hardcoded gridmetm job list, also
# complete) and knows nothing about Gate A, the fused family, NLDAS, the 4-product
# fuse, or the NH member.
#
# ⚠️ And the previous resume was a NO-OP for a worse reason: a crontab entry existed
# for a script that had never been rsync'd to the box. ⇒ m1_boot_resume.sh now merely
# delegates here, so the EXISTING crontab line keeps working and there is one file to
# get onto the box instead of two. Verify the COMPOSITION (run it and read the log),
# never just the components.
#
# ⛔ GPU SERIALISATION: only chain_fused3m2 schedules GPU work. chain_nh3b waits for
# chain_fused3m2 to be GONE (a quiet box is not an idle box), and chain_fuse4 launches
# no GPU work at all. So starting all of them together is safe.
#
# usage:  bash gpu1080/m1_rearm.sh          # re-arm what is unfinished
#         touch gpu1080/M1_STOP             # disarm everything
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_rearm.log
exec >>"$LOG" 2>&1
echo "=== M1 RE-ARM $(date +%F_%T) uptime:$(uptime -p) ==="
[ -f gpu1080/M1_STOP ] && { echo "M1_STOP present - disarmed, doing nothing"; exit 0; }
PY=gpu1080/.venv/bin/python

alive () { [ "$(ps -eo cmd --no-headers | grep -c "$1")" -gt 0 ]; }   # caller passes a char class
start () { # <name> <script>
  echo "[$(date +%T)] starting $1"
  nohup setsid bash "$2" > /dev/null 2>&1 < /dev/null &
  sleep 3
}

# ---- state ------------------------------------------------------------------
GAV=none
if [ -f benchmarks/m1_gate_a_verdict.json ]; then
  GAV=$($PY -c 'import json;print("pass" if json.load(open("benchmarks/m1_gate_a_verdict.json"))["passed"] else "FAIL")' 2>/dev/null || echo unreadable)
fi
DMM=$(ls data/mblstm/l51_dumps/m1_daymetm_s*_test14.csv.gz 2>/dev/null | wc -l)
FQ=$(ls data/mblstm/l51_dumps/m1_fused3md3q_s*_val1.csv.gz 2>/dev/null | wc -l)
NLM=$(ls data/modern_corpus/_nldas_monthly/*.npz 2>/dev/null | sed 's/.*nldas_//;s/\.npz//' | awk '$1<201400' | wc -l)
F4=$(ls data/modern_corpus/v3/camels_corpus_fused4m_m3_cohort/*.csv.gz 2>/dev/null | wc -l)
NH3=$(ls data/mblstm/l51_dumps/m1_nhar0m3_s*_val1.csv.gz 2>/dev/null | wc -l)
echo "state: gateA=$GAV daymetm=$DMM/3 fused3md3q=$FQ/3 nldas_months=$NLM/408 fused4m=$F4 nhar0m3=$NH3/3"

# ---- 1. chain_fused3m2 -- the only chain that schedules GPU work ------------
# Every step of it is idempotent: the daymet wait passes at once, assemble and fuse
# skip when their artifacts exist, cohort trees are rebuilt from scratch, and
# queue_m1_withq.sh skips training whenever the checkpoint is already there.
if [ "$GAV" = "FAIL" ]; then
  echo "Gate A FAILED its falsifier -- NOT re-arming the P-MODERN family. Diagnose first."
elif alive "chain_fused3m[2]\.sh"; then
  echo "chain_fused3m2 already running"
elif [ "$GAV" = "none" ] || [ "$DMM" -lt 3 ]; then
  start chain_fused3m2 gpu1080/chain_fused3m2.sh
else
  echo "chain_fused3m2 work complete (gateA=$GAV daymetm=$DMM/3)"
fi

# ---- 2. NLDAS fetch (resumes from the month cache) -------------------------
if [ "$NLM" -ge 408 ]; then
  echo "nldas back-extension complete ($NLM/408)"
elif alive "chain_nldas4[b]\.sh"; then
  echo "chain_nldas4b already running"
else
  start chain_nldas4b gpu1080/chain_nldas4b.sh
fi

# ---- 3. the 4-product fuse (build only, no GPU) ----------------------------
if [ "$F4" -ge 400 ]; then
  echo "4-product corpus already built ($F4 cohort basins)"
elif alive "chain_fuse[4]\.sh"; then
  echo "chain_fuse4 already running"
else
  start chain_fuse4 gpu1080/chain_fuse4.sh
fi

# ---- 4. the 3-product NH member (waits for chain_fused3m2 to be gone) ------
# ⛔ A Gate A failure means "halt ALL GPU and diagnose" (PREREG_v2.md MODERN-1), and
# this is GPU work. Found by the branch test: with GAV=FAIL the family was correctly
# withheld, but nh3b would then see no chain_fused3m2 alive, take the card at once and
# train right through the registered halt -- the halt would have stopped the members it
# named and nothing else. NLDAS (network) and the fuse (build-only) are unaffected and
# continue.
if [ "$GAV" = "FAIL" ]; then
  echo "Gate A FAILED -- NOT starting chain_nh3b either; the registered halt covers ALL GPU work."
elif [ "$NH3" -ge 3 ]; then
  echo "nhar0m3 complete ($NH3/3)"
elif alive "chain_nh3[b]\.sh"; then
  echo "chain_nh3b already running"
else
  start chain_nh3b gpu1080/chain_nh3b.sh
fi

echo "--- after re-arm ---"
for p in "chain_fused3m[2]" "chain_nldas4[b]" "chain_fuse[4]" "chain_nh3[b]"; do
  echo "  $p: $(ps -eo cmd --no-headers | grep -c "$p\.sh")"
done
echo "=== M1 RE-ARM DONE $(date +%F_%T) ==="
