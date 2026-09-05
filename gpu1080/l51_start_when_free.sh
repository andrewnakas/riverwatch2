#!/usr/bin/env bash
# Wait until at most N trainers are running, then launch a lane. Keeps the 8 GB
# card from being oversubscribed (each trainer holds ~1.6 GB) while lanes A/B
# drain their final jobs.
# usage: l51_start_when_free.sh <max_running> <lane> <member:seed> ...
set -u
cd ~/riverwatch2
MAX="${1:?max}"; LANE="${2:?lane}"; shift 2
while [ "$(pgrep -fc 'train_mblstm.py --epochs 30' || echo 0)" -gt "$MAX" ]; do sleep 120; done
exec bash gpu1080/run_l51c_lane.sh "$LANE" "$@"
