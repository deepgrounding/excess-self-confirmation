#!/bin/bash
# Waits for the competing calibration_floor 4B MLX job to finish, then runs
# the ESC Week-1 item screens (math, then code) sequentially on Qwen3.5-4B.
# Self-match-safe pgrep pattern: bracketed dot so this script's own command
# line does not satisfy the regex. Marker file protocol:
#   logs/screen_status.txt in {waiting, running_math, running_code, done,
#                              timeout_waiting, failed_math, failed_code}
set -u
cd "$(dirname "$0")"
LOGDIR=../logs
mkdir -p "$LOGDIR" ../data/stage2
STATUS="$LOGDIR/screen_status.txt"

echo "waiting" > "$STATUS"
echo "[$(date '+%F %T')] watcher start; waiting for competing 4B job"

# up to 12 h wait, poll every 60 s
for i in $(seq 1 720); do
  if ! pgrep -f "run_stage2[.]py" >/dev/null 2>&1; then
    break
  fi
  if [ "$i" -eq 720 ]; then
    echo "timeout_waiting" > "$STATUS"
    echo "[$(date '+%F %T')] still busy after 12 h; aborting (no concurrent 4B)"
    exit 1
  fi
  sleep 60
done
sleep 60  # let memory settle

# Operational pass-rate window is [0.25, 0.75]: at N=4 probe granularity
# (multiples of 0.25) the design's "prefer [0.3, 0.6]" admits only rate==0.5;
# [0.25, 0.75] preserves the intent (neither floor nor ceiling) and the
# keep-sort by |rate - 0.45| still prefers mid-band items first. Math pool is
# 600 (raw resume reuses the 300-item probes already on disk); code pool 300.
run_screen () {
  local domain="$1"
  local pool="$2"
  echo "running_${domain}" > "$STATUS"
  echo "[$(date '+%F %T')] starting ${domain} screen (pool=${pool})"
  PYTHONUNBUFFERED=1 uv run --with mlx-lm --with mlx --with numpy \
    --with datasets --with scikit-learn \
    python screen_items.py \
    --model mlx-community/Qwen3.5-4B-4bit \
    --domain "${domain}" --pool "${pool}" --keep 100 --N 4 --seed 13 \
    --lo 0.25 --hi 0.75 \
    --out "../data/stage2/screen_${domain}_4b.json" \
    --raw-out "../data/stage2/screen_${domain}_4b_raw.jsonl" \
    >> "$LOGDIR/screen_${domain}_4b.log" 2>&1
  local ec=$?
  echo "[$(date '+%F %T')] ${domain} screen exit=${ec}"
  if [ "$ec" -ne 0 ]; then
    echo "failed_${domain}" > "$STATUS"
    exit "$ec"
  fi
}

pool_for () { if [ "$1" = "math" ]; then echo 600; else echo 300; fi; }

DOMAINS=("$@")
if [ ${#DOMAINS[@]} -eq 0 ]; then DOMAINS=(code math); fi
for domain in "${DOMAINS[@]}"; do
  run_screen "$domain" "$(pool_for "$domain")"
done
echo "done" > "$STATUS"
echo "[$(date '+%F %T')] all screens complete"
