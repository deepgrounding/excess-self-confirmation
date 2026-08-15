#!/bin/bash
# Wait for calibration_floor's Qwen3.5-4B run_stage2.py to exit, then start
# the ESC 4B plumbing pilot. Poll every 3 minutes; also require that no other
# Qwen3.5-4B python job remains, and that free memory looks usable.
set -u
ROOT="/Users/minggguangchen/Desktop/2026/manuscript/recursive_self_improvement/esc_manuscript"
LOG="$ROOT/logs/watch_then_pilot_4b.log"
PILOT_LOG="$ROOT/logs/pilot_4b.log"
STATUS="$ROOT/logs/watch_status.txt"
mkdir -p "$ROOT/logs"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }
set_status() { echo "$*" > "$STATUS"; }

still_running() {
  pgrep -f 'run_stage2.py --model mlx-community/Qwen3.5-4B-4bit' >/dev/null 2>&1
}

other_4b_busy() {
  # any other python holding the same 4B checkpoint (exclude this watcher)
  pgrep -af 'Qwen3.5-4B-4bit' 2>/dev/null | grep -v 'watch_then_pilot_4b' | grep -v 'grep' | grep -q .
}

free_pct() {
  memory_pressure 2>/dev/null | awk -F': ' '/System-wide memory free percentage/ {gsub(/%/,"",$2); print $2; exit}'
}

log "WATCHER_START waiting for calibration_floor run_stage2.py (Qwen3.5-4B) to finish"
set_status "waiting_for_calibration_floor"

POLL=180  # 3 minutes
MAX_WAIT_H=8
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))

while still_running; do
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    log "WATCHER_TIMEOUT after ${MAX_WAIT_H}h; giving up without starting pilot"
    set_status "timeout"
    exit 2
  fi
  fp=$(free_pct || echo '?')
  log "still running (pid $(pgrep -f 'run_stage2.py --model mlx-community/Qwen3.5-4B-4bit' | tr '\n' ' ')); free%~${fp}; sleep ${POLL}s"
  set_status "waiting_for_calibration_floor free_pct=${fp}"
  sleep "$POLL"
done

log "calibration_floor run_stage2.py gone; cooling 60s for memory release"
set_status "cooling_down"
sleep 60

# If something else grabbed the 4B, wait a bit more (up to 30 min)
cool_deadline=$(( $(date +%s) + 1800 ))
while other_4b_busy; do
  if [ "$(date +%s)" -ge "$cool_deadline" ]; then
    log "WATCHER_ABORT another Qwen3.5-4B job still present after cooldown"
    set_status "aborted_other_4b_busy"
    exit 3
  fi
  log "another Qwen3.5-4B process still present; sleep 60s"
  sleep 60
done

fp=$(free_pct || echo 0)
log "memory free%~${fp}; starting ESC 4B pilot"
set_status "starting_pilot free_pct=${fp}"

cd "$ROOT" || exit 1
export HF_HUB_OFFLINE=1
{
  echo "[$(ts)] PILOT_4B_START"
  .venv/bin/python -u code/pilot_trajectory.py \
    --model mlx-community/Qwen3.5-4B-4bit \
    --tag 4b
  ec=$?
  echo "[$(ts)] PILOT_4B_EXIT code=$ec"
  exit $ec
} 2>&1 | tee -a "$PILOT_LOG" | tee -a "$LOG"
ec=${PIPESTATUS[0]}

if [ "$ec" -eq 0 ]; then
  log "WATCHER_DONE pilot succeeded"
  set_status "pilot_done"
else
  log "WATCHER_DONE pilot failed exit=$ec"
  set_status "pilot_failed exit=$ec"
fi
exit "$ec"
