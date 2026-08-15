#!/usr/bin/env bash
# Waits for the B_n1 queue (or any other local MLX job) to finish, then
# collects a properly-sized J_self round-0 (N=4, T=0) calibration pool on
# the full math/code screened pools, and runs Appendix B calibration on it.
set -uo pipefail
cd "$(dirname "$0")"
LOGS=../logs
DATA=../data/stage2
mkdir -p "$LOGS" "$DATA"
LOG="$LOGS/calib_pool_queue.log"
exec >>"$LOG" 2>&1
echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') START calib_pool_queue ====="

gpu_busy() {
  pgrep -af 'python.*(run_stage2_esc\.py|run_B_n1)' 2>/dev/null \
    | grep -v 'run_calib_pool_queue' | grep -v 'pgrep' | grep -q .
}

POLL=0
while gpu_busy; do
  POLL=$((POLL + 1))
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU busy (poll $POLL); sleep 15m"
  sleep 900
done
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU idle; starting calibration-pool collection"

export PYTHONUNBUFFERED=1
set -a; source ../../.env; set +a

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB calib pool math (84 items, N=4, T=0)"
uv run --with mlx-lm --with mlx --with numpy --with datasets \
  python run_stage2_esc.py \
  --model mlx-community/Qwen3.5-4B-4bit \
  --domain math \
  --task-manifest "$DATA/screen_math_4b.json" \
  --N 4 --T 0 \
  --rubric faithful --judge J_self \
  --out "$DATA/calib_pool_math_4b_self.jsonl" --resume \
  >"$LOGS/calib_pool_math_4b_self.log" 2>&1
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB calib pool math rc=$?"

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB calib pool code (91 items, N=4, T=0)"
uv run --with mlx-lm --with mlx --with numpy --with datasets \
  python run_stage2_esc.py \
  --model mlx-community/Qwen3.5-4B-4bit \
  --domain code \
  --task-manifest "$DATA/screen_code_4b.json" \
  --N 4 --T 0 \
  --rubric faithful --judge J_self \
  --out "$DATA/calib_pool_code_4b_self.jsonl" --resume \
  >"$LOGS/calib_pool_code_4b_self.log" 2>&1
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB calib pool code rc=$?"

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB Appendix B calibration: math"
uv run --with numpy --with scipy \
  python calibrate_strong_noise.py \
  --self-scored "$DATA/calib_pool_math_4b_self.jsonl" \
  --task-manifest "$DATA/screen_math_4b.json" \
  --domain math --rubric faithful \
  --out "$DATA/noise_calibration_math.json" \
  >"$LOGS/noise_calibration_math.log" 2>&1
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB calibration math rc=$?"
tail -10 "$LOGS/noise_calibration_math.log"

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB Appendix B calibration: code"
uv run --with numpy --with scipy \
  python calibrate_strong_noise.py \
  --self-scored "$DATA/calib_pool_code_4b_self.jsonl" \
  --task-manifest "$DATA/screen_code_4b.json" \
  --domain code --rubric faithful \
  --out "$DATA/noise_calibration_code.json" \
  >"$LOGS/noise_calibration_code.log" 2>&1
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB calibration code rc=$?"
tail -10 "$LOGS/noise_calibration_code.log"

echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') calib_pool_queue COMPLETE ====="
echo "AGENT_CALIB_POOL_DONE"
