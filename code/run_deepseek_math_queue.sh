#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 math dual-bound cross-check (post-hoc, added 2026-08-03
# after Gemma and Haiku both failed the noise-match gate on math, to test
# whether the polarization is judge-specific or structural to math grading;
# see main.md Appendix B). Calibration already ran (noise_calibration_math_
# deepseek.json) and failed the gate a third time -- this runs the resulting
# dual-bound (optimistic/conservative tau) full-scale arms.
#
# Generation is local MLX (same GPU as any other run_stage2_esc.py job);
# judging is remote via OpenRouter. Waits for other local-inference jobs to
# clear before starting, same pattern as run_strongtilde_oracle_placebo_queue.sh.
set -uo pipefail
cd "$(dirname "$0")"
LOGS=../logs
DATA=../data/stage2
mkdir -p "$LOGS" "$DATA"
LOG="$LOGS/strongtilde_deepseek_queue.log"
exec >>"$LOG" 2>&1
echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') START strongtilde_deepseek_queue ====="

gpu_busy() {
  pgrep -af 'python.*(run_stage2_esc\.py|run_B_n1|run_calib_pool|calibrate_strong_noise)|run_peer_pipeline\.sh|run_A_pipeline\.sh|run_strongtilde_oracle_placebo_queue\.sh' 2>/dev/null \
    | grep -v 'run_deepseek_math_queue' | grep -v 'pgrep' | grep -q .
}
POLL=0
while gpu_busy; do
  POLL=$((POLL + 1))
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU busy (poll $POLL); sleep 15m"
  sleep 900
done
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU idle; starting"

export PYTHONUNBUFFERED=1
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "FATAL: OPENROUTER_API_KEY not set; aborting."
  echo "AGENT_DEEPSEEK_QUEUE_FAILED no-api-key"
  exit 1
fi

run_arm() {
  local suffix="$1"
  local tau="$2"
  local lam="$3"
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB strong~ math (deepseek $suffix) tau=$tau lambda=$lam"
  uv run --with mlx-lm --with mlx --with numpy --with datasets \
    python run_stage2_esc.py \
    --model mlx-community/Qwen3.5-4B-4bit \
    --domain math \
    --task-manifest "$DATA/screen_math_4b.json" \
    --N 4 --T 15 \
    --rubric faithful --judge J_strong_tilde \
    --strong-backend openrouter --strong-model deepseek/deepseek-v4-flash-0731 \
    --noise-tau "$tau" --noise-lambda "$lam" --noise-space linear \
    --out "$DATA/A_strongtilde_math_4b_deepseek_${suffix}.jsonl" --resume \
    >"$LOGS/A_strongtilde_math_4b_deepseek_${suffix}.log" 2>&1
  local rc=$?
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB strong~ math (deepseek $suffix) rc=$rc"
  tail -3 "$LOGS/A_strongtilde_math_4b_deepseek_${suffix}.log"
  # See run_strongtilde_oracle_placebo_queue.sh's 2026-08-01 postmortem: rc
  # alone is not trustworthy (background-thread exceptions don't change it).
  # Require the explicit success marker.
  if [ "$rc" -ne 0 ] || ! grep -q '^Done\. Wrote ' "$LOGS/A_strongtilde_math_4b_deepseek_${suffix}.log"; then
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z') AGENT_DEEPSEEK_QUEUE_FAILED arm=$suffix rc=$rc (no 'Done. Wrote' marker)"
    exit 1
  fi
}

run_arm opt 2.9802322387695312e-08 1.0
run_arm cons 0.375 1.0

echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') strongtilde_deepseek_queue COMPLETE ====="
echo "AGENT_DEEPSEEK_QUEUE_DONE"
