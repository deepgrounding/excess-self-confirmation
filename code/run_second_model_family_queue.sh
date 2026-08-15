#!/usr/bin/env bash
# Second model family generalization check (item 2 of the post-submission
# strengthening plan; item 1 is run_second_seed_queue.sh, which this queue
# waits behind on the same single-GPU lock). Tests whether the paper's
# central domain-split finding (ESC positive under J_self in code, absent/
# negative in math) is specific to the Qwen3.5 family or holds for a
# different backbone at a comparable scale.
#
# Model: mlx-community/Llama-3.2-3B-Instruct-4bit -- chosen because it is
# already cached locally (zero download), a different family/architecture
# (Meta Llama vs. Alibaba Qwen), and already validated inside this exact
# harness as the J_peer judge model, which de-risks tokenizer/prompt-format
# surprises. Caveat to disclose in the paper: 3B, not an exact match to the
# primary 4B scale -- closest available, not a deliberate scale-matched pick.
#
# Methodological note: screen_math_4b.json / screen_code_4b.json were built
# by *screening for Qwen3.5-4B's* pass-rate window [0.25, 0.75] (screen_items.py).
# Reusing that pool for Llama-3.2-3B as-is would risk a floor/ceiling
# confound if Llama's pass rate on those specific items sits outside its own
# medium-difficulty band. So this queue re-screens both domains for Llama
# first (same pipeline, same pool sizes, same window, same seed=13) before
# running J_self, producing a Llama-specific frozen item pool
# (screen_{domain}_3b_llama.json) rather than reusing the Qwen pool.
#
# J_self only (not peer/strong~/oracle/placebo) -- this is the minimum
# needed to test the one core claim (H1's ESC(T) sign under J_self) against
# a different model family; not a full cell [A] redo.
set -uo pipefail
cd "$(dirname "$0")"
LOGS=../logs
DATA=../data/stage2
mkdir -p "$LOGS" "$DATA"
LOG="$LOGS/second_model_family_queue.log"
exec >>"$LOG" 2>&1
echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') START second_model_family_queue (Llama-3.2-3B-Instruct-4bit) ====="

MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit"

gpu_busy() {
  pgrep -af 'python.*(run_stage2_esc\.py|screen_items\.py|run_B_n1|run_calib_pool|calibrate_strong_noise)|run_peer_pipeline\.sh|run_A_pipeline\.sh|run_strongtilde_oracle_placebo_queue\.sh|run_deepseek_math_queue\.sh|run_second_seed_queue\.sh' 2>/dev/null \
    | grep -v 'run_second_model_family_queue' | grep -v 'pgrep' | grep -q .
}
POLL=0
while gpu_busy; do
  POLL=$((POLL + 1))
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU busy (poll $POLL); sleep 15m"
  sleep 900
done
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU idle; starting"

export PYTHONUNBUFFERED=1

check_done_marker() {
  local logfile="$1"
  local rc="$2"
  local label="$3"
  local marker="$4"
  if [ "$rc" -ne 0 ] || ! grep -q "$marker" "$logfile"; then
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z') AGENT_SECOND_MODEL_QUEUE_FAILED arm=$label rc=$rc (no '$marker' marker)"
    exit 1
  fi
}

# --- Step 1: screen both domains for Llama-3.2-3B, same params as the
# original Qwen screen (watch_then_screen.sh): pool 600/300, keep 100,
# N=4, seed 13, window [0.25, 0.75]. ---
run_screen() {
  local domain="$1"
  local pool="$2"
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB screen llama3b $domain (pool=$pool)"
  uv run --with mlx-lm --with mlx --with numpy --with datasets --with scikit-learn \
    python screen_items.py \
    --model "$MODEL" \
    --domain "$domain" --pool "$pool" --keep 100 --N 4 --seed 13 \
    --lo 0.25 --hi 0.75 \
    --out "$DATA/screen_${domain}_3b_llama.json" \
    --raw-out "$DATA/screen_${domain}_3b_llama_raw.jsonl" \
    >"$LOGS/screen_${domain}_3b_llama.log" 2>&1
  local rc=$?
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB screen llama3b $domain rc=$rc"
  tail -5 "$LOGS/screen_${domain}_3b_llama.log"
  check_done_marker "$LOGS/screen_${domain}_3b_llama.log" "$rc" "screen_${domain}_llama3b" "^Wrote "
}
run_screen math 600
run_screen code 300

# --- Step 2: J_self, full pool, N=4, T=15, faithful, seed=13, both domains ---
run_self() {
  local domain="$1"
  local manifest="$DATA/screen_${domain}_3b_llama.json"
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB [A] self llama3b $domain"
  uv run --with mlx-lm --with mlx --with numpy --with datasets \
    python run_stage2_esc.py \
    --model "$MODEL" \
    --domain "$domain" \
    --task-manifest "$manifest" \
    --N 4 --T 15 --seed 13 \
    --rubric faithful --judge J_self \
    --out "$DATA/A_self_${domain}_3b_llama.jsonl" --resume \
    >"$LOGS/A_self_${domain}_3b_llama.log" 2>&1
  local rc=$?
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB [A] self llama3b $domain rc=$rc"
  tail -5 "$LOGS/A_self_${domain}_3b_llama.log"
  check_done_marker "$LOGS/A_self_${domain}_3b_llama.log" "$rc" "self_${domain}_llama3b" '^Done\. Wrote '
}
run_self math
run_self code

echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') second_model_family_queue COMPLETE ====="
echo "AGENT_SECOND_MODEL_QUEUE_DONE"
