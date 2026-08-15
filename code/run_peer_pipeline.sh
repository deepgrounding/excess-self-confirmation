#!/usr/bin/env bash
# Cell [A] J_peer pipeline, next in queue after J_self (main.md 6.7 order:
# self -> peer -> strong). Fully local (Llama-3.2-3B-Instruct-4bit judges
# Qwen3.5-4B), no API. Full pre-registered scale, both domains, N=4, T=15,
# on the same frozen §14.3 pools J_self already used.
set -uo pipefail
cd "$(dirname "$0")"
LOGS=../logs
DATA=../data/stage2
mkdir -p "$LOGS" "$DATA"
LOG="$LOGS/peer_pipeline.log"
exec >>"$LOG" 2>&1
echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') START peer_pipeline ====="

gpu_busy() {
  pgrep -af 'python.*(run_stage2_esc\.py|run_B_n1|run_calib_pool)' 2>/dev/null \
    | grep -v 'run_peer_pipeline' | grep -v 'pgrep' | grep -q .
}
POLL=0
while gpu_busy; do
  POLL=$((POLL + 1))
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU busy (poll $POLL); sleep 15m"
  sleep 900
done
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU idle; starting J_peer full runs"

export PYTHONUNBUFFERED=1

run_peer() {
  local domain="$1"
  local manifest="$2"
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB [A] peer $domain (full pool, N=4, T=15)"
  uv run --with mlx-lm --with mlx --with numpy --with datasets \
    python run_stage2_esc.py \
    --model mlx-community/Qwen3.5-4B-4bit \
    --domain "$domain" \
    --task-manifest "$manifest" \
    --N 4 --T 15 \
    --rubric faithful --judge J_peer \
    --peer-model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --out "$DATA/A_peer_${domain}_4b.jsonl" --resume \
    >"$LOGS/A_peer_${domain}_4b.log" 2>&1
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB [A] peer $domain rc=$?"
  tail -5 "$LOGS/A_peer_${domain}_4b.log"
}

run_peer math "$DATA/screen_math_4b.json"
run_peer code "$DATA/screen_code_4b.json"

echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') peer_pipeline COMPLETE ====="
echo "AGENT_PEER_PIPELINE_DONE"
