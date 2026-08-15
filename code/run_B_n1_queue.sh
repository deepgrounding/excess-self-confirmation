#!/usr/bin/env bash
# ESC Stage 2, cell [B], N=1 slice only (cheap signal before N=8/N=12).
set -uo pipefail
cd "$(dirname "$0")"
LOGS=../logs
mkdir -p "$LOGS"
LOG="$LOGS/B_n1_queue.log"
exec >>"$LOG" 2>&1
echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') START B_n1_queue ====="

export PYTHONUNBUFFERED=1

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB B N=1 code (91 items)"
uv run --with mlx-lm --with mlx --with numpy --with datasets \
  python run_stage2_esc.py \
  --model mlx-community/Qwen3.5-4B-4bit \
  --domain code \
  --task-manifest ../data/stage2/screen_code_4b.json \
  --N 1 --T 15 \
  --rubric faithful --judge J_self \
  --out ../data/stage2/B_n1_code_4b_self.jsonl --resume \
  >"$LOGS/B_n1_code_4b_self.log" 2>&1
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB B N=1 code rc=$?"
tail -5 "$LOGS/B_n1_code_4b_self.log"

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB B N=1 math (84 items)"
uv run --with mlx-lm --with mlx --with numpy --with datasets \
  python run_stage2_esc.py \
  --model mlx-community/Qwen3.5-4B-4bit \
  --domain math \
  --task-manifest ../data/stage2/screen_math_4b.json \
  --N 1 --T 15 \
  --rubric faithful --judge J_self \
  --out ../data/stage2/B_n1_math_4b_self.jsonl --resume \
  >"$LOGS/B_n1_math_4b_self.log" 2>&1
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB B N=1 math rc=$?"
tail -5 "$LOGS/B_n1_math_4b_self.log"

echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') B_n1_queue COMPLETE ====="
echo "AGENT_B_N1_DONE"
