#!/usr/bin/env bash
# Cell [A] pipeline, staged per main.md 6.2:
#   1. Instrument admission gate: J_oracle + J_placebo pilots (20 items/domain,
#      full T=15 depth) -> check_instrument_gate.py (|SEL|,|ESC| ~ 0).
#   2. If (and only if) the gate passes: [A] J_self full run, both domains,
#      full screened pools, N=4, T=15 (the ~49h core-signal piece).
# Does NOT auto-continue to peer/strong/strong~/oracle-full/placebo-full --
# those are queued separately once self's signal has been reviewed.
set -uo pipefail
cd "$(dirname "$0")"
LOGS=../logs
DATA=../data/stage2
mkdir -p "$LOGS" "$DATA"
LOG="$LOGS/A_pipeline.log"
exec >>"$LOG" 2>&1
echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') START A_pipeline ====="

gpu_busy() {
  pgrep -af 'python.*(run_stage2_esc\.py|run_B_n1|run_calib_pool)' 2>/dev/null \
    | grep -v 'run_A_pipeline' | grep -v 'pgrep' | grep -q .
}
POLL=0
while gpu_busy; do
  POLL=$((POLL + 1))
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU busy (poll $POLL); sleep 15m"
  sleep 900
done
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU idle; starting instrument-gate pilots"

export PYTHONUNBUFFERED=1
set -a; source ../../.env; set +a

run_pilot() {
  local judge="$1"
  local domain="$2"
  local out="$DATA/A_pilot_${judge#J_}_${domain}.jsonl"
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB pilot $judge $domain"
  uv run --with mlx-lm --with mlx --with numpy --with datasets \
    python run_stage2_esc.py \
    --model mlx-community/Qwen3.5-4B-4bit \
    --domain "$domain" \
    --task-manifest "$DATA/A_pilot_${domain}_manifest.json" \
    --N 4 --T 15 \
    --rubric faithful --judge "$judge" \
    --out "$out" --resume \
    >"$LOGS/A_pilot_${judge#J_}_${domain}.log" 2>&1
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB pilot $judge $domain rc=$?"
}

run_pilot J_oracle math
run_pilot J_oracle code
run_pilot J_placebo math
run_pilot J_placebo code

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') running instrument gate checks"
GATE_OK=1
for cond_domain in "oracle math" "oracle code" "placebo math" "placebo code"; do
  set -- $cond_domain
  cond=$1; domain=$2
  jf="$DATA/A_pilot_${cond}_${domain}.jsonl"
  uv run --with numpy \
    python check_instrument_gate.py --jsonl "$jf" --condition "$cond" \
    >"$LOGS/gate_${cond}_${domain}.log" 2>&1
  tail -10 "$LOGS/gate_${cond}_${domain}.log"
  if ! grep -q '"gate_pass": true' "$LOGS/gate_${cond}_${domain}.log"; then
    GATE_OK=0
  fi
done

if [ "$GATE_OK" -ne 1 ]; then
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') INSTRUMENT GATE FAILED -- stopping before [A] self. Review gate_*.log."
  echo "AGENT_A_PIPELINE_GATE_FAILED"
  exit 1
fi
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') instrument gate PASSED; proceeding to [A] J_self full run"

run_A_self() {
  local domain="$1"
  local manifest="$2"
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB [A] self $domain (full pool, N=4, T=15)"
  uv run --with mlx-lm --with mlx --with numpy --with datasets \
    python run_stage2_esc.py \
    --model mlx-community/Qwen3.5-4B-4bit \
    --domain "$domain" \
    --task-manifest "$manifest" \
    --N 4 --T 15 \
    --rubric faithful --judge J_self \
    --out "$DATA/A_self_${domain}_4b.jsonl" --resume \
    >"$LOGS/A_self_${domain}_4b.log" 2>&1
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB [A] self $domain rc=$?"
  tail -5 "$LOGS/A_self_${domain}_4b.log"
}

run_A_self math "$DATA/screen_math_4b.json"
run_A_self code "$DATA/screen_code_4b.json"

echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') A_pipeline COMPLETE ====="
echo "AGENT_A_PIPELINE_DONE"
