#!/usr/bin/env bash
# Post-withdrawal scope (2026-07-26): finish cell [A]'s remaining arms needed
# for the merged results paper -- J_strong~ (gemma backend, per the 4.2
# amendment) plus full-scale J_oracle / J_placebo on both frozen pools.
# [C]/[D]/[E]/second-seed stay future work.
#
# Gemma calibrations are already done and live in the canonical files
# (noise_calibration_{code,math}.json = gemma primary; *_haiku.json =
# cross-check). This queue only runs the closed-loop arms: strong~ with taus
# read from those JSONs at runtime (code single arm, math dual-bound), then
# oracle/placebo full runs. Waits for the running J_peer pipeline to finish.
set -uo pipefail
cd "$(dirname "$0")"
LOGS=../logs
DATA=../data/stage2
mkdir -p "$LOGS" "$DATA"
LOG="$LOGS/strongtilde_queue.log"
exec >>"$LOG" 2>&1
echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') START strongtilde_oracle_placebo_queue (v2: calibrations pre-done) ====="

gpu_busy() {
  pgrep -af 'python.*(run_stage2_esc\.py|run_B_n1|run_calib_pool|calibrate_strong_noise)|run_peer_pipeline\.sh|run_A_pipeline\.sh' 2>/dev/null \
    | grep -v 'run_strongtilde_oracle_placebo_queue' | grep -v 'pgrep' | grep -q .
}
POLL=0
while gpu_busy; do
  POLL=$((POLL + 1))
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU busy (poll $POLL); sleep 15m"
  sleep 900
done
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU idle; starting"

export PYTHONUNBUFFERED=1

# Ollama must be up for gemma judging.
if ! curl -s -m 5 http://localhost:11434/api/tags >/dev/null; then
  echo "FATAL: Ollama server not reachable at localhost:11434; aborting."
  echo "AGENT_STRONGTILDE_QUEUE_FAILED ollama-down"
  exit 1
fi

# Sanity: canonical calibration files must be gemma-based with expected shape.
python3 - <<'PY' || { echo "AGENT_STRONGTILDE_QUEUE_FAILED bad-calibration-files"; exit 1; }
import json, sys
code = json.loads(open("../data/stage2/noise_calibration_code.json").read())
math = json.loads(open("../data/stage2/noise_calibration_math.json").read())
assert code["strong_model"] == "gemma4:12b" and code["gate_pass"], code
assert math["strong_model"] == "gemma4:12b" and math.get("dual_bound"), math
print("calibration files OK: code single (tau=%.4g), math dual-bound" % code["tau"])
PY

# Helper: extract "tau lambda" for an arm from a calibration JSON.
get_noise() {
  python3 - "$1" "$2" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1]).read())
arm = sys.argv[2]
if arm == "single":
    if not d.get("gate_pass"):
        sys.exit(f"gate_pass false in {sys.argv[1]}")
    print(d["tau"], d["lambda"])
else:
    db = d["dual_bound"]
    print(db[arm]["tau"], db[arm].get("lambda", 1.0))
PY
}

run_arm() {
  local domain="$1"
  local manifest="$2"
  local tau="$3"
  local lam="$4"
  local suffix="$5"
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB strong~ $domain ($suffix) tau=$tau lambda=$lam"
  uv run --with mlx-lm --with mlx --with numpy --with datasets \
    python run_stage2_esc.py \
    --model mlx-community/Qwen3.5-4B-4bit \
    --domain "$domain" \
    --task-manifest "$manifest" \
    --N 4 --T 15 \
    --rubric faithful --judge J_strong_tilde \
    --strong-backend ollama --strong-model gemma4:12b \
    --noise-tau "$tau" --noise-lambda "$lam" --noise-space linear \
    --out "$DATA/A_strongtilde_${domain}_4b_${suffix}.jsonl" --resume \
    >"$LOGS/A_strongtilde_${domain}_4b_${suffix}.log" 2>&1
  local rc=$?
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB strong~ $domain ($suffix) rc=$rc"
  tail -3 "$LOGS/A_strongtilde_${domain}_4b_${suffix}.log"
  # rc alone is not trustworthy: an uncaught exception in an MLX background
  # thread (e.g. the 2026-07-30 Metal "Command buffer execution failed"
  # crash) prints a traceback but does not change the process exit code, so
  # a genuine mid-run crash was silently logged as "rc=0" and the queue
  # cascaded straight to a false COMPLETE. Require the explicit success
  # marker instead of trusting $?.
  if [ "$rc" -ne 0 ] || ! grep -q '^Done\. Wrote ' "$LOGS/A_strongtilde_${domain}_4b_${suffix}.log"; then
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z') AGENT_STRONGTILDE_QUEUE_FAILED arm=strong~_${domain}_${suffix} rc=$rc (no 'Done. Wrote' marker)"
    exit 1
  fi
}

# --- strong~ code: single arm (gemma calibration passed its gate) ---
read -r TAU LAM < <(get_noise "$DATA/noise_calibration_code.json" single)
run_arm code "$DATA/screen_code_4b.json" "$TAU" "$LAM" main

# --- strong~ math: dual-bound arms ---
read -r TAU LAM < <(get_noise "$DATA/noise_calibration_math.json" optimistic)
run_arm math "$DATA/screen_math_4b.json" "$TAU" "$LAM" opt
read -r TAU LAM < <(get_noise "$DATA/noise_calibration_math.json" conservative)
run_arm math "$DATA/screen_math_4b.json" "$TAU" "$LAM" cons

# --- oracle + placebo, full scale, both domains ---
run_cheap() {
  local judge="$1"
  local domain="$2"
  local manifest="$3"
  local name="${judge#J_}"
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB $judge full $domain"
  uv run --with mlx-lm --with mlx --with numpy --with datasets \
    python run_stage2_esc.py \
    --model mlx-community/Qwen3.5-4B-4bit \
    --domain "$domain" \
    --task-manifest "$manifest" \
    --N 4 --T 15 \
    --rubric faithful --judge "$judge" \
    --out "$DATA/A_${name}_${domain}_4b.jsonl" --resume \
    >"$LOGS/A_${name}_${domain}_4b.log" 2>&1
  local rc=$?
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB $judge full $domain rc=$rc"
  if [ "$rc" -ne 0 ] || ! grep -q '^Done\. Wrote ' "$LOGS/A_${name}_${domain}_4b.log"; then
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z') AGENT_STRONGTILDE_QUEUE_FAILED arm=${name}_${domain} rc=$rc (no 'Done. Wrote' marker)"
    exit 1
  fi
}

run_cheap J_oracle math "$DATA/screen_math_4b.json"
run_cheap J_oracle code "$DATA/screen_code_4b.json"
run_cheap J_placebo math "$DATA/screen_math_4b.json"
run_cheap J_placebo code "$DATA/screen_code_4b.json"

echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') strongtilde_oracle_placebo_queue COMPLETE ====="
echo "AGENT_STRONGTILDE_QUEUE_DONE"
