#!/usr/bin/env bash
# Second-seed replication (seed=42) on the pre-registered "critical cells"
# subset: J_self, J_strong~, N=1 (ANALYSIS_PLAN.md:25 -- H3 main [C] is
# excluded since [C] was never run in this paper's reduced scope). These
# three conditions feed every headline statistic in main.md Table 7.7
# (ESC(T), AUC_ESC, SCG_{N=1}(T), ESC_sp) except ESC_adj (needs placebo,
# not on the pre-registered second-seed list -- intentionally out of scope
# here). Generation seed 42 is the project's designated second seed
# (Appendix D: "generation seeds {13, 42}").
#
# strong~ reuses the EXISTING (seed-13-calibrated) tau/lambda from
# noise_calibration_{code,math}.json rather than re-running the noise-match
# search at seed 42 -- tau is a property of the noise-injection mechanism,
# not the underlying trajectory seed, and re-calibrating per seed would
# both cost more compute and raise its own methodological question. Analysis
# should re-check the noise-match diagnostic (|ΔAUC| <= 0.02) held at seed 42
# and disclose if it drifted.
set -uo pipefail
cd "$(dirname "$0")"
LOGS=../logs
DATA=../data/stage2
mkdir -p "$LOGS" "$DATA"
LOG="$LOGS/second_seed_queue.log"
exec >>"$LOG" 2>&1
echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') START second_seed_queue (seed=42) ====="

gpu_busy() {
  pgrep -af 'python.*(run_stage2_esc\.py|run_B_n1|run_calib_pool|calibrate_strong_noise)|run_peer_pipeline\.sh|run_A_pipeline\.sh|run_strongtilde_oracle_placebo_queue\.sh|run_deepseek_math_queue\.sh' 2>/dev/null \
    | grep -v 'run_second_seed_queue' | grep -v 'pgrep' | grep -q .
}
POLL=0
while gpu_busy; do
  POLL=$((POLL + 1))
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU busy (poll $POLL); sleep 15m"
  sleep 900
done
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU idle; starting"

export PYTHONUNBUFFERED=1

if ! curl -s -m 5 http://localhost:11434/api/tags >/dev/null; then
  echo "FATAL: Ollama server not reachable at localhost:11434; aborting."
  echo "AGENT_SECOND_SEED_QUEUE_FAILED ollama-down"
  exit 1
fi

# Sanity: canonical calibration files must still be the gemma-based ones
# this queue depends on for strong~'s tau/lambda.
python3 - <<'PY' || { echo "AGENT_SECOND_SEED_QUEUE_FAILED bad-calibration-files"; exit 1; }
import json
code = json.loads(open("../data/stage2/noise_calibration_code.json").read())
math = json.loads(open("../data/stage2/noise_calibration_math.json").read())
assert code["strong_model"] == "gemma4:12b" and code["gate_pass"], code
assert math["strong_model"] == "gemma4:12b" and math.get("dual_bound"), math
print("calibration files OK: code single (tau=%.4g), math dual-bound" % code["tau"])
PY

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

check_done() {
  local logfile="$1"
  local rc="$2"
  local label="$3"
  if [ "$rc" -ne 0 ] || ! grep -q '^Done\. Wrote ' "$logfile"; then
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z') AGENT_SECOND_SEED_QUEUE_FAILED arm=$label rc=$rc (no 'Done. Wrote' marker)"
    exit 1
  fi
}

# --- J_self, seed 42, both domains (full pool, N=4, T=15) ---
run_self() {
  local domain="$1"
  local manifest="$2"
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB [A] self seed42 $domain"
  uv run --with mlx-lm --with mlx --with numpy --with datasets \
    python run_stage2_esc.py \
    --model mlx-community/Qwen3.5-4B-4bit \
    --domain "$domain" \
    --task-manifest "$manifest" \
    --N 4 --T 15 --seed 42 \
    --rubric faithful --judge J_self \
    --out "$DATA/A_self_${domain}_4b_seed42.jsonl" --resume \
    >"$LOGS/A_self_${domain}_4b_seed42.log" 2>&1
  local rc=$?
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB [A] self seed42 $domain rc=$rc"
  tail -3 "$LOGS/A_self_${domain}_4b_seed42.log"
  check_done "$LOGS/A_self_${domain}_4b_seed42.log" "$rc" "self_${domain}_seed42"
}
run_self math "$DATA/screen_math_4b.json"
run_self code "$DATA/screen_code_4b.json"

# --- J_strong~, seed 42: code single arm + math dual-bound ---
run_strongtilde() {
  local domain="$1"
  local manifest="$2"
  local tau="$3"
  local lam="$4"
  local suffix="$5"
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB strong~ seed42 $domain ($suffix) tau=$tau lambda=$lam"
  uv run --with mlx-lm --with mlx --with numpy --with datasets \
    python run_stage2_esc.py \
    --model mlx-community/Qwen3.5-4B-4bit \
    --domain "$domain" \
    --task-manifest "$manifest" \
    --N 4 --T 15 --seed 42 \
    --rubric faithful --judge J_strong_tilde \
    --strong-backend ollama --strong-model gemma4:12b \
    --noise-tau "$tau" --noise-lambda "$lam" --noise-space linear \
    --out "$DATA/A_strongtilde_${domain}_4b_seed42_${suffix}.jsonl" --resume \
    >"$LOGS/A_strongtilde_${domain}_4b_seed42_${suffix}.log" 2>&1
  local rc=$?
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB strong~ seed42 $domain ($suffix) rc=$rc"
  tail -3 "$LOGS/A_strongtilde_${domain}_4b_seed42_${suffix}.log"
  check_done "$LOGS/A_strongtilde_${domain}_4b_seed42_${suffix}.log" "$rc" "strong~_${domain}_${suffix}_seed42"
}
read -r TAU LAM < <(get_noise "$DATA/noise_calibration_code.json" single)
run_strongtilde code "$DATA/screen_code_4b.json" "$TAU" "$LAM" main
read -r TAU LAM < <(get_noise "$DATA/noise_calibration_math.json" optimistic)
run_strongtilde math "$DATA/screen_math_4b.json" "$TAU" "$LAM" opt
read -r TAU LAM < <(get_noise "$DATA/noise_calibration_math.json" conservative)
run_strongtilde math "$DATA/screen_math_4b.json" "$TAU" "$LAM" cons

# --- Cell [B] N=1, J_self, seed 42, both domains ---
run_n1() {
  local domain="$1"
  local manifest="$2"
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB [B] N=1 self seed42 $domain"
  uv run --with mlx-lm --with mlx --with numpy --with datasets \
    python run_stage2_esc.py \
    --model mlx-community/Qwen3.5-4B-4bit \
    --domain "$domain" \
    --task-manifest "$manifest" \
    --N 1 --T 15 --seed 42 \
    --rubric faithful --judge J_self \
    --out "$DATA/B_n1_${domain}_4b_self_seed42.jsonl" --resume \
    >"$LOGS/B_n1_${domain}_4b_self_seed42.log" 2>&1
  local rc=$?
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB [B] N=1 self seed42 $domain rc=$rc"
  tail -3 "$LOGS/B_n1_${domain}_4b_self_seed42.log"
  check_done "$LOGS/B_n1_${domain}_4b_self_seed42.log" "$rc" "n1_${domain}_seed42"
}
run_n1 math "$DATA/screen_math_4b.json"
run_n1 code "$DATA/screen_code_4b.json"

echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') second_seed_queue COMPLETE ====="
echo "AGENT_SECOND_SEED_QUEUE_DONE"
