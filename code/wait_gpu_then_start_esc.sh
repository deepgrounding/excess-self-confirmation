#!/usr/bin/env bash
# Wait for calibration_floor MLX to finish, then start ESC Stage-2 next steps.
# ESC_SKIP_SLEEP=1 → skip the initial 4h wait.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE="$ROOT/code"
DATA="$ROOT/data/stage2"
LOGS="$ROOT/logs"
mkdir -p "$DATA" "$LOGS"
LOG="$LOGS/wait_gpu_then_esc.log"
MARKER="$LOGS/esc_auto_started.flag"

exec >>"$LOG" 2>&1
echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') ARMED wait_gpu_then_start_esc ====="

gpu_busy() {
  # Match real workers only (exclude this waiter and pgrep itself).
  local hits
  hits=$(pgrep -af 'python.*(run_stage2\.py|run_stage2_esc\.py|screen_items\.py|bench_throughput\.py|pilot_trajectory\.py)' 2>/dev/null \
    | grep -v 'wait_gpu_then_start_esc' \
    | grep -v 'pgrep' \
    || true)
  [ -n "$hits" ]
}

disk_ok() {
  local avail_kb
  avail_kb=$(df -k /System/Volumes/Data | awk 'NR==2{print $4}')
  [ "${avail_kb:-0}" -ge 5242880 ]
}

run_job() {
  local name="$1"; shift
  local logfile="$1"; shift
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') START $name"
  set +e
  # Avoid pipefail/tee SIGPIPE killing the waiter; capture exit code.
  "$@" >"$logfile" 2>&1
  local rc=$?
  set -e
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') DONE $name rc=$rc (log=$(basename "$logfile"))"
  # Mirror last lines into master log for quick diagnosis
  tail -20 "$logfile" || true
  return "$rc"
}

if [ -f "$MARKER" ] && [ "${ESC_FORCE:-0}" != "1" ]; then
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') marker exists ($MARKER); refusing double-start (ESC_FORCE=1 to override)."
  exit 0
fi

if [ "${ESC_SKIP_SLEEP:-0}" = "1" ]; then
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') ESC_SKIP_SLEEP=1; skip 4h wait"
else
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') sleeping 4h before first GPU check..."
  sleep 14400
fi

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') first GPU check"
POLL=0
MAX_POLLS=48
while gpu_busy; do
  POLL=$((POLL + 1))
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU busy (poll $POLL/$MAX_POLLS); sleep 15m"
  if [ "$POLL" -ge "$MAX_POLLS" ]; then
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z') gave up waiting for GPU"
    exit 2
  fi
  sleep 900
done

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') GPU idle"
if ! disk_ok; then
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') disk <5GB free; abort ESC auto-start"
  df -h /System/Volumes/Data
  exit 3
fi

touch "$MARKER"
cd "$CODE"
export PYTHONUNBUFFERED=1
set +e   # never abort the whole sequence on one job failure

# --- Job 1: 4B instrument-gate smoke (higher priority than 9B bench) ---
SMOKE_MANIFEST="$DATA/gate_smoke_math_4b_manifest.json"
python3 - <<PY
import json
from pathlib import Path
src = Path("$DATA/screen_math_4b.json")
dst = Path("$SMOKE_MANIFEST")
d = json.loads(src.read_text())
d["tasks"] = d["tasks"][:16]
d["n_kept"] = len(d["tasks"])
d["note"] = "auto gate-smoke subset (16 items) from screen_math_4b.json"
dst.write_text(json.dumps(d, indent=2, ensure_ascii=False))
print("wrote", dst, "n=", len(d["tasks"]))
PY

run_job "ESC gate-smoke math 4B J_self" "$LOGS/gate_smoke_math_4b_self.log" \
  uv run --with mlx-lm --with mlx --with numpy --with datasets \
  python run_stage2_esc.py \
  --model mlx-community/Qwen3.5-4B-4bit \
  --domain math \
  --task-manifest "$SMOKE_MANIFEST" \
  --N 4 --T 2 \
  --rubric faithful --judge J_self \
  --out "$DATA/gate_smoke_math_4b_self.jsonl" --resume

run_job "ESC gate-smoke math 4B J_placebo" "$LOGS/gate_smoke_math_4b_placebo.log" \
  uv run --with mlx-lm --with mlx --with numpy --with datasets \
  python run_stage2_esc.py \
  --model mlx-community/Qwen3.5-4B-4bit \
  --domain math \
  --task-manifest "$SMOKE_MANIFEST" \
  --N 4 --T 2 \
  --rubric faithful --judge J_placebo \
  --out "$DATA/gate_smoke_math_4b_placebo.jsonl" --resume

# --- Job 2: 9B throughput re-bench (§8); non-fatal if OOM ---
run_job "ESC 9B throughput" "$LOGS/throughput_9b.log" \
  uv run --with mlx-lm --with mlx --with numpy --with datasets \
  python bench_throughput.py \
  --model mlx-community/Qwen3.5-9B-4bit --n 20

python3 - <<'PY'
import json, re
from pathlib import Path
from datetime import date
logp = Path("../logs/throughput_9b.log")
log = logp.read_text() if logp.exists() else ""
m = re.search(r"mean_s_per_gen\+judge=([0-9.]+).*median=([0-9.]+).*p90=([0-9.]+)", log)
out = Path("../data/stage2/throughput_9b.json")
if m:
    out.write_text(json.dumps({
        "model": "mlx-community/Qwen3.5-9B-4bit",
        "n": 20,
        "mean_s_per_gen_judge": float(m.group(1)),
        "median": float(m.group(2)),
        "p90": float(m.group(3)),
        "log": "logs/throughput_9b.log",
        "date": date.today().isoformat(),
    }, indent=2) + "\n")
    print("wrote", out)
else:
    print("no SUMMARY parsed; skip throughput_9b.json")
PY

echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') ESC auto-start sequence complete ====="
echo "AGENT_ESC_AUTO_DONE {\"log\":\"$LOG\"}"
