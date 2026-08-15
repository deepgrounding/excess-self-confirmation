#!/usr/bin/env bash
# ESC next MLX jobs (no long sleep). Continues past individual failures.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE="$ROOT/code"
DATA="$ROOT/data/stage2"
LOGS="$ROOT/logs"
mkdir -p "$DATA" "$LOGS"
LOG="$LOGS/run_esc_next.log"
MARKER="$LOGS/esc_auto_started.flag"

exec >>"$LOG" 2>&1
echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') START run_esc_next ====="

if pgrep -af 'python.*(run_stage2\.py|run_stage2_esc\.py|bench_throughput\.py)' \
  | grep -v 'run_esc_next' | grep -v pgrep | grep -q .; then
  echo "Another MLX job is running; abort."
  pgrep -af 'python.*(run_stage2|bench_throughput)' || true
  exit 1
fi

touch "$MARKER"
cd "$CODE"
export PYTHONUNBUFFERED=1

SMOKE_MANIFEST="$DATA/gate_smoke_math_4b_manifest.json"
python3 - <<PY
import json
from pathlib import Path
src = Path("$DATA/screen_math_4b.json")
dst = Path("$SMOKE_MANIFEST")
d = json.loads(src.read_text())
d["tasks"] = d["tasks"][:16]
d["n_kept"] = len(d["tasks"])
d["note"] = "gate-smoke subset (16 items) from screen_math_4b.json"
dst.write_text(json.dumps(d, indent=2, ensure_ascii=False))
print("manifest n=", len(d["tasks"]))
PY

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB gate-smoke J_self"
uv run --with mlx-lm --with mlx --with numpy --with datasets \
  python run_stage2_esc.py \
  --model mlx-community/Qwen3.5-4B-4bit \
  --domain math \
  --task-manifest "$SMOKE_MANIFEST" \
  --N 4 --T 2 \
  --rubric faithful --judge J_self \
  --out "$DATA/gate_smoke_math_4b_self.jsonl" --resume \
  >"$LOGS/gate_smoke_math_4b_self.log" 2>&1
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB gate-smoke J_self rc=$?"

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB gate-smoke J_placebo"
uv run --with mlx-lm --with mlx --with numpy --with datasets \
  python run_stage2_esc.py \
  --model mlx-community/Qwen3.5-4B-4bit \
  --domain math \
  --task-manifest "$SMOKE_MANIFEST" \
  --N 4 --T 2 \
  --rubric faithful --judge J_placebo \
  --out "$DATA/gate_smoke_math_4b_placebo.jsonl" --resume \
  >"$LOGS/gate_smoke_math_4b_placebo.log" 2>&1
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB gate-smoke J_placebo rc=$?"

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB 9B throughput"
uv run --with mlx-lm --with mlx --with numpy --with datasets \
  python bench_throughput.py \
  --model mlx-community/Qwen3.5-9B-4bit --n 20 \
  >"$LOGS/throughput_9b.log" 2>&1
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') JOB 9B throughput rc=$?"

python3 - <<'PY'
import json, re
from pathlib import Path
from datetime import date
log = Path("../logs/throughput_9b.log").read_text() if Path("../logs/throughput_9b.log").exists() else ""
m = re.search(r"mean_s_per_gen\+judge=([0-9.]+).*median=([0-9.]+).*p90=([0-9.]+)", log)
out = Path("../data/stage2/throughput_9b.json")
if m:
    out.write_text(json.dumps({
        "model": "mlx-community/Qwen3.5-9B-4bit", "n": 20,
        "mean_s_per_gen_judge": float(m.group(1)),
        "median": float(m.group(2)), "p90": float(m.group(3)),
        "log": "logs/throughput_9b.log", "date": date.today().isoformat(),
    }, indent=2) + "\n")
    print("wrote", out)
else:
    print("no 9B SUMMARY")
PY

echo "===== $(date '+%Y-%m-%dT%H:%M:%S%z') run_esc_next COMPLETE ====="
