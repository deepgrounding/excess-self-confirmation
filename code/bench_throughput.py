"""Throughput re-benchmark for ESC Stage-2 (§8 commitment).

Runs 20 short gen+judge cycles on GSM8K-style prompts and prints s/call.
Does not write Stage-2 scientific data.

  PYTHONUNBUFFERED=1 uv run --with mlx-lm --with mlx --with numpy --with datasets \\
    python bench_throughput.py --model mlx-community/Qwen3.5-4B-4bit --n 20
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage2_esc import gen, chat_prompt  # noqa: E402
from prompts_frozen import GEN_MATH, JUDGE_FAITHFUL  # noqa: E402


PROBES = [
    "A store has 23 apples and sells 8. How many remain? End with '#### <number>'.",
    "What is 17 + 25? End with '#### <number>'.",
    "A train travels 60 miles then 45. Total? End with '#### <number>'.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-4bit")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    model, tokenizer = load(args.model)
    times = []
    for i in range(args.n):
        q = PROBES[i % len(PROBES)]
        t0 = time.time()
        text, _ = gen(model, tokenizer, [{"role": "user", "content": q}], 128, temp=0.7)
        jmsg = [{"role": "user", "content": JUDGE_FAITHFUL.format(task=q, solution=text)}]
        gen(model, tokenizer, jmsg, 80, temp=0.2)
        dt = time.time() - t0
        times.append(dt)
        print(f"[{i+1}/{args.n}] {dt:.2f}s", flush=True)
    arr = np.array(times)
    print(
        f"SUMMARY model={args.model} n={args.n} "
        f"mean_s_per_gen+judge={arr.mean():.2f} median={np.median(arr):.2f} "
        f"p90={np.percentile(arr,90):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
