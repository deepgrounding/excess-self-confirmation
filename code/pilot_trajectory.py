"""ESC harness pilot: real MLX self-judge closed loop.

Plumbing check only (not a hypothesis test). Exercises:
  GEN (N candidates) -> J_self score -> argmax select -> REVISE
  -> full-pool oracle G -> offline SEL/ESC via esc_core.decompose

Mirror of error_structure_manuscript/code/pilot_trajectory.py, adapted to
the ESC candidate-pool schema (Appendix D of draft/main.md).

Examples:
  python pilot_trajectory.py
  python pilot_trajectory.py --model mlx-community/Qwen3.5-4B-4bit --tag 4b
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from mlx_lm import load
from mlx_lm.generate import stream_generate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esc_core import decompose, round_quantities  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen3.5-0.8B-4bit"
N = 4
T = 2  # rounds 0..T
SEED = 13
MAX_TOKENS_GEN = 220
MAX_TOKENS_JUDGE = 80

TASKS = [
    {"id": "gsm_pilot_1", "prompt": "A store has 23 apples. They sell 8 and then receive a shipment of 15 more. How many apples does the store have now? End your answer with '#### <number>'.", "answer": "30"},
    {"id": "gsm_pilot_2", "prompt": "Maria has 4 boxes with 6 pencils each. She gives away 5 pencils. How many pencils does she have left? End your answer with '#### <number>'.", "answer": "19"},
    {"id": "gsm_pilot_3", "prompt": "A train travels 60 miles in the first hour and 45 miles in the second hour. What is the total distance traveled? End your answer with '#### <number>'.", "answer": "105"},
    {"id": "gsm_pilot_4", "prompt": "A tank holds 480 liters. A pump fills it at 15 liters per minute, but it leaks 3 liters per minute. How many minutes does it take to fill the tank? End your answer with '#### <number>'.", "answer": "40"},
    {"id": "gsm_pilot_5", "prompt": "A rectangular garden is 3 times as long as it is wide. If the perimeter is 96 meters, what is the width in meters? End your answer with '#### <number>'.", "answer": "12"},
    {"id": "gsm_pilot_6", "prompt": "Twelve friends split a restaurant bill of $312 evenly, then one friend covers an extra $18 tip alone. How much does each of the other eleven friends pay? End your answer with '#### <number>'.", "answer": "26"},
    {"id": "gsm_pilot_7", "prompt": "A number is tripled, then 7 is subtracted, then the result is doubled to give 58. What was the original number? End your answer with '#### <number>'.", "answer": "12"},
]

JUDGE_F = (
    "Evaluate whether the following solution correctly solves the task. "
    "Consider correctness only. "
    "Output JSON only: {{\"score\": <probability in [0,1]>, \"feedback\": \"<one sentence>\"}}\n\n"
    "Task:\n{task}\n\nSolution:\n{solution}"
)
REVISE = (
    "Please produce an improved version of the previous solution, "
    "taking the feedback into account. End with '#### <number>'.\n\n"
    "Previous solution:\n{prev}\n\nFeedback:\n{feedback}"
)


def extract_answer(text: str) -> str | None:
    m = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", text)
    return m[-1] if m else None


def extract_judge(text: str) -> tuple[float, str]:
    """Parse score in [0,1] and feedback; fall back gracefully on messy output."""
    score, feedback = 0.5, "Please improve overall correctness."
    m = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    blob = m.group(0) if m else text
    sm = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', blob)
    if sm:
        score = float(np.clip(float(sm.group(1)), 0.0, 1.0))
    fm = re.search(r'"feedback"\s*:\s*"([^"]*)"', blob)
    if fm:
        feedback = fm.group(1).strip() or feedback
    return score, feedback


def chat_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def gen(model, tokenizer, messages: list[dict], max_tokens: int, temp: float = 0.7):
    prompt = chat_prompt(tokenizer, messages)
    text, lp_sum, n_tok = "", 0.0, 0
    kwargs = {"max_tokens": max_tokens}
    # temperature via sampler if available; keep simple for plumbing
    try:
        from mlx_lm.sample_utils import make_sampler

        kwargs["sampler"] = make_sampler(temp=temp)
    except Exception:
        pass
    for resp in stream_generate(model, tokenizer, prompt=prompt, **kwargs):
        text += resp.text
        if resp.logprobs is not None:
            lp_sum += float(resp.logprobs[resp.token])
            n_tok += 1
    return text, lp_sum / max(n_tok, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--tag",
        default="",
        help="Output filename suffix, e.g. '4b' -> pilot_candidate_pool_4b.jsonl. "
        "Empty keeps legacy names for the 0.8B pilot.",
    )
    args = ap.parse_args()
    model_id = args.model
    tag = args.tag.strip("_")
    suffix = f"_{tag}" if tag else ""

    rng = np.random.default_rng(SEED)
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    pool_path = out_dir / f"pilot_candidate_pool{suffix}.jsonl"
    summary_path = out_dir / f"pilot_esc_summary{suffix}.json"

    print(f"Loading {model_id} (local HF cache) ...", flush=True)
    model, tokenizer = load(model_id)

    records = []
    # Per-round aggregates across the 7 tasks (one trajectory unit for the pilot)
    round_dicts = []

    for t in range(T + 1):
        j_mat = np.zeros((len(TASKS), N))
        G_mat = np.zeros((len(TASKS), N))
        sel_idx = np.zeros(len(TASKS), dtype=int)
        # carry selected text/feedback for next-round revise
        if t == 0:
            parents = {task["id"]: None for task in TASKS}
            parent_fb = {task["id"]: None for task in TASKS}

        for i, task in enumerate(TASKS):
            cands = []
            for k in range(N):
                if t == 0:
                    messages = [{"role": "user", "content": task["prompt"]}]
                else:
                    messages = [
                        {"role": "user", "content": task["prompt"]},
                        {
                            "role": "user",
                            "content": REVISE.format(
                                prev=parents[task["id"]],
                                feedback=parent_fb[task["id"]],
                            ),
                        },
                    ]
                # diversify samples with a per-candidate seed nudge via temp jitter
                temp = 0.6 + 0.15 * float(rng.random())
                text, mean_lp = gen(
                    model, tokenizer, messages, MAX_TOKENS_GEN, temp=temp
                )
                ans = extract_answer(text)
                G = 1.0 if ans == task["answer"] else 0.0

                judge_messages = [
                    {
                        "role": "user",
                        "content": JUDGE_F.format(
                            task=task["prompt"], solution=text
                        ),
                    }
                ]
                j_text, _ = gen(
                    model, tokenizer, judge_messages, MAX_TOKENS_JUDGE, temp=0.2
                )
                j_score, feedback = extract_judge(j_text)

                j_mat[i, k] = j_score
                G_mat[i, k] = G
                cands.append(
                    {
                        "task_id": task["id"],
                        "family": "gsm_pilot",
                        "difficulty_bin": "pilot",
                        "cluster": "pilot",
                        "model": model_id,
                        "judge_cond": "J_self",
                        "rubric": "faithful",
                        "N": N,
                        "seed": SEED,
                        "round": t,
                        "cand_idx": k,
                        "text": text,
                        "j_score": j_score,
                        "feedback": feedback,
                        "G": G,
                        "answer_extracted": ans,
                        "ground_truth": task["answer"],
                        "selected": False,
                        "mean_logprob": mean_lp,
                        "prompt_hash": "pilot_v1",
                    }
                )
                print(
                    f"  t={t} {task['id']} c{k}: G={int(G)} j={j_score:.2f} ans={ans}",
                    flush=True,
                )

            # select
            k_star = int(np.argmax(j_mat[i]))
            sel_idx[i] = k_star
            cands[k_star]["selected"] = True
            parents[task["id"]] = cands[k_star]["text"]
            parent_fb[task["id"]] = cands[k_star]["feedback"]
            records.extend(cands)

        rq = round_quantities(j_mat, G_mat, sel_idx)
        round_dicts.append(rq)
        print(
            f"round {t}: J_sel={rq['J_sel']:.3f} G_sel={rq['G_sel']:.3f} "
            f"J_pop={rq['J_pop']:.3f} G_pop={rq['G_pop']:.3f}",
            flush=True,
        )

    with pool_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    decomp = decompose(round_dicts)
    # numpy -> list for JSON
    summary = {
        "model": model_id,
        "N": N,
        "T": T,
        "n_tasks": len(TASKS),
        "seed": SEED,
        "rounds": round_dicts,
        "bias_sel": decomp["bias_sel"].tolist(),
        "bias_pop": decomp["bias_pop"].tolist(),
        "raw_SCG": decomp["raw_SCG"].tolist(),
        "SEL": decomp["SEL"].tolist(),
        "ESC": decomp["ESC"].tolist(),
        "ESC_identity": decomp["ESC_identity"].tolist(),
        "identity_max_abs_err": float(
            np.abs(decomp["ESC"] - decomp["ESC_identity"]).max()
        ),
        "note": (
            "Plumbing pilot only — 7 toy GSM items, not a test of H1–H3. "
            "Power gate requires >> this n."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(records)} candidate records -> {pool_path}")
    print(f"Wrote ESC summary -> {summary_path}")
    print(f"identity |ESC - bias_pop drift| max = {summary['identity_max_abs_err']:.3e}")
    print("ESC(t):", [round(x, 4) for x in summary["ESC"]])
    print("SEL(t):", [round(x, 4) for x in summary["SEL"]])
    print("raw_SCG(t):", [round(x, 4) for x in summary["raw_SCG"]])
    print("PILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
