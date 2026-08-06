"""Production ESC closed-loop harness (Stage-2).

Upgrades pilot_trajectory.py:
  - real GSM8K / HumanEval(+MBPP) task loaders + verifiers
  - resume by (task_id, round, cand_idx)
  - atomic JSONL append (tmp + rename per record batch)
  - faithful / arbitrable rubrics from prompts_frozen.py
  - Appendix D schema fields

Examples:
  PYTHONUNBUFFERED=1 uv run --with mlx-lm --with mlx --with numpy --with datasets \\
    python run_stage2_esc.py --model mlx-community/Qwen3.5-4B-4bit \\
    --domain math --n 100 --N 4 --T 15 --rubric faithful \\
    --out ../data/stage2/cellA_math_self.jsonl --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from mlx_lm import load
from mlx_lm.generate import stream_generate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esc_core import decompose, round_quantities  # noqa: E402
from judge_utils import (  # noqa: E402
    apply_noise,
    api_chat_strong,
    extract_judge,
    strong_noise_seed,
)
from prompts_frozen import (  # noqa: E402
    GEN_CODE,
    GEN_MATH,
    JUDGE_ARBITRABLE,
    JUDGE_FAITHFUL,
    PLACEBO_FEEDBACK,
    REVISE_CODE,
    REVISE_MATH,
)
from verifiers import code_correct, extract_math_answer, math_correct  # noqa: E402


def chat_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def gen(model, tokenizer, messages, max_tokens: int, temp: float = 0.7):
    prompt = chat_prompt(tokenizer, messages)
    text, lp_sum, n_tok = "", 0.0, 0
    kwargs = {"max_tokens": max_tokens}
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


def load_gsm8k(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n]
    tasks = []
    for i in idx:
        row = ds[int(i)]
        gold = extract_math_answer(row["answer"]) or row["answer"].strip().split()[-1]
        tasks.append(
            {
                "id": f"gsm8k_{int(i)}",
                "domain": "math",
                "family": "gsm8k",
                "prompt": GEN_MATH.format(question=row["question"].strip()),
                "gold": gold,
                "test_code": None,
                "entry_point": None,
                "question_text": row["question"].strip(),
            }
        )
    return tasks


MBPP_ENTRY_RE = re.compile(r"assert\s+([A-Za-z_]\w*)\s*\(")


def load_math_hendrycks(n: int, seed: int) -> list[dict]:
    """MATH items. The original `hendrycks/competition_math` repo is no longer
    downloadable from the Hub (script-based dataset, removed), so we load from
    mirrors: HuggingFaceH4/MATH-500 first (canonical 500-problem test subset,
    clean parquet, has `answer` + `level` fields), then EleutherAI's per-subject
    mirror. Per the frozen design (main.md section 4.3) sampling is stratified
    toward difficulty levels 3-5 when level metadata is available."""
    from datasets import load_dataset

    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        source = "HuggingFaceH4/MATH-500"
    except Exception:
        from datasets import concatenate_datasets

        configs = [
            "algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
        ]
        parts = [
            load_dataset("EleutherAI/hendrycks_math", c, split="test")
            for c in configs
        ]
        ds = concatenate_datasets(parts)
        source = "EleutherAI/hendrycks_math"
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))

    def level_of(row) -> int | None:
        lv = row.get("level")
        if lv is None:
            return None
        m = re.search(r"(\d)", str(lv))
        return int(m.group(1)) if m else None

    # stratify toward levels 3-5: take those first, then backfill
    preferred = [i for i in order if (level_of(ds[int(i)]) or 3) >= 3]
    rest = [i for i in order if i not in set(preferred)]
    idx = (preferred + rest)[:n]
    tasks = []
    for i in idx:
        row = ds[int(i)]
        gold = str(row.get("answer") or "").strip() or None
        if gold is None:
            sol = str(row.get("solution") or "")
            m = re.search(r"\\boxed\{([^{}]+)\}", sol)
            gold = m.group(1).strip() if m else sol.strip().split()[-1]
        tasks.append(
            {
                "id": f"math_{int(i)}",
                "domain": "math",
                "family": "math",
                "source": source,
                "prompt": GEN_MATH.format(question=row["problem"].strip()),
                "gold": gold,
                "test_code": None,
                "entry_point": None,
                "question_text": row["problem"].strip(),
            }
        )
    return tasks


def load_humaneval(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[: min(n, len(ds))]
    tasks = []
    for i in idx:
        row = ds[int(i)]
        tasks.append(
            {
                "id": f"humaneval_{row['task_id']}",
                "domain": "code",
                "family": "humaneval",
                "prompt": GEN_CODE.format(prompt=row["prompt"]),
                "gold": None,
                "test_code": row["test"],
                "entry_point": row["entry_point"],
                "question_text": row["prompt"],
            }
        )
    return tasks


def load_mbpp(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    # Prefer sanitized; fall back to full if unavailable. Bare "mbpp" repo ids
    # are rejected by current datasets/hub versions; use the full namespace.
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    except Exception:
        ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[: min(n, len(ds))]
    tasks = []
    for i in idx:
        row = ds[int(i)]
        tests = row.get("test_list") or []
        m = MBPP_ENTRY_RE.search(tests[0]) if tests else None
        entry_point = m.group(1) if m else None
        setup = row.get("test_setup_code") or ""
        test_code = (setup + "\n" if setup else "") + "\n".join(tests)
        if "prompt" in row and row["prompt"]:
            q = row["prompt"].strip()
        else:
            q = str(row.get("text") or "").strip()
        if entry_point:
            q = f"{q}\n\nImplement a function named `{entry_point}`."
        tasks.append(
            {
                "id": f"mbpp_{row['task_id']}",
                "domain": "code",
                "family": "mbpp",
                "prompt": GEN_CODE.format(prompt=q),
                "gold": None,
                "test_code": test_code,
                "entry_point": None,  # asserts run directly
                "question_text": q,
            }
        )
    return tasks


def load_tasks(domain: str, n: int, seed: int, family: str | None = None) -> list[dict]:
    """Load n items for a domain. family=None mixes families 50/50 when possible."""
    if domain == "math":
        if family == "gsm8k":
            return load_gsm8k(n, seed)
        if family == "math":
            return load_math_hendrycks(n, seed)
        a, b = n // 2, n - n // 2
        return load_gsm8k(a, seed) + load_math_hendrycks(b, seed + 1)
    if domain == "code":
        if family == "humaneval":
            return load_humaneval(n, seed)
        if family == "mbpp":
            return load_mbpp(n, seed)
        a, b = n // 2, n - n // 2
        return load_humaneval(a, seed) + load_mbpp(b, seed + 1)
    raise ValueError(f"domain must be math|code, got {domain}")


def oracle_G(task: dict, text: str) -> float:
    if task["domain"] == "math":
        return math_correct(text, task["gold"])
    return code_correct(text, task["test_code"], entry_point=task["entry_point"])


def done_keys(path: Path) -> set[tuple]:
    keys = set()
    if not path.exists():
        return keys
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            keys.add((r["task_id"], int(r["round"]), int(r["cand_idx"])))
    return keys


def atomic_append_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    # append: copy existing then new lines, rename over
    existing = path.read_text() if path.exists() else ""
    with tmp.open("w") as f:
        if existing:
            f.write(existing)
            if not existing.endswith("\n"):
                f.write("\n")
        for line in lines:
            f.write(line if line.endswith("\n") else line + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-4bit")
    ap.add_argument("--domain", choices=["math", "code"], default="math")
    ap.add_argument(
        "--family",
        default=None,
        help="optional: gsm8k|math|humaneval|mbpp (default: mix)",
    )
    ap.add_argument("--n", type=int, default=20, help="number of items")
    ap.add_argument("--N", type=int, default=4, help="candidates per round")
    ap.add_argument("--T", type=int, default=2, help="last round index (rounds 0..T)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--rubric", choices=["faithful", "arbitrable"], default="faithful")
    ap.add_argument(
        "--judge",
        default="J_self",
        choices=["J_self", "J_placebo", "J_oracle", "J_peer", "J_strong", "J_strong_tilde"],
    )
    ap.add_argument(
        "--peer-model",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="local MLX model for J_peer (Appendix, 4.2: same size class, different family)",
    )
    ap.add_argument(
        "--strong-model",
        default="gemma4:12b",
        help="model id for J_strong / J_strong_tilde (local Ollama tag, or "
        "OpenRouter slug when --strong-backend openrouter). main.md 4.2 "
        "amendment default is Gemma-4 12B via Ollama (local, free); Claude "
        "Haiku 4.5 via OpenRouter is retained as a cross-check.",
    )
    ap.add_argument(
        "--strong-backend",
        choices=["openrouter", "ollama"],
        default="ollama",
        help="ollama = amended default strong-judge path (e.g. gemma4:12b, "
        "local); openrouter = Claude Haiku 4.5 cross-check / API fallback "
        "(main.md 4.2 amendment, §10 risk 9)",
    )
    ap.add_argument(
        "--noise-tau",
        type=float,
        default=None,
        help="frozen noise sd for J_strong_tilde; required when --judge J_strong_tilde "
        "(produced by calibrate_strong_noise.py, Appendix B)",
    )
    ap.add_argument(
        "--noise-lambda",
        type=float,
        default=1.0,
        help="optional linear scale on J_strong score before noise (Appendix B step 3 fallback)",
    )
    ap.add_argument(
        "--noise-space",
        choices=["linear", "logit"],
        default="linear",
        help="must match whatever calibrate_strong_noise.py --noise-space produced "
        "the --noise-tau/--noise-lambda being used here",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-tokens-gen", type=int, default=512)
    ap.add_argument("--max-tokens-judge", type=int, default=120)
    ap.add_argument("--difficulty-bin", default="unscreened")
    ap.add_argument("--cluster", default="0")
    ap.add_argument(
        "--task-manifest",
        type=Path,
        default=None,
        help="JSON from screen_items.py; overrides --n/--family sampling",
    )
    args = ap.parse_args()

    if args.judge == "J_strong_tilde" and args.noise_tau is None:
        raise SystemExit(
            "--judge J_strong_tilde requires --noise-tau (run calibrate_strong_noise.py "
            "first, Appendix B step 1-3)."
        )

    judge_tmpl = JUDGE_FAITHFUL if args.rubric == "faithful" else JUDGE_ARBITRABLE
    revise_tmpl = REVISE_MATH if args.domain == "math" else REVISE_CODE
    prompt_hash = hashlib.sha1(
        (judge_tmpl + revise_tmpl + args.rubric).encode()
    ).hexdigest()[:12]

    if args.task_manifest:
        tasks = json.loads(args.task_manifest.read_text())["tasks"]
        tasks = [t for t in tasks if t.get("domain", args.domain) == args.domain]
    else:
        tasks = load_tasks(args.domain, args.n, args.seed, family=args.family)
    done = done_keys(args.out) if args.resume else set()
    if args.out.exists() and not args.resume:
        raise SystemExit(f"{args.out} exists; pass --resume or choose a new --out")

    print(f"Loading {args.model} ...", flush=True)
    t_load = time.time()
    model, tokenizer = load(args.model)
    print(f"loaded in {time.time()-t_load:.1f}s; items={len(tasks)} done_keys={len(done)}", flush=True)

    peer_model = peer_tokenizer = None
    if args.judge == "J_peer":
        print(f"Loading peer judge {args.peer_model} ...", flush=True)
        t_peer = time.time()
        peer_model, peer_tokenizer = load(args.peer_model)
        print(f"peer loaded in {time.time()-t_peer:.1f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    parents = {t["id"]: None for t in tasks}
    parent_fb = {t["id"]: None for t in tasks}

    # Restore parents from existing file if resuming mid-trajectory
    if done and args.out.exists():
        by_task_round: dict = {}
        with args.out.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("selected"):
                    by_task_round[(r["task_id"], r["round"])] = r
        for t in tasks:
            for rnd in range(args.T, -1, -1):
                key = (t["id"], rnd)
                if key in by_task_round:
                    parents[t["id"]] = by_task_round[key]["text"]
                    parent_fb[t["id"]] = by_task_round[key]["feedback"]
                    break

    call_times = []
    for rnd in range(args.T + 1):
        print(f"=== round {rnd} ===", flush=True)
        for task in tasks:
            lines_out = []
            scores_j = []
            scores_G = []
            texts = []
            feedbacks = []
            need_any = any(
                (task["id"], rnd, k) not in done for k in range(args.N)
            )
            if not need_any:
                # still need scores for selection restore — skip generation
                continue

            for k in range(args.N):
                key = (task["id"], rnd, k)
                if key in done:
                    # load existing for matrices
                    continue
                t0 = time.time()
                if rnd == 0:
                    messages = [{"role": "user", "content": task["prompt"]}]
                else:
                    messages = [
                        {"role": "user", "content": task["prompt"]},
                        {
                            "role": "user",
                            "content": revise_tmpl.format(
                                prev=parents[task["id"]],
                                feedback=parent_fb[task["id"]],
                            ),
                        },
                    ]
                temp = 0.6 + 0.15 * float(rng.random())
                text, mean_lp = gen(
                    model, tokenizer, messages, args.max_tokens_gen, temp=temp
                )
                G = oracle_G(task, text)

                j_score_raw = None
                if args.judge == "J_placebo":
                    j_score, feedback = float(rng.random()), PLACEBO_FEEDBACK
                elif args.judge == "J_oracle":
                    j_score = G
                    feedback = (
                        "Ground truth: solution is correct."
                        if G >= 0.5
                        else "Ground truth: solution is incorrect."
                    )
                elif args.judge == "J_peer":
                    j_msg = [
                        {
                            "role": "user",
                            "content": judge_tmpl.format(
                                task=task["prompt"], solution=text
                            ),
                        }
                    ]
                    j_text, _ = gen(
                        peer_model, peer_tokenizer, j_msg, args.max_tokens_judge, temp=0.2
                    )
                    j_score, feedback = extract_judge(j_text)
                elif args.judge in ("J_strong", "J_strong_tilde"):
                    prompt_txt = judge_tmpl.format(task=task["prompt"], solution=text)
                    j_text = api_chat_strong(
                        args.strong_backend, args.strong_model, prompt_txt, args.max_tokens_judge
                    )
                    strong_score, feedback = extract_judge(j_text)
                    j_score_raw = strong_score
                    if args.judge == "J_strong_tilde":
                        seed_i = strong_noise_seed(task["id"], rnd, k)
                        noise_rng = np.random.default_rng(seed_i)
                        j_score = float(
                            apply_noise(
                                np.array([strong_score]),
                                args.noise_tau,
                                args.noise_lambda,
                                noise_rng,
                                args.noise_space,
                            )[0]
                        )
                    else:
                        j_score = strong_score
                else:  # J_self
                    j_msg = [
                        {
                            "role": "user",
                            "content": judge_tmpl.format(
                                task=task["prompt"], solution=text
                            ),
                        }
                    ]
                    j_text, _ = gen(
                        model, tokenizer, j_msg, args.max_tokens_judge, temp=0.2
                    )
                    j_score, feedback = extract_judge(j_text)

                elapsed = time.time() - t0
                call_times.append(elapsed)
                rec = {
                    "task_id": task["id"],
                    "family": task["family"],
                    "domain": task["domain"],
                    "difficulty_bin": task.get("difficulty_bin", args.difficulty_bin),
                    "cluster": str(task.get("cluster", args.cluster)),
                    "model": args.model,
                    "judge_cond": args.judge,
                    "strong_backend": args.strong_backend if args.judge in ("J_strong", "J_strong_tilde") else None,
                    "strong_model": args.strong_model if args.judge in ("J_strong", "J_strong_tilde") else None,
                    "rubric": args.rubric,
                    "N": args.N,
                    "seed": args.seed,
                    "round": rnd,
                    "cand_idx": k,
                    "text": text,
                    "j_score": j_score,
                    "j_score_raw": j_score_raw,
                    "feedback": feedback,
                    "G": G,
                    "selected": False,
                    "mean_logprob": mean_lp,
                    "distinct_stats": None,
                    "prompt_hash": prompt_hash,
                }
                lines_out.append(json.dumps(rec, ensure_ascii=False))
                scores_j.append(j_score)
                scores_G.append(G)
                texts.append(text)
                feedbacks.append(feedback)
                print(
                    f"  {task['id']} r{rnd}c{k}: G={G:.0f} j={j_score:.2f} {elapsed:.1f}s",
                    flush=True,
                )

            if lines_out:
                # select among newly generated only if full N new; else reload
                # For simplicity when partial resume: rewrite selection after full N present
                atomic_append_lines(args.out, lines_out)

            # Recompute selection for this (task, round) from file
            cands = []
            with args.out.open() as f:
                for line in f:
                    r = json.loads(line)
                    if r["task_id"] == task["id"] and r["round"] == rnd:
                        cands.append(r)
            if len(cands) < args.N:
                continue
            # clear selected flags in-memory and pick argmax
            best = max(range(args.N), key=lambda i: cands[i]["j_score"])
            parents[task["id"]] = cands[best]["text"]
            parent_fb[task["id"]] = cands[best]["feedback"]
            # patch selected flags on disk (rewrite file — ok for Stage-2 sizes)
            _patch_selected(args.out, task["id"], rnd, best)

    if call_times:
        print(
            f"throughput: n_calls={len(call_times)} "
            f"mean_s={np.mean(call_times):.2f} median_s={np.median(call_times):.2f}",
            flush=True,
        )
    print(f"Done. Wrote {args.out}", flush=True)


def _patch_selected(path: Path, task_id: str, rnd: int, best_idx: int) -> None:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["task_id"] == task_id and r["round"] == rnd:
                r["selected"] = r["cand_idx"] == best_idx
            rows.append(r)
    tmp = path.with_suffix(path.suffix + ".patchtmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
