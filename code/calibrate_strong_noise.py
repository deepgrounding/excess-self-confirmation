"""Appendix B: J_strong~ noise-matching calibration (offline, steps 1-3).

Reads an existing round-0 J_self-scored JSONL (any run_stage2_esc.py output
with judge_cond=J_self), re-scores the same (task, candidate) pairs with
J_strong (default: local Gemma-4 12B via Ollama; main.md 4.2 amendment),
then binary-searches a frozen noise sd `tau` (and, if the SEL(0) magnitude
check fails, a linear scale `lambda`) so that AUC(J_strong~) matches
AUC(J_self) within 0.02 and the round-0 SEL magnitude ratio lands in
[0.8, 1.25]. Writes (tau, lambda) to --out for consumption by
run_stage2_esc.py --judge J_strong_tilde --noise-tau ... --noise-lambda ...

Usage:
  PYTHONUNBUFFERED=1 \\
  uv run --with numpy --with scipy \\
    python calibrate_strong_noise.py \\
    --self-scored ../data/stage2/gate_smoke_math_4b_self.jsonl \\
    --domain math --rubric faithful \\
    --out ../data/stage2/noise_calibration_math.json
  # Claude Haiku 4.5 cross-check: add --strong-backend openrouter
  # --strong-model anthropic/claude-haiku-4.5 (requires OPENROUTER_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_utils import apply_noise, api_chat_strong, extract_judge  # noqa: E402
from prompts_frozen import JUDGE_ARBITRABLE, JUDGE_FAITHFUL  # noqa: E402


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), 0.5 credit for ties."""
    from scipy.stats import rankdata

    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def sel0_by_task(scores: np.ndarray, task_ids: list[str]) -> float:
    """Mean(selected) - Mean(pool) at round 0, averaged over tasks with >=2 candidates."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for s, tid in zip(scores, task_ids):
        by_task[tid].append(s)
    diffs = []
    for tid, vals in by_task.items():
        if len(vals) < 2:
            continue
        vals = np.asarray(vals)
        diffs.append(vals.max() - vals.mean())
    return float(np.mean(diffs)) if diffs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-scored", type=Path, required=True)
    ap.add_argument(
        "--task-manifest",
        type=Path,
        required=True,
        help="the --task-manifest used to produce --self-scored; run_stage2_esc.py "
        "does not persist the task prompt text into its output rows, so this is "
        "needed to look the prompt back up by task_id",
    )
    ap.add_argument("--domain", choices=["math", "code"], required=True)
    ap.add_argument("--rubric", choices=["faithful", "arbitrable"], default="faithful")
    ap.add_argument("--strong-model", default="gemma4:12b")
    ap.add_argument(
        "--strong-backend",
        choices=["openrouter", "ollama"],
        default="ollama",
        help="ollama = amended default strong-judge path (e.g. gemma4:12b, "
        "local, free); openrouter = Claude Haiku 4.5 cross-check / API fallback",
    )
    ap.add_argument("--max-tokens-judge", type=int, default=120)
    ap.add_argument("--round", type=int, default=0, help="which round to calibrate on")
    ap.add_argument("--n-noise-draws", type=int, default=200)
    ap.add_argument("--tol", type=float, default=0.02, help="|AUC diff| stop threshold")
    ap.add_argument("--sel-ratio-lo", type=float, default=0.8)
    ap.add_argument("--sel-ratio-hi", type=float, default=1.25)
    ap.add_argument(
        "--noise-space",
        choices=["linear", "logit"],
        default="linear",
        help="linear = Appendix B frozen spec; logit = amendment to try when "
        "linear can't clear the SEL(0) gate on saturated (near-0/1) score "
        "distributions",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None, help="cache strong scores here (resume)")
    args = ap.parse_args()

    judge_tmpl = JUDGE_FAITHFUL if args.rubric == "faithful" else JUDGE_ARBITRABLE

    manifest_tasks = json.loads(args.task_manifest.read_text())["tasks"]
    prompt_by_id = {t["id"]: t["prompt"] for t in manifest_tasks}

    rows = []
    with args.self_scored.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("round") == args.round and r.get("domain", args.domain) == args.domain:
                rows.append(r)
    if not rows:
        raise SystemExit(f"no round={args.round} domain={args.domain} rows in {args.self_scored}")
    print(f"calibration pool: {len(rows)} round-0 candidates", flush=True)

    cache_path = args.cache or args.out.with_suffix(".strong_cache.json")
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    strong_scores = []
    for i, r in enumerate(rows):
        key = f"{r['task_id']}|{r['round']}|{r['cand_idx']}"
        if key in cache:
            strong_scores.append(cache[key])
            continue
        task_prompt = prompt_by_id.get(r["task_id"])
        if task_prompt is None:
            raise SystemExit(
                f"task_id {r['task_id']!r} not found in --task-manifest "
                f"{args.task_manifest}; wrong manifest for this --self-scored file?"
            )
        content = judge_tmpl.format(task=task_prompt, solution=r["text"])
        j_text = api_chat_strong(
            args.strong_backend, args.strong_model, content, args.max_tokens_judge
        )
        score, _fb = extract_judge(j_text)
        strong_scores.append(score)
        cache[key] = score
        if (i + 1) % 10 == 0:
            cache_path.write_text(json.dumps(cache))
            print(f"  strong-scored {i+1}/{len(rows)}", flush=True)
    cache_path.write_text(json.dumps(cache))

    self_scores = np.array([r["j_score"] for r in rows], dtype=float)
    strong_scores = np.array(strong_scores, dtype=float)
    G = np.array([r["G"] for r in rows], dtype=float)
    task_ids = [r["task_id"] for r in rows]

    auc_self = auc_score(self_scores, G)
    auc_strong = auc_score(strong_scores, G)
    print(f"AUC(J_self)={auc_self:.4f}  AUC(J_strong)={auc_strong:.4f}", flush=True)

    def auc_noisy(tau: float, lam: float, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        accum = 0.0
        for _ in range(args.n_noise_draws):
            noisy = apply_noise(strong_scores, tau, lam, rng, args.noise_space)
            accum += auc_score(noisy, G)
        return accum / args.n_noise_draws

    # Step 2: binary-search tau at lambda=1.
    lam = 1.0
    lo, hi = 0.0, 1.0
    # expand hi until AUC(noisy) <= AUC(self) (monotone-decreasing in tau, approx)
    for _ in range(6):
        if auc_noisy(hi, lam) <= auc_self:
            break
        hi *= 2.0
    tau = hi
    for _ in range(25):
        mid = (lo + hi) / 2.0
        a = auc_noisy(mid, lam)
        if abs(a - auc_self) <= args.tol:
            tau = mid
            break
        if a > auc_self:
            lo = mid
        else:
            hi = mid
        tau = mid
    auc_final = auc_noisy(tau, lam)
    print(f"binary search: tau={tau:.4f} -> AUC(J_strong~)={auc_final:.4f} "
          f"(target {auc_self:.4f}, tol {args.tol})", flush=True)
    tau_optimistic, auc_optimistic = tau, auc_final  # AUC-matched; kept for dual-bound reporting

    # Step 3: SEL(0) magnitude ratio check.
    rng = np.random.default_rng(7)
    noisy_final = apply_noise(strong_scores, tau, lam, rng, args.noise_space)
    sel_self = sel0_by_task(self_scores, task_ids)
    sel_strong_tilde = sel0_by_task(noisy_final, task_ids)
    ratio = abs(sel_strong_tilde) / abs(sel_self) if sel_self else float("nan")
    print(f"SEL(0): self={sel_self:.4f} strong~={sel_strong_tilde:.4f} ratio={ratio:.3f}", flush=True)

    gate_pass = abs(auc_final - auc_self) <= args.tol and args.sel_ratio_lo <= ratio <= args.sel_ratio_hi
    fallback_used = False
    if not gate_pass:
        print("SEL ratio gate failed at lambda=1; falling back to 2-param search "
              "(Appendix B step 3).", flush=True)
        fallback_used = True
        best = None
        for lam_try in np.linspace(0.5, 1.5, 11):
            lo, hi = 0.0, 1.0
            for _ in range(6):
                if auc_noisy(hi, lam_try) <= auc_self:
                    break
                hi *= 2.0
            tau_try = hi
            for _ in range(20):
                mid = (lo + hi) / 2.0
                a = auc_noisy(mid, lam_try)
                if abs(a - auc_self) <= args.tol:
                    tau_try = mid
                    break
                if a > auc_self:
                    lo = mid
                else:
                    hi = mid
                tau_try = mid
            a_final = auc_noisy(tau_try, lam_try)
            noisy_try = apply_noise(strong_scores, tau_try, lam_try, rng, args.noise_space)
            sel_try = sel0_by_task(noisy_try, task_ids)
            ratio_try = abs(sel_try) / abs(sel_self) if sel_self else float("nan")
            score_metric = abs(a_final - auc_self) + max(
                0.0, args.sel_ratio_lo - ratio_try, ratio_try - args.sel_ratio_hi
            )
            cand = (score_metric, lam_try, tau_try, a_final, ratio_try)
            if best is None or cand[0] < best[0]:
                best = cand
        _, lam, tau, auc_final, ratio = best
        gate_pass = abs(auc_final - auc_self) <= args.tol and args.sel_ratio_lo <= ratio <= args.sel_ratio_hi
        print(f"2-param result: lambda={lam:.3f} tau={tau:.4f} AUC={auc_final:.4f} "
              f"ratio={ratio:.3f} gate_pass={gate_pass}", flush=True)

    dual_bound = None
    if not gate_pass:
        # main.md line 241 / Appendix B fallback: calibration gate failed even
        # after the 2-param search. Report a dual bound instead of forcing a
        # single tau: 'optimistic' = the AUC-matched tau from step 2 (best
        # ranking fidelity, but SEL(0) magnitude under-matched); 'conservative'
        # = a tau chosen to hit the SEL(0) target directly (ratio ~ 1.0),
        # accepting whatever AUC mismatch that costs. Both are reported;
        # downstream H2 analysis runs ESC_sp under both and downgrades the
        # claim per the pre-registered contingency -- this does not "pass" the
        # gate, it documents the bracket honestly.
        print("Gate still failing after fallback; computing dual bound "
              "(optimistic AUC-matched tau vs conservative SEL-matched tau).", flush=True)

        def sel_ratio_noisy(tau_: float, lam_: float, seed: int = 11) -> float:
            rng_ = np.random.default_rng(seed)
            accum = 0.0
            for _ in range(args.n_noise_draws):
                noisy = apply_noise(strong_scores, tau_, lam_, rng_, args.noise_space)
                accum += abs(sel0_by_task(noisy, task_ids))
            mean_abs_sel = accum / args.n_noise_draws
            return mean_abs_sel / abs(sel_self) if sel_self else float("nan")

        lam_c = 1.0
        lo, hi = 0.0, 1.0
        for _ in range(8):
            if sel_ratio_noisy(hi, lam_c) >= 1.0:
                break
            hi *= 2.0
        tau_cons = hi
        for _ in range(20):
            mid = (lo + hi) / 2.0
            r = sel_ratio_noisy(mid, lam_c)
            if abs(r - 1.0) <= 0.05:
                tau_cons = mid
                break
            if r < 1.0:
                lo = mid
            else:
                hi = mid
            tau_cons = mid
        auc_cons = auc_noisy(tau_cons, lam_c)
        ratio_cons = sel_ratio_noisy(tau_cons, lam_c)
        print(f"conservative (SEL-matched): tau={tau_cons:.4f} lambda={lam_c:.3f} "
              f"-> AUC={auc_cons:.4f} (self={auc_self:.4f}) ratio={ratio_cons:.3f}", flush=True)
        dual_bound = {
            "optimistic": {
                "tau": tau_optimistic,
                "lambda": 1.0,
                "auc_strong_tilde": auc_optimistic,
                "note": "AUC-matched; SEL(0) magnitude under-matched (see sel0_ratio above)",
            },
            "conservative": {
                "tau": tau_cons,
                "lambda": lam_c,
                "auc_strong_tilde": auc_cons,
                "sel0_ratio": ratio_cons,
                "note": "SEL(0)-matched; AUC mismatch accepted as the cost",
            },
        }

    out = {
        "domain": args.domain,
        "rubric": args.rubric,
        "strong_model": args.strong_model,
        "strong_backend": args.strong_backend,
        "round": args.round,
        "n_calibration_items": len(rows),
        "auc_self": auc_self,
        "auc_strong_raw": auc_strong,
        "tau": tau,
        "lambda": lam,
        "auc_strong_tilde": auc_final,
        "sel0_self": sel_self,
        "sel0_strong_tilde": sel_strong_tilde,
        "sel0_ratio": ratio,
        "fallback_2param_used": fallback_used,
        "noise_space": args.noise_space,
        "gate_pass": gate_pass,
        "dual_bound": dual_bound,
        "n_noise_draws": args.n_noise_draws,
        "date": time.strftime("%Y-%m-%d"),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}; gate_pass={gate_pass}", flush=True)
    if not gate_pass:
        print("Calibration gate did not pass -- per main.md line 241 / Appendix B "
              "contingency, use dual_bound (optimistic/conservative tau) for H2 "
              "and downgrade claim strength. Do NOT pick a single tau and report "
              "it as gate_pass.", flush=True)


if __name__ == "__main__":
    main()
