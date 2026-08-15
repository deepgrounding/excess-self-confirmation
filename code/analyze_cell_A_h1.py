"""H1 and H2 formal statistics for cell [A] / [B], per ANALYSIS_PLAN.md:

  H1: cell [A] J_self one-sided cluster bootstrap (B=10^4) on ESC(T) and
      AUC_ESC; both domains must agree in direction. Independent path:
      cell [B] N=1 SCG(T) by the same test.
  H2: ESC_sp(T) = ESC_self(T) - ESC_strong~(T) > 0, cell [A] paired,
      cluster bootstrap on paired item-level differences (same resampled
      task-id list applied to both conditions per replicate, not two
      independent bootstraps). Math runs under both dual-bound arms
      (optimistic / conservative tau) per the §4.2/§14.4 calibration outcome.

Reuses esc_core.decompose (the reference SEL/ESC implementation) and the
same task-cluster bootstrap construction as check_instrument_gate.py.
H1's ESC(T)/AUC_ESC and cell [B] N=1 SCG(T) checks run as soon as their
input files exist; the J_peer point estimate and the H2 paired tests
degrade gracefully (print a skip notice) until J_peer / J_strong_tilde
data lands from the running strongtilde_oracle_placebo_queue.

Usage:
  python3 analyze_cell_A_h1.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esc_core import decompose, auc  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "stage2"
B = 10_000  # per §7 item 1
SEED = 7    # matches the bootstrap seed used elsewhere in this project


def load_rounds(path: Path) -> tuple[list[dict], dict]:
    by_round_task: dict[int, dict[str, list[tuple[float, float, bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_round_task[r["round"]][r["task_id"]].append(
                (r["j_score"], r["G"], r["selected"])
            )
    rounds_sorted = sorted(by_round_task)
    task_ids = sorted({tid for rt in by_round_task.values() for tid in rt})
    full_rounds = [pooled_for(by_round_task[t], task_ids) for t in rounds_sorted]
    return full_rounds, {
        "by_round_task": by_round_task,
        "rounds_sorted": rounds_sorted,
        "task_ids": task_ids,
    }


def pooled_for(rt: dict, subset: list[str]) -> dict:
    J_all, G_all, J_sel, G_sel = [], [], [], []
    for tid in subset:
        for j, g, sel in rt.get(tid, []):
            J_all.append(j)
            G_all.append(g)
            if sel:
                J_sel.append(j)
                G_sel.append(g)
    return {
        "J_sel": float(np.mean(J_sel)) if J_sel else float("nan"),
        "G_sel": float(np.mean(G_sel)) if G_sel else float("nan"),
        "J_pop": float(np.mean(J_all)) if J_all else float("nan"),
        "G_pop": float(np.mean(G_all)) if G_all else float("nan"),
    }


def auc_esc_cumulative(esc: np.ndarray) -> float:
    """Trapezoidal area under ESC(t), t = 0..T. Natural 'cumulative AUC_ESC'
    reading of the abstract's phrase; not yet given a formula elsewhere in
    the released code, so this is this script's operationalization of it."""
    t = np.arange(len(esc))
    return float(np.trapezoid(esc, t))


def cluster_bootstrap(meta: dict, statfn, B: int, seed: int) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    task_ids = meta["task_ids"]
    n = len(task_ids)
    vals = np.empty(B)
    for b in range(B):
        sample = rng.choice(task_ids, size=n, replace=True).tolist()
        rounds = [pooled_for(meta["by_round_task"][t], sample) for t in meta["rounds_sorted"]]
        d = decompose(rounds)
        vals[b] = statfn(d)
    return float(np.mean(vals)), vals


def incomplete_reason(path: Path, expected_max_round: int = 15) -> str | None:
    """Guard against silently reporting T={whatever round the file happens to
    stop at} as if it were the pre-registered T=15. A file that exists but was
    interrupted mid-run (e.g. a graceful SIGTERM stop) has every task present
    in `task_ids` (round 0 always got written first) but NOT every task
    reaching the final round -- load_rounds()/decompose() would then compute
    ESC(T) at whatever T = max round present is, quietly mislabeled as T=15.
    Returns None if the file is safe to analyze at expected_max_round, else a
    human-readable reason string.
    """
    if not path.exists():
        return "file does not exist"
    max_round_per_task: dict[str, int] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            tid = r["task_id"]
            max_round_per_task[tid] = max(max_round_per_task.get(tid, -1), r["round"])
    if not max_round_per_task:
        return "file is empty"
    n_reached = sum(1 for r in max_round_per_task.values() if r >= expected_max_round)
    n_total = len(max_round_per_task)
    if n_reached < n_total:
        return (f"only {n_reached}/{n_total} tasks reached round {expected_max_round} "
                f"(run was interrupted mid-arm; max round present overall = "
                f"{max(max_round_per_task.values())})")
    return None


def round0_j_and_g(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-candidate (not pooled) round-0 judge scores and oracle labels,
    for a fresh AUC(J_self) vs AUC(J_strong~) noise-match recheck (used for
    the seed-42 replication, which reuses seed-13-calibrated (tau,lambda)
    rather than re-running calibrate_strong_noise.py)."""
    j, g = [], []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["round"] == 0:
                j.append(r["j_score"])
                g.append(r["G"])
    return np.array(j), np.array(g)


def report(label: str, path: Path, statname: str, statfn) -> dict:
    rounds, meta = load_rounds(path)
    d = decompose(rounds)
    point = statfn(d)
    boot_mean, vals = cluster_bootstrap(meta, statfn, B=B, seed=SEED)
    se = float(np.std(vals))
    ci_lo, ci_hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    p_one_sided = float((np.sum(vals <= 0) + 1) / (B + 1))
    print(
        f"{label:32s} {statname:10s} n={len(meta['task_ids']):3d}  "
        f"point={point:+.4f}  boot_SE={se:.4f}  95%CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  "
        f"one-sided p(<=0)={p_one_sided:.4f}"
    )
    return {
        "label": label, "jsonl": str(path), "stat": statname, "n_tasks": len(meta["task_ids"]),
        "point": point, "boot_se": se, "ci95": [ci_lo, ci_hi], "p_one_sided": p_one_sided,
    }


def paired_cluster_bootstrap(
    meta_a: dict, meta_b: dict, statfn, B: int, seed: int
) -> tuple[float, float, np.ndarray]:
    """H2 spec (ANALYSIS_PLAN.md): 'cluster bootstrap on paired item-level
    differences'. At each replicate, resample the SAME task-id list once and
    apply it to both conditions (paired, not two independent bootstraps),
    then difference statfn(decompose(...)) between them. Requires meta_a and
    meta_b to share the same task universe (same frozen pool, both domains).
    """
    task_ids = meta_a["task_ids"]
    assert set(task_ids) == set(meta_b["task_ids"]), "conditions must share the same task pool"
    rng = np.random.default_rng(seed)
    n = len(task_ids)
    diffs = np.empty(B)
    point_a = statfn(decompose([pooled_for(meta_a["by_round_task"][t], task_ids) for t in meta_a["rounds_sorted"]]))
    point_b = statfn(decompose([pooled_for(meta_b["by_round_task"][t], task_ids) for t in meta_b["rounds_sorted"]]))
    for b in range(B):
        sample = rng.choice(task_ids, size=n, replace=True).tolist()
        d_a = decompose([pooled_for(meta_a["by_round_task"][t], sample) for t in meta_a["rounds_sorted"]])
        d_b = decompose([pooled_for(meta_b["by_round_task"][t], sample) for t in meta_b["rounds_sorted"]])
        diffs[b] = statfn(d_a) - statfn(d_b)
    return point_a - point_b, float(np.std(diffs)), diffs


def report_paired(label: str, statname: str, path_a: Path, path_b: Path, B: int, seed: int) -> dict | None:
    """Generic paired ESC(T) difference: statfn(A) - statfn(B), same resampled
    task-id list applied to both per bootstrap replicate. Used for both H2
    (self - strong~) and ESC_adj (self - placebo, §5.2/§6.2 risk 4 contingency)."""
    for p in (path_a, path_b):
        why = incomplete_reason(p)
        if why is not None:
            print(f"  (skip {label}: {p} -- {why})")
            return None
    _, meta_a = load_rounds(path_a)
    _, meta_b = load_rounds(path_b)
    esc_t = lambda d: d["ESC"][-1]
    diff, se, vals = paired_cluster_bootstrap(meta_a, meta_b, esc_t, B, seed)
    ci_lo, ci_hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    p_one_sided = float((np.sum(vals <= 0) + 1) / (B + 1))
    print(
        f"{label:36s} {statname:10s} n={len(meta_a['task_ids']):3d}  "
        f"point={diff:+.4f}  boot_SE={se:.4f}  95%CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  "
        f"one-sided p(<=0)={p_one_sided:.4f}"
    )
    return {
        "label": label, "stat": statname, "path_a": str(path_a), "path_b": str(path_b),
        "n_tasks": len(meta_a["task_ids"]), "point": diff, "boot_se": se,
        "ci95": [ci_lo, ci_hi], "p_one_sided": p_one_sided,
    }


def main() -> None:
    results = []
    print(f"=== H1 primary: cell [A] J_self, ESC(T) and AUC_ESC (B={B}) ===")
    for domain in ["math", "code"]:
        path = DATA / f"A_self_{domain}_4b.jsonl"
        why = incomplete_reason(path)
        if why is not None:
            print(f"  (skip A_self_{domain}: {why})")
            continue
        T_stat = lambda d: d["ESC"][-1]
        results.append(report(f"A_self_{domain}", path, "ESC(T)", T_stat))
    for domain in ["math", "code"]:
        path = DATA / f"A_self_{domain}_4b.jsonl"
        if incomplete_reason(path) is not None:
            continue
        auc_stat = lambda d: auc_esc_cumulative(d["ESC"])
        results.append(report(f"A_self_{domain}", path, "AUC_ESC", auc_stat))

    print(f"\n=== H1 independent path: cell [B] N=1, SCG(T) (B={B}) ===")
    for domain in ["math", "code"]:
        path = DATA / f"B_n1_{domain}_4b_self.jsonl"
        why = incomplete_reason(path)
        if why is not None:
            print(f"  (skip B_n1_{domain}: {why})")
            continue
        scg_stat = lambda d: d["raw_SCG"][-1]
        results.append(report(f"B_n1_{domain}", path, "SCG(T)", scg_stat))

    print(f"\n=== H1 self > peer? (ESC(T) point comparison, no formal test until strong~ lands) ===")
    for domain in ["math", "code"]:
        path = DATA / f"A_peer_{domain}_4b.jsonl"
        why = incomplete_reason(path)
        if why is not None:
            print(f"  (skip A_peer_{domain}: {why})")
            continue
        T_stat = lambda d: d["ESC"][-1]
        results.append(report(f"A_peer_{domain}", path, "ESC(T)", T_stat))

    print(f"\n=== Instrument gate at full scale: J_oracle / J_placebo (B={B}) ===")
    print("    (same test as the §14.4 pilot-scale gate, 20 items/domain; this is the full")
    print("     84/91-item pool once A_oracle_*/A_placebo_* land from the strongtilde queue)")
    for judge in ["oracle", "placebo"]:
        for domain in ["math", "code"]:
            path = DATA / f"A_{judge}_{domain}_4b.jsonl"
            why = incomplete_reason(path)
            if why is not None:
                print(f"  (skip A_{judge}_{domain}_fullscale: {why})")
                continue
            if judge == "oracle":
                results.append(report(f"A_oracle_{domain}_fullscale", path, "SEL(T)", lambda d: d["SEL"][-1]))
                results.append(report(f"A_oracle_{domain}_fullscale", path, "ESC(T)", lambda d: d["ESC"][-1]))
            else:
                results.append(report(f"A_placebo_{domain}_fullscale", path, "ESC(T)", lambda d: d["ESC"][-1]))

    print(f"\n=== H2: ESC_sp(T) = ESC_self(T) - ESC_strong~(T), paired cluster bootstrap (B={B}) ===")
    print("    (ANALYSIS_PLAN.md: cell [A] paired; code = single strong~ arm, math = dual-bound opt/cons)")
    h2_specs = [
        ("H2_code", DATA / "A_self_code_4b.jsonl", DATA / "A_strongtilde_code_4b_main.jsonl"),
        ("H2_math_optimistic", DATA / "A_self_math_4b.jsonl", DATA / "A_strongtilde_math_4b_opt.jsonl"),
        ("H2_math_conservative", DATA / "A_self_math_4b.jsonl", DATA / "A_strongtilde_math_4b_cons.jsonl"),
    ]
    for label, path_a, path_b in h2_specs:
        r = report_paired(label, "ESC_sp(T)", path_a, path_b, B, SEED)
        if r is not None:
            results.append(r)

    print(f"\n=== ESC_adj(T) = ESC_self(T) - ESC_placebo(T), paired cluster bootstrap (B={B}) ===")
    print("    (main.md §5.2 / §6.2 risk 4: triggered when ESC_placebo is significantly > 0 --")
    print("     computed for both domains regardless so both can be reported, per the risk-4 mitigation)")
    adj_specs = [
        ("ESC_adj_math", DATA / "A_self_math_4b.jsonl", DATA / "A_placebo_math_4b.jsonl"),
        ("ESC_adj_code", DATA / "A_self_code_4b.jsonl", DATA / "A_placebo_code_4b.jsonl"),
    ]
    for label, path_a, path_b in adj_specs:
        r = report_paired(label, "ESC_adj(T)", path_a, path_b, B, SEED)
        if r is not None:
            results.append(r)

    print(f"\n=== Second-seed replication (seed=42), critical cells per ANALYSIS_PLAN.md:25 (B={B}) ===")
    for domain in ["math", "code"]:
        path = DATA / f"A_self_{domain}_4b_seed42.jsonl"
        why = incomplete_reason(path)
        if why is not None:
            print(f"  (skip A_self_{domain}_seed42: {why})")
            continue
        T_stat = lambda d: d["ESC"][-1]
        results.append(report(f"A_self_{domain}_seed42", path, "ESC(T)", T_stat))
    for domain in ["math", "code"]:
        path = DATA / f"A_self_{domain}_4b_seed42.jsonl"
        if incomplete_reason(path) is not None:
            continue
        auc_stat = lambda d: auc_esc_cumulative(d["ESC"])
        results.append(report(f"A_self_{domain}_seed42", path, "AUC_ESC", auc_stat))

    print(f"\n=== Second-seed independent path: cell [B] N=1, seed=42, SCG(T) (B={B}) ===")
    for domain in ["math", "code"]:
        path = DATA / f"B_n1_{domain}_4b_self_seed42.jsonl"
        why = incomplete_reason(path)
        if why is not None:
            print(f"  (skip B_n1_{domain}_seed42: {why})")
            continue
        scg_stat = lambda d: d["raw_SCG"][-1]
        results.append(report(f"B_n1_{domain}_seed42", path, "SCG(T)", scg_stat))

    print(f"\n=== Second-seed H2: ESC_sp(T), seed=42, paired cluster bootstrap (B={B}) ===")
    h2_seed42_specs = [
        ("H2_code_seed42", DATA / "A_self_code_4b_seed42.jsonl", DATA / "A_strongtilde_code_4b_seed42_main.jsonl"),
        ("H2_math_optimistic_seed42", DATA / "A_self_math_4b_seed42.jsonl", DATA / "A_strongtilde_math_4b_seed42_opt.jsonl"),
        ("H2_math_conservative_seed42", DATA / "A_self_math_4b_seed42.jsonl", DATA / "A_strongtilde_math_4b_seed42_cons.jsonl"),
    ]
    for label, path_a, path_b in h2_seed42_specs:
        r = report_paired(label, "ESC_sp(T)", path_a, path_b, B, SEED)
        if r is not None:
            results.append(r)

    print(f"\n=== Second-seed noise-match diagnostic recheck (seed=42): reuses seed-13 (tau,lambda) ===")
    print("    (gate: |AUC(J_self) - AUC(J_strong~)| <= 0.02, same threshold as Appendix B)")
    noise_recheck_specs = [
        ("code_main", "A_self_code_4b_seed42.jsonl", "A_strongtilde_code_4b_seed42_main.jsonl"),
        ("math_optimistic", "A_self_math_4b_seed42.jsonl", "A_strongtilde_math_4b_seed42_opt.jsonl"),
        ("math_conservative", "A_self_math_4b_seed42.jsonl", "A_strongtilde_math_4b_seed42_cons.jsonl"),
    ]
    for label, self_f, strong_f in noise_recheck_specs:
        p_self, p_strong = DATA / self_f, DATA / strong_f
        if not p_self.exists() or not p_strong.exists():
            print(f"  (skip noise_match_seed42_{label}: file missing)")
            continue
        j_self, g_self = round0_j_and_g(p_self)
        j_strong, g_strong = round0_j_and_g(p_strong)
        auc_self, auc_strong = auc(j_self, g_self), auc(j_strong, g_strong)
        delta = auc_self - auc_strong
        gate = "PASS" if abs(delta) <= 0.02 else "FAIL"
        print(
            f"  {label:20s} AUC(self)={auc_self:.4f}  AUC(strong~)={auc_strong:.4f}  "
            f"Delta={delta:+.4f}  gate={gate}"
        )
        results.append({
            "label": f"noise_match_seed42_{label}", "auc_self": auc_self,
            "auc_strong": auc_strong, "delta": delta, "gate_pass": abs(delta) <= 0.02,
        })

    print(f"\n=== Second model family (Llama-3.2-3B-Instruct), J_self only (B={B}) ===")
    print("    (generalization check for H1's domain split; own screened pool, not Qwen's -- see §6.1)")
    for domain in ["math", "code"]:
        path = DATA / f"A_self_{domain}_3b_llama.jsonl"
        why = incomplete_reason(path)
        if why is not None:
            print(f"  (skip A_self_{domain}_llama3b: {why})")
            continue
        T_stat = lambda d: d["ESC"][-1]
        results.append(report(f"A_self_{domain}_llama3b", path, "ESC(T)", T_stat))
    for domain in ["math", "code"]:
        path = DATA / f"A_self_{domain}_3b_llama.jsonl"
        if incomplete_reason(path) is not None:
            continue
        auc_stat = lambda d: auc_esc_cumulative(d["ESC"])
        results.append(report(f"A_self_{domain}_llama3b", path, "AUC_ESC", auc_stat))

    out = Path(__file__).resolve().parent.parent / "data" / "stage2" / "h1_analysis.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
