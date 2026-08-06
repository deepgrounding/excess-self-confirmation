"""Regression tests for esc_core (no pytest required — run as script).

Covers: ESC identity, a_bar constants, tie-averaged AUC (incl. pilot pool),
collapse_flag boundaries, calibrate_noise on synthetic scores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from esc_core import (
    a_bar,
    auc,
    calibrate_noise,
    collapse_flag,
    decompose,
    round_quantities,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def pairwise_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Ground-truth: P(s+>s-) + 0.5 P(equal)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    total = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                total += 1.0
            elif p == n:
                total += 0.5
    return total / (len(pos) * len(neg))


def test_identity() -> None:
    print("identity")
    rng = np.random.default_rng(0)
    for _ in range(20):
        rounds = []
        for t in range(8):
            j = rng.random((30, 4))
            G = rng.random((30, 4))
            sel = rng.integers(0, 4, size=30)
            rounds.append(round_quantities(j, G, sel))
        d = decompose(rounds)
        err = float(np.max(np.abs(d["ESC"] - d["ESC_identity"])))
        if err >= 1e-15:
            check("ESC==ESC_identity", False, f"err={err}")
            return
    check("ESC==ESC_identity over 20 random traces", True, "max err < 1e-15")


def test_a_bar() -> None:
    print("a_bar")
    expected = {1: 0.0, 4: 1.02938, 8: 1.42360, 12: 1.62923}
    for N, e in expected.items():
        got = a_bar(N)
        check(f"a_bar({N})", abs(got - e) < 1e-4, f"{got:.5f} vs {e}")


def test_auc_no_ties() -> None:
    print("auc no ties")
    rng = np.random.default_rng(1)
    scores = rng.normal(size=200)
    labels = (rng.random(200) < 0.4).astype(int)
    # ensure both classes
    labels[0], labels[1] = 1, 0
    a = auc(scores, labels)
    p = pairwise_auc(scores, labels)
    check("auc matches pairwise (no ties)", abs(a - p) < 1e-12, f"{a:.6f} vs {p:.6f}")


def test_auc_with_ties() -> None:
    print("auc with ties")
    scores = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.5])
    labels = np.array([1, 0, 1, 0, 1, 0])
    a = auc(scores, labels)
    p = pairwise_auc(scores, labels)
    check("auc matches pairwise (ties)", abs(a - p) < 1e-12, f"{a:.6f} vs {p:.6f}")


def test_auc_pilot_pool() -> None:
    print("auc pilot pool 0.8B")
    path = DATA / "pilot_candidate_pool.jsonl"
    if not path.exists():
        check("pilot pool present", False, str(path))
        return
    scores, labels = [], []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            scores.append(float(r["j_score"]))
            labels.append(int(r["G"] >= 0.5) if isinstance(r["G"], float) else int(r["G"]))
            # G is 0/1 in pilot
            if "G" in r:
                labels[-1] = int(round(float(r["G"])))
    scores = np.array(scores)
    labels = np.array(labels)
    check("pilot pool has 84 rows", len(scores) == 84, f"n={len(scores)}")
    a = auc(scores, labels)
    p = pairwise_auc(scores, labels)
    check("auc == pairwise on pilot", abs(a - p) < 1e-12, f"{a:.6f}")
    check("auc ≈ 0.675 ± 0.001", abs(a - 0.675) < 0.001, f"auc={a:.4f}")


def test_collapse_flag() -> None:
    print("collapse_flag")
    # peak then drop exactly at boundaries
    g = np.array([0.5, 0.6, 0.7, 0.8, 0.8, 0.8 - 0.30 * 0.8])  # drop = 0.24 = 0.3*0.8
    # warmup=3 means late starts at index 4; need abs >= 0.10
    # indices: 0..5; warmup+1=4; late = dd[4:]
    # at t=5: peak=0.8, dd=0.24 >= 0.24 and >= 0.10 → hit
    check("exact 30% of peak hits", collapse_flag(g, rel=0.30, abs_floor=0.10, warmup=3) == 1)
    g2 = np.array([0.5, 0.6, 0.7, 0.8, 0.8, 0.8 - 0.29 * 0.8])  # 0.232 < 0.24
    check("just under 30% misses if abs also under? ", 
          collapse_flag(g2, rel=0.30, abs_floor=0.25, warmup=3) == 0)
    # small absolute drop
    g3 = np.array([0.5, 0.55, 0.6, 0.65, 0.65, 0.56])  # dd=0.09 < 0.10
    check("abs_floor blocks small drawdown", collapse_flag(g3, rel=0.10, abs_floor=0.10, warmup=3) == 0)
    # warmup boundary: drop only inside warmup should not count
    g4 = np.array([0.8, 0.2, 0.2, 0.2, 0.75, 0.75])  # big early drop, late ok
    check("warmup-only drop ignored", collapse_flag(g4, rel=0.30, abs_floor=0.10, warmup=3) == 0)


def test_calibrate_noise() -> None:
    print("calibrate_noise")
    rng = np.random.default_rng(2)
    # Strong scores nearly perfect; self target lower
    labels = np.array([1] * 40 + [0] * 40)
    scores_strong = np.where(labels == 1, 0.9, 0.1).astype(float) + rng.normal(0, 0.02, 80)
    target = 0.75
    tau, achieved = calibrate_noise(scores_strong, labels, target, seed=3, n_rep=100, tol=0.02)
    check("|ΔAUC| ≤ tol", abs(achieved - target) <= 0.02, f"tau={tau:.3f} auc={achieved:.3f}")


def main() -> None:
    print("esc_core regression tests")
    test_identity()
    test_a_bar()
    test_auc_no_ties()
    test_auc_with_ties()
    test_auc_pilot_pool()
    test_collapse_flag()
    test_calibrate_noise()
    print(f"\n{PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
