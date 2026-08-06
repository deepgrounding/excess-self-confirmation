"""Core metric implementations for the SEL/ESC decomposition.

Implements the round-level quantities, the raw self-confirmation gap (SCG),
the per-round optimizer's-curse term (SEL), the excess self-confirmation
metric (ESC) with its pool-drift identity, the analytic winner's-curse
baseline sigma_j * a_bar_N, the AUC-matched noise calibration used to build
the J_strong~ control arm, and the within-condition early-warning statistics
(weighted AUC + within-condition label permutation test).

Everything here is pure NumPy; no model calls. These functions are the
reference implementation released with the paper and the object under test
in sim_validate.py.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# Round-level quantities and the SCG = SEL + ESC decomposition
# ----------------------------------------------------------------------

def round_quantities(j: np.ndarray, G: np.ndarray, sel: np.ndarray) -> dict:
    """Per-round means over one candidate pool.

    j, G : (n_tasks, N) judge scores and oracle true scores for every
           candidate in the round's pool.
    sel  : (n_tasks,) index of the selected candidate per task.
    """
    rows = np.arange(j.shape[0])
    return {
        "J_sel": float(j[rows, sel].mean()),
        "G_sel": float(G[rows, sel].mean()),
        "J_pop": float(j.mean()),
        "G_pop": float(G.mean()),
    }


def decompose(rounds: list[dict]) -> dict:
    """Full decomposition from a list of per-round quantity dicts.

    Returns arrays indexed by round t = 0..T:
      bias_sel, bias_pop, raw_SCG, SEL, dSEL (= SEL(t)-SEL(0)),
      ESC (= raw_SCG - dSEL), ESC_identity (= bias_pop(t)-bias_pop(0)).
    The identity ESC == ESC_identity holds algebraically; sim_validate.py
    checks it to machine precision.
    """
    J_sel = np.array([r["J_sel"] for r in rounds])
    G_sel = np.array([r["G_sel"] for r in rounds])
    J_pop = np.array([r["J_pop"] for r in rounds])
    G_pop = np.array([r["G_pop"] for r in rounds])
    bias_sel = J_sel - G_sel
    bias_pop = J_pop - G_pop
    raw_scg = bias_sel - bias_sel[0]
    sel_term = bias_sel - bias_pop
    d_sel = sel_term - sel_term[0]
    esc = raw_scg - d_sel
    esc_identity = bias_pop - bias_pop[0]
    return {
        "bias_sel": bias_sel,
        "bias_pop": bias_pop,
        "raw_SCG": raw_scg,
        "SEL": sel_term,
        "dSEL": d_sel,
        "ESC": esc,
        "ESC_identity": esc_identity,
    }


# ----------------------------------------------------------------------
# Analytic winner's-curse baseline
# ----------------------------------------------------------------------

def a_bar(N: int, grid: int = 200001, lim: float = 10.0) -> float:
    """E[max of N iid standard normals], by numerical integration of
    N * x * phi(x) * Phi(x)^(N-1)."""
    if N == 1:
        return 0.0
    x = np.linspace(-lim, lim, grid)
    phi = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    # Phi via cumulative trapezoid of phi (avoids scipy dependency)
    dx = x[1] - x[0]
    Phi = np.concatenate([[0.0], np.cumsum((phi[1:] + phi[:-1]) * 0.5 * dx)])
    Phi = np.clip(Phi, 0.0, 1.0)
    integrand = N * x * phi * Phi ** (N - 1)
    return float(np.trapezoid(integrand, x))


def analytic_sel(sigma_j: float, N: int) -> float:
    """Winner's-curse magnitude E[j_(argmax) - mean(j)] for a judge with
    zero discrimination and iid N(0, sigma_j^2) noise (upper bound when the
    judge has discrimination)."""
    return sigma_j * a_bar(N)


# ----------------------------------------------------------------------
# AUC and the noise-matching calibration (J_strong~)
# ----------------------------------------------------------------------

def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC of scores discriminating labels==1.

    Uses *average ranks* for ties, so that P(score+ > score-) + 0.5 P(equal)
    is recovered exactly. Plain argsort ranks without tie-averaging bias the
    statistic when many scores collide (common for discrete LLM judge scores).
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    all_s = np.concatenate([pos, neg])
    order = np.argsort(all_s, kind="mergesort")
    sorted_s = all_s[order]
    ranks = np.empty(len(all_s), dtype=float)
    i = 0
    n = len(all_s)
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        # 1-indexed ranks i+1 .. j share the average
        avg = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg
        i = j
    r_pos = float(ranks[:n_pos].sum())
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def calibrate_noise(
    scores_strong: np.ndarray,
    labels: np.ndarray,
    target_auc: float,
    seed: int = 0,
    n_rep: int = 200,
    tol: float = 0.002,
    tau_hi: float = 20.0,
) -> tuple[float, float]:
    """Binary-search the noise scale tau such that
    AUC(scores_strong + N(0, tau^2)) matches target_auc.

    Returns (tau, achieved_auc). Averaging over n_rep noise draws per
    candidate tau keeps the search stable. This is the offline half of the
    J_strong~ protocol; the frozen tau is then injected online.
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n_rep, len(scores_strong)))

    def mean_auc(tau: float) -> float:
        return float(
            np.mean([auc(scores_strong + tau * e, labels) for e in eps])
        )

    lo, hi = 0.0, tau_hi
    if mean_auc(hi) > target_auc:
        raise ValueError("target AUC unreachable: raise tau_hi")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        a = mean_auc(mid)
        if abs(a - target_auc) <= tol:
            return mid, a
        if a > target_auc:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid, mean_auc(mid)


# ----------------------------------------------------------------------
# Early-warning statistics (H3 machinery)
# ----------------------------------------------------------------------

def early_slope(series: np.ndarray, k: int = 3) -> float:
    """OLS slope of series over rounds t = 1..k (pre-registered k = 3)."""
    t = np.arange(1, k + 1, dtype=float)
    y = np.asarray(series[1 : k + 1], dtype=float)
    t_c = t - t.mean()
    return float((t_c * (y - y.mean())).sum() / (t_c**2).sum())


def collapse_flag(
    g_sel: np.ndarray, rel: float = 0.30, abs_floor: float = 0.10, warmup: int = 3
) -> int:
    """Pre-registered collapse definition: max drawdown of G_sel after the
    early window, >= rel * running peak and >= abs_floor in absolute terms."""
    peak = np.maximum.accumulate(g_sel)
    dd = peak - g_sel
    late = dd[warmup + 1 :]
    late_peak = peak[warmup + 1 :]
    hit = (late >= rel * np.maximum(late_peak, 1e-12)) & (late >= abs_floor)
    return int(hit.any())


def within_condition_auc(
    feature: np.ndarray, label: np.ndarray, condition: np.ndarray
) -> float:
    """Weighted mean of per-condition AUC(feature -> label); weights are the
    number of trajectories in each condition with both classes present."""
    total_w, total = 0.0, 0.0
    for c in np.unique(condition):
        m = condition == c
        if label[m].min() == label[m].max():
            continue  # condition has a single class; no within-condition AUC
        total += auc(feature[m], label[m]) * m.sum()
        total_w += m.sum()
    return float(total / total_w) if total_w else float("nan")


def pooled_auc(feature: np.ndarray, label: np.ndarray) -> float:
    return auc(feature, label)


def permutation_test_within(
    feature: np.ndarray,
    label: np.ndarray,
    condition: np.ndarray,
    n_perm: int = 2000,
    seed: int = 7,
) -> tuple[float, np.ndarray]:
    """One-sided p-value for the observed within-condition AUC against a null
    built by permuting collapse labels *within* each condition."""
    rng = np.random.default_rng(seed)
    observed = within_condition_auc(feature, label, condition)
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = label.copy()
        for c in np.unique(condition):
            m = np.where(condition == c)[0]
            perm[m] = perm[rng.permutation(m)]
        null[b] = within_condition_auc(feature, perm, condition)
    p = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return p, null


def norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF via erf (no scipy dependency)."""
    from math import erf

    vec = np.vectorize(lambda v: 0.5 * (1.0 + erf(v / np.sqrt(2.0))))
    out = vec(x)
    return float(out) if np.isscalar(x) else out
