"""Runs the five instrument-validation checks reported in the paper's
"Validation on a Synthetic Sandbox" section. Every number quoted there is
reproducible by `python sim_validate.py` (NumPy only, no GPU / no LLM).

Check 1  ESC identity holds to machine precision.
Check 2  Analytic winner's-curse baseline sigma_j * a_bar_N matches
         empirical SEL for a zero-discrimination judge and upper-bounds it
         for a discriminating judge.
Check 3  Decomposition separates selection from drift: under pure selection
         (true ESC = 0) ESC does not false-alarm at any N while the
         v1-style protocol does; under injected pool-bias drift, measured
         ESC recovers the closed-form injected curve at every N.
Check 4  Noise-matching calibration converges; the matched-noise
         differential ESC_sp reads ~0 when no self-preference is present
         and recovers a positive effect when it is.
Check 5  Within-condition early-warning statistics detect a real
         within-condition signal and are NOT fooled by a condition-label
         confound that inflates the pooled AUC.
"""
from __future__ import annotations

import numpy as np

from esc_core import (
    a_bar,
    auc,
    calibrate_noise,
    collapse_flag,
    early_slope,
    permutation_test_within,
    pooled_auc,
    within_condition_auc,
)
from sim_loop import (
    injected_drift_curve,
    preference_loop,
    round0_pool_latents,
    stationary_loop,
)


def check1_identity() -> None:
    print("=" * 72)
    print("Check 1: ESC identity  ESC(t) = bias_pop(t) - bias_pop(0)")
    worst = 0.0
    for out in (
        stationary_loop(500, 8, 10, drift_per_round=0.03, seed=1),
        preference_loop(300, 6, 10, b_pref=0.4, seed=2),
    ):
        worst = max(worst, float(np.abs(out["ESC"] - out["ESC_identity"]).max()))
    print(f"  max |ESC - identity| across sandboxes: {worst:.3e}")


def check2_analytic_sel() -> None:
    print("=" * 72)
    print("Check 2: analytic winner's-curse baseline sigma_j * a_bar_N")
    print(f"  a_bar constants: N=1: {a_bar(1):.5f}  N=4: {a_bar(4):.5f}  "
          f"N=8: {a_bar(8):.5f}  N=12: {a_bar(12):.5f}")
    rng = np.random.default_rng(11)
    n, sigma = 200_000, 1.0
    print(f"  {'N':>3} {'emp SEL (no discrim.)':>22} {'analytic':>9} "
          f"{'ratio':>6} | {'emp SEL (discrim.)':>19} {'ratio':>6}")
    for N in (4, 8, 12):
        # no discrimination: judge score is pure noise; G iid
        G = rng.normal(0.0, 1.0, (n, N))
        j = rng.normal(0.0, sigma, (n, N))
        rows = np.arange(n)
        sel = np.argmax(j, axis=1)
        sel_emp = (j[rows, sel].mean() - j.mean()) - (G[rows, sel].mean() - G.mean())
        ana = sigma * a_bar(N)
        # with discrimination: j = G + eps, residual sd = sigma
        j2 = G + rng.normal(0.0, sigma, (n, N))
        sel2 = np.argmax(j2, axis=1)
        sel_emp2 = (j2[rows, sel2].mean() - j2.mean()) - (
            G[rows, sel2].mean() - G.mean()
        )
        print(f"  {N:>3} {sel_emp:>22.4f} {ana:>9.4f} {sel_emp/ana:>6.3f} "
              f"| {sel_emp2:>19.4f} {sel_emp2/ana:>6.3f}")


def check3_decomposition() -> None:
    print("=" * 72)
    print("Check 3: decomposition separates pure selection from drift")
    T, n_tasks = 12, 4000
    print("  -- null regime (true ESC = 0, pure winner's curse) --")
    print(f"  {'N':>3} {'SEL level':>10} {'raw_SCG(T)':>11} {'ESC(T)':>9} "
          f"{'v1-protocol SCG(T)':>19}")
    null_rows = []
    for N in (1, 4, 8, 12):
        v2 = stationary_loop(n_tasks, N, T, drift_per_round=0.0, seed=100 + N)
        v1 = stationary_loop(n_tasks, N, T, drift_per_round=0.0,
                             round0_pool=False, seed=100 + N)
        sel_level = float(v2["SEL"][1:].mean())
        row = (N, sel_level, float(v2["raw_SCG"][T]), float(v2["ESC"][T]),
               float(v1["raw_SCG"][T]))
        null_rows.append(row)
        print(f"  {N:>3} {row[1]:>10.4f} {row[2]:>11.4f} {row[3]:>9.4f} "
              f"{row[4]:>19.4f}")
    print("  -- injected drift regime (c = 0.03/round) --")
    truth = injected_drift_curve(T, 0.03)
    print(f"  injected ESC_true(T) = {truth[T]:.4f}")
    print(f"  {'N':>3} {'measured ESC(T)':>16} {'max |ESC - truth| over t':>25}")
    for N in (1, 4, 12):
        out = stationary_loop(n_tasks, N, T, drift_per_round=0.03, seed=200 + N)
        dev = float(np.abs(out["ESC"] - truth).max())
        print(f"  {N:>3} {out['ESC'][T]:>16.4f} {dev:>25.4f}")


def _esc_sp_arm(b_pref: float, seeds: range, n_tasks: int, N: int, T: int
                ) -> tuple[np.ndarray, dict]:
    """Calibrate tau on a round-0 pool, then run self and matched arms across
    seeds; returns ESC_sp(T) per seed plus calibration diagnostics."""
    cal = round0_pool_latents(3000, N, b_pref=b_pref, seed=99)
    target = auc(cal["lat_self"], cal["labels"])
    tau, achieved = calibrate_noise(cal["lat_strong"], cal["labels"], target,
                                    seed=5)
    esc_sp, sel_ratio = [], []
    for s in seeds:
        arm_self = preference_loop(n_tasks, N, T, b_pref=b_pref,
                                   judge="self", seed=1000 + s)
        arm_match = preference_loop(n_tasks, N, T, b_pref=b_pref,
                                    judge="strong_matched", tau_matched=tau,
                                    seed=1000 + s)
        esc_sp.append(arm_self["ESC"][T] - arm_match["ESC"][T])
        sel_ratio.append(arm_match["SEL"][0] / max(arm_self["SEL"][0], 1e-9))
    diag = {"tau": tau, "auc_self": target, "auc_matched": achieved,
            "d_auc": abs(achieved - target),
            "sel_ratio": float(np.mean(sel_ratio))}
    return np.array(esc_sp), diag


def check4_noise_matching() -> None:
    print("=" * 72)
    print("Check 4: noise-matching calibration and the ESC_sp differential")
    seeds, n_tasks, N, T = range(20), 800, 6, 12
    for label, b in (("self-preference present (b=0.4)", 0.4),
                     ("no self-preference (b=0, null)", 0.0)):
        esc_sp, d = _esc_sp_arm(b, seeds, n_tasks, N, T)
        m, sd = esc_sp.mean(), esc_sp.std(ddof=1)
        se = sd / np.sqrt(len(esc_sp))
        print(f"  scenario: {label}")
        print(f"    calibration: tau = {d['tau']:.3f}, AUC self = "
              f"{d['auc_self']:.4f}, AUC matched = {d['auc_matched']:.4f} "
              f"(|dAUC| = {d['d_auc']:.4f}); SEL(0) ratio matched/self = "
              f"{d['sel_ratio']:.3f}")
        print(f"    ESC_sp(T) over {len(esc_sp)} seeds: mean = {m:+.4f}, "
              f"sd = {sd:.4f}, mean/se = {m/se:+.1f}")


def _warning_data(scenario: str, seed: int, K: int = 6, M: int = 16,
                  T: int = 12):
    """Synthetic trajectory units for the early-warning machinery.

    Condition k has mean ESC drift rate mu_k and a collapse propensity
    alpha_k that RISES with mu_k (the confound). Trajectory (k, m) draws a
    within-condition component z; its ESC series has slope mu_k + w*z.
    scenario='informative': collapse hazard depends on z (real
    within-condition signal). scenario='label_only': hazard depends only on
    the condition label -> pooled AUC is inflated by the confound while the
    within-condition AUC should stay at chance.
    Collapse labels are produced by running collapse_flag on a G_sel series
    constructed from the hazard draw, so the pre-registered flag definition
    is exercised too.
    """
    rng = np.random.default_rng(seed)
    mu = np.linspace(0.002, 0.012, K)
    alpha = (mu - mu.mean()) * 400.0 - 0.2
    feat, lab, cond = [], [], []
    t = np.arange(T + 1)
    for k in range(K):
        for _ in range(M):
            z = rng.standard_normal()
            beta = mu[k] + 0.004 * z
            esc_series = beta * t + rng.normal(0.0, 0.004, T + 1)
            if scenario == "informative":
                p = 1.0 / (1.0 + np.exp(-(-0.2 + 2.2 * z)))
            elif scenario == "label_only":
                p = 1.0 / (1.0 + np.exp(-alpha[k]))
            else:
                raise ValueError(scenario)
            collapses = rng.random() < p
            g = 0.55 + 0.005 * t + rng.normal(0.0, 0.005, T + 1)
            if collapses:
                t_break = rng.integers(5, 9)
                depth = rng.uniform(0.18, 0.30)
                g[t_break:] -= depth * (t[t_break:] - t_break) / (T - t_break)
            flag = collapse_flag(g)
            feat.append(early_slope(esc_series))
            lab.append(flag)
            cond.append(k)
    return np.array(feat), np.array(lab), np.array(cond)


def check5_early_warning() -> None:
    print("=" * 72)
    print("Check 5: within-condition early-warning statistics")
    for scenario in ("informative", "label_only"):
        feat, lab, cond = _warning_data(scenario, seed=42)
        pooled = pooled_auc(feat, lab)
        within = within_condition_auc(feat, lab, cond)
        p, null = permutation_test_within(feat, lab, cond, n_perm=2000, seed=7)
        print(f"  scenario: {scenario} (collapse rate {lab.mean():.2f})")
        print(f"    pooled AUC = {pooled:.3f}   within-condition AUC = "
              f"{within:.3f}   permutation p = {p:.4f} "
              f"(null 95th pct = {np.quantile(null, 0.95):.3f})")


if __name__ == "__main__":
    check1_identity()
    check2_analytic_sel()
    check3_decomposition()
    check4_noise_matching()
    check5_early_warning()
    print("=" * 72)
