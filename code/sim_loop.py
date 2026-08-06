"""Synthetic closed-loop sandbox with KNOWN ground truth for the SEL/ESC
decomposition.

This is a method-validation tool, not an empirical claim about real models
or real judges. Two families of sandboxes:

1. stationary_loop -- generation-side feedback deliberately silenced, so the
   candidate pool is drawn from the same distribution every round. Ground
   truth: true ESC = 0 under any selection pressure N (regime "null"), or
   true ESC(t) equal to a closed-form injected pool-bias drift curve
   (regime "drift"). Used by Checks 1 and 3.

2. preference_loop -- a genuinely closed loop: candidates inherit a latent
   "style" trait from the previously selected parent, and the self-judge
   over-rewards that trait (weight b_sp), so self-preference compounds
   across rounds via selection. Ground truth: with b_sp = 0 the matched-
   noise differential ESC_sp should be ~0; with b_sp > 0 it should be > 0.
   Used by Check 4.

Latent-scale convention: quality latents u and judge latents are Gaussian;
observed scores are mapped to [0,1] by the standard normal CDF, matching the
paper's requirement that j and G share a [0,1] correctness scale.
"""
from __future__ import annotations

import numpy as np

from esc_core import decompose, norm_cdf, round_quantities


# ----------------------------------------------------------------------
# Sandbox 1: stationary pool (ground-truth ESC known in closed form)
# ----------------------------------------------------------------------

def stationary_loop(
    n_tasks: int,
    N: int,
    T: int,
    sigma_task: float = 1.0,
    sigma_cand: float = 0.5,
    sigma_judge: float = 0.6,
    drift_per_round: float = 0.0,
    round0_pool: bool = True,
    seed: int = 0,
) -> dict:
    """Judge-and-select loop over a stationary candidate distribution.

    Each round, each task draws N fresh candidates u = a_task + eta
    (a_task ~ N(0, sigma_task^2) fixed per task; eta ~ N(0, sigma_cand^2)).
    True score G = Phi(u). Judge latent = u + drift_per_round * t + eps,
    eps ~ N(0, sigma_judge^2); judge score j = Phi(latent). Selection is
    argmax j. Because nothing is inherited, the pool distribution is
    identical every round and the injected pool-bias drift is available in
    closed form (see injected_drift_curve).

    round0_pool=False emulates the v1 protocol (single unselected draw at
    round 0, best-of-N from round 1 on) to demonstrate the false alarm the
    v2 design removes.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sigma_task, n_tasks)
    rounds = []
    for t in range(T + 1):
        n_cand = 1 if (t == 0 and not round0_pool) else N
        u = a[:, None] + rng.normal(0.0, sigma_cand, (n_tasks, n_cand))
        G = norm_cdf(u)
        lat = u + drift_per_round * t + rng.normal(0.0, sigma_judge, (n_tasks, n_cand))
        j = norm_cdf(lat)
        sel = np.argmax(j, axis=1)
        rounds.append(round_quantities(j, G, sel))
    out = decompose(rounds)
    out["rounds"] = rounds
    return out


def injected_drift_curve(
    T: int,
    drift_per_round: float,
    sigma_task: float = 1.0,
    sigma_cand: float = 0.5,
    sigma_judge: float = 0.6,
) -> np.ndarray:
    """Closed-form expected ESC(t) for stationary_loop.

    E[j] at round t = E[Phi(u + c*t + eps)] = Phi(c*t / s_j) with
    s_j = sqrt(1 + sigma_task^2 + sigma_cand^2 + sigma_judge^2 - 1) ... i.e.
    using E[Phi(a + bZ)] = Phi(a / sqrt(1 + b^2)) with the total latent
    standard deviation; E[G] = Phi(0 / s_g) = 1/2. Hence
    ESC_true(t) = Phi(c t / s_j) - Phi(0) with the bias_pop(0) term equal to
    Phi(0/s_j) - 1/2 = 0 cancelling by symmetry.
    """
    s_j = np.sqrt(1.0 + sigma_task**2 + sigma_cand**2 + sigma_judge**2)
    t = np.arange(T + 1)
    return norm_cdf(drift_per_round * t / s_j) - norm_cdf(np.zeros(T + 1))


# ----------------------------------------------------------------------
# Sandbox 2: heritable style + self-preference (closed feedback loop)
# ----------------------------------------------------------------------

def preference_loop(
    n_tasks: int,
    N: int,
    T: int,
    b_pref: float,
    sigma_judge_self: float = 1.0,
    sigma_judge_strong: float = 0.3,
    tau_matched: float | None = None,
    judge: str = "self",
    heritability: float = 0.9,
    sigma_task: float = 1.0,
    sigma_cand: float = 0.5,
    seed: int = 0,
) -> dict:
    """Closed loop with a heritable, quality-irrelevant style trait s.

    Candidates: u = a_task + eta (quality), s = h * s_parent + sqrt(1-h^2)*xi
    (style, inherited from the selected parent). True score G = Phi(u) never
    depends on s.

    judge="self":    latent = u + b_pref * s + eps_self  (over-rewards style)
    judge="strong":  latent = u + eps_strong             (no style bias)
    judge="strong_matched": strong latent + frozen injected noise tau_matched
                     (the online J_strong~ arm).

    With b_pref > 0 and judge="self", argmax selection favors high-s
    candidates, s_parent ratchets upward, and the judge's style bonus
    inflates j for the whole pool while G is untouched -> bias_pop drifts
    up -> ESC > 0. The matched-noise strong arm shares N, T, and the
    generation process but carries no style bias, so ESC_sp = ESC_self -
    ESC_strong~ isolates the self-preference component.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sigma_task, n_tasks)
    s_parent = np.zeros(n_tasks)
    h = heritability
    rounds = []
    auc_labels, auc_scores = None, None
    for t in range(T + 1):
        u = a[:, None] + rng.normal(0.0, sigma_cand, (n_tasks, N))
        s = h * s_parent[:, None] + np.sqrt(1 - h**2) * rng.normal(
            0.0, 1.0, (n_tasks, N)
        )
        G = norm_cdf(u)
        if judge == "self":
            lat = u + b_pref * s + rng.normal(0.0, sigma_judge_self, (n_tasks, N))
        elif judge == "strong":
            lat = u + rng.normal(0.0, sigma_judge_strong, (n_tasks, N))
        elif judge == "strong_matched":
            if tau_matched is None:
                raise ValueError("tau_matched required for strong_matched")
            lat = (
                u
                + rng.normal(0.0, sigma_judge_strong, (n_tasks, N))
                + rng.normal(0.0, tau_matched, (n_tasks, N))
            )
        else:
            raise ValueError(judge)
        j = norm_cdf(lat)
        sel = np.argmax(j, axis=1)
        rounds.append(round_quantities(j, G, sel))
        if t == 0:
            auc_labels = (u > 0).astype(int).ravel()
            auc_scores = lat.ravel()
        rows = np.arange(n_tasks)
        s_parent = s[rows, sel]
    out = decompose(rounds)
    out["rounds"] = rounds
    out["round0_labels"] = auc_labels
    out["round0_latents"] = auc_scores
    return out


def round0_pool_latents(
    n_tasks: int,
    N: int,
    b_pref: float,
    sigma_judge_self: float = 1.0,
    sigma_judge_strong: float = 0.3,
    sigma_task: float = 1.0,
    sigma_cand: float = 0.5,
    seed: int = 0,
) -> dict:
    """A single round-0 pool scored by both judges (the calibration set for
    the noise-matching protocol): returns labels y = 1{u>0} and both judges'
    latent scores on the same candidates."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sigma_task, n_tasks)
    u = a[:, None] + rng.normal(0.0, sigma_cand, (n_tasks, N))
    s = rng.normal(0.0, 1.0, (n_tasks, N))  # round-0 style: no inheritance yet
    lat_self = u + b_pref * s + rng.normal(0.0, sigma_judge_self, (n_tasks, N))
    lat_strong = u + rng.normal(0.0, sigma_judge_strong, (n_tasks, N))
    return {
        "labels": (u > 0).astype(int).ravel(),
        "lat_self": lat_self.ravel(),
        "lat_strong": lat_strong.ravel(),
    }
