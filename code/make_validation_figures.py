"""Generates the three validation figures from real synthetic-sandbox
computation -- no placeholders, no fabricated numbers.
Run: python3 make_validation_figures.py ; outputs into ../draft/figures/.

Colors follow the series' colorblind-safe palette (shared with the source
survey and the companion error-structure study): blue #2a78d6, green #1baf7a,
amber #eda100, purple #4a3aa7.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from esc_core import auc, calibrate_noise, permutation_test_within, pooled_auc, within_condition_auc
from sim_loop import injected_drift_curve, preference_loop, round0_pool_latents, stationary_loop
from sim_validate import _warning_data

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "draft", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, GREEN, AMBER, PURPLE = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# --- Figure A: decomposition separates pure selection from injected drift ---
T, n_tasks = 12, 4000
Ns = [1, 4, 8, 12]

null_scg_v1, null_scg_v2, null_esc, null_sel = [], [], [], []
for N in Ns:
    v2 = stationary_loop(n_tasks, N, T, drift_per_round=0.0, seed=100 + N)
    v1 = stationary_loop(n_tasks, N, T, drift_per_round=0.0, round0_pool=False,
                         seed=100 + N)
    null_scg_v1.append(v1["raw_SCG"][T])
    null_scg_v2.append(v2["raw_SCG"][T])
    null_esc.append(v2["ESC"][T])
    null_sel.append(v2["SEL"][1:].mean())

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
ax = axes[0]
ax.plot(Ns, null_scg_v1, "o-", color=PURPLE,
        label="raw SCG, v1 protocol (unselected round 0)")
ax.plot(Ns, null_sel, "s--", color=AMBER, label="per-round SEL level (winner's curse)")
ax.plot(Ns, null_scg_v2, "d-", color=BLUE, label="raw SCG, this design")
ax.plot(Ns, null_esc, "^-", color=GREEN, label="ESC, this design")
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.set_xticks(Ns)
ax.set_xlabel("selection pressure $N$ (best-of-$N$)")
ax.set_ylabel("value at $T=12$ (score units)")
ax.set_title("Pure selection, no drift (true ESC = 0)")
ax.legend(frameon=False, fontsize=7.5, loc="center left")

ax = axes[1]
truth = injected_drift_curve(T, 0.03)
t_axis = np.arange(T + 1)
markers = {1: "o", 4: "s", 12: "^"}
shades = {1: "#7fd0ac", 4: GREEN, 12: "#0d7a52"}
for N in (1, 4, 12):
    out = stationary_loop(n_tasks, N, T, drift_per_round=0.03, seed=200 + N)
    ax.plot(t_axis, out["ESC"], markers[N] + "-", color=shades[N], markersize=4,
            label=f"measured ESC, $N$={N}")
ax.plot(t_axis, truth, "k--", lw=1.4, label="injected drift (closed form)")
ax.set_xlabel("round $t$")
ax.set_ylabel("ESC$(t)$")
ax.set_title("Injected pool-bias drift ($c$ = 0.03/round)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_decomposition.png")
print("wrote fig_decomposition.png")

# --- Figure B: noise matching and the self-preference differential ---------
n_tasks_b, N_b, T_b, seeds = 800, 6, 12, range(20)


def esc_curves(b_pref):
    cal = round0_pool_latents(3000, N_b, b_pref=b_pref, seed=99)
    target = auc(cal["lat_self"], cal["labels"])
    tau, _ = calibrate_noise(cal["lat_strong"], cal["labels"], target, seed=5)
    self_c, match_c = [], []
    for s in seeds:
        self_c.append(preference_loop(n_tasks_b, N_b, T_b, b_pref=b_pref,
                                      judge="self", seed=1000 + s)["ESC"])
        match_c.append(preference_loop(n_tasks_b, N_b, T_b, b_pref=b_pref,
                                       judge="strong_matched", tau_matched=tau,
                                       seed=1000 + s)["ESC"])
    return np.array(self_c), np.array(match_c)


self_c, match_c = esc_curves(0.4)
sp_c = self_c - match_c
self_c0, match_c0 = esc_curves(0.0)
sp_c0 = self_c0 - match_c0

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
ax = axes[0]
t_axis = np.arange(T_b + 1)
for arr, color, label in ((self_c, BLUE, r"ESC$_{\rm self}$"),
                          (match_c, AMBER, r"ESC$_{\rm strong\sim}$ (noise-matched)"),
                          (sp_c, GREEN, r"ESC$_{\rm sp}$ = difference")):
    m, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
    ax.plot(t_axis, m, "-", color=color, label=label)
    ax.fill_between(t_axis, m - sd, m + sd, color=color, alpha=0.18, lw=0)
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("round $t$")
ax.set_ylabel("ESC$(t)$")
ax.set_title("Self-preference injected ($b$ = 0.4); mean $\\pm$ sd, 20 seeds")
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axes[1]
rng = np.random.default_rng(3)
for i, (vals, label) in enumerate(((sp_c0[:, T_b], "no self-preference\n($b$ = 0, null)"),
                                   (sp_c[:, T_b], "self-preference\npresent ($b$ = 0.4)"))):
    x = i + rng.normal(0, 0.04, len(vals))
    ax.plot(x, vals, "o", color=GREEN if i else "#8a8a8a", alpha=0.6, markersize=4)
    m = vals.mean()
    ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
    ax.errorbar(i + 0.25, m, yerr=ci, fmt="D", color="black", capsize=4, markersize=5)
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.set_xticks([0.1, 1.1])
ax.set_xticklabels(["no self-preference\n($b$ = 0, null)",
                    "self-preference\npresent ($b$ = 0.4)"])
ax.set_xlim(-0.5, 1.8)
ax.set_ylabel(r"ESC$_{\rm sp}(T)$ per seed")
ax.set_title("Matched-noise differential at $T$ = 12\n(dots: seeds; diamond: mean $\\pm$ 95% CI)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_noise_matching.png")
print("wrote fig_noise_matching.png")

# --- Figure C: within-condition early warning vs. the label confound --------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
ax = axes[0]
labels, pooled_v, within_v, null95 = [], [], [], []
perm_null_info, obs_info = None, None
for scenario, name in (("informative", "within-condition\nsignal present"),
                       ("label_only", "condition-label\nconfound only")):
    feat, lab, cond = _warning_data(scenario, seed=42)
    pooled_v.append(pooled_auc(feat, lab))
    within_v.append(within_condition_auc(feat, lab, cond))
    p, null = permutation_test_within(feat, lab, cond, n_perm=2000, seed=7)
    null95.append(np.quantile(null, 0.95))
    labels.append(name)
    if scenario == "informative":
        perm_null_info, obs_info, p_info = null, within_v[-1], p

x = np.arange(len(labels))
w = 0.32
ax.bar(x - w / 2, pooled_v, w, color=PURPLE, label="pooled AUC (confounded)")
ax.bar(x + w / 2, within_v, w, color=GREEN, label="within-condition AUC")
for xi, n95 in zip(x, null95):
    ax.plot([xi + w / 2 - 0.14, xi + w / 2 + 0.14], [n95, n95], color="black",
            lw=1.2, ls="--")
ax.axhline(0.5, color="gray", lw=0.8, ls=":")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("AUC (early ESC slope $\\to$ late collapse)")
ax.set_title("Dashes: 95th pct of within-condition permutation null")
ax.legend(frameon=False, fontsize=8, loc="upper right")

ax = axes[1]
ax.hist(perm_null_info, bins=36, color="#c9c9c9", edgecolor="white")
ax.axvline(obs_info, color=GREEN, lw=2,
           label=f"observed within-AUC = {obs_info:.3f} (p = {p_info:.4f})")
ax.set_xlabel("within-condition AUC under label permutation")
ax.set_ylabel("count (2,000 permutations)")
ax.set_title("Permutation null, within-condition-signal scenario")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_early_warning.png")
print("wrote fig_early_warning.png")
