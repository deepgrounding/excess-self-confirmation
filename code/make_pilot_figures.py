"""Regenerates Figures 4-5 (real-model pilot figures) from the stored pilot
summaries in ../data/ -- no MLX, no model calls. Run after pilot_trajectory.py
has produced data/pilot_esc_summary.json (0.8B) and *_4b.json (4B):

    python3 make_pilot_figures.py

Colors follow the series palette: blue #2a78d6, green #1baf7a, amber #eda100,
purple #4a3aa7 (same as make_validation_figures.py).
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "draft", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, GREEN, AMBER, PURPLE = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


def load(tag=""):
    path = os.path.join(DATA, f"pilot_esc_summary{tag}.json")
    with open(path) as f:
        return json.load(f)


def series(summary):
    t = np.arange(len(summary["rounds"]))
    j_sel = np.array([r["J_sel"] for r in summary["rounds"]])
    g_sel = np.array([r["G_sel"] for r in summary["rounds"]])
    return t, j_sel, g_sel


def decomposition_panel(ax, summary, suffix=""):
    t = np.arange(len(summary["raw_SCG"]))
    sel_label = "SEL" + (suffix if suffix else " (winner's curse term)")
    ax.plot(t, summary["raw_SCG"], "d-", color=PURPLE, label=f"raw SCG{suffix}")
    ax.plot(t, summary["SEL"], "s--", color=AMBER, label=sel_label)
    ax.plot(t, summary["ESC"], "^-", color=GREEN, label=f"ESC{suffix}")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(t)
    ax.set_xlabel("round $t$")
    ax.set_ylabel("gap / bias units")
    ax.legend(frameon=False, fontsize=8)


# --- Figure 4: 0.8B pilot ---------------------------------------------------
s08 = load("")
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
ax = axes[0]
t, j_sel, g_sel = series(s08)
ax.plot(t, j_sel, "o-", color=AMBER, label=r"$J_{\rm sel}$ (self-judge)")
ax.plot(t, g_sel, "s-", color=BLUE, label=r"$G_{\rm sel}$ (oracle)")
ax.set_xticks(t)
ax.set_ylim(0.4, 1.05)
ax.set_xlabel("round $t$")
ax.set_ylabel("score")
ax.set_title(f"Qwen3.5-0.8B pilot: selected scores "
             f"($N$={s08['N']}, {s08['n_tasks']} GSM toys)")
ax.legend(frameon=False, fontsize=8, loc="lower left")
decomposition_panel(axes[1], s08)
axes[1].set_title("Decomposition on the real candidate pool")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_pilot_0.8b.png")
print("wrote fig_pilot_0.8b.png")

# --- Figure 5: 4B pilot vs 0.8B ---------------------------------------------
s4b = load("_4b")
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
ax = axes[0]
t, j08, g08 = series(s08)
_, j4b, g4b = series(s4b)
ax.plot(t, j08, "o-", color=AMBER, label=r"$J_{\rm sel}$ 0.8B")
ax.plot(t, g08, "s-", color=BLUE, label=r"$G_{\rm sel}$ 0.8B")
ax.plot(t, j4b, "o--", color=AMBER, label=r"$J_{\rm sel}$ 4B")
ax.plot(t, g4b, "s--", color=BLUE, label=r"$G_{\rm sel}$ 4B")
ax.set_xticks(t)
ax.set_ylim(0.3, 1.05)
ax.set_xlabel("round $t$")
ax.set_ylabel("score")
ax.set_title(f"Selected scores: 0.8B vs 4B pilots "
             f"($N$={s4b['N']}, {s4b['n_tasks']} GSM toys)")
ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
decomposition_panel(axes[1], s4b, suffix=" (4B)")
axes[1].set_title(f"4B decomposition (identity err = "
                  f"{s4b['identity_max_abs_err']:.0f})")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_pilot_4b.png")
print("wrote fig_pilot_4b.png")
