"""Draft figures for the merged findings-first paper (2026-07-26 restructure
plan, see AGENT_HANDOFF.md 0.1). Built from data that is already complete
(cell [A] J_self full scale, cell [B] N=1, and the Appendix B calibration
caches) -- no MLX/Ollama calls, no dependency on the still-running J_peer /
J_strong_tilde / J_oracle / J_placebo queue. Regenerate any time with:

    uv run --with matplotlib --with numpy python make_findings_figures.py

Figure F1: ESC(t) and raw-SCG(t) side by side across both domains -- the
    headline "cancellation vs. accumulation" finding (code's near-zero raw
    SCG is a coincidental near-cancellation of a significant negative dSEL
    and a significant positive ESC; math's raw SCG stays small because BOTH
    terms stay small).
Figure F2: True accuracy (G_pop) trajectory under N=1 pure revision (no
    selection pressure -- SEL identically 0 by construction) for both
    domains -- the "two different pathologies" finding: does true quality
    drift under revision alone, independent of any judge.
Figure F3: Round-0 judge-score histogram on math candidates for J_self vs.
    the two independently-calibrated external judges (Gemma-4 12B, Claude
    Haiku 4.5) -- the judge-saturation finding underlying the dual-bound
    report in Appendix B / S:14.4.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esc_core import decompose  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "stage2"
OUT = Path(__file__).resolve().parent.parent / "draft" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GREEN, AMBER, PURPLE = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 300})


def load_rounds(path: Path) -> list[dict]:
    by_round_task: dict[int, dict[str, list[tuple[float, float, bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_round_task[r["round"]][r["task_id"]].append((r["j_score"], r["G"], r["selected"]))
    rounds_sorted = sorted(by_round_task)
    task_ids = sorted({tid for rt in by_round_task.values() for tid in rt})

    def pooled_for(rt, subset):
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

    return [pooled_for(by_round_task[t], task_ids) for t in rounds_sorted]


# --- Figure F1: ESC(t) / raw_SCG(t), both domains --------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.9), sharey=False)
for ax, domain, color in zip(axes, ["math", "code"], [AMBER, BLUE]):
    rounds = load_rounds(DATA / f"A_self_{domain}_4b.jsonl")
    d = decompose(rounds)
    t = np.arange(len(d["ESC"]))
    ax.axhline(0, color="gray", lw=0.7, ls=":")
    ax.plot(t, d["raw_SCG"], "-", color="black", lw=1.6, label="raw SCG(t)")
    ax.plot(t, d["ESC"], "-", color=color, lw=1.8, label="ESC(t)")
    ax.plot(t, d["SEL"] - d["SEL"][0], "--", color=color, lw=1.4, alpha=0.7, label=r"$\Delta$SEL(t)")
    ax.set_title(domain, fontsize=11)
    ax.set_xlabel("round t")
    ax.set_ylabel("value")
    ax.tick_params(labelleft=True)
handles = [
    plt.Line2D([], [], color="black", lw=1.6, label="raw SCG(t)"),
    plt.Line2D([], [], color="gray", lw=1.8, label="ESC(t)"),
    plt.Line2D([], [], color="gray", lw=1.4, ls="--", alpha=0.7, label=r"$\Delta$SEL(t)"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
fig.suptitle(
    "Cell [A] $J_{self}$, full scale: raw SCG = $\\Delta$SEL + ESC\n"
    "code's near-zero raw SCG is a near-cancellation of two significant, opposite-sign terms",
    fontsize=9.5,
    y=1.13,
)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(OUT / "fig_esc_scg_decomposition.png", bbox_inches="tight")
print("wrote fig_esc_scg_decomposition.png")

# --- Figure F2: N=1 true-accuracy trajectory -------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for domain, color in [("math", AMBER), ("code", BLUE)]:
    rounds = load_rounds(DATA / f"B_n1_{domain}_4b_self.jsonl")
    t = np.arange(len(rounds))
    g_pop = np.array([r["G_pop"] for r in rounds])
    ax.plot(t, g_pop, "-o", color=color, ms=3.5, label=domain)
ax.set_xlabel("round t")
ax.set_ylabel(r"true accuracy $G_{pop}$")
ax.set_title(
    "Cell [B] N=1 (pure revision, no selection pressure: SEL $\\equiv$ 0)\n"
    "true-accuracy trajectory across 16 revision rounds",
    fontsize=9.5,
)
ax.legend(frameon=False, fontsize=9)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUT / "fig_n1_accuracy_trajectory.png")
print("wrote fig_n1_accuracy_trajectory.png")

# --- Figure F3: math judge score, split by ground truth --------------------
# The raw marginal score histogram (both judges vs. self, unconditional on
# correctness) does NOT cleanly separate self from external judges -- an
# earlier draft of this figure claimed it did and was wrong; checked against
# the numbers below before shipping. The discriminating fact is conditional
# on ground truth: both external judges are more generous specifically on
# WRONG answers than the self-judge is, which compresses AUC/SEL-relevant
# discrimination even though their scores on CORRECT answers look similar.
g_by_key: dict[str, float] = {}
self_scores, self_g = [], []
with (DATA / "calib_pool_math_4b_self.jsonl").open() as f:
    for line in f:
        r = json.loads(line)
        if r["round"] == 0:
            key = f"{r['task_id']}|0|{r['cand_idx']}"
            g_by_key[key] = r["G"]
            self_scores.append(r["j_score"])
            self_g.append(r["G"])
self_scores, self_g = np.array(self_scores), np.array(self_g)


def round0_scores_with_g(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(cache_path.read_text())
    scores, g = [], []
    for k, v in d.items():
        if k.split("|")[1] == "0" and k in g_by_key:
            scores.append(v)
            g.append(g_by_key[k])
    return np.array(scores), np.array(g)


gemma_scores, gemma_g = round0_scores_with_g(DATA / "noise_calibration_math.strong_cache.json")
haiku_scores, haiku_g = round0_scores_with_g(DATA / "noise_calibration_math_haiku.strong_cache.json")
deepseek_scores, deepseek_g = round0_scores_with_g(DATA / "noise_calibration_math_deepseek.strong_cache.json")

judges = [
    ("$J_{self}$", self_scores, self_g, PURPLE),
    ("$J_{strong}$: Gemma-4 12B", gemma_scores, gemma_g, GREEN),
    ("$J_{strong}$: Claude Haiku 4.5", haiku_scores, haiku_g, AMBER),
    ("$J_{strong}$: DeepSeek V4 Flash", deepseek_scores, deepseek_g, BLUE),
]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
positions = np.arange(len(judges))
width = 0.32
for i, (label, scores, g, color) in enumerate(judges):
    wrong, right = scores[g == 0], scores[g == 1]
    bp = ax.boxplot(
        [wrong, right],
        positions=[i - width / 2, i + width / 2],
        widths=width * 0.9,
        patch_artist=True,
        showfliers=False,
    )
    for patch, hatch in zip(bp["boxes"], ["////", ""]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55 if hatch else 0.9)
        patch.set_hatch(hatch)
    for median in bp["medians"]:
        median.set_color("black")
ax.set_xticks(positions)
ax.set_xticklabels([j[0] for j in judges], fontsize=8.5, rotation=12, ha="right")
ax.set_ylabel("round-0 judge score")
# Headroom above the boxes: every wrong-answer box spans ~0-1 and every
# correct-answer box sits at 1.0, so there is no empty corner inside the
# data range -- the legend has to live in space made for it, not found.
ax.set_ylim(-0.05, 1.28)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
from matplotlib.patches import Patch

ax.legend(
    handles=[
        Patch(facecolor="gray", alpha=0.55, hatch="////", label="wrong answers (G=0)"),
        Patch(facecolor="gray", alpha=0.9, label="correct answers (G=1)"),
    ],
    frameon=False,
    fontsize=8,
    loc="upper center",
    ncol=2,
)
ax.set_title(
    "Math calibration pool (n=336): all three external judges score WRONG answers\n"
    "more generously than the self-judge does -- correct-answer scores look similar",
    fontsize=9.5,
)
fig.tight_layout()
fig.savefig(OUT / "fig_math_judge_saturation.png")
print("wrote fig_math_judge_saturation.png")
for label, scores, g, _ in judges:
    print(
        f"  {label}: wrong mean={scores[g==0].mean():.3f}  "
        f"right mean={scores[g==1].mean():.3f}  gap={scores[g==1].mean()-scores[g==0].mean():.3f}"
    )

# --- Figure F4: seed-13 vs seed-42 replication forest plot ------------------
# Reads the analysis script's own output rather than recomputing, so the
# figure can never disagree with the numbers in the text.
res = {}
for row in json.loads((DATA / "h1_analysis.json").read_text()):
    res[(row["label"], row.get("stat"))] = row

# (display label, seed-13 key, seed-42 key) -- ESC_adj deliberately absent:
# J_placebo was not on the pre-registered second-seed list (see S:9).
ROWS = [
    (r"$\mathrm{ESC}(T)$, code", ("A_self_code", "ESC(T)"), ("A_self_code_seed42", "ESC(T)")),
    (r"$\mathrm{ESC}(T)$, math", ("A_self_math", "ESC(T)"), ("A_self_math_seed42", "ESC(T)")),
    (r"$\mathrm{ESC}_{sp}(T)$, code", ("H2_code", "ESC_sp(T)"), ("H2_code_seed42", "ESC_sp(T)")),
    (r"$\mathrm{ESC}_{sp}(T)$, math (cons)", ("H2_math_conservative", "ESC_sp(T)"), ("H2_math_conservative_seed42", "ESC_sp(T)")),
    (r"$N{=}1$ SCG$(T)$, code", ("B_n1_code", "SCG(T)"), ("B_n1_code_seed42", "SCG(T)")),
    (r"$N{=}1$ SCG$(T)$, math", ("B_n1_math", "SCG(T)"), ("B_n1_math_seed42", "SCG(T)")),
]
fig, ax = plt.subplots(figsize=(7.4, 4.0))
offset = 0.17
for i, (name, k13, k42) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    for key, dy, color, marker in [(k13, +offset, PURPLE, "o"), (k42, -offset, GREEN, "s")]:
        r = res.get(key)
        if r is None:
            continue
        lo, hi = r["ci95"]
        ax.plot([lo, hi], [y + dy, y + dy], "-", color=color, lw=1.6, solid_capstyle="round")
        ax.plot([r["point"]], [y + dy], marker, color=color, ms=5.5, zorder=3)
ax.axvline(0, color="gray", lw=0.9, ls=":")
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([n for n, _, _ in ROWS][::-1], fontsize=9)
ax.set_xlabel("effect size (95% cluster-bootstrap CI)")
ax.set_ylim(-0.6, len(ROWS) - 0.4)
# shade the N=1 rows -- the pair that fails to replicate
ax.axhspan(-0.6, 1.5, color="black", alpha=0.045, zorder=0)
ax.text(
    ax.get_xlim()[1], 0.5, " does not\n replicate", fontsize=7.5, va="center",
    ha="left", color="black", alpha=0.65,
)
ax.legend(
    handles=[
        plt.Line2D([], [], color=PURPLE, marker="o", lw=1.6, ms=5.5, label="seed 13 (primary)"),
        plt.Line2D([], [], color=GREEN, marker="s", lw=1.6, ms=5.5, label="seed 42 (replication)"),
    ],
    frameon=False, fontsize=8.5, loc="upper left",
)
ax.set_title(
    "Second-seed replication on the pre-registered critical cells\n"
    "the three statistics carrying the domain split replicate; the no-selection path does not",
    fontsize=9.5,
)
fig.tight_layout()
fig.savefig(OUT / "fig_seed_replication.png", bbox_inches="tight")
print("wrote fig_seed_replication.png")
