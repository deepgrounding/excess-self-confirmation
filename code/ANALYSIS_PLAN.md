# ANALYSIS_PLAN.md — Excess Self-Confirmation (option 2 / ESC)

Frozen with git tag `prereg-opt2-v2`. Analyses outside this list are exploratory.

## Primary hypotheses (one test each; Holm correction)

| ID | Estimand | Cell | Test | Gate before test |
|---|---|---|---|---|
| H1 | ESC(T) > 0 and AUC_ESC > 0 under J_self; independently SCG_{N=1}(T) > 0 | [A], [B] | one-sided cluster bootstrap B=10^4, task-level clusters; both domains must agree in direction | oracle/placebo instrument gate |
| H2 | ESC_sp(T) = ESC_self(T) − ESC_strong~(T) > 0 | [A] paired | cluster bootstrap on paired item-level differences | \|ΔAUC\| ≤ 0.02; SEL(0) ratio ∈ [0.8, 1.25] |
| H3 | within-condition AUC(early_slope_ESC → late_collapse_flag) > 0.5 and beats label baseline + permutation null 95th pct + early_slope_rawSCG | [C], [D] | weighted mean of within-condition AUCs; within-condition label permutation B=10^4 | ≥12 trajectories/condition; collapse rate mid-check |

## Cells (pre-registered)

```text
[A] 4B × 2 domains × {self, peer, strong, strong~, oracle, placebo}
    × N=4 × faithful × T=15 × 100 items
[B] 4B × code × J_self × N ∈ {1,8,12} × faithful × T=15
    (N=1 also math; N∈{8,12} at 50 items)
[C] 4B × 2 domains × {J_self, J_strong~} × N=8 × arbitrable × T=20 × 60 items
[D] 9B × code × J_self × N=6 × {arbitrable, faithful} × T=18 × 40 items
[E] 0.8B × code × {J_self, J_oracle} × N=4 × faithful × T=15 × 100 items
```

Second seed on critical cells: J_self, J_strong~, N=1, H3 main ([C]).

## Thresholds

- Collapse flag: drawdown ≥ 0.30 × peak AND ≥ 0.10 absolute, after warmup k=3.
- Sensitivity (secondary): rel ∈ {0.2,0.3,0.4} × abs ∈ {0.05,0.10,0.15}; k ∈ {2,3,4}.
- Early slope: OLS of ESC(t) on t ∈ {1,2,3}.
- Noise match: \|AUC(J_strong~) − AUC(J_self)\| ≤ 0.02; SEL magnitude ratio ∈ [0.8, 1.25].
- ESC_adj (conditional): report if ESC_placebo significantly > 0.

## H3 escalation (Week-5 mid-check)

If collapse event rate in [C]/[D] < 20%: raise N→12, lengthen T, and/or increase arbitrable bonus weight. Path is pre-declared; using it is not exploratory.

## Lean fallback (budget)

Drop N=12; drop [D] faithful face; second seed only on J_self and [C]. Target ≈280 h vs full ≈440 h.

## Freeze contents

- `esc_manuscript/draft/main.md` (frozen design: hypotheses, method, metrics)
- this `ANALYSIS_PLAN.md`
- Appendix C prompt/rubric skeletons in `main.md`
- `code/esc_core.py`, `sim_loop.py`, `sim_validate.py` (instrument)
- Noise calibration `(τ, λ)`: to be written into `data/noise_calibration.json` after Week-2 calibration, *before* main [A] arm; not required for the plumbing pilot

## Pilot (not a hypothesis test)

`code/pilot_trajectory.py` on Qwen3.5-0.8B-4bit, N=4, short T, 7 GSM toys, J_self only — harness / schema / offline ESC recomputation check. Results do not inform H1–H3.
