# Excess Self-Confirmation

Data, code, and paper source for **"Excess Self-Confirmation Is Domain-Dependent: Evidence from Code and Math."**

When a language model generates candidates, judges them, and revises under its own scores, reported progress can outrun true progress. Most of that gap is not self-deception — it is the optimizer's curse (argmax over a noisy judge inflates the selected score while true quality is unaffected). This repository holds the instrument, the full real-model trajectory library, and the analysis code for a decomposition that isolates the residual — "excess self-confirmation" (ESC) — from that curse, and asks whether the residual is attributable to self-preference or fully explained by generic effects any judge would show.

**Headline result:** a domain split, not a general answer. In code, self-judgment shows excess self-confirmation on four complementary statistics. In math, none of the same five statistics confirms it, and two are significantly the *wrong* sign. An independent seed reproduces the three statistics that carry this split (and sharpens math's); the one statistic that does *not* replicate is the no-selection path, which measures pure revision-conditioning drift. Full results, method, and a complete account of pre-registration deviations are in the paper (`paper/main.md`, `paper/draft_v1.pdf`).

## Layout

```text
paper/
  main.md            Full manuscript source (Markdown + pandoc-style citations)
  references.bib      Bibliography
  draft_v1.pdf         Compiled PDF
  figures/             The 7 figures referenced in the paper
code/
  esc_core.py                  Core SEL/ESC decomposition, analytic baseline, noise calibration, AUC/permutation stats
  sim_loop.py, sim_validate.py, make_validation_figures.py
                               Synthetic-sandbox instrument validation (§5.4) -- no GPU/LLM required
  run_stage2_esc.py            Real-model closed-loop harness (MLX generation + judge conditions)
  judge_utils.py               Judge backends (OpenRouter API, local Ollama)
  calibrate_strong_noise.py    Appendix B noise-matching calibration (offline)
  check_instrument_gate.py     admission-gate statistics
  analyze_cell_A_h1.py         Reproduces every number in the paper's §7 Results (H1, H2, oracle/placebo gates, ESC_adj)
  make_findings_figures.py     Regenerates the real-data results figures (Figures 4-7)
  screen_items.py              Item screening / difficulty binning
  pilot_trajectory.py, make_pilot_figures.py
                               Harness pilot (§10.1)
  verifiers.py, prompts_frozen.py, bench_throughput.py, test_*.py
  ANALYSIS_PLAN.md             Frozen pre-registered analysis plan (git tag prereg-opt2-v2)
data/stage2/
  A_self_{math,code}_4b.jsonl              Cell [A] J_self, full scale
  A_peer_{math,code}_4b.jsonl               Cell [A] J_peer, full scale
  A_strongtilde_code_4b_main.jsonl          Cell [A] J_strong~, code (single calibrated arm)
  A_strongtilde_math_4b_{opt,cons}.jsonl    Cell [A] J_strong~, math (dual-bound arms)
  A_strongtilde_math_4b_deepseek_opt.jsonl  Third-judge cross-check arm
  A_oracle_{math,code}_4b.jsonl             Cell [A] J_oracle, full scale
  A_placebo_{math,code}_4b.jsonl            Cell [A] J_placebo, full scale
  B_n1_{math,code}_4b_self.jsonl            Cell [B] N=1 no-selection path
  A_pilot_{oracle,placebo}_{math,code}.jsonl, gate_smoke_*           Pilot-scale instrument-gate data (§7.1)
  calib_pool_{math,code}_4b_self.jsonl      Round-0 calibration pools (Appendix B input)
  noise_calibration_*.json(.strong_cache.json)  Frozen (tau, lambda) per judge, and cached round-0 judge scores
  screen_{math,code}_4b.json(_raw.jsonl)    Item screening manifests and raw probe records
  *_seed42.jsonl                            Second-seed replication (§7.8): J_self, J_strong~,
                                            and N=1, both domains, seed 42
  A_self_{math,code}_3b_llama.jsonl         Second model family (§7.8): Llama-3.2-3B-Instruct,
                                            J_self only, on its own screened pools
  screen_{math,code}_3b_llama.json(_raw.jsonl)  Llama-specific item screening (not Qwen's pool --
                                            re-screened to avoid a floor/ceiling confound)
  h1_analysis.json                          Machine-readable dump of every §7 statistic
```

## Reproducing the results

No GPU or LLM access needed for the instrument validation or the real-model statistics -- both recompute from the stored trajectory library:

```bash
# Synthetic-sandbox instrument validation (§5.4)
cd code && python3 sim_validate.py

# Every number in §7 (Results), read from the trajectory library in ../data/stage2/
cd code && python3 analyze_cell_A_h1.py

# Regenerate the real-data figures (Figures 4-7)
cd code && python3 make_findings_figures.py
```

The replication arms in §7.8 were produced by `run_second_seed_queue.sh` (seed 42, pre-registered critical cells) and `run_second_model_family_queue.sh` (Llama-3.2-3B, re-screen then `J_self`); both are included alongside the primary queue scripts.

Re-running the real-model harness itself (`run_stage2_esc.py`) requires MLX on Apple Silicon, the cached `mlx-community/Qwen3.5-4B-4bit` checkpoint (or `mlx-community/Llama-3.2-3B-Instruct-4bit` for the second-family arm), and — for the `J_strong`/`J_strong~` judge conditions — either a local Ollama server (`gemma4:12b`, the default) or an OpenRouter API key (`OPENROUTER_API_KEY`) for the Claude Haiku / DeepSeek cross-checks.

## Pre-registration

Design, hypotheses, metrics, synthetic validation, and the statistical analysis plan were frozen under git tag `prereg-opt2-v2` (commit `95b0fb4`, 2026-07-12) before any real-model data collection began. The noise-matching calibration, which cannot exist before round-0 data is collected, was frozen at commit `f7faafb` (2026-07-25) under tag `prereg-opt2-v2-noise` -- five days before the first `J_strong~` arm it governs was run (`464f44e`, 2026-07-30).

Both freeze points live in the authors' development repository, which also holds unrelated manuscripts; **this repository is a snapshot of that work, not its commit history**, so the tags are not visible here. The dated commits are available to reviewers and editors on request. The paper's Section 8 ("Preregistration and Deviations") discloses every post-freeze amendment in full, in the order it occurred.

## License

Code: MIT (see `LICENSE`). Data (`data/`) and paper text (`paper/`) are released under CC BY 4.0 -- reuse with attribution.

## Citation

If you use this data or code, please cite the paper (see `paper/main.md` for the current author list and details; BibTeX entry to follow arXiv publication).
