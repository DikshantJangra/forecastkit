# Phase 2 Progress Log

Newest entries at the bottom. Experiments: `docs/PHASE2_EXPERIMENT_LEDGER.md`. Plan: `docs/PHASE2_IMPLEMENTATION_PLAN.md`.

## 2026-11-19

1. **Stage 0 (audit/baseline).** Re-verified the repo state (HEAD 929f654, clean), Phase 1 tables in
   `outputs/m5_main/report/`, 35 passing tests. Wrote the implementation plan. Copied the Phase 1 tables into the
   tracked `results/phase1/` (they were only in the gitignored `outputs/`).
2. **Stage 1 (DeepAR).** Added to `models/deepar.py`, all off by default: seasonal lags, causal age feature,
   history validation block with best-checkpoint restore / early stopping, AdamW option, per-path NB parameters,
   Rao-Blackwellised P(0), per-origin export of samples, curves and weights. Runner gained a generic
   `model.export(dir, ctx)` hook. `config.py` gained `extends:` for Phase 2 configs.
   - E01: Phase 1 DeepAR reproduced bit-for-bit (max abs diff 0.0).
   - Bug found and fixed: synthetic fixture depended on `PYTHONHASHSEED` (`datasets/synthetic.py`, set iteration).
     No effect on M5.
   - macOS note: torch and lightgbm load different OpenMP runtimes; multi-threaded torch ops after lightgbm is
     imported segfaulted in pytest. Tests pin `torch.set_num_threads(1)` (DeepAR already uses 1 thread).
3. **Stage 2 (DL experiments).** E02 on the validation origin selected `deepar_v2` = validation checkpoint + lags
   7/14/28 (age, AdamW within the tie band). Launched 3-seed test runs.
4. **Stage 3 (zero-run model).** `src/forecastkit/zerorun/`: causal features, LightGBM binary model with history
   validation, baselines (series / recent zero rate, Kaplan-Meier renewal), binary scores. E04 run.
   Finding: beats every non-learned baseline and masked DeepAR's P(0) significantly; ties unmasked DeepAR's P(0);
   explicit run-length features add nothing measurable on top of rolling non-zero rates.
5. **Stage 4 (hybrid).** `models/hybrid.py` `deepar_hurdle`: Rao-Blackwellised hurdle fusion + 4 alternative
   fusions, empirical-size (no-DL) variant; registered model, exports samples, P(0), pi, metadata.
6. **Tests.** +30 tests (`tests/test_phase2_dl.py`, `test_phase2_hybrid.py`, `test_phase2_eval.py`): NB NLL vs
   scipy, sampler moments, masked loss, validation windows, future-perturbation causality for DeepAR, zero
   features and zero model, reproducibility, KM with censoring, hurdle pmf/quantiles/sampler, fusion algebra,
   end-to-end pipeline with artifacts, randomised PIT, paired bootstrap.
7. **Evaluation.** `Evaluator.paired_bootstrap` (series resampling, seed replicates, per-origin diffs);
   `metrics.randomized_pit`, `pit_histogram`. Results package script `scripts/phase2_analysis.py`.
8. **CI.** `.github/workflows/tests.yml` runs pytest on Python 3.10 and 3.12 with CPU torch (synthetic data only).
9. **Hygiene.** Removed the hardcoded `/Users/...` paths from `app.py`.
10. **Docs.** `PHASE2_THEORY.md`, literature addendum (20 verified references), `PHASE2_REPRODUCE.md`.

Remaining items and their status are listed in `docs/PHASE2_RAPID_STATUS.md`.
