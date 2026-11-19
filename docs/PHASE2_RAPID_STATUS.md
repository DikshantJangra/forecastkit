# Phase 2 Rapid-Mode Status (2026-11-19)

Rapid mode started when the full experiment batch was using too much CPU. All 5 running jobs and the batch
runner were stopped (no background processes remain). Completed work was kept. No new training was run in
rapid mode: everything below was built from forecasts and artifacts already saved by finished runs.

## Complete

| Item | Evidence |
|---|---|
| DeepAR v2: seasonal lags, causal age option, history validation block, early stopping, best checkpoint, AdamW option, seed control, exported samples/curves/weights | `src/forecastkit/models/deepar.py`; Phase 1 params reproduce Phase 1 forecasts exactly (E01) |
| Zero-run persistence model + baselines (series/recent zero rate, Kaplan-Meier renewal) | `src/forecastkit/zerorun/`; E04 |
| Hurdle hybrid `deepar_hurdle` + fusions (`hurdle`, `zi_naive`, `zi_inflate`, `gated`, `none`) + no-DL variant | `src/forecastkit/models/hybrid.py` |
| Series-level paired bootstrap, randomised PIT | `src/forecastkit/evaluation/evaluator.py`, `metrics.py` |
| 30 new tests; full suite 65 passed in 8 s | `tests/test_phase2_*.py` |
| Full-scale runs completed before the stop: DeepAR P1/v2/v2-unmasked and hybrid x 3 seeds; hybrid ablations `zi_naive`, `zi_inflate`, `gated`, `no_dl` (seed 42); E02 validation selection | `outputs/phase2/runs/`, `outputs/phase2/e02_val/` |
| Results package (built once, 2 min, 2 GB peak) | `results/phase2/` |
| IEEE report (compiled once, 6 pages, no overfull boxes) | `paper/forecastkit_phase2.tex/.pdf` |
| Slides (13, built once with headless Chrome, visually inspected) | `slides/phase2_presentation.pdf`, source `slides/phase2/build_deck.py` |
| CI workflow running pytest | `.github/workflows/tests.yml` |
| Literature addendum (20 verified refs), theory, reproduction guide | `docs/LITERATURE_REVIEW.md`, `docs/PHASE2_THEORY.md`, `docs/PHASE2_REPRODUCE.md` |

## Stopped / skipped (resource limits)

| Item | Why it matters | Status |
|---|---|---|
| `hyb_none` (RB reconstruction control) | checks the hurdle code path is neutral | stopped at ~11 min, no output |
| `hyb_no_persist` (hybrid without run features) | covered stand-alone by E04 (`lgbm_zero_no_persistence`) | stopped |
| `hyb_nomask`, `hyb_p1` | hybrid on unmasked / Phase 1 DeepAR | stopped |
| `dar_v2_no_val`, `dar_v2_age`, `dar_v2_age_adamw` | DeepAR leave-one-out on test | stopped / not started (E02 on validation covers selection) |
| Remaining Phase 1 ablation configs (LightGBM, stockout rule, DeepAR item-id/price/context) | completeness | not run |
| Hyper-parameter tuning | - | not run, by design |
