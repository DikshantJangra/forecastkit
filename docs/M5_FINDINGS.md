# Real M5 results: findings and corrections (run 2026-10-29)

Source files: `outputs/m5_main/report/RESULTS.md`, `outputs/ablation_kalman/`, `outputs/ablation_deepar_quick/`.
Setup: 1,200 series (3 stores × 400 items, `configs/subsets/m5_subset.csv`), 1,000 days (2013-08-27 → 2016-05-22),
4 test origins (2016-01-31 … 2016-04-24), H=28, 84 simulated cold-start series, 124,992 evaluated cells,
8.7% of cells flagged by the stockout proxy. Established-series classes: 839 intermittent, 162 lumpy, 87 smooth, 27 erratic.

## Main table (all days, established series)

| model | MAE | WAPE | pinball | cov80 | cov90 | total time (s) |
|---|---|---|---|---|---|---|
| snaive4 (reference) | 0.988 | 0.772 | 0.290 | 0.838 | 0.914 | 0.2 |
| ma28 | 0.924 | 0.722 | 0.274 | 0.811 | 0.891 | 0.2 |
| kalman | 0.924 | 0.723 | 0.274 | 0.831 | 0.914 | 13.7 |
| lgbm | 0.907 | 0.709 | 0.264 | 0.871 | 0.941 | 397 |
| kalman_lgbm | 0.900 | 0.704 | 0.262 | 0.875 | 0.940 | 441 |
| deepar | **0.891** | **0.697** | **0.262** | 0.931 | 0.965 | 296 |
| ens_global_local | 0.899 | 0.703 | 0.262 | 0.827 | 0.919 | n/a |

Non-censored days only: kalman_lgbm is best (WAPE 0.660, pinball 0.269), then lgbm 0.663, deepar 0.665.
Cold start: kalman_lgbm is best (WAPE 0.760); lgbm and deepar tie at 0.773; kalman 0.839.

## Findings

1. DeepAR is best on all days: WAPE 0.697 vs 0.772 for snaive4 (−9.7%), pinball −9.7%. The top four models are within 0.012 pinball of each other.
2. Kalman does no better than a 28-day moving average (WAPE 0.723 vs 0.722), but it runs 29× faster than LightGBM and 22× faster than DeepAR.
3. The Kalman→LightGBM hybrid beats plain LightGBM everywhere and is the best model on non-censored days and on cold start.
4. DeepAR over-covers: cov80 0.931 and cov90 0.965. Its intervals are too wide, so it is under-confident.
5. The equal-weight ensemble does not beat DeepAR on WAPE (0.703 vs 0.697). It does pull coverage back near nominal (cov90 0.919 vs 0.965 for DeepAR) and has the best RMSE (1.782).
6. Calibration at q ≤ 0.25 sits at about 0.57 for every model because about 57% of actuals are zero (discrete mass at 0).

## Ablations run

Kalman (7 variants, `outputs/ablation_kalman/ablation_table.csv`):

| variant | WAPE | pinball | pinball (non-censored) |
|---|---|---|---|
| sqrt_grid (base) | 0.723 | 0.274 | 0.282 |
| sqrt_shared | **0.718** | **0.269** | **0.277** |
| sqrt_fixed | 0.725 | 0.274 | 0.282 |
| log1p_grid | 0.744 | 0.286 | 0.290 (RMSE 20.6: mean back-transform blows up) |
| none_grid | 0.786 | 0.279 | 0.286 |
| window_180 | 0.728 | 0.278 | 0.287 |
| no_masking | 0.701 | 0.270 | 0.291 |

DeepAR (2 variants, compared with the main-run DeepAR):

| variant | WAPE | pinball | pinball (non-censored) | cold-start WAPE | cold-start pinball |
|---|---|---|---|---|---|
| full (negbin, masked) | 0.697 | 0.262 | **0.272** | **0.773** | **0.277** |
| no_loss_masking | 0.677 | 0.256 | 0.276 | 0.808 | 0.293 |
| gaussian_likelihood | 0.792 | 0.278 | 0.285 | 0.888 | 0.294 |

Reading the ablations:
- Turning off stockout masking looks better on all days. That is expected, because the all-days score includes the proxy-flagged zero days, and a model trained on those zeros predicts them. On non-censored days, masking helps (Kalman pinball 0.282 vs 0.291, DeepAR 0.272 vs 0.276), and it helps DeepAR on cold start (pinball 0.277 vs 0.293). Report the non-censored metric as the masking result.
- The negative binomial likelihood beats Gaussian: WAPE −12% (0.697 vs 0.792) and pinball −5.8%.
- The √ transform is needed for Kalman: no transform gives WAPE 0.786 vs 0.723.
- One shared noise ratio beats the per-series grid search, so the per-series grid slightly overfits.

Not run (about 2 hours of CPU): the remaining DeepAR variants (no_item_id, no_price_covariates, context_56/112), the stockout ablation (rule_14d) and the LightGBM ablation.

## Claims in the current PDFs that must change

| Claim (slide/report) | Real M5 value |
|---|---|
| "12,320 evaluations, 800 days" (abstract) | 124,992 cells, 1,000 days |
| Ensemble best, WAPE 0.489 / pinball 0.515 | Synthetic numbers. Real: DeepAR best, WAPE 0.697 / pinball 0.262 |
| Kalman "30× faster than GBDTs" in 2.46 s | 13.7 s, 29× faster than LightGBM (claim holds approximately) |
| "DeepAR out-generalizes local models by 34% on cold start" | 7.9% better than Kalman on WAPE. Not better than LightGBM; the hybrid is best |
| "Removing item embeddings +31.4% cold-start pinball" | Not run |
| "NegBin beats Gaussian by 18.2%" | 12% (WAPE), 5.8% (pinball) |
| "No masking creates −18.7% downward bias" | Not measured. Measured: +1.5% (DeepAR) / +3.2% (Kalman) non-censored pinball without masking |
| "Per-series EM noise estimation beats shared priors" | Code uses grid search, not EM. Shared beats per-series |
| "Poisson cuts intermittent error by 14.2% vs L2" | Not run |
| "Ignoring stockouts: 2.2% penalty" | Not an ablation. Use the masking rows above |
