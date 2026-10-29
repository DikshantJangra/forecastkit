# Phase 1 (ML) evidence index

Scope: the Phase 1 machine-learning criteria only. All numbers come from the real M5 run of 2026-10-29: 1,200 series, 1,000 days, 4 test origins and H=28. The sources are `outputs/m5_main/report/`, `outputs/ablation_kalman/` and `outputs/ablation_deepar_quick/`.
`outputs/` is git-ignored, so the generated tables exist only on this machine. The commands to regenerate them are listed under "Reproduce".

| Criterion | Current evidence | Supporting artifact | Remaining gaps | Status |
|---|---|---|---|---|
| Literature review | 19 references, each checked against Crossref or the NeurIPS proceedings page. Covers Croston/SBA/TSB, ETS/ARIMA, LightGBM, global vs local models and DeepAR, Kalman/state-space, M5, and scoring rules. Includes a comparison table and a model-selection rationale | `docs/LITERATURE_REVIEW.md` | In the paper, Section 2. Croston, SBA, TSB, ETS and ARIMA are discussed but not implemented as baselines | Done (docs) |
| Dataset quality and EDA | Reproducible script with 11 CSV tables and 6 figures. Quality checks found 0 duplicates, 0 negatives and 0 missing values in the subset. 59.1% zero share, ADI/CV² classes, weekday/month/event/SNAP profiles, price variation, stockout-proxy rate | `scripts/eda_m5.py`, `docs/EDA_FINDINGS.md`, `docs/figures/eda/*.png`, `outputs/eda/*.csv` | In the paper, Section 3 (Figs. 1-4, appendix Figs. 5-6). The descriptive price and SNAP plots do not establish causal effects | Done (docs) |
| Feature engineering | 59 LightGBM features (lags, weekday lags, rolling stats over 7 to 112 days, recency, 23 known-future covariates, 4 static) plus 3 Kalman-state features in the hybrid. Each group lists its source data, rationale, forecast-time availability and leakage control | `docs/FEATURE_ENGINEERING.md`, `src/forecastkit/features/tabular.py`, `tests/test_leakage.py` | **Feature importance not produced**: no LightGBM booster is saved, and computing it needs one retrain of about 100 s (snippet in section 12 of the doc). Price features assume M5's convention that prices are posted ahead | Mostly done |
| Theoretical rigor | Kalman state and observation equations as implemented. Missing-observation update. Direct multi-horizon LightGBM. Pinball/quantile loss. Exact definitions of WAPE, MAE, RMSE, WQL and coverage. Rolling-origin design. Stockout-proxy assumptions. What the leakage tests prove | `docs/THEORY_AND_METHODS.md` | In the paper, Sections 4-6. The old PDF's broken math rendering is superseded by the new LaTeX build | Done (docs) |
| Model application | Real M5 benchmark of 8 models plus the ensemble. Results on all days, non-censored days, cold start, by slice and by origin. Paired bootstrap against snaive4 (all learned models significant). 7 Kalman and 2 DeepAR ablations | `docs/ML_RESULTS_AND_ABLATIONS.md`, `docs/M5_FINDINGS.md`, `outputs/m5_main/report/RESULTS.md` | Hyperparameter tuning not run. Single seed. LightGBM and stockout-rule ablations not run. Remaining DeepAR ablations not run | Mostly done |

## Headline verified results (all days, WAPE / pinball)
snaive4 0.772 / 0.290 · kalman 0.723 / 0.274 · lgbm 0.709 / 0.264 · kalman_lgbm 0.704 / 0.262 · deepar 0.697 / 0.262 · ensemble 0.703 / 0.262.
DeepAR has the best WAPE. On pinball, DeepAR, the hybrid and the ensemble are tied: their gap is 0.0004, far below the roughly 0.004 standard deviation across origins.
The hybrid is best on non-censored days (0.660 / 0.269) and on simulated cold start (WAPE 0.760). DeepAR's Cov90 is 0.965 against a 0.90 target.

## Known integrity issues outside these docs
- The old report and presentation PDFs still contain synthetic-run numbers and unsupported ablation percentages. They were moved, unmodified, to `archive/superseded/` (see its README). The current presentation is `slides/forecastkit_phase1_slides.pdf`.
- The Phase 1 paper has been rewritten from the verified evidence: `paper/forecastkit_phase1.tex`, compiled to `paper/forecastkit_phase1.pdf`. `docs/ACADEMIC_REPORT.md` is now generated from that source and no longer contains synthetic numbers.
- The old PDF report's reference list gives the wrong title and pages for the M5 paper (corrected in the new paper) ("Results, findings, and ways forward", pp. 1402-1423). Crossref records it as "M5 accuracy competition: Results, findings, and conclusions", IJF 38(4), 1346-1364, doi:10.1016/j.ijforecast.2021.11.013.
- Anonymity: the `@DJ` signature, which was in `plan.md` (now `archive/superseded/plan_report_draft.md`), has been removed. The new paper has an anonymous author block and empty PDF author metadata. The git history still records the commit author.

## Optional next steps (not run)
```bash
/tmp/fkvenv/bin/forecastkit ablate configs/ablations/lgbm.yaml       # ~45 min
/tmp/fkvenv/bin/forecastkit ablate configs/ablations/stockout.yaml   # ~40 min
/tmp/fkvenv/bin/forecastkit tune configs/m5.yaml --model lgbm --trials 8   # ~15 min
# feature importance: snippet in docs/FEATURE_ENGINEERING.md section 12 (~100 s)
```

## Reproduce
```bash
/tmp/fkvenv/bin/forecastkit all configs/m5.yaml          # main benchmark (~19 min, already done)
/tmp/fkvenv/bin/forecastkit ablate configs/ablations/kalman.yaml
/tmp/fkvenv/bin/python scripts/eda_m5.py --config configs/m5.yaml --fig-dir docs/figures/eda --out-dir outputs/eda   # ~6 s
/tmp/fkvenv/bin/python -m pytest -q                      # 35 tests
```
The two DeepAR ablation variants were run from a temporary spec, now saved verbatim as `configs/ablations/deepar_quick.yaml`. It selects `no_loss_masking` and `gaussian_likelihood` from `configs/ablations/deepar.yaml`.
