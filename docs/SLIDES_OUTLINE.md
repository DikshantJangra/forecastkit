# Slides outline (Phase 1 deck)

Source: `slides/deck.html`. Built PDF: `slides/forecastkit_phase1_slides.pdf`, 12 slides at 16:9 (1440 × 810 px). Build with `sh slides/build.sh`, as described in `slides/README.md`.
Source of truth: `paper/forecastkit_phase1.tex` and `outputs/m5_main/report/*.csv`. The three result charts are generated from the CSVs by `slides/build_charts.py`.

The old decks, now in `archive/superseded/`, contain synthetic numbers and unsupported ablation percentages. They are superseded. Do not present them.

| # | Slide | Main takeaway | Evidence |
|---|---|---|---|
| 1 | Title and research question | Local, tabular, hybrid and global models compared under one leak-safe protocol | paper Section I |
| 2 | Why it matters | Inventory needs quantiles; 59.11% of cells are zero; 90% of series are intermittent or lumpy; censoring can only be inferred | `outputs/eda/dataset_summary.csv`, `intermittency_counts.csv` |
| 3 | Dataset and EDA | 1,200 series, 1,000 days, 95.64% available, clean data (Figs 01, 02) | `outputs/eda/`, paper Section III |
| 4 | Seasonality and stockout proxy | Weekday is the strongest signal; SNAP and event effects are associations only; stockout rule `L ≥ max(28, 3·ADI)` (Figs 04, 06) | paper Section III-E |
| 5 | Forecasting approaches | Kalman (grid search, not EM), LightGBM direct, hybrid, DeepAR NegBin, ⅓ ensemble; Croston/SBA/TSB/ETS/ARIMA not implemented | paper Section V |
| 6 | Features and leakage | 59 + 3 features; what the leakage test shows and does not show; price assumption | paper Section IV |
| 7 | Experimental setup | 4 origins, H = 28; 124,992 / 114,127 / 9,408 cells; metric definitions; tuning not run | paper Section VI |
| 8 | Main results | DeepAR lowest WAPE (0.697) and MAE; pinball tied (0.26191 / 0.26198 / 0.26231, std about 0.004); ensemble lowest RMSE (1.782) | `main_all_days.csv`, `origin_stability.csv` |
| 9 | By evaluation setting | Hybrid best on non-censored days (0.660 / 0.269) and cold-start WAPE (0.760); LightGBM and DeepAR 0.773 | `main_non_censored.csv`, `cold_start.csv` |
| 10 | Ablations and calibration | Only completed variants (7 Kalman, 2 DeepAR); masking reading; NegBin vs Gaussian; DeepAR Cov90 0.965 and the discrete-band caveat | `outputs/ablation_*`, `calibration.csv` |
| 11 | Limitations | Proxy labels, narrow scope, no tuning, simulated cold start, price assumption, missing baselines; list of experiments not run | paper Section VIII |
| 12 | Conclusions, future work, references | The best model depends on the evaluation set; 7 key verified references | paper Section IX, `paper/references.bib` |

Slides 3 and 4 use the EDA figures 01, 02, 04 and 06. Figures 03 (demand/CV) and 05 (price) are left to the paper appendix to keep the deck uncluttered.
