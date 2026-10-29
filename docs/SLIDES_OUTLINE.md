# Slides outline (final presentation)

File: `slides/final_presentation.pdf`, 6 slides at 16:9. Source of truth for the numbers: `paper/forecastkit_phase1.tex` and `outputs/m5_main/report/*.csv`. Each slide follows one Phase 1 criterion.

| # | Slide | Main takeaway | Evidence |
|---|---|---|---|
| 1 | Literature review (intro) | Orders need quantiles; 59% zeros and 90% intermittent or lumpy series; few studies compare local, tabular, hybrid and global models leak-safely | paper Sections I-II, `docs/LITERATURE_REVIEW.md` |
| 2 | Dataset quality and EDA | 1,200 clean but sparse series; 914/167/89/30 class counts; weekend 1.26-1.31×; SNAP +7.0/+10.7/+5.8%; stockout proxy flags 12.97% (8.69% of test cells); descriptive, not causal | `outputs/eda/*.csv`, paper Section III |
| 3 | Feature engineering | 31 history + 23 known-future + 5 horizon/static = 59 features; hybrid adds 3 Kalman features; price assumption stated | paper Section IV, `docs/FEATURE_ENGINEERING.md` |
| 4 | Theoretical rigor | Kalman, direct LightGBM, hybrid and DeepAR formulations; 124,992 / 114,127 / 9,408 cells; what the leakage test shows; calibration caveat | paper Sections IV-VI and VIII |
| 5 | Model application | DeepAR best all-days WAPE (0.697); hybrid best non-censored (0.660) and cold start (0.760); pinball tie; completed and not-run ablations | `main_all_days.csv`, `main_non_censored.csv`, `cold_start.csv`, `outputs/ablation_*` |
| 6 | Conclusion | The best model depends on the evaluation set; limitations; future work not yet run | paper Sections VII-IX |

Known rounding note: slide 6 gives DeepAR's WAPE gain over the seasonal naive as −9.7%, as the paper does. From the unrounded CSV values (0.69664 / 0.77242) it is −9.8%.

The earlier 12-slide HTML deck is superseded: `archive/superseded/forecastkit_phase1_slides_html_v1.pdf`, source `slides/deck.html`.
