# Phase 1 vs Phase 2 Comparative Audit (saved forecasts only)

Script: `scripts/phase2_comparative_audit.py` (32 s, 1 thread, no training, no backtest). Output tables:
`results/phase2/comparative_audit/{fairness,overall,hybrid_minus_reference,subsets,bootstrap}.csv`.
Phase 2 models are averaged over seeds 42/43/44; Kalman-LightGBM (Phase 1) has one run (seed 42).

## 1. Is the comparison apples-to-apples?

Yes, with one caveat.

| Check | Result |
|---|---|
| Series | all 13 forecast sets cover the same 1,200 series |
| Origins | same 4 test origins (2016-01-31, 02-28, 03-27, 04-24) |
| Horizon | h = 1..28 for all; 134,400 median forecasts per model |
| Forecast statistics | same 7 quantiles + mean for all |
| Evaluation cells and targets | one `Evaluator` builds a single truth cube (targets, availability, hindsight flags) and every model is scored on it: 124,992 warm cells (all days), 114,127 (non-censored) |
| Metric definitions | same code (`Evaluator.agg`) for every model |
| Data, masks, cold-start split | same `configs/m5.yaml` settings (Phase 2 config `extends` it) |
| **Caveat** | Kalman-LightGBM is one seed; Phase 2 numbers are 3-seed means. Phase 2 seed sd is tiny (hybrid pinball sd 0.0003), so this does not change any conclusion, but Kalman-LightGBM's seed variability is unknown. |

## 2. Overall metrics (warm series)

Lower is better for WAPE, MAE, RMSE, pinball. Coverage: closer to nominal (0.80 / 0.90) is better.

**All days (recorded sales, 124,992 cells)**

| Model | Phase | WAPE | MAE | RMSE | Pinball | Cov80 | Cov90 |
|---|---|---:|---:|---:|---:|---:|---:|
| Kalman-LightGBM | P1 | 0.7039 | 0.9004 | 1.7982 | 0.2620 | 0.875 | 0.940 |
| DeepAR P1 | P1 config, 3 seeds | 0.6948 | 0.8888 | 1.7837 | 0.2612 | 0.930 | 0.964 |
| DeepAR v2 | P2 | 0.6913 | 0.8843 | 1.7813 | 0.2606 | 0.930 | 0.964 |
| DeepAR v2 unmasked | P2 | 0.6783 | 0.8677 | 1.7719 | 0.2567 | 0.927 | 0.962 |
| **Hybrid** | P2 | **0.6707** | **0.8579** | **1.7418** | **0.2533** | 0.931 | 0.964 |

**Non-censored days (114,127 cells)**

| Model | Phase | WAPE | MAE | RMSE | Pinball | Cov80 | Cov90 |
|---|---|---:|---:|---:|---:|---:|---:|
| **Kalman-LightGBM** | P1 | **0.6598** | **0.9244** | **1.7938** | **0.2693** | 0.873 | 0.940 |
| DeepAR P1 | P1 config | 0.6644 | 0.9309 | 1.8044 | 0.2721 | 0.926 | 0.962 |
| DeepAR v2 | P2 | 0.6634 | 0.9295 | 1.7992 | 0.2720 | 0.926 | 0.962 |
| DeepAR v2 unmasked | P2 | 0.6709 | 0.9400 | 1.8320 | 0.2764 | 0.921 | 0.959 |
| Hybrid | P2 | 0.6644 | 0.9309 | 1.8040 | 0.2734 | 0.924 | 0.961 |

**Hybrid minus reference (negative = hybrid better; relative in parentheses)**

| Reference | All-days WAPE | All-days pinball | Non-cens. WAPE | Non-cens. pinball |
|---|---|---|---|---|
| Kalman-LightGBM (P1) | -0.0332 (-4.7%) | -0.0087 (-3.3%) | +0.0046 (+0.7%) | +0.0041 (+1.5%) |
| DeepAR P1 | -0.0241 (-3.5%) | -0.0079 (-3.0%) | +0.0000 (0.0%) | +0.0013 (+0.5%) |
| DeepAR v2 | -0.0206 (-3.0%) | -0.0073 (-2.8%) | +0.0010 (+0.2%) | +0.0013 (+0.5%) |
| DeepAR v2 unmasked | -0.0076 (-1.1%) | -0.0034 (-1.3%) | -0.0065 (-1.0%) | -0.0030 (-1.1%) |

Coverage: all DeepAR-based models, the hybrid included, over-cover (0.93 / 0.96). Kalman-LightGBM is closer to the
nominal levels (0.875 / 0.940). The hybrid does **not** improve interval coverage.

## 3. Where does the hybrid win? (hybrid minus Kalman-LightGBM)

"Gap share" = the fraction of the overall all-days pinball (or absolute-error) gap between the two models that
comes from this subset. Shares sum to 100% within a family; a negative share means the hybrid loses there.

| Family | Subset | Cells | Δ pinball | Δ MAE | Δ RMSE | Gap share (pinball) | Gap share (abs. error) |
|---|---|---:|---:|---:|---:|---:|---:|
| Hindsight flag | non-censored | 91.3% | +0.0041 | +0.0065 | +0.0102 | **-43%** | -14% |
| | censored (inside flagged long zero run) | 8.7% | -0.1427 | -0.5567 | -0.9980 | **+143%** | +114% |
| Zero run at origin (causal) | 0 d (sold on origin day) | 47.4% | -0.0009 | -0.0021 | -0.0146 | 5% | 2% |
| | 1-6 d | 33.4% | -0.0008 | -0.0041 | +0.0012 | 3% | 3% |
| | 7-27 d | 11.2% | -0.0109 | -0.0542 | -0.1139 | 14% | 14% |
| | >= 28 d | 7.9% | -0.0853 | -0.4302 | -0.5765 | **78%** | **80%** |
| Outcome (realised y) | y > 0 | 43.5% | +0.0187 | +0.0481 | +0.0374 | -94% | -49% |
| | y = 0 | 56.5% | -0.0297 | -0.1123 | -0.2370 | +194% | +149% |
| Demand class | intermittent | 75.2% | -0.0079 | -0.0358 | -0.0408 | 69% | 63% |
| | lumpy | 14.5% | -0.0077 | -0.0608 | -0.0562 | 13% | 21% |
| | smooth | 7.8% | -0.0098 | -0.0438 | -0.1172 | 9% | 8% |
| | erratic | 2.4% | -0.0340 | -0.1375 | -0.1423 | 10% | 8% |

WAPE is undefined on the censored and y = 0 subsets (zero actual sales) and is not reported there.

Findings, verified from the cells:

* **The gain is driven by series already in a zero run at the origin.** Runs of 7 days or more (19% of cells)
  account for 92% of the pinball gap and 94% of the absolute-error gap. Series that sold recently (0-6 days,
  81% of cells) are a statistical tie with Kalman-LightGBM (see CIs below).
* **The censored cells (8.7%) account for more than the whole gap (143%).** On the other 91% of cells the hybrid
  loses (-43%).
* **Outcome split.** The hybrid wins on zero-sales cells and loses on non-zero cells. This split conditions on the
  realised outcome, so it rewards whichever model predicts more zeros. Use it only to describe the trade-off
  ("more zeros predicted, positives under-forecast more often"); it does not rank the models.
* **Demand class.** The hybrid is ahead of Kalman-LightGBM in every class. Only intermittent is significant on
  pinball. Lumpy and smooth are significant on WAPE only; erratic is not significant.

## 4. Statistical uncertainty (paired bootstrap, whole series resampled, 500 draws, Phase 2 = 3-seed mean)

| Cells | Hybrid - Kalman-LightGBM, pinball [95% CI] | Origins better | Hybrid - DeepAR v2, pinball [95% CI] |
|---|---|---|---|
| All days | **-0.0087** [-0.0136, -0.0051] | 4/4 | **-0.0073** [-0.0099, -0.0046] |
| All days, WAPE | **-0.0332** [-0.0474, -0.0220] | 4/4 | **-0.0206** [-0.0289, -0.0127] |
| Non-censored | **+0.0041** [+0.0014, +0.0065] (hybrid worse) | 0/4 | **+0.0013** [+0.0001, +0.0027] (worse) |
| Non-censored, WAPE | +0.0046 [-0.0003, +0.0089] (n.s.) | 1/4 | +0.0010 [-0.0021, +0.0048] (n.s.) |
| Run at origin 0 d | -0.0009 [-0.0038, +0.0021] (n.s.) | 3/4 | -0.0013 [-0.0026, +0.0001] (n.s.) |
| Run at origin 1-6 d | -0.0008 [-0.0017, +0.0002] (n.s.) | 2/4 | **-0.0013** [-0.0023, -0.0002] |
| Run at origin 7-27 d | **-0.0109** [-0.0181, -0.0051] | 4/4 | **-0.0120** [-0.0215, -0.0045] |
| Run at origin >= 28 d | **-0.0853** [-0.1320, -0.0476] | 4/4 | **-0.0618** [-0.0866, -0.0427] |
| Intermittent | **-0.0079** [-0.0127, -0.0042] | 4/4 | **-0.0046** [-0.0063, -0.0029] |
| Lumpy | -0.0077 [-0.0210, +0.0028] (n.s.) | 4/4 | **-0.0121** [-0.0195, -0.0051] |
| Smooth | -0.0098 [-0.0286, +0.0024] (n.s.) | 3/4 | -0.0101 [-0.0248, +0.0006] (n.s.) |
| Erratic | -0.0340 [-0.0818, +0.0129] (n.s.) | 3/4 | -0.0526 [-0.1233, +0.0035] (n.s.) |

The CIs cover variation across series only, not across time periods: there are just 4 origins.

## 5. Scientific interpretation

**Statement B is supported**: the Phase 2 hybrid improves recorded-sales forecasting overall, but Phase 1 remains
better on some subsets.

* A is false: on non-censored days Kalman-LightGBM is significantly better on pinball. Its intervals are also
  closer to nominal coverage.
* C is false: the all-days gains over Kalman-LightGBM are significant, consistent across all 4 origins, and about
  30 times the hybrid's seed sd.

What Phase 2 proves:

1. The hybrid forecasts **recorded sales** better than every Phase 1 and Phase 2 alternative.
2. The improvement comes from series in an ongoing zero run at the origin. There, predicting "the zeros continue"
   is right most of the time; masked models (Kalman-LightGBM, masked DeepAR) forecast latent demand instead.
3. On series that sold recently, the hybrid ties Kalman-LightGBM.
4. On days outside flagged long zero runs, the Phase 1 hybrid is still the best model. Phase 2 does not improve
   latent-demand forecasting.
5. The one comparison the hybrid wins on both views is against unmasked DeepAR. So its zero modelling is better
   than letting a DL model learn the zeros itself.

Note: the censored cells are defined by the proxy rule, and the evaluation applies it in hindsight. The
"long zero run at origin" slice uses information available at the origin only, and gives the same conclusion,
so the finding does not depend on the hindsight labels.

## 6. Limitations

* Kalman-LightGBM: one seed.
* One retailer, 3 stores, 4 origins in early 2016.
* The stockout proxy defines both the masks and the non-censored view.
* Outcome-conditioned subsets are descriptive only.
* WAPE is undefined on zero-sales subsets.
* No interval recalibration was tried for any model.

## 7. Recommended claim (report and slides)

> "On recorded sales, the Phase 2 DeepAR + zero-run persistence hybrid is the most accurate model: WAPE 0.671 vs
> 0.704 and mean pinball 0.253 vs 0.262 for the Phase 1 Kalman-LightGBM hybrid. Both differences are significant
> under a series-level paired bootstrap and hold on all 4 origins. The gain comes almost entirely from series that
> are already in a zero run of a week or more at the forecast origin (19% of cells, about 92% of the gain). On
> series that sold recently the two hybrids are tied. On days outside long zero runs, the Phase 1 Kalman-LightGBM
> remains better (pinball +0.004 for Phase 2). Phase 2 therefore improves forecasting of recorded sales,
> particularly zero-run persistence. It does not improve forecasting of latent demand."

Avoid: "Phase 2 outperforms Phase 1" without qualification; "better calibrated intervals"; any stockout wording.
