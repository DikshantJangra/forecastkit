# ML results and ablations

All numbers in this document are read directly from the output files of the `m5_main` run and the two
ablation runs. File paths are cited inline. Where a number differs from an earlier draft or from the
expected reference values I was given, the difference is flagged in the discrepancy list at the end.

## 1. Setup

Verified against `outputs/m5_main/config.yaml`, `outputs/m5_main/report/RESULTS.md`,
`outputs/m5_main/series_info.csv`, and the main table CSVs.

- Series: 1200 total (3 stores CA_1, TX_1, WI_1 times 400 items per store). 84 are held out as
  simulated cold start, leaving 1116 established series. Verified: `series_info.csv` has 1200 rows
  and `cold_start` sums to 84.
- Days: 1000, 2013-08-27 to 2016-05-22 (`config.yaml` `last_days: 1000`, dates from `RESULTS.md`).
- Test origins: 4, at 2016-01-31, 2016-02-28, 2016-03-27, 2016-04-24, with H=28 and step 28, plus one
  earlier validation origin reserved (`config.yaml` backtest block, `RESULTS.md` section 1).
- Horizon: H=28.
- Cold start: 84 series (`cold_start.fraction: 0.07`, 0.07 * 1200 = 84), only the last 28 days
  visible (`history_days: 28`), item identity set to the unknown embedding.
- Evaluated cells, all days, established series: 124992 = 1116 series * 28 horizons * 4 origins.
  Verified: `main_all_days.csv` column `n` = 124992, and 124992 / 28 / 4 = 1116.
- Cold start cells: 9408 = 84 * 28 * 4 (`cold_start.csv` column `n` = 9408).
- Non censored cells: 114127 (`main_non_censored.csv` column `n`), so 124992 - 114127 = 10865 cells
  were flagged, which is 8.69 percent, matching the stated 8.7 percent proxy flag rate.
- Quantiles: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] (`config.yaml`).
- Established series classes (`RESULTS.md` section 1): intermittent 839, lumpy 162, smooth 87,
  erratic 27, new_item 1.

Models and key config (`config.yaml`): kalman (sqrt, grid, window 365), lgbm (poisson, 200 trees,
train_days 364, bucket_size 7), kalman_lgbm (200 trees, train_days 364), deepar (negbin,
context_length 84, hidden 64, 2 layers, 2500 steps, 800 finetune steps, 200 samples, item id on),
seed 42, and an equal weight ensemble `ens_global_local = {kalman, lgbm, deepar}`.

## 2. Main table, all days, established series

Source `outputs/m5_main/report/main_all_days.csv` (full precision shown for the close models).

| model | MAE | RMSE | WAPE | pinball | WQL | cov80 | cov90 | total s |
|---|---|---|---|---|---|---|---|---|
| naive | 1.413 | 2.733 | 1.104 | 0.424 | 0.664 | 0.822 | 0.892 | 0.1 |
| snaive | 1.148 | 2.296 | 0.898 | 0.337 | 0.528 | 0.880 | 0.933 | 0.1 |
| snaive4 (ref) | 0.988 | 1.901 | 0.772 | 0.290 | 0.453 | 0.838 | 0.914 | 0.2 |
| ma28 | 0.924 | 1.864 | 0.722 | 0.274 | 0.429 | 0.811 | 0.891 | 0.2 |
| kalman | 0.924 | 1.917 | 0.723 | 0.274 | 0.428 | 0.831 | 0.914 | 13.7 |
| lgbm | 0.907 | 1.799 | 0.709 | 0.26366 | 0.412 | 0.871 | 0.941 | 397.4 |
| kalman_lgbm | 0.900 | 1.798 | 0.704 | 0.26198 | 0.410 | 0.875 | 0.940 | 441.0 |
| deepar | 0.891 | 1.790 | 0.697 | 0.26191 | 0.409 | 0.931 | 0.965 | 295.5 |
| ens_global_local | 0.899 | 1.782 | 0.703 | 0.26231 | 0.410 | 0.827 | 0.919 | n/a |

Runtime from `outputs/m5_main/report/runtime.csv`: deepar 295.5 s, kalman 13.7 s, kalman_lgbm 441.0
s, lgbm 397.4 s.

Interpretation:

- Best on all days is DeepAR: WAPE 0.697 against 0.772 for snaive4, a 9.7 percent reduction, and the
  lowest MAE (0.891). This is the "best all days" model.
- Pinball is effectively tied at the top. Full precision: deepar 0.261911, kalman_lgbm 0.261980,
  ensemble 0.262307, lgbm 0.263662. DeepAR is lower than kalman_lgbm by 0.00007, which is far inside
  the origin to origin standard deviation of about 0.004 (`origin_stability.csv`). The honest reading
  is that DeepAR, the hybrid, and the ensemble tie on pinball; DeepAR wins on the point metrics (MAE,
  WAPE).
- Kalman does no better than a 28 day moving average on this subset (WAPE 0.7226 vs 0.7222, pinball
  0.27363 vs 0.27436), but it runs far faster: 13.7 s against 397 s for LightGBM (about 29 times
  faster) and 296 s for DeepAR (about 22 times faster).
- DeepAR over covers: cov80 0.931 and cov90 0.965, both above nominal. Its intervals are wide, so it
  is under confident rather than over confident.
- Best RMSE is the ensemble (1.782), narrowly below deepar (1.790); the ensemble also pulls coverage
  back toward nominal (cov90 0.919 vs DeepAR 0.965).

The ensemble is not the best overall. It does not beat DeepAR on WAPE (0.703 vs 0.697) and only ties
on pinball. It is the best on RMSE and has the most nominal coverage, which is a different and
legitimate strength, but it should not be called the best model.

## 3. Non censored days only

Source `outputs/m5_main/report/main_non_censored.csv` (full precision for the close models).

| model | MAE | RMSE | WAPE | pinball | cov80 | cov90 |
|---|---|---|---|---|---|---|
| snaive4 | 1.064 | 1.966 | 0.759 | 0.309 | 0.826 | 0.907 |
| ma28 | 0.998 | 1.930 | 0.712 | 0.293 | 0.796 | 0.883 |
| kalman | 0.961 | 1.841 | 0.686 | 0.282 | 0.825 | 0.911 |
| lgbm | 0.929 | 1.796 | 0.66295 | 0.27071 | 0.871 | 0.941 |
| kalman_lgbm | 0.924 | 1.794 | 0.65981 | 0.26928 | 0.873 | 0.940 |
| deepar | 0.931 | 1.808 | 0.66480 | 0.27226 | 0.927 | 0.962 |
| ens_global_local | 0.931 | 1.775 | 0.66475 | 0.27058 | 0.827 | 0.919 |

Interpretation: on non censored days the hybrid kalman_lgbm is best on both WAPE (0.6598) and pinball
(0.26928), ahead of lgbm (0.6630 / 0.27071) and deepar (0.6648 / 0.27226). The ordering of DeepAR and
the hybrid swaps between the all days view and the non censored view; this is explained by the
masking ablation in section 7.

## 4. Cold start

Source `outputs/m5_main/report/cold_start.csv`. 84 simulated cold start series, 9408 cells.

| model | MAE | RMSE | WAPE | pinball | cov90 | width90 |
|---|---|---|---|---|---|---|
| snaive4 | 1.028 | 2.215 | 0.875 | 0.319 | 0.863 | 2.848 |
| ma28 | 0.920 | 2.135 | 0.783 | 0.292 | 0.853 | 2.876 |
| kalman | 0.985 | 2.495 | 0.839 | 0.323 | 0.874 | 4.468 |
| lgbm | 0.909 | 2.232 | 0.773 | 0.27359 | 0.939 | 3.619 |
| kalman_lgbm | 0.894 | 2.111 | 0.760 | 0.27482 | 0.925 | 3.362 |
| deepar | 0.908 | 2.217 | 0.773 | 0.27748 | 0.963 | 3.553 |
| ens_global_local | 0.920 | 2.221 | 0.783 | 0.283 | 0.894 | 3.880 |

Interpretation: the hybrid is best on cold start WAPE (0.760) and MAE (0.894) and RMSE (2.111). On
cold start pinball lgbm is lowest (0.27359), with kalman_lgbm 0.27482 and deepar 0.27748 close
behind. lgbm and deepar tie on cold start WAPE at 0.773. Plain Kalman is the weakest of the learned
models on cold start (WAPE 0.839), with very wide intervals (width90 4.468) from the diffuse prior.

## 5. Stability across origins

Source `outputs/m5_main/report/origin_stability.csv` (mean plus or minus std over the 4 origins).

| model | MAE | WAPE | pinball |
|---|---|---|---|
| deepar | 0.891 ± 0.012 | 0.697 ± 0.014 | 0.2619 ± 0.0042 |
| kalman_lgbm | 0.900 ± 0.015 | 0.704 ± 0.023 | 0.2620 ± 0.0036 |
| ens_global_local | 0.899 ± 0.016 | 0.703 ± 0.022 | 0.2623 ± 0.0040 |
| lgbm | 0.907 ± 0.019 | 0.710 ± 0.027 | 0.2637 ± 0.0043 |
| kalman | 0.924 ± 0.023 | 0.723 ± 0.028 | 0.2736 ± 0.0060 |
| ma28 | 0.924 ± 0.013 | 0.722 ± 0.009 | 0.2744 ± 0.0033 |
| snaive4 | 0.988 ± 0.014 | 0.773 ± 0.011 | 0.2899 ± 0.0030 |

The pinball gap between the top three models (deepar 0.2619, kalman_lgbm 0.2620, ensemble 0.2623) is
much smaller than each model's own std across origins (about 0.004), which is why the all days pinball
ranking at the top should be read as a tie.

## 6. Significance

Source `outputs/m5_main/report/significance_wape.csv` and `significance_pinball.csv`, reference
snaive4, paired bootstrap over series, 1000 resamples (section 5 of the theory doc). Negative diff
means better than snaive4.

WAPE (`significance_wape.csv`, `metric = WAPE`):

| model | diff | ci_low | ci_high | significant | series_win_rate |
|---|---|---|---|---|---|
| ma28 | -0.0502 | -0.0604 | -0.0400 | True | 0.846 |
| kalman | -0.0498 | -0.0598 | -0.0373 | True | 0.847 |
| lgbm | -0.0631 | -0.0739 | -0.0517 | True | 0.872 |
| kalman_lgbm | -0.0686 | -0.0804 | -0.0558 | True | 0.885 |
| deepar | -0.0758 | -0.0860 | -0.0666 | True | 0.867 |
| ens_global_local | -0.0694 | -0.0788 | -0.0603 | True | 0.875 |

pinball (`significance_pinball.csv`, `metric = pinball`):

| model | diff | ci_low | ci_high | significant | series_win_rate |
|---|---|---|---|---|---|
| ma28 | -0.0155 | -0.0179 | -0.0126 | True | 0.874 |
| kalman | -0.0163 | -0.0200 | -0.0116 | True | 0.874 |
| lgbm | -0.0262 | -0.0294 | -0.0227 | True | 0.906 |
| kalman_lgbm | -0.0279 | -0.0314 | -0.0235 | True | 0.921 |
| deepar | -0.0280 | -0.0310 | -0.0249 | True | 0.892 |
| ens_global_local | -0.0276 | -0.0306 | -0.0245 | True | 0.913 |

What the files support: every learned model beats snaive4 significantly on both WAPE and pinball (the
bootstrap interval excludes zero for all). The bootstrap resamples whole series with replacement and
recomputes the aggregated metric. These tables compare each model to snaive4 only; they do not test
DeepAR against the hybrid, so they do not settle the DeepAR vs hybrid tie. On pinball, deepar
(-0.0280) and kalman_lgbm (-0.0279) have overlapping magnitudes and the hybrid has the higher series
win rate (0.921 vs 0.892), consistent with the hybrid being stronger on the typical series while
DeepAR is slightly stronger in demand weighted aggregate.

## 7. Kalman ablation

Source `outputs/ablation_kalman/ablation_table.csv` (7 variants) and `ablation_cold_start.csv`. Run
with `forecastkit ablate configs/ablations/kalman.yaml`.

| variant | WAPE | pinball | WAPE noncens | pinball noncens | RMSE |
|---|---|---|---|---|---|
| sqrt_grid (base) | 0.7226 | 0.27363 | 0.6860 | 0.28223 | 1.917 |
| sqrt_shared | 0.7176 | 0.26856 | 0.6803 | 0.27716 | 1.793 |
| sqrt_fixed | 0.7249 | 0.27365 | 0.6876 | 0.28165 | 1.849 |
| log1p_grid | 0.7439 | 0.28597 | 0.7014 | 0.28997 | 20.60 |
| none_grid | 0.7862 | 0.27922 | 0.7306 | 0.28572 | 1.824 |
| window_180 | 0.7282 | 0.27765 | 0.6933 | 0.28683 | 1.922 |
| no_masking | 0.7014 | 0.26978 | 0.6935 | 0.29067 | 1.805 |

Interpretation:

- The sqrt transform matters: no transform (none_grid) raises WAPE to 0.786 from 0.723. log1p blows
  up RMSE to 20.6 because the mean back transform `expm1` amplifies large states, which is why RMSE
  is a poor metric for the log1p variant even though its WAPE is only mildly worse.
- One shared noise ratio (sqrt_shared) beats the per series grid on every metric here (WAPE 0.7176 vs
  0.7226, pinball 0.26856 vs 0.27363). The per series grid search slightly overfits on this subset.
- no_masking looks better on all days (WAPE 0.7014, pinball 0.26978) but worse on non censored days
  (pinball 0.29067 vs 0.28223 for the base). See the explanation below.
- On cold start (`ablation_cold_start.csv`) sqrt_shared is best (WAPE 0.8202, pinball 0.3156). The
  no_masking cold start row is identical to sqrt_grid, because the simulated cold start series only
  expose their last 28 days and have no long zero runs to flag, so masking changes nothing for them.

## 8. DeepAR ablation

Source `outputs/ablation_deepar_quick/ablation_table.csv` and `ablation_cold_start.csv`. Run with
`forecastkit ablate configs/ablations/deepar.yaml`. Important: this ablation directory contains only
two variants, `no_loss_masking` and `gaussian_likelihood`. It does not contain a `full` row. The
`full (negbin, masked)` numbers used for comparison are the main run DeepAR, taken from
`outputs/m5_main/report/` (same config and the same seed 42). The ablation runner reruns each variant
from scratch; it did not rerun `full`, so `full` here means the main run, not a fresh ablation run.
The quick variants were trained with the full config (2500 steps, 800 finetune), confirmed by the fit
times in `outputs/ablation_deepar_quick.log` (first origin about 125 s, warm starts about 40 s).

| variant | WAPE | pinball | WAPE noncens | pinball noncens | cov90 | cold WAPE | cold pinball |
|---|---|---|---|---|---|---|---|
| full (main run) | 0.6966 | 0.26191 | 0.6648 | 0.27226 | 0.965 | 0.7729 | 0.2775 |
| no_loss_masking | 0.6770 | 0.25649 | 0.6687 | 0.27604 | 0.960 | 0.8077 | 0.2931 |
| gaussian_likelihood | 0.7923 | 0.27836 | 0.7322 | 0.28528 | 0.922 | 0.8882 | 0.2937 |

`full` row values are from `main_all_days.csv`, `main_non_censored.csv`, and `cold_start.csv`.

Interpretation:

- Negative binomial clearly beats Gaussian. On all days WAPE drops from 0.7923 to 0.6966, a 12.1
  percent reduction, and pinball from 0.27836 to 0.26191, a 5.9 percent reduction. The reason is the
  likelihood match to the data: counts are non negative and over dispersed, and NegBin supports a
  discrete mass at small integers, while the Gaussian puts probability on negative and fractional
  values and has symmetric tails that fit zero heavy counts badly. Both are compared on the same
  completed (non censored and available) values, so the comparison is like for like.
- no_loss_masking improves all days scores (WAPE 0.6770 vs 0.6966, pinball 0.25649 vs 0.26191) but
  worsens non censored pinball (0.27604 vs 0.27226) and cold start (pinball 0.2931 vs 0.2775). See
  the explanation below.

### Why no_loss_masking helps on all days but hurts on real days

The all days score includes the proxy flagged zero days. A model whose loss is not masked learns to
predict those flagged zeros as real demand, so it scores well on exactly the days that were flagged.
The evaluation labels (which cells are censored) are frozen to the base rule across every variant
(`ablation.py` sets `masks.eval_stockout` from the base config), so the two variants are scored on
the same cells; only the training treatment differs. When you remove the flagged days and look at the
non censored subset, which is the fair view of genuine demand, masking wins (DeepAR non censored
pinball 0.27226 masked vs 0.27604 unmasked; Kalman 0.28223 vs 0.29067). So masking is the right
default for forecasting real demand, and the all days gain from turning it off is an artifact of
scoring on the very zero days the unmasked model learned. Report the non censored metric as the
masking result.

## 9. By slice highlights

Source `outputs/m5_main/report/RESULTS.md` section 4 and `by_slice.csv`.

- Intermittency (WAPE): on smooth series the models are far better (deepar 0.459, kalman 0.460,
  kalman_lgbm 0.458, ensemble 0.454) than on intermittent (deepar 0.810, kalman_lgbm 0.827) or lumpy
  (deepar 0.813, kalman_lgbm 0.818) series. On erratic series kalman_lgbm is best (WAPE 0.722). On
  the single new_item series kalman_lgbm has the lowest WAPE (1.055); this slice has one series so
  read it with caution.
- Volume tiers (WAPE): high volume is easiest (deepar 0.601, kalman_lgbm 0.611). Low volume is
  hardest and the hybrid leads there (kalman_lgbm 1.007 vs lgbm 1.045 vs deepar 1.022). Mid volume:
  deepar 0.968, kalman_lgbm 0.976.
- So which model is best depends on the slice: DeepAR leads on most high and intermittent slices in
  aggregate, while the hybrid leads on low volume, erratic, and the non censored and cold start
  views. There is no single best on slices model.

## 10. Limitations

- Stockout labels are a proxy, not ground truth; M5 has no inventory data (`masks.py`, `README.md`).
  Both model treatment and evaluation depend on the chosen rule, and the rule has false positives and
  false negatives (theory doc section 8).
- Subset size: 1200 series from 3 stores, not the full M5. Results may not transfer to the full set.
- Only 4 rolling origins, all in early 2016, so seasonal coverage of the test windows is limited.
- Cold start is simulated: a series reveals only its last 28 days and uses the unknown item
  embedding. This is not the same as a truly new product with no history at all and no hierarchy
  signal yet.
- DeepAR over covers (cov80 0.931, cov90 0.965); its intervals are wider than nominal.
- Hyperparameters are untuned. The tuning code exists (`src/forecastkit/backtest/tuning.py`) but was
  not run: there is no `configs/tuned` directory (verified, only `ablations`, `subsets`,
  `example_csv.yaml`, `m5_smoke.yaml`, `m5.yaml` are present). The reported numbers use the hand set
  defaults in `config.yaml`.
- Single seed (42). No multi seed variance is reported, so the across seed stability is unknown.

## 11. Experiments defined but not run

These variants are defined in `configs/ablations/` but have no output directory. Commands to run each
are listed.

DeepAR (`configs/ablations/deepar.yaml`), not run: `no_item_id`, `no_price_covariates`, `context_56`,
`context_112`. The only DeepAR variants with outputs are `no_loss_masking` and `gaussian_likelihood`.

    forecastkit ablate configs/ablations/deepar.yaml

Stockout sensitivity (`configs/ablations/stockout.yaml`), not run at all (no `outputs/ablation_stockout`):
variants `rule_28d` (base), `rule_14d` (min_run 14), `no_masking` for the set kalman, lgbm, deepar.

    forecastkit ablate configs/ablations/stockout.yaml

LightGBM (`configs/ablations/lgbm.yaml`), not run at all (no `outputs/ablation_lgbm`): `poisson`
(base), `tweedie`, `l2`, `no_price`, `coarse_buckets`, `with_item_id`, `no_masking`.

    forecastkit ablate configs/ablations/lgbm.yaml

Also not produced: LightGBM and hybrid feature importance (the code exists in
`LGBMForecaster.feature_importance` but no importance table is in `outputs/`), hyperparameter tuning,
and any multiple seed runs.

## 12. Discrepancies found

Checked the expected reference values I was given and the existing `docs/M5_FINDINGS.md` against the
output CSVs.

- Expected WAPE/pinball pairs all match the CSVs: snaive4 0.772/0.290, kalman 0.723/0.274 (CSV
  0.7226/0.27363), lgbm 0.709/0.264 (0.7094/0.26366), kalman_lgbm 0.704/0.262 (0.7039/0.26198),
  deepar 0.697/0.262 (0.6966/0.26191), ensemble 0.703/0.262 (0.7030/0.26231). No mismatch.
- Non censored hybrid WAPE 0.660 / pinball 0.269: matches (CSV 0.65981 / 0.26928). No mismatch.
- Cold start hybrid WAPE 0.760: matches (CSV 0.76043). No mismatch.
- DeepAR cov90 0.965: matches (CSV 0.96469). No mismatch.
- Pinball tie: the expected values round deepar, kalman_lgbm, and ensemble all to 0.262. At full
  precision deepar 0.261911 is the lowest, kalman_lgbm 0.261980 next (gap 0.00007), ensemble 0.262307.
  So the all days pinball top three are a statistical tie (gap far below the 0.004 origin std), and
  DeepAR is only nominally first. Flag: any claim that DeepAR is clearly best on pinball overstates
  the result; it is best on WAPE and MAE, tied on pinball.
- `docs/M5_FINDINGS.md` DeepAR ablation table labels a `full (negbin, masked)` row inside the "DeepAR
  (2 variants...)" block. The ablation file `outputs/ablation_deepar_quick/ablation_table.csv` has
  only 2 rows (`no_loss_masking`, `gaussian_likelihood`); `full` is the main run, not an ablation
  output. The M5_FINDINGS text does say "compared with the main-run DeepAR", so this is consistent,
  but the `full` numbers come from `m5_main`, not from the ablation directory. Noted so nobody looks
  for a `full` row in the ablation CSV.
- `docs/M5_FINDINGS.md` DeepAR full non censored pinball is written as 0.272; the CSV value is
  0.27226 (`main_non_censored.csv`), which rounds to 0.272. No mismatch, just rounding.
- `docs/M5_FINDINGS.md` reports kalman no_masking non censored pinball 0.291; CSV is 0.29067 (rounds
  to 0.291). No mismatch.
- `docs/M5_FINDINGS.md` says "DeepAR is best on all days ... pinball -9.7%". The pinball reduction vs
  snaive4 is (0.28990 - 0.26191) / 0.28990 = 9.65 percent, so -9.7 percent is correct for pinball and
  also matches the WAPE reduction of 9.69 percent. No mismatch, but note both the WAPE and pinball
  reductions happen to round to -9.7 percent.
- Kalman cold start no_masking row equals the sqrt_grid row exactly in `ablation_cold_start.csv`.
  This is expected (cold series expose only 28 days, nothing to mask), not an error, but it is worth
  stating so the duplicate is not read as a copy paste bug.
- No `configs/tuned` directory exists, confirming tuning was not run. Consistent with M5_FINDINGS.

No contradictions were found between the generated output CSVs and the expected reference values. The
only substantive caveats are the all days pinball tie (DeepAR nominal, not clear, winner) and the
clarification that the DeepAR ablation `full` row is the main run rather than a fresh ablation.
