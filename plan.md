
# Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models — Project Report

Oct 27, 2026 · **@DJ**

## Abstract

We compare three families of probabilistic demand forecasters on the M5 Walmart data under one leak-proof rolling-origin protocol: a per-SKU Kalman filter, a LightGBM lag-feature model, and a global DeepAR network. All are scored on MAE, RMSE, WAPE and pinball loss at the same 28-day horizon, with stockouts treated as censored observations, simulated cold start, and calibrated uncertainty intervals.

The work is delivered as  **forecastkit** , an open toolkit in which any dataset (via an adapter) and any model (via one `fit`/`predict` interface) can be evaluated by the same pipeline. Headline finding:  **[M5 RESULT: best model by WAPE and pinball, with margin and significance]** .

All numbers marked [M5 RESULT] are filled from `outputs/m5_main/report/RESULTS.md` after the real M5 run; none are typed by hand.

## Problem and data

Store-level retailers must set order quantities before demand is known, so a point forecast is not enough: the replenishment decision needs a calibrated upper quantile of lead-time demand. M5 (Walmart, daily unit sales, 2011–2016) is the benchmark: intermittent, hierarchical, with prices and calendar/SNAP events.

| Item                    | Setting                                                                                                                |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Series                  | ~1,200: 3 stores (CA_1, TX_1, WI_1) × ~400 items, frozen stratified by department × Syntetos–Boylan class (seed 42) |
| Target                  | Daily unit sales                                                                                                       |
| Known-future covariates | Calendar, events, SNAP, sell price                                                                                     |
| Horizon                 | 28 days                                                                                                                |
| Origins                 | 4 test origins, step 28; the last window ends on the last data day; 1 earlier validation origin used only for tuning   |
| Quantiles               | 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95                                                                                  |

Fairness rules: every model sees the identical context object (history up to the origin only), the same series subset, the same origins and the same scoring file. M5 has no inventory records, so stockouts are inferred, not observed (see Limitations).

## forecastkit: the open toolkit

forecastkit separates what changes (data sources, models) from what must not (leakage control, scoring). Datasets enter through adapters into one contract; models see only a leak-proof context; every model writes the same forecast file; the evaluator reads only that file.

[embed: node/0fde62bc-d881]

To add a dataset, write an adapter or a YAML column map for a CSV; to add a model, implement `fit` and `predict` and register it or name it by import path in the config. Neither change touches the evaluator. The same code ran on M5-format data and on an unrelated bakery CSV.

## Methods

**Stockout handling (mark, never impute).** A run of consecutive available zero-sales days of length L is flagged censored when L ≥ max(28, 3·ADI) and the series has at least 10 sales events. Leading zeros are treated as not-yet-launched. Masks are recomputed on the history slice at every origin, so a flag never depends on the future. Flagged days are missing observations for the Kalman filter, zero-weight rows for LightGBM and masked terms in the DeepAR loss. Scoring uses a separate hindsight label, frozen across ablations, and reports both all days and non-censored days.

**Local model: Kalman filter per SKU.** On the square-root scale, a local level plus a stochastic weekly seasonal:

```latex
z_t = \ell_t + \gamma_t + \varepsilon_t,\quad \ell_{t+1} = \ell_t + \xi_t,\quad \gamma_{t+1} = -\sum_{j=0}^{5}\gamma_{t-j} + \omega_t
```

The observation variance is concentrated out; the level and seasonal noise ratios are chosen per series by maximum concentrated likelihood on a small grid. Missing or censored days skip the update. A diffuse prior makes short histories produce wide intervals automatically. Quantiles are back-transformed exactly through the monotone square.

**Tabular baseline: LightGBM.** Direct multi-horizon in four 7-day buckets. A Poisson model gives the mean and one quantile-objective model per level gives the intervals. Features are strictly causal: recent lags, weekday-aligned lags, rolling mean, deviation and non-zero rate at 7 to 112 days, time since last sale, and target-date calendar and price covariates.

**Hybrid.** The Kalman filtered level, one-step predictive deviation and h-step mean are added to LightGBM as features, so the tabular model starts from a state-space estimate.

**Global model: DeepAR.** An LSTM shared across all series with static embeddings (including an unknown-item slot trained by item dropout for cold start), per-window mean scaling and a negative binomial likelihood:

```latex
\log p(y\mid\mu,\alpha)=\log\Gamma(y+r)-\log\Gamma(r)-\log\Gamma(y+1)+r\log\tfrac{r}{r+\mu}+y\log\tfrac{\mu}{r+\mu},\quad r=1/\alpha
```

Censored days are removed from the loss. Quantiles come from 200 ancestral sample paths. Later origins are fine-tuned from the previous weights.

## Evaluation protocol

All models write one long forecast file (model, series, origin, horizon, date, statistic, value); the evaluator reads only that file.

| Metric               | Definition                                                                          |
| -------------------- | ----------------------------------------------------------------------------------- |
| MAE                  | mean absolute error of the median forecast                                          |
| RMSE                 | root mean squared error of the mean forecast                                        |
| WAPE                 | sum of absolute errors divided by sum of actuals (aggregated, not a mean of ratios) |
| Pinball              | quantile loss averaged over the seven levels; weighted version (wQL) also reported  |
| Cov80 / Cov90, width | share of actuals inside the central interval, and its mean width                    |
| Calibration          | P(y < q) and P(y ≤ q) per level, shown as a band because counts have mass at zero  |

Results are also cut by horizon, origin, demand class, volume tier, cold-start status and censored versus non-censored days. Differences against the reference model (4-week seasonal naive) use a paired bootstrap over series with 1,000 resamples. Hyper-parameters are tuned on the validation origin only. Cold start is simulated: for about 7% of series only the last 28 days are visible, and global models receive the unknown-item embedding.

## Results

The real M5 run has not been executed yet; every cell below is a placeholder to copy from `outputs/m5_main/report/RESULTS.md` (main table, all days). Lower is better except coverage.

| Model                     | MAE         | RMSE        | WAPE        | Pinball     | Cov90       | Width90     | Fit+predict time |
| ------------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ---------------- |
| Seasonal naive (4-wk)     | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT]      |
| Kalman (per SKU)          | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT]      |
| LightGBM                  | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT]      |
| Kalman→LightGBM          | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT]      |
| DeepAR                    | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT]      |
| Ensemble (global + local) | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT] | [M5 RESULT]      |

Figures to insert from `report/figures/`, each with a one-line takeaway: fan charts (smooth, intermittent, cold-start series), error versus horizon, calibration band, pinball by quantile, stability across origins, slice heatmap, cold-start comparison, runtime, Kalman state diagnostics.

Questions the discussion must answer from the numbers: (1) Does the global model beat the local filter, and where (by demand class, horizon, volume tier)? (2) Does the hybrid beat plain LightGBM? (3) Which model's 90% interval is closest to 90% coverage? (4) Who wins on cold-start series? (5) Is the gap significant (bootstrap interval excludes zero)? (6) Do conclusions change between all days and non-censored days?

## Ablations and sensitivity

Each ablation is one YAML file run with `forecastkit ablate`; tables report all-days and non-censored-day columns, with the evaluation stockout labels frozen so variants are scored on identical days.

| Ablation | Variants                                                                                                             | Question                                                             |
| -------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| DeepAR   | item embedding on/off, price covariates on/off, NegBin vs Gaussian vs Student-t, context length, loss masking on/off | What drives the global model's accuracy and cold-start behaviour?    |
| Stockout | 28-day rule, 14-day rule, no masking                                                                                 | Does handling censored demand help, and is it sensitive to the rule? |
| Kalman   | sqrt vs raw scale, shared vs per-series noise, window length, masking                                                | Which design choices matter for the local filter?                    |
| LightGBM | Poisson vs Tweedie vs L2, price features, bucket size, item id                                                       | How much of the tabular result is feature engineering?               |

[M5 RESULT: one sentence per ablation naming the winner and the size of the effect.]

## Limitations and threats to validity

* Stockouts are a proxy: M5 has no inventory, so a long zero run may be genuine absence of demand. The rule is stated, tested for sensitivity, and never used to impute values.
* Cold start is simulated by hiding history, not observed; it measures behaviour with little data, not true new-product launches.
* One retailer, one subset of about 1,200 series; results may not transfer to other categories or regions.
* The Kalman model is Gaussian on a transformed scale and is a poor likelihood for very intermittent series.
* Count calibration is evaluated with strict and inclusive bounds; interval coverage uses inclusive bounds, which is generous when zeros are common.
* DeepAR is a compact in-repo implementation, not GluonTS; the CPU path is tested and the GPU path is untested here.
* Each DeepAR number is from [M5 RESULT: number of seeds] seed(s); report the spread.

## Reproducibility and references

The repository pins the subset file, the exact run config (saved with each run), seeds, and 35 tests covering leakage, metrics, splits, masks, the model contract and plug-ins. Reproduce with `forecastkit subset configs/m5.yaml` then `forecastkit all configs/m5.yaml`.

References

* Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems.
* Durbin, J. and Koopman, S. J. (2012). Time Series Analysis by State Space Methods.
* Salinas, D., Flunkert, V., Gasthaus, J., Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting.
* Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. International Journal of Forecasting.
* Syntetos, A. A. and Boylan, J. E. (2005). The accuracy of intermittent demand estimates.
* Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree.
* Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation.
