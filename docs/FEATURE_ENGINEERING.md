# Feature Engineering

This document describes every feature used by the tabular models in forecastkit: the
LightGBM model (`lgbm`) and the Kalman to LightGBM hybrid (`kalman_lgbm`). It also
covers the 23 known-future covariates built by the M5 dataset adapter, how features are
anchored at the forecast origin to prevent leakage, and the status of feature importance.

Everything here is read directly from the code. File references:

* `src/forecastkit/features/tabular.py` (lag, rolling, recency features and the feature matrix builder)
* `src/forecastkit/features/calendar.py` (calendar covariates)
* `src/forecastkit/features/intermittency.py` (demand classification, used for slicing, not as a model feature)
* `src/forecastkit/models/lgbm.py` (`LGBMForecaster`, `KalmanLGBMForecaster`)
* `src/forecastkit/datasets/m5.py` (the 23 known-future covariates)
* `src/forecastkit/backtest/context.py` and `src/forecastkit/schema.py` (`ForecastContext`, origin boundary)
* `src/forecastkit/masks.py` (availability and stockout masks)
* `configs/m5.yaml` (model parameters for the real run)
* `tests/test_leakage.py` (what the leakage tests actually check)

## 1. The direct horizon-bucket design

The LightGBM model does direct multi-horizon forecasting. It does not forecast one step
and feed the prediction back in. Instead it trains separate boosters that map features
observed at the anchor day straight to the target h days ahead.

Horizons are grouped into buckets. The bucket size is set by `bucket_size`, which is 7 in
`configs/m5.yaml`. With horizon 28 this gives 4 buckets: days 1-7, 8-14, 15-21, 22-28
(`_buckets(H, size)` in `lgbm.py`). For each bucket the model trains:

* one mean model, objective `poisson` in the M5 config (count demand)
* one quantile model per quantile in `[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]`

So with 4 buckets and 7 quantiles plus one mean per bucket, a full fit trains
4 * (1 + 7) = 32 boosters, each with `n_estimators` = 200 rounds in the M5 config.

Important: the bucket only decides which rows go into which booster. The horizon value `h`
is still passed as a numeric feature (`"h"` is the first column of the feature matrix), and
the seasonal lags and the known-future covariates are computed for the exact target day
`t + h`, not for the bucket as a whole. So within a bucket the model can still tell day 1
from day 7 through the `h` feature and the target-day covariates.

### Anchoring: features at the origin, covariates at the target day

Every training row is a triple `(series s, anchor day t, horizon h)`. The split between
origin-anchored and target-day values is the core of the leakage design.

* Lag, rolling, recency and Kalman-state features are read at the anchor day `t`. In
  training `t` ranges over past anchors with `t + h <= T0`, where `T0 = Th - 1` is the
  origin index. At prediction time every row uses `t = T0` (the origin) and `h = 1..H`.
  See `TabularFeatures.rows`: `self.day_feats[n][s, t]` and the lag columns index `y[s, t-lag]`.
* Known-future covariates are read at the target day `t + h`: `kmat = self.K[s, t + h]`.
  This is allowed because these covariates are known in advance for the whole horizon
  (calendar, events, SNAP, and the M5 weekly prices, discussed in section 5).

The training anchors are strided. `_training_rows` uses `anchor_stride` (default 7) and
`train_days` (364 in the M5 config), walking backward from `T0 - lo` in steps of `stride`
down to `first_anchor = max(T0 - train_days, 56)`. For each horizon `h` it keeps only
anchors with `a + h <= T0`, so the target of every training row lies at or before the
origin and is therefore a value the model is allowed to see.

## 2. Target lags and weekday-aligned seasonal lags

Built in `TabularFeatures.rows`. All lags index the target array `y` (which ends at the
origin) at positions less than or equal to the anchor `t`.

| Feature | What it is | Why it helps | Available at forecast time | Leakage prevention |
|---|---|---|---|---|
| `y_t0`, `y_t1`, `y_t2` | target at `t`, `t-1`, `t-2` | recent level and short momentum | yes, read at the origin | index is `y[s, max(t-lag, 0)]`, always at or before `t`, and `y` physically stops at the origin |
| `slag_1`..`slag_4` | same-weekday lags for the target day | weekly seasonality aligned to the forecast day | yes | index is `t + h - 7*(c + k - 1)` with `c = ceil(h/7)`, chosen so the value lands at or before the origin; values with `j < 0` are set to NaN |
| `slag_mean` | mean of `slag_1..slag_4` | a smoothed weekday-of-week level | yes | derived from the four seasonal lags above |

The weekday alignment works like this. For a target day `t + h`, `c = ceil(h / 7)` is the
number of weeks ahead. The k-th seasonal lag is `t + h - 7*(c + k - 1)`. For a 1 to 7 day
horizon `c = 1`, so the first seasonal lag is `t + h - 7*0 = t + h` shifted by whole weeks;
the construction guarantees each seasonal lag is a same-weekday value at or before the
anchor. This gives the model "what did this series sell on this weekday one, two, three and
four weeks ago", which is the dominant seasonal signal in daily retail demand.

## 3. Rolling mean, std, non-zero rate and mask fraction

Built by `causal_roll(Y, windows)` in `tabular.py`. The windows are fixed in the class:

```
ROLL = (7, 14, 28, 56, 112)
```

For each window `w`, four per-day features are produced (so 5 windows times 4 = 20 rolling
features). The computation uses cumulative sums, so the value at day `t` is a function of
`y[:, :t+1]` only. There are no centered windows and no forward shifts.

| Feature (per window w) | What it is | Why it helps | Leakage prevention |
|---|---|---|---|
| `mean_w` | mean of observed target over the last `w` days up to `t` | demand level at several time scales | cumulative sum differenced at `t` and `max(t-w, 0)`, NaN cells excluded via an observation mask |
| `std_w` | sample std of observed target over the window | volatility, used by quantile models | same cumulative-sum window, uses observed counts only |
| `nzrate_w` | fraction of observed days with a positive sale | intermittency at that scale | counts positive observed days over observed days in the window |
| `maskfrac_w` | fraction of the window that was missing or masked | tells the model how reliable the window is | `1 - observed_count / window_length` |

Windows 7, 14, 28 cover short-term level, 56 and 112 cover the slower trend and the
long-run non-zero rate for intermittent items.

## 4. Recency and intermittency indicators

Built by `recency_features(Y)` in `tabular.py`. Three per-day features, all cumulative or
backward-looking:

| Feature | What it is | Why it helps | Leakage prevention |
|---|---|---|---|
| `days_since_sale` | days since the last positive sale at or before `t` | key signal for intermittent demand (when is the next order likely) | running maximum of positive-sale positions, read at `t` |
| `age` | days since the first observed day of the series | separates new items from established ones | `t - first_observed`, with first observed from the history slice |
| `mean_all` | expanding mean of all observed days up to `t` | long-run base rate | cumulative sum over observed days, read at `t` |

Note: `features/intermittency.py` (`classify_demand`, ADI and CV^2, Syntetos-Boylan class
and volume tier) is NOT a model feature. It is computed once before the first test origin
and is used only to slice results in evaluation. It is listed here so it is not confused
with the per-day intermittency features above.

## 5. Calendar, event and SNAP covariates (known-future)

These come from the known-future covariate array `K`, built by the M5 adapter and by
`calendar_features`. They are read at the target day `t + h`.

Calendar features from `features/calendar.py` (`ALL_CAL`), 13 columns:

* `dow` (day of week as a number)
* `dow_0` .. `dow_6` (one-hot day of week, 7 columns)
* `dom` (day of month)
* `month` (month number)
* `month_sin`, `month_cos` (cyclic month encoding)
* `is_weekend`

Event flags from `datasets/m5.py`, built from `event_type_1` and `event_type_2`, 5 columns:

* `event_religious`, `event_national`, `event_cultural`, `event_sporting`
* `event_any` (true if either event column is non-empty)

SNAP, 1 column:

* `snap` is the SNAP flag of the series' own state. The adapter stacks `snap_CA`, `snap_TX`,
  `snap_WI` and selects the row matching each series' `state_id`.

| Group | What it is | Why it helps | Available at forecast time | Leakage prevention |
|---|---|---|---|---|
| Calendar (13) | deterministic date features | weekday and seasonal effects | yes, dates are always known | derived from `dates` only; broadcast across all series |
| Events (5) | holiday-type flags for the target day | demand spikes and dips around holidays | yes, M5 calendar publishes events ahead | read from `K` at `t + h`; the array is a fixed calendar, not a function of sales |
| SNAP (1) | food-assistance spending-day flag | demand lifts on SNAP days | yes, SNAP schedule is fixed | fixed calendar flag per state |

## 6. Price features (known-future, M5 assumption)

Built in `datasets/m5.py` from the weekly `sell_prices.csv`. Prices are weekly; each day is
mapped to its week via `w_of_day`. Four columns, all read at the target day `t + h`:

| Feature | How it is computed | Why it helps | Leakage note |
|---|---|---|---|
| `log_price` | `log(price_w)`, with missing weeks filled by the series' own mean log price | price level drives demand | uses the week's own price; see assumption below |
| `price_rel_change` | `price_w / prev_week_price - 1`, NaN filled with 0 | week-over-week price moves | uses the current and previous week only |
| `price_vs_mean` | `price_w / past_mean - 1`, NaN filled with 0 | how cheap or dear versus recent norm | `past_mean` is a `shift(1).rolling(26).mean()`, so it uses only weeks strictly before the current week |
| `promo` | `1` where `price_vs_mean < -0.05`, else `0` | a markdown proxy: a price notably below the recent mean | derived from `price_vs_mean`, same backward-looking base |

How these are causal within the price history: `past_mean` and `prev` are built with
`pw.shift(1)` before the rolling mean, so the reference level for a week never includes that
week or any later week. `price_rel_change` compares to the previous week. `log_price` uses
the current week's posted price. The missing-price fill uses the series' own global mean log
price computed across the window; this is a per-series constant used only to fill gaps where
no price was posted, not a rolling feature.

Assumption of the M5 setup: sell prices are published one or more weeks ahead in M5, so the
adapter treats all four price features as known-future covariates and reads them at the
target day. This matches the standard M5 convention. If you run on a dataset where prices
are not known ahead, these should instead be lagged to the last known week at the origin.
As written, the model does read the target week's posted `log_price`, which is only leak-free
under the M5 known-ahead-price assumption.

## 7. Kalman state features in the hybrid

Only in `KalmanLGBMForecaster` (`kalman_lgbm`), added by `extra_features`. A per-SKU Kalman
filter (local level plus stochastic weekly seasonal, on a transformed scale, sqrt by default)
is run on the history, and its filtered state is added as features.

| Feature | What it is | Why it helps | Available at forecast time | Leakage prevention |
|---|---|---|---|---|
| `kf_level` | inverse-transformed filtered local level at `t` | a smooth, adaptive per-series level | yes | filtered state at anchor `t` uses `y <= t` only; stored as a per-day array and read at `t` |
| `kf_sd1` | one-step predictive standard deviation at `t` | local uncertainty, useful for quantile models | yes | same filtered run on the history window |
| `kf_mean_h` | the h-step-ahead seasonal-aware mean `Z T^h a_t` | gives the booster the state-space forecast as an input | yes | state `a_t` is the filtered state at the anchor; `Wh[h] = Z T^h` only projects forward, it reads no future data |

The hybrid feeds local adaptivity (per-series state) into a global cross-series learner (one
booster across all series). The filter is run on a window of `train_days + 150` days ending
at the origin, and `kf_mean_h` is computed in `rows` as `einsum` of the anchor state with the
projection matrix for horizon `h`.

## 8. Static categorical features

Built by `LGBMForecaster._static`. From `meta["static_cat_cols"]` the M5 adapter sets:

* `dept_id`, `cat_id`, `store_id`, `state_id`

These are passed as LightGBM categorical features (their indices are collected in
`self.cat_idx`). `item_id` is appended only when `use_item_id` is true. The default is
`use_item_id = False` (see `DEFAULTS` in `lgbm.py`), and the M5 config does not override it,
so the real `lgbm` and `kalman_lgbm` runs do NOT use `item_id`. When it is enabled,
cold-start items have their item code set to `-1` so the model does not key on an identity it
could not know at launch.

| Feature | What it is | Why it helps | Leakage prevention |
|---|---|---|---|
| `s_dept_id`, `s_cat_id`, `s_store_id`, `s_state_id` | hierarchy codes | cross-series pooling by group | static attributes, constant over time, no temporal leakage |
| `s_item_id` (optional, off by default) | item identity code | per-item effects | cold-start items masked to `-1` |

## 9. How stockout-masked days are treated (use_mask)

Masking happens in two places.

In `build_context` (`backtest/context.py`):

* `y_raw` is the target with NaN only where the series was unavailable (no price entry, or
  leading zeros before the first sale, or cold-start truncation).
* `y` additionally sets NaN where the stockout proxy `censored` is true. The stockout mask
  is recomputed on the history slice at each origin, so a flag on day `t` never depends on
  anything after the origin.

In the model (`lgbm.py`):

* `use_mask` (default true, and true in the M5 config) selects `y` over `y_raw` via
  `ctx.target(self.pp("use_mask"))`. So by default the booster is trained on the masked
  target where stockout days and unavailable days are NaN.
* As targets: in `_training_rows`, rows whose target is NaN are dropped (`keep = ~np.isnan(target)`).
  The comment in the code calls this "weight 0 == dropped". So a day flagged as a stockout is
  never used as a label.
* As features: all rolling and recency features are computed on this same masked `y`. The
  `causal_roll` code treats NaN as missing (it uses an observation mask), so masked days do
  not contribute to the means, and `maskfrac_w` explicitly reports the masked fraction of the
  window. `days_since_sale` ignores masked days because they are not observed positives.

This is the "mark, never impute" rule from `masks.py`: a stockout zero is removed from both
training labels and feature windows rather than being imputed or treated as true zero demand.

## 10. What the leakage tests verify, and their limits

`tests/test_leakage.py` runs a future-perturbation invariance check. It builds a context at a
test origin, then builds a second context after scrambling every target after the origin with
random integers, and asserts that nothing the model can see or predict changes.

What is checked:

* `test_context_is_identical_when_the_future_is_scrambled`: `y`, `y_raw`, `available`,
  `censored`, `known_hist`, `known_fut` are identical before and after the scramble, and
  `y.shape[1] == origin + 1` (the target stops at the origin).
* `test_forecasts_do_not_depend_on_future_targets`: for `seasonal_naive`, `moving_average`,
  `kalman`, `lgbm`, `kalman_lgbm` and `deepar`, the predicted quantiles and mean match to
  `rtol=1e-5, atol=1e-5` before and after the scramble. This covers the two tabular models
  with small, fast parameters.
* `test_tabular_features_at_anchor_ignore_later_values`: directly builds `TabularFeatures`
  on an array and on a copy with everything after `t` randomized, then checks the assembled
  feature matrix `rows(...)` is identical. This is the tightest check on the feature code.
* `test_cold_start_series_expose_only_recent_days`: cold-start series expose at most
  `history_days` observed days.
* `test_stockout_flags_in_context_use_history_only`: the `censored` array is identical when
  the future is scrambled.

Limits of these tests:

* They test invariance to future values, not correctness of the features. A feature could be
  anchored correctly and still be computed wrong; the perturbation test would not catch that.
* They run on a small fixture and with reduced model sizes (for example `n_estimators: 8`),
  so they exercise the code path, not the full-scale fit.
* They check the standard context path. They do not independently re-derive what "known-future"
  should mean for a given dataset. In particular the M5 price features are known-future by
  construction of the adapter (section 6); the test confirms they do not change under a
  target scramble, but it cannot confirm that treating posted prices as known ahead is the
  right assumption for your data. That is a modeling assumption, documented above, not a bug
  the test can find.

## 11. Feature counts summary

Base `lgbm` feature vector (`TabularFeatures.feature_names`, `use_item_id=False`):

| Block | Count | Names |
|---|---|---|
| horizon | 1 | `h` |
| recent target lags | 3 | `y_t0`, `y_t1`, `y_t2` |
| seasonal weekday lags | 5 | `slag_1`..`slag_4`, `slag_mean` |
| rolling (5 windows x 4) | 20 | `mean/std/nzrate/maskfrac_{7,14,28,56,112}` |
| recency | 3 | `days_since_sale`, `age`, `mean_all` |
| known-future covariates | 23 | 13 calendar + 5 events + 1 snap + 4 price |
| static categorical | 4 | `s_dept_id`, `s_cat_id`, `s_store_id`, `s_state_id` |
| total | 59 | |

Hybrid `kalman_lgbm` adds 3 features: `kf_level`, `kf_sd1` (per-day) and `kf_mean_h`
(target-horizon), for a total of 62 features.

The 23 known-future covariates match the M5 adapter log line "N series x T days, 23
covariates".

## 12. Feature importance: status and how to produce it

Status: no feature importance artifact exists and none was produced for this document,
because producing it requires fitting LightGBM, and this workstream does not train models.

The existing run directory `outputs/m5_main/` contains only `config.yaml`, `diag_kalman.pkl`,
`forecasts.parquet`, `series_info.csv`, `timing.csv` and a `report/` folder. There is no
saved LightGBM booster and no importance CSV or PNG. The runner only saves model extras via
`_save_model_extras`, which pickles Kalman diagnostics; it does not persist the LightGBM
boosters, so importance cannot be recovered from disk. The gain-based importance would have
to be computed from a fresh fit.

`LGBMForecaster.feature_importance(top=15)` already implements the right summary: it reads
gain importance from each bucket's mean model, sums the gain per feature name across all
buckets, normalizes by the total, and returns the top features sorted descending. It uses the
mean models only, not the quantile models.

Optional snippet to produce a CSV and PNG for one origin. This fits LightGBM once on the M5
config at the latest test origin, then calls the built-in importance. It is written against
the real APIs in `backtest/runner.py` and `backtest/context.py`. It is NOT executed here.

Runtime estimate: the main run log shows `lgbm` fit at about 97 to 102 seconds per origin
(for example "lgbm origin 2016-01-31 fit 97.6s"). Data loading from cache adds a little more.
Budget roughly two to three minutes for a single-origin fit on a comparable machine. The
hybrid `kalman_lgbm` is a bit slower (about 112 seconds fit in the same log).

```python
# optional: fit lgbm for ONE origin and dump gain importance. NOT run in this workstream.
# requires the real M5 files present and the frozen subset, as configs/m5.yaml expects.
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from forecastkit.backtest.context import build_context
from forecastkit.backtest.runner import prepare
from forecastkit.utils import get_model_class, set_seed

cfg = yaml.safe_load(Path("configs/m5.yaml").read_text())
set_seed(int(cfg.get("seed", 42)))

# dataset, rolling origins and per-series info (slice labels, cold-start flags)
ds, orig, _info = prepare(cfg, use_cache=True)

bt = cfg["backtest"]
H = int(bt["horizon"])
qs = [float(q) for q in bt["quantiles"]]
o = orig["test"][-1]                       # latest test origin
cold = ds.static["cold_start"].to_numpy()

ctx = build_context(ds, o, H, cfg["masks"], cold, ds.meta["cold_days"], qs, ds.static, seed=42 + o)

# fit the SAME lgbm spec as the config (objective poisson, n_estimators 200, bucket_size 7, train_days 364)
mtype, params = cfg["models"]["lgbm"]["type"], dict(cfg["models"]["lgbm"]["params"])
model = get_model_class(mtype)(**params).fit(ctx)

# gain-based importance, summed over the per-bucket mean models and normalised
imp = model.feature_importance(top=10_000)   # large top to keep all features
imp_df = pd.DataFrame(sorted(imp.items(), key=lambda kv: -kv[1]), columns=["feature", "gain_share"])

Path("outputs").mkdir(exist_ok=True)
imp_df.to_csv("outputs/feature_importance.csv", index=False)

top = imp_df.head(25).iloc[::-1]             # top 25 for the chart
Path("docs/figures").mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8, 9))
plt.barh(top["feature"], top["gain_share"])
plt.xlabel("normalised gain importance")
plt.title("LightGBM gain importance (one origin, mean models)")
plt.tight_layout()
plt.savefig("docs/figures/feature_importance.png", dpi=120)
```

Caveats on this importance:

* Gain importance measures how much each feature reduced the training loss when it was split
  on. It is associational, not causal. A high-gain feature is not proven to drive demand.
* It is computed from the mean (Poisson) models only; the quantile models may weight features
  differently.
* SHAP values were not used. SHAP would give per-prediction attributions and a cleaner global
  ranking, but it was out of scope here and would add runtime.
* The ranking is from a single origin. Numbers will shift slightly across origins.
