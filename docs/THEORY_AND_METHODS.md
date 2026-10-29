# Theory and methods

This document explains the forecasting problem, the three model families, the exact metric
definitions, and the evaluation protocol as they are written in the code. Every equation matches a
specific source file, and every symbol is named. Where the code and a claim could differ, the code
wins. This file does not claim any property that is not either in the code or checked by a test.

## 1. The inventory demand problem and why quantiles matter

We forecast daily unit sales for many retail items (series). Let `y_{i,t}` be the sales of item `i`
on day `t`. Sales are non negative counts, and most series are intermittent: on the M5 subset about
57 percent of actual values are zero (`outputs/m5_main/report/RESULTS.md`, finding 6).

A point forecast (one number per day) is not enough for inventory decisions. The cost of ordering
too little (a lost sale) is usually not equal to the cost of ordering too much (holding stock). The
classic newsvendor result says the order quantity that minimizes expected cost is a quantile of the
demand distribution, not the mean. The target quantile is the critical fractile

    q* = Cu / (Cu + Co)

where `Cu` is the per unit underage (stockout) cost and `Co` is the per unit overage (holding) cost.
If stockouts are five times as costly as holding, `q* = 5/6 ≈ 0.83`, so you stock to roughly the
0.83 quantile of demand over the lead time, not the average. This is why the toolkit produces a set
of quantiles `[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]` (`outputs/m5_main/config.yaml`) and scores
them with pinball loss, so every model is judged on the full predictive distribution, not just a
central estimate.

Decision link as written in `docs/METHODS.md`: the 0.9 quantile of demand over the replenishment
lead time is the order up to level for about a 90 percent cycle service level, and a measured Cov90
near 0.90 means that level delivers the service it promises.

## 2. Local model: Kalman filter per item

Source: `src/forecastkit/models/kalman.py`.

### What it models

Each item gets its own small state space model on a transformed scale. The transform `g` is sqrt by
default (`TRANSFORMS` in `kalman.py`; options are `sqrt`, `log1p`, `none`). Writing `z_t = g(y_t)`:

    z_t       = level_t + gamma_t + eps_t          eps_t ~ N(0, sigma^2)        (observation)
    level_{t+1} = level_t + xi_t                   xi_t  ~ N(0, q_level * sigma^2)   (local level)
    gamma_{t+1} = -(gamma_t + gamma_{t-1} + ... + gamma_{t-m+2}) + w_t   w_t ~ N(0, q_seas * sigma^2)

Symbols:

- `level_t` is the slow moving baseline of (transformed) demand. It is a random walk, so it adapts
  over time instead of being fixed.
- `gamma_t` is a stochastic weekly seasonal term. `m = 7` is the season length (`season`, default
  from the dataset). This is the "dummy" seasonal form: the seven daily effects are constrained so
  they sum to roughly zero and drift slowly, which captures day of week shape without a fixed
  pattern.
- `eps_t` is observation noise with variance `sigma^2`.
- `q_level` and `q_seas` are noise ratios: how fast the level and the seasonal part are allowed to
  move, measured in units of `sigma^2` (the code fixes `R = 1` and scales everything by `sigma^2`).
- The hidden state vector is `alpha_t = (level_t, gamma_t, gamma_{t-1}, ..., gamma_{t-m+2})`. In
  code, `StateSpace` builds the transition matrix `T` and the selection vector `Z = (1, 1, 0, ...)`
  so that the observed value picks off `level + gamma`.

### The filter recursions

The batched filter `kf_filter` runs all series at once. For each day `t` with observation `y_t`
(`v` is the innovation, `F` its variance):

    F_t  = Z P_{t|t-1} Z' + 1
    v_t  = z_t - Z alpha_{t|t-1}
    gain = P_{t|t-1} Z' / F_t
    alpha_{t|t} = alpha_{t|t-1} + gain * v_t
    P_{t|t}     = P_{t|t-1} - (P Z)(P Z)' / F_t
    alpha_{t+1|t} = T alpha_{t|t}
    P_{t+1|t}     = T P_{t|t} T' + Q

### How missing and censored days are handled

This is the key leak and robustness point. In `kf_filter` the mask is `obs = ~np.isnan(y)`. When a
day is NaN the update is skipped: `a` and `P` are left at their predicted values
(`a = np.where(obs[:, None], a + gain*v, a)` and the matching line for `P`). The state still moves
forward in time through `T` and `Q`, but it is not corrected by that day. Days that are unavailable,
before the first sale, or flagged by the stockout proxy are set to NaN upstream (section 7), so the
filter simply treats them as gaps. Nothing is imputed.

### Picking the noise ratios

`sigma^2` is concentrated out of the likelihood: `sigma2 = max(sum(v^2/F) / n, 1e-3)` where `n`
counts the observed days after a burn in (`burn_in = 14`, so the first 14 observed days of each
series do not enter the likelihood while the diffuse prior settles). The concentrated log likelihood
is

    ll = -0.5 * (n * log(sigma2) + sum(log F_t))

`choose_params` selects `(q_level, q_seas)` per series by maximizing `ll` over a small grid
(`param_mode = "grid"`, the default and the main run setting), or picks one shared pair that
maximizes the summed likelihood across all series (`param_mode = "shared"`), or uses a fixed pair
(`param_mode = "fixed"`, default `(1e-2, 1e-3)`). Series with fewer than `min_obs = 28` observations
fall back to the fixed pair, because there is not enough data to estimate the ratios. This is a
speed versus adaptivity trade off; the ablation in section 8 of the results shows `shared` slightly
beats `grid` here, meaning the per series grid search overfits a little on this subset.

### Diffuse initialization and cold start

Initial state: `level` is the mean of the first up to 14 observed values; `P0 = kappa * I` with
`kappa = 1e3`. A large `kappa` is a diffuse (very uncertain) prior, so a series with little history
automatically produces wide intervals. That is the local model's built in cold start behavior: it
admits it does not know much yet.

### From state to quantiles

`forecast` propagates the state forward `H` steps to get a predictive mean `mu_h = Z alpha_{T0+h|T0}`
and variance `V_h = sigma2 * (Z P_{T0+h|T0} Z' + 1)` on the transformed scale. Quantiles come from
the normal on that scale and are mapped back through `g^{-1}`:

    q_hat_h(p) = g^{-1}( mu_h + Phi^{-1}(p) * sqrt(V_h) ),  clipped at 0

Because `g` is monotone, mapping the normal quantiles through `g^{-1}` gives the exact quantiles on
the original scale. The reported mean uses the back transform of the normal: for sqrt it is
`mu^2 + V` (the delta correction), for log1p `expm1(mu + 0.5 V)`. Note the Kalman likelihood is
Gaussian on a transformed scale; on very intermittent series this is a poor fit, which is a
documented weakness, not a hidden one (`README.md`, honest limitations).

## 3. Tabular model: LightGBM, direct by horizon bucket

Source: `src/forecastkit/models/lgbm.py`, features in `src/forecastkit/features/tabular.py`.

### Direct multi horizon

Instead of forecasting one step and feeding the forecast back in, LightGBM trains separate models
for each horizon bucket. `_buckets(H, size)` with `bucket_size = 7` gives the buckets 1 to 7, 8 to
14, 15 to 21, 22 to 28 for `H = 28`. For each bucket the training rows are tuples (series `i`, anchor
day `t`, horizon `h`) with target `y_{i, t+h}`, drawn from many anchor days spaced by `anchor_stride`
(7 days) over the last `train_days` (364). The constraint `t + h <= T0` is enforced so a row never
uses a future target. Rows whose target is NaN (censored or unavailable) are dropped, which is the
weight 0 treatment of masked days.

### Features

Features are built causally at the anchor `t`, using indices `<= t` only. They include recent lags,
weekday aligned lags, rolling mean, std and non zero rate over several windows, days since last sale,
series age, masked fraction, the target date known covariates (calendar, events, SNAP, price
features), and hierarchy category codes. The role of the lag and rolling features is to give a
gradient boosted model the recent level, recent volatility, and seasonality signal that a tree can
split on, so a single global model can learn patterns shared across items.

### Mean and quantile models

For each bucket the code trains one mean model with a count objective (`objective = "poisson"` is the
main run setting; `tweedie` is available with `tweedie_variance_power`) and one separate model per
quantile level with LightGBM's quantile objective (`objective = "quantile", alpha = q`). So the
probabilistic output is a set of independent quantile regressions.

### Non crossing

Independent quantile models can cross (a lower quantile predicting above a higher one). The fix is in
`BaseForecaster.finalize` (`src/forecastkit/models/base.py`): `q = np.sort(np.maximum(q, 0.0),
axis=-1)`. Every model's quantiles are clipped at 0 and then sorted along the quantile axis, which
removes crossing by construction.

### The hybrid

`KalmanLGBMForecaster` subclasses the LightGBM model and adds Kalman derived features through the
`extra_features` hook: the filtered level `kf_level`, the one step predictive standard deviation
`kf_sd1`, and the seasonal aware `h` step mean. All of these are computed from `y <= t` only, so the
hybrid keeps the same leak guarantee. The idea is that the per item Kalman filter supplies a locally
adaptive level and uncertainty, and the global tree model learns how to combine that local signal
with cross series patterns and covariates.

## 4. Quantile objective and pinball loss

For a single quantile level `q` and error `u = y - f` (actual minus forecast), the pinball (quantile)
loss is

    rho_q(u) = max(q * u, (q - 1) * u)

This is asymmetric: when the forecast is too low (`u > 0`) the loss weight is `q`; when it is too
high (`u < 0`) the weight is `1 - q`. Minimizing expected pinball loss drives `f` toward the `q`
quantile of `y`. In `src/forecastkit/evaluation/metrics.py`, `pinball(y, f, q)` returns the mean of
`rho_q` over observations, and `mean_pinball` averages `pinball` over the quantile set. The reported
"pinball" number is this mean over the quantile set; it is not scaled by actuals. LightGBM optimizes
exactly this loss per level during training (`objective = "quantile"`), so the tabular model is
scored on the same loss it is trained on, which keeps the comparison with the probabilistic models
fair.

## 5. Metric definitions exactly as coded

Source: `src/forecastkit/evaluation/metrics.py` and the aggregation in
`src/forecastkit/evaluation/evaluator.py` (`Errors` and `Evaluator.agg`).

All main table metrics are aggregated over a set of valid cells. A cell is one (origin, series,
horizon) triple. The valid mask `Evaluator.valid` starts from availability and then, depending on the
table, removes cold start series (`~self.cold`), removes censored cells (mode `non_censored`),
restricts horizons, or restricts to one origin. The main table uses established series
(`~self.cold`).

Which forecast statistic each metric uses:

- MAE: `mean |y - q_hat_0.5|`, using the median forecast (the quantile closest to 0.5). Code:
  `self.AE = |Y - med|`, `MAE = e.AE[v].mean()`.
- RMSE: `sqrt(mean (y - y_mean)^2)`, using the mean forecast (the `mean` statistic, or the median if
  no mean is provided). Code: `self.SE = (Y - M)^2`, `RMSE = sqrt(e.SE[v].mean())`.
- WAPE: aggregated, `sum |y - q_hat_0.5| / sum y` over the valid cells, using the median forecast.
  Code: `WAPE = e.AE[v].sum() / sy` where `sy = Y[v].sum()`. This is a ratio of sums, not a mean of
  per series ratios. If `sum y <= 0` it is NaN.
- pinball: `mean` over valid cells of the per cell pinball `PB`, where `PB` is itself the mean over
  the quantile set of `rho_q`. Code: `self.PBq = max(qs*d, (qs-1)*d)` shape `[O,N,H,Q]`,
  `self.PB = self.PBq.mean(-1)`, `pinball = e.PB[v].mean()`. Not scaled.
- WQL (weighted quantile loss): `mean over q of 2 * sum(pinball_q) / sum|y|`. Code in `agg`:
  `WQL = (2 * e.PBq[v].sum(0) / sy).mean()`. This is the scaled version; it divides by total actual
  so it is comparable across series of different volume. (The standalone `wql` in `metrics.py` uses
  `sum|y|` in the denominator with the same `2 * sum(pinball) / sum|y|` form.)
- Coverage: `cov80` and `cov90` are the share of actuals that fall inside the central interval, with
  inclusive bounds. For `cov = 0.8` the interval is `[q_hat_0.1, q_hat_0.9]`; for `cov = 0.9` it is
  `[q_hat_0.05, q_hat_0.95]`. Code: `(Y >= Q[..., jl]) & (Y <= Q[..., jh])`. Inclusive bounds are
  generous with zeros, which matters for count data.
- Width: mean interval width `q_hat_hi - q_hat_lo`.

Calibration (`Evaluator.calibration`): for each quantile the code reports both the non strict
empirical coverage `P(y <= q_hat)` (`hit`) and the strict version `P(y < q_hat)` (`hit_lt`). For
discrete counts the ideal lies between these two, so both are reported rather than just one. The flag
`inside_discrete_band` checks `lt <= q + 0.02 and le >= q - 0.02`.

Significance (`Evaluator.significance`): a paired bootstrap over series, not over cells. The code
resamples series indices with replacement `n_boot = 1000` times (`rng.integers(0, len(keep), size=
(n_boot, len(keep)))`) and recomputes the aggregated metric for each resample, for both the model and
the reference. The reported `diff` is the point difference `model - reference` in the aggregated
metric; a negative diff means the model is better (lower error). `ci_low`/`ci_high` are the 2.5 and
97.5 percentiles of the bootstrap difference, and `significant` is true when that interval excludes
zero. For WAPE the per series numerator and denominator are `sum |error|` and `sum y`; for pinball
the numerator is `sum PB` and the denominator is the cell count (so pinball is a per cell mean, WAPE
is a demand weighted ratio). `series_win_rate` is the fraction of series where the model's per series
ratio beats the reference. The reference model is `snaive4` (`outputs/m5_main/config.yaml`,
`evaluation.reference_model`).

Which significance table is which: `significance_wape.csv` has `metric = WAPE`, and
`significance_pinball.csv` has `metric = pinball` (confirmed in both files). In `RESULTS.md` the first
significance table (diffs around 0.33 for naive, around -0.076 for deepar) is WAPE; the second table
(diffs around 0.13 for naive, around -0.028 for deepar) is pinball.

## 6. Rolling origin evaluation and the 28 day horizon

Source: `src/forecastkit/backtest/splitter.py`.

An origin is the last observed day; the forecast window is `origin+1 .. origin+H`. The main run uses
`H = 28` with `n_origins = 4` and `step = 28` (`outputs/m5_main/config.yaml`). The four test origins
are 2016-01-31, 2016-02-28, 2016-03-27, 2016-04-24 (`RESULTS.md` section 1). Reasons:

- `H = 28` is the M5 competition horizon (28 days), so results are comparable to that benchmark.
- Multiple rolling origins reduce dependence on a single lucky or unlucky four week window; the
  origin stability table reports mean and standard deviation across the four folds.
- One earlier validation origin is reserved (`val_origins = 1`) and is strictly before the test
  origins (`val = [test[0] - step*(k+1) ...]`), so hyperparameter tuning (if run) never touches the
  test windows.

## 7. Direct versus recursive multi horizon

LightGBM here is direct: a separate model per horizon bucket predicts `y_{t+h}` straight from
features at `t`. The alternative is recursive: predict one step, feed the prediction back as an input,
and repeat. Trade offs, stated plainly:

- Direct does not feed its own forecast back, so one step errors do not compound into later steps.
- Direct needs more models and splits the data across horizons, so each horizon model sees fewer
  effective rows than a single recursive model would.
- Direct produces each horizon independently, so there is no guarantee that the forecast path is
  jointly coherent across horizons (no modeled correlation between `h` and `h+1`). DeepAR, by
  contrast, samples full paths autoregressively and so produces coherent sample paths.

## 8. The stockout proxy rule

Source: `src/forecastkit/masks.py` (`zero_run_stockout`, `_flag_runs_1d`).

The rule is a plain function. For one series with sales `y` and availability `a`:

1. Count non zero available days `n_nz`. If `n_nz < min_nonzero_events` (default 10), flag nothing.
2. Compute the average demand interval `ADI = (available days) / n_nz`.
3. Set a threshold `thr = max(min_run, adi_multiplier * ADI)` with `min_run = 28`,
   `adi_multiplier = 3.0` (defaults, matching `outputs/m5_main/config.yaml` masks block).
4. Find maximal runs of consecutive available zero days. Flag every run whose length is at least
   `thr`.

In words: a zero stretch is called a likely stockout only if it is much longer than that series' own
normal gap between sales, and only in a series that otherwise sells regularly.

Assumptions and failure modes, stated honestly:

- This is a proxy, not ground truth. M5 has no inventory records, so a zero day could be a genuine
  no demand day, a stockout, or a delisting. The toolkit never calls it ground truth (`masks.py`
  docstring, `README.md` honest limitations).
- False positives: a slow but legitimately quiet period in a normally busy item can be flagged even
  though demand really was zero.
- False negatives: a short stockout (shorter than `thr`), or a stockout in a low volume item
  (`n_nz < 10`), is not flagged.
- Why zero sales are not treated as confirmed lost demand: a zero means no units were sold, which
  could be because none were available or because none were wanted. Imputing demand into those days
  would invent numbers. The toolkit therefore marks and never imputes: flagged days become missing
  observations for Kalman, dropped rows for LightGBM, and masked loss terms for DeepAR.

## 9. The context boundary and what the leakage test proves

Source: `src/forecastkit/backtest/context.py`, `tests/test_leakage.py`.

`build_context` cuts every target array at the origin: `y_raw = ds.Y[:, :Th]` where `Th = origin+1`.
Future targets are not even present in the `ForecastContext` object. The availability tweaks (leading
zeros) and the stockout proxy are recomputed on this history slice, so a flag on day `t` can only
depend on data up to the origin. Cold start series have their availability turned off before the last
`cold_days` days, so they only expose recent history.

What `tests/test_leakage.py` actually verifies:

- `test_context_is_identical_when_the_future_is_scrambled`: replacing every target after the origin
  with random integers leaves `y`, `y_raw`, `available`, `censored`, `known_hist`, `known_fut`
  unchanged, and asserts the target array stops at the origin.
- `test_forecasts_do_not_depend_on_future_targets`: for six model families (seasonal_naive,
  moving_average, kalman, lgbm, kalman_lgbm, deepar), fitting on the real context and on the
  future scrambled context gives the same quantiles and mean within `rtol=1e-5, atol=1e-5`.
- `test_tabular_features_at_anchor_ignore_later_values`: tabular features at an anchor are unchanged
  when later values are scrambled.
- `test_cold_start_series_expose_only_recent_days`: cold start series reveal at most the configured
  number of recent days.
- `test_stockout_flags_in_context_use_history_only`: the censored flags are identical under future
  scrambling.

What it does not prove:

- It does not prove the proxy stockout labels are correct; it only proves they use history only.
- It does not prove the models are accurate or well calibrated; it only proves they do not read the
  future.
- It checks the specific perturbation (scrambling targets after the origin) for the tested models and
  the listed context arrays. It is strong evidence against target leakage, not a proof of the absence
  of every possible leak (for example through a covariate that was itself built with future
  information upstream of the adapter).

## 10. Why the Kalman to LightGBM hybrid is meaningful, and the limits of the evidence

The hybrid is technically meaningful because it combines two different strengths in one model: the
per item Kalman filter gives a locally adaptive level and an uncertainty estimate that reacts quickly
to each series on its own, and the global gradient boosted model learns patterns shared across all
series and uses covariates. Feeding the Kalman level, one step sd, and h step mean as features lets
the tree model use a local adaptive signal it would otherwise have to approximate from raw lags.

What the experiments do and do not establish:

- They show the hybrid beats plain LightGBM on this subset on all days (WAPE 0.7039 vs 0.7094,
  pinball 0.26198 vs 0.26366) and is best on non censored days and on cold start (section of the
  results doc). The margins are small.
- There is no ablation that isolates the Kalman features individually. The only comparison is lgbm
  versus kalman_lgbm, which turns the whole block of Kalman features on or off together. We cannot
  say which Kalman feature carries the gain, or whether a simpler added feature would do as well.
- The evidence is from four origins on one 1200 series subset with a single seed. Treat the hybrid
  advantage as a small, consistent, but not strongly powered result, not a guarantee.

No guarantee beyond code and tests is claimed anywhere in this document.
