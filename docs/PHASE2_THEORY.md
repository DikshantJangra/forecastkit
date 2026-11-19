# Phase 2 Theory and Methods

This document states the mathematics behind every Phase 2 component and the assumptions it rests on. Phase 1
material (Kalman model, LightGBM formulation, masks, metrics) is in `docs/THEORY_AND_METHODS.md`; it is summarised
here only where Phase 2 depends on it. Code references are given as `file:function`.

## 1. Quantities: what is observed and what is not

For series i and day t:

| symbol | meaning | observed? |
|---|---|---|
| y_{i,t} | recorded unit sales | yes |
| a_{i,t} | availability: the item has a price that week and is past its first sale | yes (from `sell_prices`) |
| d_{i,t} | latent demand | no |
| s_{i,t} | in stock | **no** (M5 has no inventory data) |
| c_{i,t} | stockout *proxy* flag: day lies in a zero run of length >= max(28, 3 ADI_i) in a series with >= 10 sales | computed |

Assumption A1 (censoring model): y = d when s = 1 and y = 0 when s = 0. Under A1 a long zero run in a regularly
selling series is *consistent with* censoring, but also with delisting, a data gap, or genuine absence of demand.
The proxy c is a heuristic, not a label. Nothing in Phase 2 claims to predict s.

Two forecasting targets follow, and both are reported:

* **recorded sales** y (all-days evaluation): what a replenishment system actually observes;
* **demand where it is visible** (non-censored evaluation): cells with c = 0 in the hindsight labels.

Hindsight labels c (computed on the full series) define the non-censored *evaluation view* only. Models only ever
see c computed on the history slice up to the origin (`backtest/context.py`), so a flag on day t never depends on
days after the origin.

## 2. Rolling-origin protocol (unchanged from Phase 1)

Origins T_1 < ... < T_4 (step 28), horizon H = 28. For each origin T the model receives a `ForecastContext` with
y_{1:T}, a_{1:T}, c^{(T)}_{1:T} (proxy recomputed on the slice), known covariates x_{1:T+H} (calendar, SNAP, price;
M5 publishes prices in advance), static codes. Targets after T are not present in the object. One earlier origin
(2016-01-03) is used only for design selection (E02). Inside each fit, the last H days of history form an internal
validation block (DeepAR checkpointing, zero-model early stopping), so no test cell influences any choice.

## 3. DeepAR (global autoregressive RNN)

Following Salinas et al. (2020), for each series a scale nu_i = 1 + mean of observed y in the context window. The
LSTM (Hochreiter & Schmidhuber, 1997) state evolves as

    h_t = LSTM(h_{t-1}, [ y_{t-1}/nu, 1[y_{t-1} observed], log nu, (y_{t-k}/nu, 1[y_{t-k} observed])_{k in K}, x_t, e_i ])

with K = {7, 14, 28} in DeepAR v2 (empty in Phase 1), x_t the standardised known covariates *for the target day*,
e_i the concatenated embeddings of dept, category, store, state and item (item id dropped with probability 0.15
during training and replaced by "unknown" for cold-start series). The output layer gives

    mu_t = softplus(w_mu' h_t) * nu,      alpha_t = softplus(w_a' h_t) / sqrt(nu)

and the likelihood is negative binomial with mean mu and dispersion alpha:

    NB(y; mu, alpha) = Gamma(y + r) / (Gamma(r) y!) * (r/(r+mu))^r * (mu/(r+mu))^y,   r = 1/alpha,
    E[y] = mu,   Var[y] = mu + alpha mu^2   (over-dispersed relative to Poisson).

Verified: the NLL matches `scipy.stats.nbinom.logpmf` to 1e-8 (`tests/test_phase2_dl.py::test_nb_nll_matches_scipy`);
the Gamma-Poisson sampler reproduces mean, variance and P(0) to < 2% (`test_nb_sampler_moments`).

**Loss masking.** Training minimises the mean NLL over observed target days only:
L = sum_{t in W} m_t * (-log NB(y_t; mu_t, alpha_t)) / sum_t m_t, m_t = 1[a_t and not c_t^{(T)}]. Masked days still
enter the input sequence as (0, flag = 0), so the network knows a value is missing without seeing it.
Consequence: masked DeepAR estimates the distribution of y *given the item sells normally*. The unmasked
variant (use_mask = false) learns the long zero runs as part of y.

**Seasonal lags.** In ancestral sampling the lag y_{T+h-k} is the observed value if h - k <= 0 and the model's own
sampled value otherwise, so each sample path is a coherent joint draw.

**Training and regularisation.** Adam (lr 1e-3, cosine decay to 1e-4), coupled L2 1e-5, gradient-norm clipping at 10,
LSTM inter-layer dropout 0.1 (Srivastava et al., 2014), item-id dropout 0.15, 2,500 steps of 128 windows from the
last 730 days; warm-started fine-tuning (800 steps, lr x 0.5) at later origins. DeepAR v2 adds a *validation block*:
training windows end at T - H; every 100 steps the teacher-forced NLL of the last H days of history (all series,
windows ending at T) is computed; the best checkpoint is restored (early stopping with patience 5; Prechelt, 1998).
At warm-started origins the incoming weights are evaluated first and compete with the fine-tuned ones. AdamW
(decoupled weight decay; Loshchilov & Hutter, 2019) is implemented and tested as an ablation.

**Product age.** age_{i,t} = log(1 + max(0, t - f_i)) / 7 with f_i the first available day *in the history slice*;
it is deterministic in the future and therefore a known covariate. In the 1,000-day window it is left-truncated
for old items (all old items look the same age at the window start), which limits its information.

## 4. Zero-run persistence model (ML component)

**Target.** pi_{i,h}(T) = P(y_{i,T+h} = 0 | F_T), h = 1..H, for available target days. It is the probability of a
*recorded* zero. It is observable, needs no inventory label, and is exactly the zero mass a forecast of recorded
sales must get right. It is not P(stockout).

**Features at anchor t** (all computed from y_{1:t}, `zerorun/features.py`; causality is tested by perturbing
the future, `tests/test_phase2_hybrid.py::test_zero_run_features_are_causal`):

* current zero-run length r_t (days since last sale), r_t / ADI_t, the causal proxy flag
  1[r_t >= max(28, 3 ADI_t) and n_sales_t >= 10];
* completed-run statistics up to t (count, mean, max, number >= 28 days), r_t / max completed run;
* expanding ADI and log number of sales;
* Phase 1 leak-safe tabular features: rolling mean / sd / non-zero rate / unavailable fraction over 7..112 days,
  lags y_t, y_{t-1}, y_{t-2}, same-weekday lags, expanding mean, age;
* known covariates at the target day (weekday, events, SNAP, price), static hierarchy codes, and h.

**Learner.** LightGBM (Ke et al., 2017) with binary log loss (a strictly proper scoring rule; Gneiting & Raftery,
2007). One model for all horizons. Training rows: (series, anchor t, h) with anchors every 7 days over the last
364 days and t + h <= T - H (capped at 600k rows); validation rows: anchor T - H, h = 1..H. Early stopping on the
validation log loss (patience 50, max 1,000 trees). Class balance: recorded zeros are the majority (~70% of cells);
no re-weighting is applied because re-weighting would bias the probabilities that the fusion consumes.

**Baselines** (`zerorun/baselines.py`): series zero rate (expanding), last-28-day zero rate, and a run-duration
renewal baseline. The latter estimates S(d) = P(run length > d) by Kaplan-Meier (Kaplan & Meier, 1958), pooled
within ADI quartiles, with the run open at the origin right-censored. For a series in a run of length r >= 1,

    s_h = S(r + h - 1) / S(r - 1)      (probability the run is still open at T + h)
    pi_h^KM = s_h + (1 - s_h) z_i       (after the run ends: zero with the series' zero rate z_i)

and pi_h^KM = z_i if the series sold on the origin day.

## 5. Fusion: Rao-Blackwellised hurdle (the hybrid)

DeepAR's predictive marginal at horizon h is a mixture over S = 200 sampled paths,
p_D(y) = (1/S) sum_s NB(y; mu_{s,h}, alpha_{s,h}). Its zero mass is computed exactly (Rao-Blackwellised) as
p_D(0) = (1/S) sum_s (r_s/(r_s + mu_s))^{r_s} instead of counting zero samples.

**Hurdle fusion** (Mullahy, 1986):

    q(0) = pi_h,        q(y) = (1 - pi_h) p_D(y) / (1 - p_D(0)),   y >= 1.

Properties: q(y) >= 0 and sum_y q(y) = pi + (1 - pi) = 1 (valid pmf on the non-negative integers);
E_q[Y] = (1 - pi) E_D[Y | Y > 0] = (1 - pi) sum_s mu_s / sum_s (1 - p0_s);
quantiles Q(tau) = 0 if tau <= pi, else F_+^{-1}((tau - pi)/(1 - pi)) with F_+ the CDF of p_D(. | Y > 0);
positive draws are exact: choose path s with probability proportional to 1 - p0_s, draw from the zero-truncated
NB(mu_s, alpha_s) by inverse CDF (`models/hybrid.py:truncated_nb_mixture_samples`, tested against closed forms).

**Why not the naive zero-inflated mixture.** q_ZI = pi delta_0 + (1 - pi) p_D gives q_ZI(0) = pi + (1 - pi) p_D(0).
If pi is already a calibrated estimate of P(Y = 0) (it is trained on exactly that event), the extra (1-pi) p_D(0)
counts DeepAR's own zeros a second time and q_ZI(0) > pi systematically. The ZI form (Lambert, 1992) is correct
only when pi models a *separate* zero-generating state, which would require stock labels we do not have.

**Variants (ablations).** All keep DeepAR's positive conditional and change only q(0): `zi_naive` (above),
`zi_inflate` q(0) = max(pi, p_D(0)) (consistent zero inflation, can only add zero mass), `gated` (pi only for series
in a causally flagged zero run at the origin), `none` q(0) = p_D(0) (control: should match DeepAR up to Monte-Carlo
noise), `no_dl` (positive part = empirical distribution of the series' positive sales in the last 112 days),
`no_persist` (zero model without explicit run features).

**Division of labour.** The ML component estimates a binary, strongly state-dependent quantity (how long a zero
run lasts) from hand-built run statistics, which trees exploit well. The DL component models the size distribution
of sales, shared across 1,200 series, and remains a latent-demand model because its loss is masked.

## 6. Evaluation

* WAPE = sum |y - q_0.5| / sum y (aggregated, never a mean of ratios); MAE on the median; RMSE on the mean.
  WAPE is an error ratio, not an accuracy percentage.
* Mean pinball loss over the 7 quantile levels (a proper scoring rule for quantiles); coverage and width of the
  80% / 90% central intervals.
* Quantile calibration: P(y <= q_tau) vs tau; for discrete y the target lies between P(y < q_tau) and P(y <= q_tau).
* Randomised PIT (Czado et al., 2009): u = F(y - 1) + V [F(y) - F(y - 1)], V ~ U(0,1); calibrated count forecasts
  give u ~ U(0,1) (`evaluation/metrics.py:randomized_pit`, tested on simulated Poisson data).
* Zero-mass scores of P(Y = 0): Brier, log loss, expected calibration error (10 bins), ROC-AUC, PR-AUC.
* Paired bootstrap (`Evaluator.paired_bootstrap`): resampling unit = series (1,000 draws in the tracked package); all days of a series move
  together, which keeps within-series autocorrelation (cf. block bootstrap, Kuensch, 1989). Seed replicates are
  averaged per series before resampling. We also report the per-origin differences (4 origins, too few for a
  formal origin-level test such as Diebold-Mariano, 1995). With 1,116 warm series, CIs reflect between-series
  variability, not variability across time periods or seeds; seed variability is reported separately (mean +- sd).

## 7. Leakage controls added in Phase 2

* DeepAR lags: positions p - k < 0 are missing; in sampling lags come from history or the same path.
* DeepAR validation block and zero-model validation rows lie inside the history slice.
* Zero-run features: day-t values depend on days <= t (perturbation test).
* `gated` fusion uses c^{(T)} (history-only proxy), never hindsight labels.
* Design selection (E02) used only the validation origin. Hyper-parameters of the zero model are defaults chosen
  before any test evaluation and were not tuned.

Known, disclosed Phase 1 issue (not changed, to keep Phase 1 reproducible): the Kalman-LightGBM hybrid estimates
shared noise parameters once per origin on a 514-day window, so training-anchor features use parameters fitted with
data up to the origin (training-time look-ahead inside the history; the test window is not involved).
