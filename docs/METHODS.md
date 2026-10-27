# Methods reference (equations for the report)

## Notation
Series *i = 1..N*, day *t*, origin *T₀*, horizon *h = 1..H* (H = 28). Target *y_{i,t}* ≥ 0; availability *a_{i,t}*; censored flag *c_{i,t}* (stockout proxy). Observed-for-training set 𝒪 = {a=1, c=0}.

## Stockout proxy (masking rule)
For series *i* with history to *T₀*: ADI_i = (#available days)/(#days with y>0). A maximal run of consecutive available zero-sales days with length *L* is flagged censored iff
  *L* ≥ max(L_min, κ·ADI_i) and the series has ≥ n_min sales events (defaults L_min = 28, κ = 3, n_min = 10).
Computed **on the history slice only** at every origin. Flagged days are *missing observations* for Kalman, weight-0 rows for LightGBM, and masked terms in the DeepAR loss. This is a proxy; it is never described as ground truth.

## Local model: Kalman filter per SKU
Observation on transformed scale *z = g(y)* (g = √· by default):
  z_t = level_t + γ_t + ε_t,  ε_t ~ N(0, σ²)
  level_{t+1} = level_t + ξ_t,  ξ_t ~ N(0, q_ℓ σ²)
  γ_{t+1} = −Σ_{j=0}^{m−2} γ_{t−j} + ω_t,  ω_t ~ N(0, q_s σ²),  m = 7
State α_t = (level_t, γ_t, …, γ_{t−m+2})′, Z = (1, 1, 0, …), T = [[1,0],[0,S]] with S the seasonal dummy-form transition.
Recursions (missing y_t ⇒ skip the update): v_t = z_t − Zα_{t|t−1}, F_t = ZP_{t|t−1}Z′ + 1, K_t = P_{t|t−1}Z′/F_t, α_{t|t} = α_{t|t−1} + K_t v_t, P_{t|t} = P_{t|t−1} − K_tF_tK_t′, then α_{t+1|t} = Tα_{t|t}, P_{t+1|t} = TP_{t|t}T′ + Q.
σ² is **concentrated out**: σ̂² = (1/n)Σ v_t²/F_t, log-lik ≈ −½[n log σ̂² + Σ log F_t]. (q_ℓ, q_s) are chosen per series on a small grid by this likelihood (`param_mode: grid`), or one shared pair for all series (`shared`), or fixed.
Initial state: level = mean of first observations, P₀ = κI with κ = 10³ (diffuse) ⇒ a series with little history automatically gets **wide** intervals (the local model's cold-start behaviour).
Forecast: μ_h = Zα_{T₀+h|T₀}, V_h = σ̂²(ZP_{T₀+h|T₀}Z′ + 1). Quantile q on the original scale = g⁻¹(μ_h + Φ⁻¹(q)√V_h), clipped at 0 (exact for monotone g). Mean uses E[g⁻¹(Z)] (√: μ² + V).

## Tabular model: LightGBM, direct by horizon bucket
For bucket *b* (h ∈ {1–7, 8–14, 15–21, 22–28}) train on rows (series *i*, anchor day *t*, horizon *h*), target y_{i,t+h} ∈ 𝒪, t + h ≤ T₀. Features use indices ≤ t only: y_t, y_{t−1}, y_{t−2}; four weekday-aligned lags y_{t+h−7c}, c = ⌈h/7⌉ + k − 1; rolling mean/std/non-zero-rate over 7/14/28/56/112 days (cumulative-sum, causal); days since last sale; age; masked-fraction; and **target-date** known covariates (calendar, events, SNAP, price features) plus hierarchy codes. One Poisson (mean) model and one quantile-objective model per level in the quantile set. Quantiles are sorted (no crossing) and clipped ≥ 0.
**Hybrid** (`kalman_lgbm`): adds the Kalman filtered level, one-step predictive sd and the seasonal-aware *h*-step mean Zᵀʰα_{t|t} (all computed from y ≤ t) as features.

## Global model: DeepAR
Per-window scale ν = 1 + mean(context y). LSTM input at step t: [y_{t−1}/ν, observed-flag, log ν, covariates x_t (standardised), embeddings of static categories (+ item id)]. Output (μ_t, α_t): μ = ν·softplus(·), α = softplus(·)/√ν.
Negative binomial with mean μ and dispersion α (r = 1/α):
  log p(y | μ, α) = logΓ(y+r) − logΓ(r) − logΓ(y+1) + r log(r/(r+μ)) + y log(μ/(r+μ)),  Var = μ + αμ².
Loss = −Σ_{(i,t)∈𝒪} log p(y_{i,t} | ·) (stockout days masked). Training windows are drawn only from t ≤ T₀. Inference: ancestral sampling of S = 200 paths over the horizon, quantiles/mean from the samples. Item-id embedding index 0 = "unknown item" (trained with item-dropout; used for cold-start series).

## Metrics (all computed on available days of established series unless stated)
- MAE = mean|y − q̂₀.₅|;  RMSE = √mean(y − ŷ_mean)²;  **WAPE = Σ|y − q̂₀.₅| / Σy** (aggregated).
- Pinball at level q: ρ_q(u) = max(q·u, (q−1)·u), u = y − q̂_q; reported as the mean over the quantile set; wQL = mean_q 2Σρ_q / Σy.
- Coverage_c = share of y inside [q̂_{(1−c)/2}, q̂_{1−(1−c)/2}] (c = 0.8, 0.9, inclusive); Width = mean interval length.
- Calibration for discrete y: report both P(y < q̂) and P(y ≤ q̂); a calibrated quantile lies between them.
- Significance: paired bootstrap over *series* (1000 resamples) of the difference in aggregated WAPE / pinball vs a reference model.
- Decision link: the 0.9 quantile of demand over the replenishment lead time is the order-up-to level for a ≈90 % cycle-service level; Cov90 ≈ 0.9 means that level delivers the service it promises.
