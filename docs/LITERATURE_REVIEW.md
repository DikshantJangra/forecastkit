# Literature Review

## 1. Introduction

This document reviews the forecasting research that motivates the models in forecastkit and the way we evaluate them. The project studies inventory demand forecasting on the M5 retail dataset, where sales are daily counts, many series are intermittent (lots of zeros), and prices, calendar events and promotions matter. forecastkit implements four families of forecasters plus simple baselines: a per series Kalman filter (local level and stochastic weekly seasonal on a square root scale), LightGBM trained direct by horizon with Poisson mean and quantile objectives, a Kalman to LightGBM hybrid that feeds the filtered state in as features, and DeepAR (a global LSTM with a negative binomial likelihood). Baselines are naive, seasonal naive at one and four weeks, and a 28 day moving average. We also form an equal weight quantile average ensemble, mask suspected stockouts, and score everything with a leak safe rolling origin protocol using pinball loss and WAPE.

Some classical intermittent demand and statistical methods appear throughout this review as related work. To be clear up front: Croston, SBA, TSB, ETS and ARIMA are not implemented in forecastkit. They are discussed because they frame the problem and explain our model choices, not because they are part of the code.

The review is organised by method family, then M5 findings, then evaluation, followed by a comparison table, a research gap statement, and the reference list.

## 2. Intermittent demand methods

Retail demand at the item level is often intermittent: many days have zero sales and positive demand arrives in irregular bursts. Simple exponential smoothing is biased on such data because it keeps updating toward zero between demand events. Croston (1972) proposed splitting the series into two parts, the size of nonzero demand and the interval between demand events, and smoothing each separately. This removed much of the bias and became the standard intermittent demand method.

Croston's estimator is itself biased, which Syntetos and Boylan (2005) showed analytically and corrected with a simple scaling factor, giving the Syntetos Boylan Approximation (SBA). Syntetos, Boylan and Croston (2005) then gave a practical rule for categorising demand patterns into smooth, erratic, intermittent and lumpy using the average inter demand interval (ADI) and the squared coefficient of variation, which tells a practitioner which method to prefer for each series. Teunter, Syntetos and Babai (2011) proposed the TSB method, which updates the demand probability every period rather than only at demand events, so it can react to demand that is obsolescing (falling toward zero), a case Croston and SBA handle poorly.

These methods are valuable and still competitive on sparse series, but they have real limits for an M5 style problem. They produce point forecasts, not full predictive distributions, so they do not directly give the quantiles needed for inventory service levels. They do not use seasonality or covariates such as price, SNAP days or events, which are strong signals in M5. Our stockout masking rule borrows the ADI idea from Syntetos, Boylan and Croston (2005) to decide when a run of available zero days looks like a stockout rather than genuine zero demand, so the categorisation literature informs the toolkit even though the forecasters themselves are not implemented.

## 3. Classical statistical methods

The two dominant classical families are exponential smoothing in state space form (ETS) and ARIMA. Hyndman, Koehler, Snyder and Grose (2002) placed exponential smoothing inside a single state space framework that gives each method a likelihood, automatic model selection, and prediction intervals, which turned a set of ad hoc recipes into a principled statistical family. Hyndman and Khandakar (2008) made both ETS and automatic ARIMA selection routine through the R forecast package, which is why they are standard benchmarks today.

On M5 style data these methods struggle for two reasons. First, they assume continuous, roughly Gaussian errors, which is a poor fit for sparse integer counts with mass at zero. Second, applied per series they cannot borrow strength across the thousands of related items, so items with short or noisy histories get unstable estimates. What survives this stress test is the humble seasonal naive forecast. Because retail demand has a strong weekly cycle, repeating the value from the same weekday last week is hard to beat, which is why seasonal naive is treated as a serious benchmark in M5 and why we include seasonal naive at one and four weeks as baselines.

## 4. Tree based methods

Gradient boosted decision trees became the workhorse of applied tabular forecasting. Ke et al. (2017) introduced LightGBM, an efficient GBDT implementation using histogram based splits, gradient based one side sampling and exclusive feature bundling, which made it fast and scalable on large, wide feature tables. For forecasting, a time series is reshaped into a supervised table: each row is a (series, day, horizon) triple and the features are lagged sales, rolling means and variances, days since last sale, calendar fields, event flags and price features. forecastkit follows exactly this recipe, training LightGBM direct by horizon bucket with a Poisson objective for the mean and a pinball objective per quantile level.

The M5 accuracy competition established how strong this approach is. Makridakis, Spiliotis and Assimakopoulos (2022) report that machine learning methods, LightGBM in particular, dominated the leaderboard, with the top solutions built on gradient boosted trees over engineered lag and calendar features, while pure statistical methods trailed and combinations of simple methods served as strong benchmarks. This result is the single clearest reason LightGBM is a central model in forecastkit.

## 5. Global versus local learning and DeepAR

A local model fits one set of parameters per series. A global model fits one model across all series, so parameters are shared and the model learns patterns that recur across items. Januschowski et al. (2020) give a careful vocabulary for this distinction and others, and Montero-Manso and Hyndman (2021) show, with theory and experiments, that a single global model can match or beat many local models precisely because sharing parameters across related series acts as a form of regularisation and lets scarce data go further. This is the core argument for deep learning on large panels like M5.

DeepAR (Salinas et al., 2020) is the canonical global probabilistic model. An autoregressive LSTM is trained across all series; it reads lagged values, covariates and learned embeddings of static categories, and outputs the parameters of a likelihood (negative binomial for counts) at each step. At inference it samples many future paths and reads quantiles off the samples, so it produces a full predictive distribution rather than a point. Cross series learning explains its cold start behaviour: a brand new item with little history still gets a reasonable forecast because the shared weights and category embeddings transfer knowledge from similar items. forecastkit implements a compact DeepAR with a negative binomial head, loss masking for stockout days, and an explicit unknown item embedding so cold start series can be handled honestly.

## 6. Kalman filter and state space models

The Kalman filter (Kalman, 1960) is the recursive solution for estimating the hidden state of a linear Gaussian system as observations arrive. Structural time series models express a series as unobserved components, a slowly moving level and a seasonal pattern, and estimate them with the filter; Harvey (1989) and Durbin and Koopman (2012) are the standard treatments. Two properties matter for this project. First, the level and seasonal components are time varying, so the model tracks demand that drifts and has an evolving weekly shape rather than assuming fixed coefficients. Second, missing observations are handled naturally: when a value is absent the filter simply skips the measurement update and propagates the state forward, so uncertainty widens in a principled way.

That second property is the direct reason the Kalman filter fits our stockout masking design. When a run of days is flagged as a likely stockout, forecastkit treats those days as missing observations for the filter instead of imputing fake sales, which keeps the estimate honest. The same filtered state (level, one step predictive standard deviation and a seasonal aware h step mean) is what the Kalman to LightGBM hybrid feeds to the tree model as a state space prior. We fit the filter on a square root scale to stabilise count variance, which is a pragmatic choice; the Gaussian likelihood is admittedly weak on the most intermittent items, a limitation we report rather than hide.

## 7. M5 findings and strong benchmarks

M5 is the most relevant empirical touchstone because it uses Walmart data of exactly the kind we target. The accuracy track (Makridakis, Spiliotis and Assimakopoulos, 2022) found that gradient boosted trees dominated, that cross learning across series helped, and that simple combinations and seasonal methods were strong benchmarks that many submissions failed to beat. The uncertainty track (Makridakis et al., 2022) scored full predictive distributions with a weighted quantile loss and reached similar conclusions, underscoring that getting the whole distribution right, not just the point, is what inventory decisions need. These two papers justify the model mix in forecastkit (trees, a global deep model, strong naive and seasonal baselines) and the decision to score quantiles directly.

## 8. Evaluation

Good evaluation is as important as good models. Gneiting and Raftery (2007) formalise proper scoring rules: a score is proper if it is optimised in expectation by the true distribution, so proper scores cannot be gamed by hedging. The pinball (quantile) loss is proper for an individual quantile and its average over a quantile grid approximates the continuous ranked probability score, which is why it is the primary probabilistic metric here and in the M5 uncertainty track. Hyndman and Koehler (2006) review point forecast accuracy measures and warn that percentage errors blow up when actuals are zero, which is common in intermittent retail. This motivates our use of WAPE, an aggregated absolute error divided by total demand, which stays well defined and interpretable when many actuals are zero and weights series by volume. Kolassa (2016) studies how to evaluate predictive count distributions specifically in retail and stresses discrete aware calibration, which is why forecastkit reports calibration as a band between P(y < q) and P(y <= q) rather than a single curve. Snyder, Ord and Beaumont (2012) is a useful reference for count based intermittent demand models that treat the integer nature of demand directly.

## 9. Comparison table

| Method | Main idea | Strengths | Limitations | Relevance (implemented?) |
|---|---|---|---|---|
| Naive / seasonal naive / moving average | Repeat last value, same weekday last week, or recent average | Trivial, robust, strong weekly benchmark | No covariates, point only | Yes (baselines) |
| Croston / SBA | Smooth demand size and interval separately; SBA debiases | Designed for intermittency | Point only, no seasonality or covariates, residual bias | No (related work) |
| TSB | Update demand probability every period | Handles obsolescence | Point only, no covariates | No (related work) |
| ETS | Exponential smoothing in state space form | Principled, intervals, auto selection | Gaussian, weak on sparse counts, local only | No (related work) |
| ARIMA | Autoregressive integrated moving average | Flexible linear dynamics | Poor on sparse counts, per series, weak with covariates | No (related work) |
| Kalman structural | Time varying level and seasonal via recursive filter | Tracks drift, handles missing data naturally, few parameters | Gaussian likelihood weak on very intermittent items | Yes |
| LightGBM (direct) | GBDT on lag, calendar and price features | Fast, uses covariates, M5 winner | Needs feature engineering, not natively probabilistic (we add quantile models) | Yes |
| Kalman to LightGBM hybrid | Feed filtered state as tree features | Combines state space prior with covariates | Features only as good as the filter | Yes |
| DeepAR | Global LSTM with a count likelihood, sampled paths | Cross series learning, full distribution, cold start via embeddings | Data and compute hungry, less interpretable | Yes |
| Quantile average ensemble | Average quantiles across models | Simple, often well calibrated | Not trained jointly | Yes |

## 10. Research gap and model selection rationale

Each implemented model maps onto a strand of the literature. The baselines encode the M5 lesson that seasonal naive is a strong benchmark (Makridakis, Spiliotis and Assimakopoulos, 2022). The Kalman filter instantiates structural state space modelling (Kalman, 1960; Harvey, 1989; Durbin and Koopman, 2012) and its missing data handling is what makes stockout masking clean. LightGBM reflects the dominance of gradient boosted trees in M5 (Ke et al., 2017; Makridakis, Spiliotis and Assimakopoulos, 2022). The hybrid tests whether a state space prior helps a tabular learner. DeepAR represents global probabilistic deep learning and the locality versus globality argument (Salinas et al., 2020; Januschowski et al., 2020; Montero-Manso and Hyndman, 2021). The ensemble follows the M5 finding that combinations are hard to beat.

To our knowledge, relatively few studies place a local state space model, a tabular gradient boosting model and a global deep model side by side under one leak safe rolling origin protocol, with explicit stockout censoring handled consistently across every model and with calibrated quantiles scored by proper rules (Gneiting and Raftery, 2007; Kolassa, 2016). Most published comparisons focus within a single family, or report point accuracy, or do not make stockout handling explicit and identical across methods. We do not claim this combination is novel in any strong sense; the individual pieces are all established. The contribution is a fair, reproducible comparison framework that treats censoring, leakage and probabilistic calibration as first class concerns, so that differences between model families reflect the models rather than the experimental setup.

## 11. References

Croston, J. D. (1972). Forecasting and stock control for intermittent demands. *Operational Research Quarterly*, 23(3), 289-303. https://doi.org/10.1057/jors.1972.50

Durbin, J., & Koopman, S. J. (2012). *Time series analysis by state space methods* (2nd ed.). Oxford University Press. https://doi.org/10.1093/acprof:oso/9780199641178.001.0001

Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359-378. https://doi.org/10.1198/016214506000001437

Harvey, A. C. (1989). *Forecasting, structural time series models and the Kalman filter*. Cambridge University Press. https://doi.org/10.1017/CBO9781107049994

Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series forecasting: The forecast package for R. *Journal of Statistical Software*, 27(3), 1-22. https://doi.org/10.18637/jss.v027.i03

Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688. https://doi.org/10.1016/j.ijforecast.2006.03.001

Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002). A state space framework for automatic forecasting using exponential smoothing methods. *International Journal of Forecasting*, 18(3), 439-454. https://doi.org/10.1016/S0169-2070(01)00110-8

Januschowski, T., Gasthaus, J., Wang, Y., Salinas, D., Flunkert, V., Bohlke-Schneider, M., & Callot, L. (2020). Criteria for classifying forecasting methods. *International Journal of Forecasting*, 36(1), 167-177. https://doi.org/10.1016/j.ijforecast.2019.05.008

Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35-45. https://doi.org/10.1115/1.3662552

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30 (NIPS 2017)* (pp. 3146-3154). https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

Kolassa, S. (2016). Evaluating predictive count data distributions in retail sales forecasting. *International Journal of Forecasting*, 32(3), 788-803. https://doi.org/10.1016/j.ijforecast.2015.12.004

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346-1364. https://doi.org/10.1016/j.ijforecast.2021.11.013

Makridakis, S., Spiliotis, E., Assimakopoulos, V., Chen, Z., Gaba, A., Tsetlin, I., & Winkler, R. L. (2022). The M5 uncertainty competition: Results, findings and conclusions. *International Journal of Forecasting*, 38(4), 1365-1385. https://doi.org/10.1016/j.ijforecast.2021.10.009

Montero-Manso, P., & Hyndman, R. J. (2021). Principles and algorithms for forecasting groups of time series: Locality and globality. *International Journal of Forecasting*, 37(4), 1632-1653. https://doi.org/10.1016/j.ijforecast.2021.03.004

Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181-1191. https://doi.org/10.1016/j.ijforecast.2019.07.001

Snyder, R. D., Ord, J. K., & Beaumont, A. (2012). Forecasting the intermittent demand for slow-moving inventories: A modelling approach. *International Journal of Forecasting*, 28(2), 485-496. https://doi.org/10.1016/j.ijforecast.2011.03.009

Syntetos, A. A., & Boylan, J. E. (2005). The accuracy of intermittent demand estimates. *International Journal of Forecasting*, 21(2), 303-314. https://doi.org/10.1016/j.ijforecast.2004.10.001

Syntetos, A. A., Boylan, J. E., & Croston, J. D. (2005). On the categorization of demand patterns. *Journal of the Operational Research Society*, 56(5), 495-503. https://doi.org/10.1057/palgrave.jors.2601841

Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011). Intermittent demand: Linking forecasting to inventory obsolescence. *European Journal of Operational Research*, 214(3), 606-615. https://doi.org/10.1016/j.ejor.2011.05.018

## 12. Needs verification

No references are unverified. Two notes on the entries above:

- Croston (1972) was published in *Operational Research Quarterly* (Vol. 23, No. 3, pp. 289-303). Crossref returns two DOIs for this article: 10.2307/3007885 (JSTOR stable record, which lists only the start page 289) and 10.1057/jors.1972.50 (the publisher record under the journal's later name, Journal of the Operational Research Society, with full pages 289-303). We cite the publisher DOI, which carries the complete page range.
- Ke et al. (2017), LightGBM, was published in the NeurIPS 2017 proceedings, which do not mint DOIs. It is verified instead against the official NeurIPS proceedings page, which lists the title, all eight authors, and the volume (Advances in Neural Information Processing Systems 30).

---

# Phase 2 addendum: deep learning, intermittent counts and hybrid forecasting

Sections 1-12 above are the Phase 1 review and are unchanged. This addendum covers the literature behind the
Phase 2 design decisions. Every reference was checked on 2026-11-19 (DOIs through api.crossref.org; arXiv,
NeurIPS, ICLR and JMLR entries on the publisher pages).

## 13. Recurrent networks for forecasting

LSTMs (Hochreiter & Schmidhuber, 1997) add gated memory cells to RNNs, which mitigates vanishing gradients and
lets the state carry information over long lags. Hewamalage, Bergmeir and Bandara (2021) benchmark RNN forecasters
across many datasets. Their findings that matter here: RNNs need per-series normalisation, they are competitive
mainly when trained *globally* across many related series, and they are sensitive to the input window and
to how seasonality is presented. Bandara, Bergmeir and Smyl (2020) show that global RNNs gain when they are trained
on groups of similar series.

*Relevance to ForecastKit.* DeepAR is a global LSTM with mean scaling, which matches these recommendations. The
DeepAR inputs in Phase 1 were only y_{t-1} and covariates, so the weekly and monthly pattern had to be carried by
the state. Following the advice on presenting seasonality explicitly, Phase 2 adds seasonal lags 7/14/28 and
tests them on a validation origin before using them (ledger E02).

## 14. DeepAR and probabilistic deep forecasters

DeepAR (Salinas et al., 2020) trains an autoregressive LSTM by maximum likelihood with a parametric output
distribution (negative binomial for counts) and produces forecasts by ancestral sampling. MQ-RNN (Wen et al., 2017)
outputs quantiles directly for all horizons (direct, non-sampled). Temporal Fusion Transformers (Lim et al., 2021)
combine recurrent encoders, attention and variable selection for interpretable multi-horizon quantile forecasts.

*Relevance.* We kept DeepAR because (i) the negative binomial gives a full count distribution, and the hybrid
needs its zero mass and positive-part shape, which quantile-only models (MQ-RNN, TFT) do not provide; (ii)
its loss can be masked per day, which is how censoring is handled; (iii) its compute cost fits a CPU-only budget.
TFT is a natural next step but would need a count likelihood to plug into the same fusion.

## 15. Deep learning for intermittent demand

Türkmen, Januschowski, Wang and Cemgil (2021) recast intermittent demand as a renewal process: a model of
inter-demand intervals plus a model of demand sizes, both parameterised by RNNs ("deep renewal processes"). This
generalises Croston's size/interval decomposition (Section 2) to a probabilistic, global model. Jeon and Seong
(2022) propose a robust RNN for intermittent series in which the network is trained to handle the bursty structure
of sparse sales.

*Relevance.* Both papers support splitting the problem into "when is there a sale" and "how much is sold". The
ForecastKit hybrid makes the same split, but assigns the "when" part to a gradient-boosted classifier using
explicit run-duration features and keeps a single DeepAR for the size part. Our Kaplan-Meier renewal baseline is
the non-learned version of the interval model.

## 16. Zero-inflated and hurdle count models

Zero-inflated models (Lambert, 1992) treat the data as a mixture of a degenerate "always zero" state and a count
distribution, P(0) = pi + (1 - pi) f(0). Hurdle models (Mullahy, 1986) instead use one binary process for zero vs.
positive and a zero-truncated count distribution for positive values, P(0) = pi. The two differ in where zeros come
from: ZI models have zeros from two sources, hurdle models from one.

*Relevance.* This distinction is the core design choice of the Phase 2 fusion. Our ML component is trained on
the event "recorded sales = 0", so it already estimates the total zero mass. Mixing it ZI-style with DeepAR
counts DeepAR's zeros twice. The hurdle form uses it directly. Both forms are implemented and compared
(ablation "Fusion: naive ZI").

## 17. Statistical + neural hybrids and deep state-space models

The M4 winner (Smyl, 2020) is a hybrid in which per-series exponential smoothing parameters provide level and
seasonality and a global RNN learns the remaining dynamics. Deep State Space Models (Rangapuram et al., 2018)
let an RNN output the parameters of per-series linear-Gaussian state-space models. Deep Factors (Wang et al., 2019)
combine global deep factors with local probabilistic models.

*Relevance.* These hybrids give each component a different role inside one probabilistic model. Averaging
outputs does not do that. This is the standard our hybrid is held to. Our Phase 1 Kalman-LightGBM is a
feature-level ML+ML hybrid. The Phase 2 hybrid is a distribution-level ML+DL hybrid. The deep state-space route
(an RNN feeding a Kalman filter) was considered, but the Gaussian state-space likelihood fits sparse counts
poorly (Section 6), which is the problem we are trying to solve.

## 18. Training and regularisation

Early stopping on a held-out validation set (Prechelt, 1998) and dropout (Srivastava et al., 2014) are the
standard regularisers for RNNs. Loshchilov and Hutter (2019) show that L2 regularisation and weight decay are not
equivalent for adaptive optimisers such as Adam and propose decoupled weight decay (AdamW).

*Relevance.* Phase 1 used Adam with a small coupled L2 penalty and a fixed step budget. Phase 2 adds a validation
block inside the history with best-checkpoint restore and implements AdamW. AdamW is kept only if it helps on the
validation origin (it fell inside the tie band and was not adopted).

## 19. Evaluating probabilistic count forecasts

Gneiting, Balabdaoui and Raftery (2007) define the paradigm "maximise sharpness subject to calibration".
For count data the standard PIT is not uniform even for a correct model. Czado, Gneiting and Held (2009) give the
randomised PIT, which is uniform under calibration, and discuss proper scores for counts. Kolassa (2016)
shows for retail count forecasts that point-error measures can favour wrong distributions, and recommends
proper scoring rules. For comparing forecasts, Diebold and Mariano (1995) test equal predictive accuracy on loss
differentials. With dependent observations, block resampling (Künsch, 1989) keeps the dependence structure.

*Relevance.* Phase 2 reports pinball loss (proper for quantiles), randomised PIT histograms, and the Brier score
and log loss of P(Y = 0), in addition to WAPE. Significance uses a paired bootstrap with whole series as the
resampling unit, so a series' days are never treated as independent draws.

## 20. Phase 2 comparison table

| Approach | Zeros handled by | Output | Hybrid type | In ForecastKit |
|---|---|---|---|---|
| DeepAR (Salinas et al., 2020) | NB likelihood only | sample paths | - | yes (P1, v2) |
| MQ-RNN / TFT | quantile loss | quantiles | - | no |
| Deep renewal processes (Türkmen et al., 2021) | RNN interval model | sample paths | DL+DL decomposition | no (KM renewal baseline instead) |
| ES-RNN (Smyl, 2020) | - | point / intervals | statistical + DL | no |
| Deep state space (Rangapuram et al., 2018) | Gaussian SSM | distribution | DL -> statistical | no |
| ZI models (Lambert, 1992) | mixture with extra zero state | distribution | - | ablation `zi_naive` |
| Hurdle models (Mullahy, 1986) | separate binary process | distribution | - | **fusion of the hybrid** |
| Kalman-LightGBM (ours, P1) | - | quantiles | statistical features -> ML | yes |
| DeepAR + zero-run hurdle (ours, P2) | LightGBM zero model + DeepAR positive part | distribution | ML + DL (distribution level) | yes |

## 21. Gap addressed in Phase 2

Published intermittent-demand deep models learn zeros either inside the count likelihood (DeepAR) or with a
dedicated neural interval model (deep renewal processes). They assume the recorded zeros are demand. When long
zero runs may be censoring (stockouts, delisting), a masked likelihood is needed to learn demand, but then
the model no longer predicts recorded zeros. ForecastKit Phase 2 tests whether a separately trained,
calibrated zero-run persistence model, joined to a masked DeepAR through a hurdle, can forecast recorded sales
without giving up the latent-demand model. It also checks whether the gain is more than the effect of simply not
masking. The answer is reported in `results/phase2/` and in the report, including where it is negative.

## 22. Phase 2 references

Bandara, K., Bergmeir, C., & Smyl, S. (2020). Forecasting across time series databases using recurrent neural networks on groups of similar series: A clustering approach. *Expert Systems with Applications*, 140, 112896. https://doi.org/10.1016/j.eswa.2019.112896

Czado, C., Gneiting, T., & Held, L. (2009). Predictive model assessment for count data. *Biometrics*, 65(4), 1254-1261. https://doi.org/10.1111/j.1541-0420.2009.01191.x

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263. https://doi.org/10.1080/07350015.1995.10524599

Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *Journal of the Royal Statistical Society: Series B*, 69(2), 243-268. https://doi.org/10.1111/j.1467-9868.2007.00587.x

Hewamalage, H., Bergmeir, C., & Bandara, K. (2021). Recurrent neural networks for time series forecasting: Current status and future directions. *International Journal of Forecasting*, 37(1), 388-427. https://doi.org/10.1016/j.ijforecast.2020.06.008

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735

Jeon, Y., & Seong, S. (2022). Robust recurrent network model for intermittent time-series forecasting. *International Journal of Forecasting*, 38(4), 1415-1425. https://doi.org/10.1016/j.ijforecast.2021.07.004

Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*, 53(282), 457-481. https://doi.org/10.1080/01621459.1958.10501452

Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217-1241. https://doi.org/10.1214/aos/1176347265

Lambert, D. (1992). Zero-inflated Poisson regression, with an application to defects in manufacturing. *Technometrics*, 34(1), 1-14. https://doi.org/10.2307/1269547

Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764. https://doi.org/10.1016/j.ijforecast.2021.03.012

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR)*. arXiv:1711.05101

Mullahy, J. (1986). Specification and testing of some modified count data models. *Journal of Econometrics*, 33(3), 341-365. https://doi.org/10.1016/0304-4076(86)90002-3

Prechelt, L. (1998). Early stopping - but when? In *Neural Networks: Tricks of the Trade* (LNCS 1524, pp. 55-69). Springer. https://doi.org/10.1007/3-540-49430-8_3

Rangapuram, S. S., Seeger, M. W., Gasthaus, J., Stella, L., Wang, Y., & Januschowski, T. (2018). Deep state space models for time series forecasting. *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*.

Smyl, S. (2020). A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting. *International Journal of Forecasting*, 36(1), 75-85. https://doi.org/10.1016/j.ijforecast.2019.03.017

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(56), 1929-1958.

Türkmen, A. C., Januschowski, T., Wang, Y., & Cemgil, A. T. (2021). Forecasting intermittent and sparse time series: A unified probabilistic framework via deep renewal processes. *PLOS ONE*, 16(11), e0259764. https://doi.org/10.1371/journal.pone.0259764

Wang, Y., Smola, A., Maddix, D., Gasthaus, J., Foster, D., & Januschowski, T. (2019). Deep factors for forecasting. *Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97*. arXiv:1905.12417

Wen, R., Torkkola, K., Narayanaswamy, B., & Madeka, D. (2017). A multi-horizon quantile recurrent forecaster. arXiv:1711.11053 (NIPS 2017 Time Series Workshop).
