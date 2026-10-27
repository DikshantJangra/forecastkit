# Design decisions (what, why, trade-off)

| Decision | Why | Trade-off |
|---|---|---|
| One Dataset contract + adapters | Reuse on any retail data; only adapters know file formats | A new source needs an adapter (≈10 lines, or a YAML column map for CSV) |
| `fit(ctx)` / `predict(quantiles)` only | A new model is one file; all models are compared identically | Models cannot request extra data outside the context |
| Leak-proof context by construction | Leakage is the most common backtest error; make it impossible, then test it | Slightly more copying at each origin |
| One long forecast file, evaluator reads only it | Decouples modelling from scoring; foreign forecasts can be scored | Larger files than wide arrays |
| Mark, never impute, stockouts | Imputing invents data; marking keeps every model honest | The stockout label is a proxy and can be wrong |
| Evaluation stockout labels frozen across ablations | Variants are scored on identical days | Hindsight labels are used for scoring only, never training |
| Kalman on √ scale with concentrated variance | Stabilises counts; few parameters per series, batched in numpy | Gaussian likelihood is weak on very intermittent items |
| LightGBM direct by horizon bucket | Avoids recursive error build-up; fast | 4 buckets × (mean + quantile models) to train |
| Hybrid Kalman→LightGBM | Gives the tabular model a state-space prior | Features are only as good as the filter |
| Own compact PyTorch DeepAR | No GluonTS dependency; full control of masking and embeddings | Less battle-tested than a library implementation |
| Quantile-average ensemble | Simple, calibrated-friendly, post-hoc | Not trained jointly |
| Tune on the validation origin only | Keeps test origins clean | Tuned parameters may drift by the last origin |
| Calibration as a band (P(y<q), P(y≤q)) | Counts have mass at zero, so one curve misleads | Harder to read than a single line |
