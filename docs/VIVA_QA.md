# Viva / Q&A cheat sheet

**Why a Kalman filter for demand?** It is a cheap, interpretable local model that handles missing days natively and gives honest uncertainty; a diffuse prior widens intervals on short histories.

**Why not just ARIMA / ETS?** The local-level plus seasonal state space is the same family as ETS; the filter form lets censored days be skipped as missing observations and gives batched numpy speed.

**How do you stop leakage?** The context handed to models contains no target after the origin; masks are recomputed on the history slice; tests scramble the future and assert unchanged forecasts for every model.

**What is censored demand and how did you handle it?** Zero sales caused by an empty shelf. M5 has no inventory, so a zero-run rule against the item's own ADI gives a proxy flag. Flagged days are missing data for training, never imputed, and results are reported with and without them.

**Why is WAPE aggregated?** A mean of per-series ratios is dominated by tiny series; summing errors over summing sales weights by volume.

**Why pinball loss?** It is a proper scoring rule for quantiles, so it rewards calibrated and sharp intervals; wQL normalises it by sales.

**Why is calibration shown as a band?** Counts have probability mass at zero, so P(y ≤ q) is inflated; a calibrated quantile lies between P(y < q) and P(y ≤ q).

**Why would a global model win (or lose)?** It shares strength across series and handles cold start through embeddings; it can lose on stable, long-history series where a local filter is already sharp. Say which happened from the by-slice table.

**How is cold start tested?** Simulated: about 7% of series reveal only their last 28 days; global models use the unknown-item embedding, local models use a diffuse prior.

**Is the difference real?** Paired bootstrap over series versus seasonal naive; report the interval, not just the mean.

**What does the toolkit not do?** No hierarchical reconciliation, no real inventory data, no GPU testing in the build environment, one dataset family tested at scale (M5-format) plus a small CSV proof.

**What would you do next?** Hierarchical reconciliation, real stockout labels, a promo-aware price model, longer DeepAR training with multiple seeds.
