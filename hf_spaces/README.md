---
title: ForecastKit - Probabilistic Demand Forecasting Toolkit
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.40.0"
app_file: app.py
pinned: false
license: mit
---

# ForecastKit: Probabilistic Demand Forecasting Studio

An open, model-agnostic toolkit for probabilistic demand forecasting — leak-safe rolling-origin backtesting, fair stockout handling, and apples-to-apples evaluation across:
- **Local State-Space**: Kalman Filter per SKU (with local level and stochastic weekly seasonal)
- **Tabular GBDT**: LightGBM direct multi-horizon models
- **Hybrid**: Kalman → LightGBM with state priors
- **Global Deep Learning**: DeepAR Autoregressive LSTM with Negative Binomial likelihood and item dropout for cold start
- **Baselines**: Seasonal Naive, Moving Average, Quantile Ensembles

## Features in this Space
- 🎯 **Interactive Forecaster Playground**: Live multi-quantile fan charts ($p_{10}, p_{25}, p_{50}, p_{75}, p_{90}$) and ground truth comparisons.
- 🔄 **Leak-Safe Rolling Backtest Studio**: Rolling evaluation folds with stockout censorship masking and simulated cold start.
- 📊 **9 Diagnostic Reports**: Error vs Horizon, Calibration reliability bounds, Pinball loss curves, Syntetos-Boylan demand slices.
- 🛠️ **YAML Config Studio**: Visual config builder and instant export.
- 📐 **Fairness Guarantees**: Formal leak-proofing and mathematical formulas.
