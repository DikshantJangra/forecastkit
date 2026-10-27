# Academic Report & Submission Draft

The complete, academic-level submittable research report is available in two formats:
1. **Full Academic Paper (PDF):** [ForecastKit_Academic_Report.pdf](ForecastKit_Academic_Report.pdf) (also mirrored at `reports/ForecastKit_Academic_Report.pdf`).
2. **Executive Presentation Slides (PDF):** [Inventory Demand Forecasting Guide.pdf](Inventory%20Demand%20Forecasting%20Guide.pdf) (also mirrored at [ForecastKit_Presentation.pdf](ForecastKit_Presentation.pdf)).
3. **Full Markdown Source:** [ACADEMIC_REPORT.md](docs/ACADEMIC_REPORT.md).

### Title:
**Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models: A Leak-Proof, Probabilistic Benchmark on M5 Hierarchical Retail Data**

### Context:
Anonymous Academic Submission · Advanced Machine Learning & Deep Learning Project Report (October 2026)

### Contents:
1. Introduction & Operational Context (Newsvendor critical fractile, CSL 90%, asymmetric inventory losses, demand censoring).
2. The `forecastkit` Architecture & Experimental Protocol (Canonical contract, stockout proxy rules, perturbation proofs).
3. Model Families & Mathematical Formulations (Kalman filter with concentrated likelihood, direct LightGBM, DeepAR with negative binomial emissions, quantile ensemble).
4. Evaluation Metrics & Statistical Protocol (Aggregated WAPE, pinball loss, discrete calibration bounds, paired bootstrap significance).
5. Empirical Results & Comparative Analysis (Tables 1-3, Figures 1-11 covering horizons, slices, cold-start, latent states, and runtime profiling).
6. Discussion & Practical Recommendations.
7. Software Engineering & Reproducibility (35 unit tests, leak-free verification).
8. References.
