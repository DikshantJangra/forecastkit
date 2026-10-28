# Slides outline (the published deck follows this)

> The published PDF deck still shows synthetic-run numbers and several unsupported ablation percentages (34%, 31.4%, 18.2%, 18.7%, 14.2%). Use only the verified values in `docs/ML_RESULTS_AND_ABLATIONS.md` and `docs/M5_FINDINGS.md` when rebuilding it.

1. Cover: forecastkit
2. Problem: ordering is a bet on a range
3. Data and experiment design
4. forecastkit architecture: two contracts
5. Rolling origins, no peeking
6. Stockouts, cold start, uncertainty
7. Three model families, one interface
8. Add a model or dataset in about ten lines
9. How every model is scored
10. M5 results (source: `outputs/m5_main/report/RESULTS.md`; DeepAR best WAPE 0.697, pinball tie with hybrid and ensemble at 0.262)
11. Example report figure (synthetic, labelled)
12. Ablations
13. Limitations
14. What you take away and reuse

Real run done (2026-10-29): replace slide 10's table from `outputs/m5_main/report/RESULTS.md`, add by-horizon, cold-start and calibration figures, and fill slide 12 only with completed ablations (Kalman 7 variants, DeepAR no_loss_masking and gaussian_likelihood). Speaker notes are in the deck.
