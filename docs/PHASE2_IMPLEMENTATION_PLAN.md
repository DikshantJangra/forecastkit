# Phase 2 Implementation Plan (living document)

Status markers: [x] done, [~] in progress, [ ] not started, [-] dropped (reason given).
Progress details: `docs/PHASE2_PROGRESS.md`. Every experiment: `docs/PHASE2_EXPERIMENT_LEDGER.md`.

## 1. Repository baseline (at the start of Phase 2)

- Branch `main`, HEAD `929f654` ("Use the 6-slide final presentation; archive the earlier HTML deck"), clean tree.
- Package `src/forecastkit` (~4.2k lines): adapters (`datasets/`), leak-safe context builder (`backtest/context.py`),
  rolling-origin runner (`backtest/runner.py`), models (`models/`), evaluator (`evaluation/`), report builder.
- Data: M5 subset, 3 stores (CA_1, TX_1, WI_1) x 400 items = 1,200 series, last 1,000 days.
  Validation origin 2016-01-03; test origins 2016-01-31, 2016-02-28, 2016-03-27, 2016-04-24; horizon 28.
  ~7% of series (84) are *simulated* cold-start (only the last 28 days are exposed).
- Tests: 35 passing (`pytest -q`). CI only deployed the frontend; it never ran tests.
- Outputs are gitignored (`outputs/*`): no result files were tracked.

## 2. Existing implementation inventory

| Component | File | Notes |
|---|---|---|
| Naive / seasonal naive / MA | `models/naive.py` | baselines |
| Kalman (local level + weekly seasonal) | `models/kalman.py` | per-series, sqrt transform, grid noise params |
| LightGBM direct multi-horizon + quantiles | `models/lgbm.py` | horizon buckets of 7 days |
| Kalman-LightGBM "hybrid" | `models/lgbm.py` | Kalman state as LightGBM features; ML+ML, not ML+DL |
| DeepAR (LSTM, NB likelihood) | `models/deepar.py` | masked loss, item embeddings, warm start; no validation/early stopping; only y_{t-1} lag |
| Stockout proxy | `masks.py` | long zero runs (>= max(28, 3 ADI)); recomputed on history per origin; NOT ground truth |
| Evaluator | `evaluation/evaluator.py` | WAPE/MAE/RMSE/pinball/WQL/coverage, slices, bootstrap vs ONE reference |

## 3. Verified Phase 1 results (from `outputs/m5_main/report/tables`, 4 test origins, warm series, all days)

| Model | WAPE | Pinball |
|---|---:|---:|
| snaive4 | 0.77242 | 0.28990 |
| kalman | 0.72261 | 0.27363 |
| lgbm | 0.70935 | 0.26366 |
| kalman_lgbm | 0.70387 | 0.26198 |
| deepar | 0.69664 | 0.26191 |
| ens_global_local | 0.70303 | 0.26231 |

Phase 1 ablation: unmasked DeepAR scores better on all days (WAPE 0.677, pinball 0.2565) because recorded
zeros inside long zero runs are learned as "demand". This is the observation that motivates Phase 2C/2D.

## 4. Missing Phase 2 requirements (from the audit)

1. No ML+DL hybrid. 2. DeepAR has no validation split, early stopping, checkpointing, saved curves or
seasonal lags. 3. Single seed. 4. Significance only vs `snaive4`. 5. No PIT / zero-probability
calibration. 6. No DL literature beyond DeepAR. 7. No CI tests. 8. No tracked results. 9. Report and
slides are Phase 1 only. 10. Hardcoded user path in `app.py`.

## 5. Planned architecture

```
                    history <= origin (leak-proof ForecastContext)
                 /                                  \
  DeepAR (masked NB LSTM)                  Zero-run persistence model (LightGBM, binary)
  latent-demand path distribution          pi_h = P(Y_{t+h} = 0 | F_t), causal run features
  per-path NB params (mu, alpha)                    |
                 \                                  /
                  Hurdle fusion (Rao-Blackwellised):
                    P(Y=0)       = pi_h
                    P(Y=y | y>0) = p_D(y) / (1 - p_D(0))      (DeepAR positive-count shape kept)
                  -> quantiles, mean, samples
```

The ML component owns the *zero / non-zero* decision (where recorded zeros come from runs that DeepAR is
told to ignore). The DL component owns the *size* of positive sales. This removes the double counting of a
naive zero-inflated mixture, which is kept as an ablation.

## 6. Implementation phases

- [x] Stage 0: audit, baseline, this plan, progress log, ledger.
- [x] Stage 1: DeepAR: seasonal lags (7/14/28), causal age feature, validation window inside history,
  early stopping + best checkpoint, AdamW option, saved curves/checkpoints/samples. Defaults keep Phase 1 behaviour.
- [x] Stage 2: DeepAR variant selection on the VALIDATION origin only; then 3 seeds on test (Phase 1 vs improved).
- [x] Stage 3: zero-run persistence: causal features, target, baselines (series zero rate, recent zero rate,
  Kaplan-Meier run survival + renewal), LightGBM; Brier, log loss, ECE, reliability, AUC, PR-AUC, by run bucket.
- [x] Stage 4: `deepar_hurdle` model (registered), fusion variants, tests.
- [~] Stage 5 (partial: 5 ablation runs stopped, see PHASE2_RAPID_STATUS.md): hybrid evaluation: 3 seeds, ablations, pairwise bootstrap, PIT, slices.
- [x] Stage 6: docs, literature, theory, README, reproduction guide, results package, IEEE report.
- [x] Stage 7: slides.
- [x] Stage 8: final audit.

## 7. Experiment plan (see ledger for actual runs)

| ID | Question | Phase |
|---|---|---|
| E01 | Do the code changes reproduce Phase 1 DeepAR exactly with Phase 1 params? | test, seed 42 |
| E02 | Which DeepAR additions help (lags, age, early stopping, AdamW)? | **validation origin only** |
| E03 | Phase 1 DeepAR vs improved DeepAR, 3 seeds | test |
| E04 | Does a learned zero-run model beat non-learned baselines (Brier/log loss/calibration)? | test origins, labels only |
| E05 | DeepAR + persistence hurdle vs DeepAR, 3 seeds | test |
| E06 | Hybrid ablations: fusion (naive ZI, inflate-only ZI, gated), no persistence features, no DL, no mask | test, seed 42 |
| E07 | Pairwise bootstrap, PIT, zero-probability calibration, slices | evaluation only |

## 8. Risks

- CPU only (8 cores, 8 GB). DeepAR full run ~5 min per seed; mitigated by parallel processes.
- The stockout proxy is a heuristic. Everything is framed as *recorded zero sales*, never inventory.
- The hybrid may improve all-days (recorded sales) metrics but worsen non-censored (latent demand) metrics.
  Both are reported.
- Hindsight eval labels (`C`) are used only to define the non-censored evaluation view, never as features.

## 9. Acceptance criteria

See Section 20 of the Phase 2 brief; tracked at the end of `docs/PHASE2_PROGRESS.md`.
