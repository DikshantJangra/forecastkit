# Phase 2 Experiment Ledger

Common setup unless stated: M5 subset (1,200 series = 3 stores x 400 items, last 1,000 days), horizon 28,
quantiles {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95}, masks as in `configs/m5.yaml` (leading zeros; stockout
*proxy* zero_run min_run 28, ADI x 3, >= 10 sales). Validation origin 2016-01-03 (selection only). Test origins
2016-01-31, 2016-02-28, 2016-03-27, 2016-04-24. Metrics on warm series (84 simulated cold-start series separate),
available days. Hardware: Apple laptop, 8 CPU cores, 8 GB RAM, CPU only (`device: auto` -> cpu), 5 jobs in parallel.
Results of every run: `outputs/phase2/...` (gitignored, large); the tracked numbers are in `results/phase2/`.

| ID | Date | Purpose / hypothesis | Config | Seeds | Split | Status | Output |
|---|---|---|---|---|---|---|---|
| E01 | 2026-11-19 | Phase 2 code with Phase 1 params reproduces Phase 1 DeepAR exactly | `configs/phase2/e01_repro.yaml` | 42 | test | DONE: max abs diff 0.0 over 1,075,200 forecast values | `outputs/phase2/e01_repro_s42` |
| E02 | 2026-11-19 | Which DeepAR additions help? (early stopping/checkpoint, lags 7/14/28, age, AdamW) | `configs/phase2/e02_deepar_select.yaml` | 42 | **validation origin only** | DONE | `outputs/phase2/e02_val/*` |
| E03 | 2026-11-19 | Phase 1 DeepAR vs selected DeepAR v2, 3 seeds; masked vs unmasked | `configs/phase2/m5_phase2.yaml` | 42,43,44 | test | see below | `outputs/phase2/runs/` |
| E04 | 2026-11-19 | Does the learned zero-run model beat non-learned baselines and DeepAR's own P(0)? | `scripts/phase2_zero_model.py` | 42 | test (labels only) | see below | `results/phase2/zero_model/` |
| E05 | 2026-11-19 | DeepAR + persistence hurdle vs DeepAR, 3 seeds | `configs/phase2/m5_phase2.yaml` (`hybrid`) | 42,43,44 | test | see below | `outputs/phase2/runs/hybrid_s*` |
| E06 | 2026-11-19 | Hybrid and DeepAR ablations | `configs/phase2/m5_phase2.yaml` (`hyb_*`, `dar_v2_*`) | 42 | test | see below | `outputs/phase2/runs/` |
| E07 | 2026-11-19 | Pairwise bootstrap, PIT, zero-mass calibration, run-length and demand-class slices | `scripts/phase2_analysis.py` | - | test | see below | `results/phase2/` |

## E01: reproduction of Phase 1 DeepAR

- Command: `forecastkit run configs/phase2/e01_repro.yaml`. Runtime 6.5 min (5 other jobs in parallel).
- Result: all 1,075,200 forecast values (1,200 series x 4 origins x 28 h x 8 stats) identical to
  `outputs/m5_main/forecasts.parquet` (max abs diff 0.0). Phase 1 numbers remain authoritative and are reused
  as `deepar_p1@s42`.
- Side finding (bug fixed): `datasets/synthetic.py` iterated over a Python `set`, so the *synthetic* test fixture
  depended on `PYTHONHASHSEED`. Fixed with `sorted(...)`. M5 data and all Phase 1 M5 results are unaffected.

## E02: DeepAR design selection (validation origin 2016-01-03 only)

Pre-declared rule: lowest mean pinball on warm, non-censored validation cells; differences < 0.5% count as ties
and go to the simpler model. Each variant trained from scratch (max 2,500 steps; eval every 100 steps, patience 5).

| variant | WAPE all | pinball all | WAPE non-cens | **pinball non-cens** | best step |
|---|---:|---:|---:|---:|---:|
| deepar_p1 (Phase 1) | 0.73047 | 0.25517 | 0.68499 | 0.27505 | (no validation) |
| + validation checkpoint (es) | 0.72897 | 0.25569 | 0.68298 | 0.27681 | 2300 |
| + lags 7/14/28 | 0.73106 | 0.25499 | 0.68077 | 0.27283 | 2100 |
| + age | 0.73241 | 0.25496 | 0.68036 | 0.27219 | 2500 |
| + AdamW (wd 0.01) | 0.73303 | 0.25486 | 0.67932 | **0.27208** | 2500 |

Interpretation and decision: the validation checkpoint alone did not help on this origin (non-censored pinball
0.27681 vs 0.27505 for Phase 1). Lags give the largest single gain (-1.4% non-censored pinball vs `es`). Age (-0.23%)
and AdamW (-0.04%) are inside the 0.5% tie band, so the simpler **es + lags** model is selected as `deepar_v2`.
Early stopping never triggered (best step 2,100-2,500 of 2,500): the step budget, not over-fitting, is binding; the
validation block acts as a best-checkpoint selector. Age and AdamW are still reported on test as ablations
(`dar_v2_age`, `dar_v2_age_adamw`), without changing the selection. One validation origin is a small sample:
this is a selection protocol, not evidence of improvement (that is E03).

## E03-E07 (executed; outcomes)

| ID | Runs completed | Runtime | Outcome |
|---|---|---|---|
| E03 | deepar_p1 (s42 = E01, s43, s44), deepar_v2 (s42-44), deepar_v2_nomask (s42-44) | ~10 min per run, 5 in parallel | v2 vs P1: WAPE -0.0035 (sig.), pinball -0.0006 (n.s.). Unmasking: better all days, worse non-censored (both sig.) |
| E04 | zero model, no-persistence zero model, 3 baselines, DeepAR P(0) x3 | ~4 min | LightGBM beats all baselines and masked DeepAR P(0); ties unmasked DeepAR; run features add nothing |
| E05 | hybrid s42, s43, s44 | ~12 min per run | best all-days WAPE/pinball (0.6707 / 0.2533); beats unmasked DeepAR on both views; slightly worse than masked DeepAR on non-censored |
| E06 | hyb_zi_naive, hyb_zi_inflate, hyb_gated, hyb_no_dl (s42) | ~12 min each | all significantly worse than the hurdle on all days |
| E06 (stopped) | hyb_none, hyb_no_persist, hyb_nomask, hyb_p1, dar_v2_no_val killed in rapid mode; dar_v2_age, dar_v2_age_adamw never started | - | **no results; not reported** |
| E07 | analysis of all completed runs | 2 min, 2 GB RAM | `results/phase2/` |

Bootstrap: 1,000 series resamples in the tracked package (the E04 baseline bootstrap uses 2,000).
