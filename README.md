# ForecastKit: leak-safe probabilistic demand forecasting on M5

**Research question.** On sparse, partly censored retail demand, how do a local state-space model (Kalman filter), a tabular model (LightGBM), a hybrid of the two, and a global deep model (DeepAR) compare when they all use the same data, origins, masks and scoring?

**Current deliverables**

| | File |
|---|---|
| Research paper (IEEE format, 8 pages) | [`paper/forecastkit_phase1.pdf`](paper/forecastkit_phase1.pdf) |
| Presentation (6 slides, PDF) | [`slides/final_presentation.pdf`](slides/final_presentation.pdf) |
| Evidence behind every number | [`docs/PHASE1_EVIDENCE_INDEX.md`](docs/PHASE1_EVIDENCE_INDEX.md) |

The final presentation was exported from slide software (PDF metadata: PptxGenJS, LibreOffice). Its editable source is not in the repository. Older PDFs with synthetic numbers are in [`archive/superseded/`](archive/superseded/README.md) and should not be graded.

## Models compared

| Name in results | Model | Code |
|---|---|---|
| `snaive4` | 4-week seasonal naive (reference baseline) | `src/forecastkit/models/naive.py` |
| `kalman` | Per-series local level + stochastic weekly seasonal on the √y scale | `src/forecastkit/models/kalman.py` |
| `lgbm` | LightGBM, direct by 7-day horizon bucket, Poisson mean + 7 quantile models | `src/forecastkit/models/lgbm.py` |
| `kalman_lgbm` | LightGBM plus 3 Kalman-state features (the hybrid) | `src/forecastkit/models/lgbm.py` |
| `deepar` | Global LSTM with negative binomial output (own PyTorch implementation) | `src/forecastkit/models/deepar.py` |
| `ens_global_local` | Equal-weight quantile average of kalman, lgbm and deepar | `src/forecastkit/evaluation/ensemble.py` |

The naive, seasonal naive and 28-day moving average baselines are also run. Croston, SBA, TSB, ETS and ARIMA appear only in the literature review; they are not implemented.

## Workflow

No assignment rubric is stored in the repository. The steps below follow the implemented pipeline. The five Phase 1 criteria used in `docs/PHASE1_EVIDENCE_INDEX.md` (literature review, dataset quality and EDA, feature engineering, theoretical rigor, model application) come from the earlier task brief, not from a repo file.

Commands assume an activated environment with the package installed (step 1) and are run from the repo root.

| # | Step | Purpose | Source files | Command | Output | Status |
|---|---|---|---|---|---|---|
| 1 | Environment | Install the package and its dependencies. | `pyproject.toml` | `pip install -e ".[all]"` | `forecastkit` CLI | Done |
| 2 | Data and frozen subset | Load the M5 files and fix the 1,200 series (400 items × CA_1, TX_1, WI_1, stratified, seed 42). | `configs/m5.yaml`, `src/forecastkit/datasets/m5.py`, `datasets/subset.py` | `forecastkit subset configs/m5.yaml` | `configs/subsets/m5_subset.csv` (committed; reused if present, never overwritten) | Done |
| 3 | Preparation and masks | Availability mask, leading-zero mask and stockout heuristic, recomputed on the history slice at each origin. | `src/forecastkit/masks.py`, `features/intermittency.py`, `backtest/runner.py` | runs inside step 7 | cache in `data/processed/` | Done |
| 4 | EDA | Descriptive statistics and figures for the subset. | `scripts/eda_m5.py` | `make eda` (about 6 s) | `outputs/eda/*.csv`, `docs/figures/eda/*.png`, write-up in `docs/EDA_FINDINGS.md` | Done |
| 5 | Features and leakage controls | Lags, rolling statistics, calendar, SNAP, price and Kalman-state features, all computed from data up to the origin. | `src/forecastkit/features/`, `backtest/context.py`, `tests/test_leakage.py` | runs inside step 7 | `docs/FEATURE_ENGINEERING.md` | Done (feature importance not produced) |
| 6 | Model configuration | One `fit(ctx)` / `predict(quantiles)` interface; hyperparameters in YAML. | `src/forecastkit/models/`, `configs/m5.yaml` | n/a | n/a | Done (defaults, not tuned) |
| 7 | Rolling-origin backtest | 4 test origins (2016-01-31 to 2016-04-24), 28-day horizon, 7 quantiles, simulated cold start for about 7% of series. | `src/forecastkit/backtest/splitter.py`, `runner.py`, `cli.py` | `forecastkit all configs/m5.yaml` (about 19 min on CPU) | `outputs/m5_main/forecasts.parquet`, `timing.csv`, `config.yaml` | Done |
| 8 | Evaluation and report tables | Metrics on all days, non-censored days, cold start, by slice, by origin; paired bootstrap against `snaive4`. | `src/forecastkit/evaluation/`, `report/build.py`, `viz/plots.py` | part of step 7; alone: `forecastkit evaluate configs/m5.yaml` then `forecastkit report configs/m5.yaml` | `outputs/m5_main/report/*.csv`, `RESULTS.md`, figures | Done |
| 9 | Ablations | Change one design choice at a time; evaluation labels frozen across variants. | `src/forecastkit/backtest/ablation.py`, `configs/ablations/` | `forecastkit ablate configs/ablations/kalman.yaml`, `forecastkit ablate configs/ablations/deepar_quick.yaml` | `outputs/ablation_kalman/`, `outputs/ablation_deepar_quick/` | Partly done (see below) |
| 10 | Tests | Leakage, metrics, splits, masks, model contract, CSV adapter, CLI end to end. | `tests/` | `make test` | 35 tests | Done |
| 11 | Paper | Build the IEEE paper from LaTeX. | `paper/forecastkit_phase1.tex`, `paper/references.bib` | `make paper` (needs Tectonic) | `paper/forecastkit_phase1.pdf` | Done |
| 12 | Presentation | Final 6-slide deck, added as a PDF. The earlier 12-slide HTML deck can still be rebuilt. | `slides/final_presentation.pdf`; earlier deck: `slides/deck.html`, `build_charts.py`, `build.sh` | none for the final deck; `make slides` rebuilds the earlier deck (needs Chrome) | `slides/final_presentation.pdf` | Done |

`make eda`, `make paper` and `make slides` do not train models. Steps 7 to 9 retrain and take minutes to hours.

## Repository map

```
configs/              m5.yaml (main run), m5_smoke.yaml, example_csv.yaml
  ablations/          kalman, deepar, deepar_quick (the 2 variants run), stockout, lgbm
  subsets/            m5_subset.csv (frozen M5 subset), m5_synth_subset.csv
src/forecastkit/      the reusable package
  datasets/           M5, generic CSV and synthetic adapters; subset selection
  masks.py            availability and stockout masks
  features/           calendar, intermittency classes, tabular features
  models/             naive, kalman, lgbm (+ hybrid), deepar, base interface
  backtest/           origins, leak-proof context, runner, ablation, tuning
  evaluation/         metrics, evaluator, ensemble, bootstrap
  report/, viz/       RESULTS.md and figures
  cli.py              the `forecastkit` command
  api/                FastAPI server (demo app, not part of the evaluation)
scripts/              eda_m5.py (step 4), make_diagrams.py (docs/architecture.png, rolling_origin.png)
tests/                regression tests (35)
docs/                 method, EDA, feature, results and evidence documents (see below)
paper/                LaTeX source, bibliography and compiled paper
slides/               final_presentation.pdf (current); earlier HTML deck source, chart and build scripts
reports/example_synthetic_run/   synthetic demo outputs used by the Streamlit app and API
archive/superseded/   old PDFs, the earlier 12-slide deck and the old report draft, kept for provenance
app.py, frontend/, hf_spaces/    Streamlit studio, React site and Hugging Face deployment (demo only)
```

Documentation in `docs/`:

| File | Content |
|---|---|
| `DELIVERABLES.md` | current paper and slides, and how they are built |
| `PHASE1_EVIDENCE_INDEX.md` | each Phase 1 criterion mapped to its evidence and gaps |
| `ML_RESULTS_AND_ABLATIONS.md` | full results and ablation tables with sources |
| `EDA_FINDINGS.md` | EDA write-up |
| `FEATURE_ENGINEERING.md` | every feature, its source and its leakage control |
| `THEORY_AND_METHODS.md` | equations as implemented |
| `LITERATURE_REVIEW.md` | 19 verified references |
| `RUN_REAL_M5.md` | step-by-step guide to reproduce the M5 run |
| `ACADEMIC_REPORT.md` | Markdown version of the paper (generated, do not edit) |
| `SLIDES_OUTLINE.md` | slide-by-slide outline with sources |
| `PHASE1_CHANGELOG.md` | what changed in each Phase 1 stage |
| `METHODS.md`, `M5_FINDINGS.md`, `DECISIONS.md`, `VIVA_QA.md` | earlier notes; the paper and the files above take precedence |

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"     # numpy, pandas, scipy, lightgbm, torch, pytest, ...
make test                   # expect 35 passed
```

`requirements.txt` lists the extra packages for the Streamlit app and API. It is not needed for the experiments.

## Data

M5 is not redistributed. Download *M5 Forecasting - Accuracy* from Kaggle and put `calendar.csv`, `sell_prices.csv` and `sales_train_evaluation.csv` in `data/raw/m5/`. `data/raw/` is git-ignored. The frozen subset `configs/subsets/m5_subset.csv` is committed, so every run uses the same 1,200 series. Full instructions are in [`docs/RUN_REAL_M5.md`](docs/RUN_REAL_M5.md).

Without the Kaggle data you can still run the tests and the synthetic demo (`make smoke`, synthetic data in M5 file format). Synthetic results are not M5 results.

## Commands

```bash
make test                                          # regression tests
make eda                                           # EDA tables and figures (needs data/raw/m5)
forecastkit all configs/m5.yaml                    # main benchmark, about 19 min (needs data/raw/m5)
forecastkit ablate configs/ablations/kalman.yaml   # Kalman ablation (7 variants)
forecastkit ablate configs/ablations/deepar_quick.yaml  # DeepAR ablation (2 variants), about 11 min
make paper                                         # paper PDF (needs Tectonic)
make slides                                        # earlier 12-slide HTML deck -> archive/ (needs Chrome and outputs/m5_main/report/)
python3 paper/tex_to_md.py                         # regenerate docs/ACADEMIC_REPORT.md
```

## Findings (real M5 run, 2026-10-29)

Established series, all days, averaged over 4 origins. Lower is better for every column except coverage (target 0.90). Source: `outputs/m5_main/report/main_all_days.csv`.

| Model | MAE | RMSE | WAPE | Pinball | Cov90 |
|---|---|---|---|---|---|
| snaive4 | 0.988 | 1.901 | 0.772 | 0.2899 | 0.914 |
| kalman | 0.924 | 1.917 | 0.723 | 0.2736 | 0.914 |
| lgbm | 0.907 | 1.799 | 0.709 | 0.2637 | 0.941 |
| kalman_lgbm | 0.900 | 1.798 | 0.704 | 0.2620 | 0.940 |
| deepar | 0.891 | 1.790 | 0.697 | 0.2619 | 0.965 |
| ens_global_local | 0.899 | 1.782 | 0.703 | 0.2623 | 0.919 |

- Every learned model beats `snaive4` on WAPE and pinball. The paired bootstrap interval over series excludes 0 in every case.
- DeepAR has the lowest all-days WAPE and MAE.
- On pinball, DeepAR, the hybrid and the ensemble are tied. They differ by 0.0004, against a standard deviation of about 0.004 across origins.
- The ensemble has the lowest RMSE (1.782). No model is best on every metric.
- On non-censored days the hybrid is best (WAPE 0.660, pinball 0.269). On simulated cold start the hybrid has the lowest WAPE (0.760); LightGBM and DeepAR are both 0.773.
- DeepAR's Cov90 of 0.965 is above target, partly because inclusive bounds count ties at zero as covered. A discrete-aware check puts DeepAR inside the calibration band at all 7 quantile levels, but its intervals are still wide.

Per-setting tables, bootstrap intervals and ablations are in [`docs/ML_RESULTS_AND_ABLATIONS.md`](docs/ML_RESULTS_AND_ABLATIONS.md).

## Completed and not run

| Completed | Not run |
|---|---|
| Main benchmark: 8 models + ensemble, 4 origins, 1,200 series | Hyperparameter tuning (the code exists; defaults were used) |
| Evaluation on all days, non-censored days, cold start, slices, origins; paired bootstrap | Multiple seeds (all results use seed 42) |
| Kalman ablation, 7 variants | LightGBM ablation (`configs/ablations/lgbm.yaml`) |
| DeepAR ablation, 2 variants: no loss masking, Gaussian likelihood | Stockout-rule ablation (`configs/ablations/stockout.yaml`) |
| EDA, feature documentation, literature review | Remaining DeepAR variants (no item id, no price covariates, context 56/112) |
| Paper and slides | LightGBM feature importance |
| | Croston, SBA, TSB, ETS and ARIMA baselines |

## Limitations and assumptions

- M5 has no inventory records, so the stockout flag is a heuristic (a zero run of length at least max(28, 3·ADI)), not ground truth.
- Cold start is simulated by hiding history of existing items. It is not a real product launch.
- Price features assume prices are known in advance. The whole-window mean log-price fill is not strictly causal, although those cells are never scored.
- The SNAP and event effects in the EDA are descriptive associations over the full window, including test days. They are not used as model input and are not causal estimates.
- One retailer, 3 stores, 1,200 series, single seed, no tuning.
- The Kalman model is Gaussian on the √y scale, which fits very intermittent series poorly.
- DeepAR is a compact in-repo implementation, not GluonTS. Only the CPU path was run.

## Reproducibility notes

- `data/raw/`, `data/processed/` and `outputs/` are git-ignored. The result CSVs behind the paper and slides exist only on the machine that ran them. To rebuild them, run steps 7 to 9 with the M5 data.
- Every run saves its exact configuration in `outputs/<run>/config.yaml`. Seeds are set in `configs/m5.yaml` (run seed 42, subset seed 42, cold-start seed 7).
- External tools: Tectonic (or TeX Live) for the paper, and Google Chrome or Chromium for the earlier HTML deck.
- `make slides` regenerates the charts from `outputs/m5_main/report/`, so it fails without those CSVs.

## Demo applications (not part of the evaluation)

`make streamlit` (Streamlit studio), `make api` (FastAPI server), `make site` (React site in `frontend/`) and `hf_spaces/` (Hugging Face deployment) demonstrate the toolkit on the synthetic example run in `reports/example_synthetic_run/`. They do not show M5 results.

## Extending the toolkit

A dataset needs one adapter that returns a `Dataset(Y, avail, K, static)` (see `src/forecastkit/datasets/csv_adapter.py`, or use `type: csv` in YAML as in `configs/example_csv.yaml`). A model needs one class with `fit(ctx)` and `predict(quantiles)` (see `tests/helpers.py` for a 10-line example). Architecture diagram: [`docs/architecture.png`](docs/architecture.png). Backtest protocol: [`docs/rolling_origin.png`](docs/rolling_origin.png).

MIT licensed.
