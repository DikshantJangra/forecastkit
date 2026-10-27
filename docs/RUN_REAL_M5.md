# Running the REAL M5 experiment (the part that produces your report numbers)

The library is built and tested; the numbers in your report/slides must come from this run on the real data.

## 0. Setup (once)

```bash
git clone <your repo> && cd forecastkit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
make test                                   # must be green before you trust anything
```
GPU (Colab/Kaggle) is optional: only DeepAR uses it (`device: auto` picks CUDA when present). Everything else is CPU.

## 1. Data

1. Kaggle → *M5 Forecasting – Accuracy* → accept the rules → download.
2. Put these three files in `data/raw/m5/`:
   `calendar.csv`, `sell_prices.csv`, `sales_train_evaluation.csv`
   (CLI alternative: `kaggle competitions download -c m5-forecasting-accuracy`).
   Do **not** commit them (`.gitignore` already blocks `data/`).

## 2. Freeze the subset (do this once, commit the CSV)

```bash
forecastkit subset configs/m5.yaml
```
This picks ~400 items × 3 stores (CA_1, TX_1, WI_1) **stratified by department × Syntetos–Boylan class**, seed 42, and writes `configs/subsets/m5_subset.csv`. It prints the class counts and the origin dates. Commit the CSV — every later experiment reuses it, which is what makes the comparison fair and reproducible.

## 3. Smoke-test the real data on a tiny slice first (5 min)

```bash
forecastkit run configs/m5.yaml --models snaive4,kalman --set dataset.params.items_per_store=40 \
      --set dataset.params.subset_file=/tmp/m5_tiny.csv --set run_name=m5_tiny
```
Check the log: series count, % of cells available, origins' dates (last origin's window must end at `d_1941`).

## 4. Full run

```bash
forecastkit all configs/m5.yaml
```
Rough CPU wall-times for ~1,200 series (4 test origins) — measure on your machine and quote **your** numbers:

| model | cost driver | expect |
|---|---|---|
| naive family, Kalman | seconds | seconds–minutes |
| LightGBM, hybrid | 8 horizon-buckets × 8 models; `max_rows` caps data | ~5–10 min **per origin** on 4 cores |
| DeepAR | `steps` (first origin) + `finetune_steps` (later origins) | ~15–40 min CPU total; far less on a GPU |

Run models separately if you want (`make baselines | kalman | lgbm | deepar`) — `forecasts.parquet` is merged, not overwritten.

**If time is short** (cut in this order): items_per_store 400→250 → DeepAR steps 2500→1200 → LightGBM `n_estimators` 200→120 → `n_origins` 4→3. Say so in the report.

## 5. Results

`outputs/m5_main/report/RESULTS.md` holds every table (main, non-censored, by horizon, by slice, cold start, significance, compute cost, calibration) and `report/figures/*.png` the 9 figures. Paste from there. Never type numbers by hand.

## 6. Ablations & sensitivity (these become the "Ablations" section)

```bash
forecastkit ablate configs/ablations/deepar.yaml     # item_id, price covariates, likelihood, context length, loss masking
forecastkit ablate configs/ablations/stockout.yaml   # 28-day rule vs 14-day rule vs no masking (evaluation labels frozen)
forecastkit ablate configs/ablations/kalman.yaml     # transform, shared vs per-series noise, window, masking
forecastkit ablate configs/ablations/lgbm.yaml       # objective, price features, bucket size, item id
```
Each writes `outputs/ablation_*/ablation_table.csv` (all-days **and** non-censored-day columns).

Hyper-parameters (optional, validation origin only): `forecastkit tune configs/m5.yaml --model lgbm --trials 8` → paste the printed snippet into the config, then re-run.

## 7. Seeds

For the headline DeepAR result run ≥3 seeds: `forecastkit run configs/m5.yaml --models deepar --set seed=1 --set run_name=m5_seed1` (and 2, 3), then report mean ± std across seeds.

## 8. Before you submit

- [ ] `make test` green · `configs/subsets/m5_subset.csv` committed · exact `config.yaml` of the run kept
- [ ] README results table replaced by your M5 numbers; `reports/example_synthetic_run/` clearly labelled synthetic (or deleted)
- [ ] Report + slides regenerated from `RESULTS.md` · every figure caption states its takeaway
- [ ] Limitations stated: stockout is a proxy · cold start simulated · one retailer · subset of series
- [ ] Tag the release: `git tag v1.0 && git push --tags`
