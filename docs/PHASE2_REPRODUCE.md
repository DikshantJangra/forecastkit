# Reproducing Phase 2

All commands run from the repository root with the package installed (`pip install -e ".[all]"`, Python >= 3.10).
The M5 files must be in `data/raw/m5/` (see `docs/RUN_REAL_M5.md`); the frozen subset
`configs/subsets/m5_subset.csv` is committed. Runtimes are for an 8-core laptop CPU with 5 jobs in parallel.

## 0. Tests (no M5 data needed, ~25 s)

```bash
pytest -q                      # 65 tests (~10 s): Phase 1 (35) + Phase 2 DL, hybrid and evaluation tests (30)
```

## 1. Phase 1 baselines (if `outputs/m5_main` is absent, ~19 min)

```bash
forecastkit all configs/m5.yaml
```

Phase 2 reuses the Phase 1 forecasts of `naive, snaive, snaive4, ma28, kalman, lgbm, kalman_lgbm` unchanged.

## 2. E01: Phase 1 DeepAR reproduction check (~5 min)

```bash
forecastkit run configs/phase2/e01_repro.yaml      # -> outputs/phase2/e01_repro_s42, identical to outputs/m5_main deepar
```

## 3. E02: DeepAR design selection on the validation origin (5 jobs, ~5 min in parallel)

```bash
for m in deepar_p1 deepar_es deepar_es_lags deepar_es_lags_age deepar_es_lags_age_adamw; do
  forecastkit run configs/phase2/e02_deepar_select.yaml --phase val --models $m --set run_name=phase2/e02_val/$m &
done; wait
python scripts/phase2_e02_table.py                 # -> results/phase2/e02_validation_selection.csv
```

## 4. E03/E05/E06: all test runs (26 jobs, ~60 min with JOBS=5; use JOBS=2 on a laptop, ~2.5 h)

The script skips runs whose `forecasts.parquet` exists. 15 of 26 completed in this project; see the ledger.

```bash
# E01 doubles as deepar_p1 seed 42: copy it, or let the script recompute it
PY=python JOBS=5 sh scripts/phase2_run_all.sh      # -> outputs/phase2/runs/<model>_s<seed>/
```

Each job is `forecastkit run configs/phase2/m5_phase2.yaml --models <m> --set seed=<s> --set run_name=phase2/runs/<m>_s<s>`.
Seeds: 42, 43, 44 for `deepar_p1, deepar_v2, deepar_v2_nomask, hybrid`; 42 for the ablations.
The top-level `seed` changes only model randomness (weight init, batches, sampling, LightGBM bagging); the
series subset, cold-start split, origins and masks are fixed by their own seeds.

## 5. E04: zero-run persistence model vs baselines (~5 min)

```bash
python scripts/phase2_zero_model.py --config configs/m5.yaml --out results/phase2/zero_model \
  --deepar deepar_v2_masked=outputs/phase2/runs/deepar_v2_s42/artifacts/deepar_v2 \
  --deepar deepar_v2_unmasked=outputs/phase2/runs/deepar_v2_nomask_s42/artifacts/deepar_v2_nomask \
  --deepar deepar_p1=outputs/phase2/runs/deepar_p1_s42/artifacts/deepar_p1
```

## 6. E07: results package (~3 min, ~2 GB RAM)

```bash
python scripts/phase2_analysis.py                  # -> results/phase2/*.csv, figures/, RESULTS_PHASE2.md
```

## 7. Report and slides

```bash
make paper2        # tectonic -> paper/forecastkit_phase2.pdf
make slides2       # Chrome headless -> slides/phase2_presentation.pdf
```

## Where things are

| What | Where (tracked) | Raw (gitignored) |
|---|---|---|
| Configs | `configs/phase2/` | - |
| Forecasts, per-origin samples, curves, weights | - | `outputs/phase2/runs/*/` |
| All Phase 2 tables and figures | `results/phase2/` | - |
| Phase 1 tables (copied from `outputs/m5_main/report`) | `results/phase1/` | `outputs/m5_main/` |
| Experiment ledger | `docs/PHASE2_EXPERIMENT_LEDGER.md` | - |
