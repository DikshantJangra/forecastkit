# Phase 1 changelog (2026-10-29)

Nothing has been committed. No model, API, frontend or test code was changed. The PDFs were not touched.

## Created
- `scripts/eda_m5.py`: the EDA script. It loads data through the repo loader and reuses `classify_demand`, `leading_zero_mask` and `stockout_mask`. Runs in about 6 s.
- `docs/EDA_FINDINGS.md`, `docs/figures/eda/01..06_*.png` (6 figures), `outputs/eda/*.csv` (11 tables, git-ignored)
- `docs/LITERATURE_REVIEW.md`: 19 references, verified through Crossref and the NeurIPS proceedings
- `docs/FEATURE_ENGINEERING.md`
- `docs/THEORY_AND_METHODS.md`
- `docs/ML_RESULTS_AND_ABLATIONS.md`
- `docs/PHASE1_EVIDENCE_INDEX.md`, `docs/PHASE1_CHANGELOG.md`
- Created earlier in this session: `docs/M5_FINDINGS.md`, `configs/subsets/m5_subset.csv` (the frozen subset; the repo instructions say to commit it), and the `data/raw/m5` symlink (git-ignored)

## Changed (backups in `/tmp/*.bak`)
- `plan.md`: replaced every `[M5 RESULT]` placeholder with verified values. Changed the tuning sentence from "tuned" to "tuning code exists, not run". Recorded that DeepAR uses a single seed.
- `docs/ACADEMIC_REPORT.md`: rewrote the abstract with real M5 numbers. Added a status banner, and a note marking Sections 5 and 6 as synthetic.
- `docs/SLIDES_OUTLINE.md`: removed the placeholders and listed the unsupported deck percentages.
- `docs/EDA_FINDINGS.md`: added a note explaining why the EDA class counts and flag rate differ from the backtest's (different scope and time window).

## Checks run
- `/tmp/fkvenv/bin/python -m pytest -q`: 35 passed
- EDA script rerun: exit 0, and the PNGs and `series_stats.csv` are byte-identical (md5) to the first run
- All headline numbers checked against `main_all_days.csv`, `main_non_censored.csv`, `cold_start.csv` and `origin_stability.csv`. No mismatches.
- New docs: 0 em-dashes and 0 placeholders. `git diff --stat` shows only the 3 Markdown files above.

## Paper finalisation (2026-10-29)

### Created
- `paper/forecastkit_phase1.tex`: the IEEEtran conference paper, 8 pages, anonymous author block.
- `paper/references.bib`: 19 references taken from `docs/LITERATURE_REVIEW.md`, with DOI URLs added (`IEEEtran.bst` does not print the `doi` field).
- `paper/forecastkit_phase1.pdf`: compiled with Tectonic 0.17.0.
- `paper/tex_to_md.py`: generates `docs/ACADEMIC_REPORT.md` from the `.tex`.
- `paper/README.md`: build instructions.

### Changed
- `docs/ACADEMIC_REPORT.md`: replaced in full and now generated from the LaTeX source.
- `docs/REPORT_DRAFT.md`: now points to the new paper and marks the old PDFs as superseded. Backup in `/tmp/REPORT_DRAFT.md.bak`.
- `plan.md`: removed the `@DJ` signature.
- `docs/PHASE1_EVIDENCE_INDEX.md`: updated the report status, anonymity and citation notes.

### Substantive corrections relative to the old report
- All synthetic smoke-run numbers were removed: 12,320 evaluations, 800 days, WAPE 0.489, pinball 0.515, Kalman 2.46 s, "30× faster", the "1.6% ensemble gain" and the "2.2% censoring penalty".
- The ensemble is no longer called the best model. DeepAR is reported as best on all-days WAPE and MAE. DeepAR, the hybrid and the ensemble are reported as tied on pinball, because their gaps (at most 0.0004) are far below the 0.004 standard deviation across origins.
- The ensemble was described as a 1/2 Kalman plus 1/2 DeepAR average. It is actually an equal-weight (1/3) average of Kalman, LightGBM and DeepAR.
- Kalman parameter selection is now described correctly as a per-series grid search over 16 noise-ratio pairs, not EM. The hybrid's internal filter uses one shared pair.
- "We mathematically prove no leakage" was replaced by a description of exactly what `tests/test_leakage.py` checks and what it does not check.
- The old report said censored days were "weighted as zero in LightGBM loss". The paper now says, correctly, that rows whose target is masked are dropped.
- The M5 citation was corrected to "Results, findings, and conclusions", IJF 38(4), 1346-1364.
- Added the missing material: a related-work section, EDA with 4 main-text and 2 appendix figures, a feature table, the price-availability assumption and the non-causal fill of missing prices, the non-censored and cold-start results, the completed ablations, the calibration nuance for discrete data, threats to validity, and a list of experiments not run.

### Checks
- `tectonic -X compile`: success, 8 pages, no overfull boxes, no undefined references or citations.
- All 19 bib keys are cited, and every cited key exists. Every figure and table label is referenced.
- Every table cell was compared with the CSVs in `outputs/` under strict rounding: 0 mismatches. Three padding and rounding errors were found and fixed during the check.
- The set of numbers is identical between the `.tex` and `docs/ACADEMIC_REPORT.md`, apart from DOI digits.
- `pytest -q`: 35 passed. `git diff` shows no changes under `src/`, `tests/`, `frontend/` or `app.py`.

## Presentation rebuild (2026-10-29)

The old deck had no source in the repo. Its PDF metadata (HeadlessChrome / Skia, 1440 × 810 px) shows it was an HTML page printed from Chrome. The new source follows the same approach and the same visual style.

### Created
- `slides/deck.html`: 12 slides.
- `slides/build_charts.py`: draws the result charts from the CSVs.
- `slides/build.sh`: the build script.
- `slides/README.md`: build instructions.
- `slides/figures/*.svg` and `values.json`: generated by the chart script.
- `slides/forecastkit_phase1_slides.pdf`: the compiled deck.

### Changed
- `docs/SLIDES_OUTLINE.md`: rewritten for the new deck.

The old presentation PDFs were not modified. They are marked as superseded in `docs/SLIDES_OUTLINE.md` and `docs/REPORT_DRAFT.md`.

### Removed claims (present in the old deck)
- Synthetic metrics: WAPE 0.485 / 0.478 and so on, "1,200 series" next to synthetic numbers, 22.3 s / 207.3 s runtimes.
- "Tabular and hybrid dominate", "quantile ensemble" shown as best, "DeepAR out-generalizes local models by 34%".
- Ablation percentages that were never run: 31.4%, 18.2%, −18.7%, 14.2%.
- "Per-series EM", "35 passing tests" presented as a result, and "production-ready".

### Checks
- `sh slides/build.sh` produces a 12-page PDF at 1080 × 607.92 pt (16:9). It opens in `pdfinfo` and `pdftoppm`.
- All 12 pages were rendered to PNG and inspected. There is no text overflow or overlap.
- 94 numbers were extracted from the deck and matched against the paper and EDA CSVs. The only unmatched items are axis ticks, CSS values, the 9× newsvendor example, and derived values checked by hand (1,081 = 914 + 167; 29× = 397.4 / 13.7).
- A search for the old synthetic values and claims (0.489, 0.515, 12,320, 2.46, 30×, 34%, 31.4, 18.2, 18.7, 14.2, 2.2%, 1.6%, EM, "dominate", "out-generalize") found no matches.
- No author names appear in the deck, and the PDF author metadata is empty.

## Repository cleanup (2026-10-29)

No model, feature, config parameter, metric, split, seed, test, API or frontend code was changed. Nothing was retrained.

### Moved (with `git mv`, history kept)
- `ForecastKit_Academic_Report.pdf`, `ForecastKit_Presentation.pdf`, `Inventory Demand Forecasting Guide.pdf` → `archive/superseded/`.
- `reports/ForecastKit_Academic_Report.pdf`, `reports/ForecastKit_Presentation.pdf` → `archive/superseded/reports/`. These are byte-identical copies (same SHA-1) of the root files.
- `plan.md` → `archive/superseded/plan_report_draft.md`. It was an old report draft, superseded by the paper.
- `docs/REPORT_DRAFT.md` → `docs/DELIVERABLES.md`. Rewritten as the index of current deliverables.

### Created
- `archive/superseded/README.md`: what each old file was and what replaced it.
- `configs/ablations/deepar_quick.yaml`: a verbatim copy of `/tmp/deepar_quick.yaml`, the spec behind `outputs/ablation_deepar_quick/`. It was not in the repo before.

### Changed
- `README.md`: rewritten around a 12-step workflow table, repository map, verified results, completed vs not-run table, limitations and reproducibility notes.
- `Makefile`: added `eda`, `paper` and `slides` targets. Existing targets were not changed.
- `docs/RUN_REAL_M5.md`: added a status note (run completed; tuning and seeds not run) and updated the checklist.
- `docs/PHASE1_EVIDENCE_INDEX.md`, `docs/SLIDES_OUTLINE.md`: updated paths to the archive and the DeepAR spec.

### Checks
- `pytest`: 35 passed, before and after.
- SHA-1 of the 83 evidence files (`outputs/`, `paper/`, `slides/`, `configs/subsets/`, `docs/figures/`) was compared before and after. Only rebuilt files changed: the rebuilt PDFs differed in creation date, and the SVGs in their embedded date. The PDF text was identical. The original paper and slides PDFs were restored, so their hashes match.
- `make eda` (5.5 s), `make paper` (8 pages, no overfull boxes or undefined references) and `make slides` (12 pages) all ran.

## Final presentation replaced (2026-10-29)

- Added `slides/final_presentation.pdf`: the 6-slide deck supplied by the user (`Claude Research Presentation Slides.pdf`). The PDF was copied unchanged.
- Moved the earlier 12-slide deck `slides/forecastkit_phase1_slides.pdf` to `archive/superseded/forecastkit_phase1_slides_html_v1.pdf` with `git mv`. `slides/build.sh` now writes there, so it can't overwrite the final deck.
- Updated `README.md`, `slides/README.md`, `docs/SLIDES_OUTLINE.md`, `docs/DELIVERABLES.md`, `docs/PHASE1_EVIDENCE_INDEX.md` and `archive/superseded/README.md`.
- Checks on the new deck: the author metadata is "Anonymous submission" and the text contains no names. A search found none of the old synthetic values. All results, EDA numbers and cell counts match the CSVs. The one rounding difference is −9.7% vs −9.8%, as noted in `docs/SLIDES_OUTLINE.md`.
