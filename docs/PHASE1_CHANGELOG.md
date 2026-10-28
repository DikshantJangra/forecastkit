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
