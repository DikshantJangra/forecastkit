# Phase 1 paper: build instructions

Files:
- `forecastkit_phase1.tex`: the paper. This is the authoritative source.
- `references.bib`: 19 verified references (see `docs/LITERATURE_REVIEW.md`).
- `forecastkit_phase1.pdf`: the compiled output.
- `tex_to_md.py`: regenerates `docs/ACADEMIC_REPORT.md` from the `.tex`.

The figures are read from `../docs/figures/eda/` through `\graphicspath`. They are not copied into this folder.

## Build the PDF

The paper uses the standard `IEEEtran` class with BibTeX and `IEEEtran.bst`.

With Tectonic (`brew install tectonic`). The first run downloads the required packages, about 5 minutes. Later runs take a few seconds.

```bash
cd paper
tectonic -X compile forecastkit_phase1.tex
```

With a full TeX Live / MacTeX installation:

```bash
cd paper
pdflatex forecastkit_phase1 && bibtex forecastkit_phase1 && pdflatex forecastkit_phase1 && pdflatex forecastkit_phase1
```

## Regenerate the Markdown version

Run this from the repo root after any edit to the `.tex` file:

```bash
python3 paper/tex_to_md.py      # writes docs/ACADEMIC_REPORT.md
```

## Where the numbers come from

All results come from `outputs/m5_main/report/*.csv`, `outputs/ablation_kalman/` and `outputs/ablation_deepar_quick/`. The EDA numbers come from `outputs/eda/*.csv`. These directories are git-ignored. Regenerate them with the commands in `docs/PHASE1_EVIDENCE_INDEX.md`.
