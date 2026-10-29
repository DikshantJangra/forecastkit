# Phase 1 slide deck

Files:
- `deck.html`: the slide source, 12 slides. It uses the old deck's design (light background, green accent, cards) and a fixed colour for each model.
- `build_charts.py`: draws `figures/main_wape.svg`, `figures/main_pinball.svg` and `figures/by_setting.svg` directly from `outputs/m5_main/report/*.csv`. It also writes `figures/values.json` for checking.
- `build.sh`: regenerates the charts, then prints the HTML to PDF with headless Chrome.
- `forecastkit_phase1_slides.pdf`: the compiled deck.

The EDA figures are loaded from `../docs/figures/eda/`. They are not copied.

## Build

```bash
sh slides/build.sh
# override the tools if needed:
PYTHON=python3 CHROME="/path/to/chrome" sh slides/build.sh
```

Requirements: Python with pandas and matplotlib, and Google Chrome (or Chromium) for printing to PDF. The `outputs/` CSVs must exist. Regenerate them with `forecastkit all configs/m5.yaml` if needed; they are git-ignored.

To edit the slides, change `deck.html` and rebuild. Keep every number traceable to `paper/forecastkit_phase1.tex` or the CSVs.
