# Slides

**Final presentation:** `final_presentation.pdf` (6 slides, 16:9, author metadata "Anonymous submission"). It was exported from slide software (PDF metadata: PptxGenJS, LibreOffice). Its editable source is not in this repository.

## Earlier 12-slide HTML deck (superseded)

These files built the previous deck. They are kept because they are its source and because `build_charts.py` documents how the result charts were drawn from the CSVs.

- `deck.html`: the 12-slide source.
- `build_charts.py`: draws `figures/*.svg` from `outputs/m5_main/report/*.csv` and writes `figures/values.json`.
- `build.sh`: regenerates the charts and prints the deck with headless Chrome to `archive/superseded/forecastkit_phase1_slides_html_v1.pdf`. It does not touch `final_presentation.pdf`.

```bash
sh slides/build.sh        # or: make slides
PYTHON=python3 CHROME="/path/to/chrome" sh slides/build.sh
```

Requirements: Python with pandas and matplotlib, Google Chrome or Chromium, and the git-ignored `outputs/` CSVs.
