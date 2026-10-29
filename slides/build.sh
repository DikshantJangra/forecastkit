#!/bin/sh
# Build the EARLIER 12-slide HTML deck (superseded by slides/final_presentation.pdf).
# Charts from the benchmark CSVs, then HTML -> PDF with headless Chrome. Output goes to the archive
# so it can never overwrite the final presentation.
# Usage (from the repo root or slides/):  sh slides/build.sh   [PYTHON=/path/to/python] [CHROME=/path/to/chrome]
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/tmp/fkvenv/bin/python}"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$PYTHON" build_charts.py
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --allow-file-access-from-files \
  --print-to-pdf="$PWD/../archive/superseded/forecastkit_phase1_slides_html_v1.pdf" "file://$PWD/deck.html" 2>/dev/null
echo "wrote archive/superseded/forecastkit_phase1_slides_html_v1.pdf"
