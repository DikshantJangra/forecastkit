#!/bin/sh
# Build the Phase 1 slide deck: charts from the benchmark CSVs, then HTML -> PDF with headless Chrome.
# Usage (from the repo root or slides/):  sh slides/build.sh   [PYTHON=/path/to/python] [CHROME=/path/to/chrome]
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/tmp/fkvenv/bin/python}"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$PYTHON" build_charts.py
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --allow-file-access-from-files \
  --print-to-pdf="$PWD/forecastkit_phase1_slides.pdf" "file://$PWD/deck.html" 2>/dev/null
echo "wrote slides/forecastkit_phase1_slides.pdf"
