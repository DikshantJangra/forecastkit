#!/bin/sh
# Phase 2 deck: numbers from results/phase2 -> deck.html -> PDF (headless Chrome) -> PNG previews for inspection.
#   sh slides/phase2/build.sh   [PYTHON=python] [CHROME=/path/to/chrome]
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$PYTHON" build_deck.py
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --allow-file-access-from-files \
  --print-to-pdf="$PWD/../phase2_presentation.pdf" "file://$PWD/deck.html" 2>/dev/null
echo "wrote slides/phase2_presentation.pdf"
if command -v pdftoppm >/dev/null; then mkdir -p preview && pdftoppm -r 40 -png ../phase2_presentation.pdf preview/p; fi
