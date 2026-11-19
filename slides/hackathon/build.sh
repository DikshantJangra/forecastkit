#!/bin/sh
# Hackathon deck: results -> deck.html -> PDF (headless Chrome) -> PNG (pdftoppm) -> PPTX (python-pptx).
#   PYTHON=/path/to/python sh slides/hackathon/build.sh
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PYTHONPATH=../../src "$PYTHON" build_hackathon.py > /dev/null
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --allow-file-access-from-files \
  --print-to-pdf="$PWD/ForecastKit_Hackathon.pdf" "file://$PWD/deck.html" 2>/dev/null
rm -rf png && mkdir -p png && pdftoppm -r 144 -png ForecastKit_Hackathon.pdf png/slide
"$PYTHON" make_pptx.py
echo "wrote slides/hackathon/ForecastKit_Hackathon.{pdf,pptx} and png/"
