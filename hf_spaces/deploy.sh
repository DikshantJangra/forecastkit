#!/usr/bin/env bash
set -e

# Deploy ForecastKit to Hugging Face Spaces
# Usage: ./hf_spaces/deploy.sh <hf-username>/<space-name>

if [ -z "$1" ]; then
  echo "Usage: $0 <username>/<space-name>"
  echo "Example: $0 dikshant/forecastkit"
  exit 1
fi

SPACE_REPO="https://huggingface.co/spaces/$1"
TMP_DIR=$(mktemp -d)
echo "Deploying ForecastKit to Hugging Face Space: $SPACE_REPO..."

git clone "$SPACE_REPO" "$TMP_DIR" || {
  echo "Cloning failed. Make sure the Space exists on Hugging Face (SDK: Streamlit)."
  exit 1
}

# Copy code and configs
cp app.py "$TMP_DIR/"
cp hf_spaces/README.md "$TMP_DIR/README.md"
cp hf_spaces/requirements.txt "$TMP_DIR/requirements.txt"
cp -r src "$TMP_DIR/"
cp -r configs "$TMP_DIR/"
cp -r examples "$TMP_DIR/"
cp -r reports "$TMP_DIR/" 2>/dev/null || true
cp pyproject.toml "$TMP_DIR/"

cd "$TMP_DIR"
git add .
git commit -m "Deploy ForecastKit Studio to Hugging Face Spaces" || echo "No changes to commit"
git push origin main

echo "Deployment complete! Visit: $SPACE_REPO"
