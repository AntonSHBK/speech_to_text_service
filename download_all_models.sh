#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/data/cache_dir}"

mkdir -p "$CACHE_DIR"

MODELS=(
  "Systran/faster-whisper-small"
  "Systran/faster-whisper-medium"
  "Systran/faster-whisper-large-v3"
)

echo "Using cache dir: $CACHE_DIR"

for model in "${MODELS[@]}"; do
  echo "Downloading $model ..."
  python -m huggingface_hub snapshot-download "$model" \
    --local-dir "$CACHE_DIR/$model" \
    --local-dir-use-symlinks False
  echo "Done: $model"
done

echo "All models downloaded to: $CACHE_DIR"
