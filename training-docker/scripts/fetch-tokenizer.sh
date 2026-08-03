#!/usr/bin/env bash
# Usage: scripts/fetch-tokenizer.sh [DEST_DIR]
#
# Downloads the PaliGemma tokenizer files that the pi0.5 policy needs, into
# DEST_DIR (default: ./paligemma_tokenizer_flat). The Dockerfile copies that
# directory into the image, so run this before `docker build`.
#
# google/paligemma-3b-pt-224 is a gated repository. Before this script can work
# you must, with your own Hugging Face account:
#   1. accept the Gemma terms of use on the model page
#      https://huggingface.co/google/paligemma-3b-pt-224
#   2. wait for access to be granted
#   3. authenticate locally, either with `hf auth login` or by exporting
#      HF_TOKEN with a token that has read access
#
# The tokenizer files are redistributed by Google under the Gemma terms, not
# under this project's license, which is why they are not vendored here.

set -euo pipefail

DEST="${1:-./paligemma_tokenizer_flat}"
REPO_ID="${PALIGEMMA_REPO_ID:-google/paligemma-3b-pt-224}"

python - "$REPO_ID" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, dest = sys.argv[1], sys.argv[2]
files = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]
path = snapshot_download(
    repo_id,
    local_dir=dest,
    allow_patterns=files,
)
print(f"tokenizer files downloaded to {path}")
PY

echo "[fetch-tokenizer] done: $DEST"
