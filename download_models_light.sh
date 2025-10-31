#!/bin/bash
set -e
echo "=== Creating folders ==="
mkdir -p models/blip2/blip2_flant5_xl
mkdir -p models/clip/ViT-L-14

echo "=== Installing HF tools ==="
pip install huggingface_hub ftfy regex tqdm

echo "=== Downloading BLIP-2 Flan-T5-XL (≈8 GB on disk, not in VRAM) ==="
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Salesforce/blip2-flan-t5-xl", local_dir="models/blip2/blip2_flant5_xl")
print("✅ BLIP-2-XL downloaded")
PY

echo "=== Downloading CLIP ViT-L/14 (≈1 GB) ==="
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("openai/clip-vit-large-patch14", local_dir="models/clip/ViT-L-14")
print("✅ CLIP downloaded")
PY

echo "=== Done ==="
du -sh models/*
