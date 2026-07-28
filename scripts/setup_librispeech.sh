#!/usr/bin/env bash
# LibriSpeech train-clean-100 のダウンロードからlcasr学習開始まで
# 実行: bash scripts/setup_librispeech.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HOME_DIR="$HOME"

SPLIT="train-clean-100"
LIBRI_TGZ="$HOME_DIR/${SPLIT}.tar.gz"
LIBRI_DIR="$HOME_DIR/LibriSpeech"
SPEC_DIR="$HOME_DIR/librispeech_processed"
MAPPING_JSON="$REPO_DIR/data/mapping.json"
CONFIG="$REPO_DIR/exp/configs/floras_test.yaml"

# ── Step 1: ダウンロード & 展開 ────────────────────────────────────────────────
echo "=== [1/3] Downloading LibriSpeech ${SPLIT} (~6.3GB) ==="
if [ -d "$LIBRI_DIR/$SPLIT" ]; then
    echo "Already exists, skipping."
else
    wget -c -O "$LIBRI_TGZ" \
        "https://www.openslr.org/resources/12/${SPLIT}.tar.gz"
    echo "Extracting..."
    tar -xzf "$LIBRI_TGZ" -C "$HOME_DIR"
    echo "Extraction done."
fi

# ── Step 2: スペクトログラム生成 & mapping.json 作成 ─────────────────────────
echo ""
echo "=== [2/3] Generating spectrograms and mapping.json ==="
mkdir -p "$REPO_DIR/data"
cd "$REPO_DIR"
uv run --extra gpu scripts/prepare_librispeech.py \
    --librispeech-dir "$LIBRI_DIR" \
    --spec-dir "$SPEC_DIR" \
    --output "$MAPPING_JSON" \
    --split "$SPLIT"

# ── Step 3: config のパスを書き換えて学習開始 ────────────────────────────────
echo ""
echo "=== [3/3] Updating config and starting training ==="
uv run --extra gpu python3 - <<EOF
import re

with open("$CONFIG", "r") as f:
    lines = f.readlines()

in_data = False
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == "data:":
        in_data = True
    elif in_data:
        if stripped and not stripped.startswith("#") and not line[0].isspace():
            in_data = False
        elif re.match(r"\s+path\s*:", line):
            lines[i] = re.sub(r"(path\s*:).*", r"\1 $MAPPING_JSON", line)
            in_data = False

with open("$CONFIG", "w") as f:
    f.writelines(lines)
print("Config updated: data.path ->", "$MAPPING_JSON")
EOF

uv run --extra gpu exp/train.py -config "$CONFIG"
