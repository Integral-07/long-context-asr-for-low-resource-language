#!/usr/bin/env bash
# TED-LIUM 3 のダウンロードからlcasr学習開始までを一括実行するスクリプト
# 実行: bash scripts/setup_tedlium.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HOME_DIR="$HOME"

TEDLIUM_TGZ="$HOME_DIR/TEDLIUM_release-3.tgz"
TEDLIUM_DIR="$HOME_DIR/TEDLIUM_release-3"
MANIFEST_DIR="$HOME_DIR/speech-datasets-remote"
SPEC_DIR="$HOME_DIR/tedlium_processed"
MAPPING_JSON="$REPO_DIR/data/mapping.json"
CONFIG="$REPO_DIR/exp/configs/floras_test.yaml"

# ── Step 1: Lhotse マニフェスト取得 ──────────────────────────────────────────
echo "=== [1/4] Fetching TED-LIUM Lhotse manifests ==="
if [ -d "$MANIFEST_DIR/longform_reconstitution/tedlium" ]; then
    echo "Already exists, skipping."
else
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/revdotcom/speech-datasets.git "$MANIFEST_DIR"
    cd "$MANIFEST_DIR"
    git sparse-checkout init --cone
    git sparse-checkout set longform_reconstitution/tedlium
    cd "$REPO_DIR"
fi

# ── Step 2: TED-LIUM 3 ダウンロード & 展開 ───────────────────────────────────
echo ""
echo "=== [2/4] Downloading TED-LIUM 3 (~14GB) ==="
if [ -d "$TEDLIUM_DIR" ]; then
    echo "Already exists, skipping."
else
    wget -c -O "$TEDLIUM_TGZ" \
        "https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz"
    echo "Extracting..."
    tar -xzf "$TEDLIUM_TGZ" -C "$HOME_DIR"
    echo "Extraction done."
fi

# ── Step 3: スペクトログラム生成 & mapping.json 作成 ─────────────────────────
echo ""
echo "=== [3/4] Generating spectrograms and mapping.json ==="
mkdir -p "$REPO_DIR/data"
cd "$REPO_DIR"
uv run --extra gpu scripts/prepare_tedlium.py \
    --tedlium-dir "$TEDLIUM_DIR" \
    --manifest-dir "$MANIFEST_DIR" \
    --spec-dir "$SPEC_DIR" \
    --output "$MAPPING_JSON" \
    --split train

# ── Step 4: config のパスを書き換えて学習開始 ────────────────────────────────
echo ""
echo "=== [4/4] Updating config and starting training ==="
uv run --extra gpu python3 - <<EOF
import re
with open("$CONFIG", "r") as f:
    content = f.read()
content = re.sub(r'path:.*', 'path: $MAPPING_JSON', content)
with open("$CONFIG", "w") as f:
    f.write(content)
print("Config updated: data.path ->", "$MAPPING_JSON")
EOF

uv run --extra gpu exp/train.py -config "$CONFIG"
