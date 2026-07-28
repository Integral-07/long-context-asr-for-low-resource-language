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
    # 複数ミラーを順に試す
    TEDLIUM_URLS=(
        "https://projets-lium.univ-lemans.fr/wp-content/uploads/corpus/TED-LIUM/TEDLIUM_release-3.tgz"
        "https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz"
    )
    DOWNLOADED=false
    for URL in "${TEDLIUM_URLS[@]}"; do
        echo "Trying: $URL"
        if wget -c -O "$TEDLIUM_TGZ" "$URL"; then
            DOWNLOADED=true
            break
        fi
        echo "Failed, trying next mirror..."
    done
    if [ "$DOWNLOADED" = false ]; then
        echo "ERROR: すべてのミラーからのダウンロードに失敗しました。"
        echo "手動でダウンロードして $TEDLIUM_TGZ に配置してください。"
        exit 1
    fi
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
