#!/bin/bash
# 実験⑤(karol wav2vec2-Ainu + 長文脈シーケンス長ウォームアップ)の
# 条件A(短文脈)vs 条件B(長文脈)コレクション単位k-fold交差検証を通して
# 実行するドライバ。scripts/run_ainu_kfold.sh(実験②③用)のwav2vec2版。
# GPUホスト(このリポジトリでは dl12)上で、リポジトリルートから実行する
# ことを想定している。
#
# 各foldについて:
#   1. データ準備(条件A: 発話単位 / 条件B: コレクション連結、wav2vec2 CNN特徴量)
#   2. 条件A/Bそれぞれ学習(トークナイザは実験④で作成済みのfold別トークナイザを再利用)
#   3. 条件A/Bそれぞれtestで評価(test_predictions.json を出力)
# 最後に、全foldの結果を scripts/aggregate_kfold_results.py でプールして
# 統計的有意性を検定する(モデル非依存のスクリプトなので②③用をそのまま再利用)。
#
# 実行前に一度 scripts/generate_kfold_configs_wav2vec2.py で
# exp/configs/ainu_wav2vec2_kfold/fold{N}_{short,long}.yaml を生成しておくこと。
# 実験④の lcasr/artifacts/ainu_kfold/fold{N}/tokenizer.model が既に存在する
# ことも前提(なければ scripts/run_ainu_kfold.sh 側で作成される)。
#
# Usage:
#   bash scripts/run_ainu_wav2vec2_kfold.sh [N_FOLDS]

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"  # uv がログインシェル経由でないと PATH に無いことがある

N_FOLDS="${1:-5}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== generating configs for $N_FOLDS folds ==="
uv run --extra gpu python scripts/generate_kfold_configs_wav2vec2.py --n-folds "$N_FOLDS"

for fold in $(seq 0 $((N_FOLDS - 1))); do
  echo
  echo "############################################"
  echo "# fold $fold / $((N_FOLDS - 1))"
  echo "############################################"

  TOKENIZER="lcasr/artifacts/ainu_kfold/fold${fold}/tokenizer.model"
  if [ ! -f "$TOKENIZER" ]; then
    echo "ERROR: $TOKENIZER が見つかりません。先に実験④(scripts/run_ainu_kfold.sh)で作成してください。" >&2
    exit 1
  fi

  echo "--- [$fold] prepare_ainu_wav2vec2_utterances.py (条件A, 発話単位) ---"
  uv run --extra gpu python scripts/prepare_ainu_wav2vec2_utterances.py \
    --corpus-dir ainu_corpus \
    --feat-dir   "ainu_w2v2_kfold_utterance_processed/fold${fold}" \
    --output-dir "data/ainu_wav2vec2_kfold_utterances/fold${fold}" \
    --fold "$fold"

  echo "--- [$fold] prepare_ainu_wav2vec2_longcontext.py (条件B, コレクション連結) ---"
  uv run --extra gpu python scripts/prepare_ainu_wav2vec2_longcontext.py \
    --corpus-dir ainu_corpus \
    --feat-dir   "ainu_w2v2_kfold_processed/fold${fold}" \
    --output-dir "data/ainu_wav2vec2_kfold_collections/fold${fold}" \
    --fold "$fold"

  echo "--- [$fold] train 条件A (short) ---"
  uv run --extra gpu python exp/train.py --config "exp/configs/ainu_wav2vec2_kfold/fold${fold}_short.yaml"

  echo "--- [$fold] train 条件B (long) ---"
  uv run --extra gpu python exp/train.py --config "exp/configs/ainu_wav2vec2_kfold/fold${fold}_long.yaml"

  SHORT_CKPT_DIR="checkpoints/ainu_wav2vec2_kfold/fold${fold}_short"
  LONG_CKPT_DIR="checkpoints/ainu_wav2vec2_kfold/fold${fold}_long"
  SHORT_LAST_CKPT="$(ls -t "$SHORT_CKPT_DIR"/step_*.pt | head -1)"
  LONG_LAST_CKPT="$(ls -t "$LONG_CKPT_DIR"/step_*.pt | head -1)"

  echo "--- [$fold] eval 条件A short (checkpoint: $SHORT_LAST_CKPT) ---"
  uv run --extra gpu python scripts/eval_ainu_lcasr.py \
    --checkpoint "$SHORT_LAST_CKPT" \
    --mapping    "data/ainu_wav2vec2_kfold_utterances/fold${fold}/test_mapping.json" \
    --tokenizer  "$TOKENIZER" \
    --output     "$SHORT_CKPT_DIR/test_predictions.json"

  echo "--- [$fold] eval 条件B long (checkpoint: $LONG_LAST_CKPT) ---"
  uv run --extra gpu python scripts/eval_ainu_lcasr.py \
    --checkpoint "$LONG_LAST_CKPT" \
    --mapping    "data/ainu_wav2vec2_kfold_collections/fold${fold}/test_mapping.json" \
    --tokenizer  "$TOKENIZER" \
    --output     "$LONG_CKPT_DIR/test_predictions.json"
done

echo
echo "=== aggregating all $N_FOLDS folds ==="
uv run --extra gpu python scripts/aggregate_kfold_results.py --checkpoints-dir checkpoints/ainu_wav2vec2_kfold --n-folds "$N_FOLDS"
