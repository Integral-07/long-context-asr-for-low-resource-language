#!/bin/bash
# 実験⑦(文字レベルトークナイザ検証): 実験⑤(karol wav2vec2-Ainu + 長文脈
# シーケンス長ウォームアップ)と全く同じモデル・学習レシピ・k-fold分割・
# 音声データ(data/ainu_wav2vec2_kfold_{utterances,collections})を再利用し、
# トークナイザだけをBPE-500から文字レベルSentencePieceに差し替えて
# 条件A/Bを再学習する。scripts/run_ainu_wav2vec2_kfold.sh の文字レベル版。
#
# 目的: INVESTIGATION.mdタスク#2で「実験①(31.47%)と実験⑤条件A(52.54%)の
# ~21ptの差の最有力候補」とされた文字レベル vs BPE-500 トークナイザの違いを、
# トークナイザ以外の条件を完全に固定した上で検証する。
#
# 音声特徴量の再抽出は不要(トークナイザは学習時にdataloader側で単語を
# encodeするだけで、data prep段階のmapping.json自体はトークナイザに依存
# しないため)。事前に実験⑤の公式run(run_ainu_wav2vec2_kfold.sh)で
# data/ainu_wav2vec2_kfold_{utterances,collections}/fold{N}/ が
# 生成済みであることが前提。
#
# 各foldについて:
#   1. 文字レベルトークナイザ学習(fold単位、未作成の場合のみ)
#   2. 条件A/Bそれぞれ学習(データは実験⑤のものを再利用)
#   3. 条件A/Bそれぞれtestで評価
# 最後に全foldの結果をプールして統計的有意性を検定する。
#
# Usage:
#   bash scripts/run_ainu_wav2vec2_char_kfold.sh [N_FOLDS] [START_FOLD]

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

N_FOLDS="${1:-5}"
START_FOLD="${2:-0}"
PREFIX="ainu_char"
DATA_PREFIX="ainu_wav2vec2"       # 実験⑤の音声特徴量を再利用
TOKENIZER_SUBDIR="ainu_char_kfold"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== generating configs for $N_FOLDS folds (prefix=$PREFIX, data reused from $DATA_PREFIX) ==="
uv run --extra gpu python scripts/generate_kfold_configs_wav2vec2.py \
  --n-folds "$N_FOLDS" \
  --short-template "exp/configs/ainu_wav2vec2_short.yaml" \
  --long-template  "exp/configs/ainu_wav2vec2_long.yaml" \
  --output-dir     "exp/configs/${PREFIX}_kfold" \
  --name-prefix    "$PREFIX" \
  --data-prefix    "$DATA_PREFIX" \
  --tokenizer-subdir "$TOKENIZER_SUBDIR"

for fold in $(seq "$START_FOLD" $((N_FOLDS - 1))); do
  echo
  echo "############################################"
  echo "# fold $fold / $((N_FOLDS - 1))"
  echo "############################################"

  SHORT_DATA="data/${DATA_PREFIX}_kfold_utterances/fold${fold}/train_mapping.json"
  LONG_DATA="data/${DATA_PREFIX}_kfold_collections/fold${fold}/train_mapping.json"
  if [ ! -f "$SHORT_DATA" ] || [ ! -f "$LONG_DATA" ]; then
    echo "ERROR: $SHORT_DATA / $LONG_DATA が見つかりません。先に実験⑤の" >&2
    echo "       run_ainu_wav2vec2_kfold.sh でこのfoldの音声データを準備してください。" >&2
    exit 1
  fi

  TOKENIZER_DIR="lcasr/artifacts/${TOKENIZER_SUBDIR}/fold${fold}"
  TOKENIZER="${TOKENIZER_DIR}/tokenizer.model"
  if [ ! -f "$TOKENIZER" ]; then
    echo "--- [$fold] train character-level tokenizer ---"
    uv run --extra gpu python scripts/train_ainu_tokenizer.py \
      --corpus-dir ainu_corpus \
      --save-dir   "$TOKENIZER_DIR" \
      --model-type char \
      --fold "$fold"
  else
    echo "--- [$fold] tokenizer already exists at $TOKENIZER, skipping ---"
  fi

  echo "--- [$fold] train 条件A (short) ---"
  uv run --extra gpu python exp/train.py --config "exp/configs/${PREFIX}_kfold/fold${fold}_short.yaml"

  echo "--- [$fold] train 条件B (long) ---"
  uv run --extra gpu python exp/train.py --config "exp/configs/${PREFIX}_kfold/fold${fold}_long.yaml"

  SHORT_CKPT_DIR="checkpoints/${PREFIX}_kfold/fold${fold}_short"
  LONG_CKPT_DIR="checkpoints/${PREFIX}_kfold/fold${fold}_long"
  SHORT_LAST_CKPT="$(ls -t "$SHORT_CKPT_DIR"/step_*.pt | head -1)"
  LONG_LAST_CKPT="$(ls -t "$LONG_CKPT_DIR"/step_*.pt | head -1)"

  echo "--- [$fold] eval 条件A short (checkpoint: $SHORT_LAST_CKPT) ---"
  uv run --extra gpu python scripts/eval_ainu_lcasr.py \
    --checkpoint "$SHORT_LAST_CKPT" \
    --mapping    "data/${DATA_PREFIX}_kfold_utterances/fold${fold}/test_mapping.json" \
    --tokenizer  "$TOKENIZER" \
    --output     "$SHORT_CKPT_DIR/test_predictions.json"

  echo "--- [$fold] eval 条件B long (checkpoint: $LONG_LAST_CKPT) ---"
  uv run --extra gpu python scripts/eval_ainu_lcasr.py \
    --checkpoint "$LONG_LAST_CKPT" \
    --mapping    "data/${DATA_PREFIX}_kfold_collections/fold${fold}/test_mapping.json" \
    --tokenizer  "$TOKENIZER" \
    --output     "$LONG_CKPT_DIR/test_predictions.json"

  echo "--- [$fold] disk cleanup: drop intermediate checkpoints, keep test_predictions.json ---"
  find "$SHORT_CKPT_DIR" "$LONG_CKPT_DIR" -name "step_*.pt" -delete
done

echo
echo "=== aggregating all $N_FOLDS folds ==="
uv run --extra gpu python scripts/aggregate_kfold_results.py --checkpoints-dir "checkpoints/${PREFIX}_kfold" --n-folds "$N_FOLDS"
