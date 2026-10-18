#!/bin/bash
# 実験⑪(self_conditioning: false の再検証、文字レベルトークナイザ):
# 実験⑦(文字レベルトークナイザ、自己条件付きlr=3e-5)と全く同じ音声データ・
# 文字トークナイザ(lcasr/artifacts/ainu_char_kfold)を再利用しつつ、モデルの
# self_conditioning だけを false に変える(exp/configs/ainu_wav2vec2_noselfcond_
# {short,long}.yaml)。scripts/run_ainu_wav2vec2_char_kfold.sh の自己条件付けなし版。
#
# 目的: 他メンバーのfinetune_xlsr53_ainu(HF Trainer + 素のWav2Vec2ForCTC、
# 自己条件付けなし)が文字レベルトークナイザで良好な結果(test WER 31.47%)を
# 出している一方、我々の自己条件付きモデルは文字レベルで大幅悪化する
# (52.54%→71.51%、既存手法)。この差が自己条件付けの中間CTC loss増幅
# (23層×重み0.5 ≈ 最終lossの12倍)によるものか検証する。
# なお self_conditioning: false は9/19のアブレーション(aeffd99)で一度
# mode collapseに陥ったが、それはlr=3e-4(訓練不安定化前)時代の結果で、
# lr=3e-5(e2ad8af、後発のstability fix)との組み合わせはまだ試していない。
#
# Usage:
#   bash scripts/run_ainu_wav2vec2_noselfcond_char_kfold.sh [N_FOLDS] [START_FOLD]
#
# nohup等でバックグラウンド実行すればSSH切断後も継続する:
#   nohup bash scripts/run_ainu_wav2vec2_noselfcond_char_kfold.sh 5 0 > noselfcond_char_kfold_run.log 2>&1 &

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

N_FOLDS="${1:-5}"
START_FOLD="${2:-0}"
PREFIX="ainu_noselfcond_char"
DATA_PREFIX="ainu_wav2vec2"       # 実験⑤の音声特徴量を再利用
TOKENIZER_SUBDIR="ainu_char_kfold" # 実験⑦の文字レベルトークナイザを再利用
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== generating configs for $N_FOLDS folds (prefix=$PREFIX, self_conditioning=false) ==="
uv run --extra gpu python scripts/generate_kfold_configs_wav2vec2.py \
  --n-folds "$N_FOLDS" \
  --short-template "exp/configs/ainu_wav2vec2_noselfcond_short.yaml" \
  --long-template  "exp/configs/ainu_wav2vec2_noselfcond_long.yaml" \
  --output-dir     "exp/configs/${PREFIX}_kfold" \
  --name-prefix    "$PREFIX" \
  --data-prefix    "$DATA_PREFIX" \
  --tokenizer-subdir "$TOKENIZER_SUBDIR"

for fold in $(seq "$START_FOLD" $((N_FOLDS - 1))); do
  echo
  echo "############################################"
  echo "# [noselfcond char] fold $fold / $((N_FOLDS - 1))"
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

  echo "--- [$fold] train 条件A (short, self_conditioning=false) ---"
  uv run --extra gpu python exp/train.py --config "exp/configs/${PREFIX}_kfold/fold${fold}_short.yaml"

  echo "--- [$fold] train 条件B (long, self_conditioning=false) ---"
  uv run --extra gpu python exp/train.py --config "exp/configs/${PREFIX}_kfold/fold${fold}_long.yaml"

  SHORT_CKPT_DIR="checkpoints/${PREFIX}_kfold/fold${fold}_short"
  LONG_CKPT_DIR="checkpoints/${PREFIX}_kfold/fold${fold}_long"
  if [ -f "$SHORT_CKPT_DIR/best_dev/best_dev.pt" ]; then
    SHORT_LAST_CKPT="$SHORT_CKPT_DIR/best_dev/best_dev.pt"
  else
    SHORT_LAST_CKPT="$(ls -t "$SHORT_CKPT_DIR"/step_*.pt | head -1)"
  fi
  if [ -f "$LONG_CKPT_DIR/best_dev/best_dev.pt" ]; then
    LONG_LAST_CKPT="$LONG_CKPT_DIR/best_dev/best_dev.pt"
  else
    LONG_LAST_CKPT="$(ls -t "$LONG_CKPT_DIR"/step_*.pt | head -1)"
  fi

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

  echo "--- [$fold] disk cleanup: drop intermediate checkpoints, keep best_dev/dev_eval_log.jsonl + test_predictions.json ---"
  find "$SHORT_CKPT_DIR" "$LONG_CKPT_DIR" -name "step_*.pt" -delete
done

echo
echo "=== aggregating all $N_FOLDS folds ==="
uv run --extra gpu python scripts/aggregate_kfold_results.py --checkpoints-dir "checkpoints/${PREFIX}_kfold" --n-folds "$N_FOLDS"
