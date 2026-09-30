#!/bin/bash
# 実験⑤条件A(既存手法)のWERが高い原因調査(タスク#1〜#3, INVESTIGATION.md)を受けて、
# 「学習率を3e-5に下げる」代わりに「元のlr=3e-4のまま、epoch平均train lossが最良の
# エポックのチェックポイントを使って評価する」方式で5fold×2条件を再実行するドライバ。
# scripts/run_ainu_wav2vec2_kfold.sh(公式lr=3e-5版)をベースに、以下だけを変える:
#   - config: scripts/make_bestepoch_kfold_configs.py で事前生成した
#     exp/configs/ainu_wav2vec2_bestepoch_kfold/fold{N}_{short,long}.yaml を使う
#     (lr=3e-4に戻し、training.diagnostics_path を追加、checkpointing.dirは
#     公式lr=3e-5版と衝突しない専用パス)
#   - データ・トークナイザは既存の実験⑤公式run(data/ainu_wav2vec2_kfold_*,
#     lcasr/artifacts/ainu_kfold)をそのまま再利用(再生成しない)
#   - 評価チェックポイントの選び方: 最終epochではなく
#     scripts/pick_best_epoch_checkpoint.py が選ぶ最良epochのチェックポイント
#
# 事前準備(このスクリプトの実行前に一度だけ):
#   uv run --extra gpu python scripts/make_bestepoch_kfold_configs.py --n-folds 5
#
# Usage:
#   bash scripts/run_ainu_wav2vec2_bestepoch_kfold.sh [N_FOLDS] [START_FOLD]

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

N_FOLDS="${1:-5}"
START_FOLD="${2:-0}"
PREFIX="ainu_wav2vec2"
BESTEPOCH_PREFIX="ainu_wav2vec2_bestepoch"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="exp/configs/${BESTEPOCH_PREFIX}_kfold"
if [ ! -d "$CONFIG_DIR" ]; then
  echo "ERROR: $CONFIG_DIR がありません。先に scripts/make_bestepoch_kfold_configs.py を実行してください。" >&2
  exit 1
fi

for fold in $(seq "$START_FOLD" $((N_FOLDS - 1))); do
  echo
  echo "############################################"
  echo "# [bestepoch] fold $fold / $((N_FOLDS - 1))"
  echo "############################################"

  TOKENIZER="lcasr/artifacts/ainu_kfold/fold${fold}/tokenizer.model"
  if [ ! -f "$TOKENIZER" ]; then
    echo "ERROR: $TOKENIZER が見つかりません。" >&2
    exit 1
  fi

  echo "--- [$fold] train 条件A (short, lr=3e-4, diagnostics有効) ---"
  uv run --extra gpu python exp/train.py --config "$CONFIG_DIR/fold${fold}_short.yaml"

  echo "--- [$fold] train 条件B (long, lr=3e-4, diagnostics有効) ---"
  uv run --extra gpu python exp/train.py --config "$CONFIG_DIR/fold${fold}_long.yaml"

  SHORT_CKPT_DIR="checkpoints/${BESTEPOCH_PREFIX}_kfold/fold${fold}_short"
  LONG_CKPT_DIR="checkpoints/${BESTEPOCH_PREFIX}_kfold/fold${fold}_long"

  echo "--- [$fold] 最良epochのチェックポイントを選定 (条件A) ---"
  SHORT_BEST_CKPT="$(uv run --extra gpu python scripts/pick_best_epoch_checkpoint.py \
    --diagnostics "$SHORT_CKPT_DIR/diagnostics.jsonl" \
    --checkpoint-dir "$SHORT_CKPT_DIR")"

  echo "--- [$fold] 最良epochのチェックポイントを選定 (条件B) ---"
  LONG_BEST_CKPT="$(uv run --extra gpu python scripts/pick_best_epoch_checkpoint.py \
    --diagnostics "$LONG_CKPT_DIR/diagnostics.jsonl" \
    --checkpoint-dir "$LONG_CKPT_DIR")"

  echo "--- [$fold] eval 条件A short (checkpoint: $SHORT_BEST_CKPT) ---"
  uv run --extra gpu python scripts/eval_ainu_lcasr.py \
    --checkpoint "$SHORT_BEST_CKPT" \
    --mapping    "data/${PREFIX}_kfold_utterances/fold${fold}/test_mapping.json" \
    --tokenizer  "$TOKENIZER" \
    --output     "$SHORT_CKPT_DIR/test_predictions.json"

  echo "--- [$fold] eval 条件B long (checkpoint: $LONG_BEST_CKPT) ---"
  uv run --extra gpu python scripts/eval_ainu_lcasr.py \
    --checkpoint "$LONG_BEST_CKPT" \
    --mapping    "data/${PREFIX}_kfold_collections/fold${fold}/test_mapping.json" \
    --tokenizer  "$TOKENIZER" \
    --output     "$LONG_CKPT_DIR/test_predictions.json"

  echo "--- [$fold] disk cleanup: drop intermediate checkpoints, keep test_predictions.json + diagnostics.jsonl ---"
  find "$SHORT_CKPT_DIR" "$LONG_CKPT_DIR" -name "step_*.pt" -delete
done

echo
echo "=== aggregating all $N_FOLDS folds (bestepoch, lr=3e-4) ==="
uv run --extra gpu python scripts/aggregate_kfold_results.py \
  --checkpoints-dir "checkpoints/${BESTEPOCH_PREFIX}_kfold" --n-folds "$N_FOLDS" --per-fold
