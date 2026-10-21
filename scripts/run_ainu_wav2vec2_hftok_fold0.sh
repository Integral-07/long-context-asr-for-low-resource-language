#!/bin/bash
# 実験⑫ fold0: 実験⑪(self_conditioning=false)に加えて、渡辺さんの
# finetune_xlsr53_ainu notebookの3点(HF Wav2Vec2CTCTokenizer・CTC loss
# reduction='mean'・argmax greedyデコード)を組み込んで再検証する。
# fold0のみ(3点同時変更のリスクが高いため、まずここで確認する)。
#
# Usage:
#   bash scripts/run_ainu_wav2vec2_hftok_fold0.sh
#
#   nohup bash scripts/run_ainu_wav2vec2_hftok_fold0.sh > hftok_fold0_run.log 2>&1 < /dev/null &

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TOKENIZER_DIR="lcasr/artifacts/ainu_hftok_kfold/fold0"
SHORT_DATA="data/ainu_wav2vec2_kfold_utterances/fold0/train_mapping.json"
LONG_DATA="data/ainu_wav2vec2_kfold_collections/fold0/train_mapping.json"

if [ ! -f "$SHORT_DATA" ] || [ ! -f "$LONG_DATA" ]; then
  echo "ERROR: $SHORT_DATA / $LONG_DATA が見つかりません。先に実験⑤の" >&2
  echo "       run_ainu_wav2vec2_kfold.sh でfold0の音声データを準備してください。" >&2
  exit 1
fi

if [ ! -f "$TOKENIZER_DIR/vocab.json" ]; then
  echo "--- train HF character-level tokenizer (fold0) ---"
  uv run --extra gpu python scripts/train_ainu_hf_tokenizer.py \
    --corpus-dir ainu_corpus \
    --save-dir   "$TOKENIZER_DIR" \
    --fold 0
else
  echo "--- tokenizer already exists at $TOKENIZER_DIR, skipping ---"
fi

echo "--- train 条件A (short) ---"
uv run --extra gpu python exp/train.py --config "exp/configs/ainu_hftok_kfold/fold0_short.yaml"

echo "--- train 条件B (long) ---"
uv run --extra gpu python exp/train.py --config "exp/configs/ainu_hftok_kfold/fold0_long.yaml"

SHORT_CKPT_DIR="checkpoints/ainu_hftok_kfold/fold0_short"
LONG_CKPT_DIR="checkpoints/ainu_hftok_kfold/fold0_long"

if [ -f "$SHORT_CKPT_DIR/best_dev/best_dev.pt" ]; then
  SHORT_CKPT="$SHORT_CKPT_DIR/best_dev/best_dev.pt"
else
  SHORT_CKPT="$(ls -t "$SHORT_CKPT_DIR"/step_*.pt | head -1)"
fi
if [ -f "$LONG_CKPT_DIR/best_dev/best_dev.pt" ]; then
  LONG_CKPT="$LONG_CKPT_DIR/best_dev/best_dev.pt"
else
  LONG_CKPT="$(ls -t "$LONG_CKPT_DIR"/step_*.pt | head -1)"
fi

echo "--- eval 条件A short (checkpoint: $SHORT_CKPT) ---"
uv run --extra gpu python scripts/eval_ainu_lcasr.py \
  --checkpoint "$SHORT_CKPT" \
  --mapping    "data/ainu_wav2vec2_kfold_utterances/fold0/test_mapping.json" \
  --tokenizer  "$TOKENIZER_DIR" \
  --decoder    greedy \
  --output     "$SHORT_CKPT_DIR/test_predictions.json"

echo "--- eval 条件B long (checkpoint: $LONG_CKPT) ---"
uv run --extra gpu python scripts/eval_ainu_lcasr.py \
  --checkpoint "$LONG_CKPT" \
  --mapping    "data/ainu_wav2vec2_kfold_collections/fold0/test_mapping.json" \
  --tokenizer  "$TOKENIZER_DIR" \
  --decoder    greedy \
  --output     "$LONG_CKPT_DIR/test_predictions.json"

find "$SHORT_CKPT_DIR" "$LONG_CKPT_DIR" -name "step_*.pt" -delete

echo
echo "=== done ==="
echo "short: $SHORT_CKPT_DIR/test_predictions.json"
echo "long:  $LONG_CKPT_DIR/test_predictions.json"
