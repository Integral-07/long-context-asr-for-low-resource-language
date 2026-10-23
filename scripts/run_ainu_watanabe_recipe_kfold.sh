#!/bin/bash
# 実験⑬のk-fold版: 渡辺さんのfinetune_xlsr53_ainu notebookのレシピ
# (scripts/finetune_ainu_watanabe_recipe.py)を、53コレクション5foldの
# k-fold交差検証で実行する。単一split(固定test: at08/30/40/13/22)の結果
# (test WER 36.37%)が固定splitの難易度に左右されている可能性があるため、
# より確証のある数値を得る目的。
#
# Usage:
#   bash scripts/run_ainu_watanabe_recipe_kfold.sh [N_FOLDS] [START_FOLD] [MAX_STEPS]
#
#   nohup bash scripts/run_ainu_watanabe_recipe_kfold.sh 5 0 4500 > watanabe_kfold_run.log 2>&1 < /dev/null &

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

N_FOLDS="${1:-5}"
START_FOLD="${2:-0}"
MAX_STEPS="${3:-4500}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

for fold in $(seq "$START_FOLD" $((N_FOLDS - 1))); do
  echo
  echo "############################################"
  echo "# [watanabe recipe] fold $fold / $((N_FOLDS - 1))"
  echo "############################################"

  DATA_DIR="data/ainu_watanabe_kfold/fold${fold}"
  TOKENIZER_DIR="lcasr/artifacts/ainu_hftok_kfold/fold${fold}"
  CKPT_DIR="checkpoints/ainu_watanabe_kfold/fold${fold}"

  if [ ! -f "$DATA_DIR/train.json" ]; then
    echo "--- [$fold] prepare raw-audio data (utterance level, k-fold) ---"
    uv run --extra gpu python scripts/prepare_ainu_wav2vec2.py \
      --corpus-dir ainu_corpus \
      --output-dir "$DATA_DIR" \
      --fold "$fold"
  else
    echo "--- [$fold] data already exists at $DATA_DIR, skipping ---"
  fi

  if [ ! -f "$TOKENIZER_DIR/vocab.json" ]; then
    echo "--- [$fold] train HF character-level tokenizer ---"
    uv run --extra gpu python scripts/train_ainu_hf_tokenizer.py \
      --corpus-dir ainu_corpus \
      --save-dir   "$TOKENIZER_DIR" \
      --fold "$fold"
  else
    echo "--- [$fold] tokenizer already exists at $TOKENIZER_DIR, skipping ---"
  fi

  echo "--- [$fold] train + eval (max_steps=$MAX_STEPS) ---"
  uv run --extra gpu --with tensorboard python scripts/finetune_ainu_watanabe_recipe.py \
    --data-dir "$DATA_DIR" \
    --vocab-path "$TOKENIZER_DIR/vocab.json" \
    --output-dir "$CKPT_DIR" \
    --max-steps "$MAX_STEPS"

  echo "--- [$fold] disk cleanup: drop intermediate checkpoints ---"
  find "$CKPT_DIR" -maxdepth 1 -name "checkpoint-*" -type d -exec rm -rf {} +
done

echo
echo "=== aggregating fold${START_FOLD}-$((N_FOLDS - 1)) ==="
uv run --extra gpu python -c "
import json, jiwer
from pathlib import Path

all_refs, all_hyps = [], []
for fold in range($START_FOLD, $N_FOLDS):
    p = Path(f'checkpoints/ainu_watanabe_kfold/fold{fold}/test_predictions.json')
    if not p.exists():
        print(f'fold{fold}: missing ({p})')
        continue
    d = json.load(open(p))
    refs = [r['reference'] for r in d['results']]
    hyps = [r['prediction'] for r in d['results']]
    wer = jiwer.wer(refs, hyps)
    print(f'fold{fold}: n={len(refs)} wer={wer*100:.2f}%')
    all_refs.extend(refs)
    all_hyps.extend(hyps)

if all_refs:
    pooled = jiwer.wer(all_refs, all_hyps)
    print(f'pooled: n={len(all_refs)} wer={pooled*100:.2f}%')
"
