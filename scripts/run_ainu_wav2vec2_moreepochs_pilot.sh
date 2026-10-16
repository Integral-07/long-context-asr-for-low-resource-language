#!/bin/bash
# タスク「過学習していなければmax_epochsを増やす」のパイロット実行。
# fold0の条件A/Bについて、実験⑧本番runと全く同じ設定(lr=3e-5, dev分割,
# sequence_scheduler)のまま max_epochs だけ 30 -> 45 に伸ばして再学習する。
# PROGRESS_LOG.mdの可視化(scripts/plot_training_curves.py)で、5fold全てが
# epoch15〜20でdev WERがプラトーし明確な過学習が見られなかったことを受けて、
# 「もっと学習すれば改善が続くか、頭打ちのままか」を確認するのが目的。
#
# 本番runと衝突しないよう checkpoints/ainu_wav2vec2_moreepochs_pilot/ に
# 保存する(本番runのcheckpoints/ainu_wav2vec2_kfold/はそのまま残る)。
# データ(data/ainu_wav2vec2_kfold_{utterances,collections}/fold0)・
# トークナイザ(lcasr/artifacts/ainu_kfold/fold0)は本番runのものを再利用する
# (再生成不要)。
#
# 事前準備(このスクリプトの実行前に一度だけ):
#   uv run --extra gpu python scripts/make_moreepochs_pilot_configs.py --folds 0 --max-epochs 45
#
# Usage:
#   bash scripts/run_ainu_wav2vec2_moreepochs_pilot.sh [FOLD]

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

FOLD="${1:-0}"
PREFIX="ainu_wav2vec2"
PILOT_PREFIX="ainu_wav2vec2_moreepochs_pilot"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="exp/configs/${PILOT_PREFIX}"
if [ ! -f "$CONFIG_DIR/fold${FOLD}_short.yaml" ]; then
  echo "ERROR: $CONFIG_DIR/fold${FOLD}_short.yaml がありません。先に scripts/make_moreepochs_pilot_configs.py を実行してください。" >&2
  exit 1
fi

TOKENIZER="lcasr/artifacts/ainu_kfold/fold${FOLD}/tokenizer.model"
if [ ! -f "$TOKENIZER" ]; then
  echo "ERROR: $TOKENIZER が見つかりません。" >&2
  exit 1
fi

echo "--- [$FOLD] train 条件A (short, max_epochs pilot) ---"
uv run --extra gpu python exp/train.py --config "$CONFIG_DIR/fold${FOLD}_short.yaml"

echo "--- [$FOLD] train 条件B (long, max_epochs pilot) ---"
uv run --extra gpu python exp/train.py --config "$CONFIG_DIR/fold${FOLD}_long.yaml"

SHORT_CKPT_DIR="checkpoints/${PILOT_PREFIX}/fold${FOLD}_short"
LONG_CKPT_DIR="checkpoints/${PILOT_PREFIX}/fold${FOLD}_long"

echo "--- [$FOLD] eval 条件A short (checkpoint: $SHORT_CKPT_DIR/best_dev/best_dev.pt) ---"
uv run --extra gpu python scripts/eval_ainu_lcasr.py \
  --checkpoint "$SHORT_CKPT_DIR/best_dev/best_dev.pt" \
  --mapping    "data/${PREFIX}_kfold_utterances/fold${FOLD}/test_mapping.json" \
  --tokenizer  "$TOKENIZER" \
  --output     "$SHORT_CKPT_DIR/test_predictions.json"

echo "--- [$FOLD] eval 条件B long (checkpoint: $LONG_CKPT_DIR/best_dev/best_dev.pt) ---"
uv run --extra gpu python scripts/eval_ainu_lcasr.py \
  --checkpoint "$LONG_CKPT_DIR/best_dev/best_dev.pt" \
  --mapping    "data/${PREFIX}_kfold_collections/fold${FOLD}/test_mapping.json" \
  --tokenizer  "$TOKENIZER" \
  --output     "$LONG_CKPT_DIR/test_predictions.json"

echo "--- [$FOLD] disk cleanup: drop intermediate checkpoints, keep best_dev/dev_eval_log.jsonl + test_predictions.json ---"
find "$SHORT_CKPT_DIR" "$LONG_CKPT_DIR" -name "step_*.pt" -delete

echo
echo "=== done. compare against official fold${FOLD} (max_epochs=30): ==="
echo "  official short: checkpoints/${PREFIX}_kfold/fold${FOLD}_short/test_predictions.json"
echo "  official long:  checkpoints/${PREFIX}_kfold/fold${FOLD}_long/test_predictions.json"
echo "  pilot short:    $SHORT_CKPT_DIR/test_predictions.json"
echo "  pilot long:     $LONG_CKPT_DIR/test_predictions.json"
