#!/bin/bash
# タスク「過学習していなければmax_epochsを増やす」のパイロット実行。
# fold0の条件A/Bについて、実験⑧本番runと全く同じ設定(lr=3e-5, dev分割,
# sequence_scheduler)のまま max_epochs だけ 30 -> 45 に伸ばして再学習する。
# PROGRESS_LOG.mdの可視化(scripts/plot_training_curves.py)で、5fold全てが
# epoch15〜20でdev WERがプラトーし明確な過学習が見られなかったことを受けて、
# 「もっと学習すれば改善が続くか、頭打ちのままか」を確認するのが目的。
#
# 本番runと衝突しないよう checkpoints/<PILOT_PREFIX>/ に保存する
# (本番runのcheckpoints/ainu_wav2vec2_kfold/はそのまま残る)。
# データ(data/ainu_wav2vec2_kfold_{utterances,collections}/fold{N})・
# トークナイザ(lcasr/artifacts/ainu_kfold/fold{N})は本番runのものを再利用する
# (再生成不要)。
#
# scripts/run_ainu_wav2vec2_kfold.sh(本番5fold driver)と同様、複数foldを
# 1プロセスで順に流せるようN_FOLDS/START_FOLDを取る。config生成
# (scripts/make_moreepochs_pilot_configs.py)もこのスクリプト内で毎回まとめて
# 行うため、事前準備は不要(既存のconfigは上書きされるだけ)。
#
# Usage:
#   bash scripts/run_ainu_wav2vec2_moreepochs_pilot.sh [N_FOLDS] [START_FOLD] [MAX_EPOCHS] [PILOT_PREFIX]
#
# 例: fold0-2の3fold分をmax_epochs=45でSSH切断後も継続するように流す場合:
#   nohup bash scripts/run_ainu_wav2vec2_moreepochs_pilot.sh 3 0 45 > moreepochs_pilot_run.log 2>&1 &

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

N_FOLDS="${1:-1}"
START_FOLD="${2:-0}"
MAX_EPOCHS="${3:-45}"
PREFIX="ainu_wav2vec2"
PILOT_PREFIX="${4:-ainu_wav2vec2_moreepochs_pilot}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="exp/configs/${PILOT_PREFIX}"
FOLDS_TO_RUN=$(seq "$START_FOLD" $((START_FOLD + N_FOLDS - 1)))

echo "=== generating moreepochs pilot configs (folds: $FOLDS_TO_RUN, max_epochs=$MAX_EPOCHS, ckpt_prefix=$PILOT_PREFIX) ==="
uv run --extra gpu python scripts/make_moreepochs_pilot_configs.py \
  --folds $FOLDS_TO_RUN \
  --max-epochs "$MAX_EPOCHS" \
  --dst-dir "$CONFIG_DIR" \
  --ckpt-prefix "$PILOT_PREFIX"

for fold in $FOLDS_TO_RUN; do
  echo
  echo "############################################"
  echo "# [moreepochs pilot] fold $fold"
  echo "############################################"

  TOKENIZER="lcasr/artifacts/ainu_kfold/fold${fold}/tokenizer.model"
  if [ ! -f "$TOKENIZER" ]; then
    echo "ERROR: $TOKENIZER が見つかりません。" >&2
    exit 1
  fi

  echo "--- [$fold] train 条件A (short, max_epochs pilot) ---"
  uv run --extra gpu python exp/train.py --config "$CONFIG_DIR/fold${fold}_short.yaml"

  echo "--- [$fold] train 条件B (long, max_epochs pilot) ---"
  uv run --extra gpu python exp/train.py --config "$CONFIG_DIR/fold${fold}_long.yaml"

  SHORT_CKPT_DIR="checkpoints/${PILOT_PREFIX}/fold${fold}_short"
  LONG_CKPT_DIR="checkpoints/${PILOT_PREFIX}/fold${fold}_long"

  echo "--- [$fold] eval 条件A short (checkpoint: $SHORT_CKPT_DIR/best_dev/best_dev.pt) ---"
  uv run --extra gpu python scripts/eval_ainu_lcasr.py \
    --checkpoint "$SHORT_CKPT_DIR/best_dev/best_dev.pt" \
    --mapping    "data/${PREFIX}_kfold_utterances/fold${fold}/test_mapping.json" \
    --tokenizer  "$TOKENIZER" \
    --output     "$SHORT_CKPT_DIR/test_predictions.json"

  echo "--- [$fold] eval 条件B long (checkpoint: $LONG_CKPT_DIR/best_dev/best_dev.pt) ---"
  uv run --extra gpu python scripts/eval_ainu_lcasr.py \
    --checkpoint "$LONG_CKPT_DIR/best_dev/best_dev.pt" \
    --mapping    "data/${PREFIX}_kfold_collections/fold${fold}/test_mapping.json" \
    --tokenizer  "$TOKENIZER" \
    --output     "$LONG_CKPT_DIR/test_predictions.json"

  echo "--- [$fold] disk cleanup: drop intermediate checkpoints, keep best_dev/dev_eval_log.jsonl + test_predictions.json ---"
  find "$SHORT_CKPT_DIR" "$LONG_CKPT_DIR" -name "step_*.pt" -delete

  echo
  echo "=== [$fold] done. compare against official fold${fold} (max_epochs=30): ==="
  echo "  official short: checkpoints/${PREFIX}_kfold/fold${fold}_short/test_predictions.json"
  echo "  official long:  checkpoints/${PREFIX}_kfold/fold${fold}_long/test_predictions.json"
  echo "  pilot short:    $SHORT_CKPT_DIR/test_predictions.json"
  echo "  pilot long:     $LONG_CKPT_DIR/test_predictions.json"
done

echo
echo "=== all folds done: $FOLDS_TO_RUN (max_epochs=$MAX_EPOCHS) ==="
