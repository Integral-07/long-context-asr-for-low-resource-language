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
#   bash scripts/run_ainu_wav2vec2_kfold.sh [N_FOLDS] [START_FOLD] [BASE_MODEL] [NAME_PREFIX]
#
# START_FOLD(省略時0)を指定すると、そのfoldから再開する(configの再生成は
# 常に行う。既存のconfigファイルは上書きされるだけで、それ以前のfoldの
# 学習済みチェックポイント/test_predictions.jsonはそのまま)。長時間ジョブが
# 途中のfoldでクラッシュ(例: GPUメモリ断片化によるOOM)した場合に、
# 完了済みfoldを再学習せずに再開するために使う。
#
# BASE_MODEL/NAME_PREFIX(省略時は実験⑤のkarolモデル)を変えると、同じ
# アーキテクチャ(wav2vec2-large系, hidden_size 1024・24層)の別の事前学習
# 済みモデルでも同じ手順を再利用できる。例(実験⑥, facebook/wav2vec2-xls-r-300m):
#   bash scripts/run_ainu_wav2vec2_kfold.sh 5 0 facebook/wav2vec2-xls-r-300m ainu_xlsr300m
# この場合、exp/configs/ainu_xlsr300m_{short,long}.yaml が存在すること
# (model.base_model が対応するBASE_MODELになっている必要がある)。

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"  # uv がログインシェル経由でないと PATH に無いことがある
# PyTorchのキャッシュアロケータが可変長シーケンス(条件Aの発話単位バッチは
# 長さが毎回違う)で断片化し、"reserved but unallocated"が積み上がって
# 十分な空きがあるはずなのにOOMする事象がfold2で発生したための対策
# (PyTorch公式ドキュメント推奨)。
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

N_FOLDS="${1:-5}"
START_FOLD="${2:-0}"
BASE_MODEL="${3:-karolnowakowski/wav2vec2-large-xlsr-53-pretrain-ain}"
PREFIX="${4:-ainu_wav2vec2}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== generating configs for $N_FOLDS folds (prefix=$PREFIX) ==="
uv run --extra gpu python scripts/generate_kfold_configs_wav2vec2.py \
  --n-folds "$N_FOLDS" \
  --short-template "exp/configs/${PREFIX}_short.yaml" \
  --long-template  "exp/configs/${PREFIX}_long.yaml" \
  --output-dir     "exp/configs/${PREFIX}_kfold" \
  --name-prefix    "$PREFIX"

for fold in $(seq "$START_FOLD" $((N_FOLDS - 1))); do
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
    --feat-dir   "${PREFIX}_kfold_utterance_processed/fold${fold}" \
    --output-dir "data/${PREFIX}_kfold_utterances/fold${fold}" \
    --base-model "$BASE_MODEL" \
    --fold "$fold"

  echo "--- [$fold] prepare_ainu_wav2vec2_longcontext.py (条件B, コレクション連結) ---"
  uv run --extra gpu python scripts/prepare_ainu_wav2vec2_longcontext.py \
    --corpus-dir ainu_corpus \
    --feat-dir   "${PREFIX}_kfold_processed/fold${fold}" \
    --output-dir "data/${PREFIX}_kfold_collections/fold${fold}" \
    --base-model "$BASE_MODEL" \
    --fold "$fold"

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
    --mapping    "data/${PREFIX}_kfold_utterances/fold${fold}/test_mapping.json" \
    --tokenizer  "$TOKENIZER" \
    --output     "$SHORT_CKPT_DIR/test_predictions.json"

  echo "--- [$fold] eval 条件B long (checkpoint: $LONG_LAST_CKPT) ---"
  uv run --extra gpu python scripts/eval_ainu_lcasr.py \
    --checkpoint "$LONG_LAST_CKPT" \
    --mapping    "data/${PREFIX}_kfold_collections/fold${fold}/test_mapping.json" \
    --tokenizer  "$TOKENIZER" \
    --output     "$LONG_CKPT_DIR/test_predictions.json"

  echo "--- [$fold] disk cleanup: drop intermediate checkpoints, keep test_predictions.json ---"
  find "$SHORT_CKPT_DIR" "$LONG_CKPT_DIR" -name "step_*.pt" -delete
done

echo
echo "=== aggregating all $N_FOLDS folds ==="
uv run --extra gpu python scripts/aggregate_kfold_results.py --checkpoints-dir "checkpoints/${PREFIX}_kfold" --n-folds "$N_FOLDS"
