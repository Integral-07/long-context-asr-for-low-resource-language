#!/bin/bash
# ②(scratch-short) vs ③(scratch-long) のコレクション単位k-fold交差検証を
# 通して実行するドライバ。GPUホスト(このリポジトリでは dl12)上で、
# リポジトリルートから実行することを想定している。
#
# 各foldについて:
#   1. データ準備 (②: 発話単位 / ③: コレクション連結)
#   2. そのfoldのtrainテキストだけからトークナイザを学習(vocab leakage防止)
#   3. ②③それぞれ学習
#   4. ②③それぞれtestで評価 (test_predictions.json を出力)
# 最後に、全foldの結果を scripts/aggregate_kfold_results.py でプールして
# 統計的有意性を検定する。
#
# 実行前に一度 scripts/generate_kfold_configs.py で
# exp/configs/ainu_kfold/fold{N}_{short,long}.yaml を生成しておくこと。
#
# Usage:
#   bash scripts/run_ainu_kfold.sh [N_FOLDS]

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"  # uv がログインシェル経由でないと PATH に無いことがある

N_FOLDS="${1:-5}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 全ステップを --extra gpu で統一する。--extra cpu は本環境ではtorchaudioの
# バイナリ不整合(undefined symbol: aoti_torch_abi_version)で壊れていた。
# データ準備自体はCPUだけで動く処理だが、gpu extraのtorch/torchaudioで
# 動かしても問題はない。

echo "=== generating configs for $N_FOLDS folds ==="
uv run --extra gpu python scripts/generate_kfold_configs.py --n-folds "$N_FOLDS"

for fold in $(seq 0 $((N_FOLDS - 1))); do
  echo
  echo "############################################"
  echo "# fold $fold / $((N_FOLDS - 1))"
  echo "############################################"

  echo "--- [$fold] prepare_ainu.py (③ long, collection-concatenated) ---"
  uv run --extra gpu python scripts/prepare_ainu.py \
    --corpus-dir ainu_corpus \
    --spec-dir   "ainu_kfold_processed/fold${fold}" \
    --output-dir "data/ainu_kfold/fold${fold}" \
    --fold "$fold"

  echo "--- [$fold] prepare_ainu_utterances.py (② short, per-utterance) ---"
  uv run --extra gpu python scripts/prepare_ainu_utterances.py \
    --corpus-dir ainu_corpus \
    --spec-dir   "ainu_utterance_kfold_processed/fold${fold}" \
    --output-dir "data/ainu_utterance_kfold/fold${fold}" \
    --fold "$fold"

  echo "--- [$fold] train_ainu_tokenizer.py (foldのtrainのみから、vocab 500) ---"
  uv run --extra gpu python scripts/train_ainu_tokenizer.py \
    --corpus-dir ainu_corpus \
    --save-dir   "lcasr/artifacts/ainu_kfold/fold${fold}" \
    --vocab-size 500 \
    --fold "$fold"

  echo "--- [$fold] train ② short ---"
  uv run --extra gpu python exp/train.py --config "exp/configs/ainu_kfold/fold${fold}_short.yaml"

  echo "--- [$fold] train ③ long ---"
  uv run --extra gpu python exp/train.py --config "exp/configs/ainu_kfold/fold${fold}_long.yaml"

  SHORT_CKPT_DIR="checkpoints/ainu_kfold/fold${fold}_short"
  LONG_CKPT_DIR="checkpoints/ainu_kfold/fold${fold}_long"
  SHORT_LAST_CKPT="$(ls -t "$SHORT_CKPT_DIR"/step_*.pt | head -1)"
  LONG_LAST_CKPT="$(ls -t "$LONG_CKPT_DIR"/step_*.pt | head -1)"

  echo "--- [$fold] eval ② short (checkpoint: $SHORT_LAST_CKPT) ---"
  uv run --extra gpu python scripts/eval_ainu_lcasr.py \
    --checkpoint "$SHORT_LAST_CKPT" \
    --mapping    "data/ainu_utterance_kfold/fold${fold}/test_mapping.json" \
    --tokenizer  "lcasr/artifacts/ainu_kfold/fold${fold}/tokenizer.model" \
    --output     "$SHORT_CKPT_DIR/test_predictions.json"

  echo "--- [$fold] eval ③ long (checkpoint: $LONG_LAST_CKPT) ---"
  uv run --extra gpu python scripts/eval_ainu_lcasr.py \
    --checkpoint "$LONG_LAST_CKPT" \
    --mapping    "data/ainu_kfold/fold${fold}/test_mapping.json" \
    --tokenizer  "lcasr/artifacts/ainu_kfold/fold${fold}/tokenizer.model" \
    --output     "$LONG_CKPT_DIR/test_predictions.json"
done

echo
echo "=== aggregating all $N_FOLDS folds ==="
python scripts/aggregate_kfold_results.py --checkpoints-dir checkpoints/ainu_kfold --n-folds "$N_FOLDS"
