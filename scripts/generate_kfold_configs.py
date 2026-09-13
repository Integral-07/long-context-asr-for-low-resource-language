#!/usr/bin/env python3
"""
exp/configs/ainu_scratch_short.yaml / ainu_scratch_long.yaml をテンプレートに、
k-fold交差検証用の設定ファイルを N_FOLDS 個ずつ生成する。

モデル・オプティマイザ・学習レシピは元のyamlから一切変更しない。
書き換えるのは以下の3つのパスだけ(foldごとのデータ/チェックポイント/トークナイザを
指し示すため):
  - data.path
  - checkpointing.dir
  - training.tokenizer_path

前提として、以下がすでに各foldごとに用意されていること(生成はしない、
このスクリプトは設定ファイルだけを作る):
  - data/ainu_kfold/fold{N}/train_mapping.json, test_mapping.json      (③長い音声用, prepare_ainu.py --fold N)
  - data/ainu_utterance_kfold/fold{N}/train_mapping.json, test_mapping.json (②短い音声用, prepare_ainu_utterances.py --fold N)
  - lcasr/artifacts/ainu_kfold/fold{N}/tokenizer.model                 (train_ainu_tokenizer.py --fold N)

Usage:
  python scripts/generate_kfold_configs.py \\
    --remote-root /home/t23cs033/long-context-asr \\
    --output-dir exp/configs/ainu_kfold
"""

import argparse
import copy
from pathlib import Path

import yaml

HEADER = (
    "# 自動生成: scripts/generate_kfold_configs.py\n"
    "# モデル/学習レシピは exp/configs/ainu_scratch_{{short,long}}.yaml と同一。\n"
    "# 変更点は data.path / checkpointing.dir / training.tokenizer_path のみ(fold {fold} 用)。\n"
)


def load_template(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--short-template', default='exp/configs/ainu_scratch_short.yaml')
    parser.add_argument('--long-template', default='exp/configs/ainu_scratch_long.yaml')
    parser.add_argument('--output-dir', default='exp/configs/ainu_kfold')
    parser.add_argument('--n-folds', type=int, default=None,
                        help='省略時は ainu_kfold_splits.N_FOLDS を使う')
    parser.add_argument('--remote-root', default='/home/t23cs033/long-context-asr',
                        help='data/checkpoints/tokenizerの絶対パスの起点(学習を実行するホスト上のパス)')
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ainu_kfold_splits import N_FOLDS
    n_folds = args.n_folds or N_FOLDS

    root = args.remote_root.rstrip('/')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    short_template = load_template(Path(args.short_template))
    long_template = load_template(Path(args.long_template))

    for fold in range(n_folds):
        # --- short (②, 発話単位) ---
        cfg = copy.deepcopy(short_template)
        cfg['data']['path'] = f'{root}/data/ainu_utterance_kfold/fold{fold}/train_mapping.json'
        cfg['checkpointing']['dir'] = f'{root}/checkpoints/ainu_kfold/fold{fold}_short'
        cfg['training']['tokenizer_path'] = f'{root}/lcasr/artifacts/ainu_kfold/fold{fold}/tokenizer.model'
        cfg['wandb']['project_name'] = f'ainu_kfold_fold{fold}_short'
        cfg['wandb']['name'] = f'ainu-kfold-fold{fold}-short'
        out_path = output_dir / f'fold{fold}_short.yaml'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(HEADER.format(fold=fold))
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f'wrote {out_path}')

        # --- long (③, コレクション連結) ---
        cfg = copy.deepcopy(long_template)
        cfg['data']['path'] = f'{root}/data/ainu_kfold/fold{fold}/train_mapping.json'
        cfg['checkpointing']['dir'] = f'{root}/checkpoints/ainu_kfold/fold{fold}_long'
        cfg['training']['tokenizer_path'] = f'{root}/lcasr/artifacts/ainu_kfold/fold{fold}/tokenizer.model'
        cfg['wandb']['project_name'] = f'ainu_kfold_fold{fold}_long'
        cfg['wandb']['name'] = f'ainu-kfold-fold{fold}-long'
        out_path = output_dir / f'fold{fold}_long.yaml'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(HEADER.format(fold=fold))
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f'wrote {out_path}')

    print(f'\n{n_folds} folds x 2 conditions = {n_folds * 2} configs written under {output_dir}/')


if __name__ == '__main__':
    main()
