#!/usr/bin/env python3
"""
タスク「過学習していなければmax_epochsを増やす」のパイロット用。
既存の exp/configs/ainu_wav2vec2_kfold/fold{N}_{short,long}.yaml
(実験⑤公式/⑧本番run、lr=3e-5、dev分割、max_epochs=30)を複製し、
以下だけを変更した新しいconfig一式を書き出す:

  - training.max_epochs: 30 -> --max-epochs(既定45)
  - training.diagnostics_path: 新規追加(epoch別train lossをJSONLに記録。
    本番run自体は記録していなかったため、このパイロットで初めて取れる)
  - checkpointing.dir: checkpoints/ainu_wav2vec2_moreepochs_pilot/... (本番run
    のcheckpoint/dev_eval_log.jsonlと衝突しない新パス)
  - wandb.project_name / wandb.name: 区別のため接尾辞 -moreepochs-pilot を追加
    (wandb自体はuse: false のまま)

data.path / data.dev_path / training.tokenizer_path は変更しない(同じfold
分割・同じper-foldトークナイザ・同じ音声特徴量を再利用、データ再生成は不要)。

Usage:
  uv run --extra gpu python scripts/make_moreepochs_pilot_configs.py \\
    --folds 0 \\
    --max-epochs 45
"""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--src-dir', default='exp/configs/ainu_wav2vec2_kfold')
    parser.add_argument('--dst-dir', default='exp/configs/ainu_wav2vec2_moreepochs_pilot')
    parser.add_argument('--folds', type=int, nargs='+', default=[0])
    parser.add_argument('--max-epochs', type=int, default=45)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        for cond in ('short', 'long'):
            src_path = src_dir / f'fold{fold}_{cond}.yaml'
            with open(src_path, encoding='utf-8') as f:
                cfg = yaml.safe_load(f)

            old_max_epochs = cfg['training']['max_epochs']
            cfg['training']['max_epochs'] = args.max_epochs

            new_ckpt_dir = cfg['checkpointing']['dir'].replace(
                'ainu_wav2vec2_kfold', 'ainu_wav2vec2_moreepochs_pilot')
            cfg['checkpointing']['dir'] = new_ckpt_dir
            cfg['training']['diagnostics_path'] = f'{new_ckpt_dir}/diagnostics.jsonl'

            cfg['wandb']['project_name'] = cfg['wandb']['project_name'] + '-moreepochs-pilot'
            cfg['wandb']['name'] = cfg['wandb']['name'] + '-moreepochs-pilot'

            dst_path = dst_dir / f'fold{fold}_{cond}.yaml'
            with open(dst_path, 'w', encoding='utf-8') as f:
                f.write(f'# 自動生成: scripts/make_moreepochs_pilot_configs.py (元: {src_path})\n')
                f.write(f'# max_epochs: {old_max_epochs} -> {args.max_epochs} '
                        f'(過学習有無の確認パイロット。lr/dev分割/sequence_schedulerは変更なし)\n')
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            print(f'wrote {dst_path} (max_epochs={old_max_epochs}->{args.max_epochs}, ckpt_dir={new_ckpt_dir})')


if __name__ == '__main__':
    main()
