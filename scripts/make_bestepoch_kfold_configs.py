#!/usr/bin/env python3
"""
実験⑤の「学習率を下げる代わりに最良epochを使う」検証用に、既存の
exp/configs/ainu_wav2vec2_kfold/fold{N}_{short,long}.yaml (lr=3e-5, 現行の
公式lr=3e-5結果を生成したconfig)を複製し、以下だけを変更した新しいconfig一式を
exp/configs/ainu_wav2vec2_bestepoch_kfold/ に書き出す:
  - optimizer.args.lr: 3e-5 -> 3e-4 (元の不安定な学習率に戻す)
  - training.diagnostics_path: 新規追加(epoch別lossをJSONLに記録)
  - checkpointing.dir: checkpoints/ainu_wav2vec2_bestepoch_kfold/... (公式lr=3e-5結果と衝突しない新パス)
  - wandb.project_name / wandb.name: 区別のため接尾辞 -bestepoch を追加(wandb自体はuse:false)

data.path / training.tokenizer_path は既存configから変更しない(同じfold分割・
同じper-foldトークナイザを再利用、データ再生成は不要)。

Usage:
  python scripts/make_bestepoch_kfold_configs.py \\
    --src-dir exp/configs/ainu_wav2vec2_kfold \\
    --dst-dir exp/configs/ainu_wav2vec2_bestepoch_kfold \\
    --n-folds 5
"""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src-dir', default='exp/configs/ainu_wav2vec2_kfold')
    parser.add_argument('--dst-dir', default='exp/configs/ainu_wav2vec2_bestepoch_kfold')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(args.n_folds):
        for cond in ('short', 'long'):
            src_path = src_dir / f'fold{fold}_{cond}.yaml'
            with open(src_path, encoding='utf-8') as f:
                cfg = yaml.safe_load(f)

            old_lr = cfg['optimizer']['args']['lr']
            cfg['optimizer']['args']['lr'] = args.lr

            new_ckpt_dir = cfg['checkpointing']['dir'].replace(
                'ainu_wav2vec2_kfold', 'ainu_wav2vec2_bestepoch_kfold')
            cfg['checkpointing']['dir'] = new_ckpt_dir

            diagnostics_path = f'{new_ckpt_dir}/diagnostics.jsonl'
            cfg['training']['diagnostics_path'] = diagnostics_path

            cfg['wandb']['project_name'] = cfg['wandb']['project_name'] + '-bestepoch'
            cfg['wandb']['name'] = cfg['wandb']['name'] + '-bestepoch'

            dst_path = dst_dir / f'fold{fold}_{cond}.yaml'
            with open(dst_path, 'w', encoding='utf-8') as f:
                f.write(f'# 自動生成: scripts/make_bestepoch_kfold_configs.py (元: {src_path})\n')
                f.write(f'# lr: {old_lr} -> {args.lr} (最良epoch選択方式の検証用、self_conditioningは有効のまま)\n')
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            print(f'wrote {dst_path} (lr={old_lr}->{args.lr}, ckpt_dir={new_ckpt_dir})')


if __name__ == '__main__':
    main()
