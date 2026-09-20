#!/usr/bin/env python3
"""
exp/configs/ainu_wav2vec2_{short,long}.yaml をテンプレートに、実験⑤
(karol wav2vec2-Ainu + 長文脈シーケンス長ウォームアップ)のk-fold交差検証用
設定ファイルを N_FOLDS 個ずつ生成する。scripts/generate_kfold_configs.py
(実験②③用)のwav2vec2版。

モデル・オプティマイザ・学習レシピは元のyamlから一切変更しない。書き換える
のは以下の3つのパスだけ(foldごとのデータ/チェックポイント/トークナイザを
指し示すため):
  - data.path
  - checkpointing.dir
  - training.tokenizer_path

トークナイザは実験④で既に作成済みの per-fold トークナイザ
(lcasr/artifacts/ainu_kfold/fold{N}/tokenizer.model, BPE-500)をそのまま
再利用する(実験⑤は実験②③と同じトークナイザ選択のため、fold単位で
学習し直す必要はない)。

前提として、以下がすでに各foldごとに用意されていること(生成はしない、
このスクリプトは設定ファイルだけを作る):
  - data/ainu_wav2vec2_kfold_utterances/fold{N}/train_mapping.json, test_mapping.json
    (条件A用, prepare_ainu_wav2vec2_utterances.py --fold N)
  - data/ainu_wav2vec2_kfold_collections/fold{N}/train_mapping.json, test_mapping.json
    (条件B用, prepare_ainu_wav2vec2_longcontext.py --fold N)
  - lcasr/artifacts/ainu_kfold/fold{N}/tokenizer.model (実験④で作成済み)

Usage:
  python scripts/generate_kfold_configs_wav2vec2.py \\
    --remote-root /home/t23cs033/long-context-asr \\
    --output-dir exp/configs/ainu_wav2vec2_kfold
"""

import argparse
import copy
from pathlib import Path

import yaml

HEADER = (
    "# 自動生成: scripts/generate_kfold_configs_wav2vec2.py\n"
    "# モデル/学習レシピは exp/configs/ainu_wav2vec2_{{short,long}}.yaml と同一。\n"
    "# 変更点は data.path / checkpointing.dir / training.tokenizer_path のみ(fold {fold} 用)。\n"
)


def load_template(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--short-template', default='exp/configs/ainu_wav2vec2_short.yaml')
    parser.add_argument('--long-template', default='exp/configs/ainu_wav2vec2_long.yaml')
    parser.add_argument('--output-dir', default='exp/configs/ainu_wav2vec2_kfold')
    parser.add_argument('--name-prefix', default='ainu_wav2vec2',
                        help='data/checkpoints/wandbのfold別パスに使う接頭辞。'
                             'ベースモデルが違う実験(例: 実験⑥は ainu_xlsr300m)では、'
                             'data/ainu_wav2vec2_kfold_*を上書きしないよう変更すること。')
    parser.add_argument('--n-folds', type=int, default=None,
                        help='省略時は ainu_kfold_splits.N_FOLDS を使う')
    parser.add_argument('--remote-root', default='/home/t23cs033/long-context-asr',
                        help='data/checkpoints/tokenizerの絶対パスの起点(学習を実行するホスト上のパス)')
    args = parser.parse_args()
    prefix = args.name_prefix

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
        # --- short (条件A, 発話単位) ---
        cfg = copy.deepcopy(short_template)
        cfg['data']['path'] = f'{root}/data/{prefix}_kfold_utterances/fold{fold}/train_mapping.json'
        cfg['checkpointing']['dir'] = f'{root}/checkpoints/{prefix}_kfold/fold{fold}_short'
        cfg['training']['tokenizer_path'] = f'{root}/lcasr/artifacts/ainu_kfold/fold{fold}/tokenizer.model'
        cfg['wandb']['project_name'] = f'{prefix}_kfold_fold{fold}_short'
        cfg['wandb']['name'] = f'{prefix}-kfold-fold{fold}-short'
        out_path = output_dir / f'fold{fold}_short.yaml'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(HEADER.format(fold=fold))
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f'wrote {out_path}')

        # --- long (条件B, コレクション連結) ---
        cfg = copy.deepcopy(long_template)
        cfg['data']['path'] = f'{root}/data/{prefix}_kfold_collections/fold{fold}/train_mapping.json'
        cfg['checkpointing']['dir'] = f'{root}/checkpoints/{prefix}_kfold/fold{fold}_long'
        cfg['training']['tokenizer_path'] = f'{root}/lcasr/artifacts/ainu_kfold/fold{fold}/tokenizer.model'
        cfg['wandb']['project_name'] = f'{prefix}_kfold_fold{fold}_long'
        cfg['wandb']['name'] = f'{prefix}-kfold-fold{fold}-long'
        out_path = output_dir / f'fold{fold}_long.yaml'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(HEADER.format(fold=fold))
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f'wrote {out_path}')

    print(f'\n{n_folds} folds x 2 conditions = {n_folds * 2} configs written under {output_dir}/')


if __name__ == '__main__':
    main()
