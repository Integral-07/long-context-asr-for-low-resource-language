#!/usr/bin/env python3
"""
既存の dev_eval_log.jsonl(と、あれば diagnostics.jsonl)から
TensorBoardのイベントファイルを作る。

このコードベースはこれまでTensorBoardを一度も使っておらず(wandb統合のみ、
しかもainu実験ではwandb.use: falseで未使用)、train.py自体にSummaryWriterの
呼び出しもない。今後の学習でリアルタイムに使うというより、既に完走した
run(実験⑤公式/⑧本番、max_epochsパイロット)の記録済みJSONLログを
事後的にTensorBoardのscalarダッシュボードとして見られるようにするための
変換スクリプト。

入力ディレクトリ構成(--source-dir): 以下のファイル名パターンを探す。
  <run>_dev_eval_log.jsonl    必須。{"epoch":E,"step":S,"dev_wer":W}
  <run>_diagnostics.jsonl     任意。{"epoch":E,"step":S,"loss":L,...}
<run> はそのまま TensorBoard の run 名(左サイドバーのrun一覧)になる。
例: fold0_short, fold0_long, fold0_short_pilot60 等。

Usage:
  uv run --with tensorboard python scripts/export_tensorboard_logs.py \\
    --source-dir tb_source/official \\
    --logdir tb_logs

  tensorboard --logdir tb_logs
  # ブラウザで http://localhost:6006 を開く
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def export_run(source_dir: Path, run_name: str, logdir: Path):
    writer = SummaryWriter(log_dir=str(logdir / run_name))

    # global_stepには'epoch'ではなく'step'(cur_podcast、recording単位の累積カウンタ、
    # exp/train.py)を使う。条件B(長文脈)はsequence_schedulerによりepochあたりの
    # ステップ数が条件A(短文脈)の約1/48しかないため、epoch数を横軸にすると
    # 全く異なる学習量を同じ目盛りで比較してしまい誤解を招く。
    dev_eval_path = source_dir / f'{run_name}_dev_eval_log.jsonl'
    with open(dev_eval_path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            writer.add_scalar('dev_wer', row['dev_wer'], global_step=row['step'])

    diagnostics_path = source_dir / f'{run_name}_diagnostics.jsonl'
    if diagnostics_path.exists():
        # diagnostics.jsonlはstep単位(1 epoch内に多数行)なので、epoch平均に
        # 落とす。x軸はそのepoch内で観測された最大step(=epoch終了時点の
        # 累積step)を使い、dev_werのstep軸と同じスケールで比較できるようにする。
        sums: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        last_step: dict[int, int] = {}
        with open(diagnostics_path, encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                sums[row['epoch']] += row['loss']
                counts[row['epoch']] += 1
                last_step[row['epoch']] = max(last_step.get(row['epoch'], 0), row['step'])
        for epoch in sorted(sums):
            writer.add_scalar(
                'train_loss_epoch_mean', sums[epoch] / counts[epoch], global_step=last_step[epoch]
            )

    writer.close()
    print(f'wrote {logdir / run_name}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--logdir', required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    logdir = Path(args.logdir)

    run_names = sorted({
        p.name.replace('_dev_eval_log.jsonl', '')
        for p in source_dir.glob('*_dev_eval_log.jsonl')
    })
    if not run_names:
        raise SystemExit(f'no *_dev_eval_log.jsonl found under {source_dir}')

    for run_name in run_names:
        export_run(source_dir, run_name, logdir)

    print(f'\n{len(run_names)} runs written to {logdir}')
    print(f'  tensorboard --logdir {logdir}')


if __name__ == '__main__':
    main()
