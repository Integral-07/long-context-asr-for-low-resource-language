#!/usr/bin/env python3
"""
training.diagnostics_path のJSONL(exp/train.py::log_training_metricsが出力)から
epoch別の平均train lossを集計し、最良(最小)epochに対応するチェックポイント
(step_*.pt)を選ぶ。「学習率を下げる代わりに、最良epochの重みを使う」検証
(実験⑤条件A/Bのlr=3e-4版)のためのスクリプト。

選び方: 最良epoch中に記録されたstep範囲[lo, hi]のうち、hi以下で最大のstepを
持つチェックポイントを選ぶ(そのepoch終了時点に最も近い保存点)。該当が
無ければlo以上で最小のstepにフォールバックする。

標準出力には選んだチェックポイントのパスのみを1行で出す(シェルの
$(...)で受け取りやすいように)。診断情報は標準エラーに出す。

Usage:
  CKPT=$(python scripts/pick_best_epoch_checkpoint.py \\
    --diagnostics checkpoints/ainu_wav2vec2_bestepoch_kfold/fold0_short/diagnostics.jsonl \\
    --checkpoint-dir checkpoints/ainu_wav2vec2_bestepoch_kfold/fold0_short)
"""

import argparse
import json
import re
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    args = parser.parse_args()

    epoch_loss_sum = {}
    epoch_loss_n = {}
    epoch_step_range = {}

    with open(args.diagnostics, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            e, loss, step = rec['epoch'], rec['loss'], rec['step']
            epoch_loss_sum[e] = epoch_loss_sum.get(e, 0.0) + loss
            epoch_loss_n[e] = epoch_loss_n.get(e, 0) + 1
            lo, hi = epoch_step_range.get(e, (step, step))
            epoch_step_range[e] = (min(lo, step), max(hi, step))

    if not epoch_loss_sum:
        sys.exit(f'ERROR: {args.diagnostics} にレコードがありません')

    mean_loss = {e: epoch_loss_sum[e] / epoch_loss_n[e] for e in epoch_loss_sum}
    best_epoch = min(mean_loss, key=mean_loss.get)
    lo, hi = epoch_step_range[best_epoch]

    ckpt_dir = Path(args.checkpoint_dir)
    available = []
    for p in ckpt_dir.glob('step_*.pt'):
        m = re.match(r'step_(\d+)\.pt$', p.name)
        if m:
            available.append((int(m.group(1)), p))
    available.sort()

    if not available:
        sys.exit(f'ERROR: {ckpt_dir} に step_*.pt が見つかりません')

    candidates = [p for step, p in available if step <= hi]
    if candidates:
        chosen = candidates[-1]
        reason = 'largest step <= best epoch末尾'
    else:
        candidates2 = [p for step, p in available if step >= lo]
        chosen = candidates2[0] if candidates2 else available[-1][1]
        reason = 'fallback: smallest step >= best epoch先頭 (or 最終チェックポイント)'

    print(f'best_epoch={best_epoch} mean_loss={mean_loss[best_epoch]:.4f} '
          f'step_range=[{lo},{hi}] reason=({reason})', file=sys.stderr)
    for e in sorted(mean_loss):
        marker = ' <== best' if e == best_epoch else ''
        print(f'  epoch {e}: mean_loss={mean_loss[e]:.4f} (n={epoch_loss_n[e]}){marker}', file=sys.stderr)
    print(f'chosen_checkpoint={chosen}', file=sys.stderr)

    print(chosen)


if __name__ == '__main__':
    main()
