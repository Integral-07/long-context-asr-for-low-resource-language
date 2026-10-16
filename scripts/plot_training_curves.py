#!/usr/bin/env python3
"""
1つのfoldについて、条件A(短文脈)・条件B(長文脈)のepoch別dev WER(と、
あれば train loss)を並べてプロットし、過学習(dev WERが学習終盤で反転して
悪化するか)を目視確認できるようにする。

dev WERは `exp/train.py` がdev分割使用時に書く
`<checkpointing.dir>/best_dev/dev_eval_log.jsonl` を直接読む
(--short-dev-eval-log / --long-dev-eval-log)。checkpoint削除等でこれが
失われている場合は、`scripts/extract_dev_wer_history.py` で標準出力ログから
復元した結合jsonl を --dev-wer-jsonl で代わりに渡せる。

【注意 / 既知の罠】checkpoints/ainu_wav2vec2_bestepoch_kfold/ というよく似た
名前のディレクトリが別実験(lr=3e-4でのepoch平均train loss最良選択、
INVESTIGATION.mdタスク#9、dev分割なし)用に存在する。dev分割を使った本実験
(実験⑤公式/⑧)のcheckpointは checkpoints/ainu_wav2vec2_kfold/fold{N}_{short,long}/
配下にある。名前が紛らわしいので必ずどちらを指しているか確認すること。

train lossは、diagnostics_path が設定されていた場合のみ
`<checkpointing.dir>/diagnostics.jsonl` (各stepごとに{epoch, loss, ...})
から読む(--short-diagnostics / --long-diagnostics、省略可)。実験⑤/⑧の
本番configはdiagnostics_pathを設定していないため、通常はdev WERのみの
1段プロットになる。

注意(INVESTIGATION.mdタスク#9で確認済みの既知の落とし穴): train lossが
取れる場合でも、条件B(長文脈)は`sequence_scheduler`によりepochが進むほど
1epoch内のチャンク数(train loss平均の標本数)が減っていく。終盤ほどepoch
平均lossの推定がノイジーになり、条件Aとは同じ土俵で比較できない。

Usage:
  # 本番(dev分割)run。checkpoint/dev_eval_log.jsonlが残っている場合
  uv run python scripts/plot_training_curves.py \\
    --fold 0 \\
    --short-dev-eval-log checkpoints/ainu_wav2vec2_kfold/fold0_short/best_dev/dev_eval_log.jsonl \\
    --long-dev-eval-log  checkpoints/ainu_wav2vec2_kfold/fold0_long/best_dev/dev_eval_log.jsonl \\
    --output fold0_training_curves.png

  # dev_eval_log.jsonlが失われている場合(標準出力ログから復元した結合jsonlを使用)
  uv run python scripts/plot_training_curves.py \\
    --fold 0 --dev-wer-jsonl dev_wer_history.jsonl \\
    --output fold0_training_curves.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_epoch_loss(diagnostics_path: str | None):
    """-> {epoch: (mean_loss, n)}"""
    if diagnostics_path is None:
        return {}
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    with open(diagnostics_path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            sums[row['epoch']] += row['loss']
            counts[row['epoch']] += 1
    return {epoch: (sums[epoch] / counts[epoch], counts[epoch]) for epoch in sums}


def load_dev_wer(dev_wer_jsonl: str | None, fold: int, condition: str):
    """extract_dev_wer_history.py形式(fold/condition列つき)の結合jsonlから -> {epoch: dev_wer}"""
    if dev_wer_jsonl is None:
        return {}
    out = {}
    with open(dev_wer_jsonl, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['fold'] == fold and row['condition'] == condition:
                out[row['epoch']] = row['dev_wer']
    return out


def load_dev_eval_log(dev_eval_log_path: str | None):
    """exp/train.pyが書くbest_dev/dev_eval_log.jsonl(fold/condition列なし)から -> {epoch: dev_wer}"""
    if dev_eval_log_path is None:
        return {}
    out = {}
    with open(dev_eval_log_path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            out[row['epoch']] = row['dev_wer']
    return out


def plot_condition(ax_loss, ax_wer, label, color, epoch_loss, dev_wer):
    if epoch_loss:
        epochs = sorted(epoch_loss)
        losses = [epoch_loss[e][0] for e in epochs]
        ns = [epoch_loss[e][1] for e in epochs]
        ax_loss.plot(epochs, losses, color=color, label=f'{label} train loss')
        ax_n = ax_loss.twinx()
        ax_n.plot(epochs, ns, color=color, alpha=0.25, linestyle=':', linewidth=1)
        ax_n.set_yticks([])
    if dev_wer:
        epochs = sorted(dev_wer)
        wers = [dev_wer[e] for e in epochs]
        ax_wer.plot(epochs, wers, color=color, marker='o', markersize=3, label=f'{label} dev WER')
        best_epoch = min(dev_wer, key=dev_wer.get)
        ax_wer.scatter([best_epoch], [dev_wer[best_epoch]], color=color, marker='*', s=150,
                        zorder=5, edgecolors='black', linewidths=0.5)
        ax_wer.axvline(best_epoch, color=color, linestyle='--', alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--short-diagnostics', default=None,
                        help='条件A(短文脈)のdiagnostics.jsonl(diagnostics_pathが設定されていた場合のみ存在)')
    parser.add_argument('--long-diagnostics', default=None,
                        help='条件B(長文脈)のdiagnostics.jsonl(同上)')
    parser.add_argument('--short-dev-eval-log', default=None,
                        help='条件A(短文脈)のbest_dev/dev_eval_log.jsonl(直接)')
    parser.add_argument('--long-dev-eval-log', default=None,
                        help='条件B(長文脈)のbest_dev/dev_eval_log.jsonl(直接)')
    parser.add_argument('--dev-wer-jsonl', default=None,
                        help='extract_dev_wer_history.py出力(fold/condition列つき結合jsonl)。'
                             'dev_eval_log.jsonlが失われている場合のフォールバック。'
                             '--short/long-dev-eval-logと併用しない。')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    short_loss = load_epoch_loss(args.short_diagnostics)
    long_loss = load_epoch_loss(args.long_diagnostics)
    short_wer = load_dev_eval_log(args.short_dev_eval_log) or load_dev_wer(args.dev_wer_jsonl, args.fold, 'short')
    long_wer = load_dev_eval_log(args.long_dev_eval_log) or load_dev_wer(args.dev_wer_jsonl, args.fold, 'long')

    has_loss = bool(short_loss or long_loss)
    if has_loss:
        fig, (ax_loss, ax_wer) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    else:
        fig, ax_wer = plt.subplots(1, 1, figsize=(10, 5))
        ax_loss = None

    plot_condition(ax_loss, ax_wer, 'A: short-context', 'tab:blue', short_loss, short_wer)
    plot_condition(ax_loss, ax_wer, 'B: long-context', 'tab:orange', long_loss, long_wer)

    if has_loss:
        ax_loss.set_yscale('log')
        ax_loss.set_ylabel('train loss (per-epoch mean, log scale)')
        ax_loss.set_title(f'fold {args.fold}: train loss / dev WER per epoch\n'
                           '(dotted = per-epoch sample count n; condition B\'s n shrinks under\n'
                           'sequence_scheduler, so its late-epoch loss is not directly comparable)',
                           fontsize=10)
        ax_loss.legend(loc='upper right', fontsize=8)
    else:
        ax_wer.set_title(f'fold {args.fold}: dev WER per epoch (no diagnostics_path logged for this run)')

    ax_wer.set_xlabel('epoch')
    ax_wer.set_ylabel('dev WER')
    ax_wer.set_ylim(0, 1)
    ax_wer.legend(loc='upper right', fontsize=8)
    ax_wer.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f'saved {args.output}')


if __name__ == '__main__':
    main()
