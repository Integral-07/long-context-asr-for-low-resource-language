#!/usr/bin/env python3
"""
k-foldの駆動スクリプト(例: scripts/run_ainu_wav2vec2_kfold.sh)が吐く
標準出力ログから、epoch境界ごとの dev WER 履歴を復元する。

`exp/train.py` はdev分割使用時に `<checkpointing.dir>/best_dev/dev_eval_log.jsonl`
を書く。通常はまずこれを直接見ればよい(例: 実験⑤公式/⑧本番runなら
checkpoints/ainu_wav2vec2_kfold/fold{N}_{short,long}/best_dev/dev_eval_log.jsonl
に残っている)。このスクリプトは、checkpointディレクトリ丸ごと削除などで
そのログ自体が失われてしまった場合のフォールバックで、標準出力に出ていた

    --- [<fold>] train 条件A (short) ---
    --- [<fold>] train 条件B (long) ---
    [dev eval] epoch <E>: dev_wer=<W> (best so far: <B>)

という行を駆動スクリプトのログファイル(nohup等でリダイレクトしたもの)から
再パースして復元する。

【紛らわしい点】checkpoints/ainu_wav2vec2_bestepoch_kfold/ という似た名前の
ディレクトリは、dev分割とは無関係の別実験(lr=3e-4でのepoch平均train loss
最良選択、INVESTIGATION.mdタスク#9)用で、そもそもbest_devディレクトリを
持たない。dev分割実験のログと混同しないこと。

複数ログファイル(クラッシュ後にresumeして再実行した分)を時系列順に渡すと、
同じ(fold, condition, epoch)は後に出てきた方で上書きする。

一部のfoldは駆動スクリプトのループ内でクラッシュした後、`exp/train.py`を
個別に直接再実行して復旧したため、上記のfold/condition境界マーカーが
ログに存在しない(例: fold1_long_retry.log)。そのようなログは
`<fold>:<short|long>:<path>` の形式で渡すと、ファイル全体をそのfold/
conditionとして強制的に扱う。

Usage:
  uv run python scripts/extract_dev_wer_history.py \\
    devsplit_production_run.log \\
    devsplit_production_run_resume.log \\
    devsplit_production_run_resume2.log \\
    1:long:fold1_long_retry.log \\
    2:long:fold2_long_retry.log \\
    4:long:fold4_long_retry.log \\
    --output dev_wer_history.jsonl
"""

import argparse
import json
import re
from pathlib import Path

FOLD_RE = re.compile(r'^#\s*fold\s+(\d+)\s*/\s*\d+\s*$')
CONDITION_RE = re.compile(r'^---\s*\[(\d+)\]\s*train\s*条件([AB])\s*\((short|long)\)\s*---$')
DEV_EVAL_RE = re.compile(r'^\[dev eval\]\s*epoch\s+(\d+):\s*dev_wer=([\d.]+)')
FORCED_CONTEXT_RE = re.compile(r'^(\d+):(short|long):(.+)$')


def parse_log(path: Path, history: dict, forced_fold: int | None = None, forced_condition: str | None = None):
    cur_fold = forced_fold
    cur_condition = forced_condition  # 'short' or 'long'
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if forced_fold is None:
                m = FOLD_RE.match(line)
                if m:
                    cur_fold = int(m.group(1))
                    cur_condition = None
                    continue
                m = CONDITION_RE.match(line)
                if m:
                    fold_from_marker, _letter, condition = m.groups()
                    cur_fold = int(fold_from_marker)
                    cur_condition = condition
                    continue
            m = DEV_EVAL_RE.match(line)
            if m and cur_fold is not None and cur_condition is not None:
                epoch, dev_wer = int(m.group(1)), float(m.group(2))
                history[(cur_fold, cur_condition, epoch)] = dev_wer


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('logs', nargs='+', help='時系列順(古い→新しい)のログファイル')
    parser.add_argument('--output', required=True, help='復元結果の保存先(.jsonl)')
    args = parser.parse_args()

    history: dict[tuple[int, str, int], float] = {}
    for log_arg in args.logs:
        m = FORCED_CONTEXT_RE.match(log_arg)
        if m:
            fold_str, condition, log_path = m.groups()
            parse_log(Path(log_path), history, forced_fold=int(fold_str), forced_condition=condition)
        else:
            parse_log(Path(log_arg), history)

    rows = [
        {'fold': fold, 'condition': condition, 'epoch': epoch, 'dev_wer': dev_wer}
        for (fold, condition, epoch), dev_wer in history.items()
    ]
    rows.sort(key=lambda r: (r['fold'], r['condition'], r['epoch']))

    with open(args.output, 'w', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')

    n_folds = len({r['fold'] for r in rows})
    n_series = len({(r['fold'], r['condition']) for r in rows})
    print(f'{len(rows)} epoch-level dev_wer entries across {n_series} fold/condition series ({n_folds} folds) -> {args.output}')


if __name__ == '__main__':
    main()
