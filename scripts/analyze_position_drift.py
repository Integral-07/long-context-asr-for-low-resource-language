#!/usr/bin/env python3
"""
コレクション内の相対位置(参照テキストの先頭からの単語インデックス割合)と
ローカル誤り率の関係を見る。タスク#7で見つけた「収録後半ほど予測が劣化する
ドリフト」仮説を定量化するための分析。

各コレクション(長文脈条件のtest_predictions.json、1コレクション=1エントリ)
についてjiwerのアラインメントを取り、参照単語ごとに正解/誤り(置換・削除)を
判定する。挿入誤りは、jiwerのアラインメントチャンクが示す参照側の挿入位置
(ref_start_idx, 通常ゼロ幅)に割り当てる。参照単語のインデックスをコレクション
長で正規化し(0=先頭, 1=末尾)、10分位(decile)ごとに誤り数/単語数をプールして
ローカルWERを算出、先頭から末尾にかけて悪化する傾向をSpearman相関で確認する。

Usage:
  uv run --extra gpu python scripts/analyze_position_drift.py \\
    --checkpoints-dir checkpoints/ainu_xlsr300m_kfold --n-folds 5
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import jiwer
except ImportError:
    sys.exit('jiwer が必要です')

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

N_DECILES = 10


def decile_errors(reference: str, hypothesis: str, err_words: list, n_words: list):
    ref_words = reference.split()
    n = len(ref_words)
    if n == 0:
        return

    out = jiwer.process_words(reference, hypothesis)
    alignment = out.alignments[0]

    is_error = [False] * n
    insert_positions = []

    for chunk in alignment:
        if chunk.type == 'equal':
            continue
        elif chunk.type in ('substitute', 'delete'):
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                is_error[i] = True
        elif chunk.type == 'insert':
            pos = min(chunk.ref_start_idx, n - 1)
            insert_positions.extend([pos] * (chunk.hyp_end_idx - chunk.hyp_start_idx))

    for i in range(n):
        decile = min(int(i / n * N_DECILES), N_DECILES - 1)
        n_words[decile] += 1
        if is_error[i]:
            err_words[decile] += 1

    for pos in insert_positions:
        decile = min(int(pos / n * N_DECILES), N_DECILES - 1)
        err_words[decile] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints-dir', required=True)
    parser.add_argument('--condition', default='long')
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()

    err_words = [0] * N_DECILES
    n_words = [0] * N_DECILES
    n_collections = 0

    ckpt_dir = Path(args.checkpoints_dir)
    for fold in range(args.n_folds):
        path = ckpt_dir / f'fold{fold}_{args.condition}' / 'test_predictions.json'
        if not path.exists():
            print(f'WARN: {path} が見つかりません、スキップ')
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        for r in data['results']:
            decile_errors(r['reference'], r['prediction'], err_words, n_words)
            n_collections += 1

    print(f'=== {args.checkpoints_dir} / {args.condition}: {n_collections} コレクション ===\n')
    print(f'{"decile":>7s} {"words":>8s} {"errors":>8s} {"local_WER":>10s}')
    local_wers = []
    for d in range(N_DECILES):
        wer = err_words[d] / n_words[d] * 100 if n_words[d] else float('nan')
        local_wers.append(wer)
        print(f'{d:>7d} {n_words[d]:>8d} {err_words[d]:>8d} {wer:>9.2f}%')

    if spearmanr is not None:
        r, p = spearmanr(list(range(N_DECILES)), local_wers)
        print(f'\nSpearman r(decile位置, local WER) = {r:+.3f} (p={p:.4f})')
        print('(正の相関 = 収録後半ほど誤り率が高い = ドリフト仮説を支持)')
    else:
        print('\nscipy未インストールのため相関係数は計算せず')


if __name__ == '__main__':
    main()
