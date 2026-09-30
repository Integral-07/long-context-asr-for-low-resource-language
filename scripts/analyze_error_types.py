#!/usr/bin/env python3
"""
k-foldのtest_predictions.jsonを対象に、挿入/削除/置換の内訳と、
反復・空予測など部分崩壊の兆候を確認する(誤り種別分析)。

Usage:
  uv run --extra gpu python scripts/analyze_error_types.py \\
    --checkpoints-dir checkpoints/ainu_wav2vec2_kfold \\
    --condition short --n-folds 5
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import jiwer
except ImportError:
    sys.exit('jiwer が必要です: pip install jiwer')

COLL_RE = re.compile(r'^([A-Za-z]+\d+)-\d+$')


def to_collection(item_id: str) -> str:
    m = COLL_RE.match(item_id)
    return m.group(1).lower() if m else item_id.lower()


def is_repetitive(text: str, min_repeats: int = 4) -> bool:
    """同一トークンがmin_repeats回以上連続していれば反復崩壊とみなす。"""
    words = text.split()
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            if run >= min_repeats:
                return True
        else:
            run = 1
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints-dir', required=True)
    parser.add_argument('--condition', required=True, choices=['short', 'long'])
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--top-n', type=int, default=15)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoints_dir)
    total_ins = total_del = total_sub = total_words = 0
    coll_agg = defaultdict(lambda: {'ins': 0, 'del': 0, 'sub': 0, 'words': 0,
                                     'empty_pred': 0, 'repetitive': 0, 'n': 0})

    for fold in range(args.n_folds):
        path = ckpt_dir / f'fold{fold}_{args.condition}' / 'test_predictions.json'
        if not path.exists():
            print(f'WARN: {path} が見つかりません、スキップ')
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        for r in data['results']:
            ref, hyp = r['reference'], r['prediction']
            cid = to_collection(r['id'])
            r_words = ref.split()
            if not r_words:
                continue
            m = jiwer.process_words(ref, hyp)
            total_ins += m.insertions
            total_del += m.deletions
            total_sub += m.substitutions
            total_words += len(r_words)

            agg = coll_agg[cid]
            agg['ins'] += m.insertions
            agg['del'] += m.deletions
            agg['sub'] += m.substitutions
            agg['words'] += len(r_words)
            agg['n'] += 1
            if not hyp.strip():
                agg['empty_pred'] += 1
            if is_repetitive(hyp):
                agg['repetitive'] += 1

    print(f'=== 条件{args.condition}: 全体(プールしたcorpus-level)誤り内訳 ===')
    print(f'  words={total_words}')
    print(f'  insertions={total_ins} ({total_ins/total_words*100:.2f}%)')
    print(f'  deletions={total_del} ({total_del/total_words*100:.2f}%)')
    print(f'  substitutions={total_sub} ({total_sub/total_words*100:.2f}%)')
    print(f'  WER={(total_ins+total_del+total_sub)/total_words*100:.2f}%\n')

    print(f'=== コレクション別、WER降順(上位{args.top_n}件) ===')
    print(f'{"id":8s} {"n":>4s} {"WER":>8s} {"ins%":>7s} {"del%":>7s} {"sub%":>7s} {"empty":>6s} {"repet":>6s}')
    rows = []
    for cid, a in coll_agg.items():
        wer = (a['ins'] + a['del'] + a['sub']) / a['words'] * 100 if a['words'] else float('nan')
        rows.append((cid, a, wer))
    for cid, a, wer in sorted(rows, key=lambda x: -x[2])[:args.top_n]:
        print(f'{cid:8s} {a["n"]:>4d} {wer:>7.2f}% '
              f'{a["ins"]/a["words"]*100:>6.2f}% {a["del"]/a["words"]*100:>6.2f}% '
              f'{a["sub"]/a["words"]*100:>6.2f}% {a["empty_pred"]:>6d} {a["repetitive"]:>6d}')

    flagged = [(cid, a) for cid, a in coll_agg.items() if a['empty_pred'] > 0 or a['repetitive'] > 0]
    if flagged:
        print(f'\n=== 空予測 or 反復崩壊が検出されたコレクション ===')
        for cid, a in sorted(flagged, key=lambda x: x[0]):
            print(f'  {cid}: n={a["n"]}, empty_pred={a["empty_pred"]}, repetitive={a["repetitive"]}')
    else:
        print('\n空予測・反復崩壊は検出されませんでした。')


if __name__ == '__main__':
    main()
