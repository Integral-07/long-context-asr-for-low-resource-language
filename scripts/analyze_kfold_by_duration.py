#!/usr/bin/env python3
"""
実験⑤/⑥のk-fold結果を、コレクションの長さ(duration)との関係で分析する。
scripts/aggregate_kfold_results.py が出す全体集計(pooled WER, 符号検定)
だけでは「どのコレクションで長文脈化が効く/効かないか」が分からないため、
コレクション単位のWER差(B-A)と収録時間の相関を見る。

前提: scripts/run_ainu_wav2vec2_kfold.sh の実行で
  checkpoints/{prefix}_kfold/fold{N}_{short,long}/test_predictions.json
  data/{prefix}_kfold_collections/fold{N}/test_mapping.json (durationの取得用)
が存在すること。

Usage:
  python scripts/analyze_kfold_by_duration.py --prefix ainu_wav2vec2 --n-folds 5
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aggregate_kfold_results import errs_words, to_collection

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = spearmanr = None


def per_collection_wer(results):
    agg = {}
    for r in results:
        e, w = errs_words(r['reference'], r['prediction'])
        cid = to_collection(r['id'])
        cur = agg.setdefault(cid, [0, 0])
        cur[0] += e
        cur[1] += w
    return {k: (v[0] / v[1] if v[1] else float('nan')) for k, v in agg.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints-dir', default=None,
                        help='省略時は checkpoints/{prefix}_kfold')
    parser.add_argument('--data-dir', default=None,
                        help='省略時は data/{prefix}_kfold_collections (durationの取得用)')
    parser.add_argument('--prefix', default='ainu_wav2vec2')
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoints_dir or f'checkpoints/{args.prefix}_kfold')
    data_dir = Path(args.data_dir or f'data/{args.prefix}_kfold_collections')

    wer_a, wer_b, duration = {}, {}, {}

    for fold in range(args.n_folds):
        short_path = ckpt_dir / f'fold{fold}_short' / 'test_predictions.json'
        long_path = ckpt_dir / f'fold{fold}_long' / 'test_predictions.json'
        mapping_path = data_dir / f'fold{fold}' / 'test_mapping.json'
        if not (short_path.exists() and long_path.exists() and mapping_path.exists()):
            print(f'WARN: fold{fold} 一部ファイル欠落、スキップ')
            continue

        short_data = json.loads(short_path.read_text(encoding='utf-8'))
        long_data = json.loads(long_path.read_text(encoding='utf-8'))
        mapping = json.loads(mapping_path.read_text(encoding='utf-8'))

        wer_a.update(per_collection_wer(short_data['results']))
        wer_b.update(per_collection_wer(long_data['results']))
        for cid, entry in mapping.items():
            duration[cid] = entry['duration']

    common = sorted(set(wer_a) & set(wer_b) & set(duration))
    print(f'=== {len(common)} コレクション ===\n')

    rows = []
    for cid in common:
        diff = wer_b[cid] - wer_a[cid]  # 負 = Bが改善
        rows.append((cid, duration[cid], wer_a[cid], wer_b[cid], diff))

    durations = [r[1] for r in rows]
    diffs = [r[4] for r in rows]

    if pearsonr is not None:
        r_p, p_p = pearsonr(durations, diffs)
        r_s, p_s = spearmanr(durations, diffs)
        print(f'収録時間(秒) vs WER差(B-A) の相関:')
        print(f'  Pearson  r={r_p:+.3f} (p={p_p:.4f})')
        print(f'  Spearman r={r_s:+.3f} (p={p_s:.4f})')
        print(f'  (負の相関 = 長いコレクションほどBが改善する傾向)\n')
    else:
        print('scipy未インストールのため相関係数は計算せず、ソート結果のみ表示\n')

    rows_by_diff = sorted(rows, key=lambda r: r[4])
    print('--- Bが最も改善したコレクション(上位10件) ---')
    print(f'{"id":8s} {"dur(min)":>9s} {"WER_A":>8s} {"WER_B":>8s} {"diff":>8s}')
    for cid, dur, a, b, diff in rows_by_diff[:10]:
        print(f'{cid:8s} {dur/60:9.1f} {a*100:7.2f}% {b*100:7.2f}% {diff*100:+7.2f}pt')

    print('\n--- Bが最も悪化したコレクション(上位10件) ---')
    print(f'{"id":8s} {"dur(min)":>9s} {"WER_A":>8s} {"WER_B":>8s} {"diff":>8s}')
    for cid, dur, a, b, diff in rows_by_diff[-10:][::-1]:
        print(f'{cid:8s} {dur/60:9.1f} {a*100:7.2f}% {b*100:7.2f}% {diff*100:+7.2f}pt')

    median_dur = sorted(durations)[len(durations) // 2]
    long_colls = [r for r in rows if r[1] >= median_dur]
    short_colls = [r for r in rows if r[1] < median_dur]
    print(f'\n--- 収録時間で中央値({median_dur/60:.1f}分)分割 ---')
    for label, group in [('長い方の半分', long_colls), ('短い方の半分', short_colls)]:
        n_improve = sum(1 for r in group if r[4] < 0)
        mean_diff = sum(r[4] for r in group) / len(group)
        print(f'{label} (n={len(group)}): Bが改善={n_improve}/{len(group)}, 平均diff={mean_diff*100:+.2f}pt')


if __name__ == '__main__':
    main()
