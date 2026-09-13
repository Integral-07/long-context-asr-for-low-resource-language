#!/usr/bin/env python3
"""
k-fold交差検証の結果(② scratch-short vs ③ scratch-long)を集計し、統計的有意性を検定する。

各foldの checkpoints/ainu_kfold/fold{i}_short/test_predictions.json と
fold{i}_long/test_predictions.json を読み込み、全foldのtestコレクション
(=全53コレクション、fold間で重複なし)をプールして、
  - corpus-level WER(総誤り数/総単語数)の全体比較
  - コレクション単位でのペア化ブートストラップ検定
  - 厳密符号検定(sign test)
を行う。significance_test.py(固定split版, n=5)の一般化版。

前提:
  - 各foldのtest_predictions.jsonは、eval_ainu_lcasr.pyが出力する形式
    ({'wer':..., 'results': [{'id', 'reference', 'prediction'}, ...]})
  - ②(short)のtest_predictions.jsonの'id'は発話単位(例: 'At08-3')、
    ③(long)は既にコレクション単位(例: 'at08')。②はコレクション単位に
    集約してから③と比較する。
  - fold間でtestコレクションが重複しないので、全foldをまたいで単純にpoolしてよい
    (同じコレクションが2回testされることはない)。

Usage:
  python scripts/aggregate_kfold_results.py \\
    --checkpoints-dir checkpoints/ainu_kfold \\
    --n-folds 5
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

try:
    import jiwer
except ImportError:
    sys.exit('jiwer が必要です: pip install jiwer')

COLL_RE = re.compile(r'^([A-Za-z]+\d+)-\d+$')
N_BOOT = 20000


def to_collection(item_id: str) -> str:
    m = COLL_RE.match(item_id)
    return m.group(1).lower() if m else item_id.lower()


def errs_words(ref: str, hyp: str):
    r = ref.split()
    if not r:
        return len(hyp.split()), 0
    m = jiwer.process_words(ref, hyp)
    return m.insertions + m.deletions + m.substitutions, len(r)


def load_predictions(path: Path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def per_collection_counts(results):
    """utterance単位/collection単位どちらのresultsでも、collection_idごとの
    (errors, words) 累計に潰す。"""
    agg = defaultdict(lambda: [0, 0])
    for r in results:
        e, w = errs_words(r['reference'], r['prediction'])
        cid = to_collection(r['id'])
        agg[cid][0] += e
        agg[cid][1] += w
    return {k: tuple(v) for k, v in agg.items()}


def pooled_wer(counts, keys):
    e = sum(counts[k][0] for k in keys)
    w = sum(counts[k][1] for k in keys)
    return e / w if w else float('nan')


def bootstrap_paired(counts_a, counts_b, keys, n_boot=N_BOOT, label=''):
    n = len(keys)
    obs_a, obs_b = pooled_wer(counts_a, keys), pooled_wer(counts_b, keys)
    diffs = []
    for _ in range(n_boot):
        sample = [keys[random.randrange(n)] for _ in range(n)]
        diffs.append(pooled_wer(counts_b, sample) - pooled_wer(counts_a, sample))
    diffs.sort()
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]
    frac_ge0 = sum(1 for d in diffs if d >= 0) / n_boot
    frac_le0 = sum(1 for d in diffs if d <= 0) / n_boot
    p = min(2 * min(frac_ge0, frac_le0), 1.0)

    print(f'--- {label} (n={n} コレクション, {n_boot}回リサンプル) ---')
    print(f'  WER_short(a) = {obs_a:.4f}, WER_long(b) = {obs_b:.4f}, diff(b-a) = {obs_b - obs_a:+.4f}')
    print(f'  95% CI of diff: [{lo:+.4f}, {hi:+.4f}]')
    print(f'  bootstrap p(two-sided) = {p:.4f}')
    print()


def exact_sign_test(counts_a, counts_b, keys, label=''):
    n = len(keys)
    wins_b = sum(1 for k in keys if counts_b[k][0] / counts_b[k][1] < counts_a[k][0] / counts_a[k][1])
    k = min(wins_b, n - wins_b)
    p = min(sum(comb(n, i) for i in range(k + 1)) * 2 / (2 ** n), 1.0)
    print(f'--- {label}: 符号検定 ---')
    print(f'  bが勝った(WER改善)コレクション数: {wins_b}/{n}')
    print(f'  exact two-sided p = {p:.4f}  (この n での理論上の最小p値 = {2 / (2 ** n):.4g})')
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints-dir', default='checkpoints/ainu_kfold')
    parser.add_argument('--n-folds', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    ckpt_dir = Path(args.checkpoints_dir)
    short_counts, long_counts = {}, {}
    missing = []

    for fold in range(args.n_folds):
        short_path = ckpt_dir / f'fold{fold}_short' / 'test_predictions.json'
        long_path = ckpt_dir / f'fold{fold}_long' / 'test_predictions.json'
        if not short_path.exists():
            missing.append(str(short_path))
            continue
        if not long_path.exists():
            missing.append(str(long_path))
            continue

        short_data = load_predictions(short_path)
        long_data = load_predictions(long_path)
        fold_short = per_collection_counts(short_data['results'])
        fold_long = per_collection_counts(long_data['results'])

        overlap = set(fold_short) ^ set(fold_long)
        if set(fold_short) != set(fold_long):
            print(f'WARN: fold{fold}: short/long のtestコレクション集合が一致しません: {overlap}')

        short_counts.update(fold_short)
        long_counts.update(fold_long)

    if missing:
        print('見つからなかった test_predictions.json:')
        for m in missing:
            print(f'  {m}')
        if not short_counts:
            sys.exit('集計対象が0件です。学習/評価がまだ完了していません。')
        print(f'(見つかった分だけで集計を続けます: {len(short_counts)}コレクション)\n')

    common = sorted(set(short_counts) & set(long_counts))
    print(f'=== 集計対象: {len(common)} コレクション (全{args.n_folds}fold中) ===\n')

    bootstrap_paired(short_counts, long_counts, common,
                      label='② scratch-short vs ③ scratch-long (k-fold全体)')
    exact_sign_test(short_counts, long_counts, common,
                    label='② vs ③ (k-fold全体)')


if __name__ == '__main__':
    main()
