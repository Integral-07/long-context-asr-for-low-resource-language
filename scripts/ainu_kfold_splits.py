"""
Ainuコーパスのコレクション単位k分割交差検証(k-fold CV)の唯一の定義。

ainu_splits.py(実験1/2/3で使った固定の train 43 / dev 5 / test 5 split)の
代わりに、53コレクション全部をk個のfoldに分け、各foldを順番にtestとして
使うことで、5コレクションしかないtestセットに起因する統計的検定力不足
(held-outが5個だと符号検定は理論上どうやってもp=0.0625までしか届かない)
を解消する。

分割はコレクション単位で固定する。②(発話単位, 非連結)と③(コレクション
連結, 長文脈)は同じ録音を粒度違いで扱っているだけなので、あるコレクション
の発話群(②)とそのコレクション自体(③)が違うfoldに分かれてしまうと、
「同じ音声が一方のtrainで他方のtestに出てくる」というリークになる。
そのため fold_for() はコレクションIDだけをキーにしており、②③(および①)
の全てのdata-prepスクリプトがこれを共有する。

fold割り当ては、各fold の合計収録時間ができるだけ均等になるように
LPT(Longest Processing Time)ヒューリスティックで決めた:
  1. 53コレクションを実測の長さ(秒)で降順にソート
  2. 各コレクションを、その時点で合計時間が最小のfoldに割り当てる
結果、5foldとも1.75〜1.80時間(10〜11コレクション)に収まっている
(再現用のロジックは末尾の `_recompute_fold_assignment()` を参照。
コーパス自体が固定の歴史的資料で増減しないため、実行時に毎回計算し直す
のではなくここに結果を焼き込んでいる)。

at33 は書き起こしファイルが空(音声なし)のため、そもそも対象外。
"""

from pathlib import Path
from typing import Dict, FrozenSet

N_FOLDS = 5
DEV_FRACTION = 0.10

# corpus収録時間: at01..at54(at33を除く53コレクション)の実測秒数から
# LPTヒューリスティックで算出した固定のfold割り当て。詳細は末尾を参照。
FOLD_ASSIGNMENT = {
    'at01': 0, 'at02': 3, 'at03': 2, 'at04': 3, 'at05': 2,
    'at06': 0, 'at07': 4, 'at08': 1, 'at09': 4, 'at10': 1,
    'at11': 0, 'at12': 3, 'at13': 1, 'at14': 2, 'at15': 0,
    'at16': 1, 'at17': 3, 'at18': 4, 'at19': 4, 'at20': 2,
    'at21': 4, 'at22': 4, 'at23': 0, 'at24': 2, 'at25': 4,
    'at26': 3, 'at27': 4, 'at28': 4, 'at29': 0, 'at30': 1,
    'at31': 1, 'at32': 1, 'at34': 3, 'at35': 2, 'at36': 1,
    'at37': 1, 'at38': 2, 'at39': 3, 'at40': 3, 'at41': 4,
    'at42': 2, 'at43': 2, 'at44': 4, 'at45': 2, 'at46': 3,
    'at47': 1, 'at48': 0, 'at49': 1, 'at50': 3, 'at51': 0,
    'at52': 0, 'at53': 2, 'at54': 0,
    # at33: 書き起こし0行(音声なし)のため割り当てなし
}

ALL_COLLECTIONS = frozenset(FOLD_ASSIGNMENT)


# 各outer foldのtrain pool(他4fold)から、持続時間の約10%(DEV_FRACTION)を
# LPT風に切り出したdevコレクション(fold_idx -> devコレクションIDの集合)。
# outer foldのtest集合(FOLD_ASSIGNMENT)は変更しない — 変わるのは各foldの
# train集合の内部だけ(train -> train' + dev)。再現用ロジックは末尾の
# `_recompute_dev_assignment()` を参照。
DEV_ASSIGNMENT: Dict[int, FrozenSet[str]] = {
    0: frozenset({'at08', 'at43'}),
    1: frozenset({'at43', 'at54'}),
    2: frozenset({'at08', 'at54'}),
    3: frozenset({'at08', 'at54'}),
    4: frozenset({'at08', 'at54'}),
}


def fold_for(collection_id: str) -> int:
    """collection_id (例: 'at08') が属するfold番号(0..N_FOLDS-1)を返す。
    at33等、割り当てのないコレクションは呼び出し側でスキップすること。"""
    return FOLD_ASSIGNMENT[collection_id]


def dev_for(collection_id: str, fold_idx: int) -> bool:
    """collection_idが、fold_idxをtestとして使うときのdev集合に属するか。"""
    return collection_id in DEV_ASSIGNMENT[fold_idx]


def train_test_for_fold(fold_idx: int):
    """fold_idx をtestとして使うときの (train_ids, test_ids) を返す。
    train_idsにはdevも含む(従来の2分割インターフェースとの後方互換用)。
    3-way分割が必要な呼び出し元は train_dev_test_for_fold() を使うこと。"""
    if not (0 <= fold_idx < N_FOLDS):
        raise ValueError(f'fold_idx must be in [0, {N_FOLDS}), got {fold_idx}')
    test_ids = {cid for cid, f in FOLD_ASSIGNMENT.items() if f == fold_idx}
    train_ids = ALL_COLLECTIONS - test_ids
    return train_ids, test_ids


def train_dev_test_for_fold(fold_idx: int):
    """fold_idx をtestとして使うときの (train_ids, dev_ids, test_ids) を返す。
    train_idsはdevを除いた純粋なtrain'集合。"""
    train_ids, test_ids = train_test_for_fold(fold_idx)
    dev_ids = set(DEV_ASSIGNMENT[fold_idx])
    return train_ids - dev_ids, dev_ids, test_ids


def split_for(collection_id: str, fold_idx: int) -> str:
    """prepare_ainu.py / prepare_ainu_utterances.py / train_ainu_tokenizer.py の
    既存の split_for(collection_id) と同じインターフェースの、fold対応版。
    'train' / 'dev' / 'test' の3値を返す(test判定を最優先、次にdev判定)。"""
    if fold_for(collection_id) == fold_idx:
        return 'test'
    if dev_for(collection_id, fold_idx):
        return 'dev'
    return 'train'


def _collection_durations(corpus_dir='ainu_corpus'):
    """各コレクションの実測音声長(秒)を計算する共有ヘルパー。
    _recompute_fold_assignment() と _recompute_dev_assignment() の両方で使う。"""
    import wave
    import contextlib
    from collections import defaultdict

    corpus_dir = Path(corpus_dir)
    durations = defaultdict(float)
    for trans_path in sorted((corpus_dir / 'transcripts').glob('at*.trans.txt')):
        cid = trans_path.stem.replace('.trans', '')
        for line in trans_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            seg_id = line.split(' ', 1)[0]
            wav_path = corpus_dir / 'audio' / f'{seg_id}.wav'
            if not wav_path.exists():
                continue
            try:
                with contextlib.closing(wave.open(str(wav_path), 'r')) as f:
                    durations[cid] += f.getnframes() / float(f.getframerate())
            except Exception:
                pass  # at37/at38 hold a couple of non-RIFF clips; duration slightly underestimated
    return durations


def _recompute_fold_assignment(corpus_dir='ainu_corpus', k=N_FOLDS):
    """FOLD_ASSIGNMENT を実測の音声長から再計算する(コーパスが変わった場合の再現用)。
    通常の実行では呼ばれない。"""
    durations = _collection_durations(corpus_dir)
    ordered = sorted(durations.items(), key=lambda x: -x[1])  # longest first (LPT)
    fold_totals = [0.0] * k
    assignment = {}
    for cid, dur in ordered:
        i = min(range(k), key=lambda f: fold_totals[f])
        fold_totals[i] += dur
        assignment[cid] = i
    return assignment


def _dev_collections_for_fold(fold_idx, durations, target_frac=DEV_FRACTION):
    """outer fold `fold_idx` のtrain pool(他4fold)を時間降順にソートし、
    累積時間が train_total * target_frac に最も近づく時点で切って、その
    先頭k件をdevにする(LPTスタイルを1段ネストさせた形)。"""
    train_ids, _ = train_test_for_fold(fold_idx)
    train_total = sum(durations[cid] for cid in train_ids)
    target = train_total * target_frac

    ordered = sorted(train_ids, key=lambda cid: -durations[cid])
    best_k, best_diff, cum = 0, target, 0.0
    for k, cid in enumerate(ordered, start=1):
        cum += durations[cid]
        diff = abs(cum - target)
        if diff < best_diff:
            best_diff = diff
            best_k = k
    return frozenset(ordered[:best_k])


def _recompute_dev_assignment(corpus_dir='ainu_corpus', target_frac=DEV_FRACTION):
    """DEV_ASSIGNMENT を実測の音声長から再計算する(コーパスが変わった場合の再現用)。
    通常の実行では呼ばれない。FOLD_ASSIGNMENT(test集合)は固定のまま、各foldの
    train poolだけを対象にdevを切り出す。"""
    durations = _collection_durations(corpus_dir)
    return {
        fold_idx: _dev_collections_for_fold(fold_idx, durations, target_frac)
        for fold_idx in range(N_FOLDS)
    }


if __name__ == '__main__':
    # 再計算して現在の FOLD_ASSIGNMENT / DEV_ASSIGNMENT と一致するか確認する簡易チェック。
    recomputed_folds = _recompute_fold_assignment()
    if recomputed_folds == FOLD_ASSIGNMENT:
        print('OK: FOLD_ASSIGNMENT matches recomputed assignment from ainu_corpus/')
    else:
        print('MISMATCH: FOLD_ASSIGNMENT is stale, recompute and update this file:')
        print(recomputed_folds)

    recomputed_dev = _recompute_dev_assignment()
    if recomputed_dev == DEV_ASSIGNMENT:
        print('OK: DEV_ASSIGNMENT matches recomputed assignment from ainu_corpus/')
    else:
        print('MISMATCH: DEV_ASSIGNMENT is stale, recompute and update this file:')
        print(recomputed_dev)
