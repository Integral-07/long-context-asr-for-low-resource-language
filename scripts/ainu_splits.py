"""
Ainuコーパスのコレクション単位train/dev/test分割の唯一の定義。
実験1(wav2vec2)/2(短い音声)/3(長い音声)すべてで同じ held-out
コレクションを使うことで、評価セットを揃える。

長さで層化(短いものと15〜19分クラスの長いものを両方含める)し、
dev/testそれぞれ合計40〜50分程度になるように選んでいる。
"""

TEST_COLLECTIONS = {'at08', 'at30', 'at40', 'at13', 'at22'}
DEV_COLLECTIONS = {'at54', 'at18', 'at27', 'at36', 'at10'}


def split_for(collection_id: str) -> str:
    if collection_id in TEST_COLLECTIONS:
        return 'test'
    if collection_id in DEV_COLLECTIONS:
        return 'dev'
    return 'train'
