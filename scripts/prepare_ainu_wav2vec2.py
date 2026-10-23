#!/usr/bin/env python3
"""
Ainuコーパスを wav2vec2 CTC ファインチューニング(実験1)用のデータに変換する。

lcasrのメルスペクトログラム前処理は使わず、生波形へのパスとテキストだけを
持つJSONマニフェスト(train/dev/test)と、文字レベルのvocab.json
(HuggingFace Wav2Vec2CTCTokenizer形式)を書き出す。
train/dev/testの分割は prepare_ainu_utterances.py と同じ held-out
コレクションを使い、実験1/2/3で評価セットを揃えている。

Usage:
  uv run --extra gpu python scripts/prepare_ainu_wav2vec2.py \\
    --corpus-dir ainu_corpus \\
    --output-dir data/ainu_wav2vec2
"""

import argparse
import json
from pathlib import Path

from prepare_ainu import parse_transcript
from ainu_splits import split_for as fixed_split_for
from ainu_kfold_splits import split_for as kfold_split_for

SPECIAL_TOKENS = ['|', '[UNK]', '[PAD]']


def build_vocab(train_texts):
    chars = set()
    for text in train_texts:
        for word in text.split():
            chars.update(word)
    vocab = {ch: i for i, ch in enumerate(sorted(chars))}
    for tok in SPECIAL_TOKENS:
        vocab[tok] = len(vocab)
    return vocab


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus-dir', default='ainu_corpus')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fold', type=int, default=None,
                        help='指定するとk-fold交差検証モード(ainu_kfold_splits.py)。'
                             '0..N_FOLDS-1のfold番号をtestとして使う。省略時は固定split'
                             '(ainu_splits.py、実験①②③と同じheld-out)。'
                             'k-foldモードではvocab.jsonは書き出さない'
                             '(scripts/train_ainu_hf_tokenizer.pyでfold別に別途構築すること、'
                             'vocab leakage防止のためtokenizer学習とデータ準備を分離している)。')
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    transcripts_dir = corpus_dir / 'transcripts'
    audio_dir = corpus_dir / 'audio'

    records = {'train': [], 'dev': [], 'test': []}
    missing_audio = 0
    empty_text = 0

    for trans_path in sorted(transcripts_dir.glob('at*.trans.txt')):
        collection_id = trans_path.stem.replace('.trans', '')
        if args.fold is not None:
            try:
                split = kfold_split_for(collection_id, args.fold)
            except KeyError:
                continue  # at33: 書き起こしが空(音声なし)、fold割り当てもない
        else:
            split = fixed_split_for(collection_id)

        for seg_id, _, text in parse_transcript(trans_path):
            wav_path = audio_dir / f'{seg_id}.wav'
            if not wav_path.exists():
                missing_audio += 1
                continue
            if not text.split():
                empty_text += 1
                continue
            records[split].append({
                'id': seg_id,
                'audio': str(wav_path.resolve()),
                'text': text,
            })

    if missing_audio:
        print(f'WARN: {missing_audio} transcript rows had no matching audio file')
    if empty_text:
        print(f'WARN: {empty_text} transcript rows had no words, skipped')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fold is None:
        vocab = build_vocab(r['text'] for r in records['train'])
        vocab_path = output_dir / 'vocab.json'
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f'vocab: {len(vocab)} tokens -> {vocab_path}')

    for split, recs in records.items():
        out_path = output_dir / f'{split}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        n_words = sum(len(r['text'].split()) for r in recs)
        print(f'{split}: {len(recs)} utterances, {n_words} words -> {out_path}')


if __name__ == '__main__':
    main()
