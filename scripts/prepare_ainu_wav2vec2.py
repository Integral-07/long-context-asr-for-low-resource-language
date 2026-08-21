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
from ainu_splits import split_for

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
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    transcripts_dir = corpus_dir / 'transcripts'
    audio_dir = corpus_dir / 'audio'

    records = {'train': [], 'dev': [], 'test': []}
    missing_audio = 0
    empty_text = 0

    for trans_path in sorted(transcripts_dir.glob('at*.trans.txt')):
        collection_id = trans_path.stem.replace('.trans', '')
        split = split_for(collection_id)

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

    vocab = build_vocab(r['text'] for r in records['train'])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
