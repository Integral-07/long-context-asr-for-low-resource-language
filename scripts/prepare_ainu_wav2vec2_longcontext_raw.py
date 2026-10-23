#!/usr/bin/env python3
"""
実験⑬の長文脈版(提案手法)用に、Ainuコーパスをコレクション単位で連結した
生波形(.pt)+単語タイムスタンプJSONを書き出す。

scripts/prepare_ainu.py(lcasr用、連結後にmel-spectrogramへ変換)と連結ロジック
は同じだが、HFのWav2Vec2FeatureExtractorは生波形を直接消費するため、
mel-spectrogramへの変換はせず生波形のまま保存する点が違う。

k-fold(ainu_kfold_splits.py)のみ対応。固定splitは実験⑬の単一split実行
(utterance単位、prepare_ainu_wav2vec2.py)で使用済みのため、長文脈側は
最初からk-fold前提とする。

Usage:
  uv run --extra cpu python scripts/prepare_ainu_wav2vec2_longcontext_raw.py \\
    --corpus-dir ainu_corpus \\
    --audio-dir  ainu_watanabe_longcontext_audio/fold0 \\
    --output-dir data/ainu_watanabe_longcontext_kfold/fold0 \\
    --fold 0
"""

import argparse
import json
from pathlib import Path

import torch

from prepare_ainu import SR, load_clip, parse_transcript
from ainu_kfold_splits import split_for


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus-dir', default='ainu_corpus')
    parser.add_argument('--audio-dir', required=True,
                        help='連結済み生波形(<collection_id>.pt)の出力先')
    parser.add_argument('--output-dir', required=True,
                        help='train/dev/test それぞれのcollections.jsonを書き出すディレクトリ')
    parser.add_argument('--fold', type=int, required=True,
                        help='0..ainu_kfold_splits.N_FOLDS-1')
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    transcripts_dir = corpus_dir / 'transcripts'
    audio_dir_in = corpus_dir / 'audio'
    audio_dir_out = Path(args.audio_dir)
    audio_dir_out.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = {'train': [], 'dev': [], 'test': []}
    skipped = []

    for trans_path in sorted(transcripts_dir.glob('at*.trans.txt')):
        collection_id = trans_path.stem.replace('.trans', '')
        try:
            split = split_for(collection_id, args.fold)
        except KeyError:
            continue  # at33: 書き起こしが空(音声なし)、fold割り当てもない

        entries = parse_transcript(trans_path)
        clips, words, cursor = [], [], 0.0
        for seg_id, _, text in entries:
            wav_path = audio_dir_in / f'{seg_id}.wav'
            if not wav_path.exists():
                continue
            waveform = load_clip(wav_path)
            dur = waveform.shape[-1] / SR
            tokens = text.split()
            if tokens:
                step = dur / len(tokens)
                for i, tok in enumerate(tokens):
                    words.append({
                        'start': round(cursor + i * step, 4),
                        'end': round(cursor + (i + 1) * step, 4),
                        'text': tok,
                    })
            clips.append(waveform)
            cursor += dur

        if not clips:
            skipped.append(collection_id)
            continue

        full_waveform = torch.cat(clips, dim=-1)
        audio_path = audio_dir_out / f'{collection_id}.pt'
        torch.save(full_waveform, str(audio_path))

        txt_path = audio_dir_out / f'{collection_id}.json'
        with open(txt_path, 'w', encoding='utf-8') as f:
            json.dump(words, f, ensure_ascii=False)

        records[split].append({
            'id': collection_id,
            'audio': str(audio_path.resolve()),
            'txt': str(txt_path.resolve()),
            'duration': round(cursor, 4),
        })

    if skipped:
        print(f'WARN: no audio found for {len(skipped)} collection(s): {skipped}')

    for split, recs in records.items():
        out_path = output_dir / f'{split}_collections.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        total_hours = sum(r['duration'] for r in recs) / 3600
        print(f'{split}: {len(recs)} collections, {total_hours:.2f}h -> {out_path}')


if __name__ == '__main__':
    main()
