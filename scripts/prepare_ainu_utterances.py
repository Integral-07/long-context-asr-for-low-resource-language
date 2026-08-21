#!/usr/bin/env python3
"""
Ainuコーパスを発話単位(transcriptsの各行=1学習サンプル)で lcasr 学習用の
mapping.json に変換する。prepare_ainu.py(コレクションを連結する長文脈版)と対になる、
非連結版。

Karolさんのベースライン実験のうち、
  1) wav2vec2-Ainu を従来手法(短い音声)でファインチューニング
  2) 論文のアーキテクチャを0から、従来手法(短い音声)で学習
の実験2で使う想定。train/dev/testの分割はコレクション単位で固定し、
prepare_ainu.py が作る連結版(実験3用)と同じ held-out コレクションを使うことで、
3実験の評価セットを揃えている。

Usage:
  uv run --extra gpu scripts/prepare_ainu_utterances.py \\
    --corpus-dir ainu_corpus \\
    --spec-dir   ainu_utterance_processed \\
    --output-dir data/ainu_utterance
"""

import argparse
import json
from pathlib import Path

import torch

from prepare_ainu import parse_transcript, load_clip, SR, HOP_LENGTH
from ainu_splits import split_for
from lcasr.utils.audio_tools import to_spectogram


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus-dir', default='ainu_corpus')
    parser.add_argument('--spec-dir', required=True,
                        help='出力先 (.spec.pt / 単語タイムスタンプJSON)')
    parser.add_argument('--output-dir', required=True,
                        help='train/dev/test それぞれの mapping.json を書き出すディレクトリ')
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    transcripts_dir = corpus_dir / 'transcripts'
    audio_dir = corpus_dir / 'audio'
    spec_dir = Path(args.spec_dir)
    txt_dir = spec_dir / 'txt'
    spec_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    trans_files = sorted(transcripts_dir.glob('at*.trans.txt'))
    mappings = {'train': {}, 'dev': {}, 'test': {}}
    missing_audio = 0
    empty_text = 0

    for trans_path in trans_files:
        collection_id = trans_path.stem.replace('.trans', '')
        split = split_for(collection_id)
        entries = parse_transcript(trans_path)

        for seg_id, _, text in entries:
            wav_path = audio_dir / f'{seg_id}.wav'
            if not wav_path.exists():
                missing_audio += 1
                continue

            tokens = text.split()
            if not tokens:
                empty_text += 1
                continue

            waveform = load_clip(wav_path)
            dur = waveform.shape[-1] / SR

            spec = to_spectogram(waveform).to(torch.float16)
            spec_path = spec_dir / f'{seg_id}.spec.pt'
            torch.save(spec, str(spec_path))

            step = dur / len(tokens)
            words = [{
                'start': round(i * step, 4),
                'end': round((i + 1) * step, 4),
                'text': tok,
            } for i, tok in enumerate(tokens)]

            txt_path = txt_dir / f'{seg_id}.json'
            with open(txt_path, 'w', encoding='utf-8') as f:
                json.dump(words, f, ensure_ascii=False)

            duration = round(spec.shape[-1] * HOP_LENGTH / SR, 2)
            mappings[split][seg_id] = {
                'audio': str(spec_path.resolve()),
                'txt': str(txt_path.resolve()),
                'duration': duration,
            }

    if missing_audio:
        print(f'WARN: {missing_audio} transcript rows had no matching audio file')
    if empty_text:
        print(f'WARN: {empty_text} transcript rows had no words after splitting, skipped')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, mapping in mappings.items():
        total_minutes = sum(v['duration'] for v in mapping.values()) / 60
        out_path = output_dir / f'{split}_mapping.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f'{split}: {len(mapping)} utterances ({total_minutes:.1f} min) -> {out_path}')


if __name__ == '__main__':
    main()
