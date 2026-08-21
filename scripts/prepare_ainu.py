#!/usr/bin/env python3
"""
Ainu語コーパス(ainu_corpus/, Tuytah Collection)を lcasr 学習用の
mapping.json 形式に変換する。

各 .trans.txt (収録単位, 例: at01) に属する発話単位の wav クリップ
(例: At01-1.wav, At01-2.wav, ...) を ID の連番順に連結し、1収録=1つの
連続音声として再構成する。クリップの境界時刻は実測(正確)だが、
クリップ内部の単語ごとのタイムスタンプはクリップの尺の中で線形補間した
近似値である点に注意。

train/dev/testの分割はコレクション単位で固定し、prepare_ainu_utterances.py
(実験1/2用の非連結版)と同じ held-out コレクションを使うことで、3実験の
評価セットを揃えている。

Usage:
  uv run --extra cpu scripts/prepare_ainu.py \\
    --corpus-dir ainu_corpus \\
    --spec-dir   ainu_processed \\
    --output-dir data/ainu
"""

import argparse
import json
import re
from pathlib import Path

import torch
import torchaudio

from lcasr.utils.audio_tools import to_spectogram

HOP_LENGTH = 160
SR = 16000

ID_RE = re.compile(r'^([A-Za-z]+\d+)-(\d+)$')

# prepare_ainu_utterances.py(実験1/2, 非連結版)と揃えた held-out コレクション。
TEST_COLLECTIONS = {'at08', 'at30', 'at40', 'at13', 'at22'}
DEV_COLLECTIONS = {'at54', 'at18', 'at27', 'at36', 'at10'}


def split_for(collection_id: str) -> str:
    if collection_id in TEST_COLLECTIONS:
        return 'test'
    if collection_id in DEV_COLLECTIONS:
        return 'dev'
    return 'train'


def parse_transcript(path: Path):
    """Return [(seg_id, suffix_num, text), ...] sorted by suffix_num."""
    entries = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        seg_id, _, text = line.partition(' ')
        m = ID_RE.match(seg_id)
        if not m:
            print(f'  WARN: unrecognized id {seg_id!r} in {path.name}, skipping line')
            continue
        entries.append((seg_id, int(m.group(2)), text.strip()))
    entries.sort(key=lambda e: e[1])
    return entries


def load_clip(path: Path) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != SR:
        waveform = torchaudio.transforms.Resample(sr, SR)(waveform)
    return waveform


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus-dir', default='ainu_corpus',
                        help='transcripts/ と audio/ を含むディレクトリ')
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
    print(f'{len(trans_files)} collections found')

    mappings = {'train': {}, 'dev': {}, 'test': {}}
    total_missing_audio = 0

    for trans_path in trans_files:
        collection_id = trans_path.stem.replace('.trans', '')
        split = split_for(collection_id)
        entries = parse_transcript(trans_path)

        clips, words, cursor = [], [], 0.0
        for seg_id, _, text in entries:
            wav_path = audio_dir / f'{seg_id}.wav'
            if not wav_path.exists():
                total_missing_audio += 1
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
            print(f'  WARN: no audio found for {collection_id}, skipping collection')
            continue

        full_waveform = torch.cat(clips, dim=-1)
        spec = to_spectogram(full_waveform).to(torch.float16)

        spec_path = spec_dir / f'{collection_id}.spec.pt'
        torch.save(spec, str(spec_path))

        txt_path = txt_dir / f'{collection_id}.json'
        with open(txt_path, 'w', encoding='utf-8') as f:
            json.dump(words, f, ensure_ascii=False)

        duration = round(spec.shape[-1] * HOP_LENGTH / SR, 2)
        mappings[split][collection_id] = {
            'audio': str(spec_path.resolve()),
            'txt': str(txt_path.resolve()),
            'duration': duration,
        }
        print(f'  {collection_id} [{split}]: {len(clips)} clips, {len(words)} words ({duration/60:.1f} min)')

    if total_missing_audio:
        print(f'\nWARN: {total_missing_audio} transcript rows had no matching audio file')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, mapping in mappings.items():
        total_hours = sum(v['duration'] for v in mapping.values()) / 3600
        out_path = output_dir / f'{split}_mapping.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f'{split}: {len(mapping)} collections ({total_hours:.2f} h) -> {out_path}')


if __name__ == '__main__':
    main()
