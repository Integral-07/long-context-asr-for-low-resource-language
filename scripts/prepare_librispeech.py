#!/usr/bin/env python3
"""
LibriSpeech data preparation for lcasr training.

Concatenates utterances within each chapter to create long-form recordings,
then approximates word-level timing by distributing words uniformly within
each utterance's time window.

Usage:
  uv run --extra gpu scripts/prepare_librispeech.py \
    --librispeech-dir /home/t23cs033/LibriSpeech \
    --spec-dir        /home/t23cs033/librispeech_processed \
    --output          /home/t23cs033/long-context-asr/data/mapping.json \
    --split           train-clean-100
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm


def load_audio(path: Path) -> torch.Tensor:
    """Load audio as mono 16kHz waveform."""
    import torchaudio
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    return waveform


def process_chapter(chapter_dir: Path, spec_dir: Path, txt_dir: Path, skip_audio: bool):
    parts = chapter_dir.name, chapter_dir.parent.name  # book_id, speaker_id
    book_id, speaker_id = parts
    rec_id = f'{speaker_id}-{book_id}'

    trans_file = chapter_dir / f'{speaker_id}-{book_id}.trans.txt'
    if not trans_file.exists():
        return None

    utterances = {}
    with open(trans_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, _, text = line.partition(' ')
            utterances[utt_id] = text.lower()

    sorted_utt_ids = sorted(utterances.keys())

    spec_path = spec_dir / f'{rec_id}.spec.pt'
    txt_path = txt_dir / f'{rec_id}.json'

    audio_chunks = []
    words_json = []
    cumulative = 0.0

    for utt_id in sorted_utt_ids:
        flac_path = chapter_dir / f'{utt_id}.flac'
        if not flac_path.exists():
            continue
        try:
            waveform = load_audio(flac_path)
        except Exception as e:
            print(f'  WARN: failed to load {flac_path.name}: {e}')
            continue

        utt_dur = waveform.shape[-1] / 16000.0
        words = utterances[utt_id].split()
        if not words:
            audio_chunks.append(waveform)
            cumulative += utt_dur
            continue

        for i, word in enumerate(words):
            word_start = cumulative + i * utt_dur / len(words)
            word_end = cumulative + (i + 1) * utt_dur / len(words)
            words_json.append({
                'start': round(word_start, 4),
                'end': round(word_end, 4),
                'text': word,
            })

        audio_chunks.append(waveform)
        cumulative += utt_dur

    if not audio_chunks or not words_json:
        return None

    if not skip_audio and not spec_path.exists():
        from lcasr.utils.audio_tools import to_spectogram
        full_audio = torch.cat(audio_chunks, dim=-1)
        spec = to_spectogram(full_audio).to(torch.float16)
        torch.save(spec, str(spec_path))

    if not spec_path.exists():
        return None

    if not txt_path.exists():
        with open(txt_path, 'w', encoding='utf-8') as f:
            json.dump(words_json, f, ensure_ascii=False)

    spec = torch.load(spec_path, map_location='cpu', weights_only=True)
    duration = round(spec.shape[-1] * 160 / 16000, 2)

    if duration < 5.0:
        return None

    return rec_id, str(spec_path.resolve()), str(txt_path.resolve()), duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--librispeech-dir', required=True,
                        help='Root that contains LibriSpeech/ (e.g. /home/user/LibriSpeech)')
    parser.add_argument('--spec-dir', required=True)
    parser.add_argument('--output', required=True, help='Output mapping.json path')
    parser.add_argument('--split', default='train-clean-100',
                        choices=['train-clean-100', 'train-clean-360', 'train-other-500',
                                 'dev-clean', 'test-clean'])
    parser.add_argument('--skip-audio', action='store_true')
    args = parser.parse_args()

    librispeech_root = Path(args.librispeech_dir) / 'LibriSpeech' / args.split
    if not librispeech_root.exists():
        # Try without the extra LibriSpeech/ subdirectory
        librispeech_root = Path(args.librispeech_dir) / args.split
    if not librispeech_root.exists():
        print(f'ERROR: split directory not found. Tried:\n'
              f'  {Path(args.librispeech_dir) / "LibriSpeech" / args.split}\n'
              f'  {Path(args.librispeech_dir) / args.split}', file=sys.stderr)
        sys.exit(1)

    spec_dir = Path(args.spec_dir)
    txt_dir = spec_dir / 'txt'
    spec_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chapter_dirs = sorted(
        d for speaker in librispeech_root.iterdir() if speaker.is_dir()
        for d in speaker.iterdir() if d.is_dir()
    )
    print(f'Found {len(chapter_dirs)} chapters in {args.split}')

    mapping = {}
    skipped = 0

    for chapter_dir in tqdm(chapter_dirs, desc='chapters'):
        result = process_chapter(chapter_dir, spec_dir, txt_dir, args.skip_audio)
        if result is None:
            skipped += 1
            continue
        rec_id, spec_path, txt_path, duration = result
        mapping[rec_id] = {'audio': spec_path, 'txt': txt_path, 'duration': duration}

    total_hours = sum(v['duration'] for v in mapping.values()) / 3600
    print(f'\nResult: {len(mapping)} chapters ({total_hours:.1f} h), {skipped} skipped')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
