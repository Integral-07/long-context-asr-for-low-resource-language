#!/usr/bin/env python3
"""
TED-LIUM 3 data preparation for lcasr training.

Steps performed:
  1. Validate required tools (ffmpeg) and paths
  2. Load Lhotse supervision manifests from speech-datasets repo
  3. Convert .sph audio -> mel spectrogram (.spec.pt) via ffmpeg + lcasr processing_chain
  4. Convert Lhotse word alignment -> lcasr JSON format
  5. Write mapping.json

Usage (on DL server):
  # 1. Clone Lhotse manifests
  git clone --depth 1 --filter=blob:none --sparse \\
    https://github.com/revdotcom/speech-datasets.git speech-datasets-remote
  cd speech-datasets-remote
  git sparse-checkout init --cone
  git sparse-checkout set longform_reconstitution/tedlium
  cd ..

  # 2. Download TED-LIUM 3 (if not done)
  wget https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz
  tar -xzf TEDLIUM_release-3.tgz

  # 3. Run this script
  uv run --extra gpu scripts/prepare_tedlium.py \\
    --tedlium-dir /home/t23cs033/TEDLIUM_release-3 \\
    --manifest-dir /home/t23cs033/speech-datasets-remote \\
    --spec-dir    /home/t23cs033/tedlium_processed \\
    --output      /home/t23cs033/long-context-asr/data/mapping.json \\
    --split train
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from tqdm import tqdm


# ── helpers ──────────────────────────────────────────────────────────────────

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print('ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg', file=sys.stderr)
        sys.exit(1)


def sph_to_wav(sph_path: Path, wav_path: Path):
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error',
         '-i', str(sph_path), '-ar', '16000', '-ac', '1', str(wav_path)],
        check=True,
    )


def make_spectrogram(wav_path: Path, spec_path: Path) -> int:
    """Return number of mel frames."""
    from lcasr.utils.audio_tools import processing_chain
    spec = processing_chain(str(wav_path)).to(torch.float16)
    torch.save(spec, str(spec_path))
    return spec.shape[-1]


def load_lhotse_gz(gz_path: Path) -> list:
    records = []
    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def lhotse_words_to_lcasr(words: list) -> list:
    """
    Lhotse word alignment:
      {"start": float, "duration": float, "symbol": str}
    -> lcasr format (floras50 style):
      {"start": float, "end": float, "text": str}
    """
    result = []
    for w in words:
        symbol = w.get('symbol', w.get('word', '')).strip()
        if not symbol or symbol in ('<eps>', '<sil>', '<unk>'):
            continue
        start = float(w['start'])
        end = start + float(w.get('duration', 0.0))
        result.append({'start': round(start, 4), 'end': round(end, 4), 'text': symbol})
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Prepare TED-LIUM 3 for lcasr training')
    parser.add_argument('--tedlium-dir', required=True,
                        help='Root of TEDLIUM_release-3 (contains data/train/sph/)')
    parser.add_argument('--manifest-dir', required=True,
                        help='Root of speech-datasets repo clone')
    parser.add_argument('--spec-dir', required=True,
                        help='Output directory for .spec.pt and word JSON files')
    parser.add_argument('--output', required=True,
                        help='Output path for mapping.json')
    parser.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    parser.add_argument('--skip-audio', action='store_true',
                        help='Skip spectrogram generation (reuse existing .spec.pt)')
    args = parser.parse_args()

    check_ffmpeg()

    tedlium_dir = Path(args.tedlium_dir)
    spec_dir = Path(args.spec_dir)
    txt_dir = spec_dir / 'txt'
    spec_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Lhotse manifest ───────────────────────────────────────────────────────
    manifest_path = (
        Path(args.manifest_dir)
        / 'longform_reconstitution' / 'tedlium'
        / f'tedlium_{args.split}_lf_linked.jsonl.gz'
    )
    if not manifest_path.exists():
        print(f'ERROR: manifest not found:\n  {manifest_path}', file=sys.stderr)
        print('Run the sparse-checkout commands in the usage comment first.', file=sys.stderr)
        sys.exit(1)

    print(f'[1/4] Loading Lhotse manifest: {manifest_path.name}')
    supervisions = load_lhotse_gz(manifest_path)
    print(f'      {len(supervisions)} supervision records')

    # recording_id -> sorted list of supervisions
    by_recording: dict[str, list] = {}
    for sup in supervisions:
        by_recording.setdefault(sup['recording_id'], []).append(sup)
    for sups in by_recording.values():
        sups.sort(key=lambda s: s['start'])

    # ── Audio files ────────────────────────────────────────────────────────────
    sph_dir = tedlium_dir / 'data' / args.split / 'sph'
    if not sph_dir.exists():
        print(f'ERROR: sph directory not found: {sph_dir}', file=sys.stderr)
        sys.exit(1)

    sph_files = sorted(sph_dir.glob('*.sph'))
    print(f'[2/4] Found {len(sph_files)} .sph files')

    # ── Process each recording ─────────────────────────────────────────────────
    print(f'[3/4] Generating spectrograms{"  (skipped)" if args.skip_audio else ""}')
    errors, skipped = [], 0
    for sph_path in tqdm(sph_files, desc='audio'):
        rec_id = sph_path.stem
        spec_path = spec_dir / f'{rec_id}.spec.pt'

        if args.skip_audio or spec_path.exists():
            continue

        wav_path = spec_dir / f'{rec_id}_tmp.wav'
        try:
            sph_to_wav(sph_path, wav_path)
            make_spectrogram(wav_path, spec_path)
        except Exception as e:
            errors.append((rec_id, str(e)))
        finally:
            if wav_path.exists():
                wav_path.unlink()

    if errors:
        print(f'  WARN: {len(errors)} recordings failed:')
        for rec_id, msg in errors[:5]:
            print(f'    {rec_id}: {msg}')

    # ── Word-level JSON ────────────────────────────────────────────────────────
    print('[4/4] Building mapping.json')
    mapping = {}

    for sph_path in tqdm(sph_files, desc='mapping'):
        rec_id = sph_path.stem
        spec_path = spec_dir / f'{rec_id}.spec.pt'
        txt_path = txt_dir / f'{rec_id}.json'

        if not spec_path.exists():
            skipped += 1
            continue

        if rec_id not in by_recording:
            skipped += 1
            continue

        # word JSON（なければ生成）
        if not txt_path.exists():
            all_words = []
            for sup in by_recording[rec_id]:
                words = sup.get('alignment', {}).get('word', [])
                all_words.extend(words)

            lcasr_words = lhotse_words_to_lcasr(all_words)
            if not lcasr_words:
                print(f'  WARN: no word alignment for {rec_id}, skipping')
                skipped += 1
                continue

            with open(txt_path, 'w', encoding='utf-8') as f:
                json.dump(lcasr_words, f, ensure_ascii=False)

        spec = torch.load(spec_path, map_location='cpu', weights_only=True)
        duration = round(spec.shape[-1] * 160 / 16000, 2)  # HOP_LENGTH / SR

        mapping[rec_id] = {
            'audio': str(spec_path.resolve()),
            'txt': str(txt_path.resolve()),
            'duration': duration,
        }

    # ── Summary ────────────────────────────────────────────────────────────────
    total_hours = sum(v['duration'] for v in mapping.values()) / 3600
    print(f'\nResult: {len(mapping)} recordings ({total_hours:.1f} h), {skipped} skipped')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
