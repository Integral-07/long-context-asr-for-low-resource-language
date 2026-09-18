#!/usr/bin/env python3
"""
Ainuコーパスを発話単位(transcriptsの各行=1学習サンプル)で、karolのwav2vec2
長文脈実験(実験⑤)の「条件A: 短い文脈」用に変換する。
scripts/prepare_ainu_wav2vec2_longcontext.py(コレクション連結版、条件B用)と
対になる、非連結版。scripts/prepare_ainu_utterances.py(実験2用、
mel-spectrogram版)の wav2vec2 版に相当する。

発話単位でチャンクを独立させることで、条件Aは「長文脈化なし」(1発話=1学習
サンプル、他の発話の情報は一切混ざらない)を保証する。コレクション連結済み
音声を単純に固定長でチャンク分割すると、隣接する複数発話が1チャンクに
混入してしまい、条件A/Bの違いが「文脈長」だけにならなくなるため、
このスクリプトを別に用意している。

train/dev/testの分割・単語タイムスタンプ計算・held-outコレクションは
prepare_ainu.py と共有する(import して再利用)。

Usage:
  uv run --extra gpu scripts/prepare_ainu_wav2vec2_utterances.py \\
    --corpus-dir ainu_corpus \\
    --feat-dir   ainu_w2v2_utterance_processed \\
    --output-dir data/ainu_wav2vec2_longcontext_utterances \\
    --base-model karolnowakowski/wav2vec2-large-xlsr-53-pretrain-ain
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from prepare_ainu import SR, load_clip, parse_transcript, split_for


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus-dir', default='ainu_corpus')
    parser.add_argument('--feat-dir', required=True,
                        help='出力先 (.w2v2feat.pt / 単語タイムスタンプJSON)')
    parser.add_argument('--output-dir', required=True,
                        help='train/dev/test それぞれの mapping.json を書き出すディレクトリ')
    parser.add_argument('--base-model', default='karolnowakowski/wav2vec2-large-xlsr-53-pretrain-ain')
    parser.add_argument('--max-duration-seconds', type=float, default=30.0,
                        help='これより長い発話は除外する(条件A側のchunk size=30.72sに収まる範囲、'
                             '実験1/2と揃えた外れ値対策)')
    args = parser.parse_args()

    from transformers import Wav2Vec2Model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w2v2 = Wav2Vec2Model.from_pretrained(args.base_model).to(device)
    w2v2.eval()
    feature_extractor = w2v2.feature_extractor

    corpus_dir = Path(args.corpus_dir)
    transcripts_dir = corpus_dir / 'transcripts'
    audio_dir = corpus_dir / 'audio'
    feat_dir = Path(args.feat_dir)
    txt_dir = feat_dir / 'txt'
    feat_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    trans_files = sorted(transcripts_dir.glob('at*.trans.txt'))
    mappings = {name: {} for name in ('train', 'dev', 'test')}
    missing_audio = 0
    empty_text = 0
    too_long = 0

    for trans_path in trans_files:
        collection_id = trans_path.stem.replace('.trans', '')
        entries = parse_transcript(trans_path)
        if not entries:
            continue  # at33: 書き起こしが空(音声なし)
        split = split_for(collection_id)

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
            if dur > args.max_duration_seconds:
                too_long += 1
                continue

            with torch.no_grad():
                feat = feature_extractor(waveform.to(device))  # (1, 512, time@50Hz)
            feat = feat.squeeze(0).to(torch.float16).cpu()

            feat_path = feat_dir / f'{seg_id}.w2v2feat.pt'
            torch.save(feat, str(feat_path))

            step = dur / len(tokens)
            words = [{
                'start': round(i * step, 4),
                'end': round((i + 1) * step, 4),
                'text': tok,
            } for i, tok in enumerate(tokens)]

            txt_path = txt_dir / f'{seg_id}.json'
            with open(txt_path, 'w', encoding='utf-8') as f:
                json.dump(words, f, ensure_ascii=False)

            mappings[split][seg_id] = {
                'audio': str(feat_path.resolve()),
                'txt': str(txt_path.resolve()),
                'duration': round(dur, 2),
            }

    if missing_audio:
        print(f'WARN: {missing_audio} transcript rows had no matching audio file')
    if empty_text:
        print(f'WARN: {empty_text} transcript rows had no words after splitting, skipped')
    if too_long:
        print(f'WARN: {too_long} transcript rows exceeded {args.max_duration_seconds}s, skipped')

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
