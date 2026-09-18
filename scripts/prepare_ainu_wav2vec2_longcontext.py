#!/usr/bin/env python3
"""
Ainu語コーパス(ainu_corpus/, Tuytah Collection)を、karolの事前学習済み
wav2vec2-Ainuを使う長文脈実験(実験⑤)用の mapping.json 形式に変換する。

scripts/prepare_ainu.py(実験③, mel-spectrogram版)とほぼ同じ処理だが、
`to_spectogram()` の代わりに、凍結した wav2vec2 の CNN feature encoder
(feature_extractor のみ、eval + no_grad)に生波形を通して得た
(conv_dim=512, time@50Hz) の特徴量を保存する点だけが異なる。この
CNN出力を使うことで、学習時にはfeature_projection以降の
Transformer(24層、事前学習済み)だけを対象にすればよく、CNNの
受容野(約25ms)に起因するチャンク境界の近似誤差も、コレクション
全体を一度にCNNへ通すことで実質発生しない。

train/dev/testの分割・単語タイムスタンプの線形補間は prepare_ainu.py と
同一のロジックを共有する(import して再利用、重複させない)。

条件B(長文脈, sequence_scheduler使用)用。条件A(短文脈, 発話単位)は
scripts/prepare_ainu_wav2vec2_utterances.py を使う。

Usage:
  uv run --extra cpu scripts/prepare_ainu_wav2vec2_longcontext.py \\
    --corpus-dir ainu_corpus \\
    --feat-dir   ainu_w2v2_processed \\
    --output-dir data/ainu_wav2vec2_longcontext_collections \\
    --base-model karolnowakowski/wav2vec2-large-xlsr-53-pretrain-ain
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from prepare_ainu import SR, load_clip, parse_transcript, split_for

HOP_LENGTH = 320  # wav2vec2-large-xlsr の累積 conv stride (5*2*2*2*2*2*2), 50Hz相当


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus-dir', default='ainu_corpus',
                        help='transcripts/ と audio/ を含むディレクトリ')
    parser.add_argument('--feat-dir', required=True,
                        help='出力先 (.w2v2feat.pt / 単語タイムスタンプJSON)')
    parser.add_argument('--output-dir', required=True,
                        help='train/dev/test それぞれの mapping.json を書き出すディレクトリ')
    parser.add_argument('--base-model', default='karolnowakowski/wav2vec2-large-xlsr-53-pretrain-ain')
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
    print(f'{len(trans_files)} collections found')

    mappings = {name: {} for name in ('train', 'dev', 'test')}
    total_missing_audio = 0

    for trans_path in trans_files:
        collection_id = trans_path.stem.replace('.trans', '')
        entries = parse_transcript(trans_path)
        if not entries:
            # at33: 書き起こしが空(音声なし)。
            continue
        split = split_for(collection_id)

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
        duration = round(full_waveform.shape[-1] / SR, 2)

        with torch.no_grad():
            feat = feature_extractor(full_waveform.to(device))  # (1, conv_dim=512, time@50Hz)
        feat = feat.squeeze(0).to(torch.float16).cpu()  # (512, time)

        feat_path = feat_dir / f'{collection_id}.w2v2feat.pt'
        torch.save(feat, str(feat_path))

        txt_path = txt_dir / f'{collection_id}.json'
        with open(txt_path, 'w', encoding='utf-8') as f:
            json.dump(words, f, ensure_ascii=False)

        mappings[split][collection_id] = {
            'audio': str(feat_path.resolve()),
            'txt': str(txt_path.resolve()),
            'duration': duration,
        }
        print(f'  {collection_id} [{split}]: {len(clips)} clips, {len(words)} words, '
              f'{feat.shape[-1]} frames ({duration/60:.1f} min)')

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
