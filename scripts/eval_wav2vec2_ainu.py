#!/usr/bin/env python3
"""
実験1(wav2vec2-Ainu ファインチューニング)の最終評価。
学習済みモデルを held-out test セットに対して実行し、WERを算出する。

Usage:
  uv run --extra gpu python scripts/eval_wav2vec2_ainu.py \\
    --model-dir ainu_wav2vec2_ft/best \\
    --data-dir  data/ainu_wav2vec2 \\
    --output    ainu_wav2vec2_ft/test_predictions.json
"""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from jiwer import wer, cer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

SR = 16000


def load_waveform(path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != SR:
        waveform = torchaudio.transforms.Resample(sr, SR)(waveform)
    return waveform.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--data-dir', default='data/ainu_wav2vec2')
    parser.add_argument('--split', default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir).to(device).eval()

    records = json.loads((Path(args.data_dir) / f'{args.split}.json').read_text(encoding='utf-8'))
    print(f'{args.split}: {len(records)} utterances')

    predictions, references, results = [], [], []
    with torch.no_grad():
        for rec in records:
            waveform = load_waveform(rec['audio'])
            inputs = processor(waveform.numpy(), sampling_rate=SR, return_tensors='pt')
            input_values = inputs.input_values.to(device)
            logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_text = processor.batch_decode(pred_ids)[0]

            predictions.append(pred_text)
            references.append(rec['text'])
            results.append({'id': rec['id'], 'reference': rec['text'], 'prediction': pred_text})

    overall_wer = wer(references, predictions)
    overall_cer = cer(references, predictions)
    print(f'{args.split} WER: {overall_wer:.4f}')
    print(f'{args.split} CER: {overall_cer:.4f}')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'wer': overall_wer, 'cer': overall_cer, 'results': results},
                   f, ensure_ascii=False, indent=2)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
