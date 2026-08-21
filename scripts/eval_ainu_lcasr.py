#!/usr/bin/env python3
"""
実験2/3(論文アーキテクチャ, lcasr)共通の評価スクリプト。
mapping.json(dev/testなど)の各エントリに対してモデルを実行し、WERを算出する。

Usage:
  uv run --extra gpu python scripts/eval_ainu_lcasr.py \\
    --checkpoint checkpoints/ainu_scratch_short/step_332550.pt \\
    --mapping data/ainu_utterance/test_mapping.json \\
    --tokenizer lcasr/artifacts/ainu/tokenizer.model \\
    --output checkpoints/ainu_scratch_short/test_predictions.json
"""

import argparse
import json
from pathlib import Path

import torch
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm

import lcasr
from lcasr.eval.utils import fetch_logits, decode_beams_lm
from lcasr.eval.wer import word_error_rate_detail
from lcasr.utils.general import load_model, get_model_class


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mapping', required=True, help='評価対象の mapping.json')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seq-len', type=int, default=-1,
                        help='-1 でチェックポイントの audio_chunking.size を使用')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_config = checkpoint['config']
    args.config = model_config

    tokenizer = lcasr.utils.audio_tools.load_tokenizer(tokenizer_path=args.tokenizer)
    model = load_model(args.config, tokenizer.vocab_size(),
                        model_class=get_model_class(config=args.config, args=args))
    model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    vocab = [tokenizer.id_to_piece(i) for i in range(tokenizer.get_piece_size())] + ['']
    decoder = build_ctcdecoder(vocab, kenlm_model_path=None, alpha=None, beta=None)

    mapping = json.loads(Path(args.mapping).read_text(encoding='utf-8'))
    ids = sorted(mapping.keys())
    print(f'{len(ids)} utterances')

    hypotheses, references, results = [], [], []
    with torch.no_grad():
        for uid in tqdm(ids):
            entry = mapping[uid]
            spec = torch.load(entry['audio'], map_location='cpu', weights_only=True).to(torch.float32)
            words = json.loads(Path(entry['txt']).read_text(encoding='utf-8'))
            reference = ' '.join(w['text'] for w in words)

            logits = fetch_logits(args, model, spec, args.seq_len, 0, tokenizer, use_tqdm=False)
            ds_factor = spec.shape[-1] / logits.shape[0]
            decoded, _ = decode_beams_lm([logits], decoder, beam_width=1, ds_factor=ds_factor)
            hypothesis = decoded[0]['text']

            hypotheses.append(hypothesis)
            references.append(reference)
            results.append({'id': uid, 'reference': reference, 'prediction': hypothesis})

    wer, words_n, ins_rate, del_rate, sub_rate = word_error_rate_detail(
        hypotheses=hypotheses, references=references)
    print(f'WER: {wer:.4f}  (words={words_n}, ins={ins_rate:.4f}, del={del_rate:.4f}, sub={sub_rate:.4f})')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'wer': wer, 'words': words_n,
            'ins_rate': ins_rate, 'del_rate': del_rate, 'sub_rate': sub_rate,
            'results': results,
        }, f, ensure_ascii=False, indent=2)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
