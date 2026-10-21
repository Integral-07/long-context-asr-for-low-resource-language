#!/usr/bin/env python3
"""
実験⑫: 渡辺さんのfinetune_xlsr53_ainu notebook(cell 8-11)と同じ手順で、HF
`Wav2Vec2CTCTokenizer`形式の文字レベルvocabをfold別に構築する。
scripts/train_ainu_tokenizer.py --model-type char のHF版。

train splitのテキストから文字集合を作り、スペースを word_delimiter_token
`|` に、[UNK]/[PAD]を追加する。アイヌ語コーパスには句読点が存在しない
ことを確認済み(PROGRESS_LOG 10/17)なので、notebookのREMOVE_CHARSに相当
する正規化は行わない(対象が存在しないため)。

dev/testのテキストは学習に含めない(vocab leakage防止、train_ainu_tokenizer.py
と同じ方針)。

Usage:
  uv run --extra gpu python scripts/train_ainu_hf_tokenizer.py \\
    --corpus-dir ainu_corpus \\
    --save-dir   lcasr/artifacts/ainu_hftok_kfold/fold0 \\
    --fold 0
"""

import argparse
import json
from pathlib import Path

from transformers import Wav2Vec2CTCTokenizer

from prepare_ainu import parse_transcript
from train_ainu_tokenizer import is_heldout


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus-dir', default='ainu_corpus')
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--fold', type=int, required=True,
                        help='このfold番号のコレクションをtestとして除外する'
                             '(0..ainu_kfold_splits.N_FOLDS-1)')
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    chars: set[str] = set()
    n_lines = 0
    for trans_path in sorted(corpus_dir.glob('transcripts/at*.trans.txt')):
        collection_id = trans_path.stem.replace('.trans', '')
        try:
            if is_heldout(collection_id, args.fold):
                continue
        except KeyError:
            continue  # at33: 書き起こしが空(音声なし)。fold割り当ても持たない。
        for _, _, text in parse_transcript(trans_path):
            if text.split():
                chars.update(text)
                n_lines += 1

    print(f'train text: {n_lines} lines, {len(chars)} unique chars')

    vocab = sorted(chars)
    vocab_dict = {ch: i for i, ch in enumerate(vocab)}
    if ' ' in vocab_dict:
        vocab_dict['|'] = vocab_dict.pop(' ')
    vocab_dict['[UNK]'] = len(vocab_dict)
    vocab_dict['[PAD]'] = len(vocab_dict)

    vocab_path = save_dir / 'vocab.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False)
    print(f'vocab size: {len(vocab_dict)}')

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_path),
        unk_token='[UNK]',
        pad_token='[PAD]',
        word_delimiter_token='|',
    )
    tokenizer.save_pretrained(str(save_dir))
    print(f'tokenizer saved under {save_dir}/')


if __name__ == '__main__':
    main()
