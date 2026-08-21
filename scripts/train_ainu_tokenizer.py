#!/usr/bin/env python3
"""
実験2/3(論文アーキテクチャを0から学習)用に、Ainu語のtrainスプリットのみから
SentencePiece(BPE)トークナイザを学習する。英語のSpotify事前学習用トークナイザ
(4095語彙)はAinu語には使えないため、コーパス規模に合わせて小さい語彙で作り直す。

dev/testのテキストは学習に含めない(vocab leakage防止)。

Usage:
  uv run --extra gpu python scripts/train_ainu_tokenizer.py \\
    --corpus-dir ainu_corpus \\
    --save-dir   lcasr/artifacts/ainu \\
    --vocab-size 500
"""

import argparse
import re
from pathlib import Path

from lcasr.utils.audio_tools import train_tokenizer

from prepare_ainu import parse_transcript

ID_RE = re.compile(r'^([A-Za-z]+\d+)$')

# prepare_ainu_utterances.py / prepare_ainu_wav2vec2.py と同じ held-out コレクション
TEST_COLLECTIONS = {'at08', 'at30', 'at40', 'at13', 'at22'}
DEV_COLLECTIONS = {'at54', 'at18', 'at27', 'at36', 'at10'}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus-dir', default='ainu_corpus')
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--vocab-size', type=int, default=500)
    parser.add_argument('--raw-txt-path', default=None,
                        help='学習に使う結合テキストの一時保存先(省略時は save-dir 内)')
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_txt_path = Path(args.raw_txt_path) if args.raw_txt_path else save_dir / 'train_text.txt'

    n_lines = 0
    with open(raw_txt_path, 'w', encoding='utf-8') as out:
        for trans_path in sorted(corpus_dir.glob('transcripts/at*.trans.txt')):
            collection_id = trans_path.stem.replace('.trans', '')
            if collection_id in TEST_COLLECTIONS or collection_id in DEV_COLLECTIONS:
                continue
            for _, _, text in parse_transcript(trans_path):
                if text.split():
                    out.write(text + '\n')
                    n_lines += 1

    print(f'train text: {n_lines} lines -> {raw_txt_path}')

    train_tokenizer(
        raw_txt=str(raw_txt_path),
        save_path=str(save_dir) + '/',
        vocab_size=args.vocab_size,
    )
    print(f'tokenizer saved under {save_dir}/ (tokenizer.model, tokenizer.vocab)')


if __name__ == '__main__':
    main()
