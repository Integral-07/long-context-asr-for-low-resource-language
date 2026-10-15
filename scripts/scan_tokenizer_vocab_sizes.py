#!/usr/bin/env python3
"""
トークナイザのvocab_size(文字レベル込み)を複数スキャンし、各設定の
tok/word比(SentencePieceでencodeした平均トークン数 / 空白区切りの単語数)
を一括計測する。

10/08・10/12の進捗ログで報告した「vocab={120,150,200,250,500,文字94}の
スキャン」はこれまでアドホックに(スクリプト化せず)行っていたため、再現性
確保のために切り出した。RESULT.md(experiment/9-small-bpe-tokenizerブランチ)
「トークナイザ選択のまとめ」セクションの tok/word 列はこのスクリプトで
再現できる。

Usage:
  uv run --extra gpu python scripts/scan_tokenizer_vocab_sizes.py \\
    --corpus-dir ainu_corpus \\
    --vocab-sizes 120 150 200 250 500 \\
    --include-char

  # k-foldのfold0のtrain側のみでスキャンする場合
  uv run --extra gpu python scripts/scan_tokenizer_vocab_sizes.py \\
    --fold 0 --vocab-sizes 120 150 200 250 500 --include-char
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from lcasr.utils.audio_tools import load_tokenizer, train_tokenizer

from prepare_ainu import parse_transcript
from train_ainu_tokenizer import is_heldout


def build_train_text(corpus_dir: Path, fold: int | None, raw_txt_path: Path) -> int:
    n_lines = 0
    with open(raw_txt_path, 'w', encoding='utf-8') as out:
        for trans_path in sorted(corpus_dir.glob('transcripts/at*.trans.txt')):
            collection_id = trans_path.stem.replace('.trans', '')
            try:
                if is_heldout(collection_id, fold):
                    continue
            except KeyError:
                continue  # at33: 書き起こしが空(音声なし)
            for _, _, text in parse_transcript(trans_path):
                if text.split():
                    out.write(text + '\n')
                    n_lines += 1
    return n_lines


def tok_per_word(tokenizer_model: Path, lines: list[str]) -> float:
    sp = load_tokenizer(str(tokenizer_model))
    n_tokens = sum(len(sp.encode(line)) for line in lines)
    n_words = sum(len(line.split()) for line in lines)
    return n_tokens / n_words


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--corpus-dir', default='ainu_corpus')
    parser.add_argument(
        '--fold', type=int, default=None,
        help='train_ainu_tokenizer.pyの--foldと同じ意味。省略時は固定split(実験②③⑤等と同じtrain/dev/test)。'
    )
    parser.add_argument('--vocab-sizes', type=int, nargs='+', default=[120, 150, 200, 250, 500])
    parser.add_argument(
        '--include-char', action='store_true',
        help='文字レベル(vocab_sizeはtrain splitの実測文字種数+3から自動算出)も比較に含める'
    )
    parser.add_argument(
        '--work-dir', default=None,
        help='各vocab_sizeの一時トークナイザの保存先。省略時は一時ディレクトリを作成し終了時に削除する。'
    )
    parser.add_argument('--keep-work-dir', action='store_true', help='--work-dir指定時、終了後も削除しない')
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    cleanup = args.work_dir is None
    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix='ainu_tok_scan_'))
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_txt_path = work_dir / 'train_text.txt'
    n_lines = build_train_text(corpus_dir, args.fold, raw_txt_path)
    lines = raw_txt_path.read_text(encoding='utf-8').splitlines()
    print(f'train text: {n_lines} lines (fold={args.fold})')

    configs: list[tuple[str, int]] = [('bpe', v) for v in sorted(set(args.vocab_sizes))]
    if args.include_char:
        chars: set[str] = set()
        for line in lines:
            chars.update(line)
        configs.append(('char', len(chars) + 3))

    results = []
    for model_type, vocab_size in configs:
        save_dir = work_dir / f'{model_type}_{vocab_size}'
        save_dir.mkdir(parents=True, exist_ok=True)
        train_tokenizer(
            raw_txt=str(raw_txt_path),
            save_path=str(save_dir) + '/',
            vocab_size=vocab_size,
            model_type=model_type,
        )
        ratio = tok_per_word(save_dir / 'tokenizer.model', lines)
        label = '文字' if model_type == 'char' else f'BPE-{vocab_size}'
        results.append((label, vocab_size, ratio))
        print(f'{label:>10}  vocab_size={vocab_size:<5}  tok/word={ratio:.2f}')

    if cleanup:
        shutil.rmtree(work_dir, ignore_errors=True)

    print()
    print('| トークナイザ | vocab_size | tok/word |')
    print('|---|---|---|')
    for label, vocab_size, ratio in results:
        print(f'| {label} | {vocab_size} | {ratio:.2f} |')


if __name__ == '__main__':
    main()
