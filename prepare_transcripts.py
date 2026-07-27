#!/usr/bin/env python3
"""
subset10のverbatim NLPファイルからfull_transcripts.jsonを生成する。
eval/earnings22/run.py が期待する {meeting_id: text} 形式で出力する。
"""

import json
import argparse
from pathlib import Path


def parse_nlp(nlp_path: Path) -> str:
    lines = nlp_path.read_text(encoding="utf-8").splitlines()
    tokens = []
    for line in lines[1:]:  # 1行目はヘッダ
        parts = line.split("|")
        if len(parts) < 1:
            continue
        token = parts[0].strip()
        if token:
            tokens.append(token)
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp-dir", required=True,
                        help="verbatim_transcripts/nlp_references ディレクトリ")
    parser.add_argument("--out", required=True,
                        help="出力 full_transcripts.json のパス")
    args = parser.parse_args()

    nlp_dir = Path(args.nlp_dir)
    transcripts = {}
    for nlp_path in sorted(nlp_dir.glob("*.nlp")):
        meeting_id = nlp_path.stem
        transcripts[meeting_id] = parse_nlp(nlp_path)
        print(f"  {meeting_id}: {len(transcripts[meeting_id].split())} words")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)

    print(f"\n完了: {len(transcripts)} 件 → {args.out}")


if __name__ == "__main__":
    main()
