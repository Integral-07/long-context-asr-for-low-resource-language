#!/usr/bin/env python3
"""Earnings-22 前処理スクリプト
force_aligned_nlp_references の word-level タイムスタンプを使って
ファインチューニング用マニフェストを作成する。
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from lcasr.utils.audio_tools import processing_chain, total_seconds


def parse_aligned_nlp(nlp_path: Path):
    """force-aligned NLPファイルをword-level alignmentリストに変換。
    ts/endTsが空の行（del扱い）はスキップ。
    """
    words = []
    for line in nlp_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        token, _, ts, end_ts = parts[0], parts[1], parts[2], parts[3]
        if not ts or not end_ts:
            continue
        try:
            start, end = float(ts), float(end_ts)
        except ValueError:
            continue
        if token.strip():
            words.append({"start": start, "end": end, "text": token.strip()})
    return words


def build_manifest(args):
    media_dir = Path(args.earnings22_root) / "media"
    nlp_dir   = Path(args.earnings22_root) / "transcripts" / "force_aligned_nlp_references"
    spec_dir  = Path(args.spec_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)

    mp3_files = sorted(media_dir.glob("*.mp3"))
    if args.max_records:
        mp3_files = mp3_files[:args.max_records]

    manifest = {}
    skipped = 0

    for mp3_path in tqdm(mp3_files, desc="Processing Earnings-22"):
        stem = mp3_path.stem
        nlp_path = nlp_dir / f"{stem}.aligned.nlp"

        if not nlp_path.exists():
            print(f"[skip] NLPなし: {stem}")
            skipped += 1
            continue

        words = parse_aligned_nlp(nlp_path)
        if not words:
            print(f"[skip] タイムスタンプなし: {stem}")
            skipped += 1
            continue

        spec_path = spec_dir / f"{stem}.spec.pt"
        if args.force or not spec_path.exists():
            spec = processing_chain(str(mp3_path))
            torch.save(spec, spec_path)
        else:
            spec = torch.load(spec_path)

        # duration in seconds (100 frames/sec: sr=16000, hop=160)
        duration = spec.shape[-1] / 100.0

        txt_path = spec_dir / f"{stem}.txt.json"
        if args.force or not txt_path.exists():
            with open(txt_path, "w", encoding="utf-8") as f:
                json.dump({"word_timestamps": words}, f, ensure_ascii=False)

        manifest[stem] = {
            "audio":    str(spec_path),
            "txt":      str(txt_path),
            "duration": duration,
        }

    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n完了: {len(manifest)} 件 → {args.manifest_out}  (スキップ: {skipped})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--earnings22-root", required=True,
                        help="speech-datasets/earnings22 のパス")
    parser.add_argument("--spec-dir", required=True,
                        help="スペクトログラム保存先ディレクトリ")
    parser.add_argument("--manifest-out", required=True,
                        help="出力マニフェストJSONのパス")
    parser.add_argument("--max-records", type=int, default=0,
                        help="処理件数上限（0=全件）")
    parser.add_argument("--force", action="store_true",
                        help="既存spec.ptを上書き")
    main_args = parser.parse_args()
    build_manifest(main_args)


if __name__ == "__main__":
    main()
