#!/usr/bin/env python3
"""
READMEに記載のHuggingFaceチェックポイントをダウンロードする。
EncDecSconformerV2のエンコーダ重み初期化に使う acoustic (CTC) モデル。
デコーダ重みは含まれないため strict=False でロードされる。

使用方法:
  python3 download_checkpoint.py
  python3 download_checkpoint.py --out-dir /path/to/dir --repo-id rjflynn2/lcasr-9L-768D-6H-RB-1p5M
"""

import argparse
from pathlib import Path

# earnings22_ft.yaml の n_layers=3 / d_model=768 に対応するモデル
DEFAULT_REPO = "rjflynn2/lcasr-3L-768D-6H-RB-1p5M"

AVAILABLE_MODELS = {
    "3L-768D":  "rjflynn2/lcasr-3L-768D-6H-RB-1p5M",
    "6L-768D":  "rjflynn2/lcasr-6L-768D-6H-RB-1p5M",
    "9L-768D":  "rjflynn2/lcasr-9L-768D-6H-RB-1p5M",
    "12L-256D": "rjflynn2/lcasr-12L-256D-8H-RB-1p5M",
    "6L-256D":  "rjflynn2/lcasr-6L-256D-8H-RB-1p5M",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="./checkpoints/pretrained",
                        help="保存先ディレクトリ (デフォルト: ./checkpoints/pretrained)")
    parser.add_argument("--repo-id", default=DEFAULT_REPO,
                        help=f"HuggingFace リポジトリID (デフォルト: {DEFAULT_REPO})")
    parser.add_argument("--list-models", action="store_true",
                        help="利用可能なモデル一覧を表示して終了")
    args = parser.parse_args()

    if args.list_models:
        print("利用可能なモデル (README より):")
        for name, repo in AVAILABLE_MODELS.items():
            marker = " ← earnings22_ft.yaml のデフォルト" if repo == DEFAULT_REPO else ""
            print(f"  {name:12s}  {repo}{marker}")
        return

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise SystemExit("huggingface_hub が未インストールです: pip install huggingface_hub")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"リポジトリ: {args.repo_id}")
    print(f"保存先:     {out_dir.resolve()}\n")

    print("ファイル一覧を取得中...")
    all_files = list(list_repo_files(args.repo_id))
    pt_files = [f for f in all_files if f.endswith(".pt")]

    if not pt_files:
        raise SystemExit(f"ERROR: {args.repo_id} に .pt ファイルが見つかりません")

    print("見つかった .pt ファイル:")
    for f in pt_files:
        print(f"  {f}")

    # step_105360.pt を優先 (bin/load_pretrained.py のデフォルト値)
    target = next((f for f in pt_files if "step_105360" in f and "repeat_1" not in f), None)
    if target is None:
        target = next((f for f in pt_files if "step_105360" in f), None)
    if target is None:
        target = pt_files[0]

    print(f"\nダウンロード: {target}")
    local_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=target,
        local_dir=str(out_dir),
    )

    print(f"\n完了: {local_path}")
    print("\nearnings22_ft.yaml の checkpointing セクションに以下を追記してください:")
    print(f"  pretrained: {local_path}")


if __name__ == "__main__":
    main()
