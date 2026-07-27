#!/usr/bin/env python3
"""GitHub LFS のポインタファイルから実際の MP3 をダウンロードする。"""

import sys
import requests
from pathlib import Path
from tqdm import tqdm

MEDIA_DIR = Path("~/long-context-asr/speech-datasets/earnings22/media").expanduser()
REPO = "revdotcom/speech-datasets"
BATCH_SIZE = 20  # 一度にリクエストするオブジェクト数


def read_pointer(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None, None
    oid = next((l.split("sha256:")[1].strip() for l in text.splitlines() if l.startswith("oid sha256:")), None)
    size = next((int(l.split()[1]) for l in text.splitlines() if l.startswith("size ")), None)
    return oid, size


def fetch_download_urls(objects: list) -> dict:
    """LFS Batch API でダウンロード URL を取得。oid -> url のマップを返す。"""
    resp = requests.post(
        f"https://github.com/{REPO}.git/info/lfs/objects/batch",
        json={"operation": "download", "transfers": ["basic"], "objects": objects},
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"Batch API エラー: {resp.status_code} {resp.text[:200]}")
        return {}
    result = {}
    for item in resp.json().get("objects", []):
        if "error" in item:
            print(f"  オブジェクトエラー {item['oid'][:12]}: {item['error']}")
            continue
        oid = item["oid"]
        url = item.get("actions", {}).get("download", {}).get("href")
        if url:
            result[oid] = url
    return result


def download_file(url: str, path: Path):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=path.name, leave=False
    ) as bar:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))


def main():
    # ポインタファイル（132〜200 bytes 程度）を収集
    pointers = []
    for f in sorted(MEDIA_DIR.glob("*.mp3")):
        if f.stat().st_size > 1000:
            continue  # 実体ファイルはスキップ
        oid, size = read_pointer(f)
        if oid and size:
            pointers.append((f, oid, size))

    if not pointers:
        print("ダウンロード対象なし（全ファイル取得済み）")
        return

    print(f"{len(pointers)} 件のポインタを検出。ダウンロード開始...")

    # BATCH_SIZE ずつ処理
    for i in range(0, len(pointers), BATCH_SIZE):
        batch = pointers[i:i + BATCH_SIZE]
        objects = [{"oid": oid, "size": size} for _, oid, size in batch]
        url_map = fetch_download_urls(objects)

        for path, oid, _ in batch:
            url = url_map.get(oid)
            if not url:
                print(f"  [skip] URL取得失敗: {path.name}")
                continue
            try:
                download_file(url, path)
                print(f"  [ok] {path.name}")
            except Exception as e:
                print(f"  [error] {path.name}: {e}")

    print("\n完了")


if __name__ == "__main__":
    main()
