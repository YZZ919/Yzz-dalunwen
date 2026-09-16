#!/usr/bin/env python3
"""Download low-light inputs from the accessible SICE mirror.

The original SICE Part1 Google Drive archive is not reachable from the
current machine.  ``okhater/SICE`` is a public mirror-derived paired dataset;
this downloader keeps its provenance explicit and only fetches one low-light
input (``low1.jpg``) per sample.  It is therefore suitable for a reproducible
clean-room training run, but it must not be described as the exact official
2422/600 Part1 split unless that archive is later obtained and verified.  The
``all_low`` mode downloads every low-light image in each sample as an
independent, mirror-derived data variant.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def fetch_json(url: str):
    request = Request(url, headers={"User-Agent": "IR-VIO-cleanroom/1.0"})
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8")), response.headers.get("Link")


def list_files(base_url: str, repo_id: str, revision: str) -> list[dict]:
    url = (
        f"{base_url.rstrip('/')}/api/datasets/{repo_id}/tree/{revision}"
        "?recursive=true&expand=false&limit=1000"
    )
    rows: list[dict] = []
    while url:
        payload, link_header = fetch_json(url)
        rows.extend(item for item in payload if item.get("type") == "file")
        next_url = None
        if link_header:
            for part in link_header.split(","):
                if 'rel="next"' in part:
                    next_url = part.split("<", 1)[1].split(">", 1)[0]
                    break
        if next_url and next_url.startswith("https://huggingface.co/"):
            next_url = base_url.rstrip("/") + next_url.split("https://huggingface.co", 1)[1]
        url = next_url
    return rows


def download_file(
    url: str,
    target: Path,
    expected_size: int | None,
    retries: int = 5,
):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        return "skipped"
    partial = target.with_suffix(target.suffix + ".part")
    attempts = max(1, int(retries) + 1)
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            request = Request(url, headers={"User-Agent": "IR-VIO-cleanroom/1.0"})
            with urlopen(request, timeout=300) as response, partial.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise IOError(
                    f"size mismatch for {url}: expected {expected_size}, "
                    f"got {partial.stat().st_size}"
                )
            partial.replace(target)
            return "downloaded"
        except Exception as exc:  # network mirrors can fail transiently
            last_error = exc
            if partial.exists():
                partial.unlink()
            if attempt < attempts:
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last_error


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="okhater/SICE")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--base-url", default="https://hf-mirror.com")
    parser.add_argument(
        "--output-root", type=Path,
        default=workspace / "datasets" / "sice_mirror_low1",
    )
    parser.add_argument(
        "--selection",
        choices=("low1", "all_low"),
        default="low1",
        help="one low1 image per sample (default) or all low*.jpg images",
    )
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel downloads (keep 1 for the historical serial behavior)",
    )
    parser.add_argument(
        "--retries", type=int, default=5,
        help="retries per file after a transient network error",
    )
    args = parser.parse_args()

    entries = list_files(args.base_url, args.repo_id, args.revision)
    selected: list[dict] = []
    for item in entries:
        path = str(item.get("path", ""))
        if args.selection == "low1":
            keep = path.endswith("/low1.jpg")
        else:
            keep = path.lower().endswith(".jpg") and "/low" in path.lower()
        if not keep:
            continue
        if path.startswith("train/"):
            split = "train"
        elif path.startswith("test/"):
            split = "val"
        else:
            continue
        selected.append({"split": split, **item})
    selected.sort(key=lambda item: (item["split"], item["path"]))
    limits = {"train": args.max_train, "val": args.max_val}
    selected = [
        item
        for split in ("train", "val")
        for item in selected
        if item["split"] == split and (limits[split] is None or sum(
            1 for prior in selected if prior["split"] == split and prior["path"] <= item["path"]
        ) <= limits[split])
    ]
    if not selected:
        raise RuntimeError("API 中没有找到 train/test 的 low1.jpg")

    def fetch_one(index_item):
        index, item = index_item
        rel = Path(item["path"])
        target = args.output_root / rel
        source_url = (
            f"{args.base_url.rstrip('/')}/datasets/{args.repo_id}/resolve/{args.revision}/"
            f"{item['path']}?download=true"
        )
        status = download_file(source_url, target, item.get("size"), retries=args.retries)
        record = {
            "index": index,
            "split": item["split"],
            "path": item["path"],
            "size": item.get("size"),
            "target": str(target),
            "url": source_url,
            "status": status,
        }
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)
        return record

    indexed = list(enumerate(selected, start=1))
    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        records = [fetch_one(item) for item in indexed]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            records = list(pool.map(fetch_one, indexed))
    for record in records:
        print(json.dumps(record, ensure_ascii=False))

    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": f"sice_mirror_{args.selection}_downloaded",
        "repo_id": args.repo_id,
        "revision": args.revision,
        "base_url": args.base_url,
        "selection": (
            "one low1.jpg per train/test sample directory"
            if args.selection == "low1"
            else "all low*.jpg images per train/test sample directory"
        ),
        "train_count": sum(record["split"] == "train" for record in records),
        "val_count": sum(record["split"] == "val" for record in records),
        "warning": "mirror-derived data; not verified identical to official SICE Part1 2422/600 split",
        "records": records,
    }
    (args.output_root / "download_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: metadata[k] for k in ("status", "train_count", "val_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
