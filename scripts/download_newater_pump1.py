#!/usr/bin/env python3
"""
Download NeWater Pump 1 CSVs from Zenodo record 13808085.

Uses the official Zenodo REST API (no browser needed).

Usage (from repo root):
    python scripts/download_newater_pump1.py
    python scripts/download_newater_pump1.py --include-optional
    python scripts/download_newater_pump1.py --also-merge   # download then run prepare script
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ZENODO_RECORD_ID = "13808085"
DEFAULT_OUT = REPO_ROOT / "unseen_data" / "zenodo_newater_pump1_raw"

DOC_FILE = "Water Pump Dataset Decription.md"


def _load_file_lists():
    prep_path = REPO_ROOT / "scripts" / "prepare_newater_pump1.py"
    spec = importlib.util.spec_from_file_location("prepare_newater_pump1", prep_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod.PUMP1_FILES, mod.OPTIONAL_FILES


def _api_files() -> dict[str, str]:
    """Return {filename: content_url} from Zenodo API."""
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        import json

        data = json.loads(resp.read().decode())
    out: dict[str, str] = {}
    for f in data.get("files", []):
        key = f.get("key", "")
        links = f.get("links") or {}
        content = links.get("content") or links.get("self", "")
        if key and content:
            out[key] = content
    return out


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "CITS5206-capstone-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        chunk = 1024 * 256
        with dest.open("wb") as f:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r    {pct:5.1f}% ({done // (1024*1024)} MB)", end="", flush=True)
    if total:
        print()


def download_all(
    out_dir: Path,
    *,
    include_optional: bool = True,
    skip_existing: bool = True,
) -> list[str]:
    pump1, optional = _load_file_lists()
    wanted = list(pump1.keys())
    if include_optional:
        wanted.extend(optional.keys())
    # Description is markdown — fetch separately if needed
    doc_wanted = [DOC_FILE]

    api = _api_files()
    missing_api = [w for w in wanted if w not in api]
    if missing_api:
        print("Warning: not on Zenodo API:", missing_api)

    all_names = wanted + doc_wanted
    downloaded: list[str] = []
    for i, name in enumerate(all_names, 1):
        if name not in api:
            continue
        dest = out_dir / name
        if skip_existing and dest.is_file() and dest.stat().st_size > 0:
            print(f"[{i}/{len(all_names)}] skip (exists) {name}")
            downloaded.append(name)
            continue

        print(f"[{i}/{len(all_names)}] downloading {name} ...")
        try:
            _download(api[name], dest)
            downloaded.append(name)
            print(f"    -> {dest} ({dest.stat().st_size // (1024*1024)} MB)")
        except urllib.error.HTTPError as e:
            print(f"    FAILED HTTP {e.code}: {name}")
        except OSError as e:
            print(f"    FAILED: {name}: {e}")

    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NeWater Pump 1 from Zenodo")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-optional", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument(
        "--also-merge",
        action="store_true",
        help="Run prepare_newater_pump1.py after download",
    )
    args = parser.parse_args()

    print(f"Zenodo record {ZENODO_RECORD_ID} -> {args.out_dir}")
    pump1, _ = _load_file_lists()
    print(f"Required sensor files: {len(pump1)} (+ doc, + optional unless --no-optional)")

    download_all(
        args.out_dir,
        include_optional=not args.no_optional,
        skip_existing=not args.force,
    )

    if args.also_merge:
        print("\nMerging ...")
        prep = REPO_ROOT / "scripts" / "prepare_newater_pump1.py"
        import subprocess

        rc = subprocess.call([sys.executable, str(prep)], cwd=REPO_ROOT)
        sys.exit(rc)


if __name__ == "__main__":
    main()
