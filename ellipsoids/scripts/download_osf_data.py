#!/usr/bin/env python3
"""Download the Hong et al. (2025) dataset from OSF into data/.

Usage
-----
    python scripts/download_osf_data.py [--data-dir PATH]

The dataset lives at https://osf.io/k27js (node ID: k27js).
Files are fetched via the public OSF v2 API and written to data/ (or the
directory supplied via --data-dir), preserving the folder structure found
on OSF.

No third-party packages are required — only stdlib urllib and json.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

OSF_NODE = "k27js"
OSF_API = "https://api.osf.io/v2"
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def _get_json(url: str) -> dict:  # type: ignore[type-arg]
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())  # type: ignore[no-any-return]
    except urllib.error.HTTPError as exc:
        print(f"  HTTP {exc.code} fetching {url}", file=sys.stderr)
        raise


def _download_file(download_url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  skip  {dest}  (already exists)")
        return
    print(f"  fetch {dest}")
    req = urllib.request.Request(download_url)
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
        while chunk := resp.read(1 << 20):  # 1 MB chunks
            fh.write(chunk)


def _walk_folder(folder_url: str, dest_dir: Path) -> None:
    """Recursively walk an OSF storage folder and download all files."""
    url = folder_url
    while url:
        data = _get_json(url)
        for item in data.get("data", []):
            attrs = item.get("attributes", {})
            kind = attrs.get("kind")
            name = attrs.get("name", "unnamed")
            links = item.get("links", {})

            if kind == "file":
                download_url = links.get("download", "")
                if download_url:
                    _download_file(download_url, dest_dir / name)
                else:
                    print(f"  warn  no download link for {name}", file=sys.stderr)

            elif kind == "folder":
                # Recurse: the folder's own file-listing URL
                folder_files_url = (
                    item.get("relationships", {})
                    .get("files", {})
                    .get("links", {})
                    .get("related", {})
                    .get("href", "")
                )
                if folder_files_url:
                    _walk_folder(folder_files_url, dest_dir / name)

        # Pagination
        url = data.get("links", {}).get("next") or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Destination directory (default: data/ next to repo root)",
    )
    args = parser.parse_args()
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading OSF node {OSF_NODE!r} → {data_dir}")
    root_url = f"{OSF_API}/nodes/{OSF_NODE}/files/osfstorage/"
    _walk_folder(root_url, data_dir)
    print("Done.")


if __name__ == "__main__":
    main()
