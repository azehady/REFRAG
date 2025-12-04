#!/usr/bin/env python3
"""
Download Bangla Wikipedia dump from Wikimedia.
"""

import os
import requests
from tqdm import tqdm
from pathlib import Path


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        dest_path: Destination file path
        chunk_size: Size of chunks to download
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def main():
    # URL for the latest Bangla Wikipedia articles dump
    url = "https://dumps.wikimedia.org/bnwiki/latest/bnwiki-latest-pages-articles.xml.bz2"

    # Destination directory
    dest_dir = Path(__file__).parent.parent.parent / "data" / "bnwiki"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_file = dest_dir / "bnwiki-latest-pages-articles.xml.bz2"

    print(f"Downloading Bangla Wikipedia dump...")
    print(f"URL: {url}")
    print(f"Destination: {dest_file}")
    print()

    try:
        download_file(url, dest_file)
        print(f"\nDownload complete! File saved to: {dest_file}")
        print(f"File size: {dest_file.stat().st_size / (1024**2):.2f} MB")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        raise


if __name__ == "__main__":
    main()
