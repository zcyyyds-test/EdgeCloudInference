"""Download MVTec Anomaly Detection dataset.

MVTec AD is a public benchmark dataset for industrial anomaly detection,
containing 15 categories of real-world industrial images with normal
and various defect types.

Usage:
    python scripts/download_mvtec.py                     # download all categories
    python scripts/download_mvtec.py --categories bottle tile  # specific categories
    python scripts/download_mvtec.py --list              # list available categories

Reference:
    Bergmann et al., "MVTec AD -- A Comprehensive Real-World Dataset for
    Unsupervised Anomaly Detection", CVPR 2019.
    https://www.mvtec.com/company/research/datasets/mvtec-ad
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

# MVTec AD download URLs (public mirror)
MVTEC_BASE_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# Individual category download IDs (from MVTec's public download)
CATEGORY_URLS = {
    "bottle": f"{MVTEC_BASE_URL}/420937370-1629951468/bottle.tar.xz",
    "cable": f"{MVTEC_BASE_URL}/420937413-1629951498/cable.tar.xz",
    "capsule": f"{MVTEC_BASE_URL}/420937454-1629951595/capsule.tar.xz",
    "carpet": f"{MVTEC_BASE_URL}/420937484-1629951672/carpet.tar.xz",
    "grid": f"{MVTEC_BASE_URL}/420937487-1629951814/grid.tar.xz",
    "hazelnut": f"{MVTEC_BASE_URL}/420937545-1629951845/hazelnut.tar.xz",
    "leather": f"{MVTEC_BASE_URL}/420937607-1629951964/leather.tar.xz",
    "metal_nut": f"{MVTEC_BASE_URL}/420937637-1629952063/metal_nut.tar.xz",
    "pill": f"{MVTEC_BASE_URL}/420938129-1629953099/pill.tar.xz",
    "screw": f"{MVTEC_BASE_URL}/420938130-1629953152/screw.tar.xz",
    "tile": f"{MVTEC_BASE_URL}/420938133-1629953189/tile.tar.xz",
    "toothbrush": f"{MVTEC_BASE_URL}/420938134-1629953256/toothbrush.tar.xz",
    "transistor": f"{MVTEC_BASE_URL}/420938166-1629953277/transistor.tar.xz",
    "wood": f"{MVTEC_BASE_URL}/420938383-1629953354/wood.tar.xz",
    "zipper": f"{MVTEC_BASE_URL}/420938385-1629953449/zipper.tar.xz",
}


def download_category(category: str, output_dir: Path, force: bool = False) -> bool:
    """Download and extract a single MVTec AD category."""
    cat_dir = output_dir / category
    if cat_dir.exists() and not force:
        print(f"  {category}: already exists, skipping (use --force to re-download)")
        return True

    url = CATEGORY_URLS.get(category)
    if not url:
        print(f"  {category}: unknown category, skipping")
        return False

    tar_path = output_dir / f"{category}.tar.xz"
    print(f"  {category}: downloading...", end=" ", flush=True)

    try:
        urllib.request.urlretrieve(url, tar_path)
        print("extracting...", end=" ", flush=True)

        with tarfile.open(tar_path, "r:xz") as tar:
            tar.extractall(path=output_dir)

        tar_path.unlink()
        print("done")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if tar_path.exists():
            tar_path.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(description="Download MVTec AD dataset")
    parser.add_argument("--output", type=str, default="data/mvtec",
                        help="Output directory (default: data/mvtec)")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Categories to download (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List available categories and exit")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if already exists")
    args = parser.parse_args()

    if args.list:
        print("Available MVTec AD categories:")
        for cat in CATEGORIES:
            print(f"  - {cat}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cats = args.categories or CATEGORIES
    invalid = [c for c in cats if c not in CATEGORIES]
    if invalid:
        print(f"Unknown categories: {invalid}")
        print(f"Available: {CATEGORIES}")
        sys.exit(1)

    print(f"Downloading MVTec AD to {output_dir}/")
    print(f"Categories: {cats}")
    print()

    success = 0
    for cat in cats:
        if download_category(cat, output_dir, args.force):
            success += 1

    print(f"\nDone: {success}/{len(cats)} categories downloaded")

    # Verify structure
    for cat in cats:
        cat_dir = output_dir / cat
        if cat_dir.exists():
            train_count = len(list((cat_dir / "train").rglob("*.png"))) if (cat_dir / "train").exists() else 0
            test_count = len(list((cat_dir / "test").rglob("*.png"))) if (cat_dir / "test").exists() else 0
            print(f"  {cat}: {train_count} train, {test_count} test images")


if __name__ == "__main__":
    main()
