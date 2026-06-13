"""Fix train/val duplicate chips.

570 chip stems appear in both images/train/ and images/val/, causing
train-val leakage. This script removes them from val/ (images + labels),
keeping the train/ copies intact.

Usage:
    python scripts/fix_split_duplicates.py          # dry run — prints what would be removed
    python scripts/fix_split_duplicates.py --apply  # actually deletes the val/ duplicates
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

YOLO_DIR = Path("data/yolo_dataset")


def find_duplicates() -> list[str]:
    """Return chip stems present in both train/ and val/ image directories."""
    train_stems = {p.stem for p in (YOLO_DIR / "images" / "train").glob("*.jpg")}
    val_stems = {p.stem for p in (YOLO_DIR / "images" / "val").glob("*.jpg")}
    return sorted(train_stems & val_stems)


def fix_duplicates(duplicates: list[str], apply: bool) -> None:
    """Remove duplicate chips from val/ images and labels directories.

    Args:
        duplicates: List of chip stems to remove from val/.
        apply: If False, only prints what would be removed (dry run).
    """
    val_img_dir = YOLO_DIR / "images" / "val"
    val_lbl_dir = YOLO_DIR / "labels" / "val"

    removed_imgs = 0
    removed_lbls = 0
    missing_lbls = 0

    for stem in duplicates:
        img_path = val_img_dir / f"{stem}.jpg"
        lbl_path = val_lbl_dir / f"{stem}.txt"

        if img_path.exists():
            if apply:
                img_path.unlink()
            removed_imgs += 1

        if lbl_path.exists():
            if apply:
                lbl_path.unlink()
            removed_lbls += 1
        else:
            missing_lbls += 1

    action = "Removed" if apply else "Would remove"
    log.info(f"{action} {removed_imgs} images and {removed_lbls} label files from val/")
    if missing_lbls:
        log.info(f"  ({missing_lbls} chips had no label file in val/ — already clean)")

    # Print updated split sizes
    if apply:
        log.info("")
        for split in ("train", "val", "test"):
            n = len(list((YOLO_DIR / "images" / split).glob("*.jpg")))
            log.info(f"  {split}: {n} chips")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove train/val duplicate chips from val/")
    parser.add_argument("--apply", action="store_true", help="Actually delete files (default: dry run)")
    args = parser.parse_args()

    duplicates = find_duplicates()
    log.info(f"Found {len(duplicates)} chip stems in both train/ and val/")

    if not duplicates:
        log.info("Nothing to do.")
        return

    if not args.apply:
        log.info("DRY RUN — pass --apply to delete")
        log.info(f"Sample duplicates: {duplicates[:5]}")

    fix_duplicates(duplicates, apply=args.apply)


if __name__ == "__main__":
    main()
