"""Pack a self-contained training bundle for execution on a GPU machine.

What this script does
---------------------
1. Preserves preliminary training results in-place (models/plots_preliminary/,
   models/helipad_yolov8s_preliminary.pt, models/helipad_run_preliminary_results.csv)
   so preliminary vs. final comparison is possible after final training.

2. Builds a training_bundle/ directory containing every file train_yolo.py needs —
   images, labels, registry CSVs, src modules, requirements — ready to copy to a
   GPU machine via USB drive or network share.

3. Writes training_bundle/run.bat for one-click training on Windows.

Usage
-----
    python scripts/pack_training.py            # build bundle at ./training_bundle/
    python scripts/pack_training.py --out D:/  # write bundle to D:/training_bundle/
    python scripts/pack_training.py --preserve-only  # only preserve preliminary results, no bundle
"""

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]

PRELIM_WEIGHTS_SRC = _ROOT / "models" / "helipad_yolov8s.pt"
PRELIM_WEIGHTS_DST = _ROOT / "models" / "helipad_yolov8s_preliminary.pt"
PRELIM_PLOTS_SRC   = _ROOT / "models" / "plots"
PRELIM_PLOTS_DST   = _ROOT / "models" / "plots_preliminary"
PRELIM_RESULTS_SRC = _ROOT / "models" / "helipad_run" / "results.csv"
PRELIM_RESULTS_DST = _ROOT / "models" / "helipad_run_preliminary_results.csv"


# ── Step 1: preserve preliminary results ─────────────────────────────────────

def preserve_preliminary() -> None:
    """Copy preliminary weights, plots, and training curves to *_preliminary paths."""
    preserved = []
    skipped = []

    for src, dst in [
        (PRELIM_WEIGHTS_SRC, PRELIM_WEIGHTS_DST),
        (PRELIM_RESULTS_SRC, PRELIM_RESULTS_DST),
    ]:
        if not src.exists():
            skipped.append(str(src))
            continue
        if dst.exists():
            log.info("  Already preserved: %s (skipping)", dst.name)
            continue
        shutil.copy2(src, dst)
        preserved.append(dst.name)

    if PRELIM_PLOTS_SRC.exists() and not PRELIM_PLOTS_DST.exists():
        shutil.copytree(PRELIM_PLOTS_SRC, PRELIM_PLOTS_DST)
        preserved.append("plots_preliminary/")
    elif PRELIM_PLOTS_DST.exists():
        log.info("  Already preserved: plots_preliminary/ (skipping)")
    else:
        skipped.append(str(PRELIM_PLOTS_SRC))

    if preserved:
        log.info("  Preserved: %s", ", ".join(preserved))
    if skipped:
        log.info("  Not found (skipped): %s", ", ".join(skipped))


# ── Step 2: build the bundle ──────────────────────────────────────────────────

BUNDLE_FILES = [
    # scripts
    ("scripts/train_yolo.py",          "scripts/train_yolo.py"),
    # src modules
    ("src/__init__.py",                "src/__init__.py"),
    ("src/analysis.py",               "src/analysis.py"),
    ("src/hie.py",                    "src/hie.py"),
    # registry CSVs (needed for baseline computation)
    ("data/faa_adip_enriched.csv",    "data/faa_adip_enriched.csv"),
    ("data/osm_helipads_raw.csv",     "data/osm_helipads_raw.csv"),
    # review decisions (informational — train_yolo.py just logs the count)
    ("data/yolo_dataset/review_decisions.csv",  "data/yolo_dataset/review_decisions.csv"),
    ("data/yolo_dataset/dataset.yaml",          "data/yolo_dataset/dataset.yaml"),
    # requirements
    ("requirements.txt",               "requirements.txt"),
]

BUNDLE_DIRS = [
    # (src relative to _ROOT, dst relative to bundle root)
    ("data/yolo_dataset/images",  "data/yolo_dataset/images"),
    ("data/yolo_dataset/labels",  "data/yolo_dataset/labels"),
]

BUNDLE_PRELIM = [
    # preliminary results included so GPU machine can show comparison plots
    ("models/helipad_yolov8s_preliminary.pt", "models/helipad_yolov8s_preliminary.pt"),
    ("models/plots_preliminary",              "models/plots_preliminary"),
    ("models/helipad_run_preliminary_results.csv", "models/helipad_run_preliminary_results.csv"),
]

RUN_BAT = r"""@echo off
setlocal
echo ============================================================
echo  SkyRoute HIE -- YOLOv8s Final Training
echo ============================================================
echo.

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python 3.11+ is on PATH.
    pause & exit /b 1
)
echo.

echo [2/3] Starting training (GPU device 0)...
echo       Results will be written to models/helipad_run/ and models/plots/
echo       Press Ctrl+C to interrupt -- training resumes with --skip-train
echo.
python scripts\train_yolo.py --device 0 --epochs 50
if errorlevel 1 (
    echo ERROR: Training script failed. Check output above.
    pause & exit /b 1
)
echo.

echo [3/3] Done. Copy the following back to your main machine:
echo       models\helipad_yolov8s.pt       (final weights)
echo       models\plots\                   (final comparison plots)
echo       models\helipad_run\results.csv  (training curves)
echo.
pause
"""


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576


def build_bundle(bundle_root: Path) -> None:
    bundle_root.mkdir(parents=True, exist_ok=True)
    log.info("Bundle root: %s", bundle_root)

    # Individual files
    for rel_src, rel_dst in BUNDLE_FILES:
        src = _ROOT / rel_src
        dst = bundle_root / rel_dst
        if not src.exists():
            log.info("  MISSING (skip): %s", rel_src)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        log.info("  Copied: %s", rel_dst)

    # Large image/label directories
    for rel_src, rel_dst in BUNDLE_DIRS:
        src = _ROOT / rel_src
        dst = bundle_root / rel_dst
        if not src.exists():
            log.info("  MISSING (skip): %s", rel_src)
            continue
        if dst.exists():
            log.info("  Already exists, removing: %s", rel_dst)
            shutil.rmtree(dst)
        log.info("  Copying %s  (%.0f MB)...", rel_dst, _dir_size_mb(src))
        shutil.copytree(src, dst)
        log.info("    -> done")

    # Preliminary results
    log.info("  Copying preliminary results for comparison...")
    for rel_src, rel_dst in BUNDLE_PRELIM:
        src = _ROOT / rel_src
        dst = bundle_root / rel_dst
        if not src.exists():
            log.info("    MISSING (skip): %s", rel_src)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        log.info("    Copied: %s", rel_dst)

    # Empty models/ dir for training output
    (bundle_root / "models").mkdir(parents=True, exist_ok=True)

    # run.bat
    bat_path = bundle_root / "run.bat"
    bat_path.write_text(RUN_BAT, encoding="utf-8")
    log.info("  Wrote: run.bat")

    # Summary
    total_mb = _dir_size_mb(bundle_root)
    log.info("")
    log.info("Bundle size: %.0f MB  (%.1f GB)", total_mb, total_mb / 1024)
    log.info("")
    log.info("Instructions:")
    log.info("  1. Copy %s to the GPU machine (USB or network share)", bundle_root.name)
    log.info("  2. On the GPU machine: double-click run.bat  (or: python scripts/train_yolo.py --device 0)")
    log.info("  3. When done, copy back:")
    log.info("       models/helipad_yolov8s.pt")
    log.info("       models/plots/")
    log.info("       models/helipad_run/results.csv")
    log.info("  4. Place copied files into THIS repo's models/ folder")
    log.info("  5. Run: python scripts/train_yolo.py --skip-train  (regenerates comparison plots)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pack training bundle for GPU machine")
    parser.add_argument("--out", type=Path, default=_ROOT,
                        help="Directory in which training_bundle/ is created (default: project root)")
    parser.add_argument("--preserve-only", action="store_true",
                        help="Only preserve preliminary results; do not build the bundle")
    args = parser.parse_args()

    log.info("=== Step 1: Preserve preliminary results ===")
    preserve_preliminary()
    log.info("")

    if args.preserve_only:
        log.info("--preserve-only set. Done.")
        return

    log.info("=== Step 2: Build training bundle ===")
    bundle_root = args.out / "training_bundle"
    build_bundle(bundle_root)


if __name__ == "__main__":
    main()
