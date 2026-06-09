"""Zero-shot model comparison on the 747 NE US NAIP test chips.

Run this BEFORE YOLO training completes to establish the academic baseline.
Reads chips from data/yolo_dataset/images/test/ and GT labels from
data/yolo_dataset/labels/test/.

Usage
-----
  python scripts/compare_zero_shot.py                      # all models, all chips
  python scripts/compare_zero_shot.py --limit 50           # smoke test
  python scripts/compare_zero_shot.py --models yolo_world  # single model
  python scripts/compare_zero_shot.py --resume             # skip already-done chips

Output
------
  data/yolo_dataset/zero_shot_results.csv  — one row per (chip, model)
  Prints per-model Precision / Recall / F1 / mAP@50 / mean latency table.

IoU note
--------
  Ground truth bboxes come from build_yolo_dataset.py (synthetic centre-based
  annotations).  Results should be re-run after manual annotation corrections
  are applied to get final numbers.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))

from src.hie import (
    IMG_PX,
    load_chip,
    detect_classical,
    detect_yolo_world,
    detect_florence2,
    detect_dino,
    load_yolo_world_model,
)

_TEST_IMAGES_DIR = _PROJ_ROOT / "data" / "yolo_dataset" / "images" / "test"
_TEST_LABELS_DIR = _PROJ_ROOT / "data" / "yolo_dataset" / "labels" / "test"
_OUTPUT_CSV = _PROJ_ROOT / "data" / "yolo_dataset" / "zero_shot_results.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("compare_zero_shot")

# ── available model keys ─────────────────────────────────────────────────────
ALL_MODELS = ["classical", "yolo_world", "florence2", "dino"]


# ────────────────────────────────────────────────────────────────────────────
# GT label parsing
# ────────────────────────────────────────────────────────────────────────────

def parse_gt_label(label_path: Path) -> list[int] | None:
    """Parse a YOLO-format label file into pixel bbox [x1,y1,x2,y2].

    Returns None if the file is empty (negative chip) or missing.
    """
    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError):
        return None
    if not text or text.startswith("#"):
        return None
    # First non-comment line: class cx_n cy_n w_n h_n
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            _, cx_n, cy_n, w_n, h_n = (float(p) for p in parts[:5])
        except ValueError:
            continue
        cx = cx_n * IMG_PX
        cy = cy_n * IMG_PX
        w  = w_n  * IMG_PX
        h  = h_n  * IMG_PX
        return [
            int(cx - w / 2),
            int(cy - h / 2),
            int(cx + w / 2),
            int(cy + h / 2),
        ]
    return None


# ────────────────────────────────────────────────────────────────────────────
# IoU
# ────────────────────────────────────────────────────────────────────────────

def compute_iou(pred: list[int], gt: list[int]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(pred[0], gt[0])
    iy1 = max(pred[1], gt[1])
    ix2 = min(pred[2], gt[2])
    iy2 = min(pred[3], gt[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    pred_area = max(0, pred[2] - pred[0]) * max(0, pred[3] - pred[1])
    gt_area   = max(0, gt[2]   - gt[0])   * max(0, gt[3]   - gt[1])
    union = pred_area + gt_area - inter
    return inter / union if union > 0 else 0.0


# ────────────────────────────────────────────────────────────────────────────
# Per-model runner
# ────────────────────────────────────────────────────────────────────────────

def _run_model(model_key: str, image, models: dict) -> dict:
    """Dispatch image to the correct detect function.

    Args:
        model_key: One of ALL_MODELS.
        image: PIL RGB Image.
        models: Dict of pre-loaded model objects (populated lazily).

    Returns:
        Detection result dict from src/hie.py.
    """
    if model_key == "classical":
        return detect_classical(image)

    elif model_key == "yolo_world":
        if "yolo_world" not in models:
            log.info("Loading YOLO-World …")
            models["yolo_world"] = load_yolo_world_model()
        return detect_yolo_world(image, model=models["yolo_world"])

    elif model_key == "florence2":
        if models.get("florence2_disabled"):
            return {"detected": False, "bbox_px": None, "cx": None, "cy": None,
                    "confidence": 0.0, "method": "florence2_disabled", "latency_s": 0.0}
        if "florence2_model" not in models:
            log.info("Loading Florence-2 (may take a minute on first run) …")
            from src.hie import load_florence2_model
            try:
                models["florence2_model"], models["florence2_proc"] = load_florence2_model()
            except ImportError as e:
                log.error("Florence-2 skipped: %s", e)
                models["florence2_disabled"] = True
                return {"detected": False, "bbox_px": None, "cx": None, "cy": None,
                        "confidence": 0.0, "method": "florence2_disabled", "latency_s": 0.0}
        return detect_florence2(
            image,
            model=models["florence2_model"],
            processor=models["florence2_proc"],
        )

    elif model_key == "dino":
        if "dino_model" not in models:
            log.info("Loading Grounding DINO (~661 MB, may be slow) …")
            from src.hie import load_dino_model
            models["dino_model"], models["dino_proc"] = load_dino_model()
        return detect_dino(
            image,
            model=models["dino_model"],
            processor=models["dino_proc"],
        )

    raise ValueError(f"Unknown model key: {model_key}")


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict], model_key: str, iou_threshold: float = 0.50) -> dict:
    """Compute Precision, Recall, F1, and mAP@50 for one model.

    Detection is a TP if detected=True AND IoU ≥ iou_threshold vs GT.
    Only chips with a GT label are positive examples.

    Args:
        rows: List of result row dicts from the comparison run.
        model_key: Which model to compute metrics for.
        iou_threshold: IoU threshold for TP (default 0.50).

    Returns:
        Dict with keys: precision, recall, f1, n_pos, n_neg, tp, fp, fn,
                        mean_latency_s, mean_iou_tp.
    """
    model_rows = [r for r in rows if r["model"] == model_key]

    tp = fp = fn = tn = 0
    iou_tp_list: list[float] = []
    latencies: list[float] = []

    for r in model_rows:
        has_gt = r["has_gt"]
        detected = r["detected"]
        iou = float(r["iou"])
        latencies.append(float(r["latency_s"]))

        if has_gt and detected and iou >= iou_threshold:
            tp += 1
            iou_tp_list.append(iou)
        elif has_gt and (not detected or iou < iou_threshold):
            fn += 1
        elif not has_gt and detected:
            fp += 1
        else:
            tn += 1

    n_pos = tp + fn
    n_neg = fp + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "model":          model_key,
        "precision":      precision,
        "recall":         recall,
        "f1":             f1,
        "n_pos":          n_pos,
        "n_neg":          n_neg,
        "tp":             tp,
        "fp":             fp,
        "fn":             fn,
        "mean_latency_s": float(np.mean(latencies)) if latencies else 0.0,
        "mean_iou_tp":    float(np.mean(iou_tp_list)) if iou_tp_list else 0.0,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot helipad detection comparison on 747 NAIP test chips."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=["classical", "yolo_world"],
        help=(
            "Models to run (default: classical yolo_world). "
            "florence2 requires transformers<4.49 — add with --models classical yolo_world florence2 "
            "only if transformers<4.49 is installed."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of chips to process (for smoke tests).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chips already present in the output CSV.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU threshold for true positive (default 0.50).",
    )
    args = parser.parse_args()

    # ── collect test chips ───────────────────────────────────────────────────
    chip_paths = sorted(_TEST_IMAGES_DIR.glob("*.jpg"))
    if not chip_paths:
        log.error("No chips found in %s — run build_yolo_dataset.py first.", _TEST_IMAGES_DIR)
        sys.exit(1)

    if args.limit:
        chip_paths = chip_paths[: args.limit]

    log.info("Found %d test chips. Models: %s", len(chip_paths), args.models)

    # ── load already-done (chip, model) pairs for resume ────────────────────
    done: set[tuple[str, str]] = set()
    existing_rows: list[dict] = []
    if args.resume and _OUTPUT_CSV.exists():
        with _OUTPUT_CSV.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                done.add((row["chip_stem"], row["model"]))
        log.info("Resuming — %d rows already in output.", len(existing_rows))

    # ── CSV setup ────────────────────────────────────────────────────────────
    fieldnames = [
        "chip_stem", "model", "has_gt", "detected",
        "confidence", "iou", "latency_s",
        "pred_x1", "pred_y1", "pred_x2", "pred_y2",
        "gt_x1",   "gt_y1",   "gt_x2",   "gt_y2",
    ]

    write_mode = "a" if (args.resume and _OUTPUT_CSV.exists()) else "w"
    csv_file = _OUTPUT_CSV.open(write_mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_mode == "w":
        writer.writeheader()

    # ── model cache ──────────────────────────────────────────────────────────
    models: dict = {}
    new_rows: list[dict] = []

    # ── main loop ────────────────────────────────────────────────────────────
    total = len(chip_paths) * len(args.models)
    processed = 0

    for chip_path in chip_paths:
        stem = chip_path.stem
        label_path = _TEST_LABELS_DIR / f"{stem}.txt"
        gt_bbox = parse_gt_label(label_path)
        has_gt = gt_bbox is not None

        # Load chip once per image, re-use across models
        try:
            image = load_chip(chip_path)
        except Exception as exc:
            log.warning("Failed to load chip %s: %s", stem, exc)
            continue

        for model_key in args.models:
            if (stem, model_key) in done:
                processed += 1
                continue

            try:
                result = _run_model(model_key, image, models)
            except Exception as exc:
                log.warning("[%s / %s] Detection failed: %s", stem, model_key, exc)
                result = {
                    "detected": False, "bbox_px": None, "cx": None, "cy": None,
                    "confidence": 0.0, "method": model_key, "latency_s": 0.0,
                }

            pred_bbox = result.get("bbox_px")
            iou = compute_iou(pred_bbox, gt_bbox) if (pred_bbox and has_gt) else 0.0

            row = {
                "chip_stem":  stem,
                "model":      model_key,
                "has_gt":     int(has_gt),
                "detected":   int(result["detected"]),
                "confidence": f"{result['confidence']:.4f}",
                "iou":        f"{iou:.4f}",
                "latency_s":  f"{result['latency_s']:.4f}",
                "pred_x1":    pred_bbox[0] if pred_bbox else "",
                "pred_y1":    pred_bbox[1] if pred_bbox else "",
                "pred_x2":    pred_bbox[2] if pred_bbox else "",
                "pred_y2":    pred_bbox[3] if pred_bbox else "",
                "gt_x1":      gt_bbox[0] if gt_bbox else "",
                "gt_y1":      gt_bbox[1] if gt_bbox else "",
                "gt_x2":      gt_bbox[2] if gt_bbox else "",
                "gt_y2":      gt_bbox[3] if gt_bbox else "",
            }
            writer.writerow(row)
            csv_file.flush()
            new_rows.append(row)

            processed += 1
            if processed % 50 == 0 or processed == total:
                pct = 100 * processed / total
                log.info("[%d/%d  %.0f%%]  last: %s / %s  IoU=%.2f",
                         processed, total, pct, stem, model_key, iou)

    csv_file.close()

    # ── metrics table ────────────────────────────────────────────────────────
    all_rows = existing_rows + new_rows
    if not all_rows:
        log.warning("No results to summarise.")
        return

    print("\n" + "=" * 80)
    print(f"{'Model':<15}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  "
          f"{'#Pos':>5}  {'TP':>5}  {'FP':>5}  {'FN':>5}  "
          f"{'mIoU(TP)':>9}  {'Lat(s)':>7}")
    print("-" * 80)

    for model_key in args.models:
        m = compute_metrics(all_rows, model_key, args.iou_threshold)
        print(f"{m['model']:<15}  "
              f"{m['precision']:>6.3f}  {m['recall']:>6.3f}  {m['f1']:>6.3f}  "
              f"{m['n_pos']:>5d}  {m['tp']:>5d}  {m['fp']:>5d}  {m['fn']:>5d}  "
              f"{m['mean_iou_tp']:>9.3f}  {m['mean_latency_s']:>7.3f}")

    print("=" * 80)
    print(f"\nResults saved to: {_OUTPUT_CSV}")
    print("Note: GT bboxes are synthetic (centre-based) until manual annotation"
          " corrections are applied.\n")


if __name__ == "__main__":
    main()
