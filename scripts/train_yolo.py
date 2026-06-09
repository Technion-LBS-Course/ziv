"""Train YOLOv8s helipad detector and evaluate against the registry-agreement baseline.

IMPORTANT: run  streamlit run scripts/annotate_dataset.py  and click
"Apply all decisions" BEFORE running this script, so your annotations
are written to the label files.

Usage
-----
  python scripts/train_yolo.py                        # full pipeline
  python scripts/train_yolo.py --skip-train           # evaluate existing weights
  python scripts/train_yolo.py --epochs 30            # override epoch count
  python scripts/train_yolo.py --device 0             # use GPU 0
  python scripts/train_yolo.py --batch 8              # smaller batch for low RAM

Outputs
-------
  models/helipad_run/weights/best.pt     trained weights
  models/helipad_yolov8s.pt             copy of best weights (used by pipeline)
  models/plots/pr_curve.png             Precision-Recall curve + baseline point
  models/plots/f1_threshold.png         Precision / Recall / F1 vs conf threshold
  models/plots/comparison_bar.png       YOLO vs Baseline — P / R / F1 side-by-side
  models/plots/confusion_matrix.png     Confusion matrix at best-F1 threshold
  models/plots/yolo_metrics_bar.png     P / R / F1 / Accuracy bar chart (YOLO only)

Baseline definition
-------------------
  TP  = FAA+OSM pairs matched by FAA-ID  AND  distance < 10 m
  FP  = FAA+OSM pairs matched by FAA-ID  AND  distance >= 10 m
  FN  = 50% of OSM-only records (helipads FAA missed entirely)
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))

from src.analysis import match_by_faa_id, haversine_matrix
from src.hie import IMG_PX

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_YAML     = _PROJ_ROOT / "data" / "yolo_dataset" / "dataset.yaml"
TEST_IMAGES_DIR  = _PROJ_ROOT / "data" / "yolo_dataset" / "images" / "test"
TEST_LABELS_DIR  = _PROJ_ROOT / "data" / "yolo_dataset" / "labels" / "test"
DECISIONS_CSV    = _PROJ_ROOT / "data" / "yolo_dataset" / "review_decisions.csv"
MODELS_DIR       = _PROJ_ROOT / "models"
PLOTS_DIR        = MODELS_DIR / "plots"
RUN_DIR          = MODELS_DIR / "helipad_run"
BEST_WEIGHTS     = RUN_DIR / "weights" / "best.pt"
PIPELINE_WEIGHTS = MODELS_DIR / "helipad_yolov8s.pt"
FAA_CSV          = _PROJ_ROOT / "data" / "faa_adip_enriched.csv"
OSM_CSV          = _PROJ_ROOT / "data" / "osm_helipads_raw.csv"

IOU_THRESH = 0.50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_yolo")

_PLT_STYLE = {
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
}


# ────────────────────────────────────────────────────────────────────────────
# Label / IoU utilities (mirrors compare_zero_shot.py)
# ────────────────────────────────────────────────────────────────────────────

def parse_gt_label(label_path: Path) -> list[int] | None:
    """Return pixel [x1,y1,x2,y2] from a YOLO label file, or None (negative chip)."""
    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError):
        return None
    if not text:
        return None
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
        cx, cy = cx_n * IMG_PX, cy_n * IMG_PX
        w, h   = w_n  * IMG_PX, h_n  * IMG_PX
        return [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]
    return None


def compute_iou(pred: list[int], gt: list[int]) -> float:
    ix1, iy1 = max(pred[0], gt[0]), max(pred[1], gt[1])
    ix2, iy2 = min(pred[2], gt[2]), min(pred[3], gt[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    pa = max(0, pred[2] - pred[0]) * max(0, pred[3] - pred[1])
    ga = max(0, gt[2]   - gt[0])   * max(0, gt[3]   - gt[1])
    union = pa + ga - inter
    return inter / union if union > 0 else 0.0


# ────────────────────────────────────────────────────────────────────────────
# Baseline (registry-agreement)
# ────────────────────────────────────────────────────────────────────────────

def compute_baseline(distance_threshold_m: float = 10.0) -> dict:
    """Precision / Recall / F1 for the FAA+OSM cross-source agreement baseline.

    TP  = FAA-ID matched pair, distance < distance_threshold_m
    FP  = FAA-ID matched pair, distance >= distance_threshold_m
    FN  = 50% of OSM-only records (helipads absent from FAA)
    """
    if not FAA_CSV.exists() or not OSM_CSV.exists():
        log.warning("FAA or OSM CSV not found — baseline skipped")
        return {}

    faa_df = pd.read_csv(FAA_CSV, low_memory=False)
    osm_df = pd.read_csv(OSM_CSV, low_memory=False)

    for df in (faa_df, osm_df):
        for c in ("lat", "lon"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    id_matches = match_by_faa_id(faa_df, osm_df)
    if id_matches.empty:
        log.warning("No FAA-ID matches found — check IDENT/faa columns in CSVs")
        return {}

    tp = fp = 0
    for _, row in id_matches.iterrows():
        faa_r = faa_df.loc[row["faa_idx"]]
        osm_r = osm_df.loc[row["osm_idx"]]
        try:
            dist = float(haversine_matrix(
                np.array([float(faa_r["lat"])]), np.array([float(faa_r["lon"])]),
                np.array([float(osm_r["lat"])]), np.array([float(osm_r["lon"])]),
            )[0, 0])
        except Exception:
            fp += 1
            continue
        if dist < distance_threshold_m:
            tp += 1
        else:
            fp += 1

    matched_osm = set(id_matches["osm_idx"].tolist())
    n_osm_only  = int((~osm_df.index.isin(matched_osm)).sum())
    fn          = int(n_osm_only * 0.5)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    log.info(
        "Baseline  TP=%d  FP=%d  FN=%d  n_osm_only=%d  P=%.3f  R=%.3f  F1=%.3f",
        tp, fp, fn, n_osm_only, precision, recall, f1,
    )
    return dict(
        tp=tp, fp=fp, fn=fn,
        n_matched=len(id_matches), n_osm_only=n_osm_only,
        precision=precision, recall=recall, f1=f1,
    )


# ────────────────────────────────────────────────────────────────────────────
# YOLO inference
# ────────────────────────────────────────────────────────────────────────────

def run_test_inference(weights_path: Path) -> list[dict]:
    """Run YOLO on every test chip; return list of per-chip result dicts."""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    chip_paths = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    log.info("Inference on %d test chips …", len(chip_paths))

    rows = []
    for i, cp in enumerate(chip_paths, 1):
        stem       = cp.stem
        gt_bbox    = parse_gt_label(TEST_LABELS_DIR / f"{stem}.txt")
        has_gt     = gt_bbox is not None

        preds = model.predict(str(cp), conf=0.001, verbose=False)
        boxes = preds[0].boxes

        if len(boxes) == 0:
            confidence, iou = 0.0, 0.0
        else:
            idx        = int(boxes.conf.argmax())
            confidence = float(boxes.conf[idx])
            x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[idx].tolist())
            iou = compute_iou([x1, y1, x2, y2], gt_bbox) if has_gt else 0.0

        rows.append({"stem": stem, "has_gt": has_gt,
                     "confidence": confidence, "iou": iou})

        if i % 100 == 0 or i == len(chip_paths):
            log.info("  %d / %d done", i, len(chip_paths))

    return rows


# ────────────────────────────────────────────────────────────────────────────
# YOLO metrics (sweep threshold)
# ────────────────────────────────────────────────────────────────────────────

def compute_yolo_metrics(rows: list[dict]) -> dict:
    """Sweep 201 confidence thresholds; return curves + best-F1 counts."""
    thresholds = np.linspace(0.0, 1.0, 201)
    precisions, recalls, f1s, accs = [], [], [], []

    for thresh in thresholds:
        tp = fp = fn = tn = 0
        for r in rows:
            hit = r["confidence"] >= thresh and r["iou"] >= IOU_THRESH
            fir = r["confidence"] >= thresh   # any detection above thresh
            if r["has_gt"] and hit:
                tp += 1
            elif r["has_gt"] and not hit:
                fn += 1
            elif not r["has_gt"] and fir:
                fp += 1
            else:
                tn += 1
        n    = tp + fp + fn + tn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc  = (tp + tn) / n  if n > 0       else 0.0
        precisions.append(prec); recalls.append(rec)
        f1s.append(f1);          accs.append(acc)

    precisions = np.array(precisions)
    recalls    = np.array(recalls)
    f1s        = np.array(f1s)
    accs       = np.array(accs)

    best = int(np.argmax(f1s))
    best_thresh = float(thresholds[best])

    # Counts at best threshold
    tp = fp = fn = tn = 0
    for r in rows:
        hit = r["confidence"] >= best_thresh and r["iou"] >= IOU_THRESH
        fir = r["confidence"] >= best_thresh
        if r["has_gt"] and hit:      tp += 1
        elif r["has_gt"] and not hit: fn += 1
        elif not r["has_gt"] and fir: fp += 1
        else:                         tn += 1

    log.info(
        "YOLO best threshold=%.3f  TP=%d  FP=%d  FN=%d  TN=%d  "
        "P=%.3f  R=%.3f  F1=%.3f  Acc=%.3f",
        best_thresh, tp, fp, fn, tn,
        precisions[best], recalls[best], f1s[best], accs[best],
    )

    return dict(
        thresholds=thresholds,
        precisions=precisions, recalls=recalls,
        f1s=f1s, accs=accs,
        best_threshold=best_thresh,
        best_precision=float(precisions[best]),
        best_recall=float(recalls[best]),
        best_f1=float(f1s[best]),
        best_accuracy=float(accs[best]),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


# ────────────────────────────────────────────────────────────────────────────
# Plots
# ────────────────────────────────────────────────────────────────────────────

def _save(fig: "plt.Figure", path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log.info("Saved plot: %s", path)


def plot_pr_curve(yolo: dict, baseline: dict, out: Path) -> None:
    with plt.rc_context(_PLT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(yolo["recalls"], yolo["precisions"],
                color="#2563eb", lw=2,
                label=f"YOLOv8s fine-tuned  F1={yolo['best_f1']:.3f}")
        ax.scatter([yolo["best_recall"]], [yolo["best_precision"]],
                   color="#2563eb", s=90, zorder=5,
                   label=f"Best conf threshold ({yolo['best_threshold']:.2f})")
        if baseline:
            ax.scatter([baseline["recall"]], [baseline["precision"]],
                       color="#dc2626", marker="D", s=110, zorder=5,
                       label=f"Registry baseline  F1={baseline['f1']:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1);       ax.set_ylim(0, 1.05)
        ax.set_title("Precision–Recall Curve")
        ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
        _save(fig, out)


def plot_f1_threshold(yolo: dict, out: Path) -> None:
    with plt.rc_context(_PLT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(yolo["thresholds"], yolo["f1s"],
                color="#7c3aed", lw=2.5, label="F1")
        ax.plot(yolo["thresholds"], yolo["precisions"],
                color="#059669", lw=1.5, ls="--", label="Precision")
        ax.plot(yolo["thresholds"], yolo["recalls"],
                color="#d97706", lw=1.5, ls="--", label="Recall")
        ax.axvline(yolo["best_threshold"], color="gray", ls=":", lw=1.5,
                   label=f"Best thresh = {yolo['best_threshold']:.2f}")
        ax.set_xlabel("Confidence Threshold"); ax.set_ylabel("Score")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_title("Precision / Recall / F1 vs Confidence Threshold")
        ax.legend(); ax.grid(True, alpha=0.3)
        _save(fig, out)


def plot_comparison_bar(yolo: dict, baseline: dict, out: Path) -> None:
    metrics   = ["Precision", "Recall", "F1"]
    yolo_vals = [yolo["best_precision"], yolo["best_recall"], yolo["best_f1"]]
    base_vals = [baseline.get("precision", 0), baseline.get("recall", 0), baseline.get("f1", 0)]

    x     = np.arange(len(metrics))
    width = 0.35

    with plt.rc_context(_PLT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        b1 = ax.bar(x - width / 2, base_vals, width, label="Registry baseline",
                    color="#dc2626", alpha=0.82)
        b2 = ax.bar(x + width / 2, yolo_vals, width,
                    label=f"YOLOv8s (thresh={yolo['best_threshold']:.2f})",
                    color="#2563eb", alpha=0.82)
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.18); ax.set_ylabel("Score")
        ax.set_title("YOLOv8s vs Registry-Agreement Baseline")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        _save(fig, out)


def plot_confusion_matrix(yolo: dict, out: Path) -> None:
    # Layout: rows = actual, cols = predicted
    #  [[TP, FN],
    #   [FP, TN]]
    cm     = np.array([[yolo["tp"], yolo["fn"]],
                       [yolo["fp"], yolo["tn"]]])
    labels = [["TP", "FN"], ["FP", "TN"]]

    with plt.rc_context(_PLT_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Helipad", "Pred: None"])
        ax.set_yticklabels(["Actual: Helipad", "Actual: None"])
        ax.set_title(f"Confusion Matrix  (threshold={yolo['best_threshold']:.2f})")
        vmax = cm.max() or 1
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > vmax / 2 else "black"
                ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}",
                        ha="center", va="center", color=color, fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax)
        _save(fig, out)


def plot_yolo_metrics_bar(yolo: dict, out: Path) -> None:
    """Four-metric bar chart for YOLO at the best-F1 threshold."""
    metrics = ["Precision", "Recall", "F1", "Accuracy"]
    vals    = [yolo["best_precision"], yolo["best_recall"],
               yolo["best_f1"],        yolo["best_accuracy"]]
    colors  = ["#059669", "#d97706", "#7c3aed", "#2563eb"]

    with plt.rc_context(_PLT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(metrics, vals, color=colors, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, 1.2); ax.set_ylabel("Score")
        ax.set_title(f"YOLOv8s — All Metrics at Best Threshold ({yolo['best_threshold']:.2f})")
        ax.grid(True, alpha=0.3, axis="y")
        _save(fig, out)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8s helipad detector and compare to registry baseline."
    )
    parser.add_argument("--skip-train",  action="store_true",
                        help="Skip training; evaluate existing weights.")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--batch",   type=int,   default=16,
                        help="Training batch size (lower if RAM is limited).")
    parser.add_argument("--device",  type=str,   default=None,
                        help="'cpu', '0', '0,1' … (default: auto-detect GPU)")
    parser.add_argument("--baseline-dist", type=float, default=10.0,
                        help="FAA-OSM distance threshold for baseline TP in metres (default: 10)")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── annotation state check ────────────────────────────────────────────────
    if DECISIONS_CSV.exists():
        n_dec = len(pd.read_csv(DECISIONS_CSV))
        log.info("%d annotation decisions found in review_decisions.csv", n_dec)
        log.info("Ensure you clicked 'Apply all decisions' in annotate_dataset.py "
                 "BEFORE running this script — otherwise your corrections are NOT used.")

    # ── prerequisites ─────────────────────────────────────────────────────────
    if not DATASET_YAML.exists():
        log.error("dataset.yaml missing — run build_yolo_dataset.py first")
        sys.exit(1)
    chips = list(TEST_IMAGES_DIR.glob("*.jpg"))
    if not chips:
        log.error("No test chips in %s — run build_yolo_dataset.py first", TEST_IMAGES_DIR)
        sys.exit(1)

    # ── train or load ─────────────────────────────────────────────────────────
    if args.skip_train:
        if BEST_WEIGHTS.exists():
            weights = BEST_WEIGHTS
        elif PIPELINE_WEIGHTS.exists():
            weights = PIPELINE_WEIGHTS
        else:
            log.error("--skip-train but no weights found at %s or %s",
                      BEST_WEIGHTS, PIPELINE_WEIGHTS)
            sys.exit(1)
        log.info("Skipping training — using weights: %s", weights)
    else:
        import torch
        from ultralytics import YOLO

        device = args.device
        if device is None:
            device = "0" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            log.warning("No GPU detected — training on CPU. This will take several hours.")
            log.warning("Tip: run on Google Colab (free T4 GPU) to finish in ~30 min:")
            log.warning("  Upload notebooks/02_yolo_training.ipynb to Colab and run there.")

        n_train = len(list((_PROJ_ROOT / "data" / "yolo_dataset" / "images" / "train").glob("*.jpg")))
        n_val   = len(list((_PROJ_ROOT / "data" / "yolo_dataset" / "images" / "val").glob("*.jpg")))
        log.info("Dataset: %d train | %d val | %d test chips", n_train, n_val, len(chips))
        log.info("Starting training: epochs=%d  batch=%d  device=%s", args.epochs, args.batch, device)

        model = YOLO("yolov8s.pt")
        model.train(
            data=str(DATASET_YAML),
            epochs=args.epochs,
            imgsz=640,
            batch=args.batch,
            device=device,
            project=str(MODELS_DIR),
            name="helipad_run",
            exist_ok=True,
            patience=20,
            save=True,
            plots=True,
        )
        weights = BEST_WEIGHTS
        if not weights.exists():
            log.error("Training finished but best.pt not found at %s", weights)
            sys.exit(1)
        shutil.copy2(weights, PIPELINE_WEIGHTS)
        log.info("Best weights copied to %s", PIPELINE_WEIGHTS)

    # ── inference on test set ─────────────────────────────────────────────────
    chip_results = run_test_inference(weights)
    n_pos = sum(1 for r in chip_results if r["has_gt"])
    n_neg = len(chip_results) - n_pos
    log.info("Test set: %d positive chips | %d negative chips", n_pos, n_neg)

    # ── metrics ───────────────────────────────────────────────────────────────
    yolo_m     = compute_yolo_metrics(chip_results)
    baseline_m = compute_baseline(args.baseline_dist)

    # ── plots ─────────────────────────────────────────────────────────────────
    log.info("Generating plots in %s …", PLOTS_DIR)
    plot_pr_curve(yolo_m, baseline_m, PLOTS_DIR / "pr_curve.png")
    plot_f1_threshold(yolo_m, PLOTS_DIR / "f1_threshold.png")
    plot_yolo_metrics_bar(yolo_m, PLOTS_DIR / "yolo_metrics_bar.png")
    plot_confusion_matrix(yolo_m, PLOTS_DIR / "confusion_matrix.png")
    if baseline_m:
        plot_comparison_bar(yolo_m, baseline_m, PLOTS_DIR / "comparison_bar.png")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  EVALUATION SUMMARY")
    print("=" * 68)
    header = f"  {'Metric':<14}{'Baseline':>12}{'YOLOv8s':>12}"
    print(header)
    print("  " + "-" * 38)
    for label, bkey, ykey in [
        ("Precision",  "precision", "best_precision"),
        ("Recall",     "recall",    "best_recall"),
        ("F1",         "f1",        "best_f1"),
    ]:
        bv = f"{baseline_m[bkey]:.3f}" if baseline_m else "  —"
        yv = f"{yolo_m[ykey]:.3f}"
        print(f"  {label:<14}{bv:>12}{yv:>12}")
    print(f"  {'Accuracy':<14}{'  —':>12}{yolo_m['best_accuracy']:>12.3f}")
    print()
    print(f"  YOLO best confidence threshold : {yolo_m['best_threshold']:.3f}")
    print(f"  YOLO counts (TP/FP/FN/TN)      : "
          f"{yolo_m['tp']} / {yolo_m['fp']} / {yolo_m['fn']} / {yolo_m['tn']}")
    if baseline_m:
        print(f"  Baseline FAA-ID matches        : {baseline_m['n_matched']}")
        print(f"  Baseline OSM-only records      : {baseline_m['n_osm_only']}")
        print(f"  Baseline counts (TP/FP/FN)     : "
              f"{baseline_m['tp']} / {baseline_m['fp']} / {baseline_m['fn']}")
    print()
    print(f"  Training curves (loss/mAP)     : {RUN_DIR / 'results.png'}")
    print(f"  All plots saved in             : {PLOTS_DIR}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
