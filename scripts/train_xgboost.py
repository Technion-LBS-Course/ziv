"""Train XGBoost structured baseline and generate comparison plots.

Requires:
    data/faa_adip_enriched.csv   — FAA + ADIP features
    data/inspector_results.csv   — gt labels (run app Inspector tab first,
                                   or: python scripts/train_yolo.py --skip-train)

Outputs:
    models/xgboost/xgboost_baseline.pkl   trained model
    models/xgboost/metrics.json            P / R / F1 + majority baseline
    models/xgboost/feature_importance.png
    models/xgboost/comparison.png          Registry / XGBoost / YOLO side-by-side

Usage:
    python scripts/train_xgboost.py
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

from src.model import (
    METRICS_PATH,
    MODEL_PATH,
    RESULTS_CSV,
    XGB_DIR,
    FAA_CSV,
    FI_PLOT_PATH,
    CMP_PLOT_PATH,
    FEATURE_COLS,
    build_features,
    build_labels,
    evaluate_model,
    plot_feature_importance,
    plot_model_comparison,
    save_model,
    train_xgboost_baseline,
)


def load_yolo_metrics() -> dict | None:
    """Load YOLO final eval metrics if available."""
    path = _PROJ_ROOT / "models" / "plots" / "eval_results.json"
    if not path.exists():
        return None
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
        return {"precision": m["precision"], "recall": m["recall"], "f1": m["f1"]}
    except Exception:
        return None


def load_baseline_metrics() -> dict | None:
    """Load registry-agreement baseline metrics if available."""
    path = _PROJ_ROOT / "models" / "plots" / "eval_results.json"
    if not path.exists():
        return None
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
        b = m.get("baseline")
        return b if b else None
    except Exception:
        return None


def main() -> None:
    # ── Prerequisites ─────────────────────────────────────────────────────────
    for p in (FAA_CSV, RESULTS_CSV):
        if not p.exists():
            log.error("Missing: %s", p)
            if p == RESULTS_CSV:
                log.error(
                    "Run the Inspector tab in the Streamlit app first, or:\n"
                    "  python scripts/train_yolo.py --skip-train"
                )
            sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    faa_df     = pd.read_csv(FAA_CSV)
    results_df = pd.read_csv(RESULTS_CSV)

    log.info("FAA records: %d", len(faa_df))
    log.info("Inspector results: %d  (gt=1: %d, gt=0: %d)",
             len(results_df),
             (results_df["gt"] == 1).sum(),
             (results_df["gt"] == 0).sum())

    # ── Features + labels ─────────────────────────────────────────────────────
    X = build_features(faa_df)
    y = build_labels(faa_df, results_df)

    log.info("Feature matrix: %d rows × %d cols", len(X), len(X.columns))
    log.info("Label distribution: positive=%d, negative=%d",
             (y == 1).sum(), (y == 0).sum())

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Training XGBoost baseline …")
    model, metrics, X_test, y_test = train_xgboost_baseline(X, y)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  XGBOOST STRUCTURED BASELINE — EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Test split: {metrics['n_test']} records")
    print(f"  Precision : {metrics['precision']:.3f}")
    print(f"  Recall    : {metrics['recall']:.3f}")
    print(f"  F1        : {metrics['f1']:.3f}")
    print(f"  Majority-class F1 (floor): {metrics['majority_f1']:.3f}")
    print(f"  F1 improvement over majority: {metrics['f1'] - metrics['majority_f1']:+.3f}")
    print(f"\n  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    print("\n  Classification report:")
    for line in metrics["report"].splitlines():
        print("   ", line)

    yolo_m     = load_yolo_metrics()
    baseline_m = load_baseline_metrics()
    if yolo_m:
        print(f"\n  YOLOv8s F1 (visual, for reference): {yolo_m['f1']:.3f}")
        print(f"  XGBoost captures {100 * metrics['f1'] / yolo_m['f1']:.0f}% of YOLO F1 "
              "using structured data only")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_model(model)
    XGB_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Metrics saved to %s", METRICS_PATH)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_feature_importance(model, FEATURE_COLS, FI_PLOT_PATH)
    plot_model_comparison(metrics, None, yolo_m, CMP_PLOT_PATH)  # registry baseline omitted
    log.info("Plots written to %s", XGB_DIR)


if __name__ == "__main__":
    main()
