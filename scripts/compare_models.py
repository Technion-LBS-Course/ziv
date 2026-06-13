"""Evaluate and compare YOLOv8s, YOLOv11s, YOLOv11m and RT-DETR-L on 747 test chips.

Reuses run_test_inference / compute_yolo_metrics from train_yolo.py so evaluation
methodology is identical across all models.

Outputs (all in models/plots_comparison/):
    comparison_metrics.json   per-model P / R / F1 at best threshold
    comparison_bar.png        side-by-side bar chart P / R / F1
    training_curves.png       mAP50 vs epoch for all 4 models

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --limit 50   # smoke test on 50 chips
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "YOLOv8s":   _PROJ_ROOT / "models" / "helipad_yolov8s.pt",
    "YOLOv11s":  _PROJ_ROOT / "models" / "helipad_run_yolo11s" / "weights" / "best.pt",
    "YOLOv11m":  _PROJ_ROOT / "models" / "helipad_run_yolo11m" / "weights" / "best.pt",
    "RT-DETR-L": _PROJ_ROOT / "models" / "helipad_run_rtdetr_l" / "weights" / "best.pt",
}

# Training results.csv for each model (for mAP50 curves)
RESULTS_CSV = {
    "YOLOv8s":   _PROJ_ROOT / "models" / "helipad_run"           / "results.csv",
    "YOLOv11s":  _PROJ_ROOT / "models" / "helipad_run_yolo11s"  / "results.csv",
    "YOLOv11m":  _PROJ_ROOT / "models" / "helipad_run_yolo11m"  / "results.csv",
    "RT-DETR-L": _PROJ_ROOT / "models" / "helipad_run_rtdetr_l" / "results.csv",
}

OUT_DIR = _PROJ_ROOT / "models" / "plots_comparison"

# Colour palette consistent with the app theme
_COLORS = {
    "YOLOv8s":   "#2563eb",  # blue
    "YOLOv11s":  "#16a34a",  # green
    "YOLOv11m":  "#9333ea",  # purple
    "RT-DETR-L": "#dc2626",  # red
}

_PLT_STYLE = {
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor":   "#1e1e2e",
    "text.color":       "#cdd6f4",
    "axes.labelcolor":  "#cdd6f4",
    "xtick.color":      "#cdd6f4",
    "ytick.color":      "#cdd6f4",
    "axes.edgecolor":   "#45475a",
    "grid.color":       "#313244",
    "legend.facecolor": "#313244",
    "legend.edgecolor": "#45475a",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_training_curve(path: Path) -> pd.DataFrame | None:
    """Read results.csv; return DataFrame with 'epoch' and 'map50' columns."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        map_col = next((c for c in df.columns if "mAP50" in c and "95" not in c), None)
        if map_col is None:
            return None
        df = df.rename(columns={"epoch": "epoch", map_col: "map50"})
        return df[["epoch", "map50"]].dropna()
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate_all(limit: int | None = None) -> tuple[dict, dict]:
    """Run test-chip inference for each model.

    Returns:
        (all_metrics, all_curves) where all_curves holds the full
        confidence-sweep arrays needed for the P/R/threshold plots.
    """
    from scripts.train_yolo import run_test_inference, compute_yolo_metrics

    all_metrics: dict = {}
    all_curves:  dict = {}
    for name, weights in MODELS.items():
        if not weights.exists():
            log.warning("Skipping %s — weights not found: %s", name, weights)
            continue
        log.info("\n═══ %s ═══", name)
        rows = run_test_inference(weights)
        if limit:
            rows = rows[:limit]
        m = compute_yolo_metrics(rows)
        all_metrics[name] = {
            "precision": m["best_precision"],
            "recall":    m["best_recall"],
            "f1":        m["best_f1"],
            "threshold": m["best_threshold"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        }
        all_curves[name] = {
            "thresholds": m["thresholds"].tolist(),
            "precisions": m["precisions"].tolist(),
            "recalls":    m["recalls"].tolist(),
            "f1s":        m["f1s"].tolist(),
        }
        log.info(
            "%s  P=%.3f  R=%.3f  F1=%.3f  (thresh=%.2f)",
            name, m["best_precision"], m["best_recall"], m["best_f1"], m["best_threshold"],
        )
    return all_metrics, all_curves


# ── Helper ─────────────────────────────────────────────────────────────────────

def _accuracy(m: dict) -> float:
    n = m["tp"] + m["fp"] + m["fn"] + m["tn"]
    return (m["tp"] + m["tn"]) / n if n else 0.0


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_radar_comparison(metrics: dict, out: Path) -> None:
    """Radar (spider) chart — Precision / Recall / F1 / Accuracy for all models.

    Uses a light background and a zoomed radial axis so differences are visible.
    The axis minimum is labelled explicitly so the chart is not misleading.
    """
    cats = ["Precision", "Recall", "F1", "Accuracy"]
    N    = len(cats)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Compute zoom range from actual data — padded 3% inside/outside
    all_vals = []
    for m in metrics.values():
        all_vals.extend([m["precision"], m["recall"], m["f1"], _accuracy(m)])
    r_min = max(0.0,  round(min(all_vals) - 0.04, 2))
    r_max = min(1.0,  round(max(all_vals) + 0.02, 2))
    tick_vals = np.round(np.linspace(r_min, r_max, 5), 2)

    # Light / bright style for this chart only
    _LIGHT = {
        "figure.facecolor": "#ffffff",
        "axes.facecolor":   "#f5f5f5",
        "text.color":       "#111111",
        "axes.labelcolor":  "#111111",
        "xtick.color":      "#111111",
        "ytick.color":      "#444444",
        "axes.edgecolor":   "#bbbbbb",
        "grid.color":       "#cccccc",
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#bbbbbb",
    }

    with plt.rc_context(_LIGHT):
        fig, ax = plt.subplots(figsize=(7, 7),
                               subplot_kw=dict(polar=True, facecolor="#f5f5f5"))
        fig.patch.set_facecolor("#ffffff")

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, fontsize=12, color="#111111", fontweight="bold")
        ax.set_ylim(r_min, r_max)
        ax.set_yticks(tick_vals)
        ax.set_yticklabels([f"{v:.2f}" for v in tick_vals], fontsize=8, color="#555555")
        ax.tick_params(colors="#111111")
        ax.spines["polar"].set_color("#bbbbbb")
        ax.grid(color="#cccccc", alpha=0.7)

        for name, m in metrics.items():
            vals  = [m["precision"], m["recall"], m["f1"], _accuracy(m)]
            vals += vals[:1]
            color = _COLORS.get(name, "#888")
            acc   = _accuracy(m)
            label = (f"{name}   P={m['precision']:.3f}  "
                     f"R={m['recall']:.3f}  F1={m['f1']:.3f}  Acc={acc:.3f}")
            ax.plot(angles, vals, color=color, lw=2.5, label=label)
            ax.fill(angles, vals, color=color, alpha=0.12)
            ax.scatter(angles[:-1], vals[:-1], color=color, s=50, zorder=5)

        ax.set_title("Model Comparison — 747 Test Chips\n(Precision / Recall / F1 / Accuracy)",
                     y=1.08, fontsize=11, color="#111111")

        # Legend below the chart — 2 columns so it stays compact
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc="upper center", bbox_to_anchor=(0.5, 0.13),
                   ncol=2, fontsize=8,
                   facecolor="#ffffff", edgecolor="#bbbbbb", labelcolor="#111111")

        # Scale note just above the legend
        fig.text(0.5, 0.07,
                 f"Note: inner ring = {r_min:.2f}  (axis zoomed to data range, not starting at 0)",
                 ha="center", fontsize=8, color="#666666", style="italic")

        fig.subplots_adjust(bottom=0.22)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved %s", out)


def plot_individual_model(name: str, m: dict, c: dict, out_dir: Path) -> None:
    """Four per-model plots from curve data — identical style for all models.

    Outputs (in out_dir):
        pr.png             Precision–Recall curve
        f1_conf.png        F1 vs confidence
        precision_conf.png Precision vs confidence
        recall_conf.png    Recall vs confidence
        confusion.png      Confusion matrix (TP/FP/FN/TN at best threshold)
    """
    t    = np.array(c["thresholds"])
    p    = np.array(c["precisions"])
    r    = np.array(c["recalls"])
    f1   = np.array(c["f1s"])
    t_opt = m["threshold"]
    color = _COLORS.get(name, "#888")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, fname):
        fig.savefig(out_dir / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)

    with plt.rc_context(_PLT_STYLE):
        # 1. PR curve
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(r, p, color=color, lw=2)
        idx = int(np.argmin(np.abs(t - t_opt)))
        ax.scatter([r[idx]], [p[idx]], color=color, s=80, zorder=6,
                   label=f"Best threshold {t_opt:.2f}  (F1={m['f1']:.3f})")
        ax.set_xlabel("Recall");  ax.set_ylabel("Precision")
        ax.set_xlim(0, 1);        ax.set_ylim(0, 1.05)
        ax.set_title(f"{name} — Precision–Recall")
        ax.legend(fontsize=8);    ax.grid(True, alpha=0.3)
        _save(fig, "pr.png")

        # 2. F1 vs confidence
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, f1, color=color, lw=2)
        ax.axvline(t_opt, color="#ffffff", lw=1.2, ls="--", alpha=0.7,
                   label=f"Best threshold {t_opt:.2f}")
        ax.scatter([t_opt], [m["f1"]], color=color, s=70, zorder=6,
                   marker="D")
        ax.set_xlabel("Confidence threshold");  ax.set_ylabel("F1")
        ax.set_xlim(0, 1);  ax.set_ylim(0, 1.05)
        ax.set_title(f"{name} — F1 vs Confidence")
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
        _save(fig, "f1_conf.png")

        # 3. Precision vs confidence
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, p, color=color, lw=2)
        ax.axvline(t_opt, color="#ffffff", lw=1.2, ls="--", alpha=0.7)
        ax.scatter([t_opt], [m["precision"]], color=color, s=70, zorder=6,
                   marker="D", label=f"At best F1 threshold: P={m['precision']:.3f}")
        ax.set_xlabel("Confidence threshold");  ax.set_ylabel("Precision")
        ax.set_xlim(0, 1);  ax.set_ylim(0, 1.05)
        ax.set_title(f"{name} — Precision vs Confidence")
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
        _save(fig, "precision_conf.png")

        # 4. Recall vs confidence
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, r, color=color, lw=2)
        ax.axvline(t_opt, color="#ffffff", lw=1.2, ls="--", alpha=0.7)
        ax.scatter([t_opt], [m["recall"]], color=color, s=70, zorder=6,
                   marker="D", label=f"At best F1 threshold: R={m['recall']:.3f}")
        ax.set_xlabel("Confidence threshold");  ax.set_ylabel("Recall")
        ax.set_xlim(0, 1);  ax.set_ylim(0, 1.05)
        ax.set_title(f"{name} — Recall vs Confidence")
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
        _save(fig, "recall_conf.png")

        # 5. Confusion matrix
        tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
        cm   = np.array([[tn, fp], [fn, tp]])
        lbls = [["TN", "FP"], ["FN", "TP"]]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{lbls[i][j]}\n{cm[i,j]}",
                        ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() * 0.5 else "#cdd6f4",
                        fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: None", "Pred: Helipad"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: None", "Actual: Helipad"])
        ax.set_title(f"{name} — Confusion Matrix  (thresh {t_opt:.2f})")
        plt.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        _save(fig, "confusion.png")

    log.info("Individual plots for %s → %s", name, out_dir)


def plot_confidence_curves(metrics: dict, curves: dict, out_dir: Path) -> None:
    """Three comparison plots using confidence-sweep data.

    1. PR curve          (Precision vs Recall)
    2. Precision vs Confidence threshold
    3. Recall    vs Confidence threshold

    Each plot overlays all models. Vertical markers show each model's
    individually optimal threshold; a dashed line marks the median.
    """
    thresholds_opt = [metrics[n]["threshold"] for n in curves]
    median_thresh  = float(np.median(thresholds_opt))
    spread         = max(thresholds_opt) - min(thresholds_opt)
    note           = (f"Median optimal threshold: {median_thresh:.2f}  "
                      f"(spread {spread:.2f}{'  ⚠ models differ — calibrate separately' if spread > 0.15 else ''})")

    def _base_fig(title, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        return fig, ax

    out_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_PLT_STYLE):
        # ── 1. PR curve ───────────────────────────────────────────────────────
        fig, ax = _base_fig("Precision–Recall Curve — all models",
                            "Recall", "Precision")
        for name, c in curves.items():
            p = np.array(c["precisions"])
            r = np.array(c["recalls"])
            color = _COLORS.get(name, "#888")
            f1_best = metrics[name]["f1"]
            ax.plot(r, p, color=color, lw=2, alpha=0.9,
                    label=f"{name}  (F1={f1_best:.3f})")
            # Mark best-F1 operating point
            t_opt = metrics[name]["threshold"]
            t_arr = np.array(c["thresholds"])
            idx   = int(np.argmin(np.abs(t_arr - t_opt)))
            ax.scatter(r[idx], p[idx], color=color, s=80, zorder=6)
        ax.legend(fontsize=8, loc="lower left")
        fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="#888")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(out_dir / "curve_pr.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        # ── 2. Precision vs Confidence ────────────────────────────────────────
        fig, ax = _base_fig("Precision vs Confidence Threshold",
                            "Confidence threshold", "Precision")
        ax.axvline(median_thresh, color="#ffffff", lw=1.2, ls="--",
                   alpha=0.6, label=f"Median opt. threshold ({median_thresh:.2f})")
        for name, c in curves.items():
            t = np.array(c["thresholds"])
            p = np.array(c["precisions"])
            color  = _COLORS.get(name, "#888")
            t_opt  = metrics[name]["threshold"]
            p_opt  = metrics[name]["precision"]
            ax.plot(t, p, color=color, lw=2, alpha=0.9, label=name)
            ax.scatter([t_opt], [p_opt], color=color, s=70, zorder=6,
                       marker="D")
            ax.annotate(f"{t_opt:.2f}", (t_opt, p_opt),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=color)
        ax.legend(fontsize=8)
        fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="#888")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(out_dir / "curve_precision_conf.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        # ── 3. Recall vs Confidence ───────────────────────────────────────────
        fig, ax = _base_fig("Recall vs Confidence Threshold",
                            "Confidence threshold", "Recall")
        ax.axvline(median_thresh, color="#ffffff", lw=1.2, ls="--",
                   alpha=0.6, label=f"Median opt. threshold ({median_thresh:.2f})")
        for name, c in curves.items():
            t = np.array(c["thresholds"])
            r = np.array(c["recalls"])
            color = _COLORS.get(name, "#888")
            t_opt = metrics[name]["threshold"]
            r_opt = metrics[name]["recall"]
            ax.plot(t, r, color=color, lw=2, alpha=0.9, label=name)
            ax.scatter([t_opt], [r_opt], color=color, s=70, zorder=6,
                       marker="D")
            ax.annotate(f"{t_opt:.2f}", (t_opt, r_opt),
                        textcoords="offset points", xytext=(4, -10),
                        fontsize=7, color=color)
        ax.legend(fontsize=8)
        fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="#888")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(out_dir / "curve_recall_conf.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    log.info("Saved 3 confidence-curve plots to %s", out_dir)
    log.info("Threshold note: %s", note)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N chips (smoke test)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip inference — regenerate plots from cached curves JSON")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT_DIR / "comparison_metrics.json"
    curves_path  = OUT_DIR / "comparison_curves.json"

    def _safe(name: str) -> str:
        return name.lower().replace("-", "").replace(" ", "_")

    if args.plots_only:
        if not metrics_path.exists() or not curves_path.exists():
            log.error("No cached data found. Run without --plots-only first.")
            sys.exit(1)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        curves  = json.loads(curves_path.read_text(encoding="utf-8"))
        plot_radar_comparison(metrics, OUT_DIR / "comparison_radar.png")
        plot_confidence_curves(metrics, curves, OUT_DIR)
        for name in metrics:
            if name in curves:
                plot_individual_model(name, metrics[name], curves[name],
                                      OUT_DIR / f"individual_{_safe(name)}")
        log.info("Done (plots only).")
        return

    # Full evaluation or load cache
    cached_ok = (
        metrics_path.exists() and curves_path.exists() and not args.limit
        and set(json.loads(metrics_path.read_text()).keys())
            == set(k for k, v in MODELS.items() if v.exists())
    )
    if cached_ok:
        log.info("Using cached data (delete JSON files to re-run)")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        curves  = json.loads(curves_path.read_text(encoding="utf-8"))
    else:
        metrics, curves = evaluate_all(args.limit)

    if not args.limit:
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        curves_path.write_text(json.dumps(curves,  indent=2), encoding="utf-8")
        log.info("Results saved to %s", OUT_DIR)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"  {'Model':<12}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}  {'Threshold':>10}")
    print("=" * 70)
    for name, m in metrics.items():
        print(f"  {name:<12}  {m['precision']:>9.3f}  {m['recall']:>9.3f}  "
              f"{m['f1']:>9.3f}  {m['threshold']:>10.3f}")
    print("=" * 70)

    plot_radar_comparison(metrics, OUT_DIR / "comparison_radar.png")
    plot_confidence_curves(metrics, curves, OUT_DIR)
    for name in metrics:
        if name in curves:
            plot_individual_model(name, metrics[name], curves[name],
                                  OUT_DIR / f"individual_{_safe(name)}")
    log.info("\nAll outputs written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
