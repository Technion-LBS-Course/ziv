"""XGBoost structured baseline — predicts helipad visual presence from ADIP registry features.

Label (y):
    gt from data/inspector_results.csv — 1 = human-annotated as visually present (approved
    or bbox-adjusted), 0 = visually absent/invisible (disqualified).

Why not adip_status:
    743 / 747 records are "Operational", providing no classification signal.
    The gt label captures whether a helipad is VISUALLY CONFIRMABLE from 0.156 m/px NAIP
    imagery — the actual question the routing engine needs to answer.

Features:
    All derived from faa_adip_enriched.csv — no imagery used.
    This model is the non-visual baseline for comparison against YOLOv8s.
"""

import json
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

log = logging.getLogger(__name__)

_PROJ_ROOT    = Path(__file__).resolve().parents[1]
DATA_DIR      = _PROJ_ROOT / "data"
MODELS_DIR    = _PROJ_ROOT / "models"
XGB_DIR       = MODELS_DIR / "xgboost"
MODEL_PATH    = XGB_DIR / "xgboost_baseline.pkl"
METRICS_PATH  = XGB_DIR / "metrics.json"
FI_PLOT_PATH  = XGB_DIR / "feature_importance.png"
CMP_PLOT_PATH = XGB_DIR / "comparison.png"

FAA_CSV       = DATA_DIR / "faa_adip_enriched.csv"
RESULTS_CSV   = DATA_DIR / "inspector_results.csv"

_PLT_STYLE = {"figure.facecolor": "#1e1e2e", "axes.facecolor": "#1e1e2e",
              "text.color": "#cdd6f4", "axes.labelcolor": "#cdd6f4",
              "xtick.color": "#cdd6f4", "ytick.color": "#cdd6f4",
              "axes.edgecolor": "#45475a", "grid.color": "#313244"}


# ── Feature engineering ───────────────────────────────────────────────────────

_OWNERSHIP_MAP = {"PU": 0, "PR": 1, "MR": 2, "MN": 3}
_USE_MAP       = {"PU": 0, "PR": 1}
# FAA airspace analysis result — ordered by decreasing operational confidence
_AIRANAL_MAP   = {"NO OBJECTION": 0, "NOT ANALYZED": 1, "CONDITIONAL": 2, "OBJECTIONABLE": 3}

_REF_DATE = pd.Timestamp("2026-06-01")   # approximate ADIP data collection date

FEATURE_COLS = [
    # Registry / ownership
    "ownership_enc",
    "use_enc",
    "mil_flag",
    "privateuse",
    # Physical
    "elevation_ft",
    "elev_surveyed",
    # Operational equipment
    "has_wind",
    "has_notam",
    "has_icao",
    # Data freshness
    "last_info_days_ago",
    "position_age_days",
    "elevation_age_days",
    # Staleness flags (engineered)
    "data_stale",
    "high_inspection_age",
    # Airspace / regulatory
    "airanal_enc",
    "has_nasp_inspection",
    # Geography
    "state_enc",
]


def build_features(faa_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the structured feature matrix from ADIP enriched records.

    Args:
        faa_df: DataFrame loaded from faa_adip_enriched.csv.

    Returns:
        DataFrame with columns matching FEATURE_COLS, index aligned to faa_df.
    """
    df = faa_df.copy()

    # ── Ownership / use ──────────────────────────────────────────────────────
    df["ownership_enc"] = df["ownership_code"].map(_OWNERSHIP_MAP).fillna(1).astype(int)
    df["use_enc"]       = df["use_code"].map(_USE_MAP).fillna(1).astype(int)
    df["mil_flag"]      = (df["MIL_CODE"] == "MIL").astype(int)
    df["privateuse"]    = df["PRIVATEUSE"].fillna(1).astype(int)

    # ── Physical ─────────────────────────────────────────────────────────────
    elev = df["ELEVATION"].copy()
    state_median  = df.groupby("STATE")["ELEVATION"].transform("median")
    overall_median = elev.median()
    df["elevation_ft"]  = elev.fillna(state_median).fillna(overall_median)
    df["elev_surveyed"] = (df["elev_method"] == "SURVEYED").astype(int)

    # ── Operational equipment ─────────────────────────────────────────────────
    df["has_wind"]  = df["wind_indicator"].isin(["Y", "Y-L"]).astype(int)
    df["has_notam"] = (df["notam_service"] == "Y").astype(int)
    df["has_icao"]  = df["ICAO_ID"].notna().astype(int)

    # ── Data freshness ────────────────────────────────────────────────────────
    median_days = df["last_info_days_ago"].median()
    df["last_info_days_ago"] = df["last_info_days_ago"].fillna(median_days)

    def _age_days(col: str) -> pd.Series:
        dt    = pd.to_datetime(df[col], errors="coerce")
        delta = (_REF_DATE - dt).dt.days.clip(lower=0)
        return delta.fillna(delta.median())

    df["position_age_days"]  = _age_days("position_date")
    df["elevation_age_days"] = _age_days("elevation_date")

    # ── Staleness flags (engineered) ─────────────────────────────────────────
    df["data_stale"]          = (df["last_info_days_ago"] > 365).astype(int)
    df["high_inspection_age"] = (df["last_info_days_ago"] > 1095).astype(int)

    # ── Airspace / regulatory ─────────────────────────────────────────────────
    # AIRANAL: NO OBJECTION=0 … OBJECTIONABLE=3  (ordered by decreasing confidence)
    df["airanal_enc"] = df["AIRANAL"].map(_AIRANAL_MAP).fillna(1).astype(int)
    # NASP inspection (method '2') is the most rigorous inspection type
    df["has_nasp_inspection"] = (df["inspection_method"] == "2").astype(int)

    # ── Geography ─────────────────────────────────────────────────────────────
    state_codes = {"NY": 0, "NJ": 1, "PA": 2, "MA": 3, "CT": 4}
    df["state_enc"] = df["STATE"].map(state_codes).fillna(5).astype(int)

    return df[FEATURE_COLS].reset_index(drop=True)


def build_labels(faa_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.Series:
    """Join gt labels from inspector_results.csv onto the FAA dataframe.

    Args:
        faa_df: FAA records DataFrame (must have IDENT column).
        results_df: inspector_results.csv DataFrame (must have ident + gt columns).

    Returns:
        Series of gt labels aligned to faa_df rows; NaN rows are dropped upstream.
    """
    merged = faa_df[["IDENT"]].merge(
        results_df[["ident", "gt"]],
        left_on="IDENT", right_on="ident", how="left",
    )
    return merged["gt"].reset_index(drop=True)


# ── Training ──────────────────────────────────────────────────────────────────

def train_xgboost_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple:
    """Train XGBoost classifier on structured features; evaluate on held-out split.

    Args:
        X: Feature matrix from build_features().
        y: Binary labels (0/1) from build_labels().
        test_size: Fraction held out for evaluation (default 0.20).
        random_state: Reproducibility seed.

    Returns:
        (fitted_model, metrics_dict, X_test, y_test)
    """
    from xgboost import XGBClassifier

    # Drop rows where label is NaN
    mask = y.notna()
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    y = y.astype(int)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    log.info("XGBoost  P=%.3f  R=%.3f  F1=%.3f  (test n=%d)",
             metrics["precision"], metrics["recall"], metrics["f1"], len(y_test))
    return model, metrics, X_test, y_test


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a trained model on the test split.

    Args:
        model: Fitted classifier with predict() method.
        X_test: Test feature matrix.
        y_test: Ground truth labels.

    Returns:
        Dict with precision, recall, f1, majority_f1, confusion_matrix, report.
    """
    y_pred = model.predict(X_test)
    y_test = y_test.astype(int)

    majority_pred = np.ones(len(y_test), dtype=int)
    majority_f1   = f1_score(y_test, majority_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    return {
        "precision":    float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":       float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":           float(f1_score(y_test, y_pred, zero_division=0)),
        "majority_f1":  float(majority_f1),
        "tp": int(cm[1, 1]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tn": int(cm[0, 0]),
        "report":       classification_report(y_test, y_pred, zero_division=0),
        "n_test":       len(y_test),
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model, path: Path = MODEL_PATH) -> None:
    """Save fitted model to disk with pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("Model saved to %s", path)


def load_model(path: Path = MODEL_PATH):
    """Load a pickled model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list[str], out: Path) -> None:
    """Horizontal bar chart of XGBoost feature importance (gain)."""
    importance = model.get_booster().get_score(importance_type="gain")
    vals  = [importance.get(name, 0.0) for name in feature_names]
    pairs = sorted(zip(feature_names, vals), key=lambda x: x[1])

    with plt.rc_context(_PLT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh([p[0] for p in pairs], [p[1] for p in pairs], color="#2563eb", alpha=0.85)
        ax.set_xlabel("Gain")
        ax.set_title("XGBoost Feature Importance")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
    log.info("Feature importance plot saved to %s", out)


def plot_model_comparison(
    xgb_metrics: dict,
    baseline_metrics: dict | None,
    yolo_metrics: dict | None,
    out: Path,
) -> None:
    """Bar chart comparing Registry baseline / XGBoost / YOLO across P, R, F1."""
    labels  = ["Precision", "Recall", "F1"]
    series  = []

    if baseline_metrics:
        series.append(("Registry baseline",
                        [baseline_metrics.get("precision", 0),
                         baseline_metrics.get("recall", 0),
                         baseline_metrics.get("f1", 0)],
                        "#dc2626"))
    series.append(("XGBoost (structured)",
                   [xgb_metrics["precision"], xgb_metrics["recall"], xgb_metrics["f1"]],
                   "#f59e0b"))
    if yolo_metrics:
        series.append(("YOLOv8s (visual)",
                       [yolo_metrics.get("precision", 0),
                        yolo_metrics.get("recall", 0),
                        yolo_metrics.get("f1", 0)],
                       "#2563eb"))

    x     = np.arange(len(labels))
    width = 0.8 / len(series)

    with plt.rc_context(_PLT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, (name, vals, color) in enumerate(series):
            offset = (k - len(series) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("Score")
        ax.set_title("XGBoost (structured) vs YOLOv8s (visual)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
    log.info("Comparison plot saved to %s", out)
