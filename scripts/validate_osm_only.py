"""Validate OSM-only NE US helipads via NAIP cascade inference.

Finds OSM records with no FAA counterpart (neither FAA-ID nor proximity match),
fetches a NAIP chip for each, runs the YOLO cascade, and writes results to
data/osm_validated.csv.

Usage:
    python scripts/validate_osm_only.py              # full run
    python scripts/validate_osm_only.py --limit 20   # smoke test first N records
    python scripts/validate_osm_only.py --resume     # skip already-done rows
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FAA_CSV = _PROJ_ROOT / "data" / "faa_adip_enriched.csv"
OSM_CSV = _PROJ_ROOT / "data" / "osm_helipads_raw.csv"
OUTPUT_CSV = _PROJ_ROOT / "data" / "osm_validated.csv"
YOLO_MODEL_PATH = _PROJ_ROOT / "models" / "helipad_yolov8s.pt"
OSM_CHIPS_DIR = _PROJ_ROOT / "data" / "osm_chips"

PROX_THRESHOLD_M = 250.0

OUTPUT_COLS = [
    "osm_id", "name", "lat", "lon",
    "hie_visual_detected", "hie_confidence",
    "hie_det_lat", "hie_det_lon", "hie_offset_m",
    "naip_status",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N OSM-only records")
    p.add_argument("--resume", action="store_true",
                   help="Skip rows already present in output CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for path in (FAA_CSV, OSM_CSV, YOLO_MODEL_PATH):
        if not path.exists():
            log.error("Missing required file: %s", path)
            sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────────
    from src.data import load_faa_data
    from src.analysis import match_by_faa_id, match_by_proximity

    faa_df = load_faa_data(FAA_CSV)
    osm_df = pd.read_csv(OSM_CSV, low_memory=False)

    for df in (faa_df, osm_df):
        for col in ("lat", "lon"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    osm_df = osm_df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    osm_df["osm_id"] = osm_df["osm_id"].astype(str)

    log.info("FAA records: %d", len(faa_df))
    log.info("OSM records (with coords): %d", len(osm_df))

    # ── Find OSM-only records ──────────────────────────────────────────────────
    faa_id_matches = match_by_faa_id(faa_df, osm_df)
    prox_matches = match_by_proximity(
        faa_df, osm_df,
        threshold_m=PROX_THRESHOLD_M,
        exclude_faa_idx=set(faa_id_matches["faa_idx"]) if len(faa_id_matches) else set(),
    )

    matched_osm_idx = set(faa_id_matches["osm_idx"]) | set(prox_matches["osm_idx"])
    osm_only_df = osm_df.loc[~osm_df.index.isin(matched_osm_idx)].copy().reset_index(drop=True)

    log.info(
        "OSM-only records (no FAA match within %.0f m): %d  (%.0f%% of all OSM)",
        PROX_THRESHOLD_M,
        len(osm_only_df),
        100.0 * len(osm_only_df) / max(len(osm_df), 1),
    )

    # ── Resume: skip already-processed rows ───────────────────────────────────
    if args.resume and OUTPUT_CSV.exists():
        done_ids = set(pd.read_csv(OUTPUT_CSV, dtype={"osm_id": str})["osm_id"])
        before = len(osm_only_df)
        osm_only_df = osm_only_df[~osm_only_df["osm_id"].isin(done_ids)].reset_index(drop=True)
        log.info("Resume: skipping %d done, %d remaining", before - len(osm_only_df), len(osm_only_df))

    if args.limit:
        osm_only_df = osm_only_df.head(args.limit)
        log.info("Limit applied: processing first %d records", len(osm_only_df))

    if len(osm_only_df) == 0:
        log.info("Nothing to process.")
        return

    # ── Initialise output file and chip directory ─────────────────────────────
    fresh_file = not (args.resume and OUTPUT_CSV.exists())
    if fresh_file:
        pd.DataFrame(columns=OUTPUT_COLS).to_csv(OUTPUT_CSV, index=False)
    OSM_CHIPS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load YOLO model ────────────────────────────────────────────────────────
    from ultralytics import YOLO
    log.info("Loading YOLO model from %s …", YOLO_MODEL_PATH.name)
    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    from src.hie import (
        fetch_naip_chip,
        detect_helipad_cascade,
        bbox_px_to_latlon,
        compute_offset_m,
    )

    # ── Inference loop ─────────────────────────────────────────────────────────
    n_total = len(osm_only_df)
    n_ok = n_no_imagery = n_failed = n_detected = 0
    batch: list[dict] = []

    for i, row in enumerate(osm_only_df.itertuples(index=False)):
        osm_id = str(row.osm_id)
        name = (row.name if isinstance(row.name, str) else "") or ""
        lat = float(row.lat)
        lon = float(row.lon)

        if (i + 1) % 10 == 0 or i == 0:
            log.info(
                "[%d/%d]  osm_id=%s  %-30s  (%.4f, %.4f)",
                i + 1, n_total, osm_id, name[:30], lat, lon,
            )

        chip = fetch_naip_chip(lat, lon)

        if chip is not None:
            # Save chip to data/osm_chips/{safe_id}.jpg for app thumbnail display.
            import re as _re
            _safe_id = _re.sub(r"[^a-zA-Z0-9_-]", "_", osm_id)
            _chip_path = OSM_CHIPS_DIR / f"{_safe_id}.jpg"
            if not _chip_path.exists():
                try:
                    chip.save(str(_chip_path), format="JPEG", quality=85)
                except Exception as _e:
                    log.warning("Could not save chip for osm_id=%s: %s", osm_id, _e)

        if chip is None:
            naip_status = "no_imagery"
            n_no_imagery += 1
            record = {
                "osm_id": osm_id, "name": name, "lat": lat, "lon": lon,
                "hie_visual_detected": False, "hie_confidence": 0.0,
                "hie_det_lat": None, "hie_det_lon": None, "hie_offset_m": None,
                "naip_status": naip_status,
            }
        else:
            try:
                result = detect_helipad_cascade(chip, yolo_model)
                naip_status = "ok"
                n_ok += 1
            except Exception as exc:
                log.warning("Detection failed for osm_id=%s: %s", osm_id, exc)
                naip_status = "failed"
                n_failed += 1
                record = {
                    "osm_id": osm_id, "name": name, "lat": lat, "lon": lon,
                    "hie_visual_detected": False, "hie_confidence": 0.0,
                    "hie_det_lat": None, "hie_det_lon": None, "hie_offset_m": None,
                    "naip_status": naip_status,
                }
                batch.append(record)
                continue

            hie_det_lat = hie_det_lon = hie_offset_m = None
            if result["detected"] and result["bbox_px"]:
                hie_det_lat, hie_det_lon = bbox_px_to_latlon(result["bbox_px"], lat, lon)
                hie_offset_m = compute_offset_m(lat, lon, hie_det_lat, hie_det_lon)
                n_detected += 1

            record = {
                "osm_id": osm_id, "name": name, "lat": lat, "lon": lon,
                "hie_visual_detected": bool(result["detected"]),
                "hie_confidence": float(result["confidence"]),
                "hie_det_lat": hie_det_lat,
                "hie_det_lon": hie_det_lon,
                "hie_offset_m": hie_offset_m,
                "naip_status": naip_status,
            }

        batch.append(record)

        # Flush batch every 10 rows (resume-safe incremental writes)
        if len(batch) >= 10:
            pd.DataFrame(batch, columns=OUTPUT_COLS).to_csv(
                OUTPUT_CSV, mode="a", header=False, index=False
            )
            batch.clear()

        time.sleep(0.2)  # USDA APFO rate courtesy

    # Flush remaining rows
    if batch:
        pd.DataFrame(batch, columns=OUTPUT_COLS).to_csv(
            OUTPUT_CSV, mode="a", header=False, index=False
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  OSM-ONLY HELIPAD VALIDATION — SUMMARY")
    print("=" * 58)
    print(f"  Total OSM-only records processed : {n_total}")
    print(f"  NAIP imagery available           : {n_ok}")
    print(f"  No imagery (out of CONUS)        : {n_no_imagery}")
    print(f"  Fetch/inference failures         : {n_failed}")
    print(f"  Visually confirmed (HIE detected): {n_detected}")
    if n_ok > 0:
        print(f"  Detection rate (on valid chips)  : {100 * n_detected / n_ok:.1f}%")
    print(f"\n  Output : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
