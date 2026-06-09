"""Build YOLO helipad detection dataset from HelipadCAT + NAIP aerial imagery.

Imagery source: Microsoft Planetary Computer NAIP collection (CONUS + AK, HI).
  Resolution: 0.3 m/px (2021+ captures) or 0.6 m/px (older states).
  All chips cover NAIP_WINDOW_M × NAIP_WINDOW_M metres, resampled to IMG_PX × IMG_PX px.
  No ESRI or Google Maps tiles are used.

Pipeline (resumable with --from N):
  1  Download HelipadCAT annotated CSV (~6 000 FAA-sourced coordinates)
  2  Download national FAA ArcGIS (all ~5 653 US heliports, current snapshot)
  3  Staleness filter — remove HelipadCAT records not found in current FAA (> 100 m)
  4  Geographic dedup — remove NE US records (our held-out test set, 747 pads)
  5  Fetch NAIP chips for training-eligible records
     groundtruth=True  → positive examples  (images/train|val/pos/)
     groundtruth=False → hard-negative examples (images/train|val/hard_neg/)
     GSD logged per chip; chips with GSD > --max-gsd are skipped.
  6  Generate easy-negative chips (random non-helipad NAIP locations)
  7  Write YOLO label files; geographic train/val split; dataset.yaml
  8  Build HTML review galleries:
     review_train/ — 50 random training chips with bbox overlaid
     review_test/  — all 747 NE US chips with synthetic bbox for manual review

Usage:
  python scripts/build_yolo_dataset.py                     # full pipeline
  python scripts/build_yolo_dataset.py --from 5           # resume from step 5
  python scripts/build_yolo_dataset.py --limit 20         # smoke-test first 20 records
  python scripts/build_yolo_dataset.py --max-gsd 0.6      # skip 1 m/px legacy NAIP
  python scripts/build_yolo_dataset.py --test-tiles-only  # fetch test-set chips only

Outputs (all under data/yolo_dataset/):
  images/{train,val,test}/   NAIP chips (640×640 px JPEG)
  labels/{train,val,test}/   YOLO label .txt (empty = negative)
  review_train/              Annotated chips for spot-check
  review_test/               Test chips + annotation_guide.html
  dataset.yaml               YOLO config (nc=1, names=[helipad])
  build_log.csv              Per-chip status (source / split / gsd / bbox / status)
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import random
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
YOLO_DIR = DATA_DIR / "yolo_dataset"
LOG_PATH = YOLO_DIR / "build_log.csv"
YAML_PATH = YOLO_DIR / "dataset.yaml"

HELIPADCAT_CSV = DATA_DIR / "helipadcat_raw.csv"
FAA_NATIONAL_CSV = DATA_DIR / "faa_national.csv"
NE_US_CSV = (
    DATA_DIR / "faa_adip_enriched.csv"
    if (DATA_DIR / "faa_adip_enriched.csv").exists()
    else DATA_DIR / "faa_helipads_raw.csv"
)

# ── remote sources ─────────────────────────────────────────────────────────────
_HELIPADCAT_URL = (
    "https://raw.githubusercontent.com/jonasbtn/helipad_detection"
    "/master/data/Helipad_DataBase_annotated.csv"
)
_FAA_ARCGIS_URL = (
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ"
    "/arcgis/rest/services/US_Airport/FeatureServer/0/query"
)
# USDA NAIP ImageServer URLs are defined in the fetch section below

# ── geometry / imagery constants ──────────────────────────────────────────────
# NAIP chip: 100m × 100m window, resized to 640×640 px.
# Effective GSD = 100 / 640 = 0.156 m/px — midway between ESRI zoom 19 (0.228) and zoom 20 (0.114).
# HelipadCAT annotated at zoom 20 (~73m coverage at lat 40°N). Scale factor ~0.73 at lat 40°N.
# At NAIP 0.3 m/px native: 333 px → 1.9× upsample (good quality).
# At NAIP 0.6 m/px native: 167 px → 3.8× upsample (acceptable).
NAIP_WINDOW_M: float = 100.0  # geographic window size (metres, square)
IMG_PX: int = 640              # output image size — also HelipadCAT reference size

# HelipadCAT was annotated on Google Maps zoom-20 chips (640×640 px).
# The HelipadCAT coordinate is at image centre (320, 320).
# Our NAIP chip is always centred on the coordinate too (320, 320).
# Bbox scale = HelipadCAT physical coverage / NAIP window (both 640 px output).
# Physical coverage varies with latitude; computed per-record by _bbox_pixel_scale().
HELIPADCAT_ZOOM: int = 20
HELIPADCAT_IMG_PX: int = 640
HELIPADCAT_CENTER: int = HELIPADCAT_IMG_PX // 2   # 320

FAA_MATCH_M: float = 100.0    # staleness cutoff (HelipadCAT coord vs current FAA)
NE_EXCL_M: float = 100.0      # radius to exclude NE US test-set records
EARTH_R: float = 6_371_000.0

VAL_FRACTION: float = 0.15    # fraction of training-eligible assigned to val
EASY_NEG_RATIO: float = 0.40  # easy negatives as fraction of positives
NEG_MIN_M: float = 500.0      # minimum offset for random negative point
NEG_MAX_M: float = 5_000.0    # maximum offset

NAIP_SLEEP: float = 0.10      # seconds between Planetary Computer requests
FAA_PAGE_SIZE: int = 2_000
# GSD filter: skip NAIP chips coarser than this (None = accept all).
# 0.3 m = ultra-high (2021+ states);  0.6 m = standard;  1.0 m = legacy.
# At 1.0 m/px the 150 m window is only 150×150 native px → 4.3× upsample → blurry.
MAX_GSD_M: Optional[float] = None   # overridden by --max-gsd CLI flag

# Synthetic bbox half-size for test-set annotation (hospital FATO, ≈9 m radius).
# At 100 m / 640 px = 0.156 m/px: 9 m → 58 px.
_DEFAULT_BBOX_HALF: int = 58

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "SkyRoute-Research/1.0 (Technion course project)"})

# ── DMS coordinate parser (same as fetch_ny_data.py) ──────────────────────────
_DMS_RE = re.compile(r"^(\d{1,3})-(\d{2})-([\d.]+)([NSEWnsew])$")


def _dms_to_decimal(dms: str) -> Optional[float]:
    """Parse FAA DMS string (e.g. '40-46-36.0000N') to decimal degrees."""
    if not dms or not isinstance(dms, str):
        return None
    m = _DMS_RE.match(dms.strip())
    if not m:
        return None
    deg, mins, secs, hem = m.groups()
    val = float(deg) + float(mins) / 60 + float(secs) / 3600
    if hem.upper() in ("S", "W"):
        val = -val
    return val


# ── haversine helpers ──────────────────────────────────────────────────────────

def _min_distance_to_set(
    q_lats: np.ndarray, q_lons: np.ndarray,
    r_lats: np.ndarray, r_lons: np.ndarray,
    chunk: int = 500,
) -> np.ndarray:
    """For each query point, return its minimum haversine distance to any ref point.

    Args:
        q_lats, q_lons: (N,) query coordinates in decimal degrees.
        r_lats, r_lons: (M,) reference coordinates in decimal degrees.
        chunk: Process N points per batch to bound memory usage.

    Returns:
        (N,) float32 array of distances in metres.
    """
    n = len(q_lats)
    min_dists = np.full(n, np.inf, dtype=np.float32)
    r_φ = np.radians(r_lats).astype(np.float32)
    r_λ = np.radians(r_lons).astype(np.float32)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        φ1 = np.radians(q_lats[start:end]).astype(np.float32)[:, None]
        λ1 = np.radians(q_lons[start:end]).astype(np.float32)[:, None]
        φ2 = r_φ[None, :]
        λ2 = r_λ[None, :]
        dφ = φ2 - φ1
        dλ = λ2 - λ1
        a = np.sin(dφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ / 2) ** 2
        d = EARTH_R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        min_dists[start:end] = d.min(axis=1)

    return min_dists


def _offset_latlon(lat: float, lon: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    """Compute a point at bearing_deg / dist_m from (lat, lon)."""
    d = dist_m / EARTH_R
    b = math.radians(bearing_deg)
    φ1 = math.radians(lat)
    λ1 = math.radians(lon)
    φ2 = math.asin(math.sin(φ1) * math.cos(d) + math.cos(φ1) * math.sin(d) * math.cos(b))
    λ2 = λ1 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(φ1),
        math.cos(d) - math.sin(φ1) * math.sin(φ2),
    )
    return math.degrees(φ2), math.degrees(λ2)


# ── NAIP imagery via USDA APFO ArcGIS ImageServer ────────────────────────────
# No extra dependencies — uses the existing requests session.
# USDA exportImage returns a ready-made JPEG chip for any lat/lon bbox.
# CONUS:  https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer
# Alaska: https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_Alaska_PRIME/ImageServer

class NoImageryError(Exception):
    """Raised when USDA returns no imagery for a coordinate."""


_USDA_CONUS_URL = (
    "https://gis.apfo.usda.gov/arcgis/rest/services"
    "/NAIP/USDA_CONUS_PRIME/ImageServer/exportImage"
)
# NAIP is a CONUS-only programme (National Agriculture Imagery Program).
# Alaska and Hawaii are not covered by USDA_CONUS_PRIME.
# HelipadCAT records outside CONUS (~5% of dataset) will be marked no_imagery.


def _naip_export_url(lat: float, lon: float) -> Optional[str]:
    """Return the USDA NAIP ImageServer URL for a coordinate, or None if outside CONUS.

    NAIP covers the continental US (roughly 24–49.5°N, 66–125°W).
    Records outside this box are gracefully skipped as no_imagery.
    """
    if 24.0 <= lat <= 49.5 and -125.0 <= lon <= -66.0:
        return _USDA_CONUS_URL
    return None


def _bbox_pixel_scale(lat: float) -> float:
    """Compute the HelipadCAT-to-NAIP-chip bbox scale factor for a given latitude.

    HelipadCAT annotated bboxes on Google Maps zoom-20 chips (640×640 px).
    The physical coverage of those chips varies with latitude (Web Mercator
    stretches north-south).  Our NAIP chip always covers NAIP_WINDOW_M × NAIP_WINDOW_M
    metres and outputs IMG_PX × IMG_PX pixels.

    Args:
        lat: Latitude in decimal degrees.

    Returns:
        Multiplicative scale: apply to (helipadcat_corner − 320) before placing
        into the 640×640 NAIP output image.
    """
    # m/px at HelipadCAT zoom 20 for this latitude
    hcat_m_per_px = 156543.03392 * math.cos(math.radians(lat)) / (2 ** HELIPADCAT_ZOOM)
    hcat_coverage_m = HELIPADCAT_IMG_PX * hcat_m_per_px
    return hcat_coverage_m / NAIP_WINDOW_M


def fetch_naip_chip(lat: float, lon: float) -> tuple[Image.Image, int, int, float]:
    """Fetch a NAIP chip centred on (lat, lon) via USDA APFO ImageServer.

    Issues a single HTTP GET to the USDA exportImage endpoint — no rasterio,
    no STAC, no authentication.  The server handles year selection and
    resampling; the response is a ready-made JPEG at the requested size.

    Coverage: CONUS only (24–49.5°N, 66–125°W).
    Records outside this box raise NoImageryError and are logged as no_imagery.

    Args:
        lat, lon: Centre coordinate in decimal degrees (EPSG:4326).

    Returns:
        (image, cx, cy, gsd_m): IMG_PX × IMG_PX RGB PIL Image, centre pixel
        (always IMG_PX//2, IMG_PX//2), and effective GSD (NAIP_WINDOW_M / IMG_PX).

    Raises:
        NoImageryError: Location outside USDA coverage, or server returns
        no-data / error response.
        requests.HTTPError: On non-200 response.
    """
    url = _naip_export_url(lat, lon)
    if url is None:
        raise NoImageryError(
            f"No USDA NAIP service for lat={lat:.5f} lon={lon:.5f} "
            f"(outside CONUS/Alaska coverage)"
        )

    dlat = NAIP_WINDOW_M / 2 / 111_320
    dlon = NAIP_WINDOW_M / 2 / (111_320 * math.cos(math.radians(lat)))

    params = {
        "bbox":       f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}",
        "bboxSR":     "4326",
        "size":       f"{IMG_PX},{IMG_PX}",
        "imageSR":    "4326",
        "format":     "jpeg",
        "pixelType":  "U8",
        "f":          "image",
    }

    # Retry up to 3 times with exponential back-off for transient network errors
    # (USDA server occasionally resets connections under load).
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(3):
        try:
            r = _SESSION.get(url, params=params, timeout=30)
            r.raise_for_status()
            break
        except Exception as exc:
            last_exc = exc
            wait = 10 * (2 ** attempt)   # 10 s, 20 s, 40 s
            log.warning("  USDA request failed (attempt %d/3), retrying in %ds: %s",
                        attempt + 1, wait, exc)
            time.sleep(wait)
    else:
        raise last_exc   # all 3 attempts exhausted

    # USDA returns XML/JSON on service errors instead of image data
    ct = r.headers.get("content-type", "")
    if not ct.startswith("image/"):
        raise NoImageryError(
            f"USDA returned non-image response ({ct}) at lat={lat:.5f} lon={lon:.5f}"
        )

    img = Image.open(BytesIO(r.content)).convert("RGB")

    # Reject blank chips (uniform colour = no-data area, e.g. ocean tile)
    arr = np.asarray(img, dtype=np.float32)
    if arr.std() < 5.0:
        raise NoImageryError(
            f"USDA returned blank chip at lat={lat:.5f} lon={lon:.5f}"
        )

    time.sleep(NAIP_SLEEP)
    gsd_m = NAIP_WINDOW_M / IMG_PX
    cx = cy = IMG_PX // 2
    return img, cx, cy, gsd_m


def _transform_bbox(
    min_x: int, min_y: int, max_x: int, max_y: int,
    cx: int, cy: int,
    scale: float,
) -> tuple[int, int, int, int]:
    """Convert HelipadCAT zoom-20 bbox into NAIP chip (IMG_PX × IMG_PX) space.

    HelipadCAT annotated bboxes on a 640×640 Google Maps image at zoom 20,
    with the coordinate at pixel (320, 320).  Our NAIP chip always centres
    the coordinate at (IMG_PX//2, IMG_PX//2) = (320, 320) as well.

    The scale factor converts HelipadCAT pixel offsets (physical distance /
    HelipadCAT m_per_px) to NAIP chip pixel offsets (same physical distance /
    NAIP m_per_px).  Computed per-record by _bbox_pixel_scale(lat).

    Args:
        min_x, min_y, max_x, max_y: Bbox corners in HelipadCAT 640×640 space.
        cx, cy: Pixel location of the coordinate in our chip (always 320, 320).
        scale: Per-latitude scale factor from _bbox_pixel_scale().

    Returns:
        Transformed bbox corners clamped to [0, IMG_PX].
    """
    def _clamp(v: float) -> int:
        return max(0, min(IMG_PX, int(round(v))))

    x1 = _clamp(cx + (min_x - HELIPADCAT_CENTER) * scale)
    y1 = _clamp(cy + (min_y - HELIPADCAT_CENTER) * scale)
    x2 = _clamp(cx + (max_x - HELIPADCAT_CENTER) * scale)
    y2 = _clamp(cy + (max_y - HELIPADCAT_CENTER) * scale)
    return x1, y1, x2, y2


def _bbox_to_yolo(x1: int, y1: int, x2: int, y2: int, img_px: int = IMG_PX) -> str:
    """Format a bbox as a YOLO label line (class 0, normalised xywh).

    Args:
        x1, y1, x2, y2: Bbox corners in pixel space.
        img_px: Image width/height in pixels.

    Returns:
        YOLO label string: '0 cx_n cy_n w_n h_n' (class index = 0 = helipad).
    """
    cx_n = ((x1 + x2) / 2) / img_px
    cy_n = ((y1 + y2) / 2) / img_px
    w_n = (x2 - x1) / img_px
    h_n = (y2 - y1) / img_px
    return f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"


# ── Step 1 — download HelipadCAT CSV ──────────────────────────────────────────

def step1_download_helipadcat() -> pd.DataFrame:
    """Download HelipadCAT annotated CSV and return as DataFrame.

    Returns:
        DataFrame with columns: latitude, longitude, groundtruth, category,
        minX, minY, maxX, maxY.

    Raises:
        RuntimeError: On download failure.
    """
    if HELIPADCAT_CSV.exists():
        log.info("HelipadCAT CSV already on disk — loading %s", HELIPADCAT_CSV.name)
        return pd.read_csv(HELIPADCAT_CSV)

    log.info("Downloading HelipadCAT CSV from GitHub ...")
    r = _SESSION.get(_HELIPADCAT_URL, timeout=60)
    r.raise_for_status()
    HELIPADCAT_CSV.parent.mkdir(parents=True, exist_ok=True)
    HELIPADCAT_CSV.write_bytes(r.content)
    df = pd.read_csv(HELIPADCAT_CSV)
    log.info("HelipadCAT: %d records", len(df))
    return df


# ── Step 2 — download national FAA ────────────────────────────────────────────

def step2_download_faa_national() -> pd.DataFrame:
    """Download all US heliport records from FAA ADDS-ArcGIS.

    Paginates through the full national dataset (TYPE_CODE='HP', no state filter).
    Results are cached to data/faa_national.csv.

    Returns:
        DataFrame with lat, lon, IDENT columns.

    Raises:
        RuntimeError: On ArcGIS API failure.
    """
    if FAA_NATIONAL_CSV.exists():
        log.info("National FAA CSV already on disk — loading %s", FAA_NATIONAL_CSV.name)
        return pd.read_csv(FAA_NATIONAL_CSV)

    log.info("Downloading national FAA heliports from ArcGIS (all states) ...")
    all_rows: list[dict] = []
    offset = 0

    while True:
        params = {
            "where":             "TYPE_CODE='HP'",
            "outFields":         "IDENT,NAME,STATE,LATITUDE,LONGITUDE",
            "returnGeometry":    "false",
            "f":                 "json",
            "resultRecordCount": FAA_PAGE_SIZE,
            "resultOffset":      offset,
        }
        r = _SESSION.get(_FAA_ARCGIS_URL, params=params, timeout=30)
        r.raise_for_status()
        body = r.json()

        if "error" in body:
            raise RuntimeError(f"ArcGIS error: {body['error']}")

        features = body.get("features", [])
        if not features:
            break

        for feat in features:
            all_rows.append(feat.get("attributes", {}))

        log.info("  offset=%d, page=%d, total=%d", offset, len(features), len(all_rows))

        if body.get("exceededTransferLimit"):
            offset += FAA_PAGE_SIZE
            time.sleep(0.3)
        else:
            break

    if not all_rows:
        raise RuntimeError("ArcGIS returned 0 records for national FAA query")

    df = pd.DataFrame(all_rows)
    df["lat"] = df["LATITUDE"].apply(_dms_to_decimal)
    df["lon"] = df["LONGITUDE"].apply(_dms_to_decimal)
    df = df.dropna(subset=["lat", "lon"])

    FAA_NATIONAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FAA_NATIONAL_CSV, index=False)
    log.info("National FAA: %d records → %s", len(df), FAA_NATIONAL_CSV.name)
    return df


# ── Step 3 — staleness filter ─────────────────────────────────────────────────

def step3_staleness_filter(hcat_df: pd.DataFrame, faa_df: pd.DataFrame) -> pd.DataFrame:
    """Remove HelipadCAT records with no matching current FAA record within FAA_MATCH_M.

    A helipad absent from the current FAA ArcGIS snapshot was likely removed from
    the active registry — do not use as a training positive.

    Args:
        hcat_df: HelipadCAT DataFrame (must have latitude, longitude).
        faa_df: National FAA DataFrame (must have lat, lon).

    Returns:
        Filtered DataFrame with a new column faa_dist_m.
    """
    hcat_valid = hcat_df.dropna(subset=["latitude", "longitude"]).copy()
    faa_valid = faa_df.dropna(subset=["lat", "lon"])

    log.info("Staleness check: %d HelipadCAT vs %d FAA records ...", len(hcat_valid), len(faa_valid))

    dists = _min_distance_to_set(
        hcat_valid["latitude"].values, hcat_valid["longitude"].values,
        faa_valid["lat"].values, faa_valid["lon"].values,
    )
    hcat_valid["faa_dist_m"] = dists

    before = len(hcat_valid)
    hcat_valid = hcat_valid[hcat_valid["faa_dist_m"] <= FAA_MATCH_M].copy()
    after = len(hcat_valid)
    log.info("Staleness filter: removed %d records (%.1f%%) — no FAA match within %.0f m",
             before - after, (before - after) / before * 100, FAA_MATCH_M)
    return hcat_valid


# ── Step 4 — geographic deduplication ─────────────────────────────────────────

def step4_geo_dedup(filtered_df: pd.DataFrame, ne_us_df: pd.DataFrame) -> pd.DataFrame:
    """Remove records that fall within NE_EXCL_M of our NE US test set.

    These records must never enter the training or validation split — they are
    reserved as the held-out test set.

    Args:
        filtered_df: Post-staleness DataFrame (latitude, longitude).
        ne_us_df: NE US FAA DataFrame (lat, lon — our 747 test records).

    Returns:
        Training-eligible DataFrame with a new column ne_us_dist_m.
    """
    ne_valid = ne_us_df.dropna(subset=["lat", "lon"])
    log.info("NE US dedup: checking %d records against %d test-set pads ...",
             len(filtered_df), len(ne_valid))

    dists = _min_distance_to_set(
        filtered_df["latitude"].values, filtered_df["longitude"].values,
        ne_valid["lat"].values, ne_valid["lon"].values,
    )
    filtered_df = filtered_df.copy()
    filtered_df["ne_us_dist_m"] = dists

    before = len(filtered_df)
    eligible = filtered_df[filtered_df["ne_us_dist_m"] > NE_EXCL_M].copy()
    after = len(eligible)
    log.info("Geographic dedup: removed %d NE US overlap records; %d training-eligible remain",
             before - after, after)
    return eligible


# ── Step 5 — fetch tiles for training records ─────────────────────────────────

def step5_fetch_tiles(
    eligible_df: pd.DataFrame,
    limit: Optional[int] = None,
) -> list[dict]:
    """Fetch NAIP chips for all training-eligible HelipadCAT records.

    Skips records whose chip image already exists on disk (safe to resume).

    Args:
        eligible_df: Post-dedup DataFrame.
        limit: If set, process only the first N records (smoke-test mode).

    Returns:
        List of tile record dicts with keys:
        helipadcat_id, lat, lon, groundtruth, category,
        bbox_x1, bbox_y1, bbox_x2, bbox_y2, tile_path, status.
    """
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("images/pos_train", "images/pos_val",
                "images/hard_neg_train", "images/hard_neg_val"):
        (YOLO_DIR / sub).mkdir(parents=True, exist_ok=True)

    records = eligible_df.head(limit) if limit else eligible_df
    tile_records: list[dict] = []

    for i, (_, row) in enumerate(records.iterrows()):
        hcat_id = int(row.get("Helipad_number", i))
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        gt = bool(row.get("groundtruth", False))
        try:
            category = int(row.get("category", -1))
        except (ValueError, TypeError):
            category = -1
        min_x = int(row.get("minX", -1))
        min_y = int(row.get("minY", -1))
        max_x = int(row.get("maxX", -1))
        max_y = int(row.get("maxY", -1))
        has_bbox = min_x != -1

        tile_type = "pos" if gt else "hard_neg"
        fname = f"hcat_{hcat_id:05d}.jpg"
        # Initially assign to train; step 7 will reassign val fraction
        tile_path = YOLO_DIR / "images" / f"{tile_type}_train" / fname

        rec: dict = {
            "helipadcat_id": hcat_id,
            "source": "helipadcat",
            "lat": lat,
            "lon": lon,
            "groundtruth": gt,
            "category": category,
            "faa_dist_m": float(row.get("faa_dist_m", -1)),
            "ne_us_dist_m": float(row.get("ne_us_dist_m", -1)),
            "tile_type": tile_type,
            "split": "train",
            "gsd_m": -1.0,
            "tile_path": str(tile_path.relative_to(DATA_DIR)),
            "bbox_x1": -1, "bbox_y1": -1, "bbox_x2": -1, "bbox_y2": -1,
            "status": "pending",
        }

        if tile_path.exists():
            rec["status"] = "skipped_exists"
            tile_records.append(rec)
            continue

        try:
            img, cx, cy, gsd_m = fetch_naip_chip(lat, lon)
            rec["gsd_m"] = gsd_m
        except NoImageryError:
            rec["status"] = "no_imagery"
            tile_records.append(rec)
            continue
        except Exception as exc:
            log.warning("  [%d/%d] FAILED lat=%.5f lon=%.5f: %s",
                        i + 1, len(records), lat, lon, exc)
            rec["status"] = "failed"
            tile_records.append(rec)
            continue

        if has_bbox:
            scale = _bbox_pixel_scale(lat)
            x1, y1, x2, y2 = _transform_bbox(min_x, min_y, max_x, max_y, cx, cy, scale)
            rec.update(bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2)

        tile_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(tile_path), quality=90)
        rec["status"] = "saved"

        if (i + 1) % 50 == 0:
            log.info("  [%d/%d] %s  lat=%.5f lon=%.5f  type=%s",
                     i + 1, len(records), fname, lat, lon, tile_type)

        tile_records.append(rec)

    n_saved = sum(1 for r in tile_records if r["status"] == "saved")
    n_skip = sum(1 for r in tile_records if r["status"] == "skipped_exists")
    n_fail = sum(1 for r in tile_records if r["status"] == "failed")
    n_no_img = sum(1 for r in tile_records if r["status"] == "no_imagery")
    log.info(
        "Tile fetch done: saved=%d  skipped=%d  no_imagery=%d  failed=%d",
        n_saved, n_skip, n_no_img, n_fail,
    )
    if n_no_img > 0:
        pct = n_no_img / max(1, n_saved + n_no_img) * 100
        log.info("  %.1f%% of records had no NAIP coverage — excluded from dataset", pct)

    # Log GSD distribution for saved chips
    gsds = [r["gsd_m"] for r in tile_records if r.get("gsd_m", -1) > 0]
    if gsds:
        from collections import Counter
        gsd_counts = Counter(round(g, 1) for g in gsds)
        log.info("  GSD distribution: %s", dict(sorted(gsd_counts.items())))

    return tile_records


# ── Step 6 — generate easy negatives ─────────────────────────────────────────

def step6_easy_negatives(
    tile_records: list[dict],
    all_lats: np.ndarray,
    all_lons: np.ndarray,
) -> list[dict]:
    """Fetch NAIP chips for random non-helipad locations.

    For each easy-negative, pick a random positive chip's coordinate, sample a
    random bearing and distance, verify the point is > 100 m from any known
    helipad, then fetch the NAIP chip.

    Args:
        tile_records: Existing tile records (positives used as starting points).
        all_lats, all_lons: All known helipad coordinates (HelipadCAT + NE US).

    Returns:
        List of easy-negative tile record dicts.
    """
    pos_records = [r for r in tile_records if r["tile_type"] == "pos" and r["status"] == "saved"]
    if not pos_records:
        # Fall back to all saved records as seed points (e.g. --limit run with only hard-negs)
        pos_records = [r for r in tile_records if r["status"] == "saved"]
    if not pos_records:
        log.warning("No saved chips to seed easy negatives from — skipping step 6")
        return []
    n_easy = max(1, int(len(pos_records) * EASY_NEG_RATIO))
    log.info("Generating %d easy-negative chips ...", n_easy)

    (YOLO_DIR / "images" / "easy_neg_train").mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "images" / "easy_neg_val").mkdir(parents=True, exist_ok=True)

    neg_records: list[dict] = []
    attempts = 0
    max_attempts = n_easy * 20

    while len(neg_records) < n_easy and attempts < max_attempts:
        attempts += 1
        seed = random.choice(pos_records)
        bearing = random.uniform(0, 360)
        dist = random.uniform(NEG_MIN_M, NEG_MAX_M)
        nlat, nlon = _offset_latlon(seed["lat"], seed["lon"], bearing, dist)

        if not (-90 <= nlat <= 90 and -180 <= nlon <= 180):
            continue

        # Check that this point is far enough from all known helipads
        d = _min_distance_to_set(
            np.array([nlat], dtype=np.float32),
            np.array([nlon], dtype=np.float32),
            all_lats, all_lons,
        )
        if d[0] < FAA_MATCH_M:
            continue

        neg_id = len(neg_records)
        fname = f"neg_{neg_id:05d}.jpg"
        tile_path = YOLO_DIR / "images" / "easy_neg_train" / fname

        if tile_path.exists():
            neg_records.append({
                "helipadcat_id": -1, "source": "easy_neg",
                "lat": nlat, "lon": nlon, "groundtruth": False,
                "category": -1, "faa_dist_m": -1, "ne_us_dist_m": -1,
                "tile_type": "easy_neg", "split": "train",
                "tile_path": str(tile_path.relative_to(DATA_DIR)),
                "bbox_x1": -1, "bbox_y1": -1, "bbox_x2": -1, "bbox_y2": -1,
                "status": "skipped_exists",
            })
            continue

        try:
            img, _, _, _ = fetch_naip_chip(nlat, nlon)
        except NoImageryError:
            continue  # point has no NAIP coverage — resample
        except Exception as exc:
            log.warning("  easy_neg %d: fetch failed: %s", neg_id, exc)
            continue

        img.save(str(tile_path), quality=90)
        neg_records.append({
            "helipadcat_id": -1, "source": "easy_neg",
            "lat": nlat, "lon": nlon, "groundtruth": False,
            "category": -1, "faa_dist_m": float(d[0]), "ne_us_dist_m": -1,
            "tile_type": "easy_neg", "split": "train",
            "tile_path": str(tile_path.relative_to(DATA_DIR)),
            "bbox_x1": -1, "bbox_y1": -1, "bbox_x2": -1, "bbox_y2": -1,
            "status": "saved",
        })

        if len(neg_records) % 50 == 0:
            log.info("  easy_neg %d/%d", len(neg_records), n_easy)

    log.info("Easy negatives: %d generated (%d attempts)", len(neg_records), attempts)
    return neg_records


# ── Step 7 — YOLO labels + split + dataset.yaml ───────────────────────────────

def step7_write_labels_and_split(all_records: list[dict]) -> None:
    """Write YOLO label files, assign val split, and generate dataset.yaml.

    For positive tiles: writes '0 cx_n cy_n w_n h_n' if bbox is available,
    otherwise an empty file (YOLO interprets empty label as background).
    For all negative tiles: empty label file (no objects).

    Args:
        all_records: Combined list of tile records from steps 5 and 6.
    """
    for split in ("train", "val", "test"):
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Assign val fraction from training-eligible records
    rng = random.Random(42)
    saved = [r for r in all_records if r["status"] in ("saved", "skipped_exists")]
    rng.shuffle(saved)
    n_val = int(len(saved) * VAL_FRACTION)
    for rec in saved[:n_val]:
        rec["split"] = "val"

    # Write labels and symlink/copy images into final split directories
    for rec in saved:
        split = rec["split"]
        src_path = DATA_DIR / rec["tile_path"]
        if not src_path.exists():
            continue

        stem = src_path.stem
        dst_img = YOLO_DIR / "images" / split / src_path.name
        dst_lbl = YOLO_DIR / "labels" / split / (stem + ".txt")

        # Move tile to split directory (rename is fast on same filesystem)
        if not dst_img.exists():
            src_path.rename(dst_img)
        rec["tile_path"] = str(dst_img.relative_to(DATA_DIR))

        # Write YOLO label
        if not dst_lbl.exists():
            has_bbox = rec["bbox_x1"] != -1
            is_positive = rec["tile_type"] == "pos"
            with open(dst_lbl, "w") as f:
                if is_positive and has_bbox:
                    line = _bbox_to_yolo(
                        rec["bbox_x1"], rec["bbox_y1"],
                        rec["bbox_x2"], rec["bbox_y2"],
                    )
                    f.write(line + "\n")
                # Negatives and positives without bbox: empty file

    # Also fetch and write test-set tiles (NE US 747 records)
    _fetch_test_tiles()

    # Write dataset.yaml
    yaml_content = f"""# SkyRoute helipad YOLO dataset — auto-generated by build_yolo_dataset.py
path: {YOLO_DIR.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: helipad
"""
    YAML_PATH.write_text(yaml_content, encoding="utf-8")
    log.info("dataset.yaml written → %s", YAML_PATH)

    # Write build log
    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        log.info("build_log.csv written → %d records", len(all_records))


def _fetch_test_tiles() -> None:
    """Fetch NAIP chips for the 747 NE US FAA records (test set).

    Uses a synthetic bbox (centre ± _DEFAULT_BBOX_HALF px) as a starting
    point for manual annotation. All chips are saved to images/test/ and
    labels/test/ with the synthetic bbox label.
    """
    if not NE_US_CSV.exists():
        log.warning("NE US CSV not found at %s — skipping test tile fetch", NE_US_CSV)
        return

    ne_df = pd.read_csv(NE_US_CSV)
    ne_df = ne_df.dropna(subset=["lat", "lon"])
    log.info("Fetching test-set tiles for %d NE US records ...", len(ne_df))

    img_dir = YOLO_DIR / "images" / "test"
    lbl_dir = YOLO_DIR / "labels" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    saved = skipped = failed = 0
    for _, row in ne_df.iterrows():
        ident = str(row.get("IDENT", row.get("ident", f"unk_{_}")) or f"unk_{_}").strip()
        lat, lon = float(row["lat"]), float(row["lon"])
        fname = f"{ident}.jpg"
        img_path = img_dir / fname
        lbl_path = lbl_dir / (ident + ".txt")

        if img_path.exists():
            skipped += 1
            continue

        try:
            img, cx, cy, gsd_m = fetch_naip_chip(lat, lon)
            log.debug("  test tile %s  GSD=%.2f m", ident, gsd_m)
        except NoImageryError:
            log.warning("  test tile %s: no NAIP coverage", ident)
            failed += 1
            continue
        except Exception as exc:
            log.warning("  test tile %s failed: %s", ident, exc)
            failed += 1
            continue

        img.save(str(img_path), quality=90)

        # Synthetic bbox: centre ± DEFAULT_BBOX_HALF (for manual review/correction)
        if not lbl_path.exists():
            half = _DEFAULT_BBOX_HALF
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(IMG_PX, cx + half), min(IMG_PX, cy + half)
            lbl_path.write_text(
                _bbox_to_yolo(x1, y1, x2, y2) + "\n# SYNTHETIC — verify manually\n",
                encoding="utf-8",
            )
        saved += 1

    log.info("Test tiles: saved=%d  skipped=%d  failed=%d", saved, skipped, failed)


# ── Step 8 — HTML review galleries ───────────────────────────────────────────

def step8_build_galleries(n_train_samples: int = 50) -> None:
    """Build HTML review galleries for spot-checking.

    Training gallery (review_train/index.html):
        n_train_samples random training tiles with bbox overlaid in red.
        Purpose: verify that bboxes align with visible helipad markers.

    Test gallery (review_test/index.html):
        All NE US test tiles with synthetic bbox in yellow.
        Purpose: reviewer confirms or corrects each bbox before YOLO training.

    Args:
        n_train_samples: Number of random training tiles to include.
    """
    _build_train_gallery(n_train_samples)
    _build_test_gallery()


def _draw_bbox_on_image(img: Image.Image, x1: int, y1: int, x2: int, y2: int,
                        color: str = "red") -> Image.Image:
    """Draw a bbox rectangle on a PIL image (non-destructive copy)."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return out


def _img_to_data_uri(img: Image.Image, max_side: int = 300) -> str:
    """Resize image and encode as a data URI for embedding in HTML."""
    import base64
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=75)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _build_train_gallery(n: int) -> None:
    """Build training spot-check gallery."""
    review_dir = YOLO_DIR / "review_train"
    review_dir.mkdir(parents=True, exist_ok=True)

    log_df = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()
    if log_df.empty:
        return

    pos_saved = log_df[
        (log_df["tile_type"] == "pos") &
        (log_df["status"].isin(["saved", "skipped_exists"])) &
        (log_df["bbox_x1"] != -1)
    ]
    sample = pos_saved.sample(min(n, len(pos_saved)), random_state=42)

    cards_html = []
    for _, row in sample.iterrows():
        img_path = DATA_DIR / row["tile_path"]
        if not img_path.exists():
            continue
        img = Image.open(img_path)
        annotated = _draw_bbox_on_image(
            img, int(row["bbox_x1"]), int(row["bbox_y1"]),
            int(row["bbox_x2"]), int(row["bbox_y2"]), color="red",
        )
        uri = _img_to_data_uri(annotated)
        label = (
            f"ID={int(row['helipadcat_id'])}  "
            f"cat={int(row['category'])}  "
            f"bbox=({int(row['bbox_x1'])},{int(row['bbox_y1'])},"
            f"{int(row['bbox_x2'])},{int(row['bbox_y2'])})"
        )
        cards_html.append(
            f'<div style="display:inline-block;margin:6px;text-align:center">'
            f'<img src="{uri}" style="display:block"><br>'
            f'<small>{label}</small></div>'
        )

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>SkyRoute — Training Review</title></head><body>"
        f"<h2>Training tiles spot-check ({len(cards_html)} samples)</h2>"
        f"<p>Red box = HelipadCAT annotation transformed to NAIP 640px chip space. "
        f"If the box does not frame the H-marking, flag the tile in build_log.csv.</p>"
        + "".join(cards_html)
        + "</body></html>"
    )
    (review_dir / "index.html").write_text(html, encoding="utf-8")
    log.info("Training review gallery → %s", review_dir / "index.html")


def _build_test_gallery() -> None:
    """Build test-set annotation guide gallery."""
    review_dir = YOLO_DIR / "review_test"
    review_dir.mkdir(parents=True, exist_ok=True)

    test_img_dir = YOLO_DIR / "images" / "test"
    test_lbl_dir = YOLO_DIR / "labels" / "test"

    if not test_img_dir.exists():
        return

    cards_html = []
    for img_path in sorted(test_img_dir.glob("*.jpg")):
        ident = img_path.stem
        lbl_path = test_lbl_dir / (ident + ".txt")

        try:
            img = Image.open(img_path)
        except Exception:
            continue
        bbox_info = "no bbox"
        if lbl_path.exists():
            try:
                label_text = lbl_path.read_text(encoding="utf-8")
            except (PermissionError, OSError):
                label_text = ""   # file locked (e.g. OneDrive sync) — skip bbox
            for line in label_text.splitlines():
                if line.startswith("0 "):
                    parts = list(map(float, line.split()[1:]))
                    cx_n, cy_n, w_n, h_n = parts
                    x1 = int((cx_n - w_n / 2) * IMG_PX)
                    y1 = int((cy_n - h_n / 2) * IMG_PX)
                    x2 = int((cx_n + w_n / 2) * IMG_PX)
                    y2 = int((cy_n + h_n / 2) * IMG_PX)
                    img = _draw_bbox_on_image(img, x1, y1, x2, y2, color="yellow")
                    bbox_info = f"bbox=({x1},{y1},{x2},{y2}) SYNTHETIC"
                    break

        uri = _img_to_data_uri(img)
        cards_html.append(
            f'<div style="display:inline-block;margin:6px;text-align:center;'
            f'border:1px solid #ccc;padding:4px">'
            f'<img src="{uri}" style="display:block"><br>'
            f'<small><b>{ident}</b><br>{bbox_info}</small></div>'
        )

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>SkyRoute — Test Set Review</title></head><body>"
        f"<h2>Test-set tiles ({len(cards_html)} records)</h2>"
        "<p><b>Yellow box = SYNTHETIC bbox (centre ± 80 px).</b> "
        "Open each tile in LabelImg (YOLO format) and draw a tight bbox around "
        "the H-marking or pad circle. If no helipad is visible, delete the label "
        "file — it becomes a true negative in the test set.</p>"
        + "".join(cards_html)
        + "</body></html>"
    )
    (review_dir / "index.html").write_text(html, encoding="utf-8")
    log.info("Test review gallery → %s (%d tiles)", review_dir / "index.html", len(cards_html))


# ── main ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build YOLO helipad detection dataset from HelipadCAT + NAIP imagery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1, metavar="N",
        help="Resume from step N (1–8). Steps before N are skipped.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process only the first N HelipadCAT records (smoke test).",
    )
    parser.add_argument(
        "--test-tiles-only", action="store_true",
        help="Only fetch NE US test-set chips and build test gallery.",
    )
    parser.add_argument(
        "--max-gsd", dest="max_gsd", type=float, default=None, metavar="M",
        help=(
            "Skip NAIP chips coarser than M metres/pixel. "
            "0.6 = keep 0.3 and 0.6 m/px; 1.0 = accept all. Default: accept all."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the full dataset construction pipeline."""
    global MAX_GSD_M
    args = _parse_args()
    start_step = args.from_step
    limit = args.limit

    if args.max_gsd is not None:
        MAX_GSD_M = args.max_gsd
        log.info("GSD filter: skipping chips with GSD > %.2f m", MAX_GSD_M)

    log.info("=" * 60)
    log.info("SkyRoute — YOLO helipad dataset builder")
    if limit:
        log.info("  LIMIT MODE: first %d HelipadCAT records only", limit)
    log.info("=" * 60)

    if args.test_tiles_only:
        log.info("Test-tiles-only mode: fetching NE US tiles and building test gallery")
        _fetch_test_tiles()
        _build_test_gallery()
        return

    # ── Steps 1–4: download and filter ────────────────────────────────────────
    hcat_df = faa_df = ne_df = eligible_df = None

    if start_step <= 1:
        log.info("\n[ 1 / 8 ]  Download HelipadCAT CSV")
        hcat_df = step1_download_helipadcat()

    if start_step <= 2:
        log.info("\n[ 2 / 8 ]  Download national FAA ArcGIS")
        faa_df = step2_download_faa_national()

    if start_step <= 3:
        if hcat_df is None:
            hcat_df = pd.read_csv(HELIPADCAT_CSV)
        if faa_df is None:
            faa_df = pd.read_csv(FAA_NATIONAL_CSV)
        log.info("\n[ 3 / 8 ]  Staleness filter")
        hcat_df = step3_staleness_filter(hcat_df, faa_df)

    if start_step <= 4:
        if hcat_df is None:
            raise RuntimeError("--from 4 requires hcat_df; run from step 3 first")
        ne_df = pd.read_csv(NE_US_CSV).dropna(subset=["lat", "lon"])
        log.info("\n[ 4 / 8 ]  Geographic dedup (remove NE US test set)")
        eligible_df = step4_geo_dedup(hcat_df, ne_df)
    else:
        # Resuming from step 5+: reconstruct eligible_df from existing log
        if LOG_PATH.exists():
            eligible_df = pd.read_csv(LOG_PATH).rename(
                columns={"lat": "latitude", "lon": "longitude"}
            )
            log.info("Resumed: %d records from build_log.csv", len(eligible_df))
        else:
            raise RuntimeError(
                f"--from {start_step} requires build_log.csv; run from step 1 first"
            )

    # ── Step 5: fetch tiles ────────────────────────────────────────────────────
    tile_records: list[dict] = []
    if start_step <= 5:
        log.info("\n[ 5 / 8 ]  Fetch NAIP chips for %d training records", len(eligible_df))
        tile_records = step5_fetch_tiles(eligible_df, limit=limit)

    # ── Step 6: easy negatives ─────────────────────────────────────────────────
    if start_step <= 6:
        if ne_df is None:
            ne_df = pd.read_csv(NE_US_CSV).dropna(subset=["lat", "lon"])
        if hcat_df is None:
            hcat_df = pd.read_csv(HELIPADCAT_CSV)

        all_lats = np.concatenate([
            eligible_df["latitude"].values,
            ne_df["lat"].values,
        ]).astype(np.float32)
        all_lons = np.concatenate([
            eligible_df["longitude"].values,
            ne_df["lon"].values,
        ]).astype(np.float32)

        log.info("\n[ 6 / 8 ]  Generate easy-negative tiles")
        neg_records = step6_easy_negatives(tile_records, all_lats, all_lons)
        tile_records.extend(neg_records)

    # ── Steps 7–8: labels, split, galleries ───────────────────────────────────
    if start_step <= 7:
        log.info("\n[ 7 / 8 ]  Write YOLO labels, split, dataset.yaml")
        step7_write_labels_and_split(tile_records)

    if start_step <= 8:
        log.info("\n[ 8 / 8 ]  Build HTML review galleries")
        step8_build_galleries(n_train_samples=50)

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("Dataset build complete")

    train_imgs = list((YOLO_DIR / "images" / "train").glob("*.jpg"))
    val_imgs   = list((YOLO_DIR / "images" / "val").glob("*.jpg"))
    test_imgs  = list((YOLO_DIR / "images" / "test").glob("*.jpg"))

    log.info("  Train : %d tiles", len(train_imgs))
    log.info("  Val   : %d tiles", len(val_imgs))
    log.info("  Test  : %d tiles", len(test_imgs))
    log.info("  YAML  : %s", YAML_PATH)
    log.info("  Log   : %s", LOG_PATH)
    log.info("")
    log.info("Next steps:")
    log.info("  1. Review  data/yolo_dataset/review_train/index.html  (spot-check bboxes)")
    log.info("  2. Review  data/yolo_dataset/review_test/index.html   (manual annotation)")
    log.info("  3. Open notebooks/02_yolo_training.ipynb in Colab to train YOLOv8s")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
