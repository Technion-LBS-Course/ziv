"""Download FAA and OSM helipad data for the Northeast US.

Usage:
    python scripts/fetch_ny_data.py

Outputs (written to data/):
    faa_helipads_raw.csv   — FAA US_Airport ArcGIS records, TYPE_CODE=HP
    osm_helipads_raw.csv   — OSM aeroway=helipad/heliport, Northeast bbox

FAA source:
    ADDS-FAA ArcGIS FeatureServer (services6.arcgis.com)
    Public, no auth, returns 26-field JSON for every US heliport.
    Coordinates are DMS strings — parsed to decimal degrees here.

OSM source:
    Overpass API, bounding box covering NY + NJ + CT + PA + MA.
"""

import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

FAA_OUT = DATA_DIR / "faa_helipads_raw.csv"
OSM_OUT = DATA_DIR / "osm_helipads_raw.csv"

# ── target states ─────────────────────────────────────────────────────────────
TARGET_STATES: list[str] = ["NY", "NJ", "CT", "PA", "MA"]

# ── FAA ArcGIS constants ──────────────────────────────────────────────────────
# ADDS-FAA US Airport FeatureServer (public, no auth required)
# Dataset: https://adds-faa.opendata.arcgis.com/datasets/e747ab91a11045e8b3f8a3efd093d3b5_0
_FAA_ARCGIS_URL = (
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ"
    "/arcgis/rest/services/US_Airport/FeatureServer/0/query"
)
_HELIPORT_TYPE_CODE = "HP"   # TYPE_CODE value for heliports in this dataset

# ── OSM constants ─────────────────────────────────────────────────────────────
# Bounding box covering NY, NJ, CT, PA, MA: (south, west, north, east)
_OSM_BBOX = (38.9, -80.5, 42.9, -69.9)
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "SkyRoute-Research/1.0 (Technion course project)"})

# ── DMS coordinate parser ─────────────────────────────────────────────────────

_DMS_RE = re.compile(r"^(\d{1,3})-(\d{2})-([\d.]+)([NSEWnsew])$")


def _dms_to_decimal(dms: str) -> float | None:
    """Parse an FAA DMS coordinate string to decimal degrees.

    FAA format: ``DD-MM-SS.SSSSH`` or ``DDD-MM-SS.SSSSH``
    e.g. ``"40-46-36.0000N"`` → 40.776667,  ``"073-58-59.0000W"`` → -73.983056

    Args:
        dms: Raw DMS string from the ArcGIS attribute.

    Returns:
        Decimal degrees (negative for S/W), or ``None`` if unparseable.
    """
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


# ── FAA fetch ─────────────────────────────────────────────────────────────────

def fetch_faa_helipads(states: list[str] = TARGET_STATES) -> pd.DataFrame:
    """Download FAA heliport records for the given states from ADDS-FAA ArcGIS.

    Queries ``TYPE_CODE='HP'`` for each state, handles ArcGIS pagination, and
    parses DMS coordinate strings to decimal degrees before saving.

    Args:
        states: List of two-letter US state abbreviations.

    Returns:
        Raw FAA DataFrame saved to ``data/faa_helipads_raw.csv``.

    Raises:
        RuntimeError: If the ArcGIS service is unreachable or returns an error.
    """
    states_sql = "','".join(s.upper() for s in states)
    where = f"TYPE_CODE='{_HELIPORT_TYPE_CODE}' AND STATE IN ('{states_sql}')"
    log.info("FAA query: %s", where)

    all_rows: list[dict] = []
    offset = 0
    page_size = 1000

    while True:
        params = {
            "where":             where,
            "outFields":         "*",
            "returnGeometry":    "false",
            "f":                 "json",
            "resultRecordCount": page_size,
            "resultOffset":      offset,
        }
        log.info("  Fetching FAA page offset=%d ...", offset)
        try:
            r = _SESSION.get(_FAA_ARCGIS_URL, params=params, timeout=30)
            r.raise_for_status()
            body = r.json()
        except Exception as exc:
            raise RuntimeError(f"FAA ArcGIS request failed: {exc}") from exc

        if "error" in body:
            raise RuntimeError(
                f"ArcGIS returned error: {body['error']}\n"
                f"WHERE clause used: {where}"
            )

        features = body.get("features", [])
        if not features:
            break

        for feat in features:
            all_rows.append(feat.get("attributes", {}))

        log.info("  Page → %d features (total so far: %d)", len(features), len(all_rows))

        if body.get("exceededTransferLimit"):
            offset += page_size
            time.sleep(0.2)
        else:
            break

    if not all_rows:
        raise RuntimeError(
            f"ArcGIS returned 0 heliport features for states {states}.\n"
            f"Verify endpoint: {_FAA_ARCGIS_URL}"
        )

    df = pd.DataFrame(all_rows)
    log.info("FAA raw: %d records × %d cols", *df.shape)
    log.info("FAA columns: %s", df.columns.tolist())

    # Parse DMS → decimal degrees and add convenience columns
    df["lat"] = df["LATITUDE"].apply(_dms_to_decimal)
    df["lon"] = df["LONGITUDE"].apply(_dms_to_decimal)

    null_coords = df["lat"].isna().sum()
    if null_coords:
        log.warning("%d records had unparseable coordinates — kept with NaN lat/lon", null_coords)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FAA_OUT, index=False)
    log.info("Saved → %s  (%d records)", FAA_OUT, len(df))
    return df


# ── OSM fetch ─────────────────────────────────────────────────────────────────

def fetch_osm_helipads(bbox: tuple[float, float, float, float] = _OSM_BBOX) -> pd.DataFrame:
    """Download OSM helipad/heliport elements for the Northeast US via Overpass.

    Args:
        bbox: ``(south, west, north, east)`` in decimal degrees.
              Default covers NY, NJ, CT, PA, MA.

    Returns:
        Flat DataFrame: osm_id, osm_type, lat, lon, plus one column per OSM tag.

    Raises:
        requests.HTTPError: If the Overpass API returns a non-200 response.
    """
    s, w, n, e = bbox
    query = f"""
[out:json][timeout:120];
(
  node[aeroway=helipad]({s},{w},{n},{e});
  node[aeroway=heliport]({s},{w},{n},{e});
  way[aeroway=helipad]({s},{w},{n},{e});
  way[aeroway=heliport]({s},{w},{n},{e});
  relation[aeroway=helipad]({s},{w},{n},{e});
  relation[aeroway=heliport]({s},{w},{n},{e});
);
out center tags;
"""
    log.info(
        "Querying Overpass — bbox S=%.1f W=%.1f N=%.1f E=%.1f "
        "(NY + NJ + CT + PA + MA) ...", s, w, n, e
    )

    r = _SESSION.post(_OVERPASS_URL, data={"data": query}, timeout=150)
    r.raise_for_status()

    elements: list[dict] = r.json().get("elements", [])
    log.info("Overpass returned %d element(s)", len(elements))

    rows: list[dict] = []
    for el in elements:
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        row: dict = {
            "osm_id":   el["id"],
            "osm_type": el["type"],
            "lat":      float(lat),
            "lon":      float(lon),
        }
        row.update(el.get("tags", {}))
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("OSM helipads parsed: %d rows × %d cols", *df.shape)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OSM_OUT, index=False)
    log.info("Saved → %s", OSM_OUT)
    return df


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("SkyRoute — helipad data fetch  (states: %s)", ", ".join(TARGET_STATES))
    log.info("=" * 60)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("[ 1 / 2 ]  FAA heliports — ADDS-FAA ArcGIS FeatureServer")
    log.info("-" * 40)
    try:
        faa_df = fetch_faa_helipads()
        log.info("FAA ✓  %d records  →  %s", len(faa_df), FAA_OUT.name)
    except Exception as exc:
        log.error("FAA fetch failed: %s", exc)
        faa_df = None

    log.info("")
    log.info("[ 2 / 2 ]  OSM helipads — Northeast US bbox")
    log.info("-" * 40)
    try:
        osm_df = fetch_osm_helipads()
        log.info("OSM ✓  %d records  →  %s", len(osm_df), OSM_OUT.name)
    except Exception as exc:
        log.error("OSM fetch failed: %s", exc)
        osm_df = None

    log.info("")
    log.info("=" * 60)
    log.info("Summary")
    log.info("  FAA : %s", f"{len(faa_df):>4d} records" if faa_df is not None else "FAILED")
    log.info("  OSM : %s", f"{len(osm_df):>4d} records" if osm_df is not None else "FAILED")
    log.info("  Data dir : %s", DATA_DIR)
    log.info("=" * 60)
