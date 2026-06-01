"""Enrich FAA helipad records with ADIP Airport Master Record data.

For each FAA location identifier (IDENT) in faa_helipads_raw.csv, POSTs to
the ADIP getAirportDetails endpoint and extracts:
  - Operational status, inspection history, coordinate/elevation quality flags
  - All airport remarks (contact info, restrictions, NOTAMs)
  - Helipad landing area dimensions: TLOF/FATO size, design category,
    elevated height AGL, weight bearing capacity
  - Raw JSON saved per-record to data/adip_raw/ for future field discovery

Usage:
    python scripts/fetch_adip_details.py

Output:
    data/faa_adip_enriched.csv  — original FAA columns + ADIP enrichment columns
    data/adip_raw/<IDENT>.json  — full raw ADIP response per heliport

ADIP endpoint:
    POST https://adip.faa.gov/agisServices/public-api/getAirportDetails
    Body: {"locId": "<IDENT>"}
    Auth: static application key embedded in ADIP Angular bundle
"""

import json
import logging
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

FAA_PATH    = DATA_DIR / "faa_helipads_raw.csv"
ADIP_OUT    = DATA_DIR / "faa_adip_enriched.csv"
ADIP_RAW_DIR = DATA_DIR / "adip_raw"   # one JSON file per IDENT

# ── constants ─────────────────────────────────────────────────────────────────
_ADIP_URL   = "https://adip.faa.gov/agisServices/public-api/getAirportDetails"
_REQ_DELAY  = 0.5   # seconds between requests (~2 req/sec)
_TIMEOUT    = 20    # seconds per request

# WKT formats seen in the wild:
#   "SRID=4326;POINT(-77.470 41.131)"
#   "POINT(-77.470 41.131)"
_WKT_RE = re.compile(r"POINT\((-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\)")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent":        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/148.0.0.0 Safari/537.36",
    # Static application key embedded in the ADIP Angular bundle
    "Authorization":     "Basic 3f647d1c-a3e7-415e-96e1-6e8415e6f209-ADIP",
    "Content-Type":      "application/json;charset=UTF-8",
    "Accept":            "application/json, text/plain, */*",
    "Origin":            "https://adip.faa.gov",
    "Referer":           "https://adip.faa.gov/agis/public/",
    "Accept-Language":   "en-US,en;q=0.9",
    "If-Modified-Since": "0",
    "Cache-Control":     "no-cache",
    "Pragma":            "no-cache",
})


def _warm_up_session() -> None:
    """Pre-fetch the ADIP public page to establish session cookies (JSESSIONID).

    The browser makes a getEnv + page load before calling getAirportDetails.
    Without a valid session, the Java backend returns 400.
    """
    urls = [
        "https://adip.faa.gov/agis/public/",
        "https://adip.faa.gov/agisServices/public-api/getEnv",
    ]
    for url in urls:
        try:
            _SESSION.get(url, timeout=15)
            log.info("Session warm-up: %s  (cookies: %s)",
                     url, list(_SESSION.cookies.keys()))
        except Exception as exc:
            log.warning("Session warm-up failed for %s: %s", url, exc)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_wkt_point(wkt: str | None) -> tuple[float | None, float | None]:
    """Return (lon, lat) from a WKT POINT string, or (None, None) on failure."""
    if not wkt:
        return None, None
    m = _WKT_RE.search(str(wkt))
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))  # lon first in WKT


def _days_since(iso_date: str | None) -> int | None:
    """Return days between an ISO-8601 date string and today, or None."""
    if not iso_date:
        return None
    try:
        d = date.fromisoformat(iso_date[:10])   # handles "2020-03-25T00:00:00"
        return (date.today() - d).days
    except ValueError:
        return None


def _fetch_one(loc_id: str) -> dict:
    """POST to ADIP and return extracted fields for one locId.

    Also saves the raw JSON response to data/adip_raw/<locId>.json so
    newly discovered field names can be exploited without re-fetching.

    Args:
        loc_id: FAA location identifier (e.g. "4PA4").

    Returns:
        Dict of enrichment columns; empty dict on failure.
    """
    try:
        r = _SESSION.post(_ADIP_URL, json={"locId": loc_id}, timeout=_TIMEOUT)
        if not r.ok:
            log.warning("  %s — HTTP %d: %s", loc_id, r.status_code, r.text[:200])
            return {}
        d = r.json()
    except Exception as exc:
        log.warning("  %s — request failed: %s", loc_id, exc)
        return {}

    # Persist raw JSON so field discovery can happen offline
    ADIP_RAW_DIR.mkdir(parents=True, exist_ok=True)
    (ADIP_RAW_DIR / f"{loc_id}.json").write_text(
        json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return _parse_response(d)


def _extract_remarks(d: dict) -> str | None:
    """Concatenate all remark text from any remarks array in the response.

    NASR/ADIP uses several field names for remarks depending on record type.
    Each remark dict may have 'remark', 'text', or 'remarkText' keys plus an
    optional 'remarkCode' / 'code' prefix.
    """
    for key in ("airportRemarks", "remarks", "otherInformation",
                "attendanceRemarks", "operationalRemarks"):
        raw = d.get(key)
        if not raw:
            continue
        if isinstance(raw, str):
            return raw.strip() or None
        if isinstance(raw, list):
            parts: list[str] = []
            for item in raw:
                if isinstance(item, str):
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    txt  = (item.get("remark") or item.get("text") or
                            item.get("remarkText") or "").strip()
                    code = (item.get("remarkCode") or item.get("code") or "").strip()
                    if txt:
                        parts.append(f"[{code}] {txt}" if code else txt)
            if parts:
                return " | ".join(parts)
    return None


def _extract_landing_area(d: dict) -> dict:
    """Pull TLOF/FATO dimensions and helipad metadata from nested arrays.

    ADIP may use different key names for the helipad landing area array.
    We try all known variants and return the first populated entry.
    Fields use multiple candidate key names to handle NASR naming variations.
    """
    for key in ("helipads", "landingAreas", "helicopterLandingAreas",
                "helicpterLandingArea",   # known NASR typo variant
                "helipadLandingAreas"):
        areas = d.get(key)
        if areas and isinstance(areas, list):
            h = areas[0]
            break
    else:
        h = {}

    def _get(*keys):
        for k in keys:
            v = h.get(k)
            if v is not None:
                return v
        return None

    return {
        "tlof_length_ft":   _get("tloflength", "tlofLength", "tlof_length"),
        "tlof_width_ft":    _get("tlofwidth",  "tlofWidth",  "tlof_width"),
        "fato_length_ft":   _get("fatoLength", "fatoDimension", "fato_length"),
        "fato_width_ft":    _get("fatoWidth",  "fato_width"),
        "design_category":  _get("designCategory", "design_category", "designCat"),
        "elevated_agl_ft":  _get("elevatedHeight", "heightAboveGround",
                                 "elevatedAgl", "elevatedHeightAgl"),
        "weight_limit_lbs": _get("weightBearingCapacity", "weightLimit",
                                 "maxWeightCapacity"),
        "surface_type":     _get("surfaceType", "surface", "pavementType"),
        "pad_count":        len(areas) if (areas := d.get("helipads")
                                          or d.get("landingAreas", [])) else None,
    }


def _parse_response(d: dict) -> dict:
    """Extract all enrichment fields from a decoded ADIP JSON response."""
    adip_lon, adip_lat = _parse_wkt_point(d.get("arp"))
    landing = _extract_landing_area(d)

    return {
        # ── operational status ────────────────────────────────────────────────
        "adip_status":        d.get("status"),
        # ── precise ARP coordinates ───────────────────────────────────────────
        "adip_lat":           adip_lat,
        "adip_lon":           adip_lon,
        "arp_method":         d.get("arpDeterminationMethod"),   # SURVEYED/ESTIMATED/OWNER
        "position_date":      d.get("positionSourceDate"),
        # ── elevation quality ─────────────────────────────────────────────────
        "elev_method":        d.get("elevationDeterminationMethod"),
        "elevation_date":     d.get("elevationSourceDate"),
        # ── data freshness (ML feature) ───────────────────────────────────────
        "last_info_date":     d.get("lastInfoRequestDate"),
        "last_info_days_ago": _days_since(d.get("lastInfoRequestDate")),
        # ── ownership / use ───────────────────────────────────────────────────
        "use_code":           d.get("facilityUseCode"),           # PR/PU
        "ownership_code":     d.get("ownershipTypeCode"),         # PR/PU/MA
        # ── infrastructure quality proxies ────────────────────────────────────
        "wind_indicator":     d.get("windIndicatorCode"),         # Y/N
        "notam_service":      d.get("notamServiceFlag"),          # Y/N
        "inspection_method":  d.get("inspectionMethod"),          # F=FAA, 2=state, N=none
        "inspection_agency":  d.get("inspectionAgency"),
        # ── remarks (contact info, restrictions, NOTAMs) ──────────────────────
        "remarks":            _extract_remarks(d),
        # ── helipad landing area dimensions ───────────────────────────────────
        "tlof_length_ft":     landing["tlof_length_ft"],
        "tlof_width_ft":      landing["tlof_width_ft"],
        "fato_length_ft":     landing["fato_length_ft"],
        "fato_width_ft":      landing["fato_width_ft"],
        "design_category":    landing["design_category"],
        "elevated_agl_ft":    landing["elevated_agl_ft"],
        "weight_limit_lbs":   landing["weight_limit_lbs"],
        "surface_type":       landing["surface_type"],
        "pad_count":          landing["pad_count"],
    }


# ── main ──────────────────────────────────────────────────────────────────────

def fetch_adip_details(faa_path: Path = FAA_PATH) -> pd.DataFrame:
    """Enrich FAA helipad records with ADIP Airport Master Record data.

    Reads faa_helipads_raw.csv, fetches ADIP details for every IDENT, and
    writes the merged result to faa_adip_enriched.csv.

    Args:
        faa_path: Path to faa_helipads_raw.csv produced by fetch_ny_data.py.

    Returns:
        Enriched DataFrame (also written to data/faa_adip_enriched.csv).

    Raises:
        FileNotFoundError: If faa_path does not exist.
        RuntimeError: If no IDENT column is found in the CSV.
    """
    if not faa_path.exists():
        raise FileNotFoundError(
            f"FAA data not found at {faa_path}.\n"
            "Run: python scripts/fetch_ny_data.py first."
        )

    _warm_up_session()

    faa_df = pd.read_csv(faa_path, dtype=str, low_memory=False)
    log.info("Loaded FAA CSV: %d records × %d cols", *faa_df.shape)

    ident_col = next(
        (c for c in faa_df.columns if c.upper() == "IDENT"), None
    )
    if ident_col is None:
        raise RuntimeError(
            f"No IDENT column found in {faa_path}.\n"
            f"Available columns: {faa_df.columns.tolist()}"
        )

    idents = faa_df[ident_col].str.strip().str.upper().fillna("").tolist()
    log.info("Fetching ADIP details for %d location identifiers ...", len(idents))

    enriched_rows: list[dict] = []
    ok_count = 0
    for i, loc_id in enumerate(idents, 1):
        if not loc_id or loc_id == "NAN":
            log.debug("  [%d/%d]  skipped (no IDENT)", i, len(idents))
            enriched_rows.append({})
            continue

        log.info("  [%d/%d]  %s", i, len(idents), loc_id)
        row = _fetch_one(loc_id)
        enriched_rows.append(row)
        if row:
            ok_count += 1
        time.sleep(_REQ_DELAY)

    log.info("ADIP fetch complete: %d / %d succeeded", ok_count, len(idents))

    enrich_df = pd.DataFrame(enriched_rows, index=faa_df.index)
    result    = pd.concat([faa_df, enrich_df], axis=1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ADIP_RAW_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(ADIP_OUT, index=False)
    log.info("Saved → %s  (%d records, %d cols)", ADIP_OUT, len(result), len(result.columns))
    log.info("Raw JSON → %s  (%d files)", ADIP_RAW_DIR, len(list(ADIP_RAW_DIR.glob("*.json"))))

    log.info("")
    log.info("Column fill rates:")
    for col in enrich_df.columns:
        n = enrich_df[col].notna().sum()
        pct = n / len(idents) * 100
        log.info("  %-24s  %4d / %d  (%.0f%%)", col, n, len(idents), pct)

    return result


if __name__ == "__main__":
    fetch_adip_details()
