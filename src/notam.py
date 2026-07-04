"""FAA NOTAM / TFR + Aviation Weather Center METAR module.

TFR data — two public FAA sources (no API key required):

  1. Live TFRs (all active NOTAMs with polygon geometry):
     FAA GeoServer WFS — the same endpoint used by tfr.faa.gov/tfr3/
     https://tfr.faa.gov/geoserver/TFR/ows  (WFS GetFeature, typeName=TFR:V_TFR_LOC)
     Returns ~230 active TFRs globally including NE US security/VIP/event TFRs.

  2. Stadiums (game-day TFR reference points, 3 nm radius during events):
     https://adds-faa.opendata.arcgis.com/datasets/faa::stadiums/

METAR data:
  Aviation Weather Center API (no key required).
  https://aviationweather.gov/api/data/metar?ids=<ICAO>&format=json
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "SkyRoute/1.0"})

# METAR in-process cache: {icao_id: (result_dict, fetched_at)}
_metar_cache: dict[str, tuple[dict, datetime]] = {}
_METAR_TTL = timedelta(minutes=5)

_GEOSERVER_WFS = "https://tfr.faa.gov/geoserver/TFR/ows"
_STADIUM_URL   = (
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services"
    "/Stadiums/FeatureServer/0"
)

# Module-level TFR cache — survives Streamlit reruns within the same process.
# Used as fallback when the FAA GeoServer is slow or unreachable.
_tfr_cache: list[dict] = []
_tfr_cache_at: datetime | None = None
_TFR_CACHE_TTL = timedelta(minutes=15)


# ── TFR / NOTAM ────────────────────────────────────────────────────────────────

# FAA WFS date field name variants (tried in order; first non-empty wins)
_TFR_EFF_FIELDS = ("EFF_DATE", "EFF_LOCAL_DATE_TIME", "EFFECTIVE", "START_DATE")
_TFR_EXP_FIELDS = ("EXP_DATE", "EXP_LOCAL_DATE_TIME", "EXPIRATION", "END_DATE", "EXPIRE_DATE")

# Common FAA date/time format patterns (UTC / Zulu)
_TFR_DT_FMTS = (
    "%m/%d/%Y %H%M",    # "09/01/2026 1400"
    "%m/%d/%Y %H:%M",   # "09/01/2026 14:00"
    "%Y/%m/%d %H:%M:%S",  # "2026/09/01 14:00:00"
    "%Y-%m-%d %H:%M:%S",  # "2026-09-01 14:00:00"
    "%Y-%m-%dT%H:%M:%S",  # "2026-09-01T14:00:00"
    "%m/%d/%Y",            # "09/01/2026" (date-only fallback)
)


def _parse_tfr_dt(raw: str) -> Optional[datetime]:
    """Parse a FAA WFS date/time string to a UTC-aware datetime.

    Tries multiple format patterns; returns None if nothing matches.
    FAA NOTAMs use Zulu (UTC) time.
    """
    if not raw:
        return None
    # Strip trailing Z/z and whitespace, normalise
    s = raw.strip().rstrip("Zz").strip()
    for fmt in _TFR_DT_FMTS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def fetch_active_tfrs() -> list[dict]:
    """Fetch all active TFR polygons from the FAA GeoServer WFS endpoint.

    No API key required.  Returns live TFRs (security, VIP, events, airshows)
    plus stadium game-day restriction points.

    The GeoServer endpoint is the same backend used by tfr.faa.gov/tfr3/.
    Same NOTAM_KEY may appear multiple times (different altitude rings);
    all rings are kept so that routing avoidance is conservative.

    Returns:
        List of dicts, each with keys:
        notam_id, text, type_code, geometry_type, coordinates.
        coordinates is [[lat, lon], ...] (Folium-compatible format).
    """
    global _tfr_cache, _tfr_cache_at
    results: list[dict] = []

    # ── 1. Live TFR polygons (GeoServer WFS) ──────────────────────────────────
    geoserver_ok = False
    try:
        resp = _SESSION.get(
            _GEOSERVER_WFS,
            params={
                "service":      "WFS",
                "version":      "1.1.0",
                "request":      "GetFeature",
                "typeName":     "TFR:V_TFR_LOC",
                "maxFeatures":  500,
                "outputFormat": "application/json",
                "srsname":      "EPSG:4326",
            },
            headers={"Referer": "https://tfr.faa.gov/tfr3/"},
            timeout=8,   # reduced from 20 — fail fast, use cache on timeout
        )
        resp.raise_for_status()
        for feat in resp.json().get("features", []):
            geo   = feat.get("geometry") or {}
            props = feat.get("properties") or {}
            geo_type = geo.get("type", "")

            if geo_type == "Polygon":
                ring_gj = geo.get("coordinates", [[]])[0]
                coordinates = [[pt[1], pt[0]] for pt in ring_gj]
            elif geo_type == "MultiPolygon":
                ring_gj = geo.get("coordinates", [[[]]])[0][0]
                coordinates = [[pt[1], pt[0]] for pt in ring_gj]
                geo_type = "Polygon"
            else:
                continue

            eff_raw = next((props.get(f) for f in _TFR_EFF_FIELDS if props.get(f)), None)
            exp_raw = next((props.get(f) for f in _TFR_EXP_FIELDS if props.get(f)), None)
            results.append({
                "notam_id":      props.get("NOTAM_KEY", str(props.get("GID", ""))),
                "text":          props.get("TITLE", "Active TFR"),
                "type_code":     props.get("LEGAL", "TFR"),
                "geometry_type": geo_type,
                "coordinates":   coordinates,
                "state":         props.get("STATE", ""),
                "effective_utc": _parse_tfr_dt(eff_raw or ""),
                "expires_utc":   _parse_tfr_dt(exp_raw or ""),
            })

        log.info("Fetched %d live TFR polygons from GeoServer WFS", len(results))
        geoserver_ok = True

    except requests.RequestException as exc:
        log.warning("TFR GeoServer WFS fetch failed: %s", exc)
        # Return stale cache (up to 15 min) rather than an empty overlay
        now = datetime.now(timezone.utc)
        if _tfr_cache and _tfr_cache_at and (now - _tfr_cache_at) < _TFR_CACHE_TTL:
            log.info("Using cached TFR data (%s old)", now - _tfr_cache_at)
            return _tfr_cache  # stadiums already included in prior cache

    # ── 2. Stadium TFR points (game-day awareness) ─────────────────────────────
    n_before = len(results)
    try:
        resp = _SESSION.get(
            _STADIUM_URL + "/query",
            params={
                "where":     "STATUS_CODE='Open'",
                "outFields": "GLOBAL_ID,NAME,CITY,STATE",
                "f":         "geojson",
                "outSR":     "4326",
            },
            timeout=8,
        )
        resp.raise_for_status()
        for feat in resp.json().get("features", []):
            geo   = feat.get("geometry") or {}
            props = feat.get("properties") or {}
            if geo.get("type") != "Point":
                continue
            lon, lat = geo["coordinates"]
            name  = props.get("NAME", "Stadium")
            city  = props.get("CITY", "")
            state = props.get("STATE", "")
            results.append({
                "notam_id":      str(props.get("GLOBAL_ID", name)),
                "text": (
                    f"Stadium TFR: {name} ({city}, {state}) — "
                    "3 nm radius, surface to 3,000 ft AGL, 1 hr before/after events"
                ),
                "type_code":     "STADIUM",
                "geometry_type": "Point",
                "coordinates":   [[lat, lon]],
                "state":         state,
            })

        log.info("Fetched %d stadium TFR points", len(results) - n_before)

    except requests.RequestException as exc:
        log.warning("Stadium fetch failed: %s", exc)

    # Update module-level cache with successful result
    if geoserver_ok:
        _tfr_cache    = results
        _tfr_cache_at = datetime.now(timezone.utc)

    return results


def notam_closes_airspace(tfr: dict) -> bool:
    """True if this TFR entry represents an airspace closure."""
    return tfr.get("type_code") in ("NDA_TFR", "DEF", "STADIUM", "D", "TFR", "SECURITY")


def tfrs_to_geojson(tfrs: list[dict]) -> dict:
    """Convert TFR list to a GeoJSON FeatureCollection for Folium / JS injection.

    Args:
        tfrs: Output of fetch_active_tfrs().

    Returns:
        GeoJSON FeatureCollection dict (JSON-serialisable).
    """
    features = []
    for tfr in tfrs:
        coords = tfr.get("coordinates", [])
        geo_type = tfr.get("geometry_type", "Point")

        if geo_type == "Polygon" and len(coords) >= 3:
            # Internal [[lat, lon], ...] → GeoJSON [[lon, lat], ...]
            ring = [[pt[1], pt[0]] for pt in coords]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            geometry = {"type": "Polygon", "coordinates": [ring]}
        elif coords:
            pt = coords[0]
            geometry = {"type": "Point", "coordinates": [pt[1], pt[0]]}
        else:
            continue

        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "notam_id": tfr.get("notam_id", ""),
                "text": tfr.get("text", ""),
                "type_code": tfr.get("type_code", ""),
            },
        })

    return {"type": "FeatureCollection", "features": features}


# ── METAR ─────────────────────────────────────────────────────────────────────

def fetch_metar(icao_id: str) -> Optional[dict]:
    """Fetch the latest METAR for an ICAO station via Aviation Weather Center.

    No API key required.  Responses are cached for 5 minutes.

    Args:
        icao_id: ICAO or FAA station identifier (e.g. 'KJFK', 'NK39').

    Returns:
        Dict with keys: wind_dir, wind_kt, visibility_sm, ceiling_ft,
        flight_category, raw.  None if no observation found.
    """
    icao_id = icao_id.upper().strip()
    now = datetime.now(timezone.utc)

    if icao_id in _metar_cache:
        cached, fetched_at = _metar_cache[icao_id]
        if now - fetched_at < _METAR_TTL:
            return cached

    try:
        resp = _SESSION.get(
            "https://aviationweather.gov/api/data/metar",
            params={"ids": icao_id, "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        log.debug("METAR fetch failed for %s: %s", icao_id, exc)
        return None

    if not data:
        log.debug("No METAR found for %s", icao_id)
        return None

    obs = data[0]

    ceiling_ft: Optional[int] = None
    for layer in obs.get("clouds", []):
        if layer.get("cover") in ("BKN", "OVC", "VV"):
            base = layer.get("base")
            if base is not None:
                ceiling_ft = int(base)
                break

    vis_raw = obs.get("visib", None)
    try:
        visibility_sm = float(str(vis_raw).replace("+", "")) if vis_raw is not None else None
    except ValueError:
        visibility_sm = None

    result = {
        "wind_dir": obs.get("wdir"),
        "wind_kt": obs.get("wspd"),
        "wind_gust_kt": obs.get("wgst"),
        "visibility_sm": visibility_sm,
        "ceiling_ft": ceiling_ft,
        "flight_category": obs.get("fltCat", "VFR"),
        "raw": obs.get("rawOb", ""),
        "temp_c": obs.get("temp"),
        "icao_id": icao_id,
    }

    _metar_cache[icao_id] = (result, now)
    return result


def route_weather_summary(helipads: list[dict]) -> dict[str, dict]:
    """Fetch METAR + active TFRs for a list of helipads.

    Args:
        helipads: List of dicts with at least 'ident' key.

    Returns:
        Dict keyed by ident: {'metar': ..., 'tfrs': [...]}
    """
    tfrs = fetch_active_tfrs()
    summary: dict[str, dict] = {}
    for pad in helipads:
        ident = pad.get("ident", "")
        summary[ident] = {
            "metar": fetch_metar(ident),
            "tfrs": tfrs,
        }
    return summary
