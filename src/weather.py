"""NWS radar precipitation module.

Data source: NOAA/NWS GeoServer (opengeo.ncep.noaa.gov) — the same backend
used by radar.weather.gov.  No API key required.

WMS endpoint:
    https://opengeo.ncep.noaa.gov/geoserver/conus/ows
    Layer: conus_bref_qcd  (CONUS Base Reflectivity, QC'd)
    CRS:   EPSG:4326  (WMS 1.3.0 axis order: lat_min,lon_min,lat_max,lon_max)

Intensity proxy: NWS colormap red channel.
    Transparent pixel → 0 (no precipitation)
    Green/cyan (light, <20 dBZ)  →  R ≈   0–50   → intensity  0–50
    Yellow      (moderate 20-40) →  R ≈  50–255   → intensity ~50–200
    Orange/red  (heavy 40-55)    →  R ≈ 200–255   → intensity ~200–255
"""

import io
import logging
from typing import Optional

import requests
from PIL import Image

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "SkyRoute/1.0",
    "Referer":    "https://radar.weather.gov/",
})

_NWS_WMS_URL     = "https://opengeo.ncep.noaa.gov/geoserver/conus/ows"
_NWS_RADAR_LAYER = "conus_bref_qcd"

# In-process sampling cache: {cache_key: intensity}
_sample_cache: dict[str, int] = {}

# Mode-specific thresholds (NWS red-channel intensity 0–255)
PRECIP_THRESHOLDS = {
    "walking":    {"warn": 60,  "avoid": 160},
    "car":        {"warn": 100, "avoid": 200},
    "helicopter": {"warn": 40,  "avoid": 120},
}


# ── WMS helpers ────────────────────────────────────────────────────────────────

def get_nws_wms_url() -> str:
    """Return the NWS GeoServer WMS base URL."""
    return _NWS_WMS_URL


def get_nws_wms_kwargs() -> dict:
    """Return kwargs for folium.WmsTileLayer targeting the NWS radar layer.

    Usage:
        folium.WmsTileLayer(**get_nws_wms_kwargs()).add_to(m)
    """
    return {
        "url":         _NWS_WMS_URL,
        "layers":      _NWS_RADAR_LAYER,
        "fmt":         "image/png",
        "transparent": True,
        "version":     "1.3.0",
        "attr":        '<a href="https://radar.weather.gov">NWS Radar</a>',
        "name":        "Precipitation radar (NWS)",
        "overlay":     True,
        "control":     True,
        "show":        True,
    }


# ── Precipitation sampling ─────────────────────────────────────────────────────

def sample_precipitation_at_latlon(
    lat: float,
    lon: float,
    *_args,          # accepts (and ignores) legacy host/path args for API compat
    **_kwargs,
) -> int:
    """Sample NWS radar precipitation intensity at a geographic point.

    Issues a tiny WMS GetMap request (3×3 pixels, 0.05° window) and reads
    the red channel of the centre pixel.  The NWS colormap encodes intensity
    via colour hue: green (light) → yellow (moderate) → red (heavy).

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        Intensity 0–255.  0 = no precipitation.  >40 = light rain.  >120 = heavy.
    """
    # Round to ~5 km grid to share cache entries across nearby helipads
    cache_key = f"{lat:.2f},{lon:.2f}"
    if cache_key in _sample_cache:
        return _sample_cache[cache_key]

    delta = 0.025  # ~2.5° half-width → 5 km window
    bbox  = f"{lat - delta},{lon - delta},{lat + delta},{lon + delta}"

    try:
        resp = _SESSION.get(
            _NWS_WMS_URL,
            params={
                "service":     "WMS",
                "version":     "1.3.0",
                "request":     "GetMap",
                "layers":      _NWS_RADAR_LAYER,
                "bbox":        bbox,
                "width":       3,
                "height":      3,
                "crs":         "EPSG:4326",
                "format":      "image/png",
                "transparent": "true",
            },
            timeout=10,
        )
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        r_val, g_val, b_val, a_val = img.getpixel((1, 1))
        intensity = r_val if a_val > 10 else 0
    except Exception as exc:
        log.debug("NWS radar sample failed (%s): %s", cache_key, exc)
        intensity = 0

    _sample_cache[cache_key] = intensity
    return intensity


def check_route_precipitation(
    waypoints: list[dict],
    *_args,
    thresholds: Optional[dict] = None,
    **_kwargs,
) -> list[dict]:
    """Check NWS radar precipitation intensity along a list of route waypoints.

    Args:
        waypoints: List of dicts with keys: lat, lon, label, mode.
            mode is one of 'walking', 'car', 'helicopter'.
        thresholds: Dict of mode-specific warn/avoid thresholds.
            Defaults to PRECIP_THRESHOLDS.

    Returns:
        List of dicts (one per waypoint) with the input fields plus:
        intensity (0–255), warning (bool), severity ('none'|'warn'|'avoid').
    """
    thr = thresholds or PRECIP_THRESHOLDS
    results = []
    for wp in waypoints:
        lat   = float(wp["lat"])
        lon   = float(wp["lon"])
        mode  = wp.get("mode", "helicopter")
        label = wp.get("label", f"{lat:.4f},{lon:.4f}")

        intensity  = sample_precipitation_at_latlon(lat, lon)
        mode_thr   = thr.get(mode, thr["helicopter"])

        if intensity >= mode_thr["avoid"]:
            severity, warning = "avoid", True
        elif intensity >= mode_thr["warn"]:
            severity, warning = "warn", True
        else:
            severity, warning = "none", False

        results.append({
            **wp,
            "lat": lat, "lon": lon, "label": label,
            "intensity": intensity, "warning": warning, "severity": severity,
        })

    return results
