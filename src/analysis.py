"""Cross-source analysis and consistency checks for SkyRoute helipad data.

All functions are pure (no side effects, no I/O) so they can be cached
safely by Streamlit's @st.cache_data decorator.
"""

import difflib
import logging
import re

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_EARTH_R = 6_371_000  # metres
_FT_PER_M = 3.28084


# ── distance ──────────────────────────────────────────────────────────────────

def haversine_matrix(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    """Pairwise haversine distances in metres between two point sets.

    Args:
        lat1, lon1: (N,) arrays — first point set (decimal degrees).
        lat2, lon2: (M,) arrays — second point set (decimal degrees).

    Returns:
        (N, M) float64 array of distances in metres.
    """
    φ1 = np.radians(lat1)[:, None]
    φ2 = np.radians(lat2)[None, :]
    λ1 = np.radians(lon1)[:, None]
    λ2 = np.radians(lon2)[None, :]
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = np.sin(dφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ / 2) ** 2
    return _EARTH_R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ── matching ──────────────────────────────────────────────────────────────────

def match_by_faa_id(
    faa_df: pd.DataFrame,
    osm_df: pd.DataFrame,
) -> pd.DataFrame:
    """Exact match on FAA location identifier: FAA.IDENT == OSM.faa.

    OSM contributors tag helipad nodes with ``faa=<LID>`` when they know
    the official FAA location identifier (present for ~24% of OSM records).

    Args:
        faa_df: FAA DataFrame with IDENT column.
        osm_df: OSM DataFrame with faa column.

    Returns:
        DataFrame with columns: faa_idx, osm_idx, match_method='faa_id'.
    """
    if "IDENT" not in faa_df.columns or "faa" not in osm_df.columns:
        return pd.DataFrame(columns=["faa_idx", "osm_idx", "match_method"])

    faa_ids = faa_df["IDENT"].str.strip().str.upper()
    osm_ids = osm_df["faa"].str.strip().str.upper()

    rows = []
    for faa_idx, faa_id in faa_ids.items():
        if pd.isna(faa_id):
            continue
        osm_matches = osm_df.index[osm_ids == faa_id]
        for osm_idx in osm_matches:
            rows.append({"faa_idx": faa_idx, "osm_idx": osm_idx,
                         "match_method": "faa_id"})

    result = pd.DataFrame(rows)
    log.info("FAA-ID exact matches: %d", len(result))
    return result


def match_by_proximity(
    faa_df: pd.DataFrame,
    osm_df: pd.DataFrame,
    threshold_m: float = 250.0,
    exclude_faa_idx: set | None = None,
) -> pd.DataFrame:
    """Nearest-neighbour spatial match: each FAA record → closest OSM record.

    Args:
        faa_df: FAA DataFrame with lat/lon columns.
        osm_df: OSM DataFrame with lat/lon columns.
        threshold_m: Maximum distance (metres) to accept as a match.
        exclude_faa_idx: FAA indices already matched (skipped here).

    Returns:
        DataFrame with columns: faa_idx, osm_idx, distance_m, match_method='proximity'.
    """
    faa_v = faa_df.dropna(subset=["lat", "lon"]).copy()
    osm_v = osm_df.dropna(subset=["lat", "lon"]).copy()

    if exclude_faa_idx:
        faa_v = faa_v[~faa_v.index.isin(exclude_faa_idx)]

    if faa_v.empty or osm_v.empty:
        return pd.DataFrame(columns=["faa_idx", "osm_idx", "distance_m", "match_method"])

    dist = haversine_matrix(
        faa_v["lat"].values, faa_v["lon"].values,
        osm_v["lat"].values, osm_v["lon"].values,
    )
    nn_idx  = dist.argmin(axis=1)
    nn_dist = dist[np.arange(len(faa_v)), nn_idx]

    mask = nn_dist <= threshold_m
    result = pd.DataFrame({
        "faa_idx":    faa_v.index[mask],
        "osm_idx":    osm_v.iloc[nn_idx[mask]].index.values,
        "distance_m": nn_dist[mask],
        "match_method": "proximity",
    }).reset_index(drop=True)

    log.info("Proximity matches (≤%.0fm): %d / %d FAA records",
             threshold_m, len(result), len(faa_v))
    return result


def match_rate_by_threshold(
    faa_df: pd.DataFrame,
    osm_df: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute FAA→OSM match rate at several distance thresholds.

    Args:
        faa_df: FAA DataFrame with lat/lon.
        osm_df: OSM DataFrame with lat/lon.
        thresholds: Distance thresholds in metres. Defaults to standard set.

    Returns:
        DataFrame with columns: threshold_m, matched, total, pct.
    """
    if thresholds is None:
        # Based on 1.5 × FATO diameter of design helicopter class (ft → m):
        #   Small R22   1.5 × 50 ft FATO  ≈  23 m
        #   Hospital    1.5 × 60 ft FATO  ≈  27 m
        #   Medium 206  1.5 × 70 ft FATO  ≈  32 m
        #   Large  S-92 1.5 × 175 ft FATO ≈  80 m
        thresholds = [23, 27, 32, 50, 80, 100, 150]

    faa_v = faa_df.dropna(subset=["lat", "lon"])
    osm_v = osm_df.dropna(subset=["lat", "lon"])

    dist = haversine_matrix(
        faa_v["lat"].values, faa_v["lon"].values,
        osm_v["lat"].values, osm_v["lon"].values,
    )
    min_dist = dist.min(axis=1)

    rows = []
    for thr in thresholds:
        n = (min_dist <= thr).sum()
        rows.append({
            "threshold_m": thr,
            "matched": int(n),
            "total": len(faa_v),
            "pct": round(n / len(faa_v) * 100, 1),
        })
    return pd.DataFrame(rows)


# ── consistency table ─────────────────────────────────────────────────────────

# Matches trailing helipad-related words in OSM names (e.g. "…Helipad", "…Heliport")
_HELI_SUFFIX_RE = re.compile(
    r"\s+(heliport|helipad|heli\s*pad|helicopter\s+(?:landing\s+)?pad|helideck)s?\s*$",
    re.IGNORECASE,
)


def _name_sim(faa_name: str, osm_name: str) -> float | None:
    """SequenceMatcher ratio between FAA and OSM names (case-insensitive).

    OSM contributors frequently append "Helipad" / "Heliport" to names that
    the FAA records without that suffix (e.g. OSM "NYU Hospital Helipad" vs
    FAA "NYU HOSPITAL").  The suffix is stripped from the OSM name before
    comparison *unless* the FAA name also contains a helipad keyword, so that
    facilities whose official name includes "Helipad" still score correctly.
    """
    a = str(faa_name or "").strip().lower()
    b = str(osm_name or "").strip().lower()
    if not a or not b or a == "nan" or b == "nan":
        return None
    b_stripped = _HELI_SUFFIX_RE.sub("", b).strip()
    if b_stripped and not _HELI_SUFFIX_RE.search(a):
        b = b_stripped
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)


def build_consistency_table(
    faa_df: pd.DataFrame,
    osm_df: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Build a row-per-match consistency table with coordinate, elevation, and name checks.

    Elevation unit detection:
        OSM ``ele`` is officially in metres (OSM wiki standard).
        This function converts to feet (× 3.28084) and computes delta vs FAA.
        A secondary "raw" delta (no conversion) is also computed so the caller
        can flag records where the raw value is suspiciously close to FAA ft
        (suggesting the OSM contributor accidentally recorded feet).

    Args:
        faa_df: FAA DataFrame.
        osm_df: OSM DataFrame.
        matches: Output of match_by_faa_id or match_by_proximity.

    Returns:
        DataFrame with one row per matched pair and the following columns:
        faa_idx, osm_idx, match_method, distance_m,
        faa_name, osm_name, name_similarity,
        faa_lat, osm_lat, lat_delta,
        faa_lon, osm_lon, lon_delta,
        faa_state, osm_state, state_match,
        faa_elev_ft, osm_ele_raw, osm_elev_ft_converted,
        elev_delta_ft, elev_plausible,
        osm_ele_likely_feet.
    """
    if matches.empty:
        return pd.DataFrame()

    rows = []
    for _, m in matches.iterrows():
        fi = m["faa_idx"]
        oi = m["osm_idx"]
        fr = faa_df.loc[fi]
        or_ = osm_df.loc[oi]

        # ── coordinates ──
        faa_lat = float(fr.get("lat", np.nan))
        faa_lon = float(fr.get("lon", np.nan))
        osm_lat = float(or_.get("lat", np.nan))
        osm_lon = float(or_.get("lon", np.nan))

        # ── location distance (may differ from match distance for id matches) ──
        dist_m = m.get("distance_m", np.nan)
        if pd.isna(dist_m) and not (np.isnan(faa_lat) or np.isnan(osm_lat)):
            d = haversine_matrix(
                np.array([faa_lat]), np.array([faa_lon]),
                np.array([osm_lat]), np.array([osm_lon]),
            )
            dist_m = float(d[0, 0])

        # ── elevation ──
        faa_elev = pd.to_numeric(fr.get("ELEVATION"), errors="coerce")
        osm_ele_raw = pd.to_numeric(or_.get("ele"), errors="coerce")
        osm_elev_converted = osm_ele_raw * _FT_PER_M if pd.notna(osm_ele_raw) else np.nan
        elev_delta = faa_elev - osm_elev_converted if pd.notna(faa_elev) and pd.notna(osm_elev_converted) else np.nan

        # Flag records where treating ele as feet gives a closer match than metres
        osm_ele_likely_feet = False
        if pd.notna(faa_elev) and pd.notna(osm_ele_raw):
            delta_if_metres = abs(faa_elev - osm_ele_raw * _FT_PER_M)
            delta_if_feet   = abs(faa_elev - osm_ele_raw)
            osm_ele_likely_feet = bool(delta_if_feet < delta_if_metres)

        # ── names ──
        faa_name = str(fr.get("NAME", "") or "").strip()
        osm_name = str(or_.get("name", "") or "").strip()
        sim = _name_sim(faa_name, osm_name)

        # ── states ──
        faa_state = str(fr.get("STATE", "") or "").strip().upper()
        osm_state = str(or_.get("addr:state", "") or "").strip().upper()
        state_match = (faa_state == osm_state) if (faa_state and osm_state and osm_state != "NAN") else None

        rows.append({
            "faa_idx":               fi,
            "osm_idx":               oi,
            "match_method":          m.get("match_method", "proximity"),
            "distance_m":            round(dist_m, 1) if not np.isnan(dist_m) else np.nan,
            "faa_name":              faa_name,
            "osm_name":              osm_name if osm_name and osm_name != "nan" else "",
            "name_similarity":       sim,
            "faa_lat":               faa_lat,
            "osm_lat":               osm_lat,
            "lat_delta":             round(faa_lat - osm_lat, 6) if not np.isnan(faa_lat + osm_lat) else np.nan,
            "faa_lon":               faa_lon,
            "osm_lon":               osm_lon,
            "lon_delta":             round(faa_lon - osm_lon, 6) if not np.isnan(faa_lon + osm_lon) else np.nan,
            "faa_state":             faa_state,
            "osm_state":             osm_state if osm_state != "NAN" else "",
            "state_match":           state_match,
            "faa_elev_ft":           faa_elev,
            "osm_ele_raw":           osm_ele_raw,
            "osm_elev_ft_converted": round(osm_elev_converted, 1) if pd.notna(osm_elev_converted) else np.nan,
            "elev_delta_ft":         round(elev_delta, 1) if pd.notna(elev_delta) else np.nan,
            "elev_plausible":        bool(abs(elev_delta) <= 100) if pd.notna(elev_delta) else None,
            "osm_ele_likely_feet":   osm_ele_likely_feet,
        })

    df = pd.DataFrame(rows)
    log.info("Consistency table built: %d matched pairs", len(df))
    return df


# ── completeness helpers ──────────────────────────────────────────────────────

def faa_completeness(faa_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column null % for FAA data, sorted descending.

    Args:
        faa_df: Raw FAA DataFrame.

    Returns:
        DataFrame with columns: field, null_pct, non_null, total.
    """
    total = len(faa_df)
    rows = [
        {
            "field": col,
            "null_pct": round(faa_df[col].isna().mean() * 100, 1),
            "non_null": int(faa_df[col].notna().sum()),
            "total": total,
        }
        for col in faa_df.columns
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("null_pct", ascending=False)
        .reset_index(drop=True)
    )


def osm_completeness(
    osm_df: pd.DataFrame,
    key_fields: list[str] | None = None,
) -> pd.DataFrame:
    """Return per-field completeness stats for OSM data.

    Args:
        osm_df: Raw OSM DataFrame.
        key_fields: Subset of columns to report. Defaults to curated list.

    Returns:
        DataFrame with columns: field, pct_present, count, total.
    """
    if key_fields is None:
        key_fields = [
            "name", "surface", "ele", "lit", "operator",
            "access", "addr:state", "faa", "icao",
            "operator:type", "aerodrome:type",
        ]
    key_fields = [f for f in key_fields if f in osm_df.columns]
    total = len(osm_df)
    rows = [
        {
            "field": f,
            "pct_present": round(osm_df[f].notna().mean() * 100, 1),
            "count": int(osm_df[f].notna().sum()),
            "total": total,
        }
        for f in key_fields
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("pct_present", ascending=False)
        .reset_index(drop=True)
    )
