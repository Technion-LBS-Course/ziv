"""SkyRoute Helipad Intelligence Engine — EDA Dashboard.

Run:
    streamlit run app.py
"""

# Windows: numpy / PyTorch / XGBoost each ship libiomp5md.dll → OMP Error #15.
# Must be set before any of those libraries are imported.
import base64
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load .env BEFORE any os.getenv() calls so API keys are available.
# .env is gitignored — never committed. See .env.example for required keys.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — keys must be set in the environment directly

# Suppress "Tried to instantiate class '__path__._path'" log noise.
# Streamlit's file-watcher inspects every sys.modules entry's __path__; when it
# reaches torch.classes it triggers a C++ path object that cannot be instantiated
# in Python.  Replacing __path__ with an empty list stops the inspection without
# affecting any registered C++ classes (they are already loaded at this point).
try:
    import torch
    torch.classes.__path__ = []  # type: ignore[assignment]
except Exception:
    pass

import json
import logging
from pathlib import Path

import folium
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import MarkerCluster, MeasureControl, HeatMap
from streamlit_folium import st_folium
import streamlit.components.v1 as components

from src.analysis import (
    build_consistency_table,
    faa_completeness,
    haversine_matrix,
    match_by_faa_id,
    match_by_proximity,
    match_rate_by_threshold,
    osm_completeness,
)
from src.hie import (
    bbox_px_to_bounds,
    bbox_px_to_latlon,
    compute_offset_m,
    detect_yolo,
    draw_detection,
    fetch_esri_chip,
    fetch_naip_chip,
    load_chip,
    load_yolo_model,
    YOLO_MODEL_PATH,
)
from src.notam import fetch_active_tfrs, tfrs_to_geojson, fetch_metar
from src.weather import (
    get_nws_wms_kwargs,
    PRECIP_THRESHOLDS,
)
from src.agent import run_agent, run_agent_v2, run_booking, is_booking_intent

log = logging.getLogger(__name__)

# ── live data helpers (TTL-cached so map rebuilds don't hammer external APIs) ──

@st.cache_data(ttl=300, show_spinner=False)
def _get_active_tfrs() -> list:
    try:
        return fetch_active_tfrs()
    except Exception:
        log.warning("TFR fetch failed; returning empty list", exc_info=True)
        return []

@st.cache_data(ttl=300, show_spinner=False)
def _get_metar_bbox(lat_min: float, lon_min: float, lat_max: float, lon_max: float):
    try:
        r = requests.get(
            "https://aviationweather.gov/api/data/metar",
            params={"bbox": f"{lat_min},{lon_min},{lat_max},{lon_max}", "format": "json"},
            timeout=8,
            headers={"User-Agent": "SkyRoute/1.0"},
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

# ── constants ──────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent / "data"
ASSETS_DIR      = Path(__file__).parent / "assets"
FAA_PATH        = DATA_DIR / "faa_helipads_raw.csv"
OSM_PATH        = DATA_DIR / "osm_helipads_raw.csv"
BOOKMARKS_PATH  = DATA_DIR / "bookmarks.json"

_GROUNDING_DINO_IMG = ASSETS_DIR / "helipad_grounding_dino.jpg"

_FAA_COLOR = "#1565C0"
_OSM_COLOR = "#E65100"
_MAP_CENTER = [40.8, -75.5]
_MAP_ZOOM   = 7

# Proximity thresholds derived from 1.5 × FATO for each helicopter class.
# FATO values (ft → m): R22 50 ft, hospital 60 ft, Bell 206 70 ft, S-92 175 ft.
_THRESHOLD_PRESETS: dict[str, int] = {
    "Small — R22 class       (23 m = 1.5 × 50 ft FATO)":  23,
    "Hospital rooftop        (27 m = 1.5 × 60 ft FATO)":  27,
    "Medium — Bell 206       (32 m = 1.5 × 70 ft FATO)":  32,
    "Large — S-92 class      (80 m = 1.5 × 175 ft FATO)": 80,
    "Custom": 0,
}
_PRESET_LABELS = list(_THRESHOLD_PRESETS.keys())

# ── bookmarks ─────────────────────────────────────────────────────────────────

def _load_bookmarks() -> list[dict]:
    if BOOKMARKS_PATH.exists():
        try:
            return json.loads(BOOKMARKS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_bookmarks(bookmarks: list[dict]) -> None:
    BOOKMARKS_PATH.write_text(
        json.dumps(bookmarks, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _on_bookmark_select() -> None:
    """on_change callback: load selected bookmark into session state."""
    sel = st.session_state.get("_bk_select", "")
    if not sel or sel == "— select —":
        return
    bks = _load_bookmarks()
    bk = next((b for b in bks if b["name"] == sel), None)
    if bk:
        st.session_state["_bk_lat"]  = bk["lat"]
        st.session_state["_bk_lon"]  = bk["lon"]
        st.session_state["_bk_zoom"] = bk.get("zoom", 15)
        st.session_state["_bk_ver"]  = st.session_state.get("_bk_ver", 0) + 1


# ── search ─────────────────────────────────────────────────────────────────────

def _search_helipads(
    query: str,
    faa_df: pd.DataFrame,
    osm_df: pd.DataFrame,
    max_results: int = 30,
) -> list[dict]:
    """Return up to max_results FAA + OSM records matching query (name or ID)."""
    q = query.strip().lower()
    results: list[dict] = []

    # FAA: match IDENT (exact prefix) or NAME (contains)
    faa_v = faa_df.dropna(subset=["lat", "lon"])
    ident_col = "IDENT" if "IDENT" in faa_v.columns else None
    name_col  = "NAME"  if "NAME"  in faa_v.columns else None
    city_col  = "SERVCITY" if "SERVCITY" in faa_v.columns else None

    masks = []
    if ident_col:
        masks.append(faa_v[ident_col].str.upper().str.contains(q.upper(), na=False))
    if name_col:
        masks.append(faa_v[name_col].str.lower().str.contains(q, na=False))
    if city_col:
        masks.append(faa_v[city_col].str.lower().str.contains(q, na=False))

    if masks:
        faa_mask = masks[0]
        for m in masks[1:]:
            faa_mask = faa_mask | m
        for _, row in faa_v[faa_mask].head(max_results).iterrows():
            ident = str(row.get(ident_col, "?") or "?").strip()
            name  = str(row.get(name_col,  "?") or "?").strip()
            results.append({
                "label":  f"FAA  {ident}  —  {name}",
                "lat":    float(row["lat"]),
                "lon":    float(row["lon"]),
                "source": "faa",
            })

    # OSM: match name or faa tag
    osm_v = osm_df.dropna(subset=["lat", "lon"])
    osm_masks = []
    if "name" in osm_v.columns:
        osm_masks.append(osm_v["name"].str.lower().str.contains(q, na=False))
    if "faa" in osm_v.columns:
        osm_masks.append(osm_v["faa"].str.upper().str.contains(q.upper(), na=False))

    if osm_masks:
        osm_mask = osm_masks[0]
        for m in osm_masks[1:]:
            osm_mask = osm_mask | m
        for _, row in osm_v[osm_mask].head(max_results).iterrows():
            osm_name = str(row.get("name", "") or "Unnamed").strip()
            osm_faa  = str(row.get("faa",  "") or "").strip()
            tag = f" [{osm_faa}]" if osm_faa else ""
            results.append({
                "label":  f"OSM  {osm_name}{tag}",
                "lat":    float(row["lat"]),
                "lon":    float(row["lon"]),
                "source": "osm",
            })

    return results[:max_results]


@st.cache_data
def build_search_entries(
    faa_df: pd.DataFrame, osm_df: pd.DataFrame
) -> list[dict]:
    """Pre-build all searchable entries for the autocomplete selectbox.

    Returns a flat list of dicts with keys: label, lat, lon, source.
    Includes FAA helipads, all OSM helipads (named and unnamed), and
    business / residential demand points.
    """
    entries: list[dict] = []

    # ── FAA helipads ──────────────────────────────────────────────────────────
    faa_v = faa_df.dropna(subset=["lat", "lon"])
    ident_col = "IDENT" if "IDENT" in faa_v.columns else None
    name_col  = "NAME"  if "NAME"  in faa_v.columns else None
    for _, row in faa_v.iterrows():
        ident = str(row.get(ident_col, "") or "").strip() if ident_col else ""
        name  = str(row.get(name_col,  "") or "").strip() if name_col  else ""
        if not (ident or name):
            continue
        label = f"FAA  {ident}  —  {name}" if ident and name else f"FAA  {ident or name}"
        entries.append({"label": label, "lat": float(row["lat"]),
                        "lon": float(row["lon"]), "source": "faa"})

    # ── OSM helipads (all, including unnamed) ─────────────────────────────────
    osm_v = osm_df.dropna(subset=["lat", "lon"])
    for _, row in osm_v.iterrows():
        osm_name = str(row.get("name", "") or "").strip()
        faa_tag  = str(row.get("faa",  "") or "").strip()
        osm_id   = str(row.get("osm_id", "") or "").strip()
        if osm_name:
            label = f"OSM  {osm_name}" + (f"  [{faa_tag}]" if faa_tag else "")
        elif faa_tag:
            label = f"OSM  (unnamed)  [{faa_tag}]"
        elif osm_id:
            label = f"OSM  #{osm_id}  ({float(row['lat']):.4f}, {float(row['lon']):.4f})"
        else:
            label = f"OSM  ({float(row['lat']):.4f}, {float(row['lon']):.4f})"
        entries.append({"label": label, "lat": float(row["lat"]),
                        "lon": float(row["lon"]), "source": "osm"})

    # ── Demand points: business centres + executive residences ────────────────
    _SEARCH_POIS = [
        {"lat": 40.7589, "lng": -73.9851, "name": "Midtown Manhattan",             "cat": "biz"},
        {"lat": 40.7127, "lng": -74.0059, "name": "Financial District (Wall St)",  "cat": "biz"},
        {"lat": 40.7504, "lng": -73.9967, "name": "Hudson Yards",                  "cat": "biz"},
        {"lat": 40.7531, "lng": -73.9772, "name": "Grand Central / Park Ave",      "cat": "biz"},
        {"lat": 41.0253, "lng": -73.6282, "name": "Greenwich, CT",                 "cat": "biz"},
        {"lat": 41.0534, "lng": -73.5387, "name": "Stamford, CT",                  "cat": "biz"},
        {"lat": 41.1220, "lng": -73.7949, "name": "White Plains, NY",              "cat": "biz"},
        {"lat": 40.7736, "lng": -73.9566, "name": "Upper East Side",               "cat": "home"},
        {"lat": 40.7870, "lng": -73.9754, "name": "Upper West Side",               "cat": "home"},
        {"lat": 40.7195, "lng": -74.0089, "name": "Tribeca",                       "cat": "home"},
        {"lat": 40.9176, "lng": -73.8282, "name": "Bronxville, NY",                "cat": "home"},
        {"lat": 40.9895, "lng": -73.7776, "name": "Scarsdale, NY",                 "cat": "home"},
        {"lat": 41.0253, "lng": -73.6282, "name": "Greenwich, CT (res.)",          "cat": "home"},
        {"lat": 40.9799, "lng": -73.6876, "name": "Rye, NY",                       "cat": "home"},
    ]
    for poi in _SEARCH_POIS:
        icon = "🏢" if poi["cat"] == "biz" else "🏠"
        entries.append({"label": f"{icon} POI  {poi['name']}",
                        "lat": poi["lat"], "lon": poi["lng"], "source": "poi"})

    return entries


# ── page ───────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyRoute HIE",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Compact sidebar: tighten dividers and element spacing */
    section[data-testid="stSidebar"] hr {
        margin-top: 0.35rem !important;
        margin-bottom: 0.35rem !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        margin-bottom: 0.1rem !important;
    }
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 0.15rem !important;
    }
    section[data-testid="stSidebar"] .stCaption {
        margin-top: 0 !important;
        margin-bottom: 0.1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw FAA and OSM CSV files."""
    for p in (FAA_PATH, OSM_PATH):
        if not p.exists():
            st.error(f"Missing: {p}  →  run: python scripts/fetch_ny_data.py")
            st.stop()
    return pd.read_csv(FAA_PATH), pd.read_csv(OSM_PATH)


@st.cache_data
def compute_matches(faa_df: pd.DataFrame, osm_df: pd.DataFrame,
                    threshold_m: float) -> pd.DataFrame:
    """Combine FAA-ID exact matches + proximity matches into one table."""
    id_matches   = match_by_faa_id(faa_df, osm_df)
    prox_matches = match_by_proximity(
        faa_df, osm_df, threshold_m=threshold_m,
        exclude_faa_idx=set(id_matches["faa_idx"]) if not id_matches.empty else None,
    )
    combined = pd.concat([id_matches, prox_matches], ignore_index=True)
    return build_consistency_table(faa_df, osm_df, combined)


@st.cache_data
def compute_threshold_curve(faa_df: pd.DataFrame,
                             osm_df: pd.DataFrame) -> pd.DataFrame:
    """Match rate vs distance threshold (cached separately — expensive)."""
    return match_rate_by_threshold(faa_df, osm_df)


@st.cache_data
def _load_osm_validated() -> pd.DataFrame | None:
    """Load osm_validated.csv produced by scripts/validate_osm_only.py."""
    path = DATA_DIR / "osm_validated.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in ("lat", "lon", "hie_confidence", "hie_offset_m"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["hie_visual_detected"] = df["hie_visual_detected"].astype(bool)
    df["osm_id"] = df["osm_id"].astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


@st.cache_data(ttl=3_600)
def fetch_imagery_meta(lat: float, lon: float) -> dict | None:
    """Query ESRI World Imagery identify endpoint for source/date at a point.

    Coordinates are rounded to 3 dp (~110 m) before caching so nearby
    pan movements don't trigger redundant requests.
    """
    delta = 0.005
    try:
        r = requests.get(
            "https://services.arcgisonline.com/ArcGIS/rest/services"
            "/World_Imagery/MapServer/identify",
            params={
                "f":             "json",
                "geometry":      f'{{"x":{lon},"y":{lat},'
                                  '"spatialReference":{"wkid":4326}}',
                "geometryType":  "esriGeometryPoint",
                "sr":            "4326",
                "layers":        "all",
                "tolerance":     "2",
                "mapExtent":     f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}",
                "imageDisplay":  "800,600,96",
                "returnGeometry": "false",
            },
            timeout=5,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0].get("attributes") if results else None
    except Exception:
        return None


faa_raw, osm_raw = load_data()

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚁 SkyRoute HIE")
    st.caption("Helipad Intelligence Engine")
    st.divider()

    # Jump-target indicator — shown whenever a location has been set
    _active_lat = st.session_state.get("_bk_lat")
    _active_lon = st.session_state.get("_bk_lon")
    if _active_lat and _active_lon:
        st.markdown(
            "<div style='background:#0a2e14;border-left:3px solid #22c55e;"
            "border-radius:5px;padding:6px 10px;font-size:11px;margin-bottom:4px'>"
            "<span style='color:#22c55e'>📍 Jump target set</span><br>"
            f"<span style='color:#86efac'>{_active_lat:.4f} N, {abs(_active_lon):.4f} W</span><br>"
            "<span style='color:#4ade80;font-weight:600'>→ Open 📊 EDA &amp; HIE tab to navigate</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Search / autocomplete ─────────────────────────────────────────────────
    st.markdown("**Search** (FAA · OSM · POIs)")
    _ac_entries = build_search_entries(faa_raw, osm_raw)
    _ac_labels  = ["— type to search —"] + [e["label"] for e in _ac_entries]
    _ac_sel = st.selectbox(
        "Name or ID",
        options=_ac_labels,
        key="_ac_select",
        label_visibility="collapsed",
        help=f"{len(_ac_entries):,} entries indexed (FAA helipads, OSM helipads, business/residential POIs). "
             "Type IDENT (e.g. 7NY7), name, or location to filter.",
    )
    # Navigate only when the user picks a real entry and it has changed
    if _ac_sel and _ac_sel != "— type to search —":
        if st.session_state.get("_ac_last") != _ac_sel:
            st.session_state["_ac_last"] = _ac_sel
            _ac_hit = next((e for e in _ac_entries if e["label"] == _ac_sel), None)
            if _ac_hit:
                _new_ver = st.session_state.get("_bk_ver", 0) + 1
                st.session_state["_bk_lat"]        = _ac_hit["lat"]
                st.session_state["_bk_lon"]        = _ac_hit["lon"]
                st.session_state["_bk_zoom"]       = 19
                st.session_state["_bk_ver"]        = _new_ver
                st.session_state["_use_satellite"] = True
                st.session_state["_satellite_ver"] = _new_ver
    st.divider()

    all_states = sorted(faa_raw["STATE"].dropna().unique().tolist())
    sel_states: list[str] = st.multiselect(
        "Filter states", options=all_states, default=all_states,
        help="Filters FAA. OSM shown for full bbox (limited state tags).",
    )
    st.divider()

    # ── Bookmarks ──────────────────────────────────────────────────────────────
    st.markdown("**Bookmarks**")
    _bks = _load_bookmarks()
    _bk_options = ["— select —"] + [b["name"] for b in _bks]
    st.selectbox(
        "Jump to",
        options=_bk_options,
        key="_bk_select",
        on_change=_on_bookmark_select,
        help="Select a saved location to fly the map there.",
    )

    # Show note for active bookmark
    _active = st.session_state.get("_bk_select", "")
    if _active and _active != "— select —":
        _active_bk = next((b for b in _bks if b["name"] == _active), None)
        if _active_bk and _active_bk.get("note"):
            st.caption(_active_bk["note"])
        if st.button("Delete bookmark", key="_bk_del"):
            _save_bookmarks([b for b in _bks if b["name"] != _active])
            st.session_state.pop("_bk_select", None)
            st.rerun()

    st.divider()

    # ── Save current view ──────────────────────────────────────────────────────
    st.markdown("**Save current view**")
    _last_center = st.session_state.get("_last_center", {})
    _hint = (
        f"{_last_center.get('lat', 0):.4f} N, "
        f"{_last_center.get('lng', 0):.4f} W  "
        f"zoom {st.session_state.get('_last_zoom', '?')}"
        if _last_center else "Pan or zoom the map first"
    )
    st.caption(_hint)
    _bk_name = st.text_input("Bookmark name", placeholder="e.g. 30th St mismatch",
                              key="_bk_name_input")
    _bk_note = st.text_input("Note (optional)", placeholder="FAA elev wrong?",
                              key="_bk_note_input")
    if st.button("Save bookmark", disabled=not (_bk_name and _last_center),
                 key="_bk_save"):
        _new = {
            "name": _bk_name,
            "lat":  _last_center.get("lat", _MAP_CENTER[0]),
            "lon":  _last_center.get("lng", _MAP_CENTER[1]),
            "zoom": st.session_state.get("_last_zoom", 14),
            "note": _bk_note,
        }
        _all = [b for b in _load_bookmarks() if b["name"] != _bk_name]
        _all.append(_new)
        _save_bookmarks(_all)
        st.success(f"Saved: {_bk_name}")

    st.divider()
    st.caption(f"**FAA** ADDS-ArcGIS · {len(faa_raw):,} records")
    st.caption(f"**OSM** Overpass API · {len(osm_raw):,} records")
    st.caption("Coverage: NY · NJ · CT · PA · MA")

# ── filtered views ─────────────────────────────────────────────────────────────
faa = faa_raw[faa_raw["STATE"].isin(sel_states)].copy() if sel_states else faa_raw.copy()
osm = osm_raw.copy()

# Default proximity threshold: Large helicopter (S-92 class), 1.5 × 175 ft FATO ≈ 80 m
prox_threshold: int = 80

# Pre-compute matches here (cached) so both the sidebar filter and the
# Consistency tab share the same result without a second expensive call.
cons = compute_matches(faa_raw, osm_raw, threshold_m=prox_threshold)

# ── sidebar (part 2) — analysis filters ────────────────────────────────────────
with st.sidebar:
    if not cons.empty:
        st.divider()
        st.markdown("**Analysis Filters**")
        st.caption(
            "Investigate FAA↔OSM matched pairs. Filter by match quality, then pick a pair "
            "from the dropdown to jump the map to that location (zoom 19, satellite view)."
        )
        with st.expander("Filter matched pairs", expanded=False):
            _af_method = st.selectbox(
                "Match method",
                ["All", "FAA-ID exact", "Proximity only"],
                key="_af_method",
            )
            _has_dist = cons["distance_m"].notna().any()
            if _has_dist:
                # Cap slider to proximity-only pairs; FAA-ID matches can have huge
                # distances (different coordinates, same IDENT) and would skew the scale.
                _prox_dists = cons.loc[
                    cons["match_method"] == "proximity", "distance_m"
                ].dropna()
                _dist_max = float(
                    max(_prox_dists.max() if not _prox_dists.empty else prox_threshold,
                        float(prox_threshold))
                )
                _af_dist = st.slider(
                    "Distance range (m)",
                    min_value=0.0, max_value=_dist_max,
                    value=(0.0, float(prox_threshold)),
                    step=1.0, key="_af_dist",
                )
            _has_sim = cons["name_similarity"].notna().any()
            if _has_sim:
                _af_sim = st.slider(
                    "Name similarity range",
                    min_value=0.0, max_value=1.0,
                    value=(0.0, 1.0),
                    step=0.05, key="_af_sim",
                )
            _has_elev = cons["elev_delta_ft"].notna().any()
            if _has_elev:
                _elev_max = float(max(cons["elev_delta_ft"].abs().max(), 500.0))
                _af_elev = st.slider(
                    "Max |elevation delta| (ft)",
                    min_value=0.0, max_value=_elev_max,
                    value=_elev_max, step=10.0, key="_af_elev",
                )
            _af_state_only = st.checkbox("State mismatch only", key="_af_state")
            _af_feet_only  = st.checkbox("OSM ele likely in feet", key="_af_feet")

        # Apply filters
        _af_filtered = cons.copy()
        if _af_method == "FAA-ID exact":
            _af_filtered = _af_filtered[_af_filtered["match_method"] == "faa_id"]
        elif _af_method == "Proximity only":
            _af_filtered = _af_filtered[_af_filtered["match_method"] == "proximity"]
        if _has_dist:
            lo, hi = _af_dist
            _af_filtered = _af_filtered[
                _af_filtered["distance_m"].isna() |
                ((_af_filtered["distance_m"] >= lo) & (_af_filtered["distance_m"] <= hi))
            ]
        if _has_sim:
            lo_s, hi_s = _af_sim
            _af_filtered = _af_filtered[
                _af_filtered["name_similarity"].isna() |
                ((_af_filtered["name_similarity"] >= lo_s) & (_af_filtered["name_similarity"] <= hi_s))
            ]
        if _has_elev:
            _af_filtered = _af_filtered[
                _af_filtered["elev_delta_ft"].isna() |
                (_af_filtered["elev_delta_ft"].abs() <= _af_elev)
            ]
        if _af_state_only:
            _af_filtered = _af_filtered[_af_filtered["state_match"] == False]  # noqa: E712
        if _af_feet_only:
            _af_filtered = _af_filtered[_af_filtered["osm_ele_likely_feet"] == True]  # noqa: E712

        st.caption(f"{len(_af_filtered):,} / {len(cons):,} pairs match filters")

        # Build labelled entries for the dropdown
        _af_entries: list[dict] = []
        for _, _r in _af_filtered.iterrows():
            fn   = str(_r.get("faa_name") or "—")[:22]
            on   = str(_r.get("osm_name") or "—")[:22]
            dist = f"{_r['distance_m']:.0f}m" if pd.notna(_r.get("distance_m")) else "—"
            sim  = f"s={_r['name_similarity']:.2f}" if pd.notna(_r.get("name_similarity")) else ""
            lbl  = f"{fn}  ↔  {on}  [{dist}{'  ' + sim if sim else ''}]"
            _af_entries.append({
                "label": lbl,
                "lat":   _r["faa_lat"],
                "lon":   _r["faa_lon"],
            })

        if _af_entries:
            _af_labels = ["— select pair —"] + [e["label"] for e in _af_entries]
            _af_sel = st.selectbox(
                "Jump to pair",
                options=_af_labels,
                key="_af_select",
                label_visibility="collapsed",
                help="Select a matched pair to jump to its FAA location on the map.",
            )
            if _af_sel and _af_sel != "— select pair —":
                if st.session_state.get("_af_last") != _af_sel:
                    st.session_state["_af_last"] = _af_sel
                    _af_hit = next((e for e in _af_entries if e["label"] == _af_sel), None)
                    if _af_hit and pd.notna(_af_hit["lat"]) and pd.notna(_af_hit["lon"]):
                        _new_ver = st.session_state.get("_bk_ver", 0) + 1
                        st.session_state["_bk_lat"]        = float(_af_hit["lat"])
                        st.session_state["_bk_lon"]        = float(_af_hit["lon"])
                        st.session_state["_bk_zoom"]       = 19
                        st.session_state["_bk_ver"]        = _new_ver
                        st.session_state["_use_satellite"] = True
                        st.session_state["_satellite_ver"] = _new_ver
        else:
            st.info("No pairs match the current filters.")

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🚁 SkyRoute — Helipad Intelligence Engine")
st.divider()

# ── outer tabs ─────────────────────────────────────────────────────────────────
tab_problem, tab_lit, tab_market, tab_eda, tab_inspector, tab_results, tab_agent = st.tabs([
    "📍 Problem", "📚 Literature", "🏪 Market", "📊 EDA & HIE", "🔍 Inspector", "📈 Results", "💬 Route Assistant",
])


# ── TAB 1 · Problem Understanding ────────────────────────────────────────────

with tab_problem:
    st.markdown("### The SkyRoute Opportunity")
    col_persona, col_journey = st.columns([1, 2])

    with col_persona:
        st.markdown("""
<div style="background:#0d2137;border-radius:12px;padding:20px;border-left:4px solid #29b6f6">
  <div style="text-align:center;margin-bottom:4px"><img src="data:image/jpeg;base64,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" style="width:130px;height:130px;border-radius:50%;object-fit:cover;border:3px solid #29b6f6;box-shadow:0 4px 16px rgba(0,0,0,.5)"></div>
  <h3 style="color:#29b6f6;margin:8px 0 4px;text-align:center">Miles Urban</h3>
  <p style="color:#90caf9;margin:0 0 14px;text-align:center;font-size:13px">VP Business Development</p>
  <table style="width:100%;font-size:13px;color:#e3f2fd;border-collapse:collapse">
    <tr><td style="padding:4px 0"><b>Age</b></td><td>44</td></tr>
    <tr><td style="padding:4px 0"><b>Location</b></td><td>Bronxville, NY</td></tr>
    <tr><td style="padding:4px 0"><b>Travel</b></td><td>4-5 x /week</td></tr>
    <tr><td style="padding:4px 0"><b>Payment</b></td><td>Corporate Amex - no spending cap</td></tr>
    <tr><td style="padding:4px 0"><b>Priority</b></td><td>Time &gt; Cost</td></tr>
  </table>
  <hr style="border-color:#1e3a5f;margin:14px 0">
  <p style="font-size:16px;font-weight:800;color:#80cbc4;font-style:italic;margin:0;line-height:1.4;text-align:center">
    "I don't need cheaper. I need faster and reliable."
  </p>
</div>
""", unsafe_allow_html=True)

    with col_journey:
        st.markdown("#### Door-to-Door Journey Comparison")
        jt_before, jt_after = st.tabs(["Without SkyRoute", "With SkyRoute"])
        with jt_before:
            st.markdown("""
| Step | Mode | Time | Pain Point |
|------|------|-----:|------------|
| Midtown to Penn Station | Walk | 12 min | Arrives rushed |
| Amtrak NYC to Stamford | Train | 55 min | Frequent delays |
| Taxi Stamford to Greenwich | Car | 25 min | Traffic unpredictable |
| **Total** | **3 apps, 3 bookings** | **~92 min** | **No single view** |
""")
        with jt_after:
            st.markdown("""
| Step | Mode | Time | Advantage |
|------|------|-----:|-----------|
| Walk to 30th St Heliport | Walk | 6 min | 10 city blocks |
| Helicopter NYC to Westchester | Air | 18 min | Above the traffic |
| Car Westchester to Greenwich | Car | 12 min | Short last mile |
| **Total** | **SkyRoute - 1 booking** | **~36 min** | **One price, one tap** |
""")
            st.success("**Save 56 minutes per trip. 4x per week = 3.7 hours saved weekly.**")

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Time saved / trip", "56 min", delta="61% faster")
    m2.metric("Trips per week", "4-5x", delta="240+ / year")
    m3.metric("Weekly hours reclaimed", "3.7 hrs", delta="via SkyRoute")
    m4.metric("Booking friction", "1 tap", delta="-2 apps")

    st.divider()
    st.markdown("#### Stakeholder Ecosystem")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("""<div style="background:#0d2137;border-radius:8px;padding:14px;border-top:3px solid #1565C0;text-align:center">
<div style="font-size:28px">&#x1F3E2;</div><b style="color:#29b6f6">Operators</b>
<p style="font-size:12px;color:#90caf9;margin:6px 0 0">Blade · Joby · Helicopters NY</p></div>""",
            unsafe_allow_html=True)
    with s2:
        st.markdown("""<div style="background:#0d2137;border-radius:8px;padding:14px;border-top:3px solid #E65100;text-align:center">
<div style="font-size:28px">&#x1F9F3;</div><b style="color:#FF8A65">Passengers</b>
<p style="font-size:12px;color:#FFCCBC;margin:6px 0 0">Business travellers - Miles Urban archetype</p></div>""",
            unsafe_allow_html=True)
    with s3:
        st.markdown("""<div style="background:#0d2137;border-radius:8px;padding:14px;border-top:3px solid #22c55e;text-align:center">
<div style="font-size:28px">&#x2708;&#xFE0F;</div><b style="color:#4ade80">Vertiport Owners</b>
<p style="font-size:12px;color:#bbf7d0;margin:6px 0 0">Helipad owners · Port Authority</p></div>""",
            unsafe_allow_html=True)
    with s4:
        st.markdown("""<div style="background:#0d2137;border-radius:8px;padding:14px;border-top:3px solid #9c27b0;text-align:center">
<div style="font-size:28px">&#x2696;&#xFE0F;</div><b style="color:#ce93d8">Regulators</b>
<p style="font-size:12px;color:#e1bee7;margin:6px 0 0">FAA · EASA · Local airspace</p></div>""",
            unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 🤖 ML Pipeline: Helipad Intelligence Engine (HIE)")
    st.caption(
        "Raw helipad registries (FAA, OSM) are incomplete, stale, and contain military or "
        "decommissioned pads. HIE is a 3-phase pipeline that validates every candidate "
        "pad before it enters the routing engine."
    )

    # ── pipeline flow banner ────────────────────────────────────────────────────
    st.markdown("""
<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;
            font-family:monospace;font-size:12px;margin:10px 0 18px">
  <div style="background:#0d2137;border:1px solid #42A5F5;border-radius:8px;
              padding:10px 14px;color:#90caf9;min-width:120px;text-align:center">
    🛰️<br><b>Raw Input</b><br><span style="font-size:10px;color:#475569">FAA · OSM<br>747 + records</span>
  </div>
  <div style="color:#475569;font-size:18px">→</div>
  <div style="background:#0d2137;border:2px solid #a78bfa;border-radius:8px;
              padding:10px 14px;color:#c4b5fd;min-width:130px;text-align:center">
    🛰️ <b>Phase 1</b><br><span style="font-size:10px">YOLO11m<br>Visual check</span>
  </div>
  <div style="color:#475569;font-size:18px">→</div>
  <div style="background:#0d2137;border:2px solid #60a5fa;border-radius:8px;
              padding:10px 14px;color:#93c5fd;min-width:130px;text-align:center">
    🛰️ <b>Phase 2</b><br><span style="font-size:10px">YOLO OSM<br>Validation</span>
  </div>
  <div style="color:#475569;font-size:18px">→</div>
  <div style="background:#0d2137;border:2px solid #34d399;border-radius:8px;
              padding:10px 14px;color:#6ee7b7;min-width:130px;text-align:center">
    📋 <b>Phase 3</b><br><span style="font-size:10px">ADIP booking<br>enrichment</span>
  </div>
  <div style="color:#475569;font-size:18px">→</div>
  <div style="background:#071a2e;border:2px solid #22c55e;border-radius:8px;
              padding:10px 14px;color:#22c55e;min-width:130px;text-align:center;font-weight:700">
    ✅<br><b>Validated pad</b><br><span style="font-size:10px">Added to routing</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Phase 1 ─────────────────────────────────────────────────────────────────
    ph1_col, ph1_img, ph1_ex = st.columns([3, 2, 2])
    with ph1_img:
        if _GROUNDING_DINO_IMG.exists():
            st.image(
                str(_GROUNDING_DINO_IMG),
                caption="YOLO11m detection on a rooftop helipad (NAIP 100 m × 100 m chip)",
                use_column_width=True,
            )
        else:
            st.markdown(
                "<div style='background:#1a1a2e;border:1px dashed #475569;border-radius:8px;"
                "padding:20px;text-align:center;color:#475569;font-size:11px'>"
                "📷 Place satellite example image at<br>"
                "<code>assets/helipad_grounding_dino.jpg</code>"
                "</div>",
                unsafe_allow_html=True,
            )
    with ph1_col:
        st.markdown("""
**Phase 1 — Visual Validation: YOLO11m (Fine-tuned)**

For every FAA and OSM candidate pad, SkyRoute fetches a **100 m × 100 m NAIP satellite chip**
(USDA APFO ImageServer, ~0.16 m/px effective GSD) centred on the registry coordinate and runs
it through a **fine-tuned YOLO11m detector** — trained on 1,200+ annotated NAIP chips derived
from the HelipadCAT dataset with manual annotation correction.

| | |
|---|---|
| **Input** | 640 × 640 px NAIP chip centred on registry coord (~100 m × 100 m window) |
| **Model** | YOLO11m fine-tuned · P=0.931 · R=0.848 · **F1=0.888** |
| **Output** | Bounding box `[x₁,y₁,x₂,y₂]` + confidence score (threshold ≥ 0.50) |
| **Action — detected** | Centroid projected back to lat/lon → coordinate corrected if offset > 15 m |
| **Action — not found** | Pad flagged `unverified`; excluded from routing pool |
| **Metric (M3 KPI)** | F1=0.888 on 747 held-out NE US test chips vs F1=0.73 XGBoost structured baseline |
""")
    with ph1_ex:
        st.markdown("""
<div style="background:#0a1628;border:1px solid #a78bfa;border-radius:8px;padding:12px 14px;font-size:12px">
<div style="color:#a78bfa;font-weight:700;margin-bottom:8px">🛰️ YOLO11m — example inference</div>
<div style="background:#1a1a2e;border-radius:6px;padding:8px;margin-bottom:8px;font-family:monospace;font-size:11px;color:#e2e8f0">
<span style="color:#fbbf24">NAIP chip</span> · 640×640 px · 100m×100m<br>
<span style="color:#34d399">▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮</span>  ← rooftop building<br>
<span style="color:#f97316">┌──────────────┐</span>  ← YOLO11m bbox<br>
<span style="color:#f97316">│</span>  <b style="color:#fff">H</b> marking     <span style="color:#f97316">│</span>  conf: <b style="color:#22c55e">0.94</b><br>
<span style="color:#f97316">│</span>  yellow border <span style="color:#f97316">│</span><br>
<span style="color:#f97316">└──────────────┘</span><br>
<span style="color:#60a5fa">centroid</span>: 40.7236 N, 74.0482 W<br>
<span style="color:#60a5fa">registry</span>: 40.7238 N, 74.0479 W<br>
<span style="color:#22c55e">offset</span>: <b>8.3 m</b> → within tolerance ✓
</div>
<div style="color:#64748b;font-size:10px">
Typical rooftop pad — yellow safety border and H marking are
the primary detection anchors. NAIP 0.16 m/px GSD resolves
the H marking at 3–4 px height (sufficient for fine-tuned YOLO).
</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Phase 2 ─────────────────────────────────────────────────────────────────
    ph2_col, ph2_ex = st.columns([3, 2])
    with ph2_col:
        st.markdown("""
**Phase 2 — YOLO Visual Validation (OSM-only pads)**

OSM records with no FAA counterpart are the highest-value but highest-risk additions
to the routing network. Phase 2 runs the same **YOLO11m cascade** on a fresh NAIP chip
for each OSM-only pad to visually confirm the helipad exists before routing uses it.
Military and private-use pads are excluded via **FAA ADIP structured flags** (`MIL_CODE`,
`PRIVATEUSE`) — no LLM web search required.

| | |
|---|---|
| **Input** | OSM coordinates → USDA NAIP 100 m × 100 m chip |
| **Model** | YOLO11m fine-tuned (same as Phase 1) |
| **Output** | `detected` · `confidence` · `hie_offset_m` |
| **Action — detected** | Pad promoted to routing pool ✅ |
| **Action — not detected** | Pad retained on map but excluded from routing 🚫 |
| **Military / private** | Excluded via ADIP `MIL_CODE` / `PRIVATEUSE` fields |
| **Result (NE US)** | **1,174 / 1,663** OSM-only pads confirmed (70.6 %) |
""")
    with ph2_ex:
        st.markdown("""
<div style="background:#0a1628;border:1px solid #ef4444;border-radius:8px;padding:12px 14px;font-size:12px">
<div style="color:#f87171;font-weight:700;margin-bottom:8px">🚫 ADIP flag — military pad excluded</div>
<div style="background:#1a0a0a;border-radius:6px;padding:8px;margin-bottom:8px;font-family:monospace;font-size:11px;color:#e2e8f0">
<span style="color:#fbbf24">OSM pad</span>: Caven Point USAR Center Heliport<br>
<span style="color:#94a3b8">FAA IDENT</span>: NJ77<br>
<span style="color:#94a3b8">Location</span>: Jersey City, NJ<br><br>
<span style="color:#60a5fa">ADIP lookup →</span><br>
<span style="color:#e2e8f0">MIL_CODE: <b style="color:#ef4444">ARMY</b><br>
PRIVATEUSE: <b style="color:#ef4444">Y</b></span><br><br>
<span style="color:#f87171">Result</span>: <b style="color:#ef4444">EXCLUDED</b><br>
<span style="color:#94a3b8;font-size:10px">U.S. Army Reserve Center · private-use<br>
New York District Corps of Engineers<br>
→ civilian routing: EXCLUDED 🚫</span>
</div>
<div style="color:#64748b;font-size:10px">
Military and private-use flags are read directly from FAA
ADIP structured fields — no inference needed.
</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Phase 3 ─────────────────────────────────────────────────────────────────
    st.markdown("""
**Phase 3 — ADIP Booking Enrichment**

Every helicopter leg booking reads the FAA ADIP per-heliport record at runtime to surface
operational details passengers need before they land: pad status, ownership type, military
or private-use flags, inspection age, METAR flight category, and coordination instructions.
Cryptic FAA remarks (e.g. `FOR CD CTC NEW YORK APCH AT 516-683-2962`) are decoded to plain
English by the LLM before display — so passengers see actionable arrival notes, not raw
FAA notation.
""")

    adip1, adip2, adip3 = st.columns(3)
    adip1.markdown("""
<div style="background:#0a1628;border:1px solid #34d399;border-radius:8px;padding:10px 12px;font-size:12px">
<div style="color:#34d399;font-weight:700;margin-bottom:6px">🛡️ Operational Status & Flags</div>
<div style="color:#94a3b8">ADIP status (Operational / Closed / Restricted), military code, and
private-use flag — shown in the booking card and used to block routing to ineligible pads.</div>
</div>""", unsafe_allow_html=True)
    adip2.markdown("""
<div style="background:#0a1628;border:1px solid #34d399;border-radius:8px;padding:10px 12px;font-size:12px">
<div style="color:#34d399;font-weight:700;margin-bottom:6px">🧭 Coordination Notes (LLM decoded)</div>
<div style="color:#94a3b8">Raw FAA remarks decoded to plain English by LLM at booking time.
Passengers see "Contact New York Approach on 516-683-2962" — not raw FAA notation.</div>
</div>""", unsafe_allow_html=True)
    adip3.markdown("""
<div style="background:#0a1628;border:1px solid #34d399;border-radius:8px;padding:10px 12px;font-size:12px">
<div style="color:#34d399;font-weight:700;margin-bottom:6px">🌤️ METAR Flight Category</div>
<div style="color:#94a3b8">Live METAR from Aviation Weather Center — VFR / MVFR / IFR / LIFR
colour-coded in booking card and FAA map popups. No API key required.</div>
</div>""", unsafe_allow_html=True)


# ── TAB 2 · Literature Review ────────────────────────────────────────────────

with tab_lit:
    st.markdown("### Literature Review — Multi-Modal Urban Air Mobility")
    st.caption("5 peer-reviewed papers & technical reports · DOIs / source URLs verified · Summaries distilled for SkyRoute relevance")

    _PAPERS = [
        {
            "num": 1,
            "authors": "O'Reilly, P.E., Rahimi, R.A., Marques, J.L.R., & Babadopulos, M.A.F.A.L.",
            "year": 2024,
            "title": "Vertiport ventures: assessing operational feasibility for eVTOL integration in São Paulo's helipad and heliport infrastructure",
            "journal": "Journal of Marketing Analytics",
            "vol_issue": "Vol. 12, pp. 873–884",
            "doi": "10.1057/s41270-024-00323-0",
            "doi_url": "https://doi.org/10.1057/s41270-024-00323-0",
            "field": "eVTOL Infrastructure & Site Scoring",
            "summary": (
                "Evaluates whether eVTOL aircraft (4+ passengers, 50 km+ range) can be integrated into "
                "São Paulo's extensive helicopter infrastructure — the world's busiest helicopter city. "
                "Analyses site suitability of existing helipads and heliports across the metropolitan region, "
                "finding that the dense rooftop pad network provides a structurally advantageous starting point. "
                "Identifies key gaps in infrastructure dimensions, regulatory alignment, and charging logistics "
                "that must be resolved before commercial eVTOL service can launch."
            ),
            "skyroute_benefit": (
                "São Paulo's helipad-reuse playbook validates SkyRoute's HIE approach of scoring existing FAA "
                "helipads as vertiport candidates — and quantifies the operational upgrade gaps that feed "
                "the scoring model."
            ),
        },
        {
            "num": 2,
            "authors": "Zhang, Y., Yang, C., Xi, H., Peng, S., Yang, J., Gan, M., Liu, X., & Ai, R.",
            "year": 2026,
            "title": "Air-ground multimodal transport planning for joint passenger mobility and parcel delivery: integration of drones, aircraft, and ground vehicles",
            "journal": "Transportation Research Part E: Logistics and Transportation Review",
            "vol_issue": "Vol. 210, Article 104825",
            "doi": "10.1016/j.tre.2026.104825",
            "doi_url": "https://doi.org/10.1016/j.tre.2026.104825",
            "field": "Multi-Modal AAM Routing & Optimisation",
            "summary": (
                "Formulates a joint optimisation model for multimodal transport that simultaneously routes passengers "
                "and parcels using drones, fixed-wing/rotary aircraft, and ground vehicles. "
                "The model integrates air-ground transfer nodes (analogous to vertiports/helipads) with last-mile "
                "ground connections, optimising vehicle types, transfer schedules, and fleet allocation in a unified "
                "mathematical programme. "
                "Demonstrates that coordinated multi-fleet planning significantly reduces total transport time and cost "
                "compared to single-mode or sequentially planned operations, with explicit treatment of passenger "
                "access/egress legs and cargo hand-off at intermodal nodes."
            ),
            "skyroute_benefit": (
                "The joint passenger-parcel multimodal optimisation framework directly parallels SkyRoute's routing "
                "architecture — replacing drones+fixed-wing with eVTOL+helicopter and cargo transfers with "
                "rideshare/subway connections. Provides rigorous mathematical grounding for the transfer-node-based "
                "routing that HIE-validated helipads feed into, and supports the multimodal comparison table "
                "already built into the app."
            ),
        },
        {
            "num": 3,
            "authors": "Singh, R., Puhl, R.B., Dhakal, K., & Sornapudi, S.",
            "year": 2025,
            "title": "Few-Shot Adaptation of Grounding DINO for Agricultural Domain",
            "journal": "arXiv preprint",
            "vol_issue": "arXiv:2504.07252",
            "doi": "10.48550/arXiv.2504.07252",
            "doi_url": "https://doi.org/10.48550/arXiv.2504.07252",
            "field": "ML: Promptable Object Detection in Aerial Imagery",
            "summary": (
                "Adapts Grounding DINO — an open-set, text-prompted object detector — for aerial and agricultural "
                "remote-sensing imagery using few-shot learning. "
                "Removes the BERT text encoder and replaces it with a lightweight trainable text embedding, "
                "substantially reducing the model's parameter count and adaptation cost. "
                "The resulting few-shot variant achieves up to 24% higher mAP than fully fine-tuned YOLO baselines "
                "on agricultural datasets, and outperforms prior state-of-the-art by ~10% on remote-sensing "
                "object-detection benchmarks under low-data conditions — demonstrating Grounding DINO's practical "
                "viability for promptable detection in overhead imagery without large labelled datasets."
            ),
            "skyroute_benefit": (
                "Directly underpins HIE Phase 1: confirms that Grounding DINO generalises to overhead/satellite "
                "imagery via text-prompt detection without helipad-specific training data. "
                "The few-shot mechanism means a small number of verified helipad examples is sufficient to "
                "guide the detector — critical given the scarcity of labelled helipad chips in public datasets."
            ),
        },
        {
            "num": 4,
            "authors": "Eyinade, J.A., & Ademusire, A.J.",
            "year": 2025,
            "title": "GeoLLMs in action: A systematic review of multimodal models for satellite image captioning and geospatial understanding",
            "journal": "Open Access Research Journal of Science and Technology",
            "vol_issue": "Vol. 14, Issue 2, pp. 049–064",
            "doi": "10.53022/oarjst.2025.14.2.0093",
            "doi_url": "https://doi.org/10.53022/oarjst.2025.14.2.0093",
            "field": "ML: LLM for Geospatial & Satellite Understanding",
            "summary": (
                "Systematic review of 42 peer-reviewed studies (2020–2025) on multimodal large language models "
                "applied to geospatial tasks: semantic segmentation, satellite image captioning, and spatial "
                "question answering. "
                "Identifies three dominant architectural patterns: frozen vision encoders with language adapters, "
                "end-to-end fine-tuned transformers, and retrieval-augmented hybrid systems. "
                "Surfaces persistent challenges — geographic generalisation, temporal reasoning, lack of standardised "
                "benchmarks, underrepresentation of non-Western regions — and recommends geometry-aware embeddings "
                "and multilingual fine-tuning as priority future directions."
            ),
            "skyroute_benefit": (
                "Underpins HIE Phase 3: the LLM concierge decodes cryptic FAA ADIP remarks into plain-English "
                "coordination notes at booking time, and handles geocoding of business-name addresses via a "
                "retrieval-augmented cascade. The GeoLLM benchmark framework is directly applicable to "
                "evaluating the LLM's spatial reasoning quality in the Route Assistant."
            ),
        },
    ]

    # ── quick-reference table ─────────────────────────────────────────────────
    st.markdown("#### Quick Reference")
    _tbl_md = (
        "| # | Article | Field of Relevance | DOI |\n"
        "|---|---------|-------------------|-----|\n"
    )
    for _p in _PAPERS:
        _a = _p["authors"].split(",")[0] + (" et al." if "," in _p["authors"] else "")
        _tbl_md += f"| {_p['num']} | **{_a} ({_p['year']})** — {_p['title']} | {_p['field']} | [🔗 DOI]({_p['doi_url']}) |\n"
    st.markdown(_tbl_md)

    st.divider()
    st.markdown("#### Paper Summaries")

    for p in _PAPERS:
        short_auth = p["authors"].split(",")[0] + (" et al." if "," in p["authors"] else "")
        _ttl = p["title"]
        label = f"[{p['num']}]  {short_auth} ({p['year']}) — {_ttl[:65]}{'…' if len(_ttl)>65 else ''}"
        with st.expander(label):
            st.markdown(f"**{p['authors']} ({p['year']})**")
            st.markdown(f"*{p['title']}*")
            st.markdown(f"📚 **Source:** *{p['journal']}*, {p['vol_issue']}")
            st.markdown(f"🔗 **DOI:** [{p['doi']}]({p['doi_url']})")
            st.divider()
            st.markdown("**Abstract summary**")
            st.markdown(p["summary"])
            st.markdown(
                f"<div style='background:#071a2e;border-left:4px solid #22c55e;"
                f"border-radius:6px;padding:9px 14px;margin-top:10px;font-size:13px'>"
                f"<span style='color:#22c55e;font-weight:700'>✈ SkyRoute benefit:&nbsp;</span>"
                f"<span style='color:#22c55e'>{p['skyroute_benefit']}</span></div>",
                unsafe_allow_html=True,
            )


# ── TAB 3 · Market Survey ────────────────────────────────────────────────────

with tab_market:
    st.markdown("### Competitive Landscape -- Urban Air Mobility")

    _comp_df = pd.DataFrame({
        "Company":              ["Blade", "Joby Aviation", "Citymapper", "VoloIQ", "SkyRoute"],
        "Consumer Booking":     ["Yes", "Yes", "No", "No", "Yes"],
        "Multi-Modal Routing":  ["No", "Partial", "Ground only", "No", "Yes"],
        "AI Helipad Data":      ["No", "No", "No", "Partial", "Yes (HIE)"],
        "eVTOL + Helicopter":   ["Helicopter", "eVTOL only", "None", "Both", "Both"],
        "Consumer Product":     ["Yes", "Yes", "Yes", "No", "Yes"],
        "Regulatory Expertise": ["Medium", "High", "Low", "High", "High"],
    })
    st.dataframe(_comp_df.set_index("Company"), use_container_width=True)

    st.divider()
    st.markdown("#### Capability Radar -- SkyRoute vs Competitors")

    _cats = ["Booking", "Routing", "AI/ML Data", "Multi-Modal", "Regulatory"]
    _scores = {
        "SkyRoute":   [5, 5, 5, 5, 5],
        "Blade":      [5, 2, 2, 2, 3],
        "Joby":       [4, 3, 2, 3, 5],
        "Citymapper": [2, 5, 3, 4, 1],
        "VoloIQ":     [3, 2, 4, 2, 4],
    }
    _colors = {
        "SkyRoute":   "#00d4ff",
        "Blade":      "#EF5350",
        "Joby":       "#FF8A65",
        "Citymapper": "#AB47BC",
        "VoloIQ":     "#26A69A",
    }
    _fig_radar = go.Figure()
    for _name, _sc in _scores.items():
        _sc_c = _sc + [_sc[0]]
        _ct_c = _cats + [_cats[0]]
        _fig_radar.add_trace(go.Scatterpolar(
            r=_sc_c, theta=_ct_c, fill="toself", name=_name,
            line=dict(color=_colors[_name], width=3 if _name == "SkyRoute" else 1.5),
            fillcolor=_colors[_name],
            opacity=0.4 if _name == "SkyRoute" else 0.2,
        ))
    _fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        title="Capability Matrix (1 = minimal, 5 = best-in-class)",
        height=440,
    )
    st.plotly_chart(_fig_radar, use_container_width=True)

    st.divider()
    c_why1, c_why2 = st.columns(2)
    with c_why1:
        st.markdown("""
**Why competitors lose:**

| Competitor | Critical Gap |
|---|---|
| Blade | Single operator, no routing, no ML |
| Joby Aviation | eVTOL only, pre-commercial |
| Citymapper | Ground-only, zero AAM data |
| VoloIQ | Ops-side only, no consumer product |
""")
    with c_why2:
        st.markdown("""
**SkyRoute's four-way moat:**

1. Consumer booking across **multiple operators**
2. **Multi-modal routing** (eVTOL + helicopter + ground)
3. **ML-powered helipad intelligence** -- unique in market
4. **Regulatory depth** (FAA/EASA) baked into data pipeline
""")


# ── routing HTML template ──────────────────────────────────────────────────────

_ROUTING_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    html, body { width:100%; height:100%; overflow:hidden; font-family:'Segoe UI',sans-serif; }
    #map { width:100%; height:570px; }
    #aerial-bar {
      width:100%; height:54px;
      background:linear-gradient(90deg,#060b14,#0d1b2a,#060b14);
      border-top:2px solid rgba(100,116,139,0.4);
      display:flex; align-items:center; justify-content:center;
      font-size:13px; color:#94a3b8; gap:24px; padding:0 20px;
      transition:border-top-color 0.3s;
    }
    #aerial-bar.has-save { border-top-color:#22c55e; }
    #aerial-bar.no-save  { border-top-color:#475569; }
    .ab-seg { display:flex; flex-direction:column; align-items:center; gap:2px; }
    .ab-lbl { font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:.6px; }
    .ab-val { font-size:15px; font-weight:700; }
    .ab-val.ground { color:#ef4444; }
    .ab-val.air    { color:#22c55e; }
    .ab-val.dim    { color:#94a3b8; }
    .ab-div { width:1px; height:30px; background:#1e2a3a; }
    .ab-badge {
      font-size:14px; font-weight:700; color:#22c55e;
      background:rgba(34,197,94,.12); border:1px solid rgba(34,197,94,.3);
      border-radius:6px; padding:5px 14px;
    }
    .ab-neutral { font-size:13px; color:#64748b; }
    .ctrl-btn {
      background:#fff; border:2px solid rgba(0,0,0,.2); border-radius:6px;
      width:36px; height:36px; font-size:17px; cursor:pointer;
      display:flex; align-items:center; justify-content:center;
    }
    .ctrl-btn:hover { background:#f0f0f0; }
    #status-bar {
      display:none; position:absolute; top:10px; left:50%;
      transform:translateX(-50%);
      background:rgba(255,255,255,.95); border-radius:20px;
      padding:7px 18px; font-size:13px;
      z-index:999; box-shadow:0 2px 10px rgba(0,0,0,.2); white-space:nowrap;
    }
    #compare-panel {
      display:none; position:absolute; bottom:10px; left:50%;
      transform:translateX(-50%);
      background:rgba(10,12,28,.97); color:#e8e8f0; border-radius:10px;
      padding:12px 18px 10px; box-shadow:0 4px 24px rgba(0,0,0,.7);
      font-size:12px; z-index:1000; min-width:500px; max-width:720px;
    }
    #compare-panel h4 { margin:0; font-size:14px; color:#00d4ff; display:inline; }
    #compare-panel .close-x { float:right; background:none; border:none; color:#6b7280; cursor:pointer; font-size:16px; }
    #compare-panel .from-to { font-size:11px; color:#64748b; margin:4px 0 8px; }
    #compare-panel table { width:100%; border-collapse:collapse; }
    #compare-panel th { padding:4px 10px; color:#475569; border-bottom:1px solid #1e2a3a; font-size:11px; text-align:left; font-weight:500; }
    #compare-panel td { padding:5px 10px; border-bottom:1px solid #0f1623; }
    #compare-panel tr.gr-best td { background:rgba(239,68,68,.1); }
    #compare-panel tr.ai-best td { background:rgba(34,197,94,.1); }
    #compare-panel .tag { display:inline-block; border-radius:3px; padding:1px 5px; font-size:10px; margin-left:4px; }
    #compare-panel .tag-f { background:#22c55e; color:#fff; }
    #compare-panel .tag-a { background:#00d4ff; color:#0a0c1c; }
    #compare-panel .legend { margin-top:8px; font-size:11px; color:#475569; display:flex; gap:18px; flex-wrap:wrap; }
    #compare-panel .ld { display:inline-block; width:14px; height:3px; border-radius:2px; margin-right:4px; vertical-align:middle; }
    #route-info {
      display:none; position:absolute; bottom:10px; left:50%;
      transform:translateX(-50%);
      background:rgba(255,255,255,.96); border-radius:8px;
      padding:8px 16px; box-shadow:0 2px 12px rgba(0,0,0,.25);
      font-size:12px; z-index:1000; text-align:center;
    }
  </style>
</head>
<body>
<div style="position:relative">
  <div id="map"></div>
  <div id="status-bar"></div>
  <div id="compare-panel">
    <button class="close-x" onclick="clearMM()">&times;</button>
    <h4>&#x1F9ED; Multi-Modal Route Comparison</h4>
    <div class="from-to" id="cp-from-to"></div>
    <table>
      <thead><tr>
        <th>Mode</th><th>Distance</th><th>Time</th><th>Aerial Leg</th>
      </tr></thead>
      <tbody id="cp-tbody"></tbody>
    </table>
    <div class="legend">
      <span><span class="ld" style="background:#ef4444"></span>Ground route (red)</span>
      <span><span class="ld" style="background:#22c55e"></span>Drive to/from helipad (green)</span>
      <span><span class="ld" style="background:#00d4ff"></span>Helicopter flight (cyan dashed)</span>
    </div>
  </div>
  <div id="route-info">
    <strong id="ri-title"></strong>&nbsp;
    <span id="ri-dist"></span> &middot; <span id="ri-dur"></span>
    <button onclick="clearHeli()" style="margin-left:10px;background:#eee;border:none;border-radius:3px;padding:2px 8px;cursor:pointer;font-size:11px">&times; Clear</button>
  </div>
</div>
<div id="aerial-bar">
  <span style="color:#475569;font-size:12px">Click &#x1F9ED; then pick two points to compare ground vs aerial routes</span>
</div>
<script>
var RANGE_KM = 300 * 1.852;
var SPEED_HELI_KMH = 250;
var SPEED_WALK_KMH = 5;

// ── map ───────────────────────────────────────────────────────────────────────
var osmDay = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors', maxZoom: 19
});
var esriSat = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {attribution: 'Tiles &copy; Esri', maxZoom: 20, maxNativeZoom: 19}
);
var radarLayer = L.tileLayer.wms('https://opengeo.ncep.noaa.gov/geoserver/conus/ows', {
  layers: 'conus_bref_qcd',
  format: 'image/png',
  transparent: true,
  version: '1.3.0',
  attribution: '<a href="https://radar.weather.gov">NWS Radar</a>',
  opacity: 0.7,
});
var _dpr = window.devicePixelRatio || 1;
var map = L.map('map', {center: [40.75, -73.5], zoom: 8, layers: [osmDay], zoomControl: false, zoomAnimation: (_dpr % 1 === 0)});
L.control.zoom({position: 'topleft'}).addTo(map);
L.control.scale({metric: true, imperial: true}).addTo(map);

// ── business POIs (Miles Urban destinations) ──────────────────────────────────
var poiData = [
  {lat:40.7589, lng:-73.9851, name:'Midtown Manhattan',              cat:'biz'},
  {lat:40.7127, lng:-74.0059, name:'Financial District (Wall St)',   cat:'biz'},
  {lat:40.7504, lng:-73.9967, name:'Hudson Yards',                   cat:'biz'},
  {lat:40.7531, lng:-73.9772, name:'Grand Central / Park Ave',       cat:'biz'},
  {lat:40.7580, lng:-73.9855, name:'Rockefeller Center',             cat:'biz'},
  {lat:40.7282, lng:-74.0776, name:'Jersey City Financial Center',   cat:'biz'},
  {lat:40.7357, lng:-74.1724, name:'Newark, NJ',                     cat:'biz'},
  {lat:40.7456, lng:-74.3204, name:'Short Hills, NJ (Corp. Park)',   cat:'biz'},
  {lat:41.0253, lng:-73.6282, name:'Greenwich, CT',                  cat:'biz'},
  {lat:41.0534, lng:-73.5387, name:'Stamford, CT',                   cat:'biz'},
  {lat:41.1220, lng:-73.7949, name:'White Plains, NY',               cat:'biz'},
  {lat:41.0662, lng:-73.8987, name:'Tarrytown, NY (Regeneron etc.)', cat:'biz'},
  {lat:40.7606, lng:-73.8296, name:'LaGuardia Airport (LGA)',        cat:'airport'},
  {lat:40.6413, lng:-73.7781, name:'JFK Airport',                    cat:'airport'},
  {lat:40.6895, lng:-74.1745, name:'Newark Liberty (EWR)',           cat:'airport'},
  {lat:41.0673, lng:-73.7076, name:'Westchester Airport (HPN)',      cat:'airport'},
  {lat:40.7060, lng:-74.0099, name:'Downtown Manhattan Heliport',    cat:'heliport'},
  {lat:40.7422, lng:-73.9750, name:'East 34th St Heliport',          cat:'heliport'},
  // ── executive residences (Miles Urban home zones) ─────────────────────────
  {lat:40.7736, lng:-73.9566, name:'Upper East Side',         cat:'res'},
  {lat:40.7870, lng:-73.9754, name:'Upper West Side',         cat:'res'},
  {lat:40.7195, lng:-74.0089, name:'Tribeca',                 cat:'res'},
  {lat:40.7339, lng:-74.0057, name:'West Village',            cat:'res'},
  {lat:40.7614, lng:-73.9776, name:'Sutton Place',            cat:'res'},
  {lat:40.7831, lng:-73.9712, name:'Carnegie Hill',           cat:'res'},
  {lat:40.7281, lng:-73.9944, name:'Chelsea / Hudson Square', cat:'res'},
  {lat:40.6960, lng:-73.9936, name:'Brooklyn Heights',        cat:'res'},
  {lat:40.7440, lng:-74.0324, name:'Hoboken, NJ',             cat:'res'},
  {lat:40.9176, lng:-73.8282, name:'Bronxville, NY',          cat:'res'},
  {lat:40.9895, lng:-73.7776, name:'Scarsdale, NY',           cat:'res'},
  {lat:41.0253, lng:-73.6282, name:'Greenwich, CT (res.)',    cat:'res'},
  {lat:40.7957, lng:-73.7269, name:'Great Neck, NY',          cat:'res'},
  {lat:40.9799, lng:-73.6876, name:'Rye, NY',                 cat:'res'},
];

function poiIcon(cat) {
  var bg  = cat==='airport' ? '#5b21b6' : cat==='heliport' ? '#0369a1' : cat==='res' ? '#065f46' : '#1e3a8a';
  var sym = cat==='airport' ? '&#x2708;' : cat==='heliport' ? 'H' : cat==='res' ? '&#x1F3E0;' : '&#x1F3E2;';
  return L.divIcon({
    html: '<div style="background:'+bg+';color:#fff;font-size:11px;font-weight:600;'+
          'padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,.25);'+
          'white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,.4)">'+sym+' </div>',
    className: '', iconAnchor: [0, 0]
  });
}
var poiLayer = L.layerGroup();    // business / airport / heliport
var resLayer  = L.layerGroup();   // executive residences
poiData.forEach(function(p) {
  var target = p.cat === 'res' ? resLayer : poiLayer;
  L.marker([p.lat, p.lng], {icon: poiIcon(p.cat), title: p.name})
   .bindTooltip(p.name, {sticky:true, direction:'top', offset:[0,-6]})
   .addTo(target);
});

// ── helipads (FAA) ─────────────────────────────────────────────────────────────
var helipadData = __GEOJSON__;
var cr = L.canvas({padding: 0.5});
var helipadLayer = L.geoJSON(helipadData, {
  renderer: cr,
  pointToLayer: function(f, ll) {
    return L.circleMarker(ll, {
      radius:5, color:'#1565C0', fillColor:'#42A5F5', fillOpacity:0.8, weight:1.5
    });
  },
  onEachFeature: function(f, l) {
    var p = f.properties, name = p.NAME || p.IDENT || 'Helipad';
    l.bindTooltip(name, {sticky:true});
    l.bindPopup(
      '<b>'+name+'</b><br>IDENT: '+(p.IDENT||'&mdash;')+'<br>'+
      (p.STATE ? 'State: '+p.STATE+(p.SERVCITY?' &middot; '+p.SERVCITY:'')+'<br>' : '')+
      (p.OPERSTATUS ? 'Status: '+p.OPERSTATUS+'<br>' : '')+
      (p.ELEVATION ? 'Elev: '+p.ELEVATION+' ft' : ''),
      {maxWidth: 220}
    );
  }
});

// ── helipads (OSM) ─────────────────────────────────────────────────────────────
var osmHelipadData = __OSM_GEOJSON__;
var crOsm = L.canvas({padding: 0.5});
var osmHelipadLayer = L.geoJSON(osmHelipadData, {
  renderer: crOsm,
  pointToLayer: function(f, ll) {
    return L.circleMarker(ll, {
      radius:4, color:'#E65100', fillColor:'#FF8A65', fillOpacity:0.8, weight:1.5
    });
  },
  onEachFeature: function(f, l) {
    var p = f.properties;
    var name = p.name || (p.faa ? 'OSM ['+p.faa+']' : 'OSM Helipad');
    l.bindTooltip(name, {sticky:true});
    l.bindPopup(
      '<b>'+name+'</b><br>'+
      (p.surface ? 'Surface: '+p.surface+'<br>' : '')+
      (p.ele ? 'Elev: '+p.ele+' m<br>' : '')+
      (p.faa ? 'FAA tag: '+p.faa : 'OSM ID: '+(p.osm_id||'—')),
      {maxWidth: 220}
    );
  }
});

// ── HIE-validated helipads (FAA pads confirmed by YOLO, gt=1) ─────────────────
var validatedData = __VALIDATED_GEOJSON__;
var crVal = L.canvas({padding: 0.5});
var validatedLayer = L.geoJSON(validatedData, {
  renderer: crVal,
  pointToLayer: function(f, ll) {
    return L.circleMarker(ll, {
      radius:6, color:'#16a34a', fillColor:'#4ade80', fillOpacity:0.9, weight:2
    });
  },
  onEachFeature: function(f, l) {
    var p=f.properties, name=p.NAME||p.IDENT||'Helipad';
    l.bindTooltip(name+' ✅', {sticky:true});
    l.bindPopup(
      '<b>'+name+'</b><br>IDENT: '+(p.IDENT||'—')+'<br>'+
      (p.STATE?'State: '+p.STATE+'<br>':'')+
      '<span style="color:#4ade80">✅ HIE Validated (YOLO confirmed)</span>',
      {maxWidth:230}
    );
  }
});

// ── NOTAM / TFR layer ─────────────────────────────────────────────────────────
var notamData = __NOTAM_GEOJSON__;
var weatherThresholds = __WEATHER_THRESHOLDS__;

// Ray-casting point-in-polygon. ring: GeoJSON format [[lon, lat], ...]
function pointInPolygon(ptLat, ptLon, ring) {
  var inside = false;
  for (var i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    var lonI = ring[i][0], latI = ring[i][1];
    var lonJ = ring[j][0], latJ = ring[j][1];
    if (((latI > ptLat) !== (latJ > ptLat)) &&
        (ptLon < (lonJ - lonI) * (ptLat - latI) / (latJ - latI) + lonI))
      inside = !inside;
  }
  return inside;
}
function isInActiveTFR(lat, lon) {
  for (var fi = 0; fi < notamData.features.length; fi++) {
    var f = notamData.features[fi];
    if (f.geometry && f.geometry.type === 'Polygon') {
      if (pointInPolygon(lat, lon, f.geometry.coordinates[0])) return true;
    }
  }
  return false;
}

var notamLayer = L.geoJSON(notamData, {
  style: function(f) {
    return f.geometry.type === 'Polygon'
      ? {color: '#ef4444', weight: 2, fillOpacity: 0.15, fillColor: '#ef4444'}
      : {};
  },
  pointToLayer: function(f, ll) {
    return L.circleMarker(ll, {radius: 8, color: '#ef4444', fill: true, fillOpacity: 0.4});
  },
  onEachFeature: function(f, l) {
    var txt = (f.properties && f.properties.text) ? f.properties.text : 'Active TFR';
    l.bindPopup('<b>🚫 Active TFR</b><br><small>' + txt + '</small>', {maxWidth: 280});
    l.bindTooltip('🚫 ' + txt.slice(0, 60), {sticky: true});
  }
});

// ── OSM-only helipads confirmed by YOLO visual detection ──────────────────────
var osmValidatedData = __OSM_VALIDATED_GEOJSON__;
var crOsmVal = L.canvas({padding: 0.5});
var osmValidatedLayer = L.geoJSON(osmValidatedData, {
  renderer: crOsmVal,
  pointToLayer: function(f, ll) {
    return L.circleMarker(ll, {
      radius:6, color:'#0369a1', fillColor:'#38bdf8', fillOpacity:0.85, weight:2
    });
  },
  onEachFeature: function(f, l) {
    var p = f.properties;
    var txt = p.name || 'OSM Helipad';
    l.bindTooltip('✅ ' + txt, {sticky: true});
    l.bindPopup(
      '<b>✅ ' + txt + '</b><br>'+
      '<small style="color:#38bdf8">HIE visual detection confirmed (OSM source)</small><br>'+
      (p.osm_id ? 'OSM ID: '+p.osm_id : ''),
      {maxWidth: 220}
    );
  }
});

var allPointLayers = [helipadLayer, osmHelipadLayer, validatedLayer, osmValidatedLayer];

// ── Mapbox traffic basemap (green/yellow/red roads) ──────────────────────────
// Uses the traffic-day-v2 style — shows live congestion colors on a full basemap.
// Added as a basemap option (not overlay) because Mapbox dropped the v4 raster overlay API.
var mapboxTrafficLayer = null;
if('__MAPBOX_TOKEN__'){
  mapboxTrafficLayer = L.tileLayer(
    'https://api.mapbox.com/styles/v1/mapbox/traffic-day-v2/tiles/256/{z}/{x}/{y}?access_token=__MAPBOX_TOKEN__',
    {attribution:'© <a href="https://www.mapbox.com">Mapbox</a>', maxZoom:22, tileSize:256}
  );
}

var _overlays = {
  'Helipads (FAA)': helipadLayer,
  'Helipads (OSM)': osmHelipadLayer,
  '✅ HIE Validated': validatedLayer,
  '✅ OSM Validated': osmValidatedLayer,
  'Business POIs':  poiLayer,
  'Exec. Residences': resLayer,
  '🚫 TFRs (airspace)': notamLayer,
  '⛈ Precipitation radar': radarLayer,
};
// Mapbox traffic is a basemap (not overlay) — full road network with live congestion colors
var _basemaps = {'Street Map': osmDay, 'Satellite (ESRI)': esriSat};
if(mapboxTrafficLayer) _basemaps['Traffic (Mapbox)'] = mapboxTrafficLayer;

L.control.layers(
  _basemaps,
  _overlays,
  {collapsed: false}
).addTo(map);
validatedLayer.addTo(map);
osmValidatedLayer.addTo(map);
poiLayer.addTo(map);
resLayer.addTo(map);
// notamLayer, radarLayer off by default — user toggles via layer control

// ── utils ─────────────────────────────────────────────────────────────────────
function haversine(lat1, lon1, lat2, lon2) {
  var R=6371, dLat=(lat2-lat1)*Math.PI/180, dLon=(lon2-lon1)*Math.PI/180;
  var a=Math.sin(dLat/2)*Math.sin(dLat/2)+
    Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLon/2)*Math.sin(dLon/2);
  return R*2*Math.atan2(Math.sqrt(a),Math.sqrt(1-a));
}
// NWS radar precipitation sample at a lat/lon point.
// Browser-side canvas read — CORS open (Access-Control-Allow-Origin: *).
// Returns red-channel intensity 0–255 (0 = no precipitation).
// Called only when a route is computed (2 requests total, not at page load).
async function samplePrecip(lat, lon) {
  var d=0.025;
  var url='https://opengeo.ncep.noaa.gov/geoserver/conus/ows?service=WMS&version=1.3.0' +
    '&request=GetMap&layers=conus_bref_qcd&bbox='+(lat-d)+','+(lon-d)+','+(lat+d)+','+(lon+d)+
    '&width=3&height=3&crs=EPSG:4326&format=image/png&transparent=true';
  try {
    var bmp=await createImageBitmap(await (await fetch(url)).blob());
    var cv=document.createElement('canvas'); cv.width=3; cv.height=3;
    var cx=cv.getContext('2d'); cx.drawImage(bmp,0,0);
    var px=cx.getImageData(1,1,1,1).data;
    return px[3]>10 ? px[0] : 0;
  } catch(e){ return 0; }
}
function fmtDur(m) {
  if (m<60) return Math.round(m)+' min';
  return Math.floor(m/60)+'h'+(Math.round(m%60)>0?' '+Math.round(m%60)+'m':'');
}
function fmtDist(km) { return km<1 ? Math.round(km*1000)+' m' : km.toFixed(1)+' km'; }
// Cache helipad list — invalidate only when layer visibility changes
var _helipadCache=null;
map.on('layeradd layerremove',function(){_helipadCache=null;});
function getAllHelipads() {
  if(_helipadCache) return _helipadCache;
  var pts=[];
  allPointLayers.forEach(function(ly){
    if (!map.hasLayer(ly)) return;
    ly.eachLayer(function(l){
      if(!l.feature) return;
      var ll=l.getLatLng?l.getLatLng():l.getBounds().getCenter();
      var p=l.feature.properties;
      pts.push({lat:ll.lat, lon:ll.lng, name:p.NAME||p.name||'Helipad'});
    });
  });
  _helipadCache=pts;
  return pts;
}

// Fast estimate for short drive legs to/from helipad.
// Uses haversine × 1.35 road-network factor, 25 km/h urban average.
// Avoids an OSRM/TomTom call for each leg — cuts API calls from 3 → 1.
function estimateDriveLeg(lat1,lng1,lat2,lng2) {
  var d=haversine(lat1,lng1,lat2,lng2)*1.35;
  return {dist:d, duration:d/25*60,
          geom:{type:'LineString',coordinates:[[lng1,lat1],[lng2,lat2]]}};
}
function nearestHelipad(lat,lng,pts) {
  var best=null,bd=Infinity;
  pts.forEach(function(p){var d=haversine(lat,lng,p.lat,p.lon);if(d<bd){bd=d;best=p;}});
  return best?{pad:best,dist:bd}:null;
}
// Adapter: findRoute expects {lat, lng} (Leaflet); pads use {lat, lon}.
// Returns the Dijkstra result with path[0]/path[last] names patched to real pad names.
function findHeliRoute(padA,padB,pts){
  var r=findRoute({lat:padA.lat,lng:padA.lon},{lat:padB.lat,lng:padB.lon},pts);
  if(r&&r.path.length>=2){
    r.path[0].name=padA.name||padA.IDENT||'Departure';
    r.path[r.path.length-1].name=padB.name||padB.IDENT||'Arrival';
  }
  return r;
}

// ── custom markers ────────────────────────────────────────────────────────────
function flagIcon(color, letter) {
  return L.divIcon({
    html:'<div style="position:relative;width:30px;height:40px">'+
      '<div style="position:absolute;bottom:0;left:4px;width:3px;height:28px;background:'+color+';border-radius:2px"></div>'+
      '<div style="position:absolute;top:0;left:4px;background:'+color+';color:#fff;font-weight:800;font-size:12px;'+
        'padding:2px 6px;border-radius:4px 4px 4px 0;box-shadow:0 2px 6px rgba(0,0,0,.4);'+
        'min-width:18px;text-align:center;border:1px solid rgba(255,255,255,.5)">'+letter+'</div>'+
    '</div>',
    className:'', iconSize:[30,40], iconAnchor:[6,40]
  });
}
function hIcon(color) {
  return L.divIcon({
    html:'<div style="background:'+color+';color:#fff;font-weight:900;font-size:13px;'+
      'width:28px;height:28px;border-radius:50%;border:3px solid #fff;'+
      'text-align:center;line-height:22px;'+
      'box-shadow:0 2px 8px rgba(0,0,0,.5)">H</div>',
    className:'', iconSize:[28,28], iconAnchor:[14,14]
  });
}

// ── state ─────────────────────────────────────────────────────────────────────
var routeLayers=[], mkA=null, mkB=null, ptA=null, ptB=null, clickState=0;

function clearHeli() {
  routeLayers.forEach(function(l){map.removeLayer(l);}); routeLayers=[];
  document.getElementById('route-info').style.display='none';
}
function clearMM() {
  routeLayers.forEach(function(l){map.removeLayer(l);}); routeLayers=[];
  if(mkA){map.removeLayer(mkA);mkA=null;} if(mkB){map.removeLayer(mkB);mkB=null;}
  ptA=null; ptB=null;
  document.getElementById('compare-panel').style.display='none';
  resetBar();
}
function clearAll() { clearMM(); clearHeli(); }

function resetBar() {
  var b=document.getElementById('aerial-bar'); b.className='';
  b.innerHTML='<span style="color:#475569;font-size:12px">Click &#x1F9ED; then pick two points to compare ground vs aerial routes</span>';
}
function setBar(groundMin, airMin) {
  var b=document.getElementById('aerial-bar');
  if (airMin===null) {
    b.className='no-save';
    b.innerHTML=
      '<div class="ab-seg"><div class="ab-lbl">Best ground</div><div class="ab-val ground">'+fmtDur(groundMin)+'</div></div>'+
      '<div class="ab-div"></div>'+
      '<div class="ab-neutral">No helipad route available for these points</div>';
    return;
  }
  var saving=groundMin-airMin;
  if (saving>0.5) {
    b.className='has-save';
    b.innerHTML=
      '<div class="ab-seg"><div class="ab-lbl">Best ground</div><div class="ab-val ground">'+fmtDur(groundMin)+'</div></div>'+
      '<div class="ab-div"></div>'+
      '<div class="ab-seg"><div class="ab-lbl">Aerial route</div><div class="ab-val air">'+fmtDur(airMin)+'</div></div>'+
      '<div class="ab-div"></div>'+
      '<div class="ab-badge">&#x2708; Aerial advantage: saves '+fmtDur(saving)+'</div>';
  } else {
    b.className='no-save';
    b.innerHTML=
      '<div class="ab-seg"><div class="ab-lbl">Best ground</div><div class="ab-val ground">'+fmtDur(groundMin)+'</div></div>'+
      '<div class="ab-div"></div>'+
      '<div class="ab-seg"><div class="ab-lbl">Aerial route</div><div class="ab-val dim">'+fmtDur(airMin)+'</div></div>'+
      '<div class="ab-div"></div>'+
      '<div class="ab-neutral">Ground is faster for this trip</div>';
  }
}

// ── Routing (TomTom traffic-aware → OSRM fallback → haversine) ───────────────
// TomTom free tier: 2,500 req/day, NO credit card required.
// Register at https://developer.tomtom.com/ (email only).
// Add to .env:  TOMTOM_API_KEY=your_key
var TOMTOM_KEY='__TOMTOM_API_KEY__';

// ── Quota guard (localStorage) ────────────────────────────────────────────────
// Hard-stops paid API calls before the daily limit to prevent any surprise charges.
// Falls back to OSRM automatically when quota is reached.
var _QUOTA_KEY='skyroute_tomtom_quota';
var _DAILY_LIMIT=2000;   // TomTom free = 2500/day; stop at 2000 for safety margin

function _quotaOk(){
  try{
    var raw=localStorage.getItem(_QUOTA_KEY);
    var q=raw?JSON.parse(raw):{date:'',count:0};
    var today=new Date().toISOString().slice(0,10);
    if(q.date!==today){q={date:today,count:0};}
    if(q.count>=_DAILY_LIMIT){
      console.warn('SkyRoute: TomTom daily quota reached ('+q.count+'/'+_DAILY_LIMIT+'). Using OSRM fallback.');
      return false;
    }
    q.count++;
    localStorage.setItem(_QUOTA_KEY,JSON.stringify(q));
    return true;
  }catch(e){return true;} // if localStorage unavailable, allow call
}
function _quotaStatus(){
  try{
    var raw=localStorage.getItem(_QUOTA_KEY);
    if(!raw) return '0/'+_DAILY_LIMIT+' today';
    var q=JSON.parse(raw);
    var today=new Date().toISOString().slice(0,10);
    if(q.date!==today) return '0/'+_DAILY_LIMIT+' today';
    return q.count+'/'+_DAILY_LIMIT+' today';
  }catch(e){return 'unknown';}
}

async function osrmRoute(lat1,lng1,lat2,lng2,profile) {
  // 1. Try TomTom (live traffic, 2500 req/day free, no credit card)
  if(TOMTOM_KEY && _quotaOk()){
    var tmUrl='https://api.tomtom.com/routing/1/calculateRoute/'+
      lat1.toFixed(6)+','+lng1.toFixed(6)+':'+lat2.toFixed(6)+','+lng2.toFixed(6)+
      '/json?travelMode='+(profile==='driving'?'car':'pedestrian')+
      '&traffic=true&routeType=fastest&key='+TOMTOM_KEY;
    try{
      var tj=await (await fetch(tmUrl)).json();
      var leg=tj.routes[0].legs[0];
      var pts=tj.routes[0].legs[0].points||[];
      var geom=pts.length?{type:'LineString',coordinates:pts.map(function(p){return[p.longitude,p.latitude];})}:null;
      return {
        dist:     tj.routes[0].summary.lengthInMeters/1000,
        duration: tj.routes[0].summary.travelTimeInSeconds/60,
        geom:     geom
      };
    }catch(e){}
  }
  // 2. OSRM fallback (no traffic)
  var oUrl='https://router.project-osrm.org/route/v1/'+profile+'/'+
    lng1.toFixed(6)+','+lat1.toFixed(6)+';'+lng2.toFixed(6)+','+lat2.toFixed(6)+
    '?overview=full&geometries=geojson';
  try{
    var r=await (await fetch(oUrl)).json();
    if(r.code==='Ok'&&r.routes&&r.routes.length)
      return {dist:r.routes[0].distance/1000, duration:r.routes[0].duration/60, geom:r.routes[0].geometry};
  }catch(e){}
  // 3. Haversine last resort
  var d=haversine(lat1,lng1,lat2,lng2);
  return {dist:d, duration:d/30*60, geom:null};
}

// ── Multi-modal computation ────────────────────────────────────────────────────
async function computeMultiModal(pA, pB) {
  var sb=document.getElementById('status-bar');
  sb.innerHTML='&#x23F3; Computing routes&hellip;'; sb.style.display='block';
  // Yield to browser so the spinner paints before any blocking work starts.
  await new Promise(function(r){setTimeout(r,0);});

  // ── 1. Synchronous work: find helipads + plan aerial path ─────────────────
  // Must complete before network calls so padA/padB are known.
  var allPts=getAllHelipads();
  var nearA=nearestHelipad(pA.lat,pA.lng,allPts);
  var nearB=nearestHelipad(pB.lat,pB.lng,allPts);
  var padA=null,padB=null,heliPath=null,hd=0,hDur=0,tfrNote='',heliPossible=false;
  if(nearA&&nearB){
    padA=nearA.pad; padB=nearB.pad;
    heliPath=[padA,padB];
    hd=haversine(padA.lat,padA.lon,padB.lat,padB.lon);
    hDur=hd/SPEED_HELI_KMH*60;
    if(map.hasLayer(notamLayer)&&hd>0.1&&hd<=RANGE_KM){
      // Fast direct-path check: 4 sample points. Skip Dijkstra if direct is clear.
      var directBlocked=false;
      for(var tk=1;tk<=4;tk++){
        var tt=tk/5;
        if(isInActiveTFR(padA.lat+tt*(padB.lat-padA.lat),padA.lon+tt*(padB.lon-padA.lon))){directBlocked=true;break;}
      }
      if(directBlocked){
        var route=findHeliRoute(padA,padB,allPts);
        if(route&&route.path&&route.path.length>=2){
          heliPath=route.path; hd=route.dist; hDur=route.dur;
        } else {
          tfrNote=' <span style="color:#ef4444;font-size:10px">🚫 All paths cross active TFR</span>';
        }
      }
      if(!tfrNote&&(isInActiveTFR(padA.lat,padA.lon)||isInActiveTFR(padB.lat,padB.lon)))
        tfrNote=' <span style="color:#f59e0b;font-size:10px">⚠ TFR near pad — auth may be required</span>';
    }
    heliPossible=hd>0.1&&hd<=RANGE_KM;
  }
  var walkDist=haversine(pA.lat,pA.lng,pB.lat,pB.lng), walkDur=walkDist/SPEED_WALK_KMH*60;

  // ── 2. Three OSRM calls in parallel — wall time = slowest single call (~1-3 s)
  // All three run simultaneously so road geometry is accurate for every leg.
  var _routePromises=[
    osrmRoute(pA.lat,pA.lng,pB.lat,pB.lng,'driving'),
    heliPossible?osrmRoute(pA.lat,pA.lng,padA.lat,padA.lon,'driving'):Promise.resolve(null),
    heliPossible?osrmRoute(padB.lat,padB.lon,pB.lat,pB.lng,'driving'):Promise.resolve(null),
  ];
  var _routeRes=await Promise.all(_routePromises);
  var driveR=_routeRes[0];
  // Fallback to straight-line estimate if OSRM leg failed (OSRM returns null on error)
  var d2a=_routeRes[1]||(heliPossible?estimateDriveLeg(pA.lat,pA.lng,padA.lat,padA.lon):null);
  var d2b=_routeRes[2]||(heliPossible?estimateDriveLeg(padB.lat,padB.lon,pB.lat,pB.lng):null);

  var driveDur=driveR.duration, taxiDur=driveDur*1.1, transitDur=driveDur*1.5;
  var heli=null;
  if(heliPossible){
    heli={dur:d2a.duration+hDur+d2b.duration, dist:d2a.dist+hd+d2b.dist,
          hd:hd, hDur:hDur, padA:padA, padB:padB, heliPath:heliPath,
          gA:d2a.geom, gB:d2b.geom, tfrNote:tfrNote};
  }

  // best ground time
  var groundTimes=[driveDur,taxiDur]; if(walkDur<90) groundTimes.push(walkDur);
  var bestGround=Math.min.apply(null,groundTimes);

  // ── draw RED: driving route ───────────────────────────────────────────────
  if(driveR.geom)
    routeLayers.push(L.geoJSON(driveR.geom,{style:{color:'#ef4444',weight:5,opacity:0.85}}).addTo(map));
  else
    routeLayers.push(L.polyline([[pA.lat,pA.lng],[pB.lat,pB.lng]],{color:'#ef4444',weight:5,opacity:0.7,dashArray:'6 4'}).addTo(map));

  // ── draw GREEN+CYAN: aerial route (multi-hop arc when TFR-avoiding) ─────────
  if(heli) {
    if(heli.gA) routeLayers.push(L.geoJSON(heli.gA,{style:{color:'#22c55e',weight:4,opacity:0.9}}).addTo(map));
    // Helicopter path — one or more segments via intermediate helipads
    var hCoords=heli.heliPath.map(function(p){return[p.lat,p.lon];});
    routeLayers.push(L.polyline(hCoords,{color:'#00d4ff',weight:3,dashArray:'10 6',opacity:0.95}).addTo(map));
    // Intermediate waypoint markers (not takeoff or landing)
    for(var mi=1;mi<heli.heliPath.length-1;mi++){
      var wp=heli.heliPath[mi];
      routeLayers.push(L.marker([wp.lat,wp.lon],{icon:hIcon('#00b8d9'),zIndexOffset:400})
        .bindTooltip('Via: '+wp.name,{direction:'top'}).addTo(map));
    }
    if(heli.gB) routeLayers.push(L.geoJSON(heli.gB,{style:{color:'#22c55e',weight:4,opacity:0.9}}).addTo(map));
    routeLayers.push(L.marker([heli.padA.lat,heli.padA.lon],{icon:hIcon('#00b8d9'),zIndexOffset:500})
      .bindTooltip('Take-off: '+heli.padA.name,{direction:'top'}).addTo(map));
    routeLayers.push(L.marker([heli.padB.lat,heli.padB.lon],{icon:hIcon('#22c55e'),zIndexOffset:500})
      .bindTooltip('Landing: '+heli.padB.name,{direction:'top'}).addTo(map));
  }

  // ── fit bounds ────────────────────────────────────────────────────────────
  var pts=[pA,pB];
  if(heli){heli.heliPath.forEach(function(p){pts.push(L.latLng(p.lat,p.lon));});}
  map.fitBounds(L.latLngBounds(pts),{padding:[50,50]});

  // ── table ─────────────────────────────────────────────────────────────────
  var rows=[
    {name:'&#x1F6B6; Walking',         dur:walkDur,    dist:walkDist,    air:'&mdash;', ground:true},
    {name:'&#x1F697; Car',             dur:driveDur,   dist:driveR.dist, air:'&mdash;', ground:true},
    {name:'&#x1F695; Taxi',            dur:taxiDur,    dist:driveR.dist, air:'&mdash;', ground:true, note:'est.'},
    {name:'&#x1F68C; Transit/Subway',  dur:transitDur, dist:driveR.dist, air:'&mdash;', ground:true, note:'est.'},
  ];
  // Precipitation warning badge — browser-side NWS WMS sample (2 req, ~0.1 s)
  var heliWx = '';
  if(heli && weatherThresholds) {
    var padIntensity = await samplePrecip(heli.padA.lat, heli.padA.lon);
    var thr = weatherThresholds.helicopter;
    if(padIntensity >= thr.avoid)
      heliWx = ' <span style="color:#ef4444;font-size:11px">⛈ Severe precip — not recommended</span>';
    else if(padIntensity >= thr.warn)
      heliWx = ' <span style="color:#eab308;font-size:11px">🌧 Rain detected at departure pad</span>';
  }

  if(heli) rows.push({
    name:'&#x1F697;&#x2708; Car + Heli + Car' + heliWx + (heli.tfrNote||''),
    dur:heli.dur, dist:heli.dist,
    air:fmtDist(heli.hd)+' &middot; '+fmtDur(heli.hDur), ground:false
  });

  var fastestDur=rows.reduce(function(a,b){return a.dur<b.dur?a:b;}).dur;
  var fastestGround=rows.filter(function(r){return r.ground;}).reduce(function(a,b){return a.dur<b.dur?a:b;}).dur;

  var tbody=document.getElementById('cp-tbody'); tbody.innerHTML='';
  rows.forEach(function(r){
    var tr=document.createElement('tr');
    tr.className=r.ground?(Math.abs(r.dur-fastestGround)<0.1?'gr-best':''):'ai-best';
    var tags='';
    if(Math.abs(r.dur-fastestDur)<0.1) tags+='<span class="tag tag-f">Fastest</span>';
    if(!r.ground) tags+='<span class="tag tag-a">Aerial</span>';
    tr.innerHTML='<td>'+r.name+tags+(r.note?'<span style="color:#475569;font-size:10px"> '+r.note+'</span>':'')+'</td>'+
      '<td>'+fmtDist(r.dist)+'</td><td>'+fmtDur(r.dur)+'</td><td>'+r.air+'</td>';
    tbody.appendChild(tr);
  });
  document.getElementById('cp-from-to').textContent='A ('+pA.lat.toFixed(4)+', '+pA.lng.toFixed(4)+')  →  B ('+pB.lat.toFixed(4)+', '+pB.lng.toFixed(4)+')';
  document.getElementById('compare-panel').style.display='block';
  sb.style.display='none';
  setBar(fastestGround, heli?heli.dur:null);
}

// ── Helipad-to-helipad Dijkstra ───────────────────────────────────────────────
function findRoute(start,end,allPts) {
  // Bound the graph to a corridor around the route — prevents O(N²) explosion
  // with 2000+ helipad pools. 1.5°lat (~165 km) / 2.5°lon (~220 km) buffer is
  // more than enough for a TFR detour in the NE US.
  var bufLat=1.5,bufLon=2.5;
  var minLat=Math.min(start.lat,end.lat)-bufLat, maxLat=Math.max(start.lat,end.lat)+bufLat;
  var minLon=Math.min(start.lng,end.lng)-bufLon, maxLon=Math.max(start.lng,end.lng)+bufLon;
  var nearby=allPts.filter(function(p){return p.lat>=minLat&&p.lat<=maxLat&&p.lon>=minLon&&p.lon<=maxLon;});
  var nodes=[{lat:start.lat,lon:start.lng,name:'Origin'}].concat(nearby).concat([{lat:end.lat,lon:end.lng,name:'Destination'}]);
  var N=nodes.length, dst=N-1, dist=[], prev=[];
  for(var i=0;i<N;i++){dist.push(Infinity);prev.push(-1);}
  dist[0]=0; var heap=[[0,0]];
  while(heap.length){
    heap.sort(function(a,b){return a[0]-b[0];});
    var top=heap.shift(),cost=top[0],u=top[1];
    if(cost>dist[u])continue; if(u===dst)break;
    for(var v=0;v<N;v++){
      if(v===u)continue;
      var d=haversine(nodes[u].lat,nodes[u].lon,nodes[v].lat,nodes[v].lon);
      if(d>RANGE_KM)continue;
      // Skip edge if any corridor sample point falls inside an active TFR
      if(map.hasLayer(notamLayer)){
        var edgeBlocked=false;
        for(var tk=1;tk<=4;tk++){
          var tt=tk/5;
          if(isInActiveTFR(
            nodes[u].lat+tt*(nodes[v].lat-nodes[u].lat),
            nodes[u].lon+tt*(nodes[v].lon-nodes[u].lon)
          )){edgeBlocked=true;break;}
        }
        if(edgeBlocked) continue;
      }
      var nc=dist[u]+d; if(nc<dist[v]){dist[v]=nc;prev[v]=u;heap.push([nc,v]);}
    }
  }
  if(dist[dst]===Infinity)return null;
  var path=[],cur=dst; while(cur!==-1){path.unshift(nodes[cur]);cur=prev[cur];}
  return {path:path, dist:dist[dst], dur:dist[dst]/SPEED_HELI_KMH*60};
}
function drawHeliRoute(result) {
  clearHeli();
  if(!result||result.path.length<2){alert('No helipad route found within range.');return;}
  var ll=result.path.map(function(p){return[p.lat,p.lon];});
  routeLayers.push(L.polyline(ll,{color:'#1565C0',weight:3,dashArray:'8 4',opacity:0.9}).addTo(map));
  for(var i=1;i<result.path.length-1;i++)
    routeLayers.push(L.marker([result.path[i].lat,result.path[i].lon],
      {icon:hIcon('#1565C0'),zIndexOffset:500}).bindTooltip(result.path[i].name,{direction:'top'}).addTo(map));
  map.fitBounds(L.latLngBounds(ll),{padding:[40,40]});
  document.getElementById('ri-title').textContent='Helipad Route ('+result.path.length+' waypoints)';
  document.getElementById('ri-dist').textContent=fmtDist(result.dist);
  document.getElementById('ri-dur').textContent=fmtDur(result.dur);
  document.getElementById('route-info').style.display='block';
}

// ── controls ──────────────────────────────────────────────────────────────────
// Routing buttons placed bottom-right to avoid overlapping the layers control panel
var HeliControl=L.Control.extend({
  options:{position:'bottomright'},
  onAdd:function(){
    var btn=L.DomUtil.create('button','ctrl-btn');
    btn.title='Helipad-to-helipad routing (click 2 points)';
    btn.style.cssText='font-size:20px;width:40px;height:40px;margin-bottom:4px;';
    btn.innerHTML='&#x1F681;';
    L.DomEvent.on(btn,'click',function(e){
      L.DomEvent.stopPropagation(e); clearAll(); clickState=1;
      var sb=document.getElementById('status-bar');
      sb.innerHTML='&#x1F4CD; Click <b>Origin</b> on the map'; sb.style.display='block';
    }); return btn;
  }
});
var MMControl=L.Control.extend({
  options:{position:'bottomright'},
  onAdd:function(){
    var btn=L.DomUtil.create('button','ctrl-btn');
    btn.title='Multi-modal comparison (click 2 points)';
    btn.style.cssText='font-size:20px;width:40px;height:40px;';
    btn.innerHTML='&#x1F9ED;';
    L.DomEvent.on(btn,'click',function(e){
      L.DomEvent.stopPropagation(e); clearAll(); clickState=10;
      var sb=document.getElementById('status-bar');
      sb.innerHTML='&#x1F4CD; Click <b>Origin (A)</b> on the map'; sb.style.display='block';
    }); return btn;
  }
});
new HeliControl().addTo(map);
new MMControl().addTo(map);

// ── click handler ─────────────────────────────────────────────────────────────
map.on('click', async function(e){
  var sb=document.getElementById('status-bar');
  if(clickState===1){
    ptA=e.latlng; if(mkA)map.removeLayer(mkA);
    mkA=L.marker(ptA,{icon:flagIcon('#ef4444','A'),zIndexOffset:1000}).addTo(map);
    sb.innerHTML='&#x1F4CD; Click <b>Destination</b> on the map'; clickState=2;
  } else if(clickState===2){
    ptB=e.latlng; if(mkB)map.removeLayer(mkB);
    mkB=L.marker(ptB,{icon:flagIcon('#1565C0','B'),zIndexOffset:1000}).addTo(map);
    sb.style.display='none'; clickState=0;
    var hpts=getAllHelipads();
    if(!hpts.length){alert('No helipad layers visible. Enable at least one helipad layer to route.');return;}
    drawHeliRoute(findRoute(ptA,ptB,hpts));
  } else if(clickState===10){
    ptA=e.latlng; if(mkA)map.removeLayer(mkA);
    mkA=L.marker(ptA,{icon:flagIcon('#ef4444','A'),zIndexOffset:1000}).addTo(map);
    sb.innerHTML='&#x1F4CD; Click <b>Destination (B)</b> on the map'; clickState=11;
  } else if(clickState===11){
    ptB=e.latlng; if(mkB)map.removeLayer(mkB);
    mkB=L.marker(ptB,{icon:flagIcon('#1565C0','B'),zIndexOffset:1000}).addTo(map);
    clickState=0; await computeMultiModal(ptA,ptB);
  }
});

// ── Auto-trigger from Route Assistant (injected at build time by Python) ──────
// __INIT_A__ and __INIT_B__ are replaced with {lat,lng} objects or null.
(function(){
  var _ia=__INIT_A__, _ib=__INIT_B__;
  if(!_ia||!_ib) return;
  setTimeout(async function(){
    ptA=L.latLng(_ia.lat,_ia.lng); ptB=L.latLng(_ib.lat,_ib.lng);
    if(mkA) map.removeLayer(mkA);
    if(mkB) map.removeLayer(mkB);
    mkA=L.marker(ptA,{icon:flagIcon('#ef4444','A'),zIndexOffset:1000}).addTo(map);
    mkB=L.marker(ptB,{icon:flagIcon('#1565C0','B'),zIndexOffset:1000}).addTo(map);
    await computeMultiModal(ptA,ptB);
  },1800);
})();
</script>
</body>
</html>"""


@st.cache_data(ttl=1800, show_spinner=False)
def build_routing_html(faa_df: pd.DataFrame, osm_df: pd.DataFrame,
                       osm_validated_df: pd.DataFrame | None = None,
                       js_v: str = "m4.18", tomtom_key: str = "",
                       mapbox_token: str = "",
                       init_lat_a: float = 0.0, init_lon_a: float = 0.0,
                       init_lat_b: float = 0.0, init_lon_b: float = 0.0) -> str:
    """Build self-contained Leaflet routing HTML with FAA and OSM helipads injected as GeoJSON.

    Args:
        faa_df: FAA helipad DataFrame with lat, lon, IDENT, NAME columns.
        osm_df: OSM helipad DataFrame with lat, lon, name, faa, surface, ele columns.
        osm_validated_df: osm_validated.csv DataFrame; only hie_visual_detected=True rows are shown.

    Returns:
        Complete HTML string for use with streamlit.components.v1.html().
    """
    # Build GeoJSON using vectorized to_dict() — 20-50× faster than iterrows()
    _faa_cols = [c for c in ["IDENT", "NAME", "STATE", "SERVCITY", "OPERSTATUS", "ELEVATION"]
                 if c in faa_df.columns]
    _faa = faa_df.dropna(subset=["lat", "lon"])[_faa_cols + ["lat", "lon"]].copy()
    for c in _faa_cols:
        _faa[c] = _faa[c].where(_faa[c].notna(), "").astype(str)
    faa_features = [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
         "properties": {c: r[c] for c in _faa_cols}}
        for r in _faa.to_dict(orient="records")
    ]

    _osm_cols = [c for c in ["name", "faa", "surface", "ele", "osm_id", "aeroway"]
                 if c in osm_df.columns]
    _osm = osm_df.dropna(subset=["lat", "lon"])[_osm_cols + ["lat", "lon"]].copy()
    for c in _osm_cols:
        _osm[c] = _osm[c].where(_osm[c].notna(), "").astype(str)
    osm_features = [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
         "properties": {c: r[c] for c in _osm_cols}}
        for r in _osm.to_dict(orient="records")
    ]

    # HIE-validated subset: visually confirmed (gt=1) OR ADIP operational + recently inspected
    _insp_path = DATA_DIR / "inspector_results.csv"
    validated_geojson = '{"type":"FeatureCollection","features":[]}'
    if _insp_path.exists():
        _insp = pd.read_csv(_insp_path)
        _val_idents = set(_insp[_insp["gt"] == 1]["ident"].str.upper())
        # ADIP fallback: no visual marking found but registry confirms operational + inspected ≤1 yr ago
        if "operational" in faa_df.columns and "data_freshness_days" in faa_df.columns:
            _adip_fallback = faa_df[
                (faa_df["operational"] == 1) &
                (faa_df["data_freshness_days"].fillna(9999) <= 365)
            ]["IDENT"].dropna().str.upper()
            _val_idents = _val_idents | set(_adip_fallback)
        _val_cols = [c for c in ["IDENT", "NAME", "STATE", "SERVCITY"] if c in faa_df.columns]
        _val = faa_df[faa_df["IDENT"].isin(_val_idents)].dropna(subset=["lat", "lon"])
        _val = _val[_val_cols + ["lat", "lon"]].copy()
        for c in _val_cols:
            _val[c] = _val[c].where(_val[c].notna(), "").astype(str)
        validated_geojson = json.dumps({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature",
                 "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                 "properties": {c: r[c] for c in _val_cols}}
                for r in _val.to_dict(orient="records")
            ]
        })

    # OSM-only helipads confirmed by YOLO visual detection
    osm_validated_geojson = '{"type":"FeatureCollection","features":[]}'
    if osm_validated_df is not None and not osm_validated_df.empty:
        _ov = osm_validated_df[osm_validated_df["hie_visual_detected"]].dropna(subset=["lat", "lon"])
        osm_validated_geojson = json.dumps({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature",
                 "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
                 "properties": {
                     "name": str(r.get("name", "") or ""),
                     "osm_id": str(r.get("osm_id", "") or ""),
                 }}
                for r in _ov.to_dict(orient="records")
            ]
        })

    # TFR GeoJSON for JS routing checks
    tfrs = _get_active_tfrs()
    notam_geojson = json.dumps(tfrs_to_geojson(tfrs))

    faa_geojson = json.dumps({"type": "FeatureCollection", "features": faa_features})
    osm_geojson = json.dumps({"type": "FeatureCollection", "features": osm_features})
    return (
        _ROUTING_HTML_TEMPLATE
        .replace("__GEOJSON__", faa_geojson)
        .replace("__OSM_GEOJSON__", osm_geojson)
        .replace("__VALIDATED_GEOJSON__", validated_geojson)
        .replace("__OSM_VALIDATED_GEOJSON__", osm_validated_geojson)
        .replace("__NOTAM_GEOJSON__", notam_geojson)
        .replace("__WEATHER_THRESHOLDS__", json.dumps(PRECIP_THRESHOLDS))
        .replace("__TOMTOM_API_KEY__", tomtom_key)
        .replace("__MAPBOX_TOKEN__", mapbox_token)
        .replace("__INIT_A__", json.dumps({"lat": init_lat_a, "lng": init_lon_a}) if init_lat_a else "null")
        .replace("__INIT_B__", json.dumps({"lat": init_lat_b, "lng": init_lon_b}) if init_lat_b else "null")
    )


with tab_eda:
    st.caption("EDA Dashboard · FAA ADDS-ArcGIS + OpenStreetMap · Northeast US")
    st.divider()

    # ── KPI row ──────────────────────────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("FAA Heliports",   f"{len(faa):,}")
    k2.metric("OSM Helipads",    f"{len(osm):,}")
    k3.metric("FAA Operational", f"{(faa['OPERSTATUS']=='OPERATIONAL').sum():,}",
              delta=f"{(faa['OPERSTATUS']=='OPERATIONAL').mean()*100:.0f}%")
    k4.metric("FAA Military",    f"{(faa['MIL_CODE']=='MIL').sum():,}")
    k5.metric("OSM Named",       f"{osm['name'].notna().sum():,}",
              delta=f"{osm['name'].notna().mean()*100:.0f}%")

    st.divider()

    # ── map ──────────────────────────────────────────────────────────────────────────────────────────────────────────

    def build_map(
        faa_df: pd.DataFrame, osm_df: pd.DataFrame,
        show_f: bool, show_o: bool,
        center: list[float] | None = None,
        zoom: int | None = None,
        use_satellite: bool = False,
    ) -> folium.Map:
        """Build a clustered Folium map with FAA and OSM layers."""
        m = folium.Map(
            location=center or _MAP_CENTER,
            zoom_start=zoom or _MAP_ZOOM,
            tiles=None,
            max_zoom=20,
            control_scale=True,
            zoom_control=False,
        )

        MeasureControl(
            position="bottomleft",
            primary_length_unit="meters",
            secondary_length_unit="feet",
            primary_area_unit="sqmeters",
            active_color=_OSM_COLOR,
            completed_color=_FAA_COLOR,
        ).add_to(m)

        folium.TileLayer(
            tiles="CartoDB positron",
            name="Street map",
            overlay=False, control=True, show=not use_satellite,
            max_zoom=20,
        ).add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Esri, Maxar, Earthstar Geographics",
            name="Satellite (ESRI)",
            overlay=False, control=True, show=use_satellite,
            max_zoom=20, max_native_zoom=19,
        ).add_to(m)

        if show_f and not faa_df.empty:
            grp = folium.FeatureGroup(name=f"FAA ({len(faa_df):,})", show=True)
            clust = MarkerCluster(disableClusteringAtZoom=12)
            v = faa_df.dropna(subset=["lat", "lon"])
            _adip_base = "https://adip.faa.gov/agis/public/#/simpleAirportMap/"
            popups = (
                "<b>" + v["NAME"].fillna("Unknown") + "</b><br>"
                + "IDENT: " + v["IDENT"].fillna("—") + "<br>"
                + "State: " + v["STATE"].fillna("—") + " · "
                + v["SERVCITY"].fillna("—") + "<br>"
                + "Status: " + v["OPERSTATUS"].fillna("—") + "<br>"
                + "Type: " + v["MIL_CODE"].fillna("—") + "<br>"
                + "Elev: " + v["ELEVATION"].astype(str) + " ft<br>"
                + v["IDENT"].fillna("").apply(
                    lambda i: (f'<a href="{_adip_base}{i}" target="_blank" '
                               f'style="color:#64B5F6">\U0001f4cb ADIP record</a>')
                    if i else ""
                )
            )
            for lat, lon, pop, name in zip(v["lat"], v["lon"], popups,
                                            v["NAME"].fillna("FAA Heliport")):
                folium.CircleMarker(
                    [lat, lon], radius=6, color=_FAA_COLOR, fill=True,
                    fill_color=_FAA_COLOR, fill_opacity=0.8, weight=1.5,
                    popup=folium.Popup(pop, max_width=230), tooltip=name,
                ).add_to(clust)
            clust.add_to(grp)
            grp.add_to(m)

        if show_o and not osm_df.empty:
            grp = folium.FeatureGroup(name=f"OSM ({len(osm_df):,})", show=True)
            clust = MarkerCluster(disableClusteringAtZoom=12)
            v = osm_df.dropna(subset=["lat", "lon"])
            names    = v["name"].fillna("Unnamed").astype(str)
            surfaces = v["surface"].fillna("unknown").astype(str)
            popups   = (
                "<b>" + names + "</b><br>"
                + "Type: " + v["aeroway"].fillna("helipad").astype(str) + "<br>"
                + "Surface: " + surfaces + "<br>"
                + "OSM ID: " + v["osm_id"].astype(str)
            )
            for lat, lon, pop, name in zip(v["lat"], v["lon"], popups, names):
                folium.CircleMarker(
                    [lat, lon], radius=4, color=_OSM_COLOR, fill=True,
                    fill_color=_OSM_COLOR, fill_opacity=0.6, weight=1,
                    popup=folium.Popup(pop, max_width=200), tooltip=name,
                ).add_to(clust)
            clust.add_to(grp)
            grp.add_to(m)

        # ── Traffic basemap (Mapbox traffic-day-v2 style — green/yellow/red roads) ──
        # Added as a basemap (not overlay) because the v4 raster overlay API is deprecated.
        _mapbox_tk = os.getenv("MAPBOX_TOKEN", "")
        if _mapbox_tk:
            folium.TileLayer(
                tiles=(
                    f"https://api.mapbox.com/styles/v1/mapbox/traffic-day-v2"
                    f"/tiles/256/{{z}}/{{x}}/{{y}}?access_token={_mapbox_tk}"
                ),
                attr='© <a href="https://www.mapbox.com">Mapbox</a>',
                name="Traffic (Mapbox)",
                overlay=False, control=True, show=False,
            ).add_to(m)

        # ── Precipitation radar (NWS radar.weather.gov) ───────────────────────
        folium.WmsTileLayer(**get_nws_wms_kwargs()).add_to(m)

        # ── Active TFRs ────────────────────────────────────────────────────────
        tfrs = _get_active_tfrs()
        if tfrs:
            tfr_group = folium.FeatureGroup(name="TFRs — Active airspace closures", show=False)
            for tfr in tfrs:
                coords = tfr.get("coordinates", [])
                tip = tfr.get("text", "TFR")[:80]
                if tfr.get("geometry_type") == "Polygon" and len(coords) >= 3:
                    folium.Polygon(
                        locations=coords, color="red", weight=2,
                        fill=True, fill_opacity=0.15, fill_color="red",
                        tooltip=tip,
                        popup=folium.Popup(tfr.get("text", "TFR"), max_width=280),
                    ).add_to(tfr_group)
                elif coords:
                    folium.CircleMarker(
                        location=coords[0], radius=8, color="red",
                        fill=True, fill_opacity=0.4,
                        tooltip=tip,
                        popup=folium.Popup(tfr.get("text", "TFR"), max_width=280),
                    ).add_to(tfr_group)
            tfr_group.add_to(m)

        # ── METAR weather stations (NE US bounding box) ────────────────────────
        metar_obs = _get_metar_bbox(38.0, -76.5, 45.5, -69.5)
        if metar_obs:
            _cat_color = {"VFR": "#22c55e", "MVFR": "#eab308", "IFR": "#ef4444", "LIFR": "#a855f7"}
            wx_group = folium.FeatureGroup(name="METAR weather stations", show=False)
            for obs in metar_obs:
                if not obs.get("lat") or not obs.get("lon"):
                    continue
                cat = obs.get("fltCat", "VFR")
                color = _cat_color.get(cat, "#94a3b8")
                raw = obs.get("rawOb", "")
                popup_html = (
                    f'<b>{obs.get("icaoId","")}</b> — '
                    f'<span style="color:{color}">{cat}</span><br>'
                    f'<small>{raw[:80]}</small>'
                )
                folium.CircleMarker(
                    location=[obs["lat"], obs["lon"]],
                    radius=8, color=color, fill=True, fill_color=color,
                    fill_opacity=0.75, weight=2,
                    popup=folium.Popup(popup_html, max_width=260),
                    tooltip=f'{obs.get("icaoId","")} {cat}',
                ).add_to(wx_group)
            wx_group.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        # Inject setView JS that fires inside the component's own iframe.
        # streamlit-folium exposes the Leaflet map as window.map in that iframe,
        # so this is the only reliable way to programmatically navigate.
        if center and zoom:
            from branca.element import Element
            _lat, _lon = float(center[0]), float(center[1])
            _z = int(zoom)
            m.get_root().html.add_child(Element(f"""
<script>
(function() {{
    var _t = 0;
    var _iv = setInterval(function() {{
        _t++;
        if (typeof window.map !== 'undefined' && typeof window.map.setView === 'function') {{
            window.map.setView([{_lat}, {_lon}], {_z});
            clearInterval(_iv);
        }} else if (_t > 60) {{ clearInterval(_iv); }}
    }}, 80);
}})();
</script>"""))

        # Poll every 400 ms: call invalidateSize() while container is 0×0
        # (hidden-tab init), then re-add the zoom control once the map has
        # proper dimensions so the +/- buttons are never clipped at the edge.
        from branca.element import Element
        m.get_root().html.add_child(Element(
            "<script>(function(){var n=0,iv=setInterval(function(){n++;"
            "var el=document.querySelector('.leaflet-container');"
            "if(el){var m=window[el.id];if(m&&typeof m.addLayer==='function'){"
            "if(!m._szZ){m._szZ=true;L.control.zoom({position:'topleft'}).addTo(m);"
            "if((window.devicePixelRatio||1)%1!==0)m.options.zoomAnimation=false;}"
            "var s=m.getSize();if(s.x===0||s.y===0)m.invalidateSize(false);}}"
            "if(n>50)clearInterval(iv);},200);})();</script>"
        ))

        return m

    st.subheader("Geographic Distribution")
    st.caption("Click a cluster to expand · Click a marker for details · Toggle layers top-right")

    _bk_lat  = st.session_state.get("_bk_lat")
    _bk_lon  = st.session_state.get("_bk_lon")
    _bk_zoom = st.session_state.get("_bk_zoom")
    _bk_ver  = st.session_state.get("_bk_ver", 0)
    _bk_center = [_bk_lat, _bk_lon] if _bk_lat and _bk_lon else None

    _use_satellite = (
        st.session_state.get("_use_satellite", False) and
        st.session_state.get("_satellite_ver") == _bk_ver
    )

    if _bk_center and _bk_zoom:
        st.session_state.setdefault("_last_center", {"lat": _bk_center[0], "lng": _bk_center[1]})
        st.session_state.setdefault("_last_zoom", _bk_zoom)
        if st.session_state.get("_last_bk_ver") != _bk_ver:
            st.session_state["_last_center"] = {"lat": _bk_center[0], "lng": _bk_center[1]}
            st.session_state["_last_zoom"]   = _bk_zoom
            st.session_state["_last_bk_ver"] = _bk_ver

    # Navigation is handled by a setView JS snippet injected into the Folium
    # map by build_map().  The key change forces a full remount (new Folium
    # map at the correct location) on every jump so the injected JS always
    # fires on a fresh component instance.
    map_state = st_folium(
        build_map(faa, osm, True, True,
                  center=_bk_center, zoom=_bk_zoom,
                  use_satellite=_use_satellite),
        width="100%", height=530, returned_objects=[],
        key=f"main_map_{_bk_ver}",
    )

    if map_state and map_state.get("center"):
        st.session_state["_last_center"] = map_state["center"]
        st.session_state["_last_zoom"]   = map_state.get("zoom", 14)

    _meta_center = st.session_state.get("_last_center", {})
    _meta_zoom   = st.session_state.get("_last_zoom", 0)
    _clat = _meta_center.get("lat")
    _clng = _meta_center.get("lng")

    if _meta_zoom >= 11 and _clat and _clng:
        meta = fetch_imagery_meta(round(_clat, 3), round(_clng, 3))
        if meta:
            src   = meta.get("SOURCE") or meta.get("NICE_NAME") or "ESRI World Imagery"
            desc  = meta.get("DESCRIPTION") or meta.get("SOURCE_INFO") or ""
            date  = meta.get("SRC_DATE2") or meta.get("DATE (YYYYMMDD)") or "—"
            res_m = meta.get("RESOLUTION (M)")
            acc_m = meta.get("ACCURACY (M)")
            label = f"{src} — {desc}" if desc and desc not in src else src
            res_str = f"{float(res_m)*100:.0f} cm/px" if res_m is not None else "—"
            acc_str = f"{float(acc_m):.1f} m"          if acc_m is not None else "—"
            st.markdown(
                f"""
<div style="
  background:#0d2137;
  border-left:4px solid #29b6f6;
  border-radius:6px;
  padding:10px 14px;
  margin:6px 0 4px 0;
  font-size:0.82rem;
  line-height:1.75;
  color:#e3f2fd;
">
  \U0001f6f0️ <b style="font-size:0.88rem;color:#29b6f6;">Satellite / Aerial Imagery</b><br>
  <b>Source&nbsp;&nbsp;&nbsp;:</b> {label}<br>
  <b>Acquired :</b> {date}<br>
  <b>Resolution:</b> {res_str} &nbsp;&nbsp;
  <b>Accuracy :</b> {acc_str}
</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div style="
  background:#1a1a2e;border-left:4px solid #546e7a;border-radius:6px;
  padding:8px 12px;margin:6px 0;font-size:0.82rem;color:#90a4ae;
">
  \U0001f6f0️ Imagery metadata unavailable at this location.
</div>""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
<div style="
  background:#1a1a2e;border-left:4px solid #37474f;border-radius:6px;
  padding:8px 12px;margin:6px 0;font-size:0.82rem;color:#78909c;
">
  \U0001f6f0️ Switch to <b>Satellite</b> layer and zoom&nbsp;in (level&nbsp;11+)<br>
  to see imagery source and acquisition date.
</div>""",
            unsafe_allow_html=True,
        )


    st.divider()
    st.markdown("### 📊 EDA — Data Intelligence")

    # ---- Chart 1: Field Completeness FAA vs OSM ----------------------------------
    # Three paired fields shown in the same order on both charts:
    #   Identification (IDENT / faa), Name (NAME / name), Elevation (ELEVATION / ele)
    _PAIR_LABEL = ["Elevation", "Name", "Identification"]   # bottom → top (ascending)
    _FAA_PAIR   = ["ELEVATION", "NAME", "IDENT"]
    _OSM_PAIR   = ["ele",       "name", "faa"]

    _fc_all = faa_completeness(faa_raw)
    _fc_map = {row["field"]: 100 - row["null_pct"] for _, row in _fc_all.iterrows()}
    _faa_pct = [round(_fc_map.get(f, 0.0), 1) for f in _FAA_PAIR]

    _osm_raw_comp = osm_completeness(osm, key_fields=_OSM_PAIR)
    _osm_map = {row["field"]: row["pct_present"] for _, row in _osm_raw_comp.iterrows()}
    _osm_pct = [round(_osm_map.get(f, 0.0), 1) for f in _OSM_PAIR]
    _osm_cnt = [int(_osm_map.get(f, 0)) for f in _OSM_PAIR]  # not needed below but keep for ref

    _faa_df = pd.DataFrame({"field": _PAIR_LABEL, "pct_present": _faa_pct,
                             "raw_field": _FAA_PAIR})
    _osm_df = pd.DataFrame({"field": _PAIR_LABEL, "pct_present": _osm_pct,
                             "raw_field": _OSM_PAIR})

    _elev_pct_faa = _faa_pct[0]
    _elev_pct_osm = _osm_pct[0]
    _faa_tag_pct  = _faa_pct[2]   # IDENT always 100
    _osm_faa_pct  = _osm_pct[2]   # faa tag completeness

    st.markdown("#### 1 — Field Completeness: FAA vs OSM")
    _cc1, _cc2 = st.columns(2)

    with _cc1:
        _fig = px.bar(
            _faa_df, x="pct_present", y="field", orientation="h",
            color="pct_present",
            color_continuous_scale=["#FFCCBC", _FAA_COLOR],
            text=_faa_df["pct_present"].apply(lambda x: f"{x:.0f}%"),
            title=f"FAA Field Completeness  (N={len(faa_raw):,})",
            labels={"pct_present": "% Records with Value", "field": ""},
            hover_data={"raw_field": True, "pct_present": True, "field": False},
        )
        _fig.update_traces(textposition="outside")
        _fig.update_layout(xaxis_range=[0, 115], coloraxis_showscale=False,
                           height=260, yaxis={"categoryorder": "array",
                                              "categoryarray": _PAIR_LABEL})
        st.plotly_chart(_fig, use_container_width=True)

    with _cc2:
        _fig = px.bar(
            _osm_df, x="pct_present", y="field", orientation="h",
            color="pct_present",
            color_continuous_scale=["#FFCCBC", _OSM_COLOR],
            text=_osm_df["pct_present"].apply(lambda x: f"{x:.0f}%"),
            title=f"OSM Field Completeness  (N={len(osm):,})",
            labels={"pct_present": "% Records with Value", "field": ""},
            hover_data={"raw_field": True, "pct_present": True, "field": False},
        )
        _fig.update_traces(textposition="outside")
        _fig.update_layout(xaxis_range=[0, 115], coloraxis_showscale=False,
                           height=260, yaxis={"categoryorder": "array",
                                              "categoryarray": _PAIR_LABEL})
        st.plotly_chart(_fig, use_container_width=True)

    st.info(
        f"💡 **Insight — OSM helipad data is critically incomplete on key attributes.**  "
        f"FAA fills every IDENT ({_faa_tag_pct:.0f}%), Name (100%), and Elevation "
        f"({_elev_pct_faa:.0f}%) field across all {len(faa_raw):,} records. "
        f"OSM carries the FAA cross-reference tag (`faa`) on only **{_osm_faa_pct:.0f}%** "
        f"of its {len(osm):,} records, elevation on **{_elev_pct_osm:.0f}%**, and names "
        f"on **{_osm_pct[1]:.0f}%**. "
        f"This incompleteness is exactly why ML-based validation is required before "
        f"OSM pads can be trusted for routing decisions."
    )

    st.divider()

    # ---- Chart 2: Elevation Consistency -----------------------------------------
    _elev_data = cons.dropna(subset=["faa_elev_ft", "osm_elev_ft_converted"])
    _n_ele     = len(_elev_data)

    st.markdown("#### 2 — Elevation Consistency: FAA vs OSM")

    if _n_ele == 0:
        st.info("No matched pairs have elevation in both sources.")
    else:
        _med_delta   = _elev_data["elev_delta_ft"].abs().median()
        _pct_plaus   = _elev_data["elev_plausible"].mean() * 100
        _n_suspect   = int(_elev_data["osm_ele_likely_feet"].sum())
        _pct_suspect = _n_suspect / _n_ele * 100

        _ec1, _ec2 = st.columns(2)
        with _ec1:
            _lim = max(_elev_data["faa_elev_ft"].max(),
                       _elev_data["osm_elev_ft_converted"].max()) * 1.05
            _fig = px.scatter(
                _elev_data, x="faa_elev_ft", y="osm_elev_ft_converted",
                color="match_method",
                hover_data=["faa_name", "osm_name", "distance_m"],
                title=f"FAA vs OSM Elevation — {_n_ele} matched pairs",
                labels={"faa_elev_ft": "FAA Elevation (ft)",
                        "osm_elev_ft_converted": "OSM elevation → ft"},
                color_discrete_map={"faa_id": _OSM_COLOR, "proximity": _FAA_COLOR},
            )
            _fig.add_shape(type="line", x0=0, x1=_lim, y0=0, y1=_lim,
                           line=dict(color="red", dash="dash"))
            _fig.add_annotation(x=_lim * 0.8, y=_lim * 0.92,
                                 text="perfect agreement",
                                 showarrow=False,
                                 font=dict(color="red", size=10))
            st.plotly_chart(_fig, use_container_width=True)

        with _ec2:
            _fig = px.histogram(
                _elev_data, x="elev_delta_ft", nbins=40,
                color="elev_plausible",
                title="Elevation Delta  (FAA − OSM converted, ft)",
                labels={"elev_delta_ft": "Delta (ft)"},
                color_discrete_map={True: _FAA_COLOR, False: "#EF5350"},
            )
            _fig.add_vline(x=0, line_dash="dash", line_color="green",
                           annotation_text="zero delta")
            _fig.update_layout(bargap=0.05, yaxis_title="Pairs",
                                legend_title="|Delta| <= 100 ft")
            st.plotly_chart(_fig, use_container_width=True)

        _em1, _em2, _em3 = st.columns(3)
        _em1.metric("Median |elevation delta|", f"{_med_delta:.0f} ft",
                    help="Lower = stronger cross-source agreement")
        _em2.metric(
            "Plausible pairs  (|Δ| ≤ 100 ft)",
            f"{int(_elev_data['elev_plausible'].sum()):,} / {_n_ele:,}",
            delta=f"{_pct_plaus:.0f}%",
        )
        _em3.metric("OSM values likely in feet", f"{_n_suspect:,}",
                    delta=f"−{_pct_suspect:.0f}% quality flag",
                    delta_color="inverse")

        st.info(
            f"💡 **Insight — High agreement with identifiable outliers.**  "
            f"**{_pct_plaus:.0f}% of matched pairs** agree within 100 ft, confirming "
            f"strong cross-source elevation signal. The median absolute delta is "
            f"**{_med_delta:.0f} ft** — well within standard obstacle-clearance margins. "
            f"{_n_suspect} OSM records ({_pct_suspect:.0f}%) appear to store elevation in "
            f"**feet** rather than the OSM-standard metres: a systematic quality issue "
            f"the HIE pipeline flags and corrects before using elevation as an ML feature."
        )

    st.divider()

    # ---- Chart 3: Location Deviation for Matched Pairs --------------------------
    _prox_only = cons[cons["match_method"] == "proximity"].dropna(subset=["distance_m"])
    _id_only   = cons[cons["match_method"] == "faa_id"].dropna(subset=["distance_m"])

    st.markdown("#### 3 — Location Deviation for Matched Pairs")

    _ldc1, _ldc2 = st.columns(2)
    with _ldc1:
        if not _id_only.empty:
            _id_med = _id_only["distance_m"].median()
            _id_p90 = _id_only["distance_m"].quantile(0.90)
            _fig = px.histogram(
                _id_only, x="distance_m", nbins=30,
                title=f"FAA-ID Exact Matches  (N={len(_id_only):,})",
                labels={"distance_m": "Distance FAA ↔ OSM (m)", "count": "Pairs"},
                color_discrete_sequence=[_OSM_COLOR],
            )
            _fig.add_vline(x=_id_med, line_dash="dash", line_color="white",
                           annotation_text=f"median {_id_med:.0f} m")
            _fig.update_layout(bargap=0.05, yaxis_title="Matched pairs")
            st.plotly_chart(_fig, use_container_width=True)
            st.caption(
                f"Median **{_id_med:.0f} m** · 90th pct **{_id_p90:.0f} m** ·"
                f" FAA-ID tag guarantees same physical pad"
            )
        else:
            st.info("No FAA-ID exact matches in current dataset.")

    with _ldc2:
        if not _prox_only.empty:
            _prox_med = _prox_only["distance_m"].median()
            _prox_p90 = _prox_only["distance_m"].quantile(0.90)
            _pct_u30  = (_prox_only["distance_m"] < 30).mean() * 100
            _fig = px.histogram(
                _prox_only, x="distance_m", nbins=40,
                title=f"Proximity Matches  (N={len(_prox_only):,}, ≤{prox_threshold} m)",
                labels={"distance_m": "Distance FAA ↔ OSM (m)", "count": "Pairs"},
                color_discrete_sequence=[_FAA_COLOR],
            )
            _fig.add_vline(x=_prox_med, line_dash="dash", line_color="white",
                           annotation_text=f"median {_prox_med:.0f} m")
            _fig.update_layout(bargap=0.05, yaxis_title="Matched pairs")
            st.plotly_chart(_fig, use_container_width=True)
            st.caption(
                f"Median **{_prox_med:.0f} m** · 90th pct **{_prox_p90:.0f} m** ·"
                f" {_pct_u30:.0f}% of pairs within 30 m"
            )
        else:
            st.info("No proximity matches in current dataset.")

    if not _id_only.empty and not _prox_only.empty:
        _id_med2   = _id_only["distance_m"].median()
        _prox_med2 = _prox_only["distance_m"].median()
        _pct_u30b  = (_prox_only["distance_m"] < 30).mean() * 100
        st.info(
            f"💡 **Insight — Near-perfect for ID matches; good accuracy for proximity.**  "
            f"FAA-ID matches cluster at **{_id_med2:.0f} m median** — near-zero deviation "
            f"confirms both sources describe the same physical pad. "
            f"Proximity matches show **{_prox_med2:.0f} m median** with "
            f"**{_pct_u30b:.0f}%** of pairs within a single helipad-width (30 m). "
            f"The long tail (>50 m) marks decommissioned pads or OSM coordinate drift "
            f"— high-priority candidates for the M3 Grounding DINO validation pass."
        )

    st.divider()

    # ── KPI: Helipad Coverage & Access Time ──────────────────────────────────────
    st.markdown("### KPI: Helipad Coverage & Access Time for AAM Routing")

    st.markdown("""
<div style="background:#071a2e;border:2px solid #22c55e;border-radius:10px;padding:16px 20px;margin-bottom:16px">
  <div style="font-size:13px;color:#4ade80;font-weight:700;letter-spacing:.5px;margin-bottom:10px">
    🎯 KPI DEFINITION
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <tr>
      <td style="color:#94a3b8;font-weight:600;padding:5px 12px 5px 0;width:90px;vertical-align:top">MODEL</td>
      <td style="color:#e2e8f0;padding:5px 0">
        HIE validation pipeline — YOLO11m visual detection on FAA candidates (Phase 1)
        followed by YOLO11m cascade on OSM-only pads (Phase 2) — promoting visually
        confirmed helipads into the live routing pool.
      </td>
    </tr>
    <tr>
      <td style="color:#94a3b8;font-weight:600;padding:5px 12px 5px 0;vertical-align:top">INDICATOR</td>
      <td style="color:#e2e8f0;padding:5px 0">
        <b style="color:#22c55e">Δ Avg First-Mile Time</b> — reduction in average drive time from a demand point
        (business centre or executive residence) to its nearest validated helipad, in minutes
        at 30 km/h city speed, comparing the <em>FAA-only baseline</em> against the
        <em>FAA + HIE-validated OSM pool</em>.
      </td>
    </tr>
    <tr>
      <td style="color:#94a3b8;font-weight:600;padding:5px 12px 5px 0;vertical-align:top">BECAUSE</td>
      <td style="color:#e2e8f0;padding:5px 0">
        The first-mile ground leg is the dominant bottleneck in door-to-door AAM journeys.
        Every minute saved driving to a helipad is a direct, quantifiable improvement to
        total journey time — the core value proposition for the traveller.
      </td>
    </tr>
  </table>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        "FAA and OSM each tell a partial story. Mapping helipad density against "
        "**business demand hotspots** and **executive residential zones** reveals "
        "coverage gaps where validating OSM helipads shortens the ground-access leg "
        "and makes aerial routing competitive — for both the office commute *and* the "
        "home-to-helipad first mile."
    )

    # ── POI definitions ───────────────────────────────────────────────────
    _BIZ_POIS = [
        {"lat": 40.7589, "lng": -73.9851, "name": "Midtown Manhattan",             "cat": "biz"},
        {"lat": 40.7127, "lng": -74.0059, "name": "Financial District (Wall St)",  "cat": "biz"},
        {"lat": 40.7504, "lng": -73.9967, "name": "Hudson Yards",                  "cat": "biz"},
        {"lat": 40.7531, "lng": -73.9772, "name": "Grand Central / Park Ave",      "cat": "biz"},
        {"lat": 40.7580, "lng": -73.9855, "name": "Rockefeller Center",            "cat": "biz"},
        {"lat": 40.7282, "lng": -74.0776, "name": "Jersey City Financial Center",  "cat": "biz"},
        {"lat": 40.7357, "lng": -74.1724, "name": "Newark, NJ",                    "cat": "biz"},
        {"lat": 40.7456, "lng": -74.3204, "name": "Short Hills, NJ (Corp. Park)",  "cat": "biz"},
        {"lat": 41.0253, "lng": -73.6282, "name": "Greenwich, CT",                 "cat": "biz"},
        {"lat": 41.0534, "lng": -73.5387, "name": "Stamford, CT",                  "cat": "biz"},
        {"lat": 41.1220, "lng": -73.7949, "name": "White Plains, NY",              "cat": "biz"},
        {"lat": 41.0662, "lng": -73.8987, "name": "Tarrytown, NY",                 "cat": "biz"},
        {"lat": 40.7606, "lng": -73.8296, "name": "LaGuardia Airport (LGA)",       "cat": "airport"},
        {"lat": 40.6413, "lng": -73.7781, "name": "JFK Airport",                   "cat": "airport"},
        {"lat": 40.6895, "lng": -74.1745, "name": "Newark Liberty (EWR)",          "cat": "airport"},
        {"lat": 41.0673, "lng": -73.7076, "name": "Westchester Airport (HPN)",     "cat": "airport"},
    ]

    # Executive residential zones — representative neighbourhoods where
    # business travellers matching the Miles Urban persona live.
    _RESIDENTIAL_POIS = [
        {"lat": 40.7736, "lng": -73.9566, "name": "Upper East Side",               "cat": "home"},
        {"lat": 40.7870, "lng": -73.9754, "name": "Upper West Side",               "cat": "home"},
        {"lat": 40.7195, "lng": -74.0089, "name": "Tribeca",                       "cat": "home"},
        {"lat": 40.7339, "lng": -74.0057, "name": "West Village",                  "cat": "home"},
        {"lat": 40.7614, "lng": -73.9776, "name": "Sutton Place",                  "cat": "home"},
        {"lat": 40.7831, "lng": -73.9712, "name": "Carnegie Hill",                 "cat": "home"},
        {"lat": 40.7281, "lng": -73.9944, "name": "Chelsea / Hudson Square",       "cat": "home"},
        {"lat": 40.6960, "lng": -73.9936, "name": "Brooklyn Heights",              "cat": "home"},
        {"lat": 40.7440, "lng": -74.0324, "name": "Hoboken, NJ",                   "cat": "home"},
        {"lat": 40.9176, "lng": -73.8282, "name": "Bronxville, NY",                "cat": "home"},
        {"lat": 40.9895, "lng": -73.7776, "name": "Scarsdale, NY",                 "cat": "home"},
        {"lat": 41.0253, "lng": -73.6282, "name": "Greenwich, CT (res.)",          "cat": "home"},
        {"lat": 40.7957, "lng": -73.7269, "name": "Great Neck, NY",                "cat": "home"},
        {"lat": 40.9799, "lng": -73.6876, "name": "Rye, NY",                       "cat": "home"},
    ]

    # ── data prep ─────────────────────────────────────────────────────────
    faa_v = faa_raw.dropna(subset=["lat", "lon"]).copy()
    osm_v = osm_raw.dropna(subset=["lat", "lon"]).copy()

    # OSM-only pads: not within 250 m of any FAA pad → M3 validation candidates
    if len(faa_v) > 0 and len(osm_v) > 0:
        _d_check = haversine_matrix(
            osm_v["lat"].values, osm_v["lon"].values,
            faa_v["lat"].values, faa_v["lon"].values,
        )
        _osm_only_mask = _d_check.min(axis=1) > 250
        osm_only_v = osm_v[_osm_only_mask].copy()
    else:
        osm_only_v = osm_v.copy()

    # combined pool: FAA + OSM-only (all unique validated-or-to-be-validated pads)
    _comb_lats = np.concatenate([faa_v["lat"].values, osm_only_v["lat"].values])
    _comb_lons = np.concatenate([faa_v["lon"].values, osm_only_v["lon"].values])

    CITY_SPEED_KMH   = 30.0
    AERIAL_SPEED_KMH = 250.0
    COVERAGE_M       = 5_000   # 5 km ≈ ~10 min drive threshold

    # ── coverage gap statistics (all demand points) ───────────────────────
    _all_demand_lats = np.array([p["lat"] for p in _BIZ_POIS + _RESIDENTIAL_POIS])
    _all_demand_lons = np.array([p["lng"] for p in _BIZ_POIS + _RESIDENTIAL_POIS])
    _ddist_faa  = haversine_matrix(_all_demand_lats, _all_demand_lons,
                                   faa_v["lat"].values, faa_v["lon"].values)
    _ddist_comb = haversine_matrix(_all_demand_lats, _all_demand_lons,
                                   _comb_lats, _comb_lons)
    _in_faa  = int((_ddist_faa.min(axis=1)  <= COVERAGE_M).sum())
    _in_comb = int((_ddist_comb.min(axis=1) <= COVERAGE_M).sum())
    _tot_d   = len(_all_demand_lats)

    # ── density map ───────────────────────────────────────────────────────
    dm = folium.Map(location=[40.8, -73.8], zoom_start=8, tiles=None, zoom_control=False)
    folium.TileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; '
             '<a href="https://carto.com/attributions">CARTO</a>',
        name="Dark Map", subdomains="abcd", max_zoom=20,
    ).add_to(dm)

    fg_faa_h = folium.FeatureGroup(name="🔵 FAA Helipad Density", show=True)
    HeatMap(
        faa_v[["lat", "lon"]].values.tolist(),
        radius=22, blur=18, min_opacity=0.35,
        gradient={"0.3": "#1565C0", "0.6": "#42A5F5", "1.0": "#E3F2FD"},
    ).add_to(fg_faa_h)
    fg_faa_h.add_to(dm)

    fg_osm_h = folium.FeatureGroup(name="🟠 OSM Helipad Density", show=True)
    HeatMap(
        osm_v[["lat", "lon"]].values.tolist(),
        radius=22, blur=18, min_opacity=0.35,
        gradient={"0.3": "#E65100", "0.6": "#FFA726", "1.0": "#FFF8E1"},
    ).add_to(fg_osm_h)
    fg_osm_h.add_to(dm)

    # Coverage rings (off by default — toggle to see spatial gaps)
    fg_faa_cov = folium.FeatureGroup(name="🔵 FAA 5 km coverage rings", show=False)
    for _, _cr in faa_v.iterrows():
        folium.Circle(
            [_cr["lat"], _cr["lon"]], radius=5000,
            color="#42A5F5", weight=1, opacity=0.25,
            fill=True, fill_color="#1565C0", fill_opacity=0.04,
        ).add_to(fg_faa_cov)
    fg_faa_cov.add_to(dm)

    fg_osm_cov = folium.FeatureGroup(name="⭐ OSM-only 5 km extension rings", show=False)
    for _, _cr in osm_only_v.iterrows():
        folium.Circle(
            [_cr["lat"], _cr["lon"]], radius=5000,
            color="#FFD700", weight=1, opacity=0.35,
            fill=True, fill_color="#FFD700", fill_opacity=0.06,
        ).add_to(fg_osm_cov)
    fg_osm_cov.add_to(dm)

    fg_faa_pts = folium.FeatureGroup(name="FAA pads (points)", show=False)
    for _, r in faa_v.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=4,
            color="#1565C0", fill=True, fill_color="#42A5F5", fill_opacity=0.7,
            tooltip=str(r.get("NAME", r.get("IDENT", "FAA"))),
        ).add_to(fg_faa_pts)
    fg_faa_pts.add_to(dm)

    fg_osm_pts = folium.FeatureGroup(name="OSM pads (points)", show=False)
    for _, r in osm_v.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=4,
            color="#E65100", fill=True, fill_color="#FFA726", fill_opacity=0.7,
            tooltip=str(r.get("name", "OSM helipad")),
        ).add_to(fg_osm_pts)
    fg_osm_pts.add_to(dm)

    fg_biz = folium.FeatureGroup(name="🏢 Business Centres & Airports", show=True)
    for poi in _BIZ_POIS:
        color = "purple" if poi["cat"] == "biz" else "darkblue"
        icon_name = "briefcase" if poi["cat"] == "biz" else "plane"
        folium.Marker(
            [poi["lat"], poi["lng"]],
            tooltip=poi["name"],
            popup=folium.Popup(f"<b>{poi['name']}</b>", max_width=180),
            icon=folium.Icon(color=color, icon=icon_name, prefix="fa"),
        ).add_to(fg_biz)
    fg_biz.add_to(dm)

    fg_res_dm = folium.FeatureGroup(name="🏠 Executive Residences", show=True)
    for poi in _RESIDENTIAL_POIS:
        folium.Marker(
            [poi["lat"], poi["lng"]],
            tooltip=poi["name"],
            popup=folium.Popup(f"<b>🏠 {poi['name']}</b>", max_width=180),
            icon=folium.Icon(color="lightblue", icon="home", prefix="fa"),
        ).add_to(fg_res_dm)
    fg_res_dm.add_to(dm)

    fg_osm_only = folium.FeatureGroup(name="⭐ OSM-only pads (M3 candidates)", show=True)
    for _, r in osm_only_v.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=6,
            color="#FFD700", fill=True, fill_color="#FFD700", fill_opacity=0.75, weight=2,
            tooltip=f"⭐ OSM-only: {str(r.get('name', 'unnamed'))}",
        ).add_to(fg_osm_only)
    fg_osm_only.add_to(dm)

    folium.LayerControl(collapsed=False).add_to(dm)

    from branca.element import Element as _El
    dm.get_root().html.add_child(_El(
        "<script>(function(){var n=0,iv=setInterval(function(){n++;"
        "var el=document.querySelector('.leaflet-container');"
        "if(el){var m=window[el.id];if(m&&typeof m.addLayer==='function'){"
        "if(!m._szZ){m._szZ=true;L.control.zoom({position:'topleft'}).addTo(m);"
            "if((window.devicePixelRatio||1)%1!==0)m.options.zoomAnimation=false;}"
        "var s=m.getSize();if(s.x===0||s.y===0)m.invalidateSize(false);}}"
        "if(n>50)clearInterval(iv);},200);})();</script>"
    ))

    # ── render density map + KPI panel ───────────────────────────────────
    col_map, col_kpi = st.columns([3, 1])
    with col_map:
        st_folium(dm, width=None, height=510, returned_objects=[], key="density_map")
    with col_kpi:
        st.metric("FAA helipads", f"{len(faa_v):,}")
        st.metric("OSM helipads", f"{len(osm_v):,}")
        ratio = len(osm_v) / max(len(faa_v), 1)
        st.metric("OSM / FAA ratio", f"{ratio:.1f}×")
        st.metric("⭐ M3 candidates", f"{len(osm_only_v):,}",
                  help="OSM-only pads >250 m from any FAA pad — Grounding DINO targets")
        st.divider()
        st.metric(
            "Demand pts within 5 km",
            f"{_in_faa} / {_tot_d}  FAA-only",
            delta=f"+{_in_comb - _in_faa} with OSM",
            delta_color="normal",
            help="Business centres + executive residences within ~10 min drive of a helipad",
        )
        st.markdown(
            "<div style='background:#071a2e;border-left:3px solid #FFD700;"
            "border-radius:6px;padding:8px 10px;font-size:11px;margin-top:8px'>"
            "<b style='color:#FFD700'>Tip: coverage rings</b><br>"
            "<span style='color:#cbd5e1'>Enable '🔵 FAA 5 km coverage rings' + "
            "'⭐ OSM-only 5 km extension rings' to see spatial gaps.</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Business hot spot analysis ────────────────────────────────────────
    st.markdown("### 🏢 Business Centre Access — Nearest Helipad")
    st.caption(
        "Blue dashed → nearest FAA pad · Orange → nearest OSM pad (similar) · "
        "Green → OSM meaningfully closer (routing gain if validated)"
    )

    poi_lats = np.array([p["lat"] for p in _BIZ_POIS])
    poi_lons = np.array([p["lng"] for p in _BIZ_POIS])

    dist_faa_m = haversine_matrix(poi_lats, poi_lons,
                                  faa_v["lat"].values, faa_v["lon"].values)
    dist_osm_m = haversine_matrix(poi_lats, poi_lons,
                                  osm_v["lat"].values, osm_v["lon"].values)

    hs_rows = []
    hs_geo  = []

    for i, poi in enumerate(_BIZ_POIS):
        fi = int(dist_faa_m[i].argmin())
        oi = int(dist_osm_m[i].argmin())
        faa_km   = dist_faa_m[i, fi] / 1000
        osm_km   = dist_osm_m[i, oi] / 1000
        faa_row  = faa_v.iloc[fi]
        osm_row  = osm_v.iloc[oi]
        faa_name = str(faa_row.get("NAME", faa_row.get("IDENT", "—")))[:28]
        osm_name = str(osm_row.get("name", "unnamed"))[:28]
        delta_km  = faa_km - osm_km
        saved_min = max(delta_km / CITY_SPEED_KMH * 60, 0)
        osm_better = delta_km > 0.3

        hs_rows.append({
            "Business Centre":    poi["name"],
            "OSM better?":        "✅" if osm_better else "—",
            "Nearest FAA pad":    f"{faa_name} ({faa_km:.1f} km)",
            "Nearest OSM pad":    f"{osm_name} ({osm_km:.1f} km)",
            "OSM closer by (km)": round(delta_km, 2),
            "Time saved (min)":   round(saved_min, 1) if saved_min > 0.1 else 0.0,
        })
        hs_geo.append({
            "poi":      poi,
            "faa_lat":  float(faa_row["lat"]), "faa_lon": float(faa_row["lon"]),
            "osm_lat":  float(osm_row["lat"]), "osm_lon": float(osm_row["lon"]),
            "faa_name": faa_name, "faa_km": faa_km,
            "osm_name": osm_name, "osm_km": osm_km,
            "osm_better": osm_better, "saved_min": saved_min,
        })

    # spider map — business
    sm = folium.Map(location=[40.85, -73.8], zoom_start=8, tiles=None, zoom_control=False)
    folium.TileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; OpenStreetMap &copy; CARTO',
        name="Dark", subdomains="abcd", max_zoom=20,
    ).add_to(sm)

    fg_faa_lines = folium.FeatureGroup(name="🔵 FAA connections", show=True)
    fg_osm_lines = folium.FeatureGroup(name="🟢 OSM connections (better)", show=True)
    fg_osm_same  = folium.FeatureGroup(name="🟠 OSM connections (similar)", show=True)
    fg_pois_sm   = folium.FeatureGroup(name="🏢 Business centres", show=True)

    seen_faa_pads = {}
    seen_osm_pads = {}

    for g in hs_geo:
        poi = g["poi"]
        plat, plng = poi["lat"], poi["lng"]

        folium.PolyLine(
            [[plat, plng], [g["faa_lat"], g["faa_lon"]]],
            color="#42A5F5", weight=2, opacity=0.7, dash_array="6 4",
            tooltip=f"{poi['name']} → {g['faa_name']} ({g['faa_km']:.1f} km)",
        ).add_to(fg_faa_lines)

        faa_key = (round(g["faa_lat"], 4), round(g["faa_lon"], 4))
        if faa_key not in seen_faa_pads:
            seen_faa_pads[faa_key] = True
            folium.CircleMarker(
                [g["faa_lat"], g["faa_lon"]], radius=5,
                color="#1565C0", fill=True, fill_color="#42A5F5", fill_opacity=0.9,
                tooltip=f"FAA: {g['faa_name']}",
            ).add_to(fg_faa_lines)

        osm_color   = "#22c55e" if g["osm_better"] else "#FFA726"
        osm_weight  = 3        if g["osm_better"] else 2
        osm_opacity = 0.9      if g["osm_better"] else 0.5
        osm_tip = (
            f"{poi['name']} → {g['osm_name']} ({g['osm_km']:.1f} km)"
            + (f" — saves {g['saved_min']:.1f} min if validated" if g["osm_better"] else "")
        )
        target_fg = fg_osm_lines if g["osm_better"] else fg_osm_same
        folium.PolyLine(
            [[plat, plng], [g["osm_lat"], g["osm_lon"]]],
            color=osm_color, weight=osm_weight, opacity=osm_opacity,
            tooltip=osm_tip,
        ).add_to(target_fg)

        osm_key = (round(g["osm_lat"], 4), round(g["osm_lon"], 4))
        if osm_key not in seen_osm_pads:
            seen_osm_pads[osm_key] = True
            folium.CircleMarker(
                [g["osm_lat"], g["osm_lon"]], radius=5,
                color=osm_color, fill=True, fill_color=osm_color, fill_opacity=0.9,
                tooltip=f"OSM: {g['osm_name']}",
            ).add_to(target_fg)

        biz_color = "purple" if poi["cat"] == "biz" else "darkblue"
        icon_name = "briefcase" if poi["cat"] == "biz" else "plane"
        folium.Marker(
            [plat, plng],
            tooltip=poi["name"],
            icon=folium.Icon(color=biz_color, icon=icon_name, prefix="fa"),
        ).add_to(fg_pois_sm)

    for fg in [fg_faa_lines, fg_osm_lines, fg_osm_same, fg_pois_sm]:
        fg.add_to(sm)
    folium.LayerControl(collapsed=False).add_to(sm)
    sm.get_root().html.add_child(_El(
        "<script>(function(){var n=0,iv=setInterval(function(){n++;"
        "var el=document.querySelector('.leaflet-container');"
        "if(el){var m=window[el.id];if(m&&typeof m.addLayer==='function'){"
        "if(!m._szZ){m._szZ=true;L.control.zoom({position:'topleft'}).addTo(m);"
            "if((window.devicePixelRatio||1)%1!==0)m.options.zoomAnimation=false;}"
        "var s=m.getSize();if(s.x===0||s.y===0)m.invalidateSize(false);}}"
        "if(n>50)clearInterval(iv);},200);})();</script>"
    ))
    st_folium(sm, width=None, height=480, returned_objects=[], key="hotspot_map")

    hs_df = pd.DataFrame(hs_rows).set_index("Business Centre")
    st.dataframe(hs_df, use_container_width=True, height=490)

    n_osm_better = int((hs_df["OSM closer by (km)"] > 0.3).sum())
    if n_osm_better > 0:
        avg_save = hs_df.loc[hs_df["OSM closer by (km)"] > 0.3, "Time saved (min)"].mean()
        max_save = hs_df["Time saved (min)"].max()
        st.markdown(
            f"<div style='background:#071a2e;border-left:4px solid #22c55e;"
            f"border-radius:6px;padding:10px 16px;margin-top:12px;font-size:13px'>"
            f"<span style='color:#22c55e;font-weight:700'>✈ Routing opportunity: </span>"
            f"<span style='color:#22c55e'>"
            f"<b>{n_osm_better} of {len(_BIZ_POIS)} business centres</b> have a closer OSM "
            f"helipad than any FAA-registered pad. Validating these would save an average "
            f"of <b>{avg_save:.1f} min</b> ground access per trip — up to "
            f"<b>{max_save:.0f} min</b> in the best case.</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Executive residential first-mile ──────────────────────────────────
    st.markdown("### 🏠 Executive Residential Access — First Mile to Helipad")
    st.markdown(
        "Miles Urban's journey begins at home. For each representative executive "
        "neighbourhood the map shows driving distance to the nearest helipad — "
        "FAA-only (validated today) versus FAA + OSM (after M3 validation). "
        "Green lines flag where a validated OSM pad would shorten the first mile."
    )

    res_lats = np.array([p["lat"] for p in _RESIDENTIAL_POIS])
    res_lons = np.array([p["lng"] for p in _RESIDENTIAL_POIS])
    dist_res_faa = haversine_matrix(res_lats, res_lons,
                                    faa_v["lat"].values, faa_v["lon"].values)
    dist_res_osm = haversine_matrix(res_lats, res_lons,
                                    osm_v["lat"].values, osm_v["lon"].values)

    res_rows = []
    res_geo  = []
    for i, poi in enumerate(_RESIDENTIAL_POIS):
        fi = int(dist_res_faa[i].argmin())
        oi = int(dist_res_osm[i].argmin())
        faa_km   = dist_res_faa[i, fi] / 1000
        osm_km   = dist_res_osm[i, oi] / 1000
        faa_row  = faa_v.iloc[fi]
        osm_row  = osm_v.iloc[oi]
        faa_name = str(faa_row.get("NAME", faa_row.get("IDENT", "—")))[:28]
        osm_name = str(osm_row.get("name", "unnamed"))[:28]
        delta_km  = faa_km - osm_km
        saved_min = max(delta_km / CITY_SPEED_KMH * 60, 0)
        osm_better = delta_km > 0.3

        res_rows.append({
            "Residence":             poi["name"],
            "OSM better?":           "✅" if osm_better else "—",
            "Nearest FAA pad":       f"{faa_name} ({faa_km:.1f} km)",
            "Nearest OSM pad":       f"{osm_name} ({osm_km:.1f} km)",
            "Closer by (km)":        round(delta_km, 2),
            "First-mile saved (min)": round(saved_min, 1) if saved_min > 0.1 else 0.0,
        })
        res_geo.append({
            "poi":      poi,
            "faa_lat":  float(faa_row["lat"]), "faa_lon": float(faa_row["lon"]),
            "osm_lat":  float(osm_row["lat"]), "osm_lon": float(osm_row["lon"]),
            "faa_name": faa_name, "faa_km": faa_km,
            "osm_name": osm_name, "osm_km": osm_km,
            "osm_better": osm_better, "saved_min": saved_min,
        })

    # spider map — residential
    rm = folium.Map(location=[40.85, -73.8], zoom_start=8, tiles=None, zoom_control=False)
    folium.TileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; OpenStreetMap &copy; CARTO',
        name="Dark", subdomains="abcd", max_zoom=20,
    ).add_to(rm)

    fg_r_faa  = folium.FeatureGroup(name="🔵 FAA connections", show=True)
    fg_r_osm  = folium.FeatureGroup(name="🟢 OSM connections (better)", show=True)
    fg_r_same = folium.FeatureGroup(name="🟠 OSM connections (similar)", show=True)
    fg_r_res  = folium.FeatureGroup(name="🏠 Residences", show=True)

    seen_r_faa = {}
    seen_r_osm = {}

    for g in res_geo:
        poi = g["poi"]
        plat, plng = poi["lat"], poi["lng"]

        folium.PolyLine(
            [[plat, plng], [g["faa_lat"], g["faa_lon"]]],
            color="#42A5F5", weight=2, opacity=0.7, dash_array="6 4",
            tooltip=f"{poi['name']} → {g['faa_name']} ({g['faa_km']:.1f} km)",
        ).add_to(fg_r_faa)

        key_f = (round(g["faa_lat"], 4), round(g["faa_lon"], 4))
        if key_f not in seen_r_faa:
            seen_r_faa[key_f] = True
            folium.CircleMarker(
                [g["faa_lat"], g["faa_lon"]], radius=5,
                color="#1565C0", fill=True, fill_color="#42A5F5", fill_opacity=0.9,
                tooltip=f"FAA: {g['faa_name']}",
            ).add_to(fg_r_faa)

        osm_color   = "#22c55e" if g["osm_better"] else "#FFA726"
        osm_weight  = 3        if g["osm_better"] else 2
        osm_opacity = 0.9      if g["osm_better"] else 0.5
        osm_tip = (
            f"{poi['name']} → {g['osm_name']} ({g['osm_km']:.1f} km)"
            + (f" — saves {g['saved_min']:.1f} min if validated" if g["osm_better"] else "")
        )
        target_r = fg_r_osm if g["osm_better"] else fg_r_same
        folium.PolyLine(
            [[plat, plng], [g["osm_lat"], g["osm_lon"]]],
            color=osm_color, weight=osm_weight, opacity=osm_opacity,
            tooltip=osm_tip,
        ).add_to(target_r)

        key_o = (round(g["osm_lat"], 4), round(g["osm_lon"], 4))
        if key_o not in seen_r_osm:
            seen_r_osm[key_o] = True
            folium.CircleMarker(
                [g["osm_lat"], g["osm_lon"]], radius=5,
                color=osm_color, fill=True, fill_color=osm_color, fill_opacity=0.9,
                tooltip=f"OSM: {g['osm_name']}",
            ).add_to(target_r)

        folium.Marker(
            [plat, plng],
            tooltip=poi["name"],
            icon=folium.Icon(color="lightblue", icon="home", prefix="fa"),
        ).add_to(fg_r_res)

    for fg in [fg_r_faa, fg_r_osm, fg_r_same, fg_r_res]:
        fg.add_to(rm)
    folium.LayerControl(collapsed=False).add_to(rm)
    rm.get_root().html.add_child(_El(
        "<script>(function(){var n=0,iv=setInterval(function(){n++;"
        "var el=document.querySelector('.leaflet-container');"
        "if(el){var m=window[el.id];if(m&&typeof m.addLayer==='function'){"
        "if(!m._szZ){m._szZ=true;L.control.zoom({position:'topleft'}).addTo(m);"
            "if((window.devicePixelRatio||1)%1!==0)m.options.zoomAnimation=false;}"
        "var s=m.getSize();if(s.x===0||s.y===0)m.invalidateSize(false);}}"
        "if(n>50)clearInterval(iv);},200);})();</script>"
    ))
    st_folium(rm, width=None, height=460, returned_objects=[], key="res_spider_map")

    res_df = pd.DataFrame(res_rows).set_index("Residence")
    st.dataframe(res_df, use_container_width=True, height=460)

    n_res_better = int((res_df["Closer by (km)"] > 0.3).sum())
    if n_res_better > 0:
        avg_res = res_df.loc[res_df["Closer by (km)"] > 0.3, "First-mile saved (min)"].mean()
        max_res = res_df["First-mile saved (min)"].max()
        st.markdown(
            f"<div style='background:#071a2e;border-left:4px solid #22c55e;"
            f"border-radius:6px;padding:10px 16px;margin-top:12px;font-size:13px'>"
            f"<span style='color:#22c55e;font-weight:700'>🏠 First-mile opportunity: </span>"
            f"<span style='color:#22c55e'>"
            f"<b>{n_res_better} of {len(_RESIDENTIAL_POIS)} executive residences</b> could "
            f"reach a helipad faster with a validated OSM pad. "
            f"Average first-mile saving: <b>{avg_res:.1f} min</b> — "
            f"up to <b>{max_res:.0f} min</b> in the best case.</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Full door-to-door journey KPI ─────────────────────────────────────
    st.markdown("### ✈ Full Door-to-Door Journey — FAA-Only vs FAA + Validated OSM")
    st.markdown(
        "Combining first-mile (home → helipad) and last-mile (helipad → office) savings. "
        "Ground-only drive shown for reference; **✈ faster** marks corridors where aerial "
        "beats driving even before OSM validation."
    )

    _poi_lookup = {p["name"]: p for p in _BIZ_POIS + _RESIDENTIAL_POIS}

    # Corridor pairs: (home, destination) — chosen where haversine > 15 km
    _JOURNEY_PAIRS = [
        ("Bronxville, NY",        "Midtown Manhattan"),
        ("Scarsdale, NY",         "Financial District (Wall St)"),
        ("Greenwich, CT (res.)",  "Midtown Manhattan"),
        ("Rye, NY",               "Financial District (Wall St)"),
        ("Great Neck, NY",        "Hudson Yards"),
        ("Upper East Side",       "Greenwich, CT"),
        ("Hoboken, NJ",           "Westchester Airport (HPN)"),
        ("Brooklyn Heights",      "Midtown Manhattan"),
    ]

    jrn_rows = []
    for (home_name, work_name) in _JOURNEY_PAIRS:
        if home_name not in _poi_lookup or work_name not in _poi_lookup:
            continue
        H = _poi_lookup[home_name]
        W = _poi_lookup[work_name]

        # direct drive (haversine × 1.4 road detour factor)
        dir_m = haversine_matrix(
            np.array([H["lat"]]), np.array([H["lng"]]),
            np.array([W["lat"]]), np.array([W["lng"]]),
        )[0, 0]
        drive_min = (dir_m / 1000 * 1.4) / CITY_SPEED_KMH * 60

        # FAA-only aerial: nearest FAA pad to home + flight + nearest FAA pad to work
        dh_faa = haversine_matrix(np.array([H["lat"]]), np.array([H["lng"]]),
                                  faa_v["lat"].values, faa_v["lon"].values)
        dw_faa = haversine_matrix(np.array([W["lat"]]), np.array([W["lng"]]),
                                  faa_v["lat"].values, faa_v["lon"].values)
        hi_f = int(dh_faa[0].argmin()); wi_f = int(dw_faa[0].argmin())
        fly_faa_km = haversine_matrix(
            np.array([faa_v.iloc[hi_f]["lat"]]), np.array([faa_v.iloc[hi_f]["lon"]]),
            np.array([faa_v.iloc[wi_f]["lat"]]), np.array([faa_v.iloc[wi_f]["lon"]]),
        )[0, 0] / 1000
        faa_aerial_min = (
            (dh_faa[0, hi_f] + dw_faa[0, wi_f]) / 1000 / CITY_SPEED_KMH * 60
            + fly_faa_km / AERIAL_SPEED_KMH * 60
        )

        # FAA+OSM aerial: use combined pad pool (_comb_lats/_comb_lons)
        dh_c = haversine_matrix(np.array([H["lat"]]), np.array([H["lng"]]),
                                _comb_lats, _comb_lons)
        dw_c = haversine_matrix(np.array([W["lat"]]), np.array([W["lng"]]),
                                _comb_lats, _comb_lons)
        hi_c = int(dh_c[0].argmin()); wi_c = int(dw_c[0].argmin())
        fly_comb_km = haversine_matrix(
            np.array([_comb_lats[hi_c]]), np.array([_comb_lons[hi_c]]),
            np.array([_comb_lats[wi_c]]), np.array([_comb_lons[wi_c]]),
        )[0, 0] / 1000
        comb_aerial_min = (
            (dh_c[0, hi_c] + dw_c[0, wi_c]) / 1000 / CITY_SPEED_KMH * 60
            + fly_comb_km / AERIAL_SPEED_KMH * 60
        )

        aerial_vs_drive = drive_min - comb_aerial_min
        osm_saves       = faa_aerial_min - comb_aerial_min

        jrn_rows.append({
            "Home":                        home_name,
            "Destination":                 work_name,
            "✈ vs Drive":                  "✈ faster" if aerial_vs_drive > 2 else "Drive ≈ same",
            "Drive only (min)":            round(drive_min),
            "Aerial FAA-only (min)":       round(faa_aerial_min),
            "Aerial FAA+OSM (min)":        round(comb_aerial_min),
            "OSM saves (min)":             round(osm_saves, 1) if osm_saves > 0.1 else 0.0,
        })

    if jrn_rows:
        jrn_df = pd.DataFrame(jrn_rows)
        st.dataframe(jrn_df, use_container_width=True,
                     height=min(390, 60 + len(jrn_rows) * 38))

        total_osm_save = sum(r["OSM saves (min)"] for r in jrn_rows)
        avg_osm_save   = total_osm_save / len(jrn_rows)
        aerial_faster  = sum(1 for r in jrn_rows if r["✈ vs Drive"] == "✈ faster")

        jc1, jc2, jc3 = st.columns(3)
        jc1.metric("Corridors where aerial wins", f"{aerial_faster} / {len(jrn_rows)}")
        jc2.metric("Avg OSM validation saves",
                   f"{avg_osm_save:.1f} min / trip",
                   help="Reduction in total door-to-door aerial time after OSM pad validation")
        jc3.metric("Total saving across corridors", f"{total_osm_save:.0f} min")

        st.markdown(
            "<div style='background:#071a2e;border-left:4px solid #60a5fa;"
            "border-radius:6px;padding:10px 16px;margin-top:12px;font-size:12px'>"
            "<span style='color:#60a5fa;font-weight:700'>📐 Assumptions: </span>"
            "<span style='color:#cbd5e1'>"
            "Ground legs at 30 km/h (NYC metro average). "
            "Flight at 250 km/h haversine (conservative helicopter; eVTOL ~180 km/h narrows margins). "
            "Road distance = haversine × 1.4 detour factor. "
            "FAA + OSM combined pool = FAA pads plus OSM-only pads >250 m from any FAA pad."
            "</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── M3 KPI pipeline ───────────────────────────────────────────────────
    st.markdown("### M3 KPI — Grounding DINO Validation Pipeline")
    st.caption(
        "Upper-bound improvement assumes 100 % validation rate. "
        "Actual M3 KPI tracks the validated fraction and its routing impact across "
        "both business centres and executive residential zones."
    )

    st.markdown(
        "<div style='display:flex;align-items:center;gap:6px;flex-wrap:wrap;"
        "font-family:monospace;font-size:12px;margin:12px 0'>"
        "<div style='background:#1e2a3a;border:1px solid #FFD700;border-radius:6px;"
        "padding:8px 12px;color:#FFD700;text-align:center'>"
        f"⭐<br><b>{len(osm_only_v)}</b><br>OSM-only pads</div>"
        "<div style='color:#475569;font-size:18px'>→</div>"
        "<div style='background:#1e2a3a;border:1px solid #60a5fa;border-radius:6px;"
        "padding:8px 12px;color:#60a5fa;text-align:center'>"
        "🛰️<br><b>ESRI</b><br>image chips<br>(zoom 19)</div>"
        "<div style='color:#475569;font-size:18px'>→</div>"
        "<div style='background:#1e2a3a;border:1px solid #a78bfa;border-radius:6px;"
        "padding:8px 12px;color:#a78bfa;text-align:center'>"
        "🤖<br><b>Grounding DINO</b><br>open-set detection<br>"
        "<i style='font-size:10px'>prompt: &quot;helipad&quot;</i></div>"
        "<div style='color:#475569;font-size:18px'>→</div>"
        "<div style='background:#1e2a3a;border:1px solid #22c55e;border-radius:6px;"
        "padding:8px 12px;color:#22c55e;text-align:center'>"
        "✅<br><b>Validated</b><br>pads added<br>to routing</div>"
        "<div style='color:#475569;font-size:18px'>→</div>"
        "<div style='background:#071a2e;border:2px solid #22c55e;border-radius:6px;"
        "padding:8px 12px;color:#22c55e;text-align:center;font-weight:700'>"
        "📈<br>M3 KPI<br>Δ coverage<br>Δ time saved</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # M3 upper-bound: run against combined demand (biz + residential)
    _m3_all_lats = np.concatenate([poi_lats, res_lats])
    _m3_all_lons = np.concatenate([poi_lons, res_lons])
    _m3_all_names = [p["name"] for p in _BIZ_POIS] + [p["name"] for p in _RESIDENTIAL_POIS]

    m3_rows = []
    if len(osm_only_v) > 0:
        dist_faa_m3  = haversine_matrix(_m3_all_lats, _m3_all_lons,
                                        faa_v["lat"].values, faa_v["lon"].values)
        dist_only_m3 = haversine_matrix(_m3_all_lats, _m3_all_lons,
                                        osm_only_v["lat"].values, osm_only_v["lon"].values)
        for i, name in enumerate(_m3_all_names):
            faa_km  = dist_faa_m3[i].min()  / 1000
            only_km = dist_only_m3[i].min() / 1000
            delta   = faa_km - only_km
            saved   = max(delta / CITY_SPEED_KMH * 60, 0)
            if delta > 0.3:
                m3_rows.append({
                    "Demand Point":               name,
                    "FAA-only access (km)":       round(faa_km, 1),
                    "OSM-only pad access (km)":   round(only_km, 1),
                    "Reduction (km)":             round(delta, 2),
                    "Time saved (min, @30 km/h)": round(saved, 1),
                })

    m3c1, m3c2, m3c3 = st.columns(3)
    m3c1.metric("Validation candidates (M3)", f"{len(osm_only_v):,}",
                help="OSM-only pads to run through Grounding DINO")
    m3c2.metric("Demand points benefiting",
                f"{len(m3_rows)} / {len(_m3_all_names)}",
                help="Business centres + residences where nearest OSM-only pad is >0.3 km closer")
    avg_m3 = (sum(r["Time saved (min, @30 km/h)"] for r in m3_rows) / len(m3_rows)
              if m3_rows else 0)
    m3c3.metric("Avg ground access saving", f"{avg_m3:.1f} min" if avg_m3 else "—",
                help="Upper bound — assumes 100 % Grounding DINO validation rate")

    if m3_rows:
        st.dataframe(
            pd.DataFrame(m3_rows).set_index("Demand Point"),
            use_container_width=True, height=min(420, 60 + len(m3_rows) * 38),
        )
        st.markdown(
            "<div style='background:#0a1628;border:1px solid #a78bfa;border-radius:8px;"
            "padding:10px 16px;margin-top:10px;font-size:12px;color:#c4b5fd'>"
            "<b>Why Grounding DINO?</b> Unlike a fine-tuned classifier it is zero-shot — "
            "it detects helipads from the text prompt <i>&quot;helipad&quot;</i> without "
            "labelled training data, which is scarce for aerial landing pads. "
            "Each OSM-only pad gets a zoom-19 ESRI satellite chip (~0.22 m/px at lat 41°); "
            "pads where Grounding DINO returns a bounding box with confidence ≥ threshold "
            "are promoted to <b>validated</b> and added to the routing engine. "
            "M3 tracks <b>validated count</b> and <b>Δ avg ground-access time</b> vs the "
            "FAA-only baseline shown above.</div>",
            unsafe_allow_html=True,
        )


# ── multi-modal routing simulator ─────────────────────────────────────────────
st.divider()
st.subheader("Multi-Modal Routing Simulator")
st.caption(
    "Click the helicopter button (top-right of map) to find a helipad-to-helipad route, "
    "or the compass button for a full multi-modal comparison. "
    "Uses existing FAA helipad data + OSRM public API."
)
_tomtom_key   = os.getenv("TOMTOM_API_KEY", "")   # routing API (not traffic tiles)
_mapbox_token = os.getenv("MAPBOX_TOKEN", "")      # traffic-day-v2 basemap
_ar = st.session_state.get("_agent_last_route")
_sim_kwargs: dict = dict(
    js_v="m4.17",
    tomtom_key=_tomtom_key,
    mapbox_token=_mapbox_token,
)
if _ar and _ar.get("origin") and _ar.get("destination"):
    _sim_kwargs.update(
        init_lat_a=_ar["origin"]["lat"],  init_lon_a=_ar["origin"]["lon"],
        init_lat_b=_ar["destination"]["lat"], init_lon_b=_ar["destination"]["lon"],
    )
components.html(
    build_routing_html(faa_raw, osm_raw, osm_validated_df=_load_osm_validated(), **_sim_kwargs),
    height=650, scrolling=False)

st.divider()
st.caption("SkyRoute HIE · Technion LBS Course 016833 · FAA ADDS-ArcGIS + OpenStreetMap")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · HIE Inspector — live YOLO inference on NAIP / ESRI imagery
# ══════════════════════════════════════════════════════════════════════════════

# ── module-level helpers (must be outside the `with` block for caching) ───────

_INSPECTOR_CSV    = DATA_DIR / "inspector_results.csv"
_TEST_IMG_DIR     = DATA_DIR / "yolo_dataset" / "images" / "test"
_TEST_LBL_DIR     = DATA_DIR / "yolo_dataset" / "labels" / "test"
_FAA_NATIONAL_CSV = DATA_DIR / "faa_national.csv"

_INSP_B_SOURCES: dict[str, str] = {
    "FAA — NE US test set (747)": "faa_neus",
    "FAA — CONUS (3 000+)":       "faa_national",
    "OSM — NE US (1 500+)":       "osm_neus",
}


@st.cache_data
def _load_insp_b_helipads(source_key: str) -> pd.DataFrame:
    """Load helipad DataFrame for Live Inference overlay markers."""
    if source_key == "faa_national":
        if _FAA_NATIONAL_CSV.exists():
            df = pd.read_csv(_FAA_NATIONAL_CSV)
            return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        source_key = "faa_neus"   # fall back if national CSV not built yet
    if source_key == "faa_neus":
        p = DATA_DIR / "faa_adip_enriched.csv"
        if not p.exists():
            p = DATA_DIR / "faa_helipads_raw.csv"
        if p.exists():
            df = pd.read_csv(p)
            return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if source_key == "osm_neus":
        p = DATA_DIR / "osm_helipads_raw.csv"
        if p.exists():
            df = pd.read_csv(p)
            return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return pd.DataFrame(columns=["lat", "lon", "name"])


@st.cache_resource(show_spinner="Loading YOLO helipad detector…")
def _load_inspector_model():
    """Load the fine-tuned YOLO model once per session."""
    try:
        return load_yolo_model(YOLO_MODEL_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner="Running YOLO on 747 test chips — first visit only, ~90 s…")
def _get_test_results(model_path_str: str) -> pd.DataFrame:
    """Run inference on all 747 test chips; persist results to inspector_results.csv."""
    # Fast path: return cached CSV if it exists and looks complete
    if _INSPECTOR_CSV.exists():
        cached = pd.read_csv(_INSPECTOR_CSV)
        if len(cached) == 747 and "category" in cached.columns:
            return cached

    _EMPTY_COLS = ["ident", "name", "lat", "lon", "gt", "pred",
                   "detected", "confidence", "bbox_px", "category"]
    if not _TEST_IMG_DIR.exists():
        return pd.DataFrame(columns=_EMPTY_COLS)

    model = load_yolo_model(Path(model_path_str))
    try:
        faa     = pd.read_csv(DATA_DIR / "faa_adip_enriched.csv")
        faa_map = {str(r.IDENT): r for _, r in faa.iterrows()}
    except FileNotFoundError:
        faa_map = {}

    rows = []
    for chip_path in sorted(_TEST_IMG_DIR.glob("*.jpg")):
        ident    = chip_path.stem
        lbl_path = _TEST_LBL_DIR / f"{ident}.txt"

        gt = 0
        if lbl_path.exists():
            gt = 1 if any(ln.startswith("0 ") for ln in lbl_path.read_text().splitlines()) else 0

        image  = load_chip(chip_path)
        result = detect_yolo(image, model)
        pred   = 1 if result["detected"] and result["confidence"] >= 0.5 else 0

        fr  = faa_map.get(ident)
        lat = float(fr.lat)  if fr is not None else 0.0
        lon = float(fr.lon)  if fr is not None else 0.0
        name = str(fr.NAME) if fr is not None else ident

        cat = {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(gt, pred)]
        rows.append({"ident": ident, "name": name, "lat": lat, "lon": lon,
                     "gt": gt, "pred": pred, "detected": result["detected"],
                     "confidence": round(result["confidence"], 4),
                     "bbox_px": str(result["bbox_px"]) if result["bbox_px"] else "",
                     "category": cat})

    df = pd.DataFrame(rows)
    _INSPECTOR_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_INSPECTOR_CSV, index=False)
    return df


def _show_pil(target, img, caption: str = "") -> None:
    """Display a PIL Image — falls back to use_column_width on older Streamlit."""
    try:
        target.image(img, caption=caption, use_container_width=True)
    except TypeError:
        target.image(img, caption=caption, use_column_width="always")


def _safe_image(target, path: Path, caption: str = "") -> None:
    """Display a plot PNG safely — handles corrupt files and Streamlit version differences."""
    from PIL import Image as _PIL
    try:
        img = _PIL.open(str(path))
        img.load()          # fully decode; raises on truncated/corrupt file
    except Exception as exc:
        target.warning(f"Could not load `{path.name}`: {exc}")
        return
    _show_pil(target, img, caption)


# ── Inspector tab ─────────────────────────────────────────────────────────────
from branca.element import Element as _BE   # used in both Mode A and Mode B maps

# Fragment decorator — isolates Inspector reruns from the heavy EDA maps.
# Falls back to a plain wrapper on Streamlit < 1.33.
try:
    _fragment = st.experimental_fragment
except AttributeError:
    def _fragment(fn):  # type: ignore[misc]
        return fn


@_fragment
def _inspector_content() -> None:
    """Render the full Inspector tab — runs as an isolated fragment."""
    insp_model = _load_inspector_model()
    if insp_model is None:
        st.warning(
            f"YOLO weights not found at `{YOLO_MODEL_PATH}`. "
            "Run training first or copy `models/helipad_yolov8s.pt` to the project root."
        )
        return

    insp_mode = st.radio(
        "Mode",
        ["🔬 Test Set Inspector", "📡 Live Inference", "🗺️ OSM Inspector"],
        horizontal=True,
        key="_insp_mode",
        help=(
            "Test Set Inspector: select a helipad from the dropdown — chip loads automatically. "
            "Live Inference: pan/zoom the map freely; click a marker to fetch a chip and run YOLO. "
            "OSM Inspector: browse OSM-only NE US pads validated by the HIE pipeline — Prev/Next or map click."
        ),
    )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # MODE A — Test Set Inspector
    # ══════════════════════════════════════════════════════════════════════════
    if insp_mode == "🔬 Test Set Inspector":
        st.caption(
            "Select a helipad from the dropdown — chip loads automatically. "
            "Green crosshair = FAA registry coordinate · Red box = YOLO detection."
        )

        test_results = _get_test_results(str(YOLO_MODEL_PATH))
        if test_results.empty:
            st.info(
                "Test set imagery is not available in this deployment. "
                "Run `scripts/build_yolo_dataset.py` locally to generate NAIP chips, "
                "then re-deploy with the generated `data/inspector_results.csv`."
            )
        cat_counts   = test_results.category.value_counts().to_dict()

        ctrl_a, view_a = st.columns([1, 2], gap="medium")

        with ctrl_a:
            st.markdown("#### Navigate test set")
            CAT_LABELS = {
                "TP": f"✅ TP  ({cat_counts.get('TP', 0)})  helipad, detected",
                "TN": f"⬜ TN  ({cat_counts.get('TN', 0)})  no helipad, not detected",
                "FP": f"🔴 FP  ({cat_counts.get('FP', 0)})  no helipad, but detected",
                "FN": f"🟡 FN  ({cat_counts.get('FN', 0)})  helipad, missed",
            }
            sel_cat = st.radio("Category", list(CAT_LABELS.keys()),
                               format_func=lambda x: CAT_LABELS[x], key="_insp_a_cat")
            cat_df  = test_results[test_results.category == sel_cat].sort_values("name")
            opts_a  = [f"{r.ident} — {r.name}" for _, r in cat_df.iterrows()]
            sel_opt = st.selectbox("Helipad", opts_a or ["(none)"], key="_insp_a_sel")

            # Auto-jump: run inference whenever the selection changes
            if opts_a and sel_opt and sel_opt != "(none)":
                ident_a = sel_opt.split(" — ")[0]
                if ident_a != st.session_state.get("_insp_a_ident"):
                    row_a  = cat_df[cat_df.ident == ident_a].iloc[0]
                    cp     = _TEST_IMG_DIR / f"{ident_a}.jpg"
                    chip_a = load_chip(cp) if cp.exists() else None
                    res_a  = detect_yolo(chip_a, insp_model) if chip_a else {
                        "detected": False, "bbox_px": None, "cx": None, "cy": None,
                        "confidence": 0.0, "method": "yolo_finetuned", "latency_s": 0.0,
                    }
                    st.session_state.update({
                        "_insp_a_lat":    float(row_a.lat),
                        "_insp_a_lon":    float(row_a.lon),
                        "_insp_a_ver":    st.session_state.get("_insp_a_ver", 0) + 1,
                        "_insp_a_ident":  ident_a,
                        "_insp_a_chip":   chip_a,
                        "_insp_a_result": res_a,
                    })

            st.divider()
            conf_a = st.slider("Confidence threshold", 0.0, 1.0, 0.50, 0.05, key="_insp_a_thr")

            res_a_now   = st.session_state.get("_insp_a_result")
            ident_a_now = st.session_state.get("_insp_a_ident", "—")
            if res_a_now:
                st.divider()
                st.markdown(f"**{ident_a_now}**")
                if res_a_now["detected"] and res_a_now["confidence"] >= conf_a:
                    st.success(f"✅ Detected   conf = {res_a_now['confidence']:.3f}")
                    lat_a = st.session_state.get("_insp_a_lat", 0.0)
                    lon_a = st.session_state.get("_insp_a_lon", 0.0)
                    if res_a_now["bbox_px"]:
                        dl, dlo = bbox_px_to_latlon(res_a_now["bbox_px"], lat_a, lon_a)
                        st.caption(f"Registry offset: {compute_offset_m(lat_a, lon_a, dl, dlo):.1f} m")
                else:
                    st.info("No detection above threshold")

        with view_a:
            lat_a     = st.session_state.get("_insp_a_lat", 40.7503)
            lon_a     = st.session_state.get("_insp_a_lon", -74.0025)
            ver_a     = st.session_state.get("_insp_a_ver", 0)
            res_a_now = st.session_state.get("_insp_a_result")

            # NAIP chip (left) and reference map (right) — side by side
            _chip_col_a, _map_col_a = st.columns([1, 1], gap="small")

            chip_a_now = st.session_state.get("_insp_a_chip")
            with _chip_col_a:
                if chip_a_now and res_a_now:
                    ann_a = draw_detection(chip_a_now, res_a_now,
                                           source_label="NAIP (disk)", conf_threshold=conf_a)
                    _show_pil(st, ann_a,
                              caption=f"100 m × 100 m NAIP  |  conf={res_a_now['confidence']:.3f}")
                else:
                    st.info("Select a helipad from the dropdown to load the NAIP chip.")

            with _map_col_a:
                # Reference map — dark base by default, light map switchable
                m_a = folium.Map(location=[lat_a, lon_a], zoom_start=16,
                                 tiles="CartoDB dark_matter", zoom_control=False)
                folium.TileLayer(tiles="OpenStreetMap", name="Light map",
                                 overlay=False, control=True).add_to(m_a)
                folium.CircleMarker(
                    location=[lat_a, lon_a], radius=7,
                    color="#4CAF50", fill=True, fill_color="#4CAF50", fill_opacity=0.9,
                    tooltip=f"FAA registry: {st.session_state.get('_insp_a_ident', '')}",
                ).add_to(m_a)
                if res_a_now and res_a_now.get("bbox_px") and res_a_now["confidence"] >= conf_a:
                    folium.Rectangle(
                        bounds=bbox_px_to_bounds(res_a_now["bbox_px"], lat_a, lon_a),
                        color="#ff3c3c", weight=3, fill=False,
                        tooltip=f"YOLO bbox  conf={res_a_now['confidence']:.2f}",
                    ).add_to(m_a)
                folium.LayerControl(collapsed=False).add_to(m_a)
                m_a.get_root().html.add_child(_BE(
                    "<script>(function(){var n=0,iv=setInterval(function(){n++;"
                    "var el=document.querySelector('.leaflet-container');"
                    "if(el){var m=window[el.id];if(m&&typeof m.addLayer==='function'){"
                    "if(!m._szZ){m._szZ=true;L.control.zoom({position:'topleft'}).addTo(m);"
            "if((window.devicePixelRatio||1)%1!==0)m.options.zoomAnimation=false;}"
                    "var s=m.getSize();if(s.x===0||s.y===0)m.invalidateSize(false);}}"
                    "if(n>50)clearInterval(iv);},200);})();</script>"
                ))
                st_folium(m_a, key=f"insp_a_map_{ver_a}", width=None, height=310,
                          returned_objects=[])

    # ══════════════════════════════════════════════════════════════════════════
    # MODE B — Live Inference
    # ══════════════════════════════════════════════════════════════════════════
    elif insp_mode == "📡 Live Inference":
        st.caption(
            "Select a helipad dataset — markers are shown on the map. "
            "Click a marker or pan the map to fetch a 100 m × 100 m chip at that location and run YOLO. "
            "Map zoom does not change the chip geometry — inference always uses the fixed 100 m window."
        )

        ctrl_b, view_b = st.columns([1, 2], gap="medium")

        with ctrl_b:
            st.markdown("#### Helipad overlays")
            show_faa_b   = st.checkbox("FAA — NE US (747)",   value=True,  key="_insp_b_faa")
            show_osm_b   = st.checkbox("OSM — NE US (1 500+)", value=True,  key="_insp_b_osm")
            show_conus_b = st.checkbox("FAA — CONUS (3 000+)", value=False, key="_insp_b_conus",
                                       help="Adds all CONUS helipads — may be slow to render")

            st.markdown("#### Imagery")
            imagery_b = st.radio(
                "Imagery source",
                ["NAIP only", "ESRI only", "NAIP + ESRI"],
                index=2,
                help="NAIP = training domain  ·  ESRI = domain-shift comparison",
                key="_insp_b_imagery",
            )
            auto_infer_b = st.checkbox(
                "Auto-fetch on marker click",
                value=True,
                key="_insp_b_auto",
                help="When checked, clicking a helipad marker immediately fetches its chip and runs YOLO.",
            )
            conf_b = st.slider("Confidence threshold", 0.0, 1.0, 0.50, 0.05, key="_insp_b_thr")

            st.divider()
            lat_b_now = st.session_state.get("_insp_b_lat", 40.843194)
            lon_b_now = st.session_state.get("_insp_b_lon", -75.205319)
            st.caption(f"Centre: {lat_b_now:.4f}°N  {abs(lon_b_now):.4f}°W")
            if st.button("Run Inference Here", use_container_width=True, key="_insp_b_run"):
                st.session_state["_insp_b_force"] = True

            res_b_naip = st.session_state.get("_insp_b_naip_result")
            res_b_esri = st.session_state.get("_insp_b_esri_result")
            if res_b_naip and imagery_b in ("NAIP only", "NAIP + ESRI"):
                st.markdown("**NAIP**")
                if res_b_naip["detected"] and res_b_naip["confidence"] >= conf_b:
                    st.success(f"✅ Detected   conf = {res_b_naip['confidence']:.3f}")
                    if res_b_naip["bbox_px"]:
                        dl, dlo = bbox_px_to_latlon(res_b_naip["bbox_px"], lat_b_now, lon_b_now)
                        st.caption(f"Centre offset: {compute_offset_m(lat_b_now, lon_b_now, dl, dlo):.1f} m")
                else:
                    st.info("No detection above threshold")
            if res_b_esri and imagery_b in ("ESRI only", "NAIP + ESRI"):
                st.markdown("**ESRI**")
                if res_b_esri["detected"] and res_b_esri["confidence"] >= conf_b:
                    st.success(f"✅ Detected   conf = {res_b_esri['confidence']:.3f}")
                else:
                    st.info("No detection above threshold")

        with view_b:
            ver_b = st.session_state.get("_insp_b_ver", 0)

            # ESRI satellite as default base — no blank tiles at any zoom level
            m_b = folium.Map(location=[lat_b_now, lon_b_now], zoom_start=15, tiles=None, zoom_control=False)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="ESRI World Imagery", name="Satellite", control=False,
            ).add_to(m_b)
            folium.TileLayer(tiles="CartoDB dark_matter", name="Dark map",
                             overlay=False, control=True).add_to(m_b)

            # Helipad overlay layers — each dataset as a separate toggleable cluster
            def _add_helipad_layer(m, df, layer_name, color):
                if df.empty:
                    return
                _id_c = "IDENT" if "IDENT" in df.columns else None
                _nm_c = ("NAME"  if "NAME"  in df.columns else
                         "name"  if "name"  in df.columns else None)
                _fg = MarkerCluster(name=layer_name, show=True).add_to(m)
                for _, _hr in df.iterrows():
                    _tid = str(_hr.get(_id_c, "") or "") if _id_c else ""
                    _tnm = str(_hr.get(_nm_c, "") or "") if _nm_c else ""
                    _tip = (_tid + "  " + _tnm).strip() or "helipad"
                    folium.CircleMarker(
                        location=[float(_hr["lat"]), float(_hr["lon"])],
                        radius=4, color=color, fill=True, fill_color=color,
                        fill_opacity=0.75, tooltip=_tip,
                    ).add_to(_fg)

            if show_faa_b:
                _add_helipad_layer(m_b, _load_insp_b_helipads("faa_neus"),
                                   "FAA NE US", "#1565C0")
            if show_osm_b:
                _add_helipad_layer(m_b, _load_insp_b_helipads("osm_neus"),
                                   "OSM NE US", "#E65100")
            if show_conus_b:
                _add_helipad_layer(m_b, _load_insp_b_helipads("faa_national"),
                                   "FAA CONUS", "#6A1B9A")

            # Detection bbox overlays (NAIP = red, ESRI = orange)
            if res_b_naip and res_b_naip.get("bbox_px") and res_b_naip["confidence"] >= conf_b:
                folium.Rectangle(
                    bounds=bbox_px_to_bounds(res_b_naip["bbox_px"], lat_b_now, lon_b_now),
                    color="#ff3c3c", weight=3, fill=False,
                    tooltip=f"YOLO NAIP  conf={res_b_naip['confidence']:.2f}",
                ).add_to(m_b)
            if res_b_esri and res_b_esri.get("bbox_px") and res_b_esri["confidence"] >= conf_b:
                folium.Rectangle(
                    bounds=bbox_px_to_bounds(res_b_esri["bbox_px"], lat_b_now, lon_b_now),
                    color="#FF9800", weight=2, fill=False,
                    tooltip=f"YOLO ESRI  conf={res_b_esri['confidence']:.2f}",
                ).add_to(m_b)
            folium.LayerControl(collapsed=False).add_to(m_b)
            m_b.get_root().html.add_child(_BE(
                "<script>(function(){var n=0,iv=setInterval(function(){n++;"
                "var el=document.querySelector('.leaflet-container');"
                "if(el){var m=window[el.id];if(m&&typeof m.addLayer==='function'){"
                "if(!m._szZ){m._szZ=true;L.control.zoom({position:'topleft'}).addTo(m);"
            "if((window.devicePixelRatio||1)%1!==0)m.options.zoomAnimation=false;}"
                "var s=m.getSize();if(s.x===0||s.y===0)m.invalidateSize(false);}}"
                "if(n>50)clearInterval(iv);},200);})();</script>"
            ))

            map_out_b = st_folium(m_b, key=f"insp_b_map_{ver_b}", width=None, height=390,
                                  returned_objects=["last_object_clicked"])

            # ── Determine inference target ────────────────────────────────────
            # Pan/zoom no longer triggers a rerun — only marker clicks and the
            # "Run Inference Here" button fire inference.
            force_b = st.session_state.get("_insp_b_force", False)
            if force_b:
                del st.session_state["_insp_b_force"]

            new_lat_b, new_lng_b = lat_b_now, lon_b_now
            if map_out_b:
                clicked_b = map_out_b.get("last_object_clicked")
                if clicked_b and isinstance(clicked_b, dict) and "lat" in clicked_b:
                    new_lat_b = float(clicked_b["lat"])
                    new_lng_b = float(clicked_b.get("lng", new_lng_b))
                    if auto_infer_b:
                        force_b = True

            do_infer = force_b

            if do_infer:
                if imagery_b in ("NAIP only", "NAIP + ESRI"):
                    with st.spinner("Fetching NAIP chip…"):
                        cn = fetch_naip_chip(new_lat_b, new_lng_b)
                    if cn:
                        rn = detect_yolo(cn, insp_model)
                        st.session_state["_insp_b_naip_chip"]   = cn
                        st.session_state["_insp_b_naip_result"] = rn
                if imagery_b in ("ESRI only", "NAIP + ESRI"):
                    with st.spinner("Fetching ESRI chip…"):
                        ce = fetch_esri_chip(new_lat_b, new_lng_b)
                    if ce:
                        re = detect_yolo(ce, insp_model)
                        st.session_state["_insp_b_esri_chip"]   = ce
                        st.session_state["_insp_b_esri_result"] = re
                st.session_state["_insp_b_inferred_at"] = (new_lat_b, new_lng_b)
                st.session_state["_insp_b_lat"]         = new_lat_b
                st.session_state["_insp_b_lon"]         = new_lng_b

            # Chip image panels
            n_cols_b   = 2 if imagery_b == "NAIP + ESRI" else 1
            img_cols_b = st.columns(n_cols_b)

            if imagery_b in ("NAIP only", "NAIP + ESRI"):
                cn = st.session_state.get("_insp_b_naip_chip")
                rn = st.session_state.get("_insp_b_naip_result")
                if cn and rn:
                    _show_pil(img_cols_b[0],
                              draw_detection(cn, rn, source_label="NAIP", conf_threshold=conf_b),
                              caption=f"NAIP — training domain  |  conf={rn['confidence']:.3f}")
                else:
                    img_cols_b[0].caption("Click a helipad marker or pan to fetch a NAIP chip.")

            if imagery_b in ("ESRI only", "NAIP + ESRI"):
                ce = st.session_state.get("_insp_b_esri_chip")
                re = st.session_state.get("_insp_b_esri_result")
                ci = 1 if imagery_b == "NAIP + ESRI" else 0
                if ce and re:
                    _show_pil(img_cols_b[ci],
                              draw_detection(ce, re, source_label="ESRI", conf_threshold=conf_b),
                              caption=f"ESRI World Imagery — domain shift  |  conf={re['confidence']:.3f}")
                else:
                    img_cols_b[ci].caption("Click a helipad marker or pan to fetch an ESRI chip.")

    # ══════════════════════════════════════════════════════════════════════════
    # MODE C — OSM Inspector
    # ══════════════════════════════════════════════════════════════════════════
    elif insp_mode == "🗺️ OSM Inspector":
        st.caption(
            "Browse OSM-only NE US helipads validated by the HIE pipeline.  "
            "🟢 Green = YOLO confirmed · 🔴 Red = not detected.  "
            "Navigate with Prev / Next or click any marker on the map."
        )

        osm_val = _load_osm_validated()
        if osm_val is None or osm_val.empty:
            st.warning(
                "No OSM validation data found.  \n"
                "Run:  `python scripts/validate_osm_only.py`"
            )
        else:
            # ── filter ────────────────────────────────────────────────────────
            filt_c, count_c = st.columns([3, 1])
            filter_val = filt_c.radio(
                "Filter",
                ["All", "✅ Detected", "❌ Not Detected"],
                horizontal=True,
                key="_insp_c_filter",
            )

            # Reset index when filter changes
            if st.session_state.get("_insp_c_last_filter") != filter_val:
                st.session_state["_insp_c_idx"]    = 0
                st.session_state["_insp_c_chip"]   = None
                st.session_state["_insp_c_result"] = None
            st.session_state["_insp_c_last_filter"] = filter_val

            if filter_val == "✅ Detected":
                df_c = osm_val[osm_val["hie_visual_detected"]].reset_index(drop=True)
            elif filter_val == "❌ Not Detected":
                df_c = osm_val[~osm_val["hie_visual_detected"]].reset_index(drop=True)
            else:
                df_c = osm_val.copy()

            n_c = len(df_c)
            count_c.markdown(
                f"<div style='text-align:right;padding-top:28px;color:#94a3b8'>"
                f"{n_c:,} pads</div>",
                unsafe_allow_html=True,
            )

            if n_c == 0:
                st.info("No records match the current filter.")
            else:
                idx_c = int(st.session_state.get("_insp_c_idx", 0)) % n_c

                # ── navigation row ────────────────────────────────────────────
                nav1, nav2, nav3 = st.columns([1, 2, 1])
                if nav1.button("◀  Prev", use_container_width=True, key="_insp_c_prev"):
                    idx_c = (idx_c - 1) % n_c
                    st.session_state.update({
                        "_insp_c_idx": idx_c,
                        "_insp_c_chip": None, "_insp_c_result": None,
                    })
                nav2.markdown(
                    f"<div style='text-align:center;font-size:15px;padding-top:8px'>"
                    f"<b>{idx_c + 1}</b> / {n_c}</div>",
                    unsafe_allow_html=True,
                )
                if nav3.button("Next  ▶", use_container_width=True, key="_insp_c_next"):
                    idx_c = (idx_c + 1) % n_c
                    st.session_state.update({
                        "_insp_c_idx": idx_c,
                        "_insp_c_chip": None, "_insp_c_result": None,
                    })

                row_c     = df_c.iloc[idx_c]
                lat_c     = float(row_c["lat"])
                lon_c     = float(row_c["lon"])
                name_c    = row_c["name"] or "Unnamed"
                osm_id_c  = row_c["osm_id"]
                detected_c = bool(row_c["hie_visual_detected"])
                conf_csv_c = float(row_c.get("hie_confidence", 0.0) or 0.0)
                offset_c   = row_c.get("hie_offset_m")

                # ── auto-fetch chip when index changes ────────────────────────
                if st.session_state.get("_insp_c_last_idx") != idx_c or \
                   st.session_state.get("_insp_c_chip") is None:
                    st.session_state["_insp_c_last_idx"] = idx_c
                    st.session_state["_insp_c_ver"] = (
                        st.session_state.get("_insp_c_ver", 0) + 1
                    )
                    with st.spinner("Fetching NAIP chip…"):
                        _chip_c = fetch_naip_chip(lat_c, lon_c)
                    if _chip_c is not None:
                        _res_c = detect_yolo(_chip_c, insp_model)
                        st.session_state["_insp_c_chip"]   = _chip_c
                        st.session_state["_insp_c_result"] = _res_c
                    else:
                        st.session_state["_insp_c_chip"]   = None
                        st.session_state["_insp_c_result"] = None

                chip_c   = st.session_state.get("_insp_c_chip")
                result_c = st.session_state.get("_insp_c_result")
                thr_c    = st.slider(
                    "Confidence threshold", 0.50, 1.0, 0.50, 0.05, key="_insp_c_thr"
                )

                st.divider()
                col_chip_c, col_map_c = st.columns([1, 2], gap="medium")

                # ── left: chip ────────────────────────────────────────────────
                with col_chip_c:
                    if chip_c is not None and result_c is not None:
                        live_conf  = result_c["confidence"]
                        live_det   = result_c["detected"] and live_conf >= thr_c
                        batch_det  = detected_c

                        if live_det:
                            st.success(f"LIVE ✅ Detected  conf = {live_conf:.3f}")
                        else:
                            st.info(f"LIVE ❌ No detection above {thr_c:.2f}")

                        # Flag disagreement between live YOLO and stored batch result
                        if live_det and not batch_det:
                            st.warning(
                                "Batch HIE said **not detected** — live YOLO disagrees. "
                                "Possible: low-confidence detection (0.25–0.49 in batch) "
                                "or imagery change."
                            )
                        elif not live_det and batch_det:
                            st.warning(
                                f"Batch HIE said **detected** (conf {conf_csv_c:.3f}) "
                                "but live YOLO is below threshold at current slider setting."
                            )

                        _show_pil(
                            col_chip_c,
                            draw_detection(chip_c, result_c,
                                           source_label="NAIP", conf_threshold=thr_c),
                            caption=f"NAIP · {name_c[:40]}",
                        )
                    elif chip_c is None:
                        st.warning("NAIP imagery not available for this location")
                    else:
                        st.caption("Fetching…")

                # ── right: info card + map ────────────────────────────────────
                with col_map_c:
                    status_color = "#22c55e" if detected_c else "#ef4444"
                    offset_html  = (
                        f"<br><span style='color:#94a3b8'>Offset:</span> {offset_c:.1f} m"
                        if detected_c and pd.notna(offset_c) else ""
                    )
                    # OSM URL — node vs way depends on osm_type in original data
                    _osm_url = f"https://www.openstreetmap.org/node/{osm_id_c}"
                    st.markdown(
                        f"<div style='background:#0f172a;border:1px solid #1e2a3a;"
                        f"border-radius:8px;padding:12px 16px;margin-bottom:10px;"
                        f"font-size:13px;line-height:1.7'>"
                        f"<b style='font-size:15px'>"
                        f"{name_c if name_c and name_c != 'Unnamed' else '(unnamed)'}"
                        f"</b><br>"
                        f"<span style='color:#94a3b8'>OSM:</span> "
                        f"<a href='{_osm_url}' target='_blank' style='color:#64B5F6'>"
                        f"node/{osm_id_c}</a><br>"
                        f"<span style='color:#94a3b8'>Coords:</span> "
                        f"{lat_c:.5f}°N &nbsp; {abs(lon_c):.5f}°W<br>"
                        f"<span style='color:#94a3b8'>Batch HIE:</span> "
                        f"<span style='color:{status_color}'>"
                        f"{'✅ Detected' if detected_c else '❌ Not detected'} "
                        f"(conf {conf_csv_c:.3f})</span>"
                        f"{offset_html}</div>",
                        unsafe_allow_html=True,
                    )

                    # Folium map — current pad + nearby pads (within 0.5°)
                    ver_c = st.session_state.get("_insp_c_ver", 0)
                    nearby_c = df_c[
                        ((df_c["lat"] - lat_c) ** 2 + (df_c["lon"] - lon_c) ** 2) <= 0.25
                    ]

                    m_c = folium.Map(
                        location=[lat_c, lon_c], zoom_start=13, tiles=None, max_zoom=20,
                        zoom_control=False
                    )
                    folium.TileLayer(
                        "CartoDB dark_matter", name="Dark",
                        overlay=False, control=False, show=True,
                    ).add_to(m_c)
                    folium.TileLayer(
                        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                        attr="OSM", name="Street map",
                        overlay=False, control=True, show=False,
                    ).add_to(m_c)

                    for _, r in nearby_c.iterrows():
                        is_cur  = str(r["osm_id"]) == osm_id_c
                        is_det  = bool(r["hie_visual_detected"])
                        color   = "#22c55e" if is_det else "#ef4444"
                        radius  = 9 if is_cur else 5
                        opacity = 1.0 if is_cur else 0.6
                        tip     = f'{"→ " if is_cur else ""}{r["name"] or r["osm_id"]}'
                        folium.CircleMarker(
                            location=[float(r["lat"]), float(r["lon"])],
                            radius=radius, color=color,
                            fill=True, fill_color=color,
                            fill_opacity=opacity, weight=3 if is_cur else 1,
                            tooltip=tip,
                        ).add_to(m_c)

                    folium.LayerControl(collapsed=True).add_to(m_c)
                    m_c.get_root().html.add_child(_BE(
                        "<script>(function(){var n=0,iv=setInterval(function(){n++;"
                        "var el=document.querySelector('.leaflet-container');"
                        "if(el){var m=window[el.id];if(m&&typeof m.addLayer==='function'){"
                        "if(!m._szZ){m._szZ=true;L.control.zoom({position:'topleft'}).addTo(m);"
            "if((window.devicePixelRatio||1)%1!==0)m.options.zoomAnimation=false;}"
                        "var s=m.getSize();if(s.x===0||s.y===0)m.invalidateSize(false);}}"
                        "if(n>50)clearInterval(iv);},200);})();</script>"
                    ))

                    map_out_c = st_folium(
                        m_c, key=f"insp_c_map_{ver_c}",
                        width=None, height=310,
                        returned_objects=["last_object_clicked"],
                    )

                    # Map click → navigate to nearest pad in filtered set
                    if map_out_c and map_out_c.get("last_object_clicked"):
                        click = map_out_c["last_object_clicked"]
                        if isinstance(click, dict) and "lat" in click:
                            clat = float(click["lat"])
                            clon = float(click.get("lng", lon_c))
                            dists = (df_c["lat"] - clat) ** 2 + (df_c["lon"] - clon) ** 2
                            new_idx = int(dists.to_numpy().argmin())
                            if new_idx != idx_c:
                                st.session_state.update({
                                    "_insp_c_idx":   new_idx,
                                    "_insp_c_chip":  None,
                                    "_insp_c_result": None,
                                })
                                st.rerun()

                    # ── Verify location ───────────────────────────────────────
                    with st.expander("Verify location — query what's here"):
                        _q_key = (osm_id_c, round(lat_c, 5), round(lon_c, 5))
                        if st.session_state.get("_insp_c_q_key") != _q_key:
                            st.session_state["_insp_c_q_key"]    = _q_key
                            st.session_state["_insp_c_q_result"] = None

                        if st.button("Query Nominatim + Overpass", key="_insp_c_query_btn"):
                            _q_res = {}
                            _hdr = {"User-Agent": "SkyRoute/1.0 helipad-inspector"}

                            # Nominatim reverse geocode
                            try:
                                _r = requests.get(
                                    "https://nominatim.openstreetmap.org/reverse",
                                    params={"lat": lat_c, "lon": lon_c,
                                            "format": "json", "zoom": 18,
                                            "addressdetails": 1},
                                    headers=_hdr, timeout=10,
                                )
                                if _r.ok:
                                    _nom = _r.json()
                                    _q_res["display_name"] = _nom.get("display_name", "")
                                    _q_res["address"] = _nom.get("address", {})
                                    _q_res["osm_type"] = _nom.get("osm_type", "")
                                    _q_res["osm_id_nom"] = _nom.get("osm_id", "")
                                    _q_res["category"] = _nom.get("category", "")
                                    _q_res["type_nom"] = _nom.get("type", "")
                            except Exception:
                                pass

                            # Overpass: aeroway/helipad/hospital within 300m
                            try:
                                _op_q = (
                                    f"[out:json][timeout:10];"
                                    f"("
                                    f'node["aeroway"](around:300,{lat_c},{lon_c});'
                                    f'node["amenity"="hospital"](around:500,{lat_c},{lon_c});'
                                    f'node["amenity"="clinic"](around:300,{lat_c},{lon_c});'
                                    f'way["aeroway"](around:300,{lat_c},{lon_c});'
                                    f'way["building"="hospital"](around:300,{lat_c},{lon_c});'
                                    f");out tags;"
                                )
                                _r2 = requests.post(
                                    "https://overpass-api.de/api/interpreter",
                                    data={"data": _op_q},
                                    headers=_hdr, timeout=15,
                                )
                                if _r2.ok:
                                    _q_res["nearby"] = _r2.json().get("elements", [])
                            except Exception:
                                _q_res["nearby"] = []

                            st.session_state["_insp_c_q_result"] = _q_res

                        _q = st.session_state.get("_insp_c_q_result")
                        if _q:
                            if _q.get("display_name"):
                                st.markdown(
                                    f"**Address:** {_q['display_name']}<br>"
                                    f"**Category:** {_q.get('category','—')} / {_q.get('type_nom','—')}",
                                    unsafe_allow_html=True,
                                )
                                _addr = _q.get("address", {})
                                _interesting = {k: v for k, v in _addr.items()
                                                if k not in ("country", "country_code",
                                                             "postcode", "state_district")}
                                if _interesting:
                                    st.json(_interesting)

                            _nearby = _q.get("nearby", [])
                            if _nearby:
                                st.markdown(f"**Nearby OSM features ({len(_nearby)}):**")
                                for _el in _nearby[:12]:
                                    _tags = _el.get("tags", {})
                                    _label = (
                                        _tags.get("name") or
                                        _tags.get("aeroway") or
                                        _tags.get("amenity") or
                                        _tags.get("building") or
                                        f"[{_el.get('type','?')} {_el.get('id','')}]"
                                    )
                                    _detail = ", ".join(
                                        f"{k}={v}" for k, v in _tags.items()
                                        if k not in ("name",) and len(k) < 20
                                    )[:80]
                                    st.markdown(f"- **{_label}** — {_detail}")
                            elif _q:
                                st.info("No aeroway / hospital features found within 500 m.")

                        st.markdown(
                            f"[Open in OpenStreetMap]({_osm_url})  ·  "
                            f"[Overpass Turbo](https://overpass-turbo.eu/?Q="
                            f"node%28around%3A200%2C{lat_c}%2C{lon_c}%29%5B%22aeroway%22%5D%3B"
                            f"out%3B)",
                            unsafe_allow_html=False,
                        )


with tab_inspector:
    st.markdown("### 🔍 HIE Inspector")
    _inspector_content()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 · Results — training plots and 3-way comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    st.markdown("### 📈 Training Results & Model Comparison")
    st.caption(
        "Visual detection plots are generated by `python scripts/train_yolo.py --skip-train`. "
        "XGBoost structured baseline is trained by `python scripts/train_xgboost.py`."
    )

    _PLOTS_DIR  = YOLO_MODEL_PATH.parent / "plots"
    _PRELIM_DIR = YOLO_MODEL_PATH.parent / "plots_preliminary"
    _XGB_DIR    = YOLO_MODEL_PATH.parent / "xgboost"

    _CMP_DIR    = YOLO_MODEL_PATH.parent / "plots_comparison"

    _res_tabs = st.tabs(["📊 XGBoost Structured Baseline", "🤖 YOLO Models Comparison"])

    # ══ XGBoost tab (now first) ═════════════════════════════════════════════════
    with _res_tabs[0]:
        st.markdown(
            "**XGBoost trained on ADIP structured features only — no imagery.**  \n"
            "Label: `gt` — human-annotated visual presence from the 747 test chips.  \n"
            "Answers: *how much helipad presence is predictable from structured data alone, without satellite imagery?*"
        )

        _xgb_metrics_path = _XGB_DIR / "metrics.json"
        _xgb_fi_plot      = _XGB_DIR / "feature_importance.png"
        _xgb_cmp_plot     = _XGB_DIR / "comparison.png"

        if not _xgb_metrics_path.exists():
            st.info(
                "XGBoost model not trained yet.  \n"
                "Run:  `python scripts/train_xgboost.py`"
            )
        else:
            import json as _json
            _xgb_m = _json.loads(_xgb_metrics_path.read_text(encoding="utf-8"))

            # Metrics summary
            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            _xgb_f1  = _xgb_m.get("f1", 0)
            _maj_f1  = _xgb_m.get("majority_f1", 0)
            _mc1.metric("Precision", f"{_xgb_m.get('precision', 0):.3f}")
            _mc2.metric("Recall",    f"{_xgb_m.get('recall', 0):.3f}")
            _mc3.metric("XGBoost F1", f"{_xgb_f1:.3f}",
                        delta=f"{_xgb_f1 - _maj_f1:+.3f} vs majority baseline",
                        delta_color="normal")
            _mc4.metric("Majority-class baseline F1", f"{_maj_f1:.3f}",
                        help="F1 of a classifier that predicts every helipad as visually present. "
                             "High because ~58 % of the test set is gt=1. "
                             "XGBoost must beat this to show real discriminative power.")
            st.caption(
                "**Interpretation:** XGBoost F1 ≈ majority baseline because ~58 % of helipads "
                "are visually present (class imbalance inflates F1 for high-recall predictors). "
                "More informative: Precision 0.74 > base rate 0.58 — the model IS discriminating. "
                "The 0.16 gap vs YOLO F1 (0.89) represents helipads visually confirmable by imagery "
                "but not predictable from registry data alone — exactly why the HIE visual pipeline exists."
            )

            st.divider()
            _xp1, _xp2 = st.columns(2)
            if _xgb_fi_plot.exists():
                _xp1.markdown("**Feature Importance (gain)**")
                _safe_image(_xp1, _xgb_fi_plot)
            if _xgb_cmp_plot.exists():
                _xp2.markdown("**XGBoost vs YOLOv8s**")
                _safe_image(_xp2, _xgb_cmp_plot)

            # ── position_age_days distribution ────────────────────────────────
            st.divider()
            st.markdown("**Position Age Distribution — days since ARP coordinates were last verified**")
            try:
                _faa_res = pd.read_csv(DATA_DIR / "faa_adip_enriched.csv")
                _ref_ts  = pd.Timestamp("2026-06-01")
                _pos_dt  = pd.to_datetime(_faa_res["position_date"], errors="coerce")
                _pos_age = (_ref_ts - _pos_dt).dt.days.clip(lower=0).dropna()
                _fig_pos = px.histogram(
                    _pos_age,
                    nbins=40,
                    labels={"value": "Days since position verified", "count": "Helipads"},
                    title="How stale are helipad coordinates? (NE US FAA, 747 records)",
                    color_discrete_sequence=["#2563eb"],
                    template="plotly_dark",
                )
                _fig_pos.add_vline(
                    x=365, line_dash="dash", line_color="#f59e0b",
                    annotation_text="1 yr", annotation_position="top right",
                )
                _fig_pos.add_vline(
                    x=1095, line_dash="dash", line_color="#dc2626",
                    annotation_text="3 yr", annotation_position="top right",
                )
                _fig_pos.update_layout(
                    showlegend=False,
                    xaxis_title="Days since position last verified",
                    yaxis_title="Number of helipads",
                    margin=dict(t=50, b=40),
                )
                _pct_stale = (_pos_age > 365).mean() * 100
                _pct_very  = (_pos_age > 1095).mean() * 100
                _pa1, _pa2, _pa3 = st.columns(3)
                _pa1.metric("Median position age", f"{int(_pos_age.median())} days")
                _pa2.metric("> 1 year old", f"{_pct_stale:.0f}% of helipads")
                _pa3.metric("> 3 years old", f"{_pct_very:.0f}% of helipads")
                try:
                    st.plotly_chart(_fig_pos, use_container_width=True)
                except TypeError:
                    st.plotly_chart(_fig_pos, use_column_width=True)
            except Exception as _e:
                st.warning(f"Could not build position age chart: {_e}")

            # ── has_wind feature spotlight ─────────────────────────────────────
            st.divider()
            st.markdown("#### Top feature spotlight — `has_wind`")
            _ws_col, _ws_txt = st.columns([1, 2], gap="large")
            with _ws_col:
                st.image(
                    "https://img.youtube.com/vi/ae1wdsoXSfQ/0.jpg",
                    caption="Helipad windsock — click link below to watch",
                    use_column_width="always",
                )
                st.markdown("[▶ Watch: Helipad Wind Indicator (YouTube)](https://youtu.be/ae1wdsoXSfQ?si=As7b0invRNb0BC-o)")
            with _ws_txt:
                st.markdown(
                    "<p style='font-size:1.15rem; line-height:1.7'>"
                    "A helipad wind indicator, commonly known as a <b>windsock</b> or wind cone, "
                    "is a mandatory visual aid used to display real-time wind direction and velocity "
                    "to helicopter pilots during approach, hover, and takeoff. Because helicopters "
                    "must ideally land and take off facing directly into the wind to maintain "
                    "stability, a functional wind indicator is critical for safety."
                    "</p>"
                    "<p style='font-size:1.15rem; line-height:1.7'>"
                    "In the XGBoost model, <code>has_wind</code> is the <b>strongest single predictor</b> of "
                    "visual helipad presence (gain = 5.2×). Helipads with registered wind "
                    "equipment are actively operated and maintained — making them far more likely "
                    "to have visible painted markings than unequipped pads."
                    "</p>",
                    unsafe_allow_html=True,
                )

    # ══ YOLO Models Comparison tab (now second) ════════════════════════════════
    with _res_tabs[1]:
        import json as _json

        _MODEL_COLORS = {
            "YOLOv8s":   "#2563eb",
            "YOLOv11s":  "#16a34a",
            "YOLOv11m":  "#9333ea",
            "RT-DETR-L": "#dc2626",
        }
        _CMP_METRICS_PATH  = _CMP_DIR / "comparison_metrics.json"
        _CMP_BAR_PATH      = _CMP_DIR / "comparison_bar.png"
        _CMP_PR_PATH       = _CMP_DIR / "curve_pr.png"
        _CMP_PREC_PATH     = _CMP_DIR / "curve_precision_conf.png"
        _CMP_REC_PATH      = _CMP_DIR / "curve_recall_conf.png"

        # ── Run instructions ──────────────────────────────────────────────────
        if not _CMP_METRICS_PATH.exists():
            st.info(
                "Full 4-model comparison not yet generated.  \n"
                "Run once (evaluates all models on 747 test chips, ~10 min):  \n"
                "```\npython scripts/compare_models.py\n```"
            )
        else:
            _cmp_m = _json.loads(_CMP_METRICS_PATH.read_text(encoding="utf-8"))
            st.caption(
                "All models evaluated on the same 747 held-out NE US test chips "
                "at their individually optimal confidence threshold."
            )

            # ── Metric tiles ──────────────────────────────────────────────────
            _model_names = list(_cmp_m.keys())
            _tile_cols   = st.columns(len(_model_names))
            for _ci, _mn in enumerate(_model_names):
                _mm = _cmp_m[_mn]
                _tile_cols[_ci].metric(
                    _mn,
                    f"F1 {_mm['f1']:.3f}",
                    f"P {_mm['precision']:.2f}  R {_mm['recall']:.2f}",
                )

            st.divider()

            # ── Radar + PR curve side by side ─────────────────────────────────
            _CMP_RADAR_PATH = _CMP_DIR / "comparison_radar.png"
            _radar_c1, _radar_c2 = st.columns([1, 1], gap="medium")
            with _radar_c1:
                st.markdown("**Radar — Precision / Recall / F1 / Accuracy**")
                if _CMP_RADAR_PATH.exists():
                    _safe_image(st, _CMP_RADAR_PATH)
                else:
                    st.caption("Run `python scripts/compare_models.py --plots-only`")
            with _radar_c2:
                st.markdown("**Precision–Recall curve**")
                if _CMP_PR_PATH.exists():
                    _safe_image(st, _CMP_PR_PATH)
                else:
                    st.caption("Run `python scripts/compare_models.py --plots-only`")

            st.markdown(
                "> **YOLO11m** is the production model — highest precision (0.931) and fewest false "
                "positives (FP = 27), directing the routing engine only to real, confirmable helipads. "
                "Choose it whenever a wrong detection creates a safety or trust liability.\n>\n"
                "> **YOLO11s** is the discovery model — highest recall (0.866), finding 8 more real "
                "helipads than YOLO11m (TP = 375 vs 367) at the cost of 11 extra false positives. "
                "Use it for initial registry sweeps where coverage matters more than confidence.\n>\n"
                "> **YOLOv8s** and **RT-DETR-L** both fall behind the YOLO11 family. RT-DETR-L confirms "
                "that transformer-based detectors can learn the NAIP aerial domain, but offers no "
                "practical advantage over YOLO11 at this dataset size."
            )

        # ── Confidence-sweep plots ────────────────────────────────────────────
        st.divider()
        st.markdown("**Confidence Threshold Analysis — all models**")
        _have_curves = _CMP_PREC_PATH.exists() and _CMP_REC_PATH.exists()
        if _have_curves:
            _cc2, _cc3 = st.columns(2, gap="small")
            _cc2.markdown("**Precision vs Confidence**")
            _safe_image(_cc2, _CMP_PREC_PATH)
            _cc3.markdown("**Recall vs Confidence**")
            _safe_image(_cc3, _CMP_REC_PATH)
            st.caption(
                "Diamond markers (◆) show each model's individually optimal confidence threshold. "
                "Dashed line = median optimal threshold across models. "
                "Note: raw confidence scores are NOT comparable across architectures — each model "
                "uses a different internal scale. Calibrate thresholds independently per model."
            )
        else:
            st.info("Run `python scripts/compare_models.py` to generate confidence-sweep plots.")

        # ── Individual model plots (consistent style, all from curve data) ──────
        st.divider()
        st.markdown("**Individual Model Plots**")

        def _ind_safe(n): return n.lower().replace("-", "").replace(" ", "_")

        _IND_CURVE_PLOTS = [
            ("pr.png",             "Precision–Recall"),
            ("f1_conf.png",        "F1 vs Confidence"),
            ("precision_conf.png", "Precision vs Confidence"),
            ("recall_conf.png",    "Recall vs Confidence"),
            ("confusion.png",      "Confusion Matrix"),
        ]
        for _mn in (_cmp_m or {}).keys():
            _ind_dir = _CMP_DIR / f"individual_{_ind_safe(_mn)}"
            _available = [(f, t) for f, t in _IND_CURVE_PLOTS if (_ind_dir / f).exists()]
            if not _available:
                continue
            with st.expander(f"{_mn}", expanded=False):
                for _pi in range(0, len(_available), 2):
                    _ec = st.columns(2)
                    for _ej, (_ef, _et) in enumerate(_available[_pi:_pi+2]):
                        _ec[_ej].markdown(f"**{_et}**")
                        _safe_image(_ec[_ej], _ind_dir / _ef)


def _mly_viewer_html(lat: float, lon: float, height: int = 260,
                     image_id: str = "", thumb_url: str = "") -> str:
    """Satellite map + Mapillary street-view thumbnail for a booking leg location.

    Satellite tab (default): ESRI World Imagery via Leaflet.
    Street View tab: shows a pre-fetched Mapillary JPEG thumbnail (fetched server-side,
    so no CORS / WebGL / sandbox issues).  Falls back to a Mapillary app link if
    thumb_url is empty.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        height: Total widget height in pixels.
        image_id: Pre-fetched Mapillary image ID (used only for the fallback deep link).
        thumb_url: Direct CDN JPEG URL from get_mapillary_thumb_url().
    """
    map_h   = height - 48
    mly_url = f"https://www.mapillary.com/app/?lat={lat:.6f}&lng={lon:.6f}&z=17"
    if image_id:
        mly_url = f"https://www.mapillary.com/app/?pKey={image_id}"
    gmap    = f"https://maps.google.com/?q={lat:.6f},{lon:.6f}"

    if thumb_url:
        sv_content = f"""<div id="mly-pane" style="width:100%;height:{map_h}px;display:none;position:relative;background:#111">
  <img src="{thumb_url}" alt="Street view"
       style="width:100%;height:100%;object-fit:cover;display:block">
  <a href="{mly_url}" target="_blank"
     style="position:absolute;bottom:4px;right:6px;font-size:10px;color:#fff;
            background:rgba(0,0,0,.55);padding:2px 6px;border-radius:3px;text-decoration:none">
    Open in Mapillary ↗
  </a>
</div>"""
    else:
        sv_content = f"""<div id="mly-pane" style="width:100%;height:{map_h}px;display:none;
     background:#111;display:none;align-items:center;justify-content:center;font-size:12px;color:#888">
  No nearby street imagery found.&nbsp;
  <a href="{mly_url}" target="_blank" style="color:#4fa3e0">Browse Mapillary ↗</a>
</div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9/dist/leaflet.css">
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#111;overflow:hidden;font-family:sans-serif;color:#ccc}}
  .tabs{{display:flex;height:28px;background:#1e1e1e;border-bottom:1px solid #333}}
  .tab{{flex:1;text-align:center;line-height:28px;font-size:12px;cursor:pointer;color:#888}}
  .tab.active{{color:#4fa3e0;border-bottom:2px solid #4fa3e0}}
  #map-pane{{width:100%;height:{map_h}px;display:none}}
  .bar{{height:20px;line-height:20px;padding:0 8px;font-size:10px;color:#555;background:#111}}
  .bar a{{color:#4fa3e0;text-decoration:none;margin-right:12px}}
</style></head><body>
<div class="tabs">
  <div class="tab active" id="tab-sat" onclick="showTab('sat')">🛰 Satellite</div>
  <div class="tab" id="tab-sv" onclick="showTab('sv')">📸 Street View</div>
</div>
<div id="map-pane"><div id="map" style="width:100%;height:{map_h}px"></div></div>
{sv_content}
<div class="bar">
  <a href="{mly_url}" target="_blank">Mapillary ↗</a>
  <a href="{gmap}" target="_blank">Google Maps ↗</a>
</div>
<script src="https://unpkg.com/leaflet@1.9/dist/leaflet.js"></script>
<script>
var LAT = {lat}, LON = {lon};
var leafMap = null;

function showTab(t) {{
  document.getElementById('tab-sat').className = 'tab' + (t==='sat' ? ' active' : '');
  document.getElementById('tab-sv').className  = 'tab' + (t==='sv'  ? ' active' : '');
  document.getElementById('map-pane').style.display   = t==='sat' ? 'block' : 'none';
  document.getElementById('mly-pane').style.display   = t==='sv'  ? 'flex'  : 'none';
  if (t === 'sat' && leafMap) {{ leafMap.invalidateSize(); }}
}}

// Init Leaflet while map-pane is visible (real dimensions)
document.getElementById('map-pane').style.display = 'block';
leafMap = L.map('map', {{zoomControl:true, attributionControl:false}}).setView([LAT, LON], 18);
L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{maxZoom:20}}
).addTo(leafMap);
var dot = L.divIcon({{
  html:'<div style="width:14px;height:14px;background:#e74c3c;border:2px solid #fff;border-radius:50%;box-shadow:0 0 6px rgba(0,0,0,.7)"></div>',
  className:'', iconSize:[14,14], iconAnchor:[7,7]
}});
L.marker([LAT, LON], {{icon:dot}}).addTo(leafMap);
</script></body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 · Route Assistant — LLM-powered natural language routing (M4)
# ══════════════════════════════════════════════════════════════════════════════

_MIA_IMG   = ASSETS_DIR / "mia.png"
_MIA_AUDIO = ASSETS_DIR / "mia_intro.mp3"


def _build_mia_card(autoplay: bool = False) -> str:
    """Return a self-contained HTML card for the Mia intro panel.

    Embeds image and audio as base64 data URIs so the component is
    fully self-contained inside Streamlit's sandboxed iframe.
    Clicking the image replays the intro audio.
    """
    img_src = ""
    if _MIA_IMG.exists():
        img_src = "data:image/png;base64," + base64.b64encode(_MIA_IMG.read_bytes()).decode()

    audio_src = ""
    if _MIA_AUDIO.exists():
        audio_src = "data:audio/mpeg;base64," + base64.b64encode(_MIA_AUDIO.read_bytes()).decode()

    autoplay_js = (
        'window.addEventListener("load",function(){'
        'setTimeout(function(){var a=document.getElementById("mia-a");if(a){a.play().catch(function(){});}},250);'
        '});'
    ) if (autoplay and audio_src) else ""

    img_html = (
        f'<img src="{img_src}" alt="Mia">'
        if img_src
        else '<div style="width:170px;height:170px;background:#0d2040;border-radius:12px;'
             'display:flex;align-items:center;justify-content:center;color:#3a6fa0;font-size:36px">🤖</div>'
    )
    play_btn  = '<div class="play-btn" title="Replay introduction">▶</div>' if audio_src else ""
    audio_tag = f'<audio id="mia-a" src="{audio_src}"></audio>' if audio_src else ""

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
.card{{display:flex;align-items:flex-start;gap:18px;
  background:linear-gradient(135deg,#060c18 0%,#0b1628 60%,#0a1220 100%);
  border:1px solid rgba(0,140,220,.35);border-radius:14px;padding:18px;
  position:relative;overflow:hidden}}
.card::before{{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent 0%,#00aaff 50%,transparent 100%)}}
.img-wrap{{position:relative;flex-shrink:0;cursor:pointer;user-select:none}}
.img-wrap img{{width:170px;height:170px;object-fit:cover;border-radius:12px;
  border:1.5px solid rgba(0,150,255,.5);box-shadow:0 0 18px rgba(0,150,255,.25);
  display:block;transition:box-shadow .25s}}
.img-wrap:hover img{{box-shadow:0 0 30px rgba(0,170,255,.6)}}
.play-btn{{position:absolute;bottom:7px;right:7px;width:26px;height:26px;
  background:rgba(0,150,255,.85);border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:11px;color:#fff;transition:transform .15s,background .15s}}
.img-wrap:hover .play-btn{{transform:scale(1.15);background:rgba(0,180,255,1)}}
.info{{flex:1;padding-top:2px}}
.name{{font-size:24px;font-weight:700;color:#fff;letter-spacing:.3px;margin-bottom:3px}}
.title{{font-size:13px;color:#00b4ff;text-transform:uppercase;letter-spacing:1.8px;margin-bottom:10px}}
.desc{{font-size:14.5px;color:#8badcc;line-height:1.65}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin-top:11px}}
.tag{{background:rgba(0,140,220,.12);border:1px solid rgba(0,140,220,.3);
  color:#5bc8f5;font-size:12px;padding:4px 11px;border-radius:20px}}
</style></head><body>
<div class="card">
  <div class="img-wrap" onclick="playAudio()" title="Click to replay introduction">
    {img_html}{play_btn}
  </div>
  <div class="info">
    <div class="name">Mia</div>
    <div class="title">Mobility Intelligence Assistant</div>
    <div class="desc">I plan your door-to-door air mobility routes across New York metro —
      combining helicopter, eVTOL, and ground transport into a single booking.
      Tell me your destination.</div>
    <div class="tags">
      <span class="tag">✈ Aerial routing</span>
      <span class="tag">🚗 Ground legs</span>
      <span class="tag">📍 POI search</span>
      <span class="tag">🌤 Live weather</span>
      <span class="tag">📋 Helipad status</span>
    </div>
  </div>
  {audio_tag}
</div>
<script>
function playAudio(){{var a=document.getElementById('mia-a');if(a){{a.currentTime=0;a.play().catch(function(){{}});}}}}
{autoplay_js}
</script>
</body></html>"""


def _metar_badge(metar: dict | None) -> str:
    """Return an HTML METAR badge string for inline rendering.

    Returns empty string when no METAR is available.
    """
    if not metar:
        return ""
    cat   = metar.get("flight_category") or ""
    color = {"VFR": "#16a34a", "MVFR": "#2563eb", "IFR": "#dc2626", "LIFR": "#7e22ce"}.get(cat, "#6b7280")
    parts = []
    if metar.get("wind_kt") is not None:
        gust = metar.get("wind_gust_kt")
        parts.append(f"💨 {metar['wind_kt']}{'G' + str(gust) if gust else ''}kt")
    if metar.get("visibility_sm") is not None:
        parts.append(f"👁 {metar['visibility_sm']}sm")
    if metar.get("ceiling_ft") is not None:
        parts.append(f"☁️ {metar['ceiling_ft']:,}ft")
    details = " · ".join(parts) if parts else "no obs"
    badge = (f'<span style="background:{color};color:#fff;padding:2px 10px;'
             f'border-radius:4px;font-weight:700;font-size:0.82em">{cat or "N/A"}</span>')
    return f"{badge} {details}"


@_fragment
def _route_assistant_content() -> None:
    """Render the Route Assistant tab as an isolated fragment.

    Isolating this tab prevents chat messages and spinner reruns from
    rebuilding the EDA maps, hotspot maps, and routing simulator HTML.
    """
    _first_visit = "_mia_intro_played" not in st.session_state
    if _first_visit:
        st.session_state["_mia_intro_played"] = True
    components.html(_build_mia_card(autoplay=_first_visit), height=225, scrolling=False)

    # ── Helipad pool + FAA ADIP data ──────────────────────────────────────────
    if "_agent_helipads" not in st.session_state:
        faa_raw, _ = load_data()
        _agent_faa = faa_raw.dropna(subset=["lat", "lon"])
        _pool: list[dict] = [
            {
                "lat":   float(r["lat"]),
                "lon":   float(r["lon"]),
                "name":  str(r.get("NAME", "") or "").strip() or None,
                "ident": str(r.get("IDENT", "") or "").strip() or None,
            }
            for r in _agent_faa.to_dict("records")
        ]
        # Append visually-confirmed OSM-only helipads (hie_visual_detected=True)
        _osm_val = _load_osm_validated()
        if _osm_val is not None and not _osm_val.empty:
            _confirmed = _osm_val[_osm_val["hie_visual_detected"]].dropna(subset=["lat", "lon"])
            for r in _confirmed.to_dict("records"):
                _pool.append({
                    "lat":   float(r["lat"]),
                    "lon":   float(r["lon"]),
                    "name":  str(r.get("name", "") or "").strip() or None,
                    "ident": None,  # OSM-only pads have no FAA IDENT
                })
        st.session_state["_agent_helipads"] = _pool
    _agent_helipads: list[dict] = st.session_state["_agent_helipads"]
    # Pass the full enriched DataFrame for ADIP lookups during booking
    # load_data() returns instantly from @st.cache_data — no disk read after first load
    _adip_df, _ = load_data()

    if not os.getenv("GROQ_API_KEY"):
        st.warning(
            "GROQ_API_KEY not set — responses will use template text. "
            "Local: add it to `.env` and restart. "
            "Streamlit Cloud: add it under **Settings → Secrets**.",
            icon="⚠️",
        )

    # ── Session state ─────────────────────────────────────────────────────────
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []
    # _agent_llm_history: [{role, content}] passed to extract_nav_params for multi-turn context.
    # assistant turns contain the extracted params JSON (not the formatted narrative) so the LLM
    # can resolve references like "arrive earlier", "same destination", "change origin to...".
    if "_agent_llm_history" not in st.session_state:
        st.session_state["_agent_llm_history"] = []
    if "_agent_booking_legs" not in st.session_state:
        st.session_state["_agent_booking_legs"] = []
    # _agent_last_route: persists the most recently computed route for booking
    # and for auto-triggering the routing simulator in the EDA tab

    # Render existing conversation
    for _msg in st.session_state["agent_messages"]:
        with st.chat_message(_msg["role"]):
            st.markdown(_msg["content"])
            if _msg.get("route_map_html"):
                components.html(_msg["route_map_html"], height=340)

    # ── Booking leg cards — rendered from session state so they survive reruns ──
    for _bl in st.session_state["_agent_booking_legs"]:
        if _bl["mode"] == "rideshare":
            rs = _bl["rideshare"]
            st.markdown(
                f"### 🚗 Leg {_bl['leg_index']+1} — Ground Transport"
                f"  \n**{_bl['from']} → {_bl['to']}**"
            )
            _gc1, _gc2, _gc3 = st.columns(3)
            _gc1.metric("Estimated fare", rs["fare_range"])
            _gc2.metric("Duration", f"{rs['duration_min']} min")
            _gc3.metric("Distance", f"{rs['dist_km']} km")
            st.markdown(
                f"**Booking ref:** `{rs['booking_ref']}`  \n"
                f"**Services:** {', '.join(rs['vehicles'])}  \n"
                f"[📱 Open Uber]({rs['uber_deeplink']}) &nbsp;·&nbsp; "
                f"[🚗 Waymo One]({rs['waymo_url']})"
            )
            st.caption("_Simulated — actual pricing and availability depend on demand._")
            with st.expander(f"📍 Pickup — {_bl['from']}", expanded=True):
                components.html(
                    _mly_viewer_html(_bl["pickup_lat"], _bl["pickup_lon"],
                                     image_id=_bl.get("pickup_mly_id", ""),
                                     thumb_url=_bl.get("pickup_mly_thumb", "")),
                    height=260,
                )
            with st.expander(f"📍 Dropoff — {_bl['to']}", expanded=True):
                components.html(
                    _mly_viewer_html(_bl["dropoff_lat"], _bl["dropoff_lon"],
                                     image_id=_bl.get("dropoff_mly_id", ""),
                                     thumb_url=_bl.get("dropoff_mly_thumb", "")),
                    height=260,
                )

        elif _bl["mode"] == "walk":
            st.markdown(
                f"### 🚶 Leg {_bl['leg_index']+1} — Walk"
                f"  \n**{_bl['from']} → {_bl['to']}**"
            )
            _wc1, _wc2 = st.columns(2)
            _wc1.metric("Distance", f"{_bl['dist_km']} km")
            _wc2.metric("Walk time", f"{_bl['duration_min']} min")
            st.info(
                f"Only {_bl['dist_km']} km — no transport needed. "
                f"Approximately {_bl['duration_min']} min on foot.",
                icon="🚶",
            )
            with st.expander("📍 Street view at start", expanded=True):
                components.html(
                    _mly_viewer_html(_bl["pickup_lat"], _bl["pickup_lon"],
                                     height=240,
                                     image_id=_bl.get("pickup_mly_id", ""),
                                     thumb_url=_bl.get("pickup_mly_thumb", "")),
                    height=240,
                )

        elif _bl["mode"] == "helicopter":
            dep = _bl["departure_helipad"]
            arr = _bl["arrival_helipad"]
            st.markdown(
                f"### 🚁 Leg {_bl['leg_index']+1} — Helicopter Flight"
                f"  \n**{dep['name']} → {arr['name']}**  "
                f"({_bl['dist_km']} km · {_bl['duration_min']} min)"
            )
            with st.expander(f"📋 Departure: {dep['name']} ({dep['ident'] or 'OSM'})", expanded=True):
                _dep_badge = _metar_badge(_bl.get("metar_dep"))
                if _dep_badge:
                    st.markdown(_dep_badge, unsafe_allow_html=True)
                _hc1, _hc2 = st.columns(2)
                _hc1.markdown(
                    f"**Status:** {dep.get('status','—')}  \n"
                    f"**City:** {dep.get('servcity','—')}  \n"
                    f"**Ownership:** {dep.get('ownership','—')}  \n"
                    f"**Private use:** {'Yes' if dep.get('private_use') else 'No'}  \n"
                    + (f"**Address:** {dep['address']}  \n" if dep.get('address') else "")
                    + f"**Coordinates:** `{dep['lat']:.5f}, {dep['lon']:.5f}`"
                )
                _hc2.markdown(f"**Coordination note:**  \n{dep.get('contact_notes','—')}")
                if dep.get("adip_url"):
                    st.markdown(f"[📋 Open ADIP record]({dep['adip_url']})")
                st.markdown(f"[📍 Google Maps ↗]({dep.get('gmaps_url','')})")
                components.html(
                    _mly_viewer_html(dep["lat"], dep["lon"],
                                     image_id=dep.get("mly_image_id", ""),
                                     thumb_url=dep.get("mly_thumb_url", "")),
                    height=260,
                )
            with st.expander(f"📋 Arrival: {arr['name']} ({arr['ident'] or 'OSM'})", expanded=True):
                _arr_badge = _metar_badge(_bl.get("metar_arr"))
                if _arr_badge:
                    st.markdown(_arr_badge, unsafe_allow_html=True)
                _ac1, _ac2 = st.columns(2)
                _ac1.markdown(
                    f"**Status:** {arr.get('status','—')}  \n"
                    f"**City:** {arr.get('servcity','—')}  \n"
                    f"**Ownership:** {arr.get('ownership','—')}  \n"
                    f"**Private use:** {'Yes' if arr.get('private_use') else 'No'}  \n"
                    + (f"**Address:** {arr['address']}  \n" if arr.get('address') else "")
                    + f"**Coordinates:** `{arr['lat']:.5f}, {arr['lon']:.5f}`"
                )
                _ac2.markdown(f"**Coordination note:**  \n{arr.get('contact_notes','—')}")
                if arr.get("adip_url"):
                    st.markdown(f"[📋 Open ADIP record]({arr['adip_url']})")
                st.markdown(f"[📍 Google Maps ↗]({arr.get('gmaps_url','')})")
                components.html(
                    _mly_viewer_html(arr["lat"], arr["lon"],
                                     image_id=arr.get("mly_image_id", ""),
                                     thumb_url=arr.get("mly_thumb_url", "")),
                    height=260,
                )

    # ── Example prompts (first load only) ─────────────────────────────────────
    _pending = st.session_state.pop("_agent_pending_input", None)
    if not st.session_state["agent_messages"] and not _pending:
        st.markdown("**Try one of these:**")
        _ex_cols = st.columns(3)
        _examples = [
            "Fastest way from Midtown to Greenwich CT for a 16:00 meeting",
            "Good restaurant near the 30th St Heliport",
            "What's the weather in Greenwich this afternoon?",
        ]
        for _ei, (_ec, _ex) in enumerate(zip(_ex_cols, _examples)):
            if _ec.button(_ex, key=f"ex_{_ei}"):
                st.session_state["_agent_pending_input"] = _ex
                st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    _user_input = (
        st.chat_input("Route, nearby places, weather… ask anything about your trip") or _pending
    )
    if _user_input:
        st.session_state["agent_messages"].append({"role": "user", "content": _user_input})
        # Clear persisted booking leg cards so stale details don't show during new request
        st.session_state["_agent_booking_legs"] = []
        with st.chat_message("user"):
            st.markdown(_user_input)

        _route_map_html = ""
        with st.chat_message("assistant"):
            _TOOL_ICONS = {
                "geocode": "📍",
                "search_nearby_places": "🔍",
                "get_weather": "🌤️",
                "compute_route": "🚁",
                "confirm_booking": "📋",
            }

            def _agent_step_label(name: str, args: dict) -> str:
                if name == "search_nearby_places":
                    return f"Searching for **{args.get('category','?')}** near _{args.get('location','?')}_"
                if name == "get_weather":
                    return f"Getting weather for _{args.get('location','?')}_"
                if name == "compute_route":
                    return f"Computing route _{args.get('origin','?')}_ → _{args.get('destination','?')}_"
                if name == "confirm_booking":
                    return f"Booking _{args.get('origin','?')}_ → _{args.get('destination','?')}_"
                if name == "geocode":
                    return f"Locating _{args.get('address','?')}_"
                return name

            def _agent_result_label(name: str, res: dict) -> str:
                if "error" in res:
                    return f"⚠️ {res['error']}"
                if name == "search_nearby_places":
                    n = len(res.get("results", []))
                    return res.get("note", f"found {n} result{'s' if n != 1 else ''}")
                if name == "get_weather":
                    return f"{res.get('conditions','')} · {res.get('temperature_f','?')}°F"
                if name == "compute_route":
                    return f"{res.get('total_min','?')} min total · saves {res.get('time_saved_min','?')} min vs driving"
                if name == "confirm_booking":
                    return f"confirmed · ref {res.get('confirmation_id','?')}"
                if name == "geocode":
                    lat, lon = res.get("lat", 0), res.get("lon", 0)
                    return f"{res.get('display_name', f'{lat:.4f}, {lon:.4f}')}"
                return "done"

            with st.status("Working…", expanded=True) as _agent_status:
                def _on_agent_step(event: str, data: dict) -> None:
                    if event == "thinking":
                        msg_txt = "🤔 Thinking…" if data.get("iteration", 0) == 0 else "🤔 Processing results…"
                        st.write(msg_txt)
                    elif event == "tool_call":
                        icon = _TOOL_ICONS.get(data["name"], "🛠️")
                        st.write(f"{icon} {_agent_step_label(data['name'], data.get('args', {}))}")
                    elif event == "tool_result":
                        st.write(f"   ↳ {_agent_result_label(data['name'], data.get('result', {}))}")
                    elif event == "booking_step":
                        st.write(data.get("msg", ""))

                _result = run_agent_v2(
                    _user_input,
                    helipads=_agent_helipads,
                    history=st.session_state["_agent_llm_history"],
                    faa_adip_df=_adip_df,
                    status_callback=_on_agent_step,
                )
                _agent_status.update(label="Done", state="complete", expanded=False)

            _reply = _result.get("response") or ""

            if _result.get("error"):
                _err_msg = f"Sorry — {_result['error']}"
                st.warning(_err_msg)
                if not _reply:
                    _reply = _err_msg
            else:
                # TFR warnings — shown before narrative
                if _result.get("tfrs_ignored"):
                    st.error("🚫 TFR OVERRIDE ACTIVE — routing through restricted airspace. For planning only; operator approval required before flight.")
                for _tfr_txt in _result.get("tfr_warnings", []):
                    st.warning(f"⚠️ Active TFR on aerial segment: {_tfr_txt}")

                # Precipitation warnings
                for _pw in _result.get("precip_warnings", []):
                    st.warning(f"🌧️ Precipitation on aerial leg — {_pw}")

                # Always show the model's final narrative
                if _reply:
                    st.markdown(_reply)

                # ── Route display (if compute_route tool was called) ───────
                _route = _result.get("route")
                if _route:
                    st.session_state["_agent_last_route"] = _result
                    st.divider()

                    _rc1, _rc2, _rc3, _rc4 = st.columns(4)
                    _rc1.metric("Total time",  f"{_route['total_min']} min")
                    _rc2.metric("Drive only",  f"{_route['drive_only_min']} min")
                    _rc3.metric("Time saved",  f"{_route['time_saved_min']} min",
                                delta=f"−{_route['time_saved_min']} min")
                    if _route.get("departure_time"):
                        _rc4.metric("Depart by", _route["departure_time"])

                    if _route["legs"]:
                        _leg_rows = []
                        for _lg in _route["legs"]:
                            _icon = {"helicopter": "🚁", "drive": "🚗", "walk": "🚶"}.get(_lg["mode"], "•")
                            _leg_rows.append({
                                "Mode": f"{_icon}  {_lg['mode'].title()}",
                                "From": _lg["from"],
                                "To":   _lg["to"],
                                "km":   _lg["dist_km"],
                                "min":  _lg["duration_min"],
                            })
                        st.dataframe(_leg_rows, use_container_width=True, hide_index=True)

                    _orig  = _result.get("origin", {})
                    _dest  = _result.get("destination", {})
                    _pad_a = _route.get("nearest_pad_origin")
                    _pad_b = _route.get("nearest_pad_dest")
                    if _orig.get("lat") and _dest.get("lat"):
                        _all_lats = [_orig["lat"], _dest["lat"]]
                        _all_lons = [_orig["lon"], _dest["lon"]]
                        if _pad_a: _all_lats.append(_pad_a["lat"]); _all_lons.append(_pad_a["lon"])
                        if _pad_b: _all_lats.append(_pad_b["lat"]); _all_lons.append(_pad_b["lon"])
                        _rm = folium.Map(
                            location=[(min(_all_lats)+max(_all_lats))/2,
                                      (min(_all_lons)+max(_all_lons))/2],
                            zoom_start=11, tiles="CartoDB positron",
                            zoom_control=False,
                        )
                        folium.Marker(
                            [_orig["lat"], _orig["lon"]], tooltip=f"Origin: {_orig.get('text','A')}",
                            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
                        ).add_to(_rm)
                        folium.Marker(
                            [_dest["lat"], _dest["lon"]], tooltip=f"Destination: {_dest.get('text','B')}",
                            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
                        ).add_to(_rm)
                        if _pad_a:
                            folium.Marker([_pad_a["lat"], _pad_a["lon"]],
                                tooltip=f"Takeoff: {_pad_a.get('name') or _pad_a.get('ident','Helipad')}",
                                icon=folium.Icon(color="cadetblue", icon="plane", prefix="fa"),
                            ).add_to(_rm)
                            folium.PolyLine([[_orig["lat"],_orig["lon"]],[_pad_a["lat"],_pad_a["lon"]]],
                                color="#22c55e", weight=3, tooltip="Ground to helipad").add_to(_rm)
                        if _pad_b:
                            folium.Marker([_pad_b["lat"], _pad_b["lon"]],
                                tooltip=f"Landing: {_pad_b.get('name') or _pad_b.get('ident','Helipad')}",
                                icon=folium.Icon(color="green", icon="plane", prefix="fa"),
                            ).add_to(_rm)
                            folium.PolyLine([[_pad_b["lat"],_pad_b["lon"]],[_dest["lat"],_dest["lon"]]],
                                color="#22c55e", weight=3, tooltip="Ground from helipad").add_to(_rm)
                        if _pad_a and _pad_b:
                            folium.PolyLine(
                                [[_pad_a["lat"],_pad_a["lon"]],[_pad_b["lat"],_pad_b["lon"]]],
                                color="#00d4ff", weight=3, dash_array="10 6",
                                tooltip="Helicopter flight",
                            ).add_to(_rm)
                        _rm.fit_bounds([[min(_all_lats)-0.02, min(_all_lons)-0.02],
                                        [max(_all_lats)+0.02, max(_all_lons)+0.02]])
                        from branca.element import Element as _BrancaEl
                        _rm.get_root().html.add_child(_BrancaEl(
                            "<script>(function(){"
                            "var _n=0,_iv=setInterval(function(){"
                            "_n++;"
                            "var el=document.querySelector('.leaflet-container');"
                            "if(el&&window[el.id]&&typeof window[el.id].addLayer==='function'){"
                            "L.control.zoom({position:'topleft'}).addTo(window[el.id]);"
                            "clearInterval(_iv);}"
                            "if(_n>30)clearInterval(_iv);"
                            "},100);})();</script>"
                        ))
                        _route_map_html = _rm.get_root().render()
                        components.html(_route_map_html, height=340, scrolling=False)

                    st.info(
                        "Type **yes** or **book it** to confirm and see helipad coordination "
                        "details + arrange your ground transport. "
                        "The route has also been sent to the **EDA & HIE → Routing Simulator** tab.",
                        icon="ℹ️",
                    )

                    with st.expander("Resolved locations", expanded=False):
                        def _fmt_loc(loc: dict) -> str:
                            parts = []
                            if loc.get("poi_name"):
                                parts.append(f"**{loc['poi_name']}**")
                            if loc.get("address"):
                                parts.append(loc["address"])
                            parts.append(
                                f"`{loc.get('lat',0):.5f}, {loc.get('lon',0):.5f}`"
                            )
                            return "  \n".join(parts)
                        _lc1, _lc2 = st.columns(2)
                        _lc1.markdown(
                            f"**Origin:** {_orig.get('text','—')}  \n"
                            + _fmt_loc(_orig)
                        )
                        _lc2.markdown(
                            f"**Destination:** {_dest.get('text','—')}  \n"
                            + _fmt_loc(_dest)
                        )

                # ── Booking display (if confirm_booking tool was called) ───
                _booking_legs = _result.get("booking_legs")
                if _booking_legs:
                    st.session_state["_agent_booking_legs"] = _booking_legs
                    st.success("**Booking confirmed (simulated)** — step-by-step details below:")

        st.session_state["agent_messages"].append({
            "role": "assistant",
            "content": _reply,
            "route_map_html": _route_map_html,
        })

        # Update multi-turn history — store full tool-call-aware messages from this turn
        if _result.get("_messages"):
            st.session_state["_agent_llm_history"] = _result["_messages"]

        # Rerun only when booking legs were just confirmed — those cards render from
        # session_state at the top of the fragment and need a second pass to appear.
        # For all other responses both messages are already rendered inline above.
        if _result.get("booking_legs"):
            st.rerun()

    # ── Clear conversation ─────────────────────────────────────────────────────
    if st.session_state["agent_messages"]:
        if st.button("Clear conversation", key="agent_clear"):
            st.session_state["agent_messages"] = []
            st.session_state["_agent_llm_history"] = []
            st.session_state["_agent_booking_legs"] = []
            st.session_state.pop("_agent_last_route", None)
            st.rerun()


with tab_agent:
    _route_assistant_content()
