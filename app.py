"""SkyRoute Helipad Intelligence Engine — EDA Dashboard.

Run:
    streamlit run app.py
"""

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

log = logging.getLogger(__name__)

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
tab_problem, tab_lit, tab_market, tab_eda = st.tabs([
    "📍 Problem", "📚 Literature", "🏪 Market", "📊 EDA & HIE",
])


# ── TAB 1 · Problem Understanding ────────────────────────────────────────────

with tab_problem:
    st.markdown("### The SkyRoute Opportunity")
    col_persona, col_journey = st.columns([1, 2])

    with col_persona:
        st.markdown("""
<div style="background:#0d2137;border-radius:12px;padding:20px;border-left:4px solid #29b6f6">
  <div style="text-align:center;margin-bottom:4px"><img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAQDAwMDAgQDAwMEBAQFBgoGBgUFBgwICQcKDgwPDg4MDQ0PERYTDxAVEQ0NExoTFRcYGRkZDxIbHRsYHRYYGRj/2wBDAQQEBAYFBgsGBgsYEA0QGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBj/wAARCADIAMgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDrgOOK5H4l3T2fw11KVZFjym0nOCQeML7muxUc5rzT41SxJ4KTfNtYShlRjhD/AL1cq3NWzxXRlhmsTdXmyK2gPTJOZGbABHc9vrmtHUrpLvzbdnRXGFdUHI9FzjgfzOfasWMNBpVtHFMYlt90zSOo/wBaxxvA9VXhR6n2rQs9kMUKQxGM/eJb5nyeuT3Y9Se1XN2Vwpx5nYsxacLQJGwHnHBKj+EV2Nig8hVQYwK5mPaJB37k+tdLpvzKQWIPpmvLqyvue9ho20RvaWimf5eg7AZNdtpcILh8kDORmuN0iJl3ShCq9s5rr9DuVlz5gORx06iuOTPRjsddbwhkVGGc89MVq28YWMrjJB47YrGtpyVBVsfWtqykDIDtOCcDPalYGXYl2oTtBz+lTwoFb5jjPFQ5XzFQHjriraZ3qygZBzx2ppmbQqjFwMA5PX3q8NykY7DpVXq+CDjkYq+mEwWA7D8aa3IkxxDNgbQAP1pdgY7AvTtipFKrGoOACfvHvUw8oH7yjvitFEyuVGhVGBEfI96zrmEs4Pb37VunZsPII9c1RmRdwVQW56ipkjSEjJADkjHNV5FPlsNvy4yCexq9LBtk4+6OhHeq11lBt2nB6j0rI13PIvi9oMM9tY695alraQF9w6j0/GuV8JPLqvgWzl854b6zMqDHJVWO7j2yn617Lr1lFqWiy2c8YcHBwRken9a8h8NWF3pHivVNP+fyvLRlLNgbdxHT0GQD7HNd+Fndcp5uNp2fMUfG1vFdaXa6xG5+yTSgueCpbGMZ9wGwK77w+wPh+3RidwTByMGvL/EGoMlrFpkBjMDFBGjZKhl4I56ZzjnpXo3hS5MmhRRKzeWmVWOTO9Mdueor0eh4z3NxgM0U40VIFhTg8ivH/jQslxJaJkOBnyoc9W/vHPQCvXx1rzz4raekmkQXaFFdWKu5BOFIxmmnqJni1to6rbRtcSjPmblU9WYDcZGPoM8D3FE0rNJ+5TYijaOOSDzj6nqa27lFvt02B5cK7EVjtwCcljjtxz7cVj3jRpbIUcAt8zE84B/qfT/ClUd9DWkralzTxuTdnLenWuk0Z2N1tK/L0zXOaWy7OVIGOnX9a6XSyI3Vjxk15tXqe1h+h3cMA8j5Pu469a1dH/dS7OAPXFUdKnUWyhgDvBBx2rTjhWKf5ScjHINcdz0kdFZEeeV4OPx963bdZAmcgAnisOwC7Q4LFj157VvWoCphhgZ4waVwZoRsysM4+Vc4NW7IJ5vzjO7kkcYqoIWdsrIVHvzVxEd5UCx5AX+9jmmZvYl2oblgrvt7c9KnhBkk8tmc8YyWqpLMIWAbcMnnj/Cp43jULt3ZK5IFUmZtF0hV2ENuUd6lViYyT0xxzVJ5sPlkkfnAAFToeOFYYHTBzVqRnyl1W3LngiqrgDLehxxSrMcHesmMdcc0jsWAGxsUN3BKxAI8qd469MDg1nXqhz0+orX3AfMPSs27BwTgde9ZyLi9Tnb2N2t3VFLEDBA61wV3BG2pLcW8wWba2wn5T/tx++R0+lehakxitndCQe5xmuJki+2XsrKFCMQ7bRyCp5Pv1zn61vhtGYYzWJ5Nrq/afHX2yeGJbWYBmSJdiMp4bjsQcHPufSvUfD8MUOkrEI5VkDH5y24Ovb6Gs3UfDccEcl9PHveBmZ88KyMew/2hn6EVvabbw2+nIiF9+Odw5PHBz34r107o8CSs7Fk0UMaKBFlQM1x3xRheT4eXRjwGGMn2zz+ldmBjmub8fQG5+H2pIGCkQk5PahbiZ8+2t2bjR2gVgqysJDhfmYZ4H06/lWdKrz3xO7cA3CjgZ/8ArU6xCf2ejiQI0h+//dAGP8aWziSecGIEwx/KD6+59eams7I2oK7sb1gn7gbQAcYyK2rJcMp28jgZrGhby1CjoDyPSt/RwsgO7GM45OK82Z7VFrY6zSpwF29dp6V1Voq3CE856DNc3ptqAiquM9T/AIV1NonyAAfMDjI7VzNHdFmxpq5GzeMAcACugtA5TaxVznOOlc1a4hu9yvtI4IPSt+C5VV7qfXOankByvodDCgNvuIPy54UirewbBwc4FZNtcfuwq8kkHjitKGcSqm44I5IH1q7GckRzqwnVcD3HemmVhgqQc9h19KnmlickqR7sKbG0LueVGB+tKwdNSzbxkBSTkjrV2KJV+dhjHvVSN41jAZsc/dNX4ngA3P1z371pFGMmxSinG0j19qgZW59fpUxkjVNwAHPApu9duTiqcbkplB93CjAFU7wjgYyD2q5c7PM+UqfYHpWfcFkyGBPYA1zyVjeOpjatDJ9gkkQOcAnC1x9k4i1ZXjYESZD5b5eRxx2JGfxAr0CYb1UngY5BFeb+JwNI1aO9iCRIxwXboDkEEj6/lmtcPK0kZYmN4M3dWB/sKKSWMT5H7wLnBTOOR/nrWfEqKgRDuRBsVt24sB0J9+1ZP9vS3ttDayfuI7lJI2EnIXn5uR74x9a1rZNlqiltzAAMcY5r2IbHz89x5FFL39qKq5BaHNZPiSye/wDC15bK23dE2fyNaoz3pJhCYDFK6AyAgIzAbvYUAfIdySulLtQxxhynpkCtPSGC6bGV6NycDpVTxlZz6RrGrWUwZCsjsisOgL9qsaIWOhW7MOdvQVFdaJm+E+I0ydz4GQKtf8JHYeHrc/aGZpW5WNOWNZl3dNbW/nDBYcdDXFX891dXrGFU3k7mduTXPCmpbnbOs6fw7nodp8TpVnwsbq542MuMfhXU2XxTTT4FlvnjBfnOOFHr715Po/hrV76RpYLhk3c44bH6VvXfw916e3LO+9wudwXAolTpJ2uEK1dq9j0OH416LJKd5kck9xtX8/Suj074paRetiOXGV4OSQa+cLzQdV0+5K3wtpABggsFNJpq6haSiVZVAHP7t805UKbV0xwxVZO0kfZem+JoL+1t3hkXI4fByee9dPYXbNEGRwQvcnFfJ2heL7iHy0lm+UHnnacd69h8G+NYbm3W2aQOTkKcknA+tcFSDgepTqKoepW8kks0kgJ2ZIJByCKi/tK3tGeSS5wA2G9BUXhqFr7TXfksQeCMAmuG8YWuqWN5LGoxE53bucnmsjRtXaO5fxJCXO11KgF/MB4ArlNe+Lmm6PdeRJI+AuT8ucD39K4oeJEsrWa0MYkjZSQegJJ7+w9BXmfiW3Opu9zI8EHJbdJLt5rena9mYVea14o9muP2g9B8rySssbnIVsEhqrr8aLq6bbEkuG+6r4U/n/hXzxYeE4769w2vWEYJz9/J/DtXoWm/DDU47ZPsmt2vljnAYMT9PSumUaa6nHGdV7xPQ5Pipqmft8cRa3JCNE/LZ7gjrj3BrvNE+JGjeJdPgQpLDckbCpG4Z9AwryTSvhnqaOZbvU3mAHAjj6+2f8K0Rpl7pMnksJikf+rZ4+R+PUjvisJqD2N6bmn7x7Ok3mKxySFGOvSuO8fWIuPCU0jsY2Uht3pzWn4Uv7m902P7cGYkY8wH74HQ+3pUXju3c+DNQVSf9V3571hBe+jeo/cZ5tZusXh6F3fcyv5g3nlVbG45+orvwVCAADoM4+lecs8ltoekwxK0m7yTtIyxUna34V6Ig/dgAcDjNezF6HzlRe9oOJ4opMGirMS8q5IFfM3jLWbnXPGV+8t0++GZkSIOQYVBwAB26V9Mrwc18p+ObR7P4zX+wbT9ofgDqCc/1rOodmDipNmXrd1qV+ol1eRrlkiEQnc5ZlByNx7ntmtnStv9jW+B1QZFU/EAjh0CYlQT0z6GrulgtpEIK4OwcGs5zbgjaFNQqtIoaos1xKIYWZce3X8aSxsUgHm3DRqicliMY+tbsOnyTqzrHkAE9QKxtQtrmaBoniYof4ScA/XHUVkp82iNvZ8r5iC68Zah5gtfD8arIekhXk+/sK5DXvE3jf8AtCa3vddlHkrk/MVUnAOAB3r0HwxpVpJcpBNAjktks3B/Ouq1b4c6Frc8MySahbXTAI22FZVIHr0/OtoVKcOhlUoVqiumef8AgqBfECaLDB4lupNTvbo2l3YT2O5YhgkSK+cMDxxwRzWl4q8JzeHNektdStzaSkHZcWoPlSAdyO30Neu/D3wJpfhPVTdWcE018oZVvbqMBYeOqRjuc43Emp/GWjWOqK4udUvJ2fhhC23r1H0qamIg32NaGDnFe8z5vklns9Q8qRg46hx/EPWvUvhxJ9p12NBnPXI5rH1jwrpul2oihZpI1XCl+SDnsfT2rofhTabPEsBIAycYrnrSUoaG9CDjVsfVvhC1QWGwAcgZJo8U6Xa3FrIZo94UdMdal8NyeWuNxXsM1c1VRMJIycgiuHaJ1ST5z5j8X6bqiaitvZ2pPmqWT+6Me/tXnSaTLqPimO0VlurljgyOCUi+gr631jSLW70aeJ4fLV49rzqMsFznaPQeteQX0emWnieZo4CLjcGE8pI3j8OK0o1uXdGsqanHc+dPHv8Awkmha5d29lqrxQWx2gCIAsQRx04610Flrmq65c+GtJ8NeK9S1C6vQIry31K2G22l44RhgsODyO1eu6z4P0DxRe/a9S0++hmlwGnspFdXwOrKR1/wrf8AAnww8OaBrKahpdjf3F+EPlXF+VCxEjBKqvQ4OM+/1r1I4iDjax4tTB1VU5lJnBaR498R+CtSfSvESSCZWJWKR98UwH/PNzyp69eK9a0y/wBD8b+G4tT0qYf7SYO5G7gj1qLVvA9hrV+INSt45pSRux8w/P8ApV/QvANv4b1ISWaMsbj52VtnmD/aHQn3615laUb+6j1YU2ormdy9oMc1i4i3BkB5YsTz+VP8ZTN/whuoMWAHknJAzxXTPZwxIAsSx/3e+K53xLEJvCl5EyE7oHG31GDWUX7yY2rxaOP8DeH7nVZYpZ2IggjMCyKBgEH/AAI59q6rxVrHh7wnY2trdyHzrpvLhhiXc7++Ow96b8OLjyvAEEVqA811I7FufXqfpiqWvaHFe+J9OluojLIZwu4jJwOcfSumpUbkRhKUUldbjlIZAwGAeQKKklULK4A4DH+dFeqtkfNz+JljOelfMfxEEp+OtysihVM36bQR+lfTi9ea8L+MOjrafEez1nGEuYM5/wBpRtP6YqKi0OnBStOxwPiby10a4VG3NIM7T2rQ00N9hhUfe2D+VYuqzxXlsLRTmQ9Me3NbOlS+dpkMoOMqKwa9yx1S/i/I6/TYo5EKFAGIHIq7P4fbygrwkA5bd39hVXw9IPtcYXGBwQa9XgsLW701YnBaTjLDt7V585OL0PRhFOJ5Vb+CtTaQXECKgb7gFdPo+j+IIJdpt1cDAyX4/Su9h0aPCQk7mQfK7c1oJa+UpVwGGO3apc29zWMbLQwotM1mW3+e7ihToVSMkj8TWbcaUsEUjF2kyP8AWN1znoD6V1s2VTJxx261zmpyLHbMXB+Ukge+aVwSex5T4ktzJdjzW+6fujuaZ4QY2/iqFQ+35xUuuO0l0xPXrUHhqOQ+IoiqljvHSt/smaVpqx9UeHCZYUVjjPatTU3IvgCCMDGKw/CbMLdGIwT2NbF6/mXbFuOaxtoay0kESJPA0PqCCPU1yOpeFbST91fWyvET8r8ZWuuiBWUZ6dRitOW1intmdeuORiiMTOUnF6HnFt4KeAj7Jf7VHJWRc49ORzWla+GdWR+TGV7sjn+VdAbMwgHlsd8cgelXbeR1iO4AMO4qnEOeRl2miQ27FwQZGGGO3pVma283bvQbIzkVoMd3PJPXNLsEaEEfe6ehqJxEpN7mQxT5jsBI6CuZ1aTzrOVQAylW4PpW/cukc7ndtxzXJa3dJBZ3THAwjfyqIblyVkV/hbJbWvhr7O5X7SC6qoGdoLEn8+K6C8xBr+mYAY5eRs/w8day/AEV0mlW8csCI8yB+R/e5/rV/wAQoba+nlO0tgQRn8Msfyrp5XKQ0404cz6Iw5W3ysw5ySaKZziivXR8o3d3LINeffGHTze+BI71R+9s5wwP+yw2n+legdqz9d0tda8N3umP/wAt4mRT6Njg/nihq6sXSlyzTPjtUcaqA77H2vsJ/vY4rpPDdx5uj7GILR5Bx9aW705JbGZp7bbPA7Iw9GHFZXh66S3v3tG74Tg98Vi/eiz0Knu1E+56Bo155V2uWBUNjFemaHq8icBjhsk88/hXj1vJtlXvhuBXoGh3KuYz5h9+PavPrRPTw89LHrFncvJErB1wMYBPWtCN9+c9fUVylhcLv2oeNg49q24JiWbsNvXP4D+dc51WC6IBzgnnk1y3iO6xblUOQM9K39QlKR5ySf6V5/4m1BLa1ldn3Dae+KEU0rXOH1O5zP5YYEk813XgLSY5JRLLGSww24dq8xtQZrsXMucvkgHt6V7V8P5FMCIpCkld3NdElZHPSfM2z1jRwII+UznAGK0po2M3muuwdqraZH5gwqjI5Xn9a29RtGhsY3OfmA/D3qeTQcpq9jLBxIVOOlbNvMfsxQ4OeBn+VYAZXBAOWX3q7Z3WW2OCXHQ1KdhON0Xi6t8uAcHIxUHkHblBls8n8asEkKWBCk/zoRlUEH5GIPPWrsRtsVmlZGAc5xznFNlvN0RBB6enWnzL5hOWGPXtVCfbHGojJwM1nJlJIo6hIGjyvDFcHNeY+PZynhuaESmPzHSMtnsXGf0r0m6lVpJBghV6g9+K8f8AiNcjfZQqrMfPEjqP4lB5H60Uo3mhVpcsGereHiy2Gl5IJjtokOP90VS8S3K3GtOiEbEJIx6n/wDVV3TYri30azhQJDcSQB5OS3lrjP51z80nm3MkmScnjPpXZh43nc5MwqJUUl1IsjFFH4Yor0DwiyKcvFMGc0vegR4T8StNGleM7pYVHl3i/aVAHQng/qK8UsbiRPF2VG0BvmUdATX1B8VvD9xqOixavZRmSayBEiqMkxnkkeuDzXyu0cw1Ln+GY5J6nn5f0zUxWrR1VKl4wfVHpVsMhZDzjpXQaXdvHPt3svIrndKkM+nx7sBioyFNa0UhMwkQ8ryc+lcNRdD06U9mj0bSL53iDoSMHGPX8a6uxvB5LFgACQRk9K860q4/ffuyck7gPf8Aya7G1mJtcnk5II964ZKzPUpu6NC+uwsLlmzgdD/OvGvGmqvcztAGwpOGOe2a9H1icmzZY2BJ4+leWa7Y5n4wSck5NXRs5akV21HQoz3cFuheSdIkGPmY4AFehfDbxFbrerBLMsiP8ytkEH8a8B8SE3Eotbhg0YboT6UnhfxEPD+p5siWtd/zpnjd6j3r0XQ5o6HlrFqE7PY+9NO1gKBtkHP3fWtbWPEw+wqHkDMowB0r5j0v4oXFzdW6Wq+bxjGCK39c1zT9Z0dLXXNRngikfZMltIU2gj+JhzjnoK5OSa0O5Spy9657Jpuv2dxIUSaF26NtYHFaV2ZoYYr2Mny25PvXivw++GUOk+JJZobuW301sOQsuWlHbH+NfRNvbW91pf2ZEURKNgAPTHGKylDojdyUbMh067S6iAfB/A1acFuwP41zypPpeotCwBUH73r6VttMfKBB4/i45NKMtNTKUdboilAUY3t9KoTD52RWPp0qaRmkLcEH9KiEaksWck9R2xUt3HexmXJAVhnBxjpyK8O8a6i1z4ytfKkVo4HMagDGTkZyfpnn2r2XW7prWyMpVQR97J65H+NfN9tdyav46LeZlZL6Mqu7IJD7ZB+APT61thY6tnLiZ6Jd2fS+pbbDR/LicszxpGHY5JGK5vHatHU9QivTGluP3cZ69j2GKoHrXoUIcsTzcfU56nKugw/jRSnFFbnCWB04paQGl7UAIyhgRXyZ8SrCLSviprCJFHh086KNFwFZiF6DjvmvrUH0r5d+OkbW/j5rhCqme3XcoHJYA8Z/GmhHOaHf7GVJJMx5wM87j6129oqy25k3AHpnFeOaXqflzxcnodqr6nr/AEr0nTdWU2yKMhj054A6En8a5q9N9Dvw1Xozu9MaP92Mt8p4Y8fSutgu0itvMJCOc8g/KR2zXEaPeRSWZ8xyrA7SDxmsDxN4zdYFtLGJ5i3yD5to984/rXB7JzlZHrLERpwuzrNZ8So87RxuGCsVZsj0ya8813xKEMhLgLjaFTnJI6Z9aw59Qka1eSRCq43DnHmY/iPoPSuduJriVYruSMqkuVt16ksTt4H1z+Rrto4ZI87EY5taDUtbzxBqxUAIkjbFB4Xd2y345rdtfD1tF9nSJi5fmNnGA69A+PQkHGe3PerdvHpkKTwRRhrTS4lZ8khriRs7h+JwM9lFWrtLmK1Zt0LGTyAxRyWQMG2ovoWIY+yjjrXXLsjz42bvItalrraPZWNpaBY5bg7fNQA5UHr7UaxqJutHXyVy3mbmYcsCeDgfXB/E1zfieC4Oqad5G88eUuOhGBjHpnt+Fbuhacsei+bMwChyTLnO+I42sBnhgTg/72KzlFWTOhVW24o6XwF451qxuzb3MjBRbKISJQinaeGyfukDj3xzmvo/wf47uJPDX7yVJL2NQztIu3cD0fHHQnBHXvXyTKt5Fd2azXOXbBjlAH71MAg898r+ldC/jLUdMjNuYnY2o2JCZOGTg7fwzwfX0rnq0uZ3R14fEWXLM+rW8bWN5dRw3oEcmPlccj3/AArdt5RJENjK0RHBDZ46/lXyZYeN4bi7tXBVoDgFyTHgHryORn19q9b8F+OLiYLZOwd0fGw4IdSeCCDgN14HBxkVxVKbWtj0KdSLVkz1uUq+1I2IByc9KqyXIW3bqPLGctznNIl6t3F+5lOOhGOQe4PvXO6/qSWomJy8BXMuGHC4+969e9YpMcmc38RNcaO0FkH8pW2yGQAHAzg/iDgj6GvF/C6xzeLF3B2E1+rF1b5c85I9iQDj0q/458VPdXf2RrlZYxIFkLKAASCQwHpwcj1NQ/DiVL3xnpMFsR9ntbeQt5a4CkEk59zmvSw0LI8rGT6Loe5rjauAAPbtTqCB+FJ2rsPNA+9FB6UUASj0ooAoA45oAXJ7V4T+0NpC/wDCNW+oKimZJwDJjlUIPX2zj8q927VwnxW0FtZ+H18EUyOiBhGehwecDuT2oW4M+PrCSMX6uqhQMAkfqK6201AqW2HYpwQeoAxgcevtXJFTaXccZXD5Axjtnk/XmnyXLIE8p9xznb6YOMn6f0qpRuOE7Hp6anFbaT59xLgEnbHnknufpXMPqiytPeSszxKo+VPlB+bCr+Pp7e1Y2o3UqW0bb3JC+Wg6YX+9+NN0iCaWECSVvKTbKQ3QEAhf51ioKOpvOq5e6Tanqkc1xfxpOEiY7QCjHhRgDH1OSTVXSWu5dShuz5hW2YAHrjAGAPT049Sa09P8P3NxsZNqrJnDN8x292x3yavR+DrtHbyb25R/RHI4/lWntEtBRoSnqyXSjefZZ45ra2/fSK8qN1bC8YPb5ucdKvWvh65ntvKF95fmSBywOWUhdowfXbkZ7ZqbTPAGo3MyPevJIrng+YfxrvtO+EFlJbLPJdzwMSAvlyt1Jx6/Wsp1ktEz0sNlymrtHKap4FmuNNtvs18ohjG9QGGVPA5754HNTJ4d1CDTwyXMZJjKNkHBbpvA6A4xx3PPevTI/g9HBDaOmqXjiRtrL5uSRx+PetBPhFaSH95LKqbiuXlYk/risnWdtztWXU272PAb6LUTqUqR2oktgSsG5seSCc5x69sDjmrF4n2pLgXzETrsRZd43MCuDjjnHv8A0r1+f4MWVwhWJJVbkkljxzjg1n3HwAjFk80+o3InGSqrIxGfzp/WIJWbOatlnLrH8zynSruza/ks5JyInj2xvIn3HUcjA9T29zXYWWuQ2UNtdBjB5fRUPylD3b6Z6jnofUVylzoOo+G/EMlteOVWMhkkZRyM8Hnr2qVbe6tLK7EYLwspYowyFbIxgHpxn8qbUZHEnOHqj6h0TxtBLoCSzTs9werqAwdwOVJHqOh/2a4vxT4xMUi3kSi5gUNI57hCCT09Tn864Lw5rUth4ZjurlmjiljVdgYhQVyBwfulh09cVh32u/Z9YNxvae2cMxjXoYyOw784Psa5Y0NWdksReKfUxvEMz6j4iLAOsGCkkgJIOR8rZ/T8DXrvwd0iGGGa+khVbgRm3lDcknIYH64PP4V5DbZe6eCS4P2f908m5gQG+YAgdTwelfSPgrSJNJ0BYniKGU/aCeP4gAB9MAGu6Gmh5VZ3d2dLgBcAYHamkcUp60h6VoYBmijvRQBJmj60c0YoAM81Q1u0fUdAvrJYd5khZU5xlscH860MCl3EL8uM9s0AfCviXTrjStXuEuoWRoZWQjGCVbODj0znpWGEw8MmXO8Ddk/eG7ofxr6I+N3hm2iso9Rt1C3Ekh356hcZ3HPb6evtXz1eGS2u4o2+9HHycY+g/Ktorm2MW7G/HCkyO93yWCvuHTaDwB68jFWp7yIadZxIpiCqQ7Ku4tk/KfcgdKwrfV5BoclgIQXUrIhX7wB6j3GcHH19a19Htb2+skzCWG8hG2kFQoJY/mRjPvWbi09TZTT2Oy8PTW8ccfmhygAVFLcHjqfX/wCtXT29xaxsLhYwysCq7hxn2FcBbRfZ7ITJqEjSSfL93jjgkDP4Z/KtK0upJIQGkA2qVLSN05wAPf8AwrjqQ1uenQrK3Kdta+IHs9mxlaMt8rEc9fTsK6N/ia9npq3FtaQzYJdSQOeSA34kHFeTS3BJcHM7EEJwc5x2x+ArYgjaDT4YUuIbm8ldI/IijBSDJ65Gc42n24pKknqzT65OGkWeh2XxV1/yGuri3tiF+dExz0zj8K2/DfxmtdRhjm1SCKBmJjjcLkbgBnPp1/SvHXkkh8SH5SGkd2EbEkhG+VjtPUgEH/gJo8NR2ot5IplIR7jYCT91+SQpPUHaTn0PtTnh0ldBTzCblZn0M/xIsrnyoon37yeYhkDH8v8A9XrXVWF6brSfOkjH3SXXBBx1yPWvBPDEMFrf2d3ZO1xD5vKbeUyAAT7cke/Fepza5beHFVERXtgJHzu+UYIIAb+HjNefOGp6CquS1PKPjPBa2999rhDIX5IY71GeOT6H0rg1v99q1rcRxwloxkMx2lcAYB74OD9frXTfE7XI9aupvs9viVD5oTCnch+8MDvz075yOleWbpxbq08geJmVVSQkLxyuPbH616GHp3ieVi6yjPQ77xFdDTNCsdILLJcwNtYjncGOdvXBGcHPvXJWtrJPg280peG3cI5bhD0K/mQTmqE2qtldQt/OZ1UxSQzgcdgV+n+B9au6fcW5stRiijkLznyw33fJRlzk4/75Psa3hDlVjkqVeZ3Oj8DaCNV8QxBwhYbmJBOyULj5QRyDntjpivp6xhW20a0sxGR5UYG49+Oa8e+F+mzP4jt7swPthhARwed3BOB0IwRz9fWvZ9xPJOT3qrWMXK408GkOaU4zxTT6UxCUUUUATD6UUgp1AB26UY5ox60tIRg+J9Ah1jSJbcxxSs8bqElOdxKkZ9sfrXw74nUWmuzwPlCj7drHqBx1/OvsD4p+O4vB2i20UIE2p37+RaQbsEluCxxztGfxr5a+Ieita+JLiKSQMY5Dl+pdh96t6WhlUV9uhx1v5z3EtwyEJGAXx/Dzhf1r0COeb7I9sEMmWQMrPlmlY4jBUHoBvY+uCe4rktGkiS4uLqWJHgyhljY+/Bx3Oa0bvU4rbV2AZUMbs0xTkuw5DD+8dpxn61pONzOLOj1C6RbhVSeR1iCxxL0HQ9B/kDtWYl4/nuC3zA/Kq8qAOw9T2zWBHe3M8DyDeJJJBGFHHlqBzz+Q/OuggsPtGpDTbXa0jjaFQ4GMDkk+pP6VjOCSNoTdzTtLs3moM4njhWIAsZpQGOT2/vdDXWwG4gt3ayhm2RRCR2CZlbnCIvPBY46dB+NYGh6bFHO1wVjeGMECXbwcdWTPUDoCeu76V0A1Bre3mtrbyolkBVmZtzBm4+Y9zlgPQc1g0dad1dnMarHfza47XNw73SiNGaMkqJH2/IG7gEkjHXBrsI02Xeo2EKtIAqTWzgD5JFYgsB025HXjjPvWHbq0E1lc3jIwsIxPb2+fvFcBCc9fmXLH0wBXThkfQPtELK7MjQb+AzliNz4z1+8Bn+9xRUmOjT1Nnwvd2EWpKZ3itrclWkjdiFjRXXGCOh3ZGOoz7Va8ceKUMkdhbSRBPmWSWNt4ZP4WwORg49euehrAudPil3xWhaWKVd/LHduIXfj3J24P+Fc1rckdvqUcglMjCOMt5gJWQEA/g3bHf+fOqPM7nZLEOC5TPvtTE98kLRJJJbuGSUjO6M/w4HOQe1UJVhuLGS1twm6RmiaH7wDclSM9sDg9eCORUMLql9NewsxlBDNERtG4MApGOmM+33vai8vtNuPtV0lzLC7zB5YHT54iBhiSONpOfTGe1ehCKijyKk3KV2Y7C5itt08CqsYJYdASOOR25/OtLwsLm+v43gyJwxBBcHzSTjHTHO7aabdW0gYSrcRTSXH7mRgN5JBwCQR1/wAPeu++D3g64Orx628UjxoxxMcFFIHGPfPH4+1RVkoRbZdCDqTUUe/+HdJXStOj8iBIzsGd3LZxyN3+elbPSl1aew0XxBpPh9y0TX+mRXlmXORNgASKD6q3OPQ0MD0qFqkxyVm0MPWk69acRSEUxDelFONFADwead+NFFACZ5rL8Q+JNH8K6DLq2tXawQJ0H8UjdlUdzRRVQV3YUnZXPlC31+6+J/7UGlX9yrpbC5DxQFsiKKIFgv4kDPua1PHulC68UapEyF5HmZx6gZ55+h6fSiiqxL5WrFYRKcXfqeYxL9jujYTCKKAzAeYyn5ueo9SB09KHmji0VJVVPNmuCYCDkgBeWPrnIA9MGiit73sclrXIrNJxCiGQtuISKNGyVyOW92PIH4mvSvBmnxR22oSXhUPJFIHAbLIpALMW9SMAY6ZNFFZ1DWG5rWcX2iOS4uWCKm2QoeRhAXEaj0ORk++T0ArAuLqebWkjRg22cAbByfm3D/x4sD+FFFY2N09i/fMJteS1ulUSwxgMu7gR/fYn6c/mavaZd77cq4RgYlO3+Mg5+YeuMBsexooqVFNF+0aloa9nMq+NEmjlMkRijuLc9A6qcc9s8lfcE5rG8RxvcXkAgAM8E7wjLcIz72VR7bj39cUUVcFZkVJNo4qbUo44FklXat3FsfKkbRk5bPUkEKcVm6xKItTeW1SVAQu5upJIw59CCynIPHNFFbI5zqvDVrqPiLxkljawxrhhIR90Im0DKg8AbQowfY19ZaB4bsdJ0dYbexWKPiSLaoGA2T83POM/zoorzcbNt8p7OWQVubqH7Q+lzH9nDwz4qsGMWo6Jdo0VwnDIrgqce2Qv5Vi+AfGdp418JwajFIi3aAR3UAPMcg68eh6iiiu2CvRTPMqu1aSOqwcZNIcUUVmA0g4ooooQH//Z" style="width:130px;height:130px;border-radius:50%;object-fit:cover;border:3px solid #29b6f6;box-shadow:0 4px 16px rgba(0,0,0,.5)"></div>
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
        "decommissioned pads. HIE is a 3-phase ML pipeline that validates every candidate "
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
    🔍 <b>Phase 1</b><br><span style="font-size:10px">Grounding DINO<br>Visual check</span>
  </div>
  <div style="color:#475569;font-size:18px">→</div>
  <div style="background:#0d2137;border:2px solid #60a5fa;border-radius:8px;
              padding:10px 14px;color:#93c5fd;min-width:130px;text-align:center">
    🧠 <b>Phase 2</b><br><span style="font-size:10px">LLM text search<br>Status check</span>
  </div>
  <div style="color:#475569;font-size:18px">→</div>
  <div style="background:#0d2137;border:2px solid #34d399;border-radius:8px;
              padding:10px 14px;color:#6ee7b7;min-width:130px;text-align:center">
    📋 <b>Phase 3 ⭐</b><br><span style="font-size:10px">ADIP arrival<br>coordination</span>
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
                caption="Grounding DINO bounding box on a rooftop helipad (ESRI zoom 19)",
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
**Phase 1 — Visual Validation: Grounding DINO**

For every FAA and OSM candidate pad, SkyRoute fetches a zoom-19 ESRI satellite chip
(~0.22 m/px at lat 41°) and runs it through **Grounding DINO** — a zero-shot open-set
object detector prompted with the text *"helipad"*.

| | |
|---|---|
| **Input** | 768 × 768 px satellite chip centred on registry coord |
| **Model** | Grounding DINO (open-set, no helipad-labelled training data needed) |
| **Output** | Bounding box `[x₁,y₁,x₂,y₂]` + confidence score |
| **Action — detected** | Centroid projected back to lat/lon → coordinate corrected if offset > 15 m |
| **Action — not found** | Pad flagged `unverified`; excluded from routing until manually reviewed |
| **Metric (M3 KPI)** | Detection rate · mean coordinate offset (m) vs FAA registry |
""")
    with ph1_ex:
        st.markdown("""
<div style="background:#0a1628;border:1px solid #a78bfa;border-radius:8px;padding:12px 14px;font-size:12px">
<div style="color:#a78bfa;font-weight:700;margin-bottom:8px">🔍 Grounding DINO — example output</div>
<div style="background:#1a1a2e;border-radius:6px;padding:8px;margin-bottom:8px;font-family:monospace;font-size:11px;color:#e2e8f0">
<span style="color:#fbbf24">Satellite chip</span> · zoom 19 · 768×768 px<br>
<span style="color:#34d399">▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮</span>  ← rooftop building<br>
<span style="color:#f97316">┌──────────────┐</span>  ← detected bbox<br>
<span style="color:#f97316">│</span>  <b style="color:#fff">H</b> marking     <span style="color:#f97316">│</span>  conf: <b style="color:#22c55e">0.91</b><br>
<span style="color:#f97316">│</span>  yellow border <span style="color:#f97316">│</span><br>
<span style="color:#f97316">└──────────────┘</span><br>
<span style="color:#60a5fa">centroid</span>: 40.7236 N, 74.0482 W<br>
<span style="color:#60a5fa">registry</span>: 40.7238 N, 74.0479 W<br>
<span style="color:#22c55e">offset</span>: <b>8.3 m</b> → within tolerance ✓
</div>
<div style="color:#64748b;font-size:10px">
Typical rooftop pad — yellow safety border and H marking are
the primary detection anchors. Building parallax at zoom 19
introduces ±3–8 m apparent offset.
</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Phase 2 ─────────────────────────────────────────────────────────────────
    ph2_col, ph2_ex = st.columns([3, 2])
    with ph2_col:
        st.markdown("""
**Phase 2 — Text/LLM Status Validation (OSM-only pads)**

OSM pads that have no FAA match are the highest-value but highest-risk additions to
the routing network. Phase 2 uses the **OSM name tag + coordinates** as a query to
an LLM with web-search access (Google Gemini / GPT-4o) to verify operational status.

| | |
|---|---|
| **Input** | OSM pad name, coordinates, optional `operator` / `access` tags |
| **Query** | *"Is [name] helipad at [location] currently operational? Military or civilian?"* |
| **Model** | LLM + live web search (Gemini Search Grounding / Bing) |
| **Output** | `operational` · `military` · `closed` · `uncertain` |
| **Action — operational** | Pad promoted to routing pool ✅ |
| **Action — military / closed** | Pad excluded; tagged with reason 🚫 |
| **Action — uncertain** | Held for Grounding DINO visual re-check 🔄 |
| **Metric (M3 KPI)** | Classification accuracy on a hand-labelled held-out set |
""")
    with ph2_ex:
        st.markdown("""
<div style="background:#0a1628;border:1px solid #ef4444;border-radius:8px;padding:12px 14px;font-size:12px">
<div style="color:#f87171;font-weight:700;margin-bottom:8px">🚫 LLM verdict — military pad excluded</div>
<div style="background:#1a0a0a;border-radius:6px;padding:8px;margin-bottom:8px;font-family:monospace;font-size:11px;color:#e2e8f0">
<span style="color:#fbbf24">OSM pad</span>: Caven Point USAR Center Heliport<br>
<span style="color:#94a3b8">FAA IDENT</span>: NJ77 (OSM faa-tag match)<br>
<span style="color:#94a3b8">Location</span>: Jersey City, NJ<br><br>
<span style="color:#60a5fa">Query →</span> Gemini Search Grounding<br>
<span style="color:#e2e8f0">"Is Caven Point USAR Heliport<br>
 operational and open to civil use?"</span><br><br>
<span style="color:#f87171">Result</span>: <b style="color:#ef4444">MILITARY</b><br>
<span style="color:#94a3b8;font-size:10px">U.S. Army Reserve Center · private-use<br>
New York District Corps of Engineers<br>
→ civilian routing: EXCLUDED 🚫</span>
</div>
<div style="color:#64748b;font-size:10px">
Without LLM validation, this pad would be offered as a
landing option to civilian passengers — a safety and
regulatory violation.
</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Phase 3 ─────────────────────────────────────────────────────────────────
    st.markdown("""
**Phase 3 — ADIP-Guided Arrival Coordination ⭐ Stretch Goal (M3+)**

The FAA ADIP (Airport/Facility Directory Information Program) per-heliport record contains
operational data that goes far beyond the basic registry: TLOF/FATO dimensions, touchdown
bearing, ingress/egress corridors, ATC contacts, and inspection date. Phase 3 ingests this
data to upgrade routing from *"navigate to coordinates"* to *"arrive on the correct approach
bearing and clear the obstacle surface."*
""")

    adip1, adip2, adip3 = st.columns(3)
    adip1.markdown("""
<div style="background:#0a1628;border:1px solid #34d399;border-radius:8px;padding:10px 12px;font-size:12px">
<div style="color:#34d399;font-weight:700;margin-bottom:6px">📐 TLOF / FATO Dimensions</div>
<div style="color:#94a3b8">Touchdown & Lift-Off zone size + Final Approach and Take-Off surface bounds.
Used to match aircraft type to pad capacity.</div>
</div>""", unsafe_allow_html=True)
    adip2.markdown("""
<div style="background:#0a1628;border:1px solid #34d399;border-radius:8px;padding:10px 12px;font-size:12px">
<div style="color:#34d399;font-weight:700;margin-bottom:6px">🧭 Approach Bearings & Corridors</div>
<div style="color:#94a3b8">ADIP remarks encode preferred ingress/egress bearings and obstacle-clearance
notes. Enables turn-by-turn aerial approach instructions in the routing output.</div>
</div>""", unsafe_allow_html=True)
    adip3.markdown("""
<div style="background:#0a1628;border:1px solid #34d399;border-radius:8px;padding:10px 12px;font-size:12px">
<div style="color:#34d399;font-weight:700;margin-bottom:6px">📋 Facility & Inspection Data</div>
<div style="color:#94a3b8">Design category, last inspection date, EV charging availability, and ATC
contact. Feeds the HIE freshness score and infrastructure readiness flag.</div>
</div>""", unsafe_allow_html=True)

    st.caption("⭐ Phase 3 is a stretch goal for M3. Raw ADIP JSON is already fetched and stored in data/adip_raw/ for offline field discovery.")


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
                "Provides the academic umbrella for HIE Phase 2: querying an LLM (Gemini/GPT-4o with search "
                "grounding) about a helipad's operational status from its name and location. "
                "The retrieval-augmented hybrid pattern identified in the review is exactly the architecture "
                "used — OSM name + coordinates as retrieval keys, LLM as the reasoning layer — and the review's "
                "benchmarks offer a framework for evaluating HIE Phase 2 classification accuracy."
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
var map = L.map('map', {center: [40.75, -73.5], zoom: 8, layers: [osmDay]});
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

var allPointLayers = [helipadLayer, osmHelipadLayer];

L.control.layers(
  {'Street Map': osmDay, 'Satellite (ESRI)': esriSat},
  {
    'Helipads (FAA)': helipadLayer,
    'Helipads (OSM)': osmHelipadLayer,
    'Business POIs':  poiLayer,
    'Exec. Residences': resLayer,
  },
  {collapsed: false}
).addTo(map);
helipadLayer.addTo(map);
osmHelipadLayer.addTo(map);
poiLayer.addTo(map);
resLayer.addTo(map);

// ── utils ─────────────────────────────────────────────────────────────────────
function haversine(lat1, lon1, lat2, lon2) {
  var R=6371, dLat=(lat2-lat1)*Math.PI/180, dLon=(lon2-lon1)*Math.PI/180;
  var a=Math.sin(dLat/2)*Math.sin(dLat/2)+
    Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLon/2)*Math.sin(dLon/2);
  return R*2*Math.atan2(Math.sqrt(a),Math.sqrt(1-a));
}
function fmtDur(m) {
  if (m<60) return Math.round(m)+' min';
  return Math.floor(m/60)+'h'+(Math.round(m%60)>0?' '+Math.round(m%60)+'m':'');
}
function fmtDist(km) { return km<1 ? Math.round(km*1000)+' m' : km.toFixed(1)+' km'; }
function getAllHelipads() {
  var pts=[];
  allPointLayers.forEach(function(ly){
    if (!map.hasLayer(ly)) return;   // skip unchecked layers
    ly.eachLayer(function(l){
      if(!l.feature) return;
      var ll=l.getLatLng?l.getLatLng():l.getBounds().getCenter();
      var p=l.feature.properties;
      pts.push({lat:ll.lat, lon:ll.lng, name:p.NAME||p.name||'Helipad'});
    });
  });
  return pts;
}
function nearestHelipad(lat,lng,pts) {
  var best=null,bd=Infinity;
  pts.forEach(function(p){var d=haversine(lat,lng,p.lat,p.lon);if(d<bd){bd=d;best=p;}});
  return best?{pad:best,dist:bd}:null;
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

// ── OSRM ──────────────────────────────────────────────────────────────────────
async function osrmRoute(lat1,lng1,lat2,lng2,profile) {
  var url='https://router.project-osrm.org/route/v1/'+profile+'/'+
    lng1.toFixed(6)+','+lat1.toFixed(6)+';'+lng2.toFixed(6)+','+lat2.toFixed(6)+
    '?overview=full&geometries=geojson';
  try {
    var r=await (await fetch(url)).json();
    if(r.code==='Ok'&&r.routes&&r.routes.length)
      return {dist:r.routes[0].distance/1000, duration:r.routes[0].duration/60, geom:r.routes[0].geometry};
  } catch(e){}
  var d=haversine(lat1,lng1,lat2,lng2);
  return {dist:d, duration:d/30*60, geom:null};
}

// ── Multi-modal computation ────────────────────────────────────────────────────
async function computeMultiModal(pA, pB) {
  var sb=document.getElementById('status-bar');
  sb.innerHTML='&#x23F3; Computing routes&hellip;'; sb.style.display='block';

  var driveR=await osrmRoute(pA.lat,pA.lng,pB.lat,pB.lng,'driving');
  var driveDur=driveR.duration, taxiDur=driveDur*1.1, transitDur=driveDur*1.5;
  var walkDist=haversine(pA.lat,pA.lng,pB.lat,pB.lng), walkDur=walkDist/SPEED_WALK_KMH*60;

  var allPts=getAllHelipads(), nearA=nearestHelipad(pA.lat,pA.lng,allPts), nearB=nearestHelipad(pB.lat,pB.lng,allPts);
  var heli=null;
  if(nearA&&nearB) {
    var hd=haversine(nearA.pad.lat,nearA.pad.lon,nearB.pad.lat,nearB.pad.lon);
    if(hd>0.1&&hd<=RANGE_KM) {
      var d2a=await osrmRoute(pA.lat,pA.lng,nearA.pad.lat,nearA.pad.lon,'driving');
      var d2b=await osrmRoute(nearB.pad.lat,nearB.pad.lon,pB.lat,pB.lng,'driving');
      var hDur=hd/SPEED_HELI_KMH*60;
      heli={dur:d2a.duration+hDur+d2b.duration, dist:d2a.dist+hd+d2b.dist,
            hd:hd, hDur:hDur, padA:nearA.pad, padB:nearB.pad, gA:d2a.geom, gB:d2b.geom};
    }
  }

  // best ground time
  var groundTimes=[driveDur,taxiDur]; if(walkDur<90) groundTimes.push(walkDur);
  var bestGround=Math.min.apply(null,groundTimes);

  // ── draw RED: driving route ───────────────────────────────────────────────
  if(driveR.geom)
    routeLayers.push(L.geoJSON(driveR.geom,{style:{color:'#ef4444',weight:5,opacity:0.85}}).addTo(map));
  else
    routeLayers.push(L.polyline([[pA.lat,pA.lng],[pB.lat,pB.lng]],{color:'#ef4444',weight:5,opacity:0.7,dashArray:'6 4'}).addTo(map));

  // ── draw GREEN+CYAN: aerial route ─────────────────────────────────────────
  if(heli) {
    if(heli.gA) routeLayers.push(L.geoJSON(heli.gA,{style:{color:'#22c55e',weight:4,opacity:0.9}}).addTo(map));
    routeLayers.push(L.polyline([[heli.padA.lat,heli.padA.lon],[heli.padB.lat,heli.padB.lon]],
      {color:'#00d4ff',weight:3,dashArray:'10 6',opacity:0.95}).addTo(map));
    if(heli.gB) routeLayers.push(L.geoJSON(heli.gB,{style:{color:'#22c55e',weight:4,opacity:0.9}}).addTo(map));
    routeLayers.push(L.marker([heli.padA.lat,heli.padA.lon],{icon:hIcon('#00b8d9'),zIndexOffset:500})
      .bindTooltip('Take-off: '+heli.padA.name,{direction:'top'}).addTo(map));
    routeLayers.push(L.marker([heli.padB.lat,heli.padB.lon],{icon:hIcon('#22c55e'),zIndexOffset:500})
      .bindTooltip('Landing: '+heli.padB.name,{direction:'top'}).addTo(map));
  }

  // ── fit bounds ────────────────────────────────────────────────────────────
  var pts=[pA,pB];
  if(heli){pts.push(L.latLng(heli.padA.lat,heli.padA.lon));pts.push(L.latLng(heli.padB.lat,heli.padB.lon));}
  map.fitBounds(L.latLngBounds(pts),{padding:[50,50]});

  // ── table ─────────────────────────────────────────────────────────────────
  var rows=[
    {name:'&#x1F6B6; Walking',         dur:walkDur,    dist:walkDist,    air:'&mdash;', ground:true},
    {name:'&#x1F697; Car',             dur:driveDur,   dist:driveR.dist, air:'&mdash;', ground:true},
    {name:'&#x1F695; Taxi',            dur:taxiDur,    dist:driveR.dist, air:'&mdash;', ground:true, note:'est.'},
    {name:'&#x1F68C; Transit/Subway',  dur:transitDur, dist:driveR.dist, air:'&mdash;', ground:true, note:'est.'},
  ];
  if(heli) rows.push({
    name:'&#x1F697;&#x2708; Car + Heli + Car',
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
  var nodes=[{lat:start.lat,lon:start.lng,name:'Origin'}].concat(allPts).concat([{lat:end.lat,lon:end.lng,name:'Destination'}]);
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
    drawHeliRoute(findRoute(ptA,ptB,getAllHelipads()));
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
</script>
</body>
</html>"""


def build_routing_html(faa_df: pd.DataFrame, osm_df: pd.DataFrame) -> str:
    """Build self-contained Leaflet routing HTML with FAA and OSM helipads injected as GeoJSON.

    Args:
        faa_df: FAA helipad DataFrame with lat, lon, IDENT, NAME columns.
        osm_df: OSM helipad DataFrame with lat, lon, name, faa, surface, ele columns.

    Returns:
        Complete HTML string for use with streamlit.components.v1.html().
    """
    faa_features = []
    for _, row in faa_df.dropna(subset=["lat", "lon"]).iterrows():
        props: dict = {}
        for col in ["IDENT", "NAME", "STATE", "SERVCITY", "OPERSTATUS", "ELEVATION"]:
            if col in row.index:
                v = row[col]
                props[col] = str(v) if pd.notna(v) else ""
        faa_features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [float(row["lon"]), float(row["lat"])]},
            "properties": props,
        })

    osm_features = []
    for _, row in osm_df.dropna(subset=["lat", "lon"]).iterrows():
        props = {}
        for col in ["name", "faa", "surface", "ele", "osm_id", "aeroway"]:
            if col in row.index:
                v = row[col]
                props[col] = str(v) if pd.notna(v) else ""
        osm_features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [float(row["lon"]), float(row["lat"])]},
            "properties": props,
        })

    faa_geojson = json.dumps({"type": "FeatureCollection", "features": faa_features})
    osm_geojson = json.dumps({"type": "FeatureCollection", "features": osm_features})
    return (
        _ROUTING_HTML_TEMPLATE
        .replace("__GEOJSON__", faa_geojson)
        .replace("__OSM_GEOJSON__", osm_geojson)
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

        # Poll every 400 ms and call invalidateSize() whenever the Leaflet
        # container has zero dimensions (hidden tab on first render).
        # Stops automatically after 20 s so it does not run forever.
        from branca.element import Element
        m.get_root().html.add_child(Element("""
<script>
(function(){
    var _n=0;
    var _iv=setInterval(function(){
        _n++;
        if(typeof window.map!=='undefined'){
            var s=window.map.getSize();
            if(s.x===0||s.y===0){window.map.invalidateSize(true);}
        }
        if(_n>50){clearInterval(_iv);}
    },400);
})();
</script>"""))

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
        HIE validation pipeline — Grounding DINO visual detection (Phase 1) followed by
        LLM operational-status classification (Phase 2) — promoting OSM-only candidate
        pads into the live routing pool.
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
    dm = folium.Map(location=[40.8, -73.8], zoom_start=8, tiles=None)
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
    dm.get_root().html.add_child(_El("""
<script>
(function(){
    var _n=0;
    var _iv=setInterval(function(){
        _n++;
        if(typeof window.map!=='undefined'){
            var s=window.map.getSize();
            if(s.x===0||s.y===0){window.map.invalidateSize(true);}
        }
        if(_n>50){clearInterval(_iv);}
    },400);
})();
</script>"""))

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
    sm = folium.Map(location=[40.85, -73.8], zoom_start=8, tiles=None)
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
    sm.get_root().html.add_child(_El("""
<script>
(function(){var _n=0,_iv=setInterval(function(){_n++;if(typeof window.map!=='undefined'){var s=window.map.getSize();if(s.x===0||s.y===0)window.map.invalidateSize(true);}if(_n>50)clearInterval(_iv);},400);})();
</script>"""))
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
    rm = folium.Map(location=[40.85, -73.8], zoom_start=8, tiles=None)
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
    rm.get_root().html.add_child(_El("""
<script>
(function(){var _n=0,_iv=setInterval(function(){_n++;if(typeof window.map!=='undefined'){var s=window.map.getSize();if(s.x===0||s.y===0)window.map.invalidateSize(true);}if(_n>50)clearInterval(_iv);},400);})();
</script>"""))
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
components.html(build_routing_html(faa_raw, osm_raw), height=650, scrolling=False)

st.divider()
st.caption("SkyRoute HIE · Technion LBS Course 016833 · FAA ADDS-ArcGIS + OpenStreetMap")
