"""SkyRoute Helipad Intelligence Engine — EDA Dashboard.

Run:
    streamlit run app.py
"""

import logging
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
FAA_PATH = DATA_DIR / "faa_helipads_raw.csv"
OSM_PATH = DATA_DIR / "osm_helipads_raw.csv"

_FAA_COLOR = "#1565C0"
_OSM_COLOR = "#E65100"
_MAP_CENTER = [40.8, -75.5]
_MAP_ZOOM = 7

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyRoute HIE",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw FAA and OSM CSV files.

    Returns:
        Tuple of (faa_df, osm_df) as raw DataFrames.

    Raises:
        FileNotFoundError: If either CSV is missing.
    """
    if not FAA_PATH.exists():
        st.error(f"FAA data not found: {FAA_PATH}. Run: python scripts/fetch_ny_data.py")
        st.stop()
    if not OSM_PATH.exists():
        st.error(f"OSM data not found: {OSM_PATH}. Run: python scripts/fetch_ny_data.py")
        st.stop()
    faa = pd.read_csv(FAA_PATH)
    osm = pd.read_csv(OSM_PATH)
    return faa, osm


faa_raw, osm_raw = load_data()

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚁 SkyRoute HIE")
    st.caption("Helipad Intelligence Engine")
    st.divider()

    all_states = sorted(faa_raw["STATE"].dropna().unique().tolist())
    sel_states: list[str] = st.multiselect(
        "Filter states",
        options=all_states,
        default=all_states,
        help="Applies to FAA data. OSM shown for full bbox (limited state tags).",
    )
    st.divider()
    show_faa: bool = st.checkbox("FAA layer", value=True)
    show_osm: bool = st.checkbox("OSM layer", value=True)
    st.divider()
    st.caption(f"**FAA** ADDS-ArcGIS · {len(faa_raw):,} records")
    st.caption(f"**OSM** Overpass API · {len(osm_raw):,} records")
    st.caption("Coverage: NY · NJ · CT · PA · MA")

# ── apply filters ──────────────────────────────────────────────────────────────
faa: pd.DataFrame = (
    faa_raw[faa_raw["STATE"].isin(sel_states)].copy()
    if sel_states else faa_raw.copy()
)
osm: pd.DataFrame = osm_raw.copy()

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🚁 SkyRoute — Helipad Intelligence Engine")
st.caption("EDA Dashboard · FAA ADDS-ArcGIS + OpenStreetMap · Northeast US")
st.divider()

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("FAA Heliports", f"{len(faa):,}")
k2.metric("OSM Helipads", f"{len(osm):,}")
k3.metric("FAA Operational",
          f"{(faa['OPERSTATUS'] == 'OPERATIONAL').sum():,}",
          delta=f"{(faa['OPERSTATUS'] == 'OPERATIONAL').mean()*100:.0f}%")
k4.metric("FAA Military", f"{(faa['MIL_CODE'] == 'MIL').sum():,}")
k5.metric("OSM Named",
          f"{osm['name'].notna().sum():,}",
          delta=f"{osm['name'].notna().mean()*100:.0f}%")

st.divider()


# ── map ────────────────────────────────────────────────────────────────────────

def build_map(faa_df: pd.DataFrame, osm_df: pd.DataFrame,
              show_f: bool, show_o: bool) -> folium.Map:
    """Build a Folium map with clustered FAA and OSM layers.

    Args:
        faa_df: Filtered FAA DataFrame.
        osm_df: OSM DataFrame.
        show_f: Whether to add the FAA layer.
        show_o: Whether to add the OSM layer.

    Returns:
        Configured ``folium.Map`` instance.
    """
    m = folium.Map(location=_MAP_CENTER, zoom_start=_MAP_ZOOM,
                   tiles="CartoDB positron")

    if show_f and not faa_df.empty:
        faa_grp = folium.FeatureGroup(name=f"FAA ({len(faa_df):,})", show=True)
        cluster = MarkerCluster(disableClusteringAtZoom=12)

        valid = faa_df.dropna(subset=["lat", "lon"])
        # Vectorised popup construction
        popups = (
            "<b>" + valid["NAME"].fillna("Unknown") + "</b><br>"
            + "IDENT: " + valid["IDENT"].fillna("—") + "<br>"
            + "State: " + valid["STATE"].fillna("—") + "<br>"
            + "City: " + valid["SERVCITY"].fillna("—") + "<br>"
            + "Status: " + valid["OPERSTATUS"].fillna("—") + "<br>"
            + "Type: " + valid["MIL_CODE"].fillna("—") + "<br>"
            + "Elev: " + valid["ELEVATION"].astype(str) + " ft"
        )
        for lat, lon, popup_html, name in zip(
            valid["lat"], valid["lon"], popups, valid["NAME"].fillna("FAA Heliport")
        ):
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=_FAA_COLOR,
                fill=True,
                fill_color=_FAA_COLOR,
                fill_opacity=0.8,
                weight=1.5,
                popup=folium.Popup(popup_html, max_width=230),
                tooltip=name,
            ).add_to(cluster)
        cluster.add_to(faa_grp)
        faa_grp.add_to(m)

    if show_o and not osm_df.empty:
        osm_grp = folium.FeatureGroup(name=f"OSM ({len(osm_df):,})", show=True)
        cluster_o = MarkerCluster(disableClusteringAtZoom=12)

        valid = osm_df.dropna(subset=["lat", "lon"])
        names = valid["name"].fillna("Unnamed").astype(str)
        surfaces = valid["surface"].fillna("unknown").astype(str)
        aeroway_tags = valid["aeroway"].fillna("helipad").astype(str)

        popups = (
            "<b>" + names + "</b><br>"
            + "Type: " + aeroway_tags + "<br>"
            + "Surface: " + surfaces + "<br>"
            + "OSM ID: " + valid["osm_id"].astype(str)
        )
        for lat, lon, popup_html, name in zip(
            valid["lat"], valid["lon"], popups, names
        ):
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=_OSM_COLOR,
                fill=True,
                fill_color=_OSM_COLOR,
                fill_opacity=0.6,
                weight=1,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=name,
            ).add_to(cluster_o)
        cluster_o.add_to(osm_grp)
        osm_grp.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


st.subheader("Geographic Distribution")
st.caption("Click a cluster to expand · Click a marker for details · Toggle layers (top right)")
map_obj = build_map(faa, osm, show_faa, show_osm)
st_folium(map_obj, width="100%", height=530, returned_objects=[])

st.divider()


# ── analysis tabs ──────────────────────────────────────────────────────────────
st.subheader("Analysis")
tab_cov, tab_faa_tab, tab_osm_tab = st.tabs(
    ["📊 Coverage", "🔵 FAA Deep Dive", "🟠 OSM Deep Dive"]
)

# ── TAB 1: Coverage ────────────────────────────────────────────────────────────
with tab_cov:
    col_a, col_b = st.columns(2)

    with col_a:
        state_cnt = faa.groupby("STATE").size().reset_index(name="Heliports")
        fig = px.bar(
            state_cnt.sort_values("Heliports", ascending=False),
            x="STATE", y="Heliports",
            title="FAA Heliports by State",
            color="STATE",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"STATE": "State"},
            text="Heliports",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        osm_state_cnt = (
            osm[osm["addr:state"].notna()]
            .groupby("addr:state").size()
            .reset_index(name="Helipads")
            .sort_values("Helipads", ascending=False)
            .head(12)
        )
        n_tagged = osm["addr:state"].notna().sum()
        fig2 = px.bar(
            osm_state_cnt,
            x="addr:state", y="Helipads",
            title=f"OSM Helipads by State (tagged: {n_tagged:,}/{len(osm):,})",
            color="addr:state",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"addr:state": "State"},
            text="Helipads",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

    # Source comparison
    comp = pd.DataFrame({
        "Metric": ["Total records", "With name", "With elevation", "With state"],
        "FAA": [
            len(faa),
            len(faa),          # NAME always present
            faa["ELEVATION"].notna().sum(),
            faa["STATE"].notna().sum(),
        ],
        "OSM": [
            len(osm),
            osm["name"].notna().sum(),
            osm["ele"].notna().sum(),
            osm["addr:state"].notna().sum(),
        ],
    })
    fig3 = px.bar(
        comp.melt(id_vars="Metric", var_name="Source", value_name="Count"),
        x="Metric", y="Count", color="Source", barmode="group",
        title="FAA vs OSM — Record Count & Key Field Coverage",
        color_discrete_map={"FAA": _FAA_COLOR, "OSM": _OSM_COLOR},
        text="Count",
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(yaxis_title="Records", xaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

# ── TAB 2: FAA Deep Dive ───────────────────────────────────────────────────────
with tab_faa_tab:
    r1c1, r1c2, r1c3 = st.columns(3)

    with r1c1:
        oper = faa["OPERSTATUS"].value_counts().reset_index()
        oper.columns = ["Status", "Count"]
        fig = px.pie(
            oper, names="Status", values="Count",
            title="Operational Status",
            color_discrete_sequence=["#1565C0", "#EF5350"],
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        mil = faa["MIL_CODE"].value_counts().reset_index()
        mil.columns = ["Type", "Count"]
        fig = px.pie(
            mil, names="Type", values="Count",
            title="Civil vs Military",
            color_discrete_sequence=["#1565C0", "#B71C1C"],
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with r1c3:
        use = (
            faa["PRIVATEUSE"]
            .map({0: "Public", 1: "Private"})
            .value_counts().reset_index()
        )
        use.columns = ["Use", "Count"]
        fig = px.pie(
            use, names="Use", values="Count",
            title="Private vs Public Use",
            color_discrete_sequence=["#42A5F5", "#1565C0"],
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig = px.histogram(
            faa.dropna(subset=["ELEVATION"]),
            x="ELEVATION", nbins=40,
            color="STATE",
            title="Elevation Distribution by State (ft)",
            labels={"ELEVATION": "Elevation (ft)", "count": "Count"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(bargap=0.05, yaxis_title="Heliports")
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        anal = faa["AIRANAL"].value_counts().reset_index()
        anal.columns = ["Analysis", "Count"]
        fig = px.bar(
            anal.sort_values("Count"),
            x="Count", y="Analysis", orientation="h",
            title="FAA Airspace Analysis Classification",
            color="Analysis",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Heliports", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Elevation by state box plot
    fig_box = px.box(
        faa.dropna(subset=["ELEVATION"]),
        x="STATE", y="ELEVATION", color="STATE",
        title="Elevation Distribution per State",
        labels={"ELEVATION": "Elevation (ft)", "STATE": "State"},
        color_discrete_sequence=px.colors.qualitative.Set2,
        points="outliers",
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

# ── TAB 3: OSM Deep Dive ───────────────────────────────────────────────────────
with tab_osm_tab:
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fields_info = {
            "name": osm["name"].notna().sum(),
            "surface": osm["surface"].notna().sum(),
            "ele (elevation)": osm["ele"].notna().sum(),
            "lit (lighting)": osm["lit"].notna().sum(),
            "operator": osm["operator"].notna().sum(),
            "access": osm["access"].notna().sum(),
        }
        comp_df = pd.DataFrame({
            "Field": list(fields_info.keys()),
            "Pct": [v / len(osm) * 100 for v in fields_info.values()],
            "Count": list(fields_info.values()),
        }).sort_values("Pct")
        fig = px.bar(
            comp_df, x="Pct", y="Field", orientation="h",
            title=f"OSM Field Completeness (N={len(osm):,})",
            color="Pct",
            color_continuous_scale=["#FFCCBC", _OSM_COLOR],
            text=comp_df["Count"].apply(lambda x: f"{x:,}"),
            labels={"Pct": "% Records with Value", "Field": ""},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_range=[0, 105], coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        surf = osm["surface"].dropna().value_counts().reset_index()
        surf.columns = ["Surface", "Count"]
        surf = surf[surf["Count"] >= 2]
        fig = px.bar(
            surf.sort_values("Count"),
            x="Count", y="Surface", orientation="h",
            title=f"Surface Types (tagged: {osm['surface'].notna().sum():,} records)",
            color="Surface",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Helipads", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        etype = osm["osm_type"].value_counts().reset_index()
        etype.columns = ["Element Type", "Count"]
        fig = px.pie(
            etype, names="Element Type", values="Count",
            title="OSM Element Type (node / way / relation)",
            color_discrete_sequence=["#E65100", "#FF8A65", "#FFCCBC"],
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        aeroway = osm["aeroway"].value_counts().reset_index()
        aeroway.columns = ["aeroway", "Count"]
        fig = px.pie(
            aeroway, names="aeroway", values="Count",
            title="aeroway Tag  (helipad vs heliport)",
            color_discrete_sequence=["#E65100", "#FF8A65"],
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    # Lit values breakdown
    lit_counts = osm["lit"].value_counts().reset_index()
    lit_counts.columns = ["Value", "Count"]
    n_lit_tagged = osm["lit"].notna().sum()
    st.caption(f"Lighting tag (lit): {n_lit_tagged:,} records tagged out of {len(osm):,}")
    fig_lit = px.bar(
        lit_counts,
        x="Value", y="Count",
        title=f"OSM Lighting Tag Values (tagged: {n_lit_tagged:,})",
        color_discrete_sequence=[_OSM_COLOR],
        text="Count",
    )
    fig_lit.update_traces(textposition="outside")
    st.plotly_chart(fig_lit, use_container_width=True)

st.divider()
st.caption("SkyRoute HIE · Technion LBS Course 016833 · Data: FAA ADDS-ArcGIS + OpenStreetMap")
