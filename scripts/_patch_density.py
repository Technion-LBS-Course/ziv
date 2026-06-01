"""One-shot script: replace the Density & Hot Spots tab in app.py with the
enhanced version that adds residential access, coverage circles, full journey
KPI, and improved M3 metrics.

Run once:  python scripts/_patch_density.py
"""
from pathlib import Path

APP = Path(__file__).parent.parent / "app.py"

# Use line-number splicing to avoid encoding issues with box-drawing chars
# Density tab comment starts at line 2047 (0-indexed: 2046), with tab_density: at 2048
# Last line of old block: 2417 (0-indexed: 2416, the closing `)\n`)
TAB_COMMENT_LINE = 2047   # 1-based; the "# ── Density & Hot Spots" comment
TAB_END_LINE     = 2418   # 1-based; blank line after closing `)` of old block (exclusive)

NEW_BLOCK = r'''    # ── Density & Hot Spots ─────────────────────────────────────────────────────
    with tab_density:
        st.markdown("### Helipad Density vs Demand: Business & Executive Residential Access")
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
'''

# ── splice using line numbers ────────────────────────────────────────────────
lines = APP.read_text(encoding="utf-8").splitlines(keepends=True)
n = len(lines)
assert n >= TAB_END_LINE, f"app.py has only {n} lines, expected >= {TAB_END_LINE}"

before = "".join(lines[:TAB_COMMENT_LINE - 1])   # everything before the comment line
after  = "".join(lines[TAB_END_LINE - 1:])        # everything from blank line onward

new_src = before + NEW_BLOCK + after
APP.write_text(new_src, encoding="utf-8")
print(f"Patched {APP}  ({n} lines before -> {len(new_src.splitlines()):,} lines after)")
