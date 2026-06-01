#!/usr/bin/env python3
"""Patch app.py: replace 5 EDA sub-tabs with 3 insight charts + single KPI tab."""
from pathlib import Path

APP = Path(__file__).parent.parent / "app.py"
lines = APP.read_text(encoding="utf-8").splitlines(keepends=True)

# ── locate splice boundaries ──────────────────────────────────────────────────
# START: find "tab_cov, tab_faa_t," then go back 1 to include preceding st.divider()
tab_line = next(
    i for i, ln in enumerate(lines)
    if "tab_cov, tab_faa_t," in ln
)
# Go back 1 line to the blank line before st.divider(), then one more for st.divider()
# Actually we want START to be the index of "    st.divider()" that precedes the tab block
START = tab_line - 2   # tab_line-2 = the "    st.divider()" line

# END: first line that has "Density & Hot Spots" as a comment (keep this line and all after)
END = next(
    i for i, ln in enumerate(lines)
    if i > tab_line and "Density & Hot Spots" in ln and ln.strip().startswith("#")
)

print(f"Splicing: replace lines {START+1}..{END} with new EDA charts + KPI tab.")

NEW = """\
    st.divider()
    st.markdown("### \U0001f4ca EDA — Data Intelligence")

    # ---- Chart 1: Field Completeness FAA vs OSM ----------------------------------
    _FAA_KEY = ["NAME", "STATE", "SERVCITY", "OPERSTATUS", "ELEVATION", "ICAO_ID"]
    _fc_all  = faa_completeness(faa_raw)
    _fc_key  = _fc_all[_fc_all["field"].isin(_FAA_KEY)].copy()
    _fc_key["pct_present"] = (100 - _fc_key["null_pct"]).round(1)
    _fc_key  = _fc_key.sort_values("pct_present")

    _ocomp = osm_completeness(osm).sort_values("pct_present")

    _fc_n_complete  = int((_fc_key["pct_present"] >= 99).sum())
    _osm_n_complete = int((_ocomp["pct_present"] >= 99).sum())

    st.markdown("#### 1 — Field Completeness: FAA vs OSM")
    _cc1, _cc2 = st.columns(2)

    with _cc1:
        _fig = px.bar(
            _fc_key, x="pct_present", y="field", orientation="h",
            color="pct_present",
            color_continuous_scale=["#FFCCBC", _FAA_COLOR],
            text=_fc_key["pct_present"].apply(lambda x: f"{x:.0f}%"),
            title=f"FAA Field Completeness  (N={len(faa_raw):,})",
            labels={"pct_present": "% Records with Value", "field": ""},
        )
        _fig.update_traces(textposition="outside")
        _fig.update_layout(xaxis_range=[0, 115], coloraxis_showscale=False,
                           height=320)
        st.plotly_chart(_fig, use_container_width=True)

    with _cc2:
        _fig = px.bar(
            _ocomp, x="pct_present", y="field", orientation="h",
            color="pct_present",
            color_continuous_scale=["#FFCCBC", _OSM_COLOR],
            text=_ocomp["count"].apply(lambda x: f"{x:,}"),
            title=f"OSM Field Completeness  (N={len(osm):,})",
            labels={"pct_present": "% Records with Value", "field": ""},
        )
        _fig.update_traces(textposition="outside")
        _fig.update_layout(xaxis_range=[0, 115], coloraxis_showscale=False,
                           height=320)
        st.plotly_chart(_fig, use_container_width=True)

    st.info(
        f"\U0001f4a1 **Insight — OSM characteristic data is critically sparse.**  "
        f"FAA registry fills **{_fc_n_complete} of {len(_fc_key)} core fields** at ≥99 % "
        f"coverage (coordinates, name, and operational status are universal). "
        f"OSM contributors cover only **{_osm_n_complete} of {len(_ocomp)} fields** at that rate: "
        f"surface, lighting, and elevation are missing for the majority of records. "
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
            f"\U0001f4a1 **Insight — High agreement with identifiable outliers.**  "
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
            f"\U0001f4a1 **Insight — Near-perfect for ID matches; good accuracy for proximity.**  "
            f"FAA-ID matches cluster at **{_id_med2:.0f} m median** — near-zero deviation "
            f"confirms both sources describe the same physical pad. "
            f"Proximity matches show **{_prox_med2:.0f} m median** with "
            f"**{_pct_u30b:.0f}%** of pairs within a single helipad-width (30 m). "
            f"The long tail (>50 m) marks decommissioned pads or OSM coordinate drift "
            f"— high-priority candidates for the M3 Grounding DINO validation pass."
        )

    st.divider()

    [tab_density] = st.tabs(["\U0001f525 KPI"])

"""

new_lines = lines[:START] + [NEW] + lines[END:]
APP.write_text("".join(new_lines), encoding="utf-8")

n_new = len(new_lines)
print(f"Done. {len(lines)} -> {n_new} lines.")
