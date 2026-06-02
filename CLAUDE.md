# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> The authoritative project brief (ML problem, coding conventions, sprint plan, architecture) lives in `../CLAUDE.md`. Read that first. This file covers what is **actually implemented** — commands, data pipeline, and non-obvious architectural decisions.

> **Session log:** `Worklog.md` records every development session — what was built, issues hit, and how they were resolved. Read it to understand the history behind non-obvious decisions. **Update it at the end of every session before committing.**

---

## Commands

```bash
# Install dependencies (Python 3.11+)
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run app.py

# ── Data pipeline (run in order) ─────────────────────────────────────────────
# Step 1 — fetch FAA + OSM raw data (~747 FAA records, 5-state NE US)
python scripts/fetch_ny_data.py

# Step 2 — enrich FAA records with ADIP Airport Master Record data
#           reads faa_helipads_raw.csv → writes faa_adip_enriched.csv + data/adip_raw/*.json
#           ~6 min at 0.5 req/sec for 747 records
python scripts/fetch_adip_details.py

# ── Tests ─────────────────────────────────────────────────────────────────────
# tests/ directory is not yet created (M2 deliverable)
# When it exists: pytest tests/
```

There is no linter or formatter configured.

---

## Repository Layout (actual)

```
ziv/
├── CLAUDE.md               ← YOU ARE HERE
├── README.md               ← Project overview, persona, ML formulation
├── requirements.txt        ← Pinned dependencies
├── app.py                  ← Streamlit entry point (~2700 lines)
├── src/
│   ├── __init__.py
│   ├── data.py             ← Ingestion, cleaning, schema normalisation
│   └── analysis.py         ← Cross-source matching, consistency, completeness
├── scripts/
│   ├── fetch_ny_data.py    ← Download FAA (ArcGIS) + OSM (Overpass) for NE US
│   └── fetch_adip_details.py  ← Enrich FAA records with ADIP Airport Master Record
├── assets/
│   └── helipad_grounding_dino.jpg  ← Satellite chip illustration for HIE Phase 1
│                                      (copy your image here; app shows placeholder if absent)
├── data/                   ← Raw data files — NEVER commit (gitignored)
│   ├── faa_helipads_raw.csv          (fetch_ny_data.py output)
│   ├── faa_adip_enriched.csv         (fetch_adip_details.py output — preferred by load_faa_data)
│   ├── osm_helipads_raw.csv          (fetch_ny_data.py output)
│   ├── bookmarks.json                (user helipad bookmarks, persisted by app)
│   └── adip_raw/<IDENT>.json         (one JSON per helipad from ADIP API)
└── notebooks/
    └── 01_eda.ipynb        ← Not yet created (M2 deliverable)
```

---

## Data Pipeline Architecture

The pipeline is two-stage and auto-selecting:

```
fetch_ny_data.py         → data/faa_helipads_raw.csv   (FAA ADDS-ArcGIS, 28 cols)
                         → data/osm_helipads_raw.csv   (OSM Overpass API)

fetch_adip_details.py    → data/faa_adip_enriched.csv  (raw CSV + 23 ADIP cols)
                         → data/adip_raw/<IDENT>.json  (full raw response, one per heliport)

load_faa_data()          → auto-detects faa_adip_enriched.csv if present,
                           falls back to faa_helipads_raw.csv silently
```

`load_faa_data()` in `src/data.py` uses the enriched file automatically when it exists — no parameter change needed. ADIP columns are mapped at load time: `adip_status` → `operational`, `last_info_days_ago` → `data_freshness_days`, ADIP ARP coordinates upgrade ADDS-FAA ones where available.

---

## ADIP Enrichment Details

The ADIP endpoint requires:
- `POST https://adip.faa.gov/agisServices/public-api/getAirportDetails`
- Body: `{"locId": "<IDENT>"}`
- Header: `Authorization: Basic 3f647d1c-a3e7-415e-96e1-6e8415e6f209-ADIP` (static app key from Angular bundle)
- Session warm-up required: GET `https://adip.faa.gov/agis/public/` first to establish `JSESSIONID`

A richer per-heliport XLSX is available at:
`https://adip.faa.gov/agisServices/public-api/downloadHeliportDataAsExcel/<IDENT>`
Same auth header. Contains 4 sheets: heliport/vertiport dimensions (TLOF/FATO), facility data (design category, last inspection, EV charging), remarks (ATC contacts, ingress/egress bearings), and schedule. `openpyxl` is in requirements for this. **Not yet fetched by the script** — planned enrichment, raw JSON saved in `data/adip_raw/` for offline field discovery.

---

## `src/analysis.py` — Cross-Source Matching

All functions are pure (no I/O) so they're safe for `@st.cache_data`.

| Function | Description |
|----------|-------------|
| `haversine_matrix(lat1, lon1, lat2, lon2)` | Pairwise distances (N×M metres array) |
| `match_by_faa_id(faa_df, osm_df)` | Exact match: FAA.IDENT == OSM.faa tag |
| `match_by_proximity(faa_df, osm_df, threshold_m)` | Nearest-neighbour, respects exclude list |
| `match_rate_by_threshold(faa_df, osm_df)` | Curve of match % at standard thresholds |
| `build_consistency_table(faa_df, osm_df, matches)` | Per-pair coord/elevation/name consistency |
| `faa_completeness(faa_df)` | Per-column null % for FAA data |
| `osm_completeness(osm_df, key_fields=...)` | Per-field completeness for OSM data |

Two-tier matching cascade:
1. **FAA-ID exact**: `FAA.IDENT == OSM.faa` tag — ~24% of OSM records have this
2. **Proximity fallback**: nearest-neighbour haversine, excluding already-matched FAA records

Proximity thresholds are not arbitrary — they are `1.5 × FATO diameter` per helicopter design class:

| Class | FATO | Threshold |
|-------|------|-----------|
| R22 (small) | 50 ft | 23 m |
| Hospital rooftop | 60 ft | 27 m |
| Bell 206 (medium) | 70 ft | 32 m |
| S-92 (large) | 175 ft | 80 m |

Name similarity strips OSM "Helipad"/"Heliport" suffixes before comparison *unless* the FAA name also contains a helipad keyword — OSM contributors routinely append these words to names that FAA records without them.

---

## Streamlit App Architecture (`app.py`)

### Tab structure

```
st.tabs(["📍 Problem", "📚 Literature", "🏪 Market", "📊 EDA & HIE"])
```

**Tab 1 — Problem**
- Persona card: Miles Urban, VP BD, Bronxville NY, age 44
- Journey comparison: without SkyRoute (92 min, 3 apps) vs with SkyRoute (36 min, 1 booking)
- 4 KPI metrics (time saved, trips/week, hours reclaimed, booking friction)
- Stakeholder ecosystem (4 cards: Operators, Passengers, Vertiport Owners, Regulators)
- HIE ML pipeline flow banner + 3-phase detail:
  - Phase 1: Grounding DINO visual validation (satellite chip → bounding box)
  - Phase 2: LLM text/status validation (Gemini search grounding, Caven Point example)
  - Phase 3: ADIP arrival coordination — stretch goal

**Tab 2 — Literature**
- Quick Reference table: article, field of relevance, DOI link
- 4 expandable paper summaries with abstract + "SkyRoute benefit" callout:
  1. O'Reilly et al. 2024 — eVTOL site scoring (São Paulo)
  2. Zhang et al. 2026 — Air-ground multimodal routing optimisation
  3. Singh et al. 2025 — Few-shot Grounding DINO for aerial imagery
  4. Eyinade & Ademusire 2025 — GeoLLMs for geospatial understanding

**Tab 3 — Market**
- Competitor comparison table and market sizing (content not detailed here)

**Tab 4 — EDA & HIE**
- 3 inline data charts:
  1. Field completeness: FAA vs OSM (3 paired fields: IDENT/faa, NAME/name, ELEVATION/ele)
  2. Elevation consistency scatter + delta histogram
  3. Location deviation for matched pairs (FAA-ID vs proximity matches)
- KPI section: "Helipad Coverage & Access Time for AAM Routing" with formal KPI definition (model/indicator/because)
- Density & hot-spots heatmap (Folium + HeatMap plugin)
- Executive residence spider map
- Multi-modal routing simulator (HTML/JS component)

### Caching strategy

`@st.cache_data` on: `load_data()`, `compute_matches()`, `compute_threshold_curve()`, `build_search_entries()`, `fetch_imagery_meta()`.

`cons = compute_matches(...)` is computed once at the top level (after the first sidebar block) and shared across EDA charts, analysis filters, and the sidebar — do not recompute it inside a tab.

### Map jump / satellite switch mechanism

Uses `_bk_ver` (int) as the Folium component `key`. Incrementing it forces a full re-mount at the new location. `_use_satellite` is only `True` when `_satellite_ver == _bk_ver`, so the satellite layer activates only on the render triggered by a jump.

### Session state keys that matter

| Key | Purpose |
|-----|---------|
| `_bk_lat`, `_bk_lon`, `_bk_zoom`, `_bk_ver` | Active jump target |
| `_satellite_ver` | Which `_bk_ver` should auto-switch to ESRI satellite |
| `_last_center`, `_last_zoom` | Updated from `map_state`; used for imagery metadata caption |
| `_last_bk_ver` | Detects a fresh jump to pre-populate `_last_center` before user pans |
| `_ac_last`, `_af_last` | Debounce autocomplete / analysis-filter selects |

### Two `with st.sidebar:` blocks

The first defines `prox_threshold`; `cons = compute_matches(...)` is computed between them; the second block (analysis filters) uses `cons`. This ordering is intentional — Streamlit renders both blocks into the sidebar in document order.

### Leaflet hidden-tab initialization fix

Streamlit hides inactive tabs with `display:none`. Leaflet measures the container as 0×0 and skips auto-resize. Fix: inject a self-contained polling script into each Folium map's own iframe via `branca.element.Element`:

```python
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
```

This pattern is applied to the main Folium map (`build_map()` return), the density heatmap (`dm`), the hotspot map (`sm`), and the residential spider map (`rm`).

### Routing simulator (`build_routing_html`)

Builds an HTML/JS page injected via `components.html()`. Layer-aware: `getAllHelipads()` calls `map.hasLayer(ly)` for each overlay layer and skips unchecked ones — routing only uses visible helipad layers. Four overlays: FAA helipads, OSM helipads, Business POIs, Executive Residences. `HeliControl` and `MMControl` buttons are positioned `bottomright` to avoid overlap with the layer control.

### ADIP hotlink in popups

Every FAA marker popup contains:
```html
<a href="https://adip.faa.gov/agis/public/#/simpleAirportMap/{IDENT}">📋 ADIP record</a>
```

---

## What Is Not Yet Implemented (M3 scope)

| Item | Notes |
|------|-------|
| `src/model.py` — `build_features()`, `train_model()`, `evaluate_model()` | XGBoost binary classifier on structured registry features |
| `merge_helipad_sources()` spatial deduplication | Currently a simple concat; spatial dedup intentionally deferred |
| ADIP XLSX fetcher (TLOF/FATO dimensions, design category, remarks) | Raw JSON already in `data/adip_raw/` for offline field discovery |
| OSM geospatial enrichment (`dist_to_hospital_km`, `dist_to_city_center_km`) | Requires geopandas joins; pre-compute offline → cache as Parquet |
| VLM helipad bounding-box validation (interactive overlay on helipad select) | Plan in `.claude/plans/` — uses ESRI tiles + Claude vision API |
| Streamlit Cloud deployment | Set env vars from `.env.example` |
