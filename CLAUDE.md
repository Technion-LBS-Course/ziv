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

# Run annotation review tool (separate Streamlit app)
streamlit run scripts/annotate_dataset.py

# ── M2 data pipeline (run in order) ──────────────────────────────────────────
# Step 1 — fetch FAA + OSM raw data (~747 FAA records, 5-state NE US)
python scripts/fetch_ny_data.py

# Step 2 — enrich FAA records with ADIP Airport Master Record data
#           reads faa_helipads_raw.csv → writes faa_adip_enriched.csv + data/adip_raw/*.json
#           ~6 min at 0.5 req/sec for 747 records
python scripts/fetch_adip_details.py

# ── M3 YOLO dataset pipeline (run in order, resumable) ───────────────────────
# Full pipeline: download HelipadCAT + national FAA → filter → fetch NAIP chips (~3 hrs)
python scripts/build_yolo_dataset.py

# Resume from step N if interrupted (steps 1-2 read from cached CSVs)
python scripts/build_yolo_dataset.py --from 3

# Smoke test first 20 records (note: first ~300 HelipadCAT records are Alaska → no_imagery)
python scripts/build_yolo_dataset.py --limit 50

# Skip NAIP imagery coarser than 0.6 m/px (optional quality filter)
python scripts/build_yolo_dataset.py --max-gsd 0.6

# ── M3 zero-shot comparison (run NOW — no training needed) ───────────────────
# Smoke test (5 chips, ~30 sec)
python scripts/compare_zero_shot.py --limit 5

# Full run: classical + YOLO-World + Grounding DINO on all 747 test chips
python scripts/compare_zero_shot.py --models classical yolo_world dino

# Single model only
python scripts/compare_zero_shot.py --models yolo_world

# Resume after interruption
python scripts/compare_zero_shot.py --resume

# Florence-2 requires transformers<4.49 — disabled by default; enable with:
# pip install "transformers>=4.44.0,<4.49.0" then:
python scripts/compare_zero_shot.py --models classical yolo_world florence2

# ── M3 YOLO training + evaluation ────────────────────────────────────────────
# IMPORTANT: click "Apply all decisions" in annotate_dataset.py FIRST

# Full pipeline: train YOLOv8s + evaluate on 747 test chips + 5 comparison plots
python scripts/train_yolo.py

# Skip training (evaluate existing weights only)
python scripts/train_yolo.py --skip-train

# Reduce batch size if low RAM
python scripts/train_yolo.py --batch 8

# Use GPU (auto-detected; specify explicitly if needed)
python scripts/train_yolo.py --device 0

# ── Tests ─────────────────────────────────────────────────────────────────────
# tests/ directory is not yet created (M2 deliverable)
# When it exists: pytest tests/

# ── M3 XGBoost training ───────────────────────────────────────────────────────
# Train XGBoost on 17 ADIP-derived features; outputs models/xgboost/hie_xgboost.pkl
python scripts/train_xgboost.py

# ── M3 4-model comparison (requires all 4 trained .pt files) ─────────────────
# Full run: evaluate YOLOv8s + YOLO11s + YOLO11m + RT-DETR-L on 747 test chips
python scripts/compare_models.py

# Regenerate plots only (skip inference — uses cached JSON)
python scripts/compare_models.py --plots-only

# Smoke test first N chips
python scripts/compare_models.py --limit 20

# ── Post-training analysis (run after helipad_yolov8s.pt exists) ─────────────
# Registry accuracy: which source (FAA or OSM) is spatially closer to YOLO detection?
python scripts/compare_registry_accuracy.py

# Smoke test on first 10 matched pairs
python scripts/compare_registry_accuracy.py --limit 10

# Change the "possibly different helipad" distance flag threshold (default 50m)
python scripts/compare_registry_accuracy.py --flag-dist 80

# Validate OSM-only helipads via NAIP inference
python scripts/validate_osm_only.py

# Smoke test first 5 OSM-only records
python scripts/validate_osm_only.py --limit 5

# Resume after interruption (skips chips already in osm_validated.csv)
python scripts/validate_osm_only.py --resume

# ── M4 NOTAM + weather (do NOT start until M3 is submitted) ──────────────────
# Smoke test: fetch NOTAMs for 30th St Heliport (NK39)
python -c "from src.notam import fetch_notams_for_ident; import os; print(fetch_notams_for_ident('NK39', os.getenv('FAA_API_KEY')))"

# METAR for JFK (nearest large station to NE US helipads)
python -c "from src.notam import fetch_metar; print(fetch_metar('KJFK'))"

# RainViewer latest frame
python -c "from src.weather import fetch_rainviewer_frames; frames = fetch_rainviewer_frames(); print(frames['radar']['past'][-1])"

# Sample precipitation at 30th St Heliport
python -c "from src.weather import fetch_rainviewer_frames, sample_precipitation_at_latlon; f=fetch_rainviewer_frames(); path=f['radar']['past'][-1]['path']; print(sample_precipitation_at_latlon(40.7503, -74.0025, path))"
```

There is no linter or formatter configured.

---

## Repository Layout (actual)

```
ziv/
├── CLAUDE.md               ← YOU ARE HERE
├── README.md               ← Project overview, persona, ML formulation
├── Worklog.md              ← Session-by-session log of decisions and issues
├── requirements.txt        ← Pinned dependencies
├── app.py                  ← Streamlit dashboard (~3600 lines)
├── src/
│   ├── __init__.py
│   ├── data.py             ← Ingestion, cleaning, schema normalisation
│   ├── analysis.py         ← Cross-source matching, consistency, completeness
│   ├── notam.py            ← FAA NOTAM API + Aviation Weather Center METAR/TAF (M4)
│   └── weather.py          ← RainViewer radar frames, tile URLs, route precipitation check (M4)
├── scripts/
│   ├── fetch_ny_data.py       ← Download FAA (ArcGIS) + OSM (Overpass) for NE US
│   ├── fetch_adip_details.py  ← Enrich FAA records with ADIP Airport Master Record
│   ├── build_yolo_dataset.py  ← 8-step NAIP chip pipeline for YOLO training dataset
│   ├── annotate_dataset.py    ← Streamlit annotation review tool (approve/disqualify/adjust)
│   ├── compare_zero_shot.py   ← Zero-shot ablation on 747 test chips (Classical/YOLO-World/DINO)
│   ├── train_yolo.py          ← Train YOLOv8s + evaluate vs registry baseline + 5 plots
│   ├── train_xgboost.py       ← Train XGBoost on 17 ADIP features; outputs hie_xgboost.pkl
│   ├── compare_models.py      ← Unified 4-model evaluation (YOLOv8s/YOLO11s/YOLO11m/RT-DETR-L)
│   ├── fix_split_duplicates.py ← One-time fix: removes 570 train/val duplicate chips (run with --apply)
│   ├── compare_registry_accuracy.py  ← FAA vs OSM coordinate accuracy vs YOLO bbox centre
│   └── validate_osm_only.py   ← NAIP inference on OSM-only pads → data/osm_validated.csv
├── models/
│   ├── helipad_yolov8s.pt              ← YOLOv8s fine-tuned weights
│   ├── helipad_run_yolo11s/weights/best.pt  ← YOLO11s (discovery model)
│   ├── helipad_run_yolo11m/weights/best.pt  ← YOLO11m (production model — P=0.931 F1=0.888)
│   ├── helipad_run_rtdetr_l/weights/best.pt ← RT-DETR-L transformer baseline
│   ├── plots/                  ← YOLOv8s training plots (from train_yolo.py)
│   ├── plots_comparison/       ← 4-model comparison plots (from compare_models.py)
│   └── xgboost/                ← XGBoost model + feature importance
├── assets/
│   └── helipad_grounding_dino.jpg
├── data/                   ← All data files — NEVER commit (gitignored)
│   ├── faa_helipads_raw.csv
│   ├── faa_adip_enriched.csv         (preferred by load_faa_data — auto-selected)
│   ├── osm_helipads_raw.csv
│   ├── bookmarks.json
│   ├── adip_raw/<IDENT>.json
│   ├── helipadcat_raw.csv            (build_yolo_dataset.py step 1 output)
│   ├── faa_national.csv              (build_yolo_dataset.py step 2 output — all US)
│   └── yolo_dataset/                 (gitignored — large image chips ~GB)
│       ├── images/{train,val,test}/  NAIP chips (640×640 px JPEG)
│       ├── labels/{train,val,test}/  YOLO label .txt (empty = negative)
│       ├── review_decisions.csv      annotation review state (persists between sessions)
│       ├── build_log.csv             per-chip status log from pipeline
│       ├── dataset.yaml              YOLO config
│       └── review_{train,test}/      HTML spot-check galleries
└── notebooks/
    └── 01_eda.ipynb        ← EDA (M2 deliverable)
```

---

## YOLO Dataset Pipeline (`scripts/build_yolo_dataset.py`)

8-step resumable pipeline that builds a YOLO object-detection dataset for helipad visual validation.

### Architecture

```
HelipadCAT CSV (~6000 FAA coords)  ──┐
National FAA ArcGIS (~5653 records) ──┤ Steps 1-4: filter + dedup
NE US 747 test records              ──┘
          │
          ▼ Steps 5-6: NAIP chip fetch
USDA APFO ImageServer (CONUS only)
  → 100m × 100m window, 640×640 px output
  → effective GSD = 0.156 m/px
          │
          ▼ Step 7: YOLO labels + split
  Positive chips (groundtruth=True)
  Hard negatives (groundtruth=False) ← most valuable: FAA-listed but visually absent
  Easy negatives (random CONUS locations)
  Test set: 747 NE US FAA records (held out — never in training)
          │
          ▼ Step 8: HTML review galleries
```

### Key non-obvious decisions

**NAIP source**: USDA APFO ImageServer, not Microsoft Planetary Computer.
Planetary Computer was tried first but rasterio COG reads fail on Windows (pip GDAL VSICURL issue with Azure Blob SAS tokens). USDA `exportImage` is a simple HTTP GET using the existing `requests` session — no rasterio, no authentication.
- URL: `https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/exportImage`
- Native resolution: 1 m/px (standard); 0.6 m/px (enhanced, select states from 2018)
- Effective GSD in pipeline: 0.156 m/px (640×640 px export over 100m×100m window — server upsamples from 1m native)
- Horizontal accuracy: ≤6 m NMAS Class 1, 90th-percentile per USDA FSA spec; newer acquisitions ≤3 m
- Coverage: CONUS only (lat 24–49.5°N, lon 66–125°W). ~5% of HelipadCAT records are Alaska → `no_imagery`.

**HelipadCAT does not ship image chips**: the dataset contains coordinates and annotation metadata only — no images are distributed. Original authors fetched Google Maps Static API zoom-20 tiles at runtime. We cannot use those tiles (commercial ToS + domain shift vs NAIP). All imagery was re-fetched from USDA NAIP and all bounding boxes were re-examined via `scripts/annotate_dataset.py`. This is why a full annotation pass is required before training.

**Bbox scale factor**: HelipadCAT annotated bboxes on Google Maps zoom-20 chips (640×640 px, ~0.114 m/px at lat 40°N). Our NAIP chips are also 640×640 px but cover 100m × 100m. Scale = HelipadCAT coverage / NAIP coverage, computed per latitude:
```python
def _bbox_pixel_scale(lat):
    hcat_m_per_px = 156543.03392 * math.cos(math.radians(lat)) / (2 ** 20)
    return (640 * hcat_m_per_px) / 100.0   # ~0.73 at lat 40°N
```

**Hard negatives**: HelipadCAT `groundtruth=False` records are FAA-listed locations where the authors found no visible helipad. These are the highest-value training examples (decommissioned/stale pads). They go straight into the dataset as hard negatives — do not discard them.

**HelipadCAT `category` column**: Contains string values like `'other'` in addition to integers. Always wrap in `try/except (ValueError, TypeError)`.

**Retry logic**: USDA server occasionally drops connections under load. `fetch_naip_chip()` retries 3 times with 10s/20s/40s back-off before marking as `failed`.

**Resume**: `--from 3` re-runs steps 3–4 (fast filter recompute) and step 5 skips chips already on disk (`tile_path.exists()`). Always use `--from 3` rather than `--from 5` when re-running, because `--from 5` reconstructs from `build_log.csv` which lacks the original HelipadCAT bbox columns.

### .gitignore pitfall

Inline comments on gitignore lines (`pattern  # comment`) are NOT valid — git treats the `#` as a literal character, not a comment. Only lines that START with `#` are comments. Always put comments on their own line above the pattern.

---

## Annotation Review Tool (`scripts/annotate_dataset.py`)

Standalone Streamlit app for reviewing NAIP chips and correcting YOLO labels.

```bash
streamlit run scripts/annotate_dataset.py
```

**Actions per chip:**
- **Approve** — keeps current bbox. If sliders were moved, auto-saves the adjusted bbox (button label changes to "Approve + save bbox").
- **Disqualify** — writes empty label file → chip becomes a negative training example. Useful for stale FAA records where no helipad is visible.
- Navigation auto-advances to next chip after any action.

**State persistence**: all decisions are saved immediately to `data/yolo_dataset/review_decisions.csv`. Kill with Ctrl+C, re-run, and hit "Jump to first unreviewed" to resume.

**Apply**: decisions are staged until you click "Apply all decisions" in the sidebar, which writes the final YOLO label files.

**Planned experiment**: run `scripts/compare_zero_shot.py` (YOLO-World + Florence-2) on the same NAIP chips before manual correction and compare bbox IoU vs `review_decisions.csv`. Will quantify whether zero-shot models can replace or reduce manual labeling for future datasets.

---

## `src/hie.py` — HIE Visual Detection Module

All detection functions return a unified dict:
```python
{"detected": bool, "bbox_px": [x1,y1,x2,y2] | None, "cx": int | None,
 "cy": int | None, "confidence": float, "method": str, "latency_s": float}
```

| Function | Tier | Notes |
|----------|------|-------|
| `detect_classical(image)` | 1 (production) | OpenCV H-template bank at 3 scales × 2 rotations × 2 colour variants; threshold 0.72 |
| `detect_yolo(image, model)` | 2 (production) | YOLOv8s fine-tuned; requires `models/helipad_yolov8s.pt` |
| `detect_helipad_cascade(image, yolo_model)` | cascade | Tier1 → Tier2; Tier1 short-circuits if confidence ≥ 0.75 |
| `detect_yolo_world(image, model=None)` | zero-shot | YOLO-World small; auto-loads `yolov8s-worldv2.pt` (~14 MB) |
| `detect_florence2(image, model, processor)` | zero-shot | Florence-2-base; auto-loads ~460 MB; no per-box confidence score (returns 1.0) |
| `detect_dino(image, model, processor)` | zero-shot (optional) | Grounding DINO tiny; ~661 MB; documented poor nadir zero-shot |
| `bbox_px_to_latlon(bbox_px, lat, lon)` | utility | Pixel bbox centre → (lat, lon) using NAIP window geometry |
| `compute_offset_m(ref_lat, ref_lon, det_lat, det_lon)` | utility | Haversine distance in metres |

**YOLO-World class list:** `["helipad", "landing pad", "H marking"]`
**Florence-2 task prompt:** `<OPEN_VOCABULARY_DETECTION>helipad`

---

## `scripts/compare_zero_shot.py` — Zero-Shot Ablation

Runs YOLO-World, Florence-2 (and optionally classical CV + Grounding DINO) on 747 test chips. **Can run immediately** — no training needed.

Key design decisions:
- IoU computed against synthetic GT labels (bbox at centre, ~0.18 normalised). Rerun after annotation corrections for final numbers.
- GT label empty → negative chip (no helipad expected); GT label present → positive chip.
- `--resume` flag: appends to existing `zero_shot_results.csv`, skipping done (chip, model) pairs.
- Florence-2 does not return per-box confidence scores; reports 1.0 when detected.
- Closest-to-centre box selection for Florence-2 (multi-detection case).

---

## M2 Data Pipeline Architecture

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
st.tabs(["📍 Problem", "📚 Literature", "🏪 Market", "📊 EDA & HIE", "🔍 Inspector", "📈 Results"])
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

**Tab 5 — Inspector (🔍)**
- Isolated via `@st.experimental_fragment` — widget interactions in this tab don't trigger full-app reruns
- **Mode A — Test Set Inspector:** dropdown of 747 NE US helipads by TP/TN/FP/FN category; auto-jumps on selection (no Jump button); NAIP chip + YOLO bbox annotation (left) + CartoDB/OSM reference map (right) in 1:1 side-by-side layout
- **Mode B — Live Inference:** pan/click Folium map → fetch 100m NAIP chip + ESRI XYZ tile chip in real-time → run YOLO → show bbox; checkboxes to toggle FAA NE US / OSM NE US / FAA CONUS overlay layers independently
- ESRI chip fetched via 3×3 XYZ tile grid at zoom 18 (stitched, cropped, resized) — `fetch_esri_chip()` in `src/hie.py`

**Tab 6 — Results (📈)**
- Two sub-tabs: `["📊 XGBoost Structured Baseline", "🤖 YOLO Models Comparison"]`
- XGBoost sub-tab: F1 lift over majority, feature importance bar chart, has_wind spotlight (windsock image + YouTube link), position_age_days distribution chart, full classification report
- YOLO sub-tab: 4 metric KPI tiles → radar chart + PR curve side-by-side → action caption (YOLO11m = production / YOLO11s = discovery) → P vs conf + R vs conf → individual model expanders (each with 5 plots)

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
| `_insp_a_ident` | Last-rendered Inspector Mode A ident — detects selectbox change for auto-jump |
| `_insp_a_chip`, `_insp_a_res` | Cached chip PIL image and YOLO result for current Inspector A selection |
| `_insp_b_faa`, `_insp_b_osm`, `_insp_b_conus` | Inspector Mode B layer visibility checkboxes |

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

## M3 Status — COMPLETE (2026-06-13)

### Final model results (747 NE US test chips)

| Model | Precision | Recall | F1 | Accuracy | Notes |
|-------|-----------|--------|----|----------|-------|
| Registry baseline | 0.63 | 0.21 | 0.32 | — | FAA-ID cross-reference only, no imagery |
| Classical CV (zero-shot) | — | — | 0.00 | — | H-template; building-corner false positives |
| YOLO-World (zero-shot) | — | — | 0.00 | — | Total domain gap |
| Grounding DINO (zero-shot) | — | — | 0.00 | — | Partial IoU ~0.16; no reliable localisation |
| YOLOv8s fine-tuned | 0.906 | 0.801 | 0.850 | 0.837 | CNN baseline |
| RT-DETR-L fine-tuned | 0.907 | 0.815 | 0.859 | 0.845 | Transformer; training instability at epoch 32 |
| YOLO11s fine-tuned | 0.908 | 0.866 | 0.887 | 0.871 | Best recall (discovery model) |
| **YOLO11m fine-tuned** | **0.931** | 0.848 | **0.888** | **0.876** | **Production model — fewest FP=27** |

XGBoost structured baseline (17 ADIP features, no imagery): P=0.74 · R=0.72 · **F1=0.73**

### Completed deliverables

| Item | File |
|------|------|
| YOLO dataset pipeline | `scripts/build_yolo_dataset.py` |
| Annotation review tool | `scripts/annotate_dataset.py` |
| Split duplicate fix | `scripts/fix_split_duplicates.py` |
| HIE detection module | `src/hie.py` |
| Zero-shot ablation | `scripts/compare_zero_shot.py` |
| YOLOv8s training + eval | `scripts/train_yolo.py` |
| 4-model comparison | `scripts/compare_models.py` |
| XGBoost training | `scripts/train_xgboost.py` + `src/model.py` |
| Inspector tab (live HIE) | `app.py` — fragment-isolated, auto-jump, side-by-side chip+map |
| Results tab (YOLO + XGBoost) | `app.py` — radar chart, PR curve, per-model plots, action captions |

### Grounding DINO API fix (transformers 4.51)
`post_process_grounded_object_detection()` no longer accepts `box_threshold`. Fixed in `src/hie.py` — removed `box_threshold=conf`, kept only `text_threshold=conf`.

### Post-M3 items (start after M3 submission 23 Jun 2026)
| Item | Notes |
|------|-------|
| `scripts/compare_registry_accuracy.py` | FAA vs OSM coordinate accuracy vs YOLO bbox centre → `data/registry_accuracy.csv` |
| `scripts/validate_osm_only.py` | Cascade inference on OSM-only NE US pads → `data/osm_validated.csv`; feeds M4 routing pool |
| Streamlit Cloud deployment | Set env vars from `.env.example` |
| `merge_helipad_sources()` spatial dedup | Currently simple concat |

---

## Post-Training Analysis Scripts

### `scripts/compare_registry_accuracy.py`

Answers: *for matched FAA+OSM pairs, which registry coordinate is spatially closer to the YOLO-detected helipad centre?*

**Pipeline:**
1. Load `helipad_yolov8s.pt`, load `faa_adip_enriched.csv` + `osm_helipads_raw.csv`
2. Compute matches via `match_by_faa_id()` + `match_by_proximity()` (reuse `src/analysis.py`)
3. For each of the 747 test chips, run YOLO inference → get best bbox → call `bbox_px_to_latlon()` → `det_lat`, `det_lon`
4. For each detection that has a matched FAA+OSM pair: compute:
   - `dist_faa_m` = haversine(faa_lat, faa_lon, det_lat, det_lon)
   - `dist_osm_m` = haversine(osm_lat, osm_lon, det_lat, det_lon)
   - `faa_osm_dist_m` = haversine(faa_lat, faa_lon, osm_lat, osm_lon)
   - `winner` = "FAA" if dist_faa_m < dist_osm_m else "OSM"
   - `flag_different_pad` = True if faa_osm_dist_m > `--flag-dist` (default 50m)
5. Write `data/registry_accuracy.csv` — one row per matched pair with detection
6. Print summary: mean dist FAA→det, mean dist OSM→det, % FAA wins, % OSM wins, N flagged

**Non-obvious:**
- Chips where YOLO did not detect anything are excluded from the accuracy comparison (can't measure offset without a detection)
- `flag_different_pad` at 50m corresponds roughly to the Bell 206 medium FATO threshold — pairs beyond this may genuinely be two distinct pads (e.g., rooftop vs ground)
- The FAA ADIP coordinate (`adip_lat`/`adip_lon`) is preferred over the raw ArcGIS coordinate where `arp_method == 'SURVEYED'`; use the same logic as `load_faa_data()` auto-select

**Output columns:** `faa_ident`, `osm_id`, `match_method`, `faa_lat`, `faa_lon`, `osm_lat`, `osm_lon`, `det_lat`, `det_lon`, `dist_faa_m`, `dist_osm_m`, `faa_osm_dist_m`, `winner`, `flag_different_pad`

---

### `scripts/validate_osm_only.py`

Runs the trained YOLO cascade on NE US OSM helipads that have no FAA counterpart. Visually confirmed pads are added to the routing pool for M4.

**Pipeline:**
1. Load OSM + FAA data, compute matches, filter to OSM-only records (neither FAA-ID nor proximity match found)
2. For each OSM-only record: call `fetch_naip_chip(lat, lon)` — same USDA APFO logic as `build_yolo_dataset.py`; save to `data/osm_chips/<osm_id>.jpg`
3. Run `detect_helipad_cascade(image, yolo_model)` → `detected`, `confidence`, `bbox_px`
4. If detected: call `bbox_px_to_latlon()` → `hie_det_lat`, `hie_det_lon`; call `compute_offset_m()` → `hie_offset_m`
5. Write `data/osm_validated.csv`
6. Print: N OSM-only records, N with NAIP imagery, N visually confirmed, detection rate %

**Resume:** `--resume` flag skips records already in `data/osm_validated.csv` (checks by `osm_id`)

**Routing pool integration (M4):** `app.py` loads `osm_validated.csv` at startup; records where `hie_visual_detected=True` are shown in the OSM layer with a ✅ badge and are eligible as routing waypoints. Records where `hie_visual_detected=False` remain on the map but are excluded from routing.

**Output columns:** `osm_id`, `name`, `lat`, `lon`, `hie_visual_detected`, `hie_confidence`, `hie_det_lat`, `hie_det_lon`, `hie_offset_m`, `naip_status` (`ok` / `no_imagery` / `failed`)

---

## M4 Plan — NOTAM Airspace Avoidance + RainViewer Precipitation

**Do NOT start until M3 is submitted (23 Jun 2026).**

### `src/notam.py` — NOTAM + Aviation Weather

FAA Digital NOTAM API (`api.faa.gov`) requires `FAA_API_KEY` (already in `.env.example`). Header: `X-API-Key: <key>`.

```
GET https://api.faa.gov/notamSearch/api/v1/notams?icaoLocation=NK39&pageSize=100
GET https://api.faa.gov/notamSearch/api/v1/notams?locationLongitude=<lon>&locationLatitude=<lat>&locationRadius=<nm>
```

- FAA helipad IDENTs (e.g., `NK39`) are NOT always 4-letter ICAO codes. The API accepts 2–4-char FAA IDENTs directly via `icaoLocation`.
- For the NOTAM map layer: query by radius (25 nm) centred on each NE US cluster; merge and deduplicate by `notamId`.
- TFR NOTAMs (type `D`) have polygon geometry in `geometry.coordinates` — render as `folium.Polygon` with red outline.
- Point NOTAMs (no geometry) render as red circle markers at the referenced fix coordinate.

Aviation Weather Center METAR/TAF (no key required):
```
GET https://aviationweather.gov/api/data/metar?ids=<ICAO>&format=json
GET https://aviationweather.gov/api/data/taf?ids=<ICAO>&format=json
```

- Not every helipad has an ASOS station. Use the FAA IDENT directly first; fall back to the nearest station within 30 nm (query `?bbox=lat_min,lon_min,lat_max,lon_max`).
- Parse `flight_category` from METAR: `VFR` (green) / `MVFR` (yellow) / `IFR` (red) / `LIFR` (magenta).
- Cache responses for 5 minutes (`datetime.utcnow()` based TTL) — METAR updates every 20–60 min.

| Function | Description |
|----------|-------------|
| `fetch_notams_for_ident(ident, api_key)` | Active NOTAMs for one helipad IDENT |
| `fetch_notams_for_bbox(lat_min, lon_min, lat_max, lon_max, api_key)` | All active NOTAMs in bounding box (radius query per cluster) |
| `filter_active_notams(notams)` | Keep only NOTAMs whose effective window contains `datetime.utcnow()` |
| `notam_closes_airspace(notam)` | True if NOTAM is type D (TFR) or contains "airspace" classification |
| `fetch_metar(icao_id)` | METAR dict: `{wind_dir, wind_kt, visibility_sm, ceiling_ft, flight_category, raw}` |
| `fetch_taf(icao_id)` | TAF forecast periods list |
| `route_weather_summary(helipads, api_key)` | Per-helipad dict with `notams`, `metar`, `taf`; one-call-per-pad |

### `src/weather.py` — RainViewer Precipitation

RainViewer is free, no API key. Frame JSON updated every ~10 min.

```
GET https://api.rainviewer.com/public/weather-maps.json
```

Response structure:
```json
{"host": "https://tilecache.rainviewer.com",
 "radar": {"past": [{"time": 1748914800, "path": "/v2/radar/1748914800"}, ...],
           "nowcast": [{"time": ..., "path": ...}, ...]}}
```

Tile URL pattern: `{host}{path}/256/{z}/{x}/{y}/{color}/{options}.png`
- Color 2 = "universal blue" (best contrast on Folium/OSM basemap)
- Options `1_1` = smooth rendering + show snow

```python
# Route precipitation check — lat/lon → tile pixel → intensity
def _latlon_to_tile(lat, lon, zoom):  # → (x, y, pixel_x, pixel_y)
    # Standard slippy-map tile formula
    n = 2 ** zoom
    x_tile = int((lon + 180) / 360 * n)
    lat_rad = math.radians(lat)
    y_tile = int((1 - math.log(math.tan(lat_rad) + 1/math.cos(lat_rad)) / math.pi) / 2 * n)
    # pixel within 256×256 tile
    pixel_x = int((lon + 180) / 360 * n * 256) % 256
    pixel_y = int((1 - math.log(math.tan(lat_rad) + 1/math.cos(lat_rad)) / math.pi) / 2 * n * 256) % 256
    return x_tile, y_tile, pixel_x, pixel_y
```

Intensity threshold for route warning: pixel green channel > 50 at zoom=6 (covers ~600 km tile; enough resolution for NE US route legs).

| Function | Description |
|----------|-------------|
| `fetch_rainviewer_frames()` | Returns `{"past": [...], "nowcast": [...]}` with `{time, path}` per frame |
| `get_radar_tile_url(host, path, z, x, y, color=2)` | Tile URL string for `folium.TileLayer` |
| `sample_precipitation_at_latlon(lat, lon, host, path, zoom=6)` | Fetch tile PNG, sample pixel → intensity 0–255 |
| `check_route_precipitation(waypoints, host, path)` | Return list of `{lat, lon, label, intensity, warning}` |

### app.py additions

**RainViewer tile layer (all Folium maps):**
```python
# In build_map() and density/spider maps:
rv = fetch_rainviewer_frames()
latest = rv["radar"]["past"][-1]
radar_url = get_radar_tile_url(rv["host"], latest["path"], "{z}", "{x}", "{y}")
folium.TileLayer(
    tiles=radar_url,
    attr="RainViewer",
    name="Precipitation radar",
    overlay=True,
    control=True,
    opacity=0.6,
).add_to(m)
```

**NOTAM layer:**
```python
notam_group = folium.FeatureGroup(name="NOTAMs", show=False)
for n in active_notams:
    if notam_closes_airspace(n):
        # polygon TFR
        folium.Polygon(locations=..., color="red", fill=True, fill_opacity=0.15).add_to(notam_group)
    else:
        folium.CircleMarker(location=[lat, lon], radius=8, color="red", popup=n["text"]).add_to(notam_group)
notam_group.add_to(m)
```

**Per-helipad NOTAM badge in popup:** Fetch `fetch_notams_for_ident(ident, api_key)` at load time (cached); if any active closure NOTAM, append `🚫 <b>ACTIVE NOTAM</b>` to the existing popup HTML.

**Route METAR/TAF panel (routing simulator):** For each helipad leg, render a coloured badge:
```
🟢 VFR  NK39  Wind: 250°/12 kt  Vis: 10 SM  Ceiling: unlimited
🟡 MVFR  JRB  Wind: 180°/5 kt   Vis: 3 SM   Ceiling: 1500 ft
```

**Precipitation warning banner:** After route is computed, call `check_route_precipitation(waypoints, ...)`. If any waypoint `intensity > 50`: show `st.warning("⚠️ Precipitation detected along [leg name] — check latest radar.")` above the routing simulator.

### Non-obvious constraints

- `FAA_API_KEY`: register free at `https://api.faa.gov` → "Sign Up" → key delivered by email within minutes. The existing `.env.example` already has the placeholder.
- NOTAM API rate limit: 1000 req/hr on the free tier. For the full NE US cluster (~747 helipads) fetch by radius (25 nm circles centred on NYC/Boston/Philly) rather than one-per-IDENT to stay within limits.
- RainViewer tile server blocks requests without a Referer or User-Agent header. Set `headers={"User-Agent": "SkyRoute/1.0"}` in the `requests.get()` call inside `sample_precipitation_at_latlon`.
- Folium `TileLayer` with a dynamic URL (containing the radar timestamp) must be rebuilt every page load — do NOT use `@st.cache_data` on the map function when the radar layer is active, or the tile URL will be stale.
- METAR API returns an array; if the helipad IDENT has no station, the array is empty. Always check `if result` before accessing `result[0]`.
- No new packages needed: `requests`, `Pillow` (for PNG pixel sampling), and `folium` are all already in `requirements.txt`.
