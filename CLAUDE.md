# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> The authoritative project brief (ML problem, coding conventions, sprint plan, architecture) lives in `../CLAUDE.md`. Read that first. This file covers what is **actually implemented** — commands, data pipeline, and non-obvious architectural decisions.

> **Session log:** `Worklog.md` records every development session — what was built, issues hit, and how they were resolved. Read it to understand the history behind non-obvious decisions. **Update it at the end of every session before committing.**

---

## Commands

```bash
# Install dependencies (Python 3.11+)
pip install -r requirements.txt

# Run Streamlit dashboard (preferred — loads .env and checks API keys first)
.\run.bat        # Windows CMD
.\run.ps1        # PowerShell (may need: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
.\run.bat --check    # verify .env keys without starting the app

# Run directly (skips .env check)
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
├── app.py                  ← Streamlit dashboard (~4000 lines)
├── src/
│   ├── __init__.py
│   ├── data.py             ← Ingestion, cleaning, schema normalisation
│   ├── analysis.py         ← Cross-source matching, consistency, completeness
│   ├── notam.py            ← FAA NOTAM API + Aviation Weather Center METAR/TAF (M4)
│   ├── weather.py          ← NWS MRMS radar WMS layer + per-waypoint precipitation sampling (M4)
│   └── agent.py            ← LLM Route Assistant: routing, booking, geocoding, Mapillary (M4)
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
st.tabs(["📍 Problem", "📚 Literature", "🏪 Market", "📊 EDA & HIE", "🔍 Inspector", "📈 Results", "💬 Route Assistant"])
```

**Tab 1 — Problem**
- Persona card: Miles Urban, VP BD, Bronxville NY, age 44
- Journey comparison: without SkyRoute (92 min, 3 apps) vs with SkyRoute (36 min, 1 booking)
- 4 KPI metrics (time saved, trips/week, hours reclaimed, booking friction)
- Stakeholder ecosystem (4 cards: Operators, Passengers, Vertiport Owners, Regulators)
- HIE ML pipeline flow banner + 2-phase detail:
  - Phase 1: YOLO11m visual detection (NAIP satellite chip → bounding box)
  - Phase 2: ADIP structured scoring (XGBoost on 17 features; military/private flags; arrival coordination remarks decoded by LLM)

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

**Tab 7 — Route Assistant (💬)**
- Isolated via `@st.experimental_fragment` — chat messages don't trigger rebuilding of the 4 Folium maps in other tabs
- Natural language chat powered by `run_agent_v2()` — a **Level 1 agentic loop** (`src/agent.py`): model decides which tools to call; tools: `geocode`, `search_nearby_places`, `get_weather`, `compute_route`, `confirm_booking`
- **LLM:** Groq/`llama-3.3-70b-versatile`; auto-fallback to `llama-3.1-8b-instant` on quota/model error
- **Live thinking panel:** `st.status()` updated via `status_callback` parameter on `run_agent_v2()`; collapses when response is ready
- Geocoding: TomTom Fuzzy Search → LLM address extraction → Nominatim; handles business names and floor-level addresses
- **`search_nearby_places`:** TomTom POI Search + `categorySet` hard filter via `_TOMTOM_CATEGORY_MAP` (prevents off-category POI drift — e.g. "fine dining" → TomTom category `7315`)
- After route planning, user confirms with "yes" / "book it" / "book it now" → triggers `confirm_booking` tool
- **Booking flow** per leg (parallelised with `ThreadPoolExecutor`):
  - *Helicopter leg:* METAR badge (VFR/MVFR/IFR/LIFR colour pill + wind/visibility/ceiling) at top of each helipad expander; ADIP helipad info (status, ownership, coordination note decoded from raw FAA remarks via LLM), Mapillary street-level thumbnail
  - *Rideshare leg:* simulated Uber/Waymo fare + deeplinks, Mapillary thumbnails at pickup and dropoff
  - *Walk leg (< 0.5 km):* distance + walk-time card, Mapillary thumbnail at start
- **Precipitation warnings:** `check_route_precipitation()` called in `_execute_tool()` compute_route branch; `warn` severity → `result["precip_warnings"]` → `st.warning()` banner before route narrative; `avoid` severity → added to `advisory` text
- **Mapillary thumbnails:** image ID found server-side (`find_nearest_mapillary_image`), `thumb_2048_url` fetched via `get_mapillary_thumb_url`, rendered as `<img>` tag — no JS viewer, no CDN library, no WebGL, no CORS issues; "Open in Mapillary" link uses v4 `?pKey=` URL format (not v3 `?image_key=`)

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

Builds an HTML/JS page injected via `components.html()`. Layer-aware: `getAllHelipads()` calls `map.hasLayer(ly)` for each overlay layer and skips unchecked ones — routing only uses visible helipad layers. Seven overlays: FAA helipads, OSM helipads, HIE Validated, Business POIs, Executive Residences, TFRs, Precipitation radar. `HeliControl` and `MMControl` buttons are positioned `bottomright` to avoid overlap with the layer control.

Key JS routing functions (current as of m4.15):

| Function | Description |
|----------|-------------|
| `estimateDriveLeg(lat1,lng1,lat2,lng2)` | Instant drive estimate: haversine × 1.35 road factor, 25 km/h urban speed — replaces OSRM for short helipad approach legs |
| `osrmRoute(lat1,lng1,lat2,lng2,profile)` | TomTom → OSRM → haversine fallback chain; used only for the full origin→destination comparison route |
| `getAllHelipads()` | Returns flat array of `{lat,lon,name}` from all visible layers; result cached in `_helipadCache`, invalidated on `layeradd`/`layerremove` |
| `nearestHelipad(lat,lng,pts)` | O(N) haversine scan; called twice per route |
| `findRoute(start,end,allPts)` | Dijkstra TFR-avoiding arc; graph bounded to ±1.5°/2.5° corridor; skipped when direct line is TFR-free |
| `computeMultiModal(pA,pB)` | Orchestrator: sync helipad selection → browser yield → 1 OSRM call → draw + table |

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

## M4 Status — COMPLETE (2026-07-04)

### Completed deliverables

| Item | File | Notes |
|------|------|-------|
| OSM-only visual validation | `scripts/validate_osm_only.py` | 1,663 OSM-only records → 1,174 confirmed (70.6%); `data/osm_validated.csv` |
| TFR live feed | `src/notam.py` | FAA GeoServer WFS (same backend as tfr.faa.gov/tfr3/); ~230 live TFRs + stadium points; 15-min stale cache |
| METAR | `src/notam.py` | Aviation Weather Center, no key; 5-min in-process cache |
| NWS radar layer | `src/weather.py` | WMS `conus_bref_qcd` (MRMS); mode-specific thresholds |
| Precipitation sampling | `src/weather.py` | `sample_precipitation_at_latlon()` — 3×3 WMS GetMap, red channel 0–255 |
| TFR map overlay | `app.py` | `folium.Polygon` on main map + GeoJSON-injected layer in routing simulator |
| NWS radar overlay | `app.py` | `folium.WmsTileLayer` on all Folium maps; always-on |
| METAR badge in popups | `app.py` | Flight category colour-coded in FAA marker popup |
| Validated OSM routing pool | `app.py` | Inspector Mode C; `validatedLayer` in routing simulator |
| TFR arc routing | `app.py` | Dijkstra multi-hop avoidance (`findRoute`) when TFR layer ON; arcs via intermediate helipads |
| TomTom traffic-aware routing | `app.py` | TomTom Routing API v1 → OSRM fallback; daily quota guard |
| Mapbox traffic basemap | `app.py` | `traffic-day-v2` style as switchable basemap in routing simulator + Folium map; replaces deprecated v4 raster overlay |
| Routing performance | `app.py` | 3 OSRM calls → 1 (helipad legs replaced by `estimateDriveLeg`); Dijkstra corridor-bounded; `getAllHelipads()` cached; browser yield before sync work |
| Launcher scripts | `run.ps1`, `run.bat` | Load `.env`, report key status — safe to commit |
| **LLM Route Assistant** | `src/agent.py` | Groq/Llama; intent detection (route/book/off_topic); TomTom Fuzzy Search + LLM address extraction + Nominatim geocoding cascade; off-topic guard with negative few-shot examples |
| **Booking flow** | `src/agent.py` | Per-leg: helicopter (ADIP lookup + METAR badge + Mapillary), rideshare (Uber/Waymo deeplinks), walk (< 0.5 km threshold); Phase 1 now 6 parallel threads |
| **ADIP remarks decoding** | `src/agent.py` | `_decode_adip_remarks()` via Groq/Llama — cryptic FAA remarks → plain English coordination notes |
| **Mapillary street-level imagery** | `src/agent.py` + `app.py` | Server-side ID lookup (`find_nearest_mapillary_image`) + thumbnail fetch (`get_mapillary_thumb_url`); rendered as `<img>` — no JS viewer, no CORS |
| **TFR pre-booking check** | `src/agent.py` | `_check_tfrs_on_segment()` — samples 7 points on aerial leg; hard blocks (SECURITY/STADIUM/NDA_TFR/DEF) abort with error; soft TFRs shown as `st.warning()` banners in app before route narrative |

### Non-obvious decisions made during M4

**Mapbox traffic basemap (not overlay):** Mapbox deprecated the v4 raster tile endpoint (`/v4/mapbox.mapbox-traffic-v1/{z}/{x}/{y}.png`) — it renders orange road outlines without congestion colors. The correct approach is to use the `traffic-day-v2` Mapbox style as a switchable **basemap** (not an overlay), using the styles tile URL: `/styles/v1/mapbox/traffic-day-v2/tiles/256/{z}/{x}/{y}?access_token=...`. In the routing simulator this is added to the basemap radio group (`L.control.layers({'Street Map': osmDay, 'Satellite': esriSat, 'Traffic (Mapbox)': mapboxTrafficLayer}, ...)`). The TomTom Traffic Flow Tiles product (separate from the Routing API) requires paid activation on the free key — it was replaced entirely by Mapbox.

**Mapillary street-level imagery — no JS viewer:** The `mapillary-js@4` Viewer library requires WebGL and makes internal API calls that fail in Streamlit's sandboxed iframe (`sandbox="allow-scripts"` without `allow-same-origin`). The solution: (1) find the nearest image ID server-side via `find_nearest_mapillary_image()` using `Authorization: OAuth {token}` header (no CORS restriction in Python), (2) fetch the CDN JPEG thumbnail URL via `get_mapillary_thumb_url(image_id)` using `graph.mapillary.com/{id}?fields=thumb_2048_url`, (3) render as a plain `<img src="{thumb_url}">` tag — no JS library, no WebGL, no sandbox issues.

**ADIP remarks decoding:** Raw FAA coordination remarks (e.g. `FOR CD CTC NEW YORK APCH AT 516-683-2962.`) use cryptic abbreviations that passengers can't parse. `_decode_adip_remarks()` calls Groq/Llama with a system prompt listing common expansions (CD=Clearance Delivery, CTC=Contact, ATC=Air Traffic Control, etc.) and `max_tokens=150, temperature=0.1`. Results are cached in `_adip_remarks_cache` by ident and decoded text.

**Geocoding cascade:** Nominatim alone fails on business names (e.g. "Enigma Technologies at 32 Mercer St 8th Fl"). Fix: (1) TomTom Fuzzy Search as primary (`/search/2/search/{query}.json?key=...&lat=40.75&lon=-73.98&radius=200000`) handles business names natively, (2) if TomTom returns no result, LLM pre-pass via `_extract_address_with_llm()` strips the business name and floor to get a clean street address, then retry TomTom, (3) Nominatim as final fallback.

**Walk mode threshold:** Ground legs < 0.5 km use walk mode (`mode = "walk"`) instead of rideshare. Walk time computed at 5 km/h. The threshold is checked in `compute_skyroute()` before calling `simulate_rideshare()`.

**TFR data source:** FAA GeoServer WFS (`https://tfr.faa.gov/geoserver/TFR/ows`) was chosen over the FAA Digital NOTAM API because it requires no API key and returns ~230 live TFR polygons in one request. The endpoint was discovered by inspecting the tfr.faa.gov/tfr3 Nuxt SPA JS bundle (`geoWmsURL` in `__INITIAL_STATE__`). The GeoServer is the same backend the official TFR map uses.

**NWS over RainViewer:** NWS MRMS (`opengeo.ncep.noaa.gov/geoserver/conus/ows`) was chosen over RainViewer because: (1) it serves WMS tiles that render at all zoom levels without timestamp expiry; (2) `Access-Control-Allow-Origin: *` header allows client-side pixel sampling; (3) no API key or rate limit.

**TomTom over HERE:** TomTom requires no credit card for the free tier (2,500 calls/day). HERE requires a payment method even for free usage. TomTom Routing API v1 returns a points array directly — no FlexPolyline decoder needed (unlike HERE).

**TFR routing avoidance reuses Dijkstra:** Rather than a separate arc algorithm, the existing `findRoute()` Dijkstra in the JS routing template already hard-blocks edges whose 4 sample points fall inside TFR polygons. `findHeliRoute()` is a thin adapter (`lon`→`lng`) that calls `findRoute()` for the aerial leg.

**`@st.cache_data` and underscore parameters:** Streamlit excludes `_`-prefixed parameters from the cache key. `build_routing_html()` parameters `tomtom_key` and `js_v` must NOT have underscore prefixes, or the function always returns the first-cached HTML regardless of the key value.

**TomTom API key injection:** The key is inlined as a string literal (`'__TOMTOM_API_KEY__'` replaced at Python build time) rather than assigned to a JS variable before traffic layer code executes. This avoids variable hoisting failures in the iframe.

**Routing API call reduction:** `computeMultiModal` originally made 3 sequential `await osrmRoute()` calls (origin→dest, origin→padA, padB→dest). The first `await` was the only one that immediately yielded to the browser, so the "Computing routes…" spinner appeared only because it was first. After restructuring to do sync work first, the spinner stopped appearing because DOM mutations batch until the next `await`. Fix: `await new Promise(r => setTimeout(r, 0))` after setting the status bar forces a browser repaint before any blocking JS runs. The 2 helipad-leg calls were replaced by `estimateDriveLeg()` (haversine × 1.35 road factor, 25 km/h urban speed) — no OSRM request needed, and accuracy is within ±2 min for the typical 2–8 km helipad approach leg.

**Dijkstra corridor bounding:** `findRoute()` builds a graph of all visible helipads. With all layers on (~3000 nodes) and `RANGE_KM=555 km`, the graph is near-complete → O(N²) edge checks. A ±1.5°lat / ±2.5°lon bounding box filter around the route reduces active nodes to ~100–200 for a typical NE US route, cutting Dijkstra work by ~100×. The direct-path TFR check (4 sample points, sync) skips Dijkstra entirely when the straight line is already clear.

**`getAllHelipads()` caching:** The function iterated all Leaflet marker objects via `eachLayer()` on every route click (~3000 markers × 3 layers). A module-level `_helipadCache` array is invalidated on `layeradd`/`layerremove` map events — subsequent route clicks reuse the cached array.

**TFR pre-booking segment check:** `_check_tfrs_on_segment()` in `src/agent.py` calls `fetch_active_tfrs()` (15-min cached) and tests 7 evenly-spaced points along the pad_a → pad_b great-circle arc using a ray-casting point-in-polygon function. Stadium TFRs use a 3 nm (5.556 km) radius check instead of polygon containment since they are stored as points. Type codes in `_HARD_TFR_CODES = {"NDA_TFR", "DEF", "SECURITY", "STADIUM"}` block booking entirely; all other intersecting TFRs are returned as soft warnings rendered via `st.warning()` in the Route Assistant tab. The result dict gains a `tfr_warnings: list[str]` key; hard blocks set `result["error"]` and return early before the narrative formatter is called.

**Groq model configuration:** `llama-3.3-70b-versatile` and `meta-llama/llama-4-scout-17b-16e-instruct` are both project-blocked by default in Groq console. Working model as of June 2026: `llama-3.1-8b-instant` (set as `_GROQ_MODEL` primary). To use 70b: go to console.groq.com/settings/project/limits and enable it, then set `_GROQ_MODEL = "llama-3.3-70b-versatile"`. The LLM exception fallback in `extract_nav_params()` now returns `_FALLBACK_PARAMS` (with `destination_text: None`) on any API error — this triggers the destination guard and returns the routing redirect, rather than the previous bug where `user_text` was used as `destination_text`.

**`run_agent_v2()` tool-calling loop:** Replaced the single-pass `extract_nav_params()` + hardcoded routing with a `while iterations < 8` loop that gives the 70b model full autonomy to call tools and chain results. The key design choice is that routing internals (TFR check, METAR, ADIP status) are never exposed as user-facing tools — they run inside `compute_route` and `confirm_booking` and surface only as plain-English advisory text. This keeps the tool surface minimal and prevents the model from over-engineering multi-tool chains for simple route queries.

**Fragment isolation — Route Assistant tab:** The Route Assistant was previously a regular `with tab_agent:` block. Every chat message triggered a full Streamlit rerun, which rebuilt all 4 Folium maps + routing HTML (~2-3s). Wrapping the Route Assistant in `@st.experimental_fragment` isolates it — only the chat fragment reruns on message submission. The helipad pool is built from `load_data()` (which is `@st.cache_data`) on first call and stored in `st.session_state["_agent_helipads"]`. Do not close over outer-scope variables inside a fragment; they become stale on fragment-only reruns.

**`status_callback` pattern for `st.status()`:** The agentic loop has no way to directly call Streamlit from `src/agent.py` (no Streamlit dependency there). The callback indirection keeps the agent logic pure Python while letting `app.py` decide how to present progress. The callback fires at three hooks: `("thinking", {"iteration": N})` before each Groq call, `("tool_call", {"name": ..., "args": ...})` before each tool execution, and `("tool_result", {"name": ..., "result": ...})` after. The `st.status()` context manager auto-collapses the panel when `.update(state="complete")` is called.

**POI search `categorySet` filter:** TomTom's keyword `poiSearch` endpoint matches any business whose description contains the query term. Searching "fine dining" previously returned liquor stores and sandwich shops because their descriptions mention dining. The fix: `_TOMTOM_CATEGORY_MAP` maps user-facing terms to TomTom structured category codes (e.g., "fine dining" → `7315`, "steakhouse" → `7315002`, "coffee" → `9361065`). When a mapping exists, `categorySet=<code>` is added to the API request — this is a server-side hard filter, not a client-side re-rank. The keyword in the URL still provides relevance sorting within the category.

**Parallel booking with `ThreadPoolExecutor`:** `run_booking()` originally made ~28 serial HTTP calls (ADIP POST + Mapillary search + Mapillary thumbnail per leg endpoint). These are now split into two parallel phases: Phase 1 — 6 concurrent fetches (departure ADIP, arrival ADIP, departure Mapillary ID, arrival Mapillary ID, departure METAR, arrival METAR); Phase 2 — 2 concurrent thumbnail URL fetches. Total latency ~3s instead of ~12s. The `ThreadPoolExecutor` is created inside `run_booking()` (not at module level) so it doesn't leak threads if the function raises.

**Mapillary URL v4 format:** The Mapillary Graph API returns numeric image IDs (v4 format). Opening these with the legacy v3 URL `?image_key=<id>` loads the worldwide map, not the specific image. The correct v4 URL is `?pKey=<id>`. One-line fix in `_mly_viewer_html()` in `app.py`.

**TFR time-awareness — US Eastern departure time:** `_aerial_departure_dt()` computes the estimated UTC wall-clock time when the helicopter leg departs. If the user specified an arrival time (e.g. "get me there by 16:00"), that time is interpreted as **US Eastern** (via `zoneinfo.ZoneInfo("America/New_York")`) regardless of what timezone the server is in. This is correct for the NYC metro use-case and ensures DST is handled correctly. `tzdata>=2024.1` in `requirements.txt` provides the IANA timezone database on Windows and Streamlit Cloud (both lack it by default). `_check_tfrs_on_segment()` receives the computed `departure_dt` and skips TFRs whose `expires_utc` is before that time — expired TFRs are silently filtered, not counted as hard blocks.

**HIE validated layer ADIP fallback:** The `validatedLayer` GeoJSON in the routing simulator is built from two sources: (1) FAA helipads with `gt==1` in `inspector_results.csv` (YOLO visually confirmed), and (2) FAA helipads where `operational==1` AND `data_freshness_days<=365` even if YOLO found no visual marking. This recovers Type B false negatives — operational helipads with no painted H in NAIP imagery (grass fields, faded markings, post-NAIP-acquisition construction). The 365-day threshold and `fillna(9999)` guard are in `app.py` around the `_val_idents` construction block.

**METAR badge `_metar_badge()` helper:** Defined as a module-level function in `app.py` just before `@_fragment def _route_assistant_content()`. Returns an HTML string with a colour-coded `<span>` pill (green=VFR, blue=MVFR, red=IFR, purple=LIFR) followed by wind/visibility/ceiling inline text. Returns empty string when METAR is None — callers check `if _dep_badge:` before calling `st.markdown(..., unsafe_allow_html=True)`. The `metar_dep` / `metar_arr` keys are in the booking leg dict (set by `run_booking()` Phase 1 via `_fetch_metar`).

**Routing simulator zoom control clipping:** Leaflet's default zoom control is at `topleft`. The layer control panel is at `topright`. When the layer control panel expands (many layers), on some display sizes the top-left area of the iframe becomes crowded and scrolling clips the zoom buttons. Fix: `zoomControl: false` in `L.map()` init, then `L.control.zoom({position: 'bottomleft'}).addTo(map)`. `bottomleft` has no other controls, so there is nothing to compete with.

### Remaining Final items

- [x] Route METAR/TAF panel — per-leg wind/visibility/ceiling badges; VFR/MVFR/IFR colour coding (done 2026-07-04)
- [x] Precipitation warning banner — NWS per-waypoint intensity check; `st.warning()` banner when intensity > mode threshold (done 2026-07-04)
- [x] Weather card raw-cache + unit toggle fix — `_render_weather_card()`; NWS cache stores imperial raw; `_format_weather()` converts at read time (done 2026-07-11)
- [x] POI card Yelp enrichment — `_yelp_enrich()`, parallel `ThreadPoolExecutor`, star rating/price/Yelp link; TomTom `openingHours` added (done 2026-07-11)
- [x] `location_hint` injected into LLM system prompt for follow-up place/weather queries (done 2026-07-11)
- [x] `_naip_chip_b64` live NAIP fallback + 280px resolution — fetches USDA APFO when chip not on disk (done 2026-07-11)
- [x] Walk leg dropoff Mapillary fields + 2-column detailed itinerary (start + destination side-by-side) (done 2026-07-11)
- [x] Itinerary height fix — 225px per helicopter leg, 150px bottom buffer (done 2026-07-11)
- [x] URL protocol guard — TomTom/Yelp protocol-less URLs prepended with `https://` (done 2026-07-11)
- [ ] `scripts/compare_registry_accuracy.py` — FAA vs OSM coordinate accuracy vs YOLO bbox centre

### Final milestone checklist (21 Jul 2026 — Demo Day)

- [x] METAR per-leg badges + precipitation warnings (done 2026-07-04)
- [x] OSM helipad address lookup + 8B model recovery + fragment rerun fix (done 2026-07-04)
- [x] Mia intro card + tab-switch audio (done 2026-07-05)
- [x] Drive-only time estimate fix — distance-tiered speed (done 2026-07-05)
- [x] Booking pickup label fix — resolved POI name + address (done 2026-07-05)
- [x] METAR ICAO lookup fix — pre-resolve FAA ident → ICAO_ID (done 2026-07-05)
- [x] Weather card unit-toggle fix + raw NWS cache (done 2026-07-11)
- [x] Yelp Fusion enrichment for POI cards + parallel enrichment + price/rating display (done 2026-07-11)
- [x] `location_hint` in `run_agent_v2` — last route origin injected into LLM context (done 2026-07-11)
- [x] `_naip_chip_b64` live NAIP fallback + 280px chip resolution (done 2026-07-11)
- [x] Walk leg dropoff Mapillary + 2-column detailed itinerary view (done 2026-07-11)
- [x] Itinerary height formula fix (done 2026-07-11)
- [ ] `scripts/compare_registry_accuracy.py`
- [ ] End-to-end demo run: Miles Urban persona, NYC → Greenwich CT, live TFR + weather layers, multimodal route with aerial advantage callout
- [ ] `Worklog.md` updated with Final session notes

### Post-M4 fixes (2026-07-04)

| Fix | File | Detail |
|-----|------|--------|
| OSM helipad reverse geocoding | `src/agent.py` | `_reverse_geocode_tomtom(lat, lon)` — TomTom reverse geocode called in Phase 1 parallel block for OSM-only helipads; `address` field attached to `dep_info`/`arr_info`; shown above coordinates in booking card |
| 8B model tool-call recovery | `src/agent.py` | `_recover_tool_use_failed()` — detects Groq 400 `tool_use_failed`, parses `<function=name>{args}` from `failed_generation`, executes tool via `_execute_tool()`, injects synthetic tool-result turn, asks 8B to format as text; handles both fallback path (70b blocked → 8b) and direct 8b failure |
| Route Assistant rerun fix | `app.py` | `st.rerun()` inside `@st.experimental_fragment` triggers full app rerun (rebuilds all Folium maps) on Streamlit 1.36; guarded to fire only when `_result.get("booking_legs")` is non-empty — eliminates grayout on weather/routing/general queries |

#### Non-obvious decisions

**Why `_recover_tool_use_failed` parses `failed_generation`:** The 8B model does generate the right tool and right arguments — it just uses `<function=compute_route>{"origin":…}` format instead of the OpenAI schema. The `failed_generation` field in the Groq 400 error body contains the exact attempted call. Parsing and re-executing it is more reliable than retrying the model with a modified prompt.

**Why `st.rerun()` only on booking:** Both user and assistant messages are rendered inline during the fragment's current pass (not only from session_state at the top). Only booking leg cards need a second pass because they render from `st.session_state["_agent_booking_legs"]` at the top of the fragment. Guarding the rerun eliminates a ~2-3s grayout on every non-booking query.

### Post-Final fixes (2026-07-05)

| Fix | File | Detail |
|-----|------|--------|
| Mia intro card | `app.py` | `_build_mia_card()` — self-contained `components.html()` with base64 PNG + MP3; ResizeObserver triggers audio play on tab reveal; `assets/mia.png` (PIL-resized to 426×426, 255 KB) + `assets/mia_intro.mp3` (116 KB) committed to git |
| Drive-only time estimate | `src/agent.py` | Distance-tiered speed replaces flat 25 km/h: `<20 km → 30 km/h`, `20–60 km → 55 km/h`, `>60 km → 70 km/h`. Short helipad approach legs (origin→pad_a, pad_b→dest) still use 25 km/h via `_SPEED_DRIVE_KMH`. |
| Booking pickup/dropoff labels | `src/agent.py` + `app.py` | `_execute_tool()` injects `origin_poi_name`, `origin_address`, `dest_poi_name`, `dest_address` from `_geocode_rich_cache` into the route dict; `run_booking()` uses these for the ground leg `from_name`/`to_name` labels |
| METAR ICAO lookup | `src/agent.py` | `_icao_for(faa_id)` helper pre-resolves `ICAO_ID` from `faa_adip_df["ICAO_ID"]` before the Phase 1 parallel block; Aviation Weather Center requires ICAO codes (e.g. `KJFK`), not FAA idents (e.g. `NK39`) |
| Route map conditional | `app.py` | Route map rendered only when `_result` contains a non-error route or booking; prevents the map from appearing on weather/restaurant/general queries |
| Booking legs snapshot+clear | `app.py` | After rendering, `st.session_state["_agent_booking_legs"]` is snapshot-then-cleared so cards do not re-render on the next non-booking query |

#### Non-obvious decisions

**Mia audio: ResizeObserver vs `window.onload`:** `window.onload` fires when the iframe loads at page startup — before the user has clicked the Route Assistant tab. Browsers block autoplay unless a user gesture immediately preceded the play call. `ResizeObserver` on `document.body` fires when Streamlit changes the parent container's CSS from `display:none` to visible (tab switch), and the tab click is the qualifying user gesture. Pattern:
```js
var ro = new ResizeObserver(function(entries) {
    if (!played && entries[0] && entries[0].contentRect.width > 0) {
        played = true; ro.disconnect();
        document.getElementById("mia-a").play().catch(function(){});
    }
});
ro.observe(document.body);
```

**`components.html()` size limit:** Streamlit Cloud silently fails to render a `components.html()` component whose content exceeds ~2 MB (the base64 string grows ~37% vs raw). `mia.png` was originally 2 MB (1254×1254 px). Fix: PIL-resize to 426×426 before base64-encoding → 255 KB on disk, ~350 KB base64. Always pre-compress images before embedding as data URIs in `components.html()`.

**Distance-tiered drive speed:** The previous flat `25 km/h` urban speed produced `321 min` for a 99 km haversine route (NYC → New Haven), whereas OSRM returns ~112 min. The tiered speeds (`30 / 55 / 70 km/h` for short/medium/long haversine distances) are calibrated so that `haversine × 1.35 road_factor / speed` approximates OSRM times across the NE US trip range. The 25 km/h constant is preserved for short helipad approach legs (2–8 km) where urban crawl applies.

**ICAO vs FAA ident for METAR:** The Aviation Weather Center (`aviationweather.gov/api/data/metar`) indexes stations by ICAO code. Most helipads have only FAA idents (e.g. `NK39`). `faa_adip_df["ICAO_ID"]` has the mapping when it exists. `_icao_for()` falls back to the raw FAA ident so existing helipads that happen to have an ICAO code still get METAR data; helipads without an ICAO code (private, small rooftops) return `None` from `fetch_metar()` which is correctly handled as "no weather data available".

**Booking label `_geocode_rich_cache`:** The cache is keyed by `place_name.lower().strip()` and stores `{poi_name, address}` from TomTom's response. Coordinates flow separately through `geocode_place()` return value — the cache only carries the display strings. `_execute_tool()` reads the cache for both origin and destination immediately after geocoding, then injects the strings into `route["origin_poi_name"]` etc. before calling `compute_skyroute()`. This must happen before `run_booking()` consumes the route dict, not after.

### v5.0 Post-M4 fixes (2026-07-11)

| Fix | File | Detail |
|-----|------|--------|
| Weather card raw NWS cache | `app.py` | `_nws_cache` stores imperial raw dict (`temperature_f`, `temperature_c`, `wind_raw`, `periods_raw`); `_format_weather(raw, metric, units)` converts at read time — fixes unit-toggle returning wrong temperature on cache hit |
| POI card Yelp enrichment | `src/agent.py` + `app.py` | `_yelp_enrich(name, lat, lon)` calls Yelp Fusion `/v3/businesses/search`; `_yelp_cache` keyed by `(name_lower, lat4, lon4)`; parallel fetch via `ThreadPoolExecutor`; `_render_places_card()` shows star rating, review count, price pill, Yelp link; TomTom `openingHours=nextSevenDays` added for today's hours |
| URL protocol guard | `src/agent.py` + `app.py` | TomTom and Yelp return protocol-less URLs (`www.example.com`); prepend `https://` when not starting with `http://` or `https://` |
| `location_hint` injection | `src/agent.py` | `run_agent_v2(location_hint=...)` appends last route origin/destination to `_CONCIERGE_SYSTEM` prompt; follow-up place/weather queries default to last known area |
| `_naip_chip_b64` live fallback | `app.py` | When chip file not on disk and `lat`/`lon` provided, calls `fetch_naip_chip(lat, lon)` from `src.hie`; result cached by Streamlit; resolution raised from 100×100 to 280×280 px |
| Walk leg dropoff Mapillary | `src/agent.py` + `app.py` | `run_booking()` stores `dropoff_mly_id`/`dropoff_mly_thumb` in walk leg dict; detailed itinerary shows 2-column layout (start left, destination right) |
| Itinerary height formula | `app.py` | `_per_leg_h`: 225 px per helicopter leg (was 165), 118 px per ground leg; bottom buffer 150 px (was 110); POI card per-row dynamic height with Yelp/hours rows |

#### Non-obvious decisions

**Yelp parallel enrichment:** `ThreadPoolExecutor(max_workers=len(base))` runs all Yelp calls concurrently; total latency ≈ slowest single call (~1–3 s) instead of N×5 s sequential. Worker count is bounded by the result set size (default TomTom limit 5).

**NWS raw cache:** The previous cache stored already-converted strings, so switching from Fahrenheit to Celsius on a cache hit returned the original unit. Fix: always cache the raw imperial values; `_format_weather(raw, metric, units)` converts at call time. The cache key is `(lat4, lon4)` only — units are not part of the key.

**`_naip_chip_b64` cache with lat/lon args:** `@st.cache_data` uses all non-underscore parameters as cache key. Adding `lat` and `lon` as named parameters (not prefixed with `_`) means the live-fetch result is cached per location — subsequent renders of the same helipad are instant. Raising resolution to 280 px costs ~5 KB per chip in Streamlit's cache (negligible).

**Walk leg 2-column layout:** The previous walk detailed itinerary showed a single Mapillary card for the pickup (helipad), which confused users expecting to see the destination. The 2-column layout (`st.columns(2)`) shows start on the left and destination on the right, matching the itinerary summary card's directional flow.

---

## Streamlit Cloud Deployment

**Live URL:** https://skyroute.streamlit.app

### packages.txt (apt system libraries)

```
libgl1
libglib2.0-0t64
```

- `libgl1` — provides `libGL.so.1` required by `opencv-python` (full version that ultralytics installs alongside headless)
- `libglib2.0-0t64` — provides `libgthread-2.0.so.0` required by `cv2` native module. On Debian **trixie** the package was renamed from `libglib2.0-0` → `libglib2.0-0t64` as part of the 64-bit time_t transition. The old name pulls in `libffi7` which is not available on trixie and aborts apt entirely — always use `libglib2.0-0t64`.

### Python dependency constraints (requirements.txt)

- `numpy>=1.26.0,<2.0.0` — `torch==2.0.1` was compiled against NumPy 1.x ABI; `numpy>=2.0` breaks `torch.from_numpy` with `RuntimeError: Numpy is not available`
- `opencv-python-headless>=4.9.0,<4.11.0` — ultralytics also installs `opencv-python` (full), so both end up installed; `libgl1` + `libglib2.0-0t64` cover the missing shared libraries for both

### Secrets management

Streamlit Community Cloud exposes every secret from the dashboard as an OS environment variable automatically. All `os.getenv("KEY")` calls in the codebase work without any extra code. Add secrets at: **App settings → Secrets** in TOML format:

```toml
GROQ_API_KEY = "gsk_..."
TOMTOM_API_KEY = "..."
MAPBOX_TOKEN = "pk.eyJ1..."
MAPILLARY_TOKEN = "MLY|..."
FAA_API_KEY = "..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

**CRITICAL — never call `st.secrets` at module level.** Streamlit raises "Tried to use SessionInfo before it was initialized" if `st.secrets` is accessed outside a valid script run context (e.g. during import, in a `@st.cache_resource` function body, or in a background thread). Always guard with:
```python
from streamlit.runtime.scriptrunner import get_script_run_ctx
if get_script_run_ctx() is not None:
    key = st.secrets.get("MY_KEY", "")
```
Or better: rely on `os.getenv()` alone — Streamlit Cloud already put the secret there.

### Data files committed for Cloud (gitignore exceptions)

The `data/` directory is gitignored, but two paths are committed so the Inspector tab works without a local data pipeline:

| Path | Size | Purpose |
|------|------|---------|
| `data/inspector_results.csv` | ~60 KB | Pre-computed TP/TN/FP/FN for all 747 test helipads — fast-path for `_get_test_results()` |
| `data/yolo_dataset/images/test/` | 40.5 MB (747 JPEGs) | NAIP chips for the Inspector's chip viewer |

`.gitignore` uses `!data/inspector_results.csv` and replaces the blanket `data/yolo_dataset/` rule with per-subdirectory exclusions (train/, val/, labels/, review_*/ excluded; test/ kept). Do not re-add `data/yolo_dataset/` as a blanket rule — it would override the exceptions.

### Deployment pitfall history

| Error | Root cause | Fix |
|-------|-----------|-----|
| `libffi7` not installable | `libglib2.0-0` (bullseye) in packages.txt | Replace with `libglib2.0-0t64` (trixie) |
| `libGL.so.1` not found | `libgl1` missing | Add `libgl1` to packages.txt |
| `libgthread-2.0.so.0` not found | `libglib2.0-0t64` missing | Add `libglib2.0-0t64` to packages.txt |
| `RuntimeError: Numpy is not available` | numpy≥2.0 installed; torch 2.0.1 needs 1.x | Pin `numpy>=1.26.0,<2.0.0` in requirements.txt |
| `cannot import name 'bbox_px_to_bounds'` | Upstream symptom of libGL/libgthread failure | Fix the library, not the import |
| `FileNotFoundError: faa_adip_enriched.csv` | data/ gitignored | `_get_test_results()` returns empty DataFrame when `_TEST_IMG_DIR` missing |
| `Bad message format: SessionInfo before initialized` | `st.secrets` called outside script run context | Guard with `get_script_run_ctx() is not None` before any `st.secrets` access |
