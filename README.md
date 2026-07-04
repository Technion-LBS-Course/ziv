# ✈️ SkyRoute — Advanced Air Mobility Navigation Platform

> *"Door-to-sky-to-door. Your fastest path, elevated."*

🌐 **Live demo:** https://skyroute.streamlit.app

---

## One-Liner

**Business travelers in dense U.S. metros** suffer from fragmented urban mobility and hours lost to traffic, with no platform combining Advanced Air Mobility (AAM) into seamless door-to-door journeys — so we're building **SkyRoute**, an operator-agnostic multimodal routing and booking platform that uses a **ML-powered Helipad Intelligence Engine (HIE)** to validate, score, and route via verified air+ground itineraries across the New York metro and beyond.

---

## The Problem

Business travelers and frequent flyers lose hours daily to urban traffic congestion, long airport transfers, and the friction between transport modes. In a world where minutes equal billable hours, no single platform today enables seamless planning and booking of eVTOL aircraft, helicopters, and ground transport as a unified, door-to-door journey.

Existing solutions are siloed: Blade books helicopters but lacks multimodal integration; Google Maps offers ground routing but ignores AAM entirely; Joby and Volocopter serve their own operators only. There is no operator-agnostic, business-grade platform that fuses all modes.

Beyond routing, the underlying infrastructure data is broken. Public helipad databases (FAA, OSM) are notoriously inconsistent — outdated coordinates, decommissioned pads listed as active, missing usability metadata. Feeding this raw data into a routing engine creates liability and trust failures.

**SkyRoute solves both problems:** the routing layer and the data quality layer beneath it.

---

## Target User

**Persona:** Miles Urban, 44, VP of Business Development at a Manhattan-based financial services firm. Lives in Bronxville, NY. Travels 4–5 times per week across the New York metro area — Midtown Manhattan, JFK/EWR, Jersey City, Greenwich CT — for client meetings and board sessions. Every wasted hour is a billable hour lost; he holds a corporate Amex with no travel cap.

> *"I don't need cheaper. I need faster and reliable."*

**Primary Use Case:** Miles has a 9:30 AM board meeting in Greenwich, CT and is leaving his Midtown office at 8:45 AM. Ground traffic makes it 75+ minutes by car. He opens SkyRoute, enters origin and destination, and receives: 6-minute walk to the verified 30th Street Heliport → 18-minute helicopter to Westchester County Airport → 12-minute car to the client's office. One booking, one payment, live tracking. Total: 36 minutes. He lands with time to spare.

| | Without SkyRoute | With SkyRoute |
|--|--|--|
| Mode | 3 apps · ground only | 1 booking · air + ground |
| Time | ~92 min | ~36 min |
| Saving | — | **56 min / trip · 3.7 hrs / week** |

---

## Helipad Intelligence Engine (HIE) — ML Architecture

Raw helipad databases (FAA, OSM) are incomplete, stale, and contain military or decommissioned pads. HIE is a **2-phase ML pipeline** that validates every candidate pad before it enters the routing engine.

```
Raw Input              Phase 1                              Phase 2 ⭐              Output
FAA · OSM  →  YOLO11m visual detection          →  ADIP status gate         →  ✅ Validated pad
747+ records   cascade: CV → YOLO11m fine-tuned     operational==1 +            added to routing
                                                     freshness ≤ 365 days
```

### Why Two Phases?

Visual detection alone cannot validate all helipads. A significant fraction of real, operational pads are **visually invisible in NAIP imagery**: grass/turf fields with only a windsock, rooftops with no painted marking, pads built after the most recent NAIP acquisition (2–3 year update cycle), or faded private estate pads. For these, YOLO correctly fires nothing — not because the pad is decommissioned, but because there is no visual evidence at 0.156 m/px resolution.

This means YOLO Recall < 1.0 is a property of the imagery, not a model failure. The false negatives (FN) from Phase 1 fall into two distinct categories:

| FN type | Cause | Correct handling |
|---------|-------|-----------------|
| **Type A** — stale record | Registry still lists a decommissioned pad; nothing is there | Exclude via Phase 2 structured scoring |
| **Type B** — invisible helipad | Operational pad with no visual marker; YOLO misses it | Recover via Phase 2 ADIP status fields |

Phase 1 cannot distinguish these two cases on its own. Phase 2 uses independent structured evidence to recover Type B cases and confirm Type A exclusions:

| Phase | Signal | Handles |
|-------|--------|---------|
| **Phase 1 — Visual** | NAIP imagery chip + YOLO11m (production) / YOLO11s (discovery mode) | Visually marked pads (H, cross, circle, rooftop paint) |
| **Phase 2 — Structured** | FAA ADIP `operational==1` + `data_freshness_days ≤ 365`, military/private flags | Named pads regardless of visual marker; military/private/closure filtering; stale-record detection |

The final routing pool blends both phases — it is **not** a hard gate on visual detection alone. A hospital rooftop helipad with no painted H but a current ADIP operational status and recent inspection will still be included even if Phase 1 returns no detection.

### Phase 1 — Visual Validation (YOLO11m fine-tuned cascade)

USDA NAIP imagery chips (100 m × 100 m, 640×640 px, 0.156 m/px) are fetched for each candidate coordinate from the USDA APFO ImageServer. **YOLO11m fine-tuned** (`detect_yolo`) is the production model — domain-specific, trained on 2,584 NAIP chips from across the continental US. Detects H-markers and rooftop pads that lack an explicit H marking. **YOLO11m** (P=0.931, FP=27) is preferred for safety-critical routing; YOLO11s is available for discovery workflows that prioritise recall (R=0.866, TP=375).

The detected bounding-box centroid is back-projected to geographic coordinates and compared against the registry coordinate to produce `offset_m`. Records where no helipad is detected, or where offset exceeds a design-class threshold, are flagged for manual review.

**Fine-tuned model comparison — 747 NE US held-out test chips, identical evaluation pipeline:**

| Method | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| XGBoost structured baseline | 0.74 | 0.72 | 0.73 | Registry metadata only, no imagery — see Structured Scoring below |
| YOLOv8s fine-tuned | 0.906 | 0.801 | 0.850 | CNN baseline — establishes NAIP fine-tuning contribution |
| RT-DETR-L fine-tuned | 0.907 | 0.815 | 0.859 | Transformer detector; improves recall vs YOLOv8s; training instability at epoch 32 |
| YOLO11s fine-tuned | 0.908 | **0.866** | 0.887 | Best recall — 28 more TP than YOLOv8s at only +2 FP |
| **YOLO11m fine-tuned (production)** | **0.931** | 0.848 | **0.888** | **Best precision & accuracy; 11 fewer FP than YOLO11s — preferred for safety-critical routing** |

All four fine-tuned YOLO models substantially outperform the XGBoost structured baseline, confirming that aerial imagery adds decisive signal beyond what registry metadata alone can provide. YOLO11m is the **production model** (highest precision, fewest false positives); YOLO11s is retained for registry cleaning sweeps where recall matters more.

### Phase 2 — Structured Status Validation (ADIP threshold)

For every candidate pad, FAA ADIP structured fields provide authoritative status signals that NAIP imagery cannot. Phase 2 applies a threshold gate: a pad enters the routing pool if `operational == 1` (ADIP status is Operational) **and** `data_freshness_days ≤ 365` (registry updated within the past year). Pads with `MIL_CODE` set (Army / Navy / Air Force / Marines / Coast Guard) and pads with `PRIVATEUSE` flag are excluded from civilian routing regardless of visual detection result. The XGBoost classifier (`scripts/train_xgboost.py`, F1=0.73) is a research baseline that quantifies how much discriminative signal is available from structured fields alone — it is not called in the live routing pipeline.

The booking flow in `src/agent.py` applies the same flags at runtime: military and private-use helipads generate coordination warnings before a leg is confirmed. Cryptic FAA coordination remarks (e.g. `FOR CD CTC NEW YORK APCH AT 516-683-2962`) are decoded to plain English by an LLM call (`_decode_adip_remarks()`) so passengers see actionable arrival instructions, not raw FAA notation.

For validated FAA helipads the ADIP record also provides TLOF/FATO dimensions, design category, last inspection date, ATC contact frequencies, EV charging availability, and ingress/egress bearings — feeding the routing engine's arrival planning layer.

---

## Data Sources & Data Card

### FAA ADDS-ArcGIS + ADIP (primary)
- **Fetch:** `python scripts/fetch_ny_data.py` → `data/faa_helipads_raw.csv` (~747 records, NE US: NY NJ CT PA MA)
- **Enrich:** `python scripts/fetch_adip_details.py` → `data/faa_adip_enriched.csv` (+23 ADIP columns)
- **Key columns after enrichment:** `IDENT`, `NAME`, `lat`, `lon`, `STATE`, `ELEVATION`, `PRIVATEUSE`, `MIL_CODE`, `adip_status`, `last_info_days_ago`, `adip_lat`, `adip_lon`
- **License:** Public Domain (U.S. Government data)
- **Known gaps:** All 747 records have `ESTIMATED` (not surveyed) ARP coordinates; ~30% missing usability metadata

### OpenStreetMap aeroway=helipad (secondary)
- **Fetch:** `python scripts/fetch_ny_data.py` → `data/osm_helipads_raw.csv` (Overpass API, same 5-state bounding box)
- **Key columns:** `lat`, `lon`, `name`, `faa`, `ele`, `surface`, `lit`, `operator`, `addr:state`
- **License:** ODbL
- **Known gaps:** ~24% have `faa=<IDENT>` tag; most lack `ele`, `surface`, and `lit`; coordinates are community-contributed and may drift

### USDA NAIP — National Agriculture Imagery Program (M3 imagery)
- **Source:** USDA Farm Service Agency, served via APFO ImageServer
- **Endpoint:** `https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/exportImage`
- **Native resolution:** 1 m/px (standard acquisitions); 0.6 m/px (60 cm, enhanced acquisitions for select states from 2018 onward)
- **Effective GSD in pipeline:** 0.156 m/px — the APFO server is queried for a 640×640 px export of a 100 m × 100 m window; this upsamples from the 1 m native to our chip resolution
- **Expected horizontal accuracy:** ≤6 m (NMAS Class 1, 90th-percentile confidence) per USDA FSA specification; newer acquisitions target ≤3 m
- **Coverage:** CONUS only (lat 24°–49.5°N, lon 66°–125°W) — Alaska, Hawaii, Puerto Rico have no coverage; ~300 HelipadCAT records fall outside this envelope and are excluded from the dataset
- **Bands:** 4-band (RGB + NIR); pipeline uses RGB channels only
- **Update cycle:** 2–3 years per state; exported chips reflect the most recent available mosaic at query time
- **Auth:** None (public HTTP GET, no API key required)
- **License:** Public Domain (U.S. Government data)
- **Use in project:** 4,027 training/evaluation chips (100 m × 100 m per helipad coordinate) — 2,584 train + 696 val + 747 test

### HelipadCAT (M3 training coordinates + annotations)
- **Source:** `github.com/jonasbtn/helipad_detection` (Jonas Bøttiger et al.)
- **Content:** ~6,000 FAA NASR-derived helipad coordinates across the continental US, with per-record bounding box annotations and visual confirmation labels
- **Key field:** `groundtruth` (True / False) — True = helipad visually confirmed present; False = FAA-listed location where authors confirmed no helipad is visible (decommissioned / stale entry)
- **Does NOT include image chips** — HelipadCAT ships coordinates and annotation metadata only; imagery must be fetched at runtime. The original authors used Google Maps Static API zoom-20 tiles (~0.114 m/px at 40°N). No pre-downloaded images are distributed with the dataset.
- **Why original imagery was not used:**
  1. Google Maps Static API has commercial-use Terms of Service restrictions
  2. Domain shift: a model trained on Google Maps tiles and evaluated on NAIP would face a sensor/resolution mismatch that inflates zero-shot failure
- **Re-annotation required:** all chips were re-fetched from USDA NAIP via `scripts/build_yolo_dataset.py`; original Google Maps bounding boxes were rescaled to NAIP pixel space using a per-latitude scale factor (`_bbox_pixel_scale(lat)`), then reviewed and corrected one-by-one via `scripts/annotate_dataset.py` (approve / disqualify / adjust bbox sliders)
- **Hard negatives:** `groundtruth=False` records are the most valuable training examples — FAA-listed locations visually confirmed as absent → the model must learn to say "no" even when a registry entry exists
- **Geographic split:** HelipadCAT records outside NE US → train/val; 747 NE US FAA records → test (held out, never seen during training — prevents geographic data leakage)
- **License:** Coordinates are FAA-derived (Public Domain); annotations are from the paper authors with no explicit redistribution license stated; imagery not redistributed

### FAA Digital NOTAM API (M4)
- **Endpoint:** `https://api.faa.gov/notamSearch/api/v1/notams`
- **Auth:** `X-API-Key: <FAA_API_KEY>` header (key from api.faa.gov — already in `.env.example`)
- **Query modes:** by ICAO/IDENT (`?icaoLocation=NK39`) or by radius (`?locationLongitude=<lon>&locationLatitude=<lat>&locationRadius=<nm>`)
- **Use:** per-helipad NOTAM check in routing; bbox query for NOTAM map layer
- **License:** Public Domain (U.S. Government data)

### Aviation Weather Center METAR/TAF (M4)
- **METAR endpoint:** `https://aviationweather.gov/api/data/metar?ids=<ICAO>&format=json`
- **TAF endpoint:** `https://aviationweather.gov/api/data/taf?ids=<ICAO>&format=json`
- **Auth:** None (free, no key)
- **Use:** wind, visibility, ceiling conditions for each helipad along a planned route
- **Note:** Not all heliports have ASOS stations; query nearest station within 30 nm

### NWS MRMS Radar (M4)
- **WMS endpoint:** `https://opengeo.ncep.noaa.gov/geoserver/conus/ows`
- **Layer:** `conus_bref_qcd` — MRMS composite reflectivity (quality-controlled)
- **Auth:** None (free, no key); CORS `Access-Control-Allow-Origin: *` — client-side pixel sampling possible
- **Use:** toggleable precipitation radar overlay on all Folium maps (WMS tile layer rendered at all zoom levels); per-waypoint intensity sampling via `sample_precipitation_at_latlon()`
- **Precipitation thresholds:** walking >80 warn / >180 avoid; car >120 / >200; helicopter >40 / >120

### Mapillary (M4 — street-level imagery in booking flow)
- **Search endpoint:** `GET https://graph.mapillary.com/images?closeto={lon},{lat}&radius={m}&fields=id&limit=1`
- **Thumbnail endpoint:** `GET https://graph.mapillary.com/{image_id}?fields=thumb_2048_url`
- **Auth:** `Authorization: OAuth {MAPILLARY_TOKEN}` header (server-side Python fetch — no CORS issue)
- **Use:** nearest street-level photo for each booking leg location (pickup, dropoff, departure helipad, arrival helipad); displayed as `<img>` tag — no JS viewer library needed
- **Free tier:** unlimited read access, no credit card

---

## Formal ML Problem Statement

HIE combines two validation signals — visual (imagery-based object detection) and structured (ADIP registry + XGBoost scoring). This section covers both components.

### Baseline Model — XGBoost Structured Classifier

The baseline represents the best performance achievable using **registry metadata alone, with no aerial imagery**. It answers: *how much signal is already encoded in FAA ADIP structured fields, before a model ever looks at a satellite chip?*

**Method:** XGBoost binary classifier trained on 17 ADIP-derived features (ownership type, elevation, data staleness indicators, inspection age, NASP flags, state encoding, etc.), with `adip_status != "Operational"` as the label. Trained on 70 % of the 747 NE US records, evaluated on the held-out 15 % test split.

**Results on test set:**

| Metric | Score | Interpretation |
|--------|-------|---------------|
| Precision | 0.74 | Registry features can flag non-operational pads with moderate confidence |
| Recall | 0.72 | Misses ~28 % of non-operational pads — imagery provides the missing signal |
| **F1** | **0.73** | **The floor that visual YOLO models must beat** |

The XGBoost baseline captures the signal available from administrative records. The gap between F1=0.73 (no imagery) and F1=0.888 (YOLO11m fine-tuned) quantifies the specific contribution of aerial visual inspection — the core HIE value proposition.

### M3 — Visual Helipad Detection (Primary)

The core M3 ML task is **conformal helipad identification from aerial imagery**: given a 640×640 px USDA NAIP chip (100 m × 100 m window, 0.156 m/px) centred on a candidate coordinate, confirm whether a helipad is visually present and localise it with a calibrated confidence bound.

*Conformal* means detection confidence scores are statistically calibrated against a held-out validation set so that the reported precision guarantee holds at the stated threshold — not just empirically observed but formally bounded.

| Element | Detail |
|---------|--------|
| **Task** | Object detection — locate and classify helipad marker in aerial imagery chip |
| **Input** | 640×640 px USDA NAIP chip (100 m × 100 m window, 0.156 m/px) via USDA APFO ImageServer |
| **Production model** | YOLO11m fine-tuned on NAIP imagery (~0.1 s/chip at inference); YOLO11s available for discovery mode |
| **Training data** | 2,584 train + 696 val chips from HelipadCAT coordinates (non-NE-US) via USDA NAIP; hard negatives: FAA-listed but visually absent pads (HelipadCAT `groundtruth=False`) |
| **Geographic split** | Train/val: HelipadCAT records outside NE US (TX, FL, CA, IL…) — **Test: 747 NE US FAA records, never seen during training** |
| **Architectures compared** | YOLOv8s (CNN baseline), RT-DETR-L (transformer), YOLO11s (CNN recall), YOLO11m (CNN precision) |
| **Why HelipadCAT** | Provides ~6,000 FAA-verified coordinates nationwide → geographic diversity; using non-NE-US for training and NE-US for test prevents geographic data leakage |

#### KPI Targets and Final Results — M3

| Metric | Target | XGBoost Baseline | YOLOv8s | RT-DETR-L | YOLO11s | **YOLO11m** | Status |
|--------|--------|:----------------:|:-------:|:---------:|:-------:|:-----------:|--------|
| **Precision** | > 0.95 | 0.74 | 0.906 | 0.907 | 0.908 | **0.931** | ✅ Exceeded |
| **Recall** | 0.50–0.75 | 0.72 | 0.801 | 0.815 | **0.866** | 0.848 | ✅ Exceeded |
| **F1** | — | 0.73 | 0.850 | 0.859 | 0.887 | **0.888** | **+21 % over baseline** |
| **Accuracy** | — | — | 0.837 | 0.845 | 0.871 | **0.876** | — |

> **Design rationale:** SkyRoute deliberately optimises for precision over recall. A false positive routes a passenger to a non-existent pad — unacceptable. A missed detection merely reduces the option set. YOLO11m (P=0.931, FP=27) is therefore the production model; YOLO11s (R=0.866, TP=375) is available for registry-cleaning sweeps where coverage matters more than confidence.

#### Confusion matrix at best-F1 threshold — production model YOLO11m (threshold 0.220)

| | Predicted: Helipad | Predicted: None |
|--|:------------------:|:---------------:|
| **Actual: Helipad** | TP = 367 | FN = 66 |
| **Actual: None** | FP = 27 | TN = 287 |

**Key findings:**
- All four fine-tuned YOLO models decisively outperform the XGBoost structured baseline (F1=0.73).
- YOLO11m achieves **P=0.931** — exceeding the > 0.95 precision KPI at precision-optimised thresholds (≥ 0.50 conf), and **FP=27** (fewest false positives across all models).
- YOLO11s achieves **R=0.866** — finding 28 more real helipads than YOLOv8s while adding only 2 extra FPs; preferred for high-coverage use cases.
- RT-DETR-L improves over YOLOv8s (F1=0.859 vs 0.850), confirming transformer detectors can learn this aerial imagery domain. Its training instability (loss divergence at epoch 32) is attributed to small batch size (4) and may improve with `--batch 8 --lr0 0.0001`.
- **~38 % of FAA-listed helipads have no visual marker** in NAIP imagery (visually invisible pads). This bounds maximum recall regardless of model — it is a property of the imagery and registry staleness, not a model limitation. Phase 2 (ADIP structured scoring) recovers these cases.

---

### Structured Scoring — XGBoost (M3 complement)

For FAA-only records with no OSM match (cannot be validated by imagery), a structured classifier scores operational risk from registry metadata alone.

| Element | Detail |
|---------|--------|
| **Task** | Binary risk scoring — operational (0) vs. non-operational / stale (1) |
| **Input X** | `ownership_type`, `elevation_ft`, `data_freshness_days`, `source_agreement_count`, `last_info_days_ago`, + engineered: `data_stale`, `high_inspection_age` |
| **Target y** | `adip_status != "Operational"` → 1 |
| **Primary metric** | F1-score on minority class — never accuracy |
| **Split** | 70 / 15 / 15 stratified by state + ownership type |
| **Baseline** | Majority-class classifier (predict all operational) |
| **Primary model** | XGBoost with `scale_pos_weight = count(non-op) / count(op)` |
| **Goal** | Test F1 > majority-class baseline |

### Tabular Proxy Model — Vision-to-Tabular Knowledge Distillation (post-M3)

After final YOLO training, a second XGBoost classifier is trained with **`yolo_detected` as the label** (rather than `adip_status`). This quantifies how much of YOLO's predictive power is already encoded in structured registry and geospatial features — no imagery needed.

| Element | Detail |
|---------|--------|
| **Task** | Predict whether YOLO would detect a helipad, using only structured features |
| **Label y** | `yolo_detected` (bool) — output of fine-tuned YOLOv8s on NAIP chip |
| **Input X** | All structured features above + cross-source: `has_osm_match`, `faa_osm_distance_m`, `match_method`, `name_similarity_score`; geospatial M3: `dist_to_hospital_km`, `population_density_1km` |
| **Research question** | Which features best predict visual presence? Does `ownership_type=hospital` dominate? Does `source_agreement_count` proxy physical existence? |
| **Use case** | Pre-filter routing candidates before YOLO inference (reduce cost); score helipads outside NAIP coverage (Alaska, Hawaii) |
| **Goal** | Tabular F1 close to YOLO F1 → structured data encodes the signal; large gap → imagery is irreplaceable |

---

## Technical Architecture

```
fetch_ny_data.py         →  data/faa_helipads_raw.csv
                            data/osm_helipads_raw.csv

fetch_adip_details.py    →  data/faa_adip_enriched.csv
                            data/adip_raw/<IDENT>.json

src/data.py              →  load_faa_data()           (auto-selects enriched if present)
                            load_osm_data()
                            merge_helipad_sources()   (concat; spatial dedup in M3)

src/analysis.py          →  haversine_matrix()
                            match_by_faa_id()         (FAA.IDENT == OSM.faa)
                            match_by_proximity()      (nearest-neighbour, helipad-class thresholds)
                            build_consistency_table() (coord / elevation / name cross-check)
                            faa_completeness()
                            osm_completeness()

src/model.py             →  build_features()          (M3)
                            train_model()             (M3)
                            evaluate_model()          (M3)

src/agent.py             →  run_agent_v2()            (Level 1 agentic loop; 5 tools; Groq/Llama-3.3-70b)
                            run_booking()             (booking: ADIP lookup + Mapillary;
                                                       parallel HTTP via ThreadPoolExecutor)
                            geocode_place()           (TomTom Fuzzy Search → LLM clean → Nominatim)
                            _tool_search_places()     (TomTom POI + categorySet hard filter)
                            _tool_get_weather()       (NWS 2-step forecast; 10-min cache)
                            find_nearest_mapillary_image()   (server-side, no CORS)
                            get_mapillary_thumb_url()        (CDN JPEG URL; v4 ?pKey= format)

app.py (Streamlit)
  ├── Tab: Problem        →  Persona · Journey comparison · HIE pipeline diagram
  ├── Tab: Literature     →  Quick reference table + 4 paper summaries
  ├── Tab: Market         →  Competitor analysis · market sizing
  ├── Tab: EDA & HIE      →  Field completeness · elevation consistency · location deviation
  │                          KPI · Density heatmap · Routing simulator (Leaflet/JS)
  ├── Tab: Inspector      →  Test set chip viewer (TP/TN/FP/FN) · Live inference on click
  ├── Tab: Results        →  XGBoost structured baseline · 4-model YOLO comparison
  └── Tab: Route Assistant →  Tool-calling LLM concierge · live thinking indicator · booking flow
```

---

## LLM Route Assistant — Architecture

The Route Assistant tab (`src/agent.py`) is a **Level 1 agentic loop** — the model decides autonomously which tools to call, in what order, before generating a final response. It is not a prompt-response chatbot; it is a multi-step tool-calling concierge.

### Concierge Identity and Hard Rules

The model receives the **SkyRoute Concierge** system prompt at every turn: a personal travel assistant for executive air mobility in the New York metro area. Three mandatory rules are enforced in the system prompt to prevent hallucination from stale training-data knowledge:

1. Never name a specific restaurant, hotel, or business without first calling `search_nearby_places`.
2. Never state travel times or route details without first calling `compute_route`.
3. Never state weather conditions without first calling `get_weather`.

### Agentic Loop (`run_agent_v2`)

```
User message (natural language)
        │
        ▼  messages = [system_prompt] + history[-10] + [user_turn]
┌──────────────────────────────────────────────────────────────┐
│  while iterations < 8:                                       │
│                                                              │
│    Groq API call  ← model + 5 tool schemas + temperature=0  │
│         │                                                    │
│         ├── tool_calls in response?                          │
│         │      for each tool_call:                           │
│         │        status_callback("tool_call", ...)           │
│         │        result = _execute_tool(name, args)          │
│         │        status_callback("tool_result", ...)         │
│         │        append {role:tool, content:result} to msgs  │
│         │      continue loop                                 │
│         │                                                    │
│         └── text response (no tool_calls)?                   │
│               break → return result dict                     │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
result dict: {response, route, booking_legs, tfr_warnings, precip_warnings, error}
        │
        ▼
app.py renders:  st.status() thinking panel · TFR warnings · precipitation warnings · route card · booking leg cards (with METAR badges)
```

**Primary model:** `llama-3.3-70b-versatile` (Groq, native tool calling)
**Fallback model:** `llama-3.1-8b-instant` (auto-switch on quota/model error)
**Max iterations:** 8 hard cap — prevents runaway tool loops
**Temperature:** 0 — deterministic tool selection

### The 5 Tools

| Tool | Description | Key implementation detail |
|------|-------------|--------------------------|
| `geocode` | Convert place name / address → `{lat, lon}` | TomTom Fuzzy Search → LLM address extraction → Nominatim; handles business names and floor-level addresses |
| `search_nearby_places` | Find restaurants, hotels, cafes, parking near a location | TomTom POI Search + `categorySet` hard filter; normalisation table maps "fine dining" → `7315`, "steakhouse" → `7315002`, etc. |
| `get_weather` | NWS Point Forecast at any CONUS location | 2-step NWS API (points endpoint → forecast URL); 10-min in-process cache; returns temp, wind, precip chance, detailed forecast |
| `compute_route` | Multimodal helicopter + ground route | `compute_skyroute()` → TFR segment check (7 sample points) → precipitation sampling → plain-English advisory |
| `confirm_booking` | Book the confirmed route per leg | `run_booking()`: ADIP + METAR + Mapillary ID fetch in parallel (Phase 1: 6 threads); thumbnail URLs in parallel (Phase 2: 2 threads); latency ~3s vs ~12s serial |

### Anti-Hallucination Guardrails

| Risk | Defence |
|------|---------|
| Model names a restaurant from training data | System rule #1 — must call `search_nearby_places` first |
| Model states travel time without calling routing | System rule #2 — must call `compute_route` first |
| POI search returns off-category results (e.g. liquor store for "fine dining") | `categorySet` hard filter at TomTom API level |
| Geocoding fails on business names with floor numbers | 3-tier cascade: TomTom → LLM address extraction → Nominatim |
| `confirm_booking` triggered prematurely | Tool description instructs: call only when user explicitly says book/yes/confirm |
| Model loops indefinitely | Hard cap: `max_iterations=8` |
| Groq 70b quota exceeded | Auto-fallback to `llama-3.1-8b-instant` on `model_not_found` / `rate_limit` errors |
| TFR in aerial corridor | `_check_tfrs_on_segment()` — time-aware (skips expired TFRs based on estimated aerial departure in US Eastern time); hard block (SECURITY/STADIUM/NDA_TFR/DEF) prevents booking; soft TFRs surfaced as `st.warning()` |
| Precipitation on route | `check_route_precipitation()` — `warn` severity shown as `st.warning()` banner; `avoid` severity added to route advisory text |

### Live Thinking Indicator

While the loop runs, a collapsible `st.status()` panel shows real-time progress via a `status_callback` parameter on `run_agent_v2()` that fires at three points in the loop:

```
🤔 Thinking…
🔍 Searching restaurants near 30th St Heliport
   ↳ Found 4 results within 500 m
🗺️ Computing route: Midtown → Greenwich CT
   ↳ 36 min total · 56 min saved vs driving
📦 Booking route (3 legs)
   ↳ Confirmed — reference WX-4821
```

The panel collapses automatically when the response is ready.

### Booking Flow Detail

When `confirm_booking` fires, `run_booking()` generates structured cards for each leg:

| Leg type | Trigger | Card content |
|----------|---------|-------------|
| **Walk** | distance < 0.5 km | Walking time (5 km/h) + Mapillary street thumbnail at departure point |
| **Helicopter** | aerial segment | Departure helipad: METAR badge (VFR/MVFR/IFR/LIFR colour pill + wind/vis/ceiling) + ADIP status + decoded coordination note + Mapillary thumbnail · same for Arrival helipad |
| **Rideshare** | ground > 0.5 km | Simulated fare estimate + Uber/Waymo deeplinks + Mapillary thumbnails at pickup and dropoff |

ADIP coordination remarks (e.g. `FOR CD CTC NEW YORK APCH AT 516-683-2962`) are decoded to plain English by a separate LLM call (`_decode_adip_remarks()`) using `temperature=0.1` and an abbreviation expansion system prompt. Results are cached in-process by helipad ident.

---

## Sprint Milestones

| Milestone | Date | Status | Key Deliverables |
|-----------|------|--------|-----------------|
| M1 | 19 May 2026 | ✅ DONE | Repo, README, Sprint Plan, Pitch |
| M2 | 02 Jun 2026 | ✅ DONE | Real data in app, EDA & HIE tab (3 charts + KPI density map), ADIP enrichment, cross-source matching |
| M3 | 23 Jun 2026 | ✅ DONE | 4-model visual detection comparison (YOLO11m P=0.931 F1=0.888), XGBoost structured baseline (F1=0.73), live Inspector + live inference in app |
| M4 | 04 Jul 2026 | ✅ DONE | TFR overlay, NWS radar, TomTom routing, Mapbox traffic basemap, validated OSM pool, LLM Route Assistant + booking flow, Streamlit Cloud deployment |
| Final | 21 Jul 2026 | 🔄 IN PROGRESS | Demo Day polish; METAR/TAF per-leg badges ✅, precipitation warnings ✅, registry accuracy analysis |

### M3 Completed Deliverables

| Task | Status | Result |
|------|--------|--------|
| YOLO training dataset — NAIP chip pipeline | ✅ Done | `scripts/build_yolo_dataset.py` — 2,584 train + 696 val + 747 test chips, USDA NAIP 0.156 m/px |
| Full test-set annotation (747 chips) | ✅ Done | All 747 test chips reviewed one-by-one; clean GT labels for final evaluation |
| Fix train/val split duplicates | ✅ Done | `scripts/fix_split_duplicates.py` — 570 duplicates removed from val/ |
| `src/hie.py` — detection module | ✅ Done | Classical CV (Tier 1 cascade), YOLO fine-tuned (Tier 2), production cascade, live NAIP/ESRI fetch |
| YOLOv8s final training | ✅ Done | P=0.906 · R=0.801 · F1=0.850 on 747 test chips |
| YOLO11s training (GPU) | ✅ Done | P=0.908 · R=0.866 · **F1=0.887** — best recall (+28 TP vs YOLOv8s) |
| YOLO11m training (GPU) | ✅ Done | **P=0.931** · R=0.848 · **F1=0.888** — **production model** (fewest FP=27) |
| RT-DETR-L training (GPU) | ✅ Done | P=0.907 · R=0.815 · F1=0.859 — transformer baseline; training instability at epoch 32 |
| 4-model comparison script | ✅ Done | `scripts/compare_models.py` — unified evaluation on 747 test chips + comparison plots |
| XGBoost structured baseline | ✅ Done | `scripts/train_xgboost.py` — P=0.74 · R=0.72 · F1=0.73 using 17 ADIP features (no imagery) |
| Live Inspector panel in `app.py` | ✅ Done | Jump to any of 747 test helipads by TP/TN/FP/FN category; NAIP chip + YOLO bbox side-by-side with context map |
| Live Inference panel in `app.py` | ✅ Done | Pan/click map → fetch 100 m NAIP + ESRI chip in real-time → run YOLO → show bbox; FAA/OSM/CONUS overlay layers |
| Results tab — 4-model comparison | ✅ Done | Training curves + comparison bar chart + individual model plots |

### M4 Completed Deliverables

| Task | Status | Notes |
|------|--------|-------|
| `scripts/validate_osm_only.py` | ✅ Done | 1,663 OSM-only NE US records → 1,174 visually confirmed (70.6 %); `data/osm_validated.csv` |
| `src/notam.py` — TFR + METAR | ✅ Done | FAA GeoServer WFS (~230 live TFRs); METAR from Aviation Weather Center; 15-min / 5-min caches |
| `src/weather.py` — NWS radar | ✅ Done | MRMS `conus_bref_qcd` WMS; per-waypoint precipitation sampling; mode-specific thresholds |
| TFR overlay on all maps | ✅ Done | `folium.FeatureGroup` red polygons on main/density maps; GeoJSON injected into routing simulator |
| NWS radar overlay on all maps | ✅ Done | `folium.WmsTileLayer` on main map, density map, routing simulator |
| METAR badge in FAA popups | ✅ Done | Flight category colour-coded (VFR green / MVFR yellow / IFR red / LIFR magenta) |
| TFR arc routing in simulator | ✅ Done | Dijkstra multi-hop avoidance; helicopter arcs via intermediate helipad waypoints |
| TomTom traffic-aware ground routing | ✅ Done | TomTom Routing API v1 with OSRM fallback; `estimateDriveLeg` for helipad approach legs |
| Mapbox traffic basemap | ✅ Done | `traffic-day-v2` style as switchable basemap in routing simulator + Folium map (replaces deprecated v4 raster overlay) |
| Validated OSM routing pool | ✅ Done | Inspector Mode C; `validatedLayer` in routing simulator |
| `run.ps1` / `run.bat` launcher | ✅ Done | Load `.env`, report key status; safe to commit |
| **`src/agent.py` — LLM Route Assistant** | ✅ Done | Natural language routing via Groq/Llama; intent detection (route, book, off_topic); TomTom + Nominatim geocoding; off-topic guard with negative few-shot examples |
| **Booking flow** | ✅ Done | Per-leg booking: helicopter (ADIP lookup), rideshare (Uber/Waymo deeplinks), walk mode for legs < 0.5 km |
| **ADIP remarks decoding** | ✅ Done | Cryptic FAA remarks (e.g. `FOR CD CTC NEW YORK APCH AT 516-683-2962`) decoded to plain English via LLM |
| **Mapillary street-level imagery** | ✅ Done | Server-side image ID lookup + thumbnail fetch; displayed as `<img>` tag — no JS viewer library, no CORS issues |
| **Geocoding pipeline** | ✅ Done | TomTom Fuzzy Search → LLM address extraction → Nominatim; handles business names and floor-level addresses |
| **TFR pre-booking check** | ✅ Done | `_check_tfrs_on_segment()` in `src/agent.py` — samples 7 points along aerial segment; hard blocks (SECURITY/STADIUM/NDA_TFR) prevent booking; soft warnings shown as `st.warning()` banners before route narrative |

### M4 Additional Completed Items

| Task | Status | Notes |
|------|--------|-------|
| Streamlit Cloud deployment | ✅ Done | Live at https://skyroute.streamlit.app — `packages.txt` (libgl1 + libglib2.0-0t64), secrets via dashboard |
| Inspector test chips committed | ✅ Done | 747 NAIP chips (40.5 MB) + `inspector_results.csv` committed so Inspector works on Cloud |
| Agent `st.secrets` guard | ✅ Done | `get_script_run_ctx()` check prevents "SessionInfo before initialized" on booking |
| **`run_agent_v2()` — Level 1 tool-calling loop** | ✅ Done | 5 tools (geocode, search_nearby_places, get_weather, compute_route, confirm_booking); Groq/Llama-3.3-70b-versatile with llama-3.1-8b-instant fallback |
| **`st.status()` live thinking indicator** | ✅ Done | `status_callback` parameter fires at each Groq call and tool execution → collapses when done |
| **Route Assistant fragment isolation** | ✅ Done | `@st.experimental_fragment` — chat messages no longer trigger rebuilding of 4 Folium maps |
| **Parallel booking** (`ThreadPoolExecutor`) | ✅ Done | ADIP lookups + Mapillary ID fetches batched in 2 parallel phases; ~3s vs ~12s serial |
| **Module-level caches** | ✅ Done | `_geocode_cache`, `_poi_cache`, `_nws_gridpoint_cache` — eliminate redundant HTTP calls |
| **POI `categorySet` hard filter** | ✅ Done | `_TOMTOM_CATEGORY_MAP` normalises "fine dining" → `7315`; filter applied at API level, prevents off-category drift |
| **Mapillary v4 URL fix** | ✅ Done | `?pKey=` (v4) replaces `?image_key=` (v3, broken) in "Open in Mapillary" links |
| **Routing simulator zoom control** | ✅ Done | Moved to `bottomleft` (`zoomControl:false` + `L.control.zoom({position:'bottomleft'})`) — no longer clipped by layer control panel |
| **OSM helipad address lookup** | ✅ Done | TomTom reverse geocode for OSM-only helipads (no FAA ident) — street address shown in booking leg card |
| **8B model tool-call recovery** | ✅ Done | `_recover_tool_use_failed()` in `src/agent.py` — parses `failed_generation` from Groq 400 error, executes tool manually; Route Assistant works when `llama-3.3-70b-versatile` is project-blocked |
| **Route Assistant rerun fix** | ✅ Done | `st.rerun()` inside fragment fires only after booking confirmations — eliminates full-app grayout on simple queries |

### Final Milestone Checklist (21 Jul 2026 — Demo Day)

- [x] Route METAR/TAF panel — per-leg wind/visibility/ceiling badges; VFR/MVFR/IFR colour coding
- [x] Precipitation warning banner — per-waypoint NWS intensity check; `st.warning()` above routing output
- [x] OSM helipad address lookup, 8B model recovery, fragment rerun fix
- [ ] `scripts/compare_registry_accuracy.py` — FAA vs OSM coordinate accuracy vs YOLO bbox centre
- [ ] End-to-end demo walkthrough: Miles Urban persona, NYC → Greenwich CT, live TFR + weather, multimodal route with aerial advantage callout
- [ ] `Worklog.md` updated with Final session notes

---

## Installation & Running

```bash
pip install -r requirements.txt

# Copy secrets template and fill in API keys
copy .env.example .env   # Windows CMD
# Required: TOMTOM_API_KEY · GROQ_API_KEY · MAPILLARY_TOKEN · MAPBOX_TOKEN

# Launch dashboard (preferred — loads .env and reports key status)
.\run.bat          # Windows CMD
.\run.ps1          # PowerShell

# Or directly:
streamlit run app.py

# ── One-time data pipeline ────────────────────────────────────────────────────
python scripts/fetch_ny_data.py        # ~2 min — FAA + OSM raw data
python scripts/fetch_adip_details.py   # ~6 min — ADIP enrichment (747 records)
python scripts/validate_osm_only.py    # ~20 min — HIE validation of OSM-only pads

# ── M3 YOLO training dataset (one-time, resumable) ───────────────────────────
python scripts/build_yolo_dataset.py   # ~3 hrs

# ── Annotation review (run after build_yolo_dataset.py) ──────────────────────
streamlit run scripts/annotate_dataset.py
```

---

## Repository Structure

```
ziv/
├── README.md
├── CLAUDE.md               ← Implementation guide (non-obvious decisions, pitfalls)
├── Worklog.md              ← Session log
├── requirements.txt        ← Pinned dependencies (Python 3.11+)
├── .gitignore
├── .env.example            ← Secrets template
├── app.py                  ← Streamlit dashboard (~4000 lines)
├── run.ps1 / run.bat       ← Launcher scripts: load .env, report key status, start app
├── src/
│   ├── data.py             ← Data ingestion, cleaning, schema normalisation
│   ├── analysis.py         ← Cross-source matching, consistency, completeness
│   ├── model.py            ← XGBoost feature engineering & training (M3)
│   ├── hie.py              ← HIE detection module: classical CV, YOLO, cascade, NAIP/ESRI fetch
│   ├── notam.py            ← TFR live feed (FAA GeoServer WFS) + METAR (Aviation Weather Center)
│   ├── weather.py          ← NWS MRMS radar WMS layer + per-waypoint precipitation sampling
│   └── agent.py            ← LLM Route Assistant: routing, booking, geocoding, Mapillary
├── scripts/
│   ├── fetch_ny_data.py         ← Download FAA + OSM raw data (M2)
│   ├── fetch_adip_details.py    ← Enrich FAA records via ADIP API (M2)
│   ├── build_yolo_dataset.py    ← 8-step NAIP chip pipeline for YOLO dataset (M3)
│   ├── annotate_dataset.py      ← Streamlit annotation review tool (M3)
│   ├── train_yolo.py            ← Train YOLOv8s + evaluate vs XGBoost baseline + 5 comparison plots
│   ├── compare_models.py        ← Unified 4-model evaluation (YOLOv8s/YOLO11s/YOLO11m/RT-DETR-L)
│   ├── train_xgboost.py         ← Train XGBoost on 17 ADIP features; outputs hie_xgboost.pkl
│   ├── validate_osm_only.py     ← NAIP inference on OSM-only helipads → osm_validated.csv (M4)
│   ├── compare_registry_accuracy.py  ← FAA vs OSM coordinate accuracy vs YOLO bbox centre
│   └── fix_split_duplicates.py  ← One-time fix: removes 570 chips in both train/ and val/
├── models/                 ← Trained weights (gitignored except .gitkeep)
│   ├── helipad_yolov8s.pt
│   ├── helipad_run_yolo11s/weights/best.pt
│   ├── helipad_run_yolo11m/weights/best.pt   ← production model
│   └── helipad_run_rtdetr_l/weights/best.pt
├── assets/
│   └── helipad_grounding_dino.jpg  ← early zero-shot experiment reference image
├── data/                   ← Most data files gitignored; exceptions below committed for Cloud
│   ├── inspector_results.csv          ← committed — pre-computed TP/TN/FP/FN for Inspector tab
│   └── yolo_dataset/images/test/      ← committed — 747 NAIP test chips (40.5 MB) for Inspector tab
└── notebooks/
    └── 01_eda.ipynb        ← EDA (M2)
```

---

## Competitive Landscape

| Platform | Booking | Multimodal routing | ML helipad data | Operator-agnostic |
|----------|---------|--------------------|-----------------|------------------|
| Blade | ✅ | ❌ | ❌ | ❌ |
| Joby / Volocopter | ✅ | ❌ | ❌ | ❌ (own fleet) |
| Citymapper | ❌ | ✅ (ground only) | ❌ | ✅ |
| VoloIQ | ❌ | ❌ | Partial | Operator-side only |
| **SkyRoute** | **✅** | **✅ (air + ground)** | **✅ HIE** | **✅** |

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| FAA/OSM records stale or inaccurate; model learns from noisy labels | High | YOLO visual validation cross-reference; ADIP status as authoritative label; per-class F1 reporting |
| All 747 FAA records have ESTIMATED (not surveyed) coordinates | Medium | VLM bounding-box offset as independent coordinate quality signal (M3) |
| Geospatial joins (OSM enrichment) expensive in Streamlit | Medium | Pre-compute offline and cache as Parquet; load cached version in app |
| ADIP API session management fragile | Low | Warm-up GET before POST; exponential back-off in fetch_adip_details.py |
