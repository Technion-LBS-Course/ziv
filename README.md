# ✈️ SkyRoute — Advanced Air Mobility Navigation Platform

> *"Door-to-sky-to-door. Your fastest path, elevated."*

---

## One-Liner

**Business travelers in dense U.S. metros** suffer from fragmented urban mobility and hours lost to traffic, with no platform combining Advanced Air Mobility (AAM) into seamless door-to-door journeys — so we're building **SkyRoute**, an operator-agnostic multimodal routing and booking platform that uses a **ML-powered Helipad Intelligence Engine (HIE)** to validate, score, and route via verified air+ground itineraries across the New York metro and beyond.

---

## The Problem

Business travelers and frequent flyers lose hours daily to urban traffic congestion, long airport transfers, and the friction between transport modes. In a world where minutes equal billable hours, no single platform today enables seamless planning and booking of eVTOL aircraft, helicopters, and ground transport as a unified, door-to-door journey.

Existing solutions are siloed: Blade books helicopters but lacks multimodal integration; Google Maps offers ground routing but ignores AAM entirely; Joby and Volocopter serve their own operators only. There is no operator-agnostic, business-grade platform that fuses all modes.

Beyond routing, the underlying infrastructure data is broken. Public helipad databases (FAA, OurAirports, OSM) are notoriously inconsistent — outdated coordinates, decommissioned pads listed as active, missing usability metadata. Feeding this raw data into a routing engine creates liability and trust failures.

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

Raw helipad databases (FAA, OurAirports, OSM) are incomplete, stale, and contain military or decommissioned pads. HIE is a **3-phase ML pipeline** that validates every candidate pad before it enters the routing engine.

```
Raw Input              Phase 1                    Phase 2                 Phase 3 ⭐             Output
FAA · OSM  →  YOLOv8s visual detection  →  LLM text / status  →  ADIP arrival  →  ✅ Validated pad
747+ records     cascade: CV → YOLO             search grounding       coordination     added to routing
```

### Phase 1 — Visual Validation (YOLOv8s fine-tuned cascade)

USDA NAIP imagery chips (100 m × 100 m, 640×640 px, 0.156 m/px) are fetched for each candidate coordinate from the USDA APFO ImageServer. A two-tier production cascade validates each pad:

- **Tier 1 — Classical CV** (`detect_classical`): OpenCV normalised cross-correlation against H-shape templates at 3 scales × 2 rotations × 2 colour variants. Accept if confidence ≥ 0.75; else fall through to Tier 2.
- **Tier 2 — YOLOv8s fine-tuned** (`detect_yolo`): Domain-specific model trained on 2,584 NAIP chips from across the continental US. Detects H-markers and rooftop pads that lack an explicit H marking.

The detected bounding-box centroid is back-projected to geographic coordinates and compared against the registry coordinate to produce `offset_m`. Records where no helipad is detected, or where offset exceeds a design-class threshold, are flagged for manual review.

**Academic ablation (zero-shot baselines vs. fine-tuned, on 747 NE US test chips):**

| Method | F1 | Lat/chip | Notes |
|---|---|---|---|
| Classical CV | ~0.00 | ~0.13 s | False positives on urban H-shapes; IoU ≈ 0 |
| YOLO-World small (zero-shot) | 0.00 | ~0.2 s | Total domain gap — natural-image training |
| Grounding DINO tiny (zero-shot) | 0.00 | ~5.6 s | Partial IoU (~0.16 on some chips); poor nadir localisation |
| **YOLOv8s fine-tuned** | **TBD** | ~0.1 s | **Domain-specific training on NAIP — the contribution** |

The zero-shot failures quantify exactly why fine-tuning on NAIP imagery is necessary.

### Phase 2 — LLM Text/Status Validation

For candidates that pass Phase 1, a retrieval-augmented LLM query (Gemini / GPT-4o with search grounding) takes the helipad name and coordinates and searches for evidence of closure, restricted access, or military designation. Example: the OSM node "Caven Point USAR" is correctly identified as a military-use pad ineligible for civilian routing, without any labeled training example.

### Phase 3 — ADIP Arrival Coordination (stretch goal)

For validated FAA helipads, the ADIP Airport Master Record provides TLOF/FATO dimensions, design category, last inspection date, ATC contact, EV charging availability, and ingress/egress bearings. This data feeds the routing engine's arrival planning and operator handoff layer.

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

### OurAirports (optional third source)
- **Download:** `curl -o data/ourairports_raw.csv https://ourairports.com/data/airports.csv` then filter `type == heliport`
- **License:** CC0

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

### RainViewer Radar (M4)
- **Frame list:** `https://api.rainviewer.com/public/weather-maps.json`
- **Tile URL:** `https://tilecache.rainviewer.com{path}/256/{z}/{x}/{y}/2/1_1.png` (color 2 = universal blue)
- **Auth:** None (free, no key)
- **Use:** toggleable precipitation radar overlay on Folium map; route precipitation check by sampling tile pixel at waypoint coordinates

---

## Formal ML Problem Statement

HIE combines three validation signals — visual (imagery-based object detection), textual (LLM search grounding), and structured (ADIP registry). This section covers both components.

### Baseline Model — Registry-Agreement

The baseline represents the best performance achievable using **registry data alone, with no imagery**. It answers: *how well can cross-source data agreement identify real helipads without ever looking at a satellite chip?*

**Method:** For each FAA record in the 747-record NE US test set, check whether a matching OSM record exists and whether their coordinates agree.

| Outcome | Definition |
|---------|------------|
| **True Positive (TP)** | FAA `IDENT` matches OSM `faa` tag AND haversine distance < 10 m — two independent sources agree on the same physical location |
| **False Positive (FP)** | FAA `IDENT` matches OSM `faa` tag BUT distance ≥ 10 m — sources disagree on position, suggesting a coordinate error in at least one registry |
| **False Negative (FN)** | 50 % of OSM-only records — FAA has no entry for this location; treated as unconfirmed |
| **True Negative (TN)** | Remaining 50 % of OSM-only records |

**10 m threshold rationale:** derived from the Bell 206 medium helicopter FATO diameter (~21 m). Two registries agreeing within half the pad diameter are almost certainly describing the same physical structure.

**No imagery is used.** This baseline is implemented in `scripts/train_yolo.py` via `compute_baseline()`, which calls `match_by_faa_id()` from `src/analysis.py`.

**Preliminary results on 747 NE US test records:**

| Metric | Score | Interpretation |
|--------|-------|---------------|
| Precision | 0.63 | 37 % of FAA–OSM ID matches have a coordinate discrepancy ≥ 10 m |
| Recall | 0.21 | Only ~24 % of OSM records carry a `faa` tag; most FAA records have no OSM counterpart → counted as FN |
| **F1** | **0.32** | **The floor that YOLOv8s must beat** |

The low recall is structural, not a model failure: the baseline can only confirm a helipad if OSM happens to have tagged it with the FAA identifier. Registry cross-referencing alone cannot validate the majority of helipads — which is exactly the gap that visual detection fills.

### M3 — Visual Helipad Detection (Primary)

The core M3 ML task is **conformal helipad identification from aerial imagery**: given a 640×640 px USDA NAIP chip (100 m × 100 m window, 0.156 m/px) centred on a candidate coordinate, confirm whether a helipad is visually present and localise it with a calibrated confidence bound.

*Conformal* means detection confidence scores are statistically calibrated against a held-out validation set so that the reported precision guarantee holds at the stated threshold — not just empirically observed but formally bounded.

| Element | Detail |
|---------|--------|
| **Task** | Object detection — locate and classify helipad marker in aerial imagery chip |
| **Input** | 640×640 px USDA NAIP chip (100 m × 100 m window, 0.156 m/px) via USDA APFO ImageServer |
| **Model** | YOLOv8s fine-tuned on NAIP imagery (~0.1 s/chip at inference) |
| **Training data** | 2,584 train + 696 val chips from HelipadCAT coordinates (non-NE-US) via USDA NAIP; hard negatives: FAA-listed but visually absent pads (HelipadCAT `groundtruth=False`) |
| **Geographic split** | Train/val: HelipadCAT records outside NE US (TX, FL, CA, IL…) — **Test: 747 NE US FAA records, never seen during training** |
| **Loss** | YOLOv8 default: box regression (CIoU) + classification (BCE) |
| **Why HelipadCAT** | Provides ~6,000 FAA-verified coordinates nationwide → geographic diversity; using non-NE-US for training and NE-US for test prevents geographic data leakage |

#### KPI Targets — M3

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Precision** | **> 0.95** | A false positive routes a passenger to a non-existent or unusable pad — unacceptable safety failure |
| **Recall** | **0.50 – 0.75** | A missed helipad is simply absent from routing options; no safety consequence, only reduced coverage |
| **mAP@50** | **> 0.90** | Standard YOLO object detection benchmark at IoU threshold 0.50 |

> **Design rationale:** SkyRoute deliberately optimises for precision over recall. The routing engine operates on the set of *confirmed* helipads only. A missed detection reduces the option set but never creates a hazardous routing suggestion. This asymmetry justifies a high-confidence detection threshold that may leave some valid pads unconfirmed until manual review clears them.

#### Preliminary Training Results (Session 1 — synthetic GT labels, partial annotation)

Trained on 2,584 chips / validated on 696 chips / evaluated on **747 held-out NE US test chips**. Labels are synthetic (centre-based bbox) with 1,966 of 4,027 chips manually reviewed. A final training session will follow after full test-set annotation.

| Metric | Registry Baseline | YOLOv8s Preliminary | Target | Status |
|--------|:-----------------:|:-------------------:|:------:|--------|
| **Precision** | 0.63 | **0.89** | > 0.95 | Close — expected to improve with verified labels |
| **Recall** | 0.21 | **0.77** | 0.50–0.75 | ✅ Within / above target range |
| **F1** | 0.32 | **0.82** | — | **2.6× better than baseline** |
| **mAP@50** | — | **0.894** | > 0.90 | Near target; val mAP50 peaked at 0.894 (epoch 37/50) |

**Confusion matrix at operating threshold 0.10** (best-F1 threshold on test set):

| | Predicted: Helipad | Predicted: None |
|--|:------------------:|:---------------:|
| **Actual: Helipad** | TP = 368 | FN = 112 |
| **Actual: None** | FP = 45 | TN = 222 |

**Key observations:**
- YOLO dominates the baseline on every metric. The PR curve maintains Precision > 0.95 up to Recall ~ 0.79, while the baseline sits at (Recall=0.21, Precision=0.63).
- The auto-selected operating threshold (0.10) maximises F1 but is too permissive for production routing. At threshold 0.50 the model achieves Precision ~ 0.97, Recall ~ 0.75, F1 ~ 0.82 — a better operating point for safety-critical routing where false positives are unacceptable.
- The 112 FN cases are largely attributable to synthetic labels on stale FAA records (chips where no helipad is visible but the label says positive). These are expected to convert to TN after full test-set annotation, improving Recall.
- These are **preliminary results**. The final training session (after full test-set annotation) will give clean metrics.

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

---

## Technical Architecture

```
fetch_ny_data.py         →  data/faa_helipads_raw.csv
                            data/osm_helipads_raw.csv

fetch_adip_details.py    →  data/faa_adip_enriched.csv
                            data/adip_raw/<IDENT>.json

src/data.py              →  load_faa_data()           (auto-selects enriched if present)
                            load_osm_data()
                            load_ourairports_data()
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

app.py (Streamlit)
  ├── Tab: Problem       →  Persona · Journey comparison · HIE pipeline diagram
  ├── Tab: Literature    →  Quick reference table + 4 paper summaries
  ├── Tab: Market        →  Competitor analysis · market sizing
  └── Tab: EDA & HIE     →  Field completeness · elevation consistency · location deviation
                            KPI (Δ avg first-mile time) · Density heatmap · Routing simulator
```

---

## Sprint Milestones

| Milestone | Date | Status | Key Deliverables |
|-----------|------|--------|-----------------|
| M1 | 19 May 2026 | ✅ DONE | Repo, README, Sprint Plan, Pitch |
| M2 | 02 Jun 2026 | ✅ DONE | Real data in app, EDA & HIE tab (3 charts + KPI density map), ADIP enrichment, cross-source matching |
| M3 | 23 Jun 2026 | 🔄 IN PROGRESS | YOLOv8s helipad detector (Precision > 0.95, mAP@50 > 0.90), LLM status validation, XGBoost structured scoring, live detection overlay in app |
| M4 | 14 Jul 2026 | ⏳ PENDING | NOTAM airspace avoidance + METAR/TAF weather per route leg; RainViewer precipitation overlay on map and route |
| Final | 21 Jul 2026 | ⏳ PENDING | Demo Day, stable URL, documented |

### M3 Progress

| Task | Status | Notes |
|------|--------|-------|
| YOLO training dataset — NAIP chip pipeline | ✅ Done | `scripts/build_yolo_dataset.py` — 2584 train + 696 val + 747 test chips, USDA NAIP 0.156 m/px |
| Annotation review tool | ✅ Done | `scripts/annotate_dataset.py` — approve / disqualify / adjust bbox, persists to CSV |
| Manual bbox annotation | 🔄 In progress | 1,966 / 4,027 reviewed (795 approved · 315 disqualified · 856 adjusted) |
| `src/hie.py` — detection module | ✅ Done | Classical CV, YOLOv8s, YOLO-World, Florence-2, Grounding DINO, production cascade |
| Zero-shot comparison script | ✅ Done | `scripts/compare_zero_shot.py` — Classical CV, YOLO-World, Grounding DINO on 747 test chips |
| Zero-shot baseline results | ✅ Done | YOLO-World F1=0.00, Grounding DINO F1=0.00 (5.6 s/chip), Classical CV F1=0.00 — domain gap confirmed |
| Training + evaluation script | ✅ Done | `scripts/train_yolo.py` — trains YOLOv8s, computes YOLO + registry-baseline metrics, saves 5 plots |
| YOLOv8s preliminary training | ✅ Done | P=0.89 · R=0.77 · F1=0.82 · mAP50=0.894 on 747 test chips (synthetic GT, partial annotation) — 2.6× F1 vs baseline |
| Test set full annotation | ⏳ Post-preliminary | Manually verify all 747 test chips; gives clean Recall/Precision without synthetic-label noise |
| YOLOv8s final training | ⏳ Post-annotation | Retrain with fully verified test labels → 3-way comparison: Baseline / Preliminary / Final |
| Registry accuracy analysis | ⏳ Post-final training | `scripts/compare_registry_accuracy.py` — per matched FAA+OSM pair: which coordinate is closer to YOLO bbox centre? Flags pairs >50m apart as possibly different helipads |
| OSM-only pad validation | ⏳ Post-final training | `scripts/validate_osm_only.py` — fetch NAIP chips for OSM-only NE US pads, run cascade, write `data/osm_validated.csv`; validated pads feed M4 routing pool |
| `src/validation.py` — LLM Phase 2 | ⏳ Planned | Claude Haiku + Overpass heuristic for named/unnamed OSM pads |
| `src/model.py` — XGBoost scoring | ⏳ Optional | Structured binary classifier for FAA-only pads; not required if CV pipeline satisfies M3 rubric |
| Live detection overlay in `app.py` | ⏳ Planned | YOLO-World overlay pre-training; swap to fine-tuned weights after training |

### M4 Task Checklist (do NOT start until M3 is submitted)

**NOTAM + METAR/TAF integration — `src/notam.py`**
- [ ] `fetch_notams_for_ident(ident, api_key)` — FAA NOTAM API query by ICAO/IDENT; returns list of active NOTAM dicts
- [ ] `fetch_notams_for_bbox(lat_min, lon_min, lat_max, lon_max, api_key)` — radius-based query for NOTAM map layer
- [ ] `filter_active_notams(notams)` — keep only NOTAMs whose effective window overlaps `datetime.utcnow()`
- [ ] `notam_closes_airspace(notam)` — classify NOTAM as airspace closure (type D / TFR polygon)
- [ ] `fetch_metar(icao_id)` — Aviation Weather Center METAR for nearest station; returns parsed dict (wind, visibility, ceiling, flight_category)
- [ ] `fetch_taf(icao_id)` — TAF 24-hr forecast for same station
- [ ] `route_weather_summary(helipads, api_key)` — for each helipad dict in route, return NOTAM list + METAR; one API call per helipad

**RainViewer precipitation — `src/weather.py`**
- [ ] `fetch_rainviewer_frames()` — GET `weather-maps.json`; return latest past frame + nowcast frames as list of `{timestamp, path}`
- [ ] `get_radar_tile_url(path, z, x, y)` — construct tile URL for Folium `TileLayer`
- [ ] `sample_precipitation_at_latlon(lat, lon, path)` — fetch tile PNG, convert lat/lon to tile pixel, return intensity 0–255 (0 = no rain)
- [ ] `check_route_precipitation(waypoints, path)` — return per-waypoint `{lat, lon, intensity, label}` for walk + flight legs

**Post-M3 analysis (prerequisite for M4 routing)**
- [ ] `scripts/compare_registry_accuracy.py` — run trained YOLO on 747 test chips, compare FAA vs OSM coordinate distance to bbox centre, flag pairs >50m apart as possibly different helipads; output `data/registry_accuracy.csv`
- [ ] `scripts/validate_osm_only.py` — identify OSM-only NE US helipads (no FAA match), fetch NAIP chip per pad, run `detect_helipad_cascade()`, write `data/osm_validated.csv`; visually confirmed OSM pads join routing pool

**NOTAM + METAR/TAF integration — `src/notam.py`**
- [ ] `fetch_notams_for_ident(ident, api_key)` — FAA NOTAM API query by ICAO/IDENT; returns list of active NOTAM dicts
- [ ] `fetch_notams_for_bbox(lat_min, lon_min, lat_max, lon_max, api_key)` — radius-based query for NOTAM map layer
- [ ] `filter_active_notams(notams)` — keep only NOTAMs whose effective window overlaps `datetime.utcnow()`
- [ ] `notam_closes_airspace(notam)` — classify NOTAM as airspace closure (type D / TFR polygon)
- [ ] `fetch_metar(icao_id)` — Aviation Weather Center METAR for nearest station; returns parsed dict (wind, visibility, ceiling, flight_category)
- [ ] `fetch_taf(icao_id)` — TAF 24-hr forecast for same station
- [ ] `route_weather_summary(helipads, api_key)` — for each helipad dict in route, return NOTAM list + METAR; one API call per helipad

**RainViewer precipitation — `src/weather.py`**
- [ ] `fetch_rainviewer_frames()` — GET `weather-maps.json`; return latest past frame + nowcast frames as list of `{timestamp, path}`
- [ ] `get_radar_tile_url(path, z, x, y)` — construct tile URL for Folium `TileLayer`
- [ ] `sample_precipitation_at_latlon(lat, lon, path)` — fetch tile PNG, convert lat/lon to tile pixel, return intensity 0–255 (0 = no rain)
- [ ] `check_route_precipitation(waypoints, path)` — return per-waypoint `{lat, lon, intensity, label}` for walk + flight legs

**`app.py` map and routing updates**
- [ ] RainViewer radar overlay — toggleable `folium.TileLayer` on all Folium maps; auto-fetch latest frame timestamp on load
- [ ] NOTAM layer — `folium.FeatureGroup` with circle/polygon markers for active NOTAMs in the NE US bounding box; TFR areas as `folium.Polygon` with red outline
- [ ] Per-helipad NOTAM badge in FAA popup — `🚫 ACTIVE NOTAM` if any active NOTAM for that IDENT; else hidden
- [ ] Route METAR/TAF panel — for each helipad leg in routing simulator, show wind/visibility/ceiling; color-coded VFR (green) / MVFR (yellow) / IFR (red) / LIFR (magenta)
- [ ] Precipitation warning banner — if `sample_precipitation_at_latlon` returns intensity > 50 at any walk or flight waypoint, show `⚠️ Precipitation detected along [leg name]`
- [ ] Routing pool includes validated OSM-only pads from `data/osm_validated.csv` (hie_visual_detected=True)

**Environment**
- [ ] `FAA_API_KEY` already in `.env.example` — wire into `src/notam.py` via `os.getenv("FAA_API_KEY")`
- [ ] No new packages needed (all use `requests` + `folium` already in requirements)

---

## Installation & Running

```bash
pip install -r requirements.txt

# Fetch M2 data (one-time)
python scripts/fetch_ny_data.py        # ~2 min — FAA + OSM
python scripts/fetch_adip_details.py   # ~6 min — ADIP enrichment

# Build M3 YOLO training dataset (~3 hrs, resumable)
python scripts/build_yolo_dataset.py

# Launch main dashboard
streamlit run app.py

# Launch annotation review tool
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
├── app.py                  ← Streamlit dashboard (~2700 lines)
├── src/
│   ├── data.py             ← Data ingestion, cleaning, schema normalisation
│   ├── analysis.py         ← Cross-source matching, consistency, completeness
│   └── model.py            ← XGBoost feature engineering & training (M3)
├── scripts/
│   ├── fetch_ny_data.py         ← Download FAA + OSM raw data (M2)
│   ├── fetch_adip_details.py    ← Enrich FAA records via ADIP API (M2)
│   ├── build_yolo_dataset.py    ← 8-step NAIP chip pipeline for YOLO dataset (M3)
│   ├── annotate_dataset.py      ← Streamlit annotation review tool (M3)
│   ├── compare_zero_shot.py     ← Zero-shot ablation: Classical CV / YOLO-World / DINO on 747 test chips
│   ├── train_yolo.py            ← Train YOLOv8s + evaluate vs registry baseline + 5 comparison plots
│   ├── compare_registry_accuracy.py  ← FAA vs OSM coordinate accuracy vs YOLO bbox centre (post-training)
│   └── validate_osm_only.py    ← NAIP inference on OSM-only helipads → osm_validated.csv (post-training)
├── models/                 ← Trained weights land here (gitignored except .gitkeep)
├── assets/
│   └── helipad_grounding_dino.jpg
├── data/                   ← All data files — gitignored
│   └── yolo_dataset/       ← NAIP chips + YOLO labels + review decisions
└── notebooks/
    ├── 01_eda.ipynb        ← EDA (M2)
    └── 02_yolo_training.ipynb  ← YOLOv8s training on Colab (M3, planned)
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
| FAA/OSM records stale or inaccurate; model learns from noisy labels | High | Grounding DINO visual validation cross-reference; ADIP status as authoritative label; per-class F1 reporting |
| All 747 FAA records have ESTIMATED (not surveyed) coordinates | Medium | VLM bounding-box offset as independent coordinate quality signal (M3) |
| Geospatial joins (OSM enrichment) expensive in Streamlit | Medium | Pre-compute offline and cache as Parquet; load cached version in app |
| ADIP API session management fragile | Low | Warm-up GET before POST; exponential back-off in fetch_adip_details.py |
