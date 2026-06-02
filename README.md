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
Raw Input              Phase 1                 Phase 2                 Phase 3 ⭐             Output
FAA · OSM  →  Grounding DINO visual  →  LLM text / status  →  ADIP arrival  →  ✅ Validated pad
747+ records     bounding box check         search grounding       coordination     added to routing
```

### Phase 1 — Visual Validation (Grounding DINO)

Satellite imagery chips (ESRI World Imagery, zoom 19 ≈ 0.22 m/px) are fetched for each candidate coordinate. Grounding DINO — an open-set, text-prompted object detector — locates the helipad marker and returns a bounding box and centroid. The centroid is back-projected to geographic coordinates and compared against the registry coordinate to produce `vlm_offset_m`. Records where the VLM finds no helipad, or where offset exceeds a design-class threshold, are flagged for manual review.

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

---

## Formal ML Problem Statement

HIE combines three validation signals — visual (imagery-based object detection), textual (LLM search grounding), and structured (ADIP registry). This section covers both components.

### M3 — Visual Helipad Detection (Primary)

The core M3 ML task is **conformal helipad identification from satellite imagery**: given a 768×768 px ESRI World Imagery chip centred on a candidate coordinate, confirm whether a helipad is visually present and localise it with a calibrated confidence bound.

*Conformal* means detection confidence scores are statistically calibrated against a held-out validation set so that the reported precision guarantee holds at the stated threshold — not just empirically observed but formally bounded.

| Element | Detail |
|---------|--------|
| **Task** | Object detection — locate and classify helipad marker in satellite imagery chip |
| **Input** | 768×768 px ESRI World Imagery tile (zoom 19, ~0.22 m/px at lat 42°) |
| **Detection tiers** | Tier 1: OpenCV H-template matching (classical, <0.1 s); Tier 2: YOLOv8s fine-tuned (ML, ~0.1 s); Tier 3: Grounding DINO zero-shot (academic baseline only) |
| **Training data** | ~3,500 positive tiles from HelipadCAT coordinates re-fetched via ESRI; ~3,500 negative tiles (random non-helipad locations, same source) |
| **Geographic split** | Train/val: national FAA records outside NE US — Test: 747 NE US FAA records (never seen in training) |
| **Loss** | YOLOv8 default: box regression (CIoU) + classification (BCE) |

#### KPI Targets — M3

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Precision** | **> 0.95** | A false positive routes a passenger to a non-existent or unusable pad — unacceptable safety failure |
| **Recall** | **0.50 – 0.75** | A missed helipad is simply absent from routing options; no safety consequence, only reduced coverage |
| **mAP@50** | **> 0.90** | Standard YOLO object detection benchmark at IoU threshold 0.50 |

> **Design rationale:** SkyRoute deliberately optimises for precision over recall. The routing engine operates on the set of *confirmed* helipads only. A missed detection reduces the option set but never creates a hazardous routing suggestion. This asymmetry justifies a high-confidence detection threshold that may leave some valid pads unconfirmed until manual review clears them.

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
| M3 | 23 Jun 2026 | ⏳ PENDING | YOLOv8s helipad detector (Precision > 0.95, mAP@50 > 0.90), LLM status validation, XGBoost structured scoring, live detection overlay in app |
| Final | 21 Jul 2026 | ⏳ PENDING | Demo Day, stable URL, documented |

---

## Installation & Running

```bash
pip install -r requirements.txt

# Fetch data (one-time)
python scripts/fetch_ny_data.py        # ~2 min — FAA + OSM
python scripts/fetch_adip_details.py   # ~6 min — ADIP enrichment

# Launch dashboard
streamlit run app.py
```

---

## Repository Structure

```
ziv/
├── README.md           ← This document
├── CLAUDE.md           ← Implementation guide for Claude Code agents
├── requirements.txt    ← Pinned dependencies (Python 3.11+)
├── .gitignore
├── .env.example        ← Secrets template (ANTHROPIC_API_KEY, FAA_API_KEY)
├── app.py              ← Streamlit dashboard (~2700 lines)
├── src/
│   ├── data.py         ← Data ingestion, cleaning, schema normalisation
│   ├── analysis.py     ← Cross-source matching, consistency, completeness
│   └── model.py        ← Feature engineering & ML (stub — M3)
├── scripts/
│   ├── fetch_ny_data.py        ← Download FAA + OSM raw data
│   └── fetch_adip_details.py   ← Enrich FAA records via ADIP API
├── assets/
│   └── helipad_grounding_dino.jpg  ← Satellite chip for HIE Phase 1 illustration
├── data/               ← Local data files (gitignored)
└── notebooks/
    └── 01_eda.ipynb    ← EDA notebook (M2 deliverable)
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
