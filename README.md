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

| Element | Detail |
|---------|--------|
| **Task** | Binary classification — helipad operational (1) vs. unreliable/decommissioned (0) |
| **Input X** | Structured registry fields + geospatial enrichment (see Feature List in `../CLAUDE.md`) |
| **Target y** | `operational` — binary label derived from ADIP status or OPERSTATUS field |
| **Loss** | Binary cross-entropy |
| **Primary metric** | F1-score — **never use accuracy** (class imbalance: most raw records are labeled "active") |
| **Split** | 70% train / 15% val / 15% test, stratified by U.S. state + ownership type |
| **Baseline 1** | Majority-class classifier (predict all operational) |
| **Baseline 2** | Logistic regression on structured fields only |
| **Primary model** | XGBoost (M3) |
| **Comparison model** | Random Forest (M3) |

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
| M2 | 02 Jun 2026 | 🔄 IN PROGRESS | Real data in app, 3 EDA charts, baseline F1, density KPI map |
| M3 | 23 Jun 2026 | ⏳ PENDING | XGBoost model, VLM overlay, OSM enrichment, Streamlit Cloud |
| Final | 21 Jul 2026 | ⏳ PENDING | Demo Day, stable URL, documented |

**M2 remaining:**
- [ ] `tests/test_smoke.py`
- [ ] `notebooks/01_eda.ipynb` (4 charts: class distribution, missing value heatmap, correlation matrix, geographic scatter)
- [ ] `merge_helipad_sources()` spatial deduplication (currently a simple concat)

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
