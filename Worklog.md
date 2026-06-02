# SkyRoute — Worklog

Chronological log of development sessions: what was built, issues hit, and how they were resolved.
Update this file at the end of every session before committing.

**Format per entry:**
- **Done** — features built, code written, decisions made
- **Issues** — problems encountered during the session
- **Resolved** — how each issue was fixed

---

## 2026-05-17 — M1: Project Initialisation

**Commits:** `e3f3913`, `da5cf31`

### Done
- Initialised Git repository (`Technion-LBS-Course/ziv`)
- Created `README.md` with project description, persona (Miles Urban), and one-liner
- Created `app.py` stub (Streamlit entry point)
- Created `src/data.py` and `src/__init__.py`
- Created `requirements.txt` with initial dependencies

### Issues
- None recorded

---

## 2026-05-22 — M2: Cross-Source Matching Enhancements

**Commit:** `3deba21`

### Done
- Added helipad-class-based proximity thresholds to `src/analysis.py`:
  - R22 small (23 m = 1.5 × 50 ft FATO)
  - Hospital rooftop (27 m = 1.5 × 60 ft FATO)
  - Bell 206 medium (32 m = 1.5 × 70 ft FATO)
  - S-92 large (80 m = 1.5 × 175 ft FATO)
- Added OSM name suffix stripping in `_name_sim()` — removes "Helipad"/"Heliport" from OSM names before similarity comparison unless the FAA name also contains the keyword

### Issues
- OSM contributors routinely append "Helipad" or "Heliport" to names that FAA records without the suffix, causing artificially low name similarity scores

### Resolved
- Regex `_HELI_SUFFIX_RE` strips the suffix from OSM names pre-comparison; conditional on FAA name not also containing a helipad keyword

---

## 2026-06-01 — M2: Dashboard, ADIP Enrichment, HIE Pipeline

**Commits:** `a5ffabe`, `ab30799`, `867cb75`, `3e84a34`

### Done

**Data pipeline:**
- Implemented `scripts/fetch_adip_details.py` — enriches 747 FAA records via ADIP Airport Master Record API, adds 16 columns (`adip_status`, `adip_lat`, `adip_lon`, `arp_method`, `last_info_days_ago`, etc.)
- Updated `src/data.py` — `load_faa_data()` auto-detects enriched CSV; ADIP coordinate upgrade for SURVEYED records; `adip_status` mapped to `operational` label

**Analysis module:**
- Implemented full `src/analysis.py`: `haversine_matrix()`, `match_by_faa_id()`, `match_by_proximity()`, `match_rate_by_threshold()`, `build_consistency_table()`, `faa_completeness()`, `osm_completeness()`

**Streamlit dashboard (`app.py`) — 4-tab layout:**
- Tab 1 (Problem): Miles Urban persona card (Bronxville NY, quote styled large/bold/centred), door-to-door journey comparison table, 4 KPI metrics, stakeholder ecosystem cards
- HIE ML Pipeline section: 3-phase flow banner (Grounding DINO → LLM → ADIP), Phase 1 with satellite chip illustration, Phase 2 with Caven Point USAR military-exclusion example, Phase 3 ADIP stretch goal
- Tab 2 (Literature): Quick Reference table (full titles, no truncation) + 4 expandable paper summaries with SkyRoute benefit callouts
- Tab 3 (Market): Competitor analysis
- Tab 4 (EDA & HIE): 3 inline data charts (field completeness, elevation consistency, location deviation) + KPI density map + routing simulator
- Routing simulator: layer-aware (only uses visible helipad layers), FAA + OSM + Business POIs + Executive Residences overlays, buttons repositioned bottom-right to avoid overlap
- Bookmarks: save/load helipad locations persisted to `data/bookmarks.json`
- ADIP hotlinks in every FAA marker popup

**Infrastructure:**
- Created `.gitignore` (excluded `data/`, `__pycache__/`, `.env`)
- Created `.env.example`
- Added `assets/` directory with `helipad_grounding_dino.jpg` and `miles_urban.jpg`
- Updated `CLAUDE.md` and `README.md` to reflect actual implementation
- Marked M2 as complete

### Issues

1. **`st.image()` TypeError** — `use_container_width` not available in Streamlit 1.36
2. **Leaflet maps not rendering on tab load** — Streamlit hides inactive tabs with `display:none`; Leaflet initialises container as 0×0 and never resizes
3. **`components.html()` resize injection broke both maps** — injecting a cross-iframe resize script via a second `components.html()` component disrupted layout
4. **`__pycache__` and `data/` files committed** — no `.gitignore` existed, bytecode and CSV files were staged
5. **Commit message empty** — all lines in the commit template started with `#` (git comment prefix); message was blank
6. **`branca` missing from `requirements.txt`** — `from branca.element import Element` used in 3 places in `app.py` but not listed as a dependency
7. **Literature table truncating article titles** — `_t = _p["title"][:62] + "…"` applied in Quick Reference table

### Resolved

1. Changed `use_container_width=True` → `use_column_width=True`
2. Injected a self-contained polling script via `branca.element.Element` into each Folium map's own iframe: checks `window.map.getSize()` every 400 ms, calls `map.invalidateSize(true)` when dimensions are 0, stops after 50 ticks (20 sec)
3. Removed the `components.html()` resize injection entirely; the in-iframe polling script is sufficient
4. Created `.gitignore`; un-staged bytecode and data files with `git restore --staged`
5. Used `git commit -m "$(cat <<'EOF' ... EOF)"` heredoc syntax to pass multi-line message
6. Added `branca>=0.8.0,<0.9.0` to `requirements.txt`
7. Removed truncation — replaced `_t` variable with `_p["title"]` directly in the table row string

---

## 2026-06-02 — M3: Planning, YOLO Dataset Architecture, README Update

**Commit:** `54ff0d1`

### Done

**M3 plan (`curious-hopping-lampson.md`):**
- Designed full HIE validation pipeline: three-tier visual detection cascade + Phase 2 LLM validation + XGBoost structured scoring
- Detection cascade: Tier 1 Classical CV (OpenCV H-template, H-marked pads only) → Tier 2 YOLOv8s fine-tuned → Tier 3 Grounding DINO (academic baseline)
- Researched HelipadCAT dataset (`github.com/jonasbtn/helipad_detection`): ~4,000 FAA-verified coordinates, images NOT pre-packaged (fetched from Google Maps), no negative examples, Mask RCNN annotations (not YOLO), same FAA source as existing data
- Decided to use HelipadCAT coordinates re-fetched via ESRI World Imagery (consistent with production, avoids Google Maps ToS risk)
- Designed YOLO training dataset pipeline: `scripts/build_yolo_dataset.py` — download helipadCAT CSV, deduplicate against NE US test records, fetch ESRI tiles, generate negative tiles, quality filter, convert to YOLO format, geographic train/val/test split
- Geographic contamination guard: train on national FAA (non-NE US), test exclusively on 747 NE US records
- Defined KPIs: Precision > 0.95, Recall 0.50–0.75, mAP@50 > 0.90

**README.md:**
- Updated Formal ML Problem Statement with two-part M3 formulation
- Added conformal identification framing (calibrated precision bounds)
- Added KPI table with precision/recall rationale (false positive = safety failure; miss = reduced routing options only)
- Updated M3 sprint milestone description

**Devil's advocate review:**
- Identified Grounding DINO domain gap (trained on terrestrial, not aerial imagery)
- Identified ground truth circularity (adip_status used as label and validation source)
- Identified XGBoost dataset too small (~35 non-operational examples out of 747)
- Identified regulatory gaps: ADIP key is undocumented government API (CFAA risk), no NOTAM integration, ODbL share-alike implications
- NOTAM integration confirmed as post-M3 scope; documented as known gap in README

### Issues
- HelipadCAT images require Google Maps API (not ESRI) — domain shift risk if used directly
- HelipadCAT has zero negative examples — YOLO would overfit without them
- HelipadCAT has no explicit license; Google imagery has commercial-use ToS risk

### Resolved
- Use HelipadCAT coordinates only; re-fetch all tiles from ESRI for consistency
- Generate ~1 negative tile per positive tile via random sampling within 5 km of known pads
- License risk mitigated by not using their imagery, only their FAA-derived coordinates (public domain)
