# SkyRoute — Worklog

Chronological log of development sessions: what was built, issues hit, and how they were resolved.
Update this file at the end of every session before committing.

**Format per entry:**
- **Done** — features built, code written, decisions made
- **Issues** — problems encountered during the session
- **Resolved** — how each issue was fixed

---

## 2026-07-13 — Post-Final Bug Sprint: Cloud Stability, UX Fixes, Agent Memory

**Commits:** `b427c73` → `212a161`

### Done

**Streamlit Cloud segfault (torch CUDA stubs on CPU-only VM):**
- Pinned `torch==2.4.1+cpu` + `torchvision==0.19.1+cpu` from `download.pytorch.org/whl/cpu` — PyPI torch 2.5.x includes CUDA stub libraries that segfault at import on Streamlit Cloud's CPU server
- Added `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false` in `app.py` before all imports to prevent double-libgomp race between torch and opencv

**Weather card not showing on Cloud:**
- Root cause: `if _result.get("weather") and not _result.get("route"):` — the 8B Groq fallback model (used on Cloud when 70B quota exhausted) calls both `get_weather` and `compute_route` for weather queries; the guard hid the card whenever a route was also present
- Fix: removed `and not _result.get("route")` from the condition

**NWS wind range not converted to metric:**
- Root cause: `_mph_to_ms()` used `re.match()` anchored at start with `/([\d.]+)\s*mph/`; NWS returns ranges like `"3 to 13 mph SW"` where `"3"` is followed by `" to"`, not `" mph"` — match failed, raw string returned unchanged
- Also `_nws_detail_to_metric()` only converted the last number before `mph` in `"3 to 13 mph"`, producing `"3 to 5.8 m/s"` instead of `"1.3 to 5.8 m/s"`
- Fix: added range pattern `/([\d.]+)\s+to\s+([\d.]+)\s*mph/` to both functions; range sub runs before single-value sub

**Booking confirmation ID mismatch:**
- Root cause: LLM text got a random `"SKYxxxxxx"` from `simulate_rideshare()`; booking card independently computed `"SR-" + md5(origin+dest+total_min)[:6].upper()`
- Fix: `confirm_booking` tool now computes the same deterministic hash as `_render_quick_itinerary()` and returns it as `confirmation_id`

**QR code missing on Streamlit Cloud:**
- Root cause: `assets/qr_github.png` was not committed to git (existed only locally)
- Fix: generated and committed `assets/qr_github.png`; added `qrcode[pil]>=7.4.0` to `requirements.txt`; added dynamic generation fallback in `_render_quick_itinerary()` when the file is absent

**ADIP link missing from routing card helipads:**
- Root cause: `_basic()` in `build_quick_booking_legs()` already populated `adip_url` in the pad dict, but `_pad_links()` in `_render_quick_itinerary()` never read it
- Fix: one-line addition to `_pad_links()` to emit `📋 ADIP` link for FAA helipads that have an IDENT

**FAA helipad preference over OSM:**
- Root cause: `find_nearest_helipad()` returned the geometrically closest helipad regardless of source; when an OSM helipad and FAA helipad describe the same physical pad, OSM was sometimes chosen, losing ADIP/METAR resolution
- Fix: after finding the closest candidate, if it is OSM and an FAA/IDENT entry exists within 50 m, the FAA entry is preferred

**POI search missing nearby restaurants (Willis Marine Center / Italian):**
- Root cause 1: default `radius_m=500` physically excluded Piccolo Restaurant at 869 m before any category filter ran
- Root cause 2: `categorySet=7315003` (Italian sub-category) is absent from most small local restaurant records in TomTom; generic restaurant category (7315) was needed
- Fix: increased default radius to 1500 m, cap to 3000 m; added zero-result fallback that retries cuisine sub-category searches with `categorySet=7315` (parent restaurant) when the specific code returns nothing

**Route memory across conversation turns:**
- Root cause: `result["route"]` reset to `None` on every `run_agent_v2()` call; "book it" messages after weather/POI follow-ups had no cached route and either failed (empty geocode args) or recomputed unnecessarily
- Fix 1 (Python): `run_agent_v2()` now accepts `last_route`, `last_origin`, `last_destination`; `app.py` passes these from `st.session_state["_agent_last_route"]`; `result{}` is pre-seeded, so `confirm_booking` skips geocoding and recompute entirely
- Fix 2 (LLM): when a route exists, `location_hint` upgraded from neutral "last known origin/dest" to an explicit instruction — "call `confirm_booking(origin='X', destination='Y')` directly — do NOT call compute_route again" — targeting 8B model failures on Cloud

**Test plan:**
- Created `SkyRoute_Agent_Test_Plan.xlsx` — 9 tabs (A–I), 55 test cases covering routing nominal/boundary, booking flow, weather, POI search, off-topic guardrails, bad inputs, multi-turn context, unit conversions

### Issues

- Streamlit Cloud WebSocket shows "Connecting…" → "Running…" during Groq API calls (~10–30s): Groq blocks the Streamlit server thread; Cloud proxy (CloudFlare) times out the WebSocket. No code fix available without async refactor — documented as known Cloud latency behavior.

### Resolved

See individual fix descriptions above. All 8 bugs fixed and deployed in this session.

---

## 2026-06-13 — M3 Final: 4-Model Comparison, Inspector Tab, Results Tab Polish

### Done

**4-model comparison (`scripts/compare_models.py`):**
- New unified evaluation script: runs YOLOv8s, YOLO11s, YOLO11m, RT-DETR-L on the same 747 NE US test chips using the same methodology from `train_yolo.py`
- Saves per-model metrics (`comparison_metrics.json`) and full confidence-sweep curve arrays (`comparison_curves.json`) to `models/plots_comparison/`
- Generates 4 comparison plots: radar chart (P/R/F1/Accuracy), PR curve, P vs conf, R vs conf
- Generates 5 per-model individual plots for each model: pr.png, f1_conf.png, precision_conf.png, recall_conf.png, confusion.png (consistent style across all 4 models)
- Final results on 747 test chips: YOLOv8s P=0.906/R=0.801/F1=0.850, RT-DETR-L P=0.907/R=0.815/F1=0.859, YOLO11s P=0.908/R=0.866/F1=0.887, **YOLO11m P=0.931/R=0.848/F1=0.888 (production model)**
- `--plots-only` flag regenerates all charts from cached JSON without re-running inference
- `--limit N` for smoke test; cache check requires both JSON files to exist

**XGBoost training (`scripts/train_xgboost.py`):**
- Implemented full training pipeline using 17 ADIP-derived features
- Fixed `results_df.gt` → `results_df["gt"]` (pandas method vs column access)
- Fixed feature importance lookup: `importance.get(name, 0.0)` using actual column names (not positional `f0`/`f1` aliases)
- Final result: P=0.74 · R=0.72 · F1=0.73 on test set
- Added `position_age_days` distribution chart and has_wind windsock image with YouTube link

**Streamlit app (`app.py`) — 6-tab layout finalised:**
- Added Inspector tab (🔍) and Results tab (📈) to the tab bar
- **Inspector tab**: `@st.experimental_fragment` isolation prevents full-app reruns on widget interaction; auto-jump on dropdown selection (no button needed); NAIP chip + reference map side-by-side in 1:1 columns; Live Inference mode with FAA NE US / OSM NE US / FAA CONUS overlay layers as separate checkboxes
- **Results tab**: two sub-tabs — XGBoost Structured Baseline (first) / YOLO Models Comparison (second); radar chart comparison + PR curve side-by-side; action caption below radar describing model recommendation (YOLO11m = production, YOLO11s = discovery)
- Reference map uses CartoDB dark_matter default with OpenStreetMap light overlay; no ESRI satellite (was returning HTTP 500)
- Radar chart: zoomed polar axis with explicit inner-ring note; light/bright background; figure-level legend placed below chart (ncol=2, bbox_to_anchor=(0.5, 0.13)); note at y=0.07

**Bug fixes:**
- `_BE` NameError at Inspector line: `from branca.element import Element as _BE` moved to module level
- `_show_pil` / `_safe_image` NameError: both helper functions moved to before `with tab_inspector:`
- `use_container_width` TypeError on older Streamlit: `_show_pil` tries `use_container_width=True`, falls back to `use_column_width="always"` on TypeError
- OMP Error #15 (Windows OpenMP duplicate): `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at file top before all imports
- `torch.classes` Streamlit watcher error: `torch.classes.__path__ = []` after torch import
- ESRI `MapServer/export` unreliable (HTTP 500, timeout): rewrote `fetch_esri_chip()` to use individual XYZ tile stitching (3×3 grid at zoom 18, mosaic → crop → resize)
- `compare_models.py` KeyError: `compute_yolo_metrics()` returns `best_precision`/`best_recall`/`best_f1` not `precision`/`recall`/`f1`

**Infrastructure:**
- `.streamlit/config.toml` — `fileWatcherType = "watchdog"` to suppress torch.classes watch error

### Issues

1. **ESRI imagery fetch HTTP 500** — `MapServer/export` endpoint returned HTTP 500 for ~30% of requests, full timeouts for others
2. **OMP Error #15** — Windows duplicate OpenMP DLL (numpy + PyTorch + XGBoost all ship their own libiomp5md.dll)
3. **torch.classes watcher error** — Streamlit file-watcher attempts to inspect `torch.classes.__path__` which raises C++ exception in Python
4. **Radar chart legend obscuring upper portion** — `ax.legend()` placed at `bbox_to_anchor=(1.55, 1.18)` overlapped the chart area

### Resolved

1. Replaced `MapServer/export` with individual XYZ tile fetch at zoom 18 (3×3 grid of 256×256 tiles, stitched into 768×768 mosaic, cropped to exact 100m window, resized to 640×640) — same URL pattern as Folium base map tiles
2. `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` before any imports at top of `app.py`
3. `torch.classes.__path__ = []` after `import torch`; confirmed non-fatal (log noise only), optional `.streamlit/config.toml` `fileWatcherType = "watchdog"` as belt-and-suspenders
4. Switched to `fig.legend()` (figure-level) at `bbox_to_anchor=(0.5, 0.13)`, `ncol=2`; `fig.subplots_adjust(bottom=0.22)`; note moved up to `fig.text(0.5, 0.07, ...)`

---

## 2026-06-08 — Post-Training Analysis Planning

### Done
- Defined two post-training analysis scripts that run after final YOLO weights exist:
  1. **`scripts/compare_registry_accuracy.py`**: for each matched FAA+OSM pair, compare FAA vs OSM coordinate distance to YOLO-detected bbox centre. Outputs `data/registry_accuracy.csv`. Pairs >50m apart flagged as possibly different helipads.
  2. **`scripts/validate_osm_only.py`**: fetch NAIP chips for OSM-only NE US helipads (no FAA match), run `detect_helipad_cascade()`, write `data/osm_validated.csv`. Visually confirmed pads join the M4 routing pool.
- Defined 3-result academic comparison structure: Baseline / Preliminary YOLO (synthetic labels) / Final YOLO (fully verified test set). Preliminary results must be saved before retraining.
- Updated README.md: M3 Progress table with 5 new rows, M4 checklist with post-training prerequisites, repo structure with 2 new scripts.
- Updated CLAUDE.md: commands for both scripts, layout, M3 Still To Do rewrite, full architecture sections for both scripts.

### Issues
None — planning session only.

### Resolved
N/A

---

## 2026-06-07 — M4 Planning: NOTAM + RainViewer

### Done
- Defined M4 milestone (target 14 Jul 2026) with two new features:
  1. **NOTAM airspace avoidance**: `src/notam.py` — FAA Digital NOTAM API (by IDENT + bbox), Aviation Weather Center METAR/TAF, per-helipad active-NOTAM badge in popups, NOTAM overlay layer (TFR polygons + point markers), route weather summary panel (VFR/MVFR/IFR colour-coded).
  2. **RainViewer precipitation**: `src/weather.py` — fetch latest radar frame from `weather-maps.json`, add as toggleable Folium `TileLayer` (color 2, opacity 0.6), route precipitation check by sampling tile pixel at each waypoint, `st.warning` banner when intensity > 50.
- Updated README.md: M4 row in sprint table, new data sources section (FAA NOTAM API, AVN Weather Center, RainViewer), M4 task checklist.
- Updated CLAUDE.md: M4 commands, `src/notam.py` + `src/weather.py` in layout, full M4 architecture section with API URLs, non-obvious constraints, and `app.py` code patterns.

### Issues
None — planning session only.

### Resolved
N/A

---

## 2026-06-07 — M3: Zero-Shot Results, Training Script, Documentation

### Done
- Ran Grounding DINO on 5 test chips — fixed API break in transformers 4.51 (`box_threshold` removed from `post_process_grounded_object_detection`). Fix: remove `box_threshold=conf` from `detect_dino()` in `src/hie.py`. Now runs at ~5.6 s/chip, F1=0.00 as expected.
- Zero-shot ablation confirmed: all three models (Classical CV, YOLO-World, Grounding DINO) achieve F1=0.00 on 747 NAIP test chips. Domain gap quantified.
- Created `scripts/train_yolo.py`: trains YOLOv8s, evaluates on 747 test chips, computes registry-agreement baseline (FAA-ID matches < 10 m = TP, ≥ 10 m = FP, 50% OSM-only = FN), saves 5 plots to `models/plots/`.
- M3 course alignment confirmed: CV detection pipeline (3 algorithms + baseline + train/test split + Streamlit demo) satisfies the course CV track. XGBoost `src/model.py` is optional.
- OSM-only strategy: keep on map, not in test set; post-training run inference on OSM-only NAIP chips.
- Updated README.md, CLAUDE.md with current state (annotation 1966/4027, zero-shot results, new scripts).

### Issues
- Florence-2 incompatible with transformers ≥ 4.49 (two missing attributes: `forced_bos_token_id`, `_supports_sdpa`). Disabled by default.

### Resolved
- Grounding DINO: removed `box_threshold` parameter (renamed/removed in transformers 4.51).
- Florence-2: graceful fallback in `compare_zero_shot.py`; requires `pip install "transformers>=4.44,<4.49"` to enable.

---

## 2026-06-06 — M3: HIE Detection Module + Zero-Shot Comparison

### Done

**Zero-shot model selection (revised from original plan):**
- Original plan: Grounding DINO tiny as sole zero-shot baseline
- Research finding: Grounding DINO has documented poor zero-shot performance on nadir aerial imagery (arxiv 2601.22164) — accuracy improves only with domain adaptation
- New choice: **YOLO-World small** as primary zero-shot comparison (already in ultralytics install, <1s/chip, designed for small objects); **Florence-2-base** as secondary
- Grounding DINO retained as optional `--models dino` flag

**`src/hie.py` — HIE visual detection module:**
- `load_chip()` — load 640×640 NAIP chip from disk
- `detect_classical()` — OpenCV H-template bank (3 scales × 2 rotations × 2 colour variants), threshold 0.72
- `detect_yolo()` — YOLOv8s fine-tuned inference (Tier 2, uses `models/helipad_yolov8s.pt`)
- `detect_yolo_world()` — YOLO-World small zero-shot, classes: `["helipad", "landing pad", "H marking"]`
- `detect_florence2()` — Florence-2-base `<OPEN_VOCABULARY_DETECTION>helipad`
- `detect_dino()` — optional Grounding DINO tiny
- `detect_helipad_cascade()` — Tier1 → Tier2 production cascade
- `bbox_px_to_latlon()` and `compute_offset_m()` — coordinate utilities

**`scripts/compare_zero_shot.py` — ablation comparison:**
- Reads 747 NAIP test chips + GT labels
- Runs selected models, computes IoU vs GT, writes `data/yolo_dataset/zero_shot_results.csv`
- Prints Precision / Recall / F1 / mean latency table
- `--resume` flag for restartability; `--models` to pick subset; `--limit` for smoke tests
- Can run TODAY — no trained YOLO weights needed

**`requirements.txt` updated:** `transformers`, `timm`, `einops` moved from comments to active deps (Florence-2 requirements)

### Issues
None — smoke tests all passed on first run.

### Resolved
N/A

---

## 2026-06-03 — M3: YOLO Dataset Pipeline + Annotation Review Tool

### Done

**YOLO dataset pipeline (`scripts/build_yolo_dataset.py`):**
- 8-step resumable pipeline: HelipadCAT CSV → national FAA staleness filter → NE US geographic dedup → NAIP chip fetch → easy negatives → YOLO labels + split → HTML review galleries
- NAIP imagery source: USDA APFO `USDA_CONUS_PRIME/ImageServer/exportImage` — pure HTTP GET, no rasterio, no authentication
- 100m × 100m window → 640×640 px → effective GSD 0.156 m/px (between ESRI zoom 19 and 20)
- Per-latitude bbox scale factor `_bbox_pixel_scale(lat)`: converts HelipadCAT zoom-20 bboxes to NAIP chip space
- Retry logic: 3 attempts with 10s/20s/40s back-off for USDA server timeouts
- Dataset: ~5000 CONUS chips (positive + hard-negative + easy-negative) + 747 NE US test chips

**Annotation review tool (`scripts/annotate_dataset.py`):**
- Standalone Streamlit app: approve / disqualify / adjust bbox with live slider preview
- Decisions persisted to `data/yolo_dataset/review_decisions.csv` — survives Ctrl+C
- Approve auto-detects if sliders were moved and saves adjusted bbox
- Disqualify → empty YOLO label → chip becomes hard negative training example
- "Apply all decisions" writes final YOLO label files

**Infrastructure:**
- Fixed `.gitignore` inline comment bug (trailing `# comment` is NOT a git comment)
- Added `models/` directory with `.gitkeep`
- Updated CLAUDE.md and README.md

### Issues

1. **`ValueError: invalid literal for int() with base 10: 'other'`** — HelipadCAT `category` column has string values, not only integers
2. **Planetary Computer + rasterio**: rasterio COG reads fail on Windows (pip GDAL VSICURL issue with Azure Blob SAS tokens) — 41% failure rate on NE US test set
3. **ESRI zoom-20 gray tiles** — `USDA_Alaska_PRIME` service doesn't exist at `gis.apfo.usda.gov`; only `USDA_CONUS_PRIME` is available
4. **`data/yolo_dataset/` not gitignored** — 2000+ image chips showing as untracked files in VS Code ("too many active changes")
5. **`PermissionError` on label file in gallery** — OneDrive locks .txt files during sync
6. **`use_container_width`** in annotation tool — Streamlit 1.36.x uses `use_column_width`
7. **`IndexError: Cannot choose from an empty sequence`** in step6 — `--limit 20` hits Alaska-only records, zero positive chips

### Resolved

1. Wrapped `int(row.get("category", -1))` in `try/except (ValueError, TypeError)` → default -1
2. Replaced Planetary Computer + rasterio with USDA APFO `exportImage` endpoint (same `requests` session, no new dependencies)
3. `_naip_export_url()` returns `None` for lat outside 24–49.5°N → clean `NoImageryError`
4. Fixed `.gitignore`: moved inline comments to their own lines (`# comment` on a separate line above the pattern)
5. Wrapped `lbl_path.read_text()` in `try/except (PermissionError, OSError)` in gallery builder
6. Changed `use_container_width=True` → `use_column_width=True`
7. Added fallback: if no positive chips, seed easy negatives from all saved records; if still empty, skip step 6 gracefully

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
