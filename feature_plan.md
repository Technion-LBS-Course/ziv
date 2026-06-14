# SkyRoute — Feature Plan

> Derived from CLAUDE.md and SPRINT_PLAN.md.
> Organized by milestone. Each feature lists the owning file, inputs/outputs, and acceptance criteria.

---

## M2 — Real Data + Baseline (Due: 02 Jun 2026)

### F-01 · FAA Data Loader
**File:** `src/data.py` — `load_faa_data(path)`

| | |
|---|---|
| Input | `data/faa_apt.csv` (fixed-width NASR APT download) |
| Output | DataFrame with standard SkyRoute schema (12 columns) |
| Key logic | Parse fixed-width fields, map ownership codes to enum strings, compute `data_freshness_days` from cycle date |
| Done when | Returns clean DataFrame; raises `FileNotFoundError` if path missing; no nulls in `lat`, `lon`, `state` |

---

### F-02 · Helipad Merge & Deduplication
**File:** `src/data.py` — `merge_helipad_sources(*dfs)`

| | |
|---|---|
| Input | 2 DataFrames (FAA, OSM) |
| Output | Single deduplicated DataFrame; `source_agreement_count` column populated |
| Key logic | Cluster records within 100 m radius (haversine); keep record with highest `source_agreement_count`; tie-break: FAA > OSM |
| Done when | No two records are within 100 m of each other; `source_agreement_count` in range [1, 2] |

---

### F-04 · EDA Notebook
**File:** `notebooks/01_eda.ipynb`

| Chart | Library | What it shows |
|---|---|---|
| Class distribution | Plotly bar | Count of `operational = 1` vs `0` |
| Missing value heatmap | seaborn / Plotly | % missing per column |
| Feature correlation matrix | Plotly heatmap | Pearson r between numeric features |
| Geographic scatter | Folium | All helipads on US basemap, colored by `operational` |

Done when: all 4 cells execute without error on the merged dataset.

---

### F-05 · Majority-Class Baseline
**File:** `src/model.py` — `compute_majority_baseline(y_test)` *(stub already exists)*

| | |
|---|---|
| Input | `y_test` Series |
| Output | `float` — F1 score of a classifier that predicts all-1 |
| Done when | Returns same value as `sklearn.metrics.f1_score(y_test, np.ones_like(y_test))` |

---

### F-06 · Streamlit Dashboard — Map + Charts
**File:** `app.py`

| Component | Library | Details |
|---|---|---|
| Helipad map | Folium + streamlit-folium | Markers colored by `operational`; tooltip shows name + state |
| Class distribution chart | Plotly bar | Mirrors EDA notebook chart |
| State coverage chart | Plotly bar | Helipad count per US state |
| Missing value % chart | Plotly bar | % null per feature column |
| Baseline F1 | Streamlit sidebar | `st.metric("Majority-class F1", value)` |

Done when: `streamlit run app.py` loads without error on real merged data and renders all 5 components.

---

## M3 — ML Model + Deployment (Due: 23 Jun 2026)

### F-07 · Feature Engineering
**File:** `src/model.py` — `build_features(df)`

| Feature | Type | Encoding |
|---|---|---|
| `ownership_type_encoded` | categorical | OrdinalEncoder (public / private / hospital / military) |
| `lighting_encoded` | binary | 0 / 1 |
| `elevation_ft` | float | fill nulls with state median |
| `source_agreement_count` | int | passthrough |
| `data_freshness_days` | float | passthrough |

Output: DataFrame with exactly these 5 columns (no leakage columns).
Done when: `build_features(df).columns.tolist() == FEATURE_COLS` and no NaNs remain.

---

### F-08 · XGBoost Classifier
**File:** `src/model.py` — `train_model(X_train, y_train)`

| | |
|---|---|
| Model | `xgboost.XGBClassifier` |
| Imbalance handling | `scale_pos_weight = neg_count / pos_count` |
| Hyperparameters | `n_estimators=300`, `max_depth=6`, `learning_rate=0.05` (tunable) |
| Output | Fitted `XGBClassifier` object |
| Done when | Returns fitted model; no `accuracy_score` used anywhere in this function |

---

### F-09 · Model Evaluation
**File:** `src/model.py` — `evaluate_model(model, X_test, y_test)`

| | |
|---|---|
| Output | `dict` with keys: `f1` (float), `report` (str), `confusion_matrix` (ndarray) |
| Primary metric | `sklearn.metrics.f1_score(..., average="binary")` |
| Done when | `result["f1"] > compute_majority_baseline(y_test)` on held-out test set |

**Stretch goal:** val F1 >= 0.70

---

### F-10 · Random Forest Comparison
**File:** `src/model.py` — `train_rf_model(X_train, y_train)`

| | |
|---|---|
| Model | `sklearn.ensemble.RandomForestClassifier` |
| Purpose | Comparison baseline against XGBoost |
| Done when | Evaluated with same `evaluate_model()` function; F1 reported alongside XGBoost in app |

---

### F-11 · Live Inference in Streamlit
**File:** `app.py`

| Component | Details |
|---|---|
| Model load | Load fitted model from `models/xgb_model.pkl` (cached with `@st.cache_resource`) |
| Usability score slider | `st.slider("Min usability score", 0.0, 1.0, 0.5)` filters map markers |
| Per-helipad score | `model.predict_proba(X)[:, 1]` — displayed as tooltip on map marker |
| Done when | Slider filters markers in real time; sidebar shows model F1 vs baseline F1 |

---

### F-12 · OSM Geospatial Enrichment
**File:** `src/data.py` — `enrich_geospatial(df)`

| Feature | Source | Logic |
|---|---|---|
| `dist_to_hospital_km` | OSM Overpass | Nearest node with `amenity=hospital`; haversine distance |
| `dist_to_city_center_km` | OSM Overpass | Nearest node with `place=city` or `place=town` |
| `population_density` | OSM / Census | 1 km radius estimate |

Done when: columns added to merged_df without NaNs; `build_features()` updated to include them.
**Note: do NOT start this before M2 is submitted.**

---

### F-13 · Streamlit Cloud Deployment

| Step | Detail |
|---|---|
| Secrets | Set `FAA_API_KEY`, `MAPBOX_TOKEN` in Streamlit Cloud dashboard (not in code) |
| Data | Upload processed `data/helipad_merged.parquet` as a Streamlit secret or GitHub LFS (decide at deploy time) |
| URL | Update README with live public URL after deploy |
| Done when | App loads at public URL with no local file dependencies |

---

## Post-M3 / Stretch

| Feature | Description | Blocked on |
|---|---|---|
| NOTAM integration | Pull active NOTAMs from FAA live feed; flag temporarily closed helipads on map | `FAA_API_KEY` + M3 done |
| Multimodal route planner | Given origin + destination, compute optimal helipad-to-helipad path with ground legs | M3 done |
| Operator booking stub | Deep-link to Blade/Joby booking for selected route | M3 done |
| Google Earth Engine layer | Satellite imagery validation of helipad surface condition | `GEE_PROJECT_ID` + post-Final |

---

## Schema Reference

Every loader in `src/data.py` must output these columns in this order:

| Column | Type | Notes |
|---|---|---|
| `source` | str | `"faa"` / `"osm"` |
| `skyroute_id` | str | UUID assigned at merge time |
| `name` | str | Raw name from source |
| `lat` | float | WGS-84 |
| `lon` | float | WGS-84 |
| `state` | str | 2-letter US state code |
| `ownership_type` | str | `public` / `private` / `hospital` / `military` |
| `lighting` | bool | Has lighting |
| `elevation_ft` | float | Nullable before imputation |
| `source_agreement_count` | int | 1–3 |
| `data_freshness_days` | float | Days since last update |
| `operational` | int | Binary label: 1 = operational, 0 = unreliable |

---

## Metric Guardrails

- **Primary metric everywhere:** F1-score (binary)
- **Never report accuracy as primary** — class imbalance makes it misleading
- Test F1 must beat majority-class baseline — this is a hard requirement, not a stretch goal
- Stretch: val F1 >= 0.70
