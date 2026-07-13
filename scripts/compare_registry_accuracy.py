"""Compare FAA vs OSM coordinate accuracy and quantify OSM network expansion.

Part 1 — Registry Accuracy
    For matched FAA+OSM pairs where YOLO detected a helipad, compare which
    registry coordinate is spatially closer to the YOLO-detected centre.
    Uses inspector_results.csv (pre-computed detections) — no YOLO re-inference.

Part 2 — OSM Network Expansion
    Quantifies how HIE-validated OSM-only helipads expand the NE US routing
    network relative to the FAA-only baseline. Samples 2,000 random origin
    points and measures nearest-helipad distances under both scenarios.

Usage:
    python scripts/compare_registry_accuracy.py              # full run
    python scripts/compare_registry_accuracy.py --limit 20  # smoke test (Part 1)
    python scripts/compare_registry_accuracy.py --flag-dist 80
    python scripts/compare_registry_accuracy.py --skip-accuracy
    python scripts/compare_registry_accuracy.py --skip-expansion
"""
import argparse
import ast
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.analysis import haversine_matrix, match_by_faa_id, match_by_proximity
from src.hie import bbox_px_to_latlon

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
_DATA      = _ROOT / "data"
_OSM_RAW   = _DATA / "osm_helipads_raw.csv"
_OSM_VAL   = _DATA / "osm_validated.csv"
_INSPECTOR = _DATA / "inspector_results.csv"
_OUT       = _DATA / "registry_accuracy.csv"

# ── NE US bounding box for random origin sampling ─────────────────────────────
_LAT_MIN, _LAT_MAX =  40.0, 45.5   # roughly New England + NJ/PA
_LON_MIN, _LON_MAX = -76.0, -70.0
_N_SAMPLE = 2_000
_ACCESS_KM = 5.0   # a pad is "accessible" if the nearest is ≤ this distance


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_bbox(val) -> list[int] | None:
    """Parse a bbox string '[x1, y1, x2, y2]' from CSV; return None if absent."""
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if s.lower() in ("nan", "", "none"):
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None


def _haversine_pair(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in metres between a single pair of points."""
    return float(haversine_matrix(
        np.array([lat1]), np.array([lon1]),
        np.array([lat2]), np.array([lon2]),
    )[0, 0])


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Registry coordinate accuracy
# ══════════════════════════════════════════════════════════════════════════════

def run_registry_accuracy(flag_dist_m: float, limit: int | None) -> pd.DataFrame:
    """Compare FAA vs OSM coordinate accuracy against YOLO-detected helipad centres.

    Args:
        flag_dist_m: Distance (m) above which FAA↔OSM pair may be different pads.
        limit: If set, process only the first N inspector rows (smoke test).

    Returns:
        DataFrame written to data/registry_accuracy.csv.
    """
    insp = pd.read_csv(_INSPECTOR)
    osm  = pd.read_csv(_OSM_RAW)

    if limit:
        insp = insp.head(limit)
        log.info("Smoke test: limited to %d chips", limit)

    # Minimal FAA df for matching — IDENT column required by match_by_faa_id
    faa = insp[["ident", "lat", "lon"]].rename(columns={"ident": "IDENT"})

    # Two-tier matching: FAA-ID exact → proximity fallback
    id_matches   = match_by_faa_id(faa, osm)
    prox_matches = match_by_proximity(
        faa, osm,
        threshold_m=250.0,
        exclude_faa_idx=set(id_matches["faa_idx"]) if not id_matches.empty else None,
    )
    all_matches = pd.concat([id_matches, prox_matches], ignore_index=True)
    log.info("FAA+OSM matched pairs: %d", len(all_matches))

    rows = []
    for _, m in all_matches.iterrows():
        faa_row = insp.loc[m["faa_idx"]]
        osm_row = osm.loc[m["osm_idx"]]

        bbox = _parse_bbox(faa_row["bbox_px"])
        if bbox is None:
            continue   # no YOLO detection — can't measure offset

        det_lat, det_lon = bbox_px_to_latlon(bbox, float(faa_row["lat"]), float(faa_row["lon"]))

        faa_lat = float(faa_row["lat"])
        faa_lon = float(faa_row["lon"])
        osm_lat = float(osm_row.get("lat", np.nan))
        osm_lon = float(osm_row.get("lon", np.nan))

        if np.isnan(osm_lat) or np.isnan(osm_lon):
            continue

        dist_faa     = _haversine_pair(faa_lat, faa_lon, det_lat, det_lon)
        dist_osm     = _haversine_pair(osm_lat, osm_lon, det_lat, det_lon)
        faa_osm_dist = _haversine_pair(faa_lat, faa_lon, osm_lat, osm_lon)

        rows.append({
            "faa_ident":          faa_row["ident"],
            "osm_id":             osm_row["osm_id"],
            "match_method":       m.get("match_method", ""),
            "faa_lat":            faa_lat,
            "faa_lon":            faa_lon,
            "osm_lat":            osm_lat,
            "osm_lon":            osm_lon,
            "det_lat":            round(det_lat, 6),
            "det_lon":            round(det_lon, 6),
            "dist_faa_m":         round(dist_faa, 1),
            "dist_osm_m":         round(dist_osm, 1),
            "faa_osm_dist_m":     round(faa_osm_dist, 1),
            "winner":             "FAA" if dist_faa <= dist_osm else "OSM",
            "flag_different_pad": bool(faa_osm_dist > flag_dist_m),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        log.warning("No matched+detected pairs found — check data files.")
        return df

    df.to_csv(_OUT, index=False)
    log.info("Saved %s  (%d rows)", _OUT.name, len(df))

    n         = len(df)
    faa_wins  = (df["winner"] == "FAA").sum()
    osm_wins  = (df["winner"] == "OSM").sum()
    flagged   = df["flag_different_pad"].sum()

    print("\n" + "=" * 62)
    print("  PART 1 -- REGISTRY COORDINATE ACCURACY")
    print("=" * 62)
    print(f"  Matched FAA+OSM pairs with YOLO detection  : {n}")
    print()
    print(f"  Mean  FAA -> detection                     : {df['dist_faa_m'].mean():.1f} m")
    print(f"  Mean  OSM -> detection                     : {df['dist_osm_m'].mean():.1f} m")
    print(f"  Median FAA -> detection                    : {df['dist_faa_m'].median():.1f} m")
    print(f"  Median OSM -> detection                    : {df['dist_osm_m'].median():.1f} m")
    print()
    print(f"  FAA coordinate more accurate               : {faa_wins:3d} / {n}  ({faa_wins/n*100:.0f}%)")
    print(f"  OSM coordinate more accurate               : {osm_wins:3d} / {n}  ({osm_wins/n*100:.0f}%)")
    print()
    print(f"  Flagged (FAA<->OSM > {flag_dist_m:.0f} m, may be != pad)  : {flagged}")
    print("=" * 62)

    return df


# ==============================================================================
# Part 2 — OSM network expansion
# ==============================================================================

def run_network_expansion() -> None:
    """Quantify how validated OSM-only helipads expand the NE US routing network.

    Samples _N_SAMPLE random origin points across the NE US bounding box and
    measures distance to the nearest helipad under two scenarios:
        Baseline  — FAA-only routing pool (747 pads)
        With HIE  — FAA + HIE-validated OSM-only pads (747 + 1,174)
    """
    insp    = pd.read_csv(_INSPECTOR)
    osm_val = pd.read_csv(_OSM_VAL)

    faa_pool = insp[["lat", "lon"]].dropna()
    osm_pool = osm_val[osm_val["hie_visual_detected"] == True][["lat", "lon"]].dropna()
    combined = pd.concat([faa_pool, osm_pool], ignore_index=True)

    log.info("FAA pool: %d  |  OSM validated: %d  |  Combined: %d",
             len(faa_pool), len(osm_pool), len(combined))

    faa_lat  = faa_pool["lat"].values
    faa_lon  = faa_pool["lon"].values
    comb_lat = combined["lat"].values
    comb_lon = combined["lon"].values

    rng      = np.random.default_rng(42)
    orig_lat = rng.uniform(_LAT_MIN, _LAT_MAX, _N_SAMPLE)
    orig_lon = rng.uniform(_LON_MIN, _LON_MAX, _N_SAMPLE)

    log.info("Computing nearest distances: %d origins × %d FAA pads …",
             _N_SAMPLE, len(faa_pool))
    d_faa_m  = haversine_matrix(orig_lat, orig_lon, faa_lat, faa_lon).min(axis=1)
    d_faa_km = d_faa_m / 1_000

    log.info("Computing nearest distances: %d origins × %d combined pads …",
             _N_SAMPLE, len(combined))
    d_comb_m  = haversine_matrix(orig_lat, orig_lon, comb_lat, comb_lon).min(axis=1)
    d_comb_km = d_comb_m / 1_000

    mean_faa  = d_faa_km.mean()
    mean_comb = d_comb_km.mean()
    pct_red   = (mean_faa - mean_comb) / mean_faa * 100

    median_faa  = float(np.median(d_faa_km))
    median_comb = float(np.median(d_comb_km))

    p90_faa  = float(np.percentile(d_faa_km, 90))
    p90_comb = float(np.percentile(d_comb_km, 90))

    within_faa  = (d_faa_km  <= _ACCESS_KM).mean() * 100
    within_comb = (d_comb_km <= _ACCESS_KM).mean() * 100

    # "Gained access" = was too far from any FAA pad, now within range of combined
    gained    = int(((d_faa_km > _ACCESS_KM) & (d_comb_km <= _ACCESS_KM)).sum())
    net_gain  = len(osm_pool) - (len(faa_pool) - int((insp["gt"] == 1).sum()))
    pct_inc   = (len(combined) / len(faa_pool) - 1) * 100

    print("\n" + "=" * 62)
    print("  PART 2 -- OSM NETWORK EXPANSION")
    print("=" * 62)
    print(f"  FAA-only routing pool (NE US)              : {len(faa_pool)}")
    print(f"  OSM-only validated pads added              : {len(osm_pool)}")
    print(f"  Combined routing pool                      : {len(combined)}  (+{pct_inc:.0f}%)")
    print()
    print(f"  Mean nearest pad  — FAA only               : {mean_faa:.2f} km")
    print(f"  Mean nearest pad  — FAA + OSM validated    : {mean_comb:.2f} km")
    print(f"  Mean distance reduction                    : {pct_red:.1f}%")
    print()
    print(f"  Median nearest    — FAA only               : {median_faa:.2f} km")
    print(f"  Median nearest    — FAA + OSM validated    : {median_comb:.2f} km")
    print()
    print(f"  90th-pct nearest  — FAA only               : {p90_faa:.2f} km")
    print(f"  90th-pct nearest  — FAA + OSM validated    : {p90_comb:.2f} km")
    print()
    print(f"  Origins within {_ACCESS_KM:.0f} km of any pad")
    print(f"    FAA only                                 : {within_faa:.1f}%")
    print(f"    FAA + OSM validated                      : {within_comb:.1f}%")
    print(f"  New origins that gained access             : {gained} / {_N_SAMPLE}  ({gained/_N_SAMPLE*100:.1f}%)")
    print("=" * 62 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Registry accuracy + OSM network expansion analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--flag-dist", type=float, default=50.0,
        help="Distance (m) above which FAA↔OSM pair is flagged as possibly ≠ pad (default 50)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Limit Part 1 to first N inspector rows — smoke test",
    )
    p.add_argument("--skip-accuracy",  action="store_true", help="Skip Part 1")
    p.add_argument("--skip-expansion", action="store_true", help="Skip Part 2")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.skip_accuracy:
        run_registry_accuracy(flag_dist_m=args.flag_dist, limit=args.limit)

    if not args.skip_expansion:
        run_network_expansion()


if __name__ == "__main__":
    main()
