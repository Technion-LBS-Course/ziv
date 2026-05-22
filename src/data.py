"""Data ingestion, cleaning, and merging for SkyRoute helipad data.

All public functions receive file paths and return DataFrames that conform to
the standard SkyRoute schema. No model logic lives here.
"""

import logging
import re
import uuid
from pathlib import Path

import pandas as pd

_DMS_RE = re.compile(r"^(\d{1,3})-(\d{2})-([\d.]+)([NSEWnsew])$")


def _dms_to_decimal(dms: str) -> float | None:
    """Parse an FAA DMS string (``DD-MM-SS.SSSSH``) to decimal degrees."""
    if not dms or not isinstance(dms, str):
        return None
    m = _DMS_RE.match(dms.strip())
    if not m:
        return None
    deg, mins, secs, hem = m.groups()
    val = float(deg) + float(mins) / 60 + float(secs) / 3600
    return -val if hem.upper() in ("S", "W") else val

log = logging.getLogger(__name__)

# ── file path constants ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FAA_PATH = DATA_DIR / "faa_helipads_raw.csv"
OSM_PATH = DATA_DIR / "osm_helipads_raw.csv"
OURAIRPORTS_PATH = DATA_DIR / "ourairports_raw.csv"

# ── standard schema ──────────────────────────────────────────────────────────
SCHEMA_COLS: list[str] = [
    "source", "skyroute_id", "name", "lat", "lon",
    "state", "ownership_type", "lighting", "elevation_ft",
    "source_agreement_count", "data_freshness_days", "operational",
]

# ── FAA ownership code → canonical label ─────────────────────────────────────
_FAA_OWNERSHIP: dict[str, str] = {
    "PU": "public",
    "PR": "private",
    "MA": "military",  # Army
    "MN": "military",  # Navy
    "MF": "military",  # Air Force
    "MQ": "military",  # Marines
    "MR": "military",  # Army Reserve
    "CG": "military",  # Coast Guard
}


# ── private helpers ───────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate found in df.columns (case-insensitive)."""
    upper_map = {c.upper(): c for c in df.columns}
    for name in candidates:
        if name.upper() in upper_map:
            return upper_map[name.upper()]
    return None


def _assign_skyroute_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a fresh UUID skyroute_id to every row."""
    df = df.copy()
    df["skyroute_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    return df


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder to SCHEMA_COLS, adding missing columns as pd.NA."""
    for col in SCHEMA_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[SCHEMA_COLS].copy()


# ── public loaders ────────────────────────────────────────────────────────────

def load_faa_data(path: Path = FAA_PATH) -> pd.DataFrame:
    """Load and standardise FAA heliport records from the ADDS-FAA ArcGIS export.

    Source: ADDS-FAA US_Airport FeatureServer (TYPE_CODE=HP).
    The fetch script pre-parses DMS coordinates into ``lat``/``lon`` columns.
    This loader handles both the pre-parsed columns and the raw DMS strings as
    a fallback so the file is usable even if the fetch script is re-run without
    the parsing step.

    ArcGIS field mapping used:
        NAME        → name
        lat / LATITUDE (DMS) → lat (decimal degrees)
        lon / LONGITUDE (DMS) → lon (decimal degrees)
        STATE       → state
        ELEVATION   → elevation_ft  (already in feet)
        PRIVATEUSE  → ownership_type: 0=public, 1=private
        MIL_CODE    → ownership_type: non-empty overrides to military
        OPERSTATUS  → operational: "O" or empty = 1, other = 0

    Args:
        path: Path to the raw FAA helipad CSV.

    Returns:
        DataFrame with standard SkyRoute schema columns. Rows with no
        valid lat/lon are dropped.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"FAA data not found at {path}.\n"
            "Run:  python scripts/fetch_ny_data.py\n"
            "Coverage: NY, NJ, CT, PA, MA"
        )

    raw = pd.read_csv(path, dtype=str, low_memory=False)
    log.info("FAA raw: %d rows × %d cols", *raw.shape)

    # ── coordinates ───────────────────────────────────────────────────────────
    # fetch_ny_data.py adds pre-parsed 'lat'/'lon' columns; use them if present,
    # otherwise fall back to parsing the DMS LATITUDE/LONGITUDE strings.
    if "lat" in raw.columns and "lon" in raw.columns:
        lat_series = pd.to_numeric(raw["lat"], errors="coerce")
        lon_series = pd.to_numeric(raw["lon"], errors="coerce")
        log.info("Using pre-parsed lat/lon columns")
    else:
        lat_dms_col = _find_col(raw, ["LATITUDE"])
        lon_dms_col = _find_col(raw, ["LONGITUDE"])
        if lat_dms_col is None or lon_dms_col is None:
            raise RuntimeError(
                "Cannot find coordinate columns.\n"
                f"Available columns: {raw.columns.tolist()}"
            )
        lat_series = raw[lat_dms_col].apply(_dms_to_decimal)
        lon_series = raw[lon_dms_col].apply(_dms_to_decimal)
        log.info("Parsed DMS coordinates from LATITUDE/LONGITUDE columns")

    # ── ownership ─────────────────────────────────────────────────────────────
    # PRIVATEUSE: "0" = public use, "1" = private use
    # MIL_CODE: non-empty string → military
    mil_col     = _find_col(raw, ["MIL_CODE"])
    private_col = _find_col(raw, ["PRIVATEUSE"])

    if mil_col and private_col:
        is_military = raw[mil_col].notna() & (raw[mil_col].str.strip() != "")
        is_private  = raw[private_col].str.strip().isin(["1", "True", "true"])
        ownership   = pd.Series("public", index=raw.index)
        ownership[is_private]  = "private"
        ownership[is_military] = "military"
    elif private_col:
        ownership = raw[private_col].str.strip().map(
            {"0": "public", "1": "private"}
        ).fillna("private")
    else:
        ownership = pd.Series("private", index=raw.index)

    # ── operational status ────────────────────────────────────────────────────
    # OPERSTATUS: "O" = operational; "CI" = closed indefinitely; empty = unknown
    status_col = _find_col(raw, ["OPERSTATUS"])
    if status_col:
        closed = raw[status_col].str.strip().str.upper().isin(["CI", "CP", "CLOSED"])
        operational = (~closed).astype(int)
    else:
        operational = pd.Series(1, index=raw.index)

    # ── assemble output ───────────────────────────────────────────────────────
    name_col  = _find_col(raw, ["NAME", "ARPTNAME"])
    state_col = _find_col(raw, ["STATE"])
    elev_col  = _find_col(raw, ["ELEVATION"])

    out = pd.DataFrame(index=raw.index)
    out["source"]                 = "faa"
    out["name"]                   = raw[name_col].str.strip() if name_col else pd.NA
    out["lat"]                    = lat_series
    out["lon"]                    = lon_series
    out["state"]                  = raw[state_col].str.strip().str.upper() if state_col else pd.NA
    out["elevation_ft"]           = pd.to_numeric(raw[elev_col], errors="coerce") if elev_col else pd.NA
    out["lighting"]               = False          # not available in ArcGIS export
    out["ownership_type"]         = ownership
    out["data_freshness_days"]    = 0              # fetched today
    out["source_agreement_count"] = 1
    out["operational"]            = operational

    out = _assign_skyroute_ids(out)
    out = _enforce_schema(out)
    out = out.dropna(subset=["lat", "lon"])
    log.info("FAA standardised: %d records", len(out))
    return out


def load_osm_data(path: Path = OSM_PATH) -> pd.DataFrame:
    """Load and standardise OSM aeroway=helipad records.

    Reads the flat CSV produced by ``scripts/fetch_ny_data.py``, where every
    OSM tag becomes a column.  Maps to the SkyRoute schema.

    Args:
        path: Path to the raw OSM helipad CSV.

    Returns:
        DataFrame with standard SkyRoute schema columns. Rows with no
        valid lat/lon are dropped.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"OSM data not found at {path}.\n"
            "Run:  python scripts/fetch_ny_data.py\n"
            "Coverage: NY, NJ, CT, PA, MA"
        )

    raw = pd.read_csv(path, dtype=str, low_memory=False)
    log.info("OSM raw: %d rows × %d cols", *raw.shape)

    name_col       = _find_col(raw, ["name", "operator"])
    ele_col        = _find_col(raw, ["ele", "elevation"])
    lit_col        = _find_col(raw, ["lit", "lighting"])
    addr_state_col = _find_col(raw, ["addr:state", "addr_state"])
    access_col     = _find_col(raw, ["operator:type", "access", "ownership"])

    out = pd.DataFrame(index=raw.index)
    out["source"] = "osm"
    out["name"]   = raw[name_col].str.strip() if name_col else pd.NA
    out["lat"]    = pd.to_numeric(raw["lat"], errors="coerce")
    out["lon"]    = pd.to_numeric(raw["lon"], errors="coerce")
    out["state"]  = raw[addr_state_col].str.strip().str.upper() if addr_state_col else pd.NA

    if ele_col:
        # OSM convention: elevation in metres — convert to feet
        out["elevation_ft"] = pd.to_numeric(raw[ele_col], errors="coerce") * 3.28084
    else:
        out["elevation_ft"] = pd.NA

    if lit_col:
        out["lighting"] = raw[lit_col].str.lower().str.strip().isin(
            ["yes", "1", "true", "24/7", "dusk-dawn"]
        )
    else:
        out["lighting"] = False

    if access_col:
        _osm_own = {"government": "public", "military": "military",
                    "private": "private", "public": "public"}
        out["ownership_type"] = (
            raw[access_col].str.lower().str.strip()
            .map(_osm_own)
            .fillna("private")
        )
    else:
        out["ownership_type"] = "private"

    # OSM data was fetched today, so freshness = 0 days
    out["data_freshness_days"]  = 0
    out["source_agreement_count"] = 1
    out["operational"]          = 1

    out = _assign_skyroute_ids(out)
    out = _enforce_schema(out)
    out = out.dropna(subset=["lat", "lon"])
    log.info("OSM standardised: %d records", len(out))
    return out


def load_ourairports_data(path: Path = OURAIRPORTS_PATH) -> pd.DataFrame:
    """Load and standardise OurAirports heliport records.

    Args:
        path: Path to airports.csv downloaded from ourairports.com,
              pre-filtered to ``type == 'heliport'``.

    Returns:
        DataFrame with standard SkyRoute schema columns.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"OurAirports data not found at {path}.\n"
            "Download: curl -o data/ourairports_raw.csv "
            "https://ourairports.com/data/airports.csv"
        )

    raw = pd.read_csv(path, dtype=str, low_memory=False)
    raw = raw[raw["type"].str.strip() == "heliport"].copy()
    log.info("OurAirports heliports: %d rows", len(raw))

    out = pd.DataFrame(index=raw.index)
    out["source"]       = "ourairports"
    out["name"]         = raw["name"].str.strip()
    out["lat"]          = pd.to_numeric(raw["latitude_deg"], errors="coerce")
    out["lon"]          = pd.to_numeric(raw["longitude_deg"], errors="coerce")
    out["state"]        = (
        raw.get("iso_region", pd.Series(dtype=str))
        .str.replace("US-", "", regex=False)
        .str.strip()
    )
    out["elevation_ft"] = pd.to_numeric(
        raw.get("elevation_ft", pd.Series(dtype=str)), errors="coerce"
    )
    out["lighting"]              = False
    out["ownership_type"]        = "private"
    out["data_freshness_days"]   = pd.NA
    out["source_agreement_count"] = 1
    out["operational"]           = 1

    out = _assign_skyroute_ids(out)
    out = _enforce_schema(out)
    out = out.dropna(subset=["lat", "lon"])
    log.info("OurAirports standardised: %d records", len(out))
    return out


# ── merge ─────────────────────────────────────────────────────────────────────

def merge_helipad_sources(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Merge standardised helipad DataFrames with geospatial deduplication.

    Clusters records within 100 m radius (haversine); keeps the record with
    the highest ``source_agreement_count``.  Tie-break priority:
    faa > ourairports > osm.

    Args:
        *dfs: Two or more DataFrames in the standard SkyRoute schema.

    Returns:
        Deduplicated DataFrame with updated ``source_agreement_count``.
    """
    # Simple concat for now — spatial deduplication implemented in M2 session
    merged = pd.concat(list(dfs), ignore_index=True)
    log.info(
        "Merged (pre-dedup): %d records from %d source(s)",
        len(merged), len(dfs)
    )
    return merged
