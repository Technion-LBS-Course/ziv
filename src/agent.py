"""SkyRoute Route Assistant — LLM-powered natural language to multimodal routing.

Architecture (per course M4 spec):
  Step 1 — User speaks freely: "I need to be at X by 14:00"
  Step 2 — LLM (Groq/Llama) extracts structured parameters → JSON
  Step 3 — Python routing engine runs on those parameters (NOT the LLM)
  Step 4 — LLM formats the computed route as a friendly narrative

The LLM is a thin NLU layer only.  All routing math (haversine, helipad
selection, time estimation) is done by deterministic Python code.

Uses Groq (OpenAI-compatible).  Import: client.chat.completions.create
NOT anthropic.messages.create.
"""

import json
import logging
import math
import os
import pathlib
import random
import string
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ── ADIP remarks helpers ──────────────────────────────────────────────────────

_ADIP_AUTH    = "Basic 3f647d1c-a3e7-415e-96e1-6e8415e6f209-ADIP"
_ADIP_RAW_DIR = pathlib.Path(__file__).parent.parent / "data" / "adip_raw"
_adip_remarks_cache: dict = {}   # in-memory cache: ident → list[str]


def _fetch_adip_remarks(ident: str) -> list:
    """Return raw ADIP remark strings for a helipad.

    Checks in-memory cache → file cache → live API (in that order).
    Caches the full API response to data/adip_raw/<ident>.json for reuse.

    Args:
        ident: FAA IDENT code (e.g. "JRB").

    Returns:
        List of raw remark strings, empty list on any failure.
    """
    if ident in _adip_remarks_cache:
        return _adip_remarks_cache[ident]

    cache_path = _ADIP_RAW_DIR / f"{ident}.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            remarks = [r["remark"] for r in (data.get("remarks") or []) if r.get("remark")]
            _adip_remarks_cache[ident] = remarks
            return remarks
        except Exception:
            pass

    try:
        sess = requests.Session()
        sess.get("https://adip.faa.gov/agis/public/", timeout=8)
        resp = sess.post(
            "https://adip.faa.gov/agisServices/public-api/getAirportDetails",
            json={"locId": ident},
            headers={"Authorization": _ADIP_AUTH},
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()
        _ADIP_RAW_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        remarks = [r["remark"] for r in (data.get("remarks") or []) if r.get("remark")]
        _adip_remarks_cache[ident] = remarks
        return remarks
    except Exception as exc:
        log.warning("ADIP remarks fetch failed for %s: %s", ident, exc)
        _adip_remarks_cache[ident] = []
        return []


def _decode_adip_remarks(remarks: list, ident: str, name: str) -> str:
    """Decode cryptic FAA ADIP remarks into plain English using Groq/Llama.

    Falls back to the raw remark text if the LLM is unavailable.

    Args:
        remarks: List of raw remark strings from ADIP.
        ident: FAA IDENT code (used for context in the prompt).
        name: Helipad display name.

    Returns:
        Human-readable coordination note string.
    """
    if not remarks:
        return ""
    raw = " | ".join(remarks)
    client = _get_groq_client()
    if client is None:
        return f"Remarks: {raw}"
    prompt = (
        f"Decode these abbreviated FAA ADIP remarks for {name} ({ident}) into "
        f"clear plain-English operational notes for a helicopter dispatcher.\n\n"
        f"Raw: {raw}\n\n"
        f"Rules: expand ALL aviation abbreviations "
        f"(CD=Clearance Delivery, CTC=Contact, APCH=Approach Control, "
        f"PPR=Prior Permission Required, etc.). "
        f"Keep phone numbers exactly as written. "
        f"Output 1–3 concise sentences, no headers or bullet points."
    )
    try:
        resp = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.warning("Remarks decode LLM failed for %s: %s", ident, exc)
        return f"Remarks: {raw}"

_GROQ_MODEL         = "llama-3.1-8b-instant"
_GROQ_MODEL_FAST    = "meta-llama/llama-4-scout-17b-16e-instruct"  # if enabled in project settings

# ── Groq client (lazy — avoids ImportError if package not installed) ──────────

def _get_groq_client():
    """Return a Groq client, or None if package/key unavailable."""
    try:
        from groq import Groq  # pip install groq
    except ImportError:
        log.warning("groq package not installed — run: pip install groq")
        return None

    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            # Only access st.secrets when inside a valid Streamlit script run context.
            # Calling it outside that context raises "SessionInfo before initialized".
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                import streamlit as st
                key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    if not key:
        log.warning("GROQ_API_KEY not set — Route Assistant will use fallback text")
        return None

    return Groq(api_key=key)


# ── Step 2: Parameter extraction ─────────────────────────────────────────────

_EXTRACTION_SYSTEM = """You are a navigation parameter extractor for SkyRoute,
an air-mobility routing app covering the New York / New Jersey metro area.

Extract routing parameters from the user's free text.
Return ONLY valid JSON — no markdown, no explanation, just the JSON object.

Required keys:
{
  "origin_text": string or null,
  "destination_text": string or null,
  "arrival_time": "HH:MM" string (24-hour) or null,
  "departure_time": "HH:MM" string (24-hour) or null,
  "notes": string or null,
  "intent": "route" | "book" | "cancel" | "off_topic",
  "ignore_tfrs": boolean
}

intent rules:
- "route": user wants to find/plan a route (default)
- "book": user confirms they want to book ("yes", "book it", "confirm", "go ahead", "proceed")
- "cancel": user wants to cancel or start over
- "off_topic": user's message is unrelated to air travel, routing, or helipad queries
  (e.g. weather questions, general knowledge, creative writing, jokes, food, sports)
  IMPORTANT: when no clear origin/destination place name is present, always use "off_topic".
  Do NOT invent or infer a destination from unrelated context.

ignore_tfrs rules:
- Set to true when the user explicitly says they want to bypass, ignore, or override TFR restrictions
  (e.g. "ignore TFRs", "bypass restrictions", "route anyway", "override TFR", "I have a waiver")
- Default: false

Rules:
- If origin is not mentioned, set origin_text to null
- If time context implies arrival deadline ("meeting at 14:00"), use arrival_time
- notes: any preference the user mentioned ("prefer helicopter", "avoid Manhattan")
- Expand abbreviations: "NJ" → "New Jersey", "NYC" → "New York City"

Examples:
User: "Meeting in Greenwich at 3pm, leaving Penn Station"
Output: {"origin_text": "Penn Station, New York", "destination_text": "Greenwich, CT", "arrival_time": "15:00", "departure_time": null, "notes": null, "intent": "route", "ignore_tfrs": false}

User: "Route to JFK, ignore TFRs"
Output: {"origin_text": null, "destination_text": "JFK Airport, New York", "arrival_time": null, "departure_time": null, "notes": null, "intent": "route", "ignore_tfrs": true}

User: "What should I have for breakfast?"
Output: {"origin_text": null, "destination_text": null, "arrival_time": null, "departure_time": null, "notes": null, "intent": "off_topic", "ignore_tfrs": false}

User: "Who won the World Cup?"
Output: {"origin_text": null, "destination_text": null, "arrival_time": null, "departure_time": null, "notes": null, "intent": "off_topic", "ignore_tfrs": false}
"""

_FALLBACK_PARAMS: dict = {
    "origin_text": None,
    "destination_text": None,
    "arrival_time": None,
    "departure_time": None,
    "notes": None,
    "intent": "route",
    "ignore_tfrs": False,
}

# Simple booking-intent keywords — faster than an LLM call for clear confirmations
_BOOK_KEYWORDS = {"yes", "book", "confirm", "go ahead", "proceed", "reserve",
                  "do it", "ok", "okay", "sure", "book it", "let's go", "yep", "yeah"}

def is_booking_intent(user_text: str) -> bool:
    """Return True if the user's message clearly signals booking confirmation."""
    words = set(user_text.lower().split())
    return bool(words & _BOOK_KEYWORDS)


def extract_nav_params(user_text: str, history: list[dict] | None = None) -> dict:
    """Extract navigation parameters from free text using Groq/Llama.

    Args:
        user_text: Free-form user request in any language/style.
        history: Prior conversation turns as [{role, content}] pairs.
            The assistant turns should contain the previously extracted JSON
            so the LLM can resolve references like "earlier", "same destination".
            Capped internally to the last 6 messages (3 exchanges).

    Returns:
        Dict with keys: origin_text, destination_text, arrival_time,
        departure_time, notes. Falls back gracefully on any error.
    """
    client = _get_groq_client()
    if client is None:
        # No LLM available — treat the whole message as the destination (graceful degradation)
        return {**_FALLBACK_PARAMS, "destination_text": user_text}

    messages: list[dict] = [{"role": "system", "content": _EXTRACTION_SYSTEM}]
    if history:
        messages.extend(history[-6:])  # last 3 exchanges — enough context, bounded cost
    messages.append({"role": "user", "content": user_text})

    try:
        resp = client.chat.completions.create(
            model=_GROQ_MODEL,
            temperature=0,  # deterministic JSON output
            messages=messages,
        )
        raw = resp.choices[0].message.content.strip()
        params = json.loads(raw)
        for k in _FALLBACK_PARAMS:
            params.setdefault(k, None)
        return params
    except (json.JSONDecodeError, Exception) as exc:
        log.warning("Groq extraction failed: %s", exc)
        # LLM call failed — return empty params so the destination guard fires
        return {**_FALLBACK_PARAMS}


# ── Geocoding ─────────────────────────────────────────────────────────────────

def _extract_address_with_llm(place_text: str) -> str:
    """Strip business names / floor info, returning the bare geocodable address.

    E.g. "Enigma Technologies at 32 Mercer St 8th Fl, New York, NY 10013"
         → "32 Mercer St, New York, NY 10013"

    Falls back to the original text if LLM is unavailable.

    Args:
        place_text: Raw place description from the user.

    Returns:
        Cleaned address string.
    """
    client = _get_groq_client()
    if client is None:
        return place_text
    prompt = (
        "Extract only the geocodable street address from the text below. "
        "Remove business names, floor/suite numbers, and anything that is not "
        "part of the street address, city, state, or ZIP. "
        "Return ONLY the clean address, nothing else.\n\n"
        f"Text: {place_text}"
    )
    try:
        resp = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0,
        )
        cleaned = resp.choices[0].message.content.strip().strip('"').strip("'")
        return cleaned if cleaned else place_text
    except Exception:
        return place_text


def _geocode_tomtom(place_name: str) -> Optional[tuple[float, float]]:
    """Geocode via TomTom Fuzzy Search — handles business names and addresses.

    Biased toward the NY/NJ metro area (200 km radius from NYC centre).

    Args:
        place_name: Place name, business name, or street address.

    Returns:
        (lat, lon) tuple, or None if not found or key unavailable.
    """
    key = os.getenv("TOMTOM_API_KEY", "")
    if not key:
        return None
    try:
        import urllib.parse
        encoded = urllib.parse.quote(place_name)
        resp = requests.get(
            f"https://api.tomtom.com/search/2/search/{encoded}.json",
            params={
                "key": key,
                "limit": 1,
                "countrySet": "US",
                "lat": 40.75,   # NYC centre bias
                "lon": -73.98,
                "radius": 200_000,
            },
            timeout=8,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            pos = results[0]["position"]
            return float(pos["lat"]), float(pos["lon"])
    except Exception as exc:
        log.warning("TomTom geocode failed for '%s': %s", place_name, exc)
    return None


def _geocode_nominatim(place_name: str) -> Optional[tuple[float, float]]:
    """Geocode via Nominatim (OSM). Fallback when TomTom unavailable or fails.

    Args:
        place_name: Human-readable place name or address.

    Returns:
        (lat, lon) tuple, or None if not found.
    """
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": place_name,
                "format": "json",
                "limit": 1,
                "countrycodes": "us",
                "viewbox": "-76.0,42.0,-72.0,40.0",
                "bounded": 0,
            },
            headers={"User-Agent": "SkyRoute/1.0 (zivg@campus.technion.ac.il)"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as exc:
        log.warning("Nominatim geocode failed for '%s': %s", place_name, exc)
    return None


def geocode_place(place_name: str) -> Optional[tuple[float, float]]:
    """Geocode a place name or address to (lat, lon).

    Pipeline:
      1. Try TomTom (business names + addresses, best accuracy)
      2. If TomTom fails, strip business name via LLM then retry TomTom
      3. Fall back to Nominatim on the cleaned address

    Args:
        place_name: Human-readable place name, business, or address.

    Returns:
        (lat, lon) tuple, or None if all sources fail.
    """
    # 1. Try TomTom directly
    result = _geocode_tomtom(place_name)
    if result:
        return result

    # 2. LLM-clean the address, retry TomTom (handles "Business at 32 Mercer St …")
    cleaned = _extract_address_with_llm(place_name)
    if cleaned != place_name:
        log.info("Geocode: cleaned '%s' → '%s'", place_name, cleaned)
        result = _geocode_tomtom(cleaned)
        if result:
            return result

    # 3. Nominatim fallback on cleaned address
    return _geocode_nominatim(cleaned)


# ── TFR segment check ─────────────────────────────────────────────────────────

def _point_in_polygon(lat: float, lon: float, ring: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon test.

    Args:
        lat: Point latitude.
        lon: Point longitude.
        ring: Polygon ring as [[lat, lon], ...] (Folium-compatible).
    """
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][1], ring[i][0]   # lon, lat
        xj, yj = ring[j][1], ring[j][0]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


_HARD_TFR_CODES = {"NDA_TFR", "DEF", "SECURITY", "STADIUM"}

def _check_tfrs_on_segment(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    n_samples: int = 7,
) -> tuple[list[dict], list[dict]]:
    """Check whether the aerial segment intersects any active TFRs.

    Samples n_samples evenly-spaced points along the great-circle segment and
    tests each against all active TFR polygons.  Stadium TFRs use a 3 nm
    (5.6 km) radius check instead of polygon containment.

    Args:
        lat1: Departure helipad latitude.
        lon1: Departure helipad longitude.
        lat2: Arrival helipad latitude.
        lon2: Arrival helipad longitude.
        n_samples: Number of points to test along the segment.

    Returns:
        (hard_blocks, soft_warnings) — each a list of TFR dicts that were hit.
        hard_blocks: type codes in _HARD_TFR_CODES → booking must be blocked.
        soft_warnings: all other intersecting TFRs → show caution but allow.
    """
    try:
        from src.notam import fetch_active_tfrs
    except ImportError:
        return [], []

    try:
        tfrs = fetch_active_tfrs()
    except Exception as exc:
        log.warning("TFR fetch failed during route check: %s", exc)
        return [], []

    sample_pts = [
        (lat1 + (lat2 - lat1) * i / (n_samples - 1),
         lon1 + (lon2 - lon1) * i / (n_samples - 1))
        for i in range(n_samples)
    ]

    hard: list[dict] = []
    soft: list[dict] = []
    seen: set[str] = set()

    for tfr in tfrs:
        nid = tfr.get("notam_id", "")
        if nid in seen:
            continue
        geo_type = tfr.get("geometry_type", "")
        coords = tfr.get("coordinates", [])
        hit = False

        if geo_type == "Polygon" and len(coords) >= 3:
            for lat, lon in sample_pts:
                if _point_in_polygon(lat, lon, coords):
                    hit = True
                    break
        elif geo_type == "Point" and coords:
            # Stadium: 3 nm ≈ 5.556 km radius
            clat, clon = coords[0]
            for lat, lon in sample_pts:
                if _haversine_km(lat, lon, clat, clon) <= 5.556:
                    hit = True
                    break

        if hit:
            seen.add(nid)
            if tfr.get("type_code") in _HARD_TFR_CODES:
                hard.append(tfr)
            else:
                soft.append(tfr)

    return hard, soft


# ── Step 3: Python routing engine ─────────────────────────────────────────────

_SPEED_HELI_KMH  = 220.0
_SPEED_DRIVE_KMH = 25.0   # urban average
_SPEED_WALK_KMH  = 5.0    # pedestrian average
_ROAD_FACTOR     = 1.35   # haversine → road distance multiplier
_MAX_RANGE_KM    = 555.0  # max helicopter range
_WALK_THRESHOLD_KM = 0.5  # legs shorter than this become walking


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def find_nearest_helipad(lat: float, lon: float,
                         helipads: list[dict]) -> Optional[dict]:
    """Return the closest helipad dict (with added dist_km key) or None.

    Args:
        lat: Query latitude.
        lon: Query longitude.
        helipads: List of dicts each with 'lat', 'lon', 'name'.
    """
    best: Optional[dict] = None
    best_d = float("inf")
    for pad in helipads:
        d = _haversine_km(lat, lon, float(pad["lat"]), float(pad["lon"]))
        if d < best_d:
            best_d = d
            best = {**pad, "dist_km": round(d, 2)}
    return best


def compute_skyroute(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    helipads: list[dict],
    arrival_time: Optional[str] = None,
) -> dict:
    """Compute the optimal multimodal route between two coordinates.

    Drive to nearest departure helipad → fly → drive from arrival helipad.
    Times use: helicopter 220 km/h, drive haversine × 1.35 / 25 km/h.

    Args:
        origin_lat: Origin latitude.
        origin_lon: Origin longitude.
        dest_lat: Destination latitude.
        dest_lon: Destination longitude.
        helipads: Available helipads as list of {lat, lon, name, ident}.
        arrival_time: Target arrival time "HH:MM" or None.

    Returns:
        Dict with total_min, drive_only_min, time_saved_min, departure_time,
        arrival_time, legs, nearest_pad_origin, nearest_pad_dest.
    """
    pad_a = find_nearest_helipad(origin_lat, origin_lon, helipads)
    pad_b = find_nearest_helipad(dest_lat, dest_lon, helipads)

    # Leg 1: origin → pad_a
    _d1_straight = (_haversine_km(origin_lat, origin_lon, pad_a["lat"], pad_a["lon"])
                    if pad_a else 0.0)
    if _d1_straight <= _WALK_THRESHOLD_KM:
        d_to_a, t_to_a, mode_to_a = _d1_straight, _d1_straight / _SPEED_WALK_KMH * 60, "walk"
    else:
        d_to_a = _d1_straight * _ROAD_FACTOR
        t_to_a = d_to_a / _SPEED_DRIVE_KMH * 60
        mode_to_a = "drive"

    # Leg 2: pad_a → pad_b (helicopter)
    d_flight = (_haversine_km(pad_a["lat"], pad_a["lon"], pad_b["lat"], pad_b["lon"])
                if pad_a and pad_b else 0.0)
    t_flight = d_flight / _SPEED_HELI_KMH * 60 if d_flight > 0.1 else 0.0

    # Leg 3: pad_b → destination
    _d3_straight = (_haversine_km(pad_b["lat"], pad_b["lon"], dest_lat, dest_lon)
                    if pad_b else 0.0)
    if _d3_straight <= _WALK_THRESHOLD_KM:
        d_from_b, t_from_b, mode_from_b = _d3_straight, _d3_straight / _SPEED_WALK_KMH * 60, "walk"
    else:
        d_from_b = _d3_straight * _ROAD_FACTOR
        t_from_b = d_from_b / _SPEED_DRIVE_KMH * 60
        mode_from_b = "drive"

    total_min = t_to_a + t_flight + t_from_b

    # Drive-only comparison (haversine × road factor at urban speed)
    drive_only_min = (_haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
                      * _ROAD_FACTOR / _SPEED_DRIVE_KMH * 60)
    time_saved_min = max(0.0, drive_only_min - total_min)

    # Work backwards from arrival deadline
    dep_str: Optional[str] = None
    if arrival_time:
        try:
            h, m = map(int, arrival_time.split(":"))
            dep_total = h * 60 + m - int(total_min)
            dep_str = f"{dep_total // 60:02d}:{dep_total % 60:02d}"
        except ValueError:
            pass

    legs = []
    if pad_a and d_to_a > 0:
        legs.append({
            "mode": mode_to_a,
            "from": "Origin",
            "to": pad_a.get("name") or pad_a.get("ident") or "Departure Helipad",
            "dist_km": round(d_to_a, 1),
            "duration_min": round(t_to_a),
        })
    if pad_a and pad_b and d_flight > 0.1:
        legs.append({
            "mode": "helicopter",
            "from": pad_a.get("name") or pad_a.get("ident") or "Departure Helipad",
            "to": pad_b.get("name") or pad_b.get("ident") or "Arrival Helipad",
            "dist_km": round(d_flight, 1),
            "duration_min": round(t_flight),
        })
    if pad_b and d_from_b > 0:
        legs.append({
            "mode": mode_from_b,
            "from": pad_b.get("name") or pad_b.get("ident") or "Arrival Helipad",
            "to": "Destination",
            "dist_km": round(d_from_b, 1),
            "duration_min": round(t_from_b),
        })

    return {
        "total_min": round(total_min),
        "drive_only_min": round(drive_only_min),
        "time_saved_min": round(time_saved_min),
        "departure_time": dep_str,
        "arrival_time": arrival_time,
        "legs": legs,
        "nearest_pad_origin": pad_a,
        "nearest_pad_dest": pad_b,
        "origin": {"lat": origin_lat, "lon": origin_lon},
        "dest": {"lat": dest_lat, "lon": dest_lon},
    }


# ── Step 4: Response formatter ─────────────────────────────────────────────────

_FORMAT_SYSTEM = """You are SkyRoute's routing assistant for air-mobility in the New York metro area.

Format a computed multimodal itinerary as a concise, friendly response.
Rules:
- Lead with the headline benefit (time saved vs driving)
- List legs as numbered steps with mode icon: 🚗 drive, 🚁 helicopter
- Include departure time and estimated arrival if available
- Mention the helipad names
- End exactly with: "Ready to book all legs in one tap?"
- Maximum 6 sentences. Plain text only — no markdown headers or bullets.
"""


def format_skyroute_response(route: dict, user_text: str) -> str:
    """Format a computed route dict into a natural language narrative.

    Uses Groq for the response; falls back to a template string if unavailable.

    Args:
        route: Output of compute_skyroute().
        user_text: Original user query (for context).

    Returns:
        Friendly itinerary string.
    """
    def _template() -> str:
        legs_str = "  →  ".join(
            f"{'🚁' if l['mode'] == 'helicopter' else '🚗'} "
            f"{l['from']} → {l['to']} ({l['duration_min']} min)"
            for l in route["legs"]
        )
        dep  = route.get("departure_time") or "now"
        arr  = route.get("arrival_time") or f"+{route['total_min']} min"
        saved = route.get("time_saved_min", 0)
        return (
            f"SkyRoute saves you {saved} min vs driving alone. "
            f"Depart at {dep}: {legs_str}. "
            f"Estimated arrival: {arr}. "
            "Ready to book all legs in one tap?"
        )

    client = _get_groq_client()
    if client is None:
        return _template()

    try:
        resp = client.chat.completions.create(
            model=_GROQ_MODEL,
            temperature=0.3,
            messages=[
                {"role": "system", "content": _FORMAT_SYSTEM},
                {"role": "user", "content": (
                    f"User asked: '{user_text}'\n\n"
                    f"Route data:\n{json.dumps(route, indent=2)}"
                )},
            ],
        )
        text = resp.choices[0].message.content.strip()
        if len(text) < 50 or "book" not in text.lower():
            return _template()
        return text
    except Exception as exc:
        log.warning("Route formatting failed: %s", exc)
        return _template()


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_agent(
    user_text: str,
    helipads: list[dict],
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
    history: list[dict] | None = None,
) -> dict:
    """Full M4 pipeline: natural language → structured route → formatted response.

    Steps:
      1. extract_nav_params  — Groq/Llama parses free text → JSON
      2. geocode_place       — Nominatim resolves place names → lat/lon
      3. compute_skyroute    — deterministic Python routing engine
      4. format_skyroute_response — Groq formats result as narrative

    Args:
        user_text: Free-form user request.
        helipads: Available helipads as list of {lat, lon, name, ident}.
        user_lat: Current user latitude (used when origin not specified).
        user_lon: Current user longitude.
        history: Prior conversation turns for multi-turn context (see extract_nav_params).

    Returns:
        Dict with: params, origin, destination, route, response, error.
        'error' is None on success; a user-friendly string on failure.
    """
    result: dict = {
        "params": None,
        "origin": None,
        "destination": None,
        "route": None,
        "response": None,
        "error": None,
    }

    # Step 1 — Extract parameters
    params = extract_nav_params(user_text, history=history)
    result["params"] = params

    if params.get("intent") == "off_topic":
        result["error"] = (
            "I handle helicopter routing in the New York metro area. "
            "Try: \"I need to be at [destination] by [time].\""
        )
        return result

    if not params.get("destination_text"):
        result["error"] = (
            "I couldn't find a destination in your message. "
            "Try: \"I need to be at [place] by [time].\""
        )
        return result

    # Step 2 — Geocode destination
    dest_ll = geocode_place(params["destination_text"])
    if dest_ll is None:
        result["error"] = (
            "I handle helicopter routing in the New York metro area. "
            "Try: \"I need to be at [destination] by [time].\""
        )
        return result
    result["destination"] = {
        "text": params["destination_text"],
        "lat": dest_ll[0],
        "lon": dest_ll[1],
    }

    # Step 2 — Resolve origin
    if params.get("origin_text"):
        origin_ll = geocode_place(params["origin_text"])
        if origin_ll is None:
            result["error"] = (
                f"I couldn't locate origin \"{params['origin_text']}\". "
                "Try a more specific address."
            )
            return result
        result["origin"] = {
            "text": params["origin_text"],
            "lat": origin_ll[0],
            "lon": origin_ll[1],
        }
    elif user_lat is not None and user_lon is not None:
        origin_ll = (user_lat, user_lon)
        result["origin"] = {"text": "Your location", "lat": user_lat, "lon": user_lon}
    else:
        # Default: 30th St Heliport / Midtown Manhattan
        origin_ll = (40.7503, -74.0025)
        result["origin"] = {
            "text": "Midtown Manhattan (default — specify origin for accuracy)",
            "lat": origin_ll[0],
            "lon": origin_ll[1],
        }

    # Step 3 — Compute route
    if not helipads:
        result["error"] = "No helipad data loaded. Ensure faa_helipads_raw.csv exists."
        return result

    route = compute_skyroute(
        origin_ll[0], origin_ll[1],
        dest_ll[0], dest_ll[1],
        helipads,
        arrival_time=params.get("arrival_time"),
    )
    result["route"] = route

    # Step 3b — TFR check on the aerial segment
    pad_a = route.get("nearest_pad_origin")
    pad_b = route.get("nearest_pad_dest")
    ignore_tfrs = bool(params.get("ignore_tfrs", False))
    if pad_a and pad_b:
        hard_tfrs, soft_tfrs = _check_tfrs_on_segment(
            float(pad_a["lat"]), float(pad_a["lon"]),
            float(pad_b["lat"]), float(pad_b["lon"]),
        )
        if hard_tfrs and not ignore_tfrs:
            names = "; ".join(t.get("text", "Unknown TFR") for t in hard_tfrs[:2])
            result["error"] = (
                f"Active airspace closure on this route: {names}. "
                "Booking is not possible until the TFR lifts. "
                "To plan anyway, add \"ignore TFRs\" to your request. "
                "Check tfr.faa.gov for the latest status."
            )
            return result
        all_warnings = [t.get("text", "Active TFR") for t in hard_tfrs + soft_tfrs]
        result["tfr_warnings"] = all_warnings
        result["tfrs_ignored"] = ignore_tfrs and bool(hard_tfrs)
    else:
        result["tfr_warnings"] = []
        result["tfrs_ignored"] = False

    # Step 4 — Format narrative
    result["response"] = format_skyroute_response(route, user_text)

    return result


# ── Mapillary helpers ─────────────────────────────────────────────────────────

def _mapillary_embed(lat: float, lon: float) -> tuple[str, str]:
    """Return (embed_url, app_url) for a Mapillary street view at lat/lon."""
    embed = (
        f"https://www.mapillary.com/embed"
        f"?lat={lat:.6f}&lng={lon:.6f}&z=15&style=photo"
    )
    app = f"https://www.mapillary.com/app/?lat={lat:.6f}&lng={lon:.6f}&z=17"
    return embed, app


_mly_id_cache: dict = {}   # in-memory cache: (lat_r, lon_r) → image_id | None


def find_nearest_mapillary_image(lat: float, lon: float,
                                  max_radius_m: int = 1000) -> Optional[str]:
    """Return the nearest Mapillary image ID to (lat, lon).

    Runs server-side (Python) — no CORS restriction.
    Tries closeto (lat,lon) then bbox, expanding radius until an image is found.
    Caches results in-memory to avoid duplicate API calls within a session.

    Args:
        lat: Latitude.
        lon: Longitude.
        max_radius_m: Maximum search radius in metres.

    Returns:
        Mapillary image ID string, or None if not found / token missing.
    """
    token = os.getenv("MAPILLARY_TOKEN", "")
    if not token:
        return None

    key = (round(lat, 4), round(lon, 4))
    if key in _mly_id_cache:
        return _mly_id_cache[key]

    headers = {"Authorization": f"OAuth {token}"}
    radii = [r for r in [150, 400, max_radius_m] if r <= max_radius_m]

    for radius in radii:
        # Try closeto (lat,lon) — v4 point search
        try:
            resp = requests.get(
                "https://graph.mapillary.com/images",
                params={"fields": "id", "closeto": f"{lon},{lat}",
                        "radius": radius, "limit": 1},
                headers=headers,
                timeout=8,
            )
            data = resp.json()
            imgs = (data.get("data") or [])
            if imgs:
                image_id = imgs[0]["id"]
                _mly_id_cache[key] = image_id
                log.info("Mapillary: found %s within %dm of (%.4f,%.4f)",
                         image_id, radius, lat, lon)
                return image_id
        except Exception as exc:
            log.debug("Mapillary closeto failed r=%d: %s", radius, exc)

        # Fallback: tiny bbox
        try:
            r_deg = radius / 111_000
            cos_lat = max(math.cos(math.radians(lat)), 0.01)
            bbox = (f"{lon - r_deg/cos_lat:.6f},{lat - r_deg:.6f},"
                    f"{lon + r_deg/cos_lat:.6f},{lat + r_deg:.6f}")
            resp = requests.get(
                "https://graph.mapillary.com/images",
                params={"fields": "id", "bbox": bbox, "limit": 1},
                headers=headers,
                timeout=8,
            )
            data = resp.json()
            imgs = (data.get("data") or [])
            if imgs:
                image_id = imgs[0]["id"]
                _mly_id_cache[key] = image_id
                return image_id
        except Exception as exc:
            log.debug("Mapillary bbox failed r=%d: %s", radius, exc)

    _mly_id_cache[key] = None
    return None


_mly_thumb_cache: dict = {}


def get_mapillary_thumb_url(image_id: str) -> Optional[str]:
    """Fetch the 2048px thumbnail URL for a Mapillary image ID.

    Runs server-side — no CORS restriction. Returns a direct CDN JPEG URL
    that can be embedded as <img src="...">, no JS viewer needed.

    Args:
        image_id: Mapillary image ID string.

    Returns:
        Direct HTTPS URL to the JPEG thumbnail, or None on failure.
    """
    if not image_id:
        return None
    if image_id in _mly_thumb_cache:
        return _mly_thumb_cache[image_id]

    token = os.getenv("MAPILLARY_TOKEN", "")
    if not token:
        return None

    try:
        resp = requests.get(
            f"https://graph.mapillary.com/{image_id}",
            params={"fields": "thumb_2048_url,thumb_original_url"},
            headers={"Authorization": f"OAuth {token}"},
            timeout=8,
        )
        data = resp.json()
        url = data.get("thumb_2048_url") or data.get("thumb_original_url")
        _mly_thumb_cache[image_id] = url
        return url
    except Exception as exc:
        log.debug("Mapillary thumb fetch failed for %s: %s", image_id, exc)
        _mly_thumb_cache[image_id] = None
        return None


def _osm_embed(lat: float, lon: float, delta: float = 0.004) -> str:
    """Return an OpenStreetMap embed URL centred on lat/lon.

    delta controls the bbox half-width in degrees (~0.004° ≈ 400 m).
    Always loads — no API key required.
    """
    return (
        f"https://www.openstreetmap.org/export/embed.html"
        f"?bbox={lon - delta},{lat - delta},{lon + delta},{lat + delta}"
        f"&layer=mapnik&marker={lat},{lon}"
    )


# ── Booking flow ──────────────────────────────────────────────────────────────

def lookup_helipad_info(ident: Optional[str], name: Optional[str],
                        lat: float, lon: float,
                        faa_adip_df=None) -> dict:
    """Return booking/coordination info for a helipad.

    Reads from faa_adip_enriched.csv columns if DataFrame is provided.
    Always returns ADIP portal link and Mapillary URL.

    Args:
        ident: FAA IDENT code (e.g. "JRB").
        name: Helipad name.
        lat: Latitude.
        lon: Longitude.
        faa_adip_df: Optional FAA ADIP enriched DataFrame.

    Returns:
        Dict with keys: ident, name, adip_url, mapillary_url, ownership,
        status, private_use, servcity, contact_notes.
    """
    _mly_embed, _mly_url = _mapillary_embed(lat, lon)
    info: dict = {
        "ident": ident or "",
        "name": name or "Helipad",
        "lat": lat,
        "lon": lon,
        "adip_url": (f"https://adip.faa.gov/agis/public/#/simpleAirportMap/{ident}"
                     if ident else None),
        "mapillary_url": _mly_url,
        "mapillary_embed": _mly_embed,
        "osm_embed": _osm_embed(lat, lon),
        "gmaps_url": f"https://maps.google.com/?q={lat:.6f},{lon:.6f}&z=17",
        "ownership": "unknown",
        "status": "unknown",
        "private_use": False,
        "servcity": "",
        "contact_notes": "Contact helipad operator for approach coordination.",
    }

    if faa_adip_df is not None and ident:
        try:
            import pandas as pd
            row = faa_adip_df[faa_adip_df["IDENT"] == ident]
            if not row.empty:
                r = row.iloc[0]
                ownership = str(r.get("ownership_code", "") or "")
                info["ownership"] = ownership
                info["status"] = str(r.get("adip_status", r.get("OPERSTATUS", "")) or "")
                info["private_use"] = str(r.get("PRIVATEUSE", "N")).strip().upper() == "Y"
                info["servcity"] = str(r.get("SERVCITY", "") or "")
                info["last_info_date"] = str(r.get("last_info_date", "") or "")
                info["notam_service"] = str(r.get("notam_service", "") or "")
                info["icao_id"] = str(r.get("ICAO_ID", "") or "")

                if ownership == "PR":
                    info["contact_notes"] = (
                        "Private helipad — prior coordination required. "
                        "Contact the operator directly via the ADIP record before flight."
                    )
                elif ownership == "PU":
                    info["contact_notes"] = (
                        "Public-use helipad. PPR (Prior Permission Required) may apply. "
                        "Check current NOTAMs and contact facility manager."
                    )
                elif ownership in ("MA", "MN", "MR"):
                    info["contact_notes"] = (
                        "Military helipad — civilian use is restricted. "
                        "Federal authorization required. Do not approach without clearance."
                    )
        except Exception as exc:
            log.warning("ADIP lookup failed for %s: %s", ident, exc)

    # Fetch ADIP remarks and decode with LLM — overrides the generic ownership note
    # when actual operational remarks exist (e.g. "FOR CD CTC NEW YORK APCH AT …")
    if ident:
        try:
            remarks = _fetch_adip_remarks(ident)
            if remarks:
                decoded = _decode_adip_remarks(remarks, ident, info["name"])
                if decoded:
                    info["contact_notes"] = decoded
                    info["raw_remarks"] = remarks
        except Exception as exc:
            log.warning("Remarks decode failed for %s: %s", ident, exc)

    return info


def simulate_rideshare(origin_lat: float, origin_lon: float,
                       dest_lat: float, dest_lon: float) -> dict:
    """Simulate a rideshare booking (Uber or Waymo autonomous taxi).

    Fare and ETA are estimates based on haversine × road factor.
    Waymo is simulated for NYC/NJ metro (where autonomous vehicles operate).
    Booking reference is randomly generated.

    Args:
        origin_lat: Pickup latitude.
        origin_lon: Pickup longitude.
        dest_lat: Dropoff latitude.
        dest_lon: Dropoff longitude.

    Returns:
        Dict with service, vehicle, fare_range, duration_min, dist_km,
        booking_ref, uber_deeplink, waymo_url.
    """
    dist_km = _haversine_km(origin_lat, origin_lon, dest_lat, dest_lon) * _ROAD_FACTOR
    duration_min = dist_km / _SPEED_DRIVE_KMH * 60
    # Fare model: base $3.50 + $1.80/km + $0.35/min
    fare = 3.50 + 1.80 * dist_km + 0.35 * duration_min
    fare = max(fare, 9.0)

    ref = "SKY" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

    uber_deeplink = (
        f"https://m.uber.com/ul/?action=setPickup"
        f"&pickup[latitude]={origin_lat:.6f}&pickup[longitude]={origin_lon:.6f}"
        f"&dropoff[latitude]={dest_lat:.6f}&dropoff[longitude]={dest_lon:.6f}"
    )
    waymo_url = "https://waymo.com/waymo-one/"

    return {
        "service": "Uber / Waymo",
        "vehicles": ["UberX", "Waymo One (autonomous)"],
        "fare_range": f"${fare * 0.85:.0f}–${fare * 1.15:.0f}",
        "fare_est": round(fare, 2),
        "duration_min": round(duration_min),
        "dist_km": round(dist_km, 1),
        "booking_ref": ref,
        "uber_deeplink": uber_deeplink,
        "waymo_url": waymo_url,
        "status": "SIMULATED",
    }


def run_booking(route: dict, faa_adip_df=None) -> list[dict]:
    """Build a per-leg booking plan for a computed route.

    Drive legs → rideshare simulation (Uber / Waymo).
    Helicopter legs → helipad coordination info (ADIP + Mapillary).

    Args:
        route: Output of compute_skyroute().
        faa_adip_df: Optional FAA ADIP enriched DataFrame for contact lookup.

    Returns:
        List of booking dicts, one per leg.
    """
    booking_legs: list[dict] = []
    legs = route.get("legs", [])

    for i, leg in enumerate(legs):
        if leg["mode"] == "helicopter":
            pad_a = route.get("nearest_pad_origin", {})
            pad_b = route.get("nearest_pad_dest", {})

            dep_info = lookup_helipad_info(
                ident=pad_a.get("ident"), name=pad_a.get("name"),
                lat=pad_a["lat"], lon=pad_a["lon"],
                faa_adip_df=faa_adip_df,
            )
            arr_info = lookup_helipad_info(
                ident=pad_b.get("ident"), name=pad_b.get("name"),
                lat=pad_b["lat"], lon=pad_b["lon"],
                faa_adip_df=faa_adip_df,
            )
            dep_info["mly_image_id"] = find_nearest_mapillary_image(
                pad_a["lat"], pad_a["lon"]) or ""
            dep_info["mly_thumb_url"] = get_mapillary_thumb_url(dep_info["mly_image_id"]) or ""
            arr_info["mly_image_id"] = find_nearest_mapillary_image(
                pad_b["lat"], pad_b["lon"]) or ""
            arr_info["mly_thumb_url"] = get_mapillary_thumb_url(arr_info["mly_image_id"]) or ""
            booking_legs.append({
                "leg_index": i,
                "mode": "helicopter",
                "from": leg["from"],
                "to": leg["to"],
                "duration_min": leg["duration_min"],
                "dist_km": leg["dist_km"],
                "departure_helipad": dep_info,
                "arrival_helipad": arr_info,
            })

        elif leg["mode"] in ("drive", "walk"):
            if i == 0:
                pad = route.get("nearest_pad_origin", {})
                o = route["origin"]
                pickup_lat, pickup_lon = o["lat"], o["lon"]
                dropoff_lat, dropoff_lon = pad["lat"], pad["lon"]
                from_name, to_name = "Your origin", pad.get("name", leg["to"])
            else:
                pad = route.get("nearest_pad_dest", {})
                d = route["dest"]
                pickup_lat, pickup_lon = pad["lat"], pad["lon"]
                dropoff_lat, dropoff_lon = d["lat"], d["lon"]
                from_name, to_name = pad.get("name", leg["from"]), "Your destination"

            _pu_embed, _pu_url = _mapillary_embed(pickup_lat, pickup_lon)
            _do_embed, _do_url = _mapillary_embed(dropoff_lat, dropoff_lon)
            _pu_mly_id = find_nearest_mapillary_image(pickup_lat, pickup_lon) or ""
            _pu_mly_thumb = get_mapillary_thumb_url(_pu_mly_id) or ""
            _do_mly_id = find_nearest_mapillary_image(dropoff_lat, dropoff_lon) or ""
            _do_mly_thumb = get_mapillary_thumb_url(_do_mly_id) or ""

            if leg["mode"] == "walk":
                booking_legs.append({
                    "leg_index": i,
                    "mode": "walk",
                    "from": from_name,
                    "to": to_name,
                    "dist_km": leg["dist_km"],
                    "duration_min": leg["duration_min"],
                    "pickup_lat": pickup_lat, "pickup_lon": pickup_lon,
                    "dropoff_lat": dropoff_lat, "dropoff_lon": dropoff_lon,
                    "pickup_mapillary_url": _pu_url,
                    "pickup_mly_id": _pu_mly_id,
                    "pickup_mly_thumb": _pu_mly_thumb,
                })
            else:
                rideshare = simulate_rideshare(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
                booking_legs.append({
                    "leg_index": i,
                    "mode": "rideshare",
                    "from": from_name,
                    "to": to_name,
                    "pickup_lat": pickup_lat, "pickup_lon": pickup_lon,
                    "dropoff_lat": dropoff_lat, "dropoff_lon": dropoff_lon,
                    "pickup_mapillary_embed": _pu_embed,
                    "dropoff_mapillary_embed": _do_embed,
                    "pickup_mapillary_url": _pu_url,
                    "dropoff_mapillary_url": _do_url,
                    "pickup_mly_id": _pu_mly_id,
                    "pickup_mly_thumb": _pu_mly_thumb,
                    "dropoff_mly_id": _do_mly_id,
                    "dropoff_mly_thumb": _do_mly_thumb,
                    "rideshare": rideshare,
                })

    return booking_legs
