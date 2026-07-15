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
from datetime import datetime, timezone, timedelta
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

_GROQ_MODEL         = "llama-3.1-8b-instant"       # fallback — always enabled
_GROQ_MODEL_FAST    = "meta-llama/llama-4-scout-17b-16e-instruct"  # if enabled in project settings
_GROQ_MODEL_70B     = "llama-3.3-70b-versatile"   # Level 1 tool calling (enable in Groq console)

# ── Level 1 tool-calling concierge (run_agent_v2) ─────────────────────────────

_CONCIERGE_SYSTEM = """You are Mia — SkyRoute's Mobility Intelligence Assistant. \
You are a female AI travel concierge specialising in executive air mobility across \
the New York metro area. Your name, Mia, stands for Mobility Intelligence Assistant. \
Your user is a busy VP who travels 4-5×/week and values time above all else.

What you can do:
- Plan door-to-door multimodal routes combining helicopter and ground transport
- Search for restaurants, hotels, cafes, and other businesses near any location
- Provide live weather forecasts for any destination
- Check helipad operational status and live airspace restrictions
- Book confirmed itineraries leg by leg

You have access to these tools:
- geocode: convert addresses or place names to coordinates
- search_nearby_places: find restaurants, hotels, cafes, or businesses near any location
- get_weather: current conditions and forecast at any location
- compute_route: multimodal helicopter + ground route with time comparison
- confirm_booking: book a confirmed route

MANDATORY rules:
1. Never name a specific restaurant, hotel, or business without first calling \
search_nearby_places. Your training knowledge about specific places is unreliable.
2. Never state travel times or route details without first calling compute_route.
3. Never state weather conditions without first calling get_weather.
4. If a tool returns {"error": ...}, tell the user that information is unavailable — \
do not substitute your own estimate.
5. Keep responses concise — the user is time-pressed. One sentence per leg is enough.
6. Airspace restrictions, METAR conditions, and helipad operational status are handled \
automatically inside compute_route. Never ask the user about them.
7. For off-topic questions unrelated to travel, routing, weather, or nearby amenities, \
politely redirect to your area of expertise.
8. If asked about your personal background, creator, how you were built, or any details \
beyond your name and capabilities, politely decline and steer back to travel planning.
9. When routing to a place that appeared in a previous search_nearby_places result, pass \
its exact address (not just the name) as the destination to compute_route. The address \
field in search results is precise; the name alone may geocode to the wrong location."""

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": (
                "Convert a place name or address to geographic coordinates. "
                "Call this first when you need the location of somewhere the user mentioned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "Place name or street address, e.g. '30th St Heliport' or '272 Park Ave, New York'",
                    }
                },
                "required": ["address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_nearby_places",
            "description": (
                "Search for restaurants, hotels, cafes, or other businesses near a location. "
                "You MUST call this before naming any specific place."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "Specific location to search near, e.g. '30th St Heliport, NYC' "
                            "or 'Westchester Airport'. "
                            "NEVER pass 'nearby', 'near me', or 'here' — always resolve to "
                            "a real place name. If the user said 'nearby' without a location, "
                            "use the origin from the session context."
                        ),
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Type of place — use the CUISINE or PLACE TYPE only, not a full phrase. "
                            "Good: 'italian', 'restaurant', 'hotel', 'coffee', 'sushi', 'french', 'seafood'. "
                            "BAD: 'italian restaurant', 'fine dining restaurant' — drop the word 'restaurant'."
                        ),
                    },
                    "radius_m": {
                        "type": "integer",
                        "description": "Search radius in metres (default 1500, max 3000). Use 2000–3000 for suburban or marina areas.",
                    },
                },
                "required": ["location", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather and forecast for any location. "
                "You MUST call this before stating any weather conditions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City, address, or landmark, e.g. 'Greenwich CT' or '30th St Heliport NYC'",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["imperial", "metric"],
                        "description": (
                            "Unit system for the response. Use 'metric' when the user asks "
                            "for Celsius, km/h, m/s, or any SI units. Default 'imperial' (°F, mph)."
                        ),
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_route",
            "description": (
                "Compute a multimodal helicopter + ground route between two locations. "
                "Returns travel time, legs breakdown, time saved vs driving, and any advisories. "
                "You MUST call this before stating any travel times or route details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Starting location",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination location",
                    },
                },
                "required": ["origin", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_booking",
            "description": (
                "Confirm and book the helicopter + ground route. "
                "Only call this when the user explicitly says 'book', 'yes', 'confirm', "
                "'go ahead', 'book it', or similar AND a route has already been discussed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Starting location (same as compute_route origin)",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination location",
                    },
                },
                "required": ["origin", "destination"],
            },
        },
    },
]

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

    # timeout=30 prevents a 600s freeze during demos.
    # max_retries=2 (SDK default) uses the Groq retry-after header on 429s.
    return Groq(api_key=key, timeout=30.0)


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
            item = results[0]
            pos = item["position"]
            _geocode_rich_cache[place_name.lower().strip()] = {
                "poi_name": item.get("poi", {}).get("name", ""),
                "address":  item.get("address", {}).get("freeformAddress", ""),
            }
            return float(pos["lat"]), float(pos["lon"])
    except Exception as exc:
        log.warning("TomTom geocode failed for '%s': %s", place_name, exc)
    return None


def _reverse_geocode_tomtom(lat: float, lon: float) -> dict:
    """Reverse geocode coordinates to a street address via TomTom.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        Dict with keys: address (str), poi_name (str). Empty strings if unavailable.
    """
    key = os.getenv("TOMTOM_API_KEY", "")
    if not key:
        return {"address": "", "poi_name": ""}
    try:
        resp = requests.get(
            f"https://api.tomtom.com/search/2/reverseGeocode/{lat:.6f},{lon:.6f}.json",
            params={"key": key},
            timeout=8,
        )
        resp.raise_for_status()
        addresses = resp.json().get("addresses", [])
        if addresses:
            addr = addresses[0].get("address", {})
            return {
                "address": addr.get("freeformAddress", ""),
                "poi_name": "",
            }
    except Exception as exc:
        log.warning("TomTom reverse geocode failed for (%s, %s): %s", lat, lon, exc)
    return {"address": "", "poi_name": ""}


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


def _geo_dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km between two points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def geocode_place(place_name: str) -> Optional[tuple[float, float]]:
    """Geocode a place name or address to (lat, lon).

    Pipeline:
      1. In-memory cache (session-scoped, avoids repeated HTTP for same place)
      2. Try TomTom (business names + addresses, best accuracy)
         — If place_name has a geographic qualifier (comma), cross-check with
           Nominatim.  When they disagree by > 3 km, Nominatim wins: TomTom's
           Manhattan-centre bias can return Grand Central instead of, e.g.,
           "Hoboken Terminal, New York" (which should resolve to Hoboken, NJ).
      3. If TomTom fails, strip business name via LLM then retry TomTom
      4. Fall back to Nominatim on the cleaned address

    Args:
        place_name: Human-readable place name, business, or address.

    Returns:
        (lat, lon) tuple, or None if all sources fail.
    """
    cache_key = place_name.lower().strip()
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    # 1. Try TomTom directly
    result = _geocode_tomtom(place_name)
    if result:
        # Cross-check with Nominatim when the query contains a geographic qualifier
        # (e.g. "Hoboken Terminal, New York").  TomTom's NYC-centre proximity bias
        # can match a Manhattan landmark instead of the named cross-state location.
        if "," in place_name:
            nom = _geocode_nominatim(place_name)
            if nom and _geo_dist_km(result[0], result[1], nom[0], nom[1]) > 3.0:
                log.info(
                    "Geocode: TomTom/Nominatim disagree by %.1f km for '%s'; using Nominatim",
                    _geo_dist_km(result[0], result[1], nom[0], nom[1]),
                    place_name,
                )
                # Discard TomTom's rich cache (wrong POI name/address for the wrong location)
                _geocode_rich_cache.pop(cache_key, None)
                result = nom
        _geocode_cache[cache_key] = result
        return result

    # 2. LLM-clean the address, retry TomTom (handles "Business at 32 Mercer St …")
    cleaned = _extract_address_with_llm(place_name)
    if cleaned != place_name:
        log.info("Geocode: cleaned '%s' → '%s'", place_name, cleaned)
        result = _geocode_tomtom(cleaned)
        if result:
            _geocode_cache[cache_key] = result
            # Propagate rich data to the original (unclean) key so callers can look it up
            if cleaned.lower().strip() in _geocode_rich_cache:
                _geocode_rich_cache[cache_key] = _geocode_rich_cache[cleaned.lower().strip()]
            return result

    # 3. Nominatim fallback on cleaned address
    result = _geocode_nominatim(cleaned)
    _geocode_cache[cache_key] = result  # cache None too (avoids re-trying dead locations)
    return result


# ── Level 1 tool implementations ──────────────────────────────────────────────

import urllib.parse as _urlparse
import datetime as _dt

_nws_cache: dict = {}           # key: "lat2,lon2" → (result_dict, fetched_at)
_NWS_TTL_S = 600                # 10-minute cache
_nws_gridpoint_cache: dict = {} # key: "lat2,lon2" → forecast_url (permanent — gridpoints never move)
_geocode_cache: dict = {}       # key: place_name.lower().strip() → (lat, lon) | None
_geocode_rich_cache: dict = {}  # key: place_name.lower().strip() → {poi_name, address}
_poi_cache: dict = {}           # key: (location, category, radius_m) → result_dict
_yelp_cache: dict = {}          # key: (name_lower, lat4, lon4) → enrichment dict

# TomTom structured category codes — hard-filter results so keyword drift can't pull
# in off-category businesses (e.g. liquor stores for "fine dining").
# https://developer.tomtom.com/search-api/documentation/product-information/supported-categories
_TOMTOM_CATEGORY_MAP: dict[str, int] = {
    # dining
    "restaurant": 7315,
    "restaurants": 7315,
    "dining": 7315,
    "fine dining": 7315,
    "dinner": 7315,
    "lunch": 7315,
    "food": 7315,
    "eat": 7315,
    "sushi": 7315025,
    "italian": 7315003,
    "french": 7315001,
    "steakhouse": 7315002,
    "seafood": 7315005,
    "japanese": 7315034,
    "american": 7315041,
    "bar": 7315,
    "bistro": 7315,
    "brasserie": 7315,
    # coffee
    "coffee": 9361065,
    "cafe": 9361065,
    "cafes": 9361065,
    "coffee shop": 9361065,
    # accommodation
    "hotel": 7314,
    "hotels": 7314,
    "motel": 7314,
    # transport
    "parking": 7013,
    "gas station": 7311,
    "fuel": 7311,
    # health
    "pharmacy": 9361,
    "hospital": 7321,
    # other
    "lounge": 9913,
    "atm": 7397001,
}


def _yelp_enrich(name: str, lat: float, lon: float) -> dict:
    """Return Yelp rating/price/review_count for a business by name + coordinates.

    Returns an empty dict when YELP_API_KEY is absent or the lookup fails —
    callers must treat all keys as optional.

    Args:
        name: Business name as returned by TomTom.
        lat: Latitude of the business.
        lon: Longitude of the business.

    Returns:
        Dict with keys: rating (float), review_count (int), price (str "$$"),
        yelp_url (str). Empty dict on failure.
    """
    api_key = os.getenv("YELP_API_KEY", "")
    if not api_key:
        return {}
    cache_k = (name.lower().strip(), round(lat, 4), round(lon, 4))
    if cache_k in _yelp_cache:
        return _yelp_cache[cache_k]
    try:
        resp = requests.get(
            "https://api.yelp.com/v3/businesses/search",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"term": name, "latitude": lat, "longitude": lon, "limit": 1},
            timeout=5,
        )
        resp.raise_for_status()
        biz = (resp.json().get("businesses") or [None])[0]
        # Skip permanently closed businesses — their URL leads to a dead page
        if biz and biz.get("is_closed", False):
            biz = None
        rating_raw = biz.get("rating") if biz else None
        out = (
            {
                # Treat rating=0 same as missing — 0.0 stars is a Yelp data artifact
                "rating":       rating_raw if (rating_raw and rating_raw > 0) else None,
                "review_count": biz.get("review_count"),
                "price":        biz.get("price", ""),
                "yelp_url":     biz.get("url", ""),
            }
            if biz else {}
        )
    except Exception as exc:
        log.debug("Yelp enrich failed for '%s': %s", name, exc)
        out = {}
    _yelp_cache[cache_k] = out
    return out


def _tool_search_places(location: str, category: str, radius_m: int = 1500) -> dict:
    """Search for POIs near a location using TomTom Search API.

    Args:
        location: Plain-English location name or address.
        category: Type of place, e.g. 'restaurant', 'hotel', 'coffee'.
        radius_m: Search radius in metres (clamped to 3000).

    Returns:
        Dict with 'results' list or 'error' key.
    """
    radius_m = min(int(radius_m), 3000)
    poi_key = (location.lower().strip(), category.lower().strip(), radius_m)
    if poi_key in _poi_cache:
        return _poi_cache[poi_key]

    coords = geocode_place(location)
    if coords is None:
        return {"error": f"Could not locate '{location}'", "results": []}
    lat, lon = coords

    key = os.getenv("TOMTOM_API_KEY", "")
    if not key:
        return {"error": "TomTom API key not configured", "results": []}

    cat_lower = category.lower().strip()
    category_code = _TOMTOM_CATEGORY_MAP.get(cat_lower)
    if category_code is None:
        # Fallback: try each word so "italian restaurant" → "italian" → 7315003
        for word in cat_lower.split():
            if word in _TOMTOM_CATEGORY_MAP:
                category_code = _TOMTOM_CATEGORY_MAP[word]
                break
    url = (
        f"https://api.tomtom.com/search/2/poiSearch/"
        f"{_urlparse.quote(category)}.json"
    )
    params: dict = {"key": key, "lat": lat, "lon": lon,
                    "radius": radius_m, "limit": 5,
                    "openingHours": "nextSevenDays"}
    if category_code:
        params["categorySet"] = category_code
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("TomTom POI search failed: %s", exc)
        return {"error": "Place search temporarily unavailable", "results": []}

    items = resp.json().get("results", [])
    # Zero-result fallback: cuisine sub-categories (7315xxx) are often absent from
    # small restaurant records in TomTom. Retry with the parent restaurant category
    # (7315) so the keyword in the URL path still drives relevance ranking.
    if not items and category_code and category_code != 7315 and str(category_code).startswith("7315"):
        params_fb = {**params, "categorySet": 7315}
        try:
            resp_fb = requests.get(url, params=params_fb, timeout=8)
            resp_fb.raise_for_status()
            items = resp_fb.json().get("results", [])
        except Exception:
            pass
    if not items:
        out = {
            "results": [],
            "note": f"No {category} found within {radius_m}m of {location} — try a larger radius",
        }
        _poi_cache[poi_key] = out
        return out

    import datetime as _dt
    from concurrent.futures import ThreadPoolExecutor as _TPE

    _today = _dt.datetime.utcnow().date().isoformat()

    def _hours_today(poi: dict) -> str:
        oh = poi.get("openingHours", {})
        for tr in oh.get("timeRanges", []):
            if tr.get("startTime", {}).get("date") == _today:
                s, e = tr["startTime"], tr["endTime"]
                def _fmt(h: int, m: int) -> str:
                    period = "AM" if h < 12 else "PM"
                    return f"{h % 12 or 12}:{m:02d} {period}"
                return f"{_fmt(s['hour'], s['minute'])} – {_fmt(e['hour'], e['minute'])}"
        return ""

    # Build base results with TomTom data; keep position coords for Yelp lookup
    base: list[dict] = []
    for item in items[:5]:
        poi  = item.get("poi", {})
        addr = item.get("address", {})
        pos  = item.get("position", {})
        base.append({
            "name":       poi.get("name", "Unknown"),
            "address":    addr.get("freeformAddress", ""),
            "distance_m": round(item.get("dist", 0)),
            "phone":      poi.get("phone", ""),
            "url":        poi.get("url", ""),
            "hours_today": _hours_today(poi),
            "_lat": pos.get("lat", lat),
            "_lon": pos.get("lon", lon),
        })

    # Enrich with Yelp in parallel — all calls run concurrently
    def _enrich(r: dict) -> dict:
        return _yelp_enrich(r["name"], r["_lat"], r["_lon"])

    with _TPE(max_workers=len(base)) as pool:
        yelp_data = list(pool.map(_enrich, base))

    results = []
    for r, yelp in zip(base, yelp_data):
        # Rename internal _lat/_lon → lat/lon so the model can use exact coordinates
        # when routing to a search result without re-geocoding from name alone.
        r["lat"] = r.pop("_lat")
        r["lon"] = r.pop("_lon")
        results.append({**r, **yelp})

    out = {"location": location, "category": category, "results": results}
    _poi_cache[poi_key] = out
    return out


def _tool_get_weather(location: str, units: str = "imperial") -> dict:
    """Get NWS Point Forecast for a location (CONUS only, no API key).

    Args:
        location: City, address, or landmark.
        units: 'imperial' (°F, mph) or 'metric' (°C, m/s).

    Returns:
        Dict with conditions/temperature/wind/precip_chance or 'error' key.
    """
    import re as _re

    def _mph_to_ms(wind_str: str) -> str:
        # NWS often returns a range: "3 to 13 mph SW"
        m = _re.match(r"([\d.]+)\s+to\s+([\d.]+)\s*mph(.*)", wind_str, _re.IGNORECASE)
        if m:
            lo = round(float(m.group(1)) * 0.44704, 1)
            hi = round(float(m.group(2)) * 0.44704, 1)
            return f"{lo} to {hi} m/s{m.group(3)}"
        m = _re.match(r"([\d.]+)\s*mph(.*)", wind_str, _re.IGNORECASE)
        if not m:
            return wind_str
        return f"{round(float(m.group(1)) * 0.44704, 1)} m/s{m.group(2)}"

    def _format_weather(raw: dict, metric: bool, units_str: str) -> dict:
        """Convert a cached raw-imperial NWS dict to the requested unit system."""
        if metric:
            disp_temp  = raw["temperature_c"]
            disp_wind  = _mph_to_ms(raw["wind_raw"])
            temp_label = "°C"
            wind_label = "m/s"
        else:
            disp_temp  = raw["temperature_f"]
            disp_wind  = raw["wind_raw"]
            temp_label = "°F"
            wind_label = "mph"

        periods = []
        for p in raw.get("periods_raw", []):
            pf = p["temperature_f"]
            periods.append({
                "name":          p["name"],
                "conditions":    p["conditions"],
                "temperature_f": pf,
                "temperature_c": round((pf - 32) * 5 / 9, 1),
                "wind":          _mph_to_ms(p["wind_raw"]) if metric else p["wind_raw"],
                "precip_chance": p["precip_chance"],
                "is_daytime":    p["is_daytime"],
            })

        return {
            "location":      raw["location"],
            "period":        raw["period"],
            "conditions":    raw["conditions"],
            "temperature_f": raw["temperature_f"],
            "temperature_c": raw["temperature_c"],
            "wind":          disp_wind,
            "precip_chance": raw["precip_chance"],
            "detailed":      raw["detailed"],
            "periods":       periods,
            "units":         units_str,
            "temp_label":    temp_label,
            "wind_label":    wind_label,
        }

    metric = (units == "metric")
    coords = geocode_place(location)
    if coords is None:
        return {"error": f"Could not locate '{location}'"}
    lat, lon = coords

    # NWS only covers CONUS (approx lat 24–50, lon -66 to -125)
    if not (24.0 <= lat <= 50.0 and -125.0 <= lon <= -66.0):
        return {"error": "Weather data not available outside the continental US"}

    cache_key = f"{lat:.2f},{lon:.2f}"
    cached = _nws_cache.get(cache_key)
    if cached:
        raw, fetched_at = cached
        if (_dt.datetime.utcnow() - fetched_at).total_seconds() < _NWS_TTL_S:
            return _format_weather(raw, metric, units)   # always convert on read

    headers = {"User-Agent": "SkyRoute/1.0 (zivg@campus.technion.ac.il)"}
    try:
        # Step 1: resolve gridpoint (permanent cache — NWS grid cells never move)
        forecast_url = _nws_gridpoint_cache.get(cache_key)
        if forecast_url is None:
            r1 = requests.get(
                f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}",
                headers=headers, timeout=8,
            )
            r1.raise_for_status()
            forecast_url = r1.json()["properties"]["forecast"]
            _nws_gridpoint_cache[cache_key] = forecast_url

        # Step 2: fetch forecast (always fresh — expires after TTL)
        r2 = requests.get(forecast_url, headers=headers, timeout=8)
        r2.raise_for_status()
        all_periods = r2.json()["properties"]["periods"]
        period = all_periods[0]
    except Exception as exc:
        log.warning("NWS weather fetch failed for %s: %s", location, exc)
        return {"error": "NWS weather service unavailable — try again shortly"}

    precip = 0
    try:
        precip = int(period.get("probabilityOfPrecipitation", {}).get("value") or 0)
    except (TypeError, ValueError):
        pass

    def _raw_period(p: dict) -> dict:
        pp = 0
        try:
            pp = int(p.get("probabilityOfPrecipitation", {}).get("value") or 0)
        except (TypeError, ValueError):
            pass
        return {
            "name":          p.get("name", ""),
            "conditions":    p.get("shortForecast", ""),
            "temperature_f": p.get("temperature", 0),
            "wind_raw":      f"{p.get('windSpeed', '')} {p.get('windDirection', '')}".strip(),
            "precip_chance": pp,
            "is_daytime":    bool(p.get("isDaytime", True)),
        }

    temp_f   = period.get("temperature", 0)
    raw_wind = f"{period.get('windSpeed', '')} {period.get('windDirection', '')}".strip()

    # Cache always stores raw imperial — unit conversion happens at read time
    raw = {
        "location":      location,
        "period":        period.get("name", "Now"),
        "conditions":    period.get("shortForecast", ""),
        "temperature_f": temp_f,
        "temperature_c": round((temp_f - 32) * 5 / 9, 1),
        "wind_raw":      raw_wind,
        "precip_chance": precip,
        "detailed":      (period.get("detailedForecast", "") or "")[:300],
        "periods_raw":   [_raw_period(p) for p in all_periods[:4]],
    }
    _nws_cache[cache_key] = (raw, _dt.datetime.utcnow())
    return _format_weather(raw, metric, units)


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


def _aerial_departure_dt(route: dict) -> datetime:
    """Best-effort UTC datetime when the helicopter segment begins.

    Uses the route's pre-flight ground leg durations to offset from now.
    If the user specified an arrival_time (HH:MM), back-computes from that
    instead, treating the time as local Eastern (UTC-4 summer approximation)
    since all routed helipads are in the NE US corridor.
    """
    now = datetime.now(timezone.utc)
    legs = route.get("legs", [])

    # Minutes of ground travel before the first helicopter leg
    heli_idx = next((i for i, l in enumerate(legs) if l.get("mode") == "helicopter"), len(legs))
    pre_flight_min = sum(legs[i].get("duration_min", 0) for i in range(heli_idx))

    arr_str = route.get("arrival_time")  # "HH:MM" or None
    if arr_str:
        try:
            h, m = map(int, arr_str.split(":"))
            total_min = route.get("total_min", 0)
            # User always types arrival time in US Eastern (the app's operating timezone).
            # Use zoneinfo so DST (EDT/EST) is handled automatically — works identically
            # whether Streamlit is running locally or on Streamlit Cloud (UTC server).
            from zoneinfo import ZoneInfo
            eastern = ZoneInfo("America/New_York")
            today_et = now.astimezone(eastern).date()
            arrival_et = datetime(today_et.year, today_et.month, today_et.day,
                                  h, m, tzinfo=eastern)
            if arrival_et.astimezone(timezone.utc) < now:
                arrival_et += timedelta(days=1)
            arrival_utc = arrival_et.astimezone(timezone.utc)
            aerial_start = arrival_utc - timedelta(minutes=total_min - pre_flight_min)
            return aerial_start
        except Exception:
            pass

    return now + timedelta(minutes=pre_flight_min)


def _tfr_active_at(tfr: dict, dt: datetime) -> bool:
    """Return True if the TFR is active (or has unknown timing) at *dt* (UTC).

    A TFR is considered inactive only when we have a confirmed expiry time
    that is strictly before *dt*.  If no time data is available, we treat the
    TFR as active (conservative — the FAA GeoServer already pre-filters to
    current/upcoming TFRs).
    """
    expires = tfr.get("expires_utc")
    effective = tfr.get("effective_utc")

    # Already expired
    if expires and dt > expires:
        return False
    # Not yet effective (don't warn about TFRs > 6 hours in the future)
    if effective and (effective - dt).total_seconds() > 6 * 3600:
        return False
    return True


def _check_tfrs_on_segment(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    departure_dt: Optional[datetime] = None,
    n_samples: int = 7,
) -> tuple[list[dict], list[dict]]:
    """Check whether the aerial segment intersects any time-active TFRs.

    Samples n_samples evenly-spaced points along the great-circle segment and
    tests each against TFR polygons that are active at *departure_dt*.
    Stadium TFRs use a 3 nm (5.6 km) radius check instead of polygon containment.

    Args:
        lat1: Departure helipad latitude.
        lon1: Departure helipad longitude.
        lat2: Arrival helipad latitude.
        lon2: Arrival helipad longitude.
        departure_dt: UTC datetime when the helicopter segment starts.
                      Defaults to now if not supplied.
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

    check_dt = departure_dt or datetime.now(timezone.utc)

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

        # Skip TFRs that are not active at the estimated departure time
        if not _tfr_active_at(tfr, check_dt):
            log.debug("TFR %s skipped — not active at %s", nid, check_dt.isoformat())
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


# ── Tool dispatcher (run_agent_v2) ────────────────────────────────────────────

def _execute_tool(
    name: str,
    args: dict,
    helipads: list[dict],
    faa_adip_df,
    result: dict,
    progress_callback=None,
) -> dict:
    """Execute one tool call and return a JSON-serializable result dict.

    Side-effects:
        - 'compute_route' and 'confirm_booking' mutate *result* in-place so
          app.py's existing route/booking display logic triggers unchanged.

    Args:
        name: Tool function name.
        args: Parsed arguments dict from the model's tool_call.
        helipads: Available helipads list (from app.py load_data).
        faa_adip_df: FAA ADIP DataFrame or None (for booking lookup).
        result: Shared result dict mutated by compute_route / confirm_booking.

    Returns:
        JSON-serializable dict passed back to the model as a tool_result.
    """
    try:
        if name == "geocode":
            address = args.get("address", "")
            coords = geocode_place(address)
            if coords is None:
                return {"error": f"Could not locate '{address}'"}
            # Try to get a display name via TomTom
            key = os.getenv("TOMTOM_API_KEY", "")
            display = address
            if key:
                try:
                    r = requests.get(
                        f"https://api.tomtom.com/search/2/search/{_urlparse.quote(address)}.json",
                        params={"key": key, "limit": 1, "countrySet": "US"},
                        timeout=6,
                    )
                    if r.ok:
                        items = r.json().get("results", [])
                        if items:
                            display = items[0].get("address", {}).get("freeformAddress", address)
                except Exception:
                    pass
            return {"lat": coords[0], "lon": coords[1], "display_name": display}

        if name == "search_nearby_places":
            places = _tool_search_places(
                location=args.get("location", ""),
                category=args.get("category", ""),
                radius_m=int(args.get("radius_m", 500)),
            )
            if places.get("results"):
                result["places"] = places
            return places

        if name == "get_weather":
            weather = _tool_get_weather(
                location=args.get("location", ""),
                units=args.get("units", "imperial"),
            )
            if not weather.get("error"):
                result["weather"] = weather
            return weather

        if name in ("compute_route", "confirm_booking"):
            origin_text = args.get("origin", "")
            dest_text   = args.get("destination", "")

            # confirm_booking short-circuit: skip geocoding when a route is
            # already cached from this turn or a previous turn (passed via
            # last_route in run_agent_v2).  This is the common case where the
            # user asks follow-up questions between planning and booking.
            if name == "confirm_booking" and result.get("route"):
                route = result["route"]
                if result.get("origin", {}).get("text"):
                    route = {**route, "origin_name": result["origin"]["text"]}
                if result.get("destination", {}).get("text"):
                    route = {**route, "dest_name": result["destination"]["text"]}
            else:
                # Geocode both ends (only needed for compute_route or
                # confirm_booking when no cached route is available)
                origin_ll = geocode_place(origin_text)
                dest_ll   = geocode_place(dest_text)
                if origin_ll is None:
                    return {"error": f"Could not locate origin '{origin_text}'"}
                if dest_ll is None:
                    return {"error": f"Could not locate destination '{dest_text}'"}

                if not helipads:
                    return {"error": "No helipad data available — data files may not be loaded"}

                route = compute_skyroute(
                    origin_ll[0], origin_ll[1],
                    dest_ll[0], dest_ll[1],
                    helipads,
                    origin_name=origin_text,
                    dest_name=dest_text,
                )
                result["route"] = route   # triggers app.py route display
                # Store geocoded endpoints with rich display info for app.py
                _orig_rich = _geocode_rich_cache.get(origin_text.lower().strip(), {})
                _dest_rich = _geocode_rich_cache.get(dest_text.lower().strip(), {})
                result["origin"] = {
                    "text": origin_text,
                    "lat":  origin_ll[0], "lon": origin_ll[1],
                    "poi_name": _orig_rich.get("poi_name", ""),
                    "address":  _orig_rich.get("address", ""),
                }
                result["destination"] = {
                    "text": dest_text,
                    "lat":  dest_ll[0],   "lon": dest_ll[1],
                    "poi_name": _dest_rich.get("poi_name", ""),
                    "address":  _dest_rich.get("address", ""),
                }
                # Inject resolved name+address into route so run_booking() can use them
                route["origin_poi_name"] = _orig_rich.get("poi_name", "")
                route["origin_address"]  = _orig_rich.get("address", "")
                route["dest_poi_name"]   = _dest_rich.get("poi_name", "")
                route["dest_address"]    = _dest_rich.get("address", "")

            # TFR + weather advisory
            advisory_parts = []
            pad_a = route.get("nearest_pad_origin")
            pad_b = route.get("nearest_pad_dest")
            departure_dt = _aerial_departure_dt(route)
            if pad_a and pad_b:
                hard, soft = _check_tfrs_on_segment(
                    float(pad_a["lat"]), float(pad_a["lon"]),
                    float(pad_b["lat"]), float(pad_b["lon"]),
                    departure_dt=departure_dt,
                )
                dep_label = departure_dt.strftime("%H:%MZ")
                if hard:
                    names = "; ".join(t.get("text", "TFR") for t in hard[:2])
                    result["tfr_warnings"] = [t.get("text", "TFR") for t in hard + soft]
                    advisory_parts.append(
                        f"AIRSPACE CLOSURE at {dep_label}: {names}"
                    )
                elif soft:
                    result["tfr_warnings"] = [t.get("text", "TFR") for t in soft]
                    advisory_parts.append(
                        f"TFR advisory at {dep_label} — check tfr.faa.gov"
                    )
                else:
                    result["tfr_warnings"] = []
            else:
                result["tfr_warnings"] = []
            result["tfrs_ignored"] = False

            # Precipitation check on route waypoints
            try:
                from src.weather import check_route_precipitation
                waypoints = []
                for leg in route.get("legs", []):
                    if leg.get("mode") == "helicopter" and pad_a and pad_b:
                        waypoints = [
                            {"lat": float(pad_a["lat"]), "lon": float(pad_a["lon"]), "label": "departure pad"},
                            {"lat": float(pad_b["lat"]), "lon": float(pad_b["lon"]), "label": "arrival pad"},
                        ]
                        break
                if waypoints:
                    precip_checks = check_route_precipitation(waypoints)
                    severe  = [c for c in precip_checks if c.get("severity") == "avoid"]
                    warned  = [c for c in precip_checks if c.get("severity") in ("warn", "avoid")]
                    if severe:
                        advisory_parts.append("Severe precipitation on aerial leg — helicopter not recommended")
                    if warned:
                        result["precip_warnings"] = [
                            f"{c['label'].title()}: precipitation intensity {c['intensity']}/255"
                            for c in warned
                        ]
            except Exception:
                pass

            advisory = " | ".join(advisory_parts) if advisory_parts else None

            if name == "compute_route":
                return {
                    "total_min": route["total_min"],
                    "drive_only_min": route["drive_only_min"],
                    "time_saved_min": route["time_saved_min"],
                    "legs": route["legs"],
                    "aerial_ok": not bool(advisory_parts),
                    "advisory": advisory,
                }

            # confirm_booking path — build quick stubs instantly (no HTTP calls).
            # The Detailed Itinerary button in app.py calls run_booking() lazily.
            booking_legs = build_quick_booking_legs(route)
            result["booking_legs"] = booking_legs   # triggers app.py booking display

            # Compute a deterministic booking reference that matches _render_quick_itinerary()
            # formula: "SR-" + md5(origin_name + dest_name + total_min)[:6].upper()
            import hashlib as _hashlib
            def _bl_dur(bl):
                if bl["mode"] == "helicopter": return bl.get("duration_min", 0)
                if bl["mode"] == "rideshare":  return bl.get("rideshare", {}).get("duration_min", 0)
                return bl.get("duration_min", 0)
            _orig_name = booking_legs[0].get("from", "") if booking_legs else ""
            _dest_name = booking_legs[-1].get("to", "")  if booking_legs else ""
            _total_min = sum(_bl_dur(bl) for bl in booking_legs)
            booking_ref = "SR-" + _hashlib.md5(
                f"{_orig_name}{_dest_name}{_total_min}".encode()
            ).hexdigest()[:6].upper()

            return {
                "confirmation_id": booking_ref,
                "legs": len(booking_legs),
                "summary": (
                    f"Booked {len(booking_legs)}-leg route: "
                    f"{origin_text} → {dest_text}. "
                    f"Reference: {booking_ref}."
                ),
            }

        return {"error": f"Unknown tool: {name}"}

    except Exception as exc:
        log.warning("Tool execution error (%s): %s", name, exc)
        return {"error": f"Tool {name} failed: {exc}"}


# ── Step 3: Python routing engine ─────────────────────────────────────────────

_SPEED_HELI_KMH  = 250.0   # matches JS simulator SPEED_HELI_KMH
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

    When an OSM helipad is geometrically closest but an FAA helipad exists
    within 50 m, the FAA entry is returned instead (FAA data is authoritative
    for ADIP lookups, METAR resolution, and IDENT-based coordination).

    Args:
        lat: Query latitude.
        lon: Query longitude.
        helipads: List of dicts each with 'lat', 'lon', 'name'.
    """
    if not helipads:
        return None
    candidates = sorted(
        [(_haversine_km(lat, lon, float(p["lat"]), float(p["lon"])), p) for p in helipads],
        key=lambda x: x[0],
    )
    best_d, best_pad = candidates[0]
    # If nearest is OSM, prefer any FAA within 50 m of the user (same physical pad)
    _FAA_PREF_KM = 0.05
    if best_pad.get("source") != "faa" and not best_pad.get("ident"):
        for d, pad in candidates:
            if d > best_d + _FAA_PREF_KM:
                break
            if pad.get("source") == "faa" or pad.get("ident"):
                best_d, best_pad = d, pad
                break
    return {**best_pad, "dist_km": round(best_d, 2)}


def compute_skyroute(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    helipads: list[dict],
    arrival_time: Optional[str] = None,
    origin_name: str = "Origin",
    dest_name: str = "Destination",
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

    # Drive-only comparison — distance-tiered speed matches real highway/suburban driving
    # (ground approach legs above stay at _SPEED_DRIVE_KMH=25 km/h; those are short urban hops)
    _d_total_km = _haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
    if _d_total_km < 20:
        _cmp_speed = 30.0   # dense urban
    elif _d_total_km < 60:
        _cmp_speed = 55.0   # suburban / mixed
    else:
        _cmp_speed = 70.0   # highway
    drive_only_min = _d_total_km * _ROAD_FACTOR / _cmp_speed * 60
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
            "from": origin_name,
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
            "to": dest_name,
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
        "origin_name": origin_name,
        "dest_name": dest_name,
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


# ── Level 1 agentic loop ──────────────────────────────────────────────────────

import re as _re_tool


def _recover_tool_use_failed(
    exc,
    client,
    model: str,
    messages: list,
    helipads,
    faa_adip_df,
    result: dict,
    status_callback,
) -> Optional[object]:
    """Recover from a Groq 400 tool_use_failed error on the 8B model.

    The 8B model sometimes generates ``<function=name>{args}`` instead of the
    OpenAI tool-calling schema.  Groq rejects it with 400/tool_use_failed but
    includes the attempted call in ``failed_generation``.  We parse it, execute
    the tool manually, append a synthetic tool-result turn, then ask the model
    to format the result as plain text.

    Returns:
        A ChatCompletion response object on success, None if not recoverable.
    """
    err_str = str(exc).lower()
    if "tool_use_failed" not in err_str and "failed to call a function" not in err_str:
        return None

    fg = ""
    if hasattr(exc, "response") and exc.response is not None:
        try:
            fg = exc.response.json().get("error", {}).get("failed_generation", "")
        except Exception:
            pass
    if not fg:
        raw = str(exc)
        # Try both quote styles; use re.DOTALL so newlines inside the value are matched
        for _pat in (
            r"'failed_generation':\s*'(.*?)'(?=\s*[,}])",
            r'"failed_generation":\s*"(.*?)"(?=\s*[,}])',
        ):
            _m = _re_tool.search(_pat, raw, _re_tool.DOTALL)
            if _m:
                fg = _m.group(1)
                break
    if not fg:
        return None

    m = _re_tool.search(r"<function=(\w+)>(.*)", fg.strip(), _re_tool.DOTALL)
    if not m:
        return None

    try:
        tool_name = m.group(1)
        tool_args = json.loads(m.group(2))
    except Exception:
        return None

    log.info("8B tool_use_failed recovery: executing %s(%s)", tool_name, list(tool_args))

    if status_callback:
        status_callback("tool_call", {"name": tool_name, "args": tool_args})

    tool_result = _execute_tool(
        name=tool_name,
        args=tool_args,
        helipads=helipads,
        faa_adip_df=faa_adip_df,
        result=result,
        progress_callback=status_callback,
    )

    if status_callback:
        status_callback("tool_result", {"name": tool_name, "result": tool_result})

    _fake_id = "recovered_0"
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": _fake_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": _fake_id,
        "content": json.dumps(tool_result),
    })

    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
    except Exception as exc_fmt:
        log.error("8B formatting call after tool_use_failed recovery failed: %s", exc_fmt)
        return None


def run_agent_v2(
    user_text: str,
    helipads: list[dict],
    history: list[dict] | None = None,
    faa_adip_df=None,
    status_callback=None,
    location_hint: str = "",
    last_route: dict | None = None,
    last_origin: dict | None = None,
    last_destination: dict | None = None,
) -> dict:
    """Tool-calling concierge: natural language → tool dispatch → response.

    The model autonomously decides which tools to call (search_nearby_places,
    get_weather, compute_route, confirm_booking, geocode) based on the user's
    intent.  All routing math, TFR checks, and booking flows remain in Python;
    the LLM orchestrates the sequence and formats the final answer.

    Args:
        user_text: Free-form user message.
        helipads: Available helipads list (lat, lon, name, ident).
        history: Prior conversation messages for multi-turn context.
                 Each dict must have 'role' and 'content' keys.
                 Tool-call turns from previous rounds are included automatically.
        faa_adip_df: FAA ADIP DataFrame for helipad booking lookup (or None).
        status_callback: Optional callable(event, data) for live UI progress.
            Called with event in {"thinking", "tool_call", "tool_result"}.
            data keys: iteration (thinking), name+args (tool_call), name+result (tool_result).

    Returns:
        Dict with keys:
          response (str | None): final model text shown in chat
          route (dict | None): compute_skyroute() output if routing was triggered
          booking_legs (list | None): run_booking() output if booking was triggered
          error (str | None): user-friendly error message
          tfr_warnings (list[str]): TFR advisory strings
          tfrs_ignored (bool): True if hard TFRs were overridden
          _messages (list[dict]): full conversation messages for next-turn history
    """
    result: dict = {
        "response": None,
        "route":       last_route,       # pre-seed from previous turn so confirm_booking
        "origin":      last_origin,      # can skip geocoding + recompute
        "destination": last_destination,
        "booking_legs": None,
        "error": None,
        "tfr_warnings": [],
        "tfrs_ignored": False,
        "precip_warnings": [],
        "model_warnings": [],
        "_messages": [],
    }

    client = _get_groq_client()
    if client is None:
        result["error"] = (
            "Route Assistant is not configured — add GROQ_API_KEY to .env "
            "(local) or Streamlit Cloud Secrets."
        )
        return result

    # Build messages: system + trimmed history + current user turn.
    # Keep only last 6 messages (3 turns) — booking tool results are verbose JSON
    # and 10 messages easily exceeds the 70b 6,000 TPM free-tier limit.
    _sys = (_CONCIERGE_SYSTEM + f"\n\nSession context: {location_hint}"
            if location_hint else _CONCIERGE_SYSTEM)
    messages: list[dict] = [{"role": "system", "content": _sys}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_text})

    max_iterations = 8
    active_model = _GROQ_MODEL_70B

    for iteration in range(max_iterations):
        if status_callback:
            status_callback("thinking", {"iteration": iteration})
        try:
            resp = client.chat.completions.create(
                model=active_model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0,
            )
        except Exception as exc:
            import time as _time
            err_str = str(exc).lower()
            # ── 70b failure ───────────────────────────────────────────────────
            if active_model == _GROQ_MODEL_70B:
                _is_not_enabled = any(kw in err_str for kw in (
                    "model_not_found", "does not exist", "blocked",
                    "permission", "403",
                ))
                _is_rate_limit = "rate_limit" in err_str or "429" in err_str

                if _is_not_enabled:
                    # Hard misconfiguration — fall back immediately and warn the user
                    _warn = (
                        f"⚠️ {_GROQ_MODEL_70B} is not enabled in this Groq project. "
                        "Enable it at console.groq.com/settings/project/limits. "
                        f"Falling back to {_GROQ_MODEL}."
                    )
                    log.warning(_warn)
                    result["model_warnings"].append(_warn)

                elif _is_rate_limit:
                    # Rate limit — extract retry-after and retry 70b once before giving up on it
                    _retry_after = 5
                    try:
                        if hasattr(exc, "response") and exc.response is not None:
                            _retry_after = int(exc.response.headers.get("retry-after", 5))
                    except Exception:
                        pass
                    _retry_after = min(_retry_after, 10)  # cap at 10 s for demo UX
                    log.warning(
                        "Groq 70b rate limit — retrying in %ds.", _retry_after,
                    )
                    _time.sleep(_retry_after)
                    try:
                        resp = client.chat.completions.create(
                            model=_GROQ_MODEL_70B,
                            messages=messages,
                            tools=TOOL_SCHEMAS,
                            tool_choice="auto",
                            temperature=0,
                        )
                        # Retry succeeded — continue the loop with this response
                        active_model = _GROQ_MODEL_70B
                        # jump to the response-processing code below
                        msg = resp.choices[0].message
                        if not msg.tool_calls:
                            result["response"] = (msg.content or "").strip()
                            messages.append({"role": "assistant", "content": result["response"]})
                            break
                        assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
                        assistant_msg["tool_calls"] = [
                            {"id": tc.id, "type": "function",
                             "function": {"name": tc.function.name,
                                          "arguments": tc.function.arguments}}
                            for tc in msg.tool_calls
                        ]
                        messages.append(assistant_msg)
                        for tc in msg.tool_calls:
                            if status_callback:
                                status_callback("tool_call", {
                                    "name": tc.function.name,
                                    "args": json.loads(tc.function.arguments or "{}"),
                                })
                            tool_result = _execute_tool(
                                name=tc.function.name,
                                args=json.loads(tc.function.arguments or "{}"),
                                helipads=helipads,
                                faa_adip_df=faa_adip_df,
                                result=result,
                                progress_callback=status_callback,
                            )
                            if status_callback:
                                status_callback("tool_result", {
                                    "name": tc.function.name, "result": tool_result,
                                })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(tool_result),
                            })
                        continue  # next iteration with 70b still active
                    except Exception:
                        _warn = (
                            f"⚠️ {_GROQ_MODEL_70B} rate limit persists — "
                            f"falling back to {_GROQ_MODEL} for this request."
                        )
                        log.warning(_warn)
                        result["model_warnings"].append(_warn)

                elif "tool_use_failed" in err_str or "failed to call a function" in err_str:
                    # 70b produced a malformed tool call — try the same parse-and-recover
                    # logic used for 8b before giving up and falling back.
                    log.warning("Groq 70b tool_use_failed — attempting parse recovery.")
                    recovered = _recover_tool_use_failed(
                        exc, client, _GROQ_MODEL_70B, messages,
                        helipads, faa_adip_df, result, status_callback,
                    )
                    if recovered is not None:
                        resp = recovered
                        continue
                    log.warning("70b tool_use_failed unrecoverable — falling back to %s.", _GROQ_MODEL)

                else:
                    # Transient error (timeout, 503, network) — fall back silently
                    log.warning("Groq 70b transient error (%s) — falling back to %s.", exc, _GROQ_MODEL)

                active_model = _GROQ_MODEL
                try:
                    resp = client.chat.completions.create(
                        model=active_model,
                        messages=messages,
                        tools=TOOL_SCHEMAS,
                        tool_choice="auto",
                        temperature=0,
                    )
                except Exception as exc2:
                    exc2_str = str(exc2).lower()
                    # If 8b is ALSO rate-limited, extract retry-after and wait, then retry
                    if "rate_limit" in exc2_str or "429" in exc2_str:
                        import time as _time
                        wait_s = 5
                        try:
                            if hasattr(exc2, "response") and exc2.response is not None:
                                wait_s = int(exc2.response.headers.get("retry-after", 5))
                        except Exception:
                            pass
                        wait_s = min(wait_s, 15)   # cap at 15 s during demo
                        log.warning("8b rate limit — waiting %ds before retry.", wait_s)
                        _time.sleep(wait_s)
                        try:
                            resp = client.chat.completions.create(
                                model=active_model,
                                messages=messages,
                                tools=TOOL_SCHEMAS,
                                tool_choice="auto",
                                temperature=0,
                            )
                        except Exception as exc2b:
                            log.error("8b still rate-limited after wait: %s", exc2b)
                            result["error"] = "Service is busy — please try again in a moment."
                            return result
                    else:
                        # 8B malformed tool call — parse failed_generation and recover
                        recovered = _recover_tool_use_failed(
                            exc2, client, active_model, messages,
                            helipads, faa_adip_df, result, status_callback,
                        )
                        if recovered is not None:
                            resp = recovered
                        else:
                            # Last resort: ask 8b for a plain-text response with no tools
                            log.warning("8b tool call failed — retrying without tools.")
                            try:
                                resp = client.chat.completions.create(
                                    model=active_model,
                                    messages=messages,
                                    temperature=0,
                                )
                            except Exception as exc3:
                                log.error("Groq 8b plain-text fallback failed: %s", exc3)
                                result["error"] = "Route Assistant is temporarily unavailable."
                                return result
            # ── 8b is active model and produced a malformed tool call ─────────
            elif "tool_use_failed" in err_str or "failed to call a function" in err_str:
                recovered = _recover_tool_use_failed(
                    exc, client, active_model, messages,
                    helipads, faa_adip_df, result, status_callback,
                )
                if recovered is not None:
                    resp = recovered
                else:
                    # Last resort: plain-text response with no tools
                    log.warning("8b tool_use_failed unrecoverable — retrying without tools.")
                    try:
                        resp = client.chat.completions.create(
                            model=active_model,
                            messages=messages,
                            temperature=0,
                        )
                    except Exception as exc_plain:
                        log.error("Groq plain-text fallback failed: %s", exc_plain)
                        result["error"] = "Route Assistant is temporarily unavailable."
                        return result
            else:
                log.error("Groq API error (iteration %d): %s", iteration, exc)
                result["error"] = "Route Assistant is temporarily unavailable."
                return result

        msg = resp.choices[0].message

        if not msg.tool_calls:
            # Model returned a final text response — done
            result["response"] = (msg.content or "").strip()
            messages.append({"role": "assistant", "content": result["response"]})
            break

        # Append the assistant's tool-call message
        assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
        # Groq returns tool_calls as objects; serialise for the messages list
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
        messages.append(assistant_msg)

        # Execute each tool and append results
        for tc in msg.tool_calls:
            try:
                tool_args = json.loads(tc.function.arguments)
            except Exception:
                tool_args = {}

            if status_callback:
                status_callback("tool_call", {"name": tc.function.name, "args": tool_args})

            tool_result = _execute_tool(
                name=tc.function.name,
                args=tool_args,
                helipads=helipads,
                faa_adip_df=faa_adip_df,
                result=result,
                progress_callback=status_callback,
            )

            if status_callback:
                status_callback("tool_result", {"name": tc.function.name, "result": tool_result})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_result),
            })
    else:
        log.warning("run_agent_v2 hit max_iterations (%d) without final response", max_iterations)
        result["error"] = "Took too many steps — please rephrase your request."

    # Store the full conversation (minus system prompt) for next-turn history
    result["_messages"] = messages[1:]   # strip system message

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


def build_quick_booking_legs(route: dict) -> list[dict]:
    """Build instant booking leg stubs from a computed route — no API calls.

    Returns the same structure as run_booking() but without ADIP data,
    Mapillary thumbnails, or METAR. Call run_booking() to enrich these stubs.

    Args:
        route: Output of compute_skyroute().

    Returns:
        List of quick booking leg dicts, one per route leg.
    """
    quick_legs: list[dict] = []

    for i, leg in enumerate(route.get("legs", [])):
        if leg["mode"] == "helicopter":
            pad_a = route.get("nearest_pad_origin", {})
            pad_b = route.get("nearest_pad_dest", {})

            def _basic(pad: dict) -> dict:
                lat    = float(pad.get("lat", 0))
                lon    = float(pad.get("lon", 0))
                ident  = pad.get("ident") or pad.get("IDENT") or ""
                osm_id = str(pad.get("osm_id", "") or "").strip()
                return {
                    "name":          pad.get("name") or ident or "Helipad",
                    "ident":         ident,
                    "osm_id":        osm_id,
                    "lat":           lat,
                    "lon":           lon,
                    "gmaps_url":     f"https://maps.google.com/?q={lat},{lon}&z=18",
                    "adip_url":      (f"https://adip.faa.gov/agis/public/#/simpleAirportMap/{ident}"
                                      if ident else ""),
                    "status":        "—",
                    "servcity":      "—",
                    "ownership":     "—",
                    "private_use":   False,
                    "address":       "",
                    "contact_notes": "",
                    "mly_image_id":  "",
                    "mly_thumb_url": "",
                }

            quick_legs.append({
                "leg_index":         i,
                "mode":              "helicopter",
                "from":              leg["from"],
                "to":                leg["to"],
                "duration_min":      leg["duration_min"],
                "dist_km":           leg["dist_km"],
                "departure_helipad": _basic(pad_a),
                "arrival_helipad":   _basic(pad_b),
                "metar_dep":         None,
                "metar_arr":         None,
            })

        elif leg["mode"] in ("drive", "walk"):
            if i == 0:
                pad = route.get("nearest_pad_origin", {})
                o   = route["origin"]
                pickup_lat, pickup_lon   = o["lat"], o["lon"]
                dropoff_lat, dropoff_lon = pad["lat"], pad["lon"]
                _orig_poi  = route.get("origin_poi_name", "")
                _orig_raw  = route.get("origin_name") or "Your origin"
                from_name  = f"Origin: {_orig_poi or _orig_raw}"
                to_name    = pad.get("name") or pad.get("ident") or leg["to"]
            else:
                pad = route.get("nearest_pad_dest", {})
                d   = route["dest"]
                pickup_lat, pickup_lon   = pad["lat"], pad["lon"]
                dropoff_lat, dropoff_lon = d["lat"], d["lon"]
                _dest_poi  = route.get("dest_poi_name", "")
                _dest_raw  = route.get("dest_name") or "Your destination"
                from_name  = pad.get("name") or pad.get("ident") or leg["from"]
                to_name    = f"Destination: {_dest_poi or _dest_raw}"

            if leg["mode"] == "walk":
                quick_legs.append({
                    "leg_index":        i,
                    "mode":             "walk",
                    "from":             from_name,
                    "to":               to_name,
                    "dist_km":          leg["dist_km"],
                    "duration_min":     leg["duration_min"],
                    "pickup_lat":       pickup_lat,
                    "pickup_lon":       pickup_lon,
                    "dropoff_lat":      dropoff_lat,
                    "dropoff_lon":      dropoff_lon,
                    "pickup_mly_id":    "",
                    "pickup_mly_thumb": "",
                })
            else:
                rs = simulate_rideshare(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
                quick_legs.append({
                    "leg_index":         i,
                    "mode":              "rideshare",
                    "from":              from_name,
                    "to":                to_name,
                    "duration_min":      rs["duration_min"],  # top-level for quick card
                    "dist_km":           leg["dist_km"],
                    "pickup_lat":        pickup_lat,
                    "pickup_lon":        pickup_lon,
                    "dropoff_lat":       dropoff_lat,
                    "dropoff_lon":       dropoff_lon,
                    "pickup_mly_id":     "",
                    "pickup_mly_thumb":  "",
                    "dropoff_mly_id":    "",
                    "dropoff_mly_thumb": "",
                    "rideshare":         rs,
                })

    return quick_legs


def run_booking(route: dict, faa_adip_df=None, progress_callback=None) -> list[dict]:
    """Build a per-leg booking plan for a computed route.

    Drive legs → rideshare simulation (Uber / Waymo).
    Helicopter legs → helipad coordination info (ADIP + Mapillary).

    HTTP calls within each leg are parallelised using ThreadPoolExecutor:
    ADIP lookups and Mapillary ID fetches run concurrently (phase 1), then
    Mapillary thumbnail fetches run concurrently (phase 2). This reduces
    a typical 3-leg route from ~28 serial calls to 2 parallel batches.

    Args:
        route: Output of compute_skyroute().
        faa_adip_df: Optional FAA ADIP enriched DataFrame for contact lookup.
        progress_callback: Optional callable(event, data) for live UI updates.

    Returns:
        List of booking dicts, one per leg.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _emit(msg: str) -> None:
        if progress_callback:
            progress_callback("booking_step", {"msg": msg})

    booking_legs: list[dict] = []
    legs = route.get("legs", [])
    n_legs = len(legs)
    _emit(f"Processing {n_legs}-leg route…")

    for i, leg in enumerate(legs):
        if leg["mode"] == "helicopter":
            pad_a = route.get("nearest_pad_origin", {})
            pad_b = route.get("nearest_pad_dest", {})
            dep_label = pad_a.get("name") or pad_a.get("ident") or "departure helipad"
            arr_label = pad_b.get("name") or pad_b.get("ident") or "arrival helipad"

            _emit(f"🚁 Leg {i+1}/{n_legs} — helicopter · {leg['from']} → {leg['to']}")
            _emit(f"   ↳ Fetching helipad status + imagery ({dep_label} & {arr_label})…")

            # Phase 1: ADIP lookups + Mapillary ID fetches + METAR — all independent, run in parallel
            from src.notam import fetch_metar as _fetch_metar
            _dep_ident = pad_a.get("ident") or pad_a.get("IDENT") or ""
            _arr_ident = pad_b.get("ident") or pad_b.get("IDENT") or ""

            # Aviation Weather Center uses ICAO codes, not FAA idents.
            # Look up ICAO_ID from the enriched DataFrame (instant pandas lookup)
            # so METAR can be fetched in parallel alongside the ADIP call.
            def _icao_for(faa_id: str) -> str:
                if not faa_id or faa_adip_df is None:
                    return ""
                rows = faa_adip_df[faa_adip_df["IDENT"] == faa_id]
                if rows.empty:
                    return ""
                return str(rows.iloc[0].get("ICAO_ID", "") or "").strip()

            _dep_metar_id = _icao_for(_dep_ident) or _dep_ident
            _arr_metar_id = _icao_for(_arr_ident) or _arr_ident

            with ThreadPoolExecutor(max_workers=8) as ex:
                f_dep        = ex.submit(lookup_helipad_info,
                                         pad_a.get("ident"), pad_a.get("name"),
                                         pad_a["lat"], pad_a["lon"], faa_adip_df)
                f_arr        = ex.submit(lookup_helipad_info,
                                         pad_b.get("ident"), pad_b.get("name"),
                                         pad_b["lat"], pad_b["lon"], faa_adip_df)
                f_mly_a      = ex.submit(find_nearest_mapillary_image, pad_a["lat"], pad_a["lon"])
                f_mly_b      = ex.submit(find_nearest_mapillary_image, pad_b["lat"], pad_b["lon"])
                f_metar_dep  = ex.submit(_fetch_metar, _dep_metar_id) if _dep_metar_id else None
                f_metar_arr  = ex.submit(_fetch_metar, _arr_metar_id) if _arr_metar_id else None
                # Reverse geocode OSM-only helipads (no FAA ident) to get street address
                f_rg_dep     = ex.submit(_reverse_geocode_tomtom, pad_a["lat"], pad_a["lon"]) if not _dep_ident else None
                f_rg_arr     = ex.submit(_reverse_geocode_tomtom, pad_b["lat"], pad_b["lon"]) if not _arr_ident else None

            dep_info  = f_dep.result()
            arr_info  = f_arr.result()
            if f_rg_dep:
                dep_info["address"] = f_rg_dep.result().get("address", "")
            if f_rg_arr:
                arr_info["address"] = f_rg_arr.result().get("address", "")
            mly_a_id  = f_mly_a.result() or ""
            mly_b_id  = f_mly_b.result() or ""
            metar_dep = f_metar_dep.result() if f_metar_dep else None
            metar_arr = f_metar_arr.result() if f_metar_arr else None

            dep_status = dep_info.get("status") or "unknown"
            arr_status = arr_info.get("status") or "unknown"
            dep_coord  = dep_info.get("coordination_note") or dep_info.get("contact") or ""
            arr_coord  = arr_info.get("coordination_note") or arr_info.get("contact") or ""
            _emit(f"   ↳ {dep_label}: {dep_status}" + (f" · {dep_coord[:60]}" if dep_coord else ""))
            _emit(f"   ↳ {arr_label}: {arr_status}" + (f" · {arr_coord[:60]}" if arr_coord else ""))
            _emit(f"   ↳ Fetching street-level imagery…")

            # Phase 2: Mapillary thumbnails — run in parallel
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_ta = ex.submit(get_mapillary_thumb_url, mly_a_id)
                f_tb = ex.submit(get_mapillary_thumb_url, mly_b_id)

            dep_info["mly_image_id"]  = mly_a_id
            dep_info["mly_thumb_url"] = f_ta.result() or ""
            arr_info["mly_image_id"]  = mly_b_id
            arr_info["mly_thumb_url"] = f_tb.result() or ""
            _emit(f"   ↳ Imagery ready · leg {i+1} confirmed ✓")

            booking_legs.append({
                "leg_index": i,
                "mode": "helicopter",
                "from": leg["from"],
                "to": leg["to"],
                "duration_min": leg["duration_min"],
                "dist_km": leg["dist_km"],
                "departure_helipad": dep_info,
                "arrival_helipad": arr_info,
                "metar_dep": metar_dep,
                "metar_arr": metar_arr,
            })

        elif leg["mode"] in ("drive", "walk"):
            if i == 0:
                pad = route.get("nearest_pad_origin", {})
                o = route["origin"]
                pickup_lat, pickup_lon = o["lat"], o["lon"]
                dropoff_lat, dropoff_lon = pad["lat"], pad["lon"]
                _orig_poi  = route.get("origin_poi_name", "")
                _orig_addr = route.get("origin_address", "")
                _orig_raw  = route.get("origin_name") or "Your origin"
                _origin_label = (_orig_poi or _orig_raw) + (f"  \n{_orig_addr}" if _orig_addr else "")
                from_name = f"Origin: {_origin_label}"
                to_name   = pad.get("name") or pad.get("ident") or leg["to"]
            else:
                pad = route.get("nearest_pad_dest", {})
                d = route["dest"]
                pickup_lat, pickup_lon = pad["lat"], pad["lon"]
                dropoff_lat, dropoff_lon = d["lat"], d["lon"]
                from_name = pad.get("name") or pad.get("ident") or leg["from"]
                _dest_poi  = route.get("dest_poi_name", "")
                _dest_addr = route.get("dest_address", "")
                _dest_raw  = route.get("dest_name") or "Your destination"
                _dest_label = (_dest_poi or _dest_raw) + (f"  \n{_dest_addr}" if _dest_addr else "")
                to_name   = f"Destination: {_dest_label}"

            _mode_icon = "🚶" if leg["mode"] == "walk" else "🚗"
            _emit(f"{_mode_icon} Leg {i+1}/{n_legs} — {leg['mode']} · {leg['from']} → {leg['to']}")
            _emit(f"   ↳ {leg['dist_km']} km · ~{leg['duration_min']} min · fetching imagery…")

            _pu_embed, _pu_url = _mapillary_embed(pickup_lat, pickup_lon)
            _do_embed, _do_url = _mapillary_embed(dropoff_lat, dropoff_lon)

            # Phase 1: Mapillary ID lookups in parallel
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_pu_id = ex.submit(find_nearest_mapillary_image, pickup_lat, pickup_lon)
                f_do_id = ex.submit(find_nearest_mapillary_image, dropoff_lat, dropoff_lon)

            _pu_mly_id = f_pu_id.result() or ""
            _do_mly_id = f_do_id.result() or ""

            # Phase 2: Mapillary thumbnails in parallel
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_pu_th = ex.submit(get_mapillary_thumb_url, _pu_mly_id)
                f_do_th = ex.submit(get_mapillary_thumb_url, _do_mly_id)

            _pu_mly_thumb = f_pu_th.result() or ""
            _do_mly_thumb = f_do_th.result() or ""

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
                    "dropoff_mly_id": _do_mly_id,
                    "dropoff_mly_thumb": _do_mly_thumb,
                })
                _emit(f"   ↳ Walk {leg['dist_km']} km · leg {i+1} confirmed ✓")
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
                _emit(f"   ↳ Rideshare: {rideshare['fare_range']} · {rideshare['duration_min']} min · leg {i+1} confirmed ✓")

    _emit(f"All {n_legs} legs confirmed — booking reference ready")
    return booking_legs
