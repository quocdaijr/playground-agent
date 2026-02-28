"""
PlaygroundAgent — Tool Registry
==========================
Local tools (pure Python, no I/O):
  calculator       — safe sandboxed math eval
  get_current_time — UTC timestamp
  word_counter     — text statistics
  list_operations  — array operations on comma-separated numbers

External tools (async HTTP via httpx, all free / no API key required):
  web_search       — DuckDuckGo Instant Answer API
  get_weather      — Open-Meteo (city name resolved via geocoding API)
  currency_exchange — Frankfurter.app (European Central Bank rates)
  ip_lookup        — ip-api.com (geolocation, ISP, ASN)

All external calls share a single reusable AsyncClient with:
  - 10s timeout
  - Up to 3 retries with exponential backoff on timeout / network errors
  - Structured JSON error returns — never raises into the graph
"""

import json
import math
from datetime import datetime
from functools import wraps

import httpx
import structlog
from langchain_core.tools import tool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

# ── Shared HTTP client ─────────────────────────────────────────────────────────

_http: httpx.AsyncClient | None = None


def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            headers={"User-Agent": "PlaygroundAgent/1.0"},
            follow_redirects=True,
        )
    return _http


async def close_http() -> None:
    global _http
    if _http and not _http.is_closed:
        await _http.aclose()
        _http = None


def _external(fn):
    """Wrap an async tool function with retry logic and structured error handling."""

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        before_sleep=lambda rs: logger.warning(
            "tool_retry", tool=fn.__name__, attempt=rs.attempt_number
        ),
    )
    @wraps(fn)
    async def wrapper(*args, **kwargs) -> str:
        try:
            return await fn(*args, **kwargs)
        except httpx.TimeoutException:
            return json.dumps({"error": "Request timed out. Please try again."})
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"})
        except Exception as exc:
            logger.error("tool_error", tool=fn.__name__, error=str(exc))
            return json.dumps({"error": str(exc)})

    return wrapper


# ── Local tools ────────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely in a sandboxed environment.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, log10, abs, round, pi, e
    Examples: 'sqrt(2025)', '2 ** 10', 'round(3.14159, 2)', 'sin(pi / 2)'
    """
    safe_builtins = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "abs": abs,
        "round": round,
        "pow": pow,
        "min": min,
        "max": max,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_builtins)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Return the current UTC date and time."""
    now = datetime.utcnow()
    return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def word_counter(text: str) -> str:
    """
    Analyse a text string and return word count, character count,
    sentence count, and average word length.
    """
    words = len(text.split())
    chars = len(text)
    chars_no_space = len(text.replace(" ", ""))
    sentences = text.count(".") + text.count("!") + text.count("?")
    return json.dumps(
        {
            "words": words,
            "characters": chars,
            "characters_without_spaces": chars_no_space,
            "sentences": max(sentences, 1),
            "avg_word_length": round(chars_no_space / max(words, 1), 2),
        },
        indent=2,
    )


@tool
def list_operations(items: str, operation: str) -> str:
    """
    Perform an operation on a comma-separated list of numbers.
    Supported operations: sum, average, min, max, sort, reverse, unique
    Example: items='1,5,3,2,4', operation='sort'
    """
    try:
        nums = [float(x.strip()) for x in items.split(",")]
    except ValueError:
        parts = [x.strip() for x in items.split(",")]
        str_ops = {
            "sort": sorted,
            "reverse": lambda x: list(reversed(x)),
            "unique": lambda x: list(dict.fromkeys(x)),
        }
        fn = str_ops.get(operation)
        return str(fn(parts)) if fn else f"Unknown operation: {operation}"

    ops = {
        "sum": lambda x: sum(x),
        "average": lambda x: sum(x) / len(x),
        "min": min,
        "max": max,
        "sort": sorted,
        "reverse": lambda x: list(reversed(x)),
        "unique": lambda x: list(dict.fromkeys(x)),
    }
    fn = ops.get(operation)
    if fn is None:
        return f"Unknown operation. Choose from: {', '.join(ops)}"
    return str(fn(nums))


# ── External tools ─────────────────────────────────────────────────────────────

@tool
async def web_search(query: str) -> str:
    """
    Search the web using the DuckDuckGo Instant Answer API.
    Returns a summary, direct answer (if available), and related topics.
    Best for: definitions, quick facts, short lookups.
    Example: query='What is LangGraph?'
    """
    @_external
    async def _fetch(q: str) -> str:
        r = await get_http().get(
            "https://api.duckduckgo.com/",
            params={"q": q, "format": "json", "no_html": "1", "skip_disambig": "1"},
        )
        r.raise_for_status()
        data = r.json()

        result: dict = {"query": q}

        if data.get("AbstractText"):
            result["summary"] = data["AbstractText"]
            result["source"] = data.get("AbstractSource", "")
            result["url"] = data.get("AbstractURL", "")

        if data.get("Answer"):
            result["answer"] = data["Answer"]

        topics = [
            t.get("Text", "")
            for t in data.get("RelatedTopics", [])
            if isinstance(t, dict) and t.get("Text")
        ][:5]
        if topics:
            result["related"] = topics

        if not result.get("summary") and not result.get("answer") and not topics:
            result["note"] = "No instant answer found. Try a more specific query."

        return json.dumps(result, ensure_ascii=False, indent=2)

    return await _fetch(query)


# WMO weather code → human-readable description
_WMO_CODES: dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Heavy thunderstorm with hail",
}


@tool
async def get_weather(location: str) -> str:
    """
    Fetch current weather conditions using the Open-Meteo API (free, no key required).
    Accepts a city name (e.g. 'Tokyo') or coordinates as 'lat,lon' (e.g. '35.68,139.69').
    Returns temperature, feels-like, humidity, wind speed, precipitation, and condition.
    """
    @_external
    async def _fetch(loc: str) -> str:
        http = get_http()

        # Resolve city name to coordinates via Open-Meteo geocoding
        parts = loc.split(",")
        if (
            len(parts) == 2
            and all(p.strip().replace(".", "").replace("-", "").isdigit() for p in parts)
        ):
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            city_name = loc
        else:
            geo = await http.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": loc, "count": 1, "language": "en", "format": "json"},
            )
            geo.raise_for_status()
            results = geo.json().get("results", [])
            if not results:
                return json.dumps({"error": f"Location not found: {loc}"})
            r0 = results[0]
            lat, lon = r0["latitude"], r0["longitude"]
            city_name = f"{r0['name']}, {r0.get('country', '')}"

        weather = await http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,apparent_temperature,"
                    "relative_humidity_2m,windspeed_10m,"
                    "weathercode,precipitation"
                ),
                "timezone": "auto",
            },
        )
        weather.raise_for_status()
        data = weather.json()
        cur = data["current"]

        return json.dumps(
            {
                "location": city_name,
                "coordinates": {"lat": lat, "lon": lon},
                "temperature_c": cur["temperature_2m"],
                "feels_like_c": cur["apparent_temperature"],
                "humidity_pct": cur["relative_humidity_2m"],
                "wind_speed_kmh": cur["windspeed_10m"],
                "precipitation_mm": cur["precipitation"],
                "condition": _WMO_CODES.get(cur["weathercode"], f"Code {cur['weathercode']}"),
                "timezone": data.get("timezone", ""),
            },
            ensure_ascii=False,
            indent=2,
        )

    return await _fetch(location)


@tool
async def currency_exchange(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert between currencies using the Frankfurter API (European Central Bank rates).
    Supports 30+ currencies: USD, EUR, GBP, JPY, AUD, SGD, CHF, CNY, KRW, THB, etc.
    Note: VND is not in the ECB basket — use SGD or THB as SEA proxies.
    Example: amount=100, from_currency='USD', to_currency='EUR'
    """
    @_external
    async def _fetch(amount: float, src: str, dst: str) -> str:
        src, dst = src.upper().strip(), dst.upper().strip()
        r = await get_http().get(
            "https://api.frankfurter.app/latest",
            params={"from": src, "to": dst},
        )
        r.raise_for_status()
        data = r.json()

        rate = data["rates"].get(dst)
        if rate is None:
            available = ", ".join(sorted(data["rates"].keys()))
            return json.dumps({"error": f"{dst} not supported. Available: {available}"})

        return json.dumps(
            {
                "from": f"{amount} {src}",
                "to": f"{round(amount * rate, 4)} {dst}",
                "rate": rate,
                "date": data.get("date", ""),
                "source": "European Central Bank via Frankfurter.app",
            },
            indent=2,
        )

    return await _fetch(amount, from_currency, to_currency)


@tool
async def ip_lookup(ip_address: str) -> str:
    """
    Look up geolocation and network metadata for an IP address via ip-api.com (free).
    Returns country, region, city, coordinates, timezone, ISP, organization, and ASN.
    Example: ip_address='8.8.8.8'
    """
    @_external
    async def _fetch(ip: str) -> str:
        r = await get_http().get(
            f"http://ip-api.com/json/{ip.strip()}",
            params={
                "fields": "status,message,country,countryCode,regionName,city,zip,lat,lon,timezone,isp,org,as"
            },
        )
        r.raise_for_status()
        data = r.json()

        if data.get("status") == "fail":
            return json.dumps({"error": data.get("message", "Lookup failed"), "ip": ip})

        return json.dumps(
            {
                "ip": ip,
                "country": f"{data.get('country')} ({data.get('countryCode')})",
                "region": data.get("regionName"),
                "city": data.get("city"),
                "zip": data.get("zip"),
                "coordinates": {"lat": data.get("lat"), "lon": data.get("lon")},
                "timezone": data.get("timezone"),
                "isp": data.get("isp"),
                "org": data.get("org"),
                "asn": data.get("as"),
            },
            ensure_ascii=False,
            indent=2,
        )

    return await _fetch(ip_address)


# ── Tool registry ──────────────────────────────────────────────────────────────

LOCAL_TOOLS = [calculator, get_current_time, word_counter, list_operations]
EXTERNAL_TOOLS = [web_search, get_weather, currency_exchange, ip_lookup]
TOOLS = LOCAL_TOOLS + EXTERNAL_TOOLS
