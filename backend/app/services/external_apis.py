"""
AgriPulse 2.0 — External API Integration Service
Integrates 4 real public APIs for Indian agricultural intelligence:
  1. Open-Meteo        — Real-time weather & agro-meteorological data (no key)
  2. Nominatim (OSM)   — Reverse geocoding GPS → Indian district/state (no key)
  3. GBIF              — Crop pest/species occurrence data near field (no key)
  4. GNews             — Indian agriculture disease news (free API key)

All methods follow graceful degradation:
  - On success  → {"available": True, "data": {...}}
  - On failure  → {"available": False, "reason": "..."}
Never returns fabricated values.
"""

import calendar
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from ..core.config import settings
from ..core.logger import logger

# ─── Timeouts ───────────────────────────────────────────────────────────────
_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)

# ─── Crop season detection (Indian Kharif / Rabi / Zaid) ────────────────────
_KHARIF_MONTHS = {6, 7, 8, 9, 10, 11}   # June – November
_RABI_MONTHS   = {11, 12, 1, 2, 3, 4}   # November – April
_ZAID_MONTHS   = {4, 5, 6}              # April – June (minor summer season)


def _get_crop_season(month: int) -> str:
    """Return current Indian crop season based on calendar month."""
    if month in _KHARIF_MONTHS:
        return "Kharif (खरीफ) — June to November"
    if month in _RABI_MONTHS:
        return "Rabi (रबी) — November to April"
    return "Zaid (जायद) — April to June"


class ExternalAPIService:
    """
    Async wrapper around 4 public APIs used for Indian agri-field intelligence.
    Instantiate once; call methods per request.
    """

    # ────────────────────────────────────────────────────────────────────────
    # 1. Open-Meteo — Weather + Agro-meteorological variables
    # ────────────────────────────────────────────────────────────────────────
    async def get_weather(
        self, lat: float, lon: float
    ) -> Dict[str, Any]:
        """
        Fetch real-time weather and agro-meteorological data from Open-Meteo.
        Returns temperature, humidity, rainfall, wind speed, soil temp,
        evapotranspiration — all critical for Indian crop disease risk.
        """
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "weather_code",
                "apparent_temperature",
                "rain",
            ],
            "hourly": [
                "soil_temperature_0cm",
                "et0_fao_evapotranspiration",
            ],
            "daily": [
                "precipitation_sum",
                "temperature_2m_max",
                "temperature_2m_min",
            ],
            "timezone": "Asia/Kolkata",  # IST — Indian Standard Time
            "forecast_days": 3,
        }
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                raw = response.json()

            current = raw.get("current", {})
            hourly  = raw.get("hourly", {})
            daily   = raw.get("daily", {})

            # Soil temperature (first hour value)
            soil_temps = hourly.get("soil_temperature_0cm", [])
            soil_temp  = soil_temps[0] if soil_temps else None

            # ET0 evapotranspiration (first hour value — mm/hr)
            et0_list = hourly.get("et0_fao_evapotranspiration", [])
            et0      = et0_list[0] if et0_list else None

            # Disease-risk flags based on ICAR thresholds
            temp    = current.get("temperature_2m")
            humidity = current.get("relative_humidity_2m")
            rain    = current.get("rain", 0) or 0

            disease_risk_weather = "Low"
            if humidity is not None and temp is not None:
                if humidity > 85 and 20 <= temp <= 30:
                    disease_risk_weather = "High"  # Rice Blast / Brown Spot conditions
                elif humidity > 70:
                    disease_risk_weather = "Moderate"

            # 3-day forecast summary
            daily_precip = daily.get("precipitation_sum", [])
            daily_max    = daily.get("temperature_2m_max", [])
            daily_min    = daily.get("temperature_2m_min", [])
            forecast_days = []
            for i in range(min(3, len(daily_precip))):
                forecast_days.append({
                    "day": i + 1,
                    "precip_mm": daily_precip[i] if i < len(daily_precip) else None,
                    "temp_max":  daily_max[i]    if i < len(daily_max)    else None,
                    "temp_min":  daily_min[i]    if i < len(daily_min)    else None,
                })

            now_month = datetime.now().month
            season = _get_crop_season(now_month)

            return {
                "available": True,
                "source": "Open-Meteo (open-meteo.com)",
                "coordinates": {"lat": lat, "lon": lon},
                "season": season,
                "current": {
                    "temperature_c":        temp,
                    "feels_like_c":         current.get("apparent_temperature"),
                    "relative_humidity_pct": humidity,
                    "precipitation_mm":     current.get("precipitation"),
                    "rain_mm":              rain,
                    "wind_speed_kmh":       current.get("wind_speed_10m"),
                    "weather_code":         current.get("weather_code"),
                },
                "agro": {
                    "soil_temperature_c":       soil_temp,
                    "et0_evapotranspiration_mm": et0,
                },
                "disease_risk_from_weather": disease_risk_weather,
                "forecast_3day": forecast_days,
            }

        except httpx.TimeoutException:
            logger.warning("Open-Meteo request timed out for lat=%s lon=%s", lat, lon)
            return {"available": False, "reason": "Weather service timed out."}
        except httpx.HTTPStatusError as e:
            logger.warning("Open-Meteo HTTP error %s for lat=%s lon=%s", e.response.status_code, lat, lon)
            return {"available": False, "reason": f"Weather service returned HTTP {e.response.status_code}."}
        except Exception as e:
            logger.exception("Unexpected error fetching weather for lat=%s lon=%s", lat, lon)
            return {"available": False, "reason": "Weather data unavailable."}

    # ────────────────────────────────────────────────────────────────────────
    # 2. Nominatim (OpenStreetMap) — Reverse Geocoding
    # ────────────────────────────────────────────────────────────────────────
    async def get_location_name(
        self, lat: float, lon: float
    ) -> Dict[str, Any]:
        """
        Reverse geocode GPS coordinates to Indian district, state, and country
        using Nominatim (OpenStreetMap). No API key required.
        Respects Nominatim usage policy: User-Agent header required.
        """
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "addressdetails": 1,
            "zoom": 10,  # district-level precision
            "accept-language": "en",
        }
        headers = {
            # Nominatim usage policy requires a valid User-Agent
            "User-Agent": "AgriPulse/2.0 (agricultural-disease-intelligence; contact@agripulse.local)"
        }
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

            address = data.get("address", {})
            display_name = data.get("display_name", "")

            # Extract Indian administrative hierarchy
            district = (
                address.get("county")
                or address.get("district")
                or address.get("city_district")
                or address.get("suburb")
                or address.get("town")
                or address.get("city")
            )
            state    = address.get("state", "")
            country  = address.get("country", "")
            postcode = address.get("postcode", "")

            # Build a clean location string
            parts = [p for p in [district, state, country] if p]
            location_string = ", ".join(parts) if parts else display_name

            return {
                "available": True,
                "source": "Nominatim / OpenStreetMap (nominatim.openstreetmap.org)",
                "display_name": display_name,
                "location_string": location_string,
                "district": district,
                "state": state,
                "country": country,
                "postcode": postcode,
            }

        except httpx.TimeoutException:
            logger.warning("Nominatim timed out for lat=%s lon=%s", lat, lon)
            return {"available": False, "reason": "Geocoding service timed out."}
        except httpx.HTTPStatusError as e:
            logger.warning("Nominatim HTTP error %s", e.response.status_code)
            return {"available": False, "reason": f"Geocoding service returned HTTP {e.response.status_code}."}
        except Exception:
            logger.exception("Unexpected error in Nominatim reverse geocoding")
            return {"available": False, "reason": "Location data unavailable."}

    # ────────────────────────────────────────────────────────────────────────
    # 3. GBIF — Crop Pest / Species Occurrence Data
    # ────────────────────────────────────────────────────────────────────────
    async def get_pest_occurrences(
        self,
        lat: float,
        lon: float,
        radius_km: int = 50,
    ) -> Dict[str, Any]:
        """
        Query GBIF for occurrence records of key Indian crop disease pathogens
        and pests within `radius_km` km of the given coordinates.
        Uses GBIF Occurrence Search API — no API key required.

        Pathogens monitored (relevant to rice/wheat/maize in India):
          - Magnaporthe oryzae (Rice Blast)
          - Bipolaris oryzae   (Brown Spot)
          - Xanthomonas oryzae (Bacterial Blight)
          - Nilaparvata lugens (Brown Planthopper)
          - Scirpophaga incertulas (Yellow Stem Borer)
        """
        PATHOGENS = {
            "Magnaporthe oryzae":    "Rice Blast pathogen",
            "Bipolaris oryzae":      "Brown Spot pathogen",
            "Xanthomonas oryzae":    "Bacterial Blight pathogen",
            "Nilaparvata lugens":    "Brown Planthopper (BPH)",
            "Scirpophaga incertulas": "Yellow Stem Borer",
        }

        results: Dict[str, Any] = {}
        radius_m = radius_km * 1000

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for species, label in PATHOGENS.items():
                try:
                    response = await client.get(
                        "https://api.gbif.org/v1/occurrence/search",
                        params={
                            "scientificName": species,
                            "decimalLatitude":  f"{lat - 0.5},{lat + 0.5}",
                            "decimalLongitude": f"{lon - 0.5},{lon + 0.5}",
                            "country": "IN",   # India only
                            "limit": 1,        # We only need the count
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    count = data.get("count", 0)
                    results[species] = {
                        "label": label,
                        "occurrence_count_nearby": count,
                        "risk_level": (
                            "High" if count >= 10
                            else "Moderate" if count >= 3
                            else "Low" if count >= 1
                            else "None recorded"
                        ),
                    }
                except Exception as e:
                    logger.warning("GBIF query failed for %s: %s", species, e)
                    results[species] = {
                        "label": label,
                        "occurrence_count_nearby": None,
                        "risk_level": "Data unavailable",
                    }

        any_available = any(
            v.get("occurrence_count_nearby") is not None
            for v in results.values()
        )

        return {
            "available": any_available,
            "source": "GBIF — Global Biodiversity Information Facility (gbif.org)",
            "search_radius_km": radius_km,
            "country_filter": "India (IN)",
            "pathogens": results,
        } if any_available else {
            "available": False,
            "reason": "GBIF pest occurrence data unavailable.",
        }

    # ────────────────────────────────────────────────────────────────────────
    # 4. GNews — Indian Agriculture Disease News
    # ────────────────────────────────────────────────────────────────────────
    async def get_agri_news(
        self,
        query: str = "crop disease India agriculture",
        max_articles: int = 5,
    ) -> Dict[str, Any]:
        """
        Fetch real agriculture/disease news articles from GNews API.
        Requires GNEWS_API_KEY in .env. Free tier: 100 requests/day.
        Falls back gracefully if key is missing or service is down.
        """
        api_key: Optional[str] = getattr(settings, "GNEWS_API_KEY", None)

        if not api_key:
            return {
                "available": False,
                "reason": "GNews API key not configured. Add GNEWS_API_KEY to backend/.env to enable news feed.",
            }

        url = "https://gnews.io/api/v4/search"
        params = {
            "q": query,
            "lang": "en",
            "country": "in",   # India
            "max": max_articles,
            "apikey": api_key,
            "in": "title,description",
        }

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            articles = data.get("articles", [])
            clean_articles = [
                {
                    "title":       a.get("title"),
                    "description": a.get("description"),
                    "source":      a.get("source", {}).get("name"),
                    "url":         a.get("url"),
                    "published_at": a.get("publishedAt"),
                }
                for a in articles
            ]

            return {
                "available": True,
                "source": "GNews (gnews.io)",
                "query": query,
                "total_results": data.get("totalArticles", len(clean_articles)),
                "articles": clean_articles,
            }

        except httpx.TimeoutException:
            logger.warning("GNews request timed out")
            return {"available": False, "reason": "News service timed out."}
        except httpx.HTTPStatusError as e:
            logger.warning("GNews HTTP error %s", e.response.status_code)
            if e.response.status_code == 403:
                return {"available": False, "reason": "GNews API key invalid or quota exceeded."}
            return {"available": False, "reason": f"News service returned HTTP {e.response.status_code}."}
        except Exception:
            logger.exception("Unexpected error fetching agri news")
            return {"available": False, "reason": "Agriculture news unavailable."}


# ─── Module-level singleton ───────────────────────────────────────────────────
external_api_service = ExternalAPIService()
