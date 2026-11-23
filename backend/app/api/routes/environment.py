"""
AgriPulse 2.0 — Environment & External Data API Routes
Exposes real weather, geolocation, pest occurrence, and agri news
data for Indian agricultural field intelligence.

Endpoints:
  GET /api/v1/environment          — weather + location + pest data
  GET /api/v1/environment/news     — Indian agriculture disease news
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from ...core.limiter import limiter
from ...core.logger import logger
from ...services.external_apis import external_api_service

router = APIRouter()

# Default Indian location fallback: Shivamogga, Karnataka (PESITM / Malnad Agricultural Region)
DEFAULT_LAT = 13.9299
DEFAULT_LON = 75.5681


@router.get("/environment")
@limiter.limit("30/minute")
async def get_environment_data(
    request: Request,
    lat: Optional[float] = Query(
        default=None,
        description="Field latitude. Falls back to Shivamogga, Karnataka if not provided.",
        ge=-90.0,
        le=90.0,
    ),
    lon: Optional[float] = Query(
        default=None,
        description="Field longitude. Falls back to Shivamogga, Karnataka if not provided.",
        ge=-180.0,
        le=180.0,
    ),
    radius_km: int = Query(
        default=50,
        description="Search radius in km for GBIF pest occurrence data.",
        ge=1,
        le=500,
    ),
):
    """
    Fetch real-time environmental intelligence for an Indian agricultural field.

    Returns:
    - **weather**: Temperature, humidity, rainfall, wind speed, soil temp,
      evapotranspiration, 3-day forecast, disease risk level from weather
      (Open-Meteo — IST timezone, no API key required)
    - **location**: Indian district, state, country name via reverse geocoding
      (Nominatim / OpenStreetMap — no API key required)
    - **pests**: GBIF occurrence counts for key Indian crop pathogens
      (Magnaporthe oryzae, Bipolaris oryzae, Xanthomonas oryzae, BPH, Stem Borer)
    - **using_default_location**: True if lat/lon were not provided

    All fields degrade gracefully — if any source is unavailable, the
    corresponding section includes `"available": false` with a reason string.
    No fabricated values are ever returned.
    """
    using_default = lat is None or lon is None
    effective_lat = lat if lat is not None else DEFAULT_LAT
    effective_lon = lon if lon is not None else DEFAULT_LON

    request_id = getattr(request.state, "request_id", "N/A")
    logger.info(
        "[%s] Environment data request — lat=%.4f lon=%.4f radius=%dkm default=%s",
        request_id,
        effective_lat,
        effective_lon,
        radius_km,
        using_default,
    )

    # Run all three external calls concurrently
    import asyncio

    weather_task  = external_api_service.get_weather(effective_lat, effective_lon)
    location_task = external_api_service.get_location_name(effective_lat, effective_lon)
    pests_task    = external_api_service.get_pest_occurrences(
        effective_lat, effective_lon, radius_km
    )

    weather_data, location_data, pest_data = await asyncio.gather(
        weather_task,
        location_task,
        pests_task,
        return_exceptions=False,
    )

    return {
        "using_default_location": using_default,
        "default_location_note": (
            "No coordinates provided. Showing data for Shivamogga, Karnataka "
            "(PESITM / Malnad Agricultural Region). Allow browser location for field-specific data."
            if using_default else None
        ),
        "coordinates": {
            "lat": effective_lat,
            "lon": effective_lon,
        },
        "weather": weather_data,
        "location": location_data,
        "pest_occurrences": pest_data,
    }


@router.get("/environment/news")
@limiter.limit("20/minute")
async def get_agri_news(
    request: Request,
    query: str = Query(
        default="crop disease India agriculture",
        description="Search query for Indian agriculture disease news.",
        max_length=200,
    ),
    max_articles: int = Query(
        default=5,
        description="Maximum number of news articles to return.",
        ge=1,
        le=10,
    ),
):
    """
    Fetch latest Indian agriculture disease news from GNews.

    Requires `GNEWS_API_KEY` set in `backend/.env`.
    Free tier: 100 requests/day. Get a free key at https://gnews.io

    Returns gracefully with `available: false` if the key is missing
    or the service is down — never returns fabricated news.
    """
    request_id = getattr(request.state, "request_id", "N/A")
    logger.info(
        "[%s] Agri news request — query='%s' max=%d",
        request_id,
        query,
        max_articles,
    )

    news_data = await external_api_service.get_agri_news(
        query=query,
        max_articles=max_articles,
    )

    return news_data
