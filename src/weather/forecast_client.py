"""
NOAA Weather Forecast Client.

Phase 1: Point forecast from api.weather.gov + Gaussian fallback for bucket probabilities.
Phase 2 (future): NBM probabilistic data for calibrated bucket distributions.

Key design decisions:
  - api.weather.gov is free, no API key, requires User-Agent header
  - Cache forecasts for 1 hour (NOAA updates every 1-6 hours)
  - Gaussian fallback: point forecast +/- historical forecast error → bucket probabilities
  - Historical std-dev varies by city, season, and lead time (hardcoded Phase 1 estimates)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ── City grid-point mappings for api.weather.gov ──────────────────────────
# Format: (Weather Forecast Office, gridX, gridY, typical forecast std-dev in °F)
# Std-dev estimates are conservative 1-day forecast errors (Phase 1 placeholders).
# Phase 2: replace with per-city/season/lead-time calibrated values.
CITY_GRIDS: dict[str, dict] = {
    "new-york": {"wfo": "OKX", "x": 37, "y": 39, "std_f": 3.5},
    "chicago": {"wfo": "LOT", "x": 66, "y": 77, "std_f": 4.0},
    "miami": {"wfo": "MFL", "x": 75, "y": 67, "std_f": 2.5},
    "los-angeles": {"wfo": "LOX", "x": 155, "y": 44, "std_f": 3.0},
    "houston": {"wfo": "HGX", "x": 65, "y": 97, "std_f": 3.5},
    "phoenix": {"wfo": "PSR", "x": 161, "y": 58, "std_f": 3.0},
    "denver": {"wfo": "BOU", "x": 62, "y": 60, "std_f": 5.0},
    "atlanta": {"wfo": "FFC", "x": 52, "y": 88, "std_f": 3.5},
    "dallas": {"wfo": "FWD", "x": 80, "y": 103, "std_f": 3.5},
    "san-francisco": {"wfo": "MTR", "x": 85, "y": 105, "std_f": 3.0},
    "seattle": {"wfo": "SEW", "x": 125, "y": 67, "std_f": 3.5},
}


@dataclass
class WeatherForecast:
    """NOAA forecast for a city on a specific date."""
    city: str
    date: str  # YYYY-MM-DD
    high_temp_f: float  # predicted daily high (°F)
    std_dev_f: float  # forecast uncertainty (°F)
    bucket_probabilities: dict[str, float]  # bucket_label -> probability
    source: str  # "gaussian_fallback" or "nbm" (Phase 2)
    fetched_at: float = field(default_factory=time.time)


class WeatherForecastClient:
    """Fetches NOAA forecasts and converts to bucket probability distributions."""

    USER_AGENT = "(polymarket-weather-bot, contact@example.com)"
    BASE_URL = "https://api.weather.gov"

    def __init__(self, cache_ttl_seconds: float = 3600.0):
        self._cache: dict[str, WeatherForecast] = {}  # "city|date" -> forecast
        self._cache_ttl = cache_ttl_seconds
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": self.USER_AGENT, "Accept": "application/geo+json"},
            )
        return self._session

    async def get_forecast(
        self,
        city: str,
        date: str,
        bucket_edges: list[float],
    ) -> Optional[WeatherForecast]:
        """
        Get forecast for a city on a date, with bucket probabilities.

        Args:
            city: City slug (e.g. "new-york")
            date: Target date "YYYY-MM-DD"
            bucket_edges: Sorted temperature thresholds from the market
                          (e.g. [60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80])

        Returns:
            WeatherForecast with bucket_probabilities keyed by label,
            or None if the city is unknown or the forecast is unavailable.
        """
        cache_key = f"{city}|{date}"
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached.fetched_at) < self._cache_ttl:
            return cached

        grid = CITY_GRIDS.get(city)
        if grid is None:
            logger.warning("Unknown city %s — no grid mapping", city)
            return None

        high_temp = await self._fetch_daily_high(grid, date)
        if high_temp is None:
            logger.warning("Could not fetch forecast for %s on %s", city, date)
            return None

        # Phase 1: Gaussian fallback — convert point forecast to bucket probabilities
        std_dev = grid["std_f"]
        bucket_probs = self._gaussian_bucket_probs(high_temp, std_dev, bucket_edges)

        forecast = WeatherForecast(
            city=city,
            date=date,
            high_temp_f=high_temp,
            std_dev_f=std_dev,
            bucket_probabilities=bucket_probs,
            source="gaussian_fallback",
        )
        self._cache[cache_key] = forecast

        logger.info(
            "Forecast for %s on %s: high=%.1f°F std=%.1f°F (%d buckets, source=%s)",
            city, date, high_temp, std_dev, len(bucket_probs), forecast.source,
        )
        return forecast

    async def _fetch_daily_high(self, grid: dict, target_date: str) -> Optional[float]:
        """
        Fetch the daily high temperature from NOAA hourly forecast.

        Scans hourly periods on the target date and returns the max temperature.
        """
        session = await self._ensure_session()
        url = f"{self.BASE_URL}/gridpoints/{grid['wfo']}/{grid['x']},{grid['y']}/forecast/hourly"

        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.error("NOAA API returned %d for %s", resp.status, url)
                    return None
                data = await resp.json()
        except Exception:
            logger.exception("NOAA API request failed for %s", url)
            return None

        periods = data.get("properties", {}).get("periods", [])
        if not periods:
            logger.warning("No forecast periods returned for %s", url)
            return None

        # Find max temperature on the target date
        max_temp: Optional[float] = None
        for period in periods:
            start = period.get("startTime", "")
            if not start.startswith(target_date):
                continue
            temp = period.get("temperature")
            temp_unit = period.get("temperatureUnit", "F")
            if temp is None:
                continue
            temp_f = float(temp)
            if temp_unit == "C":
                temp_f = temp_f * 9.0 / 5.0 + 32.0
            if max_temp is None or temp_f > max_temp:
                max_temp = temp_f

        return max_temp

    @staticmethod
    def _gaussian_bucket_probs(
        mean_f: float,
        std_f: float,
        bucket_edges: list[float],
    ) -> dict[str, float]:
        """
        Convert a point forecast + std-dev into bucket probabilities.

        Bucket structure matches Polymarket weather markets:
        - Buckets defined by edges: [60, 62, 64, ..., 80]
        - First bucket: "below {edges[0]}°F"   → P(T < edges[0])
        - Middle buckets: "{edges[i]}–{edges[i+1]}°F" → P(edges[i] <= T < edges[i+1])
        - Last bucket: "{edges[-1]}°F or higher" → P(T >= edges[-1])

        Returns dict keyed by human-readable label.
        """
        if not bucket_edges:
            return {}

        dist = norm(loc=mean_f, scale=std_f)
        probs: dict[str, float] = {}

        # First bucket: below lowest edge
        label_low = f"below {bucket_edges[0]:.0f}"
        probs[label_low] = float(dist.cdf(bucket_edges[0]))

        # Middle buckets
        for i in range(len(bucket_edges) - 1):
            lo = bucket_edges[i]
            hi = bucket_edges[i + 1]
            label = f"{lo:.0f}-{hi:.0f}"
            probs[label] = float(dist.cdf(hi) - dist.cdf(lo))

        # Last bucket: at or above highest edge
        label_high = f"{bucket_edges[-1]:.0f} or higher"
        probs[label_high] = float(1.0 - dist.cdf(bucket_edges[-1]))

        return probs

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
