"""
Weather Market Scanner — discovers and parses Polymarket weather markets.

Polymarket weather market structure (from actual API data):
  - event_slug: "{city}-daily-weather" (e.g., "nyc-daily-weather", "chicago-daily-weather")
  - Market slug: "highest-temperature-in-{city}-on-{month}-{day}-{year}-{bucket}"
  - Each event has ~11 sub-markets (condition IDs) for temperature buckets
  - Buckets are 2°F wide, e.g., "48°F or higher", "30-31°F"
  - negRisk structure: each sub-market is a YES/NO pair
  - Resolution: via Weather Underground airport station data
  - Question format: "Will the highest temperature in {City} be {range} on {Month} {Day}?"

Key discovery: market.event_slug is "nyc-daily-weather" (short), while the full
slug with city/date/bucket is on the market.slug field itself.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.market.models import Market

logger = logging.getLogger(__name__)

# ── Detection patterns ────────────────────────────────────────────────────

# Match event_slug like "nyc-daily-weather", "chicago-daily-weather"
DAILY_WEATHER_PATTERN = re.compile(r"^(?P<city>.+)-daily-weather$")

# Match market slug like "highest-temperature-in-chicago-on-april-2-2026-34-35f"
MARKET_SLUG_PATTERN = re.compile(
    r"highest-temperature-in-(?P<city>[a-z-]+)-on-(?P<month>\w+)-(?P<day>\d+)-(?P<year>\d{4})"
)

# Month name → number
MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

# ── Temperature extraction from question text ─────────────────────────────
# Real examples from Polymarket:
#   "Will the highest temperature in Chicago be 48°F or higher on April 2?"
#   "Will the highest temperature in Chicago be between 30-31°F on April 2?"
#   "Will the highest temperature in NYC be 59°F or below on April 2?"

TEMP_OR_HIGHER = re.compile(r"(\d+)\s*°?\s*F?\s*or\s+higher", re.IGNORECASE)
TEMP_OR_BELOW = re.compile(r"(\d+)\s*°?\s*F?\s*or\s+below", re.IGNORECASE)
TEMP_BETWEEN = re.compile(r"between\s+(\d+)\s*[-–]\s*(\d+)\s*°?\s*F?", re.IGNORECASE)
TEMP_RANGE_DASH = re.compile(r"be\s+(\d+)\s*[-–]\s*(\d+)\s*°?\s*F?", re.IGNORECASE)

# City slug normalization: Polymarket uses abbreviated forms
# Maps Polymarket event_slug city part → NOAA city key
CITY_SLUG_MAP: dict[str, str] = {
    "nyc": "new-york",
    "new-york": "new-york",
    "chicago": "chicago",
    "miami": "miami",
    "los-angeles": "los-angeles",
    "la": "los-angeles",
    "houston": "houston",
    "phoenix": "phoenix",
    "denver": "denver",
    "atlanta": "atlanta",
    "dallas": "dallas",
    "san-francisco": "san-francisco",
    "sf": "san-francisco",
    "seattle": "seattle",
    # International cities (Phase 1: skip, no NOAA grid data)
    # "london": None, "tokyo": None, "seoul": None, "shanghai": None, etc.
}


@dataclass
class WeatherBucket:
    """One temperature bucket in a weather event."""
    condition_id: str
    token_id_yes: str
    token_id_no: str
    label: str  # e.g. "48 or higher", "30-31", "below 59"
    lower_bound_f: Optional[float]  # None for "below X" / "X or below" bucket
    upper_bound_f: Optional[float]  # None for "X or higher" bucket
    yes_price: Optional[float]
    neg_risk: bool
    tick_size: float


@dataclass
class WeatherEvent:
    """A complete weather event (one city, one date, multiple buckets)."""
    event_slug: str
    city: str  # NOAA city key, e.g. "new-york", "chicago"
    date: str  # "YYYY-MM-DD"
    buckets: list[WeatherBucket] = field(default_factory=list)

    @property
    def bucket_edges(self) -> list[float]:
        """
        Extract sorted temperature edge values from all buckets.

        E.g. if buckets span 29-48°F in 2°F steps:
        Returns [29, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
        """
        edges: set[float] = set()
        for b in self.buckets:
            if b.lower_bound_f is not None:
                edges.add(b.lower_bound_f)
            if b.upper_bound_f is not None:
                edges.add(b.upper_bound_f)
        return sorted(edges)


class WeatherMarketScanner:
    """
    Filters the main scanner's markets for weather events and structures them.

    Detection strategy (two-pass):
      1. Check market.event_slug for "{city}-daily-weather" pattern
      2. OR check market.slug for "highest-temperature-in-{city}-on-..." pattern
      3. Parse city and date from whichever matched
      4. Group by city+date, parse temperature buckets from question text
    """

    def scan(self, markets: dict[str, Market]) -> list[WeatherEvent]:
        """
        Identify and group weather markets from the full market list.

        Args:
            markets: condition_id -> Market from the main scanner

        Returns:
            List of WeatherEvent objects, each containing all buckets for that city/date.
        """
        # Group markets by (city, date) key
        groups: dict[str, list[Market]] = {}  # "city|date" -> markets

        for market in markets.values():
            city_date = self._detect_weather_market(market)
            if city_date is None:
                continue
            noaa_city, date_str = city_date
            key = f"{noaa_city}|{date_str}"
            groups.setdefault(key, []).append(market)

        events: list[WeatherEvent] = []
        for key, group_markets in groups.items():
            noaa_city, date_str = key.split("|", 1)
            event = self._build_event(noaa_city, date_str, group_markets)
            if event is not None and len(event.buckets) >= 3:
                events.append(event)

        logger.info(
            "Found %d weather events with %d total buckets",
            len(events),
            sum(len(e.buckets) for e in events),
        )
        return events

    def _detect_weather_market(self, market: Market) -> Optional[tuple[str, str]]:
        """
        Detect if a market is a weather market and extract (noaa_city, date).

        Returns (noaa_city, "YYYY-MM-DD") or None.
        """
        # Strategy 1: Check event_slug for "{city}-daily-weather"
        event_slug = market.event_slug or ""
        daily_match = DAILY_WEATHER_PATTERN.match(event_slug)

        # Strategy 2: Check market slug for "highest-temperature-in-..."
        market_slug = market.slug or ""
        slug_match = MARKET_SLUG_PATTERN.search(market_slug)

        if not daily_match and not slug_match:
            return None

        # Extract city
        if daily_match:
            poly_city = daily_match.group("city")
        elif slug_match:
            poly_city = slug_match.group("city")
        else:
            return None

        # Map to NOAA city key
        noaa_city = CITY_SLUG_MAP.get(poly_city)
        if noaa_city is None:
            # Unknown city (probably international) — skip for Phase 1
            return None

        # Extract date from market slug (more reliable than event_slug)
        if slug_match:
            month_str = slug_match.group("month").lower()
            day = slug_match.group("day")
            year = slug_match.group("year")
            month_num = MONTH_MAP.get(month_str)
            if month_num is None:
                return None
            date_str = f"{year}-{month_num}-{day.zfill(2)}"
            return (noaa_city, date_str)

        # If we only matched event_slug, try to parse date from the question
        # "Will the highest temperature in Chicago be between 30-31°F on April 2?"
        date_from_q = self._parse_date_from_question(market.question)
        if date_from_q:
            return (noaa_city, date_from_q)

        return None

    @staticmethod
    def _parse_date_from_question(question: str) -> Optional[str]:
        """Extract date from question like '...on April 2?'"""
        m = re.search(
            r"on\s+(?P<month>\w+)\s+(?P<day>\d+)",
            question,
            re.IGNORECASE,
        )
        if not m:
            return None
        month_str = m.group("month").lower()
        day = m.group("day")
        month_num = MONTH_MAP.get(month_str)
        if month_num is None:
            return None
        # Assume current year (weather markets are daily, always near-term)
        from datetime import datetime, timezone
        year = datetime.now(tz=timezone.utc).year
        return f"{year}-{month_num}-{day.zfill(2)}"

    def _build_event(
        self,
        noaa_city: str,
        date_str: str,
        markets: list[Market],
    ) -> Optional[WeatherEvent]:
        """Build a WeatherEvent from grouped markets."""
        event = WeatherEvent(
            event_slug=f"{noaa_city}-daily-weather",
            city=noaa_city,
            date=date_str,
        )

        for market in markets:
            bucket = self._parse_bucket(market)
            if bucket is not None:
                event.buckets.append(bucket)

        # Sort buckets by lower_bound (None = -inf for "below X")
        event.buckets.sort(
            key=lambda b: b.lower_bound_f if b.lower_bound_f is not None else -999
        )

        return event

    @staticmethod
    def _parse_bucket(market: Market) -> Optional[WeatherBucket]:
        """Parse a single sub-market into a WeatherBucket."""
        question = market.question

        lower: Optional[float] = None
        upper: Optional[float] = None
        label = question

        # Try "X or higher" first
        m = TEMP_OR_HIGHER.search(question)
        if m:
            lower = float(m.group(1))
            upper = None
            label = f"{lower:.0f} or higher"
        else:
            # Try "X or below"
            m = TEMP_OR_BELOW.search(question)
            if m:
                lower = None
                upper = float(m.group(1))
                label = f"below {upper:.0f}"
            else:
                # Try "between X-Y" or "be X-Y"
                m = TEMP_BETWEEN.search(question) or TEMP_RANGE_DASH.search(question)
                if m:
                    lower = float(m.group(1))
                    upper = float(m.group(2))
                    label = f"{lower:.0f}-{upper:.0f}"
                else:
                    logger.debug("Could not parse temperature from: %s", question)
                    return None

        # Get YES and NO token IDs
        yes_token = None
        no_token = None
        yes_price = None
        for token in market.tokens:
            if token.outcome.lower() == "yes":
                yes_token = token.token_id
                yes_price = token.price
            elif token.outcome.lower() == "no":
                no_token = token.token_id

        if yes_token is None or no_token is None:
            logger.debug("Missing YES/NO tokens for %s", question)
            return None

        return WeatherBucket(
            condition_id=market.condition_id,
            token_id_yes=yes_token,
            token_id_no=no_token,
            label=label,
            lower_bound_f=lower,
            upper_bound_f=upper,
            yes_price=yes_price,
            neg_risk=market.neg_risk,
            tick_size=market.tick_size,
        )
