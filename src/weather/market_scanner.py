"""
Weather Market Scanner — discovers and parses Polymarket weather markets.

Polymarket weather market structure:
  - Event slug pattern: "highest-temperature-in-{city}-on-{month}-{day}-{year}"
  - Each event has ~11 sub-markets (condition IDs) for temperature buckets
  - Buckets are 2°F wide, e.g., "72°F or higher", "70°F to 72°F"
  - negRisk structure: each sub-market is a YES/NO pair
  - Resolution: via Weather Underground airport station data

This scanner wraps the main MarketScanner and filters/structures weather markets.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.market.models import Market

logger = logging.getLogger(__name__)

# Regex to parse weather event slugs
# e.g. "highest-temperature-in-new-york-on-april-4-2026"
WEATHER_SLUG_PATTERN = re.compile(
    r"highest-temperature-in-(?P<city>.+)-on-(?P<month>\w+)-(?P<day>\d+)-(?P<year>\d{4})"
)

# Month name → number
MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

# Regex to extract temperature thresholds from market questions
# Matches patterns like:
#   "72°F or higher" → (72, None) — the top bucket
#   "70°F to 72°F"   → (70, 72)  — a middle bucket
#   "below 60°F"     → (None, 60) — the bottom bucket
TEMP_HIGHER_PATTERN = re.compile(r"(\d+)\s*°?\s*F?\s*or\s+higher", re.IGNORECASE)
TEMP_RANGE_PATTERN = re.compile(r"(\d+)\s*°?\s*F?\s*to\s+(\d+)\s*°?\s*F?", re.IGNORECASE)
TEMP_BELOW_PATTERN = re.compile(r"below\s+(\d+)\s*°?\s*F?", re.IGNORECASE)


@dataclass
class WeatherBucket:
    """One temperature bucket in a weather event."""
    condition_id: str
    token_id_yes: str
    token_id_no: str
    label: str  # e.g. "72°F or higher", "70-72", "below 60"
    lower_bound_f: Optional[float]  # None for "below X" bucket
    upper_bound_f: Optional[float]  # None for "X or higher" bucket
    yes_price: Optional[float]
    neg_risk: bool
    tick_size: float


@dataclass
class WeatherEvent:
    """A complete weather event (one city, one date, multiple buckets)."""
    event_slug: str
    city: str  # slug format, e.g. "new-york"
    date: str  # "YYYY-MM-DD"
    buckets: list[WeatherBucket] = field(default_factory=list)

    @property
    def bucket_edges(self) -> list[float]:
        """
        Extract sorted temperature edge values from all buckets.

        E.g. if buckets span 60-80°F in 2°F steps:
        Returns [60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80]
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

    Usage:
        weather_scanner = WeatherMarketScanner()
        events = weather_scanner.scan(all_markets)  # list of WeatherEvent
    """

    def scan(self, markets: dict[str, Market]) -> list[WeatherEvent]:
        """
        Identify and group weather markets from the full market list.

        Args:
            markets: condition_id -> Market from the main scanner

        Returns:
            List of WeatherEvent objects, each containing all buckets for that city/date.
        """
        # Group markets by event_slug
        event_groups: dict[str, list[Market]] = {}
        for market in markets.values():
            slug = market.event_slug
            if not slug:
                continue
            match = WEATHER_SLUG_PATTERN.search(slug)
            if match:
                event_groups.setdefault(slug, []).append(market)

        events: list[WeatherEvent] = []
        for slug, group_markets in event_groups.items():
            event = self._parse_event(slug, group_markets)
            if event is not None and len(event.buckets) >= 3:
                events.append(event)

        logger.info("Found %d weather events with %d total buckets",
                     len(events),
                     sum(len(e.buckets) for e in events))
        return events

    def _parse_event(self, slug: str, markets: list[Market]) -> Optional[WeatherEvent]:
        """Parse an event slug and its sub-markets into a WeatherEvent."""
        match = WEATHER_SLUG_PATTERN.search(slug)
        if not match:
            return None

        city = match.group("city")
        month_str = match.group("month").lower()
        day = match.group("day")
        year = match.group("year")

        month_num = MONTH_MAP.get(month_str)
        if month_num is None:
            logger.debug("Unknown month '%s' in slug %s", month_str, slug)
            return None

        date = f"{year}-{month_num}-{day.zfill(2)}"

        event = WeatherEvent(event_slug=slug, city=city, date=date)

        for market in markets:
            bucket = self._parse_bucket(market)
            if bucket is not None:
                event.buckets.append(bucket)

        # Sort buckets by lower_bound (None = -inf for "below X")
        event.buckets.sort(key=lambda b: b.lower_bound_f if b.lower_bound_f is not None else -999)

        return event

    @staticmethod
    def _parse_bucket(market: Market) -> Optional[WeatherBucket]:
        """Parse a single sub-market into a WeatherBucket."""
        question = market.question

        # Try to extract temperature bounds from the question text
        lower: Optional[float] = None
        upper: Optional[float] = None
        label = question

        m = TEMP_HIGHER_PATTERN.search(question)
        if m:
            lower = float(m.group(1))
            upper = None
            label = f"{lower:.0f} or higher"
        else:
            m = TEMP_RANGE_PATTERN.search(question)
            if m:
                lower = float(m.group(1))
                upper = float(m.group(2))
                label = f"{lower:.0f}-{upper:.0f}"
            else:
                m = TEMP_BELOW_PATTERN.search(question)
                if m:
                    lower = None
                    upper = float(m.group(1))
                    label = f"below {upper:.0f}"
                else:
                    # Can't parse — skip this bucket
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
