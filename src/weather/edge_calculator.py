"""
Weather Edge Calculator — compares NOAA forecast probabilities to market prices.

Core formula:
    edge = P_noaa(bucket) - ask_price(bucket)

If the market says "72°F or higher" is priced at $0.30 (implied 30%) but
NOAA says there's a 45% chance, that's a +15% edge.

Kelly sizing uses quarter-Kelly (short-dated: <24h resolution).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.weather.forecast_client import WeatherForecast
from src.weather.market_scanner import WeatherBucket, WeatherEvent
from src.feed.order_book import OrderBookState

logger = logging.getLogger(__name__)


@dataclass
class WeatherEdge:
    """Edge calculation result for a single bucket."""
    bucket: WeatherBucket
    p_noaa: float  # NOAA probability for this bucket
    ask_price: float  # best ask on the YES token
    edge: float  # p_noaa - ask_price
    kelly_fraction: float  # quarter-Kelly position size as fraction of bankroll
    position_size_usd: float  # dollar amount to bet
    position_size_shares: float  # shares at ask_price
    bucket_label: str


class WeatherEdgeCalculator:
    """Computes edge across all buckets in a weather event."""

    def __init__(
        self,
        min_edge: float = 0.08,
        kelly_fraction: float = 0.25,  # quarter-Kelly for <24h markets
        max_position_usd: float = 5.0,
        bankroll_usd: float = 50.0,
        max_portfolio_exposure: float = 0.50,
    ):
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.max_position_usd = max_position_usd
        self.bankroll_usd = bankroll_usd
        self.max_portfolio_exposure = max_portfolio_exposure

    def compute_edges(
        self,
        forecast: WeatherForecast,
        event: WeatherEvent,
        books: dict[str, OrderBookState],
        total_deployed_usd: float = 0.0,
    ) -> list[WeatherEdge]:
        """
        Compute edge for every bucket in the event.

        Only returns buckets with edge >= min_edge.
        Sorted by edge descending (best opportunity first).

        Args:
            forecast: NOAA forecast with bucket_probabilities
            event: WeatherEvent with parsed buckets
            books: token_id -> OrderBookState for live ask prices
            total_deployed_usd: current total portfolio deployment
        """
        edges: list[WeatherEdge] = []

        for bucket in event.buckets:
            # Match bucket to forecast probability
            p_noaa = self._match_bucket_prob(bucket, forecast)
            if p_noaa is None:
                continue

            # Get ask price from order book (true fill cost)
            ask_price = self._get_ask_price(bucket, books)
            if ask_price is None or ask_price <= 0 or ask_price >= 0.99:
                continue

            edge = p_noaa - ask_price
            if edge < self.min_edge:
                continue

            # Kelly sizing: f* = edge / (1 - ask_price), then scale by quarter-Kelly
            max_loss_per_share = 1.0 - ask_price
            if max_loss_per_share <= 0:
                continue

            f_star = edge / max_loss_per_share
            f_applied = self.kelly_fraction * f_star

            # Position size in USD
            raw_size_usd = f_applied * self.bankroll_usd
            available = min(
                self.bankroll_usd - total_deployed_usd,
                self.bankroll_usd * self.max_portfolio_exposure - total_deployed_usd,
            )
            position_usd = min(raw_size_usd, self.max_position_usd, max(0, available))

            if position_usd < 0.50:  # minimum viable trade
                continue

            position_shares = position_usd / ask_price

            edges.append(WeatherEdge(
                bucket=bucket,
                p_noaa=p_noaa,
                ask_price=ask_price,
                edge=edge,
                kelly_fraction=f_applied,
                position_size_usd=round(position_usd, 2),
                position_size_shares=round(position_shares, 2),
                bucket_label=bucket.label,
            ))

            logger.info(
                "Weather edge: %s %s P_noaa=%.3f ask=%.3f edge=%.3f size=$%.2f",
                event.city, bucket.label, p_noaa, ask_price, edge, position_usd,
            )

        # Sort by edge descending
        edges.sort(key=lambda e: e.edge, reverse=True)
        return edges

    def _match_bucket_prob(
        self,
        bucket: WeatherBucket,
        forecast: WeatherForecast,
    ) -> Optional[float]:
        """
        Match a market bucket to the corresponding NOAA probability.

        The forecast has bucket_probabilities keyed by labels like:
          "below 60", "60-62", "62-64", ..., "80 or higher"

        Market buckets have labels like:
          "below 59", "30-31", "48 or higher"

        We match based on temperature bounds. The bucket width may differ
        between forecast (generated from bucket_edges) and market (2°F wide),
        so we use the bounds directly.
        """
        # Build the expected label from the bucket bounds
        if bucket.lower_bound_f is not None and bucket.upper_bound_f is None:
            # "X or higher" bucket
            target_label = f"{bucket.lower_bound_f:.0f} or higher"
        elif bucket.lower_bound_f is None and bucket.upper_bound_f is not None:
            # "below X" / "X or below" bucket
            target_label = f"below {bucket.upper_bound_f:.0f}"
        elif bucket.lower_bound_f is not None and bucket.upper_bound_f is not None:
            # Range bucket "X-Y"
            target_label = f"{bucket.lower_bound_f:.0f}-{bucket.upper_bound_f:.0f}"
        else:
            return None

        prob = forecast.bucket_probabilities.get(target_label)
        if prob is not None:
            return prob

        # Try alternate label formats
        # "below 59" might be stored as "below 59" or "59 or below"
        alt_labels = []
        if bucket.lower_bound_f is not None and bucket.upper_bound_f is None:
            alt_labels.append(f"{bucket.lower_bound_f:.0f}orhigher")
        elif bucket.lower_bound_f is None and bucket.upper_bound_f is not None:
            alt_labels.append(f"below{bucket.upper_bound_f:.0f}")
            alt_labels.append(f"{bucket.upper_bound_f:.0f}orbelow")

        for label, p in forecast.bucket_probabilities.items():
            normalized = label.lower().replace(" ", "").replace("°f", "")
            for alt in alt_labels:
                if alt.lower() in normalized or normalized in alt.lower():
                    return p

        # Last resort: fuzzy matching on the original label
        for label, p in forecast.bucket_probabilities.items():
            if target_label.lower() in label.lower() or label.lower() in target_label.lower():
                return p

        logger.debug("No probability match for bucket %s (target: %s)", bucket.label, target_label)
        return None

    @staticmethod
    def _get_ask_price(
        bucket: WeatherBucket,
        books: dict[str, OrderBookState],
    ) -> Optional[float]:
        """Get the best ask for the YES token of this bucket."""
        book = books.get(bucket.token_id_yes)
        if book is None or not book.has_data:
            # Fallback to the Gamma API price if no live book data
            return bucket.yes_price
        return book.best_ask if book.best_ask is not None else bucket.yes_price
