"""
Weather Trading Strategy — the main entry point for weather market trading.

This strategy runs as a parallel path alongside the Bayesian pipeline:
  1. Discover weather markets via WeatherMarketScanner
  2. Fetch NOAA forecasts via WeatherForecastClient
  3. Compute edges via WeatherEdgeCalculator
  4. Place trades via the existing OrderManager

Key decisions:
  - Max 1 position per city per day (pick highest-edge bucket)
  - Skip markets <6h to resolution (forecast too stale)
  - Skip markets >3 days out (forecast uncertainty too high)
  - Global weather exposure cap: 40% of bankroll
  - Uses existing exit infrastructure (stop-loss, emergency floor, time backstop)
  - Logs every forecast + outcome for future calibration (Phase 2)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.config import BotConfig
from src.execution.order_manager import OrderManager, TradeRequest
from src.execution.position_tracker import PositionTracker
from src.feed.order_book import OrderBookState
from src.market.scanner import MarketScanner
from src.persistence.state_store import StateStore
from src.weather.edge_calculator import WeatherEdge, WeatherEdgeCalculator
from src.weather.forecast_client import WeatherForecastClient
from src.weather.market_scanner import WeatherEvent, WeatherMarketScanner

logger = logging.getLogger(__name__)


@dataclass
class WeatherTradeResult:
    """Record of a weather trade attempt."""
    city: str
    date: str
    bucket_label: str
    edge: float
    p_noaa: float
    ask_price: float
    position_usd: float
    order_status: str
    forecast_source: str
    timestamp: float = field(default_factory=time.time)


class WeatherStrategy:
    """
    Evaluates and executes weather market trades.

    Called periodically from the main bot loop (every 5 minutes).
    """

    def __init__(
        self,
        config: BotConfig,
        forecast_client: WeatherForecastClient,
        order_mgr: OrderManager,
        positions: PositionTracker,
        scanner: MarketScanner,
        state_store: StateStore,
        books: dict[str, OrderBookState],
    ):
        self.config = config
        self.forecast_client = forecast_client
        self.order_mgr = order_mgr
        self.positions = positions
        self.scanner = scanner
        self.state_store = state_store
        self._books = books

        wcfg = getattr(config, "weather", None)

        self.enabled = getattr(wcfg, "enabled", True) if wcfg else True
        self.min_edge = getattr(wcfg, "min_edge", 0.08) if wcfg else 0.08
        self.max_positions_per_city = getattr(wcfg, "max_positions_per_city", 1) if wcfg else 1
        self.min_hours_to_resolution = getattr(wcfg, "min_hours_to_resolution", 6.0) if wcfg else 6.0
        self.max_days_to_resolution = getattr(wcfg, "max_days_to_resolution", 3) if wcfg else 3
        self.max_weather_exposure_pct = getattr(wcfg, "max_weather_exposure_pct", 0.40) if wcfg else 0.40
        self.cities = getattr(wcfg, "cities", list(self._default_cities())) if wcfg else list(self._default_cities())

        self.weather_scanner = WeatherMarketScanner()
        self.edge_calculator = WeatherEdgeCalculator(
            min_edge=self.min_edge,
            kelly_fraction=0.25,  # quarter-Kelly for <24h resolution
            max_position_usd=config.kelly.max_position_usd,
            bankroll_usd=config.kelly.total_bankroll_usd,
            max_portfolio_exposure=config.kelly.max_portfolio_exposure,
        )

        # Track which city/date combos we've already traded (prevent duplicates)
        self._traded_today: set[str] = set()  # "city|date"

        # Forecast log for future calibration (Phase 2)
        self._forecast_log: list[dict] = []

    @staticmethod
    def _default_cities() -> list[str]:
        from src.weather.forecast_client import CITY_GRIDS
        return list(CITY_GRIDS.keys())

    async def evaluate_and_trade(self) -> tuple[list[WeatherTradeResult], list[str]]:
        """
        Main entry point — scan for weather markets, compute edges, place trades.

        Returns (trade_results, new_token_ids) where new_token_ids are tokens
        that need WebSocket subscription for exit monitoring.
        """
        if not self.enabled:
            return [], []

        results: list[WeatherTradeResult] = []
        new_token_ids: list[str] = []

        # 1. Discover weather markets from the scanner's market list
        all_markets = self.scanner.markets
        events = self.weather_scanner.scan(all_markets)

        if not events:
            logger.debug("No weather markets found")
            return [], []

        logger.info("Weather strategy: found %d events", len(events))

        # 2. Filter events by time horizon and city
        viable_events = self._filter_events(events)

        # 3. Check global weather exposure cap
        total_deployed = self.positions.get_total_deployed()
        max_weather_usd = self.max_weather_exposure_pct * self.config.kelly.total_bankroll_usd
        weather_deployed = self._get_weather_deployed()

        if weather_deployed >= max_weather_usd:
            logger.info(
                "Weather exposure cap: $%.2f deployed (max $%.2f) — skipping all",
                weather_deployed, max_weather_usd,
            )
            return [], []

        # 4. For each viable event, fetch forecast and compute edges
        for event in viable_events:
            city_date_key = f"{event.city}|{event.date}"

            # Skip if we already have a position for this city/date
            if city_date_key in self._traded_today:
                continue

            # Check per-city position limit
            city_positions = self._count_city_positions(event.city)
            if city_positions >= self.max_positions_per_city:
                continue

            # Fetch NOAA forecast
            bucket_edges = event.bucket_edges
            if not bucket_edges:
                continue

            forecast = await self.forecast_client.get_forecast(
                event.city, event.date, bucket_edges,
            )
            if forecast is None:
                continue

            # Log forecast for future calibration
            self._log_forecast(forecast, event)

            # Compute edges
            edges = self.edge_calculator.compute_edges(
                forecast, event, self._books, total_deployed,
            )

            if not edges:
                continue

            # Pick the single best-edge bucket (Phase 1: one position per city/date)
            best_edge = edges[0]

            # Re-check weather exposure cap
            if weather_deployed + best_edge.position_size_usd > max_weather_usd:
                remaining = max_weather_usd - weather_deployed
                if remaining < 0.50:
                    continue
                best_edge = WeatherEdge(
                    bucket=best_edge.bucket,
                    p_noaa=best_edge.p_noaa,
                    ask_price=best_edge.ask_price,
                    edge=best_edge.edge,
                    kelly_fraction=best_edge.kelly_fraction,
                    position_size_usd=round(remaining, 2),
                    position_size_shares=round(remaining / best_edge.ask_price, 2),
                    bucket_label=best_edge.bucket_label,
                )

            # Dedup: skip if we already hold a position on this token
            token_id = best_edge.bucket.token_id_yes
            if self.positions.get_position(token_id) is not None:
                logger.info(
                    "Weather dedup: already holding position on %s %s %s — skipping",
                    event.city, event.date, best_edge.bucket_label,
                )
                continue

            # Place the trade
            result = await self._execute_weather_trade(best_edge, event, forecast)
            if result is not None:
                results.append(result)
                self._traded_today.add(city_date_key)
                weather_deployed += result.position_usd
                total_deployed += result.position_usd
                token_id = best_edge.bucket.token_id_yes
                if token_id not in new_token_ids:
                    new_token_ids.append(token_id)

        return results, new_token_ids

    def _filter_events(self, events: list[WeatherEvent]) -> list[WeatherEvent]:
        """Filter events by time horizon and enabled cities."""
        now = datetime.now(tz=timezone.utc)
        viable: list[WeatherEvent] = []

        for event in events:
            # Check city is enabled
            if event.city not in self.cities:
                continue

            # Parse event date and check time horizon
            try:
                event_date = datetime.strptime(event.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                # Assume resolution at end of day (23:59 UTC)
                resolution_time = event_date.replace(hour=23, minute=59)
                hours_to_resolution = (resolution_time - now).total_seconds() / 3600
            except ValueError:
                continue

            if hours_to_resolution < self.min_hours_to_resolution:
                logger.debug("Skipping %s %s — too close to resolution (%.1fh)",
                             event.city, event.date, hours_to_resolution)
                continue

            if hours_to_resolution > self.max_days_to_resolution * 24:
                logger.debug("Skipping %s %s — too far out (%.1fh)",
                             event.city, event.date, hours_to_resolution)
                continue

            viable.append(event)

        logger.info("Weather strategy: %d viable events after filtering", len(viable))
        return viable

    async def _execute_weather_trade(
        self,
        edge: WeatherEdge,
        event: WeatherEvent,
        forecast: WeatherForecast,
    ) -> Optional[WeatherTradeResult]:
        """Place a trade for a weather bucket."""
        bucket = edge.bucket

        # Bump price by 2 ticks above observed ask to cross the spread
        # and fill immediately in thin weather markets.
        tick = bucket.tick_size or 0.01
        fill_price = round(edge.ask_price + 2 * tick, 10)
        fill_price = min(fill_price, 0.99)  # never pay more than $0.99

        # Recalculate shares at the higher fill price to stay within budget
        fill_shares = round(edge.position_size_usd / fill_price, 2)

        request = TradeRequest(
            condition_id=bucket.condition_id,
            token_id=bucket.token_id_yes,
            side="BUY",
            price=fill_price,
            size=fill_shares,
            order_type="FOK",  # Fill or Kill — instant fill or cancel
            edge=edge.edge,
            kelly_fraction=edge.kelly_fraction,
            neg_risk=bucket.neg_risk,
            tick_size=bucket.tick_size,
        )

        logger.info(
            "Weather trade: %s %s %s P_noaa=%.3f ask=%.3f fill=%.3f edge=%.3f $%.2f",
            event.city, event.date, bucket.label,
            edge.p_noaa, edge.ask_price, fill_price, edge.edge, edge.position_size_usd,
        )

        order = await self.order_mgr.place_order(request)

        # Track position — use fill_shares (actual order size), not edge.position_size_shares
        if order.status in ("matched", "filled", "dry_run"):
            pos = self.positions.update_from_fill(
                condition_id=bucket.condition_id,
                token_id=bucket.token_id_yes,
                side="YES",
                fill_size=fill_shares,
                fill_price=order.price,
            )
            await self.state_store.save_position(pos)

            # Register order book so exit sweep can monitor this position
            token_id = bucket.token_id_yes
            if token_id not in self._books:
                self._books[token_id] = OrderBookState(token_id=token_id)
                logger.info(
                    "Registered weather token %s in order books for exit monitoring",
                    token_id[:12],
                )

        # Log trade to SQLite
        await self.state_store.log_trade(
            order_id=order.order_id,
            condition_id=bucket.condition_id,
            token_id=bucket.token_id_yes,
            side="BUY_YES_WEATHER",
            price=order.price,
            size=fill_shares,
            status=order.status,
            edge=edge.edge,
            kelly_fraction=edge.kelly_fraction,
            p_hat=edge.p_noaa,
            b_estimate=0.0,
            placed_at=order.placed_at,
            confidence=1.0,  # NOAA confidence (Phase 2: replace with calibrated value)
            market_question=f"Weather: {event.city} {event.date} {bucket.label}",
        )

        return WeatherTradeResult(
            city=event.city,
            date=event.date,
            bucket_label=bucket.label,
            edge=edge.edge,
            p_noaa=edge.p_noaa,
            ask_price=edge.ask_price,
            position_usd=edge.position_size_usd,
            order_status=order.status,
            forecast_source=forecast.source,
        )

    def _get_weather_deployed(self) -> float:
        """Sum of cost_basis for all weather positions."""
        total = 0.0
        for pos in self.positions.get_all_open():
            market = self.scanner.get_market(pos.condition_id)
            if market is not None and self._is_weather_market(market):
                total += pos.cost_basis
        return total

    def _count_city_positions(self, city: str) -> int:
        """Count open positions for a specific city."""
        count = 0
        for pos in self.positions.get_all_open():
            market = self.scanner.get_market(pos.condition_id)
            if market is None:
                continue
            slug = market.event_slug
            if slug and city in slug:
                count += 1
        return count

    @staticmethod
    def _is_weather_market(market) -> bool:
        """Check if a market belongs to a weather event."""
        from src.weather.market_scanner import DAILY_WEATHER_PATTERN
        return bool(DAILY_WEATHER_PATTERN.match(market.event_slug or ""))

    def _log_forecast(self, forecast: WeatherForecast, event: WeatherEvent) -> None:
        """Log forecast data for future calibration (Phase 2)."""
        self._forecast_log.append({
            "city": forecast.city,
            "date": forecast.date,
            "high_temp_f": forecast.high_temp_f,
            "std_dev_f": forecast.std_dev_f,
            "source": forecast.source,
            "bucket_probs": forecast.bucket_probabilities,
            "bucket_count": len(event.buckets),
            "fetched_at": forecast.fetched_at,
        })
        # Keep last 1000 entries in memory
        if len(self._forecast_log) > 1000:
            self._forecast_log = self._forecast_log[-500:]

    def reset_daily_tracker(self) -> None:
        """Reset the traded-today tracker (call at midnight UTC)."""
        self._traded_today.clear()
