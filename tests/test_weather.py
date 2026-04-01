"""Tests for weather market trading module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.weather.forecast_client import WeatherForecastClient, WeatherForecast, CITY_GRIDS
from src.weather.market_scanner import WeatherMarketScanner, WeatherEvent, WeatherBucket
from src.weather.edge_calculator import WeatherEdgeCalculator, WeatherEdge
from src.market.models import Market, MarketToken


# ── Fixtures ──────────────────────────────────────────────────────────────


def make_weather_market(
    condition_id: str,
    question: str,
    event_slug: str,
    yes_price: float = 0.10,
) -> Market:
    """Create a Market object that looks like a Polymarket weather sub-market."""
    return Market(
        condition_id=condition_id,
        question=question,
        slug=f"weather-{condition_id[:8]}",
        tokens=[
            MarketToken(token_id=f"yes-{condition_id[:8]}", outcome="Yes", price=yes_price),
            MarketToken(token_id=f"no-{condition_id[:8]}", outcome="No", price=1.0 - yes_price),
        ],
        end_date="2026-04-05T23:59:00Z",
        volume_24h=50000.0,
        liquidity=10000.0,
        neg_risk=True,
        tick_size=0.01,
        active=True,
        tags=["weather"],
        event_slug=event_slug,
    )


def make_weather_markets() -> dict[str, Market]:
    """Create a set of weather markets for NYC on April 5."""
    slug = "highest-temperature-in-new-york-on-april-5-2026"
    markets = {}
    # Below 60°F
    m = make_weather_market("cid-below60", "Will the temperature be below 60°F?", slug, 0.05)
    markets[m.condition_id] = m
    # 60-62°F
    m = make_weather_market("cid-60-62", "Will the temperature be 60°F to 62°F?", slug, 0.08)
    markets[m.condition_id] = m
    # 62-64°F
    m = make_weather_market("cid-62-64", "Will the temperature be 62°F to 64°F?", slug, 0.12)
    markets[m.condition_id] = m
    # 64-66°F
    m = make_weather_market("cid-64-66", "Will the temperature be 64°F to 66°F?", slug, 0.18)
    markets[m.condition_id] = m
    # 66-68°F
    m = make_weather_market("cid-66-68", "Will the temperature be 66°F to 68°F?", slug, 0.22)
    markets[m.condition_id] = m
    # 68-70°F
    m = make_weather_market("cid-68-70", "Will the temperature be 68°F to 70°F?", slug, 0.15)
    markets[m.condition_id] = m
    # 70-72°F
    m = make_weather_market("cid-70-72", "Will the temperature be 70°F to 72°F?", slug, 0.10)
    markets[m.condition_id] = m
    # 72 or higher
    m = make_weather_market("cid-72plus", "Will the temperature be 72°F or higher?", slug, 0.05)
    markets[m.condition_id] = m
    return markets


# ── WeatherMarketScanner tests ───────────────────────────────────────────


class TestWeatherMarketScanner:
    def test_scan_finds_weather_events(self):
        scanner = WeatherMarketScanner()
        markets = make_weather_markets()
        events = scanner.scan(markets)

        assert len(events) == 1
        event = events[0]
        assert event.city == "new-york"
        assert event.date == "2026-04-05"
        assert len(event.buckets) == 8

    def test_scan_ignores_non_weather_markets(self):
        scanner = WeatherMarketScanner()
        markets = {
            "cid-politics": Market(
                condition_id="cid-politics",
                question="Will Biden win?",
                slug="biden-win",
                tokens=[
                    MarketToken(token_id="yes-pol", outcome="Yes", price=0.45),
                    MarketToken(token_id="no-pol", outcome="No", price=0.55),
                ],
                end_date="2026-11-03T23:59:00Z",
                volume_24h=100000.0,
                liquidity=50000.0,
                neg_risk=False,
                tick_size=0.01,
                event_slug="us-election-2026",
            ),
        }
        events = scanner.scan(markets)
        assert len(events) == 0

    def test_bucket_edges_sorted(self):
        scanner = WeatherMarketScanner()
        markets = make_weather_markets()
        events = scanner.scan(markets)
        event = events[0]

        edges = event.bucket_edges
        assert edges == sorted(edges)
        assert 60.0 in edges
        assert 72.0 in edges

    def test_bucket_parsing_or_higher(self):
        scanner = WeatherMarketScanner()
        m = make_weather_market("cid-test", "Will the temperature be 72°F or higher?",
                                "highest-temperature-in-chicago-on-march-15-2026")
        events = scanner.scan({"cid-test": m})
        # Only 1 bucket — needs >= 3 to be a valid event
        assert len(events) == 0  # correctly filtered out (too few buckets)

    def test_bucket_parsing_range(self):
        bucket = WeatherMarketScanner._parse_bucket(
            make_weather_market("cid-test", "Will the temperature be 64°F to 66°F?",
                                "highest-temperature-in-miami-on-june-1-2026")
        )
        assert bucket is not None
        assert bucket.lower_bound_f == 64.0
        assert bucket.upper_bound_f == 66.0
        assert bucket.label == "64-66"

    def test_bucket_parsing_below(self):
        bucket = WeatherMarketScanner._parse_bucket(
            make_weather_market("cid-test", "Will the temperature be below 60°F?",
                                "highest-temperature-in-denver-on-jan-1-2026")
        )
        assert bucket is not None
        assert bucket.lower_bound_f is None
        assert bucket.upper_bound_f == 60.0
        assert bucket.label == "below 60"


# ── Gaussian Bucket Probability tests ────────────────────────────────────


class TestGaussianBucketProbs:
    def test_probabilities_sum_to_one(self):
        probs = WeatherForecastClient._gaussian_bucket_probs(
            mean_f=66.0, std_f=3.5,
            bucket_edges=[60, 62, 64, 66, 68, 70, 72],
        )
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.001, f"Probabilities sum to {total}, expected ~1.0"

    def test_peak_near_mean(self):
        probs = WeatherForecastClient._gaussian_bucket_probs(
            mean_f=66.0, std_f=3.5,
            bucket_edges=[60, 62, 64, 66, 68, 70, 72],
        )
        # Bucket containing the mean (64-66 and 66-68) should have highest probability
        p_64_66 = probs.get("64-66", 0)
        p_66_68 = probs.get("66-68", 0)
        peak = max(p_64_66, p_66_68)

        # Peak bucket should be > any tail bucket
        p_below = probs.get("below 60", 0)
        p_above = probs.get("72 or higher", 0)
        assert peak > p_below
        assert peak > p_above

    def test_wider_std_spreads_probability(self):
        narrow = WeatherForecastClient._gaussian_bucket_probs(
            mean_f=66.0, std_f=2.0,
            bucket_edges=[60, 62, 64, 66, 68, 70, 72],
        )
        wide = WeatherForecastClient._gaussian_bucket_probs(
            mean_f=66.0, std_f=6.0,
            bucket_edges=[60, 62, 64, 66, 68, 70, 72],
        )
        # Wider std should put more probability in tails
        assert wide.get("below 60", 0) > narrow.get("below 60", 0)
        assert wide.get("72 or higher", 0) > narrow.get("72 or higher", 0)

    def test_empty_bucket_edges(self):
        probs = WeatherForecastClient._gaussian_bucket_probs(66.0, 3.5, [])
        assert probs == {}


# ── Edge Calculator tests ────────────────────────────────────────────────


class TestWeatherEdgeCalculator:
    def _make_forecast_and_event(self):
        """Create a forecast and event for testing."""
        forecast = WeatherForecast(
            city="new-york",
            date="2026-04-05",
            high_temp_f=66.0,
            std_dev_f=3.5,
            bucket_probabilities={
                "below 60": 0.04,
                "60-62": 0.07,
                "62-64": 0.13,
                "64-66": 0.22,
                "66-68": 0.22,
                "68-70": 0.16,
                "70-72": 0.09,
                "72 or higher": 0.07,
            },
            source="gaussian_fallback",
        )

        event = WeatherEvent(
            event_slug="highest-temperature-in-new-york-on-april-5-2026",
            city="new-york",
            date="2026-04-05",
            buckets=[
                WeatherBucket("cid-below60", "yes-below60", "no-below60",
                              "below 60", None, 60.0, 0.05, True, 0.01),
                WeatherBucket("cid-60-62", "yes-60-62", "no-60-62",
                              "60-62", 60.0, 62.0, 0.08, True, 0.01),
                WeatherBucket("cid-62-64", "yes-62-64", "no-62-64",
                              "62-64", 62.0, 64.0, 0.04, True, 0.01),  # mispriced! NOAA says 0.13
                WeatherBucket("cid-64-66", "yes-64-66", "no-64-66",
                              "64-66", 64.0, 66.0, 0.18, True, 0.01),
                WeatherBucket("cid-66-68", "yes-66-68", "no-66-68",
                              "66-68", 66.0, 68.0, 0.15, True, 0.01),
                WeatherBucket("cid-68-70", "yes-68-70", "no-68-70",
                              "68-70", 68.0, 70.0, 0.12, True, 0.01),
                WeatherBucket("cid-70-72", "yes-70-72", "no-70-72",
                              "70-72", 70.0, 72.0, 0.10, True, 0.01),
                WeatherBucket("cid-72plus", "yes-72plus", "no-72plus",
                              "72 or higher", 72.0, None, 0.05, True, 0.01),
            ],
        )
        return forecast, event

    def test_finds_edges_above_threshold(self):
        calc = WeatherEdgeCalculator(min_edge=0.08, bankroll_usd=50.0, max_position_usd=5.0)
        forecast, event = self._make_forecast_and_event()

        edges = calc.compute_edges(forecast, event, books={})
        assert len(edges) > 0
        # All returned edges should be >= 0.08
        for e in edges:
            assert e.edge >= 0.08

    def test_highest_edge_first(self):
        calc = WeatherEdgeCalculator(min_edge=0.05, bankroll_usd=50.0, max_position_usd=5.0)
        forecast, event = self._make_forecast_and_event()

        edges = calc.compute_edges(forecast, event, books={})
        for i in range(len(edges) - 1):
            assert edges[i].edge >= edges[i + 1].edge

    def test_mispriced_bucket_detected(self):
        """Bucket 62-64 is priced at 0.04 but NOAA says 0.13 → edge = 0.09."""
        calc = WeatherEdgeCalculator(min_edge=0.08, bankroll_usd=50.0, max_position_usd=5.0)
        forecast, event = self._make_forecast_and_event()

        edges = calc.compute_edges(forecast, event, books={})
        bucket_labels = [e.bucket_label for e in edges]
        assert "62-64" in bucket_labels

        edge_62_64 = next(e for e in edges if e.bucket_label == "62-64")
        assert edge_62_64.p_noaa == pytest.approx(0.13)
        assert edge_62_64.ask_price == pytest.approx(0.04)
        assert edge_62_64.edge == pytest.approx(0.09)

    def test_position_sizing_respects_max(self):
        calc = WeatherEdgeCalculator(
            min_edge=0.05, bankroll_usd=50.0, max_position_usd=5.0,
        )
        forecast, event = self._make_forecast_and_event()

        edges = calc.compute_edges(forecast, event, books={})
        for e in edges:
            assert e.position_size_usd <= 5.0

    def test_no_edges_when_prices_match_forecast(self):
        """If market prices equal NOAA probs, no edge exists."""
        calc = WeatherEdgeCalculator(min_edge=0.08, bankroll_usd=50.0, max_position_usd=5.0)
        forecast = WeatherForecast(
            city="new-york",
            date="2026-04-05",
            high_temp_f=66.0,
            std_dev_f=3.5,
            bucket_probabilities={"64-66": 0.22},
            source="gaussian_fallback",
        )
        event = WeatherEvent(
            event_slug="highest-temperature-in-new-york-on-april-5-2026",
            city="new-york",
            date="2026-04-05",
            buckets=[
                WeatherBucket("cid-64-66", "yes-64-66", "no-64-66",
                              "64-66", 64.0, 66.0, 0.22, True, 0.01),  # price = probability
            ],
        )
        edges = calc.compute_edges(forecast, event, books={})
        assert len(edges) == 0

    def test_deployed_capital_reduces_position(self):
        """Position should be smaller when close to exposure limit."""
        calc = WeatherEdgeCalculator(
            min_edge=0.05,
            bankroll_usd=50.0,
            max_position_usd=5.0,
            max_portfolio_exposure=0.50,
        )
        forecast, event = self._make_forecast_and_event()

        # Fresh bankroll
        edges_fresh = calc.compute_edges(forecast, event, books={}, total_deployed_usd=0.0)
        # Near exposure limit
        edges_full = calc.compute_edges(forecast, event, books={}, total_deployed_usd=24.0)

        if edges_fresh and edges_full:
            assert edges_full[0].position_size_usd <= edges_fresh[0].position_size_usd


# ── City Grid Mapping tests ──────────────────────────────────────────────


class TestCityGrids:
    def test_all_eleven_cities_present(self):
        expected = {
            "new-york", "chicago", "miami", "los-angeles", "houston",
            "phoenix", "denver", "atlanta", "dallas", "san-francisco", "seattle",
        }
        assert set(CITY_GRIDS.keys()) == expected

    def test_grid_values_are_valid(self):
        for city, grid in CITY_GRIDS.items():
            assert "wfo" in grid, f"{city} missing wfo"
            assert "x" in grid, f"{city} missing x"
            assert "y" in grid, f"{city} missing y"
            assert "std_f" in grid, f"{city} missing std_f"
            assert isinstance(grid["x"], int), f"{city} grid x should be int"
            assert isinstance(grid["y"], int), f"{city} grid y should be int"
            assert 1.0 < grid["std_f"] < 10.0, f"{city} std_f out of range"
