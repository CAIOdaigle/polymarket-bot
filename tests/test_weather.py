"""Tests for weather market trading module."""

import pytest

from src.weather.forecast_client import WeatherForecastClient, WeatherForecast, CITY_GRIDS
from src.weather.market_scanner import WeatherMarketScanner, WeatherEvent, WeatherBucket
from src.weather.edge_calculator import WeatherEdgeCalculator
from src.market.models import Market, MarketToken


# ── Fixtures ──────────────────────────────────────────────────────────────


def make_weather_market(
    condition_id: str,
    question: str,
    slug: str,
    event_slug: str,
    yes_price: float = 0.10,
) -> Market:
    """Create a Market object that looks like a real Polymarket weather sub-market."""
    return Market(
        condition_id=condition_id,
        question=question,
        slug=slug,
        tokens=[
            MarketToken(token_id=f"yes-{condition_id[:8]}", outcome="Yes", price=yes_price),
            MarketToken(token_id=f"no-{condition_id[:8]}", outcome="No", price=1.0 - yes_price),
        ],
        end_date="2026-04-05T12:00:00Z",
        volume_24h=50000.0,
        liquidity=10000.0,
        neg_risk=True,
        tick_size=0.01,
        active=True,
        tags=["Weather"],
        event_slug=event_slug,
    )


def make_chicago_weather_markets() -> dict[str, Market]:
    """Create markets matching real Polymarket Chicago weather structure."""
    event_slug = "chicago-daily-weather"
    base_slug = "highest-temperature-in-chicago-on-april-5-2026"
    markets = {}

    data = [
        ("cid-below29", "Will the highest temperature in Chicago be 29°F or below on April 5?",
         f"{base_slug}-29forbelow", 0.05),
        ("cid-30-31", "Will the highest temperature in Chicago be between 30-31°F on April 5?",
         f"{base_slug}-30-31f", 0.08),
        ("cid-32-33", "Will the highest temperature in Chicago be between 32-33°F on April 5?",
         f"{base_slug}-32-33f", 0.12),
        ("cid-34-35", "Will the highest temperature in Chicago be between 34-35°F on April 5?",
         f"{base_slug}-34-35f", 0.18),
        ("cid-36-37", "Will the highest temperature in Chicago be between 36-37°F on April 5?",
         f"{base_slug}-36-37f", 0.22),
        ("cid-38-39", "Will the highest temperature in Chicago be between 38-39°F on April 5?",
         f"{base_slug}-38-39f", 0.15),
        ("cid-40-41", "Will the highest temperature in Chicago be between 40-41°F on April 5?",
         f"{base_slug}-40-41f", 0.10),
        ("cid-42-43", "Will the highest temperature in Chicago be between 42-43°F on April 5?",
         f"{base_slug}-42-43f", 0.05),
        ("cid-44-45", "Will the highest temperature in Chicago be between 44-45°F on April 5?",
         f"{base_slug}-44-45f", 0.03),
        ("cid-46-47", "Will the highest temperature in Chicago be between 46-47°F on April 5?",
         f"{base_slug}-46-47f", 0.01),
        ("cid-48plus", "Will the highest temperature in Chicago be 48°F or higher on April 5?",
         f"{base_slug}-48forhigher", 0.01),
    ]

    for cid, question, slug, price in data:
        m = make_weather_market(cid, question, slug, event_slug, price)
        markets[m.condition_id] = m
    return markets


def make_nyc_weather_markets() -> dict[str, Market]:
    """Create markets matching real Polymarket NYC weather structure."""
    event_slug = "nyc-daily-weather"
    base_slug = "highest-temperature-in-nyc-on-april-2-2026"
    markets = {}

    data = [
        ("cid-below59", "Will the highest temperature in NYC be 59°F or below on April 2?",
         f"{base_slug}-59forbelow", 0.90),
        ("cid-60-61", "Will the highest temperature in NYC be between 60-61°F on April 2?",
         f"{base_slug}-60-61f", 0.04),
        ("cid-62-63", "Will the highest temperature in NYC be between 62-63°F on April 2?",
         f"{base_slug}-62-63f", 0.02),
        ("cid-78plus", "Will the highest temperature in NYC be 78°F or higher on April 2?",
         f"{base_slug}-78forhigher", 0.01),
    ]

    for cid, question, slug, price in data:
        m = make_weather_market(cid, question, slug, event_slug, price)
        markets[m.condition_id] = m
    return markets


# ── WeatherMarketScanner tests ───────────────────────────────────────────


class TestWeatherMarketScanner:
    def test_scan_finds_chicago_event_via_event_slug(self):
        scanner = WeatherMarketScanner()
        markets = make_chicago_weather_markets()
        events = scanner.scan(markets)

        assert len(events) == 1
        event = events[0]
        assert event.city == "chicago"
        assert event.date == "2026-04-05"
        assert len(event.buckets) == 11

    def test_scan_finds_nyc_event_via_event_slug(self):
        scanner = WeatherMarketScanner()
        markets = make_nyc_weather_markets()
        events = scanner.scan(markets)

        assert len(events) == 1
        event = events[0]
        assert event.city == "new-york"  # mapped from "nyc"

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

    def test_scan_ignores_international_cities(self):
        """Seoul, London, Tokyo etc. should be skipped in Phase 1."""
        scanner = WeatherMarketScanner()
        m = make_weather_market(
            "cid-seoul",
            "Will the highest temperature in Seoul be 14°C or higher on April 2?",
            "highest-temperature-in-seoul-on-april-2-2026-14corhigher",
            "seoul-daily-weather",
            0.50,
        )
        events = scanner.scan({"cid-seoul": m})
        assert len(events) == 0

    def test_bucket_edges_sorted(self):
        scanner = WeatherMarketScanner()
        markets = make_chicago_weather_markets()
        events = scanner.scan(markets)
        event = events[0]

        edges = event.bucket_edges
        assert edges == sorted(edges)
        assert 30.0 in edges
        assert 48.0 in edges

    def test_bucket_parsing_or_higher(self):
        bucket = WeatherMarketScanner._parse_bucket(
            make_weather_market(
                "cid-test",
                "Will the highest temperature in Chicago be 48°F or higher on April 2?",
                "highest-temperature-in-chicago-on-april-2-2026-48forhigher",
                "chicago-daily-weather",
            )
        )
        assert bucket is not None
        assert bucket.lower_bound_f == 48.0
        assert bucket.upper_bound_f is None
        assert bucket.label == "48 or higher"

    def test_bucket_parsing_between_range(self):
        bucket = WeatherMarketScanner._parse_bucket(
            make_weather_market(
                "cid-test",
                "Will the highest temperature in Chicago be between 34-35°F on April 2?",
                "highest-temperature-in-chicago-on-april-2-2026-34-35f",
                "chicago-daily-weather",
            )
        )
        assert bucket is not None
        assert bucket.lower_bound_f == 34.0
        assert bucket.upper_bound_f == 35.0
        assert bucket.label == "34-35"

    def test_bucket_parsing_or_below(self):
        bucket = WeatherMarketScanner._parse_bucket(
            make_weather_market(
                "cid-test",
                "Will the highest temperature in NYC be 59°F or below on April 2?",
                "highest-temperature-in-nyc-on-april-2-2026-59forbelow",
                "nyc-daily-weather",
            )
        )
        assert bucket is not None
        assert bucket.lower_bound_f is None
        assert bucket.upper_bound_f == 59.0
        assert bucket.label == "below 59"

    def test_multiple_cities_in_one_scan(self):
        scanner = WeatherMarketScanner()
        markets = {}
        markets.update(make_chicago_weather_markets())
        markets.update(make_nyc_weather_markets())
        events = scanner.scan(markets)

        cities = {e.city for e in events}
        assert "chicago" in cities
        assert "new-york" in cities


# ── Gaussian Bucket Probability tests ────────────────────────────────────


class TestGaussianBucketProbs:
    def test_probabilities_sum_to_one(self):
        probs = WeatherForecastClient._gaussian_bucket_probs(
            mean_f=36.0, std_f=4.0,
            bucket_edges=[29, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48],
        )
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.001, f"Probabilities sum to {total}, expected ~1.0"

    def test_peak_near_mean(self):
        probs = WeatherForecastClient._gaussian_bucket_probs(
            mean_f=36.0, std_f=4.0,
            bucket_edges=[29, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48],
        )
        # Bucket containing the mean (34-36 and 36-38) should have highest probability
        p_34_36 = probs.get("34-36", 0)
        p_36_38 = probs.get("36-38", 0)
        peak = max(p_34_36, p_36_38)

        # Peak bucket should be > any tail bucket
        p_below = probs.get("below 29", 0)
        p_above = probs.get("48 or higher", 0)
        assert peak > p_below
        assert peak > p_above

    def test_wider_std_spreads_probability(self):
        edges = [29, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
        narrow = WeatherForecastClient._gaussian_bucket_probs(36.0, 2.0, edges)
        wide = WeatherForecastClient._gaussian_bucket_probs(36.0, 8.0, edges)
        # Wider std should put more probability in tails
        assert wide.get("below 29", 0) > narrow.get("below 29", 0)
        assert wide.get("48 or higher", 0) > narrow.get("48 or higher", 0)

    def test_empty_bucket_edges(self):
        probs = WeatherForecastClient._gaussian_bucket_probs(36.0, 4.0, [])
        assert probs == {}


# ── Edge Calculator tests ────────────────────────────────────────────────


class TestWeatherEdgeCalculator:
    def _make_forecast_and_event(self):
        """Create a forecast and event with one mispriced bucket."""
        forecast = WeatherForecast(
            city="chicago",
            date="2026-04-05",
            high_temp_f=36.0,
            std_dev_f=4.0,
            bucket_probabilities={
                "below 29": 0.04,
                "29-30": 0.02,
                "30-32": 0.05,
                "32-34": 0.10,
                "34-36": 0.19,
                "36-38": 0.22,
                "38-40": 0.18,
                "40-42": 0.10,
                "42-44": 0.05,
                "44-46": 0.03,
                "46-48": 0.01,
                "48 or higher": 0.01,
            },
            source="gaussian_fallback",
        )

        event = WeatherEvent(
            event_slug="chicago-daily-weather",
            city="chicago",
            date="2026-04-05",
            buckets=[
                WeatherBucket("cid-below29", "yes-below29", "no-below29",
                              "below 29", None, 29.0, 0.05, True, 0.01),
                WeatherBucket("cid-30-32", "yes-30-32", "no-30-32",
                              "30-32", 30.0, 32.0, 0.05, True, 0.01),
                WeatherBucket("cid-34-36", "yes-34-36", "no-34-36",
                              "34-36", 34.0, 36.0, 0.08, True, 0.01),  # mispriced! NOAA=0.19
                WeatherBucket("cid-36-38", "yes-36-38", "no-36-38",
                              "36-38", 36.0, 38.0, 0.15, True, 0.01),
                WeatherBucket("cid-38-40", "yes-38-40", "no-38-40",
                              "38-40", 38.0, 40.0, 0.12, True, 0.01),
                WeatherBucket("cid-48plus", "yes-48plus", "no-48plus",
                              "48 or higher", 48.0, None, 0.01, True, 0.01),
            ],
        )
        return forecast, event

    def test_finds_edges_above_threshold(self):
        calc = WeatherEdgeCalculator(min_edge=0.08, bankroll_usd=50.0, max_position_usd=5.0)
        forecast, event = self._make_forecast_and_event()

        edges = calc.compute_edges(forecast, event, books={})
        assert len(edges) > 0
        for e in edges:
            assert e.edge >= 0.08

    def test_highest_edge_first(self):
        calc = WeatherEdgeCalculator(min_edge=0.05, bankroll_usd=50.0, max_position_usd=5.0)
        forecast, event = self._make_forecast_and_event()

        edges = calc.compute_edges(forecast, event, books={})
        for i in range(len(edges) - 1):
            assert edges[i].edge >= edges[i + 1].edge

    def test_mispriced_bucket_detected(self):
        """Bucket 34-36 is priced at 0.08 but NOAA says 0.19 → edge = 0.11."""
        calc = WeatherEdgeCalculator(min_edge=0.08, bankroll_usd=50.0, max_position_usd=5.0)
        forecast, event = self._make_forecast_and_event()

        edges = calc.compute_edges(forecast, event, books={})
        bucket_labels = [e.bucket_label for e in edges]
        assert "34-36" in bucket_labels

        edge_34_36 = next(e for e in edges if e.bucket_label == "34-36")
        assert edge_34_36.p_noaa == pytest.approx(0.19)
        assert edge_34_36.ask_price == pytest.approx(0.08)
        assert edge_34_36.edge == pytest.approx(0.11)

    def test_position_sizing_respects_max(self):
        calc = WeatherEdgeCalculator(min_edge=0.05, bankroll_usd=50.0, max_position_usd=5.0)
        forecast, event = self._make_forecast_and_event()

        edges = calc.compute_edges(forecast, event, books={})
        for e in edges:
            assert e.position_size_usd <= 5.0

    def test_no_edges_when_prices_match_forecast(self):
        calc = WeatherEdgeCalculator(min_edge=0.08, bankroll_usd=50.0, max_position_usd=5.0)
        forecast = WeatherForecast(
            city="chicago", date="2026-04-05", high_temp_f=36.0, std_dev_f=4.0,
            bucket_probabilities={"34-36": 0.19},
            source="gaussian_fallback",
        )
        event = WeatherEvent(
            event_slug="chicago-daily-weather", city="chicago", date="2026-04-05",
            buckets=[
                WeatherBucket("cid-34-36", "yes-34-36", "no-34-36",
                              "34-36", 34.0, 36.0, 0.19, True, 0.01),
            ],
        )
        edges = calc.compute_edges(forecast, event, books={})
        assert len(edges) == 0

    def test_deployed_capital_reduces_position(self):
        calc = WeatherEdgeCalculator(
            min_edge=0.05, bankroll_usd=50.0, max_position_usd=5.0,
            max_portfolio_exposure=0.50,
        )
        forecast, event = self._make_forecast_and_event()

        edges_fresh = calc.compute_edges(forecast, event, books={}, total_deployed_usd=0.0)
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


# ── City Slug Mapping tests ──────────────────────────────────────────────


class TestCitySlugMap:
    def test_nyc_maps_to_new_york(self):
        from src.weather.market_scanner import CITY_SLUG_MAP
        assert CITY_SLUG_MAP["nyc"] == "new-york"

    def test_chicago_maps_to_chicago(self):
        from src.weather.market_scanner import CITY_SLUG_MAP
        assert CITY_SLUG_MAP["chicago"] == "chicago"

    def test_all_noaa_cities_reachable(self):
        from src.weather.market_scanner import CITY_SLUG_MAP
        noaa_cities = set(CITY_GRIDS.keys())
        mapped_cities = set(CITY_SLUG_MAP.values())
        # Every NOAA city should be reachable via at least one slug mapping
        for city in noaa_cities:
            assert city in mapped_cities, f"NOAA city {city} not reachable via CITY_SLUG_MAP"
