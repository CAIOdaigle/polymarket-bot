from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_yaml(name: str) -> dict:
    path = _project_root() / "config" / name
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class PolymarketConfig(BaseSettings):
    private_key: str = Field("", alias="POLYMARKET_PRIVATE_KEY")
    funder_address: str = Field("", alias="POLYMARKET_FUNDER_ADDRESS")
    chain_id: int = Field(137, alias="POLYMARKET_CHAIN_ID")
    clob_host: str = Field("https://clob.polymarket.com", alias="CLOB_HOST")
    gamma_host: str = Field("https://gamma-api.polymarket.com", alias="GAMMA_HOST")

    model_config = {"env_file": ".env", "extra": "ignore"}


class TradingConfig(BaseSettings):
    kelly_fraction: float = 0.5
    min_edge_threshold: float = 0.02
    max_position_usd: float = 100.0
    total_bankroll_usd: float = 1000.0
    max_portfolio_exposure: float = 0.5
    min_liquidity_usd: float = 500.0
    order_type: str = "GTC"
    price_tolerance: float = 0.005
    stale_order_timeout_seconds: int = 300
    daily_loss_limit_usd: float = 100.0
    dry_run: bool = True

    model_config = {"extra": "ignore"}


class LMSRConfig(BaseSettings):
    default_b: float = 100.0
    b_estimation_method: str = "order_book_fit"
    min_b: float = 10.0
    max_b: float = 10000.0

    model_config = {"extra": "ignore"}


class BayesianConfig(BaseSettings):
    default_prior: float = 0.5
    prior_strength: float = 1.0
    signal_decay_hours: float = 24.0
    min_signals_to_trade: int = 1

    model_config = {"extra": "ignore"}


class SignalConfig(BaseSettings):
    orderbook_imbalance: dict = Field(
        default_factory=lambda: {"enabled": True, "sensitivity": 1.0, "min_depth_usd": 100.0}
    )
    lmsr_deviation: dict = Field(
        default_factory=lambda: {"enabled": True, "min_deviation": 0.01}
    )
    volume: dict = Field(
        default_factory=lambda: {"enabled": True, "lookback_trades": 50}
    )
    cross_market: dict = Field(
        default_factory=lambda: {"enabled": True, "min_inconsistency": 0.05, "max_event_markets": 20}
    )
    whale_tracker: dict = Field(
        default_factory=lambda: {
            "enabled": True,
            "whale_threshold_usd": 500.0,
            "volume_spike_multiplier": 3.0,
            "lookback_trades": 100,
            "min_whale_trades": 1,
        }
    )
    related_market: dict = Field(
        default_factory=lambda: {
            "enabled": True,
            "min_similarity": 0.3,
            "max_related": 5,
            "min_price_divergence": 0.10,
        }
    )

    model_config = {"extra": "ignore"}


class ScannerConfig(BaseSettings):
    rescan_interval_seconds: int = 300
    min_volume_24h_usd: float = 100.0
    min_liquidity_usd: float = 500.0
    min_hours_to_expiry: float = 24.0

    model_config = {"extra": "ignore"}


class FeedConfig(BaseSettings):
    ws_reconnect_max_delay: int = 60
    rest_poll_interval_seconds: int = 30
    heartbeat_interval_seconds: int = 10

    model_config = {"extra": "ignore"}


class SlackConfig(BaseSettings):
    webhook_url: str = Field("", alias="SLACK_WEBHOOK_URL")
    notify_on_trade: bool = True
    notify_on_error: bool = True
    notify_on_startup: bool = True
    daily_summary_hour_utc: int = 0

    model_config = {"env_file": ".env", "extra": "ignore"}


class LoggingConfig(BaseSettings):
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/bot.log"
    max_bytes: int = 10_485_760
    backup_count: int = 5

    model_config = {"extra": "ignore"}


class MarketFilterConfig(BaseSettings):
    include_slugs: list[str] = Field(default_factory=list)
    exclude_slugs: list[str] = Field(default_factory=list)
    include_tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class BotConfig:
    def __init__(
        self,
        polymarket: PolymarketConfig,
        trading: TradingConfig,
        lmsr: LMSRConfig,
        bayesian: BayesianConfig,
        signals: SignalConfig,
        scanner: ScannerConfig,
        feed: FeedConfig,
        slack: SlackConfig,
        logging: LoggingConfig,
        market_filter: MarketFilterConfig,
    ):
        self.polymarket = polymarket
        self.trading = trading
        self.lmsr = lmsr
        self.bayesian = bayesian
        self.signals = signals
        self.scanner = scanner
        self.feed = feed
        self.slack = slack
        self.logging = logging
        self.market_filter = market_filter


def load_config(env: Optional[str] = None) -> BotConfig:
    env = env or os.getenv("BOT_ENV", "development")

    defaults = _load_yaml("default.yaml")
    overrides = _load_yaml(f"{env}.yaml") if env != "development" else {}
    merged = _deep_merge(defaults, overrides)

    filters = _load_yaml("markets_filter.yaml")

    return BotConfig(
        polymarket=PolymarketConfig(),
        trading=TradingConfig(**merged.get("trading", {})),
        lmsr=LMSRConfig(**merged.get("lmsr", {})),
        bayesian=BayesianConfig(**merged.get("bayesian", {})),
        signals=SignalConfig(**merged.get("signals", {})),
        scanner=ScannerConfig(**merged.get("scanner", {})),
        feed=FeedConfig(**merged.get("feed", {})),
        slack=SlackConfig(),
        logging=LoggingConfig(**merged.get("logging", {})),
        market_filter=MarketFilterConfig(**filters),
    )
