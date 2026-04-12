"""Configuration for the crypto sniper service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

from polymarket_common.config import (
    PolymarketConfig,
    TradingConfig,
    SlackConfig,
    LoggingConfig,
    _deep_merge,
)


class TASniperConfig(BaseSettings):
    """TA-based 5-min BTC sniper configuration."""
    enabled: bool = True
    mode: str = "safe"  # safe, aggressive, degen
    min_confidence: float = 0.30
    entry_seconds_before_close: int = 10
    min_ev_edge: float = 0.05  # Black-Scholes EV gate
    max_trades_per_session: int = 0  # 0 = unlimited
    eval_interval_seconds: float = 2.0  # polling interval during entry window
    use_timesfm: bool = False  # optional TimesFM forecast signal

    model_config = {"extra": "ignore"}


class OracleSniperConfig(BaseSettings):
    """Oracle lag arbitrage configuration."""
    enabled: bool = False
    poll_interval_ms: int = 200
    entry_window_min_seconds: int = 8  # T-8s: latest entry
    entry_window_max_seconds: int = 55  # T-55s: earliest entry
    min_lag_score: float = 0.60
    min_net_edge: float = 0.05
    confirmation_threshold: float = 0.003  # 0.3% delta minimum
    chainlink_rtds_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    polygon_rpc_url: str = "https://polygon-rpc.com"

    model_config = {"extra": "ignore"}


class KellyConfig(BaseSettings):
    """Token-price-aware Kelly sizing."""
    fraction: float = 0.25  # quarter Kelly
    min_bet_usd: float = 4.75  # Polymarket minimum
    # Mode-specific max bet as fraction of bankroll
    max_bet_fraction_safe: float = 0.01
    max_bet_fraction_aggressive: float = 0.02
    max_bet_fraction_degen: float = 0.03
    max_bet_fraction_oracle: float = 0.02
    # Token price ceilings per mode
    max_token_price_safe: float = 0.62
    max_token_price_aggressive: float = 0.70
    max_token_price_degen: float = 0.80
    max_token_price_oracle: float = 0.80

    model_config = {"extra": "ignore"}


class FeedsConfig(BaseSettings):
    """Exchange feed configuration."""
    binance_ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    binance_rest_url: str = "https://api.binance.com/api/v3"
    binance_symbol: str = "BTCUSDT"

    model_config = {"extra": "ignore"}


class StrategiesConfig(BaseSettings):
    """Master toggle for crypto strategies."""
    ta_sniper: bool = True
    oracle_sniper: bool = False

    model_config = {"extra": "ignore"}


class SniperConfig:
    """Top-level config aggregating all sub-configs."""
    def __init__(
        self,
        polymarket: PolymarketConfig,
        trading: TradingConfig,
        slack: SlackConfig,
        logging: LoggingConfig,
        strategies: StrategiesConfig,
        ta_sniper: TASniperConfig,
        oracle_sniper: OracleSniperConfig,
        kelly: KellyConfig,
        feeds: FeedsConfig,
    ):
        self.polymarket = polymarket
        self.trading = trading
        self.slack = slack
        self.logging = logging
        self.strategies = strategies
        self.ta_sniper = ta_sniper
        self.oracle_sniper = oracle_sniper
        self.kelly = kelly
        self.feeds = feeds


def load_sniper_config(env: Optional[str] = None) -> SniperConfig:
    env = env or os.getenv("SNIPER_ENV", "development")
    config_dir = Path(__file__).resolve().parent.parent / "config"

    defaults = {}
    default_path = config_dir / "default.yaml"
    if default_path.exists():
        with open(default_path) as f:
            defaults = yaml.safe_load(f) or {}

    overrides = {}
    if env != "development":
        override_path = config_dir / f"{env}.yaml"
        if override_path.exists():
            with open(override_path) as f:
                overrides = yaml.safe_load(f) or {}

    merged = _deep_merge(defaults, overrides)

    return SniperConfig(
        polymarket=PolymarketConfig(),
        trading=TradingConfig(**merged.get("trading", {})),
        slack=SlackConfig(**merged.get("slack", {})),
        logging=LoggingConfig(**merged.get("logging", {})),
        strategies=StrategiesConfig(**merged.get("strategies", {})),
        ta_sniper=TASniperConfig(**merged.get("ta_sniper", {})),
        oracle_sniper=OracleSniperConfig(**merged.get("oracle_sniper", {})),
        kelly=KellyConfig(**merged.get("kelly", {})),
        feeds=FeedsConfig(**merged.get("feeds", {})),
    )
