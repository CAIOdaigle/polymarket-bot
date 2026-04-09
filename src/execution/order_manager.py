"""Order lifecycle management via py-clob-client."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, CreateOrderOptions

from src.config import PolymarketConfig, TradingConfig
from src.execution.auth import create_clob_client
from src.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Monkey-patch py-clob-client's decimal_places to handle float noise.
# The original uses Decimal(x.__str__()) which sees artifacts like
# 15.219999999999999 as 15 decimal places, breaking amount rounding.
# Also patch to_token_decimals to round amounts to 2dp before conversion,
# because the CLOB server requires maker amounts with max 2 decimal places
# but the library's ROUNDING_CONFIG allows up to 4.
import py_clob_client.order_builder.helpers as _clob_helpers
from decimal import ROUND_DOWN as _ROUND_DOWN

_orig_to_token_decimals = _clob_helpers.to_token_decimals

def _patched_decimal_places(x: float) -> int:
    rounded = round(x, 10)
    return abs(Decimal(str(rounded)).normalize().as_tuple().exponent)

def _patched_to_token_decimals(x: float) -> int:
    # Round to 2 decimal places (CLOB server requirement) before converting
    d = Decimal(str(round(x, 10))).quantize(Decimal("0.01"), rounding=_ROUND_DOWN)
    return int(d * 10**6)

_clob_helpers.decimal_places = _patched_decimal_places
_clob_helpers.to_token_decimals = _patched_to_token_decimals

BUY = "BUY"
SELL = "SELL"


@dataclass
class TradeRequest:
    condition_id: str
    token_id: str
    side: str  # BUY or SELL
    price: float
    size: float
    order_type: str  # GTC, FOK, FAK
    edge: float
    kelly_fraction: float
    neg_risk: bool
    tick_size: float


@dataclass
class OrderRecord:
    order_id: str
    condition_id: str
    token_id: str
    side: str
    price: float
    size: float
    filled_size: float = 0.0
    status: str = "pending"
    placed_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class OrderManager:
    def __init__(self, poly_config: PolymarketConfig, trading_config: TradingConfig):
        self._poly_config = poly_config
        self._trading_config = trading_config
        self._client: Optional[ClobClient] = None
        self._orders: dict[str, OrderRecord] = {}
        self._lock = asyncio.Lock()
        self._rate_limiter = RateLimiter()
        self._dry_run = trading_config.dry_run

    async def initialize(self) -> None:
        loop = asyncio.get_event_loop()
        self._client = await loop.run_in_executor(
            None, create_clob_client, self._poly_config
        )
        logger.info("OrderManager initialized (dry_run=%s)", self._dry_run)

    async def place_order(self, request: TradeRequest) -> OrderRecord:
        async with self._lock:
            await self._rate_limiter.orders.acquire()

            if self._dry_run:
                order_id = f"DRY-{int(time.time()*1000)}"
                logger.info(
                    "[DRY RUN] Would place %s %s %.2f shares @ %.4f (edge=%.4f)",
                    request.side,
                    request.token_id[:8],
                    request.size,
                    request.price,
                    request.edge,
                )
                record = OrderRecord(
                    order_id=order_id,
                    condition_id=request.condition_id,
                    token_id=request.token_id,
                    side=request.side,
                    price=request.price,
                    size=request.size,
                    status="dry_run",
                    placed_at=time.time(),
                )
                self._orders[order_id] = record
                return record

            # Real order
            order_type_map = {
                "GTC": OrderType.GTC,
                "FOK": OrderType.FOK,
                "GTD": OrderType.GTD,
            }
            ot = order_type_map.get(request.order_type, OrderType.GTC)

            # Round price to tick size
            price = round(
                round(request.price / request.tick_size) * request.tick_size,
                10,
            )

            # Round size to 2dp (CLOB max for taker amount)
            size = round(request.size, 2)

            order_args = OrderArgs(
                price=price,
                size=size,
                side=request.side,
                token_id=request.token_id,
            )

            loop = asyncio.get_event_loop()
            try:
                # Always pass CreateOrderOptions with tick_size so the library
                # applies correct rounding for maker/taker amounts.
                options = CreateOrderOptions(
                    neg_risk=request.neg_risk,
                    tick_size=str(request.tick_size),
                )
                signed_order = await loop.run_in_executor(
                    None,
                    lambda: self._client.create_order(order_args, options),
                )
                resp = await loop.run_in_executor(
                    None,
                    lambda: self._client.post_order(signed_order, orderType=ot),
                )

                order_id = resp.get("orderID", resp.get("id", f"UNK-{int(time.time()*1000)}"))
                status = resp.get("status", "live")

                record = OrderRecord(
                    order_id=order_id,
                    condition_id=request.condition_id,
                    token_id=request.token_id,
                    side=request.side,
                    price=price,
                    size=request.size,
                    status=status,
                    placed_at=time.time(),
                )
                self._orders[order_id] = record

                logger.info(
                    "Placed order %s: %s %s %.2f @ %.4f (status=%s, edge=%.4f)",
                    order_id,
                    request.side,
                    request.token_id[:8],
                    request.size,
                    price,
                    status,
                    request.edge,
                )
                return record

            except Exception as exc:
                logger.exception(
                    "Failed to place order for %s (side=%s token=%s price=%.4f size=%.2f neg_risk=%s tick=%s): %s",
                    request.condition_id, request.side, request.token_id[:10],
                    price, request.size, request.neg_risk, request.tick_size, exc,
                )
                record = OrderRecord(
                    order_id=f"FAILED-{int(time.time()*1000)}",
                    condition_id=request.condition_id,
                    token_id=request.token_id,
                    side=request.side,
                    price=price,
                    size=request.size,
                    status="failed",
                    placed_at=time.time(),
                )
                return record

    async def cancel_order(self, order_id: str) -> bool:
        if self._dry_run:
            logger.info("[DRY RUN] Would cancel order %s", order_id)
            return True

        loop = asyncio.get_event_loop()
        try:
            await self._rate_limiter.orders.acquire()
            await loop.run_in_executor(None, self._client.cancel, order_id)
            if order_id in self._orders:
                self._orders[order_id].status = "cancelled"
            logger.info("Cancelled order %s", order_id)
            return True
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    async def cancel_all(self) -> None:
        if self._dry_run:
            logger.info("[DRY RUN] Would cancel all orders")
            return

        loop = asyncio.get_event_loop()
        try:
            await self._rate_limiter.orders.acquire()
            await loop.run_in_executor(None, self._client.cancel_all)
            for rec in self._orders.values():
                if rec.status in ("pending", "live", "open"):
                    rec.status = "cancelled"
            logger.info("Cancelled all open orders")
        except Exception:
            logger.exception("Failed to cancel all orders")

    async def cancel_stale_orders(self, max_age_seconds: Optional[float] = None) -> int:
        max_age = max_age_seconds or self._trading_config.stale_order_timeout_seconds
        now = time.time()
        cancelled = 0
        for oid, rec in list(self._orders.items()):
            if rec.status in ("live", "open", "pending") and (now - rec.placed_at) > max_age:
                if await self.cancel_order(oid):
                    cancelled += 1
        if cancelled:
            logger.info("Cancelled %d stale orders", cancelled)
        return cancelled

    async def get_usdc_balance(self) -> float:
        if self._client is None:
            return 0.0
        loop = asyncio.get_event_loop()
        try:
            await self._rate_limiter.public.acquire()
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            bal = await loop.run_in_executor(
                None,
                lambda: self._client.get_balance_allowance(
                    BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
                ),
            )
            return self._normalize_usdc_balance(bal)
        except Exception:
            logger.exception("Failed to fetch USDC balance")
            return 0.0

    @staticmethod
    def _normalize_usdc_balance(balance_payload: dict) -> float:
        """Normalize balance payload into USDC units.

        The Polymarket CLOB balance-allowance endpoint returns USDC in base
        units (6 decimals on Polygon). Always divide by 1e6.
        """
        raw = balance_payload.get("balance", 0)
        raw_str = str(raw).strip()
        if not raw_str:
            return 0.0

        try:
            amount = Decimal(raw_str)
        except (InvalidOperation, TypeError, ValueError):
            return float(raw or 0.0)

        # USDC on Polygon always has 6 decimals — the CLOB API returns base units
        USDC_DECIMALS = Decimal("1000000")

        # If a decimals field is present, use it; otherwise default to 6
        decimals = None
        for key in ("decimals", "assetDecimals", "tokenDecimals"):
            if key in balance_payload:
                try:
                    decimals = int(balance_payload[key])
                    break
                except (TypeError, ValueError):
                    continue

        if decimals is not None:
            return float(amount / (Decimal(10) ** decimals))

        # No decimals field — normalize from base units (6 decimals)
        # Values that look already-normalized (e.g. "365.71") would be < 1000
        # after division, which is fine. Values in base units like "365706340"
        # become ~365.71.
        if amount > Decimal("1000"):
            normalized = amount / USDC_DECIMALS
            logger.info(
                "Balance %s -> $%.2f USDC (divided by 1e6)",
                raw_str, float(normalized),
            )
            return float(normalized)

        return float(amount)

    async def poll_open_orders(self) -> list[OrderRecord]:
        """Poll the CLOB API for status of all 'live' orders.

        Returns list of orders whose status changed (e.g. filled).
        """
        if self._client is None or self._dry_run:
            return []

        changed: list[OrderRecord] = []
        loop = asyncio.get_event_loop()

        for oid, rec in list(self._orders.items()):
            if rec.status not in ("live", "open", "pending"):
                continue
            try:
                await self._rate_limiter.orders.acquire()
                resp = await loop.run_in_executor(
                    None, self._client.get_order, oid
                )
                new_status = resp.get("status", rec.status)
                size_matched = float(resp.get("size_matched", 0) or 0)

                if new_status != rec.status or size_matched != rec.filled_size:
                    old_status = rec.status
                    rec.status = new_status
                    rec.filled_size = size_matched
                    rec.updated_at = time.time()
                    changed.append(rec)
                    logger.info(
                        "Order %s status: %s -> %s (filled=%.2f/%.2f)",
                        oid, old_status, new_status, size_matched, rec.size,
                    )
            except Exception:
                logger.debug("Failed to poll order %s", oid, exc_info=True)

        return changed

    @property
    def open_orders(self) -> dict[str, OrderRecord]:
        return {
            oid: rec
            for oid, rec in self._orders.items()
            if rec.status in ("live", "open", "pending")
        }

    @property
    def all_orders(self) -> dict[str, OrderRecord]:
        return self._orders
