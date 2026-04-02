"""Order lifecycle management via py-clob-client."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

from src.config import PolymarketConfig, TradingConfig
from src.execution.auth import create_clob_client
from src.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

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

            order_args = OrderArgs(
                price=price,
                size=request.size,
                side=request.side,
                token_id=request.token_id,
            )

            loop = asyncio.get_event_loop()
            try:
                if request.neg_risk:
                    resp = await loop.run_in_executor(
                        None,
                        lambda: self._client.create_and_post_order(
                            order_args,
                            options={"neg_risk": True, "tick_size": str(request.tick_size)},
                        ),
                    )
                else:
                    resp = await loop.run_in_executor(
                        None,
                        lambda: self._client.create_and_post_order(order_args),
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
            bal = await loop.run_in_executor(
                None,
                lambda: self._client.get_balance_allowance(asset_type=0),  # USDC
            )
            return float(bal.get("balance", 0))
        except Exception:
            logger.exception("Failed to fetch USDC balance")
            return 0.0

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
