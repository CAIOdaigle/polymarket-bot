"""Chainlink oracle feeds — strike price + real-time BTC/USD.

Combines:
- ChainlinkOracleFeed: strike price for settlement (Vatic API + Binance fallback)
- OracleLagDetector: real-time Chainlink monitoring via Polymarket RTDS WebSocket
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

VATIC_API_URL = "https://api.vatic.trade/chainlink/price"
DEFAULT_RTDS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
DEFAULT_RPC_URL = "https://polygon-rpc.com"
CHAINLINK_BTC_POLYGON = "0xc907E116054Ad103354f2D350FD2514433D57F6f"


class ChainlinkOracleFeed:
    """Fetches BTC/USD strike price for settlement + real-time Chainlink monitoring."""

    def __init__(
        self,
        rtds_url: str = DEFAULT_RTDS_URL,
        rpc_url: str = DEFAULT_RPC_URL,
    ):
        self._rtds_url = rtds_url
        self._rpc_url = rpc_url
        self._strike_cache: dict[int, float] = {}
        self._chainlink_price: float = 0.0
        self._chainlink_ts: float = 0.0
        self._price_history: list[tuple[float, float]] = []
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Start the real-time Chainlink RTDS listener."""
        self._session = aiohttp.ClientSession()
        self._ws_task = asyncio.create_task(self._rtds_listener())
        await asyncio.sleep(2)  # Wait for initial price
        logger.info(
            "ChainlinkOracleFeed connected. BTC: $%.2f",
            self._chainlink_price,
        )

    async def _rtds_listener(self) -> None:
        """Subscribe to Polymarket RTDS crypto_prices_chainlink topic."""
        subscribe_msg = json.dumps({
            "action": "subscribe",
            "subscriptions": [{
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": '{"symbol":"btc/usd"}',
            }],
        })

        while True:
            try:
                async with self._session.ws_connect(
                    self._rtds_url,
                    heartbeat=5,
                    timeout=aiohttp.ClientTimeout(total=None),
                ) as ws:
                    await ws.send_str(subscribe_msg)
                    logger.info("RTDS WebSocket connected — Chainlink BTC/USD feed active")

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._handle_rtds_message(msg.data)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("RTDS WS error: %s — reconnecting in 3s", e)
                await asyncio.sleep(3)
                # Fallback: try Polygon on-chain
                try:
                    price = await self._fetch_polygon_onchain()
                    if price > 0:
                        self._update_price(price, time.time())
                except Exception:
                    pass

    def _handle_rtds_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
            if msg.get("topic") == "crypto_prices_chainlink":
                payload = msg.get("payload", {})
                if payload.get("symbol") == "btc/usd":
                    price = float(payload["value"])
                    ts = float(payload.get("timestamp", time.time() * 1000)) / 1000
                    self._update_price(price, ts)
        except Exception:
            pass

    def _update_price(self, price: float, ts: float) -> None:
        self._chainlink_price = price
        self._chainlink_ts = ts
        self._price_history.append((ts, price))
        cutoff = time.time() - 120
        self._price_history = [(t, p) for t, p in self._price_history if t > cutoff]

    async def _fetch_polygon_onchain(self) -> float:
        """Read Chainlink BTC/USD from Polygon via JSON-RPC."""
        data = "0xfeaf968c"  # latestRoundData() selector
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": CHAINLINK_BTC_POLYGON, "data": data}, "latest"],
            "id": 1,
        }
        async with self._session.post(
            self._rpc_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as r:
            resp = await r.json()

        result = resp.get("result", "0x")
        if len(result) < 10:
            return 0.0

        answer_hex = result[66:130]
        answer_int = int(answer_hex, 16)
        if answer_int > (2**255):
            answer_int -= 2**256
        price = answer_int / 1e8
        return price if price > 1000 else 0.0

    # --- Strike price (for settlement) ---

    async def get_strike(self, window_ts: int) -> float:
        """Get BTC/USD strike price for a window."""
        if window_ts in self._strike_cache:
            return self._strike_cache[window_ts]

        # Try Vatic API first
        try:
            price = await self._fetch_vatic(window_ts)
            if price and price > 10000:
                self._strike_cache[window_ts] = price
                return price
        except Exception as e:
            logger.warning("Vatic API failed: %s", e)

        # Fallback: Binance kline open at window_ts
        try:
            price = await self._fetch_binance_at(window_ts)
            if price and price > 10000:
                self._strike_cache[window_ts] = price
                return price
        except Exception as e:
            logger.warning("Binance fallback failed: %s", e)

        # Last resort: current Chainlink price
        if self._chainlink_price > 10000:
            return self._chainlink_price

        raise RuntimeError(f"Could not fetch strike for window {window_ts}")

    async def _fetch_vatic(self, window_ts: int) -> float:
        url = f"{VATIC_API_URL}?feed=BTC/USD&timestamp={window_ts}&chain=polygon"
        async with self._session.get(
            url, timeout=aiohttp.ClientTimeout(total=5)
        ) as r:
            if r.status == 200:
                data = await r.json()
                return float(data.get("price", 0))
        return 0.0

    async def _fetch_binance_at(self, window_ts: int) -> float:
        ts_ms = window_ts * 1000
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime={ts_ms}&limit=1"
        async with self._session.get(url) as r:
            data = await r.json()
            if data:
                return float(data[0][1])
        return 0.0

    # --- Real-time Chainlink data ---

    def get_chainlink_price(self) -> tuple[float, float]:
        """Returns (price, timestamp) of latest Chainlink update."""
        return self._chainlink_price, self._chainlink_ts

    def get_price_velocity(self, lookback_seconds: float = 10) -> float:
        """Rate of price change over last N seconds ($/sec)."""
        if len(self._price_history) < 2:
            return 0.0
        cutoff = time.time() - lookback_seconds
        recent = [(t, p) for t, p in self._price_history if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        dt = recent[-1][0] - recent[0][0]
        dp = recent[-1][1] - recent[0][1]
        return dp / dt if dt > 0 else 0.0

    async def close(self) -> None:
        if self._ws_task:
            self._ws_task.cancel()
        if self._session:
            await self._session.close()
