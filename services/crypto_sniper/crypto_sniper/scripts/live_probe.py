"""ONE-SHOT live $1 FOK probe.

The premise: every dry-run trade we've recorded "filled" at the quoted CLOB
best-ask. The strategy's entire apparent edge depends on those quotes being
real-fillable. Our anomaly system has flagged the same phantom-liquidity
pattern twice now. The only way to definitively answer "is the depth real"
is to submit ONE actual order and see what the exchange does.

This script:
  1. Picks the current 5-min window for the configured asset
  2. Fetches the live ask book for the UP token (default — overridable)
  3. Computes the fill price for ~$1 of shares at quote + $0.02 limit
  4. Submits a single FOK BUY for that size
  5. Prints what happened: filled? at what avg price? killed?

It is INTENTIONALLY NOT integrated with the runner. The runner stays in
dry-run; this script reaches around it and uses the same OrderManager
authentication path, but with dry_run forcibly set to False.

USAGE (from inside the BTC container):
  SNIPER_ASSET=BTC python -m crypto_sniper.scripts.live_probe \
      --direction UP --notional-usd 1.00

Without --execute, the script does a full pre-flight (authenticates, fetches
the book, computes the order) but DOES NOT submit. Add --execute to actually
place the order.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from polymarket_common.config import PolymarketConfig, TradingConfig
from polymarket_common.execution.order_manager import OrderManager, TradeRequest
from polymarket_common.utils.logging_config import setup_logging
from polymarket_common.config import LoggingConfig

from crypto_sniper.config import resolve_slug_prefix, resolve_binance_symbol  # noqa
from crypto_sniper.market.discovery import MarketDiscovery
from crypto_sniper.runner import (
    SLIPPAGE_BUMP, compute_fok_fill, top_n_depth_usd, WINDOW_SECONDS,
)


def _current_window_ts() -> int:
    now = int(time.time())
    return now - (now % WINDOW_SECONDS)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Live $1 FOK probe")
    parser.add_argument("--asset", default=os.environ.get("SNIPER_ASSET", "BTC"))
    parser.add_argument("--direction", choices=["UP", "DOWN"], default="UP")
    parser.add_argument("--notional-usd", type=float, default=1.00,
                        help="USD size of the probe order")
    parser.add_argument("--execute", action="store_true",
                        help="Actually submit the order. Without this, dry pre-flight only.")
    parser.add_argument("--max-token-price", type=float, default=0.95,
                        help="Refuse to submit if the quote exceeds this")
    args = parser.parse_args()

    setup_logging(LoggingConfig())
    logger = logging.getLogger("live_probe")

    asset = args.asset.upper()
    slug_prefix = resolve_slug_prefix(asset)
    window_ts = _current_window_ts()
    secs_left = (window_ts + WINDOW_SECONDS) - time.time()

    logger.info("PROBE asset=%s direction=%s notional=$%.2f window_ts=%d secs_left=%.1f",
                asset, args.direction, args.notional_usd, window_ts, secs_left)
    logger.info("Slug prefix: %s", slug_prefix)

    if secs_left < 30:
        logger.warning("Only %.1fs left in window — fill window may be too short", secs_left)
    if secs_left < 5:
        logger.error("Window is essentially closed; aborting")
        return 1

    # ── Fetch market + book ──────────────────────────────────────────────
    discovery = MarketDiscovery(slug_prefix=slug_prefix)
    try:
        market = await discovery.get_market(window_ts)
        if not market:
            logger.error("No market found for slug %s-%d", slug_prefix, window_ts)
            return 1

        token_id = market["up_token"] if args.direction == "UP" else market["down_token"]
        condition_id = market["condition_id"]
        tick_size = float(market.get("tick_size", 0.01))
        neg_risk = bool(market.get("neg_risk", False))

        asks = await discovery.get_ask_book(token_id)
        if not asks:
            logger.error("Ask book empty for token %s — nobody is selling", token_id[:20])
            return 1

        depth_usd = top_n_depth_usd(asks, n=3)
        top_ask = asks[0][0]
        logger.info("Top ask: $%.4f  Top-3 depth: $%.2f  Levels: %s",
                    top_ask, depth_usd, asks[:5])

        if top_ask > args.max_token_price:
            logger.error("Top ask $%.4f exceeds --max-token-price $%.4f — aborting",
                         top_ask, args.max_token_price)
            return 1

        # Compute order size: ~notional_usd worth of shares at the top ask.
        shares_wanted = round(args.notional_usd / top_ask, 2)
        limit_price = min(top_ask + SLIPPAGE_BUMP, 0.99)

        avg_fill, would_fill = compute_fok_fill(asks, shares_wanted, limit_price)
        logger.info(
            "Order plan: BUY %.2f shares @ limit $%.4f (top ask $%.4f, +2c slippage)",
            shares_wanted, limit_price, top_ask,
        )
        logger.info(
            "FOK simulation says: %s, walked-avg fill would be $%.4f",
            "FILL" if would_fill else "KILL", avg_fill or 0,
        )

        if not args.execute:
            logger.info("─── DRY PRE-FLIGHT COMPLETE — re-run with --execute to submit ───")
            return 0

        # ── Authenticate live ───────────────────────────────────────────
        poly_cfg = PolymarketConfig()
        if not poly_cfg.private_key:
            logger.error("POLYMARKET_PRIVATE_KEY env var is empty — cannot submit")
            return 1

        # Force live mode for the OrderManager
        live_trading = TradingConfig(dry_run=False)
        order_mgr = OrderManager(poly_cfg, live_trading)
        await order_mgr.initialize()
        logger.warning("*** LIVE MODE — submitting REAL FOK order for $%.2f ***",
                       args.notional_usd)

        request = TradeRequest(
            condition_id=condition_id,
            token_id=token_id,
            side="BUY",
            price=limit_price,
            size=shares_wanted,
            order_type="FOK",
            edge=0.0,
            kelly_fraction=0.0,
            neg_risk=neg_risk,
            tick_size=tick_size,
        )

        logger.info("Submitting: %s", asdict(request))
        t0 = time.time()
        record = await order_mgr.place_order(request)
        elapsed = time.time() - t0

        result = {
            "asset": asset,
            "direction": args.direction,
            "submitted_at": t0,
            "elapsed_seconds": round(elapsed, 3),
            "limit_price": limit_price,
            "shares_requested": shares_wanted,
            "notional_usd": args.notional_usd,
            "top_ask_at_submit": top_ask,
            "depth_top3_usd": depth_usd,
            "asks_snapshot": asks[:10],
            "order_record": asdict(record),
        }

        # Persist next to the audit trail
        out_path = Path("/app/data") / f"live_probe_{int(t0)}.json"
        try:
            out_path.write_text(json.dumps(result, indent=2, default=str))
        except Exception:
            logger.warning("Could not persist result to %s", out_path)

        logger.warning("─── RESULT ───")
        logger.warning("  status: %s", record.status)
        logger.warning("  order_id: %s", record.order_id)
        logger.warning("  filled_size: %.4f / %.4f requested",
                       record.filled_size, record.size)
        logger.warning("  fill_price submitted: $%.4f", record.price)
        logger.warning("  elapsed: %.2fs", elapsed)
        logger.warning("  full record persisted to %s", out_path)

        return 0
    finally:
        await discovery.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
