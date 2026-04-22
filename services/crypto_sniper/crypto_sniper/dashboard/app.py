"""Crypto Sniper Dashboard — lightweight Flask app (port 5051)."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template

# Writable data dir (shared volume in Docker)
DATA_DIR = Path(os.environ.get("SNIPER_DATA_DIR", "/app/data"))
DB_PATH = DATA_DIR / "bot.db"
STATS_PATH = DATA_DIR / "sniper_stats.json"

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))


def _get_db():
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _get_recent_trades(limit: int = 50) -> list[dict]:
    """Return only crypto-sniper trades (filtered by order_id prefix)."""
    conn = _get_db()
    if not conn:
        return []
    try:
        rows = conn.execute(
            """SELECT order_id, condition_id, token_id, side, price, size,
                      status, edge, p_hat, confidence, market_question,
                      placed_at, btc_open, btc_close, outcome, pnl_usd, exit_price
               FROM orders
               WHERE order_id LIKE 'sniper-%' OR condition_id LIKE 'btc-%'
               ORDER BY placed_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        trades = []
        for r in rows:
            t = dict(r)
            t["placed_at_fmt"] = (
                datetime.fromtimestamp(t["placed_at"], tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if t.get("placed_at")
                else "—"
            )

            # Extract strategy from order_id: "sniper-<window_ts>-<strategy>"
            oid = t.get("order_id") or ""
            if oid.startswith("sniper-"):
                parts = oid.split("-", 2)
                t["strategy"] = parts[2] if len(parts) > 2 else "unknown"
            else:
                t["strategy"] = "—"

            # Direction display
            raw_side = (t.get("side") or "").upper()
            if raw_side == "BUY_UP":
                t["side_display"] = "UP ▲"
                t["direction"] = "buy"
            elif raw_side == "BUY_DOWN":
                t["side_display"] = "DOWN ▼"
                t["direction"] = "sell"
            else:
                t["side_display"] = raw_side or "—"
                t["direction"] = "unknown"

            t["price_fmt"] = f"${t['price']:.3f}" if t.get("price") else "—"
            t["edge_pct"] = f"{t['edge'] * 100:.1f}%" if t.get("edge") else "—"
            conf = t.get("confidence")
            t["conf_fmt"] = f"{conf * 100:.0f}%" if conf else "—"

            # Bet size in USD = shares * price
            price = t.get("price") or 0
            shares = t.get("size") or 0
            t["size_display"] = f"${shares * price:.2f}" if price > 0 and shares > 0 else "—"

            mq = t.get("market_question") or ""
            t["market_short"] = (mq[:60] + "…") if len(mq) > 60 else (mq or "—")

            # Entry/exit token prices (binary market: resolves $1 win / $0 loss)
            t["entry_fmt"] = f"${t['price']:.3f}" if t.get("price") else "—"
            exit_p = t.get("exit_price")
            if exit_p is not None:
                t["exit_fmt"] = f"${exit_p:.2f}"
            else:
                t["exit_fmt"] = "—"

            # BTC reference prices
            btc_o = t.get("btc_open")
            btc_c = t.get("btc_close")
            t["btc_open_fmt"] = f"${btc_o:,.2f}" if btc_o else "—"
            t["btc_close_fmt"] = f"${btc_c:,.2f}" if btc_c else "—"
            if btc_o and btc_c:
                move = btc_c - btc_o
                t["btc_move_fmt"] = f"{'+' if move >= 0 else ''}{move:,.2f}"
            else:
                t["btc_move_fmt"] = "—"

            # Outcome & PnL
            outcome = t.get("outcome") or ""
            t["outcome"] = outcome
            pnl = t.get("pnl_usd")
            if pnl is not None:
                t["pnl_fmt"] = f"{'+' if pnl >= 0 else ''}${pnl:.2f}"
                t["pnl_class"] = "green" if pnl >= 0 else "red"
            else:
                t["pnl_fmt"] = "—"
                t["pnl_class"] = ""

            trades.append(t)
        return trades
    except Exception:
        app.logger.exception("Failed to load trades")
        return []
    finally:
        conn.close()


def _get_stats() -> dict:
    """Read runtime stats snapshot written by StrategyRunner."""
    if STATS_PATH.exists():
        try:
            return json.loads(STATS_PATH.read_text())
        except Exception:
            pass
    return {
        "bankroll": 0.0,
        "starting_bankroll": 0.0,
        "pnl": 0.0,
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "btc_price": None,
        "dry_run": True,
        "updated_at": None,
    }


def _get_trade_count() -> int:
    conn = _get_db()
    if not conn:
        return 0
    try:
        row = conn.execute(
            "SELECT COUNT(*) as n FROM orders WHERE order_id LIKE 'sniper-%'"
        ).fetchone()
        return row["n"] or 0
    except Exception:
        return 0
    finally:
        conn.close()


@app.route("/")
def dashboard():
    trades = _get_recent_trades(50)
    stats = _get_stats()
    trade_count = _get_trade_count()

    return render_template(
        "dashboard.html",
        trades=trades,
        stats=stats,
        trade_count=trade_count,
        is_live=not stats.get("dry_run", True),
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


@app.route("/api/trades")
def api_trades():
    return jsonify(_get_recent_trades(50))


@app.route("/api/stats")
def api_stats():
    return jsonify(_get_stats())


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 5051))
    print(f"Crypto Sniper Dashboard running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
