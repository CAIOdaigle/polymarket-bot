"""Polymarket Bot Dashboard — lightweight Flask app."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data" / "bot.db"
CSV_PATH = BASE_DIR / "data" / "trades.csv"
CONFIG_PATH = BASE_DIR / "config" / "default.yaml"
PORTFOLIO_PATH = BASE_DIR / "data" / "portfolio.json"

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))


def _get_db():
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _load_config():
    try:
        import yaml

        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def _get_recent_trades(limit: int = 50) -> list[dict]:
    conn = _get_db()
    if not conn:
        return []
    try:
        rows = conn.execute(
            """SELECT order_id, condition_id, token_id, side, price, size,
                      filled_size, status, edge, kelly_fraction, p_hat,
                      b_estimate, confidence, market_question,
                      placed_at, updated_at
               FROM orders ORDER BY placed_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        trades = []
        for r in rows:
            t = dict(r)
            if t.get("placed_at"):
                t["placed_at_fmt"] = datetime.fromtimestamp(
                    t["placed_at"], tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
            else:
                t["placed_at_fmt"] = "—"
            if t.get("edge"):
                t["edge_pct"] = f"{t['edge'] * 100:.1f}%"
            else:
                t["edge_pct"] = "—"
            if t.get("kelly_fraction"):
                t["kelly_pct"] = f"{t['kelly_fraction'] * 100:.1f}%"
            else:
                t["kelly_pct"] = "—"
            if t.get("price"):
                t["price_fmt"] = f"${t['price']:.4f}"
            else:
                t["price_fmt"] = "—"

            # Human-readable side display
            raw_side = (t.get("side") or "").upper()
            if raw_side == "BUY_YES_WEATHER":
                t["side_display"] = "WX YES ▲"
                t["direction"] = "weather"
            elif raw_side == "BUY_YES":
                t["side_display"] = "YES ▲"
                t["direction"] = "buy"
            elif raw_side == "BUY_NO":
                t["side_display"] = "NO ▼"
                t["direction"] = "sell"
            elif raw_side.startswith("SELL"):
                t["side_display"] = "EXIT"
                t["direction"] = "exit"
            else:
                t["side_display"] = raw_side or "—"
                t["direction"] = "unknown"

            # Truncate market question for table display
            mq = t.get("market_question") or ""
            t["market_short"] = (mq[:60] + "…") if len(mq) > 60 else (mq or "—")

            # Confidence formatting
            conf = t.get("confidence")
            t["conf_fmt"] = f"{conf:.2f}" if conf else "—"

            # Size field is in SHARES, compute USD cost for display
            price = t.get("price") or 0
            shares = t.get("size") or 0
            if price > 0 and shares > 0:
                cost_usd = shares * price
                t["size_display"] = f"${cost_usd:.2f}"
            else:
                t["size_display"] = "—"

            # Actual P&L: only computed for EXIT trades
            t["actual_pnl"] = "—"
            t["actual_roi"] = "—"
            t["pnl_class"] = ""

            trades.append(t)

        # Second pass: compute actual P&L for EXIT trades by looking up entry prices
        _enrich_exit_pnl(conn, trades)

        return trades
    except Exception:
        return []
    finally:
        conn.close()


def _enrich_exit_pnl(conn, trades: list[dict]) -> None:
    """For each EXIT trade, find its entry VWAP and compute actual P&L."""
    # Cache: condition_id → VWAP entry price (from all BUY orders for that condition)
    entry_cache: dict[str, float] = {}

    for t in trades:
        raw_side = (t.get("side") or "").upper()
        if not raw_side.startswith("SELL"):
            continue

        cond_id = t.get("condition_id")
        if not cond_id:
            continue

        # Compute VWAP entry price across all BUY orders for this condition
        if cond_id not in entry_cache:
            rows = conn.execute(
                """SELECT price, size FROM orders
                   WHERE condition_id = ? AND side NOT LIKE 'SELL%'
                     AND price > 0 AND size > 0
                   ORDER BY placed_at""",
                (cond_id,),
            ).fetchall()
            total_cost = sum(r["price"] * r["size"] for r in rows)
            total_shares = sum(r["size"] for r in rows)
            entry_cache[cond_id] = (total_cost / total_shares) if total_shares > 0 else 0

        entry_price = entry_cache[cond_id]
        exit_price = t.get("price") or 0
        shares = t.get("size") or 0

        if entry_price > 0 and exit_price > 0 and shares > 0:
            pnl_usd = (exit_price - entry_price) * shares
            cost_basis = entry_price * shares
            roi_pct = (pnl_usd / cost_basis * 100) if cost_basis > 0 else 0

            t["actual_pnl"] = f"${pnl_usd:+.2f}"
            t["actual_roi"] = f"{roi_pct:+.1f}%"
            t["pnl_class"] = "green" if pnl_usd >= 0 else "red"


def _get_daily_pnl() -> list[dict]:
    conn = _get_db()
    if not conn:
        return []
    try:
        rows = conn.execute(
            "SELECT date, realized_pnl, trades FROM daily_pnl ORDER BY date DESC LIMIT 30"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _get_trade_stats() -> dict:
    conn = _get_db()
    if not conn:
        return {"total_trades": 0, "dry_run_trades": 0, "live_trades": 0, "avg_edge": 0}
    try:
        row = conn.execute(
            """SELECT COUNT(*) as total,
                      SUM(CASE WHEN status='dry_run' THEN 1 ELSE 0 END) as dry_runs,
                      SUM(CASE WHEN status NOT IN ('dry_run','failed','cancelled') THEN 1 ELSE 0 END) as live,
                      AVG(edge) as avg_edge
               FROM orders"""
        ).fetchone()
        return {
            "total_trades": row["total"] or 0,
            "dry_run_trades": row["dry_runs"] or 0,
            "live_trades": row["live"] or 0,
            "avg_edge": row["avg_edge"] or 0,
        }
    except Exception:
        return {"total_trades": 0, "dry_run_trades": 0, "live_trades": 0, "avg_edge": 0}
    finally:
        conn.close()


def _get_portfolio() -> dict:
    """Read live portfolio snapshot written by the bot."""
    try:
        if PORTFOLIO_PATH.exists():
            data = json.loads(PORTFOLIO_PATH.read_text())
            return data
    except Exception:
        app.logger.exception("Failed to load portfolio snapshot from %s", PORTFOLIO_PATH)
    return {
        "usdc_balance": 0,
        "deployed_cost": 0,
        "market_value": 0,
        "portfolio_value": 0,
        "open_positions": 0,
        "max_position_usd": 0,
        "daily_loss_limit_usd": 0,
        "updated_at": None,
    }


def _get_csv_trade_count() -> int:
    if not CSV_PATH.exists():
        return 0
    try:
        with open(CSV_PATH) as f:
            return sum(1 for _ in csv.reader(f)) - 1  # minus header
    except Exception:
        return 0


@app.route("/")
def dashboard():
    config = _load_config()
    trading = config.get("trading", {})
    trades = _get_recent_trades(50)
    stats = _get_trade_stats()
    daily_pnl = _get_daily_pnl()
    csv_count = _get_csv_trade_count()
    portfolio = _get_portfolio()

    # Prepare chart data
    pnl_dates = [d["date"] for d in reversed(daily_pnl)]
    pnl_values = [d["realized_pnl"] for d in reversed(daily_pnl)]

    is_live = not trading.get("dry_run", True)

    return render_template(
        "dashboard.html",
        config=trading,
        trades=trades,
        stats=stats,
        daily_pnl=daily_pnl,
        csv_count=csv_count,
        pnl_dates=pnl_dates,
        pnl_values=pnl_values,
        is_live=is_live,
        portfolio=portfolio,
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


@app.route("/api/trades")
def api_trades():
    """JSON endpoint for trades (for AJAX refresh)."""
    from flask import jsonify

    trades = _get_recent_trades(50)
    return jsonify(trades)


@app.route("/api/stats")
def api_stats():
    from flask import jsonify

    stats = _get_trade_stats()
    return jsonify(stats)


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 5050))
    print(f"Dashboard running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "").lower() == "true")
