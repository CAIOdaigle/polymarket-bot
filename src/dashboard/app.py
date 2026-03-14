"""Polymarket Bot Dashboard — lightweight Flask app."""

from __future__ import annotations

import csv
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data" / "bot.db"
CSV_PATH = BASE_DIR / "data" / "trades.csv"
CONFIG_PATH = BASE_DIR / "config" / "default.yaml"

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

            # Truncate market question for table display
            mq = t.get("market_question") or ""
            t["market_short"] = (mq[:60] + "…") if len(mq) > 60 else (mq or "—")

            # Confidence formatting
            conf = t.get("confidence")
            t["conf_fmt"] = f"{conf:.2f}" if conf else "—"

            # Projected profit: if correct, payout is $1/share - cost
            price = t.get("price") or 0
            size_usd = t.get("size") or 0
            side = t.get("side") or ""
            if price > 0 and price < 1 and size_usd > 0:
                shares = size_usd / price
                profit_usd = shares * (1.0 - price)  # payout - cost = shares*(1-p)
                profit_pct = (1.0 - price) / price * 100  # ROI %
                t["proj_profit_usd"] = f"${profit_usd:.2f}"
                t["proj_profit_pct"] = f"{profit_pct:.0f}%"
            else:
                t["proj_profit_usd"] = "—"
                t["proj_profit_pct"] = "—"

            trades.append(t)
        return trades
    except Exception:
        return []
    finally:
        conn.close()


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

    # Prepare chart data
    pnl_dates = [d["date"] for d in reversed(daily_pnl)]
    pnl_values = [d["realized_pnl"] for d in reversed(daily_pnl)]

    return render_template(
        "dashboard.html",
        config=trading,
        trades=trades,
        stats=stats,
        daily_pnl=daily_pnl,
        csv_count=csv_count,
        pnl_dates=pnl_dates,
        pnl_values=pnl_values,
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
    app.run(host="0.0.0.0", port=port, debug=True)
