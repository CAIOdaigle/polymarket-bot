"""Crypto Sniper Dashboard — lightweight Flask app (port 5051)."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Writable data dir (shared volume in Docker)
DATA_DIR = Path(os.environ.get("SNIPER_DATA_DIR", "/app/data"))
DB_PATH = DATA_DIR / "bot.db"
CALIBRATION_PATH = DATA_DIR / "calibration.json"

# Assets supported by the multi-asset dashboard. Each asset writes its own
# sniper_stats_{asset}.json snapshot from the runner process.
SUPPORTED_ASSETS = ["BTC", "ETH", "SOL"]

# Mirror of ConfidenceCalibrator.MIN_SAMPLES_PER_BUCKET from sizing/calibration.py.
# Kept as a local constant to keep the dashboard package dependency-free
# (doesn't need to import from sizing/ which pulls in more modules).
MIN_SAMPLES_PER_BUCKET = 20


def _iso_utc(ts: float | None) -> str | None:
    """Epoch seconds -> ISO 8601 UTC string with trailing 'Z'.

    The client-side JS uses this with Intl.DateTimeFormat to render each
    timestamp in the user's selected timezone. Returning ISO strings (not
    pre-formatted) is what makes the TZ picker work without server restarts.
    """
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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
        # Only show trades captured with the NEW real-price pipeline.
        # Old contaminated rows (pre-fix) have estimated_price = NULL and
        # used fake token prices. They're intentionally hidden.
        rows = conn.execute(
            """SELECT order_id, condition_id, token_id, side, price, size,
                      status, edge, p_hat, confidence, market_question,
                      placed_at,
                      COALESCE(spot_open, btc_open) AS spot_open,
                      COALESCE(spot_close, btc_close) AS spot_close,
                      outcome, pnl_usd,
                      exit_price, estimated_price
               FROM orders
               WHERE order_id LIKE 'sniper-%'
                 AND estimated_price IS NOT NULL
               ORDER BY placed_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        trades = []
        for r in rows:
            t = dict(r)
            # Keep the pre-formatted UTC string as a fallback, but also emit an
            # ISO 8601 UTC string that the client-side TZ picker re-renders
            # in the user's chosen timezone.
            t["placed_at_fmt"] = (
                datetime.fromtimestamp(t["placed_at"], tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if t.get("placed_at")
                else "—"
            )
            t["iso_ts"] = _iso_utc(t.get("placed_at"))

            # Extract asset + strategy from order_id.
            # New format:    "sniper-<asset>-<window_ts>-<strategy>"  (4+ parts)
            # Legacy format: "sniper-<window_ts>-<strategy>"          (3 parts)
            oid = t.get("order_id") or ""
            parts = oid.split("-") if oid.startswith("sniper-") else []
            if len(parts) >= 4 and not parts[1].isdigit():
                t["asset"] = parts[1].upper()
                t["strategy"] = "-".join(parts[3:]) or "unknown"
            elif len(parts) >= 3:
                # legacy BTC-only rows
                t["asset"] = "BTC"
                t["strategy"] = "-".join(parts[2:]) or "unknown"
            else:
                t["asset"] = "?"
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

            # Entry = REAL best-ask from Polymarket CLOB
            # Estimated = what the old model would have said (for comparison)
            t["entry_fmt"] = f"${t['price']:.3f}" if t.get("price") else "—"
            est = t.get("estimated_price")
            t["est_fmt"] = f"${est:.3f}" if est else "—"
            if est and t.get("price"):
                t["slippage_fmt"] = f"{((t['price'] - est) / est * 100):+.1f}%"
            else:
                t["slippage_fmt"] = "—"
            exit_p = t.get("exit_price")
            if exit_p is not None:
                t["exit_fmt"] = f"${exit_p:.2f}"
            else:
                t["exit_fmt"] = "—"

            # Spot price at window open/close (the underlying asset's price,
            # NOT always BTC — misleadingly named in pre-fix schema).
            so = t.get("spot_open")
            sc = t.get("spot_close")
            t["spot_open_fmt"] = f"${so:,.2f}" if so else "—"
            t["spot_close_fmt"] = f"${sc:,.2f}" if sc else "—"
            if so and sc:
                move = sc - so
                t["spot_move_fmt"] = f"{'+' if move >= 0 else ''}{move:,.2f}"
            else:
                t["spot_move_fmt"] = "—"
            # Back-compat keys (old template references)
            t["btc_open_fmt"] = t["spot_open_fmt"]
            t["btc_close_fmt"] = t["spot_close_fmt"]
            t["btc_move_fmt"] = t["spot_move_fmt"]

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


def _get_all_stats() -> list[dict]:
    """Return a per-asset stats snapshot for every runner that is writing a
    sniper_stats_{asset}.json file. Unknown assets are silently ignored.
    """
    out: list[dict] = []
    for asset in SUPPORTED_ASSETS:
        path = DATA_DIR / f"sniper_stats_{asset.lower()}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            data.setdefault("asset", asset)
            out.append(data)
        except Exception:
            continue

    # Legacy single-file fallback (pre-multi-asset)
    if not out:
        legacy = DATA_DIR / "sniper_stats.json"
        if legacy.exists():
            try:
                data = json.loads(legacy.read_text())
                data.setdefault("asset", "BTC")
                out.append(data)
            except Exception:
                pass
    return out


def _aggregate_stats(per_asset: list[dict]) -> dict:
    """Sum-up view across all assets for the header cards."""
    if not per_asset:
        return {
            "bankroll": 0.0, "starting_bankroll": 0.0, "pnl": 0.0,
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "dry_run": True, "asset_count": 0,
        }
    total_bankroll = sum(s.get("bankroll", 0) for s in per_asset)
    starting = sum(s.get("starting_bankroll", 0) for s in per_asset)
    wins = sum(s.get("wins", 0) for s in per_asset)
    losses = sum(s.get("losses", 0) for s in per_asset)
    trades = sum(s.get("trades", 0) for s in per_asset)
    total = wins + losses
    return {
        "bankroll": round(total_bankroll, 2),
        "starting_bankroll": round(starting, 2),
        "pnl": round(total_bankroll - starting, 2),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / total) if total else 0.0,
        "dry_run": any(s.get("dry_run", True) for s in per_asset),
        "asset_count": len(per_asset),
    }


def _get_trade_count() -> int:
    """Count only the NEW real-price trades."""
    conn = _get_db()
    if not conn:
        return 0
    try:
        row = conn.execute(
            """SELECT COUNT(*) as n FROM orders
               WHERE order_id LIKE 'sniper-%' AND estimated_price IS NOT NULL"""
        ).fetchone()
        return row["n"] or 0
    except Exception:
        return 0
    finally:
        conn.close()


def _get_calibration() -> dict:
    """Load calibration.json written by sizing/calibration.py.

    Annotates each bucket with:
      - `gap`: empirical - mean(stated_min, stated_max)
      - `status`: 'locked' | 'active' | 'unprofitable'
        - 'locked'        -> n < MIN_SAMPLES_PER_BUCKET (Kelly capped at 0.55)
        - 'active'        -> n >= MIN and empirical >= 0.50
        - 'unprofitable'  -> n >= MIN and empirical <  0.50 (Kelly returns 0)

    Returns an empty skeleton if the file doesn't exist yet (first run).
    """
    skeleton = {"generated_at_iso": None, "total_trades": 0, "by_asset": {}}
    if not CALIBRATION_PATH.exists():
        return skeleton
    try:
        raw = json.loads(CALIBRATION_PATH.read_text())
    except Exception:
        app.logger.exception("Failed to parse calibration.json")
        return skeleton

    out: dict = {
        "generated_at": raw.get("generated_at"),
        "generated_at_iso": _iso_utc(raw.get("generated_at")),
        "total_trades": raw.get("total_trades", 0),
        "by_asset": {},
    }
    for asset, buckets in (raw.get("by_asset") or {}).items():
        enriched = []
        for b in buckets:
            n = int(b.get("n", 0))
            empirical = float(b.get("empirical", 0.0))
            stated_mid = (float(b.get("stated_min", 0)) + float(b.get("stated_max", 0))) / 2
            gap = empirical - stated_mid
            if n < MIN_SAMPLES_PER_BUCKET:
                status = "locked"
            elif empirical >= 0.50:
                status = "active"
            else:
                status = "unprofitable"
            enriched.append({
                "stated_min": b.get("stated_min"),
                "stated_max": b.get("stated_max"),
                "empirical": empirical,
                "gap": round(gap, 4),
                "n": n,
                "status": status,
            })
        out["by_asset"][asset] = enriched
    return out


def _get_pnl_series(days: int = 7) -> dict[str, list[dict]]:
    """Return cumulative PnL timeseries per asset for the last `days` days.

    Structure: {'BTC': [{'t': iso, 'cumulative_pnl': 12.34}, ...], 'ETH': [...]}
    One data point per resolved trade, ordered chronologically.
    """
    out: dict[str, list[dict]] = {}
    conn = _get_db()
    if not conn:
        return out
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
        rows = conn.execute(
            """SELECT order_id, placed_at, pnl_usd
               FROM orders
               WHERE order_id LIKE 'sniper-%'
                 AND estimated_price IS NOT NULL
                 AND pnl_usd IS NOT NULL
                 AND placed_at >= ?
               ORDER BY placed_at ASC""",
            (cutoff,),
        ).fetchall()
    except Exception:
        app.logger.exception("Failed to load pnl series")
        return out
    finally:
        conn.close()

    running: dict[str, float] = {}
    for r in rows:
        oid = r["order_id"] or ""
        parts = oid.split("-")
        if len(parts) >= 4 and not parts[1].isdigit():
            asset = parts[1].upper()
        else:
            asset = "BTC"  # legacy rows
        running[asset] = running.get(asset, 0.0) + float(r["pnl_usd"] or 0)
        out.setdefault(asset, []).append({
            "t": _iso_utc(r["placed_at"]),
            "cumulative_pnl": round(running[asset], 4),
        })
    return out


def _get_legacy_count() -> int:
    """Count the old, contaminated, pre-fix trades (shown only in banner)."""
    conn = _get_db()
    if not conn:
        return 0
    try:
        row = conn.execute(
            """SELECT COUNT(*) as n FROM orders
               WHERE order_id LIKE 'sniper-%' AND estimated_price IS NULL"""
        ).fetchone()
        return row["n"] or 0
    except Exception:
        return 0
    finally:
        conn.close()


@app.route("/")
def dashboard():
    trades = _get_recent_trades(50)
    per_asset_stats = _get_all_stats()
    totals = _aggregate_stats(per_asset_stats)
    trade_count = _get_trade_count()
    legacy_count = _get_legacy_count()
    calibration = _get_calibration()
    now = datetime.now(timezone.utc)

    return render_template(
        "dashboard.html",
        trades=trades,
        stats=totals,
        per_asset=per_asset_stats,
        trade_count=trade_count,
        legacy_count=legacy_count,
        calibration=calibration,
        is_live=not totals.get("dry_run", True),
        now=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        now_iso=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        min_samples=MIN_SAMPLES_PER_BUCKET,
    )


@app.route("/api/trades")
def api_trades():
    return jsonify(_get_recent_trades(50))


@app.route("/api/stats")
def api_stats():
    return jsonify({
        "totals": _aggregate_stats(_get_all_stats()),
        "per_asset": _get_all_stats(),
    })


@app.route("/api/calibration")
def api_calibration():
    return jsonify(_get_calibration())


@app.route("/api/pnl_series")
def api_pnl_series():
    """Cumulative PnL timeseries per asset.

    Query params:
      days (int, default 7) — clamped to [1, 90]
    """
    try:
        days = max(1, min(90, int(request.args.get("days", "7"))))
    except ValueError:
        days = 7
    return jsonify({"days": days, "series": _get_pnl_series(days=days)})


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 5051))
    print(f"Crypto Sniper Dashboard running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
