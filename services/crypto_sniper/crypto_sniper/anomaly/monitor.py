"""Anomaly monitor — runs the detector battery, ranks hypotheses, writes
incident reports, and drives the kill switch.

Responsibilities:
  1. Pull recent trade window from bot.db
  2. Run every detector in detectors.ALL_DETECTORS
  3. If anything fires at warn/halt level, rank hypotheses and write incident
     to /app/data/anomaly_log.jsonl + /app/data/anomaly_latest.json
  4. On halt-severity, touch /app/data/halt.flag (runner checks before trades)

The runner calls run_anomaly_check() periodically (every N trades / every
M minutes). Read-only against the DB and file-based for everything else,
so it can also be invoked from a CLI for ad-hoc reviews.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from crypto_sniper.anomaly.detectors import ALL_DETECTORS, DetectorResult, _asset_of
from crypto_sniper.anomaly.hypotheses import rank_hypotheses

logger = logging.getLogger(__name__)

ANOMALY_LOG = "anomaly_log.jsonl"
ANOMALY_LATEST = "anomaly_latest.json"
HALT_FLAG = "halt.flag"


def _load_recent_trades(db_path: Path, limit: int = 100) -> list[dict]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT order_id, price, size, pnl_usd, outcome, confidence,
                      COALESCE(spot_open, btc_open)  AS spot_open,
                      COALESCE(spot_close, btc_close) AS spot_close,
                      placed_at
               FROM orders
               WHERE order_id LIKE 'sniper-%'
                 AND estimated_price IS NOT NULL
               ORDER BY placed_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["asset"] = _asset_of(d["order_id"])
        out.append(d)
    return out


def run_anomaly_check(
    data_dir: Path,
    db_path: Path | None = None,
    window: int = 100,
) -> dict:
    """Run every detector, rank hypotheses, persist results.

    Returns the incident report dict, regardless of whether anything fired.
    On halt-severity firing, writes the halt flag (runner checks before trades).
    """
    db_path = db_path or (data_dir / "bot.db")
    trades = _load_recent_trades(db_path, limit=window)

    detector_results: list[DetectorResult] = []
    for fn in ALL_DETECTORS:
        try:
            r = fn(trades)
        except Exception as e:
            logger.warning("detector %s raised: %s", fn.__name__, e)
            continue
        detector_results.append(r)

    fired = [r for r in detector_results if r.fired]
    halt_fired = any(r.severity == "halt" for r in fired)
    warn_fired = any(r.severity == "warn" for r in fired)

    hypotheses = rank_hypotheses(fired, top_k=3) if fired else []

    incident = {
        "ts": time.time(),
        "ts_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_size": len(trades),
        "fired": [asdict(r) for r in fired],
        "all_results": [asdict(r) for r in detector_results],
        "hypotheses": [asdict(h) for h in hypotheses],
        "halt": halt_fired,
        "severity": "halt" if halt_fired else ("warn" if warn_fired else "info"),
    }

    # Always write latest snapshot (dashboard reads this)
    try:
        latest = data_dir / ANOMALY_LATEST
        tmp = latest.with_suffix(".tmp")
        tmp.write_text(json.dumps(incident, indent=2, default=str))
        tmp.replace(latest)
    except Exception:
        logger.exception("failed to write anomaly_latest.json")

    # Append to the incident log only when something fires
    if fired:
        try:
            with (data_dir / ANOMALY_LOG).open("a") as f:
                f.write(json.dumps(incident, default=str) + "\n")
        except Exception:
            logger.exception("failed to append anomaly_log.jsonl")

        log_fn = logger.error if halt_fired else logger.warning
        for d in fired:
            log_fn("ANOMALY %s [%s]: %s", d.name, d.severity, d.message)
        for h in hypotheses:
            log_fn("  -> hypothesis (%d/100): %s — %s",
                   h.score, h.name, h.recommended_fix)

    # Kill switch
    flag = data_dir / HALT_FLAG
    if halt_fired and not flag.exists():
        flag.write_text(json.dumps({
            "tripped_at": incident["ts_iso"],
            "halt_detectors": [r.name for r in fired if r.severity == "halt"],
            "top_hypothesis": hypotheses[0].name if hypotheses else None,
            "clear_with": "delete this file (rm /app/data/halt.flag)",
        }, indent=2))
        logger.error("HALT FLAG SET — runner will stop placing trades. "
                     "Review %s and clear the flag to resume.", flag)

    return incident


def is_halted(data_dir: Path) -> bool:
    """Cheap check the runner can call before each trade."""
    return (data_dir / HALT_FLAG).exists()
