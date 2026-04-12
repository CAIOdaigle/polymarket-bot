"""Slack Block Kit message formatters."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def trade_blocks(
    market_question: str,
    side: str,
    price: float,
    size_shares: float,
    size_usd: float,
    edge: float,
    half_kelly: float,
    confidence: float,
    order_id: str,
    market_slug: str,
    dry_run: bool = False,
) -> list[dict]:
    prefix = "[DRY RUN] " if dry_run else ""
    emoji = ":chart_with_upwards_trend:" if "YES" in side else ":chart_with_downwards_trend:"

    # Projected profit: if YES wins, each share pays $1.00
    # Profit = (shares * $1.00) - cost, ROI = profit / cost
    proj_profit = (size_shares * 1.0) - size_usd
    proj_roi = (proj_profit / size_usd * 100) if size_usd > 0 else 0

    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{prefix}Trade Executed"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"{emoji} *{market_question}*"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Side:*\n{side}"},
                {"type": "mrkdwn", "text": f"*Price:*\n${price:.4f}"},
                {"type": "mrkdwn", "text": f"*Size:*\n${size_usd:.2f}"},
                {"type": "mrkdwn", "text": f"*Edge:*\n{edge:.2%}"},
                {"type": "mrkdwn", "text": f"*Proj. Profit:*\n${proj_profit:.2f}"},
                {"type": "mrkdwn", "text": f"*Proj. ROI:*\n{proj_roi:.0f}%"},
            ],
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"Confidence: {confidence:.2f} | Half-Kelly: {half_kelly:.4f} | "
                        f"Order: {order_id[:12]} | "
                        f"<https://polymarket.com/event/{market_slug}|View Market>"
                    ),
                },
            ],
        },
        {"type": "divider"},
    ]


def error_blocks(error_msg: str, context: str) -> list[dict]:
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Bot Error"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Context:* {context}\n```{error_msg}```"},
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                },
            ],
        },
    ]


def startup_blocks(market_count: int, bankroll: float, dry_run: bool) -> list[dict]:
    mode = "DRY RUN" if dry_run else "LIVE"
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Bot Started ({mode})"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Markets:*\n{market_count}"},
                {"type": "mrkdwn", "text": f"*Bankroll:*\n${bankroll:,.2f}"},
                {"type": "mrkdwn", "text": f"*Mode:*\n{mode}"},
                {
                    "type": "mrkdwn",
                    "text": f"*Time:*\n{datetime.now(timezone.utc).strftime('%H:%M UTC')}",
                },
            ],
        },
    ]


def exit_blocks(
    market_question: str,
    reason: str,
    entry_price: float,
    exit_price: float,
    size_shares: float,
    pnl_pct: float,
    realized_pnl: float,
    edge_at_exit: float,
    confidence: float,
    market_slug: str,
    dry_run: bool = False,
) -> list[dict]:
    prefix = "[DRY RUN] " if dry_run else ""
    pnl_emoji = ":white_check_mark:" if realized_pnl >= 0 else ":x:"
    reason_display = reason.replace("_", " ").title()

    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{prefix}Position Closed"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"{pnl_emoji} *{market_question}*"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Reason:*\n{reason_display}"},
                {"type": "mrkdwn", "text": f"*Entry:*\n${entry_price:.4f}"},
                {"type": "mrkdwn", "text": f"*Exit:*\n${exit_price:.4f}"},
                {"type": "mrkdwn", "text": f"*Shares:*\n{size_shares:.2f}"},
                {"type": "mrkdwn", "text": f"*PnL:*\n{pnl_pct:.1%} (${realized_pnl:.2f})"},
                {"type": "mrkdwn", "text": f"*Edge@Exit:*\n{edge_at_exit:.4f}"},
            ],
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"Confidence: {confidence:.2f} | "
                        f"<https://polymarket.com/event/{market_slug}|View Market>"
                    ),
                },
            ],
        },
        {"type": "divider"},
    ]


def daily_summary_blocks(
    positions_count: int,
    total_deployed: float,
    daily_pnl: float,
    trades_today: int,
    bankroll: float,
) -> list[dict]:
    pnl_str = f"+${daily_pnl:.2f}" if daily_pnl >= 0 else f"-${abs(daily_pnl):.2f}"
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Daily Summary"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Positions:*\n{positions_count}"},
                {"type": "mrkdwn", "text": f"*Deployed:*\n${total_deployed:,.2f}"},
                {"type": "mrkdwn", "text": f"*Daily PnL:*\n{pnl_str}"},
                {"type": "mrkdwn", "text": f"*Trades:*\n{trades_today}"},
                {"type": "mrkdwn", "text": f"*Bankroll:*\n${bankroll:,.2f}"},
            ],
        },
    ]
