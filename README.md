# polymarket-bot

Automated trading bot for [Polymarket](https://polymarket.com/) prediction markets.
Uses Bayesian signal fusion, LMSR order-book modelling, and Half-Kelly position sizing
to find and trade edges on binary-outcome markets, including a specialised weather-market
pipeline backed by NOAA forecasts.

## Quick start

### 1. Environment variables

```bash
cp .env.example .env
# then fill in your values
```

| Variable | Required | Description |
|---|---|---|
| `POLYMARKET_PRIVATE_KEY` | yes | Polygon wallet private key (with USDC) |
| `POLYMARKET_FUNDER_ADDRESS` | no | Funder/proxy address (leave blank for self-funded) |
| `POLYMARKET_CHAIN_ID` | no | Default `137` (Polygon mainnet) |
| `CLOB_HOST` | no | CLOB API endpoint (default: `https://clob.polymarket.com`) |
| `GAMMA_HOST` | no | Gamma API endpoint (default: `https://gamma-api.polymarket.com`) |
| `SLACK_WEBHOOK_URL` | no | Slack incoming-webhook for trade/error notifications |
| `BOT_ENV` | no | `development` or `production` |

### 2. Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m src.main          # starts the bot
python -m src.dashboard.app # starts the status dashboard on :5050 (optional)
```

Set `dry_run: true` in `config/default.yaml` under `trading:` to paper-trade without
placing real orders.

### 3. Run with Docker

```bash
docker-compose up -d        # builds image, starts bot + dashboard
docker-compose logs -f      # tail logs
```

Data (SQLite) and logs are stored in named Docker volumes (`bot-data`, `bot-logs`) and
persist across container rebuilds.

### 4. Run tests

```bash
pytest                      # all tests
pytest -x -q                # stop on first failure, quiet output
pytest --cov=src            # with coverage
```

## Architecture

```
polymarket-bot/
  config/
    default.yaml          # all tunable parameters (Kelly, signals, exit, weather, ...)
  src/
    main.py               # bot loop: scan -> analyse -> trade -> exit sweep
    config.py             # Pydantic settings loaded from YAML + env
    analysis/             # Bayesian engine, signal registry, LMSR fitting
    execution/            # OrderManager, ExitManager, PositionTracker, auth
    feed/                 # WebSocket + REST order-book feed
    market/               # Market scanner (Gamma API)
    persistence/          # SQLite state store (positions, trades, daily P&L)
    signals/              # Signal generators (order-book imbalance, volume, whale, ...)
    sizing/               # Kelly criterion position sizer
    weather/              # Weather-market pipeline (NOAA forecasts, edge calc)
    dashboard/            # Flask status dashboard (:5050)
    notifications/        # Slack alerting
    utils/                # Rate limiter, helpers
  scripts/
    check_balances.py     # Query on-chain USDC balance
    setup_wallet.py       # Wallet setup helper
  tests/                  # pytest suite (Bayesian, exits, Kelly, LMSR, signals, weather)
  Dockerfile
  docker-compose.yml
  start.sh                # Entrypoint: dashboard (background) + bot (foreground)
```

### Decision pipeline (per market, every ~30 s)

1. **Scan** active markets via the Gamma API; filter by volume, liquidity, time-to-expiry.
2. **Fit LMSR** to the live order book to estimate the liquidity parameter *b*.
3. **Fuse signals** (order-book imbalance, LMSR deviation, volume momentum, whale activity,
   cross-market inconsistency, related-market divergence) through a Bayesian engine to
   produce a posterior probability estimate.
4. **Size** the position with Half-Kelly, clamped by per-market and portfolio-wide exposure
   caps.
5. **Enter** via the Polymarket CLOB API (GTC or FOK orders).
6. **Exit sweep** runs continuously: stop-loss, take-profit, edge-convergence, emergency
   floor, and time backstop.

### Weather pipeline

Runs in parallel every 5 minutes:

1. Discover temperature-bucket markets from the scanner output.
2. Fetch NOAA point forecasts for each city/date.
3. Fit a normal distribution to the forecast and compute bucket probabilities.
4. Compare to market ask prices, apply quarter-Kelly sizing, and trade edges > 8 %.

## Key risks and assumptions

- **Market risk.** The bot trades real money on Polymarket. Prediction-market prices can
  move against you. Past edges do not guarantee future profitability.
- **Model risk.** The Bayesian posterior and LMSR fit are approximations. Thin or
  manipulated order books can produce spurious edges. An edge sanity cap (default 20 %)
  and LMSR max-*b* guard reject the most obvious model failures.
- **Execution risk.** Orders may not fill, may fill at worse prices, or may be rejected by
  the CLOB API. The bot uses FOK for weather trades and GTC for general trades.
- **Churn risk.** Without the post-exit re-entry cooldown (default 60 min), the bot can
  enter-exit-enter the same market repeatedly, bleeding money through the bid-ask spread.
- **Weather forecast staleness.** NOAA forecasts are cached for 1 hour. Markets < 6 h or
  > 3 days from resolution are skipped.
- **Single point of failure.** The bot runs as a single Docker container. If it crashes or
  loses connectivity, open orders remain on the exchange until cancelled or filled.
- **Private key exposure.** The `.env` file contains your wallet private key. Never commit
  it to version control.

## License

Private repository. All rights reserved.
