#!/usr/bin/env python3
"""Interactive setup helper for Polymarket wallet and API credentials."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    print("=" * 60)
    print("  Polymarket Trading Bot — Wallet Setup")
    print("=" * 60)
    print()
    print("This script will help you configure your bot.\n")

    print("STEP 1: Wallet Private Key")
    print("-" * 40)
    print("You need an Ethereum wallet private key (on Polygon).")
    print("IMPORTANT: Use a DEDICATED wallet, not your personal one.")
    print()

    pk = input("Enter your private key (0x...): ").strip()
    if not pk.startswith("0x"):
        pk = "0x" + pk

    print()
    print("STEP 2: Funder Address (optional)")
    print("-" * 40)
    print("If you use a proxy/contract wallet, enter its address.")
    print("Otherwise, press Enter to skip.")
    print()

    funder = input("Funder address (or Enter to skip): ").strip()

    print()
    print("STEP 3: Slack Webhook URL")
    print("-" * 40)
    print("Go to: https://api.slack.com/messaging/webhooks")
    print("Create an app, enable Incoming Webhooks, add to your channel.")
    print()

    slack = input("Slack webhook URL: ").strip()

    # Write .env file
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    with open(env_path, "w") as f:
        f.write(f"POLYMARKET_PRIVATE_KEY={pk}\n")
        f.write(f"POLYMARKET_FUNDER_ADDRESS={funder}\n")
        f.write("POLYMARKET_CHAIN_ID=137\n")
        f.write("CLOB_HOST=https://clob.polymarket.com\n")
        f.write("GAMMA_HOST=https://gamma-api.polymarket.com\n")
        f.write(f"SLACK_WEBHOOK_URL={slack}\n")
        f.write("BOT_ENV=development\n")

    os.chmod(env_path, 0o600)

    print()
    print(f"Configuration written to {env_path} (permissions: 600)")
    print()
    print("STEP 4: Verify API credentials")
    print("-" * 40)

    try:
        from py_clob_client.client import ClobClient

        client = ClobClient(
            "https://clob.polymarket.com",
            key=pk,
            chain_id=137,
            signature_type=1,
            funder=funder or None,
        )
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        print("API credentials derived successfully!")
        print(f"  API Key: {creds.api_key[:12]}...")
    except Exception as e:
        print(f"WARNING: Could not derive API credentials: {e}")
        print("The bot will attempt this again on startup.")

    print()
    print("SETUP COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Fund your wallet with USDC on Polygon")
    print("  2. Visit polymarket.com and deposit USDC")
    print("  3. Configure bankroll in config/default.yaml")
    print("  4. Run: python -m src.main")
    print()
    print("The bot starts in DRY RUN mode by default.")
    print("Set dry_run: false in config/default.yaml to go live.")


if __name__ == "__main__":
    main()
