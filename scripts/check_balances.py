#!/usr/bin/env python3
"""Quick balance and position check."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()


def main():
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import AssetType, BalanceAllowanceParams, OpenOrderParams

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set in .env")
        sys.exit(1)

    funder = os.getenv("POLYMARKET_FUNDER_ADDRESS", "") or None
    host = os.getenv("CLOB_HOST", "https://clob.polymarket.com")

    print("Connecting to Polymarket...")
    client = ClobClient(host, key=pk, chain_id=137, signature_type=1, funder=funder)
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)

    print()
    print("Balance & Allowance:")
    print("-" * 40)
    try:
        bal = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        print(f"  USDC Balance:   {bal.get('balance', 'N/A')}")
        print(f"  USDC Allowance: {bal.get('allowance', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")

    print()
    print("Open Orders:")
    print("-" * 40)
    try:
        orders = client.get_orders(OpenOrderParams())
        if not orders:
            print("  No open orders")
        else:
            for o in orders[:10]:
                print(f"  {o.get('id', '')[:12]} {o.get('side')} {o.get('size')} @ {o.get('price')}")
            if len(orders) > 10:
                print(f"  ... and {len(orders) - 10} more")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()
