"""Polymarket authentication wrapper."""

from __future__ import annotations

import logging

from py_clob_client.client import ClobClient

from polymarket_common.config import PolymarketConfig

logger = logging.getLogger(__name__)


def create_clob_client(config: PolymarketConfig) -> ClobClient:
    """
    Initialize and authenticate a ClobClient.

    Steps:
      1. Create client with private key
      2. Derive API credentials (HMAC key/secret/passphrase)
      3. Set creds on client
    """
    client = ClobClient(
        config.clob_host,
        key=config.private_key,
        chain_id=config.chain_id,
        signature_type=1,  # POLY_PROXY (email/Magic login)
        funder=config.funder_address or None,
    )

    # Derive or create HMAC credentials for L2 auth
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)

    logger.info("Authenticated with Polymarket CLOB (chain_id=%d)", config.chain_id)
    return client
