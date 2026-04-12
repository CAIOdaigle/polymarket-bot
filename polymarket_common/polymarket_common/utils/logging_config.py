from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from pythonjsonlogger import jsonlogger

from polymarket_common.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    log_path = Path(config.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    # Clear existing handlers
    root.handlers.clear()

    if config.format == "json":
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    else:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        config.file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Stdout handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Quiet noisy libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
