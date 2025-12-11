from __future__ import annotations

import logging
import os
import re
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

# Base paths
APP_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = APP_DIR.parent
LOG_DIR = ROOT_DIR / "logs"

# Default logging configuration
DEFAULT_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def _current_level(level_name: Optional[str] = None) -> int:
    """Translate a level name to its numeric value with INFO as fallback."""
    return getattr(logging, (level_name or DEFAULT_LEVEL_NAME), logging.INFO)


def _build_file_handler(level: int) -> TimedRotatingFileHandler:
    """Create a handler that writes a .txt log file per day."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    handler = TimedRotatingFileHandler(
        filename=str(LOG_DIR / "application.log"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
        utc=False,
    )
    handler.suffix = "%Y-%m-%d.txt"
    handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}\.txt$")

    def _namer(default_name: str) -> str:
        """Rename rotated files to use only the date (YYYY-MM-DD.txt)."""
        path = Path(default_name)
        stamp = path.name.split(".")[-1]
        return str(path.parent / f"{stamp}.txt")

    handler.namer = _namer

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    return handler


def _build_console_handler(level: int) -> logging.Handler:
    """Send log output to stdout for easier debugging."""
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    return console_handler


def configure_logging(level_name: Optional[str] = None, reset: bool = False) -> None:
    """
    Configure application-wide logging.

    Parameters
    ----------
    level_name: Optional[str]
        Desired logging level name (e.g., "DEBUG"); defaults to LOG_LEVEL env value.
    reset: bool
        If True, reconfigures handlers even when already configured.
    """
    global _CONFIGURED

    if _CONFIGURED and not reset:
        return

    root_logger = logging.getLogger()
    current_level = _current_level(level_name)
    root_logger.setLevel(current_level)

    # Remove existing handlers to avoid duplicate logs.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(_build_console_handler(current_level))
    root_logger.addHandler(_build_file_handler(current_level))
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retrieve a logger with the global configuration already applied."""
    configure_logging()
    return logging.getLogger(name)