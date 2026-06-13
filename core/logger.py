"""
core/logger.py — Rich-powered logger.
Every agent imports get_logger(__name__).
"""
import logging
import os
from rich.logging import RichHandler
from rich.console import Console

_console = Console(stderr=True)


def get_logger(name: str) -> logging.Logger:
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(console=_console, rich_tracebacks=True, markup=True)
        handler.setLevel(level)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
