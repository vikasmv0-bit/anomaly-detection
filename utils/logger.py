"""
utils/logger.py
---------------
Provides a unified logger with colored console output and rotating file handler.
"""

import logging
import os
from logging.handlers import RotatingFileHandler


# ANSI color codes for console output
_COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes based on log level."""

    FMT = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
    DATE_FMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        formatter = logging.Formatter(
            f"{color}{self.FMT}{_RESET}", datefmt=self.DATE_FMT
        )
        return formatter.format(record)


def get_logger(name: str, log_dir: str = "logs", level: int = logging.DEBUG) -> logging.Logger:
    """
    Create (or retrieve) a named logger with:
    - Colored console handler (INFO+)
    - Rotating file handler (DEBUG+, max 5 MB, 3 backups)

    Args:
        name:    Logger name (typically __name__ of the calling module).
        log_dir: Directory where log files are stored.
        level:   Root logging level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── Console handler ──────────────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ColorFormatter())
    logger.addHandler(ch)

    # ── File handler ─────────────────────────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setLevel(logging.DEBUG)
    plain_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(plain_fmt)
    logger.addHandler(fh)

    return logger
