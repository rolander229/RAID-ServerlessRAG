import logging
import logging.handlers
import os
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Logging formatter that adds ANSI colors to the console output based on level."""

    COLOR_RESET = "\x1b[0m"
    COLOR_MAP = {
        logging.DEBUG: "\x1b[36m",    # cyan
        logging.INFO: "\x1b[32m",     # green
        logging.WARNING: "\x1b[33m",  # yellow
        logging.ERROR: "\x1b[31m",    # red
        logging.CRITICAL: "\x1b[35m", # magenta
    }

    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt)

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self.COLOR_MAP.get(record.levelno, "")
        if color:
            return f"{color}{message}{self.COLOR_RESET}"
        return message


def get_logger(name: str, level: str = "INFO", log_dir: Optional[str] = None) -> logging.Logger:
    """Create and return a configured logger.

    - Adds a colorized console handler and a rotating file handler (file is plain text).
    - Ensures the log directory exists if provided (defaults to `./logs`).

    Args:
        name: Logger name.
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Optional directory path to write log files to. If None, will use
                 a `logs` folder next to this module.

    Returns:
        Configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid adding handlers multiple times when called repeatedly
    if logger.handlers:
        return logger

    # Console handler with colors
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_formatter = ColoredFormatter()
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # File handler (rotating) - plain (no colors)
    if log_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "logs")

    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{name}.log")
        fh = logging.handlers.RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setLevel(level)
        fh_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
    except Exception:
        # If file handler setup fails for any reason, keep console logging only.
        logger = logging.getLogger(name)
        logger.exception("Failed to initialize file handler for logger; continuing with console only.")

    return logger
