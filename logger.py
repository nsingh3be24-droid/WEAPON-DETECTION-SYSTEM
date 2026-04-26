# logger.py — Centralized logging for the Weapon Detection System

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import LOG_CFG

_loggers: dict[str, logging.Logger] = {}   # module-level cache


def get_logger(name: str = "weapon_detection") -> logging.Logger:
    """
    Return a named logger configured with console + rotating file handlers.
    Subsequent calls with the same `name` return the cached instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(LOG_CFG["level"])
    logger.propagate = False  # avoid duplicate root-logger output

    formatter = logging.Formatter(
        fmt=LOG_CFG["format"],
        datefmt=LOG_CFG["datefmt"],
    )

    # ── Console handler ──────────────────────────────────────────────────────
    if LOG_CFG["console"]:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(LOG_CFG["level"])
        logger.addHandler(ch)

    # ── Rotating file handler ────────────────────────────────────────────────
    if LOG_CFG["file"]:
        log_path = Path(LOG_CFG["log_file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fh = RotatingFileHandler(
            filename=log_path,
            maxBytes=LOG_CFG["max_bytes"],
            backupCount=LOG_CFG["backup_count"],
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        fh.setLevel(LOG_CFG["level"])
        logger.addHandler(fh)

    _loggers[name] = logger
    return logger


# ── Convenience shortcuts ────────────────────────────────────────────────────

def set_level(level: str, name: str = "weapon_detection") -> None:
    """Dynamically change log level at runtime (e.g., switch to DEBUG)."""
    logger = get_logger(name)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def log_system_info(logger: logging.Logger) -> None:
    """Log Python version, torch availability, and OpenCV build info."""
    import platform
    logger.info("─" * 60)
    logger.info("Weapon Detection System — startup")
    logger.info("Platform : %s", platform.platform())
    logger.info("Python   : %s", platform.python_version())

    try:
        import torch
        logger.info(
            "PyTorch  : %s  (CUDA available: %s)",
            torch.__version__,
            torch.cuda.is_available(),
        )
        if torch.cuda.is_available():
            logger.info("CUDA GPU : %s", torch.cuda.get_device_name(0))
    except ImportError:
        logger.warning("PyTorch not found — GPU inference unavailable")

    try:
        import cv2
        logger.info("OpenCV   : %s", cv2.__version__)
    except ImportError:
        logger.warning("OpenCV not found")

    logger.info("─" * 60)
