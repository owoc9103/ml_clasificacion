"""
Structured logging utilities for quantfund.

Provides centralized logging configuration with JSON formatting
for production observability.
"""
import logging
import sys
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure structured logger for a module.

    Args:
        name: Logger name (typically __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for file logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with structured format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Structured format: timestamp | level | module | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create logger for module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set up default console handler
    if not logger.handlers:
        return setup_logger(name)

    return logger


__all__ = ["setup_logger", "get_logger"]
