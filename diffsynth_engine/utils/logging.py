import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Formatter that implements loguru-style colorful output style"""

    # ANSI color codes
    _RED = "\033[31m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _MAGENTA = "\033[35m"
    _CYAN = "\033[36m"
    _RESET = "\033[0m"

    # Log level color mapping
    LEVEL_COLORS = {
        "DEBUG": _CYAN,
        "INFO": _GREEN,
        "WARNING": _YELLOW,
        "ERROR": _RED,
        "CRITICAL": _MAGENTA,
    }

    def format(self, record):
        # Format time
        log_time = self.formatTime(record, "%Y-%m-%d %H:%M:%S")

        # Get level color
        level_color = self.LEVEL_COLORS.get(record.levelname, self._RESET)

        # Format level name (left-aligned, width 8)
        level_name = f"{record.levelname:<8}"

        # Build log message
        log_message = (
            f"{self._GREEN}{log_time}{self._RESET} | "
            f"{level_color}{level_name}{self._RESET} | "
            f"{self._CYAN}{record.name}{self._RESET}:"
            f"{self._CYAN}{record.funcName}{self._RESET}:"
            f"{self._CYAN}{record.lineno}{self._RESET} - "
            f"{level_color}{record.getMessage()}{self._RESET}"
        )

        # Handle exception information
        if record.exc_info:
            log_message += "\n" + self.formatException(record.exc_info)

        return log_message


def get_logger(name: Optional[str] = None, level=logging.DEBUG) -> logging.Logger:
    """
    Set up and return a loguru-style logger

    Args:
        name: Logger name, defaults to None (root logger)
        level: Log level, defaults to DEBUG

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate logs
    logger.handlers.clear()

    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())

    logger.addHandler(console_handler)

    # Prevent log propagation to parent logger
    logger.propagate = False

    return logger
