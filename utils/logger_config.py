from datetime import datetime
from pathlib import Path

from loguru import logger

LOG_DATE = datetime.now().strftime("%Y-%m-%d")

LOG_DIR = (
    Path("logs")
    / LOG_DATE
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

logger.remove()

logger.add(
    sink=lambda msg: print(msg, end=""),
    colorize=True,
    level="INFO",
)

logger.add(
    LOG_DIR / f"{datetime.now()}_pipeline.log",
    rotation="50 MB",
    retention="100 days",
    level="DEBUG",
)

__all__ = ["logger"]