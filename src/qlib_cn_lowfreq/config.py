"""Configuration helpers for the low-frequency Qlib workflow."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path("data")
AKSHARE_DATA_DIR = DATA_ROOT / "akshare_ths"
QLIB_PROVIDER_URI = AKSHARE_DATA_DIR / "qlib_data"
EXP_ROOT = Path("experiments")

# Default location of the official Qlib Yahoo! 财经数据（cn market）的存储目录。
# 当需要将同花顺数据合并到已有的 Yahoo 数据集时，可使用该默认路径或通过命令行参数覆盖。
DEFAULT_YAHOO_PROVIDER = Path.home() / ".qlib" / "qlib_data" / "cn_data"


@dataclass
class DateRange:
    """Simple container to hold start and end date strings."""

    start: str
    end: str


DEFAULT_DATE_RANGE = DateRange(start="2015-01-01", end="2024-12-31")

# Weekly frequency is more appropriate for low/medium-horizon research.
DEFAULT_FREQ = "day"
DEFAULT_RESAMPLE_RULE = "W-FRI"

# Factor/feature configuration can be expanded as needed.
FACTOR_CONFIG = {
    "price_factors": ["OPEN", "HIGH", "LOW", "CLOSE"],
    "volume_factors": ["VOLUME"],
    "alpha158_like": True,
}


def ensure_directories() -> None:
    """Create all expected directories if they do not yet exist."""

    AKSHARE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    QLIB_PROVIDER_URI.mkdir(parents=True, exist_ok=True)
    EXP_ROOT.mkdir(parents=True, exist_ok=True)


__all__ = [
    "AKSHARE_DATA_DIR",
    "QLIB_PROVIDER_URI",
    "DEFAULT_YAHOO_PROVIDER",
    "DATA_ROOT",
    "EXP_ROOT",
    "DateRange",
    "DEFAULT_DATE_RANGE",
    "DEFAULT_FREQ",
    "DEFAULT_RESAMPLE_RULE",
    "FACTOR_CONFIG",
    "ensure_directories",
]
