"""通用工具函数。"""

import sys
from datetime import datetime, date
from pathlib import Path
from typing import Union

import yaml
from loguru import logger


def load_config(config_path: Union[str, Path] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(log_dir: Union[str, Path] = "logs", level: str = "INFO") -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        str(log_dir / "{time:YYYY-MM-DD}.log"),
        rotation="1 day",
        retention="30 days",
        encoding="utf-8",
        level="DEBUG",
    )


def today_str() -> str:
    return date.today().strftime("%Y%m%d")


def fmt_date(d: str) -> str:
    """20260622 → 2026-06-22"""
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def yi(val: float) -> str:
    """元 → 亿元字符串"""
    if val is None:
        return "N/A"
    return f"{val / 1e8:.2f} 亿"
