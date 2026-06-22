"""
Phase 1 — 数据拉取层。

职责：
  - 从 akshare 拉取全市场行情、涨停池、跌停池、炸板池
  - 每次请求带 retry + 指数退避
  - 成功后写本地 CSV 缓存
  - 失败时降级读最近缓存，不抛异常给上层

接口调用频率限制：每次调用后有可配置的 gap，不做秒级高频请求。
"""

import time
import functools
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from loguru import logger


# ── retry 装饰器 ─────────────────────────────────────────────────────────────

def retry(max_attempts: int = 3, delay: float = 2.0):
    """指数退避重试。delay 秒起步，每次翻倍。"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt < max_attempts - 1:
                        wait = delay * (2 ** attempt)
                        logger.warning(
                            f"[{func.__name__}] 第{attempt + 1}/{max_attempts}次失败: "
                            f"{exc!r}，{wait:.0f}s 后重试"
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            f"[{func.__name__}] 已重试 {max_attempts} 次，最终失败: {exc!r}"
                        )
                        raise
        return wrapper
    return decorator


# ── DataFetcher ──────────────────────────────────────────────────────────────

class DataFetcher:
    """
    数据拉取器。

    Parameters
    ----------
    data_dir : str
        缓存根目录，子目录按日期组织，如 data/20260622/a_spot.csv
    request_gap : float
        两次 API 调用之间的最短间隔（秒），默认 1.5
    retry_attempts : int
        每个接口最大重试次数
    retry_delay : float
        首次重试延迟（秒）
    """

    def __init__(
        self,
        data_dir: str = "data",
        request_gap: float = 1.5,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
    ):
        self.data_dir = Path(data_dir)
        self.request_gap = request_gap
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
        self._last_request_time: float = 0.0

    # ── 缓存工具 ──────────────────────────────────────────────────────────────

    def _cache_path(self, date: str, name: str) -> Path:
        return self.data_dir / date / f"{name}.csv"

    def _load_cache(self, date: str, name: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(date, name)
        if path.exists() and path.stat().st_size > 10:
            try:
                df = pd.read_csv(path, dtype=str)
                logger.debug(f"命中缓存 → {path}  ({len(df)} 行)")
                return df
            except Exception as e:
                logger.warning(f"读取缓存失败 {path}: {e}")
        return None

    def _save_cache(self, df: pd.DataFrame, date: str, name: str) -> None:
        path = self._cache_path(date, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"已缓存 → {path}  ({len(df)} 行)")

    def _throttle(self) -> None:
        """控制请求频率。"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_gap:
            time.sleep(self.request_gap - elapsed)
        self._last_request_time = time.time()

    # ── 底层 akshare 调用（带 retry）────────────────────────────────────────

    def _call_api(self, func: Callable, *args, **kwargs) -> pd.DataFrame:
        """统一的限频 + retry 调用入口。"""
        self._throttle()

        @retry(max_attempts=self._retry_attempts, delay=self._retry_delay)
        def _inner():
            return func(*args, **kwargs)

        return _inner()

    # ── 公开拉取接口（缓存优先 → 请求 → 降级）────────────────────────────────

    def fetch_a_spot(self, date: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        全市场实时行情。
        akshare: ak.stock_zh_a_spot_em()
        主要字段: 代码, 名称, 最新价, 涨跌幅, 成交额, 换手率, 量比, 涨速 ...
        """
        if not force_refresh:
            cached = self._load_cache(date, "a_spot")
            if cached is not None:
                return cached

        try:
            import akshare as ak
            df = self._call_api(ak.stock_zh_a_spot_em)
            if df is not None and not df.empty:
                self._save_cache(df, date, "a_spot")
                return df
            logger.warning("stock_zh_a_spot_em 返回空数据")
        except Exception as e:
            logger.error(f"fetch_a_spot 请求失败，降级读缓存: {e}")

        return self._load_cache(date, "a_spot")

    def fetch_limit_up_pool(self, date: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        涨停池。
        akshare: ak.stock_zt_pool_em(date=date)
        主要字段: 代码, 名称, 涨跌幅, 最新价, 成交额, 流通市值, 换手率,
                  封板资金, 最后封板时间, 炸板次数, 涨停统计, 连板数, 所属行业
        """
        if not force_refresh:
            cached = self._load_cache(date, "zt_pool")
            if cached is not None:
                return cached

        try:
            import akshare as ak
            df = self._call_api(ak.stock_zt_pool_em, date=date)
            if df is not None and not df.empty:
                self._save_cache(df, date, "zt_pool")
                return df
            logger.warning(f"stock_zt_pool_em({date}) 返回空数据（可能非交易日）")
        except Exception as e:
            logger.error(f"fetch_limit_up_pool 请求失败，降级读缓存: {e}")

        return self._load_cache(date, "zt_pool")

    def fetch_limit_down_pool(self, date: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        跌停池。
        akshare: ak.stock_zt_pool_dtgc_em(date=date)
        主要字段: 代码, 名称, 涨跌幅, 最新价, 成交额, 流通市值, 换手率 ...
        """
        if not force_refresh:
            cached = self._load_cache(date, "dt_pool")
            if cached is not None:
                return cached

        try:
            import akshare as ak
            df = self._call_api(ak.stock_zt_pool_dtgc_em, date=date)
            if df is not None and not df.empty:
                self._save_cache(df, date, "dt_pool")
                return df
            logger.warning(f"stock_zt_pool_dtgc_em({date}) 返回空数据")
        except Exception as e:
            logger.error(f"fetch_limit_down_pool 请求失败，降级读缓存: {e}")

        return self._load_cache(date, "dt_pool")

    def fetch_failed_board_pool(self, date: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        炸板池。
        akshare: ak.stock_zt_pool_zbgc_em(date=date)
        主要字段: 代码, 名称, 涨跌幅, 最新价, 成交额, 流通市值, 换手率,
                  炸板次数, 涨停统计, 连板数, 所属行业
        """
        if not force_refresh:
            cached = self._load_cache(date, "zb_pool")
            if cached is not None:
                return cached

        try:
            import akshare as ak
            df = self._call_api(ak.stock_zt_pool_zbgc_em, date=date)
            if df is not None and not df.empty:
                self._save_cache(df, date, "zb_pool")
                return df
            logger.warning(f"stock_zt_pool_zbgc_em({date}) 返回空数据")
        except Exception as e:
            logger.error(f"fetch_failed_board_pool 请求失败，降级读缓存: {e}")

        return self._load_cache(date, "zb_pool")

    def fetch_all(
        self, date: str, force_refresh: bool = False
    ) -> dict[str, Optional[pd.DataFrame]]:
        """一次性拉取全部数据，返回 {key: DataFrame | None}。"""
        logger.info(f"═══ 开始拉取 {date} 数据 ═══")
        result = {
            "a_spot": self.fetch_a_spot(date, force_refresh),
            "zt_pool": self.fetch_limit_up_pool(date, force_refresh),
            "dt_pool": self.fetch_limit_down_pool(date, force_refresh),
            "zb_pool": self.fetch_failed_board_pool(date, force_refresh),
        }
        ok = sum(1 for v in result.values() if v is not None)
        logger.info(f"数据拉取完成: {ok}/{len(result)} 项成功")
        return result
