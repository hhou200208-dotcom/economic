"""
Phase 2 — 市场情绪评分。

输入: data_fetcher 返回的四张表
输出: dict (所有指标) + 写 data/{date}/market_summary.csv
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class EmotionScorer:
    """
    计算并保存市场情绪评分。

    评分公式（总分 0-100）:
        score =
            min(涨停数 / 150, 1) * 25
          + min(最高连板 / 8,  1) * 15
          + min(20cm数  / 30,  1) * 10
          + min(成交额比值, 1.5) / 1.5 * 15
          + min(涨跌家数比, 2.0) / 2.0 * 15
          - min(跌停数 / 80,   1) * 10
          - min(炸板率,        1) * 10
    """

    # akshare 返回的列名（优先顺序，适配可能的列名变化）
    _COL_PCT_CHG  = ["涨跌幅"]
    _COL_AMOUNT   = ["成交额"]
    _COL_BOARDS   = ["连板数"]
    _COL_CODE     = ["代码"]

    def __init__(self, data_dir: str = "data", cfg: dict = None):
        self.data_dir = Path(data_dir)
        cfg = cfg or {}
        ecfg = cfg.get("emotion", {})
        self.limit_up_base     = ecfg.get("limit_up_base",    150)
        self.board_height_base = ecfg.get("board_height_base", 8)
        self.twenty_cm_base    = ecfg.get("twenty_cm_base",   30)
        self.amount_ratio_cap  = ecfg.get("amount_ratio_cap", 1.5)
        self.up_down_ratio_cap = ecfg.get("up_down_ratio_cap", 2.0)
        self.limit_down_base   = ecfg.get("limit_down_base",  80)

    # ── 内部工具 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _num(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def _find_col(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # ── 指标计算 ─────────────────────────────────────────────────────────────

    def _limit_up_count(self, zt: Optional[pd.DataFrame]) -> int:
        return len(zt) if zt is not None and not zt.empty else 0

    def _limit_down_count(self, dt: Optional[pd.DataFrame]) -> int:
        return len(dt) if dt is not None and not dt.empty else 0

    def _failed_board_count(self, zb: Optional[pd.DataFrame]) -> int:
        return len(zb) if zb is not None and not zb.empty else 0

    def _max_board_height(self, zt: Optional[pd.DataFrame]) -> int:
        if zt is None or zt.empty:
            return 0
        col = self._find_col(zt, self._COL_BOARDS)
        if col:
            val = self._num(zt[col]).max()
            return int(val) if not pd.isna(val) else 1
        return 1  # 有涨停池但没连板列，至少算1板

    def _twenty_cm_count(
        self,
        a_spot: Optional[pd.DataFrame],
        zt: Optional[pd.DataFrame],
    ) -> int:
        """
        20cm涨停 = 科创板(688xxx)/创业板(300xxx) 当日涨跌幅 >= 19.9%。
        优先从涨停池统计（更准确），次选从全市场行情统计。
        """
        def _count_from_df(df: pd.DataFrame) -> Optional[int]:
            col = self._find_col(df, self._COL_PCT_CHG)
            if col is None:
                return None
            pct = self._num(df[col])
            return int((pct >= 19.9).sum())

        if zt is not None and not zt.empty:
            cnt = _count_from_df(zt)
            if cnt is not None:
                return cnt

        if a_spot is not None and not a_spot.empty:
            cnt = _count_from_df(a_spot)
            if cnt is not None:
                return cnt

        return 0

    def _failed_board_rate(self, limit_up: int, failed: int) -> float:
        """炸板率 = 炸板数 / (涨停数 + 炸板数)"""
        total = limit_up + failed
        return failed / total if total > 0 else 0.0

    def _up_down_counts(
        self, a_spot: Optional[pd.DataFrame]
    ) -> tuple[int, int, int, int]:
        """返回 (上涨家数, 下跌家数, 涨幅>5%家数, 跌幅>5%家数)"""
        if a_spot is None or a_spot.empty:
            return 0, 0, 0, 0
        col = self._find_col(a_spot, self._COL_PCT_CHG)
        if col is None:
            return 0, 0, 0, 0
        pct = self._num(a_spot[col])
        return (
            int((pct > 0).sum()),
            int((pct < 0).sum()),
            int((pct >= 5).sum()),
            int((pct <= -5).sum()),
        )

    def _total_amount(self, a_spot: Optional[pd.DataFrame]) -> float:
        if a_spot is None or a_spot.empty:
            return 0.0
        col = self._find_col(a_spot, self._COL_AMOUNT)
        if col is None:
            return 0.0
        return float(self._num(a_spot[col]).sum())

    def _amount_5d_avg(self, current_date: str) -> Optional[float]:
        """
        从过去最多 30 个自然日中找 5 个有缓存的交易日，计算其成交额均值。
        """
        amounts: list[float] = []
        dt = datetime.strptime(current_date, "%Y%m%d")
        delta = 1
        checked = 0
        while len(amounts) < 5 and checked < 30:
            candidate = (dt - timedelta(days=delta)).strftime("%Y%m%d")
            delta += 1
            checked += 1
            cache = self.data_dir / candidate / "a_spot.csv"
            if not cache.exists():
                continue
            try:
                df = pd.read_csv(cache, dtype=str)
                col = self._find_col(df, self._COL_AMOUNT)
                if col:
                    total = self._num(df[col]).sum()
                    if total > 0:
                        amounts.append(float(total))
            except Exception:
                pass

        if len(amounts) >= 2:
            return sum(amounts) / len(amounts)
        return None

    # ── 评分主函数 ────────────────────────────────────────────────────────────

    def compute(
        self,
        date: str,
        a_spot: Optional[pd.DataFrame],
        zt_pool: Optional[pd.DataFrame],
        dt_pool: Optional[pd.DataFrame],
        zb_pool: Optional[pd.DataFrame],
    ) -> dict:
        """
        计算全部指标并返回结果字典。
        任何数据缺失时对应指标标注 None，不影响其他指标计算。
        """
        lu  = self._limit_up_count(zt_pool)
        ld  = self._limit_down_count(dt_pool)
        fb  = self._failed_board_count(zb_pool)
        mh  = self._max_board_height(zt_pool)
        t20 = self._twenty_cm_count(a_spot, zt_pool)
        fbr = self._failed_board_rate(lu, fb)

        up, dn, up5, dn5 = self._up_down_counts(a_spot)
        total_amt  = self._total_amount(a_spot)
        avg_5d     = self._amount_5d_avg(date)
        amt_ratio  = (total_amt / avg_5d) if (avg_5d and avg_5d > 0) else None

        # 成交额比值用于评分时，若无历史数据以 1.0（持平）代入，避免误扣分
        amt_ratio_for_score = amt_ratio if amt_ratio is not None else 1.0

        # ── 评分公式 ──────────────────────────────────────────────────────────
        up_dn_ratio = up / max(dn, 1)

        score = (
            min(lu  / self.limit_up_base,    1.0) * 25
          + min(mh  / self.board_height_base, 1.0) * 15
          + min(t20 / self.twenty_cm_base,    1.0) * 10
          + min(amt_ratio_for_score, self.amount_ratio_cap) / self.amount_ratio_cap * 15
          + min(up_dn_ratio, self.up_down_ratio_cap) / self.up_down_ratio_cap * 15
          - min(ld  / self.limit_down_base,   1.0) * 10
          - min(fbr, 1.0) * 10
        )
        score = round(max(0.0, min(100.0, score)), 2)

        # ── 状态标签 ──────────────────────────────────────────────────────────
        if score >= 80:
            status = "强情绪：可做主线核心和强分支"
        elif score >= 60:
            status = "结构性行情：只做核心，不碰杂毛"
        elif score >= 40:
            status = "分化行情：低吸优先，谨慎追高"
        else:
            status = "退潮期：防守，保护利润"

        result = {
            "date":               date,
            "emotion_score":      score,
            "status":             status,
            # 涨跌停
            "limit_up_count":     lu,
            "limit_down_count":   ld,
            "failed_board_count": fb,
            "failed_board_rate":  round(fbr * 100, 2),   # %
            "max_board_height":   mh,
            "twenty_cm_count":    t20,
            # 市场宽度
            "up_count":           up,
            "down_count":         dn,
            "up_5pct_count":      up5,
            "down_5pct_count":    dn5,
            # 成交额
            "total_amount_yi":    round(total_amt / 1e8, 2),
            "amount_5d_avg_yi":   round(avg_5d / 1e8, 2) if avg_5d else None,
            "amount_ratio":       round(amt_ratio, 3) if amt_ratio is not None else None,
            # 分项得分（便于调参）
            "_score_lu":    round(min(lu / self.limit_up_base,    1.0) * 25, 2),
            "_score_board": round(min(mh / self.board_height_base, 1.0) * 15, 2),
            "_score_20cm":  round(min(t20 / self.twenty_cm_base,   1.0) * 10, 2),
            "_score_amt":   round(min(amt_ratio_for_score, self.amount_ratio_cap) / self.amount_ratio_cap * 15, 2),
            "_score_updn":  round(min(up_dn_ratio, self.up_down_ratio_cap) / self.up_down_ratio_cap * 15, 2),
            "_penalty_ld":  round(min(ld / self.limit_down_base, 1.0) * 10, 2),
            "_penalty_fb":  round(min(fbr, 1.0) * 10, 2),
        }

        logger.info(
            f"情绪评分: {score:.1f}  |  {status}  |  "
            f"涨停={lu} 跌停={ld} 炸板={fb} 连板={mh} 20cm={t20}"
        )
        return result

    def save(self, result: dict, date: str) -> Path:
        path = self.data_dir / date / "market_summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"市场概况已保存 → {path}")
        return path
