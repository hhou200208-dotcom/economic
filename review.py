"""
A股复盘脚本 — 基于 akshare 真实数据，分析个股盘面并给出次日概率倾向。

【重要声明】
- 数据来自 akshare（底层接口：东方财富、新浪等），可能有延迟，以券商终端为准。
- 本脚本不构成任何投资建议，分析结论仅为技术规则的机械输出，盈亏自负。

【使用方法】
  pip install akshare
  python review.py

【依赖】
  akshare >= 1.10
  pandas >= 1.3
"""

# ──────────────────────────────────────────────
# 用户配置区（改这里即可复用）
# ──────────────────────────────────────────────
STOCKS = [
    {"code": "603773", "name": "沃格光电"},
    {"code": "600487", "name": "亨通光电"},
]

# 目标复盘日期（格式 YYYYMMDD），None 表示取今天
TARGET_DATE = None  # 例: "20250617"

# 日K回看天数（含目标日）
HIST_DAYS = 10

# 量能对比窗口（前 N 日均值）
VOLUME_WINDOW = 5
# ──────────────────────────────────────────────

import sys
import warnings
from datetime import date, datetime, timedelta

warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import akshare as ak
except ImportError as e:
    sys.exit(f"[FATAL] 缺少依赖: {e}\n请执行: pip install akshare pandas")

# ─── 工具函数 ───

DIVIDER = "=" * 70
THIN = "-" * 70


def _today_str() -> str:
    return date.today().strftime("%Y%m%d")


def _date_str(d: str) -> str:
    """20250617 → 2025-06-17"""
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _pct(val) -> str:
    if pd.isna(val):
        return "N/A"
    return f"{val:+.2f}%"


def _fmt(val, decimal=2, suffix="") -> str:
    if pd.isna(val):
        return "N/A"
    return f"{val:.{decimal}f}{suffix}"


def _safe(func, *args, **kwargs):
    """执行 func，任何异常返回 None 并打印警告。"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return None


# ─── 数据拉取 ───

def fetch_daily_kline(code: str, target_date: str) -> pd.DataFrame | None:
    """
    拉取含目标日在内的最近 HIST_DAYS+5 个交易日日K（不复权）。
    列: 日期 开盘 最高 最低 收盘 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
    """
    # 往前多取一些，保证有足够数据计算前5日均值
    extra = HIST_DAYS + VOLUME_WINDOW + 5
    end_dt = datetime.strptime(target_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=extra * 2)
    start_str = start_dt.strftime("%Y%m%d")

    df = _safe(
        ak.stock_zh_a_hist,
        symbol=code,
        period="daily",
        start_date=start_str,
        end_date=target_date,
        adjust="",
    )
    if df is None or df.empty:
        return None

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # 统一列名（akshare 中文列名）
    col_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",   # 手
        "成交额": "amount",   # 元
        "振幅": "amplitude",  # %
        "涨跌幅": "pct_chg",  # %
        "涨跌额": "change",
        "换手率": "turnover", # %
    }
    df.rename(columns=col_map, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_minute_data(code: str, target_date: str) -> pd.DataFrame | None:
    """
    拉取目标日 1 分钟 K 线（不复权）。
    返回列含: 时间 开盘 最高 最低 收盘 成交量 成交额
    """
    # stock_zh_a_hist_min_em：东财分钟线
    start_str = f"{_date_str(target_date)} 09:30:00"
    end_str = f"{_date_str(target_date)} 15:00:00"

    df = _safe(
        ak.stock_zh_a_hist_min_em,
        symbol=code,
        start_date=start_str,
        end_date=end_str,
        period="1",
        adjust="",
    )
    if df is None or df.empty:
        return None

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    col_map = {
        "时间": "time",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # 过滤只保留目标日
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df[df["time"].dt.strftime("%Y%m%d") == target_date]
    return df.reset_index(drop=True) if not df.empty else None


def fetch_index_sh(target_date: str) -> float | None:
    """拉取上证指数当日涨跌幅。"""
    df = _safe(
        ak.stock_zh_index_daily_em,
        symbol="sh000001",
        start_date=target_date,
        end_date=target_date,
    )
    if df is None or df.empty:
        return None
    df.columns = [c.strip() for c in df.columns]
    # akshare 返回列: date open high low close volume amount
    # 需要计算涨跌幅：当天close vs 上一交易日close，此处另拉两天
    df2 = _safe(
        ak.stock_zh_index_daily_em,
        symbol="sh000001",
        start_date=(datetime.strptime(target_date, "%Y%m%d") - timedelta(days=10)).strftime("%Y%m%d"),
        end_date=target_date,
    )
    if df2 is None or len(df2) < 2:
        return None
    df2 = df2.sort_values(df2.columns[0]).reset_index(drop=True)
    last_close = df2.iloc[-2]["close"] if "close" in df2.columns else df2.iloc[-2].iloc[4]
    today_close = df2.iloc[-1]["close"] if "close" in df2.columns else df2.iloc[-1].iloc[4]
    return (today_close - last_close) / last_close * 100


def fetch_concept_boards(code: str) -> list[str] | None:
    """查询个股所属概念板块列表（东财）。"""
    # stock_board_concept_cons_em 需要板块名称 → 反向查: 遍历所有概念板块太慢
    # 改用 stock_individual_info_em 看能否拿到概念
    df = _safe(ak.stock_individual_info_em, symbol=code)
    if df is None or df.empty:
        return None
    # 该接口返回 item/value 结构
    if df.shape[1] >= 2:
        info = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        # 可能没有概念字段
        concept = info.get("概念") or info.get("所属概念") or info.get("概念板块")
        if concept and str(concept).strip():
            return [c.strip() for c in str(concept).split(",") if c.strip()]
    return None


def fetch_concept_pct(concept_name: str, target_date: str) -> float | None:
    """查询指定概念板块当日涨跌幅（东财）。"""
    df = _safe(
        ak.stock_board_concept_hist_em,
        symbol=concept_name,
        period="daily",
        start_date=_date_str(target_date),
        end_date=_date_str(target_date),
        adjust="",
    )
    if df is None or df.empty:
        return None
    df.columns = [c.strip() for c in df.columns]
    for col in ["涨跌幅", "change_pct", "pct_chg"]:
        if col in df.columns:
            return df.iloc[-1][col]
    return None


# ─── 分析函数 ───

def calc_avg_price_line(min_df: pd.DataFrame) -> pd.DataFrame:
    """
    均价线 = 截至该分钟累计成交额 / 累计成交量（手→股：×100）
    """
    df = min_df.copy()
    df["cum_amount"] = df["amount"].cumsum()
    df["cum_volume"] = df["volume"].cumsum() * 100  # 手→股
    df["avg_price"] = df["cum_amount"] / df["cum_volume"]
    return df


def analyze_price_vs_avg(min_df: pd.DataFrame) -> dict:
    """
    分析价格线 vs 均价线的位置关系。
    返回: above_ratio, tail30_trend, final_position
    """
    df = min_df.copy()
    above = df["close"] > df["avg_price"]
    below = df["close"] < df["avg_price"]

    total = len(df)
    above_cnt = above.sum()
    below_cnt = below.sum()
    above_ratio = above_cnt / total if total > 0 else 0

    # 全天站上/跌破/反复
    if above_ratio >= 0.8:
        day_position = f"全天站稳均价线上方（{above_ratio*100:.0f}%时间在上方）"
    elif above_ratio <= 0.2:
        day_position = f"全天跌破均价线下方（{above_ratio*100:.0f}%时间在上方）"
    else:
        day_position = f"价格线反复穿越均价线（{above_ratio*100:.0f}%时间在上方）"

    # 尾盘最后 30 分钟
    tail = df.tail(30)
    if len(tail) >= 2:
        tail_above_start = tail.iloc[0]["close"] > tail.iloc[0]["avg_price"]
        tail_above_end = tail.iloc[-1]["close"] > tail.iloc[-1]["avg_price"]
        tail_close_end = tail.iloc[-1]["close"]
        tail_avg_end = tail.iloc[-1]["avg_price"]
        gap = tail_close_end - tail_avg_end

        if tail_above_end and gap > 0:
            tail_trend = f"尾盘收于均价线上方（+{gap:.3f}元，强势）"
        elif not tail_above_end and gap < 0:
            tail_trend = f"尾盘跌破均价线（{gap:.3f}元，偏弱）"
        else:
            tail_trend = "尾盘收平均价线附近"
    else:
        tail_trend = "尾盘数据不足（<30根1分K）"

    return {
        "above_ratio": above_ratio,
        "day_position": day_position,
        "tail_trend": tail_trend,
    }


def analyze_candle(row: pd.Series) -> dict:
    """分析当日K线形态：阴/阳、收盘位置、光头光脚。"""
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    amp = h - l  # 振幅（绝对值，非百分比）

    candle_type = "阳线" if c >= o else "阴线"

    # 收盘在振幅区间的位置
    if amp > 0:
        close_pos = (c - l) / amp
        if close_pos >= 0.7:
            close_pos_label = f"收在高位（位置={close_pos:.1%}）"
        elif close_pos <= 0.3:
            close_pos_label = f"收在低位（位置={close_pos:.1%}）"
        else:
            close_pos_label = f"收在中位（位置={close_pos:.1%}）"
    else:
        close_pos = 0.5
        close_pos_label = "当日无振幅（一字板或停牌）"

    # 上下影线
    body = abs(c - o)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    shadow_desc = []
    if amp > 0:
        if upper_shadow / amp < 0.05:
            shadow_desc.append("光头（无上影）")
        elif upper_shadow / amp > 0.3:
            shadow_desc.append(f"长上影（{upper_shadow/amp:.1%}）")
        if lower_shadow / amp < 0.05:
            shadow_desc.append("光脚（无下影）")
        elif lower_shadow / amp > 0.3:
            shadow_desc.append(f"长下影（{lower_shadow/amp:.1%}）")
        if body / amp < 0.05:
            shadow_desc.append("十字星（实体极小）")

    if not shadow_desc:
        shadow_desc.append("有上下影线")

    return {
        "candle_type": candle_type,
        "close_pos_label": close_pos_label,
        "close_pos": close_pos,
        "shadow_desc": "、".join(shadow_desc),
        "upper_shadow_ratio": upper_shadow / amp if amp > 0 else 0,
    }


def analyze_volume(today_row: pd.Series, hist_df: pd.DataFrame) -> dict:
    """对比今日量能与前 VOLUME_WINDOW 日均值。"""
    # 前5日（不含今日）
    prev = hist_df[hist_df["date"] < today_row["date"]].tail(VOLUME_WINDOW)
    if len(prev) < 2:
        return {"volume_status": "前期数据不足，无法对比", "turnover_status": "前期数据不足"}

    avg_vol = prev["volume"].mean()
    avg_amt = prev["amount"].mean()
    avg_turn = prev["turnover"].mean() if "turnover" in prev.columns else None

    vol_ratio = today_row["volume"] / avg_vol if avg_vol > 0 else None
    amt_ratio = today_row["amount"] / avg_amt if avg_amt > 0 else None

    def _label(ratio):
        if ratio is None:
            return "无法判断"
        if ratio >= 2.0:
            return f"显著放量（{ratio:.1f}倍）"
        elif ratio >= 1.5:
            return f"温和放量（{ratio:.1f}倍）"
        elif ratio >= 0.8:
            return f"量能持平（{ratio:.1f}倍）"
        elif ratio >= 0.5:
            return f"缩量（{ratio:.1f}倍）"
        else:
            return f"显著缩量（{ratio:.1f}倍）"

    result = {
        "avg_vol_5d": avg_vol,
        "avg_amt_5d": avg_amt,
        "avg_turn_5d": avg_turn,
        "vol_ratio": vol_ratio,
        "amt_ratio": amt_ratio,
        "volume_status": _label(vol_ratio),
        "amount_status": _label(amt_ratio),
    }

    if avg_turn is not None:
        turn_ratio = today_row["turnover"] / avg_turn if avg_turn > 0 else None
        result["turn_ratio"] = turn_ratio
        result["turnover_status"] = _label(turn_ratio)
    else:
        result["turnover_status"] = "换手率数据未取到"

    return result


def judge_next_day(
    candle: dict,
    vol: dict,
    price_vs_avg: dict,
    index_pct: float | None,
    today_row: pd.Series,
    has_minute: bool,
) -> str:
    """
    基于技术规则输出次日概率倾向。
    注意：仅为技术规则机械推断，不是买卖建议。
    """
    signals_strong = []
    signals_weak = []

    # 1. 均价线位置
    if has_minute:
        if price_vs_avg["above_ratio"] >= 0.75:
            signals_strong.append("全天站稳均价线上方")
        elif price_vs_avg["above_ratio"] <= 0.3:
            signals_weak.append("全天跌破均价线下方")
        # 尾盘
        if "跌破" in price_vs_avg["tail_trend"]:
            signals_weak.append("尾盘跌破均价线")
        elif "上方" in price_vs_avg["tail_trend"] and "强势" in price_vs_avg["tail_trend"]:
            signals_strong.append("尾盘收于均价线上方")
    else:
        signals_strong.append("（分时数据未取到，均价线信号缺失）")

    # 2. 量能
    vol_ratio = vol.get("vol_ratio")
    if vol_ratio is not None:
        if vol_ratio >= 1.5:
            if candle["close_pos"] >= 0.7:
                signals_strong.append(f"放量（{vol_ratio:.1f}倍）收高位")
            elif candle["close_pos"] <= 0.4:
                signals_weak.append(f"放量（{vol_ratio:.1f}倍）但收中低位（高位放量滞涨信号）")
        elif vol_ratio < 0.6:
            signals_weak.append(f"成交量明显萎缩（{vol_ratio:.1f}倍）")

    # 3. K线形态
    pct = today_row.get("pct_chg", 0) or 0
    if candle["candle_type"] == "阳线":
        if candle["close_pos"] >= 0.7:
            signals_strong.append("阳线收高位")
            if candle["upper_shadow_ratio"] < 0.08:
                signals_strong.append("光头（无明显上影）")
        if candle["upper_shadow_ratio"] > 0.35:
            signals_weak.append(f"长上影线（{candle['upper_shadow_ratio']:.1%}），压力明显")
    else:
        if candle["close_pos"] <= 0.3:
            signals_weak.append("阴线收低位，弱势形态")
        else:
            signals_weak.append("收阴线")

    # 4. 大盘配合
    if index_pct is not None:
        if index_pct >= 0.5:
            signals_strong.append(f"大盘当日上涨（+{index_pct:.2f}%，有市场情绪支撑）")
        elif index_pct <= -0.5:
            signals_weak.append(f"大盘当日下跌（{index_pct:.2f}%，市场整体承压）")
        else:
            signals_strong.append(f"大盘基本持平（{index_pct:+.2f}%）")

    # 综合判断
    strong_n = len([s for s in signals_strong if "缺失" not in s and "不足" not in s])
    weak_n = len(signals_weak)

    if strong_n >= 3 and weak_n == 0:
        tendency = "偏多：多项强势信号共振，次日高开/惯性冲高概率偏高"
        coping = "持仓：正常持有，可设尾盘或次日均价线跌破为止损参考"
    elif strong_n >= 2 and weak_n <= 1:
        tendency = "中性偏多：强信号占优，但需警惕个别弱信号"
        coping = "持仓：轻仓持有，若次日低开或跌破均价线则减仓"
    elif weak_n >= 3:
        tendency = "偏空：多项转弱信号叠加，次日低开/承压概率偏高"
        coping = "持仓：建议减仓或严设止损，参考均价线或当日低点"
    elif weak_n >= 2 and strong_n <= 1:
        tendency = "中性偏空：弱信号偏多，需谨慎"
        coping = "持仓：控制仓位，次日若开盘弱势可逐步减仓"
    else:
        tendency = "中性：多空信号均衡，方向不明"
        coping = "持仓：维持仓位不动，观察次日开盘后首30分钟均价线方向再决策"

    lines = [tendency, ""]
    if signals_strong:
        lines.append("  [强势因素]")
        for s in signals_strong:
            lines.append(f"    ✦ {s}")
    if signals_weak:
        lines.append("  [转弱因素]")
        for s in signals_weak:
            lines.append(f"    ✧ {s}")
    lines.append("")
    lines.append(f"  [应对思路（方法论参考）] {coping}")
    return "\n".join(lines)


# ─── 主流程 ───

def analyze_stock(stock: dict, target_date: str, index_pct: float | None):
    code = stock["code"]
    name = stock["name"]

    print(f"\n{DIVIDER}")
    print(f"  {name}（{code}）  |  复盘日期: {_date_str(target_date)}")
    print(DIVIDER)

    # ── 1. 日K数据 ──
    print("\n[1] 日K数据（最近交易日）")
    print(THIN)
    hist_df = fetch_daily_kline(code, target_date)

    if hist_df is None or hist_df.empty:
        print("  ✗ 日K数据未取到（接口返回空或网络异常）")
        print(f"\n[原始数据核对区 — {name}（{code}）]")
        print("  日K数据未取到，无法核对。")
        return

    # 过滤到目标日及之前
    hist_df = hist_df[hist_df["date"] <= target_date]
    recent = hist_df.tail(HIST_DAYS)

    # 打印最近日K
    display_cols = ["date", "open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"]
    available_cols = [c for c in display_cols if c in recent.columns]
    print(f"  {'日期':>10} {'开':>7} {'高':>7} {'低':>7} {'收':>7} {'量(手)':>10} {'额(万元)':>12} {'涨跌幅':>8} {'换手率':>8}")
    for _, row in recent.iterrows():
        amt_wan = row["amount"] / 10000 if not pd.isna(row.get("amount")) else float("nan")
        print(
            f"  {row['date']:>10} "
            f"{_fmt(row.get('open')):>7} "
            f"{_fmt(row.get('high')):>7} "
            f"{_fmt(row.get('low')):>7} "
            f"{_fmt(row.get('close')):>7} "
            f"{int(row['volume']) if not pd.isna(row.get('volume')) else 'N/A':>10} "
            f"{_fmt(amt_wan, 0):>12} "
            f"{_pct(row.get('pct_chg')):>8} "
            f"{_fmt(row.get('turnover')):>7}%"
        )

    # 找目标日行
    today_rows = hist_df[hist_df["date"] == target_date]
    if today_rows.empty:
        print(f"\n  ✗ 目标日 {target_date} 在日K中未找到（可能非交易日或数据未更新）")
        today_row = None
    else:
        today_row = today_rows.iloc[0]
        print(f"\n  目标日 {target_date} 数据已取到 ✓")

    # ── 2. 分时数据 & 均价线 ──
    print(f"\n[2] 分时数据（1分钟K线）& 均价线")
    print(THIN)
    min_df = fetch_minute_data(code, target_date)

    has_minute = False
    price_vs_avg = {
        "above_ratio": None,
        "day_position": "分时数据未取到",
        "tail_trend": "分时数据未取到",
    }

    if min_df is None or min_df.empty:
        print(f"  ✗ {target_date} 分时数据未取到（免费接口覆盖窗口有限，或当日数据尚未更新）")
    else:
        has_minute = True
        min_df = calc_avg_price_line(min_df)
        price_vs_avg = analyze_price_vs_avg(min_df)

        # 打印关键时间点均价线
        checkpoints = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:00", "13:30", "14:00", "14:30", "14:57"]
        print(f"  {'时刻':>6} {'价格':>8} {'均价':>8} {'差值':>8}")
        for cp in checkpoints:
            mask = min_df["time"].dt.strftime("%H:%M") == cp
            if mask.any():
                r = min_df[mask].iloc[-1]
                diff = r["close"] - r["avg_price"]
                side = "↑上" if diff >= 0 else "↓下"
                print(f"  {cp:>6} {_fmt(r['close']):>8} {_fmt(r['avg_price']):>8} {_fmt(diff, 3):>8} ({side})")
        print(f"\n  全天均价线分析: {price_vs_avg['day_position']}")
        print(f"  尾盘30分钟走向: {price_vs_avg['tail_trend']}")

    # ── 3. 当日盘面形态（需 today_row） ──
    print(f"\n[3] 当日盘面形态")
    print(THIN)

    if today_row is None:
        print("  ✗ 无目标日数据，跳过形态分析")
        candle = None
        vol_analysis = None
    else:
        candle = analyze_candle(today_row)
        vol_analysis = analyze_volume(today_row, hist_df)

        print(f"  K线类型    : {candle['candle_type']}")
        print(f"  收盘位置   : {candle['close_pos_label']}")
        print(f"  影线特征   : {candle['shadow_desc']}")
        print(f"  开高低收   : {_fmt(today_row.get('open'))} / {_fmt(today_row.get('high'))} / {_fmt(today_row.get('low'))} / {_fmt(today_row.get('close'))}")
        print(f"  涨跌幅     : {_pct(today_row.get('pct_chg'))}")
        print(f"  振幅       : {_fmt(today_row.get('amplitude'))}%")
        print()
        print(f"  成交量(手) : {int(today_row['volume']) if not pd.isna(today_row.get('volume')) else '未取到'}")
        print(f"  成交额(万) : {_fmt(today_row['amount']/10000, 0) if not pd.isna(today_row.get('amount')) else '未取到'}")
        print(f"  换手率     : {_fmt(today_row.get('turnover'))}%")
        print(f"  量能状态   : {vol_analysis['volume_status']}")
        print(f"  成交额状态 : {vol_analysis['amount_status']}")
        print(f"  换手对比   : {vol_analysis['turnover_status']}")
        print(f"  前{VOLUME_WINDOW}日均量(手): {_fmt(vol_analysis.get('avg_vol_5d'), 0) if vol_analysis.get('avg_vol_5d') else '未取到'}")

    # ── 4. 大盘 & 板块 ──
    print(f"\n[4] 大盘 & 板块背景")
    print(THIN)
    if index_pct is not None:
        print(f"  上证指数当日涨跌幅: {_pct(index_pct)}")
    else:
        print("  上证指数当日涨跌幅: 该数据未取到")

    # 板块
    concepts = fetch_concept_boards(code)
    if concepts:
        print(f"  所属概念板块(前3): {', '.join(concepts[:3])}")
        # 取第一个概念的当日涨跌幅
        concept_results = []
        for concept in concepts[:3]:
            pct_val = fetch_concept_pct(concept, target_date)
            if pct_val is not None:
                concept_results.append(f"{concept}: {_pct(pct_val)}")
            else:
                concept_results.append(f"{concept}: 当日涨跌幅未取到")
        for cr in concept_results:
            print(f"    {cr}")
    else:
        print("  所属概念板块: 未取到")

    # ── 5. 次日概率倾向 ──
    print(f"\n[5] 次日概率倾向（技术规则推断，不构成投资建议）")
    print(THIN)
    if today_row is not None and candle is not None:
        judgment = judge_next_day(candle, vol_analysis, price_vs_avg, index_pct, today_row, has_minute)
        for line in judgment.split("\n"):
            print(f"  {line}")
    else:
        print("  目标日数据缺失，无法生成概率倾向分析")

    # ── 6. 原始数据核对区 ──
    print(f"\n{'#'*70}")
    print(f"  原始数据核对区 — {name}（{code}）  | 请与券商终端逐项核对")
    print(f"{'#'*70}")
    if today_row is not None:
        print(f"  日期    : {today_row['date']}")
        print(f"  开盘价  : {today_row.get('open', '未取到')}")
        print(f"  最高价  : {today_row.get('high', '未取到')}")
        print(f"  最低价  : {today_row.get('low', '未取到')}")
        print(f"  收盘价  : {today_row.get('close', '未取到')}")
        print(f"  成交量  : {int(today_row['volume']) if not pd.isna(today_row.get('volume')) else '未取到'} 手")
        amt_raw = today_row.get("amount")
        print(f"  成交额  : {amt_raw if pd.isna(amt_raw) else f'{amt_raw:.0f} 元'}")
        print(f"  换手率  : {today_row.get('turnover', '未取到')} %")
        print(f"  涨跌幅  : {today_row.get('pct_chg', '未取到')} %")
        print(f"  振幅    : {today_row.get('amplitude', '未取到')} %")
        print(f"  涨跌额  : {today_row.get('change', '未取到')} 元")
    else:
        print("  目标日数据未取到")
    print(f"{'#'*70}")


def main():
    target_date = TARGET_DATE or _today_str()

    print(DIVIDER)
    print("  A股复盘脚本 | 数据来源: akshare（东方财富等）")
    print("  本脚本不构成投资建议；数据可能有延迟，以券商终端为准。")
    print(f"  复盘日期: {_date_str(target_date)}   股票数: {len(STOCKS)}")
    print(DIVIDER)

    # 上证指数（只取一次）
    print("\n正在拉取上证指数数据...")
    index_pct = _safe(lambda: fetch_index_sh(target_date))
    if index_pct is None:
        print("  上证指数当日涨跌幅: 该数据未取到")
    else:
        print(f"  上证指数当日涨跌幅: {_pct(index_pct)}")

    # 逐只股票分析
    for stock in STOCKS:
        analyze_stock(stock, target_date, index_pct)

    print(f"\n{DIVIDER}")
    print("  复盘完成。以上数据请与券商终端核对，不构成投资建议。")
    print(DIVIDER)


if __name__ == "__main__":
    main()
