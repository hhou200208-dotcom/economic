"""
A股短线情绪与题材强度监控系统 — 主入口。

用法:
    python src/main.py                  # 今天
    python src/main.py --date 20260617  # 指定日期
    python src/main.py --force          # 强制刷新，忽略缓存
"""

import sys
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import setup_logger, today_str, fmt_date, load_config, yi
from src.data_fetcher import DataFetcher
from src.emotion_score import EmotionScorer

console = Console()


# ── 输出函数 ─────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 80:
        return "bold red"
    elif score >= 60:
        return "bold yellow"
    elif score >= 40:
        return "bold cyan"
    else:
        return "bold blue"


def print_data_status(data: dict) -> None:
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("接口", style="cyan")
    table.add_column("状态")
    for key, df in data.items():
        if df is not None:
            table.add_row(key, f"[green]✓ {len(df)} 行[/]")
        else:
            table.add_row(key, "[red]✗ 未获取（接口失败且无缓存）[/]")
    console.print(table)


def print_emotion_result(result: dict) -> None:
    score = result["emotion_score"]
    color = _score_color(score)

    table = Table(
        title=f"市场情绪评分  —  {fmt_date(result['date'])}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white on dark_blue",
    )
    table.add_column("指标", style="cyan", width=18)
    table.add_column("数值", justify="right", width=14)
    table.add_column("说明", style="dim", width=28)

    rows = [
        ("情绪总分(0-100)", f"[{color}]{score}[/]", result["status"]),
        ("─" * 16, "─" * 10, ""),
        ("涨停数",    str(result["limit_up_count"]),   f"基准={150}"),
        ("跌停数",    str(result["limit_down_count"]),  f"基准={80}"),
        ("炸板数",    str(result["failed_board_count"]), ""),
        ("炸板率",    f"{result['failed_board_rate']}%", ""),
        ("最高连板",  f"{result['max_board_height']} 板", f"基准={8}"),
        ("20cm涨停", str(result["twenty_cm_count"]),   f"基准={30}"),
        ("─" * 16, "─" * 10, ""),
        ("上涨家数",  str(result["up_count"]),   ""),
        ("下跌家数",  str(result["down_count"]),  ""),
        ("涨幅>5%",  str(result["up_5pct_count"]),  ""),
        ("跌幅>5%",  str(result["down_5pct_count"]), ""),
        ("─" * 16, "─" * 10, ""),
        ("今日成交额", yi(result["total_amount_yi"] * 1e8), ""),
        (
            "5日均成交额",
            yi(result["amount_5d_avg_yi"] * 1e8) if result.get("amount_5d_avg_yi") else "N/A（历史数据不足）",
            "",
        ),
        (
            "成交额比值",
            str(result.get("amount_ratio", "N/A")),
            "1.0=持平 >1放量 <1缩量",
        ),
    ]

    for label, val, desc in rows:
        table.add_row(label, val, desc)

    console.print()
    console.print(table)

    # 分项得分明细
    detail = Table(title="分项得分明细", box=box.SIMPLE, show_header=True)
    detail.add_column("分项", style="cyan")
    detail.add_column("得分", justify="right")
    detail.add_column("满分", justify="right")

    detail.add_row("涨停数",   str(result["_score_lu"]),    "25")
    detail.add_row("连板高度", str(result["_score_board"]), "15")
    detail.add_row("20cm涨停", str(result["_score_20cm"]),  "10")
    detail.add_row("成交额比", str(result["_score_amt"]),   "15")
    detail.add_row("涨跌比",   str(result["_score_updn"]),  "15")
    detail.add_row("—跌停扣分", f"-{result['_penalty_ld']}", "-10")
    detail.add_row("—炸板扣分", f"-{result['_penalty_fb']}", "-10")
    detail.add_row(
        "[bold]合计[/]",
        f"[bold]{result['emotion_score']}[/]",
        "[bold]100[/]",
    )
    console.print(detail)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(date: str = None, force_refresh: bool = False) -> None:
    cfg = load_config(ROOT / "config.yaml")
    setup_logger(log_dir=ROOT / cfg["dirs"]["logs"])

    if date is None:
        date = today_str()

    console.print(
        Panel(
            f"[bold]A股短线情绪监控系统[/bold]\n"
            f"复盘日期: [cyan]{fmt_date(date)}[/]  |  "
            f"数据目录: [dim]{ROOT / cfg['dirs']['data']}[/]",
            style="blue",
        )
    )

    # ── Phase 1: 数据拉取 ────────────────────────────────────────────────────
    fetch_cfg = cfg.get("fetch", {})
    fetcher = DataFetcher(
        data_dir=str(ROOT / cfg["dirs"]["data"]),
        request_gap=fetch_cfg.get("request_gap", 1.5),
        retry_attempts=fetch_cfg.get("retry_attempts", 3),
        retry_delay=fetch_cfg.get("retry_delay", 2.0),
    )

    console.rule("[bold]Phase 1 — 数据拉取[/]")
    data = fetcher.fetch_all(date, force_refresh=force_refresh)
    print_data_status(data)

    # ── Phase 2: 情绪评分 ────────────────────────────────────────────────────
    console.rule("[bold]Phase 2 — 市场情绪评分[/]")
    scorer = EmotionScorer(
        data_dir=str(ROOT / cfg["dirs"]["data"]),
        cfg=cfg,
    )
    result = scorer.compute(
        date=date,
        a_spot=data["a_spot"],
        zt_pool=data["zt_pool"],
        dt_pool=data["dt_pool"],
        zb_pool=data["zb_pool"],
    )
    scorer.save(result, date)
    print_emotion_result(result)

    console.print(
        f"\n[dim]数据已保存至 {ROOT / cfg['dirs']['data'] / date}/  "
        f"（market_summary.csv + 各池 CSV）[/]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股情绪监控系统")
    parser.add_argument("--date",  type=str, default=None,  help="交易日期 YYYYMMDD，默认今天")
    parser.add_argument("--force", action="store_true",     help="强制刷新，忽略本地缓存")
    args = parser.parse_args()
    main(date=args.date, force_refresh=args.force)
