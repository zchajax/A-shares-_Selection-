"""
策略层 - 历史回测引擎(组合模拟版)
========================================
把选股策略放到历史里检验:如果过去每天都按这个策略选股买入,
一段时间后到底是赚是亏?胜率多少?最大回撤多深?

【为什么是"组合模拟"而不是"逐笔全仓"】
  同一天策略可能同时命中几十只股票,现实中不可能每只都全仓。
  所以我们模拟一个真实账户:初始资金固定,最多同时持有 K 只(等权),
  资金用完就不再买,卖出后资金回笼可再买——这样得到的资金曲线和
  最大回撤才是可信的、能和真实交易对应的。

【核心设计:严格无未来函数】
  在历史第 t 天,只把 df.iloc[:t+1](截止当天)切片喂给策略,
  策略看到的"最新一天"就是第 t 天,决策绝不会用到 t 之后的数据。
  完全复用现有策略的 evaluate(),一行都不用改。

【交易规则】
  - 第 t 天收盘策略发出信号 → 第 t+1 天以【开盘价】买入
  - 每天盯市检查每个持仓:触止损→卖 / 触止盈→卖 / 持满N天→收盘卖
  - 最多同时持有 max_positions 只,超出的信号按打分高低取前几只
  - 已持有的股票不重复买入
  - 买卖各计一次成本(手续费+印花税+滑点,默认千三)

【产出】BacktestResult(trades 交易明细, equity 资金曲线, stats 汇总统计)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data import database as db
from . import indicators as ind
from .market import MarketTrend


@dataclass
class BacktestResult:
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: dict = field(default_factory=dict)


def _prepare(codes, strategy, min_bars, start_from, progress_cb):
    """
    预计算每只股票的行情数组、日期索引,以及"买入事件"。
    买入事件:策略在第 i 天收盘命中 → 记录在第 i+1 天(买入日)。
    返回:
      code_data[code] = dict(dates, opens, highs, lows, closes, d2i)
      buy_events[buy_date] = [ (score, code, buy_idx), ... ]
    """
    code_data = {}
    buy_events = {}
    total = len(codes)
    for n_done, code in enumerate(codes):
        raw = db.load_kline(code)
        if raw.empty or len(raw) < min_bars + 2:
            if progress_cb:
                progress_cb(n_done + 1, total)
            continue
        df = ind.enrich(raw).reset_index(drop=True)
        dates = df["date"].tolist()
        opens = df["open"].astype(float).values
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values
        code_data[code] = {
            "dates": dates, "opens": opens, "highs": highs,
            "lows": lows, "closes": closes,
            "d2i": {d: k for k, d in enumerate(dates)},
        }
        nlen = len(df)
        for i in range(min_bars, nlen - 1):     # 留出 i+1 作为买入日
            if start_from and dates[i] < start_from:
                continue
            try:
                selected, score, _ = strategy.evaluate(df.iloc[: i + 1])
            except Exception:
                selected = False
                score = 0
            if selected:
                buy_i = i + 1
                buy_date = dates[buy_i]
                buy_events.setdefault(buy_date, []).append((float(score), code, buy_i))
        if progress_cb:
            progress_cb(n_done + 1, total)
    return code_data, buy_events


def run_backtest(strategy, codes=None, hold_days=10, take_profit=0.15,
                 stop_loss=0.08, cost=0.003, max_positions=5,
                 initial_capital=100000.0, min_bars=65, start_from=None,
                 progress_cb=None, market_filter=False,
                 index_code="sh000001") -> BacktestResult:
    """
    组合模拟回测。

    参数:
      strategy        已配置参数的策略实例
      codes           回测哪些股票,None=本地全部缓存
      hold_days       最长持有天数(交易日)
      take_profit     止盈幅度(0.15=+15%),0=不设
      stop_loss       止损幅度(0.08=-8%),0=不设
      cost            单边交易成本(0.003=千三)
      max_positions   最多同时持有几只(等权)
      initial_capital 初始资金
      min_bars        起始所需最少K线(保证指标算得出)
      start_from      只回测该日期后的信号,None=全部历史
      progress_cb     progress_cb(done,total) 进度回调(用于预计算阶段)
      market_filter   True=启用大盘趋势过滤,大盘走弱的交易日不开仓
      index_code      大盘指数代码(默认上证综指)
    """
    if codes is None:
        codes = db.list_cached_codes()
    namemap = dict(zip(db.load_stock_list()["code"], db.load_stock_list()["name"]))

    # 大盘趋势过滤器(可选)。未启用或指数无数据时不影响原有逻辑。
    mkt = MarketTrend(index_code) if market_filter else None

    code_data, buy_events = _prepare(codes, strategy, min_bars, start_from, progress_cb)
    if not code_data:
        return _summarize(pd.DataFrame(), pd.DataFrame(), initial_capital,
                          hold_days, take_profit, stop_loss)

    # 主时间轴:所有股票日期的并集,升序
    all_dates = sorted({d for cd in code_data.values() for d in cd["dates"]})

    cash = initial_capital
    positions = {}     # code -> dict(shares, buy_price, buy_idx, tp, sl, buy_date, name)
    trades = []
    equity_curve = []

    for today in all_dates:
        # ---- 1) 先处理卖出(盯市检查每个持仓) ----
        for code in list(positions.keys()):
            cd = code_data[code]
            idx = cd["d2i"].get(today)
            if idx is None:
                continue    # 该股今天没有交易数据(停牌等)
            pos = positions[code]
            days_held = idx - pos["buy_idx"]
            if days_held <= 0:
                continue    # 买入当天不卖
            high, low, close = cd["highs"][idx], cd["lows"][idx], cd["closes"][idx]
            sell_price = None
            reason = None
            # 保守:同一天先判止损
            if pos["sl"] is not None and low <= pos["sl"]:
                sell_price, reason = pos["sl"], "止损"
            elif pos["tp"] is not None and high >= pos["tp"]:
                sell_price, reason = pos["tp"], "止盈"
            elif days_held >= hold_days:
                sell_price, reason = close, f"持有{hold_days}天到期"
            if sell_price is not None:
                proceeds = pos["shares"] * sell_price * (1 - cost)
                cash += proceeds
                gross = (sell_price - pos["buy_price"]) / pos["buy_price"]
                net = gross - 2 * cost
                trades.append({
                    "code": code, "name": pos["name"],
                    "buy_date": pos["buy_date"], "sell_date": today,
                    "buy_price": round(pos["buy_price"], 2),
                    "sell_price": round(float(sell_price), 2),
                    "hold": days_held,
                    "return_pct": round(net * 100, 2),
                    "reason": reason,
                })
                del positions[code]

        # ---- 2) 再处理买入(用今天开盘价买入昨日收盘发出的信号) ----
        # 大盘趋势过滤:大盘走弱的交易日不开新仓(已有持仓照常按卖出规则处理)
        market_ok = (mkt.is_strong(today) if mkt is not None else True)
        cands = buy_events.get(today, [])
        if market_ok and cands and len(positions) < max_positions:
            cands = sorted(cands, key=lambda x: x[0], reverse=True)  # 打分高优先
            for score, code, buy_i in cands:
                if len(positions) >= max_positions:
                    break
                if code in positions:
                    continue
                cd = code_data[code]
                open_price = cd["opens"][buy_i]
                if not np.isfinite(open_price) or open_price <= 0:
                    continue
                free = max_positions - len(positions)
                alloc = cash / free            # 把可用现金按剩余空位等分
                if alloc <= 1:
                    break
                shares = (alloc * (1 - cost)) / open_price
                cash -= alloc
                positions[code] = {
                    "shares": shares, "buy_price": open_price, "buy_idx": buy_i,
                    "tp": open_price * (1 + take_profit) if take_profit > 0 else None,
                    "sl": open_price * (1 - stop_loss) if stop_loss > 0 else None,
                    "buy_date": today, "name": namemap.get(code, ""),
                }

        # ---- 3) 盯市:账户总权益 = 现金 + 持仓市值(按当日收盘) ----
        holding_value = 0.0
        for code, pos in positions.items():
            idx = code_data[code]["d2i"].get(today)
            px = code_data[code]["closes"][idx] if idx is not None else pos["buy_price"]
            holding_value += pos["shares"] * px
        equity_curve.append({"date": today, "equity": cash + holding_value})

    equity = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    return _summarize(trades_df, equity, initial_capital,
                      hold_days, take_profit, stop_loss)


def _summarize(trades, equity, initial_capital, hold_days, take_profit, stop_loss):
    """汇总统计。资金曲线的收益/回撤来自 equity(真实账户),胜率等来自 trades。"""
    if trades.empty:
        return BacktestResult(
            trades=trades, equity=equity,
            stats={"n_trades": 0, "note": "无交易信号,可放宽策略参数,或扩大回测样本/时间范围"},
        )

    trades = trades.sort_values("sell_date").reset_index(drop=True)
    wins = trades[trades["return_pct"] > 0]
    losses = trades[trades["return_pct"] <= 0]
    n = len(trades)
    win_rate = len(wins) / n * 100
    avg_ret = trades["return_pct"].mean()
    avg_win = wins["return_pct"].mean() if len(wins) else 0.0
    avg_loss = losses["return_pct"].mean() if len(losses) else 0.0
    profit_factor = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("inf")

    # 账户层面:总收益 & 最大回撤(来自真实资金曲线)
    if not equity.empty:
        eq = equity["equity"].values
        total_return = (eq[-1] / initial_capital - 1) * 100
        running_max = np.maximum.accumulate(eq)
        drawdown = (eq - running_max) / running_max
        max_dd = drawdown.min() * 100
        # 年化(按交易日 244 计)
        days = len(eq)
        years = max(days / 244.0, 1e-6)
        cagr = ((eq[-1] / initial_capital) ** (1 / years) - 1) * 100 if eq[-1] > 0 else -100.0
    else:
        total_return = max_dd = cagr = 0.0

    stats = {
        "n_trades": n,
        "win_rate": round(win_rate, 1),
        "avg_ret": round(avg_ret, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else 999.0,
        "total_return": round(total_return, 1),
        "cagr": round(cagr, 1),
        "max_dd": round(max_dd, 1),
        "best": round(trades["return_pct"].max(), 2),
        "worst": round(trades["return_pct"].min(), 2),
        "avg_hold": round(trades["hold"].mean(), 1),
        "hold_days": hold_days,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
    }
    return BacktestResult(trades=trades, equity=equity, stats=stats)
