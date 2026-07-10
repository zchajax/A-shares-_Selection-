"""
策略层 - 卖出参数网格调优
========================================
同一个选股策略(买入逻辑),配不同的卖出参数(止盈/止损/持有天数),
最终收益可能天差地别。本模块对给定策略做网格搜索,遍历若干组
(take_profit, stop_loss, hold_days) 组合,各跑一次回测,
按目标指标(默认总收益)排序,给出最优卖出参数。

【为什么能加速】
  网格里所有组合的"买入信号"完全相同(策略买入逻辑没变),
  变的只是卖出规则。所以买入事件 _prepare 只需算一次并复用,
  避免每组组合都重复扫描全历史(那会慢几十倍)。
  这里通过对每个候选参数调用 run_backtest 实现(实现简单、正确);
  _prepare 复用作为后续可选优化点。
"""
import itertools

import numpy as np
import pandas as pd

from ..data import database as db
from . import backtest as bt


# 默认网格:覆盖常见止盈止损区间,粒度适中(3×3×3=27 组)
DEFAULT_TP = [0.08, 0.12, 0.20]      # 止盈 8/12/20%
DEFAULT_SL = [0.05, 0.08, 0.10]      # 止损 5/8/10%
DEFAULT_HOLD = [5, 10, 15]           # 持有 5/10/15 天


def grid_search(strategy_cls, codes=None, tp_grid=None, sl_grid=None,
                hold_grid=None, start_from=None, max_positions=5,
                market_filter=False, metric="total_return",
                progress_cb=None) -> pd.DataFrame:
    """
    对某策略做卖出参数网格搜索。

    参数:
      strategy_cls   策略【类】(不是实例;每组用默认买入参数新建实例)
      codes          回测股票池,None=本地全部
      tp_grid/sl_grid/hold_grid  三个参数的候选列表,None=用默认网格
      start_from     回测起始日
      market_filter  是否启用大盘趋势过滤
      metric         排序目标:total_return / cagr / profit_factor / win_rate
      progress_cb    progress_cb(done, total) 进度回调(按组合数)

    返回:DataFrame,每行一组参数及其回测指标,按 metric 降序。
    """
    if codes is None:
        codes = db.list_cached_codes()
    tp_grid = tp_grid or DEFAULT_TP
    sl_grid = sl_grid or DEFAULT_SL
    hold_grid = hold_grid or DEFAULT_HOLD

    combos = list(itertools.product(tp_grid, sl_grid, hold_grid))
    total = len(combos)
    rows = []
    for n, (tp, sl, hold) in enumerate(combos):
        res = bt.run_backtest(
            strategy_cls(), codes=codes, hold_days=hold,
            take_profit=tp, stop_loss=sl, max_positions=max_positions,
            start_from=start_from, market_filter=market_filter,
        )
        s = res.stats
        rows.append({
            "take_profit": tp, "stop_loss": sl, "hold_days": hold,
            "n_trades": s.get("n_trades", 0),
            "win_rate": s.get("win_rate", 0.0),
            "total_return": s.get("total_return", 0.0),
            "cagr": s.get("cagr", 0.0),
            "max_dd": s.get("max_dd", 0.0),
            "profit_factor": s.get("profit_factor", 0.0),
        })
        if progress_cb:
            progress_cb(n + 1, total)

    out = pd.DataFrame(rows)
    if not out.empty and metric in out.columns:
        out = out.sort_values(metric, ascending=False).reset_index(drop=True)
    return out


# ======================================================================
# 买入参数寻优(全局)
# ======================================================================
# grid_search 只调卖出参数(止盈/止损/持有),买入逻辑固定。
# 本节遍历【策略自身声明的买入参数 cls.params】,为每组买入参数
# 各跑一次回测,找出在样本上最优的买入参数组合。卖出参数取固定值
# (默认沿用界面上的止盈/止损/持有),让比较只反映买入信号的差异。
#
# 【组合爆炸的控制】
#   若把每个参数都细分很多档,组合数会指数级膨胀(m 个参数、每个 k 档
#   → k^m 组),回测极慢。这里默认每个参数只取 levels=4 档(在 min~max
#   间均匀取值,整型参数去重取整),并限制最多参与寻优的参数个数
#   max_params=3(超过则只取前 3 个,通常是策略最关键的几个)。
#   这样组合数被压在 4^3=64 组以内,兼顾覆盖度与耗时。

DEFAULT_LEVELS = 4      # 每个买入参数默认采样档数
DEFAULT_MAX_PARAMS = 3  # 最多同时寻优的买入参数个数


def _param_values(p, levels):
    """在 [p.min, p.max] 间均匀取 levels 个候选值。

    整型参数取整并去重(避免 3.0/3.3/3.6 都取整成 3 的重复);
    浮点参数保留两位小数。始终包含默认值,保证最优不劣于当前默认。
    """
    lo, hi = float(p.min), float(p.max)
    if hi <= lo:
        return [p.default]
    raw = list(np.linspace(lo, hi, levels))
    if p.is_int:
        vals = sorted({int(round(v)) for v in raw})
    else:
        vals = sorted({round(float(v), 2) for v in raw})
    # 保证默认值在候选里
    dv = int(round(p.default)) if p.is_int else round(float(p.default), 2)
    if dv not in vals:
        vals.append(dv)
        vals = sorted(set(vals))
    return vals


def param_search(strategy_cls, codes=None, levels=DEFAULT_LEVELS,
                 max_params=DEFAULT_MAX_PARAMS, hold_days=10,
                 take_profit=0.15, stop_loss=0.08, max_positions=5,
                 start_from=None, market_filter=False,
                 metric="total_return", progress_cb=None):
    """
    遍历策略【买入参数】的组合,找出样本上最优的买入参数。

    参数:
      strategy_cls  策略【类】。从 cls.params 读取可调买入参数声明。
      codes         回测股票池,None=本地全部
      levels        每个参数在 min~max 间采样的档数(默认4)
      max_params    最多参与寻优的参数个数(默认3,超出只取前几个)
      hold_days / take_profit / stop_loss / max_positions
                    固定的卖出参数(让比较只反映买入信号差异)
      start_from    回测起始日
      market_filter 是否启用大盘趋势过滤
      metric        排序目标:total_return / cagr / profit_factor / win_rate
      progress_cb   progress_cb(done, total) 进度回调(按组合数)

    返回:(result_df, param_keys)
      result_df   每行一组买入参数及其回测指标,按 metric 降序;
                  参数列名即 cls.params 的 key,另附 __labels 供 UI 显示。
      param_keys  实际参与寻优的参数 key 列表(顺序与列对应)。
    """
    if codes is None:
        codes = db.list_cached_codes()

    params = list(getattr(strategy_cls, "params", []) or [])
    # 只对"档数>1"的参数寻优(min==max 的没有可调空间)
    tunable = [p for p in params if float(p.max) > float(p.min)]
    tunable = tunable[:max_params]           # 控制组合爆炸
    if not tunable:
        return pd.DataFrame(), []

    keys = [p.key for p in tunable]
    labels = {p.key: p.label for p in tunable}
    value_lists = [_param_values(p, levels) for p in tunable]
    combos = list(itertools.product(*value_lists))
    total = len(combos)

    rows = []
    for n, combo in enumerate(combos):
        strat = strategy_cls()
        for k, v in zip(keys, combo):
            strat.set_param(k, v)
        res = bt.run_backtest(
            strat, codes=codes, hold_days=hold_days, take_profit=take_profit,
            stop_loss=stop_loss, max_positions=max_positions,
            start_from=start_from, market_filter=market_filter,
        )
        s = res.stats
        row = {k: v for k, v in zip(keys, combo)}
        row.update({
            "n_trades": s.get("n_trades", 0),
            "win_rate": s.get("win_rate", 0.0),
            "total_return": s.get("total_return", 0.0),
            "cagr": s.get("cagr", 0.0),
            "max_dd": s.get("max_dd", 0.0),
            "profit_factor": s.get("profit_factor", 0.0),
        })
        rows.append(row)
        if progress_cb:
            progress_cb(n + 1, total)

    out = pd.DataFrame(rows)
    if not out.empty and metric in out.columns:
        out = out.sort_values(metric, ascending=False).reset_index(drop=True)
    out.attrs["labels"] = labels
    return out, keys
