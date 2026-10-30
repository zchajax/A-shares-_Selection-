"""
因子 RankIC。
t_stat 是 Newey-West t，不是把每期 IC 当成独立样本的 t。
相关的是处理后的因子（去极值、行业去均值）和开盘到开盘的可成交毛收益。
"""
import numpy as np
import pandas as pd

from . import panel as pnl
from . import protocol as proto


def _spearman(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if int(m.sum()) < proto.MIN_NAMES:
        return np.nan
    ra = a[m].rank()
    rb = b[m].rank()
    if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def prepare_sample(panel: pd.DataFrame, step: int, factors=None) -> pd.DataFrame:
    """先取不重叠的调仓日，再只在这些截面上做因子处理。"""
    if panel is None or panel.empty:
        return pd.DataFrame()
    factors = list(factors or proto.TECH_FACTORS)
    dates = set(pnl.sample_rebalance_dates(panel, step))
    if not dates:
        return pd.DataFrame()
    sub = panel.loc[panel["date"].astype(str).isin(dates)].copy()
    return proto.attach_processed(sub, factors)


def ic_from_prepared(sub: pd.DataFrame, factors=None):
    """sub 已经是调仓日截面，且含 *_p 列。"""
    factors = list(factors or proto.TECH_FACTORS)
    empty = pd.DataFrame(columns=["ic_mean", "ic_ir", "positive_ratio", "t_stat", "n_periods"])
    if sub is None or sub.empty:
        return empty, {}
    series = {f: {} for f in factors}
    for d, g in sub.groupby("date", sort=False):
        for f in factors:
            ic = _spearman(g[f + "_p"], g["fwd_ret"])
            if np.isfinite(ic):
                series[f][str(d)] = ic
    rows = {}
    series_out = {}
    for f in factors:
        s = pd.Series(series[f], dtype=float).sort_index()
        series_out[f] = s
        if len(s) == 0:
            rows[f] = dict(ic_mean=np.nan, ic_ir=np.nan,
                           positive_ratio=np.nan, t_stat=np.nan, n_periods=0, nw_lags=0)
            continue
        mu, t, lags = proto.newey_west_mean(s.to_numpy())
        sd = float(s.std(ddof=1)) if len(s) > 1 else np.nan
        ir = mu / sd if sd and np.isfinite(sd) and sd != 0 else np.nan
        rows[f] = dict(
            ic_mean=mu,
            ic_ir=ir,
            positive_ratio=float((s > 0).mean()),
            t_stat=t,
            n_periods=int(len(s)),
            nw_lags=int(lags),
        )
    summary = pd.DataFrame(rows).T
    summary = summary[["ic_mean", "ic_ir", "positive_ratio", "t_stat", "n_periods", "nw_lags"]]
    summary.index = [pnl._FACTOR_CN.get(f, f) for f in summary.index]
    return summary, series_out


def compute_ic(fwd_days: int = 5, step: int = None, panel: pd.DataFrame = None,
               progress_cb=None):
    """与旧接口兼容。t_stat 列为 Newey-West t。"""
    if step is None:
        step = fwd_days
    if panel is None:
        panel = pnl.build_panel(fwd_days=fwd_days, progress_cb=progress_cb)
    sub = prepare_sample(panel, step)
    return ic_from_prepared(sub)
