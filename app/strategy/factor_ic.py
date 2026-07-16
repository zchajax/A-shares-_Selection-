"""
策略层 - L2 因子 IC 检验(回答"这个因子到底有没有预测力")
=========================================================
IC(Information Coefficient,信息系数)是量化里检验因子有效性的黄金标准。

做法:在【历史每个截面日】,把全市场当日的【因子值排名】和【它之后 N 日的真实
收益排名】求 Spearman 秩相关,得到该日一个 IC 值。滚过所有截面日 → 一条 IC 序列。
用秩相关(RankIC)而非普通相关,是因为我们只关心"因子大的票是不是真的涨得多",
不关心线性程度,对极端值更稳健。

一条 IC 序列能回答的问题:
  · IC 均值:因子平均预测方向与强度。|IC|>0.03 算有效,>0.05 已相当不错(A股现实)。
  · IC_IR = IC均值/IC标准差:稳定性。IR 越高说明"每次都朝一个方向使劲",越可靠。
  · 正 IC 占比:多少比例的截面日方向预测对了(>55% 说明不是靠运气)。
  · t 值:IC 均值是否显著异于 0(|t|>2 约等于 95% 置信不是偶然)。

红线:严格用【当日因子】配【之后收益】,shift(-N) 保证方向,绝无前视;不编造。
主入口: compute_ic(fwd_days, step, progress_cb) -> (ic_summary_df, ic_series_dict)
"""
import numpy as np
import pandas as pd

from . import panel as pnl


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """两列的 Spearman 秩相关。样本<5 或某列无波动时返回 NaN。"""
    if len(a) < 5:
        return np.nan
    ra = a.rank()
    rb = b.rank()
    if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_ic(fwd_days: int = 5, step: int = None, panel: pd.DataFrame = None,
               progress_cb=None):
    """
    计算各技术因子的 IC 序列与汇总统计。

    fwd_days: 未来收益窗口(交易日)。
    step:     截面日采样间隔(交易日)。None=默认取 fwd_days(相邻窗口不重叠,更严谨)。
    panel:    已构建好的面板(避免重复构建);None 则内部现建。
    progress_cb: 进度回调(面板构建阶段)。

    返回 (summary_df, series_dict):
      summary_df: index=因子, 列= ic_mean / ic_ir / positive_ratio / t_stat / n_periods
      series_dict: {因子: pd.Series(index=截面日, values=当日IC)}  供画 IC 曲线
    """
    if step is None:
        step = fwd_days
    if panel is None:
        panel = pnl.build_panel(fwd_days=fwd_days, progress_cb=progress_cb)
    if panel is None or panel.empty:
        return pd.DataFrame(), {}

    dates = pnl.sample_rebalance_dates(panel, step)
    date_set = set(dates)

    series = {f: {} for f in pnl.TECH_FACTORS}
    for d, g in panel.groupby("date"):
        if d not in date_set:
            continue
        if len(g) < 10:  # 截面股票太少,IC 无意义
            continue
        for f in pnl.TECH_FACTORS:
            ic = _spearman(g[f], g["fwd_ret"])
            if not np.isnan(ic):
                series[f][d] = ic

    rows = {}
    series_out = {}
    for f in pnl.TECH_FACTORS:
        s = pd.Series(series[f]).sort_index()
        series_out[f] = s
        if len(s) == 0:
            rows[f] = dict(ic_mean=np.nan, ic_ir=np.nan,
                           positive_ratio=np.nan, t_stat=np.nan, n_periods=0)
            continue
        mu = s.mean()
        sd = s.std(ddof=1)
        n = len(s)
        ir = mu / sd if sd and sd != 0 else np.nan
        pos = float((s > 0).mean())
        # t = IC均值 / (IC标准差/根号n) = IR * 根号n
        t = mu / (sd / np.sqrt(n)) if sd and sd != 0 else np.nan
        rows[f] = dict(ic_mean=mu, ic_ir=ir, positive_ratio=pos,
                       t_stat=t, n_periods=n)

    summary = pd.DataFrame(rows).T
    summary = summary[["ic_mean", "ic_ir", "positive_ratio", "t_stat", "n_periods"]]
    # 中文因子名做 index,展示更友好
    summary.index = [pnl._FACTOR_CN.get(f, f) for f in summary.index]
    return summary, series_out
