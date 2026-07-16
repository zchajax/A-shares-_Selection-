"""
策略层 - L2 分位分组回测(回答"照这套横截面打分选票,组合到底赚不赚钱")
==============================================================
注意:本模块与 backtest.py 是两种不同的回测,各司其职、互不替代:
  · backtest.py(run_backtest) —— "组合模拟":拿【单一策略的 evaluate】逐日选股,
    模拟真实账户资金曲线(止盈止损/最多持仓K只/含成本)。面向"一个策略实盘像不像"。
  · quantile_bt.py(本模块)   —— "分位分组体检":拿【横截面合成因子】把全市场分5档,
    检验强档是否稳定跑赢弱档。面向"这套多因子打分有没有选股能力"。是 L2 配套件。

IC 检验回答"因子有没有预测力",但预测力≠真金白银。分组回测直接模拟:
每期按合成因子把全市场分5档,买最强档(Q5)等权持有到下期,看净值曲线。

做法(每个调仓日):
  1. 取当日全市场技术因子,winsorize+z-score(与L1同口径),加权合成总分。
  2. 按总分把股票分成 n_groups 组(Q1最弱 … Qn最强)。
  3. 每组等权持有到下一调仓日,组合收益=组内成员未来收益均值。
  4. 逐期累乘 → 各组净值 + 多空组合(Qn-Q1)净值。

多空组合(Qn-Q1)剥离了大盘涨跌(同期多空对冲),只留"强档vs弱档"纯 alpha:
  长期向上=打分确实能区分强弱;走平/向下=这套权重历史上没选股能力。

诚实边界:
  · 只用有真实历史序列的技术因子(动量/趋势/量能),同 panel/factor_ic。
  · 等权、不计交易成本/滑点/停牌——这是"因子能力体检",非实盘收益承诺。
    成本会侵蚀高换手策略收益,实盘务必打折看。绝不编造。
主入口: run_quantile_backtest(weights, fwd_days, n_groups, ...) -> dict
"""
import numpy as np
import pandas as pd

from . import panel as pnl


def _winsorize_z(s: pd.Series, n: float = 3.0) -> pd.Series:
    """去极值 + 标准化(与 cross_section 口径一致)。无波动时返回全 0。"""
    x = s.astype(float)
    mu, sd = x.mean(), x.std(ddof=0)
    if np.isfinite(sd) and sd != 0:
        x = x.clip(mu - n * sd, mu + n * sd)
    mu, sd = x.mean(), x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (x - mu) / sd


# 回测默认权重(与 L1 技术因子部分对齐,已归一到技术三因子)
DEFAULT_TECH_WEIGHTS = {"momentum": 40.0, "trend": 40.0, "volume": 20.0}


def run_quantile_backtest(weights: dict = None, fwd_days: int = 5,
                          n_groups: int = 5, step: int = None,
                          panel: pd.DataFrame = None, progress_cb=None) -> dict:
    """
    分位分组回测。

    weights:  技术因子权重(momentum/trend/volume);None=DEFAULT_TECH_WEIGHTS。
    fwd_days: 持有 / 未来收益窗口(交易日)。
    n_groups: 分组数(默认5=quintile)。
    step:     调仓间隔(交易日)。None=fwd_days(不重叠,无前视)。
    panel:    复用已建面板;None 则内部建。

    返回 dict:
      group_curves: {组名: pd.Series(净值, index=调仓日)}  Q1..Qn + 多空
      metrics:      {组名: {ann_return, sharpe, max_drawdown, total_return}}
      periods:      调仓期数
      period_days:  每期交易日数(=fwd_days)
      monotonic:    分组年化收益是否随档位单调递增
    """
    w = dict(DEFAULT_TECH_WEIGHTS)
    if weights:
        # L3 支持带符号权重:允许传入负权重(反向使用该因子)。
        # 若外部只想改部分因子,未提及的沿用默认;显式传 0 = 关闭该因子。
        for k in list(w.keys()):
            if k in weights and weights[k] is not None:
                w[k] = float(weights[k])
    # 入选=权重非 0(正=顺势,负=反向)。全为 0 时回退默认,避免空合成。
    active = {k: v for k, v in w.items() if v != 0} or dict(DEFAULT_TECH_WEIGHTS)
    if step is None:
        step = fwd_days

    if panel is None:
        panel = pnl.build_panel(fwd_days=fwd_days, progress_cb=progress_cb)
    if panel is None or panel.empty:
        return {"group_curves": {}, "metrics": {}, "periods": 0,
                "period_days": fwd_days, "monotonic": False}

    dates = pnl.sample_rebalance_dates(panel, step)
    date_set = set(dates)
    # 带符号权重:归一化用【绝对值之和】,否则正负相抵会把分母缩小甚至归零。
    # 合成时 z * (带符号权重/Σ|权重|):负权重自动实现"该因子越小得分越高"。
    wsum = sum(abs(v) for v in active.values()) or 1.0

    period_rows = []
    used_dates = []
    for d, g in panel.groupby("date"):
        if d not in date_set or len(g) < n_groups * 4:
            continue
        comp = None
        for f, wt in active.items():
            z = _winsorize_z(g[f]) * (wt / wsum)
            comp = z if comp is None else comp + z
        gg = g.assign(_score=comp.values)
        try:
            gg["_grp"] = pd.qcut(gg["_score"].rank(method="first"),
                                 n_groups, labels=False)
        except ValueError:
            continue
        row = {"date": d}
        for q in range(n_groups):
            members = gg[gg["_grp"] == q]
            row[f"Q{q + 1}"] = members["fwd_ret"].mean()
        period_rows.append(row)
        used_dates.append(d)

    if not period_rows:
        return {"group_curves": {}, "metrics": {}, "periods": 0,
                "period_days": fwd_days, "monotonic": False}

    ret_df = pd.DataFrame(period_rows).set_index("date").sort_index()
    ls_name = "多空(Q%d-Q1)" % n_groups
    ret_df[ls_name] = ret_df[f"Q{n_groups}"] - ret_df["Q1"]

    group_curves = {}
    for col in ret_df.columns:
        nav = (1.0 + ret_df[col].fillna(0.0)).cumprod()
        group_curves[col] = nav

    periods_per_year = 244.0 / fwd_days
    metrics = {}
    for col, nav in group_curves.items():
        r = ret_df[col].dropna()
        n = len(r)
        if n == 0:
            continue
        total = nav.iloc[-1] - 1.0
        mean_p = r.mean()
        std_p = r.std(ddof=1)
        ann = (1.0 + mean_p) ** periods_per_year - 1.0
        sharpe = (mean_p / std_p * np.sqrt(periods_per_year)
                  if std_p and std_p != 0 else np.nan)
        peak = nav.cummax()
        dd = (nav / peak - 1.0).min()
        metrics[col] = {"ann_return": ann, "sharpe": sharpe,
                        "max_drawdown": float(dd), "total_return": total}

    q_anns = [metrics.get(f"Q{q + 1}", {}).get("ann_return", np.nan)
              for q in range(n_groups)]
    monotonic = all(q_anns[i] <= q_anns[i + 1]
                    for i in range(len(q_anns) - 1)
                    if not (np.isnan(q_anns[i]) or np.isnan(q_anns[i + 1])))

    return {"group_curves": group_curves, "metrics": metrics,
            "periods": len(used_dates), "period_days": fwd_days,
            "n_groups": n_groups, "monotonic": monotonic,
            "ls_name": ls_name, "dates": used_dates}
