"""
策略层 - L2 因子面板构建(IC检验 + 分组回测的共享地基)
======================================================
把全市场历史日线摊平成一张"长面板":每只票的每个交易日一行,带上当日的
技术因子值 + 未来 N 日收益(fwd_ret)。IC 检验和分组回测都吃这同一张表。

为什么是"面板"而不是 L1 那种"只取最后一根K线":
  L1 回答"此刻全市场谁排第一",只需当前截面。
  L2 要回答"这些因子过去到底有没有预测力",必须把时间轴铺开,
  在【历史每个截面日】把因子值和它【之后】的真实涨跌配对,才能算相关性。

诚实边界(必须明确,不可绕过):
  · 技术因子(动量/趋势/量能)——close/volume 的历史逐日都在本地,有真实时序,
    可做时序 IC 与回测。
  · 基本面因子(PE/ROE)——本地只有【最新一期快照】,没有历史序列。把当前值
    广播回历史 = 用"未来才知道的估值"去解释过去 = 前视偏差,结论会虚高。
    故 L2 时序检验【不纳入】基本面因子。它们的有效性靠 L1 截面逻辑,
    待未来积累历史基本面后再单独验证。
红线:全程本地量化、可复现,绝不编造历史基本面。
"""
import numpy as np
import pandas as pd

from ..data import database as db
from . import indicators as ind


# L2 时序检验/回测纳入的因子(仅有真实历史序列的技术因子)。方向统一"越大越好"。
TECH_FACTORS = ["momentum", "trend", "volume"]

_FACTOR_CN = {"momentum": "动量", "trend": "趋势", "volume": "量能"}


def build_panel(fwd_days: int = 5, min_len: int = 90, progress_cb=None) -> pd.DataFrame:
    """
    构建全市场因子面板(长表)。

    fwd_days: 未来收益窗口(交易日)。fwd_ret = 该日之后 fwd_days 日的涨跌幅。
    min_len:  历史不足 min_len 根K线的票跳过(算不出 ma60 + 稳定因子)。
    progress_cb(done, total): 进度回调。

    返回 DataFrame(long): date, code, industry, momentum, trend, volume, fwd_ret
      —— 已 dropna,每行都是一个"可用于检验的截面样本点"。
    """
    codes = db.list_cached_codes()
    slist = db.load_stock_list()
    indmap = dict(zip(slist["code"], slist["industry"]))

    frames = []
    total = len(codes)
    for i, code in enumerate(codes):
        if progress_cb:
            progress_cb(i + 1, total)
        df = db.load_kline(code)
        if df is None or df.empty or len(df) < min_len:
            continue
        df = ind.enrich(df)
        c = df["close"]

        # --- 逐日技术因子(定义与 L1 cross_section 完全一致,保证口径统一) ---
        momentum = df["chg_20"]                                   # 近20日涨幅%
        ma60 = df["ma60"].replace(0, np.nan)
        trend = (df["ma5"] - ma60) / ma60                        # 均线发散度
        v5 = df["volume"].rolling(5).mean()
        v20 = df["volume"].rolling(20).mean().replace(0, np.nan)
        volume = v5 / v20                                        # 量能放大倍数

        # --- 未来 fwd_days 日收益(这是"答案",最后 fwd_days 行必然为 NaN) ---
        fwd_ret = c.shift(-fwd_days) / c - 1.0
        # 脏点防御:A股 fwd_days(≤~20)日收益理论极限约±(涨跌停×天数),
        # 出现 |收益|>100% 几乎必然是前复权断裂/停牌复牌跳变等数据异常。
        # 这类极端值会把分组均值污染成 inf,故直接剔除(置 NaN,后面 dropna)。
        fwd_ret = fwd_ret.where(fwd_ret.abs() <= 1.0, np.nan)

        sub = pd.DataFrame({
            "date": df["date"].astype(str),
            "code": code,
            "industry": indmap.get(code) or "未分类",
            "momentum": momentum.values,
            "trend": trend.values,
            "volume": volume.values,
            "fwd_ret": fwd_ret.values,
        })
        # 任一因子或答案缺失的行不能用于检验,直接剔除
        sub = sub.replace([np.inf, -np.inf], np.nan)
        sub = sub.dropna(subset=TECH_FACTORS + ["fwd_ret"])
        if not sub.empty:
            frames.append(sub)

    if not frames:
        return pd.DataFrame(
            columns=["date", "code", "industry"] + TECH_FACTORS + ["fwd_ret"])
    return pd.concat(frames, ignore_index=True)


def sample_rebalance_dates(panel: pd.DataFrame, step: int) -> list:
    """
    从面板全局交易日里,每隔 step 个交易日取一个作为"调仓/截面日"。
    这样所有票在同一批日期上对齐,且相邻截面的未来收益窗口不重叠(step=fwd_days时),
    避免重叠采样带来的伪显著。
    """
    dates = sorted(panel["date"].unique())
    if step <= 1:
        return dates
    return dates[::step]
