"""
策略层 - 技术指标计算
纯 pandas 实现常用指标,不依赖 TA-Lib(免安装编译),方便在 Windows 上直接跑。
所有函数输入日线 DataFrame(含 close/high/low/volume),返回 Series 或新列。
"""
import numpy as np
import pandas as pd


def ma(close: pd.Series, n: int) -> pd.Series:
    """简单移动平均"""
    return close.rolling(n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    """指数移动平均"""
    return close.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """相对强弱指标 RSI"""
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(n).mean()
    loss = (-diff.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def boll(close: pd.Series, n: int = 20, k: float = 2.0):
    """布林带,返回 (中轨, 上轨, 下轨)"""
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    return mid, mid + k * std, mid - k * std


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    """MACD,返回 (dif, dea, macd柱)"""
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, (dif - dea) * 2


def pct_change_n(close: pd.Series, n: int) -> pd.Series:
    """近 n 日涨跌幅"""
    return close.pct_change(n)


def resample_period(df: pd.DataFrame, period: str = "W") -> pd.DataFrame:
    """
    把日线聚合成周线/月线。period: 'D'=日(原样返回) / 'W'=周 / 'M'=月。
    聚合规则:开=首日开盘,高=期内最高,低=期内最低,收=末日收盘,量/额=期内求和。
    A股习惯:周线以周五(最后交易日)收盘,月线以月末收盘。
    返回列与日线一致(date/open/high/low/close/volume/amount),date 为该周期最后交易日。
    """
    if df is None or df.empty:
        return df
    p = (period or "D").upper()
    if p == "D":
        return df.reset_index(drop=True)

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()

    # pandas 新版周期别名:周五结束用 W-FRI,月末用 ME(旧版 M)
    rule = "W-FRI" if p == "W" else "ME"
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    if "amount" in d.columns:
        agg["amount"] = "sum"
    try:
        out = d.resample(rule).agg(agg)
    except ValueError:
        # 老 pandas 不认识 ME,回退到 M
        rule = "W-FRI" if p == "W" else "M"
        out = d.resample(rule).agg(agg)

    out = out.dropna(subset=["close"]).reset_index()
    # date 落到该周期实际最后一个交易日(而非周五/月末的日历日)
    last_dates = d.reset_index().groupby(
        pd.Grouper(key="date", freq=rule))["date"].max()
    out["date"] = out["date"].map(
        lambda x: last_dates.get(x, x)).dt.strftime("%Y-%m-%d")
    if "amount" not in out.columns:
        out["amount"] = 0.0
    cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """给日线 df 附加一批常用指标列,供策略和图表直接使用。"""
    df = df.copy()
    c = df["close"]
    df["ma5"] = ma(c, 5)
    df["ma10"] = ma(c, 10)
    df["ma20"] = ma(c, 20)
    df["ma60"] = ma(c, 60)
    df["rsi14"] = rsi(c, 14)
    mid, up, low = boll(c)
    df["boll_mid"], df["boll_up"], df["boll_low"] = mid, up, low

    # 成交量均线 + 量比(当日量 / 近5日均量)
    df["vol_ma5"] = ma(df["volume"], 5)
    df["vol_ma10"] = ma(df["volume"], 10)
    df["vol_ratio"] = df["volume"] / (df["vol_ma5"].shift(1) + 1e-9)

    # MACD(12,26,9): dif, dea, 柱
    dif, dea, bar = macd(c)
    df["dif"], df["dea"], df["macd_bar"] = dif, dea, bar

    # N 日最高/最低(不含当日),用于"突破新高/跌破新低"类策略
    df["high_20"] = df["high"].rolling(20).max().shift(1)
    df["high_60"] = df["high"].rolling(60).max().shift(1)
    df["low_20"] = df["low"].rolling(20).min().shift(1)

    # 近 N 日涨跌幅(百分比)
    df["chg_5"] = c.pct_change(5) * 100
    df["chg_20"] = c.pct_change(20) * 100
    return df
