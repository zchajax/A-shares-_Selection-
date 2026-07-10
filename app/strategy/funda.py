"""
策略层 - 基本面 / 估值过滤
========================================
把"技术面选股结果"再用基本面门槛筛一道:过滤掉估值过高、
盈利能力过差、市值过小的票,留下"技术形态好 + 基本面不拖后腿"的标的。

设计成【后置过滤器】而非嵌进策略 evaluate():
  · 策略只看日线(df),职责单一、可回测;
  · 基本面数据是"每只一行"的截面数据,天然适合对选股结果做二次过滤;
  · 过滤是可选项(阈值留空=不启用该项),不破坏原有纯技术选股。

对外主入口:
  apply_filter(df, thresholds) -> 过滤后的 df(带 pe_ttm/pb/roe/total_mv 列)
"""
import pandas as pd

from ..data import database as db


def enrich_with_fundamental(df: pd.DataFrame) -> pd.DataFrame:
    """给选股结果 df(含 code 列)补上 pe_ttm/pb/ps_ttm/total_mv/roe 列。"""
    if df is None or df.empty:
        return df
    fmap = db.load_fundamental_map()
    out = df.copy()
    for col in ("pe_ttm", "pb", "ps_ttm", "total_mv", "roe"):
        out[col] = out["code"].map(lambda c, k=col: (fmap.get(c) or {}).get(k))
    return out


def apply_filter(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    对选股结果按基本面门槛过滤。

    thresholds 支持的键(任一为 None 表示不启用该项):
      pe_max     市盈率(TTM) 上限,>0 才启用(<=0 或 None 忽略;负PE的票在启用时被剔除)
      pb_max     市净率 上限
      roe_min    净资产收益率(%) 下限
      mv_min     总市值(亿) 下限
      mv_max     总市值(亿) 上限
      drop_missing 缺失基本面数据的票是否剔除(默认 False:数据缺失不误杀)

    返回过滤后的 df(已补基本面列),保持原排序。
    """
    if df is None or df.empty:
        return df
    out = enrich_with_fundamental(df)
    drop_missing = bool(thresholds.get("drop_missing", False))

    def keep(row) -> bool:
        pe = row.get("pe_ttm")
        pb = row.get("pb")
        roe = row.get("roe")
        mv = row.get("total_mv")

        pe_max = thresholds.get("pe_max")
        if pe_max and pe_max > 0:
            if pe is None or pd.isna(pe):
                if drop_missing:
                    return False
            else:
                # 负PE(亏损)在启用PE上限时视为不达标剔除
                if pe <= 0 or pe > pe_max:
                    return False

        pb_max = thresholds.get("pb_max")
        if pb_max and pb_max > 0:
            if pb is None or pd.isna(pb):
                if drop_missing:
                    return False
            elif pb > pb_max:
                return False

        roe_min = thresholds.get("roe_min")
        if roe_min is not None and roe_min != 0:
            if roe is None or pd.isna(roe):
                if drop_missing:
                    return False
            elif roe < roe_min:
                return False

        mv_min = thresholds.get("mv_min")
        if mv_min and mv_min > 0:
            if mv is None or pd.isna(mv):
                if drop_missing:
                    return False
            elif mv < mv_min:
                return False

        mv_max = thresholds.get("mv_max")
        if mv_max and mv_max > 0:
            if mv is None or pd.isna(mv):
                if drop_missing:
                    return False
            elif mv > mv_max:
                return False

        return True

    mask = out.apply(keep, axis=1)
    return out[mask].reset_index(drop=True)
