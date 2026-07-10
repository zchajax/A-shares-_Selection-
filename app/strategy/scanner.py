"""
策略层 - 全市场扫描器
把某个策略应用到本地所有已缓存的股票上,返回排序后的选股结果。
"""
import pandas as pd

from ..data import database as db
from . import indicators as ind


def scan(strategy, progress_cb=None) -> pd.DataFrame:
    """
    用给定策略扫描本地所有缓存股票。
    strategy: BaseStrategy 实例(参数已设置好)
    返回 DataFrame: code,name,score,reason,close  按 score 降序
    """
    codes = db.list_cached_codes()
    namemap = dict(zip(db.load_stock_list()["code"], db.load_stock_list()["name"]))
    results = []
    total = len(codes)
    for i, code in enumerate(codes):
        df = db.load_kline(code)
        if df.empty:
            continue
        df = ind.enrich(df)
        try:
            selected, score, reason = strategy.evaluate(df)
        except Exception as e:
            selected, score, reason = False, 0, f"err:{e}"
        if selected:
            results.append({
                "code": code,
                "name": namemap.get(code, ""),
                "score": round(score, 2),
                "close": round(float(df.iloc[-1]["close"]), 2),
                "reason": reason,
            })
        if progress_cb:
            progress_cb(i + 1, total)
    out = pd.DataFrame(results)
    if not out.empty:
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out
