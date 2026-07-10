"""
策略层 - 策略排行榜 / 今日推荐
========================================
这是"选股软件"的灵魂功能:软件打开就给答案,而不是让用户
一个个策略手动点。做两件事:

1) 策略排行榜:对每个策略做一次历史回测,按综合表现打分排名,
   让用户一眼看清"哪个策略在这段历史里最靠谱"。

2) 今日推荐:用当前最新数据跑每个策略的选股,把结果汇总,
   每只票标注"来自哪个策略 + 该策略历史年化/胜率",
   并按 (策略排名, 个股得分) 综合排序,最强策略选出的票排最前。

【综合评分口径】(排行榜用)
   score = 年化收益(cagr) * 0.6 + 盈亏比映射 * 0.25 - 回撤惩罚 * 0.15
   —— 年化为主,盈亏比加分,回撤扣分。可按需调整权重。

结果会缓存到内存 STATE,UI 启动时后台算一次即可。
"""
import pandas as pd

from ..data import database as db
from . import base, scanner, backtest as bt

import json
import os
from datetime import datetime

# 策略排行榜结果缓存(JSON)。排行基于固定历史区间回测,一天内不会变,
# 故落盘缓存供"今日推荐"复用,避免每次生成推荐都重跑 8 策略回测。
_CACHE_PATH = os.path.join(
    os.path.dirname(db.DB_PATH), "rank_cache.json")


def save_rank_cache(rank_df: pd.DataFrame, market_filter: bool) -> None:
    """把排行榜结果落盘,记录计算时间、依据的数据日期、是否大盘过滤。"""
    if rank_df is None or rank_df.empty:
        return
    payload = {
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_date": db.cache_summary().get("last_date"),
        "market_filter": bool(market_filter),
        "rows": rank_df.to_dict(orient="records"),
    }
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa
        print(f"[warn] 排行缓存写入失败: {e}")


def load_rank_cache() -> dict:
    """读取排行缓存,返回 {computed_at, data_date, market_filter, rank_df};
    无缓存或损坏返回 None。"""
    if not os.path.exists(_CACHE_PATH):
        return None
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("rows") or []
        if not rows:
            return None
        payload["rank_df"] = pd.DataFrame(rows)
        return payload
    except Exception as e:  # noqa
        print(f"[warn] 排行缓存读取失败: {e}")
        return None



def _combo_score(stats: dict) -> float:
    """把回测统计压成一个综合分,用于策略排名。越大越好。"""
    if not stats or stats.get("n_trades", 0) == 0:
        return -999.0
    cagr = stats.get("cagr", 0.0)
    pf = stats.get("profit_factor", 0.0) or 0.0
    dd = abs(stats.get("max_dd", 0.0))
    # 盈亏比>1 才算加分,映射到 0~1 量级;回撤按百分比扣分
    pf_bonus = max(min((pf - 1.0), 1.0), -0.5) * 30
    return cagr * 0.6 + pf_bonus * 0.25 - dd * 0.15


def rank_strategies(codes=None, start_from="2021-07-01", hold_days=10,
                    take_profit=0.15, stop_loss=0.08, max_positions=5,
                    market_filter=True, progress_cb=None) -> pd.DataFrame:
    """
    对全部策略做回测并排名。
    返回 DataFrame: key,name,score(综合分),cagr,total_return,win_rate,
                    max_dd,profit_factor,n_trades  按 score 降序。
    progress_cb(done,total,name) 按策略数回调。
    """
    if codes is None:
        codes = db.list_cached_codes()
    rows = []
    items = list(base.ALL_STRATEGIES.items())
    for n, (key, cls) in enumerate(items):
        res = bt.run_backtest(
            cls(), codes=codes, hold_days=hold_days, take_profit=take_profit,
            stop_loss=stop_loss, max_positions=max_positions,
            start_from=start_from, market_filter=market_filter,
        )
        s = res.stats
        rows.append({
            "key": key, "name": cls.name,
            "score": round(_combo_score(s), 1),
            "cagr": s.get("cagr", 0.0),
            "total_return": s.get("total_return", 0.0),
            "win_rate": s.get("win_rate", 0.0),
            "max_dd": s.get("max_dd", 0.0),
            "profit_factor": s.get("profit_factor", 0.0),
            "n_trades": s.get("n_trades", 0),
        })
        if progress_cb:
            progress_cb(n + 1, len(items), cls.name)
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def today_picks(rank_df: pd.DataFrame, top_strategies=3, per_strategy=10,
                progress_cb=None) -> pd.DataFrame:
    """
    生成"今日推荐票":取排行榜前 top_strategies 个策略,各自跑选股,
    把选出的票汇总去重,标注来源策略与其排名/年化。
    同一只票被多个策略同时选中时,保留排名更高(策略更强)的那条,
    并累计命中次数(hits)——被越多好策略选中越值得关注。

    返回 DataFrame: code,name,close,strategy,strat_rank,strat_cagr,
                    score(个股得分),hits,reason  排序:先按命中策略排名,再按个股得分。
    """
    if rank_df is None or rank_df.empty:
        return pd.DataFrame()
    top = rank_df.head(top_strategies)
    merged = {}       # code -> row(dict)
    n = 0
    for _, srow in top.iterrows():
        key = srow["key"]
        strat = base.ALL_STRATEGIES[key]()
        picks = scanner.scan(strat)
        if picks is not None and not picks.empty:
            picks = picks.head(per_strategy)
            for _, p in picks.iterrows():
                code = p["code"]
                cand = {
                    "code": code, "name": p["name"], "close": p["close"],
                    "strategy": srow["name"], "strat_rank": int(srow["rank"]),
                    "strat_cagr": srow["cagr"], "score": p["score"],
                    "hits": 1, "reason": p["reason"],
                }
                if code in merged:
                    merged[code]["hits"] += 1
                    # 保留策略排名更高(数字更小)的来源
                    if cand["strat_rank"] < merged[code]["strat_rank"]:
                        cand["hits"] = merged[code]["hits"]
                        merged[code] = cand
                else:
                    merged[code] = cand
        n += 1
        if progress_cb:
            progress_cb(n, len(top), srow["name"])

    if not merged:
        return pd.DataFrame()
    out = pd.DataFrame(list(merged.values()))
    # 排序:命中次数多优先 -> 来源策略排名高 -> 个股得分高
    out = out.sort_values(
        ["hits", "strat_rank", "score"],
        ascending=[False, True, False]
    ).reset_index(drop=True)
    return out
