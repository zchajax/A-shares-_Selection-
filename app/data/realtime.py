"""
数据层 - 实时行情(盘中盯盘)
========================================
盘中给自选/推荐票刷新最新价与涨跌幅,并检查价格预警是否触发。

【数据源】新浪 stock_zh_a_spot:
  · 一次调用返回【全市场 5000+ 只】的实时快照(约 10-15s),
    比逐只查高效得多——刷新一次就能覆盖所有关注的票。
  · 返回列含: 代码/名称/最新价/涨跌幅/昨收/今开/最高/最低/成交量/成交额...
  · 代码带交易所前缀(如 sh600519 / sz000001 / bj920000),这里统一转 6 位。
  · 非交易时段返回的是最近收盘快照,依然可用(涨跌幅为当日收盘涨跌)。

对外主入口:
  fetch_spot() -> {code(6位): {price, chg_pct, name, open, high, low, prev_close}}
  check_alerts(spot) -> list[dict] 已触发的预警明细
"""
import socket

import pandas as pd

from . import database as db

try:
    import akshare as ak
except ImportError:
    ak = None


def _strip_prefix(sym: str) -> str:
    """sh600519 -> 600519;已是6位数字则原样。"""
    s = str(sym).strip().lower()
    for p in ("sh", "sz", "bj"):
        if s.startswith(p):
            return s[len(p):]
    return s.zfill(6) if s.isdigit() else s


def fetch_spot() -> dict:
    """
    拉取全市场实时快照,返回 {6位代码: 明细dict}。
    明细含: price(最新价), chg_pct(涨跌幅%), name, open, high, low, prev_close。
    网络异常时返回空 dict(调用方据此提示,不崩)。
    """
    if ak is None:
        return {}
    # spot 接口较大,给足超时(全局默认15s对5000+行可能偏紧)
    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)
    try:
        df = ak.stock_zh_a_spot()
    except Exception as e:  # noqa
        print(f"[warn] 实时行情拉取失败: {e}")
        return {}
    finally:
        socket.setdefaulttimeout(old)
    if df is None or df.empty:
        return {}

    out = {}
    for r in df.itertuples(index=False):
        d = r._asdict() if hasattr(r, "_asdict") else dict(zip(df.columns, r))
        code = _strip_prefix(d.get("代码", ""))
        if not code:
            continue
        def _f(key):
            try:
                return float(d.get(key))
            except Exception:
                return None
        price = _f("最新价")
        chg = _f("涨跌幅")
        prev = _f("昨收")
        # 盘前/午间休市/停牌时新浪返回"最新价=0",此时用"昨收"兜底(仍是有效
        # 参考价),并标记 stale=True,让上层知道这不是当日成交价、涨跌幅无意义。
        stale = False
        if price is None or price <= 0:
            if prev is not None and prev > 0:
                price = prev
                chg = None
                stale = True
            else:
                price = None
        out[code] = {
            "name": d.get("名称", ""),
            "price": price,
            "chg_pct": chg,
            "open": _f("今开"),
            "high": _f("最高"),
            "low": _f("最低"),
            "prev_close": prev,
            "stale": stale,
        }
    return out


def quotes_for(codes: list, spot: dict = None) -> dict:
    """从快照里取指定 codes 的行情;spot=None 时现拉一次。"""
    if spot is None:
        spot = fetch_spot()
    return {c: spot.get(c) for c in codes}


def check_alerts(spot: dict = None) -> list:
    """
    用实时快照检查所有预警是否触发。
    返回已触发列表:每项 dict(code,name,price,chg_pct,reasons[list]),
    reasons 说明命中了哪些条件(可能同时命中多条)。
    """
    alerts = db.load_alerts()
    if alerts is None or alerts.empty:
        return []
    if spot is None:
        spot = fetch_spot()
    fired = []
    for _, a in alerts.iterrows():
        code = a["code"]
        q = spot.get(code)
        if not q or q.get("price") is None:
            continue
        # 盘前/休市/停牌:价格是昨收兜底、涨跌幅无意义,跳过以免每天开盘前误报。
        if q.get("stale"):
            continue
        price = q["price"]
        chg = q.get("chg_pct")
        reasons = []
        pl, ph = a.get("price_low"), a.get("price_high")
        cl, ch = a.get("chg_low"), a.get("chg_high")
        if pl is not None and not pd.isna(pl) and pl > 0 and price <= pl:
            reasons.append(f"价格≤{pl:g}(现{price:g})")
        if ph is not None and not pd.isna(ph) and ph > 0 and price >= ph:
            reasons.append(f"价格≥{ph:g}(现{price:g})")
        if chg is not None:
            if cl is not None and not pd.isna(cl) and cl != 0 and chg <= cl:
                reasons.append(f"跌幅≤{cl:g}%(现{chg:+.2f}%)")
            if ch is not None and not pd.isna(ch) and ch != 0 and chg >= ch:
                reasons.append(f"涨幅≥{ch:g}%(现{chg:+.2f}%)")
        if reasons:
            fired.append({
                "code": code, "name": a.get("name", "") or (q.get("name") or ""),
                "price": price, "chg_pct": chg, "reasons": reasons,
            })
    return fired
