"""选股 AI 点评:把本地算好/实时拉取的结构化指标翻译成人话点评 + 风险提示。

流程: build_facts(code) 汇集"客观事实"(技术面 + 基本面 + 实时价 + 行业估值分位)
→ build_prompt 拼成提示词 → client.chat 让模型输出结构化点评。指标全部由系统
计算/拉取,AI 不碰原始数据、不给买卖结论,只做"翻译 + 提示风险 + 定性评级"。

主要能力:
- 实时价喂入:点评前取一次分时快照,用当日实时价 + 真昨收,避免本地日线过时。
- 按需基本面:本地缺则实时拉一只(估值 + 成长/质量),含毛利率/净利率/营收增速等。
- 行业估值分位:PE/PB 在本地同行业中的分位,让"贵不贵"有横向参照。
- 结构化评级:AI 额外吐出 综合评级(偏多/中性/偏空) + 风险等级(高/中/低),可复用。
- 当天缓存:同一 code+交易日 只请求一次 API,可强制刷新。
"""

import re
import threading

from app.data import database as db
from app.strategy import indicators as ind
from .client import chat, AIError

DISCLAIMER = "以上为 AI 依据技术与基本面指标生成的解读,仅供参考,不构成任何投资建议。"

SYSTEM_PROMPT = (
    "你是一名严谨的 A 股分析助手。用户会给你某只股票由系统算好/实时拉取的客观事实,"
    "包含【技术面】(趋势/量能/动能)、【基本面】(估值/盈利/成长/负债)与【行业对比】。\n"
    "你的输出必须严格分为四部分,顺序与标签固定:\n"
    "【综合】先给一句总定性(如'趋势偏强、估值偏贵的成长股'),再在同一行末尾用固定格式"
    "标注两项评级,便于程序解析:  评级:偏多/中性/偏空 | 风险:高/中/低\n"
    "【技术面】2-3 句:趋势、量能、动能强弱;\n"
    "【基本面】2-3 句:估值高低(结合行业分位)、盈利能力(ROE/毛利率/净利率)、"
    "成长性(营收/净利增速)、负债水平;数据缺失就说明未获取到,不得编造;\n"
    "【风险】1-2 条客观风险点(如超买、放量滞涨、指标背离、临近压力位、估值高于行业、"
    "亏损或增速转负、负债偏高等)。\n"
    "严格禁止:不得给出'买入/卖出/加仓/减仓'等操作建议,不得预测目标价或涨跌幅,"
    "不得编造事实里没有给出的信息(尤其不得虚构财报数字、消息面、新闻)。"
    "语气客观中立,总字数控制在 260 字内。"
)


def _fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:  # noqa
        return "-"


# ==================== 实时价 ====================
def get_realtime_snapshot(code: str) -> dict:
    """取当日实时价 + 真昨收(来自分时接口,与涨跌幅修复同源)。

    返回 {price, prev_close, day_chg, trade_date, _ok} 或 _ok=False。
    失败(无网络/非交易品种)时降级 _ok=False,不抛异常,由调用方回退日线。
    """
    try:
        from app.data import fetcher as ft
        df = ft.fetch_intraday(code)
        if df is None or df.empty or "price" not in df.columns:
            return {"_ok": False}
        price = float(df["price"].iloc[-1])
        prev = df.attrs.get("prev_close")
        prev = float(prev) if prev is not None else None
        day_chg = (price / prev - 1) * 100 if prev else None
        return {"price": price, "prev_close": prev, "day_chg": day_chg,
                "trade_date": df.attrs.get("trade_date"), "_ok": True}
    except Exception:  # noqa
        return {"_ok": False}


# ==================== 基本面(按需) ====================
def get_fundamental_ondemand(code: str) -> dict:
    """取单只股票基本面:本地有则直接用;缺失则实时拉取一次(约1-2s)并存库。

    返回含 pe_ttm/pb/ps_ttm/total_mv/roe/gross_margin/net_margin/rev_yoy/
    profit_yoy/debt_ratio/dividend_ratio/report_date 及 _source。
    实时拉取失败时降级 _source="none",不抛异常。
    """
    _keys = ("pe_ttm", "pb", "ps_ttm", "total_mv", "roe", "gross_margin",
             "net_margin", "rev_yoy", "profit_yoy", "debt_ratio",
             "dividend_ratio", "report_date")
    row = db.get_fundamental(code)
    # 本地已有"较全"(估值或成长任一非空)则用本地
    if row and any(row.get(k) is not None
                   for k in ("pe_ttm", "pb", "roe", "total_mv",
                             "gross_margin", "rev_yoy")):
        out = {k: row.get(k) for k in _keys}
        out["_source"] = "local"
        return out
    # 本地缺失 → 实时拉取一只
    try:
        from app.data import fetcher as ft
        d = ft._fetch_one_fundamental(code)
        if d and any(v is not None for v in d.values()):
            try:
                db.save_fundamental({code: d})
            except Exception:  # noqa 存库失败不影响本次点评
                pass
            out = {k: d.get(k) for k in _keys}
            out["_source"] = "fetched"
            return out
    except Exception:  # noqa 无网络/未装 akshare 等
        pass
    return {**{k: None for k in _keys}, "_source": "none"}


def build_facts(code: str) -> dict:
    """汇集一组客观事实(技术面 + 基本面 + 实时价 + 行业分位)。

    返回 dict;若本地无日线返回 {"error": "..."}。
    """
    df = db.load_kline(code)
    if df is None or df.empty or len(df) < 5:
        return {"error": f"{code} 本地无足够日线数据"}
    df = ind.enrich(df)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    name = db.name_of(code) or ""
    try:
        industry = db.load_industry_map().get(code, "")
    except Exception:  # noqa
        industry = ""

    close = float(last["close"])
    kl_prev_close = float(prev["close"])
    day_chg = (close / kl_prev_close - 1) * 100 if kl_prev_close else 0.0

    # 实时价:优先用当日实时快照覆盖(解决本地日线过时的"最新价失真")
    rt = get_realtime_snapshot(code)
    if rt.get("_ok") and rt.get("price"):
        close = rt["price"]
        if rt.get("day_chg") is not None:
            day_chg = rt["day_chg"]
        price_source = "realtime"
    else:
        price_source = "daily"  # 回退本地日线收盘

    # 均线多空排列
    ma5, ma20, ma60 = float(last["ma5"]), float(last["ma20"]), float(last["ma60"])
    if ma5 > ma20 > ma60:
        ma_state = "多头排列(MA5>MA20>MA60)"
    elif ma5 < ma20 < ma60:
        ma_state = "空头排列(MA5<MA20<MA60)"
    else:
        ma_state = "均线交织(方向不明)"

    # MACD 金叉/死叉判定
    dif, dea = float(last["dif"]), float(last["dea"])
    dif_p, dea_p = float(prev["dif"]), float(prev["dea"])
    if dif_p <= dea_p and dif > dea:
        macd_state = "今日 MACD 金叉"
    elif dif_p >= dea_p and dif < dea:
        macd_state = "今日 MACD 死叉"
    elif dif > dea:
        macd_state = "MACD 处多头区(DIF在DEA上方)"
    else:
        macd_state = "MACD 处空头区(DIF在DEA下方)"

    rsi = float(last["rsi14"])
    if rsi >= 80:
        rsi_state = "严重超买"
    elif rsi >= 70:
        rsi_state = "偏超买"
    elif rsi <= 20:
        rsi_state = "严重超卖"
    elif rsi <= 30:
        rsi_state = "偏超卖"
    else:
        rsi_state = "中性"

    vol_ratio = float(last["vol_ratio"])
    if vol_ratio >= 2:
        vol_state = "明显放量"
    elif vol_ratio >= 1.2:
        vol_state = "温和放量"
    elif vol_ratio <= 0.6:
        vol_state = "明显缩量"
    else:
        vol_state = "量能平稳"

    # 相对 20/60 日高低位置
    high_60 = float(last["high_60"])
    low_20 = float(last["low_20"])
    pos_60 = (close - low_20) / (high_60 - low_20) * 100 if high_60 > low_20 else 50.0

    # 基本面:本地有则用,缺失则实时拉取一次(失败降级为空)
    fund = get_fundamental_ondemand(code)

    # 行业估值分位(仅本地缓存足够时才有值)
    try:
        pe_pct = db.industry_valuation_percentile(code, "pe_ttm")
    except Exception:  # noqa
        pe_pct = None
    try:
        pb_pct = db.industry_valuation_percentile(code, "pb")
    except Exception:  # noqa
        pb_pct = None

    return {
        "code": code,
        "name": name,
        "industry": industry or "未知",
        "close": close,
        "price_source": price_source,
        "trade_date": rt.get("trade_date") if rt.get("_ok") else None,
        "day_chg": day_chg,
        "chg_5": float(last["chg_5"]),
        "chg_20": float(last["chg_20"]),
        "ma_state": ma_state,
        "macd_state": macd_state,
        "rsi": rsi,
        "rsi_state": rsi_state,
        "vol_ratio": vol_ratio,
        "vol_state": vol_state,
        "pos_60": pos_60,
        "high_60": high_60,
        "boll_up": float(last["boll_up"]),
        "boll_low": float(last["boll_low"]),
        # 基本面(估值 + 成长/质量)
        "pe_ttm": fund.get("pe_ttm"),
        "pb": fund.get("pb"),
        "ps_ttm": fund.get("ps_ttm"),
        "total_mv": fund.get("total_mv"),
        "roe": fund.get("roe"),
        "gross_margin": fund.get("gross_margin"),
        "net_margin": fund.get("net_margin"),
        "rev_yoy": fund.get("rev_yoy"),
        "profit_yoy": fund.get("profit_yoy"),
        "debt_ratio": fund.get("debt_ratio"),
        "dividend_ratio": fund.get("dividend_ratio"),
        "report_date": fund.get("report_date"),
        "fund_source": fund.get("_source", "none"),
        # 行业估值分位
        "pe_pct": pe_pct,
        "pb_pct": pb_pct,
    }


def _fundamental_lines(f: dict) -> str:
    """基本面要点(估值 + 成长/质量)。整体缺失时明确告诉模型"未获取到"。"""
    val_keys = ("pe_ttm", "pb", "roe", "total_mv", "ps_ttm",
                "gross_margin", "net_margin", "rev_yoy", "profit_yoy",
                "debt_ratio", "dividend_ratio")
    if all(f.get(k) is None for k in val_keys):
        return "- 基本面: 未获取到该股估值/盈利数据(请勿编造,分析中说明缺失即可)"

    pe = f.get("pe_ttm")
    pe_s = _fmt(pe)
    if pe is not None and float(pe) < 0:
        pe_s += "(为负,公司当前亏损)"

    # 估值行(附行业分位)
    val_parts = [
        f"PE(TTM)={pe_s}", f"PB={_fmt(f.get('pb'))}",
        f"PS={_fmt(f.get('ps_ttm'))}", f"总市值={_fmt(f.get('total_mv'))}亿",
    ]
    lines = ["- 估值: " + "; ".join(val_parts)]

    pe_pct = f.get("pe_pct")
    if pe_pct:
        lines.append(
            f"- 行业估值对比: 该股 PE 高于所在【{pe_pct['industry']}】行业约 "
            f"{pe_pct['percentile']:.0f}% 的公司(同业样本 {pe_pct['peers']} 只);"
            f"分位越高越贵")
    pb_pct = f.get("pb_pct")
    if pb_pct:
        lines.append(
            f"- 行业PB对比: 该股 PB 高于同行业约 {pb_pct['percentile']:.0f}% 的公司")

    # 盈利/成长/负债行
    prof_parts = [f"ROE={_fmt(f.get('roe'))}%"]
    if f.get("gross_margin") is not None:
        prof_parts.append(f"毛利率={_fmt(f.get('gross_margin'))}%")
    if f.get("net_margin") is not None:
        prof_parts.append(f"净利率={_fmt(f.get('net_margin'))}%")
    lines.append("- 盈利能力: " + "; ".join(prof_parts))

    grow_parts = []
    if f.get("rev_yoy") is not None:
        grow_parts.append(f"营收同比={_fmt(f.get('rev_yoy'))}%")
    if f.get("profit_yoy") is not None:
        grow_parts.append(f"净利润同比={_fmt(f.get('profit_yoy'))}%")
    if grow_parts:
        lines.append("- 成长性: " + "; ".join(grow_parts))

    fin_parts = []
    if f.get("debt_ratio") is not None:
        fin_parts.append(f"资产负债率={_fmt(f.get('debt_ratio'))}%")
    if f.get("dividend_ratio") is not None:
        fin_parts.append(f"股息发放率={_fmt(f.get('dividend_ratio'))}%")
    if fin_parts:
        lines.append("- 财务/分红: " + "; ".join(fin_parts))

    if f.get("report_date"):
        lines.append(f"- 财报期: {f['report_date']}(估值为最新实时,财务指标为该报告期)")
    return "\n".join(lines)


def facts_to_lines(f: dict) -> str:
    """把事实 dict 拼成给模型看的要点列表(纯文本)。"""
    price_tag = "当日实时" if f.get("price_source") == "realtime" else "本地日线收盘"
    return (
        f"- 股票: {f['code']} {f['name']} (所属行业: {f['industry']})\n"
        f"- 最新价({price_tag}): {_fmt(f['close'])}  当日涨跌: {_fmt(f['day_chg'])}%\n"
        f"- 近5日涨跌: {_fmt(f['chg_5'])}%  近20日涨跌: {_fmt(f['chg_20'])}%\n"
        f"- 均线形态: {f['ma_state']}\n"
        f"- 动能: {f['macd_state']}; RSI(14)={_fmt(f['rsi'],1)}({f['rsi_state']})\n"
        f"- 量能: 量比={_fmt(f['vol_ratio'])}({f['vol_state']})\n"
        f"- 位置: 处于近60日区间约 {_fmt(f['pos_60'],0)}% 分位"
        f"(60日高={_fmt(f['high_60'])})\n"
        f"- 布林带: 上轨={_fmt(f['boll_up'])} 下轨={_fmt(f['boll_low'])}\n"
        f"{_fundamental_lines(f)}"
    )


def build_prompt(f: dict, strategy_hint: str = "") -> list:
    """构造 messages。strategy_hint: 可选,说明该股被哪个策略选中/命中。"""
    extra = f"\n- 量化系统备注: {strategy_hint}" if strategy_hint else ""
    user = (
        "请解读以下这只股票(以下均为系统算好/实时拉取的客观事实,"
        "含技术面、基本面与行业对比):\n\n"
        f"{facts_to_lines(f)}{extra}\n\n"
        "请严格按【综合】【技术面】【基本面】【风险】四段输出,"
        "并在【综合】行尾用固定格式标注:  评级:偏多/中性/偏空 | 风险:高/中/低"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ==================== 结构化评级解析 ====================
_RATING_MAP = {"偏多": "偏多", "中性": "中性", "偏空": "偏空"}
_RISK_MAP = {"高": "高", "中": "中", "低": "低"}


def parse_rating(text: str) -> dict:
    """从点评文本里解析 综合评级 与 风险等级(供彩色标签/榜单排序复用)。

    识别形如 "评级:偏多 | 风险:高" 的标注;解析不到返回 None 值。
    """
    rating, risk = None, None
    m = re.search(r"评级[:：]\s*(偏多|中性|偏空)", text)
    if m:
        rating = m.group(1)
    m = re.search(r"风险[:：]\s*(高|中|低)", text)
    if m:
        risk = m.group(1)
    return {"rating": rating, "risk": risk}


# ==================== 当天缓存 ====================
# key = (code, trade_date) -> 完整结果 dict。进程内缓存,重启即清。
_CACHE = {}
_CACHE_LOCK = threading.Lock()


def comment_stock(code: str, strategy_hint: str = "",
                  force_refresh: bool = False) -> dict:
    """
    对单只股票生成 AI 点评。返回:
      {"ok": True, "facts": {...}, "text": "...", "rating": "...",
       "risk": "...", "cached": bool, "disclaimer": "..."}
      {"ok": False, "error": "原因"}
    - 当天缓存: 同 code+交易日 命中缓存直接复用(force_refresh=True 跳过);
    - 调用方(UI)负责放到后台线程执行,避免阻塞界面。
    """
    facts = build_facts(code)
    if "error" in facts:
        return {"ok": False, "error": facts["error"]}

    # 缓存键用实时快照的交易日(build_facts 已取过,直接复用;拿不到则用今天)
    trade_date = facts.get("trade_date")
    if not trade_date:
        import datetime as _dt
        trade_date = _dt.date.today().isoformat()
    key = (code, trade_date)

    if not force_refresh:
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
        if hit:
            out = dict(hit)
            out["cached"] = True
            return out

    try:
        text = chat(build_prompt(facts, strategy_hint))
    except AIError as e:  # noqa
        return {"ok": False, "error": str(e), "facts": facts}

    r = parse_rating(text)
    result = {"ok": True, "facts": facts, "text": text,
              "rating": r["rating"], "risk": r["risk"],
              "cached": False, "disclaimer": DISCLAIMER}
    with _CACHE_LOCK:
        _CACHE[key] = result
    return result
