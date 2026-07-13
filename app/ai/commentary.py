"""选股 AI 点评:把本地算好的结构化指标翻译成人话点评 + 风险提示。

流程: build_facts(code) 用本地日线算出一组"客观事实" → build_prompt 拼成
提示词 → client.chat 让模型输出 2~3 句点评。指标全部由本地计算,AI 不碰原始
数据、不给买卖结论,只做"翻译 + 提示风险"。
"""

from app.data import database as db
from app.strategy import indicators as ind
from .client import chat, AIError

DISCLAIMER = "以上为 AI 依据技术与基本面指标生成的解读,仅供参考,不构成任何投资建议。"

SYSTEM_PROMPT = (
    "你是一名严谨的 A 股分析助手。用户会给你某只股票由系统算好/拉取的客观事实,"
    "包含【技术面】(趋势/量能/动能)与【基本面】(估值/盈利)。你的任务:\n"
    "1. 用通俗中文先点评技术面走势(2-3 句:趋势、量能、动能强弱);\n"
    "2. 再点评基本面(1-2 句:估值高低、盈利能力ROE、市值规模,若数据缺失就说明未获取到);\n"
    "3. 客观提示 1-2 条风险点(如超买、放量滞涨、指标背离、临近压力位、估值偏高、亏损或ROE偏低等);\n"
    "严格禁止:不得给出'买入/卖出/加仓/减仓'等操作建议,不得预测目标价或涨跌幅,"
    "不得编造事实里没有给出的信息(尤其不得虚构财报数字、消息面、新闻)。"
    "语气客观中立,总字数控制在 220 字内。"
)


def _fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:  # noqa
        return "-"


def get_fundamental_ondemand(code: str) -> dict:
    """取单只股票基本面:本地有则直接用;缺失则实时拉取一次(约1-2s)并存库。

    返回 {pe_ttm,pb,ps_ttm,total_mv,roe, _source:"local"/"fetched"/"none"}。
    实时拉取失败(无网络/接口异常)时降级返回 _source="none",不抛异常。
    """
    row = db.get_fundamental(code)
    if row and any(row.get(k) is not None
                   for k in ("pe_ttm", "pb", "roe", "total_mv")):
        row["_source"] = "local"
        return row
    # 本地缺失 → 实时拉取一只
    try:
        from app.data import fetcher as ft
        d = ft._fetch_one_fundamental(code)
        if d and any(v is not None for v in d.values()):
            try:
                db.save_fundamental({code: d})
            except Exception:  # noqa 存库失败不影响本次点评
                pass
            d["_source"] = "fetched"
            return d
    except Exception:  # noqa 无网络/未装 akshare 等
        pass
    return {"pe_ttm": None, "pb": None, "ps_ttm": None,
            "total_mv": None, "roe": None, "_source": "none"}


def build_facts(code: str) -> dict:
    """用本地日线数据算出一组客观技术事实(供 prompt 使用,也可单独展示)。

    返回 dict;若本地无数据返回 {"error": "..."}。
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
    prev_close = float(prev["close"])
    day_chg = (close / prev_close - 1) * 100 if prev_close else 0.0

    # 均线多空排列
    ma5, ma20, ma60 = float(last["ma5"]), float(last["ma20"]), float(last["ma60"])
    if ma5 > ma20 > ma60:
        ma_state = "多头排列(MA5>MA20>MA60)"
    elif ma5 < ma20 < ma60:
        ma_state = "空头排列(MA5<MA20<MA60)"
    else:
        ma_state = "均线交织(方向不明)"

    # MACD 金叉/死叉判定(用最近两日 dif-dea 的符号变化)
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

    return {
        "code": code,
        "name": name,
        "industry": industry or "未知",
        "close": close,
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
        # 基本面
        "pe_ttm": fund.get("pe_ttm"),
        "pb": fund.get("pb"),
        "ps_ttm": fund.get("ps_ttm"),
        "total_mv": fund.get("total_mv"),
        "roe": fund.get("roe"),
        "fund_source": fund.get("_source", "none"),
    }


def _fundamental_lines(f: dict) -> str:
    """基本面要点。若整体缺失,明确告诉模型"未获取到",避免其编造。"""
    pe, pb, roe = f.get("pe_ttm"), f.get("pb"), f.get("roe")
    mv, ps = f.get("total_mv"), f.get("ps_ttm")
    if all(v is None for v in (pe, pb, roe, mv, ps)):
        return "- 基本面: 未获取到该股估值/盈利数据(请勿编造,分析中说明缺失即可)"
    pe_s = _fmt(pe)
    if pe is not None and float(pe) < 0:
        pe_s += "(为负,公司当前亏损)"
    parts = [
        f"市盈率PE(TTM)={pe_s}",
        f"市净率PB={_fmt(pb)}",
        f"市销率PS={_fmt(ps)}",
        f"总市值={_fmt(mv)}亿",
        f"净资产收益率ROE={_fmt(roe)}%",
    ]
    return "- 基本面: " + "; ".join(parts)


def facts_to_lines(f: dict) -> str:
    """把事实 dict 拼成给模型看的要点列表(纯文本)。"""
    return (
        f"- 股票: {f['code']} {f['name']} (所属行业: {f['industry']})\n"
        f"- 最新收盘价: {_fmt(f['close'])}  当日涨跌: {_fmt(f['day_chg'])}%\n"
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
    """构造 messages。strategy_hint: 可选,说明该股是被哪个策略选中/命中。"""
    extra = f"\n- 量化系统备注: {strategy_hint}" if strategy_hint else ""
    user = (
        "请解读以下这只股票(以下均为系统算好/拉取的客观事实,含技术面与基本面):\n\n"
        f"{facts_to_lines(f)}{extra}\n\n"
        "请按要求输出:技术面走势解读 + 基本面点评 + 风险提示。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def comment_stock(code: str, strategy_hint: str = "") -> dict:
    """
    对单只股票生成 AI 点评。返回:
      {"ok": True,  "facts": {...}, "text": "点评...", "disclaimer": "..."}
      {"ok": False, "error": "原因"}
    调用方(UI)负责放到后台线程执行,避免阻塞界面。
    """
    facts = build_facts(code)
    if "error" in facts:
        return {"ok": False, "error": facts["error"]}
    try:
        text = chat(build_prompt(facts, strategy_hint))
    except AIError as e:  # noqa
        return {"ok": False, "error": str(e), "facts": facts}
    return {"ok": True, "facts": facts, "text": text, "disclaimer": DISCLAIMER}
