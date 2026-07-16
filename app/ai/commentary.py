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
from app.strategy import chip
from app.strategy import indicators as ind
from .client import chat, chat_stream, AIError

DISCLAIMER = "以上为 AI 依据技术与基本面指标生成的解读,仅供参考,不构成任何投资建议。"

SYSTEM_PROMPT = (
    "你是一名严谨的 A 股分析助手。用户会给你某只股票由系统算好/实时拉取的客观事实,"
    "包含【技术面】(趋势/量能/动能)、【基本面】(估值/盈利/成长/负债)与【行业对比】。\n"
    "你的输出必须严格分为四部分,顺序与标签固定:\n"
    "【综合】先给一句总定性(如'趋势偏强、估值偏贵的成长股'),再在同一行末尾用固定格式"
    "标注两项评级,便于程序解析:  评级:偏多/中性/偏空 | 风险:高/中/低\n"
    "【技术面】2-3 句:趋势、量能、动能强弱;若给出【多周期】信息,须结合日线与"
    "周线是否共振(大小周期同向更可靠,背离需警惕短期反弹或回调);若给出【筹码结构】,"
    "可结合获利盘/平均成本/成本密集区判断上方套牢抛压或下方支撑(筹码为本地估算,只作"
    "定性参考,不得报精确套牢比例);\n"
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


# ==================== 多周期共振(日线 vs 周线) ====================
def _weekly_trend(df_daily) -> dict:
    """把日线重采样成周线,判断周线级别趋势方向(多头/空头/震荡)。

    纯本地计算、零接口依赖。返回 {state, ma_up, macd_up, weeks} 或 None(数据不足)。
      · state: '周线多头'/'周线空头'/'周线震荡'
      · ma_up: 周线是否均线多头(MA5>MA20)
      · macd_up: 周线 MACD 是否在多头区(DIF>DEA)
    """
    try:
        import pandas as pd
        d = df_daily.copy()
        d["_dt"] = pd.to_datetime(d["date"])
        d = d.set_index("_dt")
        # 按自然周重采样(周五对齐),取 OHLCV
        wk = d.resample("W-FRI").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        if len(wk) < 22:  # 至少 22 周才够算周线 MA20 + MACD
            return None
        wk = ind.enrich(wk.reset_index(drop=True))
        r = wk.iloc[-1]
        ma_up = bool(r["ma5"] > r["ma20"])
        macd_up = bool(r["dif"] > r["dea"])
        if ma_up and macd_up:
            state = "周线多头"
        elif (not ma_up) and (not macd_up):
            state = "周线空头"
        else:
            state = "周线震荡"
        return {"state": state, "ma_up": ma_up, "macd_up": macd_up,
                "weeks": int(len(wk))}
    except Exception:  # noqa
        return None


def _resonance(daily_ma_state: str, weekly: dict) -> str:
    """综合日线均线形态 + 周线趋势,给出多周期共振判断(供 AI 参考的客观描述)。"""
    if not weekly:
        return ""
    wk = weekly["state"]
    day_bull = "多头" in daily_ma_state
    day_bear = "空头" in daily_ma_state
    if wk == "周线多头" and day_bull:
        return "日线与周线共振向上(大小周期同步走多,趋势较可靠)"
    if wk == "周线空头" and day_bear:
        return "日线与周线共振向下(大小周期同步走空)"
    if wk == "周线多头" and day_bear:
        return "周线多头但日线转弱(大周期偏多、短期回调,或为洗盘)"
    if wk == "周线空头" and day_bull:
        return "日线反弹但周线仍空(大周期未转好,警惕短线反弹后继续下行)"
    return f"周线{wk.replace('周线','')}、日线{daily_ma_state[:2]},多周期方向不一致"


def build_facts(code: str) -> dict:
    """汇集一组客观事实(技术面 + 基本面 + 实时价 + 行业分位 + 多周期共振)。

    返回 dict;若本地无日线返回 {"error": "..."}。
    """
    df = db.load_kline(code)
    if df is None or df.empty or len(df) < 5:
        return {"error": f"{code} 本地无足够日线数据"}
    weekly = _weekly_trend(df)   # 重采样需原始日线,须在 enrich 前用 date 列
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

    # 筹码分布(成本结构):本地日线三角分布衰减模型,与通达信标准一致。
    # 定位为"辅助参考、说趋势不报精确值"(股本近似,禁 AI 编造精确套牢比例)。
    try:
        _chip = chip.compute_chip_distribution(
            df, total_mv=fund.get("total_mv"))
    except Exception:  # noqa
        _chip = None

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
        # 多周期共振(周线趋势 + 日周共振描述)
        "weekly_state": weekly["state"] if weekly else None,
        "resonance": _resonance(ma_state, weekly),
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
        # 筹码结构(成本分布派生;股本近似,作趋势参考)
        "chip_profit_ratio": (_chip["profit_ratio"] if _chip else None),
        "chip_avg_cost": (_chip["avg_cost"] if _chip else None),
        "chip_cost_low": (_chip["cost_low"] if _chip else None),
        "chip_cost_high": (_chip["cost_high"] if _chip else None),
        "chip_concentration": (_chip["concentration"] if _chip else None),
        "chip_shares_known": (_chip["shares_known"] if _chip else None),
    }


def _valuation_verdict(pct: dict, metric: str, industry: str,
                       peers: int) -> str:
    """把行业估值分位直接翻译成"贵/便宜"的明确结论,避免 AI 误读。

    percentile 定义(见 database.industry_valuation_percentile):
      = 同行业中 PE/PB 比本股【更低(更便宜)】的公司占比。
    所以 percentile 越【小】→ 比它便宜的同行越少 → 它反而越便宜;
        percentile 越【大】→ 大多数同行都比它便宜 → 它越贵。
    这里直接给出结论 + 双向表述,不让 AI 自己换算,消除歧义。
    """
    p = pct["percentile"]
    cheaper_than = 100.0 - p   # 它比多少比例的同行更便宜
    if p <= 30:
        verdict = "估值偏低(比同行便宜)"
    elif p >= 70:
        verdict = "估值偏高(比同行贵)"
    else:
        verdict = "估值居中"
    return (f"{verdict};在【{industry}】行业 {peers} 只同行中,"
            f"它比其中约 {cheaper_than:.0f}% 的公司更便宜"
            f"(即仅约 {p:.0f}% 的同行 {metric} 比它更低)。"
            f"注意:分位低=便宜,分位高=贵,勿与'高于X%'的字面混淆")


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
        lines.append("- 行业估值对比(PE): " + _valuation_verdict(
            pe_pct, "PE", pe_pct["industry"], pe_pct["peers"]))
    pb_pct = f.get("pb_pct")
    if pb_pct:
        lines.append("- 行业估值对比(PB): " + _valuation_verdict(
            pb_pct, "PB", pb_pct["industry"], pb_pct["peers"]))

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


def _chip_line(f: dict) -> str:
    """筹码结构要点。定位为辅助参考:说清获利盘/成本区/集中度的"趋势含义",
    并明确告诉模型这是本地估算(股本近似)、只可定性、禁止编造精确数字。"""
    pr = f.get("chip_profit_ratio")
    ac = f.get("chip_avg_cost")
    if pr is None or ac is None:
        return ""
    close = f.get("close")
    pf = pr * 100
    # 获利盘定性
    if pf >= 85:
        pf_tag = "绝大多数持仓浮盈,上方套牢抛压小但存在获利了结压力"
    elif pf >= 50:
        pf_tag = "多数持仓浮盈"
    elif pf >= 15:
        pf_tag = "多数持仓套牢,上方成本区抛压较重"
    else:
        pf_tag = "几乎全员套牢,反弹需消化上方密集套牢盘"
    # 现价相对平均成本
    if close and ac:
        if close >= ac * 1.02:
            pos = f"现价({_fmt(close)})高于平均成本({_fmt(ac)}),持仓者整体浮盈"
        elif close <= ac * 0.98:
            pos = f"现价({_fmt(close)})低于平均成本({_fmt(ac)}),持仓者整体浮亏、上方即套牢区"
        else:
            pos = f"现价({_fmt(close)})基本贴近平均成本({_fmt(ac)})"
    else:
        pos = f"平均成本约 {_fmt(ac)}"
    parts = [f"获利盘约{pf:.0f}%({pf_tag})", pos]
    lo, hi = f.get("chip_cost_low"), f.get("chip_cost_high")
    if lo and hi:
        parts.append(f"90%筹码集中在 {_fmt(lo)}~{_fmt(hi)}")
    conc = f.get("chip_concentration")
    if conc is not None:
        c_tag = "筹码集中(易拉升/易控盘)" if conc < 0.3 else (
            "筹码分散(换手充分/分歧大)" if conc > 0.6 else "集中度中等")
        parts.append(f"集中度{conc:.2f}({c_tag})")
    note = "近1年本地估算、股本近似,仅作成本结构的定性参考,请勿据此报精确套牢比例"
    return f"- 筹码结构: " + "; ".join(parts) + f"。[{note}]"


def facts_to_lines(f: dict) -> str:
    """把事实 dict 拼成给模型看的要点列表(纯文本)。"""
    price_tag = "当日实时" if f.get("price_source") == "realtime" else "本地日线收盘"
    reson = ""
    if f.get("resonance"):
        wk = f.get("weekly_state") or ""
        reson = f"- 多周期: 周线趋势={wk}; {f['resonance']}\n"
    return (
        f"- 股票: {f['code']} {f['name']} (所属行业: {f['industry']})\n"
        f"- 最新价({price_tag}): {_fmt(f['close'])}  当日涨跌: {_fmt(f['day_chg'])}%\n"
        f"- 近5日涨跌: {_fmt(f['chg_5'])}%  近20日涨跌: {_fmt(f['chg_20'])}%\n"
        f"- 均线形态(日线): {f['ma_state']}\n"
        f"- 动能: {f['macd_state']}; RSI(14)={_fmt(f['rsi'],1)}({f['rsi_state']})\n"
        f"{reson}"
        f"- 量能: 量比={_fmt(f['vol_ratio'])}({f['vol_state']})\n"
        f"- 位置: 处于近60日区间约 {_fmt(f['pos_60'],0)}% 分位"
        f"(60日高={_fmt(f['high_60'])})\n"
        f"- 布林带: 上轨={_fmt(f['boll_up'])} 下轨={_fmt(f['boll_low'])}\n"
        f"{_chip_line(f) + chr(10) if _chip_line(f) else ''}"
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

    优先识别固定格式 "评级:偏多 | 风险:高";AI 偶尔不按格式时,回退到
    正文关键词兜底推断,尽量不返回空。
    """
    rating, risk = None, None
    m = re.search(r"评级[:：]\s*(偏多|中性|偏空)", text)
    if m:
        rating = m.group(1)
    m = re.search(r"风险[:：]\s*(高|中|低)", text)
    if m:
        risk = m.group(1)

    # 兜底:未按固定格式标注时,从正文关键词推断
    if rating is None:
        if "偏多" in text:
            rating = "偏多"
        elif "偏空" in text:
            rating = "偏空"
        elif "中性" in text:
            rating = "中性"
    if risk is None:
        # 取"风险高/风险较高/风险高企"等表述
        if re.search(r"风险\s*(较?高|偏高|高企)", text) or "风险高" in text:
            risk = "高"
        elif re.search(r"风险\s*(较?低|偏低)", text) or "风险低" in text:
            risk = "低"
        elif "风险" in text and "中" in text:
            risk = "中"
    return {"rating": rating, "risk": risk}


# ==================== 当天缓存 ====================
# key = (code, trade_date) -> 完整结果 dict。进程内缓存,重启即清。
_CACHE = {}
_CACHE_LOCK = threading.Lock()


def comment_stock(code: str, strategy_hint: str = "",
                  force_refresh: bool = False, on_delta=None) -> dict:
    """
    对单只股票生成 AI 点评。返回:
      {"ok": True, "facts": {...}, "text": "...", "rating": "...",
       "risk": "...", "cached": bool, "disclaimer": "..."}
      {"ok": False, "error": "原因"}
    - 当天缓存: 同 code+交易日 命中缓存直接复用(force_refresh=True 跳过);
    - on_delta(piece): 若提供,则走流式接口边生成边回调(命中缓存时不回调);
      流式失败自动回退到普通 chat();
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

    prompt = build_prompt(facts, strategy_hint)
    try:
        if on_delta is not None:
            try:
                text = chat_stream(prompt, on_delta)
            except AIError:  # noqa 流式不可用则回退普通调用
                text = chat(prompt)
        else:
            text = chat(prompt)
    except AIError as e:  # noqa
        return {"ok": False, "error": str(e), "facts": facts}

    r = parse_rating(text)
    result = {"ok": True, "facts": facts, "text": text,
              "rating": r["rating"], "risk": r["risk"],
              "cached": False, "disclaimer": DISCLAIMER}
    with _CACHE_LOCK:
        _CACHE[key] = result
    # 落历史存档(失败不影响点评返回)
    try:
        db.save_ai_commentary(code, facts.get("name", ""), trade_date,
                              r["rating"], r["risk"], text)
    except Exception:  # noqa
        pass
    return result


# ==================== 批量点评(自选池晨报) ====================
def comment_batch(codes: list, progress_cb=None,
                  force_refresh: bool = False) -> dict:
    """对一批股票逐只生成点评(串行,复用当天缓存)。

    返回 {"ok": True, "items": [{code,name,rating,risk,text,...}], "n": N}。
    progress_cb(done, total, code) 用于 UI 进度提示。单只失败不中断整体。
    """
    codes = [c for c in (codes or []) if c]
    if not codes:
        return {"ok": False, "error": "没有可点评的股票"}
    items = []
    total = len(codes)
    for i, code in enumerate(codes):
        try:
            r = comment_stock(code, force_refresh=force_refresh)
            if r.get("ok"):
                f = r["facts"]
                items.append({
                    "code": code, "name": f.get("name", ""),
                    "industry": f.get("industry", ""),
                    "rating": r.get("rating"), "risk": r.get("risk"),
                    "text": r["text"], "cached": r.get("cached", False),
                })
            else:
                items.append({"code": code, "name": db.name_of(code),
                              "error": r.get("error", "点评失败")})
        except Exception as e:  # noqa
            items.append({"code": code, "name": db.name_of(code),
                          "error": str(e)})
        if progress_cb:
            progress_cb(i + 1, total, code)
    return {"ok": True, "items": items, "n": len(items),
            "disclaimer": DISCLAIMER}


# ==================== 组合解读(对一批票做全局画像研判) ====================
PORTFOLIO_SYSTEM_PROMPT = (
    "你是一名严谨的 A 股组合分析助手。用户会给你一篮子股票由系统算好的客观统计"
    "(数量、行业分布、估值区间、技术面多空计数、多周期共振计数)。\n"
    "请从组合整体视角输出研判,严格分三部分,顺序与标签固定:\n"
    "【组合画像】2-3 句:板块集中度(是否押注单一行业)、整体估值高低、"
    "技术面整体偏多还是偏空;\n"
    "【集中度风险】1-2 句:行业/风格是否过度集中,分散程度如何;\n"
    "【组合风险】1-2 条客观风险点(如行业高度集中、整体估值偏高、"
    "多数个股技术面转弱、多周期普遍背离等)。\n"
    "严格禁止:不得给出'买入/卖出/调仓/加减仓'等操作建议,不得预测涨跌幅或目标价,"
    "不得编造未给出的信息。语气客观中立,总字数控制在 260 字内。"
)


def _portfolio_profile(codes: list) -> dict:
    """纯本地收集一篮子股票的组合画像(不走网络):行业分布、估值、技术面计数。

    返回 dict;供 _portfolio_lines 拼提示词,也可单独给 UI 展示。
    """
    imap = db.load_industry_map()
    fmap = db.load_fundamental_map()

    n = 0
    industry_cnt = {}
    pe_vals, pb_vals, mv_vals = [], [], []
    ma_long = ma_short = ma_mix = 0        # 均线多头/空头/交织
    reso_up = reso_down = reso_mix = 0     # 多周期共振向上/向下/背离或震荡
    missing_funda = 0

    for code in codes:
        df = db.load_kline(code)
        if df is None or df.empty or len(df) < 5:
            continue
        n += 1
        ind_name = imap.get(code) or "未分类"
        industry_cnt[ind_name] = industry_cnt.get(ind_name, 0) + 1

        fd = fmap.get(code, {})
        for key, bucket in (("pe_ttm", pe_vals), ("pb", pb_vals),
                            ("total_mv", mv_vals)):
            v = fd.get(key)
            try:
                if v is not None and v == v:
                    bucket.append(float(v))
            except Exception:  # noqa
                pass
        if not fd:
            missing_funda += 1

        # 技术面:均线排列 + 多周期共振(复用现有逻辑,纯本地)
        weekly = _weekly_trend(df)
        try:
            e = ind.enrich(df)
            last = e.iloc[-1]
            ma5, ma20, ma60 = (float(last["ma5"]), float(last["ma20"]),
                               float(last["ma60"]))
            if ma5 > ma20 > ma60:
                ma_state = "多头排列"
                ma_long += 1
            elif ma5 < ma20 < ma60:
                ma_state = "空头排列"
                ma_short += 1
            else:
                ma_state = "均线交织"
                ma_mix += 1
        except Exception:  # noqa
            ma_state = "均线交织"
            ma_mix += 1

        reso = _resonance(ma_state, weekly)
        if "共振向上" in reso:
            reso_up += 1
        elif "共振向下" in reso:
            reso_down += 1
        else:
            reso_mix += 1

    def _stat(vals):
        if not vals:
            return None
        vs = sorted(vals)
        mid = vs[len(vs) // 2]
        return {"min": min(vs), "max": max(vs), "median": mid, "n": len(vs)}

    top_ind = sorted(industry_cnt.items(), key=lambda kv: -kv[1])
    top_share = (top_ind[0][1] / n) if (n and top_ind) else 0.0

    return {
        "n": n,
        "industries": top_ind,           # [(行业, 数量), ...] 降序
        "top_industry": top_ind[0][0] if top_ind else "",
        "top_share": top_share,          # 最大行业占比 0~1
        "pe": _stat(pe_vals),
        "pb": _stat(pb_vals),
        "mv": _stat(mv_vals),
        "ma_long": ma_long, "ma_short": ma_short, "ma_mix": ma_mix,
        "reso_up": reso_up, "reso_down": reso_down, "reso_mix": reso_mix,
        "missing_funda": missing_funda,
    }


def _portfolio_lines(p: dict) -> str:
    """把组合画像拼成给 AI 的客观事实文本。"""
    lines = [f"- 组合规模: {p['n']} 只(有效日线)"]

    # 行业分布(取前 6 个)
    inds = p.get("industries") or []
    if inds:
        head = "; ".join(f"{name}×{cnt}" for name, cnt in inds[:6])
        more = f" 等{len(inds)}个行业" if len(inds) > 6 else ""
        lines.append(f"- 行业分布: {head}{more}")
        lines.append(f"- 最大行业: {p['top_industry']} "
                     f"(占比 {p['top_share']*100:.0f}%)")

    def _rng(label, s, nd=1):
        if not s:
            return None
        return (f"- {label}: 中位 {s['median']:.{nd}f}, "
                f"区间 {s['min']:.{nd}f}~{s['max']:.{nd}f} "
                f"(有数据 {s['n']} 只)")

    for line in (_rng("PE(TTM)", p.get("pe")),
                 _rng("PB", p.get("pb"), 2),
                 _rng("市值(亿)", p.get("mv"), 0)):
        if line:
            lines.append(line)
    if p.get("missing_funda"):
        lines.append(f"- 缺基本面数据: {p['missing_funda']} 只")

    lines.append(f"- 均线形态: 多头排列 {p['ma_long']} 只 / "
                 f"空头排列 {p['ma_short']} 只 / 交织 {p['ma_mix']} 只")
    lines.append(f"- 多周期共振: 向上 {p['reso_up']} 只 / "
                 f"向下 {p['reso_down']} 只 / 背离或震荡 {p['reso_mix']} 只")
    return "\n".join(lines)


def comment_portfolio(codes: list, title: str = "组合", on_delta=None) -> dict:
    """对一篮子股票做全局组合解读(板块集中度/整体估值/组合风险)。

    - 组合画像纯本地收集(不走网络),再单次调用 AI 做全局研判;
    - on_delta 提供时走流式(边生成边回调),失败自动回退普通调用;
    - 调用方(UI)负责放到后台线程执行。
    返回 {"ok": True, "profile": {...}, "text": "...", "disclaimer": "..."}
        或 {"ok": False, "error": "..."}。
    """
    codes = [c for c in (codes or []) if c]
    if not codes:
        return {"ok": False, "error": "没有可解读的股票"}
    profile = _portfolio_profile(codes)
    if profile["n"] == 0:
        return {"ok": False, "error": "这批股票本地都没有足够的日线数据"}

    facts_text = _portfolio_lines(profile)
    prompt = [
        {"role": "system", "content": PORTFOLIO_SYSTEM_PROMPT},
        {"role": "user",
         "content": (f"以下是「{title}」这一篮子股票的客观统计,请做组合层面研判:\n\n"
                     f"{facts_text}")},
    ]
    try:
        if on_delta is not None:
            try:
                text = chat_stream(prompt, on_delta, max_tokens=600)
            except AIError:  # noqa
                text = chat(prompt, max_tokens=600)
        else:
            text = chat(prompt, max_tokens=600)
    except AIError as e:  # noqa
        return {"ok": False, "error": str(e), "profile": profile}

    return {"ok": True, "profile": profile, "text": text,
            "disclaimer": DISCLAIMER}
