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
# ETF 无个股基本面,免责声明改用基金语境(技术面 + 折溢价等指标)。
ETF_DISCLAIMER = "以上为 AI 依据技术面与折溢价等指标生成的解读,仅供参考,不构成任何投资建议。"

SYSTEM_PROMPT = (
    "你是一名严谨的 A 股分析助手。用户会给你某只股票由系统算好/实时拉取的客观事实,"
    "包含【技术面】(趋势/量能/动能)、【基本面】(估值/盈利/成长/负债)与【行业对比】。\n"
    "你的输出必须严格分为四部分,顺序与标签固定:\n"
    "【综合】先给一句总定性(如'趋势偏强、估值偏贵的成长股'),再在同一行末尾用固定格式"
    "标注两项评级,便于程序解析:  评级:偏多/中性/偏空 | 风险:高/中/低\n"
    "【技术面】2-3 句:趋势、量能、动能强弱;若给出【多周期】信息,须结合日线与"
    "周线是否共振(大小周期同向更可靠,背离需警惕短期反弹或回调);若给出【筹码结构】,"
    "切忌仅凭获利盘比例高低下结论(获利盘高≠必然获利了结、低≠必然抛压严重):获利盘只是"
    "'燃料量',须联合筹码密集/分散(以事实行给出的定性标签为准,不要自行从集中度数值推方向)"
    "与价格位置综合研判——筹码密集时持仓者浮盈相近、抛压弱、易同涨同跌(常是主升浪特征),"
    "筹码分散且下方堆积获利盘时了结冲动才强;现价处于筹码峰顶部(上方几乎无套牢盘)则'抛压"
    "严重'无从谈起,现价上方压着又近又厚的套牢盘才构成实打实的反弹阻力。据此定性判断上方"
    "抛压或下方支撑(筹码为本地估算,只作定性参考,不得报精确套牢比例);\n"
    "【基本面】2-3 句:估值高低(结合行业分位)、盈利能力(ROE/毛利率/净利率)、"
    "成长性(营收/净利增速)、负债水平;数据缺失就说明未获取到,不得编造;\n"
    "【风险】1-2 条客观风险点(如超买、放量滞涨、指标背离、临近压力位、估值高于行业、"
    "亏损或增速转负、负债偏高等)。\n"
    "严格禁止:不得给出'买入/卖出/加仓/减仓'等操作建议,不得预测目标价或涨跌幅,"
    "不得编造事实里没有给出的信息(尤其不得虚构财报数字、消息面、新闻)。"
    "语气客观中立,总字数控制在 260 字内。"
)

# ETF 是指数基金,没有个股的 PE/PB/ROE/财报/筹码套牢盘等概念,
# 因此用一套基金语境的提示词:聚焦跟踪指数的价格趋势、折溢价、规模流动性。
ETF_SYSTEM_PROMPT = (
    "你是一名严谨的 A 股 ETF(交易所交易基金)分析助手。用户会给你某只 ETF 由系统"
    "算好/实时拉取的客观事实,包含【技术面】(趋势/量能/动能/位置)与【ETF专属指标】"
    "(折溢价率/IOPV估值/基金规模/换手率/资金流)。\n"
    "重要认知:ETF 是一篮子成分股组成的指数基金,跟踪某个指数或行业主题,"
    "本身【没有】个股的市盈率/市净率/ROE/营收利润/财报,也【不存在】个股意义上的"
    "'套牢盘/获利盘'(ETF 靠一二级市场申赎套利,价格锚定净值)。因此严禁套用个股"
    "财务或筹码套牢逻辑,更不要说'未获取到基本面数据'这类话——对 ETF 而言这些指标"
    "本就不适用,属正常。\n"
    "你的输出必须严格分为四部分,顺序与标签固定:\n"
    "【综合】先给一句总定性(如'跟踪某行业指数、当前趋势偏弱的行业ETF'),再在同一行"
    "末尾用固定格式标注两项评级,便于程序解析:  评级:偏多/中性/偏空 | 风险:高/中/低\n"
    "【技术面】2-3 句:结合均线/MACD/RSI/量能判断趋势与动能强弱;若给出【多周期】,"
    "须结合日线与周线是否共振(同向更可靠,背离需警惕);说明当前处于近60日区间的高/低位。\n"
    "【资金与折溢价】2-3 句:①折溢价——折价率为正=现价低于净值(场内买相对划算),"
    "为负=溢价(现价高于净值,追高需谨慎,溢价过高有回归风险);②基金规模——规模越大"
    "流动性越好、越不易被边缘化(规模过小如低于2亿有清盘/流动性风险);③换手率/资金流"
    "反映交投活跃度与主力动向。数据缺失就说明未获取到,不得编造。\n"
    "【风险】1-2 条客观风险点(如高位偏热、放量滞涨、日周共振向下、溢价偏高存回归风险、"
    "规模偏小流动性弱、跟踪指数系统性回调等)。\n"
    "严格禁止:不得给出'买入/卖出/加仓/减仓'等操作建议,不得预测目标价或涨跌幅,"
    "不得编造事实里没有给出的信息;不得虚构成分股、跟踪指数名称或净值数字。"
    "语气客观中立,总字数控制在 260 字内。"
)


def _is_etf(code: str) -> bool:
    """判断是否为 ETF(以本地已登记的 etf_list 为准,可靠且不联网)。"""
    try:
        return str(code).zfill(6) in db.load_etf_codes()
    except Exception:  # noqa
        return False


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


def get_etf_metrics_ondemand(code: str) -> dict:
    """取单只 ETF 的专属指标(折溢价/规模/换手/资金流),实时拉取一次。

    返回含 iopv/discount_rate/turnover/vol_ratio/scale_yi/main_inflow_yi/
    main_inflow_pct 及 _source。失败降级 _source="none",不抛异常。
    """
    _keys = ("iopv", "discount_rate", "turnover", "vol_ratio",
             "scale_yi", "main_inflow_yi", "main_inflow_pct")
    try:
        from app.data import fetcher as ft
        d = ft.fetch_etf_metrics(code)
        if d and any(d.get(k) is not None for k in _keys):
            out = {k: d.get(k) for k in _keys}
            out["_source"] = "fetched"
            return out
    except Exception:  # noqa 无网络/接口变更等
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
    elif rsi >= 65:
        rsi_state = "偏热(接近超买)"
    elif rsi <= 20:
        rsi_state = "严重超卖"
    elif rsi <= 30:
        rsi_state = "偏超卖"
    elif rsi <= 35:
        rsi_state = "偏冷(接近超卖)"
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

    is_etf = _is_etf(code)

    # ETF 走基金专属分支:没有个股估值/财报/筹码套牢盘概念,
    # 改注入 ETF 专属指标(折溢价/规模/换手/资金流),其余技术面字段与股票共用。
    if is_etf:
        etfm = get_etf_metrics_ondemand(code)
        base = {
            "code": code,
            "name": name,
            "industry": industry or "ETF/指数基金",
            "is_etf": True,
            "close": close,
            "price_source": price_source,
            "trade_date": rt.get("trade_date") if rt.get("_ok") else None,
            "day_chg": day_chg,
            "chg_5": float(last["chg_5"]),
            "chg_20": float(last["chg_20"]),
            "ma_state": ma_state,
            "macd_state": macd_state,
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
            # ETF 专属
            "etf_iopv": etfm.get("iopv"),
            "etf_discount_rate": etfm.get("discount_rate"),
            "etf_turnover": etfm.get("turnover"),
            "etf_vol_ratio": etfm.get("vol_ratio"),
            "etf_scale_yi": etfm.get("scale_yi"),
            "etf_inflow_yi": etfm.get("main_inflow_yi"),
            "etf_inflow_pct": etfm.get("main_inflow_pct"),
            "etf_metrics_source": etfm.get("_source", "none"),
        }
        return base

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

    # 个股估值历史分位(纵向: PE/PB 在自身过去一年的位置, 与上面"行业横向对比"互补)。
    # 一次请求即出(百度估值接口), 失败降级为空; 放在这里让持仓/单只点评都能用。
    try:
        from app.data import market as _mkt
        _selfval = _mkt.stock_valuation_percentile(code)
    except Exception:  # noqa
        _selfval = {"pe": None, "pb": None}

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
        "is_etf": False,
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
        # 个股估值历史分位(纵向, 比自身过去一年)
        "self_pe_pct": _selfval.get("pe"),
        "self_pb_pct": _selfval.get("pb"),
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


def _self_hist_val_line(f: dict) -> str:
    """个股估值历史分位行(纵向, PE/PB 在自身过去一年区间的位置)。

    与 _valuation_verdict(行业横向对比)互补: 这里比的是"这只票现在 vs 它自己
    过去一年", 分位越小=越接近一年低位=相对自己历史越便宜。两个指标都有时,
    AI 能同时看"和同行比贵不贵""和自己历史比贵不贵"。任一缺失则跳过。
    """
    def _one(cell, label):
        if not cell:
            return None
        v = cell.get("value")
        p = cell.get("percentile")
        if v is None:
            return None
        if p is None:
            n = cell.get("samples")
            return f"{label}当前{v}(历史样本仅{n}天,不足一年,分位暂略)"
        if p <= 30:
            tag = "处近一年低位, 相对自身历史偏便宜"
        elif p >= 70:
            tag = "处近一年高位, 相对自身历史偏贵"
        else:
            tag = "处近一年中间区域"
        rng = cell.get("min"), cell.get("max")
        rng_s = (f", 一年区间{rng[0]}~{rng[1]}"
                 if rng[0] is not None and rng[1] is not None else "")
        return f"{label}当前{v}(近一年 {p:.0f}% 分位{rng_s}, {tag})"

    parts = [x for x in (_one(f.get("self_pe_pct"), "PE"),
                         _one(f.get("self_pb_pct"), "PB")) if x]
    if not parts:
        return ""
    return ("- 个股估值历史分位(纵向比自身过去一年,分位低=处历史低位=相对便宜): "
            + "; ".join(parts))


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

    # 个股估值历史分位(纵向, 比自身过去一年): 分位越小=越接近一年低位=越便宜
    self_hist = _self_hist_val_line(f)
    if self_hist:
        lines.append(self_hist)

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


def _etf_metrics_lines(f: dict) -> str:
    """ETF 专属指标要点:折溢价/规模/换手/资金流。整体缺失时明确说明未获取到。"""
    keys = ("etf_iopv", "etf_discount_rate", "etf_turnover",
            "etf_scale_yi", "etf_inflow_yi")
    if all(f.get(k) is None for k in keys):
        return ("- ETF专属指标: 未获取到折溢价/规模等数据(可能非交易时段或网络问题,"
                "请在分析中说明缺失,不得编造)")
    lines = []

    dr = f.get("etf_discount_rate")
    iopv = f.get("etf_iopv")
    if dr is not None:
        # 东财"基金折价率":正=折价(现价<净值,买相对划算),负=溢价(现价>净值)
        if dr > 0.3:
            dr_tag = "折价(现价低于净值,场内买入相对划算)"
        elif dr < -0.3:
            dr_tag = "溢价(现价高于净值,追高需谨慎、有向净值回归的风险)"
        else:
            dr_tag = "基本贴近净值(折溢价很小)"
        iopv_s = f"、IOPV实时估值≈{_fmt(iopv, 3)}" if iopv is not None else ""
        lines.append(f"- 折溢价: 折价率{_fmt(dr)}%{iopv_s}({dr_tag})")

    sc = f.get("etf_scale_yi")
    if sc is not None:
        if sc < 2:
            sc_tag = "规模偏小(低于2亿,存在流动性弱/清盘风险)"
        elif sc < 10:
            sc_tag = "规模中等"
        else:
            sc_tag = "规模较大(流动性较好)"
        lines.append(f"- 基金规模: 约{_fmt(sc)}亿元({sc_tag})")

    flow_parts = []
    tv = f.get("etf_turnover")
    if tv is not None:
        flow_parts.append(f"换手率={_fmt(tv)}%")
    inflow = f.get("etf_inflow_yi")
    ipct = f.get("etf_inflow_pct")
    if inflow is not None:
        direction = "净流入" if inflow >= 0 else "净流出"
        pct_s = f"(占比{_fmt(ipct)}%)" if ipct is not None else ""
        flow_parts.append(f"主力{direction}{_fmt(abs(inflow))}亿{pct_s}")
    if flow_parts:
        lines.append("- 交投/资金: " + "; ".join(flow_parts))

    return "\n".join(lines)


def _pos_60_tag(pos) -> str:
    """把近60日区间分位翻译成明确的高/中/低位标签, 避免 AI 把裸数字误读。
    位置分位 = (现价-近20日低)/(近60日高-近20日低)*100, 越高=越接近区间顶部=相对高位。
    注意: 这是'价格在近期区间的位置', 与用户买入价浮盈浮亏是两码事, 勿混淆。"""
    try:
        p = float(pos)
    except (TypeError, ValueError):
        return ""
    if p >= 80:
        return "相对高位(接近60日高点)"
    if p >= 60:
        return "偏上位置"
    if p <= 20:
        return "相对低位(接近60日低点)"
    if p <= 40:
        return "偏下位置"
    return "区间中部"


def _etf_facts_to_lines(f: dict) -> str:
    """ETF 专属事实拼装:技术面与股票共用,基本面/筹码替换为 ETF 专属指标。"""
    price_tag = "当日实时" if f.get("price_source") == "realtime" else "本地日线收盘"
    reson = ""
    if f.get("resonance"):
        wk = f.get("weekly_state") or ""
        reson = f"- 多周期: 周线趋势={wk}; {f['resonance']}\n"
    return (
        f"- ETF: {f['code']} {f['name']}(交易所交易基金/指数基金,无个股财报与筹码套牢盘概念)\n"
        f"- 最新价({price_tag}): {_fmt(f['close'], 3)}  当日涨跌: {_fmt(f['day_chg'])}%\n"
        f"- 近5日涨跌: {_fmt(f['chg_5'])}%  近20日涨跌: {_fmt(f['chg_20'])}%\n"
        f"- 均线形态(日线): {f['ma_state']}\n"
        f"- 动能: {f['macd_state']}; RSI(14)={_fmt(f['rsi'],1)}({f['rsi_state']})\n"
        f"{reson}"
        f"- 量能: 量比={_fmt(f['vol_ratio'])}({f['vol_state']})\n"
        f"- 位置: 处于近60日区间约 {_fmt(f['pos_60'],0)}% 分位, {_pos_60_tag(f['pos_60'])}"
        f"(60日高={_fmt(f['high_60'], 3)})\n"
        f"- 布林带: 上轨={_fmt(f['boll_up'], 3)} 下轨={_fmt(f['boll_low'], 3)}\n"
        f"{_etf_metrics_lines(f)}"
    )


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
        # 集中度低有两种【成因相反】的解释,不可一概贴"易控盘":
        #   · 小盘股筹码集中 → 可能真是主力高度控盘(易被拉升/打压);
        #   · 大盘蓝筹筹码集中 → 多是日换手极低(如银行0.4~0.8%)、成本长期沉淀
        #     所致,流通盘动辄数千亿根本无从"控盘",只意味波动小、成本稳定。
        # 用总市值(亿元)分流,避免把平安银行/工行这类巨无霸误判成"易被控盘"。
        mv = f.get("total_mv")
        if conc < 0.3:
            if mv and mv >= 800:
                c_tag = ("筹码高度集中,但此为大盘低换手、成本长期沉淀所致"
                         "(巨盘难被控盘),通常意味波动小、成本稳定")
            elif mv and mv <= 150:
                c_tag = "筹码高度集中,小盘股需留意主力控盘可能(易被拉升/打压)"
            else:
                c_tag = ("筹码分布偏窄(集中),成因需结合换手:"
                         "低换手多为长期沉淀,高换手才近控盘")
        elif conc > 0.6:
            c_tag = "筹码分散(换手充分/分歧大)"
        else:
            c_tag = "集中度中等"
        parts.append(f"集中度{conc:.2f}({c_tag})")
    note = "近1年本地估算、股本近似,仅作成本结构的定性参考,请勿据此报精确套牢比例"
    return f"- 筹码结构: " + "; ".join(parts) + f"。[{note}]"


def facts_to_lines(f: dict) -> str:
    """把事实 dict 拼成给模型看的要点列表(纯文本)。ETF 走基金专属分支。"""
    if f.get("is_etf"):
        return _etf_facts_to_lines(f)
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
        f"- 位置: 处于近60日区间约 {_fmt(f['pos_60'],0)}% 分位, {_pos_60_tag(f['pos_60'])}"
        f"(60日高={_fmt(f['high_60'])})\n"
        f"- 布林带: 上轨={_fmt(f['boll_up'])} 下轨={_fmt(f['boll_low'])}\n"
        f"{_chip_line(f) + chr(10) if _chip_line(f) else ''}"
        f"{_fundamental_lines(f)}"
    )


def build_prompt(f: dict, strategy_hint: str = "") -> list:
    """构造 messages。strategy_hint: 可选,说明该股被哪个策略选中/命中。
    ETF 走基金专属提示词与段落结构。"""
    extra = f"\n- 量化系统备注: {strategy_hint}" if strategy_hint else ""
    if f.get("is_etf"):
        user = (
            "请解读以下这只 ETF(交易所交易基金,以下均为系统算好/实时拉取的客观事实,"
            "含技术面与 ETF 专属指标):\n\n"
            f"{facts_to_lines(f)}{extra}\n\n"
            "请严格按【综合】【技术面】【资金与折溢价】【风险】四段输出,"
            "并在【综合】行尾用固定格式标注:  评级:偏多/中性/偏空 | 风险:高/中/低"
        )
        return [
            {"role": "system", "content": ETF_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
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
              "cached": False,
              "disclaimer": ETF_DISCLAIMER if facts.get("is_etf") else DISCLAIMER}
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


# ==================== 大盘/行业点评(对整个市场做研判) ====================
MARKET_SYSTEM_PROMPT = (
    "你是一名严谨的 A 股大盘分析助手。用户会给你当前大盘由系统实时拉取的客观事实,"
    "包含【指数行情】(上证/深证/创业板/科创50/沪深300 等涨跌幅)、【市场情绪】"
    "(涨跌家数、涨停跌停、赚钱效应/活跃度、两市成交额)、【行业板块】"
    "(领涨/领跌行业及其涨跌幅、资金净额)。\n"
    "请从整体市场视角输出研判,严格分四部分,顺序与标签固定:\n"
    "【大盘】2-3 句:先给一句总定性(如'指数分化、量能萎缩的普跌格局'),说明主要"
    "指数强弱、成交量水平;并在本段末尾用固定格式标注情绪评级,便于程序解析:"
    "  情绪:偏暖/中性/偏冷\n"
    "【市场情绪】2-3 句:结合涨跌家数与赚钱效应判断个股普涨还是普跌、是否存在"
    "'指数失真'(少数权重护盘但多数个股下跌);涨停跌停家数反映的多空强度;\n"
    "【热点板块】2-3 句:今日领涨与领跌的行业方向、是否有明确主线,结合资金净额"
    "说明资金流向(注意:净额为全单口径流入−流出,方向可参考但数值偏小,非主力资金);\n"
    "【风险提示】1-2 条客观风险点(如量能不足、赚钱效应低迷、指数与个股背离、"
    "板块普跌无主线、涨停跌停比恶化等)。\n"
    "严格禁止:不得给出'加仓/减仓/满仓/空仓/买入/卖出'等操作建议,不得预测"
    "点位或涨跌幅,不得编造事实里没有给出的信息(尤其不得虚构消息面、政策、新闻)。"
    "语气客观中立,总字数控制在 300 字内。"
)


def _market_facts_lines(snapshot: dict) -> str:
    """把行情页已拉取的快照(指数/情绪/总貌/板块)统计成给 AI 的客观事实文本。

    snapshot 结构(全部来自 UI 已展示的同一份数据,保证'点评'与'屏幕'一致):
      {
        "indexes": [{"name","chg_pct","price"}...],   # 核心指数
        "sh_amount","sz_amount","total_amount": 元,    # 两市成交额
        "activity": {上涨/下跌/平盘/涨停/真实涨停/跌停/真实跌停/停牌/活跃度...},
        "boards": [{"name","chg_pct","net_inflow"}...] # 行业板块(已按涨幅降序)
      }
    纯本地拼装,不发网络请求。
    """
    lines = []

    # ---- 指数行情 ----
    idx = snapshot.get("indexes") or []
    if idx:
        parts = []
        for r in idx:
            cp = r.get("chg_pct")
            cps = f"{cp:+.2f}%" if (cp is not None and cp == cp) else "-"
            parts.append(f"{r.get('name', '')} {cps}")
        lines.append("- 主要指数涨跌: " + "; ".join(parts))

    # ---- 两市成交额 ----
    tot = snapshot.get("total_amount")
    sh = snapshot.get("sh_amount")
    sz = snapshot.get("sz_amount")

    def _yi(v):
        try:
            x = float(v)
        except Exception:  # noqa
            return None
        if x != x:
            return None
        return x / 1e8

    tot_yi, sh_yi, sz_yi = _yi(tot), _yi(sh), _yi(sz)
    if tot_yi is not None:
        amt_line = f"- 两市成交额: {tot_yi:,.0f} 亿"
        if sh_yi is not None and sz_yi is not None:
            amt_line += f"(沪 {sh_yi:,.0f} 亿 + 深 {sz_yi:,.0f} 亿)"
        # 量能定性(经验阈值,仅作粗略参考)
        if tot_yi >= 12000:
            amt_line += ";量能较活跃"
        elif tot_yi <= 7000:
            amt_line += ";量能偏低迷"
        else:
            amt_line += ";量能中等"
        lines.append(amt_line)

    # ---- 市场情绪/涨跌家数 ----
    act = snapshot.get("activity") or {}

    def _int(k):
        try:
            return int(float(act.get(k)))
        except Exception:  # noqa
            return None
    up, down, flat = _int("上涨"), _int("下跌"), _int("平盘")
    zt, zt_real = _int("涨停"), _int("真实涨停")
    dt, dt_real = _int("跌停"), _int("真实跌停")
    if up is not None and down is not None:
        total_ud = up + down + (flat or 0)
        ratio = (up / (up + down) * 100) if (up + down) > 0 else None
        line = f"- 涨跌家数: 上涨 {up} / 下跌 {down}"
        if flat is not None:
            line += f" / 平盘 {flat}"
        if ratio is not None:
            line += f"(上涨占比约 {ratio:.0f}%)"
        lines.append(line)
    zt_parts = []
    if zt is not None:
        zt_parts.append(f"涨停 {zt}" + (f"(真实 {zt_real})"
                                        if zt_real is not None else ""))
    if dt is not None:
        zt_parts.append(f"跌停 {dt}" + (f"(真实 {dt_real})"
                                        if dt_real is not None else ""))
    if zt_parts:
        lines.append("- 涨跌停: " + " / ".join(zt_parts))
    active = act.get("活跃度")
    if active is not None:
        lines.append(f"- 赚钱效应(活跃度): {active}"
                     "(即上涨股票占比,越高市场越普涨;<40% 偏冷,>60% 偏暖)")

    # ---- 行业板块:领涨 / 领跌 各取若干 ----
    boards = snapshot.get("boards") or []
    if boards:
        # boards 已按涨跌幅降序;取头尾
        def _bd(r):
            cp = r.get("chg_pct")
            cps = f"{cp:+.2f}%" if (cp is not None and cp == cp) else "-"
            ni = r.get("net_inflow")
            nis = ""
            if ni is not None and ni == ni:
                nis = f",净额 {ni / 1e8:+.1f} 亿"
            return f"{r.get('name', '')} {cps}{nis}"
        up_n = min(6, len(boards))
        down_n = min(6, len(boards))
        top = [b for b in boards
               if (b.get("chg_pct") == b.get("chg_pct"))][:up_n]
        bottom = [b for b in boards
                  if (b.get("chg_pct") == b.get("chg_pct"))][-down_n:]
        n_boards = len(boards)
        n_up = sum(1 for b in boards
                   if (b.get("chg_pct") or 0) > 0
                   and b.get("chg_pct") == b.get("chg_pct"))
        lines.append(f"- 行业板块共 {n_boards} 个,其中上涨 {n_up} 个、"
                     f"下跌 {n_boards - n_up} 个")
        if top:
            lines.append("- 领涨行业: " + "; ".join(_bd(b) for b in top))
        if bottom:
            lines.append("- 领跌行业: "
                         + "; ".join(_bd(b) for b in reversed(bottom)))

    return "\n".join(lines) if lines else "- (未能获取到有效的大盘数据)"


def build_market_facts(snapshot: dict) -> dict:
    """校验 snapshot 是否含足够数据。返回 {"ok":bool, "lines":str, "error":str}。"""
    if not snapshot or not isinstance(snapshot, dict):
        return {"ok": False, "error": "没有可用的行情数据,请先在行情页点『刷新行情』"}
    has = bool(snapshot.get("indexes") or snapshot.get("activity")
               or snapshot.get("boards"))
    if not has:
        return {"ok": False, "error": "行情数据为空,请先在行情页点『刷新行情』再点评"}
    return {"ok": True, "lines": _market_facts_lines(snapshot)}


def comment_market(snapshot: dict, on_delta=None) -> dict:
    """对当前大盘 + 行业做整体 AI 研判。

    - 事实全部来自 UI 已拉取的行情快照(不发网络请求),保证与屏幕显示一致;
    - on_delta 提供时走流式(边生成边回调),失败自动回退普通调用;
    - 调用方(UI)负责放到后台线程执行。
    返回 {"ok":True, "text":..., "sentiment":偏暖/中性/偏冷, "disclaimer":...}
        或 {"ok":False, "error":...}。
    """
    fb = build_market_facts(snapshot)
    if not fb.get("ok"):
        return {"ok": False, "error": fb.get("error", "行情数据不足")}

    facts_text = fb["lines"]
    prompt = [
        {"role": "system", "content": MARKET_SYSTEM_PROMPT},
        {"role": "user",
         "content": ("以下是当前 A 股大盘由系统实时拉取的客观事实,"
                     "请做大盘 + 行业层面的整体研判:\n\n"
                     f"{facts_text}\n\n"
                     "请严格按【大盘】【市场情绪】【热点板块】【风险提示】四段输出,"
                     "并在【大盘】段末尾用固定格式标注:  情绪:偏暖/中性/偏冷")},
    ]
    try:
        if on_delta is not None:
            try:
                text = chat_stream(prompt, on_delta, max_tokens=700)
            except AIError:  # noqa
                text = chat(prompt, max_tokens=700)
        else:
            text = chat(prompt, max_tokens=700)
    except AIError as e:  # noqa
        return {"ok": False, "error": str(e)}

    # 解析情绪评级(供彩色标签)
    sentiment = None
    m = re.search(r"情绪[:：]\s*(偏暖|中性|偏冷)", text)
    if m:
        sentiment = m.group(1)
    return {"ok": True, "text": text, "sentiment": sentiment,
            "facts": facts_text, "disclaimer": DISCLAIMER}


# ==================== 持仓点评(结合买入价给持有/加减仓倾向) ====================
# 与 comment_stock 的关键区别: 这是"我已经持有"的视角,带入买入成本,
# 允许给出持有/加仓/减仓的倾向性判断(仍基于客观事实、列明风险、标注仅供参考)。
HOLDING_SYSTEM_PROMPT = (
    "你是一名严谨的 A 股持仓顾问。用户【已经持有】某只股票,会给你该股由系统"
    "算好/实时拉取的客观事实(技术面/基本面/行业对比/筹码结构),以及用户的"
    "【买入成本价、当前浮动盈亏】。请站在'持有者该如何管理这笔仓位'的角度输出,"
    "严格分四部分,顺序与标签固定,【四段缺一不可,尤其不得省略最后的【风险】段】:\n"
    "【持仓诊断】2-3 句:结合买入价与现价,说明当前是浮盈还是浮亏、盈亏幅度大小,"
    "以及现价相对成本、相对近期趋势所处的位置(如成本已被套牢盘覆盖、或已有安全垫)。"
    "严格区分两个不同概念,切勿混用:'买入价'是用户这一笔的个人持仓成本,只用于算用户的"
    "浮动盈亏;'平均成本/成本区'是筹码结构里全市场持仓者的平均成本,用于判断整体套牢/浮盈"
    "格局。谈用户盈亏时引用买入价,谈市场筹码位置时引用平均成本,不得把买入价当成平均成本;\n"
    "【技术面】2-3 句:趋势、量能、动能强弱;若给出【多周期】须结合日线与周线是否共振;"
    "若给出【筹码结构】,切忌仅凭获利盘比例高低就下减仓或持有的结论(获利盘高不必然要止盈、"
    "低不必然因抛压而离场):获利盘只是'燃料量',须联合筹码密集/分散(以事实行给出的定性标签"
    "为准,不要自行从集中度数值推方向)与价格位置综合研判——筹码密集且现价站上成本区,浮盈"
    "相近、抛压弱,即便获利盘高也常是主升浪、可继续持有;筹码分散、下方堆积大量低成本获利盘"
    "且已拉开距离,才需警惕获利了结;现价处于筹码峰顶部(上方几乎无套牢盘)则上方抛压小,现价"
    "上方压着又近又厚的套牢盘才构成反弹阻力、对浮亏仓位尤其不利(筹码为本地估算,只作定性"
    "参考);\n"
    "【操作倾向】给出明确但审慎的仓位管理倾向,在'继续持有 / 考虑减仓(止盈或止损)"
    "/ 逢低加仓 / 观望'中选择并说明理由,理由必须落在前面列出的客观事实上。"
    "止盈/止损用词纪律:'止盈'仅用于当前【浮盈】时的减仓,'止损/减仓控制风险'用于当前"
    "【浮亏】时的减仓——浮亏状态严禁说'止盈',浮盈状态减仓才叫'止盈'(以事实行的浮动盈亏"
    "正负为准)。位置纪律:'现价相对近60日区间的高/低位'(见位置行的高/中/低位标签)与"
    "'用户这笔的浮盈浮亏'是两个独立概念,不得混为一谈——例如浮亏但价格处于区间相对高位是"
    "完全可能的,描述位置时一律以事实行给出的高/中/低位标签为准,不得因为浮亏就说成'处于低位'。"
    "两条硬约束:(1)【归因纪律】'接近60日高点/处于高分位'本身是【高位】信号,不得反向"
    "解读成'还有上涨空间/上行空间'('上涨空间''上行空间'在高位语境下为禁用表述,一律不得"
    "出现);位置高低只陈述事实,是否有空间需另有依据(如刚突破、"
    "缩量回踩不破等),不得凭'接近高点'就推涨。(2)【高位禁加仓】当同时出现高位(近60日≥80%"
    "分位)与偏热(RSI≥65)且已有相当浮盈时,不得主动建议加仓或把回调说成'加仓良机',此种情形"
    "只能在'继续持有 / 考虑减仓止盈 / 观望'中选择;\n"
    "【风险】此段【必须独立输出、不得省略或留空,至少给满 2 条】客观风险点(如高位放量"
    "滞涨、指标背离、临近压力位、估值偏高、浮亏扩大跌破关键支撑等)。当出现下列偏热/高位"
    "信号时必须在此点出、不得因浮盈而回避:RSI 明显偏高(≥65 偏热、≥70 超买)、位置处于近60日"
    "高分位(≥80%)、获利盘比例很高(≥85%)叠加已积累相当浮盈、估值偏贵或公司当前亏损(PE 为负)"
    "——此类情形应提示短线追高谨防获利回吐;若当日实为收跌或滞涨,也不得把当日描述成上涨/"
    "动能增强。\n"
    "在【操作倾向】段末尾用固定格式标注,便于程序解析:  操作:持有/加仓/减仓/观望 | 风险:高/中/低\n"
    "重要边界:你可以给出倾向性的仓位管理建议(这是持仓顾问的职责),但必须审慎、"
    "基于事实、附带风险提示;不得预测精确目标价或涨跌幅;不得编造事实里没有的信息"
    "(尤其不得虚构财报数字、消息面、新闻);不得承诺收益。语气客观务实,"
    "总字数控制在 300 字内。"
)

# ETF 持仓点评专用:持有者视角 + 基金语境(无个股财报/筹码套牢盘)。
# 与股票版 HOLDING_SYSTEM_PROMPT 段落结构一致(便于 UI 解析),仅内容换成基金逻辑。
ETF_HOLDING_SYSTEM_PROMPT = (
    "你是一名严谨的 A 股 ETF(交易所交易基金)持仓顾问。用户【已经持有】某只 ETF,"
    "会给你该 ETF 由系统算好/实时拉取的客观事实——【技术面】(趋势/量能/动能/位置)"
    "与【ETF专属指标】(折溢价率/IOPV估值/基金规模/换手率/资金流),以及用户的"
    "【买入成本价、当前浮动盈亏】。请站在'持有者该如何管理这笔仓位'的角度输出。\n"
    "重要认知:ETF 是一篮子成分股组成的指数基金,跟踪某指数或行业主题,本身【没有】"
    "个股的市盈率/市净率/ROE/营收利润/财报,也【不存在】个股意义上的'套牢盘/获利盘/"
    "平均成本区'(ETF 靠一二级市场申赎套利,价格锚定净值)。因此严禁套用个股财务或"
    "筹码套牢逻辑,更不得出现'上方套牢盘压制''筹码结构''获利盘''安全垫(以筹码论)'"
    "'未获取到基本面数据'这类话——对 ETF 而言这些概念本就不适用。谈用户盈亏只用"
    "'买入价 vs 现价',谈估值贵贱只用'折溢价率(现价 vs 基金净值IOPV)'。\n"
    "严格分四部分,顺序与标签固定,【四段缺一不可,尤其不得省略最后的【风险】段】:\n"
    "【持仓诊断】2-3 句:结合买入价与现价,说明当前是浮盈还是浮亏、盈亏幅度大小;"
    "并结合折溢价率说明现价相对基金净值是偏贵(溢价)还是偏便宜(折价)。\n"
    "【技术面】2-3 句:结合均线/MACD/RSI/量能判断趋势与动能强弱;若给出【多周期】"
    "须结合日线与周线是否共振(同向更可靠);说明当前处于近60日区间的高/低位。\n"
    "【操作倾向】给出明确但审慎的仓位管理倾向,在'继续持有 / 考虑减仓(止盈或止损)"
    "/ 逢低加仓 / 观望'中选择并说明理由,理由必须落在前面列出的客观事实上(趋势方向、"
    "折溢价、规模流动性、资金流、位置高低)。硬约束:'处于近60日高分位'本身是【高位】"
    "信号,不得反向解读成'还有上涨空间';溢价偏高时不得建议追高加仓。\n"
    "【风险】此段【必须独立输出、不得省略,至少给满 2 条】客观风险点,如:日周线共振"
    "向下趋势偏弱、溢价偏高存在向净值回归的风险、基金规模偏小(低于2亿)有流动性/清盘"
    "风险、所跟踪指数或行业系统性回调、浮亏扩大跌破关键支撑等。切勿用'套牢盘'措辞。\n"
    "在【操作倾向】段末尾用固定格式标注,便于程序解析:  操作:持有/加仓/减仓/观望 | 风险:高/中/低\n"
    "重要边界:你可以给出倾向性的仓位管理建议(这是持仓顾问的职责),但必须审慎、"
    "基于事实、附带风险提示;不得预测精确目标价或涨跌幅;不得编造事实里没有的信息"
    "(尤其不得虚构成分股、跟踪指数名称、净值数字、消息面);不得承诺收益。语气客观"
    "务实,总字数控制在 300 字内。"
)

_HOLD_ACTION_MAP = {"持有": "持有", "加仓": "加仓", "减仓": "减仓", "观望": "观望"}


def parse_holding_action(text: str) -> dict:
    """从持仓点评文本里解析 操作倾向 与 风险等级。"""
    action, risk = None, None
    m = re.search(r"操作[:：]\s*(持有|加仓|减仓|观望)", text)
    if m:
        action = m.group(1)
    m = re.search(r"风险[:：]\s*(高|中|低)", text)
    if m:
        risk = m.group(1)
    # 兜底:未按固定格式时从正文关键词推断操作
    if action is None:
        if "减仓" in text or "止盈" in text or "止损" in text:
            action = "减仓"
        elif "加仓" in text:
            action = "加仓"
        elif "观望" in text:
            action = "观望"
        elif "持有" in text or "继续持有" in text:
            action = "持有"
    if risk is None:
        if re.search(r"风险\s*(较?高|偏高|高企)", text) or "风险高" in text:
            risk = "高"
        elif re.search(r"风险\s*(较?低|偏低)", text) or "风险低" in text:
            risk = "低"
        elif "风险" in text:
            risk = "中"
    return {"action": action, "risk": risk}


def _has_risk_section(text: str) -> bool:
    """判断文本是否含【风险】段且该段有实际内容(非空、非仅标签)。"""
    m = re.search(r"【风险】([\s\S]*)$", text or "")
    if not m:
        return False
    # 去掉尾部固定标注/免责声明行后,看是否还有正文
    body = m.group(1)
    body = re.split(r"操作[:：]", body)[0]
    body = re.sub(r"[\s\u3000·。、,,;;]+", "", body)
    return len(body) >= 6  # 至少要有一句实质内容


def _fnum(v):
    """安全取 float:None/NaN 返回 None。"""
    try:
        if v is None or v != v:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _auto_risk_lines(facts: dict) -> str:
    """当 AI 漏掉【风险】段时,依据客观事实程序补一段标准风险提示。
    覆盖高位偏热(追高回吐)与低位空头(下行套牢)两类场景,只陈述事实里
    已有的信号,不编造。ETF 走基金语境分支(无套牢盘概念)。"""
    if facts.get("is_etf"):
        return _auto_risk_lines_etf(facts)
    pts = []
    rsi = _fnum(facts.get("rsi"))
    pos = _fnum(facts.get("pos_60"))
    cpr = _fnum(facts.get("chip_profit_ratio"))
    pe = _fnum(facts.get("pe_ttm"))
    ma = str(facts.get("ma_state") or "")
    macd = str(facts.get("macd_state") or "")
    weekly = str(facts.get("weekly_state") or "")

    # —— 高位/偏热类(追高、获利回吐)——
    if rsi is not None and rsi >= 65:
        lvl = "超买" if rsi >= 70 else "偏热"
        pts.append(f"RSI(14)={rsi:.1f} 已{lvl}(≥65),短线追高需防获利回吐")
    if pos is not None and pos >= 80:
        pts.append(f"价格处于近60日 {pos:.0f}% 高分位,属相对高位,谨防冲高回落")
    if cpr is not None and cpr >= 0.85:
        pts.append(f"获利盘约 {cpr * 100:.0f}%,一旦风吹草动易引发集中获利了结")

    # —— 低位/空头类(下行、套牢)——
    if "空头" in ma:
        pts.append("日线呈空头排列,均线系统压制,短期仍有下行压力")
    if "空头" in weekly:
        pts.append("周线亦为空头,中期趋势偏弱,反弹易受上方套牢盘压制")
    elif "空头" in macd and "空头" not in ma:
        pts.append("MACD 位于空头区,动能偏弱,反弹持续性存疑")
    if pos is not None and pos <= 20:
        pts.append(f"价格处于近60日 {pos:.0f}% 低位,弱势中抄底需防'越跌越买'扩大浮亏")
    if cpr is not None and cpr <= 0.10:
        pts.append(f"获利盘仅约 {cpr * 100:.0f}%,几乎全员套牢,上方抛压沉重、反弹阻力大")

    # —— 估值类(多空通用)——
    if pe is not None and pe < 0:
        pts.append(f"公司当前 PE 为负({pe:.1f},处于亏损),估值缺乏盈利支撑")

    # 去重并保序,不足 2 条补一句通用纪律
    seen, uniq = set(), []
    for p in pts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    if len(uniq) < 2:
        uniq.append("需关注大盘系统性风险及个股自身波动,严守个人止盈止损纪律")
    return "【风险】\n" + "".join(f"- {p};\n" for p in uniq)


def _auto_risk_lines_etf(facts: dict) -> str:
    """ETF 版风险兜底:基金语境,不使用'套牢盘/获利盘'措辞,
    改用趋势/折溢价/规模/位置等 ETF 适用维度。只陈述事实,不编造。"""
    pts = []
    rsi = _fnum(facts.get("rsi"))
    pos = _fnum(facts.get("pos_60"))
    ma = str(facts.get("ma_state") or "")
    macd = str(facts.get("macd_state") or "")
    weekly = str(facts.get("weekly_state") or "")
    dr = _fnum(facts.get("etf_discount_rate"))
    scale = _fnum(facts.get("etf_scale_yi"))

    # —— 趋势/动能类 ——
    if "空头" in ma:
        pts.append("日线呈空头排列,均线系统压制,短期仍有下行压力")
    if "空头" in weekly:
        pts.append("周线亦为空头,中期趋势偏弱,反弹动能有限")
    elif "空头" in macd and "空头" not in ma:
        pts.append("MACD 位于空头区,动能偏弱,反弹持续性存疑")
    # —— 位置类 ——
    if pos is not None and pos <= 20:
        pts.append(f"价格处于近60日 {pos:.0f}% 低位,弱势中抄底需防'越跌越买'扩大浮亏")
    if rsi is not None and rsi >= 65:
        lvl = "超买" if rsi >= 70 else "偏热"
        pts.append(f"RSI(14)={rsi:.1f} 已{lvl}(≥65),短线追高需防冲高回落")
    if pos is not None and pos >= 80:
        pts.append(f"价格处于近60日 {pos:.0f}% 高分位,属相对高位,谨防回落")
    # —— ETF 专属:折溢价 / 规模 ——
    if dr is not None and dr <= -0.5:
        pts.append(f"当前溢价约 {abs(dr):.2f}%(现价高于净值),存在向净值回归的风险")
    if scale is not None and scale < 2.0:
        pts.append(f"基金规模仅约 {scale:.2f} 亿,偏小,存在流动性不足或清盘风险")
    # —— 通用 ——
    pts.append("ETF 净值随所跟踪指数/行业波动,需关注板块系统性回调风险")

    seen, uniq = set(), []
    for p in pts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    if len(uniq) < 2:
        uniq.append("需关注大盘系统性风险及标的自身波动,严守个人止盈止损纪律")
    return "【风险】\n" + "".join(f"- {p};\n" for p in uniq)


def _ensure_risk_section(text: str, facts: dict):
    """保证输出含【风险】段。缺失则程序补一段,返回 (新文本, 补充片段或None)。
    补充片段用于流式场景补推给 UI。"""
    if _has_risk_section(text):
        return text, None
    supp = "\n\n" + _auto_risk_lines(facts).rstrip("\n")
    # 若原文末尾已有'操作:.. | 风险:..'标注行,把补充段插到该标注行之前
    m = re.search(r"\n?\s*操作[:：]\s*(持有|加仓|减仓|观望)\s*[|｜]\s*风险",
                  text or "")
    if m:
        idx = m.start()
        new_text = text[:idx].rstrip() + supp + "\n" + text[idx:].lstrip("\n")
    else:
        new_text = (text or "").rstrip() + supp
    return new_text, supp


def _holding_position_line(f: dict, buy_price) -> str:
    """构造持仓上下文行(买入价/现价/浮盈)。buy_price 无效时返回空。"""
    try:
        bp = float(buy_price)
    except (TypeError, ValueError):
        bp = 0.0
    if bp <= 0:
        return ""
    cur = f.get("close")
    try:
        cur = float(cur)
    except (TypeError, ValueError):
        return f"- 持仓成本: 买入价={_fmt(bp)}(现价未取到,无法计算浮动盈亏)\n"
    pnl = (cur - bp) / bp * 100 if bp else 0.0
    zt = "浮盈" if pnl > 0 else ("浮亏" if pnl < 0 else "持平")
    # 明确止盈/止损语义, 避免 AI 在浮亏时误用"止盈"(浮亏减仓=止损, 浮盈减仓=止盈)
    if pnl > 0:
        hint = "  [当前浮盈, 若谈减仓应称'止盈', 不得称'止损']"
    elif pnl < 0:
        hint = "  [当前浮亏, 若谈减仓应称'止损'或'减仓控制风险', 严禁称'止盈']"
    else:
        hint = ""
    return (f"- 持仓成本: 买入价={_fmt(bp)}  当前价={_fmt(cur)}  "
            f"浮动盈亏={_fmt(pnl)}%({zt}){hint}\n")


def comment_holding(code: str, buy_price=0.0, on_delta=None) -> dict:
    """对【已持仓】的单只股票生成 AI 点评,结合买入成本给持有/加减仓倾向。

    返回 {"ok":True, "facts":..., "text":..., "action":持有/加仓/减仓/观望,
          "risk":高/中/低, "buy_price":.., "pnl":.., "disclaimer":...}
      或 {"ok":False, "error":...}。
    - 不做当天缓存(浮盈随实时价变,且带用户成本,不宜跨次复用);
    - on_delta 提供时走流式,失败自动回退普通调用;
    - 调用方(UI)负责放到后台线程执行。
    """
    facts = build_facts(code)
    if "error" in facts:
        return {"ok": False, "error": facts["error"]}

    pos_line = _holding_position_line(facts, buy_price)
    try:
        bp = float(buy_price)
    except (TypeError, ValueError):
        bp = 0.0
    cur = facts.get("close")
    try:
        cur = float(cur)
    except (TypeError, ValueError):
        cur = None
    pnl = ((cur - bp) / bp * 100) if (cur and bp > 0) else None

    is_etf = bool(facts.get("is_etf"))
    if is_etf:
        user = (
            "我【已经持有】下面这只 ETF,请结合我的买入成本,帮我诊断这笔仓位并给出"
            "仓位管理倾向(以下均为系统算好/实时拉取的客观事实):\n\n"
            f"{pos_line}{facts_to_lines(facts)}\n\n"
            "请严格按【持仓诊断】【技术面】【操作倾向】【风险】四段输出,ETF 无个股财报与"
            "筹码套牢盘概念,谈估值贵贱只用折溢价率,"
            "并在【操作倾向】段末尾用固定格式标注:  操作:持有/加仓/减仓/观望 | 风险:高/中/低"
        )
        sys_prompt = ETF_HOLDING_SYSTEM_PROMPT
    else:
        user = (
            "我【已经持有】下面这只股票,请结合我的买入成本,帮我诊断这笔仓位并给出"
            "仓位管理倾向(以下均为系统算好/实时拉取的客观事实):\n\n"
            f"{pos_line}{facts_to_lines(facts)}\n\n"
            "请严格按【持仓诊断】【技术面】【操作倾向】【风险】四段输出,"
            "并在【操作倾向】段末尾用固定格式标注:  操作:持有/加仓/减仓/观望 | 风险:高/中/低"
        )
        sys_prompt = HOLDING_SYSTEM_PROMPT
    prompt = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]
    try:
        if on_delta is not None:
            try:
                text = chat_stream(prompt, on_delta)
            except AIError:  # noqa
                text = chat(prompt)
        else:
            text = chat(prompt)
    except AIError as e:  # noqa
        return {"ok": False, "error": str(e), "facts": facts}

    # 兜底:AI 偶尔漏掉独立【风险】段(纯提示词压不住采样波动),
    # 缺失则依据客观事实程序补一段,保证结构 100% 有风险提示。
    text, supp = _ensure_risk_section(text, facts)
    if supp and on_delta is not None:
        try:
            on_delta(supp)   # 流式场景把补充段也推给 UI,保持显示一致
        except Exception:
            pass

    r = parse_holding_action(text)
    return {"ok": True, "facts": facts, "text": text,
            "action": r["action"], "risk": r["risk"],
            "buy_price": bp, "pnl": pnl,
            "disclaimer": ETF_DISCLAIMER if is_etf else DISCLAIMER}
