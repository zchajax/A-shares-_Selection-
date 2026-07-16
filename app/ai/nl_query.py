"""自然语言选股:把用户的人话翻译成量化引擎可执行的"策略+参数+基本面过滤"。

核心红线(与项目一贯原则一致):
  AI 不碰全市场数据、不直接挑票、不产可执行代码。它只做一件事——
  把"人话"翻译成一个受约束的 JSON 参数字典;真正的筛选由本地量化引擎
  (scanner + funda 过滤器)执行,所以结果 100% 可复现、可回测。

翻译产物 schema(AI 只能在此范围内输出):
  {
    "strategy": "multi_factor" | "breakout" | ...  (必须是 ALL_STRATEGIES 的 key),
    "params":   {参数key: 值},                      (仅限该策略 params 声明的 key),
    "funda":    {pe_max, pb_max, roe_min, mv_min, mv_max}  (基本面后置过滤,可空),
    "top_n":    整数,
    "explain":  "一句话说明这样翻译的理由"
  }

任何越界(未知策略/参数/字段)都会被校验层丢弃或钳制,AI 无法让引擎做它不该做的事。
"""

import json
import re

from app.strategy import base
from .client import chat, AIError


# 基本面过滤支持的键(与 funda.apply_filter 对齐),以及合理范围钳制
_FUNDA_KEYS = {
    "pe_max": (0, 1000),
    "pb_max": (0, 100),
    "roe_min": (-100, 100),
    "mv_min": (0, 100000),   # 亿元
    "mv_max": (0, 100000),
}


def _strategy_catalog() -> str:
    """把可用策略与其参数(key/说明/默认/范围)整理成给 AI 看的目录。"""
    lines = []
    for key, cls in base.ALL_STRATEGIES.items():
        params = getattr(cls, "params", []) or []
        pdesc = "; ".join(
            f"{p.key}({p.label},默认{p.default},范围{p.min}~{p.max})"
            for p in params
        ) or "无可调参数"
        lines.append(f'- "{key}": {cls.name} —— {cls.desc}\n    参数: {pdesc}')
    return "\n".join(lines)


def _system_prompt() -> str:
    return (
        "你是一个 A 股量化选股的【意图翻译器】。用户用自然语言描述想找什么样的股票,"
        "你要把它翻译成量化引擎可执行的 JSON 参数,而【不是】自己去挑股票、不给股票代码。\n\n"
        "可用策略目录(strategy 只能取以下 key 之一):\n"
        f"{_strategy_catalog()}\n\n"
        "基本面后置过滤(funda,可选,不需要就省略对应键):\n"
        "- pe_max: 市盈率TTM上限(找'估值不高/便宜'时用,如 pe_max:30)\n"
        "- pb_max: 市净率上限\n"
        "- roe_min: 净资产收益率%下限(找'盈利能力强'时用,如 roe_min:10)\n"
        "- mv_min: 总市值下限(亿元,找'大盘股'时用)\n"
        "- mv_max: 总市值上限(亿元,找'中小盘/市值XX以内'时用)\n\n"
        "翻译规则:\n"
        "1. 先根据用户描述的【技术形态】选最贴切的一个策略(如'突破新高'→breakout,"
        "'金叉'→macd_cross,'超跌反弹'→oversold_rebound,'回踩均线/低吸'→pullback,"
        "'放量上涨/主升'→vol_price_rise,'综合/稳健/好票'→multi_factor)。\n"
        "2. 用户提到的技术数值(如放量倍数、涨幅、周期天数)映射到该策略的参数 key,"
        "不在范围内就钳到范围内;没提到的参数不要写(用默认)。\n"
        "3. 用户提到的【估值/盈利/市值】诉求映射到 funda。'市值300亿以内'→mv_max:300;"
        "'估值不高'→pe_max 给个合理值(如30);'盈利能力强'→roe_min(如10)。\n"
        "4. top_n: 用户说'前10/10只'就 10,没说默认 20。\n"
        "5. 只输出一个 JSON 对象,不要任何多余文字、不要代码块标记、不要股票代码。\n"
        "6. 如果用户的话完全无法对应任何策略,strategy 取 \"multi_factor\" 兜底,"
        "并在 explain 里说明。\n\n"
        "输出格式(严格 JSON):\n"
        '{"strategy":"...","params":{},"funda":{},"top_n":20,"explain":"..."}'
    )


def _extract_json(text: str) -> dict:
    """从模型输出里稳健地抽出第一个 JSON 对象(容忍 ```json 包裹或前后噪声)。"""
    if not text:
        raise ValueError("空响应")
    # 去掉 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))
    # 否则找第一个 { 到最后一个 } 之间
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        return json.loads(text[s:e + 1])
    raise ValueError(f"未找到 JSON: {text[:120]}")


def _sanitize(spec: dict) -> dict:
    """严格校验/钳制 AI 翻译结果,过滤一切越界,保证引擎只做被允许的事。"""
    # 1) 策略:必须是已注册 key,否则兜底 multi_factor
    strat = spec.get("strategy")
    if strat not in base.ALL_STRATEGIES:
        strat = "multi_factor"

    # 2) 参数:只保留该策略声明的 key,并钳到 [min,max];类型按 is_int 归一
    cls = base.ALL_STRATEGIES[strat]
    allowed = {p.key: p for p in (getattr(cls, "params", []) or [])}
    clean_params = {}
    for k, v in (spec.get("params") or {}).items():
        if k not in allowed:
            continue
        p = allowed[k]
        try:
            v = float(v)
        except Exception:  # noqa
            continue
        v = max(p.min, min(p.max, v))          # 钳制
        clean_params[k] = int(round(v)) if p.is_int else round(v, 3)

    # 3) 基本面过滤:只保留已知键并钳到合理范围
    clean_funda = {}
    for k, v in (spec.get("funda") or {}).items():
        if k not in _FUNDA_KEYS:
            continue
        try:
            v = float(v)
        except Exception:  # noqa
            continue
        lo, hi = _FUNDA_KEYS[k]
        clean_funda[k] = max(lo, min(hi, v))

    # 4) top_n
    try:
        top_n = int(spec.get("top_n", 20))
    except Exception:  # noqa
        top_n = 20
    top_n = max(1, min(100, top_n))

    return {
        "strategy": strat,
        "strategy_name": cls.name,
        "params": clean_params,
        "funda": clean_funda,
        "top_n": top_n,
        "explain": str(spec.get("explain", ""))[:200],
    }


def translate(query: str) -> dict:
    """把自然语言查询翻译成受约束的选股参数字典。

    返回 {"ok": True, "spec": {...}, "raw": "..."} 或 {"ok": False, "error": "..."}。
    spec 已经过 _sanitize,可直接交给 run_nl_scan 执行。
    """
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "请输入选股描述"}
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": f"用户描述:{query}\n\n请翻译成 JSON 参数。"},
    ]
    try:
        raw = chat(messages, temperature=0.1, max_tokens=400)
    except AIError as e:  # noqa
        return {"ok": False, "error": str(e)}
    try:
        spec = _sanitize(_extract_json(raw))
    except Exception as e:  # noqa
        return {"ok": False, "error": f"AI 未能翻译成有效参数({e});请换个说法再试", "raw": raw}
    return {"ok": True, "spec": spec, "raw": raw}


def build_strategy(spec: dict):
    """根据 spec 构造一个参数已设置好的策略实例(供 scanner.scan 使用)。"""
    cls = base.ALL_STRATEGIES[spec["strategy"]]
    strat = cls()
    for k, v in spec.get("params", {}).items():
        strat.set_param(k, v)
    return strat


def run_nl_scan(query: str, progress_cb=None) -> dict:
    """自然语言选股全流程:翻译 → 量化引擎筛选 → 基本面过滤 → 截断 top_n。

    返回 {"ok": True, "spec": {...}, "df": DataFrame, "explain": "..."} 或
         {"ok": False, "error": "..."}。df 列: code,name,score,close,reason(+基本面列)。
    """
    tr = translate(query)
    if not tr["ok"]:
        return tr
    spec = tr["spec"]

    from app.strategy import scanner
    from app.strategy import funda

    strat = build_strategy(spec)
    df = scanner.scan(strat, progress_cb=progress_cb)

    # 基本面后置过滤(有条件才过滤)
    if spec.get("funda"):
        try:
            df = funda.apply_filter(df, spec["funda"])
        except Exception:  # noqa 过滤异常不致命,退回未过滤结果
            pass

    if df is not None and not df.empty:
        df = df.head(spec["top_n"]).reset_index(drop=True)

    return {"ok": True, "spec": spec, "df": df, "explain": spec.get("explain", "")}
