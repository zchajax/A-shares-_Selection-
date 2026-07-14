"""AI 精排(路径 B):对量化引擎选出的一批候选股做二次优选排序。

定位: 量化引擎(scanner)负责"从全市场筛出候选"这一步(碰全市场数据、
可复现); 本模块只把候选池里每只票的【客观事实】喂给 AI, 让它横向比较后
给出 Top-N 排名 + 一句话入选理由 + 综合评级。AI 全程不产出新代码、不碰
原始行情, 只在"给定候选集合"内排序, 因此结果不会编造、可解释。

与逐只点评(commentary.comment_stock)的区别:
- 逐只点评: 一次一只, 深度长文, 无法横向比较;
- 精排: 一批一起, 每只压成一行紧凑事实, AI 横向比较后统一排序(省 token、
  且"谁比谁好"才是排序的关键)。

红线: prompt 明令 AI 只能在给定代码列表内排序; 解析结果时再做一次
"代码必须来自候选池"的白名单校验, 双保险防止编造。
"""

import json
import re

from app.data import database as db
from .client import chat, AIError
from . import commentary as cm

DISCLAIMER = ("以上为 AI 依据量化候选池的客观指标做的二次优选排序,"
              "仅供研究参考,不构成任何投资建议。")

# 一次最多喂给 AI 的候选数量(过多会超 token 且稀释质量; 量化结果通常取前若干)
MAX_CANDIDATES = 25

SYSTEM_PROMPT = (
    "你是一名严谨的 A 股组合优选分析师。用户会给你一批【已由量化系统从全市场"
    "筛选出来的候选股】, 每只附带客观指标事实(技术面 + 基本面 + 行业对比)。\n"
    "你的任务: 在这批候选中横向比较, 挑出综合质量最高的若干只并排序。\n"
    "严格规则:\n"
    "1. 只能从给定的候选代码里挑选和排序, 绝对不得新增、替换或编造任何代码;\n"
    "2. 不得给出买入/卖出/加减仓建议, 不得预测目标价或涨跌幅;\n"
    "3. 只依据给出的事实, 不得虚构财报、消息面、新闻等未提供的信息;\n"
    "4. 排序依据要兼顾: 趋势与动能(技术面)、估值是否合理(结合行业分位)、"
    "盈利与成长质量(ROE/毛利/营收净利增速)、风险(负债/亏损/超买等)。\n"
    "输出必须是严格的 JSON(不要 markdown 代码块、不要多余文字), 形如:\n"
    '{"ranking":[{"code":"600519","rating":"偏多","risk":"中",'
    '"reason":"趋势多头且估值处行业中位,盈利稳健"}, ...]}\n'
    "其中 rating 取值仅限 偏多/中性/偏空, risk 仅限 高/中/低, "
    "reason 为不超过 30 字的一句话入选理由。ranking 按优选程度从高到低排列。"
)


def _compact_line(f: dict) -> str:
    """把一只票的事实压成一行紧凑文本(供批量横向比较, 尽量省 token)。"""
    def n(x, nd=1):
        try:
            if x is None or x != x:
                return "-"
            return f"{float(x):.{nd}f}"
        except Exception:  # noqa
            return "-"

    parts = [
        f"{f['code']} {f['name']}({f.get('industry', '未知')})",
        f"现价{n(f.get('close'), 2)} 当日{n(f.get('day_chg'))}%",
        f"5日{n(f.get('chg_5'))}% 20日{n(f.get('chg_20'))}%",
        f["ma_state"].replace("(", "").replace(")", ""),
        f["macd_state"],
        f"RSI{n(f.get('rsi'))}({f.get('rsi_state')})",
        f"量比{n(f.get('vol_ratio'))}({f.get('vol_state')})",
        f"60日位置{n(f.get('pos_60'), 0)}%",
    ]
    # 基本面(有才加)
    fund = []
    if f.get("pe_ttm") is not None:
        fund.append(f"PE{n(f.get('pe_ttm'))}")
    if f.get("pb") is not None:
        fund.append(f"PB{n(f.get('pb'), 2)}")
    if f.get("roe") is not None:
        fund.append(f"ROE{n(f.get('roe'))}%")
    if f.get("rev_yoy") is not None:
        fund.append(f"营收同比{n(f.get('rev_yoy'))}%")
    if f.get("profit_yoy") is not None:
        fund.append(f"净利同比{n(f.get('profit_yoy'))}%")
    if f.get("debt_ratio") is not None:
        fund.append(f"负债率{n(f.get('debt_ratio'))}%")
    pe_pct = f.get("pe_pct")
    if pe_pct:
        fund.append(f"PE行业分位{pe_pct['percentile']:.0f}%")
    if fund:
        parts.append("; ".join(fund))
    else:
        parts.append("基本面未获取")
    return " | ".join(parts)


def _extract_json(text: str) -> dict:
    """从模型输出里稳健地抽出 JSON(容忍 ```json 代码块或前后杂字)。"""
    if not text:
        raise ValueError("空响应")
    # 去掉可能的 markdown 代码块围栏
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(),
               flags=re.MULTILINE).strip()
    try:
        return json.loads(t)
    except Exception:  # noqa
        pass
    # 兜底: 抓第一个 {...} 大括号块
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"无法解析为 JSON, 原文前120字: {text[:120]}")


def rank_stocks(codes: list, top_n: int = 10, strategy_hint: str = "",
                progress_cb=None) -> dict:
    """对候选股列表做 AI 精排。

    codes: 候选股代码列表(通常来自 scanner 结果的前若干只)。
    top_n: 最终返回排名的数量(不超过候选数)。
    progress_cb(done, total): 可选, 汇报"正在收集第 done/total 只事实"。

    返回:
      {"ok": True, "ranking": [ {code,name,industry,rating,risk,reason,
        close,facts}, ... ], "disclaimer": "...", "n_candidates": N}
      {"ok": False, "error": "原因"}

    健壮性: 单只事实收集失败自动跳过; AI 返回的代码若不在候选白名单内一律
    丢弃(防编造); 完全解析失败则回退为"按候选原序"给出结果并标注 degraded。
    """
    codes = [c for c in dict.fromkeys(codes) if c]  # 去重保序
    if not codes:
        return {"ok": False, "error": "候选股为空, 请先执行选股。"}
    codes = codes[:MAX_CANDIDATES]

    # 1) 收集每只票的事实(复用 commentary.build_facts)
    facts_map = {}
    total = len(codes)
    for i, code in enumerate(codes):
        try:
            f = cm.build_facts(code)
            if "error" not in f:
                facts_map[code] = f
        except Exception:  # noqa 单只失败跳过, 不影响整体
            pass
        if progress_cb:
            progress_cb(i + 1, total)
    if not facts_map:
        return {"ok": False, "error": "候选股均无足够本地数据, 无法精排。"}

    valid_codes = list(facts_map.keys())
    lines = [f"{idx + 1}. {_compact_line(facts_map[c])}"
             for idx, c in enumerate(valid_codes)]
    hint = f"\n量化系统备注: {strategy_hint}" if strategy_hint else ""
    user = (
        f"以下是量化系统选出的 {len(valid_codes)} 只候选股(均为客观事实):\n\n"
        + "\n".join(lines)
        + hint
        + f"\n\n请在以上候选中横向比较, 选出综合质量最高的前 {top_n} 只并排序, "
        "严格按系统要求的 JSON 格式输出(只能用上面出现过的代码)。"
    )

    # 2) 调 AI
    try:
        raw = chat([{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}],
                   temperature=0.2, max_tokens=900)
    except AIError as e:  # noqa
        return {"ok": False, "error": str(e)}

    # 3) 解析 + 白名单校验(防编造)
    degraded = False
    ranking = []
    white = set(valid_codes)
    try:
        obj = _extract_json(raw)
        for item in obj.get("ranking", []):
            code = str(item.get("code", "")).strip()
            if code not in white:
                continue  # 不在候选池 → 丢弃(AI 编造/写错)
            ranking.append({
                "code": code,
                "rating": item.get("rating") or "",
                "risk": item.get("risk") or "",
                "reason": (item.get("reason") or "").strip()[:40],
            })
    except Exception:  # noqa 解析彻底失败 → 降级为原序
        degraded = True

    seen = {r["code"] for r in ranking}
    if degraded or not ranking:
        # 降级: 按候选原顺序(通常已是量化得分序)兜底给出
        degraded = True
        ranking = [{"code": c, "rating": "", "risk": "",
                    "reason": "AI 排序解析失败, 暂按量化得分序展示"}
                   for c in valid_codes[:top_n]]
    else:
        # AI 漏排的候选补在末尾(不丢票)
        for c in valid_codes:
            if c not in seen:
                ranking.append({"code": c, "rating": "", "risk": "",
                                "reason": "(AI 未列入优选)"})

    ranking = ranking[:top_n]
    # 4) 补充展示所需字段(名称/行业/现价)
    for r in ranking:
        f = facts_map.get(r["code"], {})
        r["name"] = f.get("name") or db.name_of(r["code"]) or ""
        r["industry"] = f.get("industry") or ""
        r["close"] = f.get("close")
        r["facts"] = f

    return {"ok": True, "ranking": ranking, "disclaimer": DISCLAIMER,
            "n_candidates": len(valid_codes), "degraded": degraded}
