"""
本地 AI 智能选基与智能选股引擎 (app/ai/fund_screener.py)
===========================================================
完全基于本地 config.json (app/ai/client.py) 与本地量化数据库运行，不依赖任何外部三方 Agent。

核心三阶段流水线 (以 AI 选基为例)：
  第 1 阶段【意图解析 (LLM + 本地金融正则双引擎)】：
      将用户自然语言（如：“帮我选几个年化百分之3左右，最大回撤百分之二，基金规模10-50亿，机构占比不超过百分之八十的基金”）
      解析为严格受约束的结构化过滤条件 JSON：
      {
        "fund_type": "mutual" | "etf",
        "category": "债券/固收" | "宽基指数" | ...,
        "ret_1y_min": 2.2, "ret_1y_max": 4.8,
        "ret_3y_min": null,
        "max_dd_max": 2.0,
        "scale_min": 10.0, "scale_max": 50.0,
        "inst_ratio_max": 80.0,
        "fee_max": null,
        "keyword": "",
        "top_n": 15,
        "explain": "..."
      }
  第 2 阶段【本地 20,000+ 基金初筛 + 16 线程极速深度核验 (Scale / InstRatio / MaxDD)】：
      先在本地 mutual_fund_rank / etf_snapshot 表中按收益率、赛道、费率初筛候选池；
      若用户指定了「最大回撤」、「基金规模」或「机构占比」，自动对候选池中尚未缓存深度指标的基金
      并发请求东财 pingzhongdata (16线程，80只仅需 1.5~2s)，精确核验并永久写回 SQLite 数据库！
  第 3 阶段【AI 优选排序与专业解读】：
      将严格满足全部硬指标的真实基金喂给本地配置的 AI 模型，生成组合建议与逐只亮点点评。
"""
from __future__ import annotations

import json
import re
from typing import Optional

import numpy as np
import pandas as pd

from app.data import database as db
from app.data import fetcher
from .client import chat, is_configured, AIError


FUND_NL_SYSTEM_PROMPT = (
    "你是一名专业的公募基金与 ETF 量化筛选意图翻译器。用户会用自然语言描述选基条件，"
    "请将其翻译为严格的 JSON 过滤参数（不要输出任何多余解释或 markdown 代码块标记）。\n\n"
    "支持的 JSON 字段如下（未提及的条件请设为 null）：\n"
    "- fund_type: \"mutual\"(默认，场外公募基金) 或 \"etf\"(仅当用户明确提到 ETF/场内 时)\n"
    "- category: 只能取 [\"全部类别\",\"债券/固收\",\"宽基指数\",\"红利/高股息\",\"科技/AI/芯片\","
    "\"医药/医疗\",\"大消费/食品\",\"新能源/汽车\",\"金融/地产\",\"军工/制造/周期\",\"跨境/海外\",\"商品/黄金\",\"混合/其他主题\"] 之一。"
    "注意：若用户要求年化2%~6%左右且最大回撤很小(<=3%)，通常隐含稳健债基，category 可设为 \"债券/固收\"\n"
    "- ret_1y_min: 近1年(或年化)收益率下限(%)。若用户说“年化3%左右”，请给区间 ret_1y_min: 2.2, ret_1y_max: 4.8\n"
    "- ret_1y_max: 近1年(或年化)收益率上限(%)\n"
    "- ret_3y_min: 近3年收益率下限(%)\n"
    "- max_dd_max: 最大回撤上限(%，正数，如“最大回撤2%以内”填 2.0)\n"
    "- scale_min: 基金规模下限(亿元，如“10-50亿”则 scale_min: 10.0)\n"
    "- scale_max: 基金规模上限(亿元，如“10-50亿”则 scale_max: 50.0)\n"
    "- inst_ratio_max: 机构持有人占比上限(%，如“机构占比不超过80%”填 80.0)\n"
    "- fee_max: 手续费率上限(%)\n"
    "- keyword: 名称关键字(如“国债”、“黄金”、“沪深300”，无则空串 \"\")\n"
    "- top_n: 返回只数(默认 15)\n"
    "- explain: 一句话说明解析出的筛选逻辑\n\n"
    "必须只输出一行合法 JSON，例如：\n"
    "{\"fund_type\":\"mutual\",\"category\":\"债券/固收\",\"ret_1y_min\":2.2,\"ret_1y_max\":4.8,"
    "\"ret_3y_min\":null,\"max_dd_max\":2.0,\"scale_min\":10.0,\"scale_max\":50.0,"
    "\"inst_ratio_max\":80.0,\"fee_max\":null,\"keyword\":\"\",\"top_n\":15,\"explain\":\"筛选年化2.2%~4.8%、最大回撤<=2%、规模10~50亿、机构占比<=80%的稳健债券基金\"}"
)

FUND_REVIEW_SYSTEM_PROMPT = (
    "你是一名专业的 FOF 基金投研顾问。用户给出了选基需求，量化引擎已经严格按硬指标从 20,000+ 只基金中"
    "筛选并核验出以下完全达标的真实基金列表（含近1年/近3年收益、真实复权最大回撤、最新规模、机构持有占比）。\n"
    "请用简洁、客观、专业的语言（控制在 220 字以内）输出：\n"
    "【筛选总结】1-2句话总结本批达标基金的整体风险收益特征；\n"
    "【重点标的亮点】挑出 2-3 只综合性价比最高（如回撤极小、夏普高、规模适中、机构占比健康）的基金，点明其代码、名称与核心优势；\n"
    "【配置提示】1句话提示申赎费率或持有期注意事项。严禁编造未给出的数据。"
)


_CN_NUM_MAP = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _cn_to_float(s: str) -> Optional[float]:
    """把中文数字(如 '三'、'二十'、'八十'、'3.5')转为 float。"""
    s = str(s).strip()
    try:
        return float(s)
    except Exception:
        pass
    if not s:
        return None
    if s == "十":
        return 10.0
    if "十" in s:
        parts = s.split("十")
        tens = _CN_NUM_MAP.get(parts[0], 1) if parts[0] else 1
        ones = _CN_NUM_MAP.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return float(tens * 10 + ones)
    if len(s) == 1 and s in _CN_NUM_MAP:
        return float(_CN_NUM_MAP[s])
    return None


def _normalize_cn_numbers(text: str) -> str:
    """将 '百分之八十'、'百分之3'、'百分之二' 等统一替换为 '80%'、'3%'、'2%' 便于正则提取。"""
    def _repl(m):
        raw = m.group(1)
        v = _cn_to_float(raw)
        return f"{v}%" if v is not None else m.group(0)
    out = re.sub(r"百分之([零一二两三四五六七八九十0-9.]+)", _repl, text)
    return out


def _base_fund_name(name: str) -> str:
    """提取基金主名称（去除末尾 A/B/C/D/E/H/I 份额后缀），用于同基金多份额去重。"""
    s = str(name or "").strip()
    s = re.sub(r"(\([A-Za-z0-9]+\))?[A-Za-z](/[A-Za-z])?$", "", s)
    s = re.sub(r"(发起式|发起)$", "", s)
    return s.strip() or str(name or "")


def parse_fund_query_fallback(query: str) -> dict:
    """
    本地高精度金融语义正则解析器（同时解析数值约束与定性语义如“成立时间长/回撤小/排名靠前/高夏普/老将管理/低费率”）。
    """
    q = _normalize_cn_numbers(query or "")
    spec = {
        "fund_type": "etf" if ("etf" in q.lower() or "场内" in q) else "mutual",
        "category": "全部类别",
        "ret_1y_min": None,
        "ret_1y_max": None,
        "ret_3y_min": None,
        "ret_since_min": None,
        "max_dd_max": None,
        "max_dd_all_max": None,
        "scale_min": None,
        "scale_max": None,
        "inst_ratio_max": None,
        "establish_years_min": None,
        "sharpe_1y_min": None,
        "vol_1y_max": None,
        "manager_years_min": None,
        "manager_kw": "",
        "fee_max": None,
        "long_history": False,
        "low_dd_pref": False,
        "top_rank": False,
        "keyword": "",
        "top_n": 100,
        "explain": "",
    }

    m_topn = re.search(r"(?:前|选|挑)\s*([0-9]+)\s*(?:只|个|支)", q)
    if m_topn:
        spec["top_n"] = max(1, min(200, int(m_topn.group(1))))

    # 1) 识别赛道类别
    for cat_k, cat_v in (
        ("债", "债券/固收"), ("固收", "债券/固收"), ("纯债", "债券/固收"),
        ("红利", "红利/高股息"), ("高股息", "红利/高股息"),
        ("宽基", "宽基指数"), ("沪深300", "宽基指数"), ("中证500", "宽基指数"),
        ("科技", "科技/AI/芯片"), ("半导体", "科技/AI/芯片"), ("芯片", "科技/AI/芯片"), ("人工智能", "科技/AI/芯片"),
        ("医药", "医药/医疗"), ("医疗", "医药/医疗"),
        ("消费", "大消费/食品"), ("白酒", "大消费/食品"),
        ("新能源", "新能源/汽车"), ("光伏", "新能源/汽车"),
        ("黄金", "商品/黄金"), ("商品", "商品/黄金"),
        ("纳指", "跨境/海外"), ("标普", "跨境/海外"), ("恒生", "跨境/海外"), ("海外", "跨境/海外"),
    ):
        if cat_k in q:
            spec["category"] = cat_v
            break

    # 2) 提取年化 / 近1年收益率 (支持 "年化3%左右"、"年化3-5%"、"收益大于5%")
    m_ret_range = re.search(r"(?:年化|近1年|近一年|收益[率]?)\s*([0-9.]+)\s*%?\s*(?:到|至|-|~|—)\s*([0-9.]+)\s*%", q)
    if m_ret_range:
        spec["ret_1y_min"] = float(m_ret_range.group(1))
        spec["ret_1y_max"] = float(m_ret_range.group(2))
    else:
        m_ret_around = re.search(r"(?:年化|近1年|近一年|收益[率]?)[约在]*\s*([0-9.]+)\s*%\s*(?:左右|上下|附近)", q)
        if m_ret_around:
            center = float(m_ret_around.group(1))
            band = max(0.8, center * 0.35)
            spec["ret_1y_min"] = round(max(0.0, center - band), 2)
            spec["ret_1y_max"] = round(center + band * 1.4, 2)
        else:
            m_ret_min = re.search(r"(?:年化|近1年|近一年|收益[率]?)(?:大于|高于|超过|不低于|>=|至少)?\s*([0-9.]+)\s*%", q)
            if m_ret_min:
                val = float(m_ret_min.group(1))
                if "以内" in q[m_ret_min.start() : m_ret_min.end() + 4] or "以下" in q[m_ret_min.start() : m_sc_min.end() + 4 if (m_sc_min := None) else m_ret_min.end() + 4]:
                    spec["ret_1y_max"] = val
                else:
                    spec["ret_1y_min"] = val

    # 3) 提取最大回撤 (如 "最大回撤2%"、"回撤不超过2%"、"回撤2%以内")
    m_dd = re.search(r"回撤[^0-9，,、；;。]{0,6}([0-9.]+)\s*%", q)
    if m_dd:
        spec["max_dd_max"] = float(m_dd.group(1))

    # 4) 提取基金规模 (如 "规模10-50亿"、"规模50亿以上")
    m_sc_range = re.search(r"规模[^0-9，,、；;。]{0,6}([0-9.]+)\s*(?:到|至|-|~|—)\s*([0-9.]+)\s*亿", q)
    if m_sc_range:
        spec["scale_min"] = float(m_sc_range.group(1))
        spec["scale_max"] = float(m_sc_range.group(2))
    else:
        m_sc_min = re.search(r"规模[^0-9，,、；;。]{0,6}([0-9.]+)\s*亿\s*(?:以上|大于|及以上)?", q)
        if m_sc_min:
            val_sc = float(m_sc_min.group(1))
            if "以内" in q[m_sc_min.start() : m_sc_min.end() + 4] or "以下" in q[m_sc_min.start() : m_sc_min.end() + 4]:
                spec["scale_max"] = val_sc
            else:
                spec["scale_min"] = val_sc

    # 5) 提取机构占比 (如 "机构占比不超过80%"、"机构占比80%以下")
    m_inst = re.search(r"机构[^0-9，,、；;。]{0,8}([0-9.]+)\s*%", q)
    if m_inst:
        spec["inst_ratio_max"] = float(m_inst.group(1))

    # 6) 提取成立年限、夏普比率、基金经理年限、低费率
    m_est = re.search(r"成立[^0-9，,、；;。]{0,4}([0-9.]+)\s*年\s*(?:以上|以上)?", q)
    if m_est:
        spec["establish_years_min"] = float(m_est.group(1))
        spec["long_history"] = True

    m_shp = re.search(r"夏普[^0-9，,、；;。]{0,5}([0-9.]+)", q)
    if m_shp:
        spec["sharpe_1y_min"] = float(m_shp.group(1))
    elif any(w in q for w in ("高夏普", "夏普高", "高性价比", "风险收益比高")):
        spec["sharpe_1y_min"] = 1.2 if spec["category"] == "债券/固收" else 0.8

    m_mgr_yr = re.search(r"(?:经理|从业|任职)[^0-9，,、；;。]{0,5}([0-9.]+)\s*年", q)
    if m_mgr_yr:
        spec["manager_years_min"] = float(m_mgr_yr.group(1))
    elif any(w in q for w in ("老将", "资深经理", "经验丰富", "十年老将")):
        spec["manager_years_min"] = 8.0 if "十年" in q else 5.0

    if any(w in q for w in ("免申购费", "0费率", "零费率", "C类")):
        spec["fee_max"] = 0.0
    elif any(w in q for w in ("低费率", "费率低", "手续费低")):
        spec["fee_max"] = 0.08

    # 7) 识别定性语义约束：成立时间长 / 回撤小 / 排名靠前
    if any(w in q for w in ("成立时间长", "成立时长", "成立久", "老牌", "长跑", "穿越牛熊", "三年以上", "5年以上")):
        spec["long_history"] = True
        if spec["establish_years_min"] is None:
            spec["establish_years_min"] = 5.0
        if spec["ret_3y_min"] is None:
            spec["ret_3y_min"] = 11.0 if spec["category"] == "债券/固收" else 15.0
        if spec["ret_since_min"] is None:
            spec["ret_since_min"] = 28.0 if spec["category"] == "债券/固收" else 35.0

    if any(w in q for w in ("回撤小", "低回撤", "回撤低", "稳健", "波动小", "纯债")):
        spec["low_dd_pref"] = True
        if spec["max_dd_max"] is None:
            spec["max_dd_max"] = 1.8 if spec["category"] == "债券/固收" else 8.0
        if spec["category"] == "债券/固收" and spec["max_dd_all_max"] is None:
            spec["max_dd_all_max"] = 4.0

    if any(w in q for w in ("排名靠前", "排名高", "前排", "绩优", "业绩好", "收益高", "表现好", "头部")):
        spec["top_rank"] = True

    # 若要求低回撤(<=3%)且年化目标温和(<=6%)且未显式指定赛道，优先聚焦债券/固收赛道
    if spec["category"] == "全部类别" and spec["max_dd_max"] is not None and spec["max_dd_max"] <= 3.0:
        if spec["ret_1y_max"] is None or spec["ret_1y_max"] <= 7.0:
            spec["category"] = "债券/固收"

    parts = []
    if spec["category"] != "全部类别":
        parts.append(f"赛道={spec['category']}")
    if spec["establish_years_min"] is not None:
        parts.append(f"成立>={spec['establish_years_min']:.0f}年")
    if spec["long_history"]:
        parts.append(f"近3年收益>={spec['ret_3y_min']}%·成立来累计>={spec['ret_since_min']}%")
    if spec["top_rank"]:
        parts.append("多周期业绩靠前")
    if spec["ret_1y_min"] is not None or spec["ret_1y_max"] is not None:
        lo_s = f"{spec['ret_1y_min']}%" if spec["ret_1y_min"] is not None else "不限"
        hi_s = f"{spec['ret_1y_max']}%" if spec["ret_1y_max"] is not None else "不限"
        parts.append(f"近1年收益=[{lo_s}~{hi_s}]")
    if spec["max_dd_max"] is not None:
        dd_str = f"近1年回撤<={spec['max_dd_max']}%"
        if spec.get("max_dd_all_max") is not None:
            dd_str += f"(成立来<={spec['max_dd_all_max']}%)"
        parts.append(dd_str)
    if spec["sharpe_1y_min"] is not None:
        parts.append(f"近1年夏普>={spec['sharpe_1y_min']}")
    if spec["manager_years_min"] is not None:
        parts.append(f"基金经理从业>={spec['manager_years_min']:.0f}年")
    if spec["scale_min"] is not None or spec["scale_max"] is not None:
        parts.append(f"规模=[{spec['scale_min'] or 0}~{spec['scale_max'] or '不限'}亿]")
    if spec["inst_ratio_max"] is not None:
        parts.append(f"机构占比<={spec['inst_ratio_max']}%")
    if spec["fee_max"] is not None:
        parts.append(f"费率<={spec['fee_max']}%")
    spec["explain"] = " · ".join(parts) if parts else "综合绩优筛选"
    return spec


def parse_fund_query(query: str) -> dict:
    """
    优先使用本地高精度金融正则解析器提取数值约束（0毫秒响应）；
    仅当查询语句没有匹配到任何数值或赛道条件时，才调用本地 LLM (config.json) 做语义意图翻译，
    把省下的 LLM 调用时间留给最终的 AI 优选解读，整体速度提升一倍！
    """
    rule_spec = parse_fund_query_fallback(query)
    has_rule_constraint = (
        rule_spec.get("category") != "全部类别"
        or any(
            rule_spec.get(k) is not None
            for k in ("ret_1y_min", "ret_1y_max", "ret_3y_min", "max_dd_max", "scale_min", "scale_max", "inst_ratio_max", "fee_max")
        )
    )
    if has_rule_constraint or not is_configured():
        return rule_spec

    try:
        raw = chat(
            [
                {"role": "system", "content": FUND_NL_SYSTEM_PROMPT},
                {"role": "user", "content": query.strip()},
            ],
            temperature=0.0,
            max_tokens=300,
            retries=1,
        )
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            llm_spec = json.loads(m.group(0))
            for k in ("ret_1y_min", "ret_1y_max", "ret_3y_min", "max_dd_max", "scale_min", "scale_max", "inst_ratio_max", "fee_max"):
                if llm_spec.get(k) is None and rule_spec.get(k) is not None:
                    llm_spec[k] = rule_spec[k]
            if not llm_spec.get("category") or llm_spec.get("category") == "全部类别":
                if rule_spec.get("category") != "全部类别":
                    llm_spec["category"] = rule_spec["category"]
            if not llm_spec.get("explain"):
                llm_spec["explain"] = rule_spec["explain"]
            return llm_spec
    except Exception:
        pass
    return rule_spec


def run_ai_fund_screen(query: str, progress_cb=None, on_rows_ready=None) -> dict:
    """
    执行完整的端到端 AI 自然语言选基：
      1) 解析自然语言条件 spec；
      2) 在本地 20,659 只公募基金库(或 1,696 只 ETF 库)中初筛候选集；
      3) 对候选集并发拉取并核验【最新基金规模、最新机构持有比例、真实复权最大回撤】，写回本地数据库；
      4) 严格过滤所有硬指标，并调用本地 AI 生成投研总结与单基优选亮点。
    """
    if progress_cb:
        progress_cb("🤖 正在解析你的选基条件...")
    spec = parse_fund_query(query)
    ftype = spec.get("fund_type") or "mutual"

    if ftype == "etf":
        df = db.load_etf_snapshot()
    else:
        df = db.load_mutual_fund_rank()

    if df is None or df.empty:
        return {"ok": False, "error": "本地基金库为空，请先点『⚡ 同步最新基金数据』", "spec": spec}

    d = df.copy()
    cat = spec.get("category")
    if cat and cat != "全部类别":
        d = d[d["category"] == cat]
    kw = (spec.get("keyword") or "").strip()
    if kw:
        d = d[d["code"].astype(str).str.contains(kw) | d["name"].astype(str).str.contains(kw, case=False)]

    # 初筛收益率
    r1y_col = "ret_250d" if ftype == "etf" else "ret_1y"
    r1y_min = spec.get("ret_1y_min")
    r1y_max = spec.get("ret_1y_max")
    if r1y_min is not None:
        d = d[pd.to_numeric(d[r1y_col], errors="coerce") >= float(r1y_min)]
    if r1y_max is not None:
        d = d[pd.to_numeric(d[r1y_col], errors="coerce") <= float(r1y_max)]

    r3y_min = spec.get("ret_3y_min")
    if r3y_min is not None and "ret_3y" in d.columns:
        d = d[pd.to_numeric(d["ret_3y"], errors="coerce") >= float(r3y_min)]

    rsince_min = spec.get("ret_since_min")
    if rsince_min is not None and "ret_since" in d.columns:
        d = d[pd.to_numeric(d["ret_since"], errors="coerce") >= float(rsince_min)]

    fee_max = spec.get("fee_max")
    if fee_max is not None and "fee_pct" in d.columns:
        d = d[pd.to_numeric(d["fee_pct"], errors="coerce").fillna(0.15) <= float(fee_max)]

    # 若是债券赛道且要求低回撤(或显式要求回撤小/稳健)，剔除高波动转债基并要求各阶段全正收益，避免转债基挤占初筛池
    is_low_dd_bond = (
        ftype == "mutual"
        and cat == "债券/固收"
        and (spec.get("low_dd_pref") or (spec.get("max_dd_max") is not None and float(spec["max_dd_max"]) <= 3.5))
    )
    if is_low_dd_bond:
        d_steady = d[
            (~d["name"].astype(str).str.contains("转债|可转债", na=False))
            & (pd.to_numeric(d["ret_1m"], errors="coerce").fillna(0) >= -0.05)
            & (pd.to_numeric(d["ret_3m"], errors="coerce").fillna(0) >= 0.0)
            & (pd.to_numeric(d["ret_6m"], errors="coerce").fillna(0) >= 0.0)
            & (pd.to_numeric(d["ret_1y"], errors="coerce").fillna(0) >= float(r1y_min or 2.2))
            & (pd.to_numeric(d["ret_1y"], errors="coerce").fillna(99) <= float(r1y_max or 8.5))
        ]
        if len(d_steady) >= 15:
            d = d_steady

    if d.empty:
        return {
            "ok": False,
            "error": f"初筛未找到满足收益率条件 ({spec.get('explain')}) 的基金，建议适当放宽收益率区间。",
            "spec": spec,
        }

    # 为避免同一只基金的 A/C/E 份额占满候选名额，计算多周期综合稳健得分并对同名基金份额去重
    if ftype == "mutual":
        d["_r1y"] = pd.to_numeric(d["ret_1y"], errors="coerce").fillna(0.0)
        d["_r2y"] = pd.to_numeric(d["ret_2y"], errors="coerce").fillna(0.0)
        d["_r3y"] = pd.to_numeric(d["ret_3y"], errors="coerce").fillna(0.0)
        d["_rsince"] = pd.to_numeric(d["ret_since"], errors="coerce").fillna(0.0)

        if r1y_min is not None and r1y_max is not None:
            target_mid = 0.5 * (float(r1y_min) + float(r1y_max))
            d["_score"] = -((d["_r1y"] - target_mid).abs()) + np.where(d["_r3y"] >= float(r1y_min) * 2.5, 0.35, 0.0)
        elif is_low_dd_bond or spec.get("long_history") or spec.get("top_rank"):
            is_pure = d["name"].astype(str).str.contains(
                "纯债|国企债|金融债|政金债|国开|农发|信用|利率|短债|中债|高等级", na=False
            )
            d["_score"] = (
                d["_r3y"] * 1.2
                + d["_r2y"] * 1.0
                + d["_r1y"] * 1.5
                + d["_rsince"].clip(upper=100.0) * 0.25
                + is_pure.astype(float) * (8.0 if is_low_dd_bond else 2.0)
            )
        else:
            d["_score"] = d["_r1y"] * 1.0 + d["_r3y"] * 0.25

        # A类主份额微幅加权(通常规模更大、历史更完整)，并对同名基金去重保留最优份额进入候选核验池
        is_a_share = d["name"].astype(str).str.endswith("A")
        d["_sort_score"] = d["_score"] + is_a_share.astype(float) * 0.15
        d["_base_name"] = d["name"].astype(str).map(_base_fund_name)
        d_dedup = d.sort_values("_sort_score", ascending=False).drop_duplicates(subset=["_base_name"], keep="first").reset_index(drop=True)

        # 检查是否需要核验 规模 / 机构占比 / 最大回撤 / 成立年限 / 夏普 / 基金经理年限
        need_deep = any(
            spec.get(k) is not None
            for k in ("max_dd_max", "max_dd_all_max", "scale_min", "scale_max", "inst_ratio_max", "establish_years_min", "sharpe_1y_min", "manager_years_min")
        )
        missing_codes = [
            str(r.code).zfill(6)
            for r in d_dedup.head(160).itertuples(index=False)
            if pd.isna(getattr(r, "scale_yi", None))
            or pd.isna(getattr(r, "inst_ratio", None))
            or pd.isna(getattr(r, "max_dd_1y", None))
            or pd.isna(getattr(r, "establish_years", None))
            or pd.isna(getattr(r, "sharpe_1y", None))
        ]
        if missing_codes and (need_deep or len(missing_codes) <= 60):
            to_pull = missing_codes[:125]
            if progress_cb:
                progress_cb(f"正在并发核验 {len(to_pull)} 只候选基金的【真实最大回撤 / 基金规模 / 机构占比 / 夏普 / 经理年限】...")
            fetcher.fetch_fund_deep_metrics_batch(
                to_pull,
                progress_cb=lambda done, tot, c: progress_cb(f"正在核验深度指标 {done}/{tot} ({c})...") if progress_cb else None,
            )
            fresh_all = db.load_mutual_fund_rank()
            d_codes_set = set(d["code"].astype(str).str.zfill(6))
            cand_base = fresh_all[fresh_all["code"].astype(str).str.zfill(6).isin(d_codes_set)].copy()
        else:
            cand_base = d.copy()

        def _apply_deep_filter(src_df: pd.DataFrame, relax: float = 0.0):
            cur = src_df.copy()
            f_steps = [f"全库{len(df)}只", f"初筛{len(d_dedup)}只"]
            dd_max = spec.get("max_dd_max")
            dd_all_max = spec.get("max_dd_all_max")
            sc_min = spec.get("scale_min")
            sc_max = spec.get("scale_max")
            inst_max = spec.get("inst_ratio_max")
            est_min = spec.get("establish_years_min")
            shp_min = spec.get("sharpe_1y_min")
            mgr_min = spec.get("manager_years_min")

            if dd_max is not None:
                lim = float(dd_max) * (1.0 + 0.35 * relax)
                cur = cur[pd.to_numeric(cur["max_dd_1y"], errors="coerce") <= lim]
                f_steps.append(f"回撤达标{len(cur)}只")
            if dd_all_max is not None:
                lim_all = float(dd_all_max) * (1.0 + 0.30 * relax)
                cur_all = cur[pd.to_numeric(cur["max_dd_all"], errors="coerce") <= lim_all]
                if len(cur_all) >= 3:
                    cur = cur_all
            if sc_min is not None:
                lim_sc = float(sc_min) * (1.0 - 0.35 * relax)
                cur = cur[pd.to_numeric(cur["scale_yi"], errors="coerce") >= lim_sc]
                f_steps.append(f"规模达标{len(cur)}只")
            if sc_max is not None:
                lim_sc_hi = float(sc_max) * (1.0 + 0.35 * relax)
                cur = cur[pd.to_numeric(cur["scale_yi"], errors="coerce") <= lim_sc_hi]
            if inst_max is not None:
                lim_inst = min(100.0, float(inst_max) + 10.0 * relax)
                cur = cur[pd.to_numeric(cur["inst_ratio"], errors="coerce") <= lim_inst]
                f_steps.append(f"机构占比达标{len(cur)}只")
            if est_min is not None and "establish_years" in cur.columns:
                lim_est = float(est_min) * (1.0 - 0.25 * relax)
                # 兼容尚未拉取 establish_years 但近3年收益有值的老基
                has_est = (pd.to_numeric(cur["establish_years"], errors="coerce") >= lim_est) | (
                    cur["establish_years"].isna() & cur["ret_3y"].notna() & (lim_est <= 5.0)
                )
                cur = cur[has_est]
            if shp_min is not None and "sharpe_1y" in cur.columns:
                lim_shp = float(shp_min) * (1.0 - 0.25 * relax)
                cur = cur[pd.to_numeric(cur["sharpe_1y"], errors="coerce") >= lim_shp]
            if mgr_min is not None and "manager_years" in cur.columns:
                lim_mgr = float(mgr_min) * (1.0 - 0.25 * relax)
                cur = cur[pd.to_numeric(cur["manager_years"], errors="coerce") >= lim_mgr]
            return cur, " → ".join(f_steps)

        cand_df, funnel_str = _apply_deep_filter(cand_base, relax=0.0)
        relaxed_note = ""
        if len(cand_df) < 2 and len(cand_base) >= 3:
            cand_relax, _ = _apply_deep_filter(cand_base, relax=1.0)
            if len(cand_relax) > len(cand_df):
                cand_df = cand_relax
                relaxed_note = "（严选命中较少，已自动为您弹性放宽约25%边际阈值补充高匹配候补标的）"

        # 结合多周期收益、夏普比率与最大回撤计算综合评分排序，并对同基金多份额去重
        r1_s = pd.to_numeric(cand_df["ret_1y"], errors="coerce").fillna(0.0)
        r3_s = pd.to_numeric(cand_df["ret_3y"], errors="coerce").fillna(0.0)
        rs_s = pd.to_numeric(cand_df["ret_since"], errors="coerce").fillna(0.0).clip(upper=100.0)
        dd_s = pd.to_numeric(cand_df["max_dd_1y"], errors="coerce").fillna(5.0).clip(lower=0.35)
        dd_all_s = pd.to_numeric(cand_df["max_dd_all"], errors="coerce").fillna(8.0).clip(lower=1.20)
        shp_s = pd.to_numeric(cand_df.get("sharpe_1y"), errors="coerce").fillna(1.2)
        if spec.get("long_history") or spec.get("top_rank") or cat == "债券/固收":
            cand_df["_calmar"] = (r1_s * 1.2 + r3_s * 0.45 + rs_s * 0.08 + shp_s * 1.0) / (dd_s * 0.65 + dd_all_s * 0.25)
        else:
            cand_df["_calmar"] = (r1_s + shp_s * 0.5) / dd_s
        cand_df["_base_name"] = cand_df["name"].astype(str).map(_base_fund_name)
        prefer_c = (spec.get("fee_max") is not None and float(spec["fee_max"]) == 0.0)
        cand_df["_share_pref"] = cand_df["name"].astype(str).str.endswith("C" if prefer_c else "A").astype(int)
        cand_df = (
            cand_df.sort_values(["_share_pref", "ret_1y", "scale_yi"], ascending=[False, False, False])
            .drop_duplicates(subset=["_base_name"], keep="first")
            .sort_values(["_calmar", "ret_1y"], ascending=[False, False])
            .reset_index(drop=True)
        )
        final_df = cand_df.head(int(spec.get("top_n") or 100))
        spec["funnel"] = funnel_str + f" → 优选去重{len(final_df)}只" + relaxed_note
    else:
        sc_min = spec.get("scale_min")
        sc_max = spec.get("scale_max")
        if sc_min is not None:
            d = d[pd.to_numeric(d["scale_yi"], errors="coerce") >= float(sc_min)]
        if sc_max is not None:
            d = d[pd.to_numeric(d["scale_yi"], errors="coerce") <= float(sc_max)]
        final_df = d.head(int(spec.get("top_n") or 100)).reset_index(drop=True)
        spec["funnel"] = f"ETF全库{len(df)}只 → 筛选命中{len(final_df)}只"

    if final_df.empty:
        return {
            "ok": False,
            "error": f"深度核验后暂无同时满足 [{spec.get('explain')}] 的基金，建议适当放宽回撤或规模范围。",
            "spec": spec,
        }

    if on_rows_ready is not None:
        try:
            on_rows_ready(ftype, spec, final_df)
        except Exception:
            pass

    # 第 3 阶段：若本地配置了 AI 模型，生成专业组合解读
    ai_summary = ""
    if is_configured():
        if progress_cb:
            progress_cb(f"已筛出 {len(final_df)} 只完全达标基金，正在生成 AI 优选点评...")
        fact_lines = []
        for idx, r in enumerate(final_df.head(12).itertuples(index=False)):
            if ftype == "mutual":
                est_y = getattr(r, "establish_years", None)
                shp = getattr(r, "sharpe_1y", None)
                mgr = getattr(r, "manager_name", None)
                mgr_y = getattr(r, "manager_years", None)
                mgr_s = f"{mgr}({mgr_y}年)" if (mgr and not pd.isna(mgr)) else "-"
                fact_lines.append(
                    f"{idx+1}. {r.code} {r.name} [{r.category}] | "
                    f"成立:{est_y if (est_y and not pd.isna(est_y)) else '-'}年 | "
                    f"近1年收益:{r.ret_1y}% 近3年:{r.ret_3y}% 成立来:{getattr(r, 'ret_since', '-')}% | "
                    f"近1年回撤:{getattr(r, 'max_dd_1y', '-')}% (成立来回撤:{getattr(r, 'max_dd_all', '-')}%) | "
                    f"夏普:{shp if (shp and not pd.isna(shp)) else '-'} | "
                    f"规模:{getattr(r, 'scale_yi', '-')}亿 | 机构占比:{getattr(r, 'inst_ratio', '-')}% | "
                    f"基金经理:{mgr_s} | 费率:{r.fee_pct}%"
                )
            else:
                fact_lines.append(
                    f"{idx+1}. {r.code} {r.name} [{r.category}] | "
                    f"近1年:{r.ret_250d}% 近60日:{r.ret_60d}% | 规模:{r.scale_yi}亿 | 溢价率:{r.discount_rate}%"
                )
        user_msg = (
            f"用户选基需求：{query}\n"
            f"量化解析条件：{spec.get('explain')}\n"
            f"以下为严格达标的前 {len(fact_lines)} 只基金实测数据：\n"
            + "\n".join(fact_lines)
        )
        try:
            ai_summary = chat(
                [
                    {"role": "system", "content": FUND_REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=450,
                retries=1,
            )
        except AIError as e:
            ai_summary = f"(AI解读暂不可用: {e})"
    else:
        ai_summary = f"已按条件【{spec.get('explain')}】为您精准筛出 {len(final_df)} 只达标基金（配置 config.json 后可附加 AI 深度对比点评）。"

    return {
        "ok": True,
        "fund_type": ftype,
        "spec": spec,
        "df": final_df,
        "ai_summary": ai_summary,
    }


# ---------------------------------------------------------------------------
# 二、AI 自然语言选股 (结合全市场 5000+ 股票作战台因子 + 5年PE/PB历史分位 + 连续增长季度数 + 资金流 + 排雷)
# ---------------------------------------------------------------------------

STOCK_NL_SYSTEM_PROMPT = (
    "你是一名专业的 A 股多因子量化选股意图翻译器。用户会用自然语言（包括模糊定性描述，如“市盈率历史百分位30%以下，业绩持续增长的大蓝筹股”）描述选股条件，"
    "请将其翻译为严格的 JSON 过滤参数（不要输出任何多余解释或 markdown 代码块标记）。\n\n"
    "支持的 JSON 字段如下（未提及的条件请设为 null）：\n"
    "- industry_kw: 行业或细分题材关键词(如 \"半导体\"、\"银行\"、\"医药\"、\"有色\"，无则 \"\")\n"
    "- setup_kw: 形态或催化关键词(如 \"主升\"、\"反转\"、\"超跌\"、\"预增\"、\"资金抢筹\"，无则 \"\")\n"
    "- p_tb_min: 实战胜率下限(0~1之间的小数，如“胜率55%以上”填 0.55)\n"
    "- payoff_min: 盈亏比下限(如“盈亏比1.3以上”填 1.3)\n"
    "- pos250_max: 年内股价位置上限(0~1之间的小数，如“年内低位/位置40%以内”填 0.40)\n"
    "- pos250_min: 年内股价位置下限(0~1之间的小数)\n"
    "- price_max: 股价上限(元)\n"
    "- price_min: 股价下限(元)\n"
    "- pe_max: 市盈率 PE(TTM) 绝对值上限(如“市盈率25以内”填 25.0)\n"
    "- pe_hist_pct_max: 5年市盈率(PE)历史百分位上限(0~100，如“市盈率历史百分位30%以下/估值历史低位”填 30.0)\n"
    "- pb_max: 市净率 PB 绝对值上限\n"
    "- pb_hist_pct_max: 5年市净率(PB)历史百分位上限(0~100)\n"
    "- roe_min: ROE(%) 下限(如“ROE大于10%”或“大蓝筹/白马”填 10.0)\n"
    "- growth_streak_min: 归母净利润同比连续正增长的季度数下限(如“业绩持续增长/连续增长”填 3)\n"
    "- profit_yoy_min: 最新季报归母净利润同比增速(%)下限(如“业绩增长20%以上”填 20.0；“持续增长”填 5.0)\n"
    "- avg_profit_yoy_min: 近3期平均净利同比增速(%)下限(如“业绩持续增长”填 5.0)\n"
    "- rev_yoy_min: 最新季报营收同比增速(%)下限(如“持续增长”填 0.0)\n"
    "- mv_min: 总市值下限(亿元，如“大蓝筹/大盘股/白马龙头”填 500.0；“千亿巨头”填 1000.0)\n"
    "- mv_max: 总市值上限(亿元，如“小盘股”填 200.0)\n"
    "- div_yield_min: 当前股息率(TTM %)下限(如“股息率高于3%”填 3.0；“高股息”填 4.0)\n"
    "- div_years_min: 过去5年连续现金分红年数下限(0~5，如“持续稳定分红/连续分红”填 4；“连续5年分红”填 5)\n"
    "- div_min_3y_min: 近3年每年最低股息率(%)下限(用于核验分红稳定性，如“持续稳定分红”填 2.0)\n"
    "- stable_dividend: 是否为持续稳定分红/红利偏好(true/false)\n"
    "- blue_chip: 是否为大蓝筹/白马龙头偏好(true/false)\n"
    "- exclude_veto: 是否剔除解禁/业绩预亏排雷票(默认 true)\n"
    "- top_n: 返回只数(默认 20)\n"
    "- explain: 一句话说明解析出的选股逻辑\n\n"
    "必须只输出一行合法 JSON。"
)

STOCK_REVIEW_SYSTEM_PROMPT = (
    "你是一名实战型 A 股量化基金经理。量化引擎已根据用户的自然语言条件，从全市场 5000+ 只股票中"
    "结合 5年PE/PB历史分位数、24个季度连续业绩增长序列、5年连续现金分红与股息率、ROE、总市值与多因子实战胜率筛出以下达标个股。\n"
    "请用简洁、客观、专业的语言（控制在 220 字以内）输出：\n"
    "【筛选画像】1-2句话概括本批入选股票的分红/估值/成长/ROE 与市值特征；\n"
    "【精选看点】挑出 2-3 只综合指标最突出且实战胜率优秀的代表标的，点明其代码、名称与核心数据；\n"
    "【纪律提示】1句话提示按作战台挂单上限与 ATR 止损位执行。严禁编造数据。"
)


def parse_stock_query_fallback(query: str) -> dict:
    """
    本地高精度选股语义+正则解析器（支持21项定量与模糊定性语义：
      5年PE/PB历史分位、同业PE分位、PEG、连续3年ROE、毛利率、资产负债率、
      连续增长季度数、业绩加速、5年连续现金分红与股息率、大蓝筹/中小盘、
      主力净流入、均线多头排列、突破60日新高、实战胜率、盈亏比）。
    """
    q = _normalize_cn_numbers(query or "")
    spec = {
        "industry_kw": "",
        "setup_kw": "",
        "p_tb_min": None,
        "payoff_min": None,
        "pos250_max": None,
        "pos250_min": None,
        "price_max": None,
        "price_min": None,
        "pe_max": None,
        "pe_hist_pct_max": None,
        "pe_ind_pct_max": None,
        "pb_max": None,
        "pb_hist_pct_max": None,
        "peg_max": None,
        "roe_min": None,
        "roe_3y_min": None,
        "gross_margin_min": None,
        "debt_ratio_max": None,
        "growth_streak_min": None,
        "rev_growth_streak_min": None,
        "profit_yoy_min": None,
        "avg_profit_yoy_min": None,
        "rev_yoy_min": None,
        "profit_accel": False,
        "div_yield_min": None,
        "div_years_min": None,
        "div_min_3y_min": None,
        "stable_dividend": False,
        "inflow_min": None,
        "ma_bull": False,
        "break_high60": False,
        "mv_min": None,
        "mv_max": None,
        "blue_chip": False,
        "require_increase_or_buyback": False,
        "exclude_decrease": False,
        "exclude_veto": True,
        "top_n": 20,
        "explain": "",
    }

    for ind in ("半导体", "芯片", "人工智能", "医药", "医疗", "银行", "煤炭", "电力", "军工", "消费", "白酒", "新能源", "汽车", "化工", "机械", "有色", "黄金", "软件", "通信", "证券", "保险", "家电", "航运", "港口"):
        if ind in q:
            spec["industry_kw"] = ind
            break

    for st in ("主升", "突破", "反转", "超跌", "预增", "抢筹", "吸筹", "回踩"):
        if st in q:
            spec["setup_kw"] = st
            break

    m_win = re.search(r"胜率[^0-9，,、；;。]{0,5}([0-9.]+)\s*%", q)
    if m_win:
        spec["p_tb_min"] = round(float(m_win.group(1)) / 100.0, 3)

    m_pay = re.search(r"盈亏比[^0-9，,、；;。]{0,5}([0-9.]+)", q)
    if m_pay:
        spec["payoff_min"] = float(m_pay.group(1))

    # 1) 识别 PE / PB 历史百分位、同业分位、PEG 与绝对值
    m_pe_ind = re.search(r"(?:行业|同业).*?(?:市盈率|pe|PE|估值).*?(?:分位|前)[^0-9，,、；;。]{0,5}([0-9.]+)\s*%", q)
    if m_pe_ind:
        spec["pe_ind_pct_max"] = float(m_pe_ind.group(1))
    elif any(w in q for w in ("同业低估", "比同行便宜", "行业内低估", "低于行业平均")):
        spec["pe_ind_pct_max"] = 35.0

    m_pe_pct = re.search(r"(?:市盈率|(?<![A-Za-z])(?:pe|PE)(?![GgA-Za-z])|估值)[^0-9，,、；;。]{0,8}(?:历史)?(?:百分位|分位[数]?)[^0-9，,、；;。]{0,5}([0-9.]+)\s*%", q)
    if m_pe_pct:
        spec["pe_hist_pct_max"] = float(m_pe_pct.group(1))
    elif any(w in q for w in ("历史低估值", "估值历史低位", "估值底", "PE历史低位")):
        spec["pe_hist_pct_max"] = 30.0
    else:
        m_pe = re.search(r"(?:市盈率|(?<![A-Za-z])(?:pe|PE)(?![GgA-Za-z]))[^0-9，,、；;。%]{0,6}([0-9.]+)(?!\s*%)", q)
        if m_pe:
            spec["pe_max"] = float(m_pe.group(1))
        elif "低估值" in q or "估值低" in q:
            spec["pe_max"] = 20.0
            spec["pe_hist_pct_max"] = 35.0

    m_pb_pct = re.search(r"(?:市净率|pb|PB)[^0-9，,、；;。]{0,8}(?:历史)?(?:百分位|分位[数]?)[^0-9，,、；;。]{0,5}([0-9.]+)\s*%", q)
    if m_pb_pct:
        spec["pb_hist_pct_max"] = float(m_pb_pct.group(1))
    else:
        m_pb = re.search(r"(?:市净率|pb|PB)[^0-9，,、；;。%]{0,6}([0-9.]+)(?!\s*%)", q)
        if m_pb:
            spec["pb_max"] = float(m_pb.group(1))
        elif "破净" in q:
            spec["pb_max"] = 1.0

    m_peg = re.search(r"(?:peg|PEG)[^0-9，,、；;。]{0,5}([0-9.]+)", q)
    if m_peg:
        spec["peg_max"] = float(m_peg.group(1))
    elif any(w in q for w in ("GARP", "成长低估", "高成长低估值")):
        spec["peg_max"] = 1.0
        if spec["growth_streak_min"] is None:
            spec["growth_streak_min"] = 2

    # 2) 识别年内股价位置与技术趋势 (均线多头/突破新高/主力净流入)
    if "股价低位" in q or "年内低位" in q or "底部" in q or "超跌" in q:
        spec["pos250_max"] = 0.40
    m_pos = re.search(r"位置[^0-9，,、；;。]{0,5}([0-9.]+)\s*%\s*(?:以内|以下)", q)
    if m_pos:
        spec["pos250_max"] = round(float(m_pos.group(1)) / 100.0, 3)

    if any(w in q for w in ("均线多头", "多头排列", "上升趋势", "右侧多头")):
        spec["ma_bull"] = True
    if any(w in q for w in ("突破新高", "60日新高", "创新高", "平台突破")):
        spec["break_high60"] = True
    if any(w in q for w in ("主力净流入", "资金净流入", "主力流入", "大单净买入", "主力抢筹")):
        spec["inflow_min"] = 0.01
    if any(w in q for w in ("增持", "回购", "大股东买入", "高管买入")):
        spec["require_increase_or_buyback"] = True
    if any(w in q for w in ("无减持", "不减持", "剔除减持", "排除减持", "没有减持")):
        spec["exclude_decrease"] = True

    # 3) 识别 ROE、连续3年ROE、毛利率、资产负债率
    m_roe3 = re.search(r"连续\s*[3三]\s*年.*?(?:净资产收益率|roe|ROE)[^0-9，,、；;。]{0,5}([0-9.]+)", q)
    if m_roe3:
        spec["roe_3y_min"] = float(m_roe3.group(1))
    else:
        m_roe = re.search(r"(?:净资产收益率|roe|ROE)[^0-9，,、；;。]{0,6}([0-9.]+)", q)
        if m_roe:
            spec["roe_min"] = float(m_roe.group(1))
        if any(w in q for w in ("长期高ROE", "连续高ROE", "巴菲特")):
            spec["roe_3y_min"] = float(spec["roe_min"] or 15.0)

    m_gm = re.search(r"毛利[率]?[^0-9，,、；;。]{0,5}([0-9.]+)\s*%", q)
    if m_gm:
        spec["gross_margin_min"] = float(m_gm.group(1))
    elif any(w in q for w in ("高毛利", "护城河")):
        spec["gross_margin_min"] = 35.0

    m_debt = re.search(r"负债[率]?[^0-9，,、；;。]{0,6}([0-9.]+)\s*%", q)
    if m_debt:
        spec["debt_ratio_max"] = float(m_debt.group(1))
    elif any(w in q for w in ("低负债", "财务稳健", "轻负债")):
        spec["debt_ratio_max"] = 50.0

    # 4) 识别模糊语义：“业绩持续增长” / “连续增长” / “业绩加速” / “营收净利双增”
    if any(w in q for w in ("业绩加速", "加速增长", "增速加快")):
        spec["profit_accel"] = True
        if spec["growth_streak_min"] is None:
            spec["growth_streak_min"] = 2

    if any(w in q for w in ("双增", "营收净利双增", "营收利润双增")):
        spec["rev_growth_streak_min"] = 2
        if spec["growth_streak_min"] is None:
            spec["growth_streak_min"] = 2

    m_streak = re.search(r"连续\s*([0-9]+)\s*(?:个)?(?:季度|期).*?增长", q)
    if m_streak:
        spec["growth_streak_min"] = int(m_streak.group(1))
    elif any(w in q for w in ("持续增长", "连续增长", "稳步增长", "业绩长虹", "持续高增", "业绩增长")):
        spec["growth_streak_min"] = 3
        if spec["profit_yoy_min"] is None:
            spec["profit_yoy_min"] = 5.0
        if spec["avg_profit_yoy_min"] is None:
            spec["avg_profit_yoy_min"] = 5.0
        if spec["rev_yoy_min"] is None:
            spec["rev_yoy_min"] = 0.0

    m_pyoy = re.search(r"(?:净利[润]?|业绩)(?:增速|增长|同比)[^0-9，,、；;。]{0,6}([0-9.]+)\s*%", q)
    if m_pyoy:
        spec["profit_yoy_min"] = float(m_pyoy.group(1))

    # 5) 识别分红与股息率语义：“持续稳定分红”、“连续5年分红”、“股息率高于3%”、“高股息”、“红利”
    m_dy = re.search(r"股息[率]?[^0-9，,、；;。]{0,6}([0-9.]+)\s*%", q)
    if m_dy:
        spec["div_yield_min"] = float(m_dy.group(1))
    elif any(w in q for w in ("高股息", "高分红", "红利")):
        spec["div_yield_min"] = 3.5

    m_div_yrs = re.search(r"连续\s*([0-9]+)\s*年.*?分红", q)
    if m_div_yrs:
        spec["div_years_min"] = min(5, int(m_div_yrs.group(1)))
        spec["stable_dividend"] = True
    elif any(w in q for w in ("持续稳定分红", "稳定分红", "持续分红", "连续分红", "常年分红", "红利", "高股息")):
        spec["div_years_min"] = 4
        spec["stable_dividend"] = True

    if spec["stable_dividend"]:
        if spec["div_yield_min"] is None:
            spec["div_yield_min"] = 3.0
        spec["div_min_3y_min"] = round(min(2.5, float(spec["div_yield_min"]) * 0.65), 2)
        if spec["roe_min"] is None and spec["roe_3y_min"] is None:
            spec["roe_min"] = 6.0

    # 6) 识别模糊语义：“大蓝筹股” / “白马股” / “大盘股” / “行业龙头” vs 具体市值
    m_mv_range = re.search(r"市值[^0-9，,、；;。]{0,5}([0-9.]+)\s*(?:到|至|-|~|—)\s*([0-9.]+)\s*亿", q)
    if m_mv_range:
        spec["mv_min"] = float(m_mv_range.group(1))
        spec["mv_max"] = float(m_mv_range.group(2))
    else:
        m_mv_max = re.search(r"市值[^0-9，,、；;。]{0,5}([0-9.]+)\s*亿\s*(?:以内|以下)", q)
        if m_mv_max:
            spec["mv_max"] = float(m_mv_max.group(1))
        m_mv_min = re.search(r"市值[^0-9，,、；;。]{0,5}([0-9.]+)\s*亿\s*(?:以上|大于)", q)
        if m_mv_min:
            spec["mv_min"] = float(m_mv_min.group(1))

    if any(w in q for w in ("大蓝筹", "蓝筹", "白马", "大盘", "核心资产", "行业龙头", "巨头", "权重股")):
        spec["blue_chip"] = True
        if spec["mv_min"] is None:
            spec["mv_min"] = 1000.0 if "千亿" in q else 500.0
        if spec["roe_min"] is None and spec["roe_3y_min"] is None:
            spec["roe_min"] = 10.0
        if spec["pe_max"] is None:
            spec["pe_max"] = 35.0
    elif any(w in q for w in ("小盘", "微盘")):
        if spec["mv_max"] is None:
            spec["mv_max"] = 150.0

    parts = []
    if spec["industry_kw"]:
        parts.append(f"行业/题材含'{spec['industry_kw']}'")
    if spec["blue_chip"]:
        parts.append("大蓝筹/白马龙头")
    if spec["stable_dividend"] or spec["div_years_min"] is not None:
        parts.append(f"连续分红>={spec['div_years_min'] or 4}年(近3年最低股息>={spec['div_min_3y_min'] or 2.0}%)")
    if spec["div_yield_min"] is not None:
        parts.append(f"股息率(TTM)>={spec['div_yield_min']}%")
    if spec["pe_hist_pct_max"] is not None:
        parts.append(f"5年PE分位<={spec['pe_hist_pct_max']:.0f}%")
    if spec["pe_ind_pct_max"] is not None:
        parts.append(f"同业PE分位<={spec['pe_ind_pct_max']:.0f}%")
    if spec["pb_hist_pct_max"] is not None:
        parts.append(f"5年PB分位<={spec['pb_hist_pct_max']:.0f}%")
    if spec["pe_max"] is not None:
        parts.append(f"PE(TTM)<={spec['pe_max']:.0f}")
    if spec["pb_max"] is not None:
        parts.append(f"PB<={spec['pb_max']:.1f}")
    if spec["peg_max"] is not None:
        parts.append(f"PEG<={spec['peg_max']}")
    if spec["roe_3y_min"] is not None:
        parts.append(f"连续3年ROE>={spec['roe_3y_min']:.0f}%")
    elif spec["roe_min"] is not None:
        parts.append(f"ROE>={spec['roe_min']:.0f}%")
    if spec["gross_margin_min"] is not None:
        parts.append(f"毛利率>={spec['gross_margin_min']:.0f}%")
    if spec["debt_ratio_max"] is not None:
        parts.append(f"负债率<={spec['debt_ratio_max']:.0f}%")
    if spec["growth_streak_min"] is not None:
        parts.append(f"净利连增>={spec['growth_streak_min']}个季度")
    elif spec["profit_yoy_min"] is not None:
        parts.append(f"净利同比>={spec['profit_yoy_min']:.0f}%")
    if spec["profit_accel"]:
        parts.append("净利增速环比加速")
    if spec["ma_bull"]:
        parts.append("均线多头排列")
    if spec["break_high60"]:
        parts.append("突破60日新高")
    if spec["inflow_min"] is not None:
        parts.append("主力资金净流入")
    if spec["require_increase_or_buyback"]:
        parts.append("有股东增持/股份回购计划")
    if spec["exclude_decrease"]:
        parts.append("剔除股东减持")
    if spec["mv_min"] is not None or spec["mv_max"] is not None:
        parts.append(f"总市值=[{spec['mv_min'] or 0:.0f}~{spec['mv_max'] or '不限'}亿]")
    if spec["p_tb_min"] is not None:
        parts.append(f"实战胜率>={spec['p_tb_min']*100:.0f}%")
    if spec["pos250_max"] is not None:
        parts.append(f"年内位置<={spec['pos250_max']*100:.0f}%")
    if spec["setup_kw"]:
        parts.append(f"信号含'{spec['setup_kw']}'")
    spec["explain"] = " · ".join(parts) if parts else "按实战胜率与多因子共振优选"
    return spec


def parse_stock_query(query: str) -> dict:
    """优先使用本地高精度金融语义+正则解析器提取选股约束，复杂未知表达由本地 LLM 补充。"""
    rule_spec = parse_stock_query_fallback(query)
    has_rule = (
        bool(rule_spec.get("industry_kw"))
        or bool(rule_spec.get("setup_kw"))
        or bool(rule_spec.get("blue_chip"))
        or bool(rule_spec.get("stable_dividend"))
        or bool(rule_spec.get("ma_bull"))
        or bool(rule_spec.get("break_high60"))
        or bool(rule_spec.get("profit_accel"))
        or bool(rule_spec.get("require_increase_or_buyback"))
        or bool(rule_spec.get("exclude_decrease"))
        or any(
            rule_spec.get(k) is not None
            for k in (
                "p_tb_min", "payoff_min", "pos250_max", "pe_max", "pe_hist_pct_max", "pe_ind_pct_max",
                "pb_max", "pb_hist_pct_max", "peg_max", "roe_min", "roe_3y_min",
                "gross_margin_min", "debt_ratio_max", "growth_streak_min",
                "profit_yoy_min", "div_yield_min", "div_years_min", "inflow_min", "mv_min", "mv_max",
            )
        )
    )
    if has_rule or not is_configured():
        return rule_spec
    try:
        raw = chat(
            [
                {"role": "system", "content": STOCK_NL_SYSTEM_PROMPT},
                {"role": "user", "content": query.strip()},
            ],
            temperature=0.0,
            max_tokens=320,
            retries=1,
        )
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            llm_spec = json.loads(m.group(0))
            for k, v in rule_spec.items():
                if llm_spec.get(k) is None and v is not None:
                    llm_spec[k] = v
            if not llm_spec.get("explain"):
                llm_spec["explain"] = rule_spec["explain"]
            return llm_spec
    except Exception:
        pass
    return rule_spec


def run_ai_stock_screen(query: str, all_rows: list[dict], progress_cb=None, on_rows_ready=None) -> dict:
    """
    在今日作战台的全市场股票列表 (all_rows) 上执行本地 AI 自然语言选股：
      1) 解析自然语言条件 spec（支持 21 项定量与模糊语义指标）；
      2) 结合 fundamental 特征库与作战台多因子信号严格过滤，生成透明筛选漏斗，并在严选过少时自动弹性补齐；
      3) 先在 50ms 内秒出表格，再异步调用本地 AI 输出组合综述与重点个股看点。
    """
    if progress_cb:
        progress_cb("🤖 正在解析你的选股条件...")
    spec = parse_stock_query(query)
    if not all_rows:
        return {"ok": False, "error": "当前作战台暂无股票数据，请先点『重算模型』", "spec": spec}

    # 加载全量 21 项基本面衍生特征库与风险/增减持/回购日历
    funda_map = db.load_fundamental_map()
    rc_map = db.load_risk_calendar_map()

    def _run_filter(relax: float = 0.0):
        matched_list = []
        ind_kw = (spec.get("industry_kw") or "").strip()
        st_kw = (spec.get("setup_kw") or "").strip()
        p_tb_min = spec.get("p_tb_min")
        payoff_min = spec.get("payoff_min")
        pos_max = spec.get("pos250_max")
        pos_min = spec.get("pos250_min")
        pr_max = spec.get("price_max")
        pr_min = spec.get("price_min")
        pe_max = spec.get("pe_max")
        pe_pct_max = spec.get("pe_hist_pct_max")
        pe_ind_max = spec.get("pe_ind_pct_max")
        pb_max = spec.get("pb_max")
        pb_pct_max = spec.get("pb_hist_pct_max")
        peg_max = spec.get("peg_max")
        roe_min = spec.get("roe_min")
        roe3_min = spec.get("roe_3y_min")
        gm_min = spec.get("gross_margin_min")
        debt_max = spec.get("debt_ratio_max")
        streak_min = spec.get("growth_streak_min")
        r_streak_min = spec.get("rev_growth_streak_min")
        pyoy_min = spec.get("profit_yoy_min")
        avg_pyoy_min = spec.get("avg_profit_yoy_min")
        ryoy_min = spec.get("rev_yoy_min")
        req_accel = bool(spec.get("profit_accel", False))
        dy_min = spec.get("div_yield_min")
        div_yrs_min = spec.get("div_years_min")
        div_min3_min = spec.get("div_min_3y_min")
        is_stable_div = bool(spec.get("stable_dividend", False))
        inflow_min = spec.get("inflow_min")
        req_ma_bull = bool(spec.get("ma_bull", False))
        req_brk60 = bool(spec.get("break_high60", False))
        mv_min = spec.get("mv_min")
        mv_max = spec.get("mv_max")
        is_blue_chip = bool(spec.get("blue_chip", False))
        req_inc_bb = bool(spec.get("require_increase_or_buyback", False))
        excl_dec = bool(spec.get("exclude_decrease", False))
        excl_veto = bool(spec.get("exclude_veto", True))

        n_no_veto = 0
        n_val_pass = 0
        for r in all_rows:
            code = str(r.get("code") or "").zfill(6)
            evs = rc_map.get(code) or []
            has_inc_bb = any(e.get("event_type") in ("股东增持", "股份回购") for e in evs)
            has_dec = any(e.get("event_type") == "股东减持" for e in evs)
            has_veto_ev = any(e.get("severity") == "veto" for e in evs)
            if excl_veto and (r.get("veto") or has_veto_ev):
                continue
            if (excl_dec or req_inc_bb) and has_dec:
                continue
            if req_inc_bb and not has_inc_bb:
                continue
            if ind_kw:
                ind_str = f"{r.get('industry') or ''} {r.get('subsector') or ''} {r.get('name') or ''}"
                if ind_kw not in ind_str:
                    continue
            if st_kw:
                st_str = " ".join(r.get("setups") or [])
                if st_kw not in st_str:
                    continue
            n_no_veto += 1

            p_tb = r.get("p_tb", r.get("p_cond"))
            if p_tb_min is not None and (p_tb is None or float(p_tb) < float(p_tb_min) - 0.03 * relax):
                continue
            payoff = r.get("payoff")
            if payoff_min is not None and (payoff is None or float(payoff) < float(payoff_min) * (1.0 - 0.15 * relax)):
                continue
            pos250 = r.get("pos250")
            if pos_max is not None and (pos250 is None or float(pos250) > float(pos_max) + 0.12 * relax):
                continue
            if pos_min is not None and (pos250 is None or float(pos250) < float(pos_min) - 0.10 * relax):
                continue
            close = r.get("close")
            if pr_max is not None and (close is None or float(close) > float(pr_max)):
                continue
            if pr_min is not None and (close is None or float(close) < float(pr_min)):
                continue
            if inflow_min is not None:
                inf_v = r.get("net_inflow_yi")
                nb_v = r.get("net_big_ratio")
                if (inf_v is None or float(inf_v) <= 0) and (nb_v is None or float(nb_v) <= 0):
                    continue

            f_info = funda_map.get(code, {})
            pe = f_info.get("pe_ttm")
            if pe_max is not None:
                if pe is None or pd.isna(pe) or float(pe) <= 0 or float(pe) > float(pe_max) * (1.0 + 0.25 * relax):
                    continue
            if pe_pct_max is not None:
                pe_pct = f_info.get("pe_hist_pct")
                if pe_pct is None or pd.isna(pe_pct) or pe is None or float(pe) <= 0 or float(pe_pct) > float(pe_pct_max) + 12.0 * relax:
                    continue
            if pe_ind_max is not None:
                pe_ind = f_info.get("pe_ind_pct")
                if pe_ind is None or pd.isna(pe_ind) or pe is None or float(pe) <= 0 or float(pe_ind) > float(pe_ind_max) + 12.0 * relax:
                    continue
            pb = f_info.get("pb")
            if pb_max is not None:
                if pb is None or pd.isna(pb) or float(pb) <= 0 or float(pb) > float(pb_max) * (1.0 + 0.25 * relax):
                    continue
            if pb_pct_max is not None:
                pb_pct = f_info.get("pb_hist_pct")
                if pb_pct is None or pd.isna(pb_pct) or pb is None or float(pb) <= 0 or float(pb_pct) > float(pb_pct_max) + 12.0 * relax:
                    continue
            if peg_max is not None:
                peg = f_info.get("peg")
                if peg is None or pd.isna(peg) or float(peg) <= 0 or float(peg) > float(peg_max) * (1.0 + 0.25 * relax):
                    continue
            n_val_pass += 1

            roe = f_info.get("roe")
            if roe_min is not None:
                if roe is None or pd.isna(roe) or float(roe) < float(roe_min) * (1.0 - 0.20 * relax):
                    continue
            roe3 = f_info.get("roe_3y_min")
            if roe3_min is not None:
                if roe3 is None or pd.isna(roe3) or float(roe3) < float(roe3_min) * (1.0 - 0.20 * relax):
                    continue
            gm = f_info.get("gross_margin")
            if gm_min is not None:
                if gm is None or pd.isna(gm) or float(gm) < float(gm_min) * (1.0 - 0.20 * relax):
                    continue
            debt = f_info.get("debt_ratio")
            if debt_max is not None:
                if debt is None or pd.isna(debt) or float(debt) > float(debt_max) + 10.0 * relax:
                    continue
            dy = f_info.get("dividend_ratio")
            if dy_min is not None:
                if dy is None or pd.isna(dy) or float(dy) < float(dy_min) * (1.0 - 0.20 * relax):
                    continue
                if pe is None or pd.isna(pe) or float(pe) <= 0:
                    continue
            div_yrs = f_info.get("div_years_5y")
            if div_yrs_min is not None:
                req_y = max(2, int(div_yrs_min) - int(relax))
                if div_yrs is None or pd.isna(div_yrs) or int(div_yrs) < req_y:
                    continue
            div_min3 = f_info.get("div_min_3y")
            if div_min3_min is not None:
                if div_min3 is None or pd.isna(div_min3) or float(div_min3) < float(div_min3_min) * (1.0 - 0.30 * relax):
                    continue
            streak = f_info.get("growth_streak")
            if streak_min is not None:
                req_st = max(1, int(streak_min) - int(relax))
                if streak is None or pd.isna(streak) or int(streak) < req_st:
                    continue
            r_streak = f_info.get("rev_growth_streak")
            if r_streak_min is not None:
                req_rst = max(1, int(r_streak_min) - int(relax))
                if r_streak is None or pd.isna(r_streak) or int(r_streak) < req_rst:
                    continue
            pyoy = f_info.get("profit_yoy")
            if pyoy_min is not None:
                if pyoy is None or pd.isna(pyoy) or float(pyoy) < float(pyoy_min) * (1.0 - 0.30 * relax):
                    continue
            avg_pyoy = f_info.get("avg_profit_yoy_3q")
            if avg_pyoy_min is not None:
                if avg_pyoy is None or pd.isna(avg_pyoy) or float(avg_pyoy) < float(avg_pyoy_min) * (1.0 - 0.30 * relax):
                    continue
            ryoy = f_info.get("rev_yoy")
            if ryoy_min is not None and relax == 0.0:
                if ryoy is None or pd.isna(ryoy) or float(ryoy) < float(ryoy_min):
                    continue
            if req_accel and relax == 0.0:
                if not f_info.get("profit_accel"):
                    continue
            if req_ma_bull and relax == 0.0:
                if not f_info.get("ma_bull"):
                    continue
            if req_brk60 and relax == 0.0:
                if not f_info.get("break_high60"):
                    continue
            mv = f_info.get("total_mv")
            if mv_min is not None or mv_max is not None:
                if mv is None or pd.isna(mv):
                    continue
                mv_yi = float(mv) / 1e8 if float(mv) > 1e6 else float(mv)
                if mv_min is not None and mv_yi < float(mv_min) * (1.0 - 0.30 * relax):
                    continue
                if mv_max is not None and mv_yi > float(mv_max) * (1.0 + 0.30 * relax):
                    continue

            # 为入选股票附加直观的基本面核验标签，方便在作战台表格和详情栏一眼看到
            r_copy = dict(r)
            orig_setups = list(r_copy.get("setups") or [])
            extra_tags = []
            for e in evs:
                if e.get("event_type") in ("股东增持", "股份回购"):
                    s_tag = f"★{e.get('summary')}"
                    if s_tag not in orig_setups and s_tag not in extra_tags:
                        extra_tags.append(s_tag)
            if (dy_min is not None or is_stable_div) and dy is not None and not pd.isna(dy):
                d_y = int(div_yrs) if (div_yrs is not None and not pd.isna(div_yrs)) else 0
                d_avg = float(f_info.get("div_avg_5y") or dy)
                extra_tags.append(f"股息{float(dy):.1f}%(连分{d_y}年·均{d_avg:.1f}%)")
            if pe_pct_max is not None and f_info.get("pe_hist_pct") is not None:
                extra_tags.append(f"PE={float(pe):.1f}({float(f_info['pe_hist_pct']):.0f}%分位)")
            elif pe_ind_max is not None and f_info.get("pe_ind_pct") is not None:
                extra_tags.append(f"PE={float(pe):.1f}(同业{float(f_info['pe_ind_pct']):.0f}%分位)")
            elif peg_max is not None and f_info.get("peg") is not None:
                extra_tags.append(f"PEG={float(f_info['peg']):.2f}(PE={float(pe):.1f})")
            elif pe is not None and not pd.isna(pe):
                extra_tags.append(f"PE={float(pe):.1f}")
            if roe3_min is not None and roe3 is not None and not pd.isna(roe3):
                extra_tags.append(f"3年最低ROE={float(roe3):.1f}%")
            if streak is not None and not pd.isna(streak) and int(streak) >= 2:
                extra_tags.append(f"连增{int(streak)}期({float(pyoy or 0):+.0f}%)")
            if mv is not None and not pd.isna(mv) and (is_blue_chip or is_stable_div):
                mv_v = float(mv) / 1e8 if float(mv) > 1e6 else float(mv)
                extra_tags.append(f"{mv_v:.0f}亿")
            if extra_tags:
                r_copy["setups"] = extra_tags + orig_setups[:2]
            r_copy["_funda"] = f_info
            matched_list.append(r_copy)

        f_str = f"全市场{len(all_rows)}只 → 排雷/行业过审{n_no_veto}只 → 估值过审{n_val_pass}只 → 盈利/分红/成长达标{len(matched_list)}只"
        return matched_list, f_str

    matched, funnel_str = _run_filter(relax=0.0)
    relaxed_note = ""
    if len(matched) < 3:
        m_relax, _ = _run_filter(relax=1.0)
        if len(m_relax) > len(matched):
            matched = m_relax
            relaxed_note = "（严选命中较少，已自动为您弹性放宽约20%边际阈值补充高匹配候补股）"
    spec["funnel"] = funnel_str + relaxed_note

    is_stable_div = bool(spec.get("stable_dividend", False))
    dy_min = spec.get("div_yield_min")
    is_blue_chip = bool(spec.get("blue_chip", False))

    if is_stable_div or dy_min is not None:
        matched.sort(
            key=lambda x: (
                float((x.get("_funda") or {}).get("div_years_5y") or 0.0) * 3.0
                + min(10.0, float((x.get("_funda") or {}).get("div_avg_5y") or 0.0)) * 2.2
                + min(10.0, float((x.get("_funda") or {}).get("dividend_ratio") or 0.0)) * 1.8
                + min(25.0, float((x.get("_funda") or {}).get("roe") or 0.0)) * 0.5
                + float(x.get("p_tb") or x.get("p_cond") or 0.5) * 45.0
            ),
            reverse=True,
        )
    elif is_blue_chip:
        matched.sort(
            key=lambda x: (
                float((x.get("_funda") or {}).get("roe") or 0.0) * 0.8
                + min(12.0, float((x.get("_funda") or {}).get("growth_streak") or 0.0)) * 1.5
                + np.log10(max(10.0, float((x.get("_funda") or {}).get("total_mv") or 10.0))) * 8.0
                + float(x.get("p_tb") or x.get("p_cond") or 0.5) * 40.0
            ),
            reverse=True,
        )
    else:
        matched.sort(
            key=lambda x: (
                1 if (x.get("setups") and not x.get("veto")) else 0,
                float(x.get("p_tb") or x.get("p_cond") or 0.0),
                float(x.get("payoff") or 0.0),
            ),
            reverse=True,
        )
    top_rows = matched[: int(spec.get("top_n") or 25)]
    if not top_rows:
        return {
            "ok": False,
            "error": f"未找到同时满足 [{spec.get('explain')}] 的股票（{funnel_str}），建议适当放宽条件。",
            "spec": spec,
        }

    if on_rows_ready is not None:
        try:
            on_rows_ready(spec, top_rows, len(matched))
        except Exception:
            pass

    ai_summary = ""
    if is_configured():
        if progress_cb:
            progress_cb(f"已筛出 {len(matched)} 只达标股票，正在调用本地 AI 生成优选解读...")
        lines = []
        for i, r in enumerate(top_rows[:12]):
            c = str(r.get("code") or "").zfill(6)
            fi = r.get("_funda") or funda_map.get(c, {})
            pe_v = fi.get("pe_ttm")
            pe_s = f"{float(pe_v):.1f}" if pe_v and not pd.isna(pe_v) else "-"
            pe_pct_s = f"{float(fi['pe_hist_pct']):.1f}%" if fi.get("pe_hist_pct") is not None and not pd.isna(fi.get("pe_hist_pct")) else "-"
            roe_s = f"{float(fi['roe']):.1f}%" if fi.get("roe") and not pd.isna(fi.get("roe")) else "-"
            mv_s = f"{float(fi['total_mv']):.0f}亿" if fi.get("total_mv") and not pd.isna(fi.get("total_mv")) else "-"
            dy_s = f"{float(fi['dividend_ratio']):.2f}%" if fi.get("dividend_ratio") is not None and not pd.isna(fi.get("dividend_ratio")) else "-"
            dy_yrs = int(fi["div_years_5y"]) if fi.get("div_years_5y") is not None and not pd.isna(fi.get("div_years_5y")) else 0
            dy_avg = f"{float(fi['div_avg_5y']):.2f}%" if fi.get("div_avg_5y") is not None and not pd.isna(fi.get("div_avg_5y")) else "-"
            st_k = int(fi["growth_streak"]) if fi.get("growth_streak") is not None and not pd.isna(fi.get("growth_streak")) else 0
            py_s = f"{float(fi['profit_yoy']):+.1f}%" if fi.get("profit_yoy") is not None and not pd.isna(fi.get("profit_yoy")) else "-"
            lines.append(
                f"{i+1}. {c} {r.get('name')} [{r.get('subsector') or r.get('industry') or '-'}] | "
                f"总市值:{mv_s} | 股息率:{dy_s}(连续分红:{dy_yrs}年·5年均:{dy_avg}) | PE:{pe_s}(5年分位:{pe_pct_s}) | ROE:{roe_s} | "
                f"净利连增:{st_k}个季度(最新同比:{py_s}) | 实战胜率:{(r.get('p_tb') or 0)*100:.1f}%"
            )
        try:
            ai_summary = chat(
                [
                    {"role": "system", "content": STOCK_REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": f"用户选股需求：{query}\n解析条件：{spec.get('explain')}\n筛选漏斗：{spec.get('funnel', '')}\n入选股票：\n" + "\n".join(lines)},
                ],
                temperature=0.2,
                max_tokens=420,
                retries=1,
            )
        except AIError as e:
            ai_summary = f"(AI解读暂不可用: {e})"
    else:
        ai_summary = f"已按条件【{spec.get('explain')}】筛出 {len(matched)} 只股票。"

    return {
        "ok": True,
        "spec": spec,
        "rows": top_rows,
        "total_matched": len(matched),
        "ai_summary": ai_summary,
    }

