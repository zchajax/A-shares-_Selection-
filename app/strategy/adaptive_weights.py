"""
策略层 - L3 市场自适应权重(让数据反推权重,替代拍脑袋)
====================================================
L1 的权重(动量25/趋势25/量能15/低估20/质量15)是人凭经验拍的。
L2 用 IC 检验证明:这三个技术因子在近年 A 股【方向可能是反的】。
L3 就是把 L2 的 IC 结论【反过来喂给权重】——用数据定权重,而非拍脑袋。

核心思想(IC_IR 加权,量化业界标准做法):
  · 一个因子该给多大权重,取决于它【预测得准不准】+【稳不稳】。
  · IC 均值 → 预测强度与方向(正=顺势有效,负=反向有效)。
  · IC_IR = IC均值/IC标准差 → 稳定性(每次都朝一个方向使劲才可靠)。
  · 故权重 ∝ IC_IR:既奖励有效因子,也惩罚忽正忽负的噪声因子。
  · 【保留符号】:负 IC 因子给【负权重】= 反向使用(买该因子低的档),
    这正是 L2 "买超跌弱档反而赚"结论的直接落地。

诚实边界(必须讲清,不可粉饰):
  · 只对有真实历史序列的技术因子(动量/趋势/量能)反推权重。
    基本面因子(低估/质量)本地无历史 IC,不参与自适应,沿用 L1 默认或手动。
  · IC 来自单一时间窗(约2021→2026,整体偏震荡/结构市)。牛市里动量常转正,
    所以"负权重"是这段行情的产物,不是永恒真理。换窗口结论可能变——
    这就是为什么 L3 要配 backtest 对比验证,且默认不自动覆盖用户权重。
  · 只筛显著因子(|t|>阈值)入选,避免把噪声当信号放大。
红线:全程本地量化、可复现,IC 严格 shift(-N) 无前视,不编造。
主入口: derive_weights(fwd_days, ...) -> dict{weights, detail, note}
"""
import numpy as np
import pandas as pd

from . import panel as pnl
from . import factor_ic as fic


# IC 显著性门槛:|t| 达此值才认为因子"有料",纳入自适应;否则权重归 0(不表态)。
# 2.0 约等于 95% 置信。设为可调,保守用户可调高。
_T_THRESHOLD = 2.0

# 反推后技术因子权重的目标总量纲(绝对值之和),便于和 L1 的百分制权重衔接。
# 例:技术三因子最终 |w| 之和≈这个数,再和基本面因子拼成完整 L1 权重。
_TECH_BUDGET = 60.0


def derive_weights(fwd_days: int = 5, step: int = None,
                   t_threshold: float = _T_THRESHOLD,
                   tech_budget: float = _TECH_BUDGET,
                   panel: pd.DataFrame = None,
                   ic_summary: pd.DataFrame = None,
                   progress_cb=None) -> dict:
    """
    基于 IC_IR 反推技术因子的自适应权重(带符号)。

    fwd_days/step: 传给 IC 计算(与 L2 口径一致)。
    t_threshold:   |t| 门槛,未达标的因子权重记 0(视为噪声,不参与)。
    tech_budget:   入选因子最终 |权重| 之和归一到该量纲(和 L1 百分制衔接)。
    panel/ic_summary: 允许外部传入已算好的面板/IC 表复用,避免重复计算。
    progress_cb:   进度回调(面板构建阶段)。

    返回 dict:
      weights: {因子: 带符号权重}  —— 正=顺势用,负=反向用,0=剔除。仅技术三因子。
      detail:  DataFrame(因子, ic_mean, ic_ir, t_stat, 是否入选, 原始IR权重, 最终权重)
      note:    人话总结(哪些因子入选、方向如何、可信度提醒)
      raw_ic:  透传的 IC summary,供 UI 展示
    """
    if step is None:
        step = fwd_days
    if panel is None:
        panel = pnl.build_panel(fwd_days=fwd_days, progress_cb=progress_cb)
    if ic_summary is None:
        ic_summary, _ = fic.compute_ic(fwd_days=fwd_days, step=step, panel=panel)

    if ic_summary is None or ic_summary.empty:
        return {"weights": {}, "detail": pd.DataFrame(), "raw_ic": ic_summary,
                "note": "IC 计算无结果(可能数据不足),无法反推权重。"}

    # ic_summary 的 index 是中文因子名,映射回英文键
    cn2en = {v: k for k, v in pnl._FACTOR_CN.items()}

    rows = []
    for cn_name, r in ic_summary.iterrows():
        en = cn2en.get(cn_name, cn_name)
        ic_mean = r.get("ic_mean", np.nan)
        ic_ir = r.get("ic_ir", np.nan)
        t_stat = r.get("t_stat", np.nan)
        selected = (np.isfinite(t_stat) and abs(t_stat) >= t_threshold
                    and np.isfinite(ic_ir))
        # 原始信号强度 = IC_IR(带符号)。未入选记 0。
        raw_w = float(ic_ir) if selected else 0.0
        rows.append({
            "factor": en, "factor_cn": cn_name,
            "ic_mean": float(ic_mean) if np.isfinite(ic_mean) else np.nan,
            "ic_ir": float(ic_ir) if np.isfinite(ic_ir) else np.nan,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "selected": bool(selected),
            "raw_ir_weight": raw_w,
        })
    detail = pd.DataFrame(rows)

    # 归一化:按 |raw_w| 之和缩放到 tech_budget,保留符号
    abs_sum = detail["raw_ir_weight"].abs().sum()
    if abs_sum > 0:
        scale = tech_budget / abs_sum
        detail["weight"] = (detail["raw_ir_weight"] * scale).round(2)
    else:
        detail["weight"] = 0.0

    weights = {r["factor"]: r["weight"]
               for _, r in detail.iterrows() if r["weight"] != 0.0}

    note = _build_note(detail, fwd_days, t_threshold)
    return {"weights": weights, "detail": detail,
            "raw_ic": ic_summary, "note": note}


def _build_note(detail: pd.DataFrame, fwd_days: int, t_threshold: float) -> str:
    """把结果翻译成人话(方向 + 可信度提醒,绝不夸大)。"""
    sel = detail[detail["selected"]]
    if sel.empty:
        return ("这几个指标过去都没表现出足够可信的规律,数据帮不了忙。"
                "建议先别改权重,或者把\"预测天数\"调一调再试。")
    parts = []
    for _, r in sel.iterrows():
        direction = "顺着用(得分越高越好)" if r["weight"] > 0 else "反着用(得分越低越好)"
        parts.append("%s:过去%s,建议权重%+.1f"
                     % (r["factor_cn"], direction, r["weight"]))
    body = "；".join(parts)
    neg = (sel["weight"] < 0).sum()
    hint = ""
    if neg > 0:
        hint = ("；其中 %d 个指标是【反着用】——就是说近 %d 天里,\"跌得多的反弹\"比"
                "\"强的继续强\"更赚钱。" % (neg, fwd_days))
    tail = ("。提醒:这只是过去这段行情的规律,行情风格一变(比如转牛市)可能就反过来了。"
            "一定先看下面的历史验证确认它更强,别长期照搬。")
    return "结论:" + body + hint + tail
