"""
筹码分布(成本分布)本地计算。

【为什么本地算】东财筹码接口 stock_cyq_em 长期断连/超时(项目已知问题),
不可依赖。筹码分布本可用日线 OHLC+成交量在本地推算,更可靠、可复现,
符合本项目"本地量化可复现、不依赖外部接口"的红线。

【算法:三角形分布衰减模型】(与东财/通达信同款思路)
- 每个交易日的成交量,按当日价格区间[low, high]分摊到若干价格档上。
  分摊形状用三角形分布:均价(近似(high+low+close)/3)处筹码最多,
  两端递减 —— 比均匀分布更贴近真实成交结构。
- 历史筹码随时间"换手衰减":新一天换手率为 t 时,
  昨日存量筹码乘以 (1 - t*A) 被"换手转移"出来,再叠加当日新增筹码。
  A 为衰减系数(通常取 1,可调),t=当日成交量/流通股本。
- 最终得到"当前时点每个价格档上沉淀的筹码量",归一化后即成本分布。

【派生指标】
- 获利盘比例:成本低于现价的筹码占比(越高说明浮盈盘越多,潜在抛压)。
- 平均成本:筹码量加权的价格均值。
- 90%成本区间:剔除上下各5%极端后,筹码集中的价格上下沿。
- 集中度:90%区间宽度/均价,越小越集中(易拉升),越大越分散。

【流通股本来源】优先用传入值;否则用总市值/现价近似(总股本≈流通股本,
小盘/次新会偏差,但作为衰减权重的量级足够)。换手率用于衰减,量级正确即可。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_chip_distribution(
    df: pd.DataFrame,
    float_shares: Optional[float] = None,
    total_mv: Optional[float] = None,
    bins: int = 150,
    decay: float = 1.0,
    lookback: int = 250,
) -> Optional[dict]:
    """
    计算筹码分布。

    参数
    ----
    df : 日线 DataFrame,需含 open/high/low/close/volume(volume 单位:股)。
    float_shares : 流通股本(股)。缺失时用 total_mv/现价 近似。
    total_mv : 总市值(亿元),用于近似流通股本。
    bins : 价格分档数。
    decay : 换手衰减系数(1=标准)。
    lookback : 只用最近多少个交易日累积(筹码有时效,过老的意义不大)。

    返回
    ----
    dict 或 None(数据不足)。字段:
      prices        : np.ndarray  各价格档中值
      chips         : np.ndarray  各价格档筹码量(已归一化,和=1)
      last_close    : float       最新收盘价
      profit_ratio  : float       获利盘比例(0~1)
      avg_cost      : float       平均成本
      cost_low/high : float       90%成本区间上下沿
      concentration : float       集中度(90%区间宽度/均价)
    """
    if df is None or df.empty or len(df) < 20:
        return None
    # 【健壮性】清洗:只保留 OHLCV 关键列均有效(非 NaN)的行,避免停牌/缺失日的空值
    #   在下游 int(np.clip((c-lo)/step)) 处触发 "cannot convert NaN to integer"。
    need = ["high", "low", "close", "volume"]
    d = df.copy()
    for col in need:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=need)
    if len(d) < 20:
        return None
    d = d.tail(lookback).reset_index(drop=True)
    highs = d["high"].astype(float).to_numpy()
    lows = d["low"].astype(float).to_numpy()
    closes = d["close"].astype(float).to_numpy()
    vols = d["volume"].astype(float).to_numpy()
    last_close = float(closes[-1])

    # 流通股本(股):优先显式传入,否则用总市值/现价近似
    if float_shares and float_shares > 0:
        fs = float(float_shares)
    elif total_mv and total_mv > 0 and last_close > 0:
        fs = float(total_mv) * 1e8 / last_close   # 总市值(亿)→股
    else:
        fs = None

    # 【换手率量级保护】本地K线多为复权数据,做过大比例送转/转增的票(如688166
    #   博瑞医药),复权会把历史成交量放大数倍~数十倍,导致 turn=v/fs 虚高、单日
    #   换手率甚至 >100%(物理不可能)。此时凸组合 alpha 常年≈1,每天几乎全量清空
    #   重建,分布塌缩到最近一两天价位(只剩两根畸高柱),高位套牢盘被抹掉。
    #   修法:检测中位日换手率,若异常偏高(>15%,正常A股个股日换手多在0.1%~10%),
    #   将整段换手率等比缩放到合理中位(5%),只压量级、保留日间相对活跃度。
    #   对正常换手股(中位≤15%)不触发,数值分毫不变,不误伤。
    turn_scale = 1.0
    if fs and fs > 0:
        _raw = vols / fs
        _pos = _raw[_raw > 0]
        if _pos.size > 0:
            _med = float(np.median(_pos))
            _MAX_MED = 0.15   # 正常个股中位日换手率上限
            _TARGET = 0.05    # 缩放目标中位
            if _med > _MAX_MED:
                turn_scale = _TARGET / _med

    # 价格分档:覆盖历史全区间,留 2% 余量
    pmin = float(lows.min())
    pmax = float(highs.max())
    if pmax <= pmin:
        return None
    span = pmax - pmin
    lo = pmin - span * 0.02
    hi = pmax + span * 0.02
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    step = edges[1] - edges[0]

    def _today_dist(h, low_, c):
        """当日成交的三角形分布(归一化,和=1):均价处最多,两端递减。"""
        avg = (h + low_ + c) / 3.0
        t = np.zeros(bins, dtype=float)
        mask = (centers >= low_ - step) & (centers <= h + step)
        if not mask.any():
            if not np.isfinite(c):
                return t   # c 异常:返回全零(该日不贡献),不崩
            j = int(np.clip((c - lo) / step, 0, bins - 1))
            t[j] = 1.0
            return t
        seg = centers[mask]
        half = max(h - avg, avg - low_, step)
        w = np.clip(1.0 - np.abs(seg - avg) / half, 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(seg)
        t[np.where(mask)[0]] = w / w.sum()
        return t

    # 初始存量筹码:用【整个窗口的成交量加权分布】铺底,代表"进入窗口时已沉淀
    #   的存量筹码"的成本基准。
    # 【为什么必须是全窗口量加权,而不是最早N天】低换手股(银行/大盘蓝筹,日换手
    #   仅~0.04%)在 250 天里累计换手可能只有 10%,意味着约 90% 的筹码在窗口期内
    #   几乎没被换手过——它们的真实成本应贴近这段时间的整体成交价分布。
    #   · 若初始设成"第一天单根尖峰":该尖峰权重=(1-turn)^250 永远稀释不掉,分布被
    #     第一天绑架 → 虚假极度集中(红柱畸长)。
    #   · 若只用"最早20天"铺底:这20天的窄价格代表不了真实进场成本,仍把90%存量
    #     筹码压在过窄区间(实测建行90%区间被压成8.86~9.51,真实成交价却是
    #     8.42~10.02),集中度虚高一倍。
    #   · 用【全窗口量加权分布】铺底:存量筹码贴合真实成交结构,再经每日凸组合演化,
    #     低换手股分布自然铺开(建行集中度0.071→0.177,贴合真实0.12~0.20)。
    #   对高换手股(累计换手数百%~数百倍):初始存量在窗口内被反复换手冲刷殆尽,
    #   用何种铺底结果完全一致(实测当升/博瑞三种铺底数值分毫不差),故不误伤。
    base = np.zeros(bins, dtype=float)
    for i in range(len(d)):
        if vols[i] > 0 and highs[i] >= lows[i]:
            base += _today_dist(highs[i], lows[i], closes[i]) * vols[i]
    if base.sum() > 0:
        chips = base / base.sum()
        inited = True
    else:
        chips = np.zeros(bins, dtype=float)
        inited = False

    for i in range(len(d)):
        h, low_, c, v = highs[i], lows[i], closes[i], vols[i]
        if v <= 0 or h < low_:
            continue
        # 当日换手率:被换手的老筹码比例 = 当日新沉淀筹码比例。
        # 无流通股本时用一个温和默认值(量级合理即可)。
        # turn_scale: 复权放量股的换手率量级保护(见上方注释),正常股恒为1。
        if fs:
            turn = min(v / fs * turn_scale, 1.0)
        else:
            turn = 0.03   # 兜底:假设日换手3%
        alpha = float(np.clip(turn * decay, 0.0, 1.0))   # 凸组合权重,∈[0,1]

        today = _today_dist(h, low_, c)

        # 凸组合(通达信标准):每天有 alpha 比例的老筹码被换手到当日成交价位,
        #   剩余 (1-alpha) 保持不动。chips 始终为归一化分布(和=1),不受成交量绝对
        #   量级影响 —— 避免"注入量∝v² 把放量日筹码平方放大"的错误。
        if not inited:
            chips = today.copy()
            inited = True
        else:
            chips = chips * (1.0 - alpha) + today * alpha

    total = chips.sum()
    if total <= 0:
        return None
    chips = chips / total   # 数值兜底再归一化(理论上已=1)

    # 派生指标
    profit_ratio = float(chips[centers <= last_close].sum())   # 成本≤现价=浮盈
    avg_cost = float((centers * chips).sum())

    # 90% 成本区间:按价格排序累积,取 5%~95% 分位
    order = np.argsort(centers)
    cp = np.cumsum(chips[order])
    p_sorted = centers[order]
    cost_low = float(p_sorted[np.searchsorted(cp, 0.05)])
    idx_hi = min(np.searchsorted(cp, 0.95), len(p_sorted) - 1)
    cost_high = float(p_sorted[idx_hi])
    concentration = float((cost_high - cost_low) / avg_cost) if avg_cost else 0.0

    return {
        "prices": centers,
        "chips": chips,
        "last_close": last_close,
        "profit_ratio": profit_ratio,
        "avg_cost": avg_cost,
        "cost_low": cost_low,
        "cost_high": cost_high,
        "concentration": concentration,
        "shares_known": fs is not None,
    }
