"""
策略层 - L1 横截面多因子打分引擎(智能版选股地基)
==================================================
和现有"逐股 evaluate"策略的本质区别:
  · 老策略只看单只票自己的历史,用绝对阈值(RSI<30 / 跌幅>15%)卡线,
    孤立判断"这只票达标了吗"。市场涨落时同一阈值含义完全不同。
  · 本引擎同时握有【全市场】的因子矩阵,把每个因子做横截面标准化 + 行业中性化,
    问的是"这只票在全市场里排第几",阈值自动适应市场温度,且消除行业系统性偏差。

流水线:
  原始因子矩阵 → ① winsorize去极值 → ② z-score标准化 → ③ 行业中性化 → ④ 加权合成 → ⑤ 全市场分位排名(0~100)

5 个因子(方向统一为"越大越好"),数据全部本地现成、零新增接口:
  · momentum 动量   —— 近20日涨幅(强者恒强)
  · trend    趋势   —— 均线发散度 (ma5-ma60)/ma60
  · volume   量能   —— 近5日/近20日均量放大倍数
  · value    低估值 —— 盈利收益率 1/PE(越大越便宜;负PE=亏损视为缺失)
  · quality  质量   —— ROE(净资产收益率)

缺失值语义:某因子缺失(如 ETF 没 PE)→ 该因子 z-score 记 0(中性),
  既不误杀也不白给分。这与 funda 过滤器的"启用门槛即须有值"是两套逻辑:
  这里是打分(缺失=不表态),那里是硬门槛(缺失=不达标)。

红线:全程本地量化、可复现,不碰全市场原始数据以外的东西,不编造。
对外主入口: scan_cross_section(weights, top_n, neutralize, progress_cb)
"""
import numpy as np
import pandas as pd

from ..data import database as db
from . import indicators as ind


# 默认因子权重(可在 UI 调节)。等权偏动量/趋势,价值/质量作稳健补充。
DEFAULT_WEIGHTS = {
    "momentum": 25.0,
    "trend": 25.0,
    "volume": 15.0,
    "value": 20.0,
    "quality": 15.0,
}

# 因子中文名(reason 展示用)
_FACTOR_CN = {
    "momentum": "动量", "trend": "趋势", "volume": "量能",
    "value": "低估", "quality": "质量",
}

# 组内中性化的最小样本数:少于此数的行业组回退到全市场 z-score
_MIN_GROUP = 5


def _winsorize(s: pd.Series, n: float = 3.0) -> pd.Series:
    """去极值:把超过 均值±n倍标准差 的值裁剪到边界,防单只妖股带歪整列。"""
    x = s.astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return x
    lo, hi = mu - n * sd, mu + n * sd
    return x.clip(lower=lo, upper=hi)


def _zscore(s: pd.Series) -> pd.Series:
    """标准化: (x-均值)/标准差。缺失值保持 NaN(后续统一填 0=中性)。"""
    x = s.astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (x - mu) / sd


def _neutralize_by_group(z: pd.Series, groups: pd.Series) -> pd.Series:
    """
    行业中性化:在每个行业组内分别做 z-score,消除"低估值全是银行/高动量全是小盘"
    这类系统性偏差。组样本过少(<_MIN_GROUP)或组内无波动时,回退用已算好的全市场 z。
    """
    out = z.copy()
    for g, idx in groups.groupby(groups).groups.items():
        sub = z.loc[idx]
        valid = sub.dropna()
        if len(valid) < _MIN_GROUP or valid.std(ddof=0) == 0:
            continue  # 回退:保留全市场 z
        out.loc[idx] = _zscore(sub)
    return out


def _extract_raw_factors(progress_cb=None) -> pd.DataFrame:
    """
    遍历全市场缓存,为每只票取最新一根 K 线上的原始因子值。
    返回 DataFrame(index=code): momentum/trend/volume/value/quality/industry/name/close
    这是横截面引擎与老 scanner 的分水岭 —— 一次性把全市场摊成一张因子表。
    """
    codes = db.list_cached_codes()
    slist = db.load_stock_list()
    namemap = dict(zip(slist["code"], slist["name"]))
    indmap = dict(zip(slist["code"], slist["industry"]))
    fmap = db.load_fundamental_map()

    rows = []
    total = len(codes)
    for i, code in enumerate(codes):
        if progress_cb:
            progress_cb(i + 1, total)
        df = db.load_kline(code)
        if df is None or df.empty or len(df) < 65:
            continue
        df = ind.enrich(df)
        row = df.iloc[-1]

        # --- 技术因子(单票历史即可算,方向统一为"越大越好") ---
        # 动量:近20日涨幅%
        momentum = row.get("chg_20")
        # 趋势:均线发散度 (ma5-ma60)/ma60
        ma60 = row.get("ma60")
        trend = ((row.get("ma5") - ma60) / ma60
                 if ma60 and np.isfinite(ma60) and ma60 != 0 else np.nan)
        # 量能:近5日/近20日均量倍数
        v5 = df["volume"].rolling(5).mean().iloc[-1]
        v20 = df["volume"].rolling(20).mean().iloc[-1]
        volume = v5 / v20 if v20 and np.isfinite(v20) and v20 != 0 else np.nan

        # --- 基本面因子(截面数据,缺失=中性) ---
        f = fmap.get(code) or {}
        pe = f.get("pe_ttm")
        roe = f.get("roe")
        # 价值:盈利收益率 1/PE(越大越便宜);负PE=亏损,无估值意义 -> 视为缺失
        value = (1.0 / pe) if (pe is not None and pd.notna(pe) and pe > 0) else np.nan
        quality = roe if (roe is not None and pd.notna(roe)) else np.nan

        rows.append({
            "code": code,
            "name": namemap.get(code, ""),
            "industry": indmap.get(code) or "未分类",
            "close": round(float(row["close"]), 2),
            "momentum": momentum,
            "trend": trend,
            "volume": volume,
            "value": value,
            "quality": quality,
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("code")
    return out


def scan_cross_section(weights: dict = None, top_n: int = 50,
                       neutralize: bool = True, progress_cb=None) -> pd.DataFrame:
    """
    L1 横截面多因子选股主入口。

    weights:    因子权重 dict(如 {"momentum":25,"trend":25,...}),None=用 DEFAULT_WEIGHTS。
                权重为 0 即关闭该因子。
    top_n:      返回前 N 只。
    neutralize: 是否做行业中性化(True=在行业内排名,消除行业偏差)。
    progress_cb(done, total): 进度回调(与 scanner.scan 兼容)。

    返回 DataFrame: code,name,score(0~100全市场分位),close,reason,industry
                    按 score 降序。列与 scanner.scan 兼容,UI 可直接复用渲染。
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k in w:
            if k in weights and weights[k] is not None:
                w[k] = float(weights[k])
    # 保留权重非0的因子参与合成。支持负权重(L3自适应):负=反向使用该因子。
    active = {k: v for k, v in w.items() if v != 0}
    if not active:
        active = dict(DEFAULT_WEIGHTS)

    raw = _extract_raw_factors(progress_cb=progress_cb)
    if raw.empty:
        return pd.DataFrame(columns=["code", "name", "score", "close", "reason", "industry"])

    factors = list(active.keys())
    groups = raw["industry"]

    # ① winsorize → ② z-score → ③ 行业中性化,得每个因子的横截面得分
    z_cols = {}
    for fac in factors:
        wz = _winsorize(raw[fac])
        z = _zscore(wz)
        if neutralize:
            z = _neutralize_by_group(z, groups)
        # 缺失填 0(中性:等于该组/全市场均值,不奖不罚)
        z_cols[fac] = z.fillna(0.0)
    zdf = pd.DataFrame(z_cols, index=raw.index)

    # ④ 加权合成(权重归一化;带符号,负权重=反向使用该因子)
    wsum = sum(abs(v) for v in active.values()) or 1.0
    composite = sum(zdf[fac] * (active[fac] / wsum) for fac in factors)

    # ⑤ 转全市场分位排名(0~100),越大越好
    score = composite.rank(pct=True) * 100.0

    out = raw[["name", "close", "industry"]].copy()
    out["score"] = score.round(2)
    out["_composite"] = composite

    # reason:列出各因子的横截面 z 值(让用户看懂"为什么它排前面")。
    # 负权重因子加"↓"标记,提示"这里是反向使用:该值越小反而越加分"。
    def _mk_reason(code):
        parts = []
        for fac in factors:
            mark = "↓" if active[fac] < 0 else ""
            parts.append(f"{_FACTOR_CN[fac]}{mark}{zdf.loc[code, fac]:+.2f}")
        return " ".join(parts)

    out["reason"] = [_mk_reason(c) for c in out.index]
    out = out.reset_index()  # code 变回列
    out = out.sort_values("_composite", ascending=False).reset_index(drop=True)
    out = out.drop(columns=["_composite"])

    if top_n and top_n > 0:
        out = out.head(int(top_n)).reset_index(drop=True)
    return out[["code", "name", "score", "close", "reason", "industry"]]
