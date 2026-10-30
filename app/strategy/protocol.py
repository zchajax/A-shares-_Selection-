"""
有效性检验的冻结规则与共用计算。

下面这些常数在看测试窗结果之前写死。重跑检验时不要为了让结论变好而修改它们。
IC、分档和组合用的是同一套处理后的因子、同一种开盘到开盘可成交收益。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --- 冻结常数。训练截止日按日历写下，不随收益移动。 ---
TRAIN_END = "2023-12-31"
# 持有期候选只在训练窗里比较，测试窗只跑选中的那一个。
HORIZONS = (5, 10, 20)
# 训练窗截面相关达到此值，动量与趋势只留 |Newey-West t| 更大的一个。
CORR_MAX = 0.70
# 因子要进入权重，训练窗 |t| 须达到此值。
T_MIN = 2.0
# 佣金单边万 2.5。不搜索。
COMMISSION = 0.00025
# 卖出印花税：2023-08-28 起由千一改为万五。
STAMP_DATE = "2023-08-28"
STAMP_BEFORE = 0.001
STAMP_AFTER = 0.0005
# 开盘涨跌幅距涨跌停在 0.2 个百分点以内，视为封板开盘。
LIMIT_EPS = 0.002
# 相邻收盘涨跌幅超过 31%（宽于北交所 30% 限制）记为复权断裂候选。
BREAK_ABS = 0.31
MIN_NAMES = 20
MIN_BOOK = 5
MIN_INDUSTRY = 5
WARMUP = 90

TECH_FACTORS = ("momentum", "trend", "volume")


def stamp_rate(date) -> float:
    """卖出日印花税。date 为 YYYY-MM-DD 字符串。"""
    return STAMP_AFTER if str(date) >= STAMP_DATE else STAMP_BEFORE


def limit_ratio(code: str) -> float:
    """板块涨跌停幅度。样本从 2021 年起，创业板已是 20%。没有逐日 ST 标记，不用 5%。"""
    c = str(code).zfill(6)
    if c.startswith(("688", "689", "300", "301")):
        return 0.20
    if c.startswith(("8", "4", "92")):
        return 0.30
    return 0.10


def is_limit_up_gap(gap: float, limit: float) -> bool:
    return np.isfinite(gap) and gap >= limit - LIMIT_EPS


def is_limit_down_gap(gap: float, limit: float) -> bool:
    return np.isfinite(gap) and gap <= -(limit - LIMIT_EPS)


def net_of_costs(entry: float, exit_px: float, exit_date) -> float:
    """整段换手都成交时的净收益。组合层按实际换手另计，避免连续持有被收两次。"""
    sell = exit_px * (1.0 - COMMISSION - stamp_rate(exit_date))
    buy = entry * (1.0 + COMMISSION)
    return sell / buy - 1.0


def newey_west_mean(x, lags=None):
    """
    均值的 Newey-West t。lags=None 时用 Bartlett 自动滞后 floor(4*(n/100)^(2/9))。
    返回 (mean, t, lags)。方差非正时 t 为 nan。
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 3:
        return (float(arr.mean()) if n else np.nan), np.nan, 0
    mu = float(arr.mean())
    u = arr - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = int(max(0, lags))
    gamma0 = float(np.dot(u, u) / n)
    acc = gamma0
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        gamma = float(np.dot(u[L:], u[:-L]) / n)
        acc += 2.0 * w * gamma
    var_mean = acc / n
    if not np.isfinite(var_mean) or var_mean <= 0:
        return mu, np.nan, lags
    t = mu / np.sqrt(var_mean)
    return mu, float(t), lags


def rebalance_cost(prev_w: dict, new_w: dict, trade_date) -> float:
    """换仓日的佣金和卖出印花税。权重是组合权重，买入额与卖出额按权重变化计算。"""
    buy = 0.0
    sell = 0.0
    for code in set(prev_w) | set(new_w):
        dw = float(new_w.get(code, 0.0)) - float(prev_w.get(code, 0.0))
        if dw > 0:
            buy += dw
        elif dw < 0:
            sell -= dw
    return buy * COMMISSION + sell * (COMMISSION + stamp_rate(trade_date))


def turnover_parts(prev_w: dict, new_w: dict):
    buy = 0.0
    sell = 0.0
    for code in set(prev_w) | set(new_w):
        dw = float(new_w.get(code, 0.0)) - float(prev_w.get(code, 0.0))
        if dw > 0:
            buy += dw
        elif dw < 0:
            sell -= dw
    return buy, sell


def one_way_turnover(prev_w: dict, new_w: dict) -> float:
    buy, sell = turnover_parts(prev_w, new_w)
    return 0.5 * (buy + sell)


def apply_collinearity(weights: dict, tstats: dict, corr_mt: float) -> dict:
    """动量与趋势相关过高时，丢掉训练窗 |t| 较小的一个。|t| 相同则丢趋势。"""
    out = dict(weights)
    if not np.isfinite(corr_mt) or abs(corr_mt) < CORR_MAX:
        return out
    tm = abs(tstats.get("momentum", np.nan))
    tt = abs(tstats.get("trend", np.nan))
    drop = "trend" if (not np.isfinite(tt) or not np.isfinite(tm) or tm >= tt) else "momentum"
    out[drop] = 0.0
    return out


def weights_from_ic(ic_mean: dict, ic_ir: dict, tstats: dict) -> dict:
    """训练窗 IC_IR 带符号权重。|t| 不够的因子权重为 0，不回退到默认权重。"""
    out = {}
    for f in TECH_FACTORS:
        t = tstats.get(f, np.nan)
        ir = ic_ir.get(f, np.nan)
        if np.isfinite(t) and abs(t) >= T_MIN and np.isfinite(ir):
            out[f] = float(ir)
        else:
            out[f] = 0.0
    return out


def train_slice(panel: pd.DataFrame) -> pd.DataFrame:
    """只保留卖出日不晚于训练截止日的行。没有 exit_date 时原样返回，避免用信号日误切。"""
    if panel is None or panel.empty or "exit_date" not in panel.columns:
        return panel
    m = panel["exit_date"].astype(str) <= TRAIN_END
    return panel.loc[m].copy()


def test_slice(panel: pd.DataFrame) -> pd.DataFrame:
    """信号日严格晚于训练截止日。与 train_slice 中间的几天是隔离带，两边都不用。"""
    if panel is None or panel.empty:
        return panel
    m = panel["date"].astype(str) > TRAIN_END
    return panel.loc[m].copy()


def attach_processed(df: pd.DataFrame, factors=TECH_FACTORS) -> pd.DataFrame:
    """
    每个截面日内：1%/99% 去极值，再做行业去均值。
    行业样本少于 MIN_INDUSTRY 的，改为减去当日全市场均值。
    随后在截面内做 z-score，供加权。秩相关对 z-score 不变，所以 IC 用去均值后的列。
    """
    if df is None or df.empty:
        return df
    if all(f + "_p" in df.columns for f in factors):
        return df
    out = df.copy()
    if "industry" not in out.columns:
        out["industry"] = "未分类"
    out["industry"] = out["industry"].fillna("未分类").replace("", "未分类")

    def _win(s: pd.Series) -> pd.Series:
        v = s.to_numpy(dtype=float)
        if np.isfinite(v).sum() < 10:
            return pd.Series(np.nan, index=s.index)
        lo, hi = np.nanpercentile(v, [1, 99])
        return s.clip(lo, hi)

    for f in factors:
        w = out.groupby("date", sort=False)[f].transform(_win)
        tmp = pd.DataFrame({
            "date": out["date"].to_numpy(),
            "industry": out["industry"].to_numpy(),
            "_w": w.to_numpy(),
        }, index=out.index)
        cnt = tmp.groupby(["date", "industry"], sort=False)["_w"].transform("count")
        mu = tmp.groupby(["date", "industry"], sort=False)["_w"].transform("mean")
        mkt = tmp.groupby("date", sort=False)["_w"].transform("mean")
        dem = w - mu
        dem = dem.where(cnt >= MIN_INDUSTRY, w - mkt)
        out[f + "_p"] = dem

        def _z(s: pd.Series) -> pd.Series:
            sd = s.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                return pd.Series(np.nan, index=s.index)
            return (s - s.mean()) / sd

        out[f + "_z"] = out.groupby("date", sort=False)[f + "_p"].transform(_z)
        out.loc[out[f + "_p"].isna(), f + "_z"] = np.nan
    return out


def composite_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    """处理后的 z-score 按冻结权重相加。权重全 0 时返回全 nan。"""
    active = {k: float(v) for k, v in weights.items() if v}
    if not active:
        return pd.Series(np.nan, index=df.index)
    wsum = sum(abs(v) for v in active.values())
    score = None
    for f, w in active.items():
        col = df[f + "_z"] * (w / wsum)
        score = col if score is None else score + col
    return score


def self_check() -> list[str]:
    """不读行情库的机械核对。任一失败，检验脚本不得写测试结论。"""
    lines = []

    def check(name, cond):
        lines.append(("PASS " if cond else "FAIL ") + name)
        return cond

    ok = True
    ok &= check("印花税 2023-08-27 为千一", stamp_rate("2023-08-27") == STAMP_BEFORE)
    ok &= check("印花税 2023-08-28 为万五", stamp_rate("2023-08-28") == STAMP_AFTER)
    ok &= check("主板 10% 开盘视为涨停", is_limit_up_gap(0.10, 0.10))
    ok &= check("9% 开盘不是涨停", not is_limit_up_gap(0.09, 0.10))
    ok &= check("创业板代码幅度为 20%", limit_ratio("300001") == 0.20)
    ok &= check("主板代码幅度为 10%", limit_ratio("600000") == 0.10)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mu, t, lags = newey_west_mean(x, lags=0)
    se = np.sqrt(0.4)
    ok &= check("Newey-West lags=0 与总体标准误一致",
                lags == 0 and abs(mu - 3.0) < 1e-12 and abs(t - 3.0 / se) < 1e-9)
    mu1, t1, lags1 = newey_west_mean(x, lags=1)
    se1 = np.sqrt(2.8 / 5.0)
    ok &= check("Newey-West lags=1 与手算一致",
                lags1 == 1 and abs(t1 - 3.0 / se1) < 1e-9)

    gross = net_of_costs(10.0, 11.0, "2024-01-02")
    hand = (11.0 * (1 - COMMISSION - STAMP_AFTER)) / (10.0 * (1 + COMMISSION)) - 1
    ok &= check("单笔净收益公式", abs(gross - hand) < 1e-12)

    cost0 = rebalance_cost({}, {"000001": 1.0}, "2024-01-02")
    ok &= check("空仓建仓只付买入佣金", abs(cost0 - COMMISSION) < 1e-12)
    cost1 = rebalance_cost({"000001": 1.0}, {}, "2023-08-01")
    ok &= check("清仓付佣金加调整前印花税",
                abs(cost1 - (COMMISSION + STAMP_BEFORE)) < 1e-12)

    w = {"momentum": 1.0, "trend": 1.0, "volume": 0.0}
    tstats = {"momentum": 3.0, "trend": 2.5, "volume": 0.4}
    dropped = apply_collinearity(w, tstats, 0.85)
    ok &= check("相关 0.85 时丢掉 |t| 较小的趋势", dropped["trend"] == 0.0 and dropped["momentum"] == 1.0)
    kept = apply_collinearity(w, tstats, 0.4)
    ok &= check("相关 0.40 时两个都留", kept["trend"] == 1.0 and kept["momentum"] == 1.0)

    ww = weights_from_ic(
        {"momentum": 0.02, "trend": -0.01, "volume": 0.0},
        {"momentum": 0.3, "trend": -0.2, "volume": 0.01},
        {"momentum": 2.5, "trend": 1.2, "volume": 4.0},
    )
    ok &= check("权重只留给 |t|>=2 的因子且保留符号",
                ww["momentum"] == 0.3 and ww["trend"] == 0.0 and ww["volume"] == 0.01)

    toy = pd.DataFrame({
        "date": ["2023-12-29", "2024-01-03"],
        "exit_date": ["2023-12-29", "2024-01-08"],
    })
    # 第一行卖出日在截止日，留下；第二行是测试信号。
    # 再造一行隔离带：信号在截止日前，卖出日越过截止日。
    toy = pd.DataFrame({
        "date": ["2023-12-20", "2023-12-29", "2024-01-03"],
        "exit_date": ["2024-01-04", "2023-12-29", "2024-01-10"],
    })
    tr = set(train_slice(toy)["date"])
    te = set(test_slice(toy)["date"])
    ok &= check("隔离带不进训练窗也不进测试窗",
                tr == {"2023-12-29"} and te == {"2024-01-03"})

    if not ok:
        raise RuntimeError("机械核对失败:\n" + "\n".join(lines))
    return lines
