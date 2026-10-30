"""
四层量化决策与多周期样本外评测引擎 (Multi-Horizon Cockpit Engine v2)
====================================================================
升级亮点（基于本地全量数据的高阶量化工程）：
  1. 15 维高阶微观结构 + 板块动量溢出因子库：
     - 新增 Amihud 非流动性 (amihud20)、下行/上行波动不对称比 (down_up_vol20)、
       下影线承接减去上影线抛压 (shadow_net10)、量价相关性 (pv_corr20)、
       60日特质波动率 (ivol60)，以及行业 5 日动量溢出 (sec_mom5) 与个股行业内超额 (alpha_ind5)。
  2. 「三重屏障（Triple-Barrier）」实战胜率与收益引擎：
     - 同时评估固定持有期开盘平仓 (fwd_ret / net_ret) 与实盘带止盈/止损的三重屏障结果 (tb_win / tb_net_ret)，
       真实模拟交易员“触及 +k1·ATR 止盈落袋、触及 -k2·ATR 止损出局、未触及则到期平仓”的实战路径。
  3. 行业中性超额概率 + 三重屏障实战模型双轨拟合（仅在 <=2023-12-31 训练窗拟合）。
  4. 组合级 2021~2026 净值曲线 (Equity Curve)、水下回撤序列 (Drawdown)、夏普/卡玛比率与分年度归因表。
  5. 一票否决排雷卫士 (Risk Veto) + 实盘推荐信号自动记账本 (Signal Ledger) + 自选持仓移动止盈/止损监控。
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from ..data import database as db
from . import panel as pnl
from . import protocol as proto
from . import winmodel as wm


SUPPORTED_HORIZONS = (10, 60, 120)

HORIZON_LABELS = {
    10: "短波段 (10日 · 两周)",
    60: "中线 (60日 · 一季度)",
    120: "长线 (120日 · 半年)",
}

# 各周期三重屏障的 (止损ATR倍数, 止盈ATR倍数, 单笔风险预算)
TB_PARAMS = {
    10: (1.5, 2.5, 0.010),
    60: (3.2, 7.5, 0.015),
    120: (4.2, 11.0, 0.020),
}

REGIME_BULL = "BULL"
REGIME_NEUTRAL = "NEUTRAL"
REGIME_BEAR = "BEAR"

REGIME_CN_SHORT = {
    REGIME_BULL: "绿灯 · 进攻期(多头主升/冰点企稳反攻)",
    REGIME_NEUTRAL: "黄灯 · 观察期(结构分化/冰点磨底待确认)",
    REGIME_BEAR: "红灯 · 防守期(破位普跌/恐慌寻底/高位派发)",
}

REGIME_CN_LONG = {
    REGIME_BULL: "绿灯 · 长线击球区(低位企稳/长牛健康通道)",
    REGIME_NEUTRAL: "黄灯 · 左侧分批区(低位寻底未稳/结构分化)",
    REGIME_BEAR: "红灯 · 周期高位风险区(全市场过热，防长线站岗)",
}

REGIME_POS_CAP = {
    REGIME_BULL: 0.80,
    REGIME_NEUTRAL: 0.40,
    REGIME_BEAR: 0.00,
}

REGIME_SCALE = {
    REGIME_BULL: 1.0,
    REGIME_NEUTRAL: 0.5,
    REGIME_BEAR: 0.0,
}

# 高阶微观结构与板块联动因子集（短波段 15 维）
SHORT_ADV_FEATURES = (
    "rev1", "rev5", "mom20", "mom60ex5", "vol20",
    "volratio", "overnight5", "intraday5", "range20", "highdist",
    "amihud20", "down_up_vol20", "shadow_net10", "pv_corr20", "alpha_ind5",
)

# 中长线特征集（16 维：融入 Point-in-Time 历史真实季报 ROE 与利润增速）
LONG_FEATURES = (
    "pos250", "mom120ex10", "mom60ex5", "mom20",
    "vol60", "vol20", "range20", "volratio", "dist_ma60", "highdist",
    "amihud20", "down_up_vol20", "shadow_net10", "pv_corr20",
    "pit_roe", "pit_profit_yoy",
)

ALL_FEATURE_CN = {
    **wm.FEATURE_CN,
    "pos250": "一年价格分位",
    "mom120ex10": "半年涨跌(不含近10日)",
    "vol60": "60日长期波动",
    "dist_ma60": "距60日季线偏离",
    "amihud20": "Amihud非流动性",
    "down_up_vol20": "下行/上行波动比",
    "shadow_net10": "下影承接净值",
    "pv_corr20": "20日量价相关性",
    "alpha_ind5": "5日行业内超额",
    "pit_roe": "历史时点季报ROE",
    "pit_profit_yoy": "历史时点净利同比",
}

LONG_FEATURE_CN = ALL_FEATURE_CN
REGIME_CN = REGIME_CN_SHORT


def regime_cn_of(reg: str, horizon: int = 10) -> str:
    if int(horizon) >= 60:
        return REGIME_CN_LONG.get(reg, reg)
    return REGIME_CN_SHORT.get(reg, reg)


def _root() -> str:
    return os.path.dirname(os.path.dirname(db.DB_PATH))


def cockpit_state_path() -> str:
    return os.path.join(_root(), "cockpit_state.json")


def cockpit_record_path() -> str:
    return os.path.join(_root(), "cockpit_record.txt")


def load_cockpit_state() -> Optional[dict]:
    path = cockpit_state_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


_PIT_CACHE_BY_CODE: Optional[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = None


def _get_pit_cache() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """加载并缓存每只股票按公告日 notice_date 排序的历史季报 (notice_dates, roe_arr, profit_yoy_arr)。"""
    global _PIT_CACHE_BY_CODE
    if _PIT_CACHE_BY_CODE is not None:
        return _PIT_CACHE_BY_CODE
    df = db.load_pit_financials()
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if df is not None and not df.empty:
        df = df.dropna(subset=["code", "notice_date"]).sort_values(["code", "notice_date", "report_date"])
        for code, grp in df.groupby("code", sort=False):
            nds = grp["notice_date"].astype(str).to_numpy()
            roes = pd.to_numeric(grp["roe"], errors="coerce").to_numpy(dtype=float)
            pyoys = pd.to_numeric(grp["profit_yoy"], errors="coerce").clip(-200.0, 500.0).to_numpy(dtype=float)
            cache[str(code).zfill(6)] = (nds, roes, pyoys)
    _PIT_CACHE_BY_CODE = cache
    return _PIT_CACHE_BY_CODE


def _extended_item_arrays(item: dict) -> dict:
    """为单只股票计算并缓存 15+ 维高阶微观结构、均线趋势与 ATR 时间序列（零未来函数）。"""
    cached = item.get("_cockpit_ext")
    if cached is not None:
        return cached

    feat = wm._features_of(item)
    close = item["close"]
    open_ = item["open"]
    high = item["high"]
    low = item["low"]
    vol = item["volume"]
    n = close.size

    c_ser = pd.Series(close)
    ret = c_ser.pct_change().to_numpy()
    ret_ser = pd.Series(ret)

    ma5 = c_ser.rolling(5).mean().to_numpy()
    ma20 = c_ser.rolling(20).mean().to_numpy()
    ma60 = c_ser.rolling(60).mean().to_numpy()
    ma120 = c_ser.rolling(120, min_periods=60).mean().to_numpy()
    ma250 = c_ser.rolling(250, min_periods=90).mean().to_numpy()

    ma20_prev5 = np.full(n, np.nan)
    ma60_prev10 = np.full(n, np.nan)
    ma120_prev20 = np.full(n, np.nan)
    if n > 5:
        ma20_prev5[5:] = ma20[:-5]
    if n > 10:
        ma60_prev10[10:] = ma60[:-10]
    if n > 20:
        ma120_prev20[20:] = ma120[:-20]

    high20_prev = pd.Series(high).rolling(20).max().shift(1).to_numpy()
    hi250 = pd.Series(high).rolling(250, min_periods=60).max().to_numpy()
    lo250 = pd.Series(low).rolling(250, min_periods=60).min().to_numpy()
    vol60 = ret_ser.rolling(60, min_periods=20).std(ddof=0).to_numpy()

    mom120ex10 = np.full(n, np.nan)
    if n > 120:
        denom = close[:-120]
        mom120ex10[120:] = np.divide(
            close[110:-10], denom, out=np.full(n - 120, np.nan), where=denom > 0,
        ) - 1.0

    # ATR14
    prev_c = np.r_[np.nan, close[:-1]]
    tr1 = high - low
    tr2 = np.abs(high - prev_c)
    tr3 = np.abs(low - prev_c)
    tr = np.fmax(tr1, np.fmax(tr2, tr3))
    atr14 = pd.Series(tr).rolling(14).mean().to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        above_ma20 = (close > ma20).astype(float)
        above_ma20[~np.isfinite(ma20)] = np.nan
        above_ma60 = (close > ma60).astype(float)
        above_ma60[~np.isfinite(ma60)] = np.nan

        ma20_slope5 = np.divide(
            ma20 - ma20_prev5, ma20_prev5,
            out=np.full(n, np.nan), where=np.isfinite(ma20_prev5) & (ma20_prev5 > 0),
        )
        ma60_slope10 = np.divide(
            ma60 - ma60_prev10, ma60_prev10,
            out=np.full(n, np.nan), where=np.isfinite(ma60_prev10) & (ma60_prev10 > 0),
        )
        ma120_slope20 = np.divide(
            ma120 - ma120_prev20, ma120_prev20,
            out=np.full(n, np.nan), where=np.isfinite(ma120_prev20) & (ma120_prev20 > 0),
        )
        dist_ma20 = np.divide(
            close - ma20, ma20,
            out=np.full(n, np.nan), where=np.isfinite(ma20) & (ma20 > 0),
        )
        dist_ma60 = np.divide(
            close - ma60, ma60,
            out=np.full(n, np.nan), where=np.isfinite(ma60) & (ma60 > 0),
        )
        dist_ma120 = np.divide(
            close - ma120, ma120,
            out=np.full(n, np.nan), where=np.isfinite(ma120) & (ma120 > 0),
        )
        span250 = hi250 - lo250
        pos250 = np.divide(
            close - lo250, span250,
            out=np.full(n, 0.5), where=np.isfinite(span250) & (span250 > 1e-6),
        )
        day_intra = np.divide(
            close - open_, open_,
            out=np.full(n, np.nan), where=np.isfinite(open_) & (open_ > 0),
        )
        atr_pct = np.divide(
            atr14, close,
            out=np.full(n, np.nan), where=np.isfinite(close) & (close > 0),
        )

        # ---- 高阶微观结构因子 ----
        # 1) Amihud 非流动性: log1p(20日平均 |日收益| / 成交额(亿元))
        amt_yi = (vol * close) / 1e8
        illiq_day = np.divide(
            np.abs(ret), amt_yi,
            out=np.full(n, np.nan), where=np.isfinite(amt_yi) & (amt_yi > 1e-4),
        )
        amihud20 = np.log1p(pd.Series(illiq_day).rolling(20, min_periods=10).mean().clip(lower=0).to_numpy())

        # 2) 下行波动 vs 上行波动不对称性: 下行平方收益均值 / 上行平方收益均值
        up_sq = np.where(ret > 0, ret * ret, 0.0)
        dn_sq = np.where(ret < 0, ret * ret, 0.0)
        up_m = pd.Series(up_sq).rolling(20, min_periods=10).mean().to_numpy()
        dn_m = pd.Series(dn_sq).rolling(20, min_periods=10).mean().to_numpy()
        down_up_vol20 = np.sqrt(np.divide(dn_m, up_m + 1e-9, out=np.ones(n), where=(up_m + dn_m) > 0))

        # 3) K线上下影线净承接比: 10日均值 ((下影线 - 上影线) / (high - low))
        hl_span = high - low
        body_lo = np.minimum(open_, close)
        body_hi = np.maximum(open_, close)
        sh_diff = np.divide(
            (body_lo - low) - (high - body_hi), hl_span,
            out=np.zeros(n), where=np.isfinite(hl_span) & (hl_span > 1e-6),
        )
        shadow_net10 = pd.Series(sh_diff).rolling(10, min_periods=5).mean().to_numpy()

        # 4) 20日量价相关性 Corr(close, volume, 20): 向量化协方差计算
        v_ser = pd.Series(vol)
        c_mean = c_ser.rolling(20).mean()
        v_mean = v_ser.rolling(20).mean()
        cov_cv = (c_ser * v_ser).rolling(20).mean() - c_mean * v_mean
        denom_cv = (c_ser.rolling(20).std(ddof=0) * v_ser.rolling(20).std(ddof=0)).to_numpy()
        pv_corr20 = np.divide(
            cov_cv.to_numpy(), denom_cv,
            out=np.zeros(n), where=np.isfinite(denom_cv) & (denom_cv > 1e-9),
        )
        pv_corr20 = np.clip(pv_corr20, -1.0, 1.0)

        # 5) 历史时点基本面 (Point-in-Time ROE & 净利同比，严格按 notice_date <= date 对齐，零未来函数)
        pit_roe = np.full(n, np.nan)
        pit_profit_yoy = np.full(n, np.nan)
        pit_entry = _get_pit_cache().get(str(item["code"]).zfill(6))
        if pit_entry is not None:
            nds, roes, pyoys = pit_entry
            if len(nds) > 0:
                pos_idx = np.searchsorted(nds, item["date"].astype(str), side="right") - 1
                valid_m = pos_idx >= 0
                if np.any(valid_m):
                    pit_roe[valid_m] = roes[pos_idx[valid_m]]
                    pit_profit_yoy[valid_m] = pyoys[pos_idx[valid_m]]

    ext = {
        **feat,
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60,
        "ma120": ma120,
        "ma250": ma250,
        "ma20_slope5": ma20_slope5,
        "ma60_slope10": ma60_slope10,
        "ma120_slope20": ma120_slope20,
        "dist_ma20": dist_ma20,
        "dist_ma60": dist_ma60,
        "dist_ma120": dist_ma120,
        "pos250": pos250,
        "mom120ex10": mom120ex10,
        "vol60": vol60,
        "high20_prev": high20_prev,
        "day_intra": day_intra,
        "atr14": atr14,
        "atr_pct": atr_pct,
        "above_ma20": above_ma20,
        "above_ma60": above_ma60,
        "amihud20": amihud20,
        "down_up_vol20": down_up_vol20,
        "shadow_net10": shadow_net10,
        "pv_corr20": pv_corr20,
        "pit_roe": pit_roe,
        "pit_profit_yoy": pit_profit_yoy,
    }
    item["_cockpit_ext"] = ext
    return ext


def _compute_triple_barrier_for_item(item: dict, idx_arr: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算一只股票在给定信号索引 idx_arr 下的【三重屏障 (Triple-Barrier)】路径结果：
      - entry = i + 1 开盘买入 (entry_open)
      - 在 t = i + 1 .. i + 1 + H 期间，逐日检查最高价/最低价是否触及：
          止损下轨 sl_price = entry_open - sl_mult * atr14[i]
          止盈上轨 tp_price = entry_open + tp_mult * atr14[i]
      - 保守原则：若同一天既破止损又触止盈，按先触止损处理；若开盘直接跳空越过止损/止盈，按当日开盘价结算；
      - 若 H 天内均未触发，则在 i + 1 + H 开盘价平仓。
    返回: (tb_gross_ret, tb_outcome, tb_hold_days)
      tb_outcome: 1=止盈出局, -1=止损出局, 0=到期平仓
    """
    m = len(idx_arr)
    if m == 0:
        return np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    ext = _extended_item_arrays(item)
    open_ = item["open"]
    high = item["high"]
    low = item["low"]
    close = item["close"]
    atr = ext["atr14"]

    sl_mult, tp_mult, _ = TB_PARAMS.get(int(horizon), (1.5, 2.5, 0.01))
    entry = idx_arr + 1
    entry_open = open_[entry]
    atr_sig = np.where(np.isfinite(atr[idx_arr]) & (atr[idx_arr] > 0), atr[idx_arr], close[idx_arr] * 0.025)

    sl_price = np.maximum(0.01, entry_open - sl_mult * atr_sig)
    tp_price = entry_open + tp_mult * atr_sig

    active = np.isfinite(entry_open) & (entry_open > 0)
    exit_price = np.full(m, np.nan)
    outcome = np.zeros(m, dtype=int)
    hold_days = np.full(m, int(horizon), dtype=int)

    H = int(horizon)
    for step in range(1, H + 1):
        if not np.any(active):
            break
        bar_idx = entry + step
        o_s = open_[bar_idx]
        h_s = high[bar_idx]
        l_s = low[bar_idx]

        # 先判止损（保守：跳空低开破止损按开盘价出，盘中破止损按止损价出）
        hit_sl = active & np.isfinite(l_s) & (l_s <= sl_price)
        if np.any(hit_sl):
            exit_price[hit_sl] = np.minimum(o_s[hit_sl], sl_price[hit_sl])
            outcome[hit_sl] = -1
            hold_days[hit_sl] = step
            active[hit_sl] = False

        # 再判止盈（跳空高开超止盈按开盘价出，盘中触及按止盈价出）
        hit_tp = active & np.isfinite(h_s) & (h_s >= tp_price)
        if np.any(hit_tp):
            exit_price[hit_tp] = np.maximum(o_s[hit_tp], tp_price[hit_tp])
            outcome[hit_tp] = 1
            hold_days[hit_tp] = step
            active[hit_tp] = False

    # 未触及止盈止损的，在 i + 1 + H 开盘平仓
    still = active
    if np.any(still):
        exit_price[still] = open_[entry[still] + H]
        outcome[still] = 0
        hold_days[still] = H

    with np.errstate(invalid="ignore", divide="ignore"):
        tb_gross = exit_price / entry_open - 1.0
    return tb_gross, outcome, hold_days


def compute_market_regimes(prepared: list[dict]) -> pd.DataFrame:
    """
    计算每个交易日的全市场广度与指数趋势，并同时产出：
      - regime_short: 短波段(10日)市场环境闸门
      - regime_long:  中长线(60日/120日)市场环境闸门
    """
    date_buckets: dict[str, list[tuple[float, float, float, float]]] = {}
    for item in prepared:
        ext = _extended_item_arrays(item)
        dates = item["date"]
        ab20 = ext["above_ma20"]
        ab60 = ext["above_ma60"]
        r1 = ext["rev1"]
        r5 = ext["rev5"]
        for i in range(proto.WARMUP - 1, len(dates)):
            d = str(dates[i])
            date_buckets.setdefault(d, []).append((ab20[i], ab60[i], r1[i], r5[i]))

    idx_df = db.load_index_kline("sh000001")
    idx_map = {}
    if idx_df is not None and not idx_df.empty:
        idx_df = idx_df.sort_values("date").reset_index(drop=True)
        ic = pd.to_numeric(idx_df["close"], errors="coerce")
        ima20 = ic.rolling(20).mean()
        ima60 = ic.rolling(60).mean()
        ima20_prev5 = ima20.shift(5)
        islope = (ima20 - ima20_prev5) / ima20_prev5
        for d, c_val, m20_val, m60_val, s_val in zip(
            idx_df["date"].astype(str), ic, ima20, ima60, islope
        ):
            if np.isfinite(c_val) and np.isfinite(m20_val) and m20_val > 0:
                idx_map[d] = (
                    float(c_val),
                    float(m20_val),
                    float(m60_val) if np.isfinite(m60_val) else float(m20_val),
                    float(s_val) if np.isfinite(s_val) else 0.0,
                )

    rows = []
    for d in sorted(date_buckets.keys()):
        arr = np.asarray(date_buckets[d], dtype=float)
        n_act = len(arr)
        b20 = float(np.nanmean(arr[:, 0])) if n_act else 0.5
        b60 = float(np.nanmean(arr[:, 1])) if n_act else 0.5
        u1 = float(np.nanmean(arr[:, 2] > 0)) if n_act else 0.5
        u5 = float(np.nanmean(arr[:, 3] > 0)) if n_act else 0.5

        if d in idx_map:
            ic, im20, im60, islope = idx_map[d]
            idx_ok = bool(ic > im20 and islope >= -0.003)
            idx_ok60 = bool(ic >= im60 * 0.99)
        else:
            ic, im20, im60, islope = np.nan, np.nan, np.nan, np.nan
            idx_ok = bool(b20 >= 0.50)
            idx_ok60 = bool(b60 >= 0.45)

        # ---- 1) 短波段 (10日) 市场闸门：严禁在单边普跌日左侧盲目接飞刀 ----
        # a) 冰点右侧反转确认：前期已跌入冰点(b20 <= 26%) 且 当日出现超42%个股翻红反攻(止跌确认)
        is_washout_confirmed = (b20 <= 0.26) and (u1 >= 0.42)
        # b) 多头主升共振：指数在 MA20 之上 且 广度健康 且 当日无恐慌普跌(u1 >= 38%)
        is_bull_impulse = (
            (idx_ok and (0.45 <= b20 <= 0.78) and u5 >= 0.46 and u1 >= 0.38)
            or (b20 > 0.78 and u1 >= 0.50)
        )
        # c) 恐慌宣泄/单边普跌(如全场仅10%~25%个股上涨，严禁接飞刀) 或 破位阴跌中段 或 高位滞涨派发
        is_panic_freefall = (not idx_ok) and (u1 < 0.25)
        is_mid_decline = (not idx_ok) and (0.26 < b20 < 0.62) and (u5 < 0.45)
        is_overheated_stall = (b20 > 0.70) and (u1 < 0.40)

        if is_washout_confirmed or is_bull_impulse:
            reg_short = REGIME_BULL
        elif is_panic_freefall or is_mid_decline or is_overheated_stall:
            reg_short = REGIME_BEAR
        else:
            reg_short = REGIME_NEUTRAL

        # ---- 2) 中长线 (60日/120日) 市场周期与位置闸门 ----
        # 长线虽看重低位赔率，但若当日正在发生极端恐慌踩踏(u1 < 22% 且指数破位)，降为黄灯分批观察，不亮满仓绿灯
        if (b60 <= 0.42 or (idx_ok60 and b60 <= 0.52)) and (u1 >= 0.25 or idx_ok60):
            reg_long = REGIME_BULL
        elif b60 > 0.62:
            reg_long = REGIME_BEAR
        else:
            reg_long = REGIME_NEUTRAL

        rows.append({
            "date": d,
            "n_active": n_act,
            "breadth_ma20": round(b20, 4),
            "breadth_ma60": round(b60, 4),
            "breadth_up1": round(u1, 4),
            "breadth_up5": round(u5, 4),
            "idx_close": None if not np.isfinite(ic) else round(ic, 2),
            "idx_ma20": None if not np.isfinite(im20) else round(im20, 2),
            "idx_ma60": None if not np.isfinite(im60) else round(im60, 2),
            "idx_slope5": None if not np.isfinite(islope) else round(islope, 4),
            "idx_above_ma20": idx_ok,
            "regime": reg_short,
            "regime_short": reg_short,
            "regime_long": reg_long,
            "regime_cn": REGIME_CN_SHORT[reg_short],
            "pos_cap": REGIME_POS_CAP[reg_short],
        })
    return pd.DataFrame(rows)


def _attach_sector_spillover(df: pd.DataFrame) -> pd.DataFrame:
    """
    在单日或多日截面上计算板块动量溢出 (sec_mom5) 与个股行业内超额 (alpha_ind5)。
    优先使用东财二级细分题材板块 (subsector_map，如半导体、白酒Ⅱ、消费电子等 130+ 细分领域) 聚合，
    比宽泛的一级行业更能捕捉真实题材主线联动。
    """
    if df is None or df.empty or "rev5" not in df.columns:
        return df
    out = df.copy()
    sub_map = db.load_subsector_map()
    if sub_map:
        grp_col = out["code"].astype(str).str.zfill(6).map(lambda c: sub_map.get(c) or "")
        grp_col = np.where(grp_col != "", grp_col, out["industry"].astype(str))
        out["_subsec"] = grp_col
    else:
        out["_subsec"] = out["industry"].astype(str)
    sec_m = out.groupby(["date", "_subsec"], sort=False)["rev5"].transform("mean")
    out["sec_mom5"] = sec_m.fillna(0.0)
    out["alpha_ind5"] = (out["rev5"] - out["sec_mom5"]).fillna(0.0)
    return out.drop(columns=["_subsec"])


def _build_rich_panel(
    horizon: int,
    prepared: list[dict],
    regime_map: dict[str, str],
    step_days: Optional[int] = None,
) -> pd.DataFrame:
    """构建指定持有期 horizon 的截面面板（含高阶微观结构因子与三重屏障路径标签）。"""
    step = int(step_days) if step_days is not None else (10 if int(horizon) <= 20 else 20)
    all_dates = sorted({str(d) for item in prepared for d in item["date"][proto.WARMUP - 1 : -1 - int(horizon)]})
    keep = set(all_dates[::step])
    frames = []
    item_feat_cols = tuple(
        f for f in dict.fromkeys(SHORT_ADV_FEATURES + LONG_FEATURES)
        if f != "alpha_ind5"
    )
    extra_cols = (
        "ma5", "ma20", "ma60", "ma120", "ma250",
        "ma20_slope5", "ma60_slope10", "ma120_slope20",
        "dist_ma20", "dist_ma60", "dist_ma120",
        "high20_prev", "day_intra", "atr14", "atr_pct",
    )
    for item in prepared:
        if "high" not in item:
            continue
        sig = pnl._signals_from_prepared(item, int(horizon), proto.WARMUP)
        if sig is None or sig.empty:
            continue
        sig = sig.loc[sig["date"].astype(str).isin(keep)].copy()
        if sig.empty:
            continue
        ext = _extended_item_arrays(item)
        loc = {str(d): i for i, d in enumerate(item["date"])}
        idx = np.asarray([loc[str(d)] for d in sig["date"]], dtype=int)
        sig["close"] = item["close"][idx]
        for name in item_feat_cols:
            sig[name] = ext[name][idx]
        for name in extra_cols:
            sig[name] = ext[name][idx]

        # 计算三重屏障 (Triple-Barrier) 实战路径收益与结局
        tb_gross, tb_out, tb_hd = _compute_triple_barrier_for_item(item, idx, int(horizon))
        ok_mask = (sig["block"].to_numpy() == "ok")
        sig["tb_ret"] = np.where(ok_mask, tb_gross, np.nan)
        sig["tb_outcome"] = np.where(ok_mask, tb_out, 0)
        sig["tb_hold"] = np.where(ok_mask, tb_hd, int(horizon))
        frames.append(sig)

    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = _attach_sector_spillover(raw)
    all_feat_cols = tuple(dict.fromkeys(SHORT_ADV_FEATURES + LONG_FEATURES))
    raw = proto.attach_processed(raw, all_feat_cols)
    raw["regime"] = raw["date"].astype(str).map(lambda d: regime_map.get(d, REGIME_NEUTRAL))

    # 计算固定持有期净收益 net_ret 与三重屏障实战净收益 tb_net_ret
    filled_mask = (raw["block"] == "ok") & raw["fwd_ret"].notna()
    net_arr = np.full(len(raw), np.nan)
    tb_net_arr = np.full(len(raw), np.nan)
    if filled_mask.any():
        sub = raw.loc[filled_mask]
        stamp = np.where(sub["exit_date"].astype(str) < proto.STAMP_DATE, proto.STAMP_BEFORE, proto.STAMP_AFTER)
        gross = sub["fwd_ret"].to_numpy(dtype=float)
        tb_g = sub["tb_ret"].to_numpy(dtype=float)
        net_vals = (1.0 + gross) * (1.0 - proto.COMMISSION - stamp) / (1.0 + proto.COMMISSION) - 1.0
        tb_net_vals = (1.0 + tb_g) * (1.0 - proto.COMMISSION - stamp) / (1.0 + proto.COMMISSION) - 1.0
        net_arr[filled_mask.to_numpy()] = net_vals
        tb_net_arr[filled_mask.to_numpy()] = tb_net_vals
    raw["net_ret"] = net_arr
    raw["tb_net_ret"] = tb_net_arr

    # 行业中性超额标签：跑赢当日同行业平均毛收益
    ind_day_mean = raw.groupby(["date", "industry"], sort=False)["fwd_ret"].transform("mean")
    raw["excess_ind"] = raw["fwd_ret"] - ind_day_mean
    return raw


def _fit_tb_and_excess_model(train_df: pd.DataFrame, features: tuple[str, ...], lam: float = 0.25) -> np.ndarray:
    """
    在训练窗 (exit_date <= 2023-12-31) 拟合融合【三重屏障实战胜率】与【行业超额胜率】的高阶逻辑回归系数。
    """
    filled = train_df.loc[(train_df["block"] == "ok") & train_df["tb_net_ret"].notna() & train_df["fwd_ret"].notna()]
    if len(filled) < 300:
        return np.zeros(1 + len(features))
    Z = np.zeros((len(filled), len(features)), dtype=float)
    for j, f in enumerate(features):
        col = filled[f + "_z"].to_numpy(dtype=float)
        Z[:, j] = np.where(np.isfinite(col), col, 0.0)
    X = np.column_stack([np.ones(len(filled)), Z])
    # 实战复合正标签：三重屏障净收益 > 0 (即止盈或到期净赚)
    y_tb = (filled["tb_net_ret"].to_numpy(dtype=float) > 0).astype(float)
    return wm.fit_logit(X, y_tb, lam=lam)


def _assign_deciles(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(5, index=df.index, dtype=int)
    for _, idx in df.groupby("date", sort=False).groups.items():
        sub = df.loc[idx, "p"]
        valid = sub.dropna()
        if len(valid) < 20:
            continue
        ranks = valid.rank(method="first", pct=True)
        dec = np.clip(np.ceil(ranks * 10.0), 1, 10).astype(int)
        out.loc[valid.index] = dec
    return out


def calibrate_conditional_cells(train_df: pd.DataFrame) -> dict:
    """在训练窗 (exit_date <= 2023-12-31) 校准 (regime, decile) 固定持有期胜率、三重屏障实战胜率与盈亏比。"""
    filled = train_df.loc[(train_df["block"] == "ok") & train_df["fwd_ret"].notna() & train_df["tb_net_ret"].notna()].copy()
    if filled.empty:
        return {"regimes": {}, "cells": {}}

    filled["decile"] = _assign_deciles(filled)

    g_win_gross = float((filled["fwd_ret"] > 0).mean())
    g_win_net = float((filled["net_ret"] > 0).mean())
    g_win_tb = float((filled["tb_net_ret"] > 0).mean())
    g_net_mean = float(filled["net_ret"].mean())
    g_tb_mean = float(filled["tb_net_ret"].mean())
    pos_all = filled.loc[filled["tb_net_ret"] > 0, "tb_net_ret"]
    neg_all = filled.loc[filled["tb_net_ret"] <= 0, "tb_net_ret"]
    g_avg_win = float(pos_all.mean()) if len(pos_all) else 0.06
    g_avg_loss = float(abs(neg_all.mean())) if len(neg_all) else 0.04

    regime_stats = {}
    for reg in (REGIME_BULL, REGIME_NEUTRAL, REGIME_BEAR):
        sub = filled.loc[filled["regime"] == reg]
        if len(sub) < 50:
            regime_stats[reg] = {
                "n_dates": 0, "n_trades": len(sub),
                "win_gross": g_win_gross, "win_net": g_win_net, "win_tb": g_win_tb,
                "mean_net": g_net_mean, "mean_tb_net": g_tb_mean,
                "avg_win": g_avg_win, "avg_loss": g_avg_loss,
                "payoff": g_avg_win / max(g_avg_loss, 1e-4),
            }
            continue
        day_wg = sub.groupby("date")["fwd_ret"].apply(lambda s: float((s > 0).mean()))
        day_wn = sub.groupby("date")["net_ret"].apply(lambda s: float((s > 0).mean()))
        day_wtb = sub.groupby("date")["tb_net_ret"].apply(lambda s: float((s > 0).mean()))
        day_mn = sub.groupby("date")["net_ret"].mean()
        day_mtb = sub.groupby("date")["tb_net_ret"].mean()
        r_pos = sub.loc[sub["tb_net_ret"] > 0, "tb_net_ret"]
        r_neg = sub.loc[sub["tb_net_ret"] <= 0, "tb_net_ret"]
        aw = float(r_pos.mean()) if len(r_pos) else g_avg_win
        al = float(abs(r_neg.mean())) if len(r_neg) else g_avg_loss
        regime_stats[reg] = {
            "n_dates": int(sub["date"].nunique()),
            "n_trades": int(len(sub)),
            "win_gross": float(day_wg.mean()),
            "win_net": float(day_wn.mean()),
            "win_tb": float(day_wtb.mean()),
            "mean_net": float(day_mn.mean()),
            "mean_tb_net": float(day_mtb.mean()),
            "avg_win": aw,
            "avg_loss": al,
            "payoff": aw / max(al, 1e-4),
        }

    cells = {}
    prior_n = 200.0
    for reg in (REGIME_BULL, REGIME_NEUTRAL, REGIME_BEAR):
        base = regime_stats[reg]
        for dec in range(1, 11):
            key = f"{reg}_{dec}"
            sub = filled.loc[(filled["regime"] == reg) & (filled["decile"] == dec)]
            n = len(sub)
            if n == 0:
                cells[key] = {
                    "n": 0,
                    "p_gross": base["win_gross"],
                    "p_net": base["win_net"],
                    "p_tb": base["win_tb"],
                    "mean_net": base["mean_net"],
                    "mean_tb_net": base["mean_tb_net"],
                    "avg_win": base["avg_win"],
                    "avg_loss": base["avg_loss"],
                    "payoff": base["payoff"],
                    "cvar5": -base["avg_loss"] * 2.0,
                }
                continue
            w = n / (n + prior_n)
            raw_wg = float(sub.groupby("date")["fwd_ret"].apply(lambda s: float((s > 0).mean())).mean())
            raw_wn = float(sub.groupby("date")["net_ret"].apply(lambda s: float((s > 0).mean())).mean())
            raw_wtb = float(sub.groupby("date")["tb_net_ret"].apply(lambda s: float((s > 0).mean())).mean())
            raw_mn = float(sub.groupby("date")["net_ret"].mean().mean())
            raw_mtb = float(sub.groupby("date")["tb_net_ret"].mean().mean())
            s_pos = sub.loc[sub["tb_net_ret"] > 0, "tb_net_ret"]
            s_neg = sub.loc[sub["tb_net_ret"] <= 0, "tb_net_ret"]
            aw = float(s_pos.mean()) if len(s_pos) >= 20 else base["avg_win"]
            al = float(abs(s_neg.mean())) if len(s_neg) >= 20 else base["avg_loss"]
            cvar5 = float(np.nanpercentile(sub["tb_net_ret"].to_numpy(dtype=float), 5)) if n >= 20 else -al * 2.0
            cells[key] = {
                "n": int(n),
                "p_gross": round(w * raw_wg + (1.0 - w) * base["win_gross"], 4),
                "p_net": round(w * raw_wn + (1.0 - w) * base["win_net"], 4),
                "p_tb": round(w * raw_wtb + (1.0 - w) * base["win_tb"], 4),
                "mean_net": round(w * raw_mn + (1.0 - w) * base["mean_net"], 4),
                "mean_tb_net": round(w * raw_mtb + (1.0 - w) * base["mean_tb_net"], 4),
                "avg_win": round(aw, 4),
                "avg_loss": round(al, 4),
                "payoff": round(aw / max(al, 1e-4), 2),
                "cvar5": round(cvar5, 4),
            }

    return {"regimes": regime_stats, "cells": cells}


# ============================================================================
# 短波段 (10日) 与 中长线 (60日/120日) 策略矩阵（融入高阶微观结构与板块溢出）
# ============================================================================
SHORT_STRATEGY_SPECS = [
    {
        "key": "quiet_pullback",
        "name": "低波缩量企稳",
        "desc": "振幅与波动处于低位(z<-0.4) + 60日结构完好 + 近20日温和回调(-15%~-2%) + 下影承接良好",
        "top_k": 12,
        "allow_neutral": False,
    },
    {
        "key": "sector_spillover_dip",
        "name": "主线板块低吸共振",
        "desc": "所属行业5日动量处于全市场前35%(板块走强) + 个股短期滞涨回调(alpha_ind5<0) + 低波缩量承接",
        "top_k": 12,
        "allow_neutral": False,
    },
    {
        "key": "logit_top",
        "name": "高阶概率精选",
        "desc": "15维高阶微观结构与三重屏障模型概率最高前15只 + 剔除放量滞涨与单日过热股",
        "top_k": 15,
        "allow_neutral": False,
    },
    {
        "key": "oversold_rebound",
        "name": "恐慌极值反转",
        "desc": "近20日深跌超12%且近5日急跌 + 低波标的当日收阳下影承接止跌，博取极值均值回归",
        "top_k": 12,
        "allow_neutral": False,
    },
    {
        "key": "trend_pullback",
        "name": "多头缩量回踩",
        "desc": "中期均线多头(MA20>MA60且MA20向上) + 低波标的近5日缩量回踩MA20支撑位(±2.2%)",
        "top_k": 12,
        "allow_neutral": False,
    },
    {
        "key": "high_mom_chase",
        "name": "传统追高动量(对照)",
        "desc": "均线多头 + 近20日大涨(>8%) + 放量(>1.3倍)追涨，用作检验散户追高胜率的反面对照组",
        "top_k": 12,
        "allow_neutral": False,
    },
]

LONG_STRATEGY_SPECS = [
    {
        "key": "pit_quality_compounder",
        "name": "长线·高ROE慢牛回踩",
        "desc": "历史时点季报ROE>=6%(真实盈利护城河) + 长期均线多头(MA60>MA120) + 缩量回踩季线低波企稳",
        "top_k": 15,
        "allow_neutral": True,
    },
    {
        "key": "trend_compounder_dip",
        "name": "长线·慢牛季线回踩",
        "desc": "长期上升趋势(MA60>MA120且半年线向上) + 股价缩量回踩MA60季线附近(-3.5%~+5%) + 低波稳健",
        "top_k": 15,
        "allow_neutral": True,
    },
    {
        "key": "deep_value_turnaround",
        "name": "长线·底部右侧拐点",
        "desc": "一年价格分位<=35%(跌透) + 历史时点ROE>0(非亏损股) + 长期波动收敛 + 股价站上MA20且季线企稳",
        "top_k": 15,
        "allow_neutral": True,
    },
    {
        "key": "multiyear_base_lock",
        "name": "长线·大底筹码锁定",
        "desc": "一年价格分位<=30% + 振幅与60日波动极度收敛(z<=-0.35) + 下影承接良好且站上MA20",
        "top_k": 15,
        "allow_neutral": True,
    },
    {
        "key": "long_logit_top",
        "name": "长线·高阶概率精选",
        "desc": "16维(含PIT季报ROE/增速)长周期模型概率最高前15只 + 剔除一年高位(>60%分位)与高波动透支股",
        "top_k": 15,
        "allow_neutral": True,
    },
    {
        "key": "high_flyer_chase",
        "name": "长线·高位热门追涨(对照)",
        "desc": "一年价格分位>80% + 半年大涨超25% + 高波动追高，检验长线买在山顶的胜率与回撤",
        "top_k": 15,
        "allow_neutral": False,
    },
]


def get_strategy_specs(horizon: int) -> list[dict]:
    return LONG_STRATEGY_SPECS if int(horizon) >= 60 else SHORT_STRATEGY_SPECS


def select_strategy_candidates(day_df: pd.DataFrame, strat_key: str, top_k: int = 12) -> pd.DataFrame:
    """在单日截面 day_df 上按指定策略筛选并排序前 top_k 只标的。"""
    if day_df is None or day_df.empty:
        return pd.DataFrame()

    df = day_df
    v_z = df["vol20_z"].to_numpy(dtype=float)
    r_z = df["range20_z"].to_numpy(dtype=float)
    hd_z = df["highdist_z"].to_numpy(dtype=float)
    m20 = df["mom20"].to_numpy(dtype=float)
    r5 = df["rev5"].to_numpy(dtype=float)
    r1 = df["rev1"].to_numpy(dtype=float)
    vr = df["volratio"].to_numpy(dtype=float)
    intra = df["day_intra"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    ma5 = df["ma5"].to_numpy(dtype=float)
    ma20 = df["ma20"].to_numpy(dtype=float)
    ma60 = df["ma60"].to_numpy(dtype=float)
    slope5 = df["ma20_slope5"].to_numpy(dtype=float)
    dist20 = df["dist_ma20"].to_numpy(dtype=float)
    hi20 = df["high20_prev"].to_numpy(dtype=float)
    p_up = df["p"].to_numpy(dtype=float) if "p" in df.columns else np.full(len(df), 0.5)
    p_tb = df["p_tb"].to_numpy(dtype=float) if "p_tb" in df.columns else p_up
    p_blend = 0.5 * p_up + 0.5 * p_tb

    sh_z = df["shadow_net10_z"].to_numpy(dtype=float) if "shadow_net10_z" in df.columns else np.zeros(len(df))
    du_z = df["down_up_vol20_z"].to_numpy(dtype=float) if "down_up_vol20_z" in df.columns else np.zeros(len(df))
    sec_m5 = df["sec_mom5"].to_numpy(dtype=float) if "sec_mom5" in df.columns else np.zeros(len(df))
    alp_i5 = df["alpha_ind5"].to_numpy(dtype=float) if "alpha_ind5" in df.columns else np.zeros(len(df))

    pos250 = df["pos250"].to_numpy(dtype=float) if "pos250" in df.columns else np.full(len(df), 0.5)
    v60_z = df["vol60_z"].to_numpy(dtype=float) if "vol60_z" in df.columns else v_z
    dist60 = df["dist_ma60"].to_numpy(dtype=float) if "dist_ma60" in df.columns else dist20
    ma120 = df["ma120"].to_numpy(dtype=float) if "ma120" in df.columns else ma60
    slope60 = df["ma60_slope10"].to_numpy(dtype=float) if "ma60_slope10" in df.columns else slope5
    slope120 = df["ma120_slope20"].to_numpy(dtype=float) if "ma120_slope20" in df.columns else slope5
    m120 = df["mom120ex10"].to_numpy(dtype=float) if "mom120ex10" in df.columns else m20

    with np.errstate(invalid="ignore"):
        if strat_key == "quiet_pullback":
            mask = (
                (v_z <= -0.40)
                & (r_z <= -0.40)
                & (hd_z >= -0.50)
                & (m20 >= -0.15) & (m20 <= -0.015)
                & (vr <= 0.92)
                & (intra >= -0.005)
                & (r1 <= 0.04)
            )
            score = p_blend * 100.0 - 0.5 * r_z - 0.5 * v_z + 0.2 * np.where(np.isfinite(sh_z), sh_z, 0.0)
        elif strat_key == "sector_spillover_dip":
            sec_th = float(np.nanpercentile(sec_m5, 65)) if np.isfinite(sec_m5).any() else 0.0
            mask = (
                (sec_m5 >= sec_th)
                & (alp_i5 <= -0.01) & (alp_i5 >= -0.10)
                & (v_z <= -0.25)
                & (r_z <= -0.25)
                & (close >= ma60 * 0.96)
                & (r1 >= -0.02) & (r1 <= 0.035)
            )
            score = p_blend * 100.0 + sec_m5 * 15.0 - 0.4 * r_z - 0.4 * v_z
        elif strat_key == "trend_pullback":
            mask = (
                (ma20 > ma60)
                & (slope5 > 0.0)
                & (close >= ma60)
                & (np.abs(dist20) <= 0.022)
                & (r5 <= -0.01)
                & (vr <= 0.95)
                & (r_z <= -0.20)
                & (v_z <= -0.20)
            )
            score = p_blend * 100.0 - r_z - v_z
        elif strat_key == "oversold_rebound":
            mask = (
                (m20 <= -0.12)
                & (r5 <= -0.04)
                & (intra >= 0.005)
                & (v_z <= 0.30)
                & (r_z <= 0.30)
                & (r1 <= 0.05)
            )
            score = p_blend * 100.0 + (-m20) * 5.0 - 0.5 * r_z
        elif strat_key == "logit_top":
            mask = (
                np.isfinite(p_blend)
                & (vr <= 1.10)
                & (r1 >= -0.025) & (r1 <= 0.04)
                & (v_z <= -0.30)
                & (r_z <= -0.30)
            )
            score = p_blend * 100.0
        elif strat_key == "high_mom_chase":
            mask = (
                (ma5 > ma20) & (ma20 > ma60)
                & (m20 >= 0.08)
                & (vr >= 1.3)
            )
            score = m20 * 10.0 + vr

        # ---- 中长线 (60日 / 120日) 策略 ----
        elif strat_key == "pit_quality_compounder":
            pit_roe = df["pit_roe"].to_numpy(dtype=float) if "pit_roe" in df.columns else np.full(len(df), 8.0)
            pit_roe_z = df["pit_roe_z"].to_numpy(dtype=float) if "pit_roe_z" in df.columns else np.zeros(len(df))
            mask = (
                (np.where(np.isfinite(pit_roe), pit_roe, 6.0) >= 6.0)
                & (ma60 > ma120)
                & (slope60 >= 0.0)
                & (dist60 >= -0.04) & (dist60 <= 0.05)
                & (pos250 <= 0.70)
                & (v60_z <= 0.0)
                & (r_z <= 0.0)
            )
            score = p_blend * 100.0 + 0.5 * np.where(np.isfinite(pit_roe_z), pit_roe_z, 0.0) - v60_z - r_z
        elif strat_key == "deep_value_turnaround":
            pit_roe = df["pit_roe"].to_numpy(dtype=float) if "pit_roe" in df.columns else np.full(len(df), 3.0)
            mask = (
                (pos250 <= 0.35)
                & (np.where(np.isfinite(pit_roe), pit_roe, 1.0) > 0.0)
                & (v60_z <= -0.20)
                & (r_z <= -0.20)
                & (close >= ma20)
                & (slope5 >= 0.0)
                & (dist60 >= -0.02) & (dist60 <= 0.08)
            )
            score = p_blend * 100.0 - 0.5 * v60_z - 0.5 * r_z + (0.35 - pos250) * 10.0
        elif strat_key == "trend_compounder_dip":
            mask = (
                (ma60 > ma120)
                & (slope60 >= 0.0)
                & (slope120 >= -0.005)
                & (dist60 >= -0.035) & (dist60 <= 0.05)
                & (pos250 <= 0.72)
                & (v60_z <= 0.0)
                & (r_z <= 0.0)
            )
            score = p_blend * 100.0 - v60_z - r_z - np.abs(dist60) * 10.0
        elif strat_key == "multiyear_base_lock":
            mask = (
                (pos250 <= 0.30)
                & (v60_z <= -0.35)
                & (r_z <= -0.35)
                & (vr <= 1.05)
                & (close >= ma20 * 0.99)
            )
            score = p_blend * 100.0 - v60_z - r_z + 0.2 * np.where(np.isfinite(sh_z), sh_z, 0.0)
        elif strat_key == "long_logit_top":
            mask = (
                np.isfinite(p_blend)
                & (pos250 <= 0.60)
                & (v60_z <= -0.25)
                & (r_z <= -0.25)
                & (close >= ma20 * 0.98)
            )
            score = p_blend * 100.0
        elif strat_key == "high_flyer_chase":
            mask = (
                (pos250 >= 0.80)
                & (m120 >= 0.25)
                & (v60_z >= 0.30)
            )
            score = pos250 * 20.0 + m120 * 10.0
        else:
            return pd.DataFrame()

    mask = mask & np.isfinite(score)
    if not np.any(mask):
        return pd.DataFrame()
    sub = df.loc[mask].copy()
    sub["_strat_score"] = score[mask]
    sub = sub.sort_values(["_strat_score", "code"], ascending=[False, True], kind="mergesort")
    return sub.iloc[:top_k]


def _eval_strategy_slice(df: pd.DataFrame, spec: dict, gated: bool) -> dict:
    """在给定样本切片 (train 或 test) 上评测单个策略（同时产出固定期指标与三重屏障实战指标）。"""
    empty_res = {
        "n_days": 0, "n_trades": 0,
        "win_gross": np.nan, "win_net": np.nan, "win_tb": np.nan, "mkt_win": np.nan,
        "lift": np.nan, "lift_t": np.nan,
        "mean_net": np.nan, "net_t": np.nan,
        "mean_tb_net": np.nan, "tb_net_t": np.nan,
        "excess_net": np.nan, "excess_t": np.nan,
        "payoff": np.nan, "tb_payoff": np.nan,
        "limit_up": 0, "limit_down": 0,
    }
    if df is None or df.empty:
        return empty_res

    key = spec["key"]
    top_k = spec["top_k"]
    allow_neutral = spec["allow_neutral"]

    day_win_gross = []
    day_win_net = []
    day_win_tb = []
    day_mkt_win = []
    day_lift = []
    day_net_ret = []
    day_tb_net_ret = []
    day_excess_net = []
    all_wins = []
    all_losses = []
    tb_wins = []
    tb_losses = []
    n_trades = 0
    n_lim_up = 0
    n_lim_dn = 0

    for d, g in df.groupby("date", sort=True):
        reg = str(g["regime"].iloc[0])
        if gated:
            if reg == REGIME_BEAR:
                continue
            if reg == REGIME_NEUTRAL and not allow_neutral:
                continue

        filled_all = g.loc[(g["block"] == "ok") & g["fwd_ret"].notna()]
        if len(filled_all) < proto.MIN_NAMES:
            continue

        picks = select_strategy_candidates(g, key, top_k=top_k)
        if picks.empty:
            continue

        n_lim_up += int((picks["block"] == "limit_up").sum())
        n_lim_dn += int((picks["block"] == "limit_down").sum())

        filled_picks = picks.loc[(picks["block"] == "ok") & picks["fwd_ret"].notna() & picks["tb_net_ret"].notna()]
        if len(filled_picks) == 0:
            continue

        n_trades += len(filled_picks)
        wg = float((filled_picks["fwd_ret"] > 0).mean())
        wn = float((filled_picks["net_ret"] > 0).mean())
        wtb = float((filled_picks["tb_net_ret"] > 0).mean())
        mw = float((filled_all["fwd_ret"] > 0).mean())
        mn = float(filled_picks["net_ret"].mean())
        mtb = float(filled_picks["tb_net_ret"].mean())
        mkt_mn = float(filled_all["net_ret"].mean())

        day_win_gross.append(wg)
        day_win_net.append(wn)
        day_win_tb.append(wtb)
        day_mkt_win.append(mw)
        day_lift.append(wg - mw)
        day_net_ret.append(mn)
        day_tb_net_ret.append(mtb)
        day_excess_net.append(mn - mkt_mn)

        nets = filled_picks["net_ret"].to_numpy(dtype=float)
        all_wins.extend(nets[nets > 0].tolist())
        all_losses.extend(np.abs(nets[nets <= 0]).tolist())

        tbn = filled_picks["tb_net_ret"].to_numpy(dtype=float)
        tb_wins.extend(tbn[tbn > 0].tolist())
        tb_losses.extend(np.abs(tbn[tbn <= 0]).tolist())

    if not day_win_gross:
        empty_res["limit_up"] = n_lim_up
        empty_res["limit_down"] = n_lim_dn
        return empty_res

    lift_mu, lift_t, _ = proto.newey_west_mean(np.asarray(day_lift, dtype=float))
    net_mu, net_t, _ = proto.newey_west_mean(np.asarray(day_net_ret, dtype=float))
    tb_mu, tb_t, _ = proto.newey_west_mean(np.asarray(day_tb_net_ret, dtype=float))
    exc_mu, exc_t, _ = proto.newey_west_mean(np.asarray(day_excess_net, dtype=float))
    avg_w = float(np.mean(all_wins)) if all_wins else 0.0
    avg_l = float(np.mean(all_losses)) if all_losses else 0.04
    payoff = avg_w / max(avg_l, 1e-4) if avg_w > 0 else 0.0

    tb_aw = float(np.mean(tb_wins)) if tb_wins else 0.0
    tb_al = float(np.mean(tb_losses)) if tb_losses else 0.03
    tb_payoff = tb_aw / max(tb_al, 1e-4) if tb_aw > 0 else 0.0

    return {
        "n_days": len(day_win_gross),
        "n_trades": int(n_trades),
        "win_gross": round(float(np.mean(day_win_gross)), 4),
        "win_net": round(float(np.mean(day_win_net)), 4),
        "win_tb": round(float(np.mean(day_win_tb)), 4),
        "mkt_win": round(float(np.mean(day_mkt_win)), 4),
        "lift": round(float(lift_mu), 4) if np.isfinite(lift_mu) else None,
        "lift_t": round(float(lift_t), 2) if np.isfinite(lift_t) else None,
        "mean_net": round(float(net_mu), 4) if np.isfinite(net_mu) else None,
        "net_t": round(float(net_t), 2) if np.isfinite(net_t) else None,
        "mean_tb_net": round(float(tb_mu), 4) if np.isfinite(tb_mu) else None,
        "tb_net_t": round(float(tb_t), 2) if np.isfinite(tb_t) else None,
        "excess_net": round(float(exc_mu), 4) if np.isfinite(exc_mu) else None,
        "excess_t": round(float(exc_t), 2) if np.isfinite(exc_t) else None,
        "payoff": round(float(payoff), 2),
        "tb_payoff": round(float(tb_payoff), 2),
        "limit_up": int(n_lim_up),
        "limit_down": int(n_lim_dn),
    }


def _unconditional_benchmark(df: pd.DataFrame) -> tuple[float, float, float]:
    if df is None or df.empty:
        return 0.48, 0.0, 0.48
    filled = df.loc[(df["block"] == "ok") & df["fwd_ret"].notna() & df["tb_net_ret"].notna()]
    if filled.empty:
        return 0.48, 0.0, 0.48
    day_w = filled.groupby("date")["fwd_ret"].apply(lambda s: float((s > 0).mean()))
    day_n = filled.groupby("date")["net_ret"].mean()
    day_tb = filled.groupby("date")["tb_net_ret"].apply(lambda s: float((s > 0).mean()))
    return float(day_w.mean()), float(day_n.mean()), float(day_tb.mean())


def evaluate_all_strategies(train_df: pd.DataFrame, test_df: pd.DataFrame, horizon: int = 10) -> list[dict]:
    """对当前周期的全部候选策略在训练窗与测试窗上分别评测【裸跑】与【开启绿灯闸门】。"""
    tr_base_w, tr_base_n, tr_base_tb = _unconditional_benchmark(train_df)
    te_base_w, te_base_n, te_base_tb = _unconditional_benchmark(test_df)
    specs = get_strategy_specs(horizon)

    min_win = 0.515 if int(horizon) == 60 else (0.54 if int(horizon) >= 120 else 0.52)
    min_net = 0.030 if int(horizon) == 60 else (0.050 if int(horizon) >= 120 else 0.010)

    results = []
    for spec in specs:
        tr_raw = _eval_strategy_slice(train_df, spec, gated=False)
        tr_gate = _eval_strategy_slice(train_df, spec, gated=True)
        te_raw = _eval_strategy_slice(test_df, spec, gated=False)
        te_gate = _eval_strategy_slice(test_df, spec, gated=True)

        for d_stat, bw, bn, btb in (
            (tr_raw, tr_base_w, tr_base_n, tr_base_tb),
            (tr_gate, tr_base_w, tr_base_n, tr_base_tb),
            (te_raw, te_base_w, te_base_n, te_base_tb),
            (te_gate, te_base_w, te_base_n, te_base_tb),
        ):
            d_stat["uncond_mkt_win"] = round(bw, 4)
            d_stat["uncond_mkt_net"] = round(bn, 4)
            d_stat["uncond_mkt_tb_win"] = round(btb, 4)
            wg = d_stat.get("win_gross")
            wtb = d_stat.get("win_tb")
            mn = d_stat.get("mean_net")
            d_stat["total_win_lift"] = round(float(wg - bw), 4) if wg is not None and np.isfinite(wg) else None
            d_stat["total_tb_lift"] = round(float(wtb - btb), 4) if wtb is not None and np.isfinite(wtb) else None
            d_stat["total_net_lift"] = round(float(mn - bn), 4) if mn is not None and np.isfinite(mn) else None

        wg = te_gate.get("win_gross")
        wtb = te_gate.get("win_tb")
        mn = te_gate.get("mean_net")
        mtb = te_gate.get("mean_tb_net")
        pf = te_gate.get("payoff")
        is_control = spec["key"] in ("high_mom_chase", "high_flyer_chase")
        best_win = max(wg or 0.0, wtb or 0.0)
        best_net = max(mn or -1.0, mtb or -1.0)
        verified = bool(
            (not is_control)
            and best_win >= min_win
            and best_net >= min_net
            and pf is not None and np.isfinite(pf) and pf >= 1.30
        )
        results.append({
            "key": spec["key"],
            "name": spec["name"],
            "desc": spec["desc"],
            "allow_neutral": spec["allow_neutral"],
            "verified": verified,
            "train_raw": tr_raw,
            "train_gated": tr_gate,
            "test_raw": te_raw,
            "test_gated": te_gate,
        })
    results.sort(
        key=lambda r: (
            1 if r["verified"] else 0,
            max(r["test_gated"].get("win_tb") or -1.0, r["test_gated"].get("win_gross") or -1.0),
            max(r["test_gated"].get("mean_tb_net") or -1.0, r["test_gated"].get("mean_net") or -1.0),
        ),
        reverse=True,
    )
    for idx, r in enumerate(results):
        r["rank"] = idx + 1
    return results


def build_portfolio_curves_and_tearsheet(
    rich_panel: pd.DataFrame,
    strat_evals: list[dict],
    horizon: int,
) -> dict:
    """
    构建 2021~2026 全时间轴组合净值曲线 (Equity Curve)、水下回撤序列 (Drawdown)、
    夏普比率 (Sharpe)、卡玛比率 (Calmar) 与分年度 (2021..2026) 归因统计表。
    对比三条核心曲线：
      1) gated_top:  开启市场闸门 + 优选策略组合 (实盘遵守四层纪律)
      2) raw_top:    全天候裸跑策略组合 (不看市场红绿灯硬买)
      3) mkt_bench:  全市场等权盲买基准
    """
    if rich_panel is None or rich_panel.empty:
        return {}

    # 选取当前周期排名前 2 的非对照策略作为组合代表
    top_specs = [
        next(sp for sp in get_strategy_specs(horizon) if sp["key"] == r["key"])
        for r in strat_evals[:2]
        if r["key"] not in ("high_mom_chase", "high_flyer_chase")
    ]
    if not top_specs:
        top_specs = get_strategy_specs(horizon)[:1]

    step = 10 if int(horizon) <= 20 else 20
    periods_per_yr = 244.0 / float(step)
    overlap_k = max(1.0, float(horizon) / float(step))

    dates = []
    eq_gated = [1.0]
    eq_raw = [1.0]
    eq_bench = [1.0]
    ret_gated_list = []
    ret_raw_list = []
    ret_bench_list = []
    yr_buckets: dict[str, dict[str, list[float]]] = {}

    for d, g in rich_panel.groupby("date", sort=True):
        filled_all = g.loc[(g["block"] == "ok") & g["fwd_ret"].notna() & g["tb_net_ret"].notna()]
        if len(filled_all) < proto.MIN_NAMES:
            continue

        reg = str(g["regime"].iloc[0])
        # 若持有期 H 大于调仓步长 step (如 H=60, step=20)，资金分成 overlap_k (3) 份滚动入场，
        # 每步长周期的组合收益率为单笔 H 期收益除以 overlap_k
        b_ret = float(filled_all["net_ret"].mean()) / overlap_k

        pick_frames = []
        allow_n = any(sp["allow_neutral"] for sp in top_specs)
        for sp in top_specs:
            cands = select_strategy_candidates(g, sp["key"], top_k=10)
            if not cands.empty:
                pick_frames.append(cands)
        if pick_frames:
            comb = pd.concat(pick_frames, ignore_index=True).drop_duplicates("code")
            filled_p = comb.loc[(comb["block"] == "ok") & comb["tb_net_ret"].notna()]
            raw_ret = (float(filled_p["tb_net_ret"].mean()) if not filled_p.empty else 0.0) / overlap_k
        else:
            raw_ret = 0.0

        # 市场闸门仓位控制：红灯空仓(0.0)，黄灯若允许则半仓(0.5)否则空仓，绿灯满仓(1.0)
        if reg == REGIME_BULL:
            pos_w = 1.0
        elif reg == REGIME_NEUTRAL and allow_n:
            pos_w = 0.5
        else:
            pos_w = 0.0
        gated_ret = raw_ret * pos_w

        dates.append(str(d))
        ret_gated_list.append(gated_ret)
        ret_raw_list.append(raw_ret)
        ret_bench_list.append(b_ret)

        eq_gated.append(eq_gated[-1] * (1.0 + gated_ret))
        eq_raw.append(eq_raw[-1] * (1.0 + raw_ret))
        eq_bench.append(eq_bench[-1] * (1.0 + b_ret))

        yr = str(d)[:4]
        yb = yr_buckets.setdefault(yr, {"gated": [], "raw": [], "bench": []})
        yb["gated"].append(gated_ret)
        yb["raw"].append(raw_ret)
        yb["bench"].append(b_ret)

    if not dates:
        return {}

    eq_g = np.asarray(eq_gated[1:], dtype=float)
    eq_r = np.asarray(eq_raw[1:], dtype=float)
    eq_b = np.asarray(eq_bench[1:], dtype=float)

    def _dd_series(eq_arr: np.ndarray) -> np.ndarray:
        peaks = np.maximum.accumulate(np.maximum(1.0, eq_arr))
        return (eq_arr - peaks) / peaks

    dd_g = _dd_series(eq_g)
    dd_r = _dd_series(eq_r)
    dd_b = _dd_series(eq_b)

    def _perf_summary(rets: list[float], eq_arr: np.ndarray, dd_arr: np.ndarray) -> dict:
        r_arr = np.asarray(rets, dtype=float)
        n_p = len(r_arr)
        if n_p == 0:
            return {"tot_ret": 0.0, "cagr": 0.0, "max_dd": 0.0, "sharpe": 0.0, "calmar": 0.0}
        tot = float(eq_arr[-1] - 1.0)
        yrs = max(0.5, n_p / periods_per_yr)
        cagr = float(eq_arr[-1] ** (1.0 / yrs) - 1.0) if eq_arr[-1] > 0 else -0.99
        mdd = float(np.min(dd_arr))
        sd = float(np.std(r_arr, ddof=1)) if n_p > 1 else 0.0
        sharpe = float((np.mean(r_arr) / sd) * np.sqrt(periods_per_yr)) if sd > 1e-6 else 0.0
        calmar = float(cagr / abs(mdd)) if abs(mdd) > 1e-4 else 0.0
        return {
            "tot_ret": round(tot, 4),
            "cagr": round(cagr, 4),
            "max_dd": round(mdd, 4),
            "sharpe": round(sharpe, 2),
            "calmar": round(calmar, 2),
        }

    # 切分 2024 之后纯样本外区间的组合绩效
    oos_idx = [i for i, d in enumerate(dates) if d > proto.TRAIN_END]
    if oos_idx:
        i0 = oos_idx[0]
        base_g = eq_g[i0 - 1] if i0 > 0 else 1.0
        base_r = eq_r[i0 - 1] if i0 > 0 else 1.0
        base_b = eq_b[i0 - 1] if i0 > 0 else 1.0
        oos_eq_g = eq_g[i0:] / base_g
        oos_eq_r = eq_r[i0:] / base_r
        oos_eq_b = eq_b[i0:] / base_b
        oos_stats = {
            "gated": _perf_summary(ret_gated_list[i0:], oos_eq_g, _dd_series(oos_eq_g)),
            "raw": _perf_summary(ret_raw_list[i0:], oos_eq_r, _dd_series(oos_eq_r)),
            "bench": _perf_summary(ret_bench_list[i0:], oos_eq_b, _dd_series(oos_eq_b)),
        }
    else:
        oos_stats = {}

    yearly_rows = []
    for yr in sorted(yr_buckets.keys()):
        yb = yr_buckets[yr]
        g_r = float(np.prod([1.0 + x for x in yb["gated"]]) - 1.0)
        r_r = float(np.prod([1.0 + x for x in yb["raw"]]) - 1.0)
        b_r = float(np.prod([1.0 + x for x in yb["bench"]]) - 1.0)
        yearly_rows.append({
            "year": yr,
            "is_oos": yr >= "2024",
            "gated_ret": round(g_r, 4),
            "raw_ret": round(r_r, 4),
            "bench_ret": round(b_r, 4),
            "excess_ret": round(g_r - b_r, 4),
        })

    return {
        "strategy_names": [sp["name"] for sp in top_specs],
        "dates": dates,
        "eq_gated": [round(float(x), 4) for x in eq_g],
        "eq_raw": [round(float(x), 4) for x in eq_r],
        "eq_bench": [round(float(x), 4) for x in eq_b],
        "dd_gated": [round(float(x), 4) for x in dd_g],
        "dd_raw": [round(float(x), 4) for x in dd_r],
        "dd_bench": [round(float(x), 4) for x in dd_b],
        "full_stats": {
            "gated": _perf_summary(ret_gated_list, eq_g, dd_g),
            "raw": _perf_summary(ret_raw_list, eq_r, dd_r),
            "bench": _perf_summary(ret_bench_list, eq_b, dd_b),
        },
        "oos_stats": oos_stats,
        "yearly": yearly_rows,
    }


def _fundamental_and_risk_veto(
    fmap: dict,
    code: str,
    close: float,
    ma20: float,
    ma60: float,
    ma120: float,
    slope60: float,
    rev1: float,
    volratio: float,
    pos250: float,
    horizon: int,
) -> tuple[str, float, list[str]]:
    """
    结合当下真实基本面与量价结构，执行：
      1) 基本面安全垫加减分 (-0.025 ~ +0.030)
      2) 一票否决排雷卫士 (Risk Veto)：返回 (f_tag, f_bonus, veto_reasons)
    """
    is_long = int(horizon) >= 60
    f = fmap.get(code) or {}
    pe = f.get("pe_ttm")
    pb = f.get("pb")
    roe = f.get("roe")
    mv = f.get("total_mv")
    float_mv = f.get("float_mv")
    debt = f.get("debt_ratio")
    pyoy = f.get("profit_yoy")

    parts = []
    veto = []
    bonus = 0.0

    # 基本面打分与排雷
    if pe is not None and np.isfinite(pe):
        if 0 < pe <= 22:
            parts.append(f"PE={pe:.1f}低估")
            bonus += 0.012
        elif 22 < pe <= 45:
            parts.append(f"PE={pe:.1f}")
        elif pe < 0:
            parts.append("亏损")
            bonus -= 0.018
            if is_long:
                veto.append("业绩亏损(PE<0)")
        elif pe > 220:
            veto.append(f"估值极度透支(PE={pe:.0f})")

    if roe is not None and np.isfinite(roe):
        if roe >= 12.0:
            parts.append(f"ROE={roe:.1f}%高质")
            bonus += 0.015
        elif roe >= 6.0:
            parts.append(f"ROE={roe:.1f}%")
            bonus += 0.005
        elif roe < -5.0:
            veto.append(f"ROE深亏({roe:.1f}%)")

    if pb is not None and np.isfinite(pb) and 0 < pb <= 1.8:
        parts.append(f"PB={pb:.2f}低PB")
        bonus += 0.008

    if debt is not None and np.isfinite(debt) and debt >= 85.0 and is_long:
        veto.append(f"高负债风险({debt:.0f}%)")
    if pyoy is not None and np.isfinite(pyoy) and pyoy <= -50.0 and is_long:
        veto.append(f"净利断崖({pyoy:.0f}%)")

    eff_mv = float_mv if (float_mv is not None and np.isfinite(float_mv) and float_mv > 0) else mv
    if eff_mv is not None and np.isfinite(eff_mv) and 0 < eff_mv < 15.0:
        veto.append(f"微盘流动性风险(流通{eff_mv:.1f}亿)")

    # 量价结构排雷
    if np.isfinite(ma20) and np.isfinite(ma60) and np.isfinite(ma120):
        if close < ma20 < ma60 < ma120 and (np.isfinite(slope60) and slope60 < -0.025):
            veto.append("空头阴跌通道")
    if np.isfinite(pos250) and pos250 >= 0.85 and np.isfinite(volratio) and volratio >= 2.2 and np.isfinite(rev1) and rev1 < 0:
        veto.append("高位天量收阴派发")

    return " ".join(parts), float(np.clip(bonus, -0.025, 0.030)), veto


def build_latest_cockpit(
    prepared: list[dict],
    regime_df: pd.DataFrame,
    beta_fixed: np.ndarray,
    beta_tb: np.ndarray,
    features: tuple[str, ...],
    calib: dict,
    strat_evals: list[dict],
    horizon: int,
) -> tuple[str, dict, list[dict], list[dict]]:
    """计算最新交易日 (asof) 指定周期 horizon 的作战台状态（含三重屏障胜率与一票否决排雷）。"""
    if not prepared or regime_df.empty:
        return "", {}, [], []

    is_long = int(horizon) >= 60
    asof = wm._latest_full_date(prepared)
    reg_row = regime_df.loc[regime_df["date"] == asof]
    if reg_row.empty:
        reg_info = regime_df.iloc[-1].to_dict()
    else:
        reg_info = reg_row.iloc[-1].to_dict()

    cur_reg = reg_info["regime_long"] if is_long else reg_info["regime_short"]
    reg_info = dict(reg_info)
    reg_info["regime"] = cur_reg
    reg_info["regime_cn"] = regime_cn_of(cur_reg, horizon)
    reg_info["pos_cap"] = REGIME_POS_CAP.get(cur_reg, 0.4)

    reg_scale = REGIME_SCALE.get(cur_reg, 0.5)
    reg_base = calib.get("regimes", {}).get(cur_reg, {})
    cells = calib.get("cells", {})

    slist = db.load_stock_list()
    names = {}
    if slist is not None and not slist.empty:
        slist = slist.copy()
        slist["code"] = slist["code"].astype(str).str.zfill(6)
        names = dict(zip(slist["code"], slist["name"].astype(str)))

    fmap = db.load_fundamental_map()
    cap_flow_map = db.load_latest_capital_flow()
    risk_cal_map = db.load_risk_calendar_map()
    subsector_map = db.load_subsector_map()

    item_feat_cols = tuple(
        f for f in dict.fromkeys(SHORT_ADV_FEATURES + LONG_FEATURES)
        if f != "alpha_ind5"
    )
    extra_cols = (
        "ma5", "ma20", "ma60", "ma120", "ma250",
        "ma20_slope5", "ma60_slope10", "ma120_slope20",
        "dist_ma20", "dist_ma60", "dist_ma120",
        "high20_prev", "day_intra", "atr14", "atr_pct",
    )
    rows = []
    for item in prepared:
        if "high" not in item:
            continue
        dates = item["date"]
        if str(dates[-1]) == asof:
            pos = -1
        else:
            m = np.where(dates == asof)[0]
            if len(m) == 0:
                continue
            pos = int(m[-1])
        ext = _extended_item_arrays(item)
        close_val = float(item["close"][pos])
        if not np.isfinite(close_val) or close_val <= 0:
            continue
        rec = {
            "date": asof,
            "code": str(item["code"]),
            "name": names.get(str(item["code"]), ""),
            "industry": item["industry"] or "未分类",
            "close": close_val,
        }
        for f in item_feat_cols:
            rec[f] = float(ext[f][pos])
        for c in extra_cols:
            rec[c] = float(ext[c][pos])
        rows.append(rec)

    if not rows:
        return asof, reg_info, [], []

    df = _attach_sector_spillover(pd.DataFrame(rows))
    all_feat_cols = tuple(dict.fromkeys(SHORT_ADV_FEATURES + LONG_FEATURES))
    df = proto.attach_processed(df, all_feat_cols)
    df["p"] = wm.predict_p(df, features, beta_fixed)
    df["p_tb"] = wm.predict_p(df, features, beta_tb)
    df["decile"] = _assign_deciles(df)

    specs = get_strategy_specs(horizon)
    verified_keys = {r["key"] for r in strat_evals if r["verified"]}
    if not verified_keys:
        verified_keys = {r["key"] for r in strat_evals[:2] if r["key"] not in ("high_mom_chase", "high_flyer_chase")}

    strat_hits: dict[str, list[str]] = {}
    strat_best_win: dict[str, float] = {}
    strat_best_tb: dict[str, float] = {}
    for spec in specs:
        s_key = spec["key"]
        if s_key in ("high_mom_chase", "high_flyer_chase"):
            continue
        ev_map = next((r for r in strat_evals if r["key"] == s_key), None)
        if ev_map is None:
            continue
        cands = select_strategy_candidates(df, s_key, top_k=25)
        if cands.empty:
            continue
        te_w = ev_map["test_gated"].get("win_gross") or ev_map["train_gated"].get("win_gross") or 0.50
        te_tb = ev_map["test_gated"].get("win_tb") or ev_map["train_gated"].get("win_tb") or te_w
        for code in cands["code"].astype(str):
            strat_hits.setdefault(code, []).append(spec["name"])
            if s_key in verified_keys:
                strat_best_win[code] = max(strat_best_win.get(code, 0.0), float(te_w))
                strat_best_tb[code] = max(strat_best_tb.get(code, 0.0), float(te_tb))

    b = np.asarray(beta_tb if len(beta_tb) == 1 + len(features) else beta_fixed, dtype=float)
    df = df.sort_values(["p_tb", "p", "code"], ascending=[False, False, True], kind="mergesort").reset_index(drop=True)
    k_top = max(proto.MIN_BOOK, int(round(len(df) * wm.TOP_FRAC)))

    sl_mult, tp_mult, risk_unit = TB_PARAMS.get(int(horizon), (1.5, 2.5, 0.010))

    all_stocks = []
    today_picks = []

    for rank, r in enumerate(df.itertuples(index=False)):
        code = str(r.code)
        close = float(r.close)
        dec = int(r.decile)
        cell = cells.get(f"{cur_reg}_{dec}", {})
        p_lin = float(r.p)
        p_tb_raw = float(r.p_tb)
        p_cell = float(cell.get("p_gross", p_lin))
        p_cell_tb = float(cell.get("p_tb", p_tb_raw))

        p_cond = 0.65 * p_cell + 0.35 * p_lin
        p_tb_cond = 0.65 * p_cell_tb + 0.35 * p_tb_raw
        if code in strat_best_win:
            p_cond = 0.50 * p_cond + 0.50 * strat_best_win[code]
            p_tb_cond = 0.50 * p_tb_cond + 0.50 * strat_best_tb.get(code, p_tb_cond)

        f_tag, f_bonus, veto_list = _fundamental_and_risk_veto(
            fmap, code, close, float(r.ma20), float(r.ma60), float(r.ma120),
            float(r.ma60_slope10), float(r.rev1), float(r.volratio), float(r.pos250), horizon,
        )

        # ---- 融入新补齐的三大数据源：主力盘口资金流 + 解禁/业绩预告日历 + 二级细分题材 ----
        cf = cap_flow_map.get(code) or {}
        net_inflow_yi = cf.get("net_inflow_yi")
        net_big_r = cf.get("net_big_ratio")
        act_imb = cf.get("active_buy_imb")
        flow_bonus = 0.0
        flow_tag = ""
        if net_big_r is not None and act_imb is not None:
            if net_big_r >= 0.08 and act_imb >= 0.05:
                flow_bonus = 0.015 if not is_long else 0.008
                flow_tag = f"主力抢筹({net_inflow_yi:+.2f}亿)" if net_inflow_yi else "主力大单净买"
            elif net_big_r <= -0.15 and act_imb <= -0.12:
                flow_bonus = -0.018
                if not is_long and net_inflow_yi is not None and net_inflow_yi <= -0.50:
                    veto_list.append(f"主力大单出逃({net_inflow_yi:.2f}亿)")

        cal_events = risk_cal_map.get(code) or []
        cal_tags = []
        for ev in cal_events:
            sev = ev.get("severity")
            summ = ev.get("summary") or ""
            if sev == "veto":
                veto_list.append(summ)
            elif sev == "warn":
                cal_tags.append("⚠" + summ)
                flow_bonus -= 0.010
            elif sev == "bull":
                cal_tags.append("★" + summ)
                flow_bonus += 0.012

        p_cond = float(np.clip(p_cond + (f_bonus if is_long else f_bonus * 0.4) + flow_bonus, 0.15, 0.88))
        p_tb_cond = float(np.clip(p_tb_cond + (f_bonus if is_long else f_bonus * 0.4) + flow_bonus, 0.15, 0.88))

        atr = float(r.atr14) if np.isfinite(r.atr14) and r.atr14 > 0 else close * 0.025
        atr_pct = atr / close
        max_buy = round(close * (1.02 if is_long else 1.015), 2)
        stop_loss = round(max(0.01, close - sl_mult * atr), 2)
        take_profit = round(close + tp_mult * atr, 2)
        raw_pos = risk_unit / max(sl_mult * atr_pct, 0.03)
        pos_pct = round(min(15.0, raw_pos * 100.0) * reg_scale, 1)

        drivers = []
        for j, f in enumerate(features):
            z = getattr(r, f + "_z")
            z = float(z) if np.isfinite(z) else 0.0
            drivers.append((abs(b[j + 1] * z), f, z, float(b[j + 1] * z)))
        drivers.sort(reverse=True)

        hits = list(strat_hits.get(code, []))
        if flow_tag and hits:
            hits.append(flow_tag)
        if f_tag and hits:
            hits.append(f_tag)
        for ct in cal_tags[:1]:
            if hits:
                hits.append(ct)
        if veto_list:
            hits.append("⚠排雷:" + "/".join(veto_list[:2]))

        subsec_name = subsector_map.get(code) or str(r.industry)
        f_row = fmap.get(code) or {}
        pe_ttm_v = f_row.get("pe_ttm")
        pe_hp_v = f_row.get("pe_hist_pct")
        pb_hp_v = f_row.get("pb_hist_pct")

        stock_item = {
            "code": code,
            "name": str(r.name),
            "industry": str(r.industry),
            "subsector": subsec_name,
            "close": round(close, 2),
            "p_cond": round(p_cond, 4),
            "p_tb": round(p_tb_cond, 4),
            "p_up": round(p_lin, 4),
            "decile": dec,
            "top": rank < k_top,
            "pos250": round(float(r.pos250), 3) if np.isfinite(r.pos250) else 0.5,
            "pe_ttm": round(float(pe_ttm_v), 2) if (pe_ttm_v is not None and np.isfinite(float(pe_ttm_v))) else None,
            "pe_hist_pct": round(float(pe_hp_v), 1) if (pe_hp_v is not None and np.isfinite(float(pe_hp_v))) else None,
            "pb_hist_pct": round(float(pb_hp_v), 1) if (pb_hp_v is not None and np.isfinite(float(pb_hp_v))) else None,
            "net_inflow_yi": net_inflow_yi,
            "net_big_ratio": net_big_r,
            "active_buy_imb": act_imb,
            "mean_net": cell.get("mean_tb_net", cell.get("mean_net", 0.0)),
            "avg_win": cell.get("avg_win", 0.05),
            "avg_loss": cell.get("avg_loss", 0.04),
            "payoff": cell.get("payoff", 1.25),
            "atr_pct": round(atr_pct, 4),
            "max_buy": max_buy,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "pos_pct": pos_pct if not veto_list else 0.0,
            "veto": veto_list,
            "setups": hits,
            "drivers": [
                {"feature": f, "name": ALL_FEATURE_CN.get(f, f), "z": round(z, 3), "contrib": round(c, 3)}
                for _, f, z, c in drivers[:3]
            ],
        }
        all_stocks.append(stock_item)

        # 一票否决排雷：凡触发 veto_list 的风险票，一律禁止进入候选池
        # 若当天为红灯防守期，建议仓位 pos_pct 已归 0%，仍保留最强抗跌观察名单(标明建议仓位 0.0% 仅作企稳跟踪)
        if strat_hits.get(code) and not veto_list:
            if cur_reg == REGIME_BEAR:
                stock_item["setups"] = ["[红灯仅观察·勿开仓]"] + list(stock_item["setups"])
            today_picks.append(stock_item)

    today_picks.sort(
        key=lambda x: (len(x["setups"]), x["p_tb"], x["p_cond"], x["p_up"]),
        reverse=True,
    )
    today_picks = today_picks[:30]
    all_stocks.sort(key=lambda x: (-x["p_tb"], -x["p_cond"], -x["p_up"], x["code"]))

    reg_info["base_win_gross"] = round(float(reg_base.get("win_gross", 0.48)), 4)
    reg_info["base_win_tb"] = round(float(reg_base.get("win_tb", 0.50)), 4)
    reg_info["base_win_net"] = round(float(reg_base.get("win_net", 0.46)), 4)
    reg_info["base_payoff"] = round(float(reg_base.get("payoff", 1.2)), 2)
    reg_info["horizon"] = int(horizon)
    return asof, reg_info, all_stocks, today_picks


def evaluate_holding_stop_and_trailing(
    code: str,
    buy_price: float,
    add_date: str = "",
    cur_price: Optional[float] = None,
    horizon: int = 10,
) -> dict:
    """
    为「自选持仓 / 盯盘」页计算单只持仓股的 ATR 初始止损线、最高价移动止盈线 (Trailing Stop) 与操作建议：
      - 若未填买入价 (buy_price <= 0)，按最新收盘价作为参考基准；
      - 扫描自 add_date (或最近 20 个交易日) 以来的最高价 max_high 与最新 ATR14；
      - 初始止损价 init_stop = buy_price - sl_mult * atr14；
      - 移动止损/止盈线 trail_stop：
          * 当最大浮盈 >= 1.8 * atr14 时，激活保本垫 (止损至少上移至 buy_price * 1.005)；
          * 同时按 max_high - (sl_mult * 1.1) * atr14 动态上移锁利；
      - 判定状态：
          * 「🚨 破位止损」(cur_price <= active_stop)
          * 「🎯 触及止盈」(cur_price >= tp_price)
          * 「🔒 移动锁利中」(active_stop > buy_price)
          * 「✅ 正常持有」
    """
    code = str(code).zfill(6)
    kl = db.load_kline(code)
    if kl is None or kl.empty or len(kl) < 15:
        return {
            "stop_price": None, "tp_price": None, "status": "数据不足",
            "tone": "neutral", "dist_stop_pct": None,
        }
    c = pd.to_numeric(kl["close"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(kl["high"], errors="coerce").to_numpy(dtype=float)
    l = pd.to_numeric(kl["low"], errors="coerce").to_numpy(dtype=float)
    dates = kl["date"].astype(str).to_numpy()

    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.fmax(h - l, np.fmax(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr_arr = pd.Series(tr).rolling(14, min_periods=5).mean().to_numpy()
    atr = float(atr_arr[-1]) if np.isfinite(atr_arr[-1]) and atr_arr[-1] > 0 else float(c[-1]) * 0.025

    now_p = float(cur_price) if (cur_price is not None and cur_price > 0) else float(c[-1])
    ref_buy = float(buy_price) if (buy_price is not None and buy_price > 0) else float(c[-1])

    sl_mult, tp_mult, _ = TB_PARAMS.get(int(horizon), (1.5, 2.5, 0.010))
    init_stop = max(0.01, ref_buy - sl_mult * atr)
    tp_price = ref_buy + tp_mult * atr

    # 统计自买入日(add_date)以来的最高价
    since_d = str(add_date)[:10] if add_date else ""
    if since_d and (dates >= since_d).any():
        sub_h = h[dates >= since_d]
        max_high = float(np.nanmax(sub_h)) if len(sub_h) else max(now_p, ref_buy)
    else:
        max_high = max(now_p, ref_buy)
    max_high = max(max_high, now_p, ref_buy)

    active_stop = init_stop
    if buy_price and buy_price > 0:
        if (max_high - ref_buy) >= 1.8 * atr:
            # 已走出显著浮盈，激活保本与移动止盈线
            trail = max_high - (sl_mult * 1.1) * atr
            active_stop = max(init_stop, ref_buy * 1.004, trail)

    dist_stop_pct = (now_p - active_stop) / active_stop * 100.0 if active_stop > 0 else 0.0
    ma20 = float(pd.Series(c).rolling(20, min_periods=10).mean().iloc[-1])

    if now_p <= active_stop:
        if active_stop > ref_buy:
            status = f"🚨 触移动止盈({active_stop:.2f})"
            tone = "warn"
        else:
            status = f"🚨 破位止损({active_stop:.2f})"
            tone = "danger"
    elif now_p >= tp_price:
        status = f"🎯 达目标止盈({tp_price:.2f})"
        tone = "bull"
    elif active_stop > ref_buy:
        status = f"🔒 移动锁利(底线{active_stop:.2f})"
        tone = "bull"
    elif now_p < ma20:
        status = f"⚠ 跌破MA20(守{active_stop:.2f})"
        tone = "warn"
    else:
        status = f"✅ 正常(距止损+{dist_stop_pct:.1f}%)"
        tone = "ok"

    return {
        "stop_price": round(active_stop, 2),
        "tp_price": round(tp_price, 2),
        "status": status,
        "tone": tone,
        "dist_stop_pct": round(dist_stop_pct, 2),
    }


def _seed_and_update_signal_ledger(
    rich_panel: pd.DataFrame,
    prepared: list[dict],
    strat_evals: list[dict],
    today_picks: list[dict],
    asof: str,
    horizon: int,
) -> dict:
    """
    维护并结算「历史推荐信号自动记账本 (Signal Ledger)」：
      1) 从样本外测试窗最近的 6 个绿灯调仓日提取当时的 Top5 共振推荐信号写入账本（供用户直观核验近期历史信号真实表现）；
      2) 将今日 (asof) 的 Top10 共振推荐信号记入账本；
      3) 用每只股票的最新日线自动结算账本中所有信号的状态（止盈出局 / 止损出局 / 到期平仓 / 持仓浮盈中 / 待开盘）。
    """
    h = int(horizon)
    sl_mult, tp_mult, _ = TB_PARAMS.get(h, (1.5, 2.5, 0.010))
    slist = db.load_stock_list()
    names = {}
    if slist is not None and not slist.empty:
        slist = slist.copy()
        slist["code"] = slist["code"].astype(str).str.zfill(6)
        names = dict(zip(slist["code"], slist["name"].astype(str)))

    top_specs = [
        next(sp for sp in get_strategy_specs(h) if sp["key"] == r["key"])
        for r in strat_evals[:2]
        if r["key"] not in ("high_mom_chase", "high_flyer_chase")
    ]
    if not top_specs:
        top_specs = get_strategy_specs(h)[:1]

    ledger_rows = []

    # 1) 回溯最近 12 个绿灯调仓日的历史信号（每期取前 3 只代表票）
    if rich_panel is not None and not rich_panel.empty:
        test_slice = rich_panel.loc[
            (rich_panel["date"].astype(str) > proto.TRAIN_END)
            & (rich_panel["regime"] == REGIME_BULL)
        ]
        recent_dates = sorted(test_slice["date"].astype(str).unique())[-12:]
        for d in recent_dates:
            g = test_slice.loc[test_slice["date"].astype(str) == d]
            for sp in top_specs[:1]:
                cands = select_strategy_candidates(g, sp["key"], top_k=3)
                for r in cands.itertuples(index=False):
                    code = str(r.code).zfill(6)
                    sig_c = float(r.close)
                    atr = float(r.atr14) if np.isfinite(r.atr14) and r.atr14 > 0 else sig_c * 0.025
                    ledger_rows.append({
                        "signal_date": str(d),
                        "horizon": h,
                        "code": code,
                        "name": names.get(code, ""),
                        "industry": str(r.industry),
                        "setups": sp["name"],
                        "p_cond": round(float(getattr(r, "p_tb", r.p)), 4),
                        "p_tb": round(float(getattr(r, "p_tb", r.p)), 4),
                        "signal_close": round(sig_c, 2),
                        "max_buy": round(sig_c * (1.02 if h >= 60 else 1.015), 2),
                        "stop_loss": round(max(0.01, sig_c - sl_mult * atr), 2),
                        "take_profit": round(sig_c + tp_mult * atr, 2),
                    })

    # 2) 记录最新交易日 (asof) 的 Top8 信号
    for p in today_picks[:8]:
        ledger_rows.append({
            "signal_date": str(asof),
            "horizon": h,
            "code": str(p["code"]).zfill(6),
            "name": p.get("name", ""),
            "industry": p.get("industry", ""),
            "setups": " / ".join(p.get("setups") or [])[:40],
            "p_cond": p.get("p_cond"),
            "p_tb": p.get("p_tb", p.get("p_cond")),
            "signal_close": p.get("close"),
            "max_buy": p.get("max_buy"),
            "stop_loss": p.get("stop_loss"),
            "take_profit": p.get("take_profit"),
        })

    # 3) 对所有账本记录按真实日线序列逐日结算
    item_map = {str(it["code"]).zfill(6): it for it in prepared}
    settled_rows = []
    for row in ledger_rows:
        code = row["code"]
        sig_d = row["signal_date"]
        item = item_map.get(code)
        if item is None:
            row["status"] = "待开盘"
            settled_rows.append(row)
            continue

        dates = item["date"]
        m = np.where(dates == sig_d)[0]
        if len(m) == 0:
            row["status"] = "待开盘"
            settled_rows.append(row)
            continue
        i = int(m[-1])
        n = len(dates)
        if i + 1 >= n:
            row["status"] = "今日信号·待明日开盘"
            settled_rows.append(row)
            continue

        open_ = item["open"]
        high = item["high"]
        low = item["low"]
        close = item["close"]
        ext = _extended_item_arrays(item)
        atr_val = float(ext["atr14"][i]) if np.isfinite(ext["atr14"][i]) and ext["atr14"][i] > 0 else float(close[i]) * 0.025

        entry_idx = i + 1
        entry_d = str(dates[entry_idx])
        entry_o = float(open_[entry_idx])
        lim = item["limit"]
        gap_e = entry_o / float(close[i]) - 1.0 if close[i] > 0 else 0.0
        if gap_e >= lim - proto.LIMIT_EPS:
            row["entry_date"] = entry_d
            row["status"] = "一字涨停未买进(剔除)"
            settled_rows.append(row)
            continue

        sl_p = round(max(0.01, entry_o - sl_mult * atr_val), 2)
        tp_p = round(entry_o + tp_mult * atr_val, 2)
        row["entry_date"] = entry_d
        row["entry_price"] = round(entry_o, 2)
        row["stop_loss"] = sl_p
        row["take_profit"] = tp_p

        exit_d = None
        exit_p = None
        status = None
        max_step = min(n - 1 - entry_idx, h)
        for step in range(1, max_step + 1):
            b_idx = entry_idx + step
            o_s, h_s, l_s = float(open_[b_idx]), float(high[b_idx]), float(low[b_idx])
            if l_s <= sl_p:
                exit_d = str(dates[b_idx])
                exit_p = min(o_s, sl_p)
                status = f"止损出局({step}日)"
                break
            if h_s >= tp_p:
                exit_d = str(dates[b_idx])
                exit_p = max(o_s, tp_p)
                status = f"🎯止盈落袋({step}日)"
                break

        if exit_p is None:
            if entry_idx + h < n:
                b_idx = entry_idx + h
                exit_d = str(dates[b_idx])
                exit_p = float(open_[b_idx])
                status = f"到期平仓({h}日)"
            else:
                exit_d = str(dates[-1])
                exit_p = float(close[-1])
                held_n = n - 1 - entry_idx
                status = f"持仓跟踪中({held_n}/{h}日)"

        stamp = proto.STAMP_BEFORE if str(exit_d) < proto.STAMP_DATE else proto.STAMP_AFTER
        gross = exit_p / entry_o - 1.0
        net = (1.0 + gross) * (1.0 - proto.COMMISSION - stamp) / (1.0 + proto.COMMISSION) - 1.0
        row["exit_date"] = exit_d
        row["exit_price"] = round(float(exit_p), 2)
        row["return_pct"] = round(float(net * 100.0), 2)
        row["status"] = status
        settled_rows.append(row)

    try:
        db.save_signal_ledger(settled_rows)
    except Exception:
        pass

    closed = [r for r in settled_rows if r.get("return_pct") is not None and ("止盈" in str(r.get("status")) or "止损" in str(r.get("status")) or "到期" in str(r.get("status")))]
    win_cnt = sum(1 for r in closed if float(r["return_pct"]) > 0)
    tp_cnt = sum(1 for r in closed if "止盈" in str(r.get("status")))
    avg_ret = float(np.mean([float(r["return_pct"]) for r in closed])) if closed else 0.0

    return {
        "n_closed": len(closed),
        "win_rate": round(win_cnt / len(closed), 4) if closed else None,
        "tp_hit_rate": round(tp_cnt / len(closed), 4) if closed else None,
        "avg_ret_pct": round(avg_ret, 2),
        "rows": settled_rows[:50],
    }


def _pct(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{float(x) * 100:.2f}%"


def _num(x, nd=2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{float(x):.{nd}f}"


def _test_regime_breakdown(test_df: pd.DataFrame) -> dict:
    filled = test_df.loc[(test_df["block"] == "ok") & test_df["fwd_ret"].notna() & test_df["tb_net_ret"].notna()].copy()
    if filled.empty:
        return {}
    filled["decile"] = _assign_deciles(filled)
    out = {}
    for reg in (REGIME_BULL, REGIME_NEUTRAL, REGIME_BEAR):
        sub = filled.loc[filled["regime"] == reg]
        if sub.empty:
            out[reg] = {
                "n_dates": 0, "win_all": None, "win_tb_all": None,
                "win_top_dec": None, "win_tb_top_dec": None,
                "mean_net_all": None, "mean_net_top": None, "mean_tb_top": None,
            }
            continue
        day_all = sub.groupby("date")["fwd_ret"].apply(lambda s: float((s > 0).mean()))
        day_tb_all = sub.groupby("date")["tb_net_ret"].apply(lambda s: float((s > 0).mean()))
        day_net = sub.groupby("date")["net_ret"].mean()
        top_sub = sub.loc[sub["decile"] >= 9]
        day_top = top_sub.groupby("date")["fwd_ret"].apply(lambda s: float((s > 0).mean())) if not top_sub.empty else day_all
        day_tb_top = top_sub.groupby("date")["tb_net_ret"].apply(lambda s: float((s > 0).mean())) if not top_sub.empty else day_tb_all
        day_top_net = top_sub.groupby("date")["net_ret"].mean() if not top_sub.empty else day_net
        day_top_tb_net = top_sub.groupby("date")["tb_net_ret"].mean() if not top_sub.empty else day_net
        out[reg] = {
            "n_dates": int(sub["date"].nunique()),
            "win_all": round(float(day_all.mean()), 4),
            "win_tb_all": round(float(day_tb_all.mean()), 4),
            "win_top_dec": round(float(day_top.mean()), 4),
            "win_tb_top_dec": round(float(day_tb_top.mean()), 4),
            "mean_net_all": round(float(day_net.mean()), 4),
            "mean_net_top": round(float(day_top_net.mean()), 4),
            "mean_tb_top": round(float(day_top_tb_net.mean()), 4),
        }
    return out


def _evaluate_one_horizon(
    horizon: int,
    prepared: list[dict],
    regime_df: pd.DataFrame,
    wm_state: dict,
    progress_cb=None,
) -> dict:
    """构建并评测单个持有周期 (10 / 60 / 120) 的完整四层模型、三重屏障实战表现、净值曲线与记账本。"""
    h = int(horizon)
    is_long = h >= 60
    reg_col = "regime_long" if is_long else "regime_short"
    regime_map = dict(zip(regime_df["date"].astype(str), regime_df[reg_col].astype(str)))

    if progress_cb:
        progress_cb(f"正在构建 [{HORIZON_LABELS.get(h, h)}] 15维高阶因子与三重屏障面板...")
    rich_panel = _build_rich_panel(h, prepared, regime_map, step_days=10 if h == 10 else 20)
    train_df = proto.train_slice(rich_panel).copy()
    test_df = proto.test_slice(rich_panel).copy()

    feats = LONG_FEATURES if is_long else SHORT_ADV_FEATURES
    X_tr, y_tr = wm._design(train_df, feats)
    beta_fixed = wm.fit_logit(X_tr, y_tr, lam=0.35) if len(y_tr) >= 200 else np.zeros(1 + len(feats))
    beta_tb = _fit_tb_and_excess_model(train_df, feats, lam=0.35)

    rich_panel["p"] = wm.predict_p(rich_panel, feats, beta_fixed)
    rich_panel["p_tb"] = wm.predict_p(rich_panel, feats, beta_tb)
    train_df = proto.train_slice(rich_panel).copy()
    test_df = proto.test_slice(rich_panel).copy()

    if progress_cb:
        progress_cb(f"正在校准 [{HORIZON_LABELS.get(h, h)}] 三重屏障胜率、净值曲线与信号记账本...")
    calib = calibrate_conditional_cells(train_df)
    test_reg_breakdown = _test_regime_breakdown(test_df)
    strat_evals = evaluate_all_strategies(train_df, test_df, horizon=h)
    tearsheet = build_portfolio_curves_and_tearsheet(rich_panel, strat_evals, horizon=h)

    asof, today_reg, all_stocks, today_picks = build_latest_cockpit(
        prepared, regime_df, beta_fixed, beta_tb, feats, calib, strat_evals, h,
    )
    ledger = _seed_and_update_signal_ledger(
        rich_panel, prepared, strat_evals, today_picks, asof, h,
    )

    verified_strats = [s["name"] for s in strat_evals if s["verified"]]
    bull_te = test_reg_breakdown.get(REGIME_BULL, {})
    bear_te = test_reg_breakdown.get(REGIME_BEAR, {})
    oos_g = (tearsheet.get("oos_stats") or {}).get("gated") or {}
    summary_headline = (
        f"[{HORIZON_LABELS.get(h, h)}] 样本外实测(2024~2026)："
        f"绿灯期高分档实战胜率 {_pct(bull_te.get('win_tb_top_dec') or bull_te.get('win_top_dec'))} "
        f"(单期净赚 {_pct(bull_te.get('mean_tb_top') or bull_te.get('mean_net_top'))})；"
        f"红灯期胜率降至 {_pct(bear_te.get('win_all'))}。"
        f"组合样本外累计收益 {_pct(oos_g.get('tot_ret'))} (最大回撤 {_pct(oos_g.get('max_dd'))}, 夏普 {_num(oos_g.get('sharpe'))})。"
    )

    return {
        "asof": asof,
        "train_end": proto.TRAIN_END,
        "horizon": h,
        "horizon_label": HORIZON_LABELS.get(h, f"{h}日"),
        "features": list(feats),
        "beta": [float(x) for x in beta_fixed],
        "beta_tb": [float(x) for x in beta_tb],
        "summary_headline": summary_headline,
        "winmodel_conclusion": wm_state.get("conclusion", ""),
        "today_regime": today_reg,
        "train_regimes": calib.get("regimes", {}),
        "calib_cells": calib.get("cells", {}),
        "test_regimes": test_reg_breakdown,
        "strategies": strat_evals,
        "tearsheet": tearsheet,
        "ledger": ledger,
        "today_picks": today_picks,
        "stocks": all_stocks,
    }


def compute_latest_regime_only(prepared: list[dict], asof: str) -> pd.DataFrame:
    """仅计算最新交易日 asof 的市场环境闸门（耗时 < 0.1 秒）。"""
    bucket = []
    for item in prepared:
        ext = _extended_item_arrays(item)
        dates = item["date"]
        if str(dates[-1]) == asof:
            pos = -1
        else:
            m = np.where(dates == asof)[0]
            if len(m) == 0:
                continue
            pos = int(m[-1])
        bucket.append((
            ext["above_ma20"][pos],
            ext["above_ma60"][pos],
            ext["rev1"][pos],
            ext["rev5"][pos],
        ))
    arr = np.asarray(bucket, dtype=float) if bucket else np.zeros((0, 4))
    n_act = len(arr)
    b20 = float(np.nanmean(arr[:, 0])) if n_act else 0.5
    b60 = float(np.nanmean(arr[:, 1])) if n_act else 0.5
    u1 = float(np.nanmean(arr[:, 2] > 0)) if n_act else 0.5
    u5 = float(np.nanmean(arr[:, 3] > 0)) if n_act else 0.5

    idx_df = db.load_index_kline("sh000001")
    ic, im20, im60, islope = np.nan, np.nan, np.nan, np.nan
    idx_ok = bool(b20 >= 0.50)
    idx_ok60 = bool(b60 >= 0.45)
    if idx_df is not None and not idx_df.empty:
        idx_df = idx_df.sort_values("date").reset_index(drop=True)
        c_ser = pd.to_numeric(idx_df["close"], errors="coerce")
        m20_ser = c_ser.rolling(20).mean()
        m60_ser = c_ser.rolling(60).mean()
        s_ser = (m20_ser - m20_ser.shift(5)) / m20_ser.shift(5)
        ic = float(c_ser.iloc[-1])
        im20 = float(m20_ser.iloc[-1])
        im60 = float(m60_ser.iloc[-1]) if np.isfinite(m60_ser.iloc[-1]) else im20
        islope = float(s_ser.iloc[-1]) if np.isfinite(s_ser.iloc[-1]) else 0.0
        if np.isfinite(ic) and np.isfinite(im20) and im20 > 0:
            idx_ok = bool(ic > im20 and islope >= -0.003)
            idx_ok60 = bool(ic >= im60 * 0.99)

    is_washout_confirmed = (b20 <= 0.26) and (u1 >= 0.42)
    is_bull_impulse = (
        (idx_ok and (0.45 <= b20 <= 0.78) and u5 >= 0.46 and u1 >= 0.38)
        or (b20 > 0.78 and u1 >= 0.50)
    )
    is_panic_freefall = (not idx_ok) and (u1 < 0.25)
    is_mid_decline = (not idx_ok) and (0.26 < b20 < 0.62) and (u5 < 0.45)
    is_overheated_stall = (b20 > 0.70) and (u1 < 0.40)
    if is_washout_confirmed or is_bull_impulse:
        reg_short = REGIME_BULL
    elif is_panic_freefall or is_mid_decline or is_overheated_stall:
        reg_short = REGIME_BEAR
    else:
        reg_short = REGIME_NEUTRAL

    if (b60 <= 0.42 or (idx_ok60 and b60 <= 0.52)) and (u1 >= 0.25 or idx_ok60):
        reg_long = REGIME_BULL
    elif b60 > 0.62:
        reg_long = REGIME_BEAR
    else:
        reg_long = REGIME_NEUTRAL

    return pd.DataFrame([{
        "date": asof,
        "n_active": n_act,
        "breadth_ma20": round(b20, 4),
        "breadth_ma60": round(b60, 4),
        "breadth_up1": round(u1, 4),
        "breadth_up5": round(u5, 4),
        "idx_close": None if not np.isfinite(ic) else round(ic, 2),
        "idx_ma20": None if not np.isfinite(im20) else round(im20, 2),
        "idx_ma60": None if not np.isfinite(im60) else round(im60, 2),
        "idx_slope5": None if not np.isfinite(islope) else round(islope, 4),
        "idx_above_ma20": idx_ok,
        "regime": reg_short,
        "regime_short": reg_short,
        "regime_long": reg_long,
        "regime_cn": REGIME_CN_SHORT[reg_short],
        "pos_cap": REGIME_POS_CAP[reg_short],
    }])


def refresh_latest_cockpit_fast(progress_cb=None) -> Optional[dict]:
    """增量更新日线后的极速刷新通道：仅加载最近450天窗口重算三周期今日概率与候选池。"""
    st = load_cockpit_state()
    if not st or not st.get("horizons"):
        return run_cockpit(rerun_winmodel=False, progress_cb=progress_cb)

    pnl.invalidate_cache()
    if progress_cb:
        progress_cb("[本地计算] 今日增量已入库，正在从本地库读取均线窗口重算今日概率...")
    prepared = pnl.load_recent_market(cal_days=450)
    if not prepared:
        return st

    asof = wm._latest_full_date(prepared)
    if progress_cb:
        progress_cb(f"[本地计算] 正在刷新 ({asof}) 市场环境闸门与 10日/60日/120日 候选池...")
    regime_df = compute_latest_regime_only(prepared, asof)

    h_map = st.get("horizons") or {}
    for hk in ("10", "60", "120"):
        hd = h_map.get(hk)
        if not hd:
            continue
        h = int(hd.get("horizon") or hk)
        feats = tuple(hd.get("features") or (LONG_FEATURES if h >= 60 else SHORT_ADV_FEATURES))
        beta_fixed = np.asarray(hd.get("beta"), dtype=float)
        beta_tb = np.asarray(hd.get("beta_tb") or hd.get("beta"), dtype=float)
        calib = {
            "regimes": hd.get("train_regimes") or {},
            "cells": hd.get("calib_cells") or {},
        }
        strat_evals = hd.get("strategies") or []
        asof_h, today_reg, all_stocks, today_picks = build_latest_cockpit(
            prepared, regime_df, beta_fixed, beta_tb, feats, calib, strat_evals, h,
        )
        hd["asof"] = asof_h
        hd["today_regime"] = today_reg
        hd["stocks"] = all_stocks
        hd["today_picks"] = today_picks
        hd["ledger"] = _seed_and_update_signal_ledger(
            None, prepared, strat_evals, today_picks, asof_h, h,
        )

    default_h = h_map.get("10") or st
    st["asof"] = default_h.get("asof")
    st["today_regime"] = default_h.get("today_regime")
    st["stocks"] = default_h.get("stocks")
    st["today_picks"] = default_h.get("today_picks")

    state = wm._jsonable(st)
    with open(cockpit_state_path(), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    return state


def run_cockpit(rerun_winmodel: bool = False, progress_cb=None) -> dict:
    """一键运行三周期(10日/60日/120日)完整高阶量化评测、净值归因与信号记账引擎。"""
    from ..data import fetcher

    db.init_db()
    if progress_cb:
        progress_cb("正在同步上证指数最新日线...")
    try:
        fetcher.update_index_kline("sh000001", start_date="20210101")
    except Exception:
        pass

    wm_state = wm.load_state()
    if rerun_winmodel or wm_state is None:
        if progress_cb:
            progress_cb("正在全量股票池上运行训练窗滚动选型与截面逻辑回归...")
        wm_state = wm.run()
    elif len(wm_state.get("stocks") or []) < 1000:
        if progress_cb:
            progress_cb("正在刷新最近完整交易日的全市场截面概率...")
        wm_state = wm.refresh_latest_only()

    prepared, audit = pnl.load_market()
    if progress_cb:
        progress_cb("正在计算全历史市场环境闸门(短波段情绪广度 + 中长线周期位置)...")
    regime_df = compute_market_regimes(prepared)

    horizons_data = {}
    for h in SUPPORTED_HORIZONS:
        horizons_data[str(h)] = _evaluate_one_horizon(
            h, prepared, regime_df, wm_state, progress_cb=progress_cb,
        )

    default_h = horizons_data["10"]
    state = wm._jsonable({
        **default_h,
        "audit": audit,
        "supported_horizons": list(SUPPORTED_HORIZONS),
        "horizons": horizons_data,
    })

    with open(cockpit_state_path(), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)

    # 写入三周期完整审计报告 cockpit_record.txt
    lines = [
        "四层量化决策系统 (v2 高阶因子 + 三重屏障实战 + 组合净值归因) · 样本外验证审计记录",
        "========================================================================================",
        f"数据范围: {audit.get('date_min')} ~ {audit.get('date_max')}  股票数: {audit.get('n_codes')}",
        f"训练截止日(卖出日<=): {proto.TRAIN_END}  |  10日无条件截面基准结论: {wm_state.get('conclusion')}",
        "",
    ]

    for h in SUPPORTED_HORIZONS:
        hd = horizons_data[str(h)]
        ts = hd.get("tearsheet") or {}
        oos_g = (ts.get("oos_stats") or {}).get("gated") or {}
        oos_r = (ts.get("oos_stats") or {}).get("raw") or {}
        oos_b = (ts.get("oos_stats") or {}).get("bench") or {}
        led = hd.get("ledger") or {}

        lines.append(f"==================== 【{hd['horizon_label']}】 ====================")
        lines.append("一、第一层：市场环境闸门 (Market Regime Gate) 样本外 (2024~2026) 真实表现")
        for reg in (REGIME_BULL, REGIME_NEUTRAL, REGIME_BEAR):
            tr_r = hd["train_regimes"].get(reg, {})
            te_r = hd["test_regimes"].get(reg, {})
            lines.append(
                f"  {regime_cn_of(reg, h)}: "
                f"训练窗实战胜率={_pct(tr_r.get('win_tb') or tr_r.get('win_gross'))} (盈亏比={_num(tr_r.get('payoff'))}) | "
                f"测试窗期数={te_r.get('n_dates', 0)} 全市场胜率={_pct(te_r.get('win_all'))} "
                f"高分档固定胜率={_pct(te_r.get('win_top_dec'))} 高分档三重屏障胜率={_pct(te_r.get('win_tb_top_dec'))} "
                f"高分档单期净赚={_pct(te_r.get('mean_tb_top') or te_r.get('mean_net_top'))}"
            )
        lines.append("二、第二层：选股策略 样本外测试窗 (2024~2026) 固定持有 vs 三重屏障实战 (开启绿灯闸门)")
        for s in hd["strategies"]:
            raw_s = s["test_raw"]
            gate_s = s["test_gated"]
            tag = "[★样本外验证通过]" if s["verified"] else "[观察/对照]"
            lines.append(
                f"  {s['rank']}. {s['name']} {tag}\n"
                f"     · 全天候裸跑: 期数={raw_s.get('n_days')} 成交={raw_s.get('n_trades')}笔 "
                f"固定毛胜率={_pct(raw_s.get('win_gross'))} 三重屏障胜率={_pct(raw_s.get('win_tb'))} "
                f"单期净收益={_pct(raw_s.get('mean_net'))} 盈亏比={_num(raw_s.get('payoff'))}\n"
                f"     · 开启绿灯闸门: 期数={gate_s.get('n_days')} 成交={gate_s.get('n_trades')}笔 "
                f"固定毛胜率={_pct(gate_s.get('win_gross'))} 三重屏障胜率={_pct(gate_s.get('win_tb'))} "
                f"(较盲买提升={_pct(gate_s.get('total_tb_lift') or gate_s.get('total_win_lift'))}) "
                f"单期净赚={_pct(gate_s.get('mean_tb_net') or gate_s.get('mean_net'))} "
                f"实战盈亏比={_num(gate_s.get('tb_payoff') or gate_s.get('payoff'))}"
            )
        lines.append(
            f"三、组合净值归因 (2024~2026 样本外): "
            f"开启闸门累计收益={_pct(oos_g.get('tot_ret'))} (年化={_pct(oos_g.get('cagr'))}, 最大回撤={_pct(oos_g.get('max_dd'))}, 夏普={_num(oos_g.get('sharpe'))}) | "
            f"全天候裸跑累计={_pct(oos_r.get('tot_ret'))} (最大回撤={_pct(oos_r.get('max_dd'))}) | "
            f"等权基准累计={_pct(oos_b.get('tot_ret'))} (最大回撤={_pct(oos_b.get('max_dd'))})"
        )
        lines.append(
            f"四、近期绿灯信号实盘跟踪账本 (Signal Ledger): "
            f"已平仓={led.get('n_closed', 0)}笔 真实胜率={_pct(led.get('win_rate'))} "
            f"主动触及止盈率={_pct(led.get('tp_hit_rate'))} 单笔平均净收益={led.get('avg_ret_pct', 0):.2f}%"
        )
        t_reg = hd["today_regime"]
        picks = hd["today_picks"]
        lines.append(
            f"五、最新交易日 ({hd['asof']}) 状态: {t_reg.get('regime_cn')} | "
            f"今日通过排雷的共振候选 ({len(picks)} 只):"
        )
        for p in picks[:8]:
            lines.append(
                f"  {p['code']} {p['name']} ({p['industry']}) 收盘={p['close']:.2f} "
                f"实战胜率(三重屏障)={_pct(p['p_tb'])} 条件胜率={_pct(p['p_cond'])} 年分位={_pct(p.get('pos250'))} "
                f"止损={p['stop_loss']:.2f} 止盈={p['take_profit']:.2f} 命中=[{', '.join(p['setups'])}]"
            )
        lines.append("")

    with open(cockpit_record_path(), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return state


if __name__ == "__main__":
    st = run_cockpit(rerun_winmodel=False, progress_cb=lambda msg: print(msg, flush=True))
    for hk in ("10", "60", "120"):
        print("DONE", hk, ":", st["horizons"][hk]["summary_headline"])






