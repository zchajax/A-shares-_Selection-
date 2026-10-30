"""
策略层 - 因子面板
================
技术因子的定义只在这里：
  动量 = 近 20 日涨幅(%)
  趋势 = (ma5 - ma60) / ma60
  量能 = 近 5 日均量 / 近 20 日均量

fwd_ret 是可成交的毛收益：信号日收盘之后，次日开盘买入，持有 horizon 个交易日，
到期开盘卖出。涨停开盘、跌停开盘、停牌、复权断裂都不给出收益（记在 block 列）。
基本面没有历史序列，不进这张表。
"""
import numpy as np
import pandas as pd

from ..data import database as db
from . import protocol as proto


TECH_FACTORS = list(proto.TECH_FACTORS)
_FACTOR_CN = {"momentum": "动量", "trend": "趋势", "volume": "量能"}

_PREPARED = None
_AUDIT = None


def _load_raw() -> pd.DataFrame:
    etf = {str(c).zfill(6) for c in db.load_etf_codes()}
    with db.get_conn() as conn:
        raw = pd.read_sql(
            "SELECT code, date, open, high, low, close, volume FROM daily_kline",
            conn,
        )
    raw["code"] = raw["code"].astype(str).str.zfill(6)
    raw["date"] = raw["date"].astype(str)
    raw = raw.loc[~raw["code"].isin(etf)]
    # 0/3/6 开头是股票。15/51/58 等基金代码不进这次检验。
    raw = raw.loc[raw["code"].str.startswith(("0", "3", "6"))]
    raw = raw.sort_values(["code", "date"], kind="mergesort")
    return raw.reset_index(drop=True)


def invalidate_cache():
    """当本地数据库日线更新后，清空内存面板缓存以便重新加载最新行情。"""
    global _PREPARED, _AUDIT
    _PREPARED = None
    _AUDIT = None


def load_recent_market(cal_days: int = 450) -> list[dict]:
    """
    仅加载最近 cal_days 个自然日(约 300 个交易日)的日线面板，
    专供每日增量更新后极速重算当日 10/60/120 日截面特征与候选池(耗时仅 2~3 秒)。
    """
    from datetime import date, timedelta
    since = (date.today() - timedelta(days=int(cal_days))).isoformat()
    etf = {str(c).zfill(6) for c in db.load_etf_codes()}
    with db.get_conn() as conn:
        raw = pd.read_sql(
            "SELECT code, date, open, high, low, close, volume "
            "FROM daily_kline WHERE date >= ?",
            conn,
            params=(since,),
        )
    if raw.empty:
        return []
    raw["code"] = raw["code"].astype(str).str.zfill(6)
    raw = raw.loc[~raw["code"].isin(etf)]
    raw = raw.loc[raw["code"].str.startswith(("0", "3", "6"))]
    raw = raw.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)

    slist = db.load_stock_list()
    indmap = {}
    if slist is not None and not slist.empty:
        slist = slist.copy()
        slist["code"] = slist["code"].astype(str).str.zfill(6)
        indmap = dict(zip(slist["code"], slist["industry"]))

    frames = []
    for code, g in raw.groupby("code", sort=False):
        if len(g) < proto.WARMUP:
            continue
        close = g["close"].to_numpy(dtype=float)
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        vol = g["volume"].to_numpy(dtype=float)
        frames.append({
            "code": str(code),
            "industry": indmap.get(str(code)) or "未分类",
            "date": g["date"].astype(str).to_numpy(),
            "open": g["open"].to_numpy(dtype=float),
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "limit": proto.limit_ratio(str(code)),
        })
    return frames


def load_market(progress_cb=None):
    """读入并算好因子的逐股表。同一进程里只做一次。"""
    global _PREPARED, _AUDIT
    if _PREPARED is not None:
        if progress_cb:
            progress_cb(1, 1)
        return _PREPARED, _AUDIT

    raw = _load_raw()
    slist = db.load_stock_list()
    if slist is None or slist.empty:
        indmap = {}
        names = pd.DataFrame(columns=["code", "name"])
    else:
        slist = slist.copy()
        slist["code"] = slist["code"].astype(str).str.zfill(6)
        indmap = dict(zip(slist["code"], slist["industry"]))
        names = slist

    last = raw.groupby("code", sort=False)["date"].agg(["min", "max", "count"])
    global_max = str(raw["date"].max()) if not raw.empty else ""
    global_min = str(raw["date"].min()) if not raw.empty else ""
    n_survive = int((last["max"] == global_max).sum()) if len(last) else 0
    n_delist_name = 0
    if not names.empty and "name" in names.columns:
        n_delist_name = int(names["name"].fillna("").astype(str).str.contains("退").sum())

    prev_close = raw.groupby("code", sort=False)["close"].shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        day_ret = raw["close"].to_numpy(dtype=float) / prev_close.to_numpy(dtype=float) - 1.0
    n_break = int(np.nansum(np.abs(day_ret) > proto.BREAK_ABS))

    _AUDIT = {
        "n_rows": int(len(raw)),
        "n_codes": int(raw["code"].nunique()) if len(raw) else 0,
        "date_min": global_min,
        "date_max": global_max,
        "n_end_on_last": n_survive,
        "n_name_delist": n_delist_name,
        "n_stock_list": int(len(names)),
        "n_break_rows": n_break,
        "n_start_after_2022": int((last["min"] > "2022-01-01").sum()) if len(last) else 0,
    }

    frames = []
    total = int(raw["code"].nunique()) if len(raw) else 0
    for i, (code, g) in enumerate(raw.groupby("code", sort=False)):
        if progress_cb and (i % 50 == 0 or i + 1 == total):
            progress_cb(i + 1, total)
        g = g.reset_index(drop=True)
        if len(g) < proto.WARMUP + 2:
            continue
        close = g["close"].to_numpy(dtype=float)
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        vol = g["volume"].to_numpy(dtype=float)
        mom = pd.Series(close).pct_change(20).to_numpy() * 100.0
        ma5 = pd.Series(close).rolling(5).mean().to_numpy()
        ma60 = pd.Series(close).rolling(60).mean().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            trend = (ma5 - ma60) / ma60
        vser = pd.Series(vol)
        with np.errstate(invalid="ignore", divide="ignore"):
            volf = (vser.rolling(5).mean() / vser.rolling(20).mean().replace(0, np.nan)).to_numpy()
        industry = indmap.get(code) or "未分类"
        frames.append({
            "code": code,
            "industry": industry,
            "date": g["date"].astype(str).to_numpy(),
            "open": g["open"].to_numpy(dtype=float),
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "momentum": mom,
            "trend": trend,
            "volume_f": volf,
            "limit": proto.limit_ratio(code),
        })
    _PREPARED = frames
    return _PREPARED, _AUDIT


def _signals_from_prepared(item: dict, horizon: int, min_hist: int) -> pd.DataFrame | None:
    """由一只股票的数组生成信号行。block: ok/limit_up/limit_down/suspend/break/missing。"""
    close = item["close"]
    open_ = item["open"]
    vol = item["volume"]
    dates = item["date"]
    n = close.size
    last = n - 1 - (1 + horizon)
    start = min_hist - 1
    if start > last:
        return None
    i = np.arange(start, last + 1)
    entry = i + 1
    exit_ = i + 1 + horizon
    brk = np.zeros(n, dtype=np.int32)
    with np.errstate(invalid="ignore", divide="ignore"):
        day_ret = np.empty(n)
        day_ret[0] = np.nan
        day_ret[1:] = close[1:] / close[:-1] - 1.0
    brk[1:] = np.abs(day_ret[1:]) > proto.BREAK_ABS
    cbrk = np.cumsum(brk)
    # brk[i+1 .. i+1+horizon] = cbrk[exit_] - cbrk[i]
    n_break = cbrk[exit_] - cbrk[i]
    window_break = n_break > 0

    entry_open = open_[entry]
    exit_open = open_[exit_]
    entry_prev = close[i]
    exit_prev = close[i + horizon]
    entry_vol = vol[entry]
    exit_vol = vol[exit_]
    lim = item["limit"]
    with np.errstate(invalid="ignore", divide="ignore"):
        gap_e = entry_open / entry_prev - 1.0
        gap_x = exit_open / exit_prev - 1.0

    block = np.full(i.size, "ok", dtype=object)
    bad = ~(
        np.isfinite(entry_open) & np.isfinite(exit_open)
        & np.isfinite(entry_prev) & np.isfinite(exit_prev)
        & (entry_open > 0) & (exit_open > 0)
    )
    block[bad] = "missing"
    block[(block == "ok") & window_break] = "break"
    block[(block == "ok") & (entry_vol <= 0)] = "suspend"
    up = (block == "ok") & np.isfinite(gap_e) & (gap_e >= lim - proto.LIMIT_EPS)
    block[up] = "limit_up"
    block[(block == "ok") & (exit_vol <= 0)] = "suspend"
    down = (block == "ok") & np.isfinite(gap_x) & (gap_x <= -(lim - proto.LIMIT_EPS))
    block[down] = "limit_down"

    with np.errstate(invalid="ignore", divide="ignore"):
        gross = exit_open / entry_open - 1.0
    gross = np.where(block == "ok", gross, np.nan)

    mom = item["momentum"][i]
    trend = item["trend"][i]
    volf = item["volume_f"][i]
    keep = np.isfinite(mom) & np.isfinite(trend) & np.isfinite(volf) & (block != "missing")
    if not np.any(keep):
        return None
    return pd.DataFrame({
        "date": dates[i[keep]],
        "code": item["code"],
        "industry": item["industry"],
        "momentum": mom[keep],
        "trend": trend[keep],
        "volume": volf[keep],
        "fwd_ret": gross[keep],
        "block": block[keep],
        "entry_date": dates[entry[keep]],
        "exit_date": dates[exit_[keep]],
    })


def build_panel(fwd_days: int = 5, min_len: int = None, progress_cb=None) -> pd.DataFrame:
    """
    长表面板。date 是信号日（因子只用到这一天收盘）。
    fwd_ret 只在 block=='ok' 时有值，单位是开盘到开盘的毛收益。
    """
    if min_len is None:
        min_len = proto.WARMUP
    prepared, _ = load_market(progress_cb=progress_cb)
    frames = []
    for item in prepared:
        sig = _signals_from_prepared(item, int(fwd_days), int(min_len))
        if sig is not None and not sig.empty:
            frames.append(sig)
    cols = ["date", "code", "industry", "momentum", "trend", "volume",
            "fwd_ret", "block", "entry_date", "exit_date"]
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def sample_rebalance_dates(panel: pd.DataFrame, step: int) -> list:
    """每隔 step 个信号日取一个调仓日，使持有期首尾相接、收益不重叠。"""
    dates = sorted(panel["date"].astype(str).unique())
    if step <= 1:
        return dates
    return dates[::step]


def self_check_signals() -> list[str]:
    """用构造出来的价格核对涨停分类，不读数据库。"""
    n = 100
    close = np.full(n, 10.0)
    open_ = np.full(n, 10.0)
    open_[90] = 11.0  # 相对前收 10，正好 +10%
    dates = pd.bdate_range("2024-01-01", periods=n).strftime("%Y-%m-%d").to_numpy()
    item = {
        "code": "600000",
        "industry": "银行",
        "date": dates,
        "open": open_,
        "close": close,
        "volume": np.full(n, 1000.0),
        "momentum": np.full(n, 1.0),
        "trend": np.full(n, 0.01),
        "volume_f": np.full(n, 1.0),
        "limit": 0.10,
    }
    # 因子在检验里直接给定，warmup 仍按索引截断。
    sig = _signals_from_prepared(item, horizon=5, min_hist=90)
    ok = True
    lines = []

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        lines.append(("PASS " if cond else "FAIL ") + name)

    check("合成序列能生成信号", sig is not None and not sig.empty)
    if sig is not None and not sig.empty:
        hit = sig.loc[sig["entry_date"] == dates[90]]
        check("10% 开盘被标成涨停且没有收益",
              len(hit) == 1 and hit.iloc[0]["block"] == "limit_up"
              and not np.isfinite(hit.iloc[0]["fwd_ret"]))
        flat = sig.loc[sig["block"] == "ok"]
        check("其余平盘有成交且毛收益为 0",
              len(flat) > 0 and np.nanmax(np.abs(flat["fwd_ret"].to_numpy())) < 1e-12)
    if not ok:
        raise RuntimeError("信号分类核对失败:\n" + "\n".join(lines))
    return lines
