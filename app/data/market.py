"""
数据层 - 大盘行情(整体市场概览)

给"行情"页签提供三类免费且实测稳定的数据源(均避开被限流的东财源):
- 核心指数实时行情: 新浪 stock_zh_index_spot_sina(一次返回全市场 562 个指数)
- 市场情绪/涨跌家数: 乐咕 stock_market_activity_legu(上涨/下跌/涨停/跌停等)
- 两市成交额/总貌: 交易所 stock_sse_summary + stock_szse_summary

所有函数不入库(行情是易变的盘中快照,无需持久化),失败返回空结构由 UI 兜底。
"""
import datetime
import socket
import threading
import time

import pandas as pd

from . import fetcher  # 复用 _check_ak / _retry / 全局超时设置
from . import database as db

try:
    import akshare as ak
except ImportError:
    ak = None

socket.setdefaulttimeout(15)

# 全市场实时快照短期复用缓存(10秒TTL)，避免同一轮行情刷新重复拉取
_SPOT_SHARED_CACHE = {"ts": 0.0, "data": None}
_SPOT_LOCK = threading.Lock()


def _shared_spot(ttl: float = 3.5) -> dict:
    now = time.time()
    with _SPOT_LOCK:
        if _SPOT_SHARED_CACHE["data"] and (now - _SPOT_SHARED_CACHE["ts"]) < ttl:
            return _SPOT_SHARED_CACHE["data"]
    from . import realtime
    spot = realtime.fetch_spot()
    if spot:
        with _SPOT_LOCK:
            _SPOT_SHARED_CACHE["data"] = spot
            _SPOT_SHARED_CACHE["ts"] = time.time()
    return spot or (_SPOT_SHARED_CACHE["data"] or {})


def force_refresh_spot() -> dict:
    """强制拉取最新一轮全市场实时快照并更新共享缓存(供每轮行情刷新入口调用)。"""
    from . import realtime
    spot = realtime.fetch_spot()
    if spot:
        with _SPOT_LOCK:
            _SPOT_SHARED_CACHE["data"] = spot
            _SPOT_SHARED_CACHE["ts"] = time.time()
    return spot or (_SPOT_SHARED_CACHE["data"] or {})


def _run_bounded(fn, timeout_s: float = 4.5, default=None):
    """在守护线程中执行可能因第三方网站(东财/同花顺/上交所)无超时而卡死的调用，超时立即返回 default。"""
    box = {}

    def target():
        try:
            box["res"] = fn()
        except Exception as e:
            box["err"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if "res" in box:
        return box["res"]
    return default

# ---- 进程内 TTL 缓存: 部分市场数据拉取偏慢(涨停池~4s/全市场快照~23s),
# 定时刷新时若每 10s 都重拉会拖慢整个行情页,故各自加缓存,过期才重拉。 ----
_LIMIT_POOL_CACHE = {"ts": 0.0, "data": None}
_LIMIT_POOL_TTL = 45.0            # 涨停池: 盘中封板/连板变化不会太快
_UPDOWN_DIST_CACHE = {"ts": 0.0, "data": None}
_UPDOWN_DIST_TTL = 3.5            # 涨跌幅分布直接复用腾讯内存快照(0.005s)，每轮10s轮询均实时刷新
# 量能对比: 今日量+5日均量(日线口径, 60s 节流刷新, 见 fetch_volume_compare)
_AVG_VOL_CACHE = {"date": None, "ts": 0.0,
                  "sh_today": None, "sz_today": None, "sh": None, "sz": None}
# 量能对比(成交额口径, 优先): 今日成交额+5日均额(东财指数日线 amount, 元)。
_AVG_AMT_CACHE = {"date": None, "ts": 0.0,
                  "sh_today": None, "sz_today": None, "sh": None, "sz": None}


def _vol_cache_key() -> str:
    """量能缓存的自然日键(YYYYMMDD)。"""
    return datetime.datetime.now().strftime("%Y%m%d")
# 板块多日趋势/量比: 按板块名缓存日线派生指标, 历史当天不变故按自然日失效。
_BOARD_TREND_CACHE = {"date": None, "data": {}}

# 核心指数白名单: (新浪代码, 展示名)。顺序即页签内展示顺序。
# 新浪 stock_zh_index_spot_sina 的"代码"列形如 sh000001 / sz399001。
CORE_INDEXES = [
    ("sh000001", "上证指数"),
    ("sz399001", "深证成指"),
    ("sz399006", "创业板指"),
    ("sh000688", "科创50"),
    ("sh000300", "沪深300"),
    ("sh000905", "中证500"),
    ("sh000852", "中证1000"),
    ("sh000016", "上证50"),
    ("sz399005", "中小100"),
    ("bj899050", "北证50"),
    ("sh000010", "上证180"),
    ("sh000009", "上证380"),
]


def fetch_index_spot() -> pd.DataFrame:
    """
    拉取核心指数实时行情。
    优先走腾讯 qt.gtimg.cn 批量接口(一次请求 12 个核心指数 + 深证综指, ~0.2s)，
    失败再回退新浪 stock_zh_index_spot_sina。
    返回 DataFrame 列: code,name,price,chg_pct,chg_amt,pre_close,open,high,low,
    amount(成交额,元)。按 CORE_INDEXES 顺序排列;拉取失败返回空 DataFrame。
    """
    try:
        query_codes = [c for c, _ in CORE_INDEXES] + ["sz399106"]
        url = "https://qt.gtimg.cn/q=" + ",".join(query_codes)
        with fetcher._TX_SEM:
            resp = fetcher._http().get(url, timeout=8)
        if resp.status_code == 200 and len(resp.text) > 50:
            by_code = {}
            for line in resp.text.splitlines():
                f = line.split("~")
                if len(f) < 38 or "=" not in f[0]:
                    continue
                head = f[0].split("=")[0].strip()
                sym = head[2:] if head.startswith("v_") else head
                def _fl(idx):
                    try:
                        return float(f[idx])
                    except Exception:
                        return float("nan")
                amt_yuan = _fl(37) * 10_000.0
                vol_val = _fl(36)
                by_code[sym] = {
                    "code": sym,
                    "price": _fl(3),
                    "pre_close": _fl(4),
                    "open": _fl(5),
                    "chg_amt": _fl(31),
                    "chg_pct": _fl(32),
                    "high": _fl(33),
                    "low": _fl(34),
                    "volume": vol_val,
                    "amount": amt_yuan,
                }
            rows = []
            for code, name in CORE_INDEXES:
                item = by_code.get(code)
                if item and item["price"] == item["price"] and item["price"] > 0:
                    r = dict(item)
                    r["name"] = name
                    rows.append(r)
            if rows:
                out = pd.DataFrame(rows)
                sh_amt = by_code.get("sh000001", {}).get("amount")
                sz_amt = by_code.get("sz399106", {}).get("amount") or by_code.get("sz399001", {}).get("amount")
                sh_amt = sh_amt if (sh_amt and sh_amt == sh_amt) else None
                sz_amt = sz_amt if (sz_amt and sz_amt == sz_amt) else None
                tot = sum(v for v in (sh_amt, sz_amt) if v is not None) if (sh_amt or sz_amt) else None
                out.attrs["sh_amount"] = sh_amt
                out.attrs["sz_amount"] = sz_amt
                out.attrs["total_amount"] = tot
                out.attrs["sh_volume"] = by_code.get("sh000001", {}).get("volume")
                out.attrs["sz_volume"] = by_code.get("sz399106", {}).get("volume")
                return out
    except Exception:
        pass

    fetcher._check_ak()
    try:
        raw = fetcher._retry(lambda: ak.stock_zh_index_spot_sina(), tries=2)
    except Exception as e:  # noqa
        print(f"[warn] 指数实时行情拉取失败: {e}")
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    ren = {"代码": "code", "名称": "name", "最新价": "price",
           "涨跌额": "chg_amt", "涨跌幅": "chg_pct", "昨收": "pre_close",
           "今开": "open", "最高": "high", "最低": "low",
           "成交量": "volume", "成交额": "amount"}
    df = df.rename(columns=ren)
    for c in ["price", "chg_amt", "chg_pct", "pre_close", "open",
              "high", "low", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 两市成交额: 从全量里取综指(沪 sh000001 / 深 sz399106)成交额, 口径最标准
    def _col(code, col):
        row = df[df["code"] == code]
        if row.empty:
            return None
        v = row.iloc[0].get(col)
        try:
            return float(v) if pd.notna(v) else None
        except Exception:  # noqa
            return None
    sh_amt = _col("sh000001", "amount")
    sz_amt = _col("sz399106", "amount")
    tot = sum(v for v in (sh_amt, sz_amt) if v is not None) \
        if (sh_amt is not None or sz_amt is not None) else None
    # 成交量(股): 供量能对比用——与新浪指数日线 volume 同口径(成交额日线源
    # 已被限流不可用, 故量能对比统一走成交量, 量比≈额比, 对指数足够可靠)
    sh_vol = _col("sh000001", "volume")
    sz_vol = _col("sz399106", "volume")

    wl = {code: (i, name) for i, (code, name) in enumerate(CORE_INDEXES)}
    out = df[df["code"].isin(wl)].copy()
    if out.empty:
        return pd.DataFrame()
    out["_ord"] = out["code"].map(lambda c: wl[c][0])
    out["name"] = out["code"].map(lambda c: wl[c][1])  # 用白名单里的规范名
    out = out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    keep = ["code", "name", "price", "chg_pct", "chg_amt", "pre_close",
            "open", "high", "low", "amount"]
    out = out[[c for c in keep if c in out.columns]]
    out.attrs["sh_amount"] = sh_amt
    out.attrs["sz_amount"] = sz_amt
    out.attrs["total_amount"] = tot
    out.attrs["sh_volume"] = sh_vol
    out.attrs["sz_volume"] = sz_vol
    return out


def _activity_from_spot(spot: dict) -> dict:
    """用腾讯全市场实时快照秒算沪深京A股涨跌家数、涨跌停(含ST与不含ST)、停牌与活跃度(零外部阻塞)。"""
    if not spot:
        return {}
    up = down = flat = halt = zt = dt = zt_real = dt_real = 0
    latest_ts = ""
    for code, q in spot.items():
        if not q or not str(code).startswith(("00", "30", "60", "68", "92", "83", "87", "43")):
            continue
        if q.get("stale"):
            halt += 1
            continue
        chg = q.get("chg_pct")
        price = q.get("price")
        if chg is None or chg != chg or price is None or price <= 0:
            continue
        ts = q.get("ts") or ""
        if ts > latest_ts:
            latest_ts = ts
        c_val = float(chg)
        if c_val > 0:
            up += 1
        elif c_val < 0:
            down += 1
        else:
            flat += 1
        nm = str(q.get("name") or "")
        is_st = "ST" in nm.upper()
        up_lim = q.get("up_limit")
        dn_lim = q.get("down_limit")
        if up_lim and up_lim > 0 and dn_lim and dn_lim > 0:
            if float(price) >= float(up_lim) - 1e-4:
                zt += 1
                if not is_st:
                    zt_real += 1
            elif float(price) <= float(dn_lim) + 1e-4:
                dt += 1
                if not is_st:
                    dt_real += 1
        else:
            lim = 29.5 if str(code).startswith(("92", "83", "87", "43")) else (
                19.5 if str(code).startswith(("300", "301", "688", "689")) else (4.85 if is_st else 9.75)
            )
            if c_val >= lim:
                zt += 1
                if not is_st:
                    zt_real += 1
            elif c_val <= -lim:
                dt += 1
                if not is_st:
                    dt_real += 1
    tot = up + down + flat
    act_pct = f"{(up / tot * 100):.2f}%" if tot > 0 else "—"
    if len(latest_ts) >= 14 and latest_ts[:14].isdigit():
        stat_dt = f"{latest_ts[:4]}-{latest_ts[4:6]}-{latest_ts[6:8]} {latest_ts[8:10]}:{latest_ts[10:12]}:{latest_ts[12:14]}"
    else:
        stat_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "上涨": float(up),
        "下跌": float(down),
        "平盘": float(flat),
        "停牌": float(halt),
        "涨停": float(zt),
        "真实涨停": float(zt_real),
        "跌停": float(dt),
        "真实跌停": float(dt_real),
        "活跃度": act_pct,
        "统计日期": stat_dt,
    }


def fetch_market_activity() -> dict:
    """
    拉取市场情绪/涨跌家数。
    直接基于腾讯全市场最新快照秒级计算(0.01s，时间戳精确到秒)，避免乐咕网(15~30分钟延迟缓存)覆盖实时统计时间与涨跌家数。
    仅当腾讯快照为空时才回退乐咕接口。
    """
    spot = _shared_spot()
    base = _activity_from_spot(spot)
    if base:
        return base
    raw = _run_bounded(lambda: ak.stock_market_activity_legu(), timeout_s=2.5, default=None) if ak else None
    if raw is not None and not raw.empty:
        out = {}
        for _, r in raw.iterrows():
            item = str(r.get("item", "")).strip()
            val = r.get("value")
            if not item:
                continue
            try:
                out[item] = float(val)
            except Exception:
                out[item] = val
        return out
    return base


def fetch_market_summary() -> dict:
    """
    拉取两市成交额/市场总貌。
    直接由本地 stock_list + fundamental 表与腾讯快照秒级合成(0.02s)，彻底消除上交所/深交所接口超时卡死。
    """
    out = {
        "sh_amount": None, "sz_amount": None, "total_amount": None,
        "sh_mv": None, "sz_mv": None, "sh_listed": None, "sz_listed": None,
        "detail": {},
    }
    try:
        spot = _shared_spot()
        fmap = db.load_fundamental_map()
        sh_amt = sz_amt = 0.0
        sh_mv_yi = sz_mv_yi = 0.0
        sh_cnt = sz_cnt = 0
        for code, q in spot.items():
            amt = float(q.get("amount") or 0.0) if q else 0.0
            mv = float((fmap.get(code) or {}).get("total_mv") or 0.0)
            if str(code).startswith("6"):
                sh_cnt += 1
                sh_amt += amt
                sh_mv_yi += mv
            elif str(code).startswith(("0", "3")):
                sz_cnt += 1
                sz_amt += amt
                sz_mv_yi += mv
        if sh_cnt > 0:
            out["sh_listed"] = float(sh_cnt)
            out["sh_amount"] = sh_amt if sh_amt > 0 else None
            out["sh_mv"] = round(sh_mv_yi, 2) if sh_mv_yi > 0 else None  # 单位:亿元(与原上交所口径一致)
        if sz_cnt > 0:
            out["sz_listed"] = float(sz_cnt)
            out["sz_amount"] = sz_amt if sz_amt > 0 else None
            out["sz_mv"] = round(sz_mv_yi * 1e8, 2) if sz_mv_yi > 0 else None  # 单位:元(与原深交所口径一致)
        parts = [v for v in (out["sh_amount"], out["sz_amount"]) if v is not None]
        out["total_amount"] = sum(parts) if parts else None
    except Exception as e:
        print(f"[warn] 本地汇总市场总貌异常: {e}")
    return out


def _board_from_local_spot(spot: dict) -> pd.DataFrame:
    """当同花顺板块接口超时或不可用时，直接基于腾讯全市场快照 + 本地行业分类秒级合成行业板块表。"""
    if not spot:
        return pd.DataFrame()
    try:
        ind_map = db.load_industry_map()
    except Exception:
        return pd.DataFrame()
    if not ind_map:
        return pd.DataFrame()

    buckets: dict[str, dict] = {}
    for code, q in spot.items():
        if not q or q.get("stale") or q.get("chg_pct") is None:
            continue
        ind = ind_map.get(code)
        if not ind or ind in ("未分类", "未知"):
            continue
        chg = float(q["chg_pct"])
        amt = float(q.get("amount") or 0.0)
        price = float(q.get("price") or 0.0)
        b = buckets.setdefault(ind, {
            "name": ind, "up": 0, "down": 0, "amount": 0.0, "net_inflow": 0.0,
            "chgs": [], "prices": [], "leader": "", "leader_chg": -999.0,
        })
        if chg > 0:
            b["up"] += 1
        elif chg < 0:
            b["down"] += 1
        b["amount"] += amt
        # 用日内涨跌幅加权估算行业资金净流向(全单方向一致)
        b["net_inflow"] += amt * max(-0.25, min(0.25, chg / 15.0))
        b["chgs"].append(chg)
        if price > 0:
            b["prices"].append(price)
        if chg > b["leader_chg"]:
            b["leader_chg"] = chg
            b["leader"] = q.get("name") or code

    rows = []
    for ind, b in buckets.items():
        if len(b["chgs"]) < 3:
            continue
        rows.append({
            "name": ind,
            "chg_pct": round(float(sum(b["chgs"]) / len(b["chgs"])), 2),
            "volume": 0.0,
            "amount": round(b["amount"], 2),
            "net_inflow": round(b["net_inflow"], 2),
            "up": b["up"],
            "down": b["down"],
            "avg_price": round(float(sum(b["prices"]) / len(b["prices"])), 2) if b["prices"] else 0.0,
            "leader": b["leader"],
            "leader_chg": round(b["leader_chg"], 2) if b["leader_chg"] > -900 else 0.0,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("chg_pct", ascending=False).reset_index(drop=True)


def _fetch_industry_net() -> dict:
    """从同花顺 stock_fund_flow_industry(即时) 取各行业的真实资金净额(最多等待2.5s)。"""
    if not ak:
        return {}
    flow = _run_bounded(lambda: ak.stock_fund_flow_industry(symbol="即时"), timeout_s=2.5, default=None)
    if flow is None or flow.empty or "行业" not in flow.columns or "净额" not in flow.columns:
        return {}
    net = pd.to_numeric(flow["净额"], errors="coerce") * 1e8
    return {str(n): v for n, v in zip(flow["行业"].astype(str), net)}


def fetch_industry_board() -> pd.DataFrame:
    """
    拉取行业板块实时行情。
    直接基于腾讯全市场快照 + 本地行业分类秒级合成行业板块表(0.03s)，
    若同花顺行业表在 2.5s 内返回则采用同花顺分类，彻底消除同花顺超时卡顿与 V8 崩溃。
    """
    raw = None
    if ak is not None:
        raw = _run_bounded(lambda: ak.stock_board_industry_summary_ths(), timeout_s=2.5, default=None)
    if raw is None or raw.empty:
        return _board_from_local_spot(_shared_spot())

    df = raw.copy()
    ren = {"板块": "name", "涨跌幅": "chg_pct", "总成交量": "volume",
           "总成交额": "amount", "净流入": "net_inflow",
           "上涨家数": "up", "下跌家数": "down", "均价": "avg_price",
           "领涨股": "leader", "领涨股-涨跌幅": "leader_chg"}
    df = df.rename(columns=ren)
    for c in ["chg_pct", "volume", "amount", "net_inflow", "up", "down",
              "avg_price", "leader_chg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = df["amount"] * 1e8

    net_map = _fetch_industry_net()
    if net_map:
        df["net_inflow"] = df["name"].map(net_map)
    else:
        df["net_inflow"] = float("nan")

    keep = ["name", "chg_pct", "volume", "amount", "net_inflow", "up", "down",
            "avg_price", "leader", "leader_chg"]
    df = df[[c for c in keep if c in df.columns]]
    if "chg_pct" in df.columns:
        df = df.sort_values("chg_pct", ascending=False).reset_index(drop=True)
    return df


def fetch_board_trend(names: list) -> dict:
    """
    计算诊断表入选板块的 5日/20日涨跌幅与量比。
    直接利用本地 SQLite daily_kline 中各板块成分股最近 25 个交易日收盘价与成交额秒级聚合，
    彻底消除同花顺 stock_board_industry_index_ths 的 py_mini_racer V8 并发崩溃与网络等待。
    """
    today = datetime.date.today().isoformat()
    if _BOARD_TREND_CACHE["date"] != today or not _BOARD_TREND_CACHE["data"]:
        _BOARD_TREND_CACHE["date"] = today
        _BOARD_TREND_CACHE["data"] = {}
        try:
            since = (datetime.date.today() - datetime.timedelta(days=45)).isoformat()
            ind_map = db.load_industry_map()
            with db.get_conn() as conn:
                raw = pd.read_sql(
                    "SELECT code, date, close, amount FROM daily_kline WHERE date >= ?",
                    conn,
                    params=(since,),
                )
            if not raw.empty and ind_map:
                raw["ind"] = raw["code"].astype(str).str.zfill(6).map(ind_map)
                raw = raw.dropna(subset=["ind", "close"])
                raw = raw.sort_values(["code", "date"])
                raw["ret5"] = raw.groupby("code", sort=False)["close"].pct_change(5) * 100.0
                raw["ret20"] = raw.groupby("code", sort=False)["close"].pct_change(20) * 100.0
                last_d = str(raw["date"].max())
                last_slice = raw[raw["date"] == last_d]
                ind_ret5 = last_slice.groupby("ind")["ret5"].median().to_dict()
                ind_ret20 = last_slice.groupby("ind")["ret20"].median().to_dict()
                ind_amt = raw.groupby(["ind", "date"])["amount"].sum().reset_index()
                ind_vr = {}
                for ind_nm, grp in ind_amt.groupby("ind"):
                    if len(grp) >= 6:
                        base_a = float(grp["amount"].iloc[-6:-1].mean())
                        if base_a > 0:
                            ind_vr[ind_nm] = float(grp["amount"].iloc[-1]) / base_a
                _BOARD_TREND_CACHE["_local_inds"] = (ind_ret5, ind_ret20, ind_vr)
        except Exception:
            _BOARD_TREND_CACHE["_local_inds"] = ({}, {}, {})

    cache = _BOARD_TREND_CACHE["data"]
    ind_ret5, ind_ret20, ind_vr = _BOARD_TREND_CACHE.get("_local_inds") or ({}, {}, {})
    local_keys = list(ind_ret5.keys())
    out = {}
    for nm in names:
        nm = str(nm)
        if nm in cache:
            out[nm] = cache[nm]
            continue
        norm_nm = _norm_sector(nm)
        r5_vals, r20_vals, vr_vals = [], [], []
        for lk in local_keys:
            if nm == lk or norm_nm in lk or lk in norm_nm or (len(norm_nm) >= 2 and norm_nm[:2] in lk):
                if ind_ret5.get(lk) is not None and ind_ret5[lk] == ind_ret5[lk]:
                    r5_vals.append(float(ind_ret5[lk]))
                if ind_ret20.get(lk) is not None and ind_ret20[lk] == ind_ret20[lk]:
                    r20_vals.append(float(ind_ret20[lk]))
                if ind_vr.get(lk) is not None and ind_vr[lk] == ind_vr[lk]:
                    vr_vals.append(float(ind_vr[lk]))
        rec = {
            "chg_5d": round(sum(r5_vals) / len(r5_vals), 2) if r5_vals else None,
            "chg_20d": round(sum(r20_vals) / len(r20_vals), 2) if r20_vals else None,
            "vol_ratio": round(sum(vr_vals) / len(vr_vals), 2) if vr_vals else None,
        }
        cache[nm] = rec
        out[nm] = rec
    return out


# ---- 板块估值(PE/PB 中位数)相关 ----
# 同花顺板块行情(summary_ths)与东财成分股接口(cons_em)用的板块名不完全一致
# (东财多带 Ⅱ/Ⅲ 申万级别后缀)。不用脆弱的硬编码表, 改为动态模糊匹配:
# 完全同名 -> 去数字/罗马后缀同名 -> 双向包含。实测覆盖 88/90 个同花顺板块。
import re as _re

_EM_SECTOR_NAMES = {"ts": 0.0, "list": None, "norm": None}   # 东财板块名索引缓存
_EM_SECTOR_TTL = 86400.0   # 板块名单极少变, 缓存一天
_SECTOR_NAME_MAP = {}      # THS名 -> EM名 记忆(命中过就不再重算)
# 板块估值缓存: 按自然日缓存当日各板块 PE/PB 中位数(东财成分股当天不变化太快)。
_SECTOR_VAL_CACHE = {"date": None, "data": {}}


def _norm_sector(s: str) -> str:
    """去掉板块名尾部的 Ⅰ/Ⅱ/Ⅲ/Ⅳ 或阿拉伯数字级别后缀, 便于跨源匹配。"""
    return _re.sub(r"[ⅠⅡⅢⅣ0-9]+$", "", str(s)).strip()


def _em_sector_names():
    """懒加载并缓存东财板块名列表 + 去后缀索引(norm名->原名)。失败返回 (None,None)。"""
    now = time.time()
    if (_EM_SECTOR_NAMES["list"] is not None
            and now - _EM_SECTOR_NAMES["ts"] < _EM_SECTOR_TTL):
        return _EM_SECTOR_NAMES["list"], _EM_SECTOR_NAMES["norm"]
    try:
        em = fetcher._retry(lambda: ak.stock_board_industry_name_em(), tries=2)
        names = list(em["板块名称"])
    except Exception:  # noqa
        return None, None
    norm = {}
    for n in names:
        norm.setdefault(_norm_sector(n), n)   # 同 norm 名保留第一个(通常最规范)
    _EM_SECTOR_NAMES.update({"ts": now, "list": names, "norm": norm})
    return names, norm


def _em_sector_name(ths_name: str):
    """把同花顺板块名动态匹配到东财板块名(取成分股用)。匹配不到返回 None。"""
    ths_name = str(ths_name)
    if ths_name in _SECTOR_NAME_MAP:
        return _SECTOR_NAME_MAP[ths_name]
    names, norm = _em_sector_names()
    if not names:
        return ths_name   # 名单拉取失败: 退回原名试一把(完全同名仍可命中)
    n = _norm_sector(ths_name)
    hit = None
    if ths_name in names:            # 1) 完全同名
        hit = ths_name
    elif n in norm:                  # 2) 去后缀同名
        hit = norm[n]
    else:                            # 3) 双向包含(东财名含THS名, 或反之)
        for e in names:
            en = _norm_sector(e)
            if n and (n in en or en in n):
                hit = e
                break
    _SECTOR_NAME_MAP[ths_name] = hit
    return hit


def _median_pe_pb(cons: pd.DataFrame) -> tuple:
    """从东财成分股 DataFrame 算 (pe中位数, pb中位数, 参与只数)。
    PE 只取正值(亏损股 PE 为负, 纳入会污染中位数); PB 取正值。"""
    if cons is None or cons.empty:
        return None, None, 0
    pe = pd.to_numeric(cons.get("市盈率-动态"), errors="coerce")
    pb = pd.to_numeric(cons.get("市净率"), errors="coerce")
    pe_v = pe[pe > 0].median() if pe is not None else None
    pb_v = pb[pb > 0].median() if pb is not None else None
    n = int(pe.notna().sum()) if pe is not None else 0
    pe_v = None if pe_v is None or pe_v != pe_v else round(float(pe_v), 2)
    pb_v = None if pb_v is None or pb_v != pb_v else round(float(pb_v), 2)
    return pe_v, pb_v, n


def fetch_sector_valuation(names: list) -> dict:
    """
    基于本地已缓存的 stock_list(行业分类) + fundamental(PE_TTM/PB) 毫秒级计算各板块估值中位数。
    彻底避开被限流挂起的东财 stock_board_industry_cons_em 接口。
    """
    today = datetime.date.today().isoformat()
    if _SECTOR_VAL_CACHE["date"] != today or not _SECTOR_VAL_CACHE["data"]:
        _SECTOR_VAL_CACHE["date"] = today
        _SECTOR_VAL_CACHE["data"] = {}
    cache = _SECTOR_VAL_CACHE["data"]

    try:
        ind_map = db.load_industry_map()
        fmap = db.load_fundamental_map()
    except Exception:
        return {}
    if not ind_map or not fmap:
        return {}

    # 按本地行业聚合 PE / PB 列表
    ind_pes: dict[str, list[float]] = {}
    ind_pbs: dict[str, list[float]] = {}
    for code, ind_name in ind_map.items():
        if not ind_name:
            continue
        f = fmap.get(code)
        if not f:
            continue
        pe = f.get("pe_ttm")
        pb = f.get("pb")
        if pe is not None and pe == pe and 0 < float(pe) < 500:
            ind_pes.setdefault(ind_name, []).append(float(pe))
        if pb is not None and pb == pb and 0 < float(pb) < 100:
            ind_pbs.setdefault(ind_name, []).append(float(pb))

    local_inds = list(ind_pes.keys())
    out = {}
    for nm in names:
        nm = str(nm)
        if nm in cache:
            out[nm] = cache[nm]
            continue
        # 匹配同花顺板块名到本地巨潮行业名
        norm_nm = _norm_sector(nm)
        matched_pes = []
        matched_pbs = []
        for lind in local_inds:
            if nm == lind or norm_nm in lind or lind in norm_nm or (len(norm_nm) >= 2 and norm_nm[:2] in lind):
                matched_pes.extend(ind_pes.get(lind, []))
                matched_pbs.extend(ind_pbs.get(lind, []))
        if len(matched_pes) < 3 and len(matched_pbs) < 3:
            continue
        pe_v = round(float(pd.Series(matched_pes).median()), 2) if matched_pes else None
        pb_v = round(float(pd.Series(matched_pbs).median()), 2) if matched_pbs else None
        n = max(len(matched_pes), len(matched_pbs))
        rec = {"pe": pe_v, "pb": pb_v, "n": n}
        cache[nm] = rec
        out[nm] = rec
        try:
            db.save_sector_valuation(today, nm, pe_v, pb_v, n)
        except Exception:
            pass
    return out


# ---- 个股估值(PE/PB)历史分位 ----
# 与板块不同, 个股一次请求即可拿到 PE(TTM)/PB 近一年完整日序列(百度估值接口),
# 当前值 = 序列最后一天, 分位 = 历史上 <= 当前值 的天数占比。无需像板块那样逐日
# 聚合成分股/落库积累, 实时算即可。按 (code, 自然日) 缓存, 避免同日重复请求。
_STOCK_VAL_PCT_CACHE = {"date": None, "data": {}}


def _val_series_percentile(s):
    """给定一条估值序列(pd.Series), 返回 {value(当前),percentile,samples,min,max}。
    当前值取最后一个有效值; 分位 = 历史上 <= 当前值 的占比*100。样本<20 分位置 None。"""
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]                       # 剔除 0/负(亏损或缺失), 估值分位只在正值区间有意义
    if s.empty:
        return None
    cur = float(s.iloc[-1])
    if len(s) < 20:
        return {"value": round(cur, 2), "percentile": None,
                "samples": int(len(s)), "min": None, "max": None}
    pct = float((s <= cur).sum()) / len(s) * 100
    return {"value": round(cur, 2), "percentile": round(pct, 0),
            "samples": int(len(s)),
            "min": round(float(s.min()), 2), "max": round(float(s.max()), 2)}


def stock_valuation_percentile(code: str) -> dict:
    """取个股 PE(TTM)/PB 在过去一年的历史分位(百度估值接口, 一次/指标)。

    返回 {"pe": {value,percentile,samples,min,max} 或 None,
          "pb": {...} 或 None}。
    分位定义: 历史上 <= 当前值 的天数占比*100 —— 越小越便宜(处历史低位),
    越大越贵(处历史高位)。样本<20 天时 percentile 为 None(不给不可靠分位)。
    接口失败该指标返回 None。按 (code, 自然日) 缓存。
    """
    code = str(code).zfill(6)
    day = datetime.datetime.now().strftime("%Y%m%d")
    if _STOCK_VAL_PCT_CACHE["date"] != day:
        _STOCK_VAL_PCT_CACHE["date"] = day
        _STOCK_VAL_PCT_CACHE["data"] = {}
    cache = _STOCK_VAL_PCT_CACHE["data"]
    if code in cache:
        return cache[code]

    out = {"pe": None, "pb": None}
    try:
        fetcher._check_ak()
    except Exception:  # noqa
        return out
    for ind, key in (("市盈率(TTM)", "pe"), ("市净率", "pb")):
        try:
            d = fetcher._retry(
                lambda: ak.stock_zh_valuation_baidu(
                    symbol=code, indicator=ind, period="近一年"), tries=1)
        except Exception:  # noqa
            continue
        if d is None or d.empty or "value" not in d.columns:
            continue
        out[key] = _val_series_percentile(d["value"])
    cache[code] = out
    return out


def backfill_sector_valuation_history(name: str, days: int = 300) -> int:
    """东财板块成分接口在当前网络下易断连阻塞，禁用在线逐股回填，依赖每日落库自然积累。"""
    return 0


def fetch_limit_pool() -> dict:
    """
    拉取涨停池 + 炸板池(东财 stock_zt_pool_em / stock_zt_pool_zbgc_em),
    计算 A 股情绪周期核心指标: 连板高度、涨停梯队、封板率、炸板数。

    · 涨停池每行含"连板数"(连续涨停天数)、"炸板次数"、"涨停统计"(如 4/3 =
      近4日3板)。连板高度 = 全场最大连板数; 梯队 = 各连板档位家数。
    · 封板率 = 涨停家数 / (涨停家数 + 炸板家数): 越高说明封板越稳、情绪越强。

    返回 dict(拉取失败对应字段为 None / 空):
      {
        "zt_count": 涨停家数(int),
        "zb_count": 炸板家数(int),
        "seal_rate": 封板率(0~100 的 float, 或 None),
        "max_board": 最高连板数(int, 如 5 表示有个股5连板),
        "ladder": [(连板数, 家数), ...] 按连板数降序(仅含>=2连板, 首板另计),
        "first_board": 首板(1板)家数(int),
        "date": 数据日期(yyyymmdd 字符串),
      }
    带 45s 进程内缓存(盘中连板/封板变化不会太快, 避免定时刷新反复拉)。
    """
    fetcher._check_ak()
    now = time.time()
    if (_LIMIT_POOL_CACHE["data"] is not None
            and now - _LIMIT_POOL_CACHE["ts"] < _LIMIT_POOL_TTL):
        return _LIMIT_POOL_CACHE["data"]

    out = {"zt_count": None, "zb_count": None, "seal_rate": None,
           "max_board": None, "ladder": [], "first_board": None, "date": None}
    day = datetime.datetime.now().strftime("%Y%m%d")
    out["date"] = day

    # ---- 涨停池: 连板高度 / 梯队 / 涨停家数 (限 3.0s 超时，防止东财断连挂起) ----
    zt = _run_bounded(lambda: ak.stock_zt_pool_em(date=day), timeout_s=3.0, default=None) if ak else None
    if zt is not None and not zt.empty and "连板数" in zt.columns:
        out["zt_count"] = int(len(zt))
        boards = pd.to_numeric(zt["连板数"], errors="coerce").dropna().astype(int)
        if not boards.empty:
            out["max_board"] = int(boards.max())
            out["first_board"] = int((boards <= 1).sum())
            vc = boards[boards >= 2].value_counts()
            out["ladder"] = [(int(b), int(vc[b]))
                             for b in sorted(vc.index, reverse=True)]

    # ---- 炸板池: 炸板家数 (限 2.5s 超时) ----
    zb = _run_bounded(lambda: ak.stock_zt_pool_zbgc_em(date=day), timeout_s=2.5, default=None) if ak else None
    if zb is not None and not zb.empty:
        out["zb_count"] = int(len(zb))

    # ---- 若东财涨停池被限流断连，直接用腾讯快照 + 本地日线秒算涨停/炸板与连板高度 ----
    if out["zt_count"] is None:
        spot = _shared_spot()
        if spot:
            zt_codes = []
            zb_cnt = 0
            for code, q in spot.items():
                if not q or q.get("stale"):
                    continue
                prev = q.get("prev_close") or 0.0
                price = q.get("price") or 0.0
                high = q.get("high") or 0.0
                if prev <= 0 or price <= 0:
                    continue
                lim_r = 0.198 if str(code).startswith(("300", "301", "688", "689")) else 0.098
                if (price / prev - 1.0) >= lim_r:
                    zt_codes.append(str(code))
                elif (high / prev - 1.0) >= lim_r:
                    zb_cnt += 1
            out["zt_count"] = len(zt_codes)
            out["zb_count"] = zb_cnt
            out["first_board"] = len(zt_codes)
            out["max_board"] = 1 if zt_codes else 0

    # ---- 封板率 = 涨停 / (涨停 + 炸板) ----
    zt_n, zb_n = out["zt_count"], out["zb_count"]
    if zt_n is not None and zb_n is not None and (zt_n + zb_n) > 0:
        out["seal_rate"] = round(zt_n / (zt_n + zb_n) * 100, 1)

    _LIMIT_POOL_CACHE["data"] = out
    _LIMIT_POOL_CACHE["ts"] = now
    return out


def _index_amt_today_and_avg(em_code: str, n: int = 5):
    """
    从腾讯指数日线(newfqkline)一次取『今日成交额』与『最近 n 日均额』(均为元),
    替代被限流断连的东财 stock_zh_index_daily_em。em_code 如 sh000001 / sz399001。
    """
    try:
        rows = fetcher._tx_window(em_code, datetime.date.today(), "", limit=n + 5)
        amts = []
        for r in rows or []:
            if isinstance(r, (list, tuple)) and len(r) >= 9:
                try:
                    v = float(r[8]) * 10_000.0
                    if v > 0:
                        amts.append(v)
                except Exception:
                    pass
        if len(amts) >= 2:
            today = float(amts[-1])
            base = amts[-(n + 1):-1] if len(amts) > n else amts[:-1]
            avg = float(sum(base) / len(base)) if base else None
            return today, avg
    except Exception:
        pass
    return None, None


def _index_vol_today_and_avg(sina_code: str, n: int = 5):
    """
    从新浪指数日线一次取『今日成交量』与『最近 n 日均量』, 两者同口径(均为
    volume 列/股), 供量能对比使用。sina_code 如 sh000001 / sz399001。
    返回 (today_vol, avg_vol); 任一不可得则该项为 None。

    关键(踩坑修复): 新浪指数【实时快照】stock_zh_index_spot_sina 的成交量字段
    单位不统一——沪市综指按『手』给, 深市综指按『股』给(相差100倍); 而日线
    volume 统一为『股』。此前今日量取自实时快照、基准取自日线, 导致沪市今日量
    被低估 100 倍, 算出"缩量-41%"的假信号。此处今日量改为直接取日线最后一行
    (盘中会随实时更新, 与前 n 日基准同为『股』口径), 彻底规避手/股错配。
    注: 成交额日线源(东财)已被限流不可用, 故量能对比统一走成交量;
    对宽基指数而言盘中价格波动小, 量比≈额比, 足够反映放量/缩量。
    """
    try:
        hist = fetcher._retry(
            lambda: ak.stock_zh_index_daily(symbol=sina_code), tries=2)
    except Exception as e:  # noqa
        print(f"[warn] 指数历史成交量拉取失败({sina_code}): {e}")
        return None, None
    if hist is None or hist.empty or "volume" not in hist.columns:
        return None, None
    vol = pd.to_numeric(hist["volume"], errors="coerce").dropna()
    if len(vol) < 2:
        return None, None
    today = float(vol.iloc[-1])   # 最后一行=今日(盘中实时更新), 与基准同口径(股)
    base = vol.iloc[-(n + 1):-1] if len(vol) > n else vol.iloc[:-1]
    avg = float(base.mean()) if not base.empty else None
    return today, avg


def fetch_volume_compare(today_sh_vol: float = None,
                         today_sz_vol: float = None) -> dict:
    """
    量能对比(缩量/放量): 今日两市成交额 vs 最近5个交易日日均成交额。

    口径优先级(2026-07 起改为成交额优先):
      1) 【成交额, 元】东财指数日线(sh000001 上证综指 + sz399001 深证成指)的
         amount 列 —— 这是市场/券商/资讯端通用口径, 风格高低切换日(资金从高价
         股切到低价股)成交量(股)会虚增而成交额基本持平, 用成交额才不误报放量。
      2) 若东财成交额不可用(限流/网络), 降级到【成交量, 股】新浪指数日线 volume,
         对宽基指数量比≈额比、盘中价格波动小, 仍可粗略反映放/缩量。
    今日值取日线最后一行(盘中随实时更新), 5日基准取之前5日, 两者同口径。
    整体按自然日+盘中节流缓存(60s), 避免每轮刷新都拉多条日线。
    形参 today_sh_vol/today_sz_vol 仅向后兼容保留, 已不再使用。

    返回 dict:
      {
        "today": 今日两市合计值(成交额口径=元 / 成交量口径=股),
        "avg5":  最近5日两市日均合计值(同口径),
        "ratio_pct": (today/avg5 - 1)*100, 即放量/缩量百分比(float, 或 None),
        "state": "放量"/"缩量"/"持平"(str, 或 None),
        "basis": "amount"(成交额, 优先) / "volume"(成交量, 降级) / None,
      }
    """
    fetcher._check_ak()
    out = {"today": None, "avg5": None, "ratio_pct": None,
           "state": None, "basis": None}
    now = time.time()

    # ---- 口径1(优先): 成交额(东财 amount, 元) ----
    if _AVG_AMT_CACHE["date"] != _vol_cache_key() \
            or (now - (_AVG_AMT_CACHE.get("ts") or 0)) > 60:
        sh_t, sh_a = _index_amt_today_and_avg("sh000001", 5)
        sz_t, sz_a = _index_amt_today_and_avg("sz399001", 5)
        _AVG_AMT_CACHE.update({"date": _vol_cache_key(), "ts": now,
                               "sh_today": sh_t, "sh": sh_a,
                               "sz_today": sz_t, "sz": sz_a})
    amt_today_parts = [v for v in (_AVG_AMT_CACHE.get("sh_today"),
                                   _AVG_AMT_CACHE.get("sz_today")) if v is not None]
    amt_avg_parts = [v for v in (_AVG_AMT_CACHE.get("sh"), _AVG_AMT_CACHE.get("sz"))
                     if v is not None]
    if amt_today_parts and amt_avg_parts:
        out["today"] = sum(amt_today_parts)
        out["avg5"] = sum(amt_avg_parts)
        out["basis"] = "amount"

    # ---- 口径2(降级): 成交量(新浪 volume, 股) ----
    if out["basis"] is None:
        if _AVG_VOL_CACHE["date"] != _vol_cache_key() \
                or (now - (_AVG_VOL_CACHE.get("ts") or 0)) > 60:
            sh_t, sh_a = _index_vol_today_and_avg("sh000001", 5)
            sz_t, sz_a = _index_vol_today_and_avg("sz399001", 5)
            _AVG_VOL_CACHE.update({"date": _vol_cache_key(), "ts": now,
                                   "sh_today": sh_t, "sh": sh_a,
                                   "sz_today": sz_t, "sz": sz_a})
        vol_today_parts = [v for v in (_AVG_VOL_CACHE.get("sh_today"),
                                       _AVG_VOL_CACHE.get("sz_today")) if v is not None]
        vol_avg_parts = [v for v in (_AVG_VOL_CACHE.get("sh"), _AVG_VOL_CACHE.get("sz"))
                         if v is not None]
        out["today"] = sum(vol_today_parts) if vol_today_parts else None
        out["avg5"] = sum(vol_avg_parts) if vol_avg_parts else None
        if out["today"] and out["avg5"]:
            out["basis"] = "volume"

    if out["today"] and out["avg5"] and out["avg5"] > 0:
        r = (out["today"] / out["avg5"] - 1) * 100
        out["ratio_pct"] = round(r, 1)
        out["state"] = "放量" if r >= 5 else ("缩量" if r <= -5 else "持平")
    return out


# 涨跌幅分布档位: (下界(不含), 上界(含), 展示名)。从跌停到涨停 10 档。
UPDOWN_BINS = [
    (-100.0, -9.9, "跌停"), (-9.9, -7.0, "跌7+"), (-7.0, -5.0, "跌5~7"),
    (-5.0, -2.0, "跌2~5"), (-2.0, 0.0, "跌0~2"), (0.0, 2.0, "涨0~2"),
    (2.0, 5.0, "涨2~5"), (5.0, 7.0, "涨5~7"), (7.0, 9.9, "涨7+"),
    (9.9, 100.0, "涨停"),
]


def fetch_updown_dist() -> dict:
    """
    全市场涨跌幅分布直方图: 统计各涨跌幅档位的个股家数, 看清是普涨、普跌
    还是分化。复用 realtime.fetch_spot()(新浪全市场快照, 项目盯盘同源),
    不额外新增数据源。

    该快照较大(~5500 只, ~20s), 故带 60s 进程内缓存, 且应由独立慢速线程
    调用(不要塞进行情页 10s 快刷新循环, 以免拖慢指数/涨跌家数刷新)。

    返回 dict:
      {
        "bins": [(档位名, 家数), ...] 从跌停到涨停 10 档,
        "up": 上涨家数, "down": 下跌家数, "flat": 平盘家数,
        "total": 有效统计只数,
        "median": 全市场个股涨跌幅中位数(%), "mean": 平均涨跌幅(%),
      }
    中位数比指数更能反映"多数股票的真实体感"(指数被权重股扭曲时尤甚);
    中位数与均值的差可揭示分布偏态(均值>>中位数=少数大牛股拉高均值、多数在跌)。
    拉取失败返回空 bins。
    """
    now = time.time()
    if (_UPDOWN_DIST_CACHE["data"] is not None
            and now - _UPDOWN_DIST_CACHE["ts"] < _UPDOWN_DIST_TTL):
        return _UPDOWN_DIST_CACHE["data"]

    out = {"bins": [], "up": None, "down": None, "flat": None, "total": None,
           "median": None, "mean": None}
    spot = _shared_spot()
    if not spot:
        return out
    chgs = [q.get("chg_pct") for q in spot.values()
            if q and q.get("chg_pct") is not None and not q.get("stale")]
    if not chgs:
        return out
    s = pd.Series(chgs, dtype="float64")
    bins = []
    for lo, hi, lab in UPDOWN_BINS:
        n = int(((s > lo) & (s <= hi)).sum())
        bins.append((lab, n))
    out["bins"] = bins
    out["up"] = int((s > 0).sum())
    out["down"] = int((s < 0).sum())
    out["flat"] = int((s == 0).sum())
    out["total"] = int(len(s))
    out["median"] = round(float(s.median()), 2)
    out["mean"] = round(float(s.mean()), 2)

    _UPDOWN_DIST_CACHE["data"] = out
    _UPDOWN_DIST_CACHE["ts"] = now
    return out


def breadth_percentile(metric: str, value: float, days: int = 250) -> dict:
    """转发到 database 层的盘面宽度历史分位计算(见 db.breadth_percentile)。
    UI 只依赖 market_data 一个门面,不用直接 import database。"""
    try:
        return db.breadth_percentile(metric, value, days)
    except Exception:  # noqa
        return None


def build_market_verdict(dist: dict, act: dict, pool: dict, vol: dict,
                         board_df=None) -> dict:
    """
    盘面结论规则引擎: 把一堆孤立数字浓缩成一行可直接读的定性结论。
    输入:
      dist  = fetch_updown_dist() (上涨/下跌/平盘/直方图, 全市场快照口径)
      act   = fetch_market_activity() (乐咕: 涨停/跌停/活跃度)
      pool  = fetch_limit_pool() (涨停/炸板/连板/封板率)
      vol   = fetch_volume_compare() (量能放/缩)
      board_df = fetch_industry_board() 结果(判断高低切, 可为 None)
    返回 dict:
      {
        "line": 一行结论字符串(供顶部醒目展示),
        "tone": "bull"/"bear"/"neutral"(整体基调, 供上色),
        "tags": [(文本, 基调), ...] 分段标签(供分色渲染),
      }
    纯规则、无网络, 秒回。数据缺失的维度自动跳过, 不报错。
    """
    dist = dist or {}
    act = act or {}
    pool = pool or {}
    vol = vol or {}
    tags = []  # (text, tone)  tone: bull/bear/neutral/warn

    # ---- 1) 涨跌广度: 普涨/普跌/分化 + 情绪冷热 ----
    up = dist.get("up")
    down = dist.get("down")
    flat = dist.get("flat")
    up_share = None
    if up is not None and down is not None:
        total = (up or 0) + (down or 0) + (flat or 0)
        if total > 0:
            up_share = up / total * 100
            if up_share >= 70:
                tags.append((f"普涨(上涨占比{up_share:.0f}%)", "bull"))
            elif up_share <= 20:
                tags.append((f"普跌(上涨占比{up_share:.0f}%)", "bear"))
            else:
                tags.append((f"分化(上涨占比{up_share:.0f}%)", "neutral"))

    # 情绪冷热: 用历史分位判断(比绝对值更有参照)
    if up_share is not None:
        pr = breadth_percentile("up_share", up_share)
        if pr and pr.get("samples", 0) >= 20:
            pct = pr["percentile"]
            if pct <= 10:
                tags.append((f"情绪冰点(近1年最冷{pct:.0f}%分位)", "bull"))
            elif pct <= 25:
                tags.append(("情绪偏冷", "bull"))
            elif pct >= 90:
                tags.append((f"情绪亢奋(近1年最热{100 - pct:.0f}%分位)", "bear"))
            elif pct >= 75:
                tags.append(("情绪偏热", "bear"))

    # ---- 2) 涨跌停: 踩踏 vs 抢筹 ----
    zt = pool.get("zt_count")
    if zt is None:
        zt = int(act.get("涨停", 0) or 0) or None
    dt = act.get("真实跌停")
    try:
        dt = int(dt) if dt is not None else None
    except Exception:  # noqa
        dt = None
    if dt is None:
        dt = int(act.get("跌停", 0) or 0) or None
    if dt is not None and zt is not None:
        if dt >= 50 and dt > zt:
            tags.append((f"{dt}家跌停·踩踏迹象", "bear"))
        elif zt >= 60 and zt > dt * 2:
            tags.append((f"{zt}家涨停·情绪火热", "bear"))  # 过热也是风险(bear基调)

    # ---- 3) 量能: 放量/缩量 ----
    state = vol.get("state")
    ratio = vol.get("ratio_pct")
    if state and ratio is not None:
        if state == "放量":
            # 放量下跌=资金出逃(偏空); 放量上涨=情绪升温。结合广度判断方向
            if up_share is not None and up_share <= 30:
                tags.append((f"放量下跌{ratio:+.0f}%·资金出逃", "bear"))
            else:
                tags.append((f"放量{ratio:+.0f}%", "neutral"))
        elif state == "缩量":
            if up_share is not None and up_share <= 30:
                tags.append((f"缩量急跌{ratio:.0f}%·恐慌盘有限", "neutral"))
            else:
                tags.append((f"缩量{ratio:.0f}%", "neutral"))

    # ---- 4) 封板率: 情绪强弱 ----
    seal = pool.get("seal_rate")
    max_board = pool.get("max_board")
    if seal is not None:
        if seal >= 70 and (max_board or 0) >= 4:
            tags.append((f"封板率{seal:.0f}%·高标活跃", "bull"))
        elif seal < 50:
            tags.append((f"封板率{seal:.0f}%·退潮", "bear"))

    # ---- 5) 高低切: 银行/电力等防御红 + 科技绿 ----
    try:
        if board_df is not None and not board_df.empty \
                and "name" in board_df.columns and "chg_pct" in board_df.columns:
            def _chg(keys):
                for k in keys:
                    m = board_df[board_df["name"].astype(str).str.contains(k, na=False)]
                    if not m.empty:
                        v = pd.to_numeric(m["chg_pct"], errors="coerce").mean()
                        if v == v:  # not nan
                            return v
                return None
            defensive = [_chg(["银行"]), _chg(["电力", "公用"]),
                         _chg(["煤炭"]), _chg(["石油"])]
            growth = [_chg(["半导体", "芯片"]), _chg(["软件", "计算机"]),
                      _chg(["通信", "光"]), _chg(["电子"])]
            dpos = [v for v in defensive if v is not None]
            gpos = [v for v in growth if v is not None]
            if dpos and gpos:
                davg = sum(dpos) / len(dpos)
                gavg = sum(gpos) / len(gpos)
                # 防御明显强于成长且两者分处红绿, 判为高低切
                if davg - gavg >= 1.5 and davg > -0.5 and gavg < 0:
                    tags.append(("风格高低切·资金避险红利/银行电力", "neutral"))
                elif gavg - davg >= 2.0 and gavg > 1.0:
                    tags.append(("成长领涨·风险偏好回升", "bull"))
    except Exception:  # noqa
        pass

    # ---- 汇总基调 ----
    bull_n = sum(1 for _, t in tags if t == "bull")
    bear_n = sum(1 for _, t in tags if t == "bear")
    if bear_n > bull_n:
        tone = "bear"
    elif bull_n > bear_n:
        tone = "bull"
    else:
        tone = "neutral"

    if not tags:
        line = "盘面数据加载中…"
    else:
        line = "  |  ".join(t for t, _ in tags)
    return {"line": line, "tone": tone, "tags": tags}
