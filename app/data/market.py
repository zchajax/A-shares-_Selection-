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
import time

import pandas as pd

from . import fetcher  # 复用 _check_ak / _retry / 全局超时设置

try:
    import akshare as ak
except ImportError:
    ak = None

socket.setdefaulttimeout(15)

# ---- 进程内 TTL 缓存: 部分市场数据拉取偏慢(涨停池~4s/全市场快照~23s),
# 定时刷新时若每 10s 都重拉会拖慢整个行情页,故各自加缓存,过期才重拉。 ----
_LIMIT_POOL_CACHE = {"ts": 0.0, "data": None}
_LIMIT_POOL_TTL = 45.0            # 涨停池: 盘中封板/连板变化不会太快
_UPDOWN_DIST_CACHE = {"ts": 0.0, "data": None}
_UPDOWN_DIST_TTL = 60.0           # 涨跌幅分布: 全市场快照慢, 放宽到 60s
_AVG_VOL_CACHE = {"date": None, "sh": None, "sz": None}   # 5日均量(按自然日缓存)

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
    拉取核心指数实时行情(新浪源,一次全市场后按白名单筛选)。
    返回 DataFrame 列: code,name,price,chg_pct,chg_amt,pre_close,open,high,low,
    amount(成交额,元)。按 CORE_INDEXES 顺序排列;拉取失败返回空 DataFrame。

    附带(df.attrs, 复用同一次全量抓取, 不额外请求网络):
      - sh_amount: 沪市成交额(上证综指 sh000001 成交额, 元)
      - sz_amount: 深市成交额(深证综指 sz399106 成交额, 元)
      - total_amount: 两市合计成交额(元)
    这是"两市成交额"最标准的口径(沪市综指 + 深市综指)。
    """
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


def fetch_market_activity() -> dict:
    """
    拉取市场情绪/涨跌家数(乐咕 stock_market_activity_legu)。
    返回扁平 dict, 键为中文项目名(上涨/下跌/涨停/跌停/真实涨停/真实跌停/
    st st*涨停/st st*跌停/平盘/停牌/活跃度 等), 值为 float。
    拉取失败返回空 dict。
    """
    fetcher._check_ak()
    try:
        raw = fetcher._retry(lambda: ak.stock_market_activity_legu(), tries=2)
    except Exception as e:  # noqa
        print(f"[warn] 市场情绪拉取失败: {e}")
        return {}
    if raw is None or raw.empty:
        return {}
    out = {}
    for _, r in raw.iterrows():
        item = str(r.get("item", "")).strip()
        val = r.get("value")
        if not item:
            continue
        # "活跃度" 是形如 "50%" 的字符串, 其余为数值
        try:
            out[item] = float(val)
        except Exception:  # noqa
            out[item] = val
    return out


def fetch_market_summary() -> dict:
    """
    拉取两市成交额/市场总貌(上交所 + 深交所)。
    返回 dict:
      {
        "sh_amount": 沪市股票成交额(元, 主板+科创),
        "sz_amount": 深市股票成交额(元),
        "total_amount": 两市合计成交额(元),
        "sh_mv": 沪市总市值(元), "sz_mv": 深市总市值(元),
        "sh_listed": 沪市上市公司数, "sz_listed": 深市股票数,
        "detail": {原始明细, 供需要时展开},
      }
    任一交易所失败其对应字段为 None, 不影响另一边。
    注: sse_summary 金额单位为"亿元"(需 ×1e8 转元); szse_summary 金额单位为"元"。
    """
    fetcher._check_ak()
    out = {"sh_amount": None, "sz_amount": None, "total_amount": None,
           "sh_mv": None, "sz_mv": None, "sh_listed": None, "sz_listed": None,
           "detail": {}}

    # ---- 上交所: 项目 / 股票 / 主板 / 科创板 (金额单位: 亿元) ----
    try:
        sse = fetcher._retry(lambda: ak.stock_sse_summary(), tries=2)
        if sse is not None and not sse.empty:
            s = sse.set_index("项目")
            def _sse(item, col="股票"):
                try:
                    return float(s.loc[item, col])
                except Exception:  # noqa
                    return None
            out["sh_mv"] = _sse("总市值")            # 亿元
            out["sh_listed"] = _sse("上市公司")
            out["detail"]["sse"] = sse.to_dict("records")
    except Exception as e:  # noqa
        print(f"[warn] 上交所总貌拉取失败: {e}")

    # ---- 深交所: 证券类别 / 数量 / 成交金额 / 总市值 / 流通市值 (元) ----
    try:
        szse = fetcher._retry(lambda: ak.stock_szse_summary(), tries=2)
        if szse is not None and not szse.empty:
            z = szse.set_index("证券类别")
            def _szse(cat, col):
                try:
                    return float(z.loc[cat, col])
                except Exception:  # noqa
                    return None
            out["sz_amount"] = _szse("股票", "成交金额")   # 元
            out["sz_mv"] = _szse("股票", "总市值")
            out["sz_listed"] = _szse("股票", "数量")
            out["detail"]["szse"] = szse.to_dict("records")
    except Exception as e:  # noqa
        print(f"[warn] 深交所总貌拉取失败: {e}")

    # 沪市成交额: sse_summary 无直接成交额列, 从深市能拿到元级成交额;
    # 沪市成交额改由指数成交额兜底(在 UI 层用上证指数 amount 近似)不在此处强求。
    # 两市合计: 仅当两边都有元级成交额时才计算; 沪市此处通常缺, 由 UI 结合指数处理。
    parts = [v for v in (out["sh_amount"], out["sz_amount"]) if v is not None]
    out["total_amount"] = sum(parts) if parts else None
    return out


def _fetch_industry_net() -> dict:
    """
    从同花顺 stock_fund_flow_industry(即时) 取各行业的真实资金净额。
    返回 {板块名: 净额(元)}, 失败返回空 dict。

    净额 = 流入资金 − 流出资金(全单口径: 几乎覆盖全部成交, 双边合计≈总成交额)。
    数值方向正确(与涨跌大势、券商 App 同向), 但因是全单轧差而非"主力资金"
    (大单+超大单)口径, 绝对值会明显小于东财/券商的主力净流入; 这是口径差异,
    非错误。原 summary_ths 的"净流入"列方向都不可信, 故弃用改由此覆盖。
    单位: 接口为"亿元", 此处 ×1e8 统一到元。
    """
    try:
        flow = fetcher._retry(lambda: ak.stock_fund_flow_industry(symbol="即时"),
                              tries=2)
    except Exception as e:  # noqa
        print(f"[warn] 行业资金净额拉取失败: {e}")
        return {}
    if flow is None or flow.empty or "行业" not in flow.columns \
            or "净额" not in flow.columns:
        return {}
    net = pd.to_numeric(flow["净额"], errors="coerce") * 1e8
    return {str(n): v for n, v in zip(flow["行业"].astype(str), net)}


def fetch_industry_board() -> pd.DataFrame:
    """
    拉取行业板块实时行情(同花顺, 两接口合并)。
    实测稳定(~4s), 避开被限流的东财源, 一次返回约 90 个行业板块。

    主表 stock_board_industry_summary_ths 提供涨跌幅/成交额/涨跌家数/领涨股;
    净额改用 stock_fund_flow_industry(即时) 的真实"流入−流出"覆盖
    (summary_ths 原"净流入"口径混乱、方向不可信, 已弃用)。

    返回 DataFrame 列(单位已统一为元):
      name(板块名), chg_pct(涨跌幅%), amount(总成交额,元),
      net_inflow(净额,元,全单口径,方向正确但数值偏小), up(上涨家数),
      down(下跌家数), leader(领涨股), leader_chg(领涨股涨跌幅%)
    默认按涨跌幅降序。主表拉取失败返回空 DataFrame; 净额源失败时 net_inflow 置空。
    """
    fetcher._check_ak()
    try:
        raw = fetcher._retry(lambda: ak.stock_board_industry_summary_ths(),
                             tries=2)
    except Exception as e:  # noqa
        print(f"[warn] 行业板块行情拉取失败: {e}")
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    ren = {"板块": "name", "涨跌幅": "chg_pct", "总成交额": "amount",
           "净流入": "net_inflow", "上涨家数": "up", "下跌家数": "down",
           "领涨股": "leader", "领涨股-涨跌幅": "leader_chg"}
    df = df.rename(columns=ren)
    for c in ["chg_pct", "amount", "net_inflow", "up", "down", "leader_chg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 同花顺金额单位为亿元, 统一 ×1e8 到元(此处仅 amount; net_inflow 下方覆盖)
    if "amount" in df.columns:
        df["amount"] = df["amount"] * 1e8

    # 用真实资金净额(流入−流出)覆盖 summary_ths 那个方向不可信的"净流入"
    net_map = _fetch_industry_net()
    if net_map:
        df["net_inflow"] = df["name"].map(net_map)
    else:
        df["net_inflow"] = float("nan")  # 净额源失败: 置空, 不显示错误方向数据

    keep = ["name", "chg_pct", "amount", "net_inflow", "up", "down",
            "leader", "leader_chg"]
    df = df[[c for c in keep if c in df.columns]]
    if "chg_pct" in df.columns:
        df = df.sort_values("chg_pct", ascending=False).reset_index(drop=True)
    return df


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

    # ---- 涨停池: 连板高度 / 梯队 / 涨停家数 ----
    try:
        zt = fetcher._retry(lambda: ak.stock_zt_pool_em(date=day), tries=2)
    except Exception as e:  # noqa
        print(f"[warn] 涨停池拉取失败: {e}")
        zt = None
    if zt is not None and not zt.empty and "连板数" in zt.columns:
        out["zt_count"] = int(len(zt))
        boards = pd.to_numeric(zt["连板数"], errors="coerce").dropna().astype(int)
        if not boards.empty:
            out["max_board"] = int(boards.max())
            out["first_board"] = int((boards <= 1).sum())
            vc = boards[boards >= 2].value_counts()
            out["ladder"] = [(int(b), int(vc[b]))
                             for b in sorted(vc.index, reverse=True)]

    # ---- 炸板池: 炸板家数(用于封板率) ----
    try:
        zb = fetcher._retry(lambda: ak.stock_zt_pool_zbgc_em(date=day), tries=2)
    except Exception as e:  # noqa
        print(f"[warn] 炸板池拉取失败: {e}")
        zb = None
    if zb is not None and not zb.empty:
        out["zb_count"] = int(len(zb))

    # ---- 封板率 = 涨停 / (涨停 + 炸板) ----
    zt_n, zb_n = out["zt_count"], out["zb_count"]
    if zt_n is not None and zb_n is not None and (zt_n + zb_n) > 0:
        out["seal_rate"] = round(zt_n / (zt_n + zb_n) * 100, 1)

    _LIMIT_POOL_CACHE["data"] = out
    _LIMIT_POOL_CACHE["ts"] = now
    return out


def _index_avg_volume(sina_code: str, n: int = 5) -> float:
    """
    取某指数最近 n 个交易日的日均成交量(与新浪指数日线 volume 同口径),
    用于量能对比的基准。sina_code 如 sh000001 / sz399001。失败返回 None。
    注: 成交额日线源(东财)已被限流不可用, 故量能对比统一走成交量;
    对宽基指数而言盘中价格波动小, 量比≈额比, 足够反映放量/缩量。
    """
    try:
        hist = fetcher._retry(
            lambda: ak.stock_zh_index_daily(symbol=sina_code), tries=2)
    except Exception as e:  # noqa
        print(f"[warn] 指数历史成交量拉取失败({sina_code}): {e}")
        return None
    if hist is None or hist.empty or "volume" not in hist.columns:
        return None
    # 排除今日(最后一行可能是当日实时, 用之前 n 日算基准更稳)
    vol = pd.to_numeric(hist["volume"], errors="coerce").dropna()
    if len(vol) < 2:
        return None
    base = vol.iloc[-(n + 1):-1] if len(vol) > n else vol.iloc[:-1]
    if base.empty:
        return None
    return float(base.mean())


def fetch_volume_compare(today_sh_vol: float = None,
                         today_sz_vol: float = None) -> dict:
    """
    量能对比(缩量/放量): 当前两市成交量 vs 最近5个交易日日均成交量。
    传入今日沪深成交量(与新浪日线 volume 同口径; 一般由 fetch_index_spot 的
    attrs['sh_volume']/['sz_volume'] 提供, 避免重复拉取)。

    统一用成交量口径(而非成交额): 因带成交额的指数日线源(东财)已被限流,
    而新浪日线只给成交量; 对宽基指数盘中价格波动小, 量比≈额比, 可靠反映
    放量/缩量。5 日均量按自然日缓存(历史日线当天不变)。

    返回 dict:
      {
        "today": 今日两市合计成交量(与基准同口径),
        "avg5": 最近5日两市日均合计成交量,
        "ratio_pct": (today/avg5 - 1)*100, 即放量/缩量百分比(float, 或 None),
        "state": "放量"/"缩量"/"持平"(str, 或 None),
      }
    """
    fetcher._check_ak()
    out = {"today": None, "avg5": None, "ratio_pct": None, "state": None}

    # 今日成交量: 优先用调用方传入的实时值
    parts = [v for v in (today_sh_vol, today_sz_vol) if v is not None]
    out["today"] = sum(parts) if parts else None

    # 5 日均量: 按自然日缓存
    day = datetime.datetime.now().strftime("%Y%m%d")
    if _AVG_VOL_CACHE["date"] != day:
        _AVG_VOL_CACHE["sh"] = _index_avg_volume("sh000001", 5)
        _AVG_VOL_CACHE["sz"] = _index_avg_volume("sz399001", 5)
        _AVG_VOL_CACHE["date"] = day
    avg_parts = [v for v in (_AVG_VOL_CACHE["sh"], _AVG_VOL_CACHE["sz"])
                 if v is not None]
    out["avg5"] = sum(avg_parts) if avg_parts else None

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
      }
    拉取失败返回空 bins。
    """
    now = time.time()
    if (_UPDOWN_DIST_CACHE["data"] is not None
            and now - _UPDOWN_DIST_CACHE["ts"] < _UPDOWN_DIST_TTL):
        return _UPDOWN_DIST_CACHE["data"]

    out = {"bins": [], "up": None, "down": None, "flat": None, "total": None}
    from . import realtime
    spot = realtime.fetch_spot()
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

    _UPDOWN_DIST_CACHE["data"] = out
    _UPDOWN_DIST_CACHE["ts"] = now
    return out
