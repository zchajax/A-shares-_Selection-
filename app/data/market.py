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
from . import database as db

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
    ren = {"板块": "name", "涨跌幅": "chg_pct", "总成交量": "volume",
           "总成交额": "amount", "净流入": "net_inflow",
           "上涨家数": "up", "下跌家数": "down", "均价": "avg_price",
           "领涨股": "leader", "领涨股-涨跌幅": "leader_chg"}
    df = df.rename(columns=ren)
    for c in ["chg_pct", "volume", "amount", "net_inflow", "up", "down",
              "avg_price", "leader_chg"]:
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

    keep = ["name", "chg_pct", "volume", "amount", "net_inflow", "up", "down",
            "avg_price", "leader", "leader_chg"]
    df = df[[c for c in keep if c in df.columns]]
    if "chg_pct" in df.columns:
        df = df.sort_values("chg_pct", ascending=False).reset_index(drop=True)
    return df


def fetch_board_trend(names: list) -> dict:
    """
    对指定的少数板块拉取同花顺行业指数日线, 计算多日趋势与量比。
    数据源 stock_board_industry_index_ths(同花顺, 不限流, 单板块~1s),
    板块名与 stock_board_industry_summary_ths 的"板块"列一致, 可直接复用。

    仅对最终入选诊断表两端的少量板块(约 12 个)调用, 绝不全量 90 个
    (那样 ~45s 会阻塞刷新), 由 UI 放在慢速独立线程里回填。

    返回 {板块名: {"chg_5d": %, "chg_20d": %, "vol_ratio": 倍}}:
      · chg_5d/chg_20d: 收盘价相对 5/20 交易日前的涨跌幅(%)
      · vol_ratio: 最新一日成交量 / 前 5 日均量(>1 放量, <1 缩量)
    历史当天不变, 按自然日缓存; 单板块失败则跳过(不塞错误值)。
    """
    today = datetime.date.today().isoformat()
    if _BOARD_TREND_CACHE["date"] != today:
        _BOARD_TREND_CACHE["date"] = today
        _BOARD_TREND_CACHE["data"] = {}
    cache = _BOARD_TREND_CACHE["data"]

    end = datetime.date.today().strftime("%Y%m%d")
    start = (datetime.date.today() -
             datetime.timedelta(days=60)).strftime("%Y%m%d")
    out = {}
    for nm in names:
        nm = str(nm)
        if nm in cache:
            out[nm] = cache[nm]
            continue
        try:
            h = fetcher._retry(
                lambda: ak.stock_board_industry_index_ths(
                    symbol=nm, start_date=start, end_date=end), tries=2)
        except Exception:  # noqa
            continue
        if h is None or h.empty or "收盘价" not in h.columns \
                or len(h) < 6:
            continue
        close = pd.to_numeric(h["收盘价"], errors="coerce").dropna()
        if len(close) < 6:
            continue
        last = close.iloc[-1]
        rec = {"chg_5d": None, "chg_20d": None, "vol_ratio": None}
        if len(close) >= 6 and close.iloc[-6]:
            rec["chg_5d"] = (last / close.iloc[-6] - 1) * 100
        if len(close) >= 21 and close.iloc[-21]:
            rec["chg_20d"] = (last / close.iloc[-21] - 1) * 100
        if "成交量" in h.columns:
            vol = pd.to_numeric(h["成交量"], errors="coerce").dropna()
            if len(vol) >= 6:
                base = vol.iloc[-6:-1].mean()
                if base:
                    rec["vol_ratio"] = vol.iloc[-1] / base
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
    对给定同花顺板块名, 取东财成分股算 PE/PB 中位数(当前值)。
    数据源 stock_board_industry_cons_em(东财, 每股直接带"市盈率-动态""市净率"),
    取正值中位数代表板块估值(避开亏损股负 PE 与异常极值)。

    仅对少量板块调用(逐板块一次网络请求, ~1-2s/个), 由 UI 放慢速线程回填,
    结果按自然日缓存。返回 {板块名: {"pe":中位数,"pb":中位数,"n":成分股数}}。
    映射不到东财板块或拉取失败的板块不出现在返回里(留空由 UI 兜底)。
    """
    fetcher._check_ak()
    today = datetime.date.today().isoformat()
    if _SECTOR_VAL_CACHE["date"] != today:
        _SECTOR_VAL_CACHE["date"] = today
        _SECTOR_VAL_CACHE["data"] = {}
    cache = _SECTOR_VAL_CACHE["data"]

    out = {}
    for nm in names:
        nm = str(nm)
        if nm in cache:
            out[nm] = cache[nm]
            continue
        em_nm = _em_sector_name(nm)
        if not em_nm:
            continue   # 匹配不到东财板块: 留空, 不硬凑
        try:
            cons = fetcher._retry(
                lambda: ak.stock_board_industry_cons_em(symbol=em_nm), tries=2)
        except Exception:  # noqa
            continue
        pe_v, pb_v, n = _median_pe_pb(cons)
        if pe_v is None and pb_v is None:
            continue
        rec = {"pe": pe_v, "pb": pb_v, "n": n}
        cache[nm] = rec
        out[nm] = rec
        # 落库当日快照(有 date+sector 主键, 盘中重复写只留最新)
        try:
            db.save_sector_valuation(today, nm, pe_v, pb_v, n)
        except Exception:  # noqa
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
    """
    回填某板块过去 days 天的 PE/PB 历史(用于首次就能算分位, 不用等天数积累)。
    做法: 取东财成分股名单, 逐股拉百度近一年 PE(TTM)/PB 历史序列, 按日期对齐后
    取每日中位数, 得到板块 PE/PB 日序列, 批量落库 sector_valuation。

    成本较高(成分股数 × 2 次百度请求), 仅在首次/历史不足时对少量板块调用,
    务必放后台线程。返回成功写入的交易日数(0 表示回填失败)。
    """
    fetcher._check_ak()
    em_nm = _em_sector_name(name)
    if not em_nm:
        return 0
    try:
        cons = fetcher._retry(
            lambda: ak.stock_board_industry_cons_em(symbol=em_nm), tries=2)
    except Exception:  # noqa
        return 0
    if cons is None or cons.empty or "代码" not in cons.columns:
        return 0
    codes = [str(c).zfill(6) for c in cons["代码"].tolist()]
    # 成分股过多时抽样(按市值/成交额排序取前 N, 中位数对样本量不敏感, 40 只足够)
    if "成交额" in cons.columns:
        order = pd.to_numeric(cons["成交额"], errors="coerce").fillna(0)
        codes = [str(c).zfill(6) for c in
                 cons.assign(_o=order).sort_values("_o", ascending=False)
                 ["代码"].tolist()]
    codes = codes[:40]

    pe_frames, pb_frames = [], []
    for code in codes:
        for ind, bucket in (("市盈率(TTM)", pe_frames), ("市净率", pb_frames)):
            try:
                d = fetcher._retry(
                    lambda: ak.stock_zh_valuation_baidu(
                        symbol=code, indicator=ind, period="近一年"), tries=1)
            except Exception:  # noqa
                continue
            if d is None or d.empty or "value" not in d.columns:
                continue
            s = pd.Series(
                pd.to_numeric(d["value"], errors="coerce").values,
                index=pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"))
            bucket.append(s)

    def _daily_median(frames):
        if not frames:
            return None
        m = pd.concat(frames, axis=1)
        m = m[m > 0]  # 只取正值
        return m.median(axis=1, skipna=True)

    pe_med = _daily_median(pe_frames)
    pb_med = _daily_median(pb_frames)
    if pe_med is None and pb_med is None:
        return 0

    dates = sorted(set(
        (list(pe_med.index) if pe_med is not None else []) +
        (list(pb_med.index) if pb_med is not None else [])))
    rows = []
    for d in dates:
        pe_v = pe_med.get(d) if pe_med is not None else None
        pb_v = pb_med.get(d) if pb_med is not None else None
        pe_v = None if pe_v is None or pe_v != pe_v else round(float(pe_v), 2)
        pb_v = None if pb_v is None or pb_v != pb_v else round(float(pb_v), 2)
        if pe_v is None and pb_v is None:
            continue
        rows.append((d, name, pe_v, pb_v, len(codes)))
    if not rows:
        return 0
    try:
        db.save_sector_valuation_batch(rows)
    except Exception:  # noqa
        return 0
    return len(rows)


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


def _index_amt_today_and_avg(em_code: str, n: int = 5):
    """
    从东财指数日线(stock_zh_index_daily_em)一次取『今日成交额』与『最近 n 日
    均额』, 两者同口径(均为 amount 列/元), 供量能对比用。em_code 如
    sh000001(上证综指) / sz399001(深证成指)。返回 (today_amt, avg_amt);
    任一不可得则该项为 None。

    成交额是市场通用口径(与券商/资讯端一致), 优先于成交量: 风格高低切换日
    (资金从高价股切到低价股)成交量(股)会被放大而成交额基本持平, 用成交额才
    不会误报"放量"。今日额取日线最后一行(盘中随实时更新), 基准取之前 n 日。
    """
    try:
        hist = fetcher._retry(
            lambda: ak.stock_zh_index_daily_em(symbol=em_code), tries=2)
    except Exception as e:  # noqa
        print(f"[warn] 指数成交额拉取失败({em_code}): {e}")
        return None, None
    if hist is None or hist.empty or "amount" not in hist.columns:
        return None, None
    amt = pd.to_numeric(hist["amount"], errors="coerce").dropna()
    if len(amt) < 2:
        return None, None
    today = float(amt.iloc[-1])
    base = amt.iloc[-(n + 1):-1] if len(amt) > n else amt.iloc[:-1]
    avg = float(base.mean()) if not base.empty else None
    return today, avg


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
