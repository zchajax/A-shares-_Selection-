"""
数据层 - 数据源接口
用 akshare 拉取 A 股列表和历史日线,并写入本地 SQLite。

【数据源说明 - 重要】
东财(eastmoney)接口在部分网络下会被限流/断连(Connection aborted /
RemoteDisconnected),不可用。

日线数据源默认使用【腾讯(tx)】stock_zh_a_hist_tx,原因:
- 稳定,实测每只约 3s;
- 不依赖 py_mini_racer(V8),因此【可多线程并发】(新浪源用 py_mini_racer
  解密,多线程会触发原生 DLL 崩溃,只能串行且每只 8-18s,400 只近 1 小时);
- 并发 10 线程后 400 只可压到 2-4 分钟。
腾讯源返回列: date open close high low amount(无 volume,用 amount/close 估算)。
代码需带 sh/sz 前缀。若腾讯源某只失败,自动回退新浪 stock_zh_a_daily 兜底。

股票列表仍用新浪 stock_info_a_code_name()(不分页,一次返回全A股)。

主要函数:
- update_stock_list()      更新全市场股票列表
- update_kline(code)       更新单只股票日线
- update_all_kline(codes)  批量更新(带进度回调)
"""
import time
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import database as db

try:
    import akshare as ak
except ImportError:
    ak = None

# 关闭 akshare 内部 tqdm 进度条(否则会往 stderr 疯狂刷屏,干扰日志)。
try:
    from functools import partialmethod
    from tqdm import tqdm
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
except Exception:  # noqa
    pass

# 全局 socket 超时:防止某只股票的网络请求无限期挂起(界面"卡在第一个不动"的根因)。
# akshare 底层用 requests,不显式设超时会一直等,设了之后卡住会抛异常被重试/跳过。
socket.setdefaulttimeout(15)

# 写库锁:并发拉取时多线程会同时写 SQLite,用锁串行化写入,避免 database is locked。
_DB_LOCK = threading.Lock()


def _check_ak():
    if ak is None:
        raise RuntimeError("未安装 akshare,请先运行: pip install akshare")


def _to_sina_symbol(code: str) -> str:
    """
    把 6 位纯数字代码转成新浪需要的带交易所前缀格式。
    6xx -> sh(上海)   0xx/3xx -> sz(深圳)
    北交所(8xx/920/4xx)新浪日线支持不稳,调用方一般已过滤。
    """
    code = str(code).zfill(6)
    if code.startswith("6"):
        return "sh" + code
    if code.startswith(("0", "3")):
        return "sz" + code
    if code.startswith(("8", "4", "920")):
        return "bj" + code
    return "sh" + code  # 兜底


def _retry(fn, tries=3, delay=1.0):
    """简单重试:网络类异常时重试若干次。"""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa
            last = e
            if i < tries - 1:
                time.sleep(delay)
    raise last


def _minute_symbol(code: str) -> str:
    """
    分时(stock_zh_a_minute)用的带前缀代码。股票 6/0/3、ETF 5/1 都支持。
    ETF:5开头->sh,1开头->sz;其余走 _to_sina_symbol。
    """
    code = str(code).zfill(6)
    if code.startswith("5"):
        return "sh" + code
    if code.startswith("1"):
        return "sz" + code
    return _to_sina_symbol(code)


def fetch_intraday(code: str, period: str = "1") -> pd.DataFrame:
    """
    拉取当日分时数据(新浪 stock_zh_a_minute,1分钟线,含盘中实时最新分钟)。
    股票和 ETF 均可用。返回只保留"今天"的分钟序列,列:
      time(HH:MM) / price(收盘价当作该分钟价) / avg(累计均价) / volume / amount
    盘前/无数据时返回空 DataFrame。均价 avg = 累计成交额 / 累计成交量(分时黄线)。

    注:新浪分钟接口返回最近若干日的分钟线,这里按最后一个交易日截取当天。
    不入库(分时是临时盘中数据,不像日线需要持久化)。
    """
    _check_ak()
    sym = _minute_symbol(code)
    try:
        raw = _retry(lambda: ak.stock_zh_a_minute(
            symbol=sym, period=str(period), adjust=""), tries=2)
    except Exception as e:  # noqa
        print(f"[warn] 分时 {code} 拉取失败: {e}")
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["day"] = pd.to_datetime(df["day"])
    # 新浪分钟接口返回最近若干个交易日的分钟线。
    # 真昨收 = 上一交易日最后一分钟的收盘价(比本地日线可靠:同源、
    # 且必是真实的上一交易日,不受本地日线是否已更新到最新影响)。
    all_days = sorted(df["day"].dt.date.unique())
    last_day = all_days[-1]
    prev_close = None
    if len(all_days) >= 2:
        prev_rows = df[df["day"].dt.date == all_days[-2]]
        prev_rows = prev_rows[pd.to_numeric(prev_rows["close"], errors="coerce").notna()]
        if not prev_rows.empty:
            try:
                prev_close = float(prev_rows.iloc[-1]["close"])
            except Exception:  # noqa
                prev_close = None
    # 只保留最后一个交易日(当天)的分钟
    df = df[df["day"].dt.date == last_day].copy()
    if df.empty:
        return pd.DataFrame()
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    # 新浪分钟线用"分钟结束时刻"标注(09:30-09:31 这根标成 09:31),
    # 比系统时钟/主流软件超前1分钟。回退到"分钟起始时刻"对齐,
    # 使开盘首点=09:30、最新一根不再超前于当前时间。
    df["time"] = (df["day"] - pd.Timedelta(minutes=1)).dt.strftime("%H:%M")
    df["price"] = df["close"]
    # 分时均价线(黄线):累计成交额 / 累计成交量
    cum_amt = df["amount"].cumsum()
    cum_vol = df["volume"].cumsum()
    df["avg"] = (cum_amt / (cum_vol + 1e-9)).round(3)
    keep = ["time", "price", "avg", "volume", "amount"]
    out = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    out.attrs["trade_date"] = str(last_day)
    out.attrs["prev_close"] = prev_close  # 真昨收(上一交易日收盘);拿不到为 None
    return out


def get_mainstream_codes() -> list:
    """
    获取"主流股"代码集合:沪深300 + 中证500 成分股(共约800只)。
    这些是沪深两市市值最大、流动性最好的核心标的,回测/选股最有代表性。
    用中证官方接口 index_stock_cons_csindex,实测稳定。
    任一接口失败则跳过,不影响主流程。
    返回:保持"沪深300在前、中证500在后"的有序去重列表。
    """
    ordered = []
    seen = set()
    for idx in ("000300", "000905"):  # 沪深300, 中证500
        try:
            df = _retry(lambda idx=idx: ak.index_stock_cons_csindex(symbol=idx))
            for c in df["成分券代码"].astype(str).str.zfill(6):
                if c not in seen:
                    seen.add(c)
                    ordered.append(c)
        except Exception as e:  # noqa
            print(f"[warn] 获取指数 {idx} 成分股失败: {e}")
    return ordered


def update_stock_list() -> pd.DataFrame:
    """
    拉取沪深 A 股列表(代码+名称)并存库。
    使用新浪源 stock_info_a_code_name(),不分页,稳定。

    排序策略:把"主流股"(沪深300+中证500 成分股)排在列表最前面,
    这样按前 N 只拉取时,优先拿到大盘蓝筹核心标的,而不是按代码升序
    堆一批深市小盘股。主流股内部也按 沪深300 -> 中证500 顺序。
    """
    _check_ak()
    df = _retry(lambda: ak.stock_info_a_code_name())
    df = df[["code", "name"]].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    # 过滤北交所(8xx/920/4xx),新浪日线支持不稳,MVP 阶段先聚焦沪深主板/创业板/科创板
    df = df[~df["code"].str.startswith(("8", "4", "92"))]
    # 过滤退市股(名称含"退")和风险警示股(ST/*ST),这些不是正常可交易标的
    name_up = df["name"].astype(str)
    mask_bad = name_up.str.contains("退") | name_up.str.contains("ST", case=False)
    df = df[~mask_bad].reset_index(drop=True)

    # ---- 主流股优先排序 ----
    main_codes = get_mainstream_codes()
    if main_codes:
        # 给主流股一个从0开始的优先级序号,非主流股排在最后(用大数)
        rank = {c: i for i, c in enumerate(main_codes)}
        df["_rank"] = df["code"].map(lambda c: rank.get(c, 10 ** 9))
        df = df.sort_values(["_rank", "code"]).drop(columns="_rank").reset_index(drop=True)

    db.save_stock_list(df)
    return df


def _fetch_daily_raw(sina_symbol: str, start_date: str, adjust: str) -> pd.DataFrame:
    """
    拉取单只日线原始数据,统一返回列: date open high low close volume amount。
    优先用腾讯源(快、可并发),失败回退新浪源兜底。
    sina_symbol 形如 sh600519 / sz000001。
    """
    # ---- 主力:腾讯源(不依赖 py_mini_racer,可并发) ----
    try:
        df = _retry(lambda: ak.stock_zh_a_hist_tx(
            symbol=sina_symbol, start_date=start_date,
            end_date="20991231", adjust=adjust,
        ), tries=2)
        if df is not None and not df.empty:
            df = df.rename(columns=str.lower).copy()
            # 腾讯列: date open close high low amount
            # 注意!! 腾讯这个 amount 列实际是"成交量(单位:手)",不是成交额(元)。
            #   实测: 工行 amount=6507835 ×100 = 6.5亿股 ≈ 新浪真实 volume,完全吻合。
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            for c in ["open", "high", "low", "close", "amount"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            # amount(手) → volume(股): ×100
            df["volume"] = (df["amount"].fillna(0.0) * 100).astype(float)
            # 腾讯源无真实成交额,用 成交量(股)×收盘价 估算(元)。
            #   大盘股 close≈日内均价,误差极小(工行估49亿 vs 真实48.6亿,<1%)。
            df["amount"] = (df["volume"] * df["close"]).astype(float)
            return df[["date", "open", "high", "low", "close", "volume", "amount"]]
    except Exception as e:  # noqa
        print(f"[info] 腾讯源 {sina_symbol} 失败,回退新浪: {e}")

    # ---- 兜底:新浪源(慢、不可并发,但稳) ----
    raw = _retry(lambda: ak.stock_zh_a_daily(
        symbol=sina_symbol, start_date=start_date, adjust=adjust
    ))
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if "amount" not in df.columns:
        df["amount"] = (df["volume"] * df["close"]).astype(float)
    df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def update_kline(code: str, start_date: str = "20230101", adjust: str = "qfq",
                 incremental: bool = True) -> pd.DataFrame:
    """
    拉取单只股票日线并存库(腾讯源为主,新浪兜底)。
    adjust: qfq=前复权(推荐), hfq=后复权, ""=不复权
    incremental=True 时:若本地已有数据,只从最新日期次日开始增量拉取并追加,
                        大幅减少重复下载;否则全量拉取并覆盖。
    """
    _check_ak()
    sina_symbol = _to_sina_symbol(code)

    real_start = start_date
    do_append = False
    if incremental:
        last = db.get_last_date(code)  # YYYY-MM-DD 或 None
        if last:
            # 从最新日期的次日开始拉;若已是最新,基本拉不到新数据
            nxt = (pd.to_datetime(last) + pd.Timedelta(days=1)).strftime("%Y%m%d")
            real_start = nxt
            do_append = True
            # 已经很新(次日晚于今天),直接返回,连网络都不请求
            if pd.to_datetime(nxt) > pd.Timestamp.now().normalize():
                return db.load_kline(code)

    df = _fetch_daily_raw(sina_symbol, real_start, adjust)
    if df is None or df.empty:
        return db.load_kline(code) if do_append else pd.DataFrame()

    df = df.dropna(subset=["close"]).reset_index(drop=True)

    if do_append:
        with _DB_LOCK:
            db.append_kline(code, df)   # 增量:只追加新日期,不动旧数据
    else:
        with _DB_LOCK:
            db.save_kline(code, df)     # 全量:覆盖
    return db.load_kline(code)


def update_index_kline(index_code: str = "sh000001",
                       start_date: str = "20210101") -> pd.DataFrame:
    """
    拉取大盘指数日线并存库,用于"大盘趋势过滤"。
    默认上证综指 sh000001。用新浪 stock_zh_index_daily(稳定,单只无需并发)。
    每次全量覆盖(指数就一条,开销很小),保证到最新交易日。
    返回存库后的 DataFrame。
    """
    _check_ak()
    try:
        raw = _retry(lambda: ak.stock_zh_index_daily(symbol=index_code))
    except Exception as e:  # noqa
        print(f"[warn] 拉取指数 {index_code} 失败: {e}")
        return db.load_index_kline(index_code)
    if raw is None or raw.empty:
        return db.load_index_kline(index_code)
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[df["date"] >= pd.to_datetime(start_date).strftime("%Y-%m-%d")]
    if "amount" not in df.columns:
        df["amount"] = 0.0
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    db.save_index_kline(index_code, df)
    return db.load_index_kline(index_code)


def _fetch_one_industry(code: str) -> str:
    """
    查单只股票的所属行业(巨潮 stock_industry_change_cninfo,取最新变更记录)。
    返回行业名称字符串;查不到或异常返回空串。
    巨潮接口【不可并发】(并发会被限流挂起),必须串行调用。
    """
    try:
        df = _retry(lambda: ak.stock_industry_change_cninfo(
            symbol=code, start_date="19910101", end_date="20991231"), tries=2)
        if df is None or df.empty:
            return ""
        r = df.sort_values("变更日期").iloc[-1]
        # 优先"行业大类"(如"汽车制造业"),缺失则退到"行业门类"
        val = r.get("行业大类") or r.get("行业门类") or ""
        val = str(val).strip()
        return "" if val in ("", "nan", "None") else val
    except Exception:  # noqa
        return ""


def update_industry(codes: list, incremental: bool = True,
                    progress_cb=None) -> int:
    """
    拉取并存储一批股票的所属行业(巨潮行业分类)。
    - 巨潮接口不可并发,这里串行拉取(每只约 0.2-0.5s),
      对本地几百只股票一次几分钟,且行业极少变,结果长期缓存即可。
    - incremental=True 时只查还没有行业的股票,已分类的跳过(秒回)。
    - 允许部分失败:查不到的留空,不影响其它。
    progress_cb(done, total, code)。返回成功写入的条数。
    """
    _check_ak()
    targets = db.codes_without_industry(codes) if incremental else list(codes)
    total = len(targets)
    if total == 0:
        return 0
    mapping = {}
    ok = 0
    BATCH = 30  # 每积累一批就写库一次,避免中途中断全丢
    for i, code in enumerate(targets):
        ind = _fetch_one_industry(code)
        if ind:
            mapping[code] = ind
            ok += 1
        if len(mapping) >= BATCH:
            db.save_industry_map(mapping)
            mapping = {}
        if progress_cb:
            progress_cb(i + 1, total, code)
    if mapping:
        db.save_industry_map(mapping)
    return ok


def _baidu_latest(code: str, indicator: str):
    """
    取百度估值接口某指标的最新一个值。indicator 例:
    "市盈率(TTM)" / "市净率" / "市销率(TTM)" / "总市值"。
    返回 float 或 None。
    """
    try:
        df = ak.stock_zh_valuation_baidu(symbol=code, indicator=indicator,
                                         period="近一年")
        if df is None or df.empty:
            return None
        v = df.sort_values("date").iloc[-1]["value"]
        return float(v)
    except Exception:
        return None


def _fetch_one_fundamental(code: str) -> dict:
    """
    查单只股票的基本面/估值 + 成长/质量指标(百度估值 + 巨潮财务指标)。
    这些接口【不可高并发】(会被限流),必须串行,每只约 1-2s。
    返回 {pe_ttm, pb, ps_ttm, total_mv, roe, gross_margin, net_margin,
          rev_yoy, profit_yoy, debt_ratio, dividend_ratio, report_date},
    取不到的字段为 None。
      · total_mv 百度返回单位为亿元,直接沿用。
      · 成长/质量字段来自巨潮同一张财务分析指标表,与 ROE 同源同一次调用。
    """
    d = {
        "pe_ttm": _baidu_latest(code, "市盈率(TTM)"),
        "pb": _baidu_latest(code, "市净率"),
        "ps_ttm": _baidu_latest(code, "市销率(TTM)"),
        "total_mv": _baidu_latest(code, "总市值"),
        "roe": None,
        "gross_margin": None, "net_margin": None,
        "rev_yoy": None, "profit_yoy": None,
        "debt_ratio": None, "dividend_ratio": None,
        "report_date": None,
    }
    # 巨潮财务分析指标:最新一期同时取 ROE + 毛利率/净利率/营收增速/净利增速/
    # 资产负债率/股息发放率(同一次调用,不额外增加请求)
    try:
        fin = _retry(lambda: ak.stock_financial_analysis_indicator(
            symbol=code, start_year="2023"), tries=2)
        if fin is not None and not fin.empty:
            row = fin.sort_values("日期").iloc[-1]

            def _pick(keys, exclude=None):
                """按列名关键词取最新一期的值(容错列名差异)。"""
                for col in fin.columns:
                    if exclude and exclude in col:
                        continue
                    if all(k in col for k in keys):
                        try:
                            v = row[col]
                            return float(v) if pd.notna(v) else None
                        except Exception:
                            return None
                return None

            d["roe"] = _pick(["净资产收益率"], exclude="加权")
            d["gross_margin"] = _pick(["销售毛利率"])
            d["net_margin"] = _pick(["销售净利率"])
            d["rev_yoy"] = _pick(["主营业务收入增长率"])
            d["profit_yoy"] = _pick(["净利润增长率"])
            d["debt_ratio"] = _pick(["资产负债率"])
            d["dividend_ratio"] = _pick(["股息发放率"])
            try:
                d["report_date"] = str(row["日期"])
            except Exception:
                pass
    except Exception:
        pass
    return d


def update_fundamental(codes: list, incremental: bool = True,
                       progress_cb=None) -> int:
    """
    拉取并存储一批股票的基本面/估值(百度估值 + 巨潮 ROE)。
    - 接口不可并发,串行拉取(每只约 1-2s),对几百只票需几分钟;
      估值/季报变动不频繁,结果长期缓存即可,增量只补缺失。
    - 能拉多少算多少:某只/某字段失败留空,不阻塞其它。
    progress_cb(done, total, code)。返回成功写入(至少一个字段非空)的条数。
    """
    _check_ak()
    targets = db.codes_without_fundamental(codes) if incremental else list(codes)
    total = len(targets)
    if total == 0:
        return 0
    buf = {}
    ok = 0
    BATCH = 20  # 每积累一批写库一次,避免中途中断全丢
    for i, code in enumerate(targets):
        d = _fetch_one_fundamental(code)
        if any(v is not None for v in d.values()):
            buf[code] = d
            ok += 1
        if len(buf) >= BATCH:
            db.save_fundamental(buf)
            buf = {}
        if progress_cb:
            progress_cb(i + 1, total, code)
    if buf:
        db.save_fundamental(buf)
    return ok


# ==================== ETF ====================
# 主流宽基 + 热门行业 ETF 白名单(约 50 只):覆盖核心指数与主要赛道,
# 流动性好、有代表性,够日常选股/盯盘用。全市场 1500+ 只多为冷门低流动性,
# 不默认拉取(可后续扩展"全市场ETF"开关)。
MAINSTREAM_ETFS = [
    # --- 宽基指数 ---
    "510300",  # 沪深300ETF
    "510500",  # 中证500ETF
    "510050",  # 上证50ETF
    "159949",  # 创业板50ETF
    "159915",  # 创业板ETF
    "588000",  # 科创50ETF
    "588080",  # 科创板50ETF
    "512100",  # 中证1000ETF
    "159845",  # 中证1000ETF
    "510880",  # 红利ETF
    "512890",  # 红利低波ETF
    "515180",  # 红利ETF易方达
    "159919",  # 沪深300ETF嘉实
    "512500",  # 中证500ETF嘉实
    # --- 科技/成长 ---
    "515790",  # 光伏ETF
    "159806",  # 新能源车ETF
    "515030",  # 新能源车ETF
    "512760",  # 半导体ETF
    "159995",  # 芯片ETF
    "512480",  # 半导体ETF
    "515050",  # 5G通信ETF
    "159939",  # 信息技术ETF
    "515000",  # 科技ETF
    "159852",  # 软件ETF
    "512720",  # 计算机ETF
    "516010",  # 游戏ETF
    "159801",  # 芯片ETF广发
    # --- 消费/医药 ---
    "159928",  # 消费ETF
    "512690",  # 酒ETF
    "159936",  # 消费ETF嘉实
    "512010",  # 医药ETF
    "159929",  # 医药ETF
    "512170",  # 医疗ETF
    "159938",  # 医药卫生ETF
    "512290",  # 生物医药ETF
    "512670",  # 国防军工ETF
    # --- 金融/周期 ---
    "512800",  # 银行ETF
    "512880",  # 证券ETF
    "512000",  # 券商ETF
    "512200",  # 房地产ETF
    "515220",  # 煤炭ETF
    "159825",  # 农业ETF
    "512400",  # 有色金属ETF
    "159611",  # 电力ETF
    # --- 主题/海外/商品 ---
    "518880",  # 黄金ETF
    "159934",  # 黄金ETF易方达
    "513050",  # 中概互联网ETF
    "513100",  # 纳指ETF
    "159941",  # 纳指ETF广发
    "513500",  # 标普500ETF
    "510900",  # H股ETF
    "513180",  # 恒生科技ETF
]


def _etf_sina_symbol(code: str) -> str:
    """ETF 6位代码 -> 新浪带前缀:5开头(沪)->sh,1开头(深)->sz。"""
    code = str(code).zfill(6)
    if code.startswith("5"):
        return "sh" + code
    if code.startswith("1"):
        return "sz" + code
    return _to_sina_symbol(code)


def update_etf_list(only_mainstream: bool = True) -> pd.DataFrame:
    """
    拉取 ETF 列表并存库(新浪 fund_etf_category_sina,一次返回全市场 ~1500 只)。
    only_mainstream=True 时只保留 MAINSTREAM_ETFS 白名单(约50只主流宽基+行业);
    结果写入 etf_list(并同步进 stock_list 供 namemap/扫描)。
    返回存库的 DataFrame(code,name)。
    """
    _check_ak()
    raw = _retry(lambda: ak.fund_etf_category_sina(symbol="ETF基金"))
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["code", "name"])
    df = raw.copy()
    # 代码带 sh/sz 前缀,转 6 位纯数字
    df["code"] = df["代码"].astype(str).str.replace(
        r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
    df = df.rename(columns={"名称": "name"})[["code", "name"]]
    df = df.drop_duplicates("code").reset_index(drop=True)
    if only_mainstream:
        wl = set(MAINSTREAM_ETFS)
        df = df[df["code"].isin(wl)].reset_index(drop=True)
    db.save_etf_list(df)
    return df


def fetch_etf_spot(only_registered: bool = True) -> pd.DataFrame:
    """
    拉取 ETF 实时快照(新浪 fund_etf_category_sina,一次全市场)。
    only_registered=True 时只返回本地已登记(etf_list)的 ETF。
    返回 DataFrame: code,name,price,chg_pct,open,high,low,amount。
    盘前/休市时新浪返回最新价可能为0,这里保留原值由 UI 处理。
    """
    _check_ak()
    try:
        raw = _retry(lambda: ak.fund_etf_category_sina(symbol="ETF基金"))
    except Exception as e:  # noqa
        print(f"[warn] ETF 实时快照拉取失败: {e}")
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df["code"] = df["代码"].astype(str).str.replace(
        r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
    ren = {"名称": "name", "最新价": "price", "涨跌幅": "chg_pct",
           "今开": "open", "最高": "high", "最低": "low", "成交额": "amount"}
    df = df.rename(columns=ren)
    for c in ["price", "chg_pct", "open", "high", "low", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["code", "name", "price", "chg_pct", "open", "high", "low", "amount"]
    df = df[[c for c in keep if c in df.columns]]
    if only_registered:
        reg = db.load_etf_codes()
        if reg:
            df = df[df["code"].isin(reg)].reset_index(drop=True)
    return df.reset_index(drop=True)


def update_etf_kline(code: str, start_date: str = "20230101",
                     incremental: bool = True) -> pd.DataFrame:
    """
    拉取单只 ETF 日线并存库(新浪 fund_etf_hist_sina)。
    ETF 日线复用 daily_kline 表(格式与股票一致),因此策略/回测/K线自动支持。
    新浪 ETF 接口返回全部历史,这里本地按 start_date 截取;
    incremental=True 时若本地已有数据,只追加比最新日期更晚的行。
    """
    _check_ak()
    sym = _etf_sina_symbol(code)
    last = db.get_last_date(code) if incremental else None

    try:
        raw = _retry(lambda: ak.fund_etf_hist_sina(symbol=sym), tries=2)
    except Exception as e:  # noqa
        print(f"[warn] ETF {code} 日线拉取失败: {e}")
        return db.load_kline(code)
    if raw is None or raw.empty:
        return db.load_kline(code)

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    # 新浪 ETF hist 无 amount 时用 成交额=volume*close 估算
    if "amount" not in df.columns:
        df["amount"] = 0.0
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    # 本地按起始日期截取
    start_fmt = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    df = df[df["date"] >= start_fmt].reset_index(drop=True)
    if df.empty:
        return db.load_kline(code)

    if last:
        df = df[df["date"] > last].reset_index(drop=True)
        if df.empty:
            return db.load_kline(code)
        with _DB_LOCK:
            db.append_kline(code, df)
    else:
        with _DB_LOCK:
            db.save_kline(code, df)
    return db.load_kline(code)


def update_all_etf_kline(codes: list = None, start_date: str = "20230101",
                         incremental: bool = True, progress_cb=None) -> int:
    """
    批量更新 ETF 日线。codes=None 时用 etf_list 里全部已登记 ETF。
    新浪 ETF hist 用 py_mini_racer 解密,【不可并发】(多线程触发原生DLL崩溃),
    故串行拉取(每只约 0.4-0.7s,50 只约半分钟)。progress_cb(done,total,code)。
    返回失败数。
    """
    if codes is None:
        codes = db.load_etf_list()["code"].tolist()
    total = len(codes)
    fail = 0
    for i, code in enumerate(codes):
        try:
            update_etf_kline(code, start_date=start_date, incremental=incremental)
        except Exception as e:  # noqa
            fail += 1
            print(f"[warn] ETF {code} 拉取失败: {e}")
        if progress_cb:
            progress_cb(i + 1, total, code)
    return fail


def update_all_kline(codes: list, start_date: str = "20230101",
                     incremental: bool = True, progress_cb=None,
                     workers: int = 10):
    """
    批量更新日线(并发)。progress_cb(done, total, code) 用于界面显示进度。

    改为线程池并发拉取:日线下载是网络 IO 密集型,串行时 400 只要逐个等待,
    且任一只卡住会拖住整体。用 workers 个线程并行请求(配合全局 15s 超时),
    速度约提升 workers 倍,单只卡死也只会拖累它自己那条线程。
    写库已用 _DB_LOCK 串行化,SQLite 不会冲突。
    incremental=True 时对已有数据只做增量补齐,速度更快。
    """
    total = len(codes)
    fail = 0
    done = 0
    lock = threading.Lock()

    def work(code):
        try:
            update_kline(code, start_date=start_date, incremental=incremental)
            return code, None
        except Exception as e:  # noqa
            return code, e

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, c) for c in codes]
        for fut in as_completed(futures):
            code, err = fut.result()
            with lock:
                done += 1
                if err is not None:
                    fail += 1
                    print(f"[warn] {code} 拉取失败: {err}")
                cur_done = done
            if progress_cb:
                progress_cb(cur_done, total, code)
    return fail
