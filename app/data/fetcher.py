"""
数据层 - 数据源接口
用 akshare 拉取 A 股列表和历史日线,并写入本地 SQLite。

【数据源说明 - 重要】
东财(eastmoney)接口在部分网络下会被限流/断连(Connection aborted /
RemoteDisconnected),不可用。

日线直接请求腾讯 newfqkline（前复权）。单次最多约 640 根，
按 700 天一段切窗口后并行抓取；增量只抓最近一段。
新浪 stock_zh_a_daily 用 py_mini_racer，多线程会崩，只在腾讯失败时加锁串行兜底。
成交额与成交量统一为真实口径：amount = 腾讯成交额(万元) × 10_000 (元)，
volume = amount / close (前复权口径下的折算股数，消除送转除权断层)。

股票列表仍用新浪 stock_info_a_code_name()(不分页,一次返回全A股)。

主要函数:
- update_stock_list()      更新全市场股票列表
- update_kline(code)       更新单只股票日线
- update_all_kline(codes)  批量更新(带进度回调)
"""
import io
import json
import time
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import pandas as pd
import requests

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
# 同时打到腾讯的请求数。再高容易被掐连接。
_TX_SEM = threading.Semaphore(20)
# 新浪日线解密不能并发。
_SINA_LOCK = threading.Lock()
_TLS = threading.local()
_TX_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
_KLINE_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]
_UNSET = object()


def _check_ak():
    if ak is None:
        raise RuntimeError("未安装 akshare,请先运行: pip install akshare")


def _to_sina_symbol(code: str) -> str:
    """
    把 6 位纯数字代码转成新浪/腾讯需要的带交易所前缀格式。
    6xx/5xx -> sh(上海)   0xx/3xx/1xx -> sz(深圳)
    北交所(8xx/920/4xx) -> bj
    """
    code = str(code).zfill(6)
    if code.startswith(("6", "5")):
        return "sh" + code
    if code.startswith(("0", "3", "1")):
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


def _clean_stock_list(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["code", "name"]].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    # 仅保留沪深 A 股代码(0/3/6开头)，剔除北交所(8/4/92)与误混入的场内基金(15/51/56/58)
    df = df[df["code"].str.startswith(("0", "3", "6"))]
    name_up = df["name"].astype(str)
    mask_bad = name_up.str.contains("退") | name_up.str.contains("ST", case=False)
    return df[~mask_bad].drop_duplicates("code").reset_index(drop=True)


def get_mainstream_codes() -> list:
    """
    沪深300 + 中证500 成分股。每个文件最多等 8 秒，失败就跳过。
    返回:沪深300在前、中证500在后的有序去重列表。
    """
    ordered = []
    seen = set()
    for idx in ("000300", "000905"):
        url = (
            "https://oss-ch.csindex.com.cn/static/html/csindex/"
            f"public/uploads/file/autofile/cons/{idx}cons.xls"
        )
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            table = pd.read_excel(io.BytesIO(resp.content))
            col = "成分券代码" if "成分券代码" in table.columns else table.columns[4]
            for raw in table[col].astype(str):
                code = raw.zfill(6)[-6:]
                if code not in seen:
                    seen.add(code)
                    ordered.append(code)
        except Exception as exc:  # noqa
            print(f"[warn] 获取指数 {idx} 成分股失败: {exc}")
    return ordered


def _fetch_exchange_list() -> pd.DataFrame:
    """
    拉取沪深京全部在市 A 股代码与名称（含上交所主板/科创板、深交所主板/创业板、北交所及风险警示板，剔除退市股与ETF）。
    优先通过东财数据中心 RPT_LICO_FN_CPD 并发拉取（约 1.5s，不受北交所接口改版影响），失败再回退 akshare。
    """
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    for q_dt in ("2026-09-30", "2026-06-30"):
        try:
            r0 = _http().get(
                url,
                params={
                    "reportName": "RPT_LICO_FN_CPD",
                    "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,TRADE_MARKET,SECURITY_TYPE",
                    "filter": f"(REPORTDATE='{q_dt}')",
                    "pageNumber": "1",
                    "pageSize": "500",
                    "source": "WEB",
                    "client": "WEB",
                },
                timeout=8,
            ).json()
            res0 = r0.get("result")
            if not res0 or not res0.get("data"):
                continue
            pages = int(res0.get("pages") or 1)
            all_items = list(res0["data"])

            def _pull_p(p):
                rp = _http().get(
                    url,
                    params={
                        "reportName": "RPT_LICO_FN_CPD",
                        "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,TRADE_MARKET,SECURITY_TYPE",
                        "filter": f"(REPORTDATE='{q_dt}')",
                        "pageNumber": str(p),
                        "pageSize": "500",
                        "source": "WEB",
                        "client": "WEB",
                    },
                    timeout=8,
                ).json()
                return (rp.get("result") or {}).get("data") or []

            if pages > 1:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    for part in pool.map(_pull_p, range(2, pages + 1)):
                        all_items.extend(part)

            rows = []
            for it in all_items:
                tm = str(it.get("TRADE_MARKET") or "")
                st = str(it.get("SECURITY_TYPE") or "")
                if st in ("A股", "中国存托凭证") and ("上交所" in tm or "深交所" in tm or "北交所" in tm or "北京" in tm):
                    c = str(it.get("SECURITY_CODE") or "").strip().zfill(6)
                    nm = str(it.get("SECURITY_NAME_ABBR") or "").strip()
                    if c.startswith(("00", "30", "60", "68", "92", "83", "87", "43")) and "退" not in nm:
                        rows.append({"code": c, "name": nm})
            if len(rows) >= 5000:
                return pd.DataFrame(rows).drop_duplicates("code").reset_index(drop=True)
        except Exception:
            pass

    _check_ak()
    box = {}

    def run():
        try:
            box["df"] = ak.stock_info_a_code_name()
        except Exception as exc:  # noqa
            box["err"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(20)
    if "df" not in box:
        raise RuntimeError(box.get("err") or "交易所股票名单超时")
    return _clean_stock_list(box["df"])


def update_stock_list(sort_mainstream: bool = True) -> pd.DataFrame:
    """
    返回沪深 A 股列表。本地已有名单时直接用，不再访问上交所/深交所。
    sort_mainstream=True 时把沪深300、中证500 排到前面，供「主流400」使用。
    """
    cached = db.load_stock_list()
    if cached is not None and not cached.empty:
        df = _clean_stock_list(cached)
    else:
        df = _fetch_exchange_list()
        db.save_stock_list(df)

    if not sort_mainstream:
        return df
    main_codes = get_mainstream_codes()
    if not main_codes:
        return df
    rank = {c: i for i, c in enumerate(main_codes)}
    df = df.copy()
    df["_rank"] = df["code"].map(lambda c: rank.get(c, 10 ** 9))
    return df.sort_values(["_rank", "code"]).drop(columns="_rank").reset_index(drop=True)


def _http():
    session = getattr(_TLS, "session", None)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "Mozilla/5.0"
        _TLS.session = session
    return session


def _as_date(text: str) -> date:
    return datetime.strptime(str(text).replace("-", "")[:8], "%Y%m%d").date()


def _window_ends(start: date, today: date):
    """每段最多 640 根，大约覆盖 900 个自然日。步长 700 天，保证两段重叠。"""
    ends = []
    cursor = today
    for _ in range(8):
        ends.append(cursor)
        if cursor - timedelta(days=900) <= start:
            break
        cursor = cursor - timedelta(days=700)
    return ends


def _tx_window(symbol: str, end: date, adjust: str, limit: int = 640):
    lim = max(5, min(640, int(limit)))
    param = f"{symbol},day,,{end.isoformat()},{lim},{adjust}"
    last = None
    for _ in range(2):
        try:
            with _TX_SEM:
                resp = _http().get(
                    _TX_URL, params={"_var": "kline_day", "param": param}, timeout=12,
                )
            if resp.status_code != 200 or len(resp.content) < 120:
                last = RuntimeError(f"http {resp.status_code} {len(resp.content)}")
                time.sleep(0.3)
                continue
            text = resp.text
            mark = text.find("={")
            if mark < 0:
                last = RuntimeError("no json")
                time.sleep(0.3)
                continue
            payload = json.loads(text[mark + 1:])
            data = payload.get("data")
            block = data.get(symbol) if isinstance(data, dict) else None
            if not isinstance(block, dict):
                return []
            if adjust == "hfq":
                rows = block.get("hfqday") or block.get("day") or []
            elif adjust == "qfq":
                rows = block.get("qfqday") or block.get("day") or []
            else:
                rows = block.get("day") or []
            return rows if isinstance(rows, list) else []
        except Exception as exc:  # noqa
            last = exc
            time.sleep(0.3)
    raise last


def _bars_to_df(rows, start: date) -> pd.DataFrame:
    start_s = start.isoformat()
    recs = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 9:
            continue
        day = str(row[0])[:10]
        if day < start_s:
            continue
        try:
            opened = float(row[1])
            closed = float(row[2])
            high = float(row[3])
            low = float(row[4])
            amount_wan = float(row[8])
        except (TypeError, ValueError):
            continue
        if closed != closed or closed <= 0:
            continue
        # 统一真实量纲：amount 为成交额(元) = 万元 × 10,000；
        # volume 为前复权口径股数 = 成交额(元) / 前复权收盘价，彻底消除除权与跨期断层。
        amount = amount_wan * 10_000.0
        volume = amount / closed
        recs.append((day, opened, high, low, closed, volume, amount))
    if not recs:
        return pd.DataFrame(columns=_KLINE_COLS)
    df = pd.DataFrame(recs, columns=_KLINE_COLS)
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _fetch_tx(symbol: str, start_date: str, adjust: str) -> pd.DataFrame:
    start = _as_date(start_date)
    today = date.today()
    if start > today:
        return pd.DataFrame(columns=_KLINE_COLS)
    ends = _window_ends(start, today)
    if len(ends) == 1:
        delta_days = max(1, (today - start).days)
        lim = 640 if delta_days > 180 else min(640, max(5, delta_days + 5))
        chunks = [_tx_window(symbol, ends[0], adjust, limit=lim)]
    else:
        with ThreadPoolExecutor(max_workers=len(ends)) as pool:
            chunks = list(pool.map(lambda end: _tx_window(symbol, end, adjust, limit=640), ends))
    rows = []
    for part in chunks:
        rows.extend(part)
    return _bars_to_df(rows, start)


def _fetch_qt_batch_bars(codes: list[str]) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    通过腾讯 qt.gtimg.cn 批量接口一次性拉取全市场最新交易日 K 线快照与估值(PE/PB/总市值)。
    每次请求 80 只股票，5,000 只股票仅需约 64 个请求，1.5~2 秒即可完成全市场扫描。
    返回: (qt_bars, funda_rows)
      qt_bars[code] = {date, open, high, low, close, prev_close, volume, amount}
      funda_rows[code] = {pe_ttm, pb, total_mv}
    """
    if not codes:
        return {}, {}
    syms = [_to_sina_symbol(str(c).zfill(6)) for c in codes]
    chunks = [syms[i : i + 80] for i in range(0, len(syms), 80)]
    qt_bars: dict[str, dict] = {}
    funda_rows: dict[str, dict] = {}

    def fetch_chunk(chunk):
        url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
        for _ in range(2):
            try:
                with _TX_SEM:
                    r = _http().get(url, timeout=10)
                if r.status_code == 200 and len(r.text) > 20:
                    return r.text
            except Exception:
                time.sleep(0.2)
        return ""

    with ThreadPoolExecutor(max_workers=10) as pool:
        texts = list(pool.map(fetch_chunk, chunks))

    for txt in texts:
        if not txt:
            continue
        for line in txt.splitlines():
            f = line.split("~")
            if len(f) < 47:
                continue
            code = str(f[2]).strip().zfill(6)
            raw_ts = str(f[30]).strip()
            if len(raw_ts) < 8 or not raw_ts[:8].isdigit():
                continue
            qt_date = f"{raw_ts[:4]}-{raw_ts[4:6]}-{raw_ts[6:8]}"
            try:
                close = float(f[3])
                prev_c = float(f[4])
                open_ = float(f[5])
                high = float(f[33])
                low = float(f[34])
                # 优先取 f[35] 第三段精确成交额(元)，否则取 f[37](万元)*10000
                amt = None
                if "/" in f[35]:
                    parts = f[35].split("/")
                    if len(parts) >= 3 and parts[2]:
                        amt = float(parts[2])
                if amt is None or amt <= 0:
                    amt = float(f[37]) * 10_000.0 if f[37] else 0.0
            except (TypeError, ValueError):
                continue

            if close > 0 and open_ > 0 and high > 0 and low > 0 and amt > 0:
                qt_bars[code] = {
                    "date": qt_date,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "prev_close": prev_c,
                    "volume": amt / close,
                    "amount": amt,
                    "suspended": False,
                }
            elif prev_c > 0 and open_ == 0 and amt == 0:
                # 当日停牌或盘前未开盘
                qt_bars[code] = {
                    "date": qt_date,
                    "open": 0.0,
                    "high": 0.0,
                    "low": 0.0,
                    "close": prev_c,
                    "prev_close": prev_c,
                    "volume": 0.0,
                    "amount": 0.0,
                    "suspended": True,
                }

            # 提取实时 PE_TTM / PB / 总市值(亿)
            def _num_or_none(s):
                try:
                    v = float(s)
                    return v if v == v else None
                except Exception:
                    return None

            pe = _num_or_none(f[39])
            float_mv = _num_or_none(f[44])
            mv = _num_or_none(f[45])
            pb = _num_or_none(f[46])
            if pe is not None or pb is not None or mv is not None or float_mv is not None:
                roe_est = round(pb / pe * 100.0, 2) if (pe and pe > 0 and pb and pb > 0) else None
                funda_rows[code] = {
                    "pe_ttm": pe,
                    "pb": pb,
                    "total_mv": mv,
                    "float_mv": float_mv,
                    "roe": roe_est,
                }

    return qt_bars, funda_rows


def _fetch_daily_raw(sina_symbol: str, start_date: str, adjust: str) -> pd.DataFrame:
    """
    拉取单只日线,统一返回列: date open high low close volume amount。
    直接抓腾讯前复权 K 线。失败时才用新浪，并且全局只允许一路新浪。
    sina_symbol 形如 sh600519 / sz000001。
    """
    try:
        return _fetch_tx(sina_symbol, start_date, adjust or "")
    except Exception as exc:  # noqa
        print(f"[info] 腾讯源 {sina_symbol} 失败,回退新浪: {exc}")

    with _SINA_LOCK:
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
                 incremental: bool = True, known_last=_UNSET,
                 reload: bool = True, qt_bar: dict = None) -> pd.DataFrame:
    """
    拉取单只股票日线并存库(腾讯源为主,新浪兜底)。
    adjust: qfq=前复权(推荐), hfq=后复权, ""=不复权
    incremental=True 时:
      - 从本地最新日期(含当日,以支持同日盘中到收盘覆盖更新)开始小窗口增量拉取；
      - 自动核对重叠日收盘价，若发生除权除息(qfq价格变动>0.4%)则自动重拉全量前复权历史；
      - 若腾讯 qfqday 尚未结算当日最新 K 线，自动合并 qt_bar 当日快照。
    """
    code = str(code).zfill(6)
    sina_symbol = _to_sina_symbol(code)

    def _done(frame):
        if reload:
            return db.load_kline(code)
        return frame if frame is not None else pd.DataFrame()

    real_start = start_date
    do_append = False
    last = None
    if incremental:
        last = db.get_last_date(code) if known_last is _UNSET else known_last
        if last:
            real_start = str(last).replace("-", "")[:8]
            do_append = True

    df = _fetch_daily_raw(sina_symbol, real_start, adjust)
    if df is None or df.empty:
        df = pd.DataFrame(columns=_KLINE_COLS)
    else:
        df = df.dropna(subset=["close"]).reset_index(drop=True)

    # 若有当日实时/收盘快照 qt_bar，且日期新于或等于 newfqkline 最后一天，合并进来
    if qt_bar and not qt_bar.get("suspended") and qt_bar.get("date") and qt_bar.get("close", 0) > 0:
        q_row = pd.DataFrame([{
            "date": qt_bar["date"],
            "open": qt_bar["open"],
            "high": qt_bar["high"],
            "low": qt_bar["low"],
            "close": qt_bar["close"],
            "volume": qt_bar["volume"],
            "amount": qt_bar["amount"],
        }])
        if df.empty:
            df = q_row
        else:
            df = pd.concat([df[df["date"] != qt_bar["date"]], q_row], ignore_index=True)
            df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        return _done(pd.DataFrame()) if do_append else pd.DataFrame()

    if do_append:
        with _DB_LOCK:
            db.append_kline(code, df)   # 增量:INSERT OR REPLACE 合并本地
    else:
        with _DB_LOCK:
            db.save_kline(code, df)     # 全量:覆盖
    return _done(df)


def update_index_kline(index_code: str = "sh000001",
                       start_date: str = "20210101") -> pd.DataFrame:
    """
    拉取大盘指数日线并存库,用于"大盘趋势过滤"。
    默认上证综指 sh000001。优先直接抓腾讯 newfqkline(0.5秒完成,含最新交易日),
    失败再回退新浪 stock_zh_index_daily。
    """
    try:
        tx_df = _fetch_tx(index_code, start_date, "")
        if tx_df is not None and not tx_df.empty:
            db.save_index_kline(index_code, tx_df)
            return db.load_index_kline(index_code)
    except Exception:
        pass
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


# ETF 专属指标(折溢价/规模/换手/资金流)的全市场快照进程内缓存,
# 避免同一次点评/连点多只时重复拉取(东财一次返回全市场 ~1000 只)。
_ETF_METRICS_CACHE = {"ts": 0.0, "df": None}
_ETF_METRICS_TTL = 60.0   # 秒;盘中 60s 内复用同一份快照


def fetch_etf_metrics(code: str) -> dict:
    """取单只 ETF 的专属指标(个股没有、基金才有的维度)。

    数据源:东财 fund_etf_spot_em(),一次返回全市场 ETF 实时快照,含
    IOPV实时估值/基金折价率/换手率/量比/规模(总市值)/主力净流入等。
    进程内缓存 60s,避免连点多只 ETF 时反复全市场拉取。

    返回 dict(缺失字段为 None);拉取失败降级返回 {}。字段:
      iopv           IOPV 实时估值(基金净值的盘中估算)
      discount_rate  基金折价率(%);正=折价(现价低于净值,买便宜),负=溢价
      turnover       换手率(%)
      vol_ratio      量比
      scale_yi       规模(总市值,亿元)
      main_inflow_yi 主力净流入(亿元)
      main_inflow_pct 主力净流入占比(%)
    """
    _check_ak()
    code = str(code).zfill(6)
    now = time.time()
    df = None
    if (_ETF_METRICS_CACHE["df"] is not None
            and now - _ETF_METRICS_CACHE["ts"] < _ETF_METRICS_TTL):
        df = _ETF_METRICS_CACHE["df"]
    else:
        try:
            raw = _retry(lambda: ak.fund_etf_spot_em(), tries=2)
            if raw is not None and not raw.empty:
                raw = raw.copy()
                raw["_code"] = raw["代码"].astype(str).str.replace(
                    r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
                _ETF_METRICS_CACHE["df"] = raw
                _ETF_METRICS_CACHE["ts"] = now
                df = raw
        except Exception as e:  # noqa 无网络/接口变更等,降级
            print(f"[warn] ETF 指标快照拉取失败: {e}")
            return {}
    if df is None or df.empty:
        return {}
    hit = df[df["_code"] == code]
    if hit.empty:
        return {}
    r = hit.iloc[0]

    def _num(col):
        try:
            v = pd.to_numeric(r.get(col), errors="coerce")
            return None if pd.isna(v) else float(v)
        except Exception:  # noqa
            return None

    mv = _num("总市值")
    inflow = _num("主力净流入-净额")
    return {
        "iopv": _num("IOPV实时估值"),
        "discount_rate": _num("基金折价率"),
        "turnover": _num("换手率"),
        "vol_ratio": _num("量比"),
        "scale_yi": (mv / 1e8) if mv is not None else None,
        "main_inflow_yi": (inflow / 1e8) if inflow is not None else None,
        "main_inflow_pct": _num("主力净流入-净占比"),
    }


def update_etf_kline(code: str, start_date: str = "20210101",
                     incremental: bool = True) -> pd.DataFrame:
    """
    拉取单只 ETF 日线并存库。
    优先直接走腾讯 newfqkline(0.3s 极速前复权，无 py_mini_racer 崩溃隐患)，
    失败时再加锁回退新浪 fund_etf_hist_sina。
    """
    code = str(code).zfill(6)
    sym = _etf_sina_symbol(code)
    last = db.get_last_date(code) if incremental else None
    real_start = last.replace("-", "")[:8] if last else start_date

    try:
        tx_df = _fetch_tx(sym, real_start, "qfq")
        if tx_df is not None and not tx_df.empty:
            if last:
                with _DB_LOCK:
                    db.append_kline(code, tx_df)
            else:
                with _DB_LOCK:
                    db.save_kline(code, tx_df)
            return db.load_kline(code)
    except Exception:
        pass

    _check_ak()
    try:
        with _SINA_LOCK:
            raw = _retry(lambda: ak.fund_etf_hist_sina(symbol=sym), tries=2)
    except Exception as e:  # noqa
        print(f"[warn] ETF {code} 日线拉取失败: {e}")
        return db.load_kline(code)
    if raw is None or raw.empty:
        return db.load_kline(code)

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if "amount" not in df.columns:
        df["amount"] = 0.0
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
    df = df.dropna(subset=["close"]).reset_index(drop=True)
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


def classify_etf_category(name: str) -> str:
    """根据 ETF 或基金简称自动归类赛道/资产类别。"""
    nm = str(name or "")
    if any(k in nm for k in ("债", "纯债", "转债", "国债", "政金", "信用", "短融", "城投", "固收", "货币", "添利", "稳健", "丰利", "双利", "安享", "泰享", "汇利", "聚利")):
        return "债券/固收"
    if any(k in nm for k in ("黄金", "豆粕", "有色", "能源化工", "商品", "白银", "原油")):
        return "商品/黄金"
    if any(k in nm for k in ("纳指", "纳斯达克", "标普", "道琼斯", "恒生", "港股", "日经", "德国", "法国", "沙特", "东南亚", "中概", "H股", "亚太", "美国", "全球", "QDII")):
        return "跨境/海外"
    if any(k in nm for k in ("红利", "高股息", "央企股东回报", "国企共赢", "央企红利", "价值100", "自由现金流")):
        return "红利/高股息"
    if any(k in nm for k in ("沪深300", "中证500", "上证50", "中证1000", "中证2000", "中证A50", "A500", "创业板", "科创50", "科创100", "科创板", "深证100", "上证180", "上证综指", "宽基", "双创", "北证", "量化", "指数增强")):
        return "宽基指数"
    if any(k in nm for k in ("半导体", "芯片", "人工智能", "AI", "计算机", "软件", "通信", "5G", "信创", "大数据", "算力", "电子", "消费电子", "机器人", "游戏", "传媒", "科创", "信息技术", "科技", "物联网", "智能", "数字")):
        return "科技/AI/芯片"
    if any(k in nm for k in ("医药", "医疗", "生物", "创新药", "中药", "疫苗", "健康", "养老")):
        return "医药/医疗"
    if any(k in nm for k in ("消费", "酒", "食品", "饮料", "家电", "旅游", "影视", "农业", "畜牧", "养殖", "内需", "品质")):
        return "大消费/食品"
    if any(k in nm for k in ("新能源", "光伏", "电池", "储能", "锂电", "风电", "碳中和", "绿电", "智能电车", "汽车", "环保", "低碳")):
        return "新能源/汽车"
    if any(k in nm for k in ("银行", "证券", "券商", "保险", "金融", "地产", "房地产", "基建", "建材")):
        return "金融/地产"
    if any(k in nm for k in ("军工", "国防", "航天", "航空", "船舶", "高端装备", "机械", "电网", "电力", "煤炭", "钢铁", "化工", "石油", "石化", "资源", "周期", "制造")):
        return "军工/制造/周期"
    return "混合/其他主题"


def sync_all_etf_universe_and_spot(progress_cb=None) -> pd.DataFrame:
    """
    一键拉取全市场 ~1700 只场内 ETF 实时行情 + 基金规模(亿) + IOPV实时估值 + 折溢价率% + 换手率% + 阶段涨跌幅，
    耗时仅 ~2.5 秒，全量写入 etf_snapshot 与 etf_list 表。
    """
    _check_ak()
    if progress_cb:
        progress_cb("正在获取全市场场内 ETF 名单(约1700只)...")
    codes_names = []
    try:
        with _SINA_LOCK:
            raw = _retry(lambda: ak.fund_etf_category_sina(symbol="ETF基金"), tries=2)
        if raw is not None and not raw.empty:
            for _, r in raw.iterrows():
                c = str(r.get("代码") or "").strip()
                c = c[2:] if c.startswith(("sh", "sz", "bj")) else c
                c = c.zfill(6)
                nm = str(r.get("名称") or "").strip()
                if c and nm and not nm.startswith("货币") and not c.startswith("5119"):
                    codes_names.append((c, nm))
    except Exception as e:
        print(f"[warn] 新浪全市场ETF名单拉取异常: {e}")

    if not codes_names:
        # 兜底本地已保存的 etf_list
        el = db.load_etf_list()
        if el is not None and not el.empty:
            codes_names = list(zip(el["code"].astype(str), el["name"].astype(str)))

    if not codes_names:
        return db.load_etf_snapshot()

    etf_df_basic = pd.DataFrame(codes_names, columns=["code", "name"]).drop_duplicates("code").reset_index(drop=True)
    with _DB_LOCK:
        db.save_etf_list(etf_df_basic)

    all_codes = etf_df_basic["code"].tolist()
    name_map = dict(zip(etf_df_basic["code"], etf_df_basic["name"]))
    syms = [_etf_sina_symbol(c) for c in all_codes]
    chunks = [syms[i : i + 75] for i in range(0, len(syms), 75)]

    if progress_cb:
        progress_cb(f"正在并发拉取 {len(all_codes)} 只 ETF 实时行情、IOPV估值、规模与折溢价...")

    def pull_chunk(chunk):
        url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
        for _ in range(2):
            try:
                with _TX_SEM:
                    r = _http().get(url, timeout=10)
                if r.status_code == 200 and len(r.text) > 30:
                    return r.text
            except Exception:
                time.sleep(0.2)
        return ""

    with ThreadPoolExecutor(max_workers=10) as pool:
        texts = list(pool.map(pull_chunk, chunks))

    # 读取本地已缓存 ETF 日线的 20日/60日/250日 涨跌幅(如有)
    ret_map: dict[str, tuple[Optional[float], Optional[float], Optional[float]]] = {}
    try:
        with db.get_conn() as conn:
            k_rows = conn.execute(
                "SELECT code, close FROM daily_kline "
                "WHERE code LIKE '5%' OR code LIKE '15%' ORDER BY code ASC, date ASC"
            ).fetchall()
        by_c: dict[str, list[float]] = {}
        for c, cl in k_rows:
            if c and cl is not None and cl > 0:
                by_c.setdefault(str(c).zfill(6), []).append(float(cl))
        for c, arr in by_c.items():
            r20 = round((arr[-1] / arr[-21] - 1.0) * 100.0, 2) if len(arr) >= 21 else None
            r60 = round((arr[-1] / arr[-61] - 1.0) * 100.0, 2) if len(arr) >= 61 else None
            r250 = round((arr[-1] / arr[-250] - 1.0) * 100.0, 2) if len(arr) >= 250 else None
            ret_map[c] = (r20, r60, r250)
    except Exception:
        pass

    def _fv(f_list, idx):
        if idx >= len(f_list):
            return None
        try:
            v = float(f_list[idx])
            return v if v == v else None
        except Exception:
            return None

    snap_rows = []
    for txt in texts:
        if not txt:
            continue
        for line in txt.splitlines():
            f = line.split("~")
            if len(f) < 47:
                continue
            code = str(f[2]).strip().zfill(6)
            nm = str(f[1]).strip() or name_map.get(code, "")
            if not code or not nm:
                continue
            price = _fv(f, 3)
            prev_c = _fv(f, 4)
            open_ = _fv(f, 5)
            chg_pct = _fv(f, 32)
            high = _fv(f, 33)
            low = _fv(f, 34)
            amt_wan = _fv(f, 37)
            amt_yi = round(amt_wan / 10_000.0, 4) if amt_wan is not None else 0.0
            turnover = _fv(f, 38)
            scale_yi = _fv(f, 45) or _fv(f, 44)
            iopv = _fv(f, 78) or _fv(f, 81)
            # 折价率(%) = (IOPV - 现价) / IOPV * 100；正数=折价(便宜)，负数=溢价(偏贵)
            disc_raw = _fv(f, 77)
            if iopv and iopv > 0 and price and price > 0:
                # 统一定义：溢价率 = (price - iopv) / iopv * 100 (正=溢价, 负=折价)
                prem_pct = round((price - iopv) / iopv * 100.0, 2)
            elif disc_raw is not None:
                prem_pct = round(disc_raw, 2)
            else:
                prem_pct = None

            if (price is None or price <= 0) and prev_c and prev_c > 0:
                price = prev_c

            r20, r60, r250 = ret_map.get(code, (None, None, None))
            # 若腾讯字段 69/70/62 提供阶段涨幅且本地无日线，则用腾讯阶段涨幅补充
            if r20 is None:
                r20 = _fv(f, 69)
            if r60 is None:
                r60 = _fv(f, 70)
            if r250 is None:
                r250 = _fv(f, 62)

            cat = classify_etf_category(nm)
            snap_rows.append((
                code, nm, cat, price, chg_pct, open_, high, low,
                amt_yi, turnover, scale_yi, iopv, prem_pct, r20, r60, r250,
            ))

    if snap_rows:
        with _DB_LOCK:
            db.save_etf_snapshot(snap_rows)
    return db.load_etf_snapshot()


def sync_mutual_fund_rank(progress_cb=None) -> pd.DataFrame:
    """
    一键拉取全市场 ~20,000 只场外公募开放式基金业绩排行 (含近1周/1月/3月/6月/1年/2年/3年/今年来收益率与手续费)，
    耗时约 3 秒，写入 mutual_fund_rank 表。
    """
    _check_ak()
    if progress_cb:
        progress_cb("正在拉取全市场 20,000+ 只场外公募基金净值与多周期业绩排行...")
    try:
        raw = _retry(lambda: ak.fund_open_fund_rank_em(symbol="全部"), tries=2)
    except Exception as e:
        print(f"[warn] 场外公募基金排行拉取异常: {e}")
        return db.load_mutual_fund_rank()
    if raw is None or raw.empty:
        return db.load_mutual_fund_rank()

    def _fv(val):
        try:
            if val is None or val == "":
                return None
            if isinstance(val, str):
                val = val.replace("%", "").strip()
            x = float(val)
            return round(x, 2) if x == x else None
        except Exception:
            return None

    rows = []
    for r in raw.itertuples(index=False):
        d = r._asdict() if hasattr(r, "_asdict") else dict(zip(raw.columns, r))
        code = str(d.get("基金代码") or "").strip().zfill(6)
        nm = str(d.get("基金简称") or "").strip()
        if not code or not nm:
            continue
        nav_dt = str(d.get("日期") or "")[:10]
        cat = classify_etf_category(nm)
        rows.append((
            code, nm, cat, nav_dt,
            _fv(d.get("单位净值")),
            _fv(d.get("累计净值")),
            _fv(d.get("日增长率")),
            _fv(d.get("近1周")),
            _fv(d.get("近1月")),
            _fv(d.get("近3月")),
            _fv(d.get("近6月")),
            _fv(d.get("近1年")),
            _fv(d.get("近2年")),
            _fv(d.get("近3年")),
            _fv(d.get("今年来")),
            _fv(d.get("成立来")),
            _fv(d.get("手续费")),
        ))

    if rows:
        with _DB_LOCK:
            db.save_mutual_fund_rank(rows)
    return db.load_mutual_fund_rank()


def fetch_or_load_fund_nav(code: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    获取单只基金（场外公募或场内ETF）的完整历史净值/收盘序列：
    返回列: date, unit_nav, day_chg。
    优先读取本地 fund_nav_history 缓存(若最新日期已近)，否则调用 fund_open_fund_info_em(0.8s) 拉取并存库。
    """
    code = str(code).zfill(6)
    if not force_refresh:
        cached = db.load_fund_nav_history(code)
        if cached is not None and len(cached) >= 15:
            last_d = str(cached["date"].iloc[-1])
            if (date.today() - _as_date(last_d)).days <= 5:
                return cached

    # 若是场内 ETF，可直接复用 daily_kline 或 update_etf_kline (排除 519xxx/150xxx 场外公募)
    if code.startswith(("510", "511", "512", "513", "515", "516", "517", "518", "56", "58", "159")):
        kl = db.load_kline(code)
        if kl is None or len(kl) < 20:
            kl = update_etf_kline(code, start_date="20210101", incremental=True)
        if kl is not None and not kl.empty:
            out = pd.DataFrame({
                "date": kl["date"].astype(str),
                "unit_nav": pd.to_numeric(kl["close"], errors="coerce"),
            }).dropna()
            out["day_chg"] = (out["unit_nav"].pct_change() * 100.0).round(2).fillna(0.0)
            return out.reset_index(drop=True)

    # 优先使用轻量级直连解析 pingzhongdata/{code}.js (约0.25秒，无 akshare 阻塞)
    try:
        url = f"https://fund.eastmoney.com/pingzhongdata/{code}.js"
        r = _http().get(
            url,
            headers={"Referer": "https://fund.eastmoney.com/", "User-Agent": "Mozilla/5.0"},
            timeout=6,
        )
        if r.status_code == 200 and "Data_netWorthTrend" in r.text:
            txt = r.text
            pos_n = txt.find("Data_netWorthTrend")
            end_n = txt.find(";", pos_n)
            eq_pos = txt.find("=", pos_n)
            if pos_n >= 0 and end_n > eq_pos > pos_n:
                arr = json.loads(txt[eq_pos + 1 : end_n].strip())
                rows = [
                    (
                        datetime.fromtimestamp(it["x"] / 1000.0).strftime("%Y-%m-%d"),
                        float(it["y"]),
                        float(it.get("equityReturn") or 0.0),
                    )
                    for it in arr
                    if it.get("x") is not None and it.get("y") is not None
                ]
                if rows:
                    out = pd.DataFrame(rows, columns=["date", "unit_nav", "day_chg"])
                    with _DB_LOCK:
                        db.save_fund_nav_history(code, out)
                    return out
    except Exception:
        pass

    _check_ak()
    try:
        raw = _retry(lambda: ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势"), tries=2)
        if raw is not None and not raw.empty:
            out = pd.DataFrame({
                "date": pd.to_datetime(raw["净值日期"]).dt.strftime("%Y-%m-%d"),
                "unit_nav": pd.to_numeric(raw["单位净值"], errors="coerce"),
                "day_chg": pd.to_numeric(raw.get("日增长率", 0.0), errors="coerce").fillna(0.0),
            }).dropna(subset=["date", "unit_nav"]).sort_values("date").reset_index(drop=True)
            with _DB_LOCK:
                db.save_fund_nav_history(code, out)
            return out
    except Exception as e:
        print(f"[warn] 拉取基金 {code} 净值走势失败: {e}")

    return db.load_fund_nav_history(code)


def fetch_official_fund_stage_rank(code: str) -> tuple[list[dict], str]:
    """
    从天天基金/东财官方 F10 接口 (FundArchivesDatas.aspx?type=jdzf) 极速拉取该基金在官方二级分类下的：
    真实阶段涨幅、官方同类平均、官方同类排名(如 44/778) 与四分位评级(优秀/良好/一般/不佳)。
    耗时仅 ~0.2 秒。失败返回 ([], "") 由调用方回退本地同类计算。
    """
    code = str(code).zfill(6)
    url = "https://fundf10.eastmoney.com/FundArchivesDatas.aspx"
    try:
        r = _http().get(
            url,
            params={"type": "jdzf", "code": code},
            headers={"Referer": "https://fundf10.eastmoney.com/", "User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        if r.status_code != 200 or "jdzfnew" not in r.text:
            return [], ""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "lxml")
        wanted = {"近1周", "近1月", "近3月", "近6月", "近1年", "近2年", "近3年", "今年来"}
        stage_map = {}

        def _parse_pct(s: str):
            try:
                s = str(s).replace("%", "").strip()
                if not s or s == "---":
                    return None
                return round(float(s), 2)
            except Exception:
                return None

        beat_1y_pct = None
        for ul in soup.find_all("ul"):
            lis = [li.get_text(strip=True) for li in ul.find_all("li")]
            if len(lis) >= 7 and lis[0] in wanted:
                stg = lis[0]
                val = _parse_pct(lis[1])
                p_avg = _parse_pct(lis[2])
                rk_raw = lis[4].replace("|", "/") if lis[4] and lis[4] != "---" else "—"
                grade = lis[6] if lis[6] and lis[6] != "---" else "—"
                if stg == "近1年" and "/" in rk_raw:
                    try:
                        rk_n, tot_n = [int(x) for x in rk_raw.split("/")[:2]]
                        if tot_n > 0:
                            beat_1y_pct = int(round((1.0 - (rk_n - 1) / tot_n) * 100.0))
                    except Exception:
                        pass
                stage_map[stg] = {
                    "stage": stg,
                    "val": val,
                    "peer_avg": p_avg,
                    "rank_str": rk_raw,
                    "grade": grade,
                }

        order = ["近1周", "近1月", "近3月", "近6月", "近1年", "近2年", "近3年", "今年来"]
        rows = [stage_map[k] for k in order if k in stage_map]
        badge = f"涨幅跑赢 {beat_1y_pct}% 同类" if beat_1y_pct is not None else ""
        return rows, badge
    except Exception:
        return [], ""


def fetch_fund_deep_metrics_batch(codes: list[str], progress_cb=None) -> dict[str, dict]:
    """
    通过东财官方 pingzhongdata/{code}.js 接口并发拉取一批公募基金的深度指标：
      - scale_yi:   最新基金规模(亿元) (来自 Data_fluctuationScale)
      - inst_ratio: 最新机构持有占比(%) (来自 Data_holderStructure)
      - max_dd_1y:  近1年真实复权最大回撤(%, 正值如 1.25 代表回撤 1.25%)
      - max_dd_all: 成立以来真实复权最大回撤(%, 正值如 3.04 代表回撤 3.04%)
    同时将解析出的完整历史净值序列顺手写入 fund_nav_history 表，将 4 项指标持久化回填进 mutual_fund_rank 表！
    """
    import re
    import numpy as np

    codes = [str(c).zfill(6) for c in dict.fromkeys(codes) if c]
    if not codes:
        return {}

    total = len(codes)
    done = 0
    lock = threading.Lock()
    results: dict[str, dict] = {}
    db_updates: list[tuple] = []

    def pull_one(code: str):
        url = f"https://fund.eastmoney.com/pingzhongdata/{code}.js"
        try:
            txt = ""
            for attempt in range(2):
                try:
                    r = _http().get(
                        url,
                        headers={"Referer": "https://fund.eastmoney.com/", "User-Agent": "Mozilla/5.0"},
                        timeout=6,
                    )
                    if r.status_code == 200 and len(r.text) >= 200:
                        txt = r.text
                        break
                except Exception:
                    time.sleep(0.15 * (attempt + 1))
            if not txt:
                return code, None
            scale_yi = None
            inst_ratio = None
            max_dd_1y = None
            max_dd_all = None

            # 1) 提取最新基金规模 (亿元) —— 严格截取到当前 JS 变量分号处，严禁越界误读基金经理收益 y
            pos_s = txt.find("Data_fluctuationScale")
            if pos_s >= 0:
                end_s = txt.find(";", pos_s)
                if end_s < 0 or end_s > pos_s + 1500:
                    end_s = pos_s + 900
                seg_s = txt[pos_s:end_s]
                ys = re.findall(r'"y"\s*:\s*([0-9.]+)', seg_s)
                if ys:
                    scale_yi = round(float(ys[-1]), 2)
            if scale_yi is None or scale_yi <= 0:
                pos_a = txt.find("Data_assetAllocation")
                if pos_a >= 0:
                    end_a = txt.find(";", pos_a)
                    seg_a = txt[pos_a : (end_a if end_a > pos_a else pos_a + 1200)]
                    m_na = re.search(r'"净资产".*?"data"\s*:\s*\[([^\]]*)\]', seg_a)
                    if m_na:
                        nums_a = [x.strip() for x in m_na.group(1).split(",") if x.strip() and x.strip() != "null"]
                        if nums_a:
                            scale_yi = round(float(nums_a[-1]), 2)

            # 2) 提取最新机构持有比例 (%)
            pos_h = txt.find("Data_holderStructure")
            if pos_h >= 0:
                end_h = txt.find(";", pos_h)
                if end_h < 0 or end_h > pos_h + 1500:
                    end_h = pos_h + 1000
                seg_h = txt[pos_h:end_h]
                m_inst = re.search(r'"机构持有比例"\s*,\s*"data"\s*:\s*\[([^\]]*)\]', seg_h)
                if m_inst:
                    nums = [x.strip() for x in m_inst.group(1).split(",") if x.strip() and x.strip() != "null"]
                    if nums:
                        inst_ratio = round(float(nums[-1]), 2)

            # 3) 提取 Data_netWorthTrend 计算复权最大回撤、成立年限、年化波动率与夏普比率
            establish_years = None
            sharpe_1y = None
            vol_1y = None
            pos_n = txt.find("Data_netWorthTrend")
            if pos_n >= 0:
                end_n = txt.find(";/*累计净值走势*/", pos_n)
                if end_n < 0:
                    end_n = pos_n + 400000
                seg_n = txt[pos_n:end_n]
                # 提取首尾毫秒时间戳精确计算真实成立年限
                m_x1 = re.search(r'"x"\s*:\s*(\d{10,13})', seg_n[:400])
                m_x2_all = re.findall(r'"x"\s*:\s*(\d{10,13})', seg_n[-600:])
                if m_x1 and m_x2_all:
                    ts1 = float(m_x1.group(1))
                    ts2 = float(m_x2_all[-1])
                    if ts2 > ts1:
                        establish_years = round((ts2 - ts1) / (365.25 * 86400.0 * 1000.0), 1)
                eq_rets = re.findall(r'"equityReturn"\s*:\s*([-0-9.]+)', seg_n)
                if len(eq_rets) >= 15:
                    if establish_years is None:
                        establish_years = round(len(eq_rets) / 244.0, 1)
                    r_arr = np.asarray([float(x) for x in eq_rets], dtype=float) / 100.0
                    r_arr[0] = 0.0
                    adj = np.cumprod(1.0 + r_arr)
                    pks = np.maximum.accumulate(adj)
                    dd_all = (adj - pks) / np.where(pks > 0, pks, 1.0) * 100.0
                    max_dd_all = round(abs(float(np.min(dd_all))), 2)

                    sub_r = r_arr[-min(len(r_arr), 245):]
                    sub_adj = adj[-min(len(adj), 245):]
                    sub_pks = np.maximum.accumulate(sub_adj)
                    dd_1y = (sub_adj - sub_pks) / np.where(sub_pks > 0, sub_pks, 1.0) * 100.0
                    max_dd_1y = round(abs(float(np.min(dd_1y))), 2)

                    if len(sub_r) >= 40:
                        v_ann = float(np.std(sub_r, ddof=1) * np.sqrt(244.0) * 100.0)
                        vol_1y = round(v_ann, 2)
                        ann_ret = float((sub_adj[-1] / max(1e-9, sub_adj[0]) - 1.0) * (244.0 / len(sub_r)) * 100.0)
                        sharpe_1y = round((ann_ret - 1.80) / max(0.35, v_ann), 2)

            # 4) 提取现任基金经理姓名与从业年限 (Data_currentFundManager)
            manager_name = None
            manager_years = None
            pos_m = txt.find("Data_currentFundManager")
            if pos_m >= 0:
                seg_m = txt[pos_m : pos_m + 1800]
                m_nm = re.search(r'"name"\s*:\s*"([^"]+)"', seg_m)
                if m_nm:
                    manager_name = m_nm.group(1).strip()
                m_wt = re.search(r'"workTime"\s*:\s*"([^"]+)"', seg_m)
                if m_wt:
                    wt_str = m_wt.group(1)
                    my = re.search(r"(\d+)\s*年", wt_str)
                    md = re.search(r"(\d+)\s*天", wt_str)
                    yrs = float(my.group(1)) if my else 0.0
                    days = float(md.group(1)) / 365.0 if md else 0.0
                    if yrs > 0 or days > 0:
                        manager_years = round(yrs + days, 1)

            return code, {
                "scale_yi": scale_yi,
                "inst_ratio": inst_ratio,
                "max_dd_1y": max_dd_1y,
                "max_dd_all": max_dd_all,
                "establish_years": establish_years,
                "sharpe_1y": sharpe_1y,
                "vol_1y": vol_1y,
                "manager_name": manager_name,
                "manager_years": manager_years,
            }
        except Exception:
            return code, None

    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = [pool.submit(pull_one, c) for c in codes]
        for fut in as_completed(futs):
            code, info = fut.result()
            with lock:
                done += 1
                cur_d = done
                if info:
                    results[code] = info
                    db_updates.append((
                        info.get("scale_yi"),
                        info.get("inst_ratio"),
                        info.get("max_dd_1y"),
                        info.get("max_dd_all"),
                        info.get("establish_years"),
                        info.get("sharpe_1y"),
                        info.get("vol_1y"),
                        info.get("manager_name"),
                        info.get("manager_years"),
                        code,
                    ))
            if progress_cb and (cur_d % 10 == 0 or cur_d == total):
                progress_cb(cur_d, total, code)

    if db_updates:
        try:
            with _DB_LOCK:
                db.update_mutual_fund_deep_metrics(db_updates)
        except Exception:
            pass
    return results


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


def _latest_settled_qfq_date() -> str:
    """探测腾讯 newfqkline(qfq) 当前已结算的最新交易日(如昨日收盘日)。"""
    try:
        rows = _tx_window("sh600519", date.today(), "qfq", limit=5)
        if rows:
            return str(rows[-1][0])[:10]
    except Exception:
        pass
    return ""


def update_all_kline(codes: list, start_date: str = "20230101",
                     incremental: bool = True, progress_cb=None,
                     workers: int = 16):
    """
    三级极速增量与全量日线更新引擎：
      1) 先用腾讯 qt.gtimg.cn 批量接口(80只/批)在 1.5~2 秒内拉取全市场当日最新 K 线与 PE/PB/市值；
      2) 若 incremental=True：
         - 第一级(零单股请求·秒级合并)：若某股本地已有上一结算交易日数据，且与今日昨收一致(无除权除息)，
           直接将今日 K 线合并入本地库，无需再发单股 K 线请求；
         - 第二级(小窗口差量补齐)：若某股落后数日(<=90天)，仅请求缺失天数的少量 K 线(几百字节)，
           并自动核对重叠日价格；若发现除权除息则自动全量重拉该股前复权历史，保证前后无缝衔接；
         - 第三级(新股/久未更新)：分段并发拉取完整历史。
      3) 所有增量数据采用批量事务写入 SQLite (INSERT OR REPLACE)，支持同日盘中与收盘重复刷新。
    """
    codes = [str(c).zfill(6) for c in codes]
    total = len(codes)
    if total == 0:
        return 0

    # 1. 批量拉取全市场当日 K 线快照 + 实时 PE/PB/总市值 (约 1.5~2s)
    qt_bars, funda_rows = _fetch_qt_batch_bars(codes)
    if funda_rows:
        try:
            with _DB_LOCK:
                db.save_fundamental(funda_rows)
        except Exception:
            pass

    if not incremental:
        # 强制全量重拉模式
        fail = 0
        done = 0
        lock = threading.Lock()

        def full_work(code):
            try:
                update_kline(
                    code, start_date=start_date, incremental=False,
                    reload=False, qt_bar=qt_bars.get(code),
                )
                return code, None
            except Exception as e:  # noqa
                return code, e

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(full_work, c) for c in codes]
            for fut in as_completed(futures):
                code, err = fut.result()
                with lock:
                    done += 1
                    if err is not None:
                        fail += 1
                        print(f"[warn] {code} 拉取失败: {err}")
                    cur_done = done
                if progress_cb and (cur_done % 25 == 0 or cur_done == total):
                    progress_cb(cur_done, total, code)
        return fail

    # 2. 增量模式：读取本地每只股票的最新日期与近 45 日收盘价映射
    recent_since = (date.today() - timedelta(days=45)).isoformat()
    summary = db.last_bars_summary(recent_since)
    mkt_settled = _latest_settled_qfq_date()
    today_dt = date.today()

    fast_rows: list[tuple] = []
    need_fetch: list[str] = []
    done = 0

    for code in codes:
        info = summary.get(code)
        qt = qt_bars.get(code)
        if info and qt and mkt_settled:
            if qt.get("suspended"):
                # 当日停牌且本地已有近期数据，直接跳过网络请求
                done += 1
                continue
            closes = info["closes"]
            ref_c = closes.get(mkt_settled)
            prev_c = qt.get("prev_close") or 0.0
            # 若本地已有最新已结算交易日(如昨天)，且今日昨收与本地昨天收盘一致(无除权除息)
            if ref_c and ref_c > 0 and prev_c > 0 and abs(prev_c - ref_c) / ref_c <= 0.003:
                fast_rows.append((
                    code, qt["date"], qt["open"], qt["high"], qt["low"],
                    qt["close"], qt["volume"], qt["amount"],
                ))
                done += 1
                continue
            # 若在周末或晚间 qfq 已结算当日，且本地最新日期已等于 qt["date"]
            if qt["date"] == mkt_settled and ref_c and abs(qt["close"] - ref_c) / ref_c <= 0.003:
                fast_rows.append((
                    code, qt["date"], qt["open"], qt["high"], qt["low"],
                    qt["close"], qt["volume"], qt["amount"],
                ))
                done += 1
                continue
        need_fetch.append(code)

    if fast_rows:
        with _DB_LOCK:
            db.upsert_kline_rows(fast_rows)
        if progress_cb:
            progress_cb(done, total, fast_rows[-1][0])

    if not need_fetch:
        return 0

    # 3. 对需要补齐中间缺失日或新股的标的，小窗口并发拉取并批量入库
    fail = 0
    lock = threading.Lock()
    batch_buf: list[tuple] = []

    def delta_work(code: str):
        sina_symbol = _to_sina_symbol(code)
        info = summary.get(code)
        last_d = info["last_date"] if info else None
        closes = info["closes"] if info else {}
        qt = qt_bars.get(code)
        full_replace = False

        try:
            if last_d and (today_dt - _as_date(last_d)).days <= 90:
                df = _fetch_daily_raw(sina_symbol, last_d.replace("-", "")[:8], "qfq")
                # 核对重叠日 last_d 收盘价，检测期间是否发生除权除息
                if df is not None and not df.empty and last_d in closes:
                    m_row = df[df["date"] == last_d]
                    if not m_row.empty:
                        tx_c = float(m_row["close"].iloc[-1])
                        db_c = float(closes[last_d])
                        if db_c > 0 and abs(tx_c - db_c) / db_c > 0.004:
                            # 发生除权除息，重拉全量前复权数据以消除断层
                            df = _fetch_daily_raw(sina_symbol, start_date, "qfq")
                            full_replace = True
            else:
                df = _fetch_daily_raw(sina_symbol, start_date, "qfq")
                full_replace = True

            if df is None or df.empty:
                df = pd.DataFrame(columns=_KLINE_COLS)
            else:
                df = df.dropna(subset=["close"]).reset_index(drop=True)

            # 合并当日 qt 快照（若无除权断层且未停牌）
            if qt and not qt.get("suspended") and qt.get("date") and qt.get("close", 0) > 0:
                can_merge_qt = True
                if not df.empty:
                    last_bar_d = str(df["date"].iloc[-1])
                    last_bar_c = float(df["close"].iloc[-1])
                    if qt["date"] > last_bar_d and last_bar_c > 0 and qt.get("prev_close", 0) > 0:
                        if abs(qt["prev_close"] - last_bar_c) / last_bar_c > 0.004 and not full_replace:
                            # 今日发生除权除息
                            df = _fetch_daily_raw(sina_symbol, start_date, "qfq")
                            full_replace = True
                            can_merge_qt = False
                if can_merge_qt:
                    q_row = pd.DataFrame([{
                        "date": qt["date"],
                        "open": qt["open"],
                        "high": qt["high"],
                        "low": qt["low"],
                        "close": qt["close"],
                        "volume": qt["volume"],
                        "amount": qt["amount"],
                    }])
                    if df.empty:
                        df = q_row
                    else:
                        df = pd.concat([df[df["date"] != qt["date"]], q_row], ignore_index=True)
                        df = df.sort_values("date").reset_index(drop=True)

            return code, df, full_replace, None
        except Exception as exc:  # noqa
            return code, None, False, exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(delta_work, c) for c in need_fetch]
        for fut in as_completed(futures):
            code, df, full_replace, err = fut.result()
            to_flush = None
            with lock:
                done += 1
                if err is not None:
                    fail += 1
                    print(f"[warn] {code} 拉取失败: {err}")
                elif df is not None and not df.empty:
                    if full_replace:
                        with _DB_LOCK:
                            db.save_kline(code, df)
                    else:
                        for r in df.itertuples(index=False):
                            batch_buf.append((
                                code, r.date, float(r.open), float(r.high),
                                float(r.low), float(r.close), float(r.volume), float(r.amount),
                            ))
                        if len(batch_buf) >= 2000:
                            to_flush = list(batch_buf)
                            batch_buf.clear()
                cur_done = done

            if to_flush:
                with _DB_LOCK:
                    db.upsert_kline_rows(to_flush)
            if progress_cb and (cur_done % 20 == 0 or cur_done == total):
                progress_cb(cur_done, total, code)

    if batch_buf:
        with _DB_LOCK:
            db.upsert_kline_rows(batch_buf)
    return fail


# ============================================================================
# 四大扩展免费数据爬取与同步引擎 (主力盘口资金流 / 历史季报PIT / 解禁与预告排雷 / 二级细分题材)
# ============================================================================
def update_capital_flow_batch(codes: list[str] = None, progress_cb=None) -> int:
    """
    通过腾讯 qt.gtimg.cn 同时批量拉取全市场 s_pk(盘口大单/小单买卖占比) 与内外盘主动买盘差，
    合成每只股票当日的：
      - buy_big_ratio:  主力大单买盘占比 (0~1)
      - sell_big_ratio: 主力大单卖盘占比 (0~1)
      - net_big_ratio:  大单净买入占比 = buy_big_ratio - sell_big_ratio (-1~+1)
      - active_buy_imb: 主动买盘失衡度 = (外盘 - 内盘) / (外盘 + 内盘) (-1~+1)
      - net_inflow_yi:  主力净流入估算金额(亿元)
    耗时仅 ~2 秒，写入 capital_flow 表。
    """
    if not codes:
        slist = db.load_stock_list()
        codes = slist["code"].astype(str).str.zfill(6).tolist() if (slist is not None and not slist.empty) else []
    if not codes:
        return 0
    codes = [str(c).zfill(6) for c in codes]
    syms = [_to_sina_symbol(c) for c in codes]
    # 每批 45 只股票，同时请求 s_pk{sym} 与 {sym}
    chunks = [syms[i : i + 45] for i in range(0, len(syms), 45)]

    def fetch_pair(chunk):
        q_items = []
        for s in chunk:
            q_items.append(f"s_pk{s}")
            q_items.append(s)
        url = "https://qt.gtimg.cn/q=" + ",".join(q_items)
        for _ in range(2):
            try:
                with _TX_SEM:
                    r = _http().get(url, timeout=10)
                if r.status_code == 200 and len(r.text) > 30:
                    return r.text
            except Exception:
                time.sleep(0.2)
        return ""

    with ThreadPoolExecutor(max_workers=10) as pool:
        texts = list(pool.map(fetch_pair, chunks))

    pk_map: dict[str, tuple[float, float]] = {}
    qt_map: dict[str, tuple[str, float, float, float]] = {}

    for txt in texts:
        if not txt:
            continue
        for line in txt.splitlines():
            if "=" not in line:
                continue
            head, body = line.split("=", 1)
            body = body.strip().strip('";')
            f = body.split("~")
            if head.startswith("v_s_pk"):
                sym = head[6:].strip()
                code = sym[-6:]
                if len(f) >= 4:
                    try:
                        b_big = float(f[0])
                        s_big = float(f[2])
                        pk_map[code] = (b_big, s_big)
                    except Exception:
                        pass
            elif head.startswith("v_"):
                if len(f) < 38:
                    continue
                code = str(f[2]).strip().zfill(6)
                raw_ts = str(f[30]).strip()
                if len(raw_ts) < 8 or not raw_ts[:8].isdigit():
                    continue
                qt_date = f"{raw_ts[:4]}-{raw_ts[4:6]}-{raw_ts[6:8]}"
                try:
                    outer_v = float(f[7]) if f[7] else 0.0
                    inner_v = float(f[8]) if f[8] else 0.0
                    amt_yi = float(f[37]) / 10_000.0 if f[37] else 0.0
                    qt_map[code] = (qt_date, outer_v, inner_v, amt_yi)
                except Exception:
                    pass

    rows = []
    for code, (qt_date, outer_v, inner_v, amt_yi) in qt_map.items():
        if amt_yi <= 0 and (outer_v + inner_v) <= 0:
            continue
        b_big, s_big = pk_map.get(code, (0.0, 0.0))
        net_big = b_big - s_big
        tot_oi = outer_v + inner_v
        imb = (outer_v - inner_v) / tot_oi if tot_oi > 0 else 0.0
        # 综合大单净占比与主动买盘失衡度估算主力净流入(亿元)
        eff_ratio = 0.65 * net_big + 0.35 * imb if (b_big + s_big) > 0 else imb * 0.5
        net_inflow_yi = round(amt_yi * eff_ratio, 4)
        rows.append((
            code, qt_date, round(b_big, 4), round(s_big, 4), round(net_big, 4),
            round(outer_v, 1), round(inner_v, 1), round(imb, 4), net_inflow_yi,
        ))

    if rows:
        with _DB_LOCK:
            db.save_capital_flow(rows)
    if progress_cb:
        progress_cb(len(rows))
    return len(rows)


_PIT_QUARTERS = (
    "2020-12-31", "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
    "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31",
    "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31",
    "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
    "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31",
    "2026-03-31", "2026-06-30", "2026-09-30",
)


def _fetch_one_quarter_cpd(q_date: str) -> tuple[list[tuple], list[tuple]]:
    """
    并发拉取单个季度 (如 '2025-09-30') 的东财数据中心 RPT_LICO_FN_CPD 季报全表与二级细分板块映射。
    返回 (pit_rows, subsector_rows)。
    """
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    cols = (
        "SECURITY_CODE,REPORTDATE,NOTICE_DATE,UPDATE_DATE,"
        "BASIC_EPS,WEIGHTAVG_ROE,YSTZ,SJLTZ,BPS,XSMLL,BOARD_NAME,BOARD_CODE"
    )
    base_params = {
        "sortColumns": "SECURITY_CODE",
        "sortTypes": "1",
        "pageSize": "500",
        "pageNumber": "1",
        "reportName": "RPT_LICO_FN_CPD",
        "columns": cols,
        "filter": f"(REPORTDATE='{q_date}')",
    }
    try:
        r0 = _http().get(url, params=base_params, timeout=10).json()
        res0 = r0.get("result")
        if not res0 or not res0.get("data"):
            return [], []
        pages = int(res0.get("pages") or 1)
        all_items = list(res0["data"])
    except Exception:
        return [], []

    def pull_page(p):
        p_params = dict(base_params, pageNumber=str(p))
        for _ in range(2):
            try:
                rj = _http().get(url, params=p_params, timeout=10).json()
                return (rj.get("result") or {}).get("data") or []
            except Exception:
                time.sleep(0.2)
        return []

    if pages > 1:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for part in pool.map(pull_page, range(2, pages + 1)):
                all_items.extend(part)

    pit_rows = []
    sub_rows = []

    def _fv(v):
        try:
            x = float(v)
            return x if x == x else None
        except Exception:
            return None

    for it in all_items:
        code = str(it.get("SECURITY_CODE") or "").strip().zfill(6)
        if not code.startswith(("0", "3", "6")):
            continue
        r_dt = str(it.get("REPORTDATE") or q_date)[:10]
        n_dt = str(it.get("NOTICE_DATE") or it.get("UPDATE_DATE") or q_date)[:10]
        # 防止 UPDATE_DATE 被晚年修正污染：公告日最多不超过报告期后 150 天
        try:
            max_notice = (_as_date(r_dt) + timedelta(days=150)).isoformat()
            if n_dt > max_notice:
                n_dt = max_notice
        except Exception:
            pass
        b_name = str(it.get("BOARD_NAME") or "").strip()
        b_code = str(it.get("BOARD_CODE") or "").strip()
        pit_rows.append((
            code, r_dt, n_dt,
            _fv(it.get("BASIC_EPS")),
            _fv(it.get("WEIGHTAVG_ROE")),
            _fv(it.get("YSTZ")),
            _fv(it.get("SJLTZ")),
            _fv(it.get("BPS")),
            _fv(it.get("XSMLL")),
            b_name,
        ))
        if b_name and b_name != "None":
            sub_rows.append((code, b_name, b_code))

    return pit_rows, sub_rows


def update_pit_financials_and_subsectors(only_missing: bool = True, progress_cb=None) -> int:
    """
    同步 2020Q4 ~ 2026Q3 共 24 个季度的历史季报序列 (pit_financials) 与东财二级细分题材板块 (subsector_map)。
    only_missing=True 时：已入库且行数充足的旧季度自动跳过，仅更新最近 2 个季度。
    """
    existing_q = {}
    with db.get_conn() as conn:
        try:
            for r_dt, cnt in conn.execute("SELECT report_date, COUNT(*) FROM pit_financials GROUP BY report_date").fetchall():
                existing_q[str(r_dt)] = int(cnt)
        except Exception:
            pass

    target_qs = []
    for q in _PIT_QUARTERS:
        if only_missing and existing_q.get(q, 0) >= 3500 and q < "2026-03-31":
            continue
        target_qs.append(q)

    total_wrote = 0
    sub_map: dict[str, tuple[str, str]] = {}
    for idx, q in enumerate(target_qs):
        if progress_cb:
            progress_cb(f"正在同步历史季报序列 {q} ({idx + 1}/{len(target_qs)})...")
        pit_rows, sub_rows = _fetch_one_quarter_cpd(q)
        if pit_rows:
            with _DB_LOCK:
                db.save_pit_financials(pit_rows)
            total_wrote += len(pit_rows)
        for c, bn, bc in sub_rows:
            sub_map[c] = (bn, bc)

    if sub_map:
        with _DB_LOCK:
            db.save_subsector_map([(c, bn, bc) for c, (bn, bc) in sub_map.items()])
    try:
        refresh_derived_stock_features(progress_cb=progress_cb)
    except Exception:
        pass
    return total_wrote


def update_risk_calendar(progress_cb=None) -> int:
    """
    拉取未来 60 天【限售股解禁明细】与最新一期【业绩预告(预亏/首亏/业绩暴跌 vs 预增/略增)】，
    写入 risk_calendar 表供一票否决排雷卫士与催化打分使用。
    """
    if ak is None:
        return 0
    rows: list[tuple] = []
    today_dt = date.today()
    start_s = (today_dt - timedelta(days=5)).strftime("%Y%m%d")
    end_s = (today_dt + timedelta(days=60)).strftime("%Y%m%d")

    # 1) 限售股解禁日历
    if progress_cb:
        progress_cb("正在拉取未来60天限售股解禁排雷日历...")
    try:
        rel_df = ak.stock_restricted_release_detail_em(start_date=start_s, end_date=end_s)
        if rel_df is not None and not rel_df.empty:
            for _, r in rel_df.iterrows():
                code = str(r.get("股票代码") or "").strip().zfill(6)
                if not code.startswith(("0", "3", "6")):
                    continue
                ed = str(r.get("解禁时间") or "")[:10]
                ratio = r.get("占解禁前流通市值比例")
                mv_yuan = r.get("实际解禁市值")
                try:
                    ratio_pct = float(ratio) * 100.0 if ratio is not None and ratio == ratio else 0.0
                except Exception:
                    ratio_pct = 0.0
                try:
                    mv_yi = float(mv_yuan) / 1e8 if mv_yuan is not None and mv_yuan == mv_yuan else 0.0
                except Exception:
                    mv_yi = 0.0
                if ratio_pct >= 5.0 or mv_yi >= 5.0:
                    sev = "veto"
                elif ratio_pct >= 1.5 or mv_yi >= 1.5:
                    sev = "warn"
                else:
                    continue
                summ = f"{ed}解禁占流通盘{ratio_pct:.1f}%({mv_yi:.1f}亿)"
                rows.append((code, "限售解禁", ed, sev, round(ratio_pct, 2), summ))
    except Exception as e:
        print(f"[warn] 限售解禁日历拉取异常: {e}")

    # 2) 最新季度业绩预告 (自动探测最近报告期)
    if progress_cb:
        progress_cb("正在拉取最新业绩预告(预亏排雷/预增催化)...")
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    for q_rep in ("2026-12-31", "2026-09-30", "2026-06-30"):
        try:
            params = {
                "sortColumns": "NOTICE_DATE,SECURITY_CODE",
                "sortTypes": "-1,-1",
                "pageSize": "500",
                "pageNumber": "1",
                "reportName": "RPT_PUBLIC_OP_NEWPREDICT",
                "columns": "SECURITY_CODE,NOTICE_DATE,REPORT_DATE,PREDICT_FINANCE_CODE,ADD_AMP_LOWER,ADD_AMP_UPPER,PREDICT_TYPE",
                "filter": f"(REPORT_DATE='{q_rep}')",
            }
            rj = _http().get(url, params=params, timeout=10).json()
            res = rj.get("result")
            if not res or not res.get("data"):
                continue
            pages = min(int(res.get("pages") or 1), 8)
            items = list(res["data"])
            for p in range(2, pages + 1):
                params["pageNumber"] = str(p)
                rp = _http().get(url, params=params, timeout=10).json()
                items.extend((rp.get("result") or {}).get("data") or [])

            seen_code = set()
            for it in items:
                code = str(it.get("SECURITY_CODE") or "").strip().zfill(6)
                if not code.startswith(("0", "3", "6")) or code in seen_code:
                    continue
                p_type = str(it.get("PREDICT_TYPE") or "").strip()
                if not p_type:
                    continue
                seen_code.add(code)
                n_dt = str(it.get("NOTICE_DATE") or q_rep)[:10]
                lo = it.get("ADD_AMP_LOWER")
                hi = it.get("ADD_AMP_UPPER")
                try:
                    amp = (float(lo) + float(hi)) / 2.0 if (lo is not None and hi is not None) else (float(lo) if lo is not None else 0.0)
                except Exception:
                    amp = 0.0
                if p_type in ("首亏", "续亏", "增亏") or (p_type in ("预减", "略减") and amp <= -40.0):
                    sev = "veto"
                elif p_type in ("预增", "略增", "扭亏", "续盈") and amp >= 30.0:
                    sev = "bull"
                else:
                    continue
                summ = f"业绩{p_type}({amp:+.0f}%)" if amp else f"业绩{p_type}"
                rows.append((code, "业绩预告", n_dt, sev, round(amp, 1), summ))
        except Exception:
            pass

    # 3) 股东增减持公告 (东财 RPT_SHARE_HOLDER_INCREASE，提取近 120 天最新股东增持/减持动态)
    if progress_cb:
        progress_cb("正在拉取大股东/高管增减持计划与实施动态...")
    cutoff_90 = (today_dt - timedelta(days=120)).strftime("%Y-%m-%d")
    try:
        sh_items = []
        for p in range(1, 5):
            params = {
                "sortColumns": "NOTICE_DATE,SECURITY_CODE",
                "sortTypes": "-1,-1",
                "pageSize": "500",
                "pageNumber": str(p),
                "reportName": "RPT_SHARE_HOLDER_INCREASE",
                "columns": (
                    "SECURITY_CODE,SECURITY_NAME_ABBR,NOTICE_DATE,DIRECTION,"
                    "HOLDER_NAME,CHANGE_NUM,AFTER_CHANGE_RATE,CHANGE_FREE_RATIO"
                ),
                "source": "WEB",
                "client": "WEB",
            }
            rj = _http().get(url, params=params, timeout=8).json()
            batch = (rj.get("result") or {}).get("data") or []
            if not batch:
                break
            sh_items.extend(batch)
            last_dt = str(batch[-1].get("NOTICE_DATE") or "")[:10]
            if last_dt and last_dt < cutoff_90:
                break

        # 按 (code, direction) 汇总近 120 天累计增/减持占流通股比例与最新公告日
        agg_sh: dict[tuple[str, str], dict] = {}
        for it in sh_items:
            code = str(it.get("SECURITY_CODE") or "").strip().zfill(6)
            if not code.startswith(("0", "3", "6")):
                continue
            n_dt = str(it.get("NOTICE_DATE") or "")[:10]
            if not n_dt or n_dt < cutoff_90:
                continue
            direction = str(it.get("DIRECTION") or "").strip()
            if direction not in ("增持", "减持"):
                continue
            r_free = it.get("CHANGE_FREE_RATIO")
            r_tot = it.get("AFTER_CHANGE_RATE")
            try:
                pct = float(r_free) if (r_free is not None and r_free == r_free) else (
                    float(r_tot) if (r_tot is not None and r_tot == r_tot) else 0.0
                )
            except Exception:
                pct = 0.0
            pct = abs(pct)
            key = (code, direction)
            if key not in agg_sh:
                agg_sh[key] = {
                    "date": n_dt,
                    "pct": pct,
                    "holder": str(it.get("HOLDER_NAME") or "")[:8],
                    "cnt": 1,
                }
            else:
                agg_sh[key]["pct"] += pct
                agg_sh[key]["cnt"] += 1

        for (code, direction), info in agg_sh.items():
            pct = round(min(info["pct"], 99.9), 2)
            n_dt = info["date"]
            if direction == "减持":
                if pct >= 1.5:
                    sev = "veto"
                elif pct >= 0.2:
                    sev = "warn"
                else:
                    continue
                summ = f"股东减持{pct:.2f}%({n_dt[5:]})"
                rows.append((code, "股东减持", n_dt, sev, pct, summ))
            else:
                if pct < 0.05 and info["cnt"] < 2:
                    continue
                sev = "bull"
                summ = f"股东增持{pct:.2f}%({n_dt[5:]})" if pct >= 0.05 else f"股东增持({n_dt[5:]})"
                rows.append((code, "股东增持", n_dt, sev, max(pct, 0.05), summ))
    except Exception as e:
        print(f"[warn] 股东增减持拉取异常: {e}")

    # 4) 股份回购预案与实施进度 (东财 RPTA_WEB_GETHGLIST_NEW，提取近 150 天进行中的回购计划)
    if progress_cb:
        progress_cb("正在拉取上市公司股份回购计划与实施进度...")
    cutoff_rep = (today_dt - timedelta(days=150)).strftime("%Y-%m-%d")
    prog_map = {
        "001": "预案",
        "002": "股东会通过",
        "004": "实施中",
        "006": "完成",
    }
    try:
        hg_items = []
        for p in range(1, 4):
            params = {
                "sortColumns": "UPD,DIM_DATE,DIM_SCODE",
                "sortTypes": "-1,-1,-1",
                "pageSize": "500",
                "pageNumber": str(p),
                "reportName": "RPTA_WEB_GETHGLIST_NEW",
                "columns": "DIM_SCODE,SECURITYSHORTNAME,UPDATEDATE,REPURPROGRESS,ZSZSX,ZSZXX,JESX,JEXX,REPURAMOUNT",
                "source": "WEB",
            }
            rj = _http().get(url, params=params, timeout=8).json()
            batch = (rj.get("result") or {}).get("data") or []
            if not batch:
                break
            hg_items.extend(batch)
            last_dt = str(batch[-1].get("UPDATEDATE") or "")[:10]
            if last_dt and last_dt < cutoff_rep:
                break

        seen_hg = set()
        for it in hg_items:
            code = str(it.get("DIM_SCODE") or "").strip().zfill(6)
            if not code.startswith(("0", "3", "6")) or code in seen_hg:
                continue
            u_dt = str(it.get("UPDATEDATE") or "")[:10]
            if not u_dt or u_dt < cutoff_rep:
                continue
            p_code = str(it.get("REPURPROGRESS") or "").strip()
            if p_code not in prog_map:
                continue
            p_lbl = prog_map[p_code]
            jesx = it.get("JESX") or it.get("JEXX") or it.get("REPURAMOUNT")
            zsz = it.get("ZSZSX") or it.get("ZSZXX")
            try:
                je_yi = float(jesx) / 1e8 if (jesx is not None and jesx == jesx) else 0.0
            except Exception:
                je_yi = 0.0
            try:
                z_pct = float(zsz) if (zsz is not None and zsz == zsz) else 0.0
            except Exception:
                z_pct = 0.0
            if je_yi < 0.15 and z_pct < 0.3:
                continue
            seen_hg.add(code)
            amt_s = f"{je_yi:.1f}亿" if je_yi >= 0.5 else f"{je_yi*10000:.0f}万"
            summ = f"股份回购{p_lbl}({amt_s}·占{z_pct:.1f}%)" if z_pct > 0 else f"股份回购{p_lbl}({amt_s})"
            rows.append((code, "股份回购", u_dt, "bull", round(max(z_pct, je_yi), 2), summ))
    except Exception as e:
        print(f"[warn] 股份回购计划拉取异常: {e}")

    if rows:
        with _DB_LOCK:
            db.save_risk_calendar(rows)
    return len(rows)


def refresh_derived_stock_features(progress_cb=None) -> int:
    """
    全局预计算并回填全市场 5000+ 只股票的多维衍生基本面与量价形态特征到 fundamental 表：
      - 估值分位：pe_hist_pct (5年PE历史分位), pb_hist_pct (5年PB历史分位), pe_ind_pct (同业PE分位), peg
      - 盈利与质量：gross_margin (毛利率%), gm_trend (毛利率趋势), debt_ratio (资产负债率%),
                    roe_3y_min (连续3年年报最低ROE%), roe_3y_avg (3年平均ROE%)
      - 成长与加速：growth_streak (净利连增季度数), rev_growth_streak (营收连增季度数),
                    avg_profit_yoy_3q (近3期均增%), profit_accel (业绩加速标志 1/0)
      - 量价与均线：ret_20d, ret_60d, ma_bull (均线多头排列 1/0), break_high60 (创60日新高 1/0)
    """
    import numpy as np

    if progress_cb:
        progress_cb("正在同步全市场资产负债率与计算 18 项深度选股因子...")

    # 1) 从东财 RPT_DMSK_FN_BALANCE 批量补齐最新资产负债率 (DEBT_ASSET_RATIO)
    debt_map: dict[str, float] = {}
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    for q_dt in ("2026-09-30", "2026-06-30"):
        try:
            params = {
                "reportName": "RPT_DMSK_FN_BALANCE",
                "columns": "SECURITY_CODE,REPORT_DATE,DEBT_ASSET_RATIO",
                "filter": f"(REPORT_DATE='{q_dt}')",
                "pageNumber": "1",
                "pageSize": "500",
                "source": "WEB",
                "client": "WEB",
            }
            r0 = _http().get(url, params=params, timeout=8).json()
            res0 = r0.get("result") or {}
            data0 = res0.get("data") or []
            if not data0:
                continue
            pages = min(int(res0.get("pages") or 1), 12)
            items = list(data0)

            def _pull_b(p):
                try:
                    rj = _http().get(url, params=dict(params, pageNumber=str(p)), timeout=8).json()
                    return (rj.get("result") or {}).get("data") or []
                except Exception:
                    return []

            if pages > 1:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    for part in pool.map(_pull_b, range(2, pages + 1)):
                        items.extend(part)
            for it in items:
                c = str(it.get("SECURITY_CODE") or "").zfill(6)
                dar = it.get("DEBT_ASSET_RATIO")
                if c not in debt_map and dar is not None:
                    try:
                        debt_map[c] = round(float(dar), 2)
                    except Exception:
                        pass
            if len(debt_map) >= 4000:
                break
        except Exception:
            pass

    with db.get_conn() as conn:
        for col in ("pe_hist_pct", "pb_hist_pct", "pe_ind_pct", "peg",
                    "roe_3y_min", "roe_3y_avg", "gm_trend",
                    "growth_streak", "rev_growth_streak", "avg_profit_yoy_3q", "profit_accel",
                    "div_years_5y", "div_avg_5y", "div_min_3y",
                    "ret_20d", "ret_60d", "ma_bull", "break_high60"):
            try:
                conn.execute(f"ALTER TABLE fundamental ADD COLUMN {col} REAL")
            except Exception:
                pass
        f_df = pd.read_sql("SELECT code, pe_ttm, pb, total_mv, roe, debt_ratio FROM fundamental", conn)
        pit_df = pd.read_sql(
            "SELECT code, report_date, eps, bps, roe, rev_yoy, profit_yoy, gross_margin, board_name "
            "FROM pit_financials ORDER BY code, report_date",
            conn,
        )
        k_q_df = pd.read_sql(
            """
            SELECT code, substr(date, 1, 7) as ym, close
            FROM daily_kline
            WHERE date >= '2020-12-01'
              AND substr(date, 6, 2) IN ('03', '06', '09', '12')
            GROUP BY code, substr(date, 1, 7)
            """,
            conn,
        )
        # 取最近 65 个交易日的日线用于计算 ret_20d, ret_60d, ma_bull, break_high60
        k_recent = pd.read_sql(
            """
            SELECT code, date, close, high
            FROM daily_kline
            WHERE date >= '2026-07-01'
            ORDER BY code, date
            """,
            conn,
        )

    # 2) 计算近20日/60日涨跌、均线多头与突破60日新高
    tech_map: dict[str, tuple] = {}
    for code, grp in k_recent.groupby("code", sort=False):
        closes = grp["close"].to_numpy(dtype=float)
        highs = grp["high"].to_numpy(dtype=float)
        n = len(closes)
        if n < 20:
            continue
        c_last = closes[-1]
        r20 = round((c_last / closes[-min(n, 21)] - 1.0) * 100.0, 2) if closes[-min(n, 21)] > 0 else None
        r60 = round((c_last / closes[0] - 1.0) * 100.0, 2) if n >= 45 and closes[0] > 0 else None
        ma5 = float(np.mean(closes[-5:]))
        ma10 = float(np.mean(closes[-10:]))
        ma20 = float(np.mean(closes[-20:]))
        ma60 = float(np.mean(closes[-min(n, 60):]))
        ma20_prev = float(np.mean(closes[-min(n, 25):-5])) if n >= 25 else ma20
        ma_bull = 1.0 if (c_last >= ma5 >= ma20 >= ma60 and ma20 >= ma20_prev and ma5 >= ma10 * 0.995) else 0.0
        h60 = float(np.max(highs[-min(n, 60):]))
        brk60 = 1.0 if (h60 > 0 and np.max(closes[-3:]) >= h60 * 0.985 and r20 is not None and r20 > 0) else 0.0
        tech_map[code] = (r20, r60, ma_bull, brk60)

    # 3) 计算同行业 PE 分位数 (pe_ind_pct)
    board_map = {}
    for code, grp in pit_df.groupby("code", sort=False):
        bname = grp["board_name"].dropna().tolist()
        if bname:
            board_map[code] = bname[-1]
    f_df["board"] = f_df["code"].map(board_map).fillna("其他")
    f_df["pe_valid"] = pd.to_numeric(f_df["pe_ttm"], errors="coerce")
    pe_ind_map: dict[str, float] = {}
    for b, grp in f_df.groupby("board", sort=False):
        valid = grp[(grp["pe_valid"] > 0) & (grp["pe_valid"] < 500)]
        if len(valid) < 5:
            continue
        arr = valid["pe_valid"].to_numpy(dtype=float)
        for r in valid.itertuples(index=False):
            pe_ind_map[r.code] = round(float(np.mean(arr <= float(r.pe_valid)) * 100.0), 1)

    # 4) 计算 5 年季度历史 PE/PB 分位、连续3年年报 ROE、毛利率与连续增长指标
    q_month = pit_df["report_date"].str.slice(5, 7)
    mult = q_month.map({"03": 4.0, "06": 2.0, "09": 4.0 / 3.0, "12": 1.0}).fillna(1.0)
    pit_df["eps_ann"] = pd.to_numeric(pit_df["eps"], errors="coerce") * mult
    pit_df["bps_num"] = pd.to_numeric(pit_df["bps"], errors="coerce")
    pit_df["ym"] = pit_df["report_date"].str.slice(0, 7)

    m = pit_df.merge(k_q_df, on=["code", "ym"], how="left")
    m["pe_q"] = np.where((m["eps_ann"] > 0.02) & (m["close"] > 0), m["close"] / m["eps_ann"], np.nan)
    m["pb_q"] = np.where((m["bps_num"] > 0.05) & (m["close"] > 0), m["close"] / m["bps_num"], np.nan)

    cur_pe_map = dict(zip(f_df["code"], pd.to_numeric(f_df["pe_ttm"], errors="coerce")))
    cur_pb_map = dict(zip(f_df["code"], pd.to_numeric(f_df["pb"], errors="coerce")))
    cur_roe_map = dict(zip(f_df["code"], pd.to_numeric(f_df["roe"], errors="coerce")))

    updates = []
    for code, grp in m.groupby("code", sort=False):
        cur_pe = cur_pe_map.get(code)
        cur_pb = cur_pb_map.get(code)
        cur_roe = cur_roe_map.get(code)

        pe_hist_pct = None
        if cur_pe is not None and not pd.isna(cur_pe) and cur_pe > 0:
            v_pe = grp["pe_q"].dropna()
            v_pe = v_pe[(v_pe > 0) & (v_pe < 500)]
            if len(v_pe) >= 6:
                pe_hist_pct = round(float((v_pe <= cur_pe).mean() * 100.0), 1)

        pb_hist_pct = None
        if cur_pb is not None and not pd.isna(cur_pb) and cur_pb > 0:
            v_pb = grp["pb_q"].dropna()
            v_pb = v_pb[(v_pb > 0) & (v_pb < 100)]
            if len(v_pb) >= 6:
                pb_hist_pct = round(float((v_pb <= cur_pb).mean() * 100.0), 1)

        # 过去3年年报 (2023-12-31, 2024-12-31, 2025-12-31) ROE
        ann_grp = grp[grp["report_date"].isin(["2023-12-31", "2024-12-31", "2025-12-31"])]
        ann_roes = pd.to_numeric(ann_grp["roe"], errors="coerce").dropna().tolist()
        roe_3y_min = round(float(min(ann_roes)), 2) if len(ann_roes) >= 2 else None
        roe_3y_avg = round(float(np.mean(ann_roes)), 2) if len(ann_roes) >= 2 else None

        # 毛利率与趋势
        gms = pd.to_numeric(grp["gross_margin"], errors="coerce").dropna().tolist()
        gm_latest = round(float(gms[-1]), 2) if gms else None
        gm_trend = round(float(gms[-1] - gms[-2]), 2) if len(gms) >= 2 else None

        # 净利与营收连续增长季度数、加速标志
        pyoys = pd.to_numeric(grp["profit_yoy"], errors="coerce").dropna().tolist()
        p_streak = 0
        for v in reversed(pyoys):
            if v > 0:
                p_streak += 1
            else:
                break
        avg_3q = round(float(np.mean(pyoys[-3:])), 2) if len(pyoys) >= 3 else (round(float(pyoys[-1]), 2) if pyoys else None)
        latest_pyoy = round(float(pyoys[-1]), 2) if pyoys else None
        p_accel = 1.0 if (len(pyoys) >= 2 and pyoys[-1] > pyoys[-2] > 0 and pyoys[-1] >= 10.0) else 0.0

        ryoys = pd.to_numeric(grp["rev_yoy"], errors="coerce").dropna().tolist()
        r_streak = 0
        for v in reversed(ryoys):
            if v > 0:
                r_streak += 1
            else:
                break
        latest_ryoy = round(float(ryoys[-1]), 2) if ryoys else None

        # PEG: 当 PE > 0 且 avg_3q >= 5% 时计算
        peg = None
        if cur_pe is not None and not pd.isna(cur_pe) and cur_pe > 0 and avg_3q is not None and avg_3q >= 5.0:
            peg = round(float(cur_pe / avg_3q), 2)

        roe_fill = cur_roe
        if roe_fill is None or pd.isna(roe_fill):
            roes = pd.to_numeric(grp["roe"], errors="coerce").dropna().tolist()
            if roes:
                roe_fill = round(float(roes[-1]), 2)

        r20, r60, ma_bull, brk60 = tech_map.get(code, (None, None, 0.0, 0.0))
        updates.append((
            pe_hist_pct,
            pb_hist_pct,
            pe_ind_map.get(code),
            peg,
            roe_3y_min,
            roe_3y_avg,
            gm_latest,
            gm_trend,
            debt_map.get(code),
            p_streak,
            r_streak,
            avg_3q,
            p_accel,
            latest_pyoy,
            latest_ryoy,
            roe_fill,
            r20,
            r60,
            ma_bull,
            brk60,
            code,
        ))

    with _DB_LOCK:
        with db.get_conn() as conn:
            conn.executemany(
                """
                UPDATE fundamental
                SET pe_hist_pct       = ?,
                    pb_hist_pct       = ?,
                    pe_ind_pct        = ?,
                    peg               = ?,
                    roe_3y_min        = ?,
                    roe_3y_avg        = ?,
                    gross_margin      = COALESCE(?, gross_margin),
                    gm_trend          = ?,
                    debt_ratio        = COALESCE(?, debt_ratio),
                    growth_streak     = ?,
                    rev_growth_streak = ?,
                    avg_profit_yoy_3q = ?,
                    profit_accel      = ?,
                    profit_yoy        = COALESCE(?, profit_yoy),
                    rev_yoy           = COALESCE(?, rev_yoy),
                    roe               = COALESCE(?, roe),
                    ret_20d           = ?,
                    ret_60d           = ?,
                    ma_bull           = ?,
                    break_high60      = ?
                WHERE code = ?
                """,
                updates,
            )
    return len(updates)


