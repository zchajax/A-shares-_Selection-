"""
数据层 - SQLite 数据库封装
负责本地存储 A 股列表和日线行情数据。
所有数据都缓存在本地,避免每次都去网络拉取。
"""
import os
import sqlite3
from contextlib import contextmanager

import pandas as pd

# 数据库文件放在项目根目录的 data_cache 下
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(_BASE_DIR, "data_cache", "stock.db")


@contextmanager
def get_conn():
    """获取数据库连接的上下文管理器,自动提交/关闭。"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    # 并发拉取时多个线程可能同时读写,让 SQLite 最多等 30s 而不是立刻报 locked
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """初始化表结构。第一次运行时调用。"""
    with get_conn() as conn:
        cur = conn.cursor()
        # 股票基础列表
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_list (
                code   TEXT PRIMARY KEY,   -- 股票代码,如 000001
                name   TEXT,               -- 股票名称
                industry TEXT,             -- 所属行业(巨潮行业分类,可为空)
                updated_at TEXT            -- 最后更新时间
            )
            """
        )
        # 旧库兼容:若 stock_list 已存在但没有 industry 列,补上
        info = cur.execute("PRAGMA table_info(stock_list)").fetchall()
        cols = [r[1] for r in info]
        if "industry" not in cols:
            cur.execute("ALTER TABLE stock_list ADD COLUMN industry TEXT")
        # 旧库兼容:早期版本的 stock_list 可能由 pandas to_sql 建表,
        # 没有 code 主键 -> INSERT ... ON CONFLICT(code) 会报
        # "ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE constraint"。
        # 检测到无主键时,原地迁移重建(保留数据含 industry)。
        has_pk = any(r[5] for r in info)  # PRAGMA 第 6 列(index 5)=pk 标志
        if info and not has_pk:
            cur.execute("ALTER TABLE stock_list RENAME TO stock_list_legacy")
            cur.execute(
                """
                CREATE TABLE stock_list (
                    code   TEXT PRIMARY KEY,
                    name   TEXT,
                    industry TEXT,
                    updated_at TEXT
                )
                """
            )
            # 去重(按 code 保留一行)迁移旧数据
            cur.execute(
                "INSERT OR IGNORE INTO stock_list (code, name, industry, updated_at) "
                "SELECT code, name, industry, updated_at FROM stock_list_legacy "
                "WHERE code IS NOT NULL"
            )
            cur.execute("DROP TABLE stock_list_legacy")
        # 日线行情(前复权)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_kline (
                code   TEXT,
                date   TEXT,
                open   REAL,
                high   REAL,
                low    REAL,
                close  REAL,
                volume REAL,
                amount REAL,
                PRIMARY KEY (code, date)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_kline_code ON daily_kline(code)"
        )
        # 大盘指数日线(如上证000001/沪深300),用于"大盘趋势过滤"
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS index_kline (
                code   TEXT,               -- 指数代码,如 sh000001
                date   TEXT,
                open   REAL,
                high   REAL,
                low    REAL,
                close  REAL,
                volume REAL,
                amount REAL,
                PRIMARY KEY (code, date)
            )
            """
        )
        # 自选股 / 持仓:选出的票存下来,记录买入价与加入日期,显示浮动盈亏
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS watchlist (
                code       TEXT PRIMARY KEY,
                name       TEXT,
                add_date   TEXT,           -- 加入日期
                buy_price  REAL,           -- 买入价(0=仅观察,未持仓)
                note       TEXT            -- 备注(来源策略等)
            )
            """
        )
        # 基本面 / 估值:每只股票一行,长期缓存(季报/估值变动不频繁,增量补齐)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fundamental (
                code       TEXT PRIMARY KEY,
                pe_ttm     REAL,           -- 市盈率(TTM)
                pb         REAL,           -- 市净率
                ps_ttm     REAL,           -- 市销率(TTM)
                total_mv   REAL,           -- 总市值(亿元)
                roe        REAL,           -- 净资产收益率(%)
                gross_margin REAL,         -- 销售毛利率(%)
                net_margin   REAL,         -- 销售净利率(%)
                rev_yoy      REAL,         -- 主营业务收入增长率(%)
                profit_yoy   REAL,         -- 净利润增长率(%)
                debt_ratio   REAL,         -- 资产负债率(%)
                dividend_ratio REAL,       -- 股息发放率(%)
                report_date  TEXT,         -- 财务指标所属报告期(如 2025-03-31)
                updated_at TEXT
            )
            """
        )
        # 旧库兼容:早期 fundamental 只有估值5字段,缺成长/质量列时补上
        _fund_cols = [r[1] for r in
                      cur.execute("PRAGMA table_info(fundamental)").fetchall()]
        for _c, _t in (("gross_margin", "REAL"), ("net_margin", "REAL"),
                       ("rev_yoy", "REAL"), ("profit_yoy", "REAL"),
                       ("debt_ratio", "REAL"), ("dividend_ratio", "REAL"),
                       ("report_date", "TEXT")):
            if _c not in _fund_cols:
                cur.execute(f"ALTER TABLE fundamental ADD COLUMN {_c} {_t}")
        # 价格预警:对个股设置涨跌幅/价位阈值,盘中实时行情触发时高亮提示
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                code       TEXT PRIMARY KEY,
                name       TEXT,
                price_low  REAL,           -- 价格下限(<=触发),0/NULL=不启用
                price_high REAL,           -- 价格上限(>=触发)
                chg_low    REAL,           -- 涨跌幅下限%(<=触发,如 -5)
                chg_high   REAL,           -- 涨跌幅上限%(>=触发,如 5)
                note       TEXT,
                created_at TEXT
            )
            """
        )
        # ETF 注册表:标记哪些 code 是 ETF(用于 ETF 专属榜单与"是否ETF"判定)。
        # ETF 的日线数据复用 daily_kline 表(格式与股票完全一致),因此策略扫描/
        # 回测/K线/参数寻优全部自动支持 ETF,无需改动。ETF 名称也同步写入
        # stock_list,让 scanner 的 namemap 能显示 ETF 名称。
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS etf_list (
                code       TEXT PRIMARY KEY,   -- 6位代码,如 510300
                name       TEXT,               -- ETF 名称,如 沪深300ETF
                updated_at TEXT
            )
            """
        )
        # AI 点评历史存档:每次成功点评落一条,供历史回看("这只票上周AI怎么说")。
        # 同一 code 保留多条(按时间),trade_date 标注点评所依据的交易日。
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_commentary (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                code       TEXT,
                name       TEXT,
                trade_date TEXT,               -- 点评所依据的交易日
                rating     TEXT,               -- 偏多/中性/偏空
                risk       TEXT,               -- 高/中/低
                text       TEXT,               -- 点评全文
                created_at TEXT
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ai_comm_code "
            "ON ai_commentary(code, created_at)"
        )


def save_stock_list(df: pd.DataFrame):
    """
    保存股票列表。df 需含 code, name 两列。
    为避免覆盖已抓取的行业信息,采用 UPSERT:更新 code/name/updated_at,
    但不动已有的 industry 列(行业单独由 update_industry 维护)。
    """
    if df is None or df.empty:
        return
    df = df[["code", "name"]].copy()
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        rows = [(r.code, r.name, now) for r in df.itertuples(index=False)]
        conn.executemany(
            "INSERT INTO stock_list (code, name, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(code) DO UPDATE SET name=excluded.name, "
            "updated_at=excluded.updated_at",
            rows,
        )


def load_stock_list() -> pd.DataFrame:
    """读取本地股票列表(含行业列)。"""
    with get_conn() as conn:
        try:
            return pd.read_sql(
                "SELECT code, name, industry FROM stock_list", conn)
        except Exception:
            return pd.DataFrame(columns=["code", "name", "industry"])


# ==================== 行业分类 ====================
def save_industry_map(mapping: dict):
    """
    批量写入 code->行业 映射(巨潮行业分类)。
    只更新已在 stock_list 中的股票的 industry 列。mapping: {code: industry}。
    """
    if not mapping:
        return
    with get_conn() as conn:
        conn.executemany(
            "UPDATE stock_list SET industry = ? WHERE code = ?",
            [(v, k) for k, v in mapping.items()],
        )


def load_industry_map() -> dict:
    """返回 {code: industry},仅含已分类(industry 非空)的股票。"""
    with get_conn() as conn:
        try:
            rows = conn.execute(
                "SELECT code, industry FROM stock_list "
                "WHERE industry IS NOT NULL AND industry <> ''"
            ).fetchall()
            return {r[0]: r[1] for r in rows}
        except Exception:
            return {}


def codes_without_industry(codes: list = None) -> list:
    """返回还没有行业信息的股票代码(用于增量补齐)。codes=None 时看全表。"""
    with get_conn() as conn:
        try:
            rows = conn.execute(
                "SELECT code FROM stock_list "
                "WHERE industry IS NULL OR industry = ''"
            ).fetchall()
            miss = {r[0] for r in rows}
            if codes is None:
                return list(miss)
            return [c for c in codes if c in miss]
        except Exception:
            return list(codes or [])


def save_kline(code: str, df: pd.DataFrame):
    """保存单只股票的日线数据。df 列: date,open,high,low,close,volume,amount"""
    if df is None or df.empty:
        return
    df = df.copy()
    df["code"] = code
    cols = ["code", "date", "open", "high", "low", "close", "volume", "amount"]
    df = df[cols]
    with get_conn() as conn:
        # 先删除该股票旧数据再插入,保证幂等
        conn.execute("DELETE FROM daily_kline WHERE code = ?", (code,))
        df.to_sql("daily_kline", conn, if_exists="append", index=False)


def load_kline(code: str) -> pd.DataFrame:
    """读取单只股票的日线数据,按日期升序。"""
    with get_conn() as conn:
        return pd.read_sql(
            "SELECT date,open,high,low,close,volume,amount "
            "FROM daily_kline WHERE code = ? ORDER BY date ASC",
            conn,
            params=(code,),
        )


def list_cached_codes() -> list:
    """返回本地已缓存日线的股票代码列表。"""
    with get_conn() as conn:
        try:
            rows = conn.execute(
                "SELECT DISTINCT code FROM daily_kline"
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []


def get_last_date(code: str):
    """返回某只股票本地已缓存的最新日期(YYYY-MM-DD),没有则返回 None。"""
    with get_conn() as conn:
        try:
            row = conn.execute(
                "SELECT MAX(date) FROM daily_kline WHERE code = ?", (code,)
            ).fetchone()
            return row[0] if row and row[0] else None
        except Exception:
            return None


def append_kline(code: str, df: pd.DataFrame):
    """增量追加日线(不删旧数据),用 INSERT OR IGNORE 避免主键冲突。"""
    if df is None or df.empty:
        return
    df = df.copy()
    df["code"] = code
    cols = ["code", "date", "open", "high", "low", "close", "volume", "amount"]
    df = df[cols]
    with get_conn() as conn:
        rows = [tuple(r) for r in df.itertuples(index=False)]
        conn.executemany(
            "INSERT OR IGNORE INTO daily_kline "
            "(code,date,open,high,low,close,volume,amount) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )


# ==================== 大盘指数 ====================
def save_index_kline(code: str, df: pd.DataFrame):
    """保存指数日线(先删后插,幂等)。df 列: date,open,high,low,close,volume,amount"""
    if df is None or df.empty:
        return
    df = df.copy()
    df["code"] = code
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c not in df.columns:
            df[c] = 0.0
    df = df[["code", "date", "open", "high", "low", "close", "volume", "amount"]]
    with get_conn() as conn:
        conn.execute("DELETE FROM index_kline WHERE code = ?", (code,))
        df.to_sql("index_kline", conn, if_exists="append", index=False)


def load_index_kline(code: str) -> pd.DataFrame:
    """读取指数日线,按日期升序。"""
    with get_conn() as conn:
        try:
            return pd.read_sql(
                "SELECT date,open,high,low,close,volume,amount "
                "FROM index_kline WHERE code = ? ORDER BY date ASC",
                conn, params=(code,),
            )
        except Exception:
            return pd.DataFrame()


def get_index_last_date(code: str):
    """指数本地已缓存的最新日期,无则 None。"""
    with get_conn() as conn:
        try:
            row = conn.execute(
                "SELECT MAX(date) FROM index_kline WHERE code = ?", (code,)
            ).fetchone()
            return row[0] if row and row[0] else None
        except Exception:
            return None


# ==================== 自选股 / 持仓 ====================
def add_watch(code: str, name: str = "", buy_price: float = 0.0, note: str = ""):
    """加入自选(已存在则更新)。buy_price=0 表示仅观察。"""
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO watchlist (code,name,add_date,buy_price,note) "
            "VALUES (?,?,?,?,?)",
            (code, name, pd.Timestamp.now().strftime("%Y-%m-%d"),
             float(buy_price or 0.0), note),
        )


def remove_watch(code: str):
    """移除自选。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM watchlist WHERE code = ?", (code,))


def load_watchlist() -> pd.DataFrame:
    """读取自选列表。"""
    with get_conn() as conn:
        try:
            return pd.read_sql(
                "SELECT code,name,add_date,buy_price,note FROM watchlist "
                "ORDER BY add_date DESC", conn)
        except Exception:
            return pd.DataFrame(columns=["code", "name", "add_date", "buy_price", "note"])


# ==================== 基本面 / 估值 ====================
def save_fundamental(rows: dict):
    """
    批量写入基本面数据。rows: {code: {pe_ttm, pb, ps_ttm, total_mv, roe,
      gross_margin, net_margin, rev_yoy, profit_yoy, debt_ratio,
      dividend_ratio, report_date}}。
    UPSERT:已存在则更新非空字段(None 不覆盖旧值)。
    """
    if not rows:
        return
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        for code, d in rows.items():
            conn.execute(
                """
                INSERT INTO fundamental
                    (code, pe_ttm, pb, ps_ttm, total_mv, roe,
                     gross_margin, net_margin, rev_yoy, profit_yoy,
                     debt_ratio, dividend_ratio, report_date, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(code) DO UPDATE SET
                    pe_ttm   = COALESCE(excluded.pe_ttm,   fundamental.pe_ttm),
                    pb       = COALESCE(excluded.pb,       fundamental.pb),
                    ps_ttm   = COALESCE(excluded.ps_ttm,   fundamental.ps_ttm),
                    total_mv = COALESCE(excluded.total_mv, fundamental.total_mv),
                    roe      = COALESCE(excluded.roe,       fundamental.roe),
                    gross_margin   = COALESCE(excluded.gross_margin,   fundamental.gross_margin),
                    net_margin     = COALESCE(excluded.net_margin,     fundamental.net_margin),
                    rev_yoy        = COALESCE(excluded.rev_yoy,        fundamental.rev_yoy),
                    profit_yoy     = COALESCE(excluded.profit_yoy,     fundamental.profit_yoy),
                    debt_ratio     = COALESCE(excluded.debt_ratio,     fundamental.debt_ratio),
                    dividend_ratio = COALESCE(excluded.dividend_ratio, fundamental.dividend_ratio),
                    report_date    = COALESCE(excluded.report_date,    fundamental.report_date),
                    updated_at = excluded.updated_at
                """,
                (code, d.get("pe_ttm"), d.get("pb"), d.get("ps_ttm"),
                 d.get("total_mv"), d.get("roe"),
                 d.get("gross_margin"), d.get("net_margin"),
                 d.get("rev_yoy"), d.get("profit_yoy"),
                 d.get("debt_ratio"), d.get("dividend_ratio"),
                 d.get("report_date"), now),
            )


def load_fundamental() -> pd.DataFrame:
    """读取全部基本面数据(含成长/质量字段)。"""
    _cols = ["code", "pe_ttm", "pb", "ps_ttm", "total_mv", "roe",
             "gross_margin", "net_margin", "rev_yoy", "profit_yoy",
             "debt_ratio", "dividend_ratio", "report_date", "updated_at"]
    with get_conn() as conn:
        try:
            return pd.read_sql(
                "SELECT " + ",".join(_cols) + " FROM fundamental", conn)
        except Exception:
            return pd.DataFrame(columns=_cols)


def load_fundamental_map() -> dict:
    """返回 {code: {pe_ttm, pb, ps_ttm, total_mv, roe}},供选股/展示快速查。"""
    df = load_fundamental()
    if df is None or df.empty:
        return {}
    out = {}
    for r in df.itertuples(index=False):
        out[r.code] = {
            "pe_ttm": r.pe_ttm, "pb": r.pb, "ps_ttm": r.ps_ttm,
            "total_mv": r.total_mv, "roe": r.roe,
        }
    return out


def industry_valuation_percentile(code: str, metric: str = "pe_ttm") -> dict:
    """计算某股某估值指标在其所属行业内的分位(0-100,越高越贵)。

    仅用本地已缓存的 fundamental + 行业映射;同行业有效样本 < 5 时返回 None
    (样本太少分位无意义)。返回 {percentile, industry, peers, value} 或 None。
    """
    fmap = load_fundamental_map()
    if code not in fmap or fmap[code].get(metric) is None:
        return None
    try:
        val = float(fmap[code][metric])
    except Exception:
        return None
    if val <= 0:  # PE/PB 为负无比较意义
        return None
    try:
        ind_map = load_industry_map()
    except Exception:
        return None
    my_ind = ind_map.get(code)
    if not my_ind:
        return None
    peers = []
    for c, ind in ind_map.items():
        if ind != my_ind or c == code:
            continue
        v = fmap.get(c, {}).get(metric)
        try:
            v = float(v)
        except Exception:
            continue
        if v > 0:
            peers.append(v)
    if len(peers) < 5:  # 同行业有效样本太少,分位无意义
        return None
    below = sum(1 for v in peers if v < val)
    pct = below / len(peers) * 100
    return {"percentile": pct, "industry": my_ind,
            "peers": len(peers), "value": val}


def get_fundamental(code: str) -> dict:
    """返回单只股票的基本面(含成长/质量字段);无则 None。"""
    with get_conn() as conn:
        try:
            r = conn.execute(
                "SELECT pe_ttm,pb,ps_ttm,total_mv,roe,gross_margin,net_margin,"
                "rev_yoy,profit_yoy,debt_ratio,dividend_ratio,report_date,updated_at "
                "FROM fundamental WHERE code=?", (code,)).fetchone()
        except Exception:
            return None
    if not r:
        return None
    return {"pe_ttm": r[0], "pb": r[1], "ps_ttm": r[2],
            "total_mv": r[3], "roe": r[4], "gross_margin": r[5],
            "net_margin": r[6], "rev_yoy": r[7], "profit_yoy": r[8],
            "debt_ratio": r[9], "dividend_ratio": r[10],
            "report_date": r[11], "updated_at": r[12]}


def codes_without_fundamental(codes: list = None) -> list:
    """返回还没有基本面数据的股票代码(用于增量补齐)。"""
    with get_conn() as conn:
        try:
            rows = conn.execute("SELECT code FROM fundamental").fetchall()
            have = {r[0] for r in rows}
        except Exception:
            have = set()
    if codes is None:
        # 没给范围就看 stock_list 全部
        codes = [r for r in load_stock_list()["code"].tolist()]
    return [c for c in codes if c not in have]


# ==================== 价格预警 ====================
def save_alert(code: str, name: str = "", price_low=None, price_high=None,
               chg_low=None, chg_high=None, note: str = ""):
    """新增/更新一条价格预警(按 code 主键 UPSERT)。阈值 None=不启用该项。"""
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO alerts "
            "(code,name,price_low,price_high,chg_low,chg_high,note,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (code, name, price_low, price_high, chg_low, chg_high, note,
             pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")),
        )


def remove_alert(code: str):
    """删除某只股票的预警。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM alerts WHERE code = ?", (code,))


def load_alerts() -> pd.DataFrame:
    """读取全部预警。"""
    with get_conn() as conn:
        try:
            return pd.read_sql(
                "SELECT code,name,price_low,price_high,chg_low,chg_high,note "
                "FROM alerts ORDER BY created_at DESC", conn)
        except Exception:
            return pd.DataFrame(columns=["code", "name", "price_low", "price_high",
                                         "chg_low", "chg_high", "note"])


# ==================== ETF ====================
def save_etf_list(df: pd.DataFrame):
    """
    保存/更新 ETF 注册表。df 需含 code, name。
    同时把 ETF 名称写入 stock_list(UPSERT),让 scanner 的 namemap
    能显示 ETF 名称、ETF 得以参与统一的策略扫描。
    """
    if df is None or df.empty:
        return
    df = df[["code", "name"]].copy()
    df["code"] = df["code"].astype(str)
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [(r.code, r.name, now) for r in df.itertuples(index=False)]
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO etf_list (code, name, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(code) DO UPDATE SET name=excluded.name, "
            "updated_at=excluded.updated_at",
            rows,
        )
        # 同步进 stock_list(便于 namemap 统一取名 + 参与扫描)
        conn.executemany(
            "INSERT INTO stock_list (code, name, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(code) DO UPDATE SET name=excluded.name, "
            "updated_at=excluded.updated_at",
            rows,
        )


def load_etf_list() -> pd.DataFrame:
    """读取 ETF 注册表(code, name)。"""
    with get_conn() as conn:
        try:
            return pd.read_sql("SELECT code, name FROM etf_list", conn)
        except Exception:
            return pd.DataFrame(columns=["code", "name"])


def load_etf_codes() -> set:
    """返回所有已登记的 ETF 代码集合(用于'是否ETF'快速判定)。"""
    with get_conn() as conn:
        try:
            rows = conn.execute("SELECT code FROM etf_list").fetchall()
            return {r[0] for r in rows}
        except Exception:
            return set()


def search_local(keyword: str, limit: int = 30) -> list:
    """
    在"本地已拉取(有日线数据)"的股票/ETF 中模糊搜索。
    支持:代码子串匹配 + 名称模糊匹配(拼音简称不做,只按名称文字包含)。
    keyword 为空则返回空。返回 list[dict]:{code,name,is_etf},
    优先级:代码完全相等 > 代码前缀 > 代码子串 > 名称包含,再按代码升序。
    只在有本地日线的标的里搜(用户明确要求搜索范围=已拉取的数据)。
    """
    kw = (keyword or "").strip()
    if not kw:
        return []
    with get_conn() as conn:
        try:
            # 只取本地真正有日线的 code,并 join 名称(stock_list 已含 ETF 名称)
            rows = conn.execute(
                "SELECT DISTINCT k.code, COALESCE(s.name, '') AS name "
                "FROM daily_kline k LEFT JOIN stock_list s ON k.code = s.code"
            ).fetchall()
        except Exception:
            return []
        etf_codes = load_etf_codes()

    kw_low = kw.lower()
    scored = []
    for code, name in rows:
        c = code or ""
        nm = name or ""
        c_low = c.lower()
        nm_low = nm.lower()
        # 匹配打分:越小越靠前
        if c_low == kw_low:
            rank = 0
        elif c_low.startswith(kw_low):
            rank = 1
        elif kw_low in c_low:
            rank = 2
        elif kw_low in nm_low:
            rank = 3
        else:
            continue
        scored.append((rank, c, {"code": c, "name": nm, "is_etf": c in etf_codes}))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [item for _, _, item in scored[:limit]]



def name_of(code: str) -> str:
    """按代码取名称(stock_list 已含 ETF 名称)。查不到返回空串。"""
    if not code:
        return ""
    with get_conn() as conn:
        try:
            row = conn.execute(
                "SELECT name FROM stock_list WHERE code = ?", (code,)
            ).fetchone()
            return (row[0] if row and row[0] else "") or ""
        except Exception:
            return ""


# ==================== AI 点评历史存档 ====================
def save_ai_commentary(code: str, name: str, trade_date: str,
                       rating: str, risk: str, text: str):
    """存一条 AI 点评历史。同 code+trade_date 已存在则更新(当天多次点评只留最新)。"""
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        try:
            # 当天同一只票已有记录则更新,否则插入(避免历史表被同日重复点评灌满)
            row = conn.execute(
                "SELECT id FROM ai_commentary WHERE code=? AND trade_date=?",
                (code, trade_date)).fetchone()
            if row:
                conn.execute(
                    "UPDATE ai_commentary SET name=?,rating=?,risk=?,text=?,"
                    "created_at=? WHERE id=?",
                    (name, rating, risk, text, now, row[0]))
            else:
                conn.execute(
                    "INSERT INTO ai_commentary "
                    "(code,name,trade_date,rating,risk,text,created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (code, name, trade_date, rating, risk, text, now))
        except Exception:  # noqa 存档失败不影响点评本身
            pass


def load_ai_commentary(code: str = None, limit: int = 50) -> pd.DataFrame:
    """读取 AI 点评历史。code=None 取全部(最新在前),否则只取该股。"""
    cols = ["id", "code", "name", "trade_date", "rating", "risk",
            "text", "created_at"]
    with get_conn() as conn:
        try:
            if code:
                return pd.read_sql(
                    "SELECT " + ",".join(cols) + " FROM ai_commentary "
                    "WHERE code=? ORDER BY created_at DESC LIMIT ?",
                    conn, params=(code, limit))
            return pd.read_sql(
                "SELECT " + ",".join(cols) + " FROM ai_commentary "
                "ORDER BY created_at DESC LIMIT ?", conn, params=(limit,))
        except Exception:
            return pd.DataFrame(columns=cols)


def cache_summary() -> dict:
    """返回本地缓存概况:股票数、日线总行数、最新日期。供 UI 启动时显示。"""
    with get_conn() as conn:
        try:
            n_codes = conn.execute(
                "SELECT COUNT(DISTINCT code) FROM daily_kline"
            ).fetchone()[0]
            n_rows = conn.execute("SELECT COUNT(*) FROM daily_kline").fetchone()[0]
            last = conn.execute("SELECT MAX(date) FROM daily_kline").fetchone()[0]
            return {"codes": n_codes or 0, "rows": n_rows or 0, "last_date": last}
        except Exception:
            return {"codes": 0, "rows": 0, "last_date": None}
