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
_WAL_READY = False


@contextmanager
def get_conn():
    """获取数据库连接的上下文管理器,自动提交/关闭。"""
    global _WAL_READY
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    # 并发拉取时多个线程可能同时读写,让 SQLite 最多等 30s 而不是立刻报 locked
    conn.execute("PRAGMA busy_timeout=30000")
    if not _WAL_READY:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            _WAL_READY = True
        except sqlite3.OperationalError:
            pass
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
                note       TEXT,           -- 备注(来源策略等)
                last_price REAL,           -- 最近一次实时/刷新到的现价(持久化,重开软件不丢)
                last_price_time TEXT       -- 该现价的时间戳
            )
            """
        )
        # 旧库兼容:早期 watchlist 无 last_price/last_price_time,缺列时补上
        _wl_cols = [r[1] for r in
                    cur.execute("PRAGMA table_info(watchlist)").fetchall()]
        for _c, _t in (("last_price", "REAL"), ("last_price_time", "TEXT")):
            if _c not in _wl_cols:
                cur.execute(f"ALTER TABLE watchlist ADD COLUMN {_c} {_t}")
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
                       ("report_date", "TEXT"), ("float_mv", "REAL"),
                       ("pe_hist_pct", "REAL"), ("pb_hist_pct", "REAL"),
                       ("pe_ind_pct", "REAL"), ("peg", "REAL"),
                       ("roe_3y_min", "REAL"), ("roe_3y_avg", "REAL"), ("gm_trend", "REAL"),
                       ("growth_streak", "REAL"), ("rev_growth_streak", "REAL"),
                       ("avg_profit_yoy_3q", "REAL"), ("profit_accel", "REAL"),
                       ("div_years_5y", "REAL"), ("div_avg_5y", "REAL"), ("div_min_3y", "REAL"),
                       ("ret_20d", "REAL"), ("ret_60d", "REAL"),
                       ("ma_bull", "REAL"), ("break_high60", "REAL")):
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
        # 每日盘面宽度/情绪历史: 每个交易日落一条(盘中反复写则更新当天), 用于
        # 计算"今日读数在过去一年里处于什么冷热分位"——把"上涨占比8%"这种孤立
        # 数字变成"过去一年最冷的5%"这种有参照的信号(散户看不到、最能反映极值)。
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS market_breadth (
                date       TEXT PRIMARY KEY,   -- 交易日 YYYY-MM-DD
                up_share   REAL,               -- 上涨占比%: up/(up+down+flat)*100
                up_cnt     INTEGER,            -- 上涨家数(全市场快照口径)
                down_cnt   INTEGER,            -- 下跌家数
                flat_cnt   INTEGER,            -- 平盘家数
                limit_up   INTEGER,            -- 涨停家数(涨停池)
                limit_down INTEGER,            -- 跌停家数(乐咕真实跌停,可为空)
                zb_cnt     INTEGER,            -- 炸板家数
                seal_rate  REAL,               -- 封板率%
                max_board  INTEGER,            -- 最高连板
                amount_yi  REAL,               -- 两市成交额(亿元)
                vol_ratio  REAL,               -- 量能比%(较5日均量)
                updated_at TEXT
            )
            """
        )
        # 板块估值历史: 每交易日每个行业板块一条, 存该板块成分股 PE/PB 中位数。
        # 用于把"当前板块估值贵不贵"变成"处在过去一年的第几分位"这种有参照的信号。
        # (date, sector) 联合主键, 盘中多次写只留最新。
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sector_valuation (
                date       TEXT,                -- 交易日 YYYY-MM-DD
                sector     TEXT,                -- 板块名(同花顺口径)
                pe         REAL,                -- 成分股动态市盈率中位数(仅正值)
                pb         REAL,                -- 成分股市净率中位数
                cons_n     INTEGER,             -- 参与计算的成分股数
                updated_at TEXT,
                PRIMARY KEY (date, sector)
            )
            """
        )
        # 实盘/模拟盘信号跟踪账本 (Signal Ledger): 自动记录每次推荐信号并在后续更新时跟踪触及止盈/止损
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS signal_ledger (
                signal_date  TEXT,
                horizon      INTEGER,
                code         TEXT,
                name         TEXT,
                industry     TEXT,
                setups       TEXT,
                p_cond       REAL,
                p_tb         REAL,
                signal_close REAL,
                max_buy      REAL,
                stop_loss    REAL,
                take_profit  REAL,
                entry_date   TEXT,
                entry_price  REAL,
                exit_date    TEXT,
                exit_price   REAL,
                return_pct   REAL,
                status       TEXT,
                updated_at   TEXT,
                PRIMARY KEY (signal_date, horizon, code)
            )
            """
        )
        # 1) 历史季报序列 (Point-in-Time Financials): 按公告日 notice_date 对齐历史真实 ROE / 利润同比 / BPS
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pit_financials (
                code         TEXT,
                report_date  TEXT,
                notice_date  TEXT,
                eps          REAL,
                roe          REAL,
                rev_yoy      REAL,
                profit_yoy   REAL,
                bps          REAL,
                gross_margin REAL,
                board_name   TEXT,
                PRIMARY KEY (code, report_date)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pit_code_notice ON pit_financials(code, notice_date)"
        )
        # 2) 日内盘口与主力资金流 (Capital Flow & Order Imbalance)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS capital_flow (
                code           TEXT,
                date           TEXT,
                buy_big_ratio  REAL,
                sell_big_ratio REAL,
                net_big_ratio  REAL,
                outer_vol      REAL,
                inner_vol      REAL,
                active_buy_imb REAL,
                net_inflow_yi  REAL,
                updated_at     TEXT,
                PRIMARY KEY (code, date)
            )
            """
        )
        # 3) 事件与排雷日历 (Risk Calendar: 限售股解禁 + 业绩预告预亏/暴雷)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_calendar (
                code         TEXT,
                event_type   TEXT,
                event_date   TEXT,
                severity     TEXT,
                ratio_val    REAL,
                summary      TEXT,
                updated_at   TEXT,
                PRIMARY KEY (code, event_type, event_date)
            )
            """
        )
        # 4) 东财二级细分题材板块映射 (Subsector / Concept Map)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS subsector_map (
                code       TEXT PRIMARY KEY,
                board_name TEXT,
                board_code TEXT,
                updated_at TEXT
            )
            """
        )
        # 5) 场内 ETF 全量指标快照表 (1700+ 只场内 ETF，含分类、规模、折溢价、换手、阶段涨幅)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS etf_snapshot (
                code          TEXT PRIMARY KEY,
                name          TEXT,
                category      TEXT,
                price         REAL,
                chg_pct       REAL,
                open          REAL,
                high          REAL,
                low           REAL,
                amount_yi     REAL,
                turnover      REAL,
                scale_yi      REAL,
                iopv          REAL,
                discount_rate REAL,
                ret_20d       REAL,
                ret_60d       REAL,
                ret_250d      REAL,
                updated_at    TEXT
            )
            """
        )
        # 6) 场外开放式公募基金业绩库 (20000+ 只公募基金，含净值与各周期收益率、费率、规模、机构占比、最大回撤)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS mutual_fund_rank (
                code       TEXT PRIMARY KEY,
                name       TEXT,
                category   TEXT,
                nav_date   TEXT,
                unit_nav   REAL,
                acc_nav    REAL,
                day_chg    REAL,
                ret_1w     REAL,
                ret_1m     REAL,
                ret_3m     REAL,
                ret_6m     REAL,
                ret_1y     REAL,
                ret_2y     REAL,
                ret_3y     REAL,
                ret_ytd    REAL,
                ret_since  REAL,
                fee_pct    REAL,
                scale_yi   REAL,
                inst_ratio REAL,
                max_dd_1y  REAL,
                max_dd_all REAL,
                updated_at TEXT
            )
            """
        )
        _mf_cols = [r[1] for r in cur.execute("PRAGMA table_info(mutual_fund_rank)").fetchall()]
        for _c, _t in (("scale_yi", "REAL"), ("inst_ratio", "REAL"), ("max_dd_1y", "REAL"), ("max_dd_all", "REAL")):
            if _c not in _mf_cols:
                cur.execute(f"ALTER TABLE mutual_fund_rank ADD COLUMN {_c} {_t}")
        # 7) 场外公募基金历史净值走势表 (供基金产品详情页绘制业绩走势与净值曲线)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fund_nav_history (
                code     TEXT,
                date     TEXT,
                unit_nav REAL,
                day_chg  REAL,
                PRIMARY KEY (code, date)
            )
            """
        )


def save_market_breadth(date: str, row: dict):
    """写入/更新某交易日的盘面宽度快照(按 date 主键 UPSERT, 盘中多次写只留最新)。
    row 键与 market_breadth 列对应(缺的置 None)。date 为 YYYY-MM-DD。"""
    if not date:
        return
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    cols = ["up_share", "up_cnt", "down_cnt", "flat_cnt", "limit_up",
            "limit_down", "zb_cnt", "seal_rate", "max_board", "amount_yi",
            "vol_ratio"]
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO market_breadth "
            "(date," + ",".join(cols) + ",updated_at) "
            "VALUES (?," + ",".join(["?"] * len(cols)) + ",?) "
            "ON CONFLICT(date) DO UPDATE SET " +
            ",".join(f"{c}=excluded.{c}" for c in cols) +
            ",updated_at=excluded.updated_at",
            (date, *[row.get(c) for c in cols], now),
        )


def load_market_breadth(days: int = 250) -> pd.DataFrame:
    """读取最近 days 个交易日的盘面宽度历史(按日期升序)。"""
    cols = ["date", "up_share", "up_cnt", "down_cnt", "flat_cnt", "limit_up",
            "limit_down", "zb_cnt", "seal_rate", "max_board", "amount_yi",
            "vol_ratio"]
    with get_conn() as conn:
        try:
            df = pd.read_sql(
                "SELECT " + ",".join(cols) + " FROM market_breadth "
                "ORDER BY date DESC LIMIT ?", conn, params=(days,))
            return df.sort_values("date").reset_index(drop=True)
        except Exception:
            return pd.DataFrame(columns=cols)


def breadth_percentile(metric: str, value: float, days: int = 250) -> dict:
    """计算某盘面读数 value 在过去 days 个交易日历史中的分位(0-100)。
    分位 = 历史上 <= value 的天数占比 * 100。越低=越冷(该 metric 越小越极端时),
    如 up_share 分位越低代表越接近"历史最冷/最恐慌"。样本 < 20 返回 None。
    返回 {percentile, samples, min, max, cur} 或 None。"""
    df = load_market_breadth(days)
    if df is None or df.empty or metric not in df.columns:
        return None
    s = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(s) < 20:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    below = int((s <= v).sum())
    pct = below / len(s) * 100
    return {"percentile": pct, "samples": int(len(s)),
            "min": float(s.min()), "max": float(s.max()), "cur": v}


def save_sector_valuation(date: str, sector: str, pe, pb, cons_n=None):
    """写入/更新某交易日某板块的估值中位数(按 (date,sector) UPSERT)。
    pe/pb 可为 None(缺一不影响另一个)。date 为 YYYY-MM-DD。"""
    if not date or not sector:
        return
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sector_valuation "
            "(date,sector,pe,pb,cons_n,updated_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(date,sector) DO UPDATE SET "
            "pe=excluded.pe,pb=excluded.pb,cons_n=excluded.cons_n,"
            "updated_at=excluded.updated_at",
            (date, sector, pe, pb, cons_n, now),
        )


def save_sector_valuation_batch(rows: list):
    """批量写入板块估值历史(用于回填)。rows = [(date,sector,pe,pb,cons_n), ...]。"""
    if not rows:
        return
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO sector_valuation "
            "(date,sector,pe,pb,cons_n,updated_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(date,sector) DO UPDATE SET "
            "pe=excluded.pe,pb=excluded.pb,cons_n=excluded.cons_n,"
            "updated_at=excluded.updated_at",
            [(d, s, pe, pb, n, now) for (d, s, pe, pb, n) in rows],
        )


def load_sector_valuation(sector: str, days: int = 300) -> pd.DataFrame:
    """读取某板块最近 days 个交易日的估值历史(按日期升序)。"""
    cols = ["date", "pe", "pb", "cons_n"]
    with get_conn() as conn:
        try:
            df = pd.read_sql(
                "SELECT date,pe,pb,cons_n FROM sector_valuation "
                "WHERE sector=? ORDER BY date DESC LIMIT ?",
                conn, params=(sector, days))
            return df.sort_values("date").reset_index(drop=True)
        except Exception:
            return pd.DataFrame(columns=cols)


def sector_val_percentile(sector: str, metric: str, value: float,
                          days: int = 300) -> dict:
    """计算某板块当前 metric(pe/pb) 读数 value 在过去 days 交易日历史中的分位。
    分位 = 历史上 <= value 的天数占比 * 100。
    分位越高=当前估值处于历史偏贵区间; 越低=偏便宜。样本 < 20 返回 None。
    返回 {percentile, samples, min, max, cur} 或 None。"""
    if metric not in ("pe", "pb"):
        return None
    df = load_sector_valuation(sector, days)
    if df is None or df.empty or metric not in df.columns:
        return None
    s = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(s) < 20:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    below = int((s <= v).sum())
    pct = below / len(s) * 100
    return {"percentile": pct, "samples": int(len(s)),
            "min": float(s.min()), "max": float(s.max()), "cur": v}


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


def last_dates() -> dict:
    """一次取出每只股票的本地最新日期，避免逐只查询。"""
    with get_conn() as conn:
        try:
            rows = conn.execute(
                "SELECT code, MAX(date) FROM daily_kline GROUP BY code"
            ).fetchall()
        except Exception:
            return {}
    return {r[0]: r[1] for r in rows if r[0] and r[1]}


def last_bars_summary(recent_since: str) -> dict:
    """
    一次性取出每只股票的本地最新日期，以及 recent_since 之后的每日收盘价映射。
    返回: {code: {"last_date": str, "closes": {date: float}}}
    用于极速增量更新与除权除息(qfq)断层检测。
    """
    out = {}
    with get_conn() as conn:
        try:
            max_rows = conn.execute(
                "SELECT code, MAX(date) FROM daily_kline GROUP BY code"
            ).fetchall()
            for code, max_d in max_rows:
                if code and max_d:
                    out[str(code)] = {"last_date": str(max_d), "closes": {}}
            recent_rows = conn.execute(
                "SELECT code, date, close FROM daily_kline WHERE date >= ?",
                (str(recent_since),),
            ).fetchall()
            for code, d, c in recent_rows:
                item = out.get(str(code))
                if item is not None and d and c is not None:
                    item["closes"][str(d)] = float(c)
        except Exception:
            pass
    return out


def upsert_kline_rows(rows: list[tuple]):
    """
    批量增量合并日线行: rows 元素为 (code, date, open, high, low, close, volume, amount)。
    使用 INSERT OR REPLACE，既能追加新交易日，也能在盘中/收盘再次更新时刷新当日最新K线。
    """
    if not rows:
        return
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO daily_kline "
            "(code,date,open,high,low,close,volume,amount) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )


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
    """增量合并日线(不删历史旧数据),用 INSERT OR REPLACE 支持同日盘中到收盘覆盖更新。"""
    if df is None or df.empty:
        return
    df = df.copy()
    df["code"] = code
    cols = ["code", "date", "open", "high", "low", "close", "volume", "amount"]
    df = df[cols]
    with get_conn() as conn:
        rows = [tuple(r) for r in df.itertuples(index=False)]
        conn.executemany(
            "INSERT OR REPLACE INTO daily_kline "
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


def update_buy_price(code: str, buy_price: float):
    """只更新买入价(不动加入日/备注/现价)。用户手动改买入价时用。"""
    with get_conn() as conn:
        conn.execute("UPDATE watchlist SET buy_price = ? WHERE code = ?",
                     (float(buy_price or 0.0), code))


def save_watch_prices(prices: dict):
    """
    批量持久化自选的最近现价(重开软件不丢)。
    prices: {code: price(float)}。只更新已在自选里的 code。
    """
    if not prices:
        return
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.executemany(
            "UPDATE watchlist SET last_price = ?, last_price_time = ? "
            "WHERE code = ?",
            [(float(p), ts, c) for c, p in prices.items()
             if p is not None and float(p) > 0])


def remove_watch(code: str):
    """移除自选。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM watchlist WHERE code = ?", (code,))


def load_watchlist() -> pd.DataFrame:
    """读取自选列表(含持久化的最近现价 last_price)。"""
    cols = ["code", "name", "add_date", "buy_price", "note",
            "last_price", "last_price_time"]
    with get_conn() as conn:
        try:
            return pd.read_sql(
                "SELECT code,name,add_date,buy_price,note,"
                "last_price,last_price_time FROM watchlist "
                "ORDER BY add_date DESC", conn)
        except Exception:
            return pd.DataFrame(columns=cols)


# ==================== 基本面 / 估值 ====================
def save_fundamental(rows: dict):
    """
    批量写入基本面数据。rows: {code: {pe_ttm, pb, ps_ttm, total_mv, float_mv, roe,
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
                    (code, pe_ttm, pb, ps_ttm, total_mv, float_mv, roe,
                     gross_margin, net_margin, rev_yoy, profit_yoy,
                     debt_ratio, dividend_ratio, report_date, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(code) DO UPDATE SET
                    pe_ttm   = COALESCE(excluded.pe_ttm,   fundamental.pe_ttm),
                    pb       = COALESCE(excluded.pb,       fundamental.pb),
                    ps_ttm   = COALESCE(excluded.ps_ttm,   fundamental.ps_ttm),
                    total_mv = COALESCE(excluded.total_mv, fundamental.total_mv),
                    float_mv = COALESCE(excluded.float_mv, fundamental.float_mv),
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
                 d.get("total_mv"), d.get("float_mv"), d.get("roe"),
                 d.get("gross_margin"), d.get("net_margin"),
                 d.get("rev_yoy"), d.get("profit_yoy"),
                 d.get("debt_ratio"), d.get("dividend_ratio"),
                 d.get("report_date"), now),
            )


def load_fundamental() -> pd.DataFrame:
    """读取全部基本面数据(含成长/质量/流通市值/5年估值分位/同业分位/连续3年ROE/PEG/连续增长季度/5年分红与股息率字段)。"""
    _cols = ["code", "pe_ttm", "pb", "ps_ttm", "total_mv", "float_mv", "roe",
             "gross_margin", "net_margin", "rev_yoy", "profit_yoy",
             "debt_ratio", "dividend_ratio", "report_date",
             "pe_hist_pct", "pb_hist_pct", "pe_ind_pct", "peg",
             "roe_3y_min", "roe_3y_avg", "gm_trend",
             "growth_streak", "rev_growth_streak", "avg_profit_yoy_3q", "profit_accel",
             "div_years_5y", "div_avg_5y", "div_min_3y",
             "ret_20d", "ret_60d", "ma_bull", "break_high60",
             "updated_at"]
    with get_conn() as conn:
        for col in ("pe_hist_pct", "pb_hist_pct", "pe_ind_pct", "peg",
                    "roe_3y_min", "roe_3y_avg", "gm_trend",
                    "growth_streak", "rev_growth_streak", "avg_profit_yoy_3q", "profit_accel",
                    "div_years_5y", "div_avg_5y", "div_min_3y",
                    "ret_20d", "ret_60d", "ma_bull", "break_high60"):
            try:
                conn.execute(f"ALTER TABLE fundamental ADD COLUMN {col} REAL")
            except Exception:
                pass
        try:
            return pd.read_sql(
                "SELECT " + ",".join(_cols) + " FROM fundamental", conn)
        except Exception:
            return pd.DataFrame(columns=_cols)


load_fundamentals = load_fundamental


def load_fundamental_map() -> dict:
    """返回全量个股基本面与衍生特征字典。"""
    df = load_fundamental()
    if df is None or df.empty:
        return {}
    out = {}
    for r in df.itertuples(index=False):
        out[r.code] = {
            "pe_ttm": r.pe_ttm, "pb": r.pb, "ps_ttm": r.ps_ttm,
            "total_mv": r.total_mv, "float_mv": getattr(r, "float_mv", None),
            "roe": r.roe,
            "gross_margin": getattr(r, "gross_margin", None),
            "net_margin": getattr(r, "net_margin", None),
            "rev_yoy": getattr(r, "rev_yoy", None),
            "profit_yoy": getattr(r, "profit_yoy", None),
            "debt_ratio": getattr(r, "debt_ratio", None),
            "dividend_ratio": getattr(r, "dividend_ratio", None),
            "pe_hist_pct": getattr(r, "pe_hist_pct", None),
            "pb_hist_pct": getattr(r, "pb_hist_pct", None),
            "pe_ind_pct": getattr(r, "pe_ind_pct", None),
            "peg": getattr(r, "peg", None),
            "roe_3y_min": getattr(r, "roe_3y_min", None),
            "roe_3y_avg": getattr(r, "roe_3y_avg", None),
            "gm_trend": getattr(r, "gm_trend", None),
            "growth_streak": getattr(r, "growth_streak", None),
            "rev_growth_streak": getattr(r, "rev_growth_streak", None),
            "avg_profit_yoy_3q": getattr(r, "avg_profit_yoy_3q", None),
            "profit_accel": getattr(r, "profit_accel", None),
            "div_years_5y": getattr(r, "div_years_5y", None),
            "div_avg_5y": getattr(r, "div_avg_5y", None),
            "div_min_3y": getattr(r, "div_min_3y", None),
            "ret_20d": getattr(r, "ret_20d", None),
            "ret_60d": getattr(r, "ret_60d", None),
            "ma_bull": getattr(r, "ma_bull", None),
            "break_high60": getattr(r, "break_high60", None),
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
    """返回单只股票的基本面(含成长/质量/流通市值字段);无则 None。"""
    with get_conn() as conn:
        try:
            r = conn.execute(
                "SELECT pe_ttm,pb,ps_ttm,total_mv,float_mv,roe,gross_margin,net_margin,"
                "rev_yoy,profit_yoy,debt_ratio,dividend_ratio,report_date,updated_at "
                "FROM fundamental WHERE code=?", (code,)).fetchone()
        except Exception:
            return None
    if not r:
        return None
    return {"pe_ttm": r[0], "pb": r[1], "ps_ttm": r[2],
            "total_mv": r[3], "float_mv": r[4], "roe": r[5],
            "gross_margin": r[6], "net_margin": r[7], "rev_yoy": r[8],
            "profit_yoy": r[9], "debt_ratio": r[10], "dividend_ratio": r[11],
            "report_date": r[12], "updated_at": r[13]}


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
            rows = conn.execute(
                "SELECT DISTINCT k.code, COALESCE(s.name, e.name, '') AS name "
                "FROM daily_kline k "
                "LEFT JOIN stock_list s ON k.code = s.code "
                "LEFT JOIN etf_list e ON k.code = e.code"
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
    """按代码取名称(依次查询 stock_list -> etf_list -> mutual_fund_rank)。查不到返回空串。"""
    if not code:
        return ""
    code = str(code).zfill(6)
    with get_conn() as conn:
        for tbl in ("stock_list", "etf_list", "mutual_fund_rank"):
            try:
                row = conn.execute(
                    f"SELECT name FROM {tbl} WHERE code = ?", (code,)
                ).fetchone()
                if row and row[0]:
                    return str(row[0])
            except Exception:
                pass
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


def save_signal_ledger(rows: list[dict]):
    """写入或更新实盘/模拟盘信号跟踪账本。"""
    if not rows:
        return
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        for r in rows:
            conn.execute(
                """
                INSERT INTO signal_ledger
                    (signal_date, horizon, code, name, industry, setups,
                     p_cond, p_tb, signal_close, max_buy, stop_loss, take_profit,
                     entry_date, entry_price, exit_date, exit_price, return_pct,
                     status, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(signal_date, horizon, code) DO UPDATE SET
                    name         = COALESCE(excluded.name, signal_ledger.name),
                    industry     = COALESCE(excluded.industry, signal_ledger.industry),
                    setups       = COALESCE(excluded.setups, signal_ledger.setups),
                    p_cond       = COALESCE(excluded.p_cond, signal_ledger.p_cond),
                    p_tb         = COALESCE(excluded.p_tb, signal_ledger.p_tb),
                    signal_close = COALESCE(excluded.signal_close, signal_ledger.signal_close),
                    max_buy      = COALESCE(excluded.max_buy, signal_ledger.max_buy),
                    stop_loss    = COALESCE(excluded.stop_loss, signal_ledger.stop_loss),
                    take_profit  = COALESCE(excluded.take_profit, signal_ledger.take_profit),
                    entry_date   = COALESCE(excluded.entry_date, signal_ledger.entry_date),
                    entry_price  = COALESCE(excluded.entry_price, signal_ledger.entry_price),
                    exit_date    = COALESCE(excluded.exit_date, signal_ledger.exit_date),
                    exit_price   = COALESCE(excluded.exit_price, signal_ledger.exit_price),
                    return_pct   = COALESCE(excluded.return_pct, signal_ledger.return_pct),
                    status       = COALESCE(excluded.status, signal_ledger.status),
                    updated_at   = excluded.updated_at
                """,
                (
                    str(r.get("signal_date")), int(r.get("horizon", 10)), str(r.get("code")).zfill(6),
                    r.get("name"), r.get("industry"), r.get("setups"),
                    r.get("p_cond"), r.get("p_tb"), r.get("signal_close"),
                    r.get("max_buy"), r.get("stop_loss"), r.get("take_profit"),
                    r.get("entry_date"), r.get("entry_price"),
                    r.get("exit_date"), r.get("exit_price"), r.get("return_pct"),
                    r.get("status", "待开盘"), now,
                ),
            )


def load_signal_ledger(horizon: int = None, limit: int = 120) -> pd.DataFrame:
    """读取信号跟踪账本，按 signal_date 降序、p_cond 降序。"""
    cols = [
        "signal_date", "horizon", "code", "name", "industry", "setups",
        "p_cond", "p_tb", "signal_close", "max_buy", "stop_loss", "take_profit",
        "entry_date", "entry_price", "exit_date", "exit_price", "return_pct",
        "status", "updated_at",
    ]
    with get_conn() as conn:
        try:
            if horizon is not None:
                return pd.read_sql(
                    "SELECT " + ",".join(cols) + " FROM signal_ledger "
                    "WHERE horizon = ? ORDER BY signal_date DESC, p_cond DESC LIMIT ?",
                    conn, params=(int(horizon), int(limit)),
                )
            return pd.read_sql(
                "SELECT " + ",".join(cols) + " FROM signal_ledger "
                "ORDER BY signal_date DESC, p_cond DESC LIMIT ?",
                conn, params=(int(limit),),
            )
        except Exception:
            return pd.DataFrame(columns=cols)


def save_pit_financials(rows: list[tuple]):
    """
    批量写入历史季报序列 (Point-in-Time Financials)。
    rows 元素为: (code, report_date, notice_date, eps, roe, rev_yoy, profit_yoy, bps, gross_margin, board_name)
    """
    if not rows:
        return
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO pit_financials
                (code, report_date, notice_date, eps, roe, rev_yoy, profit_yoy, bps, gross_margin, board_name)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )


def load_pit_financials() -> pd.DataFrame:
    """读取全部历史季报序列，按 code, notice_date, report_date 升序排列。"""
    cols = [
        "code", "report_date", "notice_date", "eps", "roe",
        "rev_yoy", "profit_yoy", "bps", "gross_margin", "board_name",
    ]
    with get_conn() as conn:
        try:
            df = pd.read_sql(
                "SELECT " + ",".join(cols) + " FROM pit_financials "
                "ORDER BY code ASC, notice_date ASC, report_date ASC",
                conn,
            )
            return df
        except Exception:
            return pd.DataFrame(columns=cols)


def save_capital_flow(rows: list[tuple]):
    """
    批量写入日内盘口大单比例与内外盘主动买盘流向。
    rows 元素为: (code, date, buy_big_ratio, sell_big_ratio, net_big_ratio, outer_vol, inner_vol, active_buy_imb, net_inflow_yi)
    """
    if not rows:
        return
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO capital_flow
                (code, date, buy_big_ratio, sell_big_ratio, net_big_ratio,
                 outer_vol, inner_vol, active_buy_imb, net_inflow_yi, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            [(*r, now) for r in rows],
        )


def load_latest_capital_flow() -> dict:
    """读取每只股票最新一个交易日的主力资金流与盘口主动买盘失衡度。"""
    with get_conn() as conn:
        try:
            max_d = conn.execute("SELECT MAX(date) FROM capital_flow").fetchone()[0]
            if not max_d:
                return {}
            rows = conn.execute(
                "SELECT code, date, buy_big_ratio, sell_big_ratio, net_big_ratio, "
                "active_buy_imb, net_inflow_yi FROM capital_flow WHERE date = ?",
                (str(max_d),),
            ).fetchall()
            return {
                str(r[0]).zfill(6): {
                    "date": str(r[1]),
                    "buy_big_ratio": r[2],
                    "sell_big_ratio": r[3],
                    "net_big_ratio": r[4],
                    "active_buy_imb": r[5],
                    "net_inflow_yi": r[6],
                }
                for r in rows if r[0]
            }
        except Exception:
            return {}


def save_risk_calendar(rows: list[tuple]):
    """
    批量写入风险与催化事件日历 (限售股解禁 / 业绩预告预亏或暴增)。
    rows 元素为: (code, event_type, event_date, severity, ratio_val, summary)
    """
    if not rows:
        return
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO risk_calendar
                (code, event_type, event_date, severity, ratio_val, summary, updated_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            [(*r, now) for r in rows],
        )


def load_risk_calendar_map() -> dict[str, list[dict]]:
    """返回 {code: [{event_type, event_date, severity, ratio_val, summary}, ...]}。"""
    with get_conn() as conn:
        try:
            rows = conn.execute(
                "SELECT code, event_type, event_date, severity, ratio_val, summary "
                "FROM risk_calendar ORDER BY event_date ASC"
            ).fetchall()
            out: dict[str, list[dict]] = {}
            for code, et, ed, sev, rv, summ in rows:
                if not code:
                    continue
                out.setdefault(str(code).zfill(6), []).append({
                    "event_type": str(et or ""),
                    "event_date": str(ed or ""),
                    "severity": str(sev or "neutral"),
                    "ratio_val": float(rv) if rv is not None else 0.0,
                    "summary": str(summ or ""),
                })
            return out
        except Exception:
            return {}


def save_subsector_map(rows: list[tuple]):
    """写入股票->东财二级细分板块映射: rows 元素为 (code, board_name, board_code)。"""
    if not rows:
        return
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO subsector_map (code, board_name, board_code, updated_at)
            VALUES (?,?,?,?)
            """,
            [(str(c).zfill(6), str(bn or ""), str(bc or ""), now) for c, bn, bc in rows if c and bn],
        )


def load_subsector_map() -> dict[str, str]:
    """返回 {code: 二级细分板块名称}。"""
    with get_conn() as conn:
        try:
            rows = conn.execute("SELECT code, board_name FROM subsector_map").fetchall()
            return {str(r[0]).zfill(6): str(r[1]) for r in rows if r[0] and r[1]}
        except Exception:
            return {}


def save_etf_snapshot(rows: list[tuple]):
    """
    批量写入场内 ETF 全量指标快照。
    rows 元素为: (code, name, category, price, chg_pct, open, high, low,
                 amount_yi, turnover, scale_yi, iopv, discount_rate, ret_20d, ret_60d, ret_250d)
    """
    if not rows:
        return
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO etf_snapshot
                (code, name, category, price, chg_pct, open, high, low,
                 amount_yi, turnover, scale_yi, iopv, discount_rate,
                 ret_20d, ret_60d, ret_250d, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [(*r, now) for r in rows],
        )


def load_etf_snapshot() -> pd.DataFrame:
    """读取本地已缓存的场内 ETF 全量指标快照表。"""
    cols = [
        "code", "name", "category", "price", "chg_pct", "open", "high", "low",
        "amount_yi", "turnover", "scale_yi", "iopv", "discount_rate",
        "ret_20d", "ret_60d", "ret_250d", "updated_at",
    ]
    with get_conn() as conn:
        try:
            return pd.read_sql("SELECT " + ",".join(cols) + " FROM etf_snapshot", conn)
        except Exception:
            return pd.DataFrame(columns=cols)


def save_mutual_fund_rank(rows: list[tuple]):
    """
    批量写入场外公募基金业绩库（保留已核验缓存的 scale_yi, inst_ratio, max_dd_1y, max_dd_all）。
    rows 元素为: (code, name, category, nav_date, unit_nav, acc_nav, day_chg,
                 ret_1w, ret_1m, ret_3m, ret_6m, ret_1y, ret_2y, ret_3y, ret_ytd, ret_since, fee_pct)
    """
    if not rows:
        return
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO mutual_fund_rank
                (code, name, category, nav_date, unit_nav, acc_nav, day_chg,
                 ret_1w, ret_1m, ret_3m, ret_6m, ret_1y, ret_2y, ret_3y,
                 ret_ytd, ret_since, fee_pct, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(code) DO UPDATE SET
                name       = excluded.name,
                category   = excluded.category,
                nav_date   = excluded.nav_date,
                unit_nav   = excluded.unit_nav,
                acc_nav    = excluded.acc_nav,
                day_chg    = excluded.day_chg,
                ret_1w     = excluded.ret_1w,
                ret_1m     = excluded.ret_1m,
                ret_3m     = excluded.ret_3m,
                ret_6m     = excluded.ret_6m,
                ret_1y     = excluded.ret_1y,
                ret_2y     = excluded.ret_2y,
                ret_3y     = excluded.ret_3y,
                ret_ytd    = excluded.ret_ytd,
                ret_since  = excluded.ret_since,
                fee_pct    = excluded.fee_pct,
                updated_at = excluded.updated_at
            """,
            [(*r, now) for r in rows],
        )


def update_mutual_fund_deep_metrics(rows: list[tuple]):
    """
    批量回填场外公募基金的深度核验指标：
    支持 5 元组 (scale_yi, inst_ratio, max_dd_1y, max_dd_all, code)
    或 10 元组 (scale_yi, inst_ratio, max_dd_1y, max_dd_all, establish_years, sharpe_1y, vol_1y, manager_name, manager_years, code)
    """
    if not rows:
        return
    with get_conn() as conn:
        for col, ctype in (
            ("scale_yi", "REAL"), ("inst_ratio", "REAL"), ("max_dd_1y", "REAL"), ("max_dd_all", "REAL"),
            ("establish_years", "REAL"), ("sharpe_1y", "REAL"), ("vol_1y", "REAL"),
            ("manager_name", "TEXT"), ("manager_years", "REAL"),
        ):
            try:
                conn.execute(f"ALTER TABLE mutual_fund_rank ADD COLUMN {col} {ctype}")
            except Exception:
                pass
        if len(rows[0]) >= 10:
            conn.executemany(
                """
                UPDATE mutual_fund_rank
                SET scale_yi        = COALESCE(?, scale_yi),
                    inst_ratio      = COALESCE(?, inst_ratio),
                    max_dd_1y       = COALESCE(?, max_dd_1y),
                    max_dd_all      = COALESCE(?, max_dd_all),
                    establish_years = COALESCE(?, establish_years),
                    sharpe_1y       = COALESCE(?, sharpe_1y),
                    vol_1y          = COALESCE(?, vol_1y),
                    manager_name    = COALESCE(?, manager_name),
                    manager_years   = COALESCE(?, manager_years)
                WHERE code = ?
                """,
                rows,
            )
        else:
            conn.executemany(
                """
                UPDATE mutual_fund_rank
                SET scale_yi   = COALESCE(?, scale_yi),
                    inst_ratio = COALESCE(?, inst_ratio),
                    max_dd_1y  = COALESCE(?, max_dd_1y),
                    max_dd_all = COALESCE(?, max_dd_all)
                WHERE code = ?
                """,
                rows,
            )


def load_mutual_fund_rank() -> pd.DataFrame:
    """读取本地已缓存的场外公募开放式基金业绩表(含规模、机构占比、最大回撤、成立年限、夏普比率、波动率、基金经理)。"""
    cols = [
        "code", "name", "category", "nav_date", "unit_nav", "acc_nav", "day_chg",
        "ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_2y", "ret_3y",
        "ret_ytd", "ret_since", "fee_pct",
        "scale_yi", "inst_ratio", "max_dd_1y", "max_dd_all",
        "establish_years", "sharpe_1y", "vol_1y", "manager_name", "manager_years",
        "updated_at",
    ]
    with get_conn() as conn:
        for col, ctype in (
            ("scale_yi", "REAL"), ("inst_ratio", "REAL"), ("max_dd_1y", "REAL"), ("max_dd_all", "REAL"),
            ("establish_years", "REAL"), ("sharpe_1y", "REAL"), ("vol_1y", "REAL"),
            ("manager_name", "TEXT"), ("manager_years", "REAL"),
        ):
            try:
                conn.execute(f"ALTER TABLE mutual_fund_rank ADD COLUMN {col} {ctype}")
            except Exception:
                pass
        try:
            return pd.read_sql("SELECT " + ",".join(cols) + " FROM mutual_fund_rank", conn)
        except Exception:
            return pd.DataFrame(columns=cols)


def save_fund_nav_history(code: str, df: pd.DataFrame):
    """保存单只场外基金的历史净值序列 (date, unit_nav, day_chg)。"""
    if df is None or df.empty:
        return
    code = str(code).zfill(6)
    rows = [
        (code, str(r.date)[:10], float(r.unit_nav), float(r.day_chg) if r.day_chg is not None and r.day_chg == r.day_chg else 0.0)
        for r in df.itertuples(index=False)
        if r.unit_nav is not None and r.unit_nav == r.unit_nav
    ]
    if not rows:
        return
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO fund_nav_history (code, date, unit_nav, day_chg) VALUES (?,?,?,?)",
            rows,
        )


def load_fund_nav_history(code: str) -> pd.DataFrame:
    """读取单只场外基金的历史净值序列，按 date 升序。"""
    code = str(code).zfill(6)
    with get_conn() as conn:
        try:
            return pd.read_sql(
                "SELECT date, unit_nav, day_chg FROM fund_nav_history WHERE code = ? ORDER BY date ASC",
                conn, params=(code,),
            )
        except Exception:
            return pd.DataFrame(columns=["date", "unit_nav", "day_chg"])


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
