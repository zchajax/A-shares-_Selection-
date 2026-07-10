"""
UI 层 - DearPyGui 主界面
========================================
布局:
  左侧:数据状态 + 大盘状态 + 数据更新 + 策略选择 + 参数滑块 + 选股/回测
  右侧(标签页):
    ① 选股 & K线:选股结果表(可加自选/回测这批)+ 三层联动K线
    ② 今日推荐:一键跑策略排行榜,把最强策略选出的票置顶
    ③ 自选持仓:已加入自选的票,显示浮动盈亏

运行:  python main.py
"""
import os
import threading
import time as _time
from datetime import datetime

import pandas as pd
import dearpygui.dearpygui as dpg

from app.data import database as db
from app.data import fetcher
from app.data import realtime
from app.strategy import indicators as ind
from app.strategy import scanner
from app.strategy import backtest as bt
from app.strategy import ranking
from app.strategy import optimizer
from app.strategy import funda
from app.strategy.market import MarketTrend
from app.strategy.base import ALL_STRATEGIES
from app.report import exporter

# A 股配色:涨=红 跌=绿
RED = (220, 40, 40, 255)
GREEN = (0, 170, 90, 255)


# ---------- 中文字体 ----------
def load_chinese_font():
    """加载 Windows 系统中文字体,否则中文显示为方块。"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",   # 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",  # 黑体
        r"C:\Windows\Fonts\simsun.ttc",  # 宋体
    ]
    font_path = next((p for p in candidates if os.path.exists(p)), None)
    with dpg.font_registry():
        if font_path:
            with dpg.font(font_path, 18) as f:
                # 新版 DearPyGui 中文范围已自动处理,add_font_range_hint 为 no-op
                # 且会告警,故不再调用,直接绑定即可。
                dpg.bind_font(f)


# ---------- 全局状态 ----------
STATE = {
    "current_strategy": None,   # 当前策略实例
    "results": None,            # 选股结果 DataFrame
    "backtest": None,           # 最近一次回测结果 BacktestResult
    "rank_df": None,            # 策略排行榜 DataFrame
    "picks_df": None,           # 今日推荐 DataFrame
    "etf_spot": None,           # ETF 实时快照 DataFrame
    "cur_code": None,           # 当前查看K线的代码
    "cur_period": "D",          # 当前K线周期: D/W/M/MIN(分时)
    "poll_on": False,           # 分时轮询开关
    "poll_thread": None,        # 分时轮询线程
    "poll_code": None,          # 轮询目标代码(仅拉这只)
    "intraday_code": None,      # 当前分时图已绘制的代码(增量更新判定用)
    "kl_bars": None,            # 当前主图K线数据(用于鼠标悬停提示): list[dict]
    "_last_click_code": None,   # 上次点击的代码(双击判定)
    "_last_click_ts": 0.0,      # 上次点击时间戳(双击判定)
}

# 分时轮询间隔(秒):免费源限流,只盯 1 只,5 秒够用
POLL_INTERVAL = 5


# ---------- 柱状图红/绿主题(涨红跌绿) ----------
def _make_bar_themes():
    """创建红、绿两个柱状图主题,用于成交额/MACD 柱按涨跌上色。"""
    with dpg.theme(tag="bar_red"):
        with dpg.theme_component(dpg.mvBarSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Fill, RED, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, RED, category=dpg.mvThemeCat_Plots)
    with dpg.theme(tag="bar_green"):
        with dpg.theme_component(dpg.mvBarSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Fill, GREEN, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, GREEN, category=dpg.mvThemeCat_Plots)
    # 周期按钮选中态主题(蓝底)
    with dpg.theme(tag="period_on"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 110, 200))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (55, 130, 220))
    # 分时价格线红/绿主题(轮询增量更新时复用,避免反复新建 theme)
    with dpg.theme(tag="line_red"):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, RED, category=dpg.mvThemeCat_Plots)
    with dpg.theme(tag="line_green"):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, GREEN, category=dpg.mvThemeCat_Plots)


# ---------- 业务回调 ----------
def on_strategy_change(sender, app_data):
    """切换策略时,重建参数控件区。"""
    key = STRATEGY_LABEL2KEY[app_data]
    cls = ALL_STRATEGIES[key]
    STATE["current_strategy"] = cls()
    dpg.delete_item("param_area", children_only=True)
    dpg.add_text(cls.desc, parent="param_area", wrap=280, color=(150, 150, 150))
    for p in cls.params:
        dpg.add_slider_float(
            label=p.label, parent="param_area",
            default_value=p.default, min_value=p.min, max_value=p.max,
            format="%.0f" if p.is_int else "%.2f",
            width=170, tag=f"param_{p.key}",
            callback=lambda s, a, u: STATE["current_strategy"].set_param(u, a),
            user_data=p.key,
        )


def _set_status(msg):
    dpg.set_value("status_text", msg)


def _refresh_cache_info():
    """刷新左侧"本地数据"状态文本。"""
    s = db.cache_summary()
    if s["codes"] > 0:
        txt = f"本地已有 {s['codes']} 只 · 最新 {s['last_date']}\n(可不更新, 直接选股)"
    else:
        txt = "本地暂无数据, 请先更新"
    dpg.set_value("cache_info", txt)


def _refresh_market_state():
    """刷新左侧"大盘状态"——多头(红)/弱势(绿),供择时参考。"""
    try:
        mt = MarketTrend("sh000001")
        if not mt.ready:
            dpg.set_value("market_info", "大盘: 指数未拉取")
            dpg.configure_item("market_info", color=(150, 150, 150))
            return
        d, strong = mt.latest_state()
        if strong:
            dpg.set_value("market_info", f"大盘: 多头 (走强)  {d}")
            dpg.configure_item("market_info", color=RED)
        else:
            dpg.set_value("market_info", f"大盘: 弱势 (观望)  {d}")
            dpg.configure_item("market_info", color=GREEN)
    except Exception:
        dpg.set_value("market_info", "大盘: 未知")


def on_update_data():
    """后台线程:更新股票列表 + 增量拉取部分日线 + 大盘指数。"""
    def worker():
        try:
            _set_status("正在获取列表(含主流股排序,首次稍慢)...")
            df = fetcher.update_stock_list()
            codes = df["code"].tolist()
            n = min(int(dpg.get_value("update_count")), len(codes))
            _set_status(f"共 {len(codes)} 只,准备拉取前 {n} 只(主流股优先)日线...")

            def cb(done, total, code):
                _set_status(f"拉取日线 {done}/{n}  当前:{code}")

            # 首次全量从 2021 年起,保证回测有 4 年+样本;已有数据则自动增量
            # 腾讯源可并发,10 线程拉取,400 只约 3-5 分钟
            fetcher.update_all_kline(codes[:n], start_date="20210101",
                                     incremental=True, progress_cb=cb, workers=10)
            # 顺带更新大盘指数(用于大盘趋势过滤)
            _set_status("拉取大盘指数(上证综指)...")
            fetcher.update_index_kline("sh000001", start_date="20210101")

            # 拉取行业分类(巨潮,串行,增量:只查还没行业的;已分类的秒跳过)
            cached = db.list_cached_codes()
            miss = db.codes_without_industry(cached)
            if miss:
                _set_status(f"拉取行业分类 0/{len(miss)}(仅补缺失,可稍候)...")
                fetcher.update_industry(
                    cached, incremental=True,
                    progress_cb=lambda d, t, c: _set_status(
                        f"拉取行业分类 {d}/{t}  当前:{c}"))
            # 拉取基本面/估值(百度+巨潮,串行,增量:只补缺失的;可选)
            if bool(dpg.get_value("fetch_funda")):
                miss_f = db.codes_without_fundamental(cached)
                if miss_f:
                    _set_status(f"拉取基本面 0/{len(miss_f)}(每只1-2s,可稍候)...")
                    fetcher.update_fundamental(
                        cached, incremental=True,
                        progress_cb=lambda d, t, c: _set_status(
                            f"拉取基本面 {d}/{t}  当前:{c}"))
            # 拉取 ETF(列表+日线,可选):约50只主流宽基/行业ETF,串行,增量
            if bool(dpg.get_value("fetch_etf")):
                _set_status("拉取 ETF 列表(主流宽基+行业约50只)...")
                etf_df = fetcher.update_etf_list(only_mainstream=True)
                _set_status(f"拉取 ETF 日线 0/{len(etf_df)}...")
                fetcher.update_all_etf_kline(
                    start_date="20210101", incremental=True,
                    progress_cb=lambda d, t, c: _set_status(
                        f"拉取 ETF 日线 {d}/{t}  当前:{c}"))
            cached = db.list_cached_codes()
            _set_status(f"数据更新完成,已缓存 {len(cached)} 只")
            _refresh_cache_info()
            _refresh_market_state()
        except Exception as e:
            _set_status(f"更新失败: {e}")

    threading.Thread(target=worker, daemon=True).start()


def on_run_scan():
    """后台线程:执行选股。"""
    if STATE["current_strategy"] is None:
        _set_status("请先选择策略")
        return
    if not db.list_cached_codes():
        _set_status("本地无数据,请先点『更新股票数据』")
        return

    def worker():
        _set_status("正在扫描全市场...")
        res = scanner.scan(
            STATE["current_strategy"],
            progress_cb=lambda d, t: _set_status(f"扫描中 {d}/{t}"),
        )
        # 基本面过滤(可选):按界面阈值对选股结果做二次筛选
        raw_n = 0 if res is None else len(res)
        res = _apply_funda_filter(res)
        STATE["results"] = res
        _render_results(res)
        kept = 0 if res is None else len(res)
        if _funda_filter_on() and raw_n:
            _set_status(f"选股完成,命中 {raw_n} 只,基本面过滤后剩 {kept} 只")
        else:
            _set_status(f"选股完成,共命中 {kept} 只")

    threading.Thread(target=worker, daemon=True).start()


def _funda_filter_on() -> bool:
    """基本面过滤是否启用(总开关)。"""
    return dpg.does_item_exist("funda_on") and bool(dpg.get_value("funda_on"))


def _apply_funda_filter(df):
    """按左侧基本面控件的阈值过滤选股结果;未启用则原样返回。"""
    if df is None or df.empty or not _funda_filter_on():
        return df
    try:
        th = {
            "pe_max": float(dpg.get_value("funda_pe")),
            "pb_max": float(dpg.get_value("funda_pb")),
            "roe_min": float(dpg.get_value("funda_roe")),
            "mv_min": float(dpg.get_value("funda_mv")),
            "drop_missing": bool(dpg.get_value("funda_dropmiss")),
        }
        return funda.apply_filter(df, th)
    except Exception:
        return df


# ---------- 报告导出 ----------
def _current_market_state():
    """返回 (date, strong) 或 None,供报告标注大盘状态。"""
    try:
        mt = MarketTrend("sh000001")
        if not mt.ready:
            return None
        return mt.latest_state()
    except Exception:
        return None


def on_export_report():
    """把当前分析结果(排行榜/今日推荐/选股结果/自选)一键导出为 Excel + HTML。"""
    def worker():
        try:
            _set_status("正在生成报告(Excel + HTML)...")
            snap = exporter.build_snapshot(
                rank_df=STATE.get("rank_df"),
                picks_df=STATE.get("picks_df"),
                results_df=STATE.get("results"),
                results_name=(type(STATE["current_strategy"]).name
                              if STATE.get("current_strategy") else ""),
                market=_current_market_state(),
            )
            xlsx, html = exporter.export_all(snap)
            STATE["last_report_dir"] = os.path.dirname(html)
            STATE["last_report_html"] = html
            _set_status(f"报告已导出到 reports/ :\n{os.path.basename(xlsx)}\n{os.path.basename(html)}")
        except Exception as e:
            _set_status(f"导出失败: {e}")

    threading.Thread(target=worker, daemon=True).start()


def on_open_report_dir():
    """在资源管理器中打开报告输出目录。"""
    d = STATE.get("last_report_dir") or exporter.REPORT_DIR
    try:
        os.makedirs(d, exist_ok=True)
        os.startfile(d)  # Windows 专用
    except Exception as e:
        _set_status(f"打开目录失败: {e}")


# ---------- 回测 ----------
def on_open_backtest():
    """打开回测弹窗。"""
    if STATE["current_strategy"] is None:
        _set_status("请先选择策略")
        return
    if not db.list_cached_codes():
        _set_status("本地无数据,无法回测,请先更新数据")
        return
    if dpg.does_item_exist("bt_window"):
        dpg.delete_item("bt_window")

    cls = type(STATE["current_strategy"])
    with dpg.window(label=f"策略回测 - {cls.name}", tag="bt_window",
                    width=920, height=720, pos=(180, 60), modal=False):
        dpg.add_text(f"对「{cls.name}」做历史回测:过去每天按此策略选股买入,统计真实盈亏",
                     wrap=880, color=(160, 200, 255))
        dpg.add_text("原理:每个交易日只用截止当天的数据做决策(无未来函数),"
                     "次日开盘买入,触发止盈/止损或持满N天卖出。",
                     wrap=880, color=(130, 130, 130))
        dpg.add_separator()

        # ---- 回测参数 ----
        with dpg.group(horizontal=True):
            dpg.add_input_int(label="持有天数", default_value=10, min_value=1,
                              max_value=120, width=120, tag="bt_hold")
            dpg.add_input_int(label="最多持仓数", default_value=5, min_value=1,
                              max_value=50, width=120, tag="bt_maxpos")
        with dpg.group(horizontal=True):
            dpg.add_slider_float(label="止盈(%)", default_value=15, min_value=0,
                                 max_value=50, format="%.0f", width=200, tag="bt_tp")
            dpg.add_slider_float(label="止损(%)", default_value=8, min_value=0,
                                 max_value=30, format="%.0f", width=200, tag="bt_sl")
        with dpg.group(horizontal=True):
            dpg.add_input_text(label="起始日期(留空=全部历史)", default_value="2021-07-01",
                               width=200, tag="bt_start", hint="YYYY-MM-DD")
            dpg.add_checkbox(label="大盘趋势过滤(推荐:抬胜率砍回撤)",
                             default_value=True, tag="bt_market")
        dpg.add_text("回测范围:", color=(150, 150, 150))
        with dpg.group(horizontal=True):
            dpg.add_radio_button(("全池(本地全部)", "仅当前选股结果"),
                                 horizontal=True, tag="bt_scope",
                                 default_value="全池(本地全部)")
        with dpg.group(horizontal=True):
            dpg.add_button(label="开始回测", callback=on_run_backtest,
                           width=160, height=32, tag="bt_run_btn")
            dpg.add_button(label="卖出参数网格调优", callback=on_run_optimize,
                           width=200, height=32, tag="bt_opt_btn")
            dpg.add_button(label="买入参数寻优", callback=on_run_param_search,
                           width=160, height=32, tag="bt_psearch_btn")
        dpg.add_text("准备就绪", tag="bt_status", color=(255, 200, 100))
        dpg.add_separator()

        # ---- 结果区(回测后填充) ----
        dpg.add_text("回测结果", color=(120, 220, 160))
        dpg.add_child_window(tag="bt_stats_area", height=110, border=True)
        dpg.add_text("资金曲线(初始10万)")
        with dpg.plot(tag="bt_equity_plot", height=170, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="交易笔序/时间", tag="bt_eqx")
            dpg.add_plot_axis(dpg.mvYAxis, label="账户权益", tag="bt_eqy")
        dpg.add_text("交易明细(最近50笔)", tag="bt_detail_title")
        with dpg.table(tag="bt_trades_table", header_row=True, resizable=True,
                       policy=dpg.mvTable_SizingStretchProp, height=170, scrollY=True):
            for col in ["代码", "名称", "买入日", "卖出日", "买入价", "卖出价", "持有", "收益%", "原因"]:
                dpg.add_table_column(label=col)


def on_run_backtest():
    """后台线程执行回测,完成后渲染结果。"""
    strat = STATE["current_strategy"]
    if strat is None:
        return
    hold = int(dpg.get_value("bt_hold"))
    maxpos = int(dpg.get_value("bt_maxpos"))
    tp = float(dpg.get_value("bt_tp")) / 100.0
    sl = float(dpg.get_value("bt_sl")) / 100.0
    start = dpg.get_value("bt_start").strip() or None
    mkt = bool(dpg.get_value("bt_market"))
    scope = dpg.get_value("bt_scope")

    # 回测范围:仅当前选股结果 时,把选出的代码传给回测
    codes = None
    if scope.startswith("仅当前") and STATE["results"] is not None \
            and not STATE["results"].empty:
        codes = STATE["results"]["code"].tolist()

    def worker():
        dpg.configure_item("bt_run_btn", enabled=False)
        dpg.configure_item("bt_opt_btn", enabled=False)
        dpg.configure_item("bt_psearch_btn", enabled=False)
        dpg.set_value("bt_status", "回测中... (遍历历史逐日模拟,请稍候)")
        try:
            res = bt.run_backtest(
                strat, codes=codes, hold_days=hold, take_profit=tp, stop_loss=sl,
                max_positions=maxpos, start_from=start, market_filter=mkt,
                progress_cb=lambda d, t: dpg.set_value(
                    "bt_status", f"回测中... 处理股票 {d}/{t}"),
            )
            STATE["backtest"] = res
            _render_backtest(res)
            n = res.stats.get("n_trades", 0)
            tag = "  [大盘过滤]" if mkt else ""
            dpg.set_value("bt_status", f"回测完成,共 {n} 笔交易{tag}")
        except Exception as e:
            dpg.set_value("bt_status", f"回测失败: {e}")
        finally:
            dpg.configure_item("bt_run_btn", enabled=True)
            dpg.configure_item("bt_opt_btn", enabled=True)
            dpg.configure_item("bt_psearch_btn", enabled=True)

    threading.Thread(target=worker, daemon=True).start()


def on_run_optimize():
    """后台线程:对当前策略做卖出参数网格调优,弹窗展示 Top 组合。"""
    strat = STATE["current_strategy"]
    if strat is None:
        return
    cls = type(strat)
    start = dpg.get_value("bt_start").strip() or None
    mkt = bool(dpg.get_value("bt_market"))

    def worker():
        dpg.configure_item("bt_run_btn", enabled=False)
        dpg.configure_item("bt_opt_btn", enabled=False)
        dpg.configure_item("bt_psearch_btn", enabled=False)
        dpg.set_value("bt_status", "网格调优中... (27组参数,每组一次回测,较慢请稍候)")
        try:
            grid = optimizer.grid_search(
                cls, start_from=start, market_filter=mkt, metric="total_return",
                progress_cb=lambda d, t: dpg.set_value(
                    "bt_status", f"网格调优 {d}/{t} 组..."),
            )
            _render_optimize(cls.name, grid)
            if not grid.empty:
                best = grid.iloc[0]
                dpg.set_value(
                    "bt_status",
                    f"调优完成 最优: 止盈{best['take_profit']*100:.0f}% "
                    f"止损{best['stop_loss']*100:.0f}% 持有{int(best['hold_days'])}天 "
                    f"总收益{best['total_return']:.0f}%")
            else:
                dpg.set_value("bt_status", "调优完成,但无有效结果")
        except Exception as e:
            dpg.set_value("bt_status", f"调优失败: {e}")
        finally:
            dpg.configure_item("bt_run_btn", enabled=True)
            dpg.configure_item("bt_opt_btn", enabled=True)
            dpg.configure_item("bt_psearch_btn", enabled=True)

    threading.Thread(target=worker, daemon=True).start()


def _render_optimize(name, grid):
    """把网格调优结果渲染成弹窗表格(Top 组合)。"""
    if dpg.does_item_exist("opt_window"):
        dpg.delete_item("opt_window")
    with dpg.window(label=f"卖出参数调优 - {name}", tag="opt_window",
                    width=680, height=480, pos=(260, 120), modal=False):
        dpg.add_text("按总收益排序,取前若干组合。第一行即该策略在本样本上的最优卖出参数。",
                     wrap=640, color=(160, 200, 255))
        dpg.add_text("提示:参数是对历史样本的最优拟合,实盘请留出安全边际。",
                     wrap=640, color=(200, 160, 120))
        dpg.add_separator()
        if grid is None or grid.empty:
            dpg.add_text("无有效结果", color=(255, 180, 120))
            return
        with dpg.table(header_row=True, resizable=True,
                       policy=dpg.mvTable_SizingStretchProp,
                       height=-1, scrollY=True):
            for col in ["排名", "止盈%", "止损%", "持有天", "交易数",
                        "胜率%", "总收益%", "年化%", "回撤%", "盈亏比"]:
                dpg.add_table_column(label=col)
            for i, (_, r) in enumerate(grid.head(15).iterrows()):
                with dpg.table_row():
                    dpg.add_text(str(i + 1))
                    dpg.add_text(f"{r['take_profit']*100:.0f}")
                    dpg.add_text(f"{r['stop_loss']*100:.0f}")
                    dpg.add_text(f"{int(r['hold_days'])}")
                    dpg.add_text(f"{int(r['n_trades'])}")
                    dpg.add_text(f"{r['win_rate']:.1f}")
                    tr = r['total_return']
                    dpg.add_text(f"{tr:.1f}", color=RED if tr > 0 else GREEN)
                    dpg.add_text(f"{r['cagr']:.1f}")
                    dpg.add_text(f"{r['max_dd']:.1f}", color=GREEN)
                    dpg.add_text(f"{r['profit_factor']:.2f}")


def on_run_param_search():
    """后台线程:遍历当前策略【买入参数】组合,弹窗展示最优组合。

    卖出参数取界面上当前的止盈/止损/持有(固定),让比较只反映
    买入信号差异。回测范围沿用回测面板的"全池/仅当前选股结果"。
    """
    strat = STATE["current_strategy"]
    if strat is None:
        return
    cls = type(strat)
    hold = int(dpg.get_value("bt_hold"))
    maxpos = int(dpg.get_value("bt_maxpos"))
    tp = float(dpg.get_value("bt_tp")) / 100.0
    sl = float(dpg.get_value("bt_sl")) / 100.0
    start = dpg.get_value("bt_start").strip() or None
    mkt = bool(dpg.get_value("bt_market"))
    scope = dpg.get_value("bt_scope")

    codes = None
    if scope.startswith("仅当前") and STATE["results"] is not None \
            and not STATE["results"].empty:
        codes = STATE["results"]["code"].tolist()

    btns = ("bt_run_btn", "bt_opt_btn", "bt_psearch_btn")

    def worker():
        for b in btns:
            dpg.configure_item(b, enabled=False)
        dpg.set_value("bt_status", "买入参数寻优中... (遍历参数组合,每组一次回测,较慢请稍候)")
        try:
            grid, keys = optimizer.param_search(
                cls, codes=codes, hold_days=hold, take_profit=tp, stop_loss=sl,
                max_positions=maxpos, start_from=start, market_filter=mkt,
                metric="total_return",
                progress_cb=lambda d, t: dpg.set_value(
                    "bt_status", f"买入参数寻优 {d}/{t} 组..."),
            )
            _render_param_search(cls.name, grid, keys)
            if grid is not None and not grid.empty:
                best = grid.iloc[0]
                labels = grid.attrs.get("labels", {})
                parts = [f"{labels.get(k, k)}={_pfmt(best[k])}" for k in keys]
                dpg.set_value(
                    "bt_status",
                    "寻优完成 最优买入参数: " + ", ".join(parts) +
                    f" | 总收益{best['total_return']:.0f}%")
            else:
                dpg.set_value("bt_status", "寻优完成:该策略无可调买入参数或无有效结果")
        except Exception as e:
            dpg.set_value("bt_status", f"买入参数寻优失败: {e}")
        finally:
            for b in btns:
                dpg.configure_item(b, enabled=True)

    threading.Thread(target=worker, daemon=True).start()


def _pfmt(v):
    """参数值显示:整数不带小数,浮点保留两位。"""
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:.2f}"
    except (TypeError, ValueError):
        return str(v)


def _render_param_search(name, grid, keys):
    """把买入参数寻优结果渲染成弹窗表格(Top 组合)。

    参数列是动态的(取决于策略声明了哪些买入参数),后接固定的回测指标列。
    """
    if dpg.does_item_exist("psearch_window"):
        dpg.delete_item("psearch_window")
    with dpg.window(label=f"买入参数寻优 - {name}", tag="psearch_window",
                    width=720, height=480, pos=(240, 110), modal=False):
        dpg.add_text("遍历该策略声明的关键买入参数组合,按总收益排序。第一行即样本上的最优买入参数。",
                     wrap=680, color=(160, 200, 255))
        dpg.add_text("卖出参数固定为回测面板当前值,所以差异只来自买入信号。参数为历史最优拟合,实盘留安全边际。",
                     wrap=680, color=(200, 160, 120))
        dpg.add_separator()
        if grid is None or grid.empty:
            dpg.add_text("该策略无可调买入参数,或无有效结果。", color=(255, 180, 120))
            return
        labels = grid.attrs.get("labels", {})
        param_cols = [labels.get(k, k) for k in keys]
        metric_cols = ["交易数", "胜率%", "总收益%", "年化%", "回撤%", "盈亏比"]
        with dpg.table(header_row=True, resizable=True,
                       policy=dpg.mvTable_SizingStretchProp,
                       height=-1, scrollY=True):
            dpg.add_table_column(label="排名")
            for c in param_cols:
                dpg.add_table_column(label=c)
            for c in metric_cols:
                dpg.add_table_column(label=c)
            for i, (_, r) in enumerate(grid.head(20).iterrows()):
                with dpg.table_row():
                    if i == 0:
                        dpg.add_text("1", color=(255, 215, 0))
                    else:
                        dpg.add_text(str(i + 1))
                    for k in keys:
                        dpg.add_text(_pfmt(r[k]))
                    dpg.add_text(f"{int(r['n_trades'])}")
                    dpg.add_text(f"{r['win_rate']:.1f}")
                    tr = r['total_return']
                    dpg.add_text(f"{tr:.1f}", color=RED if tr > 0 else GREEN)
                    dpg.add_text(f"{r['cagr']:.1f}")
                    dpg.add_text(f"{r['max_dd']:.1f}", color=GREEN)
                    dpg.add_text(f"{r['profit_factor']:.2f}")
    """在指标区加一个小卡片。good=True 红色(盈利),False 绿色(亏损),None 中性。"""
    if good is True:
        color = RED
    elif good is False:
        color = GREEN
    else:
        color = (200, 200, 200)
    with dpg.group(parent=parent):
        dpg.add_text(label, color=(150, 150, 150))
        dpg.add_text(value, color=color)


def _render_backtest(res):
    """渲染回测结果:指标卡片 + 资金曲线 + 交易明细。"""
    s = res.stats
    dpg.delete_item("bt_stats_area", children_only=True)
    if s.get("n_trades", 0) == 0:
        dpg.add_text(s.get("note", "无交易"), parent="bt_stats_area",
                     wrap=840, color=(255, 180, 120))
    else:
        row1 = dpg.add_group(horizontal=True, parent="bt_stats_area")
        _stat_card(row1, "交易笔数", str(s["n_trades"]))
        _stat_card(row1, "胜率", f"{s['win_rate']}%", good=s["win_rate"] >= 50)
        _stat_card(row1, "总收益", f"{s['total_return']}%", good=s["total_return"] > 0)
        _stat_card(row1, "年化", f"{s['cagr']}%", good=s["cagr"] > 0)
        _stat_card(row1, "最大回撤", f"{s['max_dd']}%", good=False)
        row2 = dpg.add_group(horizontal=True, parent="bt_stats_area")
        _stat_card(row2, "平均每笔", f"{s['avg_ret']}%", good=s["avg_ret"] > 0)
        _stat_card(row2, "平均盈利", f"{s['avg_win']}%", good=True)
        _stat_card(row2, "平均亏损", f"{s['avg_loss']}%", good=False)
        _stat_card(row2, "盈亏比", f"{s['profit_factor']}")
        _stat_card(row2, "平均持有", f"{s['avg_hold']}天")

    if dpg.does_item_exist("bt_equity_series"):
        dpg.delete_item("bt_equity_series")
    if not res.equity.empty:
        eq = res.equity["equity"].astype(float).tolist()
        xs = list(range(len(eq)))
        dpg.add_line_series(xs, eq, label="账户权益", parent="bt_eqy",
                            tag="bt_equity_series")
        base = [100000.0] * len(eq)
        if dpg.does_item_exist("bt_base_series"):
            dpg.delete_item("bt_base_series")
        dpg.add_line_series(xs, base, label="初始10万", parent="bt_eqy",
                            tag="bt_base_series")
        dpg.fit_axis_data("bt_eqx")
        dpg.fit_axis_data("bt_eqy")

    dpg.delete_item("bt_trades_table", children_only=True)
    for col in ["代码", "名称", "买入日", "卖出日", "买入价", "卖出价", "持有", "收益%", "原因"]:
        dpg.add_table_column(label=col, parent="bt_trades_table")
    if not res.trades.empty:
        recent = res.trades.sort_values("sell_date", ascending=False).head(50)
        for _, r in recent.iterrows():
            with dpg.table_row(parent="bt_trades_table"):
                dpg.add_text(r["code"])
                dpg.add_text(str(r["name"]))
                dpg.add_text(r["buy_date"])
                dpg.add_text(r["sell_date"])
                dpg.add_text(str(r["buy_price"]))
                dpg.add_text(str(r["sell_price"]))
                dpg.add_text(str(r["hold"]))
                ret = r["return_pct"]
                dpg.add_text(f"{ret}", color=RED if ret > 0 else GREEN)
                dpg.add_text(str(r["reason"]))


# ---------- 选股结果表 ----------
def _render_results(df):
    """把选股结果渲染到表格。第一列代码可点看K线,含行业+估值列,末列"加自选"按钮。"""
    dpg.delete_item("result_table", children_only=True)
    for col in ["代码", "名称", "行业", "现价", "PE", "PB", "ROE%", "市值亿", "得分", "说明", "操作"]:
        dpg.add_table_column(label=col, parent="result_table")
    if df is None or df.empty:
        return
    imap = db.load_industry_map()
    fmap = db.load_fundamental_map()

    def _fnum(v, d=1):
        try:
            if v is None or v != v:
                return "-"
            return f"{float(v):.{d}f}"
        except Exception:
            return "-"

    for _, r in df.iterrows():
        code = r["code"]
        fd = fmap.get(code, {})
        with dpg.table_row(parent="result_table"):
            _sel = dpg.add_selectable(
                label=code, span_columns=False, user_data=code,
                callback=_on_code_click,   # 双击代码 → 跳K线页
            )
            dpg.add_text(r["name"])
            dpg.add_text(imap.get(code, "-"))
            dpg.add_text(str(r["close"]))
            dpg.add_text(_fnum(fd.get("pe_ttm")))
            dpg.add_text(_fnum(fd.get("pb"), 2))
            dpg.add_text(_fnum(fd.get("roe")))
            dpg.add_text(_fnum(fd.get("total_mv"), 0))
            dpg.add_text(str(r["score"]))
            dpg.add_text(r["reason"])
            dpg.add_button(
                label="+自选", width=60,
                callback=lambda s, a, u: _add_watch_from(u),
                user_data=(code, r["name"], float(r["close"]), str(r["reason"])),
            )


def _add_watch_from(u):
    """把某只票加入自选(买入价记为当前现价,备注记来源)。"""
    code, name, close, note = u
    db.add_watch(code, name, buy_price=close, note=note[:40])
    _set_status(f"已加入自选: {code} {name} @ {close}")
    _refresh_watchlist()


def _add_watch_code(code, note="搜索添加"):
    """仅凭代码加入自选:名称取本地登记名,买入价取本地日线最新收盘价。"""
    name = db.name_of(code) or code
    close = 0.0
    try:
        d = db.load_kline(code)
        if d is not None and len(d):
            close = float(d["close"].iloc[-1])
    except Exception:  # noqa
        pass
    db.add_watch(code, name, buy_price=close, note=note)
    _set_status(f"已加入自选: {code} {name} @ {close}")
    _refresh_watchlist()


# ---------- ETF 榜单 ----------
def on_refresh_etf():
    """后台线程:拉取 ETF 实时快照并渲染榜单。"""
    def worker():
        dpg.configure_item("etf_run_btn", enabled=False)
        dpg.set_value("etf_status", "正在拉取 ETF 实时行情...")
        try:
            df = fetcher.fetch_etf_spot(only_registered=True)
            if df is None or df.empty:
                dpg.set_value("etf_status",
                              "未获取到 ETF 行情(请先在左侧勾选『同时拉取ETF』更新数据)")
                STATE["etf_spot"] = None
            else:
                STATE["etf_spot"] = df
                stale = (df["price"].fillna(0) <= 0).all()
                tip = " (盘前/休市:最新价可能为0)" if stale else ""
                dpg.set_value("etf_status", f"已加载 {len(df)} 只 ETF{tip}")
                _render_etf()
        except Exception as e:
            dpg.set_value("etf_status", f"刷新失败: {e}")
        finally:
            dpg.configure_item("etf_run_btn", enabled=True)

    threading.Thread(target=worker, daemon=True).start()


def _render_etf():
    """渲染 ETF 榜单表(按下拉框选择的字段排序)。"""
    df = STATE.get("etf_spot")
    if not dpg.does_item_exist("etf_table"):
        return
    for ch in dpg.get_item_children("etf_table", 1) or []:
        dpg.delete_item(ch)
    if df is None or df.empty:
        return
    sort_by = dpg.get_value("etf_sort") if dpg.does_item_exist("etf_sort") else "按涨跌幅"
    col = {"按涨跌幅": "chg_pct", "按成交额": "amount", "按最新价": "price"}.get(
        sort_by, "chg_pct")
    d = df.copy()
    if col in d.columns:
        d = d.sort_values(col, ascending=False, na_position="last").reset_index(drop=True)
    for r in d.itertuples(index=False):
        code = r.code
        name = getattr(r, "name", "")
        price = getattr(r, "price", None)
        chg = getattr(r, "chg_pct", None)
        with dpg.table_row(parent="etf_table"):
            _sel = dpg.add_selectable(label=code, span_columns=False, user_data=code,
                                      callback=_on_code_click)   # 双击代码 → 跳K线页
            dpg.add_text(str(name))
            dpg.add_text(f"{price:.3f}" if price and price > 0 else "-")
            if chg is not None and not pd.isna(chg):
                dpg.add_text(f"{chg:+.2f}",
                             color=RED if chg > 0 else (GREEN if chg < 0 else (150, 150, 150)))
            else:
                dpg.add_text("-", color=(150, 150, 150))
            dpg.add_text(f"{getattr(r, 'open', 0) or 0:.3f}")
            dpg.add_text(f"{getattr(r, 'high', 0) or 0:.3f}")
            dpg.add_text(f"{getattr(r, 'low', 0) or 0:.3f}")
            amt = getattr(r, "amount", None)
            dpg.add_text(_fmt_amount(amt) if amt else "-")
            cur = float(price) if price and price > 0 else 0.0
            dpg.add_button(label="+自选", small=True,
                           callback=lambda s, a, u: _add_watch_from(u),
                           user_data=(code, name, cur, "ETF榜单"))


def _fmt_amount(v):
    """成交额格式化:亿/万。"""
    try:
        v = float(v)
    except Exception:
        return "-"
    if v >= 1e8:
        return f"{v / 1e8:.2f}亿"
    if v >= 1e4:
        return f"{v / 1e4:.0f}万"
    return f"{v:.0f}"


# ---------- 今日推荐 / 策略排行榜 ----------
def on_run_ranking():
    """后台线程:只跑【策略排行榜】(8策略回测,较慢),结果落盘缓存。
    排行基于固定历史区间,一天内不变,算一次即可反复给推荐复用。"""
    if not db.list_cached_codes():
        dpg.set_value("rank_status", "本地无数据,请先更新数据")
        return

    def worker():
        dpg.configure_item("rank_run_btn", enabled=False)
        dpg.configure_item("picks_run_btn", enabled=False)
        try:
            mkt = bool(dpg.get_value("rank_market"))
            dpg.set_value("rank_status", "正在评估各策略历史表现(回测排名,较慢)...")
            rank_df = ranking.rank_strategies(
                market_filter=mkt,
                progress_cb=lambda d, t, nm: dpg.set_value(
                    "rank_status", f"回测策略 {d}/{t}: {nm}"),
            )
            STATE["rank_df"] = rank_df
            ranking.save_rank_cache(rank_df, mkt)   # 落盘,供今日推荐复用
            _render_ranking(rank_df)
            cache = ranking.load_rank_cache() or {}
            dpg.set_value(
                "rank_status",
                f"排行已更新({cache.get('computed_at', '')}) · "
                f"共 {0 if rank_df is None else len(rank_df)} 个策略 · "
                f"现在可点『生成今日推荐』(秒出)")
        except Exception as e:
            dpg.set_value("rank_status", f"排行失败: {e}")
        finally:
            dpg.configure_item("rank_run_btn", enabled=True)
            dpg.configure_item("picks_run_btn", enabled=True)

    threading.Thread(target=worker, daemon=True).start()


def on_gen_picks():
    """后台线程:只【生成今日推荐】。直接复用已缓存的策略排行(不再回测),
    仅对最强的几个策略跑一次当日选股,秒级出结果。"""
    if not db.list_cached_codes():
        dpg.set_value("rank_status", "本地无数据,请先更新数据")
        return
    # 优先用内存里的排行,没有再读磁盘缓存
    rank_df = STATE.get("rank_df")
    cache = ranking.load_rank_cache()
    if (rank_df is None or rank_df.empty) and cache:
        rank_df = cache["rank_df"]
        STATE["rank_df"] = rank_df
        _render_ranking(rank_df)
    if rank_df is None or rank_df.empty:
        dpg.set_value(
            "rank_status",
            "还没有策略排行,请先点『刷新策略排行』(只需算一次,之后推荐秒出)")
        return

    def worker():
        dpg.configure_item("rank_run_btn", enabled=False)
        dpg.configure_item("picks_run_btn", enabled=False)
        try:
            tip = ""
            if cache and cache.get("computed_at"):
                tip = f"(排行算于 {cache['computed_at']}) "
            dpg.set_value("rank_status", f"正在用最强策略生成今日推荐票...{tip}")
            picks = ranking.today_picks(
                rank_df, top_strategies=3, per_strategy=10,
                progress_cb=lambda d, t, nm: dpg.set_value(
                    "rank_status", f"选股 {d}/{t}: {nm}"))
            STATE["picks_df"] = picks
            _render_picks(picks)
            n = 0 if picks is None or picks.empty else len(picks)
            dpg.set_value("rank_status", f"完成 · 今日推荐 {n} 只 {tip}")
        except Exception as e:
            dpg.set_value("rank_status", f"生成推荐失败: {e}")
        finally:
            dpg.configure_item("rank_run_btn", enabled=True)
            dpg.configure_item("picks_run_btn", enabled=True)

    threading.Thread(target=worker, daemon=True).start()


def _load_cached_ranking():
    """启动时读取磁盘排行缓存,若有则渲染排行表并提示可直接生成推荐。"""
    if not dpg.does_item_exist("rank_table"):
        return
    cache = ranking.load_rank_cache()
    if not cache:
        return
    rank_df = cache["rank_df"]
    STATE["rank_df"] = rank_df
    _render_ranking(rank_df)
    dpg.set_value(
        "rank_status",
        f"已载入上次排行(算于 {cache.get('computed_at', '?')}"
        f",数据日 {cache.get('data_date', '?')}) · "
        f"可直接点『生成今日推荐』;数据更新后建议重刷排行")


def _render_ranking(df):
    """渲染策略排行榜表。"""
    dpg.delete_item("rank_table", children_only=True)
    for col in ["排名", "策略", "综合分", "年化%", "总收益%", "胜率%", "回撤%", "盈亏比", "交易数"]:
        dpg.add_table_column(label=col, parent="rank_table")
    if df is None or df.empty:
        return
    for _, r in df.iterrows():
        with dpg.table_row(parent="rank_table"):
            dpg.add_text(str(int(r["rank"])))
            dpg.add_text(str(r["name"]))
            dpg.add_text(str(r["score"]))
            cg = r["cagr"]
            dpg.add_text(f"{cg}", color=RED if cg > 0 else GREEN)
            tr = r["total_return"]
            dpg.add_text(f"{tr}", color=RED if tr > 0 else GREEN)
            dpg.add_text(f"{r['win_rate']}")
            dpg.add_text(f"{r['max_dd']}", color=GREEN)
            dpg.add_text(f"{r['profit_factor']}")
            dpg.add_text(str(int(r["n_trades"])))


def _render_picks(df):
    """渲染今日推荐票表(可点看K线、加自选),含行业列。"""
    dpg.delete_item("picks_table", children_only=True)
    for col in ["代码", "名称", "行业", "现价", "来源策略", "策略排名", "命中", "得分", "操作"]:
        dpg.add_table_column(label=col, parent="picks_table")
    if df is None or df.empty:
        _render_hot_sectors(None)
        return
    imap = db.load_industry_map()
    for _, r in df.iterrows():
        code = r["code"]
        with dpg.table_row(parent="picks_table"):
            _sel = dpg.add_selectable(
                label=code, span_columns=False, user_data=code,
                callback=_on_code_click,   # 双击代码 → 跳K线页
            )
            dpg.add_text(str(r["name"]))
            dpg.add_text(imap.get(code, "-"))
            dpg.add_text(str(r["close"]))
            dpg.add_text(str(r["strategy"]))
            dpg.add_text(str(int(r["strat_rank"])))
            dpg.add_text(str(int(r["hits"])))
            dpg.add_text(str(r["score"]))
            dpg.add_button(
                label="+自选", width=60,
                callback=lambda s, a, u: _add_watch_from(u),
                user_data=(code, r["name"], float(r["close"]), f"来自{r['strategy']}"),
            )
    # 统计热门板块:推荐票按行业聚合,数量降序
    _render_hot_sectors(df)


def _render_hot_sectors(df):
    """统计今日推荐票的行业分布,展示上榜数量最多的热门板块。"""
    if not dpg.does_item_exist("hot_sectors"):
        return
    dpg.delete_item("hot_sectors", children_only=True)
    if df is None or df.empty:
        dpg.add_text("(暂无推荐,生成后显示热门板块)", parent="hot_sectors",
                     color=(130, 130, 130))
        return
    imap = db.load_industry_map()
    from collections import Counter
    cnt = Counter(imap.get(c, "未分类") for c in df["code"])
    top = cnt.most_common(8)
    if not top:
        dpg.add_text("(无行业数据,请先更新数据补齐行业)", parent="hot_sectors",
                     color=(130, 130, 130))
        return
    with dpg.group(horizontal=True, parent="hot_sectors"):
        for name, c in top:
            dpg.add_text(f"[{name} x{c}]", color=RED if c >= 2 else (200, 200, 200))


# ---------- 自选持仓 / 实时盯盘 ----------
def _refresh_watchlist(use_realtime=False):
    """
    刷新自选/持仓表,计算浮动盈亏。
    use_realtime=False: 用本地最新收盘价(默认,离线可用)。
    use_realtime=True : 先拉一次全市场实时快照,用实时价算盈亏并显示涨跌幅。
    """
    if not dpg.does_item_exist("watch_table"):
        return
    cols = ["代码", "名称", "加入日", "买入价", "现价", "今日涨跌%", "浮动盈亏%", "备注", "操作"]
    dpg.delete_item("watch_table", children_only=True)
    for col in cols:
        dpg.add_table_column(label=col, parent="watch_table")
    wl = db.load_watchlist()
    if wl is None or wl.empty:
        dpg.set_value("watch_status", "自选为空。可在选股结果或今日推荐里点『+自选』加入。")
        return

    spot = STATE.get("spot") if use_realtime else None
    # 判断快照是否"陈旧"(盘前/午休/停牌:最新价为0,已用昨收兜底)。
    # 只要有任一票带 stale 标记,就说明当前非连续竞价时段。
    is_stale = bool(spot) and any(
        (v or {}).get("stale") for v in spot.values()) if spot else False
    if use_realtime and spot:
        src_txt = "昨收(未开盘/休市)" if is_stale else "实时"
    else:
        src_txt = "本地收盘"
    total_pnl = []
    for _, r in wl.iterrows():
        code = r["code"]
        q = spot.get(code) if spot else None
        cur = None
        chg = None
        if q and q.get("price") is not None:
            cur = float(q["price"])
            chg = q.get("chg_pct")  # stale(盘前/停牌)时为 None
        else:
            kl = db.load_kline(code)
            if kl is not None and not kl.empty:
                cur = float(kl.iloc[-1]["close"])
        buy = float(r["buy_price"] or 0.0)
        # 现价有效(>0)且有买入价才算浮动盈亏,否则留空,避免拉不到价时误显示 -100%
        has_cur = cur is not None and cur > 0
        pnl = ((cur - buy) / buy * 100) if (has_cur and buy > 0) else None
        if pnl is not None:
            total_pnl.append(pnl)
        with dpg.table_row(parent="watch_table"):
            _sel = dpg.add_selectable(label=code, span_columns=False, user_data=code,
                                      callback=_on_code_click)   # 双击代码 → 跳K线页
            dpg.add_text(str(r["name"]))
            dpg.add_text(str(r["add_date"]))
            dpg.add_text(f"{buy:.2f}" if buy > 0 else "-")
            if has_cur:
                dpg.add_text(f"{cur:.2f}")
            else:
                dpg.add_text("-", color=(150, 150, 150))
            if chg is not None:
                dpg.add_text(f"{chg:+.2f}", color=RED if chg > 0 else (GREEN if chg < 0 else (150, 150, 150)))
            else:
                dpg.add_text("-", color=(150, 150, 150))
            if pnl is not None:
                dpg.add_text(f"{pnl:+.2f}", color=RED if pnl > 0 else GREEN)
            elif buy > 0:
                dpg.add_text("待行情", color=(150, 150, 150))
            else:
                dpg.add_text("观察", color=(150, 150, 150))
            dpg.add_text(str(r["note"]))
            dpg.add_button(label="移除", width=60,
                           callback=lambda s, a, u: _remove_watch(u), user_data=code)
    if total_pnl:
        avg = sum(total_pnl) / len(total_pnl)
        dpg.set_value("watch_status",
                      f"持仓 {len(total_pnl)} 只 · 平均浮动盈亏 {avg:+.2f}% · 价格源:{src_txt}")
    else:
        dpg.set_value("watch_status", f"自选 {len(wl)} 只(均为观察) · 价格源:{src_txt}")


def on_refresh_realtime():
    """后台线程:拉全市场实时快照 → 刷新自选盈亏 + 检查预警。"""
    def worker():
        dpg.set_value("watch_status", "正在拉取实时行情(全市场快照,约10-15s)...")
        spot = realtime.fetch_spot()
        STATE["spot"] = spot
        if not spot:
            dpg.set_value("watch_status", "实时行情拉取失败(网络/非交易时段),请稍后重试")
            return
        _refresh_watchlist(use_realtime=True)
        # 检查预警
        fired = realtime.check_alerts(spot)
        _render_alerts(fired)
        if fired:
            _set_status(f"⚠ {len(fired)} 条预警触发! 见自选持仓页")
    threading.Thread(target=worker, daemon=True).start()


def _remove_watch(code):
    db.remove_watch(code)
    _set_status(f"已移除自选: {code}")
    _refresh_watchlist()


# ---------- 价格预警 ----------
def on_add_alert():
    """根据输入框新增一条价格预警。"""
    code = dpg.get_value("alert_code").strip().zfill(6) if dpg.get_value("alert_code").strip() else ""
    if not code:
        dpg.set_value("alert_status", "请填写股票代码(6位)")
        return

    def _v(tag):
        try:
            v = float(dpg.get_value(tag))
            return v if v != 0 else None
        except Exception:
            return None

    nm = ""
    sl = db.load_stock_list()
    if sl is not None and not sl.empty:
        hit = sl[sl["code"] == code]
        if not hit.empty:
            nm = hit.iloc[0]["name"]
    db.save_alert(code, nm,
                  price_low=_v("alert_plow"), price_high=_v("alert_phigh"),
                  chg_low=_v("alert_clow"), chg_high=_v("alert_chigh"),
                  note=dpg.get_value("alert_note").strip())
    dpg.set_value("alert_status", f"已添加预警: {code} {nm}")
    _refresh_alerts_table()


def _refresh_alerts_table():
    """刷新预警设置列表。"""
    if not dpg.does_item_exist("alerts_table"):
        return
    dpg.delete_item("alerts_table", children_only=True)
    for col in ["代码", "名称", "价≤", "价≥", "跌幅≤%", "涨幅≥%", "备注", "操作"]:
        dpg.add_table_column(label=col, parent="alerts_table")
    al = db.load_alerts()
    if al is None or al.empty:
        return
    def _s(v):
        return f"{v:g}" if v is not None and v == v and v != 0 else "-"
    for _, r in al.iterrows():
        with dpg.table_row(parent="alerts_table"):
            dpg.add_text(r["code"])
            dpg.add_text(str(r["name"]))
            dpg.add_text(_s(r["price_low"]))
            dpg.add_text(_s(r["price_high"]))
            dpg.add_text(_s(r["chg_low"]))
            dpg.add_text(_s(r["chg_high"]))
            dpg.add_text(str(r["note"]))
            dpg.add_button(label="删除", width=56,
                           callback=lambda s, a, u: _remove_alert(u), user_data=r["code"])


def _remove_alert(code):
    db.remove_alert(code)
    dpg.set_value("alert_status", f"已删除预警: {code}")
    _refresh_alerts_table()


def _render_alerts(fired):
    """把已触发的预警渲染到提示区(涨红跌绿)。"""
    if not dpg.does_item_exist("alert_fired"):
        return
    dpg.delete_item("alert_fired", children_only=True)
    if not fired:
        dpg.add_text("(暂无触发的预警;点『刷新实时行情』检查)", parent="alert_fired",
                     color=(130, 130, 130))
        return
    for f in fired:
        chg = f.get("chg_pct")
        color = RED if (chg is not None and chg > 0) else GREEN
        txt = f"⚠ {f['code']} {f['name']}  现价{f['price']:g}"
        if chg is not None:
            txt += f"  {chg:+.2f}%"
        txt += "  |  " + "; ".join(f["reasons"])
        dpg.add_text(txt, parent="alert_fired", color=color, wrap=1000)


# ---------- K 线绘制 ----------
def _to_ts(d):
    """日期字符串 -> Unix 时间戳(candle/time 轴需要)。"""
    return _time.mktime(datetime.strptime(d, "%Y-%m-%d").timetuple())


def _line_xy(xs, series):
    """把 (时间戳, 值) 配对并过滤 NaN,返回两个列表。"""
    xy = [(x, v) for x, v in zip(xs, series.tolist()) if v == v]
    if not xy:
        return [], []
    px, py = zip(*xy)
    return list(px), list(py)


def _sync_period_buttons(period):
    """高亮当前选中的周期按钮(其余恢复默认)。"""
    mapping = {"D": "pd_day", "W": "pd_week", "M": "pd_month", "MIN": "pd_min"}
    for p, tag in mapping.items():
        if not dpg.does_item_exist(tag):
            continue
        if p == period:
            dpg.bind_item_theme(tag, "period_on")
        else:
            dpg.bind_item_theme(tag, 0)


def _switch_period(period):
    """周期切换按钮回调。切到非分时时停轮询;切到分时时开轮询(只拉当前票)。"""
    code = STATE.get("cur_code")
    if not code:
        _set_status("请先在结果表点一只股票")
        return
    if period != "MIN":
        _stop_poll()
    show_kline(code, period)
    if period == "MIN":
        _start_poll(code)


def _draw_intraday(code, incremental=False):
    """
    画当日分时曲线:价格线(涨红跌绿相对昨收) + 均价线(黄) + 成交量。
    X 轴用 HH:MM 时间刻度。incremental=True 时(轮询)只更新已有 series 数据,
    不删除重建 —— 避免闪烁,并保留用户的缩放/拖动状态。
    """
    try:
        df = fetcher.fetch_intraday(code)
    except Exception as e:  # noqa
        _set_status(f"分时拉取失败: {e}")
        df = None
    nm = db.name_of(code)
    label = f"{code} {nm}".strip() if nm else code
    if df is None or df.empty:
        if not incremental:
            dpg.delete_item("chart_area", children_only=True)
            with dpg.group(parent="chart_area"):
                dpg.add_text("暂无分时数据(可能盘前未开盘或该标的当日无成交)",
                             color=(255, 200, 100))
            dpg.set_value("kline_title", f"分时 - {label}")
            STATE["intraday_code"] = None
        return

    # 昨收:用日线最后一根收盘价作参考(判断涨跌上色)
    prev_close = None
    kl = db.load_kline(code)
    if kl is not None and not kl.empty:
        prev_close = float(kl.iloc[-1]["close"])

    times = df["time"].tolist()
    xs = list(range(len(df)))
    prices = df["price"].astype(float).tolist()
    avgs = df["avg"].astype(float).tolist()
    vols = (df["volume"].astype(float) / 100).tolist()
    n = len(xs)
    last = prices[-1]
    base = prev_close if prev_close else prices[0]
    up = last >= base

    # X 轴时间刻度:均匀取 ~7 个点,标 HH:MM
    step = max(1, n // 7)
    ticks = [(times[i], xs[i]) for i in range(0, n, step)]
    if ticks and ticks[-1][1] != xs[-1]:
        ticks.append((times[-1], xs[-1]))
    ticks = tuple(ticks)

    can_update = (incremental and STATE.get("intraday_code") == code
                  and dpg.does_item_exist("it_price"))

    if can_update:
        # —— 增量更新:只改数据,不删控件(无闪烁,保留缩放) ——
        dpg.set_value("it_price", [xs, prices])
        dpg.set_value("it_avg", [xs, avgs])
        dpg.set_value("it_vol", [xs, vols])
        if dpg.does_item_exist("it_prev") and prev_close:
            dpg.set_value("it_prev", [[0, n - 1], [prev_close, prev_close]])
        dpg.bind_item_theme("it_price", "line_red" if up else "line_green")
        dpg.bind_item_theme("it_vol", "bar_red" if up else "bar_green")
        dpg.set_axis_ticks("mx", ticks)
        dpg.set_axis_ticks("mvx", ticks)
    else:
        # —— 首次绘制:完整重建 ——
        dpg.delete_item("chart_area", children_only=True)
        with dpg.subplots(2, 1, label="", width=-1, height=-1, parent="chart_area",
                          row_ratios=[3.0, 1.0], link_all_x=True, no_title=True):
            with dpg.plot(label=f"分时 {code}", height=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="mx")
                my = dpg.add_plot_axis(dpg.mvYAxis, label="价格", tag="my")
                if prev_close:
                    dpg.add_line_series([0, n - 1], [prev_close, prev_close],
                                        label="昨收", parent=my, tag="it_prev")
                dpg.add_line_series(xs, prices, label="价格", parent=my, tag="it_price")
                dpg.bind_item_theme("it_price", "line_red" if up else "line_green")
                dpg.add_line_series(xs, avgs, label="均价", parent=my, tag="it_avg")
                dpg.set_axis_ticks("mx", ticks)
                dpg.fit_axis_data("mx")
                dpg.fit_axis_data("my")
            with dpg.plot(label="成交量", height=-1):
                dpg.add_plot_axis(dpg.mvXAxis, tag="mvx")
                mvy = dpg.add_plot_axis(dpg.mvYAxis, label="量(手)", tag="mvy")
                dpg.add_bar_series(xs, vols, parent=mvy, weight=0.7, label="量", tag="it_vol")
                dpg.bind_item_theme("it_vol", "bar_red" if up else "bar_green")
                dpg.set_axis_ticks("mvx", ticks)
                dpg.fit_axis_data("mvx")
                dpg.fit_axis_data("mvy")
        STATE["intraday_code"] = code

    tdate = df.attrs.get("trade_date", "")
    chg = ((last - base) / base * 100) if base else 0.0
    dpg.set_value("kline_title",
                  f"分时 {label}  {tdate}  现价 {last:.3f}  "
                  f"{'+' if chg >= 0 else ''}{chg:.2f}%  "
                  f"({'轮询中' if STATE.get('poll_on') else '已停'})")


def _poll_worker():
    """后台轮询线程:每 POLL_INTERVAL 秒只重绘当前轮询目标的分时。"""
    while STATE.get("poll_on"):
        code = STATE.get("poll_code")
        if code and STATE.get("cur_period") == "MIN":
            try:
                _draw_intraday(code, incremental=True)
            except Exception as e:  # noqa
                print(f"[warn] 轮询重绘失败: {e}")
        for _ in range(POLL_INTERVAL * 2):
            if not STATE.get("poll_on"):
                break
            _time.sleep(0.5)


def _start_poll(code):
    """开启分时轮询,目标为 code。已在跑则只切换目标。"""
    STATE["poll_code"] = code
    if STATE.get("poll_on"):
        return
    STATE["poll_on"] = True
    t = threading.Thread(target=_poll_worker, daemon=True)
    STATE["poll_thread"] = t
    t.start()
    _set_status(f"已开启分时轮询(每{POLL_INTERVAL}秒刷新 {code})")


def _stop_poll():
    """停止分时轮询。"""
    if STATE.get("poll_on"):
        STATE["poll_on"] = False
        _set_status("已停止分时轮询")


def _on_pick_code(code):
    """点结果/榜单/自选表里的代码:跳到「选股 & K线」页并画K线;切换标的时停旧轮询、回到日线。"""
    _stop_poll()
    STATE["intraday_code"] = None   # 换标的:分时图需完整重建
    show_kline(code, "D")           # 先画好K线
    # 再切到「选股 & K线」tab(用 int id 最稳,别名字符串在部分DPG版本 set_value 不接受)
    try:
        if dpg.does_item_exist("main_tabs") and dpg.does_item_exist("tab_kline"):
            dpg.set_value("main_tabs", dpg.get_alias_id("tab_kline"))
    except Exception:
        pass


def _on_code_click(sender, app_data, user_data):
    """榜单/表格里代码单元格的单击回调:用点击时间差判定双击。
    同一 code 在 DBL_MS 毫秒内连点两次 → 视为双击 → 跳K线页看图。
    (DPG 的 selectable 无原生双击,共享 item_handler_registry 又无法区分是哪个
     item 触发[app_data 是鼠标键号而非 item id],故改用时间差方案,最稳。)"""
    code = user_data
    if not code:
        return
    now = _time.monotonic()
    last_code = STATE.get("_last_click_code")
    last_ts = STATE.get("_last_click_ts", 0.0)
    STATE["_last_click_code"] = code
    STATE["_last_click_ts"] = now
    if code == last_code and (now - last_ts) <= 0.40:
        STATE["_last_click_ts"] = 0.0   # 重置,避免三连击误判
        _on_pick_code(code)


def _bind_code_dbl(item):
    """兼容旧调用:时间差双击方案无需绑定,保留空实现避免改动四处调用点。"""
    return


def _on_search_hit(code):
    """点击搜索结果:看K线并收起结果框。"""
    dpg.configure_item("kl_search_box", show=False)
    _on_pick_code(code)


def on_search_kline(sender=None, app_data=None, user_data=None):
    """
    K线搜索回调:在本地已拉取的股票/ETF 中模糊搜索(代码或名称)。
    输入框每次改动即触发(实时搜索);结果以按钮列表展示,点击看K线。
    """
    kw = dpg.get_value("kl_search_in") or ""
    kw = kw.strip()
    # 清空旧结果
    if dpg.does_item_exist("kl_search_box"):
        dpg.delete_item("kl_search_box", children_only=True)
    if not kw:
        dpg.configure_item("kl_search_box", show=False)
        dpg.set_value("kl_search_hint", "")
        return
    try:
        hits = db.search_local(kw, limit=30)
    except Exception as e:  # noqa
        dpg.set_value("kl_search_hint", f"搜索出错: {e}")
        return
    if not hits:
        dpg.configure_item("kl_search_box", show=True)
        dpg.add_text("无匹配(仅在已拉取的股票/ETF中搜索)",
                     color=(200, 160, 100), parent="kl_search_box")
        dpg.set_value("kl_search_hint", "0 条")
        return
    dpg.set_value("kl_search_hint", f"{len(hits)} 条")
    dpg.configure_item("kl_search_box", show=True)
    for h in hits:
        code, name = h["code"], h["name"] or "-"
        tag = "ETF" if h["is_etf"] else "股"
        label = f"[{tag}] {code}  {name}"
        with dpg.group(horizontal=True, parent="kl_search_box"):
            dpg.add_button(label=label, width=-90,
                           user_data=code,
                           callback=lambda s, a, u: _on_search_hit(u))
            dpg.add_button(label="+自选", width=-1,
                           user_data=code,
                           callback=lambda s, a, u: _add_watch_code(u, "搜索添加"))


def show_kline(code, period=None):
    """
    绘制该标的图表。period: D/W/M 画蜡烛图(K线+成交额+MACD),MIN 画当日分时曲线。
    period=None 时沿用当前选择(STATE['cur_period'])。切换标的时默认回到日线。
    涨红跌绿。
    """
    if period is None:
        period = STATE.get("cur_period", "D")
    STATE["cur_code"] = code
    STATE["cur_period"] = period
    # 高亮当前周期按钮
    _sync_period_buttons(period)

    if period == "MIN":
        _draw_intraday(code)
        return

    df = db.load_kline(code)
    if df.empty:
        _set_status(f"{code} 无本地数据")
        return
    # 日线聚合成周/月线
    if period in ("W", "M"):
        df = ind.resample_period(df, period)
    df = ind.enrich(df)
    # 按周期取合理根数,使各周期时间跨度接近(否则同为120根:日线仅半年、周线2.3年、月线10年,对不上)
    # DPG 渲染上千根蜡烛也很流畅,性能非瓶颈;日线放到全量(本地约1335根≈5年),支持滚轮缩放/拖动查看
    tail_n = {"D": 2000, "W": 250, "M": 200}.get(period, 2000)
    df = df.tail(tail_n).reset_index(drop=True)
    pname = {"D": "日线", "W": "周线", "M": "月线"}.get(period, "日线")
    nm = db.name_of(code)
    label = f"{code} {nm}".strip() if nm else code   # 代码+名称,如 "601985 中国核电"

    # X 轴统一用整数序号索引(0,1,2...),而不是时间戳。
    # 好处:相邻点间距恒为 1,蜡烛/柱宽固定,日/周/月三种周期都不会糊墙,
    #       X 轴也不会被 Time scale 撑到未来年份。日期用 set_axis_ticks 手动标注。
    dates = df["date"].tolist()
    xs = list(range(len(df)))
    opens = df["open"].astype(float).tolist()
    closes = df["close"].astype(float).tolist()
    highs = df["high"].astype(float).tolist()
    lows = df["low"].astype(float).tolist()
    up_mask = df["close"] >= df["open"]

    # 相邻点间距恒为 1 → 蜡烛/柱固定宽度即可,无需按时间换算
    # 相邻蜡烛索引间距恒为 1。candle 实体带描边+影线,视觉占宽比 weight 偏大,
    # 取 0.5 留出间隙不重叠;成交额/MACD 柱同宽保持视觉一致。
    cw = 0.5   # 蜡烛宽度(索引单位)
    bw = 0.5   # 成交额/MACD 柱宽度(索引单位)

    # X 轴日期刻度:均匀取 ~8 个点,标 年-月-日 或 年-月
    n = len(xs)
    step = max(1, n // 8)
    fmt = "%y/%m" if period in ("W", "M") else "%m/%d"

    def _fmt_date(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d").strftime(fmt)
        except Exception:  # noqa
            return s

    _ticks = [(_fmt_date(dates[i]), xs[i]) for i in range(0, n, step)]
    if _ticks and _ticks[-1][1] != xs[-1]:
        _ticks.append((_fmt_date(dates[-1]), xs[-1]))
    kticks = tuple(_ticks)

    dpg.delete_item("chart_area", children_only=True)
    with dpg.subplots(
        3, 1, label="", width=-1, height=-1, parent="chart_area",
        row_ratios=[3.0, 1.0, 1.2], link_all_x=True, no_title=True, tag="sp",
    ):
        # 第1层: K线主图 + 均线
        with dpg.plot(label=f"{pname} {label}", no_title=False, height=-1, tag="kplot"):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, tag="kx", no_tick_labels=True)
            ky = dpg.add_plot_axis(dpg.mvYAxis, label="价格", tag="ky")
            dpg.add_candle_series(
                xs, opens, closes, lows, highs, parent=ky, label=code,
                bull_color=RED, bear_color=GREEN, weight=cw,
                tooltip=False,   # 关掉内置提示(会把整数索引X当成时间戳显示 1/1/70)
            )
            for col, lbl in [("ma5", "MA5"), ("ma20", "MA20"), ("ma60", "MA60")]:
                lx, ly = _line_xy(xs, df[col])
                if lx:
                    dpg.add_line_series(lx, ly, label=lbl, parent=ky)
            # 自定义悬停提示(跟随鼠标定位到最近蜡烛),初始隐藏
            dpg.add_plot_annotation(
                tag="kanno", label="", default_value=(0, 0),
                offset=(12, -12), color=(255, 255, 255, 255),
                clamped=True, show=False, parent="kplot",
            )
            dpg.set_axis_ticks("kx", kticks)
            dpg.fit_axis_data("kx")
            dpg.fit_axis_data("ky")

        # 第2层: 成交额
        with dpg.plot(label="成交额", no_title=False, height=-1):
            dpg.add_plot_axis(dpg.mvXAxis, tag="ax2", no_tick_labels=True)
            ay2 = dpg.add_plot_axis(dpg.mvYAxis, label="成交额(亿)", tag="ay2")
            amt_yi = (df["amount"].astype(float) / 1e8)
            rx = [x for x, m in zip(xs, up_mask) if m]
            ry = [v for v, m in zip(amt_yi.tolist(), up_mask) if m]
            if rx:
                bid = dpg.add_bar_series(rx, ry, parent=ay2, weight=bw, label="涨")
                dpg.bind_item_theme(bid, "bar_red")
            gx = [x for x, m in zip(xs, up_mask) if not m]
            gy = [v for v, m in zip(amt_yi.tolist(), up_mask) if not m]
            if gx:
                bid = dpg.add_bar_series(gx, gy, parent=ay2, weight=bw, label="跌")
                dpg.bind_item_theme(bid, "bar_green")
            dpg.set_axis_ticks("ax2", kticks)
            dpg.fit_axis_data("ax2")
            dpg.fit_axis_data("ay2")

        # 第3层: MACD
        with dpg.plot(label="MACD (12,26,9)", no_title=False, height=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, tag="ax3")
            ay3 = dpg.add_plot_axis(dpg.mvYAxis, label="MACD", tag="ay3")
            bar_up = df["macd_bar"] >= 0
            bx = [x for x, m in zip(xs, bar_up) if m]
            by = [v for v, m in zip(df["macd_bar"].tolist(), bar_up) if m]
            if bx:
                bid = dpg.add_bar_series(bx, by, parent=ay3, weight=bw, label="红柱")
                dpg.bind_item_theme(bid, "bar_red")
            nx = [x for x, m in zip(xs, bar_up) if not m]
            ny = [v for v, m in zip(df["macd_bar"].tolist(), bar_up) if not m]
            if nx:
                bid = dpg.add_bar_series(nx, ny, parent=ay3, weight=bw, label="绿柱")
                dpg.bind_item_theme(bid, "bar_green")
            for col, lbl in [("dif", "DIF"), ("dea", "DEA")]:
                lx, ly = _line_xy(xs, df[col])
                if lx:
                    dpg.add_line_series(lx, ly, label=lbl, parent=ay3)
            dpg.set_axis_ticks("ax3", kticks)
            dpg.fit_axis_data("ax3")
            dpg.fit_axis_data("ay3")

    # 缓存本次绘制的K线数据,供鼠标悬停提示回调按索引取值
    STATE["kl_bars"] = {
        "n": n,
        "dates": dates,
        "open": opens, "close": closes, "high": highs, "low": lows,
    }

    dpg.set_value("kline_title", f"{pname} / 成交额 / MACD - {label}")
    _set_status(f"已绘制 {label} 的{pname}图表")


def _on_kline_hover(sender, app_data):
    """鼠标在K线主图移动:定位到最近的蜡烛,更新悬停提示(真实日期+OHLC)。"""
    bars = STATE.get("kl_bars")
    if not bars or not dpg.does_item_exist("kanno") or not dpg.does_item_exist("kplot"):
        return
    # 仅当鼠标悬停在K线主图上才显示
    if not dpg.is_item_hovered("kplot"):
        if dpg.get_item_configuration("kanno").get("show"):
            dpg.configure_item("kanno", show=False)
        return
    try:
        mx, my = dpg.get_plot_mouse_pos()
    except Exception:  # noqa
        return
    n = bars["n"]
    idx = int(round(mx))
    if idx < 0 or idx >= n:
        dpg.configure_item("kanno", show=False)
        return
    o, c = bars["open"][idx], bars["close"][idx]
    h, low = bars["high"][idx], bars["low"][idx]
    d = bars["dates"][idx]
    chg = ((c - o) / o * 100) if o else 0.0
    txt = (f"{d}\n开 {o:.2f}  收 {c:.2f}\n"
           f"高 {h:.2f}  低 {low:.2f}\n涨跌 {'+' if chg >= 0 else ''}{chg:.2f}%")
    # 提示框锚定到该蜡烛索引、Y 跟随鼠标,颜色随涨跌
    col = (230, 90, 90, 255) if c >= o else (90, 200, 120, 255)
    dpg.configure_item("kanno", label=txt, default_value=(idx, my),
                       color=col, show=True)


# ---------- 构建界面 ----------
STRATEGY_LABEL2KEY = {cls.name: key for key, cls in ALL_STRATEGIES.items()}


def build_ui():
    dpg.create_context()
    load_chinese_font()
    db.init_db()
    _make_bar_themes()

    with dpg.window(tag="root"):
        with dpg.group(horizontal=True):
            # ===== 左侧控制面板 =====
            with dpg.child_window(width=320, tag="left_panel"):
                dpg.add_text("A 股选股工具", color=(80, 160, 255))
                dpg.add_separator()

                dpg.add_text("本地数据", color=(120, 220, 160))
                dpg.add_text("", tag="cache_info", wrap=300, color=(160, 200, 255))
                dpg.add_text("", tag="market_info", wrap=300, color=(150, 150, 150))
                dpg.add_separator()

                dpg.add_text("1. 更新数据(可选)")
                dpg.add_input_int(label="拉取只数", default_value=400,
                                  tag="update_count", width=120, min_value=1)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="主流400", width=88,
                                   callback=lambda: dpg.set_value("update_count", 400))
                    dpg.add_button(label="1500", width=88,
                                   callback=lambda: dpg.set_value("update_count", 1500))
                    dpg.add_button(label="全A股", width=88,
                                   callback=lambda: dpg.set_value("update_count", 6000))
                dpg.add_button(label="更新股票数据", callback=on_update_data, width=280)
                dpg.add_checkbox(label="同时拉取基本面(PE/PB/ROE/市值,较慢)",
                                 default_value=False, tag="fetch_funda")
                dpg.add_checkbox(label="同时拉取ETF(主流宽基+行业约50只)",
                                 default_value=False, tag="fetch_etf")
                dpg.add_text("主流股(沪深300+中证500)优先,并发拉取并同步\n"
                             "大盘指数+行业。400只约3-5分钟;全A股(5000+)\n"
                             "首次约30-50分钟,选股/回测也更慢,按需选择。\n"
                             "基本面每只1-2s串行,勾选后首次较慢,增量缓存。",
                             wrap=300, color=(130, 130, 130))
                dpg.add_separator()

                dpg.add_text("2. 选择策略")
                dpg.add_combo(
                    list(STRATEGY_LABEL2KEY.keys()),
                    default_value=list(STRATEGY_LABEL2KEY.keys())[0],
                    callback=on_strategy_change, tag="strategy_combo", width=280,
                )
                dpg.add_child_window(tag="param_area", height=170, border=False)
                dpg.add_separator()

                dpg.add_checkbox(label="基本面过滤(对选股结果二次筛)",
                                 default_value=False, tag="funda_on")
                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="PE≤", default_value=60, min_value=0,
                                         max_value=200, format="%.0f", width=110,
                                         tag="funda_pe")
                    dpg.add_slider_float(label="PB≤", default_value=10, min_value=0,
                                         max_value=30, format="%.1f", width=110,
                                         tag="funda_pb")
                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="ROE%≥", default_value=0, min_value=-20,
                                         max_value=40, format="%.0f", width=110,
                                         tag="funda_roe")
                    dpg.add_slider_float(label="市值亿≥", default_value=0, min_value=0,
                                         max_value=2000, format="%.0f", width=110,
                                         tag="funda_mv")
                dpg.add_checkbox(label="剔除无基本面数据的票", default_value=False,
                                 tag="funda_dropmiss")
                dpg.add_text("阈值为0=该项不启用;需先勾选『同时拉取基本面』\n"
                             "更新过数据。滑到最大值也相当于不限制。",
                             wrap=300, color=(130, 130, 130))
                dpg.add_separator()

                dpg.add_text("3. 执行")
                dpg.add_button(label="开始选股", callback=on_run_scan, width=280, height=36)
                dpg.add_button(label="回测当前策略", callback=on_open_backtest, width=280, height=30)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="导出报告", callback=on_export_report,
                                   width=160, height=30)
                    dpg.add_button(label="打开目录", callback=on_open_report_dir,
                                   width=112, height=30)
                dpg.add_text("导出为 Excel + HTML(含排行榜/今日推荐/选股\n"
                             "结果/自选持仓),存到 reports/ 目录。",
                             wrap=300, color=(130, 130, 130))
                dpg.add_separator()
                dpg.add_text("就绪", tag="status_text", wrap=300, color=(255, 200, 100))

            # ===== 右侧:标签页 =====
            with dpg.child_window(tag="right_panel"):
                with dpg.tab_bar(tag="main_tabs"):
                    # --- Tab 1: 选股 & K线 ---
                    with dpg.tab(label="选股 & K线", tag="tab_kline"):
                        dpg.add_text("选股结果(点代码看K线,点+自选加入持仓跟踪)")
                        with dpg.table(tag="result_table", header_row=True,
                                       resizable=True, policy=dpg.mvTable_SizingStretchProp,
                                       height=210, scrollY=True):
                            for col in ["代码", "名称", "行业", "现价", "PE", "PB",
                                        "ROE%", "市值亿", "得分", "说明", "操作"]:
                                dpg.add_table_column(label=col)
                        dpg.add_separator()
                        # --- 搜索:在本地已拉取的股票/ETF 中模糊搜索(代码或名称) ---
                        with dpg.group(horizontal=True):
                            dpg.add_text("搜索:")
                            dpg.add_input_text(
                                tag="kl_search_in", hint="输入代码或名称(如 600519 / 茅台 / 沪深300)",
                                width=320, callback=on_search_kline,
                                on_enter=False)
                            dpg.add_button(label="搜索", width=54, height=26,
                                           callback=on_search_kline)
                            dpg.add_button(label="清除", width=54, height=26,
                                           callback=lambda: (dpg.set_value("kl_search_in", ""),
                                                             on_search_kline()))
                            dpg.add_text("", tag="kl_search_hint", color=(130, 130, 130))
                        # 搜索结果区(默认隐藏,有结果时显示按钮列表,点击看K线)
                        dpg.add_child_window(tag="kl_search_box", height=118,
                                             border=True, show=False)
                        dpg.add_separator()
                        with dpg.group(horizontal=True):
                            dpg.add_text("周期:")
                            dpg.add_button(label="日", width=44, height=26,
                                           tag="pd_day",
                                           callback=lambda: _switch_period("D"))
                            dpg.add_button(label="周", width=44, height=26,
                                           tag="pd_week",
                                           callback=lambda: _switch_period("W"))
                            dpg.add_button(label="月", width=44, height=26,
                                           tag="pd_month",
                                           callback=lambda: _switch_period("M"))
                            dpg.add_button(label="分时", width=54, height=26,
                                           tag="pd_min",
                                           callback=lambda: _switch_period("MIN"))
                            dpg.add_text("(分时自动每5秒刷新,仅拉当前这只)",
                                         color=(130, 130, 130))
                        dpg.add_text("K线 / 成交额 / MACD", tag="kline_title")
                        dpg.add_child_window(tag="chart_area", border=False, height=-1)

                    # --- Tab 2: 今日推荐 ---
                    with dpg.tab(label="今日推荐"):
                        dpg.add_text("策略排行(慢·算一次即可)与今日推荐(快)已分开:"
                                     "先刷新排行,之后每天点『生成今日推荐』秒出",
                                     color=(160, 200, 255), wrap=760)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="刷新策略排行", callback=on_run_ranking,
                                           width=150, height=32, tag="rank_run_btn")
                            dpg.add_button(label="生成今日推荐", callback=on_gen_picks,
                                           width=150, height=32, tag="picks_run_btn")
                            dpg.add_checkbox(label="大盘趋势过滤", default_value=True,
                                             tag="rank_market")
                            dpg.add_text("(推荐开启)", color=(130, 130, 130))
                        dpg.add_text("点『刷新策略排行』开始", tag="rank_status",
                                     color=(255, 200, 100), wrap=700)
                        dpg.add_separator()
                        dpg.add_text("策略排行榜(按综合分)", color=(120, 220, 160))
                        with dpg.table(tag="rank_table", header_row=True, resizable=True,
                                       policy=dpg.mvTable_SizingStretchProp,
                                       height=190, scrollY=True):
                            for col in ["排名", "策略", "综合分", "年化%", "总收益%",
                                        "胜率%", "回撤%", "盈亏比", "交易数"]:
                                dpg.add_table_column(label=col)
                        dpg.add_separator()
                        dpg.add_text("热门板块(今日推荐票的行业分布,x2+标红)",
                                     color=(120, 220, 160))
                        dpg.add_child_window(tag="hot_sectors", height=40, border=False)
                        dpg.add_text("今日推荐票(最强策略选出,命中越多越值得关注)",
                                     color=(120, 220, 160))
                        with dpg.table(tag="picks_table", header_row=True, resizable=True,
                                       policy=dpg.mvTable_SizingStretchProp,
                                       height=-1, scrollY=True):
                            for col in ["代码", "名称", "行业", "现价", "来源策略",
                                        "策略排名", "命中", "得分", "操作"]:
                                dpg.add_table_column(label=col)

                    # --- Tab 3: ETF 榜单 ---
                    with dpg.tab(label="ETF榜单"):
                        dpg.add_text("主流宽基+行业ETF实时榜单。点『刷新ETF行情』拉最新价,"
                                     "点代码看K线,点+加入自选跟踪。",
                                     color=(160, 200, 255), wrap=760)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="刷新ETF行情", callback=on_refresh_etf,
                                           width=150, height=32, tag="etf_run_btn")
                            dpg.add_combo(["按涨跌幅", "按成交额", "按最新价"],
                                          default_value="按涨跌幅", width=140,
                                          tag="etf_sort", callback=lambda s, a: _render_etf())
                        dpg.add_text("未加载(先在左侧勾选『同时拉取ETF』更新数据,再点刷新)",
                                     tag="etf_status", color=(255, 200, 100), wrap=740)
                        dpg.add_separator()
                        with dpg.table(tag="etf_table", header_row=True, resizable=True,
                                       policy=dpg.mvTable_SizingStretchProp,
                                       height=-1, scrollY=True):
                            for col in ["代码", "名称", "现价", "涨跌幅%", "今开",
                                        "最高", "最低", "成交额", "操作"]:
                                dpg.add_table_column(label=col)

                    # --- Tab 4: 自选持仓 ---
                    with dpg.tab(label="自选持仓 / 盯盘"):
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="刷新盈亏", callback=lambda: _refresh_watchlist(),
                                           width=110, height=30)
                            dpg.add_button(label="刷新实时行情", callback=on_refresh_realtime,
                                           width=140, height=30)
                            dpg.add_text("", tag="watch_status", color=(255, 200, 100), wrap=520)
                        dpg.add_text("买入价默认记为加入时的现价;『刷新实时行情』拉全市场快照算"
                                     "当日涨跌+浮动盈亏并检查预警;点代码看K线",
                                     wrap=900, color=(130, 130, 130))
                        dpg.add_separator()
                        with dpg.table(tag="watch_table", header_row=True, resizable=True,
                                       policy=dpg.mvTable_SizingStretchProp,
                                       height=240, scrollY=True):
                            for col in ["代码", "名称", "加入日", "买入价", "现价",
                                        "今日涨跌%", "浮动盈亏%", "备注", "操作"]:
                                dpg.add_table_column(label=col)
                        dpg.add_separator()

                        # ---- 价格预警 ----
                        dpg.add_text("价格预警(刷新实时行情时检查触发)", color=(120, 220, 160))
                        dpg.add_child_window(tag="alert_fired", height=70, border=True)
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(label="代码", width=90, tag="alert_code",
                                               hint="6位")
                            dpg.add_input_float(label="价≤", width=110, tag="alert_plow",
                                                default_value=0, step=0)
                            dpg.add_input_float(label="价≥", width=110, tag="alert_phigh",
                                                default_value=0, step=0)
                        with dpg.group(horizontal=True):
                            dpg.add_input_float(label="跌幅≤%", width=110, tag="alert_clow",
                                                default_value=0, step=0)
                            dpg.add_input_float(label="涨幅≥%", width=110, tag="alert_chigh",
                                                default_value=0, step=0)
                            dpg.add_input_text(label="备注", width=150, tag="alert_note")
                            dpg.add_button(label="添加预警", callback=on_add_alert,
                                           width=100, height=26)
                        dpg.add_text("阈值填 0 = 该条件不启用。例:跌幅≤填 -5 表示跌超5%报警;"
                                     "涨幅≥填 8 表示涨超8%报警。", wrap=900,
                                     color=(130, 130, 130))
                        dpg.add_text("", tag="alert_status", color=(255, 200, 100), wrap=900)
                        with dpg.table(tag="alerts_table", header_row=True, resizable=True,
                                       policy=dpg.mvTable_SizingStretchProp,
                                       height=140, scrollY=True):
                            for col in ["代码", "名称", "价≤", "价≥", "跌幅≤%", "涨幅≥%", "备注", "操作"]:
                                dpg.add_table_column(label=col)

    # 初始化
    on_strategy_change(None, list(STRATEGY_LABEL2KEY.keys())[0])
    _refresh_cache_info()
    _refresh_market_state()
    _refresh_watchlist()
    _refresh_alerts_table()
    _render_alerts([])
    _load_cached_ranking()   # 启动时载入上次排行(若有),推荐可直接秒出
    _sync_period_buttons("D")  # 默认高亮"日"周期

    # 全局鼠标移动:驱动K线主图的自定义悬停提示(跟随鼠标定位到最近蜡烛)
    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=_on_kline_hover)

    dpg.create_viewport(title="A-Share Stock Picker", width=1320, height=840)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("root", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    build_ui()
