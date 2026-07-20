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
from app.data import market as market_data
from app.strategy import indicators as ind
from app.strategy import scanner
from app.strategy import backtest as bt
from app.strategy import ranking
from app.strategy import optimizer
from app.strategy import funda
from app.strategy import chip
from app.strategy.market import MarketTrend
from app.ai import commentary as ai_commentary, is_configured as ai_configured, config_hint as ai_config_hint
from app.ai import ranker as ai_ranker
from app.ai import nl_query as ai_nl
from app.strategy.base import ALL_STRATEGIES
from app.report import exporter

# A 股配色:涨=红 跌=绿
RED = (220, 40, 40, 255)
GREEN = (0, 170, 90, 255)
# 列表行悬停高亮色(中性蓝,不与涨红跌绿冲突):hover 较亮,选中/按下稍暗
ROW_HOVER = (58, 96, 150, 170)
ROW_ACTIVE = (70, 116, 180, 200)
ROW_SELECT = (48, 78, 122, 130)
# 持仓表专用:单元格背景高亮色(用于 highlight_table_cell,只染背景不抢点击,
# 故按钮/输入框仍可点)。alpha 适中,盖背景但不遮文字。
WATCH_CELL_HL = (58, 96, 150, 90)


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
    "ai_hint": None,            # 当前标的的策略上下文(AI点评 strategy_hint)
    "cs_weights": None,         # L1横截面因子权重 dict(选中横截面策略时用)
    "cs_neutralize": True,      # L1横截面 行业中性化开关
    "mkt_poll_on": False,       # 行情页定时刷新开关
    "mkt_poll_thread": None,    # 行情页定时刷新线程
    "mkt_board_df": None,       # 热门板块最近一次数据(DataFrame),供排序切换复用
    "mkt_board_sort": "chg_pct",  # 热门板块当前排序字段
    "mkt_sector_val": {},       # {板块名:{pe,pb,pe_pct,pb_pct}} PE/PB中位数+历史分位(慢速回填,按天缓存)
    "mkt_sector_val_inflight": False,  # 板块估值当日中位数回填 in-flight 标记
    "mkt_sector_bf_done": set(),  # 本次会话已回填过历史的板块名(避免重复60s回填)
    "mkt_sector_bf_inflight": False,  # 历史回填 in-flight 标记(串行, 一次只回填一个)
    "mkt_idx_df": None,         # 核心指数最近一次数据(DataFrame),供AI大盘点评复用
    "mkt_act": None,            # 涨跌家数/情绪最近一次数据(dict),供AI大盘点评复用
    "mkt_summ": None,           # 两市总貌最近一次数据(dict),供AI大盘点评复用
    "mkt_drag": None,           # 行情页正在拖拽的分隔条(child_window tag),None=未拖拽
    "mkt_drag_y0": 0.0,         # 拖拽起始鼠标Y(viewport绝对坐标)
    "mkt_drag_h0": 0.0,         # 拖拽起始时该块高度
}

# 行情页各可调高度区块的默认高度(px)。拖拽分隔条实时改写这里对应的 child_window。
MKT_SECTION_HEIGHTS = {
    "sec_activity": 150,   # 涨跌家数/情绪 + 盘面量能/情绪温度
    "sec_dist": 235,       # 涨跌幅分布直方图(竖直柱状图)
    "sec_diag": 240,       # 板块强弱·资金诊断(高潜力/高风险左右平行)
    "sec_index": 240,      # 核心指数表
    "sec_board": 300,      # 热门板块表
}
MKT_SECTION_MIN = 80       # 拖拽时单块最小高度
MKT_SECTION_MAX = 900      # 拖拽时单块最大高度

# L1 横截面多因子(智能版)在策略下拉框里的特殊 key。它不走逐股 evaluate,
# 而是独立的全市场横截面打分路径(见 app/strategy/cross_section),故用 sentinel 区分。
CROSS_SECTION_KEY = "__cross_section__"
CROSS_SECTION_LABEL = "★ 横截面多因子(智能版)"

# 分时轮询间隔(秒):免费源限流,只盯 1 只,5 秒够用
POLL_INTERVAL = 5

# 行情页定时刷新间隔(秒):指数实时接口约 3s,大盘概览无需太频繁,10 秒足够
MKT_POLL_INTERVAL = 10


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
    # 列表行悬停高亮主题:绑定到跨列 selectable(span_columns=True),
    # 鼠标悬停整行变亮,选中/按下时略深。用于所有股票/ETF 列表。
    with dpg.theme(tag="row_hover"):
        with dpg.theme_component(dpg.mvSelectable):
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, ROW_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, ROW_ACTIVE)
            dpg.add_theme_color(dpg.mvThemeCol_Header, ROW_SELECT)
    # 分时价格线红/绿主题(轮询增量更新时复用,避免反复新建 theme)
    with dpg.theme(tag="line_red"):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, RED, category=dpg.mvThemeCat_Plots)
    with dpg.theme(tag="line_green"):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, GREEN, category=dpg.mvThemeCat_Plots)
    # 拖拽分隔条主题:一条低调的细横条(button 伪装), 悬停/按下变亮提示可拖
    with dpg.theme(tag="mkt_drag_bar"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (70, 70, 78))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (90, 150, 220))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (110, 170, 240))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 2)


# ---------- 行情页可拖拽分隔条(调整各大块高度) ----------
def _mkt_drag_start(section_tag):
    """按下某分隔手柄:记录当前块与起始鼠标Y、起始高度,进入拖拽态。"""
    def _cb(sender=None, app_data=None, user_data=None):
        if not dpg.does_item_exist(section_tag):
            return
        STATE["mkt_drag"] = section_tag
        _, my = dpg.get_mouse_pos(local=False)
        STATE["mkt_drag_y0"] = my
        STATE["mkt_drag_h0"] = float(dpg.get_item_height(section_tag) or
                                     MKT_SECTION_HEIGHTS.get(section_tag, 200))
    return _cb


def _mkt_drag_move(sender=None, app_data=None, user_data=None):
    """拖拽中:按鼠标Y位移实时改写当前块高度(带 min/max 约束)。"""
    tag = STATE.get("mkt_drag")
    if not tag or not dpg.does_item_exist(tag):
        return
    _, my = dpg.get_mouse_pos(local=False)
    dy = my - STATE["mkt_drag_y0"]
    h = STATE["mkt_drag_h0"] + dy
    h = max(MKT_SECTION_MIN, min(MKT_SECTION_MAX, h))
    dpg.set_item_height(tag, int(h))
    MKT_SECTION_HEIGHTS[tag] = int(h)


def _mkt_drag_end(sender=None, app_data=None, user_data=None):
    """松开鼠标:退出拖拽态。"""
    STATE["mkt_drag"] = None


def _add_mkt_section_handle(section_tag):
    """在某可调块下方加一条可上下拖拽的分隔手柄(细横条按钮)。
    向下拖=放大该块,向上拖=缩小。松手保存高度到 MKT_SECTION_HEIGHTS。"""
    h = dpg.add_button(label="", height=6, width=-1,
                       callback=None, tag=f"{section_tag}_handle")
    dpg.bind_item_theme(h, "mkt_drag_bar")
    with dpg.item_handler_registry() as reg:
        dpg.add_item_clicked_handler(dpg.mvMouseButton_Left,
                                     callback=_mkt_drag_start(section_tag))
    dpg.bind_item_handler_registry(h, reg)


# ---------- 业务回调 ----------
def on_strategy_change(sender, app_data):
    """切换策略时,重建参数控件区。"""
    # L1 横截面多因子(智能版):独立范式,渲染因子权重面板而非普通策略参数
    if app_data == CROSS_SECTION_LABEL:
        _render_cross_section_params()
        return
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


def _cs_weights_path():
    """横截面权重持久化文件路径(与数据库同目录)。"""
    import os
    from app.data import database as db
    return os.path.join(os.path.dirname(db.DB_PATH), "cs_weights.json")


def _load_saved_cs_weights():
    """读取上次保存的横截面权重+中性化开关;无则返回 None。"""
    import os, json
    path = _cs_weights_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cs_weights(weights, neutralize):
    """把当前横截面权重+中性化开关落盘,下次打开自动恢复。"""
    import json
    try:
        with open(_cs_weights_path(), "w", encoding="utf-8") as f:
            json.dump({"weights": {k: float(v) for k, v in weights.items()},
                       "neutralize": bool(neutralize)},
                      f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _render_cross_section_params():
    """渲染 L1 横截面多因子(智能版)的因子权重面板 + 行业中性化开关。"""
    from app.strategy import cross_section as cs
    # 标记当前为横截面模式(current_strategy 置 None,执行时走独立路径)
    STATE["current_strategy"] = None
    # 优先恢复上次保存的权重,没有才用默认(解决"关闭后权重还原"问题)
    _saved = _load_saved_cs_weights()
    if _saved and isinstance(_saved.get("weights"), dict):
        STATE["cs_weights"] = dict(cs.DEFAULT_WEIGHTS)
        STATE["cs_weights"].update({k: float(v) for k, v in _saved["weights"].items()
                                    if k in cs.DEFAULT_WEIGHTS})
        STATE["cs_neutralize"] = bool(_saved.get("neutralize", True))
    else:
        STATE["cs_weights"] = dict(cs.DEFAULT_WEIGHTS)
        STATE["cs_neutralize"] = True
    dpg.delete_item("param_area", children_only=True)
    dpg.add_text("全市场横截面打分:每个因子做z-score标准化+行业中性化,"
                 "问『这只票在全市场/同行业里排第几』,而非用绝对阈值卡线。",
                 parent="param_area", wrap=280, color=(150, 150, 150))
    dpg.add_text("因子权重(0=关闭该因子):", parent="param_area",
                 color=(180, 180, 120))
    _cs_labels = {"momentum": "动量(近20日涨幅)", "trend": "趋势(均线发散度)",
                  "volume": "量能(放量倍数)", "value": "低估(盈利收益率1/PE)",
                  "quality": "质量(ROE)"}
    for fk, flabel in _cs_labels.items():
        dpg.add_slider_float(
            label=flabel, parent="param_area",
            default_value=float(STATE["cs_weights"].get(fk, cs.DEFAULT_WEIGHTS[fk])),
            min_value=-100, max_value=100,
            format="%.0f", width=170, tag=f"cs_w_{fk}",
            callback=lambda s, a, u: STATE["cs_weights"].update({u: a}),
            user_data=fk,
        )
    dpg.add_text("权重可为负:正=顺势用(越大越好),负=反向用(越小越好),0=关闭。",
                 parent="param_area", wrap=280, color=(130, 130, 130))
    dpg.add_checkbox(label="行业中性化(在同行业内排名,消除行业偏差)",
                     parent="param_area",
                     default_value=bool(STATE.get("cs_neutralize", True)),
                     tag="cs_neutralize",
                     callback=lambda s, a: STATE.update({"cs_neutralize": a}))
    dpg.add_text("提示:开启中性化 → 选各行业内最强(分散);关闭 → 可能被强势板块霸屏。",
                 parent="param_area", wrap=280, color=(130, 130, 130))
    # 权重持久化:保存当前权重 / 恢复出厂默认(解决关闭软件后权重还原)
    dpg.add_separator(parent="param_area")
    with dpg.group(horizontal=True, parent="param_area"):
        dpg.add_button(label="保存当前权重", width=122, height=26,
                       callback=on_save_cs_weights)
        dpg.add_button(label="恢复默认权重", width=122, height=26,
                       callback=on_reset_cs_weights)
    dpg.add_text("权重会自动记住:回填或手动保存后,下次开软件仍是这套权重。",
                 parent="param_area", wrap=280, color=(130, 130, 130))
    # L3 自适应权重入口:用历史 IC 反推"数据说了算"的权重,一键回填上面滑块
    dpg.add_separator(parent="param_area")
    dpg.add_button(label="✦ 用数据推荐权重 (L3 自适应)", parent="param_area",
                   width=250, height=30, callback=on_open_adaptive_weights)
    dpg.add_text("L3:跑历史 IC 反推技术因子权重(可能为负=反向),对比回测后一键回填。",
                 parent="param_area", wrap=280, color=(130, 130, 130))
    # L2 因子体检入口:检验这套技术因子在历史上到底有没有预测力
    dpg.add_separator(parent="param_area")
    dpg.add_button(label="因子体检 (IC + 分组回测)", parent="param_area",
                   width=250, height=30, callback=on_open_factor_lab)
    dpg.add_text("L2:用历史数据检验因子有没有预测力、这套打分能不能选出强票。",
                 parent="param_area", wrap=280, color=(130, 130, 130))


def _collect_cs_weights_from_ui():
    """从当前滑块读回 5 个因子权重(以界面为准)。"""
    from app.strategy import cross_section as cs
    w = {}
    for fk in cs.DEFAULT_WEIGHTS:
        tag = f"cs_w_{fk}"
        w[fk] = float(dpg.get_value(tag)) if dpg.does_item_exist(tag) \
            else float(cs.DEFAULT_WEIGHTS[fk])
    return w


def on_save_cs_weights():
    """手动保存当前横截面权重+中性化开关到磁盘。"""
    w = _collect_cs_weights_from_ui()
    neu = bool(dpg.get_value("cs_neutralize")) if dpg.does_item_exist("cs_neutralize") \
        else bool(STATE.get("cs_neutralize", True))
    STATE["cs_weights"] = dict(w)
    STATE["cs_neutralize"] = neu
    ok = _save_cs_weights(w, neu)
    _set_status("已保存权重,下次打开自动恢复: "
                + ", ".join(f"{k}={v:+.0f}" for k, v in w.items())
                if ok else "保存失败,请检查磁盘权限")


def on_reset_cs_weights():
    """恢复出厂默认权重,并删除已保存的权重文件。"""
    import os
    from app.strategy import cross_section as cs
    for fk, dv in cs.DEFAULT_WEIGHTS.items():
        if dpg.does_item_exist(f"cs_w_{fk}"):
            dpg.set_value(f"cs_w_{fk}", float(dv))
    if dpg.does_item_exist("cs_neutralize"):
        dpg.set_value("cs_neutralize", True)
    STATE["cs_weights"] = dict(cs.DEFAULT_WEIGHTS)
    STATE["cs_neutralize"] = True
    try:
        p = _cs_weights_path()
        if os.path.exists(p):
            os.remove(p)
    except Exception:
        pass
    _set_status("已恢复出厂默认权重(动量25/趋势25/量能15/低估20/质量15)")


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

            # 强制全量重拉:覆盖本地历史(用于修正旧脏数据),否则增量只补最新
            force_full = bool(dpg.get_value("force_full"))
            incr = not force_full
            mode_tip = "全量重拉(覆盖历史)" if force_full else "增量"

            def cb(done, total, code):
                _set_status(f"[{mode_tip}]拉取日线 {done}/{n}  当前:{code}")

            # 首次全量从 2021 年起,保证回测有 4 年+样本;已有数据则自动增量
            # 腾讯源可并发,10 线程拉取,400 只约 3-5 分钟
            fetcher.update_all_kline(codes[:n], start_date="20210101",
                                     incremental=incr, progress_cb=cb, workers=10)
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
                    start_date="20210101", incremental=incr,
                    progress_cb=lambda d, t, c: _set_status(
                        f"[{mode_tip}]拉取 ETF 日线 {d}/{t}  当前:{c}"))
            cached = db.list_cached_codes()
            _set_status(f"数据更新完成,已缓存 {len(cached)} 只")
            _refresh_cache_info()
            _refresh_market_state()
        except Exception as e:
            _set_status(f"更新失败: {e}")

    threading.Thread(target=worker, daemon=True).start()


def on_run_scan():
    """后台线程:执行选股。"""
    # 判断当前是否为 L1 横截面模式(下拉框选中智能版时 current_strategy 为 None 但 cs_weights 有值)
    is_cross = (STATE["current_strategy"] is None
                and dpg.does_item_exist("strategy_combo")
                and dpg.get_value("strategy_combo") == CROSS_SECTION_LABEL)
    if STATE["current_strategy"] is None and not is_cross:
        _set_status("请先选择策略")
        return
    if not db.list_cached_codes():
        _set_status("本地无数据,请先点『更新股票数据』")
        return

    def worker():
        if is_cross:
            _run_cross_section_scan()
            return
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


def _run_cross_section_scan():
    """执行 L1 横截面多因子扫描(在 on_run_scan 的 worker 线程内调用)。"""
    from app.strategy import cross_section as cs
    _set_status("横截面打分:构建全市场因子矩阵...")
    weights = STATE.get("cs_weights") or dict(cs.DEFAULT_WEIGHTS)
    neutralize = bool(STATE.get("cs_neutralize", True))
    top_n = 50
    if dpg.does_item_exist("cs_top_n"):
        try:
            top_n = int(dpg.get_value("cs_top_n"))
        except Exception:  # noqa
            top_n = 50
    res = cs.scan_cross_section(
        weights=weights, top_n=top_n, neutralize=neutralize,
        progress_cb=lambda d, t: _set_status(f"横截面打分 {d}/{t}"),
    )
    # 横截面结果自带 value/quality 因子,基本面过滤仍可叠加(按界面阈值)
    raw_n = 0 if res is None else len(res)
    res = _apply_funda_filter(res)
    STATE["results"] = res
    _render_results(res)
    kept = 0 if res is None else len(res)
    tag = "行业中性" if neutralize else "全市场"
    if _funda_filter_on() and raw_n:
        _set_status(f"横截面选股完成({tag}),Top{raw_n},基本面过滤后剩 {kept} 只")
    else:
        _set_status(f"横截面选股完成({tag}),共 {kept} 只(分数=全市场分位)")


# ---------- 自然语言选股(路径A:一句人话 → AI 翻译成筛选参数 → 量化引擎执行) ----------
def on_nl_scan(sender=None, app_data=None, user_data=None):
    """自然语言选股:把一句人话交给 AI 翻译成『策略+参数+基本面过滤』,再由本地
    量化引擎执行筛选。AI 只做翻译,不直接挑票,保证结果可复现(见 app/ai/nl_query)。
    """
    query = (dpg.get_value("nl_query_in") or "").strip()
    if not query:
        _set_status("请先在自然语言选股框里输入你的选股想法")
        return
    if not ai_configured():
        dpg.set_value("nl_query_hint", "未配置 AI 模型,无法翻译。见左侧 AI 点评的配置说明。")
        return
    if not db.list_cached_codes():
        _set_status("本地无数据,请先点『更新股票数据』")
        return

    dpg.configure_item("nl_scan_btn", enabled=False, label="翻译中...")
    dpg.set_value("nl_query_hint", "正在把你的描述翻译成筛选条件...")

    def worker():
        try:
            def pcb(d, t):
                _set_status(f"AI选股扫描中 {d}/{t}")
            r = ai_nl.run_nl_scan(query, progress_cb=pcb)
            if not r.get("ok"):
                dpg.set_value("nl_query_hint", f"翻译失败:{r.get('error', '未知错误')}")
                _set_status("AI 选股未成功")
                return
            spec = r["spec"]
            df = r["df"]
            # 复用普通选股的结果表与后续流程(精排/点评/导出都能直接用)
            STATE["results"] = df
            _render_results(df)
            # 展示 AI 的翻译结果,让用户看懂它把人话理解成了什么
            strat_label = KEY2STRATEGY_LABEL.get(spec.get("strategy"),
                                                 spec.get("strategy", ""))
            parts = [f"策略={strat_label}"]
            if spec.get("params"):
                parts.append("参数=" + ", ".join(
                    f"{k}:{v}" for k, v in spec["params"].items()))
            if spec.get("funda"):
                parts.append("基本面=" + ", ".join(
                    f"{k}:{v}" for k, v in spec["funda"].items()))
            parts.append(f"取前{spec.get('top_n')}只")
            explain = spec.get("explain", "")
            hint = "AI 理解为 → " + "; ".join(parts)
            if explain:
                hint += f"\n{explain}"
            dpg.set_value("nl_query_hint", hint)
            n = 0 if df is None else len(df)
            _set_status(f"AI 选股完成,共 {n} 只(可精排/点评/导出)")
        except Exception as e:  # noqa
            dpg.set_value("nl_query_hint", f"AI 选股异常:{e}")
            _set_status("AI 选股异常")
        finally:
            if dpg.does_item_exist("nl_scan_btn"):
                dpg.configure_item("nl_scan_btn", enabled=True, label="AI 选股")

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


# ---------- L3 自适应权重(用历史 IC 反推权重 + 对比回测) ----------
# 缓存最近一次反推的权重,供"一键回填"按钮使用
_ADAPTIVE_CACHE = {"weights": None}


def on_open_adaptive_weights():
    """打开 L3 自适应权重弹窗:跑历史 IC 反推权重,对比回测,一键回填 L1 滑块。"""
    if not db.list_cached_codes():
        _set_status("本地无数据,无法反推权重,请先更新数据")
        return
    if dpg.does_item_exist("aw_window"):
        dpg.delete_item("aw_window")

    with dpg.window(label="用数据推荐权重 (L3:市场自适应)", tag="aw_window",
                    width=880, height=720, pos=(180, 60), modal=False):
        dpg.add_text("这里帮你自动算出一套权重:拿近 5 年真实历史,看每个因子过去到底管不管用,"
                     "据此给出建议权重,并当场用历史验证它比你现在的权重强多少。",
                     wrap=840, color=(160, 200, 255))
        dpg.add_text(
            "只对有完整历史走势的 3 个技术因子(动量/趋势/量能)自动算权重。"
            "低估、质量这两个基本面因子本地只有当前值、没有历史,没法验证,所以不动它们,保持你现在的设置。",
            wrap=840, color=(200, 160, 100))
        dpg.add_text(
            "怎么算的:一个因子过去越能稳定预测涨跌,给的权重越大;如果它过去是\"反着走\"的"
            "(得分高的反而后来跌),就给负权重表示反向使用;信号太弱、不够可信的,权重记 0、不参与。",
            wrap=840, color=(130, 130, 130))
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_input_int(label="预测未来几天(交易日)", default_value=5,
                              min_value=2, max_value=60, width=150, tag="aw_fwd")
            dpg.add_input_float(label="信号可信度要求(越大越严)", default_value=2.0,
                                min_value=0.0, max_value=6.0, step=0.5,
                                format="%.1f", width=170, tag="aw_tthr")
            dpg.add_button(label="开始计算权重", callback=on_run_adaptive,
                           width=160, height=30, tag="aw_run_btn")
        dpg.add_text("准备就绪 (点上面按钮开始;读取全市场历史约需 15~30 秒)", tag="aw_status",
                     color=(255, 200, 100))
        dpg.add_separator()

        # ---- ① 反推明细 ----
        dpg.add_text("① 每个因子过去管不管用 → 该给多少权重", color=(120, 220, 160))
        with dpg.table(tag="aw_detail_table", header_row=True, resizable=True,
                       policy=dpg.mvTable_SizingStretchProp, height=130, scrollY=True):
            for col in ["因子", "预测力(越偏离0越强)", "稳定性", "可信度",
                        "是否采用", "建议权重", "使用方式"]:
                dpg.add_table_column(label=col)
        dpg.add_text("", tag="aw_note", wrap=840, color=(200, 200, 160))
        dpg.add_separator()

        # ---- ② 对比回测 ----
        dpg.add_text("② 拿历史验证:你现在的权重 vs 系统建议的权重,哪个选股更强",
                     color=(120, 220, 160))
        dpg.add_text("下面数字是在近 5 年历史上模拟选股的表现(已剔除大盘涨跌因素);"
                     "未扣手续费,只用来比较两套权重谁更强,不是保证能赚这么多。",
                     wrap=840, color=(130, 130, 130))
        with dpg.table(tag="aw_cmp_table", header_row=True, resizable=True,
                       policy=dpg.mvTable_SizingStretchProp, height=150, scrollY=True):
            for col in ["权重方案", "选股超额年化%", "稳定性(夏普)", "最大回撤%",
                        "最强档年化%", "最弱档年化%"]:
                dpg.add_table_column(label=col)
        dpg.add_separator()

        # ---- ③ 回填 ----
        dpg.add_text("③ 采用建议", color=(120, 220, 160))
        dpg.add_text("确认第②步里\"建议权重\"确实比你现在的强之后,点下面按钮,把这套权重填到左侧滑块上,"
                     "然后直接点『开始选股』就能用了(基本面那两项不变)。",
                     wrap=840, color=(130, 130, 130))
        dpg.add_button(label="↩ 采用这套权重(填到左侧滑块)", callback=on_apply_adaptive,
                       width=280, height=32, tag="aw_apply_btn", enabled=False)


def on_run_adaptive():
    """后台线程:构建面板 → 反推权重 → 默认/自适应双回测 → 渲染。"""
    fwd = int(dpg.get_value("aw_fwd"))
    tthr = float(dpg.get_value("aw_tthr"))

    def worker():
        from app.strategy import panel as pnl, factor_ic as fic
        from app.strategy import adaptive_weights as aw, quantile_bt as qbt
        dpg.configure_item("aw_run_btn", enabled=False)
        dpg.configure_item("aw_apply_btn", enabled=False)
        try:
            dpg.set_value("aw_status", "构建全市场历史行情面板...")
            p = pnl.build_panel(
                fwd_days=fwd,
                progress_cb=lambda d, t: dpg.set_value(
                    "aw_status", f"读取历史行情 {d}/{t}"))
            if p is None or p.empty:
                dpg.set_value("aw_status", "没有足够的历史数据,请先更新数据")
                return
            dpg.set_value("aw_status", "计算各因子历史预测力,得出建议权重...")
            summ, _ = fic.compute_ic(fwd_days=fwd, panel=p)
            der = aw.derive_weights(fwd_days=fwd, t_threshold=tthr,
                                    panel=p, ic_summary=summ)
            adaptive_w = der["weights"]
            dpg.set_value("aw_status", "历史验证:用你现在的权重模拟选股...")
            default_w = {"momentum": 40.0, "trend": 40.0, "volume": 20.0}
            res_def = qbt.run_quantile_backtest(
                weights=default_w, fwd_days=fwd, panel=p)
            dpg.set_value("aw_status", "历史验证:用系统建议的权重模拟选股...")
            res_ada = (qbt.run_quantile_backtest(
                weights=adaptive_w, fwd_days=fwd, panel=p)
                if adaptive_w else None)
            _render_adaptive(der, res_def, res_ada, default_w, adaptive_w, p)
            _ADAPTIVE_CACHE["weights"] = adaptive_w
            if adaptive_w:
                dpg.configure_item("aw_apply_btn", enabled=True)
            dpg.set_value(
                "aw_status",
                f"完成:用了 {len(p):,} 条历史记录 · {p['code'].nunique()} 只股票 · "
                f"{p['date'].min()}~{p['date'].max()}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            dpg.set_value("aw_status", f"计算失败: {e}")
        finally:
            dpg.configure_item("aw_run_btn", enabled=True)

    threading.Thread(target=worker, daemon=True).start()


def _render_adaptive(der, res_def, res_ada, default_w, adaptive_w, panel):
    """渲染反推明细表 + 人话总结 + 默认/自适应对比表。"""
    # ---- ① 反推明细 ----
    dpg.delete_item("aw_detail_table", children_only=True)
    for col in ["因子", "预测力(越偏离0越强)", "稳定性", "可信度",
                "是否采用", "建议权重", "使用方式"]:
        dpg.add_table_column(label=col, parent="aw_detail_table")
    detail = der.get("detail")
    if detail is not None and not detail.empty:
        for _, r in detail.iterrows():
            w = r["weight"]
            selected = bool(r["selected"])
            if not selected:
                direction, dcolor = "信号太弱·不采用", (150, 150, 150)
            elif w > 0:
                direction, dcolor = "顺着用(得分越高越好)", RED
            else:
                direction, dcolor = "反着用(得分越低越好)", GREEN
            with dpg.table_row(parent="aw_detail_table"):
                dpg.add_text(str(r["factor_cn"]))
                dpg.add_text(f"{r['ic_mean']:+.4f}")
                dpg.add_text(f"{r['ic_ir']:+.3f}")
                dpg.add_text(f"{r['t_stat']:+.2f}")
                dpg.add_text("✓ 采用" if selected else "✗ 不用")
                dpg.add_text(f"{w:+.1f}", color=dcolor)
                dpg.add_text(direction, color=dcolor)
    dpg.set_value("aw_note", der.get("note", ""))

    # ---- ② 对比表 ----
    dpg.delete_item("aw_cmp_table", children_only=True)
    for col in ["权重方案", "选股超额年化%", "稳定性(夏普)", "最大回撤%",
                "最强档年化%", "最弱档年化%"]:
        dpg.add_table_column(label=col, parent="aw_cmp_table")

    def _row(tag, w, res):
        if res is None:
            with dpg.table_row(parent="aw_cmp_table"):
                dpg.add_text(tag)
                for _ in range(5):
                    dpg.add_text("—")
            return None
        ls = res.get("ls_name", "")
        m_ls = res["metrics"].get(ls, {})
        m_q5 = res["metrics"].get("Q5", {})
        m_q1 = res["metrics"].get("Q1", {})
        ann = m_ls.get("ann_return", float("nan"))
        color = RED if (ann == ann and ann >= 0) else GREEN
        with dpg.table_row(parent="aw_cmp_table"):
            dpg.add_text(tag)
            dpg.add_text(f"{ann*100:+.2f}", color=color)
            dpg.add_text(f"{m_ls.get('sharpe', float('nan')):+.2f}", color=color)
            dpg.add_text(f"{m_ls.get('max_drawdown', float('nan'))*100:+.2f}")
            dpg.add_text(f"{m_q5.get('ann_return', float('nan'))*100:+.2f}")
            dpg.add_text(f"{m_q1.get('ann_return', float('nan'))*100:+.2f}")
        return m_ls.get("sharpe", float("nan"))

    s_def = _row("你现在的权重", default_w, res_def)
    s_ada = _row("系统建议的权重", adaptive_w, res_ada)
    # 结论行
    if res_ada is not None and s_def == s_def and s_ada == s_ada:
        better = s_ada > s_def
        verdict = ("✓ 历史验证:系统建议的权重更强,可以采用(注意这只是这段历史的结论)。"
                   if better else
                   "✗ 历史验证:系统建议的没比你现在的强,建议先别改,或换个预测天数再试。")
        dpg.set_value("aw_note", dpg.get_value("aw_note") + "\n\n【一句话结论】" + verdict)


def on_apply_adaptive():
    """把反推的技术因子权重回填到 L1 滑块(基本面因子保持不变)。"""
    w = _ADAPTIVE_CACHE.get("weights")
    if not w:
        _set_status("尚无可回填的权重,请先运行反推")
        return
    applied = []
    for fk, val in w.items():
        tag = f"cs_w_{fk}"
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, float(val))
            if STATE.get("cs_weights") is not None:
                STATE["cs_weights"][fk] = float(val)
            applied.append(f"{fk}={val:+.1f}")
    _set_status("已采用系统建议的权重: " + ", ".join(applied) +
                " (基本面因子不变,可直接点开始选股)")
    # 回填即持久化:下次打开软件自动恢复这套权重,不必每次重跑 L3
    try:
        w_now = _collect_cs_weights_from_ui()
        neu = bool(dpg.get_value("cs_neutralize")) \
            if dpg.does_item_exist("cs_neutralize") \
            else bool(STATE.get("cs_neutralize", True))
        _save_cs_weights(w_now, neu)
    except Exception:
        pass


# ---------- L2 因子体检(IC + 分位分组回测) ----------
def on_open_factor_lab():
    """打开 L2 因子体检弹窗:检验技术因子的历史预测力 + 分组回测。"""
    if not db.list_cached_codes():
        _set_status("本地无数据,无法体检,请先更新数据")
        return
    if dpg.does_item_exist("fl_window"):
        dpg.delete_item("fl_window")

    with dpg.window(label="因子体检 (L2:IC + 分组回测)", tag="fl_window",
                    width=940, height=760, pos=(160, 50), modal=False):
        dpg.add_text("拿近 5 年真实历史,检验\"动量/趋势/量能\"这几个选股指标过去到底管不管用,"
                     "以及照这套打分选出来的票,是不是真能跑赢差的票。",
                     wrap=900, color=(160, 200, 255))
        dpg.add_text(
            "只检验有完整历史走势的 3 个技术指标(动量/趋势/量能)。"
            "低估、质量这两个基本面指标本地只有当前值、没有历史走势,拿它硬凑历史会\"偷看答案\"、结论不可信,所以这里不检验它们。",
            wrap=900, color=(200, 160, 100))
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_input_int(label="预测未来几天(交易日)", default_value=5,
                              min_value=2, max_value=60, width=140, tag="fl_fwd")
            dpg.add_input_int(label="分成几档", default_value=5, min_value=3,
                              max_value=10, width=120, tag="fl_ngrp")
        dpg.add_text("下面三个权重只用于\"分档打分\";每个指标单独的好坏检验(见①)跟权重无关:",
                     color=(180, 180, 120))
        with dpg.group(horizontal=True):
            dpg.add_slider_float(label="动量", default_value=40, min_value=0,
                                 max_value=100, format="%.0f", width=180, tag="fl_w_momentum")
            dpg.add_slider_float(label="趋势", default_value=40, min_value=0,
                                 max_value=100, format="%.0f", width=180, tag="fl_w_trend")
            dpg.add_slider_float(label="量能", default_value=20, min_value=0,
                                 max_value=100, format="%.0f", width=180, tag="fl_w_volume")
        with dpg.group(horizontal=True):
            dpg.add_button(label="开始体检", callback=on_run_factor_lab,
                           width=160, height=32, tag="fl_run_btn")
        dpg.add_text("准备就绪 (点上面按钮开始;读取全市场历史约需 15~30 秒)", tag="fl_status",
                     color=(255, 200, 100))
        dpg.add_separator()

        # ---- IC 结果 ----
        dpg.add_text("① 每个指标过去到底管不管用", color=(120, 220, 160))
        dpg.add_text("怎么看:预测力越偏离 0 越强;稳定性越高越可靠;可信度>2 才算真信号。"
                     "如果是负的=这个指标\"反着走\"(得分高的后来反而跌,反着用可能更好)。",
                     wrap=900, color=(130, 130, 130))
        with dpg.table(tag="fl_ic_table", header_row=True, resizable=True,
                       policy=dpg.mvTable_SizingStretchProp, height=130, scrollY=True):
            for col in ["指标", "预测力(越偏离0越强)", "稳定性", "赢的比例", "可信度", "检验次数", "结论"]:
                dpg.add_table_column(label=col)
        dpg.add_separator()

        # ---- 分组回测 ----
        dpg.add_text("② 分档验证:把全市场按打分分成几档,看最强档能不能跑赢最弱档",
                     color=(120, 220, 160))
        dpg.add_text("下面的\"最强档减最弱档\"已经剔除了大盘涨跌,只看纯选股能力:正=能选出好票,负=方向反了。"
                     "未扣手续费,是能力体检、不是保证能赚这么多。", wrap=900, color=(130, 130, 130))
        with dpg.plot(tag="fl_nav_plot", height=200, width=-1, no_box_select=True):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="第几次换仓", tag="fl_navx")
            dpg.add_plot_axis(dpg.mvYAxis, label="累计涨了多少(起点1.0)", tag="fl_navy")
        with dpg.table(tag="fl_bt_table", header_row=True, resizable=True,
                       policy=dpg.mvTable_SizingStretchProp, height=180, scrollY=True):
            for col in ["组别", "年化%", "夏普", "最大回撤%", "累计收益%"]:
                dpg.add_table_column(label=col)


def on_run_factor_lab():
    """后台线程:构建面板 → 算 IC → 分组回测 → 渲染。"""
    fwd = int(dpg.get_value("fl_fwd"))
    ngrp = int(dpg.get_value("fl_ngrp"))
    weights = {
        "momentum": float(dpg.get_value("fl_w_momentum")),
        "trend": float(dpg.get_value("fl_w_trend")),
        "volume": float(dpg.get_value("fl_w_volume")),
    }

    def worker():
        from app.strategy import panel as pnl, factor_ic as fic, quantile_bt as qbt
        dpg.configure_item("fl_run_btn", enabled=False)
        try:
            dpg.set_value("fl_status", "构建全市场历史行情面板...")
            p = pnl.build_panel(
                fwd_days=fwd,
                progress_cb=lambda d, t: dpg.set_value(
                    "fl_status", f"读取历史行情 {d}/{t}"))
            if p is None or p.empty:
                dpg.set_value("fl_status", "没有足够的历史数据,请先更新数据")
                return
            dpg.set_value("fl_status", "检验每个指标的历史预测力...")
            summ, series = fic.compute_ic(fwd_days=fwd, panel=p)
            dpg.set_value("fl_status", "分档模拟选股...")
            res = qbt.run_quantile_backtest(
                weights=weights, fwd_days=fwd, n_groups=ngrp, panel=p)
            _render_factor_lab(summ, res, p)
            dpg.set_value(
                "fl_status",
                f"体检完成:用了 {len(p):,} 条历史记录 · {p['code'].nunique()} 只股票 · "
                f"{p['date'].min()}~{p['date'].max()} · 换仓 {res['periods']} 次")
        except Exception as e:
            import traceback
            traceback.print_exc()
            dpg.set_value("fl_status", f"体检失败: {e}")
        finally:
            dpg.configure_item("fl_run_btn", enabled=True)

    threading.Thread(target=worker, daemon=True).start()


def _render_factor_lab(summ, res, panel):
    """渲染 IC 表 + 分组净值曲线 + 分组指标表。"""
    # ---- ① IC 表 ----
    dpg.delete_item("fl_ic_table", children_only=True)
    for col in ["指标", "预测力(越偏离0越强)", "稳定性", "赢的比例", "可信度", "检验次数", "结论"]:
        dpg.add_table_column(label=col, parent="fl_ic_table")
    for fac in summ.index:
        r = summ.loc[fac]
        ic = r["ic_mean"]
        t = r["t_stat"]
        # 结论:方向 + 显著性
        if abs(ic) < 0.02 or (t == t and abs(t) < 2):
            verdict, vcolor = "太弱·没啥用", (150, 150, 150)
        elif ic > 0:
            verdict, vcolor = "有用·顺着用", RED
        else:
            verdict, vcolor = "反着走·反着用更好", GREEN
        with dpg.table_row(parent="fl_ic_table"):
            dpg.add_text(str(fac))
            dpg.add_text(f"{ic:+.4f}")
            dpg.add_text(f"{r['ic_ir']:+.3f}")
            dpg.add_text(f"{r['positive_ratio']*100:.1f}%")
            dpg.add_text(f"{t:+.2f}")
            dpg.add_text(f"{int(r['n_periods'])}")
            dpg.add_text(verdict, color=vcolor)

    # ---- ② 分组净值曲线 ----
    for tag in ("fl_navx", "fl_navy"):
        dpg.delete_item(tag, children_only=True)
    curves = res.get("group_curves", {})
    ls_name = res.get("ls_name", "")
    # 分组曲线:Q1..Qn 用冷→暖渐变,多空单独醒目色
    n_groups = res.get("n_groups", 5)
    for name, nav in curves.items():
        ys = [float(v) for v in nav.values]
        xs = list(range(len(ys)))
        series = dpg.add_line_series(xs, ys, label=name, parent="fl_navy")
        if name == ls_name:
            # 多空曲线加粗醒目(主题:金色)
            with dpg.theme() as th:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 190, 60, 255),
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 3.0,
                                        category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme(series, th)
    dpg.set_axis_limits_auto("fl_navx")
    dpg.set_axis_limits_auto("fl_navy")

    # ---- ② 分组指标表 ----
    dpg.delete_item("fl_bt_table", children_only=True)
    for col in ["档位", "年化%", "稳定性(夏普)", "最大回撤%", "累计收益%"]:
        dpg.add_table_column(label=col, parent="fl_bt_table")
    metrics = res.get("metrics", {})
    for name in curves.keys():
        m = metrics.get(name, {})
        ann = m.get("ann_return", float("nan"))
        # A股惯例:正收益红,负收益绿
        color = RED if (ann == ann and ann >= 0) else GREEN
        with dpg.table_row(parent="fl_bt_table"):
            dpg.add_text(name, color=(255, 190, 60) if name == ls_name else None)
            dpg.add_text(f"{ann*100:+.2f}", color=color)
            dpg.add_text(f"{m.get('sharpe', float('nan')):.2f}")
            dpg.add_text(f"{m.get('max_drawdown', float('nan'))*100:+.2f}")
            dpg.add_text(f"{m.get('total_return', float('nan'))*100:+.2f}", color=color)


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

    # 当前策略名(作为 AI 点评的上下文备注)
    try:
        _strat_name = (type(STATE["current_strategy"]).name
                       if STATE.get("current_strategy") else "")
    except Exception:  # noqa
        _strat_name = ""

    for _, r in df.iterrows():
        code = r["code"]
        fd = fmap.get(code, {})
        _hint = (f"被『{_strat_name}』策略选中,得分{r['score']},"
                 f"命中理由:{r['reason']}") if _strat_name else \
                f"被量化策略选中,得分{r['score']},命中理由:{r['reason']}"
        with dpg.table_row(parent="result_table"):
            _sel = dpg.add_selectable(
                label=code, span_columns=True, user_data=(code, _hint),
                callback=_on_result_click,   # 同页,单击即画K线
            )
            dpg.bind_item_theme(_sel, "row_hover")
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


# ---------- AI 精排(路径B:对选股结果做二次优选排序) ----------
def _rating_color(rating):
    """评级 → 颜色(A股惯例:偏多=红, 偏空=绿, 中性=灰)。"""
    if rating == "偏多":
        return RED
    if rating == "偏空":
        return GREEN
    return (170, 170, 170)


def _risk_color(risk):
    """风险等级 → 颜色(高=红醒目, 中=橙, 低=绿)。"""
    if risk == "高":
        return (230, 80, 80)
    if risk == "中":
        return (230, 170, 60)
    if risk == "低":
        return (120, 190, 120)
    return (170, 170, 170)


def on_ai_rank(sender=None, app_data=None, user_data=None):
    """对当前选股结果做 AI 精排优选(后台线程,不阻塞界面)。

    量化引擎已选出候选,这里让 AI 在候选内横向比较、排出 Top10 并给理由/评级。
    AI 只在给定候选集合内排序,不产出新代码(见 app/ai/ranker)。
    """
    res = STATE.get("results")
    if res is None or getattr(res, "empty", True):
        dpg.set_value("ai_rank_hint", "请先执行选股,有结果后再精排")
        return
    if not ai_configured():
        dpg.configure_item("ai_rank_win", show=True)
        dpg.set_value("ai_rank_disclaimer", ai_config_hint())
        dpg.delete_item("ai_rank_table", children_only=True)
        for col in ["#", "代码", "名称", "行业", "评级", "风险", "入选理由"]:
            dpg.add_table_column(label=col, parent="ai_rank_table")
        return

    codes = list(res["code"])[:ai_ranker.MAX_CANDIDATES]
    try:
        strat_name = (type(STATE["current_strategy"]).name
                      if STATE.get("current_strategy") else "")
    except Exception:  # noqa
        strat_name = ""
    hint = (f"由『{strat_name}』策略从全市场选出的候选,已按综合得分排序"
            if strat_name else "由量化策略从全市场选出的候选")

    dpg.configure_item("ai_rank_win", show=True)
    dpg.set_value("ai_rank_hint", "")
    dpg.set_value("ai_rank_disclaimer", "")
    dpg.set_value("ai_rank_status", f"正在精排 {len(codes)} 只候选,请稍候...")
    dpg.configure_item("ai_rank_btn", enabled=False, label="精排中...")

    def worker():
        try:
            def pcb(d, t):
                if dpg.does_item_exist("ai_rank_status"):
                    dpg.set_value("ai_rank_status", f"收集事实 {d}/{t}...")
            r = ai_ranker.rank_stocks(codes, top_n=10, strategy_hint=hint,
                                      progress_cb=pcb)
            dpg.delete_item("ai_rank_table", children_only=True)
            for col in ["#", "代码", "名称", "行业", "评级", "风险", "入选理由"]:
                dpg.add_table_column(label=col, parent="ai_rank_table")
            if not r.get("ok"):
                dpg.set_value("ai_rank_status", "")
                dpg.set_value("ai_rank_disclaimer",
                              f"精排失败: {r.get('error', '未知错误')}")
                return
            for i, it in enumerate(r["ranking"], 1):
                with dpg.table_row(parent="ai_rank_table"):
                    dpg.add_text(str(i))
                    code = it["code"]
                    _sel = dpg.add_selectable(label=code, span_columns=True,
                                       user_data=code,
                                       callback=lambda s, a, u: _on_pick_code(u))
                    dpg.bind_item_theme(_sel, "row_hover")
                    dpg.add_text(it.get("name", ""))
                    dpg.add_text(it.get("industry", ""))
                    dpg.add_text(it.get("rating") or "-",
                                 color=_rating_color(it.get("rating")))
                    dpg.add_text(it.get("risk") or "-",
                                 color=_risk_color(it.get("risk")))
                    dpg.add_text(it.get("reason") or "")
            tag = "(解析降级:暂按量化得分序)" if r.get("degraded") else ""
            dpg.set_value("ai_rank_status",
                          f"完成,共 {r.get('n_candidates', 0)} 只候选 {tag}")
            dpg.set_value("ai_rank_disclaimer", r.get("disclaimer", ""))
        except Exception as e:  # noqa
            dpg.set_value("ai_rank_status", "")
            dpg.set_value("ai_rank_disclaimer", f"精排异常: {e}")
        finally:
            if dpg.does_item_exist("ai_rank_btn"):
                dpg.configure_item("ai_rank_btn", enabled=True,
                                   label="AI精排Top10")

    threading.Thread(target=worker, daemon=True).start()



# ---------- AI 批量点评(晨报) & 组合解读 ----------
def _codes_from_source(source: str):
    """按来源取一批代码 + 标题。source: 'results'(选股结果) / 'watch'(自选池)。"""
    if source == "watch":
        wl = db.load_watchlist()
        if wl is None or wl.empty:
            return [], "自选池"
        col = "代码" if "代码" in wl.columns else ("code" if "code" in wl.columns else None)
        codes = list(wl[col]) if col else []
        return [str(c) for c in codes], "自选池"
    # 默认:当前选股结果
    res = STATE.get("results")
    if res is None or res.empty:
        return [], "选股结果"
    return [str(c) for c in res["code"]], "选股结果"


def _ai_ui_tags(source):
    """按来源返回该展示哪一套 AI 面板的 tag。

    盯盘页(source='watch')用自己的面板 watch_ai_*,结果就地显示,不再跳到选股页;
    其余(选股结果)用选股页的 ai_batch_*/ai_pf_* 面板。
    """
    if source == "watch":
        return {"batch_win": "watch_ai_win", "batch_title": "watch_ai_title",
                "batch_text": "watch_ai_text", "batch_report": "watch_ai_report",
                "pf_win": "watch_ai_win", "pf_title": "watch_ai_title",
                "pf_text": "watch_ai_text", "pf_disclaimer": "watch_ai_disclaimer",
                "batch_btn": None, "pf_btn": None}
    return {"batch_win": "ai_batch_win", "batch_title": "ai_batch_title",
            "batch_text": "ai_batch_text", "batch_report": "ai_batch_report",
            "pf_win": "ai_pf_win", "pf_title": "ai_pf_title",
            "pf_text": "ai_pf_text", "pf_disclaimer": "ai_pf_disclaimer",
            "batch_btn": "ai_batch_btn", "pf_btn": "ai_pf_btn"}


def on_ai_batch(source="results", sender=None, app_data=None, user_data=None):
    """对一批股票(选股结果/自选池)逐只 AI 点评,汇总成 HTML 晨报并导出。

    结果显示在对应页面自己的面板里(盯盘页不再跳到选股页)。
    """
    codes, title = _codes_from_source(source)
    T = _ai_ui_tags(source)
    if not codes:
        _set_status(f"{title}为空,先选股或加入自选再批量点评")
        return
    if not ai_configured():
        dpg.set_value(T["batch_text"], ai_config_hint())
        if T.get("batch_report") and dpg.does_item_exist(T["batch_report"]):
            dpg.set_value(T["batch_report"], "")
        if T.get("pf_disclaimer") and dpg.does_item_exist(T["pf_disclaimer"]):
            dpg.configure_item(T["pf_disclaimer"], show=False)
        dpg.configure_item(T["batch_win"], show=True)
        return
    # 控制规模,避免一次点评过多(串行 + API 限流)
    MAX_BATCH = 20
    codes = codes[:MAX_BATCH]

    dpg.configure_item(T["batch_win"], show=True)
    dpg.set_value(T["batch_title"], f"批量点评 · {title}({len(codes)} 只)")
    dpg.set_value(T["batch_text"], "正在逐只点评,请稍候...")
    if T.get("batch_report") and dpg.does_item_exist(T["batch_report"]):
        dpg.set_value(T["batch_report"], "")
    if T.get("pf_disclaimer") and dpg.does_item_exist(T["pf_disclaimer"]):
        dpg.configure_item(T["pf_disclaimer"], show=False)
    if T.get("batch_btn") and dpg.does_item_exist(T["batch_btn"]):
        dpg.configure_item(T["batch_btn"], enabled=False, label="点评中...")

    def worker():
        try:
            def pcb(d, t, code):
                if dpg.does_item_exist(T["batch_text"]):
                    dpg.set_value(T["batch_text"],
                                  f"点评进度 {d}/{t}(当前 {code})...")
            r = ai_commentary.comment_batch(codes, progress_cb=pcb)
            if not r.get("ok"):
                dpg.set_value(T["batch_text"],
                              f"批量点评失败:{r.get('error', '未知错误')}")
                return
            items = r["items"]
            # 屏内摘要:每只一行(评级/风险)
            lines = []
            ok_n = 0
            for it in items:
                if it.get("error"):
                    lines.append(f"× {it['code']} {it.get('name', '')} "
                                 f"— {it['error']}")
                else:
                    ok_n += 1
                    lines.append(f"· {it['code']} {it.get('name', '')} "
                                 f"[{it.get('industry', '')}] "
                                 f"评级:{it.get('rating') or '-'} / "
                                 f"风险:{it.get('risk') or '-'}")
            dpg.set_value(T["batch_text"], "\n".join(lines))
            # 导出 HTML 晨报
            try:
                path = exporter.export_ai_report(
                    items, title=f"AI 点评晨报 · {title}",
                    disclaimer=r.get("disclaimer", ""))
                STATE["last_report_dir"] = os.path.dirname(path)
                STATE["last_ai_report"] = path
                if dpg.does_item_exist(T["batch_report"]):
                    dpg.set_value(T["batch_report"],
                                  f"已生成晨报:{os.path.basename(path)}"
                                  f"(成功 {ok_n}/{len(items)} 只)· "
                                  f"点『打开目录』查看")
            except Exception as e:  # noqa
                if dpg.does_item_exist(T["batch_report"]):
                    dpg.set_value(T["batch_report"], f"晨报导出失败:{e}")
        except Exception as e:  # noqa
            dpg.set_value(T["batch_text"], f"批量点评异常:{e}")
        finally:
            if T.get("batch_btn") and dpg.does_item_exist(T["batch_btn"]):
                dpg.configure_item(T["batch_btn"], enabled=True,
                                   label="批量点评(晨报)")

    threading.Thread(target=worker, daemon=True).start()


def on_ai_portfolio(source="results", sender=None, app_data=None, user_data=None):
    """对一批股票(选股结果/自选池)做全局组合解读(板块集中度/估值/组合风险),流式展示。

    结果显示在对应页面自己的面板里(盯盘页不再跳到选股页)。
    """
    codes, title = _codes_from_source(source)
    T = _ai_ui_tags(source)
    if not codes:
        _set_status(f"{title}为空,先选股或加入自选再做组合解读")
        return
    if not ai_configured():
        dpg.set_value(T["pf_text"], ai_config_hint())
        dpg.configure_item(T["pf_win"], show=True)
        return

    dpg.configure_item(T["pf_win"], show=True)
    dpg.set_value(T["pf_title"], f"组合解读 · {title}({len(codes)} 只)")
    dpg.set_value(T["pf_text"], "正在汇总组合画像并生成研判(边生成边显示)...")
    if T.get("batch_report") and dpg.does_item_exist(T["batch_report"]):
        dpg.set_value(T["batch_report"], "")
    if T.get("pf_btn") and dpg.does_item_exist(T["pf_btn"]):
        dpg.configure_item(T["pf_btn"], enabled=False, label="解读中...")

    # 流式:用可变缓冲累积,回调里刷新文本
    buf = {"s": ""}

    def _on_delta(piece):
        buf["s"] += piece
        if dpg.does_item_exist(T["pf_text"]):
            dpg.set_value(T["pf_text"], buf["s"])

    def worker():
        try:
            r = ai_commentary.comment_portfolio(codes, title=title,
                                                on_delta=_on_delta)
            if not r.get("ok"):
                dpg.set_value(T["pf_text"],
                              f"组合解读失败:{r.get('error', '未知错误')}")
                return
            # 末尾补免责声明(流式正文已在 buf 里)
            dpg.set_value(T["pf_text"], r["text"])
            if T.get("pf_disclaimer") and dpg.does_item_exist(T["pf_disclaimer"]):
                dpg.set_value(T["pf_disclaimer"], r.get("disclaimer", ""))
                dpg.configure_item(T["pf_disclaimer"], show=True)
        except Exception as e:  # noqa
            dpg.set_value(T["pf_text"], f"组合解读异常:{e}")
        finally:
            if T.get("pf_btn") and dpg.does_item_exist(T["pf_btn"]):
                dpg.configure_item(T["pf_btn"], enabled=True, label="组合解读")

    threading.Thread(target=worker, daemon=True).start()


def on_ai_hold_comment(sender=None, app_data=None, user_data=None):
    """对单只【已持仓】自选股做 AI 持仓点评(结合买入成本给持有/加减仓倾向)。

    user_data = (code, buy_price)。结果流式显示在盯盘页 watch_ai_win 面板,
    带操作倾向(持有/加仓/减仓/观望)与风险彩色标签。
    """
    if not user_data:
        return
    code, buy_price = (user_data if isinstance(user_data, (tuple, list))
                       else (user_data, 0.0))
    code = str(code)
    nm = db.name_of(code) or ""
    if not ai_configured():
        dpg.set_value("watch_ai_text", ai_config_hint())
        dpg.set_value("watch_ai_title", "AI 持仓点评")
        if dpg.does_item_exist("watch_ai_report"):
            dpg.set_value("watch_ai_report", "")
        for _b in ("watch_ai_action", "watch_ai_risk"):
            if dpg.does_item_exist(_b):
                dpg.configure_item(_b, show=False)
        dpg.configure_item("watch_ai_win", show=True)
        return

    dpg.configure_item("watch_ai_win", show=True)
    try:
        bp = float(buy_price)
    except (TypeError, ValueError):
        bp = 0.0
    bp_txt = f"买入价 {bp:.2f}" if bp > 0 else "无买入价(按观察处理)"
    dpg.set_value("watch_ai_title", f"AI 持仓点评 · {code} {nm}({bp_txt})")
    if dpg.does_item_exist("watch_ai_report"):
        dpg.set_value("watch_ai_report", "")
    if dpg.does_item_exist("watch_ai_disclaimer"):
        dpg.configure_item("watch_ai_disclaimer", show=False)
    for _b in ("watch_ai_action", "watch_ai_risk"):
        if dpg.does_item_exist(_b):
            dpg.configure_item(_b, show=False)
    dpg.set_value("watch_ai_text", f"正在为 {code} {nm} 生成持仓点评(边生成边显示)...")

    def _badge(tag, text, color):
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, text)
            dpg.configure_item(tag, color=color, show=True)

    def worker():
        stream = {"buf": "", "head": f"【{code} {nm}】\n" if nm else f"【{code}】\n"}

        def _on_delta(piece):
            stream["buf"] += piece
            if dpg.does_item_exist("watch_ai_text"):
                dpg.set_value("watch_ai_text", stream["head"] + stream["buf"])

        try:
            res = ai_commentary.comment_holding(code, buy_price=bp,
                                                on_delta=_on_delta)
            if res.get("ok"):
                f = res["facts"]
                head = f"【{f['code']} {f['name']}】{f['industry']}\n"
                pnl = res.get("pnl")
                if pnl is not None:
                    head += f"买入价 {res['buy_price']:.2f} · 浮动盈亏 {pnl:+.2f}%\n"
                dpg.set_value("watch_ai_text", head + res["text"])
                if dpg.does_item_exist("watch_ai_disclaimer"):
                    dpg.set_value("watch_ai_disclaimer", res["disclaimer"])
                    dpg.configure_item("watch_ai_disclaimer", show=True)
                # 操作倾向彩色标签(加仓=红,减仓=绿,持有=黄,观望=灰;A股惯例)
                act = res.get("action")
                if act == "加仓":
                    _badge("watch_ai_action", "  操作:加仓", (230, 60, 60))
                elif act == "减仓":
                    _badge("watch_ai_action", "  操作:减仓", (40, 170, 80))
                elif act == "持有":
                    _badge("watch_ai_action", "  操作:持有", (200, 180, 90))
                elif act == "观望":
                    _badge("watch_ai_action", "  操作:观望", (150, 150, 150))
                risk = res.get("risk")
                if risk == "高":
                    _badge("watch_ai_risk", "  风险:高", (230, 60, 60))
                elif risk == "中":
                    _badge("watch_ai_risk", "  风险:中", (220, 160, 60))
                elif risk == "低":
                    _badge("watch_ai_risk", "  风险:低", (120, 180, 120))
            else:
                dpg.set_value("watch_ai_text",
                              f"持仓点评失败:\n{res.get('error', '未知错误')}")
                if dpg.does_item_exist("watch_ai_disclaimer"):
                    dpg.configure_item("watch_ai_disclaimer", show=False)
        except Exception as e:  # noqa
            dpg.set_value("watch_ai_text", f"持仓点评异常:{e}")

    threading.Thread(target=worker, daemon=True).start()


def on_open_ai_history(sender=None, app_data=None, user_data=None):
    """点评历史回看:弹窗展示 ai_commentary 表里存档的历史点评(最新在前)。"""
    if dpg.does_item_exist("ai_hist_win"):
        dpg.delete_item("ai_hist_win")
    with dpg.window(label="AI 点评历史", tag="ai_hist_win",
                    width=880, height=620, pos=(200, 70), modal=False):
        with dpg.group(horizontal=True):
            dpg.add_text("按代码筛选:")
            dpg.add_input_text(tag="ai_hist_code", width=120,
                               hint="留空=全部")
            dpg.add_button(label="查询", width=64, height=26,
                           callback=_refresh_ai_history)
            dpg.add_button(label="全部", width=64, height=26,
                           callback=lambda: (dpg.set_value("ai_hist_code", ""),
                                             _refresh_ai_history()))
            dpg.add_text("", tag="ai_hist_status", color=(255, 200, 100))
        dpg.add_separator()
        dpg.add_child_window(tag="ai_hist_box", border=False, height=-1)
    _refresh_ai_history()


def _refresh_ai_history(sender=None, app_data=None, user_data=None):
    """读取并渲染 AI 点评历史列表。"""
    if not dpg.does_item_exist("ai_hist_box"):
        return
    dpg.delete_item("ai_hist_box", children_only=True)
    code = (dpg.get_value("ai_hist_code") or "").strip() if \
        dpg.does_item_exist("ai_hist_code") else ""
    try:
        df = db.load_ai_commentary(code=code or None, limit=80)
    except Exception as e:  # noqa
        dpg.add_text(f"读取失败: {e}", parent="ai_hist_box",
                     color=(230, 120, 120))
        return
    if df is None or df.empty:
        dpg.set_value("ai_hist_status", "暂无历史(点评过股票后会自动存档)")
        dpg.add_text("还没有任何点评存档。到『选股 & K线』页点 AI点评,"
                     "或用批量点评后即可在此回看。",
                     parent="ai_hist_box", wrap=830, color=(150, 150, 150))
        return
    dpg.set_value("ai_hist_status", f"共 {len(df)} 条")

    def _rc(rating):
        return {"偏多": (230, 60, 60), "偏空": (40, 170, 80),
                "中性": (200, 180, 90)}.get(rating, (180, 180, 180))

    for _, r in df.iterrows():
        with dpg.group(parent="ai_hist_box"):
            with dpg.group(horizontal=True):
                dpg.add_text(f"{r['code']} {r.get('name', '')}",
                             color=(160, 200, 255))
                dpg.add_text(f"· {r.get('trade_date', '')}",
                             color=(150, 150, 150))
                if r.get("rating"):
                    dpg.add_text(f"评级:{r['rating']}", color=_rc(r["rating"]))
                if r.get("risk"):
                    dpg.add_text(f"风险:{r['risk']}", color=(200, 160, 90))
            dpg.add_text(str(r.get("text", "")), wrap=830,
                         color=(220, 220, 220))
            dpg.add_separator()


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
            _sel = dpg.add_selectable(label=code, span_columns=True, user_data=code,
                                      callback=_on_code_click)   # 双击代码 → 跳K线页
            dpg.bind_item_theme(_sel, "row_hover")
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
                label=code, span_columns=True, user_data=code,
                callback=_on_code_click,   # 双击代码 → 跳K线页
            )
            dpg.bind_item_theme(_sel, "row_hover")
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
    cols = ["代码", "名称", "加入日", "买入价", "现价", "今日涨跌%", "浮动盈亏%",
            "PE分位", "PB分位", "备注", "操作"]
    dpg.delete_item("watch_table", children_only=True)
    STATE["_watch_hl_row"] = None   # 行重建后旧高亮索引失效,重置避免误清/越界
    for col in cols:
        dpg.add_table_column(label=col, parent="watch_table")
    wl = db.load_watchlist()
    if wl is None or wl.empty:
        dpg.set_value("watch_status", "自选为空。可在选股结果或今日推荐里点『+自选』加入。")
        return

    spot = STATE.get("spot") if use_realtime else None
    # 陈旧判断必须【按自选票各自】判断:每只票的 stale 标记表示它自己是否用了昨收兜底
    # (盘前/午休/停牌)。不能拿全市场 any(stale) 一刀切——全市场几乎永远有停牌票,
    # 会导致所有票都被误判为陈旧、实时价永远不落库。
    total_pnl = []
    live_prices = {}   # 本次拿到的【真实时价】(非昨收兜底),批量写库持久化
    n_saved = 0        # 用到库存历史现价的只数
    n_live = 0         # 拿到真实时价的只数(用于表头标签)
    for _, r in wl.iterrows():
        code = r["code"]
        q = spot.get(code) if spot else None
        cur = None
        chg = None
        price_src = None   # live=实时/快照, saved=库存历史现价, kline=本地日线
        if q and q.get("price") is not None:
            cur = float(q["price"])
            chg = q.get("chg_pct")  # stale(盘前/停牌)时为 None
            price_src = "live"
            q_stale = bool(q.get("stale"))   # 这只票是否用了昨收兜底
            if cur > 0 and not q_stale:
                live_prices[code] = cur   # 只把真实时价(非昨收兜底)写库
                n_live += 1
        else:
            # 无实时快照:优先用库里持久化的最近现价(重开软件不丢),
            # 没有再退回本地日线收盘价
            saved = r.get("last_price") if "last_price" in r.index else None
            if saved is not None and saved == saved and float(saved) > 0:
                cur = float(saved)
                price_src = "saved"
                n_saved += 1
            else:
                kl = db.load_kline(code)
                if kl is not None and not kl.empty:
                    cur = float(kl.iloc[-1]["close"])
                    price_src = "kline"
        buy = float(r["buy_price"] or 0.0)
        # 现价有效(>0)且有买入价才算浮动盈亏,否则留空,避免拉不到价时误显示 -100%
        has_cur = cur is not None and cur > 0
        pnl = ((cur - buy) / buy * 100) if (has_cur and buy > 0) else None
        if pnl is not None:
            total_pnl.append(pnl)
        with dpg.table_row(parent="watch_table"):
            # 注:此表行内含"持仓点评/移除"按钮与"买入价"输入框,不能用
            # span_columns=True(跨列 selectable 会铺满整行并抢占点击,导致
            # 同行按钮/输入框点不动)。故此表只高亮代码格,保留行内交互。
            _sel = dpg.add_selectable(label=code, span_columns=False, user_data=code,
                                      callback=_on_code_click)   # 双击代码 → 跳K线页
            dpg.bind_item_theme(_sel, "row_hover")
            dpg.add_text(str(r["name"]))
            dpg.add_text(str(r["add_date"]))
            # 买入价:内联可编辑(加入价不一定是真实买入价)。改完即存库并重算盈亏
            dpg.add_input_float(default_value=buy, width=90, step=0,
                                format="%.2f", tag=f"watch_buy_{code}",
                                user_data=code, on_enter=True,
                                callback=_on_edit_buy_price)
            if has_cur:
                # 库存历史价(saved)标灰,提示这不是最新实时价
                dpg.add_text(f"{cur:.2f}",
                             color=(150, 150, 150) if price_src == "saved"
                             else (230, 230, 230))
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
            # PE/PB 历史分位: 先占位, 后台线程批量算完回填(见下方 _kick_watch_valuation)
            dpg.add_text("…", tag=f"watch_peval_{code}", color=(120, 120, 130))
            dpg.add_text("…", tag=f"watch_pbval_{code}", color=(120, 120, 130))
            dpg.add_text(str(r["note"]))
            with dpg.group(horizontal=True):
                dpg.add_button(label="持仓点评", width=76,
                               callback=on_ai_hold_comment,
                               user_data=(code, buy))
                dpg.add_button(label="移除", width=54,
                               callback=lambda s, a, u: _remove_watch(u),
                               user_data=code)
    # 把本次拿到的实时价持久化,重开软件后现价不再还原成旧价
    if live_prices:
        try:
            db.save_watch_prices(live_prices)
        except Exception:
            pass
    # 价格源标签:按本次自选票实际用到的源细化
    if use_realtime and spot:
        if n_live and n_saved:
            src_txt = f"实时{n_live}只/昨收或历史{n_saved}只"
        elif n_live:
            src_txt = "实时"
        else:
            src_txt = "昨收(未开盘/休市)"
    else:
        src_txt = f"上次实时价(共{n_saved}只)" if n_saved else "本地收盘"
    if total_pnl:
        avg = sum(total_pnl) / len(total_pnl)
        dpg.set_value("watch_status",
                      f"持仓 {len(total_pnl)} 只 · 平均浮动盈亏 {avg:+.2f}% · 价格源:{src_txt}")
    else:
        dpg.set_value("watch_status", f"自选 {len(wl)} 只(均为观察) · 价格源:{src_txt}")

    # 渲染完成后, 后台批量补每只持仓票的 PE/PB 历史分位(不阻塞界面)
    try:
        _codes = [str(c) for c in wl["code"].tolist()]
    except Exception:  # noqa
        _codes = []
    if _codes:
        _kick_watch_valuation(_codes)


def _kick_watch_valuation(codes):
    """后台线程: 逐只算持仓票 PE/PB 历史分位, 回填到表格对应格子。
    个股估值一次请求即出(约1s/只), 持仓票少(十几只)故串行即可, 不阻塞界面。
    只读传入的 code 列表(纯数据), 不读 UI 值; 算完用 set_value 回填。"""
    if STATE.get("_watch_val_inflight"):
        return
    STATE["_watch_val_inflight"] = True

    def _fill(code, kind, cell):
        """cell: market.stock_valuation_percentile 返回的 pe/pb 子 dict 或 None。"""
        tag = f"watch_{kind}val_{code}"
        if not dpg.does_item_exist(tag):
            return
        if not cell:
            dpg.set_value(tag, "-")
            dpg.configure_item(tag, color=(120, 120, 130))
            return
        txt, col = _val_pct_cell(cell.get("value"), cell.get("percentile"))
        dpg.set_value(tag, txt)
        dpg.configure_item(tag, color=col)

    def worker():
        try:
            for code in codes:
                # ETF 无个股 PE/PB, 跳过(显示 '-')
                try:
                    r = market_data.stock_valuation_percentile(code)
                except Exception:  # noqa
                    r = {"pe": None, "pb": None}
                _fill(code, "pe", r.get("pe"))
                _fill(code, "pb", r.get("pb"))
        finally:
            STATE["_watch_val_inflight"] = False

    threading.Thread(target=worker, daemon=True).start()


def _on_edit_buy_price(sender=None, app_data=None, user_data=None):
    """盯盘表内联修改买入价:存库并重算浮动盈亏。"""
    code = user_data
    try:
        val = round(float(app_data), 2)   # input_float 为单精度,四舍五入到分,避免 10.32→10.3199
    except Exception:
        val = 0.0
    if val < 0:
        val = 0.0
    db.update_buy_price(code, val)
    _set_status(f"已更新 {code} 买入价为 {val:.2f}")
    # 用当前价格源(实时/库存)重算,不重新拉网络
    _refresh_watchlist(use_realtime=bool(STATE.get("spot")))


def on_copy_watch_ai(sender=None, app_data=None, user_data=None):
    """把盯盘页 AI 点评面板的全文(标题+操作/风险+正文+免责)复制到系统剪贴板。"""
    parts = []
    for tag in ("watch_ai_title", "watch_ai_action", "watch_ai_risk",
                "watch_ai_report", "watch_ai_text", "watch_ai_disclaimer"):
        if dpg.does_item_exist(tag):
            v = dpg.get_value(tag)
            if v and str(v).strip():
                parts.append(str(v).strip())
    text = "\n".join(parts).strip()
    if not text:
        _set_status("暂无可复制的点评内容")
        return
    try:
        dpg.set_clipboard_text(text)
        _set_status(f"已复制点评到剪贴板({len(text)} 字)")
    except Exception as e:  # noqa
        _set_status(f"复制失败:{e}")


def _copy_ai_text(sender=None, app_data=None, user_data=None):
    """通用:把指定 AI 点评面板的文本 tags 拼接后复制到系统剪贴板。
    user_data 为该面板要复制的文本 tag 列表(标题/评级/正文/免责等,按显示顺序)。
    供选股单只点评、批量晨报、组合解读、行情大盘点评等各面板共用。"""
    tags = user_data or []
    parts = []
    for tag in tags:
        if dpg.does_item_exist(tag):
            v = dpg.get_value(tag)
            if v and str(v).strip():
                parts.append(str(v).strip())
    text = "\n".join(parts).strip()
    if not text:
        _set_status("暂无可复制的点评内容")
        return
    try:
        dpg.set_clipboard_text(text)
        _set_status(f"已复制点评到剪贴板({len(text)} 字)")
    except Exception as e:  # noqa
        _set_status(f"复制失败:{e}")


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
            if dpg.does_item_exist("kline_title_chg"):
                dpg.set_value("kline_title_chg", "")
            if dpg.does_item_exist("kline_title_suffix"):
                dpg.set_value("kline_title_suffix", "")
            STATE["intraday_code"] = None
        return

    # 昨收基准(判断涨跌上色 + 算涨跌幅):
    # 优先用分时接口返回的"真昨收"(上一交易日最后一分钟收盘)——同源、
    # 且必是真实上一交易日,不受本地日线是否已更新到最新影响。
    # 仅在接口未给出时才回退本地日线最后一根收盘(可能过时,会导致涨跌幅算错)。
    prev_close = df.attrs.get("prev_close")
    if not prev_close:
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
    dpg.set_value("kline_title", f"分时 {label}  {tdate}  现价 {last:.3f}  ")
    # 涨跌幅单独上色:红涨绿跌(相对昨收),平盘用中性灰
    if dpg.does_item_exist("kline_title_chg"):
        dpg.set_value("kline_title_chg",
                      f"{'+' if chg >= 0 else ''}{chg:.2f}%")
        if chg > 0:
            dpg.configure_item("kline_title_chg", color=RED)
        elif chg < 0:
            dpg.configure_item("kline_title_chg", color=GREEN)
        else:
            dpg.configure_item("kline_title_chg", color=(180, 180, 180))
    if dpg.does_item_exist("kline_title_suffix"):
        dpg.set_value("kline_title_suffix",
                      f"  ({'轮询中' if STATE.get('poll_on') else '已停'})")


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


# ---------- 行情页(大盘概览) ----------
def _fmt_yi(amount_yuan):
    """把成交额/市值(元)格式化为『亿』或『万亿』字符串;None/异常返回 '-'。"""
    try:
        v = float(amount_yuan)
    except Exception:  # noqa
        return "-"
    if v != v:  # NaN
        return "-"
    yi = v / 1e8
    if abs(yi) >= 10000:
        return f"{yi / 10000:.2f} 万亿"
    return f"{yi:,.0f} 亿"


def _chg_color(v):
    """涨跌幅 → 颜色(A股惯例:涨红跌绿,0 灰)。"""
    try:
        x = float(v)
    except Exception:  # noqa
        return (200, 200, 200)
    if x > 0:
        return (230, 60, 60)
    if x < 0:
        return (40, 180, 90)
    return (190, 190, 190)


def _render_market(idx_df, act, summ):
    """把拉取到的行情数据渲染进「行情」页的各控件(必须在主线程/回调里调)。"""
    # ---- 指数表 ----
    if dpg.does_item_exist("mkt_index_table"):
        dpg.delete_item("mkt_index_table", children_only=True)
        for col in ["指数", "最新价", "涨跌幅", "涨跌点", "今开",
                    "最高", "最低", "成交额"]:
            dpg.add_table_column(label=col, parent="mkt_index_table")
        if idx_df is not None and not idx_df.empty:
            for _, r in idx_df.iterrows():
                col = _chg_color(r.get("chg_pct"))
                with dpg.table_row(parent="mkt_index_table"):
                    dpg.add_text(str(r.get("name", "")))
                    dpg.add_text(f"{r.get('price', float('nan')):.2f}", color=col)
                    cp = r.get("chg_pct")
                    dpg.add_text(f"{cp:+.2f}%" if cp == cp else "-", color=col)
                    ca = r.get("chg_amt")
                    dpg.add_text(f"{ca:+.2f}" if ca == ca else "-", color=col)
                    dpg.add_text(f"{r.get('open', float('nan')):.2f}")
                    dpg.add_text(f"{r.get('high', float('nan')):.2f}")
                    dpg.add_text(f"{r.get('low', float('nan')):.2f}")
                    dpg.add_text(_fmt_yi(r.get("amount")))

    # ---- 成交额/总貌 ----
    sh_amt = (idx_df.attrs.get("sh_amount") if idx_df is not None else None)
    sz_amt = (idx_df.attrs.get("sz_amount") if idx_df is not None else None)
    tot = (idx_df.attrs.get("total_amount") if idx_df is not None else None)
    if dpg.does_item_exist("mkt_amount_text"):
        line = (f"两市成交额  {_fmt_yi(tot)}    "
                f"(沪 {_fmt_yi(sh_amt)} + 深 {_fmt_yi(sz_amt)})")
        dpg.set_value("mkt_amount_text", line)
    if dpg.does_item_exist("mkt_summary_text"):
        sh_mv = summ.get("sh_mv")   # 亿元
        sz_mv = summ.get("sz_mv")   # 元
        parts = []
        if sh_mv is not None:
            parts.append(f"沪市总市值 {sh_mv / 1e4:,.2f} 万亿")
        if sz_mv is not None:
            parts.append(f"深市总市值 {_fmt_yi(sz_mv)}")
        if summ.get("sh_listed") is not None:
            parts.append(f"沪市上市 {int(summ['sh_listed'])} 家")
        if summ.get("sz_listed") is not None:
            parts.append(f"深市股票 {int(summ['sz_listed'])} 只")
        dpg.set_value("mkt_summary_text", "    ".join(parts) if parts else "-")

    # ---- 涨跌家数/情绪 ----
    if dpg.does_item_exist("mkt_activity_group"):
        dpg.delete_item("mkt_activity_group", children_only=True)
        # 【数据一致性修复】上涨/下跌/平盘 统一采用与"涨跌幅分布直方图"同源的
        # 新浪全市场快照(STATE["mkt_dist"]),而不是乐咕 legu 源——两个源的快照
        # 时间与股票池不同,曾导致"上涨 405 / 482 / 4051"三个不一致的上涨数并存。
        # legu 源只保留它独有、且 spot 无法直接得到的字段: 涨停/跌停/真实涨停/
        # 真实跌停/停牌/活跃度。这样全页只有一个权威"上涨数"。
        dist = STATE.get("mkt_dist") or {}
        src_spot = dist.get("up") is not None and dist.get("down") is not None
        if src_spot:
            up = int(dist.get("up") or 0)
            down = int(dist.get("down") or 0)
            flat = int(dist.get("flat") or 0)
            total_stat = dist.get("total")
            src_note = f"(全市场快照口径 · 共{total_stat}只)" if total_stat else "(全市场快照口径)"
        else:
            # 分布快照还没拉回来(首次~20s),先用 legu 家数占位,拉回后自动统一
            up = int(act.get("上涨", 0) or 0)
            down = int(act.get("下跌", 0) or 0)
            flat = int(act.get("平盘", 0) or 0)
            src_note = "(乐咕口径·分布快照加载中,稍后统一)"
        zt = int(act.get("涨停", 0) or 0)
        zt_real = int(act.get("真实涨停", 0) or 0)
        dt = int(act.get("跌停", 0) or 0)
        dt_real = int(act.get("真实跌停", 0) or 0)
        halt = int(act.get("停牌", 0) or 0)
        active = act.get("活跃度")
        with dpg.group(horizontal=True, parent="mkt_activity_group"):
            dpg.add_text(f"上涨 {up}", color=(230, 60, 60))
            dpg.add_text(" / ", color=(140, 140, 140))
            dpg.add_text(f"下跌 {down}", color=(40, 180, 90))
            dpg.add_text(" / ", color=(140, 140, 140))
            dpg.add_text(f"平盘 {flat}", color=(190, 190, 190))
            dpg.add_text(f"  {src_note}", color=(120, 120, 130))
        # 中位数涨跌: 比指数更能反映"多数股票的真实体感"(指数被权重股扭曲时尤甚)。
        # 中位数与均值的差揭示分布偏态(均值>>中位数=少数大牛股拉高均值、多数在跌)。
        med = dist.get("median")
        mean = dist.get("mean")
        if med is not None:
            with dpg.group(horizontal=True, parent="mkt_activity_group"):
                dpg.add_text("中位数涨跌 ", color=(190, 190, 190))
                dpg.add_text(f"{med:+.2f}%", color=_chg_color(med))
                if mean is not None:
                    dpg.add_text("   平均涨跌 ", color=(190, 190, 190))
                    dpg.add_text(f"{mean:+.2f}%", color=_chg_color(mean))
                    # 偏态提示: 均值显著高于中位数=少数股拉高均值,多数股更弱
                    skew = mean - med
                    if abs(skew) >= 0.5:
                        if skew > 0:
                            tip = "  ⚠均值>>中位数·少数大票拉高指数,多数股更弱"
                        else:
                            tip = "  均值<<中位数·少数大票拖累指数,多数股更强"
                        dpg.add_text(tip, color=(200, 170, 110))
        with dpg.group(horizontal=True, parent="mkt_activity_group"):
            dpg.add_text(f"涨停 {zt}(真实 {zt_real})", color=(230, 60, 60))
            dpg.add_text(" / ", color=(140, 140, 140))
            dpg.add_text(f"跌停 {dt}(真实 {dt_real})", color=(40, 180, 90))
            dpg.add_text("      ", color=(140, 140, 140))
            dpg.add_text(f"停牌 {halt}", color=(190, 190, 190))
            if active is not None:
                dpg.add_text("      ", color=(140, 140, 140))
                dpg.add_text(f"赚钱效应(活跃度) {active}", color=(230, 200, 120))
        # 涨跌强弱条(上涨占比,红涨绿跌)——与上面家数同源,口径一致
        total_ud = up + down
        if total_ud > 0:
            ratio = up / total_ud
            up_share = up / (up + down + flat) if (up + down + flat) > 0 else ratio
            # 历史分位: 今日上涨占比在过去一年里处于什么冷热位置
            pct_note = ""
            try:
                pr = market_data.breadth_percentile("up_share", up_share * 100)
                if pr is not None:
                    pct = pr.get("percentile")
                    n = pr.get("samples")
                    if pct is not None and n and n >= 20:
                        if pct <= 15:
                            hot = "偏冷(恐慌区)"
                        elif pct >= 85:
                            hot = "偏热(亢奋区)"
                        else:
                            hot = "中性"
                        pct_note = (f"   历史分位 {pct:.0f}%·{hot} "
                                    f"(近{n}日样本)")
            except Exception:  # noqa
                pass
            with dpg.group(horizontal=True, parent="mkt_activity_group"):
                dpg.add_text(f"多空比 {ratio * 100:.0f}%", color=(200, 200, 200))
                dpg.add_progress_bar(default_value=ratio, width=300,
                                     overlay=f"{up}↑ / {down}↓")
                if pct_note:
                    dpg.add_text(pct_note, color=(180, 180, 130))


def _render_market_mood(vol, pool):
    """渲染『盘面量能 / 情绪温度』区(mkt_mood_group):量能对比(缩量/放量)+
    连板高度 + 涨停梯队 + 封板率 + 炸板数。
    vol = market_data.fetch_volume_compare 结果;pool = fetch_limit_pool 结果。
    任一为空则该部分显示占位,不报错。"""
    if not dpg.does_item_exist("mkt_mood_group"):
        return
    dpg.delete_item("mkt_mood_group", children_only=True)

    # ---- 第一行:量能对比 ----
    with dpg.group(horizontal=True, parent="mkt_mood_group"):
        vol = vol or {}
        state = vol.get("state")
        ratio = vol.get("ratio_pct")
        today = vol.get("today")
        avg5 = vol.get("avg5")
        # 放量=红(情绪升温) 缩量=绿 持平=灰,契合A股涨红跌绿的情绪direction
        vcol = {"放量": (230, 60, 60), "缩量": (40, 180, 90),
                "持平": (190, 190, 190)}.get(state, (190, 190, 190))
        dpg.add_text("量能", color=(140, 140, 140))
        if state and ratio is not None:
            dpg.add_text(f"{state} {ratio:+.0f}%", color=vcol)
            dpg.add_text("(较近5日均量)", color=(150, 150, 150))
        else:
            dpg.add_text("数据未就绪", color=(150, 150, 150))

    # ---- 第二行:情绪温度(连板高度 / 封板率 / 炸板) ----
    pool = pool or {}
    max_board = pool.get("max_board")
    seal = pool.get("seal_rate")
    zt_n = pool.get("zt_count")
    zb_n = pool.get("zb_count")
    with dpg.group(horizontal=True, parent="mkt_mood_group"):
        dpg.add_text("情绪", color=(140, 140, 140))
        if max_board is not None:
            dpg.add_text(f"最高 {max_board} 连板", color=(230, 60, 60))
        else:
            dpg.add_text("涨停梯队数据未就绪", color=(150, 150, 150))
        if zt_n is not None:
            dpg.add_text("   ", color=(140, 140, 140))
            dpg.add_text(f"涨停 {zt_n}", color=(230, 60, 60))
        if zb_n is not None:
            dpg.add_text(" / ", color=(140, 140, 140))
            dpg.add_text(f"炸板 {zb_n}", color=(230, 160, 60))
        if seal is not None:
            dpg.add_text(" / ", color=(140, 140, 140))
            # 封板率高=情绪强(红) 低=退潮(绿),60% 为大致分界
            scol = (230, 60, 60) if seal >= 60 else (40, 180, 90)
            dpg.add_text(f"封板率 {seal:.0f}%", color=scol)

    # ---- 第三行:涨停梯队(各连板档位家数) ----
    ladder = pool.get("ladder") or []
    first = pool.get("first_board")
    if ladder or first is not None:
        with dpg.group(horizontal=True, parent="mkt_mood_group"):
            dpg.add_text("梯队", color=(140, 140, 140))
            for b, n in ladder:   # 已按连板数降序(高标在前)
                dpg.add_text(f"{b}板×{n}", color=(230, 90, 90))
                dpg.add_text(" ", color=(140, 140, 140))
            if first is not None:
                dpg.add_text(f"首板×{first}", color=(230, 150, 120))


_BAR_THEME_CACHE = {}   # color(tuple) -> theme id, 竖直柱体背景色复用


def _bar_theme(color):
    """给竖直柱体的 child_window 生成/复用一个纯色背景 theme。"""
    key = tuple(color)
    t = _BAR_THEME_CACHE.get(key)
    if t is not None:
        return t
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, color)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 2)
    _BAR_THEME_CACHE[key] = t
    return t


def _render_updown_dist(dist):
    """渲染『涨跌幅分布直方图』(mkt_dist_group):从跌停到涨停 10 档各家数,
    竖直柱状图(涨档红、跌档绿,柱高=家数占比,柱顶标数字、柱底标档位)。
    DPG 无竖直进度条,故用染色 child_window 当柱体 + 顶部 spacer 顶到等高
    基线:每列 spacer(留白)+数字+柱体三者高度和恒为 H,故所有柱的数字顶端
    与底部标签都自动对齐。dist = fetch_updown_dist 结果。数据未就绪显示占位。"""
    if not dpg.does_item_exist("mkt_dist_group"):
        return
    dist = dist or {}
    bins = dist.get("bins") or []
    if not bins:
        # 空数据(拉取失败)进来:若柱状图已画好,保持原图不动,绝不用占位覆盖;
        # 仅当从未画过(group 为空)时才显示首次加载占位。防"有文字没图"残缺态。
        try:
            _existing = dpg.get_item_children("mkt_dist_group", 1) or []
        except Exception:  # noqa
            _existing = []
        if _existing:
            return
        dpg.delete_item("mkt_dist_group", children_only=True)
        dpg.add_text("(涨跌幅分布数据加载中… 全市场快照约 20 秒,首次稍慢)",
                     color=(150, 150, 150), parent="mkt_dist_group")
        return
    dpg.delete_item("mkt_dist_group", children_only=True)
    if dpg.does_item_exist("mkt_dist_note"):
        up = dist.get("up"); down = dist.get("down"); flat = dist.get("flat")
        dpg.set_value("mkt_dist_note",
                      f"  上涨 {up} / 平盘 {flat} / 下跌 {down} "
                      f"(共 {dist.get('total')} 只)")
    max_n = max((n for _, n in bins), default=0) or 1
    H = 130          # 柱区最大高度(px)
    BW = 46          # 柱宽
    COLW = 62        # 每列宽度(容纳档位标签,让柱间距均匀)
    with dpg.group(horizontal=True, parent="mkt_dist_group"):
        for lab, n in bins:
            # 涨档红、跌档绿(涨停/涨x = 红系,跌停/跌x = 绿系)
            is_up = lab.startswith("涨")
            col = (230, 60, 60) if is_up else (40, 180, 90)
            ratio = n / max_n
            bar_h = max(2, int(round(H * ratio)))
            pad_h = max(0, H - bar_h)      # 顶部留白,把柱压到统一基线
            with dpg.group():              # 一列: 留白+数字+柱体+标签, 总高恒定
                dpg.add_spacer(height=pad_h)
                dpg.add_text(str(n), color=(210, 210, 210))
                cw = dpg.add_child_window(width=BW, height=bar_h,
                                          border=False, no_scrollbar=True)
                dpg.bind_item_theme(cw, _bar_theme(col))
                dpg.add_text(lab, color=col)
            dpg.add_spacer(width=max(2, COLW - BW))


_BOARD_SORT_KEYS = {
    "chg_pct": "涨跌幅",
    "net_inflow": "净流入",
    "amount": "成交额",
}


def _persist_breadth(dist, pool, idx_df):
    """把当日盘面宽度快照落库(每交易日一条, 盘中反复写只更新当天),
    供『历史分位』和后续研判使用。dist=涨跌幅分布(spot口径), pool=涨停池,
    idx_df=指数行情(取成交额)。任一缺失该字段置空, 不阻断。"""
    dist = dist or {}
    up = dist.get("up"); down = dist.get("down"); flat = dist.get("flat")
    if up is None or down is None:
        return  # 没有权威广度数据就不落库(避免脏数据污染分位基准)
    total = (up or 0) + (down or 0) + (flat or 0)
    up_share = (up / total * 100) if total > 0 else None
    pool = pool or {}
    act = STATE.get("mkt_act") or {}
    vol = STATE.get("mkt_vol") or {}
    # 成交额(亿元): 优先两市合计
    amt_yi = None
    try:
        tot = idx_df.attrs.get("total_amount") if idx_df is not None else None
        if tot:
            amt_yi = round(float(tot) / 1e8, 1)
    except Exception:  # noqa
        pass
    dt_real = act.get("真实跌停")
    try:
        dt_real = int(dt_real) if dt_real is not None else None
    except Exception:  # noqa
        dt_real = None
    row = {
        "up_share": round(up_share, 2) if up_share is not None else None,
        "up_cnt": int(up), "down_cnt": int(down),
        "flat_cnt": int(flat or 0),
        "limit_up": pool.get("zt_count"),
        "limit_down": dt_real,
        "zb_cnt": pool.get("zb_count"),
        "seal_rate": pool.get("seal_rate"),
        "max_board": pool.get("max_board"),
        "amount_yi": amt_yi,
        "vol_ratio": vol.get("ratio_pct"),
    }
    day = datetime.now().strftime("%Y-%m-%d")
    try:
        db.save_market_breadth(day, row)
    except Exception:  # noqa
        pass


def _render_market_verdict():
    """渲染顶部『盘面一句话结论』(mkt_verdict_group)。
    综合涨跌广度/涨跌停/量能/封板率/高低切, 用规则引擎生成一行定性结论,
    按基调上色(偏空红、偏多按A股惯例也是红但语义不同, 这里中性灰、偏空橙红、
    偏多绿——避免和涨跌颜色混淆, 结论区改用: 偏多=金、偏空=橙红、中性=灰)。"""
    if not dpg.does_item_exist("mkt_verdict_group"):
        return
    dpg.delete_item("mkt_verdict_group", children_only=True)
    try:
        verdict = market_data.build_market_verdict(
            STATE.get("mkt_dist"), STATE.get("mkt_act"),
            STATE.get("mkt_pool"), STATE.get("mkt_vol"),
            STATE.get("mkt_board_df"))
    except Exception:  # noqa
        verdict = None
    if not verdict or not verdict.get("tags"):
        dpg.add_text("盘面结论生成中… (需涨跌分布快照, 首次约20秒)",
                     color=(150, 150, 150), parent="mkt_verdict_group")
        return
    # 分段标签分色: bull=金(230,200,120) bear=橙红(235,110,70) neutral=浅灰
    tone_col = {"bull": (230, 200, 120), "bear": (235, 110, 70),
                "neutral": (185, 195, 210), "warn": (235, 160, 60)}
    with dpg.group(horizontal=True, parent="mkt_verdict_group"):
        dpg.add_text("今日盘面:", color=(120, 220, 160))
        first = True
        for text, tone in verdict["tags"]:
            if not first:
                dpg.add_text("|", color=(90, 95, 105))
            dpg.add_text(text, color=tone_col.get(tone, (185, 195, 210)))
            first = False


def _board_diag_score(df):
    """给行业板块算综合强弱分并筛出两端(高潜力/高风险)。
    综合分 = 当日涨幅(40%) + 资金净额(30%) + 板块内上涨占比广度(30%),
    三项各自 min-max 归一化到 [0,1] 再加权; 三项都秒回可算(无需历史)。
    返回 (up_df, dn_df):按综合分降序的前 6 / 升序的前 6(最弱在前)。"""
    import pandas as _pd
    d = df.copy()

    def _norm(s):
        s = _pd.to_numeric(s, errors="coerce")
        lo, hi = s.min(), s.max()
        if not (hi - lo) or hi != hi or lo != lo:
            return s * 0 + 0.5
        return (s - lo) / (hi - lo)

    # 广度: 板块内上涨占比 up/(up+down), 缺失取中性 0.5
    if "up" in d.columns and "down" in d.columns:
        tot = (d["up"].fillna(0) + d["down"].fillna(0)).replace(0, float("nan"))
        d["breadth"] = (d["up"].fillna(0) / tot).fillna(0.5)
    else:
        d["breadth"] = 0.5

    chg_n = _norm(d["chg_pct"]).fillna(0.5)
    if "net_inflow" in d.columns and d["net_inflow"].notna().any():
        net_n = _norm(d["net_inflow"]).fillna(0.5)
    else:
        net_n = chg_n * 0 + 0.5
    d["score"] = 0.4 * chg_n + 0.3 * net_n + 0.3 * d["breadth"]

    d = d.sort_values("score", ascending=False).reset_index(drop=True)
    up = d.head(6)
    dn = d.tail(6).sort_values("score", ascending=True)
    return up, dn


def _render_board_diagnosis(df):
    """渲染『板块强弱·资金诊断』(mkt_board_diag_group):复用行业板块数据,
    用综合评分(当日涨幅+资金净额+板块内广度)筛出两端——上栏『高潜力』(综合
    最强)、下栏『高风险』(综合最弱)。多日趋势(5/20日)与量比由慢速线程回填。

    口径:当日涨幅+资金净额+板块内涨跌广度(均秒回);5日/20日涨跌与量比来自
    同花顺行业指数日线(慢速回填,未就绪显示…)。不含PE/PB历史分位(本地无)。"""
    if not dpg.does_item_exist("mkt_board_diag_group"):
        return
    dpg.delete_item("mkt_board_diag_group", children_only=True)
    if df is None or getattr(df, "empty", True):
        dpg.add_text("(板块数据加载中…)", color=(150, 150, 150),
                     parent="mkt_board_diag_group")
        return

    up, dn = _board_diag_score(df)
    # 入选两端板块名存 STATE, 供慢速线程拉多日趋势/量比
    STATE["mkt_diag_names"] = (up["name"].tolist() + dn["name"].tolist())
    trend = STATE.get("mkt_board_trend") or {}

    def _vol_tag(vr):
        """量比 → (标记文字, 颜色)。放量橙 / 缩量灰绿 / 未回填空。"""
        if vr is None or vr != vr:
            return "", (120, 120, 120)
        if vr >= 1.15:
            return "放", (235, 150, 60)
        if vr <= 0.85:
            return "缩", (120, 165, 140)
        return "平", (150, 150, 150)

    def _col_block(title, sub, rows, accent, is_up):
        with dpg.group():
            dpg.add_text(title, color=accent)
            dpg.add_text(sub, color=(130, 130, 130))
            with dpg.table(header_row=True, resizable=False, width=430,
                           policy=dpg.mvTable_SizingFixedFit,
                           borders_innerH=False, borders_outerH=True,
                           borders_innerV=False, borders_outerV=True):
                dpg.add_table_column(label="板块", width_fixed=True,
                                     init_width_or_weight=112)
                dpg.add_table_column(label="当日", width_fixed=True,
                                     init_width_or_weight=58)
                dpg.add_table_column(label="5日", width_fixed=True,
                                     init_width_or_weight=56)
                dpg.add_table_column(label="20日", width_fixed=True,
                                     init_width_or_weight=58)
                dpg.add_table_column(label="广度", width_fixed=True,
                                     init_width_or_weight=54)
                dpg.add_table_column(label="净额", width_fixed=True,
                                     init_width_or_weight=70)
                if rows is None or len(rows) == 0:
                    with dpg.table_row():
                        dpg.add_text("—", color=(150, 150, 150))
                        for _ in range(5):
                            dpg.add_text("")
                for _, r in rows.iterrows():
                    nm = str(r.get("name", ""))
                    cp = r.get("chg_pct")
                    ni = r.get("net_inflow")
                    br = r.get("breadth")
                    tr = trend.get(nm, {}) or {}
                    c5 = tr.get("chg_5d")
                    c20 = tr.get("chg_20d")
                    vr = tr.get("vol_ratio")
                    vtag, vcol = _vol_tag(vr)
                    with dpg.table_row():
                        with dpg.group(horizontal=True, horizontal_spacing=3):
                            dpg.add_text(nm[:5], color=_chg_color(cp))
                            if vtag:
                                dpg.add_text(vtag, color=vcol)
                        dpg.add_text(f"{cp:+.1f}%" if cp == cp else "-",
                                     color=_chg_color(cp))
                        dpg.add_text(f"{c5:+.1f}%" if c5 is not None else "…",
                                     color=_chg_color(c5) if c5 is not None
                                     else (110, 110, 110))
                        dpg.add_text(f"{c20:+.1f}%" if c20 is not None else "…",
                                     color=_chg_color(c20) if c20 is not None
                                     else (110, 110, 110))
                        # 广度: 上涨占比%, >50 偏红 <50 偏绿
                        if br == br:
                            bp = br * 100
                            bcol = (210, 90, 90) if bp >= 50 else (90, 190, 120)
                            dpg.add_text(f"{bp:.0f}%", color=bcol)
                        else:
                            dpg.add_text("-", color=(130, 130, 130))
                        dpg.add_text(_fmt_yi(ni), color=_chg_color(ni))

    with dpg.group(parent="mkt_board_diag_group", horizontal=True,
                   horizontal_spacing=18):
        _col_block("▲ 高潜力(综合最强)", "涨幅40%+资金30%+广度30%", up,
                   (230, 90, 90), True)
        _col_block("▼ 高风险(综合最弱)", "跌幅+资金出逃+内部普跌", dn,
                   (90, 200, 130), False)
    dpg.add_text("综合分=当日涨幅+资金净额(全单口径)+板块内涨跌广度;"
                 "5日/20日为行业指数日线趋势,量比放/缩标记(慢速回填,首次稍慢);"
                 "不含PE/PB历史分位(本地数据源无)。仅反映资金与趋势,非投资建议",
                 color=(120, 120, 120), wrap=560,
                 parent="mkt_board_diag_group")


def _val_pct_cell(value, pct):
    """把 PE/PB 中位数 + 历史分位格式化成一格文本与颜色。
    显示形如 '12.3(35%)': 括号内是历史分位。
    颜色语义(估值角度, 与涨跌色分开): 分位低=便宜=暖色(金/黄), 分位高=贵=冷色(青),
    中间=浅灰。无估值显示 '-', 有估值但分位未就绪(历史不足)显示 '12.3(…)'。"""
    if value is None or value != value:
        return "-", (120, 120, 130)
    if pct is None:
        return f"{value:.1f}(…)", (150, 150, 155)
    # 分位配色: <30 偏便宜(金), >70 偏贵(青), 其余浅灰
    if pct <= 30:
        col = (235, 190, 90)      # 金: 历史低位, 相对便宜
    elif pct >= 70:
        col = (90, 200, 210)      # 青: 历史高位, 相对贵
    else:
        col = (185, 185, 190)
    return f"{value:.1f}({pct:.0f}%)", col


def _render_board(df):
    """渲染热门板块表(已按 STATE['mkt_board_sort'] 排序)。"""
    if not dpg.does_item_exist("mkt_board_table"):
        return
    dpg.delete_item("mkt_board_table", children_only=True)
    for col in ["排名", "板块", "涨跌幅", "成交额", "净额(全单)",
                "涨/跌家数", "PE分位", "PB分位", "领涨股"]:
        dpg.add_table_column(label=col, parent="mkt_board_table")
    if df is None or df.empty:
        return
    key = STATE.get("mkt_board_sort", "chg_pct")
    if key in df.columns:
        asc = False  # 涨幅/净流入/成交额均降序看头部
        df = df.sort_values(key, ascending=asc).reset_index(drop=True)
    # 记录渲染顺序对应的板块名 → 供定时刷新时按位置原地更新数值(不重建表、不重排序)
    names_order = []
    for i, r in df.iterrows():
        name = str(r.get("name", ""))
        names_order.append(name)
        cp = r.get("chg_pct")
        col = _chg_color(cp)
        with dpg.table_row(parent="mkt_board_table"):
            dpg.add_text(str(i + 1), color=(150, 150, 150))
            dpg.add_text(name, color=col, tag=f"bcell{i}_name")
            dpg.add_text(f"{cp:+.2f}%" if cp == cp else "-", color=col,
                         tag=f"bcell{i}_chg")
            dpg.add_text(_fmt_yi(r.get("amount")), tag=f"bcell{i}_amt")
            ni = r.get("net_inflow")
            dpg.add_text(_fmt_yi(ni), color=_chg_color(ni), tag=f"bcell{i}_net")
            up = r.get("up")
            down = r.get("down")
            up_s = str(int(up)) if up == up else "-"
            down_s = str(int(down)) if down == down else "-"
            with dpg.group(horizontal=True, horizontal_spacing=0):
                dpg.add_text(up_s, color=(230, 60, 60), tag=f"bcell{i}_up")
                dpg.add_text("/", color=(140, 140, 140))
                dpg.add_text(down_s, color=(40, 180, 90), tag=f"bcell{i}_down")
            # PE/PB 历史分位(慢速回填, 从 STATE 缓存读; 未就绪显示 …)
            val = (STATE.get("mkt_sector_val") or {}).get(name) or {}
            pe_txt, pe_col = _val_pct_cell(val.get("pe"), val.get("pe_pct"))
            pb_txt, pb_col = _val_pct_cell(val.get("pb"), val.get("pb_pct"))
            dpg.add_text(pe_txt, color=pe_col, tag=f"bcell{i}_pepct")
            dpg.add_text(pb_txt, color=pb_col, tag=f"bcell{i}_pbpct")
            lc = r.get("leader_chg")
            with dpg.group(horizontal=True, horizontal_spacing=4):
                dpg.add_text(str(r.get("leader", "")), color=(200, 200, 200),
                             tag=f"bcell{i}_leader")
                dpg.add_text(f"{lc:+.2f}%" if lc == lc else "",
                             color=_chg_color(lc), tag=f"bcell{i}_lc")
    STATE["mkt_board_names"] = names_order


def _update_board_values(df):
    """定时刷新用:只原地更新板块表已有行的数值/颜色,不删表、不重排序。
    这样用户滚动到列表任意位置时刷新,滚动位置保持不变(体感不跳)。
    按板块名匹配到重建时记录的行位置;若板块集合发生变化(新增/消失),
    则退回完整重建 _render_board(此时重置滚动可接受,属少见情况)。"""
    if not dpg.does_item_exist("mkt_board_table"):
        return
    names_order = STATE.get("mkt_board_names")
    if not names_order or df is None or df.empty:
        _render_board(df)
        return
    by_name = {str(r.get("name", "")): r for _, r in df.iterrows()}
    # 板块集合变了 → 结构不同,重建
    if set(by_name.keys()) != set(names_order):
        _render_board(df)
        return
    for i, name in enumerate(names_order):
        r = by_name.get(name)
        if r is None:
            continue
        cp = r.get("chg_pct")
        col = _chg_color(cp)
        _set_cell(f"bcell{i}_name", name, col)
        _set_cell(f"bcell{i}_chg",
                  f"{cp:+.2f}%" if cp == cp else "-", col)
        _set_cell(f"bcell{i}_amt", _fmt_yi(r.get("amount")))
        ni = r.get("net_inflow")
        _set_cell(f"bcell{i}_net", _fmt_yi(ni), _chg_color(ni))
        up = r.get("up")
        down = r.get("down")
        _set_cell(f"bcell{i}_up",
                  str(int(up)) if up == up else "-")
        _set_cell(f"bcell{i}_down",
                  str(int(down)) if down == down else "-")
        # PE/PB 分位(从 STATE 缓存读; 慢速回填线程负责更新缓存)
        val = (STATE.get("mkt_sector_val") or {}).get(name) or {}
        pe_txt, pe_col = _val_pct_cell(val.get("pe"), val.get("pe_pct"))
        pb_txt, pb_col = _val_pct_cell(val.get("pb"), val.get("pb_pct"))
        _set_cell(f"bcell{i}_pepct", pe_txt, pe_col)
        _set_cell(f"bcell{i}_pbpct", pb_txt, pb_col)
        _set_cell(f"bcell{i}_leader", str(r.get("leader", "")))
        lc = r.get("leader_chg")
        _set_cell(f"bcell{i}_lc",
                  f"{lc:+.2f}%" if lc == lc else "", _chg_color(lc))


def _set_cell(tag, value, color=None):
    """原地更新单个单元格文本(可选颜色);控件不存在则跳过。"""
    if not dpg.does_item_exist(tag):
        return
    dpg.set_value(tag, value)
    if color is not None:
        dpg.configure_item(tag, color=color)


def on_sort_board(sender=None, app_data=None, user_data=None):
    """切换板块排序字段并重绘(user_data 为排序键)。"""
    key = user_data or "chg_pct"
    STATE["mkt_board_sort"] = key
    for k in _BOARD_SORT_KEYS:
        tag = f"mkt_board_sort_{k}"
        if dpg.does_item_exist(tag):
            dpg.bind_item_theme(tag, "period_on" if k == key else 0)
    _render_board(STATE.get("mkt_board_df"))


def _fetch_market_all():
    """后台拉取行情主数据,返回 (idx_df, activity_dict, summary_dict, board_df,
    vol_dict, pool_dict)。这些都相对快(<5s),可进 10s 定时刷新循环。
    涨跌幅分布(全市场快照~20s)不在此列,由独立慢速线程 _mkt_dist_worker 处理。"""
    idx_df = market_data.fetch_index_spot()
    act = market_data.fetch_market_activity()
    summ = market_data.fetch_market_summary()
    board = market_data.fetch_industry_board()
    # 量能对比: 今日量与5日基准都在数据层从新浪日线取(同为"股"口径,
    # 规避实时快照"沪按手/深按股"的单位陷阱), 此处无需再传今日量。
    try:
        vol = market_data.fetch_volume_compare()
    except Exception:  # noqa
        vol = {}
    try:
        pool = market_data.fetch_limit_pool()   # 涨停/炸板池(~4s, 带45s缓存)
    except Exception:  # noqa
        pool = {}
    return idx_df, act, summ, board, vol, pool


def _kick_updown_dist():
    """起一个一次性慢速线程拉『涨跌幅分布』(全市场快照~20s)并渲染。
    带一个 in-flight 去重标记,避免定时刷新每轮都并发起多个全市场快照拉取
    (数据层本身也有 60s TTL 缓存兜底)。"""
    if STATE.get("mkt_dist_inflight"):
        return

    def worker():
        STATE["mkt_dist_inflight"] = True
        try:
            dist = market_data.fetch_updown_dist()
            if not (dist and dist.get("bins")):
                # 本轮全市场快照拉取失败/全被判 stale → bins 为空。此时【绝不能】用空数据
                # 覆盖已画好的柱状图(否则出现"note 有旧数字、柱图却变加载中占位"的残缺态)。
                # 已有成功分布则原样保留、直接跳过本轮;从未成功过才渲染首次占位提示。
                prev = STATE.get("mkt_dist")
                if not (prev and prev.get("bins")):
                    _render_updown_dist(dist)
                return
            STATE["mkt_dist"] = dist
            _render_updown_dist(dist)
            # 分布(spot 全市场快照)是"上涨/下跌家数"的权威口径来源。它比涨跌家数区
            # 慢~20s 回填,回来后重绘一次涨跌家数区,把首屏用 legu 占位的数字统一成
            # spot 口径(消除"405/482"两个上涨数不一致),并落一条当日盘面宽度历史。
            act = STATE.get("mkt_act")
            summ = STATE.get("mkt_summ")
            idx_df = STATE.get("mkt_idx_df")
            if act is not None:
                try:
                    _render_market(idx_df, act, summ)
                except Exception:  # noqa
                    pass
            try:
                _persist_breadth(dist, STATE.get("mkt_pool"), idx_df)
            except Exception:  # noqa
                pass
            # 分布回来后, 结论才有权威广度口径, 重绘一次盘面速读
            try:
                _render_market_verdict()
            except Exception:  # noqa
                pass
        except Exception as e:  # noqa
            import traceback
            traceback.print_exc()
        finally:
            STATE["mkt_dist_inflight"] = False

    threading.Thread(target=worker, daemon=True).start()


def _kick_board_trend():
    """起一个一次性慢速线程, 对诊断表已入选的两端板块(约12个)拉多日趋势/量比
    (同花顺行业指数日线, 每个~1s, 带按自然日缓存), 回填后重绘诊断表。
    带 in-flight 去重标记, 避免定时刷新每轮并发重拉; 历史当天不变故缓存后极快。"""
    if STATE.get("mkt_trend_inflight"):
        return
    names = STATE.get("mkt_diag_names") or []
    if not names:
        return

    def worker():
        STATE["mkt_trend_inflight"] = True
        try:
            trend = market_data.fetch_board_trend(names)
            if trend:
                STATE["mkt_board_trend"] = {
                    **(STATE.get("mkt_board_trend") or {}), **trend}
                # 用最新板块数据重绘诊断表(把回填的趋势/量比显示出来)
                board = STATE.get("mkt_board_df")
                if board is not None:
                    _render_board_diagnosis(board)
        except Exception:  # noqa
            import traceback
            traceback.print_exc()
        finally:
            STATE["mkt_trend_inflight"] = False

    threading.Thread(target=worker, daemon=True).start()


def _kick_sector_val():
    """起慢速线程回填热门板块的 PE/PB 中位数 + 历史分位(不阻塞主表10s刷新)。
    两步:
      1) 当日中位数: 对当前板块表里的板块一次性算成分股 PE/PB 中位数(东财, 按天缓存),
         写 STATE['mkt_sector_val'] 并落库当日快照。首次~90s, 之后走缓存极快。
      2) 历史分位: 对每个板块从 SQLite 读历史算分位; 历史不足(<20天)的板块, 挑
         排名靠前的 1 个做一次性历史回填(每个~60s, 串行, 本会话去重), 分位随天数积累。
    分位是慢变量, 全程只读缓存/DB, 主表10s刷新不受影响。"""
    if STATE.get("mkt_sector_val_inflight"):
        return
    board = STATE.get("mkt_board_df")
    if board is None or board.empty or "name" not in board.columns:
        return
    names = [str(n) for n in board["name"].tolist()]

    def worker():
        STATE["mkt_sector_val_inflight"] = True
        try:
            # --- 步骤1: 当日 PE/PB 中位数(按天缓存, 落库) ---
            vals = market_data.fetch_sector_valuation(names)
            cache = dict(STATE.get("mkt_sector_val") or {})
            for nm, v in vals.items():
                rec = dict(cache.get(nm) or {})
                rec["pe"] = v.get("pe")
                rec["pb"] = v.get("pb")
                cache[nm] = rec
            STATE["mkt_sector_val"] = cache

            # --- 步骤2: 历史分位(读DB); 历史不足的挑排名靠前1个回填 ---
            need_backfill = None
            for nm in names:
                rec = dict((STATE.get("mkt_sector_val") or {}).get(nm) or {})
                pe = rec.get("pe")
                pb = rec.get("pb")
                pe_p = db.sector_val_percentile(nm, "pe", pe) if pe else None
                pb_p = db.sector_val_percentile(nm, "pb", pb) if pb else None
                rec["pe_pct"] = pe_p["percentile"] if pe_p else None
                rec["pb_pct"] = pb_p["percentile"] if pb_p else None
                cache2 = dict(STATE.get("mkt_sector_val") or {})
                cache2[nm] = rec
                STATE["mkt_sector_val"] = cache2
                # 历史不足且本会话没回填过 -> 记下第一个待回填板块
                if (pe or pb) and pe_p is None and pb_p is None \
                        and nm not in STATE.get("mkt_sector_bf_done", set()) \
                        and need_backfill is None:
                    need_backfill = nm

            # 重绘一次板块表, 把估值分位显示出来
            board2 = STATE.get("mkt_board_df")
            if board2 is not None:
                try:
                    _render_board(board2)
                except Exception:  # noqa
                    pass
        except Exception:  # noqa
            import traceback
            traceback.print_exc()
        finally:
            STATE["mkt_sector_val_inflight"] = False

        # --- 历史回填(独立串行线程, 一次一个, 慢~60s, 不与上面并发) ---
        if need_backfill and not STATE.get("mkt_sector_bf_inflight"):
            _kick_sector_backfill(need_backfill)

    threading.Thread(target=worker, daemon=True).start()


def _kick_sector_backfill(name):
    """对单个板块一次性回填过去一年 PE/PB 历史(慢~60s), 完成后重算该板块分位并重绘。
    串行执行(一次只回填一个板块), 本会话去重, 避免拖垮网络与反复回填。"""
    if STATE.get("mkt_sector_bf_inflight"):
        return

    def worker():
        STATE["mkt_sector_bf_inflight"] = True
        try:
            n = market_data.backfill_sector_valuation_history(name, days=300)
            done = set(STATE.get("mkt_sector_bf_done") or set())
            done.add(name)   # 无论成功与否都标记, 避免本会话反复重试同一个
            STATE["mkt_sector_bf_done"] = done
            if n > 0:
                rec = dict((STATE.get("mkt_sector_val") or {}).get(name) or {})
                pe = rec.get("pe")
                pb = rec.get("pb")
                pe_p = db.sector_val_percentile(name, "pe", pe) if pe else None
                pb_p = db.sector_val_percentile(name, "pb", pb) if pb else None
                rec["pe_pct"] = pe_p["percentile"] if pe_p else None
                rec["pb_pct"] = pb_p["percentile"] if pb_p else None
                cache = dict(STATE.get("mkt_sector_val") or {})
                cache[name] = rec
                STATE["mkt_sector_val"] = cache
                board = STATE.get("mkt_board_df")
                if board is not None:
                    try:
                        _render_board(board)
                    except Exception:  # noqa
                        pass
        except Exception:  # noqa
            import traceback
            traceback.print_exc()
        finally:
            STATE["mkt_sector_bf_inflight"] = False

    threading.Thread(target=worker, daemon=True).start()


def on_refresh_market(sender=None, app_data=None, user_data=None):
    """刷新「行情」页数据(后台线程拉取,回主线程渲染,不阻塞界面)。"""
    if dpg.does_item_exist("mkt_status"):
        dpg.set_value("mkt_status", "正在拉取大盘行情...")

    def worker():
        try:
            idx_df, act, summ, board, vol, pool = _fetch_market_all()
        except Exception as e:  # noqa
            if dpg.does_item_exist("mkt_status"):
                dpg.set_value("mkt_status", f"行情拉取失败: {e}")
            return
        try:
            _render_market(idx_df, act, summ)
            _render_market_mood(vol, pool)
            STATE["mkt_board_df"] = board
            STATE["mkt_idx_df"] = idx_df
            STATE["mkt_act"] = act
            STATE["mkt_summ"] = summ
            STATE["mkt_vol"] = vol
            STATE["mkt_pool"] = pool
            _render_board(board)
            _render_board_diagnosis(board)
            _render_market_verdict()
            # 多日趋势/量比(同花顺行业指数日线)对入选两端板块慢速回填
            _kick_board_trend()
            # 板块 PE/PB 中位数+历史分位(慢速回填, 按天缓存, 不阻塞主表)
            _kick_sector_val()
            ts = act.get("统计日期") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tag = "(定时刷新中)" if STATE.get("mkt_poll_on") else ""
            if dpg.does_item_exist("mkt_status"):
                dpg.set_value("mkt_status", f"已更新 · 数据时间 {ts} {tag}")
        except Exception as e:  # noqa
            if dpg.does_item_exist("mkt_status"):
                dpg.set_value("mkt_status", f"行情渲染失败: {e}")

    threading.Thread(target=worker, daemon=True).start()
    # 涨跌幅分布(全市场快照慢)单独起一次慢速拉取, 不阻塞上面的主刷新
    _kick_updown_dist()


def _selected_is_market(sel) -> bool:
    """判断某个 tab 选中值 sel 是否指向『行情』tab。
    兼容不同 DPG 版本:sel 可能是 int id,也可能是 str 别名。
    - int:直接与 get_alias_id('tab_market') 比;
    - str:先与别名字符串 'tab_market' 比,再尝试解析成 id 比。"""
    if sel is None or sel == 0:
        return False
    try:
        mid = dpg.get_alias_id("tab_market")
    except Exception:
        mid = None
    # str 别名直接比对
    if isinstance(sel, str):
        if sel == "tab_market":
            return True
        try:
            return dpg.get_alias_id(sel) == mid
        except Exception:
            return False
    # int id 比对
    try:
        return int(sel) == int(mid)
    except Exception:
        return False


def _on_main_tab_changed(sender=None, app_data=None, user_data=None):
    """主标签栏切换回调(主线程触发)。把『当前是否停在行情页』记进 STATE,
    供定时刷新线程读取。app_data 为当前选中 tab 的 id/别名。
    注意:UI 值读取非线程安全,所以只在此主线程回调里读一次并缓存,
    后台线程只读 STATE 这个纯字典。"""
    try:
        STATE["active_is_market"] = _selected_is_market(app_data)
    except Exception:
        pass


def _is_market_tab_active() -> bool:
    """当前是否正停在『行情』tab(供定时刷新线程调用)。
    只读 STATE 缓存的纯布尔值,不在后台线程碰 UI(线程安全)。"""
    return bool(STATE.get("active_is_market"))


def _mkt_poll_worker():
    """行情页定时刷新线程:每 MKT_POLL_INTERVAL 秒拉一次并重绘。
    仅当界面正停在『行情』tab 时才真正拉取;切到其他 tab 时静默跳过本轮,
    避免用户在别的界面时后台仍在发网络请求、刷新看不到的页面。"""
    while STATE.get("mkt_poll_on"):
        if _is_market_tab_active():
            try:
                idx_df, act, summ, board, vol, pool = _fetch_market_all()
                _render_market(idx_df, act, summ)
                _render_market_mood(vol, pool)
                STATE["mkt_board_df"] = board
                STATE["mkt_idx_df"] = idx_df
                STATE["mkt_act"] = act
                STATE["mkt_summ"] = summ
                STATE["mkt_vol"] = vol
                STATE["mkt_pool"] = pool
                # 定时刷新只原地更新板块表数值,不重建表/不重排序,
                # 保持用户当前滚动位置(体感不跳)
                _update_board_values(board)
                _render_board_diagnosis(board)
                _render_market_verdict()
                # 涨跌幅分布慢(全市场快照~20s),独立慢速线程拉,不拖累本轮
                _kick_updown_dist()
                # 多日趋势/量比对入选两端板块慢速回填
                _kick_board_trend()
                # 板块 PE/PB 中位数+历史分位(慢速回填, 按天缓存)
                _kick_sector_val()
                ts = act.get("统计日期") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if dpg.does_item_exist("mkt_status"):
                    dpg.set_value("mkt_status", f"已更新 · 数据时间 {ts} (定时刷新中)")
            except Exception as e:  # noqa
                # 拉取/渲染失败也显示到界面,别只 print 到看不见的控制台
                import traceback
                traceback.print_exc()
                if dpg.does_item_exist("mkt_status"):
                    dpg.set_value("mkt_status", f"定时刷新失败:{e} ({datetime.now().strftime('%H:%M:%S')})")
        else:
            # 不在行情页,跳过本轮(把原因显示出来,便于排查"勾了却不刷新")
            if dpg.does_item_exist("mkt_status"):
                cur = dpg.get_value("mkt_status") or ""
                if "已暂停" not in cur:
                    dpg.set_value("mkt_status", "定时刷新已暂停(当前不在行情页,切回自动恢复)")
        for _ in range(MKT_POLL_INTERVAL * 2):
            if not STATE.get("mkt_poll_on"):
                break
            _time.sleep(0.5)


def on_toggle_mkt_poll(sender=None, app_data=None, user_data=None):
    """行情页『盘中定时刷新』开关回调。app_data 为 checkbox 新值。"""
    want = bool(app_data)
    if want and not STATE.get("mkt_poll_on"):
        # 关键:这个复选框本身就在行情页上,用户能勾到它,当下必然停在行情页。
        # 所以勾选即直接置 True——不去读 tab_bar 的选中值(其类型在不同 DPG
        # 版本 int/str 不一,硬比较易恒 False,正是之前"勾了也不刷新"的根因)。
        # 之后切走由 _on_main_tab_changed 回调更新为 False,切回再置回 True。
        STATE["active_is_market"] = True
        STATE["mkt_poll_on"] = True
        t = threading.Thread(target=_mkt_poll_worker, daemon=True)
        STATE["mkt_poll_thread"] = t
        t.start()
    elif not want:
        STATE["mkt_poll_on"] = False


def _build_market_snapshot():
    """把行情页已拉取的指数/情绪/总貌/板块打包成 commentary.comment_market 所需
    的 snapshot(纯读 STATE,不发网络请求,保证点评与屏幕显示同源)。
    数据未就绪时返回 None。"""
    idx_df = STATE.get("mkt_idx_df")
    act = STATE.get("mkt_act")
    board = STATE.get("mkt_board_df")
    if (idx_df is None or getattr(idx_df, "empty", True)) and not act:
        return None

    snap = {}
    # 指数
    indexes = []
    if idx_df is not None and not idx_df.empty:
        for _, r in idx_df.iterrows():
            indexes.append({"name": r.get("name", ""),
                            "chg_pct": r.get("chg_pct"),
                            "price": r.get("price")})
        snap["sh_amount"] = idx_df.attrs.get("sh_amount")
        snap["sz_amount"] = idx_df.attrs.get("sz_amount")
        snap["total_amount"] = idx_df.attrs.get("total_amount")
    snap["indexes"] = indexes
    # 情绪
    snap["activity"] = act or {}
    # 板块(转成轻量 list;已按涨幅降序)
    boards = []
    if board is not None and not board.empty:
        for _, r in board.iterrows():
            boards.append({"name": r.get("name", ""),
                           "chg_pct": r.get("chg_pct"),
                           "net_inflow": r.get("net_inflow")})
    snap["boards"] = boards
    return snap


def on_ai_comment_market(sender=None, app_data=None, user_data=None):
    """对当前大盘 + 行业生成 AI 点评(后台线程流式,不阻塞界面)。

    事实全部复用行情页已拉取的快照(指数/涨跌家数/成交额/板块),
    与屏幕显示同源;AI 只做研判 + 风险提示,不给操作建议(见 app/ai)。
    """
    dpg.configure_item("mkt_ai_win", show=True)
    dpg.configure_item("mkt_ai_disclaimer", show=False)
    if dpg.does_item_exist("mkt_ai_sentiment"):
        dpg.configure_item("mkt_ai_sentiment", show=False)

    # 未配置模型:直接给引导
    if not ai_configured():
        dpg.set_value("mkt_ai_text", ai_config_hint())
        return

    snapshot = _build_market_snapshot()
    if not snapshot:
        dpg.set_value("mkt_ai_text",
                      "请先点『刷新行情』拉取大盘数据,再生成 AI 点评。")
        return

    dpg.set_value("mkt_ai_text", "正在综合指数/涨跌家数/成交额/行业板块生成大盘点评,请稍候...")
    dpg.configure_item("mkt_ai_btn", enabled=False, label="点评中...")

    def worker():
        buf = {"s": ""}

        def _on_delta(piece):
            buf["s"] += piece
            if dpg.does_item_exist("mkt_ai_text"):
                dpg.set_value("mkt_ai_text", buf["s"])

        try:
            res = ai_commentary.comment_market(snapshot, on_delta=_on_delta)
            if res.get("ok"):
                dpg.set_value("mkt_ai_text", res["text"])
                dpg.set_value("mkt_ai_disclaimer", res["disclaimer"])
                dpg.configure_item("mkt_ai_disclaimer", show=True)
                # 情绪评级彩色标签(偏暖=红 偏冷=绿,遵循 A 股惯例)
                sent = res.get("sentiment")
                if dpg.does_item_exist("mkt_ai_sentiment") and sent:
                    color = {"偏暖": (230, 60, 60), "偏冷": (40, 170, 80),
                             "中性": (200, 180, 90)}.get(sent, (200, 200, 200))
                    dpg.set_value("mkt_ai_sentiment", f"  市场情绪:{sent}")
                    dpg.configure_item("mkt_ai_sentiment", color=color, show=True)
            else:
                dpg.set_value("mkt_ai_text",
                              f"点评失败:\n{res.get('error', '未知错误')}")
                dpg.configure_item("mkt_ai_disclaimer", show=False)
        finally:
            if dpg.does_item_exist("mkt_ai_btn"):
                dpg.configure_item("mkt_ai_btn", enabled=True, label="AI点评大盘")

    threading.Thread(target=worker, daemon=True).start()


def _on_pick_code(code, hint=None):
    """点结果/榜单/自选表里的代码:跳到「选股 & K线」页并画K线;切换标的时停旧轮询、回到日线。

    hint: 可选的策略上下文(如"多因子策略命中,得分85,理由:..."),记入 STATE
          供 AI 点评作为 strategy_hint;换标的时无 hint 则清空,避免张冠李戴。
    """
    _stop_poll()
    STATE["intraday_code"] = None   # 换标的:分时图需完整重建
    STATE["ai_hint"] = hint          # 策略上下文(仅本次选中的标的有效)
    show_kline(code, "D")           # 先画好K线
    # 再切到「选股 & K线」tab(用 int id 最稳,别名字符串在部分DPG版本 set_value 不接受)
    try:
        if dpg.does_item_exist("main_tabs") and dpg.does_item_exist("tab_kline"):
            dpg.set_value("main_tabs", dpg.get_alias_id("tab_kline"))
            # 程序化切 tab 不触发 tab_bar callback,手动同步(切走行情页)
            STATE["active_is_market"] = False
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


def _on_watch_row_hover(sender, app_data):
    """持仓表整行悬停高亮(全局鼠标移动驱动)。

    持仓表行内含"持仓点评/移除"按钮与"买入价"输入框,不能用跨列 selectable
    (span_columns=True 会铺满整行抢占点击)。故改用 highlight_table_cell:
    它只染单元格背景、不占点击区域,鼠标落在某行任意位置时,
    把该行【全部9列】背景染成高亮色。第9列"操作"(点评/移除按钮)一起高亮,
    但 highlight_table_cell 只画背景不抢点击,按钮/输入框仍可正常点击/编辑。

    命中判定:不用表格容器自身的 rect(table 的 get_item_rect_min/max 常返回
    (0,0)/无效,会让整行判定失效)。改用【行内单元格】的 rect 拼行矩形:
    X = 该行首格左边 ~ 末格右边(覆盖所有列及格内留白、按钮右侧空白),
    Y = 首格上下边界(定行高)。单元格叶子控件的 rect 覆盖整格(含留白),
    故鼠标落在行内任意位置(有无文字都一样)都能命中整行。
    坐标系:get_item_rect_min/max 与 get_mouse_pos(local=False) 均为 viewport
    绝对坐标,一致。若坐标全拿不到,回退 is_item_hovered 逐格兜底(不全灭)。
    """
    tbl = "watch_table"
    if not dpg.does_item_exist(tbl):
        return
    try:
        rows = dpg.get_item_children(tbl, 1) or []
    except Exception:
        return
    hover_idx = None
    try:
        mx, my = dpg.get_mouse_pos(local=False)   # 相对 viewport 的绝对坐标
    except Exception:
        mx = my = None
    coord_ok = False   # 本帧是否成功用坐标判定(有一行拿到有效 rect 即视为可用)
    if mx is not None:
        for idx, row in enumerate(rows):
            try:
                cells = dpg.get_item_children(row, 1) or []
            except Exception:
                continue
            if not cells:
                continue
            try:
                r_min = dpg.get_item_rect_min(cells[0])    # 首格左上(定行高+左边界)
                r_max = dpg.get_item_rect_max(cells[0])    # 首格右下
                x_right = dpg.get_item_rect_max(cells[-1])[0]  # 末格右边界
            except Exception:
                continue
            if not r_min or not r_max:
                continue
            left, top, bot = r_min[0], r_min[1], r_max[1]
            if bot <= top:   # rect 尚未渲染,跳过
                continue
            if x_right <= left:      # 末格 rect 无效,退用首格右边界兜底
                x_right = r_max[0]
            coord_ok = True
            # 鼠标落在该行矩形(表宽 x 行高,含所有留白)即命中
            if left <= mx <= x_right and top <= my <= bot:
                hover_idx = idx
                break
    if not coord_ok:   # 坐标方案失效(rect 全无效)→ 回退逐格 hover 检测,至少不全灭
        for idx, row in enumerate(rows):
            try:
                cells = dpg.get_item_children(row, 1) or []
            except Exception:
                continue
            if any(dpg.is_item_hovered(c) for c in cells):
                hover_idx = idx
                break
    prev = STATE.get("_watch_hl_row")
    if hover_idx == prev:
        return
    # 动态取列数(表列会随功能增减,如加 PE分位/PB分位),避免写死漏染尾列
    ncols = 11
    for _r in rows:
        try:
            _c = dpg.get_item_children(_r, 1) or []
        except Exception:
            _c = []
        if _c:
            ncols = len(_c)
            break
    if prev is not None:   # 清除上一行高亮
        for col in range(ncols):
            try:
                dpg.unhighlight_table_cell(tbl, prev, col)
            except Exception:
                pass
    if hover_idx is not None:   # 染当前行全部列(含备注/操作列)
        for col in range(ncols):
            try:
                dpg.highlight_table_cell(tbl, hover_idx, col, WATCH_CELL_HL)
            except Exception:
                pass
    STATE["_watch_hl_row"] = hover_idx


def _on_result_click(sender, app_data, user_data):
    """选股结果表专用:选股结果和K线本就在同一页,无需跳转,单击即画K线。
    user_data=(code, hint):hint 为该股命中的策略上下文,带给 AI 点评。"""
    if not user_data:
        return
    if isinstance(user_data, (tuple, list)):
        code, hint = (user_data + (None,))[:2]
    else:
        code, hint = user_data, None
    if code:
        _on_pick_code(code, hint=hint)


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


def _fmt_num(x, nd=2, suffix=""):
    """格式化数字;None/异常返回 '-'。"""
    try:
        if x is None:
            return "-"
        return f"{float(x):.{nd}f}{suffix}"
    except Exception:  # noqa
        return "-"


# 信息栏配色:中性事实用淡灰白(不涂红绿,避免误导价值判断);
# 好坏/涨跌型字段才用红涨绿跌
INFO_NEUTRAL = (200, 210, 225)   # 中性事实值(PE/PB/市值/负债率)
INFO_LABEL = (140, 148, 160)     # 字段标签(灰)


def _seg_neutral(label, val_txt):
    """中性事实字段:标签灰 + 数值淡白,不涂红绿。"""
    return [(label + " ", INFO_LABEL), (val_txt, INFO_NEUTRAL)]


def _seg_signed(label, val, nd=2, suffix="%", zero_is_good=True):
    """好坏/涨跌型字段:正值涂红(好/涨),负值涂绿(差/跌),缺失灰显。"""
    if val is None:
        return [(label + " ", INFO_LABEL), ("-", INFO_NEUTRAL)]
    try:
        v = float(val)
    except Exception:  # noqa
        return [(label + " ", INFO_LABEL), ("-", INFO_NEUTRAL)]
    color = RED if (v > 0 if zero_is_good else v >= 0) else GREEN
    return [(label + " ", INFO_LABEL), (f"{v:.{nd}f}{suffix}", color)]


# RSI 状态配色:超买(过热,追高风险)橙红警示;超卖(超跌,反弹机会)青绿提示;
# 中间区间(健康/中性)用灰白,不喧宾夺主。这里的红/绿是"状态语义"(热/冷),
# 非涨跌方向,故单独取色而非复用 RED/GREEN,避免与涨跌红绿混淆。
RSI_OVERBOUGHT = (245, 130, 60)   # 超买:橙红(过热警示)
RSI_OVERSOLD = (60, 200, 170)     # 超卖:青绿(超跌提示)


def _latest_rsi(code, period, n=14):
    """按当前周期(D/W/M)计算该标的最新一根 K 线的 RSI(14)。取不到返回 None。"""
    try:
        df = db.load_kline(code)
        if df is None or df.empty:
            return None
        if period in ("W", "M"):
            df = ind.resample_period(df, period)
        s = ind.rsi(df["close"].astype(float), n).dropna()
        if s.empty:
            return None
        v = float(s.iloc[-1])
        return v if v == v else None   # 过滤 NaN
    except Exception:  # noqa
        return None


def _seg_rsi(val):
    """RSI 分段:超买橙红(>=70)、超卖青绿(<=30)、中间灰白,并附中文状态。"""
    if val is None:
        return [("RSI ", INFO_LABEL), ("-", INFO_NEUTRAL)]
    if val >= 70:
        color, state = RSI_OVERBOUGHT, "超买"
    elif val <= 30:
        color, state = RSI_OVERSOLD, "超卖"
    else:
        color, state = INFO_NEUTRAL, "中性"
    return [("RSI ", INFO_LABEL), (f"{val:.1f}", color),
            (f"({state})", color)]


def _compose_kline_info(code):
    """
    组装基本面信息栏的【分段】数据(供 K 线上方信息栏分色显示)。
    返回 list[list[(text, color)]]:外层每项是一个字段,内层是该字段的文字分段。
    完全没取到基本面时返回 None。
    分色三类:①中性事实(PE/PB/市值/负债率)灰白;②越高越好(ROE/毛利/净利)正红负绿;
    ③涨跌语义(营收/净利同比)增长红、下滑绿。
    """
    f = ai_commentary.get_fundamental_ondemand(code)
    if f.get("_source") == "none" and all(
            f.get(k) is None for k in ("pe_ttm", "pb", "roe", "total_mv")):
        return None   # 完全没取到,交由调用方决定提示
    pe_pct = db.industry_valuation_percentile(code, "pe_ttm")
    fields = [
        # ① 中性事实值:不涂红绿
        _seg_neutral("PE", _fmt_num(f.get("pe_ttm"))),
        _seg_neutral("PB", _fmt_num(f.get("pb"))),
        # ② 越高越好:ROE 正红负绿
        _seg_signed("ROE", f.get("roe")),
        _seg_neutral("市值", _fmt_num(f.get("total_mv"), 1) + "亿"),
    ]
    # 技术面:RSI(14) 按当前周期实时算,超买/超卖分色
    rsi_val = _latest_rsi(code, STATE.get("cur_period", "D"))
    fields.append(_seg_rsi(rsi_val))
    # 筹码面:获利盘比例 + 平均成本(始终按日线算,价格维度指标)
    try:
        _cd = db.load_kline(code)
        _chip = chip.compute_chip_distribution(_cd, total_mv=f.get("total_mv"))
    except Exception:  # noqa
        _chip = None
    if _chip is not None:
        pf = _chip["profit_ratio"] * 100
        lc = _chip["last_close"]
        ac = _chip["avg_cost"]
        # 获利盘:高=浮盈多(潜在抛压),低=普遍套牢(超跌)。语义偏中性,用状态色
        pf_col = RSI_OVERBOUGHT if pf >= 85 else (
            RSI_OVERSOLD if pf <= 15 else INFO_NEUTRAL)
        fields.append([("获利盘 ", INFO_LABEL), (f"{pf:.0f}%", pf_col)])
        # 现价相对平均成本:上方红(强)、下方绿(弱)
        ac_col = RED if lc >= ac else GREEN
        fields.append([("平均成本 ", INFO_LABEL), (f"{ac:.2f}", ac_col)])
    if f.get("gross_margin") is not None:
        fields.append(_seg_signed("毛利率", f.get("gross_margin")))
    if f.get("net_margin") is not None:
        fields.append(_seg_signed("净利率", f.get("net_margin")))
    # ③ 涨跌语义:营收/净利同比,增长红、下滑绿(最有信息量,重点分色)
    if f.get("rev_yoy") is not None:
        fields.append(_seg_signed("营收同比", f.get("rev_yoy")))
    if f.get("profit_yoy") is not None:
        fields.append(_seg_signed("净利同比", f.get("profit_yoy")))
    # 负债率:中性偏事实(高未必坏,行业差异大),不涂红绿
    if f.get("debt_ratio") is not None:
        fields.append(_seg_neutral("负债率",
                                   _fmt_num(f.get("debt_ratio"), 2, "%")))
    # 尾部补充:行业分位 + 财报期(灰显)
    # 说明:percentile = 同行中比它更便宜(PE更低)的占比。低=便宜,高=贵。
    # 直接用"比X%同行便宜"的正向措辞,避免"高于X%"字面歧义(见AI点评同源修复)。
    tail = ""
    if pe_pct:
        p = pe_pct["percentile"]
        cheaper = 100.0 - p
        vt = "偏低" if p <= 30 else ("偏高" if p >= 70 else "居中")
        ind_name = db.load_industry_map().get(code, "同行业")
        tail += (f" | 估值{vt}:{ind_name}内比约{cheaper:.0f}%同行更便宜")
    if f.get("report_date"):
        tail += f"  (财报期 {f['report_date']})"
    if tail:
        fields.append([(tail.strip(), INFO_LABEL)])
    return fields


def _render_info_segments(fields):
    """把 _compose_kline_info 返回的分段数据渲染进横向信息栏容器。"""
    if not dpg.does_item_exist("kline_info"):
        return
    dpg.delete_item("kline_info", children_only=True)
    if not fields:
        return
    for i, field in enumerate(fields):
        # 每个字段之间用一段间隔;同字段内的分段(标签+数值)紧挨
        for text, color in field:
            dpg.add_text(text, parent="kline_info", color=color)
        if i < len(fields) - 1:
            dpg.add_text("  ", parent="kline_info", color=INFO_LABEL)


def _set_info_message(msg, color=INFO_LABEL):
    """在信息栏显示一句提示文字(加载中/无数据等)。"""
    if not dpg.does_item_exist("kline_info"):
        return
    dpg.delete_item("kline_info", children_only=True)
    if msg:
        dpg.add_text(msg, parent="kline_info", color=color)


def _update_kline_info(code):
    """更新 K 线上方基本面信息栏。本地有秒显示,缺失则后台拉取回填,不卡界面。"""
    if not dpg.does_item_exist("kline_info"):
        return
    # ETF 无个股基本面,直接清空信息栏
    try:
        if code in db.load_etf_codes():
            _set_info_message("")
            return
    except Exception:  # noqa
        pass
    row = db.get_fundamental(code)
    has_local = row and any(
        row.get(k) is not None
        for k in ("pe_ttm", "pb", "roe", "total_mv", "gross_margin", "rev_yoy"))
    if has_local:
        # 本地已有,直接同步显示
        _render_info_segments(_compose_kline_info(code))
        return
    # 本地缺失:先占位,后台现拉再回填(只回填仍在看这只票时)
    _set_info_message("基本面加载中...")

    def worker(target):
        fields = _compose_kline_info(target)
        # 用户可能已切换到别的票,回填前校验当前仍是这只
        if STATE.get("cur_code") == target and dpg.does_item_exist("kline_info"):
            if fields:
                _render_info_segments(fields)
            else:
                _set_info_message("该标的暂无基本面数据")

    threading.Thread(target=worker, args=(code,), daemon=True).start()


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
    # 更新 K 线上方基本面信息栏(本地有秒显,缺失后台拉取回填)
    _update_kline_info(code)

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
    rsis = df["rsi14"].tolist() if "rsi14" in df.columns else [None] * len(df)
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
        3, 2, label="", width=-1, height=-1, parent="chart_area",
        row_ratios=[3.0, 1.0, 1.2], column_ratios=[8.0, 1.0],
        link_rows=True, link_columns=True, column_major=True,
        no_title=True, tag="sp",
    ):
        # 第1层: K线主图 + 均线
        with dpg.plot(label=f"{pname} {label}", no_title=False, height=-1,
                      tag="kplot", no_box_select=True):
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
            # 平均成本参考线(横贯K线主图,与右侧筹码栏同价位对齐)
            try:
                _cdf0 = db.load_kline(code)
                _fm0 = db.get_fundamental(code) or {}
                _chip0 = chip.compute_chip_distribution(
                    _cdf0, total_mv=_fm0.get("total_mv"))
            except Exception:  # noqa
                _chip0 = None
            if _chip0 is not None:
                _ac0 = _chip0["avg_cost"]
                dpg.add_line_series([xs[0], xs[-1]], [_ac0, _ac0], parent=ky,
                                    label=f"平均成本 {_ac0:.2f}")
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
        with dpg.plot(label="成交额", no_title=False, height=-1, no_box_select=True):
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
        with dpg.plot(label="MACD (12,26,9)", no_title=False, height=-1,
                      no_box_select=True):
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

        # === 右列(column_major): 筹码栏固定在窗口右侧,X轴独立不随K线缩放,
        #     Y轴(价格)经 link_rows 与K线主图联动对齐 ===
        # 右上: 筹码分布(成本分布)独立子图
        with dpg.plot(label="筹码", no_title=True, height=-1, no_box_select=True,
                      no_mouse_pos=True, tag="chipplot"):
            dpg.add_plot_axis(dpg.mvXAxis, tag="chipx", no_tick_labels=True,
                              no_gridlines=True)
            # 价格刻度显示在筹码栏右侧(opposite),与K线主图价格联动对齐,
            # 方便直接读出筹码峰对应的价位。
            cy = dpg.add_plot_axis(dpg.mvYAxis, tag="chipy", opposite=True,
                                   tick_format="%.2f")
            try:
                _cdf = db.load_kline(code)   # 用完整日线算成本沉淀
                _fm = db.get_fundamental(code) or {}
                _chip = chip.compute_chip_distribution(
                    _cdf, total_mv=_fm.get("total_mv"))
            except Exception:  # noqa
                _chip = None
            if _chip is not None and _chip["chips"].max() > 0:
                _pr = _chip["prices"]
                _ch = _chip["chips"]
                _lc = _chip["last_close"]
                _thick = float(_pr[1] - _pr[0]) if len(_pr) > 1 else 0.1
                # 获利盘(成本≤现价)红,套牢盘(成本>现价)绿。横向柱从 x=0 向右
                win_p = [float(p) for p, c in zip(_pr, _ch) if p <= _lc]
                win_x = [float(c) for p, c in zip(_pr, _ch) if p <= _lc]
                los_p = [float(p) for p, c in zip(_pr, _ch) if p > _lc]
                los_x = [float(c) for p, c in zip(_pr, _ch) if p > _lc]
                _pf = _chip["profit_ratio"] * 100
                if win_x:
                    bid = dpg.add_bar_series(
                        win_x, win_p, parent=cy, horizontal=True,
                        weight=_thick, label=f"获利盘 {_pf:.0f}%")
                    dpg.bind_item_theme(bid, "bar_red")
                if los_x:
                    bid = dpg.add_bar_series(
                        los_x, los_p, parent=cy, horizontal=True,
                        weight=_thick, label=f"套牢盘 {100 - _pf:.0f}%")
                    dpg.bind_item_theme(bid, "bar_green")
                # 平均成本线(在筹码栏内也标一条,与左侧K线的成本线同价位)
                _ac = _chip["avg_cost"]
                _xm = float(max(_ch))
                dpg.add_line_series([0.0, _xm], [_ac, _ac], parent=cy,
                                    label=f"均本 {_ac:.2f}")
                dpg.fit_axis_data("chipx")
                dpg.fit_axis_data("chipy")

        # 右中/右下: 占位空图(3×2网格补齐,不显示内容)
        for _ph in ("chip_ph2", "chip_ph3"):
            with dpg.plot(no_title=True, height=-1, no_box_select=True,
                          no_menus=True, tag=_ph):
                dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True,
                                  no_gridlines=True)
                dpg.add_plot_axis(dpg.mvYAxis, no_tick_labels=True,
                                  no_gridlines=True)

    # 缓存本次绘制的K线数据,供鼠标悬停提示回调按索引取值
    STATE["kl_bars"] = {
        "n": n,
        "dates": dates,
        "open": opens, "close": closes, "high": highs, "low": lows,
        "rsi": rsis,
    }

    dpg.set_value("kline_title", f"{pname} / 成交额 / MACD - {label}")
    # 切回 K 线:清空分时专用的涨跌幅/后缀,避免残留上一只票的红绿数字
    if dpg.does_item_exist("kline_title_chg"):
        dpg.set_value("kline_title_chg", "")
    if dpg.does_item_exist("kline_title_suffix"):
        dpg.set_value("kline_title_suffix", "")
    _set_status(f"已绘制 {label} 的{pname}图表")


def on_ai_comment(sender=None, app_data=None, user_data=None, force=False):
    """对当前 K 线标的生成 AI 综合点评(后台线程,不阻塞界面)。

    AI 把系统算好/实时拉取的技术面+基本面指标翻译成人话+提示风险+结构化评级,
    不给买卖建议(见 app/ai)。force=True 时跳过当天缓存强制重新生成。
    """
    code = STATE.get("cur_code")
    if not code:
        dpg.set_value("ai_comment_text", "请先在左侧或搜索选中一只股票再点评。")
        dpg.configure_item("ai_comment_win", show=True)
        return
    # 未配置模型:直接给引导,不发起请求
    if not ai_configured():
        dpg.set_value("ai_comment_text", ai_config_hint())
        dpg.configure_item("ai_comment_disclaimer", show=False)
        dpg.configure_item("ai_comment_win", show=True)
        return

    nm = db.name_of(code) or ""
    hint = STATE.get("ai_hint") or ""
    dpg.configure_item("ai_comment_win", show=True)
    dpg.configure_item("ai_comment_disclaimer", show=False)
    for _b in ("ai_rating_badge", "ai_risk_badge", "ai_cache_badge"):
        if dpg.does_item_exist(_b):
            dpg.configure_item(_b, show=False)
    tip = "重新生成中..." if force else "生成 AI 点评,请稍候..."
    dpg.set_value("ai_comment_text", f"正在为 {code} {nm} {tip}")
    dpg.configure_item("ai_comment_btn", enabled=False, label="点评中...")

    def _badge(tag, text, color):
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, text)
            dpg.configure_item(tag, color=color, show=True)

    def worker():
        # 流式回调:边生成边把增量追加到点评文本(缓存命中时不会触发)
        stream = {"buf": "", "head": ""}

        def _on_delta(piece):
            stream["buf"] += piece
            if dpg.does_item_exist("ai_comment_text"):
                dpg.set_value("ai_comment_text", stream["head"] + stream["buf"])

        # 先算出 head 前缀(需要 facts,但流式回调早于 res 返回,故预取名称)
        stream["head"] = f"【{code} {nm}】\n" if nm else f"【{code}】\n"
        res = ai_commentary.comment_stock(code, strategy_hint=hint,
                                          force_refresh=force,
                                          on_delta=_on_delta)
        try:
            if res.get("ok"):
                f = res["facts"]
                head = f"【{f['code']} {f['name']}】{f['industry']}\n"
                dpg.set_value("ai_comment_text", head + res["text"])
                dpg.set_value("ai_comment_disclaimer", res["disclaimer"])
                dpg.configure_item("ai_comment_disclaimer", show=True)
                # 结构化评级彩色标签(A股惯例:偏多=红,偏空=绿)
                rating = res.get("rating")
                if rating == "偏多":
                    _badge("ai_rating_badge", "  评级:偏多", (230, 60, 60))
                elif rating == "偏空":
                    _badge("ai_rating_badge", "  评级:偏空", (40, 170, 80))
                elif rating == "中性":
                    _badge("ai_rating_badge", "  评级:中性", (200, 180, 90))
                risk = res.get("risk")
                if risk == "高":
                    _badge("ai_risk_badge", "  风险:高", (230, 60, 60))
                elif risk == "中":
                    _badge("ai_risk_badge", "  风险:中", (220, 160, 60))
                elif risk == "低":
                    _badge("ai_risk_badge", "  风险:低", (120, 180, 120))
                if res.get("cached"):
                    _badge("ai_cache_badge", "  (当日缓存,点刷新重生成)",
                           (130, 130, 130))
            else:
                dpg.set_value("ai_comment_text", f"点评失败:\n{res.get('error', '未知错误')}")
                dpg.configure_item("ai_comment_disclaimer", show=False)
        finally:
            if dpg.does_item_exist("ai_comment_btn"):
                dpg.configure_item("ai_comment_btn", enabled=True, label="AI点评")

    threading.Thread(target=worker, daemon=True).start()


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
    # RSI(14):该根K线的相对强弱,附超买/超卖状态
    rsi_line = ""
    rv = bars.get("rsi", [None] * bars["n"])[idx]
    try:
        rv = float(rv)
        if rv == rv:   # 非 NaN
            st = "超买" if rv >= 70 else ("超卖" if rv <= 30 else "中性")
            rsi_line = f"\nRSI {rv:.1f}({st})"
    except (TypeError, ValueError):
        pass
    txt = (f"{d}\n开 {o:.2f}  收 {c:.2f}\n"
           f"高 {h:.2f}  低 {low:.2f}\n涨跌 {'+' if chg >= 0 else ''}{chg:.2f}%"
           f"{rsi_line}")
    # 提示框锚定到该蜡烛索引、Y 跟随鼠标,颜色随涨跌
    col = (230, 90, 90, 255) if c >= o else (90, 200, 120, 255)
    dpg.configure_item("kanno", label=txt, default_value=(idx, my),
                       color=col, show=True)


# ---------- 构建界面 ----------
STRATEGY_LABEL2KEY = {cls.name: key for key, cls in ALL_STRATEGIES.items()}
KEY2STRATEGY_LABEL = {key: cls.name for key, cls in ALL_STRATEGIES.items()}


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
                dpg.add_checkbox(label="强制全量重拉(覆盖历史,修复脏数据)",
                                 default_value=False, tag="force_full",
                                 callback=lambda s, a: dpg.configure_item(
                                     "force_full_hint", show=bool(a)))
                dpg.add_text("已勾选:将覆盖本地全部历史日线(而非只补最新)。\n"
                             "用于修正旧的成交量/成交额脏数据,耗时与首次相当。",
                             tag="force_full_hint", show=False,
                             wrap=300, color=(255, 180, 90))
                dpg.add_text("主流股(沪深300+中证500)优先,并发拉取并同步\n"
                             "大盘指数+行业。400只约3-5分钟;全A股(5000+)\n"
                             "首次约30-50分钟,选股/回测也更慢,按需选择。\n"
                             "基本面每只1-2s串行,勾选后首次较慢,增量缓存。",
                             wrap=300, color=(130, 130, 130))
                dpg.add_separator()

                dpg.add_text("2. 选择策略")
                dpg.add_text("用一句话选股(AI 翻译成筛选条件,本地引擎执行):",
                             color=(120, 220, 160), wrap=300)
                dpg.add_input_text(
                    tag="nl_query_in", width=280,
                    hint="如:市值500亿以内的MACD金叉前8只",
                    callback=on_nl_scan, on_enter=True)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="AI 选股", callback=on_nl_scan,
                                   width=138, height=28, tag="nl_scan_btn")
                    dpg.add_button(
                        label="示例", width=138, height=28,
                        callback=lambda: dpg.set_value(
                            "nl_query_in", "低估值高ROE的白马股,PE不超过25,前10只"))
                dpg.add_text("", tag="nl_query_hint", wrap=300,
                             color=(160, 200, 255))
                dpg.add_separator()

                dpg.add_text("或手动选择策略")
                dpg.add_combo(
                    [CROSS_SECTION_LABEL] + list(STRATEGY_LABEL2KEY.keys()),
                    default_value=list(STRATEGY_LABEL2KEY.keys())[0],
                    callback=on_strategy_change, tag="strategy_combo", width=280,
                )
                with dpg.group(horizontal=True):
                    dpg.add_text("取前N只(仅智能版):", color=(130, 130, 130))
                    dpg.add_input_int(tag="cs_top_n", default_value=50,
                                      min_value=1, max_value=300, width=90,
                                      step=10)
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
                with dpg.tab_bar(tag="main_tabs",
                                 callback=_on_main_tab_changed):
                    # --- Tab 1: 选股 & K线 ---
                    with dpg.tab(label="选股 & K线", tag="tab_kline"):
                        with dpg.group(horizontal=True):
                            dpg.add_text("选股结果(点代码看K线,点+自选加入持仓跟踪)")
                            dpg.add_button(label="AI精排Top10", width=110, height=24,
                                           tag="ai_rank_btn", callback=on_ai_rank)
                            dpg.add_button(label="批量点评(晨报)", width=120, height=24,
                                           tag="ai_batch_btn",
                                           callback=lambda: on_ai_batch("results"))
                            dpg.add_button(label="组合解读", width=90, height=24,
                                           tag="ai_pf_btn",
                                           callback=lambda: on_ai_portfolio("results"))
                            dpg.add_text("", tag="ai_rank_hint", color=(130, 130, 130))
                        with dpg.table(tag="result_table", header_row=True,
                                       resizable=True, policy=dpg.mvTable_SizingStretchProp,
                                       height=210, scrollY=True, freeze_rows=1):
                            for col in ["代码", "名称", "行业", "现价", "PE", "PB",
                                        "ROE%", "市值亿", "得分", "说明", "操作"]:
                                dpg.add_table_column(label=col)
                        # --- AI 精排结果面板(默认隐藏,精排后展开) ---
                        with dpg.child_window(tag="ai_rank_win", height=228,
                                              border=True, show=False):
                            with dpg.group(horizontal=True):
                                dpg.add_text("AI 精排优选", color=(160, 200, 255))
                                dpg.add_text("", tag="ai_rank_status",
                                             color=(150, 150, 150))
                                dpg.add_button(label="关闭", width=48, height=22,
                                               callback=lambda: dpg.configure_item(
                                                   "ai_rank_win", show=False))
                            dpg.add_text(
                                "说明:此处评级/风险是 AI 在本批候选池内『横向相对比较』"
                                "后给出的,依据为压缩后的关键指标;与 K 线页『AI 综合点评』"
                                "对单只做『深度绝对评级』(事实更全、含筹码/多周期)口径不同,"
                                "两者可能不一致,均属正常。",
                                wrap=820, color=(150, 150, 150))
                            with dpg.table(tag="ai_rank_table", header_row=True,
                                           resizable=True,
                                           policy=dpg.mvTable_SizingStretchProp,
                                           height=150, scrollY=True):
                                for col in ["#", "代码", "名称", "行业",
                                            "评级", "风险", "入选理由"]:
                                    dpg.add_table_column(label=col)
                            dpg.add_text("", tag="ai_rank_disclaimer",
                                         wrap=820, color=(150, 150, 150))
                        # --- AI 批量点评晨报面板(默认隐藏) ---
                        with dpg.child_window(tag="ai_batch_win", height=200,
                                              border=True, show=False):
                            with dpg.group(horizontal=True):
                                dpg.add_text("AI 批量点评晨报", tag="ai_batch_title",
                                             color=(160, 200, 255))
                                dpg.add_button(label="复制", width=48, height=22,
                                               callback=_copy_ai_text,
                                               user_data=["ai_batch_title",
                                                          "ai_batch_report",
                                                          "ai_batch_text"])
                                dpg.add_button(label="打开目录", width=76, height=22,
                                               callback=on_open_report_dir)
                                dpg.add_button(label="关闭", width=48, height=22,
                                               callback=lambda: dpg.configure_item(
                                                   "ai_batch_win", show=False))
                            dpg.add_text("", tag="ai_batch_report",
                                         wrap=820, color=(120, 220, 160))
                            with dpg.child_window(height=120, border=False):
                                dpg.add_text("", tag="ai_batch_text", wrap=810,
                                             color=(225, 225, 225))
                        # --- AI 组合解读面板(默认隐藏,流式) ---
                        with dpg.child_window(tag="ai_pf_win", height=190,
                                              border=True, show=False):
                            with dpg.group(horizontal=True):
                                dpg.add_text("AI 组合解读", tag="ai_pf_title",
                                             color=(160, 200, 255))
                                dpg.add_button(label="复制", width=48, height=22,
                                               callback=_copy_ai_text,
                                               user_data=["ai_pf_title",
                                                          "ai_pf_text",
                                                          "ai_pf_disclaimer"])
                                dpg.add_button(label="关闭", width=48, height=22,
                                               callback=lambda: dpg.configure_item(
                                                   "ai_pf_win", show=False))
                            dpg.add_text("", tag="ai_pf_text", wrap=820,
                                         color=(225, 225, 225))
                            dpg.add_text("", tag="ai_pf_disclaimer", wrap=820,
                                         color=(150, 150, 150), show=False)
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
                            dpg.add_button(label="AI点评", width=68, height=26,
                                           tag="ai_comment_btn",
                                           callback=on_ai_comment)
                        # AI 点评结果面板(默认隐藏,点评时展开)
                        with dpg.child_window(tag="ai_comment_win", height=170,
                                              border=True, show=False):
                            with dpg.group(horizontal=True):
                                dpg.add_text("AI 综合点评", color=(160, 200, 255))
                                # 结构化评级彩色标签(点评后填充)
                                dpg.add_text("", tag="ai_rating_badge", show=False)
                                dpg.add_text("", tag="ai_risk_badge", show=False)
                                dpg.add_text("", tag="ai_cache_badge",
                                             color=(130, 130, 130), show=False)
                                dpg.add_button(label="复制", width=48, height=22,
                                               callback=_copy_ai_text,
                                               user_data=["ai_rating_badge",
                                                          "ai_risk_badge",
                                                          "ai_comment_text",
                                                          "ai_comment_disclaimer"])
                                dpg.add_button(label="刷新", width=48, height=22,
                                               tag="ai_refresh_btn",
                                               callback=lambda: on_ai_comment(
                                                   force=True))
                                dpg.add_button(label="关闭", width=48, height=22,
                                               callback=lambda: dpg.configure_item(
                                                   "ai_comment_win", show=False))
                            dpg.add_text("", tag="ai_comment_text", wrap=820,
                                         color=(230, 230, 230))
                            dpg.add_text(
                                "说明:此评级/风险为 AI 对本只个股做『深度绝对评级』"
                                "(事实含筹码/多周期/行业分位等,单独判定);与『AI精排Top10』"
                                "中在候选池内『横向相对比较』的评级口径不同,两者可能不一致,"
                                "均属正常。",
                                wrap=820, color=(150, 150, 150))
                            dpg.add_text("", tag="ai_comment_disclaimer", wrap=820,
                                         color=(150, 150, 150), show=False)
                        # 标题行:横向 group,把"涨跌幅"单独拆出来上色(单 text 无法分段着色)
                        with dpg.group(tag="kline_title_bar", horizontal=True,
                                       horizontal_spacing=0):
                            dpg.add_text("K线 / 成交额 / MACD", tag="kline_title")
                            dpg.add_text("", tag="kline_title_chg")   # 涨跌幅,红涨绿跌
                            dpg.add_text("", tag="kline_title_suffix",
                                         color=(160, 160, 160))       # 后缀(轮询中/已停)
                        # 基本面信息栏(PE/PB/ROE/市值/成长性),紧贴图上方常驻显示
                        # 用横向 group 容纳多个分色 text(单个 text 无法分字段着色)
                        dpg.add_group(tag="kline_info", horizontal=True,
                                      horizontal_spacing=0)
                        dpg.add_child_window(tag="chart_area", border=False, height=-1)

                    # --- Tab: 行情(大盘概览) ---
                    with dpg.tab(label="行情", tag="tab_market"):
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="刷新行情", width=100, height=28,
                                           tag="mkt_refresh_btn",
                                           callback=on_refresh_market)
                            dpg.add_checkbox(label="盘中定时刷新", tag="mkt_poll_cb",
                                             callback=on_toggle_mkt_poll)
                            dpg.add_text(f"(每{MKT_POLL_INTERVAL}秒)",
                                         color=(130, 130, 130))
                            dpg.add_button(label="AI点评大盘", width=110, height=28,
                                           tag="mkt_ai_btn",
                                           callback=on_ai_comment_market)
                            dpg.add_text("点『刷新行情』拉取大盘数据",
                                         tag="mkt_status", color=(255, 200, 100))
                        dpg.add_separator()
                        # AI 大盘点评面板(默认隐藏,点『AI点评大盘』展开)
                        with dpg.child_window(tag="mkt_ai_win", height=190,
                                              border=True, show=False):
                            with dpg.group(horizontal=True):
                                dpg.add_text("AI 大盘点评", color=(160, 200, 255))
                                dpg.add_text("", tag="mkt_ai_sentiment",
                                             show=False)
                                dpg.add_button(label="复制", width=48, height=22,
                                               callback=_copy_ai_text,
                                               user_data=["mkt_ai_sentiment",
                                                          "mkt_ai_text",
                                                          "mkt_ai_disclaimer"])
                                dpg.add_button(label="重新生成", width=80, height=22,
                                               callback=on_ai_comment_market)
                                dpg.add_button(label="关闭", width=48, height=22,
                                               callback=lambda: dpg.configure_item(
                                                   "mkt_ai_win", show=False))
                            dpg.add_text("综合指数、涨跌家数、成交额、行业板块由 AI 研判,"
                                         "只做解读与风险提示,不构成投资建议",
                                         color=(130, 130, 130), wrap=880)
                            dpg.add_text("", tag="mkt_ai_text", wrap=880,
                                         color=(230, 230, 230))
                            dpg.add_text("", tag="mkt_ai_disclaimer", wrap=880,
                                         color=(140, 140, 140), show=False)
                        dpg.add_separator()
                        # 成交额 + 市场总貌(一行醒目文字)
                        dpg.add_text("", tag="mkt_amount_text", color=(230, 200, 120))
                        dpg.add_text("", tag="mkt_summary_text", color=(160, 200, 255),
                                     wrap=900)
                        dpg.add_separator()
                        # ===== 盘面一句话结论(规则引擎自动生成, 顶部醒目) =====
                        with dpg.child_window(tag="sec_verdict", height=58,
                                              border=True):
                            with dpg.group(horizontal=True):
                                dpg.add_text("盘面速读", color=(120, 220, 160))
                                dpg.add_text("(规则自动研判 · 仅客观描述, 非投资建议)",
                                             color=(120, 130, 150))
                            dpg.add_group(tag="mkt_verdict_group")
                        dpg.add_separator()
                        dpg.add_text("提示:各区块间的细横条可上下拖拽,放大你关注的板块",
                                     color=(120, 130, 150))
                        # ===== 可拖拽块1:涨跌家数/情绪 + 盘面量能/情绪温度 =====
                        with dpg.child_window(
                                tag="sec_activity",
                                height=MKT_SECTION_HEIGHTS["sec_activity"],
                                border=True):
                            dpg.add_text("涨跌家数 / 市场情绪", color=(120, 220, 160))
                            dpg.add_group(tag="mkt_activity_group")
                            dpg.add_separator()
                            dpg.add_text("盘面量能 / 情绪温度", color=(120, 220, 160))
                            dpg.add_group(tag="mkt_mood_group")
                        _add_mkt_section_handle("sec_activity")
                        # ===== 可拖拽块2:涨跌幅分布直方图(整行) =====
                        with dpg.child_window(
                                tag="sec_dist",
                                height=MKT_SECTION_HEIGHTS["sec_dist"],
                                border=True):
                            with dpg.group(horizontal=True):
                                dpg.add_text("涨跌幅分布(全市场·涨=红 跌=绿)",
                                             color=(120, 220, 160))
                                dpg.add_text("", tag="mkt_dist_note",
                                             color=(130, 130, 130))
                            dpg.add_group(tag="mkt_dist_group")
                        _add_mkt_section_handle("sec_dist")
                        # ===== 可拖拽块3:板块强弱·资金诊断(整行,高潜力/高风险左右平行) =====
                        with dpg.child_window(
                                tag="sec_diag",
                                height=MKT_SECTION_HEIGHTS["sec_diag"],
                                border=True):
                            with dpg.group(horizontal=True):
                                dpg.add_text("板块强弱 · 资金诊断",
                                             color=(120, 220, 160))
                                dpg.add_text("(当日盘面口径)",
                                             color=(130, 130, 130))
                            dpg.add_group(tag="mkt_board_diag_group")
                        _add_mkt_section_handle("sec_diag")
                        # ===== 可拖拽块4:核心指数实时行情表 =====
                        with dpg.child_window(
                                tag="sec_index",
                                height=MKT_SECTION_HEIGHTS["sec_index"],
                                border=True):
                            dpg.add_text("核心指数(涨=红 跌=绿)", color=(120, 220, 160))
                            with dpg.table(tag="mkt_index_table", header_row=True,
                                           resizable=True,
                                           policy=dpg.mvTable_SizingStretchProp,
                                           height=-1, scrollY=True):
                                for col in ["指数", "最新价", "涨跌幅", "涨跌点", "今开",
                                            "最高", "最低", "成交额"]:
                                    dpg.add_table_column(label=col)
                        _add_mkt_section_handle("sec_index")
                        # ===== 块5:热门板块(最后一块,height=-1 自动填满剩余空间) =====
                        with dpg.child_window(
                                tag="sec_board",
                                height=-1,
                                border=True):
                            with dpg.group(horizontal=True):
                                dpg.add_text("热门板块(行业·涨=红 跌=绿)",
                                             color=(120, 220, 160))
                                dpg.add_text("  排序:", color=(140, 140, 140))
                                dpg.add_button(label="涨跌幅", width=64, height=22,
                                               tag="mkt_board_sort_chg_pct",
                                               user_data="chg_pct",
                                               callback=on_sort_board)
                                dpg.bind_item_theme("mkt_board_sort_chg_pct",
                                                    "period_on")
                                dpg.add_button(label="净额", width=64, height=22,
                                               tag="mkt_board_sort_net_inflow",
                                               user_data="net_inflow",
                                               callback=on_sort_board)
                                dpg.add_button(label="成交额", width=64, height=22,
                                               tag="mkt_board_sort_amount",
                                               user_data="amount",
                                               callback=on_sort_board)
                            dpg.add_text("净额=板块内全部资金 流入−流出(全单口径),"
                                         "方向反映资金进出但非券商『主力资金』口径,"
                                         "数值会明显偏小,仅供参考",
                                         color=(130, 130, 130), wrap=880)
                            with dpg.table(tag="mkt_board_table", header_row=True,
                                           resizable=True,
                                           policy=dpg.mvTable_SizingStretchProp,
                                           height=-1, scrollY=True):
                                for col in ["排名", "板块", "涨跌幅", "成交额",
                                            "净额(全单)", "涨/跌家数", "领涨股"]:
                                    dpg.add_table_column(label=col)

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
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="AI 点评自选(晨报)", width=150, height=28,
                                           callback=lambda: on_ai_batch("watch"))
                            dpg.add_button(label="AI 组合解读", width=120, height=28,
                                           callback=lambda: on_ai_portfolio("watch"))
                            dpg.add_button(label="点评历史", width=90, height=28,
                                           callback=on_open_ai_history)
                        dpg.add_text("买入价默认记为加入时的现价;『刷新实时行情』拉全市场快照算"
                                     "当日涨跌+浮动盈亏并检查预警;点代码看K线;"
                                     "点每行『持仓点评』结合你的买入价给持有/加减仓倾向",
                                     wrap=900, color=(130, 130, 130))
                        # --- AI 输出面板(晨报/组合解读/单只持仓点评共用,就地显示) ---
                        with dpg.child_window(tag="watch_ai_win", height=220,
                                              border=True, show=False):
                            with dpg.group(horizontal=True):
                                dpg.add_text("AI 点评", tag="watch_ai_title",
                                             color=(160, 200, 255))
                                dpg.add_text("", tag="watch_ai_action", show=False)
                                dpg.add_text("", tag="watch_ai_risk", show=False)
                                dpg.add_button(label="复制", width=48, height=22,
                                               callback=on_copy_watch_ai)
                                dpg.add_button(label="打开目录", width=76, height=22,
                                               callback=on_open_report_dir)
                                dpg.add_button(label="关闭", width=48, height=22,
                                               callback=lambda: dpg.configure_item(
                                                   "watch_ai_win", show=False))
                            dpg.add_text("", tag="watch_ai_report",
                                         wrap=900, color=(120, 220, 160))
                            with dpg.child_window(height=130, border=False):
                                dpg.add_text("", tag="watch_ai_text", wrap=880,
                                             color=(230, 230, 230))
                            dpg.add_text("", tag="watch_ai_disclaimer", wrap=900,
                                         color=(150, 150, 150), show=False)
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
        # 持仓表整行悬停高亮(highlight_table_cell 染背景,不抢点击,按钮仍可点)
        dpg.add_mouse_move_handler(callback=_on_watch_row_hover)
        # 行情页分隔条拖拽:move 实时调高度, release 结束拖拽
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,
                                   callback=_mkt_drag_move)
        dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,
                                      callback=_mkt_drag_end)

    dpg.create_viewport(title="A-Share Stock Picker", width=1320, height=840)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("root", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    build_ui()
