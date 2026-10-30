"""
今日作战台 & 逐股涨跌概率 (Multi-Horizon Cockpit & Probability Desk v3)
=======================================================================
参考专业金融终端 (Wind / 同花顺 iFinD / TradingView) 布局设计：
  - 顶部 HUD 6格核心指标看板 (KPI Matrix)：一眼扫视当前市场红绿灯、广度、仓位上限、样本外胜率、净值与回撤；
  - 支持三大持有周期 (10日短波段 / 60日中线 / 120日长线) 与三视图 (今日共振池 / 全市场字典 / 历史信号记账本) 一键切换；
  - 每个上下分区均配备可拖拽高度手柄，松手自动持久化保存到 ui_layout.json。
"""
from __future__ import annotations

import threading
import time

import dearpygui.dearpygui as dpg

from app.data import database as db
from app.strategy import cockpit as ck
from app.strategy import winmodel as wm

_PICK = None
_ADD_WATCH = None
_POST_UI = None
_ON_HORIZON_CHANGED = None
_ON_TOGGLE_SIDEBAR = None
_COCKPIT = None
_WM_STATE = None
_FUNDA_MAP: dict[str, dict] = {}
_RISK_CAL_MAP: dict[str, list[dict]] = {}
_CUR_HORIZON = 10      # 10 | 60 | 120
_ALL_ROWS = []
_PICK_ROWS = []
_AI_ROWS = []
_LEDGER_INFO = {}
_VIEW_MODE = "picks"   # "picks" | "all" | "ledger" | "ai"
_SORT_KEY = "p_tb"     # 当前股票视图排序列 key
_SORT_ASC = False      # False=降序(▼), True=升序(▲)
_LEDGER_SORT_KEY = "signal_date"
_LEDGER_SORT_ASC = False
_COL_SCHEMA_MODE = None
_SKIP_NEXT_AUTO_SORT = False
_AI_SORT_TOUCHED = False

_STOCK_COL_SPECS = (
    ("代码", "code"),
    ("名称", None),
    ("细分题材", None),
    ("收盘", "close"),
    ("实战胜率(止盈)", "p_tb"),
    ("条件胜率", "p_cond"),
    ("截面概率", "p_up"),
    ("盈亏比", "payoff"),
    ("价格年分位", "pos250"),
    ("PE分位(5Y)", "pe_hist_pct"),
    ("PB分位(5Y)", "pb_hist_pct"),
    ("挂单上限", "max_buy"),
    ("ATR止损", "stop_loss"),
    ("目标止盈", "take_profit"),
    ("建议仓位", "pos_pct"),
    ("命中形态与排雷/催化", None),
    ("操作", None),
)

_LEDGER_COL_SPECS = (
    ("信号日期", "signal_date"),
    ("代码", "code"),
    ("名称", None),
    ("细分题材", None),
    ("命中策略", None),
    ("预期胜率", "p_tb"),
    ("买入日", "entry_date"),
    ("开盘买入价", "entry_price"),
    ("止损线", "stop_loss"),
    ("目标止盈", "take_profit"),
    ("平仓/最新日", "exit_date"),
    ("结算价", "exit_price"),
    ("扣费净盈亏%", "return_pct"),
    ("实盘跟踪状态", None),
)


def _dispatch_ui(fn):
    if _POST_UI is not None:
        _POST_UI(fn)
    else:
        fn()


def get_current_horizon() -> int:
    return int(_CUR_HORIZON)


def get_current_horizon_data() -> dict:
    if not _COCKPIT:
        return {}
    h_map = _COCKPIT.get("horizons") or {}
    sub = h_map.get(str(_CUR_HORIZON))
    if sub:
        return sub
    return _COCKPIT


def get_current_picks() -> list[dict]:
    """供左侧快速看盘抽屉读取当前周期的高胜率共振候选池。"""
    return list(_PICK_ROWS)


def get_stock_quant_info(code: str) -> dict | None:
    """供 K 线页与自选页查询单只股票在当前周期下的四层量化风控数据。"""
    code = str(code).zfill(6)
    for r in _ALL_ROWS:
        if r.get("code") == code:
            out = dict(r)
            out["horizon"] = _CUR_HORIZON
            return out
    return None


def mount(
    font_big=None,
    on_code=None,
    on_add_watch=None,
    post_ui=None,
    on_horizon_changed=None,
    section_heights=None,
    add_drag_handle=None,
    on_reset_layout=None,
    on_toggle_sidebar=None,
):
    global _PICK, _ADD_WATCH, _POST_UI, _ON_HORIZON_CHANGED, _ON_TOGGLE_SIDEBAR
    _PICK = on_code
    _ADD_WATCH = on_add_watch
    _POST_UI = post_ui
    _ON_HORIZON_CHANGED = on_horizon_changed
    _ON_TOGGLE_SIDEBAR = on_toggle_sidebar
    sh = section_heights or {}

    # ---- 0. 顶栏：侧栏折叠开关 + 持有周期模式一键切换 + 重置布局 ----
    with dpg.group(horizontal=True):
        if _ON_TOGGLE_SIDEBAR is not None:
            dpg.add_button(
                label="◀ 侧栏", tag="prob_toggle_side_btn",
                width=62, height=28, callback=_ON_TOGGLE_SIDEBAR,
            )
        dpg.add_text(" 持有周期:", color=(255, 210, 110))
        dpg.add_button(
            label="● 短波段 (10日·两周)", tag="prob_h_10",
            width=152, height=28, callback=lambda: switch_horizon(10),
        )
        dpg.add_button(
            label="○ 中线 (60日·一季度)", tag="prob_h_60",
            width=152, height=28, callback=lambda: switch_horizon(60),
        )
        dpg.add_button(
            label="○ 长线 (120日·半年)", tag="prob_h_120",
            width=152, height=28, callback=lambda: switch_horizon(120),
        )
        dpg.add_text(
            " (拖拽横条可调框高·自动保存)",
            color=(140, 165, 185),
        )
        if on_reset_layout is not None:
            dpg.add_button(
                label="重置框高", width=68, height=26,
                callback=on_reset_layout,
            )

    # ---- 1. 顶部：专业终端 HUD 6格核心指标看板 + 策略精要 (可上下拖拽调节高度) ----
    with dpg.child_window(
        tag="prob_regime_box",
        height=int(sh.get("prob_regime_box", 136)),
        border=True,
    ):
        with dpg.table(
            header_row=False, resizable=False,
            policy=dpg.mvTable_SizingStretchProp,
            borders_innerV=True, borders_outerV=True,
            borders_innerH=False, borders_outerH=True,
        ):
            for _ in range(6):
                dpg.add_table_column()
            with dpg.table_row():
                # Card 1: 市场环境闸门
                with dpg.group():
                    dpg.add_text("① 市场环境闸门", color=(140, 155, 175))
                    dpg.add_text("加载中...", tag="hud_c1_val", color=(80, 220, 120))
                    dpg.add_text("", tag="hud_c1_sub", color=(165, 170, 180))
                # Card 2: 全市场广度与指数
                with dpg.group():
                    dpg.add_text("② 趋势广度与指数", color=(140, 155, 175))
                    dpg.add_text("—", tag="hud_c2_val", color=(225, 225, 230))
                    dpg.add_text("", tag="hud_c2_sub", color=(165, 170, 180))
                # Card 3: 建议仓位与过审数
                with dpg.group():
                    dpg.add_text("③ 建议总仓位上限", color=(140, 155, 175))
                    dpg.add_text("—", tag="hud_c3_val", color=(255, 205, 95))
                    dpg.add_text("", tag="hud_c3_sub", color=(165, 170, 180))
                # Card 4: 样本外绿灯胜率
                with dpg.group():
                    dpg.add_text("④ 样本外(24-26)绿灯胜率", color=(140, 155, 175))
                    dpg.add_text("—", tag="hud_c4_val", color=(235, 85, 85))
                    dpg.add_text("", tag="hud_c4_sub", color=(165, 170, 180))
                # Card 5: 样本外组合累计净值
                with dpg.group():
                    dpg.add_text("⑤ 样本外组合累计收益", color=(140, 155, 175))
                    dpg.add_text("—", tag="hud_c5_val", color=(235, 85, 85))
                    dpg.add_text("", tag="hud_c5_sub", color=(165, 170, 180))
                # Card 6: 样本外回撤与记账本
                with dpg.group():
                    dpg.add_text("⑥ 组合最大回撤 / 记账本", color=(140, 155, 175))
                    dpg.add_text("—", tag="hud_c6_val", color=(80, 215, 135))
                    dpg.add_text("", tag="hud_c6_sub", color=(165, 170, 180))

        # 精炼策略与执行铁律提示行 (保留旧 tag 兼容)
        dpg.add_text("", tag="prob_metrics", wrap=1150, color=(175, 215, 190))
        dpg.add_text("", tag="prob_strategy", wrap=1150, color=(240, 185, 105))
        dpg.add_text("", tag="prob_regime_badge", show=False)
        dpg.add_text("", tag="prob_regime_metrics", show=False)
        dpg.add_text("", tag="prob_conclusion", show=False)
        dpg.add_text("", tag="prob_meaning", show=False)
    if add_drag_handle is not None:
        add_drag_handle("prob_regime_box")

    # ---- 2. 控制栏：三视图切换 + 搜索 + 排序 + 重算 ----
    with dpg.group(horizontal=True):
        dpg.add_button(
            label="★ 今日高胜率共振池", tag="prob_btn_view_picks",
            width=156, height=28, callback=lambda: _switch_view("picks"),
        )
        dpg.add_button(
            label="全市场概率字典(5000+)", tag="prob_btn_view_all",
            width=165, height=28, callback=lambda: _switch_view("all"),
        )
        dpg.add_button(
            label="历史信号实盘记账本", tag="prob_btn_view_ledger",
            width=155, height=28, callback=lambda: _switch_view("ledger"),
        )
        dpg.add_text(" 搜索:")
        dpg.add_input_text(
            tag="prob_search", hint="代码/名称/细分题材/策略/排雷", width=165,
            callback=lambda s, a: _render_table(a), on_enter=False,
        )
        dpg.add_button(
            label="清空", width=42, height=26,
            callback=lambda: (dpg.set_value("prob_search", ""), _render_table("")),
        )
        dpg.add_combo(
            [
                "按实战胜率(三重屏障)", "按条件胜率", "按截面概率", "按历史盈亏比",
                "按PE历史分位(低优先)", "按PB历史分位(低优先)", "按价格年分位(低优先)",
            ],
            default_value="按实战胜率(三重屏障)", width=165, tag="prob_sort_combo",
            callback=_on_sort_change,
        )
        dpg.add_button(
            label="重算三周期模型", tag="prob_run_btn",
            callback=_on_run, width=116, height=28,
        )
        dpg.add_text("", tag="prob_status", color=(255, 200, 100))

    # ---- 2b. 本地 AI 自然语言选股栏 (调用本地 config.json 模型 + 多因子/PIT财报/分红/排雷共振过滤) ----
    with dpg.group(horizontal=True):
        dpg.add_text("🤖 AI选股:", color=(255, 215, 110))
        dpg.add_input_text(
            tag="prob_ai_query_in",
            hint="输入自然语言选股条件，如：市盈率历史百分位30%以下，业绩持续增长的大蓝筹股",
            default_value="市盈率历史百分位30%以下，业绩持续增长的大蓝筹股",
            width=450,
            on_enter=True,
            callback=lambda s, a: _on_ai_stock_screen(),
        )
        dpg.add_button(
            label="🤖 AI 智能选股", tag="prob_ai_btn",
            width=112, height=26,
            callback=lambda: _on_ai_stock_screen(),
        )
        dpg.add_text(" 灵感:", color=(155, 195, 240))
        for _lbl, _q in (
            ("红利低波蓝筹", "持续稳定分红、股息率高于3%、市盈率历史百分位30%以下的大蓝筹"),
            ("GARP成长低估", "PEG小于1、连续3个季度业绩增长、ROE大于12%的股票"),
            ("巴菲特护城河", "连续3年ROE大于15%、毛利率大于35%、负债率低于50%的白马龙头"),
            ("股东增持回购", "有股东增持或股份回购计划、无减持排雷、市盈率历史百分位50%以下、业绩增长"),
            ("多头主力抢筹", "均线多头排列、主力净流入、实战胜率53%以上、无排雷风险"),
        ):
            dpg.add_button(
                label=_lbl, height=24, user_data=_q,
                callback=lambda s, a, u: (dpg.set_value("prob_ai_query_in", u), _on_ai_stock_screen()),
            )

    # ---- 3. 选中个股的四层诊断与执行计划卡片 (可上下拖拽调节高度) ----
    with dpg.child_window(
        tag="prob_detail_box",
        height=int(sh.get("prob_detail_box", 66)),
        border=True,
    ):
        dpg.add_text(
            "单击任意一行查看【个股四层诊断 + 高阶驱动因子 + 主力资金流 + 止损/止盈计划】；双击代码直接打开 K 线图。",
            tag="prob_detail", wrap=1150, color=(230, 230, 230),
        )
    if add_drag_handle is not None:
        add_drag_handle("prob_detail_box")

    # ---- 4. 主表格 (支持点击任意可量化表头列排序、高亮当前排序列、再点一次反向排序) ----
    with dpg.table(
        tag="prob_table", header_row=True, resizable=True,
        policy=dpg.mvTable_SizingStretchProp,
        height=-1, scrollY=True, freeze_rows=1,
        sortable=True, sort_tristate=False, callback=_on_prob_table_sort,
    ):
        _ensure_columns("")

    refresh()


def switch_horizon(horizon: int):
    global _CUR_HORIZON
    _CUR_HORIZON = int(horizon)
    _sync_horizon_buttons()
    refresh()
    if _ON_HORIZON_CHANGED is not None:
        try:
            _ON_HORIZON_CHANGED(_CUR_HORIZON)
        except Exception:
            pass


def _sync_horizon_buttons():
    labels = {
        10: ("prob_h_10", "短波段 (10日·两周)"),
        60: ("prob_h_60", "中线 (60日·一季度)"),
        120: ("prob_h_120", "长线 (120日·半年)"),
    }
    for h, (tag, text) in labels.items():
        if dpg.does_item_exist(tag):
            prefix = "● " if h == _CUR_HORIZON else "○ "
            dpg.configure_item(tag, label=prefix + text)


def _switch_view(mode: str):
    global _VIEW_MODE
    _VIEW_MODE = mode
    word = ""
    if dpg.does_item_exist("prob_search"):
        word = dpg.get_value("prob_search") or ""
    _render_table(word)


def _on_sort_change(sender, app_data):
    global _SORT_KEY, _SORT_ASC, _AI_SORT_TOUCHED
    _AI_SORT_TOUCHED = True
    _SORT_ASC = False
    if app_data == "按截面概率":
        _SORT_KEY = "p_up"
    elif app_data == "按历史盈亏比":
        _SORT_KEY = "payoff"
    elif app_data == "按条件胜率":
        _SORT_KEY = "p_cond"
    elif app_data == "按PE历史分位(低优先)":
        _SORT_KEY = "pe_hist_pct"
        _SORT_ASC = True
    elif app_data == "按PB历史分位(低优先)":
        _SORT_KEY = "pb_hist_pct"
        _SORT_ASC = True
    elif app_data == "按价格年分位(低优先)":
        _SORT_KEY = "pos250"
        _SORT_ASC = True
    else:
        _SORT_KEY = "p_tb"
    word = dpg.get_value("prob_search") if dpg.does_item_exist("prob_search") else ""
    _render_table(word or "")


def _on_prob_table_sort(sender, app_data):
    """点击作战台表格表头列时触发：可量化列按该列排序并高亮，再点一次反向排序；非量化列无反应。"""
    global _SKIP_NEXT_AUTO_SORT, _SORT_KEY, _SORT_ASC, _LEDGER_SORT_KEY, _LEDGER_SORT_ASC, _AI_SORT_TOUCHED
    if _SKIP_NEXT_AUTO_SORT:
        _SKIP_NEXT_AUTO_SORT = False
        return
    if not app_data:
        return
    col_id = app_data[0][0]
    if not dpg.does_item_exist(col_id):
        return
    ud = dpg.get_item_user_data(col_id)
    if not ud or not ud[2]:
        return
    _, _, s_key = ud
    word = (dpg.get_value("prob_search") or "").strip() if dpg.does_item_exist("prob_search") else ""
    if _VIEW_MODE == "ledger" and not word:
        if _LEDGER_SORT_KEY == s_key:
            _LEDGER_SORT_ASC = not _LEDGER_SORT_ASC
        else:
            _LEDGER_SORT_KEY = s_key
            _LEDGER_SORT_ASC = (s_key == "code")
    else:
        _AI_SORT_TOUCHED = True
        if _SORT_KEY == s_key:
            _SORT_ASC = not _SORT_ASC
        else:
            _SORT_KEY = s_key
            _SORT_ASC = s_key in ("code", "pe_hist_pct", "pb_hist_pct", "pos250")
    _render_table(word)


def _enrich_rows_with_funda(rows: list[dict]):
    """将本地 fundamental 特征库与 risk_calendar(增减持/回购/解禁/业绩预告)实时注入股票行。"""
    global _FUNDA_MAP, _RISK_CAL_MAP
    if not _FUNDA_MAP:
        try:
            _FUNDA_MAP = db.load_fundamental_map()
        except Exception:
            _FUNDA_MAP = {}
    if not _RISK_CAL_MAP:
        try:
            _RISK_CAL_MAP = db.load_risk_calendar_map()
        except Exception:
            _RISK_CAL_MAP = {}
    for r in rows:
        code = str(r.get("code") or "").zfill(6)
        fi = _FUNDA_MAP.get(code)
        if fi:
            if r.get("pe_hist_pct") is None:
                r["pe_hist_pct"] = fi.get("pe_hist_pct")
            if r.get("pe_ttm") is None:
                r["pe_ttm"] = fi.get("pe_ttm")
            if r.get("pb") is None:
                r["pb"] = fi.get("pb")
            if r.get("pb_hist_pct") is None:
                r["pb_hist_pct"] = fi.get("pb_hist_pct")
            if r.get("pe_ind_pct") is None:
                r["pe_ind_pct"] = fi.get("pe_ind_pct")
        evs = _RISK_CAL_MAP.get(code) or []
        if evs:
            cur_setups = list(r.get("setups") or [])
            cur_veto = list(r.get("veto") or [])
            cur_str = " ".join(cur_setups + cur_veto)
            sh_tags = []
            for ev in evs:
                et = ev.get("event_type") or ""
                sev = ev.get("severity") or ""
                summ = ev.get("summary") or ""
                if not summ:
                    continue
                sh_tags.append(summ)
                if et in ("股东增持", "股份回购", "股东减持") and summ not in cur_str:
                    if sev == "veto":
                        cur_veto.append(summ)
                        cur_setups.append("🚨" + summ)
                    elif sev == "warn":
                        cur_setups.append("⚠" + summ)
                    elif sev == "bull":
                        cur_setups.insert(0, "★" + summ)
            r["setups"] = cur_setups
            r["veto"] = cur_veto
            r["holder_events"] = sh_tags


def refresh():
    global _COCKPIT, _WM_STATE, _ALL_ROWS, _PICK_ROWS, _LEDGER_INFO, _FUNDA_MAP, _RISK_CAL_MAP
    _COCKPIT = ck.load_cockpit_state()
    _WM_STATE = wm.load_state()
    try:
        _FUNDA_MAP = db.load_fundamental_map()
        _RISK_CAL_MAP = db.load_risk_calendar_map()
    except Exception:
        pass
    _sync_horizon_buttons()

    h_data = get_current_horizon_data()
    if h_data and h_data.get("stocks"):
        _ALL_ROWS = list(h_data.get("stocks") or [])
        _PICK_ROWS = list(h_data.get("today_picks") or [])
        _LEDGER_INFO = dict(h_data.get("ledger") or {})
    else:
        _ALL_ROWS = list((_WM_STATE or {}).get("stocks") or [])
        _PICK_ROWS = [r for r in _ALL_ROWS if r.get("top")][:30]
        _LEDGER_INFO = {}
    _enrich_rows_with_funda(_ALL_ROWS)
    _enrich_rows_with_funda(_PICK_ROWS)
    _paint(h_data)
    word = ""
    if dpg.does_item_exist("prob_search"):
        word = dpg.get_value("prob_search") or ""
    _render_table(word)


def _paint(st: dict):
    if not st:
        _set("hud_c1_val", "未初始化")
        _set("prob_metrics", "尚未生成四层量化状态，请点「重算三周期模型」。")
        return

    h = int(st.get("horizon") or _CUR_HORIZON)
    h_label = ck.HORIZON_LABELS.get(h, f"{h}日")

    t_reg = st.get("today_regime") or {}
    reg_code = t_reg.get("regime") or ck.REGIME_NEUTRAL
    reg_cn = t_reg.get("regime_cn") or ck.regime_cn_of(reg_code, h)
    pos_cap = t_reg.get("pos_cap", 0.4)
    b20 = t_reg.get("breadth_ma20")
    b60 = t_reg.get("breadth_ma60")
    u1 = t_reg.get("breadth_up1")
    ic = t_reg.get("idx_close")
    im20 = t_reg.get("idx_ma20")

    if reg_code == ck.REGIME_BULL:
        badge_col = (65, 225, 130)
    elif reg_code == ck.REGIME_BEAR:
        badge_col = (240, 80, 80)
    else:
        badge_col = (255, 200, 70)

    te_regs = st.get("test_regimes") or {}
    r_bull = te_regs.get(ck.REGIME_BULL) or {}
    r_bear = te_regs.get(ck.REGIME_BEAR) or {}
    ts = st.get("tearsheet") or {}
    oos_g = (ts.get("oos_stats") or {}).get("gated") or {}
    veto_n = sum(1 for r in _ALL_ROWS if r.get("veto"))

    # 填充 6 格 HUD KPI 矩阵
    short_reg = reg_cn.split("(")[0].strip()
    _set("hud_c1_val", short_reg)
    if dpg.does_item_exist("hud_c1_val"):
        dpg.configure_item("hud_c1_val", color=badge_col)
    _set("hud_c1_sub", f"模式: {h_label} ({st.get('asof')})")

    _set("hud_c2_val", f"MA20: {_pct(b20)} | MA60: {_pct(b60)}")
    idx_sub = f"上证 {ic:.0f} (MA20={im20:.0f}) · 涨占比{_pct(u1)}" if (ic and im20) else f"当日上涨占比 {_pct(u1)}"
    _set("hud_c2_sub", idx_sub)

    _set("hud_c3_val", f"仓位上限 {_pct(pos_cap)}")
    _set("hud_c3_sub", f"今日过审 {len(_PICK_ROWS)} 只 (已排雷 {veto_n} 只)")

    w_tb_bull = r_bull.get("win_tb_top_dec") or r_bull.get("win_top_dec")
    w_fix_bull = r_bull.get("win_top_dec")
    _set("hud_c4_val", f"固定 {_pct(w_fix_bull)} / 止盈 {_pct(w_tb_bull)}")
    _set("hud_c4_sub", f"单期净赚 {_pct(r_bull.get('mean_tb_top') or r_bull.get('mean_net_top'))} (红灯期 {_pct(r_bear.get('win_all'))})")

    tot_ret = oos_g.get("tot_ret")
    cagr = oos_g.get("cagr")
    _set("hud_c5_val", f"{_pct(tot_ret)} (年化 {_pct(cagr)})")
    if dpg.does_item_exist("hud_c5_val"):
        dpg.configure_item("hud_c5_val", color=(235, 85, 85) if (tot_ret and tot_ret > 0) else (80, 210, 120))
    _set("hud_c5_sub", f"夏普 {_num(oos_g.get('sharpe'))} | 卡玛 {_num(oos_g.get('calmar'))}")

    mdd = oos_g.get("max_dd")
    led_w = _LEDGER_INFO.get("win_rate")
    _set("hud_c6_val", f"最大回撤 {_pct(mdd)}")
    _set(
        "hud_c6_sub",
        f"记账本 {_LEDGER_INFO.get('n_closed', 0)}笔 胜率 {_pct(led_w)} ({_LEDGER_INFO.get('avg_ret_pct', 0):+.2f}%)"
        if led_w is not None else "历史信号自动跟踪中",
    )

    strats = st.get("strategies") or []
    v_parts = []
    for s in strats[:3]:
        tg = s.get("test_gated") or {}
        w_show = tg.get("win_gross")
        wtb_show = tg.get("win_tb")
        n_show = tg.get("mean_tb_net") or tg.get("mean_net")
        v_parts.append(
            f"{s['name']}(胜率 {_pct(w_show)}/止盈 {_pct(wtb_show)}, 净赚 {_pct(n_show)}, 盈亏比 {_num(tg.get('tb_payoff') or tg.get('payoff'))})"
        )
    _set("prob_metrics", "★ 绿灯期主力策略样本外表现: " + "  |  ".join(v_parts))

    sl_m, tp_m, _ = ck.TB_PARAMS.get(h, (1.5, 2.5, 0.01))
    p_tbs = [r.get("p_tb", r.get("p_cond")) for r in _ALL_ROWS if r.get("p_cond") is not None]
    span_s = f"全市场 {len(_ALL_ROWS)} 只实战胜率分布 {_pct(min(p_tbs))}~{_pct(max(p_tbs))} · " if p_tbs else ""
    _set(
        "prob_strategy",
        f"{span_s}执行纪律：次日开盘高开超「挂单上限」不追 → 盘中触 +{tp_m}×ATR 止盈落袋 → 跌破 -{sl_m}×ATR 无条件止损（最长持有 {h} 日）。",
    )

    _set(
        "side_strategy",
        f"{h_label} · {short_reg} | 仓位上限 {_pct(pos_cap)} | 过审 {len(_PICK_ROWS)} 只",
    )
    _set("prob_status", f"[{h}日模式] {st.get('asof')}")


def _ensure_columns(word: str):
    """按当前视图模式确保表头列(slot 0)存在，仅清空数据行(slot 1)并高亮当前排序列。"""
    global _COL_SCHEMA_MODE, _SKIP_NEXT_AUTO_SORT
    is_ledger = (_VIEW_MODE == "ledger" and not (word or "").strip())
    schema_mode = "ledger" if is_ledger else "stocks"
    specs = _LEDGER_COL_SPECS if is_ledger else _STOCK_COL_SPECS
    cols_0 = dpg.get_item_children("prob_table", 0) or []

    if _COL_SCHEMA_MODE != schema_mode or len(cols_0) != len(specs):
        _SKIP_NEXT_AUTO_SORT = True
        dpg.delete_item("prob_table", children_only=True)
        for idx, (lbl, s_key) in enumerate(specs):
            if s_key is None:
                dpg.add_table_column(
                    label=lbl, parent="prob_table",
                    no_sort=True, user_data=(idx, lbl, None),
                )
            else:
                dpg.add_table_column(
                    label=lbl, parent="prob_table",
                    no_sort=False, prefer_sort_descending=True, prefer_sort_ascending=False,
                    user_data=(idx, lbl, s_key),
                )
        _COL_SCHEMA_MODE = schema_mode
        cols_0 = dpg.get_item_children("prob_table", 0) or []
    else:
        for r_id in dpg.get_item_children("prob_table", 1) or []:
            dpg.delete_item(r_id)

    act_key = _LEDGER_SORT_KEY if is_ledger else _SORT_KEY
    act_asc = _LEDGER_SORT_ASC if is_ledger else _SORT_ASC
    show_sort_hl = not (_VIEW_MODE == "ai" and not _AI_SORT_TOUCHED and not is_ledger)
    for col_id in cols_0:
        ud = dpg.get_item_user_data(col_id)
        if not ud:
            continue
        idx, base_lbl, s_key = ud
        if show_sort_hl and s_key and s_key == act_key:
            arrow = "▲" if act_asc else "▼"
            dpg.configure_item(col_id, label=f"★ {base_lbl} {arrow}")
            try:
                dpg.highlight_table_column("prob_table", idx, (42, 82, 145, 70))
            except Exception:
                pass
        else:
            dpg.configure_item(col_id, label=base_lbl)
            try:
                dpg.unhighlight_table_column("prob_table", idx)
            except Exception:
                pass


def _render_table(word: str):
    if not dpg.does_item_exist("prob_table"):
        return
    _ensure_columns(word)

    led_rows = list(_LEDGER_INFO.get("rows") or [])
    if dpg.does_item_exist("prob_btn_view_picks"):
        dpg.configure_item(
            "prob_btn_view_picks",
            label=f"{'● ' if _VIEW_MODE == 'picks' else ''}★ 今日高胜率共振池 ({len(_PICK_ROWS)})",
        )
    if dpg.does_item_exist("prob_btn_view_all"):
        dpg.configure_item(
            "prob_btn_view_all",
            label=f"{'● ' if _VIEW_MODE == 'all' else ''}全市场概率字典 ({len(_ALL_ROWS)})",
        )
    if dpg.does_item_exist("prob_btn_view_ledger"):
        dpg.configure_item(
            "prob_btn_view_ledger",
            label=f"{'● ' if _VIEW_MODE == 'ledger' else ''}历史信号记账本 ({len(led_rows)})",
        )

    if _VIEW_MODE == "ledger" and not (word or "").strip():
        _render_ledger_rows(led_rows)
        return

    rows = _filter(word)
    for row in rows:
        code = row["code"]
        name = row.get("name") or ""
        close = float(row.get("close") or 0.0)
        p_tb = row.get("p_tb", row.get("p_cond", row.get("p_up")))
        p_cond = row.get("p_cond", row.get("p_up"))
        p_up = row.get("p_up")
        payoff = row.get("payoff", 1.25)
        pos250 = row.get("pos250")
        pe_pct = row.get("pe_hist_pct")
        pe_ttm = row.get("pe_ttm")
        pb_pct = row.get("pb_hist_pct")
        pb_val = row.get("pb")
        if isinstance(row.get("_funda"), dict):
            if pe_pct is None:
                pe_pct = row["_funda"].get("pe_hist_pct")
            if pe_ttm is None:
                pe_ttm = row["_funda"].get("pe_ttm")
            if pb_pct is None:
                pb_pct = row["_funda"].get("pb_hist_pct")
            if pb_val is None:
                pb_val = row["_funda"].get("pb")
        max_buy = row.get("max_buy", close * 1.015)
        stop_loss = row.get("stop_loss", close * 0.95)
        take_profit = row.get("take_profit", close * 1.08)
        pos_pct = row.get("pos_pct", 0.0)
        veto = row.get("veto") or []
        setups = row.get("setups") or ([] if not row.get("top") else ["截面前20%"])
        setup_str = " / ".join(setups) if setups else ("截面高分" if row.get("top") else "—")

        p_col = (235, 75, 75) if (p_tb and p_tb >= 0.52) else (
            (70, 195, 120) if (p_tb and p_tb < 0.45) else (225, 225, 225)
        )
        setup_col = (240, 95, 80) if veto else ((150, 210, 255) if setups else (150, 150, 150))

        # PE分位(5Y)显示与颜色：低估(<=30%)标绿，高估(>=75%)标橙红，亏损(PE<=0)标灰并提示亏损
        if pe_ttm is not None and not (pe_ttm != pe_ttm) and float(pe_ttm) <= 0:
            pe_pct_str = "亏损"
            pe_pct_col = (140, 140, 145)
        elif pe_pct is not None and not (pe_pct != pe_pct):
            fp = float(pe_pct)
            pe_pct_str = f"{fp:.1f}%"
            pe_pct_col = (80, 215, 140) if fp <= 30.0 else ((240, 110, 90) if fp >= 75.0 else (220, 220, 225))
        else:
            pe_pct_str = "—"
            pe_pct_col = (140, 140, 145)

        # PB分位(5Y)显示与颜色：低估(<=30%)标绿，高估(>=75%)标橙红
        if pb_val is not None and not (pb_val != pb_val) and float(pb_val) <= 0:
            pb_pct_str = "资不抵债"
            pb_pct_col = (140, 140, 145)
        elif pb_pct is not None and not (pb_pct != pb_pct):
            fb = float(pb_pct)
            pb_pct_str = f"{fb:.1f}%"
            pb_pct_col = (80, 215, 140) if fb <= 30.0 else ((240, 110, 90) if fb >= 75.0 else (220, 220, 225))
        else:
            pb_pct_str = "—"
            pb_pct_col = (140, 140, 145)

        with dpg.table_row(parent="prob_table"):
            sel = dpg.add_selectable(
                label=code, span_columns=False, user_data=code,
                callback=_on_row,
            )
            dpg.bind_item_theme(sel, "row_hover")
            sel_nm = dpg.add_selectable(
                label=name, span_columns=False, user_data=code,
                callback=_on_row,
            )
            dpg.bind_item_theme(sel_nm, "row_hover")
            subsec = row.get("subsector") or row.get("industry") or ""
            dpg.add_text(subsec[:9])
            dpg.add_text(f"{close:.2f}")
            dpg.add_text(_pct(p_tb), color=p_col)
            dpg.add_text(_pct(p_cond))
            dpg.add_text(_pct(p_up))
            dpg.add_text(f"{float(payoff):.2f}" if payoff is not None else "—")
            dpg.add_text(_pct(pos250) if pos250 is not None else "—")
            dpg.add_text(pe_pct_str, color=pe_pct_col)
            dpg.add_text(pb_pct_str, color=pb_pct_col)
            dpg.add_text(f"{float(max_buy):.2f}", color=(255, 210, 120))
            dpg.add_text(f"{float(stop_loss):.2f}", color=(80, 200, 130))
            dpg.add_text(f"{float(take_profit):.2f}", color=(235, 90, 90))
            dpg.add_text(f"{float(pos_pct):.1f}%", color=(150, 150, 150) if veto else (225, 225, 225))
            dpg.add_text(setup_str, color=setup_col)
            note = f"[{_CUR_HORIZON}日] 实战胜率{_pct(p_tb)} 止损{stop_loss:.2f} 止盈{take_profit:.2f} {setup_str}"
            dpg.add_button(
                label="+自选", width=50,
                user_data=(code, name, close, note),
                callback=lambda s, a, u: _ADD_WATCH(u) if _ADD_WATCH else None,
            )


def _render_ledger_rows(led_rows: list[dict]):
    rows = list(led_rows or [])
    k = _LEDGER_SORT_KEY
    asc = _LEDGER_SORT_ASC
    if k in ("signal_date", "code", "entry_date", "exit_date"):
        valid = [r for r in rows if r.get(k) and str(r.get(k)) != "—"]
        nulls = [r for r in rows if not r.get(k) or str(r.get(k)) == "—"]
        valid.sort(key=lambda r: (str(r.get(k) or ""), str(r.get("code") or "")), reverse=not asc)
        rows = valid + nulls
    else:
        def _lval(r):
            if k == "p_tb":
                return r.get("p_tb") if r.get("p_tb") is not None else r.get("p_cond")
            return r.get(k)
        valid = [r for r in rows if _lval(r) is not None]
        nulls = [r for r in rows if _lval(r) is None]
        valid.sort(key=lambda r: (float(_lval(r)), str(r.get("code") or "")), reverse=not asc)
        rows = valid + nulls

    for r in rows:
        code = str(r.get("code") or "")
        ret = r.get("return_pct")
        status = str(r.get("status") or "")
        ret_col = (235, 75, 75) if (ret is not None and float(ret) > 0) else (
            (60, 195, 115) if (ret is not None and float(ret) < 0) else (190, 190, 190)
        )
        st_col = (235, 90, 90) if "止盈" in status else (
            (70, 200, 120) if "止损" in status else (230, 200, 110)
        )
        with dpg.table_row(parent="prob_table"):
            dpg.add_text(str(r.get("signal_date") or ""))
            sel = dpg.add_selectable(label=code, span_columns=False, user_data=code, callback=_on_row)
            dpg.bind_item_theme(sel, "row_hover")
            dpg.add_text(str(r.get("name") or ""))
            dpg.add_text(str(r.get("industry") or "")[:9])
            dpg.add_text(str(r.get("setups") or "")[:16], color=(150, 210, 255))
            dpg.add_text(_pct(r.get("p_tb") or r.get("p_cond")))
            dpg.add_text(str(r.get("entry_date") or "—"))
            ep = r.get("entry_price")
            dpg.add_text(f"{float(ep):.2f}" if ep else "—")
            sl = r.get("stop_loss")
            tp = r.get("take_profit")
            dpg.add_text(f"{float(sl):.2f}" if sl else "—", color=(80, 200, 130))
            dpg.add_text(f"{float(tp):.2f}" if tp else "—", color=(235, 90, 90))
            dpg.add_text(str(r.get("exit_date") or "—"))
            xp = r.get("exit_price")
            dpg.add_text(f"{float(xp):.2f}" if xp else "—")
            dpg.add_text(f"{float(ret):+.2f}%" if ret is not None else "—", color=ret_col)
            dpg.add_text(status, color=st_col)


def _stock_sort_val(r: dict, k: str):
    if k == "p_tb":
        return r.get("p_tb", r.get("p_cond", r.get("p_up")))
    if k == "p_cond":
        return r.get("p_cond", r.get("p_up"))
    if k == "pe_hist_pct":
        pe = r.get("pe_ttm")
        if pe is None and isinstance(r.get("_funda"), dict):
            pe = r["_funda"].get("pe_ttm")
        if pe is not None and not (pe != pe) and float(pe) <= 0:
            return None  # 亏损股在按PE分位排序时排在末尾
        v = r.get("pe_hist_pct")
        if v is None and isinstance(r.get("_funda"), dict):
            v = r["_funda"].get("pe_hist_pct")
        return v
    if k == "pb_hist_pct":
        pb = r.get("pb")
        if pb is None and isinstance(r.get("_funda"), dict):
            pb = r["_funda"].get("pb")
        if pb is not None and not (pb != pb) and float(pb) <= 0:
            return None
        v = r.get("pb_hist_pct")
        if v is None and isinstance(r.get("_funda"), dict):
            v = r["_funda"].get("pb_hist_pct")
        return v
    c = float(r.get("close") or 0.0)
    if k == "max_buy":
        return r.get("max_buy", c * 1.015)
    if k == "stop_loss":
        return r.get("stop_loss", c * 0.95)
    if k == "take_profit":
        return r.get("take_profit", c * 1.08)
    return r.get(k)


def _filter(word: str) -> list[dict]:
    text = (word or "").strip()
    if _VIEW_MODE == "ai" and _AI_ROWS and not text:
        base = list(_AI_ROWS)
        if not _AI_SORT_TOUCHED:
            return base[:250]
    else:
        base = list(_ALL_ROWS) if (text or _VIEW_MODE == "all") else list(_PICK_ROWS)
    if text:
        base = [
            r for r in base
            if text in r["code"]
            or text in (r.get("name") or "")
            or text in (r.get("industry") or "")
            or text in (r.get("subsector") or "")
            or any(text in s for s in (r.get("setups") or []))
            or any(text in v for v in (r.get("veto") or []))
        ]
    if _SORT_KEY == "code":
        base.sort(key=lambda r: str(r.get("code") or ""), reverse=not _SORT_ASC)
        return base[:250]
    valid = [r for r in base if _stock_sort_val(r, _SORT_KEY) is not None]
    nulls = [r for r in base if _stock_sort_val(r, _SORT_KEY) is None]
    valid.sort(
        key=lambda r: (float(_stock_sort_val(r, _SORT_KEY)), float(r.get("p_tb") or 0.0), str(r.get("code") or "")),
        reverse=not _SORT_ASC,
    )
    return (valid + nulls)[:250]


def _on_ai_stock_screen():
    """在今日作战台上调用本地 AI 模型(config.json) + 多因子/PIT财报/排雷共振选股。"""
    global _AI_ROWS, _VIEW_MODE, _AI_SORT_TOUCHED
    q = (dpg.get_value("prob_ai_query_in") or "").strip() if dpg.does_item_exist("prob_ai_query_in") else ""
    if not q:
        return
    _AI_SORT_TOUCHED = False
    if dpg.does_item_exist("prob_ai_btn"):
        dpg.configure_item("prob_ai_btn", enabled=False)
    _set("prob_status", f"🤖 正在调用本地 AI 解析选股条件...")

    def worker():
        global _AI_ROWS, _VIEW_MODE
        try:
            from app.ai import fund_screener

            def _early_render(spec, top_rows, total_n):
                def _do_early():
                    global _AI_ROWS, _VIEW_MODE
                    _AI_ROWS = list(top_rows or [])
                    _VIEW_MODE = "ai"
                    fnl = spec.get("funnel") or f"命中 {total_n} 只"
                    _set("prob_status", f"🤖 {fnl} (正在调用本地 AI 生成解读...)")
                    _set(
                        "prob_detail",
                        f"🤖 AI 选股条件: 【{spec.get('explain', '')}】 | 漏斗: {fnl}\n正在调用本地 AI 模型生成组合画像与精选看点...",
                    )
                    _render_table("")
                _dispatch_ui(_do_early)

            res = fund_screener.run_ai_stock_screen(
                q,
                list(_ALL_ROWS),
                progress_cb=lambda m: _dispatch_ui(lambda msg=m: _set("prob_status", msg)),
                on_rows_ready=_early_render,
            )

            def apply_ui():
                global _AI_ROWS, _VIEW_MODE
                if not res.get("ok"):
                    _set("prob_status", f"⚠️ {res.get('error')}")
                    return
                _AI_ROWS = list(res.get("rows") or [])
                _VIEW_MODE = "ai"
                spec = res.get("spec") or {}
                fnl = spec.get("funnel") or f"命中 {res.get('total_matched', len(_AI_ROWS))} 只"
                _set("prob_status", f"🤖 {fnl}")
                _set(
                    "prob_detail",
                    f"🤖 AI 选股条件: 【{spec.get('explain', '')}】 | 漏斗: {fnl}\n{res.get('ai_summary', '')}",
                )
                _render_table("")

            _dispatch_ui(apply_ui)
        except Exception as e:
            _dispatch_ui(lambda err=e: _set("prob_status", f"AI 选股异常: {err}"))
        finally:
            _dispatch_ui(
                lambda: dpg.configure_item("prob_ai_btn", enabled=True)
                if dpg.does_item_exist("prob_ai_btn") else None
            )

    threading.Thread(target=worker, daemon=True).start()


def _on_row(sender, app_data, user_data):
    code = user_data
    row = next((r for r in (_AI_ROWS if _VIEW_MODE == "ai" else _ALL_ROWS) if r["code"] == code), None)
    if row is None:
        row = next((r for r in _ALL_ROWS if r["code"] == code), None)
    if row is not None:
        _show_detail(row)
    now = time.monotonic()
    last = _on_row.last
    _on_row.last = (code, now)
    if last and last[0] == code and (now - last[1]) <= 0.40 and _PICK:
        _on_row.last = (code, 0.0)
        hint = None
        if row is not None:
            setups = "、".join(row.get("setups") or []) or "全市场截面评估"
            hint = (
                f"[{_CUR_HORIZON}日周期] 四层量化：{setups} | "
                f"三重屏障实战胜率 {_pct(row.get('p_tb'))} | 条件胜率 {_pct(row.get('p_cond'))} | "
                f"年价格分位 {_pct(row.get('pos250'))} | 止损 {row.get('stop_loss')} | 止盈 {row.get('take_profit')}"
            )
        _PICK(code, hint=hint)


_on_row.last = None


def _show_detail(row: dict):
    p_tb = row.get("p_tb", row.get("p_cond"))
    p_cond = row.get("p_cond")
    p_up = row.get("p_up")
    bits = []
    for d in row.get("drivers") or []:
        c = d.get("contrib") or 0
        side = "推高胜率" if c > 0 else "拉低胜率"
        bits.append(f"{d.get('name')}(z={d.get('z')}) {side}")
    why = "；".join(bits) if bits else "无显著偏离特征"
    veto = row.get("veto") or []
    veto_txt = f" | 🚨排雷拦截: {'、'.join(veto)}" if veto else " | ✅排雷通过"
    h_evs = row.get("holder_events") or []
    sh_only = [x for x in h_evs if ("增持" in x or "减持" in x or "回购" in x or "解禁" in x)]
    sh_txt = f" | 🏢股东/回购动态: {' · '.join(sh_only[:3])}" if sh_only else ""
    inflow = row.get("net_inflow_yi")
    big_r = row.get("net_big_ratio")
    flow_txt = f" | 主力净流: {float(inflow):+.2f}亿(大单净比{float(big_r)*100:+.1f}%)" if (inflow is not None and big_r is not None) else ""
    pe_ttm = row.get("pe_ttm")
    pe_pct = row.get("pe_hist_pct")
    pe_ind = row.get("pe_ind_pct")
    pb_val = row.get("pb")
    pb_pct = row.get("pb_hist_pct")
    val_parts = []
    if pe_ttm is not None and float(pe_ttm) > 0 and pe_pct is not None:
        ind_s = f"·同业{float(pe_ind):.0f}%" if pe_ind is not None else ""
        val_parts.append(f"PE={float(pe_ttm):.1f}(5Y分位{float(pe_pct):.1f}%{ind_s})")
    elif pe_ttm is not None and float(pe_ttm) <= 0:
        val_parts.append("PE=亏损")
    if pb_val is not None and float(pb_val) > 0 and pb_pct is not None:
        val_parts.append(f"PB={float(pb_val):.2f}(5Y分位{float(pb_pct):.1f}%)")
    val_txt = (" | " + " · ".join(val_parts)) if val_parts else ""
    subsec = row.get("subsector") or row.get("industry") or ""
    setups = "、".join(row.get("setups") or []) or "未触发极值形态(仅作截面展示)"
    _set(
        "prob_detail",
        f"【{_CUR_HORIZON}日周期 · {row['code']} {row.get('name') or ''} · {subsec}】 "
        f"收盘 {float(row.get('close') or 0):.2f} (价格年分位 {_pct(row.get('pos250'))}){val_txt} | "
        f"实战胜率(三重屏障) {_pct(p_tb)} | 条件胜率 {_pct(p_cond)} | 截面概率 {_pct(p_up)} | 盈亏比 {_num(row.get('payoff'))}{flow_txt}{sh_txt}{veto_txt}\n"
        f"🎯 执行计划 → 挂单上限: {row.get('max_buy'):.2f} | 🛡️ 止损线: {row.get('stop_loss'):.2f} | "
        f"🚩 目标止盈: {row.get('take_profit'):.2f} | 建议仓位: {row.get('pos_pct'):.1f}% | 形态: {setups} | 驱动: {why}",
    )


def _on_run():
    if not dpg.does_item_exist("prob_run_btn"):
        return
    dpg.configure_item("prob_run_btn", enabled=False)
    _set("prob_status", "正在后台重算 10日/60日/120日 三周期高阶模型...")

    def worker():
        try:
            ck.run_cockpit(
                rerun_winmodel=False,
                progress_cb=lambda msg: _dispatch_ui(lambda m=msg: _set("prob_status", m)),
            )
            _dispatch_ui(refresh)
            if _ON_HORIZON_CHANGED is not None:
                _dispatch_ui(lambda: _ON_HORIZON_CHANGED(_CUR_HORIZON))
            _dispatch_ui(lambda: _set("prob_status", "三周期重算完成"))
        except Exception as e:
            _dispatch_ui(lambda err=e: _set("prob_status", f"计算失败: {err}"))
        finally:
            _dispatch_ui(
                lambda: dpg.configure_item("prob_run_btn", enabled=True)
                if dpg.does_item_exist("prob_run_btn") else None
            )

    threading.Thread(target=worker, daemon=True).start()


def _set(tag: str, value: str):
    if dpg.does_item_exist(tag):
        dpg.set_value(tag, value)


def _pct(x) -> str:
    try:
        if x is None or x != x:
            return "—"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "—"


def _num(x) -> str:
    try:
        if x is None or x != x:
            return "—"
        return f"{float(x):.2f}"
    except Exception:
        return "—"
