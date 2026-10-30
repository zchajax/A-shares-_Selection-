"""
冻结协议的一次检验。测试窗只在训练窗选定持有期和权重之后进入一次。

运行:  venv\\Scripts\\python.exe -m app.strategy.validity
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..data import database as db
from . import factor_ic
from . import panel as pnl
from . import protocol as proto
from . import quantile_bt as qbt


def _cn2en():
    return {v: k for k, v in pnl._FACTOR_CN.items()}


def _num(x, nd=4):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def _pct(x):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x * 100:.3f}%"


def _corr_matrix(sub: pd.DataFrame) -> dict:
    out = {}
    for a in proto.TECH_FACTORS:
        for b in proto.TECH_FACTORS:
            if a > b:
                continue
            vals = []
            for _, g in sub.groupby("date", sort=False):
                x = g[a + "_p"]
                y = g[b + "_p"]
                m = x.notna() & y.notna()
                if int(m.sum()) < 30:
                    continue
                if float(x[m].std(ddof=0)) == 0 or float(y[m].std(ddof=0)) == 0:
                    continue
                if a == b:
                    vals.append(1.0)
                    break
                vals.append(float(x[m].corr(y[m])))
            out[(a, b)] = float(np.mean(vals)) if vals else np.nan
    return out


def _ic_maps(summary: pd.DataFrame):
    cn2en = _cn2en()
    means, irs, ts, ns = {}, {}, {}, {}
    if summary is None or summary.empty:
        return means, irs, ts, ns
    for cn, row in summary.iterrows():
        en = cn2en.get(str(cn), str(cn))
        means[en] = float(row["ic_mean"]) if pd.notna(row["ic_mean"]) else np.nan
        irs[en] = float(row["ic_ir"]) if pd.notna(row["ic_ir"]) else np.nan
        ts[en] = float(row["t_stat"]) if pd.notna(row["t_stat"]) else np.nan
        ns[en] = int(row["n_periods"]) if pd.notna(row["n_periods"]) else 0
    return means, irs, ts, ns


def _eval_horizon(panel: pd.DataFrame, horizon: int, sample: str):
    part = proto.train_slice(panel) if sample == "train" else proto.test_slice(panel)
    prepared = factor_ic.prepare_sample(part, horizon)
    summary, _ = factor_ic.ic_from_prepared(prepared)
    means, irs, ts, ns = _ic_maps(summary)
    corr = _corr_matrix(prepared) if prepared is not None and not prepared.empty else {}
    raw_w = proto.weights_from_ic(means, irs, ts)
    corr_mt = corr.get(("momentum", "trend"), np.nan)
    weights = proto.apply_collinearity(raw_w, ts, corr_mt)
    book = qbt.run_quantile_backtest(weights=weights, fwd_days=horizon, prepared=prepared)
    dates = []
    if prepared is not None and not prepared.empty:
        dates = sorted(prepared["date"].astype(str).unique())
    return {
        "means": means, "irs": irs, "ts": ts, "ns": ns,
        "corr": corr, "raw_w": raw_w, "weights": weights,
        "book": book, "dates": dates, "summary": summary,
    }


def _fmt_factors(block: dict) -> list[str]:
    lines = []
    for f in proto.TECH_FACTORS:
        cn = pnl._FACTOR_CN[f]
        lines.append(
            f"  {cn}: RankIC={_num(block['means'].get(f))}  "
            f"IR={_num(block['irs'].get(f), 3)}  "
            f"NW-t={_num(block['ts'].get(f), 2)}  "
            f"期数={block['ns'].get(f, 0)}  "
            f"权重={_num(block['weights'].get(f), 4)}"
        )
    return lines


def _fmt_book(book: dict) -> list[str]:
    return [
        f"  期数={book.get('periods', 0)}",
        f"  扣费后强档相对等权超额均值={_pct(book.get('excess_mean'))}",
        f"  超额 Newey-West t={_num(book.get('excess_t'), 2)}  滞后={book.get('excess_lags')}",
        f"  未扣费超额均值={_pct(book.get('gross_excess_mean'))}",
        f"  强档平均单边换手={_num(book.get('turnover_mean'), 3)}",
        f"  涨停未买入={book.get('n_limit_up', 0)}  跌停未卖出={book.get('n_limit_down', 0)}",
        f"  停牌未成交={book.get('n_suspend', 0)}  窗口内复权断裂剔除={book.get('n_break', 0)}",
        f"  合成因子 RankIC={_num(book.get('composite_ic_mean'))}  "
        f"NW-t={_num(book.get('composite_ic_t'), 2)}  滞后={book.get('composite_ic_lags')}",
        f"  费用恒等式={'通过' if book.get('cost_identity_ok') else '失败'}",
    ]


def run() -> str:
    lines = []
    lines.append("A 股技术因子有效性检验")
    lines.append("规则在看测试窗之前写死，测试窗只进入一次。")
    lines.append("")
    lines.append("冻结常数")
    lines.append(f"  训练截止日（卖出日不晚于此日）: {proto.TRAIN_END}")
    lines.append(f"  持有期候选（只在训练窗比较）: {proto.HORIZONS}")
    lines.append(f"  动量与趋势相关阈值: {proto.CORR_MAX}")
    lines.append(f"  权重门槛 |Newey-West t|: {proto.T_MIN}")
    lines.append(f"  佣金单边: {proto.COMMISSION}")
    lines.append(f"  印花税: {proto.STAMP_DATE} 之前 {proto.STAMP_BEFORE}，之后 {proto.STAMP_AFTER}，只在卖出收取")
    lines.append(f"  涨跌停容差: {proto.LIMIT_EPS}")
    lines.append("  原假设: 测试窗合成因子 RankIC 均值 = 0，且扣费后强档相对等权超额 <= 0")
    lines.append("  拒绝原假设的条件: 合成 RankIC 均值 > 0 且 NW t > 2，并且超额均值 > 0 且 NW t > 2")
    lines.append("  不满足则结论为: 没有可交易的样本外边际")
    lines.append("")
    lines.append("机械核对（不使用行情库）")
    try:
        for row in proto.self_check():
            lines.append("  " + row)
        for row in pnl.self_check_signals():
            lines.append("  " + row)
    except Exception as e:
        lines.append("  FAIL " + str(e))
        text = "\n".join(lines) + "\n"
        _write(text)
        raise

    print("核对通过，开始读行情...", flush=True)
    lines.append("")
    lines.append("数据清点")
    # 先建最短持有期的面板，顺带完成全库扫描。
    panel_cache = {}
    for h in proto.HORIZONS:
        print(f"构建持有期 {h} 的可成交收益...", flush=True)
        panel_cache[h] = pnl.build_panel(fwd_days=h, progress_cb=_progress(h))
    _, audit = pnl.load_market()
    lines.append(f"  股票数（已去掉 ETF，只留 0/3/6 开头）: {audit['n_codes']}")
    lines.append(f"  日线行数: {audit['n_rows']}")
    lines.append(f"  日期范围: {audit['date_min']} ~ {audit['date_max']}")
    lines.append(f"  复权断裂候选行数（|相邻收盘涨跌幅| > {proto.BREAK_ABS}）: {audit['n_break_rows']}")
    lines.append(f"  股票列表行数: {audit['n_stock_list']}")
    lines.append(f"  名称中含「退」的行数: {audit['n_name_delist']}")
    lines.append(f"  最后一根 K 线落在全库最后交易日的股票数: {audit['n_end_on_last']} / {audit['n_codes']}")
    lines.append(f"  起始日晚于 2022-01-01 的股票数: {audit['n_start_after_2022']}")
    if audit["n_codes"] and audit["n_end_on_last"] >= audit["n_codes"] - 2 and audit["n_name_delist"] == 0:
        lines.append("  股票池判断: 是现存代码。缓存里几乎每只股票都活到最后一天，列表里没有退市名称。")
        lines.append("  这不是含退市股的历史全市场。有 K 线的只是列表中被下载进来的那一部分。")
    else:
        lines.append("  股票池判断: 存在提前结束的代码，见上面的计数。名称里仍没有退市标记。")
    lines.append("  行业用的是股票列表里的当前行业，不是历史时点行业。")
    lines.append("  没有逐日 ST 标记，所以没有使用 5% 的 ST 涨跌停。")

    # 隔离带：用持有期 5 的面板数一下，三个持有期各自不同，后面分项再写。
    lines.append("")
    lines.append("训练窗估计（不是结论）")
    train_blocks = {}
    eligible = []
    for h in proto.HORIZONS:
        print(f"训练窗估计，持有期 {h}...", flush=True)
        panel = panel_cache[h]
        embargo = panel.loc[
            (panel["date"].astype(str) <= proto.TRAIN_END)
            & (panel["exit_date"].astype(str) > proto.TRAIN_END)
        ]
        block = _eval_horizon(panel, h, "train")
        train_blocks[h] = block
        lines.append(f"持有期 {h} 个交易日")
        lines.append(f"  调仓日: {block['dates'][0] if block['dates'] else '无'} ~ "
                     f"{block['dates'][-1] if block['dates'] else '无'}  共 {len(block['dates'])} 个")
        lines.append(f"  隔离带信号日数（卖出日越过 {proto.TRAIN_END}）: {embargo['date'].nunique() if len(embargo) else 0}")
        c = block["corr"]
        lines.append("  训练窗截面相关（调仓日截面皮尔逊相关，再对日期取平均）")
        for a in proto.TECH_FACTORS:
            row = []
            for b in proto.TECH_FACTORS:
                key = (a, b) if a <= b else (b, a)
                row.append(f"{pnl._FACTOR_CN[b]}={_num(c.get(key), 3)}")
            lines.append(f"    {pnl._FACTOR_CN[a]}: " + "  ".join(row))
        lines.append(f"  动量与趋势相关={_num(c.get(('momentum', 'trend')), 3)}  阈值={proto.CORR_MAX}")
        dropped = [f for f in proto.TECH_FACTORS
                   if block["raw_w"].get(f, 0) != 0 and block["weights"].get(f, 0) == 0]
        # 相关筛选会把未入选的也置 0。单独标出「本来能进权重但被相关规则丢掉」的。
        dropped = [f for f in ("momentum", "trend")
                   if block["raw_w"].get(f, 0.0) != 0.0 and block["weights"].get(f, 0.0) == 0.0]
        if abs(c.get(("momentum", "trend"), 0) or 0) >= proto.CORR_MAX:
            lines.append("  相关规则触发，丢掉: " + ("、".join(pnl._FACTOR_CN[f] for f in dropped) or "无"))
        else:
            lines.append("  相关规则未触发")
        lines.extend(_fmt_factors(block))
        lines.extend(_fmt_book(block["book"]))
        alive = any(float(block["weights"].get(f, 0.0)) != 0.0 for f in proto.TECH_FACTORS)
        t_ex = block["book"].get("excess_t")
        if alive and t_ex == t_ex:
            eligible.append(h)
            lines.append("  本持有期进入候选")
        else:
            lines.append("  本持有期没有可交易权重，或超额 t 不可用，不进入候选")
        lines.append("")

    chosen = None
    if eligible:
        chosen = max(eligible, key=lambda h: (train_blocks[h]["book"]["excess_t"], -h))
        lines.append(f"训练窗选中的持有期: {chosen}。并列时取较短的持有期。其他持有期的测试窗不计算。")
    else:
        lines.append("训练窗没有任何持有期形成可交易权重。按规则不进入测试窗，也不改门槛再搜。")

    lines.append("")
    lines.append("测试窗")
    reject = False
    if chosen is None:
        lines.append("  未计算。训练窗没有通过规则的权重。")
        lines.append("没有可交易的样本外边际")
    else:
        print(f"测试窗只跑持有期 {chosen}...", flush=True)
        panel = panel_cache[chosen]
        frozen = dict(train_blocks[chosen]["weights"])
        prepared = factor_ic.prepare_sample(proto.test_slice(panel), chosen)
        summary, _ = factor_ic.ic_from_prepared(prepared)
        means, irs, ts, ns = _ic_maps(summary)
        book = qbt.run_quantile_backtest(
            weights=frozen, fwd_days=chosen, prepared=prepared)
        dates = sorted(prepared["date"].astype(str).unique()) if len(prepared) else []
        lines.append(f"  持有期: {chosen}")
        lines.append(f"  信号日范围: {dates[0] if dates else '无'} ~ {dates[-1] if dates else '无'}")
        lines.append(f"  调仓日数: {len(dates)}")
        lines.append("  冻结权重（只来自训练窗）: " + ", ".join(
            f"{pnl._FACTOR_CN[f]}={_num(frozen.get(f), 4)}" for f in proto.TECH_FACTORS))
        lines.append("  测试窗各因子 RankIC。这些数不改权重。")
        for f in proto.TECH_FACTORS:
            lines.append(
                f"  {pnl._FACTOR_CN[f]}: RankIC={_num(means.get(f))}  "
                f"NW-t={_num(ts.get(f), 2)}  期数={ns.get(f, 0)}"
            )
        lines.append("  组合（上面的冻结权重）")
        lines.extend(_fmt_book(book))
        ic_ok = (np.isfinite(book.get("composite_ic_mean"))
                 and np.isfinite(book.get("composite_ic_t"))
                 and book["composite_ic_mean"] > 0
                 and book["composite_ic_t"] > proto.T_MIN)
        ex_ok = (np.isfinite(book.get("excess_mean"))
                 and np.isfinite(book.get("excess_t"))
                 and book["excess_mean"] > 0
                 and book["excess_t"] > proto.T_MIN)
        reject = bool(ic_ok and ex_ok and book.get("cost_identity_ok"))
        if not book.get("cost_identity_ok"):
            lines.append("  费用恒等式没有通过，不拒绝原假设。")
        lines.append("")
        if reject:
            lines.append("结论: 测试窗拒绝原假设。合成因子 RankIC 与扣费后超额都为正，且 Newey-West t 大于 2。")
            lines.append("按规则到此停止，不再换参数。")
        else:
            lines.append("没有可交易的样本外边际")

    lines.append("")
    lines.append("重跑: venv\\Scripts\\python.exe -m app.strategy.validity")
    text = "\n".join(lines) + "\n"
    path = _write(text)
    print(text)
    print("记录已写入", path)
    return text


def _progress(horizon):
    def cb(done, total):
        if done == 1 or done == total or done % 200 == 0:
            print(f"  持有期 {horizon} 读取 {done}/{total}", flush=True)
    return cb


def _write(text: str) -> str:
    root = os.path.dirname(os.path.dirname(db.DB_PATH))
    path = os.path.join(root, "validity_record.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


if __name__ == "__main__":
    run()
