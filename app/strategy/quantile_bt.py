"""
分档与强档组合。

和 factor_ic 共用 prepare_sample：同一批调仓日、同一套处理后的因子、
同一列开盘到开盘毛收益 fwd_ret。
费用按组合换手扣（连续入选不重复收一轮佣金）。强档相对的是当日全部可成交股票的等权组合。
"""
import numpy as np
import pandas as pd

from . import factor_ic
from . import protocol as proto


DEFAULT_TECH_WEIGHTS = {"momentum": 40.0, "trend": 40.0, "volume": 20.0}


def _empty(fwd_days, n_groups):
    return {
        "group_curves": {}, "metrics": {}, "periods": 0,
        "period_days": fwd_days, "monotonic": False, "n_groups": n_groups,
        "excess_mean": np.nan, "excess_t": np.nan, "excess_lags": 0,
        "gross_excess_mean": np.nan, "turnover_mean": np.nan,
        "n_limit_up": 0, "n_limit_down": 0, "n_suspend": 0, "n_break": 0,
        "cost_identity_ok": True, "composite_ic_mean": np.nan,
        "composite_ic_t": np.nan, "composite_ic_lags": 0,
        "dates": [],
    }


def _book(members: pd.DataFrame):
    """已成交成员等权。人不够就空仓。"""
    filled = members.loc[(members["block"] == "ok") & members["fwd_ret"].notna()]
    if len(filled) < proto.MIN_BOOK:
        return {}, {}
    w = 1.0 / len(filled)
    weights = {str(c): w for c in filled["code"]}
    gross = {str(c): float(r) for c, r in zip(filled["code"], filled["fwd_ret"])}
    return weights, gross


def _mode(s: pd.Series) -> str:
    if s is None or len(s) == 0:
        return ""
    m = s.astype(str).mode()
    return str(m.iloc[0]) if len(m) else str(s.iloc[0])


def _settle(df: pd.DataFrame, dates, n_groups: int, fwd_days: int) -> dict:
    out = _empty(fwd_days, n_groups)
    packed = []
    n_up = n_dn = n_sus = n_brk = 0
    for d in dates:
        g = df.loc[df["date"].astype(str) == d]
        scored = g.loc[g["_score"].notna()]
        if len(scored) < max(proto.MIN_NAMES, n_groups * proto.MIN_BOOK):
            continue
        try:
            q = pd.qcut(scored["_score"].rank(method="first"), n_groups, labels=False)
        except ValueError:
            continue
        scored = scored.assign(_q=np.asarray(q))
        top = scored.loc[scored["_q"] == n_groups - 1]
        n_up += int((top["block"] == "limit_up").sum())
        n_dn += int((top["block"] == "limit_down").sum())
        n_sus += int((top["block"] == "suspend").sum())
        n_brk += int((top["block"] == "break").sum())

        books, grosses, good = [], [], True
        for qi in range(n_groups):
            w, gr = _book(scored.loc[scored["_q"] == qi])
            books.append(w)
            grosses.append(gr)
            if not w:
                good = False
        ew_w, ew_g = _book(scored)
        if (not good) or (not ew_w):
            continue

        ic = np.nan
        filled = scored.loc[(scored["block"] == "ok") & scored["fwd_ret"].notna(),
                            ["_score", "fwd_ret"]]
        if len(filled) >= proto.MIN_NAMES:
            ra = filled["_score"].rank()
            rb = filled["fwd_ret"].rank()
            if float(ra.std(ddof=0)) > 0 and float(rb.std(ddof=0)) > 0:
                ic = float(np.corrcoef(ra.to_numpy(), rb.to_numpy())[0, 1])
        packed.append({
            "date": d,
            "entry": _mode(scored["entry_date"]),
            "exit": _mode(scored["exit_date"]),
            "books": books,
            "grosses": grosses,
            "ew_w": ew_w,
            "ew_g": ew_g,
            "ic": ic,
        })

    prev = {q: {} for q in range(n_groups)}
    prev_ew = {}
    group_rets = {q: [] for q in range(n_groups)}
    excess_net, excess_gross, turns, ics, used = [], [], [], [], []
    ident_err = 0.0
    for i, p in enumerate(packed):
        is_last = i == len(packed) - 1

        def settle(prev_w, w, gr, _p=p, _last=is_last):
            cost = proto.rebalance_cost(prev_w, w, _p["entry"])
            if _last:
                cost += sum(w.values()) * (proto.COMMISSION + proto.stamp_rate(_p["exit"]))
            gross = sum(w[c] * gr[c] for c in w)
            return gross - cost, gross, cost, proto.one_way_turnover(prev_w, w)

        for qi in range(n_groups):
            net, _, _, _ = settle(prev[qi], p["books"][qi], p["grosses"][qi])
            group_rets[qi].append(net)
        net_q, gross_q, cost_q, turn_q = settle(prev[n_groups - 1], p["books"][-1], p["grosses"][-1])
        net_e, gross_e, cost_e, _ = settle(prev_ew, p["ew_w"], p["ew_g"])
        for qi in range(n_groups):
            prev[qi] = p["books"][qi]
        prev_ew = p["ew_w"]
        ex_n = net_q - net_e
        ex_g = gross_q - gross_e
        ident_err = max(ident_err, abs((ex_n - ex_g) - (cost_e - cost_q)))
        excess_net.append(ex_n)
        excess_gross.append(ex_g)
        turns.append(turn_q)
        used.append(p["date"])
        if np.isfinite(p["ic"]):
            ics.append(p["ic"])

    excess_net = np.asarray(excess_net, dtype=float)
    excess_gross = np.asarray(excess_gross, dtype=float)
    mu, t, lags = proto.newey_west_mean(excess_net) if len(excess_net) else (np.nan, np.nan, 0)
    ic_mu, ic_t, ic_lags = proto.newey_west_mean(ics) if ics else (np.nan, np.nan, 0)

    ls_name = "多空(Q%d-Q1)" % n_groups
    ret_map = {f"Q{q + 1}": np.asarray(group_rets[q], dtype=float) for q in range(n_groups)}
    if n_groups >= 2 and all(len(group_rets[q]) == len(used) for q in range(n_groups)):
        ret_map[ls_name] = ret_map[f"Q{n_groups}"] - ret_map["Q1"]

    curves, metrics = {}, {}
    periods_per_year = 244.0 / float(fwd_days)
    for name, r in ret_map.items():
        if len(r) == 0:
            continue
        nav = np.cumprod(1.0 + np.nan_to_num(r, nan=0.0))
        curves[name] = pd.Series(nav, index=used[:len(nav)])
        std = float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
        mean_p = float(np.mean(r))
        ann = (1.0 + mean_p) ** periods_per_year - 1.0 if mean_p > -1 else np.nan
        sharpe = (mean_p / std * np.sqrt(periods_per_year)
                  if std and np.isfinite(std) and std != 0 else np.nan)
        peak = np.maximum.accumulate(nav)
        dd = float(np.min(nav / peak - 1.0))
        metrics[name] = {
            "ann_return": float(ann) if np.isfinite(ann) else np.nan,
            "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
            "max_drawdown": dd,
            "total_return": float(nav[-1] - 1.0),
        }

    q_anns = [metrics.get(f"Q{q + 1}", {}).get("ann_return", np.nan) for q in range(n_groups)]
    monotonic = all(
        q_anns[i] <= q_anns[i + 1]
        for i in range(len(q_anns) - 1)
        if np.isfinite(q_anns[i]) and np.isfinite(q_anns[i + 1])
    )
    out.update({
        "group_curves": curves,
        "metrics": metrics,
        "periods": len(used),
        "period_days": fwd_days,
        "n_groups": n_groups,
        "monotonic": bool(monotonic),
        "ls_name": ls_name,
        "dates": used,
        "excess_mean": mu,
        "excess_t": t,
        "excess_lags": int(lags),
        "gross_excess_mean": float(np.mean(excess_gross)) if len(excess_gross) else np.nan,
        "turnover_mean": float(np.mean(turns)) if turns else np.nan,
        "n_limit_up": int(n_up),
        "n_limit_down": int(n_dn),
        "n_suspend": int(n_sus),
        "n_break": int(n_brk),
        "cost_identity_ok": bool(ident_err < 1e-8),
        "composite_ic_mean": ic_mu,
        "composite_ic_t": ic_t,
        "composite_ic_lags": int(ic_lags),
    })
    return out


def run_quantile_backtest(weights: dict = None, fwd_days: int = 5,
                          n_groups: int = 5, step: int = None,
                          panel: pd.DataFrame = None, progress_cb=None,
                          prepared: pd.DataFrame = None) -> dict:
    """prepared 给定时不再重新抽样，保证和同一次 IC 用同一张表。"""
    if step is None:
        step = fwd_days
    if weights is None:
        w = dict(DEFAULT_TECH_WEIGHTS)
    else:
        w = {k: float(weights.get(k, 0.0)) for k in proto.TECH_FACTORS}
    if not any(v != 0.0 for v in w.values()):
        empty = _empty(fwd_days, n_groups)
        empty["reason"] = "no_weights"
        return empty
    if prepared is None:
        if panel is None:
            from . import panel as pnl
            panel = pnl.build_panel(fwd_days=fwd_days, progress_cb=progress_cb)
        if panel is None or panel.empty:
            return _empty(fwd_days, n_groups)
        prepared = factor_ic.prepare_sample(panel, step)
    if prepared is None or prepared.empty:
        return _empty(fwd_days, n_groups)
    df = prepared.copy()
    df["_score"] = proto.composite_score(df, w)
    dates = sorted(df["date"].astype(str).unique())
    return _settle(df, dates, n_groups, fwd_days)
