"""
上涨概率。回答的是：一只股票按规则成交之后，收益为正的概率。

持有期用有效性检验已经选定的 20 日，这里不再比较 5 日和 10 日。
权重只在训练窗按原规则重算。测试窗不改权重，也不改档位。

标签：能成交且 fwd_ret > 0 记为上涨。收益刚好为 0 记为不上涨。
买不进、卖不出的交易没有收益，不记成跌，也不进入概率。

条件模型：每个调仓日按合成得分的截面秩分 5 档。
某一档的上涨概率 = 训练窗里该档各调仓日上涨频率的平均。
基准 = 训练窗里各调仓日全体可成交股票上涨频率的平均。

启用条件模型（三条同时，看测试窗之前写死）：
  1. 训练窗第 5 档的上涨频率不低于第 1 档，且训练、测试的有效调仓日都不少于 10。
  2. 测试窗模型的日期 Brier 低于基准。
  3. （基准 Brier − 模型 Brier）的 Newey-West t > 2。
否则每只股票都用基准概率，选股名单为空。
结论句固定，不随结果改写规则。

界面上的概率用训练窗基准（或启用时的训练窗分档）。
测试窗频率只作对照，不回写进概率。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ..data import database as db
from . import factor_ic
from . import panel as pnl
from . import protocol as proto
from . import validity


# 有效性检验在训练窗选定的持有期。此脚本不重新挑选。
HORIZON = 20
N_BINS = 5
MIN_BIN_DATES = 10

NOT_DISTINGUISHABLE = "个股之间没有可区分的上涨概率"
DISTINGUISHABLE = "上涨概率按训练窗分档使用，名单只含得分最高的一档"


def name_net(gross: float, exit_date) -> float:
    """单名持有一整段的净收益：买入佣金、卖出佣金、卖出印花税。"""
    return (1.0 + float(gross)) * (1.0 - proto.COMMISSION - proto.stamp_rate(exit_date)) / (1.0 + proto.COMMISSION) - 1.0


def decide(p_bins, n_train_days: int, n_test_days: int, brier_model: float, brier_base: float, diff_t: float) -> bool:
    """启用条件概率的唯一入口。三条都要满足。"""
    bins = list(p_bins)
    if len(bins) != N_BINS or any(not np.isfinite(p) for p in bins):
        return False
    if int(n_train_days) < MIN_BIN_DATES or int(n_test_days) < MIN_BIN_DATES:
        return False
    if float(bins[-1]) < float(bins[0]):
        return False
    if not (np.isfinite(brier_model) and np.isfinite(brier_base) and brier_model < brier_base):
        return False
    if not (np.isfinite(diff_t) and float(diff_t) > proto.T_MIN):
        return False
    return True


def self_check() -> list[str]:
    lines = []
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        lines.append(("PASS " if cond else "FAIL ") + name)

    hand = proto.net_of_costs(10.0, 11.0, "2024-01-02")
    check("单名净收益与整段公式一致", abs(name_net(0.10, "2024-01-02") - hand) < 1e-12)
    check("方向对、Brier 更低且 t>2 时启用",
          decide([0.40, 0.45, 0.50, 0.55, 0.60], 30, 30, 0.20, 0.25, 2.5))
    check("高分档频率更低时不启用",
          not decide([0.60, 0.55, 0.50, 0.45, 0.40], 30, 30, 0.20, 0.25, 3.0))
    check("Brier 没有更低时不启用",
          not decide([0.40, 0.45, 0.50, 0.55, 0.60], 30, 30, 0.30, 0.25, 3.0))
    check("t 不超过 2 时不启用",
          not decide([0.40, 0.45, 0.50, 0.55, 0.60], 30, 30, 0.20, 0.25, 1.9))
    check("调仓日不足时不启用",
          not decide([0.40, 0.45, 0.50, 0.55, 0.60], 9, 30, 0.20, 0.25, 3.0))

    # 高分的六只全部上涨，低分全部下跌。第 5 档频率应为 1，第 1 档为 0。
    rows = []
    for d in ("2021-06-01", "2021-07-01", "2021-08-02"):
        for i in range(30):
            rows.append({
                "date": d, "code": f"{i:06d}", "industry": "银行",
                "_score": float(i), "block": "ok",
                "fwd_ret": 0.02 if i >= 24 else -0.01,
                "exit_date": "2021-06-20",
            })
    toy = _annotate_days(pd.DataFrame(rows))
    stats = _day_stats(toy)
    rates = _rates(stats)
    check("构造样本高分档上涨频率为 1", abs(rates["p"][4] - 1.0) < 1e-12)
    check("构造样本低分档上涨频率为 0", abs(rates["p"][0] - 0.0) < 1e-12)
    if not ok:
        raise RuntimeError("概率核对失败:\n" + "\n".join(lines))
    return lines


def _annotate_days(prepared: pd.DataFrame) -> pd.DataFrame:
    """只留下 5 档都能成交够数的调仓日，并写上 _q。第 5 档是得分最高的一档。"""
    if prepared is None or prepared.empty or "_score" not in prepared.columns:
        return pd.DataFrame()
    frames = []
    need = max(proto.MIN_NAMES, N_BINS * proto.MIN_BOOK)
    for _, g in prepared.groupby("date", sort=False):
        scored = g.loc[g["_score"].notna()].copy()
        if len(scored) < need:
            continue
        try:
            q = pd.qcut(scored["_score"].rank(method="first"), N_BINS, labels=False)
        except ValueError:
            continue
        scored["_q"] = np.asarray(q, dtype=int)
        good = True
        for qi in range(N_BINS):
            filled = _filled(scored.loc[scored["_q"] == qi])
            if len(filled) < proto.MIN_BOOK:
                good = False
                break
        if len(_filled(scored)) < proto.MIN_NAMES:
            good = False
        if good:
            frames.append(scored)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _filled(g: pd.DataFrame) -> pd.DataFrame:
    return g.loc[(g["block"] == "ok") & g["fwd_ret"].notna()]


def _day_stats(frame: pd.DataFrame) -> list[dict]:
    rows = []
    if frame is None or frame.empty:
        return rows
    for d, g in frame.groupby("date", sort=False):
        filled = _filled(g)
        y = (filled["fwd_ret"] > 0).to_numpy(dtype=float)
        nets = [name_net(r, dt) for r, dt in zip(filled["fwd_ret"], filled["exit_date"])]
        bin_p, bin_net, bin_n = [], [], []
        for qi in range(N_BINS):
            part = _filled(g.loc[g["_q"] == qi])
            bin_n.append(int(len(part)))
            if len(part) == 0:
                bin_p.append(np.nan)
                bin_net.append(np.nan)
            else:
                bin_p.append(float((part["fwd_ret"] > 0).mean()))
                bin_net.append(float(np.mean([
                    name_net(r, dt) for r, dt in zip(part["fwd_ret"], part["exit_date"])
                ])))
        rows.append({
            "date": str(d),
            "p": float(y.mean()) if len(y) else np.nan,
            "gross": float(filled["fwd_ret"].mean()) if len(filled) else np.nan,
            "net": float(np.mean(nets)) if nets else np.nan,
            "bin_p": bin_p,
            "bin_net": bin_net,
            "bin_n": bin_n,
            "n_filled": int(len(filled)),
            "n_limit_up": int((g["block"] == "limit_up").sum()),
            "n_limit_down": int((g["block"] == "limit_down").sum()),
            "n_suspend": int((g["block"] == "suspend").sum()),
            "n_break": int((g["block"] == "break").sum()),
            "y": y,
            "q": filled["_q"].to_numpy(dtype=int) if len(filled) else np.array([], dtype=int),
        })
    rows.sort(key=lambda r: r["date"])
    return rows


def _rates(stats: list[dict]) -> dict:
    """各档上涨概率 = 调仓日频率的平均，不是把所有股票堆在一起。"""
    p = []
    net = []
    nd = []
    for qi in range(N_BINS):
        ps = [s["bin_p"][qi] for s in stats if np.isfinite(s["bin_p"][qi])]
        ns = [s["bin_net"][qi] for s in stats if np.isfinite(s["bin_net"][qi])]
        p.append(float(np.mean(ps)) if ps else np.nan)
        net.append(float(np.mean(ns)) if ns else np.nan)
        nd.append(int(len(ps)))
    return {"p": p, "net": net, "n_days": nd}


def _brier_series(stats: list[dict], p_bins, p0: float):
    model, base, diff = [], [], []
    for s in stats:
        if s["y"].size < proto.MIN_NAMES or not np.isfinite(p0):
            continue
        pq = np.array([p_bins[int(q)] for q in s["q"]], dtype=float)
        if not np.all(np.isfinite(pq)):
            continue
        bm = float(np.mean((s["y"] - pq) ** 2))
        bb = float(np.mean((s["y"] - p0) ** 2))
        model.append(bm)
        base.append(bb)
        diff.append(bb - bm)
    return model, base, diff


def _mean_t(xs):
    if not xs:
        return np.nan, np.nan, 0
    return proto.newey_west_mean(np.asarray(xs, dtype=float))


def _score_frame(panel: pd.DataFrame, weights: dict) -> pd.DataFrame:
    prepared = factor_ic.prepare_sample(panel, HORIZON)
    if prepared is None or prepared.empty:
        return pd.DataFrame()
    out = prepared.copy()
    out["_score"] = proto.composite_score(out, weights)
    return out


def _latest_snapshot(weights: dict) -> tuple[str, list[dict]]:
    prepared, _ = pnl.load_market()
    if not prepared:
        return "", []
    asof = max(str(item["date"][-1]) for item in prepared)
    rows = []
    for item in prepared:
        if str(item["date"][-1]) != asof:
            continue
        mom = float(item["momentum"][-1])
        trend = float(item["trend"][-1])
        volf = float(item["volume_f"][-1])
        if not (np.isfinite(mom) and np.isfinite(trend) and np.isfinite(volf)):
            continue
        rows.append({
            "date": asof,
            "code": str(item["code"]),
            "industry": item["industry"] or "未分类",
            "momentum": mom,
            "trend": trend,
            "volume": volf,
            "close": float(item["close"][-1]),
        })
    if not rows:
        return asof, []
    df = proto.attach_processed(pd.DataFrame(rows))
    df["score"] = proto.composite_score(df, weights)
    df["quintile"] = np.nan
    scored = df["score"].notna()
    if int(scored.sum()) >= max(proto.MIN_NAMES, N_BINS * proto.MIN_BOOK):
        q = pd.qcut(df.loc[scored, "score"].rank(method="first"), N_BINS, labels=False)
        df.loc[scored, "quintile"] = np.asarray(q, dtype=float) + 1.0
    names = {}
    slist = db.load_stock_list()
    if slist is not None and not slist.empty and "name" in slist.columns:
        slist = slist.copy()
        slist["code"] = slist["code"].astype(str).str.zfill(6)
        names = dict(zip(slist["code"], slist["name"].astype(str)))
    stocks = []
    for r in df.itertuples(index=False):
        qv = float(r.quintile) if np.isfinite(r.quintile) else None
        stocks.append({
            "code": str(r.code),
            "name": names.get(str(r.code), ""),
            "industry": str(r.industry),
            "close": round(float(r.close), 4),
            "momentum": None if not np.isfinite(r.momentum) else round(float(r.momentum), 4),
            "trend": None if not np.isfinite(r.trend) else round(float(r.trend), 6),
            "volume": None if not np.isfinite(r.volume) else round(float(r.volume), 4),
            "score": None if not np.isfinite(r.score) else round(float(r.score), 4),
            "quintile": None if qv is None else int(qv),
        })
    stocks.sort(key=lambda s: s["code"])
    return asof, stocks


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not np.isfinite(v) else v
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def _root() -> str:
    return os.path.dirname(os.path.dirname(db.DB_PATH))


def state_path() -> str:
    return os.path.join(_root(), "probability_state.json")


def record_path() -> str:
    return os.path.join(_root(), "probability_record.txt")


def load_state() -> dict | None:
    path = state_path()
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _num(x, nd=4):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{float(x):.{nd}f}"


def _pct(x):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{float(x) * 100:.2f}%"


def _block_counts(stats: list[dict]) -> dict:
    return {
        "limit_up": int(sum(s["n_limit_up"] for s in stats)),
        "limit_down": int(sum(s["n_limit_down"] for s in stats)),
        "suspend": int(sum(s["n_suspend"] for s in stats)),
        "break": int(sum(s["n_break"] for s in stats)),
    }


def run() -> dict:
    lines = []
    lines.append("上涨概率")
    lines.append("规则在看测试窗之前写死。测试窗只用来决定是否启用分档概率，不回写概率本身。")
    lines.append(f"持有期 {HORIZON} 来自已经完成的有效性检验，本脚本不比较其他持有期。")
    lines.append("")
    lines.append("冻结规则")
    lines.append(f"  训练截止日（卖出日不晚于此日）: {proto.TRAIN_END}")
    lines.append(f"  持有期: {HORIZON} 个交易日。信号日收盘之后，次日开盘买入，到期开盘卖出。")
    lines.append(f"  分档数: {N_BINS}。第 5 档是得分最高的一档。")
    lines.append(f"  启用所需的最少调仓日: {MIN_BIN_DATES}")
    lines.append(f"  Newey-West t 门槛: {proto.T_MIN}")
    lines.append("  上涨: 能成交且毛收益 > 0。毛收益为 0 不算上涨。")
    lines.append("  未成交（涨停开盘、跌停开盘、停牌、复权断裂）不进入概率，也不记成下跌。")
    lines.append("  基准概率: 训练窗调仓日上涨频率的平均。")
    lines.append("  条件概率: 训练窗各档调仓日上涨频率的平均。")
    lines.append("  启用条件模型: 第5档频率 >= 第1档，测试窗 Brier 更低，且 Brier 差的 NW t > 2。")
    lines.append(f"  不启用时的结论句: {NOT_DISTINGUISHABLE}")
    lines.append(f"  启用时的结论句: {DISTINGUISHABLE}")
    lines.append("  收益检验的结论仍以 validity_record.txt 为准。本脚本不改那句话。")
    lines.append("")
    lines.append("机械核对（不使用行情库）")
    try:
        for row in proto.self_check():
            lines.append("  " + row)
        for row in pnl.self_check_signals():
            lines.append("  " + row)
        for row in self_check():
            lines.append("  " + row)
    except Exception as e:
        lines.append("  FAIL " + str(e))
        text = "\n".join(lines) + "\n"
        _write_text(text)
        raise

    print("核对通过，构建 20 日可成交收益...", flush=True)
    panel = pnl.build_panel(fwd_days=HORIZON)
    _, audit = pnl.load_market()
    lines.append("")
    lines.append("数据")
    lines.append(f"  股票数: {audit['n_codes']}  日期: {audit['date_min']} ~ {audit['date_max']}")
    lines.append("  股票池与有效性检验相同：现存代码，不是含退市股的历史全市场。")

    print("训练窗权重...", flush=True)
    train_block = validity._eval_horizon(panel, HORIZON, "train")
    weights = {f: float(train_block["weights"].get(f, 0.0)) for f in proto.TECH_FACTORS}
    lines.append("")
    lines.append("训练窗权重（与有效性检验同一套规则，不是在测试窗上重估的）")
    for f in proto.TECH_FACTORS:
        lines.append(f"  {pnl._FACTOR_CN[f]}={_num(weights.get(f), 4)}")

    train_scored = _annotate_days(_score_frame(proto.train_slice(panel), weights))
    test_scored = _annotate_days(_score_frame(proto.test_slice(panel), weights))
    train_stats = _day_stats(train_scored)
    test_stats = _day_stats(test_scored)
    train_rates = _rates(train_stats)
    test_rates = _rates(test_stats)
    p0, p0_t, p0_lags = _mean_t([s["p"] for s in train_stats])
    p_test, p_test_t, p_test_lags = _mean_t([s["p"] for s in test_stats])
    g0, g0_t, _ = _mean_t([s["gross"] for s in train_stats])
    n0, n0_t, n0_lags = _mean_t([s["net"] for s in train_stats])
    g1, _, _ = _mean_t([s["gross"] for s in test_stats])
    n1, n1_t, n1_lags = _mean_t([s["net"] for s in test_stats])

    model_b, base_b, diff_b = _brier_series(test_stats, train_rates["p"], p0)
    brier_model, _, _ = _mean_t(model_b)
    brier_base, _, _ = _mean_t(base_b)
    diff_mean, diff_t, diff_lags = _mean_t(diff_b)
    adopted = decide(
        train_rates["p"], len(train_stats), len(test_stats),
        brier_model, brier_base, diff_t,
    )
    # 权重全 0 时没有得分，分档没有意义。
    if not any(weights.values()):
        adopted = False
    conclusion = DISTINGUISHABLE if adopted else NOT_DISTINGUISHABLE

    def _fmt_bins(title, rates, stats):
        out = [title]
        if not stats:
            out.append("  无有效调仓日")
            return out
        out.append(f"  调仓日: {stats[0]['date']} ~ {stats[-1]['date']}  共 {len(stats)} 个")
        counts = _block_counts(stats)
        out.append(
            f"  涨停未买入={counts['limit_up']}  跌停未卖出={counts['limit_down']}  "
            f"停牌={counts['suspend']}  复权断裂={counts['break']}"
        )
        for i in range(N_BINS):
            out.append(
                f"  第{i + 1}档: 上涨频率={_pct(rates['p'][i])}  "
                f"平均扣费收益={_pct(rates['net'][i])}  调仓日={rates['n_days'][i]}"
            )
        return out

    lines.append("")
    lines.extend(_fmt_bins("训练窗分档（概率用这些数）", train_rates, train_stats))
    se = (p0 / p0_t) if (np.isfinite(p0) and np.isfinite(p0_t) and p0_t != 0) else np.nan
    t_half = (p0 - 0.5) / se if (np.isfinite(se) and se > 0) else np.nan
    lines.append(
        f"  基准上涨概率={_pct(p0)}  标准误={_pct(se) if np.isfinite(se) else 'nan'}  "
        f"相对 50% 的 t={_num(t_half, 2)}  滞后={p0_lags}"
    )
    lines.append("  上面的 t 检验的是概率是否等于 50%，不是概率是否等于 0。")
    lines.append(f"  基准平均毛收益={_pct(g0)}  NW-t={_num(g0_t, 2)}")
    lines.append(f"  基准平均扣费收益={_pct(n0)}  NW-t={_num(n0_t, 2)}  滞后={n0_lags}")
    lines.append("  扣费按单名往返：买入佣金、卖出佣金、卖出当日印花税。")
    lines.append("")
    lines.extend(_fmt_bins("测试窗分档（对照，不回写概率，也不用来挑档）", test_rates, test_stats))
    se_test = (p_test / p_test_t) if (np.isfinite(p_test) and np.isfinite(p_test_t) and p_test_t != 0) else np.nan
    t_half_test = (p_test - 0.5) / se_test if (np.isfinite(se_test) and se_test > 0) else np.nan
    lines.append(
        f"  事后上涨频率={_pct(p_test)}  标准误={_pct(se_test) if np.isfinite(se_test) else 'nan'}  "
        f"相对 50% 的 t={_num(t_half_test, 2)}  滞后={p_test_lags}"
    )
    lines.append(f"  事后平均毛收益={_pct(g1)}")
    lines.append(f"  事后平均扣费收益={_pct(n1)}  NW-t={_num(n1_t, 2)}  滞后={n1_lags}")
    lines.append("")
    lines.append("Brier（越低越好）。日期内先对可成交股票平均，再对日期做 Newey-West。")
    lines.append(f"  模型 Brier={_num(brier_model)}  基准 Brier={_num(brier_base)}")
    lines.append(f"  基准减模型={_num(diff_mean)}  NW-t={_num(diff_t, 2)}  滞后={diff_lags}")
    lines.append(f"  第5档频率不低于第1档: {'是' if np.isfinite(train_rates['p'][4]) and train_rates['p'][4] >= train_rates['p'][0] else '否'}")
    lines.append(f"  测试窗 Brier 更低: {'是' if np.isfinite(brier_model) and np.isfinite(brier_base) and brier_model < brier_base else '否'}")
    lines.append(f"  Brier 差的 t > {proto.T_MIN}: {'是' if np.isfinite(diff_t) and diff_t > proto.T_MIN else '否'}")
    lines.append(f"  启用条件概率: {'是' if adopted else '否'}")
    lines.append("")
    lines.append(conclusion)

    print("最近一个收盘日的截面...", flush=True)
    asof, stocks = _latest_snapshot(weights)
    lines.append("")
    lines.append(f"最近收盘日: {asof}  可计算分数的股票: {len(stocks)}")
    lines.append("最近收盘日没有未来收益，不参与上面的估计，只供界面查个股。")
    lines.append("")
    lines.append("重跑: venv\\Scripts\\python.exe -m app.strategy.probability")

    state = {
        "horizon": HORIZON,
        "train_end": proto.TRAIN_END,
        "n_bins": N_BINS,
        "adopted": adopted,
        "conclusion": conclusion,
        "weights": weights,
        "p_up": p0,
        "p_up_t": p0_t,
        "p_up_se": se,
        "p_up_lags": p0_lags,
        "train_net": n0,
        "train_net_t": n0_t,
        "train_net_lags": n0_lags,
        "train_gross": g0,
        "test_p_up": p_test,
        "test_net": n1,
        "test_net_t": n1_t,
        "brier_model": brier_model,
        "brier_base": brier_base,
        "brier_diff_t": diff_t,
        "brier_diff_lags": diff_lags,
        "n_train_days": len(train_stats),
        "n_test_days": len(test_stats),
        "train_range": [train_stats[0]["date"], train_stats[-1]["date"]] if train_stats else [],
        "test_range": [test_stats[0]["date"], test_stats[-1]["date"]] if test_stats else [],
        "train_bins": [
            {"quintile": i + 1, "p": train_rates["p"][i], "net": train_rates["net"][i], "n_days": train_rates["n_days"][i]}
            for i in range(N_BINS)
        ],
        "test_bins": [
            {"quintile": i + 1, "p": test_rates["p"][i], "net": test_rates["net"][i], "n_days": test_rates["n_days"][i]}
            for i in range(N_BINS)
        ],
        "train_blocks": _block_counts(train_stats),
        "test_blocks": _block_counts(test_stats),
        "asof": asof,
        "stocks": stocks,
        "validity_conclusion": "没有可交易的样本外边际",
    }
    text = "\n".join(lines) + "\n"
    _write_text(text)
    with open(state_path(), "w", encoding="utf-8") as f:
        json.dump(_jsonable(state), f, ensure_ascii=False, indent=2)
    print(text)
    print("记录已写入", record_path())
    return state


def _write_text(text: str) -> None:
    with open(record_path(), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    run()
