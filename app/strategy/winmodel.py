"""
每只股票一个上涨概率。

上涨 = 按规则能成交，且开盘到开盘的毛收益 > 0。
概率来自逻辑回归。特征、惩罚和持有期候选都写在下面，跑完之前不改。

持有期和模型只在训练窗里用滚动预测比较：
  预测某一天时，只用卖出日早于这一天的成交样本。
  比较的是高概率两成的上涨比例，减去当天全体可成交股票的上涨比例。
  取 Newey-West t 最大的一组。t 相同则取较短持有期，再取参数更少的反转模型。

测试窗只进入一次，用训练窗全样本重新拟合的那一组。
启用策略的条件（三条同时）：
  高概率两成的赢率提升 NW t > 2，
  扣费后相对等权的超额均值 > 0，
  超额的 NW t > 2。
不满足则结论为下面这句，概率仍按模型逐只给出，但不作为已经提高赢率的策略。

界面上的概率用这组训练窗系数，乘最近一个收盘日的截面。
测试窗不参与拟合，也不改系数。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ..data import database as db
from . import panel as pnl
from . import protocol as proto


FEATURES = (
    "rev1", "rev5", "mom20", "mom60ex5", "vol20",
    "volratio", "overnight5", "intraday5", "range20", "highdist",
)
REVERSAL = ("rev1", "rev5")
MODELS = ("full", "reversal")
HORIZONS = (5, 10, 20)
# 惩罚加在平均对数似然上，不惩罚截距。网格只在训练窗的滚动预测里比较。
LAMBDA_GRID = (0.1, 1.0)
TOP_FRAC = 0.2
MIN_FOLD_DATES = 12

FEATURE_CN = {
    "rev1": "昨日涨跌",
    "rev5": "近5日涨跌",
    "mom20": "近20日涨跌",
    "mom60ex5": "60日涨跌(不含近5日)",
    "vol20": "20日波动",
    "volratio": "量能",
    "overnight5": "近5日隔夜",
    "intraday5": "近5日日内",
    "range20": "20日振幅",
    "highdist": "距60日高点",
}

ACCEPTED = "高概率一档在测试窗的上涨比例显著更高，扣费后也更好。可以只买预测概率最高的两成。"
REJECTED = "按模型概率从高到低买，没有提高样本外赢率。"


def sigmoid(eta):
    x = np.clip(np.asarray(eta, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logit(X: np.ndarray, y: np.ndarray, lam: float, beta=None) -> np.ndarray:
    """平均对数似然加 L2。X 第一列是截距。lam 不作用在截距上。"""
    n, k = X.shape
    b = np.zeros(k) if beta is None else np.array(beta, dtype=float, copy=True)
    pen = np.full(k, float(lam) * n)
    pen[0] = 0.0
    y = np.asarray(y, dtype=float)
    for _ in range(30):
        eta = X @ b
        p = sigmoid(eta)
        w = np.clip(p * (1.0 - p), 1e-6, None)
        z = eta + (y - p) / w
        A = X.T @ (X * w[:, None])
        A.flat[:: k + 1] += pen
        rhs = X.T @ (w * z)
        step = np.linalg.solve(A, rhs)
        if np.max(np.abs(step - b)) < 1e-8:
            return step
        b = step
    return b


def predict_p(df: pd.DataFrame, features, beta: np.ndarray) -> np.ndarray:
    """缺失的截面标准分按 0（当天平均）计入。"""
    Z = np.zeros((len(df), len(features)), dtype=float)
    for j, f in enumerate(features):
        col = df[f + "_z"].to_numpy(dtype=float)
        Z[:, j] = np.where(np.isfinite(col), col, 0.0)
    return sigmoid(float(beta[0]) + Z @ np.asarray(beta[1:], dtype=float))


def accept_strategy(lift_t, excess_mean, excess_t) -> bool:
    return bool(
        np.isfinite(lift_t) and lift_t > proto.T_MIN
        and np.isfinite(excess_mean) and excess_mean > 0
        and np.isfinite(excess_t) and excess_t > proto.T_MIN
    )


def _features_of(item: dict) -> dict:
    cached = item.get("_win_feat")
    if cached is not None:
        return cached
    close = item["close"]
    open_ = item["open"]
    high = item["high"]
    low = item["low"]
    vol = item["volume"]
    n = close.size
    c = pd.Series(close)
    ret = c.pct_change()
    prev = np.r_[np.nan, close[:-1]]
    gap = np.divide(open_, prev, out=np.full(n, np.nan), where=np.isfinite(prev) & (prev > 0)) - 1.0
    intra = np.divide(close, open_, out=np.full(n, np.nan), where=open_ > 0) - 1.0
    amp = np.divide(high - low, close, out=np.full(n, np.nan), where=close > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mom60 = np.full(n, np.nan)
        if n > 60:
            denom = close[:-60]
            mom60[60:] = np.divide(
                close[55:-5], denom, out=np.full(n - 60, np.nan), where=denom > 0,
            ) - 1.0
        hi = pd.Series(high).rolling(60).max().to_numpy()
        highdist = np.divide(close, hi, out=np.full(n, np.nan), where=np.isfinite(hi) & (hi > 0)) - 1.0
        v = pd.Series(vol)
        volratio = (v.rolling(5).mean() / v.rolling(20).mean().replace(0, np.nan)).to_numpy()
    res = {
        "rev1": c.pct_change(1).to_numpy(),
        "rev5": c.pct_change(5).to_numpy(),
        "mom20": c.pct_change(20).to_numpy(),
        "mom60ex5": mom60,
        "vol20": ret.rolling(20).std(ddof=0).to_numpy(),
        "volratio": volratio,
        "overnight5": pd.Series(gap).rolling(5).mean().to_numpy(),
        "intraday5": pd.Series(intra).rolling(5).mean().to_numpy(),
        "range20": pd.Series(amp).rolling(20).mean().to_numpy(),
        "highdist": highdist,
    }
    item["_win_feat"] = res
    return res


def _panel(horizon: int) -> pd.DataFrame:
    prepared, _ = pnl.load_market()
    # 先收集全市场日期并按 horizon 抽样调仓日，只对调仓日提取信号，避免拼接 600 万行再丢弃 90%
    all_dates = sorted({str(d) for item in prepared for d in item["date"][proto.WARMUP - 1 : -1 - int(horizon)]})
    keep = set(all_dates[:: int(horizon)])
    frames = []
    for item in prepared:
        if "high" not in item:
            continue
        sig = pnl._signals_from_prepared(item, int(horizon), proto.WARMUP)
        if sig is None or sig.empty:
            continue
        sig = sig.loc[sig["date"].astype(str).isin(keep)].copy()
        if sig.empty:
            continue
        feat = _features_of(item)
        loc = {str(d): i for i, d in enumerate(item["date"])}
        idx = [loc[str(d)] for d in sig["date"]]
        for name, arr in feat.items():
            sig[name] = arr[idx]
        frames.append(sig)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    return proto.attach_processed(raw, FEATURES)


def _design(df: pd.DataFrame, features) -> tuple[np.ndarray, np.ndarray]:
    filled = df.loc[(df["block"] == "ok") & df["fwd_ret"].notna()]
    if filled.empty:
        return np.zeros((0, 1 + len(features))), np.zeros(0)
    Z = np.zeros((len(filled), len(features)), dtype=float)
    for j, f in enumerate(features):
        col = filled[f + "_z"].to_numpy(dtype=float)
        Z[:, j] = np.where(np.isfinite(col), col, 0.0)
    X = np.column_stack([np.ones(len(filled)), Z])
    y = (filled["fwd_ret"].to_numpy(dtype=float) > 0).astype(float)
    return X, y


def _day_records(df: pd.DataFrame) -> list[dict]:
    """df 已经带有 p。高概率两成按当天的预测概率取，平手时代码小的在前。"""
    rows = []
    if df is None or df.empty or "p" not in df.columns:
        return rows
    for d, g in df.groupby("date", sort=False):
        g = g.loc[np.isfinite(g["p"].to_numpy(dtype=float))].copy()
        if len(g) < proto.MIN_NAMES:
            continue
        g = g.sort_values(["p", "code"], ascending=[False, True], kind="mergesort")
        k = max(proto.MIN_BOOK, int(round(len(g) * TOP_FRAC)))
        top = g.iloc[:k]
        filled_top = top.loc[(top["block"] == "ok") & top["fwd_ret"].notna()]
        filled_all = g.loc[(g["block"] == "ok") & g["fwd_ret"].notna()]
        if len(filled_top) < proto.MIN_BOOK or len(filled_all) < proto.MIN_NAMES:
            continue
        def pack(part):
            w = 1.0 / len(part)
            weights = {str(c): w for c in part["code"]}
            gross = {str(c): float(r) for c, r in zip(part["code"], part["fwd_ret"])}
            return weights, gross
        tw, tg = pack(filled_top)
        ew, eg = pack(filled_all)
        rows.append({
            "date": str(d),
            "entry": str(filled_all["entry_date"].astype(str).mode().iloc[0]),
            "exit": str(filled_all["exit_date"].astype(str).mode().iloc[0]),
            "win_top": float((filled_top["fwd_ret"] > 0).mean()),
            "win_all": float((filled_all["fwd_ret"] > 0).mean()),
            "top_w": tw, "top_g": tg, "ew_w": ew, "ew_g": eg,
            "n_limit_up": int((top["block"] == "limit_up").sum()),
            "n_limit_down": int((top["block"] == "limit_down").sum()),
        })
    rows.sort(key=lambda r: r["date"])
    return rows


def _summarize(records: list[dict]) -> dict:
    if not records:
        return {
            "n": 0, "lift": np.nan, "lift_t": np.nan, "lift_lags": 0,
            "win_top": np.nan, "win_all": np.nan,
            "excess": np.nan, "excess_t": np.nan, "excess_lags": 0,
            "limit_up": 0, "limit_down": 0,
            "d0": "", "d1": "",
        }
    lift = [r["win_top"] - r["win_all"] for r in records]
    mu, t, lags = proto.newey_west_mean(np.asarray(lift, dtype=float))
    prev_top, prev_ew = {}, {}
    excess = []
    for i, r in enumerate(records):
        last = i == len(records) - 1

        def settle(prev, w, gr, _r=r, _last=last):
            cost = proto.rebalance_cost(prev, w, _r["entry"])
            if _last:
                cost += sum(w.values()) * (proto.COMMISSION + proto.stamp_rate(_r["exit"]))
            gross = sum(w[c] * gr[c] for c in w)
            return gross - cost

        excess.append(settle(prev_top, r["top_w"], r["top_g"]) - settle(prev_ew, r["ew_w"], r["ew_g"]))
        prev_top, prev_ew = r["top_w"], r["ew_w"]
    em, et, el = proto.newey_west_mean(np.asarray(excess, dtype=float))
    return {
        "n": len(records),
        "lift": mu,
        "lift_t": t,
        "lift_lags": lags,
        "win_top": float(np.mean([r["win_top"] for r in records])),
        "win_all": float(np.mean([r["win_all"] for r in records])),
        "excess": em,
        "excess_t": et,
        "excess_lags": el,
        "limit_up": int(sum(r["n_limit_up"] for r in records)),
        "limit_down": int(sum(r["n_limit_down"] for r in records)),
        "d0": records[0]["date"],
        "d1": records[-1]["date"],
    }


def _walk(train: pd.DataFrame, features, lam: float) -> dict:
    if train is None or train.empty:
        return _summarize([]) | {"n_fit": 0}
    dates = sorted(train["date"].astype(str).unique())
    filled = train.loc[(train["block"] == "ok") & train["fwd_ret"].notna()]
    if filled.empty:
        return _summarize([]) | {"n_fit": 0}
    X_all, y_all = _design(filled, features)
    exit_arr = filled["exit_date"].astype(str).to_numpy()
    f_date_arr = filled["date"].astype(str).to_numpy()
    # 预先计算每个日期对应的已卖出样本数及不同日期数
    uniq_f_dates, inv_idx = np.unique(f_date_arr, return_inverse=True)
    n_uniq_total = len(uniq_f_dates)
    day_groups = {str(d): g.copy() for d, g in train.groupby("date", sort=False)}

    beta = None
    parts = []
    n_fit = 0
    for d in dates:
        mask = exit_arr < d
        if int(np.sum(mask)) < 200:
            continue
        if int(np.sum(np.bincount(inv_idx[mask], minlength=n_uniq_total) > 0)) < MIN_FOLD_DATES:
            continue
        X = X_all[mask]
        y = y_all[mask]
        if y.min() == y.max():
            continue
        beta = fit_logit(X, y, lam, beta)
        n_fit += 1
        day = day_groups[d]
        day["p"] = predict_p(day, features, beta)
        parts.append(day)
    pred = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    stats = _summarize(_day_records(pred))
    stats["n_fit"] = n_fit
    return stats


def _rank_key(stats: dict, horizon: int, model: str):
    t = stats.get("lift_t", np.nan)
    t = float(t) if np.isfinite(t) else -1e9
    # t 最大；其次持有期短；其次反转模型（参数少）。
    return (t, -int(horizon), 1 if model == "reversal" else 0)


def _latest_full_date(prepared: list[dict]) -> str:
    from collections import Counter
    cnt = Counter()
    for item in prepared:
        for d in item["date"][-3:]:
            cnt[str(d)] += 1
    if not cnt:
        return ""
    threshold = max(proto.MIN_NAMES, len(prepared) // 2)
    valid = [d for d, c in cnt.items() if c >= threshold]
    return max(valid) if valid else max(cnt.keys())


def _latest(beta, features) -> tuple[str, list[dict]]:
    prepared, _ = pnl.load_market()
    if not prepared:
        return "", []
    asof = _latest_full_date(prepared)
    rows = []
    for item in prepared:
        if "high" not in item:
            continue
        dates = item["date"]
        if str(dates[-1]) == asof:
            pos = -1
        else:
            m = np.where(dates == asof)[0]
            if len(m) == 0:
                continue
            pos = int(m[-1])
        feat = _features_of(item)
        rec = {
            "date": asof,
            "code": str(item["code"]),
            "industry": item["industry"] or "未分类",
            "close": float(item["close"][pos]),
        }
        ok = True
        for name in FEATURES:
            val = float(feat[name][pos])
            rec[name] = val
            if name in features and not np.isfinite(val):
                ok = False
        if np.isfinite(rec["close"]):
            rows.append(rec)
        else:
            continue
        if not ok:
            # 仍保留该行，缺失项在预测时按 0。
            pass
    if not rows:
        return asof, []
    df = proto.attach_processed(pd.DataFrame(rows), FEATURES)
    df["p"] = predict_p(df, features, beta)
    df = df.sort_values(["p", "code"], ascending=[False, True], kind="mergesort")
    k = max(proto.MIN_BOOK, int(round(len(df) * TOP_FRAC)))
    names = {}
    slist = db.load_stock_list()
    if slist is not None and not slist.empty and "name" in slist.columns:
        slist = slist.copy()
        slist["code"] = slist["code"].astype(str).str.zfill(6)
        names = dict(zip(slist["code"], slist["name"].astype(str)))
    b = np.asarray(beta, dtype=float)
    stocks = []
    for rank, r in enumerate(df.itertuples(index=False)):
        drivers = []
        for j, f in enumerate(features):
            z = getattr(r, f + "_z")
            z = float(z) if np.isfinite(z) else 0.0
            drivers.append((abs(b[j + 1] * z), f, z, float(b[j + 1] * z)))
        drivers.sort(reverse=True)
        stocks.append({
            "code": str(r.code),
            "name": names.get(str(r.code), ""),
            "industry": str(r.industry),
            "close": round(float(r.close), 4),
            "p_up": float(r.p),
            "top": rank < k,
            "drivers": [
                {"feature": f, "name": FEATURE_CN[f], "z": round(z, 3), "contrib": round(c, 3)}
                for _, f, z, c in drivers[:3]
            ],
        })
    return asof, stocks


def self_check() -> list[str]:
    lines = []
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        lines.append(("PASS " if cond else "FAIL ") + name)

    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    p = sigmoid(-0.1 + 1.1 * x)
    y = rng.random(500) < p
    X = np.column_stack([np.ones(500), x])
    b = fit_logit(X, y.astype(float), 0.1)
    check("正系数方向被找回来", b[1] > 0.4)
    p_hi = float(sigmoid(b[0] + b[1] * 2.0))
    p_lo = float(sigmoid(b[0] + b[1] * -2.0))
    check("高特征值的上涨概率更高", p_hi > p_lo + 0.2)
    check("三条都过才启用策略", accept_strategy(2.1, 0.01, 2.1))
    check("赢率 t 不够不启用", not accept_strategy(1.9, 0.01, 3))
    check("超额为负不启用", not accept_strategy(3, -0.01, 3))
    check("超额 t 不够不启用", not accept_strategy(3, 0.01, 1.5))

    # 卖出日不早于预测日的样本不能进入拟合。这里只核对照。
    toy = pd.DataFrame({
        "date": ["2021-06-01", "2021-07-01"],
        "exit_date": ["2021-07-01", "2021-08-01"],
    })
    # 预测 2021-08-01 时，卖出日等于这一天的样本还不知道结果。
    past = toy.loc[toy["exit_date"].astype(str) < "2021-08-01"]
    check("只把已经卖出的样本放进拟合", list(past["date"]) == ["2021-06-01"])
    if not ok:
        raise RuntimeError("赢率模型核对失败:\n" + "\n".join(lines))
    return lines


def _root() -> str:
    return os.path.dirname(os.path.dirname(db.DB_PATH))


def state_path() -> str:
    return os.path.join(_root(), "win_state.json")


def record_path() -> str:
    return os.path.join(_root(), "win_record.txt")


def load_state() -> dict | None:
    path = state_path()
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _num(x, nd=2):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{float(x):.{nd}f}"


def _pct(x):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{float(x) * 100:.2f}%"


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


def run() -> dict:
    lines = []
    lines.append("逐股上涨概率")
    lines.append("持有期、模型和惩罚只在训练窗的滚动预测里选定。测试窗只评价胜出的那一组。")
    lines.append("")
    lines.append("冻结规则")
    lines.append(f"  训练截止日（卖出日不晚于此日）: {proto.TRAIN_END}")
    lines.append(f"  持有期候选: {HORIZONS}")
    lines.append(f"  模型: 全部特征 / 只留昨日和近5日涨跌。惩罚网格: {LAMBDA_GRID}")
    lines.append(f"  特征: {', '.join(FEATURE_CN[f] for f in FEATURES)}")
    lines.append("  上涨: 能成交且开盘到开盘毛收益 > 0。未成交不记成跌，也不进拟合。")
    lines.append(f"  高概率一档: 当天预测概率最高的 {TOP_FRAC:.0%}。")
    lines.append("  训练窗比较指标: 这一档的上涨比例减去全体，Newey-West t 最大者胜出。")
    lines.append("  启用策略: 测试窗赢率提升 t > 2，且扣费后超额均值 > 0、超额 t > 2。")
    lines.append(f"  启用时的结论: {ACCEPTED}")
    lines.append(f"  不启用时的结论: {REJECTED}")
    lines.append("")
    lines.append("机械核对")
    try:
        for row in proto.self_check():
            lines.append("  " + row)
        for row in pnl.self_check_signals():
            lines.append("  " + row)
        for row in self_check():
            lines.append("  " + row)
    except Exception as e:
        lines.append("  FAIL " + str(e))
        _write("\n".join(lines) + "\n")
        raise

    print("核对通过，开始构建特征...", flush=True)
    panels = {}
    for h in HORIZONS:
        print(f"持有期 {h} ...", flush=True)
        panels[h] = _panel(h)
    _, audit = pnl.load_market()
    lines.append("")
    lines.append("数据")
    lines.append(f"  股票数: {audit['n_codes']}  日期: {audit['date_min']} ~ {audit['date_max']}")
    lines.append("  股票池是现存代码，不是含退市股的历史全市场。行业是当前行业。")

    lines.append("")
    lines.append("训练窗滚动预测（用来选模型，不是测试结论）")
    cand = []
    for h in HORIZONS:
        train = proto.train_slice(panels[h])
        for model in MODELS:
            feats = FEATURES if model == "full" else REVERSAL
            for lam in LAMBDA_GRID:
                print(f"  滚动预测 持有期{h} {model} λ={lam}", flush=True)
                stats = _walk(train, feats, lam)
                cand.append((h, model, lam, stats))
                lines.append(
                    f"  持有期{h} {model} λ={lam}: "
                    f"滚动期数={stats['n']} 拟合次数={stats['n_fit']}  "
                    f"高档赢率={_pct(stats['win_top'])} 全体={_pct(stats['win_all'])}  "
                    f"提升={_pct(stats['lift'])} t={_num(stats['lift_t'])}"
                )
    best = max(cand, key=lambda it: _rank_key(it[3], it[0], it[1]))
    h, model, lam, cv = best
    feats = FEATURES if model == "full" else REVERSAL
    lines.append("")
    lines.append(f"训练窗选中: 持有期 {h}，模型 {model}，λ={lam}。其他组合的测试窗不计算。")

    print("用训练窗全样本重拟合，并只评价一次测试窗...", flush=True)
    train = proto.train_slice(panels[h])
    X, y = _design(train, feats)
    beta = fit_logit(X, y, lam)
    test = proto.test_slice(panels[h]).copy()
    test["p"] = predict_p(test, feats, beta)
    ev = _summarize(_day_records(test))
    adopted = accept_strategy(ev["lift_t"], ev["excess"], ev["excess_t"])
    conclusion = ACCEPTED if adopted else REJECTED

    lines.append("")
    lines.append("测试窗")
    lines.append(f"  信号日: {ev['d0']} ~ {ev['d1']}  调仓日={ev['n']}")
    lines.append(f"  高概率一档上涨比例={_pct(ev['win_top'])}")
    lines.append(f"  全体可成交上涨比例={_pct(ev['win_all'])}")
    lines.append(f"  提升={_pct(ev['lift'])}  NW-t={_num(ev['lift_t'])}  滞后={ev['lift_lags']}")
    lines.append(f"  扣费后超额={_pct(ev['excess'])}  NW-t={_num(ev['excess_t'])}  滞后={ev['excess_lags']}")
    lines.append(f"  高档涨停未买入={ev['limit_up']}  跌停未卖出={ev['limit_down']}")
    lines.append("  系数（截距，然后是各特征；特征已在截面内标准化）")
    lines.append(f"    截距={_num(beta[0], 3)}")
    for j, f in enumerate(feats):
        lines.append(f"    {FEATURE_CN[f]}={_num(beta[j + 1], 3)}")
    lines.append("")
    lines.append(conclusion)

    print("最近收盘日...", flush=True)
    asof, stocks = _latest(beta, feats)
    lines.append("")
    lines.append(f"最近收盘日: {asof}  股票数: {len(stocks)}")
    if stocks:
        ps = [s["p_up"] for s in stocks]
        lines.append(f"  上涨概率从 {_pct(min(ps))} 到 {_pct(max(ps))}，中位数 {_pct(float(np.median(ps)))}")
    lines.append("最近收盘日没有未来收益，不参与拟合和测试。")
    lines.append("")
    lines.append("重跑: venv\\Scripts\\python.exe -m app.strategy.winmodel")
    text = "\n".join(lines) + "\n"
    _write(text)
    state = {
        "adopted": adopted,
        "conclusion": conclusion,
        "horizon": h,
        "model": model,
        "lam": lam,
        "features": list(feats),
        "beta": [float(v) for v in beta],
        "feature_cn": {f: FEATURE_CN[f] for f in feats},
        "cv_lift_t": cv["lift_t"],
        "cv_win_top": cv["win_top"],
        "cv_win_all": cv["win_all"],
        "test": {k: ev[k] for k in (
            "n", "lift", "lift_t", "win_top", "win_all", "excess", "excess_t",
            "limit_up", "limit_down", "d0", "d1",
        )},
        "asof": asof,
        "stocks": stocks,
        "train_end": proto.TRAIN_END,
    }
    with open(state_path(), "w", encoding="utf-8") as f:
        json.dump(_jsonable(state), f, ensure_ascii=False)
    print(text)
    print("记录已写入", record_path(), flush=True)
    return state


def refresh_latest_only() -> dict:
    """利用已保存的训练窗系数，仅重新生成最近一个完整交易日的全市场截面概率。"""
    st = load_state()
    if not st or not st.get("beta") or not st.get("features"):
        return run()
    feats = tuple(st["features"])
    beta = np.asarray(st["beta"], dtype=float)
    asof, stocks = _latest(beta, feats)
    st["asof"] = asof
    st["stocks"] = stocks
    with open(state_path(), "w", encoding="utf-8") as f:
        json.dump(_jsonable(st), f, ensure_ascii=False)
    rec_file = record_path()
    if os.path.exists(rec_file):
        with open(rec_file, encoding="utf-8") as f:
            old_txt = f.read()
        marker = "\n最近收盘日:"
        pos = old_txt.find(marker)
        if pos >= 0:
            tail = [f"\n最近收盘日: {asof}  股票数: {len(stocks)}"]
            if stocks:
                ps = [s["p_up"] for s in stocks]
                tail.append(f"  上涨概率从 {_pct(min(ps))} 到 {_pct(max(ps))}，中位数 {_pct(float(np.median(ps)))}")
            tail.append("最近收盘日没有未来收益，不参与拟合和测试。")
            tail.append("")
            tail.append("重跑: venv\\Scripts\\python.exe -m app.strategy.winmodel")
            _write(old_txt[:pos] + "\n".join(tail) + "\n")
    return st


def _write(text: str) -> None:
    with open(record_path(), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    run()
