"""
策略层 - 策略基类与内置策略
============================================
【如何自定义你自己的策略】
  1. 新建一个类继承 BaseStrategy
  2. 在 params 里声明可调参数(界面会自动生成滑块)
  3. 实现 evaluate(df) -> (是否入选, 打分, 说明)
  4. 在 ALL_STRATEGIES 里注册
就这么简单。参数会自动出现在 UI 上供用户调节。
"""
from dataclasses import dataclass, field

import pandas as pd

from . import indicators as ind
from . import chip


@dataclass
class Param:
    """一个可调参数的描述,UI 用它生成控件。"""
    key: str
    label: str
    default: float
    min: float
    max: float
    is_int: bool = True


class BaseStrategy:
    name = "基础策略"
    desc = ""
    params: list = []

    def __init__(self):
        # 用默认值初始化参数
        self.values = {p.key: p.default for p in self.params}

    def set_param(self, key, value):
        self.values[key] = value

    def evaluate(self, df: pd.DataFrame):
        """
        输入:单只股票已计算指标的日线 df(升序,最后一行是最新)
        返回:(selected: bool, score: float, reason: str)
        score 用于结果排序,越大越靠前。
        子类必须重写。
        """
        raise NotImplementedError


class MeanReversion(BaseStrategy):
    """均值回归:股价短期超跌、偏离均线较多时,预期回归。"""
    name = "均值回归"
    desc = "收盘价跌破布林下轨且 RSI 超卖,预期反弹"
    params = [
        Param("rsi_th", "RSI 低于", 30, 10, 50),
        Param("dev_pct", "低于MA20幅度(%)", 5, 1, 20),
    ]

    def evaluate(self, df):
        if len(df) < 25:
            return False, 0, "数据不足"
        row = df.iloc[-1]
        rsi_th = self.values["rsi_th"]
        dev = self.values["dev_pct"] / 100.0
        # 条件1: RSI 超卖
        cond_rsi = row["rsi14"] < rsi_th
        # 条件2: 收盘价明显低于 MA20
        below = (row["ma20"] - row["close"]) / row["ma20"]
        cond_dev = below > dev
        # 条件3: 跌破布林下轨
        cond_boll = row["close"] < row["boll_low"]
        selected = bool(cond_rsi and cond_dev)
        # 打分:超卖越深、偏离越大,分越高
        score = (rsi_th - row["rsi14"]) + below * 100 + (10 if cond_boll else 0)
        reason = f"RSI={row['rsi14']:.1f}, 低于MA20 {below*100:.1f}%"
        return selected, float(score), reason


class PotentialStock(BaseStrategy):
    """
    潜力股:处于上升趋势、温和放量、有上涨动能的股票。
    设计原则:用"打分制"而非"全部硬门槛 and 连乘",避免条件过严导致选不出。
    只要趋势方向对、动量为正,就纳入候选并按强弱打分排序。
    """
    name = "潜力股"
    desc = "均线向上、温和放量、上升动能强,处于上升趋势"
    params = [
        Param("vol_ratio", "放量倍数(近5日/近20日)", 1.1, 1.0, 3.0, is_int=False),
        Param("mom_days", "动量天数", 20, 5, 60),
        Param("min_up_pct", "期间涨幅(%)", 3, 0, 50),
    ]

    def evaluate(self, df):
        if len(df) < 65:
            return False, 0, "数据不足"
        row = df.iloc[-1]
        mom_days = int(self.values["mom_days"])
        vol_ratio = self.values["vol_ratio"]
        min_up = self.values["min_up_pct"]

        # --- 硬门槛(必要条件,尽量宽松) ---
        # 趋势向上:短期均线在长期均线之上(ma5>ma20),且 ma20>ma60(中期也向上)
        cond_trend = row["ma5"] > row["ma20"] and row["ma20"] > row["ma60"]
        # 动量为正:近 mom_days 涨幅达标
        past = df["close"].iloc[-mom_days]
        up = (row["close"] - past) / past * 100
        cond_mom = up > min_up

        # 温和放量:近5日均量 / 近20日均量(比"当日放量"稳健,不易被单日噪声左右)
        vol_ma5 = df["volume"].rolling(5).mean().iloc[-1]
        vol_ma20 = df["volume"].rolling(20).mean().iloc[-1]
        vol_expand = vol_ma5 / (vol_ma20 + 1e-9)
        cond_vol = vol_expand > vol_ratio

        # 入选:趋势 + 动量为必要,放量作为加分而非硬门槛
        selected = bool(cond_trend and cond_mom)

        # --- 打分:趋势强度 + 动量 + 放量,越强分越高 ---
        ma_spread = (row["ma5"] - row["ma60"]) / (row["ma60"] + 1e-9) * 100  # 均线发散度
        score = up + ma_spread + (vol_expand - 1) * 20 + (5 if cond_vol else 0)
        reason = f"{mom_days}日涨{up:.1f}%, 量能放大{vol_expand:.2f}x"
        return selected, float(score), reason


class MacdGoldenCross(BaseStrategy):
    """
    MACD 金叉:DIF 上穿 DEA(由下向上),且发生在 0 轴附近或之上,
    表示短期动能转强,是经典的买点信号。
    """
    name = "MACD金叉"
    desc = "DIF 上穿 DEA 形成金叉,短期动能转强"
    params = [
        Param("recent_days", "金叉发生在近N日内", 3, 1, 10),
        Param("min_dif", "DIF 不低于(0轴过滤)", 0, -1, 1, is_int=False),
    ]

    def evaluate(self, df):
        if len(df) < 40:
            return False, 0, "数据不足"
        recent = int(self.values["recent_days"])
        min_dif = self.values["min_dif"]
        dif = df["dif"]
        dea = df["dea"]
        # 找近 recent 天内是否发生金叉:前一日 dif<dea,当日 dif>=dea
        crossed = False
        cross_ago = None
        for k in range(1, recent + 1):
            if dif.iloc[-k] >= dea.iloc[-k] and dif.iloc[-k - 1] < dea.iloc[-k - 1]:
                crossed = True
                cross_ago = k - 1
                break
        row = df.iloc[-1]
        cond_dif = row["dif"] >= min_dif  # 过滤深水区(0轴下太远)的弱金叉
        selected = bool(crossed and cond_dif)
        # 打分:柱状体越大、DIF 位置越高越好
        score = row["macd_bar"] * 10 + row["dif"] * 5
        if crossed:
            ago_txt = "今日" if cross_ago == 0 else f"{cross_ago}日前"
            reason = f"{ago_txt}金叉, DIF={row['dif']:.2f} 柱={row['macd_bar']:.2f}"
        else:
            reason = f"近{recent}日无金叉, DIF={row['dif']:.2f}"
        return selected, float(score), reason


class BreakoutNewHigh(BaseStrategy):
    """
    放量突破:股价创近 N 日新高,同时明显放量,
    是趋势启动/突破形态的典型信号(强者恒强)。
    """
    name = "放量突破新高"
    desc = "创近N日新高 + 放量,突破形态,强势启动"
    params = [
        Param("high_days", "突破周期(日)", 20, 10, 120),
        Param("vol_ratio", "放量倍数(量比)", 1.5, 1.0, 5.0, is_int=False),
    ]

    def evaluate(self, df):
        need = int(self.values["high_days"]) + 5
        if len(df) < need:
            return False, 0, "数据不足"
        row = df.iloc[-1]
        hd = int(self.values["high_days"])
        # 突破:当日收盘 > 过去 hd 日最高(不含当日)
        prev_high = df["high"].rolling(hd).max().shift(1).iloc[-1]
        cond_break = row["close"] > prev_high
        # 放量:当日量比达标
        cond_vol = row["vol_ratio"] > self.values["vol_ratio"]
        selected = bool(cond_break and cond_vol)
        score = (row["close"] / (prev_high + 1e-9) - 1) * 100 * 5 + row["vol_ratio"] * 3
        reason = f"破{hd}日高, 量比{row['vol_ratio']:.2f}"
        return selected, float(score), reason


class PullbackMa(BaseStrategy):
    """
    回踩均线:上升趋势中(ma20 向上),股价回调到 MA20 附近获得支撑,
    是"上车"的低吸点,风险相对突破买入更低。
    """
    name = "回踩均线支撑"
    desc = "上升趋势中股价回踩MA20附近,低吸机会"
    params = [
        Param("near_pct", "距MA20幅度(%)", 3, 1, 10, is_int=False),
        Param("trend_days", "趋势确认(MA20向上N日)", 5, 3, 20),
    ]

    def evaluate(self, df):
        if len(df) < 65:
            return False, 0, "数据不足"
        row = df.iloc[-1]
        near = self.values["near_pct"] / 100.0
        td = int(self.values["trend_days"])
        # 趋势:ma20 处于上升(当前 > td 日前),且 ma20>ma60
        cond_trend = df["ma20"].iloc[-1] > df["ma20"].iloc[-1 - td] and row["ma20"] > row["ma60"]
        # 回踩:收盘价在 MA20 附近(上下 near 幅度内),且股价仍在 MA60 之上(趋势未破)
        dist = abs(row["close"] - row["ma20"]) / (row["ma20"] + 1e-9)
        cond_near = dist < near and row["close"] > row["ma60"]
        selected = bool(cond_trend and cond_near)
        # 打分:越贴近 MA20、趋势越强越好
        ma_slope = (df["ma20"].iloc[-1] - df["ma20"].iloc[-1 - td]) / (df["ma20"].iloc[-1 - td] + 1e-9) * 100
        score = (near - dist) * 100 + ma_slope
        reason = f"距MA20 {dist*100:.1f}%, MA20 {td}日斜率{ma_slope:.1f}%"
        return selected, float(score), reason


class VolumePriceRise(BaseStrategy):
    """
    量价齐升(主升浪):连续放量上涨,短中期均线多头,
    捕捉正在加速上涨的强势股。
    """
    name = "量价齐升"
    desc = "连续放量上涨、均线多头,主升浪加速中"
    params = [
        Param("up_pct_5", "近5日涨幅(%)", 5, 1, 30, is_int=False),
        Param("vol_ratio", "量能放大倍数", 1.3, 1.0, 3.0, is_int=False),
    ]

    def evaluate(self, df):
        if len(df) < 65:
            return False, 0, "数据不足"
        row = df.iloc[-1]
        # 近5日涨幅
        cond_up = row["chg_5"] > self.values["up_pct_5"]
        # 均线多头:ma5>ma10>ma20
        cond_ma = row["ma5"] > row["ma10"] > row["ma20"]
        # 量能放大:近5日均量 > 近10日均量 * ratio
        vol_expand = row["vol_ma5"] / (row["vol_ma10"] + 1e-9)
        cond_vol = vol_expand > self.values["vol_ratio"]
        selected = bool(cond_up and cond_ma and cond_vol)
        score = row["chg_5"] + (vol_expand - 1) * 30
        reason = f"5日涨{row['chg_5']:.1f}%, 量能{vol_expand:.2f}x"
        return selected, float(score), reason


class OversoldRebound(BaseStrategy):
    """
    超跌反弹:短期大幅下跌后 RSI 极度超卖,且当日出现放量止跌(阳线),
    博取快速反弹。比纯均值回归更强调"止跌信号"。
    """
    name = "超跌反弹"
    desc = "短期大跌+RSI超卖+当日放量阳线,博反弹"
    params = [
        Param("down_pct", "近20日跌幅超(%)", 15, 5, 50, is_int=False),
        Param("rsi_th", "RSI 低于", 35, 10, 50),
    ]

    def evaluate(self, df):
        if len(df) < 30:
            return False, 0, "数据不足"
        row = df.iloc[-1]
        # 近20日大幅下跌
        cond_down = row["chg_20"] < -self.values["down_pct"]
        # RSI 超卖
        cond_rsi = row["rsi14"] < self.values["rsi_th"]
        # 当日止跌信号:收阳(close>open)且放量(量比>1)
        cond_stop = row["close"] > row["open"] and row["vol_ratio"] > 1.0
        selected = bool(cond_down and cond_rsi and cond_stop)
        score = (-row["chg_20"]) + (self.values["rsi_th"] - row["rsi14"]) + row["vol_ratio"] * 5
        reason = f"20日跌{row['chg_20']:.1f}%, RSI={row['rsi14']:.1f}, 放量阳"
        return selected, float(score), reason


class MultiFactor(BaseStrategy):
    """
    多因子综合选股(旗舰策略):把多个维度的因子标准化后加权合成一个综合分,
    避免单一信号的偶然性,追求更稳健的选股。融合了各单策略的优点:

      · 趋势因子   —— 均线多头排列强度(ma5/ma20/ma60 发散度)
      · 动量因子   —— 近 20 日涨幅(强者恒强)
      · 量能因子   —— 近5日/近20日均量放大倍数(资金流入)
      · 位置因子   —— 距离近60日高点的空间(不追太高)
      · 健康度因子 —— RSI 处于健康区间(不超买不超卖)

    设计:每个因子先各自打分(0~1 量级),再按权重线性合成。
    权重可在界面调节,权重为0即关闭该因子。只要综合分达到阈值即入选,
    按综合分排序 —— 这本质是一个"可调权重的选股打分器"。
    """
    name = "多因子综合"
    desc = "趋势+动量+量能+位置+健康度 多因子加权综合打分,稳健选股"
    params = [
        Param("w_trend", "趋势权重", 30, 0, 100),
        Param("w_mom", "动量权重", 25, 0, 100),
        Param("w_vol", "量能权重", 20, 0, 100),
        Param("w_pos", "位置权重", 15, 0, 100),
        Param("w_rsi", "健康度权重", 10, 0, 100),
        Param("min_score", "入选综合分(0-100)", 55, 0, 100),
    ]

    def evaluate(self, df):
        if len(df) < 65:
            return False, 0, "数据不足"
        row = df.iloc[-1]

        # ---- 各因子归一到 0~1 ----
        # 1) 趋势:多头排列 + 均线发散度(ma5 相对 ma60 的领先幅度)
        spread = (row["ma5"] - row["ma60"]) / (row["ma60"] + 1e-9)  # 可正可负
        f_trend = max(0.0, min(spread / 0.15, 1.0))  # 领先15%封顶
        if not (row["ma5"] > row["ma20"] > row["ma60"]):
            f_trend *= 0.3  # 非多头排列大幅折扣

        # 2) 动量:近20日涨幅,0~30% 映射到 0~1
        past = df["close"].iloc[-20]
        mom = (row["close"] - past) / (past + 1e-9)
        f_mom = max(0.0, min(mom / 0.30, 1.0))

        # 3) 量能:近5日/近20日均量,1.0~2.0x 映射到 0~1
        v5 = df["volume"].rolling(5).mean().iloc[-1]
        v20 = df["volume"].rolling(20).mean().iloc[-1]
        vexp = v5 / (v20 + 1e-9)
        f_vol = max(0.0, min((vexp - 1.0) / 1.0, 1.0))

        # 4) 位置:距近60日高点的回落空间(离高点 0~20% 给高分,越接近高点越低)
        hi60 = df["high"].rolling(60).max().iloc[-1]
        gap = (hi60 - row["close"]) / (hi60 + 1e-9)  # 0=在高点, 越大离高点越远
        # 甜蜜区:离高点 3%~15%(既有突破潜力又不追高)
        if gap <= 0.03:
            f_pos = 0.5
        elif gap <= 0.15:
            f_pos = 1.0
        else:
            f_pos = max(0.0, 1.0 - (gap - 0.15) / 0.25)

        # 5) 健康度:RSI 在 45~70 最佳(有动能又不超买)
        rsi = row["rsi14"]
        if 45 <= rsi <= 70:
            f_rsi = 1.0
        elif rsi < 45:
            f_rsi = max(0.0, rsi / 45.0)
        else:  # >70 超买,递减
            f_rsi = max(0.0, 1.0 - (rsi - 70) / 30.0)

        # ---- 权重归一化后线性合成,得 0~100 综合分 ----
        w = [self.values["w_trend"], self.values["w_mom"], self.values["w_vol"],
             self.values["w_pos"], self.values["w_rsi"]]
        f = [f_trend, f_mom, f_vol, f_pos, f_rsi]
        wsum = sum(w) + 1e-9
        comp = sum(wi * fi for wi, fi in zip(w, f)) / wsum * 100

        selected = bool(comp >= self.values["min_score"])
        reason = (f"综合{comp:.0f} | 趋势{f_trend:.2f} 动量{f_mom:.2f} "
                  f"量{f_vol:.2f} 位{f_pos:.2f} RSI{f_rsi:.2f}")
        return selected, float(comp), reason


# ============================================================================
# 筹码形态策略 (同花顺"筹码形态洞察"同款: 低位锁定/低位密集/双峰/高位密集)
# ----------------------------------------------------------------------------
# 复用 chip.compute_chip_distribution(通达信同款筹码模型) + chip.analyze_chip_shape
# 提取的形态中间量。scanner 传入的 df 无流通股本, 筹码计算走 turn=3% 兜底; 形态判据
# 用的都是"相对形状"(现价在历史区间位置/集中度/获利盘/横盘度/峰形), 不依赖精确股本,
# 结论稳健。四个策略各自独立出现在下拉框, 命名直观。
# ============================================================================
class _ChipShapeBase(BaseStrategy):
    """筹码形态策略基类: 统一算好筹码分布与形态中间量, 子类只写 _judge。"""
    # 子类可覆盖: 筹码计算至少需要的日线长度
    _min_len = 60

    def _shape(self, df):
        """返回 chip.analyze_chip_shape 的中间量 dict, 数据不足返回 None。"""
        if df is None or len(df) < self._min_len:
            return None
        cr = chip.compute_chip_distribution(df)
        return chip.analyze_chip_shape(cr, df)

    def evaluate(self, df):
        sh = self._shape(df)
        if not sh:
            return False, 0, "数据不足/筹码不可算"
        return self._judge(sh)

    def _judge(self, sh):  # 子类实现
        raise NotImplementedError


class ChipLowLock(_ChipShapeBase):
    """
    低位锁定(主力拉升): 筹码高度集中在低位并已被锁定, 现价从低位抬起,
    低位筹码普遍浮盈(获利盘高), 温和上行 —— 主力控盘后拉升的典型形态。
    """
    name = "筹码-低位锁定"
    desc = "低位筹码密集锁定+已从底部抬升,获利盘高,主力控盘拉升"
    params = [
        Param("max_conc", "集中度上限(越小越密)", 25, 10, 60),
        Param("max_pos", "现价位置上限(0-100)", 45, 20, 70),
        Param("min_profit", "获利盘下限(%)", 55, 30, 90),
    ]

    def _judge(self, sh):
        conc = sh["concentration"] * 100
        pos = sh["price_pos"] * 100
        profit = sh["profit_ratio"] * 100
        cond_conc = conc <= self.values["max_conc"]          # 筹码密集
        cond_pos = pos <= self.values["max_pos"]             # 现价仍处相对低位
        cond_profit = profit >= self.values["min_profit"]    # 低位筹码浮盈
        cond_up = sh["trend_20"] > 0                         # 已从底部抬起
        selected = bool(cond_conc and cond_pos and cond_profit and cond_up)
        # 打分: 越密集、获利盘越高、抬升越温和越好
        score = (self.values["max_conc"] - conc) + profit * 0.5 + sh["trend_20"]
        reason = (f"集中度{conc:.0f} 位置{pos:.0f} 获利{profit:.0f}% "
                  f"20日{sh['trend_20']:+.1f}%")
        return selected, float(score), reason


class ChipLowDense(_ChipShapeBase):
    """
    低位密集(主力建仓): 大跌后在低位横盘, 筹码在低价区高度密集,
    现价贴近平均成本(获利盘中性, 多空成本趋同) —— 主力低位吸筹的典型形态。
    """
    name = "筹码-低位密集"
    desc = "低位横盘+筹码密集+现价贴近平均成本,主力建仓吸筹"
    params = [
        Param("max_conc", "集中度上限(越小越密)", 22, 10, 50),
        Param("max_pos", "现价位置上限(0-100)", 40, 20, 60),
        Param("max_consol", "横盘振幅上限(%)", 18, 8, 40),
    ]

    def _judge(self, sh):
        conc = sh["concentration"] * 100
        pos = sh["price_pos"] * 100
        consol = (sh["consolidation"] * 100) if sh["consolidation"] is not None else 999
        # 现价与均价的偏离(%),越接近说明多空成本趋同
        dev = abs(sh["last_close"] - sh["avg_cost"]) / (sh["avg_cost"] + 1e-9) * 100
        cond_conc = conc <= self.values["max_conc"]
        cond_pos = pos <= self.values["max_pos"]
        cond_consol = consol <= self.values["max_consol"]    # 近期横盘
        cond_dev = dev <= 8                                  # 现价≈平均成本
        selected = bool(cond_conc and cond_pos and cond_consol and cond_dev)
        score = (self.values["max_conc"] - conc) + (self.values["max_consol"] - consol) \
            + (8 - dev)
        reason = (f"集中度{conc:.0f} 位置{pos:.0f} 振幅{consol:.0f}% "
                  f"距均价{dev:.1f}%")
        return selected, float(score), reason


class ChipTwinPeak(_ChipShapeBase):
    """
    双峰形态(高抛低吸): 成本分布出现上下两个明显的筹码密集峰, 中间有低谷,
    对应一轮上涨后套牢盘(上峰)与低位成本盘(下峰)并存 —— 关注筹码是上移还是下沉。
    """
    name = "筹码-双峰形态"
    desc = "成本分布上下双峰+中间低谷,套牢盘与低成本盘并存"
    params = [
        Param("min_gap", "双峰最小价差(%)", 12, 5, 40),
        Param("max_gap", "双峰最大价差(%)", 80, 40, 200),
        Param("max_valley", "谷深上限(0-100)", 55, 20, 80),
    ]

    def _judge(self, sh):
        tw = sh["twin"]
        if not tw:
            return False, 0, "无双峰"
        gap = tw["gap_pct"]
        valley = tw["valley_ratio"] * 100
        cond_gap = self.values["min_gap"] <= gap <= self.values["max_gap"]  # 价差在合理区间
        cond_valley = valley <= self.values["max_valley"]    # 谷够深(双峰分明)
        selected = bool(cond_gap and cond_valley)
        # 打分: 谷越深、两峰越均衡越"标准", 价差适中加分
        score = (self.values["max_valley"] - valley) + tw["peak_balance"] * 30
        reason = (f"下峰{tw['lower_price']:.2f} 上峰{tw['upper_price']:.2f} "
                  f"价差{gap:.0f}% 谷深{valley:.0f} 均衡{tw['peak_balance']:.2f}")
        return selected, float(score), reason


class ChipHighDense(_ChipShapeBase):
    """
    高位密集(高位套牢/派发风险): 筹码在高价区高度密集, 现价处于历史高位,
    获利盘极高(浮盈盘沉重, 潜在抛压大) —— 高位横盘密集, 需警惕见顶派发。
    这是低位锁定的镜像形态: 密集但位置在顶, 意义相反(风险 > 机会)。
    """
    name = "筹码-高位密集"
    desc = "高位筹码密集+现价处历史高位+获利盘沉重,警惕见顶派发"
    params = [
        Param("max_conc", "集中度上限(越小越密)", 25, 10, 60),
        Param("min_pos", "现价位置下限(0-100)", 65, 50, 90),
        Param("min_profit", "获利盘下限(%)", 80, 60, 99),
    ]

    def _judge(self, sh):
        conc = sh["concentration"] * 100
        pos = sh["price_pos"] * 100
        profit = sh["profit_ratio"] * 100
        cond_conc = conc <= self.values["max_conc"]          # 筹码密集
        cond_pos = pos >= self.values["min_pos"]             # 现价处历史高位
        cond_profit = profit >= self.values["min_profit"]    # 获利盘沉重
        selected = bool(cond_conc and cond_pos and cond_profit)
        # 打分: 越密集、位置越高、获利盘越重 → 派发风险越大, 分越高
        score = (self.values["max_conc"] - conc) + pos * 0.5 + profit * 0.3
        reason = (f"集中度{conc:.0f} 位置{pos:.0f} 获利{profit:.0f}% "
                  f"⚠高位")
        return selected, float(score), reason


# 注册所有可用策略,UI 从这里读取。新增策略只要在这里加一行即可。
ALL_STRATEGIES = {
    "multi_factor": MultiFactor,
    "mean_reversion": MeanReversion,
    "potential": PotentialStock,
    "macd_cross": MacdGoldenCross,
    "breakout": BreakoutNewHigh,
    "pullback": PullbackMa,
    "vol_price_rise": VolumePriceRise,
    "oversold_rebound": OversoldRebound,
    "chip_low_lock": ChipLowLock,
    "chip_low_dense": ChipLowDense,
    "chip_twin_peak": ChipTwinPeak,
    "chip_high_dense": ChipHighDense,
}
