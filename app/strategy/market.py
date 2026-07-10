"""
策略层 - 大盘趋势过滤
========================================
核心思想:个股再好,大盘系统性下跌时也难独善其身(A股尤甚)。
所以给所有策略加一层"择时开关":只在大盘走强时才允许开仓,
大盘走弱时空仓观望。这能显著砍掉最大回撤、抬升胜率。

判定口径(默认上证综指 sh000001):
  大盘"走强" = 指数收盘价 > 指数 MA(默认20日) 且 MA 本身向上(近5日抬升)
  两个条件都满足才算多头环境,否则视为弱势、不开仓。

无未来函数:回测里查询某一天的大盘状态时,只用截止当天的指数数据。
用 date->状态 的预计算字典,O(1) 查询,回测里逐日调用也不慢。
"""
import pandas as pd

from ..data import database as db


class MarketTrend:
    """
    大盘趋势判定器。构造时一次性把指数日线算好,
    之后用 is_strong(date) 查询任意交易日大盘是否走强。
    """

    def __init__(self, index_code: str = "sh000001", ma: int = 20,
                 slope_days: int = 5):
        self.index_code = index_code
        self.ma = ma
        self.slope_days = slope_days
        self._status = {}      # date(str) -> bool 是否走强
        self._ready = False
        self._build()

    def _build(self):
        df = db.load_index_kline(self.index_code)
        if df is None or df.empty or len(df) < self.ma + self.slope_days + 1:
            self._ready = False
            return
        df = df.copy()
        c = pd.to_numeric(df["close"], errors="coerce")
        maN = c.rolling(self.ma).mean()
        # MA 向上:当前 MA > slope_days 日前的 MA
        ma_prev = maN.shift(self.slope_days)
        strong = (c > maN) & (maN > ma_prev)
        self._status = dict(zip(df["date"].tolist(), strong.tolist()))
        self._ready = True

    @property
    def ready(self) -> bool:
        """指数数据是否足够,能否做判定。"""
        return self._ready

    def is_strong(self, date: str) -> bool:
        """
        查询某交易日大盘是否走强。
        - 指数没数据(未拉取)时:返回 True(不过滤,退化为原行为,避免误伤)。
        - 指数当天停市/无该日:用最近的一个已知交易日状态兜底。
        """
        if not self._ready:
            return True
        v = self._status.get(date)
        if v is not None:
            return bool(v)
        # 该日期无指数数据:找 <= date 的最近交易日
        prior = [d for d in self._status if d <= date]
        if not prior:
            return True
        return bool(self._status[max(prior)])

    def latest_state(self):
        """返回 (最新日期, 是否走强),供界面显示当前大盘环境。"""
        if not self._ready or not self._status:
            return None, None
        last = max(self._status)
        return last, bool(self._status[last])
