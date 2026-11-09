"""
合约种子策略：EMA 金叉做多 / 死叉做空 + RSI 过滤，杠杆 1~10 倍（参数 lev 控制）。

注意：freqtrade 合约模式下 stoploss / ROI 都是按“含杠杆的收益率”计算的，
例如 stoploss=-0.10、5 倍杠杆 ≈ 价格反向波动 2% 就止损。
"""
from datetime import datetime

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IntParameter, IStrategy


class EvoFuturesSeed(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = True

    minimal_roi = {"0": 0.30, "720": 0.10, "2880": 0.03}
    stoploss = -0.15
    trailing_stop = False

    process_only_new_candles = True
    startup_candle_count = 200

    ema_fast = IntParameter(5, 30, default=12, space="buy")
    ema_slow = IntParameter(30, 120, default=50, space="buy")
    rsi_long_max = IntParameter(50, 75, default=65, space="buy")
    rsi_short_min = IntParameter(25, 50, default=35, space="sell")
    lev = IntParameter(1, 10, default=3, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for period in set(self.ema_fast.range) | set(self.ema_slow.range):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast = dataframe[f"ema_{self.ema_fast.value}"]
        slow = dataframe[f"ema_{self.ema_slow.value}"]
        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        vol = dataframe["volume"] > 0
        dataframe.loc[cross_up & (dataframe["rsi"] < self.rsi_long_max.value) & vol, "enter_long"] = 1
        dataframe.loc[cross_dn & (dataframe["rsi"] > self.rsi_short_min.value) & vol, "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast = dataframe[f"ema_{self.ema_fast.value}"]
        slow = dataframe[f"ema_{self.ema_slow.value}"]
        dataframe.loc[(fast < slow) & (fast.shift(1) >= slow.shift(1)), "exit_long"] = 1
        dataframe.loc[(fast > slow) & (fast.shift(1) <= slow.shift(1)), "exit_short"] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return float(min(self.lev.value, 10, max_leverage))
