# pragma pylint: disable=missing-docstring, invalid-name
"""
种子策略：EMA 金叉 + RSI 过滤。

- 作为“进化”的起点（evolve/evolve.py 会让大模型在它基础上改写）
- 参数都声明成 IntParameter/DecimalParameter，可以直接用 freqtrade hyperopt 调参
"""
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


class EvoSeedStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False

    # 止盈表：持仓 0 分钟后 10% 止盈，6 小时后 4%，24 小时后 1%
    minimal_roi = {"0": 0.10, "360": 0.04, "1440": 0.01}
    stoploss = -0.08

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    startup_candle_count = 200

    # ---- 可优化参数（hyperopt 会搜索这些区间）----
    ema_fast = IntParameter(5, 30, default=12, space="buy")
    ema_slow = IntParameter(30, 120, default=50, space="buy")
    rsi_buy_max = IntParameter(40, 75, default=65, space="buy")
    rsi_sell_min = IntParameter(60, 90, default=75, space="sell")
    atr_mult = DecimalParameter(0.0, 3.0, default=0.0, decimals=1, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 预先算好参数区间内所有 EMA，hyperopt 时就不用反复计算
        for period in set(self.ema_fast.range) | set(self.ema_slow.range):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast = dataframe[f"ema_{self.ema_fast.value}"]
        slow = dataframe[f"ema_{self.ema_slow.value}"]
        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))

        # atr_mult = 0 时不过滤；>0 时要求波动率高于其长期均值的一定倍数
        vol_ok = dataframe["atr_pct"] >= (
            dataframe["atr_pct"].rolling(100).mean() * float(self.atr_mult.value) / 2
        )

        dataframe.loc[
            cross_up & (dataframe["rsi"] < self.rsi_buy_max.value) & vol_ok
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast = dataframe[f"ema_{self.ema_fast.value}"]
        slow = dataframe[f"ema_{self.ema_slow.value}"]
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        dataframe.loc[
            (cross_down | (dataframe["rsi"] > self.rsi_sell_min.value))
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
