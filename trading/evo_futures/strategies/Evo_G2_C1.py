"""
G2_C1（父代 Evo_G1_C1）：杠杆 3->2，加移动止盈（盈利 10% 后回撤 5% 离场），目标降回撤。
原 G1_C1：
改动：只在 EMA200 方向做单（价在 EMA200 上方只做多、下方只做空），并要求 ADX>阈值，
减少震荡市里来回打脸的交叉信号。
"""
from datetime import datetime

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IntParameter, IStrategy


class Evo_G2_C1(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    minimal_roi = {"0": 0.30, "720": 0.10, "2880": 0.03}
    stoploss = -0.15
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True
    process_only_new_candles = True
    startup_candle_count = 220

    ema_fast = IntParameter(5, 30, default=12, space="buy")
    ema_slow = IntParameter(30, 120, default=50, space="buy")
    adx_min = IntParameter(15, 35, default=25, space="buy")
    lev = IntParameter(1, 10, default=2, space="buy")

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        for p in set(self.ema_fast.range) | set(self.ema_slow.range):
            df[f"ema_{p}"] = ta.EMA(df, timeperiod=p)
        df["ema200"] = ta.EMA(df, timeperiod=200)
        df["adx"] = ta.ADX(df, timeperiod=14)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        f, s = df[f"ema_{self.ema_fast.value}"], df[f"ema_{self.ema_slow.value}"]
        up = (f > s) & (f.shift(1) <= s.shift(1))
        dn = (f < s) & (f.shift(1) >= s.shift(1))
        trend = df["adx"] > self.adx_min.value
        df.loc[up & trend & (df["close"] > df["ema200"]) & (df["volume"] > 0), "enter_long"] = 1
        df.loc[dn & trend & (df["close"] < df["ema200"]) & (df["volume"] > 0), "enter_short"] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        f, s = df[f"ema_{self.ema_fast.value}"], df[f"ema_{self.ema_slow.value}"]
        df.loc[(f < s) & (f.shift(1) >= s.shift(1)), "exit_long"] = 1
        df.loc[(f > s) & (f.shift(1) <= s.shift(1)), "exit_short"] = 1
        return df

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return float(min(self.lev.value, 10, max_leverage))
