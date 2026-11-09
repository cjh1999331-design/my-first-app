"""
G1_C3（父代 EvoFuturesSeed）：趋势中回调入场。
改动：价格在 EMA200 上方时，等 RSI 从超卖区回升再做多；下方时 RSI 从超买区回落再做空。
用 RSI 回到中性区出场，持仓时间短、单笔风险小。
"""
from datetime import datetime

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IntParameter, IStrategy


class Evo_G1_C3(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    minimal_roi = {"0": 0.15, "240": 0.06, "1440": 0.02}
    stoploss = -0.12
    process_only_new_candles = True
    startup_candle_count = 220

    rsi_low = IntParameter(20, 40, default=30, space="buy")
    rsi_high = IntParameter(60, 80, default=70, space="sell")
    lev = IntParameter(1, 10, default=3, space="buy")

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["ema200"] = ta.EMA(df, timeperiod=200)
        df["rsi"] = ta.RSI(df, timeperiod=14)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        r, v = df["rsi"], df["volume"] > 0
        lo, hi = self.rsi_low.value, self.rsi_high.value
        df.loc[(df["close"] > df["ema200"]) & (r > lo) & (r.shift(1) <= lo) & v, "enter_long"] = 1
        df.loc[(df["close"] < df["ema200"]) & (r < hi) & (r.shift(1) >= hi) & v, "enter_short"] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[df["rsi"] > 60, "exit_long"] = 1
        df.loc[df["rsi"] < 40, "exit_short"] = 1
        return df

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return float(min(self.lev.value, 10, max_leverage))
