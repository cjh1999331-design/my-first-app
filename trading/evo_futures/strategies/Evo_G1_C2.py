"""
G1_C2（父代 EvoFuturesSeed）：换成唐奇安通道突破（经典趋势跟踪）。
改动：突破过去 N 根 K 线最高价做多 / 跌破最低价做空，且需与 EMA200 方向一致；
出场用更短的反向通道（海龟法则），让盈利单跑得更远。
"""
from datetime import datetime

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IntParameter, IStrategy


class Evo_G1_C2(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    minimal_roi = {"0": 10.0}  # 不用 ROI，交给通道出场
    stoploss = -0.20
    process_only_new_candles = True
    startup_candle_count = 220

    entry_len = IntParameter(24, 120, default=48, space="buy")
    exit_len = IntParameter(12, 60, default=24, space="sell")
    lev = IntParameter(1, 10, default=3, space="buy")

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["ema200"] = ta.EMA(df, timeperiod=200)
        for n in set(self.entry_len.range) | set(self.exit_len.range):
            df[f"hh_{n}"] = df["high"].rolling(n).max().shift(1)
            df[f"ll_{n}"] = df["low"].rolling(n).min().shift(1)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        n = self.entry_len.value
        v = df["volume"] > 0
        df.loc[(df["close"] > df[f"hh_{n}"]) & (df["close"] > df["ema200"]) & v, "enter_long"] = 1
        df.loc[(df["close"] < df[f"ll_{n}"]) & (df["close"] < df["ema200"]) & v, "enter_short"] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        n = self.exit_len.value
        df.loc[df["close"] < df[f"ll_{n}"], "exit_long"] = 1
        df.loc[df["close"] > df[f"hh_{n}"], "exit_short"] = 1
        return df

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return float(min(self.lev.value, 10, max_leverage))
