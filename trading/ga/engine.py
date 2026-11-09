"""
快速回测引擎（遗传算法专用）。

- 数据：freqtrade 下载的 OKX 永续合约 K 线（user_data/data/okx/futures/*.feather）
- 周期是基因的一部分：15m（直接读取）、1h（直接读取）、4h / 1d（由 1h 合成）；
  指标周期、最长持仓都按该周期的 K 线根数计算
- 每个币对独立运行、各分 1/N 资金，结果合并
- 信号在 K 线收盘时判断，下一根开盘价入场
- 止损 = 入场价 ∓ stop_atr × ATR；止盈 = 止损距离 × tp_r；超过 max_hold 根 K 线按收盘价离场
- 基因 exit="trail" 时不设止盈，止损随最高价（做空为最低价）上移，保持 stop_atr × ATR 的距离
- 同一根 K 线同时碰到止损和止盈时按止损算（保守）
- 仓位：每笔风险 = 账户 × risk；名义仓位 / 账户 = 杠杆，最高 10 倍
- 成本：手续费单边 0.05%，资金费按每 8 小时 0.01% 计（多空都按付出算，保守）

它比 freqtrade 简化很多（没有爆仓、盘口、滑点模型），只用于进化时快速打分；
选出来的策略上线前应再用 freqtrade 完整回测一次。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "user_data" / "data" / "okx" / "futures"
UNIVERSE = Path(__file__).resolve().parent / "universe.json"


def default_pairs() -> list[str]:
    """优先用 universe.json（OKX 成交额前 100），否则用最初的 5 个币。"""
    if UNIVERSE.exists():
        import json
        return [c["base"] for c in json.loads(UNIVERSE.read_text())["coins"]]
    return ["BTC", "ETH", "SOL", "BNB", "XRP"]

FEE = 0.0005          # 单边手续费
FUNDING_PER_H = 0.0001 / 8
MAX_LEV = 10.0
START_EQUITY = 1000.0

SPECIES = {
    "vol_breakout": "波动突破",
    "ma_cross": "均线趋势",
    "rsi_revert": "均值回归",
    "channel": "通道突破",
    "vol_surge": "放量异动",
    "rel_strength": "相对强弱",
}


# ------------------------------------------------------------------ 数据和指标
@dataclass
class PairData:
    name: str
    t: np.ndarray       # datetime64[ns]
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    atr: np.ndarray
    cache: dict
    bar_h: float = 1.0  # 每根 K 线多少小时
    v: np.ndarray | None = None      # 成交量
    ref_c: np.ndarray | None = None  # 同一时刻 BTC 收盘价（相对强弱用）


TF_HOURS = {"15m": 0.25, "1h": 1.0, "4h": 4.0, "1d": 24.0}
TF_NAME = {"15m": "15 分钟", "1h": "1 小时", "4h": "4 小时", "1d": "日线"}


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def _read(p: str, tf: str) -> pd.DataFrame | None:
    src = "15m" if tf == "15m" else "1h"
    f = DATA_DIR / f"{p}_USDT_USDT-{src}-futures.feather"
    if not f.exists():
        return None
    df = pd.read_feather(f).sort_values("date")
    if tf in ("4h", "1d"):
        rule = {"4h": "4h", "1d": "1D"}[tf]
        df = df.assign(date=df["date"].dt.tz_convert(None))   # 去掉时区再合成，按 UTC 0 点切分
        df = (df.set_index("date").resample(rule, origin="epoch")
              .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
              .dropna().reset_index())
        df["date"] = df["date"].dt.tz_localize("UTC")
    return df


def load_data(pairs: list[str] | None = None, tf: str = "1h") -> list[PairData]:
    out = []
    for p in pairs or default_pairs():
        df = _read(p, tf)
        if df is None or len(df) * TF_HOURS[tf] < 500:
            continue
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        prev_c = np.concatenate([[c[0]], c[:-1]])
        tr = np.maximum(h - l, np.maximum(abs(h - prev_c), abs(l - prev_c)))
        atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
        t = df["date"].dt.tz_convert(None).to_numpy("datetime64[ns]")
        out.append(PairData(p, t, o, h, l, c, atr, {}, TF_HOURS[tf], df["volume"].to_numpy(float)))
    btc = next((d for d in out if d.name == "BTC"), None)
    for d in out:   # 把 BTC 收盘价对齐到每个币的时间轴（只用当根及之前的 BTC 价格）
        if btc is not None:
            k = np.searchsorted(btc.t, d.t, side="right") - 1
            d.ref_c = np.where(k >= 0, btc.c[np.clip(k, 0, None)], np.nan)
    return out


def load_all(pairs: list[str] | None = None, tfs=("15m", "1h", "4h", "1d")) -> dict[str, list[PairData]]:
    """每个周期一组数据；没有下载的周期返回空列表。"""
    return {tf: load_data(pairs, tf) for tf in tfs}


def pick(data, g: dict) -> list[PairData]:
    """按基因里的周期取数据；data 既可以是单个周期的列表，也可以是 load_all 的字典。"""
    return data[g.get("tf", "1h")] if isinstance(data, dict) else data


def ema(pd_: PairData, n: int) -> np.ndarray:
    key = ("ema", n)
    if key not in pd_.cache:
        pd_.cache[key] = _ema(pd_.c, n)
    return pd_.cache[key]


def rsi(pd_: PairData, n: int = 14) -> np.ndarray:
    key = ("rsi", n)
    if key not in pd_.cache:
        d = np.diff(pd_.c, prepend=pd_.c[0])
        up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
        dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
        pd_.cache[key] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50).to_numpy()
    return pd_.cache[key]


def donchian(pd_: PairData, n: int) -> tuple[np.ndarray, np.ndarray]:
    key = ("dc", n)
    if key not in pd_.cache:
        hh = pd.Series(pd_.h).rolling(n).max().shift(1).to_numpy()
        ll = pd.Series(pd_.l).rolling(n).min().shift(1).to_numpy()
        pd_.cache[key] = (hh, ll)
    return pd_.cache[key]


# ------------------------------------------------------------------ 信号
def signals(g: dict, d: PairData) -> tuple[np.ndarray, np.ndarray]:
    """返回 (做多信号, 做空信号)，都只用到当根及之前的数据。"""
    c, o = d.c, d.o
    sp = g["species"]
    if sp == "vol_breakout":
        atr_prev = np.concatenate([[np.nan], d.atr[:-1]])
        big = (d.h - d.l) > g["k"] * atr_prev
        long_, short_ = big & (c > o), big & (c < o)
    elif sp == "ma_cross":
        f, s = _ema(d.c, g["fast"]), _ema(d.c, g["slow"])
        above = f > s
        prev = np.concatenate([[False], above[:-1]])
        long_, short_ = above & ~prev, ~above & prev
    elif sp == "rsi_revert":
        r = rsi(d)
        rp = np.concatenate([[50.0], r[:-1]])
        long_ = (r > g["lo"]) & (rp <= g["lo"])
        short_ = (r < g["hi"]) & (rp >= g["hi"])
    elif sp == "channel":
        n = g["n"]
        hh = pd.Series(d.h).rolling(n).max().shift(1).to_numpy()
        ll = pd.Series(d.l).rolling(n).min().shift(1).to_numpy()
        long_, short_ = c > hh, c < ll
    elif sp == "vol_surge":
        # 成交量是过去 n 根均量的 m 倍以上，顺这根 K 线的方向开仓
        avg = pd.Series(d.v).rolling(g["n"]).mean().shift(1).to_numpy()
        surge = d.v > g["m"] * avg
        long_, short_ = surge & (c > o), surge & (c < o)
    elif sp == "rel_strength":
        # n 根 K 线内本币涨幅减去 BTC 涨幅，上穿 +th 做多、下穿 -th 做空
        n = g["n"]
        if d.ref_c is None or d.name == "BTC" or len(c) <= n:
            z = np.zeros(len(c), bool)
            long_, short_ = z, z.copy()
        else:
            cp = np.concatenate([np.full(n, np.nan), c[:-n]])
            rp = np.concatenate([np.full(n, np.nan), d.ref_c[:-n]])
            rs = np.nan_to_num(c / cp - d.ref_c / rp)
            prev = np.concatenate([[0.0], rs[:-1]])
            long_ = (rs > g["th"]) & (prev <= g["th"])
            short_ = (rs < -g["th"]) & (prev >= -g["th"])
    else:
        raise ValueError(sp)
    if g["trend"] > 0:
        e = ema(d, g["trend"])
        long_ &= c > e
        short_ &= c < e
    if g["dir"] == "long":
        short_ = np.zeros_like(short_)
    elif g["dir"] == "short":
        long_ = np.zeros_like(long_)
    return long_, short_


# ------------------------------------------------------------------ 回测
def all_signals(g: dict, data) -> list:
    return [signals(g, d) for d in pick(data, g)]


def backtest(g: dict, data: list[PairData], t0: np.datetime64, t1: np.datetime64,
             want_trades: bool = False, fee: float = FEE, min_trades: int = 30, sigs: list | None = None) -> dict:
    data = pick(data, g)
    windows = []
    for idx, d in enumerate(data):
        i0 = int(np.searchsorted(d.t, t0))
        i1 = int(np.searchsorted(d.t, t1))
        if (i1 - i0) * d.bar_h >= 300:
            windows.append((idx, d, i0, i1))
    eq0 = START_EQUITY / max(1, len(windows))
    exits_t, exits_pnl, trades = [], [], []
    for idx, d, i0, i1 in windows:
        L, S = sigs[idx] if sigs is not None else signals(g, d)
        sig_idx = np.flatnonzero((L | S)[i0:i1 - 1]) + i0
        o, h, l, c, atr = d.o, d.h, d.l, d.c, d.atr
        E = eq0
        k = 0
        i = i0
        n_sig = len(sig_idx)
        while k < n_sig:
            s = sig_idx[k]
            if s < i:
                k += 1
                continue
            e_i = s + 1
            if e_i >= i1 or E <= 0:
                break
            side = 1 if L[s] else -1
            entry = o[e_i]
            dist = g["stop_atr"] * atr[s]
            if not (dist > 0) or not math.isfinite(dist):
                k += 1
                continue
            qty = E * g["risk"] / dist
            if qty * entry > E * MAX_LEV:
                qty = E * MAX_LEV / entry
            stop = entry - side * dist
            trail = g.get("exit", "tp") == "trail"          # trail：不设止盈，止损跟随最高/最低价移动
            tp = np.inf * side if trail else entry + side * dist * g["tp_r"]
            j = e_i
            last = min(i1 - 1, e_i + int(g["max_hold"]))
            exit_p = None
            while j <= last:
                if side == 1:
                    if l[j] <= stop:
                        exit_p = min(stop, o[j]) if j > e_i else stop
                        break
                    if h[j] >= tp:
                        exit_p = max(tp, o[j]) if j > e_i else tp
                        break
                else:
                    if h[j] >= stop:
                        exit_p = max(stop, o[j]) if j > e_i else stop
                        break
                    if l[j] <= tp:
                        exit_p = min(tp, o[j]) if j > e_i else tp
                        break
                if j == last:
                    exit_p = c[j]
                    break
                if trail:                                     # 收盘后再上移止损，不用本根之后的信息
                    stop = max(stop, h[j] - dist) if side == 1 else min(stop, l[j] + dist)
                j += 1
            hours = (j - e_i + 1) * d.bar_h
            notional = qty * entry
            pnl = side * qty * (exit_p - entry) - notional * fee - qty * exit_p * fee - notional * FUNDING_PER_H * hours
            pnl = max(pnl, -E)  # 逐仓：最多亏光这一份
            E += pnl
            exits_t.append(d.t[j])
            exits_pnl.append(pnl)
            if want_trades:
                trades.append((d.name, str(d.t[e_i])[:16], str(d.t[j])[:16], side, round(float(entry), 6),
                               round(float(exit_p), 6), round(float(pnl), 2), round(notional / max(E - pnl, 1e-9), 2),
                               float(qty), round(float(stop), 6), round(float(tp), 6)))
            i = j + 1
            k += 1
    return _metrics(exits_t, exits_pnl, t0, t1, trades if want_trades else None, min_trades)


def _metrics(exits_t, exits_pnl, t0, t1, trades, min_trades: int = 30) -> dict:
    n = len(exits_pnl)
    if n == 0:
        return {"trades": 0, "ret": 0.0, "dd": 0.0, "win": 0.0, "pf": 0.0, "fit": -1.0, "_t": [], "_eq": []}
    order = np.argsort(np.array(exits_t))
    t = np.array(exits_t)[order]
    pnl = np.array(exits_pnl)[order]
    eq = START_EQUITY + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[START_EQUITY], eq]))[1:]
    dd = float(np.max((peak - eq) / peak)) if n else 0.0
    ret = float(eq[-1] / START_EQUITY - 1)
    wins = pnl[pnl > 0].sum()
    loss = -pnl[pnl < 0].sum()
    days = max(1.0, (t1 - t0) / np.timedelta64(1, "D"))
    if ret > -1:
        ann = (1 + ret) ** (365 / days) - 1
    else:
        ann = -1.0
    fit = ann / max(dd, 0.05)
    if n < min_trades:
        fit = min(fit, 0) - (min_trades - n) / max(min_trades, 1)
    m = {"trades": n, "ret": round(ret * 100, 2), "dd": round(dd * 100, 2),
         "win": round(float((pnl > 0).mean() * 100), 1),
         "pf": round(float(wins / loss), 2) if loss > 0 else 99.0,
         "fit": round(float(fit), 3), "_t": t, "_eq": eq}
    if trades is not None:
        m["trade_list"] = trades
    return m


def daily_equity(m: dict, t0: str, t1: str) -> list[list]:
    """把逐笔平仓的净值折成每日净值（无交易的日子沿用前值）。"""
    days = pd.date_range(t0, t1, freq="D")
    if not len(m["_t"]):
        return [[d.strftime("%Y-%m-%d"), START_EQUITY] for d in days]
    s = pd.Series(m["_eq"], index=pd.to_datetime(m["_t"])).groupby(lambda x: x.floor("D")).last()
    s = s.reindex(days).ffill().fillna(START_EQUITY)
    return [[d.strftime("%Y-%m-%d"), round(float(v), 1)] for d, v in s.items()]
