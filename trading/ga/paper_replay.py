#!/usr/bin/env python3
"""
模拟盘回放：假设进化选出的策略在训练期结束那天（样本外起点）上线，
用 OKX 真实 1h K 线逐根跑到最新数据，记录每笔交易和逐 4 小时的账户权益（含未平仓浮盈亏）。

没有真实下单；页面上标为“回放”。

  python ga/paper_replay.py [--results ga/results.json] [--out ga/replay.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import START_EQUITY, backtest, load_data

HERE = Path(__file__).parent
BAR = np.timedelta64(4, "h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results.json"))
    ap.add_argument("--out", default=str(HERE / "replay.json"))
    args = ap.parse_args()
    r = json.loads(Path(args.results).read_text())
    I = {i["id"]: i for i in r["individuals"]}
    data = load_data()
    t0 = np.datetime64(r["meta"]["oos"][0])
    t_last = min(d.t[-1] for d in data) + np.timedelta64(1, "h")
    t_end = t0 + ((t_last - t0) // BAR) * BAR
    grid = np.arange(t0, t_end, BAR)            # 每根 4h K 线的开盘时间
    ends = grid + BAR                            # 收盘时间
    n = len(grid)

    # 4h K 线（权益按全期计算；页面只放最近 CN 根，紧凑编码）
    CN = min(n, 90 * 6)
    candles, closes = {}, {}
    for d in data:
        df = pd.DataFrame({"o": d.o, "h": d.h, "l": d.l, "c": d.c}, index=pd.to_datetime(d.t))
        df = df[(df.index >= pd.Timestamp(t0)) & (df.index < pd.Timestamp(t_end))]
        agg = df.resample("4h", origin=pd.Timestamp(t0)).agg({"o": "first", "h": "max", "l": "min", "c": "last"})
        agg = agg.reindex(pd.to_datetime(grid))
        closes[d.name] = agg["c"].ffill().bfill().to_numpy()
        tail = agg.iloc[-CN:]
        if tail["c"].notna().sum() < 30:
            continue
        first = int(tail["c"].notna().to_numpy().argmax())     # 新币上线前的 K 线不导出
        tail = tail.iloc[first:].ffill()
        px = float(tail["c"].iloc[-1])
        e = 4 - int(np.floor(np.log10(px)))                     # 保留约 5 位有效数字
        ints = np.round(tail[["o", "h", "l", "c"]].to_numpy() * 10 ** e).astype(np.int64).ravel().tolist()
        candles[d.name] = {"e": e, "off": CN - len(tail), "d": ints}

    ids = list(dict.fromkeys([r["champion"], *r["species_champions"]]))
    strategies = []
    for sid in ids:
        ind = I[sid]
        m = backtest(ind["genes"], data, t0, t_end, want_trades=True)
        trades = []
        realized = np.zeros(n)
        unreal = np.zeros(n)
        for (pair, te, tx, side, entry, exit_p, pnl, lev, qty, stop, tp) in m["trade_list"]:
            te64 = np.datetime64(te)
            tx64 = np.datetime64(tx) + np.timedelta64(1, "h")    # 在这根 1h K 线内平仓
            ks = int(np.searchsorted(ends, te64, side="right"))   # 第一根收盘晚于入场的 4h K 线
            kx = int(np.searchsorted(ends, tx64, side="left"))    # 平仓所在的 4h K 线
            if tx64 >= t_end:                                     # 回测到数据末尾被强制平仓：其实仍在持仓
                kx = n
            if kx < n:
                realized[kx:] += pnl
            hi = min(kx, n)
            if ks < hi:
                unreal[ks:hi] += side * qty * (closes[pair][ks:hi] - entry)
            trades.append({"pair": pair, "side": side, "te": te, "tx": tx, "entry": entry, "exit": exit_p,
                           "pnl": pnl, "lev": lev, "stop": stop, "tp": tp,
                           "ki": int(np.searchsorted(ends, te64, side="right")), "kx": kx})
        equity = START_EQUITY + realized + unreal
        strategies.append({"id": sid, "species": ind["species"], "desc": ind["desc"],
                           "equity": [round(float(v), 2) for v in equity], "trades": trades})
        print(f"{sid}: {len(trades)} 笔, 最终权益 {equity[-1]:.1f}")

    fmt = lambda x: pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M")
    out = {"start": fmt(t0), "bar_hours": 4, "n": n, "cn": CN, "end": fmt(t_end),
           "pairs": list(candles), "candles": candles, "strategies": strategies, "start_equity": START_EQUITY}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"{n} 根 4h K 线 ({out['start']} ~ {out['end']}) -> {args.out}")


if __name__ == "__main__":
    main()
