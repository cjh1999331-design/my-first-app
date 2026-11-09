#!/usr/bin/env python3
"""
模拟盘回放：假设进化选出的策略在训练期结束那天（样本外起点）上线，
用 OKX 真实 1h K 线一直跑到最新数据。没有真实下单；页面上标为“回放”。

输出账户权益的 K 线：先按小时计算权益（已实现盈亏 + 未平仓浮动盈亏，按每小时收盘价），
再合成 4 小时 OHLC。页面上的日线、周线由 4 小时 K 线再合成。

  python ga/paper_replay.py [--results ga/results.json] [--out ga/replay.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import START_EQUITY, backtest, load_data

HERE = Path(__file__).parent
HOUR = np.timedelta64(1, "h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results.json"))
    ap.add_argument("--out", default=str(HERE / "replay.json"))
    args = ap.parse_args()
    r = json.loads(Path(args.results).read_text())
    I = {i["id"]: i for i in r["individuals"]}
    data = load_data()
    t0 = np.datetime64(r["meta"]["oos"][0])
    t_last = min(d.t[-1] for d in data[:5]) + HOUR           # 以主流币的最新 K 线为准
    t_end = t0 + ((t_last - t0) // np.timedelta64(4, "h")) * np.timedelta64(4, "h")
    hours = np.arange(t0, t_end, HOUR)
    nh = len(hours)

    # 每个币每小时收盘价（上线前用首个价格填充，不会有仓位）
    closes = {}
    for d in data:
        s = pd.Series(d.c, index=pd.to_datetime(d.t)).reindex(pd.to_datetime(hours))
        closes[d.name] = s.ffill().bfill().to_numpy()

    ids = list(dict.fromkeys([r["champion"], *r["species_champions"]]))
    strategies = []
    for sid in ids:
        ind = I[sid]
        m = backtest(ind["genes"], data, t0, t_end, want_trades=True)
        realized = np.zeros(nh)
        unreal = np.zeros(nh)
        open_pos, closed = [], []
        per = {}   # 每个币：已实现盈亏、交易数、盈利笔数、持仓
        def coin(p):
            return per.setdefault(p, {"realized": 0.0, "n": 0, "wins": 0, "pos": None})
        for (pair, te, tx, side, entry, exit_p, pnl, lev, qty, stop, tp) in m["trade_list"]:
            hs = int(np.searchsorted(hours, np.datetime64(te)))        # 入场所在小时
            hx = int(np.searchsorted(hours, np.datetime64(tx)))        # 离场所在小时
            still_open = np.datetime64(tx) + HOUR >= t_end              # 数据末尾被强制平仓 = 实际仍持仓
            end = nh if still_open else hx
            if hs < end:
                unreal[hs:end] += side * qty * (closes[pair][hs:end] - entry)
            cs = coin(pair)
            if still_open:
                last = closes[pair][-1]
                fl = side * qty * (last - entry)
                notional = qty * last
                open_pos.append([pair, side, te[:16], entry, round(lev, 2), round(fl, 2),
                                 round(side * (last / entry - 1) * lev * 100, 2)])
                cs["pos"] = {"side": side, "notional": round(notional, 2), "margin": round(qty * entry / max(lev, 1e-9), 2),
                             "entry": entry, "last": float(f"{last:.6g}"), "float": round(fl, 2), "lev": round(lev, 2), "since": te[:16]}
            else:
                realized[hx:] += pnl
                closed.append([pair, side, tx[:16], round(pnl, 2)])
                cs["realized"] += pnl
                cs["n"] += 1
                cs["wins"] += pnl > 0
        eq = START_EQUITY + realized + unreal
        # 合成 4 小时 OHLC：开盘 = 上一根收盘
        e4 = eq.reshape(-1, 4)
        prev = np.concatenate([[START_EQUITY], e4[:-1, -1]])
        ohlc = np.stack([prev, np.maximum(prev, e4.max(1)), np.minimum(prev, e4.min(1)), e4[:, -1]], 1)
        open_pos.sort(key=lambda x: -abs(x[5]))
        strategies.append({"id": sid, "species": ind["species"], "desc": ind["desc"],
                           "ohlc": np.round(ohlc, 2).tolist(), "n_trades": len(m["trade_list"]),
                           "open": open_pos, "recent": closed[-10:][::-1],
                           "coins": sorted([{"pair": k, "realized": round(v["realized"], 2), "n": v["n"], "wins": v["wins"], "pos": v["pos"],
                                             "total": round(v["realized"] + (v["pos"]["float"] if v["pos"] else 0), 2)}
                                            for k, v in per.items()], key=lambda x: -x["total"])})
        print(f"{sid}: {len(m['trade_list'])} 笔, 持仓 {len(open_pos)}, 最终权益 {eq[-1]:.1f}")

    fmt = lambda x: pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M")
    out = {"start": fmt(t0), "end": fmt(t_end), "bar_hours": 4, "start_equity": START_EQUITY, "strategies": strategies}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(strategies[0]['ohlc'])} 根 4h 权益 K 线 ({out['start']} ~ {out['end']}) -> {args.out}")


if __name__ == "__main__":
    main()
