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

from engine import START_EQUITY, TF_HOURS, TF_NAME, load_all
from portfolio import PortfolioRules, run_portfolio

HERE = Path(__file__).parent
HOUR = np.timedelta64(1, "h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results.json"))
    ap.add_argument("--out", default=str(HERE / "replay.json"))
    ap.add_argument("--shuffles", type=int, default=20, help="随机开仓顺序重跑的次数")
    args = ap.parse_args()
    r = json.loads(Path(args.results).read_text())
    I = {i["id"]: i for i in r["individuals"]}
    data = load_all(tfs=r["meta"].get("tfs", ["1h"]))
    t0 = np.datetime64(r["meta"]["oos"][0])
    t_last = min(d.t[-1] for d in data["1h"][:5]) + HOUR if data.get("1h") else min(d.t[-1] for v in data.values() for d in v[:5])
    t_end = t0 + ((t_last - t0) // np.timedelta64(4, "h")) * np.timedelta64(4, "h")
    # 一个账户：4 个物种冠军共用同一个资金池（规则见 portfolio.py），训练期得分高的策略优先
    ids = sorted(dict.fromkeys(r["species_champions"]), key=lambda x: -I[x]["train"]["fit"])
    rules = PortfolioRules(start_equity=START_EQUITY)
    grid, eq_total, step_h, open_pos, closed_raw = run_portfolio([(i, I[i]["genes"]) for i in ids], data, t0, t_end, rules)
    tstr = lambda t: str(t)[:16]
    sig = lambda v: float(f"{float(v):.6g}")
    positions = []
    for p in open_pos:
        lc = p["lc"]
        positions.append({"pair": p["name"], "side": int(p["side"]), "sid": ids[p["si"]], "since": tstr(p["t_entry"]),
                          "entry": sig(p["entry"]), "last": sig(lc), "lev": rules.lev, "notional": round(p["qty"] * lc, 2),
                          "margin": round(p["margin"], 2), "float": round(p["side"] * p["qty"] * (lc - p["entry"]), 2),
                          "stop": sig(p["stop"]), "tp": sig(p["tp"]) if np.isfinite(p["tp"]) else None})
    positions.sort(key=lambda p: -p["notional"])
    closed = [{"pair": t["name"], "side": int(t["side"]), "sid": ids[t["si"]], "te": tstr(t["t_entry"]), "tx": tstr(t["t_exit"]),
               "entry": sig(t["entry"]), "exit": sig(t["exit"]), "lev": rules.lev, "notional": round(t["qty"] * t["entry"], 2),
               "pnl": round(float(t["pnl"]), 2), "reason": t.get("reason", "")} for t in closed_raw]
    closed.sort(key=lambda t: t["tx"], reverse=True)
    strategies = []
    for k, sid in enumerate(ids):
        mine = [t for t in closed if t["sid"] == sid]
        op = [p for p in positions if p["sid"] == sid]
        tf = I[sid]["genes"].get("tf", "1h")
        strategies.append({"id": sid, "species": I[sid]["species"], "desc": I[sid]["desc"], "tf": tf, "tf_name": TF_NAME[tf], "n_closed": len(mine), "n_open": len(op),
                           "pnl": round(sum(t["pnl"] for t in mine) + sum(p["float"] for p in op), 2)})
    per4h = int(round(4 / step_h))
    e4 = eq_total[: len(eq_total) // per4h * per4h].reshape(-1, per4h)
    prev = np.concatenate([[START_EQUITY], e4[:-1, -1]])
    ohlc = np.stack([prev, np.maximum(prev, e4.max(1)), np.minimum(prev, e4.min(1)), e4[:, -1]], 1)
    stats = {"n": len(closed), "wins": int(sum(t["pnl"] > 0 for t in closed)), "realized": round(sum(t["pnl"] for t in closed), 2)}
    rules_d = {"risk": rules.risk, "risk_gene": rules.use_gene_risk, "min_risk": rules.min_risk, "lev": rules.lev, "max_pos": rules.per_strategy * len(ids), "per_strategy": rules.per_strategy, "max_notional": rules.max_notional,
               "daily_stop": rules.daily_stop, "stops": [str(x)[:16] for x in run_portfolio.stops]}
    print(f"账户: {START_EQUITY} -> {eq_total[-1]:.2f}, 持仓 {len(positions)}, 平仓 {len(closed)}")
    for st in strategies:
        print("  ", st["id"], st["n_closed"], st["n_open"], st["pnl"])

    # 对照：同样的策略换成旧的保守规则（每笔 0.2%、5x、账户最多 25 个持仓、不停手）
    old = PortfolioRules(start_equity=START_EQUITY, use_gene_risk=False, risk=0.002, lev=5.0, max_pos=25, per_strategy=999, daily_stop=0.0)
    _, eq_old, _, _, _ = run_portfolio([(i, I[i]["genes"]) for i in ids], data, t0, t_end, old)
    pk = np.maximum.accumulate(np.concatenate([[START_EQUITY], eq_old]))[1:]
    compare = {"label": "旧规则：每笔 0.2%、5x、最多 25 个持仓、不停手", "final": round(float(eq_old[-1]), 2),
               "dd": round(float(np.max((pk - eq_old) / pk)) * 100, 1)}
    print(f"旧规则对照: {compare}")

    # 稳健性：随机打乱同一时刻的开仓顺序，看最终权益和回撤的分布
    finals, dds = [], []
    for seed in range(args.shuffles):
        _, eq_s, _, _, _ = run_portfolio([(i, I[i]["genes"]) for i in ids], data, t0, t_end,
                                         PortfolioRules(start_equity=START_EQUITY, order_seed=seed))
        peak = np.maximum.accumulate(np.concatenate([[START_EQUITY], eq_s]))[1:]
        finals.append(round(float(eq_s[-1]), 2))
        dds.append(round(float(np.max((peak - eq_s) / peak)) * 100, 1))
    if finals:
        print(f"随机顺序 {len(finals)} 次：最终权益 中位 {np.median(finals):.1f}，范围 {min(finals):.1f} ~ {max(finals):.1f}")

    fmt = lambda x: pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M")
    out = {"start": fmt(t0), "end": fmt(t_end), "bar_hours": 4, "start_equity": START_EQUITY, "strategies": strategies,
           "ohlc": np.round(ohlc, 2).tolist(), "positions": positions, "closed": closed[:300], "closed_stats": stats, "rules": rules_d,
           "shuffle": {"finals": finals, "dds": dds}, "compare": compare}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(ohlc)} 根 4h 权益 K 线 ({out['start']} ~ {out['end']}) -> {args.out}")


if __name__ == "__main__":
    main()
