"""
一个量化账户：多个策略共用同一个资金池，按小时逐根撮合。

规则（可在 PortfolioRules 里改）：
- 每笔风险：止损时最多亏当前账户权益的 risk（默认 0.2%），由此反推仓位大小；
  25 个持仓同时止损最多亏约 5%（组合总风险上限）
- 逐仓保证金：统一 lev 倍（默认 5x），所有持仓保证金合计不超过账户权益的 95%
- 每个币同时只有一个持仓；最多同时 max_pos 个持仓；单笔名义仓位不超过账户权益 × max_notional
- 同一小时多个信号时，按策略优先级（训练期得分高的在前）依次处理
- 信号在 K 线收盘判断，下一根开盘入场；止损、止盈、最长持仓时间沿用策略基因
- 同一根 K 线同时碰到止损和止盈按止损算；开仓扣手续费，平仓扣手续费 + 资金费
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine import FEE, FUNDING_PER_H, PairData, signals

HOUR = np.timedelta64(1, "h")


@dataclass
class PortfolioRules:
    start_equity: float = 1000.0
    risk: float = 0.002
    lev: float = 5.0
    max_pos: int = 25
    max_notional: float = 1.0
    margin_cap: float = 0.95
    min_notional: float = 5.0


def run_portfolio(strats: list[tuple[str, dict]], data: list[PairData], t0, t_end, rules: PortfolioRules = PortfolioRules()):
    """strats: [(策略 id, 基因)]，按优先级排序。返回每小时权益、未平仓持仓、已平仓交易。"""
    hours = np.arange(t0, t_end, HOUR)
    nh = len(hours)
    # 每个币：小时网格 -> 该币 K 线序号（没有数据为 -1）
    hidx = []
    for d in data:
        j = np.searchsorted(d.t, hours)
        j = np.minimum(j, len(d.t) - 1)
        hidx.append(np.where(d.t[j] == hours, j, -1))
    # 入场事件：events[h] = [(优先级, 币, 策略序号, 方向, 信号所在 K 线)]
    events: list[list] = [[] for _ in range(nh)]
    for si, (_, g) in enumerate(strats):
        for ci, d in enumerate(data):
            L, S = signals(g, d)
            idx = np.flatnonzero(L | S)
            idx = idx[idx + 1 < len(d.t)]
            te = d.t[idx + 1]
            ok = (te >= t0) & (te < t_end)
            for i, t in zip(idx[ok], te[ok]):
                h = int((t - t0) // HOUR)
                events[h].append((si, ci, si, 1 if L[i] else -1, int(i)))
    for ev in events:
        ev.sort()

    cash = rules.start_equity
    eq = np.zeros(nh)
    open_pos: dict[int, dict] = {}       # 币序号 -> 持仓
    closed = []
    last_close = [np.nan] * len(data)
    floating = 0.0
    for h in range(nh):
        equity_now = cash + floating
        # 1) 入场（下一根开盘价）
        for (_, ci, si, side, i) in events[h]:
            if ci in open_pos or len(open_pos) >= rules.max_pos:
                continue
            d = data[ci]
            j = hidx[ci][h]
            if j < 0 or j != i + 1:
                continue
            g = strats[si][1]
            entry = d.o[j]
            dist = g["stop_atr"] * d.atr[i]
            if not (dist > 0) or not np.isfinite(dist) or equity_now <= 0:
                continue
            notional = equity_now * rules.risk / dist * entry
            notional = min(notional, equity_now * rules.max_notional)
            used = sum(p["margin"] for p in open_pos.values())
            free = equity_now * rules.margin_cap - used
            notional = min(notional, free * rules.lev)
            if notional < rules.min_notional:
                continue
            qty = notional / entry
            cash -= notional * FEE
            open_pos[ci] = {"ci": ci, "si": si, "side": side, "entry": entry, "qty": qty, "margin": notional / rules.lev,
                            "stop": entry - side * dist, "tp": entry + side * dist * g["tp_r"], "e_j": j,
                            "max_hold": int(g["max_hold"]), "t_entry": d.t[j]}
        # 2) 本根 K 线内的止损 / 止盈 / 到期
        for ci in list(open_pos):
            p = open_pos[ci]
            d = data[ci]
            j = hidx[ci][h]
            if j < 0:
                continue
            o, hi, lo, c = d.o[j], d.h[j], d.l[j], d.c[j]
            first = j == p["e_j"]
            exit_p = None
            if p["side"] == 1:
                if lo <= p["stop"]:
                    exit_p = p["stop"] if first else min(p["stop"], o)
                elif hi >= p["tp"]:
                    exit_p = p["tp"] if first else max(p["tp"], o)
            else:
                if hi >= p["stop"]:
                    exit_p = p["stop"] if first else max(p["stop"], o)
                elif lo <= p["tp"]:
                    exit_p = p["tp"] if first else min(p["tp"], o)
            if exit_p is None and j - p["e_j"] >= p["max_hold"]:
                exit_p = c
            last_close[ci] = c
            if exit_p is not None:
                held = j - p["e_j"] + 1
                gross = p["side"] * p["qty"] * (exit_p - p["entry"])
                cost = p["qty"] * exit_p * FEE + p["qty"] * p["entry"] * FUNDING_PER_H * held
                pnl = max(gross - cost, -p["margin"])            # 逐仓：最多亏掉这笔保证金
                cash += pnl
                closed.append({**p, "exit": exit_p, "t_exit": d.t[j], "pnl": pnl - p["qty"] * p["entry"] * FEE})
                del open_pos[ci]
        # 3) 收盘权益（含浮动盈亏）
        floating = 0.0
        for ci, p in open_pos.items():
            lc = last_close[ci] if np.isfinite(last_close[ci]) else p["entry"]
            floating += p["side"] * p["qty"] * (lc - p["entry"])
        eq[h] = cash + floating
    return hours, eq, list(open_pos.values()), closed, last_close
