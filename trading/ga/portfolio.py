"""
一个量化账户：多个策略（可以是不同周期）共用同一个资金池，按最小周期逐步撮合。

规则（可在 PortfolioRules 里改）：
- 每笔风险：止损时最多亏当前账户权益的 risk（默认 1%），由此反推仓位大小
- 逐仓保证金：统一 lev 倍（默认 10x），所有持仓保证金合计不超过账户权益的 95%
- 日亏停手：账户权益比当天（UTC 0 点）开盘时亏损达到 daily_stop（默认 10%），立即平掉全部持仓，
  当天不再开仓，第二天 0 点恢复
- 盈利不设上限：基因 exit="trail" 的策略没有止盈，止损随价格移动
- 每个策略最多同时 per_strategy 个持仓（默认 5）；一个币已经有持仓（不管哪个策略开的）就不重复开；
  单笔名义仓位不超过账户权益 × max_notional
- 同一时刻多个信号时，按策略优先级（训练期得分高的在前）依次处理
- 每个策略在自己周期的 K 线收盘判断信号，下一根开盘入场；在自己周期的 K 线收盘时检查止损/止盈/到期
- 同一根 K 线同时碰到止损和止盈按止损算；开仓扣手续费，平仓扣手续费 + 资金费
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine import FEE, FUNDING_PER_H, TF_HOURS, pick, signals


@dataclass
class PortfolioRules:
    start_equity: float = 1000.0
    risk: float = 0.01
    lev: float = 10.0
    max_pos: int = 999            # 账户总持仓上限（默认不限，由每个策略的上限决定）
    per_strategy: int = 5
    daily_stop: float = 0.10
    max_notional: float = 1.0
    margin_cap: float = 0.95
    min_notional: float = 5.0
    order_seed: int | None = None   # None：同一时刻按成交额排名开仓；整数：随机打乱币的优先顺序（检验结果对顺序是否敏感）


def run_portfolio(strats: list[tuple[str, dict]], data, t0, t_end, rules: PortfolioRules = PortfolioRules()):
    """strats: [(策略 id, 基因)]，按优先级排序；data: load_all() 的字典或单周期列表。
    返回 (时间网格, 每步权益, 网格步长小时数, 未平仓持仓, 已平仓交易)。"""
    step_h = min(TF_HOURS[g.get("tf", "1h")] for _, g in strats)
    step = np.timedelta64(int(step_h * 60), "m")
    grid = np.arange(t0, t_end, step)
    ng = len(grid)

    series = {}   # (周期, 币) -> (数据, 开盘映射, 收盘映射)
    def get_series(d):
        key = (d.bar_h, d.name)
        if key not in series:
            bar = np.timedelta64(int(d.bar_h * 60), "m")
            ok = (d.t >= t0) & (d.t < t_end)
            idx = np.flatnonzero(ok)
            open_map = np.full(ng, -1)
            open_map[((d.t[idx] - t0) // step).astype(int)] = idx
            close_g = ((d.t[idx] + bar - step - t0) // step).astype(int)
            keep = close_g < ng
            close_map = np.full(ng, -1)
            close_map[close_g[keep]] = idx[keep]
            series[key] = (d, open_map, close_map)
        return key

    rank = {}
    if rules.order_seed is not None:
        rng = np.random.default_rng(rules.order_seed)
    events: list[list] = [[] for _ in range(ng)]
    for si, (_, g) in enumerate(strats):
        for ci, d in enumerate(pick(data, g)):     # ci：币在成交额排名里的顺序，同一时刻排名靠前的先开
            if rules.order_seed is not None:
                ci = rank.setdefault(d.name, float(rng.random()))
            key = get_series(d)
            L, S = signals(g, d)
            idx = np.flatnonzero(L | S)
            idx = idx[idx + 1 < len(d.t)]
            te = d.t[idx + 1]
            ok = (te >= t0) & (te < t_end)
            for i, t in zip(idx[ok], te[ok]):
                events[int((t - t0) // step)].append((si, ci, d.name, si, 1 if L[i] else -1, int(i), key))
    for ev in events:
        ev.sort(key=lambda e: (e[0], e[1]))
    active_keys = list(series)

    cash = rules.start_equity
    eq = np.zeros(ng)
    open_pos: dict[str, dict] = {}     # 币名 -> 持仓
    closed = []
    floating = 0.0
    day = None
    day_start = rules.start_equity
    paused = False
    stops = []                        # 触发日亏停手的时间
    for gi in range(ng):
        equity_now = cash + floating
        d_now = grid[gi].astype("datetime64[D]")
        if d_now != day:                              # 新的一天：记录开盘权益，解除停手
            day, day_start, paused = d_now, equity_now, False
        # 1) 入场（下一根开盘价）
        for (_, _ci, name, si, side, i, key) in events[gi]:
            if paused or name in open_pos or len(open_pos) >= rules.max_pos or equity_now <= 0:
                continue
            if sum(1 for p in open_pos.values() if p["si"] == si) >= rules.per_strategy:
                continue
            d, open_map, _ = series[key]
            j = open_map[gi]
            if j != i + 1:
                continue
            g = strats[si][1]
            entry = d.o[j]
            dist = g["stop_atr"] * d.atr[i]
            if not (dist > 0) or not np.isfinite(dist):
                continue
            notional = equity_now * rules.risk / dist * entry
            notional = min(notional, equity_now * rules.max_notional)
            used = sum(p["margin"] for p in open_pos.values())
            notional = min(notional, (equity_now * rules.margin_cap - used) * rules.lev)
            if notional < rules.min_notional:
                continue
            qty = notional / entry
            cash -= notional * FEE
            trail = g.get("exit", "tp") == "trail"
            open_pos[name] = {"name": name, "si": si, "key": key, "side": side, "entry": entry, "qty": qty, "trail": trail, "dist": dist,
                              "margin": notional / rules.lev, "stop": entry - side * dist,
                              "tp": (np.inf * side) if trail else entry + side * dist * g["tp_r"],
                              "e_j": j, "max_hold": int(g["max_hold"]), "t_entry": d.t[j], "lc": entry, "bar_h": d.bar_h}
        # 2) 在各自周期的 K 线收盘时检查止损 / 止盈 / 到期
        for name in list(open_pos):
            p = open_pos[name]
            d, _, close_map = series[p["key"]]
            j = close_map[gi]
            if j < 0 or j < p["e_j"]:
                continue
            o, hi, lo, c = d.o[j], d.h[j], d.l[j], d.c[j]
            first = j == p["e_j"]
            exit_p, reason = None, ""
            stop_name = "移动止损" if p["side"] * (p["stop"] - p["entry"]) > 0 else "止损"
            if p["side"] == 1:
                if lo <= p["stop"]:
                    exit_p, reason = (p["stop"] if first else min(p["stop"], o)), stop_name
                elif hi >= p["tp"]:
                    exit_p, reason = (p["tp"] if first else max(p["tp"], o)), "止盈"
            else:
                if hi >= p["stop"]:
                    exit_p, reason = (p["stop"] if first else max(p["stop"], o)), stop_name
                elif lo <= p["tp"]:
                    exit_p, reason = (p["tp"] if first else min(p["tp"], o)), "止盈"
            if exit_p is None and j - p["e_j"] >= p["max_hold"]:
                exit_p, reason = c, "到期"
            p["lc"] = c
            if exit_p is None and p["trail"]:
                p["stop"] = max(p["stop"], hi - p["dist"]) if p["side"] == 1 else min(p["stop"], lo + p["dist"])
            if exit_p is not None:
                held_h = (j - p["e_j"] + 1) * p["bar_h"]
                gross = p["side"] * p["qty"] * (exit_p - p["entry"])
                cost = p["qty"] * exit_p * FEE + p["qty"] * p["entry"] * FUNDING_PER_H * held_h
                pnl = max(gross - cost, -p["margin"])            # 逐仓：最多亏掉这笔保证金
                cash += pnl
                closed.append({**p, "exit": exit_p, "t_exit": d.t[j], "pnl": pnl - p["qty"] * p["entry"] * FEE, "reason": reason})
                del open_pos[name]
        # 3) 权益（含浮动盈亏，按各持仓最近一根已收盘 K 线）
        floating = sum(p["side"] * p["qty"] * (p["lc"] - p["entry"]) for p in open_pos.values())
        if not paused and rules.daily_stop and cash + floating <= day_start * (1 - rules.daily_stop) and open_pos:
            # 日亏达到上限：按最近收盘价全部平仓，当天停手
            for name in list(open_pos):
                p = open_pos.pop(name)
                held_h = max(1, (int((grid[gi] - p["t_entry"]) / np.timedelta64(1, "m")) / 60))
                gross = p["side"] * p["qty"] * (p["lc"] - p["entry"])
                cost = p["qty"] * p["lc"] * FEE + p["qty"] * p["entry"] * FUNDING_PER_H * held_h
                pnl = max(gross - cost, -p["margin"])
                cash += pnl
                closed.append({**p, "exit": p["lc"], "t_exit": grid[gi], "pnl": pnl - p["qty"] * p["entry"] * FEE, "reason": "日亏停手"})
            floating = 0.0
            paused = True
            stops.append(grid[gi])
        eq[gi] = cash + floating
    run_portfolio.stops = stops
    return grid, eq, step_h, list(open_pos.values()), closed
