#!/usr/bin/env python3
"""
遗传算法策略进化：4 个物种（策略家族），每个物种一个小种群，突变 + 交叉 + 随机移民，跑 N 代。

  python ga/evolve_ga.py --generations 60 --per-species 8 --out ga/results.json

- 选择只看训练区间的适应度；样本外区间每个个体都会算，但只用于展示，绝不参与选择
- 适应度 = 年化收益 / max(最大回撤, 5%)；训练期少于 30 笔交易会被扣分
- 输出 JSON 供 ga/build_page.py 生成可视化页面
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np

from multiprocessing import Pool

from engine import SPECIES, TF_HOURS, TF_NAME, all_signals, backtest, daily_equity, load_all

W = {}  # 子进程共享的数据（fork 时继承）


def evaluate(job):
    genes, neighbors = job
    w = W
    data = w["data"]
    sigs = all_signals(genes, data)
    tr = backtest(genes, data, w["tr0"], w["tr1"], sigs=sigs)
    oo = backtest(genes, data, w["oo0"], w["oo1"], sigs=sigs) if w["oo1"] > w["oo0"] else \
        {"trades": 0, "ret": 0.0, "dd": 0.0, "win": 0.0, "pf": 0.0, "fit": 0.0}
    strip = lambda m: {k: v for k, v in m.items() if not k.startswith("_")}
    train, extra = strip(tr), {}
    if w["mode"] == "robust":
        fw = w.get("fold_w")
        base, rets = robust_eval(genes, data, w["folds"], w["fee"], sigs, fw)
        nb = [robust_eval(n, data, w["folds"], w["fee"], weights=fw)[0] for n in neighbors]
        score = 0.5 * base + 0.5 * min(nb) if nb else base
        train["calmar"] = train["fit"]
        train["fit"] = round(score, 3)
        extra = {"folds": rets, "fold_score": round(base, 3), "neighbor_min": round(min(nb), 3) if nb else None}
    return train, strip(oo), extra

TRAIN = ("2024-02-01", "2025-06-01")
OOS = ("2025-06-01", "2026-11-01")

# 基因范围：(最小, 最大, 是否整数)
COMMON = {
    "stop_atr": (0.8, 4.0, False),
    "tp_r": (0.8, 6.0, False),
    "max_hold": (4, 96, True),
    "risk": (0.01, 0.01, False),     # 每笔风险固定 1%：先只进化开仓和离场，证明能稳定赚钱再放开
}
SPECIFIC = {
    "vol_breakout": {"k": (1.2, 3.5, False)},
    "ma_cross": {"fast": (5, 40, True), "slow": (20, 150, True)},
    "rsi_revert": {"lo": (15, 40, True), "hi": (60, 85, True)},
    "channel": {"n": (12, 120, True)},
    "vol_surge": {"m": (1.5, 6.0, False), "n": (10, 100, True)},
    "rel_strength": {"n": (6, 120, True), "th": (0.02, 0.3, False)},
}
TREND_CHOICES = [0, 50, 100, 150, 200, 300]
DIRS = ["long", "short", "both"]
TFS = ["15m", "1h", "4h", "1d"]   # 周期基因；--tfs 可以限制
EXITS = ["trail", "tp", "tp_trail"]  # 离场基因：移动止损不设止盈 / 固定止盈止损 / 止盈 + 移动止损，由进化决定


def clamp(v, lo, hi, is_int):
    v = min(hi, max(lo, v))
    return int(round(v)) if is_int else round(v, 3 if hi < 0.1 else 2)


def fix(g):
    if g["species"] == "ma_cross" and g["slow"] < g["fast"] + 5:
        g["slow"] = min(150, g["fast"] + 5 + random.randint(0, 30))
    if g["species"] == "rsi_revert" and g["hi"] - g["lo"] < 25:
        g["hi"] = min(85, g["lo"] + 25)
    return g


def random_genes(sp):
    g = {"species": sp, "tf": random.choice(TFS), "exit": random.choice(EXITS), "dir": random.choice(DIRS), "trend": random.choice(TREND_CHOICES)}
    for k, (lo, hi, it) in {**COMMON, **SPECIFIC[sp]}.items():
        g[k] = clamp(random.uniform(lo, hi), lo, hi, it)
    return fix(g)


def distance(a, b) -> float:
    """两个同物种个体的基因距离：数值基因按范围归一化取平均，周期 / 方向 / 趋势过滤不同各加一点。"""
    keys = {**COMMON, **SPECIFIC[a["species"]]}
    keys = {k: v for k, v in keys.items() if v[1] > v[0]}       # 固定不变的基因（如风险）不算距离
    d = sum(abs(a[k] - b[k]) / (hi - lo) for k, (lo, hi, _) in keys.items()) / len(keys)
    return (d + 0.3 * (a.get("tf") != b.get("tf")) + 0.2 * (a["dir"] != b["dir"]) + 0.1 * (a["trend"] != b["trend"])
            + 0.2 * (a.get("exit") != b.get("exit")))


def mutate(p, strength=0.18):
    g = copy.deepcopy(p)
    keys = list({**COMMON, **SPECIFIC[g["species"]]}.items())
    for k, (lo, hi, it) in random.sample(keys, k=random.randint(1, 3)):
        g[k] = clamp(g[k] + random.gauss(0, strength * (hi - lo)), lo, hi, it)
    if random.random() < 0.12:
        g["dir"] = random.choice(DIRS)
    if random.random() < 0.12:
        g["trend"] = random.choice(TREND_CHOICES)
    if random.random() < 0.12:
        g["tf"] = random.choice(TFS)
    if random.random() < 0.12:
        g["exit"] = random.choice(EXITS)
    return fix(g)


def crossover(a, b):
    g = {"species": a["species"]}
    for k in a:
        if k != "species":
            g[k] = copy.deepcopy(random.choice([a, b])[k])
    return fix(g)


def describe(g):
    sp = g["species"]
    d = {"long": "只做多", "short": "只做空", "both": "多空双向"}[g["dir"]]
    if sp == "vol_breakout":
        core = f"单根振幅 > {g['k']}×ATR 顺 K 线方向"
    elif sp == "ma_cross":
        core = f"EMA{g['fast']} 穿越 EMA{g['slow']}"
    elif sp == "rsi_revert":
        core = {"long": f"RSI 回升过 {g['lo']} 做多", "short": f"RSI 回落过 {g['hi']} 做空",
                "both": f"RSI 回升过 {g['lo']} 做多 / 回落过 {g['hi']} 做空"}[g["dir"]]
    elif sp == "vol_surge":
        core = f"成交量 > {g['m']}× 前 {g['n']} 根均量 顺 K 线方向"
    elif sp == "rel_strength":
        core = f"{g['n']} 根内比 BTC 多涨 / 多跌 {round(g['th'] * 100, 1)}%"
    else:
        core = f"突破 {g['n']} 根 K 线高低点"
    trend = f" · EMA{g['trend']} 趋势过滤" if g["trend"] else ""
    tf = g.get("tf", "1h")
    hold_h = g["max_hold"] * TF_HOURS[tf]
    hold = f"{hold_h / 24:.1f} 天" if hold_h >= 48 else f"{hold_h:g} 小时"
    ex = g.get("exit", "tp")
    exit_txt = {"trail": f"移动止损 {g['stop_atr']}×ATR · 不设止盈",
                "tp_trail": f"移动止损 {g['stop_atr']}×ATR · 止盈 {g['tp_r']}R"}.get(ex, f"止损 {g['stop_atr']}×ATR · 止盈 {g['tp_r']}R")
    return (f"{SPECIES[sp]} · {TF_NAME[tf]} · {core}{trend} · {d} · {exit_txt}"
            f" · 最长 {g['max_hold']} 根（{hold}） · 风险 {round(g['risk'] * 100, 2)}%")


def fold_edges(t0: str, t1: str, k: int):
    a, b = np.datetime64(t0), np.datetime64(t1)
    step = (b - a) / k
    return [(a + step * i, a + step * (i + 1)) for i in range(k)]


DD_FREE, DD_SCALE = 15.0, 15.0     # 单段回撤 15% 以内不扣分，每多 15% 扣 1 分


def fold_score(m: dict) -> float:
    """单段得分：年化收益 / 回撤，截断到 [-3, 3]，避免某一段的极端值主导。
    回撤超过 DD_FREE（%）的部分额外扣分，否则进化会一味把风险基因推到上限（多个策略同账户时风险会叠加）。"""
    return float(np.clip(m["fit"], -3, 3)) - max(0.0, m["dd"] - DD_FREE) / DD_SCALE


def robust_eval(genes, data, folds, stress_fee, sigs=None, weights=None):
    """在每个时间段分别回测（加压手续费），返回 (得分, 各段收益%)。"""
    sigs = sigs if sigs is not None else all_signals(genes, data)
    scores, rets, trades = [], [], 0
    for f0, f1 in folds:
        m = backtest(genes, data, f0, f1, fee=stress_fee, min_trades=8, sigs=sigs)
        scores.append(fold_score(m))
        rets.append(m["ret"])
        trades += m["trades"]
    sc = np.array(scores)
    w = np.ones(len(sc)) if weights is None else np.asarray(weights, float)   # 越近的时间段权重越大
    mu = float((sc * w).sum() / w.sum())
    base = mu - 0.5 * float(np.sqrt((w * (sc - mu) ** 2).sum() / w.sum()))
    profitable = sum(r > 0 for r in rets)
    need = len(folds) - 1
    if profitable < need:
        base -= (need - profitable) * 1.0
    if trades < 40:
        base -= (40 - trades) / 40
    return base, rets


def neighbor(genes, scale=0.10):
    g = copy.deepcopy(genes)
    for k, (lo, hi, it) in {**COMMON, **SPECIFIC[g["species"]]}.items():
        g[k] = clamp(g[k] * (1 + random.uniform(-scale, scale)), lo, hi, it)
    return fix(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--per-species", type=int, default=8)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results.json"))
    ap.add_argument("--mode", choices=["robust", "simple"], default="robust",
                    help="robust：滚动窗口 + 参数邻居 + 成本加压；simple：第 1 轮的单段训练打分")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--train-start", default="2021-02-01", help="训练起点；结束点固定为样本外起点")
    ap.add_argument("--neighbors", type=int, default=2)
    ap.add_argument("--stress-fee", type=float, default=0.0008)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tfs", default="15m,1h,4h,1d", help="允许的周期，逗号分隔")
    args = ap.parse_args()
    global TRAIN
    TRAIN = (args.train_start, OOS[0])
    random.seed(args.seed)
    np.random.seed(args.seed)

    global TFS
    TFS = args.tfs.split(",")
    data = load_all(tfs=TFS)
    print("数据：" + "，".join(f"{tf} {len(v)} 个币" for tf, v in data.items()), flush=True)
    tr0, tr1 = (np.datetime64(x) for x in TRAIN)
    oo0, oo1 = (np.datetime64(x) for x in OOS)

    folds = fold_edges(TRAIN[0], TRAIN[1], args.folds)
    inds: dict[str, dict] = {}
    counter = {sp: 0 for sp in SPECIES}
    short = {"vol_breakout": "v", "ma_cross": "m", "rsi_revert": "r", "channel": "c"}

    W.update(data=data, tr0=tr0, tr1=tr1, oo0=oo0, oo1=oo1, folds=folds, fee=args.stress_fee, mode=args.mode)
    pool = Pool(args.workers)
    pending = []

    def make(genes, gen, origin, parents):
        sp = genes["species"]
        iid = f"{short[sp]}{counter[sp]}"
        counter[sp] += 1
        ind = {"id": iid, "species": sp, "gen": gen, "origin": origin, "parents": parents,
               "genes": genes, "desc": describe(genes), "alive_until": gen, "champ_gens": []}
        inds[iid] = ind
        pending.append((ind, [neighbor(genes) for _ in range(args.neighbors)] if args.mode == "robust" else []))
        return ind

    def flush():
        res = pool.map(evaluate, [(ind["genes"], nb) for ind, nb in pending], chunksize=1)
        for (ind, _), (train, oos, extra) in zip(pending, res):
            ind.update(train=train, oos=oos, **extra)
            ind["oos_pass"] = oos["ret"] > 0 and oos["dd"] < 30 and oos["trades"] >= 10
        pending.clear()

    t_start = time.time()
    pop = {sp: [make(random_genes(sp), 0, "初始随机", []) for _ in range(args.per_species)] for sp in SPECIES}
    flush()
    history = []

    def record(gen):
        row = {"gen": gen}
        for sp, members in pop.items():
            members.sort(key=lambda x: x["train"]["fit"], reverse=True)
            for m in members:
                m["alive_until"] = gen
            members[0]["champ_gens"].append(gen)
            row[sp] = members[0]["train"]["fit"]
            row[sp + "_id"] = members[0]["id"]
        history.append(row)

    record(0)
    for gen in range(1, args.generations + 1):
        for sp, members in pop.items():
            members.sort(key=lambda x: x["train"]["fit"], reverse=True)
            nxt = members[: args.elite]

            def pick():
                a, b = random.sample(members, 2)
                return a if a["train"]["fit"] >= b["train"]["fit"] else b

            while len(nxt) < args.per_species:
                r = random.random()
                if r < 0.6:
                    p = pick()
                    child = make(mutate(p["genes"]), gen, "突变", [p["id"]])
                elif r < 0.88:
                    a, b = pick(), pick()
                    child = make(crossover(a["genes"], b["genes"]), gen, "交叉", sorted({a["id"], b["id"]}))
                else:
                    child = make(random_genes(sp), gen, "随机移民", [])
                nxt.append(child)
            pop[sp] = nxt
        flush()
        record(gen)
        best = max((m for ms in pop.values() for m in ms), key=lambda x: x["train"]["fit"])
        print(f"gen {gen:>3}  best {best['id']:<6} fit={best['train']['fit']:<7} "
              f"train={best['train']['ret']}%/{best['train']['dd']}%  oos={best['oos']['ret']}%/{best['oos']['dd']}%  "
              f"({time.time() - t_start:.0f}s)", flush=True)

    final = [m for ms in pop.values() for m in ms]
    final.sort(key=lambda x: x["train"]["fit"], reverse=True)
    champion = final[0]
    species_champs = [max(ms, key=lambda x: x["train"]["fit"]) for ms in pop.values()]
    oos_best = max(final, key=lambda x: x["oos"]["fit"])

    # 资金曲线：物种冠军 + 最终种群前 12，训练+样本外连续跑
    curves = {}
    for ind in {i["id"]: i for i in species_champs + final[:12] + [oos_best]}.values():
        m = backtest(ind["genes"], data, tr0, oo1, want_trades=True)
        curves[ind["id"]] = daily_equity(m, TRAIN[0], OOS[1])
        ind["recent_trades"] = m["trade_list"][-8:]

    out = {
        "meta": {"generations": args.generations, "per_species": args.per_species, "train": TRAIN, "oos": OOS, "tfs": TFS,
                 "pairs": ["BTC", "ETH", "SOL", "BNB", "XRP"], "exchange": "OKX USDT 永续", "timeframe": "1h",
                 "created": time.strftime("%Y-%m-%d %H:%M"), "species": SPECIES,
                 "mode": args.mode, "folds": [[str(a)[:10], str(b)[:10]] for a, b in folds],
                 "stress_fee": args.stress_fee, "neighbors": args.neighbors},
        "champion": champion["id"], "species_champions": [i["id"] for i in species_champs], "oos_best": oos_best["id"],
        "final_population": [i["id"] for i in final],
        "history": history, "individuals": list(inds.values()), "curves": curves,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"\n{len(inds)} 个个体，冠军 {champion['id']}：{champion['desc']}")
    print(f"训练 {champion['train']}  样本外 {champion['oos']}")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
