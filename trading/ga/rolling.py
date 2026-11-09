#!/usr/bin/env python3
"""
持续（滚动）进化 + 冠军考核 + 模拟账户。

每次运行（每天一次，或模拟时每周一步）：
  1. 训练窗口 = 2021-02-01 到“现在”，把最新行情加进来；
  2. 从上次保存的种群接着进化：先用新窗口重新给存活个体打分，再进化若干代；
  3. 各物种冠军（得分 > 0 的）组成“挑战者”组合，从现在起影子交易（只记账不进账户）；
     考核满 probation 天后，和当前上线组合比同一段时间的账户收益，跑赢才上线，否则淘汰；
  4. 模拟账户只由“上线组合”交易；换组合时按当时价格全部平仓，权益接着算。

所有状态存在 live/state.json，下次运行接着用。用法：

  # 实盘式：每天跑一次（先用 freqtrade 更新数据，见 live/daily.sh）
  python ga/rolling.py daily --gens 5

  # 历史模拟：从 2025-06-01 开始每 7 天走一步，检验整套流程
  python ga/rolling.py simulate --start 2025-06-01 --step-days 7 --gens 5 --init-gens 30
"""
from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import evolve_ga as G
from engine import SPECIES, START_EQUITY, TF_NAME, load_all
from portfolio import PortfolioRules, run_portfolio

HERE = Path(__file__).parent
LIVE = HERE / "live"
TRAIN_START = "2021-02-01"
SHORT = {"vol_breakout": "v", "ma_cross": "m", "rsi_revert": "r", "channel": "c", "vol_surge": "u", "rel_strength": "s"}
SHARE_SIGMA = 0.25      # 适应度共享：基因距离小于它的个体互相“分摊”得分
SHARE_COST = 0.1        # 每多一个相近个体扣的分
MIN_DIST = 0.08         # 精英之间至少要这么不同
IMMIGRANT = 0.2         # 每个新个体是随机移民的概率
STAGNATION = 25         # 物种冠军连续这么多代没换人 -> 大灭绝重启
MAX_CORR = 0.6          # 上线组合里两个策略的日收益相关性上限
TEAM_MAX = 5
DAY = np.timedelta64(1, "D")


def d64(s) -> np.datetime64:
    return np.datetime64(str(s)[:16]) if "T" in str(s) else np.datetime64(str(s)[:10])


def day_str(t) -> str:
    return str(np.datetime64(t, "m"))[:16].replace("T00:00", "")


# ------------------------------------------------------------------ 状态
def new_state(args) -> dict:
    return {"gen": 0, "next_id": {sp: 0 for sp in SPECIES}, "pop": {sp: [] for sp in SPECIES}, "archive": [],
            "log": [], "active": None, "challenger": None, "changes": [], "segments": [], "now": None,
            "stag": {sp: {"champ": None, "gens": 0} for sp in SPECIES}, "events": [],
            "config": {"per_species": args.per_species, "elite": args.elite, "probation": args.probation,
                       "tfs": args.tfs, "train_start": TRAIN_START, "train_years": args.train_years}}


def _np(o):
    """numpy 标量转成普通 Python 值，方便写 JSON。"""
    return o.item() if hasattr(o, "item") else str(o)


def load_state(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def save_state(st: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, separators=(",", ":"), default=_np))
    tmp.replace(path)


# ------------------------------------------------------------------ 进化
def train_start(st, t_end):
    """滑动训练窗口：只用最近 train_years 年（最早不早于 TRAIN_START）。"""
    yrs = st["config"].get("train_years") or 0
    t0 = np.datetime64(TRAIN_START)
    if yrs > 0:
        t0 = max(t0, np.datetime64(t_end, "D") - int(round(365.25 * yrs)) * DAY)
    return t0


def evaluate_all(genes_list, data, t0, t_end, workers):
    """在训练窗口 [t0, t_end) 上给一批基因打稳健分（并行）。6 段时间越近权重越大（0.5 -> 1.5）。"""
    G.W.update(data=data, tr0=t0, tr1=t_end, oo0=t_end, oo1=t_end,
               folds=G.fold_edges(t0, t_end, 6), fee=0.0008, mode="robust", fold_w=np.linspace(0.5, 1.5, 6))
    jobs = [(g, [G.neighbor(g) for _ in range(2)]) for g in genes_list]
    with Pool(workers) as pool:
        return pool.map(G.evaluate, jobs, chunksize=1)


def evolve(st: dict, data, t_end, gens: int, workers: int):
    cfg = st["config"]

    def new_ind(genes, origin, parents):
        sp = genes["species"]
        iid = f"{SHORT[sp]}{st['next_id'][sp]}"
        st["next_id"][sp] += 1
        return {"id": iid, "species": sp, "genes": genes, "gen": st["gen"], "origin": origin, "parents": parents,
                "desc": G.describe(genes), "fit": None, "train": None}

    t0 = train_start(st, t_end)

    def score(inds):
        res = evaluate_all([i["genes"] for i in inds], data, t0, t_end, workers)
        for i, (train, _, extra) in zip(inds, res):
            i["fit"], i["train"], i["folds"] = train["fit"], train, extra.get("folds")

    def shared(members):
        """适应度共享：周围相近个体越多，选择时的得分越低，防止全种群挤到同一个参数上。"""
        out = {}
        for x in members:
            niche = sum(max(0.0, 1 - G.distance(x["genes"], y["genes"]) / SHARE_SIGMA) for y in members)
            out[x["id"]] = x["fit"] - SHARE_COST * (niche - 1)
        return out

    def random_tf(sp, tf):
        g = G.random_genes(sp)
        g["tf"] = tf
        return g

    fresh = []
    for sp in SPECIES:                                   # 第一次（或新加的物种）：随机初始种群
        st["pop"].setdefault(sp, [])
        st["next_id"].setdefault(sp, 0)
        st.setdefault("stag", {}).setdefault(sp, {"champ": None, "gens": 0})
        if not st["pop"][sp]:
            st["pop"][sp] = [new_ind(G.random_genes(sp), "初始随机", []) for _ in range(cfg["per_species"])]
            fresh += st["pop"][sp]
    # 训练窗口变了，存活个体重新打分
    score([i for ms in st["pop"].values() for i in ms])
    for c in fresh:
        st["archive"].append({k: c[k] for k in ("id", "species", "gen", "origin", "parents", "desc", "fit")})
    tfs = cfg["tfs"].split(",") if isinstance(cfg["tfs"], str) else cfg["tfs"]
    n = cfg["per_species"]
    for _ in range(gens):
        st["gen"] += 1
        born = []
        for sp, members in st["pop"].items():
            members.sort(key=lambda x: x["fit"], reverse=True)
            sg = st["stag"][sp]
            if members[0]["id"] == sg["champ"]:
                sg["gens"] += 1
            else:
                sg["champ"], sg["gens"] = members[0]["id"], 0
            if sg["gens"] >= STAGNATION:
                # 大灭绝：只留前 3 名，其余换成冠军的大幅突变体和随机新个体
                nxt = members[:3]
                while len(nxt) < n:
                    if len(nxt) % 2:
                        g = members[0]["genes"]
                        for _ in range(3):
                            g = G.mutate(g, strength=0.4)
                        c = new_ind(g, "灭绝后突变", [members[0]["id"]])
                    else:
                        c = new_ind(G.random_genes(sp), "灭绝后移民", [])
                    nxt.append(c)
                    born.append(c)
                st["events"].append({"date": str(t_end)[:10], "gen": st["gen"], "species": sp, "type": "灭绝重启",
                                     "champ": members[0]["id"], "stale": sg["gens"]})
                sg["gens"] = 0
                st["pop"][sp] = nxt
                continue
            sh = shared(members)
            # 精英：按得分挑，但和已选精英太像的跳过
            nxt = []
            for m in members:
                if len(nxt) >= cfg["elite"]:
                    break
                if all(G.distance(m["genes"], e["genes"]) >= MIN_DIST for e in nxt):
                    nxt.append(m)
            # 周期保种：每个周期至少留下它最好的一个
            for tf in tfs:
                if not any(x["genes"].get("tf") == tf for x in nxt):
                    best = next((m for m in members if m["genes"].get("tf") == tf), None)
                    if best is not None:
                        nxt.append(best)

            def pick():
                a, b = random.sample(members, 2)
                return a if sh[a["id"]] >= sh[b["id"]] else b

            while len(nxt) < n:
                r = random.random()
                if r < IMMIGRANT:
                    c = new_ind(G.random_genes(sp), "随机移民", [])
                elif r < IMMIGRANT + 0.5:
                    p = pick()
                    c = new_ind(G.mutate(p["genes"]), "突变", [p["id"]])
                else:
                    a, b = pick(), pick()
                    c = new_ind(G.crossover(a["genes"], b["genes"]), "交叉", sorted({a["id"], b["id"]}))
                nxt.append(c)
                born.append(c)
            # 每个周期至少 2 个个体：不够就把新生儿换成该周期的随机个体
            for tf in tfs:
                while sum(x["genes"].get("tf") == tf for x in nxt) < 2:
                    victims = [x for x in nxt if x in born and sum(y["genes"].get("tf") == x["genes"].get("tf") for y in nxt) > 2]
                    if not victims:
                        break
                    v = victims[-1]
                    nxt.remove(v)
                    born.remove(v)
                    c = new_ind(random_tf(sp, tf), "周期保种", [])
                    nxt.append(c)
                    born.append(c)
            st["pop"][sp] = nxt
        score(born)
        for c in born:                                   # 存档（页面画谱系用），只留最近 70 代
            st["archive"].append({k: c[k] for k in ("id", "species", "gen", "origin", "parents", "desc", "fit")})
    st["archive"] = [a for a in st["archive"] if a["gen"] > st["gen"] - 70]
    for ms in st["pop"].values():
        ms.sort(key=lambda x: x["fit"], reverse=True)


def champions(st) -> list[dict]:
    """各物种冠军中得分 > 0 的，按得分排（优先级）。"""
    cs = [ms[0] for ms in st["pop"].values() if ms and ms[0]["fit"] is not None and ms[0]["fit"] > 0]
    return sorted(cs, key=lambda x: -x["fit"])


def daily_returns(genes, data, t0, t1):
    m = G.backtest(genes, data, t0, t1, min_trades=1)
    eq = np.array([v for _, v in G.daily_equity(m, str(t0)[:10], str(t1)[:10])])
    return np.diff(eq) / eq[:-1]


def select_team(st, data, t_end) -> list[dict]:
    """挑上线组合：每个物种前 3 名（得分 > 0）做候选，按得分从高到低加入，
    但和已入选策略最近一年日收益相关性超过 MAX_CORR 的跳过，每个物种最多一个，最多 TEAM_MAX 个。"""
    cands = [m for ms in st["pop"].values() for m in ms[:3] if m["fit"] is not None and m["fit"] > 0]
    cands.sort(key=lambda x: -x["fit"])
    t0 = t_end - 365 * DAY
    rets, team = {}, []
    for c in cands:
        if len(team) >= TEAM_MAX:
            break
        if any(t["species"] == c["species"] for t in team):
            continue
        rets[c["id"]] = daily_returns(c["genes"], data, t0, t_end)
        corr = [np.corrcoef(rets[c["id"]], rets[t["id"]])[0, 1] for t in team]
        corr = [x for x in corr if np.isfinite(x)]
        if corr and max(corr) > MAX_CORR:
            continue
        team.append(c)
    return team


def team_of(inds) -> dict:
    return {"ids": [i["id"] for i in inds], "genes": {i["id"]: i["genes"] for i in inds},
            "desc": {i["id"]: i["desc"] for i in inds}, "species": {i["id"]: i["species"] for i in inds}}


# ------------------------------------------------------------------ 账户
def run_team(team, data, t0, t1, equity, rules: PortfolioRules):
    if not team or not team["ids"] or t1 <= t0:
        return None
    r = PortfolioRules(**{**rules.__dict__, "start_equity": equity})
    return run_portfolio([(i, team["genes"][i]) for i in team["ids"]], data, t0, t1, r)


def manage(st, data, now, rules: PortfolioRules):
    probation = st["config"]["probation"]
    champ_team = team_of(select_team(st, data, now))
    if st["active"] is None:
        if champ_team["ids"]:
            st["active"] = {**champ_team, "since": str(now)}
            st["segments"].append({"start": str(now), "team": champ_team, "eq_start": START_EQUITY})
        return
    ch = st["challenger"]
    if ch and now - d64(ch["since"]) >= probation * DAY:
        t0 = d64(ch["since"])
        rc = run_team(ch, data, t0, now, START_EQUITY, rules)
        ra = run_team(st["active"], data, t0, now, START_EQUITY, rules)
        c_ret = (rc[1][-1] / START_EQUITY - 1) * 100 if rc else 0.0
        a_ret = (ra[1][-1] / START_EQUITY - 1) * 100 if ra else 0.0
        promoted = c_ret > a_ret
        st["changes"].append({"date": str(now)[:10], "since": ch["since"][:10], "from": st["active"]["ids"], "to": ch["ids"],
                              "active_ret": round(a_ret, 2), "chall_ret": round(c_ret, 2), "promoted": promoted})
        if promoted:
            eq_now = account_equity(st, data, now, rules)
            st["segments"][-1]["end"] = str(now)
            st["active"] = {k: ch[k] for k in ("ids", "genes", "desc", "species")} | {"since": str(now)}
            st["segments"].append({"start": str(now), "team": {k: ch[k] for k in ("ids", "genes", "desc", "species")},
                                   "eq_start": eq_now})
        st["challenger"] = None
    if st["challenger"] is None and champ_team["ids"] and set(champ_team["ids"]) != set(st["active"]["ids"]):
        st["challenger"] = {**champ_team, "since": str(now)}


def account_equity(st, data, t, rules):
    seg = st["segments"][-1]
    r = run_team(seg["team"], data, d64(seg["start"]), t, seg["eq_start"], rules)
    return float(r[1][-1]) if r else seg["eq_start"]


def build_account(st, data, now, rules: PortfolioRules) -> dict:
    """把各段上线组合的交易拼成一个账户，输出页面用的格式（与 paper_replay 一致）。"""
    ohlc, closed, positions, stops, last_eq = [], [], [], [], START_EQUITY
    sig = lambda v: float(f"{float(v):.6g}")
    strat_pnl = {}
    for k, seg in enumerate(st["segments"]):
        t0 = d64(seg["start"])
        t1 = d64(seg.get("end", str(now)))
        r = run_team(seg["team"], data, t0, t1, last_eq, rules)
        if r is None:
            continue
        grid, eq, step_h, open_pos, closed_raw = r
        stops += [str(x)[:16] for x in run_portfolio.stops]
        if step_h > 4:          # 组合里只有日线策略时按天记账，展开成 4 小时
            eq = np.repeat(eq, int(round(step_h / 4)))
            step_h = 4
        per4h = int(round(4 / step_h))
        e4 = eq[: len(eq) // per4h * per4h].reshape(-1, per4h)
        prev = np.concatenate([[last_eq], e4[:-1, -1]])
        ohlc += np.round(np.stack([prev, np.maximum(prev, e4.max(1)), np.minimum(prev, e4.min(1)), e4[:, -1]], 1), 2).tolist()
        ids = seg["team"]["ids"]
        for t in closed_raw:
            sid = ids[t["si"]]
            closed.append({"pair": t["name"], "side": int(t["side"]), "sid": sid, "te": str(t["t_entry"])[:16],
                           "tx": str(t["t_exit"])[:16], "entry": sig(t["entry"]), "exit": sig(t["exit"]), "lev": rules.lev,
                           "notional": round(t["qty"] * t["entry"], 2), "pnl": round(float(t["pnl"]), 2),
                           "reason": t.get("reason", "")})
            strat_pnl[sid] = strat_pnl.get(sid, 0) + float(t["pnl"])
        last = k == len(st["segments"]) - 1
        if last:
            for p in open_pos:
                sid = ids[p["si"]]
                fl = p["side"] * p["qty"] * (p["lc"] - p["entry"])
                positions.append({"pair": p["name"], "side": int(p["side"]), "sid": sid, "since": str(p["t_entry"])[:16],
                                  "entry": sig(p["entry"]), "last": sig(p["lc"]), "lev": rules.lev,
                                  "notional": round(p["qty"] * p["lc"], 2), "margin": round(p["margin"], 2),
                                  "float": round(fl, 2), "stop": sig(p["stop"]), "tp": None if p["trail"] else sig(p["tp"])})
                strat_pnl[sid] = strat_pnl.get(sid, 0) + fl
        else:  # 换组合时按当时价格平仓
            for p in open_pos:
                sid = ids[p["si"]]
                pnl = p["side"] * p["qty"] * (p["lc"] - p["entry"])
                closed.append({"pair": p["name"], "side": int(p["side"]), "sid": sid, "te": str(p["t_entry"])[:16],
                               "tx": str(t1)[:16], "entry": sig(p["entry"]), "exit": sig(p["lc"]), "lev": rules.lev,
                               "notional": round(p["qty"] * p["entry"], 2), "pnl": round(pnl, 2), "reason": "换组合"})
                strat_pnl[sid] = strat_pnl.get(sid, 0) + pnl
        last_eq = float(eq[-1])
    closed.sort(key=lambda t: t["tx"], reverse=True)
    positions.sort(key=lambda p: -p["notional"])
    descs, species = {}, {}
    for seg in st["segments"]:
        descs |= seg["team"]["desc"]
        species |= seg["team"]["species"]
    act = st["active"]["ids"] if st["active"] else []
    strategies = [{"id": i, "species": species[i], "desc": descs[i], "tf": "", "tf_name": TF_NAME[act_tf(st, i)],
                   "pnl": round(strat_pnl.get(i, 0.0), 2), "active": i in act} for i in descs]
    strategies.sort(key=lambda s: (not s["active"], -s["pnl"]))
    first = st["segments"][0]["start"] if st["segments"] else str(now)
    return {"start": first[:16] if "T" in first else first + "T00:00", "end": str(now)[:16] if "T" in str(now) else str(now) + "T00:00",
            "bar_hours": 4, "start_equity": START_EQUITY, "strategies": strategies, "ohlc": ohlc,
            "positions": positions, "closed": closed[:300],
            "closed_stats": {"n": len(closed), "wins": sum(t["pnl"] > 0 for t in closed), "realized": round(sum(t["pnl"] for t in closed), 2)},
            "rules": {"risk": rules.risk, "lev": rules.lev, "per_strategy": rules.per_strategy, "daily_stop": rules.daily_stop,
                      "max_pos": rules.per_strategy * TEAM_MAX, "max_notional": rules.max_notional, "stops": stops},
            "shuffle": {"finals": [], "dds": []}}


def act_tf(st, sid):
    for seg in st["segments"]:
        if sid in seg["team"]["genes"]:
            return seg["team"]["genes"][sid].get("tf", "1h")
    return "1h"


# ------------------------------------------------------------------ 一步
def step(st, data, now, gens, workers, rules):
    t = time.time()
    evolve(st, data, now, gens, workers)
    manage(st, data, now, rules)
    st["now"] = str(now)
    ch = champions(st)
    div = {sp: round(float(np.mean([G.distance(a["genes"], b["genes"]) for k, a in enumerate(ms) for b in ms[k + 1:]])), 3)
           for sp, ms in st["pop"].items() if len(ms) > 1}
    tfc = {}
    for ms in st["pop"].values():
        for m in ms:
            tfc[m["genes"].get("tf")] = tfc.get(m["genes"].get("tf"), 0) + 1
    st["log"].append({"date": str(now)[:10], "gen": st["gen"], "best": {i["species"]: i["fit"] for i in ch},
                      "div": div, "tfs": tfc, "train_start": str(train_start(st, now))[:10],
                      "champs": [i["id"] for i in ch], "active": st["active"]["ids"] if st["active"] else [],
                      "challenger": st["challenger"]["ids"] if st["challenger"] else []})
    print(f"{str(now)[:10]}  第 {st['gen']} 代  冠军 {[(i['id'], round(i['fit'], 2)) for i in ch]}  "
          f"上线 {st['active']['ids'] if st['active'] else '-'}  ({time.time() - t:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["daily", "simulate"])
    ap.add_argument("--state", default=str(LIVE / "state.json"))
    ap.add_argument("--gens", type=int, default=5, help="每一步进化几代")
    ap.add_argument("--init-gens", type=int, default=30, help="第一次运行时进化几代")
    ap.add_argument("--start", default="2025-06-01", help="simulate：从哪天开始")
    ap.add_argument("--end", default=None, help="simulate：到哪天结束（默认数据最后一天）")
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--probation", type=int, default=28, help="挑战者考核天数")
    ap.add_argument("--per-species", type=int, default=30)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--train-years", type=float, default=3, help="滑动训练窗口长度（年），0 = 从 2021-02 起全部")
    ap.add_argument("--tfs", default="15m,1h,4h,1d")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    G.TFS = args.tfs.split(",")
    rules = PortfolioRules()
    data = load_all(tfs=G.TFS)
    ref = data["1h"] if data.get("1h") else next(v for v in data.values() if v)
    last_bar = min(d.t[-1] for d in ref[:5]) + np.timedelta64(1, "h")
    data_end = np.datetime64(last_bar, "D") + ((last_bar - np.datetime64(last_bar, "D")) // np.timedelta64(4, "h")) * np.timedelta64(4, "h")
    path = Path(args.state)
    st = load_state(path)

    if args.mode == "simulate":
        st = new_state(args)                      # 模拟总是从头开始
        now = d64(args.start)
        end = d64(args.end) if args.end else data_end
        first = True
        while now <= end:
            step(st, data, now, args.init_gens if first else args.gens, args.workers, rules)
            first = False
            save_state(st, path)
            nxt = now + args.step_days * DAY
            if now < end < nxt:
                nxt = end
            now = nxt if nxt != now else now + DAY
    else:
        if st is None:
            st = new_state(args)
            step(st, data, data_end, args.init_gens, args.workers, rules)
        else:
            step(st, data, data_end, args.gens, args.workers, rules)
        save_state(st, path)

    acc = build_account(st, data, d64(st["now"]), rules)
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "account.json").write_text(json.dumps(acc, ensure_ascii=False, separators=(",", ":"), default=_np))
    e = acc["ohlc"][-1][3] if acc["ohlc"] else START_EQUITY
    print(f"账户 {START_EQUITY:.0f} -> {e:.2f} USDT，共 {len(st['segments'])} 段上线组合，换代考核 {len(st['changes'])} 次")


if __name__ == "__main__":
    main()
