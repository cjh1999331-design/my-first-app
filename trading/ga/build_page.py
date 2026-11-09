#!/usr/bin/env python3
"""把 results.json 压缩后嵌入 page_template.html，生成单文件页面 dashboard.html。

  python ga/build_page.py [--results ga/results.json] [--out ga/dashboard.html]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results.json"))
    ap.add_argument("--out", default=str(HERE / "dashboard.html"))
    args = ap.parse_args()
    d = json.loads(Path(args.results).read_text())
    sp = list(d["meta"]["species"])
    keys = ["trades", "ret", "dd", "win", "pf", "fit"]
    inds = [[i["id"], sp.index(i["species"]), i["gen"], i["origin"], i["parents"], i["desc"],
             [i["train"][k] for k in keys], [i["oos"][k] for k in keys], int(i["oos_pass"]),
             i["alive_until"], len(i["champ_gens"]), i.get("folds")] for i in d["individuals"]]
    curves = {k: [p for n, p in enumerate(v) if n % 2 == 0 or n == len(v) - 1] for k, v in d["curves"].items()}
    trades = {i["id"]: i["recent_trades"] for i in d["individuals"] if i.get("recent_trades")}
    rounds = []
    labels = {"results_r1.json": ("第 1 轮", "单段训练 2024-02 起"),
              "results_r2.json": ("第 2 轮", "+ 滚动窗口、参数邻居、成本加压"),
              "results_r3.json": ("第 3 轮", "+ 训练期拉长到 2021-02 起"),
              "results_r4.json": ("第 4 轮", "+ 币种扩展到成交额前 100"),
              Path(args.results).name: ("第 5 轮", "+ 周期基因（15 分钟 / 1 小时 / 4 小时 / 日线）")}
    for fn, (name, how) in labels.items():
        f = HERE / fn
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        A = r["individuals"]
        I = {i["id"]: i for i in A}
        fin = [I[x] for x in r["final_population"]]
        ok = sorted([i for i in A if i["train"]["trades"] >= 30], key=lambda i: -i["train"]["fit"])
        top = ok[:100]
        ch = I[r["champion"]]
        rounds.append({"name": name, "how": how, "final_pass": sum(i["oos_pass"] for i in fin), "final_n": len(fin),
                       "final_oos": round(sum(i["oos"]["ret"] for i in fin) / len(fin), 1),
                       "top_oos": round(sum(i["oos"]["ret"] for i in top) / len(top), 1),
                       "champ": ch["id"], "champ_train": ch["train"]["ret"], "champ_oos": ch["oos"]["ret"]})
    replay = None
    rp = HERE / "replay.json"
    if rp.exists():
        rj = json.loads(rp.read_text())
        replay = rj
    uni = json.loads((HERE / "universe.json").read_text()) if (HERE / "universe.json").exists() else None
    # 持续进化：有 live/state.json 时，账户页显示滚动进化的模拟账户，进化页显示换代记录
    live = None
    sp_, ap_ = HERE / "live" / "state.json", HERE / "live" / "account.json"
    if sp_.exists() and ap_.exists():
        st = json.loads(sp_.read_text())
        replay = json.loads(ap_.read_text())
        team = lambda t: None if not t else {"ids": t["ids"], "since": t["since"][:10],
                                             "items": [{"id": i, "species": t["species"][i], "desc": t["desc"][i]} for i in t["ids"]]}
        live = {"gen": st["gen"], "now": str(st["now"])[:16].replace("T00:00", ""), "probation": st["config"]["probation"],
                "start": st["segments"][0]["start"][:10] if st["segments"] else None,
                "active": team(st["active"]), "challenger": team(st["challenger"]), "changes": st["changes"],
                "log": [{"date": l["date"], "gen": l["gen"], "best": l["best"]} for l in st["log"]],
                "segments": [{"start": g["start"][:10], "ids": g["team"]["ids"]} for g in st["segments"]]}
        from engine import SPECIES
        from rolling import STAGNATION
        live["species"] = {sp: SPECIES[sp] for sp in st["pop"]}
        lg = st["log"][-1] if st["log"] else {}
        if lg.get("div"):
            tf = lg.get("tfs", {})
            ev = st.get("events", [])
            live["diversity"] = (f" 防止策略越来越固定：训练窗口只用最近 {st['config'].get('train_years', 0):g} 年（现在从 {lg.get('train_start')} 起），"
                                 f"越近的数据权重越大；每个物种 {st['config']['per_species']} 个个体，相近个体分摊得分，每个周期都保种；"
                                 f"冠军 {STAGNATION} 代不换人就大灭绝重启（已发生 {len(ev)} 次）；上线组合要求策略之间日收益相关性不超过 0.6。"
                                 f"当前种群周期分布：" + "、".join(f"{k} {v}" for k, v in sorted(tf.items(), key=lambda x: -x[1])) + "。")
    page = {"universe": uni, "meta": d["meta"], "champion": d["champion"], "species_champions": d["species_champions"],
            "oos_best": d["oos_best"], "final_population": d["final_population"], "history": d["history"],
            "inds": inds, "rounds": rounds, "replay": replay, "live": live}
    blob = json.dumps(page, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (HERE / "page_template.html").read_text().replace("__DATA__", blob)
    Path(args.out).write_text(html)
    print(f"{args.out}: {len(html) / 1024:.0f} KB, {len(inds)} 个个体")


if __name__ == "__main__":
    main()
