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
             i["alive_until"], len(i["champ_gens"])] for i in d["individuals"]]
    curves = {k: [p for n, p in enumerate(v) if n % 2 == 0 or n == len(v) - 1] for k, v in d["curves"].items()}
    trades = {i["id"]: i["recent_trades"] for i in d["individuals"] if i.get("recent_trades")}
    page = {"meta": d["meta"], "champion": d["champion"], "species_champions": d["species_champions"],
            "oos_best": d["oos_best"], "final_population": d["final_population"], "history": d["history"],
            "inds": inds, "curves": curves, "trades": trades}
    blob = json.dumps(page, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (HERE / "page_template.html").read_text().replace("__DATA__", blob)
    Path(args.out).write_text(html)
    print(f"{args.out}: {len(html) / 1024:.0f} KB, {len(inds)} 个个体")


if __name__ == "__main__":
    main()
