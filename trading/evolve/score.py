#!/usr/bin/env python3
"""
手动/Agent 模式的打分工具：策略文件由人或 Claude Code 直接写进 --dir，
这个脚本负责安全检查 + 回测 + 记录。适合不想用 API Key、直接在 Claude Code 会话里进化的情况。

  python evolve/score.py --dir evo_futures/strategies --range 20240201-20250601 \
      --gen 1 --parent EvoFuturesSeed --note "加了 ADX 过滤" Evo_G1_C1

验证集请用 --tag valid 单独跑，并且只在进化结束后跑一次。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evolve import ROOT, check_code, fitness, run_backtest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="策略文件所在目录")
    ap.add_argument("--range", required=True, help="回测区间")
    ap.add_argument("--tag", default="train", help="train / valid，写进记录")
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--parent", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("names", nargs="+", help="策略类名（文件名需与类名相同）")
    args, extra = ap.parse_known_args()

    sdir = (ROOT / args.dir).resolve()
    log = sdir.parent / "history.jsonl"
    for name in args.names:
        rec = {"name": name, "gen": args.gen, "parent": args.parent, "tag": args.tag,
               "range": args.range, "note": args.note, "time": time.strftime("%F %T")}
        try:
            check_code((sdir / f"{name}.py").read_text(), name)
            m = run_backtest(sdir, name, args.range, ROOT / "user_data" / "backtest_results" / name, extra)
            rec.update(metrics=m, fitness=fitness(m, args.min_trades))
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)[-1500:]
        with log.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
