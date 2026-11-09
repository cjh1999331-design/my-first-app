#!/usr/bin/env python3
"""
策略进化循环：Claude 改写策略 -> freqtrade 回测打分 -> 保留最好的 -> 继续迭代。

流程（每一代）：
  1. 从当前精英池里挑最好的策略作为“父代”
  2. 把父代代码 + 训练集回测指标 + 历史记录发给 Claude，让它写出一个改进版“子代”
  3. 静态安全检查（import 白名单、禁止 exec/open 等）
  4. 在【训练区间】回测，算 fitness
  5. 子代进入精英池，按 fitness 排序，只保留前 N 名
最后：精英池里的策略在【验证区间】（进化过程中从没用过的数据）再回测一次，
      报告训练分 vs 验证分，用来判断是不是过拟合。

用法（在 trading/ 目录下）：
  python evolve/evolve.py --generations 5 --children 2 \
      --train 20240101-20250101 --valid 20250101-20250901

需要：
  - freqtrade 在 PATH 里（pip install -r requirements.txt）
  - 已经下载好数据（freqtrade download-data ...，见 README）
  - 环境变量 ANTHROPIC_API_KEY（或 `ant auth login` 登录过）
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent  # trading/
USER_DATA = ROOT / "user_data"
CONFIG = USER_DATA / "config.json"
SEED = USER_DATA / "strategies" / "EvoSeedStrategy.py"

# 生成的策略允许 import 的顶层模块（其他一律拒绝）
ALLOWED_IMPORTS = {
    "freqtrade", "talib", "pandas", "numpy", "technical", "pandas_ta",
    "datetime", "typing", "functools", "math", "__future__",
}
# 禁止出现的名字（防止大模型生成读写文件、执行命令之类的代码）
FORBIDDEN_NAMES = {
    "exec", "eval", "compile", "open", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "input", "breakpoint",
}

SYSTEM_PROMPT = """\
你是一名严谨的量化策略研究员，负责改进 freqtrade（INTERFACE_VERSION = 3）策略（现货或 USDT 永续合约，以父代为准）。

硬性要求：
- 只输出一个完整的 Python 文件，放在一个 ```python 代码块里，代码块外可以写一两句改动说明。
- 类名必须正好是用户指定的名字，继承 IStrategy，保持 timeframe 和 can_short 与父代一致。
- 合约策略的 leverage() 返回值必须在 1~10 之间；注意 stoploss/ROI 是按含杠杆收益率计算的。
- 只能 import：freqtrade、talib、pandas、numpy、technical、datetime、typing、functools、math。
- 禁止未来函数：不能用 shift(-n)、不能用 center=True 的 rolling、不能用整列统计量（如 df['close'].max()）
  去生成信号；每根 K 线的信号只能用当根及之前的数据。
- 入场写 enter_long / enter_short 列，出场写 exit_long / exit_short 列；保持 startup_candle_count 足够覆盖最长指标周期。
- 不要读写文件、不要联网、不要用 exec/eval/open。

研究原则：
- 每次只做 1~2 个有逻辑依据的改动（新指标过滤、出场逻辑、止损/ROI、市场状态判断等），
  并在文件顶部 docstring 里写明这次改了什么、为什么。
- 目标是“稳健”：交易笔数足够、回撤小、收益稳定，而不是把训练集收益拟合到极致。
- 参数尽量用 IntParameter / DecimalParameter 声明，便于之后 hyperopt。
"""


@dataclass
class Candidate:
    name: str
    path: str
    generation: int
    parent: str | None
    train: dict = field(default_factory=dict)
    fitness: float = float("-inf")
    valid: dict = field(default_factory=dict)
    note: str = ""
    error: str = ""


# ---------------------------------------------------------------- 回测与打分
def run_backtest(strategy_dir: Path, class_name: str, timerange: str, out_dir: Path,
                 extra_args: list[str]) -> dict:
    """调用 freqtrade 回测，返回该策略的统计 dict；失败时抛 RuntimeError。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "freqtrade", "backtesting",
        "--config", str(CONFIG),
        "--userdir", str(USER_DATA),
        "--strategy-path", str(strategy_dir),
        "--strategy", class_name,
        "--timerange", timerange,
        "--export", "trades",
        "--backtest-directory", str(out_dir),
        *extra_args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-2500:]
        raise RuntimeError(f"freqtrade 回测失败 (exit {proc.returncode}):\n{tail}")

    from freqtrade.data.btanalysis import load_backtest_stats  # 延迟导入，启动更快

    stats = load_backtest_stats(out_dir)
    s = stats["strategy"][class_name]
    return {
        "trades": int(s.get("total_trades", 0)),
        "profit_pct": round(100 * float(s.get("profit_total", 0.0)), 2),
        "max_dd_pct": round(100 * float(s.get("max_drawdown_account", 0.0)), 2),
        "winrate_pct": round(100 * float(s.get("winrate", 0.0) or 0.0), 1),
        "profit_factor": round(float(s.get("profit_factor", 0.0) or 0.0), 2),
        "sharpe": round(float(s.get("sharpe", 0.0) or 0.0), 2),
        "calmar": round(float(s.get("calmar", 0.0) or 0.0), 2),
    }


def fitness(m: dict, min_trades: int) -> float:
    """收益 - 回撤惩罚；交易太少直接判负（太少的样本没有统计意义）。"""
    if m.get("trades", 0) < min_trades:
        return -1000.0 + m.get("trades", 0)
    return round(m["profit_pct"] - 1.5 * m["max_dd_pct"], 2)


# ---------------------------------------------------------------- 安全检查
def check_code(code: str, class_name: str) -> None:
    """静态检查：语法、类名、import 白名单、危险函数。不是沙箱，只是基本防线。"""
    tree = ast.parse(code)
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    if class_name not in classes:
        raise ValueError(f"代码里没有类 {class_name}（找到: {classes}）")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        else:
            mods = []
        for mod in mods:
            if mod not in ALLOWED_IMPORTS:
                raise ValueError(f"不允许 import {mod}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"不允许使用 {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__":
            raise ValueError(f"不允许访问 {node.attr}")
        # 粗查未来函数：shift(负数)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "shift" and node.args):
            arg = node.args[0]
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                raise ValueError("检测到 shift(负数)，疑似未来函数")


def extract_code(text: str) -> str:
    blocks = re.findall(r"```python\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        raise ValueError("回复里没有 ```python 代码块")
    return max(blocks, key=len)


# ---------------------------------------------------------------- 调用 Claude
def ask_claude(client: anthropic.Anthropic, model: str, effort: str, user_msg: str) -> str:
    with client.beta.messages.stream(
        model=model,
        max_tokens=64000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        # 安全分类器拒答时自动换模型重试（服务端处理）
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("模型拒绝了这次请求")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("回复被 max_tokens 截断")
    return "".join(b.text for b in msg.content if b.type == "text")


def build_prompt(parent: Candidate, parent_code: str, child_name: str,
                 history: list[Candidate], train_range: str) -> str:
    # 只给训练集指标——验证集结果绝不能反馈给模型，否则验证集就被“污染”了
    hist_lines = []
    for c in sorted(history, key=lambda c: c.generation)[-12:]:
        status = f"错误: {c.error[:120]}" if c.error else f"train={json.dumps(c.train, ensure_ascii=False)} fitness={c.fitness}"
        hist_lines.append(f"- {c.name}（父代 {c.parent}）: {c.note[:150]} | {status}")
    return f"""\
请基于下面的父代策略，写一个改进版。

新类名：{child_name}
训练区间：{train_range}
打分公式：fitness = 收益% - 1.5 × 最大回撤%；交易笔数不足会直接判负。

父代 {parent.name} 的训练集回测：
{json.dumps(parent.train, ensure_ascii=False)}  fitness={parent.fitness}

最近的进化历史（避免重复失败的方向）：
{chr(10).join(hist_lines) or "（暂无）"}

父代代码：
```python
{parent_code}
```"""


# ---------------------------------------------------------------- 主循环
def main() -> None:
    ap = argparse.ArgumentParser(description="用 Claude + freqtrade 回测做策略进化")
    ap.add_argument("--generations", type=int, default=5, help="进化代数")
    ap.add_argument("--children", type=int, default=2, help="每代生成几个子代")
    ap.add_argument("--elite", type=int, default=4, help="精英池大小")
    ap.add_argument("--train", required=True, help="训练区间，如 20240101-20250101")
    ap.add_argument("--valid", required=True, help="验证区间（必须在训练区间之后、且不重叠）")
    ap.add_argument("--min-trades", type=int, default=30, help="训练集最少交易笔数")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--seed", default=str(SEED), help="种子策略文件")
    ap.add_argument("--dry", action="store_true", help="不调用 Claude，只回测种子（用于检查环境）")
    args, extra = ap.parse_known_args()  # 其余参数原样传给 freqtrade backtesting

    run_dir = ROOT / "evolve" / "runs" / time.strftime("%Y%m%d-%H%M%S")
    strat_dir = run_dir / "strategies"
    strat_dir.mkdir(parents=True)
    history_file = run_dir / "history.jsonl"
    print(f"本次运行目录: {run_dir}")

    def log(c: Candidate) -> None:
        with history_file.open("a") as f:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    # 种子
    seed_src = Path(args.seed)
    seed_name = seed_src.stem
    shutil.copy(seed_src, strat_dir / seed_src.name)
    seed = Candidate(seed_name, str(strat_dir / seed_src.name), 0, None, note="种子策略")
    seed.train = run_backtest(strat_dir, seed_name, args.train, run_dir / "bt" / seed_name, extra)
    seed.fitness = fitness(seed.train, args.min_trades)
    log(seed)
    print(f"[gen 0] {seed_name}: {seed.train} fitness={seed.fitness}")

    elite: list[Candidate] = [seed]
    history: list[Candidate] = [seed]

    if not args.dry:
        client = anthropic.Anthropic()
        for gen in range(1, args.generations + 1):
            parent = elite[0]
            parent_code = Path(parent.path).read_text()
            for i in range(1, args.children + 1):
                name = f"Evo_G{gen}_C{i}"
                child = Candidate(name, str(strat_dir / f"{name}.py"), gen, parent.name)
                prompt = build_prompt(parent, parent_code, name, history, args.train)
                try:
                    code, reply = None, ""
                    for attempt in range(2):  # 第一次失败时把报错发回去修一次
                        reply = ask_claude(client, args.model, args.effort, prompt)
                        try:
                            code = extract_code(reply)
                            check_code(code, name)
                            Path(child.path).write_text(code)
                            child.train = run_backtest(strat_dir, name, args.train,
                                                       run_dir / "bt" / name, extra)
                            break
                        except (ValueError, SyntaxError, RuntimeError) as e:
                            if attempt == 1:
                                raise
                            prompt += f"\n\n你上一版代码出错了，请修正后重新输出完整文件：\n{e}"
                    ast_doc = ast.get_docstring(ast.parse(code)) or ""
                    child.note = (ast_doc or reply.split("```")[0]).strip().replace("\n", " ")[:300]
                    child.fitness = fitness(child.train, args.min_trades)
                    elite.append(child)
                    print(f"[gen {gen}] {name}: {child.train} fitness={child.fitness}")
                except Exception as e:  # noqa: BLE001 - 单个子代失败不影响整体
                    child.error = str(e)[-1500:]
                    print(f"[gen {gen}] {name}: 失败 -> {str(e)[:200]}")
                history.append(child)
                log(child)
            elite = sorted(elite, key=lambda c: c.fitness, reverse=True)[: args.elite]
            print(f"  精英池: {[(c.name, c.fitness) for c in elite]}")

    # 验证集：只在最后跑一次
    print(f"\n==== 验证区间 {args.valid}（进化过程中未使用）====")
    for c in elite:
        try:
            c.valid = run_backtest(strat_dir, c.name, args.valid, run_dir / "bt_valid" / c.name, extra)
        except RuntimeError as e:
            c.valid = {"error": str(e)[-300:]}
        vfit = fitness(c.valid, max(1, args.min_trades // 3)) if "error" not in c.valid else None
        print(f"{c.name:<18} train_fit={c.fitness:>8}  valid_fit={vfit}  valid={c.valid}")

    summary = [asdict(c) for c in elite]
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {run_dir / 'summary.json'}")
    print("提示：训练分高但验证分差很多 = 过拟合。挑验证集也稳的策略，先 dry-run 观察，再考虑实盘。")


if __name__ == "__main__":
    sys.exit(main())
