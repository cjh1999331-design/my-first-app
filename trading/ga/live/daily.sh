#!/usr/bin/env bash
# 每天运行一次的持续进化流程（放在云服务器的 crontab 里）：
#   1. 用 freqtrade 增量更新前 100 热门币的 15m / 1h 合约 K 线
#   2. 在新的训练窗口上接着进化（从 live/state.json 的种群继续）
#   3. 挑战者考核 / 上线，更新模拟账户
#   4. 生成页面 ga/dashboard.html；设置了 GIT_PUSH=1 时提交并推送
#
# 环境变量（可选）：
#   PY            Python 解释器（默认 .venv/bin/python）
#   GENS          每天进化几代（默认 5）
#   EXTRA_CONFIG  额外的 freqtrade 配置文件（例如代理设置）
#   GIT_PUSH=1    跑完后 git commit + push
set -euo pipefail
cd "$(dirname "$0")/../.."                # trading/
PY="${PY:-.venv/bin/python}"
FT="${FT:-.venv/bin/freqtrade}"
GENS="${GENS:-5}"
LOG="ga/live/daily.log"
mkdir -p ga/live
exec >>"$LOG" 2>&1
echo "===== $(date -u '+%F %T') UTC ====="

PAIRS=$("$PY" -c 'import json;print(" ".join(c["base"]+"/USDT:USDT" for c in json.load(open("ga/universe.json"))["coins"]))')
CFG=(--config user_data/config.json)
[ -n "${EXTRA_CONFIG:-}" ] && CFG+=(--config "$EXTRA_CONFIG")
"$FT" download-data "${CFG[@]}" --userdir user_data -t 15m 1h --days 5 --candle-types futures --pairs $PAIRS \
  || echo "数据更新失败，用已有数据继续"

(cd ga && "../$PY" -W ignore rolling.py daily --gens "$GENS")
(cd ga && "../$PY" build_page.py)

if [ "${GIT_PUSH:-0}" = "1" ]; then
  git add ga/dashboard.html ga/live/state.json ga/live/account.json
  git commit -m "持续进化：$(date -u +%F) 更新" || true
  git push || echo "推送失败"
fi
echo "完成"
