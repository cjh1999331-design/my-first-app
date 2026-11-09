# freqtrade + Claude 策略进化

一个最小可用的骨架：用 [freqtrade](https://www.freqtrade.io) 回测，用 Claude 改写策略代码，
按回测成绩优胜劣汰，一代一代“进化”策略。

```
trading/
├── user_data/
│   ├── config.json                 # 默认 OKX USDT 永续合约、逐仓、dry_run=true
│   └── strategies/
│       ├── EvoSeedStrategy.py      # 现货种子（只做多）
│       └── EvoFuturesSeed.py       # 合约种子（多空，1~10 倍杠杆）
├── evolve/evolve.py                # 进化循环（API 模式）：Claude 改写 -> 回测打分 -> 保留精英
├── evolve/score.py                 # 打分工具（会话模式）：策略由 Claude Code 直接写，脚本只负责检查+回测+记录
├── evo_futures/                    # 第 1 轮合约进化（5 代）的全部策略、记录和结论
├── docker-compose.yml              # 不想装 Python 环境时用 Docker 跑 freqtrade
└── requirements.txt
```

## 1. 安装

```bash
cd trading
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 下载历史数据

```bash
freqtrade download-data --config user_data/config.json --userdir user_data \
  -t 1h 4h --timerange 20240101-
```

> 现在默认是合约模式。想做现货：把 `config.json` 里 `trading_mode` 改成 `spot`、删掉 `margin_mode`，币对去掉 `:USDT` 后缀。

> 默认交易所是 OKX。要换成 Binance 等，改 `config.json` 里的 `exchange.name`。
> 如果你的网络需要代理，新建 `user_data/config.private.json`（已在 .gitignore 里）：
> ```json
> {"exchange": {"ccxt_config": {"httpsProxy": "http://127.0.0.1:7890"},
>               "ccxt_async_config": {"aiohttp_trust_env": true}}}
> ```
> 然后每条 freqtrade 命令后面加 `--config user_data/config.private.json`（evolve.py 也一样，会原样传过去）。

## 3. 回测 / 调参（freqtrade 自带功能）

```bash
# 回测种子策略
freqtrade backtesting --config user_data/config.json --userdir user_data \
  --strategy EvoSeedStrategy --timerange 20240201-20250601

# 参数优化（只调参数，不改逻辑）
freqtrade hyperopt --config user_data/config.json --userdir user_data \
  --strategy EvoSeedStrategy --hyperopt-loss CalmarHyperOptLoss \
  --spaces buy sell --timerange 20240201-20250601 -e 200
```

## 4. 策略进化（Claude 改代码）

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # https://platform.claude.com 获取

# 先检查环境：不调用 Claude，只回测种子
python evolve/evolve.py --dry --train 20240201-20250601 --valid 20250601-20261101

# 正式进化：5 代，每代 2 个子代
python evolve/evolve.py --generations 5 --children 2 \
  --train 20240201-20250601 --valid 20250601-20261101
```

每一代做的事：

1. 取精英池里 fitness 最高的策略当父代
2. 把父代代码 + **训练区间**回测指标 + 最近历史发给 Claude，让它写一个改进版
3. 静态检查（import 白名单、禁止 `exec`/`open`、禁止 `shift(负数)` 这类未来函数）
4. 在训练区间回测，`fitness = 收益% − 1.5 × 最大回撤%`，交易笔数不够直接判负
5. 出错时把报错发回 Claude 修一次；最后按 fitness 保留前 N 名

全部进化结束后，精英池在**验证区间**再跑一次。验证区间在进化过程中从未使用，
它的结果也不会发给 Claude，用来判断有没有过拟合。

结果都在 `evolve/runs/<时间戳>/`：
- `strategies/` 每一代生成的策略文件
- `history.jsonl` 每个候选的指标、改动说明、错误
- `summary.json` 精英池的训练/验证成绩

常用参数：`--elite 4`（精英池大小）、`--min-trades 30`、`--model claude-opus-5-5`、
`--effort high`（low/medium/high/xhigh/max，越高越贵越仔细）。
其余未识别的参数会原样传给 `freqtrade backtesting`。

**费用**：每个子代调用 1~2 次 Claude。5 代 × 2 个子代大约 10~20 次请求。

## 5. 把选中的策略放上模拟盘

```bash
cp evolve/runs/<时间戳>/strategies/Evo_G3_C1.py user_data/strategies/
freqtrade lookahead-analysis --config user_data/config.json --userdir user_data \
  --strategy Evo_G3_C1 --timerange 20250601-20261101    # 再查一遍未来函数
freqtrade trade --config user_data/config.json --userdir user_data --strategy Evo_G3_C1
```

`config.json` 里 `dry_run: true`，不会真下单。

## ⚠️ 注意

- **过拟合是最大的坑。** 训练分高、验证分差很多的策略不要用。不要反复看验证结果再调进化参数——
  那样验证集也会慢慢被“拟合”掉。最好再留一段最近的数据，最后才看一次。
- **生成的代码会在你电脑上执行。** evolve.py 做了基本静态检查，但不是沙箱；在不放重要数据的环境里跑。
- 回测没有完全模拟滑点和流动性，实际表现通常更差。先 dry-run 至少几周，再小资金试。
- 交易所 API Key 只开交易权限，**不要开提币**；密钥写到 `config.private.json`，别提交到 git。
- 这只是研究工具，不构成投资建议。

## 6. 持续进化（部署到云服务器）

`ga/rolling.py` 实现了滚动进化：每天把最新行情加进训练窗口，接着上一次的种群进化几代；
各物种冠军组成“挑战者”先影子交易 28 天，同期收益跑赢当前上线组合才替换；模拟账户只由上线组合交易。
状态保存在 `ga/live/state.json`，每次运行接着上一次继续。

**在一台 Linux 云服务器上（建议香港 / 新加坡 / 日本，2 核 4G 以上）：**

```bash
git clone <你的仓库地址> && cd my-first-app/trading
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 第一次：下载前 100 热门币的历史 K 线（15m 数据较大，约 1 小时）
PAIRS=$(.venv/bin/python -c 'import json;print(" ".join(c["base"]+"/USDT:USDT" for c in json.load(open("ga/universe.json"))["coins"]))')
.venv/bin/freqtrade download-data --config user_data/config.json --userdir user_data \
  -t 15m 1h --timerange 20210101- --candle-types futures --pairs $PAIRS

# 试跑一次
ga/live/daily.sh && tail ga/live/daily.log

# 每天 UTC 01:17 自动运行（crontab -e 加入下面一行）
17 1 * * * cd $HOME/my-first-app/trading && GIT_PUSH=1 ga/live/daily.sh
```

想先看看这套流程在历史上表现如何，可以做“走一步看一步”的模拟（每一步只用当时已有的数据）：

```bash
cd ga && ../.venv/bin/python rolling.py simulate --start 2025-06-01 --step-days 7 --gens 5
```

注意：页面里的模拟账户没有真实下单。要接 OKX 模拟盘真实下单，还需要在服务器上运行 freqtrade 的 dry-run 或 OKX 模拟交易 API。
