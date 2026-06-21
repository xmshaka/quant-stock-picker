# quant-stock-picker 每日开工计划：短线策略重构与买卖点规则

更新时间：2026-06-20 12:55 GMT+8

## 0. 每日开工先读

新一天/新会话开始时，先读本文件，再读相关源码：

```bash
cat docs/DAILY_START_PLAN.md
cat docs/SESSION_HANDOFF.md
cat docs/买卖点基本逻辑.md
cat docs/开仓加仓规则契约.md
cat strategy/schemes.py
cat signals/layers.py
cat market/timing.py
cat backtest/scheme_backtest.py
```

本文件是当前项目策略方向与落地顺序的主计划。任何策略、回测、K线复盘、记录页改动，都先对齐这里的目标和优先级。

---

## 1. 当前状态

### 1.1 回测记录清理

2026-06-19 已将现有回测记录全部软删除：

```text
原有回测记录：45 条
当前可见记录：0 条
软删除目录：data/backtest_runs/.trash
软删除批次后缀：*_20260619_194349
```

说明：采用软删除，不物理删除，避免误删后不可恢复。

### 1.2 最新已修复关键问题

- K线图主方向已切换为 Apache ECharts 单实例多 grid。
- K线图买卖点优先使用 `exec_date`，并保留 `signal_date`。
- 单股回测修复为 T 日信号、T+1 开盘撮合。
- ATR 跟踪止盈必须进入扣成本盈利保护区，且 projected_pnl > 0。
- 不允许亏损交易记录为 `ATR跟踪止盈`。
- P1 已完成：`trend_momentum` / `pullback` / `breakout` 均有策略专属 `resonance_config`，并落盘到回测 `config.json` / 历史页 / 信号页展示。
- P2 已完成两批：单股回测路径接入策略专属 `exit_config`、时间止损、最长持仓退出、策略失败退出、大盘防御减仓；信号页风险退出提示同步使用持仓/观察上下文与 `exit_config`。
- P3 已完成四批：全池 Backtrader 路径接入 P2 退出审计元数据；全池默认 cheat-on-open，信号 T 日收盘后在 T+1 开盘撮合；K线复盘事件源统一为实际成交事件，历史页按 `exec_date` 落点并保留 `signal_date` 审计；补齐全池执行→落盘→读取→K线事件转换闭环测试。
- P4 已完成六批：新增参数网格验证基础模块，支持保守参数空间、策略克隆、结果归一化、低回撤优先排序和 runner 注入式执行；新增轻量 SchemeBacktester runner、CSV/Parquet 输入输出工具和 `scripts/run_param_grid.py` CLI，默认 `max_runs=3` 防止误触发全A重任务；补齐真实 CLI 小样本 smoke，修复 CSV 读取导致股票代码前导 0 丢失的问题；新增参数网格审计目录，输出 `grid_results.csv/parquet`、`config.json`、`summary.md`；新增参数网格结果页面 `dashboard/pages/10_参数网格.py`，读取审计目录并展示实验列表、配置、Top结果和低回撤优先结果表；已真实生成受控小样本审计目录并验证页面辅助函数可读取，`data/grid_results/` 已加入 `.gitignore` 防止产物污染提交。
- P2 退出体系新增可调开关：`enable_market_defense_exit`、`enable_strategy_failure_exit`、`enable_trailing_exit`、`enable_time_stop`、`enable_max_holding_exit`；回测页新增“短线退出规则”面板，可调整开关、最长持仓、时间止损天数、最低收益、策略失败窗口、大盘防御分数、跟踪止盈激活浮盈%和ATR激活倍数；单股回测、全池 Backtrader、信号扫描均尊重开关，配置随回测记录落盘。
- P2 跟踪止盈语义已收紧：必须先达到 `highest >= avg_cost × (1 + trailing_activation_pct)` 或 `highest >= avg_cost + trailing_activation_atr_mult × ATR` 后才激活；默认 `5%` 或 `1ATR`，触发后扣成本盈利才记为 `ATR跟踪止盈`，否则记为 `ATR跟踪回撤止损`。
- 时间止损和最长持仓统一为交易日口径：买入执行日为第 0 个持仓交易日，周末/节假日不计入。
- L4 风险可交易性检查已补齐：信号扫描过滤 OHLC 异常、一字涨跌停、封死涨跌停、低成交额，买入原因追加 `L4可交易性` 审计文本。
- `docs/买卖点基本逻辑.md` 已升级到 v2.0，记录当前 layered 买点、T+1 开盘执行、`signals_executed` K线事件源、P2 退出体系、退出开关、600143 日期排查结论；下一次 agent 必须优先读取。
- `docs/开仓加仓规则契约.md` 已新增：明确买卖点专业化原则、置信度必须进入开仓/加仓执行契约、禁止围绕单个 run 拟合、`balanced` 长期应组合器化；下一次涉及买点/加仓/策略调整必须先读。
- `docs/SESSION_HANDOFF.md` 已新增：作为跨会话/跨 agent 的固定交接文件。每次重要收尾必须写入“已完成、测试、未提交变更、已知问题、下一步、禁止回退”，避免最后报告只留在聊天里丢失。
- 当前常规测试结果：`.venv/bin/pytest tests -q` 为 `296 passed, 6 skipped`。
- 根目录 `pytest -q` 被历史诊断脚本 `direct_dataflow_test.py` 阻塞，非当前修复引入。

---

## 2. 项目目标

项目定位：

```text
基于量化因子的 A 股短线择股择时系统
持仓周期：≤20 个交易日
覆盖范围：全 A 符合过滤条件股票
交易方式：不接自动交易
核心目标：低回撤、稳定小幅盈利、全链路可审计
```

2026-06-21 纠偏：技术指标（MACD/KDJ/RSI/BOLL/MA）只能作为买卖点确认项，不能长期作为主驱动。允许引入新的量化因子、市场环境、资金流、情绪和相对换手判断，以提高买卖点准确性；但禁止未来函数、禁止围绕单个 run 拟合、禁止回测美化。若执行中偏离“基于量化因子的短线择股择时”目标，必须在本计划中及时记录和纠偏。

核心原则：

1. 低回撤优先，稳定小幅盈利优先。
2. 撮合用不复权 OHLCV，趋势展示可用前复权。
3. 禁止未来函数：信号 T 日收盘生成，最早 T+1 开盘成交。
4. 回测必须内置佣金、印花税、过户费、滑点。
5. 买卖点必须保留 `signal_date` 与 `exec_date`。
6. 历史回测记录不静默改写，旧错误记录只能通过重新回测生成新记录修正。
7. 买卖点改造必须先有专家规则契约，再改代码与回测；禁止围绕单个 run 的胜负事后拟合。
8. `TradePoint.confidence` 是开仓/加仓执行契约的一部分，不是仅用于展示；低置信度信号不能与高置信度信号同等开仓。
9. 不允许“指标罗列触发”：必须先定义买点类型、主触发、确认项、否决项、风险标签和缺失数据，再决定是否交易。
10. 买卖点应以“因子证据 + 市场环境 + 资金/情绪/换手上下文 + 技术确认 + 风控否决”组合判断；数据缺失必须落入 `missing_fields`，不得假装已验证。

---

## 3. 当前架构

```text
数据层
  ├─ 不复权 OHLCV：撮合/成本/风控
  ├─ 前复权 OHLCV：趋势显示/技术指标可选
  └─ 因子快照：全A截面排序

股票池过滤
  ├─ ST / 退市 / 停牌 / 北交所过滤
  ├─ 成交额过滤
  ├─ 涨跌停距离过滤
  ├─ 市值过滤
  └─ 数据质量过滤

大盘择时
  ├─ 防御 10%
  ├─ 低仓 30%
  ├─ 中等 50%
  ├─ 高仓 70%
  └─ 满仓 90%

因子评分
  ├─ trend_momentum
  ├─ pullback
  ├─ breakout
  └─ balanced

信号层
  ├─ L1 趋势过滤
  ├─ L2 策略形态匹配
  ├─ L3 多条件共振
  └─ L4 风险可交易性检查

执行层
  ├─ T日收盘信号
  ├─ T+1开盘撮合
  ├─ 信号置信度 confidence → 开仓/加仓门槛与仓位调制（待重接入）
  ├─ 成本/滑点四项审计
  └─ A股100股整数手

退出层
  ├─ ATR硬止损
  ├─ ATR盈利保护型跟踪止盈
  ├─ 固定止盈
  ├─ 时间止损（交易日口径）
  ├─ 策略失败退出
  └─ 大盘降档减仓
```

---

## 4. 当前已有策略与买卖点规则

### 4.1 四套策略方案

源码：`strategy/schemes.py`

| scheme_id | 名称 | 定位 | 当前问题 |
|---|---|---|---|
| `trend_momentum` | 强势追涨 | 强势上涨市，追趋势延续 | L3 共振仍和其他策略共用，风格边界不够硬 |
| `pullback` | 回调低吸 | 上升趋势中回调买入 | 仍可能买到弱势下跌中的便宜，需要更强支撑/缩量/大盘条件 |
| `breakout` | 横盘突破 | 横盘后放量突破 | 缺少假突破过滤与突破失败退出 |
| `balanced` | 均衡择时 | 全行情适配 | 不应作为独立信号策略，建议改为多策略组合器 |

### 4.2 三层信号主路径

源码：`signals/layers.py`

#### Layer 1：趋势过滤

当前逻辑：

```text
非 pullback:
  Price > MA20
  MA20 > MA40
  ADX > 20
  非20日最低

pullback / balanced:
  近10日曾经 Price > MA20
  MA20 > MA40 加分
  Price < MA20 允许，视为回调中
  ADX > 20
  非20日最低

通过阈值：score >= 0.4
```

#### Layer 2：策略形态匹配

| 策略 | 当前形态 |
|---|---|
| `trend_momentum` | 距20日高点 <5%，5日动量 >1%，20日动量 >2% |
| `pullback` | 距20日高点回撤 >5%，且 RSI <40 或回撤 >10% |
| `breakout` | 横盘振幅 <8%，突破前期区间 1%，量比 >1.3 |
| `balanced` | 任一形态匹配即可 |

#### Layer 3：多条件共振

买入 6 条，当前默认满足 ≥2：

```text
1. RSI < 40
2. MA5 金叉 MA20
3. MACD 翻红
4. 布林位置 < 0.3
5. 放量 > 1.2x
6. KDJ 金叉且 K < 50
```

卖出 6 条，当前默认满足 ≥2：

```text
1. RSI > 70
2. MA5 死叉 MA20
3. MACD 翻绿
4. 布林位置 > 0.8
5. 放量下跌
6. KDJ 死叉且 K > 50
```

### 4.3 当前交易执行与风控

源码：`backtest/scheme_backtest.py`

```text
T日信号 → T+1 开盘价执行
买入滑点：open × (1 + slippage)
卖出滑点：open × (1 - slippage)
```

成本：

```text
佣金：双向万2.5，最低5元
印花税：卖出千1
过户费：双向万0.001
滑点：
  成交额 >5亿：0.002
  1亿-5亿：0.005
  <1亿：0.010
```

仓位：

```text
初始资金：100万
单票上限：硬限制 20%
总仓位：大盘择时映射 10% / 30% / 50% / 70% / 90%
```

退出：

```text
1. ATR硬止损
2. ATR跟踪止盈：必须进入扣成本盈利保护区且 projected_pnl > 0
3. ATR固定止盈：projected_pnl > 0
4. 信号卖出
5. 末日清仓
```

---

## 5. 主要问题判断

### 问题 A：策略数量够，但风格边界不够硬

当前 4 个策略最终仍共用 L3 共振条件，导致不同策略可能被同一批 RSI/MACD/KDJ 条件触发。

修复方向：每个策略单独定义必要条件、共振权重、退出模板。

### 问题 B：pullback 仍可能买到弱势下跌

需要增加：

```text
不破 MA40 / 不破前低
缩量回调优先
回调后首次放量转强
大盘不能处于防御档
```

### 问题 C：退出体系缺少短线必需项

必须增加：

```text
时间止损
策略失败退出
大盘降档减仓
突破失败退出
回调破位退出
```

### 问题 D：单股模式和全池模式执行逻辑仍需统一

要统一：

```text
signal_date / exec_date
T+1 开盘撮合
止盈止损
退出原因枚举
成本审计字段
K线复盘事件来源
```

---

## 6. 推荐策略重构方案

### 6.1 强势追涨 `trend_momentum`

定位：买强势延续，不买超跌反弹。

必要条件：

```text
大盘评分 >= 60
个股 close > MA20
MA20 > MA40
20日新高距离 <= 5%
近5日涨幅 > 1%
成交额 > 1亿
```

买入共振至少满足 3 条：

```text
1. 放量 > 1.3
2. MACD翻红
3. MA5 > MA20
4. 收盘价接近20日高点
5. 相对强度 > 市场中位数
6. KDJ未极端超买，K < 85
```

退出：

```text
硬止损：2ATR
跟踪止盈：最高价 - 2ATR，且必须进入盈利保护区
时间止损：5日内未盈利 >2%，退出
最大持仓：10日
```

### 6.2 回调低吸 `pullback`

定位：上升趋势中的健康回调，不抄弱势底。

必要条件：

```text
大盘评分 >= 40
近10日曾经站上 MA20
MA20 >= MA40 或 MA40走平
当前回撤 5%-15%
不破近20日低点
回调日缩量优先
```

买入共振至少满足 3 条：

```text
1. RSI < 40
2. 布林位置 < 0.3
3. 回撤 5%-15%
4. 当日不创新低
5. 缩量回调：量比 < 1.0
6. KDJ金叉或 J值拐头
```

退出：

```text
硬止损：跌破买入低点或 2ATR
固定止盈：2.5ATR
跟踪止盈：盈利保护后 high - 2ATR
时间止损：7日未盈利退出
最大持仓：15日
```

### 6.3 横盘突破 `breakout`

定位：横盘压缩后的放量突破。

必要条件：

```text
大盘评分 >= 50
过去10-15日振幅 < 8%
突破前高 1%
量比 > 1.3
非涨停封死，仍可成交
```

买入共振至少满足 3 条：

```text
1. 价格突破平台上沿
2. 量比 > 1.5
3. MACD翻红
4. MA5 > MA20
5. 布林带宽处于低位后扩张
6. 收盘价位于日内高位附近
```

退出：

```text
失败退出：2日内跌回平台内
硬止损：平台下沿或 2ATR
固定止盈：3ATR
跟踪止盈：盈利保护后 high - 2ATR
最大持仓：10日
```

### 6.4 均衡择时 `balanced`

建议不再作为独立买卖点策略，而是作为组合器：

```text
trend_score
pullback_score
breakout_score
取最高分，但必须超过阈值
```

买入条件：

```text
大盘评分 >= 50
任一策略子评分 >= 70
L3共振 >= 3/6
成交额/涨跌停/数据质量通过
```

退出：

```text
按实际触发的子策略使用对应退出模板
```

---

## 7. 建议新增统一交易状态机

### 7.1 PositionState

```python
PositionState:
    symbol
    entry_signal_date
    entry_exec_date
    entry_price
    avg_cost
    shares
    highest
    lowest
    atr
    stop_loss
    take_profit
    trailing_stop
    trailing_active
    strategy_id
    entry_reason
    holding_days
```

### 7.2 ExitDecision

```python
ExitDecision:
    signal_date
    exec_date
    exit_type
    reason
    expected_price
    exec_price
    projected_pnl
    realized_pnl
    cost_breakdown
```

### 7.3 退出优先级

```text
1. 一字跌停/不可成交检查
2. 硬止损
3. 大盘防御强制降仓
4. 策略失败退出
5. ATR盈利保护型跟踪止盈 / 跟踪回撤止损
6. 固定止盈
7. 时间止损 / 最长持仓
8. 信号卖出
9. 末日清仓
```

---

## 8. 落地顺序

### P0：统一交易口径

```text
1. 单股模式 / 全池模式统一 signal_date + exec_date
2. 全部成交统一 T+1 开盘撮合
3. 全部退出原因统一枚举，不再自由字符串
4. 回测记录新增 exit_type / exit_subtype / trigger_price / projected_pnl
5. 历史页展示成交日、信号日、退出类型、真实盈亏
```

### P1：策略专属共振参数化

```text
1. 给每个 scheme 增加 resonance_config
2. 不同策略使用不同必要条件和权重
3. balanced 改为多策略打分器，不再任意匹配
```

### P2：短线退出体系补齐

```text
1. 时间止损
2. 策略失败退出
3. 大盘降档减仓
4. 突破失败退出
5. 回调破位退出
```

### P3：全池回测统一执行模型

```text
1. Backtrader 执行点和单股执行点统一 schema
2. 成本、滑点、退出原因完全一致
3. K线复盘永远使用统一 executed events
```

### P4：参数网格与验证

```text
按策略分别做：
  stop_loss_atr_mult
  take_profit_atr_mult
  trailing_atr_mult
  max_holding_days
  time_stop_days
  trailing_activation_pct
  trailing_activation_atr_mult
  min_confirmations
  market_score_threshold

输出：
  年化收益
  最大回撤
  胜率
  盈亏比
  平均持仓天数
  单票集中度
  换手率
```

2026-06-20 补充：`balanced` 已纳入参数网格 CLI 白名单和默认参数空间。小样本 `max_runs=4` 会优先覆盖 `(max_holding_days, time_stop_days)` 的 `(15,7)/(15,10)/(20,7)/(20,10)` 四组组合，并包含 `trailing_activation_pct` 与 `trailing_activation_atr_mult` 审计字段。已生成审计目录 `data/grid_results/20260620_150523_balanced`。

---

## 9. 每日启动检查清单

每次继续开发前执行：

```bash
cd /root/.openclaw/workspace/quant-stock-picker
cat docs/DAILY_START_PLAN.md
cat docs/SESSION_HANDOFF.md
git status --short
pytest tests/test_backtest_records_p0.py tests/test_kline_chart_regressions.py -q
```

如果要动交易逻辑，必须额外检查：

```bash
pytest tests/test_backtest_engine_p1.py tests/test_backtest_records_p0.py -q
```

如果要动 K线/回测记录页，必须额外检查：

```bash
pytest tests/test_kline_cache_isolation.py tests/test_kline_chart_regressions.py -q
```

---

## 10. 关键提醒

- 不要回到 Plotly K线方案。
- 当前 K线主方案是 ECharts 单实例多 grid。
- 不要把 signal_date 当作实际成交日期。
- 不要用 T+1 收盘价模拟开盘成交。
- 不要把亏损交易归因为止盈。
- 不要静默改写旧历史 run。
- 默认一天汇总提交一次 Git，不要每个小修立即提交，除非用户明确要求。

---

## 11. 2026-06-21 数据层可行性验证：资金流 / 大单 / 相对换手

已做极小样本 Tushare 验证，不改策略代码：

```text
Tushare token: settings.tushare_token 可用，长度 56
Tushare version: 1.4.29
pro.moneyflow(trade_date='20260618'): 返回 5188 行
字段包含：buy_sm/md/lg/elg_vol/amount、sell_sm/md/lg/elg_vol/amount、net_mf_vol、net_mf_amount
与本地 data/daily_factors/factors_20260618.parquet 按 symbol 合并：4480/4480 覆盖，coverage=1.0000
pro.moneyflow_hsgt(start_date='20260618', end_date='20260618'): 返回 north_money/south_money 等字段
pro.stk_factor(trade_date='20260618'): 返回 5507 行，包含 Tushare 计算的 MACD/KDJ/RSI/BOLL/CCI 等字段，可作校验源，不作为未来函数
```

结论：

```text
1. 个股主力/大单/超大单资金流在 Tushare moneyflow 源可用。
2. 大盘北向资金流 moneyflow_hsgt 可用。
3. 本地 daily_factors 目前仅有 turnover_ratio / north_hold_change / margin_change 等基础字段，尚未接入 moneyflow 大单字段。
4. 换手率应优先做相对换手因子，而非只用绝对换手率：relative_turnover_5d、relative_turnover_20d、turnover_percentile_60d。
```

后续落地顺序：

```text
P0: 核实 resonance_config key 与 L3 ConditionResult key 是否一致，避免策略专属共振失效。=> 2026-06-21 已复现并修复，补测试。
P1: 核实全池 Backtrader 单票 20% 硬限制；目标股票不足时不得把单票推到超过 20%。=> 2026-06-21 已复现并修复，补测试。
P2: 增加资金流数据缓存/因子：main_net_mf_amount、large_net_mf_amount、elg_net_mf_amount、*_pct_amount、*_rank。=> 2026-06-21 已完成第一阶段数据层接入，不硬过滤。
P3: 增加相对换手因子：relative_turnover_5d、relative_turnover_20d、turnover_percentile_60d、amount_percentile_60d。=> 2026-06-21 已完成第一阶段；`amount_percentile_60d` 依赖历史 amount，当前 price snapshot 缺 amount 时保留缺失。
P4: 买点结构化输出：entry_model、main_trigger、factor_evidence、market_context、fund_flow_context、technical_confirmations、veto_checks、risk_tags、missing_fields。=> 2026-06-21 已完成第一阶段字段、落盘和 layered BUY 基础审计；后续需填充真实因子/资金/市场上下文。
P5: 开仓/加仓契约改造：加仓必须盈利、同模型、结构未破、confidence 更强、加仓后单票 <=20%。=> 2026-06-21 已完成单股加仓执行契约第一阶段；后续补端到端回测用例与前端审计展示。
```

