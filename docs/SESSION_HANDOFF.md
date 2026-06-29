## 2026-06-28 12:05 数据流审计 + Stage 2 准备

### 一、数据流架构审计

#### 1.1 根因：因子数据未传入信号层

回测调用链：
```
SchemeBacktester.run()
  ├─ factor_df (27列: 因子+资金流+换手)   ← 用于截面打分
  ├─ price_df  (8列: OHLCV)              ← 用于K线
  └─ _run_single_stock_backtest()
       └─ evaluate_layered(sym_bars, ...)  ← sym_bars 来自 price_df → 只有 OHLCV!
            ├─ L3 ResonanceChecker._get_moneyflow_values(bars, idx)
            │    └─ _get_factor_value(bars, idx, 'main_net_mf_amount') 
            │         → bars.columns 中没有此列 → fallback 0.0 → 条件永远不满足
            └─ _get_turnover_values(bars, idx) → 同样问题
```

**问题**：factor_df 有27列（含资金流/换手/排名因子），price_df 只有8列（OHLCV）。
evaluate_layered 通过 bars.columns 读取因子值，但传入的 bars 来自 price_df，
从未合并 factor_df 的因子列。导致 L3 全部资金流+换手条件 fallback 为默认值(0.0/0.5)。

**修复**：新增 `_merge_factor_columns()` 辅助函数，在 evaluate_layered 调用前
按 symbol + trade_date 将 factor_df 的因子列 merge 到 sym_bars。

#### 1.2 全池路径冗余K线拉取

全池回测中 `_fetch_ohlcv_for_backtest` 逐只拉取 856 只股票K线，
但 price_df 已含完整 OHLCV（open/high/low/close/volume）。
修复：price_df 已含 OHLCV 时跳过逐只拉取。

#### 1.3 执行时序问题（已确认）

06-27 执行顺序：回测(23:28) → 代码变更(00:12-00:16) → 快照完成(00:43)
回测用的是旧代码(entry_contract=True) + 旧因子数据 → 全部被 observe_only 拦截。

### 二、昨天 3 阶段计划进度

| 阶段 | 内容 | 状态 |
|------|------|:--:|
| 1 | condition_count 字段 + entry_contract 默认 False | ✅ 完成 |
| 2 | 关闭 entry_contract，全池回测，按 condition_count 分组统计后验胜率 | ⏳ 待执行 |
| 3 | 基于后验数据设 entry 过滤规则（非 confidence 阈值） | ⏳ 未开始 |

### 三、今日代码变更

| 文件 | 变更 | 理由 |
|------|------|------|
| backtest/scheme_backtest.py | 新增 `_merge_factor_columns()` | P0修复：因子数据传入信号层 |
| backtest/scheme_backtest.py | price_df 已有 OHLCV 时跳过逐只拉取 | 性能优化 |
| backtest/scheme_backtest.py | `_fetch_ohlcv` 改用 CacheManager+Fallback | 统一数据源 |

### 四、当前数据状态

- factors_20260628.parquet: 337,259行 × 27列，78个交易日（2026-01-29~2026-06-26），4449只股票
- prices_20260628.parquet: 426,239行 × 8列
- 资金流覆盖率: 3-6月全部100%（3月79,691/79,691, 4月93,134/93,134, 5月79,977/79,977, 6月84,457/84,457）
- main_net_mf_rank / large_elg_net_mf_rank: 同样100%

### 五、下一步

Stage 2 执行：
1. `enable_entry_confidence_contract=False`
2. 单股/全池回测（_merge_factor_columns 已就绪）
3. 按 condition_count 分组统计后验胜率/收益/盈亏比
4. 用后验数据反推 Stage 3 的 entry 过滤规则

### 禁止回退
- 不允许删除 `_merge_factor_columns` 及其调用
- 不允许恢复 condition_count → confidence 的 entry_contract 逻辑（等 Stage 3 后验数据）

## 2026-06-28 23:32 Dashboard 参数统一重构

### 一、Git 提交（已完成）
- 20 files changed, +1505/-1836

### 二、Dashboard 重构计划

**目标**: 所有影响交易的参数所见即所得，一个页面完整控制

**Phase 1 — 重构 8_策略回测.py**
- 新增参数区：因子权重、L3共振条件、开仓契约(min_entry_condition_count)、ATR止盈止损、仓位管理、大盘择时

**Phase 2 — 删除冗余页面** ✅ 已完成 (2026-06-28)
- ✅ 删除 7_策略方案.py（signal_rules旧逻辑已废弃）
- ✅ 删除 策略条件总览.py（内容并入8）
- ✅ 10_参数网格.py 保留但标注 CLI-only

### 三、关键发现
- min_entry_condition_count 直接决定信号是否执行，当前在全部5个Dashboard页面中完全不可见