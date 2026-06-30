# 信号引擎统一方案

> 创建日期: 2026-06-30  
> 状态: 方案评估阶段  
> 驱动: Scanner/Backtest 0% 一致率根因分析

---

## 1. 问题定义

Scanner（量化选股/截面模式）和 Backtest（策略回测/时序模式）输出完全不同的信号。根因不是数据时效差异（资金流 T+1），而是**两套独立的三层过滤实现**，仅在过热检测这一个模块共享代码。

## 2. 现状：两条路径对比

### 2.1 调用链

```
量化选股 (Scanner)                     策略回测 (Backtest)
─────────────────────────────────      ─────────────────────────────
signals/scanner.py::scan_signals()     scheme_backtest.py::
                                         _run_single_stock_backtest()
  ├─ _factor_scores()                    ├─ _score_day() (纯z-score)
  │   rank*0.6 + z*0.4                  │
  ├─ _check_layer1() [内联]             ├─ layers.py::TrendFilter.check()
  ├─ _check_layer2() [内联]             ├─ layers.py::StrategyMatcher.match()
  ├─ _check_layer3() [内联]             ├─ layers.py::ResonanceChecker.check_buy()
  ├─ check_overheat()                   ├─ check_overheat()  ← 唯一共享
  │   └→ signals.layers                  │   └→ 同一个函数
  ├─ _check_l4_tradability()            ├─ (无L4 — 模拟循环等效)
  └─ 置信度合成:                          └─ 置信度合成:
     f*0.35+l1*0.20+l2*0.25+l3*0.20       l1*0.25+l2*0.35+l3*0.40
```

### 2.2 差异清单

| 维度 | Scanner | Backtest | 是否共享 |
|------|---------|----------|----------|
| L1 趋势过滤 | `_check_layer1()` 内联 | `TrendFilter.check()` | ❌ |
| L2 策略匹配 | `_check_layer2()` 内联 | `StrategyMatcher.match()` | ❌ |
| L3 共振确认 | `_check_layer3()` 内联 | `ResonanceChecker.check_buy()` | ❌ |
| 过热检测 | → `layers.check_overheat()` | → `layers.check_overheat()` | ✅ |
| L4 可交易性 | 有 | 无（模拟循环等效） | ❌ |
| 因子评分 | rank×0.6 + z×0.4 | 纯 z-score | ❌ |
| 置信度合成 | 含 factor_score | 不含 factor_score | ❌ |
| Balanced | 顶层组合器 | ResonanceChecker 内部路由 | ❌ |

### 2.3 根因

Scanner 的 `_check_layer1/2/3` 是独立内联实现，与 `signals/layers.py` 中的 `TrendFilter`/`StrategyMatcher`/`ResonanceChecker` 是两套代码。**同一个策略规格在两个模块里各自编码了一次**。6 条信号生成路径中仅 1 条（过热检测）真正共享，自然产生不同结果。

---

## 3. 目标架构

```
BUILTIN_SCHEMES (factor_weights / resonance_config / exit_config)
        │
        ▼
  统一信号生成引擎 (L1+L2+L3+Factor+Overheat+L4)
        │
   ┌────┴────┐
   ▼         ▼
量化选股    策略回测
截面模式    时序模式
全A×1日    单股×N日
```

**核心原则：同一引擎、同一配置、同一逻辑。** 区别仅运行模式（截面 vs 时序）。

---

## 4. 统一方案（三步走）

### Step 1：统一 L1+L2 —— 低风险

两版 L1/L2 逻辑几乎一致（见差异清单），差异主要在调用方式，不在算法。

**改动：**

Scanner 的 `_check_layer1()` → 改为调用 `layers.py::TrendFilter.check()`  
Scanner 的 `_check_layer2()` → 改为调用 `layers.py::StrategyMatcher.match()`

**验证：**
- Scanner Top5 列表不变
- 4策略×60天回测绩效不变

**风险：低。** L1/L2 算法在两版中高度一致，主要是消除重复代码。

---

### Step 2：统一 L3 + 因子评分 + 置信度公式 —— 中风险

L3 差异最大：Scanner 用线性强度映射，Backtest 用置信度加权。因子评分算法也不同。

**改动：**

以 Scanner 版为基准（更专业：rank+z-score 混合评分优于纯 z-score）：

1. 因子评分 → Scanner 的 rank×0.6 + z×0.4 植入 `SchemeBacktester._score_day()`
2. L3 条件 → Scanner 的线性强度逻辑植入 `layers.py::ResonanceChecker`
3. 置信度公式 → 统一为 `factor×0.35 + l1×0.20 + l2×0.25 + l3×0.20`（含 factor_score）
4. 移除 `ResonanceChecker` 中旧的条件生成逻辑

**验证：**
- 4策略×60天回测绩效对比（统一前后）
- Scanner Top5 列表不变

**风险：中。** L3 条件集合需要逐策略核对，确保统一后策略绩效不退化。

---

### Step 3：统一 L4 + Balanced 路由 —— 架构级

L4 可交易性和 Balanced 策略在两个模块中是两种完全不同设计。

**改动：**

1. **L4 可交易性**：Scanner 的 `_check_l4_tradability()` 逻辑融入回测模拟循环
   - 当前 Backtest 已有等效逻辑（ATR 止损/时间止损/策略失败退出）
   - 需要对齐的条件：涨跌停距离、停牌检测、最小成交额
2. **Balanced 策略**：裁决 Scanner 版（顶层组合器）vs Backtest 版（内部路由器）
   - MEMORY.md 已记录：Balanced 长期定位应为组合器/路由器，不应作为独立策略
   - 倾向保留 Scanner 版：作为组合器跑 3 策略取最优

**验证：**
- 全量回归：4策略×60天回测 + Scanner 全A Top5
- Dashboard 所有页面数据一致

**风险：中高。** Balanced 设计变更影响面大，需独立评估。

---

## 5. 不变性原则

统一过程遵循：

1. **因子评分不变性**：全A统一打分，得分不随池子范围变化
2. **不复权混用**：撮合/信号生成用不复权，趋势展示用前复权
3. **成本硬编码**：佣金+印花税+过户费+滑点必须在所有路径生效
4. **NaN 阻断**：关键字段空值率 >20% 阻断信号生成
5. **每步可独立验证**：Step 完成 → 回测对比 → 确认 → 下一步

---

## 6. 执行前置条件

- [ ] 确认当前 4策略×60天回测基准数据（统一前快照）
- [ ] 确认 Scanner Top5 基准列表（统一前快照）
- [ ] 所有改动分支在统一前完成 Git 提交

---

## 7. 附录：关键文件索引

| 文件 | 职责 |
|------|------|
| `signals/scanner.py` | 量化选股截面扫描（含内联 L1/L2/L3） |
| `signals/layers.py` | 三层过滤类（TrendFilter/StrategyMatcher/ResonanceChecker） |
| `signals/engine.py` | SignalEngine 封装层 |
| `backtest/scheme_backtest.py` | 方案回测引擎（含 `_run_single_stock_backtest`） |
| `backtest/engine.py` | BacktestEngine（资金/仓位/撮合） |
| `strategy/schemes.py` | BUILTIN_SCHEMES 策略配置定义 |
| `market/timing.py` | 大盘择时模型 |
