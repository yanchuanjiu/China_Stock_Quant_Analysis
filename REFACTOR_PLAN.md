# 项目重构计划

## 一、当前状态分析

### 1.1 问题汇总

| 问题 | 说明 | 影响 |
|------|------|------|
| **脚本混乱** | 根目录 17 个 Python 脚本，功能重叠 | 难以维护、容易出错 |
| **实验目录分散** | `experiments/`, `experiments_full/`, `mlruns/` 三处存储 | 结果难以追溯 |
| **配置散乱** | 参数硬编码在各脚本中 | 策略对比困难 |
| **缺少统一入口** | 每次运行需找对应脚本 | 使用门槛高 |
| **交易信号未独立** | 信号生成与回测耦合 | 无法对接交易平台 |

### 1.2 当前脚本分析

| 脚本 | 功能 | 保留建议 |
|------|------|----------|
| `run_volume_factor_backtest.py` | 量比因子回测 | ✅ 保留，整合 |
| `run_qlib_full_backtest.py` | 完整回测 | ✅ 保留，整合 |
| `run_consistent_backtest.py` | 一致性时间回测 | ❌ 整合到主程序 |
| `run_full_analysis.py` | 全面分析 | ❌ 整合到主程序 |
| `run_paper_trading.py` | 模拟交易 | ✅ 改造为信号服务 |
| `update_data_baostock.py` | 数据更新 | ✅ 保留 |
| `visualize_results.py` | 可视化 | ✅ 整合到报告模块 |
| `debug_*.py`, `diagnose_*.py` | 调试脚本 | ❌ 删除 |
| `test_*.py` (根目录) | 临时测试 | ❌ 移到 tests/ |

---

## 二、目标架构

### 2.1 目录结构

```
QT_China/
├── configs/                    # 策略配置（YAML）
│   ├── strategies/
│   │   ├── alpha158_lgb.yaml
│   │   ├── alpha158_xgb.yaml
│   │   ├── volume_factor_lgb.yaml
│   │   └── custom_factor.yaml
│   ├── backtest.yaml           # 回测参数
│   └── trading.yaml            # 交易参数
│
├── src/
│   ├── data/                   # 数据模块
│   │   ├── __init__.py
│   │   ├── fetcher.py          # Baostock/Akshare 数据获取
│   │   ├── processor.py        # 数据处理
│   │   └── updater.py          # 每日更新
│   │
│   ├── factors/                # 因子模块
│   │   ├── __init__.py
│   │   ├── alpha158.py         # Alpha158 因子
│   │   ├── volume.py           # 量比因子
│   │   └── custom.py           # 自定义因子
│   │
│   ├── models/                 # 模型模块
│   │   ├── __init__.py
│   │   ├── lightgbm.py
│   │   ├── xgboost.py
│   │   └── ensemble.py         # 集成模型
│   │
│   ├── backtest/               # 回测模块
│   │   ├── __init__.py
│   │   ├── engine.py           # 回测引擎
│   │   ├── metrics.py          # 评估指标
│   │   └── report.py           # 报告生成
│   │
│   ├── signals/                # 信号模块 ⭐
│   │   ├── __init__.py
│   │   ├── generator.py        # 信号生成
│   │   ├── validator.py        # 信号验证
│   │   └── publisher.py        # 信号发布（API/文件）
│   │
│   └── trading/                # 交易模块
│       ├── __init__.py
│       ├── simulator.py        # 模拟交易
│       └── connector.py        # 交易平台接口
│
├── output/                     # 输出目录
│   ├── signals/                # 交易信号
│   ├── reports/                # 分析报告
│   └── logs/                   # 运行日志
│
├── scripts/                    # 入口脚本
│   ├── run_backtest.py         # 主回测入口
│   ├── run_signal.py           # 信号生成入口
│   ├── update_data.py          # 数据更新入口
│   └── compare_strategies.py   # 策略对比入口
│
├── tests/                      # 测试
├── docs/                       # 文档
├── requirements.txt
├── README.md
└── process.md
```

### 2.2 核心模块设计

#### 策略配置 (YAML)
```yaml
# configs/strategies/volume_factor_lgb.yaml
strategy:
  name: "VolumeFactor_LightGBM"
  description: "基于量比因子的LightGBM策略"

data:
  instruments: "csi300"
  train_start: "2023-01-01"
  train_end: "2025-08-31"
  test_start: "2025-09-01"
  test_end: "2025-11-25"

factors:
  type: "alpha158"  # 或 "volume" / "custom"
  # 量比因子权重增强
  volume_boost: 1.5

model:
  class: "LGBModel"
  params:
    learning_rate: 0.03
    max_depth: 10
    num_leaves: 256
    feature_fraction: 0.8

backtest:
  account: 100000
  topk: 10
  n_drop: 2
  exchange:
    deal_price: "open"
    open_cost: 0.0003
    close_cost: 0.0003
    impact_cost: 0.001
    trade_unit: 100
```

#### 信号输出格式
```json
// output/signals/2025-11-27.json
{
  "date": "2025-11-27",
  "strategy": "VolumeFactor_LightGBM",
  "signals": [
    {
      "stock": "SZ002459",
      "action": "BUY",
      "weight": 0.12,
      "score": 0.85,
      "price_ref": 1.05,
      "target_amount": 9500
    },
    {
      "stock": "SH600000",
      "action": "SELL",
      "weight": 0,
      "score": -0.32,
      "current_holding": 1000
    }
  ],
  "portfolio_summary": {
    "total_buy_value": 50000,
    "total_sell_value": 32000,
    "cash_required": 18000
  }
}
```

---

## 三、达成目标路线图

### 目标：年化收益 > 50%, Sharpe > 2

### 3.1 第一阶段：代码重构 (1周)

| 任务 | 说明 | 优先级 |
|------|------|--------|
| 整理目录结构 | 按上述架构重组 | P0 |
| 删除冗余脚本 | 保留核心功能 | P0 |
| 统一配置管理 | YAML 配置化 | P0 |
| 信号模块独立 | 与回测解耦 | P0 |

### 3.2 第二阶段：策略优化 (2周)

| 任务 | 目标 | 方法 |
|------|------|------|
| **因子工程** | 提升因子有效性 | - 量比因子深度优化<br>- 动量/反转因子组合<br>- 行业轮动因子 |
| **模型调优** | 提升预测精度 | - 超参数搜索<br>- 集成学习<br>- 时序交叉验证 |
| **风控增强** | 降低回撤 | - 动态止损<br>- 仓位管理<br>- 波动率过滤 |

### 3.3 第三阶段：信号系统 (1周)

| 任务 | 说明 |
|------|------|
| 信号生成服务 | 每日自动生成买卖信号 |
| 信号验证 | 回测信号有效性 |
| 信号发布 | JSON/API/邮件通知 |

### 3.4 第四阶段：交易对接 (2周)

| 任务 | 说明 |
|------|------|
| 模拟盘对接 | 同花顺/雪球模拟盘 |
| 实盘准备 | 券商API对接准备 |
| 风险监控 | 实时仓位/收益监控 |

---

## 四、策略优化方向

### 4.1 当前瓶颈分析

从最新回测结果看：
- **最佳策略**: VolumeFactor_LightGBM (-2.90%)
- **基准**: Buy & Hold (-5.98%)
- **超额收益**: +3.08%

**问题**: 测试期市场整体下跌，绝对收益为负

### 4.2 提升方向

#### A. 因子优化
```
1. 量比因子增强
   - VSUMD (量比RSI) 动量突破
   - WVMA 波动率过滤
   - CORR/CORD 量价背离信号

2. 组合因子
   - 动量因子: ROC, RSI
   - 价值因子: PB, PE (需基本面数据)
   - 技术因子: MACD, KDJ

3. 市场状态因子
   - 市场情绪指数
   - 波动率指数 (VIX类似)
   - 资金流向
```

#### B. 模型优化
```
1. 集成学习
   - Stacking: LightGBM + XGBoost + CatBoost
   - Blending: 加权平均

2. 深度学习
   - LSTM 序列建模
   - Transformer 注意力机制

3. 强化学习
   - PPO/A2C 策略优化
```

#### C. 风控优化
```
1. 动态止损
   - ATR止损
   - 移动止损

2. 仓位管理
   - Kelly公式
   - 风险平价

3. 择时
   - 趋势跟踪
   - 均值回归
```

---

## 五、实施计划

### 立即执行 (今天)

1. **清理冗余文件**
   ```bash
   # 删除调试脚本
   rm debug_network.py diagnose_network_and_data.py
   rm test_akshare_connection.py
   
   # 整理实验目录
   rm -rf experiments_full/ mlruns/
   ```

2. **创建新目录结构**
   ```bash
   mkdir -p src/{data,factors,models,backtest,signals,trading}
   mkdir -p configs/strategies
   mkdir -p output/{signals,reports,logs}
   mkdir -p scripts docs
   ```

3. **整合核心脚本**
   - 合并 `run_*.py` 为统一入口

### 本周完成

1. 完成代码重构
2. 策略配置 YAML 化
3. 信号生成模块

### 下周目标

1. 因子优化实验
2. 模型集成
3. 达成 Sharpe > 1.5

---

## 六、监控指标

| 指标 | 当前值 | 目标值 | 期限 |
|------|--------|--------|------|
| 年化收益率 | -33.8% | **>50%** | 4周 |
| Sharpe Ratio | -1.40 | **>2** | 4周 |
| 最大回撤 | -5.79% | <15% | 4周 |
| 胜率 | ~50% | >55% | 4周 |
| 盈亏比 | ~1.0 | >1.5 | 4周 |

---

## 七、风险提示

1. **回测过拟合**: 需要严格的样本外测试
2. **数据质量**: Baostock 数据可能有误差
3. **交易成本**: 实际交易成本可能更高
4. **市场变化**: 策略可能在不同市场环境失效

---

*文档创建时间: 2025-11-27*
*版本: v1.0*

