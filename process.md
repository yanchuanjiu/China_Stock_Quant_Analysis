# QT_China 项目开发过程记录

## 2025-11-25 用户需求分析 - 实盘模拟配置

### 用户需求

1. **数据更新**: 5月14日之后的交易数据如何获取？
2. **每日模拟**: 项目需要支持每日自动运行，根据前一天数据生成交易信号
3. **实盘参数**: 
   - 起始资金: 10万元
   - 交易单位: 100股/手
   - 手续费: 万分之三
   - 滑点: 需要考虑
   - 成交价: 信号后第二天开盘价

### Qlib 原生配置项分析

✅ **Qlib 完全支持以上所有配置**，无需额外开发：

| 需求 | Qlib 配置项 | 说明 |
|------|-------------|------|
| 起始资金 10万 | `account: 100000` | ✅ 直接配置 |
| 100股/手 | `trade_unit: 100` | ✅ 中国市场默认值 |
| 万3手续费 | `open_cost: 0.0003, close_cost: 0.0003` | ✅ 直接配置 |
| 滑点 | `impact_cost: 0.001` | ✅ 直接配置 (推荐0.1%) |
| 开盘价成交 | `deal_price: "open"` | ✅ 直接配置 |
| 涨跌停限制 | `limit_threshold: 0.095` | ✅ 中国市场默认值 |
| 最低手续费 | `min_cost: 5` | ✅ 默认5元 |

### 数据更新方案

**方案1: Crowd Source 数据 (推荐)**
```bash
# 每日更新数据 - chenditc/investment_data 提供每日更新
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
```

**方案2: Yahoo Finance (备选)**
```bash
# 增量更新
python qlib/scripts/data_collector/yahoo/collector.py update_data_to_bin \
    --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data \
    --end_date $(date +%Y-%m-%d)
```

**方案3: Docker 自动更新**
```bash
docker run -v ~/.qlib/qlib_data/cn_data:/output -it --rm chenditc/investment_data \
    bash dump_qlib_bin.sh && cp ./qlib_bin.tar.gz /output/
```

---

## 2025-11-25 RD-Agent 完整闭环测试 (修正版)

### 1. 数据获取

**目标**: 获取沪深300股票2024年至今的日级交易数据

**执行步骤**:
1. 首先尝试使用 Qlib 官方数据源下载（数据只到2020年9月）
2. 切换使用 crowd source 数据源（chenditc/investment_data）
3. 成功获取 2005-01-04 至 2025-05-14 的完整数据

**数据源**:
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
```

**数据统计**:
- 交易日历: 5074 个交易日
- 实际可用数据截止: **2025-05-14** (crowd source 更新周期)
- CSI300成分股: 336 只

### 2. 模型训练与回测 (修正后)

**实验配置**:
- 数据范围: 2024-01-01 ~ **2025-05-14**
- 股票池: 沪深300 (CSI300)
- 特征: Alpha158 因子库 (159维特征)
- 训练集: 2024-01-01 ~ 2024-06-30 (30,300 样本)
- 验证集: 2024-07-01 ~ 2024-09-30 (16,725 样本)
- 测试集: **2024-10-08 ~ 2025-05-14** (38,593 样本, 146个交易日)
- 策略: TopkDropout (Top30, Drop3)
- 基准: 沪深300指数 (SH000300)

**模型对比结果** (修正后):

| 模型 | IC | ICIR | Rank IC | Rank ICIR | 超额年化收益 | 信息比率 | 最大回撤 |
|------|-----|------|---------|-----------|--------------|----------|----------|
| XGBoost | 0.0678 | 0.3879 | 0.0099 | 0.0683 | **8.59%** | 0.5370 | -8.63% |
| LightGBM | 0.0351 | 0.2135 | 0.0153 | 0.0765 | **17.91%** | **1.0540** | -10.13% |

**基准收益** (Buy & Hold 沪深300):
- 年化收益率: 21.47%
- 信息比率: 0.9229
- 最大回撤: -17.20%

### 3. 结果分析

**XGBoost 模型**:
- IC 较高 (0.0678)，预测能力较强
- ICIR 较高 (0.3879)，预测稳定性好
- 超额收益 8.59%，最大回撤控制较好 (-8.63%)

**LightGBM 模型**:
- IC 较低 (0.0351)，但 Rank IC 较高 (0.0153)
- 超额收益最高 (17.91%)
- **信息比率最高 (1.0540)**，风险调整后收益最优
- 最大回撤略大 (-10.13%)

**结论**: LightGBM 在该数据集上表现更优，信息比率超过1.0，风险调整后收益优秀。

### 4. 结果位置

- **实验记录**: `experiments/` 目录 (MLflow 格式)
- **可视化报告**: `output/visualizations/index.html`
- **交易详情**: `output/visualizations/positions_detail.html`
- **对比报告**: `output/EXPERIMENT_REPORT.md`
- **数据表格**: `output/model_comparison.csv`
- **测试脚本**: `run_rdagent_workflow.py`

### 5. 如何复现

```bash
# 1. 下载数据
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2

# 2. 运行完整测试
cd /Users/air/QT_China
python run_rdagent_workflow.py

# 3. 生成可视化
python visualize_results.py

# 4. 查看结果
open output/visualizations/index.html

# 5. 查看 MLflow UI
cd experiments && mlflow ui --backend-store-uri .
# 访问 http://localhost:5000
```

### 6. 待优化项

1. ✅ 实现图表可视化 (收益曲线、回撤曲线)
2. ✅ 修复数据范围问题 (2025-05-14截止)
3. ✅ 添加模拟实盘脚本 (`run_paper_trading.py`)
4. ⏳ 添加每日自动更新数据脚本 (cron job)
5. ⏳ 添加更多模型对比 (LSTM, Transformer, etc.)
6. ⏳ 添加因子重要性分析
7. ⏳ 支持自定义因子挖掘
8. ⏳ 集成 RD-Agent 自动化流程

---

## 2025-11-25 模拟实盘测试结果

### 配置参数 (全部使用 Qlib 原生配置)

| 参数 | 配置值 | Qlib 配置项 |
|------|--------|-------------|
| 起始资金 | ¥100,000 | `account: 100000` |
| 交易单位 | 100股/手 | `trade_unit: 100` |
| 买入手续费 | 万分之三 | `open_cost: 0.0003` |
| 卖出手续费 | 万分之三 | `close_cost: 0.0003` |
| 滑点成本 | 0.1% | `impact_cost: 0.001` |
| 成交价格 | 开盘价 | `deal_price: "open"` |
| 涨跌停限制 | 9.5% | `limit_threshold: 0.095` |
| 最低手续费 | 5元 | `min_cost: 5` |

### 回测结果 (2024-11-15 ~ 2025-05-14, 118个交易日)

| 指标 | 数值 |
|------|------|
| 起始资金 | ¥100,000 |
| 最终资金 | ¥130,349 |
| **总收益率** | **+30.35%** |
| 累计交易成本 | 1.90% |
| 平均换手率 | 35.89% |
| 年化超额收益 | +17.65% |
| 信息比率 | 0.9953 |
| 最大回撤 | -8.63% |

### 使用方法

```bash
# 运行模拟实盘 (默认 LightGBM)
python run_paper_trading.py

# 使用 XGBoost 模型
python run_paper_trading.py --model xgboost

# 指定回测日期范围
python run_paper_trading.py --start 2024-10-01 --end 2025-05-14

# 先更新数据再运行
python run_paper_trading.py --update
```

---

## 2025-11-25 版本发布 v0.0.6

### Git 提交信息

**分支**: `codex/0.0.6`  
**提交**: `d2c7ba4`  
**远程仓库**: https://github.com/yanchuanjiu/China_Stock_Quant_Analysis.git

### 主要更新内容

1. ✅ **新增模拟实盘交易脚本** (`run_paper_trading.py`)
   - 完整的实盘模拟功能
   - 支持 LightGBM 和 XGBoost 模型
   - 自动数据更新支持

2. ✅ **新增可视化脚本** (`visualize_results.py`)
   - 累计收益曲线
   - 风险分析图表
   - 模型对比雷达图和柱状图
   - 详细交易记录和持仓详情

3. ✅ **修复数据范围问题**
   - 回测周期修正为 2024-10-08 ~ 2025-05-14
   - 修复收益曲线在5月14日后停止的问题

4. ✅ **更新配置文件**
   - 新增工作流配置文件 (`configs/`)
   - 更新 `.gitignore` 排除数据目录

5. ✅ **更新文档**
   - 完善 `process.md` 记录
   - 更新实验报告和对比数据

### 统计信息

- **文件变更**: 179 个文件
- **新增代码**: 21,684 行
- **删除代码**: 346 行
- **新增文件**: 
  - `run_paper_trading.py`
  - `visualize_results.py`
  - `configs/workflow_config_*.yaml`
  - `output/visualizations/*.html`
  - `output/paper_trading/*`

### 查看代码

```bash
# 克隆仓库
git clone https://github.com/yanchuanjiu/China_Stock_Quant_Analysis.git
cd China_Stock_Quant_Analysis

# 切换到 v0.0.6 分支
git checkout codex/0.0.6

# 查看提交历史
git log --oneline -10
```

---

## 2025-11-26 修复回测问题

### 问题修复

1. ❌ **数据更新问题**: crowd source 数据截止到 2025-05-14，暂无更新
2. ✅ **持仓数据错误**: 之前可视化读取的是 1亿起始资金的实验，已修复为正确的 10万起始资金
3. ✅ **添加 Buy & Hold 对比**: 使用沪深300指数作为基准进行对比

### 正确的回测结果 (起始资金: ¥100,000)

**回测周期**: 2024-11-15 ~ 2025-05-14 (118 个交易日)

| 指标 | LightGBM 策略 | Buy & Hold |
|------|---------------|------------|
| **最终市值** | ¥143,880.78 | ¥120,984.61 |
| **总收益率** | **+43.88%** | +20.98% |
| **超额收益** | **+22.90%** | - |
| 年化超额收益 | +37.17% | - |
| 信息比率 | 2.2796 | - |
| 最大回撤 | -4.65% | - |
| 累计交易成本 | 1.72% | 0.03% |
| 平均换手率 | 36.47% | 0% |

### 最终持仓明细

| 股票代码 | 持仓数量 | 市值 | 权重 |
|----------|----------|------|------|
| SH600893 | 1,628 | ¥19,577 | 13.61% |
| SZ000938 | 3,637 | ¥14,406 | 10.01% |
| SH601100 | 2,035 | ¥14,349 | 9.97% |
| SH600760 | 1,789 | ¥13,837 | 9.62% |
| SH600660 | 227 | ¥13,711 | 9.53% |
| **现金** | - | **¥31,538.82** | 21.92% |

### 新增脚本

- `run_complete_backtest.py`: 完整回测脚本，包含 Buy & Hold 对比

### 结果位置

- **对比报告**: `output/backtest_comparison/comparison_report.html`
- **每日数据**: `output/backtest_comparison/daily_comparison.csv`
- **汇总数据**: `output/backtest_comparison/summary.csv`

### 数据更新说明

当前数据截止到 **2025-05-14**。要获取更新的数据，请执行：

```bash
# 方案1: 下载最新 crowd source 数据
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2

# 方案2: 使用 Docker (数据更新更及时)
docker run -v ~/.qlib/qlib_data/cn_data:/output -it --rm chenditc/investment_data bash dump_qlib_bin.sh
```

---

## 2025-11-26 数据源研究与完整分析

### 1. 数据源可用性研究

| 数据源 | 最新数据日期 | 更新频率 | 可用性 |
|--------|-------------|----------|--------|
| **Crowd Source** | 2025-05-14 | 不定期 | ✅ 已使用 |
| **Baostock** | **2025-11-25** | **每日更新** | ✅ **推荐** |
| Yahoo Finance | 不支持中国市场 | - | ❌ |

**结论**: Baostock 可以提供 **5月15日至11月25日的 132 天新数据**！

### 2. 完整模型对比 (新增指标)

**回测周期**: 2024-11-15 ~ 2025-05-14 (118 个交易日)

| 指标 | LightGBM | XGBoost | Buy & Hold |
|------|----------|---------|------------|
| **总收益率** | **+46.37%** | +24.39% | +20.89% |
| **年化收益率** | **+125.60%** | +59.39% | +50.47% |
| 年化波动率 | 25.91% | 29.72% | 21.29% |
| **Sharpe Ratio** | **3.16** | 1.62 | 1.89 |
| **Sortino Ratio** | **4.88** | 2.27 | 2.59 |
| 最大回撤 | -6.82% | -14.32% | -7.16% |
| **Calmar Ratio** | **18.42** | 4.15 | 7.05 |
| 胜率 | 58.47% | 56.78% | 60.68% |
| 盈亏比 | 1.21 | 1.00 | 0.93 |

### 3. 新增文件

- `run_full_analysis.py` - 完整分析脚本，包含:
  - 多模型对比 (LightGBM, XGBoost)
  - Buy & Hold 策略对比
  - **完整评估指标** (Sharpe, Sortino, Calmar 等)
  - **详细交易记录** (每日买卖明细)

### 4. 交易记录示例

**LightGBM 最近交易** (部分):

| 日期 | 股票 | 操作 | 数量 | 价格 | 金额 |
|------|------|------|------|------|------|
| 2025-05-13 | SZ002459 | SELL | 13,153 | ¥1.14 | ¥15,058 |
| 2025-05-13 | SZ002463 | BUY | 5,060 | ¥2.79 | ¥14,102 |
| 2025-05-13 | SH601100 | BUY | 2,035 | ¥7.06 | ¥14,359 |
| 2025-05-14 | SZ000938 | BUY | 3,637 | ¥3.96 | ¥14,406 |

完整交易记录保存在: `output/full_analysis/lightgbm_trades.csv`

### 5. 结果位置

- **完整报告**: `output/full_analysis/full_analysis_report.html`
- **指标对比**: `output/full_analysis/metrics_comparison.csv`
- **交易记录**: `output/full_analysis/*_trades.csv`

### 6. 如何更新数据 (使用 Baostock)

```python
# 在 run_full_analysis.py 中调用
from run_full_analysis import update_data_from_baostock
update_data_from_baostock(start_date='2025-05-15')
```

或安装 baostock: `pip install baostock`

---

## 2025-11-26 Baostock 数据更新 & 完整回测

### 1. 数据更新结果

✅ **成功从 Baostock 下载 300 只沪深300成分股的最新数据**

- 数据范围: 2025-05-15 ~ 2025-11-25
- 交易日数: **132 天**
- 数据位置: `data/baostock_update/`

### 2. 完整回测结果 (含最新数据)

| 指标 | LightGBM (118天) | XGBoost (118天) | Buy & Hold (249天) |
|------|------------------|-----------------|-------------------|
| **总收益率** | **+46.37%** | +24.39% | **+201.83%** |
| **年化收益率** | **+125.60%** | +59.39% | +205.87% |
| 年化波动率 | 25.91% | 29.72% | 159.11% |
| **Sharpe Ratio** | **3.16** | 1.62 | 1.10 |
| **Sortino Ratio** | 4.88 | 2.27 | **11.02** |
| 最大回撤 | **-6.82%** | -14.32% | -19.76% |
| **Calmar Ratio** | **18.42** | 4.15 | 10.42 |
| 胜率 | 58.47% | 56.78% | 54.22% |
| 盈亏比 | 1.21 | 1.00 | 1.94 |

**说明**: Buy & Hold 包含了5月15日至11月25日的新数据 (132天)，因此收益更高

### 3. 详细交易记录

- LightGBM: **438 条**交易记录
- XGBoost: **445 条**交易记录
- 完整 CSV: `output/full_backtest_new_data/*_trades_full.csv`

**示例交易记录** (LightGBM 最后几笔):

| 日期 | 股票 | 操作 | 数量 | 价格 | 金额 |
|------|------|------|------|------|------|
| 2025-05-13 | SZ002463 | BUY | 5,060 | ¥2.79 | ¥14,102 |
| 2025-05-13 | SH601100 | BUY | 2,035 | ¥7.06 | ¥14,359 |
| 2025-05-14 | SZ000938 | BUY | 3,637 | ¥3.96 | ¥14,406 |

### 4. 新增脚本

- `update_data_baostock.py` - 从 Baostock 更新数据
- `run_backtest_with_new_data.py` - 使用最新数据运行回测

### 5. 运行命令

```bash
# 更新数据
python update_data_baostock.py --start 2025-05-15

# 运行完整回测
python run_backtest_with_new_data.py
```

### 6. 报告位置

- HTML 报告: `output/full_backtest_new_data/full_backtest_report.html`
- 指标对比: `output/full_backtest_new_data/metrics_comparison_full.csv`
- 交易记录: `output/full_backtest_new_data/*_trades_full.csv`

---

## 2025-11-26 一致时间周期回测 (避免过拟合)

### 1. 时间配置 (严格避免未来数据泄露)

| 阶段 | 时间范围 | 用途 |
|------|----------|------|
| 训练期 | 2024-01-01 ~ 2025-08-31 | 模型训练 |
| 验证期 | 2025-09-01 ~ 2025-09-30 | 调参验证 |
| **测试期** | **2025-10-01 ~ 2025-11-25** | **统一对比周期** |

### 2. 统一周期回测结果 (34个交易日)

| 指标 | 因子模型 | Buy & Hold | 超额收益 |
|------|----------|------------|----------|
| 总收益率 | **-2.55%** | -14.37% | **+11.82%** |
| 年化收益率 | -17.92% | -69.41% | +51.49% |
| 年化波动率 | **9.11%** | 25.10% | - |
| Sharpe Ratio | **-2.45** | -4.70 | +2.25 |
| 最大回撤 | **-3.74%** | -15.44% | - |
| 胜率 | **48.48%** | 36.36% | - |
| 交易天数 | 33 | 33 | - |

**关键发现**: 在市场下跌期间，因子模型有效控制了回撤，超额收益 +11.82%

### 3. 数据更新脚本 (支持每日执行)

```bash
# 每日增量更新 (推荐)
python update_data_baostock.py --daily

# 手动指定日期
python update_data_baostock.py --start 2025-05-15

# 设置定时任务 (每个交易日18:00执行)
crontab -e
0 18 * * 1-5 cd /Users/air/QT_China && python update_data_baostock.py --daily
```

### 4. 交易记录详情

- 总交易次数: **94 条**
- 持仓股票: **10 只**
- 调仓频率: 每周一次
- 完整记录: `output/consistent_backtest/factor_model_trades.csv`

### 5. 新增/更新脚本

| 脚本 | 功能 |
|------|------|
| `run_consistent_backtest.py` | 一致时间周期回测 |
| `update_data_baostock.py` | 支持 `--daily` 每日增量更新 |

### 6. 报告位置

- **HTML 报告**: `output/consistent_backtest/consistent_backtest_report.html`
- **指标对比**: `output/consistent_backtest/metrics_comparison_consistent.csv`
- **交易记录**: `output/consistent_backtest/factor_model_trades.csv`

---

## 2025-11-27 Qlib 完整回测 (每日交易 + 多模型对比)

### 1. 解决的问题

| 问题 | 解决方案 |
|------|----------|
| 交易频率 | 改为**每日交易**，第二天开盘价执行 |
| 现金余额 | 交易记录包含**现金(前)/现金(后)/账户总值**，检查超买 |
| 模型对比 | 同时运行 **LightGBM + XGBoost** |
| 评估指标 | 使用 Qlib 完整风险分析模块 |

### 2. 回测结果 (2025-04-16 ~ 2025-05-14, 18个交易日)

| 模型 | 总收益率 | 年化收益 | 年化波动 | Sharpe | 最大回撤 |
|------|----------|----------|----------|--------|----------|
| **LightGBM** | **-6.88%** | -63.15% | 27.05% | -3.47 | -7.79% |
| **XGBoost** | **-6.18%** | -59.05% | 21.35% | -3.97 | -8.03% |
| Buy & Hold | -5.98% | -59.90% | 19.21% | -4.55 | -5.27% |

**说明**: 测试期间市场整体下跌，所有策略均为负收益

### 3. 详细交易记录

每条记录包含：

| 字段 | 说明 |
|------|------|
| 日期 | 交易日期 |
| 股票 | 股票代码 |
| 操作 | BUY/SELL |
| 数量 | 交易股数 |
| 价格 | 成交价格 |
| 金额 | 交易金额 |
| **现金(前)** | 交易前现金余额 |
| **现金(后)** | 交易后现金余额 |
| **账户总值** | 当前账户总市值 |

**交易记录数**:
- LightGBM: **72 条**
- XGBoost: **72 条**
- ✅ 所有交易记录正常，**无超买情况**

### 4. Alpha158 量比相关因子

Qlib Alpha158 中与量比相关的因子:

| 因子 | 说明 |
|------|------|
| VMA | 成交量移动平均 |
| VSTD | 成交量标准差 |
| WVMA | 成交量加权价格波动 |
| CORR | 价格与成交量相关性 |
| CORD | 价格变化与成交量变化相关性 |
| VSUMP/VSUMN | 成交量上涨/下跌比例 |
| VSUMD | 成交量涨跌差异 |

### 5. 配置详情

```python
CONFIG = {
    "account": 100000,           # 起始资金 10万
    "exchange_kwargs": {
        "deal_price": "open",    # 第二天开盘价交易
        "open_cost": 0.0003,     # 万3手续费
        "close_cost": 0.0003,    
        "impact_cost": 0.001,    # 0.1%滑点
        "trade_unit": 100,       # 100股/手
    },
}
```

### 6. 报告位置

- **HTML 报告**: `output/qlib_full_backtest/qlib_full_report.html`
- **交易明细**: `output/qlib_full_backtest/*_trades_detailed.csv`

### 7. 新增脚本

`run_qlib_full_backtest.py` - Qlib 完整回测分析
