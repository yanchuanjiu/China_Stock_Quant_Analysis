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
