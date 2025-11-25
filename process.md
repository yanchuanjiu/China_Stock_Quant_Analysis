# QT_China 项目开发过程记录

## 2025-11-25 RD-Agent 完整闭环测试

### 1. 数据获取

**目标**: 获取沪深300股票2024年至今的日级交易数据

**执行步骤**:
1. 首先尝试使用 Qlib 官方数据源下载（数据只到2020年9月）
2. 切换使用 crowd source 数据源（chenditc/investment_data）
3. 成功获取 2005-01-04 至 2025-11-24 的完整数据

**数据源**:
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
```

**数据统计**:
- 交易日历: 5074 个交易日
- 2024年交易日: 242 天
- CSI300成分股: 336 只

### 2. 模型训练与回测

**实验配置**:
- 数据范围: 2024-01-01 ~ 2025-11-24
- 股票池: 沪深300 (CSI300)
- 特征: Alpha158 因子库 (159维特征)
- 训练集: 2024-01-01 ~ 2024-06-30 (30,300 样本)
- 验证集: 2024-07-01 ~ 2024-09-30 (16,725 样本)
- 测试集: 2024-10-01 ~ 2025-11-24 (38,593 样本)
- 策略: TopkDropout (Top30, Drop3)
- 基准: 沪深300指数 (SH000300)

**模型对比结果**:

| 模型 | IC | ICIR | Rank IC | Rank ICIR | 超额年化收益 | 信息比率 | 最大回撤 |
|------|-----|------|---------|-----------|--------------|----------|----------|
| XGBoost | 0.0678 | 0.3879 | 0.0099 | 0.0683 | 4.53% | 0.1866 | -8.63% |
| LightGBM | 0.0351 | 0.2135 | 0.0153 | 0.0765 | 9.44% | 0.5742 | -10.13% |

**基准收益** (Buy & Hold 沪深300):
- 年化收益率: 11.32%
- 信息比率: 0.6705
- 最大回撤: -17.20%

### 3. 结果分析

**XGBoost 模型**:
- IC 较高 (0.0678)，预测能力较强
- ICIR 较高 (0.3879)，预测稳定性好
- 超额收益较低 (4.53%)，但最大回撤控制较好 (-8.63%)

**LightGBM 模型**:
- IC 较低 (0.0351)，但 Rank IC 较高 (0.0153)
- 超额收益较高 (9.44%)
- 信息比率最高 (0.5742)，风险调整后收益最优
- 最大回撤略大 (-10.13%)

**结论**: LightGBM 在该数据集上表现更优，风险调整后收益更高。

### 4. 结果位置

- **实验记录**: `experiments/` 目录 (MLflow 格式)
- **对比报告**: `output/EXPERIMENT_REPORT.md`
- **数据表格**: `output/model_comparison.csv`
- **测试脚本**: `run_rdagent_workflow.py`

### 5. 如何复现

```bash
# 1. 下载数据
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2

# 2. 复制日历和instruments文件到正确位置
cp ~/.qlib/qlib_data/cn_data/day.txt ~/.qlib/qlib_data/cn_data/calendars/day.txt
mkdir -p ~/.qlib/qlib_data/cn_data/instruments
cp ~/.qlib/qlib_data/cn_data/csi300.txt ~/.qlib/qlib_data/cn_data/instruments/csi300.txt

# 3. 运行完整测试
cd /Users/air/QT_China
python run_rdagent_workflow.py

# 4. 查看 MLflow UI
cd experiments && mlflow ui
# 访问 http://localhost:5000
```

### 6. 待优化项

1. 添加更多模型对比 (LSTM, Transformer, etc.)
2. 实现图表可视化 (收益曲线、回撤曲线)
3. 添加因子重要性分析
4. 支持自定义因子挖掘
5. 集成 RD-Agent 自动化流程
