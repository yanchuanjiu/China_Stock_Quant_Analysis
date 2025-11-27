# 🚀 中国 A 股量化交易系统

基于 [Microsoft Qlib](https://github.com/microsoft/qlib) 的量化投资平台，专注于沪深 300 股票的因子选股策略。

## 🎯 目标

| 指标 | 目标 | 当前最佳 |
|------|------|----------|
| 年化收益率 | **>50%** | -33.8% |
| 夏普比率 | **>2** | -1.40 |
| 最大回撤 | <15% | -5.79% |

> ⚠️ 当前测试期市场下跌，正在优化策略中

## 📁 项目结构

```
QT_China/
├── configs/                    # 策略配置
│   └── strategies/             # YAML 策略文件
│       ├── volume_factor_lgb.yaml
│       └── alpha158_xgb.yaml
│
├── src/                        # 核心代码
│   ├── signals/                # 信号生成模块
│   │   └── generator.py        # 交易信号生成器
│   ├── data/                   # 数据模块
│   ├── factors/                # 因子模块
│   ├── models/                 # 模型模块
│   ├── backtest/               # 回测模块
│   └── trading/                # 交易模块
│
├── scripts/                    # 入口脚本
│   └── run_backtest.py         # 统一回测入口
│
├── output/                     # 输出目录
│   ├── signals/                # 交易信号 (JSON)
│   ├── reports/                # 分析报告 (HTML/CSV)
│   └── logs/                   # 运行日志
│
├── tests/                      # 测试
├── docs/                       # 文档
├── REFACTOR_PLAN.md           # 重构计划
└── process.md                  # 开发日志
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

```bash
# 下载 Qlib 数据
python -m qlib.tests.data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 或使用 Baostock 更新最新数据
python update_data_baostock.py --daily
```

### 3. 运行回测

```bash
# 单策略回测
python scripts/run_backtest.py --config configs/strategies/volume_factor_lgb.yaml

# 多策略对比
python scripts/run_backtest.py --compare
```

### 4. 生成交易信号

```bash
python src/signals/generator.py
```

信号输出示例 (`output/signals/2025-11-27_VolumeFactor_LightGBM.json`):

```json
{
  "date": "2025-11-27",
  "strategy": "VolumeFactor_LightGBM",
  "signals": [
    {
      "stock": "SZ002459",
      "action": "BUY",
      "weight": 0.12,
      "score": 0.85,
      "target_amount": 9500
    }
  ]
}
```

## 📊 策略列表

| 策略 | 因子 | 模型 | 配置文件 |
|------|------|------|----------|
| 量比因子 | Alpha158 (量比增强) | LightGBM | `volume_factor_lgb.yaml` |
| 标准因子 | Alpha158 | XGBoost | `alpha158_xgb.yaml` |

## 🔬 Alpha158 量比因子

| 因子 | 说明 |
|------|------|
| **VMA** | 成交量移动平均 |
| **VSTD** | 成交量标准差 |
| **WVMA** | 成交量加权价格波动 |
| **CORR** | 价格与成交量相关性 |
| **CORD** | 价格变化与成交量变化相关性 |
| **VSUMP/VSUMN** | 成交量上涨/下跌比例 |
| **VSUMD** | 成交量涨跌差异 |

## ⚙️ 交易配置

```yaml
backtest:
  account: 100000        # 起始资金 10万
  strategy:
    topk: 10             # 持有前10只
    n_drop: 2            # 每次最多调仓2只
  exchange:
    deal_price: "open"   # 第二天开盘价
    open_cost: 0.0003    # 买入万3
    close_cost: 0.0003   # 卖出万3
    impact_cost: 0.001   # 滑点0.1%
    trade_unit: 100      # 100股/手
```

## 📈 回测结果 (2025-04-16 ~ 2025-05-14)

| 模型 | 总收益率 | Sharpe | 最大回撤 |
|------|----------|--------|----------|
| **VolumeFactor_LightGBM** | **-2.90%** | -1.40 | -5.79% |
| VolumeFactor_XGBoost | -3.80% | -1.72 | -7.52% |
| Alpha158_LightGBM | -6.88% | -3.47 | -7.79% |
| Alpha158_XGBoost | -6.18% | -3.97 | -8.03% |
| Buy & Hold | -5.98% | -4.55 | -5.27% |

## 📝 开发路线图

- [x] 项目重构
- [x] 策略配置 YAML 化
- [x] 信号生成模块
- [ ] 因子优化 (进行中)
- [ ] 模型集成
- [ ] 交易平台对接
- [ ] 实盘模拟

## 📚 参考

- [Microsoft Qlib](https://github.com/microsoft/qlib)
- [Alpha158 因子](https://qlib.readthedocs.io/en/latest/component/data.html#alpha158)
- [Baostock](http://baostock.com/)

## 📄 License

MIT License

---

*最后更新: 2025-11-27*
