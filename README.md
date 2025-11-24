# China Stock Quant Analysis (Qlib-based)

本项目以 [Microsoft Qlib](https://github.com/microsoft/qlib) 为蓝本，面向低频中国 A 股量化研究场景。核心目标是基于同花顺数据源（通过 akshare 提供）完成数据接入、因子挖掘、模型训练与回测分析，寻找中期回报率更高的因子与模型组合，为低频交易提供决策支持。

## 项目结构

```
.
├── README.md
├── requirements.txt            # 运行依赖（Qlib + akshare）
└── src
    └── qlib_cn_lowfreq
        ├── __init__.py
        ├── config.py           # Qlib 初始化与默认路径配置
        ├── data_pipeline.py    # 同花顺数据获取与 Qlib 数据转换
        └── workflow.py         # 低频因子挖掘、模型训练和回测示例
```

## 安装说明

> 建议使用 Python 3.9+，并在全局或虚拟环境下执行以下步骤。

1. **在线安装**（默认方式）：

   ```bash
   pip install -r requirements.txt
   ```

2. **离线/代理安装**（网络受限或需要内网环境部署时）：

   ```bash
   # 在可联网环境提前下载 requirements.txt 中列出的 wheel 包
   # 将 wheel 拷贝到 <wheel_dir> 后，在目标环境执行
   pip install --no-index --find-links <wheel_dir> -r requirements.txt
   # 或者在具备 HTTP 代理的环境追加 --proxy 参数
   # pip install --proxy http://<host>:<port> -r requirements.txt
   ```

3. **安装自检**：即便第三方依赖尚未安装完毕，也可提前验证 CLI 可用性：

   ```bash
   PYTHONPATH=src python -m qlib_cn_lowfreq.data_pipeline --help
   PYTHONPATH=src python -m qlib_cn_lowfreq.workflow --help
   ```

   如未安装依赖，命令会提示需要的第三方库；成功安装后，以上命令会输出参数说明。

## 快速开始

1. 初始化数据（日频存储，支持镜像到 Qlib 官方 Yahoo! 财经数据目录）：

   ```bash
   # 下载单个股票数据（例如：比亚迪 002594.SZ）
   PYTHONPATH=src python -m qlib_cn_lowfreq.data_pipeline --start 2016-01-01 --end 2024-12-31 --symbols 002594.SZ

   # 下载多个股票数据
   PYTHONPATH=src python -m qlib_cn_lowfreq.data_pipeline --start 2015-01-01 --end 2024-12-31 --symbols 000001.SZ 600000.SH --mirror-yahoo
   ```

   该命令将从 akshare 拉取日线数据（前复权），并转换为 Qlib 所需的**日频**二进制格式存放到 `./data/akshare_ths/qlib_data`；使用 `--mirror-yahoo` 选项会将生成数据复制到 `~/.qlib/qlib_data/cn_data`，方便与官方 Yahoo! 财经数据合并使用。

2. 运行示例训练与回测：

   ```bash
   PYTHONPATH=src python -m qlib_cn_lowfreq.workflow --market csi300 --start 2017-01-01 --end 2023-12-31
   ```

   示例基于 Qlib 的 Alpha158 因子处理器与 LightGBM 模型，展示低频（周频）回测流程。

3. 验证数据有效性：

   数据生成后，可以通过 Qlib 的标准接口验证数据：

   ```python
   import qlib
   qlib.init(provider_uri='./data/akshare_ths/qlib_data', region='cn')
   from qlib.data import D
   
   # 获取日历
   cal = D.calendar(start_time='2016-01-01', end_time='2024-12-31')
   print(f"交易日历数量: {len(cal)}")
   
   # 获取股票数据
   data = D.features(['002594.SZ'], ['$close', '$volume'], start_time='2016-01-01', end_time='2024-12-31')
   print(data.head())
   ```

## 测试说明

项目附带完整的回归测试以验证核心逻辑（路径初始化、依赖守护、工作流解析、数据获取与转换等）。安装依赖后，可运行：

```bash
# 运行所有测试
PYTHONPATH=src python -m unittest discover -v

# 运行测试并生成覆盖率报告
PYTHONPATH=src coverage run --source=src/qlib_cn_lowfreq -m unittest discover
coverage report
coverage html  # 生成 HTML 报告
```

当前测试覆盖率达到 **100%**，包括：
- 配置模块（`config.py`）
- 数据管线（`data_pipeline.py`）：包括 akshare API 兼容性、数据转换、Qlib 格式写入等
- 工作流（`workflow.py`）：包括参数解析、实验运行等

如需在不同环境验证，可添加 `-k` 过滤器仅运行特定用例，例如 `-k data_pipeline`。

> **提示：** `python -m qlib_cn_lowfreq.data_pipeline --help` 与 `python -m qlib_cn_lowfreq.workflow --help` 在缺少第三方依赖时也可正常输出
> 用法说明，便于在受限环境中先行查看参数；真正执行数据拉取与训练时依然需要按上述方式安装依赖。

## 设计要点

- **同花顺数据源覆盖**：`data_pipeline.py` 中通过 akshare 的历史行情接口获取数据：
  - 优先使用 `stock_zh_a_hist_ths`（同花顺接口）
  - 自动降级到 `stock_zh_a_hist`（通用接口）作为兼容性备选
  - 使用 `stock_zh_a_spot_em` 获取股票列表
  - 统一转换到 Qlib 期望的字段格式（open, high, low, close, volume, amount）
- **Qlib 数据格式兼容**：
  - 优先使用 Qlib 的 `dump_bin` API（如果可用）
  - 自动降级到本地 vendored `qlib/scripts/dump_bin.py` 的 CLI 调用
  - 支持与官方 Yahoo! 财经数据合并使用
- **低频友好**：默认将日线数据聚合为周频，便于中期（数周至数月）因子与策略研究。
- **因子与模型扩展**：`workflow.py` 使用 Qlib 原生的 `Alpha158` 作为示例，可在 `FACTOR_CONFIG` 中自由增减因子，也可替换模型为其他 Qlib 内置或自定义模型。
- **测试覆盖**：项目包含完整的单元测试，代码覆盖率达到 100%，确保核心功能稳定可靠。

## 后续方向

- 丰富同花顺特有的特色因子（如资金流、龙虎榜等），并补充相应的聚合逻辑。
- 构建面向多市场/多周期的组合优化与风险控制模块。
- 引入更贴合低频场景的交易成本建模与持仓约束。

