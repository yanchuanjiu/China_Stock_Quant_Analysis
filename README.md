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

## 快速开始

1. 安装依赖（如处于离线/受限网络环境，可提前下载 wheel 并使用 `pip install --no-index --find-links <dir> -r requirements.txt`）：

   ```bash
   pip install -r requirements.txt
   ```

   > **排障提示：** 如果在线安装因代理/网络限制失败，可在可联网环境提前下载对应版本的 wheel 包（含 `akshare`、`pyqlib`、`lightgbm` 等）拷贝到本地目录，
   > 通过 `pip install --no-index --find-links <wheel_dir> -r requirements.txt` 完成离线安装；或者在具备稳定代理的环境下追加 `--proxy http://<host>:<port>` 参数。

2. 初始化数据（日频存储，支持镜像到 Qlib 官方 Yahoo! 财经数据目录）：

   ```bash
   python -m qlib_cn_lowfreq.data_pipeline --start 2015-01-01 --end 2024-12-31 --symbols 000001.SZ 600000.SH --mirror-yahoo
   ```

   该命令将从 akshare 的同花顺接口拉取日线数据（前复权），并转换为 Qlib 所需的**日频**二进制格式存放到 `./data/akshare_ths`；同时会将生成数据复制到 `~/.qlib/qlib_data/cn_data`，方便与官方 Yahoo! 财经数据合并使用。

3. 运行示例训练与回测：

   ```bash
   python -m qlib_cn_lowfreq.workflow --market csi300 --start 2017-01-01 --end 2023-12-31
   ```

   示例基于 Qlib 的 Alpha158 因子处理器与 LightGBM 模型，展示低频（周频）回测流程。

> **提示：** `python -m qlib_cn_lowfreq.data_pipeline --help` 与 `python -m qlib_cn_lowfreq.workflow --help` 在缺少第三方依赖时也可正常输出
> 用法说明，便于在受限环境中先行查看参数；真正执行数据拉取与训练时依然需要按上述方式安装依赖。

## 设计要点

- **同花顺数据源覆盖**：`data_pipeline.py` 中通过 akshare 的同花顺历史行情接口（`stock_zh_a_hist_ths`）与股票列表接口（`stock_zh_a_spot_em`）获取数据，并统一转换到 Qlib 期望的字段格式。
- **低频友好**：默认将日线数据聚合为周频，便于中期（数周至数月）因子与策略研究。
- **因子与模型扩展**：`workflow.py` 使用 Qlib 原生的 `Alpha158` 作为示例，可在 `FACTOR_CONFIG` 中自由增减因子，也可替换模型为其他 Qlib 内置或自定义模型。

## 后续方向

- 丰富同花顺特有的特色因子（如资金流、龙虎榜等），并补充相应的聚合逻辑。
- 构建面向多市场/多周期的组合优化与风险控制模块。
- 引入更贴合低频场景的交易成本建模与持仓约束。

