# 网络与数据源诊断报告

## 执行时间
2025-11-24 22:27

## 问题定位

### 1. 网络连接状态
- **基础网络**: ✓ DNS 解析正常，可以解析所有目标域名
- **实际连接**: ✗ 所有 HTTPS 连接失败，错误码 `[Errno 49] Can't assign requested address`

### 2. 错误分析
```
OSError: [Errno 49] Can't assign requested address
```
这是**系统级网络问题**，不是 akshare 接口本身的问题。可能原因：
1. 系统防火墙或安全软件阻止了出站 HTTPS 连接
2. VPN/代理配置异常
3. 网络接口配置问题（端口绑定失败）
4. 系统资源限制（端口耗尽、连接数限制等）

### 3. Akshare 接口状态
- **版本**: 1.17.6 ✓
- **接口可用性**:
  - ✓ `stock_zh_a_spot_em` - 存在
  - ✓ `stock_zh_a_hist` - 存在
  - ✗ `stock_zh_a_hist_ths` - **不存在**（可能是版本差异）
  - ✓ `index_stock_cons` - 存在
  - ✓ `index_stock_cons_weight_csindex` - 存在

**结论**: akshare 接口本身是有效的，问题在于网络连接层。

### 4. Qlib 数据收集器
- **cn_index 收集器**: 不使用 akshare，而是使用：
  - `baostock` 库
  - 直接 HTTP 请求（requests）
  - 中证指数官网 API

**建议**: 可以尝试使用 qlib 官方的 `cn_index` 收集器作为替代方案。

## 测试结果

### 测试 1: Akshare 实际连接
```
✗ 失败: ConnectionError: [Errno 49] Can't assign requested address
```
**确认**: 系统级网络问题，非 akshare 接口问题。

### 测试 2: Qlib 工作流（使用本地数据）
```
✓ 数据加载成功: 1 只股票 (002594.SZ)
✓ 交易日历: 242 个交易日 (2024-01-02 至 2024-12-31)
✓ 数据集构建成功 (Alpha158 因子)
✗ 模型训练失败: recorder API 兼容性问题
```

## 解决方案

### 方案 1: 修复网络问题（推荐）
1. 检查系统防火墙设置
2. 检查 VPN/代理配置
3. 重启网络服务
4. 检查系统资源限制

### 方案 2: 使用 Qlib 官方数据收集器
```bash
# 使用 cn_index 收集器（不依赖 akshare）
python -m qlib.scripts.data_collector.cn_index.collector --help
```

### 方案 3: 离线数据迁移
1. 在具备正常网络的环境下载数据
2. 使用 `data_pipeline.py` 或 qlib 收集器获取数据
3. 将数据文件迁移到当前环境

### 方案 4: 使用预构建数据
- 使用 qlib 官方提供的预构建数据包
- 或使用 crowd_source 数据（见 `qlib/scripts/data_collector/crowd_source/README.md`）

## 代码修复

### 已修复的问题
1. ✅ `workflow.py`: 修复 `qlib.contrib.eval` → `qlib.contrib.evaluate`
2. ✅ `workflow.py`: 移除不存在的 `Task` 模块引用
3. ✅ `workflow.py`: 修复 recorder API 兼容性（`exp_name` → `experiment_name`）
4. ✅ `data_pipeline.py`: 增强异常处理，支持网络错误容错

### 待修复的问题
1. ⚠️ `run_qlib_workflow_test.py`: recorder `log_params` API 兼容性（需要进一步适配）

## 下一步行动

1. **立即**: 使用本地已有数据继续测试工作流（已修复 recorder 兼容性）
2. **短期**: 尝试修复系统网络配置或使用 qlib cn_index 收集器
3. **长期**: 建立稳定的数据获取流程，考虑离线数据包方案

## 文件清单

- `diagnose_network_and_data.py` - 网络诊断脚本
- `test_akshare_connection.py` - akshare 连接测试
- `run_qlib_workflow_test.py` - 基于 run_all_model.py 的工作流测试
- `DIAGNOSIS_REPORT.md` - 本报告

