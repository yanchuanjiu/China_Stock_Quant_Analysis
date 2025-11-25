#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import datetime
import akshare as ak
import pandas as pd

# Ensure src is in python path
PROJECT_ROOT = Path(__file__).resolve().parent

# Critical: Remove project root from sys.path to prevent 'import qlib' 
# from importing the uncompiled local source directory 'qlib/' instead of 
# the installed 'pyqlib' package.
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qlib_cn_lowfreq import data_pipeline, workflow, config

def get_csi300_symbols():
    print("正在获取沪深300成分股列表...")
    try:
        # 尝试从 akshare 获取最新的沪深300成分股
        df = ak.index_stock_cons_weight_csindex(symbol="000300")
        if "成分券代码" in df.columns:
             codes = df["成分券代码"].astype(str).tolist()
        else:
             print("警告: '成分券代码' 列未找到，尝试备用接口...")
             raise ValueError("Unexpected columns")
    except Exception as e:
        print(f"主要接口调用失败: {e}。尝试备用接口...")
        try:
            # 备用接口
            df = ak.index_stock_cons(symbol="399300") 
            codes = df["variety"].tolist()
        except Exception as e2:
            print(f"备用接口也失败: {e2}。将使用预设的 Top 10 股票列表进行测试。")
            codes = ["600519", "300750", "601318", "600036", "002594", "000858", "600276", "000333", "600900", "601166"]

    formatted = []
    for c in codes:
        c = str(c).zfill(6)
        if c.startswith("6"):
            formatted.append(f"{c}.SH")
        else:
            formatted.append(f"{c}.SZ")
    
    print(f"共获取 {len(formatted)} 只股票。")
    return formatted

def main():
    # 设定时间范围：为了适配本地样例数据（截止 2024-12-31），我们使用固定时间窗口
    # 如果能下载新数据，可以使用动态时间。但考虑到网络限制，固定时间更稳健。
    # end_date = datetime.date.today()
    # start_date = end_date - datetime.timedelta(days=365)
    start_str = "2024-01-01"
    end_str = "2024-12-31"
    
    print(f"测试时间范围: {start_str} 至 {end_str}")
    
    symbols = get_csi300_symbols()
    
    # 步骤 1: 下载并处理数据
    print("\n[Step 1] 运行数据流水线 (下载 & 转换)...")
    # 使用 config 中默认的 QLIB_PROVIDER_URI (data/akshare_ths/qlib_data)
    data_args = [
        "--start", start_str,
        "--end", end_str,
        "--symbols", *symbols
    ]
    
    try:
        data_pipeline.main(data_args)
        print("数据下载与转换完成。")
    except Exception as e:
        print(f"Warning: 数据流水线执行遇到错误 (可能是网络限制): {e}")
        print("尝试跳过下载，直接使用本地已有数据...")
    
    # 检查本地是否有数据
    # 注意：这里硬编码了默认路径，应与 config 保持一致
    default_uri = Path("data/akshare_ths/qlib_data")
    if not (default_uri / "instruments" / "all.txt").exists():
         print(f"Critical Error: {default_uri} 下没有可用的 Qlib 数据，且无法下载。测试无法继续。")
         sys.exit(1)

    # 步骤 2: 运行量化工作流 (因子挖掘 & 模型训练)
    print("\n[Step 2] 运行量化分析工作流...")
    
    # 读取 instruments/all.txt 确定可用的股票
    with open(default_uri / "instruments" / "all.txt", "r") as f:
        available_stocks = [line.split()[0] for line in f.readlines()]
    
    if not available_stocks:
        print("Error: instruments/all.txt 是空的。")
        sys.exit(1)
        
    print(f"本地可用股票: {len(available_stocks)} 只 ({available_stocks[:5]}...)")
    
    # 使用本地存在的第一只股票作为基准
    benchmark_symbol = available_stocks[0]
    print(f"使用 {benchmark_symbol} 作为回测基准。")
    
    workflow_args = [
        "--market", "all", 
        "--start", start_str,
        "--end", end_str,
        "--benchmark", benchmark_symbol,
        "--horizon", "5",
        "--topk", str(min(10, len(available_stocks))), # 动态调整 topk
        "--n-drop", str(min(2, len(available_stocks))) # 动态调整 n-drop
    ]
    
    try:
        args = workflow.parse_args(workflow_args)
        workflow.run_experiment(args)
    except Exception as e:
        print(f"工作流执行失败: {e}")
        sys.exit(1)

    print("\n=== 全流程测试成功完成 ===")

if __name__ == "__main__":
    main()
