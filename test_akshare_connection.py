#!/usr/bin/env python3
"""测试 akshare 实际网络连接能力"""
import sys
import time
import akshare as ak

print("测试 akshare 实际网络连接...")
print("=" * 60)

# 测试 1: 获取股票列表（轻量级）
print("\n[测试 1] 获取 A 股实时行情（轻量级）...")
try:
    start = time.time()
    df = ak.stock_zh_a_spot_em()
    elapsed = time.time() - start
    print(f"  ✓ 成功获取 {len(df)} 只股票，耗时 {elapsed:.2f} 秒")
    print(f"  样例股票: {df.head(3)['代码'].tolist()}")
except Exception as e:
    print(f"  ✗ 失败: {type(e).__name__}: {e}")
    sys.exit(1)

# 测试 2: 获取单只股票历史数据
print("\n[测试 2] 获取单只股票历史数据（002594.SZ）...")
try:
    start = time.time()
    df = ak.stock_zh_a_hist(
        symbol="002594",
        period="daily",
        start_date="20240101",
        end_date="20240131",
        adjust="qfq"
    )
    elapsed = time.time() - start
    print(f"  ✓ 成功获取 {len(df)} 条记录，耗时 {elapsed:.2f} 秒")
    print(f"  日期范围: {df['日期'].min()} 至 {df['日期'].max()}")
except Exception as e:
    print(f"  ✗ 失败: {type(e).__name__}: {e}")
    if "Can't assign requested address" in str(e):
        print("  确认: 这是系统级网络问题，不是 akshare 接口问题")

# 测试 3: 获取沪深300成分股
print("\n[测试 3] 获取沪深300成分股列表...")
try:
    start = time.time()
    df = ak.index_stock_cons_weight_csindex(symbol="000300")
    elapsed = time.time() - start
    print(f"  ✓ 成功获取 {len(df)} 只成分股，耗时 {elapsed:.2f} 秒")
    print(f"  样例: {df.head(3)['成分券代码'].tolist()}")
except Exception as e:
    print(f"  ✗ 失败: {type(e).__name__}: {e}")
    # 尝试备用接口
    try:
        print("  尝试备用接口 index_stock_cons...")
        df = ak.index_stock_cons(symbol="399300")
        print(f"  ✓ 备用接口成功，获取 {len(df)} 只成分股")
    except Exception as e2:
        print(f"  ✗ 备用接口也失败: {type(e2).__name__}: {e2}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

