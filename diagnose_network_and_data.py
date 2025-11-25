#!/usr/bin/env python3
"""诊断网络连接和 akshare 接口有效性，并测试 qlib 数据收集器"""
import sys
import socket
import subprocess
from pathlib import Path

print("=" * 60)
print("网络与数据源诊断报告")
print("=" * 60)

# 1. 检查基础网络连接
print("\n[1] 检查基础网络连接...")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex(('www.baidu.com', 443))
    sock.close()
    if result == 0:
        print("  ✓ 基础网络连接正常")
    else:
        print(f"  ✗ 基础网络连接失败 (错误码: {result})")
        print("  原因: 系统级网络配置问题，可能是防火墙、VPN 或网络代理设置")
except Exception as e:
    print(f"  ✗ 网络连接测试异常: {e}")

# 2. 检查 DNS 解析
print("\n[2] 检查 DNS 解析...")
test_hosts = [
    'www.baidu.com',
    'push2his.eastmoney.com',
    '82.push2.eastmoney.com',
    'oss-ch.csindex.com.cn',
    'vip.stock.finance.sina.com.cn'
]
for host in test_hosts:
    try:
        ip = socket.gethostbyname(host)
        print(f"  ✓ {host} -> {ip}")
    except Exception as e:
        print(f"  ✗ {host} DNS 解析失败: {e}")

# 3. 检查 akshare 版本和接口
print("\n[3] 检查 akshare 版本和接口...")
try:
    import akshare as ak
    print(f"  ✓ akshare 版本: {ak.__version__}")
    
    # 检查常用接口是否存在
    interfaces = [
        'stock_zh_a_spot_em',
        'stock_zh_a_hist',
        'stock_zh_a_hist_ths',
        'index_stock_cons',
        'index_stock_cons_weight_csindex'
    ]
    for iface in interfaces:
        if hasattr(ak, iface):
            print(f"  ✓ 接口存在: {iface}")
        else:
            print(f"  ✗ 接口不存在: {iface}")
except ImportError as e:
    print(f"  ✗ akshare 未安装: {e}")
except Exception as e:
    print(f"  ✗ akshare 检查异常: {e}")

# 4. 检查 qlib 数据收集器
print("\n[4] 检查 qlib 数据收集器...")
qlib_collectors = Path("qlib/scripts/data_collector")
if qlib_collectors.exists():
    collectors = [d.name for d in qlib_collectors.iterdir() if d.is_dir() and not d.name.startswith('_')]
    print(f"  ✓ 找到 {len(collectors)} 个数据收集器:")
    for c in sorted(collectors):
        print(f"    - {c}")
    
    # 检查是否有 akshare 相关的收集器
    if 'cn_index' in collectors:
        print("\n  [4.1] 检查 cn_index 收集器...")
        cn_index_path = qlib_collectors / "cn_index" / "collector.py"
        if cn_index_path.exists():
            with open(cn_index_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'akshare' in content.lower():
                    print("    ✓ cn_index 收集器可能使用 akshare")
                else:
                    print("    - cn_index 收集器未直接使用 akshare")
else:
    print("  ✗ qlib 数据收集器目录不存在")

# 5. 检查系统网络配置
print("\n[5] 检查系统网络配置...")
try:
    # 检查是否有代理设置
    import os
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    has_proxy = False
    for var in proxy_vars:
        if var in os.environ:
            print(f"  - 检测到代理设置: {var}={os.environ[var]}")
            has_proxy = True
    if not has_proxy:
        print("  - 未检测到代理设置")
    
    # 检查网络接口
    result = subprocess.run(['ifconfig'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("  ✓ 网络接口配置可访问")
    else:
        print("  - 无法检查网络接口配置")
except Exception as e:
    print(f"  - 网络配置检查异常: {e}")

# 6. 总结和建议
print("\n" + "=" * 60)
print("诊断总结")
print("=" * 60)
print("""
问题定位:
  - 错误码 [Errno 49] 'Can't assign requested address' 表明是系统级网络问题
  - 可能原因:
    1. 防火墙或安全软件阻止了出站连接
    2. VPN 或代理配置异常
    3. 网络接口配置问题
    4. 系统资源限制（端口耗尽等）

建议解决方案:
  1. 检查系统防火墙和安全软件设置
  2. 检查 VPN/代理配置，尝试禁用后重试
  3. 使用 qlib 官方数据收集器（如 cn_index）作为替代方案
  4. 在具备正常网络的环境中进行数据下载，然后迁移数据文件
  5. 考虑使用离线数据包或预构建的 qlib 数据

qlib 数据收集器替代方案:
  - qlib 提供了 cn_index 收集器用于获取中国股票数据
  - 可以尝试使用 qlib 官方数据收集器替代直接调用 akshare
""")

print("\n下一步操作建议:")
print("  1. 运行: python -m qlib.scripts.data_collector.cn_index.collector --help")
print("  2. 或使用本地已有数据继续测试工作流")
print("  3. 在具备正常网络的环境下载数据后迁移")

