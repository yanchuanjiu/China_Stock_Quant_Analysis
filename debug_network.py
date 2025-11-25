import akshare as ak
import time

print("Testing connectivity to EastMoney via Akshare...")
try:
    df = ak.stock_zh_a_hist(symbol="002594", period="daily", start_date="20241101", end_date="20241120", adjust="qfq")
    print("Success!")
    print(df.head())
except Exception as e:
    print(f"Failed: {e}")

